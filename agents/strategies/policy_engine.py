"""Deterministic policy engine for trigger-gated position sizing.

Phase 1 contract: transforms (trigger_state, signal_strength, vol_hat) into
deterministic target_weight while respecting override precedence.

See docs/branching/18-phase1-deterministic-policy-integration.md for spec.

INVARIANTS:
- Policy NEVER creates exposure without trigger permission
- Policy NEVER overrides trigger direction
- If no trigger is active, target_weight is exactly 0
- Replay determinism: same inputs produce same target_weight_final
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from schemas.policy import (
    PolicyConfig,
    PolicyDecisionRecord,
    TriggerState,
    TriggerStateResult,
)

logger = logging.getLogger(__name__)

# Minimum epsilon for division safety
EPS = 1e-9


@dataclass
class RiskOverrides:
    """Risk caps and overrides from risk engine."""

    max_weight: float = 1.0
    max_symbol_weight: Dict[str, float] = field(default_factory=dict)
    emergency_exit_active: bool = False
    emergency_exit_reason: Optional[str] = None


@dataclass
class PolicyState:
    """Mutable state tracked across bars for smoothing."""

    current_weights: Dict[str, float] = field(default_factory=dict)
    last_rebalance_bar: Dict[str, int] = field(default_factory=dict)
    bar_counter: int = 0

    def get_current_weight(self, symbol: str) -> float:
        return self.current_weights.get(symbol, 0.0)

    def update_weight(self, symbol: str, weight: float) -> None:
        self.current_weights[symbol] = weight

    def record_rebalance(self, symbol: str) -> None:
        self.last_rebalance_bar[symbol] = self.bar_counter

    def bars_since_rebalance(self, symbol: str) -> int:
        last = self.last_rebalance_bar.get(symbol, -999)
        return self.bar_counter - last

    def increment_bar(self) -> None:
        self.bar_counter += 1


def _policy_config_hash(config: PolicyConfig) -> str:
    """Generate deterministic hash of policy config for versioning."""
    data = config.model_dump_json(exclude={"stand_down_state"})
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def classify_trigger_state(
    symbol: str,
    entry_fired: bool,
    exit_fired: bool,
    entry_direction: Optional[Literal["long", "short"]],
    has_position: bool,
    position_direction: Optional[Literal["long", "short"]],
    active_trigger_ids: Optional[List[str]] = None,
    confidence_grade: Optional[Literal["A", "B", "C"]] = None,
    signal_strength: float = 0.0,
) -> TriggerStateResult:
    """Classify trigger evaluation results into permission state.

    Rules:
    - inactive: no entry trigger fired and no position to exit
    - long_allowed: entry fired with long direction
    - short_allowed: entry fired with short direction
    - exit_only: no entry but has position (exit allowed, no new entries)

    Args:
        symbol: Asset symbol
        entry_fired: Whether an entry trigger fired this bar
        exit_fired: Whether an exit trigger fired this bar
        entry_direction: Direction of entry trigger (if fired)
        has_position: Whether we currently have a position
        position_direction: Direction of current position
        active_trigger_ids: List of trigger IDs that fired
        confidence_grade: Highest confidence grade of active triggers
        signal_strength: Aggregated signal strength [0, 1]

    Returns:
        TriggerStateResult with classified state
    """
    state: TriggerState
    direction: Optional[Literal["long", "short"]] = None

    if entry_fired and entry_direction:
        # Entry trigger fired - permission granted for that direction
        if entry_direction == "long":
            state = "long_allowed"
            direction = "long"
        else:
            state = "short_allowed"
            direction = "short"
    elif has_position:
        # No entry but have position - exit only mode
        state = "exit_only"
        direction = position_direction
    else:
        # No entry, no position - inactive
        state = "inactive"

    return TriggerStateResult(
        symbol=symbol,
        state=state,
        active_trigger_ids=active_trigger_ids or [],
        entry_triggers_fired=1 if entry_fired else 0,
        exit_triggers_fired=1 if exit_fired else 0,
        direction=direction,
        highest_confidence=confidence_grade,
        signal_strength=signal_strength,
    )


def apply_policy(
    trigger_result: TriggerStateResult,
    vol_hat: float,
    config: PolicyConfig,
    state: PolicyState,
    risk_overrides: Optional[RiskOverrides] = None,
    timestamp: Optional[datetime] = None,
    plan_id: str = "unknown",
    trade_set_id: Optional[str] = None,
) -> Tuple[float, PolicyDecisionRecord]:
    """Apply deterministic policy to compute target weight.

    Implements the canonical math from the runbook:
    1. Trigger gate
    2. Signal sanitation
    3. Deadband
    4. Volatility scaling
    5. Raw magnitude + bounds
    6. Smoothing / monotone exit
    7. Risk caps
    8. Override precedence

    Args:
        trigger_result: Classified trigger state for symbol
        vol_hat: Estimated volatility (annualized)
        config: Policy configuration (immutable intra-plan)
        state: Mutable policy state for smoothing
        risk_overrides: Risk caps and emergency overrides
        timestamp: Bar timestamp for decision record
        plan_id: Plan ID for attribution
        trade_set_id: Trade set ID for attribution

    Returns:
        Tuple of (target_weight_final, PolicyDecisionRecord)
    """
    symbol = trigger_result.symbol
    trigger_state = trigger_result.state
    signal_strength = trigger_result.signal_strength
    current_weight = state.get_current_weight(symbol)
    risk_overrides = risk_overrides or RiskOverrides()
    timestamp = timestamp or datetime.now()

    # Initialize tracking variables
    override_applied: Optional[str] = None
    override_reason: Optional[str] = None
    precedence_tier = 4  # Default: policy output

    # Direction from trigger state
    if trigger_state == "long_allowed":
        direction = 1
    elif trigger_state == "short_allowed":
        direction = -1
    else:
        direction = 0

    # ---------- Step 1: Trigger gate ----------
    if trigger_state == "inactive":
        # No trigger permission - force zero
        target_weight_raw = 0.0
        target_weight_policy = 0.0
        target_weight_capped = 0.0
        target_weight_final = 0.0
        signal_deadbanded = 0.0
        vol_scale = 0.0

        record = PolicyDecisionRecord(
            timestamp=timestamp,
            symbol=symbol,
            plan_id=plan_id,
            trade_set_id=trade_set_id,
            trigger_state=trigger_state,
            active_trigger_ids=trigger_result.active_trigger_ids,
            signal_strength=signal_strength,
            signal_deadbanded=signal_deadbanded,
            vol_hat=vol_hat,
            vol_scale=vol_scale,
            current_weight=current_weight,
            target_weight_raw=target_weight_raw,
            target_weight_policy=target_weight_policy,
            target_weight_capped=target_weight_capped,
            target_weight_final=target_weight_final,
            delta_weight=target_weight_final - current_weight,
            policy_config_hash=_policy_config_hash(config),
            should_rebalance=False,
            rebalance_blocked_reason="trigger_inactive",
        )
        return target_weight_final, record

    # ---------- Step 2: Signal sanitation ----------
    s = max(0.0, min(1.0, signal_strength))

    # ---------- Step 3: Deadband ----------
    tau = config.tau
    if s < tau:
        s_db = 0.0
    else:
        s_db = (s - tau) / (1.0 - tau + EPS)
    signal_deadbanded = s_db

    # ---------- Step 4: Volatility scaling ----------
    vol_target = config.vol_target
    vol_scale = min(1.0, max(0.0, vol_target / max(vol_hat, EPS)))

    # ---------- Step 5: Raw magnitude + bounds ----------
    m_raw = s_db * vol_scale
    w_min = config.w_min
    w_max = config.w_max

    if m_raw == 0:
        m_bounded = 0.0
    else:
        m_bounded = max(w_min, min(m_raw, w_max))

    target_weight_raw = direction * m_bounded

    # ---------- Step 6: Smoothing / monotone exit ----------
    alpha = config.get_alpha()

    if trigger_state == "exit_only":
        # Monotone convergence toward zero
        target_weight_policy = (1.0 - alpha) * current_weight
        # Ensure we're moving toward zero (not away)
        if abs(target_weight_policy) > abs(current_weight):
            target_weight_policy = current_weight
    else:
        # Normal smoothing
        target_weight_policy = (1.0 - alpha) * current_weight + alpha * target_weight_raw

    # ---------- Step 7: Risk caps ----------
    target_weight_capped = target_weight_policy

    # Symbol-specific cap
    symbol_cap = risk_overrides.max_symbol_weight.get(symbol, config.w_max)
    if abs(target_weight_capped) > symbol_cap:
        target_weight_capped = symbol_cap * (1 if target_weight_capped > 0 else -1)
        override_applied = "risk_cap"
        override_reason = f"symbol_cap={symbol_cap}"
        precedence_tier = 3

    # Global cap
    if abs(target_weight_capped) > risk_overrides.max_weight:
        target_weight_capped = risk_overrides.max_weight * (1 if target_weight_capped > 0 else -1)
        override_applied = "risk_cap"
        override_reason = f"global_cap={risk_overrides.max_weight}"
        precedence_tier = 3

    target_weight_final = target_weight_capped

    # ---------- Step 8: Override precedence ----------
    # Precedence: emergency_exit > stand_down > risk_caps > policy

    # Check stand_down
    if config.stand_down_state is not None:
        if timestamp < config.stand_down_state.stand_down_until_ts:
            target_weight_final = 0.0
            override_applied = "stand_down"
            override_reason = config.stand_down_state.stand_down_reason
            precedence_tier = 2

    # Check emergency_exit (highest precedence)
    if risk_overrides.emergency_exit_active:
        target_weight_final = 0.0
        override_applied = "emergency_exit"
        override_reason = risk_overrides.emergency_exit_reason
        precedence_tier = 1

    # ---------- Execution decision ----------
    delta_weight = target_weight_final - current_weight
    should_rebalance = abs(delta_weight) > config.epsilon_w
    rebalance_blocked_reason: Optional[str] = None

    if should_rebalance:
        # Check symbol allowlist
        if not config.is_symbol_allowed(symbol):
            should_rebalance = False
            rebalance_blocked_reason = "symbol_not_allowed"

        # Check cooldown (non-emergency only)
        if should_rebalance and precedence_tier > 1:
            bars_since = state.bars_since_rebalance(symbol)
            if bars_since < config.rebalance_cooldown_bars:
                should_rebalance = False
                rebalance_blocked_reason = f"cooldown ({bars_since}/{config.rebalance_cooldown_bars} bars)"

    # Build decision record
    record = PolicyDecisionRecord(
        timestamp=timestamp,
        symbol=symbol,
        plan_id=plan_id,
        trade_set_id=trade_set_id,
        trigger_state=trigger_state,
        active_trigger_ids=trigger_result.active_trigger_ids,
        signal_strength=signal_strength,
        signal_deadbanded=signal_deadbanded,
        vol_hat=vol_hat,
        vol_scale=vol_scale,
        current_weight=current_weight,
        target_weight_raw=target_weight_raw,
        target_weight_policy=target_weight_policy,
        target_weight_capped=target_weight_capped,
        target_weight_final=target_weight_final,
        delta_weight=delta_weight,
        override_applied=override_applied,
        override_reason=override_reason,
        precedence_tier=precedence_tier,
        policy_config_hash=_policy_config_hash(config),
        should_rebalance=should_rebalance,
        rebalance_blocked_reason=rebalance_blocked_reason,
    )

    return target_weight_final, record


class PolicyEngine:
    """Stateful policy engine for continuous position management.

    Wraps apply_policy() with state management and decision history.
    """

    def __init__(
        self,
        config: PolicyConfig,
        plan_id: str = "unknown",
    ) -> None:
        self.config = config
        self.plan_id = plan_id
        self.state = PolicyState()
        self.decision_history: List[PolicyDecisionRecord] = []
        self._config_hash = _policy_config_hash(config)

    def on_bar(
        self,
        trigger_result: TriggerStateResult,
        vol_hat: float,
        timestamp: datetime,
        risk_overrides: Optional[RiskOverrides] = None,
        trade_set_id: Optional[str] = None,
    ) -> Tuple[float, PolicyDecisionRecord]:
        """Process a bar and compute target weight.

        Args:
            trigger_result: Classified trigger state
            vol_hat: Estimated volatility
            timestamp: Bar timestamp
            risk_overrides: Risk caps and emergency state
            trade_set_id: Trade set for attribution

        Returns:
            Tuple of (target_weight_final, decision_record)
        """
        self.state.increment_bar()

        target_weight, record = apply_policy(
            trigger_result=trigger_result,
            vol_hat=vol_hat,
            config=self.config,
            state=self.state,
            risk_overrides=risk_overrides,
            timestamp=timestamp,
            plan_id=self.plan_id,
            trade_set_id=trade_set_id,
        )

        # Update state if rebalance would occur
        if record.should_rebalance:
            self.state.update_weight(trigger_result.symbol, target_weight)
            self.state.record_rebalance(trigger_result.symbol)

        self.decision_history.append(record)
        return target_weight, record

    def force_flatten(
        self,
        symbol: str,
        reason: str,
        timestamp: datetime,
    ) -> PolicyDecisionRecord:
        """Force immediate flatten (emergency exit path).

        Bypasses all policy math and sets target_weight to 0.
        """
        current_weight = self.state.get_current_weight(symbol)

        record = PolicyDecisionRecord(
            timestamp=timestamp,
            symbol=symbol,
            plan_id=self.plan_id,
            trigger_state="inactive",
            signal_strength=0.0,
            signal_deadbanded=0.0,
            vol_hat=0.0,
            vol_scale=0.0,
            current_weight=current_weight,
            target_weight_raw=0.0,
            target_weight_policy=0.0,
            target_weight_capped=0.0,
            target_weight_final=0.0,
            delta_weight=-current_weight,
            override_applied="emergency_exit",
            override_reason=reason,
            precedence_tier=1,
            policy_config_hash=self._config_hash,
            should_rebalance=True,
        )

        self.state.update_weight(symbol, 0.0)
        self.state.record_rebalance(symbol)
        self.decision_history.append(record)

        return record

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary statistics from decision history."""
        if not self.decision_history:
            return {"total_decisions": 0}

        rebalances = [d for d in self.decision_history if d.should_rebalance]
        overrides = [d for d in self.decision_history if d.override_applied]

        state_counts: Dict[str, int] = {}
        for d in self.decision_history:
            state_counts[d.trigger_state] = state_counts.get(d.trigger_state, 0) + 1

        return {
            "total_decisions": len(self.decision_history),
            "rebalances": len(rebalances),
            "overrides_applied": len(overrides),
            "trigger_state_distribution": state_counts,
            "override_types": {
                d.override_applied: sum(1 for x in overrides if x.override_applied == d.override_applied)
                for d in overrides if d.override_applied
            },
        }
