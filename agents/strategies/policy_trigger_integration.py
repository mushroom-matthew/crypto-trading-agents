"""Integration layer between TriggerEngine and PolicyEngine.

Phase 1 contract: trigger-gated policy produces target weights without
modifying existing trigger evaluation behavior.

This module provides opt-in policy integration that:
1. Wraps TriggerEngine to capture trigger evaluation results
2. Classifies trigger state per symbol per bar
3. Passes to PolicyEngine for target weight calculation
4. Collects decision records for telemetry

The trigger engine continues to produce orders as before; the policy
layer provides advisory target weights for observability and future
execution integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    PortfolioState,
    StrategyPlan,
)
from schemas.policy import (
    PolicyConfig,
    PolicyDecisionRecord,
    TriggerStateResult,
    get_policy_config_from_plan,
)
from schemas.learning_gate import LearningGateStatus

from .policy_engine import PolicyEngine, PolicyState, RiskOverrides, classify_trigger_state
from .risk_engine import RiskEngine
from .trigger_engine import Bar, Order, TriggerEngine

logger = logging.getLogger(__name__)


@dataclass
class BarEvaluationResult:
    """Result of evaluating a single bar with policy integration."""

    # Original trigger engine outputs
    orders: List[Order]
    blocks: List[dict]

    # Policy layer outputs
    trigger_state: TriggerStateResult
    target_weight: float
    decision_record: PolicyDecisionRecord

    # Metadata
    symbol: str
    timestamp: datetime


@dataclass
class PolicyIntegrationState:
    """State tracked across bars for policy integration."""

    # Trigger state per symbol
    last_trigger_state: Dict[str, TriggerStateResult] = field(default_factory=dict)

    # Accumulated decision records
    decision_records: List[PolicyDecisionRecord] = field(default_factory=list)

    # Entry tracking for exit_only detection
    entry_trigger_by_symbol: Dict[str, str] = field(default_factory=dict)
    position_direction_by_symbol: Dict[str, Literal["long", "short"]] = field(default_factory=dict)


class PolicyTriggerIntegration:
    """Integration layer that adds policy-based position sizing to trigger evaluation.

    This class wraps TriggerEngine and PolicyEngine to provide:
    - Trigger state classification from trigger evaluations
    - Target weight calculation from policy engine
    - Decision record collection for telemetry

    Usage:
        integration = PolicyTriggerIntegration(plan, risk_engine)
        result = integration.on_bar(bar, indicator, portfolio)
        # result.orders from trigger engine
        # result.target_weight from policy engine
        # result.decision_record for telemetry
    """

    def __init__(
        self,
        plan: StrategyPlan,
        risk_engine: RiskEngine,
        trigger_engine: Optional[TriggerEngine] = None,
        policy_config: Optional[PolicyConfig] = None,
        **trigger_engine_kwargs,
    ) -> None:
        """Initialize integration layer.

        Args:
            plan: Strategy plan with triggers
            risk_engine: Risk engine for position sizing
            trigger_engine: Optional pre-configured trigger engine
            policy_config: Optional policy config (extracted from plan if not provided)
            **trigger_engine_kwargs: Kwargs passed to TriggerEngine if created internally
        """
        self.plan = plan
        self.risk_engine = risk_engine

        # Create or use provided trigger engine
        if trigger_engine is not None:
            self.trigger_engine = trigger_engine
        else:
            self.trigger_engine = TriggerEngine(
                plan=plan,
                risk_engine=risk_engine,
                **trigger_engine_kwargs,
            )

        # Get policy config from plan or use provided
        if policy_config is not None:
            self._policy_config = policy_config
        else:
            self._policy_config = get_policy_config_from_plan(plan)

        # Initialize policy engine if config available
        self.policy_engine: Optional[PolicyEngine] = None
        if self._policy_config is not None:
            self.policy_engine = PolicyEngine(
                config=self._policy_config,
                plan_id=plan.plan_id,
            )

        # Integration state
        self._state = PolicyIntegrationState()

    @property
    def policy_enabled(self) -> bool:
        """Whether policy integration is active."""
        return self.policy_engine is not None

    def on_bar(
        self,
        bar: Bar,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        asset_state: AssetState | None = None,
        market_structure: dict[str, float | str | None] | None = None,
        position_meta: Mapping[str, dict[str, Any]] | None = None,
        learning_gate_status: LearningGateStatus | None = None,
        risk_overrides: Optional[RiskOverrides] = None,
        trade_set_id: Optional[str] = None,
    ) -> BarEvaluationResult:
        """Evaluate bar with trigger engine and policy integration.

        Returns BarEvaluationResult with both trigger orders and policy target weight.
        """
        symbol = bar.symbol
        timestamp = bar.timestamp

        # 1. Run trigger engine evaluation
        orders, blocks = self.trigger_engine.on_bar(
            bar=bar,
            indicator=indicator,
            portfolio=portfolio,
            asset_state=asset_state,
            market_structure=market_structure,
            position_meta=position_meta,
            learning_gate_status=learning_gate_status,
        )

        # 2. Classify trigger state from evaluation results
        trigger_state = self._classify_from_orders(
            symbol=symbol,
            orders=orders,
            portfolio=portfolio,
        )

        # 3. Get volatility estimate for policy
        vol_hat = self._estimate_volatility(indicator)

        # 4. Run policy engine if enabled
        if self.policy_engine is not None:
            target_weight, decision_record = self.policy_engine.on_bar(
                trigger_result=trigger_state,
                vol_hat=vol_hat,
                timestamp=timestamp,
                risk_overrides=risk_overrides,
                trade_set_id=trade_set_id,
            )
            self._state.decision_records.append(decision_record)
        else:
            # No policy - create placeholder record
            target_weight = 0.0
            decision_record = self._create_placeholder_record(
                symbol=symbol,
                timestamp=timestamp,
                trigger_state=trigger_state,
            )

        # Update state
        self._state.last_trigger_state[symbol] = trigger_state

        return BarEvaluationResult(
            orders=orders,
            blocks=blocks,
            trigger_state=trigger_state,
            target_weight=target_weight,
            decision_record=decision_record,
            symbol=symbol,
            timestamp=timestamp,
        )

    def _classify_from_orders(
        self,
        symbol: str,
        orders: List[Order],
        portfolio: PortfolioState,
    ) -> TriggerStateResult:
        """Classify trigger state from trigger engine order output."""
        # Check current position
        position_qty = portfolio.positions.get(symbol, 0.0)
        has_position = position_qty != 0.0
        position_direction: Optional[Literal["long", "short"]] = None
        if position_qty > 0:
            position_direction = "long"
        elif position_qty < 0:
            position_direction = "short"

        # Analyze orders
        entry_fired = False
        exit_fired = False
        entry_direction: Optional[Literal["long", "short"]] = None
        active_trigger_ids: List[str] = []
        max_confidence: Optional[Literal["A", "B", "C"]] = None

        for order in orders:
            active_trigger_ids.append(order.reason)

            if order.intent == "entry":
                entry_fired = True
                entry_direction = "long" if order.side == "buy" else "short"
                # Track confidence
                if order.trigger_category:
                    # Could extract confidence from trigger lookup, for now skip
                    pass
            elif order.intent in ("exit", "flat"):
                exit_fired = True

        # Compute signal strength from order confidence (placeholder)
        # In full integration, this would come from trigger evaluation metadata
        signal_strength = 0.7 if entry_fired else (0.3 if exit_fired else 0.0)

        return classify_trigger_state(
            symbol=symbol,
            entry_fired=entry_fired,
            exit_fired=exit_fired,
            entry_direction=entry_direction,
            has_position=has_position,
            position_direction=position_direction,
            active_trigger_ids=active_trigger_ids,
            confidence_grade=max_confidence,
            signal_strength=signal_strength,
        )

    def _estimate_volatility(self, indicator: IndicatorSnapshot) -> float:
        """Estimate annualized volatility from indicator snapshot."""
        # Use realized_vol_short if available, else fall back to ATR-based estimate
        if indicator.realized_vol_short is not None:
            return indicator.realized_vol_short

        if indicator.realized_vol_medium is not None:
            return indicator.realized_vol_medium

        # ATR-based fallback (approximate annualized from daily)
        if indicator.atr_14 is not None and indicator.close > 0:
            daily_vol = indicator.atr_14 / indicator.close
            return daily_vol * (252 ** 0.5)  # Annualize

        # Default fallback
        return 0.20

    def _create_placeholder_record(
        self,
        symbol: str,
        timestamp: datetime,
        trigger_state: TriggerStateResult,
    ) -> PolicyDecisionRecord:
        """Create placeholder decision record when policy is disabled."""
        return PolicyDecisionRecord(
            timestamp=timestamp,
            symbol=symbol,
            plan_id=self.plan.plan_id,
            trigger_state=trigger_state.state,
            active_trigger_ids=trigger_state.active_trigger_ids,
            signal_strength=trigger_state.signal_strength,
            signal_deadbanded=0.0,
            vol_hat=0.0,
            vol_scale=0.0,
            current_weight=0.0,
            target_weight_raw=0.0,
            target_weight_policy=0.0,
            target_weight_capped=0.0,
            target_weight_final=0.0,
            delta_weight=0.0,
            should_rebalance=False,
            rebalance_blocked_reason="policy_disabled",
        )

    def get_decision_records(self) -> List[PolicyDecisionRecord]:
        """Get accumulated decision records."""
        return list(self._state.decision_records)

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of policy decisions."""
        if self.policy_engine is not None:
            return self.policy_engine.get_decision_summary()
        return {"total_decisions": 0, "policy_enabled": False}

    def record_fill(self, symbol: str, is_entry: bool, timestamp: datetime) -> None:
        """Record fill in underlying trigger engine."""
        self.trigger_engine.record_fill(symbol, is_entry, timestamp)


def create_integration_from_plan(
    plan: StrategyPlan,
    risk_engine: RiskEngine,
    **trigger_engine_kwargs,
) -> Tuple[TriggerEngine, Optional[PolicyTriggerIntegration]]:
    """Factory to create trigger engine and optional policy integration.

    Returns tuple of (trigger_engine, integration).
    If plan has no policy_config, integration is None.
    """
    trigger_engine = TriggerEngine(
        plan=plan,
        risk_engine=risk_engine,
        **trigger_engine_kwargs,
    )

    policy_config = get_policy_config_from_plan(plan)
    if policy_config is None:
        return trigger_engine, None

    integration = PolicyTriggerIntegration(
        plan=plan,
        risk_engine=risk_engine,
        trigger_engine=trigger_engine,
        policy_config=policy_config,
    )

    return trigger_engine, integration
