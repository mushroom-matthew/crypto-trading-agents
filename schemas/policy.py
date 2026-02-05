"""Schema definitions for the deterministic policy engine.

Phase 1 contract: trigger-gated, deterministic position sizing.
See docs/branching/18-phase1-deterministic-policy-integration.md for spec.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from schemas.llm_strategist import StrategyPlan


# Trigger permission states - policy MUST respect these at all times
TriggerState = Literal["inactive", "long_allowed", "short_allowed", "exit_only"]


class StandDownState(BaseModel):
    """Explicit stand-down payload for forced neutralization."""

    stand_down_until_ts: datetime
    stand_down_reason: str
    stand_down_source: Literal["judge", "ops", "risk_engine"]

    model_config = {"extra": "forbid"}


class PolicyConfig(BaseModel):
    """Plan-level policy configuration, immutable intra-plan.

    Embedded in StrategyPlan and fixed for the full plan lifecycle.
    """

    tau: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Deadband threshold - signals below this produce zero weight",
    )
    vol_target: float = Field(
        default=0.15,
        gt=0.0,
        description="Volatility target for scale normalization (annualized)",
    )
    w_min: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum non-zero absolute weight",
    )
    w_max: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Maximum absolute weight",
    )
    horizon_id: str = Field(
        default="default",
        description="Horizon key for deterministic smoothing constants lookup",
    )
    symbol_allowlist: List[str] = Field(
        default_factory=list,
        description="Symbols eligible for policy output (empty = all allowed)",
    )
    stand_down_state: Optional[StandDownState] = Field(
        default=None,
        description="Active stand-down state for forced neutralization",
    )
    # Smoothing alpha by horizon (deterministic lookup)
    alpha_by_horizon: Dict[str, float] = Field(
        default_factory=lambda: {"default": 0.3, "scalper": 0.5, "swing": 0.1},
        description="Smoothing constants per horizon_id",
    )
    # Execution thresholds
    epsilon_w: float = Field(
        default=0.005,
        ge=0.0,
        description="Minimum delta_weight to trigger rebalance",
    )
    min_trade_notional: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum notional for rebalance execution",
    )
    rebalance_cooldown_bars: int = Field(
        default=2,
        ge=0,
        description="Bars between non-emergency rebalances",
    )

    model_config = {"extra": "forbid"}

    def get_alpha(self) -> float:
        """Get smoothing alpha for current horizon."""
        return self.alpha_by_horizon.get(self.horizon_id, 0.3)

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Check if symbol is in allowlist (empty list = all allowed)."""
        if not self.symbol_allowlist:
            return True
        return symbol in self.symbol_allowlist


class PolicyDecisionRecord(BaseModel):
    """Per-bar, per-symbol policy decision for audit and attribution.

    Every bar MUST produce a decision record, even when no trade occurs.
    Records are replayable for deterministic audit.
    """

    # Identification
    timestamp: datetime
    symbol: str
    plan_id: str
    trade_set_id: Optional[str] = None

    # Trigger state (from trigger engine)
    trigger_state: TriggerState
    active_trigger_ids: List[str] = Field(default_factory=list)

    # Signal inputs
    signal_strength: float = Field(ge=0.0, le=1.0, description="Raw signal [0, 1]")
    signal_deadbanded: float = Field(ge=0.0, le=1.0, description="After deadband")

    # Volatility
    vol_hat: float = Field(ge=0.0, description="Estimated volatility")
    vol_scale: float = Field(ge=0.0, le=1.0, description="Volatility scaling factor")

    # Weight progression
    current_weight: float = Field(description="Current position weight")
    target_weight_raw: float = Field(description="Before smoothing")
    target_weight_policy: float = Field(description="After smoothing")
    target_weight_capped: float = Field(description="After risk caps")
    target_weight_final: float = Field(description="Final after all overrides")
    delta_weight: float = Field(description="target_final - current")

    # Override tracking
    override_applied: Optional[str] = Field(
        default=None,
        description="Override that modified the result (emergency_exit, stand_down, risk_cap)",
    )
    override_reason: Optional[str] = Field(default=None)
    precedence_tier: int = Field(
        default=4,
        description="1=emergency, 2=stand_down, 3=risk_cap, 4=policy",
    )

    # Policy config hash for version tracking
    policy_config_hash: Optional[str] = Field(default=None)

    # Execution decision
    should_rebalance: bool = Field(default=False)
    rebalance_blocked_reason: Optional[str] = Field(default=None)

    model_config = {"extra": "forbid"}

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """Convert to dict for telemetry storage."""
        return self.model_dump(mode="json")


class TriggerStateResult(BaseModel):
    """Result of trigger state classification for a symbol.

    Aggregates trigger evaluations into a single permission state.
    """

    symbol: str
    state: TriggerState
    active_trigger_ids: List[str] = Field(default_factory=list)
    entry_triggers_fired: int = Field(default=0)
    exit_triggers_fired: int = Field(default=0)
    direction: Optional[Literal["long", "short"]] = Field(default=None)
    highest_confidence: Optional[Literal["A", "B", "C"]] = Field(default=None)
    signal_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated signal strength from active triggers",
    )

    model_config = {"extra": "forbid"}


def get_policy_config_from_plan(plan: "StrategyPlan") -> Optional[PolicyConfig]:
    """Extract PolicyConfig from a StrategyPlan.

    Returns PolicyConfig if plan has policy_config dict, None otherwise.
    """
    if plan.policy_config is None:
        return None
    return PolicyConfig(**plan.policy_config)
