"""Precommitted position exit contracts and portfolio meta-risk overlay (Runbook 60).

PositionExitContract materializes at entry time and defines the complete exit plan
for a position before capital is committed.  This enforces the core invariant:

    "We must know how we will get out before we get in."

PortfolioMetaRiskPolicy is a separate deterministic overlay that may trim or
rebalance at portfolio level — distinct from strategy exits and emergency exits.

Three exit classes (must remain distinct):
  1. PositionExitContract  — per-position, strategy-authored, at-entry
  2. PortfolioMetaRiskPolicy — portfolio-wide, deterministic overlay
  3. emergency_exit           — system/market safety interrupt (separate path)

No LLM calls.  No I/O.  Pure data containers with validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

POSITION_EXIT_CONTRACT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# ExitLeg — a single action in the target ladder
# ---------------------------------------------------------------------------


class ExitLeg(BaseModel):
    """A single exit action within a PositionExitContract's target ladder.

    Each leg defines one exit event: take a profit at a price level, scale out
    at an R-multiple, time out, or fully close.  Legs are ordered by priority
    and may be enabled/disabled at runtime.
    """

    model_config = {"extra": "forbid"}

    leg_id: str = Field(default_factory=lambda: f"leg_{uuid4().hex[:8]}")
    kind: Literal["take_profit", "risk_reduce", "time_exit", "full_exit"]
    trigger_mode: Literal["price_level", "r_multiple", "time", "structure_event"]
    fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of remaining position to exit on this leg (0 < f <= 1).",
    )

    # Mode-specific fields (one must be set per trigger_mode)
    price_abs: Optional[float] = None           # required when trigger_mode="price_level"
    r_multiple: Optional[float] = None          # required when trigger_mode="r_multiple"
    structure_level_id: Optional[str] = None    # optional for structure_event legs

    priority: int = 0       # lower = higher priority; used to sequence concurrent legs
    enabled: bool = True
    fired: bool = False     # runtime state: True once this leg has executed

    @model_validator(mode="after")
    def _validate_leg(self) -> "ExitLeg":
        if self.trigger_mode == "price_level" and self.price_abs is None:
            raise ValueError(
                "price_abs is required when trigger_mode='price_level'"
            )
        if self.trigger_mode == "r_multiple" and self.r_multiple is None:
            raise ValueError(
                "r_multiple is required when trigger_mode='r_multiple'"
            )
        if self.trigger_mode == "r_multiple" and self.r_multiple is not None and self.r_multiple <= 0:
            raise ValueError("r_multiple must be positive")
        return self


# ---------------------------------------------------------------------------
# TimeExitRule — time-based position expiry
# ---------------------------------------------------------------------------


class TimeExitRule(BaseModel):
    """Time-based exit rule that expires a position after a duration threshold.

    At least one of max_hold_bars or max_hold_minutes must be set.
    session_boundary_action defines behavior at daily/session close.
    """

    model_config = {"extra": "forbid"}

    max_hold_bars: Optional[int] = Field(default=None, gt=0)
    max_hold_minutes: Optional[int] = Field(default=None, gt=0)
    session_boundary_action: Literal["hold", "flatten", "reassess"] = "reassess"

    @model_validator(mode="after")
    def _at_least_one_time_limit(self) -> "TimeExitRule":
        if self.max_hold_bars is None and self.max_hold_minutes is None:
            raise ValueError(
                "TimeExitRule requires at least one of max_hold_bars or max_hold_minutes"
            )
        return self


# ---------------------------------------------------------------------------
# PositionExitContract — per-position precommitted exit plan
# ---------------------------------------------------------------------------


class PositionExitContract(BaseModel):
    """Precommitted exit plan created at entry time for a single position.

    Core invariant: every normal (non-emergency) entry must produce a valid
    PositionExitContract before capital is committed.  If a contract cannot
    be formed, the entry is rejected.

    A contract includes:
    - stop invalidation (hard price stop)
    - optional target ladder (ExitLeg list)
    - optional time-based expiry (TimeExitRule)
    - amendment policy (tighten_only by default)
    - audit provenance fields
    """

    model_config = {"extra": "forbid"}

    contract_id: str = Field(default_factory=lambda: f"contract_{uuid4().hex}")
    contract_version: str = POSITION_EXIT_CONTRACT_VERSION

    # Position identity
    position_id: str
    symbol: str
    side: Literal["long", "short"]
    created_at: datetime

    # Audit / provenance
    source_plan_id: Optional[str] = None
    source_trigger_id: str
    source_category: Optional[str] = None
    template_id: Optional[str] = None
    playbook_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    snapshot_hash: Optional[str] = None

    # Entry state (captured at fill time)
    entry_price: float = Field(gt=0.0)
    initial_qty: float = Field(gt=0.0)
    stop_price_abs: float = Field(gt=0.0)  # hard stop price (invalidation level)

    # Exit plan
    target_legs: List[ExitLeg] = Field(default_factory=list)
    time_exit: Optional[TimeExitRule] = None
    trailing_rule: Optional[Dict[str, Any]] = None  # reserved for future typed model

    # Amendment constraints
    amendment_policy: Literal["none", "tighten_only", "policy_approved"] = "tighten_only"
    allow_portfolio_overlay_trims: bool = True

    # Runtime state (updated by execution layer)
    remaining_qty: Optional[float] = None  # None until position fill is confirmed

    @model_validator(mode="after")
    def _validate_contract(self) -> "PositionExitContract":
        # Stop must be on the correct side of entry
        if self.side == "long" and self.stop_price_abs >= self.entry_price:
            raise ValueError(
                f"Long contract: stop_price_abs ({self.stop_price_abs}) "
                f"must be strictly below entry_price ({self.entry_price})"
            )
        if self.side == "short" and self.stop_price_abs <= self.entry_price:
            raise ValueError(
                f"Short contract: stop_price_abs ({self.stop_price_abs}) "
                f"must be strictly above entry_price ({self.entry_price})"
            )
        # Partial-exit leg fractions must not exceed 1.0 combined
        partial_legs = [leg for leg in self.target_legs if leg.kind != "full_exit"]
        if partial_legs:
            total = sum(leg.fraction for leg in partial_legs)
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"target_legs partial-exit fractions sum to {total:.4f}, "
                    "which exceeds 1.0"
                )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def stop_r_distance(self) -> float:
        """Absolute price distance from entry to stop."""
        return abs(self.entry_price - self.stop_price_abs)

    def compute_r_multiple(self, target_price: float) -> Optional[float]:
        """Compute R multiple for a hypothetical target price.

        Returns None when stop distance is zero (degenerate contract).
        For longs:  R = (target - entry) / stop_distance
        For shorts: R = (entry - target) / stop_distance
        """
        if self.stop_r_distance == 0:
            return None
        if self.side == "long":
            return (target_price - self.entry_price) / self.stop_r_distance
        else:
            return (self.entry_price - target_price) / self.stop_r_distance

    @property
    def active_legs(self) -> List[ExitLeg]:
        """Return legs that are enabled and have not yet fired."""
        return [leg for leg in self.target_legs if leg.enabled and not leg.fired]

    @property
    def has_price_target(self) -> bool:
        """True if contract has at least one enabled price-level or R-multiple leg."""
        return any(
            leg.trigger_mode in ("price_level", "r_multiple")
            for leg in self.active_legs
        )


# ---------------------------------------------------------------------------
# PortfolioMetaAction — a single preapproved portfolio-level action
# ---------------------------------------------------------------------------


class PortfolioMetaAction(BaseModel):
    """A single preapproved portfolio-level action within PortfolioMetaRiskPolicy.

    Actions are declared in advance in the policy and can only be executed
    if the linked condition fires.  No ad hoc actions are permitted at runtime.
    """

    model_config = {"extra": "forbid"}

    action_id: str = Field(default_factory=lambda: f"action_{uuid4().hex[:8]}")
    condition_id: str  # which deterministic policy condition triggers this action
    kind: Literal[
        "trim_largest_position_to_cap",
        "reduce_gross_exposure_pct",
        "rebalance_to_cash_pct",
        "freeze_new_entries",
        "tighten_position_stops",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    cooldown_minutes: Optional[int] = Field(default=None, gt=0)


# ---------------------------------------------------------------------------
# PortfolioMetaRiskPolicy — portfolio-wide deterministic overlay
# ---------------------------------------------------------------------------


class PortfolioMetaRiskPolicy(BaseModel):
    """Deterministic portfolio-level risk overlay policy.

    Defines when portfolio-level risk conditions (concentration, drawdown,
    correlation, regime) may trigger preapproved trim/rebalance/freeze actions.

    Invariant: the overlay may only execute actions declared in `actions`.
    No runtime action fabrication is allowed.

    This is explicitly separate from:
    - per-position PositionExitContract (strategy exits)
    - emergency_exit triggers (system safety interrupts)
    """

    model_config = {"extra": "forbid"}

    policy_id: str
    version: str = "1.0.0"
    enabled: bool = True

    # Deterministic condition thresholds (None = condition disabled)
    max_symbol_concentration_pct: Optional[float] = Field(
        default=None, gt=0.0, le=100.0,
        description="Max % of portfolio in a single symbol before trim action fires.",
    )
    max_sector_or_cluster_concentration_pct: Optional[float] = Field(
        default=None, gt=0.0, le=100.0,
        description="Max % of portfolio in correlated cluster before rebalance fires.",
    )
    portfolio_drawdown_reduce_threshold_pct: Optional[float] = Field(
        default=None, gt=0.0,
        description="Portfolio drawdown % threshold that triggers exposure reduction.",
    )
    correlation_reduce_threshold: Optional[float] = Field(
        default=None, gt=0.0, le=1.0,
        description="Correlation spike level that triggers clustered-exposure reduction.",
    )
    hostile_regime_reduce_enabled: bool = Field(
        default=False,
        description="If True, overlay can reduce sizing when regime transitions to hostile state.",
    )

    # Preapproved action registry
    actions: List[PortfolioMetaAction] = Field(default_factory=list)
