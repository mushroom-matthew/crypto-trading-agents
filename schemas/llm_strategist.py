"""Schema definitions for the LLM strategist protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class SerializableModel(BaseModel):
    """Base-model that standardizes JSON helpers and validation."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, raw: str) -> "SerializableModel":
        return cls.model_validate_json(raw)


class IndicatorSnapshot(SerializableModel):
    """Pre-computed indicator payload that the LLM consumes."""

    symbol: str
    timeframe: str
    as_of: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None
    volume_multiple: float | None = None
    sma_short: float | None = None
    sma_medium: float | None = None
    sma_long: float | None = None
    ema_short: float | None = None
    ema_medium: float | None = None
    ema_long: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    roc_short: float | None = None
    roc_medium: float | None = None
    realized_vol_short: float | None = None
    realized_vol_medium: float | None = None
    donchian_upper_short: float | None = None
    donchian_lower_short: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_middle: float | None = None
    # Cycle indicators (200-bar window for cyclical analysis)
    cycle_high_200: float | None = None
    cycle_low_200: float | None = None
    cycle_range_200: float | None = None
    cycle_position: float | None = None
    # Fibonacci retracement levels (from cycle high/low)
    fib_236: float | None = None
    fib_382: float | None = None
    fib_500: float | None = None
    fib_618: float | None = None
    fib_786: float | None = None
    # Expansion/contraction ratios (swing-based)
    last_expansion_pct: float | None = None
    last_contraction_pct: float | None = None
    expansion_contraction_ratio: float | None = None
    # Fast indicators for scalpers (5m/1m timeframes)
    ema_fast: float | None = None  # EMA 5
    ema_very_fast: float | None = None  # EMA 8
    realized_vol_fast: float | None = None  # Fast vol window (10 or less)
    ewma_vol: float | None = None  # EWMA volatility
    vwap: float | None = None  # Volume-weighted average price
    vwap_distance_pct: float | None = None  # Distance from VWAP as %
    vol_burst: bool | None = None  # True when volume_multiple >= threshold


class AssetState(SerializableModel):
    """Summarized asset view injected into the LLM context."""

    symbol: str
    indicators: List[IndicatorSnapshot]
    trend_state: Literal["uptrend", "downtrend", "sideways"]
    vol_state: Literal["low", "normal", "high", "extreme"]
    regime_assessment: "RegimeAssessment | None" = None


class PortfolioState(SerializableModel):
    """Rolling-account statistics shared with the LLM."""

    timestamp: datetime
    equity: float
    cash: float
    positions: Dict[str, float]
    realized_pnl_7d: float
    realized_pnl_30d: float
    sharpe_30d: float
    max_drawdown_90d: float
    win_rate_30d: float
    profit_factor_30d: float


TriggerDirection = Literal["long", "short", "flat", "exit", "flat_exit"]
TriggerCategory = Literal[
    "trend_continuation",
    "reversal",
    "volatility_breakout",
    "mean_reversion",
    "risk_reduce",      # Partial exit to trim exposure (0 < exit_fraction < 1.0)
    "risk_off",         # Full exit to defensive posture (regime-dependent)
    "emergency_exit",   # Hard safety interrupt - unconditional flatten
    "other",
]


class TriggerSummary(SerializableModel):
    """Compact trigger payload for continuity across replans."""

    id: str
    symbol: str
    timeframe: str
    direction: TriggerDirection
    category: TriggerCategory | None = Field(default=None)
    confidence_grade: Literal["A", "B", "C"] | None = Field(default=None)
    entry_rule: str
    exit_rule: str
    hold_rule: str | None = Field(default=None)
    stop_loss_pct: float | None = Field(default=None, ge=0.0)
    exit_fraction: float | None = Field(default=None, description="Partial exit fraction for risk_reduce")


class LLMInput(SerializableModel):
    """Structured payload that is serialized and sent to the LLM strategist."""

    portfolio: PortfolioState
    assets: List[AssetState]
    risk_params: Dict[str, Any]
    global_context: Dict[str, Any] = Field(default_factory=dict)
    market_structure: Dict[str, Any] = Field(default_factory=dict)
    previous_triggers: List[TriggerSummary] = Field(default_factory=list)


class TriggerCondition(SerializableModel):
    """Entry/exit specification provided by the LLM."""

    id: str
    symbol: str
    category: TriggerCategory | None = Field(default=None)
    confidence_grade: Literal["A", "B", "C"] | None = Field(default=None)
    direction: TriggerDirection
    timeframe: str
    entry_rule: str
    exit_rule: str
    hold_rule: str | None = Field(
        default=None,
        description="Optional rule that, when True, suppresses exit_rule to maintain position. "
        "Use to prevent premature exits from minor fluctuations. Emergency exits still fire."
    )
    stop_loss_pct: float | None = Field(default=None, ge=0.0)
    exit_fraction: float | None = Field(
        default=None,
        description="Fraction of position to close (0 < f <= 1.0). "
        "Used with risk_reduce for partial exits. None = full exit (backward compatible)."
    )
    learning_book: bool = Field(default=False, description="True if this trigger belongs to the learning book.")
    experiment_id: str | None = Field(default=None, description="Experiment that spawned this trigger, if any.")
    exit_binding_exempt: bool = Field(
        default=False,
        description="Internal flag: bypasses category-based exit binding checks at runtime. "
        "Set only by enforce_exit_binding(); external input is always reset to False.",
    )

    @model_validator(mode="after")
    def _validate_trigger(self) -> "TriggerCondition":
        # Safety: exit_binding_exempt is internal-only — reset any externally supplied value.
        # Only enforce_exit_binding() in the compiler should set this to True.
        # Use object.__setattr__ to bypass validate_assignment (avoids recursion).
        object.__setattr__(self, "exit_binding_exempt", False)

        # emergency_exit requires exit_rule
        if self.category == "emergency_exit" and not (self.exit_rule or "").strip():
            raise ValueError("emergency_exit triggers must define a non-empty exit_rule")

        # exit_fraction must be 0 < f <= 1.0 when set
        if self.exit_fraction is not None:
            if self.exit_fraction <= 0.0 or self.exit_fraction > 1.0:
                raise ValueError("exit_fraction must be in range (0, 1.0]")

        # risk_reduce should have exit_fraction (warn but don't require for now)
        # This allows gradual adoption without breaking existing code

        return self


class RiskConstraint(SerializableModel):
    """Plan-level guardrails enforced by the deterministic engine."""

    max_position_risk_pct: float = Field(..., ge=0.0)
    max_symbol_exposure_pct: float = Field(..., ge=0.0)
    max_portfolio_exposure_pct: float = Field(..., ge=0.0)
    max_daily_loss_pct: float = Field(..., ge=0.0)
    max_daily_risk_budget_pct: float | None = Field(default=None, ge=0.0)


class PositionSizingRule(SerializableModel):
    """Symbol-level sizing behavior derived from risk params."""

    symbol: str
    sizing_mode: Literal["fixed_fraction", "vol_target", "notional"]
    target_risk_pct: float | None = Field(default=None, ge=0.0)
    vol_target_annual: float | None = Field(default=None, ge=0.0)
    notional: float | None = Field(default=None, ge=0.0)
    rationale: str | None = None


class RegimeAlert(SerializableModel):
    """Condition that would trigger strategy re-assessment."""

    indicator: str
    threshold: float
    direction: Literal["above", "below", "crosses"]
    symbol: str
    interpretation: str
    priority: Literal["high", "medium", "low"] = "medium"


class RegimeAssessment(SerializableModel):
    """Pre-computed regime classification from deterministic classifier."""

    regime: Literal["bull", "bear", "range", "volatile", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    primary_signals: List[str] = Field(default_factory=list)
    conflicting_signals: List[str] = Field(default_factory=list)


class SizingHint(SerializableModel):
    """Advisory sizing suggestion from LLM (not enforced)."""

    symbol: str
    suggested_risk_pct: float = Field(ge=0.0)
    rationale: str


class StrategyPlan(SerializableModel):
    """Complete plan payload returned by the LLM."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex}")
    run_id: str | None = None
    generated_at: datetime
    valid_until: datetime
    global_view: Optional[str] = None
    regime: Literal["bull", "bear", "range", "high_vol", "mixed"]
    stance: Literal["active", "defensive", "wait"] = "active"
    triggers: List[TriggerCondition] = Field(default_factory=list)
    risk_constraints: RiskConstraint | None = None
    sizing_rules: List[PositionSizingRule] = Field(default_factory=list)
    regime_assessment: RegimeAssessment | None = None
    regime_alerts: List[RegimeAlert] = Field(default_factory=list)
    sizing_hints: List[SizingHint] = Field(default_factory=list)
    rationale: str | None = None
    max_trades_per_day: int | None = Field(default=None, ge=0)
    min_trades_per_day: int | None = Field(default=None, ge=0)
    max_triggers_per_symbol_per_day: int | None = Field(default=None, ge=0)
    trigger_budgets: Dict[str, int] = Field(default_factory=dict)
    allowed_symbols: List[str] = Field(default_factory=list)
    allowed_directions: List[TriggerDirection] = Field(default_factory=list)
    allowed_trigger_categories: List[TriggerCategory] = Field(default_factory=list)
    policy_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional policy configuration dict for deterministic position sizing. "
        "If set, enables Phase 1 policy integration (trigger-gated target weights). "
        "Expected schema: schemas.policy.PolicyConfig fields (tau, vol_target, w_min, w_max, etc).",
    )


# Rebuild model for forward references
AssetState.model_rebuild()
