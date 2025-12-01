"""Schema definitions for the LLM strategist protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


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
    close: float
    sma_short: float | None = None
    sma_medium: float | None = None
    sma_long: float | None = None
    ema_short: float | None = None
    ema_medium: float | None = None
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


class AssetState(SerializableModel):
    """Summarized asset view injected into the LLM context."""

    symbol: str
    indicators: List[IndicatorSnapshot]
    trend_state: Literal["uptrend", "downtrend", "sideways"]
    vol_state: Literal["low", "normal", "high", "extreme"]


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


class LLMInput(SerializableModel):
    """Structured payload that is serialized and sent to the LLM strategist."""

    portfolio: PortfolioState
    assets: List[AssetState]
    risk_params: Dict[str, Any]
    global_context: Dict[str, Any] = Field(default_factory=dict)


TriggerDirection = Literal["long", "short", "flat"]
TriggerCategory = Literal[
    "trend_continuation",
    "reversal",
    "volatility_breakout",
    "mean_reversion",
    "emergency_exit",
    "other",
]


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


class RiskConstraint(SerializableModel):
    """Plan-level guardrails enforced by the deterministic engine."""

    max_position_risk_pct: float = Field(..., ge=0.0)
    max_symbol_exposure_pct: float = Field(..., ge=0.0)
    max_portfolio_exposure_pct: float = Field(..., ge=0.0)
    max_daily_loss_pct: float = Field(..., ge=0.0)


class PositionSizingRule(SerializableModel):
    """Symbol-level sizing behavior derived from risk params."""

    symbol: str
    sizing_mode: Literal["fixed_fraction", "vol_target", "notional"]
    target_risk_pct: float | None = Field(default=None, ge=0.0)
    vol_target_annual: float | None = Field(default=None, ge=0.0)
    notional: float | None = Field(default=None, ge=0.0)


class StrategyPlan(SerializableModel):
    """Complete plan payload returned by the LLM."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex}")
    run_id: str | None = None
    generated_at: datetime
    valid_until: datetime
    global_view: Optional[str] = None
    regime: Literal["bull", "bear", "range", "high_vol", "mixed"]
    triggers: List[TriggerCondition]
    risk_constraints: RiskConstraint
    sizing_rules: List[PositionSizingRule] = Field(default_factory=list)
    max_trades_per_day: int | None = Field(default=None, ge=0)
    min_trades_per_day: int | None = Field(default=None, ge=0)
    allowed_symbols: List[str] = Field(default_factory=list)
    allowed_directions: List[TriggerDirection] = Field(default_factory=list)
    allowed_trigger_categories: List[TriggerCategory] = Field(default_factory=list)
