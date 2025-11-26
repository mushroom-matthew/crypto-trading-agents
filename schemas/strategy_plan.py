"""Typed schema for LLM-generated strategy plans used in backtests."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LookbackConfig(BaseModel):
    """Configuration describing planner lookback preferences."""

    preferred_bars: int = Field(..., gt=0)
    min_bars: int = Field(..., gt=0)
    max_bars: int = Field(..., gt=0)

    @model_validator(mode="after")
    def ensure_preferred_within_bounds(self) -> "LookbackConfig":
        if self.min_bars > self.max_bars:
            raise ValueError("min_bars cannot exceed max_bars")
        if not (self.min_bars <= self.preferred_bars <= self.max_bars):
            raise ValueError("preferred_bars must fall within [min_bars, max_bars]")
        return self


class RiskManagementConfig(BaseModel):
    """Risk management envelope for a plan."""

    max_position_pct: float = Field(..., ge=0, le=1)
    max_daily_loss_pct: float = Field(..., ge=0, le=1)
    max_total_drawdown_pct: float = Field(..., ge=0, le=1)
    per_trade_risk_pct: float = Field(..., ge=0, le=1)


class Condition(BaseModel):
    """Single indicator-based condition for entry/exit rules."""

    indicator: str
    operator: Literal["<", "<=", ">", ">=", "==", "!="]
    value: Union[float, str]


class PositionSizing(BaseModel):
    """Sizing instructions attached to an entry rule."""

    type: Literal["volatility_target", "fixed_fraction", "notional"]
    target_volatility: Optional[float] = Field(default=None, ge=0)
    max_leverage: Optional[float] = Field(default=None, ge=0)
    fixed_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    notional_amount: Optional[float] = Field(default=None, ge=0)


class EntryRule(BaseModel):
    """Definition of an entry pattern for the deterministic executor."""

    id: str
    direction: Literal["long", "short"]
    conditions: List[Condition]
    position_sizing: Optional[PositionSizing] = None


class StopLossConfig(BaseModel):
    """Stop loss configuration used by exit rules."""

    type: Literal["atr_multiple", "percentage", "fixed_tick", "none"]
    atr_period: Optional[int] = Field(default=None, gt=0)
    multiple: Optional[float] = Field(default=None, gt=0)
    percentage: Optional[float] = Field(default=None, ge=0)
    ticks: Optional[float] = Field(default=None, ge=0)


class TakeProfitConfig(BaseModel):
    """Take profit configuration attached to exit rules."""

    type: Literal["rr_multiple", "percentage", "fixed_tick", "none"]
    reward_risk: Optional[float] = Field(default=None, gt=0)
    percentage: Optional[float] = Field(default=None, ge=0)
    ticks: Optional[float] = Field(default=None, ge=0)


class ExitRule(BaseModel):
    """Exit rule definition referencing an entry rule."""

    id: str
    applies_to_entry_id: Optional[str] = None
    conditions: List[Condition]
    stop_loss: Optional[StopLossConfig] = None
    take_profit: Optional[TakeProfitConfig] = None


class EMACrossOverride(BaseModel):
    enabled: bool = False
    fast_period: int = Field(default=20, gt=0)
    slow_period: int = Field(default=50, gt=0)


class RSIRegimeShift(BaseModel):
    enabled: bool = False
    overbought: float = Field(default=70, ge=0, le=100)
    oversold: float = Field(default=30, ge=0, le=100)


class ATRVolatilityShift(BaseModel):
    enabled: bool = False
    lookback_bars: int = Field(default=50, gt=0)
    high_vol_threshold: float = Field(default=1.5, gt=0)
    low_vol_threshold: float = Field(default=0.7, gt=0)


class VolumeSurgeTrigger(BaseModel):
    enabled: bool = False
    lookback_bars: int = Field(default=30, gt=0)
    multiple_of_median: float = Field(default=2.0, gt=0)


class SupportResistanceProximity(BaseModel):
    enabled: bool = False
    distance_pct: float = Field(default=1.0, gt=0)


class AllocationChangeTrigger(BaseModel):
    enabled: bool = False
    threshold_pct: float = Field(default=10.0, gt=0)


class ReplanTriggers(BaseModel):
    ema_cross_overrides: EMACrossOverride = EMACrossOverride()
    rsi_regime_shift: RSIRegimeShift = RSIRegimeShift()
    atr_volatility_shift: ATRVolatilityShift = ATRVolatilityShift()
    volume_surge: VolumeSurgeTrigger = VolumeSurgeTrigger()
    support_resistance_proximity: SupportResistanceProximity = SupportResistanceProximity()
    allocation_change: AllocationChangeTrigger = AllocationChangeTrigger()


class LLMMetadata(BaseModel):
    model_name: str
    prompt_version: str
    notes: Optional[str] = None


class StrategyPlan(BaseModel):
    """Complete planner output that guides deterministic execution."""

    plan_id: str
    created_at: datetime
    symbol: str
    timeframe: str
    lookback: LookbackConfig
    risk_management: RiskManagementConfig
    entry_rules: List[EntryRule]
    exit_rules: List[ExitRule]
    replan_triggers: ReplanTriggers
    llm_metadata: LLMMetadata

    def clamp_lookback(self, available_bars: int) -> int:
        """Return preferred lookback clipped to available history."""
        if available_bars <= 0:
            raise ValueError("available_bars must be positive")
        preferred = self.lookback.preferred_bars
        lower = min(self.lookback.min_bars, available_bars)
        upper = min(self.lookback.max_bars, available_bars)
        return max(lower, min(preferred, upper))
