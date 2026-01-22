"""Pydantic models describing deterministic trading strategies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


EntryType = Literal["breakout", "crossover", "pullback"]
Direction = Literal["above", "below"]
Indicator = Literal["ema", "sma", "rsi", "price"]
ExitType = Literal["take_profit", "stop_loss", "timed_exit", "trailing_stop"]
StrategyMode = Literal["trend", "mean_revert", "breakout"]
TradeSide = Literal["buy", "sell", "close"]


class EntryCondition(BaseModel):
    """Condition describing how to enter a trade."""

    type: EntryType
    level: Optional[float] = Field(
        default=None, description="Static price level to react to"
    )
    direction: Optional[Direction] = Field(
        default=None, description="Whether the trigger happens above/below a level"
    )
    indicator: Optional[Indicator] = Field(
        default=None, description="Technical indicator reference"
    )
    lookback: Optional[int] = Field(
        default=None, description="Candles to look back for indicator computation"
    )
    confirmation_candles: Optional[int] = Field(
        default=None, description="Candles to wait for confirmation"
    )
    min_volume_multiple: Optional[float] = Field(
        default=None, description="Minimum multiple of average volume"
    )
    side: Optional[TradeSide] = Field(
        default=None,
        description="Preferred side if trigger is directional (buy/sell)",
    )


class ExitCondition(BaseModel):
    """Condition describing how to exit a trade."""

    type: ExitType
    take_profit_pct: Optional[float] = Field(
        default=None, description="Percentage gain target"
    )
    stop_loss_pct: Optional[float] = Field(
        default=None, description="Percentage loss threshold"
    )
    max_bars_in_trade: Optional[int] = Field(
        default=None, description="Maximum candles to hold a trade"
    )
    trail_pct: Optional[float] = Field(
        default=None, description="Trailing stop percentage"
    )


class RiskSpec(BaseModel):
    """Risk management parameters attached to a strategy."""

    max_fraction_of_balance: float = Field(
        description="Maximum allocation fraction allowed for the strategy"
    )
    risk_per_trade_fraction: float = Field(
        description="Fraction of balance risked per trade"
    )
    max_drawdown_pct: float = Field(description="Maximum tolerated drawdown percentage")
    leverage: float = Field(default=1.0, description="Leverage multiplier")


class StrategySpec(BaseModel):
    """Complete specification deterministically driving trade decisions."""

    strategy_id: str
    market: str
    timeframe: str
    mode: StrategyMode
    entry_conditions: List[EntryCondition]
    exit_conditions: List[ExitCondition]
    risk: RiskSpec
    expiry_ts: Optional[str] = None
    allow_auto_execute: bool = True

    @field_validator("strategy_id")
    @classmethod
    def _normalize_strategy_id(cls, v: str) -> str:
        if not v:
            raise ValueError("strategy_id must not be empty")
        return v.strip()

    def spec_key(self) -> str:
        """Generate a deterministic key for storage."""
        return f"{self.market}:{self.timeframe}"

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check whether the strategy has reached its expiry timestamp."""
        if not self.expiry_ts:
            return False
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        try:
            expiry_dt = datetime.fromisoformat(self.expiry_ts)
        except ValueError:
            return False
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        return current >= expiry_dt


class PositionState(BaseModel):
    """Minimal state snapshot describing open exposure on a market."""

    market: str
    side: Optional[TradeSide] = None
    qty: float = 0.0
    avg_entry_price: Optional[float] = None
    opened_ts: Optional[int] = None  # epoch seconds

    def is_flat(self) -> bool:
        return not self.side or self.qty <= 0


def serialize_strategy(spec: StrategySpec) -> Dict[str, Any]:
    """Return a JSON-friendly representation."""
    return spec.model_dump()


def deserialize_strategy(data: Dict[str, Any]) -> StrategySpec:
    """Instantiate a strategy spec from serialized data."""
    return StrategySpec.model_validate(data)
