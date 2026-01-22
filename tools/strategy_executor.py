"""Deterministic evaluation of StrategySpec objects against market features."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel

from tools.strategy_spec import (
    EntryCondition,
    ExitCondition,
    PositionState,
    StrategySpec,
    TradeSide,
)


class TradeSignal(BaseModel):
    """Instruction for the execution agent to place or close positions."""

    side: Literal["buy", "sell", "close"]
    size_fraction: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str


def compute_sma(series: Sequence[float], period: int) -> Optional[float]:
    if period <= 0 or len(series) < period:
        return None
    return fmean(series[-period:])


def compute_ema(series: Sequence[float], period: int) -> Optional[float]:
    if period <= 0 or len(series) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = series[-period]
    for price in series[-period + 1 :]:
        ema = (price - ema) * multiplier + ema
    return ema


def compute_rsi(series: Sequence[float], period: int = 14) -> Optional[float]:
    if period <= 0 or len(series) <= period:
        return None
    gains = []
    losses = []
    for prev, curr in zip(series[-period - 1 : -1], series[-period:]):
        change = curr - prev
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _get_price_series(features: Dict[str, any]) -> List[float]:
    for key in ("close_prices", "closes", "prices"):
        series = features.get(key)
        if isinstance(series, list) and series:
            return [float(x) for x in series if x is not None]
    return []


def _get_latest_price(features: Dict[str, any]) -> Optional[float]:
    if "price" in features and isinstance(features["price"], (int, float)):
        return float(features["price"])
    series = _get_price_series(features)
    return series[-1] if series else None


def _resolve_indicator_value(
    features: Dict[str, any], indicator: Optional[str], lookback: Optional[int]
) -> Optional[float]:
    if not indicator:
        return None
    indicators = features.get("indicators", {})
    if isinstance(indicators, dict):
        bucket = indicators.get(indicator)
        if isinstance(bucket, dict):
            if lookback and lookback in bucket:
                return bucket[lookback]
            # Accept string keys such as "ema_20"
            key = f"{indicator}_{lookback}" if lookback else indicator
            if key in indicators:
                return indicators[key]
    # Compute from price series if absent
    series = _get_price_series(features)
    if not series:
        return None
    if indicator == "sma" and lookback:
        return compute_sma(series, lookback)
    if indicator == "ema" and lookback:
        return compute_ema(series, lookback)
    if indicator == "rsi":
        period = lookback or 14
        return compute_rsi(series, period)
    if indicator == "price":
        return series[-1]
    return None


def _entry_condition_met(
    condition: EntryCondition,
    features: Dict[str, any],
    last_price: Optional[float],
) -> bool:
    if last_price is None:
        return False
    indicator_value = _resolve_indicator_value(
        features, condition.indicator, condition.lookback
    )
    if condition.type == "breakout" and condition.level is not None:
        if condition.direction == "above":
            return last_price >= condition.level
        if condition.direction == "below":
            return last_price <= condition.level
    if condition.type == "crossover" and indicator_value is not None:
        if condition.direction == "above":
            return last_price > indicator_value
        if condition.direction == "below":
            return last_price < indicator_value
    if condition.type == "pullback" and indicator_value is not None:
        threshold = indicator_value * 0.995  # 0.5% pullback default
        if condition.direction == "above":
            return last_price <= threshold
        if condition.direction == "below":
            return last_price >= indicator_value * 1.005
    if condition.min_volume_multiple:
        volume = features.get("volume")
        avg_volume = features.get("avg_volume")
        if volume and avg_volume:
            return volume >= avg_volume * condition.min_volume_multiple
    return False


def _exit_condition_met(
    condition: ExitCondition,
    features: Dict[str, any],
    position_state: PositionState,
) -> bool:
    price = _get_latest_price(features)
    if price is None or position_state.is_flat() or not position_state.avg_entry_price:
        return False
    entry_price = position_state.avg_entry_price
    direction = 1 if position_state.side == "buy" else -1
    pct_change = (price - entry_price) / entry_price * 100 * direction
    if condition.type == "take_profit" and condition.take_profit_pct:
        return pct_change >= condition.take_profit_pct
    if condition.type == "stop_loss" and condition.stop_loss_pct:
        return pct_change <= -abs(condition.stop_loss_pct)
    if condition.type == "timed_exit" and condition.max_bars_in_trade:
        opened = position_state.opened_ts or 0
        current_ts = features.get("timestamp")
        if opened and current_ts:
            elapsed = current_ts - opened
            # assume timeframe minutes convert to seconds approx
            timeframe_sec = features.get("timeframe_seconds")
            if timeframe_sec:
                max_seconds = condition.max_bars_in_trade * timeframe_sec
                return elapsed >= max_seconds
    if condition.type == "trailing_stop" and condition.trail_pct:
        peak = features.get("peak_price")
        trough = features.get("trough_price")
        if position_state.side == "buy" and peak:
            drop_pct = (peak - price) / peak * 100
            return drop_pct >= condition.trail_pct
        if position_state.side == "sell" and trough:
            rebound_pct = (price - trough) / trough * 100
            return rebound_pct >= condition.trail_pct
    return False


def _default_entry_side(strategy: StrategySpec, condition: EntryCondition) -> TradeSide:
    if condition.side:
        return condition.side
    if strategy.mode == "mean_revert":
        return "sell"
    if condition.direction == "below":
        return "sell"
    return "buy"


def _derive_size_fraction(strategy: StrategySpec) -> float:
    return min(
        strategy.risk.max_fraction_of_balance,
        max(strategy.risk.risk_per_trade_fraction, 0.0),
    )


def _compute_target_price(
    price: float, pct: float, side: TradeSide, target_type: Literal["tp", "sl"]
) -> float:
    multiplier = 1 + pct / 100.0
    if side == "buy":
        return price * multiplier if target_type == "tp" else price / multiplier
    # sell/short
    return price / multiplier if target_type == "tp" else price * multiplier


def evaluate_signals(
    strategy: StrategySpec,
    features: Dict[str, any],
    position_state: PositionState,
) -> List[TradeSignal]:
    """Return deterministic trade signals based on strategy specs."""
    signals: List[TradeSignal] = []
    price = _get_latest_price(features)
    if price is None:
        return signals

    if position_state.is_flat():
        for condition in strategy.entry_conditions:
            if _entry_condition_met(condition, features, price):
                side = _default_entry_side(strategy, condition)
                size_fraction = _derive_size_fraction(strategy)
                take_profit = None
                stop_loss = None
                for exit_condition in strategy.exit_conditions:
                    if (
                        exit_condition.type == "take_profit"
                        and exit_condition.take_profit_pct
                    ):
                        take_profit = _compute_target_price(
                            price,
                            exit_condition.take_profit_pct,
                            side,
                            "tp",
                        )
                    if (
                        exit_condition.type == "stop_loss"
                        and exit_condition.stop_loss_pct
                    ):
                        stop_loss = _compute_target_price(
                            price,
                            exit_condition.stop_loss_pct,
                            side,
                            "sl",
                        )
                signals.append(
                    TradeSignal(
                        side=side,
                        size_fraction=size_fraction,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"entry:{condition.type}",
                    )
                )
                break
    else:
        for condition in strategy.exit_conditions:
            if _exit_condition_met(condition, features, position_state):
                signals.append(
                    TradeSignal(
                        side="close",
                        size_fraction=1.0,
                        reason=f"exit:{condition.type}",
                    )
                )
                break

    return signals
