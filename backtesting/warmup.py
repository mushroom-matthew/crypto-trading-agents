"""Helpers for computing indicator warmup buffers for backtests."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Iterable, List

from agents.analytics.indicator_snapshots import IndicatorWindowConfig, scalper_config

MAX_WARMUP_TIMEFRAME_SECONDS = 4 * 3600  # focus on minutes-to-hours indicators


def timeframe_seconds(timeframe: str) -> int:
    value = int(timeframe[:-1])
    suffix = timeframe[-1]
    if suffix == "m":
        return value * 60
    if suffix == "h":
        return value * 3600
    if suffix == "d":
        return value * 86400
    raise ValueError(f"Unsupported timeframe {timeframe}")


def _indicator_window_config(timeframe: str) -> IndicatorWindowConfig:
    if timeframe_seconds(timeframe) <= 15 * 60:
        return scalper_config(timeframe)
    return IndicatorWindowConfig(timeframe=timeframe)


def indicator_warmup_bars(timeframe: str) -> int:
    """Return the number of bars needed to populate most indicators for a timeframe."""
    config = _indicator_window_config(timeframe)
    windows: List[int] = [
        config.short_window,
        config.medium_window,
        config.long_window,
        config.rsi_period,
        config.atr_period,
        config.roc_short,
        config.roc_medium,
        config.realized_vol_short,
        config.realized_vol_medium,
        config.macd_slow + config.macd_signal,
        config.cycle_window,
        config.swing_lookback,
        config.ema_fast,
        config.ema_very_fast,
        config.realized_vol_fast,
        config.ewma_vol_span,
    ]
    if config.vwap_window:
        windows.append(config.vwap_window)
    return max(windows)


def derive_higher_timeframes(
    base_timeframe: str,
    *,
    max_ratio: int = 288,
    max_timeframe_seconds: int = MAX_WARMUP_TIMEFRAME_SECONDS,
) -> List[str]:
    """Derive higher timeframes from the base, capped to minutes-to-hours ranges."""
    base_seconds = timeframe_seconds(base_timeframe)
    candidate_timeframes = ["5m", "15m", "30m", "1h", "6h", "1d"]
    derived = [base_timeframe]
    for tf in candidate_timeframes:
        tf_seconds = timeframe_seconds(tf)
        if tf_seconds > max_timeframe_seconds:
            continue
        if tf_seconds > base_seconds and tf_seconds % base_seconds == 0:
            ratio = tf_seconds // base_seconds
            if ratio <= max_ratio:
                derived.append(tf)
    seen = set()
    result: List[str] = []
    for tf in derived:
        if tf not in seen:
            seen.add(tf)
            result.append(tf)
    return result


def compute_indicator_warmup_candles(
    base_timeframe: str,
    *,
    include_derived: bool = True,
    max_timeframe_seconds: int = MAX_WARMUP_TIMEFRAME_SECONDS,
) -> int:
    """Return warmup candle count in base timeframe units."""
    base_seconds = timeframe_seconds(base_timeframe)
    if include_derived:
        timeframes: Iterable[str] = derive_higher_timeframes(
            base_timeframe,
            max_timeframe_seconds=max_timeframe_seconds,
        )
    else:
        timeframes = [base_timeframe]
    max_base_bars = 1
    for tf in timeframes:
        tf_seconds = timeframe_seconds(tf)
        if tf_seconds > max_timeframe_seconds:
            continue
        bars_needed = indicator_warmup_bars(tf)
        ratio = tf_seconds / base_seconds
        base_bars = int(math.ceil(bars_needed * ratio))
        if base_bars > max_base_bars:
            max_base_bars = base_bars
    return max_base_bars


def compute_indicator_warmup_delta(
    base_timeframe: str,
    *,
    include_derived: bool = True,
    max_timeframe_seconds: int = MAX_WARMUP_TIMEFRAME_SECONDS,
) -> timedelta:
    """Return warmup timedelta for the base timeframe."""
    base_seconds = timeframe_seconds(base_timeframe)
    candles = compute_indicator_warmup_candles(
        base_timeframe,
        include_derived=include_derived,
        max_timeframe_seconds=max_timeframe_seconds,
    )
    return timedelta(seconds=base_seconds * max(candles - 1, 0))


__all__ = [
    "MAX_WARMUP_TIMEFRAME_SECONDS",
    "timeframe_seconds",
    "indicator_warmup_bars",
    "derive_higher_timeframes",
    "compute_indicator_warmup_candles",
    "compute_indicator_warmup_delta",
]
