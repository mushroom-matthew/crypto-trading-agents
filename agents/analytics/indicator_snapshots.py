"""Indicator snapshot generation built on the metrics package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from metrics import technical as tech
from metrics.base import MetricResult, prepare_ohlcv_df
from schemas.llm_strategist import AssetState, IndicatorSnapshot


@dataclass(slots=True)
class IndicatorWindowConfig:
    """Timeframe-level indicator configuration."""

    timeframe: str
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    roc_short: int = 10
    roc_medium: int = 20
    realized_vol_short: int = 20
    realized_vol_medium: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    cycle_window: int = 200
    swing_lookback: int = 100


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return prepare_ohlcv_df(df)
    working = df.reset_index().rename(columns={"index": "timestamp"})
    if "time" in working.columns:
        working = working.rename(columns={"time": "timestamp"})
    return prepare_ohlcv_df(working)


def _latest(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _result_to_value(result: MetricResult) -> float | None:
    if not result.series_list:
        return None
    series = result.series_list[0].series
    return _latest(series)


def _macd_values(df: pd.DataFrame, fast: int, slow: int, signal: int) -> tuple[float | None, float | None, float | None]:
    macd_result = tech.macd(df, fast=fast, slow=slow, signal=signal)
    macd_line = macd_result.series_list[0].series
    signal_line = macd_result.series_list[1].series
    hist = macd_result.series_list[2].series
    return _latest(macd_line), _latest(signal_line), _latest(hist)


def _roc(series: pd.Series, period: int) -> float | None:
    if len(series) <= period:
        return None
    past = series.iloc[-period - 1]
    if past == 0:
        return None
    return float((series.iloc[-1] - past) / past)


def _realized_vol(series: pd.Series, window: int) -> float | None:
    log_returns = np.log(series / series.shift()).dropna()
    if len(log_returns) < window:
        return None
    return float(log_returns.tail(window).std(ddof=0))


def _cycle_indicators(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int = 200
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute cycle-based min/max over rolling window.

    Returns:
        cycle_high: Rolling max high over window
        cycle_low: Rolling min low over window
        cycle_range: (high - low) / close as percentage
        cycle_position: Where price sits in range (0=low, 1=high)
    """
    if len(high) < window:
        return None, None, None, None
    cycle_high = float(high.tail(window).max())
    cycle_low = float(low.tail(window).min())
    current_close = float(close.iloc[-1])
    if cycle_high == cycle_low or current_close == 0:
        return cycle_high, cycle_low, None, None
    cycle_range = (cycle_high - cycle_low) / current_close
    cycle_position = (current_close - cycle_low) / (cycle_high - cycle_low)
    return cycle_high, cycle_low, cycle_range, cycle_position


def _fibonacci_levels(
    cycle_high: float | None, cycle_low: float | None
) -> Dict[str, float | None]:
    """Compute Fibonacci retracement levels from cycle extremes.

    Retracement levels are measured from the high, representing pullback targets.
    """
    if cycle_high is None or cycle_low is None:
        return {
            "fib_236": None,
            "fib_382": None,
            "fib_500": None,
            "fib_618": None,
            "fib_786": None,
        }
    range_val = cycle_high - cycle_low
    return {
        "fib_236": cycle_high - range_val * 0.236,
        "fib_382": cycle_high - range_val * 0.382,
        "fib_500": cycle_high - range_val * 0.500,
        "fib_618": cycle_high - range_val * 0.618,
        "fib_786": cycle_high - range_val * 0.786,
    }


def _expansion_contraction(
    high: pd.Series, low: pd.Series, lookback: int = 100
) -> tuple[float | None, float | None, float | None]:
    """Compute expansion/contraction ratios from recent swings.

    Uses simple pivot detection to find local min/max points and measures
    the percentage moves between them.

    Returns:
        last_expansion_pct: Last swing low→high move (%)
        last_contraction_pct: Last swing high→low move (%)
        expansion_contraction_ratio: expansion / contraction
    """
    if len(high) < lookback:
        return None, None, None

    # Use recent data for swing detection
    recent_high = high.tail(lookback)
    recent_low = low.tail(lookback)

    # Find the highest and lowest points in the lookback
    high_idx = recent_high.idxmax()
    low_idx = recent_low.idxmin()
    high_val = float(recent_high.loc[high_idx])
    low_val = float(recent_low.loc[low_idx])

    if low_val == 0:
        return None, None, None

    # Determine which came first to calculate expansion vs contraction
    if high_idx > low_idx:
        # Low came first, then high: this is an expansion (bullish move)
        expansion_pct = ((high_val - low_val) / low_val) * 100.0
        # Look for contraction after the high
        after_high = recent_low.loc[high_idx:]
        if len(after_high) > 1:
            low_after = float(after_high.min())
            contraction_pct = ((high_val - low_after) / high_val) * 100.0
        else:
            contraction_pct = None
    else:
        # High came first, then low: this is a contraction (bearish move)
        contraction_pct = ((high_val - low_val) / high_val) * 100.0
        # Look for expansion after the low
        after_low = recent_high.loc[low_idx:]
        if len(after_low) > 1:
            high_after = float(after_low.max())
            expansion_pct = ((high_after - low_val) / low_val) * 100.0
        else:
            expansion_pct = None

    # Calculate ratio if both are available
    if expansion_pct is not None and contraction_pct is not None and contraction_pct > 0:
        ratio = expansion_pct / contraction_pct
    else:
        ratio = None

    return expansion_pct, contraction_pct, ratio


def compute_indicator_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: IndicatorWindowConfig | None = None,
) -> IndicatorSnapshot:
    """Produce an IndicatorSnapshot from OHLCV data."""

    window = config or IndicatorWindowConfig(timeframe=timeframe)
    prepared = _ensure_timestamp(df)
    close = prepared["close"]
    high = prepared["high"]
    low = prepared["low"]

    sma_short = _result_to_value(tech.sma(prepared, period=window.short_window))
    sma_medium = _result_to_value(tech.sma(prepared, period=window.medium_window))
    sma_long = _result_to_value(tech.sma(prepared, period=window.long_window))
    ema_short = _result_to_value(tech.ema(prepared, period=window.short_window))
    ema_medium = _result_to_value(tech.ema(prepared, period=window.medium_window))
    rsi_val = _result_to_value(tech.rsi(prepared, period=window.rsi_period))
    macd_val, macd_signal, macd_hist = _macd_values(prepared, window.macd_fast, window.macd_slow, window.macd_signal)
    atr_val = _result_to_value(tech.atr(prepared, period=window.atr_period))
    boll = tech.bollinger_bands(prepared, period=window.short_window, mult=2.0)
    boll_upper = _latest(boll.series_list[1].series)
    boll_lower = _latest(boll.series_list[2].series)
    donchian_upper = float(close.tail(window.short_window).max()) if len(close) >= window.short_window else None
    donchian_lower = float(close.tail(window.short_window).min()) if len(close) >= window.short_window else None
    roc_short = _roc(close, window.roc_short)
    roc_medium = _roc(close, window.roc_medium)
    realized_short = _realized_vol(close, window.realized_vol_short)
    realized_medium = _realized_vol(close, window.realized_vol_medium)

    # Cycle indicators (200-bar window for cyclical analysis)
    cycle_high, cycle_low, cycle_range, cycle_position = _cycle_indicators(
        close, high, low, window=window.cycle_window
    )

    # Fibonacci retracement levels
    fib_levels = _fibonacci_levels(cycle_high, cycle_low)

    # Expansion/contraction ratios
    expansion_pct, contraction_pct, exp_cont_ratio = _expansion_contraction(
        high, low, lookback=window.swing_lookback
    )

    as_of = prepared["timestamp"].iloc[-1].to_pydatetime()
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of,
        close=float(close.iloc[-1]),
        sma_short=sma_short,
        sma_medium=sma_medium,
        sma_long=sma_long,
        ema_short=ema_short,
        ema_medium=ema_medium,
        rsi_14=rsi_val,
        macd=macd_val,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr_14=atr_val,
        roc_short=roc_short,
        roc_medium=roc_medium,
        realized_vol_short=realized_short,
        realized_vol_medium=realized_medium,
        donchian_upper_short=donchian_upper,
        donchian_lower_short=donchian_lower,
        bollinger_upper=boll_upper,
        bollinger_lower=boll_lower,
        cycle_high_200=cycle_high,
        cycle_low_200=cycle_low,
        cycle_range_200=cycle_range,
        cycle_position=cycle_position,
        fib_236=fib_levels["fib_236"],
        fib_382=fib_levels["fib_382"],
        fib_500=fib_levels["fib_500"],
        fib_618=fib_levels["fib_618"],
        fib_786=fib_levels["fib_786"],
        last_expansion_pct=expansion_pct,
        last_contraction_pct=contraction_pct,
        expansion_contraction_ratio=exp_cont_ratio,
    )


def compute_indicator_matrix(
    symbol: str,
    frames_by_timeframe: Mapping[str, pd.DataFrame],
    base_config: IndicatorWindowConfig | None = None,
) -> list[IndicatorSnapshot]:
    """Return snapshots for all requested timeframes."""

    snapshots: list[IndicatorSnapshot] = []
    for timeframe, frame in frames_by_timeframe.items():
        config = base_config or IndicatorWindowConfig(timeframe=timeframe)
        snapshots.append(compute_indicator_snapshot(frame, symbol=symbol, timeframe=timeframe, config=config))
    return snapshots


def _trend_from_snapshot(snapshot: IndicatorSnapshot) -> str:
    if snapshot.sma_short and snapshot.sma_medium and snapshot.sma_long:
        if snapshot.sma_short > snapshot.sma_medium > snapshot.sma_long:
            return "uptrend"
        if snapshot.sma_short < snapshot.sma_medium < snapshot.sma_long:
            return "downtrend"
    if snapshot.ema_short and snapshot.ema_medium:
        if snapshot.ema_short > snapshot.ema_medium * 1.005:
            return "uptrend"
        if snapshot.ema_short < snapshot.ema_medium * 0.995:
            return "downtrend"
    return "sideways"


def _vol_from_snapshot(snapshot: IndicatorSnapshot) -> str:
    atr = snapshot.atr_14 or 0.0
    realized = snapshot.realized_vol_short or 0.0
    price = snapshot.close or 1.0
    atr_ratio = atr / price if price else 0.0
    vol_metric = max(atr_ratio, realized)
    if vol_metric < 0.01:
        return "low"
    if vol_metric < 0.02:
        return "normal"
    if vol_metric < 0.05:
        return "high"
    return "extreme"


def build_asset_state(symbol: str, snapshots: Iterable[IndicatorSnapshot]) -> AssetState:
    snapshot_list = list(snapshots)
    if not snapshot_list:
        raise ValueError("asset state requires at least one snapshot")
    primary = snapshot_list[0]
    # TODO: attach MarketStructureTelemetry alongside indicator snapshots when structure schema is plumbed into LLM inputs.
    return AssetState(
        symbol=symbol,
        indicators=snapshot_list,
        trend_state=_trend_from_snapshot(primary),  # type: ignore[arg-type]
        vol_state=_vol_from_snapshot(primary),  # type: ignore[arg-type]
    )
