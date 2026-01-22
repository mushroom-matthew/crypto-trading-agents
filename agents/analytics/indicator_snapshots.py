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


def _roc_series(series: pd.Series, period: int) -> pd.Series:
    shifted = series.shift(period)
    return (series - shifted) / shifted.replace(0.0, np.nan)


def _realized_vol_series(series: pd.Series, window: int) -> pd.Series:
    log_returns = np.log(series / series.shift())
    return log_returns.rolling(window=window, min_periods=window).std(ddof=0)


def _cycle_indicator_series(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    cycle_high = high.rolling(window=window, min_periods=window).max()
    cycle_low = low.rolling(window=window, min_periods=window).min()
    range_val = (cycle_high - cycle_low).replace(0.0, np.nan)
    close_safe = close.replace(0.0, np.nan)
    cycle_range = range_val / close_safe
    cycle_position = (close - cycle_low) / range_val
    return cycle_high, cycle_low, cycle_range, cycle_position


def _expansion_contraction_series(
    high: pd.Series, low: pd.Series, lookback: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    size = len(high)
    expansion = np.full(size, np.nan, dtype="float64")
    contraction = np.full(size, np.nan, dtype="float64")
    ratio = np.full(size, np.nan, dtype="float64")
    highs = high.to_numpy(dtype="float64", copy=False)
    lows = low.to_numpy(dtype="float64", copy=False)

    if size < lookback or lookback <= 0:
        return (
            pd.Series(expansion, index=high.index),
            pd.Series(contraction, index=high.index),
            pd.Series(ratio, index=high.index),
        )

    for idx in range(lookback - 1, size):
        window_high = highs[idx - lookback + 1 : idx + 1]
        window_low = lows[idx - lookback + 1 : idx + 1]
        if np.isnan(window_high).all() or np.isnan(window_low).all():
            continue
        high_idx = int(np.nanargmax(window_high))
        low_idx = int(np.nanargmin(window_low))
        high_val = window_high[high_idx]
        low_val = window_low[low_idx]
        if low_val == 0 or np.isnan(low_val) or np.isnan(high_val):
            continue
        if high_idx > low_idx:
            expansion_pct = (high_val - low_val) / low_val * 100.0
            low_after = np.nanmin(window_low[high_idx:]) if high_idx < lookback else np.nan
            contraction_pct = (high_val - low_after) / high_val * 100.0 if high_val else np.nan
        else:
            contraction_pct = (high_val - low_val) / high_val * 100.0 if high_val else np.nan
            high_after = np.nanmax(window_high[low_idx:]) if low_idx < lookback else np.nan
            expansion_pct = (high_after - low_val) / low_val * 100.0 if low_val else np.nan
        expansion[idx] = expansion_pct
        contraction[idx] = contraction_pct
        if contraction_pct and not np.isnan(contraction_pct) and contraction_pct > 0:
            ratio[idx] = expansion_pct / contraction_pct

    return (
        pd.Series(expansion, index=high.index),
        pd.Series(contraction, index=high.index),
        pd.Series(ratio, index=high.index),
    )


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
    volume_series = prepared["volume"] if "volume" in prepared.columns else None

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
    volume_val = _latest(volume_series) if volume_series is not None else None
    volume_multiple = None
    if volume_series is not None and len(volume_series) >= window.short_window:
        mean_volume = float(volume_series.tail(window.short_window).mean())
        if mean_volume and not np.isnan(mean_volume):
            volume_multiple = float(volume_val / mean_volume) if volume_val is not None else None

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
        volume=volume_val,
        volume_multiple=volume_multiple,
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


def precompute_indicator_frame(
    df: pd.DataFrame,
    config: IndicatorWindowConfig | None = None,
) -> pd.DataFrame:
    """Precompute indicator series for efficient snapshot lookup."""
    window = config or IndicatorWindowConfig(timeframe="unknown")
    prepared = _ensure_timestamp(df)
    close = prepared["close"]
    high = prepared["high"]
    low = prepared["low"]
    volume_series = prepared["volume"] if "volume" in prepared.columns else None

    sma_short = tech.sma(prepared, period=window.short_window).series_list[0].series
    sma_medium = tech.sma(prepared, period=window.medium_window).series_list[0].series
    sma_long = tech.sma(prepared, period=window.long_window).series_list[0].series
    ema_short = tech.ema(prepared, period=window.short_window).series_list[0].series
    ema_medium = tech.ema(prepared, period=window.medium_window).series_list[0].series
    rsi_val = tech.rsi(prepared, period=window.rsi_period).series_list[0].series
    macd_result = tech.macd(prepared, fast=window.macd_fast, slow=window.macd_slow, signal=window.macd_signal)
    macd_line = macd_result.series_list[0].series
    macd_signal = macd_result.series_list[1].series
    macd_hist = macd_result.series_list[2].series
    atr_val = tech.atr(prepared, period=window.atr_period).series_list[0].series
    boll = tech.bollinger_bands(prepared, period=window.short_window, mult=2.0)
    boll_upper = boll.series_list[1].series
    boll_lower = boll.series_list[2].series
    donchian_upper = close.rolling(window=window.short_window, min_periods=window.short_window).max()
    donchian_lower = close.rolling(window=window.short_window, min_periods=window.short_window).min()
    roc_short = _roc_series(close, window.roc_short)
    roc_medium = _roc_series(close, window.roc_medium)
    realized_short = _realized_vol_series(close, window.realized_vol_short)
    realized_medium = _realized_vol_series(close, window.realized_vol_medium)

    if volume_series is None:
        volume_series = pd.Series(np.nan, index=prepared.index, dtype="float64")
    mean_volume = volume_series.rolling(window=window.short_window, min_periods=window.short_window).mean()
    volume_multiple = volume_series / mean_volume.replace(0.0, np.nan)

    cycle_high, cycle_low, cycle_range, cycle_position = _cycle_indicator_series(
        close, high, low, window=window.cycle_window
    )
    range_val = (cycle_high - cycle_low)
    fib_236 = cycle_high - range_val * 0.236
    fib_382 = cycle_high - range_val * 0.382
    fib_500 = cycle_high - range_val * 0.500
    fib_618 = cycle_high - range_val * 0.618
    fib_786 = cycle_high - range_val * 0.786

    expansion_pct, contraction_pct, exp_ratio = _expansion_contraction_series(
        high, low, lookback=window.swing_lookback
    )

    frame = pd.DataFrame(
        {
            "timestamp": prepared["timestamp"],
            "close": close,
            "volume": volume_series,
            "volume_multiple": volume_multiple,
            "sma_short": sma_short,
            "sma_medium": sma_medium,
            "sma_long": sma_long,
            "ema_short": ema_short,
            "ema_medium": ema_medium,
            "rsi_14": rsi_val,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr_14": atr_val,
            "roc_short": roc_short,
            "roc_medium": roc_medium,
            "realized_vol_short": realized_short,
            "realized_vol_medium": realized_medium,
            "donchian_upper_short": donchian_upper,
            "donchian_lower_short": donchian_lower,
            "bollinger_upper": boll_upper,
            "bollinger_lower": boll_lower,
            "cycle_high_200": cycle_high,
            "cycle_low_200": cycle_low,
            "cycle_range_200": cycle_range,
            "cycle_position": cycle_position,
            "fib_236": fib_236,
            "fib_382": fib_382,
            "fib_500": fib_500,
            "fib_618": fib_618,
            "fib_786": fib_786,
            "last_expansion_pct": expansion_pct,
            "last_contraction_pct": contraction_pct,
            "expansion_contraction_ratio": exp_ratio,
        }
    )
    frame.set_index("timestamp", inplace=True)
    return frame


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def snapshot_from_frame(
    frame: pd.DataFrame,
    timestamp: datetime,
    symbol: str,
    timeframe: str,
) -> IndicatorSnapshot | None:
    """Build an IndicatorSnapshot from a precomputed indicator frame."""
    if frame is None or frame.empty:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    if ts in frame.index:
        row = frame.loc[ts]
    else:
        pos = frame.index.get_indexer([ts], method="pad")
        if pos[0] == -1:
            return None
        row = frame.iloc[pos[0]]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    as_of = row.name
    if isinstance(as_of, pd.Timestamp):
        as_of_dt = as_of.to_pydatetime()
    else:
        as_of_dt = pd.Timestamp(as_of, utc=True).to_pydatetime()

    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of_dt,
        close=float(row["close"]),
        volume=_optional_float(row.get("volume")),
        volume_multiple=_optional_float(row.get("volume_multiple")),
        sma_short=_optional_float(row.get("sma_short")),
        sma_medium=_optional_float(row.get("sma_medium")),
        sma_long=_optional_float(row.get("sma_long")),
        ema_short=_optional_float(row.get("ema_short")),
        ema_medium=_optional_float(row.get("ema_medium")),
        rsi_14=_optional_float(row.get("rsi_14")),
        macd=_optional_float(row.get("macd")),
        macd_signal=_optional_float(row.get("macd_signal")),
        macd_hist=_optional_float(row.get("macd_hist")),
        atr_14=_optional_float(row.get("atr_14")),
        roc_short=_optional_float(row.get("roc_short")),
        roc_medium=_optional_float(row.get("roc_medium")),
        realized_vol_short=_optional_float(row.get("realized_vol_short")),
        realized_vol_medium=_optional_float(row.get("realized_vol_medium")),
        donchian_upper_short=_optional_float(row.get("donchian_upper_short")),
        donchian_lower_short=_optional_float(row.get("donchian_lower_short")),
        bollinger_upper=_optional_float(row.get("bollinger_upper")),
        bollinger_lower=_optional_float(row.get("bollinger_lower")),
        cycle_high_200=_optional_float(row.get("cycle_high_200")),
        cycle_low_200=_optional_float(row.get("cycle_low_200")),
        cycle_range_200=_optional_float(row.get("cycle_range_200")),
        cycle_position=_optional_float(row.get("cycle_position")),
        fib_236=_optional_float(row.get("fib_236")),
        fib_382=_optional_float(row.get("fib_382")),
        fib_500=_optional_float(row.get("fib_500")),
        fib_618=_optional_float(row.get("fib_618")),
        fib_786=_optional_float(row.get("fib_786")),
        last_expansion_pct=_optional_float(row.get("last_expansion_pct")),
        last_contraction_pct=_optional_float(row.get("last_contraction_pct")),
        expansion_contraction_ratio=_optional_float(row.get("expansion_contraction_ratio")),
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
