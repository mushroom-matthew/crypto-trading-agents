"""Indicator snapshot generation built on the metrics package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from metrics import technical as tech
from metrics import candlestick as cs
from metrics.base import MetricResult, prepare_ohlcv_df
from schemas.llm_strategist import AssetState, IndicatorSnapshot


@dataclass(slots=True)
class IndicatorWindowConfig:
    """Timeframe-level indicator configuration."""

    timeframe: str
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 200
    impulse_window: int = 24
    impulse_atr_mult: float = 0.8
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
    # Fast indicator presets for scalpers
    ema_fast: int = 5
    ema_very_fast: int = 8
    realized_vol_fast: int = 10
    ewma_vol_span: int = 10
    vwap_window: int | None = None  # None = cumulative VWAP
    vol_burst_threshold: float = 1.5  # volume_multiple threshold for burst detection


def scalper_config(timeframe: str) -> IndicatorWindowConfig:
    """Return optimized config for scalper (1m/5m) timeframes."""
    return IndicatorWindowConfig(
        timeframe=timeframe,
        short_window=10,
        medium_window=20,
        long_window=50,
        rsi_period=7,
        atr_period=7,
        roc_short=5,
        roc_medium=10,
        realized_vol_short=10,
        realized_vol_medium=20,
        macd_fast=6,
        macd_slow=13,
        macd_signal=5,
        cycle_window=50,
        swing_lookback=30,
        ema_fast=3,
        ema_very_fast=5,
        realized_vol_fast=5,
        ewma_vol_span=5,
        vwap_window=20,
        vol_burst_threshold=1.3,
    )


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


def _ewma_vol(series: pd.Series, span: int) -> float | None:
    """Compute exponentially weighted moving average volatility."""
    log_returns = np.log(series / series.shift()).dropna()
    if len(log_returns) < span:
        return None
    ewma_var = log_returns.ewm(span=span, adjust=False).var()
    return float(np.sqrt(ewma_var.iloc[-1])) if not pd.isna(ewma_var.iloc[-1]) else None


def _ewma_vol_series(series: pd.Series, span: int) -> pd.Series:
    """Compute EWMA volatility series."""
    log_returns = np.log(series / series.shift())
    ewma_var = log_returns.ewm(span=span, adjust=False).var()
    return np.sqrt(ewma_var)


def _vol_burst(volume_multiple: float | None, threshold: float) -> bool:
    """Detect volume burst (spike above threshold)."""
    if volume_multiple is None:
        return False
    return volume_multiple >= threshold


def _vwap_distance(close: float, vwap: float | None) -> float | None:
    """Compute percentage distance from VWAP."""
    if vwap is None or vwap == 0:
        return None
    return (close - vwap) / vwap * 100.0


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


def _daily_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR on daily bars using true-range rolling mean."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=period).mean()
    last = atr_series.iloc[-1]
    return float(last) if not pd.isna(last) else float((high - low).mean())


def compute_htf_structural_fields(
    bar_timestamp: "datetime",
    daily_df: pd.DataFrame | None,
) -> dict:
    """Compute prior-session daily anchor fields for a given bar timestamp.

    Finds the two most recently completed daily sessions before bar_timestamp
    and returns structural anchor values: high, low, open, close, ATR, 5-day
    rolling extremes, and position-in-range metric.

    Returns an empty dict if daily_df is None, empty, or has fewer than 2
    completed sessions before bar_timestamp (ATR warmup not yet satisfied).
    """
    if daily_df is None or daily_df.empty:
        return {}

    idx = daily_df.index
    if hasattr(idx, "date"):
        bar_date = bar_timestamp.date() if hasattr(bar_timestamp, "date") else bar_timestamp.to_pydatetime().date()
        completed = daily_df[idx.date < bar_date]
    else:
        # Fallback: compare as timestamps
        ts = pd.Timestamp(bar_timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        completed = daily_df[daily_df.index < ts.normalize()]

    if len(completed) < 2:
        return {}

    prev = completed.iloc[-1]    # Most recently completed session (yesterday)
    prev2 = completed.iloc[-2]   # Session before that

    daily_atr_val = _daily_atr(completed)
    last_5 = completed.tail(5)
    five_day_high = float(last_5["high"].max())
    five_day_low = float(last_5["low"].min())

    daily_high = float(prev["high"])
    daily_low = float(prev["low"])
    daily_close = float(prev["close"])
    daily_mid = (daily_high + daily_low) / 2.0

    return {
        "htf_daily_open": float(prev["open"]),
        "htf_daily_high": daily_high,
        "htf_daily_low": daily_low,
        "htf_daily_close": daily_close,
        "htf_prev_daily_high": float(prev2["high"]),
        "htf_prev_daily_low": float(prev2["low"]),
        "htf_prev_daily_open": float(prev2["open"]),
        "htf_daily_atr": daily_atr_val,
        "htf_daily_range_pct": (daily_high - daily_low) / max(daily_close, 1e-9) * 100.0,
        "htf_5d_high": five_day_high,
        "htf_5d_low": five_day_low,
        "htf_prev_daily_mid": (float(prev2["high"]) + float(prev2["low"])) / 2.0,
    }


def compute_indicator_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: IndicatorWindowConfig | None = None,
    daily_df: pd.DataFrame | None = None,
) -> IndicatorSnapshot:
    """Produce an IndicatorSnapshot from OHLCV data."""

    window = config or IndicatorWindowConfig(timeframe=timeframe)
    prepared = _ensure_timestamp(df)
    close = prepared["close"]
    open_series = prepared["open"] if "open" in prepared.columns else None
    high = prepared["high"]
    low = prepared["low"]
    volume_series = prepared["volume"] if "volume" in prepared.columns else None

    sma_short = _result_to_value(tech.sma(prepared, period=window.short_window))
    sma_medium = _result_to_value(tech.sma(prepared, period=window.medium_window))
    sma_long = _result_to_value(tech.sma(prepared, period=window.long_window))
    ema_short = _result_to_value(tech.ema(prepared, period=window.short_window))
    ema_medium_series = tech.ema(prepared, period=window.medium_window).series_list[0].series
    ema_medium = _latest(ema_medium_series)
    ema_long = _result_to_value(tech.ema(prepared, period=window.long_window))
    rsi_val = _result_to_value(tech.rsi(prepared, period=window.rsi_period))
    macd_val, macd_signal, macd_hist = _macd_values(prepared, window.macd_fast, window.macd_slow, window.macd_signal)
    atr_result = tech.atr(prepared, period=window.atr_period)
    atr_series = atr_result.series_list[0].series
    atr_val = _latest(atr_series)
    adx_result = tech.adx(prepared, period=14)
    adx_val = _latest(adx_result.series_list[2].series)
    boll = tech.bollinger_bands(prepared, period=window.short_window, mult=2.0)
    boll_middle = _latest(boll.series_list[0].series)
    boll_upper = _latest(boll.series_list[1].series)
    boll_lower = _latest(boll.series_list[2].series)
    donchian_upper = float(high.tail(window.short_window).max()) if len(high) >= window.short_window else None
    donchian_lower = float(low.tail(window.short_window).min()) if len(low) >= window.short_window else None
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

    # Candlestick morphology features (Runbook 38)
    candle_feats = cs.compute_candlestick_features(prepared, atr_series)
    _cf = candle_feats.iloc[-1]

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

    # Fast indicators for scalpers
    ema_fast_val = _result_to_value(tech.ema(prepared, period=window.ema_fast))
    ema_very_fast_val = _result_to_value(tech.ema(prepared, period=window.ema_very_fast))
    realized_fast = _realized_vol(close, window.realized_vol_fast)
    ewma_vol_val = _ewma_vol(close, window.ewma_vol_span)

    # VWAP calculation
    vwap_result = tech.vwap(prepared, window=window.vwap_window)
    vwap_val = _result_to_value(vwap_result)
    vwap_dist = _vwap_distance(float(close.iloc[-1]), vwap_val)

    # Volume burst detection
    vol_burst_flag = _vol_burst(volume_multiple, window.vol_burst_threshold)

    def _prev_value(series: pd.Series, offset: int) -> float | None:
        if len(series) <= offset:
            return None
        value = series.iloc[-(offset + 1)]
        if pd.isna(value):
            return None
        return float(value)

    prev_open = _prev_value(open_series, 1) if open_series is not None else None
    prev_high = _prev_value(high, 1)
    prev_low = _prev_value(low, 1)
    prev_close = _prev_value(close, 1)
    atr_prev1 = _prev_value(atr_series, 1)
    atr_prev2 = _prev_value(atr_series, 2)
    atr_prev3 = _prev_value(atr_series, 3)
    atr_rising_3 = None
    if atr_val is not None and atr_prev1 is not None and atr_prev2 is not None and atr_prev3 is not None:
        atr_rising_3 = atr_val > atr_prev1 > atr_prev2 > atr_prev3
    impulse_ema50_24 = None
    if atr_val is not None and atr_val > 0 and len(close) >= window.impulse_window:
        try:
            diff_series = close - ema_medium_series
            max_diff = diff_series.tail(window.impulse_window).max()
            if not pd.isna(max_diff):
                impulse_ema50_24 = float(max_diff) >= (window.impulse_atr_mult * atr_val)
        except Exception:
            impulse_ema50_24 = None

    as_of = prepared["timestamp"].iloc[-1].to_pydatetime()

    # HTF structural anchor fields (Runbook 41)
    htf = compute_htf_structural_fields(as_of, daily_df)
    if htf:
        current_close = float(close.iloc[-1])
        daily_atr_htf = htf.get("htf_daily_atr", 1.0) or 1.0
        daily_mid_htf = (htf.get("htf_daily_high", current_close) + htf.get("htf_daily_low", current_close)) / 2.0
        htf["htf_price_vs_daily_mid"] = (current_close - daily_mid_htf) / max(daily_atr_htf, 1e-9)

    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of,
        open=float(open_series.iloc[-1]) if open_series is not None else None,
        high=float(high.iloc[-1]) if high is not None else None,
        low=float(low.iloc[-1]) if low is not None else None,
        close=float(close.iloc[-1]),
        volume=volume_val,
        volume_multiple=volume_multiple,
        sma_short=sma_short,
        sma_medium=sma_medium,
        sma_long=sma_long,
        ema_short=ema_short,
        ema_medium=ema_medium,
        ema_long=ema_long,
        ema_50=ema_medium,
        ema_200=ema_long,
        rsi_14=rsi_val,
        macd=macd_val,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr_14=atr_val,
        atr_14_prev1=atr_prev1,
        atr_14_prev2=atr_prev2,
        atr_14_prev3=atr_prev3,
        atr_14_rising_3=atr_rising_3,
        adx_14=adx_val,
        impulse_ema50_24=impulse_ema50_24,
        prev_open=prev_open,
        prev_high=prev_high,
        prev_low=prev_low,
        prev_close=prev_close,
        roc_short=roc_short,
        roc_medium=roc_medium,
        realized_vol_short=realized_short,
        realized_vol_medium=realized_medium,
        donchian_upper_short=donchian_upper,
        donchian_lower_short=donchian_lower,
        bollinger_upper=boll_upper,
        bollinger_lower=boll_lower,
        bollinger_middle=boll_middle,
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
        # Fast indicators for scalpers
        ema_fast=ema_fast_val,
        ema_very_fast=ema_very_fast_val,
        realized_vol_fast=realized_fast,
        ewma_vol=ewma_vol_val,
        vwap=vwap_val,
        vwap_distance_pct=vwap_dist,
        vol_burst=vol_burst_flag,
        # Candlestick morphology (Runbook 38)
        candle_body_pct=float(_cf["candle_body_pct"]),
        candle_upper_wick_pct=float(_cf["candle_upper_wick_pct"]),
        candle_lower_wick_pct=float(_cf["candle_lower_wick_pct"]),
        candle_strength=float(_cf["candle_strength"]),
        is_bullish=float(_cf["is_bullish"]),
        is_bearish=float(_cf["is_bearish"]),
        is_doji=float(_cf["is_doji"]),
        is_hammer=float(_cf["is_hammer"]),
        is_shooting_star=float(_cf["is_shooting_star"]),
        is_pin_bar=float(_cf["is_pin_bar"]),
        is_bullish_engulfing=float(_cf["is_bullish_engulfing"]),
        is_bearish_engulfing=float(_cf["is_bearish_engulfing"]),
        is_inside_bar=float(_cf["is_inside_bar"]),
        is_outside_bar=float(_cf["is_outside_bar"]),
        is_impulse_candle=float(_cf["is_impulse_candle"]),
        # HTF structural anchors (Runbook 41)
        htf_daily_open=htf.get("htf_daily_open"),
        htf_daily_high=htf.get("htf_daily_high"),
        htf_daily_low=htf.get("htf_daily_low"),
        htf_daily_close=htf.get("htf_daily_close"),
        htf_prev_daily_high=htf.get("htf_prev_daily_high"),
        htf_prev_daily_low=htf.get("htf_prev_daily_low"),
        htf_prev_daily_open=htf.get("htf_prev_daily_open"),
        htf_daily_atr=htf.get("htf_daily_atr"),
        htf_daily_range_pct=htf.get("htf_daily_range_pct"),
        htf_price_vs_daily_mid=htf.get("htf_price_vs_daily_mid"),
        htf_5d_high=htf.get("htf_5d_high"),
        htf_5d_low=htf.get("htf_5d_low"),
        htf_prev_daily_mid=htf.get("htf_prev_daily_mid"),
    )


def precompute_indicator_frame(
    df: pd.DataFrame,
    config: IndicatorWindowConfig | None = None,
) -> pd.DataFrame:
    """Precompute indicator series for efficient snapshot lookup."""
    window = config or IndicatorWindowConfig(timeframe="unknown")
    prepared = _ensure_timestamp(df)
    close = prepared["close"]
    open_series = prepared["open"] if "open" in prepared.columns else None
    high = prepared["high"]
    low = prepared["low"]
    volume_series = prepared["volume"] if "volume" in prepared.columns else None

    sma_short = tech.sma(prepared, period=window.short_window).series_list[0].series
    sma_medium = tech.sma(prepared, period=window.medium_window).series_list[0].series
    sma_long = tech.sma(prepared, period=window.long_window).series_list[0].series
    ema_short = tech.ema(prepared, period=window.short_window).series_list[0].series
    ema_medium = tech.ema(prepared, period=window.medium_window).series_list[0].series
    ema_long = tech.ema(prepared, period=window.long_window).series_list[0].series
    rsi_val = tech.rsi(prepared, period=window.rsi_period).series_list[0].series
    macd_result = tech.macd(prepared, fast=window.macd_fast, slow=window.macd_slow, signal=window.macd_signal)
    macd_line = macd_result.series_list[0].series
    macd_signal = macd_result.series_list[1].series
    macd_hist = macd_result.series_list[2].series
    atr_val = tech.atr(prepared, period=window.atr_period).series_list[0].series
    adx_val = tech.adx(prepared, period=14).series_list[2].series
    boll = tech.bollinger_bands(prepared, period=window.short_window, mult=2.0)
    boll_middle = boll.series_list[0].series
    boll_upper = boll.series_list[1].series
    boll_lower = boll.series_list[2].series
    donchian_upper = high.rolling(window=window.short_window, min_periods=window.short_window).max()
    donchian_lower = low.rolling(window=window.short_window, min_periods=window.short_window).min()
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

    # Fast indicators for scalpers
    ema_fast = tech.ema(prepared, period=window.ema_fast).series_list[0].series
    ema_very_fast = tech.ema(prepared, period=window.ema_very_fast).series_list[0].series
    realized_fast = _realized_vol_series(close, window.realized_vol_fast)
    ewma_vol = _ewma_vol_series(close, window.ewma_vol_span)

    # VWAP calculation
    vwap_series = tech.vwap(prepared, window=window.vwap_window).series_list[0].series
    vwap_distance_pct = (close - vwap_series) / vwap_series.replace(0.0, np.nan) * 100.0

    # Volume burst detection
    vol_burst = volume_multiple >= window.vol_burst_threshold

    # Candlestick morphology features (Runbook 38)
    candle_feats_frame = cs.compute_candlestick_features(prepared, atr_val)

    prev_open = open_series.shift(1) if open_series is not None else pd.Series(np.nan, index=prepared.index)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    atr_prev1 = atr_val.shift(1)
    atr_prev2 = atr_val.shift(2)
    atr_prev3 = atr_val.shift(3)
    atr_rising_3 = (atr_val > atr_prev1) & (atr_prev1 > atr_prev2) & (atr_prev2 > atr_prev3)
    impulse_ema50_24 = (close - ema_medium).rolling(
        window=window.impulse_window,
        min_periods=window.impulse_window,
    ).max() >= (window.impulse_atr_mult * atr_val)

    frame = pd.DataFrame(
        {
            "timestamp": prepared["timestamp"],
            "open": open_series,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume_series,
            "volume_multiple": volume_multiple,
            "sma_short": sma_short,
            "sma_medium": sma_medium,
            "sma_long": sma_long,
            "ema_short": ema_short,
            "ema_medium": ema_medium,
            "ema_long": ema_long,
            "ema_50": ema_medium,
            "ema_200": ema_long,
            "rsi_14": rsi_val,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr_14": atr_val,
            "atr_14_prev1": atr_prev1,
            "atr_14_prev2": atr_prev2,
            "atr_14_prev3": atr_prev3,
            "atr_14_rising_3": atr_rising_3,
            "adx_14": adx_val,
            "impulse_ema50_24": impulse_ema50_24,
            "prev_open": prev_open,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "prev_close": prev_close,
            "roc_short": roc_short,
            "roc_medium": roc_medium,
            "realized_vol_short": realized_short,
            "realized_vol_medium": realized_medium,
            "donchian_upper_short": donchian_upper,
            "donchian_lower_short": donchian_lower,
            "bollinger_upper": boll_upper,
            "bollinger_lower": boll_lower,
            "bollinger_middle": boll_middle,
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
            # Fast indicators for scalpers
            "ema_fast": ema_fast,
            "ema_very_fast": ema_very_fast,
            "realized_vol_fast": realized_fast,
            "ewma_vol": ewma_vol,
            "vwap": vwap_series,
            "vwap_distance_pct": vwap_distance_pct,
            "vol_burst": vol_burst,
            # Candlestick morphology (Runbook 38)
            "candle_body_pct": candle_feats_frame["candle_body_pct"],
            "candle_upper_wick_pct": candle_feats_frame["candle_upper_wick_pct"],
            "candle_lower_wick_pct": candle_feats_frame["candle_lower_wick_pct"],
            "candle_strength": candle_feats_frame["candle_strength"],
            "is_bullish": candle_feats_frame["is_bullish"],
            "is_bearish": candle_feats_frame["is_bearish"],
            "is_doji": candle_feats_frame["is_doji"],
            "is_hammer": candle_feats_frame["is_hammer"],
            "is_shooting_star": candle_feats_frame["is_shooting_star"],
            "is_pin_bar": candle_feats_frame["is_pin_bar"],
            "is_bullish_engulfing": candle_feats_frame["is_bullish_engulfing"],
            "is_bearish_engulfing": candle_feats_frame["is_bearish_engulfing"],
            "is_inside_bar": candle_feats_frame["is_inside_bar"],
            "is_outside_bar": candle_feats_frame["is_outside_bar"],
            "is_impulse_candle": candle_feats_frame["is_impulse_candle"],
        }
    )
    frame.set_index("timestamp", inplace=True)
    return frame


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


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
        open=_optional_float(row.get("open")),
        high=_optional_float(row.get("high")),
        low=_optional_float(row.get("low")),
        close=float(row["close"]),
        volume=_optional_float(row.get("volume")),
        volume_multiple=_optional_float(row.get("volume_multiple")),
        sma_short=_optional_float(row.get("sma_short")),
        sma_medium=_optional_float(row.get("sma_medium")),
        sma_long=_optional_float(row.get("sma_long")),
        ema_short=_optional_float(row.get("ema_short")),
        ema_medium=_optional_float(row.get("ema_medium")),
        ema_long=_optional_float(row.get("ema_long")),
        ema_50=_optional_float(row.get("ema_50")),
        ema_200=_optional_float(row.get("ema_200")),
        rsi_14=_optional_float(row.get("rsi_14")),
        macd=_optional_float(row.get("macd")),
        macd_signal=_optional_float(row.get("macd_signal")),
        macd_hist=_optional_float(row.get("macd_hist")),
        atr_14=_optional_float(row.get("atr_14")),
        atr_14_prev1=_optional_float(row.get("atr_14_prev1")),
        atr_14_prev2=_optional_float(row.get("atr_14_prev2")),
        atr_14_prev3=_optional_float(row.get("atr_14_prev3")),
        atr_14_rising_3=_optional_bool(row.get("atr_14_rising_3")),
        adx_14=_optional_float(row.get("adx_14")),
        impulse_ema50_24=_optional_bool(row.get("impulse_ema50_24")),
        prev_open=_optional_float(row.get("prev_open")),
        prev_high=_optional_float(row.get("prev_high")),
        prev_low=_optional_float(row.get("prev_low")),
        prev_close=_optional_float(row.get("prev_close")),
        roc_short=_optional_float(row.get("roc_short")),
        roc_medium=_optional_float(row.get("roc_medium")),
        realized_vol_short=_optional_float(row.get("realized_vol_short")),
        realized_vol_medium=_optional_float(row.get("realized_vol_medium")),
        donchian_upper_short=_optional_float(row.get("donchian_upper_short")),
        donchian_lower_short=_optional_float(row.get("donchian_lower_short")),
        bollinger_upper=_optional_float(row.get("bollinger_upper")),
        bollinger_lower=_optional_float(row.get("bollinger_lower")),
        bollinger_middle=_optional_float(row.get("bollinger_middle")),
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
        # Fast indicators for scalpers
        ema_fast=_optional_float(row.get("ema_fast")),
        ema_very_fast=_optional_float(row.get("ema_very_fast")),
        realized_vol_fast=_optional_float(row.get("realized_vol_fast")),
        ewma_vol=_optional_float(row.get("ewma_vol")),
        vwap=_optional_float(row.get("vwap")),
        vwap_distance_pct=_optional_float(row.get("vwap_distance_pct")),
        vol_burst=_optional_bool(row.get("vol_burst")),
        # Candlestick morphology (Runbook 38) — precomputed in frame
        candle_body_pct=_optional_float(row.get("candle_body_pct")),
        candle_upper_wick_pct=_optional_float(row.get("candle_upper_wick_pct")),
        candle_lower_wick_pct=_optional_float(row.get("candle_lower_wick_pct")),
        candle_strength=_optional_float(row.get("candle_strength")),
        is_bullish=_optional_float(row.get("is_bullish")),
        is_bearish=_optional_float(row.get("is_bearish")),
        is_doji=_optional_float(row.get("is_doji")),
        is_hammer=_optional_float(row.get("is_hammer")),
        is_shooting_star=_optional_float(row.get("is_shooting_star")),
        is_pin_bar=_optional_float(row.get("is_pin_bar")),
        is_bullish_engulfing=_optional_float(row.get("is_bullish_engulfing")),
        is_bearish_engulfing=_optional_float(row.get("is_bearish_engulfing")),
        is_inside_bar=_optional_float(row.get("is_inside_bar")),
        is_outside_bar=_optional_float(row.get("is_outside_bar")),
        is_impulse_candle=_optional_float(row.get("is_impulse_candle")),
        # HTF structural anchors (Runbook 41) — not precomputed; always None from frame
        # Applied by the runner via model_copy after snapshot_from_frame returns.
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
    if vol_metric < 0.015:
        return "low"
    if vol_metric < 0.03:
        return "normal"
    if vol_metric < 0.07:
        return "high"
    return "extreme"


def build_asset_state(
    symbol: str,
    snapshots: Iterable[IndicatorSnapshot],
    include_regime_assessment: bool = False,
) -> AssetState:
    snapshot_list = list(snapshots)
    if not snapshot_list:
        raise ValueError("asset state requires at least one snapshot")
    primary = snapshot_list[0]

    # Optionally compute regime assessment using deterministic classifier
    regime_assessment = None
    if include_regime_assessment:
        from trading_core.regime_classifier import classify_regime
        regime_assessment = classify_regime(primary)

    return AssetState(
        symbol=symbol,
        indicators=snapshot_list,
        trend_state=_trend_from_snapshot(primary),  # type: ignore[arg-type]
        vol_state=_vol_from_snapshot(primary),  # type: ignore[arg-type]
        regime_assessment=regime_assessment,
    )
