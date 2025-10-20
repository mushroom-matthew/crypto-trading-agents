"""Tier I technical indicators built on OHLCV data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import base


def _close(df: pd.DataFrame) -> pd.Series:
    return df["close"]


def _high(df: pd.DataFrame) -> pd.Series:
    return df["high"]


def _low(df: pd.DataFrame) -> pd.Series:
    return df["low"]


def _volume(df: pd.DataFrame) -> pd.Series:
    return df["volume"]


def sma(df: pd.DataFrame, period: int = 20) -> base.MetricResult:
    period = int(period)
    values = _close(df).rolling(window=period, min_periods=period).mean()
    column = f"SMA_{period}"
    return base.build_single_series("SMA", "value", column, values)


def ema(df: pd.DataFrame, period: int = 20) -> base.MetricResult:
    period = int(period)
    values = _close(df).ewm(span=period, adjust=False).mean()
    column = f"EMA_{period}"
    return base.build_single_series("EMA", "value", column, values)


def wma(df: pd.DataFrame, period: int = 20) -> base.MetricResult:
    period = int(period)
    values = base.linear_weighted_moving_average(_close(df), period)
    column = f"WMA_{period}"
    return base.build_single_series("WMA", "value", column, values)


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> base.MetricResult:
    close = _close(df)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=int(signal), adjust=False).mean()
    histogram = macd_line - signal_line

    label = f"MACD_{int(fast)}_{int(slow)}_{int(signal)}"

    return base.MetricResult(
        feature="MACD",
        series_list=[
            base.MetricSeries(key="value", column=label, series=macd_line),
            base.MetricSeries(key="signal", column=f"{label}_signal", series=signal_line),
            base.MetricSeries(key="hist", column=f"{label}_hist", series=histogram),
        ],
    )


def rsi(df: pd.DataFrame, period: int = 14) -> base.MetricResult:
    period = int(period)
    close = _close(df)
    delta = close.diff()
    gains = delta.where(delta > 0.0, 0.0)
    losses = (-delta).where(delta < 0.0, 0.0)

    avg_gain = base.wilder_smoothing(gains, period)
    avg_loss = base.wilder_smoothing(losses, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    rsi_values = rsi_values.where(avg_loss != 0.0, 100.0)
    rsi_values = rsi_values.where(avg_gain != 0.0, 0.0)
    rsi_values = rsi_values.fillna(0.0)
    rsi_values = base.ensure_min_period(rsi_values, period)

    column = f"RSI_{period}"
    return base.build_single_series("RSI", "value", column, rsi_values)


def bollinger_bands(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> base.MetricResult:
    period = int(period)
    mult = float(mult)

    close = _close(df)
    basis = close.rolling(window=period, min_periods=period).mean()
    std = base.rolling_std(close, period)

    upper = basis + mult * std
    lower = basis - mult * std

    bandwidth = base.safe_divide(upper - lower, basis.abs())
    pct_b = base.safe_divide(close - lower, (upper - lower).replace(0.0, np.nan))

    label = f"BBANDS_{period}_{mult}"
    return base.MetricResult(
        feature="BollingerBands",
        series_list=[
            base.MetricSeries(key="basis", column=f"{label}_basis", series=basis),
            base.MetricSeries(key="upper", column=f"{label}_upper", series=upper),
            base.MetricSeries(key="lower", column=f"{label}_lower", series=lower),
            base.MetricSeries(key="bandwidth", column=f"{label}_bandwidth", series=bandwidth),
            base.MetricSeries(key="pctB", column=f"{label}_pctB", series=pct_b),
        ],
    )


def atr(df: pd.DataFrame, period: int = 14) -> base.MetricResult:
    period = int(period)
    high = _high(df)
    low = _low(df)
    close = _close(df)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr_values = base.wilder_smoothing(true_range, period)
    atr_values = base.ensure_min_period(atr_values, period)

    column = f"ATR_{period}"
    return base.build_single_series("ATR", "value", column, atr_values)


def adx(df: pd.DataFrame, period: int = 14) -> base.MetricResult:
    period = int(period)
    high = _high(df)
    low = _low(df)
    close = _close(df)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr_values = base.wilder_smoothing(true_range, period)

    plus_smoothed = base.wilder_smoothing(pd.Series(plus_dm, index=df.index), period)
    minus_smoothed = base.wilder_smoothing(pd.Series(minus_dm, index=df.index), period)
    plus_di = 100.0 * base.safe_divide(plus_smoothed, atr_values.replace(0.0, np.nan))
    minus_di = 100.0 * base.safe_divide(minus_smoothed, atr_values.replace(0.0, np.nan))

    plus_di = base.ensure_min_period(plus_di, period)
    minus_di = base.ensure_min_period(minus_di, period)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * base.safe_divide(di_diff, di_sum.replace(0.0, np.nan))
    adx_values = base.wilder_smoothing(dx, period)
    adx_values = base.ensure_min_period(adx_values, period)

    label = f"ADX_{period}"
    return base.MetricResult(
        feature="ADX",
        series_list=[
            base.MetricSeries(key="+DI", column=f"{label}_plus_di", series=plus_di),
            base.MetricSeries(key="-DI", column=f"{label}_minus_di", series=minus_di),
            base.MetricSeries(key="ADX", column=label, series=adx_values),
        ],
    )


def roc(df: pd.DataFrame, period: int = 12) -> base.MetricResult:
    period = int(period)
    close = _close(df)
    roc_values = close / close.shift(period) - 1.0
    roc_values = base.ensure_min_period(roc_values, period + 1)
    column = f"ROC_{period}"
    return base.build_single_series("ROC", "value", column, roc_values)


def obv(df: pd.DataFrame) -> base.MetricResult:
    close = _close(df)
    volume = _volume(df)
    base.validate_non_negative(volume.fillna(0.0), "volume")

    direction = close.diff().apply(np.sign).fillna(0.0)
    obv_change = direction * volume.fillna(0.0)
    obv_series = obv_change.cumsum()

    return base.build_single_series("OBV", "value", "OBV", obv_series)


def vwap(
    df: pd.DataFrame,
    window: Optional[int] = None,
    session_column: str = "session_id",
) -> base.MetricResult:
    volume = _volume(df).fillna(0.0)
    tp = (_high(df) + _low(df) + _close(df)) / 3.0

    volume = volume.astype("float64")
    tp_volume = tp * volume

    if session_column in df.columns:
        session_ids = df[session_column]
        cum_tp_vol = base.cumulative_by_group(tp_volume, session_ids)
        cum_vol = base.cumulative_by_group(volume, session_ids)
        vwap_series = base.safe_divide(cum_tp_vol, cum_vol.replace(0.0, np.nan))
        column_name = "VWAP_session"
        key = "value"
    elif window is not None:
        window = int(window)
        cum_tp_vol = tp_volume.rolling(window=window, min_periods=1).sum()
        cum_vol = volume.rolling(window=window, min_periods=1).sum()
        vwap_series = base.safe_divide(cum_tp_vol, cum_vol.replace(0.0, np.nan))
        column_name = f"VWAP_{window}"
        key = "value"
    else:
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = volume.cumsum()
        vwap_series = base.safe_divide(cum_tp_vol, cum_vol.replace(0.0, np.nan))
        column_name = "VWAP"
        key = "value"

    return base.build_single_series("VWAP", key, column_name, vwap_series)
