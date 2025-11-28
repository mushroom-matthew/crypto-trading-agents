"""Indicator snapshot generation built on the metrics package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Mapping

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
    return AssetState(
        symbol=symbol,
        indicators=snapshot_list,
        trend_state=_trend_from_snapshot(primary),  # type: ignore[arg-type]
        vol_state=_vol_from_snapshot(primary),  # type: ignore[arg-type]
    )
