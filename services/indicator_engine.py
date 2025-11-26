"""Indicator engine that produces summaries for planner and signal agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from services.market_data_worker import Candle


@dataclass
class IndicatorSummary:
    symbol: str
    ema_fast: float
    ema_slow: float
    atr: float
    rolling_high: float
    rolling_low: float
    volume_multiple: float


def summarize_indicators(symbol: str, candles: List[Candle]) -> IndicatorSummary:
    """Compute a handful of indicators needed by planner and signal agent."""

    if not candles:
        raise ValueError("No candles supplied")
    df = pd.DataFrame([c.__dict__ for c in candles])
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    volumes = df["volume"]
    ema_fast = closes.ewm(span=min(len(closes), 20), adjust=False).mean().iloc[-1]
    ema_slow = closes.ewm(span=min(len(closes), 50), adjust=False).mean().iloc[-1]
    tr = pd.concat(
        [
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs(),
        ],
        axis=1,
    )
    atr = tr.max(axis=1).rolling(window=min(len(tr), 14)).mean().iloc[-1]
    rolling_high = highs.tail(min(len(highs), 50)).max()
    rolling_low = lows.tail(min(len(lows), 50)).min()
    vol_ma = volumes.rolling(window=min(len(volumes), 20)).mean().iloc[-1]
    volume_multiple = volumes.iloc[-1] / vol_ma if vol_ma else 1.0
    return IndicatorSummary(symbol, ema_fast, ema_slow, atr, rolling_high, rolling_low, volume_multiple)
