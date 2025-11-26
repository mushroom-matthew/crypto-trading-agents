"""Indicator engine that produces summaries for planner and signal agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    ema_fast = sum(closes[-20:]) / min(len(closes), 20)
    ema_slow = sum(closes[-50:]) / min(len(closes), 50)
    atr = sum(high - low for high, low in zip(highs[-14:], lows[-14:])) / max(1, min(len(highs), 14))
    rolling_high = max(highs[-50:])
    rolling_low = min(lows[-50:])
    volume_multiple = 1.0
    return IndicatorSummary(symbol, ema_fast, ema_slow, atr, rolling_high, rolling_low, volume_multiple)
