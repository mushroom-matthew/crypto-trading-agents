"""Pure-Python technical indicator helpers for walk-forward analysis."""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple


def _ensure_sequence(values: Iterable[float]) -> List[float]:
    seq = [float(v) for v in values]
    if not seq:
        raise ValueError("indicator requires at least one value")
    return seq


def sma(values: Sequence[float], period: int) -> float:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    seq = _ensure_sequence(values)
    if len(seq) < period:
        raise ValueError("period longer than available data")
    return sum(seq[-period:]) / period


def ema(values: Sequence[float], period: int) -> float:
    """Exponential moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    seq = _ensure_sequence(values)
    if len(seq) < period:
        raise ValueError("period longer than available data")
    multiplier = 2 / (period + 1)
    ema_val = seq[-period]
    for price in seq[-period + 1 :]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def rsi(values: Sequence[float], period: int = 14) -> float:
    """Relative strength index."""
    if period <= 0:
        raise ValueError("period must be positive")
    seq = _ensure_sequence(values)
    if len(seq) <= period:
        raise ValueError("not enough data for RSI")
    gains = []
    losses = []
    for prev, curr in zip(seq[-period - 1 : -1], seq[-period:]):
        change = curr - prev
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> float:
    """Average True Range."""
    if period <= 0:
        raise ValueError("period must be positive")
    highs_seq = _ensure_sequence(highs)
    lows_seq = _ensure_sequence(lows)
    closes_seq = _ensure_sequence(closes)
    if not (len(highs_seq) == len(lows_seq) == len(closes_seq)):
        raise ValueError("high, low, close lengths must match")
    if len(highs_seq) <= period:
        raise ValueError("not enough data for ATR")
    trs: List[float] = []
    for i in range(len(highs_seq) - period, len(highs_seq)):
        high = highs_seq[i]
        low = lows_seq[i]
        prev_close = closes_seq[i - 1] if i > 0 else closes_seq[i]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


def rolling_median(values: Sequence[float], period: int) -> float:
    """Rolling median helper."""
    if period <= 0:
        raise ValueError("period must be positive")
    seq = _ensure_sequence(values)
    if len(seq) < period:
        raise ValueError("period longer than available data")
    return median(seq[-period:])


def volume_moving_average(volumes: Sequence[float], period: int) -> float:
    """Moving average for volume series."""
    return sma(volumes, period)


def ema_crossed(fast_series: Sequence[float], slow_series: Sequence[float]) -> bool:
    """Return True if fast EMA crossed above slow EMA on the most recent bar."""
    fast = _ensure_sequence(fast_series)
    slow = _ensure_sequence(slow_series)
    if len(fast) < 2 or len(slow) < 2:
        return False
    if len(fast) != len(slow):
        raise ValueError("EMA series must have equal lengths")
    prev_fast, prev_slow = fast[-2], slow[-2]
    curr_fast, curr_slow = fast[-1], slow[-1]
    return prev_fast <= prev_slow and curr_fast > curr_slow


def detect_support_levels(prices: Sequence[float], lookback: int = 20, tolerance: float = 0.003) -> List[float]:
    """Detect simple swing support levels."""
    seq = _ensure_sequence(prices)
    if len(seq) < lookback:
        return []
    supports: List[float] = []
    for i in range(lookback, len(seq) - lookback):
        window = seq[i - lookback : i + lookback + 1]
        pivot = seq[i]
        if pivot == min(window):
            if not supports or abs(pivot - supports[-1]) / supports[-1] > tolerance:
                supports.append(pivot)
    return supports[-5:]


def detect_resistance_levels(prices: Sequence[float], lookback: int = 20, tolerance: float = 0.003) -> List[float]:
    """Detect simple swing resistance levels."""
    seq = _ensure_sequence(prices)
    if len(seq) < lookback:
        return []
    resistances: List[float] = []
    for i in range(lookback, len(seq) - lookback):
        window = seq[i - lookback : i + lookback + 1]
        pivot = seq[i]
        if pivot == max(window):
            if not resistances or abs(pivot - resistances[-1]) / resistances[-1] > tolerance:
                resistances.append(pivot)
    return resistances[-5:]
