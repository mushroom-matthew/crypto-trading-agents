"""Market data worker responsible for fetching OHLCV and prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_ohlcv_history(symbol: str, timeframe: str, lookback_days: int) -> List[Candle]:
    """Retrieve OHLCV data from Coinbase or another provider.

    TODO: Implement actual API call and error handling. For now returns dummy candles.
    """

    now = datetime.utcnow()
    candles: List[Candle] = []
    for i in range(lookback_days * 24):
        price = 100 + i * 0.1
        ts = now - timedelta(hours=lookback_days * 24 - i)
        candles.append(Candle(ts, price - 0.5, price + 0.5, price - 1, price, 1000))
    return candles


def fetch_recent_prices(symbols: List[str]) -> dict[str, float]:
    """Return the latest price per symbol."""

    return {symbol: 100.0 for symbol in symbols}
