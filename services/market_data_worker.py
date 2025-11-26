"""Market data worker responsible for fetching OHLCV and prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import ccxt


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _exchange() -> ccxt.Exchange:
    exchange = ccxt.coinbase()
    exchange.enableRateLimit = True
    return exchange


def fetch_ohlcv_history(symbol: str, timeframe: str, lookback_days: int) -> List[Candle]:
    """Retrieve OHLCV data from Coinbase; falls back to synthetic data on failure."""

    end_time = int(datetime.utcnow().timestamp() * 1000)
    since = end_time - lookback_days * 24 * 60 * 60 * 1000
    candles: List[Candle] = []
    try:
        exchange = _exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        for ts, op, high, low, close, vol in ohlcv:
            candles.append(Candle(datetime.fromtimestamp(ts / 1000), op, high, low, close, vol))
    except Exception:
        now = datetime.utcnow()
        for i in range(lookback_days * 24):
            price = 100 + i * 0.1
            ts = now - timedelta(hours=lookback_days * 24 - i)
            candles.append(Candle(ts, price - 0.5, price + 0.5, price - 1, price, 1000))
    return candles


def fetch_recent_prices(symbols: List[str]) -> dict[str, float]:
    exchange = _exchange()
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = float(ticker["last"])
        except Exception:
            prices[symbol] = 100.0
    return prices
