"""Market data worker responsible for fetching OHLCV and prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import logging

from data_loader import CCXTAPILoader, DataCache


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


logger = logging.getLogger(__name__)

_BACKEND = CCXTAPILoader(
    exchange_id="coinbase",
    cache=DataCache(root=Path("data/live_market_cache")),
)


def fetch_ohlcv_history(
    symbol: str,
    timeframe: str,
    lookback_days: int,
    *,
    allow_synthetic_fallback: bool = True,
) -> List[Candle]:
    """Retrieve OHLCV data from Coinbase.

    By default this falls back to synthetic data (useful for tests / non-critical
    flows). Callers that need real-only data (e.g. universe screening) should pass
    ``allow_synthetic_fallback=False`` and handle the exception.
    """

    try:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=lookback_days)
        frame = _BACKEND.fetch_history(symbol, start, end, timeframe)
        candles = [
            Candle(
                timestamp=ts.to_pydatetime().replace(tzinfo=None),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for ts, row in frame.iterrows()
        ]
        return candles
    except Exception:
        if not allow_synthetic_fallback:
            logger.exception("Real-only OHLCV fetch failed for %s %s", symbol, timeframe)
            raise
        logger.exception("Falling back to synthetic OHLCV for %s %s", symbol, timeframe)
        return _synthetic_candles(lookback_days)


def fetch_recent_prices(symbols: List[str]) -> dict[str, float]:
    client = _BACKEND.client
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            ticker = client.fetch_ticker(symbol)
            prices[symbol] = float(ticker["last"])
        except Exception:
            prices[symbol] = 100.0
    return prices


def _synthetic_candles(lookback_days: int) -> List[Candle]:
    now = datetime.utcnow()
    candles: List[Candle] = []
    for i in range(lookback_days * 24):
        price = 100 + i * 0.1
        ts = now - timedelta(hours=lookback_days * 24 - i)
        candles.append(Candle(ts, price - 0.5, price + 0.5, price - 1, price, 1000))
    return candles
