"""CCXT-based data loader for remote exchange APIs."""

from __future__ import annotations

from datetime import datetime
import logging
import time
from typing import Callable

import ccxt
import pandas as pd

from .base import MarketDataBackend
from .caching import DataCache
from .normalization import ensure_datetime_index
from .utils import ensure_utc, timeframe_to_seconds

logger = logging.getLogger(__name__)


class CCXTAPILoader(MarketDataBackend):
    """Fetch OHLCV candles via ccxt and optionally cache the results."""

    name = "ccxt-api"

    def __init__(
        self,
        exchange_id: str = "coinbase",
        cache: DataCache | None = None,
        client_factory: Callable[[], ccxt.Exchange] | None = None,
    ) -> None:
        super().__init__()
        self.exchange_id = exchange_id
        self.cache = cache
        self._client_factory = client_factory or self._default_factory
        self._client: ccxt.Exchange | None = None

    def _default_factory(self) -> ccxt.Exchange:
        cls = getattr(ccxt, self.exchange_id)
        client = cls()
        client.enableRateLimit = True
        return client

    @property
    def client(self) -> ccxt.Exchange:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def fetch_history(
        self, symbol: str, start: datetime, end: datetime, granularity: str
    ) -> pd.DataFrame:
        start = ensure_utc(start)
        end = ensure_utc(end)
        cached = self.cache.load(symbol, granularity) if self.cache else None
        if cached is not None:
            frame = cached
        else:
            frame = self._download(symbol, start, end, granularity)
            if self.cache:
                self.cache.store(symbol, granularity, frame)

        sliced = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        if sliced.empty:
            logger.info(
                "cache miss or insufficient coverage for %s %s; downloading explicit window",
                symbol,
                granularity,
            )
            sliced = self._download(symbol, start, end, granularity)
            if self.cache:
                self.cache.store(symbol, granularity, sliced)
        ensure_datetime_index(sliced)
        self.validate_data(sliced)
        return sliced

    def _download(
        self, symbol: str, start: datetime, end: datetime, granularity: str
    ) -> pd.DataFrame:
        client = self.client
        gran_seconds = timeframe_to_seconds(granularity)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[float]] = []
        while since < end_ms:
            batch = self._fetch_with_retry(client, symbol, granularity, since)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts == since:
                break
            since = last_ts + gran_seconds * 1000
        if not rows:
            raise ValueError(f"No OHLCV data returned for {symbol} {granularity}")
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("time").drop(columns=["timestamp"])
        logger.debug(
            "downloaded rows=%s symbol=%s granularity=%s start=%s end=%s",
            len(frame),
            symbol,
            granularity,
            start.isoformat(),
            end.isoformat(),
        )
        return frame

    def _fetch_with_retry(
        self, client: ccxt.Exchange, symbol: str, granularity: str, since: int
    ) -> list[list[float]]:
        delay = 1.0
        for attempt in range(1, 4):
            try:
                return client.fetch_ohlcv(symbol, granularity, since=since)
            except ccxt.NetworkError as exc:
                if attempt >= 3:
                    raise
                logger.warning(
                    "Transient network error fetching %s %s (attempt %d/3): %s",
                    symbol,
                    granularity,
                    attempt,
                    exc,
                )
                time.sleep(delay)
                delay *= 2
        return []


__all__ = ["CCXTAPILoader"]
