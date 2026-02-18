"""Historical OHLCV loaders built on the shared data-loader interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from data_loader import CCXTAPILoader, DataCache, MarketDataBackend
from data_loader.utils import ensure_utc


@dataclass
class OHLCVConfig:
    pair: str
    timeframe: str = "1h"
    granularity_seconds: int = 3600


OHLCVColumns = Literal["open", "high", "low", "close", "volume"]


def _default_backend() -> MarketDataBackend:
    cache = DataCache(root=Path("data/backtesting"))
    return CCXTAPILoader(exchange_id="coinbase", cache=cache)


_BACKEND = _default_backend()


def load_ohlcv(
    pair: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1h",
    use_cache: bool = True,
    backend: MarketDataBackend | None = None,
) -> pd.DataFrame:
    """Return OHLCV data for the specified pair/timeframe between start and end."""

    selected_backend = backend or _BACKEND
    if not use_cache and isinstance(selected_backend, CCXTAPILoader) and selected_backend.cache is not None:
        selected_backend = CCXTAPILoader(exchange_id=selected_backend.exchange_id, cache=None)
    start = ensure_utc(start)
    end = ensure_utc(end)
    return selected_backend.fetch_history(pair, start, end, timeframe)


def load_with_htf(
    pair: str,
    start: datetime,
    end: datetime,
    base_timeframe: str = "1h",
    use_cache: bool = True,
    backend: MarketDataBackend | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load base timeframe OHLCV plus daily OHLCV for the same date range.

    The daily bars include a 20-day buffer before start for ATR(14) warmup.

    Returns:
        (base_df, daily_df) where daily_df covers start - 20d to end.
        If base_timeframe is already '1d', daily_df == base_df.
    """
    base_df = load_ohlcv(pair, start, end, base_timeframe, use_cache, backend)
    if base_timeframe == "1d":
        return base_df, base_df
    daily_start = ensure_utc(start) - timedelta(days=20)
    daily_df = load_ohlcv(pair, daily_start, end, "1d", use_cache, backend)
    return base_df, daily_df
