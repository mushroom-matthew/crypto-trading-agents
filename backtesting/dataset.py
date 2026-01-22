"""Historical OHLCV loaders built on the shared data-loader interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
