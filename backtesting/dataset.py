"""Historical OHLCV loaders with local caching for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt
import pandas as pd


OHLCVColumns = Literal["open", "high", "low", "close", "volume"]


@dataclass
class OHLCVConfig:
    pair: str
    timeframe: str = "1h"
    granularity_seconds: int = 3600


CACHE_DIR = Path("data/backtesting")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    suffix = timeframe[-1]
    value = int(timeframe[:-1])
    return value * units[suffix]


def _exchange() -> ccxt.Exchange:
    exchange = ccxt.coinbase()
    exchange.enableRateLimit = True
    return exchange


def _download(pair: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    exchange = _exchange()
    gran_seconds = _timeframe_seconds(timeframe)
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    data: list[list[float]] = []
    while since < end_ms:
        batch = exchange.fetch_ohlcv(pair, timeframe, since=since)
        if not batch:
            break
        data.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == since:
            break
        since = last_ts + gran_seconds * 1000
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns=["timestamp"])
    return df


def _cache_path(pair: str, timeframe: str) -> Path:
    safe_pair = pair.replace("/", "-")
    return CACHE_DIR / f"{safe_pair}_{timeframe}.csv"


def _load_from_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["time"], index_col="time")


def load_ohlcv(
    pair: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1h",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return OHLCV data for the specified pair/timeframe between start and end."""

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_path = _cache_path(pair, timeframe)
    df = _load_from_cache(cache_path) if use_cache else None
    if df is None:
        df = _download(pair, timeframe, start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=True, index_label="time")
    filtered = df.loc[(df.index >= start) & (df.index <= end)]
    if filtered.empty:
        # fallback: fetch with explicit start/end if cache doesn't cover range
        filtered = _download(pair, timeframe, start, end)
    return filtered
