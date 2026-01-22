"""Helper utilities to bridge the metrics package with MCP tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from data_loader import CCXTAPILoader, DataCache
from data_loader.utils import timeframe_to_seconds
from metrics import compute_metrics, list_metrics
from metrics.base import REQUIRED_COLUMNS


CACHE_DIR = Path("data/market_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_CACHE_DIR = CACHE_DIR / "raw_feed"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BACKEND = CCXTAPILoader(exchange_id="coinbase", cache=DataCache(root=RAW_CACHE_DIR))


@dataclass(slots=True)
class MetricsRequest:
    symbol: str
    timeframe: str = "1h"
    limit: int = 500
    use_cache: bool = True
    cache_path: Optional[Path] = None
    data_path: Optional[Path] = None  # explicit CSV override


def cache_file_for(symbol: str, timeframe: str, limit: int) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}_{limit}.csv"


def load_cached_dataframe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def save_cache_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    seconds = timeframe_to_seconds(timeframe)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(seconds=seconds * (limit + 5))
    frame = _BACKEND.fetch_history(symbol, start, end, timeframe).tail(limit).copy()
    frame = frame.reset_index()
    if "time" in frame.columns:
        frame = frame.rename(columns={"time": "timestamp"})
    else:
        frame = frame.rename(columns={"index": "timestamp"})
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = frame[columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"})


def resolve_dataframe(request: MetricsRequest, fetch_if_missing: bool = True) -> pd.DataFrame:
    if request.data_path:
        path = request.data_path
        if not Path(path).exists():
            raise FileNotFoundError(f"Provided data_path does not exist: {path}")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    cache_path = request.cache_path or cache_file_for(request.symbol, request.timeframe, request.limit)
    if request.use_cache:
        cached = load_cached_dataframe(cache_path)
        if cached is not None:
            return cached
        if not fetch_if_missing:
            raise FileNotFoundError(f"No cached data at {cache_path}")

    df = fetch_ohlcv(request.symbol, request.timeframe, request.limit)
    if request.use_cache:
        save_cache_dataframe(df, cache_path)
    return df


def compute_metrics_payload(
    df: pd.DataFrame,
    feature_names: Iterable[str],
    params: Optional[dict] = None,
    output: str = "wide",
) -> pd.DataFrame:
    params = params or {}
    return compute_metrics(df, features=list(feature_names), params=params, output=output)


async def load_dataframe_async(request: MetricsRequest, fetch_if_missing: bool = True) -> pd.DataFrame:
    return await asyncio.to_thread(resolve_dataframe, request, fetch_if_missing)


def fetch_and_cache_dataframe(
    symbol: str,
    timeframe: str,
    limit: int,
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, Path]:
    df = fetch_ohlcv(symbol, timeframe, limit)
    path = cache_path or cache_file_for(symbol, timeframe, limit)
    save_cache_dataframe(df, path)
    return df, path


async def fetch_and_cache_async(
    symbol: str,
    timeframe: str,
    limit: int,
    cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, Path]:
    return await asyncio.to_thread(fetch_and_cache_dataframe, symbol, timeframe, limit, cache_path)


async def compute_metrics_async(
    df: pd.DataFrame,
    feature_names: Iterable[str],
    params: Optional[dict] = None,
    output: str = "wide",
) -> pd.DataFrame:
    return await asyncio.to_thread(compute_metrics_payload, df, feature_names, params, output)


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required OHLCV columns: {missing}")
