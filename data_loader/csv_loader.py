"""Local CSV/Parquet-backed data loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .base import MarketDataBackend
from .normalization import ensure_datetime_index
from .utils import ensure_utc


@dataclass
class CSVLoaderConfig:
    root: Path
    filename_template: str = "{symbol}_{granularity}.csv"


class CSVDataLoader(MarketDataBackend):
    """Load normalized OHLCV data from CSV files."""

    name = "csv"

    def __init__(self, config: CSVLoaderConfig) -> None:
        super().__init__()
        self.config = config

    def _resolve_path(self, symbol: str, granularity: str) -> Path:
        safe_symbol = symbol.replace("/", "-")
        return self.config.root / self.config.filename_template.format(
            symbol=safe_symbol,
            granularity=granularity,
        )

    def fetch_history(
        self, symbol: str, start: datetime, end: datetime, granularity: str
    ) -> pd.DataFrame:
        path = self._resolve_path(symbol, granularity)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=True)
        frame = ensure_datetime_index(frame)
        start = ensure_utc(start)
        end = ensure_utc(end)
        sliced = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        if sliced.empty:
            raise ValueError(f"No rows available for {symbol} between {start} and {end}")
        self.validate_data(sliced)
        return sliced


__all__ = ["CSVDataLoader", "CSVLoaderConfig"]
