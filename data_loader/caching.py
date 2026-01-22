"""Simple CSV-based cache for normalized market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .normalization import TIME_INDEX_NAME, ensure_datetime_index


class DataCache:
    """Persist normalized OHLCV frames under ``data/market_data`` by default."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("data/market_data")
        self.root.mkdir(parents=True, exist_ok=True)

    def cache_path(self, symbol: str, granularity: str) -> Path:
        safe_symbol = symbol.replace("/", "-")
        return self.root / f"{safe_symbol}_{granularity}.csv"

    def load(self, symbol: str, granularity: str) -> pd.DataFrame | None:
        path = self.cache_path(symbol, granularity)
        if not path.exists():
            return None
        frame = pd.read_csv(path, parse_dates=[TIME_INDEX_NAME])
        frame = frame.set_index(TIME_INDEX_NAME)
        ensure_datetime_index(frame)
        return frame

    def store(self, symbol: str, granularity: str, df: pd.DataFrame) -> None:
        path = self.cache_path(symbol, granularity)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_write = df.copy()
        if TIME_INDEX_NAME not in df_to_write.columns:
            df_to_write.insert(0, TIME_INDEX_NAME, df_to_write.index)
        df_to_write.to_csv(path, index=False)


__all__ = ["DataCache"]
