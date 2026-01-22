"""Base interfaces for pluggable market data ingestion backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

import pandas as pd

from .normalization import (
    DEFAULT_OHLCV_COLUMNS,
    ensure_datetime_index,
    ensure_required_columns,
    validate_ohlcv,
)


class MarketDataBackend(ABC):
    """Abstract interface for data-ingestion sources."""

    name: str = "base"

    def __init__(self, required_columns: tuple[str, ...] | None = None) -> None:
        self.required_columns = required_columns or DEFAULT_OHLCV_COLUMNS

    @abstractmethod
    def fetch_history(
        self, symbol: str, start: datetime, end: datetime, granularity: str
    ) -> pd.DataFrame:
        """Return historical OHLCV data."""

    def stream_data(
        self, symbol: str, start: datetime, end: datetime, granularity: str
    ) -> Iterator[pd.Series]:
        """Default streaming implementation that iterates over ``fetch_history`` results."""

        frame = self.fetch_history(symbol, start, end, granularity)
        for _, row in frame.iterrows():
            yield row

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate standard OHLCV requirements."""

        ensure_datetime_index(df)
        ensure_required_columns(df, self.required_columns)
        validate_ohlcv(df)
        return True


__all__ = ["MarketDataBackend"]
