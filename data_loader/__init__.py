"""Composable market data ingestion adapters."""

from .api_loader import CCXTAPILoader
from .base import MarketDataBackend
from .caching import DataCache
from .csv_loader import CSVDataLoader, CSVLoaderConfig

__all__ = [
    "CCXTAPILoader",
    "CSVDataLoader",
    "CSVLoaderConfig",
    "DataCache",
    "MarketDataBackend",
]
