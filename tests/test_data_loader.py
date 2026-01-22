from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data_loader import CCXTAPILoader, CSVDataLoader, CSVLoaderConfig, DataCache
from data_loader.utils import timeframe_to_seconds


class DummyExchange:
    def __init__(self, batches: list[list[list[float]]]) -> None:
        self._batches = batches
        self.fetch_calls = 0

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int) -> list[list[float]]:
        self.fetch_calls += 1
        if self._batches:
            return self._batches.pop(0)
        return []


def test_timeframe_to_seconds() -> None:
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("2h") == 7200


def test_data_cache_round_trip(tmp_path) -> None:
    cache = DataCache(root=tmp_path)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [1.2, 2.2],
            "low": [0.8, 1.8],
            "close": [1.1, 2.1],
            "volume": [100, 110],
        },
        index=[now, now + timedelta(hours=1)],
    )
    cache.store("BTC/USD", "1h", frame)
    loaded = cache.load("BTC/USD", "1h")
    assert loaded is not None
    assert list(loaded.index) == list(frame.index)


def test_csv_loader_filters_requested_window(tmp_path) -> None:
    cache = DataCache(root=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [1.2, 2.2, 3.2],
            "low": [0.8, 1.8, 2.8],
            "close": [1.1, 2.1, 3.1],
            "volume": [100, 110, 120],
        },
        index=[start, start + timedelta(hours=1), start + timedelta(hours=2)],
    )
    cache.store("BTC/USD", "1h", frame)
    loader = CSVDataLoader(
        CSVLoaderConfig(root=tmp_path),
    )
    window = loader.fetch_history(
        "BTC/USD",
        start + timedelta(minutes=30),
        start + timedelta(hours=1, minutes=30),
        "1h",
    )
    assert len(window) == 1
    assert window["close"].iloc[0] == 2.1


def test_ccxt_loader_uses_client_factory(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    batches = [
        [
            [int(start.timestamp() * 1000), 1, 1.2, 0.8, 1.1, 100],
            [int((start + timedelta(hours=1)).timestamp() * 1000), 1.1, 1.3, 0.9, 1.2, 110],
        ]
    ]
    dummy = DummyExchange(batches=batches)
    loader = CCXTAPILoader(cache=DataCache(root=tmp_path), client_factory=lambda: dummy)
    frame = loader.fetch_history(
        "BTC/USD",
        start,
        start + timedelta(hours=2),
        "1h",
    )
    assert not frame.empty
    assert dummy.fetch_calls == 1
    assert frame["close"].iloc[-1] == 1.2
