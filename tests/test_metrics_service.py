from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import tools.metrics_service as metrics_service


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime, datetime, str]] = []

    def fetch_history(self, symbol: str, start: datetime, end: datetime, granularity: str) -> pd.DataFrame:
        self.calls.append((symbol, start, end, granularity))
        idx = pd.DatetimeIndex(
            [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(3)],
        )
        return pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9],
                "close": [1.05, 2.05, 3.05],
                "volume": [100, 110, 120],
            },
            index=idx,
        )


def test_fetch_ohlcv_uses_shared_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DummyBackend()
    monkeypatch.setattr(metrics_service, "_BACKEND", backend)
    df = metrics_service.fetch_ohlcv("BTC/USD", "1h", limit=2)
    assert len(df) == 2
    assert backend.calls
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
