from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import tools.market_data as market_data


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime, datetime, str]] = []

    def fetch_history(self, symbol: str, start: datetime, end: datetime, granularity: str) -> pd.DataFrame:
        self.calls.append((symbol, start, end, granularity))
        idx = pd.DatetimeIndex(
            [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(5)],
        )
        return pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [1.1, 2.1, 3.1, 4.1, 5.1],
                "low": [0.9, 1.9, 2.9, 3.9, 4.9],
                "close": [1.05, 2.05, 3.05, 4.05, 5.05],
                "volume": [100, 110, 120, 130, 140],
            },
            index=idx,
        )


@pytest.mark.asyncio
async def test_fetch_historical_ohlcv_returns_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DummyBackend()
    monkeypatch.setattr(market_data, "_HISTORICAL_BACKEND", backend)
    ticks = await market_data.fetch_historical_ohlcv("BTC/USD", "1m", limit=3)
    assert len(ticks) == 3
    assert backend.calls
    assert ticks[0]["last"] == pytest.approx(3.05)
    assert "datetime" in ticks[0]
