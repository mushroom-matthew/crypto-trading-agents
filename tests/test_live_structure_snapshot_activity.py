from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tools.paper_trading import (
    _build_structure_snapshot_payloads,
    build_structure_snapshots_activity,
)


def _indicator_snapshot(symbol: str = "BTC-USD", timeframe: str = "1m") -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": datetime(2026, 4, 22, 20, 0, tzinfo=timezone.utc).isoformat(),
        "close": 50000.0,
        "volume": 1000.0,
        "atr_14": 750.0,
        "rsi_14": 55.0,
    }


def _structure_snapshot(symbol: str = "BTC-USD"):
    from schemas.structure_engine import StructureSnapshot

    now = datetime(2026, 4, 22, 20, 0, tzinfo=timezone.utc)
    return StructureSnapshot(
        snapshot_id=str(uuid4()),
        snapshot_hash="abcd1234" * 8,
        symbol=symbol,
        as_of_ts=now,
        generated_at_ts=now,
        source_timeframe="1m",
        reference_price=50000.0,
        levels=[],
    )


def test_build_structure_snapshot_payloads_returns_serialized_snapshots(monkeypatch):
    from services import structure_engine

    expected = _structure_snapshot()

    def fake_build_structure_snapshot(*args, **kwargs):
        return expected

    monkeypatch.setattr(structure_engine, "build_structure_snapshot", fake_build_structure_snapshot)

    payloads = _build_structure_snapshot_payloads(
        ["BTC-USD"],
        {"BTC-USD": _indicator_snapshot()},
        "1m",
        raw_ohlcv_data={},
        prior_snapshots={},
    )

    assert "BTC-USD" in payloads
    assert payloads["BTC-USD"]["snapshot_id"] == expected.snapshot_id
    assert payloads["BTC-USD"]["reference_price"] == expected.reference_price


@pytest.mark.asyncio
async def test_build_structure_snapshots_activity_returns_payloads(monkeypatch):
    from services import structure_engine

    expected = _structure_snapshot()

    def fake_build_structure_snapshot(*args, **kwargs):
        return expected

    monkeypatch.setattr(structure_engine, "build_structure_snapshot", fake_build_structure_snapshot)

    payloads = await build_structure_snapshots_activity(
        ["BTC-USD"],
        {"BTC-USD": _indicator_snapshot()},
        "1m",
        {},
        {},
    )

    assert payloads["BTC-USD"]["snapshot_hash"] == expected.snapshot_hash
