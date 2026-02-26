"""Tests for ops_api/routers/structure.py — Runbook 58 Ops API contracts.

Validates:
- GET /structure/version returns engine version
- POST /structure/snapshots/{symbol}/compute builds and caches a snapshot
- GET /structure/snapshots/{symbol}/latest returns 404 when no snapshot
- GET /structure/snapshots/{symbol}/levels returns level list with filters
- GET /structure/snapshots/{symbol}/ladder returns multi-timeframe ladder
- GET /structure/snapshots/{symbol}/events returns events with filters
- GET /structure/snapshots/{symbol}/candidates returns stop/target/entry candidates
- GET /structure/snapshots returns all cached snapshots
"""
from __future__ import annotations

import os
# Set DB_DSN before any ops_api imports that may trigger database initialisation
# Must use aiosqlite driver since the ops_api database layer uses SQLAlchemy asyncio
os.environ.setdefault("DB_DSN", "sqlite+aiosqlite:///test_structure.db")

import pytest
from fastapi.testclient import TestClient

from ops_api.routers.structure import router, _snapshot_store
from schemas.structure_engine import STRUCTURE_ENGINE_VERSION


# Build a minimal FastAPI app for testing the structure router
from fastapi import FastAPI as _FastAPI
_app = _FastAPI()
_app.include_router(router)
client = TestClient(_app)


def _clear_store():
    _snapshot_store.clear()


@pytest.fixture(autouse=True)
def clean_store():
    """Clear the in-memory snapshot store before each test."""
    _clear_store()
    yield
    _clear_store()


BTC_REQUEST = {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "close": 50000.0,
    "atr_14": 750.0,
    "htf_daily_high": 51000.0,
    "htf_daily_low": 49000.0,
    "htf_prev_daily_high": 52000.0,
    "htf_prev_daily_low": 48000.0,
    "htf_5d_high": 53000.0,
    "htf_5d_low": 47000.0,
}


# ---------------------------------------------------------------------------
# /structure/version
# ---------------------------------------------------------------------------

def test_version_endpoint():
    resp = client.get("/structure/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == STRUCTURE_ENGINE_VERSION


# ---------------------------------------------------------------------------
# POST /structure/snapshots/{symbol}/compute
# ---------------------------------------------------------------------------

def test_compute_returns_summary():
    resp = client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTC-USD"
    assert data["reference_price"] == pytest.approx(50000.0)
    assert data["level_count"] > 0
    assert "snapshot_id" in data
    assert "snapshot_hash" in data
    assert len(data["snapshot_hash"]) == 64


def test_compute_stores_in_cache():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    assert "BTC-USD" in _snapshot_store


def test_compute_second_call_uses_prior_as_prior_snapshot():
    # First compute — no events expected (within 5d band)
    resp1 = client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    assert resp1.status_code == 200

    # Second compute with same data — prior snapshot now available
    resp2 = client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    assert resp2.status_code == 200
    # No assertion on events count since same data means no role changes
    # Just ensure it doesn't crash
    assert resp2.json()["symbol"] == "BTC-USD"


def test_compute_range_breakout_event_detected():
    req = {**BTC_REQUEST, "close": 54000.0, "htf_5d_high": 53000.0}
    resp = client.post("/structure/snapshots/BTC-USD/compute", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_count"] >= 1
    assert "Range Breakout" in " ".join(data["policy_trigger_reasons"])
    assert data["policy_event_priority"] == "high"


def test_compute_returns_quality_flags():
    resp = client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    data = resp.json()
    assert "available_timeframes" in data
    assert "missing_timeframes" in data
    assert "1d" in data["available_timeframes"]


# ---------------------------------------------------------------------------
# GET /structure/snapshots/{symbol}/latest
# ---------------------------------------------------------------------------

def test_latest_returns_404_when_no_snapshot():
    resp = client.get("/structure/snapshots/BTC-USD/latest")
    assert resp.status_code == 404


def test_latest_returns_summary_after_compute():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTC-USD"


# ---------------------------------------------------------------------------
# GET /structure/snapshots/{symbol}/levels
# ---------------------------------------------------------------------------

def test_levels_returns_404_when_no_snapshot():
    resp = client.get("/structure/snapshots/BTC-USD/levels")
    assert resp.status_code == 404


def test_levels_returns_list():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/levels")
    assert resp.status_code == 200
    levels = resp.json()
    assert isinstance(levels, list)
    assert len(levels) > 0


def test_levels_filter_by_role_resistance():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/levels?role=resistance")
    assert resp.status_code == 200
    levels = resp.json()
    assert all(l["role_now"] == "resistance" for l in levels)


def test_levels_filter_by_role_support():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/levels?role=support")
    assert resp.status_code == 200
    levels = resp.json()
    assert all(l["role_now"] == "support" for l in levels)


def test_levels_filter_by_eligible_for_stop():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/levels?eligible_for=stop_anchor")
    assert resp.status_code == 200
    levels = resp.json()
    assert all(l["eligible_for_stop_anchor"] for l in levels)


def test_levels_sorted_by_distance():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/levels")
    levels = resp.json()
    distances = [l["distance_abs"] for l in levels]
    assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# GET /structure/snapshots/{symbol}/ladder
# ---------------------------------------------------------------------------

def test_ladder_returns_404_when_no_snapshot():
    resp = client.get("/structure/snapshots/BTC-USD/ladder")
    assert resp.status_code == 404


def test_ladder_returns_dict_keyed_by_timeframe():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/ladder")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "1d" in data


def test_ladder_timeframe_filter():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/ladder?timeframe=1d")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"1d"}


def test_ladder_unknown_timeframe_returns_404():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/ladder?timeframe=1w")
    # 1w is not in S1 extracted from indicator (no daily_df provided)
    assert resp.status_code == 404


def test_ladder_contains_near_mid_far_buckets():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/ladder?timeframe=1d")
    data = resp.json()
    ladder_1d = data["1d"]
    expected_keys = {"near_supports", "mid_supports", "far_supports",
                     "near_resistances", "mid_resistances", "far_resistances",
                     "source_timeframe"}
    assert expected_keys.issubset(set(ladder_1d.keys()))


# ---------------------------------------------------------------------------
# GET /structure/snapshots/{symbol}/events
# ---------------------------------------------------------------------------

def test_events_returns_404_when_no_snapshot():
    resp = client.get("/structure/snapshots/BTC-USD/events")
    assert resp.status_code == 404


def test_events_returns_list():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/events")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_events_filter_reassessment_only():
    req = {**BTC_REQUEST, "close": 54000.0}  # triggers range_breakout
    client.post("/structure/snapshots/BTC-USD/compute", json=req)
    resp = client.get("/structure/snapshots/BTC-USD/events?reassessment_only=true")
    assert resp.status_code == 200
    events = resp.json()
    assert all(e["trigger_policy_reassessment"] for e in events)


def test_events_filter_by_severity():
    req = {**BTC_REQUEST, "close": 54000.0}
    client.post("/structure/snapshots/BTC-USD/compute", json=req)
    resp = client.get("/structure/snapshots/BTC-USD/events?severity=high")
    assert resp.status_code == 200
    events = resp.json()
    assert all(e["severity"] == "high" for e in events)


# ---------------------------------------------------------------------------
# GET /structure/snapshots/{symbol}/candidates
# ---------------------------------------------------------------------------

def test_candidates_returns_404_when_no_snapshot():
    resp = client.get("/structure/snapshots/BTC-USD/candidates")
    assert resp.status_code == 404


def test_candidates_returns_three_lists():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert "stop_candidates" in data
    assert "target_candidates" in data
    assert "entry_candidates" in data


def test_stop_candidates_are_eligible(self=None):
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/candidates")
    data = resp.json()
    for level in data["stop_candidates"]:
        assert level["eligible_for_stop_anchor"] is True


def test_target_candidates_are_eligible():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots/BTC-USD/candidates")
    data = resp.json()
    for level in data["target_candidates"]:
        assert level["eligible_for_target_anchor"] is True


# ---------------------------------------------------------------------------
# GET /structure/snapshots (list all)
# ---------------------------------------------------------------------------

def test_list_snapshots_empty_initially():
    resp = client.get("/structure/snapshots")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_snapshots_after_compute():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.get("/structure/snapshots")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["symbol"] == "BTC-USD"


def test_list_snapshots_multiple_symbols():
    eth_req = {**BTC_REQUEST, "symbol": "ETH-USD", "close": 3000.0}
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    client.post("/structure/snapshots/ETH-USD/compute", json=eth_req)
    resp = client.get("/structure/snapshots")
    assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# DELETE /structure/snapshots/{symbol}
# ---------------------------------------------------------------------------

def test_clear_snapshot():
    client.post("/structure/snapshots/BTC-USD/compute", json=BTC_REQUEST)
    resp = client.delete("/structure/snapshots/BTC-USD")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"
    assert "BTC-USD" not in _snapshot_store


def test_clear_nonexistent_symbol_ok():
    resp = client.delete("/structure/snapshots/UNKNOWN-USD")
    assert resp.status_code == 200
