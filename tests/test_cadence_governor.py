"""Tests for R77 CadenceGovernor."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from services.cadence_governor import CadenceGovernor, CadenceGovernorState


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_governor(
    start: datetime | None = None,
    target_min: int = 5,
    target_max: int = 10,
) -> CadenceGovernor:
    gov = CadenceGovernor()
    gov.initialize(session_start=start or _now(), target_min=target_min, target_max=target_max)
    return gov


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initial_state():
    gov = _make_governor()
    assert gov.state.round_trips_completed == 0
    assert gov.state.cadence_status == "not_started"
    assert gov.state.round_trips_target_min == 5
    assert gov.state.round_trips_target_max == 10


def test_initialize_idempotent():
    gov = _make_governor()
    start = gov.state.session_start_iso
    gov.initialize(session_start=_now() + timedelta(hours=1))  # should not overwrite
    assert gov.state.session_start_iso == start


# ---------------------------------------------------------------------------
# Record round trip
# ---------------------------------------------------------------------------

def test_record_single_win():
    gov = _make_governor()
    gov.record_round_trip_complete("BTC-USD", "win", r_achieved=1.5)
    assert gov.state.round_trips_completed == 1
    assert len(gov.state.round_trip_history) == 1


def test_record_multiple():
    gov = _make_governor()
    for i in range(5):
        gov.record_round_trip_complete("ETH-USD", "win", r_achieved=float(i))
    assert gov.state.round_trips_completed == 5


def test_round_trip_history_capped_at_100():
    gov = _make_governor()
    for i in range(120):
        gov.record_round_trip_complete("BTC-USD", "neutral")
    assert len(gov.state.round_trip_history) <= 100


# ---------------------------------------------------------------------------
# Projection and status
# ---------------------------------------------------------------------------

def test_projection_below_target():
    # Session started 8h ago with 0 round trips → projected = 0 → below_target
    start = _now() - timedelta(hours=8)
    gov = _make_governor(start=start)
    gov._refresh_projection(_now())
    assert gov.state.cadence_status == "below_target"
    assert gov.state.projected_24h_rate == 0.0


def test_projection_above_target():
    # Session started 1h ago with 12 round trips → projected = 12/1 * 24 = 288 → above_target
    start = _now() - timedelta(hours=1)
    gov = _make_governor(start=start)
    for _ in range(12):
        gov.record_round_trip_complete("BTC-USD", "win")
    gov._refresh_projection(_now())
    assert gov.state.cadence_status == "above_target"
    assert gov.state.projected_24h_rate > 10.0


def test_projection_on_track():
    # 7.5 completed in ~24h → on_track (target 5–10)
    start = _now() - timedelta(hours=24)
    gov = _make_governor(start=start)
    for _ in range(7):
        gov.record_round_trip_complete("ETH-USD", "win")
    gov._refresh_projection(_now())
    assert gov.state.cadence_status == "on_track"


# ---------------------------------------------------------------------------
# should_widen_symbol_set
# ---------------------------------------------------------------------------

def test_should_widen_when_below_and_sufficient_time():
    start = _now() - timedelta(hours=4)
    gov = _make_governor(start=start)
    # 0 completed → below target, 4h elapsed → should widen
    assert gov.should_widen_symbol_set(_now()) is True


def test_should_not_widen_too_soon():
    start = _now() - timedelta(minutes=30)
    gov = _make_governor(start=start)
    # < 1h elapsed → too soon to judge
    assert gov.should_widen_symbol_set(_now()) is False


def test_should_not_widen_when_on_track():
    start = _now() - timedelta(hours=24)
    gov = _make_governor(start=start)
    for _ in range(7):
        gov.record_round_trip_complete("BTC-USD", "win")
    assert gov.should_widen_symbol_set(_now()) is False


def test_should_not_widen_during_cooldown():
    start = _now() - timedelta(hours=6)
    gov = _make_governor(start=start)
    # Force a recent widen
    import os
    os.environ["CADENCE_GOVERNOR_WIDEN_COOLDOWN_H"] = "2.0"
    gov._state = gov._state.model_copy(update={
        "last_widen_at_iso": (_now() - timedelta(minutes=30)).isoformat(),
    })
    # Should not widen because cooldown hasn't passed
    assert gov.should_widen_symbol_set(_now()) is False


# ---------------------------------------------------------------------------
# get_adaptation_recommendation
# ---------------------------------------------------------------------------

def test_recommendation_hold_when_on_track():
    start = _now() - timedelta(hours=24)
    gov = _make_governor(start=start)
    for _ in range(7):
        gov.record_round_trip_complete("BTC-USD", "win")
    rec = gov.get_adaptation_recommendation(_now())
    assert rec["action"] == "hold"


def test_recommendation_widen_when_below():
    start = _now() - timedelta(hours=5)
    gov = _make_governor(start=start)
    rec = gov.get_adaptation_recommendation(_now())
    assert rec["action"] == "widen_breadth"
    assert "below" in rec["reason"].lower()


def test_recommendation_throttle_when_above():
    start = _now() - timedelta(hours=1)
    gov = _make_governor(start=start)
    for _ in range(15):
        gov.record_round_trip_complete("BTC-USD", "win")
    rec = gov.get_adaptation_recommendation(_now())
    assert rec["action"] == "throttle"
    assert "above" in rec["reason"].lower()


def test_recommendation_always_includes_quality_floor():
    gov = _make_governor()
    rec = gov.get_adaptation_recommendation(_now())
    assert "quality_floor_min_rr" in rec
    assert rec["quality_floor_min_rr"] >= 1.0


def test_widen_recorded_in_state():
    start = _now() - timedelta(hours=5)
    gov = _make_governor(start=start)
    gov.get_adaptation_recommendation(_now())
    assert gov.state.last_adaptation_action == "widen_breadth"
    assert gov.state.last_widen_at_iso is not None


def test_throttle_recorded_in_state():
    start = _now() - timedelta(hours=1)
    gov = _make_governor(start=start)
    for _ in range(15):
        gov.record_round_trip_complete("BTC-USD", "win")
    gov.get_adaptation_recommendation(_now())
    assert gov.state.last_adaptation_action == "throttle"


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

def test_get_summary():
    gov = _make_governor()
    summary = gov.get_summary()
    assert "round_trips_completed" in summary
    assert "cadence_status" in summary
    assert "projected_24h_rate" in summary
    assert "added_symbols" in summary


# ---------------------------------------------------------------------------
# State round-trip serialization
# ---------------------------------------------------------------------------

def test_state_serialization_round_trip():
    start = _now() - timedelta(hours=2)
    gov = _make_governor(start=start)
    gov.record_round_trip_complete("BTC-USD", "win", r_achieved=1.5)
    gov.record_round_trip_complete("ETH-USD", "loss", r_achieved=-0.8)

    state_dict = gov.state.model_dump(mode="json")
    restored_state = CadenceGovernorState.model_validate(state_dict)
    restored_gov = CadenceGovernor(state=restored_state)

    assert restored_gov.state.round_trips_completed == 2
    assert restored_gov.state.session_start_iso == gov.state.session_start_iso


def test_mark_symbol_added():
    gov = _make_governor()
    gov.mark_symbol_added("SOL-USD")
    gov.mark_symbol_added("SOL-USD")  # idempotent
    gov.mark_symbol_added("AVAX-USD")
    assert gov.state.added_symbols == ["SOL-USD", "AVAX-USD"]
