"""Tests for services/structural_target_selector.py (Runbook 56).

Covers:
- CANDIDATE_SOURCE_REGISTRY membership and structure
- evaluate_expectancy_gate: happy paths, rejection reasons, selection modes
- resolve_candidate_prices_from_indicator helper
- StructureSnapshot-based candidate selectors (R58 API)
"""
from __future__ import annotations

import pytest

from datetime import datetime, timezone
from uuid import uuid4

from services.structural_target_selector import (
    CANDIDATE_SOURCE_REGISTRY,
    CANDIDATE_SOURCE_REGISTRY_VERSION,
    STRUCTURAL_TARGET_SELECTOR_VERSION,
    CandidateRejectionReason,
    ExpectancyGateTelemetry,
    StructuralCandidateRejection,
    evaluate_expectancy_gate,
    resolve_candidate_prices_from_indicator,
    select_entry_candidates,
    select_stop_candidates,
    select_target_candidates,
)
from schemas.structure_engine import StructureLevel, StructureSnapshot


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_has_required_runbook_41_fields():
    required = {"htf_daily_high", "htf_daily_low", "htf_prev_daily_high", "htf_prev_daily_low",
                "htf_5d_high", "htf_5d_low"}
    assert required.issubset(CANDIDATE_SOURCE_REGISTRY.keys())


def test_registry_has_required_runbook_40_fields():
    required = {"donchian_upper", "donchian_lower", "measured_move", "range_projection"}
    assert required.issubset(CANDIDATE_SOURCE_REGISTRY.keys())


def test_registry_has_r_multiples():
    assert "r_multiple_2" in CANDIDATE_SOURCE_REGISTRY
    assert "r_multiple_3" in CANDIDATE_SOURCE_REGISTRY


def test_registry_entries_have_required_keys():
    for name, entry in CANDIDATE_SOURCE_REGISTRY.items():
        assert "indicator_field" in entry, f"Missing indicator_field for {name}"
        assert "origin" in entry, f"Missing origin for {name}"
        assert "direction" in entry, f"Missing direction for {name}"


def test_registry_versions_are_set():
    assert STRUCTURAL_TARGET_SELECTOR_VERSION
    assert CANDIDATE_SOURCE_REGISTRY_VERSION


# ---------------------------------------------------------------------------
# evaluate_expectancy_gate — basic telemetry structure
# ---------------------------------------------------------------------------


def test_gate_empty_candidates_returns_not_passed():
    t = evaluate_expectancy_gate(
        declared_candidates=[],
        candidate_prices={},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert t.structural_target_source is None
    assert t.structural_r is None
    assert t.all_candidate_sources_evaluated == []


def test_gate_zero_stop_distance_rejects_all():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 110.0},
        entry_price=100.0,
        stop_price=100.0,  # same as entry
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert len(t.candidate_rejections) == 1
    assert t.candidate_rejections[0].reason == "price_not_available"


def test_gate_telemetry_records_entry_stop_direction():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 110.0},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert t.entry_price == 100.0
    assert t.stop_price == 95.0
    assert t.direction == "long"
    assert t.selector_version == STRUCTURAL_TARGET_SELECTOR_VERSION
    assert t.registry_version == CANDIDATE_SOURCE_REGISTRY_VERSION


# ---------------------------------------------------------------------------
# evaluate_expectancy_gate — rejection reasons
# ---------------------------------------------------------------------------


def test_gate_rejects_source_not_in_registry():
    t = evaluate_expectancy_gate(
        declared_candidates=["unknown_source"],
        candidate_prices={"unknown_source": 110.0},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert t.candidate_rejections[0].reason == "source_not_in_registry"


def test_gate_rejects_price_not_available():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": None},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert t.candidate_rejections[0].reason == "price_not_available"


def test_gate_rejects_price_equals_entry():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 100.0},  # same as entry
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert t.candidate_rejections[0].reason == "price_equals_entry"


def test_gate_rejects_wrong_direction_long():
    """Long trade: target below entry is wrong direction."""
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 90.0},  # below entry 100
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert not t.expectancy_gate_passed
    assert t.candidate_rejections[0].reason == "wrong_direction_long"


def test_gate_rejects_wrong_direction_short():
    """Short trade: target above entry is wrong direction."""
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 110.0},  # above entry 100
        entry_price=100.0,
        stop_price=105.0,
        direction="short",
    )
    assert not t.expectancy_gate_passed
    assert t.candidate_rejections[0].reason == "wrong_direction_short"


def test_gate_rejects_insufficient_r_multiple():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 101.0},  # R = 1/5 = 0.2
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        minimum_structural_r_multiple=1.5,
    )
    assert not t.expectancy_gate_passed
    rejection = t.candidate_rejections[0]
    assert rejection.reason == "insufficient_r_multiple"
    assert rejection.computed_r == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# evaluate_expectancy_gate — happy paths
# ---------------------------------------------------------------------------


def test_gate_passes_long_simple():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 115.0},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert t.expectancy_gate_passed
    assert t.structural_target_source == "htf_daily_high"
    assert t.selected_target_price == 115.0
    # R = (115 - 100) / (100 - 95) = 15/5 = 3.0
    assert t.structural_r == pytest.approx(3.0)
    assert t.candidate_rejections == []


def test_gate_passes_short_simple():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_low"],
        candidate_prices={"htf_daily_low": 85.0},
        entry_price=100.0,
        stop_price=105.0,
        direction="short",
    )
    assert t.expectancy_gate_passed
    assert t.structural_target_source == "htf_daily_low"
    # R = (100 - 85) / (105 - 100) = 15/5 = 3.0
    assert t.structural_r == pytest.approx(3.0)


def test_gate_passes_with_minimum_r_satisfied():
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high"],
        candidate_prices={"htf_daily_high": 120.0},  # R = 20/5 = 4.0
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        minimum_structural_r_multiple=2.0,
    )
    assert t.expectancy_gate_passed
    assert t.structural_r == pytest.approx(4.0)
    assert t.minimum_r_required == 2.0


# ---------------------------------------------------------------------------
# Selection modes
# ---------------------------------------------------------------------------


def test_priority_mode_selects_first_valid():
    """In priority mode, first valid candidate in declared order wins."""
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high", "htf_5d_high"],
        candidate_prices={
            "htf_daily_high": 110.0,  # R = 2.0
            "htf_5d_high": 120.0,    # R = 4.0 (higher but declared second)
        },
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        target_selection_mode="priority",
    )
    assert t.structural_target_source == "htf_daily_high"
    assert t.target_selection_mode == "priority"


def test_ranked_mode_selects_highest_r():
    """In ranked mode, candidate with highest R wins."""
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high", "htf_5d_high"],
        candidate_prices={
            "htf_daily_high": 110.0,  # R = 2.0
            "htf_5d_high": 120.0,    # R = 4.0
        },
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        target_selection_mode="ranked",
    )
    assert t.structural_target_source == "htf_5d_high"
    assert t.structural_r == pytest.approx(4.0)


def test_scored_mode_falls_back_to_ranked():
    """scored mode is not yet implemented, falls back to ranked behavior."""
    t = evaluate_expectancy_gate(
        declared_candidates=["htf_daily_high", "htf_5d_high"],
        candidate_prices={
            "htf_daily_high": 110.0,
            "htf_5d_high": 120.0,
        },
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        target_selection_mode="scored",
    )
    assert t.structural_target_source == "htf_5d_high"


def test_mixed_valid_and_invalid_candidates():
    """Some candidates rejected, gate passes on first valid."""
    t = evaluate_expectancy_gate(
        declared_candidates=["unknown_source", "htf_daily_high"],
        candidate_prices={"htf_daily_high": 115.0},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
        target_selection_mode="priority",
    )
    assert t.expectancy_gate_passed
    assert t.structural_target_source == "htf_daily_high"
    assert len(t.candidate_rejections) == 1
    assert t.candidate_rejections[0].reason == "source_not_in_registry"


def test_all_candidates_evaluated_field():
    declared = ["htf_daily_high", "htf_5d_high"]
    t = evaluate_expectancy_gate(
        declared_candidates=declared,
        candidate_prices={"htf_daily_high": 115.0, "htf_5d_high": 120.0},
        entry_price=100.0,
        stop_price=95.0,
        direction="long",
    )
    assert t.all_candidate_sources_evaluated == declared


# ---------------------------------------------------------------------------
# resolve_candidate_prices_from_indicator
# ---------------------------------------------------------------------------


def test_resolve_maps_source_to_indicator_field():
    indicator_data = {"htf_daily_high": 108.5, "htf_daily_low": 96.0}
    result = resolve_candidate_prices_from_indicator(
        declared_candidates=["htf_daily_high", "htf_daily_low"],
        indicator_data=indicator_data,
    )
    assert result["htf_daily_high"] == 108.5
    assert result["htf_daily_low"] == 96.0


def test_resolve_returns_none_for_missing_field():
    result = resolve_candidate_prices_from_indicator(
        declared_candidates=["htf_daily_high"],
        indicator_data={},  # field absent
    )
    assert result["htf_daily_high"] is None


def test_resolve_returns_none_for_unknown_source():
    result = resolve_candidate_prices_from_indicator(
        declared_candidates=["bogus_source"],
        indicator_data={"bogus_source": 100.0},
    )
    assert result["bogus_source"] is None


def test_resolve_uses_indicator_field_name_not_source_name():
    """donchian_upper source maps to donchian_upper_short indicator field."""
    indicator_data = {"donchian_upper_short": 105.0}
    result = resolve_candidate_prices_from_indicator(
        declared_candidates=["donchian_upper"],
        indicator_data=indicator_data,
    )
    assert result["donchian_upper"] == 105.0


# ---------------------------------------------------------------------------
# StructureSnapshot-based selectors (R58 API)
# ---------------------------------------------------------------------------


_NOW = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def _make_level(
    price: float,
    role: str,
    dist: float,
    stop: bool = True,
    target: bool = True,
    entry: bool = False,
) -> StructureLevel:
    snap_id = str(uuid4())
    kind = "swing_high" if role == "resistance" else "swing_low"
    return StructureLevel(
        level_id=f"BTC-USD|{kind}|1h|{price:.4f}",
        snapshot_id=snap_id,
        symbol="BTC-USD",
        as_of_ts=_NOW,
        price=price,
        source_timeframe="1h",
        kind=kind,
        source_label=f"{kind} @ {price}",
        role_now=role,
        distance_abs=dist,
        distance_pct=dist / 100.0,
        eligible_for_stop_anchor=stop,
        eligible_for_target_anchor=target,
        eligible_for_entry_trigger=entry,
    )


def _make_snapshot(*levels: StructureLevel) -> StructureSnapshot:
    return StructureSnapshot(
        snapshot_id=str(uuid4()),
        snapshot_hash="deadbeef" * 8,
        symbol="BTC-USD",
        as_of_ts=_NOW,
        generated_at_ts=_NOW,
        source_timeframe="1h",
        reference_price=100.0,
        levels=list(levels),
    )


def test_select_stop_candidates_long_returns_support():
    snap = _make_snapshot(
        _make_level(95.0, "support", 5.0),
        _make_level(110.0, "resistance", 10.0),
    )
    stops = select_stop_candidates(snap, direction="long")
    assert all(lvl.role_now == "support" for lvl in stops)
    assert len(stops) == 1


def test_select_stop_candidates_short_returns_resistance():
    snap = _make_snapshot(
        _make_level(95.0, "support", 5.0),
        _make_level(110.0, "resistance", 10.0),
    )
    stops = select_stop_candidates(snap, direction="short")
    assert all(lvl.role_now == "resistance" for lvl in stops)
    assert len(stops) == 1


def test_select_target_candidates_long_returns_resistance():
    snap = _make_snapshot(
        _make_level(95.0, "support", 5.0),
        _make_level(110.0, "resistance", 10.0),
    )
    targets = select_target_candidates(snap, direction="long")
    assert all(lvl.role_now == "resistance" for lvl in targets)


def test_select_stop_candidates_sorted_by_distance():
    snap = _make_snapshot(
        _make_level(90.0, "support", 10.0),
        _make_level(96.0, "support", 4.0),
    )
    stops = select_stop_candidates(snap)
    assert stops[0].distance_abs < stops[1].distance_abs
