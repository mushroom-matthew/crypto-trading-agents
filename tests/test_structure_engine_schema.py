"""Tests for schemas/structure_engine.py — Runbook 58 schema contracts.

Validates:
- StructureLevel: field types, extra-forbid enforcement
- StructureEvent: event types, direction values, evidence dict
- LevelLadder: bucket separation, all_supports/all_resistances properties
- StructureSnapshot: provenance fields, version constant, hash helper
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.structure_engine import (
    STRUCTURE_ENGINE_VERSION,
    LevelLadder,
    StructureEvent,
    StructureLevel,
    StructureQuality,
    StructureSnapshot,
    compute_structure_snapshot_hash,
)

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)
SID = str(uuid4())


# ---------------------------------------------------------------------------
# StructureLevel
# ---------------------------------------------------------------------------

class TestStructureLevel:
    def _level(self, **kwargs) -> StructureLevel:
        defaults = dict(
            level_id="BTC-USD|prior_session_high|1d|50000.0000",
            snapshot_id=SID,
            symbol="BTC-USD",
            as_of_ts=NOW,
            price=50000.0,
            source_timeframe="1d",
            kind="prior_session_high",
            source_label="D-1 High",
            role_now="resistance",
            distance_abs=1000.0,
            distance_pct=2.0,
        )
        defaults.update(kwargs)
        return StructureLevel(**defaults)

    def test_basic_construction(self):
        level = self._level()
        assert level.price == 50000.0
        assert level.kind == "prior_session_high"
        assert level.role_now == "resistance"
        assert level.distance_abs == 1000.0
        assert level.distance_pct == 2.0

    def test_optional_fields_default_none(self):
        level = self._level()
        assert level.distance_atr is None
        assert level.strength_score is None
        assert level.touch_count is None
        assert level.last_touch_ts is None
        assert level.age_bars is None

    def test_eligibility_flags_default_false(self):
        level = self._level()
        assert level.eligible_for_entry_trigger is False
        assert level.eligible_for_stop_anchor is False
        assert level.eligible_for_target_anchor is False

    def test_eligibility_flags_can_be_set(self):
        level = self._level(
            eligible_for_entry_trigger=True,
            eligible_for_stop_anchor=True,
            eligible_for_target_anchor=True,
        )
        assert level.eligible_for_entry_trigger is True
        assert level.eligible_for_stop_anchor is True
        assert level.eligible_for_target_anchor is True

    def test_valid_role_values(self):
        for role in ("support", "resistance", "neutral"):
            level = self._level(role_now=role)
            assert level.role_now == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            self._level(role_now="unknown")

    def test_invalid_kind_raises(self):
        with pytest.raises(ValidationError):
            self._level(kind="made_up_kind")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            self._level(made_up_field="value")

    def test_source_metadata_defaults_empty(self):
        level = self._level()
        assert level.source_metadata == {}

    def test_distance_atr_optional_float(self):
        level = self._level(distance_atr=2.5)
        assert level.distance_atr == 2.5

    def test_strength_score_range(self):
        level = self._level(strength_score=0.75)
        assert level.strength_score == 0.75

    def test_swing_high_kind(self):
        level = self._level(kind="swing_high", source_label="Swing High")
        assert level.kind == "swing_high"

    def test_swing_low_kind(self):
        level = self._level(kind="swing_low", source_label="Swing Low")
        assert level.kind == "swing_low"

    def test_rolling_window_high_kind(self):
        level = self._level(kind="rolling_window_high", source_label="5D High", source_timeframe="5d")
        assert level.kind == "rolling_window_high"


# ---------------------------------------------------------------------------
# StructureEvent
# ---------------------------------------------------------------------------

class TestStructureEvent:
    def _event(self, **kwargs) -> StructureEvent:
        defaults = dict(
            event_id=str(uuid4()),
            snapshot_id=SID,
            symbol="BTC-USD",
            as_of_ts=NOW,
            eval_timeframe="1h",
            event_type="level_broken",
            severity="medium",
        )
        defaults.update(kwargs)
        return StructureEvent(**defaults)

    def test_basic_construction(self):
        ev = self._event()
        assert ev.event_type == "level_broken"
        assert ev.severity == "medium"
        assert ev.direction == "neutral"  # default

    def test_all_event_types_valid(self):
        types = [
            "level_broken", "level_reclaimed", "liquidity_sweep_reject",
            "range_breakout", "range_breakdown", "trendline_break", "structure_shift",
        ]
        for et in types:
            ev = self._event(event_type=et)
            assert ev.event_type == et

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValidationError):
            self._event(event_type="made_up")

    def test_all_severities_valid(self):
        for sev in ("low", "medium", "high"):
            ev = self._event(severity=sev)
            assert ev.severity == sev

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            self._event(severity="critical")

    def test_all_directions_valid(self):
        for direction in ("up", "down", "neutral"):
            ev = self._event(direction=direction)
            assert ev.direction == direction

    def test_policy_flags_default_false(self):
        ev = self._event()
        assert ev.trigger_policy_reassessment is False
        assert ev.trigger_activation_review is False

    def test_policy_flags_can_be_set(self):
        ev = self._event(trigger_policy_reassessment=True, trigger_activation_review=True)
        assert ev.trigger_policy_reassessment is True
        assert ev.trigger_activation_review is True

    def test_evidence_dict_default_empty(self):
        ev = self._event()
        assert ev.evidence == {}

    def test_evidence_dict_accepts_arbitrary_values(self):
        ev = self._event(evidence={"prior_role": "support", "level_price": 49000.0})
        assert ev.evidence["prior_role"] == "support"

    def test_optional_fields_default_none(self):
        ev = self._event()
        assert ev.level_id is None
        assert ev.level_kind is None
        assert ev.price_ref is None
        assert ev.close_ref is None
        assert ev.threshold_ref is None
        assert ev.confirmation_rule is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            self._event(fake_field="value")

    def test_range_breakout_with_evidence(self):
        ev = self._event(
            event_type="range_breakout",
            severity="high",
            direction="up",
            price_ref=52000.0,
            threshold_ref=51000.0,
            confirmation_rule="close_above_5d_high",
            evidence={"window": "5d", "5d_high": 51000.0},
            trigger_policy_reassessment=True,
            trigger_activation_review=True,
        )
        assert ev.event_type == "range_breakout"
        assert ev.trigger_activation_review is True
        assert ev.evidence["window"] == "5d"


# ---------------------------------------------------------------------------
# LevelLadder
# ---------------------------------------------------------------------------

def _make_level(price: float, role: str, tf: str = "1d", dist_atr: float = 1.0) -> StructureLevel:
    return StructureLevel(
        level_id=f"BTC-USD|prior_session_high|{tf}|{price:.4f}",
        snapshot_id=SID,
        symbol="BTC-USD",
        as_of_ts=NOW,
        price=price,
        source_timeframe=tf,
        kind="prior_session_high",
        source_label="D-1 High",
        role_now=role,
        distance_abs=abs(price - 50000.0),
        distance_pct=abs(price - 50000.0) / 50000.0 * 100,
        distance_atr=dist_atr,
    )


class TestLevelLadder:
    def test_empty_ladder(self):
        ladder = LevelLadder(source_timeframe="1d")
        assert ladder.all_supports == []
        assert ladder.all_resistances == []

    def test_all_supports_aggregates_buckets(self):
        near = _make_level(49500.0, "support", dist_atr=0.5)
        mid = _make_level(49000.0, "support", dist_atr=2.0)
        far = _make_level(47000.0, "support", dist_atr=5.0)
        ladder = LevelLadder(
            source_timeframe="1d",
            near_supports=[near],
            mid_supports=[mid],
            far_supports=[far],
        )
        all_sup = ladder.all_supports
        assert len(all_sup) == 3
        assert near in all_sup
        assert mid in all_sup
        assert far in all_sup

    def test_all_resistances_aggregates_buckets(self):
        near = _make_level(50500.0, "resistance", dist_atr=0.5)
        mid = _make_level(51000.0, "resistance", dist_atr=2.0)
        ladder = LevelLadder(
            source_timeframe="1d",
            near_resistances=[near],
            mid_resistances=[mid],
        )
        all_res = ladder.all_resistances
        assert len(all_res) == 2

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            LevelLadder(source_timeframe="1d", fake_field="x")


# ---------------------------------------------------------------------------
# StructureSnapshot
# ---------------------------------------------------------------------------

class TestStructureSnapshot:
    def _snap(self, **kwargs) -> StructureSnapshot:
        defaults = dict(
            snapshot_id=str(uuid4()),
            snapshot_hash="abc123",
            symbol="BTC-USD",
            as_of_ts=NOW,
            generated_at_ts=NOW,
            source_timeframe="1h",
            reference_price=50000.0,
        )
        defaults.update(kwargs)
        return StructureSnapshot(**defaults)

    def test_basic_construction(self):
        snap = self._snap()
        assert snap.symbol == "BTC-USD"
        assert snap.reference_price == 50000.0
        assert snap.snapshot_version == STRUCTURE_ENGINE_VERSION
        assert snap.levels == []
        assert snap.events == []
        assert snap.ladders == {}

    def test_version_constant(self):
        assert STRUCTURE_ENGINE_VERSION == "1.0.0"
        snap = self._snap()
        assert snap.snapshot_version == "1.0.0"

    def test_policy_integration_fields_default_empty(self):
        snap = self._snap()
        assert snap.policy_trigger_reasons == []
        assert snap.policy_event_priority is None

    def test_policy_priority_valid_values(self):
        for prio in ("low", "medium", "high"):
            snap = self._snap(policy_event_priority=prio)
            assert snap.policy_event_priority == prio

    def test_invalid_policy_priority_raises(self):
        with pytest.raises(ValidationError):
            self._snap(policy_event_priority="critical")

    def test_quality_defaults(self):
        snap = self._snap()
        assert snap.quality.is_partial is False
        assert snap.quality.available_timeframes == []
        assert snap.quality.missing_timeframes == []

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            self._snap(made_up_field="value")

    def test_levels_and_events_stored(self):
        level = _make_level(51000.0, "resistance")
        event = StructureEvent(
            event_id=str(uuid4()),
            snapshot_id=SID,
            symbol="BTC-USD",
            as_of_ts=NOW,
            eval_timeframe="1h",
            event_type="range_breakout",
            severity="high",
        )
        snap = self._snap(levels=[level], events=[event])
        assert len(snap.levels) == 1
        assert len(snap.events) == 1

    def test_reference_atr_optional(self):
        snap = self._snap(reference_atr=750.0)
        assert snap.reference_atr == 750.0

        snap2 = self._snap()
        assert snap2.reference_atr is None


# ---------------------------------------------------------------------------
# compute_structure_snapshot_hash
# ---------------------------------------------------------------------------

class TestComputeStructureSnapshotHash:
    def test_deterministic(self):
        content = {"symbol": "BTC-USD", "price": 50000.0, "ts": "2026-01-01"}
        h1 = compute_structure_snapshot_hash(content)
        h2 = compute_structure_snapshot_hash(content)
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_structure_snapshot_hash({"price": 50000.0})
        h2 = compute_structure_snapshot_hash({"price": 51000.0})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = compute_structure_snapshot_hash({"a": 1, "b": 2})
        h2 = compute_structure_snapshot_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_is_hex_string(self):
        h = compute_structure_snapshot_hash({"x": 1})
        assert len(h) == 64  # sha256 hex digest
        assert all(c in "0123456789abcdef" for c in h)
