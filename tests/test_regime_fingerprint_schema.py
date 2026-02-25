"""Tests for schemas/regime_fingerprint.py — Runbook 55.

Covers:
  - All schema classes instantiate with valid data.
  - extra='forbid' is enforced on all classes.
  - Field bounds (ge/le) are enforced on confidence and vol_percentile etc.
  - Required fields must be present.
  - NUMERIC_VECTOR_FEATURE_NAMES_V1 is version-stable and has correct length.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.regime_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    FINGERPRINT_VERSION,
    NUMERIC_VECTOR_FEATURE_NAMES_V1,
    RegimeDistanceResult,
    RegimeFingerprint,
    RegimeTransitionDecision,
    RegimeTransitionDetectorState,
    RegimeTransitionTelemetryEvent,
)

_TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_TS2 = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)


def _fingerprint(**overrides) -> RegimeFingerprint:
    defaults = dict(
        symbol="BTC-USD",
        scope="symbol",
        as_of_ts=_TS,
        bar_id="BTC-USD|1h|2026-01-01T12:00:00+00:00",
        source_timeframe="1h",
        trend_state="up",
        vol_state="mid",
        structure_state="neutral",
        trend_confidence=0.8,
        vol_confidence=0.5,
        structure_confidence=0.5,
        regime_confidence=0.7,
        vol_percentile=0.4,
        atr_percentile=0.4,
        volume_percentile=0.5,
        range_expansion_percentile=0.5,
        realized_vol_z=0.0,
        distance_to_htf_anchor_atr=1.0,
        numeric_vector=[0.4, 0.4, 0.5, 0.5, 0.5, 0.2],
        numeric_vector_feature_names=NUMERIC_VECTOR_FEATURE_NAMES_V1,
    )
    defaults.update(overrides)
    return RegimeFingerprint(**defaults)


# ---------------------------------------------------------------------------
# NUMERIC_VECTOR_FEATURE_NAMES_V1
# ---------------------------------------------------------------------------

def test_numeric_vector_names_length():
    """R55 mandates 6 normalized features in the vector."""
    assert len(NUMERIC_VECTOR_FEATURE_NAMES_V1) == 6


def test_numeric_vector_names_content():
    assert "vol_percentile" in NUMERIC_VECTOR_FEATURE_NAMES_V1
    assert "atr_percentile" in NUMERIC_VECTOR_FEATURE_NAMES_V1
    assert "volume_percentile" in NUMERIC_VECTOR_FEATURE_NAMES_V1
    assert "range_expansion_percentile" in NUMERIC_VECTOR_FEATURE_NAMES_V1
    assert "realized_vol_z_normed" in NUMERIC_VECTOR_FEATURE_NAMES_V1
    assert "distance_to_htf_anchor_normed" in NUMERIC_VECTOR_FEATURE_NAMES_V1


# ---------------------------------------------------------------------------
# RegimeFingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_basic_instantiation():
    fp = _fingerprint()
    assert fp.symbol == "BTC-USD"
    assert fp.fingerprint_version == FINGERPRINT_VERSION
    assert fp.schema_version == FINGERPRINT_SCHEMA_VERSION


def test_fingerprint_extra_forbid():
    with pytest.raises(ValidationError):
        RegimeFingerprint(
            **{
                "symbol": "BTC-USD",
                "scope": "symbol",
                "as_of_ts": _TS,
                "bar_id": "x",
                "source_timeframe": "1h",
                "trend_state": "up",
                "vol_state": "mid",
                "structure_state": "neutral",
                "trend_confidence": 0.8,
                "vol_confidence": 0.5,
                "structure_confidence": 0.5,
                "regime_confidence": 0.7,
                "vol_percentile": 0.4,
                "atr_percentile": 0.4,
                "volume_percentile": 0.5,
                "range_expansion_percentile": 0.5,
                "realized_vol_z": 0.0,
                "distance_to_htf_anchor_atr": 1.0,
                "numeric_vector": [0.4, 0.4, 0.5, 0.5, 0.5, 0.2],
                "numeric_vector_feature_names": NUMERIC_VECTOR_FEATURE_NAMES_V1,
                "unknown_field": "forbidden",
            }
        )


def test_fingerprint_trend_states():
    for ts in ("up", "down", "sideways"):
        fp = _fingerprint(trend_state=ts)
        assert fp.trend_state == ts


def test_fingerprint_invalid_trend_state():
    with pytest.raises(ValidationError):
        _fingerprint(trend_state="uptrend")  # legacy label — not valid in R55 schema


def test_fingerprint_vol_states():
    for vs in ("low", "mid", "high", "extreme"):
        fp = _fingerprint(vol_state=vs)
        assert fp.vol_state == vs


def test_fingerprint_invalid_vol_state():
    with pytest.raises(ValidationError):
        _fingerprint(vol_state="normal")  # legacy label — not valid in R55 schema


def test_fingerprint_structure_states():
    for ss in ("compression", "expansion", "mean_reverting",
               "breakout_active", "breakdown_active", "neutral"):
        fp = _fingerprint(structure_state=ss)
        assert fp.structure_state == ss


def test_fingerprint_confidence_bounds_valid():
    fp = _fingerprint(
        trend_confidence=0.0,
        vol_confidence=1.0,
        structure_confidence=0.5,
        regime_confidence=1.0,
    )
    assert fp.trend_confidence == 0.0


def test_fingerprint_confidence_below_zero():
    with pytest.raises(ValidationError):
        _fingerprint(trend_confidence=-0.01)


def test_fingerprint_confidence_above_one():
    with pytest.raises(ValidationError):
        _fingerprint(vol_confidence=1.01)


def test_fingerprint_percentile_bounds_valid():
    fp = _fingerprint(vol_percentile=0.0, atr_percentile=1.0)
    assert fp.vol_percentile == 0.0


def test_fingerprint_percentile_below_zero():
    with pytest.raises(ValidationError):
        _fingerprint(vol_percentile=-0.001)


def test_fingerprint_percentile_above_one():
    with pytest.raises(ValidationError):
        _fingerprint(range_expansion_percentile=1.001)


def test_fingerprint_scope_cohort():
    fp = _fingerprint(scope="cohort", cohort_id="crypto-majors")
    assert fp.scope == "cohort"
    assert fp.cohort_id == "crypto-majors"


def test_fingerprint_trend_strength_z_optional():
    fp = _fingerprint()
    assert fp.trend_strength_z is None
    fp2 = _fingerprint(trend_strength_z=1.5)
    assert fp2.trend_strength_z == 1.5


# ---------------------------------------------------------------------------
# RegimeDistanceResult
# ---------------------------------------------------------------------------

def _distance_result(**overrides) -> RegimeDistanceResult:
    defaults = dict(
        distance_value=0.25,
        threshold_enter=0.30,
        threshold_exit=0.15,
        threshold_used=0.30,
        threshold_type="enter",
        component_contributions={"trend_state": 0.25},
        component_deltas={"trend_state": "up -> down"},
        weights={"trend_state": 0.25},
        confidence_delta=-0.1,
    )
    defaults.update(overrides)
    return RegimeDistanceResult(**defaults)


def test_distance_result_basic():
    dr = _distance_result()
    assert dr.distance_value == 0.25
    assert dr.threshold_type == "enter"


def test_distance_result_extra_forbid():
    with pytest.raises(ValidationError):
        _distance_result(unexpected="field")


def test_distance_result_distance_below_zero():
    with pytest.raises(ValidationError):
        _distance_result(distance_value=-0.01)


def test_distance_result_distance_above_one():
    with pytest.raises(ValidationError):
        _distance_result(distance_value=1.001)


def test_distance_result_suppression_flags_default_false():
    dr = _distance_result()
    assert dr.suppressed_by_hysteresis is False
    assert dr.suppressed_by_cooldown is False
    assert dr.suppressed_by_min_dwell is False


# ---------------------------------------------------------------------------
# RegimeTransitionDecision
# ---------------------------------------------------------------------------

def test_decision_initial_state():
    fp = _fingerprint()
    d = RegimeTransitionDecision(
        transition_fired=True,
        reason_code="initial_state",
        prior_fingerprint=None,
        new_fingerprint=fp,
        suppressed=False,
        shock_override_used=False,
        htf_gate_eligible=True,
        as_of_ts=_TS,
        symbol="BTC-USD",
        scope="symbol",
    )
    assert d.transition_fired is True
    assert d.reason_code == "initial_state"
    assert d.prior_fingerprint is None


def test_decision_invalid_reason_code():
    fp = _fingerprint()
    with pytest.raises(ValidationError):
        RegimeTransitionDecision(
            transition_fired=False,
            reason_code="made_up_code",
            new_fingerprint=fp,
            as_of_ts=_TS,
            symbol="BTC-USD",
            scope="symbol",
        )


def test_decision_extra_forbid():
    fp = _fingerprint()
    with pytest.raises(ValidationError):
        RegimeTransitionDecision(
            transition_fired=True,
            reason_code="initial_state",
            new_fingerprint=fp,
            as_of_ts=_TS,
            symbol="BTC-USD",
            scope="symbol",
            unknown="forbidden",
        )


# ---------------------------------------------------------------------------
# RegimeTransitionTelemetryEvent
# ---------------------------------------------------------------------------

def test_telemetry_event_basic():
    fp = _fingerprint()
    decision = RegimeTransitionDecision(
        transition_fired=True,
        reason_code="initial_state",
        new_fingerprint=fp,
        as_of_ts=_TS,
        symbol="BTC-USD",
        scope="symbol",
    )
    event = RegimeTransitionTelemetryEvent(
        event_id="test-id-1",
        emitted_at=_TS,
        decision=decision,
        dwell_seconds=0.0,
        cooldown_remaining_seconds=300.0,
        consecutive_stable_evals=0,
    )
    assert event.event_id == "test-id-1"
    assert event.consecutive_stable_evals == 0


def test_telemetry_event_extra_forbid():
    fp = _fingerprint()
    decision = RegimeTransitionDecision(
        transition_fired=False,
        reason_code="distance_below_threshold",
        new_fingerprint=fp,
        prior_fingerprint=fp,
        as_of_ts=_TS,
        symbol="BTC-USD",
        scope="symbol",
    )
    with pytest.raises(ValidationError):
        RegimeTransitionTelemetryEvent(
            event_id="x",
            emitted_at=_TS,
            decision=decision,
            unexpected_field="value",
        )


# ---------------------------------------------------------------------------
# RegimeTransitionDetectorState
# ---------------------------------------------------------------------------

def test_detector_state_initial():
    state = RegimeTransitionDetectorState(symbol="ETH-USD", scope="symbol")
    assert state.current_fingerprint is None
    assert state.total_transitions == 0
    assert state.consecutive_stable_evals == 0


def test_detector_state_extra_forbid():
    with pytest.raises(ValidationError):
        RegimeTransitionDetectorState(
            symbol="ETH-USD",
            scope="symbol",
            bogus_field=True,
        )


def test_detector_state_with_fingerprint():
    fp = _fingerprint()
    state = RegimeTransitionDetectorState(
        symbol="BTC-USD",
        scope="symbol",
        current_fingerprint=fp,
        last_transition_ts=_TS,
        current_regime_entered_ts=_TS,
        cooldown_until_ts=_TS2,
        consecutive_stable_evals=3,
        total_transitions=2,
    )
    assert state.total_transitions == 2
    assert state.current_fingerprint.trend_state == "up"
