"""Tests for regime_fingerprint_distance() in services/regime_transition_detector.py.

Covers:
  - Same fingerprint → distance = 0.0
  - Single categorical change → correct component contribution
  - Numeric vector change → correct contribution
  - All categoricals changed → higher distance than single change
  - distance_value is always bounded [0, 1]
  - component_contributions approximately sum to distance_value
  - Distance is deterministic (same inputs → same output)
  - component_deltas describe changes correctly
  - weights are emitted with correct values
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.regime_fingerprint import (
    NUMERIC_VECTOR_FEATURE_NAMES_V1,
    RegimeFingerprint,
)
from services.regime_transition_detector import regime_fingerprint_distance

_TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _fp(
    trend="up",
    vol="mid",
    structure="neutral",
    regime_confidence=0.7,
    numeric_vector=None,
    symbol="BTC-USD",
) -> RegimeFingerprint:
    vec = numeric_vector if numeric_vector is not None else [0.4, 0.4, 0.5, 0.5, 0.5, 0.2]
    return RegimeFingerprint(
        symbol=symbol,
        scope="symbol",
        as_of_ts=_TS,
        bar_id=f"BTC-USD|1h|{_TS.isoformat()}",
        source_timeframe="1h",
        trend_state=trend,
        vol_state=vol,
        structure_state=structure,
        trend_confidence=0.8,
        vol_confidence=0.5,
        structure_confidence=0.5,
        regime_confidence=regime_confidence,
        vol_percentile=0.4,
        atr_percentile=0.4,
        volume_percentile=0.5,
        range_expansion_percentile=0.5,
        realized_vol_z=0.0,
        distance_to_htf_anchor_atr=1.0,
        numeric_vector=vec,
        numeric_vector_feature_names=NUMERIC_VECTOR_FEATURE_NAMES_V1,
    )


# ---------------------------------------------------------------------------
# Identical fingerprints
# ---------------------------------------------------------------------------

def test_distance_identical_fingerprints_is_zero():
    fp = _fp()
    result = regime_fingerprint_distance(fp, fp)
    assert result.distance_value == 0.0


def test_distance_identical_trend_vol_structure_zero():
    fp1 = _fp(trend="down", vol="high", structure="compression")
    fp2 = _fp(trend="down", vol="high", structure="compression")
    result = regime_fingerprint_distance(fp1, fp2)
    assert result.distance_value == 0.0


# ---------------------------------------------------------------------------
# Single categorical change
# ---------------------------------------------------------------------------

def test_distance_trend_change_only():
    prev = _fp(trend="up")
    curr = _fp(trend="down")
    result = regime_fingerprint_distance(curr, prev)
    assert result.distance_value > 0.0
    assert result.component_contributions["trend_state"] == pytest.approx(0.25, abs=1e-4)
    assert result.component_contributions["vol_state"] == pytest.approx(0.0, abs=1e-4)
    assert result.component_contributions["structure_state"] == pytest.approx(0.0, abs=1e-4)


def test_distance_vol_change_only():
    prev = _fp(vol="low")
    curr = _fp(vol="extreme")
    result = regime_fingerprint_distance(curr, prev)
    assert result.component_contributions["vol_state"] == pytest.approx(0.20, abs=1e-4)
    assert result.component_contributions["trend_state"] == pytest.approx(0.0, abs=1e-4)


def test_distance_structure_change_only():
    prev = _fp(structure="neutral")
    curr = _fp(structure="compression")
    result = regime_fingerprint_distance(curr, prev)
    assert result.component_contributions["structure_state"] == pytest.approx(0.25, abs=1e-4)
    assert result.component_contributions["trend_state"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Multiple categorical changes → larger distance
# ---------------------------------------------------------------------------

def test_distance_all_categoricals_changed_larger():
    prev = _fp(trend="up", vol="low", structure="neutral")
    curr = _fp(trend="down", vol="extreme", structure="compression")
    result_all = regime_fingerprint_distance(curr, prev)

    only_trend = _fp(trend="down", vol="low", structure="neutral")
    result_one = regime_fingerprint_distance(only_trend, prev)

    assert result_all.distance_value > result_one.distance_value


def test_distance_max_categorical_contribution():
    """If all 3 categoricals fire: 0.25 + 0.20 + 0.25 = 0.70."""
    prev = _fp(trend="up", vol="low", structure="neutral")
    curr = _fp(trend="down", vol="extreme", structure="compression")
    result = regime_fingerprint_distance(curr, prev)
    cat_sum = (
        result.component_contributions["trend_state"]
        + result.component_contributions["vol_state"]
        + result.component_contributions["structure_state"]
    )
    assert cat_sum == pytest.approx(0.70, abs=1e-4)


# ---------------------------------------------------------------------------
# Numeric vector contribution
# ---------------------------------------------------------------------------

def test_distance_numeric_vector_change():
    vec_a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vec_b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    prev = _fp(numeric_vector=vec_a)
    curr = _fp(numeric_vector=vec_b)
    result = regime_fingerprint_distance(curr, prev)
    # L1/n = 6/6 = 1.0, weighted by 0.20 → 0.20
    assert result.component_contributions["numeric_vector"] == pytest.approx(0.20, abs=1e-4)


def test_distance_numeric_vector_partial_change():
    vec_a = [0.0] * 6
    vec_b = [0.5] * 6
    prev = _fp(numeric_vector=vec_a)
    curr = _fp(numeric_vector=vec_b)
    result = regime_fingerprint_distance(curr, prev)
    # L1/n = 0.5, weighted by 0.20 → 0.10
    assert result.component_contributions["numeric_vector"] == pytest.approx(0.10, abs=1e-4)


# ---------------------------------------------------------------------------
# Confidence delta contribution
# ---------------------------------------------------------------------------

def test_distance_confidence_delta():
    prev = _fp(regime_confidence=0.4)
    curr = _fp(regime_confidence=0.9)
    result = regime_fingerprint_distance(curr, prev)
    # delta = 0.5, weight = 0.10 → contribution = 0.10 * min(1.0, 0.5) = 0.05
    assert result.component_contributions["confidence"] == pytest.approx(0.05, abs=1e-4)
    assert result.confidence_delta == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Bounds [0, 1]
# ---------------------------------------------------------------------------

def test_distance_bounded_max_case():
    """Maximum scenario: all categoricals + full numeric change + full confidence delta."""
    prev = _fp(trend="up", vol="low", structure="neutral",
               regime_confidence=0.0, numeric_vector=[0.0] * 6)
    curr = _fp(trend="down", vol="extreme", structure="compression",
               regime_confidence=1.0, numeric_vector=[1.0] * 6)
    result = regime_fingerprint_distance(curr, prev)
    assert 0.0 <= result.distance_value <= 1.0


def test_distance_always_non_negative():
    fp1 = _fp()
    fp2 = _fp(trend="down")
    result = regime_fingerprint_distance(fp1, fp2)
    assert result.distance_value >= 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_distance_deterministic():
    prev = _fp(trend="up", vol="mid", structure="neutral", regime_confidence=0.7)
    curr = _fp(trend="down", vol="high", structure="compression", regime_confidence=0.5)
    r1 = regime_fingerprint_distance(curr, prev)
    r2 = regime_fingerprint_distance(curr, prev)
    assert r1.distance_value == r2.distance_value
    assert r1.component_contributions == r2.component_contributions


# ---------------------------------------------------------------------------
# Threshold metadata
# ---------------------------------------------------------------------------

def test_distance_threshold_metadata_enter():
    fp = _fp()
    result = regime_fingerprint_distance(fp, fp, enter_threshold=0.30, exit_threshold=0.15,
                                         threshold_type="enter")
    assert result.threshold_enter == 0.30
    assert result.threshold_exit == 0.15
    assert result.threshold_used == 0.30
    assert result.threshold_type == "enter"


def test_distance_threshold_metadata_exit():
    fp = _fp()
    result = regime_fingerprint_distance(fp, fp, enter_threshold=0.30, exit_threshold=0.15,
                                         threshold_type="exit")
    assert result.threshold_used == 0.15
    assert result.threshold_type == "exit"


# ---------------------------------------------------------------------------
# component_deltas inspectability
# ---------------------------------------------------------------------------

def test_distance_deltas_trend_changed():
    prev = _fp(trend="up")
    curr = _fp(trend="down")
    result = regime_fingerprint_distance(curr, prev)
    assert "up -> down" in result.component_deltas["trend_state"]


def test_distance_deltas_unchanged_label():
    fp = _fp()
    result = regime_fingerprint_distance(fp, fp)
    assert result.component_deltas["trend_state"] == "unchanged"
    assert result.component_deltas["vol_state"] == "unchanged"


# ---------------------------------------------------------------------------
# Weights emitted
# ---------------------------------------------------------------------------

def test_distance_weights_emitted():
    fp = _fp()
    result = regime_fingerprint_distance(fp, fp)
    assert "trend_state" in result.weights
    assert "vol_state" in result.weights
    assert "structure_state" in result.weights
    assert "numeric_vector" in result.weights
    assert "confidence" in result.weights
    # Weights must sum to 1.0
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
