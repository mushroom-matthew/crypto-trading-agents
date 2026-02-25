"""Tests for RegimeTransitionDetector state machine in services/regime_transition_detector.py.

Covers:
  - Initial state fires with reason_code='initial_state'
  - Small distance → no transition (distance_below_threshold)
  - Large distance → transition fires (distance_enter_threshold_crossed)
  - HTF gate not ready → htf_gate_not_ready reason, no transition
  - Cooldown suppression after transition
  - Min dwell suppression before min time in regime
  - Oscillation prevention: up→down, then down→up suppressed by min_dwell
  - Shock override: extreme vol fires even with HTF gate closed
  - Shock override fires once, then suppressed by cooldown (repetition guard)
  - Under-trigger guard: real regime change not suppressed when threshold crossed and dwell met
  - Deterministic replay: same inputs produce identical decision sequence
  - Telemetry emitted on every evaluation (fired or not)
  - cohort scope supported
  - load_state() restores detector from serialized state
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from schemas.regime_fingerprint import (
    NUMERIC_VECTOR_FEATURE_NAMES_V1,
    RegimeFingerprint,
    RegimeTransitionDetectorState,
)
from services.regime_transition_detector import RegimeTransitionDetector

_TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _fp(
    trend="up",
    vol="mid",
    structure="neutral",
    regime_confidence=0.7,
    vol_percentile=0.4,
    realized_vol_z=0.0,
    symbol="BTC-USD",
) -> RegimeFingerprint:
    vec = [vol_percentile, vol_percentile, 0.5, 0.5, (realized_vol_z + 3.0) / 6.0, 0.2]
    vec = [min(1.0, max(0.0, v)) for v in vec]
    return RegimeFingerprint(
        symbol=symbol,
        scope="symbol",
        as_of_ts=_TS,
        bar_id=f"{symbol}|1h|{_TS.isoformat()}",
        source_timeframe="1h",
        trend_state=trend,
        vol_state=vol,
        structure_state=structure,
        trend_confidence=0.8,
        vol_confidence=0.5,
        structure_confidence=0.5,
        regime_confidence=regime_confidence,
        vol_percentile=vol_percentile,
        atr_percentile=vol_percentile,
        volume_percentile=0.5,
        range_expansion_percentile=0.5,
        realized_vol_z=realized_vol_z,
        distance_to_htf_anchor_atr=1.0,
        numeric_vector=vec,
        numeric_vector_feature_names=NUMERIC_VECTOR_FEATURE_NAMES_V1,
    )


def _detector(
    min_dwell_seconds: float = 900.0,
    cooldown_seconds: float = 300.0,
    enter_threshold: float = 0.30,
    exit_threshold: float = 0.15,
) -> RegimeTransitionDetector:
    return RegimeTransitionDetector(
        "BTC-USD",
        min_dwell_seconds=min_dwell_seconds,
        cooldown_seconds=cooldown_seconds,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_fires_transition():
    d = _detector()
    fp = _fp()
    event = d.evaluate(fp, _TS)
    assert event.decision.transition_fired is True
    assert event.decision.reason_code == "initial_state"
    assert event.decision.prior_fingerprint is None
    assert d.state.total_transitions == 1


def test_initial_state_emits_telemetry():
    d = _detector()
    fp = _fp()
    event = d.evaluate(fp, _TS)
    assert event.event_id is not None
    assert event.emitted_at == _TS


def test_initial_state_sets_current_fingerprint():
    d = _detector()
    fp = _fp(trend="down")
    d.evaluate(fp, _TS)
    assert d.state.current_fingerprint.trend_state == "down"


# ---------------------------------------------------------------------------
# Distance threshold gating
# ---------------------------------------------------------------------------

def test_no_transition_same_fingerprint():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)  # initial state
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp, ts2)
    assert event.decision.transition_fired is False
    assert event.decision.reason_code == "distance_below_threshold"


def test_transition_fires_on_large_distance():
    """Full trend+vol+structure change should exceed 0.30 threshold."""
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp_init = _fp(trend="up", vol="low", structure="neutral")
    d.evaluate(fp_init, _TS)

    fp_new = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_new, ts2)
    assert event.decision.transition_fired is True
    assert event.decision.reason_code == "distance_enter_threshold_crossed"
    assert d.state.total_transitions == 2


def test_no_transition_small_distance_below_threshold():
    """Only vol_state change → 0.20 contribution, which is below 0.30 threshold."""
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0, enter_threshold=0.30)
    fp_init = _fp(vol="low")
    d.evaluate(fp_init, _TS)

    fp_small = _fp(vol="mid")
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_small, ts2)
    assert event.decision.transition_fired is False
    assert event.decision.reason_code == "distance_below_threshold"


# ---------------------------------------------------------------------------
# HTF gate
# ---------------------------------------------------------------------------

def test_htf_gate_not_eligible_suppresses():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)  # initial

    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_change, ts2, htf_gate_eligible=False)
    assert event.decision.transition_fired is False
    assert event.decision.reason_code == "htf_gate_not_ready"


def test_htf_gate_eligible_allows_transition():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)

    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_change, ts2, htf_gate_eligible=True)
    assert event.decision.transition_fired is True


# ---------------------------------------------------------------------------
# Cooldown suppression
# ---------------------------------------------------------------------------

def test_cooldown_suppresses_second_transition():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=3600.0)
    fp_init = _fp(trend="up", vol="low")
    d.evaluate(fp_init, _TS)

    # First eligible transition
    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    event2 = d.evaluate(fp_change, ts2)
    assert event2.decision.transition_fired is True

    # Second attempt within cooldown window
    fp_change2 = _fp(trend="up", vol="extreme", structure="expansion")
    ts3 = _TS + timedelta(seconds=2)
    event3 = d.evaluate(fp_change2, ts3)
    assert event3.decision.transition_fired is False
    assert event3.decision.reason_code == "suppressed_cooldown"


def test_cooldown_expires_allows_next_transition():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=300.0)
    fp_init = _fp(trend="up", vol="low")
    d.evaluate(fp_init, _TS)

    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    d.evaluate(fp_change, ts2)  # first transition, starts cooldown

    # After cooldown expires
    fp_change2 = _fp(trend="up", vol="low")
    ts4 = _TS + timedelta(seconds=400)  # 400s > 300s cooldown
    event4 = d.evaluate(fp_change2, ts4)
    # trend+vol change = 0.25 + 0.20 = 0.45 > 0.30 threshold
    assert event4.decision.transition_fired is True


# ---------------------------------------------------------------------------
# Min dwell
# ---------------------------------------------------------------------------

def test_min_dwell_suppresses_early_transition():
    d = _detector(min_dwell_seconds=900.0, cooldown_seconds=0.0)
    fp_init = _fp(trend="up", vol="low")
    d.evaluate(fp_init, _TS)

    # Only 60s later — below min_dwell of 900s
    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=60)
    event = d.evaluate(fp_change, ts2)
    assert event.decision.transition_fired is False
    assert event.decision.reason_code == "suppressed_min_dwell"


def test_min_dwell_allows_after_threshold():
    d = _detector(min_dwell_seconds=900.0, cooldown_seconds=0.0)
    fp_init = _fp(trend="up", vol="low")
    d.evaluate(fp_init, _TS)

    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1000)  # 1000s > 900s dwell
    event = d.evaluate(fp_change, ts2)
    assert event.decision.transition_fired is True


# ---------------------------------------------------------------------------
# Oscillation prevention (hysteresis)
# ---------------------------------------------------------------------------

def test_oscillation_suppressed_by_min_dwell():
    """Up → Down fires. Immediately Down → Up suppressed by min_dwell."""
    d = _detector(min_dwell_seconds=900.0, cooldown_seconds=0.0)
    fp_up = _fp(trend="up", vol="low")
    d.evaluate(fp_up, _TS)

    fp_down = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1000)
    event2 = d.evaluate(fp_down, ts2)
    assert event2.decision.transition_fired is True

    # Immediately try to go back to "up" (only 30s after transition)
    fp_up_again = _fp(trend="up", vol="low")
    ts3 = _TS + timedelta(seconds=1030)
    event3 = d.evaluate(fp_up_again, ts3)
    assert event3.decision.transition_fired is False
    assert event3.decision.reason_code == "suppressed_min_dwell"


# ---------------------------------------------------------------------------
# Shock override
# ---------------------------------------------------------------------------

def test_shock_override_fires_on_extreme_vol():
    """Extreme vol_percentile (>= 0.90) bypasses HTF gate and fires transition."""
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp_init = _fp(vol_percentile=0.4)
    d.evaluate(fp_init, _TS)

    fp_shock = _fp(
        trend="down",
        vol="extreme",
        structure="compression",
        vol_percentile=0.95,  # >= 0.90 shock threshold
        realized_vol_z=3.0,   # >= 2.5 shock threshold
    )
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_shock, ts2, htf_gate_eligible=False)  # gate closed
    assert event.decision.transition_fired is True
    assert event.decision.shock_override_used is True
    assert event.decision.reason_code == "shock_override_volatility_jump"


def test_shock_override_fires_once_then_cooldown():
    """Shock fires first transition, then cooldown prevents immediate repetition."""
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=300.0)
    fp_init = _fp(vol_percentile=0.4)
    d.evaluate(fp_init, _TS)

    fp_shock = _fp(
        trend="down",
        vol="extreme",
        structure="compression",
        vol_percentile=0.95,
        realized_vol_z=3.0,
    )
    ts2 = _TS + timedelta(seconds=1)
    event1 = d.evaluate(fp_shock, ts2, htf_gate_eligible=False)
    assert event1.decision.transition_fired is True

    # 5s later — same shock conditions, but now in cooldown
    ts3 = _TS + timedelta(seconds=6)
    event2 = d.evaluate(fp_shock, ts3, htf_gate_eligible=False)
    assert event2.decision.transition_fired is False
    assert event2.decision.reason_code == "suppressed_cooldown"


def test_shock_override_realized_vol_z():
    """realized_vol_z >= 2.5 also triggers shock override."""
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp_init = _fp()
    d.evaluate(fp_init, _TS)

    # vol_percentile below shock threshold but realized_vol_z is extreme
    fp_shock = _fp(
        trend="down",
        vol="extreme",
        vol_percentile=0.5,    # below shock_vol_percentile=0.90
        realized_vol_z=2.6,    # above shock_realized_vol_z=2.5
    )
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp_shock, ts2, htf_gate_eligible=False)
    assert event.decision.shock_override_used is True
    assert event.decision.transition_fired is True


# ---------------------------------------------------------------------------
# Under-trigger guard: real change is NOT suppressed
# ---------------------------------------------------------------------------

def test_real_regime_change_fires_after_dwell():
    """A real large-distance change fires after min_dwell is satisfied."""
    d = _detector(min_dwell_seconds=300.0, cooldown_seconds=0.0, enter_threshold=0.30)
    fp_init = _fp(trend="up", vol="low")
    d.evaluate(fp_init, _TS)

    # Wait past min_dwell
    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=400)  # 400 > 300 min_dwell
    event = d.evaluate(fp_change, ts2)
    assert event.decision.transition_fired is True
    assert not event.decision.suppressed


# ---------------------------------------------------------------------------
# Telemetry emitted on every call
# ---------------------------------------------------------------------------

def test_telemetry_emitted_no_transition():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp, ts2)
    assert event.event_id != ""
    assert event.emitted_at == ts2
    # distance_result is present even when not transitioning
    assert event.decision.distance_result is not None


def test_telemetry_emitted_on_htf_gate():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)
    ts2 = _TS + timedelta(seconds=1)
    event = d.evaluate(fp, ts2, htf_gate_eligible=False)
    assert event.event_id != ""
    assert event.decision.reason_code == "htf_gate_not_ready"


def test_consecutive_stable_evals_increments():
    d = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp()
    d.evaluate(fp, _TS)  # initial state, resets counter to 0

    ts2 = _TS + timedelta(seconds=1)
    e2 = d.evaluate(fp, ts2)
    assert e2.consecutive_stable_evals >= 1

    ts3 = _TS + timedelta(seconds=2)
    e3 = d.evaluate(fp, ts3)
    assert e3.consecutive_stable_evals >= 2


# ---------------------------------------------------------------------------
# Deterministic replay invariance
# ---------------------------------------------------------------------------

def test_deterministic_replay():
    """Same input sequence produces identical decisions on fresh detector."""
    def _run(ts_offset_secs: list) -> list:
        d = RegimeTransitionDetector("BTC-USD", min_dwell_seconds=0.0, cooldown_seconds=0.0)
        fps = [
            _fp(trend="up", vol="low"),
            _fp(trend="down", vol="extreme", structure="compression"),
            _fp(trend="up", vol="low"),
        ]
        results = []
        for fp, offset in zip(fps, ts_offset_secs):
            ts = _TS + timedelta(seconds=offset)
            event = d.evaluate(fp, ts)
            results.append((event.decision.transition_fired, event.decision.reason_code))
        return results

    run1 = _run([0, 1, 2])
    run2 = _run([0, 1, 2])
    assert run1 == run2


# ---------------------------------------------------------------------------
# Cohort scope
# ---------------------------------------------------------------------------

def test_cohort_scope_supported():
    d = RegimeTransitionDetector("crypto-majors", scope="cohort")
    fp = RegimeFingerprint(
        symbol="crypto-majors",
        scope="cohort",
        cohort_id="crypto-majors",
        as_of_ts=_TS,
        bar_id="crypto-majors|1h|ts",
        source_timeframe="1h",
        trend_state="up",
        vol_state="mid",
        structure_state="neutral",
        trend_confidence=0.7,
        vol_confidence=0.5,
        structure_confidence=0.5,
        regime_confidence=0.7,
        vol_percentile=0.4,
        atr_percentile=0.4,
        volume_percentile=0.5,
        range_expansion_percentile=0.5,
        realized_vol_z=0.0,
        distance_to_htf_anchor_atr=0.5,
        numeric_vector=[0.4, 0.4, 0.5, 0.5, 0.5, 0.1],
        numeric_vector_feature_names=NUMERIC_VECTOR_FEATURE_NAMES_V1,
    )
    event = d.evaluate(fp, _TS)
    assert event.decision.scope == "cohort"
    assert event.decision.transition_fired is True


# ---------------------------------------------------------------------------
# load_state() for deterministic replay / restore
# ---------------------------------------------------------------------------

def test_load_state_restores_detector():
    d1 = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    fp = _fp(trend="up")
    d1.evaluate(fp, _TS)

    # Serialize state and restore into new detector
    saved_state = d1.state
    d2 = _detector(min_dwell_seconds=0.0, cooldown_seconds=0.0)
    d2.load_state(saved_state)

    assert d2.state.total_transitions == 1
    assert d2.state.current_fingerprint.trend_state == "up"

    # New transition should fire on large distance
    fp_change = _fp(trend="down", vol="extreme", structure="compression")
    ts2 = _TS + timedelta(seconds=1)
    event = d2.evaluate(fp_change, ts2)
    assert event.decision.transition_fired is True
    assert d2.state.total_transitions == 2
