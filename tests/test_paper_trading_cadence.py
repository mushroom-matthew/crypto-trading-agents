"""Tests for Runbook 61: Policy Loop Cadence Wiring.

Verifies that:
- PolicyLoopGate blocks plan generation when state is HOLD_LOCK.
- PolicyLoopGate allows plan generation on regime transition trigger.
- PolicyLoopGate allows plan generation on heartbeat expiry.
- State machine transitions to HOLD_LOCK on fill (IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK).
- State machine transitions to COOLDOWN on position close.
- SessionState includes policy cadence fields and round-trips through model_dump.
- Detector state persists and restores from the SessionState dict.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from schemas.policy_state import PolicyStateMachineRecord
from schemas.reasoning_cadence import PolicyLoopTriggerEvent, get_cadence_config
from services.policy_loop_gate import PolicyLoopGate
from services.policy_state_machine import PolicyStateMachine
from tools.paper_trading import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _make_trigger(trigger_type: str) -> PolicyLoopTriggerEvent:
    return PolicyLoopTriggerEvent(
        trigger_type=trigger_type,
        fired_at=datetime.now(timezone.utc),
    )


def _idle_record() -> PolicyStateMachineRecord:
    return PolicyStateMachineRecord()


def _hold_lock_record() -> PolicyStateMachineRecord:
    """Build a HOLD_LOCK record via legitimate state machine transitions."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record)
    record, _ = sm.activate_position(record, position_id="test-pos-1")
    record = sm.lock_hold(record)
    return record


def _position_open_record() -> PolicyStateMachineRecord:
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record)
    record, _ = sm.activate_position(record, position_id="test-pos-1")
    return record


# ---------------------------------------------------------------------------
# Gate: HOLD_LOCK blocks plan generation
# ---------------------------------------------------------------------------


def test_gate_blocks_hold_lock():
    """Gate must block when policy state is HOLD_LOCK."""
    gate = PolicyLoopGate()
    record = _hold_lock_record()
    allowed, skip_event = gate.evaluate(
        scope="session-hold",
        state_record=record,
        trigger_events=[],
        last_eval_at=None,
        indicator_timeframe="5m",
    )
    assert not allowed, "Gate should block HOLD_LOCK state"
    assert skip_event is not None
    assert skip_event.skip_reason == "policy_frozen_hold_lock"


def test_gate_blocks_thesis_armed_no_trigger():
    """Gate must block when policy state is THESIS_ARMED (frozen state)."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record)

    gate = PolicyLoopGate()
    allowed, skip_event = gate.evaluate(
        scope="session-armed",
        state_record=record,
        trigger_events=[],
        last_eval_at=None,
        indicator_timeframe="5m",
    )
    assert not allowed
    assert skip_event is not None
    assert skip_event.skip_reason == "policy_frozen_thesis_armed"


# ---------------------------------------------------------------------------
# Gate: regime transition trigger allows plan generation
# ---------------------------------------------------------------------------


def test_gate_allows_on_regime_transition():
    """Gate must allow when a regime_state_changed trigger fires (IDLE state)."""
    gate = PolicyLoopGate()
    record = _idle_record()
    triggers = [_make_trigger("regime_state_changed")]

    # Set last_eval_at far in the past so cooldown is not the limiting factor.
    old_eval = _utc(datetime.now(timezone.utc) - timedelta(hours=2))

    allowed, skip_event = gate.evaluate(
        scope="session-regime-trigger",
        state_record=record,
        trigger_events=triggers,
        last_eval_at=old_eval,
        indicator_timeframe="5m",
    )
    assert allowed, f"Gate should allow on regime trigger; skip_event={skip_event}"
    # Must release the lock after allowed
    gate.release("session-regime-trigger")


# ---------------------------------------------------------------------------
# Gate: heartbeat expiry allows plan generation
# ---------------------------------------------------------------------------


def test_gate_allows_on_heartbeat_expiry():
    """Gate must allow when heartbeat has expired (no trigger events)."""
    gate = PolicyLoopGate()
    record = _idle_record()
    cfg = get_cadence_config()
    heartbeat_5m = cfg.heartbeat_for_timeframe("5m")

    # last_eval_at older than heartbeat
    old_eval = _utc(
        datetime.now(timezone.utc) - timedelta(seconds=heartbeat_5m + 60)
    )
    allowed, skip_event = gate.evaluate(
        scope="session-heartbeat",
        state_record=record,
        trigger_events=[],  # no triggers — heartbeat alone must allow
        last_eval_at=old_eval,
        indicator_timeframe="5m",
    )
    assert allowed, f"Gate should allow on heartbeat expiry; skip_event={skip_event}"
    gate.release("session-heartbeat")


def test_gate_blocks_when_no_trigger_and_heartbeat_not_expired():
    """Gate must block when neither a trigger nor heartbeat expiry is present."""
    gate = PolicyLoopGate()
    record = _idle_record()

    # last_eval_at very recent — heartbeat has not expired
    recent_eval = _utc(datetime.now(timezone.utc) - timedelta(seconds=10))

    allowed, skip_event = gate.evaluate(
        scope="session-no-trigger",
        state_record=record,
        trigger_events=[],
        last_eval_at=recent_eval,
        indicator_timeframe="5m",
    )
    assert not allowed
    assert skip_event is not None
    assert skip_event.skip_reason in (
        "no_trigger_and_no_heartbeat",
        "cooldown_not_expired",
    )


# ---------------------------------------------------------------------------
# State machine: IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK on fill
# ---------------------------------------------------------------------------


def test_state_machine_fill_transitions():
    """Filling an entry order from IDLE must reach HOLD_LOCK."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    assert record.current_state == "IDLE"

    # Simulate the fill handler logic from paper_trading._execute_order
    if record.current_state == "IDLE":
        record, _ = sm.arm_thesis(record)
    assert record.current_state == "THESIS_ARMED"

    if record.current_state == "THESIS_ARMED":
        record, _ = sm.activate_position(record, position_id="fill-pos-1")
    assert record.current_state == "POSITION_OPEN"

    if record.current_state == "POSITION_OPEN":
        record = sm.lock_hold(record)
    assert record.current_state == "HOLD_LOCK"


def test_state_machine_fill_from_thesis_armed():
    """If already THESIS_ARMED (plan just generated), fill transitions to HOLD_LOCK."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record)
    assert record.current_state == "THESIS_ARMED"

    # Simulate fill handler
    if record.current_state == "THESIS_ARMED":
        record, _ = sm.activate_position(record, position_id="fill-pos-2")
    if record.current_state == "POSITION_OPEN":
        record = sm.lock_hold(record)

    assert record.current_state == "HOLD_LOCK"


# ---------------------------------------------------------------------------
# State machine: HOLD_LOCK → COOLDOWN on position close
# ---------------------------------------------------------------------------


def test_state_machine_close_position():
    """Closing a position from HOLD_LOCK must transition to COOLDOWN."""
    sm = PolicyStateMachine()
    record = _hold_lock_record()
    assert record.current_state == "HOLD_LOCK"

    # Simulate the close handler logic from paper_trading._sweep_stop_target
    if record.current_state in ("POSITION_OPEN", "HOLD_LOCK"):
        record = sm.close_position(record)

    assert record.current_state == "COOLDOWN"


def test_state_machine_close_from_position_open():
    """Closing a position from POSITION_OPEN (stop hit before lock_hold) → COOLDOWN."""
    sm = PolicyStateMachine()
    record = _position_open_record()
    assert record.current_state == "POSITION_OPEN"

    if record.current_state in ("POSITION_OPEN", "HOLD_LOCK"):
        record = sm.close_position(record)

    assert record.current_state == "COOLDOWN"


# ---------------------------------------------------------------------------
# SessionState: new cadence fields present and round-trip correctly
# ---------------------------------------------------------------------------


def test_session_state_has_cadence_fields():
    """SessionState must include all three Runbook 61 cadence fields."""
    state = SessionState(
        session_id="s1",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
    )
    assert state.policy_state_machine_record is None
    assert state.regime_detector_state is None
    assert state.last_policy_eval_at is None


def test_session_state_roundtrip_cadence():
    """Cadence fields must survive a model_dump / model_validate round-trip."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record)
    record_dict = record.model_dump()

    iso_ts = datetime.now(timezone.utc).isoformat()

    state = SessionState(
        session_id="s2",
        symbols=["ETH-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
        policy_state_machine_record=record_dict,
        regime_detector_state={"symbol": "ETH-USD", "scope": "symbol"},
        last_policy_eval_at=iso_ts,
    )
    raw = state.model_dump()
    restored = SessionState.model_validate(raw)

    assert restored.policy_state_machine_record is not None
    assert restored.regime_detector_state is not None
    assert restored.last_policy_eval_at == iso_ts

    # Verify the state machine record can be re-hydrated
    rehydrated = PolicyStateMachineRecord.model_validate(
        restored.policy_state_machine_record
    )
    assert rehydrated.current_state == "THESIS_ARMED"


def test_session_state_none_cadence_fields_survive_roundtrip():
    """None cadence fields must still be None after round-trip."""
    state = SessionState(
        session_id="s3",
        symbols=["SOL-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
    )
    raw = state.model_dump()
    restored = SessionState.model_validate(raw)

    assert restored.policy_state_machine_record is None
    assert restored.regime_detector_state is None
    assert restored.last_policy_eval_at is None


# ---------------------------------------------------------------------------
# Gate: release is always called (verify single-flight guard is freed)
# ---------------------------------------------------------------------------


def test_gate_release_frees_scope():
    """After gate.release(), the same scope can be acquired again."""
    gate = PolicyLoopGate()
    record = _idle_record()
    old_eval = _utc(datetime.now(timezone.utc) - timedelta(hours=2))

    allowed, _ = gate.evaluate(
        scope="session-release-test",
        state_record=record,
        trigger_events=[_make_trigger("position_opened")],
        last_eval_at=old_eval,
        indicator_timeframe="1h",
    )
    assert allowed

    gate.release("session-release-test")

    # Now the scope is free — a second acquire should succeed
    allowed2, _ = gate.evaluate(
        scope="session-release-test",
        state_record=record,
        trigger_events=[_make_trigger("position_closed")],
        last_eval_at=_utc(datetime.now(timezone.utc) - timedelta(hours=1)),
        indicator_timeframe="1h",
    )
    assert allowed2
    gate.release("session-release-test")
