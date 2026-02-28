"""Tests for policy loop gate, state machine boundaries, single-flight guards (Runbook 54).

Covers:
- PolicyLoopGate: allow / skip decision paths
- Single-flight lock: duplicate-run suppression
- THESIS_ARMED boundary: policy loop suppressed during activation window
- HOLD_LOCK boundary: policy loop and target-reopt suppressed
- State machine transition graph: valid and invalid transitions
- Policy cooldown enforcement
- Heartbeat expiry gate
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from schemas.policy_state import (
    PolicyStateMachineRecord,
    is_playbook_switch_suppressed,
    is_policy_loop_suppressed,
    is_target_reopt_suppressed,
    is_transition_allowed,
)
from schemas.reasoning_cadence import (
    CadenceConfig,
    PolicyLoopTriggerEvent,
)
from services.policy_loop_gate import PolicyLoopGate
from services.policy_state_machine import PolicyStateMachine

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    heartbeat_5m: int = 3600,
    min_reeval: int = 900,
    require_event: bool = True,
) -> CadenceConfig:
    cfg = CadenceConfig.__new__(CadenceConfig)
    cfg.tick_engine_deterministic_only = True
    cfg.tick_validation_timeout_ms = 50
    cfg.policy_loop_enabled = True
    cfg.policy_loop_heartbeat_1m_seconds = 900
    cfg.policy_loop_heartbeat_5m_seconds = heartbeat_5m
    cfg.policy_loop_min_reeval_seconds = min_reeval
    cfg.policy_loop_require_event_or_heartbeat = require_event
    cfg.policy_thesis_activation_timeout_bars = 3
    cfg.policy_hold_lock_enforced = True
    cfg.policy_target_reopt_enabled = False
    cfg.policy_rearm_requires_next_boundary = True
    cfg.policy_level_reflection_enabled = True
    cfg.policy_level_reflection_timeout_ms = 250
    cfg.memory_retrieval_timeout_ms = 150
    cfg.memory_retrieval_required = True
    cfg.memory_retrieval_reuse_enabled = True
    cfg.memory_requery_regime_delta_threshold = 0.15
    cfg.high_level_reflection_enabled = True
    cfg.high_level_reflection_min_interval_hours = 24
    cfg.high_level_reflection_min_episodes = 20
    cfg.playbook_metadata_refresh_hours = 168
    cfg.playbook_metadata_drift_trigger = True
    cfg.regime_transition_detector_enabled = True
    cfg.regime_transition_min_confidence_delta = 0.20
    cfg.vol_percentile_band_shift_trigger = True
    cfg.regime_fingerprint_relearn_days = 30
    cfg.regime_fingerprint_drift_threshold = 0.30
    return cfg


def _trigger(trigger_type: str = "regime_state_changed") -> PolicyLoopTriggerEvent:
    return PolicyLoopTriggerEvent(trigger_type=trigger_type, fired_at=_NOW)  # type: ignore[arg-type]


def _idle_record() -> PolicyStateMachineRecord:
    return PolicyStateMachineRecord()


def _armed_record() -> PolicyStateMachineRecord:
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record, at=_NOW)
    return record


def _hold_lock_record() -> PolicyStateMachineRecord:
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record, at=_NOW)
    record, _ = sm.activate_position(record, position_id="pos-1", at=_NOW)
    record = sm.lock_hold(record, at=_NOW)
    return record


# Use a unique scope per test to avoid cross-test lock state
import uuid


def _scope() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# PolicyLoopGate: allow path
# ---------------------------------------------------------------------------


class TestPolicyLoopGateAllow:
    def test_allow_on_first_eval_with_trigger(self):
        """First eval + trigger event → allowed."""
        gate = PolicyLoopGate(config=_cfg())
        scope = _scope()
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            indicator_timeframe="5m",
            at=_NOW,
        )
        assert allowed
        assert skip is None
        gate.release(scope)

    def test_allow_when_heartbeat_expired_no_trigger(self):
        """No trigger but heartbeat expired → allowed."""
        gate = PolicyLoopGate(config=_cfg(heartbeat_5m=100, min_reeval=0))
        scope = _scope()
        old_eval = _NOW - timedelta(seconds=200)
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[],
            last_eval_at=old_eval,
            indicator_timeframe="5m",
            at=_NOW,
        )
        assert allowed
        gate.release(scope)

    def test_allow_on_first_eval_no_trigger_require_event_false(self):
        """require_event_or_heartbeat=False → allow even without trigger."""
        gate = PolicyLoopGate(config=_cfg(require_event=False, min_reeval=0))
        scope = _scope()
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[],
            last_eval_at=_NOW - timedelta(hours=2),
            indicator_timeframe="5m",
            at=_NOW,
        )
        assert allowed
        gate.release(scope)


# ---------------------------------------------------------------------------
# PolicyLoopGate: skip paths
# ---------------------------------------------------------------------------


class TestPolicyLoopGateSkip:
    def test_skip_thesis_armed_state(self):
        """THESIS_ARMED → skip with policy_frozen_thesis_armed."""
        gate = PolicyLoopGate(config=_cfg())
        scope = _scope()
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_armed_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert not allowed
        assert skip.skip_reason == "policy_frozen_thesis_armed"

    def test_skip_hold_lock_state(self):
        """HOLD_LOCK → skip with policy_frozen_hold_lock."""
        gate = PolicyLoopGate(config=_cfg())
        scope = _scope()
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_hold_lock_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert not allowed
        assert skip.skip_reason == "policy_frozen_hold_lock"

    def test_safety_override_bypasses_thesis_armed(self):
        """safety_override=True bypasses THESIS_ARMED suppression."""
        gate = PolicyLoopGate(config=_cfg(min_reeval=0))
        scope = _scope()
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_armed_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            safety_override=True,
            at=_NOW,
        )
        assert allowed
        gate.release(scope)

    def test_skip_cooldown_not_expired(self):
        """Within min_reeval_seconds → skip with cooldown_not_expired."""
        gate = PolicyLoopGate(config=_cfg(min_reeval=900))
        scope = _scope()
        recent = _NOW - timedelta(seconds=100)
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=recent,
            at=_NOW,
        )
        assert not allowed
        assert skip.skip_reason == "cooldown_not_expired"
        assert skip.next_eligible_at is not None

    def test_skip_no_trigger_and_no_heartbeat(self):
        """No trigger events and heartbeat not expired → skip."""
        gate = PolicyLoopGate(config=_cfg(heartbeat_5m=3600, min_reeval=0))
        scope = _scope()
        recent = _NOW - timedelta(seconds=100)
        allowed, skip = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[],
            last_eval_at=recent,
            indicator_timeframe="5m",
            at=_NOW,
        )
        assert not allowed
        assert skip.skip_reason == "no_trigger_and_no_heartbeat"
        assert skip.next_eligible_at is not None

    def test_skip_event_always_has_skipped_at(self):
        """All skip events include skipped_at timestamp."""
        gate = PolicyLoopGate(config=_cfg())
        scope = _scope()
        _, skip = gate.evaluate(
            scope=scope,
            state_record=_armed_record(),
            trigger_events=[],
            last_eval_at=None,
            at=_NOW,
        )
        assert skip.skipped_at is not None

    def test_skip_event_includes_policy_state(self):
        """Skip event records the policy state at skip time."""
        gate = PolicyLoopGate(config=_cfg())
        _, skip = gate.evaluate(
            scope=_scope(),
            state_record=_armed_record(),
            trigger_events=[],
            last_eval_at=None,
            at=_NOW,
        )
        assert skip.policy_state_at_skip == "THESIS_ARMED"


# ---------------------------------------------------------------------------
# Single-flight guard
# ---------------------------------------------------------------------------


class TestSingleFlightGuard:
    def test_second_call_blocked_while_first_lock_held(self):
        """A second evaluate() for the same scope is blocked when lock is held."""
        gate = PolicyLoopGate(config=_cfg(min_reeval=0))
        scope = _scope()

        # First call acquires lock
        allowed1, _ = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert allowed1

        # Second call should be blocked (already_running)
        allowed2, skip2 = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert not allowed2
        assert skip2.skip_reason == "already_running"

        gate.release(scope)

    def test_after_release_second_call_allowed(self):
        """After lock release, a new evaluate() is allowed."""
        gate = PolicyLoopGate(config=_cfg(min_reeval=0))
        scope = _scope()

        allowed1, _ = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert allowed1
        gate.release(scope)

        allowed2, skip2 = gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert allowed2
        gate.release(scope)

    def test_different_scopes_independent(self):
        """Two different scopes have independent locks."""
        gate = PolicyLoopGate(config=_cfg(min_reeval=0))
        scope_a = _scope()
        scope_b = _scope()

        allowed_a, _ = gate.evaluate(
            scope=scope_a,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        allowed_b, _ = gate.evaluate(
            scope=scope_b,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert allowed_a
        assert allowed_b
        gate.release(scope_a)
        gate.release(scope_b)

    def test_is_scope_active_reflects_lock_state(self):
        gate = PolicyLoopGate(config=_cfg(min_reeval=0))
        scope = _scope()
        assert not gate.is_scope_active(scope)

        gate.evaluate(
            scope=scope,
            state_record=_idle_record(),
            trigger_events=[_trigger()],
            last_eval_at=None,
            at=_NOW,
        )
        assert gate.is_scope_active(scope)
        gate.release(scope)
        assert not gate.is_scope_active(scope)


# ---------------------------------------------------------------------------
# State machine transition graph
# ---------------------------------------------------------------------------


class TestStateMachineTransitions:
    def test_valid_idle_to_thesis_armed(self):
        assert is_transition_allowed("IDLE", "THESIS_ARMED")

    def test_valid_thesis_armed_to_position_open(self):
        assert is_transition_allowed("THESIS_ARMED", "POSITION_OPEN")

    def test_valid_thesis_armed_to_invalidated(self):
        assert is_transition_allowed("THESIS_ARMED", "INVALIDATED")

    def test_valid_position_open_to_hold_lock(self):
        assert is_transition_allowed("POSITION_OPEN", "HOLD_LOCK")

    def test_valid_hold_lock_to_cooldown(self):
        assert is_transition_allowed("HOLD_LOCK", "COOLDOWN")

    def test_valid_cooldown_to_idle(self):
        assert is_transition_allowed("COOLDOWN", "IDLE")

    def test_invalid_idle_to_hold_lock(self):
        assert not is_transition_allowed("IDLE", "HOLD_LOCK")

    def test_invalid_hold_lock_to_thesis_armed(self):
        assert not is_transition_allowed("HOLD_LOCK", "THESIS_ARMED")

    def test_invalid_cooldown_to_thesis_armed(self):
        assert not is_transition_allowed("COOLDOWN", "THESIS_ARMED")

    def test_arm_thesis_raises_from_hold_lock(self):
        sm = PolicyStateMachine()
        record = _hold_lock_record()
        with pytest.raises(ValueError, match="transition not allowed"):
            sm.arm_thesis(record, at=_NOW)

    def test_activate_position_emits_telemetry(self):
        sm = PolicyStateMachine()
        record = _armed_record()
        new_record, telemetry = sm.activate_position(record, position_id="pos-1", at=_NOW)
        assert new_record.current_state == "POSITION_OPEN"
        assert telemetry.outcome == "activated"

    def test_full_lifecycle(self):
        """IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK → COOLDOWN → IDLE."""
        sm = PolicyStateMachine()
        record = PolicyStateMachineRecord()
        assert record.current_state == "IDLE"

        record, _ = sm.arm_thesis(record, at=_NOW)
        assert record.current_state == "THESIS_ARMED"

        record, telemetry = sm.activate_position(record, position_id="pos-1", at=_NOW)
        assert record.current_state == "POSITION_OPEN"
        assert telemetry.outcome == "activated"

        record = sm.lock_hold(record, at=_NOW)
        assert record.current_state == "HOLD_LOCK"

        record = sm.close_position(record, at=_NOW)
        assert record.current_state == "COOLDOWN"

        record = sm.reset_to_idle(record, at=_NOW)
        assert record.current_state == "IDLE"

    def test_transition_history_recorded(self):
        """Each transition is appended to record.transitions."""
        sm = PolicyStateMachine()
        record = PolicyStateMachineRecord()
        record, _ = sm.arm_thesis(record, at=_NOW)
        assert len(record.transitions) == 1
        assert record.transitions[0].from_state == "IDLE"
        assert record.transitions[0].to_state == "THESIS_ARMED"


# ---------------------------------------------------------------------------
# Guard functions (schema-level)
# ---------------------------------------------------------------------------


class TestGuardFunctions:
    def test_idle_not_suppressed(self):
        assert not is_policy_loop_suppressed("IDLE")

    def test_thesis_armed_suppressed(self):
        assert is_policy_loop_suppressed("THESIS_ARMED")

    def test_hold_lock_suppressed(self):
        assert is_policy_loop_suppressed("HOLD_LOCK")

    def test_safety_override_lifts_suppression(self):
        assert not is_policy_loop_suppressed("THESIS_ARMED", safety_override=True)
        assert not is_policy_loop_suppressed("HOLD_LOCK", safety_override=True)

    def test_hold_lock_suppresses_target_reopt(self):
        assert is_target_reopt_suppressed("HOLD_LOCK")

    def test_idle_does_not_suppress_target_reopt(self):
        assert not is_target_reopt_suppressed("IDLE")

    def test_cooldown_suppresses_playbook_switch(self):
        assert is_playbook_switch_suppressed("COOLDOWN")

    def test_invalidation_trigger_lifts_playbook_switch_suppression(self):
        assert not is_playbook_switch_suppressed("COOLDOWN", has_invalidation_trigger=True)


# ---------------------------------------------------------------------------
# has_trigger_type helper
# ---------------------------------------------------------------------------


class TestHasTriggerType:
    def test_found_trigger_type(self):
        events = [_trigger("regime_state_changed"), _trigger("position_closed")]
        assert PolicyLoopGate.has_trigger_type(events, "position_closed")

    def test_missing_trigger_type(self):
        events = [_trigger("regime_state_changed")]
        assert not PolicyLoopGate.has_trigger_type(events, "position_closed")

    def test_empty_events_returns_false(self):
        assert not PolicyLoopGate.has_trigger_type([], "heartbeat_expired")
