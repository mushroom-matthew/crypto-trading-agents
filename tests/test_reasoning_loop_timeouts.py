"""Tests for timeout and degraded-mode behavior (Runbook 54).

Covers:
- CadenceConfig timeout settings are explicit and non-zero
- PolicyStateMachine pre-entry invalidation (timeout, vol_shock, etc.)
- ActivationWindowTelemetry emitted on all invalidation kinds
- Memory retrieval degraded-mode configuration is explicit (not silent)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.policy_state import PolicyStateMachineRecord
from schemas.reasoning_cadence import (
    ActivationWindowTelemetry,
    CadenceConfig,
    PreEntryInvalidationKind,
)
from services.policy_state_machine import PolicyStateMachine


_NOW = datetime.now(timezone.utc)


def _armed_record() -> PolicyStateMachineRecord:
    """Return a record already in THESIS_ARMED with a 3-bar activation window."""
    sm = PolicyStateMachine()
    record = PolicyStateMachineRecord()
    record, _ = sm.arm_thesis(record, playbook_timeout_bars=None, at=_NOW)
    return record


# ---------------------------------------------------------------------------
# Timeout configuration is explicit and non-zero
# ---------------------------------------------------------------------------


class TestTimeoutConfiguration:
    def test_tick_validation_timeout_is_positive(self):
        cfg = CadenceConfig()
        assert cfg.tick_validation_timeout_ms > 0

    def test_policy_reflection_timeout_is_positive(self):
        cfg = CadenceConfig()
        assert cfg.policy_level_reflection_timeout_ms > 0

    def test_memory_retrieval_timeout_is_positive(self):
        cfg = CadenceConfig()
        assert cfg.memory_retrieval_timeout_ms > 0

    def test_memory_retrieval_required_default_true(self):
        """memory_retrieval_required=True means degraded mode is explicit, not silent."""
        cfg = CadenceConfig()
        assert cfg.memory_retrieval_required is True

    def test_all_timeout_values_are_reasonable(self):
        """Timeout values should be in plausible ranges (ms)."""
        cfg = CadenceConfig()
        assert 1 <= cfg.tick_validation_timeout_ms <= 500
        assert 10 <= cfg.policy_level_reflection_timeout_ms <= 5000
        assert 10 <= cfg.memory_retrieval_timeout_ms <= 2000

    def test_policy_loop_min_reeval_is_positive(self):
        cfg = CadenceConfig()
        assert cfg.policy_loop_min_reeval_seconds > 0


# ---------------------------------------------------------------------------
# Pre-entry invalidation: activation timeout
# ---------------------------------------------------------------------------


class TestActivationTimeout:
    def test_activation_timeout_fires_after_timeout_bars(self):
        """After global_timeout_bars ticks, is_timed_out returns True."""
        sm = PolicyStateMachine()
        record = _armed_record()
        timeout = record.activation_window.global_timeout_bars

        for _ in range(timeout):
            record = sm.tick_activation_bar(record)

        assert sm.is_timed_out(record)

    def test_activation_timeout_not_fired_before_timeout_bars(self):
        """Before timeout_bars are elapsed, is_timed_out returns False."""
        sm = PolicyStateMachine()
        record = _armed_record()
        timeout = record.activation_window.global_timeout_bars

        for _ in range(timeout - 1):
            record = sm.tick_activation_bar(record)

        assert not sm.is_timed_out(record)

    def test_playbook_override_takes_precedence_over_global(self):
        """Playbook-specific timeout_bars overrides global default."""
        sm = PolicyStateMachine()
        record = PolicyStateMachineRecord()
        record, _ = sm.arm_thesis(record, playbook_timeout_bars=5, at=_NOW)

        window = record.activation_window
        assert window.effective_timeout_bars == 5  # overrides default 3

    def test_no_playbook_override_uses_global(self):
        """No playbook override → effective_timeout_bars == global default."""
        sm = PolicyStateMachine()
        record = _armed_record()
        window = record.activation_window
        assert window.effective_timeout_bars == window.global_timeout_bars

    def test_invalidate_pre_entry_timeout_emits_telemetry(self):
        """Activation timeout invalidation emits ActivationWindowTelemetry."""
        sm = PolicyStateMachine()
        record = _armed_record()
        # tick to timeout
        timeout = record.activation_window.global_timeout_bars
        for _ in range(timeout):
            record = sm.tick_activation_bar(record)

        new_record, telemetry = sm.invalidate_pre_entry(
            record, kind="activation_timeout", detail="3 bars elapsed"
        )
        assert isinstance(telemetry, ActivationWindowTelemetry)
        assert telemetry.outcome == "timed_out"
        assert telemetry.activation_expired_reason == "activation_timeout"
        assert telemetry.armed_duration_bars == timeout

    def test_telemetry_armed_duration_bars_matches_elapsed(self):
        """armed_duration_bars in telemetry matches bars ticked."""
        sm = PolicyStateMachine()
        record = _armed_record()

        for _ in range(2):
            record = sm.tick_activation_bar(record)

        # Invalidate before timeout
        new_record, telemetry = sm.invalidate_pre_entry(
            record, kind="structure_break", detail="key level broken"
        )
        assert telemetry.armed_duration_bars == 2


# ---------------------------------------------------------------------------
# All pre-entry invalidation kinds produce telemetry
# ---------------------------------------------------------------------------


class TestPreEntryInvalidationKinds:
    @pytest.mark.parametrize("kind", [
        "activation_timeout",
        "structure_break",
        "vol_shock",
        "regime_cancel",
    ])
    def test_all_kinds_produce_activation_window_telemetry(self, kind: PreEntryInvalidationKind):
        sm = PolicyStateMachine()
        record = _armed_record()
        new_record, telemetry = sm.invalidate_pre_entry(
            record, kind=kind, detail=f"test:{kind}"
        )
        assert isinstance(telemetry, ActivationWindowTelemetry)
        assert telemetry.activation_expired_reason == kind
        assert new_record.current_state == "INVALIDATED"
        assert new_record.activation_window is None

    def test_activation_timeout_outcome_label_is_timed_out(self):
        sm = PolicyStateMachine()
        record = _armed_record()
        _, telemetry = sm.invalidate_pre_entry(record, kind="activation_timeout")
        assert telemetry.outcome == "timed_out"

    def test_other_kinds_outcome_label_is_invalidated_pre_entry(self):
        sm = PolicyStateMachine()
        record = _armed_record()
        _, telemetry = sm.invalidate_pre_entry(record, kind="structure_break")
        assert telemetry.outcome == "invalidated_pre_entry"

    def test_invalidation_does_not_trigger_strategist_call(self):
        """Pre-entry invalidation is deterministic — no LLM output in telemetry."""
        sm = PolicyStateMachine()
        record = _armed_record()
        new_record, telemetry = sm.invalidate_pre_entry(
            record, kind="regime_cancel"
        )
        # Telemetry has no LLM-output fields; only structural fields
        assert hasattr(telemetry, "activation_expired_reason")
        assert hasattr(telemetry, "armed_duration_bars")
        assert not hasattr(telemetry, "llm_output")  # never present

    def test_pre_entry_invalidation_requires_thesis_armed_state(self):
        """Calling invalidate_pre_entry from IDLE raises ValueError."""
        sm = PolicyStateMachine()
        record = PolicyStateMachineRecord()  # IDLE
        with pytest.raises(ValueError, match="THESIS_ARMED"):
            sm.invalidate_pre_entry(record, kind="activation_timeout")


# ---------------------------------------------------------------------------
# HOLD_LOCK deterministic management only
# ---------------------------------------------------------------------------


class TestHoldLockDeterministicOnly:
    def _make_hold_lock_record(self) -> PolicyStateMachineRecord:
        sm = PolicyStateMachine()
        record = PolicyStateMachineRecord()
        record, _ = sm.arm_thesis(record, at=_NOW)
        record, _ = sm.activate_position(record, position_id="pos-1", at=_NOW)
        record = sm.lock_hold(record, at=_NOW)
        return record

    def test_hold_lock_suppresses_policy_loop(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert not sm.policy_loop_allowed(record)

    def test_hold_lock_suppresses_target_reopt(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert not sm.target_reopt_allowed(record)

    def test_hold_lock_safety_override_allows_policy_loop(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert sm.policy_loop_allowed(record, safety_override=True)

    def test_hold_lock_playbook_allows_reopt_flag(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert sm.target_reopt_allowed(record, playbook_allows_reopt=True)

    def test_hold_lock_suppresses_playbook_switch(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert not sm.playbook_switch_allowed(record)

    def test_hold_lock_invalidation_trigger_allows_playbook_switch(self):
        sm = PolicyStateMachine()
        record = self._make_hold_lock_record()
        assert sm.playbook_switch_allowed(record, has_invalidation_trigger=True)
