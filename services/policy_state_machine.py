"""Deterministic intra-policy state machine (Runbook 54).

Manages THESIS_ARMED / HOLD_LOCK boundary enforcement and emits
ActivationWindowTelemetry on every activation-window exit.

All methods are pure-deterministic (no LLM, no memory retrieval, no I/O).
State mutation returns a *new* record; input records are not modified.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from schemas.policy_state import (
    ActivationWindowState,
    PolicyStateKind,
    PolicyStateMachineRecord,
    PolicyStateTransition,
    is_playbook_switch_suppressed,
    is_policy_loop_suppressed,
    is_target_reopt_suppressed,
    is_transition_allowed,
)
from schemas.reasoning_cadence import (
    ActivationWindowTelemetry,
    CadenceConfig,
    PreEntryInvalidationKind,
    get_cadence_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add_transition(
    record: PolicyStateMachineRecord,
    to_state: PolicyStateKind,
    reason: str,
    triggered_by: Optional[str],
    at: datetime,
) -> PolicyStateMachineRecord:
    """Return a new record with the transition appended and current_state updated."""
    transition = PolicyStateTransition(
        from_state=record.current_state,
        to_state=to_state,
        transitioned_at=at,
        reason=reason,
        triggered_by=triggered_by,
    )
    new_transitions = list(record.transitions) + [transition]
    return record.model_copy(
        update={
            "current_state": to_state,
            "entered_at": at,
            "transitions": new_transitions,
        }
    )


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------


class PolicyStateMachine:
    """Deterministic state machine managing intra-policy execution boundaries.

    Usage::

        sm = PolicyStateMachine()
        record, _ = sm.arm_thesis(record, playbook_timeout_bars=5, config=config)
        record = sm.tick_activation_bar(record)
        if sm.is_timed_out(record):
            record, telemetry = sm.invalidate_pre_entry(
                record, kind="activation_timeout", detail="3 bars elapsed"
            )
    """

    def __init__(self, config: Optional[CadenceConfig] = None) -> None:
        self._cfg = config or get_cadence_config()

    # --- Transition helpers -------------------------------------------------

    def arm_thesis(
        self,
        record: PolicyStateMachineRecord,
        *,
        playbook_timeout_bars: Optional[int] = None,
        at: Optional[datetime] = None,
        reason: str = "thesis_qualified_at_policy_boundary",
    ) -> Tuple[PolicyStateMachineRecord, None]:
        """Transition IDLE → THESIS_ARMED and open the activation window."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "THESIS_ARMED"):
            raise ValueError(
                f"Cannot arm thesis from state '{record.current_state}': transition not allowed"
            )
        window = ActivationWindowState(
            thesis_armed_at=at,
            global_timeout_bars=self._cfg.policy_thesis_activation_timeout_bars,
            playbook_timeout_bars=playbook_timeout_bars,
            bars_elapsed=0,
        )
        new_record = _add_transition(record, "THESIS_ARMED", reason, "arm_thesis", at)
        new_record = new_record.model_copy(update={"activation_window": window})
        return new_record, None

    def activate_position(
        self,
        record: PolicyStateMachineRecord,
        *,
        position_id: str,
        at: Optional[datetime] = None,
        reason: str = "activation_trigger_fired",
    ) -> Tuple[PolicyStateMachineRecord, ActivationWindowTelemetry]:
        """Transition THESIS_ARMED → POSITION_OPEN; emit activation telemetry."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "POSITION_OPEN"):
            raise ValueError(
                f"Cannot activate position from state '{record.current_state}'"
            )
        window = record.activation_window
        telemetry = ActivationWindowTelemetry(
            thesis_armed_at=window.thesis_armed_at if window else at,
            resolved_at=at,
            outcome="activated",
            armed_duration_bars=window.bars_elapsed if window else 0,
        )
        new_record = _add_transition(record, "POSITION_OPEN", reason, "activation_trigger", at)
        new_record = new_record.model_copy(
            update={"activation_window": None, "position_id": position_id}
        )
        return new_record, telemetry

    def lock_hold(
        self,
        record: PolicyStateMachineRecord,
        *,
        at: Optional[datetime] = None,
        reason: str = "position_fill_received",
    ) -> PolicyStateMachineRecord:
        """Transition POSITION_OPEN → HOLD_LOCK."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "HOLD_LOCK"):
            raise ValueError(
                f"Cannot enter HOLD_LOCK from state '{record.current_state}'"
            )
        return _add_transition(record, "HOLD_LOCK", reason, "fill_event", at)

    def invalidate_pre_entry(
        self,
        record: PolicyStateMachineRecord,
        *,
        kind: PreEntryInvalidationKind,
        detail: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> Tuple[PolicyStateMachineRecord, ActivationWindowTelemetry]:
        """Exit THESIS_ARMED via deterministic pre-entry invalidation.

        Does NOT trigger a strategist call — the tick engine handles this.
        Transitions to INVALIDATED (caller should follow with enter_cooldown).
        """
        at = at or _now()
        if record.current_state != "THESIS_ARMED":
            raise ValueError(
                f"Pre-entry invalidation requires THESIS_ARMED, got '{record.current_state}'"
            )
        window = record.activation_window
        outcome: str = "timed_out" if kind == "activation_timeout" else "invalidated_pre_entry"
        telemetry = ActivationWindowTelemetry(
            thesis_armed_at=window.thesis_armed_at if window else at,
            resolved_at=at,
            outcome=outcome,  # type: ignore[arg-type]
            activation_expired_reason=kind,
            armed_duration_bars=window.bars_elapsed if window else 0,
            invalidation_detail=detail,
        )
        reason = f"pre_entry_invalidation:{kind}"
        new_record = _add_transition(record, "INVALIDATED", reason, "tick_engine", at)
        new_record = new_record.model_copy(update={"activation_window": None})
        logger.debug(
            "PolicySM pre-entry invalidation kind=%s bars=%d detail=%s",
            kind,
            telemetry.armed_duration_bars,
            detail,
        )
        return new_record, telemetry

    def close_position(
        self,
        record: PolicyStateMachineRecord,
        *,
        at: Optional[datetime] = None,
        reason: str = "position_exit",
    ) -> PolicyStateMachineRecord:
        """Exit POSITION_OPEN or HOLD_LOCK to COOLDOWN."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "COOLDOWN"):
            raise ValueError(
                f"Cannot close position from state '{record.current_state}'"
            )
        new_record = _add_transition(record, "COOLDOWN", reason, "exit_event", at)
        return new_record.model_copy(update={"position_id": None})

    def enter_cooldown(
        self,
        record: PolicyStateMachineRecord,
        *,
        at: Optional[datetime] = None,
        reason: str = "post_invalidation_cooldown",
    ) -> PolicyStateMachineRecord:
        """Transition INVALIDATED → COOLDOWN."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "COOLDOWN"):
            raise ValueError(
                f"Cannot enter cooldown from state '{record.current_state}'"
            )
        return _add_transition(record, "COOLDOWN", reason, "cooldown_timer", at)

    def reset_to_idle(
        self,
        record: PolicyStateMachineRecord,
        *,
        at: Optional[datetime] = None,
        reason: str = "cooldown_expired",
    ) -> PolicyStateMachineRecord:
        """Transition COOLDOWN → IDLE (ready to re-arm)."""
        at = at or _now()
        if not is_transition_allowed(record.current_state, "IDLE"):
            raise ValueError(
                f"Cannot reset to IDLE from state '{record.current_state}'"
            )
        return _add_transition(record, "IDLE", reason, "cooldown_timer", at)

    # --- Activation window management ---------------------------------------

    def tick_activation_bar(
        self, record: PolicyStateMachineRecord
    ) -> PolicyStateMachineRecord:
        """Increment bars_elapsed in the activation window (called once per bar).

        Only meaningful in THESIS_ARMED state; no-op otherwise.
        """
        if record.current_state != "THESIS_ARMED" or record.activation_window is None:
            return record
        new_window = record.activation_window.model_copy(
            update={"bars_elapsed": record.activation_window.bars_elapsed + 1}
        )
        return record.model_copy(update={"activation_window": new_window})

    def is_timed_out(self, record: PolicyStateMachineRecord) -> bool:
        """Return True when the THESIS_ARMED window has exceeded its timeout."""
        if record.current_state != "THESIS_ARMED" or record.activation_window is None:
            return False
        return record.activation_window.is_timed_out

    # --- Guard helpers (wrappers for readability) ---------------------------

    def policy_loop_allowed(
        self, record: PolicyStateMachineRecord, *, safety_override: bool = False
    ) -> bool:
        """Return True when the policy loop (strategist/judge) MAY run."""
        return not is_policy_loop_suppressed(record.current_state, safety_override=safety_override)

    def playbook_switch_allowed(
        self,
        record: PolicyStateMachineRecord,
        *,
        has_invalidation_trigger: bool = False,
        has_safety_override: bool = False,
    ) -> bool:
        """Return True when a playbook switch is permitted."""
        return not is_playbook_switch_suppressed(
            record.current_state,
            has_invalidation_trigger=has_invalidation_trigger,
            has_safety_override=has_safety_override,
        )

    def target_reopt_allowed(
        self,
        record: PolicyStateMachineRecord,
        *,
        has_safety_override: bool = False,
        playbook_allows_reopt: bool = False,
    ) -> bool:
        """Return True when target re-optimization is permitted."""
        return not is_target_reopt_suppressed(
            record.current_state,
            has_safety_override=has_safety_override,
            playbook_allows_reopt=playbook_allows_reopt,
        )
