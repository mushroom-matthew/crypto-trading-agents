"""Policy loop gate — event-driven cadence enforcement (Runbook 54).

Determines whether the policy loop (strategist + memory retrieval + judge) may
run for a given evaluation cycle.  All decisions are deterministic; no LLM
calls are made here.

Gate logic:
  1. If policy is frozen (THESIS_ARMED / HOLD_LOCK without override) → skip
  2. If single-flight lock held for this scope → skip (already_running)
  3. If no event trigger fired AND heartbeat not expired → skip
  4. If minimum reevaluation cooldown not elapsed → skip
  5. Otherwise → allow, acquire lock

Every skip emits a ``PolicyLoopSkipEvent`` for observability (Rule #5 of
Runbook 54 Operational Rules).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from schemas.policy_state import PolicyStateMachineRecord, is_policy_loop_suppressed
from schemas.reasoning_cadence import (
    CadenceConfig,
    PolicyLoopSkipEvent,
    PolicyLoopSkipReason,
    PolicyLoopTriggerEvent,
    PolicyLoopTriggerType,
    get_cadence_config,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Single-flight lock (per scope key)
# ---------------------------------------------------------------------------


class _SingleFlightRegistry:
    """Thread-safe registry of single-flight locks keyed by scope string."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Dict[str, bool] = {}

    def try_acquire(self, scope: str) -> bool:
        """Return True and mark active; return False if already active."""
        with self._lock:
            if self._active.get(scope, False):
                return False
            self._active[scope] = True
            return True

    def release(self, scope: str) -> None:
        """Mark scope as no longer running."""
        with self._lock:
            self._active.pop(scope, None)

    def is_active(self, scope: str) -> bool:
        with self._lock:
            return self._active.get(scope, False)


_registry = _SingleFlightRegistry()


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class PolicyLoopGate:
    """Evaluate whether the policy loop may run this cycle.

    Usage::

        gate = PolicyLoopGate()
        allowed, skip_event = gate.evaluate(
            scope="BTC-USD",
            state_record=record,
            trigger_events=[PolicyLoopTriggerEvent(...)],
            last_eval_at=last_eval,
            indicator_timeframe="1m",
        )
        if allowed:
            try:
                # ... run policy loop ...
            finally:
                gate.release(scope="BTC-USD")
        else:
            log(skip_event)
    """

    def __init__(self, config: Optional[CadenceConfig] = None) -> None:
        self._cfg = config or get_cadence_config()

    # --- Public API ---------------------------------------------------------

    def evaluate(
        self,
        *,
        scope: str,
        state_record: PolicyStateMachineRecord,
        trigger_events: List[PolicyLoopTriggerEvent],
        last_eval_at: Optional[datetime],
        indicator_timeframe: str = "5m",
        safety_override: bool = False,
        at: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[PolicyLoopSkipEvent]]:
        """Evaluate gate conditions and return (allowed, skip_event_or_None).

        If allowed=True, the single-flight lock for ``scope`` is acquired.
        The caller MUST call ``release(scope)`` after the policy loop finishes.
        """
        now = at or _now()

        # 1. Policy frozen (THESIS_ARMED / HOLD_LOCK)
        if is_policy_loop_suppressed(state_record.current_state, safety_override=safety_override):
            reason: PolicyLoopSkipReason = (
                "policy_frozen_thesis_armed"
                if state_record.current_state == "THESIS_ARMED"
                else "policy_frozen_hold_lock"
            )
            return False, self._skip(reason, now, state_record.current_state)

        # 2. Single-flight check
        if not _registry.try_acquire(scope):
            return False, self._skip("already_running", now, state_record.current_state)

        # Check remaining conditions; release lock on skip
        try:
            # 3. Cooldown / min reevaluation interval
            if last_eval_at is not None:
                elapsed = (now - last_eval_at).total_seconds()
                if elapsed < self._cfg.policy_loop_min_reeval_seconds:
                    next_eligible = last_eval_at + timedelta(
                        seconds=self._cfg.policy_loop_min_reeval_seconds
                    )
                    _registry.release(scope)
                    return False, self._skip(
                        "cooldown_not_expired", now, state_record.current_state,
                        next_eligible=next_eligible,
                        detail=f"elapsed={elapsed:.0f}s < min={self._cfg.policy_loop_min_reeval_seconds}s",
                    )

            # 4. Require event or heartbeat
            if self._cfg.policy_loop_require_event_or_heartbeat:
                has_trigger = bool(trigger_events)
                heartbeat = self._cfg.heartbeat_for_timeframe(indicator_timeframe)
                heartbeat_expired = (
                    last_eval_at is None
                    or (now - last_eval_at).total_seconds() >= heartbeat
                )
                if not has_trigger and not heartbeat_expired:
                    next_heartbeat = (last_eval_at or now) + timedelta(seconds=heartbeat)
                    _registry.release(scope)
                    return False, self._skip(
                        "no_trigger_and_no_heartbeat", now, state_record.current_state,
                        next_eligible=next_heartbeat,
                        detail=f"heartbeat={heartbeat}s not expired",
                    )

        except Exception:
            _registry.release(scope)
            raise

        # Allowed — lock remains held
        logger.debug(
            "PolicyLoopGate ALLOW scope=%s triggers=%s state=%s",
            scope,
            [e.trigger_type for e in trigger_events],
            state_record.current_state,
        )
        return True, None

    def release(self, scope: str) -> None:
        """Release the single-flight lock for ``scope``."""
        _registry.release(scope)

    def is_scope_active(self, scope: str) -> bool:
        """Return True if a policy loop is currently running for ``scope``."""
        return _registry.is_active(scope)

    # --- Convenience: summarise trigger events ------------------------------

    @staticmethod
    def has_trigger_type(
        trigger_events: List[PolicyLoopTriggerEvent],
        trigger_type: PolicyLoopTriggerType,
    ) -> bool:
        """Return True when trigger_events contains a specific trigger type."""
        return any(e.trigger_type == trigger_type for e in trigger_events)

    # --- Internal -----------------------------------------------------------

    @staticmethod
    def _skip(
        reason: PolicyLoopSkipReason,
        now: datetime,
        state: str,
        next_eligible: Optional[datetime] = None,
        detail: Optional[str] = None,
    ) -> PolicyLoopSkipEvent:
        return PolicyLoopSkipEvent(
            skip_reason=reason,
            skipped_at=now,
            next_eligible_at=next_eligible,
            policy_state_at_skip=state,
            detail=detail,
        )
