"""Intra-policy state machine schemas (Runbook 54).

Models the deterministic state transitions within a single policy window:

  IDLE ──► THESIS_ARMED ──► POSITION_OPEN ──► HOLD_LOCK ──► COOLDOWN ──► IDLE
                │                                  │
                └──────────── INVALIDATED ◄────────┘

Transition guards enforced here:
- THESIS_ARMED: policy loop, reflection, and playbook switching are suppressed
- HOLD_LOCK: strategist re-selection and target re-optimization are suppressed
- Both states allow safety/invalidation overrides to create a new policy boundary

Telemetry recorded:
- activation_expired_reason (PreEntryInvalidationKind) and armed_duration_bars
  on every THESIS_ARMED exit
"""

from __future__ import annotations

from datetime import datetime
from typing import FrozenSet, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

PolicyStateKind = Literal[
    "IDLE",           # no active thesis or position
    "THESIS_ARMED",   # thesis qualified, waiting for activation trigger
    "POSITION_OPEN",  # activated, position live; may enter HOLD_LOCK
    "HOLD_LOCK",      # position open, deterministic management only
    "INVALIDATED",    # thesis/position invalidated, pending cooldown
    "COOLDOWN",       # post-invalidation/exit cooling period before re-arm
]

# ---------------------------------------------------------------------------
# Allowed state transitions (deterministic graph)
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[PolicyStateKind, FrozenSet[PolicyStateKind]] = {
    "IDLE":          frozenset({"THESIS_ARMED"}),
    "THESIS_ARMED":  frozenset({"POSITION_OPEN", "INVALIDATED", "COOLDOWN"}),
    "POSITION_OPEN": frozenset({"HOLD_LOCK", "INVALIDATED", "COOLDOWN"}),
    "HOLD_LOCK":     frozenset({"INVALIDATED", "COOLDOWN"}),
    "INVALIDATED":   frozenset({"COOLDOWN"}),
    "COOLDOWN":      frozenset({"IDLE"}),
}


def is_transition_allowed(from_state: PolicyStateKind, to_state: PolicyStateKind) -> bool:
    """Return True when the transition is in the allowed graph."""
    return to_state in _ALLOWED_TRANSITIONS.get(from_state, frozenset())


# ---------------------------------------------------------------------------
# Per-state guard rules
# ---------------------------------------------------------------------------

# States that suppress strategist / judge policy evaluation
_POLICY_LOOP_SUPPRESSED_STATES: FrozenSet[PolicyStateKind] = frozenset(
    {"THESIS_ARMED", "HOLD_LOCK"}
)

# States that suppress playbook switching
_PLAYBOOK_SWITCH_SUPPRESSED_STATES: FrozenSet[PolicyStateKind] = frozenset(
    {"THESIS_ARMED", "HOLD_LOCK", "COOLDOWN"}
)

# States that suppress target re-optimization
_TARGET_REOPT_SUPPRESSED_STATES: FrozenSet[PolicyStateKind] = frozenset({"HOLD_LOCK"})


def is_policy_loop_suppressed(state: PolicyStateKind, *, safety_override: bool = False) -> bool:
    """Return True when the policy loop (strategist/judge) must not run.

    A safety_override (explicitly audited) can bypass THESIS_ARMED and HOLD_LOCK
    to create a new policy boundary.
    """
    if safety_override:
        return False
    return state in _POLICY_LOOP_SUPPRESSED_STATES


def is_playbook_switch_suppressed(
    state: PolicyStateKind,
    *,
    has_invalidation_trigger: bool = False,
    has_safety_override: bool = False,
) -> bool:
    """Return True when a playbook switch is not permitted in the current state."""
    if has_invalidation_trigger or has_safety_override:
        return False
    return state in _PLAYBOOK_SWITCH_SUPPRESSED_STATES


def is_target_reopt_suppressed(
    state: PolicyStateKind,
    *,
    has_safety_override: bool = False,
    playbook_allows_reopt: bool = False,
) -> bool:
    """Return True when target re-optimization is blocked by HOLD_LOCK."""
    if has_safety_override or playbook_allows_reopt:
        return False
    return state in _TARGET_REOPT_SUPPRESSED_STATES


# ---------------------------------------------------------------------------
# State record schemas
# ---------------------------------------------------------------------------


class PolicyStateTransition(BaseModel):
    """A single recorded state transition in the intra-policy state machine."""

    model_config = {"extra": "forbid"}

    from_state: PolicyStateKind
    to_state: PolicyStateKind
    transitioned_at: datetime
    reason: str  # human-readable trigger reason
    triggered_by: Optional[str] = None  # e.g., "activation_trigger_fired", "stop_hit"


class ActivationWindowState(BaseModel):
    """Tracks an active THESIS_ARMED window.

    Created when state enters THESIS_ARMED; consumed on exit.
    """

    model_config = {"extra": "forbid"}

    thesis_armed_at: datetime
    global_timeout_bars: int  # from CadenceConfig.policy_thesis_activation_timeout_bars
    playbook_timeout_bars: Optional[int] = None   # playbook-specific override (takes precedence)
    bars_elapsed: int = 0

    @property
    def effective_timeout_bars(self) -> int:
        """Playbook override takes precedence over global default."""
        if self.playbook_timeout_bars is not None:
            return self.playbook_timeout_bars
        return self.global_timeout_bars

    @property
    def is_timed_out(self) -> bool:
        """Return True when bars_elapsed exceeds the effective timeout."""
        return self.bars_elapsed >= self.effective_timeout_bars


class PolicyStateMachineRecord(BaseModel):
    """Full state record for the intra-policy state machine.

    Carried by the policy loop between evaluations.
    """

    model_config = {"extra": "forbid"}

    current_state: PolicyStateKind = "IDLE"
    entered_at: Optional[datetime] = None
    transitions: List[PolicyStateTransition] = Field(default_factory=list)

    # Set when state == THESIS_ARMED; cleared on exit
    activation_window: Optional[ActivationWindowState] = None

    # Set when position is live
    position_id: Optional[str] = None

    # Cooldown tracking
    cooldown_started_at: Optional[datetime] = None
    cooldown_expires_at: Optional[datetime] = None
