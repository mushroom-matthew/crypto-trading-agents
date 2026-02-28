"""Cadence configuration and telemetry schemas for the reasoning-agent stack (Runbook 54).

Centralizes three-tier scheduling rules so fast loops stay fast and slow loops
remain statistically meaningful.  All settings have env-var overrides so
operators can tune cadence without code changes.

Three-tier model:
  Layer 1 — Tick Engine (deterministic only, every bar)
  Layer 2 — Policy Loop (event-driven + heartbeat, not every tick)
  Layer 3 — Structural Learning Loop (daily / weekly / monthly)

Intra-policy state machine (within Layer 1/2):
  IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK → INVALIDATED/COOLDOWN

Policy boundary rules enforced here:
- No LLM / memory retrieval inside THESIS_ARMED activation window
- No target re-optimization inside HOLD_LOCK
- No playbook switch inside policy cooldown without invalidation/safety path
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyLoopSkipReason = Literal[
    "no_trigger_and_no_heartbeat",   # no event fired and heartbeat not expired
    "policy_frozen_thesis_armed",    # suppressed because THESIS_ARMED
    "policy_frozen_hold_lock",       # suppressed because HOLD_LOCK
    "cooldown_not_expired",          # within policy-loop min reevaluation window
    "already_running",               # single-flight: prior eval still in progress
    "operator_override_required",    # requires explicit operator audit event
]

PolicyLoopTriggerType = Literal[
    "regime_state_changed",         # deterministic regime transition detector fired
    "position_opened",              # position state transition: fill received
    "position_closed",              # position state transition: exit completed
    "vol_band_shift",               # volatility percentile band crossed threshold
    "heartbeat_expired",            # configurable heartbeat timer elapsed
    "operator_override",            # explicit audited override event
]

PreEntryInvalidationKind = Literal[
    "activation_timeout",           # activation-window bar count exceeded
    "structure_break",              # key structural level broken before entry
    "vol_shock",                    # volatility shock outside envelope
    "regime_cancel",                # regime transition cancels thesis (R55 event)
]

HighLevelReflectionSkipReason = Literal[
    "interval_not_elapsed",         # minimum hours since last run not met
    "insufficient_sample",          # episode count below minimum
    "already_running",              # single-flight guard active
    "forced_skip",                  # caller explicitly skipped
]


# ---------------------------------------------------------------------------
# Cadence configuration (loaded from environment with safe defaults)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


class CadenceConfig:
    """Immutable cadence configuration resolved from environment variables.

    Instantiate once at module load time (or pass explicitly in tests).
    All intervals are in seconds unless the name suffix says otherwise.
    """

    # --- Tick Engine ---
    tick_engine_deterministic_only: bool
    tick_validation_timeout_ms: int

    # --- Policy Loop ---
    policy_loop_enabled: bool
    policy_loop_heartbeat_1m_seconds: int    # heartbeat for 1m-timeframe systems
    policy_loop_heartbeat_5m_seconds: int    # heartbeat for 5m-timeframe systems
    policy_loop_min_reeval_seconds: int      # cooldown between policy evaluations
    policy_loop_require_event_or_heartbeat: bool

    # Intra-policy state machine
    policy_thesis_activation_timeout_bars: int   # global default activation-window length
    policy_hold_lock_enforced: bool
    policy_target_reopt_enabled: bool            # False = HOLD_LOCK blocks target re-opt
    policy_rearm_requires_next_boundary: bool

    # --- Policy-Level Reflection ---
    policy_level_reflection_enabled: bool
    policy_level_reflection_timeout_ms: int

    # --- Memory Retrieval ---
    memory_retrieval_timeout_ms: int
    memory_retrieval_required: bool             # if True, degrade rather than silently skip
    memory_retrieval_reuse_enabled: bool
    memory_requery_regime_delta_threshold: float

    # --- High-Level Reflection ---
    high_level_reflection_enabled: bool
    high_level_reflection_min_interval_hours: int
    high_level_reflection_min_episodes: int

    # --- Playbook Metadata Refresh ---
    playbook_metadata_refresh_hours: int
    playbook_metadata_drift_trigger: bool

    # --- Regime Transition Detector ---
    regime_transition_detector_enabled: bool
    regime_transition_min_confidence_delta: float
    vol_percentile_band_shift_trigger: bool

    # --- Regime Fingerprint Relearning ---
    regime_fingerprint_relearn_days: int
    regime_fingerprint_drift_threshold: float

    def __init__(self) -> None:
        self.tick_engine_deterministic_only = _env_bool("TICK_ENGINE_DETERMINISTIC_ONLY", True)
        self.tick_validation_timeout_ms = _env_int("TICK_VALIDATION_TIMEOUT_MS", 50)

        self.policy_loop_enabled = _env_bool("POLICY_LOOP_ENABLED", True)
        self.policy_loop_heartbeat_1m_seconds = _env_int("POLICY_LOOP_HEARTBEAT_1M_SECONDS", 900)
        self.policy_loop_heartbeat_5m_seconds = _env_int("POLICY_LOOP_HEARTBEAT_5M_SECONDS", 3600)
        self.policy_loop_min_reeval_seconds = _env_int("POLICY_LOOP_MIN_REEVAL_SECONDS", 900)
        self.policy_loop_require_event_or_heartbeat = _env_bool(
            "POLICY_LOOP_REQUIRE_EVENT_OR_HEARTBEAT", True
        )

        self.policy_thesis_activation_timeout_bars = _env_int(
            "POLICY_THESIS_ACTIVATION_TIMEOUT_BARS", 3
        )
        self.policy_hold_lock_enforced = _env_bool("POLICY_HOLD_LOCK_ENFORCED", True)
        self.policy_target_reopt_enabled = _env_bool("POLICY_TARGET_REOPT_ENABLED", False)
        self.policy_rearm_requires_next_boundary = _env_bool(
            "POLICY_REARM_REQUIRES_NEXT_BOUNDARY", True
        )

        self.policy_level_reflection_enabled = _env_bool("POLICY_LEVEL_REFLECTION_ENABLED", True)
        self.policy_level_reflection_timeout_ms = _env_int(
            "POLICY_LEVEL_REFLECTION_TIMEOUT_MS", 250
        )

        self.memory_retrieval_timeout_ms = _env_int("MEMORY_RETRIEVAL_TIMEOUT_MS", 150)
        self.memory_retrieval_required = _env_bool("MEMORY_RETRIEVAL_REQUIRED", True)
        self.memory_retrieval_reuse_enabled = _env_bool("MEMORY_RETRIEVAL_REUSE_ENABLED", True)
        self.memory_requery_regime_delta_threshold = _env_float(
            "MEMORY_REQUERY_REGIME_DELTA_THRESHOLD", 0.15
        )

        self.high_level_reflection_enabled = _env_bool("HIGH_LEVEL_REFLECTION_ENABLED", True)
        self.high_level_reflection_min_interval_hours = _env_int(
            "HIGH_LEVEL_REFLECTION_MIN_INTERVAL_HOURS", 24
        )
        self.high_level_reflection_min_episodes = _env_int(
            "HIGH_LEVEL_REFLECTION_MIN_EPISODES", 20
        )

        self.playbook_metadata_refresh_hours = _env_int("PLAYBOOK_METADATA_REFRESH_HOURS", 168)
        self.playbook_metadata_drift_trigger = _env_bool("PLAYBOOK_METADATA_DRIFT_TRIGGER", True)

        self.regime_transition_detector_enabled = _env_bool(
            "REGIME_TRANSITION_DETECTOR_ENABLED", True
        )
        self.regime_transition_min_confidence_delta = _env_float(
            "REGIME_TRANSITION_MIN_CONFIDENCE_DELTA", 0.20
        )
        self.vol_percentile_band_shift_trigger = _env_bool("VOL_PERCENTILE_BAND_SHIFT_TRIGGER", True)

        self.regime_fingerprint_relearn_days = _env_int("REGIME_FINGERPRINT_RELEARN_DAYS", 30)
        self.regime_fingerprint_drift_threshold = _env_float(
            "REGIME_FINGERPRINT_DRIFT_THRESHOLD", 0.30
        )

    def heartbeat_for_timeframe(self, indicator_timeframe: str) -> int:
        """Return the policy-loop heartbeat in seconds for the given indicator timeframe."""
        if indicator_timeframe in {"1m", "1min"}:
            return self.policy_loop_heartbeat_1m_seconds
        # Default to 5m/1h heartbeat for all other timeframes
        return self.policy_loop_heartbeat_5m_seconds


# Module-level singleton (tests can instantiate CadenceConfig directly)
_default_config: Optional[CadenceConfig] = None


def get_cadence_config() -> CadenceConfig:
    """Return the module-level cadence config singleton (created lazily)."""
    global _default_config
    if _default_config is None:
        _default_config = CadenceConfig()
    return _default_config


# ---------------------------------------------------------------------------
# Telemetry schemas
# ---------------------------------------------------------------------------


class PolicyLoopTriggerEvent(BaseModel):
    """A single policy-loop trigger event that may unlock a policy evaluation."""

    model_config = {"extra": "forbid"}

    trigger_type: PolicyLoopTriggerType
    fired_at: datetime
    source_detail: Optional[str] = None   # e.g. "regime=bull→bear distance=0.35"


class PolicyLoopSkipEvent(BaseModel):
    """Emitted whenever the policy loop decides NOT to evaluate this cycle.

    Every skip must be observable (Runbook 54 Operational Rule #5).
    """

    model_config = {"extra": "forbid"}

    skip_reason: PolicyLoopSkipReason
    skipped_at: datetime
    next_eligible_at: Optional[datetime] = None
    policy_state_at_skip: Optional[str] = None   # e.g. "THESIS_ARMED"
    detail: Optional[str] = None


class ActivationWindowTelemetry(BaseModel):
    """Telemetry for a resolved THESIS_ARMED activation window.

    Recorded on every exit from the activation window (fired, timeout,
    or pre-entry invalidation), required by Runbook 54 acceptance criteria.
    """

    model_config = {"extra": "forbid"}

    thesis_armed_at: datetime
    resolved_at: datetime
    outcome: Literal["activated", "timed_out", "invalidated_pre_entry"]

    # Present when outcome != "activated"
    activation_expired_reason: Optional[PreEntryInvalidationKind] = None

    # How long the thesis was armed (in execution-timeframe bars)
    armed_duration_bars: int = 0

    # Present for pre-entry invalidation
    invalidation_detail: Optional[str] = None


class CadenceTelemetrySnapshot(BaseModel):
    """Point-in-time snapshot of cadence state for each loop layer.

    Intended for ops-API telemetry endpoints and dashboard exposure.
    """

    model_config = {"extra": "forbid"}

    snapshot_at: datetime

    # Per-layer: last run time and next eligible time
    last_policy_eval_at: Optional[datetime] = None
    next_policy_eval_eligible_at: Optional[datetime] = None

    last_high_level_reflection_at: Optional[datetime] = None
    next_high_level_reflection_at: Optional[datetime] = None

    last_playbook_metadata_refresh_at: Optional[datetime] = None
    next_playbook_metadata_refresh_at: Optional[datetime] = None

    last_regime_relearn_at: Optional[datetime] = None
    next_regime_relearn_at: Optional[datetime] = None

    # Skip counts since last reset (keyed by reason)
    policy_loop_skip_counts: dict[str, int] = Field(default_factory=dict)
    high_level_reflection_skip_counts: dict[str, int] = Field(default_factory=dict)

    # Last trigger that fired the policy loop (for observability)
    last_policy_trigger_type: Optional[PolicyLoopTriggerType] = None
    last_policy_trigger_at: Optional[datetime] = None

    # Current intra-policy state
    current_policy_state: Optional[str] = None

    # Latency summaries (p50/p95 in ms) for bounded loops
    policy_reflection_latency_p50_ms: Optional[float] = None
    policy_reflection_latency_p95_ms: Optional[float] = None
    memory_retrieval_latency_p50_ms: Optional[float] = None
    memory_retrieval_latency_p95_ms: Optional[float] = None
