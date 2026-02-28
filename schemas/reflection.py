"""Reflection schemas for dual-level reflection framework (Runbook 50).

Three distinct reflection layers:

1. TickValidationResult — deterministic-only, every tick/bar, NO LLM.
2. PolicyLevelReflectionResult — event-driven policy loop, after strategist
   proposal, before policy freeze.  Optional LLM consolidation, bounded.
3. HighLevelReflectionReport — scheduled slow-loop, batch episode analysis,
   gated by sample-size and minimum elapsed interval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Invocation metadata (shared across all reflection layers)
# ---------------------------------------------------------------------------


class ReflectionInvocationMeta(BaseModel):
    """Records how and why a reflection pass was invoked or skipped."""

    model_config = {"extra": "forbid"}

    invoked_at: datetime
    source: str  # e.g. "plan_provider", "judge_eval", "scheduler"
    policy_event_type: Optional[str] = None  # e.g. "regime_change", "position_opened"
    skip_reason: Optional[str] = None  # non-None means this was a skip record
    latency_ms: Optional[int] = None
    reflection_kind: Literal["tick", "policy", "high_level"]


# ---------------------------------------------------------------------------
# Tick-level deterministic validation (NO LLM — enforcement only)
# ---------------------------------------------------------------------------


class TickValidationResult(BaseModel):
    """Deterministic per-tick/bar validation outcome.

    Captures stop/target/time/risk-envelope breach flags and any position
    state transitions that occurred.  No LLM calls, no memory retrieval.
    """

    model_config = {"extra": "forbid"}

    tick_ts: datetime
    symbol: str
    timeframe: str

    # Breach flags
    stop_breach: bool = False
    target_breach: bool = False
    time_stop_breach: bool = False
    risk_envelope_breach: bool = False

    # State transition info
    position_state_before: Optional[str] = None  # e.g. "THESIS_ARMED", "POSITION_OPEN"
    position_state_after: Optional[str] = None

    # Supplementary context (prices, bar index)
    close_price: Optional[float] = None
    active_stop_price: Optional[float] = None
    active_target_price: Optional[float] = None
    bar_index: Optional[int] = None

    # Actions triggered by deterministic validation
    actions_taken: List[str] = Field(default_factory=list)  # e.g. ["stop_exit_triggered"]

    meta: ReflectionInvocationMeta


# ---------------------------------------------------------------------------
# Policy-level reflection (fast path, event-driven)
# ---------------------------------------------------------------------------


class PolicyLevelReflectionRequest(BaseModel):
    """Inputs for a fast policy-boundary reflection check.

    Must be assembled from already-available in-memory objects (PolicySnapshot,
    StrategyPlan, memory bundle) — no I/O may happen inside the reflection
    service itself.
    """

    model_config = {"extra": "forbid"}

    # Snapshot (R49)
    snapshot_id: Optional[str] = None
    snapshot_hash: Optional[str] = None

    # Proposed plan fields (summary — not the full plan to keep payload small)
    plan_id: Optional[str] = None
    playbook_id: Optional[str] = None
    template_id: Optional[str] = None
    direction_summary: Optional[str] = None  # e.g. "long", "short", "mixed"
    trigger_count: int = 0
    allowed_directions: List[str] = Field(default_factory=list)
    regime: Optional[str] = None
    rationale_excerpt: Optional[str] = None  # first 400 chars of rationale

    # Policy state context
    policy_state: Optional[str] = None  # e.g. "THESIS_ARMED", "POSITION_OPEN", "HOLD_LOCK"
    is_activation_window_tick: bool = False
    is_hold_lock_tick: bool = False

    # Memory summary (R51) — pre-fetched, bounded
    memory_failure_modes: List[str] = Field(default_factory=list)
    memory_winning_count: int = 0
    memory_losing_count: int = 0
    memory_bundle_id: Optional[str] = None

    # Invariant context
    risk_constraints_present: bool = False
    disabled_trigger_ids: List[str] = Field(default_factory=list)
    disabled_categories: List[str] = Field(default_factory=list)
    kill_switch_active: bool = False

    # Playbook expectation context (from R52)
    playbook_expected_hold_bars_p50: Optional[float] = None
    playbook_mae_budget_pct: Optional[float] = None
    stated_conviction: Optional[str] = None  # e.g. "high", "medium", "low"

    meta: ReflectionInvocationMeta


class PolicyLevelReflectionResult(BaseModel):
    """Typed output of a policy-level reflection check.

    status:
      - "pass"   → proposal is coherent and consistent with invariants
      - "revise" → specific issues found; requested_revisions lists what to fix
      - "block"  → hard invariant violation; proposal must not proceed as-is
    """

    model_config = {"extra": "forbid"}

    status: Literal["pass", "revise", "block"]
    coherence_findings: List[str] = Field(default_factory=list)
    invariant_findings: List[str] = Field(default_factory=list)
    memory_findings: List[str] = Field(default_factory=list)
    expectation_findings: List[str] = Field(default_factory=list)
    requested_revisions: List[str] = Field(default_factory=list)
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# High-level reflection (slow path, scheduled batch)
# ---------------------------------------------------------------------------


class HighLevelReflectionRequest(BaseModel):
    """Parameters for a scheduled high-level batch reflection run."""

    model_config = {"extra": "forbid"}

    window_start: datetime
    window_end: datetime

    # Episodes to analyze (list of episode IDs; actual records fetched by service)
    episode_ids: List[str] = Field(default_factory=list)

    # Optional filter dimensions
    symbol_filter: Optional[str] = None
    playbook_id_filter: Optional[str] = None
    regime_filter: Optional[str] = None

    # Minimum sample gates
    min_episodes_for_structural_recommendation: int = 20
    min_regime_cluster_samples: int = 10

    # Scheduling context
    scheduled_cadence: Literal["daily", "weekly", "on_demand"] = "daily"
    force_run: bool = False  # bypass time-gate for on_demand / test runs

    meta: ReflectionInvocationMeta


class RegimeClusterSummary(BaseModel):
    """Outcome statistics for one regime/playbook cluster."""

    model_config = {"extra": "forbid"}

    cluster_key: str  # e.g. "playbook=bollinger_squeeze|regime=range_bound"
    n_episodes: int
    win_rate: float  # 0.0–1.0
    avg_r_achieved: Optional[float] = None
    avg_hold_bars: Optional[float] = None
    dominant_failure_modes: List[str] = Field(default_factory=list)
    expectancy_z_score: Optional[float] = None  # deviation from playbook prior


class PlaybookFinding(BaseModel):
    """Findings about a specific playbook's performance in the window."""

    model_config = {"extra": "forbid"}

    playbook_id: str
    n_episodes: int
    win_rate: float
    avg_r_achieved: Optional[float] = None
    hold_time_deviation_pct: Optional[float] = None  # actual vs expected P50
    mae_drift: Optional[float] = None
    mfe_drift: Optional[float] = None
    dominant_failure_modes: List[str] = Field(default_factory=list)
    recommended_action: Literal[
        "hold", "replan", "policy_adjust", "research_experiment"
    ] = "hold"
    structural_change_eligible: bool = False  # True only when sample-size gate passes
    insufficient_sample_reason: Optional[str] = None


class HighLevelReflectionReport(BaseModel):
    """Output of a scheduled high-level batch reflection run.

    Structural recommendations (playbook eligibility/threshold changes) are
    only emitted when sample_size and significance gates pass.  If they fail,
    ``insufficient_sample`` is set and findings are monitor-only.
    """

    model_config = {"extra": "forbid"}

    window_start: datetime
    window_end: datetime
    n_episodes: int

    regime_cluster_summary: List[RegimeClusterSummary] = Field(default_factory=list)
    playbook_findings: List[PlaybookFinding] = Field(default_factory=list)
    drift_findings: List[str] = Field(default_factory=list)
    recommendations: List[Dict] = Field(default_factory=list)  # typed downstream
    evidence_refs: List[str] = Field(default_factory=list)  # episode IDs, signal IDs

    # Gating results
    insufficient_sample: bool = False
    insufficient_sample_reason: Optional[str] = None
    structural_recommendations_suppressed: bool = False

    meta: ReflectionInvocationMeta
