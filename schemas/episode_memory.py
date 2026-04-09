"""Episode memory schemas for diversified retrieval (Runbook 51).

EpisodeMemoryRecord captures the complete context of a resolved trade episode:
regime fingerprint, outcome metrics, and failure mode labels.

DiversifiedMemoryBundle is the result of a retrieval query: winning examples,
losing examples, and failure-mode patterns, quota-capped and recency-weighted.

MemoryRetrievalRequest specifies what to retrieve, with weights for each
similarity dimension.

DISCLAIMER: These schemas record research observations for strategy calibration.
They carry no position-sizing advice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Episode source constants
# ---------------------------------------------------------------------------

EpisodeSource = Literal["backtest", "paper", "live"]

# Default retrieval weight multipliers by source.
# Live episodes carry full weight; paper trading carries 70%; backtest 40%.
# These reflect decreasing fidelity of the execution environment.
DEFAULT_SOURCE_WEIGHT_MULTIPLIERS: Dict[str, float] = {
    "live": 1.0,
    "paper": 0.7,
    "backtest": 0.4,
}

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

FAILURE_MODE_TAXONOMY: List[str] = [
    "false_breakout_reversion",
    "trend_exhaustion_after_extension",
    "low_volume_breakout_failure",
    "macro_news_whipsaw",
    "signal_conflict_chop",
    "late_entry_poor_r_multiple",
    "stop_too_tight_noise_out",
]


# ---------------------------------------------------------------------------
# Core record
# ---------------------------------------------------------------------------

class EpisodeMemoryRecord(BaseModel):
    """A fully resolved trade episode with regime context and outcome labels."""

    model_config = {"extra": "forbid"}

    # identity
    episode_id: str  # uuid
    signal_id: Optional[str] = None
    trade_id: Optional[str] = None

    # timestamps
    entry_ts: Optional[datetime] = None
    exit_ts: Optional[datetime] = None
    resolution_ts: Optional[datetime] = None

    # market snapshot refs
    snapshot_id: Optional[str] = None
    snapshot_hash: Optional[str] = None

    # regime
    regime_fingerprint: Optional[Dict[str, float]] = None  # normalized features dict
    regime_version: Optional[str] = None

    # strategy metadata
    symbol: str
    timeframe: Optional[str] = None
    playbook_id: Optional[str] = None
    template_id: Optional[str] = None
    trigger_category: Optional[str] = None
    direction: Optional[Literal["long", "short"]] = None

    # outcome
    pnl: Optional[float] = None
    r_achieved: Optional[float] = None
    hold_bars: Optional[int] = None
    hold_minutes: Optional[float] = None

    # excursion
    mae: Optional[float] = None
    mfe: Optional[float] = None
    mae_pct: Optional[float] = None
    mfe_pct: Optional[float] = None

    # decision metadata
    stance: Optional[str] = None

    # labels
    outcome_class: Literal["win", "loss", "neutral"]
    failure_modes: List[str] = Field(default_factory=list)

    # execution environment — used to weight retrieval for live trade decisions
    episode_source: EpisodeSource = "live"

    # retrieval
    retrieval_scope: Optional[Literal["symbol", "cohort", "global"]] = None

    # R79: hypothesis attribution fields (optional — present when hypothesis model is active)
    hypothesis_id: Optional[str] = None
    thesis_text: Optional[str] = None
    stop_hit: Optional[bool] = None
    target_hit: Optional[bool] = None
    invalidation_hit: Optional[bool] = None
    attribution_tags: List[str] = Field(default_factory=list)
    r_multiple_realized: Optional[float] = None   # alias for r_achieved; explicit for attribution
    bars_held: Optional[int] = None               # alias for hold_bars; explicit for attribution
    expected_bars: Optional[int] = None           # LLM's estimated_bars_to_resolution
    timing_accuracy: Optional[float] = None       # expected_bars / bars_held (1.0 = perfect)
    source_session_id: Optional[str] = None       # cross-session indexing


# ---------------------------------------------------------------------------
# Retrieval metadata
# ---------------------------------------------------------------------------

class MemoryRetrievalMeta(BaseModel):
    """Metadata describing how a DiversifiedMemoryBundle was assembled."""

    model_config = {"extra": "forbid"}

    policy_event_type: Optional[str] = None
    regime_fingerprint_delta: Optional[float] = None
    bundle_reused: bool = False
    reuse_reason: Optional[str] = None
    requery_reason: Optional[str] = None
    candidate_pool_size: int = 0
    insufficient_buckets: List[str] = Field(default_factory=list)
    retrieval_latency_ms: Optional[float] = None
    retrieval_scope: Literal["symbol", "cohort", "global"] = "symbol"
    similarity_spec_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class DiversifiedMemoryBundle(BaseModel):
    """Result of a diversified memory retrieval query.

    Contains three quota-capped buckets: winning episodes, losing episodes,
    and failure-mode pattern episodes. Each bucket is recency-weighted and
    regime-similarity-scored.
    """

    model_config = {"extra": "forbid"}

    bundle_id: str  # uuid
    symbol: str
    created_at: datetime
    winning_contexts: List[EpisodeMemoryRecord] = Field(default_factory=list)
    losing_contexts: List[EpisodeMemoryRecord] = Field(default_factory=list)
    failure_mode_patterns: List[EpisodeMemoryRecord] = Field(default_factory=list)
    retrieval_meta: MemoryRetrievalMeta


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class MemoryRetrievalRequest(BaseModel):
    """Parameters for a diversified memory retrieval query."""

    model_config = {"extra": "forbid"}

    symbol: str
    regime_fingerprint: Dict[str, float]  # normalized features

    playbook_id: Optional[str] = None
    template_id: Optional[str] = None
    direction: Optional[str] = None
    timeframe: Optional[str] = None
    policy_event_type: Optional[str] = None

    # Bucket quotas
    win_quota: int = 3
    loss_quota: int = 3
    failure_mode_quota: int = 2

    # Scoring weights (must not necessarily sum to 1.0 — they are normalized internally)
    recency_decay_lambda: float = 0.01  # per-day decay
    regime_weight: float = 0.40
    playbook_weight: float = 0.30
    timeframe_weight: float = 0.15
    feature_vector_weight: float = 0.15

    # Source-fidelity multipliers — applied after all other scoring
    # (overriding with {} disables source weighting)
    source_weight_multipliers: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHT_MULTIPLIERS)
    )

    # Global fallback threshold
    global_fallback_max_fingerprint_distance: float = 0.30
