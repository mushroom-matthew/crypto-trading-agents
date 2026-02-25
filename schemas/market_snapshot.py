"""Market snapshot contracts for deterministic execution and policy-loop reasoning.

Two snapshot types:

- TickSnapshot: lightweight, built every bar/tick for the trigger engine.
  Embeds key fields from IndicatorSnapshot and basic provenance.

- PolicySnapshot: heavier, built only on policy events for strategist/judge.
  Carries numerical, derived, and optional text/visual signal blocks plus
  provenance and quality metadata.

Both types are versioned, atomic, and immutable once constructed.  The
snapshot_hash ties every reasoning decision back to the exact input state.

Runbook 49: Market Snapshot Definition.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SNAPSHOT_SCHEMA_VERSION = "1.0"
SNAPSHOT_SCHEMA_VERSION_KEY = "snapshot_schema_version"


# ---------------------------------------------------------------------------
# Feature derivation log
# ---------------------------------------------------------------------------

class FeatureDerivationEntry(BaseModel):
    """Records a single transform step in the feature pipeline."""

    model_config = {"extra": "forbid"}

    transform: str
    version: str = "1.0"
    input_window_bars: Optional[int] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    output_fields: List[str] = Field(default_factory=list)


class FeatureDerivationLog(BaseModel):
    """Ordered list of derivation steps that produced the snapshot's derived signals."""

    model_config = {"extra": "forbid"}

    entries: List[FeatureDerivationEntry] = Field(default_factory=list)
    pipeline_hash: str = ""  # sha256 of sorted canonical entries; set by builder


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class SnapshotProvenance(BaseModel):
    """Identity, lineage, and versioning metadata for a snapshot."""

    model_config = {"extra": "forbid"}

    snapshot_version: str = SNAPSHOT_SCHEMA_VERSION
    snapshot_kind: Literal["tick", "policy"]
    snapshot_id: str
    snapshot_hash: str
    feature_pipeline_hash: str
    as_of_ts: datetime                      # market state timestamp (UTC)
    generated_at_ts: datetime               # builder invocation timestamp (UTC)
    created_at_bar_id: str                  # canonical bar key used to freeze
    symbol: str
    timeframe: str

    # Optional linkage
    policy_event_id: Optional[str] = None
    parent_tick_snapshot_id: Optional[str] = None
    data_window_start_ts: Optional[datetime] = None
    data_window_end_ts: Optional[datetime] = None
    feature_pipeline_version: Optional[str] = None
    news_digest_version: Optional[str] = None
    visual_encoder_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Quality / staleness
# ---------------------------------------------------------------------------

class SnapshotQuality(BaseModel):
    """Staleness and completeness flags for a snapshot."""

    model_config = {"extra": "forbid"}

    is_stale: bool = False
    staleness_seconds: float = 0.0
    missing_sections: List[str] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal blocks
# ---------------------------------------------------------------------------

class NumericalSignalBlock(BaseModel):
    """Key numerical market values for a single symbol."""

    model_config = {"extra": "forbid"}

    close: Optional[float] = None
    volume: Optional[float] = None
    atr_14: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_bandwidth_pct_rank: Optional[float] = None
    realized_vol_proxy: Optional[float] = None   # e.g. (high-low)/close
    htf_daily_atr: Optional[float] = None


class DerivedSignalBlock(BaseModel):
    """Deterministic derived signals and regime flags for a single symbol.

    normalized_features is intentionally sparse until Runbook 55 (regime fingerprint)
    populates it.  The version key records which R55 contract was used.
    """

    model_config = {"extra": "forbid"}

    regime: Optional[str] = None
    trend_state: Optional[Literal["uptrend", "downtrend", "sideways"]] = None
    vol_state: Optional[Literal["low", "normal", "high", "extreme"]] = None
    compression_flag: Optional[bool] = None
    expansion_flag: Optional[bool] = None
    breakout_confirmed: Optional[bool] = None
    screener_anomaly_score: Optional[float] = None      # from R39 when active
    template_id: Optional[str] = None                   # suggested by R46 retrieval

    # Normalized features for cross-symbol retrieval and regime distance (R55/R51)
    normalized_features: Dict[str, float] = Field(default_factory=dict)
    normalized_features_version: Optional[str] = None  # links to R55 fingerprint contract


class TextSignalDigest(BaseModel):
    """Summarized, timestamped text context.  Raw news is NOT stored here."""

    model_config = {"extra": "forbid"}

    headline_summary: Optional[str] = None
    event_summary: Optional[str] = None
    event_ts: Optional[datetime] = None
    sentiment: Optional[Literal["bullish", "bearish", "neutral"]] = None
    impact_label: Optional[str] = None
    confidence: Optional[float] = None
    coverage_count: Optional[int] = None
    source_provenance: Optional[str] = None


class VisualSignalFingerprint(BaseModel):
    """Encoded visual / chart-pattern metadata.  Raw image bytes are NOT stored here."""

    model_config = {"extra": "forbid"}

    pattern_tags: List[str] = Field(default_factory=list)
    structural_tags: List[str] = Field(default_factory=list)
    extractor_version: Optional[str] = None
    source_candle_window: Optional[int] = None


# ---------------------------------------------------------------------------
# TickSnapshot — deterministic execution layer
# ---------------------------------------------------------------------------

class TickSnapshot(BaseModel):
    """Lightweight snapshot for the trigger engine and position manager.

    Does NOT include text/news digests, memory bundles, or other policy-only
    context.  References the source IndicatorSnapshot by symbol/timeframe/as_of
    to avoid field duplication.
    """

    model_config = {"extra": "forbid"}

    provenance: SnapshotProvenance
    quality: SnapshotQuality = Field(default_factory=SnapshotQuality)
    feature_derivation_log: FeatureDerivationLog = Field(
        default_factory=FeatureDerivationLog
    )

    # Key fields inlined for fast rule evaluation (source: IndicatorSnapshot)
    close: Optional[float] = None
    volume: Optional[float] = None
    atr_14: Optional[float] = None
    rsi_14: Optional[float] = None
    compression_flag: Optional[bool] = None
    expansion_flag: Optional[bool] = None
    breakout_confirmed: Optional[bool] = None
    trend_state: Optional[Literal["uptrend", "downtrend", "sideways"]] = None
    vol_state: Optional[Literal["low", "normal", "high", "extreme"]] = None


# ---------------------------------------------------------------------------
# PolicySnapshot — event-driven policy loop
# ---------------------------------------------------------------------------

class PolicySnapshot(BaseModel):
    """Heavier snapshot for strategist / judge invocations.

    Built only on policy events (regime change, heartbeat, position_closed, etc.)
    — not every bar.  Aggregates per-symbol numerical and derived signal blocks,
    optional text/visual context, and memory/calibration references.
    """

    model_config = {"extra": "forbid"}

    provenance: SnapshotProvenance
    quality: SnapshotQuality = Field(default_factory=SnapshotQuality)
    feature_derivation_log: FeatureDerivationLog = Field(
        default_factory=FeatureDerivationLog
    )

    # Per-symbol data
    numerical: Dict[str, NumericalSignalBlock] = Field(default_factory=dict)
    derived: Dict[str, DerivedSignalBlock] = Field(default_factory=dict)

    # Optional modalities (marked absent via quality.missing_sections when not provided)
    text_digest: Optional[TextSignalDigest] = None
    visual_fingerprint: Optional[VisualSignalFingerprint] = None

    # Memory / calibration context (populated by R48/R51)
    memory_bundle_id: Optional[str] = None
    memory_bundle_summary: Optional[str] = None
    expectation_summary: Dict[str, Any] = Field(default_factory=dict)

    # Policy-event context
    policy_event_type: Optional[str] = None
    policy_event_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Portfolio summary at snapshot time
    equity: Optional[float] = None
    cash: Optional[float] = None
    open_positions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_snapshot_hash(data: Dict[str, Any]) -> str:
    """Compute a stable sha256 over canonical JSON of snapshot data fields.

    ``data`` should contain the content fields only (not the hash itself).
    Keys are sorted recursively for determinism.
    """
    canonical = json.dumps(data, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
