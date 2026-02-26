"""Deterministic structure engine schemas.

Runbook 58: Deterministic Structure Engine and Context Exposure.

Three levels of typed output:
  StructureLevel   — a single raw level with provenance + current role classification
  StructureEvent   — a deterministic structural change (break, reclaim, breakout, shift)
  LevelLadder      — ranked near/mid/far ladders of supports and resistances per timeframe
  StructureSnapshot — atomic, versioned, hashable aggregation of all structure outputs

Design rules:
  1. Raw level identity (kind, price, timeframe) is always separated from current role.
  2. All outputs are scoped by source timeframe and carry as_of timestamps.
  3. Events are deterministic, replayable, and evidence-bearing.
  4. Engine outputs are context, not direct orders — execution remains in trigger/risk layer.

Version contract:
  STRUCTURE_ENGINE_VERSION tracks the schema; bump when field shapes change.
  StructureSnapshot.snapshot_hash is computed over canonical JSON.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

STRUCTURE_ENGINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# StructureLevel
# ---------------------------------------------------------------------------

_KindLiteral = Literal[
    "prior_session_high",
    "prior_session_low",
    "prior_session_open",
    "prior_session_mid",
    "rolling_window_high",
    "rolling_window_low",
    "swing_high",
    "swing_low",
    "trendline",
    "channel_upper",
    "channel_lower",
    "measured_move_projection",
]


class StructureLevel(BaseModel):
    """A single market structure level with identity, role, and eligibility metadata.

    Fields preserve raw level identity separately from the dynamic role classification
    so that role flips over time are visible in event history rather than silent overwrites.
    """

    model_config = {"extra": "forbid"}

    # Identity — immutable after creation
    level_id: str                            # deterministic: "<symbol>|<kind>|<tf>|<price:.4f>"
    snapshot_id: str                         # parent StructureSnapshot this was emitted from
    symbol: str
    as_of_ts: datetime

    price: float
    source_timeframe: str                    # e.g. "1d", "1w", "1M", "5d"
    kind: _KindLiteral
    source_label: str                        # human-readable, deterministic (e.g. "D-1 High")
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Dynamic role relative to reference price at snapshot time
    role_now: Literal["support", "resistance", "neutral"]
    distance_abs: float                      # |price - reference_price|
    distance_pct: float                      # distance_abs / reference_price * 100
    distance_atr: Optional[float] = None     # distance_abs / atr_14 (None when ATR unavailable)

    # Strength / quality metadata
    strength_score: Optional[float] = None   # 0.0–1.0 composite or None when not computed
    touch_count: Optional[int] = None        # number of documented touches (S2+)
    last_touch_ts: Optional[datetime] = None
    age_bars: Optional[int] = None           # bars since level was established

    # Use-case eligibility flags (set by engine; consumed by R52/R56 compiler)
    eligible_for_entry_trigger: bool = False
    eligible_for_stop_anchor: bool = False
    eligible_for_target_anchor: bool = False


# ---------------------------------------------------------------------------
# StructureEvent
# ---------------------------------------------------------------------------

_EventTypeLiteral = Literal[
    "level_broken",
    "level_reclaimed",
    "liquidity_sweep_reject",
    "range_breakout",
    "range_breakdown",
    "trendline_break",
    "structure_shift",
]


class StructureEvent(BaseModel):
    """A deterministic structural change at a snapshot boundary.

    Events are emitted by comparing current structure state against a prior
    snapshot.  Each event carries enough evidence to be replayed offline and
    attributed to specific levels and indicator values.
    """

    model_config = {"extra": "forbid"}

    event_id: str                            # uuid4; stable within a snapshot
    snapshot_id: str
    symbol: str
    as_of_ts: datetime
    eval_timeframe: str                      # timeframe used for close-confirmation

    event_type: _EventTypeLiteral
    severity: Literal["low", "medium", "high"]
    level_id: Optional[str] = None           # linked StructureLevel when applicable
    level_kind: Optional[str] = None
    direction: Literal["up", "down", "neutral"] = "neutral"

    # Deterministic evidence
    price_ref: Optional[float] = None        # reference price at event time
    close_ref: Optional[float] = None        # close value used in confirmation
    threshold_ref: Optional[float] = None    # the level price that was crossed
    confirmation_rule: Optional[str] = None  # e.g. "close_below_support"
    evidence: Dict[str, Any] = Field(default_factory=dict)

    # Policy integration hints (advisory — actual cadence follows Runbook 54)
    trigger_policy_reassessment: bool = False
    trigger_activation_review: bool = False


# ---------------------------------------------------------------------------
# LevelLadder
# ---------------------------------------------------------------------------

class LevelLadder(BaseModel):
    """Ranked near/mid/far support and resistance levels for a single source timeframe.

    Near  = within 1 ATR of reference price
    Mid   = 1–3 ATR from reference price
    Far   = more than 3 ATR from reference price

    When ATR is unavailable, levels are placed into mid bucket by default.
    Within each bucket, levels are sorted by proximity (closest first).
    """

    model_config = {"extra": "forbid"}

    source_timeframe: str
    near_supports: List[StructureLevel] = Field(default_factory=list)
    mid_supports: List[StructureLevel] = Field(default_factory=list)
    far_supports: List[StructureLevel] = Field(default_factory=list)
    near_resistances: List[StructureLevel] = Field(default_factory=list)
    mid_resistances: List[StructureLevel] = Field(default_factory=list)
    far_resistances: List[StructureLevel] = Field(default_factory=list)

    @property
    def all_supports(self) -> List[StructureLevel]:
        return self.near_supports + self.mid_supports + self.far_supports

    @property
    def all_resistances(self) -> List[StructureLevel]:
        return self.near_resistances + self.mid_resistances + self.far_resistances


# ---------------------------------------------------------------------------
# StructureSnapshot
# ---------------------------------------------------------------------------

class StructureQuality(BaseModel):
    """Completeness and staleness flags for a structure snapshot."""

    model_config = {"extra": "forbid"}

    available_timeframes: List[str] = Field(default_factory=list)
    missing_timeframes: List[str] = Field(default_factory=list)
    is_partial: bool = False
    quality_warnings: List[str] = Field(default_factory=list)


class StructureSnapshot(BaseModel):
    """Atomic, versioned, immutable aggregation of all structure engine outputs.

    Follows Runbook 49 evidence-artifact rules:
      - snapshot_id: uuid4
      - snapshot_hash: sha256 over canonical JSON of content fields
      - snapshot_version: STRUCTURE_ENGINE_VERSION
      - as_of_ts / generated_at_ts: UTC market time vs builder time

    Embeddable in TickSnapshot / PolicySnapshot via snapshot_id reference.
    """

    model_config = {"extra": "forbid"}

    # Provenance
    snapshot_id: str
    snapshot_hash: str
    snapshot_version: str = STRUCTURE_ENGINE_VERSION
    symbol: str
    as_of_ts: datetime
    generated_at_ts: datetime
    source_timeframe: str                    # primary timeframe used for event detection
    data_source: str = "indicator_snapshot"  # "indicator_snapshot" | "ohlcv_df"

    # Market context at snapshot time
    reference_price: float
    reference_atr: Optional[float] = None   # atr_14 for distance_atr calculations

    # Structure content
    levels: List[StructureLevel] = Field(default_factory=list)
    ladders: Dict[str, LevelLadder] = Field(default_factory=dict)    # keyed by source_timeframe
    events: List[StructureEvent] = Field(default_factory=list)

    # Policy integration (Runbook 54 forward-compat)
    policy_trigger_reasons: List[str] = Field(default_factory=list)
    policy_event_priority: Optional[Literal["low", "medium", "high"]] = None

    # Quality
    quality: StructureQuality = Field(default_factory=StructureQuality)


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def compute_structure_snapshot_hash(content: Dict[str, Any]) -> str:
    """Compute a stable sha256 over canonical JSON of structure snapshot content.

    ``content`` should contain identity + level prices only (not the hash itself).
    """
    canonical = json.dumps(content, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
