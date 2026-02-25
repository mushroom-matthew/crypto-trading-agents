"""Regime fingerprint and transition detector schemas (Runbook 55).

Key design contracts:
- RegimeFingerprint contains ONLY normalized/percentile/z-scored numeric features.
  Raw price levels, raw ATR values, raw volume, raw RSI are FORBIDDEN in the
  numeric_vector (they corrupt cohort/global comparability).
- numeric_vector_feature_names is version-locked by FINGERPRINT_VERSION.
- distance_value is bounded [0, 1] with decomposable per-component contributions.
- All Pydantic models use extra='forbid'.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FINGERPRINT_SCHEMA_VERSION = "1.0"
FINGERPRINT_VERSION = "1.0.0"

# Fixed numeric vector feature order — version-locked with FINGERPRINT_VERSION.
# Changing this list requires bumping FINGERPRINT_VERSION and FINGERPRINT_SCHEMA_VERSION.
NUMERIC_VECTOR_FEATURE_NAMES_V1: List[str] = [
    "vol_percentile",
    "atr_percentile",
    "volume_percentile",
    "range_expansion_percentile",
    "realized_vol_z_normed",        # (realized_vol_z + 3) / 6, clamped [0, 1]
    "distance_to_htf_anchor_normed",  # clamp(distance_to_htf_anchor_atr / 5.0, 0, 1)
]


class RegimeFingerprint(BaseModel):
    """Normalized regime fingerprint for a symbol at a point in time.

    CONSTRAINT: numeric_vector must contain ONLY normalized components in [0, 1].
    Raw symbol-scale values (price, ATR, volume) are forbidden in the vector.
    Named scalar fields (realized_vol_z, distance_to_htf_anchor_atr) carry the
    raw values for diagnostics; their normed counterparts go into numeric_vector.
    """

    model_config = {"extra": "forbid"}

    fingerprint_version: str = FINGERPRINT_VERSION
    schema_version: str = FINGERPRINT_SCHEMA_VERSION

    symbol: str
    scope: Literal["symbol", "cohort"]
    cohort_id: Optional[str] = None

    as_of_ts: datetime
    bar_id: str
    source_timeframe: str  # timeframe used for this evaluation (HTF by default)

    # Stable categorical states (fixed vocabulary — R55 canonical)
    trend_state: Literal["up", "down", "sideways"]
    vol_state: Literal["low", "mid", "high", "extreme"]
    structure_state: Literal[
        "compression",
        "expansion",
        "mean_reverting",
        "breakout_active",
        "breakdown_active",
        "neutral",
    ]

    # Confidence values (normalized [0, 1])
    trend_confidence: float = Field(ge=0.0, le=1.0)
    vol_confidence: float = Field(ge=0.0, le=1.0)
    structure_confidence: float = Field(ge=0.0, le=1.0)
    regime_confidence: float = Field(ge=0.0, le=1.0)

    # Normalized numeric components (all [0, 1])
    vol_percentile: float = Field(ge=0.0, le=1.0)
    atr_percentile: float = Field(ge=0.0, le=1.0)
    volume_percentile: float = Field(ge=0.0, le=1.0)
    range_expansion_percentile: float = Field(ge=0.0, le=1.0)

    # Raw z-scores (diagnostic fields, NOT in numeric_vector)
    realized_vol_z: float
    distance_to_htf_anchor_atr: float  # ATR multiples from HTF anchor level

    trend_strength_z: Optional[float] = None  # optional — ADX-derived if available

    # Version-locked distance vector (all components clamped to [0, 1])
    numeric_vector: List[float]
    numeric_vector_feature_names: List[str]


class RegimeDistanceResult(BaseModel):
    """Output of regime_fingerprint_distance(curr, prev).

    distance_value is bounded [0, 1].
    component_contributions sum (approximately) to distance_value.
    All contributing weights are emitted for full telemetry traceability.
    """

    model_config = {"extra": "forbid"}

    distance_value: float = Field(ge=0.0, le=1.0)
    threshold_enter: float
    threshold_exit: float
    threshold_used: float
    threshold_type: Literal["enter", "exit"]
    component_contributions: Dict[str, float]
    component_deltas: Dict[str, Any]
    weights: Dict[str, float]
    confidence_delta: float
    suppressed_by_hysteresis: bool = False
    suppressed_by_cooldown: bool = False
    suppressed_by_min_dwell: bool = False


class RegimeTransitionDecision(BaseModel):
    """Output of one detector evaluation pass.

    transition_fired=True means a regime change was detected and the policy
    loop should reevaluate. suppressed=True means a potential change was detected
    but was held back by hysteresis/dwell/cooldown controls.
    """

    model_config = {"extra": "forbid"}

    transition_fired: bool
    reason_code: Literal[
        "distance_enter_threshold_crossed",
        "distance_below_threshold",
        "suppressed_hysteresis",
        "suppressed_min_dwell",
        "suppressed_cooldown",
        "shock_override_volatility_jump",
        "htf_gate_not_ready",
        "missing_required_features",
        "initial_state",
    ]
    prior_fingerprint: Optional[RegimeFingerprint] = None
    new_fingerprint: RegimeFingerprint
    distance_result: Optional[RegimeDistanceResult] = None
    suppressed: bool = False
    shock_override_used: bool = False
    htf_gate_eligible: bool = True
    as_of_ts: datetime
    symbol: str
    scope: Literal["symbol", "cohort"]


class RegimeTransitionTelemetryEvent(BaseModel):
    """Full telemetry payload for one detector evaluation (fired or not).

    Every evaluation emits one of these, regardless of whether a transition fired.
    This is non-negotiable: if you cannot explain why a transition fired (or did not
    fire), the detector is not operationally acceptable.
    """

    model_config = {"extra": "forbid"}

    event_id: str
    emitted_at: datetime
    decision: RegimeTransitionDecision
    dwell_seconds: Optional[float] = None
    cooldown_remaining_seconds: Optional[float] = None
    consecutive_stable_evals: int = 0
    detector_state_version: str = "1.0"


class RegimeTransitionDetectorState(BaseModel):
    """Serializable mutable state for RegimeTransitionDetector (one per symbol/scope).

    All timestamps are UTC-aware. None values indicate the detector has not yet
    established its initial state.
    """

    model_config = {"extra": "forbid"}

    symbol: str
    scope: Literal["symbol", "cohort"]
    current_fingerprint: Optional[RegimeFingerprint] = None
    last_transition_ts: Optional[datetime] = None
    current_regime_entered_ts: Optional[datetime] = None
    cooldown_until_ts: Optional[datetime] = None
    consecutive_stable_evals: int = 0
    total_transitions: int = 0
    detector_state_version: str = "1.0"
