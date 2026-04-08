"""WorldState: shared representation across Strategist, Judge, and Execution (R80).

WorldState is the single source of truth that all three agents reference.
Instead of each component working from different projections of reality,
they all read from and write to this shared object.

Layer 1 (S³): Persistent WorldState — shared, stable across cycles.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class RegimeFingerprintPoint(BaseModel):
    """A single point in the regime trajectory (fingerprint + timestamp + bar index)."""

    model_config = {"extra": "forbid"}

    fingerprint: Dict[str, float]  # numeric_vector as {feature_name: value}
    as_of_ts: datetime
    bar_index: int
    trend_state: Optional[str] = None
    vol_state: Optional[str] = None
    regime_confidence: float = 0.5


class RegimeTrajectory(BaseModel):
    """Rolling history of regime fingerprints, capturing phase information.

    Unlike a single RegimeFingerprint (point-in-time snapshot), the trajectory
    records HOW the regime got to its current state: direction_vector captures
    the trend of each dimension, velocity_scalar captures the magnitude of change,
    and stability_score captures how stable the regime has been.

    A regime moving fast toward vol_high is different from one that has been
    stable at vol_high for 20 bars — this is the phase information that was
    previously being discarded.
    """

    model_config = {"extra": "forbid"}

    snapshots: List[RegimeFingerprintPoint] = Field(default_factory=list)
    direction_vector: Dict[str, float] = Field(
        default_factory=dict,
        description="(current - oldest) / N — trend of each feature dimension",
    )
    velocity_scalar: float = Field(
        default=0.0,
        description="Magnitude of direction_vector — how fast regime is changing",
    )
    stability_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1 - stddev across snapshots. High = stable regime.",
    )
    inflection_bars_ago: Optional[int] = Field(
        default=None,
        description="Bars since last regime direction reversal, if detected",
    )
    window_size: int = Field(default=20, description="Max snapshots to retain")


class ConfidenceCalibration(BaseModel):
    """Per-dimension trust weights for plan components.

    Updated by JudgeFeedbackService via apply_judge_guidance().
    Read by plan_provider (weighting plan sections) and HypothesisCompiler
    (deciding whether to override LLM stop with structural anchor).
    """

    model_config = {"extra": "forbid"}

    regime_assessment_confidence: float = Field(default=1.0, ge=0.0, le=2.0)
    stop_placement_confidence: float = Field(default=1.0, ge=0.0, le=2.0)
    target_placement_confidence: float = Field(default=1.0, ge=0.0, le=2.0)
    entry_timing_confidence: float = Field(default=1.0, ge=0.0, le=2.0)
    hypothesis_model_confidence: float = Field(default=1.0, ge=0.0, le=2.0)
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_by: Literal[
        "judge_evaluation", "backtest_result", "init"
    ] = "init"


class StructureDigest(BaseModel):
    """Lightweight summary of current StructureSnapshot (avoids embedding the full object)."""

    model_config = {"extra": "forbid"}

    snapshot_id: Optional[str] = None
    snapshot_hash: Optional[str] = None
    symbol: Optional[str] = None
    nearest_support_pct: Optional[float] = None
    nearest_resistance_pct: Optional[float] = None
    active_level_count: int = 0
    computed_at: Optional[datetime] = None


class EpisodeDigest(BaseModel):
    """Summary stats from the last retrieved DiversifiedMemoryBundle."""

    model_config = {"extra": "forbid"}

    bundle_id: Optional[str] = None
    win_count: int = 0
    loss_count: int = 0
    dominant_failure_mode: Optional[str] = None
    avg_r_achieved: Optional[float] = None
    retrieved_at: Optional[datetime] = None


class WorldState(BaseModel):
    """Shared world model referenced by Strategist, Judge, and Execution.

    Every decision (plan generation, trigger evaluation, judge evaluation)
    should reference the same WorldState. Changes to regime, judge guidance,
    or confidence calibration flow through a single update path.

    world_state_id changes on every structural update (regime transition,
    judge guidance update, or confidence calibration change).
    """

    model_config = {"extra": "forbid"}

    world_state_id: str = Field(default_factory=lambda: str(uuid4()))
    as_of_ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Layer 1: Regime representation
    regime_fingerprint: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current normalized regime fingerprint (numeric_vector as dict)",
    )
    regime_fingerprint_meta: Optional[Dict[str, str]] = Field(
        default=None,
        description="Categorical regime states: trend_state, vol_state, structure_state",
    )
    regime_trajectory: RegimeTrajectory = Field(
        default_factory=RegimeTrajectory,
        description="Rolling history of fingerprints — captures phase information",
    )

    # Layer 4: Judge structured output (replaces textual DisplayConstraints injection)
    judge_guidance: Optional[Dict[str, object]] = Field(
        default=None,
        description="JudgeGuidanceVector as dict (avoids circular import with judge_feedback)",
    )

    # Confidence calibration per plan dimension
    confidence_calibration: ConfidenceCalibration = Field(
        default_factory=ConfidenceCalibration
    )

    # Structure digest (lightweight ref to current StructureSnapshot)
    structure_digest: StructureDigest = Field(default_factory=StructureDigest)

    # Episode digest (summary of last memory retrieval)
    episode_digest: EpisodeDigest = Field(default_factory=EpisodeDigest)

    # Policy FSM state (mirrors PolicyStateMachineRecord.state for quick access)
    policy_state: Optional[str] = Field(
        default=None,
        description="Current PolicyStateKind: IDLE/THESIS_ARMED/POSITION_OPEN/etc.",
    )

    def to_dict(self) -> Dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: Dict) -> "WorldState":
        return cls.model_validate(d)
