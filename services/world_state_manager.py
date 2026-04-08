"""WorldState manager: CRUD operations for the shared world model (R80).

Provides update_regime(), apply_judge_guidance(), tick_trajectory(),
and get_snapshot() — the single update path for WorldState.

All mutations return a new WorldState (immutable update pattern,
consistent with PolicyStateMachine).
"""
from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from schemas.world_state import (
    ConfidenceCalibration,
    EpisodeDigest,
    RegimeFingerprintPoint,
    RegimeTrajectory,
    StructureDigest,
    WorldState,
)

logger = logging.getLogger(__name__)

_TRAJECTORY_WINDOW = 20  # rolling deque size


def _compute_trajectory(
    snapshots: List[RegimeFingerprintPoint],
) -> tuple[Dict[str, float], float, float]:
    """Compute (direction_vector, velocity_scalar, stability_score) from snapshot list."""
    if len(snapshots) < 2:
        return {}, 0.0, 1.0

    oldest = snapshots[0].fingerprint
    newest = snapshots[-1].fingerprint
    n = len(snapshots)

    # Direction vector: (newest - oldest) / n
    all_keys = set(oldest.keys()) | set(newest.keys())
    direction: Dict[str, float] = {}
    for k in all_keys:
        direction[k] = (newest.get(k, 0.0) - oldest.get(k, 0.0)) / n

    # Velocity: L2 norm of direction vector
    velocity = math.sqrt(sum(v ** 2 for v in direction.values()))

    # Stability: 1 - mean(stddev per dimension)
    stddevs = []
    for k in all_keys:
        vals = [s.fingerprint.get(k, 0.0) for s in snapshots]
        try:
            stddevs.append(statistics.stdev(vals))
        except statistics.StatisticsError:
            stddevs.append(0.0)
    mean_stddev = statistics.mean(stddevs) if stddevs else 0.0
    stability = max(0.0, min(1.0, 1.0 - mean_stddev))

    return direction, velocity, stability


def update_regime(
    world_state: WorldState,
    fingerprint: Dict[str, float],
    *,
    trend_state: Optional[str] = None,
    vol_state: Optional[str] = None,
    structure_state: Optional[str] = None,
    regime_confidence: float = 0.5,
    bar_index: int = 0,
    as_of_ts: Optional[datetime] = None,
) -> WorldState:
    """Update WorldState with new regime fingerprint. Returns new WorldState.

    Called on every regime fingerprint update (not just transitions).
    Updates both the current fingerprint and the rolling trajectory.
    """
    now = as_of_ts or datetime.now(timezone.utc)

    # Add point to trajectory
    point = RegimeFingerprintPoint(
        fingerprint=fingerprint,
        as_of_ts=now,
        bar_index=bar_index,
        trend_state=trend_state,
        vol_state=vol_state,
        regime_confidence=regime_confidence,
    )
    new_snapshots = list(world_state.regime_trajectory.snapshots) + [point]
    # Trim to window
    window = world_state.regime_trajectory.window_size
    if len(new_snapshots) > window:
        new_snapshots = new_snapshots[-window:]

    direction, velocity, stability = _compute_trajectory(new_snapshots)

    new_trajectory = RegimeTrajectory(
        snapshots=new_snapshots,
        direction_vector=direction,
        velocity_scalar=velocity,
        stability_score=stability,
        window_size=window,
    )

    # Build meta
    meta: Dict[str, str] = {}
    if trend_state:
        meta["trend_state"] = trend_state
    if vol_state:
        meta["vol_state"] = vol_state
    if structure_state:
        meta["structure_state"] = structure_state

    return world_state.model_copy(update={
        "world_state_id": str(uuid4()),
        "as_of_ts": now,
        "regime_fingerprint": fingerprint,
        "regime_fingerprint_meta": meta or world_state.regime_fingerprint_meta,
        "regime_trajectory": new_trajectory,
    })


def apply_judge_guidance(
    world_state: WorldState,
    guidance_dict: Dict[str, Any],
) -> WorldState:
    """Apply JudgeGuidanceVector to WorldState. Returns new WorldState.

    guidance_dict is the JudgeGuidanceVector as a dict (avoids circular import).
    Also updates ConfidenceCalibration from guidance confidence_adjustments.
    """
    now = datetime.now(timezone.utc)

    # Update confidence calibration from guidance
    confidence_adjustments = guidance_dict.get("confidence_adjustments", {})
    calibration = world_state.confidence_calibration
    calibration_updates: Dict[str, Any] = {"last_updated_at": now, "updated_by": "judge_evaluation"}
    field_map = {
        "regime_assessment": "regime_assessment_confidence",
        "stop_placement": "stop_placement_confidence",
        "target_placement": "target_placement_confidence",
        "entry_timing": "entry_timing_confidence",
        "hypothesis_model": "hypothesis_model_confidence",
    }
    for key, field_name in field_map.items():
        if key in confidence_adjustments:
            calibration_updates[field_name] = float(confidence_adjustments[key])

    new_calibration = calibration.model_copy(update=calibration_updates)

    return world_state.model_copy(update={
        "world_state_id": str(uuid4()),
        "as_of_ts": now,
        "judge_guidance": guidance_dict,
        "confidence_calibration": new_calibration,
    })


def update_structure_digest(
    world_state: WorldState,
    *,
    snapshot_id: Optional[str] = None,
    snapshot_hash: Optional[str] = None,
    symbol: Optional[str] = None,
    nearest_support_pct: Optional[float] = None,
    nearest_resistance_pct: Optional[float] = None,
    active_level_count: int = 0,
) -> WorldState:
    """Update the StructureDigest component of WorldState."""
    new_digest = StructureDigest(
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        symbol=symbol,
        nearest_support_pct=nearest_support_pct,
        nearest_resistance_pct=nearest_resistance_pct,
        active_level_count=active_level_count,
        computed_at=datetime.now(timezone.utc),
    )
    return world_state.model_copy(update={
        "world_state_id": str(uuid4()),
        "as_of_ts": datetime.now(timezone.utc),
        "structure_digest": new_digest,
    })


def update_episode_digest(
    world_state: WorldState,
    bundle_id: Optional[str],
    win_count: int,
    loss_count: int,
    dominant_failure_mode: Optional[str] = None,
    avg_r_achieved: Optional[float] = None,
) -> WorldState:
    """Update the EpisodeDigest component of WorldState."""
    new_digest = EpisodeDigest(
        bundle_id=bundle_id,
        win_count=win_count,
        loss_count=loss_count,
        dominant_failure_mode=dominant_failure_mode,
        avg_r_achieved=avg_r_achieved,
        retrieved_at=datetime.now(timezone.utc),
    )
    return world_state.model_copy(update={
        "as_of_ts": datetime.now(timezone.utc),
        "episode_digest": new_digest,
    })


def update_policy_state(
    world_state: WorldState,
    policy_state: str,
) -> WorldState:
    """Update the policy_state field from PolicyStateMachineRecord."""
    return world_state.model_copy(update={
        "as_of_ts": datetime.now(timezone.utc),
        "policy_state": policy_state,
    })


def get_risk_multiplier(world_state: Optional[WorldState]) -> float:
    """Extract the current judge risk_multiplier from WorldState (default: 1.0)."""
    if world_state is None:
        return 1.0
    guidance = world_state.judge_guidance
    if not guidance:
        return 1.0
    return float(guidance.get("risk_multiplier", 1.0))


def get_playbook_penalties(world_state: Optional[WorldState]) -> Dict[str, float]:
    """Extract playbook penalty weights from WorldState judge_guidance."""
    if world_state is None:
        return {}
    guidance = world_state.judge_guidance
    if not guidance:
        return {}
    return dict(guidance.get("playbook_penalties", {}))


def get_playbook_bonuses(world_state: Optional[WorldState]) -> Dict[str, float]:
    """Extract playbook bonus weights from WorldState judge_guidance."""
    if world_state is None:
        return {}
    guidance = world_state.judge_guidance
    if not guidance:
        return {}
    return dict(guidance.get("playbook_bonuses", {}))


def get_symbol_vetoes(world_state: Optional[WorldState]) -> List[str]:
    """Extract symbol vetoes from WorldState judge_guidance."""
    if world_state is None:
        return []
    guidance = world_state.judge_guidance
    if not guidance:
        return []
    return list(guidance.get("symbol_vetoes", []))
