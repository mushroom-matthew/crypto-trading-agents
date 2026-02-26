"""Ops API router for structure engine inspection.

Runbook 58: Deterministic Structure Engine and Context Exposure.

Endpoints expose:
  - On-demand structure snapshot computation from indicator data
  - Multi-timeframe level ladder view
  - Structural events timeline
  - Paper-trading session structure snapshot lookup (when sessions are running)

All endpoints are read-only analytics — no state mutation.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from schemas.llm_strategist import IndicatorSnapshot
from schemas.structure_engine import (
    STRUCTURE_ENGINE_VERSION,
    StructureEvent,
    StructureLevel,
    StructureSnapshot,
)
from services.structure_engine import (
    build_structure_snapshot,
    get_entry_candidates,
    get_stop_candidates,
    get_target_candidates,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/structure", tags=["structure"])

# In-memory snapshot store: {symbol: StructureSnapshot}
# Populated by paper-trading workflow or external callers via POST /structure/snapshots
_snapshot_store: Dict[str, StructureSnapshot] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StructureLevelResponse(BaseModel):
    level_id: str
    symbol: str
    as_of_ts: str
    price: float
    source_timeframe: str
    kind: str
    source_label: str
    role_now: str
    distance_abs: float
    distance_pct: float
    distance_atr: Optional[float] = None
    strength_score: Optional[float] = None
    eligible_for_entry_trigger: bool = False
    eligible_for_stop_anchor: bool = False
    eligible_for_target_anchor: bool = False


class StructureEventResponse(BaseModel):
    event_id: str
    symbol: str
    as_of_ts: str
    eval_timeframe: str
    event_type: str
    severity: str
    level_id: Optional[str] = None
    level_kind: Optional[str] = None
    direction: str
    price_ref: Optional[float] = None
    close_ref: Optional[float] = None
    threshold_ref: Optional[float] = None
    confirmation_rule: Optional[str] = None
    trigger_policy_reassessment: bool = False
    trigger_activation_review: bool = False
    evidence: Dict[str, Any] = Field(default_factory=dict)


class LevelLadderResponse(BaseModel):
    source_timeframe: str
    near_supports: List[StructureLevelResponse] = Field(default_factory=list)
    mid_supports: List[StructureLevelResponse] = Field(default_factory=list)
    far_supports: List[StructureLevelResponse] = Field(default_factory=list)
    near_resistances: List[StructureLevelResponse] = Field(default_factory=list)
    mid_resistances: List[StructureLevelResponse] = Field(default_factory=list)
    far_resistances: List[StructureLevelResponse] = Field(default_factory=list)


class StructureSnapshotSummary(BaseModel):
    snapshot_id: str
    snapshot_hash: str
    snapshot_version: str
    symbol: str
    as_of_ts: str
    generated_at_ts: str
    source_timeframe: str
    reference_price: float
    reference_atr: Optional[float] = None
    level_count: int
    event_count: int
    policy_trigger_reasons: List[str] = Field(default_factory=list)
    policy_event_priority: Optional[str] = None
    available_timeframes: List[str] = Field(default_factory=list)
    missing_timeframes: List[str] = Field(default_factory=list)
    is_partial: bool = False
    quality_warnings: List[str] = Field(default_factory=list)


class ComputeStructureRequest(BaseModel):
    """Request body for on-demand structure snapshot computation."""
    symbol: str
    timeframe: str = "1h"
    close: float
    atr_14: Optional[float] = None
    htf_daily_high: Optional[float] = None
    htf_daily_low: Optional[float] = None
    htf_daily_open: Optional[float] = None
    htf_prev_daily_high: Optional[float] = None
    htf_prev_daily_low: Optional[float] = None
    htf_5d_high: Optional[float] = None
    htf_5d_low: Optional[float] = None
    as_of: Optional[str] = None   # ISO8601 UTC; defaults to now


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_to_response(level: StructureLevel) -> StructureLevelResponse:
    return StructureLevelResponse(
        level_id=level.level_id,
        symbol=level.symbol,
        as_of_ts=level.as_of_ts.isoformat(),
        price=level.price,
        source_timeframe=level.source_timeframe,
        kind=level.kind,
        source_label=level.source_label,
        role_now=level.role_now,
        distance_abs=level.distance_abs,
        distance_pct=level.distance_pct,
        distance_atr=level.distance_atr,
        strength_score=level.strength_score,
        eligible_for_entry_trigger=level.eligible_for_entry_trigger,
        eligible_for_stop_anchor=level.eligible_for_stop_anchor,
        eligible_for_target_anchor=level.eligible_for_target_anchor,
    )


def _event_to_response(event: StructureEvent) -> StructureEventResponse:
    return StructureEventResponse(
        event_id=event.event_id,
        symbol=event.symbol,
        as_of_ts=event.as_of_ts.isoformat(),
        eval_timeframe=event.eval_timeframe,
        event_type=event.event_type,
        severity=event.severity,
        level_id=event.level_id,
        level_kind=event.level_kind,
        direction=event.direction,
        price_ref=event.price_ref,
        close_ref=event.close_ref,
        threshold_ref=event.threshold_ref,
        confirmation_rule=event.confirmation_rule,
        trigger_policy_reassessment=event.trigger_policy_reassessment,
        trigger_activation_review=event.trigger_activation_review,
        evidence=event.evidence,
    )


def _snapshot_to_summary(snap: StructureSnapshot) -> StructureSnapshotSummary:
    return StructureSnapshotSummary(
        snapshot_id=snap.snapshot_id,
        snapshot_hash=snap.snapshot_hash,
        snapshot_version=snap.snapshot_version,
        symbol=snap.symbol,
        as_of_ts=snap.as_of_ts.isoformat(),
        generated_at_ts=snap.generated_at_ts.isoformat(),
        source_timeframe=snap.source_timeframe,
        reference_price=snap.reference_price,
        reference_atr=snap.reference_atr,
        level_count=len(snap.levels),
        event_count=len(snap.events),
        policy_trigger_reasons=snap.policy_trigger_reasons,
        policy_event_priority=snap.policy_event_priority,
        available_timeframes=snap.quality.available_timeframes,
        missing_timeframes=snap.quality.missing_timeframes,
        is_partial=snap.quality.is_partial,
        quality_warnings=snap.quality.quality_warnings,
    )


def _build_indicator_from_request(req: ComputeStructureRequest) -> IndicatorSnapshot:
    """Construct a minimal IndicatorSnapshot from the request body."""
    as_of: datetime
    if req.as_of:
        try:
            as_of = datetime.fromisoformat(req.as_of)
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
        except ValueError:
            as_of = datetime.now(timezone.utc)
    else:
        as_of = datetime.now(timezone.utc)

    return IndicatorSnapshot(
        symbol=req.symbol,
        timeframe=req.timeframe,
        as_of=as_of,
        close=req.close,
        atr_14=req.atr_14,
        htf_daily_high=req.htf_daily_high,
        htf_daily_low=req.htf_daily_low,
        htf_daily_open=req.htf_daily_open,
        htf_prev_daily_high=req.htf_prev_daily_high,
        htf_prev_daily_low=req.htf_prev_daily_low,
        htf_5d_high=req.htf_5d_high,
        htf_5d_low=req.htf_5d_low,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/version")
async def structure_engine_version() -> Dict[str, str]:
    """Return the current structure engine schema version."""
    return {"version": STRUCTURE_ENGINE_VERSION}


@router.post("/snapshots/{symbol}/compute", response_model=StructureSnapshotSummary)
async def compute_structure_for_symbol(
    symbol: str,
    req: ComputeStructureRequest,
) -> StructureSnapshotSummary:
    """Compute a structure snapshot on-demand from indicator data.

    Builds a StructureSnapshot using S1 anchor levels from the provided
    htf_* fields, caches it in memory, and returns a summary.
    The prior snapshot for this symbol (if any) is used for event detection.

    POST body: ComputeStructureRequest (JSON)
    """
    if req.symbol != symbol:
        req = req.model_copy(update={"symbol": symbol})

    indicator = _build_indicator_from_request(req)
    prior_snap = _snapshot_store.get(symbol)

    try:
        snap = build_structure_snapshot(indicator, prior_snapshot=prior_snap)
    except Exception as exc:
        logger.error("Structure engine computation failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=f"Structure computation failed: {exc}") from exc

    _snapshot_store[symbol] = snap
    return _snapshot_to_summary(snap)


@router.get("/snapshots/{symbol}/latest", response_model=StructureSnapshotSummary)
async def get_latest_snapshot(symbol: str) -> StructureSnapshotSummary:
    """Get the most recently computed structure snapshot summary for a symbol.

    Returns 404 if no snapshot has been computed for this symbol.
    Use POST /structure/snapshots/{symbol}/compute to create one.
    """
    snap = _snapshot_store.get(symbol)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No structure snapshot found for {symbol}")
    return _snapshot_to_summary(snap)


@router.get("/snapshots/{symbol}/levels", response_model=List[StructureLevelResponse])
async def get_snapshot_levels(
    symbol: str,
    role: Optional[str] = Query(default=None, description="Filter by role: support|resistance|neutral"),
    timeframe: Optional[str] = Query(default=None, description="Filter by source_timeframe"),
    eligible_for: Optional[str] = Query(
        default=None,
        description="Filter: entry_trigger|stop_anchor|target_anchor",
    ),
) -> List[StructureLevelResponse]:
    """List all structure levels for a symbol with optional filters.

    Levels are sorted by distance from reference price (nearest first).
    """
    snap = _snapshot_store.get(symbol)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No structure snapshot found for {symbol}")

    levels = snap.levels
    if role:
        levels = [l for l in levels if l.role_now == role]
    if timeframe:
        levels = [l for l in levels if l.source_timeframe == timeframe]
    if eligible_for == "entry_trigger":
        levels = [l for l in levels if l.eligible_for_entry_trigger]
    elif eligible_for == "stop_anchor":
        levels = [l for l in levels if l.eligible_for_stop_anchor]
    elif eligible_for == "target_anchor":
        levels = [l for l in levels if l.eligible_for_target_anchor]

    levels = sorted(levels, key=lambda l: l.distance_abs)
    return [_level_to_response(l) for l in levels]


@router.get("/snapshots/{symbol}/ladder", response_model=Dict[str, LevelLadderResponse])
async def get_level_ladder(
    symbol: str,
    timeframe: Optional[str] = Query(default=None, description="Filter to a single source timeframe"),
) -> Dict[str, LevelLadderResponse]:
    """Return the multi-timeframe level ladder for a symbol.

    Response is keyed by source_timeframe (e.g. "1d", "5d", "1h").
    When ``timeframe`` is provided, only that timeframe's ladder is returned.
    """
    snap = _snapshot_store.get(symbol)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No structure snapshot found for {symbol}")

    ladders = snap.ladders
    if timeframe:
        if timeframe not in ladders:
            raise HTTPException(
                status_code=404,
                detail=f"No ladder for timeframe {timeframe} in snapshot for {symbol}",
            )
        ladders = {timeframe: ladders[timeframe]}

    result: Dict[str, LevelLadderResponse] = {}
    for tf, ladder in ladders.items():
        result[tf] = LevelLadderResponse(
            source_timeframe=tf,
            near_supports=[_level_to_response(l) for l in ladder.near_supports],
            mid_supports=[_level_to_response(l) for l in ladder.mid_supports],
            far_supports=[_level_to_response(l) for l in ladder.far_supports],
            near_resistances=[_level_to_response(l) for l in ladder.near_resistances],
            mid_resistances=[_level_to_response(l) for l in ladder.mid_resistances],
            far_resistances=[_level_to_response(l) for l in ladder.far_resistances],
        )
    return result


@router.get("/snapshots/{symbol}/events", response_model=List[StructureEventResponse])
async def get_structural_events(
    symbol: str,
    severity: Optional[str] = Query(default=None, description="Filter: low|medium|high"),
    reassessment_only: bool = Query(default=False, description="Only events that trigger policy reassessment"),
    limit: int = Query(default=50, le=200),
) -> List[StructureEventResponse]:
    """Return structural events from the latest snapshot for a symbol.

    Events are returned newest-first (by as_of_ts).
    Use reassessment_only=true to filter to events that drive policy-loop cadence (R54).
    """
    snap = _snapshot_store.get(symbol)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No structure snapshot found for {symbol}")

    events = snap.events
    if severity:
        events = [e for e in events if e.severity == severity]
    if reassessment_only:
        events = [e for e in events if e.trigger_policy_reassessment]

    events = sorted(events, key=lambda e: e.as_of_ts, reverse=True)[:limit]
    return [_event_to_response(e) for e in events]


@router.get("/snapshots/{symbol}/candidates", response_model=Dict[str, List[StructureLevelResponse]])
async def get_structural_candidates(
    symbol: str,
    direction: Optional[str] = Query(default=None, description="long|short"),
) -> Dict[str, List[StructureLevelResponse]]:
    """Return ordered stop/target/entry candidates for a symbol.

    This is the R56 structural target selector's candidate pool.
    Keys: "stop_candidates", "target_candidates", "entry_candidates".
    Ordered by distance from reference price (nearest first).
    """
    snap = _snapshot_store.get(symbol)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No structure snapshot found for {symbol}")

    return {
        "stop_candidates": [_level_to_response(l) for l in get_stop_candidates(snap)],
        "target_candidates": [_level_to_response(l) for l in get_target_candidates(snap)],
        "entry_candidates": [_level_to_response(l) for l in get_entry_candidates(snap)],
    }


@router.get("/snapshots", response_model=List[StructureSnapshotSummary])
async def list_structure_snapshots() -> List[StructureSnapshotSummary]:
    """List all cached structure snapshots (one per symbol)."""
    return [_snapshot_to_summary(snap) for snap in _snapshot_store.values()]


@router.delete("/snapshots/{symbol}")
async def clear_symbol_snapshot(symbol: str) -> Dict[str, str]:
    """Clear the cached structure snapshot for a symbol (admin/reset use)."""
    if symbol in _snapshot_store:
        del _snapshot_store[symbol]
    return {"status": "cleared", "symbol": symbol}
