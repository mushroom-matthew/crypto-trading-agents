"""Agent monitoring and event endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ops_api.materializer import Materializer
from ops_api.event_store import EventStore
from ops_api.schemas import Event, RunSummary, LLMTelemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Initialize materializer
_materializer = Materializer()


# Response Schemas
class EventResponse(BaseModel):
    """Event record for API responses."""

    event_id: str
    timestamp: datetime
    source: str
    type: str
    payload: dict
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None


class WorkflowSummary(BaseModel):
    """Workflow status summary."""

    run_id: str
    status: str
    mode: str
    last_updated: datetime
    latest_plan_id: Optional[str] = None
    latest_judge_id: Optional[str] = None


@router.get("/events", response_model=List[EventResponse])
async def get_events(
    type: Optional[str] = Query(default=None, description="Filter by event type"),
    source: Optional[str] = Query(default=None, description="Filter by source"),
    run_id: Optional[str] = Query(default=None, description="Filter by run_id"),
    correlation_id: Optional[str] = Query(default=None, description="Filter by correlation_id"),
    since: Optional[datetime] = Query(default=None, description="Events after this timestamp"),
    limit: int = Query(default=100, le=500, description="Max events to return")
):
    """
    Get events with optional filtering.

    Returns event log with support for multiple filter criteria.
    """
    try:
        events = _materializer.list_events(limit=limit)

        # Apply filters
        if type:
            events = [e for e in events if e.type == type]
        if source:
            events = [e for e in events if e.source == source]
        if run_id:
            events = [e for e in events if e.run_id == run_id]
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]
        if since:
            events = [e for e in events if e.ts >= since]

        return [
            EventResponse(
                event_id=event.event_id,
                timestamp=event.ts,
                source=event.source,
                type=event.type,
                payload=event.payload,
                run_id=event.run_id,
                correlation_id=event.correlation_id
            )
            for event in events
        ]

    except Exception as e:
        logger.error("Failed to get events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/correlation/{correlation_id}", response_model=List[EventResponse])
async def get_event_chain(correlation_id: str):
    """
    Get full event chain by correlation ID.

    Returns all events with the same correlation_id to show complete decision chain
    (e.g., intent → plan → decision → block/fill).
    """
    try:
        events = _materializer.list_events(limit=500)

        # Filter by correlation_id
        chain_events = [e for e in events if e.correlation_id == correlation_id]

        # Sort by timestamp
        chain_events.sort(key=lambda e: e.ts)

        return [
            EventResponse(
                event_id=event.event_id,
                timestamp=event.ts,
                source=event.source,
                type=event.type,
                payload=event.payload,
                run_id=event.run_id,
                correlation_id=event.correlation_id
            )
            for event in chain_events
        ]

    except Exception as e:
        logger.error("Failed to get event chain: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows", response_model=List[WorkflowSummary])
async def get_workflows():
    """
    Get workflow status summaries.

    Returns status and metadata for all active workflows.
    """
    try:
        runs = await _materializer.list_runs_async()

        return [
            WorkflowSummary(
                run_id=run.run_id,
                status=run.status,
                mode=run.mode,
                last_updated=run.last_updated,
                latest_plan_id=run.latest_plan_id,
                latest_judge_id=run.latest_judge_id
            )
            for run in runs
        ]

    except Exception as e:
        logger.error("Failed to get workflows: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/telemetry", response_model=List[LLMTelemetry])
async def get_llm_telemetry(
    run_id: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    limit: int = Query(default=100, le=500)
):
    """
    Get LLM call telemetry.

    Returns token counts, costs, and performance metrics for LLM calls.
    """
    try:
        telemetry = _materializer.list_llm(limit=limit)

        # Filter by run_id if provided
        if run_id:
            telemetry = [t for t in telemetry if t.run_id == run_id]
        if since:
            telemetry = [t for t in telemetry if t.ts >= since]

        return telemetry

    except Exception as e:
        logger.error("Failed to get LLM telemetry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/summary")
async def get_llm_summary(run_id: Optional[str] = Query(default=None)):
    """
    Get aggregated LLM usage summary.

    Returns total tokens, costs, and call counts.
    """
    try:
        telemetry = _materializer.list_llm(limit=1000)

        # Filter by run_id if provided
        if run_id:
            telemetry = [t for t in telemetry if t.run_id == run_id]

        # Aggregate statistics
        total_calls = len(telemetry)
        total_tokens_in = sum(t.tokens_in for t in telemetry)
        total_tokens_out = sum(t.tokens_out for t in telemetry)
        total_cost = sum(t.cost_estimate for t in telemetry)
        avg_duration_ms = sum(t.duration_ms for t in telemetry) / total_calls if total_calls > 0 else 0

        # Group by model
        model_stats = {}
        for t in telemetry:
            if t.model not in model_stats:
                model_stats[t.model] = {"calls": 0, "tokens": 0, "cost": 0}
            model_stats[t.model]["calls"] += 1
            model_stats[t.model]["tokens"] += t.tokens_in + t.tokens_out
            model_stats[t.model]["cost"] += t.cost_estimate

        return {
            "run_id": run_id,
            "total_calls": total_calls,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_cost_usd": total_cost,
            "avg_duration_ms": avg_duration_ms,
            "by_model": model_stats
        }

    except Exception as e:
        logger.error("Failed to get LLM summary: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
