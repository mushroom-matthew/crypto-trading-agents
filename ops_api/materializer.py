"""Event materializer to produce read models for the Ops API."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime
from typing import List
import logging

from ops_api.event_store import EventStore
from ops_api.schemas import (
    BlockReason,
    BlockReasonsAggregate,
    Event,
    FillRecord,
    LLMTelemetry,
    PositionSnapshot,
    RunSummary,
)
from ops_api.temporal_client import get_runtime_mode, get_temporal_client
from temporalio.client import WorkflowExecutionStatus

logger = logging.getLogger(__name__)


class Materializer:
    """Builds read models from the append-only event log."""

    def __init__(self, store: EventStore | None = None) -> None:
        self.store = store or EventStore()

    def list_events(self, limit: int = 500) -> List[Event]:
        return self.store.list_events(limit=limit)

    def block_reasons(self, run_id: str | None = None, limit: int = 1000) -> BlockReasonsAggregate:
        events = self.store.list_events(limit=limit)
        counter: Counter[str] = Counter()
        for event in events:
            if event.type == "trade_blocked":
                reason = event.payload.get("reason", "unknown")
                if run_id is None or event.run_id == run_id:
                    counter[reason] += 1
        reasons = [BlockReason(reason=r, count=c) for r, c in counter.items()]
        return BlockReasonsAggregate(run_id=run_id, reasons=reasons)

    def list_fills(self, limit: int = 500) -> List[FillRecord]:
        events = self.store.list_events(limit=limit)
        fills: List[FillRecord] = []
        for event in events:
            if event.type != "fill":
                continue
            payload = event.payload
            fills.append(
                FillRecord(
                    order_id=str(payload.get("order_id", event.event_id)),
                    symbol=payload.get("symbol", ""),
                    side=payload.get("side", "BUY"),
                    qty=float(payload.get("qty", 0)),
                    price=float(payload.get("price", 0)),
                    ts=event.ts,
                    run_id=event.run_id,
                    correlation_id=event.correlation_id,
                )
            )
        return fills

    def list_positions(self, limit: int = 500) -> List[PositionSnapshot]:
        events = self.store.list_events(limit=limit)
        latest: dict[str, PositionSnapshot] = {}
        for event in events:
            if event.type != "position_update":
                continue
            payload = event.payload
            sym = payload.get("symbol")
            if not sym:
                continue
            snap = PositionSnapshot(
                symbol=sym,
                qty=float(payload.get("qty", 0)),
                mark_price=float(payload.get("mark_price", 0)),
                pnl=float(payload.get("pnl", 0)),
                ts=event.ts,
            )
            latest[sym] = snap
        return list(latest.values())

    def list_llm(self, limit: int = 500) -> List[LLMTelemetry]:
        events = self.store.list_events(limit=limit)
        telemetry: List[LLMTelemetry] = []
        for event in events:
            if event.type != "llm_call":
                continue
            payload = event.payload
            telemetry.append(
                LLMTelemetry(
                    run_id=event.run_id,
                    plan_id=payload.get("plan_id"),
                    prompt_hash=payload.get("prompt_hash"),
                    model=payload.get("model", ""),
                    tokens_in=int(payload.get("tokens_in", 0)),
                    tokens_out=int(payload.get("tokens_out", 0)),
                    cost_estimate=float(payload.get("cost_estimate", 0)),
                    duration_ms=int(payload.get("duration_ms", 0)),
                    ts=event.ts,
                )
            )
        return telemetry

    async def _get_workflow_status(self, workflow_id: str) -> str:
        """Query Temporal for actual workflow status.

        Returns: "running", "completed", "failed", "stopped", or "unknown"
        """
        try:
            client = await get_temporal_client()
            handle = client.get_workflow_handle(workflow_id)
            desc = await handle.describe()

            # Map Temporal status to our status enum
            status_map = {
                WorkflowExecutionStatus.RUNNING: "running",
                WorkflowExecutionStatus.COMPLETED: "completed",
                WorkflowExecutionStatus.FAILED: "failed",
                WorkflowExecutionStatus.CANCELED: "stopped",
                WorkflowExecutionStatus.TERMINATED: "stopped",
                WorkflowExecutionStatus.CONTINUED_AS_NEW: "running",
                WorkflowExecutionStatus.TIMED_OUT: "failed",
            }
            return status_map.get(desc.status, "unknown")
        except Exception as e:
            logger.debug(f"Could not get Temporal status for {workflow_id}: {e}")
            return "unknown"

    async def list_runs_async(self) -> List[RunSummary]:
        """
        Synthesize run summaries from events and Temporal workflows.

        Status is determined by:
        1. Query Temporal for known workflows (execution-ledger, broker-agent, etc.)
        2. Fall back to event-based heuristic (events within 5 min = running)

        Mode is read from actual runtime configuration.
        """
        events = self.store.list_events(limit=500)
        summaries: dict[str, RunSummary] = {}
        now = datetime.utcnow()

        # Get actual runtime mode
        try:
            actual_mode = get_runtime_mode()
        except Exception as e:
            logger.warning("Failed to get runtime mode, defaulting to 'paper': %s", e)
            actual_mode = "paper"

        # Track which workflows to query Temporal for
        known_workflows = set()

        for event in events:
            rid = event.run_id or "default"
            if rid not in summaries:
                # Initialize with event-based heuristic
                time_since_update = (now - event.ts.replace(tzinfo=None)).total_seconds()
                status = "running" if time_since_update < 300 else "stopped"

                summaries[rid] = RunSummary(
                    run_id=rid,
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status=status,  # type: ignore[arg-type]
                    last_updated=event.ts,
                    mode=actual_mode,
                )

                # Track known workflow IDs for Temporal queries
                if rid.startswith(("execution-ledger", "broker-agent", "judge-agent", "backtest-")):
                    known_workflows.add(rid)

            summaries[rid].last_updated = max(summaries[rid].last_updated, event.ts)

            # Update event-based status
            time_since_update = (now - summaries[rid].last_updated.replace(tzinfo=None)).total_seconds()
            summaries[rid].status = "running" if time_since_update < 300 else "stopped"  # type: ignore[assignment]

            if event.type == "plan_generated":
                summaries[rid].latest_plan_id = event.payload.get("plan_id")
            if event.type == "plan_judged":
                summaries[rid].latest_judge_id = event.payload.get("judge_id")

        # Query Temporal for actual status of known workflows
        for workflow_id in known_workflows:
            temporal_status = await self._get_workflow_status(workflow_id)
            if temporal_status != "unknown" and workflow_id in summaries:
                summaries[workflow_id].status = temporal_status  # type: ignore[assignment]

        # Only fall back to default if no events exist at all
        if not summaries:
            return [
                RunSummary(
                    run_id="no_events",
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status="stopped",
                    last_updated=now,
                    mode=actual_mode,
                )
            ]

        return list(summaries.values())

    def list_runs(self) -> List[RunSummary]:
        """Synchronous wrapper for list_runs_async.

        Note: This runs the async version in a new event loop.
        Prefer calling list_runs_async directly from async contexts.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                logger.warning("list_runs() called from async context, use list_runs_async() instead")
                # Fall back to event-based heuristic only
                return self._list_runs_event_based()
            return loop.run_until_complete(self.list_runs_async())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.list_runs_async())

    def _list_runs_event_based(self) -> List[RunSummary]:
        """Event-based heuristic fallback (original implementation)."""
        events = self.store.list_events(limit=500)
        summaries: dict[str, RunSummary] = {}
        now = datetime.utcnow()

        try:
            actual_mode = get_runtime_mode()
        except Exception as e:
            logger.warning("Failed to get runtime mode, defaulting to 'paper': %s", e)
            actual_mode = "paper"

        for event in events:
            rid = event.run_id or "default"
            if rid not in summaries:
                time_since_update = (now - event.ts.replace(tzinfo=None)).total_seconds()
                status = "running" if time_since_update < 300 else "stopped"

                summaries[rid] = RunSummary(
                    run_id=rid,
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status=status,  # type: ignore[arg-type]
                    last_updated=event.ts,
                    mode=actual_mode,
                )
            summaries[rid].last_updated = max(summaries[rid].last_updated, event.ts)

            time_since_update = (now - summaries[rid].last_updated.replace(tzinfo=None)).total_seconds()
            summaries[rid].status = "running" if time_since_update < 300 else "stopped"  # type: ignore[assignment]

            if event.type == "plan_generated":
                summaries[rid].latest_plan_id = event.payload.get("plan_id")
            if event.type == "plan_judged":
                summaries[rid].latest_judge_id = event.payload.get("judge_id")

        if not summaries:
            return [
                RunSummary(
                    run_id="no_events",
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status="stopped",
                    last_updated=now,
                    mode=actual_mode,
                )
            ]

        return list(summaries.values())
