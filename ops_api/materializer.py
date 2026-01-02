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
from ops_api.temporal_client import get_runtime_mode

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

    def list_runs(self) -> List[RunSummary]:
        """
        Synthesize run summaries from events.

        Note: Status is determined by recent activity (events within last 5 minutes = running).
        Mode is read from actual runtime configuration.
        TODO: Integrate with Temporal visibility API for true workflow status.
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

        for event in events:
            rid = event.run_id or "default"
            if rid not in summaries:
                # Determine status based on recent activity
                # If last event is within 5 minutes, consider it running, otherwise stopped
                time_since_update = (now - event.ts.replace(tzinfo=None)).total_seconds()
                status = "running" if time_since_update < 300 else "stopped"

                summaries[rid] = RunSummary(
                    run_id=rid,
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status=status,  # type: ignore[arg-type]
                    last_updated=event.ts,
                    mode=actual_mode,  # Use actual mode from config
                )
            summaries[rid].last_updated = max(summaries[rid].last_updated, event.ts)

            # Update status based on most recent event
            time_since_update = (now - summaries[rid].last_updated.replace(tzinfo=None)).total_seconds()
            summaries[rid].status = "running" if time_since_update < 300 else "stopped"  # type: ignore[assignment]

            if event.type == "plan_generated":
                summaries[rid].latest_plan_id = event.payload.get("plan_id")
            if event.type == "plan_judged":
                summaries[rid].latest_judge_id = event.payload.get("judge_id")

        # Only fall back to default if no events exist at all
        if not summaries:
            return [
                RunSummary(
                    run_id="no_events",
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status="stopped",  # No activity = stopped
                    last_updated=now,
                    mode=actual_mode,
                )
            ]

        return list(summaries.values())
