"""Event materializer to produce read models for the Ops API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import List

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
        # For now, synthesize from events; later use Temporal visibility + durable state.
        events = self.store.list_events(limit=500)
        summaries: dict[str, RunSummary] = {}
        now = datetime.utcnow()
        for event in events:
            rid = event.run_id or "default"
            if rid not in summaries:
                summaries[rid] = RunSummary(
                    run_id=rid,
                    latest_plan_id=None,
                    latest_judge_id=None,
                    status="running",
                    last_updated=event.ts,
                    mode="paper",
                )
            summaries[rid].last_updated = max(summaries[rid].last_updated, event.ts)
            if event.type == "plan_generated":
                summaries[rid].latest_plan_id = event.payload.get("plan_id")
            if event.type == "plan_judged":
                summaries[rid].latest_judge_id = event.payload.get("judge_id")
        return list(summaries.values()) or [
            RunSummary(
                run_id="execution",
                latest_plan_id=None,
                latest_judge_id=None,
                status="running",
                last_updated=now,
                mode="paper",
            )
        ]
