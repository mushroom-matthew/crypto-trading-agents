"""Contract-first schemas for events, read models, and Ops API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal[
    "tick",
    "intent",
    "plan_generated",
    "plan_judged",
    "trigger_fired",
    "trade_blocked",
    "order_submitted",
    "fill",
    "position_update",
    "llm_call",
]


class Event(BaseModel):
    event_id: str
    ts: datetime
    emitted_at: Optional[datetime] = None
    source: str
    type: EventType
    payload: Dict[str, Any]
    dedupe_key: Optional[str] = None
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None


class BlockReason(BaseModel):
    reason: str
    count: int = 0


class BlockReasonsAggregate(BaseModel):
    run_id: Optional[str]
    reasons: List[BlockReason] = Field(default_factory=list)


class PositionSnapshot(BaseModel):
    symbol: str
    qty: float
    mark_price: float
    pnl: float
    ts: datetime


class FillRecord(BaseModel):
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float
    price: float
    ts: datetime
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    # Risk stats (Phase 6 trade-level visibility)
    fee: Optional[float] = None
    pnl: Optional[float] = None
    trigger_id: Optional[str] = None
    risk_used_abs: Optional[float] = None
    actual_risk_at_stop: Optional[float] = None
    stop_distance: Optional[float] = None
    r_multiple: Optional[float] = None


class RunSummary(BaseModel):
    run_id: str
    latest_plan_id: Optional[str] = None
    latest_judge_id: Optional[str] = None
    status: Literal["running", "paused", "stopped"]
    last_updated: datetime
    mode: Literal["paper", "live"]


class LLMTelemetry(BaseModel):
    run_id: Optional[str]
    plan_id: Optional[str]
    prompt_hash: Optional[str]
    model: str
    tokens_in: int
    tokens_out: int
    cost_estimate: float
    duration_ms: int
    ts: datetime
