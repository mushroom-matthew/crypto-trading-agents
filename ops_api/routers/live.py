"""Live trading monitoring endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ops_api.materializer import Materializer
from ops_api.event_store import EventStore
from app.db.repo import Database
from ops_api.utils.live_daily_reporter import generate_live_daily_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])

# Initialize materializer and database
_materializer = Materializer()
_db = Database()


# Response Schemas
class Position(BaseModel):
    """Current position snapshot."""

    symbol: str
    qty: float
    avg_entry_price: float
    mark_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    timestamp: datetime


class PortfolioSummary(BaseModel):
    """Portfolio summary with cash and equity."""

    cash: float
    equity: float
    day_pnl: Optional[float] = None
    total_pnl: Optional[float] = None
    positions_count: int
    updated_at: datetime


class Fill(BaseModel):
    """Trade fill record with risk stats."""

    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    # Risk stats (Phase 6 trade-level visibility)
    fee: Optional[float] = None
    pnl: Optional[float] = None
    trigger_id: Optional[str] = None
    risk_used_abs: Optional[float] = Field(default=None, description="Risk budget allocated for this trade")
    actual_risk_at_stop: Optional[float] = Field(default=None, description="Actual risk at stop distance")
    stop_distance: Optional[float] = Field(default=None, description="Distance to stop loss in price units")
    r_multiple: Optional[float] = Field(default=None, description="P&L divided by actual risk")


class BlockEvent(BaseModel):
    """Trade block event with reason."""

    timestamp: datetime
    symbol: str
    side: str
    qty: float
    reason: str
    detail: Optional[str] = None
    trigger_id: str
    correlation_id: Optional[str] = None


class RiskBudget(BaseModel):
    """Daily risk budget status."""

    date: str
    budget_total: float
    budget_used: float
    budget_available: float
    utilization_pct: float
    allocations: List[dict] = Field(default_factory=list)


@router.get("/positions", response_model=List[Position])
async def get_positions(run_id: Optional[str] = Query(default=None)):
    """
    Get current positions.

    Returns latest position snapshots for all symbols.
    """
    try:
        positions = _materializer.list_positions(limit=500)

        # Filter by run_id if provided
        # Note: Current materializer doesn't support run_id filtering yet
        # TODO: Add run_id filtering to materializer

        return [
            Position(
                symbol=pos.symbol,
                qty=pos.qty,
                avg_entry_price=0,  # Not tracked in current position_update events
                mark_price=pos.mark_price,
                unrealized_pnl=pos.pnl,
                timestamp=pos.ts
            )
            for pos in positions
        ]

    except Exception as e:
        logger.error("Failed to get positions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio", response_model=PortfolioSummary)
async def get_portfolio(run_id: Optional[str] = Query(default=None)):
    """
    Get portfolio summary.

    Returns cash, equity, P&L, and position count.
    """
    try:
        # TODO: Query actual portfolio status from ExecutionLedgerWorkflow
        # For now, compute from positions
        positions = _materializer.list_positions(limit=500)

        # Placeholder calculations
        positions_value = sum(pos.qty * (pos.mark_price or 0) for pos in positions)

        return PortfolioSummary(
            cash=0,  # TODO: Get from ledger workflow query
            equity=positions_value,
            day_pnl=None,
            total_pnl=None,
            positions_count=len(positions),
            updated_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Failed to get portfolio: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fills", response_model=List[Fill])
async def get_fills(
    limit: int = Query(default=50, le=500),
    symbol: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None)
):
    """
    Get recent fills.

    Returns trade execution records with optional filtering.
    """
    try:
        fills = _materializer.list_fills(limit=limit)

        # Filter by symbol if provided
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]

        # Filter by timestamp if provided
        if since:
            fills = [f for f in fills if f.ts >= since]

        return [
            Fill(
                order_id=fill.order_id,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                price=fill.price,
                timestamp=fill.ts,
                run_id=fill.run_id,
                correlation_id=fill.correlation_id,
                fee=fill.fee,
                pnl=fill.pnl,
                trigger_id=fill.trigger_id,
                risk_used_abs=fill.risk_used_abs,
                actual_risk_at_stop=fill.actual_risk_at_stop,
                stop_distance=fill.stop_distance,
                r_multiple=fill.r_multiple,
            )
            for fill in fills
        ]

    except Exception as e:
        logger.error("Failed to get fills: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks", response_model=List[BlockEvent])
async def get_block_events(
    limit: int = Query(default=100, le=500),
    run_id: Optional[str] = Query(default=None),
    reason: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None)
):
    """
    Get trade block events.

    Returns individual block events with reasons and context.
    """
    try:
        # Query events directly from event store
        store = EventStore()
        events = store.list_events(limit=limit)

        # Filter for trade_blocked events
        block_events = [e for e in events if e.type == "trade_blocked"]

        # Apply filters
        if run_id:
            block_events = [e for e in block_events if e.run_id == run_id]
        if reason:
            block_events = [e for e in block_events if e.payload.get("reason") == reason]
        if since:
            block_events = [e for e in block_events if e.ts >= since]

        return [
            BlockEvent(
                timestamp=event.ts,
                symbol=event.payload.get("symbol", "unknown"),
                side=event.payload.get("side", "unknown"),
                qty=event.payload.get("qty", 0),
                reason=event.payload.get("reason", "unknown"),
                detail=event.payload.get("detail"),
                trigger_id=event.payload.get("trigger_id", ""),
                correlation_id=event.correlation_id
            )
            for event in block_events
        ]

    except Exception as e:
        logger.error("Failed to get block events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-budget", response_model=RiskBudget)
async def get_risk_budget(date: Optional[str] = Query(default=None)):
    """
    Get daily risk budget status.

    Returns budget allocation and utilization for specified date (defaults to today).
    """
    try:
        # TODO: Query RiskAllocation table from database
        # For now, return placeholder

        target_date = date or datetime.utcnow().date().isoformat()

        return RiskBudget(
            date=target_date,
            budget_total=1000,  # Placeholder
            budget_used=0,
            budget_available=1000,
            utilization_pct=0,
            allocations=[]
        )

    except Exception as e:
        logger.error("Failed to get risk budget: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/block-reasons")
async def get_block_reasons_aggregate(run_id: Optional[str] = Query(default=None)):
    """
    Get aggregated block reasons.

    Returns counts of blocks grouped by reason.
    """
    try:
        aggregate = _materializer.block_reasons(run_id=run_id, limit=1000)

        return {
            "run_id": aggregate.run_id,
            "reasons": [
                {"reason": r.reason, "count": r.count}
                for r in aggregate.reasons
            ]
        }

    except Exception as e:
        logger.error("Failed to get block reasons: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-report/{date}")
async def get_daily_report(
    date: str,
    run_id: Optional[str] = Query(default=None)
) -> Dict[str, Any]:
    """
    Get daily trading report for a specific date.

    Returns backtest-style daily report with:
    - Trade count and P&L metrics
    - Block events by reason
    - Risk budget utilization
    - Win rate and performance stats

    Args:
        date: Date in YYYY-MM-DD format
        run_id: Optional run ID to filter by

    Example:
        GET /live/daily-report/2026-01-05
    """
    try:
        # Validate date format
        try:
            datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {date}. Use YYYY-MM-DD format."
            )

        # Generate report using database session
        async with _db.session() as session:
            report = await generate_live_daily_report(
                db=session,
                date=date,
                run_id=run_id
            )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate daily report for %s: %s", date, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
