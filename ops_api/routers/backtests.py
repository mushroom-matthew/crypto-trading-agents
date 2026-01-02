"""Backtest control and monitoring endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtests", tags=["backtests"])


# Request/Response Schemas
class BacktestConfig(BaseModel):
    """Configuration for starting a new backtest."""

    symbols: List[str] = Field(..., description="List of symbols to backtest (e.g., ['BTC-USD', 'ETH-USD'])")
    timeframe: str = Field(..., description="Candle timeframe (e.g., '1m', '5m', '15m', '1h', '4h')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    initial_cash: float = Field(default=10000, description="Starting cash balance")
    strategy: Optional[str] = Field(default="baseline", description="Strategy to use (baseline, llm_strategist, etc.)")


class BacktestCreateResponse(BaseModel):
    """Response when starting a backtest."""

    run_id: str
    status: str
    message: str


class BacktestStatus(BaseModel):
    """Backtest status and progress."""

    run_id: str
    status: str  # queued, running, completed, failed
    progress: float = Field(default=0, description="Progress percentage (0-100)")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    candles_total: Optional[int] = None
    candles_processed: Optional[int] = None
    error: Optional[str] = None


class BacktestResults(BaseModel):
    """Backtest results summary."""

    run_id: str
    status: str
    final_equity: Optional[float] = None
    equity_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None


class EquityCurvePoint(BaseModel):
    """Single point in equity curve."""

    timestamp: str
    equity: float


class BacktestTrade(BaseModel):
    """Individual backtest trade record."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: Optional[float] = None
    pnl: Optional[float] = None
    trigger_id: Optional[str] = None


@router.post("", response_model=BacktestCreateResponse)
async def start_backtest(config: BacktestConfig):
    """
    Start a new backtest run.

    This endpoint queues a backtest for execution and returns immediately.
    Use GET /backtests/{run_id} to poll for status and results.
    """
    try:
        # Generate unique run ID
        run_id = f"backtest-{uuid4()}"

        # TODO: Actually start backtest via Temporal workflow
        # For now, return a placeholder response
        logger.info("Backtest requested: %s", config.model_dump())

        # In the future:
        # from temporalio.client import Client
        # from agents.workflows.backtest_workflow import BacktestWorkflow
        #
        # client = await Client.connect("localhost:7233")
        # await client.start_workflow(
        #     BacktestWorkflow.run,
        #     config.model_dump(),
        #     id=run_id,
        #     task_queue="mcp-tools"
        # )

        return BacktestCreateResponse(
            run_id=run_id,
            status="queued",
            message="Backtest queued for execution. Use GET /backtests/{run_id} to check status."
        )

    except Exception as e:
        logger.error("Failed to start backtest: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}", response_model=BacktestStatus)
async def get_backtest_status(run_id: str):
    """
    Get backtest status and progress.

    Returns current status (queued/running/completed/failed) and progress percentage.
    """
    try:
        # TODO: Query BacktestRun table from database
        # For now, return placeholder
        logger.info("Backtest status requested: %s", run_id)

        # In the future:
        # from app.db.repo import Database
        # from app.core.config import get_settings
        #
        # settings = get_settings()
        # db = Database(settings)
        # async with db.session() as session:
        #     result = await session.execute(
        #         select(BacktestRun).where(BacktestRun.run_id == run_id)
        #     )
        #     backtest = result.scalar_one_or_none()
        #     if not backtest:
        #         raise HTTPException(404, "Backtest not found")
        #
        #     progress = 0
        #     if backtest.candles_total and backtest.candles_total > 0:
        #         progress = (backtest.candles_processed / backtest.candles_total) * 100
        #
        #     return BacktestStatus(
        #         run_id=backtest.run_id,
        #         status=backtest.status.value,
        #         progress=progress,
        #         started_at=backtest.started_at,
        #         completed_at=backtest.completed_at,
        #         candles_total=backtest.candles_total,
        #         candles_processed=backtest.candles_processed
        #     )

        return BacktestStatus(
            run_id=run_id,
            status="not_implemented",
            progress=0,
            error="Backtest status querying not yet implemented - requires database integration"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backtest status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/results", response_model=BacktestResults)
async def get_backtest_results(run_id: str):
    """
    Get backtest results summary.

    Returns performance metrics, equity stats, and trade statistics.
    Only available when backtest status is 'completed'.
    """
    try:
        # TODO: Query BacktestRun table and parse results JSON
        logger.info("Backtest results requested: %s", run_id)

        return BacktestResults(
            run_id=run_id,
            status="not_implemented",
        )

    except Exception as e:
        logger.error("Failed to get backtest results: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/equity", response_model=List[EquityCurvePoint])
async def get_equity_curve(run_id: str):
    """
    Get equity curve data for charting.

    Returns time-series data of equity values suitable for line charts.
    """
    try:
        # TODO: Parse equity_curve from BacktestRun.results JSON
        logger.info("Equity curve requested: %s", run_id)

        # Placeholder data
        return []

    except Exception as e:
        logger.error("Failed to get equity curve: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/trades", response_model=List[BacktestTrade])
async def get_backtest_trades(
    run_id: str,
    limit: int = Query(default=100, le=1000, description="Max trades to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination")
):
    """
    Get trade log from backtest.

    Returns individual trade records with pagination support.
    """
    try:
        # TODO: Parse trades from BacktestRun.results JSON
        logger.info("Backtest trades requested: %s (limit=%d, offset=%d)", run_id, limit, offset)

        # Placeholder data
        return []

    except Exception as e:
        logger.error("Failed to get backtest trades: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[BacktestStatus])
async def list_backtests(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=200, description="Max backtests to return")
):
    """
    List all backtests with optional filtering.

    Returns summary of all backtest runs with status and basic info.
    """
    try:
        # TODO: Query BacktestRun table with filters
        logger.info("List backtests requested (status=%s, limit=%d)", status, limit)

        # Placeholder data
        return []

    except Exception as e:
        logger.error("Failed to list backtests: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
