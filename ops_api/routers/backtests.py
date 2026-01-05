"""Backtest control and monitoring endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

import pandas as pd
import math
import pickle
from pathlib import Path

from backtesting.simulator import run_portfolio_backtest, PortfolioBacktestResult
from backtesting.strategies import StrategyWrapperConfig
from backtesting.dataset import load_ohlcv
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from agents.strategies.llm_client import LLMClient
from metrics.technical import sma, ema, rsi, macd, atr, bollinger_bands

logger = logging.getLogger(__name__)

# Disk-based persistence directory
BACKTEST_CACHE_DIR = Path(".cache/backtests")
BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_backtest_to_disk(run_id: str, data: Dict[str, Any]):
    """Persist backtest to disk so it survives server restarts."""
    cache_file = BACKTEST_CACHE_DIR / f"{run_id}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved backtest {run_id} to disk")


def load_backtest_from_disk(run_id: str) -> Dict[str, Any] | None:
    """Load backtest from disk if it exists."""
    cache_file = BACKTEST_CACHE_DIR / f"{run_id}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load backtest {run_id} from disk: {e}")
    return None


def list_cached_backtests() -> List[str]:
    """List all cached backtest IDs on disk."""
    return [f.stem for f in BACKTEST_CACHE_DIR.glob("*.pkl")]


def get_backtest_cached(run_id: str) -> Dict[str, Any] | None:
    """Get backtest from cache, with disk fallback."""
    with CACHE_LOCK:
        cached = BACKTEST_CACHE.get(run_id)

    # Fallback to disk if not in memory
    if not cached:
        cached = load_backtest_from_disk(run_id)
        if cached:
            # Load back into memory
            with CACHE_LOCK:
                BACKTEST_CACHE[run_id] = cached

    return cached


def clean_nan(value):
    """Convert NaN, inf, and -inf to None for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# NOTE: Thread-based backtest execution has been removed.
# Backtests now run via Temporal workflows - see BacktestWorkflow in tools/backtest_execution.py
# This provides:
# - Deterministic replay for audit trails
# - Automatic retries on failure
# - Continue-as-new for long-running backtests
# - Real-time progress tracking via workflow queries


router = APIRouter(prefix="/backtests", tags=["backtests"])

# In-memory cache for computed indicators (used by playback endpoints)
# NOTE: Backtest results are now stored in Temporal workflows, not this cache
BACKTEST_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = Lock()
MAX_CACHED_BACKTESTS = 10  # For indicator caching only


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


# Playback schemas for time-series navigation
class CandleWithIndicators(BaseModel):
    """OHLCV candle with pre-computed technical indicators."""

    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_20: Optional[float] = None
    # Oscillators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    # Volatility
    atr_14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None


class PlaybackEvent(BaseModel):
    """Event in backtest timeline (trades, triggers, blocks)."""

    timestamp: str
    event_type: str  # "trade", "trigger_fired", "trade_blocked"
    symbol: str
    data: Dict[str, Any]


class PortfolioStateSnapshot(BaseModel):
    """Portfolio state at a specific timestamp."""

    timestamp: str
    cash: float
    positions: Dict[str, float]
    equity: float
    pnl: float
    return_pct: float


@router.post("", response_model=BacktestCreateResponse)
async def start_backtest(config: BacktestConfig):
    """
    Start a new backtest run via Temporal workflow.

    Returns immediately with run_id. Backtest executes in workflow.
    Use GET /backtests/{run_id} to poll status and progress.
    Use GET /backtests/{run_id}/results to retrieve performance metrics when completed.
    """
    try:
        from ops_api.temporal_client import get_temporal_client
        from tools.backtest_execution import BacktestWorkflow

        # Generate unique run ID
        run_id = f"backtest-{uuid4()}"

        logger.info("Starting backtest workflow: %s", config.model_dump())

        # Start Temporal workflow (non-blocking)
        client = await get_temporal_client()
        await client.start_workflow(
            BacktestWorkflow.run,
            args=[{
                "run_id": run_id,
                "symbols": config.symbols,
                "timeframe": config.timeframe,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "initial_cash": config.initial_cash,
                "strategy": config.strategy or "baseline",
            }],
            id=run_id,
            task_queue="mcp-tools",
        )

        return BacktestCreateResponse(
            run_id=run_id,
            status="running",
            message=f"Backtest started. Use GET /backtests/{run_id} to monitor progress."
        )

    except Exception as e:
        logger.error("Failed to start backtest: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}", response_model=BacktestStatus)
async def get_backtest_status(run_id: str):
    """
    Get backtest status and progress by querying Temporal workflow.

    Returns current status (queued/running/completed/failed) and progress percentage.
    """
    try:
        from ops_api.temporal_client import get_temporal_client
        from temporalio.client import WorkflowHandle
        from temporalio.service import RPCError

        client = await get_temporal_client()

        try:
            # Get workflow handle
            handle: WorkflowHandle = client.get_workflow_handle(run_id)

            # Query workflow for status
            status_data = await handle.query("get_status")

            # Parse datetime strings
            started_at = datetime.fromisoformat(status_data["started_at"]) if status_data.get("started_at") else None
            completed_at = datetime.fromisoformat(status_data["completed_at"]) if status_data.get("completed_at") else None

            return BacktestStatus(
                run_id=status_data["run_id"],
                status=status_data["status"],
                progress=status_data["progress"],
                started_at=started_at,
                completed_at=completed_at,
                candles_total=status_data.get("candles_total"),
                candles_processed=status_data.get("candles_processed"),
                error=status_data.get("error"),
            )

        except RPCError as e:
            # Workflow not found, try disk cache fallback
            logger.warning(f"Workflow {run_id} not found in Temporal, checking disk cache")
            cached = load_backtest_from_disk(run_id)

            if cached:
                # Return cached status
                return BacktestStatus(
                    run_id=run_id,
                    status=cached.get("status", "completed"),
                    progress=100.0,
                    started_at=cached.get("started_at"),
                    completed_at=cached.get("completed_at"),
                    candles_total=cached.get("candles_total"),
                    candles_processed=cached.get("candles_processed"),
                    error=cached.get("error"),
                )

            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backtest status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/results", response_model=BacktestResults)
async def get_backtest_results(run_id: str):
    """
    Get backtest results summary by querying Temporal workflow.

    Returns performance metrics, equity stats, and trade statistics.
    Only available when backtest status is 'completed'.
    """
    try:
        from ops_api.temporal_client import get_temporal_client
        from temporalio.service import RPCError

        client = await get_temporal_client()

        try:
            # Get workflow handle
            handle = client.get_workflow_handle(run_id)

            # Query workflow for results
            results_data = await handle.query("get_results")

            # Check if completed
            if "error" in results_data:
                raise HTTPException(status_code=400, detail=results_data["error"])

            return BacktestResults(
                run_id=results_data["run_id"],
                status=results_data["status"],
                final_equity=clean_nan(results_data.get("final_equity")),
                equity_return_pct=clean_nan(results_data.get("equity_return_pct")),
                sharpe_ratio=clean_nan(results_data.get("sharpe_ratio")),
                max_drawdown_pct=clean_nan(results_data.get("max_drawdown_pct")),
                win_rate=clean_nan(results_data.get("win_rate")),
                total_trades=results_data.get("total_trades"),
                avg_win=clean_nan(results_data.get("avg_win")),
                avg_loss=clean_nan(results_data.get("avg_loss")),
                profit_factor=clean_nan(results_data.get("profit_factor")),
            )

        except RPCError as e:
            # Workflow not found, try disk cache fallback
            logger.warning(f"Workflow {run_id} not found in Temporal, checking disk cache")
            cached = load_backtest_from_disk(run_id)

            if cached:
                # Return results from disk cache
                result: PortfolioBacktestResult = cached.get("result")
                if result:
                    summary = result.summary
                    return BacktestResults(
                        run_id=run_id,
                        status="completed",
                        final_equity=clean_nan(summary.get("final_equity")),
                        equity_return_pct=clean_nan(summary.get("return_pct")),
                        sharpe_ratio=clean_nan(summary.get("sharpe_ratio")),
                        max_drawdown_pct=clean_nan(summary.get("max_drawdown_pct")),
                        win_rate=clean_nan(summary.get("win_rate")),
                        total_trades=summary.get("total_trades"),
                        avg_win=clean_nan(summary.get("avg_win")),
                        avg_loss=clean_nan(summary.get("avg_loss")),
                        profit_factor=clean_nan(summary.get("profit_factor")),
                    )

            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

    except HTTPException:
        raise
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
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        result: PortfolioBacktestResult = cached["result"]
        equity_series = result.equity_curve

        # Convert pandas Series to list of EquityCurvePoint
        equity_points = [
            EquityCurvePoint(
                timestamp=timestamp.isoformat(),
                equity=float(equity)
            )
            for timestamp, equity in equity_series.items()
        ]

        return equity_points

    except HTTPException:
        raise
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
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        result: PortfolioBacktestResult = cached["result"]
        trades_df = result.trades

        if trades_df.empty:
            return []

        # Apply pagination
        trades_subset = trades_df.iloc[offset:offset + limit]

        # Convert DataFrame to list of BacktestTrade
        trades_list = []
        for _, row in trades_subset.iterrows():
            trades_list.append(
                BacktestTrade(
                    timestamp=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
                    symbol=row.get("symbol", "UNKNOWN"),
                    side=row["side"],
                    qty=float(row["qty"]),
                    price=float(row["price"]),
                    fee=float(row["fee"]) if "fee" in row else None,
                    pnl=float(row["pnl"]) if "pnl" in row else None,
                    trigger_id=row.get("trigger_id"),
                )
            )

        return trades_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backtest trades: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Playback endpoints for interactive time-series navigation
@router.get("/{run_id}/playback/candles", response_model=List[CandleWithIndicators])
async def get_playback_candles(
    run_id: str,
    symbol: str = Query(..., description="Symbol to get candles for"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=2000, le=2000, description="Max candles to return")
):
    """
    Get OHLCV candles with pre-computed technical indicators for playback.

    Returns paginated candle data suitable for interactive chart playback.
    Indicators are computed server-side and cached.
    """
    try:
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        config = cached["config"]
        start_date = datetime.fromisoformat(config["start_date"])
        end_date = datetime.fromisoformat(config["end_date"])

        # Check if indicators are already cached
        cache_key = f"{run_id}:{symbol}:indicators"
        if cache_key in BACKTEST_CACHE:
            df_with_indicators = BACKTEST_CACHE[cache_key]
        else:
            # Load OHLCV data
            df = load_ohlcv(symbol, start_date, end_date)

            if df.empty:
                return []

            # Compute indicators
            sma_20_result = sma(df, period=20)
            sma_50_result = sma(df, period=50)
            ema_20_result = ema(df, period=20)
            rsi_14_result = rsi(df, period=14)
            macd_result = macd(df, fast=12, slow=26, signal=9)
            atr_14_result = atr(df, period=14)
            bb_result = bollinger_bands(df, period=20, mult=2.0)

            # Merge all indicators into the dataframe
            df_with_indicators = df.copy()
            df_with_indicators["sma_20"] = sma_20_result.series_list[0].series
            df_with_indicators["sma_50"] = sma_50_result.series_list[0].series
            df_with_indicators["ema_20"] = ema_20_result.series_list[0].series
            df_with_indicators["rsi_14"] = rsi_14_result.series_list[0].series

            # MACD has 3 series: value, signal, hist
            df_with_indicators["macd"] = macd_result.series_list[0].series
            df_with_indicators["macd_signal"] = macd_result.series_list[1].series
            df_with_indicators["macd_hist"] = macd_result.series_list[2].series

            df_with_indicators["atr_14"] = atr_14_result.series_list[0].series

            # Bollinger Bands has 3 series: middle, upper, lower
            df_with_indicators["bb_middle"] = bb_result.series_list[0].series
            df_with_indicators["bb_upper"] = bb_result.series_list[1].series
            df_with_indicators["bb_lower"] = bb_result.series_list[2].series

            # Cache the computed indicators
            with CACHE_LOCK:
                BACKTEST_CACHE[cache_key] = df_with_indicators

        # Apply pagination
        df_subset = df_with_indicators.iloc[offset:offset + limit]

        # Convert to response format
        candles = []
        for timestamp, row in df_subset.iterrows():
            candles.append(
                CandleWithIndicators(
                    timestamp=timestamp.isoformat(),
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in row else 0.0,
                    sma_20=float(row["sma_20"]) if pd.notna(row.get("sma_20")) else None,
                    sma_50=float(row["sma_50"]) if pd.notna(row.get("sma_50")) else None,
                    ema_20=float(row["ema_20"]) if pd.notna(row.get("ema_20")) else None,
                    rsi_14=float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else None,
                    macd=float(row["macd"]) if pd.notna(row.get("macd")) else None,
                    macd_signal=float(row["macd_signal"]) if pd.notna(row.get("macd_signal")) else None,
                    macd_hist=float(row["macd_hist"]) if pd.notna(row.get("macd_hist")) else None,
                    atr_14=float(row["atr_14"]) if pd.notna(row.get("atr_14")) else None,
                    bb_upper=float(row["bb_upper"]) if pd.notna(row.get("bb_upper")) else None,
                    bb_middle=float(row["bb_middle"]) if pd.notna(row.get("bb_middle")) else None,
                    bb_lower=float(row["bb_lower"]) if pd.notna(row.get("bb_lower")) else None,
                )
            )

        return candles

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get playback candles: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/playback/events", response_model=List[PlaybackEvent])
async def get_playback_events(
    run_id: str,
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=1000, le=5000, description="Max events to return")
):
    """
    Get chronological timeline events for playback (trades, triggers, blocks).

    Returns all events in chronological order for stepping through backtest history.
    """
    try:
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        result: PortfolioBacktestResult = cached["result"]
        trades_df = result.trades

        if trades_df.empty:
            return []

        # Filter by symbol if specified
        if symbol:
            trades_df = trades_df[trades_df["symbol"] == symbol]

        # Convert trades to events
        events = []
        for _, row in trades_df.iterrows():
            event_data = {
                "symbol": row.get("symbol", "UNKNOWN"),
                "side": row["side"],
                "qty": float(row["qty"]),
                "price": float(row["price"]),
                "fee": float(row.get("fee", 0)),
                "pnl": float(row.get("pnl", 0)),
                "trigger_id": row.get("trigger_id"),
            }

            events.append(
                PlaybackEvent(
                    timestamp=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
                    event_type="trade",
                    symbol=event_data["symbol"],
                    data=event_data,
                )
            )

        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by timestamp and limit
        events = sorted(events, key=lambda e: e.timestamp)[:limit]

        return events

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get playback events: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/playback/state/{timestamp}", response_model=PortfolioStateSnapshot)
async def get_portfolio_state_snapshot(run_id: str, timestamp: str):
    """
    Get portfolio state at a specific timestamp.

    Reconstructs portfolio state by replaying all trades up to the specified timestamp.
    Shows cash, positions, equity, P&L, and return percentage.
    """
    try:
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        result: PortfolioBacktestResult = cached["result"]
        config = cached["config"]
        initial_cash = config["initial_cash"]
        trades_df = result.trades

        # Parse the target timestamp
        target_ts = pd.Timestamp(timestamp)

        # Filter trades up to the target timestamp
        if not trades_df.empty:
            trades_before = trades_df[trades_df["time"] <= target_ts]
        else:
            trades_before = trades_df

        # Reconstruct portfolio state
        cash = initial_cash
        positions: Dict[str, float] = {}
        total_pnl = 0.0

        for _, trade in trades_before.iterrows():
            symbol = trade.get("symbol", "UNKNOWN")
            side = trade["side"]
            qty = float(trade["qty"])
            price = float(trade["price"])
            fee = float(trade.get("fee", 0))
            pnl = float(trade.get("pnl", 0))

            if side == "BUY":
                cash -= (qty * price + fee)
                positions[symbol] = positions.get(symbol, 0.0) + qty
            elif side == "SELL":
                cash += (qty * price - fee)
                positions[symbol] = positions.get(symbol, 0.0) - qty
                total_pnl += pnl

            # Remove positions that are effectively zero
            if symbol in positions and abs(positions[symbol]) < 1e-8:
                del positions[symbol]

        # Calculate equity (cash + position value at current prices)
        # For simplicity, we'll use the equity curve if available
        equity = cash  # Simplified - actual equity would need current prices

        # Try to get equity from equity curve
        if not result.equity_curve.empty:
            equity_at_time = result.equity_curve[result.equity_curve.index <= target_ts]
            if not equity_at_time.empty:
                equity = float(equity_at_time.iloc[-1])

        return_pct = ((equity - initial_cash) / initial_cash) * 100

        return PortfolioStateSnapshot(
            timestamp=target_ts.isoformat(),
            cash=cash,
            positions=positions,
            equity=equity,
            pnl=total_pnl,
            return_pct=return_pct,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get portfolio state snapshot: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/progress")
async def get_backtest_progress(run_id: str):
    """
    Get detailed live progress updates for a running backtest.

    Returns progress percentage, current phase, logs, and LLM plan generations.
    """
    try:
        cached = get_backtest_cached(run_id)
        with CACHE_LOCK:
            progress_data = BACKTEST_PROGRESS.get(run_id, {})

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        return {
            "run_id": run_id,
            "status": cached["status"],
            "progress_pct": progress_data.get("progress_pct", 0.0),
            "current_phase": progress_data.get("current_phase", "Initializing"),
            "current_timestamp": progress_data.get("current_timestamp"),
            "total_candles": progress_data.get("total_candles"),
            "processed_candles": progress_data.get("processed_candles"),
            "plans_generated": progress_data.get("plans_generated", 0),
            "llm_calls_made": progress_data.get("llm_calls_made", 0),
            "latest_logs": progress_data.get("logs", [])[-10:],  # Last 10 log entries
            "strategy": cached.get("strategy", "baseline"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backtest progress: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/llm-insights")
async def get_llm_insights(run_id: str):
    """
    Get LLM strategist insights for a backtest.

    Returns AI-generated strategy plans, costs, and daily reports if the backtest
    used the LLM strategist. Returns 404 if backtest not found or didn't use LLM.
    """
    try:
        cached = get_backtest_cached(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        if cached.get("strategy") != "llm_strategist" or not cached.get("llm_data"):
            raise HTTPException(
                status_code=404,
                detail="This backtest did not use the LLM strategist or LLM data is not available"
            )

        llm_data = cached["llm_data"]

        return {
            "run_id": run_id,
            "strategy": "llm_strategist",
            "plan_log": llm_data.get("plan_log", []),
            "llm_costs": llm_data.get("llm_costs", {}),
            "daily_reports": llm_data.get("daily_reports", []),
            "final_cash": llm_data.get("final_cash", 0),
            "final_positions": llm_data.get("final_positions", {}),
            "total_plans_generated": len(llm_data.get("plan_log", [])),
            "total_cost_usd": sum(llm_data.get("llm_costs", {}).values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get LLM insights: %s", e, exc_info=True)
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
        with CACHE_LOCK:
            cached_items = list(BACKTEST_CACHE.items())

        backtest_list = []
        for run_id, cached in cached_items[:limit]:
            # Skip entries without proper structure
            if not isinstance(cached, dict) or "status" not in cached:
                logger.warning(f"Skipping malformed cache entry: {run_id}")
                continue

            # Filter by status if specified
            if status and cached["status"] != status:
                continue

            backtest_list.append(
                BacktestStatus(
                    run_id=run_id,
                    status=cached.get("status", "unknown"),
                    progress=100.0 if cached.get("status") == "completed" else 0.0,
                    started_at=cached.get("started_at"),
                    completed_at=cached.get("completed_at"),
                    candles_total=None,
                    candles_processed=None,
                    error=cached.get("error"),
                )
            )

        return backtest_list

    except Exception as e:
        logger.error("Failed to list backtests: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
