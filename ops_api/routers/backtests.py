"""Backtest control and monitoring endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Literal
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
from data_loader.utils import ensure_utc, timeframe_to_seconds
from app.db.repo import Database
from app.db.models import BacktestRun, BacktestStatus
from sqlalchemy import select, desc

logger = logging.getLogger(__name__)

_db = Database()

# Task queue configuration - must match worker's TASK_QUEUE
BACKTEST_TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")

# Disk-based persistence directory
BACKTEST_CACHE_DIR = Path(".cache/backtests")
BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Keep in sync with backtesting.simulator._compute_features defaults.
MIN_FEATURE_CANDLES = 50


def save_backtest_to_disk(run_id: str, data: Dict[str, Any]):
    """Persist backtest to disk so it survives server restarts."""
    cache_file = BACKTEST_CACHE_DIR / f"{run_id}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved backtest {run_id} to disk")

def _normalize_cached_backtest(cached: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize cached backtest data for API consumers.

    Only normalizes status if it's missing or empty. Preserves running/queued
    status to avoid incorrect progress reporting.
    """
    if not isinstance(cached, dict):
        return cached
    status = cached.get("status")
    # Only normalize if status is missing or empty (not for running/queued)
    if not status:
        cached = dict(cached)
        cached["status"] = "completed"
    return cached


def load_backtest_from_disk(run_id: str) -> Dict[str, Any] | None:
    """Load backtest from disk if it exists."""
    cache_file = BACKTEST_CACHE_DIR / f"{run_id}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return _normalize_cached_backtest(pickle.load(f))
        except Exception as e:
            logger.error(f"Failed to load backtest {run_id} from disk: {e}")
    return None


async def load_backtest_from_db(run_id: str) -> Dict[str, Any] | None:
    try:
        async with _db.session() as session:
            result = await session.execute(select(BacktestRun).where(BacktestRun.run_id == run_id))
            record = result.scalar_one_or_none()
            if not record:
                return None
            payload: Dict[str, Any] = {}
            if record.results:
                try:
                    payload = json.loads(record.results)
                except json.JSONDecodeError:
                    payload = {}
            payload.setdefault("run_id", record.run_id)
            payload.setdefault("status", record.status.value if isinstance(record.status, BacktestStatus) else str(record.status))
            payload.setdefault("config", json.loads(record.config) if record.config else {})
            payload.setdefault("started_at", record.started_at.isoformat() if record.started_at else None)
            payload.setdefault("completed_at", record.completed_at.isoformat() if record.completed_at else None)
            payload.setdefault("candles_total", record.candles_total)
            payload.setdefault("candles_processed", record.candles_processed)
            return payload
    except Exception as exc:
        logger.warning("DB backtest lookup failed for %s: %s", run_id, exc)
        return None


def _merge_disk_payload(db_payload: Dict[str, Any], disk_payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(disk_payload)
    for key in ("run_id", "status", "started_at", "completed_at", "candles_total", "candles_processed", "config", "strategy"):
        if db_payload.get(key) is not None:
            merged[key] = db_payload[key]
    return merged


def _needs_disk_payload(payload: Dict[str, Any]) -> bool:
    if payload.get("equity_curve") or payload.get("trades"):
        return False
    llm_data = payload.get("llm_data") or {}
    if isinstance(llm_data, dict) and llm_data.get("artifact_path"):
        return True
    return False


async def list_backtests_from_db(status: str | None, limit: int) -> List[BacktestRun]:
    async with _db.session() as session:
        query = select(BacktestRun).order_by(desc(BacktestRun.completed_at), desc(BacktestRun.started_at))
        if status:
            try:
                query = query.where(BacktestRun.status == BacktestStatus(status))
            except ValueError:
                return []
        result = await session.execute(query.limit(limit))
        return list(result.scalars().all())


async def get_backtest_cached_async(run_id: str) -> Dict[str, Any] | None:
    cached = await load_backtest_from_db(run_id)
    if cached:
        if _needs_disk_payload(cached):
            disk = load_backtest_from_disk(run_id)
            if disk:
                return _merge_disk_payload(cached, disk)
        return cached
    return get_backtest_cached(run_id)


def list_cached_backtests() -> List[str]:
    """List all cached backtest IDs on disk."""
    return [f.stem for f in BACKTEST_CACHE_DIR.glob("*.pkl")]


def get_backtest_cached(run_id: str) -> Dict[str, Any] | None:
    """Get backtest from cache, with disk fallback."""
    with CACHE_LOCK:
        cached = BACKTEST_CACHE.get(run_id)
    if cached:
        cached = _normalize_cached_backtest(cached)
        with CACHE_LOCK:
            BACKTEST_CACHE[run_id] = cached

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
    initial_allocations: Optional[Dict[str, float]] = Field(
        default=None,
        description="Starting portfolio allocation in USD (keys: cash and/or symbols)",
    )
    strategy: Optional[str] = Field(default="baseline", description="Strategy to use (baseline, llm_strategist, etc.)")
    strategy_id: Optional[str] = Field(default=None, description="Strategy template ID for LLM strategist")
    strategy_prompt: Optional[str] = Field(default=None, description="Custom strategy prompt for LLM strategist")

    # ============================================================================
    # Risk Engine Parameters
    # ============================================================================
    max_position_risk_pct: Optional[float] = Field(
        default=None, ge=0.1, le=20.0,
        description="Max risk per trade as % of equity (default: 2%)"
    )
    max_symbol_exposure_pct: Optional[float] = Field(
        default=None, ge=5.0, le=100.0,
        description="Max notional exposure per symbol as % of equity (default: 25%)"
    )
    max_portfolio_exposure_pct: Optional[float] = Field(
        default=None, ge=10.0, le=500.0,
        description="Max total portfolio exposure as % of equity (default: 80%, >100% = leverage)"
    )
    max_daily_loss_pct: Optional[float] = Field(
        default=None, ge=1.0, le=50.0,
        description="Daily loss limit as % of equity - stops trading when hit (default: 3%)"
    )
    max_daily_risk_budget_pct: Optional[float] = Field(
        default=None, ge=1.0, le=50.0,
        description="Max cumulative risk allocated per day as % of equity"
    )

    # ============================================================================
    # Trade Frequency Parameters
    # ============================================================================
    max_trades_per_day: Optional[int] = Field(
        default=None, ge=1, le=200,
        description="Maximum number of trades per day (default: 10)"
    )
    max_triggers_per_symbol_per_day: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Maximum triggers per symbol per day (default: 5)"
    )
    judge_cadence_hours: Optional[float] = Field(
        default=4.0, ge=0.5, le=24.0,
        description="Minimum hours between judge evaluations (default: 4.0)"
    )
    judge_check_after_trades: Optional[int] = Field(
        default=3, ge=1, le=100,
        description="Trigger judge after N trades, regardless of cadence (default: 3)"
    )
    replan_on_day_boundary: Optional[bool] = Field(
        default=True,
        description="Allow start-of-day replans in adaptive mode (default: true)"
    )
    llm_calls_per_day: Optional[int] = Field(
        default=1, ge=1, le=24,
        description="Number of LLM strategist calls per day (1=daily, 24=hourly)"
    )

    # ============================================================================
    # LLM Shim Options
    # ============================================================================
    use_llm_shim: Optional[bool] = Field(
        default=False,
        description="Use deterministic strategist shim instead of calling a real LLM"
    )
    use_judge_shim: Optional[bool] = Field(
        default=False,
        description="Use deterministic judge shim instead of LLM/deterministic judge feedback"
    )

    # ============================================================================
    # Whipsaw / Anti-Flip-Flop Controls
    # ============================================================================
    min_hold_hours: Optional[float] = Field(
        default=None, ge=0.0, le=24.0,
        description="Minimum hours to hold position before exit allowed (default: 2.0, 0=disabled)"
    )
    min_flat_hours: Optional[float] = Field(
        default=None, ge=0.0, le=24.0,
        description="Minimum hours between trades for same symbol (default: 2.0, 0=disabled)"
    )
    confidence_override_threshold: Optional[str] = Field(
        default=None,
        description="Min confidence grade for entry to override exit: 'A', 'B', 'C', or null (default: 'A')"
    )
    exit_binding_mode: Literal["none", "category"] = Field(
        default="category",
        description="Exit binding policy: none (global exits) or category (entry/exit category must match). Emergency exits always allowed."
    )
    conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = Field(
        default="reverse",
        description="Resolver policy when opposing entry signals fire while in-position."
    )

    # ============================================================================
    # Execution Gating Parameters
    # ============================================================================
    min_price_move_pct: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Minimum price movement % to consider trading (default: 0.5)"
    )

    # ============================================================================
    # Walk-Away Threshold
    # ============================================================================
    walk_away_enabled: Optional[bool] = Field(
        default=False,
        description="Enable walk-away mode - stop trading after hitting profit target"
    )
    walk_away_profit_target_pct: Optional[float] = Field(
        default=25.0, ge=1.0, le=100.0,
        description="Profit target % to trigger walk-away (default: 25%)"
    )

    # ============================================================================
    # Flattening Options
    # ============================================================================
    flatten_positions_daily: Optional[bool] = Field(
        default=False,
        description="Close all positions at end of each trading day"
    )

    # ============================================================================
    # Debug / Diagnostics
    # ============================================================================
    debug_trigger_sample_rate: Optional[float] = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability (0.0-1.0) of sampling trigger evaluations for debugging. Set to 0 to disable, 1 to sample all."
    )
    debug_trigger_max_samples: Optional[int] = Field(
        default=100, ge=1, le=1000,
        description="Maximum number of trigger evaluation samples to collect (default: 100)"
    )
    indicator_debug_mode: Optional[str] = Field(
        default=None,
        description="Indicator debug mode: off, full, keys"
    )
    indicator_debug_keys: Optional[List[str]] = Field(
        default=None,
        description="Indicator keys to capture when indicator_debug_mode=keys"
    )

    # ============================================================================
    # Vector Store / RAG
    # ============================================================================
    use_trigger_vector_store: Optional[bool] = Field(
        default=False,
        description="Use vector store to retrieve relevant trigger examples for the LLM strategist"
    )


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


class BacktestListItem(BaseModel):
    """Extended backtest info for listing with rich metadata."""

    run_id: str
    status: str
    progress: float = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration metadata
    symbols: List[str] = Field(default_factory=list, description="Traded symbols")
    strategy: Optional[str] = Field(default=None, description="Strategy type (baseline/llm_strategist)")
    strategy_id: Optional[str] = Field(default=None, description="Strategy template name")
    timeframe: Optional[str] = Field(default=None, description="Primary timeframe")
    start_date: Optional[str] = Field(default=None, description="Backtest start date")
    end_date: Optional[str] = Field(default=None, description="Backtest end date")
    initial_cash: Optional[float] = Field(default=None, description="Starting capital")

    # Performance metrics (for completed backtests)
    return_pct: Optional[float] = Field(default=None, description="Total return percentage")
    final_equity: Optional[float] = Field(default=None, description="Final portfolio value")
    total_trades: Optional[int] = Field(default=None, description="Number of trades executed")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    max_drawdown_pct: Optional[float] = Field(default=None, description="Maximum drawdown percentage")
    win_rate: Optional[float] = Field(default=None, description="Win rate percentage")

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
    """Individual backtest trade record with risk stats."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: Optional[float] = None
    pnl: Optional[float] = None
    trigger_id: Optional[str] = None
    # Risk stats (Phase 6 trade-level visibility)
    risk_used_abs: Optional[float] = Field(default=None, description="Risk budget allocated for this trade")
    actual_risk_at_stop: Optional[float] = Field(default=None, description="Actual risk at stop distance (qty * stop_distance)")
    stop_distance: Optional[float] = Field(default=None, description="Distance to stop loss in price units")
    allocated_risk_abs: Optional[float] = Field(default=None, description="Risk allocated from daily budget")
    profile_multiplier: Optional[float] = Field(default=None, description="Risk multiplier from profile/regime")
    r_multiple: Optional[float] = Field(default=None, description="P&L divided by actual risk (risk-adjusted return)")


class PairedTrade(BaseModel):
    """Round-trip trade record pairing entry and exit fills."""

    symbol: str
    side: str = Field(description="Entry side (buy/sell)")
    entry_timestamp: str
    exit_timestamp: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_trigger: Optional[str] = None
    exit_trigger: Optional[str] = None
    entry_timeframe: Optional[str] = None
    qty: Optional[float] = None
    pnl: Optional[float] = None
    fees: Optional[float] = Field(default=None, description="Sum of entry + exit fees")
    hold_duration_hours: Optional[float] = None
    risk_used_abs: Optional[float] = None
    actual_risk_at_stop: Optional[float] = None
    r_multiple: Optional[float] = None


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

        try:
            requested_start_dt = ensure_utc(datetime.fromisoformat(config.start_date))
            requested_end_dt = ensure_utc(datetime.fromisoformat(config.end_date))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}") from exc
        if requested_end_dt <= requested_start_dt:
            raise HTTPException(status_code=400, detail="end_date must be after start_date")
        try:
            granularity_seconds = timeframe_to_seconds(config.timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        available_candles = int((requested_end_dt - requested_start_dt).total_seconds() // granularity_seconds) + 1
        if available_candles < MIN_FEATURE_CANDLES:
            required_range = timedelta(seconds=granularity_seconds * (MIN_FEATURE_CANDLES - 1))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Date range too short for feature windows: need at least "
                    f"{MIN_FEATURE_CANDLES} candles for timeframe {config.timeframe} "
                    f"(>= {required_range}); got ~{available_candles}."
                ),
            )

        # Generate unique run ID
        run_id = f"backtest-{uuid4()}"

        symbols = [symbol.upper() for symbol in config.symbols]
        warmup_delta = timedelta(seconds=granularity_seconds * (MIN_FEATURE_CANDLES - 1))
        buffered_start_dt = requested_start_dt - warmup_delta
        initial_allocations = None
        if config.initial_allocations:
            initial_allocations = {}
            symbol_set = set(symbols)
            for key, value in config.initial_allocations.items():
                normalized_key = "cash" if key.lower() == "cash" else key.upper()
                if value < 0:
                    raise HTTPException(status_code=400, detail=f"Allocation for {key} must be >= 0")
                if normalized_key == "cash":
                    initial_allocations["cash"] = initial_allocations.get("cash", 0.0) + float(value)
                    continue
                if normalized_key in symbol_set:
                    initial_allocations[normalized_key] = initial_allocations.get(normalized_key, 0.0) + float(value)
                    continue
                base_matches = [symbol for symbol in symbols if symbol.split("-")[0] == normalized_key]
                if len(base_matches) == 1:
                    mapped = base_matches[0]
                    initial_allocations[mapped] = initial_allocations.get(mapped, 0.0) + float(value)
                    continue
                if len(base_matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Allocation symbol {key} is ambiguous (matches {', '.join(base_matches)})",
                    )
                initial_allocations[normalized_key] = float(value)
            invalid_keys = [
                key for key in initial_allocations if key != "cash" and key not in symbol_set
            ]
            if invalid_keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"Allocation symbols not in backtest list: {', '.join(invalid_keys)}",
                )

        logger.info("Starting backtest workflow: %s", config.model_dump())

        # Build risk_params dict from config
        risk_params = {}
        if config.max_position_risk_pct is not None:
            risk_params["max_position_risk_pct"] = config.max_position_risk_pct
        if config.max_symbol_exposure_pct is not None:
            risk_params["max_symbol_exposure_pct"] = config.max_symbol_exposure_pct
        if config.max_portfolio_exposure_pct is not None:
            risk_params["max_portfolio_exposure_pct"] = config.max_portfolio_exposure_pct
        if config.max_daily_loss_pct is not None:
            risk_params["max_daily_loss_pct"] = config.max_daily_loss_pct
        if config.max_daily_risk_budget_pct is not None:
            risk_params["max_daily_risk_budget_pct"] = config.max_daily_risk_budget_pct

        try:
            async with _db.session() as session:
                record = BacktestRun(
                    run_id=run_id,
                    config=json.dumps(config.model_dump()),
                    status=BacktestStatus.running,
                    started_at=datetime.now(timezone.utc),
                    completed_at=None,
                    candles_total=available_candles,
                    candles_processed=0,
                    results=None,
                )
                session.add(record)
        except Exception as exc:
            logger.warning("Failed to persist backtest run metadata for %s: %s", run_id, exc)

        # Start Temporal workflow (non-blocking)
        client = await get_temporal_client()
        await client.start_workflow(
            BacktestWorkflow.run,
            args=[{
                "run_id": run_id,
                "symbols": symbols,
                "timeframe": config.timeframe,
                "start_date": buffered_start_dt.isoformat(),
                "end_date": requested_end_dt.isoformat(),
                "requested_start_date": requested_start_dt.isoformat(),
                "requested_end_date": requested_end_dt.isoformat(),
                "initial_cash": config.initial_cash,
                "initial_allocations": initial_allocations,
                "strategy": config.strategy or "baseline",
                "strategy_prompt": config.strategy_prompt,
                # Risk parameters
                "risk_params": risk_params if risk_params else None,
                # Trade frequency
                "max_trades_per_day": config.max_trades_per_day,
                "max_triggers_per_symbol_per_day": config.max_triggers_per_symbol_per_day,
                "llm_calls_per_day": config.llm_calls_per_day or 1,
                "judge_cadence_hours": config.judge_cadence_hours,
                "judge_check_after_trades": config.judge_check_after_trades,
                "replan_on_day_boundary": (
                    config.replan_on_day_boundary if config.replan_on_day_boundary is not None else True
                ),
                # Whipsaw controls
                "min_hold_hours": config.min_hold_hours,
                "min_flat_hours": config.min_flat_hours,
                "confidence_override_threshold": config.confidence_override_threshold,
                "exit_binding_mode": config.exit_binding_mode,
                "conflicting_signal_policy": config.conflicting_signal_policy,
                # Execution gating
                "min_price_move_pct": config.min_price_move_pct,
                # Walk-away threshold
                "walk_away_enabled": config.walk_away_enabled,
                "walk_away_profit_target_pct": config.walk_away_profit_target_pct,
                # Flattening
                "flatten_positions_daily": config.flatten_positions_daily,
                # Debug / diagnostics
                "debug_trigger_sample_rate": config.debug_trigger_sample_rate or 0.0,
                "debug_trigger_max_samples": config.debug_trigger_max_samples or 100,
                "indicator_debug_mode": config.indicator_debug_mode,
                "indicator_debug_keys": config.indicator_debug_keys,
                # Vector store
                "use_trigger_vector_store": config.use_trigger_vector_store or False,
                # LLM shim flags
                "use_llm_shim": bool(config.use_llm_shim),
                "use_judge_shim": bool(config.use_judge_shim),
            }],
            id=run_id,
            task_queue=BACKTEST_TASK_QUEUE,
        )

        return BacktestCreateResponse(
            run_id=run_id,
            status="running",
            message=f"Backtest started. Use GET /backtests/{run_id} to monitor progress."
        )

    except HTTPException:
        raise
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
            cached = await load_backtest_from_db(run_id)
            if not cached:
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
            cached = await load_backtest_from_db(run_id)
            if not cached:
                cached = load_backtest_from_disk(run_id)

            if cached:
                summary = cached.get("results_summary") or cached.get("summary") or {}
                if not summary and cached.get("result") is not None and hasattr(cached["result"], "summary"):
                    summary = cached["result"].summary  # type: ignore[assignment]
                run_status = cached.get("status", "completed")
                return BacktestResults(
                    run_id=run_id,
                    status=run_status,
                    final_equity=clean_nan(summary.get("final_equity")),
                    equity_return_pct=clean_nan(summary.get("return_pct") or summary.get("equity_return_pct")),
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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        # Handle both formats: legacy (PortfolioBacktestResult) and new (workflow dict)
        result_obj = cached.get("result")
        if result_obj is not None and hasattr(result_obj, "equity_curve"):
            # Legacy format: PortfolioBacktestResult object
            result: PortfolioBacktestResult = result_obj
            equity_series = result.equity_curve
            equity_points = [
                EquityCurvePoint(
                    timestamp=timestamp.isoformat(),
                    equity=float(equity)
                )
                for timestamp, equity in equity_series.items()
            ]
        elif "equity_curve" in cached:
            # New format: list of dicts from Temporal workflow
            equity_curve = cached["equity_curve"]
            equity_points = []
            for point in equity_curve:
                # Handle different timestamp key names
                ts = point.get("timestamp") or point.get("time") or point.get("index")
                if ts is None:
                    continue
                # Format timestamp if it's not already a string
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                equity_points.append(
                    EquityCurvePoint(
                        timestamp=str(ts),
                        equity=float(point.get("equity", point.get("value", 0)))
                    )
                )
        else:
            return []

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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        # Handle both formats: legacy (PortfolioBacktestResult) and new (workflow dict)
        result_obj = cached.get("result")
        if result_obj is not None and hasattr(result_obj, "trades"):
            # Legacy format: PortfolioBacktestResult object
            result: PortfolioBacktestResult = result_obj
            trades_df = result.trades

            if trades_df.empty:
                return []

            # Apply pagination
            trades_subset = trades_df.iloc[offset:offset + limit]

            # Convert DataFrame to list of BacktestTrade
            trades_list = []
            for _, row in trades_subset.iterrows():
                # Compute R-multiple if we have actual risk and pnl
                r_multiple = None
                pnl_val = float(row["pnl"]) if "pnl" in row and row["pnl"] is not None else None
                actual_risk = float(row["actual_risk_at_stop"]) if "actual_risk_at_stop" in row and row["actual_risk_at_stop"] is not None else None
                if pnl_val is not None and actual_risk is not None and actual_risk > 0:
                    r_multiple = pnl_val / actual_risk

                trades_list.append(
                    BacktestTrade(
                        timestamp=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
                        symbol=row.get("symbol", "UNKNOWN"),
                        side=row["side"],
                        qty=float(row["qty"]),
                        price=float(row["price"]),
                        fee=float(row["fee"]) if "fee" in row else None,
                        pnl=pnl_val,
                        trigger_id=row.get("trigger_id"),
                        risk_used_abs=float(row["risk_used_abs"]) if "risk_used_abs" in row and row["risk_used_abs"] is not None else None,
                        actual_risk_at_stop=actual_risk,
                        stop_distance=float(row["stop_distance"]) if "stop_distance" in row and row["stop_distance"] is not None else None,
                        allocated_risk_abs=float(row["allocated_risk_abs"]) if "allocated_risk_abs" in row and row["allocated_risk_abs"] is not None else None,
                        profile_multiplier=float(row["profile_multiplier"]) if "profile_multiplier" in row and row["profile_multiplier"] is not None else None,
                        r_multiple=r_multiple,
                    )
                )
        elif "trades" in cached:
            # New format: list of dicts from Temporal workflow
            trades = cached["trades"]
            if not trades:
                return []

            # Apply pagination
            trades_subset = trades[offset:offset + limit]

            trades_list = []
            for trade in trades_subset:
                # Handle different timestamp key names
                ts = trade.get("timestamp") or trade.get("time")
                if ts is None:
                    continue
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()

                # Compute R-multiple if we have actual risk and pnl
                r_multiple = None
                pnl_val = float(trade.get("pnl", 0)) if trade.get("pnl") is not None else None
                actual_risk = float(trade.get("actual_risk_at_stop", 0)) if trade.get("actual_risk_at_stop") is not None else None
                if pnl_val is not None and actual_risk is not None and actual_risk > 0:
                    r_multiple = pnl_val / actual_risk

                # Map field names: backtest stores "risk_used", API exposes "risk_used_abs"
                risk_used = trade.get("risk_used")
                if risk_used is None:
                    risk_used = trade.get("risk_used_abs")
                allocated_risk = trade.get("allocated_risk_abs")

                trades_list.append(
                    BacktestTrade(
                        timestamp=str(ts),
                        symbol=trade.get("symbol", "UNKNOWN"),
                        side=trade.get("side", "UNKNOWN"),
                        qty=float(trade.get("qty", 0)),
                        price=float(trade.get("price", 0)),
                        fee=float(trade.get("fee", 0)) if trade.get("fee") is not None else None,
                        pnl=pnl_val,
                        # Try trigger_id first, fall back to reason (LLM strategist uses "reason")
                        trigger_id=trade.get("trigger_id") or trade.get("reason"),
                        risk_used_abs=float(risk_used) if risk_used is not None else None,
                        actual_risk_at_stop=actual_risk,
                        stop_distance=float(trade.get("stop_distance", 0)) if trade.get("stop_distance") is not None else None,
                        allocated_risk_abs=float(allocated_risk) if allocated_risk is not None else None,
                        profile_multiplier=float(trade.get("profile_multiplier", 0)) if trade.get("profile_multiplier") is not None else None,
                        r_multiple=r_multiple,
                    )
                )
        else:
            return []

        return trades_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get backtest trades: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/paired_trades", response_model=List[PairedTrade])
async def get_paired_trades(run_id: str):
    """
    Get round-trip (entry/exit paired) trade records from backtest.

    Returns paired trade data from the portfolio tracker's trade_log.
    For legacy runs without trade_log, returns an empty list.
    """
    try:
        cached = await get_backtest_cached_async(run_id)
        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        trade_log = cached.get("trade_log")
        if not trade_log:
            return []

        # Build a lookup of fills by (symbol, timestamp) for fee/risk enrichment
        fills = cached.get("trades", [])
        fill_index: Dict[tuple, Dict[str, Any]] = {}
        for fill in fills:
            ts = fill.get("timestamp") or fill.get("time")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            sym = fill.get("symbol", "")
            fill_index[(sym, str(ts))] = fill

        paired: List[PairedTrade] = []
        for entry in trade_log:
            exit_ts = entry.get("timestamp")
            if hasattr(exit_ts, "isoformat"):
                exit_ts = exit_ts.isoformat()
            entry_ts = entry.get("entry_timestamp")
            if hasattr(entry_ts, "isoformat"):
                entry_ts = entry_ts.isoformat()

            symbol = entry.get("symbol", "UNKNOWN")
            entry_side = entry.get("entry_side", "buy")

            # Look up matching fills for fee/risk data
            entry_fill = fill_index.get((symbol, str(entry_ts)), {})
            exit_fill = fill_index.get((symbol, str(exit_ts)), {})

            entry_fee = float(entry_fill.get("fee", 0)) if entry_fill.get("fee") is not None else 0.0
            exit_fee = float(exit_fill.get("fee", 0)) if exit_fill.get("fee") is not None else 0.0
            total_fees = entry_fee + exit_fee

            # Compute hold duration
            hold_duration_hours = None
            if entry_ts and exit_ts:
                try:
                    from dateutil.parser import isoparse
                    entry_dt = isoparse(str(entry_ts))
                    exit_dt = isoparse(str(exit_ts))
                    delta = (exit_dt - entry_dt).total_seconds()
                    hold_duration_hours = round(delta / 3600, 2)
                except Exception:
                    pass

            # Risk fields from entry fill
            risk_used_raw = entry_fill.get("risk_used")
            if risk_used_raw is None:
                risk_used_raw = entry_fill.get("risk_used_abs")
            actual_risk = entry_fill.get("actual_risk_at_stop")

            pnl_val = float(entry.get("pnl", 0)) if entry.get("pnl") is not None else None
            r_multiple = None
            if pnl_val is not None and actual_risk is not None:
                actual_risk_f = float(actual_risk)
                if actual_risk_f > 0:
                    r_multiple = pnl_val / actual_risk_f

            # Compute qty from entry fill if available
            qty = entry_fill.get("qty") or exit_fill.get("qty")

            paired.append(PairedTrade(
                symbol=symbol,
                side=entry_side,
                entry_timestamp=str(entry_ts) if entry_ts else "",
                exit_timestamp=str(exit_ts) if exit_ts else "",
                entry_price=float(entry.get("entry_price")) if entry.get("entry_price") is not None else None,
                exit_price=float(entry.get("exit_price")) if entry.get("exit_price") is not None else None,
                entry_trigger=entry.get("entry_reason"),
                exit_trigger=entry.get("reason"),
                entry_timeframe=entry.get("entry_timeframe"),
                qty=float(qty) if qty is not None else None,
                pnl=pnl_val,
                fees=total_fees if total_fees > 0 else None,
                hold_duration_hours=hold_duration_hours,
                risk_used_abs=float(risk_used_raw) if risk_used_raw is not None else None,
                actual_risk_at_stop=float(actual_risk) if actual_risk is not None else None,
                r_multiple=r_multiple,
            ))

        return paired

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get paired trades: %s", e)
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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        config = cached["config"]
        start_date = datetime.fromisoformat(config.get("requested_start_date", config["start_date"]))
        end_date = datetime.fromisoformat(config.get("requested_end_date", config["end_date"]))
        timeframe = config.get("timeframe", "1h")
        granularity_seconds = timeframe_to_seconds(timeframe)
        warmup_delta = timedelta(seconds=granularity_seconds * (MIN_FEATURE_CANDLES - 1))
        buffered_start = start_date - warmup_delta

        # Check if indicators are already cached
        cache_key = f"{run_id}:{symbol}:indicators"
        if cache_key in BACKTEST_CACHE:
            df_with_indicators = BACKTEST_CACHE[cache_key]
        else:
            # Load OHLCV data
            df = load_ohlcv(symbol, buffered_start, end_date, timeframe=timeframe)

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

        df_window = df_with_indicators[(df_with_indicators.index >= start_date) & (df_with_indicators.index <= end_date)]

        # Apply pagination
        df_subset = df_window.iloc[offset:offset + limit]

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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        events = []
        result_obj = cached.get("result")
        if result_obj is not None and hasattr(result_obj, "trades"):
            result: PortfolioBacktestResult = result_obj
            trades_df = result.trades

            if trades_df.empty:
                return []

            # Filter by symbol if specified
            if symbol:
                trades_df = trades_df[trades_df["symbol"] == symbol]

            # Convert trades to events
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
        elif "trades" in cached:
            trades = cached["trades"]
            for trade in trades:
                trade_symbol = trade.get("symbol")
                if symbol and trade_symbol != symbol:
                    continue
                ts = trade.get("timestamp") or trade.get("time")
                if ts is None:
                    continue
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                event_data = {
                    "symbol": trade_symbol or "UNKNOWN",
                    "side": trade.get("side"),
                    "qty": float(trade.get("qty", 0)),
                    "price": float(trade.get("price", 0)),
                    "fee": float(trade.get("fee", 0)) if trade.get("fee") is not None else 0.0,
                    "pnl": float(trade.get("pnl", 0)) if trade.get("pnl") is not None else 0.0,
                    "trigger_id": trade.get("trigger_id"),
                }
                events.append(
                    PlaybackEvent(
                        timestamp=str(ts),
                        event_type="trade",
                        symbol=event_data["symbol"],
                        data=event_data,
                    )
                )
        else:
            return []

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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        config = cached.get("config", {})
        initial_cash = config.get("initial_cash", 0.0)

        # Parse the target timestamp
        target_ts = pd.Timestamp(timestamp)

        cash = initial_cash
        positions: Dict[str, float] = {}
        total_pnl = 0.0
        equity = cash

        result_obj = cached.get("result")
        if result_obj is not None and hasattr(result_obj, "trades"):
            result: PortfolioBacktestResult = result_obj
            trades_df = result.trades

            # Filter trades up to the target timestamp
            if not trades_df.empty:
                trades_before = trades_df[trades_df["time"] <= target_ts]
            else:
                trades_before = trades_df

            for _, trade in trades_before.iterrows():
                symbol = trade.get("symbol", "UNKNOWN")
                side = str(trade["side"]).upper()
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

                if symbol in positions and abs(positions[symbol]) < 1e-8:
                    del positions[symbol]

            if not result.equity_curve.empty:
                equity_at_time = result.equity_curve[result.equity_curve.index <= target_ts]
                if not equity_at_time.empty:
                    equity = float(equity_at_time.iloc[-1])
        elif "trades" in cached:
            trades = cached["trades"]
            for trade in trades:
                ts = trade.get("timestamp") or trade.get("time")
                if ts is None:
                    continue
                trade_ts = pd.Timestamp(ts)
                if trade_ts > target_ts:
                    continue
                symbol = trade.get("symbol", "UNKNOWN")
                side = str(trade.get("side", "")).upper()
                qty = float(trade.get("qty", 0))
                price = float(trade.get("price", 0))
                fee = float(trade.get("fee", 0)) if trade.get("fee") is not None else 0.0
                pnl = float(trade.get("pnl", 0)) if trade.get("pnl") is not None else 0.0

                if side == "BUY":
                    cash -= (qty * price + fee)
                    positions[symbol] = positions.get(symbol, 0.0) + qty
                elif side == "SELL":
                    cash += (qty * price - fee)
                    positions[symbol] = positions.get(symbol, 0.0) - qty
                    total_pnl += pnl

                if symbol in positions and abs(positions[symbol]) < 1e-8:
                    del positions[symbol]

            equity_curve = cached.get("equity_curve", [])
            if equity_curve:
                equity_points = []
                for point in equity_curve:
                    ts = point.get("timestamp") or point.get("time") or point.get("index")
                    if ts is None:
                        continue
                    equity_points.append((pd.Timestamp(ts), float(point.get("equity", point.get("value", 0)))))
                if equity_points:
                    equity_points.sort(key=lambda item: item[0])
                    eligible = [val for ts, val in equity_points if ts <= target_ts]
                    if eligible:
                        equity = eligible[-1]
        else:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

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
        from ops_api.temporal_client import get_temporal_client
        from temporalio.service import RPCError
        from ops_api.event_store import EventStore

        client = await get_temporal_client()

        try:
            handle = client.get_workflow_handle(run_id)
            status_data = await handle.query("get_status")
        except RPCError:
            cached = await get_backtest_cached_async(run_id)
            if not cached:
                raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")
            cached_strategy = cached.get("strategy") or (cached.get("config") or {}).get("strategy", "baseline")
            llm_costs = (cached.get("llm_data") or {}).get("llm_costs") or {}
            return {
                "run_id": run_id,
                "status": cached.get("status", "completed"),
                "progress_pct": 100.0 if cached.get("status") == "completed" else 0.0,
                "current_phase": "Completed",
                "current_timestamp": None,
                "total_candles": None,
                "processed_candles": None,
                "plans_generated": 0,
                "llm_calls_made": llm_costs.get("num_llm_calls", 0),
                "latest_logs": [],
                "strategy": cached_strategy,
            }

        def _parse_dt(value: Any) -> datetime | None:
            if not value:
                return None
            if isinstance(value, datetime):
                return value
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None

        def _format_event_message(event: Any) -> str:
            payload = event.payload or {}
            if event.type == "plan_generated":
                trigger_count = payload.get("num_triggers")
                if trigger_count is None:
                    trigger_count = len(payload.get("triggers") or [])
                return f"Plan generated ({trigger_count} triggers)"
            if event.type == "plan_judged":
                score = payload.get("score") or payload.get("overall_score")
                return f"Plan judged (score {score})" if score is not None else "Plan judged"
            if event.type == "fill":
                side = payload.get("side", "trade")
                symbol = payload.get("symbol", "unknown")
                qty = payload.get("qty")
                if qty is not None:
                    return f"Fill {side} {symbol} qty {qty}"
                return f"Fill {side} {symbol}"
            if event.type == "trade_blocked":
                reason = payload.get("reason", "unknown")
                return f"Trade blocked ({reason})"
            if event.type == "llm_call":
                model = payload.get("model", "llm")
                return f"LLM call ({model})"
            return event.type.replace("_", " ").title()

        events = EventStore().list_events(limit=500)
        run_events = [event for event in events if event.run_id == run_id]
        run_events.sort(key=lambda event: event.ts)
        plans_generated = sum(1 for event in run_events if event.type == "plan_generated")
        llm_calls_made = sum(1 for event in run_events if event.type == "llm_call")
        latest_plan_ts = next(
            (event.ts for event in reversed(run_events) if event.type == "plan_generated"),
            None,
        )
        latest_llm_ts = next(
            (event.ts for event in reversed(run_events) if event.type == "llm_call"),
            None,
        )

        timeframe = status_data.get("timeframe")
        requested_start = status_data.get("requested_start_date") or status_data.get("start_date")
        requested_end = status_data.get("requested_end_date") or status_data.get("end_date")
        start_dt = _parse_dt(requested_start)
        end_dt = _parse_dt(requested_end)
        total_candles = status_data.get("candles_total") or 0
        processed_candles = status_data.get("candles_processed") or 0
        current_timestamp = status_data.get("current_timestamp")
        current_ts = _parse_dt(current_timestamp) if current_timestamp else None

        if not current_timestamp:
            current_timestamp = latest_plan_ts.isoformat() if latest_plan_ts else None
            if not current_timestamp and latest_llm_ts:
                current_timestamp = latest_llm_ts.isoformat()
            current_ts = _parse_dt(current_timestamp) if current_timestamp else None

        if timeframe and start_dt and end_dt and not total_candles:
            try:
                tf_seconds = timeframe_to_seconds(timeframe)
                total_candles = int((end_dt - start_dt).total_seconds() // tf_seconds) + 1
            except ValueError:
                total_candles = 0

        if timeframe and start_dt and total_candles and not processed_candles:
            latest_ts = current_ts
            if not latest_ts and status_data.get("status") != "running":
                latest_ts = latest_plan_ts or latest_llm_ts
            if latest_ts:
                try:
                    tf_seconds = timeframe_to_seconds(timeframe)
                except ValueError:
                    tf_seconds = None
                if tf_seconds:
                    elapsed = (latest_ts - start_dt).total_seconds()
                    if elapsed >= 0:
                        processed_candles = int(elapsed // tf_seconds) + 1

        if total_candles and processed_candles:
            processed_candles = max(0, min(processed_candles, total_candles))

        progress_pct = status_data.get("progress", 0.0) or 0.0
        if total_candles and processed_candles:
            progress_pct = max(progress_pct, (processed_candles / total_candles) * 100.0)
        if status_data.get("status") == "completed":
            progress_pct = 100.0

        latest_logs = [
            {"timestamp": event.ts.isoformat(), "message": _format_event_message(event)}
            for event in run_events[-8:]
        ]

        return {
            "run_id": run_id,
            "status": status_data.get("status", "running"),
            "progress_pct": progress_pct,
            "current_phase": status_data.get("current_phase", "Initializing"),
            "current_timestamp": current_timestamp,
            "total_candles": total_candles or None,
            "processed_candles": processed_candles or None,
            "plans_generated": plans_generated,
            "llm_calls_made": llm_calls_made,
            "latest_logs": latest_logs,
            "strategy": status_data.get("strategy", "baseline"),
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
        cached = await get_backtest_cached_async(run_id)

        if not cached:
            raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")

        strategy = cached.get("strategy") or (cached.get("config") or {}).get("strategy")
        llm_data = cached.get("llm_data")
        if strategy != "llm_strategist" or not llm_data:
            raise HTTPException(
                status_code=404,
                detail="This backtest did not use the LLM strategist or LLM data is not available"
            )

        return {
            "run_id": run_id,
            "strategy": "llm_strategist",
            "plan_log": llm_data.get("plan_log", []),
            "llm_costs": llm_data.get("llm_costs", {}),
            "daily_reports": llm_data.get("daily_reports", []),
            "final_cash": llm_data.get("final_cash", 0),
            "final_positions": llm_data.get("final_positions", {}),
            "total_plans_generated": len(llm_data.get("plan_log", [])),
            "total_cost_usd": (llm_data.get("llm_costs", {}) or {}).get("estimated_cost_usd", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get LLM insights: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[BacktestListItem])
async def list_backtests(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=200, description="Max backtests to return")
):
    """
    List all backtests with optional filtering.

    Returns summary of all backtest runs with rich metadata for display.
    Merges in-memory cache with disk-persisted results.
    Sorted by completed_at or started_at (newest first).
    """
    try:
        records = await list_backtests_from_db(status, limit)
        if records:
            backtest_list = []
            for record in records:
                cached: Dict[str, Any] = {}
                if record.results:
                    try:
                        cached = json.loads(record.results)
                    except json.JSONDecodeError:
                        cached = {}

                cached_status = cached.get("status") or record.status.value
                config = cached.get("config") or (json.loads(record.config) if record.config else {})
                symbols = config.get("symbols") or []
                strategy = config.get("strategy")
                strategy_id = config.get("strategy_id")
                timeframe = config.get("timeframe")
                start_date = config.get("start_date") or config.get("requested_start_date")
                end_date = config.get("end_date") or config.get("requested_end_date")
                initial_cash = config.get("initial_cash")

                return_pct = None
                final_equity = None
                total_trades = None
                sharpe_ratio = None
                max_drawdown_pct = None
                win_rate = None

                if cached_status == "completed":
                    summary = cached.get("results_summary") or cached.get("summary") or {}
                    return_pct = summary.get("equity_return_pct") or summary.get("return_pct")
                    final_equity = summary.get("final_equity")
                    run_summary = summary.get("run_summary") or {}
                    metrics = run_summary.get("metrics") or summary.get("metrics") or {}
                    sharpe_ratio = metrics.get("sharpe_ratio")
                    max_drawdown_pct = metrics.get("max_drawdown_pct")
                    win_rate = metrics.get("win_rate")
                    fills = cached.get("fills") or cached.get("trades") or []
                    if isinstance(fills, list):
                        total_trades = len(fills)

                backtest_list.append(
                    BacktestListItem(
                        run_id=record.run_id,
                        status=cached_status,
                        progress=100.0 if cached_status == "completed" else 0.0,
                        started_at=record.started_at,
                        completed_at=record.completed_at,
                        symbols=symbols,
                        strategy=strategy,
                        strategy_id=strategy_id,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        initial_cash=initial_cash,
                        return_pct=clean_nan(return_pct),
                        final_equity=clean_nan(final_equity),
                        total_trades=total_trades,
                        sharpe_ratio=clean_nan(sharpe_ratio),
                        max_drawdown_pct=clean_nan(max_drawdown_pct),
                        win_rate=clean_nan(win_rate),
                        error=cached.get("error"),
                    )
                )

            return backtest_list[:limit]

        # Merge in-memory cache with disk cache
        all_run_ids = set()

        with CACHE_LOCK:
            for run_id in BACKTEST_CACHE.keys():
                all_run_ids.add(run_id)

        # Also include disk-cached backtests
        disk_run_ids = list_cached_backtests()
        all_run_ids.update(disk_run_ids)

        backtest_list = []
        for run_id in all_run_ids:
            # Try to get from memory or disk
            cached = get_backtest_cached(run_id)

            if not cached:
                continue

            # Skip entries without proper structure
            if not isinstance(cached, dict):
                logger.warning(f"Skipping malformed cache entry: {run_id}")
                continue

            cached_status = cached.get("status", "unknown")

            # Filter by status if specified
            if status and cached_status != status:
                continue

            # Extract config metadata
            config = cached.get("config") or {}
            symbols = config.get("symbols") or []
            strategy = config.get("strategy")
            strategy_id = config.get("strategy_id")
            timeframe = config.get("timeframe")
            start_date = config.get("start_date") or config.get("requested_start_date")
            end_date = config.get("end_date") or config.get("requested_end_date")
            initial_cash = config.get("initial_cash")

            # Extract performance metrics for completed backtests
            return_pct = None
            final_equity = None
            total_trades = None
            sharpe_ratio = None
            max_drawdown_pct = None
            win_rate = None

            if cached_status == "completed":
                summary = cached.get("summary") or {}
                return_pct = summary.get("equity_return_pct") or summary.get("return_pct")
                final_equity = summary.get("final_equity")

                # Try to get from metrics or run_summary
                run_summary = summary.get("run_summary") or {}
                metrics = run_summary.get("metrics") or summary.get("metrics") or {}
                sharpe_ratio = metrics.get("sharpe_ratio")
                max_drawdown_pct = metrics.get("max_drawdown_pct")
                win_rate = metrics.get("win_rate")

                # Count trades from fills
                fills = cached.get("fills") or cached.get("trades") or []
                if isinstance(fills, list):
                    total_trades = len(fills)

            # Parse timestamps
            started_at = None
            completed_at = None
            try:
                if cached.get("started_at"):
                    started_at = datetime.fromisoformat(str(cached["started_at"]).replace("Z", "+00:00"))
                if cached.get("completed_at"):
                    completed_at = datetime.fromisoformat(str(cached["completed_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

            backtest_list.append(
                BacktestListItem(
                    run_id=run_id,
                    status=cached_status,
                    progress=100.0 if cached_status == "completed" else cached.get("progress", 0.0),
                    started_at=started_at,
                    completed_at=completed_at,
                    symbols=symbols,
                    strategy=strategy,
                    strategy_id=strategy_id,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash,
                    return_pct=clean_nan(return_pct),
                    final_equity=clean_nan(final_equity),
                    total_trades=total_trades,
                    sharpe_ratio=clean_nan(sharpe_ratio),
                    max_drawdown_pct=clean_nan(max_drawdown_pct),
                    win_rate=clean_nan(win_rate),
                    error=cached.get("error"),
                )
            )

        # Sort by completed_at or started_at (newest first)
        def sort_key(item: BacktestListItem):
            if item.completed_at:
                return item.completed_at
            if item.started_at:
                return item.started_at
            return datetime.min.replace(tzinfo=timezone.utc)

        backtest_list.sort(key=sort_key, reverse=True)

        return backtest_list[:limit]

    except Exception as e:
        logger.error("Failed to list backtests: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
