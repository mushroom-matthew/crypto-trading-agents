"""API router for paper trading session management."""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])

TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")


# ============================================================================
# Request/Response Models
# ============================================================================

class PaperTradingSessionConfig(BaseModel):
    """Configuration for starting a paper trading session."""

    symbols: List[str] = Field(..., description="List of symbols to trade (e.g., ['BTC-USD', 'ETH-USD'])")
    initial_cash: float = Field(default=10000.0, description="Starting cash amount")
    initial_allocations: Optional[Dict[str, float]] = Field(
        default=None,
        description="Initial portfolio allocations in USD (e.g., {'cash': 5000, 'BTC-USD': 3000, 'ETH-USD': 2000})"
    )
    strategy_prompt: Optional[str] = Field(
        default=None,
        description="Custom strategy prompt for the LLM strategist"
    )
    strategy_id: Optional[str] = Field(
        default=None,
        description="ID of a predefined strategy template to use"
    )
    plan_interval_hours: float = Field(
        default=4.0,
        description="How often to regenerate strategy plans (in hours)"
    )
    replan_on_day_boundary: Optional[bool] = Field(
        default=True,
        description="Allow start-of-day replans in adaptive mode (default: true)"
    )
    enable_symbol_discovery: bool = Field(
        default=False,
        description="Enable daily discovery of new trading pairs"
    )
    min_volume_24h: float = Field(
        default=1_000_000,
        description="Minimum 24h volume for symbol discovery (USD)"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model to use (defaults to gpt-4o-mini)"
    )

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
    judge_check_after_trades: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Trigger judge after N trades, regardless of cadence (default: 3)"
    )

    # ============================================================================
    # Debug / Diagnostics
    # ============================================================================
    debug_trigger_sample_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Probability (0.0-1.0) of sampling trigger evaluations for debugging."
    )
    debug_trigger_max_samples: Optional[int] = Field(
        default=None, ge=1, le=1000,
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
    # Learning Book
    # ============================================================================
    learning_book_enabled: Optional[bool] = Field(
        default=None,
        description="Enable the learning book for exploratory/experimental trades"
    )
    learning_daily_risk_budget_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Max cumulative risk the learning book can allocate per day as % of equity"
    )
    learning_max_trades_per_day: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="Maximum number of learning trades per day"
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment ID to associate with this run"
    )
    experiment_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Experiment specification (see ExperimentSpec schema)"
    )


class SessionStartResponse(BaseModel):
    """Response when starting a paper trading session."""

    session_id: str
    status: str
    message: str


class SessionStatus(BaseModel):
    """Status of a paper trading session."""

    session_id: str
    status: str
    symbols: List[str]
    cycle_count: int
    has_plan: bool
    last_plan_time: Optional[str]
    plan_interval_hours: float


class PortfolioStatus(BaseModel):
    """Current portfolio status."""

    cash: float
    positions: Dict[str, float]
    entry_prices: Dict[str, float]
    last_prices: Dict[str, float]
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float


class StrategyPlanSummary(BaseModel):
    """Summary of a strategy plan."""

    generated_at: Optional[str]
    valid_until: Optional[str]
    trigger_count: int
    allowed_symbols: List[str]
    max_trades_per_day: Optional[int]


class TradeRecord(BaseModel):
    """A single trade record."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: Optional[float]
    pnl: Optional[float]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/sessions", response_model=SessionStartResponse)
async def start_session(config: PaperTradingSessionConfig):
    """Start a new paper trading session.

    Creates a PaperTradingWorkflow that will:
    - Initialize portfolio with specified allocations
    - Start streaming market data for specified symbols
    - Generate and execute strategy plans on the configured interval
    """
    from ops_api.temporal_client import get_temporal_client
    from tools.paper_trading import PaperTradingWorkflow

    session_id = f"paper-trading-{uuid4().hex[:8]}"

    try:
        # Load strategy prompt from template if strategy_id provided
        strategy_prompt = config.strategy_prompt
        if config.strategy_id and not strategy_prompt:
            try:
                from ops_api.routers.prompts import STRATEGIES_DIR, STRATEGIST_PROMPT_FILE

                if config.strategy_id == "default":
                    if STRATEGIST_PROMPT_FILE.exists():
                        strategy_prompt = STRATEGIST_PROMPT_FILE.read_text()
                else:
                    strategy_file = STRATEGIES_DIR / f"{config.strategy_id}.txt"
                    if strategy_file.exists():
                        strategy_prompt = strategy_file.read_text()
            except Exception as e:
                logger.warning(f"Failed to load strategy template: {e}")

        # Normalize symbols
        symbols = [s.upper() if "-" in s else f"{s.upper()}-USD" for s in config.symbols]

        # Normalize allocations
        initial_allocations = None
        if config.initial_allocations:
            initial_allocations = {}
            for key, value in config.initial_allocations.items():
                normalized_key = "cash" if key.lower() == "cash" else key.upper()
                if "-" not in normalized_key and normalized_key != "cash":
                    normalized_key = f"{normalized_key}-USD"
                initial_allocations[normalized_key] = float(value)

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
        if config.max_trades_per_day is not None:
            risk_params["max_trades_per_day"] = config.max_trades_per_day
        if config.max_triggers_per_symbol_per_day is not None:
            risk_params["max_triggers_per_symbol_per_day"] = config.max_triggers_per_symbol_per_day

        # Build workflow config
        workflow_config = {
            "session_id": session_id,
            "symbols": symbols,
            "initial_cash": config.initial_cash,
            "initial_allocations": initial_allocations,
            "strategy_prompt": strategy_prompt,
            "plan_interval_hours": config.plan_interval_hours,
            "replan_on_day_boundary": (
                config.replan_on_day_boundary if config.replan_on_day_boundary is not None else True
            ),
            "enable_symbol_discovery": config.enable_symbol_discovery,
            "min_volume_24h": config.min_volume_24h,
            "llm_model": config.llm_model,
            # Risk parameters
            "risk_params": risk_params if risk_params else None,
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
            # Learning book
            "learning_book_enabled": config.learning_book_enabled,
            "learning_daily_risk_budget_pct": config.learning_daily_risk_budget_pct,
            "learning_max_trades_per_day": config.learning_max_trades_per_day,
            "experiment_id": config.experiment_id,
            "experiment_spec": config.experiment_spec,
        }

        # Start workflow
        client = await get_temporal_client()
        await client.start_workflow(
            PaperTradingWorkflow.run,
            args=[workflow_config, None],  # config, resume_state
            id=session_id,
            task_queue=TASK_QUEUE,
        )

        logger.info(f"Started paper trading session: {session_id}")

        return SessionStartResponse(
            session_id=session_id,
            status="running",
            message=f"Paper trading session started. Use GET /paper-trading/sessions/{session_id} to monitor."
        )

    except Exception as e:
        logger.error(f"Failed to start paper trading session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get the status of a paper trading session."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        # Query workflow status
        status = await handle.query("get_session_status")

        return SessionStatus(
            session_id=status["session_id"],
            status="running" if not status["stopped"] else "stopped",
            symbols=status["symbols"],
            cycle_count=status["cycle_count"],
            has_plan=status["has_plan"],
            last_plan_time=status["last_plan_time"],
            plan_interval_hours=status["plan_interval_hours"],
        )

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop a paper trading session."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        # Signal workflow to stop
        await handle.signal("stop_session")

        return {"session_id": session_id, "status": "stopping", "message": "Stop signal sent"}

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/portfolio", response_model=PortfolioStatus)
async def get_portfolio(session_id: str):
    """Get current portfolio status for a paper trading session."""
    from ops_api.temporal_client import get_temporal_client
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()

        # First verify the session exists
        session_handle = client.get_workflow_handle(session_id)
        try:
            await session_handle.query("get_session_status")
        except RPCError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            raise

        # Query the shared ledger workflow
        ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
        portfolio = await ledger_handle.query("get_portfolio_status")

        return PortfolioStatus(
            cash=float(portfolio.get("cash", 0)),
            positions={k: float(v) for k, v in portfolio.get("positions", {}).items()},
            entry_prices={k: float(v) for k, v in portfolio.get("entry_prices", {}).items()},
            last_prices={k: float(v) for k, v in portfolio.get("last_prices", {}).items()},
            total_equity=float(portfolio.get("total_equity", 0)),
            unrealized_pnl=float(portfolio.get("unrealized_pnl", 0)),
            realized_pnl=float(portfolio.get("realized_pnl", 0)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/plan", response_model=StrategyPlanSummary)
async def get_current_plan(session_id: str):
    """Get the current strategy plan for a session."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        plan = await handle.query("get_current_plan")

        if not plan:
            raise HTTPException(status_code=404, detail="No strategy plan generated yet")

        return StrategyPlanSummary(
            generated_at=plan.get("generated_at"),
            valid_until=plan.get("valid_until"),
            trigger_count=len(plan.get("triggers", [])),
            allowed_symbols=plan.get("allowed_symbols", []),
            max_trades_per_day=plan.get("max_trades_per_day"),
        )

    except HTTPException:
        raise
    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/replan")
async def force_replan(session_id: str):
    """Force regeneration of strategy plan."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        await handle.signal("force_replan")

        return {"session_id": session_id, "status": "replanning", "message": "Replan signal sent"}

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to trigger replan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/trades", response_model=List[TradeRecord])
async def get_trades(session_id: str, limit: int = 100):
    """Get trade history for a session."""
    from ops_api.temporal_client import get_temporal_client
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()

        # Verify session exists
        session_handle = client.get_workflow_handle(session_id)
        try:
            await session_handle.query("get_session_status")
        except RPCError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            raise

        # Query ledger for transaction history
        ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
        transactions = await ledger_handle.query("get_transaction_history", {"limit": limit})

        trades = []
        for tx in transactions:
            trades.append(TradeRecord(
                timestamp=tx.get("timestamp", ""),
                symbol=tx.get("symbol", ""),
                side=tx.get("side", ""),
                qty=float(tx.get("qty", 0)),
                price=float(tx.get("price", 0)),
                fee=float(tx.get("fee", 0)) if tx.get("fee") else None,
                pnl=float(tx.get("pnl", 0)) if tx.get("pnl") else None,
            ))

        return trades

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/symbols")
async def update_symbols(session_id: str, symbols: List[str]):
    """Update the symbols being traded in a session."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        # Normalize symbols
        normalized = [s.upper() if "-" in s else f"{s.upper()}-USD" for s in symbols]

        await handle.signal("update_symbols", normalized)

        return {"session_id": session_id, "symbols": normalized, "message": "Symbols updated"}

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(status: Optional[str] = None, limit: int = 20):
    """List paper trading sessions.

    Note: This queries Temporal for workflow executions matching the paper-trading prefix.
    """
    from ops_api.temporal_client import get_temporal_client

    try:
        client = await get_temporal_client()

        # Build query
        query = 'WorkflowId STARTS_WITH "paper-trading-"'
        if status == "running":
            query += ' AND ExecutionStatus = "Running"'
        elif status == "stopped":
            query += ' AND ExecutionStatus != "Running"'

        sessions = []
        async for workflow in client.list_workflows(query=query):
            sessions.append({
                "session_id": workflow.id,
                "status": workflow.status.name.lower() if workflow.status else "unknown",
                "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
                "close_time": workflow.close_time.isoformat() if workflow.close_time else None,
            })
            if len(sessions) >= limit:
                break

        return {"sessions": sessions, "count": len(sessions)}

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/plans")
async def get_plan_history(session_id: str, limit: int = 50):
    """Get history of all strategy plans generated for this session.

    Returns a list of plan summaries with timestamps, trigger counts, and market regimes.
    This enables LLM insights and strategy analysis similar to backtesting.
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        plan_history = await handle.query("get_plan_history")

        # Return most recent plans first, limited
        plans = plan_history[-limit:] if len(plan_history) > limit else plan_history
        plans = list(reversed(plans))

        return {
            "session_id": session_id,
            "total_plans": len(plan_history),
            "plans": plans,
        }

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get plan history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/equity")
async def get_equity_curve(session_id: str, limit: int = 500):
    """Get equity curve history for a session.

    Returns periodic snapshots of portfolio equity over time for charting.
    Snapshots are taken every 5 minutes during active trading.
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        equity_history = await handle.query("get_equity_history")

        # Return most recent snapshots, limited
        snapshots = equity_history[-limit:] if len(equity_history) > limit else equity_history

        return {
            "session_id": session_id,
            "total_snapshots": len(equity_history),
            "equity_curve": snapshots,
        }

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get equity curve: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class UpdateStrategyRequest(BaseModel):
    """Request to update strategy prompt mid-session."""

    strategy_prompt: str = Field(..., description="New strategy prompt for the LLM")


@router.put("/sessions/{session_id}/strategy")
async def update_strategy(session_id: str, request: UpdateStrategyRequest):
    """Update the strategy prompt for a running session.

    The new prompt will take effect on the next strategy planning cycle.
    You can trigger immediate replanning by calling POST /sessions/{id}/replan.
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        await handle.signal("update_strategy_prompt", request.strategy_prompt)

        return {
            "session_id": session_id,
            "status": "updated",
            "message": "Strategy prompt updated. Will take effect on next planning cycle."
        }

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
