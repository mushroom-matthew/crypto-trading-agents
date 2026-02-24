"""API router for paper trading session management."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])

TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")


def _paper_ledger_workflow_id(session_id: str) -> str:
    """Return the dedicated ledger workflow id for a paper-trading session."""
    return f"{session_id}-ledger"


def _is_not_found(err: Exception) -> bool:
    """Return True when Temporal says the workflow/execution doesn't exist.

    Temporal can surface "not found" in two ways:
    1. RPCStatusCode.NOT_FOUND with message containing "not found"
    2. "sql: no rows in result set" — execution record purged from the DB
       (e.g. after a retention-period expiry or a container wipe).
    """
    msg = str(err).lower()
    return "not found" in msg or "no rows in result set" in msg


async def _cleanup_session_ledger(
    client: Any,
    session_id: str,
    terminate: bool = False,
) -> None:
    """Best-effort cleanup of the session-scoped ledger workflow."""
    from temporalio.service import RPCError, RPCStatusCode

    ledger_workflow_id = _paper_ledger_workflow_id(session_id)
    ledger_handle = client.get_workflow_handle(ledger_workflow_id)
    try:
        if terminate:
            await ledger_handle.terminate(reason=f"paper_session_terminated:{session_id}")
        else:
            await ledger_handle.signal("stop_workflow")
    except RPCError as err:
        if err.status == RPCStatusCode.NOT_FOUND or "not found" in str(err).lower():
            return
        logger.warning("Failed to cleanup ledger workflow %s: %s", ledger_workflow_id, err)
    except Exception as err:
        logger.warning("Failed to cleanup ledger workflow %s: %s", ledger_workflow_id, err)


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
        description="LLM model to use (defaults to gpt-5-mini)"
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
    position_meta: Dict[str, Any] = {}


class TriggerSummary(BaseModel):
    """Lightweight summary of a single trigger for the UI."""
    id: str
    symbol: str
    category: str
    direction: str
    timeframe: str
    confidence: Optional[str] = None
    entry_rule: Optional[str] = None
    exit_rule: Optional[str] = None
    hold_rule: Optional[str] = None


class StrategyPlanSummary(BaseModel):
    """Summary of a strategy plan."""

    generated_at: Optional[str]
    valid_until: Optional[str]
    trigger_count: int
    allowed_symbols: List[str]
    max_trades_per_day: Optional[int]
    # Enriched fields for the activity UI
    global_view: Optional[str] = None
    regime: Optional[str] = None
    triggers: List[TriggerSummary] = []


class TradeRecord(BaseModel):
    """A single trade record."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: Optional[float]
    pnl: Optional[float]


class TriggerRuleUpdateRequest(BaseModel):
    """Request payload for manual trigger rule edits."""
    entry_rule: Optional[str] = None
    exit_rule: Optional[str] = None
    hold_rule: Optional[str] = None
    reason: Optional[str] = Field(
        default=None,
        description="Optional human note for why this edit was made (audit trail).",
    )
    source: Optional[str] = Field(
        default="ui.paper_trading.trigger_editor",
        description="Source tag for audit trail.",
    )


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

            # Allocation semantics:
            # - If cash is omitted, symbol allocations are funded from initial_cash.
            # - If cash is provided, cash + symbol allocations must not exceed initial_cash.
            non_cash_total = sum(
                max(0.0, float(v))
                for k, v in initial_allocations.items()
                if k != "cash"
            )
            explicit_cash = initial_allocations.get("cash")
            if explicit_cash is None:
                if non_cash_total > float(config.initial_cash) + 1e-6:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Initial allocations ({non_cash_total:.2f}) exceed initial_cash "
                            f"({float(config.initial_cash):.2f})."
                        ),
                    )
                initial_allocations["cash"] = max(0.0, float(config.initial_cash) - non_cash_total)
            else:
                total_budget = float(explicit_cash) + non_cash_total
                if total_budget > float(config.initial_cash) + 1e-6:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"cash + allocations ({total_budget:.2f}) exceed initial_cash "
                            f"({float(config.initial_cash):.2f})."
                        ),
                    )

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

        ledger_workflow_id = _paper_ledger_workflow_id(session_id)

        # Build workflow config
        workflow_config = {
            "session_id": session_id,
            "ledger_workflow_id": ledger_workflow_id,
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

        # Ensure the session-scoped execution ledger workflow is running
        from agents.workflows.execution_ledger_workflow import ExecutionLedgerWorkflow
        from temporalio.service import RPCError, RPCStatusCode

        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            await ledger_handle.describe()
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                await client.start_workflow(
                    ExecutionLedgerWorkflow.run,
                    id=ledger_workflow_id,
                    task_queue=TASK_QUEUE,
                )
                logger.info("Started ledger workflow %s", ledger_workflow_id)
            else:
                raise

        try:
            await client.start_workflow(
                PaperTradingWorkflow.run,
                args=[workflow_config, None],  # config, resume_state
                id=session_id,
                task_queue=TASK_QUEUE,
            )
        except Exception:
            # Avoid orphaning a ledger if parent start fails.
            await _cleanup_session_ledger(client, session_id, terminate=True)
            raise

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
        if _is_not_found(e):
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
        # Best-effort stop of session-scoped ledger.
        await _cleanup_session_ledger(client, session_id, terminate=False)

        return {"session_id": session_id, "status": "stopping", "message": "Stop signal sent to session and ledger"}

    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/terminate")
async def terminate_session(session_id: str):
    """Hard terminate a paper trading session and its session-scoped ledger."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError, RPCStatusCode

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        try:
            await handle.terminate(reason=f"user_requested_terminate:{session_id}")
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND and "not found" not in str(err).lower():
                raise
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        await _cleanup_session_ledger(client, session_id, terminate=True)
        return {"session_id": session_id, "status": "terminated", "message": "Session and ledger terminated"}
    except HTTPException:
        raise
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to terminate session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/portfolio", response_model=PortfolioStatus)
async def get_portfolio(session_id: str):
    """Get current portfolio status for a paper trading session."""
    from ops_api.temporal_client import get_temporal_client
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from temporalio.service import RPCError, RPCStatusCode

    try:
        client = await get_temporal_client()

        # First verify the session exists
        session_handle = client.get_workflow_handle(session_id)
        try:
            await session_handle.query("get_session_status")
        except RPCError as e:
            if _is_not_found(e):
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            raise

        # Query the session-scoped ledger workflow, fallback to legacy shared ledger for older sessions.
        ledger_workflow_id = _paper_ledger_workflow_id(session_id)
        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            portfolio = await ledger_handle.query("get_portfolio_status")
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND:
                raise
            legacy_ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
            portfolio = await legacy_ledger_handle.query("get_portfolio_status")

        return PortfolioStatus(
            cash=float(portfolio.get("cash", 0)),
            positions={k: float(v) for k, v in portfolio.get("positions", {}).items()},
            entry_prices={k: float(v) for k, v in portfolio.get("entry_prices", {}).items()},
            last_prices={k: float(v) for k, v in portfolio.get("last_prices", {}).items()},
            total_equity=float(portfolio.get("total_equity", 0)),
            unrealized_pnl=float(portfolio.get("unrealized_pnl", 0)),
            realized_pnl=float(portfolio.get("realized_pnl", 0)),
            position_meta=portfolio.get("position_meta") or {},
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

        raw_triggers = plan.get("triggers", [])
        trigger_summaries = [
            TriggerSummary(
                id=t.get("id", ""),
                symbol=t.get("symbol", ""),
                category=t.get("category", ""),
                direction=t.get("direction", ""),
                timeframe=t.get("timeframe", ""),
                confidence=t.get("confidence"),
                entry_rule=(t.get("entry_rule") or "") or None,
                exit_rule=(t.get("exit_rule") or "") or None,
                hold_rule=(t.get("hold_rule") or "") or None,
            )
            for t in raw_triggers
            if isinstance(t, dict)
        ]
        return StrategyPlanSummary(
            generated_at=plan.get("generated_at"),
            valid_until=plan.get("valid_until"),
            trigger_count=len(raw_triggers),
            allowed_symbols=plan.get("allowed_symbols", []),
            max_trades_per_day=plan.get("max_trades_per_day"),
            global_view=plan.get("global_view"),
            regime=plan.get("regime"),
            triggers=trigger_summaries,
        )

    except HTTPException:
        raise
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}/triggers/{trigger_id}")
async def update_trigger_rule(
    session_id: str,
    trigger_id: str,
    request: TriggerRuleUpdateRequest,
):
    """Patch entry/exit/hold rules for one trigger in the active plan.

    Note: the workflow only accepts this update if the edited plan remains
    compilable by the deterministic trigger compiler.
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    payload: Dict[str, Any] = {
        "request_id": uuid4().hex,
        "trigger_id": trigger_id,
        "reason": (request.reason or "manual_ui_edit"),
        "source": (request.source or "ui.paper_trading.trigger_editor"),
    }
    for field_name in ("entry_rule", "exit_rule", "hold_rule"):
        value = getattr(request, field_name)
        if value is not None:
            payload[field_name] = value

    changed_fields = [k for k in ("entry_rule", "exit_rule", "hold_rule") if k in payload]
    if not changed_fields:
        raise HTTPException(
            status_code=400,
            detail="At least one of entry_rule, exit_rule, hold_rule must be provided",
        )

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)
        await handle.signal("update_trigger_rule", payload)

        # Try to confirm acceptance/rejection immediately for better UX.
        for _ in range(8):
            edits = await handle.query("get_trigger_rule_edits", 50)
            match = next(
                (entry for entry in (edits or []) if entry.get("request_id") == payload["request_id"]),
                None,
            )
            if match is not None:
                if match.get("status") == "rejected":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Trigger edit rejected: {match.get('error', 'compile validation failed')}",
                    )
                return {
                    "session_id": session_id,
                    "trigger_id": trigger_id,
                    "request_id": payload["request_id"],
                    "status": "applied",
                    "changed_fields": changed_fields,
                    "message": "Trigger rule edit applied and passed compile validation.",
                }
            await asyncio.sleep(0.1)

        return {
            "session_id": session_id,
            "trigger_id": trigger_id,
            "request_id": payload["request_id"],
            "status": "update_requested",
            "changed_fields": changed_fields,
            "message": "Trigger rule edit requested. Acceptance depends on compile validation in workflow.",
        }
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update trigger rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/trigger-rule-edits")
async def get_trigger_rule_edits(session_id: str, limit: int = 100):
    """Return manual trigger-rule edit history for auditability."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)
        edits = await handle.query("get_trigger_rule_edits", limit)
        return {
            "session_id": session_id,
            "count": len(edits or []),
            "edits": edits or [],
        }
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get trigger rule edits: {e}", exc_info=True)
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
        if _is_not_found(e):
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
    from temporalio.service import RPCError, RPCStatusCode

    try:
        client = await get_temporal_client()

        # Verify session exists
        session_handle = client.get_workflow_handle(session_id)
        try:
            await session_handle.query("get_session_status")
        except RPCError as e:
            if _is_not_found(e):
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            raise

        # Query session-scoped ledger, fallback to legacy shared ledger for older sessions.
        ledger_workflow_id = _paper_ledger_workflow_id(session_id)
        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            transactions = await ledger_handle.query("get_transaction_history", {"limit": limit})
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND:
                raise
            legacy_ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
            transactions = await legacy_ledger_handle.query("get_transaction_history", {"limit": limit})

        from datetime import datetime, timezone as _tz

        trades = []
        for tx in transactions:
            ts_raw = tx.get("timestamp", "")
            if isinstance(ts_raw, (int, float)):
                ts_str = datetime.fromtimestamp(ts_raw, tz=_tz.utc).isoformat()
            else:
                ts_str = str(ts_raw)
            trades.append(TradeRecord(
                timestamp=ts_str,
                symbol=tx.get("symbol", ""),
                side=(tx.get("side", "") or "").lower(),
                qty=float(tx.get("qty", tx.get("quantity", 0))),
                price=float(tx.get("price", tx.get("fill_price", 0))),
                fee=float(tx.get("fee", 0)) if tx.get("fee") is not None else None,
                pnl=float(tx.get("pnl", 0)) if tx.get("pnl") is not None else None,
            ))

        return trades

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/trade_sets")
async def get_trade_sets(session_id: str, limit: int = 50):
    """Get completed round-trip trades (entry + exit pairs) for a session.

    Pairs BUY fills with subsequent SELL fills for the same symbol and returns
    per-trade metrics: P&L, fees, hold time, and trigger info.  Only fully
    closed trades are returned; open positions are excluded.
    """
    from ops_api.temporal_client import get_temporal_client
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from temporalio.service import RPCError, RPCStatusCode
    from datetime import datetime, timezone as _tz

    try:
        client = await get_temporal_client()

        session_handle = client.get_workflow_handle(session_id)
        try:
            await session_handle.query("get_session_status")
        except RPCError as e:
            if _is_not_found(e):
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            raise

        ledger_workflow_id = _paper_ledger_workflow_id(session_id)
        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            transactions = await ledger_handle.query("get_transaction_history", {"limit": 2000})
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND:
                raise
            legacy_ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
            transactions = await legacy_ledger_handle.query("get_transaction_history", {"limit": 2000})

        # Sort oldest-first for pairing
        txs = sorted(transactions, key=lambda x: x.get("timestamp", 0))

        def _ts_to_iso(raw) -> str:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw, tz=_tz.utc).isoformat()
            return str(raw)

        # Pair BUY → SELL for the same symbol using a FIFO stack per symbol
        open_entries: Dict[str, List[Dict]] = {}  # symbol → list of open entry dicts
        trade_sets = []

        for tx in txs:
            symbol = tx.get("symbol", "")
            side = (tx.get("side", "") or "").upper()
            qty = float(tx.get("qty", tx.get("quantity", 0)))
            price = float(tx.get("price", tx.get("fill_price", 0)))
            fee = float(tx.get("fee", 0) or 0)
            ts_raw = tx.get("timestamp", 0)
            intent = tx.get("intent", "entry" if side == "BUY" else "exit")

            if side == "BUY" or intent == "entry":
                open_entries.setdefault(symbol, []).append(tx)
            elif side == "SELL" or intent == "exit":
                entries = open_entries.get(symbol, [])
                if not entries:
                    continue
                entry = entries.pop(0)

                entry_price = float(entry.get("price", entry.get("fill_price", 0)))
                entry_qty = float(entry.get("qty", entry.get("quantity", 0)))
                entry_ts = entry.get("timestamp", 0)
                entry_fee = float(entry.get("fee", 0) or 0)

                used_qty = min(qty, entry_qty)
                gross_pnl = (price - entry_price) * used_qty
                total_fee = fee + entry_fee
                net_pnl = gross_pnl - total_fee
                pnl_pct = (gross_pnl / (entry_price * used_qty) * 100) if entry_price > 0 else 0.0

                entry_ts_sec = entry_ts if entry_ts < 1e12 else entry_ts / 1000.0
                exit_ts_sec = ts_raw if ts_raw < 1e12 else ts_raw / 1000.0
                hold_minutes = (exit_ts_sec - entry_ts_sec) / 60.0

                # R-per-hour: net_pnl / initial_risk / hold_hours
                stop_px = entry.get("stop_price_abs")
                r_per_hour = None
                if stop_px and entry_price > 0 and hold_minutes > 0:
                    initial_risk = abs(entry_price - float(stop_px)) * used_qty
                    if initial_risk > 0:
                        r_return = net_pnl / initial_risk
                        r_per_hour = round(r_return / (hold_minutes / 60.0), 4)

                ts_rec: Dict[str, Any] = {
                    "symbol": symbol,
                    "entry_time": _ts_to_iso(entry_ts),
                    "exit_time": _ts_to_iso(ts_raw),
                    "hold_minutes": round(hold_minutes, 1),
                    "direction": "long",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "qty": used_qty,
                    "gross_pnl": round(gross_pnl, 4),
                    "fee": round(total_fee, 4),
                    "net_pnl": round(net_pnl, 4),
                    "pnl_pct": round(pnl_pct, 3),
                    "entry_trigger": entry.get("trigger_id"),
                    "exit_trigger": tx.get("trigger_id"),
                    "category": entry.get("trigger_category"),
                    "winner": net_pnl > 0,
                    "stop_price_abs": stop_px,
                    "r_per_hour": r_per_hour,
                    "estimated_bars_to_resolution": entry.get("estimated_bars_to_resolution"),
                }
                trade_sets.append(ts_rec)

        # Return most recent completed trades first
        trade_sets = list(reversed(trade_sets))[:limit]

        # Summary stats
        total = len(trade_sets)
        winners = sum(1 for t in trade_sets if t["winner"])
        total_pnl = sum(t["net_pnl"] for t in trade_sets)
        win_rate = winners / total * 100 if total else 0.0
        rph_values = [t["r_per_hour"] for t in trade_sets if t.get("r_per_hour") is not None]
        median_rph = sorted(rph_values)[len(rph_values) // 2] if rph_values else None
        hold_values = [t["hold_minutes"] for t in trade_sets]
        median_hold_minutes = sorted(hold_values)[len(hold_values) // 2] if hold_values else None

        return {
            "session_id": session_id,
            "total_completed_trades": total,
            "win_rate_pct": round(win_rate, 1),
            "total_net_pnl": round(total_pnl, 4),
            "median_r_per_hour": median_rph,
            "median_hold_minutes": median_hold_minutes,
            "trade_sets": trade_sets,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade sets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/activity")
async def get_session_activity(session_id: str, limit: int = 40):
    """Get recent activity events for a paper trading session.

    Returns tick events (live prices), trigger_fired, trade_blocked,
    order_executed, plan_generated, session_started events — everything
    needed to power the live activity feed in the UI.
    """
    from ops_api.event_store import EventStore

    try:
        store = EventStore()
        events = store.list_events_filtered(
            limit=limit,
            run_id=session_id,
            order="desc",
        )
        return {
            "session_id": session_id,
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.type,
                    "ts": e.ts.isoformat(),
                    "payload": e.payload,
                    "source": e.source,
                }
                for e in events
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get activity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/structure")
async def get_structure_snapshot(session_id: str, symbol: Optional[str] = None):
    """Get latest indicator snapshot(s) used for structure-aware planning."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    symbol_norm = symbol.upper() if symbol else None

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        # Prefer live-ish 1m snapshots when available; fall back to plan-time
        # snapshots for backward compatibility with older workflow runs.
        try:
            if symbol_norm:
                indicators = await handle.query("get_live_indicators", symbol_norm)
            else:
                indicators = await handle.query("get_live_indicators")
        except RPCError:
            if symbol_norm:
                indicators = await handle.query("get_last_indicators", symbol_norm)
            else:
                indicators = await handle.query("get_last_indicators")

        if not isinstance(indicators, dict):
            indicators = {}

        return {
            "session_id": session_id,
            "count": len(indicators),
            "indicators": indicators,
        }
    except RPCError as e:
        msg = str(e).lower()
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        # Backward compatibility for sessions started before this query existed.
        if "query" in msg and "get_last_indicators" in msg:
            return {
                "session_id": session_id,
                "count": 0,
                "indicators": {},
            }
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get structure snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/candles")
async def get_candles(session_id: str, symbol: str = "BTC-USD", timeframe: str = "1m", limit: int = 120):
    """Fetch recent OHLCV candles for charting.

    Returns up to `limit` candles in ascending order. Used by the live
    candlestick chart component. Fetches directly from the exchange (ccxt)
    so it reflects real market data regardless of session state.
    """
    import ccxt.async_support as ccxt  # type: ignore

    # Map friendly pair format to ccxt symbol
    ccxt_symbol = symbol.replace("-", "/")
    # Map UI timeframe labels to ccxt timeframe strings
    tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = timeframe.strip()
    ccxt_tf = tf_map.get(tf, "1m")
    ohlcv: List[List[float]] = []

    def _aggregate_fixed_hour_bars(rows: List[List[float]], hours: int) -> List[List[float]]:
        """Aggregate lower timeframe rows into fixed-hour candles."""
        if hours <= 1 or not rows:
            return rows
        bucket_ms = hours * 60 * 60 * 1000
        buckets: Dict[int, Dict[str, float]] = {}
        order: List[int] = []
        for row in rows:
            if len(row) < 6:
                continue
            ts_ms, o, h, l, c, v = row[:6]
            bucket_start = int((int(ts_ms) // bucket_ms) * bucket_ms)
            if bucket_start not in buckets:
                buckets[bucket_start] = {
                    "time": float(bucket_start),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
                order.append(bucket_start)
            else:
                b = buckets[bucket_start]
                b["high"] = max(b["high"], float(h))
                b["low"] = min(b["low"], float(l))
                b["close"] = float(c)
                b["volume"] += float(v)
        order.sort()
        return [
            [
                int(buckets[k]["time"]),
                buckets[k]["open"],
                buckets[k]["high"],
                buckets[k]["low"],
                buckets[k]["close"],
                buckets[k]["volume"],
            ]
            for k in order
        ]

    exchange = ccxt.coinbase({"enableRateLimit": True})
    try:
        # Coinbase often lacks native 1w/1M bars in ccxt. Build them from 1d.
        if tf in {"1w", "1M"}:
            # Weekly needs ~7 days/bar; monthly ~31 days/bar.
            lookback_days = max(limit * (7 if tf == "1w" else 31), 90)
            daily = await exchange.fetch_ohlcv(ccxt_symbol, "1d", limit=lookback_days)
            buckets: Dict[str, Dict[str, float]] = {}
            order: List[str] = []

            for row in daily:
                ts_ms, o, h, l, c, v = row
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                if tf == "1w":
                    iso = dt.isocalendar()
                    key = f"{iso.year}-W{iso.week:02d}"
                else:
                    key = f"{dt.year}-{dt.month:02d}"

                if key not in buckets:
                    buckets[key] = {
                        "time": float(ts_ms),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                    }
                    order.append(key)
                else:
                    b = buckets[key]
                    b["high"] = max(b["high"], float(h))
                    b["low"] = min(b["low"], float(l))
                    b["close"] = float(c)
                    b["volume"] += float(v)

            ohlcv = [
                [
                    int(buckets[k]["time"]),
                    buckets[k]["open"],
                    buckets[k]["high"],
                    buckets[k]["low"],
                    buckets[k]["close"],
                    buckets[k]["volume"],
                ]
                for k in order
            ]
            if len(ohlcv) > limit:
                ohlcv = ohlcv[-limit:]
        elif tf == "4h":
            # Prefer native 4h bars, but fallback to aggregating 1h if unavailable.
            try:
                ohlcv = await exchange.fetch_ohlcv(ccxt_symbol, "4h", limit=limit)
            except Exception as native_err:
                logger.info("Native 4h candles unavailable for %s: %s", symbol, native_err)
                ohlcv = []
            if not ohlcv:
                lookback_hours = max(limit * 4 + 8, 240)
                hourly = await exchange.fetch_ohlcv(ccxt_symbol, "1h", limit=lookback_hours)
                ohlcv = _aggregate_fixed_hour_bars(hourly, 4)
                if len(ohlcv) > limit:
                    ohlcv = ohlcv[-limit:]
        else:
            ohlcv = await exchange.fetch_ohlcv(ccxt_symbol, ccxt_tf, limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Exchange error: {e}")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

    candles = [
        {
            "time": row[0],       # ms timestamp
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5],
        }
        for row in ohlcv
    ]
    return {"symbol": symbol, "timeframe": timeframe, "candles": candles}


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
        if _is_not_found(e):
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
        if _is_not_found(e):
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
        if _is_not_found(e):
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
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
