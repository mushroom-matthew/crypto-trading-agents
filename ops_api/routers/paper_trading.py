"""API router for paper trading session management."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

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


async def _verify_session_exists(client: Any, session_id: str) -> None:
    """Lightweight existence check that avoids workflow query-buffer pressure."""
    from temporalio.service import RPCError

    handle = client.get_workflow_handle(session_id)
    try:
        await handle.describe()
    except RPCError as err:
        if _is_not_found(err):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise


def _count_completed_trade_sets(transactions: List[Dict[str, Any]]) -> int:
    """Count FIFO entry/exit pairs from ledger transaction history."""
    txs = sorted(transactions, key=lambda x: x.get("timestamp", 0))
    open_entries: Dict[str, List[Dict[str, Any]]] = {}
    completed = 0

    for tx in txs:
        symbol = tx.get("symbol", "")
        side = (tx.get("side", "") or "").upper()
        intent = tx.get("intent", "entry" if side == "BUY" else "exit")
        if side == "BUY" or intent == "entry":
            open_entries.setdefault(symbol, []).append(tx)
        elif side == "SELL" or intent == "exit":
            entries = open_entries.get(symbol, [])
            if not entries:
                continue
            entries.pop(0)
            completed += 1

    return completed


async def _get_session_list_summary(client: Any, session_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort summary for the sessions dropdown."""
    try:
        from agents.constants import MOCK_LEDGER_WORKFLOW_ID
        from temporalio.service import RPCError, RPCStatusCode

        ledger_workflow_id = _paper_ledger_workflow_id(session_id)
        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            portfolio = await ledger_handle.query("get_portfolio_status")
            transactions = await ledger_handle.query("get_transaction_history", {"limit": 2000})
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND:
                raise
            legacy_ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
            portfolio = await legacy_ledger_handle.query("get_portfolio_status")
            transactions = await legacy_ledger_handle.query("get_transaction_history", {"limit": 2000})

        positions = portfolio.get("positions") or {}
        open_positions = sum(1 for qty in positions.values() if abs(float(qty or 0)) > 1e-12)
        completed_positions = _count_completed_trade_sets(transactions or [])

        realized_pnl = float(portfolio.get("realized_pnl", 0) or 0)
        unrealized_pnl = float(portfolio.get("unrealized_pnl", 0) or 0)
        total_pnl = realized_pnl + unrealized_pnl

        return {
            "open_positions": open_positions,
            "completed_positions": completed_positions,
            "total_pnl": round(total_pnl, 4),
            "realized_pnl": round(realized_pnl, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
        }
    except Exception as err:
        logger.warning("Failed to build session summary for %s: %s", session_id, err)
        return None


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
        description=(
            "How often to regenerate strategy plans (in hours). "
            "When left at default (4.0), auto-computed from indicator_timeframe: "
            "1m→0.25h, 5m→0.5h, 15m→0.75h, 1h→4.0h, 6h→12.0h, 1d→24.0h. "
            "Explicit values always override auto-computation."
        ),
    )

    # Sentinel to detect whether plan_interval_hours was explicitly set by the caller.
    # Pydantic does not expose a "was this field provided?" flag directly, so we track
    # it via a private flag set in model_post_init.
    _plan_interval_explicit: bool = False

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        # Mark plan_interval as explicitly set only when it differs from the default.
        # This allows auto-computation to kick in when the caller omits the field.
        object.__setattr__(self, "_plan_interval_explicit", self.plan_interval_hours != 4.0)
    indicator_timeframe: str = Field(
        default="1h",
        description=(
            "OHLCV timeframe for indicator computation — must be a Coinbase-supported "
            "granularity: '1m', '5m', '15m', '1h', '6h', '1d'."
        ),
    )

    direction_bias: str = Field(
        default="neutral",
        description=(
            "Directional bias for the session: 'long', 'short', or 'neutral'. "
            "Injected as a direction hint into the LLM prompt when non-neutral. "
            "Sourced from screener candidate direction_bias."
        ),
    )

    @field_validator("indicator_timeframe")
    @classmethod
    def validate_indicator_timeframe(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "6h", "1d"}
        if v not in allowed:
            raise ValueError(
                f"indicator_timeframe '{v}' is not supported by Coinbase. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v

    @field_validator("direction_bias")
    @classmethod
    def validate_direction_bias(cls, v: str) -> str:
        allowed = {"long", "short", "neutral"}
        if v not in allowed:
            raise ValueError(f"direction_bias '{v}' invalid. Allowed: {sorted(allowed)}")
        return v

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
    exit_binding_mode: Literal["none", "category", "exact"] = Field(
        default="exact",
        description="Exit binding policy: none (global exits), category (entry/exit category must match), or exact (only the originating trigger may exit). Emergency exits always allowed."
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

    # ============================================================================
    # Research Budget (Runbook 48)
    # ============================================================================
    research_budget_enabled: bool = Field(
        default=True,
        description="Allocate a separate capital pool for hypothesis testing"
    )
    research_budget_fraction: Optional[float] = Field(
        default=None, ge=0.01, le=0.50,
        description="Fraction of initial_cash allocated to research (0.01–0.50, default 0.10)"
    )
    research_max_loss_pct: Optional[float] = Field(
        default=None, ge=5.0, le=100.0,
        description="Max loss as % of research capital before pausing (5–100%, default 50%)"
    )

    # ============================================================================
    # Screener Integration (Runbook 68)
    # ============================================================================
    screener_regime: Optional[str] = Field(
        default=None,
        description="Regime label from screener (e.g. 'bull_trending'). "
                    "Used to pre-filter eligible playbooks before LLM generation.",
    )

    # ============================================================================
    # AI Portfolio Planner (Runbook 76)
    # ============================================================================
    use_ai_planner: bool = Field(
        default=False,
        description=(
            "Enable the AI portfolio planner (R76). When true, generates a SessionIntent "
            "before the first strategy plan, overriding the symbol list with LLM-selected "
            "symbols and injecting a SESSION_INTENT block into the strategist prompt."
        ),
    )

    # ============================================================================
    # Trailing Stop (Runbook 85)
    # ============================================================================
    default_trailing_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Session-level trailing stop config applied to every new position. "
            "Matches TrailingStopConfig schema. "
            "Modes: none, breakeven_only, atr_trail, pct_trail, step_trail. "
            "None = static stop (default). "
            "Example: {\"mode\": \"atr_trail\", \"breakeven_at_r\": 1.0, \"trail_activation_r\": 1.5, \"atr_trail_multiple\": 2.0}"
        ),
    )

    # ============================================================================
    # R:R Gate
    # ============================================================================
    min_rr_ratio: float = Field(
        default=1.75,
        ge=0.5,
        le=5.0,
        description=(
            "Minimum reward:risk ratio required for entry (default 1.75). "
            "Entries whose resolved structural stop/target fall below this are blocked. "
            "The LLM never sees this threshold — it picks structural levels honestly."
        ),
    )


class SessionStartResponse(BaseModel):
    """Response when starting a paper trading session."""

    session_id: str
    status: str
    message: str
    plan_interval_hours: Optional[float] = None
    plan_interval_auto: Optional[bool] = None  # True when derived from indicator_timeframe


class SessionStatus(BaseModel):
    """Status of a paper trading session."""

    session_id: str
    status: str
    symbols: List[str]
    cycle_count: int
    has_plan: bool
    last_plan_time: Optional[str]
    plan_interval_hours: float
    indicator_timeframe: Optional[str] = None
    direction_bias: Optional[str] = None
    enable_symbol_discovery: Optional[bool] = None
    # R77: CadenceGovernor summary
    cadence_summary: Optional[Dict[str, Any]] = None
    # R76: AI planner session intent
    session_intent_symbols: Optional[List[str]] = None
    session_intent: Optional[Dict[str, Any]] = None
    # R80: WorldState summary
    world_state_summary: Optional[Dict[str, Any]] = None


class PortfolioStatus(BaseModel):
    """Current portfolio status."""

    cash: float
    initial_cash: Optional[float] = None
    positions: Dict[str, float]
    entry_prices: Dict[str, float]
    last_prices: Dict[str, float]
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    position_meta: Dict[str, Any] = {}


class SessionListSummary(BaseModel):
    """Compact session-level metrics for dropdown displays."""

    open_positions: int = 0
    completed_positions: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class SessionListItem(BaseModel):
    """List item for the session picker."""

    session_id: str
    status: str
    start_time: Optional[str] = None
    close_time: Optional[str] = None
    summary: Optional[SessionListSummary] = None


class SessionListResponse(BaseModel):
    """Response payload for the session picker."""

    sessions: List[SessionListItem]
    count: int


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
    # Hypothesis visibility (Fix C)
    rationale: Optional[str] = None
    stop_anchor_type: Optional[str] = None
    target_anchor_type: Optional[str] = None
    entry_reference_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    rr_ratio: Optional[float] = None


class HypothesisSummary(BaseModel):
    """Lightweight summary of a TradeHypothesis for the UI."""
    id: str
    symbol: str
    direction: str
    timeframe: str
    confidence_grade: str
    thesis: str
    indicator_basis: Optional[str] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    rr_ratio: Optional[float] = None
    playbook_id: Optional[str] = None
    regime_context: Optional[str] = None


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
    plan_rationale: Optional[str] = None
    triggers: List[TriggerSummary] = []
    hypotheses: List[HypothesisSummary] = []
    # R68/R49: snapshot provenance fields for plan inspector UI
    snapshot_id: Optional[str] = None
    snapshot_hash: Optional[str] = None
    snapshot_missing_sections: List[str] = []
    snapshot_staleness_seconds: Optional[float] = None
    snapshot_as_of_ts: Optional[str] = None


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

        # Auto-compute plan_interval_hours from indicator_timeframe when not explicitly set.
        # Only applies when the caller left plan_interval_hours at the default (4.0).
        _TF_DEFAULT_INTERVALS: Dict[str, float] = {
            "1m": 0.25, "5m": 0.5, "15m": 0.75, "1h": 4.0, "6h": 12.0, "1d": 24.0,
        }
        plan_interval_hours = config.plan_interval_hours
        plan_interval_auto = False
        if not config._plan_interval_explicit:
            derived = _TF_DEFAULT_INTERVALS.get(config.indicator_timeframe)
            if derived is not None and derived != plan_interval_hours:
                plan_interval_hours = derived
                plan_interval_auto = True

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
            "plan_interval_hours": plan_interval_hours,
            "indicator_timeframe": config.indicator_timeframe,
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
            # Research budget
            "research_budget_enabled": config.research_budget_enabled,
            "research_budget_fraction": config.research_budget_fraction,
            "research_max_loss_pct": config.research_max_loss_pct,
            # R59: directional bias
            "direction_bias": config.direction_bias,
            # R68: screener-derived regime for playbook filtering
            "screener_regime": config.screener_regime,
            # R76: AI portfolio planner
            "use_ai_planner": config.use_ai_planner,
            # R85: trailing stop session defaults
            "default_trailing_config": config.default_trailing_config,
            # R:R gate (LLM-invisible — structural targets screened post-resolution)
            "min_rr_ratio": config.min_rr_ratio,
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

        _msg = f"Paper trading session started. Use GET /paper-trading/sessions/{session_id} to monitor."
        if plan_interval_auto:
            _msg += f" plan_interval_hours auto-set to {plan_interval_hours}h from indicator_timeframe={config.indicator_timeframe}."
        return SessionStartResponse(
            session_id=session_id,
            status="running",
            message=_msg,
            plan_interval_hours=plan_interval_hours,
            plan_interval_auto=plan_interval_auto,
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

        # Execution status from describe() is the source of truth for lifecycle:
        # running/completed/failed/terminated/etc. Query state can be stale after
        # closure (e.g. a failed workflow may still return stopped=False in query).
        desc = await handle.describe()
        execution_status = desc.status.name.lower() if desc.status else "unknown"

        status: Dict[str, Any] = {}
        try:
            status = await handle.query("get_session_status")
        except Exception as query_err:
            logger.warning(
                "Session %s: get_session_status query unavailable (%s); using describe()-only fallback",
                session_id,
                query_err,
            )

        return SessionStatus(
            session_id=status.get("session_id", session_id),
            status=execution_status,
            symbols=status.get("symbols") or [],
            cycle_count=int(status.get("cycle_count") or 0),
            has_plan=bool(status.get("has_plan", False)),
            last_plan_time=status.get("last_plan_time"),
            plan_interval_hours=float(status.get("plan_interval_hours") or 0.0),
            indicator_timeframe=status.get("indicator_timeframe"),
            direction_bias=status.get("direction_bias"),
            enable_symbol_discovery=status.get("enable_symbol_discovery"),
            cadence_summary=status.get("cadence_summary"),
            session_intent_symbols=status.get("session_intent_symbols"),
            session_intent=status.get("session_intent"),
            world_state_summary=status.get("world_state_summary"),
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

        # Verify session exists without consuming workflow query slots.
        await _verify_session_exists(client, session_id)

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

        # Merge live R-tracking from the paper trading workflow into position_meta.
        # This adds current_R, mfe_r, trade_state, r1_reached, r2_reached, r3_reached
        # for each open position so the frontend can show live trade state.
        position_meta = dict(portfolio.get("position_meta") or {})
        try:
            pt_handle = client.get_workflow_handle(session_id)
            r_tracking = await pt_handle.query("get_live_r_tracking")
            for sym, r_data in (r_tracking or {}).items():
                if sym in position_meta:
                    position_meta[sym] = {**position_meta[sym], **r_data}
                else:
                    position_meta[sym] = r_data
        except Exception:
            pass  # R-tracking is best-effort; don't fail the portfolio call

        return PortfolioStatus(
            cash=float(portfolio.get("cash", 0)),
            initial_cash=float(portfolio.get("initial_cash", 0)) if portfolio.get("initial_cash") is not None else None,
            positions={k: float(v) for k, v in portfolio.get("positions", {}).items()},
            entry_prices={k: float(v) for k, v in portfolio.get("entry_prices", {}).items()},
            last_prices={k: float(v) for k, v in portfolio.get("last_prices", {}).items()},
            total_equity=float(portfolio.get("total_equity", 0)),
            unrealized_pnl=float(portfolio.get("unrealized_pnl", 0)),
            realized_pnl=float(portfolio.get("realized_pnl", 0)),
            position_meta=position_meta,
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
                confidence=t.get("confidence") or t.get("confidence_grade"),
                entry_rule=(t.get("entry_rule") or "") or None,
                exit_rule=(t.get("exit_rule") or "") or None,
                hold_rule=(t.get("hold_rule") or "") or None,
                rationale=t.get("rationale") or None,
                stop_anchor_type=t.get("stop_anchor_type") or None,
                target_anchor_type=t.get("target_anchor_type") or None,
                entry_reference_price=t.get("entry_reference_price") if isinstance(t.get("entry_reference_price"), (int, float)) else None,
                stop_price=t.get("stop_price") if isinstance(t.get("stop_price"), (int, float)) else None,
                target_price=t.get("target_price") if isinstance(t.get("target_price"), (int, float)) else None,
                rr_ratio=t.get("rr_ratio") if isinstance(t.get("rr_ratio"), (int, float)) else None,
            )
            for t in raw_triggers
            if isinstance(t, dict)
        ]
        raw_hypotheses = plan.get("hypotheses", []) or []
        hypothesis_summaries = [
            HypothesisSummary(
                id=h.get("id", ""),
                symbol=h.get("symbol", ""),
                direction=h.get("direction", ""),
                timeframe=h.get("timeframe", ""),
                confidence_grade=h.get("confidence_grade", "B"),
                thesis=h.get("thesis", ""),
                indicator_basis=h.get("indicator_basis") or None,
                stop_price=h.get("stop_price") if isinstance(h.get("stop_price"), (int, float)) else None,
                target_price=h.get("target_price") if isinstance(h.get("target_price"), (int, float)) else None,
                rr_ratio=h.get("rr_ratio") if isinstance(h.get("rr_ratio"), (int, float)) else None,
                playbook_id=h.get("playbook_id") or None,
                regime_context=h.get("regime_context") or None,
            )
            for h in raw_hypotheses
            if isinstance(h, dict)
        ]
        return StrategyPlanSummary(
            generated_at=plan.get("generated_at"),
            valid_until=plan.get("valid_until"),
            trigger_count=len(raw_triggers),
            allowed_symbols=plan.get("allowed_symbols", []),
            max_trades_per_day=plan.get("max_trades_per_day"),
            global_view=plan.get("global_view"),
            regime=plan.get("regime"),
            plan_rationale=plan.get("rationale") or None,
            triggers=trigger_summaries,
            hypotheses=hypothesis_summaries,
            # R68: snapshot provenance from workflow query (R49 metadata)
            snapshot_id=plan.get("snapshot_id"),
            snapshot_hash=plan.get("snapshot_hash"),
            snapshot_missing_sections=plan.get("snapshot_missing_sections") or [],
            snapshot_staleness_seconds=plan.get("snapshot_staleness_seconds"),
            snapshot_as_of_ts=plan.get("snapshot_as_of_ts"),
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

        # Verify session exists without consuming workflow query slots.
        await _verify_session_exists(client, session_id)

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
                ts_sec = ts_raw / 1000.0 if ts_raw > 1e12 else ts_raw
                ts_str = datetime.fromtimestamp(ts_sec, tz=_tz.utc).isoformat()
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


def _build_hold_overrides_from_order_events(session_id: str, max_events: int = 10000) -> Dict[str, List[float]]:
    """Derive per-symbol FIFO hold durations from order_executed event telemetry.

    Why this exists:
    - Legacy sessions may have coarse or replay-skewed ledger timestamps.
    - order_executed events carry reliable event-store timestamps and are emitted
      on every fill. Using them as hold-time overrides keeps historical UI/metrics
      stable without changing PnL pairing logic.
    """
    from ops_api.event_store import EventStore

    store = EventStore()
    events = store.list_events_filtered(
        event_type="order_executed",
        run_id=session_id,
        limit=max_events,
        order="asc",
    )
    open_entries: Dict[str, List[datetime]] = {}
    holds_by_symbol: Dict[str, List[float]] = {}

    for event in events:
        payload = event.payload or {}
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            continue

        side = str(payload.get("side") or "").lower()
        intent = str(payload.get("intent") or ("entry" if side == "buy" else "exit")).lower()
        is_entry = intent == "entry" or side == "buy"
        is_exit = intent in {"exit", "flat", "conflict_exit", "conflict_reverse"} or side == "sell"

        if is_entry:
            open_entries.setdefault(symbol, []).append(event.ts)
            continue
        if not is_exit:
            continue

        entries = open_entries.get(symbol, [])
        if not entries:
            continue

        entry_ts = entries.pop(0)
        hold_minutes = max(0.0, (event.ts - entry_ts).total_seconds() / 60.0)
        holds_by_symbol.setdefault(symbol, []).append(hold_minutes)

    return holds_by_symbol


def _build_entry_overrides_from_order_events(
    session_id: str,
    max_events: int = 10000,
) -> Dict[str, List[Dict[str, Any]]]:
    """Derive per-symbol FIFO entry metadata from order_executed events.

    This backfills trade-set display fields for legacy sessions where the ledger
    transaction row does not retain stop/target metadata even though the event
    feed emitted it at execution time.
    """
    from ops_api.event_store import EventStore

    store = EventStore()
    events = store.list_events_filtered(
        event_type="order_executed",
        run_id=session_id,
        limit=max_events,
        order="asc",
    )

    entry_meta_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        payload = event.payload or {}
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            continue

        side = str(payload.get("side") or "").lower()
        intent = str(payload.get("intent") or ("entry" if side == "buy" else "exit")).lower()
        is_entry = intent == "entry" or side == "buy"
        if not is_entry:
            continue

        entry_meta_by_symbol.setdefault(symbol, []).append({
            "trigger_id": payload.get("trigger_id"),
            "timeframe": payload.get("timeframe"),
            "trigger_category": payload.get("trigger_category"),
            "entry_rule": payload.get("entry_rule"),
            "planned_exit_rule": payload.get("exit_rule"),
            "hold_rule": payload.get("hold_rule"),
            "stop_price_abs": payload.get("stop_price"),
            "target_price_abs": payload.get("target_price"),
            "planned_rr": payload.get("planned_rr"),
        })

    return entry_meta_by_symbol


def _build_exit_overrides_from_order_events(
    session_id: str,
    max_events: int = 10000,
) -> Dict[str, List[Dict[str, Any]]]:
    """Derive per-symbol FIFO exit metadata from order_executed events."""
    from ops_api.event_store import EventStore

    store = EventStore()
    events = store.list_events_filtered(
        event_type="order_executed",
        run_id=session_id,
        limit=max_events,
        order="asc",
    )

    exit_meta_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        payload = event.payload or {}
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            continue

        side = str(payload.get("side") or "").lower()
        intent = str(payload.get("intent") or ("entry" if side == "buy" else "exit")).lower()
        is_exit = intent in {"exit", "flat", "conflict_exit", "conflict_reverse"} or side == "sell"
        if not is_exit:
            continue

        exit_meta_by_symbol.setdefault(symbol, []).append({
            "trigger_id": payload.get("trigger_id"),
            "timeframe": payload.get("timeframe"),
            "trigger_category": payload.get("trigger_category"),
            "entry_rule": payload.get("entry_rule"),
            "exit_rule": payload.get("exit_rule"),
            "hold_rule": payload.get("hold_rule"),
        })

    return exit_meta_by_symbol


def _index_trade_set_triggers(plan_payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index trigger definitions from plan payloads in chronological order."""
    trigger_catalog: Dict[str, Dict[str, Any]] = {}
    for plan in plan_payloads:
        triggers = plan.get("triggers") or []
        if not isinstance(triggers, list):
            continue
        for trigger in triggers:
            if not isinstance(trigger, dict):
                continue
            trigger_id = str(trigger.get("id") or "").strip()
            if not trigger_id:
                continue
            trigger_catalog[trigger_id] = dict(trigger)
    return trigger_catalog


async def _build_trade_set_trigger_catalog(client: Any, session_id: str) -> Dict[str, Dict[str, Any]]:
    """Best-effort trigger catalog for entry/exit rule enrichment."""
    handle = client.get_workflow_handle(session_id)
    plan_payloads: List[Dict[str, Any]] = []
    try:
        plan_history = await asyncio.wait_for(handle.query("get_plan_history"), timeout=2.5)
        if isinstance(plan_history, list):
            plan_payloads.extend([p for p in plan_history if isinstance(p, dict)])
    except Exception as err:
        logger.debug("Failed to query plan_history for %s: %s", session_id, err)
    try:
        current_plan = await asyncio.wait_for(handle.query("get_current_plan"), timeout=2.5)
        if isinstance(current_plan, dict):
            plan_payloads.append(current_plan)
    except Exception as err:
        logger.debug("Failed to query current_plan for %s: %s", session_id, err)
    return _index_trade_set_triggers(plan_payloads)


def _resolve_trade_set_rule_fields(
    entry_tx: Dict[str, Any],
    exit_tx: Dict[str, Any],
    entry_override: Dict[str, Any],
    exit_override: Dict[str, Any],
    trigger_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge rule metadata for a paired trade set."""
    entry_trigger_id = entry_tx.get("trigger_id") or entry_override.get("trigger_id")
    exit_trigger_id = exit_tx.get("trigger_id") or exit_override.get("trigger_id")
    entry_trigger = trigger_catalog.get(str(entry_trigger_id or ""))
    exit_trigger = trigger_catalog.get(str(exit_trigger_id or ""))

    planned_exit_rule = (
        entry_tx.get("exit_rule")
        or entry_override.get("planned_exit_rule")
        or (entry_trigger or {}).get("exit_rule")
    )
    executed_exit_rule = (
        exit_tx.get("exit_rule")
        or exit_override.get("exit_rule")
        or (exit_trigger or {}).get("exit_rule")
        or planned_exit_rule
    )

    return {
        "entry_trigger": entry_trigger_id,
        "exit_trigger": exit_trigger_id,
        "category": (
            entry_tx.get("trigger_category")
            or entry_override.get("trigger_category")
            or (entry_trigger or {}).get("category")
        ),
        "entry_rule": (
            entry_tx.get("entry_rule")
            or entry_override.get("entry_rule")
            or (entry_trigger or {}).get("entry_rule")
        ),
        "planned_exit_rule": planned_exit_rule,
        "executed_exit_rule": executed_exit_rule,
        "hold_rule": (
            entry_tx.get("hold_rule")
            or entry_override.get("hold_rule")
            or (entry_trigger or {}).get("hold_rule")
        ),
        "entry_timeframe": (
            entry_tx.get("timeframe")
            or entry_override.get("timeframe")
            or (entry_trigger or {}).get("timeframe")
        ),
        "exit_timeframe": (
            exit_tx.get("timeframe")
            or exit_override.get("timeframe")
            or (exit_trigger or {}).get("timeframe")
        ),
    }


async def _resolve_trade_set_preview_fallback(
    client: Any,
    session_id: str,
    symbol: str,
    as_of_iso: str,
    entry_price: float,
    trigger_meta: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Best-effort stop/target preview backfill using session structure history."""
    if not trigger_meta or entry_price <= 0:
        return {}

    from tools.paper_trading import _compute_trigger_preview

    handle = client.get_workflow_handle(session_id)
    try:
        structure_payload = await asyncio.wait_for(
            handle.query("get_structure_snapshots", symbol, as_of_iso),
            timeout=2.5,
        )
    except Exception as err:
        logger.debug("Failed to query structure snapshot for %s @ %s: %s", symbol, as_of_iso, err)
        return {}

    indicators = {}
    if isinstance(structure_payload, dict):
        indicators = structure_payload.get("indicators") or {}
    indicator_dict = indicators.get(symbol.upper()) or indicators.get(symbol)
    if not isinstance(indicator_dict, dict):
        return {}

    try:
        preview = _compute_trigger_preview(dict(trigger_meta), indicator_dict, entry_price)
    except Exception as err:
        logger.debug("Failed to compute trigger preview fallback for %s: %s", symbol, err)
        return {}

    if not isinstance(preview, dict):
        return {}
    return {
        "stop_price_abs": preview.get("stop_price"),
        "target_price_abs": preview.get("target_price"),
        "planned_rr": preview.get("rr_ratio"),
    }


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

        await _verify_session_exists(client, session_id)

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
        hold_overrides = _build_hold_overrides_from_order_events(session_id)
        entry_overrides = _build_entry_overrides_from_order_events(session_id)
        exit_overrides = _build_exit_overrides_from_order_events(session_id)
        trigger_catalog = await _build_trade_set_trigger_catalog(client, session_id)

        def _ts_to_iso(raw) -> str:
            if isinstance(raw, (int, float)):
                ts_sec = raw / 1000.0 if raw > 1e12 else raw
                return datetime.fromtimestamp(ts_sec, tz=_tz.utc).isoformat()
            return str(raw)

        # Pair BUY → SELL for the same symbol using a FIFO stack per symbol
        open_entries: Dict[str, List[Dict]] = {}  # symbol → list of open entry dicts
        closed_trade_index: Dict[str, int] = {}
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
                pair_idx = closed_trade_index.get(symbol, 0)
                closed_trade_index[symbol] = pair_idx + 1

                entry_price = float(entry.get("price", entry.get("fill_price", 0)))
                entry_qty = float(entry.get("qty", entry.get("quantity", 0)))
                entry_ts = entry.get("timestamp", 0)
                entry_fee = float(entry.get("fee", 0) or 0)
                entry_override = {}
                symbol_entry_overrides = entry_overrides.get(symbol)
                if symbol_entry_overrides and pair_idx < len(symbol_entry_overrides):
                    entry_override = symbol_entry_overrides[pair_idx]
                exit_override = {}
                symbol_exit_overrides = exit_overrides.get(symbol)
                if symbol_exit_overrides and pair_idx < len(symbol_exit_overrides):
                    exit_override = symbol_exit_overrides[pair_idx]

                used_qty = min(qty, entry_qty)
                gross_pnl = (price - entry_price) * used_qty
                total_fee = fee + entry_fee
                net_pnl = gross_pnl - total_fee
                pnl_pct = (gross_pnl / (entry_price * used_qty) * 100) if entry_price > 0 else 0.0

                entry_ts_sec = entry_ts if entry_ts < 1e12 else entry_ts / 1000.0
                exit_ts_sec = ts_raw if ts_raw < 1e12 else ts_raw / 1000.0
                hold_minutes = (exit_ts_sec - entry_ts_sec) / 60.0
                symbol_overrides = hold_overrides.get(symbol)
                if symbol_overrides and pair_idx < len(symbol_overrides):
                    hold_minutes = symbol_overrides[pair_idx]

                # R-per-hour and R-achieved: net_pnl / initial_risk
                stop_px = entry.get("stop_price_abs")
                if stop_px is None:
                    stop_px = entry_override.get("stop_price_abs")
                target_px = entry.get("target_price_abs")
                if target_px is None:
                    target_px = entry_override.get("target_price_abs")
                rule_fields = _resolve_trade_set_rule_fields(
                    entry,
                    tx,
                    entry_override,
                    exit_override,
                    trigger_catalog,
                )
                entry_trigger_meta = trigger_catalog.get(str(rule_fields["entry_trigger"] or ""))
                if stop_px is None or target_px is None:
                    preview_fallback = await _resolve_trade_set_preview_fallback(
                        client,
                        session_id,
                        symbol,
                        _ts_to_iso(entry_ts),
                        entry_price,
                        entry_trigger_meta,
                    )
                    if stop_px is None:
                        stop_px = preview_fallback.get("stop_price_abs")
                    if target_px is None:
                        target_px = preview_fallback.get("target_price_abs")
                r_achieved = None
                r_per_hour = None
                r_planned = entry.get("planned_rr")
                if r_planned is None:
                    r_planned = entry_override.get("planned_rr")
                if stop_px and entry_price > 0:
                    initial_risk_per_unit = abs(entry_price - float(stop_px))
                    if initial_risk_per_unit > 0:
                        initial_risk = initial_risk_per_unit * used_qty
                        r_achieved = round(net_pnl / initial_risk, 3)
                        if hold_minutes > 0:
                            r_per_hour = round(r_achieved / (hold_minutes / 60.0), 4)
                        # Planned R:R from target vs stop
                        if target_px and r_planned is None:
                            r_planned = round(abs(float(target_px) - entry_price) / initial_risk_per_unit, 2)

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
                    "entry_trigger": rule_fields["entry_trigger"],
                    "exit_trigger": rule_fields["exit_trigger"],
                    "category": rule_fields["category"],
                    "entry_timeframe": rule_fields["entry_timeframe"],
                    "exit_timeframe": rule_fields["exit_timeframe"],
                    "entry_rule": rule_fields["entry_rule"],
                    "planned_exit_rule": rule_fields["planned_exit_rule"],
                    "executed_exit_rule": rule_fields["executed_exit_rule"],
                    "hold_rule": rule_fields["hold_rule"],
                    "winner": net_pnl > 0,
                    "stop_price_abs": stop_px,
                    "target_price_abs": target_px,
                    "r_achieved": r_achieved,
                    "r_planned": round(float(r_planned), 2) if r_planned is not None else None,
                    "r_per_hour": r_per_hour,
                    "target_source": entry.get("target_source"),
                    "target_structural_kind": entry.get("target_structural_kind"),
                    "stop_source": entry.get("stop_source"),
                    "estimated_bars_to_resolution": (
                        entry.get("estimated_bars_to_resolution")
                        if entry.get("estimated_bars_to_resolution") is not None
                        else entry_override.get("estimated_bars_to_resolution")
                    ),
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


class PaperTradingMetrics(BaseModel):
    """Session-level aggregated performance metrics."""

    session_id: str
    total_trades: int
    win_rate_pct: float
    total_net_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    avg_r_per_trade: float
    median_hold_minutes: float
    equity_return_pct: float
    policy_skips: int
    validation_rejections: int


@router.get("/sessions/{session_id}/metrics", response_model=PaperTradingMetrics)
async def get_session_metrics(session_id: str):
    """Get aggregated session-level performance metrics.

    Computes win/loss stats from completed trade sets, draws equity return from
    the ledger, and counts policy gate skips and validation rejections from events.
    """
    from ops_api.temporal_client import get_temporal_client
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from ops_api.event_store import EventStore
    from temporalio.service import RPCError, RPCStatusCode
    from datetime import datetime, timezone as _tz

    try:
        client = await get_temporal_client()

        # Verify session exists without consuming workflow query slots.
        await _verify_session_exists(client, session_id)

        # Fetch all transactions (same as trade_sets endpoint)
        ledger_workflow_id = _paper_ledger_workflow_id(session_id)
        ledger_handle = client.get_workflow_handle(ledger_workflow_id)
        try:
            transactions = await ledger_handle.query("get_transaction_history", {"limit": 2000})
        except RPCError as err:
            if err.status != RPCStatusCode.NOT_FOUND:
                raise
            legacy_ledger_handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
            transactions = await legacy_ledger_handle.query("get_transaction_history", {"limit": 2000})

        # Fetch portfolio state for equity return calculation
        try:
            portfolio = await ledger_handle.query("get_portfolio_status")
        except Exception:
            portfolio = {}

        # Compute FIFO trade pairs (identical logic to get_trade_sets)
        txs = sorted(transactions, key=lambda x: x.get("timestamp", 0))
        hold_overrides = _build_hold_overrides_from_order_events(session_id)
        open_entries: Dict[str, List[Dict]] = {}
        closed_trade_index: Dict[str, int] = {}
        trade_sets: List[Dict[str, Any]] = []

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
                pair_idx = closed_trade_index.get(symbol, 0)
                closed_trade_index[symbol] = pair_idx + 1
                entry_price = float(entry.get("price", entry.get("fill_price", 0)))
                entry_qty = float(entry.get("qty", entry.get("quantity", 0)))
                entry_ts = entry.get("timestamp", 0)
                entry_fee = float(entry.get("fee", 0) or 0)
                used_qty = min(qty, entry_qty)
                gross_pnl = (price - entry_price) * used_qty
                total_fee = fee + entry_fee
                net_pnl = gross_pnl - total_fee
                entry_ts_sec = entry_ts if entry_ts < 1e12 else entry_ts / 1000.0
                exit_ts_sec = ts_raw if ts_raw < 1e12 else ts_raw / 1000.0
                hold_minutes = (exit_ts_sec - entry_ts_sec) / 60.0
                symbol_overrides = hold_overrides.get(symbol)
                if symbol_overrides and pair_idx < len(symbol_overrides):
                    hold_minutes = symbol_overrides[pair_idx]
                stop_px = entry.get("stop_price_abs")
                r_return = None
                if stop_px and entry_price > 0:
                    initial_risk = abs(entry_price - float(stop_px)) * used_qty
                    if initial_risk > 0:
                        r_return = net_pnl / initial_risk
                trade_sets.append({
                    "net_pnl": net_pnl,
                    "hold_minutes": hold_minutes,
                    "winner": net_pnl > 0,
                    "r_return": r_return,
                })

        total = len(trade_sets)
        wins = [t for t in trade_sets if t["winner"]]
        losses = [t for t in trade_sets if not t["winner"]]
        win_rate = len(wins) / total * 100 if total else 0.0
        total_net_pnl = sum(t["net_pnl"] for t in trade_sets)
        avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0.0
        gross_wins = sum(t["net_pnl"] for t in wins)
        gross_losses = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0
        r_values = [t["r_return"] for t in trade_sets if t["r_return"] is not None]
        avg_r = sum(r_values) / len(r_values) if r_values else 0.0
        hold_vals = sorted(t["hold_minutes"] for t in trade_sets)
        median_hold = hold_vals[len(hold_vals) // 2] if hold_vals else 0.0

        # Max drawdown from equity history (lightweight approximation)
        initial_cash = float(portfolio.get("initial_cash", 0) or 0)
        current_equity = float(
            portfolio.get("total_equity") or portfolio.get("total_portfolio_value", initial_cash) or initial_cash
        )
        equity_return = ((current_equity - initial_cash) / initial_cash * 100) if initial_cash > 0 else 0.0

        # Compute simple max drawdown from trade PnL sequence
        running = 0.0
        peak = 0.0
        max_dd_abs = 0.0
        for t in trade_sets:
            running += t["net_pnl"]
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd_abs:
                max_dd_abs = dd
        max_dd_pct = (max_dd_abs / initial_cash * 100) if initial_cash > 0 else 0.0

        # Count policy gate skips and validation rejections from event store
        event_store = EventStore()
        policy_skip_events = event_store.list_events_filtered(
            event_type="policy_loop_skipped", run_id=session_id, limit=10000
        )
        validation_rejected_events = event_store.list_events_filtered(
            event_type="plan_validation_rejected", run_id=session_id, limit=10000
        )

        return PaperTradingMetrics(
            session_id=session_id,
            total_trades=total,
            win_rate_pct=round(win_rate, 1),
            total_net_pnl=round(total_net_pnl, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(profit_factor, 3),
            max_drawdown_pct=round(max_dd_pct, 2),
            avg_r_per_trade=round(avg_r, 4),
            median_hold_minutes=round(median_hold, 1),
            equity_return_pct=round(equity_return, 2),
            policy_skips=len(policy_skip_events),
            validation_rejections=len(validation_rejected_events),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session metrics: {e}", exc_info=True)
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


@router.get("/sessions/{session_id}/block-summary")
async def get_block_summary(session_id: str, limit: int = 500):
    """Get Phase 0 block reason summary for a paper trading session (R73).

    Returns counts of trade_blocked events broken down by block_class
    (risk_valid / quality_gate / infra / unknown) to support Phase 0 exit criteria.
    """
    from ops_api.event_store import EventStore
    from schemas.block_taxonomy import classify_block_reason, BlockClass

    try:
        store = EventStore()
        events = store.list_events_filtered(
            limit=limit,
            run_id=session_id,
            order="desc",
        )

        block_events = [e for e in events if e.type == "trade_blocked"]
        reason_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {c.value: 0 for c in BlockClass}

        for e in block_events:
            reason = e.payload.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            # Use stored block_class if present (R73), else classify on the fly
            block_class = e.payload.get("block_class") or classify_block_reason(reason).value
            class_counts[block_class] = class_counts.get(block_class, 0) + 1

        return {
            "session_id": session_id,
            "total_blocks": len(block_events),
            "block_class_counts": class_counts,
            "block_reason_counts": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
            # Phase 0 exit criteria fields
            "phase0_quality_gate_count": class_counts.get(BlockClass.QUALITY_GATE.value, 0),
            "phase0_infra_count": class_counts.get(BlockClass.INFRA.value, 0),
            "phase0_pass": (
                class_counts.get(BlockClass.QUALITY_GATE.value, 0) == 0
                and class_counts.get(BlockClass.INFRA.value, 0) == 0
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get block summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/structure")
async def get_structure_snapshot(
    session_id: str,
    symbol: Optional[str] = None,
    as_of: Optional[str] = None,
):
    """Get latest indicator snapshot(s) used for structure-aware planning."""
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    symbol_norm = symbol.upper() if symbol else None

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)

        # Prefer the richer time-aware query when available (selected-candle UI
        # lookup). Fall back to legacy latest-only queries for older sessions.
        try:
            structure_payload = await handle.query("get_structure_snapshots", args=[symbol_norm, as_of])
        except RPCError:
            structure_payload = None

        if isinstance(structure_payload, dict) and "indicators" in structure_payload:
            indicators = structure_payload.get("indicators")
            if not isinstance(indicators, dict):
                indicators = {}
            response: Dict[str, Any] = {
                "session_id": session_id,
                "count": len(indicators),
                "indicators": indicators,
            }
            # Optional metadata for selected-candle UI binding.
            for key in (
                "lookup_mode",
                "requested_as_of",
                "resolved_as_of",
                "resolved_as_of_by_symbol",
                "structure_snapshots",
                "structure_lookup_mode",
                "structure_resolved_as_of",
                "structure_resolved_as_of_by_symbol",
            ):
                if key in structure_payload:
                    response[key] = structure_payload.get(key)
            return response

        # Legacy fallback path (latest-only).
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

        response = {
            "session_id": session_id,
            "count": len(indicators),
            "indicators": indicators,
        }
        if as_of:
            response["lookup_mode"] = "latest_fallback"
            response["requested_as_of"] = as_of
        return response
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

    # Exchange symbol migrations / aliases (Coinbase migrated MATIC -> POL).
    symbol_aliases = {
        "MATIC-USD": "POL-USD",
        "MATIC/USD": "POL/USD",
    }
    symbol = symbol_aliases.get(symbol.upper(), symbol)

    # Map friendly pair format to ccxt symbol
    ccxt_symbol = symbol.replace("-", "/")
    # Map UI timeframe labels to ccxt timeframe strings
    tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "6h": "6h", "1d": "1d"}
    tf = timeframe.strip()
    ccxt_tf = tf_map.get(tf, "1m")
    ohlcv: List[List[float]] = []
    exchange_timeout_seconds = float(os.environ.get("PAPER_TRADING_CANDLE_FETCH_TIMEOUT_SECONDS", "10"))

    async def _fetch_with_timeout(fetch_tf: str, *, fetch_limit: int) -> List[List[float]]:
        return await asyncio.wait_for(
            exchange.fetch_ohlcv(ccxt_symbol, fetch_tf, limit=fetch_limit),
            timeout=exchange_timeout_seconds,
        )

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
            daily = await _fetch_with_timeout("1d", fetch_limit=lookback_days)
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
        else:
            ohlcv = await _fetch_with_timeout(ccxt_tf, fetch_limit=limit)
    except asyncio.TimeoutError:
        logger.error("Timed out fetching candles for %s (%s)", symbol, ccxt_tf)
        raise HTTPException(status_code=504, detail=f"Exchange timeout fetching candles for {symbol}")
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


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(status: Optional[str] = None, limit: int = 20):
    """List paper trading sessions.

    Note: This queries Temporal for workflow executions matching the paper-trading prefix.
    """
    from ops_api.temporal_client import get_temporal_client

    try:
        client = await get_temporal_client()

        # Query all paper-trading executions, then collapse to one canonical row
        # per session_id. This avoids showing:
        #  - session-scoped ledger workflows (suffix "-ledger")
        #  - historical continued-as-new executions for the same session_id
        # Temporal returns newest executions first; keep the first seen per id.
        query = 'WorkflowId STARTS_WITH "paper-trading-"'
        deduped: Dict[str, Dict[str, Any]] = {}
        async for workflow in client.list_workflows(query=query):
            workflow_id = workflow.id
            if not workflow_id or workflow_id.endswith("-ledger"):
                continue
            if workflow_id in deduped:
                continue
            deduped[workflow_id] = {
                "session_id": workflow_id,
                "status": workflow.status.name.lower() if workflow.status else "unknown",
                "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
                "close_time": workflow.close_time.isoformat() if workflow.close_time else None,
            }

        sessions = list(deduped.values())
        if status == "running":
            sessions = [s for s in sessions if s["status"] == "running"]
        elif status == "stopped":
            sessions = [s for s in sessions if s["status"] != "running"]

        # Preserve newest-first ordering from Temporal scan.
        sessions = sessions[:limit]

        semaphore = asyncio.Semaphore(4)

        async def _attach_summary(session: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                summary = await _get_session_list_summary(client, session["session_id"])
            return {**session, "summary": summary}

        sessions_with_summaries = await asyncio.gather(*(_attach_summary(session) for session in sessions))
        return {"sessions": sessions_with_summaries, "count": len(sessions_with_summaries)}

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


@router.get("/sessions/{session_id}/world-state")
async def get_world_state(session_id: str):
    """Return the current WorldState for a paper trading session (R80).

    WorldState is the shared world model across Strategist, Judge, and Execution.
    Contains:
    - regime_fingerprint: current normalized regime vector
    - regime_trajectory: rolling history (velocity_scalar, stability_score)
    - judge_guidance: structured JudgeGuidanceVector (risk_multiplier, playbook_penalties, etc.)
    - confidence_calibration: per-dimension trust weights
    - policy_state: current FSM state
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError
    from tools.paper_trading import PaperTradingWorkflow

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)
        world_state = await handle.query(PaperTradingWorkflow.get_world_state)
        return {
            "session_id": session_id,
            "world_state": world_state,
            "has_judge_guidance": bool(world_state.get("judge_guidance")),
            "has_regime_fingerprint": bool(world_state.get("regime_fingerprint")),
        }
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get world state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/judge-guidance")
async def apply_judge_guidance(session_id: str, guidance: Dict[str, Any]):
    """Apply structured JudgeGuidanceVector to the WorldState (R80 ii-loop).

    Posts a JudgeGuidanceVector as a signal to the paper trading workflow.
    The guidance is applied deterministically — no LLM interpretation needed.

    Example payload:
      {
        "risk_multiplier": 0.7,
        "playbook_penalties": {"donchian_breakout": 0.0},
        "symbol_vetoes": ["ETH-USD"],
        "confidence_adjustments": {"regime_assessment": 0.6}
      }
    """
    from ops_api.temporal_client import get_temporal_client
    from temporalio.service import RPCError

    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(session_id)
        await handle.signal("apply_judge_guidance", guidance)
        return {
            "session_id": session_id,
            "status": "applied",
            "risk_multiplier": guidance.get("risk_multiplier", 1.0),
        }
    except RPCError as e:
        if _is_not_found(e):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to apply judge guidance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
