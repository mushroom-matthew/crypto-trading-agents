import os
import json
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple
import logging
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from agents.utils import stream_chat_completion, check_and_process_feedback, tool_result_data
from mcp.types import CallToolResult, TextContent
from agents.context_manager import create_context_manager
from datetime import datetime, timedelta, timezone
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
    ScheduleIntervalSpec,
    RPCError,
    RPCStatusCode,
)
from tools.ensemble_nudge import EnsembleNudgeWorkflow
from agents.workflows.broker_agent_workflow import BrokerAgentWorkflow
from agents.execution_config import load_execution_gating_config
from agents.workflows.execution_agent_workflow import (
    ExecutionAgentWorkflow,
    ExecutionAgentState,
    should_call_llm,
)
from agents.workflows.strategy_spec_workflow import StrategySpecWorkflow
from tools.agent_logger import AgentLogger
from agents.constants import (
    ORANGE, PINK, RESET, DEFAULT_OPENAI_MODEL, 
    DEFAULT_TEMPORAL_ADDRESS, DEFAULT_TEMPORAL_NAMESPACE,
    DEFAULT_TASK_QUEUE, EXECUTION_WF_ID, NUDGE_SCHEDULE_ID,
EXECUTION_AGENT
)
from agents.logging_utils import setup_logging
from agents.temporal_utils import connect_temporal
from agents.langfuse_utils import init_langfuse
from agents.llm.client_factory import get_llm_client
from agents.event_emitter import emit_event
from tools.strategy_executor import evaluate_signals, TradeSignal
from tools.strategy_spec import PositionState, StrategySpec
from tools import execution_tools
from schemas.strategy_run import StrategyRun, StrategyRunConfig
from services.strategy_run_registry import strategy_run_registry

# Tools this agent is allowed to call
ALLOWED_TOOLS = {
    "place_mock_order",
    "get_historical_ticks",
    "get_portfolio_status",
    "get_user_preferences",
    "get_transaction_history",
    "get_performance_metrics",
    "get_risk_metrics",
    "list_technical_metrics",
    "compute_technical_metrics",
}

# Context management is now handled by the ContextManager class

logger = setup_logging(__name__)

init_langfuse()
openai_client = get_llm_client()

DEFAULT_STRATEGY_TIMEFRAME = os.environ.get("STRATEGY_TIMEFRAME", "15m")
USE_LEGACY_LLM_EXECUTION = os.environ.get("USE_LEGACY_LLM_EXECUTION", "0") == "1"
EXECUTION_STRATEGY_RUN_ID = os.environ.get("EXECUTION_STRATEGY_RUN_ID", "live-strategy-run")
EXECUTION_PLAN_MAX_TRADES = int(os.environ.get("EXECUTION_PLAN_MAX_TRADES", "10"))

SYSTEM_PROMPT = (
    "You are an autonomous portfolio management agent that analyzes market data and executes trading decisions "
    "based on user preferences and risk profile. You operate independently without human confirmation.\n\n"
    
    "AUTONOMOUS OPERATION:\n"
    "• Make all trading decisions independently - no human approval required\n"
    "• Execute orders immediately when your analysis indicates action\n"
    "• Never ask for confirmation or present multiple choice options\n"
    "• Report what you decided and executed, not what you recommend\n\n"
    
    "DATA ANALYSIS:\n"
    "• Combine new tick data with your conversation history for complete market picture\n"
    "• Analyze price momentum, trends, support/resistance, and volume patterns\n"
    "• Consider current portfolio, performance metrics, and risk exposure\n"
    "• Apply user risk tolerance and trading style preferences\n"
    "• CRITICAL: Always extract and use the LATEST market prices from the historical tick data\n"
    "• NEVER use example prices or hardcoded values - only use real-time data provided to you\n\n"
    
    "RISK MANAGEMENT & FINANCIAL DISCIPLINE:\n"
    "• ALWAYS use the provided portfolio data to know exact cash available and current positions\n"
    "• Calculate total cost of ALL BUY orders (qty * price * 1.02 for slippage) and ensure it's under available cash\n"
    "• Never place orders that would exceed available funds - be conservative with position sizing\n"
    "• For batch orders, sum up all BUY order costs and verify total is within the cash shown in portfolio data\n"
    "• Apply user risk tolerance and trading style preferences when making decisions\n"
    "• Only SELL positions you actually own - check current holdings in the provided portfolio data\n"
    "• ALL ORDERS MUST BE TYPE 'market' - NEVER use 'limit' orders\n"
    "• Leave some cash buffer (5-10%) for market volatility and slippage\n"
    "• Pay close attention to the 'cash' field in the portfolio data provided to you\n\n"
    
    "ORDER FORMAT:\n"
    "CRITICAL: Always use FULL symbol names exactly as provided (e.g., 'BTC/USD', 'ETH/USD', 'DOGE/USD')\n"
    "CRITICAL: NEVER use the example prices below - ALWAYS use current market prices from the provided data\n\n"
    "Single order (array with one order) - EXAMPLE FORMAT ONLY:\n"
    '{"orders": [{"symbol": "BTC/USD", "side": "BUY", "qty": 0.001, "price": CURRENT_MARKET_PRICE, "type": "market"}]}\n\n'
    
    "Multiple orders (array with multiple orders) - EXAMPLE FORMAT ONLY:\n"
    '{"orders": [{"symbol": "BTC/USD", "side": "BUY", "qty": 0.001, "price": CURRENT_MARKET_PRICE, "type": "market"}, '
    '{"symbol": "ETH/USD", "side": "SELL", "qty": 0.1, "price": CURRENT_MARKET_PRICE, "type": "market"}]}\n\n'
    
    "Execute trades decisively using `place_mock_order`. Report completed actions and reasoning."
)


async def _watch_symbols(client: Client, symbols: Set[str]) -> None:
    """Poll broker workflow for selected symbols."""
    wf_id = os.environ.get("BROKER_WF_ID", "broker-agent")
    while True:
        try:
            handle = client.get_workflow_handle(wf_id)
            syms: list[str] = await handle.query("get_symbols")
            new_set = set(syms)
            if new_set != symbols:
                symbols.clear()
                symbols.update(new_set)
                print(f"[ExecutionAgent] Active symbols updated: {sorted(symbols)}")
                
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                try:
                    await client.start_workflow(
                        BrokerAgentWorkflow.run,
                        id=wf_id,
                        task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
                    )
                    print("[ExecutionAgent] Broker workflow started")
                except Exception as exc:
                    print(f"[ExecutionAgent] Failed to start broker workflow: {exc}")
            else:
                print(f"[ExecutionAgent] Failed to query broker workflow: {err}")
        except Exception as exc:
            print(f"[ExecutionAgent] Error watching symbols: {exc}")
        await asyncio.sleep(1)


async def _watch_user_preferences(client: Client, current_preferences: dict, conversation: list) -> None:
    """Poll execution agent workflow for user preference updates."""
    wf_id = "execution-agent"
    while True:
        try:
            handle = client.get_workflow_handle(wf_id)
            prefs = await handle.query("get_user_preferences")
            
            # Check if preferences have changed
            if prefs != current_preferences:
                current_preferences.clear()
                current_preferences.update(prefs)
                print(f"[ExecutionAgent] ✅ User preferences updated: risk_tolerance={prefs.get('risk_tolerance', 'moderate')}, style={prefs.get('trading_style', 'unknown')}")
                
                # The judge agent will handle system prompt updates directly
                
        except Exception as exc:
            # Silently continue if execution agent workflow not found or other issues
            pass
        
        await asyncio.sleep(2)  # Check every 2 seconds


async def _watch_system_prompt_updates(client: Client, conversation: list) -> None:
    """Watch for system prompt updates from the judge agent."""
    wf_id = "execution-agent"
    last_prompt = ""
    
    while True:
        try:
            handle = client.get_workflow_handle(wf_id)
            current_prompt = await handle.query("get_system_prompt")
            
            # Check if prompt has changed
            if current_prompt and current_prompt != last_prompt:
                if conversation and conversation[0]["role"] == "system":
                    conversation[0]["content"] = current_prompt
                    print(f"[ExecutionAgent] 🔄 System prompt updated by judge (length: {len(current_prompt)} chars)")
                    last_prompt = current_prompt
                
        except Exception as exc:
            # Silently continue if execution agent workflow not found or other issues
            pass
        
        await asyncio.sleep(5)  # Check every 5 seconds


async def _stream_nudges(client: Client) -> AsyncIterator[int]:
    """Yield timestamps from execution-agent workflow nudges."""
    wf_id = os.environ.get("EXECUTION_WF_ID", "execution-agent")
    cursor = 0
    while True:
        try:
            handle = client.get_workflow_handle(wf_id)
            nudges: list[int] = await handle.query("get_nudges")
            for ts in nudges:
                if ts > cursor:
                    cursor = ts
                    yield ts
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                try:
                    await client.start_workflow(
                        ExecutionAgentWorkflow.run,
                        id=wf_id,
                        task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
                    )
                    print("[ExecutionAgent] Execution workflow started")
                except Exception as exc:
                    print(f"[ExecutionAgent] Failed to start execution workflow: {exc}")
            else:
                print(f"[ExecutionAgent] Failed to query execution workflow: {err}")
        except Exception as exc:
            print(f"[ExecutionAgent] Error streaming nudges: {exc}")
        await asyncio.sleep(1)


async def _ensure_schedule(client: Client) -> None:
    """Create the nudge schedule if it doesn't already exist."""
    handle = client.get_schedule_handle(NUDGE_SCHEDULE_ID)
    try:
        await handle.describe()
        return
    except RPCError as err:
        if err.status != RPCStatusCode.NOT_FOUND:
            raise

    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow=EnsembleNudgeWorkflow.run,
            args=[],
            id="ensemble-nudge-wf",
            task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
        ),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(seconds=25))]),
    )
    await client.create_schedule(NUDGE_SCHEDULE_ID, schedule)
    await client.get_schedule_handle(NUDGE_SCHEDULE_ID).trigger()


async def _ensure_strategy_store(client: Client) -> None:
    handle = client.get_workflow_handle("strategy-spec-store")
    try:
        await handle.describe()
    except RPCError as err:
        if err.status == RPCStatusCode.NOT_FOUND:
            await client.start_workflow(
                StrategySpecWorkflow.run,
                id="strategy-spec-store",
                task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
            )
            logger.info("[ExecutionAgent] StrategySpec store workflow started")
        else:
            raise


def _timeframe_to_seconds(timeframe: str) -> int:
    if not timeframe:
        return 900
    units = {"m": 60, "h": 3600, "d": 86400}
    suffix = timeframe[-1]
    try:
        value = int(timeframe[:-1])
        return value * units.get(suffix, 60)
    except (ValueError, KeyError):
        return 900


def _build_symbol_features(symbol_data: List[Dict[str, Any]], timeframe: str) -> Optional[Dict[str, Any]]:
    if not symbol_data:
        return None
    closes: List[float] = []
    timestamps: List[int] = []
    for tick in symbol_data:
        price = tick.get("price")
        if price is None:
            continue
        closes.append(float(price))
        ts = tick.get("ts") or tick.get("timestamp")
        if ts:
            timestamps.append(int(ts))
    if not closes:
        return None
    latest_ts = timestamps[-1] if timestamps else None
    return {
        "close_prices": closes,
        "price": closes[-1],
        "timestamp": latest_ts,
        "timeframe_seconds": _timeframe_to_seconds(timeframe),
        "peak_price": max(closes),
        "trough_price": min(closes),
    }


def _build_position_state(
    symbol: str, portfolio_snapshot: Dict[str, Any]
) -> PositionState:
    positions = portfolio_snapshot.get("positions", {}) or {}
    entry_prices = portfolio_snapshot.get("entry_prices", {}) or {}
    qty = float(positions.get(symbol, 0.0) or 0.0)
    entry_price = entry_prices.get(symbol)
    side: Optional[str] = None
    if qty > 0:
        side = "buy"
    elif qty < 0:
        side = "sell"
    return PositionState(
        market=symbol,
        side=side,
        qty=abs(qty),
        avg_entry_price=float(entry_price) if entry_price is not None else None,
    )


def _ensure_execution_run(symbols: Set[str]) -> StrategyRun:
    ordered = sorted(symbols) or []
    try:
        run = strategy_run_registry.get_strategy_run(EXECUTION_STRATEGY_RUN_ID)
    except KeyError:
        config = StrategyRunConfig(
            symbols=ordered,
            timeframes=[DEFAULT_STRATEGY_TIMEFRAME],
            history_window_days=30,
            plan_cadence_hours=24,
        )
        run = strategy_run_registry.create_strategy_run(config=config, run_id=EXECUTION_STRATEGY_RUN_ID)
        return run
    if ordered and ordered != run.config.symbols:
        run.config.symbols = ordered
        run = strategy_run_registry.update_strategy_run(run)
    return run


def _spec_category(spec: StrategySpec) -> str:
    mapping = {
        "trend": "trend_continuation",
        "mean_revert": "mean_reversion",
        "breakout": "volatility_breakout",
    }
    return mapping.get(spec.mode, "other")


def _plan_payload_for_spec(run: StrategyRun, spec: StrategySpec) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    risk = spec.risk
    return {
        "plan_id": spec.strategy_id,
        "run_id": run.run_id,
        "generated_at": now.isoformat(),
        "valid_until": (now + timedelta(hours=24)).isoformat(),
        "global_view": f"Spec-driven plan for {spec.market}",
        "regime": "mixed",
        "triggers": [
            {
                "id": spec.strategy_id,
                "symbol": spec.market,
                "direction": "long",
                "timeframe": spec.timeframe,
                "entry_rule": "True",
                "exit_rule": "False",
                "category": _spec_category(spec),
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": max(risk.risk_per_trade_fraction * 100.0, 0.0),
            "max_symbol_exposure_pct": max(risk.max_fraction_of_balance * 100.0, 0.0),
            "max_portfolio_exposure_pct": min(100.0, risk.max_fraction_of_balance * 100.0),
            "max_daily_loss_pct": risk.max_drawdown_pct,
        },
        "sizing_rules": [
            {
                "symbol": spec.market,
                "sizing_mode": "fixed_fraction",
                "target_risk_pct": max(risk.risk_per_trade_fraction * 100.0, 0.0),
            }
        ],
        "max_trades_per_day": EXECUTION_PLAN_MAX_TRADES,
        "min_trades_per_day": None,
        "allowed_symbols": run.config.symbols or [spec.market],
        "allowed_directions": ["long"],
        "allowed_trigger_categories": [_spec_category(spec)],
    }


def _compiled_payload_for_spec(plan_payload: Dict[str, Any]) -> Dict[str, Any]:
    trigger = plan_payload["triggers"][0]
    return {
        "plan_id": plan_payload["plan_id"],
        "run_id": plan_payload["run_id"],
        "triggers": [
            {
                "trigger_id": trigger["id"],
                "symbol": trigger["symbol"],
                "direction": trigger["direction"],
                "category": trigger.get("category"),
                "entry": {"source": "True", "normalized": "True"},
                "exit": None,
            }
        ],
    }


def _should_execute_signal(run: StrategyRun, spec: StrategySpec, ts: int) -> Tuple[bool, str]:
    if spec.strategy_id is None:
        return True, ""
    plan_payload = _plan_payload_for_spec(run, spec)
    compiled_payload = _compiled_payload_for_spec(plan_payload)
    judge_payload = run.latest_judge_feedback.model_dump() if run.latest_judge_feedback else None
    events = [
        {
            "trigger_id": spec.strategy_id,
            "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
        }
    ]
    result = execution_tools.run_live_step_tool(
        run.run_id,
        plan_payload,
        compiled_payload,
        events,
        judge_feedback_payload=judge_payload,
    )
    event_list = result.get("events", [])
    if not event_list:
        return True, ""
    outcome = event_list[-1]
    if outcome.get("action") == "executed":
        return True, ""

    # Emit a durable block reason event for Ops visibility
    try:
        asyncio.create_task(
            emit_event(
                "trade_blocked",
                {
                    "reason": outcome.get("reason", "blocked"),
                    "trigger_id": outcome.get("trigger_id"),
                    "symbol": outcome.get("symbol", "unknown"),
                    "side": outcome.get("side", "unknown"),
                    "detail": outcome.get("detail", ""),
                },
                source="execution_agent",
                run_id=run.run_id,
                correlation_id=outcome.get("trigger_id"),
            )
        )
    except Exception as e:
        logger.warning("Failed to emit trade_blocked event: %s", e)

    return False, outcome.get("reason", "blocked")


def _build_order_payload(
    symbol: str,
    signal: TradeSignal,
    price: float,
    cash_available: float,
    position_state: PositionState,
) -> Optional[Dict[str, Any]]:
    if signal.side == "buy":
        allocation = cash_available * signal.size_fraction
        if allocation <= 0 or price <= 0:
            return None
        qty = allocation / price
        return {
            "orders": [
                {
                    "symbol": symbol,
                    "side": "BUY",
                    "qty": round(qty, 6),
                    "type": "market",
                    "price": price,
                    "allow_auto_execute": True,
                }
            ]
        }
    if signal.side == "sell":
        # Shorting not supported yet
        logger.info("[ExecutionAgent] Sell signal ignored (shorting disabled)")
        return None
    if signal.side == "close":
        if position_state.is_flat():
            return None
        side = "SELL" if position_state.side == "buy" else "BUY"
        return {
            "orders": [
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": round(position_state.qty, 6),
                    "type": "market",
                    "price": price,
                    "allow_auto_execute": True,
                }
            ]
        }
    return None


async def _execute_strategy_specs(
    temporal: Client,
    symbols: Set[str],
    historical_ticks: Dict[str, Any],
    portfolio_snapshot: Dict[str, Any],
    session: ClientSession,
    agent_logger: AgentLogger,
    ts: int,
    collected_data: Dict[str, Any],
    latest_processed_ts: Optional[int],
) -> None:
    if not symbols:
        return
    run = _ensure_execution_run(symbols)
    try:
        strategy_handle = temporal.get_workflow_handle("strategy-spec-store")
        await strategy_handle.describe()
    except Exception as exc:
        logger.info("[ExecutionAgent] Strategy spec store unavailable: %s", exc)
        return

    cash_available = float(portfolio_snapshot.get("cash", 0.0) or 0.0)
    orders_executed: List[Dict[str, Any]] = []

    for symbol in sorted(symbols):
        spec_data = await strategy_handle.query("get_strategy_spec", symbol, None)
        if not spec_data:
            logger.info("[ExecutionAgent] No StrategySpec configured for %s", symbol)
            continue
        try:
            spec = StrategySpec.model_validate(spec_data)
        except Exception as exc:
            logger.warning("[ExecutionAgent] Invalid StrategySpec for %s: %s", symbol, exc)
            continue
        if spec.is_expired():
            logger.info("[ExecutionAgent] StrategySpec expired for %s", symbol)
            continue
        if not spec.allow_auto_execute:
            logger.info("[ExecutionAgent] StrategySpec disabled for auto execution (%s)", spec.strategy_id)
            continue
        symbol_ticks = historical_ticks.get(symbol)
        features = _build_symbol_features(symbol_ticks, spec.timeframe)
        if not features:
            logger.info("[ExecutionAgent] Missing features for %s", symbol)
            continue
        position_state = _build_position_state(symbol, portfolio_snapshot)
        signals = evaluate_signals(spec, features, position_state)
        if not signals:
            continue
        for signal in signals:
            payload = _build_order_payload(
                symbol,
                signal,
                features["price"],
                cash_available,
                position_state,
            )
            if not payload:
                continue
            if signal.side == "buy":
                allowed, skip_reason = _should_execute_signal(run, spec, ts)
                if not allowed:
                    logger.info(
                        "[ExecutionAgent] Skipping %s due to %s",
                        spec.strategy_id,
                        skip_reason or "limit",
                    )
                    continue
            if signal.side == "buy":
                spend = payload["orders"][0]["qty"] * features["price"]
                cash_available = max(cash_available - spend, 0.0)
            logger.info(
                "[ExecutionAgent] Strategy signal %s for %s (reason=%s)",
                signal.side,
                symbol,
                signal.reason,
            )
            result = await session.call_tool("place_mock_order", payload)
            result_data = tool_result_data(result)
            orders_executed.append(
                {
                    "symbol": symbol,
                    "signal": signal.model_dump(),
                    "order": payload,
                    "result": result_data,
                }
            )

            # Emit order_submitted and fill events
            try:
                order_payload = payload["orders"][0]
                correlation_id = f"{spec.strategy_id}-{symbol}-{ts.isoformat()}"

                # Emit order_submitted event
                asyncio.create_task(
                    emit_event(
                        "order_submitted",
                        {
                            "symbol": symbol,
                            "side": order_payload["side"],
                            "qty": order_payload["qty"],
                            "price": order_payload.get("price"),
                            "type": order_payload["type"],
                            "strategy_id": spec.strategy_id,
                            "signal_reason": signal.reason,
                        },
                        source="execution_agent",
                        run_id=run.run_id,
                        correlation_id=correlation_id,
                    )
                )

                # Emit fill event (for mock orders, fill is immediate)
                if result_data and isinstance(result_data, list) and len(result_data) > 0:
                    fill_data = result_data[0]
                    asyncio.create_task(
                        emit_event(
                            "fill",
                            {
                                "symbol": fill_data.get("symbol", symbol),
                                "side": fill_data.get("side", order_payload["side"]),
                                "qty": fill_data.get("qty", order_payload["qty"]),
                                "fill_price": fill_data.get("fill_price", order_payload.get("price")),
                                "cost": fill_data.get("cost", 0),
                                "strategy_id": spec.strategy_id,
                            },
                            source="execution_agent",
                            run_id=run.run_id,
                            correlation_id=correlation_id,
                        )
                    )
            except Exception as e:
                logger.warning("Failed to emit order/fill events: %s", e)
    if orders_executed:
        decisions = [
            {
                "action": "place_order",
                "details": entry["order"],
                "signal": entry["signal"],
            }
            for entry in orders_executed
        ]
        await agent_logger.log_decision(
            nudge_timestamp=ts,
            symbols=sorted(symbols),
            market_data={"historical_ticks": collected_data.get("historical_ticks")},
            portfolio_data=collected_data.get("portfolio_status"),
            user_preferences=collected_data.get("user_preferences"),
            decisions={
                "orders_placed": len(decisions),
                "decisions": decisions,
                "hold_decisions": max(len(symbols) - len(decisions), 0),
            },
            reasoning="Deterministic strategy execution",
            performance_metrics=collected_data.get("performance_metrics"),
            risk_metrics=collected_data.get("risk_metrics"),
            latest_processed_timestamp=latest_processed_ts,
        )
        for decision in decisions:
            agent_logger.log_action(
                action_type="order_placement",
                details=decision["details"],
                nudge_timestamp=ts,
            )


async def run_execution_agent(server_url: str = "http://localhost:8080") -> None:
    """Run the execution agent and act on scheduled nudges."""
    base_url = server_url.rstrip("/")
    mcp_url = base_url + "/mcp/"

    temporal = await connect_temporal()
    execution_handle = temporal.get_workflow_handle(EXECUTION_WF_ID)
    gating_config = load_execution_gating_config()
    
    # Initialize context manager
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    context_manager = create_context_manager(model=model, openai_client=openai_client)
    symbols: Set[str] = set()
    current_preferences: dict = {}
    _symbol_task = asyncio.create_task(_watch_symbols(temporal, symbols))
    await _ensure_schedule(temporal)
    await _ensure_strategy_store(temporal)

    async with streamablehttp_client(mcp_url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            all_tools = tools_resp.tools
            tools = [t for t in all_tools if t.name in ALLOWED_TOOLS]
            
            # Get current system prompt from workflow, fallback to default
            try:
                current_prompt = await execution_handle.query("get_system_prompt")
                if not current_prompt:
                    current_prompt = SYSTEM_PROMPT
                    # Initialize workflow with default prompt
                    await execution_handle.signal("update_system_prompt", SYSTEM_PROMPT)
                print(f"[ExecutionAgent] Using system prompt from workflow (length: {len(current_prompt)} chars)")
            except Exception as exc:
                current_prompt = SYSTEM_PROMPT
                print(f"[ExecutionAgent] Using fallback system prompt: {exc}")
            
            conversation = [{"role": "system", "content": current_prompt}]
            print(
                "[ExecutionAgent] Connected to MCP server with tools:",
                [t.name for t in tools],
            )

            # Start watching for user preference updates
            _preferences_task = asyncio.create_task(_watch_user_preferences(temporal, current_preferences, conversation))
            
            # Start watching for system prompt updates from judge
            _prompt_task = asyncio.create_task(_watch_system_prompt_updates(temporal, conversation))

            # Track latest processed timestamp for data continuity
            latest_processed_ts = None
            
            # Initialize agent logger
            agent_logger = AgentLogger("execution_agent", temporal)
            
            async for ts in _stream_nudges(temporal):
                if not symbols:
                    continue
                print(f"[ExecutionAgent] Nudge @ {ts} for {sorted(symbols)}")
                
                # Check for any pending user feedback before processing
                await check_and_process_feedback(
                    temporal, 
                    "execution-agent", 
                    conversation=conversation,
                    agent_name="ExecutionAgent"
                )
                
                # ===============================
                # DETERMINISTIC DATA COLLECTION PHASE (PARALLEL)
                # ===============================
                print(f"[ExecutionAgent] Starting mandatory data collection (parallel)...")
                
                # Determine since_ts for incremental data fetching
                if latest_processed_ts is not None:
                    since_ts = latest_processed_ts
                    print(f"[ExecutionAgent] Fetching NEW ticks since timestamp: {since_ts}")
                else:
                    since_ts = 0  # Get all historical data on first run
                    print(f"[ExecutionAgent] First run - fetching all available historical data (since_ts=0)")
                
                # Start all data collection tasks in parallel
                tasks = [
                    session.call_tool("get_historical_ticks", {
                        "symbols": sorted(symbols),
                        "since_ts": since_ts  # Incremental fetching to avoid context bloat
                    }),
                    session.call_tool("get_portfolio_status", {}),
                    session.call_tool("get_user_preferences", {}),
                    session.call_tool("get_performance_metrics", {}),
                    session.call_tool("get_risk_metrics", {})
                ]
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                historical_data, portfolio_data, user_preferences, performance_metrics, risk_metrics = results
                
                print(f"[ExecutionAgent] ✓ Collected all data in parallel")
                
                # Extract and update latest processed timestamp for next cycle
                historical_ticks_data = tool_result_data(historical_data)
                latest_symbol_context: Dict[str, Dict[str, Any]] = {}
                
                if historical_ticks_data and isinstance(historical_ticks_data, dict):
                    # Find the latest timestamp from all symbols' tick data
                    max_timestamp = 0
                    new_tick_count = 0
                    
                    for symbol, symbol_data in historical_ticks_data.items():
                        if isinstance(symbol_data, list) and symbol_data:
                            new_tick_count += len(symbol_data)
                            # Get the latest timestamp from this symbol's ticks
                            for tick in symbol_data:
                                if isinstance(tick, dict):
                                    # Try both 'ts' and 'timestamp' fields
                                    tick_ts = tick.get('ts', 0) or tick.get('timestamp', 0)
                                    if tick_ts > 0:
                                        max_timestamp = max(max_timestamp, tick_ts)
                                        latest_price = tick.get("price")
                                        if latest_price is not None:
                                            prev_ts = latest_symbol_context.get(symbol, {}).get("timestamp", -1)
                                            if tick_ts >= prev_ts:
                                                latest_symbol_context[symbol] = {
                                                    "price": float(latest_price),
                                                    "timestamp": tick_ts,
                                                }
                    
                    if max_timestamp > 0:
                        latest_processed_ts = max_timestamp
                        print(f"[ExecutionAgent] Received {new_tick_count} new ticks (latest timestamp: {latest_processed_ts})")
                    else:
                        print(f"[ExecutionAgent] Warning: No valid timestamps found in tick data")
                
                # Compile all data for LLM analysis
                collected_data = {
                    "nudge_timestamp": ts,
                    "symbols": sorted(symbols),
                    "historical_ticks": tool_result_data(historical_data),
                    "portfolio_status": tool_result_data(portfolio_data),
                    "user_preferences": tool_result_data(user_preferences),
                    "performance_metrics": tool_result_data(performance_metrics),
                    "risk_metrics": tool_result_data(risk_metrics)
                }

                portfolio_snapshot = (
                    collected_data.get("portfolio_status")
                    if isinstance(collected_data.get("portfolio_status"), dict)
                    else {}
                )

                if not USE_LEGACY_LLM_EXECUTION:
                    await _execute_strategy_specs(
                        temporal,
                        symbols,
                        historical_ticks_data or {},
                        portfolio_snapshot or {},
                        session,
                        agent_logger,
                        ts,
                        collected_data,
                        latest_processed_ts,
                    )
                    continue
                
                # ===============================
                # LLM GATING
                # ===============================
                try:
                    raw_state = await execution_handle.query("get_execution_state")
                    execution_state = ExecutionAgentState.from_dict(raw_state)
                except Exception as exc:
                    logger.warning(
                        "[ExecutionAgent] Failed to load execution state: %s", exc
                    )
                    execution_state = ExecutionAgentState()

                should_invoke_llm = True
                gating_messages: list[str] = []
                if latest_symbol_context:
                    should_invoke_llm = False
                    now_fallback = datetime.now(timezone.utc)
                    for symbol in sorted(symbols):
                        symbol_ctx = latest_symbol_context.get(symbol)
                        if not symbol_ctx:
                            gating_messages.append(f"{symbol}:NO_PRICE_DATA")
                            should_invoke_llm = True
                            continue
                        tick_ts = symbol_ctx.get("timestamp")
                        tick_dt = (
                            datetime.fromtimestamp(tick_ts, timezone.utc)
                            if tick_ts
                            else now_fallback
                        )
                        state = execution_state.get_symbol_state(symbol)
                        call_llm, reason = should_call_llm(
                            symbol_ctx["price"],
                            tick_dt,
                            state,
                            gating_config,
                            logger,
                        )
                        gating_messages.append(f"{symbol}:{reason}")
                        if call_llm:
                            should_invoke_llm = True
                else:
                    logger.info(
                        "[ExecutionAgent] No price context available, defaulting to CALL_LLM"
                    )

                if not should_invoke_llm:
                    logger.info(
                        "[ExecutionAgent] Skipping LLM call (reasons=%s)",
                        ", ".join(gating_messages),
                    )
                    continue

                print(f"[ExecutionAgent] Data collection complete. Starting analysis phase...")
                
                # ===============================
                # LLM ANALYSIS PHASE
                # ===============================
                conversation.append(
                    {
                        "role": "user",
                        "content": json.dumps(collected_data),
                    }
                )
                # Only provide order execution tool to LLM (data collection is handled deterministically)
                order_tool = next((t for t in tools if t.name == "place_mock_order"), None)
                openai_tools = []
                if order_tool:
                    openai_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": order_tool.name,
                                "description": order_tool.description,
                                "parameters": order_tool.inputSchema,
                            },
                        }
                    ]
                while True:
                    try:
                        msg = stream_chat_completion(
                            openai_client,
                            model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                            messages=conversation,
                            tools=openai_tools,
                            tool_choice="auto",
                            prefix="[ExecutionAgent] Decision: ",
                            color=PINK,
                            reset=RESET,
                        )
                    except openai.OpenAIError as exc:
                        print(f"[ExecutionAgent] LLM request failed: {exc}")
                        # Keep system prompt and last user message on error
                        if len(conversation) >= 2:
                            conversation = [conversation[0], conversation[-1]]
                        break

                    if msg.get("tool_calls"):
                        conversation.append(
                            {
                                "role": msg.get("role", "assistant"),
                                "content": msg.get("content"),
                                "tool_calls": msg["tool_calls"],
                            }
                        )
                        # Process each tool call (batch orders are handled by MCP server)
                        for tool_call in msg["tool_calls"]:
                            func_name = tool_call["function"]["name"]
                            func_args = json.loads(
                                tool_call["function"].get("arguments") or "{}"
                            )
                            # Only allow order execution in LLM phase (data collection handled separately)
                            if func_name != "place_mock_order":
                                print(f"[ExecutionAgent] Tool not allowed in analysis phase: {func_name}")
                                continue
                            
                            # Log tool usage
                            order_count = len(func_args.get("orders", []))
                            print(
                                f"{ORANGE}[ExecutionAgent] Tool requested: {func_name} "
                                f"({order_count} order{'s' if order_count != 1 else ''}){RESET}"
                            )
                            
                            result = await session.call_tool(func_name, func_args)
                            result_data = tool_result_data(result)
                            
                            conversation.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": func_name,
                                    "content": json.dumps(result_data),
                                }
                            )
                        continue

                    if msg.get("function_call"):
                        conversation.append(
                            {
                                "role": msg.get("role", "assistant"),
                                "content": msg.get("content"),
                                "function_call": msg["function_call"],
                            }
                        )
                        func_name = msg["function_call"].get("name")
                        func_args = json.loads(
                            msg["function_call"].get("arguments") or "{}"
                        )
                        # Only allow order execution in LLM phase (data collection handled separately)
                        if func_name != "place_mock_order":
                            print(f"[ExecutionAgent] Tool not allowed in analysis phase: {func_name}")
                            continue
                        
                        # Log tool usage
                        order_count = len(func_args.get("orders", []))
                        print(
                            f"{ORANGE}[ExecutionAgent] Tool requested: {func_name} "
                            f"({order_count} order{'s' if order_count != 1 else ''}){RESET}"
                        )
                        
                        result = await session.call_tool(func_name, func_args)
                        result_data = tool_result_data(result)
                        
                        conversation.append(
                            {
                                "role": "function",
                                "name": func_name,
                                "content": json.dumps(result_data),
                            }
                        )
                        continue

                    assistant_reply = msg.get("content", "")
                    conversation.append(
                        {"role": "assistant", "content": assistant_reply}
                    )
                    
                    # Log comprehensive decision with all context
                    try:
                        # Extract decisions from the current cycle only (tool calls made in this response)
                        decisions_made = []
                        if msg.get("tool_calls"):
                            for tool_call in msg["tool_calls"]:
                                if tool_call["function"]["name"] == "place_mock_order":
                                    order_args = json.loads(tool_call["function"]["arguments"])
                                    decisions_made.append({
                                        "action": "place_order",
                                        "details": order_args
                                    })
                        
                        # Log the comprehensive decision
                        await agent_logger.log_decision(
                            nudge_timestamp=ts,
                            symbols=sorted(symbols),
                            market_data={"historical_ticks": tool_result_data(historical_data)},
                            portfolio_data=tool_result_data(portfolio_data),
                            user_preferences=tool_result_data(user_preferences),
                            decisions={
                                "orders_placed": len(decisions_made),
                                "decisions": decisions_made,
                                "hold_decisions": len(symbols) - len(decisions_made)
                            },
                            reasoning=assistant_reply,
                            performance_metrics=tool_result_data(performance_metrics),
                            risk_metrics=tool_result_data(risk_metrics),
                            latest_processed_timestamp=latest_processed_ts
                        )
                        
                        # Log each individual action
                        for decision in decisions_made:
                            agent_logger.log_action(
                                action_type="order_placement",
                                details=decision["details"],
                                nudge_timestamp=ts
                            )
                            
                    except Exception as log_error:
                        print(f"[ExecutionAgent] Failed to log decision: {log_error}")
                    
                    break

                # Persist execution state after a real LLM call
                state_updates: Dict[str, Dict[str, Any]] = {}
                now_ts = int(datetime.now(timezone.utc).timestamp())
                for symbol in sorted(symbols):
                    symbol_ctx = latest_symbol_context.get(symbol)
                    if not symbol_ctx:
                        continue
                    symbol_state = execution_state.get_symbol_state(symbol)
                    symbol_state.last_eval_price = symbol_ctx["price"]
                    symbol_state.last_eval_time = symbol_ctx.get("timestamp") or now_ts
                    if symbol_state.current_window_start is None:
                        symbol_state.current_window_start = symbol_state.last_eval_time
                        symbol_state.calls_in_current_window = 0
                    symbol_state.calls_in_current_window += 1
                    state_updates[symbol] = symbol_state.to_dict()

                if state_updates:
                    try:
                        await execution_handle.signal("update_execution_state", state_updates)
                    except Exception as exc:
                        logger.warning(
                            "[ExecutionAgent] Failed to update execution state: %s", exc
                        )

                # Manage conversation context intelligently
                conversation = await context_manager.manage_context(conversation)


if __name__ == "__main__":
    asyncio.run(
        run_execution_agent(os.environ.get("MCP_SERVER", "http://localhost:8080"))
    )
