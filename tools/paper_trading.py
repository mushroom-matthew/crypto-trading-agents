"""Paper trading workflow and activities for live strategy execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Literal
from uuid import uuid4

from pydantic import BaseModel
from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from agents.constants import STREAM_CONTINUE_EVERY, STREAM_HISTORY_LIMIT
    from agents.strategies.llm_client import LLMClient
    from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
    from agents.strategies.risk_engine import RiskEngine, RiskProfile
    from agents.strategies.trade_risk import TradeRiskEvaluator
    from agents.analytics import (
        IndicatorWindowConfig,
        build_asset_state,
        compute_indicator_snapshot,
        compute_portfolio_state,
    )
    from schemas.llm_strategist import (
        AssetState,
        IndicatorSnapshot,
        LLMInput,
        PortfolioState,
        StrategyPlan,
    )
    from trading_core.trigger_compiler import compile_plan
    from ops_api.event_store import EventStore
    from ops_api.schemas import Event

logger = logging.getLogger(__name__)

# Configuration constants
PAPER_TRADING_CONTINUE_EVERY = int(os.environ.get("PAPER_TRADING_CONTINUE_EVERY", "3600"))
PAPER_TRADING_HISTORY_LIMIT = int(os.environ.get("PAPER_TRADING_HISTORY_LIMIT", "9000"))
DEFAULT_PLAN_INTERVAL_HOURS = float(os.environ.get("PAPER_TRADING_PLAN_INTERVAL_HOURS", "4"))
PLAN_CACHE_DIR = Path(os.environ.get("PAPER_TRADING_PLAN_CACHE", ".cache/paper_trading_plans"))


class PaperTradingConfig(BaseModel):
    """Configuration for a paper trading session."""

    session_id: str
    symbols: List[str]
    initial_cash: float = 10000.0
    initial_allocations: Optional[Dict[str, float]] = None
    strategy_prompt: Optional[str] = None
    plan_interval_hours: float = DEFAULT_PLAN_INTERVAL_HOURS
    replan_on_day_boundary: bool = True
    enable_symbol_discovery: bool = False
    min_volume_24h: float = 1_000_000
    llm_model: Optional[str] = None
    exit_binding_mode: Literal["none", "category"] = "category"
    conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"


class SessionState(BaseModel):
    """Serializable session state for continue-as-new."""

    session_id: str
    symbols: List[str]
    strategy_prompt: Optional[str]
    plan_interval_hours: float
    replan_on_day_boundary: bool = True
    current_plan: Optional[Dict[str, Any]] = None
    last_plan_time: Optional[str] = None
    cycle_count: int = 0
    stopped: bool = False
    enable_symbol_discovery: bool = False
    min_volume_24h: float = 1_000_000
    plan_history: List[Dict[str, Any]] = []
    equity_history: List[Dict[str, Any]] = []
    exit_binding_mode: Literal["none", "category"] = "category"
    conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"


# ============================================================================
# Activities
# ============================================================================

@activity.defn
async def generate_strategy_plan_activity(
    symbols: List[str],
    portfolio_state: Dict[str, Any],
    strategy_prompt: Optional[str],
    market_context: Dict[str, Any],
    llm_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a strategy plan using the LLM client.

    Caches plans based on input hash to reduce LLM costs.
    """
    # Build cache key from inputs
    cache_key_data = {
        "symbols": sorted(symbols),
        "portfolio": portfolio_state,
        "prompt": strategy_prompt,
        "market": market_context,
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_data, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # Check cache
    PLAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PLAN_CACHE_DIR / f"plan_{cache_key}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            logger.info(f"Using cached strategy plan: {cache_key}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load cached plan: {e}")

    # Build LLM input
    assets = []
    for symbol in symbols:
        ctx = market_context.get(symbol, {})
        assets.append(AssetState(
            symbol=symbol,
            price=ctx.get("price", 0.0),
            trend_state=ctx.get("trend_state", "unknown"),
            vol_state=ctx.get("vol_state", "normal"),
            rsi_14=ctx.get("rsi_14"),
            sma_short=ctx.get("sma_short"),
            sma_medium=ctx.get("sma_medium"),
            sma_long=ctx.get("sma_long"),
            atr_14=ctx.get("atr_14"),
            bollinger_upper=ctx.get("bollinger_upper"),
            bollinger_lower=ctx.get("bollinger_lower"),
        ))

    portfolio = PortfolioState(
        cash=portfolio_state.get("cash", 10000.0),
        total_equity=portfolio_state.get("total_equity", 10000.0),
        positions={
            symbol: {"qty": qty, "entry_price": portfolio_state.get("entry_prices", {}).get(symbol, 0)}
            for symbol, qty in portfolio_state.get("positions", {}).items()
        },
        unrealized_pnl=portfolio_state.get("unrealized_pnl", 0.0),
        realized_pnl=portfolio_state.get("realized_pnl", 0.0),
    )

    llm_input = LLMInput(
        timestamp=datetime.now(timezone.utc).isoformat(),
        portfolio=portfolio,
        assets=assets,
    )

    # Generate plan
    llm_client = LLMClient(model=llm_model) if llm_model else LLMClient()
    plan = llm_client.generate_plan(
        llm_input,
        prompt_template=strategy_prompt,
        run_id=session_id,
        plan_id=str(uuid4()),
    )

    plan_dict = plan.model_dump()

    # Cache the plan
    try:
        cache_file.write_text(json.dumps(plan_dict, default=str))
        logger.info(f"Cached strategy plan: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache plan: {e}")

    return plan_dict


@activity.defn
def evaluate_triggers_activity(
    plan_dict: Dict[str, Any],
    market_data: Dict[str, Dict[str, Any]],
    portfolio_state: Dict[str, Any],
    exit_binding_mode: str = "category",
    conflicting_signal_policy: str = "reverse",
) -> List[Dict[str, Any]]:
    """Evaluate strategy triggers against current market data.

    Returns a list of orders to execute.
    """
    try:
        plan = StrategyPlan.model_validate(plan_dict)
    except Exception as e:
        logger.error(f"Failed to validate plan: {e}")
        return []

    # Build risk engine
    risk_engine = RiskEngine(
        plan.risk_constraints,
        {rule.symbol: rule for rule in plan.sizing_rules},
        risk_profile=RiskProfile(),
    )

    # Build trigger engine
    trigger_engine = TriggerEngine(
        plan,
        risk_engine,
        trade_risk=TradeRiskEvaluator(risk_engine),
        exit_binding_mode=exit_binding_mode if exit_binding_mode else "category",
        conflicting_signal_policy=conflicting_signal_policy if conflicting_signal_policy else "reverse",
    )

    all_orders: List[Dict[str, Any]] = []

    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)

    positions = portfolio_state.get("positions") or {}
    cash = float(portfolio_state.get("cash", 0.0))
    equity = float(
        portfolio_state.get("total_equity")
        or portfolio_state.get("total_portfolio_value")
        or cash
    )
    snapshot_ts = None
    for data in market_data.values():
        if isinstance(data, dict) and data.get("timestamp"):
            snapshot_ts = _parse_timestamp(data.get("timestamp"))
            break
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc)

    portfolio_snapshot = PortfolioState(
        timestamp=snapshot_ts,
        equity=equity,
        cash=cash,
        positions=positions,
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )
    position_meta = portfolio_state.get("position_meta") if isinstance(portfolio_state, dict) else None

    for symbol, data in market_data.items():
        if not isinstance(data, dict):
            continue
        bar_ts = _parse_timestamp(data.get("timestamp"))
        close_price = float(data.get("close", data.get("price", 0.0)) or 0.0)
        open_price = float(data.get("open", close_price))
        high_price = float(data.get("high", close_price))
        low_price = float(data.get("low", close_price))
        volume = float(data.get("volume", 0.0) or 0.0)
        timeframes = sorted({t.timeframe for t in plan.triggers if t.symbol == symbol})
        if not timeframes:
            timeframes = [str(data.get("timeframe") or "1m")]

        sma_short = data.get("sma_short") or data.get("sma_20")
        sma_medium = data.get("sma_medium") or data.get("sma_50")
        sma_long = data.get("sma_long") or data.get("sma_200")

        for timeframe in timeframes:
            bar = Bar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar_ts,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )

            indicator = IndicatorSnapshot(
                symbol=symbol,
                timeframe=timeframe,
                as_of=bar_ts,
                close=close_price,
                volume=volume,
                rsi_14=data.get("rsi_14"),
                sma_short=sma_short,
                sma_medium=sma_medium,
                sma_long=sma_long,
                atr_14=data.get("atr_14"),
                bollinger_upper=data.get("bollinger_upper"),
                bollinger_lower=data.get("bollinger_lower"),
                vwap=data.get("vwap"),
            )

            orders, _ = trigger_engine.on_bar(
                bar,
                indicator,
                portfolio_snapshot,
                asset_state=None,
                market_structure=None,
                position_meta=position_meta,
            )

            for order in orders:
                all_orders.append({
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "timeframe": order.timeframe,
                    "trigger_id": order.reason,
                    "trigger_category": order.trigger_category,
                    "intent": order.intent,
                    "reason": order.reason,
                })

    return all_orders


@activity.defn
async def discover_symbols_activity(
    exchange: str = "coinbase",
    min_volume_24h: float = 1_000_000,
    quote_currency: str = "USD",
) -> List[str]:
    """Discover available trading pairs from exchange.

    Filters by 24h volume and quote currency.
    """
    import ccxt.async_support as ccxt

    if exchange == "coinbase":
        client = ccxt.coinbaseexchange()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

    try:
        await client.load_markets()

        eligible_symbols = []
        for symbol, market in client.markets.items():
            # Filter by quote currency
            if market.get("quote") != quote_currency:
                continue

            # Filter by active status
            if not market.get("active", True):
                continue

            # Try to get 24h volume
            try:
                ticker = await client.fetch_ticker(symbol)
                volume_24h = ticker.get("quoteVolume", 0) or 0
                if volume_24h >= min_volume_24h:
                    eligible_symbols.append(symbol)
            except Exception:
                continue

        return sorted(eligible_symbols)
    finally:
        await client.close()


@activity.defn
async def fetch_current_prices_activity(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of symbols."""
    import ccxt.async_support as ccxt

    client = ccxt.coinbaseexchange()
    prices = {}

    try:
        for symbol in symbols:
            try:
                ticker = await client.fetch_ticker(symbol)
                prices[symbol] = ticker.get("last", 0) or ticker.get("close", 0) or 0
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
        return prices
    finally:
        await client.close()


@activity.defn
async def query_ledger_portfolio_activity() -> Dict[str, Any]:
    """Query the execution ledger for current portfolio status (must run as activity for Temporal client access)."""
    from agents.constants import MOCK_LEDGER_WORKFLOW_ID
    from temporalio.client import Client

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    try:
        handle = client.get_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
        return await handle.query("get_portfolio_status")
    finally:
        await client.service_client.disconnect()


@activity.defn
async def emit_paper_trading_event_activity(
    session_id: str,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """Emit an event to the event store."""
    store = EventStore()
    event = Event(
        event_id=str(uuid4()),
        ts=datetime.now(timezone.utc),
        source="paper_trading",
        type=event_type,
        payload=payload,
        run_id=session_id,
    )
    store.append(event)


# ============================================================================
# Workflow
# ============================================================================

@workflow.defn
class PaperTradingWorkflow:
    """Long-running workflow for paper trading with live market data."""

    def __init__(self) -> None:
        self.session_id: str = ""
        self.symbols: List[str] = []
        self.strategy_prompt: Optional[str] = None
        self.plan_interval_hours: float = DEFAULT_PLAN_INTERVAL_HOURS
        self.replan_on_day_boundary: bool = True
        self.current_plan: Optional[Dict[str, Any]] = None
        self.last_plan_time: Optional[datetime] = None
        self.cycle_count: int = 0
        self.stopped: bool = False
        self.enable_symbol_discovery: bool = False
        self.min_volume_24h: float = 1_000_000
        self.last_discovery_date: Optional[str] = None
        self.plan_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        self.last_equity_snapshot: Optional[datetime] = None
        self.exit_binding_mode: Literal["none", "category"] = "category"
        self.conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    @workflow.signal
    def update_symbols(self, symbols: List[str]) -> None:
        """Update the list of symbols to trade."""
        self.symbols = list(symbols)
        workflow.logger.info(f"Updated symbols: {self.symbols}")

    @workflow.signal
    def force_replan(self) -> None:
        """Force regeneration of strategy plan."""
        self.last_plan_time = None
        workflow.logger.info("Forcing strategy replan")

    @workflow.signal
    def stop_session(self) -> None:
        """Stop the paper trading session."""
        self.stopped = True
        workflow.logger.info("Stopping paper trading session")

    @workflow.signal
    def update_strategy_prompt(self, prompt: str) -> None:
        """Update the strategy prompt."""
        self.strategy_prompt = prompt
        self.last_plan_time = None  # Force replan
        workflow.logger.info("Updated strategy prompt, forcing replan")

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    @workflow.query
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "symbols": self.symbols,
            "stopped": self.stopped,
            "cycle_count": self.cycle_count,
            "last_plan_time": self.last_plan_time.isoformat() if self.last_plan_time else None,
            "has_plan": self.current_plan is not None,
            "plan_interval_hours": self.plan_interval_hours,
        }

    @workflow.query
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current strategy plan."""
        return self.current_plan

    @workflow.query
    def get_symbols(self) -> List[str]:
        """Get current symbols being traded."""
        return list(self.symbols)

    @workflow.query
    def get_plan_history(self) -> List[Dict[str, Any]]:
        """Get history of all strategy plans generated for this session."""
        return list(self.plan_history)

    @workflow.query
    def get_equity_history(self) -> List[Dict[str, Any]]:
        """Get equity curve history for this session."""
        return list(self.equity_history)

    # -------------------------------------------------------------------------
    # Main Workflow
    # -------------------------------------------------------------------------

    @workflow.run
    async def run(
        self,
        config: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the paper trading session.

        Args:
            config: Initial configuration (used on first run)
            resume_state: State from continue-as-new (used on resume)
        """
        # Initialize state
        if resume_state:
            self._restore_state(resume_state)
            workflow.logger.info(f"Resumed session {self.session_id} at cycle {self.cycle_count}")
        elif config:
            parsed_config = PaperTradingConfig.model_validate(config)
            self.session_id = parsed_config.session_id
            self.symbols = list(parsed_config.symbols)
            self.strategy_prompt = parsed_config.strategy_prompt
            self.plan_interval_hours = parsed_config.plan_interval_hours
            self.replan_on_day_boundary = parsed_config.replan_on_day_boundary
            self.enable_symbol_discovery = parsed_config.enable_symbol_discovery
            self.min_volume_24h = parsed_config.min_volume_24h
            self.exit_binding_mode = parsed_config.exit_binding_mode
            self.conflicting_signal_policy = parsed_config.conflicting_signal_policy

            # Initialize portfolio with starting allocations
            if parsed_config.initial_allocations:
                await self._initialize_portfolio(
                    parsed_config.initial_cash,
                    parsed_config.initial_allocations,
                )

            # Emit session started event
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "session_started", {
                    "symbols": self.symbols,
                    "initial_cash": parsed_config.initial_cash,
                    "initial_allocations": parsed_config.initial_allocations,
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            workflow.logger.info(f"Started paper trading session: {self.session_id}")
        else:
            raise ValueError("Either config or resume_state must be provided")

        # Main loop
        plan_interval = timedelta(hours=self.plan_interval_hours)
        evaluation_interval = timedelta(seconds=30)  # Evaluate every 30 seconds

        while not self.stopped:
            now = workflow.now()

            # Check for continue-as-new
            hist_len = workflow.info().get_current_history_length()
            if (
                hist_len >= PAPER_TRADING_HISTORY_LIMIT
                or self.cycle_count >= PAPER_TRADING_CONTINUE_EVERY
                or workflow.info().is_continue_as_new_suggested()
            ):
                workflow.logger.info(f"Triggering continue-as-new at cycle {self.cycle_count}")
                await workflow.continue_as_new(
                    args=[None, self._snapshot_state()]
                )

            # Daily symbol discovery
            if self.enable_symbol_discovery:
                today = now.strftime("%Y-%m-%d")
                if self.last_discovery_date != today:
                    await self._discover_new_symbols()
                    self.last_discovery_date = today

            # Generate/update strategy plan
            if self.replan_on_day_boundary and self.last_plan_time is not None:
                if self.last_plan_time.date() != now.date():
                    self.last_plan_time = None
            if self.last_plan_time is None or (now - self.last_plan_time) >= plan_interval:
                await self._generate_plan()

            # Evaluate triggers and execute orders
            if self.current_plan:
                await self._evaluate_and_execute()

            # Track equity periodically (every 5 minutes)
            equity_interval = timedelta(minutes=5)
            if self.last_equity_snapshot is None or (now - self.last_equity_snapshot) >= equity_interval:
                await self._record_equity_snapshot()

            self.cycle_count += 1

            # Sleep until next evaluation
            await workflow.sleep(evaluation_interval)

        # Session stopped
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "session_stopped", {
                "cycle_count": self.cycle_count,
            }],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        return {
            "session_id": self.session_id,
            "status": "stopped",
            "cycle_count": self.cycle_count,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _initialize_portfolio(
        self,
        initial_cash: float,
        initial_allocations: Dict[str, float],
    ) -> None:
        """Initialize the execution ledger with starting portfolio."""
        from agents.constants import MOCK_LEDGER_WORKFLOW_ID

        # Fetch current prices for position initialization
        prices = await workflow.execute_activity(
            fetch_current_prices_activity,
            args=[list(initial_allocations.keys())],
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Signal the execution ledger to initialize
        ledger_handle = workflow.get_external_workflow_handle(MOCK_LEDGER_WORKFLOW_ID)
        await ledger_handle.signal("initialize_portfolio", {
            "cash": initial_allocations.get("cash", initial_cash),
            "positions": {
                symbol: alloc / prices.get(symbol, 1)
                for symbol, alloc in initial_allocations.items()
                if symbol != "cash" and prices.get(symbol, 0) > 0
            },
            "prices": prices,
        })

        workflow.logger.info(f"Initialized portfolio: cash={initial_cash}, allocations={initial_allocations}")

    async def _generate_plan(self) -> None:
        """Generate a new strategy plan."""
        # Get portfolio state from ledger (via activity — external handles can't query)
        portfolio_state = await workflow.execute_activity(
            query_ledger_portfolio_activity,
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Build market context (simplified for now)
        market_context = {}
        for symbol in self.symbols:
            # In full implementation, would query ComputeFeatureVector workflows
            market_context[symbol] = {
                "price": portfolio_state.get("last_prices", {}).get(symbol, 0),
                "trend_state": "unknown",
                "vol_state": "normal",
            }

        # Generate plan
        plan_dict = await workflow.execute_activity(
            generate_strategy_plan_activity,
            args=[
                self.symbols,
                portfolio_state,
                self.strategy_prompt,
                market_context,
                None,  # llm_model
                self.session_id,
            ],
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        self.current_plan = plan_dict
        self.last_plan_time = workflow.now()

        # Store plan in history
        plan_record = {
            "plan_index": len(self.plan_history),
            "generated_at": self.last_plan_time.isoformat(),
            "trigger_count": len(plan_dict.get("triggers", [])),
            "max_trades_per_day": plan_dict.get("max_trades_per_day"),
            "market_regime": plan_dict.get("market_regime"),
            "symbols": plan_dict.get("allowed_symbols", self.symbols),
            "valid_until": plan_dict.get("valid_until"),
            "triggers": plan_dict.get("triggers", []),
        }
        self.plan_history.append(plan_record)

        # Emit event
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "plan_generated", {
                "trigger_count": len(plan_dict.get("triggers", [])),
                "plan_index": len(self.plan_history) - 1,
            }],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        workflow.logger.info(f"Generated strategy plan with {len(plan_dict.get('triggers', []))} triggers")

    async def _evaluate_and_execute(self) -> None:
        """Evaluate triggers and execute orders."""
        # Get current portfolio state (via activity — external handles can't query)
        portfolio_state = await workflow.execute_activity(
            query_ledger_portfolio_activity,
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Build market data from latest prices
        market_data = {}
        for symbol in self.symbols:
            price = portfolio_state.get("last_prices", {}).get(symbol, 0)
            if price > 0:
                market_data[symbol] = {
                    "timestamp": workflow.now().isoformat(),
                    "price": price,
                    "close": price,
                    "open": price,
                    "high": price,
                    "low": price,
                    "volume": 0,
                }

        if not market_data:
            return

        # Evaluate triggers
        orders = await workflow.execute_activity(
            evaluate_triggers_activity,
            args=[
                self.current_plan,
                market_data,
                portfolio_state,
                self.exit_binding_mode,
                self.conflicting_signal_policy,
            ],
            schedule_to_close_timeout=timedelta(seconds=30),
        )

        # Execute orders
        for order in orders:
            await self._execute_order(order, ledger_handle)

    async def _execute_order(self, order: Dict[str, Any], ledger_handle) -> None:
        """Execute a single order."""
        # Record fill in ledger
        fill_price = float(order.get("price", 0.0) or 0.0)
        quantity = float(order.get("quantity", 0.0) or 0.0)
        cost = fill_price * quantity
        fee = cost * 0.001
        await ledger_handle.signal("record_fill", {
            "symbol": order["symbol"],
            "side": order["side"].upper(),
            "qty": quantity,
            "fill_price": fill_price,
            "cost": cost,
            "fee": fee,  # 0.1% fee for reporting
            "timestamp": int(workflow.now().timestamp() * 1000),
            "trigger_id": order.get("trigger_id") or order.get("reason"),
            "trigger_category": order.get("trigger_category"),
            "intent": order.get("intent"),
        })

        # Emit event
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "order_executed", order],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        workflow.logger.info(f"Executed order: {order['side']} {order['quantity']} {order['symbol']} @ {order['price']}")

    async def _discover_new_symbols(self) -> None:
        """Discover and add new trading symbols."""
        discovered = await workflow.execute_activity(
            discover_symbols_activity,
            args=["coinbase", self.min_volume_24h, "USD"],
            schedule_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        new_symbols = [s for s in discovered if s not in self.symbols]
        if new_symbols:
            self.symbols.extend(new_symbols)
            workflow.logger.info(f"Discovered {len(new_symbols)} new symbols: {new_symbols}")

            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "symbols_discovered", {
                    "new_symbols": new_symbols,
                    "total_symbols": len(self.symbols),
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

    async def _record_equity_snapshot(self) -> None:
        """Record a periodic equity snapshot for charting."""
        try:
            portfolio_state = await workflow.execute_activity(
                query_ledger_portfolio_activity,
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            snapshot = {
                "timestamp": workflow.now().isoformat(),
                "cash": portfolio_state.get("cash", 0),
                "total_equity": portfolio_state.get("total_equity", 0),
                "positions": portfolio_state.get("positions", {}),
                "unrealized_pnl": portfolio_state.get("unrealized_pnl", 0),
                "realized_pnl": portfolio_state.get("realized_pnl", 0),
            }

            # Keep only last 2000 snapshots to prevent unbounded growth
            if len(self.equity_history) >= 2000:
                self.equity_history = self.equity_history[-1500:]

            self.equity_history.append(snapshot)
            self.last_equity_snapshot = workflow.now()
        except Exception as e:
            workflow.logger.warning(f"Failed to record equity snapshot: {e}")

    def _snapshot_state(self) -> Dict[str, Any]:
        """Create state snapshot for continue-as-new."""
        return SessionState(
            session_id=self.session_id,
            symbols=self.symbols,
            strategy_prompt=self.strategy_prompt,
            plan_interval_hours=self.plan_interval_hours,
            replan_on_day_boundary=self.replan_on_day_boundary,
            current_plan=self.current_plan,
            last_plan_time=self.last_plan_time.isoformat() if self.last_plan_time else None,
            cycle_count=self.cycle_count,
            stopped=self.stopped,
            enable_symbol_discovery=self.enable_symbol_discovery,
            min_volume_24h=self.min_volume_24h,
            plan_history=self.plan_history,
            equity_history=self.equity_history[-500:],  # Keep last 500 snapshots across continue-as-new
            exit_binding_mode=self.exit_binding_mode,
            conflicting_signal_policy=self.conflicting_signal_policy,
        ).model_dump()

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from continue-as-new snapshot."""
        parsed = SessionState.model_validate(state)
        self.session_id = parsed.session_id
        self.symbols = parsed.symbols
        self.strategy_prompt = parsed.strategy_prompt
        self.plan_interval_hours = parsed.plan_interval_hours
        self.replan_on_day_boundary = parsed.replan_on_day_boundary
        self.current_plan = parsed.current_plan
        self.last_plan_time = datetime.fromisoformat(parsed.last_plan_time) if parsed.last_plan_time else None
        self.cycle_count = parsed.cycle_count
        self.stopped = parsed.stopped
        self.enable_symbol_discovery = parsed.enable_symbol_discovery
        self.min_volume_24h = parsed.min_volume_24h
        self.plan_history = parsed.plan_history
        self.equity_history = parsed.equity_history
        self.exit_binding_mode = parsed.exit_binding_mode
        self.conflicting_signal_policy = parsed.conflicting_signal_policy
