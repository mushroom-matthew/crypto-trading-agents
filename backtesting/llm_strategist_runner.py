"""Backtesting harness that wires the LLM strategist into the simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Literal

import pandas as pd

from agents.analytics import (
    IndicatorWindowConfig,
    PortfolioHistory,
    build_asset_state,
    build_market_structure_snapshot,
    compute_factor_loadings,
    compute_indicator_snapshot,
    precompute_indicator_frame,
    snapshot_from_frame,
    compute_portfolio_state,
)
from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from agents.strategies.risk_engine import RiskEngine, RiskProfile
from agents.strategies.strategy_memory import build_strategy_memory
from agents.strategies.trade_risk import TradeRiskEvaluator
from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
from backtesting.dataset import load_ohlcv
from backtesting.llm_shim import build_judge_shim_feedback, make_judge_shim_transport
from services.judge_feedback_service import JudgeFeedbackService
from backtesting.reports import write_run_summary
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from schemas.llm_strategist import AssetState, IndicatorSnapshot, LLMInput, PortfolioState, StrategyPlan, TriggerSummary
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings
from services.risk_adjustment_service import apply_judge_risk_feedback, effective_risk_limits, snapshot_adjustments, build_risk_profile
from services.strategy_run_registry import StrategyRunConfig, StrategyRunRegistry, strategy_run_registry
from services.strategist_plan_service import StrategistPlanService
from ops_api.event_store import EventStore
from ops_api.schemas import Event
from tools import execution_tools
from trading_core.trigger_compiler import compile_plan, validate_plan_timeframes
from trading_core.trigger_budget import enforce_trigger_budget
from trading_core.execution_engine import BlockReason
from trading_core.trade_quality import (
    TradeMetrics,
    compute_trade_metrics,
    format_metrics_for_judge,
    assess_position_quality,
    format_position_quality_for_judge,
)
from data_loader.utils import timeframe_to_seconds

logger = logging.getLogger(__name__)


def _new_limit_entry() -> Dict[str, Any]:
    return {
        "trades_attempted": 0,
        "trades_executed": 0,
        "priority_skips": 0,
        "skipped": defaultdict(int),
        "risk_block_breakdown": defaultdict(int),
        "risk_limit_hints": defaultdict(int),
        "blocked_details": [],
        "executed_details": [],
    }


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        # Ensure index is UTC for consistent comparison with start/end
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        return df
    if "time" in df.columns:
        df = df.set_index("time")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    else:
        raise ValueError("Dataframe must include time column")
    # Ensure index is UTC
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


@dataclass
class PortfolioTracker:
    initial_cash: float
    fee_rate: float
    initial_allocations: Dict[str, float] | None = None
    initial_prices: Dict[str, float] | None = None
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    avg_entry_price: Dict[str, float] = field(default_factory=dict)
    position_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    equity_records: List[Dict[str, Any]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

        # Apply initial allocations if provided
        if self.initial_allocations:
            # Override cash if specified in allocations
            self.cash = float(self.initial_allocations.get("cash", self.initial_cash))

            # Create initial positions from notional allocations
            prices = self.initial_prices or {}
            for symbol, notional in self.initial_allocations.items():
                if symbol == "cash":
                    continue
                notional = float(notional)
                if notional <= 0:
                    continue
                price = prices.get(symbol)
                if not price or price <= 0:
                    continue
                qty = notional / price
                self.positions[symbol] = qty
                self.avg_entry_price[symbol] = price
                self.position_meta[symbol] = {
                    "reason": "initial_allocation",
                    "timeframe": None,
                    "opened_at": None,
                    "entry_price": price,
                    "entry_side": "long",
                }

    def _record_pnl(
        self,
        symbol: str,
        pnl: float,
        timestamp: datetime,
        reason: str | None = None,
        entry_reason: str | None = None,
        entry_timeframe: str | None = None,
        entry_timestamp: datetime | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        entry_side: str | None = None,
        market_structure_entry: Dict[str, Any] | None = None,
    ) -> None:
        self.trade_log.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "pnl": pnl,
                "reason": reason,
                "entry_reason": entry_reason,
                "entry_timeframe": entry_timeframe,
                "entry_timestamp": entry_timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_side": entry_side,
                "market_structure_entry": market_structure_entry,
            }
        )

    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _update_position(
        self,
        symbol: str,
        delta_qty: float,
        price: float,
        timestamp: datetime,
        reason: str | None = None,
        timeframe: str | None = None,
        market_structure_entry: Dict[str, Any] | None = None,
    ) -> None:
        position = self.positions.get(symbol, 0.0)
        avg_price = self.avg_entry_price.get(symbol, price)
        meta = self.position_meta.get(symbol, {})
        new_position = position + delta_qty
        if position == 0 or (position > 0 and new_position > 0) or (position < 0 and new_position < 0):
            if new_position == 0:
                self.positions.pop(symbol, None)
                self.avg_entry_price.pop(symbol, None)
                self.position_meta.pop(symbol, None)
            else:
                total_notional = abs(position) * avg_price + abs(delta_qty) * price
                self.positions[symbol] = new_position
                self.avg_entry_price[symbol] = total_notional / abs(new_position)
                self.position_meta[symbol] = {
                    "reason": reason,
                    "timeframe": timeframe,
                    "opened_at": timestamp,
                    "entry_price": price,
                    "entry_side": "long" if new_position > 0 else "short",
                    "market_structure_entry": market_structure_entry or meta.get("market_structure_entry"),
                }
            return
        closing_qty = min(abs(position), abs(delta_qty))
        if closing_qty > 0:
            pnl = closing_qty * ((price - avg_price) if position > 0 else (avg_price - price))
            self._record_pnl(
                symbol,
                pnl,
                timestamp,
                reason,
                entry_reason=meta.get("reason"),
                entry_timeframe=meta.get("timeframe"),
                entry_timestamp=meta.get("opened_at"),
                entry_price=meta.get("entry_price"),
                exit_price=price,
                entry_side=meta.get("entry_side"),
                market_structure_entry=meta.get("market_structure_entry"),
            )
        remaining = position + delta_qty
        if abs(remaining) <= 1e-9:
            self.positions.pop(symbol, None)
            self.avg_entry_price.pop(symbol, None)
            self.position_meta.pop(symbol, None)
        else:
            self.positions[symbol] = remaining
            # direction flipped; reset basis to execution price
            self.avg_entry_price[symbol] = price
            self.position_meta[symbol] = {
                "reason": reason,
                "timeframe": timeframe,
                "opened_at": timestamp,
                "entry_price": price,
                "entry_side": "long" if remaining > 0 else "short",
            }

    def execute(self, order: Order, market_structure_entry: Dict[str, Any] | None = None) -> None:
        qty = max(order.quantity, 0.0)
        if qty <= 0:
            return
        timestamp = self._normalize_timestamp(order.timestamp)
        notional = qty * order.price
        fee = notional * self.fee_rate
        if order.side == "buy":
            total_cost = notional + fee
            if total_cost > self.cash and self.positions.get(order.symbol, 0.0) >= 0:
                return
            self.cash -= total_cost
            delta = qty
        else:
            proceeds = notional - fee
            self.cash += proceeds
            delta = -qty

        # Calculate realized PnL BEFORE position update (need pre-trade position)
        realized_pnl = 0.0
        pre_position = self.positions.get(order.symbol, 0.0)
        avg_price = self.avg_entry_price.get(order.symbol, order.price)
        # Check if this trade closes/reduces position (opposite direction)
        if (pre_position > 0 and order.side == "sell") or (pre_position < 0 and order.side == "buy"):
            closing_qty = min(abs(pre_position), qty)
            if pre_position > 0:  # Closing long
                realized_pnl = closing_qty * (order.price - avg_price)
            else:  # Closing short
                realized_pnl = closing_qty * (avg_price - order.price)

        self._update_position(
            order.symbol,
            delta,
            order.price,
            timestamp,
            order.reason,
            order.timeframe,
            market_structure_entry=market_structure_entry,
        )

        self.fills.append(
            {
                "timestamp": timestamp,
                "symbol": order.symbol,
                "side": order.side,
                "qty": qty,
                "price": order.price,
                "fee": fee,
                "reason": order.reason,
                "pnl": realized_pnl,
                "realized_pnl": realized_pnl,
                "timeframe": order.timeframe,
                "market_structure_entry": market_structure_entry,
            }
        )

    def mark_to_market(self, timestamp: datetime, price_map: Mapping[str, float]) -> None:
        equity = self.cash
        for symbol, qty in self.positions.items():
            price = price_map.get(symbol)
            if price is not None:
                equity += qty * price
        self.equity_records.append({"timestamp": timestamp, "equity": equity, "cash": self.cash})

    def portfolio_state(self, timestamp: datetime) -> PortfolioState:
        if not self.equity_records:
            self.mark_to_market(timestamp, {})
        equity_df = pd.DataFrame(self.equity_records)
        trade_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame(columns=["timestamp", "symbol", "pnl"])
        history = PortfolioHistory(equity_curve=equity_df, trade_log=trade_df)
        return compute_portfolio_state(history, timestamp, self.positions)


@dataclass
class StrategistBacktestResult:
    equity_curve: pd.Series
    fills: pd.DataFrame
    plan_log: List[Dict[str, Any]]
    summary: Dict[str, Any]
    llm_costs: Dict[str, float]
    final_cash: float
    final_positions: Dict[str, float]
    daily_reports: List[Dict[str, Any]]
    bar_decisions: Dict[str, List[Dict[str, Any]]]
    intraday_judge_history: List[Dict[str, Any]] = field(default_factory=list)
    judge_triggered_replans: List[Dict[str, Any]] = field(default_factory=list)


class LLMStrategistBacktester:
    def __init__(
        self,
        pairs: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        initial_cash: float,
        fee_rate: float,
        llm_client: LLMClient,
        cache_dir: Path,
        llm_calls_per_day: int,
        risk_params: Dict[str, Any] | RiskLimitSettings | None = None,
        timeframes: Sequence[str] = ("1h", "4h", "1d"),
        prompt_template_path: Path | None = None,
        strategy_prompt: str | None = None,
        plan_provider: StrategyPlanProvider | None = None,
        market_data: Dict[str, Dict[str, pd.DataFrame]] | None = None,
        run_registry: StrategyRunRegistry | None = None,
        flatten_positions_daily: bool = False,
        flatten_notional_threshold: float = 0.0,
        flatten_session_boundary_hour: int | None = None,
        session_trade_multipliers: Sequence[Mapping[str, float | int]] | None = None,
        timeframe_trigger_caps: Mapping[str, int] | None = None,
        flatten_policy: str | None = None,
        factor_data: pd.DataFrame | None = None,
        auto_hedge_market: bool = False,
        factor_data_path: Path | None = None,
        min_hold_hours: float = 2.0,
        min_flat_hours: float = 2.0,
        confidence_override_threshold: Literal["A", "B", "C"] | None = "A",
        initial_allocations: Dict[str, float] | None = None,
        walk_away_enabled: bool = False,
        walk_away_profit_target_pct: float = 25.0,
        max_trades_per_day: int | None = None,
        max_triggers_per_symbol_per_day: int | None = None,
        debug_trigger_sample_rate: float = 0.0,
        debug_trigger_max_samples: int = 100,
        use_trigger_vector_store: bool = False,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
        indicator_debug_mode: str | None = None,
        indicator_debug_keys: Sequence[str] | None = None,
        # Adaptive judge workflow parameters
        judge_cadence_hours: float = 4.0,  # Minimum hours between judge evaluations
        judge_replan_threshold: float = 40.0,  # Score below which to trigger immediate replan
        adaptive_replanning: bool = True,  # Enable judge-driven replanning
        judge_check_after_trades: int = 3,  # Check judge after this many trades
        use_judge_shim: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must be provided")
        self.pairs = list(pairs)
        self.start = self._normalize_datetime(start)
        self.end = self._normalize_datetime(end)
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate

        # Expand timeframes to include derived higher timeframes for multi-timeframe analysis
        # This allows the LLM to reference tf_1h_*, tf_4h_* even when only 5m is configured
        base_timeframes = list(timeframes)
        base_timeframe = min(base_timeframes, key=self._timeframe_seconds)
        derived_timeframes = self._derive_higher_timeframes(base_timeframe)
        # Merge: keep user-specified + add derived ones
        all_timeframes = list(base_timeframes)
        for tf in derived_timeframes:
            if tf not in all_timeframes:
                all_timeframes.append(tf)
        self.timeframes = all_timeframes
        logger.info(
            "Timeframes expanded: configured=%s, derived=%s, final=%s",
            base_timeframes, derived_timeframes, self.timeframes
        )

        self.market_data = self._normalize_market_data(market_data) if market_data is not None else self._load_data()
        if (self.start is None or self.end is None) and self.market_data:
            timestamps = sorted({ts for tf_map in self.market_data.values() for df in tf_map.values() for ts in df.index})
            if timestamps:
                first = timestamps[0]
                last = timestamps[-1]
                if isinstance(first, pd.Timestamp):
                    first = first.to_pydatetime()
                if isinstance(last, pd.Timestamp):
                    last = last.to_pydatetime()
                if self.start is None:
                    self.start = self._normalize_datetime(first)
                if self.end is None:
                    self.end = self._normalize_datetime(last)
        if self.start is None or self.end is None:
            raise ValueError("start and end must be provided when market data is not supplied")

        # Get initial prices for allocation handling
        self.initial_allocations = initial_allocations
        initial_prices: Dict[str, float] = {}
        if initial_allocations:
            base_tf = self.timeframes[0]
            for symbol in self.pairs:
                df = self.market_data.get(symbol, {}).get(base_tf)
                if df is not None and not df.empty:
                    # Get the first available price
                    initial_prices[symbol] = float(df.iloc[0]["close"])

        self.portfolio = PortfolioTracker(
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            initial_allocations=initial_allocations,
            initial_prices=initial_prices,
        )
        self.calls_per_day = max(1, llm_calls_per_day)
        self.plan_interval = timedelta(hours=24 / self.calls_per_day)
        self.plan_provider = plan_provider or StrategyPlanProvider(llm_client, cache_dir=cache_dir, llm_calls_per_day=self.calls_per_day)
        self.run_registry = run_registry or strategy_run_registry
        self.plan_service = StrategistPlanService(plan_provider=self.plan_provider, registry=self.run_registry)
        self._event_store = EventStore()
        self.window_configs = {tf: IndicatorWindowConfig(timeframe=tf) for tf in self.timeframes}
        self.indicator_frames = self._precompute_indicator_frames()
        self.indicator_debug_mode = (indicator_debug_mode or "off").strip().lower()
        self.indicator_debug_keys = [key for key in (indicator_debug_keys or []) if key]
        if self.indicator_debug_mode in {"none", "off"}:
            self.indicator_debug_mode = ""
        if isinstance(risk_params, RiskLimitSettings):
            self.base_risk_limits = risk_params
        else:
            payload = risk_params or {}
            self.base_risk_limits = RiskLimitSettings.model_validate(payload)
        self.active_risk_limits = self.base_risk_limits
        self.active_risk_adjustments: Dict[str, RiskAdjustmentState] = {}
        self.risk_params = self.active_risk_limits.to_risk_params()
        self.risk_profile = RiskProfile()
        arch_mults, arch_hour_mults, multiplier_meta = self._load_archetype_multipliers()
        self._loaded_archetype_multipliers = dict(arch_mults)
        self._loaded_archetype_hour_multipliers = dict(arch_hour_mults)
        self.risk_profile.archetype_multipliers = arch_mults
        self.risk_profile.archetype_hour_multipliers = arch_hour_mults
        self.multiplier_meta = multiplier_meta

        self.rpr_comparison_snapshot: Dict[str, Any] = self._load_rpr_comparison_snapshot()
        self.daily_risk_budget_state: Dict[str, Dict[str, float]] = {}
        self.daily_risk_budget_pct = self.active_risk_limits.max_daily_risk_budget_pct
        self.strict_fixed_caps = os.environ.get("STRATEGIST_STRICT_FIXED_CAPS", "false").lower() == "true"
        # Use strategy_prompt directly if provided, otherwise load from file path
        if strategy_prompt:
            self.prompt_template = strategy_prompt
        elif prompt_template_path and prompt_template_path.exists():
            self.prompt_template = prompt_template_path.read_text()
        else:
            self.prompt_template = None
        self.slot_reports_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.bar_decisions_by_day: Dict[str, List[Dict[str, Any]]] = {}
        self.trigger_activity_by_day: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.skipped_activity_by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.replans_by_day: Dict[str, int] = defaultdict(int)
        self.no_change_replan_suppressed_by_day: Dict[str, int] = defaultdict(int)
        self.current_day_key: str | None = None
        self.latest_daily_summary: Dict[str, Any] | None = None
        self.last_slot_report: Dict[str, Any] | None = None
        self.memory_history: List[Dict[str, Any]] = []
        self.judge_constraints: Dict[str, Any] = {}
        self.limit_enforcement_by_day: Dict[str, Dict[str, Any]] = defaultdict(_new_limit_entry)
        self.plan_limits_by_day: Dict[str, Dict[str, Any]] = {}
        self.current_run_id: str | None = None
        self.default_symbol_trigger_cap = 6
        self.latest_trigger_trim: Dict[str, int] = {}
        self.flatten_positions_daily = flatten_positions_daily
        self.flattened_days: set[str] = set()
        self.flatten_notional_threshold = max(0.0, flatten_notional_threshold)
        self.flatten_session_boundary_hour = flatten_session_boundary_hour
        self.min_hold_hours = float(min_hold_hours)
        hold_override = os.environ.get("LLM_MIN_HOLD_HOURS")
        if hold_override:
            try:
                self.min_hold_hours = float(hold_override)
            except ValueError:
                logger.warning("Invalid LLM_MIN_HOLD_HOURS=%s; using %.2f", hold_override, self.min_hold_hours)
        self.min_hold_hours = max(0.0, self.min_hold_hours)
        self.min_hold_window = timedelta(hours=self.min_hold_hours)
        self.min_hold_bars = 0
        if self.timeframes:
            try:
                base_seconds = timeframe_to_seconds(str(self.timeframes[0]))
                if base_seconds > 0:
                    self.min_hold_bars = int(math.ceil(self.min_hold_hours * 3600 / base_seconds))
            except ValueError:
                logger.warning("Invalid timeframe for min_hold_bars: %s", self.timeframes[0])
        self.min_flat_hours = float(min_flat_hours)
        flat_override = os.environ.get("LLM_MIN_FLAT_HOURS")
        if flat_override:
            try:
                self.min_flat_hours = float(flat_override)
            except ValueError:
                logger.warning("Invalid LLM_MIN_FLAT_HOURS=%s; using %.2f", flat_override, self.min_flat_hours)
        self.min_flat_hours = max(0.0, self.min_flat_hours)
        self.min_flat_window = timedelta(hours=self.min_flat_hours)
        self.session_trade_multipliers = list(session_trade_multipliers) if session_trade_multipliers else None
        self.timeframe_trigger_caps = {str(tf): int(cap) for tf, cap in (timeframe_trigger_caps or {}).items() if cap is not None}
        self.flatten_policy = flatten_policy or (
            "session_close_utc"
            if flatten_session_boundary_hour is not None
            else ("daily_close" if flatten_positions_daily else "none")
        )
        self.factor_data = self._load_factor_data(factor_data_path) if factor_data is None else factor_data
        if self.factor_data is None or (hasattr(self.factor_data, "empty") and self.factor_data.empty):
            self.factor_data = self._default_factor_data()
        self.latest_factor_exposures: Dict[str, Dict[str, Any]] = {}
        self.auto_hedge_market = auto_hedge_market
        self.confidence_override_threshold = confidence_override_threshold

        # Walk-away threshold: stop trading after hitting daily profit target
        self.walk_away_enabled = walk_away_enabled
        self.walk_away_profit_target_pct = max(1.0, walk_away_profit_target_pct)
        self.walk_away_triggered_days: set[str] = set()
        self.walk_away_events: List[Dict[str, Any]] = []
        self.plan_max_trades_per_day = int(max_trades_per_day) if max_trades_per_day is not None else None
        self.plan_max_triggers_per_symbol_per_day = (
            int(max_triggers_per_symbol_per_day) if max_triggers_per_symbol_per_day is not None else None
        )
        # Debug sampling for trigger evaluation
        self.debug_trigger_sample_rate = max(0.0, min(1.0, debug_trigger_sample_rate))
        self.debug_trigger_max_samples = debug_trigger_max_samples
        self._all_trigger_evaluation_samples: List[Dict[str, Any]] = []
        # Vector store for trigger examples
        self.use_trigger_vector_store = use_trigger_vector_store

        # Progress tracking callback for real-time updates
        self.progress_callback = progress_callback
        self.candles_processed = 0
        self.candles_total = 0
        self.current_timestamp: datetime | None = None
        self.recent_events: List[Dict[str, Any]] = []

        # Adaptive judge workflow state
        self.judge_cadence_hours = max(0.5, judge_cadence_hours)  # Min 30 minutes
        self.judge_cadence = timedelta(hours=self.judge_cadence_hours)
        self.judge_replan_threshold = judge_replan_threshold
        self.adaptive_replanning = adaptive_replanning
        self.judge_check_after_trades = max(1, judge_check_after_trades)
        self.use_judge_shim = use_judge_shim
        # Initialize judge feedback service with transport if shimming
        if use_judge_shim:
            judge_transport = make_judge_shim_transport()
        else:
            judge_transport = None
        self.judge_service = JudgeFeedbackService(transport=judge_transport, model=llm_client.model if hasattr(llm_client, 'model') else None)
        self.last_judge_time: datetime | None = None
        self.next_judge_time: datetime | None = None
        self.trades_since_last_judge = 0
        self.intraday_judge_history: List[Dict[str, Any]] = []
        self.judge_triggered_replans: List[Dict[str, Any]] = []

        if self.flatten_policy == "daily_close":
            self.flatten_positions_daily = True
            self.flatten_session_boundary_hour = None
        elif self.flatten_policy == "session_close_utc":
            self.flatten_positions_daily = False
            if self.flatten_session_boundary_hour is None:
                self.flatten_session_boundary_hour = flatten_session_boundary_hour if flatten_session_boundary_hour is not None else 0
        else:
            self.flatten_positions_daily = False
            self.flatten_session_boundary_hour = None
        self.risk_usage_by_day: Dict[str, Dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
        self.risk_usage_events_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.latency_by_day: Dict[str, Dict[tuple[str, str], List[float]]] = defaultdict(lambda: defaultdict(list))
        self.timeframe_exec_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.session_trade_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.trigger_load_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trigger_load_threshold = int(os.environ.get("TRIGGER_LOAD_THRESHOLD", "12"))
        self.archetype_load_by_day: Dict[str, Dict[tuple[str, str, int], int]] = defaultdict(lambda: defaultdict(int))
        self.archetype_load_threshold = int(os.environ.get("ARCHETYPE_LOAD_THRESHOLD", "8"))
        self.archetype_load_scale_start = float(os.environ.get("ARCHETYPE_LOAD_SCALE_START", "0.6"))
        self.last_flat_time_by_symbol: Dict[str, datetime] = {}
        self.last_flat_trigger_by_symbol: Dict[str, str] = {}
        self.current_trigger_engine: TriggerEngine | None = None
        # Daily loss anchoring: set once per day at day boundary, never refreshed intraday.
        self.daily_loss_anchor_by_day: Dict[str, float] = {}
        self.llm_generation_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        logger.debug(
            "Initialized backtester pairs=%s timeframes=%s start=%s end=%s plan_interval_hours=%.2f",
            self.pairs,
            self.timeframes,
            self.start,
            self.end,
            self.plan_interval.total_seconds() / 3600,
        )

    def _emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        run_id: str,
        correlation_id: str | None = None,
        event_ts: datetime | None = None,
    ) -> None:
        try:
            ts = event_ts or self.current_timestamp or datetime.now(timezone.utc)
            event = Event(
                event_id=str(uuid4()),
                ts=ts,
                source="llm_strategist",
                type=event_type,  # type: ignore[arg-type]
                payload=payload,
                dedupe_key=None,
                run_id=run_id,
                correlation_id=correlation_id,
            )
            self._event_store.append(event)
        except Exception:
            logger.debug("Failed to emit %s event", event_type, exc_info=True)

    def _load_archetype_multipliers(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """Load archetype and archetype|hour multipliers from a JSON blob or file path (optional)."""

        def _normalize_map(raw: Any) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
            archetypes: Dict[str, float] = {}
            archetype_hours: Dict[str, float] = {}
            meta: Dict[str, Any] = {}
            if not isinstance(raw, dict):
                return archetypes, archetype_hours, meta
            if "meta" in raw and isinstance(raw.get("meta"), dict):
                meta = dict(raw.get("meta") or {})

            def _coerce_multiplier(val: Any) -> float | None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            def _coerce_label(payload: Mapping[str, Any]) -> float | None:
                label = str(payload.get("label")).lower()
                if label == "good":
                    return 1.2
                if label == "bad":
                    return 0.5
                return 1.0

            if "archetypes" in raw and isinstance(raw.get("archetypes"), dict):
                for key, payload in raw["archetypes"].items():
                    if isinstance(payload, dict):
                        multiplier = payload.get("multiplier")
                        if multiplier is None and "label" in payload:
                            multiplier = _coerce_label(payload)
                    else:
                        multiplier = _coerce_multiplier(payload)
                    if multiplier is not None:
                        archetypes[str(key)] = float(multiplier)
            if "archetype_hours" in raw and isinstance(raw.get("archetype_hours"), dict):
                for key, payload in raw["archetype_hours"].items():
                    if isinstance(payload, dict):
                        multiplier = payload.get("multiplier")
                        if multiplier is None and "label" in payload:
                            multiplier = _coerce_label(payload)
                    else:
                        multiplier = _coerce_multiplier(payload)
                    if multiplier is not None:
                        archetype_hours[str(key)] = float(multiplier)
            # Backwards compatibility: allow a flat map with no namespaces.
            if not archetypes and not archetype_hours:
                for key, val in raw.items():
                    coerced = _coerce_multiplier(val)
                    if coerced is not None:
                        archetypes[str(key)] = coerced
            return archetypes, archetype_hours, meta

        raw_inline = os.environ.get("ARCHETYPE_MULTIPLIERS")
        source = None
        if raw_inline:
            try:
                parsed = json.loads(raw_inline)
                archetypes, archetype_hours, meta = _normalize_map(parsed)
                if archetypes or archetype_hours:
                    source = "env"
                    logger.info(
                        "Loaded archetype multipliers from env: %s archetypes, %s hours",
                        len(archetypes),
                        len(archetype_hours),
                    )
                    return archetypes, archetype_hours, meta
            except Exception as exc:  # pragma: no cover - defensive parse guard
                logger.warning("Failed to parse ARCHETYPE_MULTIPLIERS env: %s", exc)
        path_val = os.environ.get("ARCHETYPE_MULTIPLIERS_PATH")
        if path_val:
            candidate = Path(path_val)
            if candidate.exists():
                try:
                    parsed = json.loads(candidate.read_text())
                    archetypes, archetype_hours, meta = _normalize_map(parsed)
                    if archetypes or archetype_hours:
                        source = str(candidate)
                        logger.info(
                            "Loaded archetype multipliers from %s: %s archetypes, %s hours",
                            source,
                            len(archetypes),
                            len(archetype_hours),
                        )
                        return archetypes, archetype_hours, meta
                except Exception as exc:  # pragma: no cover - defensive parse guard
                    logger.warning("Failed to load archetype multipliers from %s: %s", candidate, exc)
            else:
                logger.info("ARCHETYPE_MULTIPLIERS_PATH set but file missing: %s", candidate)
        if source is None:
            logger.info("No archetype multipliers loaded (env/ARCHETYPE_MULTIPLIERS_PATH empty); using defaults of 1.0")
        return {}, {}, {}

    def _load_rpr_comparison_snapshot(self) -> Dict[str, Any]:
        """Optionally load a precomputed RPR comparison snapshot for prompt context."""

        path_val = os.environ.get("RPR_COMPARISON_PATH")
        if not path_val:
            return {}
        candidate = Path(path_val)
        if not candidate.exists():
            return {}
        try:
            parsed = json.loads(candidate.read_text())
        except Exception as exc:  # pragma: no cover - defensive parse guard
            logger.warning("Failed to load RPR comparison snapshot from %s: %s", candidate, exc)
            return {}
        if isinstance(parsed, dict) and "rpr_comparison" in parsed:
            parsed = parsed.get("rpr_comparison") or {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_market_data(self, market_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        normalized: Dict[str, Dict[str, pd.DataFrame]] = {}
        for pair, tf_map in market_data.items():
            normalized_pairs: Dict[str, pd.DataFrame] = {}
            for timeframe, df in tf_map.items():
                normalized_pairs[timeframe] = _ensure_timestamp_index(df)
            normalized[pair] = self._ensure_required_timeframes(normalized_pairs)
        return normalized

    def _load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        base_timeframe = min(self.timeframes, key=self._timeframe_seconds)
        for pair in self.pairs:
            base_df = load_ohlcv(pair, self.start, self.end, timeframe=base_timeframe)
            base_df = _ensure_timestamp_index(base_df)
            tf_map: Dict[str, pd.DataFrame] = {}
            for timeframe in self.timeframes:
                if timeframe == base_timeframe:
                    tf_map[timeframe] = base_df
                else:
                    tf_map[timeframe] = self._resample_timeframe(base_df, base_timeframe, timeframe)
            data[pair] = tf_map
        return data

    def _precompute_indicator_frames(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        frames: Dict[str, Dict[str, pd.DataFrame]] = {}
        for pair, tf_map in self.market_data.items():
            pair_frames: Dict[str, pd.DataFrame] = {}
            for timeframe, df in tf_map.items():
                config = self.window_configs.get(timeframe) or IndicatorWindowConfig(timeframe=timeframe)
                pair_frames[timeframe] = precompute_indicator_frame(df, config=config)
            frames[pair] = pair_frames
        return frames

    def _indicator_snapshot(self, symbol: str, timeframe: str, timestamp: datetime) -> IndicatorSnapshot | None:
        frame = self.indicator_frames.get(symbol, {}).get(timeframe)
        if frame is not None:
            snapshot = snapshot_from_frame(frame, timestamp, symbol, timeframe)
            if snapshot is not None:
                return snapshot
        df = self.market_data.get(symbol, {}).get(timeframe)
        if df is None:
            return None
        subset = df[df.index <= timestamp]
        if subset.empty:
            return None
        return compute_indicator_snapshot(subset, symbol=symbol, timeframe=timeframe, config=self.window_configs[timeframe])

    def _ensure_required_timeframes(self, tf_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if all(tf in tf_map for tf in self.timeframes):
            return tf_map
        if not tf_map:
            return tf_map
        base_tf = min(tf_map.keys(), key=self._timeframe_seconds)
        base_df = tf_map[base_tf]
        for timeframe in self.timeframes:
            if timeframe not in tf_map:
                tf_map[timeframe] = self._resample_timeframe(base_df, base_tf, timeframe)
        return tf_map

    def _timeframe_seconds(self, timeframe: str) -> int:
        units = {"m": 60, "h": 3600, "d": 86400}
        suffix = timeframe[-1]
        value = int(timeframe[:-1])
        return value * units[suffix]

    def _derive_higher_timeframes(self, base_timeframe: str) -> List[str]:
        """Derive commonly-used higher timeframes from a base timeframe.

        For multi-timeframe analysis, the LLM needs access to higher timeframes
        for trend confirmation. This method automatically derives reasonable
        higher timeframes that can be computed from the base.

        Examples:
            - 5m base -> derives 15m, 1h, 4h
            - 15m base -> derives 1h, 4h
            - 1h base -> derives 4h

        Returns:
            List of timeframe strings including the base and derived timeframes.
        """
        base_seconds = self._timeframe_seconds(base_timeframe)

        # Common timeframes used for multi-timeframe analysis
        # Only include timeframes that are cleanly divisible from base
        candidate_timeframes = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"]

        derived = [base_timeframe]
        for tf in candidate_timeframes:
            tf_seconds = self._timeframe_seconds(tf)
            # Only derive timeframes larger than base that divide evenly
            if tf_seconds > base_seconds and tf_seconds % base_seconds == 0:
                # Limit to reasonable derivations (max 4h from minute data, max 1d from hourly)
                ratio = tf_seconds // base_seconds
                if ratio <= 288:  # Max 288 bars to aggregate (e.g., 5m -> 1d = 288)
                    derived.append(tf)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for tf in derived:
            if tf not in seen:
                seen.add(tf)
                result.append(tf)

        return result

    def _pandas_rule(self, timeframe: str) -> str:
        value = int(timeframe[:-1])
        suffix = timeframe[-1]
        if suffix == "m":
            return f"{value}T"
        if suffix == "h":
            return f"{value}h"
        if suffix == "d":
            return f"{value}d"
        raise ValueError(f"Unsupported timeframe {timeframe}")

    def _resample_timeframe(self, base_df: pd.DataFrame, base_timeframe: str, target_timeframe: str) -> pd.DataFrame:
        base_seconds = self._timeframe_seconds(base_timeframe)
        target_seconds = self._timeframe_seconds(target_timeframe)
        if target_seconds % base_seconds != 0:
            raise ValueError(f"Cannot derive {target_timeframe} from base timeframe {base_timeframe}")
        rule = self._pandas_rule(target_timeframe)
        agg = base_df.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        return agg

    def _normalize_datetime(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.tz_convert(timezone.utc).to_pydatetime()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _asset_states(self, timestamp: datetime) -> Dict[str, AssetState]:
        asset_states: Dict[str, AssetState] = {}
        for pair, tf_map in self.market_data.items():
            snapshots: List[IndicatorSnapshot] = []
            for timeframe in tf_map.keys():
                snapshot = self._indicator_snapshot(pair, timeframe, timestamp)
                if snapshot is not None:
                    snapshots.append(snapshot)
            if snapshots:
                asset_states[pair] = build_asset_state(pair, snapshots)
        return asset_states

    def _previous_triggers_snapshot(self, plan: StrategyPlan | None) -> List[TriggerSummary]:
        if not plan:
            return []
        summaries: List[TriggerSummary] = []
        for trigger in plan.triggers[:50]:
            summaries.append(
                TriggerSummary(
                    id=trigger.id,
                    symbol=trigger.symbol,
                    timeframe=trigger.timeframe,
                    direction=trigger.direction,
                    category=trigger.category,
                    confidence_grade=trigger.confidence_grade,
                    entry_rule=trigger.entry_rule,
                    exit_rule=trigger.exit_rule,
                    hold_rule=trigger.hold_rule,
                    stop_loss_pct=trigger.stop_loss_pct,
                )
            )
        return summaries

    def _llm_input(
        self,
        timestamp: datetime,
        asset_states: Dict[str, AssetState],
        previous_plan: StrategyPlan | None = None,
    ) -> LLMInput:
        portfolio_state = self.portfolio.portfolio_state(timestamp)
        context: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "pairs": self.pairs,
            # CRITICAL: Tell the LLM which timeframes are available for cross-timeframe references
            "available_timeframes": list(self.timeframes),
        }
        market_structure_context: Dict[str, Any] = {}
        if self.last_slot_report:
            context["recent_activity"] = self.last_slot_report
            market_structure_context = self.last_slot_report.get("market_structure", {}) or {}
        if self.latest_daily_summary:
            context["latest_daily_report"] = self.latest_daily_summary
            context["judge_feedback"] = self.latest_daily_summary.get("judge_feedback")
            if not market_structure_context:
                market_structure_context = self.latest_daily_summary.get("market_structure", {}) or {}
            if not self.latest_factor_exposures:
                self.latest_factor_exposures = self.latest_daily_summary.get("factor_exposures", {}) or {}
        context["strategy_memory"] = build_strategy_memory(self.memory_history)
        if self.judge_constraints:
            context["strategist_constraints"] = self.judge_constraints
        if self.active_risk_adjustments:
            context["risk_adjustments"] = list(snapshot_adjustments(self.active_risk_adjustments))
        context["risk_limits"] = self.active_risk_limits.to_risk_params()
        if self.rpr_comparison_snapshot:
            context["rpr_comparison"] = self.rpr_comparison_snapshot
        context["market_structure"] = market_structure_context
        if self.latest_factor_exposures:
            context["factor_exposures"] = self.latest_factor_exposures
        if self.auto_hedge_market:
            context["auto_hedge_mode"] = "market"
        return LLMInput(
            portfolio=portfolio_state,
            assets=list(asset_states.values()),
            risk_params=self.risk_params,
            global_context=context,
            market_structure=market_structure_context,
            previous_triggers=self._previous_triggers_snapshot(previous_plan),
        )

    def _refresh_risk_state_from_run(self, run) -> None:
        self.base_risk_limits = run.config.risk_limits or self.base_risk_limits
        self.active_risk_adjustments = dict(run.risk_adjustments or {})
        self.active_risk_limits = effective_risk_limits(run)
        self.risk_params = self.active_risk_limits.to_risk_params()
        self.risk_profile = build_risk_profile(run)
        # Preserve externally loaded archetype/hour multipliers (baseline-relative caps).
        arch_mults = getattr(self, "_loaded_archetype_multipliers", {}) or {}
        arch_hour_mults = getattr(self, "_loaded_archetype_hour_multipliers", {}) or {}
        if arch_mults:
            self.risk_profile.archetype_multipliers.update(arch_mults)
        if arch_hour_mults:
            self.risk_profile.archetype_hour_multipliers.update(arch_hour_mults)
        self.daily_risk_budget_pct = self.active_risk_limits.max_daily_risk_budget_pct

    def _ensure_strategy_run(self, run_id: str) -> None:
        try:
            run = self.run_registry.get_strategy_run(run_id)
        except KeyError:
            history_days = max(1, (self.end - self.start).days if self.start and self.end else 30)
            cadence_hours = max(1, int(self.plan_interval.total_seconds() // 3600) or 1)
            metadata: Dict[str, Any] = {}
            if self.session_trade_multipliers:
                metadata["session_trade_multipliers"] = self.session_trade_multipliers
            if self.timeframe_trigger_caps:
                metadata["timeframe_trigger_caps"] = self.timeframe_trigger_caps
            if self.flatten_policy:
                metadata["flatten_policy"] = self.flatten_policy
            if self.debug_trigger_sample_rate > 0:
                metadata["debug_trigger_sample_rate"] = self.debug_trigger_sample_rate
            if self.debug_trigger_max_samples:
                metadata["debug_trigger_max_samples"] = self.debug_trigger_max_samples
            if self.indicator_debug_mode:
                metadata["indicator_debug_mode"] = self.indicator_debug_mode
            if self.indicator_debug_keys:
                metadata["indicator_debug_keys"] = list(self.indicator_debug_keys)
            config = StrategyRunConfig(
                symbols=self.pairs,
                timeframes=self.timeframes,
                history_window_days=history_days,
                plan_cadence_hours=cadence_hours,
                debug_trigger_sample_rate=self.debug_trigger_sample_rate or None,
                debug_trigger_max_samples=self.debug_trigger_max_samples or None,
                indicator_debug_mode=self.indicator_debug_mode or None,
                indicator_debug_keys=list(self.indicator_debug_keys),
                risk_limits=self.base_risk_limits,
                metadata=metadata,
            )
            run = self.run_registry.create_strategy_run(config=config, run_id=run_id)
        else:
            if self.session_trade_multipliers and not run.config.metadata.get("session_trade_multipliers"):
                run.config.metadata["session_trade_multipliers"] = self.session_trade_multipliers
            if self.timeframe_trigger_caps and not run.config.metadata.get("timeframe_trigger_caps"):
                run.config.metadata["timeframe_trigger_caps"] = self.timeframe_trigger_caps
            if self.flatten_policy and not run.config.metadata.get("flatten_policy"):
                run.config.metadata["flatten_policy"] = self.flatten_policy
            if self.debug_trigger_sample_rate > 0 and not run.config.metadata.get("debug_trigger_sample_rate"):
                run.config.metadata["debug_trigger_sample_rate"] = self.debug_trigger_sample_rate
            if self.debug_trigger_max_samples and not run.config.metadata.get("debug_trigger_max_samples"):
                run.config.metadata["debug_trigger_max_samples"] = self.debug_trigger_max_samples
            if self.indicator_debug_mode and not run.config.metadata.get("indicator_debug_mode"):
                run.config.metadata["indicator_debug_mode"] = self.indicator_debug_mode
            if self.indicator_debug_keys and not run.config.metadata.get("indicator_debug_keys"):
                run.config.metadata["indicator_debug_keys"] = list(self.indicator_debug_keys)
            if run.config.debug_trigger_sample_rate is None and self.debug_trigger_sample_rate:
                run.config.debug_trigger_sample_rate = self.debug_trigger_sample_rate
            if run.config.debug_trigger_max_samples is None and self.debug_trigger_max_samples:
                run.config.debug_trigger_max_samples = self.debug_trigger_max_samples
            if run.config.indicator_debug_mode is None and self.indicator_debug_mode:
                run.config.indicator_debug_mode = self.indicator_debug_mode
            if not run.config.indicator_debug_keys and self.indicator_debug_keys:
                run.config.indicator_debug_keys = list(self.indicator_debug_keys)
            run = self.run_registry.update_strategy_run(run)
        self._refresh_risk_state_from_run(run)

    def _default_factor_data(self) -> pd.DataFrame | None:
        """Build simple crypto factor proxies from available market data (EW market, BTC dominance, ETH/BTC)."""

        if not self.market_data:
            return None
        base_tf = self.timeframes[0]
        closes: Dict[str, pd.Series] = {}
        for symbol, tf_map in self.market_data.items():
            df = tf_map.get(base_tf)
            if df is not None and "close" in df.columns:
                closes[symbol] = df["close"]
        if not closes:
            return None
        aligned = pd.concat(closes.values(), axis=1, join="inner")
        aligned.columns = list(closes.keys())
        market_eqw = aligned.mean(axis=1)
        btc = aligned.get("BTC-USD")
        eth = aligned.get("ETH-USD")
        dominance = None
        if btc is not None:
            total = aligned.sum(axis=1)
            dominance = (btc / total.replace(0.0, pd.NA)).dropna()
        eth_ratio = None
        if btc is not None and eth is not None:
            eth_ratio = eth / btc
        factors = pd.DataFrame({"market": market_eqw.pct_change()})
        if dominance is not None:
            factors["dominance"] = dominance.pct_change()
        if eth_ratio is not None:
            factors["eth_beta"] = eth_ratio.pct_change()
        return factors.dropna()

    def _load_factor_data(self, path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        try:
            from data_loader.factors import load_cached_factors
        except Exception as exc:  # pragma: no cover - optional import
            logger.warning("Could not import factor loader: %s", exc)
            return None
        try:
            df = load_cached_factors(path)
            return df
        except FileNotFoundError:
            logger.warning("Factor data path not found: %s", path)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load factor data from %s: %s", path, exc)
            return None

    def _factor_exposures(self, timestamp: datetime) -> Dict[str, Dict[str, Any]]:
        """Compute factor betas if factor data is available."""

        if self.factor_data is None or self.factor_data.empty:
            return {}
        base_tf = self.timeframes[0]
        frames: Dict[str, pd.DataFrame] = {}
        for symbol, tf_map in self.market_data.items():
            df = tf_map.get(base_tf)
            if df is None:
                continue
            subset = df[df.index <= timestamp]
            if subset.empty:
                continue
            frames[symbol] = subset
        if not frames:
            return {}
        factors_subset = self.factor_data[self.factor_data.index <= timestamp]
        if factors_subset.empty:
            return {}
        exposures = compute_factor_loadings(frames, factors_subset)
        return {symbol: exp.to_dict() for symbol, exp in exposures.items()}

    def _build_bar(self, pair: str, timeframe: str, timestamp: datetime) -> Bar:
        df = self.market_data[pair][timeframe]
        row = df.loc[timestamp]
        return Bar(
            symbol=pair,
            timeframe=timeframe,
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )

    def _coerce_timestamp(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, pd.Timestamp):
            ts = value.to_pydatetime()
        elif isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _collect_fill_details(
        self,
        day_key: str,
        since_ts: datetime | None,
        until_ts: datetime,
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for fill in self.portfolio.fills:
            ts = self._coerce_timestamp(fill.get("timestamp"))
            if ts is None:
                continue
            if ts.strftime("%Y-%m-%d") != day_key:
                continue
            if since_ts and ts <= since_ts:
                continue
            if ts > until_ts:
                continue
            details.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "qty": fill.get("qty") or fill.get("quantity"),
                    "price": fill.get("price"),
                    "timeframe": fill.get("timeframe"),
                    "trigger_id": fill.get("reason") or fill.get("trigger_id"),
                    "pnl": fill.get("pnl") if fill.get("pnl") is not None else fill.get("realized_pnl"),
                    "market_structure_entry": fill.get("market_structure_entry"),
                }
            )
        return details

    def _build_trigger_attempt_stats(
        self,
        day_key: str,
        since_ts: datetime | None,
        until_ts: datetime,
    ) -> Dict[str, Any]:
        limit_entry = self.limit_enforcement_by_day.get(day_key)
        if not limit_entry:
            return {}
        stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"attempted": 0, "executed": 0, "blocked": 0, "blocked_by_reason": defaultdict(int)}
        )

        def _in_window(ts: datetime) -> bool:
            if since_ts and ts <= since_ts:
                return False
            return ts <= until_ts

        for entry in limit_entry.get("executed_details", []):
            ts = self._coerce_timestamp(entry.get("timestamp"))
            if ts is None or not _in_window(ts):
                continue
            trigger_id = entry.get("trigger_id") or "unknown"
            stat = stats[trigger_id]
            stat["attempted"] += 1
            stat["executed"] += 1

        for entry in limit_entry.get("blocked_details", []):
            ts = self._coerce_timestamp(entry.get("timestamp"))
            if ts is None or not _in_window(ts):
                continue
            trigger_id = entry.get("trigger_id") or "unknown"
            reason = entry.get("reason") or "unknown"
            stat = stats[trigger_id]
            stat["attempted"] += 1
            stat["blocked"] += 1
            stat["blocked_by_reason"][reason] += 1

        payload: Dict[str, Any] = {}
        for trigger_id, stat in stats.items():
            reasons = stat.get("blocked_by_reason", {})
            payload[trigger_id] = {
                "attempted": stat["attempted"],
                "executed": stat["executed"],
                "blocked": stat["blocked"],
                "blocked_by_reason": dict(reasons),
            }
        return payload

    def _trigger_signature(self, trigger: "TriggerCondition") -> tuple:
        return (
            trigger.symbol,
            trigger.timeframe,
            trigger.direction,
            trigger.category,
            trigger.confidence_grade,
            trigger.entry_rule,
            trigger.exit_rule,
            trigger.hold_rule,
            trigger.stop_loss_pct,
        )

    def _diff_plan_triggers(
        self,
        previous_plan: StrategyPlan | None,
        current_plan: StrategyPlan,
        limit: int = 10,
    ) -> Dict[str, Any]:
        if previous_plan is None:
            return {
                "previous_plan_id": None,
                "added": len(current_plan.triggers),
                "removed": 0,
                "changed": 0,
                "unchanged": 0,
                "added_ids": [t.id for t in current_plan.triggers[:limit]],
                "removed_ids": [],
                "changed_ids": [],
            }
        prev_map = {t.id: t for t in previous_plan.triggers}
        curr_map = {t.id: t for t in current_plan.triggers}
        prev_ids = set(prev_map)
        curr_ids = set(curr_map)
        added_ids = sorted(curr_ids - prev_ids)
        removed_ids = sorted(prev_ids - curr_ids)
        changed_ids: List[str] = []
        unchanged_ids: List[str] = []
        for trigger_id in sorted(prev_ids & curr_ids):
            if self._trigger_signature(prev_map[trigger_id]) != self._trigger_signature(curr_map[trigger_id]):
                changed_ids.append(trigger_id)
            else:
                unchanged_ids.append(trigger_id)
        return {
            "previous_plan_id": previous_plan.plan_id,
            "added": len(added_ids),
            "removed": len(removed_ids),
            "changed": len(changed_ids),
            "unchanged": len(unchanged_ids),
            "added_ids": added_ids[:limit],
            "removed_ids": removed_ids[:limit],
            "changed_ids": changed_ids[:limit],
        }

    @staticmethod
    def _plan_constraints_signature(plan: StrategyPlan) -> tuple:
        risk = plan.risk_constraints
        trigger_budgets = tuple(sorted((plan.trigger_budgets or {}).items()))
        sizing_rules = tuple(
            sorted(
                (
                    rule.symbol,
                    rule.sizing_mode,
                    rule.target_risk_pct,
                    rule.vol_target_annual,
                    rule.notional,
                )
                for rule in plan.sizing_rules
            )
        )
        return (
            risk.max_position_risk_pct,
            risk.max_symbol_exposure_pct,
            risk.max_portfolio_exposure_pct,
            risk.max_daily_loss_pct,
            risk.max_daily_risk_budget_pct,
            plan.max_trades_per_day,
            plan.min_trades_per_day,
            plan.max_triggers_per_symbol_per_day,
            tuple(sorted(plan.allowed_symbols or [])),
            tuple(sorted(plan.allowed_directions or [])),
            tuple(sorted(plan.allowed_trigger_categories or [])),
            trigger_budgets,
            sizing_rules,
        )

    def _is_no_change_replan(
        self,
        previous_plan: StrategyPlan | None,
        current_plan: StrategyPlan,
        trigger_diff: Dict[str, Any] | None,
    ) -> bool:
        if previous_plan is None or trigger_diff is None:
            return False
        triggers_unchanged = (
            trigger_diff.get("added", 0) == 0
            and trigger_diff.get("removed", 0) == 0
            and trigger_diff.get("changed", 0) == 0
            and trigger_diff.get("unchanged", 0) == len(previous_plan.triggers)
            and len(current_plan.triggers) == len(previous_plan.triggers)
        )
        if not triggers_unchanged:
            return False
        return self._plan_constraints_signature(previous_plan) == self._plan_constraints_signature(current_plan)

    def _assess_risk_limits(self, order: Order, portfolio_state: PortfolioState) -> List[str]:
        """Check a proposed order against the configured risk knobs."""

        if not order:
            return []
        equity = max(portfolio_state.equity, 0.0)
        if equity <= 0:
            return []
        warnings: List[str] = []
        notional = max(order.quantity * order.price, 0.0)
        if notional <= 0:
            return warnings
        limits = self.active_risk_limits

        def _cap(pct: float) -> float:
            return equity * (pct / 100.0)
        trade_cap = _cap(limits.max_position_risk_pct)
        if trade_cap > 0 and notional > trade_cap + 1e-9:
            warnings.append("max_position_risk_pct")
        current_symbol = abs(portfolio_state.positions.get(order.symbol, 0.0)) * order.price
        projected_symbol = current_symbol + notional
        symbol_cap = _cap(limits.max_symbol_exposure_pct)
        if symbol_cap > 0 and projected_symbol > symbol_cap + 1e-9:
            warnings.append("max_symbol_exposure_pct")
        gross_exposure = max(portfolio_state.equity - portfolio_state.cash, 0.0)
        projected_gross = gross_exposure + notional
        portfolio_cap = _cap(limits.max_portfolio_exposure_pct)
        if portfolio_cap > 0 and projected_gross > portfolio_cap + 1e-9:
            warnings.append("max_portfolio_exposure_pct")
        return warnings

    def _archetype_for_trigger(self, day_key: str, trigger_id: str, symbol: str | None = None) -> str:
        if symbol:
            return str(symbol).split("-")[0].lower()
        plan_limits = self.plan_limits_by_day.get(day_key) or {}
        catalog = plan_limits.get("trigger_catalog") or {}
        entry = catalog.get(trigger_id) or catalog.get(trigger_id.split("_exit")[0]) or {}
        category = entry.get("category") or entry.get("direction")
        if category:
            return str(category)
        # fallback: first token before underscore
        return trigger_id.split("_")[0] if trigger_id else "unknown"

    def _process_orders_with_limits(
        self,
        run_id: str,
        day_key: str,
        orders: List[Order],
        portfolio_state: PortfolioState,
        plan_payload: Dict[str, Any] | None,
        compiled_payload: Dict[str, Any] | None,
        blocked_entries: List[dict] | None = None,
        current_load: int = 0,
        market_structure: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        executed_records: List[Dict[str, Any]] = []
        risk_engine = getattr(self, "latest_risk_engine", None)
        limit_entry = self.limit_enforcement_by_day[day_key]
        reason_map = set(item.value for item in BlockReason)
        custom_reasons = {
            "timeframe_cap",
            "session_cap",
            "trigger_load",
            "min_hold",
            "min_flat",
            "emergency_exit_veto_same_bar",
            "emergency_exit_veto_min_hold",
            "emergency_exit_missing_exit_rule",
            "emergency_exit_executed",
        }
        plan_id = (plan_payload or {}).get("plan_id")

        def _base_trigger_id(reason: str) -> str:
            for suffix in ("_exit", "_flat"):
                if reason.endswith(suffix):
                    return reason[: -len(suffix)]
            return reason

        def _trigger_category(trigger_id: str) -> str | None:
            plan_limits = self.plan_limits_by_day.get(day_key) or {}
            catalog = plan_limits.get("trigger_catalog") or {}
            entry = catalog.get(trigger_id)
            if isinstance(entry, dict):
                return entry.get("category")
            return None

        def _min_hold_block(order: Order) -> str | None:
            if self.min_hold_hours <= 0:
                return None
            if order.reason == "eod_flatten":
                return None
            # IMPORTANT: Use self.portfolio.positions (live state) instead of
            # portfolio_state.positions (snapshot) to get accurate current position
            current_qty = self.portfolio.positions.get(order.symbol, 0.0)
            if abs(current_qty) <= 1e-9:
                return None
            if (current_qty > 0 and order.side == "buy") or (current_qty < 0 and order.side == "sell"):
                return None
            opened_at = self.portfolio.position_meta.get(order.symbol, {}).get("opened_at")
            if not isinstance(opened_at, datetime):
                return None
            if order.timestamp - opened_at >= self.min_hold_window:
                return None
            trigger_id = _base_trigger_id(order.reason)
            elapsed_minutes = int((order.timestamp - opened_at).total_seconds() // 60)
            return f"Min hold {self.min_hold_hours:.2f}h not met; open for {elapsed_minutes}m"

        def _min_flat_block(order: Order) -> str | None:
            if self.min_flat_hours <= 0:
                return None
            if order.reason == "eod_flatten" or order.reason.endswith("_exit") or order.reason.endswith("_flat"):
                return None
            # IMPORTANT: Use self.portfolio.positions (live state) instead of
            # portfolio_state.positions (snapshot) to get accurate current position
            current_qty = self.portfolio.positions.get(order.symbol, 0.0)
            if abs(current_qty) > 1e-9:
                return None
            last_flat_trigger = self.last_flat_trigger_by_symbol.get(order.symbol)
            current_trigger = _base_trigger_id(order.reason)
            if last_flat_trigger and current_trigger != last_flat_trigger:
                return None
            last_flat = self.last_flat_time_by_symbol.get(order.symbol)
            if not isinstance(last_flat, datetime):
                return None
            if order.timestamp - last_flat >= self.min_flat_window:
                return None
            elapsed_minutes = int((order.timestamp - last_flat).total_seconds() // 60)
            return f"Min flat {self.min_flat_hours:.2f}h not met; flat for {elapsed_minutes}m"

        def _structure_fields(snapshot: Mapping[str, Any] | None) -> Dict[str, Any]:
            if not snapshot:
                return {}
            return {
                "distance_to_support_pct": snapshot.get("distance_to_support_pct"),
                "distance_to_resistance_pct": snapshot.get("distance_to_resistance_pct"),
                "trend": snapshot.get("trend"),
            }

        def _normalized_reason(raw: str | None) -> str:
            if not raw:
                return BlockReason.OTHER.value
            if raw in reason_map or raw in custom_reasons:
                return raw
            return BlockReason.RISK.value

        def _record_block_entry(data: Dict[str, Any], source: str) -> None:
            reason = data.get("reason")
            normalized = _normalized_reason(reason)
            limit_entry["skipped"][normalized] += 1
            self.skipped_activity_by_day[day_key][normalized] += 1
            # Track block breakdown for all reasons (including session/archetype/load caps).
            if normalized == BlockReason.RISK.value:
                key = reason or BlockReason.RISK.value
                limit_entry["risk_block_breakdown"][key] += 1
            else:
                limit_entry["risk_block_breakdown"][normalized] += 1
            entry = {
                "timestamp": data.get("timestamp"),
                "symbol": data.get("symbol"),
                "side": data.get("side"),
                "price": data.get("price"),
                "quantity": data.get("quantity"),
                "timeframe": data.get("timeframe"),
                "trigger_id": data.get("trigger_id"),
                "reason": reason or normalized,
                "detail": data.get("detail"),
                "source": source,
            }
            extra_keys = {
                "timestamp",
                "symbol",
                "side",
                "price",
                "quantity",
                "timeframe",
                "trigger_id",
                "reason",
                "detail",
                "source",
            }
            for key, value in data.items():
                if key in extra_keys or key in entry:
                    continue
                entry[key] = value
            if normalized == BlockReason.OTHER.value and not entry["detail"]:
                entry["detail"] = "Unspecified block; execution engine returned no detail"
                logger.warning("Blocked trade missing detail; normalizing to OTHER: %s", entry)
            limit_entry["blocked_details"].append(entry)
            stats = self.trigger_activity_by_day[day_key][data.get("trigger_id", "unknown")]
            stats["blocked"] += 1
            stats.setdefault("blocked_by_reason", defaultdict(int))
            stats["blocked_by_reason"][normalized] += 1
            msg = data.get("detail")
            if msg:
                logger.info("Blocked trade (%s): %s", entry["reason"], msg)

        def _record_execution_detail(order: Order, source: str, risk_used: float | None = None, latency_seconds: float | None = None) -> None:
            payload = {
                "timestamp": order.timestamp.isoformat(),
                "symbol": order.symbol,
                "side": order.side,
                "price": order.price,
                "quantity": order.quantity,
                "timeframe": order.timeframe,
                "trigger_id": order.reason,
                "source": source,
                "risk_used": risk_used,
                "latency_seconds": latency_seconds,
            }
            if getattr(order, "emergency", False):
                payload["reason"] = "emergency_exit_executed"
                payload["outcome"] = "executed"
            cooldown = getattr(order, "cooldown_recommendation_bars", None)
            if cooldown is not None:
                payload["cooldown_recommendation_bars"] = cooldown
            limit_entry["executed_details"].append(payload)

        def _record_hints(order: Order) -> None:
            hints = self._assess_risk_limits(order, portfolio_state)
            if not hints:
                return
            for hint in hints:
                limit_entry["risk_limit_hints"][hint] += 1

        def _adjust_budget_post_fill(day_key: str, symbol: str | None, planned: float | None, risk_snapshot: Mapping[str, Any] | None) -> None:
            if planned is None or planned <= 0:
                return
            if not risk_snapshot:
                return
            actual = risk_snapshot.get("actual_risk_abs")
            if actual is None:
                return
            entry = self.daily_risk_budget_state.get(day_key)
            if not entry:
                return
            delta = float(actual) - float(planned)
            # Record slippage between planned allocation and observed risk, but do not
            # retroactively shrink budget usage. Budget consumption stays at planned risk;
            # actual risk is tracked separately via risk_usage_events.
            entry["slippage_adjustment"] = entry.get("slippage_adjustment", 0.0) + delta

        if blocked_entries:
            for block in blocked_entries:
                if block.get("reason") in {"SIGNAL_PRIORITY", "priority_skip"}:
                    limit_entry["priority_skips"] += 1
                    continue
                limit_entry["trades_attempted"] += 1
                _record_block_entry(block, "trigger_engine")

        if not plan_payload or not compiled_payload:
            for order in orders:
                limit_entry["trades_attempted"] += 1
                hold_detail = _min_hold_block(order)
                if hold_detail:
                    reason = "min_hold"
                    payload = {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": reason,
                        "detail": hold_detail,
                    }
                    if getattr(order, "emergency", False):
                        reason = "emergency_exit_veto_min_hold"
                        payload["reason"] = reason
                        cooldown = getattr(order, "cooldown_recommendation_bars", None)
                        if cooldown is not None:
                            payload["cooldown_recommendation_bars"] = cooldown
                    _record_block_entry(
                        payload,
                        "min_hold",
                    )
                    continue
                flat_detail = _min_flat_block(order)
                if flat_detail:
                    _record_block_entry(
                        {
                            "timestamp": order.timestamp.isoformat(),
                            "symbol": order.symbol,
                            "side": order.side,
                            "price": order.price,
                            "quantity": order.quantity,
                            "timeframe": order.timeframe,
                            "trigger_id": order.reason,
                            "reason": "min_flat",
                            "detail": flat_detail,
                        },
                        "min_flat",
                    )
                    continue
                risk_snapshot = getattr(risk_engine, "last_risk_snapshot", {}) if risk_engine else {}
                _record_hints(order)
                allowance = self._risk_budget_allowance(day_key, order)
                if allowance is None:
                    self._record_risk_budget_block(day_key, order.symbol)
                    _record_block_entry(
                        {
                            "timestamp": order.timestamp.isoformat(),
                            "symbol": order.symbol,
                            "side": order.side,
                            "price": order.price,
                            "quantity": order.quantity,
                            "timeframe": order.timeframe,
                            "trigger_id": order.reason,
                            "reason": BlockReason.RISK_BUDGET.value,
                            "detail": "Daily risk budget exhausted",
                        },
                        "risk_budget",
                    )
                    if plan_id:
                        execution_tools.engine.revert_trade(run_id, plan_id, order.timestamp, order.symbol, order.timeframe)
                    continue
                self._commit_risk_budget(day_key, allowance, order.symbol)
                limit_entry["trades_executed"] += 1
                trigger_id = order.reason
                risk_used = allowance or 0.0
                self.risk_usage_by_day[day_key][(trigger_id, order.timeframe)] += risk_used
                actual_risk = None
                if risk_snapshot:
                    actual_risk = risk_snapshot.get("actual_risk_abs")
                if actual_risk is None:
                    stop_dist = order.stop_distance or (risk_snapshot.get("stop_distance") if risk_snapshot else None)
                    if stop_dist is not None:
                        actual_risk = max(order.quantity * stop_dist, 0.0)
                if actual_risk is None:
                    actual_risk = risk_used
                self.risk_usage_events_by_day[day_key].append(
                    {
                        "trigger_id": trigger_id,
                        "timeframe": order.timeframe,
                        "hour": order.timestamp.hour,
                        "risk_used": risk_used,
                        "actual_risk_at_stop": actual_risk,
                        "profile_multiplier": risk_snapshot.get("profile_multiplier") if risk_snapshot else None,
                        "profile_multiplier_components": risk_snapshot.get("profile_multiplier_components") if risk_snapshot else None,
                        "archetype": risk_snapshot.get("archetype") if risk_snapshot else None,
                    }
                )
                self.timeframe_exec_counts[day_key][str(order.timeframe)] += 1
                session_window = None
                if self.session_trade_multipliers:
                    session_window = next(
                        (
                            f"{win.get('start_hour', -1)}-{win.get('end_hour', -1)}"
                            for win in self.session_trade_multipliers
                            if int(win.get("start_hour", -1)) <= order.timestamp.hour < int(win.get("end_hour", -1))
                        ),
                        None,
                    )
                if session_window:
                    self.session_trade_counts[day_key][session_window] += 1
                _record_execution_detail(order, "trigger_engine", risk_used=risk_used, latency_seconds=0.0)
                structure_snapshot = (market_structure or {}).get(order.symbol) if market_structure else None
                risk_snapshot = getattr(risk_engine, "last_risk_snapshot", {}) if risk_engine else {}
                pre_qty = self.portfolio.positions.get(order.symbol, 0.0)
                self.portfolio.execute(order, market_structure_entry=structure_snapshot)
                post_qty = self.portfolio.positions.get(order.symbol, 0.0)
                # Track fill in trigger engine for cooldown/hold period enforcement
                is_entry = abs(pre_qty) <= 1e-9 and abs(post_qty) > 1e-9
                if self.current_trigger_engine:
                    self.current_trigger_engine.record_fill(order.symbol, is_entry, order.timestamp)
                if abs(pre_qty) > 1e-9 and abs(post_qty) <= 1e-9:
                    self.last_flat_time_by_symbol[order.symbol] = order.timestamp
                    self.last_flat_trigger_by_symbol[order.symbol] = _base_trigger_id(order.reason)
                structure_fields = _structure_fields(structure_snapshot)
                record = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "timeframe": order.timeframe,
                    "reason": order.reason,
                    "risk_used": allowance,
                    "latency_seconds": 0.0,
                    "market_structure_entry": structure_snapshot,
                }
                if risk_snapshot:
                    record["allocated_risk_abs"] = risk_snapshot.get("allocated_risk_abs")
                    record["actual_risk_at_stop"] = risk_snapshot.get("actual_risk_abs") or actual_risk
                    record["stop_distance"] = order.stop_distance or risk_snapshot.get("stop_distance")
                    if risk_snapshot.get("profile_multiplier") is not None:
                        record["profile_multiplier"] = risk_snapshot.get("profile_multiplier")
                        record["profile_multiplier_components"] = risk_snapshot.get("profile_multiplier_components")
                        record["profile_archetype"] = risk_snapshot.get("archetype")
                _adjust_budget_post_fill(day_key, order.symbol, risk_used, risk_snapshot)
                record.update(structure_fields)
                executed_records.append(record)
                self.trigger_activity_by_day[day_key][order.reason]["executed"] += 1

                # Track trade event for progress reporting
                self._add_event("trade", {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "trigger": order.reason,
                })

                logger.debug(
                    "Executed order (no plan gating) trigger=%s symbol=%s side=%s qty=%.6f price=%.2f",
                    order.reason,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price,
                )
            return executed_records

        plan_generated_at = None
        if plan_payload:
            generated_at = plan_payload.get("generated_at")
            if isinstance(generated_at, str):
                try:
                    plan_generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                except ValueError:
                    plan_generated_at = None

        def _session_window(ts: datetime) -> tuple[str, float] | None:
            if not self.session_trade_multipliers:
                return None
            hour = ts.hour
            for window in self.session_trade_multipliers:
                start = int(window.get("start_hour", -1))
                end = int(window.get("end_hour", -1))
                if start <= hour < end:
                    multiplier = float(window.get("multiplier", 1.0))
                    return (f"{start}-{end}", multiplier)
            return None

        def _timeframe_cap_reached(timeframe: str) -> bool:
            cap = self.timeframe_trigger_caps.get(str(timeframe))
            if not cap:
                return False
            executed = self.timeframe_exec_counts[day_key].get(str(timeframe), 0)
            return executed >= cap

        def _load_cap_reached(load_val: int) -> bool:
            return self.trigger_load_threshold > 0 and load_val > self.trigger_load_threshold

        def _archetype_state(archetype: str, timeframe: str, ts: datetime) -> tuple[int, float]:
            key = (archetype, str(timeframe), ts.hour)
            current = self.archetype_load_by_day[day_key].get(key, 0)
            return current, key

        def _session_cap_reached(ts: datetime) -> bool:
            window = _session_window(ts)
            if not window:
                return False
            plan_limits = self.plan_limits_by_day.get(day_key) or {}
            policy_cap = plan_limits.get("max_trades_per_day")
            derived_cap = plan_limits.get("derived_max_trades_per_day")
            # Always use the larger of policy vs derived when both exist so session scaling
            # cannot further constrict an already conservative cap.
            if policy_cap and derived_cap:
                base_cap = max(policy_cap, derived_cap)
            else:
                base_cap = derived_cap or policy_cap
            if not base_cap:
                return False
            window_key, multiplier = window
            cap = int(base_cap * multiplier)
            if cap <= 0:
                return False
            executed = self.session_trade_counts[day_key].get(window_key, 0)
            return executed >= cap

        for order in orders:
            timestamp = order.timestamp.isoformat()
            raw_trigger_id = order.reason
            trigger_id = _base_trigger_id(raw_trigger_id)
            archetype = self._archetype_for_trigger(day_key, trigger_id)
            arche_load, arche_key = _archetype_state(archetype, order.timeframe, order.timestamp)
            events_payload = [{"trigger_id": trigger_id, "timestamp": timestamp}]
            hold_detail = _min_hold_block(order)
            if hold_detail:
                limit_entry["trades_attempted"] += 1
                reason = "min_hold"
                payload = {
                    "timestamp": order.timestamp.isoformat(),
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": order.price,
                    "quantity": order.quantity,
                    "timeframe": order.timeframe,
                    "trigger_id": order.reason,
                    "reason": reason,
                    "detail": hold_detail,
                }
                if getattr(order, "emergency", False):
                    reason = "emergency_exit_veto_min_hold"
                    payload["reason"] = reason
                    cooldown = getattr(order, "cooldown_recommendation_bars", None)
                    if cooldown is not None:
                        payload["cooldown_recommendation_bars"] = cooldown
                _record_block_entry(
                    payload,
                    "min_hold",
                )
                continue
            flat_detail = _min_flat_block(order)
            if flat_detail:
                limit_entry["trades_attempted"] += 1
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": "min_flat",
                        "detail": flat_detail,
                    },
                    "min_flat",
                )
                continue
            result = execution_tools.run_live_step_tool(
                run_id,
                plan_payload,
                compiled_payload,
                events_payload,
            )
            events = result.get("events", [])
            event = events[-1] if events else None
            limit_entry["trades_attempted"] += 1
            self.trigger_load_by_day[day_key].append(
                {
                    "timeframe": order.timeframe,
                    "hour": order.timestamp.hour,
                    "load": current_load,
                    "archetype": archetype,
                    "archetype_load": arche_load,
                    "trigger_id": trigger_id,
                }
            )
            if _timeframe_cap_reached(order.timeframe):
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": "timeframe_cap",
                        "detail": "Timeframe cap reached",
                    },
                    "timeframe_cap",
                )
                continue
            if self.archetype_load_threshold > 0 and arche_load >= self.archetype_load_threshold:
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": "archetype_load",
                        "detail": f"Archetype {archetype} load {arche_load} >= threshold {self.archetype_load_threshold}",
                    },
                    "archetype_load",
                )
                continue
            if _load_cap_reached(current_load):
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": "trigger_load",
                        "detail": f"Trigger load {current_load} above threshold {self.trigger_load_threshold}",
                    },
                    "trigger_load",
                )
                continue
            if _session_cap_reached(order.timestamp):
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": "session_cap",
                        "detail": "Session cap reached",
                    },
                    "session_cap",
                )
                continue
            if event and event.get("action") == "executed":
                risk_snapshot = getattr(risk_engine, "last_risk_snapshot", {}) if risk_engine else {}
                _record_hints(order)
                allowance = self._risk_budget_allowance(day_key, order)
                if allowance is None:
                    self._record_risk_budget_block(day_key, order.symbol)
                    _record_block_entry(
                        {
                            "timestamp": order.timestamp.isoformat(),
                            "symbol": order.symbol,
                            "side": order.side,
                            "price": order.price,
                            "quantity": order.quantity,
                            "timeframe": order.timeframe,
                            "trigger_id": order.reason,
                            "reason": BlockReason.RISK_BUDGET.value,
                            "detail": "Daily risk budget exhausted",
                        },
                        "risk_budget",
                    )
                    if plan_id:
                        execution_tools.engine.revert_trade(run_id, plan_id, order.timestamp, order.symbol, order.timeframe)
                    continue
                load_scale = 1.0
                if self.archetype_load_threshold > 0:
                    scale_start = max(1, int(self.archetype_load_threshold * self.archetype_load_scale_start))
                    if arche_load >= scale_start:
                        load_scale = max(0.25, 1 - (arche_load - scale_start) / max(1, (self.archetype_load_threshold - scale_start + 1)))
                adjusted_allowance = allowance * load_scale
                self._commit_risk_budget(day_key, adjusted_allowance, order.symbol)
                limit_entry["trades_executed"] += 1
                latency_seconds = None
                if plan_generated_at:
                    latency_seconds = (order.timestamp - plan_generated_at).total_seconds()
                risk_used = adjusted_allowance or 0.0
                self.risk_usage_by_day[day_key][(raw_trigger_id, order.timeframe)] += risk_used
                actual_risk = None
                if risk_snapshot:
                    actual_risk = risk_snapshot.get("actual_risk_abs")
                if actual_risk is None:
                    stop_dist = order.stop_distance or (risk_snapshot.get("stop_distance") if risk_snapshot else None)
                    if stop_dist is not None:
                        actual_risk = max(order.quantity * stop_dist, 0.0)
                if actual_risk is None:
                    actual_risk = risk_used
                self.risk_usage_events_by_day[day_key].append(
                    {
                        "trigger_id": raw_trigger_id,
                        "timeframe": order.timeframe,
                        "hour": order.timestamp.hour,
                        "risk_used": risk_used,
                        "actual_risk_at_stop": actual_risk,
                        "archetype": archetype,
                        "profile_multiplier": risk_snapshot.get("profile_multiplier") if risk_snapshot else None,
                        "profile_multiplier_components": risk_snapshot.get("profile_multiplier_components") if risk_snapshot else None,
                    }
                )
                if latency_seconds is not None:
                    self.latency_by_day[day_key][(raw_trigger_id, order.timeframe)].append(latency_seconds)
                self.timeframe_exec_counts[day_key][str(order.timeframe)] += 1
                session_window = _session_window(order.timestamp)
                if session_window:
                    self.session_trade_counts[day_key][session_window[0]] += 1
                self.archetype_load_by_day[day_key][arche_key] = arche_load + 1
                _record_execution_detail(order, "execution_engine", risk_used=risk_used, latency_seconds=latency_seconds)
                structure_snapshot = (market_structure or {}).get(order.symbol) if market_structure else None
                risk_snapshot = getattr(risk_engine, "last_risk_snapshot", {}) if risk_engine else {}
                pre_qty = self.portfolio.positions.get(order.symbol, 0.0)
                self.portfolio.execute(order, market_structure_entry=structure_snapshot)
                post_qty = self.portfolio.positions.get(order.symbol, 0.0)
                # Track fill in trigger engine for cooldown/hold period enforcement
                is_entry = abs(pre_qty) <= 1e-9 and abs(post_qty) > 1e-9
                if self.current_trigger_engine:
                    self.current_trigger_engine.record_fill(order.symbol, is_entry, order.timestamp)
                if abs(pre_qty) > 1e-9 and abs(post_qty) <= 1e-9:
                    self.last_flat_time_by_symbol[order.symbol] = order.timestamp
                    self.last_flat_trigger_by_symbol[order.symbol] = _base_trigger_id(order.reason)
                structure_fields = _structure_fields(structure_snapshot)
                record = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "timeframe": order.timeframe,
                    "reason": order.reason,
                    "risk_used": risk_used,
                    "latency_seconds": latency_seconds,
                    "load_scale": load_scale,
                    "archetype": archetype,
                    "market_structure_entry": structure_snapshot,
                }
                if risk_snapshot:
                    record["allocated_risk_abs"] = risk_snapshot.get("allocated_risk_abs")
                    record["actual_risk_at_stop"] = risk_snapshot.get("actual_risk_abs") or actual_risk
                    record["stop_distance"] = order.stop_distance or risk_snapshot.get("stop_distance")
                    record["profile_multiplier"] = risk_snapshot.get("profile_multiplier")
                    record["profile_multiplier_components"] = risk_snapshot.get("profile_multiplier_components")
                    record["profile_archetype"] = risk_snapshot.get("archetype")
                _adjust_budget_post_fill(day_key, order.symbol, risk_used, risk_snapshot)
                record.update(structure_fields)
                executed_records.append(record)
                self.trigger_activity_by_day[day_key][order.reason]["executed"] += 1

                # Track trade event for progress reporting
                self._add_event("trade", {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "trigger": order.reason,
                })

                logger.debug(
                    "Executed order trigger=%s symbol=%s side=%s qty=%.6f price=%.2f",
                    order.reason,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price,
                )
            else:
                detail = event.get("detail") if event else None
                reason = event.get("reason") if event else None
                _record_block_entry(
                    {
                        "timestamp": order.timestamp.isoformat(),
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.price,
                        "quantity": order.quantity,
                        "timeframe": order.timeframe,
                        "trigger_id": order.reason,
                        "reason": reason,
                        "detail": detail,
                    },
                    "execution_engine",
                )
        return executed_records

    def run(self, run_id: str) -> StrategistBacktestResult:
        logger.info(
            "Starting backtest run_id=%s pairs=%s date_range=%s->%s",
            run_id,
            self.pairs,
            self.start,
            self.end,
        )
        self._ensure_strategy_run(run_id)
        all_timestamps = sorted(
            {ts for pair in self.market_data.values() for df in pair.values() for ts in df.index if self.start <= ts <= self.end}
        )

        # Initialize progress tracking
        self.candles_total = len(all_timestamps)
        self.candles_processed = 0
        self.recent_events = []
        logger.info("Backtest will process %d candles", self.candles_total)
        self._report_progress()

        current_plan: StrategyPlan | None = None
        current_plan_payload: Dict[str, Any] | None = None
        current_compiled_payload: Dict[str, Any] | None = None
        trigger_engine: TriggerEngine | None = None
        plan_log: List[Dict[str, Any]] = []
        latest_prices: Dict[str, float] = {symbol: 0.0 for symbol in self.pairs}
        daily_reports: List[Dict[str, Any]] = []
        sizing_targets: Dict[str, float] = {}
        session_flattened: set[str] = set()

        active_assets: Dict[str, AssetState] = {}
        cache_base = Path(self.plan_provider.cache_dir) / run_id
        daily_dir = cache_base / "daily_reports"
        daily_dir.mkdir(parents=True, exist_ok=True)
        self.slot_reports_by_day.clear()
        self.bar_decisions_by_day.clear()
        self.trigger_activity_by_day.clear()
        self.skipped_activity_by_day.clear()
        self.limit_enforcement_by_day.clear()
        self.plan_limits_by_day.clear()
        self.flattened_days.clear()
        self.daily_risk_budget_state.clear()
        self.trigger_load_by_day.clear()
        self.archetype_load_by_day.clear()
        self.daily_loss_anchor_by_day.clear()
        self.llm_generation_by_day.clear()
        self.current_day_key = None
        self.latest_daily_summary = None
        self.last_slot_report = None
        self.current_run_id = run_id

        for ts in all_timestamps:
            # Update progress tracking
            self.candles_processed += 1
            self.current_timestamp = ts

            # Call progress callback every 50 candles or on significant events
            if self.progress_callback and self.candles_processed % 50 == 0:
                self._report_progress()

            day_key = ts.date().isoformat()
            if self.current_day_key and self.current_day_key != day_key:
                if self.flatten_positions_daily and self.last_slot_report:
                    flatten_ts = datetime.fromisoformat(self.last_slot_report["timestamp"])
                    self._flatten_end_of_day(self.current_day_key, flatten_ts, latest_prices)
                summary = self._finalize_day(self.current_day_key, daily_dir, run_id)
                if summary:
                    self.latest_daily_summary = summary
                    self.memory_history.append(summary)
                    judge_constraints = summary.get("judge_feedback", {}).get("strategist_constraints")
                    if judge_constraints:
                        self.judge_constraints = judge_constraints
                    daily_reports.append(summary)
            if self.current_day_key != day_key:
                self.current_day_key = day_key
                # Capture start-of-day equity once for budget base and loss anchor.
                start_equity = self.portfolio.portfolio_state(ts).equity
                self._reset_risk_budget_for_day(day_key, start_equity=start_equity)
                self._set_daily_loss_anchor(day_key, self.portfolio.portfolio_state(ts).equity)
                execution_tools.reset_run_state(run_id)

            # Walk-away check: skip trading if daily profit target already reached
            if self.walk_away_enabled and day_key in self.walk_away_triggered_days:
                # Already triggered walk-away for this day - just update prices and continue
                for pair in self.pairs:
                    base_tf = self.timeframes[0]
                    df = self.market_data.get(pair, {}).get(base_tf)
                    if df is not None and ts in df.index:
                        latest_prices[pair] = float(df.loc[ts, "close"])
                self.portfolio.mark_to_market(ts, latest_prices)
                continue

            # Check if we should trigger walk-away NOW
            if self.walk_away_enabled and day_key not in self.walk_away_triggered_days:
                daily_anchor = self.daily_loss_anchor_by_day.get(day_key)
                if daily_anchor and daily_anchor > 0:
                    current_equity = self.portfolio.portfolio_state(ts).equity
                    daily_return_pct = ((current_equity - daily_anchor) / daily_anchor) * 100
                    if daily_return_pct >= self.walk_away_profit_target_pct:
                        # Trigger walk-away!
                        self.walk_away_triggered_days.add(day_key)
                        walk_away_event = {
                            "timestamp": ts.isoformat(),
                            "day": day_key,
                            "trigger": "profit_target",
                            "daily_return_pct": round(daily_return_pct, 2),
                            "target_pct": self.walk_away_profit_target_pct,
                            "equity_at_trigger": current_equity,
                            "anchor_equity": daily_anchor,
                        }
                        self.walk_away_events.append(walk_away_event)
                        logger.info(
                            "Walk-away triggered: day=%s return=%.2f%% target=%.2f%% equity=%.2f",
                            day_key, daily_return_pct, self.walk_away_profit_target_pct, current_equity
                        )
                        # Update prices and continue without trading
                        for pair in self.pairs:
                            base_tf = self.timeframes[0]
                            df = self.market_data.get(pair, {}).get(base_tf)
                            if df is not None and ts in df.index:
                                latest_prices[pair] = float(df.loc[ts, "close"])
                        self.portfolio.mark_to_market(ts, latest_prices)
                        continue

            # Check if adaptive judge evaluation should run
            judge_triggered_replan = False
            should_run, trigger_reason = self._judge_trigger_reason(ts)
            if current_plan and should_run:
                judge_result = self._run_intraday_judge(
                    ts, run_id, day_key, latest_prices, current_plan,
                    trigger_reason=trigger_reason,
                )
                if judge_result.get("should_replan"):
                    judge_triggered_replan = True
                    # Apply judge feedback immediately
                    judge_constraints = judge_result.get("feedback", {}).get("strategist_constraints")
                    if judge_constraints:
                        self.judge_constraints = judge_constraints
                    logger.info(
                        "Judge triggered replan at %s: %s",
                        ts.isoformat(), judge_result.get("replan_reason")
                    )

            plan_expired = current_plan is not None and ts >= current_plan.valid_until

            # Track day boundary for adaptive replanning
            last_plan_day = None
            if current_plan and current_plan.generated_at:
                last_plan_day = current_plan.generated_at.strftime("%Y-%m-%d")
            is_new_day = day_key != last_plan_day

            # Track whether this is the initial plan (no existing plan)
            is_initial_plan = current_plan is None

            if self.adaptive_replanning:
                # In adaptive mode: only replan on day boundary or judge trigger
                # Do NOT replan just because valid_until expired (that's the old slot-based model)
                plan_expired = False
                new_plan_needed = is_initial_plan or is_new_day or judge_triggered_replan
            else:
                # Legacy slot-based replanning
                new_plan_needed = is_initial_plan or judge_triggered_replan or plan_expired

            if new_plan_needed:
                reason_parts = []
                if is_initial_plan:
                    reason_parts.append("initial_plan")
                if is_new_day:
                    reason_parts.append("new_day")
                if judge_triggered_replan:
                    reason_parts.append("judge_triggered")
                if plan_expired:
                    reason_parts.append("plan_expired")
                reasons = ",".join(reason_parts) if reason_parts else "unspecified"
                date_key = (run_id, day_key)
                daily_counts = getattr(self.plan_provider, "daily_counts", None)
                calls_used = daily_counts[date_key] if daily_counts is not None else None
                logger.info(
                    "Plan check ts=%s plan_id=%s new_plan_needed=%s reasons=%s adaptive=%s "
                    "calls_used=%s calls_per_day=%s last_judge=%s next_judge=%s",
                    ts.isoformat(),
                    current_plan.plan_id if current_plan else None,
                    new_plan_needed,
                    reasons,
                    self.adaptive_replanning,
                    calls_used,
                    self.calls_per_day,
                    self.last_judge_time.isoformat() if self.last_judge_time else None,
                    self.next_judge_time.isoformat() if self.next_judge_time else None,
                )

            if new_plan_needed:
                asset_states = self._asset_states(ts)
                if not asset_states:
                    continue
                active_assets = asset_states
                self.latest_factor_exposures = self._factor_exposures(ts)
                llm_input = self._llm_input(ts, asset_states, previous_plan=current_plan)
                day_start = datetime(ts.year, ts.month, ts.day, tzinfo=ts.tzinfo)
                day_end = day_start + timedelta(days=1)
                previous_plan = current_plan
                trigger_diff: Dict[str, Any] | None = None

                # In adaptive mode, plan validity is for the entire day
                # In legacy mode, use slot-based intervals
                if self.adaptive_replanning:
                    plan_start = day_start
                    plan_end = day_end
                else:
                    slot_seconds = max(1, int(self.plan_interval.total_seconds()))
                    elapsed = max(0, int((ts - day_start).total_seconds()))
                    slot_index = elapsed // slot_seconds
                    plan_start = day_start + timedelta(seconds=slot_index * slot_seconds)
                    plan_end = min(plan_start + self.plan_interval, day_end)

                # Check budget BEFORE attempting to generate
                plan_budget_exhausted = False
                date_key = (run_id, day_key)
                daily_counts = getattr(self.plan_provider, "daily_counts", None)
                if daily_counts is not None and daily_counts[date_key] >= self.calls_per_day:
                    plan_budget_exhausted = True

                plan_rebuild = True
                replan_suppressed = False
                if plan_budget_exhausted:
                    if current_plan is None:
                        raise RuntimeError(f"LLM call budget exhausted for {date_key[1]} with no existing plan")
                    # Budget exhausted - keep existing plan, extend validity to end of day
                    logger.warning(
                        "LLM call budget exhausted for %s; keeping existing plan (judge_triggered=%s).",
                        date_key[1], judge_triggered_replan,
                    )
                    self._add_event("plan_budget_exhausted", {
                        "day": date_key[1],
                        "judge_triggered": judge_triggered_replan,
                        "is_new_day": is_new_day,
                    })
                    current_plan = current_plan.model_copy(update={"valid_until": day_end})
                    plan_rebuild = False
                else:
                    current_plan = self.plan_service.generate_plan_for_run(
                        run_id,
                        llm_input,
                        plan_date=plan_start,
                        event_ts=ts,
                        prompt_template=self.prompt_template,
                        use_vector_store=self.use_trigger_vector_store,
                        emit_events=False,
                    )
                    current_plan = current_plan.model_copy(
                        update={
                            "risk_constraints": current_plan.risk_constraints.model_copy(
                                update={"max_daily_risk_budget_pct": self.active_risk_limits.max_daily_risk_budget_pct}
                            )
                        }
                    )
                    current_plan = current_plan.model_copy(
                        update={
                            "generated_at": plan_start,
                            "valid_until": plan_end,
                        }
                    )
                    current_plan = self._apply_plan_overrides(current_plan)
                    current_plan = self._apply_trigger_budget(current_plan)
                    trigger_diff = self._diff_plan_triggers(previous_plan, current_plan)
                    if judge_triggered_replan and self._is_no_change_replan(previous_plan, current_plan, trigger_diff):
                        replan_suppressed = True
                        self.no_change_replan_suppressed_by_day[day_key] += 1
                        llm_meta = getattr(current_plan, "_llm_meta", {}) or {}
                        self.llm_generation_by_day[day_key].append(llm_meta)
                        logger.info("Suppressed judge-triggered replan at %s (no material change).", ts.isoformat())
                        if previous_plan is not None:
                            run = self.run_registry.get_strategy_run(run_id)
                            run.current_plan_id = previous_plan.plan_id
                            self.run_registry.update_strategy_run(run)
                            current_plan = previous_plan
                        plan_rebuild = False
                    # Only reset judge timing on initial plan or judge-triggered replan
                    # NOT on day-boundary replans (to avoid resetting the clock each day)
                    if not replan_suppressed and (is_initial_plan or judge_triggered_replan):
                        self.last_judge_time = ts
                        self.next_judge_time = ts + self.judge_cadence
                        self.trades_since_last_judge = 0
                        logger.info(
                            "Generated new plan at %s (initial=%s, judge_triggered=%s). Next judge at %s",
                            ts.isoformat(), is_initial_plan, judge_triggered_replan, self.next_judge_time.isoformat()
                        )
                    else:
                        logger.info(
                            "Generated new plan at %s (day boundary replan). Judge timer preserved at %s",
                            ts.isoformat(), self.next_judge_time.isoformat() if self.next_judge_time else "unset"
                        )
                if plan_rebuild:
                    if not is_initial_plan:
                        self.replans_by_day[day_key] += 1
                    sizing_targets = {
                        rule.symbol: (rule.target_risk_pct or self.active_risk_limits.max_position_risk_pct)
                        for rule in current_plan.sizing_rules
                    }
                    derived_cap = getattr(current_plan, "_derived_trade_cap", None)
                    compiled_plan = compile_plan(current_plan)

                    # Validate that triggers don't reference unavailable timeframes
                    available_tfs = set(self.timeframes)
                    tf_warnings, blocked_trigger_ids = validate_plan_timeframes(current_plan, available_tfs)
                    if blocked_trigger_ids:
                        logger.warning(
                            "Plan %s has %d triggers referencing unavailable timeframes: %s. "
                            "These triggers will likely never fire. Consider adding timeframes: %s",
                            current_plan.plan_id,
                            len(blocked_trigger_ids),
                            blocked_trigger_ids,
                            sorted({tf for w in tf_warnings for tf in w.missing_timeframes}),
                        )

                    current_plan_payload = current_plan.model_dump()
                    current_compiled_payload = compiled_plan.model_dump()
                    risk_engine = RiskEngine(
                        current_plan.risk_constraints,
                        {rule.symbol: rule for rule in current_plan.sizing_rules},
                        daily_anchor_equity=self.daily_loss_anchor_by_day.get(day_key),
                        risk_profile=self.risk_profile,
                    )
                    self.latest_risk_engine = risk_engine
                    # Collect samples from previous trigger engine before creating new one
                    if trigger_engine is not None and self.debug_trigger_sample_rate > 0:
                        samples = trigger_engine.get_evaluation_samples_summary()
                        self._append_trigger_evaluation_samples(samples)
                    trigger_engine = TriggerEngine(
                        current_plan,
                        risk_engine,
                        trade_risk=TradeRiskEvaluator(risk_engine),
                        confidence_override_threshold=self.confidence_override_threshold,
                        min_hold_bars=0,
                        debug_sample_rate=self.debug_trigger_sample_rate,
                        debug_max_samples=self.debug_trigger_max_samples,
                    )
                self.current_trigger_engine = trigger_engine
                llm_meta = getattr(current_plan, "_llm_meta", {}) or {}
                self.llm_generation_by_day[day_key].append(llm_meta)
                trigger_summary = [
                    {
                        "id": trig.id,
                        "symbol": trig.symbol,
                        "direction": trig.direction,
                        "timeframe": trig.timeframe,
                        "category": trig.category,
                        "confidence": trig.confidence_grade,
                        "entry_rule": trig.entry_rule,
                        "exit_rule": trig.exit_rule,
                    }
                    for trig in current_plan.triggers[:50]
                ]
                plan_log.append(
                    {
                        "plan_id": current_plan.plan_id,
                        "previous_plan_id": trigger_diff.get("previous_plan_id") if trigger_diff else None,
                        "generated_at": current_plan.generated_at.isoformat(),
                        "valid_until": current_plan.valid_until.isoformat(),
                        "regime": current_plan.regime,
                        "market_regime": current_plan.regime,
                        "timestamp": current_plan.generated_at.isoformat(),
                        "num_triggers": len(current_plan.triggers),
                        "triggers": trigger_summary,
                        "trigger_diff": trigger_diff,
                        "max_trades_per_day": current_plan.max_trades_per_day,
                        "derived_max_trades_per_day": derived_cap or current_plan.max_trades_per_day,
                        "min_trades_per_day": current_plan.min_trades_per_day,
                        "max_triggers_per_symbol_per_day": current_plan.max_triggers_per_symbol_per_day,
                        "trigger_budgets": current_plan.trigger_budgets,
                        "allowed_symbols": current_plan.allowed_symbols,
                        "allowed_directions": current_plan.allowed_directions,
                    }
                )
                # Emit plan_generated event for UI timeline
                self._emit_plan_generated_event(run_id, current_plan, trigger_summary, trigger_diff=trigger_diff, event_ts=ts)
                policy_max_trades = getattr(current_plan, "_policy_max_trades_per_day", current_plan.max_trades_per_day)
                policy_max_triggers = getattr(
                    current_plan, "_policy_max_triggers_per_symbol_per_day", current_plan.max_triggers_per_symbol_per_day
                )
                if policy_max_trades is None:
                    policy_max_trades = current_plan.max_trades_per_day
                if policy_max_triggers is None:
                    policy_max_triggers = current_plan.max_triggers_per_symbol_per_day
                derived_trade_cap = getattr(current_plan, "_derived_trade_cap", None)
                derived_trigger_cap = getattr(current_plan, "_derived_trigger_cap", None)
                resolved_max_trades = getattr(current_plan, "_resolved_trade_cap", current_plan.max_trades_per_day)
                resolved_max_triggers = getattr(
                    current_plan, "_resolved_trigger_cap", current_plan.max_triggers_per_symbol_per_day
                )
                cap_inputs = getattr(current_plan, "_cap_inputs", None) or {}
                session_caps: Dict[str, int] = {}
                if self.session_trade_multipliers:
                    base = resolved_max_trades or policy_max_trades or derived_trade_cap
                    for window in self.session_trade_multipliers:
                        try:
                            start = int(window.get("start_hour", -1))
                            end = int(window.get("end_hour", -1))
                            mult = float(window.get("multiplier", 1.0))
                        except (TypeError, ValueError):
                            continue
                        if start < 0 or end < 0 or start >= end:
                            continue
                        if base:
                            session_caps[f"{start}-{end}"] = int(base * mult)
                self.plan_limits_by_day[day_key] = {
                    "plan_id": current_plan.plan_id,
                    "max_trades_per_day": policy_max_trades,
                    "max_triggers_per_symbol_per_day": policy_max_triggers,
                    "derived_max_trades_per_day": derived_trade_cap,
                    "derived_max_triggers_per_symbol_per_day": derived_trigger_cap,
                    "resolved_max_trades_per_day": resolved_max_trades,
                    "resolved_max_triggers_per_symbol_per_day": resolved_max_triggers,
                    "min_trades_per_day": current_plan.min_trades_per_day,
                    "allowed_symbols": current_plan.allowed_symbols,
                    "trigger_budgets": dict(current_plan.trigger_budgets or {}),
                    "trigger_budget_trimmed": dict(self.latest_trigger_trim),
                    "session_trade_multipliers": self.session_trade_multipliers,
                    "session_caps": session_caps,
                    "timeframe_trigger_caps": self.timeframe_trigger_caps,
                    "flatten_policy": self.flatten_policy,
                    "cap_inputs": cap_inputs,
                    "strict_fixed_caps": self.strict_fixed_caps,
                    "trigger_catalog": {
                        trigger.id: {"symbol": trigger.symbol, "category": trigger.category, "direction": trigger.direction}
                        for trigger in current_plan.triggers
                        if trigger.id
                    },
                }
                slot_start = current_plan.generated_at if current_plan.generated_at else plan_start
                slot_end = current_plan.valid_until if current_plan.valid_until else plan_end
                logger.info(
                    "Generated plan plan_id=%s slot_start=%s slot_end=%s triggers=%s",
                    current_plan.plan_id,
                    slot_start.isoformat(),
                    slot_end.isoformat(),
                    len(current_plan.triggers),
                )

                # Track plan generation event for progress reporting
                self._add_event("plan_generated", {
                    "plan_id": current_plan.plan_id,
                    "triggers": len(current_plan.triggers),
                    "slot_start": slot_start.isoformat(),
                    "slot_end": slot_end.isoformat(),
                })

            if trigger_engine is None or current_plan is None:
                continue

            slot_orders: List[Dict[str, Any]] = []
            market_structure_briefs: Dict[str, Any] = {}
            indicator_briefs = self._indicator_briefs(active_assets)
            indicator_debug: Dict[str, Dict[str, Any]] | None = None
            if self.indicator_debug_mode:
                indicator_debug = {}
            # Update latest prices FIRST so portfolio valuation reflects current bar
            for pair in self.pairs:
                base_tf = self.timeframes[0]
                df = self.market_data.get(pair, {}).get(base_tf)
                if df is not None and ts in df.index:
                    latest_prices[pair] = float(df.loc[ts, "close"])

            # Mark-to-market BEFORE sizing decisions so equity reflects current prices
            self.portfolio.mark_to_market(ts, latest_prices)

            for pair in self.pairs:
                for timeframe in self.timeframes:
                    df = self.market_data.get(pair, {}).get(timeframe)
                    if df is None or ts not in df.index:
                        continue
                    bar = self._build_bar(pair, timeframe, ts)
                    indicator = self._indicator_snapshot(pair, timeframe, ts)
                    if indicator is None:
                        continue
                    if indicator_debug is not None:
                        symbol_debug = indicator_debug.setdefault(pair, {})
                        symbol_debug[timeframe] = self._format_indicator_debug(indicator)
                    portfolio_state = self.portfolio.portfolio_state(ts)
                    asset_state = active_assets.get(pair)
                    logger.debug(
                        "Evaluating triggers ts=%s pair=%s timeframe=%s equity=%.2f cash=%.2f",
                        ts.isoformat(),
                        pair,
                        timeframe,
                        portfolio_state.equity,
                        portfolio_state.cash,
                    )
                    structure_snapshot = market_structure_briefs.get(pair)
                    orders, blocked_entries = trigger_engine.on_bar(bar, indicator, portfolio_state, asset_state, market_structure=structure_snapshot)
                    current_load = len(orders) + (len(blocked_entries) if blocked_entries else 0)
                    fills_before = len(self.portfolio.fills)
                    executed_records = self._process_orders_with_limits(
                        run_id,
                        day_key,
                        orders,
                        portfolio_state,
                        current_plan_payload,
                        current_compiled_payload,
                        blocked_entries,
                        current_load=current_load,
                        market_structure=market_structure_briefs,
                    )
                    slot_orders.extend(executed_records)
                    # Track trades for adaptive judge
                    fills_after = len(self.portfolio.fills)
                    if fills_after > fills_before:
                        for _ in range(fills_after - fills_before):
                            self._on_trade_executed(ts)
                    if timeframe == self.timeframes[0]:
                        latest_prices[pair] = bar.close
                        try:
                            subset = df[df.index <= ts]
                            structure_snapshot = build_market_structure_snapshot(
                                subset,
                                symbol=pair,
                                timeframe=timeframe,
                                lookback=self.window_configs[timeframe].medium_window * 3,
                                swing_window=2,
                                tolerance_mult=0.75,
                            )
                            if structure_snapshot:
                                market_structure_briefs[pair] = structure_snapshot.to_dict()
                        except Exception as exc:  # pragma: no cover - defensive only
                            logger.debug("Market structure snapshot failed for %s %s: %s", pair, timeframe, exc)
            if (
                self.flatten_session_boundary_hour is not None
                and ts.hour == self.flatten_session_boundary_hour
                and day_key not in session_flattened
            ):
                self._flatten_end_of_day(day_key, ts, latest_prices)
                session_flattened.add(day_key)
            self.portfolio.mark_to_market(ts, latest_prices)
            portfolio_state = self.portfolio.portfolio_state(ts)
            slot_report = {
                "timestamp": ts.isoformat(),
                "equity": portfolio_state.equity,
                "cash": portfolio_state.cash,
                "positions": dict(self.portfolio.positions),
                "orders": slot_orders,
                "indicator_context": indicator_briefs,
                "market_structure": market_structure_briefs,
                "factor_exposures": self.latest_factor_exposures,
            }
            if indicator_debug is not None:
                slot_report["indicator_debug"] = indicator_debug
            self.slot_reports_by_day[day_key].append(slot_report)
            self.last_slot_report = slot_report

        if self.current_day_key:
            if self.flatten_positions_daily and self.last_slot_report:
                flatten_ts = datetime.fromisoformat(self.last_slot_report["timestamp"])
                self._flatten_end_of_day(self.current_day_key, flatten_ts, latest_prices)
            summary = self._finalize_day(self.current_day_key, daily_dir, run_id)
            if summary:
                self.latest_daily_summary = summary
                self.memory_history.append(summary)
                judge_constraints = summary.get("judge_feedback", {}).get("strategist_constraints")
                if judge_constraints:
                    self.judge_constraints = judge_constraints
                daily_reports.append(summary)

        # Collect samples from final trigger engine
        if trigger_engine is not None and self.debug_trigger_sample_rate > 0:
            samples = trigger_engine.get_evaluation_samples_summary()
            self._append_trigger_evaluation_samples(samples)

        equity_df = pd.DataFrame(self.portfolio.equity_records)
        equity_curve = (
            equity_df.set_index("timestamp")["equity"]
            if not equity_df.empty
            else pd.Series([self.initial_cash], index=[self.start])
        )
        fills_df = pd.DataFrame(self.portfolio.fills)
        final_equity = float(equity_curve.iloc[-1])
        equity_return_pct = (final_equity / self.initial_cash - 1) * 100 if self.initial_cash else 0.0
        gross_trade_pnl = sum(float(entry.get("pnl", 0.0)) for entry in self.portfolio.trade_log)
        gross_trade_return_pct = (gross_trade_pnl / self.initial_cash * 100.0) if self.initial_cash else 0.0
        run_summary_path = cache_base / "run_summary.json"
        baseline_summary_path = None
        env_baseline_path = os.environ.get("BASELINE_SUMMARY_PATH")
        if env_baseline_path:
            baseline_summary_path = Path(env_baseline_path)
        else:
            baseline_run_id = os.environ.get("BASELINE_RUN_ID")
            if baseline_run_id:
                baseline_candidate = Path(self.plan_provider.cache_dir) / baseline_run_id / "run_summary.json"
                baseline_summary_path = baseline_candidate
            else:
                baseline_candidate = Path(self.plan_provider.cache_dir) / "baseline" / "run_summary.json"
                if baseline_candidate.exists():
                    baseline_summary_path = baseline_candidate
        run_summary = write_run_summary(daily_reports, run_summary_path, baseline_summary_path=baseline_summary_path)
        summary = {
            "final_equity": final_equity,
            "equity_return_pct": equity_return_pct,
            "gross_trade_return_pct": gross_trade_return_pct,
            "return_pct": equity_return_pct,
            "run_summary": run_summary,
            # Walk-away tracking
            "walk_away_enabled": self.walk_away_enabled,
            "walk_away_events": self.walk_away_events,
            "walk_away_days_triggered": list(self.walk_away_triggered_days),
            # Debug trigger evaluation samples (if enabled)
            "trigger_evaluation_samples": self._all_trigger_evaluation_samples if self.debug_trigger_sample_rate > 0 else None,
            "trigger_evaluation_sample_count": len(self._all_trigger_evaluation_samples) if self.debug_trigger_sample_rate > 0 else 0,
        }
        return StrategistBacktestResult(
            equity_curve=equity_curve,
            fills=fills_df,
            plan_log=plan_log,
            summary=summary,
            llm_costs=self.plan_provider.cost_tracker.snapshot(),
            final_cash=self.portfolio.cash,
            final_positions=dict(self.portfolio.positions),
            daily_reports=daily_reports,
            bar_decisions=dict(self.bar_decisions_by_day),
            intraday_judge_history=list(self.intraday_judge_history),
            judge_triggered_replans=list(self.judge_triggered_replans),
        )

    def _indicator_briefs(self, asset_states: Dict[str, AssetState]) -> Dict[str, Dict[str, Any]]:
        briefs: Dict[str, Dict[str, Any]] = {}
        for symbol, asset in asset_states.items():
            snapshot = asset.indicators[0] if asset.indicators else None
            if not snapshot:
                continue
            relation = None
            if snapshot.sma_medium and snapshot.close:
                relation = "above_sma_medium" if snapshot.close > snapshot.sma_medium else "below_sma_medium"
            briefs[symbol] = {
                "close": snapshot.close,
                "rsi_14": snapshot.rsi_14,
                "trend_state": asset.trend_state,
                "vol_state": asset.vol_state,
                "sma_relation": relation,
            }
        return briefs

    def _format_indicator_debug(self, snapshot: IndicatorSnapshot) -> Dict[str, Any]:
        payload = snapshot.model_dump()
        if self.indicator_debug_mode == "full":
            return payload
        if self.indicator_debug_mode == "keys":
            if not self.indicator_debug_keys:
                return {}
            return {key: payload.get(key) for key in self.indicator_debug_keys}
        return {}

    def _apply_trigger_budget(self, plan: StrategyPlan) -> StrategyPlan:
        resolved_cap = getattr(plan, "_resolved_trigger_cap", None)
        default_cap = resolved_cap or plan.max_triggers_per_symbol_per_day or self.default_symbol_trigger_cap
        trimmed_plan, stats = enforce_trigger_budget(
            plan, default_cap=default_cap, fallback_symbols=self.pairs, resolved_cap=resolved_cap
        )
        trimmed = sum(stats.values())
        meaningful = {symbol: count for symbol, count in stats.items() if count > 0}
        self.latest_trigger_trim = meaningful
        if trimmed > 0:
            logger.info("Trimmed %s triggers via budget controls: %s", trimmed, meaningful)
        return trimmed_plan

    def _apply_plan_overrides(self, plan: StrategyPlan) -> StrategyPlan:
        updates: Dict[str, Any] = {}
        if self.plan_max_trades_per_day is not None:
            updates["max_trades_per_day"] = self.plan_max_trades_per_day
        if self.plan_max_triggers_per_symbol_per_day is not None:
            updates["max_triggers_per_symbol_per_day"] = self.plan_max_triggers_per_symbol_per_day
        if not updates:
            return plan
        updated = plan.model_copy(update=updates)
        if "max_trades_per_day" in updates:
            object.__setattr__(updated, "_policy_max_trades_per_day", updates["max_trades_per_day"])
            object.__setattr__(updated, "_resolved_trade_cap", updates["max_trades_per_day"])
        if "max_triggers_per_symbol_per_day" in updates:
            object.__setattr__(updated, "_policy_max_triggers_per_symbol_per_day", updates["max_triggers_per_symbol_per_day"])
            object.__setattr__(updated, "_resolved_trigger_cap", updates["max_triggers_per_symbol_per_day"])
        return updated

    def _apply_risk_limits_to_plan(self, plan: StrategyPlan) -> StrategyPlan:
        """Inject active risk limits (including daily risk budget) and derive trade caps."""

        updated_constraints = plan.risk_constraints.model_copy(
            update={"max_daily_risk_budget_pct": self.active_risk_limits.max_daily_risk_budget_pct}
        )
        plan = plan.model_copy(update={"risk_constraints": updated_constraints})
        plan = self.plan_provider._enrich_plan(plan, llm_input=self._llm_input(plan.generated_at, self._asset_states(plan.generated_at)))  # type: ignore[arg-type]
        return plan

    def _flatten_end_of_day(self, day_key: str, timestamp: datetime, price_map: Mapping[str, float]) -> None:
        if not self.flatten_positions_daily or day_key in self.flattened_days:
            return
        orders: List[Order] = []
        for symbol, qty in list(self.portfolio.positions.items()):
            if abs(qty) <= 1e-9:
                continue
            price = price_map.get(symbol)
            if price is None:
                continue
            notional = abs(qty) * price
            if self.flatten_notional_threshold and notional < self.flatten_notional_threshold:
                continue
            side: Literal["buy", "sell"] = "sell" if qty > 0 else "buy"
            order = Order(
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                price=price,
                timeframe=self.timeframes[0],
                reason="eod_flatten",
                timestamp=timestamp,
            )
            self.portfolio.execute(order)
            self.last_flat_time_by_symbol[symbol] = timestamp
            self.last_flat_trigger_by_symbol[symbol] = "eod_flatten"
            orders.append(order)
        if not orders:
            self.flattened_days.add(day_key)
            return
        self.portfolio.mark_to_market(timestamp, price_map)
        slot_report = {
            "timestamp": timestamp.isoformat(),
            "equity": self.portfolio.portfolio_state(timestamp).equity,
            "cash": self.portfolio.cash,
            "positions": dict(self.portfolio.positions),
            "orders": [
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "timeframe": order.timeframe,
                    "reason": order.reason,
                }
                for order in orders
            ],
            "indicator_context": self.last_slot_report.get("indicator_context", {}) if self.last_slot_report else {},
        }
        self.slot_reports_by_day[day_key].append(slot_report)
        self.last_slot_report = slot_report
        self.flattened_days.add(day_key)

    def _set_daily_loss_anchor(self, day_key: str, equity: float) -> None:
        """Set daily loss anchor once per day; ignore intra-day refresh attempts."""
        if day_key in self.daily_loss_anchor_by_day:
            return
        self.daily_loss_anchor_by_day[day_key] = equity

    def _reset_risk_budget_for_day(self, day_key: str, start_equity: float | None = None) -> None:
        pct = self.daily_risk_budget_pct or 0.0
        if pct <= 0:
            self.daily_risk_budget_state.pop(day_key, None)
            return
        equity = float(start_equity) if start_equity is not None else (
            float(self.portfolio.equity_records[-1]["equity"]) if self.portfolio.equity_records else float(self.initial_cash)
        )
        budget_abs = equity * (pct / 100.0)
        self.daily_risk_budget_state[day_key] = {
            "budget_pct": pct,
            "start_equity": equity,
            "budget_abs": budget_abs,
            "used_abs": 0.0,
            "symbol_usage": defaultdict(float),
            "blocks": defaultdict(int),
            "slippage_adjustment": 0.0,
        }

    def _risk_budget_allowance(self, day_key: str, order: Order) -> float | None:
        pct = self.daily_risk_budget_pct or 0.0
        if pct <= 0:
            return 0.0
        entry = self.daily_risk_budget_state.get(day_key)
        if not entry:
            return 0.0
        budget = entry.get("budget_abs", 0.0)
        if budget <= 0:
            return None
        notional = max(order.quantity * order.price, 0.0)
        if notional <= 0:
            return 0.0
        target_risk_pct = None
        if hasattr(self, "sizing_targets"):
            target_risk_pct = self.sizing_targets.get(order.symbol)
        if target_risk_pct is None:
            target_risk_pct = self.active_risk_limits.max_position_risk_pct or 0.0
        # Adaptive boost: based on same-day usage only (no cross-day carry-over).
        prev_usage = (entry.get("used_abs", 0.0) / budget * 100.0) if budget > 0 else 100.0
        adaptive_multiplier = 3.0 if prev_usage < 10.0 else 1.0
        # Per-trade risk allowance expressed in currency, not double-scaled by notional.
        base_equity = entry.get("start_equity", self.initial_cash)
        per_trade_cap = max(target_risk_pct * adaptive_multiplier, 0.0) / 100.0
        contribution = base_equity * per_trade_cap
        if contribution <= 0:
            return 0.0
        used = entry.get("used_abs", 0.0)
        remaining = budget - used
        if remaining <= 0:
            return None
        contribution = min(contribution, remaining)
        return contribution

    def _commit_risk_budget(self, day_key: str, contribution: float, symbol: str | None) -> None:
        if contribution <= 0:
            return
        entry = self.daily_risk_budget_state.get(day_key)
        if not entry:
            return
        entry["used_abs"] = entry.get("used_abs", 0.0) + contribution
        if symbol:
            entry["symbol_usage"][symbol] += contribution

    def _record_risk_budget_block(self, day_key: str, symbol: str | None) -> None:
        entry = self.daily_risk_budget_state.get(day_key)
        if not entry or not symbol:
            return
        entry["blocks"][symbol] += 1

    def _risk_budget_summary(self, day_key: str) -> Dict[str, float] | None:
        entry = self.daily_risk_budget_state.pop(day_key, None)
        if not entry:
            return None
        budget = entry.get("budget_abs", 0.0)
        used = entry.get("used_abs", 0.0)
        used_pct = (used / budget * 100.0) if budget > 0 else 0.0
        start_equity = entry.get("start_equity", self.initial_cash)
        budget_pct = entry.get("budget_pct", 0.0)
        util_pct = 0.0
        if start_equity and budget_pct:
            util_pct = (used / (start_equity * (budget_pct / 100.0))) * 100.0
        symbol_usage = entry.get("symbol_usage", {})
        symbol_usage_pct = {symbol: (value / budget * 100.0) if budget > 0 else 0.0 for symbol, value in symbol_usage.items()}
        blocks_by_symbol = dict(entry.get("blocks", {}))
        return {
            "budget_pct": budget_pct,
            "budget_abs": budget,
            "used_abs": used,
            "used_pct": used_pct,
            "utilization_pct": util_pct,
            "start_equity": start_equity,
            "budget_base_equity": start_equity,
            "budget_allocated_abs": budget,
            "budget_allocated_pct": budget_pct,
            "budget_actual_abs": used,
            "budget_slippage_adjustment": entry.get("slippage_adjustment", 0.0),
            "symbol_usage_pct": symbol_usage_pct,
            "blocks_by_symbol": blocks_by_symbol,
        }
    def _risk_budget_fallback(
        self,
        start_equity: float,
        risk_usage_events: List[Dict[str, Any]],
        budget_pct: float | None = None,
    ) -> Dict[str, float] | None:
        """Derive risk budget stats from risk_usage_events when budget state is missing."""

        pct = budget_pct if budget_pct is not None else (self.active_risk_limits.max_daily_risk_budget_pct or 0.0)
        if pct <= 0:
            return None
        budget_abs = start_equity * (pct / 100.0)
        used_abs = sum(float(evt.get("risk_used", 0.0)) for evt in risk_usage_events or [])
        used_pct = (used_abs / budget_abs * 100.0) if budget_abs > 0 else 0.0
        return {
            "budget_pct": pct,
            "budget_abs": budget_abs,
            "used_abs": used_abs,
            "used_pct": used_pct,
            "utilization_pct": used_pct,
            "start_equity": start_equity,
            "budget_base_equity": start_equity,
            "budget_allocated_abs": budget_abs,
            "budget_allocated_pct": pct,
            "budget_actual_abs": used_abs,
            "budget_slippage_adjustment": 0.0,
            "symbol_usage_pct": {},
            "blocks_by_symbol": {},
        }

    @staticmethod
    def _timestamp_day(value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                return None
        return None

    def _daily_trade_pnl(self, day_key: str) -> float:
        """Aggregate realized trade PnL for a calendar day."""

        pnl = 0.0
        for entry in self.portfolio.trade_log:
            if self._timestamp_day(entry.get("timestamp")) == day_key:
                pnl += float(entry.get("pnl", 0.0))
        return pnl

    def _daily_costs(
        self,
        day_key: str,
        start_equity: float,
        end_equity: float,
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
        """Compute PnL components for a given day using start/end equity anchors."""

        realized_total = 0.0
        flatten_pnl = 0.0
        symbol_pnl: Dict[str, float] = defaultdict(float)
        symbol_flatten: Dict[str, float] = defaultdict(float)
        for entry in self.portfolio.trade_log:
            if self._timestamp_day(entry.get("timestamp")) != day_key:
                continue
            pnl = float(entry.get("pnl", 0.0))
            symbol = entry.get("symbol")
            realized_total += pnl
            if symbol:
                symbol_pnl[symbol] += pnl
            reason = entry.get("reason") or ""
            if reason == "eod_flatten":
                flatten_pnl += pnl
                if symbol:
                    symbol_flatten[symbol] += pnl
        fees = 0.0
        symbol_fees: Dict[str, float] = defaultdict(float)
        for fill in self.portfolio.fills:
            if self._timestamp_day(fill.get("timestamp")) != day_key:
                continue
            fee_val = float(fill.get("fee", 0.0))
            symbol = fill.get("symbol")
            fees += fee_val
            if symbol:
                symbol_fees[symbol] += fee_val

        non_flatten_pnl = realized_total - flatten_pnl
        denom = start_equity or 1.0
        component_net = (non_flatten_pnl + flatten_pnl - fees) / denom * 100.0
        equity_return_pct = ((end_equity / start_equity - 1) * 100.0) if start_equity else 0.0
        carryover_pnl = (end_equity - start_equity) - (non_flatten_pnl + flatten_pnl - fees)
        carryover_pct = (carryover_pnl / denom) * 100.0
        breakdown = {
            "gross_trade_pct": (non_flatten_pnl / denom) * 100.0,
            "flattening_pct": (flatten_pnl / denom) * 100.0,
            "fees_pct": (-fees / denom) * 100.0,
            "carryover_pct": carryover_pct,
            "component_net_pct": component_net + carryover_pct,
            "net_equity_pct": equity_return_pct,
            "net_equity_pct_delta": equity_return_pct - (component_net + carryover_pct),
        }
        symbol_breakdown: Dict[str, Dict[str, float]] = {}
        symbols = set(symbol_pnl.keys()) | set(symbol_fees.keys()) | set(symbol_flatten.keys())
        for symbol in symbols:
            gross_symbol = symbol_pnl.get(symbol, 0.0) - symbol_flatten.get(symbol, 0.0)
            flatten_symbol = symbol_flatten.get(symbol, 0.0)
            fee_symbol = symbol_fees.get(symbol, 0.0)
            symbol_breakdown[symbol] = {
                "gross_pct": (gross_symbol / denom) * 100.0,
                "flattening_pct": (flatten_symbol / denom) * 100.0,
                "fees_pct": (-fee_symbol / denom) * 100.0,
                "net_pct": ((gross_symbol + flatten_symbol - fee_symbol) / denom) * 100.0,
            }
        component_totals = {
            "realized_pnl_abs": non_flatten_pnl,
            "flattening_pnl_abs": flatten_pnl,
            "fees_abs": fees,
            "carryover_pnl_abs": carryover_pnl,
            "equity_return_abs": end_equity - start_equity,
        }
        return breakdown, symbol_breakdown, component_totals

    def _report_progress(self) -> None:
        """Report current progress via the callback."""
        if not self.progress_callback:
            return

        progress_pct = (self.candles_processed / self.candles_total * 100) if self.candles_total > 0 else 0

        self.progress_callback({
            "candles_processed": self.candles_processed,
            "candles_total": self.candles_total,
            "progress_pct": progress_pct,
            "current_timestamp": self.current_timestamp.isoformat() if self.current_timestamp else None,
            "current_day": self.current_day_key,
            "recent_events": self.recent_events[-10:],  # Last 10 events
        })

    def _add_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add an event to the recent events list for progress reporting."""
        event = {
            "type": event_type,
            "timestamp": self.current_timestamp.isoformat() if self.current_timestamp else None,
            **details,
        }
        self.recent_events.append(event)
        # Keep only last 100 events to avoid memory bloat
        if len(self.recent_events) > 100:
            self.recent_events = self.recent_events[-100:]

        # Report progress immediately on significant events
        if self.progress_callback and event_type in ("trade", "plan_generated", "trigger_blocked", "intraday_judge"):
            self._report_progress()

    def _append_trigger_evaluation_samples(self, samples: List[Dict[str, Any]]) -> None:
        if not samples:
            return
        if not self.debug_trigger_max_samples or self.debug_trigger_max_samples <= 0:
            self._all_trigger_evaluation_samples.extend(samples)
            return
        remaining = self.debug_trigger_max_samples - len(self._all_trigger_evaluation_samples)
        if remaining <= 0:
            return
        self._all_trigger_evaluation_samples.extend(samples[:remaining])

    def _finalize_day(self, day_key: str, daily_dir: Path, run_id: str) -> Dict[str, Any] | None:
        reports = self.slot_reports_by_day.pop(day_key, [])
        self.bar_decisions_by_day[day_key] = list(reports)
        if not reports:
            self.daily_risk_budget_state.pop(day_key, None)
            self.risk_usage_by_day.pop(day_key, None)
            self.risk_usage_events_by_day.pop(day_key, None)
            self.latency_by_day.pop(day_key, None)
            self.timeframe_exec_counts.pop(day_key, None)
            self.session_trade_counts.pop(day_key, None)
            self.trigger_load_by_day.pop(day_key, None)
            return None
        start_equity = reports[0]["equity"]
        end_equity = reports[-1]["equity"]
        equity_return_pct = ((end_equity / start_equity - 1) * 100.0) if start_equity else 0.0
        gross_trade_pnl = self._daily_trade_pnl(day_key)
        gross_trade_return_pct = (gross_trade_pnl / start_equity * 100.0) if start_equity else 0.0
        pnl_breakdown, symbol_pnl, pnl_totals = self._daily_costs(day_key, start_equity if start_equity else 1.0, end_equity)
        trade_count = sum(len(report.get("orders", [])) for report in reports)
        indicator_context = reports[-1].get("indicator_context", {})
        market_structure_context = reports[-1].get("market_structure", {})
        # TODO: push market_structure_context into strategy_export and per-trade logs for LLM consumption.
        trigger_activity = self.trigger_activity_by_day.pop(day_key, {})
        top_triggers = sorted(
            ((trigger_id, data.get("executed", 0)) for trigger_id, data in trigger_activity.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        limit_entry = self.limit_enforcement_by_day.pop(day_key, _new_limit_entry())
        skipped_counts = dict(limit_entry["skipped"])
        skipped_limits = self.skipped_activity_by_day.pop(day_key, skipped_counts)
        risk_breakdown = dict(limit_entry["risk_block_breakdown"])
        risk_limit_hints = dict(limit_entry["risk_limit_hints"])
        plan_limits = self.plan_limits_by_day.pop(day_key, None)
        risk_usage = self.risk_usage_by_day.pop(day_key, {})
        risk_usage_events = self.risk_usage_events_by_day.pop(day_key, [])
        latency_samples = self.latency_by_day.pop(day_key, {})
        self.timeframe_exec_counts.pop(day_key, None)
        self.session_trade_counts.pop(day_key, None)
        self.archetype_load_by_day.pop(day_key, None)
        trigger_load_events = self.trigger_load_by_day.pop(day_key, [])
        llm_meta_entries = self.llm_generation_by_day.pop(day_key, [])
        llm_failed_parse = any(entry.get("llm_failed_parse") for entry in llm_meta_entries)
        fallback_used = any(entry.get("fallback_plan_used") for entry in llm_meta_entries)
        llm_failure_reason = None
        for entry in reversed(llm_meta_entries):
            if entry.get("llm_failure_reason"):
                llm_failure_reason = entry.get("llm_failure_reason")
                break
        llm_data_quality = "fallback" if (llm_failed_parse or fallback_used) else "ok"
        replans_count = self.replans_by_day.pop(day_key, 0)
        suppressed_replans = self.no_change_replan_suppressed_by_day.pop(day_key, 0)
        allocated_risk_abs = 0.0
        actual_risk_abs = 0.0
        for evt in risk_usage_events:
            allocated_risk_abs += float(evt.get("risk_used", 0.0))
            if evt.get("actual_risk_at_stop") is not None:
                actual_risk_abs += float(evt.get("actual_risk_at_stop", 0.0))
        # Explicit block counters for observability.
        daily_loss_blocks = int(risk_breakdown.get(BlockReason.RISK.value, 0) + risk_breakdown.get("max_daily_loss_pct", 0))
        daily_cap_blocks = int(risk_breakdown.get("daily_cap", 0))
        risk_budget_blocks = int(risk_breakdown.get(BlockReason.RISK_BUDGET.value, 0))
        session_cap_blocks = int(risk_breakdown.get("session_cap", 0))
        archetype_load_blocks = int(risk_breakdown.get("archetype_load", 0))
        trigger_load_blocks = int(risk_breakdown.get("trigger_load", 0))
        symbol_cap_blocks = int(risk_breakdown.get("max_symbol_exposure_pct", 0))
        intraday_judge_entries = [
            entry for entry in self.intraday_judge_history
            if entry.get("timestamp", "").startswith(day_key)
        ]
        summary = {
            "date": day_key,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "return_pct": equity_return_pct,
            "equity_return_pct": equity_return_pct,
            "gross_trade_return_pct": gross_trade_return_pct,
            "realized_pnl_abs": pnl_totals["realized_pnl_abs"],
            "flattening_pnl_abs": pnl_totals["flattening_pnl_abs"],
            "fees_abs": pnl_totals["fees_abs"],
            "carryover_pnl": pnl_totals["carryover_pnl_abs"],
            "daily_cash_flows": 0.0,
            "trade_count": trade_count,
            "positions_end": reports[-1]["positions"],
            "indicator_context": indicator_context,
            "market_structure": market_structure_context,
            "factor_exposures": self.latest_factor_exposures,
            "intraday_judge_history": intraday_judge_entries,
            "top_triggers": top_triggers,
            "skipped_due_to_limits": skipped_limits,
            "attempted_triggers": limit_entry["trades_attempted"],
            "executed_trades": limit_entry["trades_executed"],
            "flatten_positions_daily": self.flatten_positions_daily,
            "flatten_session_hour": self.flatten_session_boundary_hour,
            "flatten_policy": self.flatten_policy,
            "pnl_breakdown": pnl_breakdown,
            "symbol_pnl": symbol_pnl,
            "allocated_risk_abs": allocated_risk_abs,
            "actual_risk_abs": actual_risk_abs,
            "risk_usage": {f"{k[0]}|{k[1]}": v for k, v in (risk_usage or {}).items()},
            "risk_usage_events": risk_usage_events,
            "daily_loss_blocks": daily_loss_blocks,
            "daily_cap_blocks": daily_cap_blocks,
            "risk_budget_blocks": risk_budget_blocks,
            "session_cap_blocks": session_cap_blocks,
            "archetype_load_blocks": archetype_load_blocks,
            "trigger_load_blocks": trigger_load_blocks,
            "symbol_cap_blocks": symbol_cap_blocks,
            "risk_profile": {
                "global_multiplier": getattr(self.risk_profile, "global_multiplier", 1.0),
                "symbol_multipliers": dict(getattr(self.risk_profile, "symbol_multipliers", {}) or {}),
                "archetype_multipliers": dict(getattr(self.risk_profile, "archetype_multipliers", {}) or {}),
                "archetype_hour_multipliers": dict(getattr(self.risk_profile, "archetype_hour_multipliers", {}) or {}),
            },
            "risk_budget_pct": self.daily_risk_budget_pct,
            "llm_failed_parse": llm_failed_parse,
            "fallback_plan_used": fallback_used,
            "llm_failure_reason": llm_failure_reason,
            "llm_data_quality": llm_data_quality,
            "replan_rate_per_day": replans_count,
            "no_change_replan_suppressed_count": suppressed_replans,
        }
        blocked_daily_cap = skipped_counts.get(BlockReason.DAILY_CAP.value, 0)
        blocked_symbol_veto = skipped_counts.get(BlockReason.SYMBOL_VETO.value, 0)
        blocked_direction = skipped_counts.get(BlockReason.DIRECTION.value, 0)
        blocked_category = skipped_counts.get(BlockReason.CATEGORY.value, 0)
        blocked_expression = skipped_counts.get(BlockReason.EXPRESSION_ERROR.value, 0)
        blocked_missing_indicator = skipped_counts.get(BlockReason.MISSING_INDICATOR.value, 0)
        blocked_plan = skipped_counts.get(BlockReason.PLAN_LIMIT.value, 0)
        blocked_other = skipped_counts.get(BlockReason.OTHER.value, 0)
        blocked_risk_budget = skipped_counts.get("risk_budget", 0)
        blocked_risk = sum(risk_breakdown.values())
        blocked_trigger_load = skipped_counts.get("trigger_load", 0)
        blocked_archetype_load = skipped_counts.get("archetype_load", 0)
        blocked_min_hold = skipped_counts.get("min_hold", 0)
        blocked_min_flat = skipped_counts.get("min_flat", 0)
        attempted = summary.get("attempted_triggers", 0)
        executed = summary.get("executed_trades", 0)
        execution_rate = (executed / attempted) if attempted else 0.0
        daily_cap_on = blocked_daily_cap > 0
        plan_cap_on = blocked_plan > 0
        if daily_cap_on and plan_cap_on:
            active_brake = "both"
        elif daily_cap_on:
            active_brake = "daily_cap"
        elif plan_cap_on:
            active_brake = "plan_limit"
        else:
            active_brake = "none"
        limit_stats = {
            "blocked_by_daily_cap": blocked_daily_cap,
            "blocked_by_symbol_veto": blocked_symbol_veto,
            "blocked_by_direction": blocked_direction,
            "blocked_by_category": blocked_category,
            "blocked_by_expression": blocked_expression,
            "blocked_by_missing_indicator": blocked_missing_indicator,
            "blocked_by_plan_limits": blocked_plan,
            "blocked_by_other": blocked_other,
            "blocked_by_risk_limits": blocked_risk,
            "blocked_by_risk_budget": blocked_risk_budget,
            "blocked_by_trigger_load": blocked_trigger_load,
            "blocked_by_archetype_load": blocked_archetype_load,
            "blocked_by_min_hold": blocked_min_hold,
            "blocked_by_min_flat": blocked_min_flat,
            "priority_skips": int(limit_entry.get("priority_skips", 0)),
            "execution_rate": execution_rate,
            "active_brake": active_brake,
            "risk_budget_used_pct": 0.0,
            "risk_budget_usage_by_symbol": {},
            "risk_budget_blocks_by_symbol": {},
            "blocked_totals": skipped_counts,
            "risk_block_breakdown": risk_breakdown,
            "risk_limit_hints": risk_limit_hints,
            "blocked_details": list(limit_entry["blocked_details"]),
            "executed_details": list(limit_entry["executed_details"]),
        }
        summary["limit_stats"] = limit_stats
        summary["limit_enforcement"] = limit_stats
        summary["active_brake"] = active_brake
        summary["execution_rate"] = execution_rate
        closes = {symbol: ctx.get("close") for symbol, ctx in indicator_context.items()}
        overnight_exposure = {}
        for symbol, qty in summary["positions_end"].items():
            close = closes.get(symbol)
            notional = qty * close if close is not None else None
            overnight_exposure[symbol] = {"quantity": qty, "notional": notional}
        summary["overnight_exposure"] = overnight_exposure
        trigger_stats_payload = {}
        for trigger_id, stats in trigger_activity.items():
            raw_reasons = stats.get("blocked_by_reason", {})
            if hasattr(raw_reasons, "items"):
                reason_map = dict(raw_reasons)
            else:
                reason_map = raw_reasons
            trigger_stats_payload[trigger_id] = {
                "executed": stats.get("executed", 0),
                "blocked": stats.get("blocked", 0),
                "blocked_by_reason": reason_map,
            }
        summary["trigger_stats"] = trigger_stats_payload
        quality_stats = self._build_quality_stats(day_key, risk_usage, risk_usage_events, latency_samples, trigger_load_events)
        if quality_stats["trigger_quality"]:
            summary["trigger_quality"] = quality_stats["trigger_quality"]
        if quality_stats["timeframe_quality"]:
            summary["timeframe_quality"] = quality_stats["timeframe_quality"]
        if quality_stats["hour_quality"]:
            summary["hour_quality"] = quality_stats["hour_quality"]
        if quality_stats["archetype_quality"]:
            summary["archetype_quality"] = quality_stats["archetype_quality"]
        if quality_stats["archetype_hour_quality"]:
            summary["archetype_hour_quality"] = quality_stats["archetype_hour_quality"]
        risk_budget_info = self._risk_budget_summary(day_key) or {}
        if not risk_budget_info:
            risk_budget_info = self._risk_budget_fallback(start_equity, risk_usage_events, budget_pct=self.daily_risk_budget_pct) or {}
        if risk_budget_info:
            summary["risk_budget"] = risk_budget_info
            limit_stats["risk_budget_used_pct"] = risk_budget_info["used_pct"]
            limit_stats["risk_budget_usage_by_symbol"] = risk_budget_info.get("symbol_usage_pct", {})
            limit_stats["risk_budget_blocks_by_symbol"] = risk_budget_info.get("blocks_by_symbol", {})
        else:
            summary["risk_budget"] = {}
        if plan_limits:
            summary["plan_limits"] = plan_limits
            min_trades = plan_limits.get("min_trades_per_day") or 0
            if min_trades and summary["executed_trades"] < min_trades:
                summary["missed_min_trades"] = True
            summary["cap_state"] = {
                "policy": {
                    "max_trades_per_day": plan_limits.get("max_trades_per_day"),
                    "max_triggers_per_symbol_per_day": plan_limits.get("max_triggers_per_symbol_per_day"),
                },
                "derived": {
                    "max_trades_per_day": plan_limits.get("derived_max_trades_per_day"),
                    "max_triggers_per_symbol_per_day": plan_limits.get("derived_max_triggers_per_symbol_per_day"),
                },
                "resolved": {
                    "max_trades_per_day": plan_limits.get("resolved_max_trades_per_day"),
                    "max_triggers_per_symbol_per_day": plan_limits.get("resolved_max_triggers_per_symbol_per_day"),
                },
                "session_caps": plan_limits.get("session_caps") or {},
                "flags": {
                    "strict_fixed_caps": plan_limits.get("strict_fixed_caps", False),
                    "legacy_mode": not plan_limits.get("strict_fixed_caps", False),
                },
                "inputs": plan_limits.get("cap_inputs") or {},
            }
        if self.adaptive_replanning:
            summary["judge_feedback"] = {}
        else:
            raw_feedback = self._judge_feedback(summary)
            summary["judge_feedback"] = raw_feedback
            # Emit plan_judged event for UI timeline
            self._emit_plan_judged_event(run_id, day_key, raw_feedback)
            feedback_obj = JudgeFeedback.model_validate(raw_feedback)
            run = self._apply_feedback_adjustments(run_id, feedback_obj, equity_return_pct > 0)
            summary["risk_adjustments"] = list(snapshot_adjustments(run.risk_adjustments or {}))
        (daily_dir / f"{day_key}.json").write_text(json.dumps(summary, indent=2))
        self.flattened_days.discard(day_key)
        return summary

    def _build_quality_stats(
        self,
        day_key: str,
        risk_usage: Dict[tuple[str, str], float],
        risk_usage_events: List[Dict[str, Any]],
        latency_samples: Dict[tuple[str, str], List[float]],
        trigger_load_events: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        entries = [
            entry
            for entry in self.portfolio.trade_log
            if self._timestamp_day(entry.get("timestamp")) == day_key
        ]
        trigger_quality: Dict[str, Dict[str, Any]] = {}
        timeframe_quality: Dict[str, Dict[str, Any]] = {}
        hour_quality: Dict[str, Dict[str, Any]] = {}
        archetype_quality: Dict[str, Dict[str, Any]] = {}
        archetype_hour_quality: Dict[str, Dict[str, Any]] = {}

        latency_lookup: Dict[tuple[str, str], float] = {
            key: (sum(vals) / len(vals)) for key, vals in latency_samples.items() if vals
        }

        def _excursions(entry: Dict[str, Any]) -> tuple[float | None, float | None, float | None]:
            symbol = entry.get("symbol")
            timeframe = entry.get("entry_timeframe") or entry.get("timeframe")
            entry_price = entry.get("entry_price")
            exit_price = entry.get("exit_price")
            entry_ts = entry.get("entry_timestamp")
            exit_ts = entry.get("timestamp")
            side = entry.get("entry_side")
            if not symbol or entry_price is None or exit_price is None or not entry_ts or not exit_ts or not timeframe:
                return (None, None, None)
            try:
                entry_dt = entry_ts if isinstance(entry_ts, datetime) else datetime.fromisoformat(str(entry_ts).replace("Z", "+00:00"))
                exit_dt = exit_ts if isinstance(exit_ts, datetime) else datetime.fromisoformat(str(exit_ts).replace("Z", "+00:00"))
            except ValueError:
                return (None, None, None)
            tf_map = self.market_data.get(symbol) or {}
            df = tf_map.get(str(timeframe))
            if df is None or df.empty:
                return (None, None, None)
            subset = df[(df.index >= entry_dt) & (df.index <= exit_dt)]
            if subset.empty:
                return (None, None, None)
            highs = subset["high"] if "high" in subset.columns else subset["close"]
            lows = subset["low"] if "low" in subset.columns else subset["close"]
            high_price = float(highs.max())
            low_price = float(lows.min())
            entry_price = float(entry_price)
            exit_price = float(exit_price)
            if side == "short":
                mae_pct = ((entry_price - high_price) / entry_price) * 100.0
                mfe_pct = ((entry_price - low_price) / entry_price) * 100.0
                best_price = low_price
                decay_pct = ((best_price - exit_price) / entry_price) * 100.0
            else:
                mae_pct = ((low_price - entry_price) / entry_price) * 100.0
                mfe_pct = ((high_price - entry_price) / entry_price) * 100.0
                best_price = high_price
                decay_pct = ((exit_price - best_price) / entry_price) * 100.0
            return (mae_pct, mfe_pct, decay_pct)

        def _archetype_for_entry(trigger_id: str | None) -> str:
            if not trigger_id:
                return "unknown"
            return self._archetype_for_trigger(day_key, trigger_id)

        for entry in entries:
            trigger_id = entry.get("entry_reason") or entry.get("reason") or "unknown"
            timeframe = entry.get("entry_timeframe") or "unknown"
            ts = entry.get("timestamp")
            hour = None
            if isinstance(ts, datetime):
                hour = ts.hour
            elif isinstance(ts, str) and len(ts) >= 13:
                try:
                    hour = int(ts[11:13])
                except ValueError:
                    hour = None
            pnl = float(entry.get("pnl", 0.0))
            key = f"{trigger_id}|{timeframe}"
            archetype = _archetype_for_entry(trigger_id)
            mae_pct, mfe_pct, decay_pct = _excursions(entry)
            payload = trigger_quality.setdefault(
                key,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                },
            )
            payload["pnl"] += pnl
            payload["trades"] += 1
            payload["wins"] += 1 if pnl > 0 else 0
            payload["losses"] += 1 if pnl < 0 else 0
            if pnl > 0:
                payload["win_abs_sum"] += abs(pnl)
            elif pnl < 0:
                payload["loss_abs_sum"] += abs(pnl)
            latency = latency_lookup.get((trigger_id, timeframe))
            if latency is not None:
                payload.setdefault("latencies", []).append(latency)
            if mae_pct is not None:
                payload["mae_sum"] += mae_pct
                payload["abs_mae_sum"] += abs(mae_pct)
            if mfe_pct is not None:
                payload["mfe_sum"] += mfe_pct
                payload["abs_mfe_sum"] += abs(mfe_pct)
            if decay_pct is not None:
                payload["decay_sum"] += decay_pct

            tf_payload = timeframe_quality.setdefault(
                str(timeframe),
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                },
            )
            tf_payload["pnl"] += pnl
            tf_payload["trades"] += 1
            tf_payload["wins"] += 1 if pnl > 0 else 0
            tf_payload["losses"] += 1 if pnl < 0 else 0
            if pnl > 0:
                tf_payload["win_abs_sum"] += abs(pnl)
            elif pnl < 0:
                tf_payload["loss_abs_sum"] += abs(pnl)
            if latency is not None:
                tf_payload.setdefault("latencies", []).append(latency)
            if mae_pct is not None:
                tf_payload["mae_sum"] += mae_pct
                tf_payload["abs_mae_sum"] += abs(mae_pct)
            if mfe_pct is not None:
                tf_payload["mfe_sum"] += mfe_pct
                tf_payload["abs_mfe_sum"] += abs(mfe_pct)
            if decay_pct is not None:
                tf_payload["decay_sum"] += decay_pct

            arch_payload = archetype_quality.setdefault(
                archetype,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                },
            )
            arch_payload["pnl"] += pnl
            arch_payload["trades"] += 1
            arch_payload["wins"] += 1 if pnl > 0 else 0
            arch_payload["losses"] += 1 if pnl < 0 else 0
            if pnl > 0:
                arch_payload["win_abs_sum"] += abs(pnl)
            elif pnl < 0:
                arch_payload["loss_abs_sum"] += abs(pnl)
            if latency is not None:
                arch_payload.setdefault("latencies", []).append(latency)
            if mae_pct is not None:
                arch_payload["mae_sum"] += mae_pct
                arch_payload["abs_mae_sum"] += abs(mae_pct)
            if mfe_pct is not None:
                arch_payload["mfe_sum"] += mfe_pct
                arch_payload["abs_mfe_sum"] += abs(mfe_pct)
            if decay_pct is not None:
                arch_payload["decay_sum"] += decay_pct

            if hour is not None:
                hr_payload = hour_quality.setdefault(
                    str(hour),
                    {
                        "pnl": 0.0,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "mae_sum": 0.0,
                        "mfe_sum": 0.0,
                        "decay_sum": 0.0,
                        "abs_mae_sum": 0.0,
                        "abs_mfe_sum": 0.0,
                        "win_abs_sum": 0.0,
                        "loss_abs_sum": 0.0,
                        "load_sum": 0.0,
                        "load_count": 0,
                    },
                )
                hr_payload["pnl"] += pnl
                hr_payload["trades"] += 1
                hr_payload["wins"] += 1 if pnl > 0 else 0
                hr_payload["losses"] += 1 if pnl < 0 else 0
                if pnl > 0:
                    hr_payload["win_abs_sum"] += abs(pnl)
                elif pnl < 0:
                    hr_payload["loss_abs_sum"] += abs(pnl)
                if latency is not None:
                    hr_payload.setdefault("latencies", []).append(latency)
                if mae_pct is not None:
                    hr_payload["mae_sum"] += mae_pct
                    hr_payload["abs_mae_sum"] += abs(mae_pct)
                if mfe_pct is not None:
                    hr_payload["mfe_sum"] += mfe_pct
                    hr_payload["abs_mfe_sum"] += abs(mfe_pct)
                if decay_pct is not None:
                    hr_payload["decay_sum"] += decay_pct
                arch_hour_key = f"{archetype}|{hour}"
                arch_hour_payload = archetype_hour_quality.setdefault(
                    arch_hour_key,
                    {
                        "pnl": 0.0,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "mae_sum": 0.0,
                        "mfe_sum": 0.0,
                        "decay_sum": 0.0,
                        "abs_mae_sum": 0.0,
                        "abs_mfe_sum": 0.0,
                        "win_abs_sum": 0.0,
                        "loss_abs_sum": 0.0,
                        "load_sum": 0.0,
                        "load_count": 0,
                    },
                )
                arch_hour_payload["pnl"] += pnl
                arch_hour_payload["trades"] += 1
                arch_hour_payload["wins"] += 1 if pnl > 0 else 0
                arch_hour_payload["losses"] += 1 if pnl < 0 else 0
                if pnl > 0:
                    arch_hour_payload["win_abs_sum"] += abs(pnl)
                elif pnl < 0:
                    arch_hour_payload["loss_abs_sum"] += abs(pnl)
                if latency is not None:
                    arch_hour_payload.setdefault("latencies", []).append(latency)
                if mae_pct is not None:
                    arch_hour_payload["mae_sum"] += mae_pct
                    arch_hour_payload["abs_mae_sum"] += abs(mae_pct)
                if mfe_pct is not None:
                    arch_hour_payload["mfe_sum"] += mfe_pct
                    arch_hour_payload["abs_mfe_sum"] += abs(mfe_pct)
                if decay_pct is not None:
                    arch_hour_payload["decay_sum"] += decay_pct

        # Attach risk usage totals (allocated + actual at stop if available).
        actual_usage: Dict[tuple[str, str], float] = defaultdict(float)
        for evt in risk_usage_events or []:
            trig = evt.get("trigger_id")
            tf = str(evt.get("timeframe") or "unknown")
            actual = evt.get("actual_risk_at_stop")
            if trig and actual is not None:
                actual_usage[(trig, tf)] += float(actual)

        for raw_key, risk_used in risk_usage.items():
            if isinstance(raw_key, str) and "|" in raw_key:
                trigger_id, timeframe = raw_key.split("|", 1)
            elif isinstance(raw_key, tuple) and len(raw_key) == 2:
                trigger_id, timeframe = raw_key
            else:
                trigger_id = str(raw_key)
                timeframe = "unknown"
            key = f"{trigger_id}|{timeframe}"
            payload = trigger_quality.setdefault(
                key,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                },
            )
            payload["risk_used_abs"] = payload.get("risk_used_abs", 0.0) + max(risk_used, 0.0)
            actual_total = actual_usage.get((trigger_id, timeframe))
            if actual_total is not None:
                payload["actual_risk_abs"] = payload.get("actual_risk_abs", 0.0) + float(actual_total)
            payload.setdefault("load_sum", 0.0)
            payload.setdefault("load_count", 0)
            payload.setdefault("latencies", [])

        timeframe_risk_usage: Dict[str, float] = defaultdict(float)
        hour_risk_usage: Dict[str, float] = defaultdict(float)
        timeframe_actual_usage: Dict[str, float] = defaultdict(float)
        hour_actual_usage: Dict[str, float] = defaultdict(float)
        archetype_risk_usage: Dict[str, float] = defaultdict(float)
        archetype_actual_usage: Dict[str, float] = defaultdict(float)
        archetype_hour_risk_usage: Dict[str, float] = defaultdict(float)
        archetype_hour_actual_usage: Dict[str, float] = defaultdict(float)
        for evt in trigger_load_events or []:
            tf = str(evt.get("timeframe") or "unknown")
            hr = evt.get("hour")
            load_val = float(evt.get("load", 0.0))
            trig = evt.get("trigger_id") or "unknown"
            arche = evt.get("archetype") or "unknown"
            trig_key = f"{trig}|{tf}"
            trig_payload = trigger_quality.setdefault(
                trig_key,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                },
            )
            trig_payload["load_sum"] += load_val
            trig_payload["load_count"] += 1
            tf_payload = timeframe_quality.setdefault(
                tf,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                },
            )
            tf_payload["load_sum"] += load_val
            tf_payload["load_count"] += 1
            if hr is not None:
                hr_payload = hour_quality.setdefault(
                    str(hr),
                    {
                        "pnl": 0.0,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "mae_sum": 0.0,
                        "mfe_sum": 0.0,
                        "decay_sum": 0.0,
                        "abs_mae_sum": 0.0,
                        "abs_mfe_sum": 0.0,
                        "win_abs_sum": 0.0,
                        "loss_abs_sum": 0.0,
                        "load_sum": 0.0,
                        "load_count": 0,
                    },
                )
                hr_payload["load_sum"] += load_val
                hr_payload["load_count"] += 1
        for evt in risk_usage_events or []:
            tf = str(evt.get("timeframe") or "unknown")
            hour_val = evt.get("hour")
            risk_used = float(evt.get("risk_used", 0.0))
            timeframe_risk_usage[tf] += max(risk_used, 0.0)
            actual_risk_val = evt.get("actual_risk_at_stop")
            if actual_risk_val is not None:
                timeframe_actual_usage[tf] += float(actual_risk_val)
            archetype = _archetype_for_entry(evt.get("trigger_id"))
            archetype_risk_usage[archetype] += max(risk_used, 0.0)
            if actual_risk_val is not None:
                archetype_actual_usage[archetype] += float(actual_risk_val)
            if hour_val is not None:
                hour_risk_usage[str(hour_val)] += max(risk_used, 0.0)
                if actual_risk_val is not None:
                    hour_actual_usage[str(hour_val)] += float(actual_risk_val)
                arch_hour_key = f"{archetype}|{hour_val}"
                archetype_hour_risk_usage[arch_hour_key] += max(risk_used, 0.0)
                if actual_risk_val is not None:
                    archetype_hour_actual_usage[arch_hour_key] += float(actual_risk_val)

        for tf, risk_used in timeframe_risk_usage.items():
            payload = timeframe_quality.setdefault(
                tf,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                },
            )
            payload["risk_used_abs"] = payload.get("risk_used_abs", 0.0) + risk_used
            actual_val = timeframe_actual_usage.get(tf)
            if actual_val is not None:
                payload["actual_risk_abs"] = payload.get("actual_risk_abs", 0.0) + actual_val
            payload.setdefault("latencies", [])

        for hour, risk_used in hour_risk_usage.items():
            payload = hour_quality.setdefault(
                hour,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                },
            )
            payload["risk_used_abs"] = payload.get("risk_used_abs", 0.0) + risk_used
            actual_val = hour_actual_usage.get(hour)
            if actual_val is not None:
                payload["actual_risk_abs"] = payload.get("actual_risk_abs", 0.0) + actual_val
            payload.setdefault("latencies", [])

        for archetype, risk_used in archetype_risk_usage.items():
            payload = archetype_quality.setdefault(
                archetype,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                },
            )
            payload["risk_used_abs"] = payload.get("risk_used_abs", 0.0) + risk_used
            actual_val = archetype_actual_usage.get(archetype)
            if actual_val is not None:
                payload["actual_risk_abs"] = payload.get("actual_risk_abs", 0.0) + actual_val
            payload.setdefault("latencies", [])

        for arch_hour, risk_used in archetype_hour_risk_usage.items():
            payload = archetype_hour_quality.setdefault(
                arch_hour,
                {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                },
            )
            payload["risk_used_abs"] = payload.get("risk_used_abs", 0.0) + risk_used
            actual_val = archetype_hour_actual_usage.get(arch_hour)
            if actual_val is not None:
                payload["actual_risk_abs"] = payload.get("actual_risk_abs", 0.0) + actual_val
            payload.setdefault("latencies", [])

        def _finalize(target: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            finalized: Dict[str, Dict[str, Any]] = {}
            for key, payload in target.items():
                trades = payload.get("trades", 0)
                pnl = payload.get("pnl", 0.0)
                risk_used = payload.get("risk_used_abs", 0.0)
                win_rate = (payload.get("wins", 0) / trades) if trades else 0.0
                latency_vals = payload.get("latencies") or []
                avg_latency = (sum(latency_vals) / len(latency_vals)) if latency_vals else None
                risk_per_trade = (risk_used / trades) if trades and risk_used else 0.0
                mean_r = (pnl / trades / risk_per_trade) if risk_per_trade else (pnl / trades if trades else 0.0)
                abs_mae_mean = (payload.get("abs_mae_sum", 0.0) / trades) if trades else 0.0
                abs_mfe_mean = (payload.get("abs_mfe_sum", 0.0) / trades) if trades else 0.0
                efficiency_mae = (mean_r / abs_mae_mean) if abs_mae_mean else 0.0
                efficiency_mfe = (mean_r / abs_mfe_mean) if abs_mfe_mean else 0.0
                asymmetry = 0.0
                win_abs = payload.get("win_abs_sum", 0.0)
                loss_abs = payload.get("loss_abs_sum", 0.0)
                if win_abs or loss_abs:
                    asymmetry = (win_abs - loss_abs) / max(win_abs + loss_abs, 1e-9)
                load_count = payload.get("load_count", 0)
                avg_load = (payload.get("load_sum", 0.0) / load_count) if load_count else 0.0
                actual_risk = payload.get("actual_risk_abs", 0.0)
                finalized[key] = {
                    "trades": trades,
                    "pnl": pnl,
                    "risk_used_abs": risk_used,
                    "actual_risk_abs": actual_risk,
                    "rpr": (pnl / risk_used) if risk_used else 0.0,
                    "rpr_allocated": (pnl / risk_used) if risk_used else 0.0,
                    "rpr_actual": (pnl / actual_risk) if actual_risk else 0.0,
                    "win_rate": win_rate,
                    "mean_r": mean_r,
                    "latency_seconds": avg_latency,
                    "wins": payload.get("wins", 0),
                    "losses": payload.get("losses", 0),
                    "mae_pct": (payload.get("mae_sum", 0.0) / trades) if trades else 0.0,
                    "mfe_pct": (payload.get("mfe_sum", 0.0) / trades) if trades else 0.0,
                    "response_decay_pct": (payload.get("decay_sum", 0.0) / trades) if trades else 0.0,
                    "relative_efficiency_mae": efficiency_mae,
                    "relative_efficiency_mfe": efficiency_mfe,
                    "asymmetry": asymmetry,
                    "avg_load": avg_load,
                    "load_count": load_count,
                }
            return finalized

        return {
            "trigger_quality": _finalize(trigger_quality),
            "timeframe_quality": _finalize(timeframe_quality),
            "hour_quality": _finalize(hour_quality),
            "archetype_quality": _finalize(archetype_quality),
            "archetype_hour_quality": _finalize(archetype_hour_quality),
        }

    def _emit_plan_generated_event(
        self,
        run_id: str,
        plan: "StrategyPlan",
        trigger_summary: List[Dict[str, Any]],
        trigger_diff: Dict[str, Any] | None = None,
        *,
        event_ts: datetime | None = None,
    ) -> None:
        """Emit a plan_generated event to the event store for UI timeline display."""
        try:
            ts = event_ts or plan.generated_at
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            event = Event(
                event_id=f"{run_id}-plan-{plan.plan_id}",
                ts=ts,
                source="llm_strategist",
                type="plan_generated",
                payload={
                    "plan_id": plan.plan_id,
                    "strategy_id": plan.plan_id,  # For UI compatibility
                    "symbol": self.pairs[0] if self.pairs else "MULTI",
                    "regime": plan.regime,
                    "market_regime": plan.regime,
                    "num_triggers": len(plan.triggers),
                    "max_trades_per_day": plan.max_trades_per_day,
                    "triggers": trigger_summary[:10],  # Limit for payload size
                    "global_view": plan.global_view,
                    "trigger_diff": trigger_diff,
                },
                run_id=run_id,
                correlation_id=run_id,
            )
            self._event_store.append(event)
        except Exception as exc:
            logger.warning("Failed to emit plan_generated event: %s", exc)

    def _emit_plan_judged_event(
        self,
        run_id: str,
        day_key: str,
        judge_feedback: Dict[str, Any],
    ) -> None:
        """Emit a plan_judged event to the event store for UI timeline display."""
        from datetime import datetime as dt
        try:
            ts = dt.fromisoformat(f"{day_key}T23:59:59+00:00")
            score = judge_feedback.get("score", 50)
            event = Event(
                event_id=f"{run_id}-judge-{day_key}",
                ts=ts,
                source="llm_strategist",
                type="plan_judged",
                payload={
                    "day": day_key,
                    "score": score,
                    "overall_score": score,  # For UI compatibility
                    "notes": judge_feedback.get("notes"),
                    "recommendations": judge_feedback.get("recommendations", []),
                    "strategist_constraints": judge_feedback.get("strategist_constraints"),
                    "risk_adjustments": judge_feedback.get("risk_adjustments"),
                },
                run_id=run_id,
                correlation_id=run_id,
            )
            self._event_store.append(event)
        except Exception as exc:
            logger.warning("Failed to emit plan_judged event: %s", exc)

    def _judge_feedback(
        self,
        summary: Dict[str, Any],
        trade_metrics: TradeMetrics | None = None,
    ) -> Dict[str, Any]:
        """Generate judge feedback using JudgeFeedbackService.

        The service computes deterministic heuristics and either:
        - Returns them directly via shim transport (fast, deterministic)
        - Passes them as context to LLM for richer evaluation (matches live judge)

        Args:
            summary: Performance snapshot with return_pct, trade_count, etc.
            trade_metrics: Optional deterministic TradeMetrics for quality assessment
        """
        strategy_context = {
            "risk_params": self.risk_params,
            "active_risk_limits": self.active_risk_limits.model_dump() if hasattr(self.active_risk_limits, 'model_dump') else self.active_risk_limits,
            "pairs": self.pairs,
        }
        feedback = self.judge_service.generate_feedback(
            summary=summary,
            trade_metrics=trade_metrics,
            strategy_context=strategy_context,
        )
        return feedback.model_dump()

    def _apply_feedback_adjustments(self, run_id: str, feedback: JudgeFeedback, winning_day: bool):
        run = self.run_registry.get_strategy_run(run_id)
        apply_judge_risk_feedback(run, feedback, winning_day)
        updated = self.run_registry.update_strategy_run(run)
        self._refresh_risk_state_from_run(updated)
        return updated

    # ── Adaptive Judge Workflow Methods ──────────────────────────────────────────

    def _judge_trigger_reason(self, ts: datetime, force_check: bool = False) -> tuple[bool, str | None]:
        """Determine if we should run the intraday judge evaluation and why.

        Conditions for running judge (any of):
        1. Enough time has passed since last evaluation (judge_cadence_hours)
        2. Enough trades have occurred since last evaluation (judge_check_after_trades)
        3. Force check requested (e.g., significant drawdown detected)
        """
        if not self.adaptive_replanning:
            return False, None

        if self.last_judge_time is None:
            if self.next_judge_time is None:
                return False, None
            time_passed = ts >= self.next_judge_time
            trades_trigger = self.trades_since_last_judge >= self.judge_check_after_trades
            return self._judge_trigger_outcome(force_check, time_passed, trades_trigger)

        # Check time-based cadence (honor adjusted next_judge_time if set)
        if self.next_judge_time is not None:
            time_passed = ts >= self.next_judge_time
        else:
            time_passed = (ts - self.last_judge_time) >= self.judge_cadence

        # Check trade-based trigger
        trades_trigger = self.trades_since_last_judge >= self.judge_check_after_trades

        return self._judge_trigger_outcome(force_check, time_passed, trades_trigger)

    def _judge_trigger_outcome(
        self,
        force_check: bool,
        time_passed: bool,
        trades_trigger: bool,
    ) -> tuple[bool, str | None]:
        if force_check:
            return True, "force"
        if time_passed and trades_trigger:
            return True, "cadence_and_trade_count"
        if time_passed:
            return True, "cadence"
        if trades_trigger:
            return True, "trade_count"
        return False, None

    def _should_run_judge(self, ts: datetime, force_check: bool = False) -> bool:
        should_run, _ = self._judge_trigger_reason(ts, force_check=force_check)
        return should_run

    def _get_intraday_performance_snapshot(
        self, ts: datetime, day_key: str, latest_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Build a performance snapshot for intraday judge evaluation."""
        # Get current equity
        current_equity = self.portfolio.cash
        for symbol, qty in self.portfolio.positions.items():
            if symbol in latest_prices and qty != 0:
                current_equity += qty * latest_prices[symbol]

        # Get today's fills so far
        today_fills = []
        invalid_timestamps = 0
        invalid_samples: List[Any] = []
        for fill in self.portfolio.fills:
            ts_raw = fill.get("timestamp")
            ts = self._coerce_timestamp(ts_raw)
            if ts is None:
                invalid_timestamps += 1
                if len(invalid_samples) < 3:
                    invalid_samples.append(ts_raw)
                continue
            if ts.strftime("%Y-%m-%d") == day_key:
                today_fills.append({**fill, "timestamp": ts})

        # Calculate daily return
        anchor_equity = self.daily_loss_anchor_by_day.get(day_key, self.initial_cash)
        daily_return_pct = ((current_equity - anchor_equity) / anchor_equity) * 100 if anchor_equity > 0 else 0.0

        # Count winning vs losing trades today
        winning_trades = 0
        losing_trades = 0
        for fill in today_fills:
            pnl = fill.get("pnl", fill.get("realized_pnl", 0.0))
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
        if invalid_timestamps:
            logger.warning("Found %d fills with invalid timestamps for day=%s", invalid_timestamps, day_key)

        return {
            "timestamp": ts.isoformat(),
            "day_key": day_key,
            "equity": current_equity,
            "anchor_equity": anchor_equity,
            "return_pct": daily_return_pct,
            "trade_count": len(today_fills),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "positions": dict(self.portfolio.positions),
            "cash": self.portfolio.cash,
            "fill_timestamp_issues": {
                "invalid_count": invalid_timestamps,
                "samples": invalid_samples,
            },
        }

    def _run_intraday_judge(
        self,
        ts: datetime,
        run_id: str,
        day_key: str,
        latest_prices: Dict[str, float],
        current_plan: StrategyPlan | None,
        *,
        trigger_reason: str | None = None,
    ) -> Dict[str, Any]:
        """Run intraday judge evaluation and decide if replan is needed.

        Returns dict with:
        - score: float 0-100
        - should_replan: bool
        - feedback: dict with constraints and recommendations
        - reason: str explaining the decision
        """
        snapshot = self._get_intraday_performance_snapshot(ts, day_key, latest_prices)

        # Compute deterministic trade quality metrics
        day_start = datetime(ts.year, ts.month, ts.day, tzinfo=ts.tzinfo)
        since_ts = day_start
        if self.last_judge_time and self.last_judge_time.strftime("%Y-%m-%d") == day_key:
            since_ts = self.last_judge_time
        trigger_catalog = self.plan_limits_by_day.get(day_key, {}).get("trigger_catalog", {})
        trade_metrics = compute_trade_metrics(
            fills=self.portfolio.fills,
            start_time=day_start,
            end_time=ts,
            trigger_catalog=trigger_catalog,
        )

        # Assess current position quality
        position_quality = assess_position_quality(
            positions=self.portfolio.positions,
            entry_prices=self.portfolio.avg_entry_price,
            current_prices=latest_prices,
            position_opened_times={
                sym: meta.get("opened_at") for sym, meta in self.portfolio.position_meta.items()
                if meta.get("opened_at")
            },
            current_time=ts,
        )

        # Build intraday summary for judge with deterministic metrics
        summary = {
            "return_pct": snapshot["return_pct"],
            "trade_count": snapshot["trade_count"],
            "positions_end": snapshot["positions"],
            "equity": snapshot["equity"],
            "anchor_equity": snapshot["anchor_equity"],
            "winning_trades": snapshot["winning_trades"],
            "losing_trades": snapshot["losing_trades"],
            "fill_timestamp_issues": snapshot.get("fill_timestamp_issues", {}),
            # Deterministic quality metrics
            "trade_metrics": trade_metrics.to_dict(),
            "position_quality": [
                {
                    "symbol": pq.symbol,
                    "unrealized_pnl_pct": pq.unrealized_pnl_pct,
                    "hold_hours": pq.hold_duration_hours,
                    "risk_score": pq.risk_score,
                    "is_underwater": pq.is_underwater,
                }
                for pq in position_quality
            ],
            "fills_since_last_judge": self._collect_fill_details(day_key, since_ts, ts),
            "trigger_attempts": self._build_trigger_attempt_stats(day_key, since_ts, ts),
        }
        if current_plan:
            summary["active_triggers"] = [
                {
                    "id": trigger.id,
                    "symbol": trigger.symbol,
                    "timeframe": trigger.timeframe,
                    "direction": trigger.direction,
                    "category": trigger.category,
                    "confidence": trigger.confidence_grade,
                    "entry_rule": trigger.entry_rule,
                    "exit_rule": trigger.exit_rule,
                    "hold_rule": trigger.hold_rule,
                    "stop_loss_pct": trigger.stop_loss_pct,
                }
                for trigger in current_plan.triggers
            ]
            if self.last_slot_report:
                summary["indicator_context"] = self.last_slot_report.get("indicator_context", {}) or {}
                summary["market_structure"] = self.last_slot_report.get("market_structure", {}) or {}
            if self.latest_factor_exposures:
                summary["factor_exposures"] = self.latest_factor_exposures
            risk_budget = self._risk_budget_summary(day_key) or {}
            if not risk_budget:
                risk_budget = self._risk_budget_fallback(
                    snapshot.get("anchor_equity") or snapshot.get("equity"),
                    self.risk_usage_events_by_day.get(day_key, []),
                    budget_pct=self.daily_risk_budget_pct,
                ) or {}
            summary["risk_state"] = {
                "max_trades_per_day": current_plan.max_trades_per_day,
                "trades_executed_today": snapshot["trade_count"],
                "daily_risk_budget_pct": current_plan.risk_constraints.max_daily_risk_budget_pct
                if current_plan.risk_constraints
                else None,
                "risk_budget_used_pct": risk_budget.get("used_pct"),
                "risk_budget_usage_by_symbol": risk_budget.get("symbol_usage_pct", {}),
                "risk_budget_blocks_by_symbol": risk_budget.get("blocks_by_symbol", {}),
            }

        # Run judge feedback with quality metrics
        feedback = self._judge_feedback(summary, trade_metrics=trade_metrics)

        # Use deterministic quality score as base, allow judge to adjust
        deterministic_score = trade_metrics.quality_score
        judge_score = feedback.get("score", 50.0)
        # Weighted average: 60% deterministic, 40% judge adjustments
        score = (deterministic_score * 0.6) + (judge_score * 0.4)

        # Determine if replan is needed
        should_replan = False
        replan_reason = None

        # Condition 1: Score below threshold
        if score < self.judge_replan_threshold:
            should_replan = True
            replan_reason = f"Judge score {score:.1f} below threshold {self.judge_replan_threshold}"

        # Condition 2: Significant drawdown (>2% intraday)
        if snapshot["return_pct"] < -2.0 and not should_replan:
            should_replan = True
            replan_reason = f"Significant intraday drawdown: {snapshot['return_pct']:.2f}%"

        # Condition 3: No trades but should have (triggers exist but not firing)
        if snapshot["trade_count"] == 0 and current_plan and len(current_plan.triggers) > 3:
            hours_elapsed = (ts - current_plan.generated_at).total_seconds() / 3600
            if hours_elapsed > 4:  # 4+ hours with triggers but no trades
                should_replan = True
                replan_reason = f"No trades in {hours_elapsed:.1f}h despite {len(current_plan.triggers)} triggers"

        shim_replan_suppressed = False
        if self.use_judge_shim and should_replan:
            shim_replan_suppressed = True
            should_replan = False
            replan_reason = "Judge shim suppresses replans"

        # Update tracking state
        self.last_judge_time = ts
        self.next_judge_time = ts + self.judge_cadence
        self.trades_since_last_judge = 0

        # Adjust cadence based on conditions
        if snapshot["return_pct"] < -1.0:
            # Drawdown: check more frequently
            self.next_judge_time = ts + timedelta(hours=max(1.0, self.judge_cadence_hours / 2))
        elif score > 70 and snapshot["return_pct"] > 0.5:
            # Good performance: can check less frequently
            self.next_judge_time = ts + timedelta(hours=min(8.0, self.judge_cadence_hours * 1.5))

        result = {
            "timestamp": ts.isoformat(),
            "score": score,
            "should_replan": should_replan,
            "replan_reason": replan_reason,
            "trigger_reason": trigger_reason,
            "feedback": feedback,
            "snapshot": snapshot,
            "next_judge_time": self.next_judge_time.isoformat(),
        }

        # Log the evaluation
        self.intraday_judge_history.append(result)
        self._add_event("intraday_judge", {
            "score": score,
            "should_replan": should_replan,
            "reason": replan_reason,
            "trigger_reason": trigger_reason,
        })
        plan_id = current_plan.plan_id if current_plan else None
        self._emit_event(
            "plan_judged",
            {
                "date": day_key,
                "plan_id": plan_id,
                "score": feedback.get("score"),
                "notes": feedback.get("notes"),
                "constraints": feedback.get("constraints"),
                "strategist_constraints": feedback.get("strategist_constraints"),
                "executed_trades": snapshot.get("trade_count"),
                "return_pct": snapshot.get("return_pct"),
                "trigger_attempts": summary.get("trigger_attempts"),
                "should_replan": should_replan,
                "replan_reason": replan_reason,
                "shim_replan_suppressed": shim_replan_suppressed,
                "trigger_reason": trigger_reason,
                "scope": "intraday",
            },
            run_id=run_id,
            correlation_id=plan_id,
            event_ts=ts,
        )

        if should_replan:
            logger.info(
                "Intraday judge triggered replan: score=%.1f reason=%s",
                score, replan_reason
            )
            self.judge_triggered_replans.append({
                "timestamp": ts.isoformat(),
                "day_key": day_key,
                "score": score,
                "reason": replan_reason,
            })

        return result

    def _on_trade_executed(self, ts: datetime) -> None:
        """Called after a trade is executed to update judge tracking."""
        self.trades_since_last_judge += 1
