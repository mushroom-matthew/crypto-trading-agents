"""Backtesting harness that wires the LLM strategist into the simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Literal

import pandas as pd

from agents.analytics import IndicatorWindowConfig, PortfolioHistory, build_asset_state, compute_indicator_snapshot, compute_portfolio_state
from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from agents.strategies.risk_engine import RiskEngine, RiskProfile
from agents.strategies.strategy_memory import build_strategy_memory
from agents.strategies.trade_risk import TradeRiskEvaluator
from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
from backtesting.dataset import load_ohlcv
from backtesting.reports import write_run_summary
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from schemas.llm_strategist import AssetState, LLMInput, PortfolioState, StrategyPlan
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings
from services.risk_adjustment_service import apply_judge_risk_feedback, effective_risk_limits, snapshot_adjustments, build_risk_profile
from services.strategy_run_registry import StrategyRunConfig, StrategyRunRegistry, strategy_run_registry
from services.strategist_plan_service import StrategistPlanService
from tools import execution_tools
from trading_core.trigger_compiler import compile_plan
from trading_core.trigger_budget import enforce_trigger_budget
from trading_core.execution_engine import BlockReason

logger = logging.getLogger(__name__)


def _new_limit_entry() -> Dict[str, Any]:
    return {
        "trades_attempted": 0,
        "trades_executed": 0,
        "skipped": defaultdict(int),
        "risk_block_breakdown": defaultdict(int),
        "risk_limit_hints": defaultdict(int),
        "blocked_details": [],
        "executed_details": [],
    }


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "time" in df.columns:
        df = df.set_index("time")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    else:
        raise ValueError("Dataframe must include time column")
    return df.sort_index()


@dataclass
class PortfolioTracker:
    initial_cash: float
    fee_rate: float
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    avg_entry_price: Dict[str, float] = field(default_factory=dict)
    position_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    equity_records: List[Dict[str, Any]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

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
            }
        )

    def _update_position(
        self,
        symbol: str,
        delta_qty: float,
        price: float,
        timestamp: datetime,
        reason: str | None = None,
        timeframe: str | None = None,
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

    def execute(self, order: Order) -> None:
        qty = max(order.quantity, 0.0)
        if qty <= 0:
            return
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
        self._update_position(order.symbol, delta, order.price, order.timestamp, order.reason, order.timeframe)
        self.fills.append(
            {
                "timestamp": order.timestamp,
                "symbol": order.symbol,
                "side": order.side,
                "qty": qty,
                "price": order.price,
                "fee": fee,
                "reason": order.reason,
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
        plan_provider: StrategyPlanProvider | None = None,
        market_data: Dict[str, Dict[str, pd.DataFrame]] | None = None,
        run_registry: StrategyRunRegistry | None = None,
        flatten_positions_daily: bool = False,
        flatten_notional_threshold: float = 0.0,
        flatten_session_boundary_hour: int | None = None,
        session_trade_multipliers: Sequence[Mapping[str, float | int]] | None = None,
        timeframe_trigger_caps: Mapping[str, int] | None = None,
        flatten_policy: str | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must be provided")
        self.pairs = list(pairs)
        self.start = self._normalize_datetime(start)
        self.end = self._normalize_datetime(end)
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.timeframes = list(timeframes)
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
        self.portfolio = PortfolioTracker(initial_cash=initial_cash, fee_rate=fee_rate)
        self.calls_per_day = max(1, llm_calls_per_day)
        self.plan_interval = timedelta(hours=24 / self.calls_per_day)
        self.plan_provider = plan_provider or StrategyPlanProvider(llm_client, cache_dir=cache_dir, llm_calls_per_day=self.calls_per_day)
        self.run_registry = run_registry or strategy_run_registry
        self.plan_service = StrategistPlanService(plan_provider=self.plan_provider, registry=self.run_registry)
        self.window_configs = {tf: IndicatorWindowConfig(timeframe=tf) for tf in self.timeframes}
        if isinstance(risk_params, RiskLimitSettings):
            self.base_risk_limits = risk_params
        else:
            payload = risk_params or {}
            self.base_risk_limits = RiskLimitSettings.model_validate(payload)
        self.active_risk_limits = self.base_risk_limits
        self.active_risk_adjustments: Dict[str, RiskAdjustmentState] = {}
        self.risk_params = self.active_risk_limits.to_risk_params()
        self.risk_profile = RiskProfile()
        self.daily_risk_budget_state: Dict[str, Dict[str, float]] = {}
        self.daily_risk_budget_pct = self.active_risk_limits.max_daily_risk_budget_pct
        self.prompt_template = prompt_template_path.read_text() if prompt_template_path and prompt_template_path.exists() else None
        self.slot_reports_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trigger_activity_by_day: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.skipped_activity_by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
        self.session_trade_multipliers = list(session_trade_multipliers) if session_trade_multipliers else None
        self.timeframe_trigger_caps = {str(tf): int(cap) for tf, cap in (timeframe_trigger_caps or {}).items() if cap is not None}
        self.flatten_policy = flatten_policy or (
            "session_close_utc"
            if flatten_session_boundary_hour is not None
            else ("daily_close" if flatten_positions_daily else "none")
        )
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
        logger.debug(
            "Initialized backtester pairs=%s timeframes=%s start=%s end=%s plan_interval_hours=%.2f",
            self.pairs,
            self.timeframes,
            self.start,
            self.end,
            self.plan_interval.total_seconds() / 3600,
        )

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
            frames: Dict[str, pd.DataFrame] = {}
            for timeframe, df in tf_map.items():
                subset = df[df.index <= timestamp]
                if not subset.empty:
                    frames[timeframe] = subset
            if not frames:
                continue
            snapshots = [
                compute_indicator_snapshot(frame, symbol=pair, timeframe=timeframe, config=self.window_configs[timeframe])
                for timeframe, frame in frames.items()
            ]
            asset_states[pair] = build_asset_state(pair, snapshots)
        return asset_states

    def _llm_input(self, timestamp: datetime, asset_states: Dict[str, AssetState]) -> LLMInput:
        portfolio_state = self.portfolio.portfolio_state(timestamp)
        context: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "pairs": self.pairs,
        }
        if self.last_slot_report:
            context["recent_activity"] = self.last_slot_report
        if self.latest_daily_summary:
            context["latest_daily_report"] = self.latest_daily_summary
            context["judge_feedback"] = self.latest_daily_summary.get("judge_feedback")
        context["strategy_memory"] = build_strategy_memory(self.memory_history)
        if self.judge_constraints:
            context["strategist_constraints"] = self.judge_constraints
        if self.active_risk_adjustments:
            context["risk_adjustments"] = list(snapshot_adjustments(self.active_risk_adjustments))
        context["risk_limits"] = self.active_risk_limits.to_risk_params()
        return LLMInput(
            portfolio=portfolio_state,
            assets=list(asset_states.values()),
            risk_params=self.risk_params,
            global_context=context,
        )

    def _refresh_risk_state_from_run(self, run) -> None:
        self.base_risk_limits = run.config.risk_limits or self.base_risk_limits
        self.active_risk_adjustments = dict(run.risk_adjustments or {})
        self.active_risk_limits = effective_risk_limits(run)
        self.risk_params = self.active_risk_limits.to_risk_params()
        self.risk_profile = build_risk_profile(run)
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
            config = StrategyRunConfig(
                symbols=self.pairs,
                timeframes=self.timeframes,
                history_window_days=history_days,
                plan_cadence_hours=cadence_hours,
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
            run = self.run_registry.update_strategy_run(run)
        self._refresh_risk_state_from_run(run)

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

    def _archetype_for_trigger(self, day_key: str, trigger_id: str) -> str:
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
    ) -> List[Dict[str, Any]]:
        executed_records: List[Dict[str, Any]] = []
        limit_entry = self.limit_enforcement_by_day[day_key]
        reason_map = set(item.value for item in BlockReason)
        custom_reasons = {"timeframe_cap", "session_cap", "trigger_load"}

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
            if normalized == BlockReason.RISK.value:
                key = reason or BlockReason.RISK.value
                limit_entry["risk_block_breakdown"][key] += 1
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
            limit_entry["executed_details"].append(
                {
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
            )

        if blocked_entries:
            for block in blocked_entries:
                limit_entry["trades_attempted"] += 1
                _record_block_entry(block, "trigger_engine")
        def _record_hints(order: Order) -> None:
            hints = self._assess_risk_limits(order, portfolio_state)
            if not hints:
                return
            for hint in hints:
                limit_entry["risk_limit_hints"][hint] += 1

        if not plan_payload or not compiled_payload:
            for order in orders:
                limit_entry["trades_attempted"] += 1
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
                    continue
                self._commit_risk_budget(day_key, allowance, order.symbol)
                limit_entry["trades_executed"] += 1
                trigger_id = order.reason
                risk_used = allowance or 0.0
                self.risk_usage_by_day[day_key][(trigger_id, order.timeframe)] += risk_used
                self.risk_usage_events_by_day[day_key].append(
                    {
                        "trigger_id": trigger_id,
                        "timeframe": order.timeframe,
                        "hour": order.timestamp.hour,
                        "risk_used": risk_used,
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
                self.portfolio.execute(order)
                executed_records.append(
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "price": order.price,
                        "timeframe": order.timeframe,
                        "reason": order.reason,
                        "risk_used": allowance,
                        "latency_seconds": 0.0,
                    }
                )
                self.trigger_activity_by_day[day_key][order.reason]["executed"] += 1
                logger.debug(
                    "Executed order (no plan gating) trigger=%s symbol=%s side=%s qty=%.6f price=%.2f",
                    order.reason,
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.price,
                )
            return executed_records

        def _gated_trigger_id(reason: str) -> str:
            for suffix in ("_exit", "_flat"):
                if reason.endswith(suffix):
                    return reason[: -len(suffix)]
            return reason

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
            base_cap = plan_limits.get("derived_max_trades_per_day") or plan_limits.get("max_trades_per_day")
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
            trigger_id = _gated_trigger_id(raw_trigger_id)
            archetype = self._archetype_for_trigger(day_key, trigger_id)
            arche_load, arche_key = _archetype_state(archetype, order.timeframe, order.timestamp)
            events_payload = [{"trigger_id": trigger_id, "timestamp": timestamp}]
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
                self.risk_usage_events_by_day[day_key].append(
                    {
                        "trigger_id": raw_trigger_id,
                        "timeframe": order.timeframe,
                        "hour": order.timestamp.hour,
                        "risk_used": risk_used,
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
                self.portfolio.execute(order)
                executed_records.append(
                    {
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
                    }
                )
                self.trigger_activity_by_day[day_key][order.reason]["executed"] += 1
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
        self.trigger_activity_by_day.clear()
        self.skipped_activity_by_day.clear()
        self.limit_enforcement_by_day.clear()
        self.plan_limits_by_day.clear()
        self.flattened_days.clear()
        self.daily_risk_budget_state.clear()
        self.trigger_load_by_day.clear()
        self.archetype_load_by_day.clear()
        self.current_day_key = None
        self.latest_daily_summary = None
        self.last_slot_report = None
        self.current_run_id = run_id

        for ts in all_timestamps:
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
                self._reset_risk_budget_for_day(day_key)
                execution_tools.reset_run_state(run_id)

            new_plan_needed = current_plan is None or ts >= current_plan.valid_until
            if new_plan_needed:
                asset_states = self._asset_states(ts)
                if not asset_states:
                    continue
                active_assets = asset_states
                llm_input = self._llm_input(ts, asset_states)
                day_start = datetime(ts.year, ts.month, ts.day, tzinfo=ts.tzinfo)
                slot_seconds = max(1, int(self.plan_interval.total_seconds()))
                elapsed = max(0, int((ts - day_start).total_seconds()))
                slot_index = elapsed // slot_seconds
                slot_start = day_start + timedelta(seconds=slot_index * slot_seconds)
                slot_end = min(slot_start + self.plan_interval, day_start + timedelta(days=1))
                current_plan = self.plan_service.generate_plan_for_run(
                    run_id,
                    llm_input,
                    plan_date=slot_start,
                    prompt_template=self.prompt_template,
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
                        "generated_at": slot_start,
                        "valid_until": slot_end,
                    }
                )
                current_plan = self._apply_trigger_budget(current_plan)
                sizing_targets = {
                    rule.symbol: (rule.target_risk_pct or self.active_risk_limits.max_position_risk_pct)
                    for rule in current_plan.sizing_rules
                }
                derived_cap = getattr(current_plan, "_derived_trade_cap", None)
                compiled_plan = compile_plan(current_plan)
                current_plan_payload = current_plan.model_dump()
                current_compiled_payload = compiled_plan.model_dump()
                risk_engine = RiskEngine(
                    current_plan.risk_constraints,
                    {rule.symbol: rule for rule in current_plan.sizing_rules},
                    daily_anchor_equity=self.portfolio.portfolio_state(ts).equity,
                    risk_profile=self.risk_profile,
                )
                trigger_engine = TriggerEngine(current_plan, risk_engine, trade_risk=TradeRiskEvaluator(risk_engine))
                plan_log.append(
                    {
                        "plan_id": current_plan.plan_id,
                        "generated_at": current_plan.generated_at.isoformat(),
                        "valid_until": current_plan.valid_until.isoformat(),
                        "regime": current_plan.regime,
                        "num_triggers": len(current_plan.triggers),
                        "max_trades_per_day": current_plan.max_trades_per_day,
                        "derived_max_trades_per_day": derived_cap or current_plan.max_trades_per_day,
                        "min_trades_per_day": current_plan.min_trades_per_day,
                        "max_triggers_per_symbol_per_day": current_plan.max_triggers_per_symbol_per_day,
                        "trigger_budgets": current_plan.trigger_budgets,
                    }
                )
                self.plan_limits_by_day[day_key] = {
                    "plan_id": current_plan.plan_id,
                    "max_trades_per_day": current_plan.max_trades_per_day,
                    "derived_max_trades_per_day": getattr(current_plan, "_derived_trade_cap", None),
                    "min_trades_per_day": current_plan.min_trades_per_day,
                    "allowed_symbols": current_plan.allowed_symbols,
                    "max_triggers_per_symbol_per_day": current_plan.max_triggers_per_symbol_per_day,
                    "trigger_budgets": dict(current_plan.trigger_budgets or {}),
                    "trigger_budget_trimmed": dict(self.latest_trigger_trim),
                    "session_trade_multipliers": self.session_trade_multipliers,
                    "timeframe_trigger_caps": self.timeframe_trigger_caps,
                    "flatten_policy": self.flatten_policy,
                    "trigger_catalog": {
                        trigger.id: {"symbol": trigger.symbol, "category": trigger.category, "direction": trigger.direction}
                        for trigger in current_plan.triggers
                        if trigger.id
                    },
                }
                logger.info(
                    "Generated plan plan_id=%s slot_start=%s slot_end=%s triggers=%s",
                    current_plan.plan_id,
                    slot_start.isoformat(),
                    slot_end.isoformat(),
                    len(current_plan.triggers),
                )
            if trigger_engine is None or current_plan is None:
                continue

            slot_orders: List[Dict[str, Any]] = []
            indicator_briefs = self._indicator_briefs(active_assets)
            for pair in self.pairs:
                for timeframe in self.timeframes:
                    df = self.market_data.get(pair, {}).get(timeframe)
                    if df is None or ts not in df.index:
                        continue
                    bar = self._build_bar(pair, timeframe, ts)
                    subset = df[df.index <= ts]
                    indicator = compute_indicator_snapshot(subset, symbol=pair, timeframe=timeframe, config=self.window_configs[timeframe])
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
                    orders, blocked_entries = trigger_engine.on_bar(bar, indicator, portfolio_state, asset_state)
                    current_load = len(orders) + (len(blocked_entries) if blocked_entries else 0)
                    executed_records = self._process_orders_with_limits(
                        run_id,
                        day_key,
                        orders,
                        portfolio_state,
                        current_plan_payload,
                        current_compiled_payload,
                        blocked_entries,
                        current_load=current_load,
                    )
                    slot_orders.extend(executed_records)
                    if timeframe == self.timeframes[0]:
                        latest_prices[pair] = bar.close
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
            }
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
        run_summary = write_run_summary(daily_reports, run_summary_path)
        summary = {
            "final_equity": final_equity,
            "equity_return_pct": equity_return_pct,
            "gross_trade_return_pct": gross_trade_return_pct,
            "return_pct": equity_return_pct,
            "run_summary": run_summary,
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

    def _apply_trigger_budget(self, plan: StrategyPlan) -> StrategyPlan:
        default_cap = plan.max_triggers_per_symbol_per_day or self.default_symbol_trigger_cap
        trimmed_plan, stats = enforce_trigger_budget(plan, default_cap=default_cap, fallback_symbols=self.pairs)
        trimmed = sum(stats.values())
        meaningful = {symbol: count for symbol, count in stats.items() if count > 0}
        self.latest_trigger_trim = meaningful
        if trimmed > 0:
            logger.info("Trimmed %s triggers via budget controls: %s", trimmed, meaningful)
        return trimmed_plan

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

    def _reset_risk_budget_for_day(self, day_key: str) -> None:
        pct = self.daily_risk_budget_pct or 0.0
        if pct <= 0:
            self.daily_risk_budget_state.pop(day_key, None)
            return
        if self.portfolio.equity_records:
            equity = float(self.portfolio.equity_records[-1]["equity"])
        else:
            equity = float(self.initial_cash)
        budget_abs = equity * (pct / 100.0)
        self.daily_risk_budget_state[day_key] = {
            "budget_pct": pct,
            "start_equity": equity,
            "budget_abs": budget_abs,
            "used_abs": 0.0,
            "symbol_usage": defaultdict(float),
            "blocks": defaultdict(int),
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
        # Adaptive boost: if prior day risk usage <10%, allow higher per-trade risk up to 3x.
        prev_usage = 100.0
        if self.latest_daily_summary and "risk_budget" in self.latest_daily_summary:
            prev_usage = self.latest_daily_summary["risk_budget"].get("used_pct", 100.0)
        adaptive_multiplier = 3.0 if prev_usage < 10.0 else 1.0
        risk_fraction = max(target_risk_pct * adaptive_multiplier, 0.0) / 100.0
        if risk_fraction <= 0:
            return 0.0
        contribution = notional * risk_fraction
        used = entry.get("used_abs", 0.0)
        if used + contribution > budget + 1e-9:
            return None
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
            "symbol_usage_pct": symbol_usage_pct,
            "blocks_by_symbol": blocks_by_symbol,
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

    def _finalize_day(self, day_key: str, daily_dir: Path, run_id: str) -> Dict[str, Any] | None:
        reports = self.slot_reports_by_day.pop(day_key, [])
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
            "top_triggers": top_triggers,
            "skipped_due_to_limits": skipped_limits,
            "attempted_triggers": limit_entry["trades_attempted"],
            "executed_trades": limit_entry["trades_executed"],
            "flatten_positions_daily": self.flatten_positions_daily,
            "flatten_session_hour": self.flatten_session_boundary_hour,
            "flatten_policy": self.flatten_policy,
            "pnl_breakdown": pnl_breakdown,
            "symbol_pnl": symbol_pnl,
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
        risk_budget_info = self._risk_budget_summary(day_key)
        if risk_budget_info:
            summary["risk_budget"] = risk_budget_info
            limit_stats["risk_budget_used_pct"] = risk_budget_info["used_pct"]
            limit_stats["risk_budget_usage_by_symbol"] = risk_budget_info.get("symbol_usage_pct", {})
            limit_stats["risk_budget_blocks_by_symbol"] = risk_budget_info.get("blocks_by_symbol", {})
        if plan_limits:
            summary["plan_limits"] = plan_limits
            min_trades = plan_limits.get("min_trades_per_day") or 0
            if min_trades and summary["executed_trades"] < min_trades:
                summary["missed_min_trades"] = True
        raw_feedback = self._judge_feedback(summary)
        summary["judge_feedback"] = raw_feedback
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

        # attach risk usage totals
        for (trigger_id, timeframe), risk_used in risk_usage.items():
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
            payload.setdefault("latencies", [])

        timeframe_risk_usage: Dict[str, float] = defaultdict(float)
        hour_risk_usage: Dict[str, float] = defaultdict(float)
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
            if hour_val is not None:
                hour_risk_usage[str(hour_val)] += max(risk_used, 0.0)

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
                finalized[key] = {
                    "trades": trades,
                    "pnl": pnl,
                    "risk_used_abs": risk_used,
                    "rpr": (pnl / risk_used) if risk_used else 0.0,
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
        }

    def _judge_feedback(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        score = 50.0
        return_pct = summary.get("return_pct", 0.0)
        score += max(min(return_pct, 5.0), -5.0) * 4.0
        trade_count = summary.get("trade_count", 0)
        notes = []
        machine_constraints = JudgeConstraints().model_dump()
        if trade_count == 0:
            notes.append("No trades executed; confirm trigger sensitivity.")
            score -= 5
        elif trade_count > 12:
            notes.append("Elevated trade count; monitor over-trading risk.")
            score -= 3
        else:
            notes.append("Trade cadence within expected bounds.")
        if return_pct > 0.5:
            notes.append("Positive daily performance.")
        elif return_pct < -0.5:
            notes.append("Drawdown detected; tighten stops.")
        constraints = {
            "must_fix": [],
            "vetoes": [],
            "boost": [],
            "regime_correction": "Maintain discipline until equity stabilizes.",
            "sizing_adjustments": {},
        }
        if trade_count == 0:
            constraints["must_fix"].append("Increase selectivity but avoid paralysis; at least one qualified trigger per day.")
        if trade_count > 12:
            constraints["vetoes"].append("Disable redundant scalp triggers bleeding cost.")
            constraints["must_fix"].append("Cap total trades at 10 until judge score > 65.")
        if return_pct > 0.5:
            constraints["boost"].append("Favor trend_continuation setups when volatility is orderly.")
            constraints["regime_correction"] = "Lean pro-trend but protect gains with trailing exits."
        if return_pct < -0.5:
            constraints["must_fix"].append("Tighten exits after 0.8% adverse move.")
            constraints["vetoes"].append("Pause volatility_breakout triggers during drawdown.")
            constraints["regime_correction"] = "Assume mixed regime until positive close recorded."
        positions = summary.get("positions_end", {})
        for symbol in positions.keys():
            if return_pct < 0:
                constraints["sizing_adjustments"][symbol] = "Cut risk by 25% until two winning days post drawdown."
            elif return_pct > 0.5:
                constraints["sizing_adjustments"][symbol] = "Allow full allocation for grade A triggers only."
        limit_stats = summary.get("limit_stats") or {}
        attempted = summary.get("attempted_triggers", 0)
        executed = summary.get("executed_trades", 0)
        trigger_budget = None
        if (attempted - executed) >= 10 or limit_stats.get("blocked_by_daily_cap", 0) >= 5:
            trigger_budget = 6
            constraints["must_fix"].append("Limit strategist to <=6 high-conviction triggers per symbol until cadence improves.")
        pnl_breakdown = summary.get("pnl_breakdown") or {}
        if pnl_breakdown:
            gross_pct = pnl_breakdown.get("gross_trade_pct", 0.0)
            fees_pct = pnl_breakdown.get("fees_pct", 0.0)
            flatten_pct = pnl_breakdown.get("flattening_pct", 0.0)
            if gross_pct > 0 and gross_pct + fees_pct + flatten_pct < 0:
                notes.append("Signals profitable pre-costs but fees/flattening erase gains.")
                constraints["must_fix"].append("Reduce churn; disable scalp triggers that only generate fee drag.")
        trigger_stats = summary.get("trigger_stats") or {}
        noisy_triggers = [
            trigger_id
            for trigger_id, stats in trigger_stats.items()
            if stats.get("executed", 0) == 0 and stats.get("blocked", 0) >= 5
        ]
        if noisy_triggers:
            constraints["vetoes"].append(f"Disable noisy triggers: {', '.join(noisy_triggers)}")
            disabled = set(machine_constraints.get("disabled_trigger_ids") or [])
            disabled.update(noisy_triggers)
            machine_constraints["disabled_trigger_ids"] = list(disabled)
        risk_budget = summary.get("risk_budget")
        if risk_budget:
            used_pct = risk_budget.get("used_pct", 0.0)
            utilization_pct = risk_budget.get("utilization_pct", used_pct)
            if utilization_pct < 25 and return_pct > 0:
                notes.append("Risk budget underutilized despite gains; lean into high-conviction setups.")
                constraints["boost"].append("Increase size on grade A triggers until at least 30% of daily risk is deployed.")
            elif utilization_pct > 75 and return_pct < 0:
                notes.append("Risk budget heavily used on a losing day; tighten selectivity.")
                constraints["must_fix"].append("Stop firing marginal triggers once 75% of daily risk is consumed.")
                if trigger_budget is None:
                    trigger_budget = 6
                else:
                    trigger_budget = min(trigger_budget, 6)
            for symbol, pct in (risk_budget.get("symbol_usage_pct") or {}).items():
                if pct >= 70:
                    constraints["sizing_adjustments"][symbol] = "Cut risk by 25% until per-symbol risk share normalizes."
        machine_constraints["max_triggers_per_symbol_per_day"] = trigger_budget
        return {
            "score": max(0.0, min(100.0, round(score, 1))),
            "notes": " ".join(notes),
            "constraints": machine_constraints,
            "strategist_constraints": constraints,
        }

    def _apply_feedback_adjustments(self, run_id: str, feedback: JudgeFeedback, winning_day: bool):
        run = self.run_registry.get_strategy_run(run_id)
        apply_judge_risk_feedback(run, feedback, winning_day)
        updated = self.run_registry.update_strategy_run(run)
        self._refresh_risk_state_from_run(updated)
        return updated
