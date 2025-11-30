"""Backtesting harness that wires the LLM strategist into the simulator."""

from __future__ import annotations

from collections import defaultdict
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from agents.analytics import IndicatorWindowConfig, PortfolioHistory, build_asset_state, compute_indicator_snapshot, compute_portfolio_state
from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from agents.strategies.risk_engine import RiskEngine
from agents.strategies.strategy_memory import build_strategy_memory
from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
from backtesting.dataset import load_ohlcv
from schemas.llm_strategist import AssetState, LLMInput, PortfolioState, StrategyPlan


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
    equity_records: List[Dict[str, Any]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def _record_pnl(self, symbol: str, pnl: float, timestamp: datetime) -> None:
        self.trade_log.append({"timestamp": timestamp, "symbol": symbol, "pnl": pnl})

    def _update_position(self, symbol: str, delta_qty: float, price: float, timestamp: datetime) -> None:
        position = self.positions.get(symbol, 0.0)
        avg_price = self.avg_entry_price.get(symbol, price)
        new_position = position + delta_qty
        if position == 0 or (position > 0 and new_position > 0) or (position < 0 and new_position < 0):
            if new_position == 0:
                self.positions.pop(symbol, None)
                self.avg_entry_price.pop(symbol, None)
            else:
                total_notional = abs(position) * avg_price + abs(delta_qty) * price
                self.positions[symbol] = new_position
                self.avg_entry_price[symbol] = total_notional / abs(new_position)
            return
        closing_qty = min(abs(position), abs(delta_qty))
        if closing_qty > 0:
            pnl = closing_qty * ((price - avg_price) if position > 0 else (avg_price - price))
            self._record_pnl(symbol, pnl, timestamp)
        remaining = position + delta_qty
        if abs(remaining) <= 1e-9:
            self.positions.pop(symbol, None)
            self.avg_entry_price.pop(symbol, None)
        else:
            self.positions[symbol] = remaining
            # direction flipped; reset basis to execution price
            self.avg_entry_price[symbol] = price

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
        self._update_position(order.symbol, delta, order.price, order.timestamp)
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
        risk_params: Dict[str, Any],
        timeframes: Sequence[str] = ("1h", "4h", "1d"),
        prompt_template_path: Path | None = None,
        plan_provider: StrategyPlanProvider | None = None,
        market_data: Dict[str, Dict[str, pd.DataFrame]] | None = None,
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
        self.window_configs = {tf: IndicatorWindowConfig(timeframe=tf) for tf in self.timeframes}
        self.risk_params = risk_params
        self.prompt_template = prompt_template_path.read_text() if prompt_template_path and prompt_template_path.exists() else None
        self.slot_reports_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trigger_activity_by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.current_day_key: str | None = None
        self.latest_daily_summary: Dict[str, Any] | None = None
        self.last_slot_report: Dict[str, Any] | None = None
        self.memory_history: List[Dict[str, Any]] = []
        self.judge_constraints: Dict[str, Any] = {}
        self.trigger_activity_by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.current_day_key: str | None = None
        self.latest_daily_summary: Dict[str, Any] | None = None
        self.last_slot_report: Dict[str, Any] | None = None
        self.slot_reports_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_day_key: str | None = None
        self.latest_daily_summary: Dict[str, Any] | None = None
        self.last_slot_report: Dict[str, Any] | None = None

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
        return LLMInput(
            portfolio=portfolio_state,
            assets=list(asset_states.values()),
            risk_params=self.risk_params,
            global_context=context,
        )

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

    def run(self, run_id: str) -> StrategistBacktestResult:
        all_timestamps = sorted(
            {ts for pair in self.market_data.values() for df in pair.values() for ts in df.index if self.start <= ts <= self.end}
        )
        current_plan: StrategyPlan | None = None
        trigger_engine: TriggerEngine | None = None
        plan_log: List[Dict[str, Any]] = []
        latest_prices: Dict[str, float] = {symbol: 0.0 for symbol in self.pairs}
        daily_reports: List[Dict[str, Any]] = []

        active_assets: Dict[str, AssetState] = {}
        cache_base = Path(self.plan_provider.cache_dir) / run_id
        daily_dir = cache_base / "daily_reports"
        daily_dir.mkdir(parents=True, exist_ok=True)
        self.slot_reports_by_day.clear()
        self.trigger_activity_by_day.clear()
        self.current_day_key = None
        self.latest_daily_summary = None
        self.last_slot_report = None

        for ts in all_timestamps:
            new_day = current_plan is None or ts >= current_plan.valid_until
            if new_day:
                day_key = ts.date().isoformat()
                if self.current_day_key and self.current_day_key != day_key:
                    summary = self._finalize_day(self.current_day_key, daily_dir)
                    if summary:
                        self.latest_daily_summary = summary
                        self.memory_history.append(summary)
                        judge_constraints = summary.get("judge_feedback", {}).get("strategist_constraints")
                        if judge_constraints:
                            self.judge_constraints = judge_constraints
                        daily_reports.append(summary)
                self.current_day_key = day_key
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
                current_plan = self.plan_provider.get_plan(run_id, slot_start, llm_input, prompt_template=self.prompt_template)
                current_plan = current_plan.model_copy(
                    update={
                        "generated_at": slot_start,
                        "valid_until": slot_end,
                    }
                )
                risk_engine = RiskEngine(
                    current_plan.risk_constraints,
                    {rule.symbol: rule for rule in current_plan.sizing_rules},
                    daily_anchor_equity=self.portfolio.portfolio_state(ts).equity,
                )
                trigger_engine = TriggerEngine(current_plan, risk_engine)
                plan_log.append(
                    {
                        "generated_at": current_plan.generated_at.isoformat(),
                        "valid_until": current_plan.valid_until.isoformat(),
                        "regime": current_plan.regime,
                        "num_triggers": len(current_plan.triggers),
                    }
                )
            if trigger_engine is None or current_plan is None:
                continue

            slot_orders: List[Dict[str, Any]] = []
            indicator_briefs = self._indicator_briefs(active_assets)
            for pair in self.pairs:
                for timeframe in self.timeframes:
                    df = self.market_data.get(pair, {}).get(timeframe)
                    if df is None:
                        continue
                    if ts not in df.index:
                        continue
                    bar = self._build_bar(pair, timeframe, ts)
                    subset = df[df.index <= ts]
                    indicator = compute_indicator_snapshot(subset, symbol=pair, timeframe=timeframe, config=self.window_configs[timeframe])
                    portfolio_state = self.portfolio.portfolio_state(ts)
                    asset_state = active_assets.get(pair)
                    orders = trigger_engine.on_bar(bar, indicator, portfolio_state, asset_state)
                    for order in orders:
                        self.portfolio.execute(order)
                        slot_orders.append(
                            {
                                "symbol": order.symbol,
                                "side": order.side,
                                "quantity": order.quantity,
                                "price": order.price,
                                "timeframe": order.timeframe,
                                "reason": order.reason,
                            }
                        )
                        self.trigger_activity_by_day[day_key][order.reason] += 1
                    if timeframe == self.timeframes[0]:
                        latest_prices[pair] = bar.close
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
            day_key = ts.date().isoformat()
            self.slot_reports_by_day[day_key].append(slot_report)
            self.last_slot_report = slot_report

        if self.current_day_key:
            summary = self._finalize_day(self.current_day_key, daily_dir)
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
        summary = {
            "final_equity": float(equity_curve.iloc[-1]),
            "return_pct": (float(equity_curve.iloc[-1]) / self.initial_cash - 1) * 100 if self.initial_cash else 0.0,
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

    def _finalize_day(self, day_key: str, daily_dir: Path) -> Dict[str, Any] | None:
        reports = self.slot_reports_by_day.pop(day_key, [])
        if not reports:
            return None
        start_equity = reports[0]["equity"]
        end_equity = reports[-1]["equity"]
        day_return = ((end_equity - start_equity) / start_equity * 100.0) if start_equity else 0.0
        trade_count = sum(len(report.get("orders", [])) for report in reports)
        indicator_context = reports[-1].get("indicator_context", {})
        triggers = self.trigger_activity_by_day.pop(day_key, {})
        top_triggers = sorted(triggers.items(), key=lambda kv: kv[1], reverse=True)[:3]
        summary = {
            "date": day_key,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "return_pct": day_return,
            "trade_count": trade_count,
            "positions_end": reports[-1]["positions"],
            "indicator_context": indicator_context,
            "top_triggers": top_triggers,
        }
        summary["judge_feedback"] = self._judge_feedback(summary)
        (daily_dir / f"{day_key}.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _judge_feedback(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        score = 50.0
        return_pct = summary.get("return_pct", 0.0)
        score += max(min(return_pct, 5.0), -5.0) * 4.0
        trade_count = summary.get("trade_count", 0)
        notes = []
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
        return {
            "score": max(0.0, min(100.0, round(score, 1))),
            "notes": " ".join(notes),
            "strategist_constraints": constraints,
        }
