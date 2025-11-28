"""Backtesting harness that wires the LLM strategist into the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from agents.analytics import IndicatorWindowConfig, PortfolioHistory, build_asset_state, compute_indicator_snapshot, compute_portfolio_state
from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from agents.strategies.risk_engine import RiskEngine
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


class LLMStrategistBacktester:
    def __init__(
        self,
        pairs: Sequence[str],
        start: datetime,
        end: datetime,
        initial_cash: float,
        fee_rate: float,
        llm_client: LLMClient,
        cache_dir: Path,
        llm_calls_per_day: int,
        risk_params: Dict[str, Any],
        timeframes: Sequence[str] = ("1h", "4h", "1d"),
        prompt_template_path: Path | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must be provided")
        self.pairs = list(pairs)
        self.start = start
        self.end = end
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.timeframes = list(timeframes)
        self.market_data = self._load_data()
        self.portfolio = PortfolioTracker(initial_cash=initial_cash, fee_rate=fee_rate)
        self.plan_provider = StrategyPlanProvider(llm_client, cache_dir=cache_dir, llm_calls_per_day=llm_calls_per_day)
        self.window_configs = {tf: IndicatorWindowConfig(timeframe=tf) for tf in self.timeframes}
        self.risk_params = risk_params
        self.prompt_template = prompt_template_path.read_text() if prompt_template_path and prompt_template_path.exists() else None

    def _load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for pair in self.pairs:
            tf_map: Dict[str, pd.DataFrame] = {}
            for timeframe in self.timeframes:
                df = load_ohlcv(pair, self.start, self.end, timeframe=timeframe)
                tf_map[timeframe] = _ensure_timestamp_index(df)
            data[pair] = tf_map
        return data

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
        return LLMInput(
            portfolio=portfolio_state,
            assets=list(asset_states.values()),
            risk_params=self.risk_params,
            global_context={
                "timestamp": timestamp.isoformat(),
                "pairs": self.pairs,
            },
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

        active_assets: Dict[str, AssetState] = {}
        for ts in all_timestamps:
            new_day = current_plan is None or ts >= current_plan.valid_until
            if new_day:
                asset_states = self._asset_states(ts)
                if not asset_states:
                    continue
                active_assets = asset_states
                llm_input = self._llm_input(ts, asset_states)
                current_plan = self.plan_provider.get_plan(run_id, ts, llm_input, prompt_template=self.prompt_template)
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

            for pair in self.pairs:
                for timeframe in self.timeframes:
                    df = self.market_data[pair][timeframe]
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
                    if timeframe == self.timeframes[0]:
                        latest_prices[pair] = bar.close
            self.portfolio.mark_to_market(ts, latest_prices)

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
        )
