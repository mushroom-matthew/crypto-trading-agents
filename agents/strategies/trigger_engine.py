"""Trigger evaluation engine that turns LLM plans into deterministic orders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Literal

from schemas.llm_strategist import AssetState, IndicatorSnapshot, PortfolioState, StrategyPlan, TriggerCondition

from .risk_engine import RiskEngine
from .rule_dsl import RuleEvaluator


@dataclass(frozen=True)
class Bar:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    timeframe: str
    reason: str
    timestamp: datetime


class TriggerEngine:
    """Evaluates TriggerCondition strings on every bar walking forward."""

    def __init__(self, plan: StrategyPlan, risk_engine: RiskEngine, evaluator: RuleEvaluator | None = None) -> None:
        self.plan = plan
        self.risk_engine = risk_engine
        self.evaluator = evaluator or RuleEvaluator()

    def _context(self, indicator: IndicatorSnapshot, asset_state: AssetState | None) -> dict[str, float | str | None]:
        context = indicator.model_dump()
        if asset_state:
            context["trend_state"] = asset_state.trend_state
            context["vol_state"] = asset_state.vol_state
        return context

    def _position_direction(self, symbol: str, portfolio: PortfolioState) -> Literal["long", "short", "flat"]:
        qty = portfolio.positions.get(symbol, 0.0)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return "flat"

    def _flatten_order(self, symbol: str, price: float, timeframe: str, portfolio: PortfolioState, reason: str, timestamp: datetime) -> Order | None:
        qty = portfolio.positions.get(symbol, 0.0)
        if abs(qty) <= 1e-9:
            return None
        side: Literal["buy", "sell"] = "sell" if qty > 0 else "buy"
        return Order(symbol=symbol, side=side, quantity=abs(qty), price=price, timeframe=timeframe, reason=reason, timestamp=timestamp)

    def _entry_order(
        self,
        trigger: TriggerCondition,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        bar: Bar,
    ) -> Order | None:
        desired = trigger.direction
        current = self._position_direction(trigger.symbol, portfolio)
        if desired == "flat":
            if current == "flat":
                return None
            return self._flatten_order(trigger.symbol, bar.close, bar.timeframe, portfolio, f"{trigger.id}_flat", bar.timestamp)
        if desired == current:
            return None
        qty = self.risk_engine.size_position(trigger.symbol, bar.close, portfolio, indicator)
        if qty <= 0:
            return None
        side: Literal["buy", "sell"] = "buy" if desired == "long" else "sell"
        return Order(symbol=trigger.symbol, side=side, quantity=qty, price=bar.close, timeframe=bar.timeframe, reason=trigger.id, timestamp=bar.timestamp)

    def on_bar(
        self,
        bar: Bar,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        asset_state: AssetState | None = None,
    ) -> List[Order]:
        orders: List[Order] = []
        context = self._context(indicator, asset_state)
        for trigger in self.plan.triggers:
            if trigger.symbol != bar.symbol or trigger.timeframe != bar.timeframe:
                continue
            if trigger.exit_rule and self.evaluator.evaluate(trigger.exit_rule, context):
                exit_order = self._flatten_order(trigger.symbol, bar.close, bar.timeframe, portfolio, f"{trigger.id}_exit", bar.timestamp)
                if exit_order:
                    orders.append(exit_order)
                    continue
            if trigger.entry_rule and self.evaluator.evaluate(trigger.entry_rule, context):
                entry = self._entry_order(trigger, indicator, portfolio, bar)
                if entry:
                    orders.append(entry)
        return orders
