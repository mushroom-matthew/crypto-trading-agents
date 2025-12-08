"""Trigger evaluation engine that turns LLM plans into deterministic orders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Literal

from schemas.llm_strategist import AssetState, IndicatorSnapshot, PortfolioState, StrategyPlan, TriggerCondition
from trading_core.execution_engine import BlockReason

from .risk_engine import RiskEngine
from .rule_dsl import MissingIndicatorError, RuleEvaluator, RuleSyntaxError
from .trade_risk import TradeRiskEvaluator


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

    def __init__(
        self,
        plan: StrategyPlan,
        risk_engine: RiskEngine,
        evaluator: RuleEvaluator | None = None,
        trade_risk: TradeRiskEvaluator | None = None,
    ) -> None:
        self.plan = plan
        self.risk_engine = risk_engine
        self.evaluator = evaluator or RuleEvaluator()
        self.trade_risk = trade_risk or TradeRiskEvaluator(risk_engine)

    def _context(
        self,
        indicator: IndicatorSnapshot,
        asset_state: AssetState | None,
        market_structure: dict[str, float | str | None] | None = None,
    ) -> dict[str, float | str | None]:
        """Build evaluation context, including cross-timeframe aliases."""

        def _alias_key(key: str) -> str | None:
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return None

        context = indicator.model_dump()
        for key, value in list(context.items()):
            alias = _alias_key(key)
            if alias and alias not in context:
                context[alias] = value
        upper = context.get("bollinger_upper")
        lower = context.get("bollinger_lower")
        if upper is not None and lower is not None:
            context["bollinger_middle"] = (upper + lower) / 2.0
        if asset_state:
            context["trend_state"] = asset_state.trend_state
            context["vol_state"] = asset_state.vol_state
            for snapshot in asset_state.indicators:
                prefix = f"tf_{snapshot.timeframe.replace('-', '_')}"
                snapshot_dict = snapshot.model_dump()
                for key, value in snapshot_dict.items():
                    if key in {"symbol", "timeframe", "as_of"}:
                        continue
                    if key == "bollinger_upper":
                        other = snapshot_dict.get("bollinger_lower")
                        if other is not None:
                            context[f"{prefix}_bollinger_middle"] = (value + other) / 2.0
                    context[f"{prefix}_{key}"] = value
                    alias = _alias_key(key)
                    if alias:
                        context[f"{prefix}_{alias}"] = value
        ms_keys = ("nearest_support", "nearest_resistance", "distance_to_support_pct", "distance_to_resistance_pct", "trend")
        if market_structure:
            # Surface market-structure telemetry fields directly for rule expressions.
            for key in ms_keys:
                if key in market_structure:
                    context[key] = market_structure.get(key)
        # Ensure keys exist to avoid unknown identifier errors when no snapshot is available.
        for key in ms_keys:
            context.setdefault(key, None)
        return context

    def _position_direction(self, symbol: str, portfolio: PortfolioState) -> Literal["long", "short", "flat"]:
        qty = portfolio.positions.get(symbol, 0.0)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return "flat"

    def _flatten_order(
        self,
        trigger: TriggerCondition,
        bar: Bar,
        portfolio: PortfolioState,
        reason: str,
        block_entries: List[dict] | None = None,
    ) -> Order | None:
        qty = portfolio.positions.get(trigger.symbol, 0.0)
        if abs(qty) <= 1e-9:
            return None
        decision = self.trade_risk.evaluate(trigger, "flatten", bar.close, portfolio, None)
        if not decision.allowed and block_entries is not None and decision.reason:
            self._record_block(block_entries, trigger, decision.reason, "Flatten blocked unexpectedly", bar)
            return None
        side: Literal["buy", "sell"] = "sell" if qty > 0 else "buy"
        return Order(symbol=trigger.symbol, side=side, quantity=abs(qty), price=bar.close, timeframe=bar.timeframe, reason=reason, timestamp=bar.timestamp)

    def _record_block(
        self,
        container: List[dict],
        trigger: TriggerCondition,
        reason: str,
        detail: str,
        bar: Bar,
    ) -> None:
        container.append(
            {
                "trigger_id": trigger.id,
                "symbol": trigger.symbol,
                "timeframe": bar.timeframe,
                "timestamp": bar.timestamp.isoformat(),
                "reason": reason,
                "detail": detail,
                "price": bar.close,
            }
        )

    def _entry_order(
        self,
        trigger: TriggerCondition,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        bar: Bar,
        block_entries: List[dict] | None = None,
    ) -> Order | None:
        desired = trigger.direction
        current = self._position_direction(trigger.symbol, portfolio)
        if desired in {"flat", "exit", "flat_exit"}:
            if current == "flat":
                return None
            return self._flatten_order(trigger, bar, portfolio, f"{trigger.id}_flat", block_entries)
        if desired == current:
            return None
        check = self.trade_risk.evaluate(trigger, "entry", bar.close, portfolio, indicator)
        if not check.allowed:
            if block_entries is not None and check.reason:
                detail = f"Risk constraint {check.reason} prevented sizing"
                self._record_block(block_entries, trigger, check.reason, detail, bar)
            return None
        side: Literal["buy", "sell"] = "buy" if desired == "long" else "sell"
        return Order(
            symbol=trigger.symbol,
            side=side,
            quantity=check.quantity,
            price=bar.close,
            timeframe=bar.timeframe,
            reason=trigger.id,
            timestamp=bar.timestamp,
        )

    def on_bar(
        self,
        bar: Bar,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        asset_state: AssetState | None = None,
        market_structure: dict[str, float | str | None] | None = None,
    ) -> tuple[List[Order], List[dict]]:
        orders: List[Order] = []
        block_entries: List[dict] = []
        context = self._context(indicator, asset_state, market_structure)
        for trigger in self.plan.triggers:
            if trigger.symbol != bar.symbol or trigger.timeframe != bar.timeframe:
                continue
            try:
                exit_fired = bool(trigger.exit_rule and self.evaluator.evaluate(trigger.exit_rule, context))
            except MissingIndicatorError as exc:
                detail = f"{exc}; exit_rule='{trigger.exit_rule}'"
                self._record_block(block_entries, trigger, BlockReason.MISSING_INDICATOR.value, detail, bar)
                continue
            except RuleSyntaxError as exc:
                detail = f"{exc}; exit_rule='{trigger.exit_rule}'"
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar)
                continue
            if exit_fired:
                exit_order = self._flatten_order(trigger, bar, portfolio, f"{trigger.id}_exit", block_entries)
                if exit_order:
                    orders.append(exit_order)
                    continue
            try:
                entry_fired = bool(trigger.entry_rule and self.evaluator.evaluate(trigger.entry_rule, context))
            except MissingIndicatorError as exc:
                detail = f"{exc}; entry_rule='{trigger.entry_rule}'"
                self._record_block(block_entries, trigger, BlockReason.MISSING_INDICATOR.value, detail, bar)
                continue
            except RuleSyntaxError as exc:
                detail = f"{exc}; entry_rule='{trigger.entry_rule}'"
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar)
                continue
            if entry_fired:
                entry = self._entry_order(trigger, indicator, portfolio, bar, block_entries)
                if entry:
                    orders.append(entry)
        return orders, block_entries
