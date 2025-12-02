"""Deterministic execution engine shared between backtests and live runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Literal, Optional

from schemas.compiled_plan import CompiledPlan, CompiledTrigger
from schemas.judge_feedback import JudgeConstraints
from schemas.llm_strategist import StrategyPlan
from schemas.strategy_run import StrategyRun


ExecutionAction = Literal["executed", "skipped"]


class BlockReason(str, Enum):
    DAILY_CAP = "daily_cap"
    SYMBOL_VETO = "symbol_veto"
    RISK = "risk"
    DIRECTION = "direction"
    CATEGORY = "category"
    PLAN_LIMIT = "plan_limit"
    EXPRESSION_ERROR = "expression_error"
    MISSING_INDICATOR = "missing_indicator"
    OTHER = "other"


@dataclass
class TradeEvent:
    timestamp: datetime
    trigger_id: str
    symbol: str
    action: ExecutionAction
    reason: str


@dataclass
class DailyExecutionState:
    trades_today: int = 0
    symbol_trades: Dict[str, int] = field(default_factory=dict)
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def log_skip(self, reason: str) -> None:
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def record_trade(self, symbol: str) -> None:
        self.symbol_trades[symbol] = self.symbol_trades.get(symbol, 0) + 1


class ExecutionEngine:
    """Enforces judge/plan limits and computes trade eligibility."""

    def __init__(self) -> None:
        self.daily_state: Dict[str, DailyExecutionState] = {}
        self.plan_trade_counts: Dict[tuple[str, str, str], int] = {}

    def _day_key(self, run_id: str, bar_time: datetime) -> str:
        day = bar_time.date().isoformat()
        return f"{run_id}:{day}"

    def _get_state(self, run_id: str, bar_time: datetime) -> DailyExecutionState:
        key = self._day_key(run_id, bar_time)
        if key not in self.daily_state:
            self.daily_state[key] = DailyExecutionState()
        return self.daily_state[key]

    def reset_run(self, run_id: str) -> None:
        prefix = f"{run_id}:"
        self.daily_state = {k: v for k, v in self.daily_state.items() if not k.startswith(prefix)}
        self.plan_trade_counts = {k: v for k, v in self.plan_trade_counts.items() if k[0] != run_id}

    def plan_trades(self, run_id: str, plan_id: str, day: str | None = None) -> int:
        if day:
            return self.plan_trade_counts.get((run_id, plan_id, day), 0)
        return sum(count for (r, p, _), count in self.plan_trade_counts.items() if r == run_id and p == plan_id)

    def _plan_key(self, run_id: str, plan_id: str, bar_time: datetime) -> tuple[str, str, str]:
        return (run_id, plan_id, bar_time.date().isoformat())

    @dataclass(frozen=True)
    class PlanLimits:
        max_trades_per_day: int
        min_trades_per_day: Optional[int]
        allowed_symbols: set[str]
        allowed_directions: set[str]
        allowed_categories: set[str]

    def _build_limits(
        self,
        run: StrategyRun,
        strategy_plan: StrategyPlan,
        constraints: JudgeConstraints | None,
    ) -> "ExecutionEngine.PlanLimits":
        judge_limit = constraints.max_trades_per_day if constraints else None
        plan_limit = strategy_plan.max_trades_per_day
        max_trades = plan_limit if judge_limit is None else (plan_limit if plan_limit is not None else judge_limit)
        if judge_limit is not None and plan_limit is not None:
            max_trades = min(judge_limit, plan_limit)
        max_trades = max_trades if max_trades is not None else 0
        min_trades = constraints.min_trades_per_day if constraints else None
        if strategy_plan.min_trades_per_day is not None:
            min_trades = max(min_trades or 0, strategy_plan.min_trades_per_day)
        allowed_symbols = set(strategy_plan.allowed_symbols or run.config.symbols)
        allowed_directions = set(strategy_plan.allowed_directions or ["long", "short"])
        allowed_categories = set(strategy_plan.allowed_trigger_categories or ["other"])
        return ExecutionEngine.PlanLimits(
            max_trades_per_day=max_trades,
            min_trades_per_day=min_trades,
            allowed_symbols=allowed_symbols,
            allowed_directions=allowed_directions,
            allowed_categories=allowed_categories,
        )

    def _validate_plan_limits(self, strategy_plan: StrategyPlan) -> None:
        if strategy_plan.max_trades_per_day is None:
            raise ValueError("StrategyPlan missing max_trades_per_day")
        if not strategy_plan.allowed_symbols:
            raise ValueError("StrategyPlan missing allowed_symbols")
        if not strategy_plan.allowed_directions:
            raise ValueError("StrategyPlan missing allowed_directions")
        if not strategy_plan.allowed_trigger_categories:
            raise ValueError("StrategyPlan missing allowed_trigger_categories")

    def evaluate_trigger(
        self,
        run: StrategyRun,
        compiled_plan: CompiledPlan,
        strategy_plan: StrategyPlan,
        constraints: JudgeConstraints | None,
        trigger: CompiledTrigger,
        bar_timestamp: datetime,
    ) -> TradeEvent | None:
        self._validate_plan_limits(strategy_plan)
        state = self._get_state(run.run_id, bar_timestamp)
        limits = self._build_limits(run, strategy_plan, constraints)

        if limits.max_trades_per_day and state.trades_today >= limits.max_trades_per_day:
            state.log_skip(BlockReason.DAILY_CAP.value)
            return TradeEvent(
                timestamp=bar_timestamp,
                trigger_id=trigger.trigger_id,
                symbol=trigger.symbol,
                action="skipped",
                reason=BlockReason.DAILY_CAP.value,
            )

        if constraints:
            if trigger.trigger_id in constraints.disabled_trigger_ids:
                state.log_skip(BlockReason.SYMBOL_VETO.value)
                return TradeEvent(
                    timestamp=bar_timestamp,
                    trigger_id=trigger.trigger_id,
                    symbol=trigger.symbol,
                    action="skipped",
                    reason=BlockReason.SYMBOL_VETO.value,
                )
            if trigger.category and trigger.category in constraints.disabled_categories:
                state.log_skip(BlockReason.CATEGORY.value)
                return TradeEvent(
                    timestamp=bar_timestamp,
                    trigger_id=trigger.trigger_id,
                    symbol=trigger.symbol,
                    action="skipped",
                    reason=BlockReason.CATEGORY.value,
                )

        if limits.allowed_symbols and trigger.symbol not in limits.allowed_symbols:
            state.log_skip(BlockReason.SYMBOL_VETO.value)
            return TradeEvent(
                timestamp=bar_timestamp,
                trigger_id=trigger.trigger_id,
                symbol=trigger.symbol,
                action="skipped",
                reason=BlockReason.SYMBOL_VETO.value,
            )

        if limits.allowed_directions and trigger.direction not in limits.allowed_directions:
            state.log_skip(BlockReason.DIRECTION.value)
            return TradeEvent(
                timestamp=bar_timestamp,
                trigger_id=trigger.trigger_id,
                symbol=trigger.symbol,
                action="skipped",
                reason=BlockReason.DIRECTION.value,
            )
        category = trigger.category or "other"
        if limits.allowed_categories and category not in limits.allowed_categories:
            state.log_skip(BlockReason.CATEGORY.value)
            return TradeEvent(
                timestamp=bar_timestamp,
                trigger_id=trigger.trigger_id,
                symbol=trigger.symbol,
                action="skipped",
                reason=BlockReason.CATEGORY.value,
            )

        state.trades_today += 1
        state.record_trade(trigger.symbol)
        key = self._plan_key(run.run_id, compiled_plan.plan_id, bar_timestamp)
        self.plan_trade_counts[key] = self.plan_trade_counts.get(key, 0) + 1
        return TradeEvent(
            timestamp=bar_timestamp,
            trigger_id=trigger.trigger_id,
            symbol=trigger.symbol,
            action="executed",
            reason="",
        )
