"""Strategy plan generation service aware of StrategyRun registry."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.judge_feedback import DisplayConstraints, JudgeFeedback
from schemas.llm_strategist import LLMInput, StrategyPlan
from services.strategy_run_registry import StrategyRunRegistry, strategy_run_registry


def _default_plan_provider() -> StrategyPlanProvider:
    cache_dir = Path(os.environ.get("STRATEGY_PLAN_CACHE_DIR", "data/strategy_plan_cache"))
    llm_calls_per_day = int(os.environ.get("STRATEGY_PLAN_LLM_CALLS_PER_DAY", "1"))
    return StrategyPlanProvider(LLMClient(), cache_dir=cache_dir, llm_calls_per_day=llm_calls_per_day)


class StrategistPlanService:
    """Generates StrategyPlan instances for a StrategyRun and enforces constraints."""

    def __init__(
        self,
        plan_provider: StrategyPlanProvider | None = None,
        registry: StrategyRunRegistry | None = None,
    ) -> None:
        self.plan_provider = plan_provider or _default_plan_provider()
        self.registry = registry or strategy_run_registry
        self.default_max_trades = int(os.environ.get("STRATEGIST_PLAN_DEFAULT_MAX_TRADES", "10"))
        self.min_trade_hint_cap = int(os.environ.get("STRATEGIST_PLAN_MIN_TRADE_CAP", "3"))

    def generate_plan_for_run(
        self,
        run_id: str,
        llm_input: LLMInput,
        plan_date: Optional[datetime] = None,
        prompt_template: str | None = None,
    ) -> StrategyPlan:
        run = self.registry.get_strategy_run(run_id)
        plan_date = plan_date or datetime.now(timezone.utc)
        plan = self.plan_provider.get_plan(run_id, plan_date, llm_input, prompt_template=prompt_template)
        plan = plan.model_copy(deep=True)
        plan.run_id = run.run_id
        combined_symbols = sorted({*(run.config.symbols or []), *(asset.symbol for asset in llm_input.assets)})
        if not plan.allowed_symbols:
            plan.allowed_symbols = combined_symbols
        if not plan.allowed_directions:
            plan.allowed_directions = ["long", "short"]
        else:
            plan.allowed_directions = sorted({direction for direction in plan.allowed_directions if direction})
        if not plan.allowed_trigger_categories:
            plan.allowed_trigger_categories = [
                "trend_continuation",
                "mean_reversion",
                "volatility_breakout",
                "reversal",
                "emergency_exit",
                "other",
            ]
        else:
            plan.allowed_trigger_categories = sorted({(category or "other") for category in plan.allowed_trigger_categories})
        if any(trigger.category is None for trigger in plan.triggers):
            if "other" not in plan.allowed_trigger_categories:
                plan.allowed_trigger_categories.append("other")
        if run.latest_judge_feedback:
            plan = self._apply_strategist_constraints(plan, run.latest_judge_feedback.strategist_constraints)
        plan.max_trades_per_day = self._resolve_max_trades(plan.max_trades_per_day, run.latest_judge_feedback)
        constraint_min = run.latest_judge_feedback.constraints.min_trades_per_day if run.latest_judge_feedback else None
        if constraint_min is not None:
            plan.min_trades_per_day = max(plan.min_trades_per_day or 0, constraint_min)
        multipliers = run.latest_judge_feedback.constraints.symbol_risk_multipliers if run.latest_judge_feedback else {}
        if multipliers:
            for rule in plan.sizing_rules:
                multiplier = multipliers.get(rule.symbol)
                if multiplier is not None and rule.target_risk_pct is not None:
                    rule.target_risk_pct *= multiplier
        run.current_plan_id = plan.plan_id
        self.registry.update_strategy_run(run)
        cache_path = self.plan_provider._cache_path(run_id, plan_date, llm_input)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(plan.to_json(indent=2))
        return plan

    def _resolve_max_trades(
        self,
        plan_value: int | None,
        judge_feedback: JudgeFeedback | None,
    ) -> int | None:
        judge_limit = None
        if judge_feedback and judge_feedback.constraints.max_trades_per_day is not None:
            judge_limit = judge_feedback.constraints.max_trades_per_day
        if judge_limit is None:
            return plan_value if plan_value is not None else self.default_max_trades
        if plan_value is None:
            return judge_limit
        return min(plan_value, judge_limit)

    def _apply_strategist_constraints(self, plan: StrategyPlan, constraints: DisplayConstraints) -> StrategyPlan:
        plan = plan.model_copy(deep=True)
        must_fix = [entry.lower() for entry in constraints.must_fix or []]
        if any("at least one qualified trigger per day" in entry for entry in must_fix):
            plan.min_trades_per_day = max(plan.min_trades_per_day or 0, 1)
            cap = self.min_trade_hint_cap or self.default_max_trades
            plan.max_trades_per_day = min(plan.max_trades_per_day or self.default_max_trades, cap)
        sizing = constraints.sizing_adjustments or {}
        for rule in plan.sizing_rules:
            text = (sizing.get(rule.symbol) or "").lower()
            if text and "cut risk" in text and "25" in text and rule.target_risk_pct is not None:
                rule.target_risk_pct *= 0.75
        return plan


plan_service = StrategistPlanService()
