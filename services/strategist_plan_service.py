"""Strategy plan generation service aware of StrategyRun registry."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.judge_feedback import JudgeFeedback
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
        plan.allowed_symbols = sorted(set(run.config.symbols))
        plan.allowed_directions = sorted({trigger.direction for trigger in plan.triggers})
        plan.allowed_trigger_categories = sorted({(trigger.category or "other") for trigger in plan.triggers})
        plan.max_trades_per_day = self._resolve_max_trades(plan.max_trades_per_day, run.latest_judge_feedback)
        run.current_plan_id = plan.plan_id
        self.registry.update_strategy_run(run)
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


plan_service = StrategistPlanService()
