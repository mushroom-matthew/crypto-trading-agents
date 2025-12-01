"""StrategyPlan caching and LLM-call budgeting."""

from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from schemas.llm_strategist import LLMInput, StrategyPlan

from .llm_client import LLMClient


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class LLMCostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 0
    cost_input_per_1k: float = 0.01
    cost_output_per_1k: float = 0.03

    def record(self, prompt_json: str, completion_json: str) -> None:
        self.input_tokens += _estimate_tokens(prompt_json)
        self.output_tokens += _estimate_tokens(completion_json)
        self.num_calls += 1

    @property
    def estimated_cost(self) -> float:
        return (
            (self.input_tokens / 1000.0) * self.cost_input_per_1k
            + (self.output_tokens / 1000.0) * self.cost_output_per_1k
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "num_llm_calls": self.num_calls,
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


class StrategyPlanProvider:
    """Loads cached plans or requests fresh ones within a daily call budget."""

    def __init__(self, llm_client: LLMClient, cache_dir: Path, llm_calls_per_day: int = 1) -> None:
        self.llm_client = llm_client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_per_day = llm_calls_per_day
        self.daily_counts = defaultdict(int)
        self.cost_tracker = LLMCostTracker()

    def _cache_path(self, run_id: str, plan_date: datetime, llm_input: LLMInput) -> Path:
        digest = hashlib.sha256(llm_input.to_json().encode("utf-8")).hexdigest()
        date_bucket = plan_date.strftime("%Y-%m-%d")
        return self.cache_dir / run_id / date_bucket / f"{digest}.json"

    def _load_cached(self, path: Path) -> StrategyPlan | None:
        if not path.exists():
            return None
        return StrategyPlan.model_validate_json(path.read_text())

    def get_plan(self, run_id: str, plan_date: datetime, llm_input: LLMInput, prompt_template: str | None = None) -> StrategyPlan:
        cache_path = self._cache_path(run_id, plan_date, llm_input)
        cached = self._load_cached(cache_path)
        if cached:
            plan = self._enrich_plan(cached, llm_input)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(plan.to_json(indent=2))
            return plan
        date_key = (run_id, plan_date.strftime("%Y-%m-%d"))
        if self.daily_counts[date_key] >= self.llm_calls_per_day:
            raise RuntimeError(f"LLM call budget exhausted for {date_key[1]}")
        plan = self.llm_client.generate_plan(llm_input, prompt_template=prompt_template)
        plan = self._enrich_plan(plan, llm_input)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(plan.to_json(indent=2))
        self.daily_counts[date_key] += 1
        self.cost_tracker.record(llm_input.to_json(), plan.to_json())
        return plan

    def _enrich_plan(self, plan: StrategyPlan, llm_input: LLMInput) -> StrategyPlan:
        """Ensure plan defaults (limits and allowed sets) are populated."""

        plan = plan.model_copy(deep=True)
        universe = sorted({asset.symbol for asset in llm_input.assets})
        default_max = int(os.environ.get("STRATEGIST_PLAN_DEFAULT_MAX_TRADES", "10"))
        if plan.max_trades_per_day is None:
            plan.max_trades_per_day = default_max
        if not plan.allowed_symbols:
            plan.allowed_symbols = universe
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
            plan.allowed_trigger_categories = sorted({category for category in plan.allowed_trigger_categories if category})
        if any(trigger.category is None for trigger in plan.triggers):
            if "other" not in plan.allowed_trigger_categories:
                plan.allowed_trigger_categories.append("other")
        return plan
