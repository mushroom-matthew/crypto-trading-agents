"""LLM planner service that emits strategy configs for the next day."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any

from schemas.strategy_plan import StrategyPlan
from trading_core.config import PlannerSettings


@dataclass
class PlannerPreferences:
    risk_mode: str
    allow_go_to_cash: bool = True


@dataclass
class PlannerRequest:
    symbol: str
    indicators: Dict[str, Any]
    preferences: PlannerPreferences


@dataclass
class PlannerResponse:
    plan: StrategyPlan
    metadata: Dict[str, Any]


class LLMPlanner:
    def __init__(self, settings: PlannerSettings) -> None:
        self.settings = settings

    def plan(self, request: PlannerRequest) -> PlannerResponse:
        """Call the LLM planner and return a validated StrategyPlan."""

        # TODO: implement actual LLM call. For now, return a dummy plan.
        payload = {
            "strategy_id": f"plan-{request.symbol}",
            "created_at": datetime.now(timezone.utc),
            "symbol": request.symbol,
            "timeframe": "1h",
            "lookback": {"preferred_bars": 100, "min_bars": 50, "max_bars": 200},
            "risk": {
                "max_fraction_of_balance": 0.3,
                "risk_per_trade_fraction": 0.01,
                "max_drawdown_pct": 0.4,
                "leverage": 1.0,
            },
            "entry_conditions": [],
            "exit_conditions": [],
            "replan_triggers": {},
            "llm_metadata": {"model_name": self.settings.llm_model, "prompt_version": "v0"},
        }
        plan = StrategyPlan.model_validate(payload)
        return PlannerResponse(plan=plan, metadata={"notes": "stub"})
