"""Strategy plan generation service aware of StrategyRun registry."""

from __future__ import annotations

import os
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Dict, Optional

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.judge_feedback import DisplayConstraints, JudgeFeedback
from schemas.llm_strategist import LLMInput, RiskConstraint, StrategyPlan
from services.strategy_run_registry import StrategyRunRegistry, strategy_run_registry
from services.risk_adjustment_service import effective_risk_limits
from schemas.strategy_run import RiskLimitSettings


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
        self.default_max_triggers = int(
            os.environ.get("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", str(self.default_max_trades))
        )
        self.min_trade_hint_cap = int(os.environ.get("STRATEGIST_PLAN_MIN_TRADE_CAP", "3"))
        self.strict_fixed_caps = os.environ.get("STRATEGIST_STRICT_FIXED_CAPS", "false").lower() == "true"

    def generate_plan_for_run(
        self,
        run_id: str,
        llm_input: LLMInput,
        plan_date: Optional[datetime] = None,
        prompt_template: str | None = None,
    ) -> StrategyPlan:
        run = self.registry.get_strategy_run(run_id)
        plan_date = plan_date or datetime.now(timezone.utc)
        risk_limits = effective_risk_limits(run)
        llm_input = self._llm_input_with_risk_overrides(llm_input, risk_limits)
        plan = self.plan_provider.get_plan(run_id, plan_date, llm_input, prompt_template=prompt_template)
        plan = plan.model_copy(deep=True)
        # Preserve declared policy caps before any derived-clamp is applied.
        if not hasattr(plan, "_policy_max_trades_per_day"):
            object.__setattr__(plan, "_policy_max_trades_per_day", plan.max_trades_per_day)
        if not hasattr(plan, "_policy_max_triggers_per_symbol_per_day"):
            object.__setattr__(plan, "_policy_max_triggers_per_symbol_per_day", plan.max_triggers_per_symbol_per_day)
        plan.risk_constraints = self._merge_plan_risk_constraints(plan.risk_constraints, risk_limits)
        if plan.risk_constraints.max_daily_risk_budget_pct is None and risk_limits.max_daily_risk_budget_pct is not None:
            plan.risk_constraints.max_daily_risk_budget_pct = risk_limits.max_daily_risk_budget_pct
        # Prefer derived caps when a daily budget exists; otherwise derive a fallback.
        derived_cap = getattr(plan, "_derived_trade_cap", None)
        if derived_cap:
            if not self.strict_fixed_caps:
                plan.max_trades_per_day = derived_cap
        elif plan.risk_constraints.max_daily_risk_budget_pct is not None and plan.risk_constraints.max_position_risk_pct > 0:
            fallback_cap = max(
                8,
                math.ceil(plan.risk_constraints.max_daily_risk_budget_pct / plan.risk_constraints.max_position_risk_pct),
            )
            if not self.strict_fixed_caps:
                plan.max_trades_per_day = fallback_cap
            object.__setattr__(plan, "_derived_trade_cap", fallback_cap)
            object.__setattr__(plan, "_derived_trigger_cap", fallback_cap)
            cap_inputs = getattr(plan, "_cap_inputs", None) or {}
            if plan.risk_constraints.max_daily_risk_budget_pct and plan.risk_constraints.max_position_risk_pct:
                cap_inputs = {
                    "risk_budget_pct": plan.risk_constraints.max_daily_risk_budget_pct,
                    "per_trade_risk_pct": plan.risk_constraints.max_position_risk_pct,
                }
            if cap_inputs:
                object.__setattr__(plan, "_cap_inputs", cap_inputs)
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
        if not self.strict_fixed_caps:
            plan.max_triggers_per_symbol_per_day = self._resolve_symbol_trigger_cap(
                plan.max_triggers_per_symbol_per_day,
                run.latest_judge_feedback,
            )
        plan.trigger_budgets = self._sanitize_trigger_budgets(plan.trigger_budgets)
        plan = self._enforce_derived_trade_cap(plan)
        judge_constraints = run.latest_judge_feedback.constraints if run.latest_judge_feedback else None
        plan = self._resolve_final_caps(plan, judge_constraints)
        run.current_plan_id = plan.plan_id
        self.registry.update_strategy_run(run)
        cache_path = self.plan_provider._cache_path(run_id, plan_date, llm_input)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(plan.to_json(indent=2))
        return plan

    def _llm_input_with_risk_overrides(self, llm_input: LLMInput, risk_limits: RiskLimitSettings) -> LLMInput:
        payload = llm_input.model_copy(deep=True)
        params = dict(payload.risk_params or {})
        params.update(risk_limits.to_risk_params())
        payload.risk_params = params
        return payload

    @staticmethod
    def _merge_plan_risk_constraints(plan_constraints: RiskConstraint, risk_limits: RiskLimitSettings) -> RiskConstraint:
        overrides = risk_limits.to_risk_params()
        return plan_constraints.model_copy(update=overrides)

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

    def _resolve_symbol_trigger_cap(
        self,
        plan_value: int | None,
        judge_feedback: JudgeFeedback | None,
    ) -> int | None:
        judge_cap = None
        if judge_feedback and judge_feedback.constraints.max_triggers_per_symbol_per_day is not None:
            judge_cap = judge_feedback.constraints.max_triggers_per_symbol_per_day
        if judge_cap is None:
            return plan_value
        if plan_value is None:
            return judge_cap
        return min(plan_value, judge_cap)

    @staticmethod
    def _sanitize_trigger_budgets(budgets: Dict[str, int] | None) -> Dict[str, int]:
        cleaned: Dict[str, int] = {}
        if not budgets:
            return cleaned
        for symbol, cap in budgets.items():
            try:
                val = int(cap)
            except (TypeError, ValueError):
                continue
            if val > 0:
                cleaned[symbol] = val
        return cleaned

    def _enforce_derived_trade_cap(self, plan: StrategyPlan) -> StrategyPlan:
        """Ensure plans with a daily risk budget enforce the derived trade cap."""

        if plan.risk_constraints.max_daily_risk_budget_pct is None:
            return plan
        derived_cap = getattr(plan, "_derived_trade_cap", None)
        if not derived_cap:
            budget_pct = plan.risk_constraints.max_daily_risk_budget_pct
            per_trade_risk = None
            if plan.sizing_rules:
                risks = [rule.target_risk_pct for rule in plan.sizing_rules if rule.target_risk_pct and rule.target_risk_pct > 0]
                if risks:
                    per_trade_risk = min(risks)
            if per_trade_risk is None or per_trade_risk <= 0:
                per_trade_risk = plan.risk_constraints.max_position_risk_pct
            if per_trade_risk and per_trade_risk > 0 and budget_pct and budget_pct > 0:
                derived_cap = max(8, math.ceil(budget_pct / per_trade_risk))
            cap_inputs = getattr(plan, "_cap_inputs", None) or {}
            if budget_pct and per_trade_risk:
                cap_inputs = {"risk_budget_pct": budget_pct, "per_trade_risk_pct": per_trade_risk}
            if cap_inputs:
                object.__setattr__(plan, "_cap_inputs", cap_inputs)
        if not derived_cap:
            return plan
        object.__setattr__(plan, "_derived_trade_cap", int(derived_cap))
        object.__setattr__(plan, "_derived_trigger_cap", int(derived_cap))
        if self.strict_fixed_caps:
            # In fixed-cap mode, do not overwrite policy caps.
            return plan
        plan = plan.model_copy(update={"max_trades_per_day": int(derived_cap)})
        if plan.max_triggers_per_symbol_per_day is None or plan.max_triggers_per_symbol_per_day < derived_cap:
            plan.max_triggers_per_symbol_per_day = int(derived_cap)
        return plan

    def _resolve_final_caps(self, plan: StrategyPlan, judge_constraints: JudgeConstraints | None) -> StrategyPlan:
        """Compute policy/derived/resolved caps for trades and triggers and apply resolved."""

        default_trades = self.default_max_trades
        default_triggers = self.default_max_triggers
        plan_value_trade = plan.max_trades_per_day
        plan_value_trigger = plan.max_triggers_per_symbol_per_day
        policy_trade = getattr(plan, "_policy_max_trades_per_day", None) or plan_value_trade or default_trades
        policy_trigger = getattr(plan, "_policy_max_triggers_per_symbol_per_day", None) or plan_value_trigger or default_triggers
        if self.strict_fixed_caps:
            policy_trade = max(policy_trade, default_trades)
            policy_trigger = max(policy_trigger, default_triggers)
        derived_trade = getattr(plan, "_derived_trade_cap", None)
        derived_trigger = getattr(plan, "_derived_trigger_cap", derived_trade)

        if self.strict_fixed_caps:
            resolved_trade = policy_trade
            resolved_trigger = policy_trigger
        else:
            resolved_trade = derived_trade if derived_trade is not None else policy_trade
            if derived_trigger is not None and policy_trigger is not None:
                resolved_trigger = min(policy_trigger, derived_trigger)
            else:
                resolved_trigger = derived_trigger if derived_trigger is not None else policy_trigger

        if judge_constraints:
            judge_trade = judge_constraints.max_trades_per_day
            judge_trigger = judge_constraints.max_triggers_per_symbol_per_day
            if judge_trade is not None and resolved_trade is not None:
                resolved_trade = min(resolved_trade, judge_trade)
            if not self.strict_fixed_caps and judge_trigger is not None and resolved_trigger is not None:
                resolved_trigger = min(resolved_trigger, judge_trigger)

        plan = plan.model_copy(
            update={
                "max_trades_per_day": resolved_trade,
                "max_triggers_per_symbol_per_day": resolved_trigger,
            }
        )
        object.__setattr__(plan, "_policy_max_trades_per_day", policy_trade)
        object.__setattr__(plan, "_policy_max_triggers_per_symbol_per_day", policy_trigger)
        if derived_trade is not None:
            object.__setattr__(plan, "_derived_trade_cap", derived_trade)
        if derived_trigger is not None:
            object.__setattr__(plan, "_derived_trigger_cap", derived_trigger)
        object.__setattr__(plan, "_resolved_trade_cap", resolved_trade)
        object.__setattr__(plan, "_resolved_trigger_cap", resolved_trigger)
        return plan

    def _apply_strategist_constraints(self, plan: StrategyPlan, constraints: DisplayConstraints) -> StrategyPlan:
        plan = plan.model_copy(deep=True)
        must_fix = [entry.lower() for entry in constraints.must_fix or []]
        if any("at least one qualified trigger per day" in entry for entry in must_fix):
            plan.min_trades_per_day = max(plan.min_trades_per_day or 0, 1)
            cap = self.min_trade_hint_cap or self.default_max_trades
            plan.max_trades_per_day = min(plan.max_trades_per_day or self.default_max_trades, cap)
        return plan


plan_service = StrategistPlanService()
