"""Fixed plan provider and helpers for the VFTP strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition, TriggerCategory


VFTP_CATEGORY: TriggerCategory = "trend_continuation"


def build_vftp_plan(symbol: str, *, risk_pct: float = 1.0) -> StrategyPlan:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    entry_rule = (
        "tf_4h_ema_50 > tf_4h_ema_200 "
        "and tf_4h_adx_14 > 22 "
        "and tf_4h_atr_14_rising_3 "
        "and close > ema_200 "
        "and rsi_14 between 40 and 50 "
        "and close > prev_high "
        "and position == 'flat'"
    )
    trigger = TriggerCondition(
        id=f"{symbol.lower().replace('-', '_')}_vftp_long",
        symbol=symbol,
        category=VFTP_CATEGORY,
        confidence_grade="A",
        direction="long",
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule="",
    )
    risk_constraints = RiskConstraint(
        max_position_risk_pct=risk_pct,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=20.0,
        max_daily_risk_budget_pct=100.0,
    )
    sizing_rules = [
        PositionSizingRule(
            symbol=symbol,
            sizing_mode="fixed_fraction",
            target_risk_pct=risk_pct,
        )
    ]
    return StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=365 * 5),
        regime="bull",
        stance="active",
        global_view="VFTP deterministic plan (fixed, LLM-free).",
        triggers=[trigger],
        risk_constraints=risk_constraints,
        sizing_rules=sizing_rules,
        max_trades_per_day=12,
        max_triggers_per_symbol_per_day=6,
        allowed_symbols=[symbol],
        allowed_directions=["long"],
        allowed_trigger_categories=[VFTP_CATEGORY],
        rationale="Volatility-Filtered Trend Pullback (VFTP) fixed plan.",
    )


@dataclass
class FixedPlanProvider(StrategyPlanProvider):
    """Plan provider that returns a fixed StrategyPlan for every request."""

    plan: StrategyPlan

    def __init__(self, plan: StrategyPlan, cache_dir: Path | None = None) -> None:
        cache = cache_dir or Path(".cache/strategy_plans")
        super().__init__(llm_client=LLMClient(transport=lambda _: ""), cache_dir=cache, llm_calls_per_day=1)
        self.plan = plan

    def get_plan(  # type: ignore[override]
        self,
        run_id: str,
        plan_date: datetime,
        llm_input: Any,
        prompt_template: str | None = None,
        use_vector_store: bool = False,
        event_ts: datetime | None = None,
        emit_events: bool = True,
    ) -> StrategyPlan:
        plan = self.plan.model_copy(update={"run_id": run_id, "generated_at": plan_date})
        plan = self._enrich_plan(plan, llm_input)
        self.last_generation_info = {
            "source": "fixed",
            "fallback_plan_used": False,
            "llm_failed_parse": False,
            "llm_failure_reason": None,
            "raw_output": None,
        }
        return plan
