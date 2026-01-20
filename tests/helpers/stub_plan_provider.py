"""Deterministic stub plan providers for tests (no LLM / network)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition


class AlwaysLongPlanProvider:
    """Stub provider that emits a single long trigger with explicit stop; at most one open position."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
        stop_loss_pct: float = 5.0,
        max_position_risk_pct: float = 1.0,
        max_daily_loss_pct: float = 1.0,
        max_daily_risk_budget_pct: float | None = 20.0,
        cache_dir: str | Path = ".cache/strategy_plans",
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.stop_loss_pct = stop_loss_pct
        self.max_position_risk_pct = max_position_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_risk_budget_pct = max_daily_risk_budget_pct
        self.cache_dir = str(cache_dir)
        self.cost_tracker = type("Cost", (), {"snapshot": lambda self: {}})()

    def _cache_path(self, run_id: str, plan_date: datetime, llm_input: Any) -> Path:
        return Path(self.cache_dir) / f"{run_id}_{plan_date.date().isoformat()}.json"

    def get_plan(
        self,
        run_id: str,
        plan_date: datetime,
        llm_input: Any,
        prompt_template: Optional[str] = None,
        event_ts: Optional[datetime] = None,
    ) -> StrategyPlan:
        plan = StrategyPlan(
            plan_id=f"plan_{plan_date.isoformat()}",
            run_id=run_id,
            generated_at=plan_date,
            valid_until=plan_date,
            global_view=None,
            regime="range",
            triggers=[
                TriggerCondition(
                    id="long_entry",
                    symbol=self.symbol,
                    direction="long",
                    timeframe=self.timeframe,
                    entry_rule="True",
                    exit_rule="position!='flat'",
                    category="mean_reversion",
                    stop_loss_pct=self.stop_loss_pct,
                )
            ],
            risk_constraints=RiskConstraint(
                max_position_risk_pct=self.max_position_risk_pct,
                max_symbol_exposure_pct=100.0,
                max_portfolio_exposure_pct=100.0,
                max_daily_loss_pct=self.max_daily_loss_pct,
                max_daily_risk_budget_pct=self.max_daily_risk_budget_pct,
            ),
            sizing_rules=[
                PositionSizingRule(symbol=self.symbol, sizing_mode="fixed_fraction", target_risk_pct=self.max_position_risk_pct)
            ],
            max_trades_per_day=200,
            allowed_symbols=[self.symbol],
            allowed_directions=["long", "short"],
            allowed_trigger_categories=["mean_reversion", "emergency_exit", "other"],
        )
        return plan
