from __future__ import annotations

import pytest

from schemas.compiled_plan import CompiledPlan
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from tools import strategy_run_tools
from trading_core.trigger_compiler import TriggerCompilationError, compile_plan

from datetime import datetime, timedelta, timezone


def _strategy_plan(run_id: str, entry_rule: str = "close > 0") -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule="close < 0",
        category="trend_continuation",
    )
    return StrategyPlan(
        plan_id="plan_test",
        run_id=run_id,
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )


def test_compile_plan_succeeds_with_valid_expressions(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "close > sma_short")
    compiled = compile_plan(plan)
    assert isinstance(compiled, CompiledPlan)
    assert compiled.triggers[0].entry.normalized == "close > sma_short"


def test_compile_plan_fails_on_invalid_identifier(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    plan = _strategy_plan(run.run_id, "evil_call()")
    with pytest.raises(TriggerCompilationError):
        compile_plan(plan)


def test_compile_tool_updates_run(tmp_path, monkeypatch):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    monkeypatch.setattr(strategy_run_tools, "registry", registry)
    plan = _strategy_plan(run.run_id).model_dump()
    compiled = strategy_run_tools.compile_plan_tool(plan)
    assert compiled["plan_id"] == "plan_test"
    stored = registry.get_strategy_run(run.run_id)
    assert stored.plan_active is True
    assert stored.compiled_plan_id == "plan_test"

