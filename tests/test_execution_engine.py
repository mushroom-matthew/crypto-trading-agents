from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from tools import execution_tools
from trading_core.trigger_compiler import compile_plan
from trading_core.execution_engine import ExecutionEngine, BlockReason


def _strategy_plan(run_id: str, plan_limit: int | None = 3) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="close > 0",
        exit_rule="close < 0",
        category="trend_continuation",
    )
    plan = StrategyPlan(
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
        max_trades_per_day=plan_limit,
    )
    plan.allowed_symbols = ["BTC-USD"]
    plan.allowed_directions = ["long"]
    plan.allowed_trigger_categories = ["trend_continuation"]
    return plan


def _events(count: int) -> list[dict]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        {"trigger_id": "btc_long", "timestamp": (base + timedelta(hours=i)).isoformat()}
        for i in range(count)
    ]


def _setup(monkeypatch, tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(execution_tools, "registry", registry)
    engine = ExecutionEngine()
    monkeypatch.setattr(execution_tools, "engine", engine)
    return registry, engine


def test_simulate_day_enforces_judge_trade_cap(tmp_path, monkeypatch):
    registry, _ = _setup(monkeypatch, tmp_path)
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    run.latest_judge_feedback = JudgeFeedback(
        constraints=JudgeConstraints(max_trades_per_day=2, disabled_trigger_ids=[], disabled_categories=[], risk_mode="normal")
    )
    registry.update_strategy_run(run)
    plan = _strategy_plan(run.run_id, plan_limit=5)
    compiled = compile_plan(plan)

    result = execution_tools.simulate_day_tool(run.run_id, plan.model_dump(), compiled.model_dump(), _events(3))
    assert result["executed"] == 2
    assert result["skipped"] == {BlockReason.DAILY_CAP.value: 1}


def test_simulate_day_respects_judge_disabled_category(tmp_path, monkeypatch):
    registry, _ = _setup(monkeypatch, tmp_path)
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    run.latest_judge_feedback = JudgeFeedback(
        constraints=JudgeConstraints(
            max_trades_per_day=None,
            disabled_trigger_ids=[],
            disabled_categories=["trend_continuation"],
            risk_mode="normal",
        )
    )
    registry.update_strategy_run(run)
    plan = _strategy_plan(run.run_id, plan_limit=5)
    compiled = compile_plan(plan)

    result = execution_tools.simulate_day_tool(run.run_id, plan.model_dump(), compiled.model_dump(), _events(2))
    assert result["executed"] == 0
    assert result["skipped"][BlockReason.CATEGORY.value] == 2


def test_run_live_step_accumulates_day_state(tmp_path, monkeypatch):
    registry, engine = _setup(monkeypatch, tmp_path)
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    registry.update_strategy_run(run)
    plan = _strategy_plan(run.run_id, plan_limit=2)
    compiled = compile_plan(plan)

    first = execution_tools.run_live_step_tool(run.run_id, plan.model_dump(), compiled.model_dump(), _events(1))
    assert first["executed"] == 1

    second_event = [{"trigger_id": "btc_long", "timestamp": "2024-01-01T01:00:00+00:00"}]
    second = execution_tools.run_live_step_tool(run.run_id, plan.model_dump(), compiled.model_dump(), second_event)
    assert second["executed"] == 1

    third_event = [{"trigger_id": "btc_long", "timestamp": "2024-01-01T02:00:00+00:00"}]
    third = execution_tools.run_live_step_tool(run.run_id, plan.model_dump(), compiled.model_dump(), third_event)
    assert third["executed"] == 0
    assert third["skipped"][BlockReason.DAILY_CAP.value] == 1


def test_emergency_exit_bypasses_daily_cap(tmp_path, monkeypatch):
    registry, engine = _setup(monkeypatch, tmp_path)
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    registry.update_strategy_run(run)
    plan = _strategy_plan(run.run_id, plan_limit=1)
    emergency = TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="timeframe=='1h'",
        exit_rule="",
        category="emergency_exit",
    )
    plan.triggers.append(emergency)
    plan.allowed_trigger_categories.append("emergency_exit")
    compiled = compile_plan(plan)

    first = execution_tools.run_live_step_tool(run.run_id, plan.model_dump(), compiled.model_dump(), _events(1))
    assert first["executed"] == 1
    # Daily cap reached now; emergency exit should still execute
    exit_event = [{"trigger_id": "btc_exit", "timestamp": "2024-01-01T01:00:00+00:00"}]
    second = execution_tools.run_live_step_tool(run.run_id, plan.model_dump(), compiled.model_dump(), exit_event)
    assert second["executed"] == 1
