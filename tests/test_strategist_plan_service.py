from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.judge_feedback import DisplayConstraints, JudgeFeedback, JudgeConstraints
from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput,
    PortfolioState,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)
from schemas.strategy_run import RiskAdjustmentState, RiskLimitSettings, StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from services.strategist_plan_service import StrategistPlanService
from tools import strategy_run_tools


def _llm_input() -> LLMInput:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    indicator = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=ts, close=40000.0)
    asset = AssetState(symbol="BTC-USD", indicators=[indicator], trend_state="uptrend", vol_state="normal")
    portfolio = PortfolioState(
        timestamp=ts,
        equity=100000.0,
        cash=100000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )
    return LLMInput(portfolio=portfolio, assets=[asset], risk_params={"max_position_risk_pct": 1.0})


def _base_plan() -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="False",
        category="trend_continuation",
    )
    return StrategyPlan(
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
        max_trades_per_day=None,
    )


class StubPlanProvider:
    def __init__(self, plan: StrategyPlan, cache_dir: Path | None = None) -> None:
        self.plan = plan
        self.cache_dir = cache_dir or Path(".cache/test_plans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_llm_input: LLMInput | None = None

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None, use_vector_store=False, event_ts=None, emit_events=True):
        self.last_llm_input = llm_input
        return self.plan.model_copy(deep=True)

    def _cache_path(self, run_id, plan_date, llm_input):
        ident = f"{run_id}_{plan_date.isoformat().replace(':', '-')}"
        return self.cache_dir / f"{ident}.json"


def test_plan_service_respects_judge_cap(tmp_path, monkeypatch):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD", "ETH-USD"], timeframes=["1h"], history_window_days=7)
    )
    run.latest_judge_feedback = JudgeFeedback(constraints=JudgeConstraints(max_trades_per_day=10, risk_mode="normal"))
    registry.update_strategy_run(run)

    plan = _base_plan()
    service = StrategistPlanService(plan_provider=StubPlanProvider(plan), registry=registry)

    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.run_id == run.run_id
    assert result.max_trades_per_day == 10
    assert set(result.allowed_symbols) == {"BTC-USD", "ETH-USD"}
    assert result.allowed_directions == ["long", "short"]
    stored = registry.get_strategy_run(run.run_id)
    assert stored.current_plan_id == result.plan_id


def test_plan_service_applies_default_when_max_trades_missing(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    service = StrategistPlanService(plan_provider=StubPlanProvider(_base_plan()), registry=registry)
    plan = service.generate_plan_for_run(run.run_id, _llm_input())
    assert plan.max_trades_per_day is not None


def test_generate_plan_tool_updates_run(tmp_path, monkeypatch):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    run.latest_judge_feedback = JudgeFeedback(constraints=JudgeConstraints(max_trades_per_day=5, risk_mode="normal"))
    registry.update_strategy_run(run)

    plan = _base_plan()
    service = StrategistPlanService(plan_provider=StubPlanProvider(plan), registry=registry)
    monkeypatch.setattr(strategy_run_tools, "plan_service", service)
    monkeypatch.setattr(strategy_run_tools, "registry", registry)

    llm_input = _llm_input().model_dump()
    result = strategy_run_tools.generate_plan_for_run_tool(run.run_id, llm_input)
    assert result["max_trades_per_day"] == 5
    stored = registry.get_strategy_run(run.run_id)
    assert stored.current_plan_id == result["plan_id"]


def test_plan_service_applies_strategist_constraints(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7)
    )
    feedback = JudgeFeedback(
        constraints=JudgeConstraints(max_trades_per_day=6, risk_mode="normal"),
        strategist_constraints=DisplayConstraints(
            must_fix=["At least one qualified trigger per day"],
            sizing_adjustments={"BTC-USD": "Cut risk by 25% until two winning days"},
        ),
    )
    run.latest_judge_feedback = feedback
    registry.update_strategy_run(run)
    plan = StrategistPlanService(plan_provider=StubPlanProvider(_base_plan()), registry=registry).generate_plan_for_run(
        run.run_id, _llm_input()
    )
    assert plan.min_trades_per_day == 1
    assert plan.max_trades_per_day == 3


def test_plan_service_respects_judge_structured_constraints(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    feedback = JudgeFeedback(
        constraints=JudgeConstraints(
            max_trades_per_day=6,
            min_trades_per_day=2,
            symbol_risk_multipliers={"BTC-USD": 0.5},
        )
    )
    run.latest_judge_feedback = feedback
    registry.update_strategy_run(run)
    plan = StrategistPlanService(plan_provider=StubPlanProvider(_base_plan()), registry=registry).generate_plan_for_run(
        run.run_id, _llm_input()
    )
    assert plan.min_trades_per_day == 2
    assert plan.max_trades_per_day == 6


def test_plan_service_prefers_derived_cap_with_risk_budget(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=2.0,
        max_symbol_exposure_pct=25.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=risk_limits)
    )
    run.latest_judge_feedback = JudgeFeedback(
        constraints=JudgeConstraints(max_trades_per_day=None, risk_mode="normal"),
        strategist_constraints=DisplayConstraints(must_fix=["At least one qualified trigger per day"]),
    )
    registry.update_strategy_run(run)
    plan = _base_plan()
    plan.max_trades_per_day = 3
    service = StrategistPlanService(plan_provider=StubPlanProvider(plan), registry=registry)
    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.risk_constraints.max_daily_risk_budget_pct == pytest.approx(10.0)
    assert getattr(result, "_derived_trade_cap") >= 8
    assert result.max_trades_per_day == getattr(result, "_derived_trade_cap")
    assert result.max_triggers_per_symbol_per_day == getattr(result, "_derived_trade_cap")


def test_plan_service_overrides_risk_constraints_and_injects_llm_input(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=3.5,
        max_symbol_exposure_pct=40.0,
        max_portfolio_exposure_pct=70.0,
        max_daily_loss_pct=2.0,
    )
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=risk_limits)
    )
    plan = _base_plan()
    stub = StubPlanProvider(plan)
    service = StrategistPlanService(plan_provider=stub, registry=registry)
    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.risk_constraints.max_position_risk_pct == pytest.approx(risk_limits.max_position_risk_pct)
    assert result.risk_constraints.max_symbol_exposure_pct == pytest.approx(risk_limits.max_symbol_exposure_pct)
    assert result.risk_constraints.max_portfolio_exposure_pct == pytest.approx(risk_limits.max_portfolio_exposure_pct)
    assert result.risk_constraints.max_daily_loss_pct == pytest.approx(risk_limits.max_daily_loss_pct)
    assert stub.last_llm_input is not None
    assert stub.last_llm_input.risk_params["max_position_risk_pct"] == pytest.approx(risk_limits.max_position_risk_pct)


def test_plan_service_scales_limits_with_active_adjustments(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=RiskLimitSettings(max_position_risk_pct=2.0))
    )
    run.risk_adjustments = {"BTC-USD": RiskAdjustmentState(multiplier=0.5, instruction="Cut risk by 50%")}
    registry.update_strategy_run(run)
    plan = _base_plan()
    stub = StubPlanProvider(plan)
    service = StrategistPlanService(plan_provider=stub, registry=registry)
    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert stub.last_llm_input is not None
    assert stub.last_llm_input.risk_params["max_position_risk_pct"] == pytest.approx(1.0)
    assert result.risk_constraints.max_position_risk_pct == pytest.approx(1.0)


def test_plan_service_respects_fixed_caps_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "true")
    registry = StrategyRunRegistry(tmp_path / "runs")
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=25.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=risk_limits)
    )
    registry.update_strategy_run(run)

    plan = _base_plan()
    plan.max_trades_per_day = 30
    plan.max_triggers_per_symbol_per_day = 30
    plan.risk_constraints.max_daily_risk_budget_pct = 10.0
    stub = StubPlanProvider(plan)
    service = StrategistPlanService(plan_provider=stub, registry=registry)

    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.max_trades_per_day == 30
    assert result.max_triggers_per_symbol_per_day == 30
    assert getattr(result, "_derived_trade_cap") == 10
    cap_inputs = getattr(result, "_cap_inputs", {})
    assert cap_inputs.get("risk_budget_pct") == 10.0
    assert cap_inputs.get("per_trade_risk_pct") == 1.0


def test_plan_service_triggers_floor_with_fixed_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "true")
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", "40")
    registry = StrategyRunRegistry(tmp_path / "runs")
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=25.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=risk_limits)
    )
    registry.update_strategy_run(run)

    plan = _base_plan()
    plan.max_triggers_per_symbol_per_day = 4
    plan.max_trades_per_day = 5
    plan.risk_constraints.max_daily_risk_budget_pct = 10.0
    stub = StubPlanProvider(plan)
    service = StrategistPlanService(plan_provider=stub, registry=registry)

    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.max_triggers_per_symbol_per_day == 40
    assert getattr(result, "_policy_max_triggers_per_symbol_per_day") == 40
    assert getattr(result, "_derived_trigger_cap") == 10
    assert getattr(result, "_resolved_trigger_cap") == 40


def test_plan_service_triggers_legacy_min_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "false")
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", "40")
    registry = StrategyRunRegistry(tmp_path / "runs")
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=25.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    run = registry.create_strategy_run(
        StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7, risk_limits=risk_limits)
    )
    registry.update_strategy_run(run)

    plan = _base_plan()
    plan.max_triggers_per_symbol_per_day = 4
    plan.max_trades_per_day = 5
    plan.risk_constraints.max_daily_risk_budget_pct = 10.0
    stub = StubPlanProvider(plan)
    service = StrategistPlanService(plan_provider=stub, registry=registry)

    result = service.generate_plan_for_run(run.run_id, _llm_input())
    assert result.max_triggers_per_symbol_per_day == 4
    assert getattr(result, "_policy_max_triggers_per_symbol_per_day") == 4
    assert getattr(result, "_derived_trigger_cap") == 10
    assert getattr(result, "_resolved_trigger_cap") == 4


def test_plan_service_enforces_judge_trigger_budget(tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    run = registry.create_strategy_run(StrategyRunConfig(symbols=["BTC-USD"], timeframes=["1h"], history_window_days=7))
    run.latest_judge_feedback = JudgeFeedback(
        constraints=JudgeConstraints(max_trades_per_day=6, max_triggers_per_symbol_per_day=4)
    )
    registry.update_strategy_run(run)
    plan = StrategistPlanService(plan_provider=StubPlanProvider(_base_plan()), registry=registry).generate_plan_for_run(
        run.run_id, _llm_input()
    )
    assert plan.max_triggers_per_symbol_per_day == 4
