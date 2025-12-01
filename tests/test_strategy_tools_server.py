from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agents.strategies.plan_provider import StrategyPlanProvider
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
from schemas.strategy_run import StrategyRunConfig
from services.strategy_run_registry import StrategyRunRegistry
from services.strategist_plan_service import StrategistPlanService
from tools import execution_tools, strategy_run_tools
from trading_core.execution_engine import ExecutionEngine
from trading_core.trigger_compiler import compile_plan
from mcp_server.strategy_tools_server import (
    compile_plan as mcp_compile_plan,
    create_strategy_run as mcp_create_strategy_run,
    generate_plan_for_run as mcp_generate_plan_for_run,
    run_live_step as mcp_run_live_step,
    simulate_day as mcp_simulate_day,
)


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


def _base_plan(run_id: str) -> StrategyPlan:
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
        max_trades_per_day=2,
    )
    plan.allowed_symbols = ["BTC-USD"]
    plan.allowed_directions = ["long"]
    plan.allowed_trigger_categories = ["trend_continuation"]
    return plan


class StubPlanProvider:
    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None):
        return _base_plan(run_id)


def _events(count: int) -> list[dict]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [{"trigger_id": "btc_long", "timestamp": (base + timedelta(hours=i)).isoformat()} for i in range(count)]


@pytest.mark.asyncio
async def test_strategy_tools_server_end_to_end(monkeypatch, tmp_path):
    registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(strategy_run_tools, "registry", registry)
    monkeypatch.setattr(execution_tools, "registry", registry)
    engine = ExecutionEngine()
    monkeypatch.setattr(execution_tools, "engine", engine)

    service = StrategistPlanService(plan_provider=StubPlanProvider(), registry=registry)
    monkeypatch.setattr(strategy_run_tools, "plan_service", service)

    config = {
        "symbols": ["BTC-USD"],
        "timeframes": ["1h"],
        "history_window_days": 7,
        "plan_cadence_hours": 24,
    }
    created = await mcp_create_strategy_run(config)
    run_id = created["run_id"]

    llm_input = _llm_input().model_dump()
    plan_payload = await mcp_generate_plan_for_run(run_id, llm_input)
    compiled_payload = await mcp_compile_plan(plan_payload)

    simulation = await mcp_simulate_day(run_id, plan_payload, compiled_payload, _events(3))
    assert simulation["executed"] == 2
    assert simulation["skipped"]["max_trades_per_day"] == 1

    next_day_event = [{"trigger_id": "btc_long", "timestamp": "2024-01-02T00:00:00+00:00"}]
    step = await mcp_run_live_step(run_id, plan_payload, compiled_payload, next_day_event)
    assert step["executed"] == 1


@pytest.mark.asyncio
async def test_tool_call_logging(monkeypatch, tmp_path):
    from mcp_server import strategy_tools_server
    from services.tool_call_logger import logger as tool_logger

    tool_logger.log_dir = tmp_path
    (tmp_path).mkdir(parents=True, exist_ok=True)
    log_file = tool_logger.log_dir / "tool_calls.jsonl"
    if log_file.exists():
        log_file.unlink()

    registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(strategy_run_tools, "registry", registry)
    config = {"symbols": ["BTC-USD"], "timeframes": ["1h"], "history_window_days": 7, "plan_cadence_hours": 24}
    await strategy_tools_server.create_strategy_run(config)
    assert log_file.exists()
    contents = log_file.read_text().strip().splitlines()
    assert contents
