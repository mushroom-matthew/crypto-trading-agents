from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import LLMCostTracker
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from services.strategy_run_registry import StrategyRunRegistry
from tools import execution_tools
from trading_core.execution_engine import ExecutionEngine


class StubPlanProvider:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan
        self.cost_tracker = LLMCostTracker()
        self.cache_dir = Path(".cache/strategy_plans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None, event_ts=None):
        return self.plan

    def _cache_path(self, run_id, plan_date, llm_input):
        return self.cache_dir / f"{run_id}.json"


def _build_candles() -> dict[str, dict[str, pd.DataFrame]]:
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz=timezone.utc)
    closes = [105.0 if i == 12 else 95.0 for i in range(24)]
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [price + 1 for price in closes],
            "low": [price - 1 for price in closes],
            "close": closes,
            "volume": [1000 for _ in closes],
        }
    ).set_index("timestamp")
    return {"BTC-USD": {"1h": data}}


def _simple_plan() -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="btc_mean_reversion",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="timeframe=='1h' and close < 100",
        exit_rule="close > 120",
        category="mean_reversion",
    )
    return StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)],
        max_trades_per_day=2,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["mean_reversion"],
        min_trades_per_day=1,
    )


def test_simple_plan_executes_one_trade(tmp_path, monkeypatch):
    plan = _simple_plan()
    market_data = _build_candles()
    run_registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=tmp_path / "cache",
        llm_calls_per_day=1,
        risk_params={
            "max_position_risk_pct": 5.0,
            "max_symbol_exposure_pct": 50.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
        },
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="test-run")
    assert result.final_positions["BTC-USD"] > 0
    report = next(entry for entry in result.daily_reports if entry["date"] == "2024-01-01")
    assert report["trade_count"] == 1
    assert report["return_pct"] == report["equity_return_pct"]
    expected_return = ((report["end_equity"] / report["start_equity"]) - 1) * 100
    assert abs(report["equity_return_pct"] - expected_return) < 1e-9
    limit_stats = report["limit_stats"]
    assert limit_stats["blocked_by_daily_cap"] == 0
    assert report["attempted_triggers"] >= report["executed_trades"]
    assert not report.get("missed_min_trades")
    assert "overnight_exposure" in report
    assert "pnl_breakdown" in report
    assert "symbol_pnl" in report
    assert limit_stats["risk_budget_usage_by_symbol"] == {}
    assert "trigger_stats" in report


def test_logs_risk_block_details(tmp_path, monkeypatch):
    plan = _simple_plan()
    market_data = _build_candles()
    run_registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=tmp_path / "cache",
        llm_calls_per_day=1,
        risk_params={
            "max_position_risk_pct": 0.0,
            "max_symbol_exposure_pct": 5.0,
            "max_portfolio_exposure_pct": 5.0,
            "max_daily_loss_pct": 1.0,
        },
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="risk-test")
    report = next(entry for entry in result.daily_reports if entry["date"] == "2024-01-01")
    limit_stats = report["limit_stats"]
    assert limit_stats["blocked_by_risk_limits"] > 0
    assert limit_stats["risk_block_breakdown"]["max_position_risk_pct"] > 0
    assert any(detail["reason"] == "max_position_risk_pct" for detail in limit_stats["blocked_details"])
    assert "pnl_breakdown" in report
    assert "trigger_stats" in report


def test_daily_risk_budget_blocks_orders(tmp_path, monkeypatch):
    plan = _simple_plan()
    plan.max_trades_per_day = 10
    plan.triggers[0].exit_rule = "True"
    market_data = _build_candles()
    run_registry = StrategyRunRegistry(tmp_path / "runs")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=tmp_path / "cache",
        llm_calls_per_day=1,
        risk_params={
            "max_position_risk_pct": 5.0,
            "max_symbol_exposure_pct": 50.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
            "max_daily_risk_budget_pct": 1.0,
        },
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="budget-test")
    report = next(entry for entry in result.daily_reports if entry["date"] == "2024-01-01")
    assert report["executed_trades"] >= 1
    limit_stats = report["limit_stats"]
    assert limit_stats["blocked_by_risk_budget"] > 0
    assert limit_stats["risk_budget_usage_by_symbol"]["BTC-USD"] >= 99.0
    assert limit_stats["risk_budget_blocks_by_symbol"]["BTC-USD"] > 0
    assert report["risk_budget"]["used_pct"] >= 90.0
    assert report["risk_budget"]["symbol_usage_pct"]["BTC-USD"] == pytest.approx(
        limit_stats["risk_budget_usage_by_symbol"]["BTC-USD"]
    )
    assert "trigger_stats" in report
