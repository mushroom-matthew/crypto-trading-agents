from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from agents.strategies.plan_provider import LLMCostTracker
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from agents.strategies.llm_client import LLMClient
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from schemas.strategy_run import RiskLimitSettings
from services.strategy_run_registry import StrategyRunRegistry
from tools import execution_tools
from trading_core.execution_engine import ExecutionEngine


class StubPlanProvider:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan
        self.cost_tracker = LLMCostTracker()
        self.cache_dir = Path(".cache/strategy_plans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None, event_ts=None):  # noqa: D401
        return self.plan

    def _cache_path(self, run_id, plan_date, llm_input):
        ident = f"{run_id}_{plan_date.isoformat().replace(':', '-')}"
        return self.cache_dir / f"{ident}.json"


def _build_candles() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h", tz=timezone.utc)
    data = {
        "timestamp": timestamps,
        "open": [100 + i for i in range(10)],
        "high": [101 + i for i in range(10)],
        "low": [99 + i for i in range(10)],
        "close": [100 + i for i in range(10)],
        "volume": [1000 + i for i in range(10)],
    }
    return pd.DataFrame(data).set_index("timestamp")


def _risk_params() -> dict[str, float]:
    return {
        "max_position_risk_pct": 5.0,
        "max_symbol_exposure_pct": 50.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
    }


def test_backtester_executes_trigger(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
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
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="test-run")
    assert result.final_positions["BTC-USD"] > 0
    assert result.fills.shape[0] == 1
    assert result.daily_reports
    summary = result.daily_reports[-1]
    assert "judge_feedback" in summary
    assert "strategist_constraints" in summary["judge_feedback"]
    assert "plan_limits" in summary
    assert "max_triggers_per_symbol_per_day" in summary["plan_limits"]
    assert "limit_stats" in summary
    assert "risk_limit_hints" in summary["limit_stats"]
    assert "blocked_details" in summary["limit_stats"]
    assert "risk_adjustments" in summary
    assert "overnight_exposure" in summary
    assert "pnl_breakdown" in summary
    assert "trigger_stats" in summary


@pytest.mark.parametrize("strict_fixed_caps", [True, False])
def test_cap_state_reports_policy_vs_derived(monkeypatch, tmp_path, strict_fixed_caps):
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "true" if strict_fixed_caps else "false")
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
                category="trend_continuation",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
            max_daily_risk_budget_pct=10.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=30,
        max_triggers_per_symbol_per_day=40,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / f"runs_cap_state_{strict_fixed_caps}")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    risk_limits = RiskLimitSettings(
        max_position_risk_pct=1.0,
        max_symbol_exposure_pct=50.0,
        max_portfolio_exposure_pct=80.0,
        max_daily_loss_pct=3.0,
        max_daily_risk_budget_pct=10.0,
    )
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=risk_limits,
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id=f"cap-state-{strict_fixed_caps}")
    summary = result.daily_reports[-1]
    cap_state = summary.get("cap_state") or {}
    policy = cap_state.get("policy") or {}
    derived = cap_state.get("derived") or {}
    resolved = cap_state.get("resolved") or {}
    flags = cap_state.get("flags") or {}
    assert policy["max_trades_per_day"] == 30
    assert policy["max_triggers_per_symbol_per_day"] == 40
    assert derived["max_trades_per_day"] == 10
    assert derived["max_triggers_per_symbol_per_day"] == 10
    if strict_fixed_caps:
        assert resolved["max_trades_per_day"] == 30
        assert resolved["max_triggers_per_symbol_per_day"] == 40
        assert flags.get("strict_fixed_caps") is True
    else:
        assert resolved["max_trades_per_day"] == 10
        assert resolved["max_triggers_per_symbol_per_day"] == 10
        assert flags.get("strict_fixed_caps") is False
    # Session caps should be derived from resolved caps when present; none configured here.
    assert cap_state.get("session_caps") == {}


def test_exit_orders_map_to_plan_triggers(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="True",
                category="trend_continuation",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long", "flat"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_exit")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="test-run-exit")
    assert result.fills.shape[0] >= 1
    exit_fills = [reason for reason in result.fills["reason"].tolist() if reason.endswith("_exit")]
    assert exit_fills, "Expected at least one exit fill routed through the execution engine"


def test_flatten_daily_zeroes_overnight_exposure(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_flatten")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
        flatten_positions_daily=True,
    )
    result = backtester.run(run_id="test-run-flatten")
    report = result.daily_reports[-1]
    assert report["flatten_positions_daily"] is True
    assert all(abs(entry["quantity"]) < 1e-9 for entry in report["overnight_exposure"].values())


def test_factor_exposures_in_reports(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="buy_breakout",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
                category="trend_continuation",
            )
        ],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=5.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)
        ],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],
        allowed_trigger_categories=["trend_continuation"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    factor_index = list(market_data["BTC-USD"]["1h"].index)
    factor_df = pd.DataFrame({"market": [0.0] * len(factor_index)}, index=factor_index)
    run_registry = StrategyRunRegistry(tmp_path / "runs_factor")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
        factor_data=factor_df,
        auto_hedge_market=True,
    )
    result = backtester.run(run_id="test-run-factor")
    summary = result.daily_reports[-1]
    assert "factor_exposures" in summary
    assert result.summary["run_summary"].get("factor_exposures") == summary["factor_exposures"]
