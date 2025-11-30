from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from agents.strategies.plan_provider import LLMCostTracker
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from agents.strategies.llm_client import LLMClient
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition


class StubPlanProvider:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan
        self.cost_tracker = LLMCostTracker()
        self.cache_dir = Path(".cache/strategy_plans")

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None):  # noqa: D401
        return self.plan


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


def test_backtester_executes_trigger(monkeypatch):
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
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
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
    )
    result = backtester.run(run_id="test-run")
    assert result.final_positions["BTC-USD"] > 0
    assert result.fills.shape[0] == 1
    assert result.daily_reports
    assert "judge_feedback" in result.daily_reports[-1]
    assert "strategist_constraints" in result.daily_reports[-1]["judge_feedback"]
