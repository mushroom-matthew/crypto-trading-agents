"""Integration check: risk_usage_events include actual_risk_at_stop."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.execution_tools as execution_tools
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from services.strategy_run_registry import StrategyRunRegistry
from agents.strategies.llm_client import LLMClient
from agents.strategies.risk_engine import RiskProfile
from tests.helpers.stub_plan_provider import AlwaysLongPlanProvider


def _trend(start: datetime, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    prices = pd.Series(1000 + np.linspace(0, 30, periods), index=idx)
    return pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1}, index=idx)


def test_risk_usage_events_capture_actual_risk(tmp_path: Path) -> None:
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=2.0,
        max_position_risk_pct=2.0,
        max_daily_loss_pct=5.0,
        max_daily_risk_budget_pct=20.0,
    )
    market_data = {"BTC-USD": {"1h": _trend(start, periods=12)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_actual_risk")
    risk_params = {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 100.0,
        "max_portfolio_exposure_pct": 100.0,
        "max_daily_loss_pct": 5.0,
        "max_daily_risk_budget_pct": 20.0,
    }
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=start,
        end=start + timedelta(days=1),
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(allow_fallback=True),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=4,
        risk_params=risk_params,
        plan_provider=provider,
        market_data=market_data,
        run_registry=run_registry,
        prompt_template_path=None,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    execution_tools.registry = run_registry
    backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
    backtester.plan_service.plan_provider = provider
    result = backtester.run(run_id="actual-risk-capture")
    events = result.daily_reports[0].get("risk_usage_events", [])
    assert events, "expected risk_usage_events to be populated"
    assert any((evt.get("actual_risk_at_stop") or 0) > 0 for evt in events), "actual_risk_at_stop should be recorded"
