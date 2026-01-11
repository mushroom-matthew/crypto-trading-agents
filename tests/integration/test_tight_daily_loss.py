"""Integration test: tight daily loss cap halts new entries after threshold."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.execution_tools as execution_tools
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from services.strategy_run_registry import StrategyRunRegistry
from agents.strategies.llm_client import LLMClient
from agents.strategies.risk_engine import RiskProfile
from tests.helpers.stub_plan_provider import AlwaysLongPlanProvider


def _downtrend(start: datetime, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    step = pd.Series(range(periods)).to_numpy()
    prices = pd.Series(1000 - (step * 20), index=idx)  # accelerated decline to ensure loss threshold is hit
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1}, index=idx)
    return df


def test_tight_daily_loss_stops_new_entries(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=5.0,
        max_position_risk_pct=10.0,
        max_daily_loss_pct=1.0,
        max_daily_risk_budget_pct=None,
    )
    market_data = {"BTC-USD": {"1h": _downtrend(start, periods=24)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_daily_loss")
    risk_params = {
        "max_position_risk_pct": 10.0,
        "max_symbol_exposure_pct": 100.0,
        "max_portfolio_exposure_pct": 100.0,
        "max_daily_loss_pct": 1.0,
        "max_daily_risk_budget_pct": None,
    }
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=start,
        end=start + timedelta(days=1),
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(allow_fallback=True),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=8,
        risk_params=risk_params,
        plan_provider=provider,
        market_data=market_data,
        run_registry=run_registry,
        prompt_template_path=None,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    # Ensure the deterministic execution tool shares the same registry as this run.
    execution_tools.registry = run_registry
    # Force unity risk profile to avoid size being zeroed by multipliers.
    backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
    backtester.plan_service.plan_provider = provider
    result = backtester.run(run_id="tight-daily-loss")
    daily = result.daily_reports[0]
    # Ensure loss threshold crossed
    assert daily["equity_return_pct"] < 0
    # Risk block breakdown should record daily loss
    risk_breakdown = (daily.get("limit_stats", {}) or {}).get("risk_block_breakdown", {}) or {}
    # risk_breakdown keys capture max_daily_loss_pct under BlockReason.RISK bucket
    assert risk_breakdown.get("max_daily_loss_pct", 0) > 0
    # Once triggered, executed trades should be fewer than attempts.
    assert daily["executed_trades"] < daily["attempted_triggers"]
