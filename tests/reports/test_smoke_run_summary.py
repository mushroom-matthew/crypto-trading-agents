"""Smoke test: one-day synthetic backtest produces sane run_summary telemetry."""

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


def _wiggle(start: datetime, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    base = 1000 + 8 * np.sin(np.linspace(0, 2 * np.pi, periods))
    prices = pd.Series(base, index=idx)
    return pd.DataFrame({"open": prices, "high": prices + 2, "low": prices - 2, "close": prices, "volume": 1}, index=idx)


def test_smoke_run_summary_contains_budget_blocks_and_rpr(tmp_path: Path) -> None:
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=3.0,
        max_position_risk_pct=2.0,
        max_daily_loss_pct=10.0,
        max_daily_risk_budget_pct=15.0,
    )
    market_data = {"BTC-USD": {"1h": _wiggle(start, periods=24)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_smoke_summary")
    risk_params = {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 100.0,
        "max_portfolio_exposure_pct": 100.0,
        "max_daily_loss_pct": 10.0,
        "max_daily_risk_budget_pct": 15.0,
    }
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=start,
        end=start + timedelta(days=1),
        initial_cash=1500.0,
        fee_rate=0.0,
        llm_client=LLMClient(allow_fallback=True),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=6,
        risk_params=risk_params,
        plan_provider=provider,
        market_data=market_data,
        run_registry=run_registry,
        prompt_template_path=None,
    )
    execution_tools.registry = run_registry
    backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
    backtester.plan_service.plan_provider = provider
    result = backtester.run(run_id="smoke-summary")
    run_summary = result.summary["run_summary"]

    # Budget metrics present and bounded.
    mean_used = run_summary["risk_budget_used_pct_mean"]
    median_used = run_summary["risk_budget_used_pct_median"]
    assert 0.0 <= mean_used <= 100.0
    assert 0.0 <= median_used <= 100.0

    # Block totals include canonical keys and are non-negative.
    blocks = run_summary["block_totals"]
    for key in [
        "max_daily_loss_pct",
        "max_daily_risk_budget_pct",
        "max_daily_cap",
        "max_symbol_exposure_pct",
        "session_cap",
        "archetype_load",
        "trigger_load",
    ]:
        assert key in blocks
        assert blocks[key] >= 0

    # Trigger quality exposes dual RPR metrics.
    tq = run_summary.get("trigger_quality") or {}
    assert tq, "trigger_quality should not be empty in smoke backtest"
    sample = next(iter(tq.values()))
    assert "rpr_allocated" in sample and "rpr_actual" in sample
    assert np.isfinite(sample["rpr_allocated"])
    assert np.isfinite(sample["rpr_actual"])
