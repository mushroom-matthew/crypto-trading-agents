"""Unit check: risk budget commits actual risk at stop, not theoretical sizing."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
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
from agents.strategies.trigger_engine import Order
from tests.helpers.stub_plan_provider import AlwaysLongPlanProvider


def _trend(start: datetime, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    prices = pd.Series(1000 + np.linspace(0, 10, periods), index=idx)
    return pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1}, index=idx)


def _bt_stub(start_equity: float = 10000.0, max_position_risk_pct: float = 2.5) -> LLMStrategistBacktester:
    """Create a minimal backtester stub for unit testing _commit_risk_budget()."""
    _cap_pct = max_position_risk_pct
    fake_limits = type("_FakeLimits", (), {"max_position_risk_pct": _cap_pct})()
    bt = LLMStrategistBacktester.__new__(LLMStrategistBacktester)  # type: ignore
    bt.daily_risk_budget_pct = 15.0
    bt.daily_risk_budget_state = {}
    bt.latest_daily_summary = None
    bt.initial_cash = start_equity
    bt.active_risk_limits = fake_limits
    bt.portfolio = type("PF", (), {"equity_records": [], "portfolio_state": lambda self, ts: None})()
    return bt


def test_deducts_actual_not_theoretical() -> None:
    """_commit_risk_budget() should deduct actual_risk_at_stop, not theoretical cap."""
    # equity=10000, max_position_risk_pct=2.5 → per_trade_cap=250
    # actual_risk_at_stop = 5.00 (tight stop, small position) — should charge 5, not 250
    bt = _bt_stub(start_equity=10000.0, max_position_risk_pct=2.5)
    bt._reset_risk_budget_for_day("2024-01-01", start_equity=10000.0)
    bt._commit_risk_budget("2024-01-01", 5.00, "BTC-USD")
    used = bt.daily_risk_budget_state["2024-01-01"]["used_abs"]
    assert used == pytest.approx(5.00, abs=1e-6), f"Expected 5.00 charged, got {used}"


def test_capped_deduction_when_contribution_exceeds_per_trade_cap() -> None:
    """If contribution exceeds per_trade_cap (shouldn't happen after fallback fix), clamp to cap."""
    bt = _bt_stub(start_equity=10000.0, max_position_risk_pct=2.5)  # cap = 250
    bt._reset_risk_budget_for_day("2024-01-01", start_equity=10000.0)
    bt._commit_risk_budget("2024-01-01", 300.0, "BTC-USD")
    used = bt.daily_risk_budget_state["2024-01-01"]["used_abs"]
    assert used <= 250.0, f"Deduction should be capped at per_trade_cap (250), got {used}"


def test_budget_gate_blocks_when_exhausted() -> None:
    """Risk gate returns None when budget is exhausted after commits."""
    bt = _bt_stub(start_equity=10000.0, max_position_risk_pct=2.5)
    bt._reset_risk_budget_for_day("2024-01-01", start_equity=10000.0)
    # Exhaust budget manually: budget_abs = 10000 * 15% = 1500
    entry = bt.daily_risk_budget_state["2024-01-01"]
    entry["used_abs"] = entry["budget_abs"]
    order = Order(
        symbol="BTC-USD", side="buy", quantity=0.01, price=10000.0,
        timeframe="1h", reason="test", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    result = bt._risk_budget_gate("2024-01-01", order)
    assert result is None, "Exhausted budget should return None (block trade)"


def test_overcharge_ratio_in_daily_report(tmp_path: Path) -> None:
    """Daily report includes risk_overcharge_ratio_median when trades execute."""
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=1.0,
        max_position_risk_pct=2.0,
        max_daily_loss_pct=5.0,
        max_daily_risk_budget_pct=20.0,
    )
    market_data = {"BTC-USD": {"1h": _trend(start, periods=6)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_overcharge_ratio")
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
        llm_calls_per_day=2,
        risk_params=risk_params,
        plan_provider=provider,
        market_data=market_data,
        run_registry=run_registry,
        prompt_template_path=None,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    _orig_registry = execution_tools.registry
    try:
        execution_tools.registry = run_registry
        backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
        backtester.plan_service.plan_provider = provider
        result = backtester.run(run_id="risk-overcharge-ratio-test")
    finally:
        execution_tools.registry = _orig_registry
    report = result.daily_reports[0]
    # Fields must be present in report (may be None if no trades with stop data)
    assert "risk_overcharge_ratio_median" in report
    assert "risk_overcharge_ratio_max" in report


def test_risk_budget_commits_actual_risk(tmp_path: Path) -> None:
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=1.0,
        max_position_risk_pct=2.0,
        max_daily_loss_pct=5.0,
        max_daily_risk_budget_pct=20.0,
    )
    market_data = {"BTC-USD": {"1h": _trend(start, periods=6)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_actual_risk_unit")
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
        llm_calls_per_day=2,
        risk_params=risk_params,
        plan_provider=provider,
        market_data=market_data,
        run_registry=run_registry,
        prompt_template_path=None,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    _orig_registry = execution_tools.registry
    try:
        execution_tools.registry = run_registry
        backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
        backtester.plan_service.plan_provider = provider
        result = backtester.run(run_id="risk-budget-actual-unit")
    finally:
        execution_tools.registry = _orig_registry
    events = result.daily_reports[0].get("risk_usage_events", [])
    assert events, "expected risk_usage_events to be populated"
    for evt in events:
        actual = evt.get("actual_risk_at_stop") or 0.0
        if actual > 0:
            used = evt.get("risk_used") or 0.0
            assert used == pytest.approx(actual, rel=1e-6, abs=1e-6)
