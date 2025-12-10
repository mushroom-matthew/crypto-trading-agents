"""Integration test: generous budget should be consumed with many trade opportunities."""

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
    """Sideways/oscillating price path to generate both wins and losses."""

    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    base = 1000 + 10 * np.sin(np.linspace(0, 4 * np.pi, periods))
    noise = np.linspace(0, 5, periods)
    prices = pd.Series(base - noise, index=idx)
    return pd.DataFrame({"open": prices, "high": prices + 2, "low": prices - 2, "close": prices, "volume": 1}, index=idx)


def test_high_budget_activity_consumes_budget(tmp_path: Path) -> None:
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    provider = AlwaysLongPlanProvider(
        symbol="BTC-USD",
        timeframe="1h",
        stop_loss_pct=5.0,
        max_position_risk_pct=2.0,
        max_daily_loss_pct=10.0,  # Loose daily loss so budget is the main gate.
        max_daily_risk_budget_pct=8.0,
    )
    market_data = {"BTC-USD": {"1h": _wiggle(start, periods=48)}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_high_budget")
    risk_params = {
        "max_position_risk_pct": 2.0,
        "max_symbol_exposure_pct": 200.0,
        "max_portfolio_exposure_pct": 100.0,
        "max_daily_loss_pct": 10.0,
        "max_daily_risk_budget_pct": 8.0,
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
    )
    execution_tools.registry = run_registry
    backtester.risk_profile = RiskProfile(global_multiplier=1.0, symbol_multipliers={"BTC-USD": 1.0})
    backtester.plan_service.plan_provider = provider
    result = backtester.run(run_id="high-budget-activity")
    daily = result.daily_reports[0]

    risk_budget = daily.get("risk_budget") or {}
    used_pct = float(risk_budget.get("used_pct", daily.get("limit_stats", {}).get("risk_budget_used_pct", 0.0)))
    budget_abs = risk_budget.get("budget_abs", 0.0) or 0.0
    assert used_pct >= 3.0  # Should meaningfully spend budget (regression target > sub-5% regime).
    assert 0.0 <= used_pct <= 100.0

    # Monotone non-decreasing cumulative usage derived from risk_usage_events.
    events = sorted(daily.get("risk_usage_events", []), key=lambda e: (e.get("hour", 0), e.get("timeframe", "")))
    cumulative = []
    total = 0.0
    for evt in events:
        total += float(evt.get("risk_used", 0.0))
        cumulative.append(total)
    if budget_abs > 0 and cumulative:
        pct_seq = [val / budget_abs * 100.0 for val in cumulative]
        assert pct_seq == sorted(pct_seq)
        assert pct_seq[-1] <= 100.0 + 1e-6

    # Blocks (if any) should be risk budget later in the day; daily loss not expected.
    risk_breakdown = (daily.get("limit_stats", {}) or {}).get("risk_block_breakdown", {}) or {}
    assert risk_breakdown.get("max_daily_loss_pct", 0) == 0

    # Dual RPR metrics should be present and finite.
    run_quality = result.summary["run_summary"]["trigger_quality"] or {}
    if run_quality:
        sample = next(iter(run_quality.values()))
        assert "rpr_allocated" in sample and "rpr_actual" in sample
        assert not np.isinf(sample["rpr_allocated"])
        assert not np.isinf(sample["rpr_actual"])
