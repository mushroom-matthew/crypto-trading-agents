from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from agents.strategies.llm_client import LLMClient
from agents.strategies.plan_provider import LLMCostTracker, StrategyPlanProvider
from backtesting.llm_strategist_runner import LLMStrategistBacktester
from schemas.llm_strategist import PositionSizingRule, RiskConstraint, StrategyPlan, TriggerCondition
from services.strategy_run_registry import StrategyRunRegistry
from tools import execution_tools
from trading_core.execution_engine import ExecutionEngine


def _build_candles() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h", tz=timezone.utc)
    data = {
        "timestamp": timestamps,
        "open": [100 + i for i in range(len(timestamps))],
        "high": [101 + i for i in range(len(timestamps))],
        "low": [99 + i for i in range(len(timestamps))],
        "close": [100 + i for i in range(len(timestamps))],
        "volume": [1000 + i for i in range(len(timestamps))],
    }
    return pd.DataFrame(data).set_index("timestamp")


class _StubPlanProvider:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan
        self.cost_tracker = LLMCostTracker()
        self.cache_dir = Path(".cache/strategy_plans")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_plan(self, run_id, plan_date, llm_input, prompt_template=None):  # noqa: D401
        return self.plan

    def _cache_path(self, run_id, plan_date, llm_input):
        ident = f"{run_id}_{plan_date.isoformat().replace(':', '-')}"
        return self.cache_dir / f"{ident}.json"


class _DummyAsset:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _DummyInput:
    def __init__(self, symbols: list[str]) -> None:
        self.assets = [_DummyAsset(symbol) for symbol in symbols]


def _risk_params() -> dict[str, float]:
    return {
        "max_position_risk_pct": 5.0,
        "max_symbol_exposure_pct": 50.0,
        "max_portfolio_exposure_pct": 80.0,
        "max_daily_loss_pct": 3.0,
    }


def test_exit_direction_not_blocked(monkeypatch, tmp_path):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    plan = StrategyPlan(
        generated_at=now,
        valid_until=now + timedelta(days=1),
        global_view="test",
        regime="range",
        triggers=[
            TriggerCondition(
                id="enter_long",
                symbol="BTC-USD",
                direction="long",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
                category="trend_continuation",
            ),
            TriggerCondition(
                id="emergency_exit",
                symbol="BTC-USD",
                direction="exit",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="timeframe=='1h'",
                category="emergency_exit",
            ),
        ],
        risk_constraints=RiskConstraint(**_risk_params()),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long", "short"],  # exit will be auto-added during enrichment
        allowed_trigger_categories=["trend_continuation", "emergency_exit"],
    )
    market_data = {"BTC-USD": {"1h": _build_candles()}}
    run_registry = StrategyRunRegistry(tmp_path / "runs_exit_direction")
    monkeypatch.setattr(execution_tools, "registry", run_registry)
    monkeypatch.setattr(execution_tools, "engine", ExecutionEngine())
    backtester = LLMStrategistBacktester(
        pairs=["BTC-USD"],
        start=None,
        end=None,
        initial_cash=1000.0,
        fee_rate=0.0,
        llm_client=LLMClient(allow_fallback=True),
        cache_dir=Path(".cache/strategy_plans"),
        llm_calls_per_day=1,
        risk_params=_risk_params(),
        plan_provider=_StubPlanProvider(plan),
        market_data=market_data,
        run_registry=run_registry,
        min_hold_hours=0.0,
        min_flat_hours=0.0,
    )
    result = backtester.run(run_id="direction-exit-ok")
    # Exit should not be blocked by allowed_directions and should flatten the position.
    assert abs(result.final_positions.get("BTC-USD", 0.0)) < 1e-9
    exit_reasons = [reason for reason in result.fills["reason"].tolist() if reason.endswith("_flat") or reason.endswith("_exit")]
    assert exit_reasons, "Expected an exit fill to execute"
    report = result.daily_reports[-1]
    assert report["limit_stats"]["blocked_by_direction"] == 0


def test_invalid_direction_rejected(tmp_path):
    provider = StrategyPlanProvider(llm_client=LLMClient(allow_fallback=True), cache_dir=tmp_path / "plans", llm_calls_per_day=1)
    plan = StrategyPlan(
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 1, 2, tzinfo=timezone.utc),
        global_view="invalid",
        regime="range",
        triggers=[
            TriggerCondition(
                id="bad_trigger",
                symbol="BTC-USD",
                direction="short",
                timeframe="1h",
                entry_rule="timeframe=='1h'",
                exit_rule="False",
                category="trend_continuation",
            )
        ],
        risk_constraints=RiskConstraint(**_risk_params()),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=5.0)],
        max_trades_per_day=5,
        allowed_symbols=["BTC-USD"],
        allowed_directions=["long"],  # disallow short; should raise on enrichment
        allowed_trigger_categories=["trend_continuation"],
    )
    dummy_input = _DummyInput(symbols=["BTC-USD"])
    with pytest.raises(ValueError):
        provider._enrich_plan(plan, dummy_input)  # type: ignore[arg-type]
