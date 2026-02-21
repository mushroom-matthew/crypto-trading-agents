from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

from agents.strategies.plan_provider import StrategyPlanProvider
from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput as StrategistInput,
    PortfolioState,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


class DummyLLMClient:
    def __init__(self, plan: StrategyPlan) -> None:
        self.plan = plan

    def generate_plan(self, llm_input, prompt_template=None, **kwargs):
        return self.plan.model_copy(deep=True)


def _llm_input() -> StrategistInput:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=ts, close=30000.0)
    asset = AssetState(symbol="BTC-USD", indicators=[snapshot], trend_state="uptrend", vol_state="normal")
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
    return StrategistInput(portfolio=portfolio, assets=[asset], risk_params={"max_position_risk_pct": 1.0})


def _plan_without_limits() -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trigger = TriggerCondition(
        id="t1",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="False",
        stop_loss_pct=2.0,
    )
    return StrategyPlan(
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
        max_trades_per_day=None,
    )


def test_plan_provider_persists_enriched_limits(tmp_path, monkeypatch):
    plan = _plan_without_limits()
    provider = StrategyPlanProvider(DummyLLMClient(plan), cache_dir=tmp_path, llm_calls_per_day=1)
    llm_input = _llm_input()
    enriched = provider.get_plan("run-1", datetime(2024, 1, 1, tzinfo=timezone.utc), llm_input)
    assert enriched.max_trades_per_day is not None
    assert enriched.allowed_symbols == ["BTC-USD"]
    assert enriched.allowed_directions == ["long", "short"]
    cache_files = list(tmp_path.rglob("*.json"))
    assert cache_files
    saved = json.loads(cache_files[0].read_text())
    assert saved["max_trades_per_day"] == enriched.max_trades_per_day
    assert saved["allowed_symbols"] == ["BTC-USD"]


def test_plan_provider_fixed_caps_preserve_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRADES", "30")
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", "30")
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "true")
    plan = _plan_without_limits()
    plan.risk_constraints.max_daily_risk_budget_pct = 10.0
    provider = StrategyPlanProvider(DummyLLMClient(plan), cache_dir=tmp_path, llm_calls_per_day=1)
    enriched = provider.get_plan("run-fixed", datetime(2024, 1, 1, tzinfo=timezone.utc), _llm_input())
    assert enriched.max_trades_per_day == 30
    assert enriched.max_triggers_per_symbol_per_day == 30
    assert getattr(enriched, "_derived_trade_cap") == 10
    cap_inputs = getattr(enriched, "_cap_inputs", {})
    assert cap_inputs.get("risk_budget_pct") == 10.0
    assert cap_inputs.get("per_trade_risk_pct") == 1.0


def test_plan_provider_legacy_caps_apply_derivation(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRADES", "30")
    monkeypatch.setenv("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", "30")
    monkeypatch.setenv("STRATEGIST_STRICT_FIXED_CAPS", "false")
    plan = _plan_without_limits()
    plan.risk_constraints.max_daily_risk_budget_pct = 10.0
    provider = StrategyPlanProvider(DummyLLMClient(plan), cache_dir=tmp_path, llm_calls_per_day=1)
    enriched = provider.get_plan("run-legacy", datetime(2024, 1, 1, tzinfo=timezone.utc), _llm_input())
    assert enriched.max_trades_per_day == 10
    assert enriched.max_triggers_per_symbol_per_day == 10
    assert getattr(enriched, "_derived_trade_cap") == 10
