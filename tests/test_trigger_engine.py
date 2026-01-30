from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trigger_engine import Bar, TriggerEngine
from schemas.llm_strategist import IndicatorSnapshot, PortfolioState, RiskConstraint, StrategyPlan, TriggerCondition


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
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


def _portfolio_with_position() -> PortfolioState:
    state = _portfolio()
    state.positions = {"BTC-USD": 1.0}
    return state


def _indicator() -> IndicatorSnapshot:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=ts, close=50000.0, atr_14=500.0, sma_medium=49000.0)


def _plan(trigger: TriggerCondition) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        triggers=[trigger],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=0.0,
            max_symbol_exposure_pct=5.0,
            max_portfolio_exposure_pct=50.0,
            max_daily_loss_pct=3.0,
        ),
    )


def _plan_with_triggers(triggers: list[TriggerCondition]) -> StrategyPlan:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        triggers=triggers,
        risk_constraints=RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=100.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=10.0,
        ),
    )


def test_trigger_engine_records_block_when_risk_denies_entry():
    trigger = TriggerCondition(
        id="btc_long",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(symbol="BTC-USD", timeframe="1h", timestamp=_portfolio().timestamp, open=50000.0, high=50050.0, low=49950.0, close=50000.0, volume=1.0)

    orders, blocks = engine.on_bar(bar, _indicator(), _portfolio())
    assert not orders
    assert blocks
    assert blocks[0]["reason"] in {"max_position_risk_pct", "sizing_zero"}


def test_emergency_exit_trigger_bypasses_risk_checks():
    trigger = TriggerCondition(
        id="btc_exit",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="false",
        exit_rule="timeframe=='1h'",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(symbol="BTC-USD", timeframe="1h", timestamp=_portfolio().timestamp, open=50000.0, high=50050.0, low=49950.0, close=50000.0, volume=1.0)
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert orders  # flatten order should be produced even though limits are zero
    assert not blocks
    assert orders[0].side == "sell"


def test_emergency_exit_vetoes_same_bar_entry():
    trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
    )
    plan = _plan(trigger)
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, trade_cooldown_bars=0)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    engine.record_fill("BTC-USD", is_entry=True, timestamp=bar.timestamp)
    portfolio = _portfolio_with_position()

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not orders
    assert blocks
    assert blocks[0]["reason"] == "emergency_exit_veto_same_bar"


def test_emergency_exit_dedup_overrides_high_conf_entry():
    entry_trigger = TriggerCondition(
        id="btc_entry",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="False",
        category="trend_continuation",
        confidence_grade="A",
    )
    emergency_trigger = TriggerCondition(
        id="btc_emergency_exit",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="False",
        exit_rule="True",
        category="emergency_exit",
        confidence_grade="A",
    )
    plan = _plan_with_triggers([entry_trigger, emergency_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, max_triggers_per_symbol_per_bar=2)
    bar = Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=_portfolio().timestamp,
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )
    portfolio = _portfolio()
    portfolio.positions = {"BTC-USD": -1.0}

    orders, blocks = engine.on_bar(bar, _indicator(), portfolio)
    assert not blocks
    assert len(orders) == 1
    assert orders[0].emergency is True
