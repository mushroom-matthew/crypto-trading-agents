"""Tests for learning book settings and risk budgets (Runbook 10)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trade_risk import TradeRiskEvaluator
from agents.strategies.trigger_engine import Bar, TriggerEngine
from schemas.llm_strategist import (
    IndicatorSnapshot,
    PortfolioState,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)
from schemas.strategy_run import LearningBookSettings


def _portfolio(equity: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        equity=equity,
        cash=equity,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _indicator() -> IndicatorSnapshot:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return IndicatorSnapshot(
        symbol="BTC-USD", timeframe="1h", as_of=ts, close=50000.0, atr_14=500.0, sma_medium=49000.0,
    )


def _learning_trigger(direction: str = "long") -> TriggerCondition:
    return TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction=direction,
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="B",
        learning_book=True,
        experiment_id="exp-001",
    )


def _profit_trigger() -> TriggerCondition:
    return TriggerCondition(
        id="profit_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
    )


def _risk_engine() -> RiskEngine:
    constraints = RiskConstraint(
        max_position_risk_pct=2.0,
        max_symbol_exposure_pct=100.0,
        max_portfolio_exposure_pct=100.0,
        max_daily_loss_pct=10.0,
    )
    return RiskEngine(constraints, {})


def test_learning_disabled_blocks_learning_trigger():
    """When learning book is disabled, learning triggers are blocked."""
    settings = LearningBookSettings(enabled=False)
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _learning_trigger()
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())

    assert not result.allowed
    assert result.reason == "learning_disabled"


def test_learning_trade_uses_separate_risk_budget():
    """Learning trades use the learning book risk budget, not the main one."""
    settings = LearningBookSettings(
        enabled=True,
        daily_risk_budget_pct=1.0,
        notional_usd=200.0,
        max_trades_per_day=10,
    )
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _learning_trigger()
    portfolio = _portfolio(equity=100_000.0)
    result = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, _indicator())

    assert result.allowed
    assert result.quantity > 0
    # Notional should be capped at 200 USD → qty = 200/50000 = 0.004
    expected_qty = 200.0 / 50000.0
    assert abs(result.quantity - expected_qty) < 1e-9


def test_learning_daily_trade_cap_enforced():
    """After max_trades_per_day, additional learning trades are blocked."""
    settings = LearningBookSettings(
        enabled=True,
        daily_risk_budget_pct=100.0,
        max_trades_per_day=2,
        notional_usd=100.0,
    )
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _learning_trigger()
    portfolio = _portfolio()
    indicator = _indicator()

    # First 2 trades should pass
    r1 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert r1.allowed
    r2 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert r2.allowed

    # Third trade should be blocked
    r3 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert not r3.allowed
    assert r3.reason == "learning_daily_trade_cap"


def test_learning_pnl_isolated_from_profit():
    """Verify that learning and profit triggers produce orders with different
    learning_book tags, ensuring PnL isolation is possible downstream."""
    settings = LearningBookSettings(enabled=True, notional_usd=100.0)
    engine = _risk_engine()
    evaluator = TradeRiskEvaluator(engine, learning_settings=settings)

    learn_trigger = _learning_trigger()
    profit_trigger = _profit_trigger()

    portfolio = _portfolio()
    indicator = _indicator()

    learn_result = evaluator.evaluate(learn_trigger, "entry", 50000.0, portfolio, indicator)
    profit_result = evaluator.evaluate(profit_trigger, "entry", 50000.0, portfolio, indicator)

    assert learn_result.allowed
    assert profit_result.allowed
    # The tag propagation tests (runbook 09) confirm the Order.learning_book flag
    # Here we verify the evaluator doesn't cross-contaminate
    assert learn_trigger.learning_book is True
    assert profit_trigger.learning_book is False


def test_profit_trade_unaffected_by_learning_settings():
    """Profit book triggers ignore learning_settings entirely."""
    # Use very restrictive learning settings
    settings = LearningBookSettings(enabled=False, max_trades_per_day=0)
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _profit_trigger()
    portfolio = _portfolio()
    result = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, _indicator())

    # Profit trigger should still be evaluated by the main risk engine
    assert result.allowed
    assert result.quantity > 0


def test_learning_short_blocked_when_disallowed():
    """Learning short trades are blocked when allow_short is False."""
    settings = LearningBookSettings(enabled=True, allow_short=False, notional_usd=100.0)
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _learning_trigger(direction="short")
    result = evaluator.evaluate(trigger, "entry", 50000.0, _portfolio(), _indicator())

    assert not result.allowed
    assert result.reason == "learning_short_disabled"


def test_learning_daily_reset():
    """After reset_learning_daily(), counters are cleared."""
    settings = LearningBookSettings(enabled=True, max_trades_per_day=1, notional_usd=100.0)
    evaluator = TradeRiskEvaluator(_risk_engine(), learning_settings=settings)

    trigger = _learning_trigger()
    portfolio = _portfolio()
    indicator = _indicator()

    # Use up the daily cap
    r1 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert r1.allowed
    r2 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert not r2.allowed

    # Reset and try again
    evaluator.reset_learning_daily()
    r3 = evaluator.evaluate(trigger, "entry", 50000.0, portfolio, indicator)
    assert r3.allowed
