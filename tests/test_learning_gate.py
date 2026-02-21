"""Tests for learning gate evaluator and kill switches (Runbook 12)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.strategies.learning_gate import LearningGateEvaluator
from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trade_risk import TradeRiskEvaluator
from agents.strategies.trigger_engine import Bar, TriggerEngine
from schemas.learning_gate import LearningGateStatus, LearningGateThresholds, LearningKillSwitchConfig
from schemas.llm_strategist import (
    IndicatorSnapshot,
    PortfolioState,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)
from schemas.strategy_run import LearningBookSettings


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        equity=100_000.0,
        cash=100_000.0,
        positions={},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _indicator() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        close=50000.0,
        atr_14=500.0,
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


def _bar() -> Bar:
    return Bar(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=50000.0,
        high=50050.0,
        low=49950.0,
        close=50000.0,
        volume=1.0,
    )


# =============================================================================
# Gate evaluator tests
# =============================================================================


def test_gate_allows_when_all_clear():
    """Gate is open when no conditions are violated."""
    evaluator = LearningGateEvaluator()
    status = evaluator.evaluate(
        equity=100_000.0,
        realized_vol=0.5,
        median_vol=0.5,
        volume=1000.0,
        median_volume=1000.0,
        spread_pct=0.1,
    )
    assert status.open is True
    assert status.reasons == []


def test_gate_blocks_on_volatility_spike():
    """Gate closes when volatility exceeds the spike threshold."""
    thresholds = LearningGateThresholds(volatility_spike_multiple=2.0)
    evaluator = LearningGateEvaluator(thresholds=thresholds)
    status = evaluator.evaluate(
        equity=100_000.0,
        realized_vol=5.0,
        median_vol=2.0,  # 5/2 = 2.5x >= 2.0 threshold
        volume=1000.0,
        median_volume=1000.0,
    )
    assert status.open is False
    assert "volatility_spike" in status.reasons


def test_gate_blocks_on_thin_liquidity():
    """Gate closes when volume is below the thin liquidity threshold."""
    thresholds = LearningGateThresholds(liquidity_thin_volume_multiple=0.3)
    evaluator = LearningGateEvaluator(thresholds=thresholds)
    status = evaluator.evaluate(
        equity=100_000.0,
        volume=200.0,
        median_volume=1000.0,  # 200/1000 = 0.2 <= 0.3 threshold
    )
    assert status.open is False
    assert "liquidity_thin" in status.reasons


def test_gate_blocks_on_wide_spread():
    """Gate closes when spread exceeds threshold."""
    thresholds = LearningGateThresholds(spread_wide_pct=0.5)
    evaluator = LearningGateEvaluator(thresholds=thresholds)
    status = evaluator.evaluate(
        equity=100_000.0,
        spread_pct=0.6,
    )
    assert status.open is False
    assert "spread_wide" in status.reasons


def test_kill_switch_daily_loss():
    """Kill switch fires when cumulative learning loss exceeds daily limit."""
    kill_config = LearningKillSwitchConfig(daily_loss_limit_pct=0.5)
    evaluator = LearningGateEvaluator(kill_switches=kill_config)

    # Record losses totaling 0.5% of 100k = $500
    evaluator.record_learning_trade(-300.0)
    evaluator.record_learning_trade(-250.0)  # Total $550 > $500

    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is False
    assert "daily_loss_limit" in status.reasons

    # Once killed, stays closed even with no further losses
    status2 = evaluator.evaluate(equity=100_000.0)
    assert status2.open is False
    assert "kill_switch_active" in status2.reasons


def test_kill_switch_consecutive_losses():
    """Kill switch fires after N consecutive learning losses."""
    kill_config = LearningKillSwitchConfig(
        consecutive_loss_limit=3,
        daily_loss_limit_pct=100.0,  # High so only consecutive matters
    )
    evaluator = LearningGateEvaluator(kill_switches=kill_config)

    evaluator.record_learning_trade(-10.0)
    evaluator.record_learning_trade(-10.0)
    # Not yet 3
    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is True

    evaluator.record_learning_trade(-10.0)  # 3rd consecutive loss
    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is False
    assert "consecutive_loss_limit" in status.reasons


def test_winning_trade_resets_consecutive_counter():
    """A winning trade resets the consecutive loss counter."""
    kill_config = LearningKillSwitchConfig(
        consecutive_loss_limit=3,
        daily_loss_limit_pct=100.0,
    )
    evaluator = LearningGateEvaluator(kill_switches=kill_config)

    evaluator.record_learning_trade(-10.0)
    evaluator.record_learning_trade(-10.0)
    evaluator.record_learning_trade(5.0)  # Win resets counter
    evaluator.record_learning_trade(-10.0)

    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is True  # Only 1 consecutive loss after the win


def test_daily_reset_clears_kill_switch():
    """reset_daily() clears kill switch state."""
    kill_config = LearningKillSwitchConfig(
        consecutive_loss_limit=2,
        daily_loss_limit_pct=100.0,
    )
    evaluator = LearningGateEvaluator(kill_switches=kill_config)

    evaluator.record_learning_trade(-10.0)
    evaluator.record_learning_trade(-10.0)  # Kill switch fires
    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is False

    evaluator.reset_daily()
    status = evaluator.evaluate(equity=100_000.0)
    assert status.open is True


# =============================================================================
# Trigger engine integration tests
# =============================================================================


def test_profit_trades_ignore_gate():
    """Profit-book triggers (learning_book=False) pass even when learning gate is closed."""
    profit_trigger = TriggerCondition(
        id="profit_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
        learning_book=False,
        stop_loss_pct=2.0,
    )
    plan = _plan_with_triggers([profit_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)

    closed_gate = LearningGateStatus(open=False, reasons=["volatility_spike"])

    orders, blocks = engine.on_bar(
        _bar(), _indicator(), _portfolio(),
        learning_gate_status=closed_gate,
    )
    assert orders  # Profit trigger should still fire
    assert orders[0].learning_book is False
    # No learning_gate_closed blocks
    gate_blocks = [b for b in blocks if b["reason"] == "learning_gate_closed"]
    assert not gate_blocks


def test_learning_trigger_blocked_by_closed_gate():
    """Learning triggers are blocked when the learning gate is closed."""
    learning_trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="B",
        learning_book=True,
        stop_loss_pct=2.0,
    )
    plan = _plan_with_triggers([learning_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    engine = TriggerEngine(plan, risk_engine, min_hold_bars=0, trade_cooldown_bars=0)

    closed_gate = LearningGateStatus(open=False, reasons=["volatility_spike", "liquidity_thin"])

    orders, blocks = engine.on_bar(
        _bar(), _indicator(), _portfolio(),
        learning_gate_status=closed_gate,
    )
    assert not orders
    gate_blocks = [b for b in blocks if b["reason"] == "learning_gate_closed"]
    assert len(gate_blocks) == 1
    assert "volatility_spike" in gate_blocks[0]["detail"]
    assert "liquidity_thin" in gate_blocks[0]["detail"]


def test_learning_trigger_passes_when_gate_open():
    """Learning triggers pass when the gate is open."""
    learning_trigger = TriggerCondition(
        id="learn_btc",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="True",
        exit_rule="",
        category="trend_continuation",
        confidence_grade="A",
        learning_book=True,
        stop_loss_pct=2.0,
    )
    plan = _plan_with_triggers([learning_trigger])
    risk_engine = RiskEngine(plan.risk_constraints, {})
    learning_settings = LearningBookSettings(enabled=True, notional_usd=500.0)
    trade_risk = TradeRiskEvaluator(risk_engine, learning_settings=learning_settings)
    engine = TriggerEngine(plan, risk_engine, trade_risk=trade_risk, min_hold_bars=0, trade_cooldown_bars=0)

    open_gate = LearningGateStatus(open=True, reasons=[])

    orders, blocks = engine.on_bar(
        _bar(), _indicator(), _portfolio(),
        learning_gate_status=open_gate,
    )
    assert orders
    assert orders[0].learning_book is True
