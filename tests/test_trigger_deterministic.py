"""Deterministic tests for trigger evaluation with known market conditions.

These tests verify that triggers fire correctly when market conditions
explicitly match the trigger rules. This helps identify issues with the
trigger evaluation logic independent of LLM-generated rules.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.strategies.risk_engine import RiskEngine
from agents.strategies.trigger_engine import Bar, TriggerEngine
from agents.strategies.rule_dsl import RuleEvaluator
from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    PortfolioState,
    PositionSizingRule,
    RiskConstraint,
    StrategyPlan,
    TriggerCondition,
)


def _portfolio(cash: float = 100000.0, positions: dict | None = None) -> PortfolioState:
    """Create a test portfolio state."""
    return PortfolioState(
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        equity=cash,
        cash=cash,
        positions=positions or {},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def _indicator(
    symbol: str = "BTC-USD",
    timeframe: str = "1h",
    close: float = 50000.0,
    **kwargs,
) -> IndicatorSnapshot:
    """Create a test indicator snapshot with customizable values."""
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        close=close,
        sma_short=kwargs.get("sma_short", close * 0.98),  # Default: below close
        sma_medium=kwargs.get("sma_medium", close * 0.95),
        ema_short=kwargs.get("ema_short", close * 0.99),
        ema_medium=kwargs.get("ema_medium", close * 0.96),
        rsi_14=kwargs.get("rsi_14", 55.0),
        macd=kwargs.get("macd", 50.0),
        macd_signal=kwargs.get("macd_signal", 40.0),
        macd_hist=kwargs.get("macd_hist", 10.0),
        atr_14=kwargs.get("atr_14", 500.0),
        bollinger_upper=kwargs.get("bollinger_upper", close * 1.02),
        bollinger_lower=kwargs.get("bollinger_lower", close * 0.98),
    )


def _asset_state(
    symbol: str = "BTC-USD",
    trend_state: str = "uptrend",
    vol_state: str = "normal",
    indicators: list[IndicatorSnapshot] | None = None,
) -> AssetState:
    """Create a test asset state."""
    return AssetState(
        symbol=symbol,
        trend_state=trend_state,
        vol_state=vol_state,
        indicators=indicators or [_indicator(symbol=symbol)],
    )


def _plan(
    triggers: list[TriggerCondition],
    max_position_risk_pct: float = 2.0,
) -> StrategyPlan:
    """Create a test strategy plan."""
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    symbols = list({t.symbol for t in triggers})
    return StrategyPlan(
        generated_at=now,
        valid_until=now,
        global_view="test",
        regime="range",
        run_id="test-run",
        triggers=triggers,
        risk_constraints=RiskConstraint(
            max_position_risk_pct=max_position_risk_pct,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[
            PositionSizingRule(symbol=s, sizing_mode="fixed_fraction", target_risk_pct=max_position_risk_pct)
            for s in symbols
        ],
    )


def _bar(
    symbol: str = "BTC-USD",
    timeframe: str = "1h",
    close: float = 50000.0,
) -> Bar:
    """Create a test bar."""
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=100.0,
    )


class TestRuleEvaluatorDeterministic:
    """Test the rule evaluator with known conditions."""

    def test_simple_comparison_true(self):
        """Test that a simple comparison evaluates correctly."""
        evaluator = RuleEvaluator()
        context = {"close": 50000.0, "sma_short": 49000.0}

        # close > sma_short should be True (50000 > 49000)
        assert evaluator.evaluate("close > sma_short", context) is True

    def test_simple_comparison_false(self):
        """Test that a simple comparison returns False when condition not met."""
        evaluator = RuleEvaluator()
        context = {"close": 48000.0, "sma_short": 49000.0}

        # close > sma_short should be False (48000 > 49000)
        assert evaluator.evaluate("close > sma_short", context) is False

    def test_compound_and_all_true(self):
        """Test that AND with all true conditions returns True."""
        evaluator = RuleEvaluator()
        context = {
            "close": 50000.0,
            "sma_short": 49000.0,
            "rsi_14": 55.0,
            "macd_hist": 10.0,
        }

        rule = "close > sma_short and rsi_14 > 50 and macd_hist > 0"
        assert evaluator.evaluate(rule, context) is True

    def test_compound_and_one_false(self):
        """Test that AND with one false condition returns False."""
        evaluator = RuleEvaluator()
        context = {
            "close": 50000.0,
            "sma_short": 49000.0,
            "rsi_14": 45.0,  # Below 50
            "macd_hist": 10.0,
        }

        rule = "close > sma_short and rsi_14 > 50 and macd_hist > 0"
        assert evaluator.evaluate(rule, context) is False

    def test_between_syntax(self):
        """Test the 'between' syntax works correctly."""
        evaluator = RuleEvaluator()
        context = {"rsi_14": 55.0}

        assert evaluator.evaluate("rsi_14 between 40 and 60", context) is True
        assert evaluator.evaluate("rsi_14 between 60 and 80", context) is False

    def test_position_filter(self):
        """Test position filter in rules."""
        evaluator = RuleEvaluator()

        context_flat = {"position": "flat", "close": 50000.0, "sma_short": 49000.0}
        context_long = {"position": "long", "close": 50000.0, "sma_short": 49000.0}

        rule = "position == 'flat' and close > sma_short"
        assert evaluator.evaluate(rule, context_flat) is True
        assert evaluator.evaluate(rule, context_long) is False

    def test_vol_state_filter(self):
        """Test vol_state filter in rules."""
        evaluator = RuleEvaluator()

        context_normal = {"vol_state": "normal", "close": 50000.0}
        context_high = {"vol_state": "high", "close": 50000.0}

        rule = "vol_state == 'normal' and close > 49000"
        assert evaluator.evaluate(rule, context_normal) is True
        assert evaluator.evaluate(rule, context_high) is False


class TestTriggerEngineDeterministic:
    """Test the trigger engine fires correctly with known conditions."""

    def test_simple_long_entry_fires(self):
        """Test that a simple long entry trigger fires when conditions are met."""
        trigger = TriggerCondition(
            id="test_long",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            category="trend_continuation",
            confidence_grade="A",
            # Simple rule: close above sma_short and not already long
            entry_rule="close > sma_short and position == 'flat'",
            exit_rule="close < sma_short",
            stop_loss_pct=2.0,
        )

        plan = _plan([trigger])
        risk_engine = RiskEngine(plan.risk_constraints, {trigger.symbol: plan.sizing_rules[0]})
        engine = TriggerEngine(plan, risk_engine)

        # Create conditions where close > sma_short
        bar = _bar(close=50000.0)
        indicator = _indicator(close=50000.0, sma_short=49000.0)  # close > sma_short
        portfolio = _portfolio()  # Flat position
        asset_state = _asset_state()

        orders, blocks = engine.on_bar(bar, indicator, portfolio, asset_state)

        # Should produce an order
        assert len(orders) > 0, f"Expected order but got blocks: {blocks}"
        assert orders[0].side == "buy"
        assert orders[0].symbol == "BTC-USD"

    def test_simple_long_entry_blocked_when_already_long(self):
        """Test that long entry is blocked when already in a long position."""
        trigger = TriggerCondition(
            id="test_long",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            category="trend_continuation",
            confidence_grade="A",
            entry_rule="close > sma_short and position == 'flat'",
            exit_rule="close < sma_short",
            stop_loss_pct=2.0,
        )

        plan = _plan([trigger])
        risk_engine = RiskEngine(plan.risk_constraints, {trigger.symbol: plan.sizing_rules[0]})
        engine = TriggerEngine(plan, risk_engine)

        bar = _bar(close=50000.0)
        indicator = _indicator(close=50000.0, sma_short=49000.0)
        portfolio = _portfolio(positions={"BTC-USD": 1.0})  # Already long
        asset_state = _asset_state()

        orders, blocks = engine.on_bar(bar, indicator, portfolio, asset_state)

        # Should NOT produce an order because position != 'flat'
        assert len(orders) == 0, f"Expected no orders but got: {orders}"

    def test_exit_trigger_fires_when_in_position(self):
        """Test that exit trigger fires when in a position and exit conditions met."""
        trigger = TriggerCondition(
            id="test_exit",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            category="trend_continuation",
            confidence_grade="A",
            entry_rule="close > sma_short and position == 'flat'",
            exit_rule="close < sma_short and position == 'long'",
            stop_loss_pct=2.0,
        )

        plan = _plan([trigger])
        risk_engine = RiskEngine(plan.risk_constraints, {trigger.symbol: plan.sizing_rules[0]})
        engine = TriggerEngine(plan, risk_engine)

        # Create conditions where close < sma_short (exit condition met)
        bar = _bar(close=48000.0)
        indicator = _indicator(close=48000.0, sma_short=49000.0)  # close < sma_short
        portfolio = _portfolio(positions={"BTC-USD": 1.0})  # In long position
        asset_state = _asset_state()

        orders, blocks = engine.on_bar(bar, indicator, portfolio, asset_state)

        # Should produce a sell order to exit
        assert len(orders) > 0, f"Expected exit order but got blocks: {blocks}"
        assert orders[0].side == "sell"

    def test_trigger_with_rsi_condition(self):
        """Test trigger with RSI condition fires correctly."""
        trigger = TriggerCondition(
            id="test_rsi_long",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            category="mean_reversion",
            confidence_grade="B",
            # Buy when RSI oversold and price above support
            entry_rule="rsi_14 < 35 and close > sma_medium and position == 'flat'",
            exit_rule="rsi_14 > 65",
            stop_loss_pct=2.0,
        )

        plan = _plan([trigger])
        risk_engine = RiskEngine(plan.risk_constraints, {trigger.symbol: plan.sizing_rules[0]})
        engine = TriggerEngine(plan, risk_engine)

        bar = _bar(close=50000.0)
        indicator = _indicator(
            close=50000.0,
            sma_medium=48000.0,  # close > sma_medium
            rsi_14=30.0,  # Oversold (< 35)
        )
        portfolio = _portfolio()
        asset_state = _asset_state()

        orders, blocks = engine.on_bar(bar, indicator, portfolio, asset_state)

        assert len(orders) > 0, f"Expected order for RSI oversold but got blocks: {blocks}"
        assert orders[0].side == "buy"

    def test_multi_timeframe_with_available_timeframes(self):
        """Test that multi-timeframe triggers work when timeframes are available."""
        trigger = TriggerCondition(
            id="test_mtf",
            symbol="BTC-USD",
            direction="long",
            timeframe="1h",
            category="trend_continuation",
            confidence_grade="A",
            # Use tf_4h_ prefix for 4h confirmation
            entry_rule="close > sma_short and tf_4h_close > tf_4h_sma_short and position == 'flat'",
            exit_rule="close < sma_short",
            stop_loss_pct=2.0,
        )

        plan = _plan([trigger])
        risk_engine = RiskEngine(plan.risk_constraints, {trigger.symbol: plan.sizing_rules[0]})
        engine = TriggerEngine(plan, risk_engine)

        bar = _bar(close=50000.0)
        indicator_1h = _indicator(timeframe="1h", close=50000.0, sma_short=49000.0)
        indicator_4h = _indicator(timeframe="4h", close=51000.0, sma_short=49500.0)

        # Asset state with multiple timeframe indicators
        asset_state = _asset_state(indicators=[indicator_1h, indicator_4h])
        portfolio = _portfolio()

        orders, blocks = engine.on_bar(bar, indicator_1h, portfolio, asset_state)

        # Should produce an order because both timeframes confirm
        assert len(orders) > 0, f"Expected order with MTF confirmation but got blocks: {blocks}"


class TestTriggerValidation:
    """Test trigger validation for common issues."""

    def test_detect_unavailable_timeframe(self):
        """Test that we can detect triggers referencing unavailable timeframes."""
        from trading_core.trigger_compiler import extract_referenced_timeframes, validate_trigger_timeframes

        trigger = TriggerCondition(
            id="test_bad_mtf",
            symbol="BTC-USD",
            direction="long",
            timeframe="5m",
            category="trend_continuation",
            entry_rule="close > sma_short and tf_4h_close > tf_4h_sma_short",
            exit_rule="tf_1h_rsi_14 < 30",
        )

        # Only 5m is available
        available = {"5m"}
        warnings = validate_trigger_timeframes(trigger, available)

        assert len(warnings) == 2  # Entry references 4h, exit references 1h
        missing_tfs = {tf for w in warnings for tf in w.missing_timeframes}
        assert "4h" in missing_tfs
        assert "1h" in missing_tfs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
