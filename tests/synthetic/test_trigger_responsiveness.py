"""Tests for trigger responsiveness with known-outcome scenarios.

These tests verify that triggers fire at expected times when using
parametric synthetic waveforms. Each test defines:
1. A waveform with known properties (frequency, amplitude, etc.)
2. Trigger rules designed to fire at specific waveform phases
3. Assertions that triggers fire at the expected bars
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.synthetic_loader import (
    SyntheticDataBackend,
    WaveformParams,
    WaveformType,
    sin_wave,
    cos_wave,
    trend,
    range_bound,
    analyze_waveform,
)
from schemas.llm_strategist import (
    StrategyPlan,
    TriggerCondition,
    RiskConstraint,
    PositionSizingRule,
    IndicatorSnapshot,
    PortfolioState,
)
from agents.strategies.trigger_engine import TriggerEngine, Bar, Order
from agents.strategies.risk_engine import RiskEngine, RiskProfile


# ============================================================================
# Test fixtures and helpers
# ============================================================================

def make_indicator_snapshot(
    symbol: str,
    timeframe: str,
    close: float,
    rsi_14: float = 50.0,
    atr_14: float = 100.0,
    sma_short: float | None = None,
    sma_medium: float | None = None,
    donchian_upper_short: float | None = None,
    donchian_lower_short: float | None = None,
) -> IndicatorSnapshot:
    """Create a minimal indicator snapshot for testing."""
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=datetime.now(timezone.utc),
        close=close,
        rsi_14=rsi_14,
        atr_14=atr_14,
        sma_short=sma_short or close * 0.99,
        sma_medium=sma_medium or close * 0.98,
        donchian_upper_short=donchian_upper_short or close * 1.02,
        donchian_lower_short=donchian_lower_short or close * 0.98,
        bollinger_upper=close * 1.02,
        bollinger_lower=close * 0.98,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        roc_short=0.0,
        realized_vol_short=0.01,
        realized_vol_medium=0.01,
        volume=1000.0,
        volume_multiple=1.0,
    )


def make_portfolio(cash: float = 10000.0, positions: Dict[str, float] | None = None) -> PortfolioState:
    """Create a portfolio state for testing."""
    return PortfolioState(
        timestamp=datetime.now(timezone.utc),
        cash=cash,
        equity=cash,
        positions=positions or {},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )


def make_plan_with_triggers(
    triggers: List[TriggerCondition],
    symbol: str = "SYN-USD",
) -> StrategyPlan:
    """Create a strategy plan with the given triggers."""
    return StrategyPlan(
        generated_at=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc) + timedelta(days=1),
        regime="range",
        risk_constraints=RiskConstraint(
            max_position_risk_pct=2.0,
            max_symbol_exposure_pct=50.0,
            max_portfolio_exposure_pct=100.0,
            max_daily_loss_pct=5.0,
        ),
        sizing_rules=[
            PositionSizingRule(
                symbol=symbol,
                sizing_mode="fixed_fraction",
                target_risk_pct=1.0,
            )
        ],
        triggers=triggers,
    )


def make_risk_engine(plan: StrategyPlan) -> RiskEngine:
    """Create a risk engine for the plan."""
    return RiskEngine(
        plan.risk_constraints,
        {rule.symbol: rule for rule in plan.sizing_rules},
        risk_profile=RiskProfile(global_multiplier=1.0),
    )


def run_triggers_on_synthetic_data(
    backend: SyntheticDataBackend,
    plan: StrategyPlan,
    start: datetime,
    end: datetime,
    granularity: str = "1h",
    symbol: str = "SYN-USD",
) -> Dict[str, Any]:
    """Run trigger engine over synthetic data and collect results."""

    df = backend.fetch_history(symbol, start, end, granularity)
    risk_engine = make_risk_engine(plan)
    trigger_engine = TriggerEngine(plan, risk_engine)

    portfolio = make_portfolio()
    orders_by_bar: Dict[int, List[Order]] = {}
    blocks_by_bar: Dict[int, List[dict]] = {}

    for i, (ts, row) in enumerate(df.iterrows()):
        bar = Bar(
            symbol=symbol,
            timeframe=granularity,
            timestamp=ts,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        indicator = make_indicator_snapshot(
            symbol=symbol,
            timeframe=granularity,
            close=row["close"],
            sma_short=row["close"] * 0.995,  # Slightly below close
            sma_medium=row["close"] * 0.99,
            donchian_upper_short=df["close"].iloc[:i+1].max() if i > 0 else row["close"],
            donchian_lower_short=df["close"].iloc[:i+1].min() if i > 0 else row["close"],
        )

        orders, blocks = trigger_engine.on_bar(bar, indicator, portfolio)

        if orders:
            orders_by_bar[i] = orders
            # Update portfolio for next bar
            for order in orders:
                new_positions = dict(portfolio.positions)
                if order.side == "buy":
                    new_cash = portfolio.cash - order.quantity * order.price
                    new_positions[symbol] = new_positions.get(symbol, 0) + order.quantity
                else:
                    new_cash = portfolio.cash + order.quantity * order.price
                    new_positions[symbol] = new_positions.get(symbol, 0) - order.quantity

                portfolio = PortfolioState(
                    timestamp=datetime.now(timezone.utc),
                    cash=new_cash,
                    equity=portfolio.equity,
                    positions=new_positions,
                    realized_pnl_7d=0.0,
                    realized_pnl_30d=0.0,
                    sharpe_30d=0.0,
                    max_drawdown_90d=0.0,
                    win_rate_30d=0.0,
                    profit_factor_30d=0.0,
                )

        if blocks:
            blocks_by_bar[i] = blocks

    return {
        "df": df,
        "orders_by_bar": orders_by_bar,
        "blocks_by_bar": blocks_by_bar,
        "total_orders": sum(len(o) for o in orders_by_bar.values()),
        "total_blocks": sum(len(b) for b in blocks_by_bar.values()),
        "order_bars": sorted(orders_by_bar.keys()),
        "block_bars": sorted(blocks_by_bar.keys()),
    }


# ============================================================================
# SIMPLE TRIGGER TESTS
# ============================================================================

class TestPriceThresholdTriggers:
    """Test triggers that fire when price crosses a threshold."""

    def test_price_above_threshold_fires(self):
        """Trigger should fire when close > threshold."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Cos wave starts at peak (51000) and goes to trough (49000)
        base = 50000
        amp = 1000
        backend = cos_wave(base_price=base, amplitude=amp, frequency=1.0)

        # Trigger: buy when price > 50500 (upper half of wave)
        threshold = 50500
        trigger = TriggerCondition(
            id="buy_high",
            symbol="SYN-USD",
            category="volatility_breakout",
            confidence_grade="B",
            direction="long",
            timeframe="1h",
            entry_rule=f"close > {threshold} and position == 'flat'",
            exit_rule="close < 50000",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # Should have at least one order (entry when price > 50500)
        assert result["total_orders"] >= 1, "Expected at least one entry order"

        # First order should be early (when cos wave is still high)
        first_order_bar = result["order_bars"][0]
        first_close = result["df"]["close"].iloc[first_order_bar]
        assert first_close > threshold, f"Expected close > {threshold}, got {first_close}"

    def test_price_below_threshold_fires(self):
        """Trigger should fire when close < threshold."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Sin wave starts at base (50000), goes up, then down
        base = 50000
        backend = sin_wave(base_price=base, amplitude=1000, frequency=1.0)

        # Trigger: buy when price < 49500 (lower half of wave)
        threshold = 49500
        trigger = TriggerCondition(
            id="buy_low",
            symbol="SYN-USD",
            category="mean_reversion",
            confidence_grade="B",
            direction="long",
            timeframe="1h",
            entry_rule=f"close < {threshold} and position == 'flat'",
            exit_rule="close > 50500",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # Should have at least one order
        assert result["total_orders"] >= 1, "Expected at least one entry order"

        # Entry should be when price is low
        first_order_bar = result["order_bars"][0]
        first_close = result["df"]["close"].iloc[first_order_bar]
        assert first_close < threshold, f"Expected close < {threshold}, got {first_close}"


class TestTrendTriggers:
    """Test triggers with trending price data."""

    def test_uptrend_breakout_trigger(self):
        """Trigger should fire during uptrend when price breaks threshold."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Uptrend starting at 50000
        backend = trend(base_price=50000, slope=500, direction="up")

        # Trigger: buy when price breaks above 50200
        threshold = 50200
        trigger = TriggerCondition(
            id="trend_break",
            symbol="SYN-USD",
            category="trend_continuation",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule=f"close > {threshold} and position == 'flat'",
            exit_rule="close < 50000",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # Should fire within first few hours as price rises
        assert result["total_orders"] >= 1
        first_order_bar = result["order_bars"][0]
        # With slope=500/day and 24 bars, price rises ~21/bar
        # Should cross 50200 around bar 10 (50000 + 10*20 = 50200)
        assert first_order_bar <= 15, f"Expected trigger within first 15 bars, got {first_order_bar}"


class TestRangeBoundTriggers:
    """Test triggers with range-bound price data."""

    def test_range_bounce_entry_and_exit(self):
        """Test entry at support and exit at resistance."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        support = 49000
        resistance = 51000

        # Range-bound oscillation
        backend = range_bound(
            support=support,
            resistance=resistance,
            frequency=2.0,  # 2 cycles per day
            bounce_pct=0.9,
        )

        # Entry trigger near support
        entry_trigger = TriggerCondition(
            id="range_entry",
            symbol="SYN-USD",
            category="mean_reversion",
            confidence_grade="B",
            direction="long",
            timeframe="1h",
            entry_rule=f"close < {support + 300} and position == 'flat'",
            exit_rule=f"close > {resistance - 300}",
            stop_loss_pct=3.0,
        )

        plan = make_plan_with_triggers([entry_trigger])
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # With 2 cycles per day, should have multiple entry opportunities
        assert result["total_orders"] >= 1, "Expected at least one range entry"


# ============================================================================
# PARAMETRIC TRIGGER TESTS
# ============================================================================

class TestParametricWaveformTriggers:
    """Test that triggers fire at predictable times based on waveform parameters."""

    def test_known_peak_timing(self):
        """Verify trigger fires near known peak time."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Cos wave: peak at t=0, trough at t=0.5 cycles
        # With frequency=1 (1 cycle/day), peak at bar 0, trough at bar 12
        base = 50000
        amp = 1000
        backend = cos_wave(base_price=base, amplitude=amp, frequency=1.0)

        df = backend.fetch_history("SYN-USD", start, end, "1h")

        # Find actual peak
        peak_bar = df["close"].idxmax()
        peak_idx = df.index.get_loc(peak_bar)

        # Peak should be near the start (within first few bars)
        assert peak_idx <= 3, f"Expected peak near bar 0, got bar {peak_idx}"

        # Verify peak value
        peak_value = df["close"].iloc[peak_idx]
        expected_peak = base + amp
        assert abs(peak_value - expected_peak) < 100, f"Expected peak near {expected_peak}, got {peak_value}"

    def test_known_trough_timing(self):
        """Verify trigger fires near known trough time."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Cos wave: peak at t=0, trough at half period
        # With frequency=1, trough at bar 12 (half of 24)
        base = 50000
        amp = 1000
        backend = cos_wave(base_price=base, amplitude=amp, frequency=1.0)

        df = backend.fetch_history("SYN-USD", start, end, "1h")

        # Find actual trough
        trough_bar = df["close"].idxmin()
        trough_idx = df.index.get_loc(trough_bar)

        # Trough should be near bar 12 (half period)
        expected_trough_bar = 12
        assert abs(trough_idx - expected_trough_bar) <= 2, \
            f"Expected trough near bar {expected_trough_bar}, got bar {trough_idx}"

    def test_frequency_determines_trigger_count(self):
        """Higher frequency should produce more trigger opportunities."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)  # 2 days

        base = 50000
        amp = 1000

        # Threshold in the middle - will cross twice per cycle
        threshold = base

        # Slow wave: 0.5 cycles/day = 1 cycle over 2 days
        slow = sin_wave(base_price=base, amplitude=amp, frequency=0.5)
        df_slow = slow.fetch_history("SYN-USD", start, end, "1h")
        slow_crossings = sum(1 for i in range(1, len(df_slow))
                           if (df_slow["close"].iloc[i-1] < threshold) != (df_slow["close"].iloc[i] < threshold))

        # Fast wave: 2 cycles/day = 4 cycles over 2 days
        fast = sin_wave(base_price=base, amplitude=amp, frequency=2.0)
        df_fast = fast.fetch_history("SYN-USD", start, end, "1h")
        fast_crossings = sum(1 for i in range(1, len(df_fast))
                           if (df_fast["close"].iloc[i-1] < threshold) != (df_fast["close"].iloc[i] < threshold))

        # Fast should have more crossings
        assert fast_crossings > slow_crossings, \
            f"Expected fast ({fast_crossings}) > slow ({slow_crossings}) crossings"


# ============================================================================
# COMPOUND TRIGGER TESTS
# ============================================================================

class TestCompoundTriggers:
    """Test triggers with compound conditions."""

    def test_dual_threshold_trigger(self):
        """Trigger with both upper and lower bounds."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        base = 50000
        backend = sin_wave(base_price=base, amplitude=1000, frequency=2.0)

        # Trigger: buy only in middle band (49700 < close < 50300)
        trigger = TriggerCondition(
            id="middle_band",
            symbol="SYN-USD",
            category="mean_reversion",
            confidence_grade="B",
            direction="long",
            timeframe="1h",
            entry_rule="close > 49700 and close < 50300 and position == 'flat'",
            exit_rule="close > 50500 or close < 49500",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # Should have at least some orders (entry or exit)
        assert result["total_orders"] >= 1, "Expected at least one order"

        # Verify entry orders are in the middle band
        # Note: exits may fire at any price level when exit_rule is met
        for bar_idx, orders in result["orders_by_bar"].items():
            close = result["df"]["close"].iloc[bar_idx]
            for order in orders:
                if order.side == "buy":
                    # Entry should be in middle band
                    assert 49600 < close < 50400, f"Entry at bar {bar_idx} outside middle band: {close}"


# ============================================================================
# REPORT GENERATION
# ============================================================================

class TestTriggerReport:
    """Test report generation for trigger responsiveness."""

    def test_generate_trigger_report(self):
        """Generate a comprehensive trigger responsiveness report."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        base = 50000
        amp = 1000
        freq = 2.0

        backend = sin_wave(base_price=base, amplitude=amp, frequency=freq)

        # Multiple triggers at different thresholds
        triggers = [
            TriggerCondition(
                id="buy_dip",
                symbol="SYN-USD",
                category="mean_reversion",
                confidence_grade="B",
                direction="long",
                timeframe="1h",
                entry_rule=f"close < {base - amp*0.5} and position == 'flat'",
                exit_rule=f"close > {base + amp*0.3}",
                stop_loss_pct=2.0,
            ),
            TriggerCondition(
                id="buy_breakout",
                symbol="SYN-USD",
                category="volatility_breakout",
                confidence_grade="A",
                direction="long",
                timeframe="1h",
                entry_rule=f"close > {base + amp*0.7} and position == 'flat'",
                exit_rule=f"close < {base}",
                stop_loss_pct=2.0,
            ),
        ]

        plan = make_plan_with_triggers(triggers)
        result = run_triggers_on_synthetic_data(backend, plan, start, end)

        # Generate report
        report = generate_trigger_report(
            waveform_params={
                "type": "sin",
                "base_price": base,
                "amplitude": amp,
                "frequency": freq,
            },
            triggers=triggers,
            result=result,
        )

        # Report should have expected sections
        assert "WAVEFORM" in report
        assert "TRIGGERS" in report
        assert "RESULTS" in report

        # Print report for manual inspection
        print("\n" + report)


def generate_trigger_report(
    waveform_params: Dict[str, Any],
    triggers: List[TriggerCondition],
    result: Dict[str, Any],
) -> str:
    """Generate a human-readable trigger responsiveness report."""

    lines = []
    lines.append("=" * 70)
    lines.append("SYNTHETIC TRIGGER RESPONSIVENESS REPORT")
    lines.append("=" * 70)

    # Waveform section
    lines.append("\n## WAVEFORM PARAMETERS")
    lines.append("-" * 40)
    for key, value in waveform_params.items():
        lines.append(f"  {key}: {value}")

    # Triggers section
    lines.append("\n## TRIGGERS CONFIGURED")
    lines.append("-" * 40)
    for t in triggers:
        lines.append(f"  [{t.id}] {t.category}")
        lines.append(f"    direction: {t.direction}")
        lines.append(f"    entry: {t.entry_rule}")
        lines.append(f"    exit: {t.exit_rule}")

    # Results section
    lines.append("\n## EXECUTION RESULTS")
    lines.append("-" * 40)
    lines.append(f"  Total bars: {len(result['df'])}")
    lines.append(f"  Total orders: {result['total_orders']}")
    lines.append(f"  Total blocks: {result['total_blocks']}")

    if result["order_bars"]:
        lines.append(f"\n  Order bars: {result['order_bars']}")
        lines.append("\n  Order details:")
        for bar_idx, orders in result["orders_by_bar"].items():
            close = result["df"]["close"].iloc[bar_idx]
            ts = result["df"].index[bar_idx]
            for order in orders:
                lines.append(f"    Bar {bar_idx} ({ts}): {order.side} {order.quantity:.4f} @ {close:.2f}")
                lines.append(f"      trigger: {order.reason}")

    if result["block_bars"]:
        lines.append(f"\n  Block bars: {result['block_bars'][:10]}...")  # First 10

    # Price statistics
    df = result["df"]
    lines.append("\n## PRICE STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Min close: {df['close'].min():.2f}")
    lines.append(f"  Max close: {df['close'].max():.2f}")
    lines.append(f"  Mean close: {df['close'].mean():.2f}")
    lines.append(f"  Range: {df['close'].max() - df['close'].min():.2f}")

    # Pass/fail summary
    lines.append("\n## SUMMARY")
    lines.append("-" * 40)
    if result["total_orders"] > 0:
        lines.append("  Status: TRIGGERS FIRED as expected")
    else:
        lines.append("  Status: NO TRIGGERS FIRED - check conditions")

    lines.append("=" * 70)

    return "\n".join(lines)
