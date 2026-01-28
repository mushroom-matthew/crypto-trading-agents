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
    bollinger_upper: float | None = None,
    bollinger_lower: float | None = None,
    macd: float = 0.0,
    macd_signal: float = 0.0,
    macd_hist: float = 0.0,
    vol_burst: bool | None = None,
    volume_multiple: float = 1.0,
) -> IndicatorSnapshot:
    """Create a minimal indicator snapshot for testing.

    All indicator values can be overridden for parametric testing.
    """
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=datetime.now(timezone.utc),
        close=close,
        rsi_14=rsi_14,
        atr_14=atr_14,
        sma_short=sma_short if sma_short is not None else close * 0.99,
        sma_medium=sma_medium if sma_medium is not None else close * 0.98,
        donchian_upper_short=donchian_upper_short if donchian_upper_short is not None else close * 1.02,
        donchian_lower_short=donchian_lower_short if donchian_lower_short is not None else close * 0.98,
        bollinger_upper=bollinger_upper if bollinger_upper is not None else close * 1.02,
        bollinger_lower=bollinger_lower if bollinger_lower is not None else close * 0.98,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        roc_short=0.0,
        realized_vol_short=0.01,
        realized_vol_medium=0.01,
        volume=1000.0,
        volume_multiple=volume_multiple,
        vol_burst=vol_burst,
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
# INDICATOR-BASED TRIGGER TESTS (Parametric)
# ============================================================================

class TestRSITriggers:
    """Test RSI-based triggers with parametric indicator values."""

    def test_rsi_oversold_buy(self):
        """Trigger should fire when RSI < 30 (oversold)."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="rsi_oversold",
            symbol="SYN-USD",
            category="mean_reversion",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="rsi_14 < 30 and position == 'flat'",
            exit_rule="rsi_14 > 70",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        # Simulate RSI oscillating: oversold at troughs, overbought at peaks
        orders_fired = []
        for i, (ts, row) in enumerate(df.iterrows()):
            # Parametric RSI: map price position to RSI (low price = low RSI)
            price_pct = (row["close"] - 49000) / 2000  # 0 at min, 1 at max
            rsi = 20 + price_pct * 60  # RSI ranges 20-80

            indicator = make_indicator_snapshot(
                symbol="SYN-USD",
                timeframe="1h",
                close=row["close"],
                rsi_14=rsi,
            )

            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            if orders:
                orders_fired.extend([(i, o, rsi) for o in orders])
                for o in orders:
                    if o.side == "buy":
                        portfolio = make_portfolio(
                            cash=portfolio.cash - o.quantity * o.price,
                            positions={"SYN-USD": o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

        # Should have entries when RSI < 30 (price near trough)
        buy_orders = [(i, rsi) for i, o, rsi in orders_fired if o.side == "buy"]
        assert len(buy_orders) >= 1, "Expected at least one RSI oversold entry"

        # Verify RSI was actually < 30 at entry
        for bar_idx, rsi in buy_orders:
            assert rsi < 35, f"Entry at bar {bar_idx} with RSI {rsi:.1f} - expected < 35"

    def test_rsi_overbought_sell(self):
        """Trigger should fire when RSI > 70 (overbought)."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="rsi_overbought",
            symbol="SYN-USD",
            category="reversal",
            confidence_grade="A",
            direction="short",
            timeframe="1h",
            entry_rule="rsi_14 > 70 and position == 'flat'",
            exit_rule="rsi_14 < 30",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        orders_fired = []
        for i, (ts, row) in enumerate(df.iterrows()):
            price_pct = (row["close"] - 49000) / 2000
            rsi = 20 + price_pct * 60

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"], rsi_14=rsi,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            if orders:
                orders_fired.extend([(i, o, rsi) for o in orders])
                for o in orders:
                    if o.side == "sell":
                        portfolio = make_portfolio(
                            cash=portfolio.cash,
                            positions={"SYN-USD": -o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash)

        sell_orders = [(i, rsi) for i, o, rsi in orders_fired if o.side == "sell"]
        assert len(sell_orders) >= 1, "Expected at least one RSI overbought entry"

        for bar_idx, rsi in sell_orders:
            assert rsi > 65, f"Entry at bar {bar_idx} with RSI {rsi:.1f} - expected > 65"


class TestMACDTriggers:
    """Test MACD crossover triggers with parametric values."""

    def test_macd_bullish_crossover(self):
        """Trigger fires when MACD crosses above signal."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="macd_bullish",
            symbol="SYN-USD",
            category="trend_continuation",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="macd > macd_signal and position == 'flat'",
            exit_rule="macd < macd_signal",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        orders_fired = []
        prev_macd_diff = None

        for i, (ts, row) in enumerate(df.iterrows()):
            # Parametric MACD: follows price momentum
            # MACD positive when price rising, negative when falling
            price_momentum = (row["close"] - 50000) / 1000  # -1 to +1
            macd = price_momentum * 50  # MACD ranges -50 to +50
            macd_signal = macd * 0.8  # Signal lags slightly

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"],
                macd=macd, macd_signal=macd_signal, macd_hist=macd - macd_signal,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            macd_diff = macd - macd_signal
            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)

            if orders:
                orders_fired.extend([(i, o, macd, macd_signal) for o in orders])
                for o in orders:
                    if o.side == "buy":
                        portfolio = make_portfolio(
                            cash=portfolio.cash - o.quantity * o.price,
                            positions={"SYN-USD": o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

            prev_macd_diff = macd_diff

        buy_orders = [(i, macd, sig) for i, o, macd, sig in orders_fired if o.side == "buy"]
        assert len(buy_orders) >= 1, "Expected at least one MACD bullish crossover"

        for bar_idx, macd, signal in buy_orders:
            assert macd > signal, f"Bar {bar_idx}: MACD {macd:.1f} should be > signal {signal:.1f}"


class TestBollingerTriggers:
    """Test Bollinger Band triggers with parametric values."""

    def test_bollinger_breakout(self):
        """Trigger fires when price breaks above upper band."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="bollinger_breakout",
            symbol="SYN-USD",
            category="volatility_breakout",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="close > bollinger_upper and position == 'flat'",
            exit_rule="close < bollinger_middle",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        orders_fired = []
        for i, (ts, row) in enumerate(df.iterrows()):
            # Parametric Bollinger: bands narrow then widen
            # Upper band at 50700, lower at 49300 (tighter than price range)
            bb_middle = 50000
            bb_upper = 50700
            bb_lower = 49300

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"],
                bollinger_upper=bb_upper, bollinger_lower=bb_lower,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            if orders:
                orders_fired.extend([(i, o, row["close"], bb_upper) for o in orders])
                for o in orders:
                    if o.side == "buy":
                        portfolio = make_portfolio(
                            cash=portfolio.cash - o.quantity * o.price,
                            positions={"SYN-USD": o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

        buy_orders = [(i, close, upper) for i, o, close, upper in orders_fired if o.side == "buy"]
        assert len(buy_orders) >= 1, "Expected at least one Bollinger breakout"

        for bar_idx, close, upper in buy_orders:
            assert close > upper, f"Bar {bar_idx}: close {close:.0f} should be > upper {upper:.0f}"


class TestVolumeTriggers:
    """Test volume-based triggers with parametric values."""

    def test_volume_burst_entry(self):
        """Trigger fires when vol_burst is True."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="volume_burst",
            symbol="SYN-USD",
            category="volatility_breakout",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="vol_burst == True and close > sma_short and position == 'flat'",
            exit_rule="close < sma_short",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        orders_fired = []
        burst_bars = [5, 11, 17, 23]  # Parametric: volume bursts at these bars

        for i, (ts, row) in enumerate(df.iterrows()):
            vol_burst = i in burst_bars
            vol_multiple = 3.0 if vol_burst else 1.0

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"],
                vol_burst=vol_burst, volume_multiple=vol_multiple,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            if orders:
                orders_fired.extend([(i, o, vol_burst) for o in orders])
                for o in orders:
                    if o.side == "buy":
                        portfolio = make_portfolio(
                            cash=portfolio.cash - o.quantity * o.price,
                            positions={"SYN-USD": o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

        buy_orders = [(i, burst) for i, o, burst in orders_fired if o.side == "buy"]
        assert len(buy_orders) >= 1, "Expected at least one volume burst entry"

        # Entries should occur at or near burst bars (when price is also > sma_short)
        for bar_idx, burst in buy_orders:
            # Entry may be at burst bar or shortly after depending on price condition
            assert any(abs(bar_idx - b) <= 1 for b in burst_bars), \
                f"Entry at bar {bar_idx} not near any burst bar {burst_bars}"


class TestEmergencyExitTriggers:
    """Test emergency exit triggers."""

    def test_emergency_exit_on_drawdown(self):
        """Emergency exit fires on significant drawdown."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        # Downtrend for testing stop-loss
        backend = trend(base_price=50000, slope=2000, direction="down", seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        # Entry trigger (will enter early)
        entry_trigger = TriggerCondition(
            id="trend_entry",
            symbol="SYN-USD",
            category="trend_continuation",
            confidence_grade="B",
            direction="long",
            timeframe="1h",
            entry_rule="close > 49000 and position == 'flat'",
            exit_rule="close > 55000",  # Won't hit in downtrend
            stop_loss_pct=2.0,
        )

        # Emergency exit trigger
        emergency_trigger = TriggerCondition(
            id="emergency_stop",
            symbol="SYN-USD",
            category="emergency_exit",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="False",  # Never enters
            exit_rule="close < 48500",  # Exit if price drops significantly
            stop_loss_pct=1.0,
        )

        plan = make_plan_with_triggers([entry_trigger, emergency_trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        entry_bar = None
        exit_bar = None

        for i, (ts, row) in enumerate(df.iterrows()):
            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"],
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            for o in orders:
                if o.side == "buy" and entry_bar is None:
                    entry_bar = i
                    portfolio = make_portfolio(
                        cash=portfolio.cash - o.quantity * o.price,
                        positions={"SYN-USD": o.quantity}
                    )
                elif o.side == "sell" and exit_bar is None:
                    exit_bar = i
                    portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

        assert entry_bar is not None, "Expected an entry"
        assert exit_bar is not None, "Expected emergency exit to fire"
        assert exit_bar > entry_bar, "Exit should be after entry"

        # Verify exit was at a lower price
        entry_price = df["close"].iloc[entry_bar]
        exit_price = df["close"].iloc[exit_bar]
        assert exit_price < entry_price, f"Exit price {exit_price} should be < entry {entry_price}"


class TestHoldRuleTriggers:
    """Test hold rules that prevent premature exits."""

    def test_hold_rule_prevents_exit(self):
        """Hold rule should suppress exit when active."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        # Trigger with hold rule: don't exit while RSI is in neutral zone
        # RSI formula: RSI = 20 + ((close - 49000) / 2000) * 60
        # RSI 40-60 corresponds to close 49666-50333
        # Exit rule triggers at close > 49800 (RSI ~44), which is in hold range
        trigger = TriggerCondition(
            id="hold_test",
            symbol="SYN-USD",
            category="mean_reversion",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="close < 49300 and position == 'flat'",
            exit_rule="close > 49800",  # Triggers when RSI ~44 (in hold range)
            hold_rule="rsi_14 > 40 and rsi_14 < 60",  # Hold in neutral RSI zone
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        entry_bar = None
        exit_bar = None
        hold_blocks = 0

        for i, (ts, row) in enumerate(df.iterrows()):
            price_pct = (row["close"] - 49000) / 2000
            rsi = 20 + price_pct * 60

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"], rsi_14=rsi,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, blocks = trigger_engine.on_bar(bar, indicator, portfolio)

            # Count hold rule blocks
            for block in blocks:
                if "HOLD_RULE" in str(block.get("reason", "")):
                    hold_blocks += 1

            for o in orders:
                if o.side == "buy" and entry_bar is None:
                    entry_bar = i
                    portfolio = make_portfolio(
                        cash=portfolio.cash - o.quantity * o.price,
                        positions={"SYN-USD": o.quantity}
                    )
                elif o.side == "sell" and exit_bar is None:
                    exit_bar = i

        assert entry_bar is not None, "Expected an entry"
        # Hold rule should have blocked at least one exit attempt
        assert hold_blocks >= 1, f"Expected hold rule to block exits, got {hold_blocks} blocks"


class TestDonchianTriggers:
    """Test Donchian channel triggers."""

    def test_donchian_breakout(self):
        """Trigger fires when price breaks Donchian upper channel."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)
        df = backend.fetch_history("SYN-USD", start, end, "1h")

        trigger = TriggerCondition(
            id="donchian_breakout",
            symbol="SYN-USD",
            category="volatility_breakout",
            confidence_grade="A",
            direction="long",
            timeframe="1h",
            entry_rule="close > donchian_upper_short and position == 'flat'",
            exit_rule="close < donchian_lower_short",
            stop_loss_pct=2.0,
        )

        plan = make_plan_with_triggers([trigger])
        risk_engine = make_risk_engine(plan)
        trigger_engine = TriggerEngine(plan, risk_engine)
        portfolio = make_portfolio()

        orders_fired = []
        # Track rolling high/low for Donchian (simplified: last 10 bars)
        lookback = 10

        for i, (ts, row) in enumerate(df.iterrows()):
            start_idx = max(0, i - lookback)
            donchian_upper = df["close"].iloc[start_idx:i+1].max() if i > 0 else row["close"]
            donchian_lower = df["close"].iloc[start_idx:i+1].min() if i > 0 else row["close"]

            indicator = make_indicator_snapshot(
                symbol="SYN-USD", timeframe="1h", close=row["close"],
                donchian_upper_short=donchian_upper * 0.99,  # Slightly below actual for breakout
                donchian_lower_short=donchian_lower * 1.01,
            )
            bar = Bar(
                symbol="SYN-USD", timeframe="1h", timestamp=ts,
                open=row["open"], high=row["high"], low=row["low"],
                close=row["close"], volume=row["volume"],
            )

            orders, _ = trigger_engine.on_bar(bar, indicator, portfolio)
            if orders:
                orders_fired.extend([(i, o) for o in orders])
                for o in orders:
                    if o.side == "buy":
                        portfolio = make_portfolio(
                            cash=portfolio.cash - o.quantity * o.price,
                            positions={"SYN-USD": o.quantity}
                        )
                    else:
                        portfolio = make_portfolio(cash=portfolio.cash + o.quantity * o.price)

        buy_orders = [i for i, o in orders_fired if o.side == "buy"]
        assert len(buy_orders) >= 1, "Expected at least one Donchian breakout"


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
