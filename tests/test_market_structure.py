from datetime import timezone

import pandas as pd

from agents.analytics.market_structure import (
    LevelTestEvent,
    MarketStructureTelemetry,
    SupportLevel,
    build_market_structure_snapshot,
    compute_support_resistance_levels,
    detect_level_tests,
    find_swing_points,
    infer_market_structure_state,
)


def _frame_from_prices(prices: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(prices), freq="h", tz=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.6 for p in prices],
            "low": [p - 0.6 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
    ).set_index("timestamp")


def test_market_structure_identifies_uptrend_and_levels():
    prices = [100, 102, 101, 103, 102, 105, 104, 106, 105, 108]
    df = _frame_from_prices(prices)

    swings = find_swing_points(df, left=1, right=1)
    assert swings, "expected swing highs/lows to be detected"
    state = infer_market_structure_state(swings)
    assert state.trend == "uptrend"

    supports, resistances, atr_val = compute_support_resistance_levels(
        df, lookback=len(df), swing_window=1, tolerance_mult=0.75, atr_period=3
    )
    assert supports and resistances
    assert atr_val is None or atr_val > 0

    telemetry = build_market_structure_snapshot(
        df,
        symbol="BTC-USD",
        timeframe="1h",
        lookback=len(df),
        swing_window=1,
        tolerance_mult=0.75,
    )
    assert isinstance(telemetry, MarketStructureTelemetry)
    assert telemetry.trend == "uptrend"
    assert telemetry.nearest_support is not None
    assert telemetry.nearest_resistance is not None
    assert telemetry.distance_to_support_pct is not None


def test_detect_level_tests_and_reclaims():
    prices = [101.0, 100.5, 99.8, 100.2, 100.8]
    df = _frame_from_prices(prices)
    support = SupportLevel(price=100.0, width=0.75)

    events = detect_level_tests(df, supports=[support], resistances=[], base_buffer=0.5, lookback=len(df))
    assert events, "expected at least one test event"
    # Ensure ordering and reclaim detection
    assert isinstance(events[0], LevelTestEvent)
    results = [event.result for event in events]
    assert "breakdown" in results or "failed_test" in results
    assert "reclaim" in results
    assert events[-1].attempts >= 1
