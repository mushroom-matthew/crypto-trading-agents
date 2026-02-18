"""Unit tests for candlestick pattern feature computation.

Each test uses a hand-crafted OHLC row and verifies the expected pattern
fires (1.0) or does not fire (0.0) for both the individual function and
the composite compute_candlestick_features() entry point.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import candlestick as cs


def make_df(rows: list) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of (O, H, L, C, V) tuples."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Scalar feature tests
# ---------------------------------------------------------------------------

def test_body_pct_marubozu():
    """Full-body bar (no wicks) should have body_pct close to 1.0."""
    df = make_df([(100, 110, 100, 110, 1000)])
    result = cs.body_pct(df)
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_body_pct_doji():
    """Open == close (zero body) should have body_pct == 0.0."""
    df = make_df([(100, 105, 95, 100, 1000)])
    result = cs.body_pct(df)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_upper_wick_pct_only_upper():
    """Bar with only an upper wick: O=C=low, H above them."""
    df = make_df([(100, 110, 100, 100, 1000)])
    result = cs.upper_wick_pct(df)
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_lower_wick_pct_only_lower():
    """Bar with only a lower wick: O=C=high, L below them."""
    df = make_df([(100, 100, 90, 100, 1000)])
    result = cs.lower_wick_pct(df)
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_candle_strength_impulse():
    """Body = 4, ATR = 4 → strength = 1.0."""
    df = make_df([(100, 105, 99, 104, 1000)])
    atr = pd.Series([4.0])
    result = cs.candle_strength(df, atr)
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_candle_strength_zero_atr():
    """Zero ATR should not produce NaN or raise — falls back to 0.0."""
    df = make_df([(100, 105, 99, 104, 1000)])
    atr = pd.Series([0.0])
    result = cs.candle_strength(df, atr)
    assert not np.isnan(result.iloc[-1])


# ---------------------------------------------------------------------------
# Directional flags
# ---------------------------------------------------------------------------

def test_is_bullish():
    df = make_df([(100, 105, 99, 103, 1000)])
    result = cs.is_bullish(df)
    assert result.iloc[-1] == 1.0


def test_is_bearish():
    df = make_df([(103, 105, 99, 100, 1000)])
    result = cs.is_bearish(df)
    assert result.iloc[-1] == 1.0


def test_is_bullish_and_bearish_mutually_exclusive():
    df = make_df([(100, 105, 99, 103, 1000)])
    assert cs.is_bullish(df).iloc[-1] == 1.0
    assert cs.is_bearish(df).iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# Single-bar reversal patterns
# ---------------------------------------------------------------------------

def test_doji_detected():
    """Body = 0.5, range = 10 → body_pct = 0.05 < 0.10 → doji."""
    df = make_df([(100, 105, 95, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_doji"].iloc[-1] == 1.0


def test_doji_not_detected_for_full_body():
    """Marubozu should not be a doji."""
    df = make_df([(100, 110, 100, 110, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_doji"].iloc[-1] == 0.0


def test_hammer_detected():
    """Long lower wick (12), body (0.8), tiny upper wick (0.2) — upper < body required."""
    df = make_df([(100, 101, 88, 100.8, 1000)])
    # lower = min(100, 100.8) - 88 = 12, upper = 101 - max(100, 100.8) = 0.2, body = 0.8
    # conditions: 12 > 2*0.8=1.6 ✓, 0.2 < 0.8 ✓, body > 0 ✓
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_hammer"].iloc[-1] == 1.0, "Expected hammer to be detected"


def test_hammer_not_detected_for_shooting_star():
    """Shooting star shape should NOT be a hammer."""
    df = make_df([(100, 112, 99, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_hammer"].iloc[-1] == 0.0


def test_shooting_star_detected():
    """Long upper wick (11.5), body (0.5), tiny lower wick (0.4) — lower < body required."""
    df = make_df([(100, 112, 99.6, 100.5, 1000)])
    # upper = 112 - max(100, 100.5) = 11.5, body = 0.5, lower = min(100, 100.5) - 99.6 = 0.4
    # conditions: 11.5 > 2*0.5=1 ✓, 0.4 < 0.5 ✓, body > 0 ✓
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_shooting_star"].iloc[-1] == 1.0, "Expected shooting star to be detected"


def test_pin_bar_lower_wick():
    """Long lower wick > 60% of range should be a pin bar."""
    df = make_df([(100, 101, 88, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_pin_bar"].iloc[-1] == 1.0


def test_pin_bar_upper_wick():
    """Long upper wick > 60% of range should be a pin bar."""
    df = make_df([(100, 112, 99, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_pin_bar"].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Two-bar patterns
# ---------------------------------------------------------------------------

def test_bullish_engulfing_detected():
    """Row 0: bearish (102→100). Row 1: bullish that engulfs row 0 body."""
    # prev: open=102, close=100 (bearish)
    # curr: open=99, close=104 → open<=prev_close(100) and close>=prev_open(102) → engulfs
    df = make_df([(102, 103, 99, 100, 1000), (99, 105, 98, 104, 1000)])
    atr = pd.Series([3.0, 3.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_bullish_engulfing"].iloc[-1] == 1.0


def test_bullish_engulfing_not_fired_on_first_bar():
    """First bar has no prior bar — shift gives NaN, result should be 0.0."""
    df = make_df([(102, 103, 99, 100, 1000)])
    atr = pd.Series([3.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_bullish_engulfing"].iloc[-1] == 0.0


def test_bearish_engulfing_detected():
    """Row 0: bullish (99→103). Row 1: bearish that engulfs row 0 body."""
    # prev: open=99, close=103 (bullish)
    # curr: open=104, close=98 → open>=prev_close(103) and close<=prev_open(99) → engulfs
    df = make_df([(99, 104, 98, 103, 1000), (104, 106, 97, 98, 1000)])
    atr = pd.Series([3.0, 3.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_bearish_engulfing"].iloc[-1] == 1.0


def test_inside_bar_detected():
    """Current H < prior H and current L > prior L."""
    df = make_df([(100, 110, 90, 105, 1000), (102, 108, 93, 104, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_inside_bar"].iloc[-1] == 1.0


def test_inside_bar_not_detected_when_range_equal():
    """Current high == prior high → NOT strictly inside."""
    df = make_df([(100, 110, 90, 105, 1000), (102, 110, 93, 104, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_inside_bar"].iloc[-1] == 0.0


def test_outside_bar_detected():
    """Current H > prior H and current L < prior L."""
    df = make_df([(100, 108, 94, 105, 1000), (97, 112, 91, 106, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_outside_bar"].iloc[-1] == 1.0


def test_impulse_candle_detected():
    """body = 4.0, ATR = 4.0 → candle_strength = 1.0 >= default threshold (1.0)."""
    df = make_df([(100, 105, 99, 104, 1000)])
    atr = pd.Series([4.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_impulse_candle"].iloc[-1] == 1.0


def test_impulse_candle_not_detected_small_body():
    """body = 0.5, ATR = 5.0 → strength = 0.1 < 1.0 → not impulse."""
    df = make_df([(100, 105, 95, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_impulse_candle"].iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# compute_candlestick_features integration
# ---------------------------------------------------------------------------

def test_all_features_present():
    """compute_candlestick_features returns all 15 expected columns."""
    expected = {
        "candle_body_pct", "candle_upper_wick_pct", "candle_lower_wick_pct",
        "candle_strength", "is_bullish", "is_bearish", "is_doji", "is_hammer",
        "is_shooting_star", "is_pin_bar", "is_bullish_engulfing",
        "is_bearish_engulfing", "is_inside_bar", "is_outside_bar",
        "is_impulse_candle",
    }
    df = make_df([(100, 110, 90, 105, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert set(result.columns) == expected


def test_all_features_are_float():
    """All features should be float dtype (0.0/1.0 for booleans)."""
    df = make_df([(100, 110, 90, 105, 1000), (103, 108, 93, 107, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    for col in result.columns:
        assert result[col].dtype in (float, "float64"), f"Column {col} has non-float dtype {result[col].dtype}"


def test_no_nan_in_single_bar_features():
    """Single-bar features (not two-bar) should not have NaN for the first bar."""
    df = make_df([(100, 110, 90, 105, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    single_bar_cols = [
        "candle_body_pct", "candle_upper_wick_pct", "candle_lower_wick_pct",
        "candle_strength", "is_bullish", "is_bearish", "is_doji", "is_hammer",
        "is_shooting_star", "is_pin_bar", "is_impulse_candle",
    ]
    for col in single_bar_cols:
        assert not pd.isna(result[col].iloc[-1]), f"Unexpected NaN in {col}"


def test_two_bar_patterns_zero_on_first_row():
    """Two-bar patterns should be 0.0 (not NaN) on the first row."""
    df = make_df([(100, 110, 90, 105, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    two_bar_cols = [
        "is_bullish_engulfing", "is_bearish_engulfing",
        "is_inside_bar", "is_outside_bar",
    ]
    for col in two_bar_cols:
        assert result[col].iloc[-1] == 0.0, f"Expected 0.0 on first row for {col}, got {result[col].iloc[-1]}"
