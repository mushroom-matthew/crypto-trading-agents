"""Unit tests for higher-timeframe structural anchor field computation.

Tests cover compute_htf_structural_fields() directly (unit) and verify that
IndicatorSnapshot accepts the new htf_* fields without Pydantic validation errors.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.analytics.indicator_snapshots import compute_htf_structural_fields
from schemas.llm_strategist import IndicatorSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_df(days: int = 10, base_price: float = 50000.0) -> pd.DataFrame:
    """Create a daily OHLCV DataFrame with a DatetimeIndex (UTC)."""
    idx = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    prices = base_price + np.linspace(0, 1000, days)
    df = pd.DataFrame(
        {
            "open": prices - 200,
            "high": prices + 500,
            "low": prices - 300,
            "close": prices,
            "volume": np.ones(days) * 1e6,
        },
        index=idx,
    )
    return df


def _bar_ts(date_str: str) -> datetime:
    """Return a UTC datetime for a given date string (midnight UTC)."""
    d = datetime.fromisoformat(date_str)
    return d.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# compute_htf_structural_fields — edge cases
# ---------------------------------------------------------------------------

def test_returns_empty_when_daily_df_is_none():
    result = compute_htf_structural_fields(_bar_ts("2024-01-15"), None)
    assert result == {}


def test_returns_empty_when_daily_df_is_empty():
    result = compute_htf_structural_fields(_bar_ts("2024-01-15"), pd.DataFrame())
    assert result == {}


def test_returns_empty_when_fewer_than_2_completed_sessions():
    """Bar at 2024-01-02 — only 1 completed daily bar (Jan 1). Need >= 2."""
    daily = _daily_df(days=5)
    # Only Jan 1 is before Jan 2
    result = compute_htf_structural_fields(_bar_ts("2024-01-02"), daily)
    assert result == {}


def test_returns_fields_when_sufficient_history():
    """Bar at 2024-01-05 — four completed daily bars available (Jan 1-4)."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-05"), daily)
    assert result != {}


# ---------------------------------------------------------------------------
# compute_htf_structural_fields — correct prior-session values
# ---------------------------------------------------------------------------

def test_prior_session_high_is_yesterday():
    """htf_daily_high should equal the prior completed session's high."""
    daily = _daily_df(days=10)
    # Bar at 2024-01-06 → yesterday is 2024-01-05 (index 4, 0-based)
    result = compute_htf_structural_fields(_bar_ts("2024-01-06"), daily)
    expected_high = float(daily.iloc[4]["high"])  # Jan 5 (index 4)
    assert result["htf_daily_high"] == pytest.approx(expected_high, rel=1e-6)


def test_prior_session_low_is_yesterday():
    """htf_daily_low should equal the prior completed session's low."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-06"), daily)
    expected_low = float(daily.iloc[4]["low"])
    assert result["htf_daily_low"] == pytest.approx(expected_low, rel=1e-6)


def test_prev2_session_high_is_two_days_ago():
    """htf_prev_daily_high should equal the session-before-prior's high."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-06"), daily)
    expected_prev2_high = float(daily.iloc[3]["high"])  # Jan 4 (index 3)
    assert result["htf_prev_daily_high"] == pytest.approx(expected_prev2_high, rel=1e-6)


def test_five_day_high_is_max_over_five_sessions():
    """htf_5d_high should be the maximum high over the 5 most recent completed bars."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-11"), daily)
    # Completed bars before Jan 11 are Jan 1-10 (indices 0-9), 5-day lookback = Jan 6-10
    last_5 = daily.tail(5)
    expected_5d_high = float(last_5["high"].max())
    assert result["htf_5d_high"] == pytest.approx(expected_5d_high, rel=1e-6)


def test_five_day_low_is_min_over_five_sessions():
    """htf_5d_low should be the minimum low over the 5 most recent completed bars."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-11"), daily)
    last_5 = daily.tail(5)
    expected_5d_low = float(last_5["low"].min())
    assert result["htf_5d_low"] == pytest.approx(expected_5d_low, rel=1e-6)


def test_daily_range_pct_positive():
    """htf_daily_range_pct should be (high - low) / close * 100 > 0."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-06"), daily)
    assert result["htf_daily_range_pct"] > 0.0


def test_prev_daily_mid_is_average_of_prev2_high_low():
    """htf_prev_daily_mid should be (prev2_high + prev2_low) / 2."""
    daily = _daily_df(days=10)
    result = compute_htf_structural_fields(_bar_ts("2024-01-06"), daily)
    prev2 = daily.iloc[3]  # Jan 4
    expected = (float(prev2["high"]) + float(prev2["low"])) / 2.0
    assert result["htf_prev_daily_mid"] == pytest.approx(expected, rel=1e-6)


def test_daily_atr_is_positive():
    """htf_daily_atr should be a positive float."""
    daily = _daily_df(days=20)  # Need at least 14 bars for ATR warmup
    result = compute_htf_structural_fields(_bar_ts("2024-01-21"), daily)
    assert result.get("htf_daily_atr", 0) > 0.0


def test_all_expected_keys_present():
    """All 12 expected keys should appear in the result."""
    daily = _daily_df(days=20)
    result = compute_htf_structural_fields(_bar_ts("2024-01-21"), daily)
    expected_keys = {
        "htf_daily_open", "htf_daily_high", "htf_daily_low", "htf_daily_close",
        "htf_prev_daily_high", "htf_prev_daily_low", "htf_prev_daily_open",
        "htf_daily_atr", "htf_daily_range_pct",
        "htf_5d_high", "htf_5d_low", "htf_prev_daily_mid",
    }
    assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# IndicatorSnapshot schema — new fields accepted
# ---------------------------------------------------------------------------

def test_indicator_snapshot_accepts_htf_fields():
    """IndicatorSnapshot should accept all htf_* fields without validation error."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
        close=50500.0,
        htf_daily_high=51000.0,
        htf_daily_low=49500.0,
        htf_daily_open=49800.0,
        htf_daily_close=50200.0,
        htf_prev_daily_high=50800.0,
        htf_prev_daily_low=49200.0,
        htf_prev_daily_open=49600.0,
        htf_daily_atr=1200.0,
        htf_daily_range_pct=3.0,
        htf_price_vs_daily_mid=0.25,
        htf_5d_high=52000.0,
        htf_5d_low=48000.0,
        htf_prev_daily_mid=50000.0,
    )
    assert snap.htf_daily_high == pytest.approx(51000.0)
    assert snap.htf_price_vs_daily_mid == pytest.approx(0.25)


def test_indicator_snapshot_htf_fields_default_none():
    """htf_* fields should default to None when not provided."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
        close=50500.0,
    )
    assert snap.htf_daily_high is None
    assert snap.htf_daily_low is None
    assert snap.htf_5d_high is None
    assert snap.htf_prev_daily_mid is None


def test_indicator_snapshot_accepts_candlestick_fields():
    """IndicatorSnapshot should accept all candlestick pattern fields."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
        close=50500.0,
        candle_body_pct=0.6,
        candle_upper_wick_pct=0.2,
        candle_lower_wick_pct=0.2,
        candle_strength=0.9,
        is_bullish=1.0,
        is_bearish=0.0,
        is_doji=0.0,
        is_hammer=0.0,
        is_shooting_star=0.0,
        is_pin_bar=0.0,
        is_bullish_engulfing=0.0,
        is_bearish_engulfing=0.0,
        is_inside_bar=0.0,
        is_outside_bar=0.0,
        is_impulse_candle=0.0,
    )
    assert snap.is_bullish == pytest.approx(1.0)
    assert snap.candle_body_pct == pytest.approx(0.6)


def test_indicator_snapshot_candlestick_fields_default_none():
    """Candlestick fields should default to None when not provided."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
        close=50500.0,
    )
    assert snap.is_hammer is None
    assert snap.is_bullish_engulfing is None
    assert snap.candle_body_pct is None
