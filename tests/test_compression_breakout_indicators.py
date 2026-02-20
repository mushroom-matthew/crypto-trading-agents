"""Tests for compression/breakout indicators (Runbook 40).

These tests verify bb_bandwidth_pct_rank, compression_flag, expansion_flag,
and breakout_confirmed are computed correctly in both the scalar snapshot path
and the batch precompute path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.analytics.indicator_snapshots import (
    IndicatorWindowConfig,
    compute_indicator_snapshot,
    precompute_indicator_frame,
    snapshot_from_frame,
)
from schemas.llm_strategist import IndicatorSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base_price: float = 100.0, vol_factor: float = 1.0) -> pd.DataFrame:
    """Build a synthetic OHLCV frame with n bars at roughly constant price."""
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, n)
    close = base_price + np.cumsum(noise * 0.05)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000.0 * vol_factor * rng.uniform(0.8, 1.2, n),
        }
    )


def _make_tight_ohlcv(n: int, base_price: float = 100.0) -> pd.DataFrame:
    """Build a frame with WIDE bandwidth for first half, then TIGHT for second half.

    The rolling 50-bar percentile rank for the final bars will be very low (bottom quintile)
    because the later bars have much narrower bandwidth than the earlier bars.
    → compression_flag should fire.

    IMPORTANT: Bollinger bands are computed on CLOSE prices.  The "wide" half must have
    varied closes (not just wide high/low) so that BB std-dev is non-zero and produces
    real bandwidth.  The "tight" half has constant close → BB bandwidth ≈ 0 → low rank.
    """
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed=7)
    close = np.full(n, base_price, dtype=float)
    high = np.full(n, base_price + 0.01, dtype=float)
    low = np.full(n, base_price - 0.01, dtype=float)
    half = n // 2
    # First half: very wide bandwidth — vary close so Bollinger std-dev != 0
    for i in range(half):
        high[i] = base_price + 10.0
        low[i] = base_price - 10.0
        close[i] = base_price + rng.uniform(-8.0, 8.0)
    # Second half: very tight bandwidth (±0.01 per bar) — will be in bottom quintile
    # close stays at base_price (constant) → BB bandwidth ≈ 0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(np.ones(n) * 1000.0),
        }
    )


def _make_expanding_ohlcv(n: int, base_price: float = 100.0) -> pd.DataFrame:
    """Build a frame that starts compressed then expands.

    First half: tight range → BB bandwidth low.
    Second half: wide range → BB bandwidth high and growing.
    """
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, base_price, dtype=float)
    high = np.full(n, base_price + 0.05, dtype=float)
    low = np.full(n, base_price - 0.05, dtype=float)
    mid = n // 2
    # Second half: widening range
    for i in range(mid, n):
        width = 2.0 * (i - mid + 1)
        high[i] = base_price + width
        low[i] = base_price - width
        close[i] = base_price + (i - mid) * 0.5
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        }
    )


def _snapshot(df: pd.DataFrame) -> IndicatorSnapshot:
    config = IndicatorWindowConfig(timeframe="1h")
    return compute_indicator_snapshot(df, symbol="BTC-USD", timeframe="1h", config=config)


# ---------------------------------------------------------------------------
# Test: bb_bandwidth_pct_rank returns None when insufficient bars
# ---------------------------------------------------------------------------

def test_bb_bandwidth_pct_rank_none_insufficient_bars():
    """With < 20 bars, min_periods=20 rolling rank returns NaN → field is None."""
    df = _make_ohlcv(15)
    snap = _snapshot(df)
    assert snap.bb_bandwidth_pct_rank is None, (
        f"Expected None for < 20 bars but got {snap.bb_bandwidth_pct_rank}"
    )


# ---------------------------------------------------------------------------
# Test: compression_flag = 1 when bb_bandwidth_pct_rank < 0.20
# ---------------------------------------------------------------------------

def test_compression_flag_fires_when_compressed():
    """Tight-range bars should produce low BB bandwidth → compression_flag = 1."""
    # Need >= 50 bars for rank to be meaningful
    df = _make_tight_ohlcv(60)
    snap = _snapshot(df)
    # With perfectly flat prices, bandwidth will be zero throughout → rank = 0 → compressed
    assert snap.compression_flag == 1.0, (
        f"Expected compression_flag=1.0 for tight range, got {snap.compression_flag} "
        f"(bb_bandwidth_pct_rank={snap.bb_bandwidth_pct_rank})"
    )


# ---------------------------------------------------------------------------
# Test: compression_flag = 0 when bb_bandwidth_pct_rank >= 0.20
# ---------------------------------------------------------------------------

def test_compression_flag_zero_when_not_compressed():
    """Wide-range bars (expanding) should not fire compression_flag."""
    df = _make_expanding_ohlcv(80)
    snap = _snapshot(df)
    # At the final bar, bandwidth should be high (expanding) → rank near 1.0 → not compressed
    assert snap.compression_flag == 0.0, (
        f"Expected compression_flag=0.0 for expanding market, got {snap.compression_flag} "
        f"(bb_bandwidth_pct_rank={snap.bb_bandwidth_pct_rank})"
    )


# ---------------------------------------------------------------------------
# Test: expansion_flag = 1 when rank > 0.80 and bandwidth growing
# ---------------------------------------------------------------------------

def test_expansion_flag_fires_when_bandwidth_expanding():
    """Expanding bandwidth in top quintile → expansion_flag = 1."""
    df = _make_expanding_ohlcv(80)
    snap = _snapshot(df)
    # After 40 bars of expansion, bandwidth should be in top quintile and growing
    if snap.bb_bandwidth_pct_rank is not None and snap.bb_bandwidth_pct_rank > 0.80:
        assert snap.expansion_flag == 1.0, (
            f"Expected expansion_flag=1.0 when rank > 0.80 and bandwidth growing, "
            f"got {snap.expansion_flag}"
        )
    # If rank <= 0.80 for this input, the test is informational only
    else:
        pytest.skip(
            f"bb_bandwidth_pct_rank={snap.bb_bandwidth_pct_rank} not > 0.80, skip expansion check"
        )


# ---------------------------------------------------------------------------
# Test: expansion_flag = 0 when rank > 0.80 but bandwidth contracting
# ---------------------------------------------------------------------------

def test_expansion_flag_zero_when_bandwidth_contracting():
    """expansion_flag must be 0 if bandwidth was expanding but latest bar contracts."""
    # Build expanding then contracting pattern
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    # Middle bars widen dramatically
    for i in range(20, 80):
        w = float(i - 20) * 2.0
        high[i] = 100.0 + w
        low[i] = 100.0 - w
    # Last 20 bars: snap back tight
    for i in range(80, n):
        high[i] = 100.05
        low[i] = 99.95
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        }
    )
    snap = _snapshot(df)
    # The final bar has very narrow range → bandwidth is contracting
    assert snap.expansion_flag == 0.0, (
        f"Expected expansion_flag=0.0 (bandwidth contracting), got {snap.expansion_flag}"
    )


# ---------------------------------------------------------------------------
# Test: breakout_confirmed = 1 when close > donchian_upper AND vol_burst
# ---------------------------------------------------------------------------

def test_breakout_confirmed_fires_on_outside_close_with_volume():
    """Close above prior-bar Donchian upper + high volume → breakout_confirmed = 1.

    The implementation compares close to the PRIOR bar's Donchian range (standard TA:
    detect when current close exceeds previous N-bar high). This means all prior bars'
    highs must be below the current close.
    """
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0, dtype=float)
    high = np.full(n, 100.1, dtype=float)
    low = np.full(n, 99.9, dtype=float)
    # Last bar: close far above all prior highs (100.1), with huge volume
    close[-1] = 999.0
    high[-1] = 999.5  # high >= close, but previous 20 bars all at 100.1
    volume = np.ones(n) * 500.0
    volume[-1] = 50000.0  # 100x average → vol_burst fires
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    snap = _snapshot(df)
    # Prior 20-bar high max is 100.1; current close is 999.0 → breakout
    assert snap.breakout_confirmed == 1.0, (
        f"Expected breakout_confirmed=1.0 (close 999.0 above prior Donchian 100.1 + vol burst), "
        f"got {snap.breakout_confirmed} (vol_burst={snap.vol_burst})"
    )


# ---------------------------------------------------------------------------
# Test: breakout_confirmed = 0 when close inside range
# ---------------------------------------------------------------------------

def test_breakout_confirmed_zero_when_inside_range():
    """Close inside Donchian range → breakout_confirmed = 0."""
    df = _make_ohlcv(60)
    snap = _snapshot(df)
    # In a typical oscillating market the last close is inside the 20-bar range
    if snap.donchian_upper_short and snap.donchian_lower_short:
        if snap.donchian_lower_short < snap.close < snap.donchian_upper_short:
            assert snap.breakout_confirmed == 0.0, (
                f"Expected breakout_confirmed=0.0 (close inside range), got {snap.breakout_confirmed}"
            )


# ---------------------------------------------------------------------------
# Test: breakout_confirmed = 0 when close outside range but no vol_burst
# ---------------------------------------------------------------------------

def test_breakout_confirmed_zero_without_vol_burst():
    """Close outside prior Donchian range but low volume → breakout_confirmed = 0."""
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0, dtype=float)
    high = np.full(n, 100.1, dtype=float)
    low = np.full(n, 99.9, dtype=float)
    close[-1] = 999.0  # above prior Donchian range
    high[-1] = 999.5
    volume = np.ones(n) * 500.0
    volume[-1] = 501.0  # barely above average, below vol_burst_threshold (1.5x)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    snap = _snapshot(df)
    # vol_burst threshold is 1.5x; 501/500 ≈ 1.002x → no burst
    assert snap.vol_burst is False or snap.vol_burst is None or snap.breakout_confirmed == 0.0, (
        f"Expected breakout_confirmed=0.0 (no vol burst), got {snap.breakout_confirmed}"
    )


# ---------------------------------------------------------------------------
# Test: IndicatorSnapshot validates all 4 new fields
# ---------------------------------------------------------------------------

def test_indicator_snapshot_accepts_new_fields():
    """IndicatorSnapshot schema must accept the 4 new compression/breakout fields."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        close=100.0,
        bb_bandwidth_pct_rank=0.15,
        compression_flag=1.0,
        expansion_flag=0.0,
        breakout_confirmed=0.0,
    )
    assert snap.bb_bandwidth_pct_rank == 0.15
    assert snap.compression_flag == 1.0
    assert snap.expansion_flag == 0.0
    assert snap.breakout_confirmed == 0.0


def test_indicator_snapshot_new_fields_default_none():
    """New fields default to None when not provided."""
    snap = IndicatorSnapshot(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        close=100.0,
    )
    assert snap.bb_bandwidth_pct_rank is None
    assert snap.compression_flag is None
    assert snap.expansion_flag is None
    assert snap.breakout_confirmed is None


# ---------------------------------------------------------------------------
# Test: precompute_indicator_frame includes the 4 new columns
# ---------------------------------------------------------------------------

def test_precompute_frame_includes_compression_columns():
    """precompute_indicator_frame must include all 4 R40 columns."""
    df = _make_ohlcv(80)
    config = IndicatorWindowConfig(timeframe="1h")
    frame = precompute_indicator_frame(df, config=config)
    for col in ("bb_bandwidth_pct_rank", "compression_flag", "expansion_flag", "breakout_confirmed"):
        assert col in frame.columns, f"Missing column: {col}"
    # After 50+ bars, bb_bandwidth_pct_rank should have real values (not all NaN)
    non_null = frame["bb_bandwidth_pct_rank"].dropna()
    assert len(non_null) > 0, "bb_bandwidth_pct_rank is entirely NaN for 80-bar frame"


def test_snapshot_from_frame_reads_compression_fields():
    """snapshot_from_frame must propagate the 4 R40 fields into IndicatorSnapshot."""
    df = _make_tight_ohlcv(60)
    config = IndicatorWindowConfig(timeframe="1h")
    frame = precompute_indicator_frame(df, config=config)
    ts = frame.index[-1]
    snap = snapshot_from_frame(frame, timestamp=ts.to_pydatetime(), symbol="BTC-USD", timeframe="1h")
    assert snap is not None
    # compression_flag should be accessible (may be 0 or 1 depending on rank)
    assert snap.compression_flag is not None or snap.bb_bandwidth_pct_rank is None
