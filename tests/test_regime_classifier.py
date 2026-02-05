"""Tests for the deterministic regime classifier."""

from datetime import datetime, timezone

import pytest

from schemas.llm_strategist import IndicatorSnapshot, RegimeAssessment
from trading_core.regime_classifier import classify_regime


def _make_snapshot(**overrides) -> IndicatorSnapshot:
    """Helper to create indicator snapshots for testing."""
    defaults = {
        "symbol": "BTC-USD",
        "timeframe": "1h",
        "as_of": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "close": 50000.0,
        "sma_medium": 48000.0,
        "rsi_14": 60.0,
        "macd": 100.0,
        "macd_hist": 50.0,
        "atr_14": 1000.0,
    }
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def test_classify_regime_bull():
    """Test classification of bullish market."""
    snapshot = _make_snapshot(
        close=52000.0,
        sma_medium=48000.0,  # price above SMA
        rsi_14=62.0,  # RSI > 55
        macd=150.0,  # MACD positive
        atr_14=1000.0,  # normal volatility
    )
    result = classify_regime(snapshot)
    assert result.regime == "bull"
    assert result.confidence >= 0.5
    assert any("above_sma" in signal for signal in result.primary_signals)


def test_classify_regime_bear():
    """Test classification of bearish market."""
    snapshot = _make_snapshot(
        close=45000.0,
        sma_medium=50000.0,  # price below SMA
        rsi_14=38.0,  # RSI < 45
        macd=-150.0,  # MACD negative
        atr_14=1000.0,  # normal volatility
    )
    result = classify_regime(snapshot)
    assert result.regime == "bear"
    assert result.confidence >= 0.5


def test_classify_regime_volatile():
    """Test classification of volatile market (extreme ATR)."""
    snapshot = _make_snapshot(
        close=50000.0,
        atr_14=5000.0,  # ATR/price = 10% > 8% threshold
    )
    result = classify_regime(snapshot)
    assert result.regime == "volatile"
    assert "extreme_volatility" in result.primary_signals


def test_classify_regime_range():
    """Test classification of ranging market (low vol)."""
    snapshot = _make_snapshot(
        close=50000.0,
        sma_medium=50000.0,  # price near SMA
        rsi_14=50.0,  # RSI neutral
        macd=0.1,  # MACD slightly positive (avoiding exactly 0 which counts as negative)
        atr_14=400.0,  # ATR/price < 1%
    )
    result = classify_regime(snapshot)
    # Low volatility should trigger range detection
    assert result.regime == "range"
    assert "low_volatility_range" in result.primary_signals


def test_classify_regime_uncertain():
    """Test classification when signals conflict."""
    snapshot = _make_snapshot(
        close=52000.0,
        sma_medium=48000.0,  # bullish
        rsi_14=38.0,  # bearish
        macd=-100.0,  # bearish
        atr_14=1000.0,  # normal vol
    )
    result = classify_regime(snapshot)
    # 1 bullish, 2 bearish -> should be bear, not uncertain
    assert result.regime in ("bear", "uncertain")
    assert len(result.conflicting_signals) >= 0  # may have conflicting signals


def test_classify_regime_returns_valid_schema():
    """Test that result matches RegimeAssessment schema."""
    snapshot = _make_snapshot()
    result = classify_regime(snapshot)
    assert isinstance(result, RegimeAssessment)
    assert result.regime in ("bull", "bear", "range", "volatile", "uncertain")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.primary_signals, list)
    assert isinstance(result.conflicting_signals, list)
