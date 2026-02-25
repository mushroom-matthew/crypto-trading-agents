"""Tests for R55 vocabulary mapping functions in services/regime_transition_detector.py.

Covers:
  - map_trend_state: all known legacy labels → R55 canonical
  - map_vol_state:   'normal' → 'mid' (key mapping difference)
  - map_structure_state: priority ordering of R40 flags + regime + trend
  - Unknown inputs fall back to safe defaults
"""

from __future__ import annotations

import pytest

from services.regime_transition_detector import (
    map_structure_state,
    map_trend_state,
    map_vol_state,
)


# ---------------------------------------------------------------------------
# map_trend_state
# ---------------------------------------------------------------------------

def test_map_trend_state_uptrend():
    assert map_trend_state("uptrend") == "up"


def test_map_trend_state_downtrend():
    assert map_trend_state("downtrend") == "down"


def test_map_trend_state_sideways_passthrough():
    assert map_trend_state("sideways") == "sideways"


def test_map_trend_state_bull():
    assert map_trend_state("bull") == "up"


def test_map_trend_state_bear():
    assert map_trend_state("bear") == "down"


def test_map_trend_state_bullish():
    assert map_trend_state("bullish") == "up"


def test_map_trend_state_bearish():
    assert map_trend_state("bearish") == "down"


def test_map_trend_state_case_insensitive():
    assert map_trend_state("UPTREND") == "up"
    assert map_trend_state("Downtrend") == "down"


def test_map_trend_state_unknown_falls_back_to_sideways():
    assert map_trend_state("") == "sideways"
    assert map_trend_state("unknown_label") == "sideways"


def test_map_trend_state_r55_canonical_passthrough():
    """R55 canonical labels should pass through unchanged."""
    assert map_trend_state("up") == "up"
    assert map_trend_state("down") == "down"


# ---------------------------------------------------------------------------
# map_vol_state
# ---------------------------------------------------------------------------

def test_map_vol_state_normal_becomes_mid():
    """Critical: 'normal' must map to 'mid', not 'normal'."""
    assert map_vol_state("normal") == "mid"


def test_map_vol_state_low_passthrough():
    assert map_vol_state("low") == "low"


def test_map_vol_state_high_passthrough():
    assert map_vol_state("high") == "high"


def test_map_vol_state_extreme_passthrough():
    assert map_vol_state("extreme") == "extreme"


def test_map_vol_state_mid_passthrough():
    assert map_vol_state("mid") == "mid"


def test_map_vol_state_medium():
    assert map_vol_state("medium") == "mid"


def test_map_vol_state_case_insensitive():
    assert map_vol_state("NORMAL") == "mid"
    assert map_vol_state("HIGH") == "high"


def test_map_vol_state_unknown_falls_back_to_mid():
    assert map_vol_state("") == "mid"
    assert map_vol_state("some_other_label") == "mid"


def test_map_vol_state_r55_canonical_passthrough():
    for vs in ("low", "mid", "high", "extreme"):
        assert map_vol_state(vs) == vs


# ---------------------------------------------------------------------------
# map_structure_state
# ---------------------------------------------------------------------------

def test_map_structure_breakout_confirmed_up_trend():
    result = map_structure_state(
        compression_flag=0.0,
        expansion_flag=0.0,
        breakout_confirmed=0.8,
        trend_state="up",
    )
    assert result == "breakout_active"


def test_map_structure_breakout_confirmed_down_trend():
    """Breakout with down trend → breakdown_active."""
    result = map_structure_state(
        compression_flag=0.0,
        expansion_flag=0.0,
        breakout_confirmed=0.9,
        trend_state="down",
    )
    assert result == "breakdown_active"


def test_map_structure_compression_takes_priority_over_expansion():
    """compression_flag > 0.5 wins over expansion_flag when both set (after breakout check)."""
    result = map_structure_state(
        compression_flag=0.8,
        expansion_flag=0.9,
        breakout_confirmed=0.2,  # not confirmed
        trend_state="sideways",
    )
    assert result == "compression"


def test_map_structure_expansion():
    result = map_structure_state(
        compression_flag=0.1,
        expansion_flag=0.8,
        breakout_confirmed=0.0,
        trend_state="up",
    )
    assert result == "expansion"


def test_map_structure_range_regime_mean_reverting():
    result = map_structure_state(
        compression_flag=None,
        expansion_flag=None,
        breakout_confirmed=None,
        trend_state="sideways",
        regime="range",
    )
    assert result == "mean_reverting"


def test_map_structure_default_neutral():
    result = map_structure_state(
        compression_flag=0.2,
        expansion_flag=0.1,
        breakout_confirmed=0.0,
        trend_state="sideways",
        regime="bull",
    )
    assert result == "neutral"


def test_map_structure_all_none_neutral():
    result = map_structure_state(
        compression_flag=None,
        expansion_flag=None,
        breakout_confirmed=None,
        trend_state="sideways",
    )
    assert result == "neutral"


def test_map_structure_breakout_priority_over_compression():
    """breakout_confirmed wins over compression_flag (checked first)."""
    result = map_structure_state(
        compression_flag=0.9,
        expansion_flag=0.0,
        breakout_confirmed=0.7,
        trend_state="up",
    )
    assert result == "breakout_active"


def test_map_structure_compression_threshold_boundary():
    """Flag must be > 0.5, not >= 0.5."""
    result_above = map_structure_state(
        compression_flag=0.51,
        expansion_flag=0.0,
        breakout_confirmed=0.0,
        trend_state="sideways",
    )
    assert result_above == "compression"

    result_at = map_structure_state(
        compression_flag=0.5,
        expansion_flag=0.0,
        breakout_confirmed=0.0,
        trend_state="sideways",
    )
    assert result_at == "neutral"  # 0.5 is NOT > 0.5
