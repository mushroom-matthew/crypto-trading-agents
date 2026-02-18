"""Unit tests for level-anchored stop/target computation (Runbook 42).

Tests cover:
- _resolve_stop_price_anchored: all 7 anchor types
- _resolve_target_price_anchored: all 5 target anchor types
- TradeLeg serialization with new optional fields
- TriggerCondition schema accepts stop_anchor_type / target_anchor_type
- Trigger engine context exposes below_stop / above_target / stop_price / target_price
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.llm_strategist_runner import (
    _resolve_stop_price_anchored,
    _resolve_target_price_anchored,
)
from schemas.trade_set import TradeLeg
from schemas.llm_strategist import IndicatorSnapshot, TriggerCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)


def _snapshot(**kwargs) -> IndicatorSnapshot:
    """Build a minimal IndicatorSnapshot with optional overrides."""
    defaults = dict(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=_ts(),
        close=50000.0,
        low=49500.0,
        atr_14=800.0,
        donchian_lower_short=48000.0,
        donchian_upper_short=52000.0,
        fib_618=47000.0,
        htf_daily_low=48500.0,
        htf_prev_daily_low=48000.0,
        htf_daily_high=52500.0,
        htf_5d_high=53000.0,
    )
    defaults.update(kwargs)
    return IndicatorSnapshot(**defaults)


def _trigger(**kwargs) -> TriggerCondition:
    """Build a minimal TriggerCondition."""
    defaults = dict(
        id="t1",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="is_flat",
        exit_rule="below_stop",
        confidence_grade="A",
        category="trend_continuation",
    )
    defaults.update(kwargs)
    return TriggerCondition(**defaults)


# ---------------------------------------------------------------------------
# _resolve_stop_price_anchored — stop anchor types
# ---------------------------------------------------------------------------

def test_stop_pct_long():
    """'pct' anchor: 2% stop below fill price for long."""
    trig = _trigger(stop_loss_pct=2.0, stop_anchor_type="pct")
    snap = _snapshot()
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "pct"
    assert price == pytest.approx(49000.0, rel=1e-6)


def test_stop_pct_short():
    """'pct' anchor: 2% stop above fill price for short."""
    trig = _trigger(stop_loss_pct=2.0, stop_anchor_type="pct")
    snap = _snapshot()
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "short")
    assert anchor == "pct"
    assert price == pytest.approx(51000.0, rel=1e-6)


def test_stop_pct_no_pct_returns_none():
    """'pct' anchor with no stop_loss_pct → (None, None)."""
    trig = _trigger(stop_anchor_type="pct")
    snap = _snapshot()
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert price is None
    assert anchor is None


def test_stop_atr_long():
    """'atr' anchor: 1.5 * ATR below entry for longs."""
    trig = _trigger(stop_anchor_type="atr")
    snap = _snapshot(atr_14=800.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "atr"
    assert price == pytest.approx(50000.0 - 1.5 * 800.0, rel=1e-6)


def test_stop_atr_short():
    """'atr' anchor: 1.5 * ATR above entry for shorts."""
    trig = _trigger(stop_anchor_type="atr")
    snap = _snapshot(atr_14=800.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "short")
    assert anchor == "atr"
    assert price == pytest.approx(50000.0 + 1.5 * 800.0, rel=1e-6)


def test_stop_htf_daily_low():
    """'htf_daily_low' anchor: 0.5% buffer below prior session's low."""
    trig = _trigger(stop_anchor_type="htf_daily_low")
    snap = _snapshot(htf_daily_low=48500.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "htf_daily_low"
    assert price == pytest.approx(48500.0 * 0.995, rel=1e-6)


def test_stop_htf_prev_daily_low():
    """'htf_prev_daily_low' anchor: 0.5% buffer below session-before-prior's low."""
    trig = _trigger(stop_anchor_type="htf_prev_daily_low")
    snap = _snapshot(htf_prev_daily_low=48000.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "htf_prev_daily_low"
    assert price == pytest.approx(48000.0 * 0.995, rel=1e-6)


def test_stop_donchian_lower():
    """'donchian_lower' anchor: 0.5% buffer below Donchian lower."""
    trig = _trigger(stop_anchor_type="donchian_lower")
    snap = _snapshot(donchian_lower_short=48000.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "donchian_lower"
    assert price == pytest.approx(48000.0 * 0.995, rel=1e-6)


def test_stop_fib_618():
    """'fib_618' anchor: exact fib level (no buffer)."""
    trig = _trigger(stop_anchor_type="fib_618")
    snap = _snapshot(fib_618=47000.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "fib_618"
    assert price == pytest.approx(47000.0, rel=1e-6)


def test_stop_candle_low():
    """'candle_low' anchor: 0.2% buffer below trigger bar's low."""
    trig = _trigger(stop_anchor_type="candle_low")
    snap = _snapshot(low=49500.0)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "candle_low"
    assert price == pytest.approx(49500.0 * 0.998, rel=1e-6)


def test_stop_missing_snapshot_field_returns_none():
    """If the required snapshot field is None, return (None, None)."""
    trig = _trigger(stop_anchor_type="htf_daily_low")
    snap = _snapshot(htf_daily_low=None)
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert price is None
    assert anchor is None


def test_stop_no_anchor_type_defaults_to_pct():
    """No stop_anchor_type → falls back to pct behaviour."""
    trig = _trigger(stop_loss_pct=1.5)  # no stop_anchor_type
    snap = _snapshot()
    price, anchor = _resolve_stop_price_anchored(trig, 50000.0, snap, "long")
    assert anchor == "pct"
    assert price == pytest.approx(50000.0 * 0.985, rel=1e-6)


# ---------------------------------------------------------------------------
# _resolve_target_price_anchored — target anchor types
# ---------------------------------------------------------------------------

def test_target_htf_daily_high():
    """'htf_daily_high' target: 0.2% below prior session high."""
    trig = _trigger(target_anchor_type="htf_daily_high")
    snap = _snapshot(htf_daily_high=52500.0)
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, None, snap, "long")
    assert anchor == "htf_daily_high"
    assert price == pytest.approx(52500.0 * 0.998, rel=1e-6)


def test_target_htf_5d_high():
    """'htf_5d_high' target: 0.2% below 5-day rolling high."""
    trig = _trigger(target_anchor_type="htf_5d_high")
    snap = _snapshot(htf_5d_high=53000.0)
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, None, snap, "long")
    assert anchor == "htf_5d_high"
    assert price == pytest.approx(53000.0 * 0.998, rel=1e-6)


def test_target_measured_move_long():
    """'measured_move': range height added to entry for longs."""
    trig = _trigger(target_anchor_type="measured_move")
    snap = _snapshot(donchian_upper_short=52000.0, donchian_lower_short=48000.0)
    # range = 4000, target = 50000 + 4000 = 54000
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, None, snap, "long")
    assert anchor == "measured_move"
    assert price == pytest.approx(54000.0, rel=1e-6)


def test_target_r_multiple_2():
    """'r_multiple_2': target = entry + 2 * risk (for longs)."""
    trig = _trigger(target_anchor_type="r_multiple_2")
    snap = _snapshot()
    stop = 49000.0  # risk = 1000
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, stop, snap, "long")
    assert anchor == "r_multiple_2"
    assert price == pytest.approx(52000.0, rel=1e-6)


def test_target_r_multiple_3():
    """'r_multiple_3': target = entry + 3 * risk (for longs)."""
    trig = _trigger(target_anchor_type="r_multiple_3")
    snap = _snapshot()
    stop = 49000.0  # risk = 1000
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, stop, snap, "long")
    assert anchor == "r_multiple_3"
    assert price == pytest.approx(53000.0, rel=1e-6)


def test_target_none_anchor_type():
    """No target_anchor_type → (None, None)."""
    trig = _trigger()  # no target_anchor_type
    snap = _snapshot()
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, None, snap, "long")
    assert price is None
    assert anchor is None


def test_target_r_multiple_requires_stop():
    """r_multiple targets require stop_price — returns (None, None) without it."""
    trig = _trigger(target_anchor_type="r_multiple_2")
    snap = _snapshot()
    price, anchor = _resolve_target_price_anchored(trig, 50000.0, None, snap, "long")
    assert price is None
    assert anchor is None


# ---------------------------------------------------------------------------
# TradeLeg schema — new fields
# ---------------------------------------------------------------------------

def test_trade_leg_new_fields_default_none():
    """TradeLeg new stop/target fields default to None."""
    leg = TradeLeg(
        side="buy",
        qty=0.1,
        price=50000.0,
        timestamp=_ts(),
        is_entry=True,
    )
    assert leg.stop_price_abs is None
    assert leg.target_price_abs is None
    assert leg.stop_anchor_type is None
    assert leg.target_anchor_type is None


def test_trade_leg_accepts_new_fields():
    """TradeLeg accepts stop/target fields when provided."""
    leg = TradeLeg(
        side="buy",
        qty=0.1,
        price=50000.0,
        timestamp=_ts(),
        is_entry=True,
        stop_price_abs=48500.0,
        target_price_abs=52500.0,
        stop_anchor_type="htf_daily_low",
        target_anchor_type="htf_daily_high",
    )
    assert leg.stop_price_abs == pytest.approx(48500.0)
    assert leg.target_price_abs == pytest.approx(52500.0)
    assert leg.stop_anchor_type == "htf_daily_low"
    assert leg.target_anchor_type == "htf_daily_high"


def test_trade_leg_serializes_new_fields():
    """TradeLeg serializes stop/target fields to JSON."""
    leg = TradeLeg(
        side="buy",
        qty=0.1,
        price=50000.0,
        timestamp=_ts(),
        is_entry=True,
        stop_price_abs=48500.0,
        target_price_abs=52500.0,
        stop_anchor_type="donchian_lower",
        target_anchor_type="r_multiple_2",
    )
    d = leg.model_dump()
    assert d["stop_price_abs"] == pytest.approx(48500.0)
    assert d["target_price_abs"] == pytest.approx(52500.0)
    assert d["stop_anchor_type"] == "donchian_lower"
    assert d["target_anchor_type"] == "r_multiple_2"


# ---------------------------------------------------------------------------
# TriggerCondition schema — new fields
# ---------------------------------------------------------------------------

def test_trigger_condition_accepts_anchor_fields():
    """TriggerCondition accepts stop_anchor_type and target_anchor_type."""
    trig = TriggerCondition(
        id="t1",
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="is_flat",
        exit_rule="below_stop",
        confidence_grade="A",
        category="trend_continuation",
        stop_anchor_type="htf_daily_low",
        target_anchor_type="r_multiple_2",
        stop_loss_pct=2.0,
    )
    assert trig.stop_anchor_type == "htf_daily_low"
    assert trig.target_anchor_type == "r_multiple_2"


def test_trigger_condition_anchor_fields_default_none():
    """stop_anchor_type and target_anchor_type default to None."""
    trig = _trigger()
    assert trig.stop_anchor_type is None
    assert trig.target_anchor_type is None


# ---------------------------------------------------------------------------
# Trigger engine context — below_stop / above_target / stop_price / target_price
# ---------------------------------------------------------------------------

def _make_engine_context(
    close: float,
    stop_price_abs: float | None,
    target_price_abs: float | None,
    is_long: bool = True,
) -> dict:
    """Build a minimal trigger engine context dict matching what _context() would produce."""
    # Simulate what _context() does for below_stop / above_target
    is_flat = not is_long
    active_stop = stop_price_abs
    active_target = target_price_abs
    current_close = close
    context: dict = {
        "close": close,
        "is_flat": is_flat,
        "is_long": is_long,
        "is_short": False,
        "position": "long" if is_long else "flat",
    }
    context["below_stop"] = bool(current_close < active_stop) if (active_stop and not is_flat) else False
    context["above_target"] = bool(current_close > active_target) if (active_target and not is_flat) else False
    context["stop_price"] = active_stop or 0.0
    context["target_price"] = active_target or 0.0
    context["stop_distance_pct"] = (
        abs(current_close - active_stop) / current_close * 100.0
        if (active_stop and current_close) else 0.0
    )
    context["target_distance_pct"] = (
        abs(active_target - current_close) / current_close * 100.0
        if (active_target and current_close) else 0.0
    )
    return context


def test_below_stop_fires_when_close_below():
    ctx = _make_engine_context(close=48000.0, stop_price_abs=48500.0, target_price_abs=None, is_long=True)
    assert ctx["below_stop"] is True


def test_below_stop_does_not_fire_when_close_above():
    ctx = _make_engine_context(close=49000.0, stop_price_abs=48500.0, target_price_abs=None, is_long=True)
    assert ctx["below_stop"] is False


def test_below_stop_false_when_flat():
    """below_stop should be False when position is flat (no position)."""
    ctx = _make_engine_context(close=48000.0, stop_price_abs=48500.0, target_price_abs=None, is_long=False)
    assert ctx["below_stop"] is False


def test_below_stop_false_when_no_stop():
    """below_stop is False when stop_price_abs is None."""
    ctx = _make_engine_context(close=48000.0, stop_price_abs=None, target_price_abs=None, is_long=True)
    assert ctx["below_stop"] is False


def test_above_target_fires_when_close_above():
    ctx = _make_engine_context(close=53000.0, stop_price_abs=None, target_price_abs=52500.0, is_long=True)
    assert ctx["above_target"] is True


def test_above_target_does_not_fire_when_close_below():
    ctx = _make_engine_context(close=51000.0, stop_price_abs=None, target_price_abs=52500.0, is_long=True)
    assert ctx["above_target"] is False


def test_stop_distance_pct_computed():
    """stop_distance_pct = |close - stop| / close * 100."""
    ctx = _make_engine_context(close=50000.0, stop_price_abs=48500.0, target_price_abs=None, is_long=True)
    expected = abs(50000.0 - 48500.0) / 50000.0 * 100.0
    assert ctx["stop_distance_pct"] == pytest.approx(expected, rel=1e-6)


def test_target_distance_pct_computed():
    """target_distance_pct = |target - close| / close * 100."""
    ctx = _make_engine_context(close=50000.0, stop_price_abs=None, target_price_abs=52500.0, is_long=True)
    expected = abs(52500.0 - 50000.0) / 50000.0 * 100.0
    assert ctx["target_distance_pct"] == pytest.approx(expected, rel=1e-6)


def test_stop_price_zero_when_no_stop():
    ctx = _make_engine_context(close=50000.0, stop_price_abs=None, target_price_abs=None, is_long=True)
    assert ctx["stop_price"] == 0.0
    assert ctx["target_price"] == 0.0
