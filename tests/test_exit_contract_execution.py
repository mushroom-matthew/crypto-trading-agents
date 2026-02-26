"""Tests for PositionExitContract state-machine properties (Runbook 60).

Covers:
- stop_r_distance computation for long and short
- compute_r_multiple for both sides
- active_legs filtering (enabled, not fired)
- has_price_target property
- remaining_qty tracking semantics
- multi-leg contract with partial + full exit legs
- amendment_policy and allow_portfolio_overlay_trims defaults
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.position_exit_contract import (
    ExitLeg,
    PositionExitContract,
    TimeExitRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _long_contract(**kwargs) -> PositionExitContract:
    defaults = dict(
        position_id="pos_long_01",
        symbol="BTC-USD",
        side="long",
        created_at=_TS,
        source_trigger_id="btc_breakout_a1",
        entry_price=50000.0,
        initial_qty=0.1,
        stop_price_abs=49000.0,
        remaining_qty=0.1,
    )
    defaults.update(kwargs)
    return PositionExitContract(**defaults)


def _short_contract(**kwargs) -> PositionExitContract:
    defaults = dict(
        position_id="pos_short_01",
        symbol="ETH-USD",
        side="short",
        created_at=_TS,
        source_trigger_id="eth_reversal_a1",
        entry_price=3000.0,
        initial_qty=1.0,
        stop_price_abs=3100.0,
        remaining_qty=1.0,
    )
    defaults.update(kwargs)
    return PositionExitContract(**defaults)


def _price_leg(price: float, fraction: float = 1.0, fired: bool = False) -> ExitLeg:
    return ExitLeg(
        kind="full_exit",
        trigger_mode="price_level",
        fraction=fraction,
        price_abs=price,
        fired=fired,
    )


def _r_leg(r: float, fraction: float = 1.0, fired: bool = False) -> ExitLeg:
    return ExitLeg(
        kind="take_profit",
        trigger_mode="r_multiple",
        fraction=fraction,
        r_multiple=r,
        fired=fired,
    )


# ---------------------------------------------------------------------------
# stop_r_distance
# ---------------------------------------------------------------------------


def test_stop_r_distance_long():
    c = _long_contract(entry_price=50_000.0, stop_price_abs=49_000.0)
    assert c.stop_r_distance == pytest.approx(1_000.0)


def test_stop_r_distance_short():
    c = _short_contract(entry_price=3_000.0, stop_price_abs=3_100.0)
    assert c.stop_r_distance == pytest.approx(100.0)


def test_stop_r_distance_is_always_positive():
    # Long: entry > stop
    long_c = _long_contract(entry_price=100.0, stop_price_abs=95.0)
    assert long_c.stop_r_distance > 0

    # Short: stop > entry
    short_c = _short_contract(entry_price=100.0, stop_price_abs=105.0)
    assert short_c.stop_r_distance > 0


# ---------------------------------------------------------------------------
# compute_r_multiple
# ---------------------------------------------------------------------------


def test_compute_r_multiple_long_at_target():
    # entry=50000, stop=49000 → R_distance=1000
    # target=52000 → R = (52000-50000)/1000 = 2.0
    c = _long_contract(entry_price=50_000.0, stop_price_abs=49_000.0)
    assert c.compute_r_multiple(52_000.0) == pytest.approx(2.0)


def test_compute_r_multiple_long_at_stop():
    c = _long_contract(entry_price=50_000.0, stop_price_abs=49_000.0)
    assert c.compute_r_multiple(49_000.0) == pytest.approx(-1.0)


def test_compute_r_multiple_short_at_target():
    # entry=3000, stop=3100 → R_distance=100
    # target=2700 → R = (3000-2700)/100 = 3.0
    c = _short_contract(entry_price=3_000.0, stop_price_abs=3_100.0)
    assert c.compute_r_multiple(2_700.0) == pytest.approx(3.0)


def test_compute_r_multiple_short_at_stop():
    c = _short_contract(entry_price=3_000.0, stop_price_abs=3_100.0)
    assert c.compute_r_multiple(3_100.0) == pytest.approx(-1.0)


def test_compute_r_multiple_zero_stop_distance_returns_none():
    # Degenerate contract would fail validation, so we test via a
    # hypothetical zero stop_r_distance by patching after construction
    c = _long_contract(entry_price=50_000.0, stop_price_abs=49_000.0)
    # Monkey-patch stop_r_distance to simulate degenerate
    object.__setattr__(c, "stop_price_abs", 50_000.0)
    # Now stop_r_distance is 0 → should return None (not crash)
    # We test the method guard directly
    assert c.stop_r_distance == 0.0
    assert c.compute_r_multiple(51_000.0) is None


# ---------------------------------------------------------------------------
# active_legs
# ---------------------------------------------------------------------------


def test_active_legs_excludes_fired_legs():
    fired = _price_leg(55_000.0, fired=True)
    active = _price_leg(60_000.0)
    c = _long_contract(target_legs=[fired, active])
    assert len(c.active_legs) == 1
    assert c.active_legs[0].price_abs == 60_000.0


def test_active_legs_excludes_disabled_legs():
    disabled = ExitLeg(
        kind="take_profit",
        trigger_mode="r_multiple",
        fraction=0.5,
        r_multiple=2.0,
        enabled=False,
    )
    active = _r_leg(3.0, fraction=0.5)
    c = _long_contract(target_legs=[disabled, active])
    assert len(c.active_legs) == 1
    assert c.active_legs[0].r_multiple == 3.0


def test_active_legs_empty_when_all_fired():
    legs = [_price_leg(55_000.0, fired=True), _price_leg(60_000.0, fired=True)]
    c = _long_contract(target_legs=legs)
    assert c.active_legs == []


def test_active_legs_returns_all_when_none_fired():
    legs = [_r_leg(2.0, fraction=0.5), _r_leg(3.0, fraction=0.5)]
    c = _long_contract(target_legs=legs)
    assert len(c.active_legs) == 2


# ---------------------------------------------------------------------------
# has_price_target
# ---------------------------------------------------------------------------


def test_has_price_target_true_with_price_level_leg():
    c = _long_contract(target_legs=[_price_leg(55_000.0)])
    assert c.has_price_target is True


def test_has_price_target_true_with_r_multiple_leg():
    c = _long_contract(target_legs=[_r_leg(2.0)])
    assert c.has_price_target is True


def test_has_price_target_false_with_no_legs():
    c = _long_contract(target_legs=[])
    assert c.has_price_target is False


def test_has_price_target_false_when_all_fired():
    c = _long_contract(target_legs=[_price_leg(55_000.0, fired=True)])
    assert c.has_price_target is False


# ---------------------------------------------------------------------------
# Multi-leg ladder
# ---------------------------------------------------------------------------


def test_multi_leg_ladder_partial_fractions():
    # 50% at 2R, 50% at 3R — should be valid (sum == 1.0)
    legs = [
        ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=0.5, r_multiple=2.0),
        ExitLeg(kind="risk_reduce", trigger_mode="r_multiple", fraction=0.5, r_multiple=3.0),
    ]
    c = _long_contract(target_legs=legs)
    assert len(c.target_legs) == 2
    assert c.has_price_target is True


def test_multi_leg_ladder_partial_fractions_exceed_one_raises():
    legs = [
        ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=0.6, r_multiple=2.0),
        ExitLeg(kind="risk_reduce", trigger_mode="r_multiple", fraction=0.6, r_multiple=3.0),
    ]
    with pytest.raises(ValueError, match="fractions sum"):
        _long_contract(target_legs=legs)


def test_full_exit_legs_not_counted_in_partial_sum():
    # Two full_exit legs: each fraction=1.0 but validator only sums partial legs
    legs = [
        ExitLeg(kind="full_exit", trigger_mode="price_level", fraction=1.0, price_abs=55_000.0),
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=2.0),
    ]
    # Should not raise — full_exit legs are not summed
    c = _long_contract(target_legs=legs)
    assert len(c.target_legs) == 2


# ---------------------------------------------------------------------------
# Time exit + amendment policy
# ---------------------------------------------------------------------------


def test_contract_with_time_exit():
    time_rule = TimeExitRule(max_hold_bars=20, session_boundary_action="flatten")
    c = _long_contract(time_exit=time_rule)
    assert c.time_exit is not None
    assert c.time_exit.max_hold_bars == 20
    assert c.time_exit.session_boundary_action == "flatten"


def test_default_amendment_policy_is_tighten_only():
    c = _long_contract()
    assert c.amendment_policy == "tighten_only"


def test_default_allow_portfolio_overlay_trims():
    c = _long_contract()
    assert c.allow_portfolio_overlay_trims is True


def test_amendment_policy_none():
    c = _long_contract(amendment_policy="none")
    assert c.amendment_policy == "none"


# ---------------------------------------------------------------------------
# remaining_qty
# ---------------------------------------------------------------------------


def test_remaining_qty_initialized_to_initial_qty():
    c = _long_contract(initial_qty=0.5, remaining_qty=0.5)
    assert c.remaining_qty == pytest.approx(0.5)


def test_remaining_qty_can_be_partial():
    c = _long_contract(initial_qty=1.0, remaining_qty=0.5)
    assert c.remaining_qty == pytest.approx(0.5)


def test_remaining_qty_can_be_none():
    c = _long_contract(remaining_qty=None)
    assert c.remaining_qty is None
