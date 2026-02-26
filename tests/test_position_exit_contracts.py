"""Tests for schemas/position_exit_contract.py and services/exit_contract_builder.py.

Covers Runbook 60 Phase M1 (schema validation) and Phase M2 (contract materialization).
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from schemas.position_exit_contract import (
    POSITION_EXIT_CONTRACT_VERSION,
    ExitLeg,
    PortfolioMetaAction,
    PortfolioMetaRiskPolicy,
    PositionExitContract,
    TimeExitRule,
)
from schemas.llm_strategist import TriggerCondition
from services.exit_contract_builder import (
    EXIT_CONTRACT_BUILDER_VERSION,
    _PRICE_LEVEL_ANCHORS,
    _R_MULTIPLE_ANCHOR_MAP,
    build_exit_contract,
    can_build_contract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _long_trigger(
    trigger_id: str = "btc_long",
    symbol: str = "BTC-USD",
    target_anchor: str | None = None,
    estimated_bars: int | None = None,
) -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol=symbol,
        direction="long",
        timeframe="1h",
        entry_rule="close > sma_short",
        exit_rule="stop_hit",
        category="trend_continuation",
        stop_anchor_type="htf_daily_low",
        target_anchor_type=target_anchor,
        estimated_bars_to_resolution=estimated_bars,
    )


def _short_trigger(
    trigger_id: str = "eth_short",
    symbol: str = "ETH-USD",
) -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol=symbol,
        direction="short",
        timeframe="1h",
        entry_rule="close < sma_short",
        exit_rule="stop_hit",
        category="trend_continuation",
        stop_anchor_type="htf_daily_high",
    )


def _flat_trigger() -> TriggerCondition:
    return TriggerCondition(
        id="btc_flat",
        symbol="BTC-USD",
        direction="flat",
        timeframe="1h",
        entry_rule="not is_flat",
        exit_rule="stop_hit",
        category="risk_off",
        stop_loss_pct=2.0,
    )


def _base_contract(**overrides) -> PositionExitContract:
    defaults = dict(
        position_id="pos_001",
        symbol="BTC-USD",
        side="long",
        created_at=_NOW,
        source_trigger_id="btc_long",
        entry_price=100.0,
        initial_qty=1.0,
        stop_price_abs=95.0,
    )
    defaults.update(overrides)
    return PositionExitContract(**defaults)


# ---------------------------------------------------------------------------
# ExitLeg schema validation
# ---------------------------------------------------------------------------


def test_exit_leg_price_level_requires_price_abs():
    with pytest.raises(ValueError, match="price_abs is required"):
        ExitLeg(kind="full_exit", trigger_mode="price_level", fraction=1.0)


def test_exit_leg_r_multiple_requires_r_multiple():
    with pytest.raises(ValueError, match="r_multiple is required"):
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0)


def test_exit_leg_r_multiple_must_be_positive():
    with pytest.raises(ValueError, match="r_multiple must be positive"):
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=-1.0)


def test_exit_leg_fraction_bounds():
    with pytest.raises(ValueError):
        ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=0.0, r_multiple=2.0)
    with pytest.raises(ValueError):
        ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=1.1, r_multiple=2.0)


def test_exit_leg_valid_price_level():
    leg = ExitLeg(kind="full_exit", trigger_mode="price_level", fraction=1.0, price_abs=110.0)
    assert leg.price_abs == 110.0
    assert leg.enabled is True
    assert leg.fired is False


def test_exit_leg_valid_r_multiple():
    leg = ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=0.5, r_multiple=2.0)
    assert leg.r_multiple == 2.0
    assert leg.fraction == 0.5


def test_exit_leg_valid_time_exit():
    leg = ExitLeg(kind="time_exit", trigger_mode="time", fraction=1.0)
    assert leg.kind == "time_exit"


# ---------------------------------------------------------------------------
# TimeExitRule schema validation
# ---------------------------------------------------------------------------


def test_time_exit_requires_at_least_one_limit():
    with pytest.raises(ValueError, match="at least one"):
        TimeExitRule()


def test_time_exit_valid_max_hold_bars():
    ter = TimeExitRule(max_hold_bars=20)
    assert ter.max_hold_bars == 20
    assert ter.session_boundary_action == "reassess"


def test_time_exit_valid_max_hold_minutes():
    ter = TimeExitRule(max_hold_minutes=120)
    assert ter.max_hold_minutes == 120


def test_time_exit_both_fields():
    ter = TimeExitRule(max_hold_bars=10, max_hold_minutes=60)
    assert ter.max_hold_bars == 10
    assert ter.max_hold_minutes == 60


# ---------------------------------------------------------------------------
# PositionExitContract schema validation
# ---------------------------------------------------------------------------


def test_contract_long_stop_must_be_below_entry():
    with pytest.raises(ValueError, match="below entry_price"):
        _base_contract(entry_price=100.0, stop_price_abs=105.0, side="long")


def test_contract_short_stop_must_be_above_entry():
    with pytest.raises(ValueError, match="above entry_price"):
        PositionExitContract(
            position_id="pos_002",
            symbol="ETH-USD",
            side="short",
            created_at=_NOW,
            source_trigger_id="eth_short",
            entry_price=100.0,
            initial_qty=1.0,
            stop_price_abs=95.0,  # below entry — wrong for short
        )


def test_contract_partial_leg_fractions_exceed_one():
    legs = [
        ExitLeg(kind="risk_reduce", trigger_mode="r_multiple", fraction=0.6, r_multiple=2.0),
        ExitLeg(kind="risk_reduce", trigger_mode="r_multiple", fraction=0.6, r_multiple=3.0),
    ]
    with pytest.raises(ValueError, match="fractions sum"):
        _base_contract(target_legs=legs)


def test_contract_full_exit_legs_not_summed():
    """full_exit legs are exempt from the fraction-sum check."""
    legs = [
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=2.0),
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=3.0),
    ]
    # Should not raise
    c = _base_contract(target_legs=legs)
    assert len(c.target_legs) == 2


def test_contract_stop_r_distance_long():
    c = _base_contract(entry_price=100.0, stop_price_abs=95.0, side="long")
    assert c.stop_r_distance == pytest.approx(5.0)


def test_contract_stop_r_distance_short():
    c = PositionExitContract(
        position_id="p",
        symbol="X",
        side="short",
        created_at=_NOW,
        source_trigger_id="t",
        entry_price=100.0,
        initial_qty=1.0,
        stop_price_abs=107.0,
    )
    assert c.stop_r_distance == pytest.approx(7.0)


def test_contract_compute_r_multiple_long():
    c = _base_contract(entry_price=100.0, stop_price_abs=95.0)
    assert c.compute_r_multiple(110.0) == pytest.approx(2.0)  # (110-100)/5


def test_contract_compute_r_multiple_short():
    c = PositionExitContract(
        position_id="p",
        symbol="X",
        side="short",
        created_at=_NOW,
        source_trigger_id="t",
        entry_price=100.0,
        initial_qty=1.0,
        stop_price_abs=105.0,
    )
    assert c.compute_r_multiple(90.0) == pytest.approx(2.0)  # (100-90)/5


def test_contract_active_legs_excludes_fired():
    legs = [
        ExitLeg(kind="take_profit", trigger_mode="r_multiple", fraction=0.5, r_multiple=2.0, fired=True),
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=3.0),
    ]
    c = _base_contract(target_legs=legs)
    assert len(c.active_legs) == 1
    assert c.active_legs[0].r_multiple == 3.0


def test_contract_has_price_target_true():
    c = _base_contract(target_legs=[
        ExitLeg(kind="full_exit", trigger_mode="r_multiple", fraction=1.0, r_multiple=2.0)
    ])
    assert c.has_price_target is True


def test_contract_has_price_target_false_when_empty():
    c = _base_contract()
    assert c.has_price_target is False


def test_contract_version_constant():
    c = _base_contract()
    assert c.contract_version == POSITION_EXIT_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# PortfolioMetaRiskPolicy
# ---------------------------------------------------------------------------


def test_portfolio_meta_risk_policy_minimal():
    p = PortfolioMetaRiskPolicy(policy_id="policy_001")
    assert p.enabled is True
    assert p.actions == []


def test_portfolio_meta_action_kind_values():
    kinds = [
        "trim_largest_position_to_cap",
        "reduce_gross_exposure_pct",
        "rebalance_to_cash_pct",
        "freeze_new_entries",
        "tighten_position_stops",
    ]
    for kind in kinds:
        a = PortfolioMetaAction(condition_id="cond_01", kind=kind)
        assert a.kind == kind


def test_portfolio_meta_risk_policy_with_actions():
    action = PortfolioMetaAction(
        condition_id="concentration_breach",
        kind="trim_largest_position_to_cap",
        params={"cap_pct": 20.0},
        cooldown_minutes=30,
    )
    policy = PortfolioMetaRiskPolicy(
        policy_id="default_policy",
        max_symbol_concentration_pct=25.0,
        portfolio_drawdown_reduce_threshold_pct=5.0,
        actions=[action],
    )
    assert len(policy.actions) == 1
    assert policy.max_symbol_concentration_pct == 25.0


# ---------------------------------------------------------------------------
# build_exit_contract
# ---------------------------------------------------------------------------


def test_build_exit_contract_non_entry_returns_none():
    t = _flat_trigger()
    result = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0)
    assert result is None


def test_build_exit_contract_long_minimal():
    t = _long_trigger()
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert c.side == "long"
    assert c.entry_price == 100.0
    assert c.stop_price_abs == 95.0
    assert c.initial_qty == 1.0
    assert c.remaining_qty == 1.0
    assert c.source_trigger_id == "btc_long"


def test_build_exit_contract_short_minimal():
    t = _short_trigger()
    c = build_exit_contract(t, "pos_002", 100.0, 0.5, 107.0, created_at=_NOW)
    assert c is not None
    assert c.side == "short"
    assert c.stop_price_abs == 107.0


def test_build_exit_contract_r_multiple_2_anchor():
    t = _long_trigger(target_anchor="r_multiple_2")
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert len(c.target_legs) == 1
    leg = c.target_legs[0]
    assert leg.trigger_mode == "r_multiple"
    assert leg.r_multiple == 2.0
    assert leg.kind == "full_exit"


def test_build_exit_contract_r_multiple_3_anchor():
    t = _long_trigger(target_anchor="r_multiple_3")
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert c.target_legs[0].r_multiple == 3.0


def test_build_exit_contract_price_level_with_target_price():
    t = _long_trigger(target_anchor="htf_daily_high")
    c = build_exit_contract(
        t, "pos_001", 100.0, 1.0, 95.0,
        target_price_abs=115.0,
        created_at=_NOW,
    )
    assert c is not None
    assert len(c.target_legs) == 1
    assert c.target_legs[0].trigger_mode == "price_level"
    assert c.target_legs[0].price_abs == 115.0


def test_build_exit_contract_price_level_without_target_price_no_leg():
    """If caller doesn't provide resolved price, no leg is created (logged as warning)."""
    t = _long_trigger(target_anchor="htf_daily_high")
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert len(c.target_legs) == 0  # no leg without resolved price


def test_build_exit_contract_no_anchor_no_leg():
    t = _long_trigger(target_anchor=None)
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert c.target_legs == []


def test_build_exit_contract_max_hold_bars_creates_time_exit():
    t = _long_trigger()
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, max_hold_bars=20, created_at=_NOW)
    assert c is not None
    assert c.time_exit is not None
    assert c.time_exit.max_hold_bars == 20


def test_build_exit_contract_estimated_bars_creates_conservative_time_exit():
    """2× estimated_bars_to_resolution used as conservative expiry."""
    t = _long_trigger(estimated_bars=6)
    c = build_exit_contract(t, "pos_001", 100.0, 1.0, 95.0, created_at=_NOW)
    assert c is not None
    assert c.time_exit is not None
    assert c.time_exit.max_hold_bars == 12  # 2 × 6


def test_build_exit_contract_max_hold_bars_overrides_estimated():
    """Explicit max_hold_bars takes priority over estimated_bars_to_resolution."""
    t = _long_trigger(estimated_bars=6)
    c = build_exit_contract(
        t, "pos_001", 100.0, 1.0, 95.0,
        max_hold_bars=30,
        created_at=_NOW,
    )
    assert c.time_exit.max_hold_bars == 30


def test_build_exit_contract_provenance_fields():
    t = _long_trigger()
    snap_id = str(uuid4())
    c = build_exit_contract(
        t, "pos_001", 100.0, 1.0, 95.0,
        plan_id="plan_abc",
        playbook_id="playbook_donchian",
        template_id="compression_breakout",
        snapshot_id=snap_id,
        snapshot_hash="deadbeef" * 8,
        created_at=_NOW,
    )
    assert c.source_plan_id == "plan_abc"
    assert c.playbook_id == "playbook_donchian"
    assert c.template_id == "compression_breakout"
    assert c.snapshot_id == snap_id


# ---------------------------------------------------------------------------
# can_build_contract
# ---------------------------------------------------------------------------


def test_can_build_contract_long_valid():
    t = _long_trigger()
    ok, reason = can_build_contract(t, 100.0, 95.0)
    assert ok is True
    assert reason == ""


def test_can_build_contract_flat_returns_false():
    t = _flat_trigger()
    ok, reason = can_build_contract(t, 100.0, 95.0)
    assert ok is False
    assert "not an entry direction" in reason


def test_can_build_contract_long_stop_above_entry_returns_false():
    t = _long_trigger()
    ok, reason = can_build_contract(t, 100.0, 105.0)
    assert ok is False
    assert "below" in reason


def test_can_build_contract_short_stop_below_entry_returns_false():
    t = _short_trigger()
    ok, reason = can_build_contract(t, 100.0, 95.0)
    assert ok is False
    assert "above" in reason


def test_can_build_contract_zero_entry_price_returns_false():
    t = _long_trigger()
    ok, reason = can_build_contract(t, 0.0, 95.0)
    assert ok is False
    assert "positive" in reason


# ---------------------------------------------------------------------------
# Registry sanity checks
# ---------------------------------------------------------------------------


def test_r_multiple_anchor_map_has_expected_keys():
    assert "r_multiple_2" in _R_MULTIPLE_ANCHOR_MAP
    assert "r_multiple_3" in _R_MULTIPLE_ANCHOR_MAP
    assert _R_MULTIPLE_ANCHOR_MAP["r_multiple_2"] == 2.0
    assert _R_MULTIPLE_ANCHOR_MAP["r_multiple_3"] == 3.0


def test_price_level_anchors_has_htf_fields():
    assert "htf_daily_high" in _PRICE_LEVEL_ANCHORS
    assert "htf_5d_high" in _PRICE_LEVEL_ANCHORS
    assert "measured_move" in _PRICE_LEVEL_ANCHORS


def test_exit_contract_builder_version_is_set():
    assert EXIT_CONTRACT_BUILDER_VERSION
