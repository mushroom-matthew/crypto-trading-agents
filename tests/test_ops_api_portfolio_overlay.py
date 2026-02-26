"""Tests for Runbook 60 ops API: exit contract response models and converter.

Verifies that:
- _contract_dict_to_response produces a valid PositionExitContractResponse
  from a raw contract dict (as stored in workflow state).
- ExitContractsSessionResponse correctly aggregates multiple contracts.
- exit_class is always "strategy_contract" for all ops API responses.
- stop_r_distance, has_price_target, and active_legs_count are computed correctly.
- ExitLegResponse fields are mapped correctly.
- TimeExitRuleResponse fields are mapped correctly.
- None / missing optional fields are handled gracefully.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from ops_api.routers.exit_contracts import (
    ExitContractsSessionResponse,
    ExitLegResponse,
    PositionExitContractResponse,
    TimeExitRuleResponse,
    _contract_dict_to_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contract_dict(
    symbol: str = "BTC-USD",
    side: str = "long",
    entry_price: float = 50000.0,
    stop_price_abs: float = 48000.0,
    target_price_abs: Optional[float] = None,
    source_trigger_id: str = "btc_long",
    source_category: Optional[str] = "trend_continuation",
    remaining_qty: Optional[float] = 0.1,
    amendment_policy: str = "tighten_only",
    allow_portfolio_overlay_trims: bool = True,
    target_legs: Optional[List[Dict[str, Any]]] = None,
    time_exit: Optional[Dict[str, Any]] = None,
    playbook_id: Optional[str] = None,
    template_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "contract_id": f"cid_{symbol}",
        "contract_version": "1.0.0",
        "position_id": f"pos_{symbol}",
        "symbol": symbol,
        "side": side,
        "created_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
        "entry_price": entry_price,
        "initial_qty": 0.1,
        "stop_price_abs": stop_price_abs,
        "remaining_qty": remaining_qty,
        "source_trigger_id": source_trigger_id,
        "source_category": source_category,
        "amendment_policy": amendment_policy,
        "allow_portfolio_overlay_trims": allow_portfolio_overlay_trims,
        "target_legs": target_legs or [],
        "time_exit": time_exit,
        "playbook_id": playbook_id,
        "template_id": template_id,
    }


def _make_price_level_leg(
    price_abs: float = 55000.0,
    kind: str = "take_profit",
    fraction: float = 0.5,
    enabled: bool = True,
    fired: bool = False,
) -> Dict[str, Any]:
    return {
        "leg_id": "leg_abc123",
        "kind": kind,
        "trigger_mode": "price_level",
        "fraction": fraction,
        "price_abs": price_abs,
        "r_multiple": None,
        "priority": 0,
        "enabled": enabled,
        "fired": fired,
    }


def _make_r_multiple_leg(r_multiple: float = 2.0) -> Dict[str, Any]:
    return {
        "leg_id": "leg_def456",
        "kind": "full_exit",
        "trigger_mode": "r_multiple",
        "fraction": 1.0,
        "price_abs": None,
        "r_multiple": r_multiple,
        "priority": 0,
        "enabled": True,
        "fired": False,
    }


# ---------------------------------------------------------------------------
# _contract_dict_to_response basic field mapping
# ---------------------------------------------------------------------------


class TestContractDictToResponse:
    def test_basic_fields_mapped(self) -> None:
        raw = _make_contract_dict(
            symbol="BTC-USD",
            entry_price=50000.0,
            stop_price_abs=48000.0,
            source_trigger_id="btc_long",
        )
        resp = _contract_dict_to_response(raw)

        assert resp.symbol == "BTC-USD"
        assert resp.side == "long"
        assert resp.entry_price == 50000.0
        assert resp.stop_price_abs == 48000.0
        assert resp.source_trigger_id == "btc_long"
        assert resp.contract_id == "cid_BTC-USD"

    def test_exit_class_is_always_strategy_contract(self) -> None:
        raw = _make_contract_dict()
        resp = _contract_dict_to_response(raw)
        assert resp.exit_class == "strategy_contract"

    def test_amendment_policy_preserved(self) -> None:
        raw = _make_contract_dict(amendment_policy="none")
        resp = _contract_dict_to_response(raw)
        assert resp.amendment_policy == "none"

    def test_remaining_qty_preserved(self) -> None:
        raw = _make_contract_dict(remaining_qty=0.05)
        resp = _contract_dict_to_response(raw)
        assert resp.remaining_qty == 0.05

    def test_source_category_preserved(self) -> None:
        raw = _make_contract_dict(source_category="trend_continuation")
        resp = _contract_dict_to_response(raw)
        assert resp.source_category == "trend_continuation"

    def test_playbook_and_template_ids_preserved(self) -> None:
        raw = _make_contract_dict(playbook_id="rsi_extremes", template_id="aggressive_long")
        resp = _contract_dict_to_response(raw)
        assert resp.playbook_id == "rsi_extremes"
        assert resp.template_id == "aggressive_long"

    def test_allow_portfolio_overlay_trims_preserved(self) -> None:
        raw = _make_contract_dict(allow_portfolio_overlay_trims=False)
        resp = _contract_dict_to_response(raw)
        assert resp.allow_portfolio_overlay_trims is False


# ---------------------------------------------------------------------------
# stop_r_distance computation
# ---------------------------------------------------------------------------


class TestStopRDistance:
    def test_stop_r_distance_computed(self) -> None:
        raw = _make_contract_dict(entry_price=50000.0, stop_price_abs=48000.0)
        resp = _contract_dict_to_response(raw)
        assert resp.stop_r_distance == pytest.approx(2000.0)

    def test_stop_r_distance_for_short(self) -> None:
        # For shorts, stop is above entry, distance is |entry - stop|
        raw = _make_contract_dict(
            side="short", entry_price=3000.0, stop_price_abs=3200.0
        )
        resp = _contract_dict_to_response(raw)
        assert resp.stop_r_distance == pytest.approx(200.0)

    def test_stop_r_distance_none_when_prices_missing(self) -> None:
        raw = _make_contract_dict(entry_price=0.0, stop_price_abs=0.0)
        resp = _contract_dict_to_response(raw)
        assert resp.stop_r_distance is None


# ---------------------------------------------------------------------------
# has_price_target and active_legs_count
# ---------------------------------------------------------------------------


class TestPriceTargetAndActiveLegCount:
    def test_has_price_target_false_when_no_legs(self) -> None:
        raw = _make_contract_dict(target_legs=[])
        resp = _contract_dict_to_response(raw)
        assert resp.has_price_target is False
        assert resp.active_legs_count == 0

    def test_has_price_target_true_with_price_level_leg(self) -> None:
        raw = _make_contract_dict(target_legs=[_make_price_level_leg()])
        resp = _contract_dict_to_response(raw)
        assert resp.has_price_target is True
        assert resp.active_legs_count == 1

    def test_has_price_target_true_with_r_multiple_leg(self) -> None:
        raw = _make_contract_dict(target_legs=[_make_r_multiple_leg()])
        resp = _contract_dict_to_response(raw)
        assert resp.has_price_target is True

    def test_fired_leg_does_not_count_as_active(self) -> None:
        fired_leg = _make_price_level_leg(fired=True)
        raw = _make_contract_dict(target_legs=[fired_leg])
        resp = _contract_dict_to_response(raw)
        assert resp.active_legs_count == 0
        assert resp.has_price_target is False

    def test_disabled_leg_does_not_count_as_active(self) -> None:
        disabled_leg = _make_price_level_leg(enabled=False)
        raw = _make_contract_dict(target_legs=[disabled_leg])
        resp = _contract_dict_to_response(raw)
        assert resp.active_legs_count == 0

    def test_mixed_active_and_fired_legs(self) -> None:
        legs = [
            _make_price_level_leg(price_abs=52000.0, fired=False),
            _make_price_level_leg(price_abs=55000.0, fired=True),
        ]
        raw = _make_contract_dict(target_legs=legs)
        resp = _contract_dict_to_response(raw)
        assert resp.active_legs_count == 1


# ---------------------------------------------------------------------------
# ExitLegResponse field mapping
# ---------------------------------------------------------------------------


class TestExitLegResponseMapping:
    def test_price_level_leg_fields_mapped(self) -> None:
        leg_dict = _make_price_level_leg(price_abs=55000.0, fraction=0.5, kind="take_profit")
        raw = _make_contract_dict(target_legs=[leg_dict])
        resp = _contract_dict_to_response(raw)

        assert len(resp.target_legs) == 1
        leg = resp.target_legs[0]
        assert leg.kind == "take_profit"
        assert leg.trigger_mode == "price_level"
        assert leg.fraction == 0.5
        assert leg.price_abs == 55000.0
        assert leg.r_multiple is None

    def test_r_multiple_leg_fields_mapped(self) -> None:
        leg_dict = _make_r_multiple_leg(r_multiple=2.0)
        raw = _make_contract_dict(target_legs=[leg_dict])
        resp = _contract_dict_to_response(raw)

        leg = resp.target_legs[0]
        assert leg.trigger_mode == "r_multiple"
        assert leg.r_multiple == 2.0
        assert leg.price_abs is None

    def test_leg_enabled_and_fired_preserved(self) -> None:
        leg_dict = _make_price_level_leg(enabled=False, fired=True)
        raw = _make_contract_dict(target_legs=[leg_dict])
        resp = _contract_dict_to_response(raw)

        leg = resp.target_legs[0]
        assert leg.enabled is False
        assert leg.fired is True

    def test_multiple_legs_in_order(self) -> None:
        legs = [
            _make_price_level_leg(price_abs=52000.0, fraction=0.5, kind="take_profit"),
            _make_r_multiple_leg(r_multiple=3.0),
        ]
        raw = _make_contract_dict(target_legs=legs)
        resp = _contract_dict_to_response(raw)

        assert len(resp.target_legs) == 2
        assert resp.target_legs[0].trigger_mode == "price_level"
        assert resp.target_legs[1].trigger_mode == "r_multiple"


# ---------------------------------------------------------------------------
# TimeExitRuleResponse
# ---------------------------------------------------------------------------


class TestTimeExitRuleResponse:
    def test_time_exit_none_when_absent(self) -> None:
        raw = _make_contract_dict(time_exit=None)
        resp = _contract_dict_to_response(raw)
        assert resp.time_exit is None

    def test_time_exit_max_hold_bars(self) -> None:
        raw = _make_contract_dict(time_exit={"max_hold_bars": 48, "session_boundary_action": "hold"})
        resp = _contract_dict_to_response(raw)

        assert resp.time_exit is not None
        assert resp.time_exit.max_hold_bars == 48
        assert resp.time_exit.session_boundary_action == "hold"
        assert resp.time_exit.max_hold_minutes is None

    def test_time_exit_max_hold_minutes(self) -> None:
        raw = _make_contract_dict(time_exit={"max_hold_minutes": 120, "session_boundary_action": "flatten"})
        resp = _contract_dict_to_response(raw)

        assert resp.time_exit is not None
        assert resp.time_exit.max_hold_minutes == 120


# ---------------------------------------------------------------------------
# ExitContractsSessionResponse
# ---------------------------------------------------------------------------


class TestExitContractsSessionResponse:
    def test_empty_session_response(self) -> None:
        response = ExitContractsSessionResponse(
            session_id="sess-001",
            contracts=[],
            total_count=0,
        )
        assert response.session_id == "sess-001"
        assert response.contracts == []
        assert response.total_count == 0

    def test_session_response_with_multiple_contracts(self) -> None:
        c1 = _contract_dict_to_response(_make_contract_dict(symbol="BTC-USD"))
        c2 = _contract_dict_to_response(_make_contract_dict(symbol="ETH-USD"))

        response = ExitContractsSessionResponse(
            session_id="sess-002",
            contracts=[c1, c2],
            total_count=2,
        )
        assert response.total_count == 2
        assert {c.symbol for c in response.contracts} == {"BTC-USD", "ETH-USD"}

    def test_all_contracts_have_strategy_contract_class(self) -> None:
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        contracts = [
            _contract_dict_to_response(_make_contract_dict(symbol=s)) for s in symbols
        ]
        response = ExitContractsSessionResponse(
            session_id="sess-003",
            contracts=contracts,
            total_count=len(contracts),
        )
        assert all(c.exit_class == "strategy_contract" for c in response.contracts)

    def test_contract_dict_to_response_roundtrip_from_workflow_state(self) -> None:
        """Simulate realistic workflow state dict → response pipeline."""
        from schemas.position_exit_contract import ExitLeg, PositionExitContract, TimeExitRule

        ts = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        contract = PositionExitContract(
            position_id="pos-001",
            symbol="BTC-USD",
            side="long",
            created_at=ts,
            source_trigger_id="btc_breakout_long",
            source_category="trend_continuation",
            entry_price=50000.0,
            initial_qty=0.1,
            stop_price_abs=48000.0,
            remaining_qty=0.1,
            target_legs=[
                ExitLeg(
                    kind="take_profit",
                    trigger_mode="price_level",
                    fraction=0.5,
                    price_abs=54000.0,
                ),
                ExitLeg(
                    kind="full_exit",
                    trigger_mode="r_multiple",
                    fraction=1.0,
                    r_multiple=2.0,
                ),
            ],
            time_exit=TimeExitRule(max_hold_bars=96),
        )
        # Simulate what workflow state stores
        state_dict = contract.model_dump(mode="json")
        resp = _contract_dict_to_response(state_dict)

        assert resp.symbol == "BTC-USD"
        assert resp.side == "long"
        assert resp.entry_price == 50000.0
        assert resp.stop_r_distance == pytest.approx(2000.0)
        assert resp.has_price_target is True
        assert resp.active_legs_count == 2
        assert resp.time_exit is not None
        assert resp.time_exit.max_hold_bars == 96
        assert resp.exit_class == "strategy_contract"
