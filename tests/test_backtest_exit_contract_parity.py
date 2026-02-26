"""Tests for Runbook 60 backtest parity: exit contract materialization in backtests.

Verifies that:
- _materialize_backtest_contract builds a valid PositionExitContract from
  resolved stop/target prices in position_meta.
- Contract is stored keyed by symbol in exit_contracts dict.
- Contracts are appended to the audit log (_exit_contract_audit).
- A target leg is added when target_price_abs is present.
- When stop is missing or invalid, no contract is created (graceful skip).
- StrategistBacktestResult includes exit_contracts field.
- M3 warning logic (backtest): entry-rule flatten with active contract is detected.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from schemas.position_exit_contract import ExitLeg, PositionExitContract, TimeExitRule


# ---------------------------------------------------------------------------
# Helpers: minimal mock of the relevant backtester state
# ---------------------------------------------------------------------------


def _make_order(
    symbol: str = "BTC-USD",
    side: str = "buy",
    price: float = 50000.0,
    quantity: float = 0.1,
    reason: str = "btc_long",
    timestamp: Optional[datetime] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        reason=reason,
        timestamp=timestamp or datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


def _make_portfolio(
    stop_price_abs: Optional[float],
    target_price_abs: Optional[float] = None,
    symbol: str = "BTC-USD",
) -> SimpleNamespace:
    meta: Dict[str, Any] = {}
    if stop_price_abs is not None:
        meta["stop_price_abs"] = stop_price_abs
    if target_price_abs is not None:
        meta["target_price_abs"] = target_price_abs
    return SimpleNamespace(position_meta={symbol: meta})


class _MockBacktester:
    """Minimal stub of LLMStrategistBacktester for contract materialization tests."""

    def __init__(self, portfolio: SimpleNamespace) -> None:
        self.portfolio = portfolio
        self.exit_contracts: Dict[str, dict] = {}
        self._exit_contract_audit: List[Dict[str, Any]] = []

    def _materialize_backtest_contract(self, order: SimpleNamespace) -> None:
        """Exact copy of the production implementation from llm_strategist_runner.py."""
        _bt_stop = self.portfolio.position_meta.get(order.symbol, {}).get("stop_price_abs")
        if not (_bt_stop and _bt_stop > 0):
            return
        try:
            _bt_side = "long" if order.side == "buy" else "short"
            _bt_target = self.portfolio.position_meta.get(order.symbol, {}).get("target_price_abs")
            _bt_legs: list = []
            if _bt_target and _bt_target > 0:
                _bt_legs.append(ExitLeg(
                    kind="full_exit",
                    trigger_mode="price_level",
                    fraction=1.0,
                    price_abs=_bt_target,
                ))
            _bt_contract = PositionExitContract(
                position_id=f"bt_{order.symbol}_{order.timestamp.strftime('%Y%m%d%H%M%S')}",
                symbol=order.symbol,
                side=_bt_side,
                created_at=order.timestamp,
                source_trigger_id=order.reason or "unknown",
                entry_price=order.price,
                initial_qty=order.quantity,
                stop_price_abs=_bt_stop,
                target_legs=_bt_legs,
                remaining_qty=order.quantity,
            )
            _contract_dict = _bt_contract.model_dump(mode="json")
            self.exit_contracts[order.symbol] = _contract_dict
            self._exit_contract_audit.append(_contract_dict)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Contract creation tests
# ---------------------------------------------------------------------------


class TestMaterializeBacktestContract:
    def test_basic_long_contract_created(self) -> None:
        order = _make_order(symbol="BTC-USD", side="buy", price=50000.0, quantity=0.1)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        assert "BTC-USD" in backtester.exit_contracts
        c = backtester.exit_contracts["BTC-USD"]
        assert c["side"] == "long"
        assert c["entry_price"] == 50000.0
        assert c["stop_price_abs"] == 48000.0
        assert c["initial_qty"] == 0.1
        assert c["symbol"] == "BTC-USD"

    def test_basic_short_contract_created(self) -> None:
        order = _make_order(symbol="ETH-USD", side="sell", price=3000.0, quantity=1.0)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=3100.0, symbol="ETH-USD"))
        backtester._materialize_backtest_contract(order)

        assert "ETH-USD" in backtester.exit_contracts
        c = backtester.exit_contracts["ETH-USD"]
        assert c["side"] == "short"
        assert c["stop_price_abs"] == 3100.0

    def test_contract_with_target_leg(self) -> None:
        order = _make_order(price=50000.0)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0, target_price_abs=55000.0))
        backtester._materialize_backtest_contract(order)

        c = backtester.exit_contracts["BTC-USD"]
        assert len(c["target_legs"]) == 1
        leg = c["target_legs"][0]
        assert leg["trigger_mode"] == "price_level"
        assert leg["price_abs"] == 55000.0
        assert leg["kind"] == "full_exit"
        assert leg["fraction"] == 1.0

    def test_contract_without_target_has_empty_legs(self) -> None:
        order = _make_order(price=50000.0)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        c = backtester.exit_contracts["BTC-USD"]
        assert c["target_legs"] == []

    def test_position_id_includes_symbol_and_timestamp(self) -> None:
        ts = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        order = _make_order(timestamp=ts)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        c = backtester.exit_contracts["BTC-USD"]
        assert "BTC-USD" in c["position_id"]
        assert "20240601" in c["position_id"]

    def test_source_trigger_id_from_order_reason(self) -> None:
        order = _make_order(reason="compression_breakout_long")
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        c = backtester.exit_contracts["BTC-USD"]
        assert c["source_trigger_id"] == "compression_breakout_long"

    def test_unknown_trigger_id_when_reason_is_none(self) -> None:
        order = _make_order(reason=None)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        c = backtester.exit_contracts["BTC-USD"]
        assert c["source_trigger_id"] == "unknown"


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_contract_appended_to_audit_list(self) -> None:
        order = _make_order()
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        assert len(backtester._exit_contract_audit) == 1

    def test_multiple_entries_in_audit_log(self) -> None:
        order1 = _make_order(symbol="BTC-USD", timestamp=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))
        order2 = _make_order(symbol="BTC-USD", timestamp=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc))
        portfolio = _make_portfolio(stop_price_abs=48000.0)
        backtester = _MockBacktester(portfolio)

        backtester._materialize_backtest_contract(order1)
        backtester._materialize_backtest_contract(order2)

        assert len(backtester._exit_contract_audit) == 2

    def test_audit_log_preserved_after_contract_cleanup(self) -> None:
        order = _make_order()
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0))
        backtester._materialize_backtest_contract(order)

        # Simulate position flat: cleanup active contract but audit remains
        backtester.exit_contracts.pop("BTC-USD", None)

        assert len(backtester._exit_contract_audit) == 1  # audit log survives


# ---------------------------------------------------------------------------
# Graceful skip tests (no contract when stop is missing/invalid)
# ---------------------------------------------------------------------------


class TestGracefulSkip:
    def test_no_contract_when_stop_is_none(self) -> None:
        order = _make_order()
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=None))
        backtester._materialize_backtest_contract(order)

        assert "BTC-USD" not in backtester.exit_contracts
        assert len(backtester._exit_contract_audit) == 0

    def test_no_contract_when_stop_is_zero(self) -> None:
        order = _make_order()
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=0.0))
        backtester._materialize_backtest_contract(order)

        assert "BTC-USD" not in backtester.exit_contracts

    def test_no_contract_when_symbol_not_in_meta(self) -> None:
        order = _make_order(symbol="SOL-USD")
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0, symbol="BTC-USD"))
        backtester._materialize_backtest_contract(order)

        assert "SOL-USD" not in backtester.exit_contracts

    def test_invalid_target_skips_leg_gracefully(self) -> None:
        order = _make_order(price=50000.0)
        backtester = _MockBacktester(_make_portfolio(stop_price_abs=48000.0, target_price_abs=0.0))
        backtester._materialize_backtest_contract(order)

        # Contract is still created — just without a target leg
        assert "BTC-USD" in backtester.exit_contracts
        assert backtester.exit_contracts["BTC-USD"]["target_legs"] == []


# ---------------------------------------------------------------------------
# StrategistBacktestResult field test
# ---------------------------------------------------------------------------


class TestStrategyBacktestResultField:
    def test_exit_contracts_field_exists_and_defaults_empty(self) -> None:
        from dataclasses import fields
        from backtesting.llm_strategist_runner import StrategistBacktestResult

        field_names = {f.name for f in fields(StrategistBacktestResult)}
        assert "exit_contracts" in field_names, (
            "StrategistBacktestResult must have an 'exit_contracts' field (R60 backtest parity)"
        )

    def test_exit_contracts_field_default_is_list(self) -> None:
        import pandas as pd
        from backtesting.llm_strategist_runner import StrategistBacktestResult

        result = StrategistBacktestResult(
            equity_curve=pd.Series(dtype=float),
            fills=pd.DataFrame(),
            plan_log=[],
            summary={},
            llm_costs={},
            final_cash=1000.0,
            final_positions={},
            daily_reports=[],
            bar_decisions={},
        )
        assert result.exit_contracts == []


# ---------------------------------------------------------------------------
# M3 backtest warning: detection logic
# ---------------------------------------------------------------------------


def _detect_m3_backtest_warning(
    order_reason: Optional[str],
    symbol: str,
    exit_contracts: Dict[str, Any],
) -> bool:
    """Mirrors the M3 backtest warning condition from llm_strategist_runner.py.

    Returns True if a warning would be logged (entry-rule flatten with active contract).
    """
    return bool(
        order_reason
        and order_reason.endswith("_flat")
        and symbol in exit_contracts
    )


class TestM3BacktestWarning:
    def test_warning_detected_for_flat_with_active_contract(self) -> None:
        contracts = {"BTC-USD": {"contract_id": "abc"}}
        assert _detect_m3_backtest_warning("risk_off_flat", "BTC-USD", contracts)

    def test_no_warning_when_no_active_contract(self) -> None:
        assert not _detect_m3_backtest_warning("risk_off_flat", "BTC-USD", {})

    def test_no_warning_for_non_flat_exit(self) -> None:
        contracts = {"BTC-USD": {"contract_id": "abc"}}
        assert not _detect_m3_backtest_warning("below_stop", "BTC-USD", contracts)

    def test_no_warning_when_reason_is_none(self) -> None:
        contracts = {"BTC-USD": {"contract_id": "abc"}}
        assert not _detect_m3_backtest_warning(None, "BTC-USD", contracts)

    def test_no_warning_for_different_symbol(self) -> None:
        contracts = {"ETH-USD": {"contract_id": "abc"}}
        assert not _detect_m3_backtest_warning("risk_off_flat", "BTC-USD", contracts)
