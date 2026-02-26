"""Tests for Runbook 60 Phase M3: contract-backed exit enforcement in paper trading.

Verifies that:
- Orders with reason ending '_flat', non-emergency, with an active exit contract
  are detected as M3-blocked exits.
- Emergency exits are NOT blocked (bypass all constraints).
- Symbols WITHOUT an active exit contract are NOT blocked (no contract = no guard).
- Orders that are NOT 'exit' intent are NOT blocked.
- The M3 event payload contains required audit fields (symbol, trigger_id, reason,
  exit_class, detail).
- Exit contract cleanup: contract is removed from active set when position goes flat.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# M3 guardrail: detection function (mirrors the inline logic in paper_trading.py
# _evaluate_and_execute method, order processing loop)
# ---------------------------------------------------------------------------


def _detect_m3_block(
    order: Dict[str, Any],
    exit_contracts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Mirrors Phase M3 block check from paper_trading.py.

    Returns a 'trade_blocked' event dict if the order should be suppressed,
    or None if the order should proceed.
    """
    _sym = order.get("symbol", "")
    _reason = order.get("reason") or order.get("trigger_id") or ""
    if (
        order.get("intent") == "exit"
        and isinstance(_reason, str)
        and _reason.endswith("_flat")
        and order.get("trigger_category") != "emergency_exit"
        and _sym in exit_contracts
    ):
        return {
            "type": "trade_blocked",
            "payload": {
                "symbol": _sym,
                "trigger_id": _reason,
                "reason": "m3_contract_backed_exit",
                "exit_class": "strategy_contract",
                "detail": (
                    "Entry-rule flatten suppressed (Phase M3): position has an "
                    "active exit contract. Contract stop/target/time rules govern "
                    "this exit path."
                ),
            },
        }
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exit_order(
    symbol: str = "BTC-USD",
    reason: str = "risk_off_flat",
    category: str = "risk_off",
    intent: str = "exit",
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "intent": intent,
        "reason": reason,
        "trigger_category": category,
        "quantity": 0.1,
        "side": "sell",
    }


def _active_contracts(*symbols: str) -> Dict[str, Any]:
    """Simulate a non-empty exit_contracts dict for the given symbols."""
    return {sym: {"contract_id": f"cid_{sym}", "symbol": sym} for sym in symbols}


# ---------------------------------------------------------------------------
# Positive cases: should be blocked
# ---------------------------------------------------------------------------


class TestM3BlockedExits:
    """Orders that SHOULD be blocked by M3 (entry-rule flatten with active contract)."""

    def test_risk_off_flat_with_active_contract_is_blocked(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is not None
        assert result["type"] == "trade_blocked"
        assert result["payload"]["reason"] == "m3_contract_backed_exit"
        assert result["payload"]["symbol"] == "BTC-USD"
        assert result["payload"]["trigger_id"] == "risk_off_flat"

    def test_generic_flat_suffix_is_blocked(self) -> None:
        order = _exit_order(symbol="ETH-USD", reason="direction_exit_flat")
        result = _detect_m3_block(order, _active_contracts("ETH-USD"))
        assert result is not None
        assert result["payload"]["exit_class"] == "strategy_contract"

    def test_any_flat_suffix_reason_is_blocked(self) -> None:
        for reason in ["strategy_exit_flat", "trend_reversal_flat", "take_profit_flat"]:
            order = _exit_order(symbol="SOL-USD", reason=reason)
            result = _detect_m3_block(order, _active_contracts("SOL-USD"))
            assert result is not None, f"Expected block for reason='{reason}'"

    def test_fallback_trigger_id_key_is_blocked(self) -> None:
        """reason is missing but trigger_id is present."""
        order = {
            "symbol": "BTC-USD",
            "intent": "exit",
            "trigger_id": "risk_off_flat",
            "trigger_category": "risk_off",
        }
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is not None
        assert result["payload"]["trigger_id"] == "risk_off_flat"

    def test_block_detail_message_is_present(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="flatten_flat")
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is not None
        assert "Phase M3" in result["payload"]["detail"]
        assert "exit contract" in result["payload"]["detail"].lower()


# ---------------------------------------------------------------------------
# Negative cases: should NOT be blocked
# ---------------------------------------------------------------------------


class TestM3NotBlocked:
    """Orders that should NOT be blocked by M3."""

    def test_emergency_exit_bypasses_m3(self) -> None:
        order = _exit_order(
            symbol="BTC-USD",
            reason="emergency_flat",
            category="emergency_exit",
        )
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is None, "Emergency exits must bypass M3"

    def test_no_active_contract_is_not_blocked(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        result = _detect_m3_block(order, {})
        assert result is None

    def test_different_symbol_contract_does_not_block(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        result = _detect_m3_block(order, _active_contracts("ETH-USD"))
        assert result is None

    def test_non_flat_exit_not_blocked(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="below_stop_exit")
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is None

    def test_entry_intent_not_blocked(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="entry_long_flat", intent="entry")
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is None

    def test_null_reason_not_blocked(self) -> None:
        order = {
            "symbol": "BTC-USD",
            "intent": "exit",
            "reason": None,
            "trigger_category": "risk_off",
        }
        result = _detect_m3_block(order, _active_contracts("BTC-USD"))
        assert result is None

    def test_empty_exit_contracts_dict_not_blocked(self) -> None:
        order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        result = _detect_m3_block(order, {})
        assert result is None


# ---------------------------------------------------------------------------
# Multi-symbol isolation
# ---------------------------------------------------------------------------


class TestM3MultiSymbolIsolation:
    """M3 is per-symbol: a contract for one symbol must not block another."""

    def test_only_matching_symbol_is_blocked(self) -> None:
        contracts = _active_contracts("BTC-USD", "ETH-USD")
        btc_order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        sol_order = _exit_order(symbol="SOL-USD", reason="risk_off_flat")

        assert _detect_m3_block(btc_order, contracts) is not None
        assert _detect_m3_block(sol_order, contracts) is None

    def test_multiple_symbols_independently_blocked(self) -> None:
        contracts = _active_contracts("BTC-USD", "ETH-USD", "SOL-USD")
        for sym in ("BTC-USD", "ETH-USD", "SOL-USD"):
            order = _exit_order(symbol=sym, reason="risk_off_flat")
            result = _detect_m3_block(order, contracts)
            assert result is not None, f"Expected block for {sym}"
            assert result["payload"]["symbol"] == sym


# ---------------------------------------------------------------------------
# Contract cleanup on position close
# ---------------------------------------------------------------------------


class TestM3ContractCleanup:
    """When a position goes flat, the exit contract should be removed."""

    def test_contract_removed_on_flat(self) -> None:
        contracts: Dict[str, Any] = {"BTC-USD": {"contract_id": "abc", "symbol": "BTC-USD"}}
        assert "BTC-USD" in contracts

        # Simulate the cleanup that happens on position flat
        contracts.pop("BTC-USD", None)
        assert "BTC-USD" not in contracts

    def test_pop_on_nonexistent_symbol_is_safe(self) -> None:
        contracts: Dict[str, Any] = {}
        contracts.pop("BTC-USD", None)  # Must not raise

    def test_after_cleanup_flat_order_not_blocked(self) -> None:
        contracts: Dict[str, Any] = {"BTC-USD": {"contract_id": "abc", "symbol": "BTC-USD"}}
        contracts.pop("BTC-USD", None)

        order = _exit_order(symbol="BTC-USD", reason="risk_off_flat")
        result = _detect_m3_block(order, contracts)
        assert result is None, "After cleanup, M3 must not block the next exit"
