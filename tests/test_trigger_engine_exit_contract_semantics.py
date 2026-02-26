"""Tests for Runbook 60 Phase M1: entry-rule flatten path detection.

Verifies that:
- Orders with reason ending in '_flat' and non-emergency category are detected
  as entry-rule flatten paths (the M1 guardrail event is produced)
- Emergency exits ending in '_flat' are NOT flagged (they bypass normal checks)
- Orders ending in '_exit' (conflict-exit path) are NOT flagged
- Entry orders are NOT flagged
- The event payload contains the required audit fields

These tests exercise the M1 guardrail logic directly using the same detection
criteria used in paper_trading.py's order processing loop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# M1 guardrail: detection function (mirrors the inline logic in paper_trading.py)
# ---------------------------------------------------------------------------


def _detect_entry_rule_flatten(order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mirrors the M1 guardrail inline check in paper_trading.py.

    Returns an event payload dict if the order is a non-emergency entry-rule
    flatten, or None otherwise.
    """
    intent = order.get("intent")
    reason = order.get("reason", "") or ""
    category = order.get("trigger_category", "")

    if (
        intent == "exit"
        and isinstance(reason, str)
        and reason.endswith("_flat")
        and category != "emergency_exit"
    ):
        return {
            "type": "entry_rule_flatten_detected",
            "payload": {
                "symbol": order.get("symbol", ""),
                "trigger_id": reason,
                "category": category,
                "exit_class": "strategy_contract_candidate",
                "detail": (
                    "Non-emergency exit trigger fired via entry-rule flatten path "
                    "(reason ends '_flat'). Migrate to PositionExitContract leg."
                ),
            },
        }
    return None


# ---------------------------------------------------------------------------
# Positive cases: should be detected as entry-rule flatten
# ---------------------------------------------------------------------------


def test_detect_exit_with_flat_suffix():
    order = {
        "symbol": "BNB-USD",
        "intent": "exit",
        "reason": "bnb_risk_off_flat",
        "trigger_category": "risk_off",
        "side": "sell",
        "quantity": 1.0,
        "price": 300.0,
    }
    event = _detect_entry_rule_flatten(order)
    assert event is not None
    assert event["type"] == "entry_rule_flatten_detected"


def test_detect_exit_flat_no_category():
    order = {
        "symbol": "ETH-USD",
        "intent": "exit",
        "reason": "eth_exit_flat",
        "trigger_category": None,
        "side": "sell",
        "quantity": 0.5,
        "price": 3000.0,
    }
    event = _detect_entry_rule_flatten(order)
    assert event is not None


def test_detect_exit_flat_unknown_category():
    order = {
        "symbol": "BTC-USD",
        "intent": "exit",
        "reason": "btc_defensive_flat",
        "trigger_category": "risk_reduce",
        "side": "sell",
        "quantity": 0.01,
        "price": 50000.0,
    }
    event = _detect_entry_rule_flatten(order)
    assert event is not None
    assert event["payload"]["exit_class"] == "strategy_contract_candidate"


def test_event_payload_has_required_audit_fields():
    order = {
        "symbol": "SOL-USD",
        "intent": "exit",
        "reason": "sol_risk_flat",
        "trigger_category": "defensive",
    }
    event = _detect_entry_rule_flatten(order)
    payload = event["payload"]
    assert "symbol" in payload
    assert "trigger_id" in payload
    assert "category" in payload
    assert "exit_class" in payload
    assert "detail" in payload


# ---------------------------------------------------------------------------
# Negative cases: should NOT be detected
# ---------------------------------------------------------------------------


def test_emergency_exit_flat_not_detected():
    """Emergency exits may use the _flat path — they are exempt."""
    order = {
        "symbol": "BTC-USD",
        "intent": "exit",
        "reason": "btc_emergency_exit_a1_flat",
        "trigger_category": "emergency_exit",
        "side": "sell",
        "quantity": 0.1,
        "price": 50000.0,
    }
    event = _detect_entry_rule_flatten(order)
    assert event is None


def test_conflict_exit_suffix_not_detected():
    """Orders with '_exit' suffix (conflicting signal path) are not flagged."""
    order = {
        "symbol": "ETH-USD",
        "intent": "exit",
        "reason": "eth_breakout_exit",
        "trigger_category": "trend_following",
    }
    event = _detect_entry_rule_flatten(order)
    assert event is None


def test_entry_order_not_detected():
    order = {
        "symbol": "BTC-USD",
        "intent": "entry",
        "reason": "btc_breakout_flat",  # even if suffix matches, intent=entry
        "trigger_category": "trend_following",
    }
    event = _detect_entry_rule_flatten(order)
    assert event is None


def test_stop_sweep_exit_not_detected():
    """Stop-sweep orders (stop_hit) use _hit suffix, not _flat."""
    order = {
        "symbol": "BTC-USD",
        "intent": "exit",
        "reason": "btc_usd_stop_hit",
        "trigger_category": "stop_loss",
    }
    event = _detect_entry_rule_flatten(order)
    assert event is None


def test_take_profit_exit_not_detected():
    order = {
        "symbol": "ETH-USD",
        "intent": "exit",
        "reason": "eth_usd_target_hit",
        "trigger_category": "take_profit",
    }
    event = _detect_entry_rule_flatten(order)
    assert event is None


# ---------------------------------------------------------------------------
# Batch processing: simulate the full order loop
# ---------------------------------------------------------------------------


def _process_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate the order processing loop in paper_trading.py."""
    events = []
    for order in orders:
        guardrail_event = _detect_entry_rule_flatten(order)
        if guardrail_event:
            events.append(guardrail_event)
        events.append({"type": "trigger_fired", "payload": {"symbol": order.get("symbol")}})
    return events


def test_only_non_emergency_flats_produce_guardrail_events():
    orders = [
        # Entry: no guardrail
        {"symbol": "BTC-USD", "intent": "entry", "reason": "btc_long_a1", "trigger_category": "trend"},
        # Normal exit via _flat: guardrail fires
        {"symbol": "BNB-USD", "intent": "exit", "reason": "bnb_risk_off_flat", "trigger_category": "risk_off"},
        # Emergency exit via _flat: no guardrail
        {"symbol": "ETH-USD", "intent": "exit", "reason": "eth_emergency_flat", "trigger_category": "emergency_exit"},
        # _exit suffix: no guardrail
        {"symbol": "SOL-USD", "intent": "exit", "reason": "sol_conflict_exit", "trigger_category": "conflict"},
    ]
    events = _process_orders(orders)
    guardrail_events = [e for e in events if e["type"] == "entry_rule_flatten_detected"]
    assert len(guardrail_events) == 1
    assert guardrail_events[0]["payload"]["symbol"] == "BNB-USD"


def test_multiple_flat_exits_produce_multiple_guardrail_events():
    orders = [
        {"symbol": "BTC-USD", "intent": "exit", "reason": "btc_flat", "trigger_category": "risk_off"},
        {"symbol": "ETH-USD", "intent": "exit", "reason": "eth_flat", "trigger_category": "risk_reduce"},
    ]
    events = _process_orders(orders)
    guardrail_events = [e for e in events if e["type"] == "entry_rule_flatten_detected"]
    assert len(guardrail_events) == 2


# ---------------------------------------------------------------------------
# Exit class tagging in telemetry
# ---------------------------------------------------------------------------


def test_exit_class_in_order_event():
    """
    When a non-emergency _flat order fires, the guardrail event should use
    exit_class='strategy_contract_candidate' (not 'emergency' or 'portfolio_overlay').
    """
    order = {
        "symbol": "AVAX-USD",
        "intent": "exit",
        "reason": "avax_exit_flat",
        "trigger_category": "defensive",
    }
    event = _detect_entry_rule_flatten(order)
    assert event["payload"]["exit_class"] == "strategy_contract_candidate"


def test_emergency_exit_would_use_emergency_class():
    """
    Confirm that emergency exits are excluded from the guardrail (they are a
    separate exit class tracked elsewhere).
    """
    order = {
        "symbol": "BTC-USD",
        "intent": "exit",
        "reason": "btc_emergency_a1_flat",
        "trigger_category": "emergency_exit",
    }
    event = _detect_entry_rule_flatten(order)
    # No guardrail event — emergency exits are managed separately
    assert event is None
