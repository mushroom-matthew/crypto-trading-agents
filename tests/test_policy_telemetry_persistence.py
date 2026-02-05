"""Tests for policy decision telemetry persistence.

Verifies that PolicyDecisionRecord serializes correctly for storage
and can be reconstructed for replay/audit.
"""

import json
from datetime import datetime, timezone

import pytest

from schemas.policy import PolicyDecisionRecord


def test_decision_record_to_telemetry_dict():
    """Decision record serializes to dict with all fields."""
    now = datetime.now(timezone.utc)
    record = PolicyDecisionRecord(
        timestamp=now,
        symbol="BTC-USD",
        plan_id="plan_123",
        trade_set_id="ts_456",
        trigger_state="long_allowed",
        active_trigger_ids=["trig_1", "trig_2"],
        signal_strength=0.75,
        signal_deadbanded=0.65,
        vol_hat=0.18,
        vol_scale=0.83,
        current_weight=0.05,
        target_weight_raw=0.20,
        target_weight_policy=0.15,
        target_weight_capped=0.15,
        target_weight_final=0.15,
        delta_weight=0.10,
        override_applied="risk_cap",
        override_reason="symbol_cap=0.15",
        precedence_tier=3,
        policy_config_hash="abc123",
        should_rebalance=True,
        rebalance_blocked_reason=None,
    )

    telemetry = record.to_telemetry_dict()

    assert telemetry["symbol"] == "BTC-USD"
    assert telemetry["plan_id"] == "plan_123"
    assert telemetry["trigger_state"] == "long_allowed"
    assert telemetry["signal_strength"] == 0.75
    assert telemetry["target_weight_final"] == 0.15
    assert telemetry["override_applied"] == "risk_cap"
    assert telemetry["precedence_tier"] == 3


def test_decision_record_json_serializable():
    """Decision record telemetry dict is JSON serializable."""
    now = datetime.now(timezone.utc)
    record = PolicyDecisionRecord(
        timestamp=now,
        symbol="BTC-USD",
        plan_id="plan_123",
        trigger_state="inactive",
        signal_strength=0.0,
        signal_deadbanded=0.0,
        vol_hat=0.0,
        vol_scale=0.0,
        current_weight=0.0,
        target_weight_raw=0.0,
        target_weight_policy=0.0,
        target_weight_capped=0.0,
        target_weight_final=0.0,
        delta_weight=0.0,
    )

    telemetry = record.to_telemetry_dict()

    # Should not raise
    json_str = json.dumps(telemetry)
    assert "BTC-USD" in json_str

    # Should round-trip
    parsed = json.loads(json_str)
    assert parsed["symbol"] == "BTC-USD"


def test_decision_record_from_json():
    """Decision record can be reconstructed from JSON."""
    now = datetime.now(timezone.utc)
    original = PolicyDecisionRecord(
        timestamp=now,
        symbol="ETH-USD",
        plan_id="plan_789",
        trigger_state="short_allowed",
        signal_strength=0.85,
        signal_deadbanded=0.75,
        vol_hat=0.22,
        vol_scale=0.68,
        current_weight=-0.10,
        target_weight_raw=-0.25,
        target_weight_policy=-0.18,
        target_weight_capped=-0.18,
        target_weight_final=-0.18,
        delta_weight=-0.08,
    )

    # Serialize to JSON
    json_str = json.dumps(original.to_telemetry_dict())

    # Reconstruct
    data = json.loads(json_str)
    reconstructed = PolicyDecisionRecord(**data)

    assert reconstructed.symbol == original.symbol
    assert reconstructed.trigger_state == original.trigger_state
    assert reconstructed.target_weight_final == original.target_weight_final


def test_decision_record_all_trigger_states():
    """All trigger states serialize correctly."""
    now = datetime.now(timezone.utc)

    for state in ["inactive", "long_allowed", "short_allowed", "exit_only"]:
        record = PolicyDecisionRecord(
            timestamp=now,
            symbol="BTC-USD",
            plan_id="test",
            trigger_state=state,
            signal_strength=0.0,
            signal_deadbanded=0.0,
            vol_hat=0.0,
            vol_scale=0.0,
            current_weight=0.0,
            target_weight_raw=0.0,
            target_weight_policy=0.0,
            target_weight_capped=0.0,
            target_weight_final=0.0,
            delta_weight=0.0,
        )

        telemetry = record.to_telemetry_dict()
        assert telemetry["trigger_state"] == state


def test_decision_record_nullable_fields():
    """Optional fields serialize as null."""
    now = datetime.now(timezone.utc)
    record = PolicyDecisionRecord(
        timestamp=now,
        symbol="BTC-USD",
        plan_id="test",
        trigger_state="inactive",
        signal_strength=0.0,
        signal_deadbanded=0.0,
        vol_hat=0.0,
        vol_scale=0.0,
        current_weight=0.0,
        target_weight_raw=0.0,
        target_weight_policy=0.0,
        target_weight_capped=0.0,
        target_weight_final=0.0,
        delta_weight=0.0,
        # These are optional and None
        trade_set_id=None,
        override_applied=None,
        override_reason=None,
        policy_config_hash=None,
        rebalance_blocked_reason=None,
    )

    telemetry = record.to_telemetry_dict()

    assert telemetry["trade_set_id"] is None
    assert telemetry["override_applied"] is None
    assert telemetry["override_reason"] is None


def test_decision_record_integration_format():
    """Decision records match expected integration format."""
    # This test documents the expected telemetry format for
    # integration with the backtest runner and ops API endpoints.

    now = datetime.now(timezone.utc)
    record = PolicyDecisionRecord(
        timestamp=now,
        symbol="BTC-USD",
        plan_id="plan_abc",
        trade_set_id="ts_xyz",
        trigger_state="long_allowed",
        active_trigger_ids=["trigger_1"],
        signal_strength=0.7,
        signal_deadbanded=0.6,
        vol_hat=0.18,
        vol_scale=0.83,
        current_weight=0.0,
        target_weight_raw=0.15,
        target_weight_policy=0.12,
        target_weight_capped=0.12,
        target_weight_final=0.12,
        delta_weight=0.12,
        precedence_tier=4,
        policy_config_hash="hash123",
        should_rebalance=True,
    )

    telemetry = record.to_telemetry_dict()

    # Required identification fields
    assert "timestamp" in telemetry
    assert "symbol" in telemetry
    assert "plan_id" in telemetry

    # Trigger state fields
    assert "trigger_state" in telemetry
    assert "active_trigger_ids" in telemetry

    # Signal path fields
    assert "signal_strength" in telemetry
    assert "signal_deadbanded" in telemetry
    assert "vol_hat" in telemetry
    assert "vol_scale" in telemetry

    # Weight progression fields (required for attribution)
    assert "current_weight" in telemetry
    assert "target_weight_raw" in telemetry
    assert "target_weight_policy" in telemetry
    assert "target_weight_capped" in telemetry
    assert "target_weight_final" in telemetry
    assert "delta_weight" in telemetry

    # Override tracking
    assert "override_applied" in telemetry
    assert "override_reason" in telemetry
    assert "precedence_tier" in telemetry

    # Execution decision
    assert "should_rebalance" in telemetry
    assert "rebalance_blocked_reason" in telemetry
