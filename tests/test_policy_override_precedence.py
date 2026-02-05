"""Tests for policy override precedence.

Override precedence (from runbook 18):
emergency_exit > stand_down > risk_caps > policy_output

Tier 1: emergency_exit - bypass policy, immediate flatten
Tier 2: stand_down - force neutral while active
Tier 3: risk_caps - clamp policy output
Tier 4: policy_output - deterministic policy result
"""

from datetime import datetime, timedelta, timezone

import pytest

from agents.strategies.policy_engine import (
    PolicyState,
    RiskOverrides,
    apply_policy,
)
from schemas.policy import PolicyConfig, StandDownState, TriggerStateResult


def _make_trigger(signal: float = 0.8) -> TriggerStateResult:
    """Helper to create long_allowed trigger."""
    return TriggerStateResult(
        symbol="BTC-USD",
        state="long_allowed",
        signal_strength=signal,
    )


# ============================================================================
# Tier 1: Emergency Exit (highest precedence)
# ============================================================================


def test_emergency_exit_overrides_all():
    """Emergency exit overrides everything."""
    config = PolicyConfig()
    state = PolicyState()
    risk = RiskOverrides(
        emergency_exit_active=True,
        emergency_exit_reason="test emergency",
    )

    weight, record = apply_policy(
        _make_trigger(), vol_hat=0.15, config=config, state=state, risk_overrides=risk
    )

    assert weight == 0.0
    assert record.target_weight_final == 0.0
    assert record.override_applied == "emergency_exit"
    assert record.precedence_tier == 1


def test_emergency_exit_overrides_stand_down():
    """Emergency exit takes precedence over stand_down."""
    now = datetime.now(timezone.utc)
    config = PolicyConfig(
        stand_down_state=StandDownState(
            stand_down_until_ts=now + timedelta(hours=1),
            stand_down_reason="drawdown",
            stand_down_source="risk_engine",
        )
    )
    risk = RiskOverrides(
        emergency_exit_active=True,
        emergency_exit_reason="emergency",
    )

    weight, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
        timestamp=now,
    )

    # Emergency takes precedence
    assert record.override_applied == "emergency_exit"
    assert record.precedence_tier == 1


def test_emergency_exit_overrides_risk_caps():
    """Emergency exit takes precedence over risk caps."""
    config = PolicyConfig()
    risk = RiskOverrides(
        max_weight=0.10,  # Low cap
        emergency_exit_active=True,
        emergency_exit_reason="test",
    )

    weight, record = apply_policy(
        _make_trigger(signal=1.0),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
    )

    assert weight == 0.0
    assert record.precedence_tier == 1


# ============================================================================
# Tier 2: Stand Down
# ============================================================================


def test_stand_down_forces_neutral():
    """Stand down forces target_weight to zero."""
    now = datetime.now(timezone.utc)
    config = PolicyConfig(
        stand_down_state=StandDownState(
            stand_down_until_ts=now + timedelta(hours=1),
            stand_down_reason="daily loss limit",
            stand_down_source="risk_engine",
        )
    )

    weight, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        timestamp=now,
    )

    assert weight == 0.0
    assert record.override_applied == "stand_down"
    assert record.override_reason == "daily loss limit"
    assert record.precedence_tier == 2


def test_stand_down_expires():
    """Stand down does not apply after expiry."""
    now = datetime.now(timezone.utc)
    config = PolicyConfig(
        stand_down_state=StandDownState(
            stand_down_until_ts=now - timedelta(hours=1),  # Expired
            stand_down_reason="expired",
            stand_down_source="ops",
        )
    )

    weight, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        timestamp=now,
    )

    # Should not be stand_down override
    assert record.override_applied != "stand_down"
    assert weight > 0  # Normal policy output


def test_stand_down_overrides_risk_caps():
    """Stand down takes precedence over risk caps."""
    now = datetime.now(timezone.utc)
    config = PolicyConfig(
        stand_down_state=StandDownState(
            stand_down_until_ts=now + timedelta(hours=1),
            stand_down_reason="stand_down_test",
            stand_down_source="judge",
        )
    )
    risk = RiskOverrides(max_weight=0.05)

    weight, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
        timestamp=now,
    )

    assert weight == 0.0
    assert record.precedence_tier == 2  # Stand down, not risk cap


# ============================================================================
# Tier 3: Risk Caps
# ============================================================================


def test_risk_cap_global():
    """Global max_weight cap is applied."""
    config = PolicyConfig(w_max=0.50)  # High config max
    risk = RiskOverrides(max_weight=0.05)  # Low risk cap

    weight, record = apply_policy(
        _make_trigger(signal=1.0),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
    )

    assert abs(weight) <= 0.05
    assert record.override_applied == "risk_cap"
    assert record.precedence_tier == 3


def test_risk_cap_symbol_specific():
    """Symbol-specific cap is applied."""
    config = PolicyConfig(w_max=0.50)
    risk = RiskOverrides(
        max_weight=1.0,  # High global
        max_symbol_weight={"BTC-USD": 0.03},  # Low symbol cap
    )

    weight, record = apply_policy(
        _make_trigger(signal=1.0),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
    )

    assert abs(weight) <= 0.03
    assert record.override_applied == "risk_cap"


def test_risk_cap_tighter_of_two():
    """Tighter of global and symbol cap is applied."""
    config = PolicyConfig(w_max=0.50)

    # Symbol cap is tighter
    risk1 = RiskOverrides(max_weight=0.10, max_symbol_weight={"BTC-USD": 0.05})
    _, record1 = apply_policy(
        _make_trigger(signal=1.0),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk1,
    )
    assert abs(record1.target_weight_capped) <= 0.05

    # Global cap is tighter
    risk2 = RiskOverrides(max_weight=0.03, max_symbol_weight={"BTC-USD": 0.10})
    _, record2 = apply_policy(
        _make_trigger(signal=1.0),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk2,
    )
    assert abs(record2.target_weight_capped) <= 0.03


# ============================================================================
# Tier 4: Policy Output (lowest precedence)
# ============================================================================


def test_policy_output_no_overrides():
    """Without overrides, policy output is final."""
    config = PolicyConfig(tau=0.0, w_max=0.25, vol_target=0.15)

    weight, record = apply_policy(
        _make_trigger(signal=0.6),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
    )

    assert record.override_applied is None
    assert record.precedence_tier == 4
    assert weight > 0


def test_policy_output_equals_capped_without_overrides():
    """Without Tier 1-2 overrides, final equals capped."""
    config = PolicyConfig(w_max=0.25)
    risk = RiskOverrides(max_weight=0.10)  # Only Tier 3

    _, record = apply_policy(
        _make_trigger(signal=0.8),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
        risk_overrides=risk,
    )

    # Tier 3 applied but no Tier 1-2
    assert record.target_weight_final == record.target_weight_capped


# ============================================================================
# Override reason tracking
# ============================================================================


def test_override_reason_captured():
    """Override reasons are captured in decision record."""
    now = datetime.now(timezone.utc)

    # Emergency
    risk_emergency = RiskOverrides(
        emergency_exit_active=True,
        emergency_exit_reason="margin call",
    )
    _, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=PolicyConfig(),
        state=PolicyState(),
        risk_overrides=risk_emergency,
    )
    assert record.override_reason == "margin call"

    # Stand down
    config_standdown = PolicyConfig(
        stand_down_state=StandDownState(
            stand_down_until_ts=now + timedelta(hours=1),
            stand_down_reason="volatility spike",
            stand_down_source="risk_engine",
        )
    )
    _, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config_standdown,
        state=PolicyState(),
        timestamp=now,
    )
    assert record.override_reason == "volatility spike"


def test_no_override_no_reason():
    """Without override, reason is None."""
    config = PolicyConfig()

    _, record = apply_policy(
        _make_trigger(),
        vol_hat=0.15,
        config=config,
        state=PolicyState(),
    )

    assert record.override_applied is None
    assert record.override_reason is None


# ============================================================================
# Decision record completeness
# ============================================================================


def test_decision_record_all_fields():
    """Decision record contains all required fields."""
    now = datetime.now(timezone.utc)
    config = PolicyConfig()

    _, record = apply_policy(
        _make_trigger(signal=0.7),
        vol_hat=0.18,
        config=config,
        state=PolicyState(),
        timestamp=now,
        plan_id="plan_123",
        trade_set_id="ts_456",
    )

    # Required identification
    assert record.timestamp == now
    assert record.symbol == "BTC-USD"
    assert record.plan_id == "plan_123"
    assert record.trade_set_id == "ts_456"

    # Signal path
    assert record.signal_strength == 0.7
    assert record.signal_deadbanded >= 0
    assert record.vol_hat == 0.18
    assert 0 <= record.vol_scale <= 1

    # Weight progression
    assert record.current_weight is not None
    assert record.target_weight_raw is not None
    assert record.target_weight_policy is not None
    assert record.target_weight_capped is not None
    assert record.target_weight_final is not None
    assert record.delta_weight is not None

    # Metadata
    assert record.precedence_tier in [1, 2, 3, 4]
    assert record.policy_config_hash is not None
