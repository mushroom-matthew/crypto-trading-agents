"""Tests for the deterministic policy engine.

Tests cover:
- Signal transformation math (deadband, vol scaling, bounds)
- Smoothing behavior
- Trigger state classification
- PolicyEngine stateful behavior
"""

from datetime import datetime, timezone

import pytest

from agents.strategies.policy_engine import (
    PolicyEngine,
    PolicyState,
    RiskOverrides,
    apply_policy,
    classify_trigger_state,
)
from schemas.policy import PolicyConfig, TriggerStateResult


# ============================================================================
# Trigger State Classification Tests
# ============================================================================


def test_classify_inactive_no_position():
    """No entry, no position -> inactive."""
    result = classify_trigger_state(
        symbol="BTC-USD",
        entry_fired=False,
        exit_fired=False,
        entry_direction=None,
        has_position=False,
        position_direction=None,
    )
    assert result.state == "inactive"
    assert result.direction is None


def test_classify_long_allowed():
    """Entry fired with long direction -> long_allowed."""
    result = classify_trigger_state(
        symbol="BTC-USD",
        entry_fired=True,
        exit_fired=False,
        entry_direction="long",
        has_position=False,
        position_direction=None,
        signal_strength=0.75,
    )
    assert result.state == "long_allowed"
    assert result.direction == "long"
    assert result.signal_strength == 0.75


def test_classify_short_allowed():
    """Entry fired with short direction -> short_allowed."""
    result = classify_trigger_state(
        symbol="BTC-USD",
        entry_fired=True,
        exit_fired=False,
        entry_direction="short",
        has_position=False,
        position_direction=None,
    )
    assert result.state == "short_allowed"
    assert result.direction == "short"


def test_classify_exit_only():
    """No entry but has position -> exit_only."""
    result = classify_trigger_state(
        symbol="BTC-USD",
        entry_fired=False,
        exit_fired=True,
        entry_direction=None,
        has_position=True,
        position_direction="long",
    )
    assert result.state == "exit_only"
    assert result.direction == "long"


def test_classify_preserves_trigger_ids():
    """Active trigger IDs are preserved."""
    result = classify_trigger_state(
        symbol="BTC-USD",
        entry_fired=True,
        exit_fired=False,
        entry_direction="long",
        has_position=False,
        position_direction=None,
        active_trigger_ids=["trig_1", "trig_2"],
        confidence_grade="A",
    )
    assert result.active_trigger_ids == ["trig_1", "trig_2"]
    assert result.highest_confidence == "A"


# ============================================================================
# Core Policy Math Tests
# ============================================================================


def test_inactive_produces_zero():
    """Inactive trigger state must produce zero target weight."""
    config = PolicyConfig()
    state = PolicyState()
    trigger = TriggerStateResult(symbol="BTC-USD", state="inactive")

    weight, record = apply_policy(
        trigger, vol_hat=0.20, config=config, state=state
    )

    assert weight == 0.0
    assert record.target_weight_final == 0.0
    assert record.trigger_state == "inactive"


def test_long_allowed_positive_weight():
    """Long allowed produces positive weight."""
    config = PolicyConfig(tau=0.1, w_min=0.02, w_max=0.25)
    state = PolicyState()
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.8
    )

    weight, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=state
    )

    assert weight > 0
    assert record.target_weight_final > 0
    assert record.trigger_state == "long_allowed"


def test_short_allowed_negative_weight():
    """Short allowed produces negative weight."""
    config = PolicyConfig(tau=0.1, w_min=0.02, w_max=0.25)
    state = PolicyState()
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="short_allowed", signal_strength=0.8
    )

    weight, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=state
    )

    assert weight < 0
    assert record.target_weight_final < 0


def test_deadband_filters_weak_signals():
    """Signals below tau produce zero after deadband."""
    config = PolicyConfig(tau=0.3)  # High deadband
    state = PolicyState()
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.2
    )

    weight, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=state
    )

    assert record.signal_deadbanded == 0.0
    # Weight might still be non-zero due to smoothing from previous state
    # but raw should be zero
    assert record.target_weight_raw == 0.0


def test_vol_scaling_reduces_weight():
    """High volatility reduces weight via vol_scale."""
    # Use high w_max to avoid capping effects masking vol scaling
    config = PolicyConfig(vol_target=0.15, w_max=1.0, tau=0.0)
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.5
    )

    # Low vol -> high scale (capped at 1.0)
    weight_low_vol, record_low = apply_policy(
        trigger, vol_hat=0.10, config=config, state=PolicyState()
    )

    # High vol -> low scale
    weight_high_vol, record_high = apply_policy(
        trigger, vol_hat=0.30, config=config, state=PolicyState()
    )

    assert record_low.vol_scale > record_high.vol_scale
    # Raw weights should differ due to vol scaling
    assert record_low.target_weight_raw > record_high.target_weight_raw


def test_weight_bounded_by_w_max():
    """Weight is capped at w_max."""
    config = PolicyConfig(w_max=0.10, tau=0.0, vol_target=1.0)
    state = PolicyState()
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=1.0
    )

    weight, record = apply_policy(
        trigger, vol_hat=0.05, config=config, state=state
    )

    # Raw might be high but bounded should respect w_max
    assert abs(record.target_weight_raw) <= config.w_max + 0.001


def test_weight_minimum_w_min():
    """Non-zero weight is at least w_min."""
    config = PolicyConfig(w_min=0.05, tau=0.0)
    state = PolicyState()
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.01
    )

    weight, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=state
    )

    # If raw is non-zero, bounded should be at least w_min
    if record.target_weight_raw != 0:
        assert abs(record.target_weight_raw) >= config.w_min


# ============================================================================
# Smoothing Tests
# ============================================================================


def test_smoothing_gradual_approach():
    """Smoothing causes gradual approach to target."""
    config = PolicyConfig(tau=0.0, alpha_by_horizon={"default": 0.3})
    state = PolicyState()

    weights = []
    for _ in range(5):
        trigger = TriggerStateResult(
            symbol="BTC-USD", state="long_allowed", signal_strength=0.8
        )
        weight, record = apply_policy(
            trigger, vol_hat=0.15, config=config, state=state
        )
        state.update_weight("BTC-USD", weight)
        weights.append(weight)

    # Each weight should be larger than previous (approaching target)
    for i in range(1, len(weights)):
        assert weights[i] >= weights[i - 1]


def test_exit_only_monotone_decay():
    """Exit only causes monotone decay toward zero."""
    config = PolicyConfig(alpha_by_horizon={"default": 0.3})
    state = PolicyState()
    state.update_weight("BTC-USD", 0.20)  # Start with position

    weights = [0.20]
    for _ in range(5):
        trigger = TriggerStateResult(
            symbol="BTC-USD", state="exit_only", signal_strength=0.0
        )
        weight, record = apply_policy(
            trigger, vol_hat=0.15, config=config, state=state
        )
        state.update_weight("BTC-USD", weight)
        weights.append(weight)

    # Each weight should be smaller (decaying toward zero)
    for i in range(1, len(weights)):
        assert weights[i] <= weights[i - 1]
    # Should converge toward zero
    assert weights[-1] < 0.10


# ============================================================================
# PolicyEngine Stateful Tests
# ============================================================================


def test_engine_tracks_state():
    """PolicyEngine maintains state across bars."""
    config = PolicyConfig()
    engine = PolicyEngine(config=config, plan_id="test_plan")
    now = datetime.now(timezone.utc)

    # Process multiple bars
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.7
    )
    weight1, _ = engine.on_bar(trigger, vol_hat=0.18, timestamp=now)
    weight2, _ = engine.on_bar(trigger, vol_hat=0.18, timestamp=now)

    # State should accumulate
    assert weight2 != weight1 or engine.state.bar_counter == 2


def test_engine_decision_history():
    """PolicyEngine records decision history."""
    config = PolicyConfig()
    engine = PolicyEngine(config=config, plan_id="test_plan")
    now = datetime.now(timezone.utc)

    for _ in range(3):
        trigger = TriggerStateResult(
            symbol="BTC-USD", state="long_allowed", signal_strength=0.7
        )
        engine.on_bar(trigger, vol_hat=0.18, timestamp=now)

    assert len(engine.decision_history) == 3
    summary = engine.get_decision_summary()
    assert summary["total_decisions"] == 3


def test_engine_force_flatten():
    """force_flatten immediately zeros position."""
    config = PolicyConfig()
    engine = PolicyEngine(config=config, plan_id="test_plan")
    engine.state.update_weight("BTC-USD", 0.15)
    now = datetime.now(timezone.utc)

    record = engine.force_flatten("BTC-USD", "emergency test", now)

    assert record.target_weight_final == 0.0
    assert record.override_applied == "emergency_exit"
    assert record.precedence_tier == 1
    assert engine.state.get_current_weight("BTC-USD") == 0.0


def test_engine_rebalance_cooldown():
    """Rebalance cooldown prevents rapid trading."""
    config = PolicyConfig(rebalance_cooldown_bars=3, epsilon_w=0.001)
    engine = PolicyEngine(config=config, plan_id="test_plan")
    now = datetime.now(timezone.utc)

    rebalances = []
    for i in range(5):
        trigger = TriggerStateResult(
            symbol="BTC-USD",
            state="long_allowed",
            signal_strength=0.5 + i * 0.1,  # Varying signal
        )
        _, record = engine.on_bar(trigger, vol_hat=0.18, timestamp=now)
        rebalances.append(record.should_rebalance)

    # First bar should rebalance, then cooldown applies
    assert rebalances[0] is True
    # Some subsequent bars blocked by cooldown
    blocked_count = sum(1 for r in rebalances[1:4] if not r)
    assert blocked_count >= 1


def test_engine_symbol_allowlist():
    """Symbols not in allowlist are blocked."""
    config = PolicyConfig(symbol_allowlist=["BTC-USD"])
    engine = PolicyEngine(config=config, plan_id="test_plan")
    now = datetime.now(timezone.utc)

    # BTC allowed
    trigger_btc = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.8
    )
    _, record_btc = engine.on_bar(trigger_btc, vol_hat=0.18, timestamp=now)

    # ETH not allowed
    trigger_eth = TriggerStateResult(
        symbol="ETH-USD", state="long_allowed", signal_strength=0.8
    )
    _, record_eth = engine.on_bar(trigger_eth, vol_hat=0.18, timestamp=now)

    assert record_btc.rebalance_blocked_reason is None or "not_allowed" not in (record_btc.rebalance_blocked_reason or "")
    assert record_eth.rebalance_blocked_reason == "symbol_not_allowed"
