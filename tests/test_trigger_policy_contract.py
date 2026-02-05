"""Tests for trigger-policy contract invariants.

MANDATORY INVARIANTS (from runbook 18):
- Policy NEVER creates exposure without trigger permission
- Policy NEVER overrides trigger direction
- If no trigger is active for symbol/direction, target_weight is exactly 0
- Replay determinism: same inputs produce same target_weight_final
"""

from datetime import datetime, timezone

import pytest

from agents.strategies.policy_engine import (
    PolicyEngine,
    PolicyState,
    apply_policy,
)
from schemas.policy import PolicyConfig, TriggerStateResult


# ============================================================================
# Invariant: No exposure without trigger permission
# ============================================================================


def test_inactive_always_zero():
    """Inactive trigger state MUST produce zero weight."""
    config = PolicyConfig()

    # Test with various signal strengths
    for signal in [0.0, 0.5, 1.0]:
        trigger = TriggerStateResult(
            symbol="BTC-USD",
            state="inactive",
            signal_strength=signal,
        )
        weight, record = apply_policy(
            trigger, vol_hat=0.15, config=config, state=PolicyState()
        )
        assert weight == 0.0, f"Inactive with signal={signal} produced non-zero weight"
        assert record.target_weight_final == 0.0


def test_inactive_with_high_vol():
    """Inactive remains zero regardless of volatility."""
    config = PolicyConfig()
    trigger = TriggerStateResult(symbol="BTC-USD", state="inactive")

    for vol in [0.01, 0.10, 0.50, 1.0]:
        weight, _ = apply_policy(
            trigger, vol_hat=vol, config=config, state=PolicyState()
        )
        assert weight == 0.0, f"Inactive with vol={vol} produced non-zero weight"


def test_inactive_ignores_config():
    """Inactive ignores all config parameters."""
    configs = [
        PolicyConfig(tau=0.0, w_max=1.0),
        PolicyConfig(tau=0.5, w_max=0.5),
        PolicyConfig(vol_target=1.0),
    ]
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="inactive", signal_strength=1.0
    )

    for config in configs:
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=PolicyState()
        )
        assert weight == 0.0


# ============================================================================
# Invariant: Policy respects trigger direction
# ============================================================================


def test_long_allowed_positive_only():
    """long_allowed MUST produce non-negative weight."""
    config = PolicyConfig(tau=0.0, w_min=0.01)

    for signal in [0.1, 0.5, 1.0]:
        trigger = TriggerStateResult(
            symbol="BTC-USD",
            state="long_allowed",
            signal_strength=signal,
        )
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=PolicyState()
        )
        assert weight >= 0.0, f"long_allowed with signal={signal} produced negative weight"


def test_short_allowed_negative_only():
    """short_allowed MUST produce non-positive weight."""
    config = PolicyConfig(tau=0.0, w_min=0.01)

    for signal in [0.1, 0.5, 1.0]:
        trigger = TriggerStateResult(
            symbol="BTC-USD",
            state="short_allowed",
            signal_strength=signal,
        )
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=PolicyState()
        )
        assert weight <= 0.0, f"short_allowed with signal={signal} produced positive weight"


def test_direction_never_flipped():
    """Direction is never flipped by policy math."""
    config = PolicyConfig()

    # Start with long position
    state = PolicyState()
    state.update_weight("BTC-USD", 0.10)

    # Even with exit_only, direction should not flip to negative
    trigger = TriggerStateResult(
        symbol="BTC-USD",
        state="exit_only",
        signal_strength=0.0,
        direction="long",
    )

    for _ in range(10):
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=state
        )
        assert weight >= 0.0, "Direction flipped during exit"
        state.update_weight("BTC-USD", weight)


# ============================================================================
# Invariant: Exit-only monotone convergence
# ============================================================================


def test_exit_only_monotone_positive():
    """exit_only with positive position converges toward zero monotonically."""
    config = PolicyConfig(alpha_by_horizon={"default": 0.5})
    state = PolicyState()
    state.update_weight("BTC-USD", 0.20)

    trigger = TriggerStateResult(symbol="BTC-USD", state="exit_only")
    prev_weight = 0.20

    for _ in range(10):
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=state
        )
        assert weight <= prev_weight, "Weight increased during exit_only"
        assert weight >= 0.0, "Weight became negative during long exit"
        prev_weight = weight
        state.update_weight("BTC-USD", weight)


def test_exit_only_monotone_negative():
    """exit_only with negative position converges toward zero monotonically."""
    config = PolicyConfig(alpha_by_horizon={"default": 0.5})
    state = PolicyState()
    state.update_weight("BTC-USD", -0.20)

    trigger = TriggerStateResult(symbol="BTC-USD", state="exit_only")
    prev_weight = -0.20

    for _ in range(10):
        weight, _ = apply_policy(
            trigger, vol_hat=0.15, config=config, state=state
        )
        assert weight >= prev_weight, "Weight decreased (more negative) during exit_only"
        assert weight <= 0.0, "Weight became positive during short exit"
        prev_weight = weight
        state.update_weight("BTC-USD", weight)


# ============================================================================
# Invariant: Replay determinism
# ============================================================================


def test_replay_determinism():
    """Same inputs MUST produce same outputs."""
    config = PolicyConfig(tau=0.1, vol_target=0.15, w_min=0.02, w_max=0.25)
    now = datetime.now(timezone.utc)

    trigger = TriggerStateResult(
        symbol="BTC-USD",
        state="long_allowed",
        signal_strength=0.75,
        active_trigger_ids=["trig_1"],
    )

    # Run twice with fresh state
    state1 = PolicyState()
    weight1, record1 = apply_policy(
        trigger, vol_hat=0.18, config=config, state=state1, timestamp=now
    )

    state2 = PolicyState()
    weight2, record2 = apply_policy(
        trigger, vol_hat=0.18, config=config, state=state2, timestamp=now
    )

    assert weight1 == weight2
    assert record1.target_weight_raw == record2.target_weight_raw
    assert record1.signal_deadbanded == record2.signal_deadbanded
    assert record1.vol_scale == record2.vol_scale


def test_replay_determinism_multi_bar():
    """Multi-bar sequence is replay deterministic."""
    config = PolicyConfig()
    now = datetime.now(timezone.utc)

    signals = [0.3, 0.5, 0.7, 0.4, 0.8]

    def run_sequence():
        engine = PolicyEngine(config=config, plan_id="test")
        weights = []
        for sig in signals:
            trigger = TriggerStateResult(
                symbol="BTC-USD", state="long_allowed", signal_strength=sig
            )
            w, _ = engine.on_bar(trigger, vol_hat=0.15, timestamp=now)
            weights.append(w)
        return weights

    weights1 = run_sequence()
    weights2 = run_sequence()

    assert weights1 == weights2


# ============================================================================
# Boundary conditions
# ============================================================================


def test_zero_signal_zero_weight():
    """Zero signal produces zero raw weight."""
    config = PolicyConfig(tau=0.0)  # No deadband
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.0
    )

    _, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=PolicyState()
    )

    assert record.target_weight_raw == 0.0


def test_max_signal_max_weight():
    """Max signal approaches max weight (subject to vol scaling)."""
    config = PolicyConfig(tau=0.0, w_max=0.25, vol_target=0.15)
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=1.0
    )

    # With vol_hat = vol_target, no vol scaling reduction
    _, record = apply_policy(
        trigger, vol_hat=0.15, config=config, state=PolicyState()
    )

    # Raw should be at w_max
    assert abs(record.target_weight_raw) == config.w_max


def test_extreme_volatility_caps_scale():
    """Extreme volatility caps vol_scale at 1.0."""
    config = PolicyConfig(vol_target=0.50)  # High vol target
    trigger = TriggerStateResult(
        symbol="BTC-USD", state="long_allowed", signal_strength=0.8
    )

    # Very low vol -> scale would be > 1 but should cap
    _, record = apply_policy(
        trigger, vol_hat=0.10, config=config, state=PolicyState()
    )

    assert record.vol_scale <= 1.0
