"""Tests for R96: TriggerRegistry, TriggerDiff, and archetype catalog."""

import pytest
from schemas.trigger_catalog import (
    TriggerInstance,
    TriggerDiff,
    TriggerArchetype,
    instance_to_trigger_condition,
    make_instance_id,
    ARCHETYPE_SPECS,
    ARCHETYPE_DIRECTIONS,
)
from services.trigger_registry import TriggerRegistry, PositionProtectedTriggerMutation


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _inst(archetype: str, symbol: str = "BTC-USD", session: str = "sess1234") -> TriggerInstance:
    return TriggerInstance(
        instance_id=make_instance_id(archetype, symbol, session),
        archetype_id=archetype,
        symbol=symbol,
        timeframe="5m",
        params={"stop_loss_pct": 1.5} if archetype not in ("emergency_exit", "profit_take_exit", "stop_loss_exit") else {},
        confidence_grade="B",
    )


def _registry(session: str = "sess1234") -> TriggerRegistry:
    return TriggerRegistry(session_id=session)


# ── Archetype catalog ────────────────────────────────────────────────────────

def test_all_archetypes_have_specs():
    archetypes = [
        "mean_reversion_long", "mean_reversion_short",
        "breakout_long", "breakout_short",
        "momentum_long", "momentum_short",
        "emergency_exit", "profit_take_exit", "stop_loss_exit",
    ]
    for a in archetypes:
        assert a in ARCHETYPE_SPECS, f"Missing spec for {a}"
        assert a in ARCHETYPE_DIRECTIONS, f"Missing direction for {a}"


def test_breakout_long_entry_rule():
    inst = _inst("breakout_long")
    tc = instance_to_trigger_condition(inst)
    assert "donchian_upper_short" in tc["entry_rule"]
    assert "breakout_confirmed > 0.0" in tc["entry_rule"]
    assert "volume_multiple" in tc["entry_rule"]
    assert tc["direction"] == "long"
    assert tc["category"] == "volatility_breakout"


def test_breakout_short_entry_rule():
    inst = _inst("breakout_short")
    tc = instance_to_trigger_condition(inst)
    assert "donchian_lower_short" in tc["entry_rule"]
    assert tc["direction"] == "short"


def test_mean_reversion_long_defaults():
    inst = _inst("mean_reversion_long")
    tc = instance_to_trigger_condition(inst)
    assert "rsi_14 < 30" in tc["entry_rule"]
    assert "bollinger_lower" in tc["entry_rule"]
    assert tc["target_anchor_type"] == "bollinger_middle"


def test_mean_reversion_long_custom_threshold():
    inst = TriggerInstance(
        instance_id="mean_reversion_long:BTC-USD:sess1234",
        archetype_id="mean_reversion_long",
        symbol="BTC-USD",
        timeframe="5m",
        params={"stop_loss_pct": 1.5, "rsi_oversold": 25},
    )
    tc = instance_to_trigger_condition(inst)
    assert "rsi_14 < 25" in tc["entry_rule"]


def test_emergency_exit_has_correct_stop():
    inst = _inst("emergency_exit")
    tc = instance_to_trigger_condition(inst)
    assert tc["stop_loss_pct"] == 0.0
    assert tc["stop_anchor_type"] is None
    assert tc["direction"] == "exit"
    assert tc["category"] == "emergency_exit"


def test_momentum_long_requires_entry_condition():
    inst = TriggerInstance(
        instance_id="momentum_long:ETH-USD:sess1234",
        archetype_id="momentum_long",
        symbol="ETH-USD",
        timeframe="5m",
        params={"entry_condition": "ema_fast > ema_medium and volume_multiple > 0.2", "stop_loss_pct": 1.5},
    )
    tc = instance_to_trigger_condition(inst)
    assert "ema_fast > ema_medium" in tc["entry_rule"]
    assert tc["direction"] == "long"


def test_instance_id_format():
    iid = make_instance_id("breakout_long", "BTC-USD", "sess1234abcd")
    assert iid == "breakout_long:BTC-USD:sess1234"


# ── TriggerRegistry core operations ─────────────────────────────────────────

def test_registry_empty_on_init():
    reg = _registry()
    assert reg.list_active() == []


def test_apply_diff_adds_triggers():
    reg = _registry()
    diff = TriggerDiff(to_add=[_inst("breakout_long"), _inst("emergency_exit")])
    summary = reg.apply_diff(diff)
    assert summary["added"] == 2
    assert summary["total_active"] == 2
    assert len(reg.list_active()) == 2


def test_apply_diff_removes_trigger():
    reg = _registry()
    inst = _inst("breakout_long")
    reg.apply_diff(TriggerDiff(to_add=[inst]))
    summary = reg.apply_diff(TriggerDiff(to_remove=[inst.instance_id]))
    assert summary["removed"] == 1
    assert len(reg.list_active()) == 0


def test_apply_diff_deduplicates_adds():
    reg = _registry()
    inst = _inst("breakout_long")
    reg.apply_diff(TriggerDiff(to_add=[inst]))
    summary = reg.apply_diff(TriggerDiff(to_add=[inst]))  # same instance_id
    assert summary["added"] == 0  # already exists
    assert len(reg.list_active()) == 1


def test_apply_diff_noop_on_empty():
    reg = _registry()
    diff = TriggerDiff()
    summary = reg.apply_diff(diff)
    assert summary["added"] == summary["removed"] == summary["modified"] == 0


# ── POSITION_OPEN guard ──────────────────────────────────────────────────────

def test_position_open_blocks_exit_removal():
    reg = _registry()
    em_inst = _inst("emergency_exit", symbol="BTC-USD")
    reg.apply_diff(TriggerDiff(to_add=[em_inst]))

    summary = reg.apply_diff(
        TriggerDiff(to_remove=[em_inst.instance_id]),
        policy_state="POSITION_OPEN",
        open_position_symbols=["BTC-USD"],
    )
    assert em_inst.instance_id in summary["blocked_mutations"]
    assert len(reg.list_active()) == 1  # still there


def test_position_open_allows_exit_removal_for_different_symbol():
    reg = _registry()
    em_btc = _inst("emergency_exit", symbol="BTC-USD")
    em_eth = _inst("emergency_exit", symbol="ETH-USD")
    reg.apply_diff(TriggerDiff(to_add=[em_btc, em_eth]))

    summary = reg.apply_diff(
        TriggerDiff(to_remove=[em_eth.instance_id]),
        policy_state="POSITION_OPEN",
        open_position_symbols=["BTC-USD"],  # only BTC is open
    )
    assert em_eth.instance_id not in summary["blocked_mutations"]
    assert len(reg.list_active()) == 1  # ETH exit removed, BTC exit kept


def test_position_open_allows_entry_removal():
    reg = _registry()
    entry = _inst("breakout_long", symbol="BTC-USD")
    reg.apply_diff(TriggerDiff(to_add=[entry]))

    summary = reg.apply_diff(
        TriggerDiff(to_remove=[entry.instance_id]),
        policy_state="POSITION_OPEN",
        open_position_symbols=["BTC-USD"],
    )
    assert entry.instance_id not in summary["blocked_mutations"]
    assert len(reg.list_active()) == 0


def test_idle_state_allows_exit_removal():
    reg = _registry()
    em_inst = _inst("emergency_exit", symbol="BTC-USD")
    reg.apply_diff(TriggerDiff(to_add=[em_inst]))

    summary = reg.apply_diff(
        TriggerDiff(to_remove=[em_inst.instance_id]),
        policy_state="IDLE",
    )
    assert em_inst.instance_id not in summary["blocked_mutations"]
    assert len(reg.list_active()) == 0


# ── to_trigger_conditions ────────────────────────────────────────────────────

def test_to_trigger_conditions_produces_valid_dicts():
    reg = _registry()
    reg.apply_diff(TriggerDiff(to_add=[
        _inst("breakout_long"),
        _inst("mean_reversion_short", symbol="ETH-USD"),
        _inst("emergency_exit"),
    ]))
    tcs = reg.to_trigger_conditions()
    assert len(tcs) == 3
    directions = {tc["direction"] for tc in tcs}
    assert "long" in directions
    assert "short" in directions
    assert "exit" in directions
    # All entry triggers have a stop
    for tc in tcs:
        if tc["direction"] in ("long", "short"):
            assert tc.get("stop_loss_pct") or tc.get("stop_anchor_type"), \
                f"Entry trigger {tc['id']} missing stop"


# ── Serialisation (CaN round-trip) ───────────────────────────────────────────

def test_registry_serialises_and_restores():
    reg = _registry()
    reg.apply_diff(TriggerDiff(to_add=[
        _inst("breakout_long"),
        _inst("emergency_exit"),
    ]))
    state = reg.to_state()
    restored = TriggerRegistry.from_state(state)
    assert len(restored.list_active()) == 2
    ids_orig = {i.instance_id for i in reg.list_active()}
    ids_restored = {i.instance_id for i in restored.list_active()}
    assert ids_orig == ids_restored


def test_registry_serialises_removed_state():
    reg = _registry()
    inst = _inst("breakout_long")
    reg.apply_diff(TriggerDiff(to_add=[inst]))
    reg.apply_diff(TriggerDiff(to_remove=[inst.instance_id]))
    state = reg.to_state()
    restored = TriggerRegistry.from_state(state)
    assert len(restored.list_active()) == 0
    assert len(restored.list_all()) == 1  # still in dict, just removed state


# ── to_context_block ─────────────────────────────────────────────────────────

def test_context_block_empty_registry():
    reg = _registry()
    block = reg.to_context_block()
    assert "(none" in block.lower()


def test_context_block_lists_active_instances():
    reg = _registry()
    reg.apply_diff(TriggerDiff(to_add=[_inst("breakout_long"), _inst("emergency_exit")]))
    block = reg.to_context_block()
    assert "breakout_long:BTC-USD" in block
    assert "emergency_exit:BTC-USD" in block
    assert "ACTIVE_TRIGGERS" in block
