"""Runbook 35 — Block reason normalization tests.

Verifies that block reasons like exit_binding_mismatch and HOLD_RULE
preserve their identity through _normalized_reason() instead of being
mis-classified as BlockReason.RISK.
"""
from __future__ import annotations

from trading_core.execution_engine import BlockReason


def _build_normalized_reason():
    """Replicate the _normalized_reason closure from LLMStrategistRunner.

    This mirrors the logic in llm_strategist_runner.py lines ~1667-1757
    so we can unit-test it without instantiating the full runner.
    """
    reason_map = set(item.value for item in BlockReason)
    custom_reasons = {
        "timeframe_cap",
        "session_cap",
        "trigger_load",
        "min_hold",
        "min_flat",
        "emergency_exit_veto_same_bar",
        "emergency_exit_veto_min_hold",
        "emergency_exit_missing_exit_rule",
        "emergency_exit_executed",
        # Runbook 35: previously mis-classified as "risk"
        "exit_binding_mismatch",
        "HOLD_RULE",
        "archetype_load",
        "priority_skip",
        "learning_gate_closed",
    }

    def _normalized_reason(raw: str | None) -> str:
        if not raw:
            return BlockReason.OTHER.value
        if raw in reason_map or raw in custom_reasons:
            return raw
        # Unknown reasons are typically risk parameter names from the engine
        return BlockReason.RISK.value

    return _normalized_reason


def test_exit_binding_mismatch_not_mapped_to_risk():
    """exit_binding_mismatch should preserve its identity, not become 'risk'."""
    normalize = _build_normalized_reason()
    result = normalize("exit_binding_mismatch")
    assert result == "exit_binding_mismatch"
    assert result != BlockReason.RISK.value


def test_hold_rule_not_mapped_to_risk():
    """HOLD_RULE should preserve its identity, not become 'risk'."""
    normalize = _build_normalized_reason()
    result = normalize("HOLD_RULE")
    assert result == "HOLD_RULE"
    assert result != BlockReason.RISK.value


def test_unknown_reason_maps_to_risk():
    """Unknown strings (typically risk param names) still map to 'risk'."""
    normalize = _build_normalized_reason()
    result = normalize("max_position_risk_pct")
    assert result == BlockReason.RISK.value


def test_none_reason_maps_to_other():
    """None input maps to 'other'."""
    normalize = _build_normalized_reason()
    assert normalize(None) == BlockReason.OTHER.value


def test_known_enum_reasons_preserved():
    """BlockReason enum values pass through unchanged."""
    normalize = _build_normalized_reason()
    for reason in BlockReason:
        assert normalize(reason.value) == reason.value


def test_archetype_load_not_mapped_to_risk():
    """archetype_load should preserve its identity."""
    normalize = _build_normalized_reason()
    result = normalize("archetype_load")
    assert result == "archetype_load"
    assert result != BlockReason.RISK.value


def test_priority_skip_not_mapped_to_risk():
    """priority_skip should preserve its identity."""
    normalize = _build_normalized_reason()
    result = normalize("priority_skip")
    assert result == "priority_skip"
    assert result != BlockReason.RISK.value
