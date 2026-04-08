"""Block reason taxonomy for Phase 0 telemetry (Runbook 73).

Classifies trade block reasons into three classes:
- RISK_VALID: desired constraints (real risk limits doing their job)
- QUALITY_GATE: fixable plan defects (should be zero after Phase 0)
- INFRA: data or infrastructure failures (should be zero after R71)
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class BlockClass(str, Enum):
    RISK_VALID = "risk_valid"    # desired: real risk constraint working correctly
    QUALITY_GATE = "quality_gate"  # undesired: fixable plan defect
    INFRA = "infra"              # undesired: data/timeout/infrastructure issue
    UNKNOWN = "unknown"          # not yet classified


BLOCK_REASON_CLASSIFICATION: Dict[str, BlockClass] = {
    # ── risk_valid: expected, healthy blocks ──────────────────────────────
    "daily_cap": BlockClass.RISK_VALID,
    "risk_budget": BlockClass.RISK_VALID,
    "sizing_zero": BlockClass.RISK_VALID,
    "insufficient_rr": BlockClass.RISK_VALID,
    "direction": BlockClass.RISK_VALID,
    "HOLD_RULE": BlockClass.RISK_VALID,
    "MIN_HOLD_PERIOD": BlockClass.RISK_VALID,
    "emergency_exit_veto_min_hold": BlockClass.RISK_VALID,
    "emergency_exit_veto_same_bar": BlockClass.RISK_VALID,
    "exit_binding_mismatch": BlockClass.RISK_VALID,
    "exit_binding_plan_mismatch": BlockClass.RISK_VALID,
    "learning_disabled": BlockClass.RISK_VALID,
    "learning_short_disabled": BlockClass.RISK_VALID,
    "experiment_symbol_filter": BlockClass.RISK_VALID,
    "experiment_category_filter": BlockClass.RISK_VALID,
    "conflicting_signal_detected": BlockClass.RISK_VALID,
    "conflict_exit_min_hold": BlockClass.RISK_VALID,
    "learning_gate_closed": BlockClass.RISK_VALID,
    "symbol_veto": BlockClass.RISK_VALID,
    "SYMBOL_VETO": BlockClass.RISK_VALID,
    "category": BlockClass.RISK_VALID,
    "CATEGORY": BlockClass.RISK_VALID,

    # ── quality_gate: fixable plan defects ───────────────────────────────
    "no_target_rr_undefined": BlockClass.QUALITY_GATE,
    "target_price_unresolvable": BlockClass.QUALITY_GATE,
    "priority_skip": BlockClass.QUALITY_GATE,
    "priority_skip_invalid_candidate": BlockClass.QUALITY_GATE,
    "compile_time_target_violation": BlockClass.QUALITY_GATE,
    "missing_stop": BlockClass.QUALITY_GATE,
    "emergency_exit_missing_exit_rule": BlockClass.QUALITY_GATE,
    "emergency_exit_tautology": BlockClass.QUALITY_GATE,
    "EXPRESSION_ERROR": BlockClass.QUALITY_GATE,

    # ── infra: data / timeout / infrastructure ────────────────────────────
    "missing_indicator": BlockClass.INFRA,
    "MISSING_INDICATOR": BlockClass.INFRA,
    "price_feed_degraded": BlockClass.INFRA,
    "price_feed_unavailable": BlockClass.INFRA,
}


def classify_block_reason(reason: str) -> BlockClass:
    """Return the BlockClass for the given reason string.

    Falls back to UNKNOWN for unrecognized reasons.
    """
    return BLOCK_REASON_CLASSIFICATION.get(reason, BlockClass.UNKNOWN)
