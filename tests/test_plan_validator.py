"""Tests for agents/strategies/plan_validator.py.

Covers:
  - Cross-TF ATR tautology detection (the bug from session paper-trading-5912e6a7)
  - Missing exit_rule on emergency_exit
  - Valid rules that must NOT be flagged
  - check_exit_rule_for_tautology runtime helper
  - repair_prompt / summary output
"""

import math

import pytest

from agents.strategies.plan_validator import (
    PlanValidationResult,
    TautologyMatch,
    TriggerError,
    _k_min,
    _parse_tf_minutes,
    _scan_atr_tautologies,
    check_exit_rule_for_tautology,
    validate_trigger_plan,
)


# ── _parse_tf_minutes ──────────────────────────────────────────────────────────

def test_parse_tf_minutes_known():
    assert _parse_tf_minutes("1h") == 60
    assert _parse_tf_minutes("6h") == 360
    assert _parse_tf_minutes("1d") == 1440
    assert _parse_tf_minutes("15m") == 15


def test_parse_tf_minutes_unknown_falls_back():
    assert _parse_tf_minutes("xyx") == 60


# ── _k_min ────────────────────────────────────────────────────────────────────

def test_k_min_1d_vs_1h():
    km = _k_min(1440, 60)
    assert abs(km - math.sqrt(24) * 1.3) < 0.01


def test_k_min_same_timeframe():
    # Same TF → ratio is 1, margin doesn't matter — k_min = 1.0
    assert _k_min(60, 60) == 1.0


def test_k_min_high_less_than_low():
    assert _k_min(60, 240) == 1.0


# ── _scan_atr_tautologies — the real bug from session paper-trading-5912e6a7 ──

ACTUAL_FAILING_RULE = (
    "not is_flat and "
    "(vol_state == 'extreme' or htf_daily_atr > 3 * atr_14 or close < nearest_support * 0.98)"
)


def test_actual_session_rule_detected():
    """The exact exit_rule from portfolio_emergency_exit that caused constant churn."""
    hits = _scan_atr_tautologies(ACTUAL_FAILING_RULE, base_minutes=60)
    assert len(hits) == 1
    t = hits[0]
    assert t.high_ident == "htf_daily_atr"
    assert t.high_minutes == 1440
    assert t.low_ident == "atr_14"
    assert t.low_minutes == 60
    assert t.k == 3.0
    assert t.k_min > t.k  # tautology confirmed
    assert "htf_daily_atr > 3 * atr_14" in t.fragment


# ── Pattern 1: IDENT > k * IDENT ──────────────────────────────────────────────

def test_cmp_k_tautology_daily_vs_hourly():
    hits = _scan_atr_tautologies("htf_daily_atr > 3 * atr_14", 60)
    assert len(hits) == 1
    assert hits[0].k == 3.0
    assert hits[0].k_min > 3.0


def test_cmp_k_passes_with_sufficient_k():
    """k=8 well above k_min≈6.4 for 1D vs 1H → no tautology."""
    hits = _scan_atr_tautologies("htf_daily_atr > 8 * atr_14", 60)
    assert hits == []


def test_cmp_k_4h_vs_1h_tautology():
    """4H vs 1H: sqrt(240/60)=2.0, k_min≈2.6. k=2 → tautology."""
    hits = _scan_atr_tautologies("tf_4h_atr > 2 * atr_14", 60)
    assert len(hits) == 1


def test_cmp_k_4h_vs_1h_safe():
    """k=3 > k_min≈2.6 → safe."""
    hits = _scan_atr_tautologies("tf_4h_atr > 3 * atr_14", 60)
    assert hits == []


# ── Pattern 2: IDENT / k ─────────────────────────────────────────────────────

def test_div_k_tautology():
    """atr_14 < htf_daily_atr / 3 is equivalent to htf_daily_atr > 3 * atr_14."""
    hits = _scan_atr_tautologies("atr_14 < htf_daily_atr / 3", 60)
    assert len(hits) == 1
    assert hits[0].k == 3.0


def test_div_k_safe():
    hits = _scan_atr_tautologies("atr_14 < htf_daily_atr / 8", 60)
    assert hits == []


# ── Pattern 3: Direct comparison (implicit k=1) ───────────────────────────────

def test_direct_comparison_tautology():
    """htf_daily_atr > atr_14 with implicit k=1 is always a tautology."""
    hits = _scan_atr_tautologies("htf_daily_atr > atr_14", 60)
    assert len(hits) == 1
    assert hits[0].k == 1.0


# ── Non-ATR comparisons must not be flagged ───────────────────────────────────

def test_non_atr_comparison_clean():
    rule = "not is_flat and close < nearest_support * 0.98"
    hits = _scan_atr_tautologies(rule, 60)
    assert hits == []


def test_vol_state_extreme_clean():
    rule = "not is_flat and vol_state == 'extreme'"
    hits = _scan_atr_tautologies(rule, 60)
    assert hits == []


def test_unrealized_pnl_clean():
    rule = "not is_flat and unrealized_pnl_pct < -5.0"
    hits = _scan_atr_tautologies(rule, 60)
    assert hits == []


def test_same_timeframe_atr_clean():
    """atr_14 > atr_14 * 1.5 — same TF, not a tautology."""
    hits = _scan_atr_tautologies("atr_14 > atr_14 * 1.5", 60)
    # Both resolve to base_minutes=60 → same TF → should NOT be flagged
    assert hits == []


# ── validate_trigger_plan ─────────────────────────────────────────────────────

def _make_plan(triggers: list) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-01-02T00:00:00+00:00",
        "regime": "bear",
        "triggers": triggers,
        "risk_constraints": {
            "max_position_risk_pct": 2.0,
            "max_symbol_exposure_pct": 25.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
        },
        "sizing_rules": [],
    }


def _emergency_exit(exit_rule: str, trigger_id: str = "test_exit") -> dict:
    return {
        "id": trigger_id,
        "symbol": "BTC-USD",
        "category": "emergency_exit",
        "direction": "exit",
        "timeframe": "1h",
        "confidence_grade": "A",
        "entry_rule": "false",
        "exit_rule": exit_rule,
    }


def test_validate_rejects_tautology():
    plan = _make_plan([_emergency_exit(ACTUAL_FAILING_RULE)])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert not result.is_valid
    assert len(result.hard_errors) == 1
    assert result.hard_errors[0].code == "EMERGENCY_EXIT_TAUTOLOGY"
    assert result.hard_errors[0].trigger_id == "test_exit"


def test_validate_rejects_missing_exit_rule():
    plan = _make_plan([_emergency_exit("")])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert not result.is_valid
    assert result.hard_errors[0].code == "MISSING_EXIT_RULE"


def test_validate_accepts_vol_state_rule():
    rule = "not is_flat and (vol_state == 'extreme' or unrealized_pnl_pct < -6.0)"
    plan = _make_plan([_emergency_exit(rule)])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert result.is_valid
    assert result.warnings == []


def test_validate_accepts_structural_break_rule():
    rule = "not is_flat and close < nearest_support * 0.97"
    plan = _make_plan([_emergency_exit(rule)])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert result.is_valid


def test_validate_skips_non_emergency_triggers():
    """Non-emergency triggers with bad ATR comparisons should NOT be flagged."""
    trigger = {
        "id": "btc_entry",
        "symbol": "BTC-USD",
        "category": "trend_continuation",
        "direction": "long",
        "timeframe": "1h",
        "confidence_grade": "A",
        "entry_rule": "is_flat and htf_daily_atr > 2 * atr_14",  # would be tautology if it were emergency
        "exit_rule": "not is_flat and stop_hit",
    }
    plan = _make_plan([trigger])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert result.is_valid  # only emergency_exit is validated for tautologies


def test_validate_multiple_triggers_mixed():
    """One clean + one tautological emergency exit → one error."""
    clean = _emergency_exit("not is_flat and vol_state == 'extreme'", trigger_id="clean_exit")
    bad = _emergency_exit(ACTUAL_FAILING_RULE, trigger_id="bad_exit")
    plan = _make_plan([clean, bad])
    result = validate_trigger_plan(plan, base_tf_minutes=60)
    assert not result.is_valid
    assert len(result.hard_errors) == 1
    assert result.hard_errors[0].trigger_id == "bad_exit"


# ── PlanValidationResult methods ──────────────────────────────────────────────

def test_summary_ok_when_valid():
    result = PlanValidationResult()
    assert result.summary() == "OK"


def test_summary_lists_errors():
    result = PlanValidationResult(
        hard_errors=[
            TriggerError(
                trigger_id="foo", category="emergency_exit",
                code="EMERGENCY_EXIT_TAUTOLOGY",
                message="tautology detected", raw_rule="htf_daily_atr > 3 * atr_14",
            )
        ]
    )
    assert "foo" in result.summary()
    assert "TAUTOLOGY" in result.summary()


def test_repair_prompt_empty_when_valid():
    result = PlanValidationResult()
    assert result.repair_prompt() == ""


def test_repair_prompt_contains_fix_options():
    plan = _make_plan([_emergency_exit(ACTUAL_FAILING_RULE)])
    result = validate_trigger_plan(plan)
    repair = result.repair_prompt()
    assert "REJECTED" in repair
    assert "vol_state" in repair
    assert "FIX OPTION" in repair


# ── check_exit_rule_for_tautology runtime helper ──────────────────────────────

def test_runtime_check_detects_tautology():
    hits = check_exit_rule_for_tautology(
        "not is_flat and htf_daily_atr > 3 * atr_14", "1h"
    )
    assert len(hits) == 1


def test_runtime_check_clean_rule():
    hits = check_exit_rule_for_tautology(
        "not is_flat and vol_state == 'extreme'", "1h"
    )
    assert hits == []


def test_runtime_check_default_timeframe():
    """Calling without explicit timeframe uses 1h as fallback."""
    hits = check_exit_rule_for_tautology("htf_daily_atr > 3 * atr_14")
    assert len(hits) == 1
