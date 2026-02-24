"""Tests for Runbook 47: Hard Template Binding via trigger compiler enforcement."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput,
    PortfolioState,
    RegimeAssessment,
    StrategyPlan,
    TriggerCondition,
)
from trading_core.trigger_compiler import (
    PlanEnforcementResult,
    TemplateViolation,
    enforce_plan_quality,
    enforce_template_identifiers,
)
from vector_store.retriever import allowed_identifiers_for_template


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime(2024, 6, 1, tzinfo=timezone.utc)


def _minimal_plan(**overrides) -> StrategyPlan:
    defaults = dict(
        generated_at=_ts(),
        valid_until=_ts(),
        global_view="test",
        regime="range",
        triggers=[],
        sizing_rules=[],
    )
    defaults.update(overrides)
    return StrategyPlan(**defaults)


def _trigger(
    trigger_id: str,
    entry_rule: str,
    category: str = "trend_continuation",
    direction: str = "long",
) -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol="BTC-USD",
        direction=direction,
        category=category,
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule="not is_flat and (stop_hit or target_hit)",
        stop_loss_pct=2.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Triggers using identifiers outside the template's set are blocked
# ---------------------------------------------------------------------------

def test_template_enforcement_blocks_invalid_identifiers(tmp_path):
    """Triggers using identifiers outside compression_breakout's allowed set are blocked."""
    plan = _minimal_plan(
        template_id="compression_breakout",
        triggers=[
            # Valid: uses compression_breakout identifiers
            _trigger("valid", "compression_flag > 0.5 and breakout_confirmed > 0.5"),
            # Invalid: uses rsi_14 and macd_hist — not in compression_breakout's allowed set
            _trigger("invalid", "rsi_14 > 60 and macd_hist > 0"),
        ],
    )

    violations = enforce_template_identifiers(plan)

    assert len(violations) == 1
    assert violations[0].trigger_id == "invalid"
    assert violations[0].template_id == "compression_breakout"
    assert "rsi_14" in violations[0].violations or "macd_hist" in violations[0].violations
    # Valid trigger should still be in the plan
    remaining_ids = {t.id for t in plan.triggers}
    assert "valid" in remaining_ids
    assert "invalid" not in remaining_ids


# ---------------------------------------------------------------------------
# Test 2: Emergency exits are exempt from template enforcement
# ---------------------------------------------------------------------------

def test_emergency_exit_exempt_from_template_enforcement():
    """emergency_exit category triggers pass enforcement regardless of identifiers."""
    plan = _minimal_plan(
        template_id="compression_breakout",
        triggers=[
            TriggerCondition(
                id="emergency",
                symbol="BTC-USD",
                direction="exit",
                category="emergency_exit",
                timeframe="1h",
                entry_rule="false",
                exit_rule="not is_flat and (vol_state == 'extreme' or unrealized_pnl_pct < -6.0)",
            ),
        ],
    )

    violations = enforce_template_identifiers(plan)

    assert violations == []
    assert len(plan.triggers) == 1
    assert plan.triggers[0].id == "emergency"


# ---------------------------------------------------------------------------
# Test 3: template_id=None → enforcement skipped entirely
# ---------------------------------------------------------------------------

def test_null_template_id_skips_enforcement():
    """When template_id is None, enforcement pass is a no-op."""
    plan = _minimal_plan(
        template_id=None,
        triggers=[
            # Uses arbitrary identifiers — would violate any template
            _trigger("any", "rsi_14 > 70 and macd_hist > 0 and some_made_up_indicator > 1"),
        ],
    )

    violations = enforce_template_identifiers(plan)

    assert violations == []
    assert len(plan.triggers) == 1  # nothing removed


# ---------------------------------------------------------------------------
# Test 4: Unknown template_id → warning logged, enforcement skipped (fail open)
# ---------------------------------------------------------------------------

def test_unknown_template_id_fails_open():
    """If template_id points to an unknown template, enforcement logs a warning and passes."""
    plan = _minimal_plan(
        template_id="completely_nonexistent_template_xyz",
        triggers=[
            _trigger("t", "rsi_14 > 50"),
        ],
    )

    with patch("trading_core.trigger_compiler.logger") as mock_log:
        violations = enforce_template_identifiers(plan)

    assert violations == []
    assert len(plan.triggers) == 1  # fail open — trigger kept
    # Verify a warning was issued
    assert mock_log.warning.called


# ---------------------------------------------------------------------------
# Test 5: StrategyPlan validates with template_id present
# ---------------------------------------------------------------------------

def test_strategy_plan_validates_with_template_id():
    """StrategyPlan with template_id='compression_breakout' validates without error."""
    plan = _minimal_plan(
        template_id="compression_breakout",
        template_parameters={"entry_vol_multiple_min": 1.5},
    )
    assert plan.template_id == "compression_breakout"
    assert plan.template_parameters == {"entry_vol_multiple_min": 1.5}


# ---------------------------------------------------------------------------
# Test 6: Existing plans without template_id still validate (backwards compatible)
# ---------------------------------------------------------------------------

def test_strategy_plan_validates_without_template_id():
    """Existing plans without template_id still validate — field is Optional."""
    plan = _minimal_plan()  # no template_id arg
    assert plan.template_id is None
    assert plan.template_parameters is None

    # Also verify round-trip JSON survives
    raw = plan.to_json()
    restored = StrategyPlan.from_json(raw)
    assert restored.template_id is None


# ---------------------------------------------------------------------------
# Test 7: allowed_identifiers_for_template returns correct set
# ---------------------------------------------------------------------------

def test_allowed_identifiers_returns_template_set():
    """allowed_identifiers_for_template('compression_breakout') returns the doc's identifiers."""
    ids = allowed_identifiers_for_template("compression_breakout")
    # compression_breakout.md has these identifiers in its frontmatter
    assert "compression_flag" in ids
    assert "breakout_confirmed" in ids
    assert "bb_bandwidth_pct_rank" in ids
    # Unknown template → empty set
    assert allowed_identifiers_for_template("nonexistent_xyz_template") == set()


# ---------------------------------------------------------------------------
# Test 8: enforce_plan_quality includes template_violations in result
# ---------------------------------------------------------------------------

def test_enforce_plan_quality_includes_template_violations():
    """enforce_plan_quality wires template enforcement and adds violations to result."""
    plan = _minimal_plan(
        template_id="compression_breakout",
        triggers=[
            _trigger("good", "compression_flag > 0.5 and vol_burst > 0"),
            _trigger("bad", "rsi_14 > 55 and macd_hist > 0"),  # not in template
        ],
    )

    result = enforce_plan_quality(plan, available_timeframes={"1h"})

    assert isinstance(result, PlanEnforcementResult)
    assert len(result.template_violations) == 1
    assert result.template_violations[0].trigger_id == "bad"
    # bad trigger should be removed from plan.triggers
    assert all(t.id != "bad" for t in plan.triggers)


# ---------------------------------------------------------------------------
# Test 9: TEMPLATE_ENFORCEMENT_ENABLED=false observes violations but does not remove triggers
# ---------------------------------------------------------------------------

def test_enforcement_disabled_observes_but_does_not_remove():
    """When TEMPLATE_ENFORCEMENT_ENABLED=false, violations are recorded but triggers kept."""
    plan = _minimal_plan(
        template_id="compression_breakout",
        triggers=[
            _trigger("bad", "rsi_14 > 55"),
        ],
    )

    with patch.dict(os.environ, {"TEMPLATE_ENFORCEMENT_ENABLED": "false"}):
        violations = enforce_template_identifiers(plan)

    assert len(violations) == 1  # violation detected
    assert len(plan.triggers) == 1  # trigger NOT removed (enforcement disabled)
