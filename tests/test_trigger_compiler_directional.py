"""Tests for R59 trigger compiler: directional target requirement + cross-direction enforcement.

Covers:
- _get_template_direction()
- enforce_directional_target_requirement() — missing_target_for_directional_template
- cross_direction_identifier violation
- directional_violations field on PlanEnforcementResult
- enforce_plan_quality() wiring
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from trading_core.trigger_compiler import (
    PlanEnforcementResult,
    TemplateViolation,
    _get_template_direction,
    enforce_directional_target_requirement,
    enforce_plan_quality,
)
from schemas.llm_strategist import (
    StrategyPlan,
    TriggerCondition,
    RiskConstraint,
    PositionSizingRule,
)

_NOW = datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trigger(
    tid: str = "t1",
    direction: str = "long",
    category: str = "volatility_breakout",
    target_anchor_type: str | None = None,
    entry_rule: str = "is_flat and close > 100",
    exit_rule: str = "not is_flat and below_stop",
) -> TriggerCondition:
    return TriggerCondition(
        id=tid,
        symbol="BTC-USD",
        direction=direction,
        timeframe="1h",
        entry_rule=entry_rule,
        exit_rule=exit_rule,
        category=category,
        stop_anchor_type="atr",
        stop_loss_atr_mult=1.5,
        target_anchor_type=target_anchor_type,
    )


def _emergency_trigger(tid: str = "em") -> TriggerCondition:
    return TriggerCondition(
        id=tid,
        symbol="BTC-USD",
        direction="long",
        timeframe="1h",
        entry_rule="is_flat and close > 100",
        exit_rule="not is_flat and below_stop",
        category="emergency_exit",
        stop_anchor_type="atr",
    )


def _plan(
    template_id: str | None = None,
    triggers: list[TriggerCondition] | None = None,
) -> StrategyPlan:
    return StrategyPlan(
        plan_id="test-plan",
        run_id="test-run",
        generated_at=_NOW,
        valid_until=_NOW + timedelta(days=1),
        global_view="test",
        regime="range",
        template_id=template_id,
        triggers=triggers or [],
        risk_constraints=RiskConstraint(
            max_position_risk_pct=1.0,
            max_symbol_exposure_pct=25.0,
            max_portfolio_exposure_pct=80.0,
            max_daily_loss_pct=3.0,
        ),
        sizing_rules=[PositionSizingRule(symbol="BTC-USD", sizing_mode="fixed_fraction", target_risk_pct=1.0)],
        max_trades_per_day=5,
    )


# ---------------------------------------------------------------------------
# _get_template_direction
# ---------------------------------------------------------------------------

class TestGetTemplateDirection:
    def test_compression_long(self):
        assert _get_template_direction("compression_breakout_long") == "long"

    def test_compression_short(self):
        assert _get_template_direction("compression_breakout_short") == "short"

    def test_volatile_long(self):
        assert _get_template_direction("volatile_breakout_long") == "long"

    def test_volatile_short(self):
        assert _get_template_direction("volatile_breakout_short") == "short"

    def test_range_long(self):
        assert _get_template_direction("range_long") == "long"

    def test_range_short(self):
        assert _get_template_direction("range_short") == "short"

    def test_bull_trending_is_long(self):
        assert _get_template_direction("bull_trending") == "long"

    def test_bear_defensive_is_short(self):
        assert _get_template_direction("bear_defensive") == "short"

    def test_compression_breakout_neutral_is_neutral(self):
        assert _get_template_direction("compression_breakout") == "neutral"

    def test_volatile_breakout_neutral_is_neutral(self):
        assert _get_template_direction("volatile_breakout") == "neutral"

    def test_none_is_neutral(self):
        assert _get_template_direction(None) == "neutral"

    def test_unknown_is_neutral(self):
        assert _get_template_direction("unknown_template") == "neutral"


# ---------------------------------------------------------------------------
# enforce_directional_target_requirement — missing target
# ---------------------------------------------------------------------------

class TestMissingTargetForDirectionalTemplate:
    def test_long_template_with_target_anchor_no_violation(self):
        t = _trigger(target_anchor_type="measured_move")
        plan = _plan("compression_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        assert violations == []

    def test_long_template_no_target_yields_violation(self):
        t = _trigger()  # target_anchor_type=None by default
        plan = _plan("compression_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        assert len(violations) == 1
        assert "missing_target_for_directional_template" in violations[0].violations[0]

    def test_short_template_no_target_yields_violation(self):
        t = _trigger(direction="short")  # no target_anchor_type
        plan = _plan("compression_breakout_short", [t])
        violations = enforce_directional_target_requirement(plan)
        missing = [v for v in violations if any("missing_target" in vv for vv in v.violations)]
        assert len(missing) >= 1

    def test_neutral_template_no_target_no_violation(self):
        """Neutral templates don't require targets."""
        t = _trigger()  # target_anchor_type=None by default
        plan = _plan("compression_breakout", [t])
        violations = enforce_directional_target_requirement(plan)
        assert violations == []

    def test_no_template_no_violation(self):
        t = _trigger()
        plan = _plan(None, [t])
        violations = enforce_directional_target_requirement(plan)
        assert violations == []

    def test_emergency_exit_exempt(self):
        em = _emergency_trigger()
        plan = _plan("compression_breakout_long", [em])
        violations = enforce_directional_target_requirement(plan)
        assert violations == []

    def test_non_entry_trigger_exempt(self):
        """Exit/flat triggers don't require targets."""
        t = TriggerCondition(
            id="risk_off_1",
            symbol="BTC-USD",
            direction="flat",
            timeframe="1h",
            entry_rule="not is_flat and below_stop",
            exit_rule="not is_flat and below_stop",
            category="risk_off",
            stop_anchor_type="atr",
        )
        plan = _plan("compression_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        missing = [v for v in violations if any("missing_target" in vv for vv in v.violations)]
        assert missing == []

    def test_violation_reports_correct_template_id(self):
        t = _trigger(target_anchor_type=None)
        plan = _plan("range_long", [t])
        violations = enforce_directional_target_requirement(plan)
        missing = [v for v in violations if any("missing_target" in vv for vv in v.violations)]
        assert len(missing) >= 1
        assert missing[0].template_id == "range_long"


# ---------------------------------------------------------------------------
# enforce_directional_target_requirement — cross-direction identifiers
# ---------------------------------------------------------------------------

class TestCrossDirectionIdentifiers:
    def test_long_template_with_donchian_lower_short_is_violation(self):
        """donchian_lower_short is a short-specific identifier; forbidden in long templates."""
        t = _trigger(
            entry_rule="is_flat and close > donchian_lower_short",
            target_anchor_type="measured_move",
        )
        plan = _plan("compression_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        cross = [v for v in violations if any("cross_direction_identifier" in vv for vv in v.violations)]
        assert len(cross) >= 1

    def test_short_template_with_donchian_upper_short_is_violation(self):
        """donchian_upper_short is a long-specific identifier; forbidden in short templates."""
        t = _trigger(
            direction="short",
            entry_rule="is_flat and close < donchian_upper_short",
            target_anchor_type="measured_move",
        )
        plan = _plan("compression_breakout_short", [t])
        violations = enforce_directional_target_requirement(plan)
        cross = [v for v in violations if any("cross_direction_identifier" in vv for vv in v.violations)]
        assert len(cross) >= 1

    def test_long_template_with_bollinger_lower_is_violation(self):
        t = _trigger(
            entry_rule="is_flat and close < bollinger_lower",
            target_anchor_type="measured_move",
        )
        plan = _plan("volatile_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        cross = [v for v in violations if any("cross_direction_identifier" in vv for vv in v.violations)]
        assert len(cross) >= 1

    def test_short_template_with_bollinger_upper_is_violation(self):
        t = _trigger(
            direction="short",
            entry_rule="is_flat and close > bollinger_upper",
            target_anchor_type="measured_move",
        )
        plan = _plan("volatile_breakout_short", [t])
        violations = enforce_directional_target_requirement(plan)
        cross = [v for v in violations if any("cross_direction_identifier" in vv for vv in v.violations)]
        assert len(cross) >= 1

    def test_long_template_using_donchian_upper_short_is_ok(self):
        """donchian_upper_short is a long-specific entry reference — allowed in long templates."""
        t = _trigger(
            entry_rule="is_flat and close > donchian_upper_short",
            target_anchor_type="measured_move",
        )
        plan = _plan("compression_breakout_long", [t])
        violations = enforce_directional_target_requirement(plan)
        cross = [v for v in violations if any("cross_direction_identifier" in vv for vv in v.violations)]
        assert cross == []

    def test_neutral_template_ignores_cross_direction(self):
        t = _trigger(
            entry_rule="is_flat and close > donchian_lower_short",
            target_anchor_type="measured_move",
        )
        plan = _plan("compression_breakout", [t])
        violations = enforce_directional_target_requirement(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# PlanEnforcementResult — directional_violations field
# ---------------------------------------------------------------------------

class TestPlanEnforcementResultDirectionalField:
    def test_field_defaults_empty(self):
        result = PlanEnforcementResult()
        assert result.directional_violations == []

    def test_directional_violations_counted_in_total_corrections(self):
        result = PlanEnforcementResult(
            directional_violations=[
                TemplateViolation(
                    trigger_id="t1",
                    template_id="compression_breakout_long",
                    violations=["missing_target_for_directional_template: ..."],
                )
            ]
        )
        assert result.total_corrections == 1

    def test_directional_plus_template_violations_summed(self):
        result = PlanEnforcementResult(
            template_violations=[
                TemplateViolation(trigger_id="t1", template_id="t", violations=["bad_ident"])
            ],
            directional_violations=[
                TemplateViolation(trigger_id="t2", template_id="t", violations=["missing_target"])
            ],
        )
        assert result.total_corrections == 2


# ---------------------------------------------------------------------------
# enforce_plan_quality — wiring check
# ---------------------------------------------------------------------------

class TestEnforcePlanQualityDirectionalWiring:
    def test_directional_violation_surfaced_in_result(self):
        t = _trigger(target_anchor_type=None)
        plan = _plan("range_long", [t])
        result = enforce_plan_quality(plan, available_timeframes={"1h"})
        assert len(result.directional_violations) >= 1

    def test_no_directional_violation_for_neutral_template(self):
        t = _trigger(target_anchor_type=None)
        plan = _plan("compression_breakout", [t])
        result = enforce_plan_quality(plan, available_timeframes={"1h"})
        directional_missing = [
            v for v in result.directional_violations
            if any("missing_target" in vv for vv in v.violations)
        ]
        assert directional_missing == []

    def test_no_directional_violation_when_target_set(self):
        t = _trigger(target_anchor_type="measured_move")
        plan = _plan("compression_breakout_long", [t])
        result = enforce_plan_quality(plan, available_timeframes={"1h"})
        directional_missing = [
            v for v in result.directional_violations
            if any("missing_target" in vv for vv in v.violations)
        ]
        assert directional_missing == []
