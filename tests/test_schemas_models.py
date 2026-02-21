from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from schemas.judge_feedback import JudgeFeedback
from schemas.llm_strategist import StrategyPlan, TriggerCondition


def _legacy_plan_payload() -> dict:
    generated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return {
        "generated_at": generated_at.isoformat(),
        "valid_until": (generated_at + timedelta(days=1)).isoformat(),
        "global_view": "legacy payload",
        "regime": "range",
        "triggers": [
            {
                "id": "buy_breakout",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "1h",
                "entry_rule": "close > sma_short",
                "exit_rule": "close < sma_medium",
                "stop_loss_pct": 2.0,
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 2.0,
            "max_symbol_exposure_pct": 25.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
        },
        "sizing_rules": [
            {"symbol": "BTC-USD", "sizing_mode": "fixed_fraction", "target_risk_pct": 1.0}
        ],
    }


def test_strategy_plan_assigns_defaults_for_legacy_json() -> None:
    payload = _legacy_plan_payload()
    plan = StrategyPlan.model_validate_json(json.dumps(payload))
    assert plan.plan_id.startswith("plan_")
    assert plan.run_id is None
    assert plan.max_trades_per_day is None
    assert plan.min_trades_per_day is None
    assert plan.allowed_symbols == []
    assert plan.allowed_directions == []
    assert plan.allowed_trigger_categories == []


def test_judge_feedback_parses_constraints() -> None:
    payload = {
        "score": 47.5,
        "notes": "oversized exposure",
        "constraints": {
            "max_trades_per_day": 5,
            "risk_mode": "conservative",
            "disabled_trigger_ids": ["btc_scalp"],
            "disabled_categories": ["scalp"],
        },
        "strategist_constraints": {
            "must_fix": ["Trim overexposed assets"],
            "vetoes": ["Avoid leverage"],
            "boost": [],
            "regime_correction": "mixed",
            "sizing_adjustments": {"BTC-USD": "cap at 1%"},
        },
    }
    feedback = JudgeFeedback.model_validate_json(json.dumps(payload))
    assert feedback.constraints.max_trades_per_day == 5
    assert feedback.constraints.risk_mode == "conservative"
    assert feedback.constraints.disabled_trigger_ids == ["btc_scalp"]
    assert feedback.strategist_constraints.must_fix == ["Trim overexposed assets"]
    assert feedback.strategist_constraints.sizing_adjustments["BTC-USD"] == "cap at 1%"


def test_judge_feedback_defaults_when_constraints_omitted() -> None:
    payload = {
        "score": 60.0,
        "strategist_constraints": {
            "must_fix": [],
            "vetoes": [],
            "boost": [],
            "regime_correction": "keep balanced",
            "sizing_adjustments": {},
        },
    }
    feedback = JudgeFeedback.model_validate(payload)
    assert feedback.constraints.max_trades_per_day is None
    assert feedback.constraints.risk_mode == "normal"
    assert feedback.strategist_constraints.regime_correction == "keep balanced"


# =============================================================================
# TriggerCondition schema tests for graduated de-risk categories
# =============================================================================


def _base_trigger() -> dict:
    """Base trigger payload without category-specific fields."""
    return {
        "id": "test_trigger",
        "symbol": "BTC-USD",
        "direction": "exit",
        "timeframe": "1h",
        "entry_rule": "false",
        "exit_rule": "not is_flat and rsi_14 > 70",
    }


class TestTriggerCategoryRiskReduce:
    """Tests for risk_reduce category."""

    def test_risk_reduce_category_valid(self) -> None:
        """risk_reduce is a valid category."""
        payload = _base_trigger()
        payload["category"] = "risk_reduce"
        payload["exit_fraction"] = 0.5
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.category == "risk_reduce"
        assert trigger.exit_fraction == 0.5

    def test_risk_reduce_without_exit_fraction_allowed(self) -> None:
        """risk_reduce without exit_fraction is allowed (gradual adoption)."""
        payload = _base_trigger()
        payload["category"] = "risk_reduce"
        # No exit_fraction - should still validate
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.category == "risk_reduce"
        assert trigger.exit_fraction is None


class TestTriggerCategoryRiskOff:
    """Tests for risk_off category."""

    def test_risk_off_category_valid(self) -> None:
        """risk_off is a valid category."""
        payload = _base_trigger()
        payload["category"] = "risk_off"
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.category == "risk_off"

    def test_risk_off_with_exit_fraction_one(self) -> None:
        """risk_off can have exit_fraction=1.0 (full exit)."""
        payload = _base_trigger()
        payload["category"] = "risk_off"
        payload["exit_fraction"] = 1.0
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction == 1.0


class TestExitFractionValidation:
    """Tests for exit_fraction field validation."""

    def test_exit_fraction_none_valid(self) -> None:
        """exit_fraction=None is valid (backward compatible)."""
        payload = _base_trigger()
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction is None

    def test_exit_fraction_half_valid(self) -> None:
        """exit_fraction=0.5 is valid."""
        payload = _base_trigger()
        payload["exit_fraction"] = 0.5
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction == 0.5

    def test_exit_fraction_one_valid(self) -> None:
        """exit_fraction=1.0 is valid (full exit)."""
        payload = _base_trigger()
        payload["exit_fraction"] = 1.0
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction == 1.0

    def test_exit_fraction_small_valid(self) -> None:
        """exit_fraction=0.1 is valid (small trim)."""
        payload = _base_trigger()
        payload["exit_fraction"] = 0.1
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction == 0.1

    def test_exit_fraction_zero_invalid(self) -> None:
        """exit_fraction=0 is invalid (must be > 0)."""
        payload = _base_trigger()
        payload["exit_fraction"] = 0.0
        with pytest.raises(ValueError, match="exit_fraction must be in range"):
            TriggerCondition.model_validate(payload)

    def test_exit_fraction_negative_invalid(self) -> None:
        """exit_fraction=-0.5 is invalid."""
        payload = _base_trigger()
        payload["exit_fraction"] = -0.5
        with pytest.raises(ValueError, match="exit_fraction must be in range"):
            TriggerCondition.model_validate(payload)

    def test_exit_fraction_greater_than_one_invalid(self) -> None:
        """exit_fraction=1.5 is invalid (cannot exit more than 100%)."""
        payload = _base_trigger()
        payload["exit_fraction"] = 1.5
        with pytest.raises(ValueError, match="exit_fraction must be in range"):
            TriggerCondition.model_validate(payload)


class TestEmergencyExitValidation:
    """Tests for emergency_exit category validation."""

    def test_emergency_exit_requires_exit_rule(self) -> None:
        """emergency_exit must have non-empty exit_rule."""
        payload = _base_trigger()
        payload["category"] = "emergency_exit"
        payload["exit_rule"] = ""
        with pytest.raises(ValueError, match="emergency_exit triggers must define a non-empty exit_rule"):
            TriggerCondition.model_validate(payload)

    def test_emergency_exit_whitespace_exit_rule_invalid(self) -> None:
        """emergency_exit with whitespace-only exit_rule is invalid."""
        payload = _base_trigger()
        payload["category"] = "emergency_exit"
        payload["exit_rule"] = "   "
        with pytest.raises(ValueError, match="emergency_exit triggers must define a non-empty exit_rule"):
            TriggerCondition.model_validate(payload)

    def test_emergency_exit_with_valid_exit_rule(self) -> None:
        """emergency_exit with valid exit_rule passes."""
        payload = _base_trigger()
        payload["category"] = "emergency_exit"
        payload["direction"] = "flat"
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.category == "emergency_exit"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing triggers."""

    def test_trigger_without_category(self) -> None:
        """Trigger without category defaults to None."""
        payload = _base_trigger()
        # No category field
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.category is None

    def test_trigger_without_exit_fraction(self) -> None:
        """Trigger without exit_fraction defaults to None."""
        payload = _base_trigger()
        payload["category"] = "trend_continuation"
        trigger = TriggerCondition.model_validate(payload)
        assert trigger.exit_fraction is None

    def test_existing_categories_still_valid(self) -> None:
        """All pre-existing categories remain valid."""
        existing_categories = [
            "trend_continuation",
            "reversal",
            "volatility_breakout",
            "mean_reversion",
            "emergency_exit",
            "other",
        ]
        for cat in existing_categories:
            payload = _base_trigger()
            payload["category"] = cat
            if cat == "emergency_exit":
                payload["direction"] = "flat"
            trigger = TriggerCondition.model_validate(payload)
            assert trigger.category == cat
