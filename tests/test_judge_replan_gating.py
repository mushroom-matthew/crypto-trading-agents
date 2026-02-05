"""Tests for judge replan/policy_adjust action gating.

Verifies:
- replan action only valid for plan/trigger attribution
- policy_adjust action only valid for policy attribution
- hold and stand_down work with any attribution
- investigate_execution works appropriately
"""

import pytest
from pydantic import ValidationError

from schemas.judge_feedback import (
    JudgeAttribution,
    AttributionEvidence,
)


def make_evidence() -> AttributionEvidence:
    """Create minimal valid evidence."""
    return AttributionEvidence(metrics=["test_metric"])


class TestReplanGating:
    """Test replan action gating rules."""

    def test_replan_allowed_for_plan(self):
        """replan is allowed when primary_attribution is plan."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="replan",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "replan"

    def test_replan_allowed_for_trigger(self):
        """replan is allowed when primary_attribution is trigger."""
        attr = JudgeAttribution(
            primary_attribution="trigger",
            recommended_action="replan",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "replan"

    def test_replan_blocked_for_policy(self):
        """replan is NOT allowed when primary_attribution is policy."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="policy",
                recommended_action="replan",
                evidence=make_evidence(),
            )
        assert "replan action requires attribution to 'plan' or 'trigger'" in str(exc_info.value)

    def test_replan_blocked_for_execution(self):
        """replan is NOT allowed when primary_attribution is execution."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="execution",
                recommended_action="replan",
                evidence=make_evidence(),
            )
        assert "replan action requires attribution to 'plan' or 'trigger'" in str(exc_info.value)

    def test_replan_blocked_for_safety(self):
        """replan is NOT allowed when primary_attribution is safety."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="safety",
                recommended_action="replan",
                evidence=make_evidence(),
            )
        assert "replan action requires attribution to 'plan' or 'trigger'" in str(exc_info.value)


class TestPolicyAdjustGating:
    """Test policy_adjust action gating rules."""

    def test_policy_adjust_allowed_for_policy(self):
        """policy_adjust is allowed when primary_attribution is policy."""
        attr = JudgeAttribution(
            primary_attribution="policy",
            recommended_action="policy_adjust",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "policy_adjust"

    def test_policy_adjust_blocked_for_plan(self):
        """policy_adjust is NOT allowed when primary_attribution is plan."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="plan",
                recommended_action="policy_adjust",
                evidence=make_evidence(),
            )
        assert "policy_adjust action requires attribution to 'policy'" in str(exc_info.value)

    def test_policy_adjust_blocked_for_trigger(self):
        """policy_adjust is NOT allowed when primary_attribution is trigger."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="trigger",
                recommended_action="policy_adjust",
                evidence=make_evidence(),
            )
        assert "policy_adjust action requires attribution to 'policy'" in str(exc_info.value)

    def test_policy_adjust_blocked_for_execution(self):
        """policy_adjust is NOT allowed when primary_attribution is execution."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="execution",
                recommended_action="policy_adjust",
                evidence=make_evidence(),
            )
        assert "policy_adjust action requires attribution to 'policy'" in str(exc_info.value)

    def test_policy_adjust_blocked_for_safety(self):
        """policy_adjust is NOT allowed when primary_attribution is safety."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="safety",
                recommended_action="policy_adjust",
                evidence=make_evidence(),
            )
        assert "policy_adjust action requires attribution to 'policy'" in str(exc_info.value)


class TestHoldAction:
    """Test hold action is universally allowed."""

    def test_hold_allowed_for_plan(self):
        """hold is allowed for plan attribution."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "hold"

    def test_hold_allowed_for_trigger(self):
        """hold is allowed for trigger attribution."""
        attr = JudgeAttribution(
            primary_attribution="trigger",
            recommended_action="hold",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "hold"

    def test_hold_allowed_for_policy(self):
        """hold is allowed for policy attribution."""
        attr = JudgeAttribution(
            primary_attribution="policy",
            recommended_action="hold",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "hold"

    def test_hold_allowed_for_execution(self):
        """hold is allowed for execution attribution."""
        attr = JudgeAttribution(
            primary_attribution="execution",
            recommended_action="hold",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "hold"

    def test_hold_allowed_for_safety(self):
        """hold is allowed for safety attribution."""
        attr = JudgeAttribution(
            primary_attribution="safety",
            recommended_action="hold",
            evidence=make_evidence(),
        )
        assert attr.recommended_action == "hold"


class TestStandDownAction:
    """Test stand_down action is universally allowed."""

    def test_stand_down_allowed_for_all_layers(self):
        """stand_down is allowed for all attribution layers."""
        layers = ["plan", "trigger", "policy", "execution", "safety"]
        for layer in layers:
            attr = JudgeAttribution(
                primary_attribution=layer,
                recommended_action="stand_down",
                evidence=make_evidence(),
            )
            assert attr.recommended_action == "stand_down"


class TestInvestigateExecutionAction:
    """Test investigate_execution action."""

    def test_investigate_allowed_for_all_layers(self):
        """investigate_execution is allowed for all layers (typically execution)."""
        layers = ["plan", "trigger", "policy", "execution", "safety"]
        for layer in layers:
            attr = JudgeAttribution(
                primary_attribution=layer,
                recommended_action="investigate_execution",
                evidence=make_evidence(),
            )
            assert attr.recommended_action == "investigate_execution"


class TestEvidenceRequirement:
    """Test evidence is required for attribution."""

    def test_empty_evidence_rejected(self):
        """Attribution without any evidence is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="plan",
                recommended_action="hold",
                evidence=AttributionEvidence(),  # Empty evidence
            )
        assert "Attribution must include evidence" in str(exc_info.value)

    def test_metrics_only_evidence_accepted(self):
        """Evidence with only metrics is accepted."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=AttributionEvidence(metrics=["score: 50"]),
        )
        assert len(attr.evidence.metrics) == 1

    def test_trade_sets_only_evidence_accepted(self):
        """Evidence with only trade_sets is accepted."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=AttributionEvidence(trade_sets=["ts_001"]),
        )
        assert len(attr.evidence.trade_sets) == 1

    def test_events_only_evidence_accepted(self):
        """Evidence with only events is accepted."""
        attr = JudgeAttribution(
            primary_attribution="safety",
            recommended_action="hold",
            evidence=AttributionEvidence(events=["emergency_001"]),
        )
        assert len(attr.evidence.events) == 1

    def test_notes_only_evidence_accepted(self):
        """Evidence with only notes is accepted."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=AttributionEvidence(notes="Manual assessment"),
        )
        assert attr.evidence.notes == "Manual assessment"


class TestGatingEdgeCases:
    """Test edge cases in action gating."""

    def test_secondary_factors_dont_affect_gating(self):
        """Secondary factors do not affect action gating (only primary matters)."""
        # Primary is trigger (allows replan), secondary includes policy
        attr = JudgeAttribution(
            primary_attribution="trigger",
            secondary_factors=["policy", "execution"],
            recommended_action="replan",
            evidence=make_evidence(),
        )
        # Should succeed because primary is trigger
        assert attr.recommended_action == "replan"

    def test_gating_uses_class_constants(self):
        """Verify gating rules use class-level constants."""
        assert "plan" in JudgeAttribution.REPLAN_ALLOWED_LAYERS
        assert "trigger" in JudgeAttribution.REPLAN_ALLOWED_LAYERS
        assert "policy" not in JudgeAttribution.REPLAN_ALLOWED_LAYERS

        assert "policy" in JudgeAttribution.POLICY_ADJUST_ALLOWED_LAYERS
        assert "plan" not in JudgeAttribution.POLICY_ADJUST_ALLOWED_LAYERS

    def test_both_validators_run(self):
        """Both action gating and evidence validators run."""
        # Empty evidence AND bad action combination
        with pytest.raises(ValidationError) as exc_info:
            JudgeAttribution(
                primary_attribution="execution",
                recommended_action="replan",
                evidence=AttributionEvidence(),  # Empty
            )
        # Should fail on one or both validators
        error_str = str(exc_info.value)
        assert "replan" in error_str or "evidence" in error_str.lower()
