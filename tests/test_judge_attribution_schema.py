"""Tests for JudgeAttribution schema validation.

Verifies:
- Attribution layers are correctly typed
- Confidence levels are validated
- Evidence requirements are enforced
- Serialization/deserialization works correctly
"""

import pytest
from pydantic import ValidationError

from schemas.judge_feedback import (
    AttributionEvidence,
    JudgeAttribution,
    JudgeFeedback,
)


class TestAttributionEvidence:
    """Test AttributionEvidence schema."""

    def test_evidence_all_fields(self):
        """Evidence can have all fields populated."""
        evidence = AttributionEvidence(
            metrics=["win_rate: 45%", "profit_factor: 0.8"],
            trade_sets=["ts_001", "ts_002"],
            events=["emergency_exit_001"],
            notes="High volatility period caused multiple stop-outs.",
        )
        assert len(evidence.metrics) == 2
        assert len(evidence.trade_sets) == 2
        assert len(evidence.events) == 1
        assert "volatility" in evidence.notes

    def test_evidence_minimal_metrics(self):
        """Evidence with only metrics is valid."""
        evidence = AttributionEvidence(metrics=["score: -5"])
        assert evidence.metrics == ["score: -5"]
        assert evidence.trade_sets == []
        assert evidence.events == []
        assert evidence.notes is None

    def test_evidence_minimal_notes(self):
        """Evidence with only notes is valid."""
        evidence = AttributionEvidence(notes="Manual override")
        assert evidence.notes == "Manual override"
        assert evidence.metrics == []

    def test_evidence_empty_is_valid_at_schema_level(self):
        """Empty evidence passes schema validation (gating is at Attribution level)."""
        evidence = AttributionEvidence()
        assert evidence.metrics == []
        assert evidence.notes is None


class TestJudgeAttribution:
    """Test JudgeAttribution schema validation."""

    def test_valid_plan_attribution(self):
        """Valid plan attribution with hold action."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            confidence="high",
            recommended_action="hold",
            evidence=AttributionEvidence(metrics=["score: 65"]),
            canonical_verdict="Plan performing as expected.",
        )
        assert attr.primary_attribution == "plan"
        assert attr.confidence == "high"
        assert attr.recommended_action == "hold"

    def test_valid_trigger_attribution_with_replan(self):
        """Trigger attribution allows replan action."""
        attr = JudgeAttribution(
            primary_attribution="trigger",
            recommended_action="replan",
            evidence=AttributionEvidence(metrics=["win_rate: 25%"]),
        )
        assert attr.primary_attribution == "trigger"
        assert attr.recommended_action == "replan"

    def test_valid_policy_attribution_with_adjust(self):
        """Policy attribution allows policy_adjust action."""
        attr = JudgeAttribution(
            primary_attribution="policy",
            recommended_action="policy_adjust",
            evidence=AttributionEvidence(notes="Sizing too aggressive"),
        )
        assert attr.primary_attribution == "policy"
        assert attr.recommended_action == "policy_adjust"

    def test_valid_execution_attribution(self):
        """Execution attribution with investigate action."""
        attr = JudgeAttribution(
            primary_attribution="execution",
            recommended_action="investigate_execution",
            evidence=AttributionEvidence(metrics=["slippage: 2.5%"]),
        )
        assert attr.primary_attribution == "execution"
        assert attr.recommended_action == "investigate_execution"

    def test_valid_safety_attribution(self):
        """Safety attribution with stand_down action."""
        attr = JudgeAttribution(
            primary_attribution="safety",
            recommended_action="stand_down",
            evidence=AttributionEvidence(events=["emergency_exit_001"]),
        )
        assert attr.primary_attribution == "safety"
        assert attr.recommended_action == "stand_down"

    def test_all_attribution_layers(self):
        """All attribution layers can be used."""
        layers = ["plan", "trigger", "policy", "execution", "safety"]
        for layer in layers:
            attr = JudgeAttribution(
                primary_attribution=layer,
                recommended_action="hold",
                evidence=AttributionEvidence(metrics=["test"]),
            )
            assert attr.primary_attribution == layer

    def test_all_confidence_levels(self):
        """All confidence levels are valid."""
        for conf in ["low", "medium", "high"]:
            attr = JudgeAttribution(
                primary_attribution="plan",
                confidence=conf,
                recommended_action="hold",
                evidence=AttributionEvidence(metrics=["test"]),
            )
            assert attr.confidence == conf

    def test_all_recommended_actions(self):
        """All recommended actions are valid (with appropriate layers)."""
        # hold works with any layer
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=AttributionEvidence(metrics=["test"]),
        )
        assert attr.recommended_action == "hold"

        # stand_down works with any layer
        attr = JudgeAttribution(
            primary_attribution="execution",
            recommended_action="stand_down",
            evidence=AttributionEvidence(metrics=["test"]),
        )
        assert attr.recommended_action == "stand_down"

    def test_secondary_factors(self):
        """Secondary factors can be specified."""
        attr = JudgeAttribution(
            primary_attribution="trigger",
            secondary_factors=["policy", "execution"],
            recommended_action="replan",
            evidence=AttributionEvidence(notes="Multiple factors contributed"),
        )
        assert attr.secondary_factors == ["policy", "execution"]

    def test_invalid_attribution_layer(self):
        """Invalid attribution layer raises error."""
        with pytest.raises(ValidationError):
            JudgeAttribution(
                primary_attribution="invalid_layer",
                evidence=AttributionEvidence(metrics=["test"]),
            )

    def test_invalid_confidence(self):
        """Invalid confidence level raises error."""
        with pytest.raises(ValidationError):
            JudgeAttribution(
                primary_attribution="plan",
                confidence="very_high",
                evidence=AttributionEvidence(metrics=["test"]),
            )

    def test_invalid_action(self):
        """Invalid action raises error."""
        with pytest.raises(ValidationError):
            JudgeAttribution(
                primary_attribution="plan",
                recommended_action="do_nothing",
                evidence=AttributionEvidence(metrics=["test"]),
            )


class TestAttributionSerialization:
    """Test serialization and deserialization."""

    def test_to_dict_and_back(self):
        """Attribution can be serialized and deserialized."""
        original = JudgeAttribution(
            primary_attribution="trigger",
            secondary_factors=["policy"],
            confidence="high",
            recommended_action="replan",
            evidence=AttributionEvidence(
                metrics=["win_rate: 30%"],
                notes="Low quality signals",
            ),
            canonical_verdict="Triggers firing in noise.",
        )

        data = original.model_dump()
        reconstructed = JudgeAttribution.model_validate(data)

        assert reconstructed.primary_attribution == original.primary_attribution
        assert reconstructed.secondary_factors == original.secondary_factors
        assert reconstructed.confidence == original.confidence
        assert reconstructed.recommended_action == original.recommended_action
        assert reconstructed.evidence.metrics == original.evidence.metrics
        assert reconstructed.canonical_verdict == original.canonical_verdict

    def test_json_serialization(self):
        """Attribution serializes to valid JSON."""
        attr = JudgeAttribution(
            primary_attribution="plan",
            recommended_action="hold",
            evidence=AttributionEvidence(metrics=["score: 70"]),
        )

        json_str = attr.model_dump_json()
        assert '"primary_attribution":"plan"' in json_str

        # Can parse back
        parsed = JudgeAttribution.model_validate_json(json_str)
        assert parsed.primary_attribution == "plan"


class TestJudgeFeedbackWithAttribution:
    """Test JudgeFeedback integration with attribution."""

    def test_feedback_with_attribution(self):
        """JudgeFeedback can include attribution."""
        feedback = JudgeFeedback(
            score=55.0,
            notes="Moderate performance with trigger issues.",
            attribution=JudgeAttribution(
                primary_attribution="trigger",
                recommended_action="replan",
                evidence=AttributionEvidence(metrics=["win_rate: 35%"]),
            ),
        )
        assert feedback.attribution is not None
        assert feedback.attribution.primary_attribution == "trigger"

    def test_feedback_without_attribution(self):
        """JudgeFeedback works without attribution."""
        feedback = JudgeFeedback(score=70.0, notes="Good performance.")
        assert feedback.attribution is None

    def test_feedback_serialization_with_attribution(self):
        """JudgeFeedback with attribution serializes correctly."""
        feedback = JudgeFeedback(
            score=45.0,
            attribution=JudgeAttribution(
                primary_attribution="policy",
                recommended_action="policy_adjust",
                evidence=AttributionEvidence(notes="Sizing needs adjustment"),
            ),
        )

        data = feedback.model_dump()
        assert data["attribution"]["primary_attribution"] == "policy"

        reconstructed = JudgeFeedback.model_validate(data)
        assert reconstructed.attribution.recommended_action == "policy_adjust"
