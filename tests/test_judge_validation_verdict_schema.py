"""Tests for R53 JudgeValidationVerdict, JudgePlanRevisionRequest, RevisionLoopResult schemas."""

import pytest
from pydantic import ValidationError

from schemas.judge_feedback import (
    JudgeConfidenceCalibration,
    JudgePlanRevisionRequest,
    JudgeValidationDecision,
    JudgeValidationFindingClass,
    JudgeValidationVerdict,
    RevisionLoopResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_verdict(
    decision: JudgeValidationDecision = "approve",
    finding_class: JudgeValidationFindingClass = "none",
    confidence: float = 0.75,
    calibration: JudgeConfidenceCalibration = "supported",
    reasons: list[str] | None = None,
) -> JudgeValidationVerdict:
    return JudgeValidationVerdict(
        decision=decision,
        finding_class=finding_class,
        judge_confidence_score=confidence,
        confidence_calibration=calibration,
        reasons=reasons or [],
    )


# ---------------------------------------------------------------------------
# JudgeValidationVerdict
# ---------------------------------------------------------------------------


class TestJudgeValidationVerdictSchema:
    def test_approve_defaults(self):
        v = make_verdict(decision="approve")
        assert v.decision == "approve"
        assert v.finding_class == "none"
        assert v.revision_count == 0
        assert v.requested_revisions == []
        assert v.failure_pattern_matches == []
        assert v.cited_episode_ids == []
        assert v.memory_evidence_refs == []

    def test_reject_structural(self):
        v = make_verdict(
            decision="reject",
            finding_class="structural_violation",
            calibration="unsupported",
        )
        assert v.decision == "reject"
        assert v.finding_class == "structural_violation"

    def test_revise_memory_contradiction(self):
        v = JudgeValidationVerdict(
            decision="revise",
            finding_class="memory_contradiction",
            judge_confidence_score=0.70,
            confidence_calibration="weakly_supported",
            failure_pattern_matches=["false_breakout_reversion"],
            cited_episode_ids=["ep-001", "ep-002"],
            requested_revisions=["address false_breakout_reversion pattern"],
        )
        assert v.decision == "revise"
        assert "false_breakout_reversion" in v.failure_pattern_matches
        assert len(v.cited_episode_ids) == 2

    def test_stand_down_verdict(self):
        v = make_verdict(decision="stand_down")
        assert v.decision == "stand_down"

    def test_confidence_score_bounds_valid(self):
        v = make_verdict(confidence=0.0)
        assert v.judge_confidence_score == 0.0
        v2 = make_verdict(confidence=1.0)
        assert v2.judge_confidence_score == 1.0

    def test_confidence_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="approve",
                judge_confidence_score=1.1,
                confidence_calibration="supported",
            )

    def test_confidence_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="approve",
                judge_confidence_score=-0.01,
                confidence_calibration="supported",
            )

    def test_invalid_decision_rejected(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="maybe",  # type: ignore[arg-type]
                confidence_calibration="supported",
            )

    def test_invalid_finding_class_rejected(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="approve",
                finding_class="unknown_class",  # type: ignore[arg-type]
                confidence_calibration="supported",
            )

    def test_invalid_calibration_rejected(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="approve",
                confidence_calibration="very_confident",  # type: ignore[arg-type]
            )

    def test_revision_count_non_negative(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="revise",
                confidence_calibration="weakly_supported",
                revision_count=-1,
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            JudgeValidationVerdict(
                decision="approve",
                confidence_calibration="supported",
                unknown_field="oops",  # type: ignore[call-arg]
            )

    def test_divergence_explanation_stored(self):
        v = make_verdict(decision="approve")
        v2 = v.model_copy(update={
            "divergence_from_nearest_losers": "Diverges because momentum structure is intact.",
        })
        assert v2.divergence_from_nearest_losers is not None

    def test_cluster_support_summary_stored(self):
        v = JudgeValidationVerdict(
            decision="approve",
            confidence_calibration="supported",
            cluster_support_summary="3W / 1L (75% win rate in similar contexts).",
        )
        assert "75%" in v.cluster_support_summary


# ---------------------------------------------------------------------------
# JudgePlanRevisionRequest
# ---------------------------------------------------------------------------


class TestJudgePlanRevisionRequest:
    def test_basic_construction(self):
        verdict = make_verdict(decision="revise", finding_class="statistical_suspicion")
        req = JudgePlanRevisionRequest(
            verdict=verdict,
            revision_number=1,
            max_revisions=2,
            failing_criteria=["REVISE: reduce stated conviction"],
            cited_failure_patterns=["trend_exhaustion_after_extension"],
        )
        assert req.revision_number == 1
        assert req.max_revisions == 2
        assert "REVISE: reduce stated conviction" in req.failing_criteria

    def test_revision_number_must_be_ge_1(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            JudgePlanRevisionRequest(
                verdict=verdict,
                revision_number=0,
                max_revisions=2,
            )

    def test_max_revisions_must_be_ge_1(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            JudgePlanRevisionRequest(
                verdict=verdict,
                revision_number=1,
                max_revisions=0,
            )

    def test_optional_guidance_field(self):
        verdict = make_verdict(decision="revise")
        req = JudgePlanRevisionRequest(
            verdict=verdict,
            revision_number=2,
            max_revisions=2,
            revision_guidance="Last revision attempt — consider withdrawing plan.",
        )
        assert "Last revision attempt" in req.revision_guidance

    def test_extra_fields_forbidden(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            JudgePlanRevisionRequest(
                verdict=verdict,
                revision_number=1,
                max_revisions=2,
                surprise_field="x",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# RevisionLoopResult
# ---------------------------------------------------------------------------


class TestRevisionLoopResult:
    def test_approved_result(self):
        verdict = make_verdict(decision="approve")
        result = RevisionLoopResult(
            final_verdict=verdict,
            revision_attempts=0,
            accepted_plan_id="plan-abc",
        )
        assert result.final_verdict.decision == "approve"
        assert result.accepted_plan_id == "plan-abc"
        assert not result.revision_budget_exhausted

    def test_rejected_result_no_plan(self):
        verdict = make_verdict(decision="reject", finding_class="structural_violation")
        result = RevisionLoopResult(
            final_verdict=verdict,
            revision_attempts=0,
        )
        assert result.accepted_plan_id is None
        assert result.stand_down_reason is None

    def test_budget_exhausted_stand_down(self):
        verdict = make_verdict(decision="stand_down")
        result = RevisionLoopResult(
            final_verdict=verdict,
            revision_attempts=3,
            revision_budget_exhausted=True,
            stand_down_reason="revision_budget_exhausted after 2 attempt(s)",
        )
        assert result.revision_budget_exhausted
        assert "revision_budget_exhausted" in result.stand_down_reason

    def test_revision_attempts_non_negative(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            RevisionLoopResult(
                final_verdict=verdict,
                revision_attempts=-1,
            )

    def test_extra_fields_forbidden(self):
        verdict = make_verdict()
        with pytest.raises(ValidationError):
            RevisionLoopResult(
                final_verdict=verdict,
                revision_attempts=0,
                extra_noise="bad",  # type: ignore[call-arg]
            )
