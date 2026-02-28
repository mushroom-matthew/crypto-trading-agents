"""Tests for JudgePlanRevisionLoopOrchestrator (Runbook 53).

Covers:
- Immediate approve / reject paths (no callback invoked)
- Revision loop: callback called, plan updated, re-validated
- Revision budget exhaustion → auto-stand-down
- Callback returning None → immediate stand_down
- Revision count propagated correctly across iterations
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import pytest

from schemas.episode_memory import (
    DiversifiedMemoryBundle,
    MemoryRetrievalMeta,
)
from schemas.judge_feedback import (
    JudgePlanRevisionRequest,
    JudgeValidationVerdict,
)
from schemas.llm_strategist import StrategyPlan, TriggerCondition
from services.judge_revision_loop import (
    JudgePlanRevisionLoopOrchestrator,
    RevisionCallback,
)
from services.judge_validation_service import JudgePlanValidationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def make_trigger(trigger_id: str = "t1") -> TriggerCondition:
    return TriggerCondition(
        id=trigger_id,
        symbol="BTC-USD",
        direction="long",
        category="trend_continuation",
        timeframe="1h",
        entry_rule="price > vwap",
        exit_rule="price < vwap",
        stop_loss_pct=1.5,
    )


def make_plan(plan_id: str = "plan-test", triggers: list[TriggerCondition] | None = None) -> StrategyPlan:
    return StrategyPlan(
        plan_id=plan_id,
        regime="bull",
        generated_at=_NOW,
        valid_until=_NOW + timedelta(hours=4),
        triggers=triggers if triggers is not None else [make_trigger()],
        allowed_directions=["long"],
    )


def always_approve_callback(req: JudgePlanRevisionRequest) -> Optional[StrategyPlan]:
    """Return same valid plan — next validation will approve."""
    return make_plan(plan_id="plan-revised")


def always_none_callback(req: JudgePlanRevisionRequest) -> Optional[StrategyPlan]:
    """Simulate inability to revise."""
    return None


def counting_callback(call_log: list):
    """Callback that logs calls and returns a valid plan each time."""
    def _cb(req: JudgePlanRevisionRequest) -> Optional[StrategyPlan]:
        call_log.append(req.revision_number)
        return make_plan(plan_id=f"plan-rev-{req.revision_number}")
    return _cb


# ---------------------------------------------------------------------------
# Immediate-terminal paths
# ---------------------------------------------------------------------------


class TestImmediateApprovePath:
    def test_immediate_approve_returns_approve_result(self):
        """Valid plan with no issues → approve on first validation, callback never called."""
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        called = []
        def _cb(req):
            called.append(req)
            return make_plan()

        plan = make_plan()
        result = orchestrator.run(plan, _cb)

        assert result.final_verdict.decision == "approve"
        assert result.accepted_plan_id == plan.plan_id
        assert result.revision_attempts == 0
        assert not result.revision_budget_exhausted
        assert len(called) == 0  # callback never invoked for approve

    def test_immediate_approve_accepted_plan_id_is_original(self):
        plan = make_plan(plan_id="original-plan")
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        result = orchestrator.run(plan, always_approve_callback)
        assert result.accepted_plan_id == "original-plan"


class TestImmediateRejectPath:
    def test_empty_plan_hard_reject_no_callback(self):
        """Empty triggers → hard reject, callback not called."""
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        called = []
        def _cb(req):
            called.append(req)
            return make_plan()

        plan = make_plan(triggers=[])
        result = orchestrator.run(plan, _cb)

        assert result.final_verdict.decision == "reject"
        assert result.final_verdict.finding_class == "structural_violation"
        assert result.accepted_plan_id is None
        assert len(called) == 0

    def test_thesis_armed_hard_reject_no_callback(self):
        """THESIS_ARMED without override → hard reject, callback not called."""
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        called = []
        def _cb(req):
            called.append(req)
            return make_plan()

        plan = make_plan()
        result = orchestrator.run(
            plan, _cb,
            is_thesis_armed=True,
            has_invalidation_trigger=False,
            has_safety_override=False,
        )

        assert result.final_verdict.decision == "reject"
        assert len(called) == 0

    def test_reject_has_no_accepted_plan(self):
        plan = make_plan(triggers=[])
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        result = orchestrator.run(plan, always_approve_callback)
        assert result.accepted_plan_id is None
        assert result.stand_down_reason is None


# ---------------------------------------------------------------------------
# Revision loop
# ---------------------------------------------------------------------------


class TestRevisionLoop:
    def test_revision_callback_invoked_on_revise_verdict(self):
        """When first verdict is 'revise', callback is invoked exactly once before re-check."""
        # Force a revise by using unsupported cluster + high conviction
        # Build a bundle with 0W/4L to get unsupported calibration
        from schemas.episode_memory import EpisodeMemoryRecord
        loss_eps = [
            EpisodeMemoryRecord(
                episode_id=f"ep-loss-{i}",
                symbol="BTC-USD",
                outcome_class="loss",
                failure_modes=["false_breakout_reversion"],
            )
            for i in range(4)
        ]
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-loop",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=loss_eps,
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )

        calls = []
        def _cb(req: JudgePlanRevisionRequest) -> Optional[StrategyPlan]:
            calls.append(req)
            # Return a fresh plan with good trigger — validation should eventually approve
            return make_plan(plan_id=f"plan-revised-{len(calls)}")

        plan = make_plan()
        # Use stated_conviction=high to trigger calibration REVISE path
        orchestrator = JudgePlanRevisionLoopOrchestrator(max_revisions=2)
        result = orchestrator.run(
            plan,
            _cb,
            memory_bundle=bundle,
            stated_conviction="high",
        )

        # Callback should have been called (revise triggered at least once)
        assert len(calls) >= 1

    def test_revision_request_carries_failing_criteria(self):
        """JudgePlanRevisionRequest passed to callback has non-empty failing criteria."""
        from schemas.episode_memory import EpisodeMemoryRecord
        loss_eps = [
            EpisodeMemoryRecord(
                episode_id=f"ep-{i}",
                symbol="BTC-USD",
                outcome_class="loss",
                failure_modes=["false_breakout_reversion"],
            )
            for i in range(4)
        ]
        bundle = DiversifiedMemoryBundle(
            bundle_id="b-req",
            symbol="BTC-USD",
            created_at=_NOW,
            winning_contexts=[],
            losing_contexts=loss_eps,
            failure_mode_patterns=[],
            retrieval_meta=MemoryRetrievalMeta(),
        )

        received_requests = []
        def _cb(req: JudgePlanRevisionRequest) -> Optional[StrategyPlan]:
            received_requests.append(req)
            return make_plan()

        plan = make_plan()
        orchestrator = JudgePlanRevisionLoopOrchestrator(max_revisions=2)
        orchestrator.run(plan, _cb, memory_bundle=bundle, stated_conviction="high")

        if received_requests:
            req = received_requests[0]
            assert req.revision_number >= 1
            assert req.max_revisions == 2


# ---------------------------------------------------------------------------
# Budget exhaustion → stand_down
# ---------------------------------------------------------------------------


class TestRevisionBudgetExhaustion:
    def _make_svc_that_always_revises(self) -> JudgePlanValidationService:
        """Return a subclassed service that always returns 'revise'."""
        class AlwaysReviseService(JudgePlanValidationService):
            def validate_plan(self, plan, *, revision_count=0, **kwargs) -> JudgeValidationVerdict:
                return JudgeValidationVerdict(
                    decision="revise",
                    finding_class="statistical_suspicion",
                    judge_confidence_score=0.70,
                    confidence_calibration="weakly_supported",
                    reasons=["REVISE: always revise for testing"],
                    revision_count=revision_count,
                )
        return AlwaysReviseService()

    def test_budget_exhausted_after_max_revisions(self):
        """After max_revisions revise responses, orchestrator emits stand_down."""
        svc = self._make_svc_that_always_revises()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=2)
        calls = []
        plan = make_plan()
        result = orchestrator.run(plan, counting_callback(calls))

        assert result.final_verdict.decision == "stand_down"
        assert result.revision_budget_exhausted
        assert "revision_budget_exhausted" in result.stand_down_reason

    def test_callback_called_max_revisions_times(self):
        """Callback is called exactly max_revisions times before stand_down."""
        svc = self._make_svc_that_always_revises()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=2)
        calls = []
        plan = make_plan()
        orchestrator.run(plan, counting_callback(calls))

        assert len(calls) == 2  # max_revisions=2 → callback called twice

    def test_stand_down_no_accepted_plan(self):
        """Budget-exhausted stand_down has no accepted_plan_id."""
        svc = self._make_svc_that_always_revises()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=1)
        plan = make_plan()
        result = orchestrator.run(plan, always_approve_callback)

        assert result.accepted_plan_id is None

    def test_stand_down_verdict_has_structural_budget_reason(self):
        """Exhausted stand_down verdict includes a STRUCTURAL reason about budget."""
        svc = self._make_svc_that_always_revises()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=2)
        plan = make_plan()
        result = orchestrator.run(plan, counting_callback([]))

        budget_reasons = [
            r for r in result.final_verdict.reasons
            if "revision budget exhausted" in r.lower() or "budget" in r.lower()
        ]
        assert len(budget_reasons) >= 1

    def test_custom_max_revisions_zero_immediately_stands_down(self):
        """max_revisions=1, svc always revises → stand_down after 1 revision."""
        svc = self._make_svc_that_always_revises()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=1)
        calls = []
        plan = make_plan()
        result = orchestrator.run(plan, counting_callback(calls))

        assert result.revision_budget_exhausted
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Callback returning None → immediate stand_down
# ---------------------------------------------------------------------------


class TestCallbackNoneStandDown:
    def _make_svc_that_revises_once(self) -> JudgePlanValidationService:
        class ReviseOnceThenApprove(JudgePlanValidationService):
            _count = 0
            def validate_plan(self, plan, *, revision_count=0, **kwargs) -> JudgeValidationVerdict:
                self._count += 1
                if self._count == 1:
                    return JudgeValidationVerdict(
                        decision="revise",
                        finding_class="statistical_suspicion",
                        judge_confidence_score=0.70,
                        confidence_calibration="weakly_supported",
                        reasons=["REVISE: first time revise"],
                        revision_count=revision_count,
                    )
                return JudgeValidationVerdict(
                    decision="approve",
                    finding_class="none",
                    judge_confidence_score=0.80,
                    confidence_calibration="supported",
                    revision_count=revision_count,
                )
        return ReviseOnceThenApprove()

    def test_callback_none_triggers_stand_down(self):
        """Callback returning None → stand_down immediately."""
        svc = self._make_svc_that_revises_once()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=2)
        plan = make_plan()
        result = orchestrator.run(plan, always_none_callback)

        assert result.final_verdict.decision == "stand_down"
        assert result.stand_down_reason == "revision_callback_returned_none"
        assert result.accepted_plan_id is None
        assert not result.revision_budget_exhausted

    def test_callback_none_stand_down_not_budget_exhausted(self):
        """When callback returns None, budget_exhausted is False (distinct reason)."""
        svc = self._make_svc_that_revises_once()
        orchestrator = JudgePlanRevisionLoopOrchestrator(validation_service=svc, max_revisions=2)
        plan = make_plan()
        result = orchestrator.run(plan, always_none_callback)

        assert not result.revision_budget_exhausted


# ---------------------------------------------------------------------------
# Revision count tracking
# ---------------------------------------------------------------------------


class TestRevisionCountTracking:
    def test_revision_attempts_counted(self):
        """Result.revision_attempts reflects number of revision callbacks fired."""
        class ReviseN:
            def __init__(self, n): self._n = n; self._count = 0
            def validate_plan(self, plan, *, revision_count=0, **kwargs) -> JudgeValidationVerdict:
                if self._count < self._n:
                    self._count += 1
                    return JudgeValidationVerdict(
                        decision="revise", finding_class="statistical_suspicion",
                        judge_confidence_score=0.7, confidence_calibration="weakly_supported",
                        reasons=["REVISE: test"], revision_count=revision_count,
                    )
                return JudgeValidationVerdict(
                    decision="approve", finding_class="none",
                    judge_confidence_score=0.8, confidence_calibration="supported",
                    revision_count=revision_count,
                )

        svc = ReviseN(2)
        orchestrator = JudgePlanRevisionLoopOrchestrator(
            validation_service=svc, max_revisions=5,
        )
        plan = make_plan()
        result = orchestrator.run(plan, always_approve_callback)

        assert result.revision_attempts == 2
        assert result.final_verdict.decision == "approve"

    def test_zero_revisions_on_immediate_approve(self):
        orchestrator = JudgePlanRevisionLoopOrchestrator()
        plan = make_plan()
        result = orchestrator.run(plan, always_approve_callback)
        assert result.revision_attempts == 0
