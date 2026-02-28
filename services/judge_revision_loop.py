"""Judge plan revision loop orchestrator (Runbook 53).

Manages the validate → revise → resubmit cycle for a single policy event.
Caps retries at `max_revisions` (default 2); auto-emits stand_down on
budget exhaustion so the event does not loop indefinitely.

Design:
- Stateless orchestrator — the caller supplies a revision callback that
  returns an updated StrategyPlan after receiving a JudgePlanRevisionRequest.
- All judge validation is delegated to JudgePlanValidationService (pure
  deterministic logic, no LLM calls inside the loop itself).
- The revision callback *may* invoke an LLM (e.g., strategist replanning),
  but that is outside the scope of this module.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from schemas.episode_memory import DiversifiedMemoryBundle
from schemas.judge_feedback import (
    JudgePlanRevisionRequest,
    JudgeValidationVerdict,
    RevisionLoopResult,
)
from schemas.llm_strategist import StrategyPlan
from services.judge_validation_service import JudgePlanValidationService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_REVISIONS = 2

# Type alias for the revision callback.
RevisionCallback = Callable[[JudgePlanRevisionRequest], Optional[StrategyPlan]]


def _build_revision_request(
    verdict: JudgeValidationVerdict,
    revision_number: int,
    max_revisions: int,
) -> JudgePlanRevisionRequest:
    """Build a typed revision request from a 'revise' verdict."""
    structural = [r for r in verdict.reasons if r.startswith("STRUCTURAL:")]
    revise = [r for r in verdict.reasons if r.startswith("REVISE:")]
    memory = [r for r in verdict.reasons if r.startswith("MEMORY:")]

    failing = structural + revise
    guidance: Optional[str] = None
    if revision_number >= max_revisions:
        guidance = (
            f"This is revision attempt {revision_number} of {max_revisions}. "
            "If the core concern cannot be addressed, consider withdrawing the "
            "plan — further revisions will trigger stand_down."
        )

    return JudgePlanRevisionRequest(
        verdict=verdict,
        revision_number=revision_number,
        max_revisions=max_revisions,
        failing_criteria=failing,
        cited_failure_patterns=verdict.failure_pattern_matches,
        expectation_mismatches=memory,
        revision_guidance=guidance,
    )


class JudgePlanRevisionLoopOrchestrator:
    """Run the judge validation + revision loop for one policy event.

    Parameters
    ----------
    validation_service:
        Pre-configured ``JudgePlanValidationService`` instance.
    max_revisions:
        Maximum revision attempts before auto-stand-down (default 2).
    """

    def __init__(
        self,
        validation_service: Optional[JudgePlanValidationService] = None,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
    ) -> None:
        self._svc = validation_service or JudgePlanValidationService()
        self._max_revisions = max_revisions

    def run(
        self,
        plan: StrategyPlan,
        revision_callback: RevisionCallback,
        *,
        # Memory / playbook context (forwarded to validation service)
        memory_bundle: Optional[DiversifiedMemoryBundle] = None,
        playbook_regime_tags: Optional[list[str]] = None,
        playbook_requires_invalidation: bool = False,
        # Policy state
        is_thesis_armed: bool = False,
        is_hold_lock: bool = False,
        policy_cooldown_active: bool = False,
        is_playbook_switch: bool = False,
        has_invalidation_trigger: bool = False,
        has_safety_override: bool = False,
        # Proposal context
        stated_conviction: Optional[str] = None,
    ) -> RevisionLoopResult:
        """Run the validation → revise loop and return a ``RevisionLoopResult``.

        The ``revision_callback`` is invoked when the verdict is ``revise``.
        It receives a ``JudgePlanRevisionRequest`` and must return either a
        revised ``StrategyPlan`` or ``None`` (treated as inability-to-revise →
        stand_down immediately).

        A ``reject`` verdict from the validation service terminates the loop
        immediately (no revision callback).  A ``stand_down`` verdict is
        forwarded as-is.
        """
        current_plan = plan
        revision_attempts = 0

        shared_kwargs: dict = dict(
            memory_bundle=memory_bundle,
            playbook_regime_tags=playbook_regime_tags,
            playbook_requires_invalidation=playbook_requires_invalidation,
            is_thesis_armed=is_thesis_armed,
            is_hold_lock=is_hold_lock,
            policy_cooldown_active=policy_cooldown_active,
            is_playbook_switch=is_playbook_switch,
            has_invalidation_trigger=has_invalidation_trigger,
            has_safety_override=has_safety_override,
            stated_conviction=stated_conviction,
        )

        while True:
            verdict = self._svc.validate_plan(
                current_plan,
                revision_count=revision_attempts,
                **shared_kwargs,
            )

            logger.debug(
                "RevisionLoop plan=%s attempt=%d decision=%s finding=%s",
                current_plan.plan_id,
                revision_attempts,
                verdict.decision,
                verdict.finding_class,
            )

            # --- Terminal: hard reject -----------------------------------------
            if verdict.decision == "reject":
                return RevisionLoopResult(
                    final_verdict=verdict,
                    revision_attempts=revision_attempts,
                    revision_budget_exhausted=False,
                    stand_down_reason=None,
                    accepted_plan_id=None,
                )

            # --- Terminal: stand_down (forwarded from service) ------------------
            if verdict.decision == "stand_down":
                return RevisionLoopResult(
                    final_verdict=verdict,
                    revision_attempts=revision_attempts,
                    revision_budget_exhausted=False,
                    stand_down_reason="Validation service issued stand_down",
                    accepted_plan_id=None,
                )

            # --- Terminal: approve ----------------------------------------------
            if verdict.decision == "approve":
                return RevisionLoopResult(
                    final_verdict=verdict,
                    revision_attempts=revision_attempts,
                    revision_budget_exhausted=False,
                    stand_down_reason=None,
                    accepted_plan_id=current_plan.plan_id,
                )

            # --- Revise path ----------------------------------------------------
            # verdict.decision == "revise"
            revision_attempts += 1

            if revision_attempts > self._max_revisions:
                # Budget exhausted — emit stand_down
                exhausted_verdict = JudgeValidationVerdict(
                    decision="stand_down",
                    finding_class=verdict.finding_class,
                    reasons=verdict.reasons
                    + [
                        f"STRUCTURAL: revision budget exhausted after "
                        f"{self._max_revisions} attempt(s) — auto stand_down"
                    ],
                    judge_confidence_score=verdict.judge_confidence_score,
                    memory_evidence_refs=verdict.memory_evidence_refs,
                    cited_episode_ids=verdict.cited_episode_ids,
                    failure_pattern_matches=verdict.failure_pattern_matches,
                    cluster_support_summary=verdict.cluster_support_summary,
                    confidence_calibration=verdict.confidence_calibration,
                    revision_count=revision_attempts,
                )
                return RevisionLoopResult(
                    final_verdict=exhausted_verdict,
                    revision_attempts=revision_attempts,
                    revision_budget_exhausted=True,
                    stand_down_reason=(
                        f"revision_budget_exhausted after {self._max_revisions} attempt(s)"
                    ),
                    accepted_plan_id=None,
                )

            revision_request = _build_revision_request(
                verdict=verdict,
                revision_number=revision_attempts,
                max_revisions=self._max_revisions,
            )

            revised_plan = revision_callback(revision_request)

            if revised_plan is None:
                # Callback signals inability to produce a valid plan
                unable_verdict = JudgeValidationVerdict(
                    decision="stand_down",
                    finding_class=verdict.finding_class,
                    reasons=verdict.reasons
                    + ["STRUCTURAL: revision callback returned None — unable to revise, stand_down"],
                    judge_confidence_score=verdict.judge_confidence_score,
                    memory_evidence_refs=verdict.memory_evidence_refs,
                    cited_episode_ids=verdict.cited_episode_ids,
                    failure_pattern_matches=verdict.failure_pattern_matches,
                    cluster_support_summary=verdict.cluster_support_summary,
                    confidence_calibration=verdict.confidence_calibration,
                    revision_count=revision_attempts,
                )
                return RevisionLoopResult(
                    final_verdict=unable_verdict,
                    revision_attempts=revision_attempts,
                    revision_budget_exhausted=False,
                    stand_down_reason="revision_callback_returned_none",
                    accepted_plan_id=None,
                )

            current_plan = revised_plan
