"""Judge plan validation service — memory-backed, policy-boundary gate (Runbook 53).

Upgrades the judge loop from pure risk/logic validation to evidence-based plan
validation using diversified memory retrieval and playbook expectation metadata.

Validation runs as a layered gate:
  1. Deterministic constraints (risk/invariants/policy/kill switches)
  2. Playbook consistency (regime eligibility, mutation cooldown, invalidation)
  3. Memory failure-pattern scan (contrastive evidence from R51)
  4. Historical cluster support (win/loss balance, expectancy)
  5. Confidence calibration (proposal conviction vs evidence strength)

Policy boundary rules (enforced here, not in tick engine):
- Judge MUST NOT re-evaluate thesis during THESIS_ARMED activation ticks unless
  an explicit invalidation or safety override creates a new policy boundary.
- Judge MUST NOT permit target re-optimization during HOLD_LOCK unless explicitly
  allowed by playbook policy-stability rules.
- No playbook switching inside the active policy cooldown window unless a structural
  invalidation or safety override path is active.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from schemas.episode_memory import DiversifiedMemoryBundle, EpisodeMemoryRecord
from schemas.judge_feedback import (
    JudgeConfidenceCalibration,
    JudgePlanRevisionRequest,
    JudgeValidationDecision,
    JudgeValidationFindingClass,
    JudgeValidationVerdict,
)
from schemas.llm_strategist import StrategyPlan

logger = logging.getLogger(__name__)

_MAX_LOSER_CLUSTER_RATIO = 0.65   # >= this fraction of losers = memory contradiction
_MIN_CLUSTER_SIZE_FOR_SUPPORT = 3  # clusters smaller than this give weak support
_HIGH_CONVICTION_LABELS = {"high", "very_high"}
_MEMORY_STRONG_FAILURE_MODES = {
    "false_breakout_reversion",
    "trend_exhaustion_after_extension",
    "macro_news_whipsaw",
}


# ---------------------------------------------------------------------------
# Helper: cluster support from memory bundle
# ---------------------------------------------------------------------------


def _cluster_support(
    bundle: DiversifiedMemoryBundle,
) -> tuple[int, int, float]:
    """Return (wins, losses, win_rate) from bundle buckets."""
    wins = len(bundle.winning_contexts)
    losses = len(bundle.losing_contexts)
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    return wins, losses, win_rate


def _matching_failure_modes(
    bundle: DiversifiedMemoryBundle,
    plan: StrategyPlan,
) -> List[str]:
    """Return failure mode labels from loser episodes that are plausibly relevant."""
    from collections import Counter

    direction = (plan.allowed_directions or ["long"])[0] if plan.allowed_directions else None
    regime = plan.regime or ""

    counts: Counter = Counter()
    for ep in bundle.failure_mode_patterns + bundle.losing_contexts:
        for mode in ep.failure_modes:
            counts[mode] += 1

    # Direction-aware filtering: false_breakout is especially relevant for longs
    relevant = []
    for mode, count in counts.most_common():
        if count < 2:
            continue  # require at least 2 occurrences to flag
        relevant.append(mode)

    return relevant


# ---------------------------------------------------------------------------
# Step 1: Deterministic constraints
# ---------------------------------------------------------------------------


def _check_deterministic(
    *,
    is_thesis_armed: bool,
    is_hold_lock: bool,
    policy_cooldown_active: bool,
    is_playbook_switch: bool,
    has_invalidation_trigger: bool,
    has_safety_override: bool,
    plan: StrategyPlan,
) -> tuple[List[str], bool]:
    """Return (reasons, hard_reject)."""
    reasons: List[str] = []
    hard_reject = False

    # THESIS_ARMED: no re-evaluation unless invalidation/safety boundary exists
    if is_thesis_armed and not has_invalidation_trigger and not has_safety_override:
        reasons.append(
            "STRUCTURAL: judge cannot re-evaluate thesis during THESIS_ARMED activation "
            "window without a valid invalidation or safety override trigger"
        )
        hard_reject = True

    # HOLD_LOCK: no target re-optimization without explicit playbook allowance
    if is_hold_lock and not has_safety_override:
        reasons.append(
            "STRUCTURAL: judge cannot permit target re-optimization during HOLD_LOCK "
            "without an explicit playbook policy-stability exception and boundary trigger"
        )
        hard_reject = True

    # Policy cooldown: no playbook switching without invalidation/safety path
    if policy_cooldown_active and is_playbook_switch and not has_invalidation_trigger and not has_safety_override:
        reasons.append(
            "STRUCTURAL: playbook switch requested inside policy mutation cooldown window "
            "without a valid invalidation or safety override path — blocked"
        )
        hard_reject = True

    # Basic plan sanity
    if not plan.triggers:
        reasons.append("STRUCTURAL: plan has no triggers — cannot approve an empty plan")
        hard_reject = True

    return reasons, hard_reject


# ---------------------------------------------------------------------------
# Step 2: Playbook consistency
# ---------------------------------------------------------------------------


def _check_playbook_consistency(
    plan: StrategyPlan,
    playbook_regime_tags: Optional[List[str]],
    playbook_requires_invalidation: bool,
) -> List[str]:
    """Return list of consistency findings (non-blocking unless STRUCTURAL prefix)."""
    reasons: List[str] = []

    # Regime eligibility check
    if playbook_regime_tags and plan.regime:
        if plan.regime not in playbook_regime_tags:
            reasons.append(
                f"STRUCTURAL: plan regime '{plan.regime}' is not in playbook eligible regimes "
                f"{playbook_regime_tags} — playbook ineligible for current regime"
            )

    # Invalidation completeness (soft)
    if playbook_requires_invalidation:
        has_exit = any(t.direction == "exit" for t in plan.triggers)
        if not has_exit:
            reasons.append(
                "REVISE: playbook requires an explicit invalidation/exit condition but "
                "no exit trigger is present in the plan"
            )

    return reasons


# ---------------------------------------------------------------------------
# Step 3: Memory failure-pattern scan
# ---------------------------------------------------------------------------


def _check_memory_failure_patterns(
    plan: StrategyPlan,
    bundle: Optional[DiversifiedMemoryBundle],
) -> tuple[List[str], List[str], bool]:
    """Return (reasons, matched_failure_modes, is_strong_contradiction)."""
    if bundle is None:
        return [], [], False

    matched = _matching_failure_modes(bundle, plan)
    if not matched:
        return [], [], False

    strong_matches = [m for m in matched if m in _MEMORY_STRONG_FAILURE_MODES]
    reasons: List[str] = []

    for mode in matched:
        reasons.append(
            f"MEMORY: failure mode '{mode}' recurs in retrieved episode memory — "
            f"{'strong known pattern' if mode in _MEMORY_STRONG_FAILURE_MODES else 'observed pattern'}"
        )

    is_strong = len(strong_matches) >= 1
    return reasons, matched, is_strong


# ---------------------------------------------------------------------------
# Step 4: Historical cluster support
# ---------------------------------------------------------------------------


def _check_cluster_support(
    bundle: Optional[DiversifiedMemoryBundle],
) -> tuple[float, str, str]:
    """Return (win_rate, cluster_summary, calibration_label)."""
    if bundle is None:
        return 0.5, "No memory bundle available.", "weakly_supported"

    wins, losses, win_rate = _cluster_support(bundle)
    total = wins + losses

    if total < _MIN_CLUSTER_SIZE_FOR_SUPPORT:
        summary = f"Insufficient cluster size ({total} episodes) — evidence inconclusive."
        return win_rate, summary, "weakly_supported"

    if win_rate >= 0.60:
        summary = f"Strong historical support: {wins}W / {losses}L ({win_rate:.0%} win rate in similar contexts)."
        calibration: JudgeConfidenceCalibration = "supported"
    elif win_rate >= 0.40:
        summary = f"Mixed historical support: {wins}W / {losses}L ({win_rate:.0%} win rate in similar contexts)."
        calibration = "weakly_supported"
    else:
        summary = f"Weak historical support: {wins}W / {losses}L ({win_rate:.0%} win rate) — loser-skewed cluster."
        calibration = "unsupported"

    return win_rate, summary, calibration


# ---------------------------------------------------------------------------
# Step 5: Confidence calibration
# ---------------------------------------------------------------------------


def _check_confidence_calibration(
    stated_conviction: Optional[str],
    calibration: JudgeConfidenceCalibration,
    win_rate: float,
    failure_mode_count: int,
) -> List[str]:
    """Return list of calibration findings."""
    reasons: List[str] = []

    if stated_conviction in _HIGH_CONVICTION_LABELS and calibration == "unsupported":
        reasons.append(
            f"REVISE: stated conviction is '{stated_conviction}' but cluster evidence is "
            f"'unsupported' (win_rate={win_rate:.0%}, {failure_mode_count} failure modes in memory)"
        )
    elif stated_conviction in _HIGH_CONVICTION_LABELS and calibration == "weakly_supported" and failure_mode_count >= 2:
        reasons.append(
            f"REVISE: stated conviction is '{stated_conviction}' but evidence is only "
            f"'weakly_supported' with {failure_mode_count} failure mode matches — reduce conviction "
            "or provide a concrete divergence explanation"
        )

    return reasons


# ---------------------------------------------------------------------------
# Derive verdict from layered results
# ---------------------------------------------------------------------------


def _derive_verdict(
    *,
    deterministic_reasons: List[str],
    hard_reject: bool,
    playbook_reasons: List[str],
    memory_reasons: List[str],
    failure_modes: List[str],
    is_strong_memory_contradiction: bool,
    calibration_reasons: List[str],
    win_rate: float,
    cluster_summary: str,
    calibration: JudgeConfidenceCalibration,
    bundle: Optional[DiversifiedMemoryBundle],
) -> tuple[JudgeValidationDecision, JudgeValidationFindingClass, float]:
    """Derive (decision, finding_class, judge_confidence_score)."""

    all_structural = [r for r in deterministic_reasons + playbook_reasons if r.startswith("STRUCTURAL:")]
    all_revise = [r for r in playbook_reasons + memory_reasons + calibration_reasons if r.startswith("REVISE:")]

    if hard_reject or all_structural:
        return "reject", "structural_violation", 0.90

    if is_strong_memory_contradiction:
        return "revise", "memory_contradiction", 0.75

    if all_revise:
        # Choose the most specific finding class
        if memory_reasons:
            fc: JudgeValidationFindingClass = "memory_contradiction"
        elif calibration_reasons:
            fc = "statistical_suspicion"
        else:
            fc = "statistical_suspicion"
        return "revise", fc, 0.70

    if calibration == "unsupported" and len(failure_modes) >= 1:
        return "revise", "statistical_suspicion", 0.65

    # Approve
    score = 0.80 if calibration == "supported" else (0.65 if calibration == "weakly_supported" else 0.55)
    return "approve", "none", score


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------


class JudgePlanValidationService:
    """Evidence-based judge plan validation gate (R53).

    Usage::

        svc = JudgePlanValidationService()
        verdict = svc.validate_plan(
            plan=plan,
            memory_bundle=bundle,
            playbook_regime_tags=["range_bound", "bull_trending"],
            ...
        )
    """

    def validate_plan(
        self,
        plan: StrategyPlan,
        *,
        memory_bundle: Optional[DiversifiedMemoryBundle] = None,
        # Playbook context (from R52)
        playbook_regime_tags: Optional[List[str]] = None,
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
        revision_count: int = 0,
    ) -> JudgeValidationVerdict:
        """Run layered validation and return a typed verdict.

        All checks are deterministic; no LLM calls are made in this method.
        Memory bundle consumption is bounded to the pre-fetched bundle.
        """
        # Step 1: Deterministic constraints
        det_reasons, hard_reject = _check_deterministic(
            is_thesis_armed=is_thesis_armed,
            is_hold_lock=is_hold_lock,
            policy_cooldown_active=policy_cooldown_active,
            is_playbook_switch=is_playbook_switch,
            has_invalidation_trigger=has_invalidation_trigger,
            has_safety_override=has_safety_override,
            plan=plan,
        )

        if hard_reject:
            cited_ep_ids: List[str] = []
            return JudgeValidationVerdict(
                decision="reject",
                finding_class="structural_violation",
                reasons=det_reasons,
                judge_confidence_score=0.95,
                cited_episode_ids=cited_ep_ids,
                confidence_calibration="unsupported",
                revision_count=revision_count,
            )

        # Step 2: Playbook consistency
        pb_reasons = _check_playbook_consistency(
            plan=plan,
            playbook_regime_tags=playbook_regime_tags,
            playbook_requires_invalidation=playbook_requires_invalidation,
        )
        # Check if any playbook reason is structural → hard reject
        pb_structural = [r for r in pb_reasons if r.startswith("STRUCTURAL:")]
        if pb_structural:
            return JudgeValidationVerdict(
                decision="reject",
                finding_class="structural_violation",
                reasons=det_reasons + pb_reasons,
                judge_confidence_score=0.90,
                confidence_calibration="unsupported",
                revision_count=revision_count,
            )

        # Step 3: Memory failure-pattern scan
        mem_reasons, failure_modes, is_strong_mem_contradiction = _check_memory_failure_patterns(
            plan=plan,
            bundle=memory_bundle,
        )

        # Step 4: Historical cluster support
        win_rate, cluster_summary, calibration = _check_cluster_support(memory_bundle)

        # Step 5: Confidence calibration
        cal_reasons = _check_confidence_calibration(
            stated_conviction=stated_conviction,
            calibration=calibration,
            win_rate=win_rate,
            failure_mode_count=len(failure_modes),
        )

        # Derive final verdict
        decision, finding_class, confidence_score = _derive_verdict(
            deterministic_reasons=det_reasons,
            hard_reject=hard_reject,
            playbook_reasons=pb_reasons,
            memory_reasons=mem_reasons,
            failure_modes=failure_modes,
            is_strong_memory_contradiction=is_strong_mem_contradiction,
            calibration_reasons=cal_reasons,
            win_rate=win_rate,
            cluster_summary=cluster_summary,
            calibration=calibration,
            bundle=memory_bundle,
        )

        all_reasons = det_reasons + pb_reasons + mem_reasons + cal_reasons
        revise_reasons = [r[len("REVISE: "):] for r in all_reasons if r.startswith("REVISE:")]
        mem_evidence_refs = [memory_bundle.bundle_id] if memory_bundle else []
        cited_ids = [ep.episode_id for ep in (memory_bundle.losing_contexts if memory_bundle else [])[:10]]

        # Divergence explanation required when approving against contradictory memory
        divergence: Optional[str] = None
        if decision == "approve" and failure_modes:
            divergence = (
                f"Approving despite {len(failure_modes)} matching failure mode(s) "
                f"({', '.join(failure_modes[:3])}) because cluster win_rate={win_rate:.0%} "
                f"remains above breakeven and calibration='{calibration}'."
            )

        logger.debug(
            "JudgePlanValidation plan=%s decision=%s finding=%s confidence=%.2f",
            plan.plan_id,
            decision,
            finding_class,
            confidence_score,
        )

        return JudgeValidationVerdict(
            decision=decision,
            finding_class=finding_class,
            reasons=all_reasons,
            judge_confidence_score=confidence_score,
            memory_evidence_refs=mem_evidence_refs,
            cited_episode_ids=cited_ids,
            failure_pattern_matches=failure_modes,
            cluster_support_summary=cluster_summary,
            confidence_calibration=calibration,
            divergence_from_nearest_losers=divergence,
            requested_revisions=revise_reasons,
            revision_count=revision_count,
        )
