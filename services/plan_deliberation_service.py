"""R92: Two-hypothesis deliberation service.

After the primary strategist generates a plan, a challenger plan arguing the
opposing or defensive stance is scored alongside it.  The comparison signal
surfaces plan robustness: if the challenger scores equally or higher, the judge
raises its approval threshold for the primary plan.

All scoring is deterministic (PlanHallucinationScorer) — no additional LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from schemas.judge_feedback import DeliberationVerdict

logger = logging.getLogger(__name__)

# Weight applied to REJECT findings vs REVISE (REJECT is more severe)
_REJECT_WEIGHT = 2.0
_REVISE_WEIGHT = 1.0


def _score_plan(plan: Any, llm_input: Any = None, judge_constraints: Any = None) -> float:
    """Score a plan with PlanHallucinationScorer.

    Returns a scalar ∈ [0, 1] where 1 = no findings, 0 = all REJECT.
    Formula: 1 - (weighted_finding_count / max_possible_weight).
    """
    try:
        from services.plan_hallucination_scorer import PlanHallucinationScorer
        report = PlanHallucinationScorer().score(plan, llm_input=llm_input, judge_constraints=judge_constraints)
        penalty = sum(
            _REJECT_WEIGHT if f.severity == "REJECT" else _REVISE_WEIGHT
            for f in report.findings
        )
        # Cap at max_possible (6 sections × REJECT_WEIGHT each)
        max_penalty = 6 * _REJECT_WEIGHT
        return max(0.0, 1.0 - penalty / max_penalty)
    except Exception as exc:
        logger.debug("plan_deliberation: scoring failed (non-fatal): %s", exc)
        return 0.5  # neutral when scorer unavailable


def _find_divergence_points(primary: Any, challenger: Any) -> list[str]:
    """Identify the key fields where the two plans disagree most."""
    points: list[str] = []
    try:
        p_regime = getattr(primary, "regime", None)
        c_regime = getattr(challenger, "regime", None)
        if p_regime and c_regime and p_regime != c_regime:
            points.append(f"regime: {p_regime} vs {c_regime}")

        p_stance = getattr(primary, "stance", None)
        c_stance = getattr(challenger, "stance", None)
        if p_stance and c_stance and p_stance != c_stance:
            points.append(f"stance: {p_stance} vs {c_stance}")

        p_dirs = sorted(getattr(primary, "allowed_directions", None) or [])
        c_dirs = sorted(getattr(challenger, "allowed_directions", None) or [])
        if p_dirs != c_dirs:
            points.append(f"allowed_directions: {p_dirs} vs {c_dirs}")

        p_trigs = len(getattr(primary, "triggers", None) or [])
        c_trigs = len(getattr(challenger, "triggers", None) or [])
        if abs(p_trigs - c_trigs) >= 2:
            points.append(f"trigger_count: {p_trigs} vs {c_trigs}")
    except Exception:
        pass
    return points[:4]  # cap for readability


class DeliberationService:
    """Score primary and challenger plans, return a structured DeliberationVerdict."""

    def deliberate(
        self,
        primary: Any,
        challenger: Any,
        llm_input: Any = None,
        judge_constraints: Any = None,
    ) -> DeliberationVerdict:
        """Run both plans through the hallucination scorer and compare.

        Args:
            primary: The primary StrategyPlan.
            challenger: The challenger StrategyPlan (opposing/defensive stance).
            llm_input: Optional LLMInput passed to hallucination scorer for context checks.
            judge_constraints: Optional judge constraints for instruction-inconsistency check.

        Returns:
            DeliberationVerdict with outcome, scores, margin, and divergence points.
        """
        primary_score = _score_plan(primary, llm_input=llm_input, judge_constraints=judge_constraints)
        challenger_score = _score_plan(challenger, llm_input=llm_input, judge_constraints=judge_constraints)
        confidence_margin = primary_score - challenger_score

        if confidence_margin > 0.1:
            outcome = "primary_wins"
        elif confidence_margin < -0.05:
            outcome = "challenger_wins"
        else:
            outcome = "inconclusive"

        divergence_points = _find_divergence_points(primary, challenger)

        logger.info(
            "R92 deliberation: outcome=%s primary_score=%.3f challenger_score=%.3f margin=%+.3f",
            outcome, primary_score, challenger_score, confidence_margin,
        )

        return DeliberationVerdict(
            outcome=outcome,
            confidence_margin=round(confidence_margin, 4),
            primary_score=round(primary_score, 4),
            challenger_score=round(challenger_score, 4),
            divergence_points=divergence_points,
        )
