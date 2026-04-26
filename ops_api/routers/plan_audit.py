"""R95: Plan audit endpoint — operator transparency surface.

GET /plan-audit/{plan_id}

Returns all available audit signals for a plan:
  - scratchpad (R88): LLM chain-of-thought
  - hallucination_findings (R90): per-section FG-PRM findings
  - field_uncertainty (R93): LLM-stated confidence per field
  - field_logprobs (R97): token-level mean log-probability per field
  - deliberation_verdict (R92): challenger debate outcome (when STRATEGIST_DEBATE=true)
  - best_of_n_meta (R89): candidate count + selection index (when STRATEGIST_BEST_OF_N>1)
  - plan_confidence_score (R95): FG-PRM log-sum aggregate score
  - regime + triggers summary: top-level plan metadata

All data is read from the plan_generated events in the EventStore — no LLM calls.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ops_api.event_store import EventStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plan-audit", tags=["plan-audit"])


def _compute_confidence_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute PlanConfidenceScore from a plan_generated event payload."""
    try:
        from schemas.judge_feedback import PlanHallucinationReport, SectionHallucinationFinding
        from services.plan_hallucination_scorer import PlanHallucinationScorer

        raw_findings = payload.get("hallucination_findings") or []
        findings = []
        for f in raw_findings:
            try:
                findings.append(SectionHallucinationFinding(
                    section_id=f.get("section", "unknown"),
                    hallucination_type=f.get("type", "unknown"),
                    severity=f.get("severity", "REVISE"),
                    detail=f.get("detail", ""),
                ))
            except Exception:
                pass
        report = PlanHallucinationReport(findings=findings)
        score = PlanHallucinationScorer().aggregate_score(
            report,
            field_uncertainty=payload.get("confidence_map"),
            field_logprobs=payload.get("field_logprobs"),
        )
        return score.model_dump()
    except Exception as exc:
        logger.debug("plan-audit: confidence score computation failed: %s", exc)
        return {"aggregate_log_reward": None, "interpretation": "Score unavailable."}


@router.get("/{plan_id}")
async def get_plan_audit(
    plan_id: str,
    run_id: Optional[str] = Query(default=None, description="Session run_id to narrow search"),
    limit: int = Query(default=500, ge=1, le=2000),
) -> Dict[str, Any]:
    """Return all audit signals for the specified plan_id.

    Searches recent plan_generated events for the matching plan.
    Returns 404 when no matching event is found.
    """
    try:
        store = EventStore()
        events = store.query(
            event_type="plan_generated",
            run_id=run_id,
            limit=limit,
            order="desc",
        )
    except Exception as exc:
        logger.error("plan-audit: EventStore query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"EventStore unavailable: {exc}")

    # Find the event with matching plan_id
    matching: Optional[Dict[str, Any]] = None
    for event in events:
        payload = event.payload if hasattr(event, "payload") else (event.get("payload") or {})
        if isinstance(payload, dict) and payload.get("plan_id") == plan_id:
            matching = payload
            break

    if matching is None:
        raise HTTPException(
            status_code=404,
            detail=f"No plan_generated event found for plan_id='{plan_id}' (searched {len(events)} events)",
        )

    # Compute aggregate confidence score
    confidence_score = _compute_confidence_score(matching)

    return {
        "plan_id": plan_id,
        # Plan metadata
        "regime": matching.get("regime"),
        "num_triggers": matching.get("num_triggers"),
        "generated_at": matching.get("generated_at"),
        "valid_until": matching.get("valid_until"),
        "run_id": matching.get("run_id"),
        # R88: scratchpad
        "scratchpad": matching.get("scratchpad_text"),
        # R90: hallucination findings
        "hallucination_findings": matching.get("hallucination_findings") or [],
        # R93: field uncertainty (LLM-stated confidence)
        "field_uncertainty": matching.get("confidence_map"),
        # R97: logprob-based uncertainty
        "field_logprobs": matching.get("field_logprobs"),
        # R92: challenger debate
        "deliberation_verdict": matching.get("deliberation_verdict"),
        # R89: best-of-N selection metadata
        "best_of_n": {
            "candidate_count": matching.get("candidate_count"),
            "selected_candidate_index": matching.get("selected_candidate_index"),
            "candidate_confidence_scores": matching.get("candidate_confidence_scores"),
        } if matching.get("candidate_count") else None,
        # R95: aggregate confidence score
        "plan_confidence_score": confidence_score,
        # Provenance
        "snapshot_id": matching.get("snapshot_id"),
        "prompt_cohort_id": matching.get("prompt_cohort_id"),
        "retrieved_template_id": matching.get("retrieved_template_id"),
    }


@router.get("")
async def list_flagged_plans(
    run_id: Optional[str] = Query(default=None),
    min_findings: int = Query(default=1, ge=1, description="Minimum number of hallucination findings"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> List[Dict[str, Any]]:
    """List recent plans with at least min_findings hallucination findings.

    Useful for quickly surfacing suspicious plans without knowing a plan_id.
    """
    try:
        store = EventStore()
        events = store.query(event_type="plan_generated", run_id=run_id, limit=limit, order="desc")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"EventStore unavailable: {exc}")

    results: List[Dict[str, Any]] = []
    for event in events:
        payload = event.payload if hasattr(event, "payload") else (event.get("payload") or {})
        if not isinstance(payload, dict):
            continue
        findings = payload.get("hallucination_findings") or []
        if len(findings) < min_findings:
            continue
        results.append({
            "plan_id": payload.get("plan_id"),
            "run_id": payload.get("run_id"),
            "generated_at": payload.get("generated_at"),
            "regime": payload.get("regime"),
            "finding_count": len(findings),
            "reject_count": sum(1 for f in findings if f.get("severity") == "REJECT"),
            "revise_count": sum(1 for f in findings if f.get("severity") == "REVISE"),
            "deliberation_outcome": (payload.get("deliberation_verdict") or {}).get("outcome"),
        })

    return results
