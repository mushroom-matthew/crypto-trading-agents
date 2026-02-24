"""Ops API router for strategy analytics and gate validation (Runbook 47).

Key endpoint: GET /analytics/template-routing
Reports template routing statistics from plan_generated events so operators can
assess whether R46 routing accuracy meets the ≥ 80% gate required before R47
enforcement is considered "validated."

Gate definition (R46 → R47):
  "R46 routing confirmed correct ≥ 80% of plan-generation days"
  Operationally: at least 80% of plan_generated events where the market had
  compression indicators active should have retrieved_template_id="compression_breakout".

This endpoint is intentionally simple — it surfaces the raw stats so a human
operator can make the gate call.  It does not auto-pass or auto-block the gate.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from ops_api.event_store import EventStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


def _template_routing_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute routing statistics from a list of plan_generated event payloads."""
    total = len(events)
    if total == 0:
        return {
            "total_plans": 0,
            "plans_with_retrieved_template": 0,
            "plans_with_declared_template": 0,
            "retrieved_template_pct": 0.0,
            "declared_template_pct": 0.0,
            "retrieved_template_distribution": {},
            "declared_template_distribution": {},
            "gate_r46_note": "No plan_generated events found — start a paper trading session.",
        }

    retrieved = [
        e.get("retrieved_template_id")
        for e in events
        if e.get("retrieved_template_id")
    ]
    declared = [
        e.get("llm_meta", {}).get("template_id") or e.get("template_id")
        for e in events
        if (e.get("llm_meta", {}) or {}).get("template_id")
        or e.get("template_id")
    ]

    retrieved_pct = round(len(retrieved) / total * 100, 1)
    declared_pct = round(len(declared) / total * 100, 1)

    return {
        "total_plans": total,
        "plans_with_retrieved_template": len(retrieved),
        "plans_with_declared_template": len(declared),
        "retrieved_template_pct": retrieved_pct,
        "declared_template_pct": declared_pct,
        "retrieved_template_distribution": dict(Counter(retrieved)),
        "declared_template_distribution": dict(Counter(d for d in declared if d)),
        "gate_r46_note": (
            f"R46 gate: ≥80% of plans should have retrieved_template_id set. "
            f"Current: {retrieved_pct}% ({len(retrieved)}/{total}). "
            + ("GATE MET ✓" if retrieved_pct >= 80.0 else "GATE NOT YET MET — continue paper trading.")
        ),
        "gate_r47_note": (
            f"R47 gate: LLM should be declaring template_id in ≥80% of plans. "
            f"Current: {declared_pct}% ({len(declared)}/{total}). "
            + ("GATE MET ✓" if declared_pct >= 80.0 else "GATE NOT YET MET — enforcement accuracy pending.")
        ),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/template-routing")
async def get_template_routing_stats(
    run_id: Optional[str] = Query(default=None, description="Filter to a specific backtest/paper trading run_id"),
    limit: int = Query(default=500, ge=1, le=5000, description="Max plan_generated events to inspect"),
) -> Dict[str, Any]:
    """Return template routing statistics for R46/R47 gate validation.

    Queries the event store for recent `plan_generated` events and computes:
    - How often retrieval returned a template_id (R46 routing rate)
    - How often the LLM declared a template_id in its plan (R47 compliance rate)
    - Distribution of which templates are being selected

    Use this to assess whether the R46 ≥ 80% accuracy gate is met before
    relying on R47's hard enforcement in a live trading context.
    """
    try:
        store = EventStore()
        all_events = store.list_events_filtered(
            event_type="plan_generated", run_id=run_id, limit=limit, order="asc"
        )
        payloads = [e.payload for e in all_events if e.payload]
        stats = _template_routing_stats(payloads)
        return {"status": "ok", "stats": stats}
    except Exception as exc:
        logger.warning("Failed to compute template routing stats: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "stats": {},
        }
