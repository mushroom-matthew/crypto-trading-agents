"""Ops API router for research budget status and playbook validation (Runbook 48)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")
PLAYBOOK_DIR = Path("vector_store/playbooks")


# ---------------------------------------------------------------------------
# In-memory store for pending playbook edit suggestions (session-scoped).
# A production implementation would persist to a DB or event store.
# ---------------------------------------------------------------------------
_pending_edit_suggestions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class PlaybookEditSuggestionResponse(BaseModel):
    suggestion_id: str
    playbook_id: str
    section: str
    suggested_text: str
    evidence_summary: str
    requires_human_review: bool = True
    status: str = "pending"


# ---------------------------------------------------------------------------
# Helper: query paper trading workflow for session state
# ---------------------------------------------------------------------------

async def _get_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Query the PaperTradingWorkflow for research state."""
    try:
        from temporalio.client import Client

        address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
        namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
        client = await Client.connect(address, namespace=namespace)
        handle = client.get_workflow_handle(f"paper-trading-{session_id}")
        state = await handle.query("get_session_status")
        return state
    except Exception as exc:
        logger.warning("Could not query session %s: %s", session_id, exc)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/budget")
async def get_research_budget(session_id: str) -> Dict[str, Any]:
    """Return the current research budget state for a session."""
    state = await _get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or not reachable")

    research = state.get("research")
    if research is None:
        return {
            "session_id": session_id,
            "research_enabled": False,
            "message": "Research budget not initialized for this session",
        }

    return {
        "session_id": session_id,
        "research_enabled": True,
        "initial_capital": research.get("initial_capital"),
        "cash": research.get("cash"),
        "total_pnl": research.get("total_pnl"),
        "paused": research.get("paused"),
        "pause_reason": research.get("pause_reason"),
        "active_experiment_id": research.get("active_experiment_id"),
        "active_playbook_id": research.get("active_playbook_id"),
        "n_trades": len(research.get("trades", [])),
    }


@router.get("/experiments")
async def list_experiments(session_id: str) -> List[Dict[str, Any]]:
    """Return all ExperimentSpecs for a session with their current status."""
    state = await _get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or not reachable")

    return list(state.get("active_experiments") or [])


@router.get("/playbooks/{playbook_id}/validation")
async def get_playbook_validation(playbook_id: str) -> Dict[str, Any]:
    """Return current validation evidence from the playbook .md frontmatter."""
    path = PLAYBOOK_DIR / f"{playbook_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Playbook '{playbook_id}' not found")

    import re
    text = path.read_text(encoding="utf-8")

    # Parse the ## Validation Evidence section
    match = re.search(r"## Validation Evidence\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if not match:
        return {"playbook_id": playbook_id, "status": "no_evidence_section"}

    evidence_text = match.group(1)
    evidence: Dict[str, Any] = {"playbook_id": playbook_id}
    for line in evidence_text.splitlines():
        line = line.strip()
        if line.startswith("<!--") or not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        evidence[key.strip()] = None if value == "null" else value

    return evidence


@router.get("/playbooks/edit-suggestions")
async def get_pending_edit_suggestions() -> List[PlaybookEditSuggestionResponse]:
    """Return pending judge-suggested playbook edits awaiting human review."""
    return [
        PlaybookEditSuggestionResponse(**s)
        for s in _pending_edit_suggestions.values()
        if s.get("status") == "pending"
    ]


@router.post("/playbooks/edit-suggestions")
async def submit_edit_suggestion(suggestion: PlaybookEditSuggestionResponse) -> Dict[str, Any]:
    """Internal endpoint: judge submits a new edit suggestion."""
    suggestion_id = str(uuid4())
    record = suggestion.model_dump()
    record["suggestion_id"] = suggestion_id
    record["status"] = "pending"
    _pending_edit_suggestions[suggestion_id] = record
    logger.info("New playbook edit suggestion queued: %s for %s", suggestion_id, suggestion.playbook_id)
    return {"suggestion_id": suggestion_id, "status": "queued"}


@router.post("/playbooks/{playbook_id}/apply-suggestion")
async def apply_edit_suggestion(playbook_id: str, suggestion_id: str) -> Dict[str, Any]:
    """Human approves a playbook edit suggestion — writes to .md file."""
    record = _pending_edit_suggestions.get(suggestion_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Suggestion '{suggestion_id}' not found")
    if record.get("playbook_id") != playbook_id:
        raise HTTPException(status_code=400, detail="suggestion_id does not match playbook_id")
    if record.get("status") != "pending":
        raise HTTPException(status_code=409, detail=f"Suggestion already {record.get('status')}")

    path = PLAYBOOK_DIR / f"{playbook_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Playbook file not found: {playbook_id}.md")

    section = record.get("section", "Notes")
    suggested_text = record.get("suggested_text", "")

    import re
    content = path.read_text(encoding="utf-8")

    # Replace the target section content
    section_pattern = rf"(## {re.escape(section)}\n)(.*?)(?=\n##|\Z)"
    replacement = rf"\g<1>{suggested_text}\n"
    new_content, count = re.subn(section_pattern, replacement, content, flags=re.DOTALL)

    if count == 0:
        # Section does not exist — append it
        new_content = content.rstrip() + f"\n\n## {section}\n{suggested_text}\n"

    path.write_text(new_content, encoding="utf-8")
    record["status"] = "applied"
    _pending_edit_suggestions[suggestion_id] = record

    logger.info("Applied edit suggestion %s to %s § %s", suggestion_id, playbook_id, section)
    return {"suggestion_id": suggestion_id, "playbook_id": playbook_id, "status": "applied"}
