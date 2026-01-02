"""Helper to query Temporal for workflow status."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def get_workflow_status(workflow_id: str, temporal_address: str = "localhost:7233") -> dict:
    """
    Query Temporal for workflow status.

    Returns:
        dict with keys: status, run_id, type
        status can be: running, completed, failed, terminated, timed_out, canceled, unknown
    """
    try:
        from temporalio.client import Client

        client = await Client.connect(temporal_address)
        handle = client.get_workflow_handle(workflow_id)

        desc = await handle.describe()
        return {
            "status": desc.status.name.lower(),  # e.g., "running", "completed"
            "run_id": desc.run_id,
            "type": desc.workflow_type,
        }
    except Exception as e:
        logger.warning("Failed to query Temporal for workflow %s: %s", workflow_id, e)
        return {"status": "unknown", "error": str(e)}


def get_runtime_mode() -> str:
    """
    Get the actual runtime mode from environment/config.

    Returns:
        "dev", "paper", or "live"
    """
    try:
        from agents.runtime_mode import get_runtime_mode as get_mode

        runtime = get_mode()
        return runtime.mode
    except Exception as e:
        logger.warning("Failed to get runtime mode: %s", e)
        return "paper"  # Safe default
