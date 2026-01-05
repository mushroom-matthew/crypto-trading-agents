"""Helper to query Temporal for workflow status."""

from __future__ import annotations

import logging
import os
from typing import Optional

from temporalio.client import Client

logger = logging.getLogger(__name__)


# Module-level cached client
_TEMPORAL_CLIENT: Optional[Client] = None
_CLIENT_SETTINGS: Optional[tuple[str, str]] = None


async def get_temporal_client() -> Client:
    """Get cached Temporal client for ops-api.

    Connection settings from environment:
    - TEMPORAL_ADDRESS: Temporal server address (default: localhost:7233)
    - TEMPORAL_NAMESPACE: Temporal namespace (default: default)

    Returns:
        Connected Temporal client (cached across calls)
    """
    global _TEMPORAL_CLIENT, _CLIENT_SETTINGS

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    settings = (address, namespace)

    # Return cached client if settings haven't changed
    if _TEMPORAL_CLIENT is not None and _CLIENT_SETTINGS == settings:
        return _TEMPORAL_CLIENT

    # Create new client
    _TEMPORAL_CLIENT = await Client.connect(address, namespace=namespace)
    _CLIENT_SETTINGS = settings

    return _TEMPORAL_CLIENT


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
