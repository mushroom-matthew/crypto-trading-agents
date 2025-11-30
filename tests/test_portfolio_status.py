import asyncio
import os

import pytest
from temporalio.testing import WorkflowEnvironment
from fastapi.testclient import TestClient

from mcp_server.app import app
from worker.main import main as worker_main
from agents import temporal_utils


@pytest.mark.asyncio
async def test_get_portfolio_status_includes_pnl():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        config_obj = env.client.config()
        service_client = config_obj.get("service_client")
        target_host = getattr(getattr(service_client, "config", None), "target_host", None)
        os.environ["TEMPORAL_ADDRESS"] = target_host or "localhost:7233"
        os.environ["TEMPORAL_NAMESPACE"] = env.client.namespace
        temporal_utils._temporal_client = env.client
        worker_task = asyncio.create_task(worker_main())
        client = TestClient(app.sse_app())
        try:
            resp = client.post("/tools/get_portfolio_status", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert "pnl" in data
        finally:
            temporal_utils._temporal_client = None
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
