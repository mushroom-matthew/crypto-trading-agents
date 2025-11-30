import asyncio
import os
from temporalio.testing import WorkflowEnvironment
from fastapi.testclient import TestClient

from mcp_server.app import app
from worker.main import main as worker_main
from agents import temporal_utils
import pytest


@pytest.mark.asyncio
async def test_start_market_stream():
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
            resp = client.post(
                "/tools/start_market_stream",
                json={
                    "symbols": ["BTC/USD"],
                    "interval_sec": 0.1,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            workflow_id = data["workflow_id"]
            run_id = data["run_id"]
            status_resp = client.get(f"/workflow/{workflow_id}/{run_id}")
            assert status_resp.status_code == 200
            payload = status_resp.json()
            assert payload["status"] in {"RUNNING", "COMPLETED"}
        finally:
            temporal_utils._temporal_client = None
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
