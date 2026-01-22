"""FastAPI server providing a web dashboard for supervising agents."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.dashboard.service import LedgerSummaryService, ManagedProcess, ProcessSupervisor
from app.db.repo import Database


LOG = get_logger(__name__)

workspace = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str((Path(__file__).parent / "templates").resolve()))

settings = get_settings()
setup_logging(settings.log_level)

database = Database(settings)
ledger_service = LedgerSummaryService(database)

base_env = os.environ.copy()
base_env.setdefault("PYTHONPATH", str(workspace))
python_bin = str(workspace / ".venv" / "bin" / "python")

SUPERVISOR_ENABLED = os.environ.get("DASHBOARD_ENABLE_SUPERVISOR", "false").lower() in {
    "1",
    "true",
    "yes",
}

supervisor = ProcessSupervisor(base_env=base_env, workspace=workspace)

if SUPERVISOR_ENABLED:
    supervisor.register(
        ManagedProcess(
            name="temporal",
            command=["temporal", "server", "start-dev"],
            env={},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="worker",
            command=[python_bin, "worker/main.py"],
            env={},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="mcp-server",
            command=[python_bin, "mcp_server/app.py"],
            env={"PYTHONPATH": str(workspace)},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="broker-agent",
            command=[python_bin, "agents/broker_agent_client.py"],
            env={"PYTHONPATH": str(workspace)},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="execution-agent",
            command=[python_bin, "agents/execution_agent_client.py"],
            env={"PYTHONPATH": str(workspace)},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="judge-agent",
            command=[python_bin, "agents/judge_agent_client.py"],
            env={"PYTHONPATH": str(workspace)},
            cwd=workspace,
        )
    )
    supervisor.register(
        ManagedProcess(
            name="ticker-ui",
            command=[python_bin, "ticker_ui_service.py"],
            env={"PYTHONPATH": str(workspace)},
            cwd=workspace,
        )
    )
else:
    LOG.info(
        "Local process supervision disabled; set DASHBOARD_ENABLE_SUPERVISOR=1 to enable start/stop controls."
    )

backlog_items = [
    "Add automated Postgres/Temporal health checks and alerts.",
    "Automate Coinbase portfolio creation/funding workflows.",
    "Implement agent self-healing (auto-restart on crash) policies.",
    "Integrate historical backtesting orchestration into dashboard.",
    "Add alerting when ledger reconciliation drifts beyond threshold.",
]

app = FastAPI(title="Crypto Trading Dashboard", version="0.1.0")


def _ensure_process(name: str) -> None:
    if name not in supervisor._processes:
        raise HTTPException(status_code=404, detail=f"Unknown process {name}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    processes = supervisor.all_status()
    try:
        wallets = await ledger_service.wallet_overview()
        db_error: Optional[str] = None
    except Exception as exc:  # pragma: no cover - depends on runtime DB availability
        LOG.warning("Failed to read wallet overview", error=str(exc))
        wallets = []
        db_error = str(exc)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "processes": processes,
            "wallets": wallets,
            "db_error": db_error,
            "backlog": backlog_items,
            "supervisor_enabled": SUPERVISOR_ENABLED,
        },
    )


@app.get("/api/processes")
async def list_processes() -> JSONResponse:
    return JSONResponse(supervisor.all_status())


@app.post("/api/processes/{name}/start")
async def start_process(name: str) -> JSONResponse:
    _ensure_process(name)
    await supervisor.start(name)
    return JSONResponse({"status": "started", "process": name})


@app.post("/api/processes/{name}/stop")
async def stop_process(name: str) -> JSONResponse:
    _ensure_process(name)
    await supervisor.stop(name)
    return JSONResponse({"status": "stopped", "process": name})


@app.get("/api/wallets")
async def list_wallets() -> JSONResponse:
    try:
        wallets = await ledger_service.wallet_overview()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(wallets)


@app.on_event("shutdown")
async def shutdown() -> None:
    await database.dispose()
    # stop all running processes
    tasks = []
    for proc_name, proc in list(supervisor._processes.items()):
        if proc.process and proc.process.returncode is None:
            tasks.append(supervisor.stop(proc_name))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
