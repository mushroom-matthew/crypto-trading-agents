"""Minimal Ops API skeleton exposing read-first endpoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents.runtime_mode import get_runtime_mode
from ops_api.schemas import (
    BlockReasonsAggregate,
    Event,
    FillRecord,
    LLMTelemetry,
    PositionSnapshot,
    RunSummary,
)
from ops_api.materializer import Materializer

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(title="ops-api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
materializer = Materializer()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.get("/status", response_model=dict)
async def status() -> dict:
    runtime = get_runtime_mode()
    return {
        "stack": runtime.stack,
        "mode": runtime.mode,
        "live_trading_ack": runtime.live_trading_ack,
        "ui_unlock": runtime.ui_unlock,
        "banner": runtime.banner,
        "ts": datetime.utcnow().isoformat(),
    }


@app.get("/workflows", response_model=List[RunSummary])
async def list_runs() -> List[RunSummary]:
    return materializer.list_runs()


@app.get("/block_reasons", response_model=BlockReasonsAggregate)
async def block_reasons() -> BlockReasonsAggregate:
    return materializer.block_reasons()


@app.get("/fills", response_model=List[FillRecord])
async def list_fills() -> List[FillRecord]:
    return materializer.list_fills()


@app.get("/positions", response_model=List[PositionSnapshot])
async def positions() -> List[PositionSnapshot]:
    return materializer.list_positions()


@app.get("/events", response_model=List[Event])
async def list_events() -> List[Event]:
    return materializer.list_events()


@app.get("/llm/telemetry", response_model=List[LLMTelemetry])
async def llm_telemetry() -> List[LLMTelemetry]:
    return materializer.list_llm()


if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
