"""Minimal Ops API skeleton exposing read-first endpoints."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

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
from ops_api.routers import agents, backtests, live, market, wallets, prompts, paper_trading, regimes, signals as signals_router, screener
from ops_api.websocket_manager import manager as ws_manager

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(
    title="Crypto Trading Agents - Unified Ops API",
    version="0.2.0",
    description="Unified API for backtest control, live trading monitoring, market data, and agent observability"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SAFETY MIDDLEWARE: Block destructive operations in live mode without acknowledgment
@app.middleware("http")
async def live_trading_safety_check(request: Request, call_next):
    """Prevent accidental live trading without explicit LIVE_TRADING_ACK=true."""

    # Define destructive endpoints that modify state or execute trades
    destructive_endpoints = {
        "/backtests",  # Starting backtests can trigger real trades if misconfigured
        "/wallets/reconcile",  # Reconciliation could trigger corrective trades
    }

    # Check if this is a destructive operation
    is_destructive = (
        request.method in ["POST", "PUT", "DELETE"] and
        any(request.url.path.startswith(endpoint) for endpoint in destructive_endpoints)
    )

    if is_destructive:
        runtime = get_runtime_mode()

        if runtime.is_live and not runtime.live_trading_ack:
            logger.error(
                "BLOCKED API CALL: %s %s in live mode without LIVE_TRADING_ACK",
                request.method, request.url.path
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "LIVE_TRADING_NOT_ACKNOWLEDGED",
                    "message": (
                        "Cannot execute destructive operations in live mode without explicit "
                        "LIVE_TRADING_ACK=true environment variable. This endpoint could trigger "
                        "real trades with real money. Set LIVE_TRADING_ACK=true to acknowledge."
                    ),
                    "endpoint": request.url.path,
                    "method": request.method,
                    "runtime_mode": runtime.mode,
                    "live_trading_ack": runtime.live_trading_ack
                }
            )

        # Log all destructive operations for audit trail
        if runtime.is_live:
            logger.critical(
                "LIVE MODE DESTRUCTIVE API CALL: %s %s (LIVE_TRADING_ACK=%s)",
                request.method, request.url.path, runtime.live_trading_ack
            )
        else:
            logger.info(
                "PAPER MODE DESTRUCTIVE API CALL: %s %s",
                request.method, request.url.path
            )

    response = await call_next(request)
    return response


materializer = Materializer()

# Include modular routers
app.include_router(backtests.router)
app.include_router(live.router)
app.include_router(market.router)
app.include_router(agents.router)
app.include_router(wallets.router)
app.include_router(prompts.router)
app.include_router(paper_trading.router)
app.include_router(regimes.router)
app.include_router(signals_router.router)
app.include_router(screener.router)


# Legacy endpoints (for backward compatibility)
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
    return await materializer.list_runs_async()


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
async def llm_telemetry(
    run_id: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    limit: int = Query(default=200, le=500),
) -> List[LLMTelemetry]:
    telemetry = materializer.list_llm(limit=limit)
    if run_id:
        telemetry = [entry for entry in telemetry if entry.run_id == run_id]
    if since:
        telemetry = [entry for entry in telemetry if entry.ts >= since]
    elif not run_id:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        telemetry = [entry for entry in telemetry if entry.ts >= cutoff]
    return telemetry


# WebSocket endpoints for real-time updates
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for live trading updates.

    Streams:
    - New fills
    - Position changes
    - Trade blocks
    - Risk budget updates
    """
    await ws_manager.connect_live(websocket)
    try:
        while True:
            # Keep connection alive and listen for client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await ws_manager.disconnect_live(websocket)
    except Exception as e:
        logger.error(f"Live WebSocket error: {e}")
        await ws_manager.disconnect_live(websocket)


@app.websocket("/ws/market")
async def websocket_market(websocket: WebSocket):
    """
    WebSocket endpoint for market data updates.

    Streams:
    - Market ticks (price updates)
    - Symbol changes
    """
    await ws_manager.connect_market(websocket)
    try:
        while True:
            # Keep connection alive and listen for client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await ws_manager.disconnect_market(websocket)
    except Exception as e:
        logger.error(f"Market WebSocket error: {e}")
        await ws_manager.disconnect_market(websocket)


@app.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return ws_manager.get_stats()


if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    # Use reload=False to avoid asyncio.run() issues with Python 3.13
    uvicorn.run(
        "ops_api.app:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info"
    )
