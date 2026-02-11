"""Shared helper to emit durable events to the Ops event store."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ops_api.event_store import EventStore
from ops_api.schemas import Event

logger = logging.getLogger(__name__)
_store = EventStore()


async def _broadcast_to_websocket(event: Event) -> None:
    """Broadcast event to appropriate WebSocket channels."""
    # Lazy import to avoid circular dependency
    try:
        from ops_api.websocket_manager import manager as ws_manager
    except ImportError:
        logger.debug("WebSocket manager not available, skipping broadcast")
        return

    # Prepare message
    message = {
        "event_id": event.event_id,
        "timestamp": event.ts.isoformat(),
        "emitted_at": (event.emitted_at or event.ts).isoformat(),
        "source": event.source,
        "type": event.type,
        "payload": event.payload,
        "run_id": event.run_id,
        "correlation_id": event.correlation_id,
    }

    # Route to appropriate WebSocket channel based on event type
    live_event_types = {
        "fill",
        "order_submitted",
        "trade_blocked",
        "position_update",
        "risk_budget_update",
        "intent",
        "plan_generated",
        "plan_judged",
        "judge_action_applied",
        "judge_action_skipped",
    }

    market_event_types = {
        "tick",  # Market tick events
        "market_tick",
        "price_update",
        "symbol_update",
    }

    if event.type in live_event_types:
        await ws_manager.broadcast_live(message)
    elif event.type in market_event_types:
        await ws_manager.broadcast_market(message)
    # Events not in either list are stored but not broadcast


async def emit_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    source: str,
    run_id: str | None = None,
    correlation_id: str | None = None,
    dedupe_key: str | None = None,
    store: EventStore | None = None,
) -> None:
    """Append a durable event asynchronously and broadcast to WebSocket clients."""
    target_store = store or _store
    event = Event(
        event_id=str(uuid4()),
        ts=datetime.now(timezone.utc),
        source=source,
        type=event_type,  # type: ignore[arg-type]
        payload=payload,
        dedupe_key=dedupe_key,
        run_id=run_id,
        correlation_id=correlation_id,
    )
    await asyncio.to_thread(target_store.append, event)

    # Broadcast to WebSocket clients (async, non-blocking)
    try:
        await _broadcast_to_websocket(event)
    except Exception as e:
        logger.warning(f"WebSocket broadcast failed: {e}")


def set_event_store(store: EventStore) -> None:
    """Override the shared event store (useful for tests or custom wiring)."""
    global _store
    _store = store
