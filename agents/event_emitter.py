"""Shared helper to emit durable events to the Ops event store."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ops_api.event_store import EventStore
from ops_api.schemas import Event

_store = EventStore()


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
    """Append a durable event asynchronously."""
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


def set_event_store(store: EventStore) -> None:
    """Override the shared event store (useful for tests or custom wiring)."""
    global _store
    _store = store
