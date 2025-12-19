"""Append-only event store abstraction (SQLite-backed)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from datetime import datetime

from ops_api.schemas import Event, EventType


@dataclass
class EventRecord:
    event_id: str
    ts: datetime
    source: str
    type: EventType
    payload: dict
    dedupe_key: Optional[str]
    run_id: Optional[str]
    correlation_id: Optional[str]

    def to_event(self) -> Event:
        return Event(
            event_id=self.event_id,
            ts=self.ts,
            source=self.source,
            type=self.type,
            payload=self.payload,
            dedupe_key=self.dedupe_key,
            run_id=self.run_id,
            correlation_id=self.correlation_id,
        )


class EventStore:
    """SQLite-backed append-only log."""

    def __init__(self, path: Path = Path("data/events.sqlite")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    dedupe_key TEXT,
                    run_id TEXT,
                    correlation_id TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")

    def append(self, event: Event) -> None:
        record = EventRecord(
            event_id=event.event_id,
            ts=event.ts,
            source=event.source,
            type=event.type,
            payload=event.payload,
            dedupe_key=event.dedupe_key,
            run_id=event.run_id,
            correlation_id=event.correlation_id,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events
                (event_id, ts, source, type, payload, dedupe_key, run_id, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.ts.isoformat(),
                    record.source,
                    record.type,
                    json.dumps(record.payload),
                    record.dedupe_key,
                    record.run_id,
                    record.correlation_id,
                ),
            )

    def list_events(self, limit: int = 500) -> List[Event]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, ts, source, type, payload, dedupe_key, run_id, correlation_id "
                "FROM events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        events: List[Event] = []
        for row in rows:
            event_id, ts, source, type_, payload, dedupe_key, run_id, correlation_id = row
            events.append(
                Event(
                    event_id=event_id,
                    ts=datetime.fromisoformat(ts),
                    source=source,
                    type=type_,  # type: ignore[arg-type]
                    payload=json.loads(payload),
                    dedupe_key=dedupe_key,
                    run_id=run_id,
                    correlation_id=correlation_id,
                )
            )
        return events
