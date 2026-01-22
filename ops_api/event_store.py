"""Append-only event store abstraction (SQLite-backed)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from datetime import datetime, timezone

from ops_api.schemas import Event, EventType


@dataclass
class EventRecord:
    event_id: str
    ts: datetime
    emitted_at: Optional[datetime]
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
            emitted_at=self.emitted_at,
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
                    emitted_at TEXT,
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
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "emitted_at" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN emitted_at TEXT")
                conn.execute("UPDATE events SET emitted_at = ts WHERE emitted_at IS NULL")
                columns.add("emitted_at")
            if "emitted_at" in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_emitted ON events(emitted_at)")

    def append(self, event: Event) -> None:
        emitted_at = event.emitted_at or datetime.now(timezone.utc)
        if emitted_at.tzinfo is None:
            emitted_at = emitted_at.replace(tzinfo=timezone.utc)
        record = EventRecord(
            event_id=event.event_id,
            ts=event.ts,
            emitted_at=emitted_at,
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
                (event_id, ts, emitted_at, source, type, payload, dedupe_key, run_id, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.ts.isoformat(),
                    record.emitted_at.isoformat() if record.emitted_at else None,
                    record.source,
                    record.type,
                    json.dumps(record.payload),
                    record.dedupe_key,
                    record.run_id,
                    record.correlation_id,
                ),
            )

    def list_events(
        self,
        limit: int = 500,
    ) -> List[Event]:
        return self.list_events_filtered(limit=limit)

    def list_events_filtered(
        self,
        *,
        limit: int = 500,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        since: Optional[datetime] = None,
        order: str = "desc",
        order_by: str = "ts",
    ) -> List[Event]:
        clauses = []
        params: list[object] = []
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if correlation_id:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        if since:
            clauses.append("ts >= ?")
            params.append(since.isoformat())
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = "ASC" if order.lower() == "asc" else "DESC"
        order_key = "ts" if order_by not in {"ts", "emitted_at"} else order_by
        order_expr = "COALESCE(emitted_at, ts)" if order_key == "emitted_at" else "ts"
        params.append(limit)
        query = (
            "SELECT event_id, ts, emitted_at, source, type, payload, dedupe_key, run_id, correlation_id "
            f"FROM events{where_sql} ORDER BY {order_expr} {order_sql} LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        events: List[Event] = []
        for row in rows:
            event_id, ts, emitted_at, source, type_, payload, dedupe_key, run_id_val, corr_id = row
            emitted_at_dt = datetime.fromisoformat(emitted_at) if emitted_at else None
            events.append(
                Event(
                    event_id=event_id,
                    ts=datetime.fromisoformat(ts),
                    emitted_at=emitted_at_dt,
                    source=source,
                    type=type_,  # type: ignore[arg-type]
                    payload=json.loads(payload),
                    dedupe_key=dedupe_key,
                    run_id=run_id_val,
                    correlation_id=corr_id,
                )
            )
        return events
