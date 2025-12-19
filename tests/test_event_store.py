from pathlib import Path
from datetime import datetime, timezone

from ops_api.event_store import EventStore
from ops_api.schemas import Event


def test_event_store_roundtrip(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite")
    event = Event(
        event_id="e1",
        ts=datetime.now(timezone.utc),
        source="test",
        type="tick",
        payload={"foo": "bar"},
        dedupe_key=None,
        run_id=None,
        correlation_id=None,
    )
    store.append(event)
    events = store.list_events()
    assert any(e.event_id == "e1" for e in events)
