from datetime import datetime, timezone

from starlette.testclient import TestClient

from agents import event_emitter
from mcp_server import app as mcp_app
from ops_api.event_store import EventStore


def test_signal_roundtrip(monkeypatch, tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    mcp_app.configure_event_store(store)
    event_emitter.set_event_store(store)

    client = TestClient(mcp_app.app)

    ts = int(datetime.now(timezone.utc).timestamp())
    payload = {"symbol": "BTC/USD", "price": 12345.0, "ts": ts}

    resp = client.post("/signal/market_tick", json=payload)
    assert resp.status_code == 204

    resp = client.get("/signal/market_tick", params={"after": ts - 1})
    assert resp.status_code == 200
    events = resp.json()

    assert any(evt.get("payload", {}).get("symbol") == "BTC/USD" for evt in events)
