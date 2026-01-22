"""Validate that event emission is working correctly."""

import asyncio
from datetime import datetime, timezone
from agents.event_emitter import emit_event
from ops_api.event_store import EventStore


async def main():
    """Emit sample events and verify they're stored."""
    print("🧪 Testing event emission...")

    # Emit sample events
    events_to_emit = [
        {
            "event_type": "intent",
            "payload": {"text": "Test user intent"},
            "source": "broker_agent",
            "correlation_id": "test-correlation-1",
        },
        {
            "event_type": "plan_generated",
            "payload": {"strategy_id": "test-strategy", "symbol": "BTC-USD"},
            "source": "strategy_planner",
            "correlation_id": "test-correlation-2",
        },
        {
            "event_type": "trade_blocked",
            "payload": {
                "reason": "risk_budget",
                "trigger_id": "test-trigger",
                "symbol": "ETH-USD",
                "side": "BUY",
                "detail": "Daily risk budget exceeded",
            },
            "source": "execution_agent",
            "run_id": "test-run",
            "correlation_id": "test-correlation-3",
        },
        {
            "event_type": "order_submitted",
            "payload": {
                "symbol": "BTC-USD",
                "side": "BUY",
                "qty": 0.001,
                "price": 42000,
                "type": "market",
                "strategy_id": "test-strategy",
            },
            "source": "execution_agent",
            "run_id": "test-run",
            "correlation_id": "test-correlation-4",
        },
        {
            "event_type": "fill",
            "payload": {
                "symbol": "BTC-USD",
                "side": "BUY",
                "qty": 0.001,
                "fill_price": 42000,
                "cost": 42,
            },
            "source": "execution_agent",
            "run_id": "test-run",
            "correlation_id": "test-correlation-4",  # Same correlation as order_submitted
        },
        {
            "event_type": "plan_judged",
            "payload": {
                "overall_score": 75,
                "component_scores": {"returns": 80, "risk": 70, "quality": 75, "consistency": 75},
                "recommendations": ["Continue with current strategy"],
            },
            "source": "judge_agent",
            "run_id": "judge-agent",
            "correlation_id": "test-correlation-5",
        },
    ]

    print(f"📤 Emitting {len(events_to_emit)} test events...")
    for event_data in events_to_emit:
        await emit_event(**event_data)
        print(f"  ✓ Emitted {event_data['event_type']} event")

    # Give async tasks time to complete
    await asyncio.sleep(0.5)

    # Query the event store
    print("\n📥 Querying event store...")
    store = EventStore()
    events = store.list_events(limit=100)

    print(f"\n✅ Found {len(events)} total events in store")

    # Group by type
    events_by_type = {}
    for event in events:
        events_by_type.setdefault(event.type, []).append(event)

    print("\n📊 Event breakdown:")
    for event_type, event_list in sorted(events_by_type.items()):
        print(f"  {event_type}: {len(event_list)} events")

    # Check for our test events
    print("\n🔍 Test event validation:")
    expected_types = {e["event_type"] for e in events_to_emit}
    found_types = {event.type for event in events if "test" in str(event.payload).lower() or "test" in (event.correlation_id or "")}

    for event_type in expected_types:
        if event_type in {e.type for e in events}:
            print(f"  ✓ {event_type} events present")
        else:
            print(f"  ✗ {event_type} events MISSING")

    # Show sample events
    print("\n📋 Sample events (most recent 3):")
    for event in events[:3]:
        print(f"  • {event.type} from {event.source} at {event.ts}")
        print(f"    correlation_id: {event.correlation_id}")
        print(f"    payload keys: {list(event.payload.keys())}")

    print("\n✅ Event emission validation complete!")
    print(f"📁 Event store location: data/events.sqlite")


if __name__ == "__main__":
    asyncio.run(main())
