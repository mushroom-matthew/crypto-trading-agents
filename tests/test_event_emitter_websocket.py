"""Integration tests for event emitter WebSocket routing."""

import sys
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from agents.event_emitter import emit_event, _broadcast_to_websocket
from ops_api.schemas import Event


@pytest.fixture
def mock_ws_manager():
    """Create a mock WebSocket manager that can be imported."""
    manager = Mock()
    manager.broadcast_live = AsyncMock()
    manager.broadcast_market = AsyncMock()

    # Create mock module
    mock_module = Mock()
    mock_module.manager = manager

    return mock_module, manager


@pytest.mark.asyncio
async def test_broadcast_fill_event_to_live_channel(mock_ws_manager):
    """Test that fill events are broadcast to /ws/live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-fill-1",
        ts=datetime.now(timezone.utc),
        source="execution_agent",
        type="fill",
        payload={"symbol": "BTC-USD", "qty": 0.01, "price": 50000, "side": "buy"},
        run_id="test-run-001",
        correlation_id="test-corr-001"
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        # Verify broadcast_live was called
        manager.broadcast_live.assert_called_once()
        call_args = manager.broadcast_live.call_args[0][0]

        assert call_args["event_id"] == "test-fill-1"
        assert call_args["type"] == "fill"
        assert call_args["source"] == "execution_agent"
        assert call_args["payload"]["symbol"] == "BTC-USD"
        assert call_args["run_id"] == "test-run-001"

        # Verify broadcast_market was NOT called
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_tick_event_to_market_channel(mock_ws_manager):
    """Test that market tick events are broadcast to /ws/market channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-tick-1",
        ts=datetime.now(timezone.utc),
        source="market_stream",
        type="tick",
        payload={"symbol": "ETH-USD", "price": 3000, "volume": 1.5},
        run_id=None,
        correlation_id=None
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        # Verify broadcast_market was called
        manager.broadcast_market.assert_called_once()
        call_args = manager.broadcast_market.call_args[0][0]

        assert call_args["event_id"] == "test-tick-1"
        assert call_args["type"] == "tick"
        assert call_args["source"] == "market_stream"
        assert call_args["payload"]["symbol"] == "ETH-USD"

        # Verify broadcast_live was NOT called
        manager.broadcast_live.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_order_submitted_to_live(mock_ws_manager):
    """Test that order_submitted events go to live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-order-1",
        ts=datetime.now(timezone.utc),
        source="execution_agent",
        type="order_submitted",
        payload={"symbol": "BTC-USD", "side": "buy", "qty": 0.01},
        run_id="test-run-002",
        correlation_id="test-corr-002"
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        manager.broadcast_live.assert_called_once()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_trade_blocked_to_live(mock_ws_manager):
    """Test that trade_blocked events go to live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-block-1",
        ts=datetime.now(timezone.utc),
        source="execution_agent",
        type="trade_blocked",
        payload={"symbol": "ETH-USD", "reason": "insufficient_budget"},
        run_id="test-run-003",
        correlation_id="test-corr-003"
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        manager.broadcast_live.assert_called_once()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_position_update_to_live(mock_ws_manager):
    """Test that position_update events go to live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-pos-1",
        ts=datetime.now(timezone.utc),
        source="execution_ledger",
        type="position_update",
        payload={"symbol": "BTC-USD", "qty": 0.05, "pnl": 250.50},
        run_id="test-run-004",
        correlation_id=None
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        manager.broadcast_live.assert_called_once()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_intent_to_live(mock_ws_manager):
    """Test that intent events go to live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-intent-1",
        ts=datetime.now(timezone.utc),
        source="broker_agent",
        type="intent",
        payload={"text": "User requested to trade BTC"},
        run_id="broker-run-001",
        correlation_id="test-corr-005"
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        manager.broadcast_live.assert_called_once()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_plan_judged_to_live(mock_ws_manager):
    """Test that plan_judged events go to live channel."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-judge-1",
        ts=datetime.now(timezone.utc),
        source="judge_agent",
        type="plan_judged",
        payload={"overall_score": 75, "plan_id": "plan-123"},
        run_id="judge-run-001",
        correlation_id="test-corr-006"
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        manager.broadcast_live.assert_called_once()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_llm_call_event_not_broadcast(mock_ws_manager):
    """Test that llm_call events are not broadcast to any channel (not in routing lists)."""
    mock_module, manager = mock_ws_manager

    event = Event(
        event_id="test-llm-1",
        ts=datetime.now(timezone.utc),
        source="execution_agent",
        type="llm_call",
        payload={"model": "gpt-4", "tokens": 100},
        run_id=None,
        correlation_id=None
    )

    with patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):
        await _broadcast_to_websocket(event)

        # Neither channel should be called for llm_call type (not in routing lists)
        manager.broadcast_live.assert_not_called()
        manager.broadcast_market.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_handles_import_error():
    """Test that broadcast gracefully handles WebSocket manager import failure."""
    event = Event(
        event_id="test-1",
        ts=datetime.now(timezone.utc),
        source="test",
        type="fill",
        payload={},
        run_id=None,
        correlation_id=None
    )

    # Remove the module if it exists to simulate ImportError
    if "ops_api.websocket_manager" in sys.modules:
        original = sys.modules["ops_api.websocket_manager"]
    else:
        original = None

    try:
        sys.modules["ops_api.websocket_manager"] = None  # Will cause ImportError

        # Should not raise exception
        await _broadcast_to_websocket(event)

    finally:
        # Restore
        if original is not None:
            sys.modules["ops_api.websocket_manager"] = original
        elif "ops_api.websocket_manager" in sys.modules:
            del sys.modules["ops_api.websocket_manager"]


@pytest.mark.asyncio
async def test_emit_event_handles_websocket_error(mock_ws_manager):
    """Test that emit_event catches WebSocket errors gracefully."""
    mock_module, manager = mock_ws_manager

    # Make broadcast_live raise an exception
    manager.broadcast_live = AsyncMock(side_effect=Exception("WebSocket error"))

    # Mock the event store
    with patch("agents.event_emitter._store") as mock_store, \
         patch.dict(sys.modules, {"ops_api.websocket_manager": mock_module}):

        # Should not raise exception - emit_event catches WebSocket errors
        await emit_event(
            event_type="fill",
            payload={"symbol": "BTC-USD"},
            source="execution_agent",
            run_id="test-run-001"
        )

        # Event store should still have been called
        assert mock_store.append.called
