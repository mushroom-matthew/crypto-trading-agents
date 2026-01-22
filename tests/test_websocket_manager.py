"""Unit tests for WebSocket connection manager and broadcasting."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import WebSocket

from ops_api.websocket_manager import ConnectionManager


@pytest.fixture
def manager():
    """Create a fresh ConnectionManager for each test."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = Mock(spec=WebSocket)
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.accept = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_live(manager, mock_websocket):
    """Test connecting a live trading WebSocket client."""
    await manager.connect_live(mock_websocket)

    assert len(manager.live_connections) == 1
    assert mock_websocket in manager.live_connections
    assert len(manager.market_connections) == 0


@pytest.mark.asyncio
async def test_connect_market(manager, mock_websocket):
    """Test connecting a market data WebSocket client."""
    await manager.connect_market(mock_websocket)

    assert len(manager.market_connections) == 1
    assert mock_websocket in manager.market_connections
    assert len(manager.live_connections) == 0


@pytest.mark.asyncio
async def test_disconnect_live(manager, mock_websocket):
    """Test disconnecting a live trading WebSocket client."""
    await manager.connect_live(mock_websocket)
    assert len(manager.live_connections) == 1

    await manager.disconnect_live(mock_websocket)
    assert len(manager.live_connections) == 0


@pytest.mark.asyncio
async def test_disconnect_market(manager, mock_websocket):
    """Test disconnecting a market data WebSocket client."""
    await manager.connect_market(mock_websocket)
    assert len(manager.market_connections) == 1

    await manager.disconnect_market(mock_websocket)
    assert len(manager.market_connections) == 0


@pytest.mark.asyncio
async def test_broadcast_live_single_client(manager, mock_websocket):
    """Test broadcasting to a single live client."""
    await manager.connect_live(mock_websocket)

    message = {"type": "fill", "payload": {"symbol": "BTC-USD", "qty": 0.01}}
    await manager.broadcast_live(message)

    mock_websocket.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_market_single_client(manager, mock_websocket):
    """Test broadcasting to a single market client."""
    await manager.connect_market(mock_websocket)

    message = {"type": "tick", "payload": {"symbol": "BTC-USD", "price": 50000}}
    await manager.broadcast_market(message)

    mock_websocket.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_live_multiple_clients(manager):
    """Test broadcasting to multiple live clients."""
    # Create multiple mock clients
    clients = [Mock(spec=WebSocket) for _ in range(3)]
    for client in clients:
        client.send_json = AsyncMock()

    # Connect all clients
    for client in clients:
        await manager.connect_live(client)

    assert len(manager.live_connections) == 3

    # Broadcast message
    message = {"type": "position_update", "payload": {"symbol": "ETH-USD"}}
    await manager.broadcast_live(message)

    # Verify all clients received the message
    for client in clients:
        client.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_market_multiple_clients(manager):
    """Test broadcasting to multiple market clients."""
    # Create multiple mock clients
    clients = [Mock(spec=WebSocket) for _ in range(3)]
    for client in clients:
        client.send_json = AsyncMock()

    # Connect all clients
    for client in clients:
        await manager.connect_market(client)

    assert len(manager.market_connections) == 3

    # Broadcast message
    message = {"type": "tick", "payload": {"symbol": "BTC-USD", "price": 49000}}
    await manager.broadcast_market(message)

    # Verify all clients received the message
    for client in clients:
        client.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_live_no_clients(manager):
    """Test broadcasting when no clients are connected (should not raise)."""
    message = {"type": "fill", "payload": {}}
    await manager.broadcast_live(message)  # Should complete without error
    assert len(manager.live_connections) == 0


@pytest.mark.asyncio
async def test_broadcast_market_no_clients(manager):
    """Test broadcasting when no clients are connected (should not raise)."""
    message = {"type": "tick", "payload": {}}
    await manager.broadcast_market(message)  # Should complete without error
    assert len(manager.market_connections) == 0


@pytest.mark.asyncio
async def test_broadcast_removes_failed_connections(manager):
    """Test that failed connections are automatically removed during broadcast."""
    # Create clients where one will fail
    good_client = Mock(spec=WebSocket)
    good_client.send_json = AsyncMock()

    bad_client = Mock(spec=WebSocket)
    bad_client.send_json = AsyncMock(side_effect=Exception("Connection broken"))

    # Connect both
    await manager.connect_live(good_client)
    await manager.connect_live(bad_client)
    assert len(manager.live_connections) == 2

    # Broadcast (should remove bad client)
    message = {"type": "fill", "payload": {}}
    await manager.broadcast_live(message)

    # Verify good client received message
    good_client.send_json.assert_called_once_with(message)

    # Verify bad client was removed
    assert len(manager.live_connections) == 1
    assert bad_client not in manager.live_connections
    assert good_client in manager.live_connections


@pytest.mark.asyncio
async def test_get_stats(manager, mock_websocket):
    """Test retrieving connection statistics."""
    # Initially no connections
    stats = manager.get_stats()
    assert stats["live_connections"] == 0
    assert stats["market_connections"] == 0

    # Connect live client
    await manager.connect_live(mock_websocket)
    stats = manager.get_stats()
    assert stats["live_connections"] == 1
    assert stats["market_connections"] == 0

    # Connect market client
    market_ws = Mock(spec=WebSocket)
    market_ws.send_json = AsyncMock()
    await manager.connect_market(market_ws)
    stats = manager.get_stats()
    assert stats["live_connections"] == 1
    assert stats["market_connections"] == 1


@pytest.mark.asyncio
async def test_concurrent_broadcasts(manager):
    """Test concurrent broadcasts to multiple channels."""
    import asyncio

    # Create clients for both channels
    live_client = Mock(spec=WebSocket)
    live_client.send_json = AsyncMock()

    market_client = Mock(spec=WebSocket)
    market_client.send_json = AsyncMock()

    await manager.connect_live(live_client)
    await manager.connect_market(market_client)

    # Broadcast to both channels concurrently
    live_msg = {"type": "fill", "payload": {"symbol": "BTC-USD"}}
    market_msg = {"type": "tick", "payload": {"symbol": "ETH-USD"}}

    await asyncio.gather(
        manager.broadcast_live(live_msg),
        manager.broadcast_market(market_msg)
    )

    # Verify messages were sent to correct channels
    live_client.send_json.assert_called_once_with(live_msg)
    market_client.send_json.assert_called_once_with(market_msg)


@pytest.mark.asyncio
async def test_disconnect_nonexistent_connection(manager, mock_websocket):
    """Test disconnecting a connection that was never connected (should not raise)."""
    # Should not raise exception
    await manager.disconnect_live(mock_websocket)
    await manager.disconnect_market(mock_websocket)

    assert len(manager.live_connections) == 0
    assert len(manager.market_connections) == 0
