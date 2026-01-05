"""WebSocket connection manager for real-time updates."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        # Active connections by channel
        self.live_connections: Set[WebSocket] = set()
        self.market_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect_live(self, websocket: WebSocket):
        """Accept a new live trading WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.live_connections.add(websocket)
        logger.info(f"Live WebSocket connected. Total: {len(self.live_connections)}")

    async def connect_market(self, websocket: WebSocket):
        """Accept a new market data WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.market_connections.add(websocket)
        logger.info(f"Market WebSocket connected. Total: {len(self.market_connections)}")

    async def disconnect_live(self, websocket: WebSocket):
        """Remove a live trading WebSocket connection."""
        async with self._lock:
            self.live_connections.discard(websocket)
        logger.info(f"Live WebSocket disconnected. Total: {len(self.live_connections)}")

    async def disconnect_market(self, websocket: WebSocket):
        """Remove a market data WebSocket connection."""
        async with self._lock:
            self.market_connections.discard(websocket)
        logger.info(f"Market WebSocket disconnected. Total: {len(self.market_connections)}")

    async def broadcast_live(self, message: dict):
        """Broadcast message to all live trading WebSocket clients."""
        if not self.live_connections:
            return

        disconnected = []
        async with self._lock:
            for connection in self.live_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to live client: {e}")
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.live_connections.discard(conn)

    async def broadcast_market(self, message: dict):
        """Broadcast message to all market data WebSocket clients."""
        if not self.market_connections:
            return

        disconnected = []
        async with self._lock:
            for connection in self.market_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to market client: {e}")
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.market_connections.discard(conn)

    def get_stats(self) -> Dict[str, int]:
        """Get connection statistics."""
        return {
            "live_connections": len(self.live_connections),
            "market_connections": len(self.market_connections),
        }


# Global connection manager instance
manager = ConnectionManager()
