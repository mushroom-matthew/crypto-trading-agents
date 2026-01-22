"""Market data endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ops_api.event_store import EventStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])


# Response Schemas
class Tick(BaseModel):
    """Market tick data."""

    symbol: str
    price: float
    volume: Optional[float] = None
    timestamp: datetime
    source: str


class Candle(BaseModel):
    """OHLCV candle data."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@router.get("/ticks", response_model=List[Tick])
async def get_ticks(
    symbol: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    limit: int = Query(default=100, le=1000)
):
    """
    Get recent market ticks.

    Returns real-time tick data with optional symbol filtering.
    """
    try:
        # Query tick events from event store
        store = EventStore()
        events = store.list_events(limit=limit)

        # Filter for tick events
        tick_events = [e for e in events if e.type == "tick"]

        # Apply filters
        if symbol:
            tick_events = [e for e in tick_events if e.payload.get("symbol") == symbol]
        if since:
            tick_events = [e for e in tick_events if e.ts >= since]

        return [
            Tick(
                symbol=event.payload.get("symbol", ""),
                price=event.payload.get("price", 0),
                volume=event.payload.get("volume"),
                timestamp=event.ts,
                source=event.source
            )
            for event in tick_events
        ]

    except Exception as e:
        logger.error("Failed to get ticks: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candles", response_model=List[Candle])
async def get_candles(
    symbol: str = Query(..., description="Symbol (e.g., BTC-USD)"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1m, 5m, 15m, 1h, 4h)"),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    limit: int = Query(default=100, le=1000)
):
    """
    Get OHLCV candle data.

    Returns candlestick data for charting. Currently placeholder - requires
    integration with data loader or external market data source.
    """
    try:
        # TODO: Query actual candle data from data loader or market data service
        # This would typically come from:
        # - data_loader/ module (historical data)
        # - Real-time aggregation from ticks
        # - External market data API

        logger.info("Candles requested: %s %s (start=%s, end=%s)", symbol, timeframe, start, end)

        # Placeholder response
        return []

    except Exception as e:
        logger.error("Failed to get candles: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols", response_model=List[str])
async def get_active_symbols():
    """
    Get list of active symbols.

    Returns symbols that have recent tick data.
    """
    try:
        # Query recent ticks and extract unique symbols
        store = EventStore()
        events = store.list_events(limit=500)

        symbols = set()
        for event in events:
            if event.type == "tick":
                symbol = event.payload.get("symbol")
                if symbol:
                    symbols.add(symbol)

        return sorted(list(symbols))

    except Exception as e:
        logger.error("Failed to get active symbols: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
