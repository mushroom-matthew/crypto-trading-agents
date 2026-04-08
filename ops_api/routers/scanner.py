"""Opportunity scanner endpoints.

Runbook 74: OpportunityCard Scorer and Scanner Service.
Runbook 75: Scanner UI Panel.

Endpoints:
  GET /scanner/opportunities           — current ranked OpportunityRanking
  GET /scanner/opportunities/{symbol}  — single OpportunityCard for a symbol
  GET /scanner/history?limit=N         — recent rankings (in-memory store, last 50)
  POST /scanner/run-once               — trigger an immediate scan (non-blocking)

The scanner results are written by PaperTradingWorkflow via the
score_opportunities_activity; this router exposes the most recent result.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from schemas.opportunity import OpportunityCard, OpportunityRanking

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanner", tags=["scanner"])

# ---------------------------------------------------------------------------
# In-memory store for recent rankings
# ---------------------------------------------------------------------------
# Written by score_opportunities_activity via ScannerStateStore.persist()
# and read by the API endpoints.
# Max 50 rankings kept (covers ~12h at 15-min cadence).
_HISTORY_MAX = 50


class ScannerStateStore:
    """Singleton in-memory store for opportunity scanner results."""

    _instance: Optional["ScannerStateStore"] = None

    def __init__(self) -> None:
        self._current: Optional[OpportunityRanking] = None
        self._history: Deque[OpportunityRanking] = deque(maxlen=_HISTORY_MAX)
        self._symbol_index: Dict[str, OpportunityCard] = {}

    @classmethod
    def get(cls) -> "ScannerStateStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def persist(self, ranking: OpportunityRanking) -> None:
        """Store a new ranking as the current result and append to history."""
        self._current = ranking
        self._history.appendleft(ranking)
        self._symbol_index = {card.symbol: card for card in ranking.cards}
        logger.info(
            "scanner: stored ranking with %d cards (universe_size=%d, duration=%dms)",
            len(ranking.cards),
            ranking.universe_size,
            ranking.scan_duration_ms,
        )

    def get_current(self) -> Optional[OpportunityRanking]:
        return self._current

    def get_history(self, limit: int = 20) -> List[OpportunityRanking]:
        return list(self._history)[:limit]

    def get_symbol(self, symbol: str) -> Optional[OpportunityCard]:
        return self._symbol_index.get(symbol)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/opportunities", response_model=Optional[OpportunityRanking])
async def get_opportunities() -> Any:
    """Return the most recent opportunity ranking.

    Returns null (204) when no scan has run yet in this process lifetime.
    Use POST /scanner/run-once to trigger an immediate scan.
    """
    store = ScannerStateStore.get()
    ranking = store.get_current()
    if ranking is None:
        return None
    return ranking


@router.get("/opportunities/{symbol}", response_model=Optional[OpportunityCard])
async def get_symbol_opportunity(symbol: str) -> Any:
    """Return the most recent OpportunityCard for a single symbol.

    Returns 404 when the symbol was not included in the last scan.
    """
    store = ScannerStateStore.get()
    card = store.get_symbol(symbol)
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in the most recent scan. "
            "Ensure the symbol is in the trading universe and a scan has run.",
        )
    return card


@router.get("/history")
async def get_scan_history(
    limit: int = Query(default=20, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Return recent opportunity rankings (most recent first).

    Returns compact summaries (ranked_at, universe_size, top card symbols,
    score range) to avoid large payloads. Use GET /scanner/opportunities
    for the full current ranking.
    """
    store = ScannerStateStore.get()
    history = store.get_history(limit=limit)

    return [
        {
            "ranked_at": r.ranked_at.isoformat(),
            "universe_size": r.universe_size,
            "top_n": r.top_n,
            "scan_duration_ms": r.scan_duration_ms,
            "top_symbols": [c.symbol for c in r.cards[:5]],
            "score_range": {
                "max": max((c.opportunity_score_norm for c in r.cards), default=0.0),
                "min": min((c.opportunity_score_norm for c in r.cards), default=0.0),
            }
            if r.cards
            else None,
        }
        for r in history
    ]


@router.post("/run-once")
async def trigger_scan_once() -> Dict[str, Any]:
    """Trigger an immediate opportunity scan using the current market data.

    This endpoint runs the scan synchronously on the most recently cached
    indicator and structure snapshots. It does NOT connect to the exchange.
    For a full fresh-data scan, start a paper trading session.
    """
    from services.universe_screener_service import ScreenerStateStore, UniverseScreenerService
    from services.opportunity_scanner import rank_universe

    # Pull latest indicator snapshots from the screener state if available
    screener_state = ScreenerStateStore.get_instance()
    latest = screener_state.get_latest() if screener_state else None

    if latest is None:
        return {
            "status": "no_data",
            "message": "No screener data available. Run /screener/run-once first or start a paper trading session.",
        }

    # Build indicator snapshots from screener result
    indicator_snapshots = {}
    for candidate in latest.top_candidates:
        if candidate.indicators:
            indicator_snapshots[candidate.symbol] = candidate.indicators

    if not indicator_snapshots:
        return {
            "status": "no_data",
            "message": "Screener result has no indicator snapshots.",
        }

    symbols = list(indicator_snapshots.keys())
    ranking = rank_universe(
        symbols=symbols,
        indicator_snapshots=indicator_snapshots,
    )

    store = ScannerStateStore.get()
    store.persist(ranking)

    return {
        "status": "ok",
        "ranked_at": ranking.ranked_at.isoformat(),
        "universe_size": ranking.universe_size,
        "top_n": ranking.top_n,
        "scan_duration_ms": ranking.scan_duration_ms,
        "top_symbols": [c.symbol for c in ranking.cards[:10]],
    }
