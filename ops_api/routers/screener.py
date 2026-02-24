"""Read-only screener endpoints for latest screening result and recommendation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from fastapi import Query

from schemas.screener import (
    InstrumentRecommendation,
    InstrumentRecommendationBatch,
    ScreenerResult,
    ScreenerSessionPreflight,
)
from services.universe_screener_service import ScreenerStateStore, UniverseScreenerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screener", tags=["screener"])


@router.get("/latest", response_model=ScreenerResult)
async def get_latest_screener_result() -> ScreenerResult:
    """Return the most recent screener result."""
    try:
        store = ScreenerStateStore()
        result = store.load_result()
        if result is None:
            raise HTTPException(status_code=404, detail="No screener result available")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load latest screener result: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/recommendation", response_model=InstrumentRecommendation)
async def get_instrument_recommendation() -> InstrumentRecommendation:
    """Return the latest instrument recommendation."""
    try:
        store = ScreenerStateStore()
        recommendation = store.load_recommendation()
        if recommendation is None:
            raise HTTPException(status_code=404, detail="No screener recommendation available")
        return recommendation
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load screener recommendation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/recommendations", response_model=InstrumentRecommendationBatch)
async def get_grouped_instrument_recommendations(
    max_per_group: int = Query(default=10, ge=1, le=10),
    annotate: bool = Query(default=False, description="Apply optional LLM annotation/re-ranking on deterministic shortlist"),
) -> InstrumentRecommendationBatch:
    """Return grouped shortlist recommendations for session-start UX."""
    try:
        store = ScreenerStateStore()
        result = store.load_result()
        if result is None:
            raise HTTPException(status_code=404, detail="No screener result available")
        service = UniverseScreenerService()
        return service.build_recommendation_batch(result, max_per_group=max_per_group, annotate_with_llm=annotate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to build grouped screener recommendations: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/session-preflight", response_model=ScreenerSessionPreflight)
async def get_screener_session_preflight(
    mode: str = Query(default="paper", pattern="^(paper|live)$"),
    annotate: bool = Query(default=False, description="Apply optional LLM annotation/re-ranking on deterministic shortlist"),
) -> ScreenerSessionPreflight:
    """Return session-start shortlist payload for paper/live UX flows."""
    try:
        store = ScreenerStateStore()
        result = store.load_result()
        if result is None:
            raise HTTPException(status_code=404, detail="No screener result available")
        service = UniverseScreenerService()
        return service.build_session_preflight_with_options(result, mode=mode, annotate_with_llm=annotate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to build screener session preflight: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
