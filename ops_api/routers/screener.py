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


@router.post("/run-once")
async def run_screener_once(
    timeframe: str | None = Query(default=None),
    timeframes: str | None = Query(default=None, description="Comma-separated sweep timeframes, e.g. 1m,5m,15m,1h,4h"),
    lookback_bars: int = Query(default=50, ge=30, le=500),
) -> dict:
    """Run a single screener pass immediately and persist latest result/recommendation."""
    try:
        service = UniverseScreenerService()
        sweep = [tf.strip().lower() for tf in (timeframes or "").split(",") if tf.strip()]
        if sweep:
            result = await service.screen_timeframe_sweep(timeframes=sweep, lookback_bars=lookback_bars)
        else:
            tf = str(timeframe or "1h")
            result = await service.screen(timeframe=tf, lookback_bars=lookback_bars)
        recommendation = service.recommend_from_result(result) if result.top_candidates else None
        service.persist_latest(result, recommendation)
        return {
            "status": "ok",
            "run_id": result.run_id,
            "as_of": result.as_of,
            "timeframe": ("sweep" if sweep else str(timeframe or "1h")),
            "timeframes": sweep or None,
            "lookback_bars": lookback_bars,
            "top_candidates": len(result.top_candidates),
            "selected_symbol": recommendation.selected_symbol if recommendation else None,
        }
    except Exception as exc:
        logger.error("Failed to run screener once: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


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
