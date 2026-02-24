"""Temporal activities/workflow for periodic universe screening."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

from schemas.screener import ScreenerResult


@activity.defn
async def run_universe_screen(config: dict[str, Any]) -> dict[str, Any]:
    """Run one universe screening pass and build a recommendation."""
    from services.universe_screener_service import UniverseScreenerService

    service = UniverseScreenerService(universe=config.get("universe"))
    timeframe = str(config.get("timeframe") or "1h")
    lookback_bars = int(config.get("lookback_bars", 50))
    result = await service.screen(timeframe=timeframe, lookback_bars=lookback_bars)
    recommendation = service.recommend_from_result(result, timeframe=timeframe) if result.top_candidates else None
    return {
        "screener_result": result.model_dump(mode="json"),
        "instrument_recommendation": recommendation.model_dump(mode="json") if recommendation else None,
    }


@activity.defn
async def emit_screener_result(payload: dict[str, Any]) -> None:
    """Persist latest screener result/recommendation for downstream consumers."""
    from services.universe_screener_service import UniverseScreenerService

    service = UniverseScreenerService()
    result_payload = payload.get("screener_result")
    if not result_payload:
        return
    result = ScreenerResult.model_validate(result_payload)
    recommendation_payload = payload.get("instrument_recommendation")
    recommendation = None
    if recommendation_payload:
        from schemas.screener import InstrumentRecommendation

        recommendation = InstrumentRecommendation.model_validate(recommendation_payload)
    service.persist_latest(result, recommendation)


@workflow.defn
class UniverseScreenerWorkflow:
    """Run universe screening on a configurable cadence."""

    @workflow.run
    async def run(self, config: dict[str, Any]) -> None:
        cadence_minutes = int(config.get("cadence_minutes", 15))
        max_iterations = config.get("max_iterations")
        iterations = 0
        while True:
            payload = await workflow.execute_activity(
                run_universe_screen,
                config,
                schedule_to_close_timeout=timedelta(minutes=5),
            )
            await workflow.execute_activity(
                emit_screener_result,
                payload,
                schedule_to_close_timeout=timedelta(seconds=30),
            )
            iterations += 1
            if max_iterations is not None and iterations >= int(max_iterations):
                return
            await workflow.sleep(timedelta(minutes=max(1, cadence_minutes)))
