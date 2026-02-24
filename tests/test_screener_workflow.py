from __future__ import annotations

from datetime import datetime, timezone

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from schemas.screener import InstrumentRecommendation, ScreenerResult, SymbolAnomalyScore
from services.universe_screener_service import UniverseScreenerService
from tools.universe_screener import UniverseScreenerWorkflow, emit_screener_result, run_universe_screen


@pytest.mark.asyncio
async def test_universe_screener_workflow_runs_and_emits(monkeypatch):
    emitted: list[tuple[ScreenerResult, InstrumentRecommendation | None]] = []

    async def fake_screen(self, timeframe: str = "1h", lookback_bars: int = 50) -> ScreenerResult:
        candidate = SymbolAnomalyScore(
            symbol="BTC-USD",
            as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
            volume_z=2.0,
            atr_expansion=0.4,
            range_expansion_z=1.0,
            bb_bandwidth_pct_rank=0.15,
            close=100.0,
            trend_state="uptrend",
            vol_state="high",
            dist_to_prior_high_pct=-0.2,
            dist_to_prior_low_pct=0.8,
            composite_score=0.82,
            score_components={"compression_score": 0.85, "expansion_score": 0.7, "template_id_suggestion": "compression_breakout"},
        )
        return ScreenerResult(
            run_id="run-1",
            as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
            universe_size=1,
            top_candidates=[candidate],
            screener_config={"weights": {"compression": 0.5, "expansion": 0.5}, "top_n": 1},
        )

    def fake_recommend(self, result: ScreenerResult, timeframe: str | None = None) -> InstrumentRecommendation:
        return InstrumentRecommendation(
            selected_symbol="BTC-USD",
            thesis="Test recommendation",
            strategy_type="compression_breakout",
            template_id="compression_breakout",
            regime_view="uptrend/high",
            key_levels={"support": 99.0, "resistance": 101.0, "pivot": 100.0},
            expected_hold_timeframe=timeframe or "1h",
            confidence="high",
            disqualified_symbols=[],
            disqualification_reasons={},
        )

    def fake_persist(self, result: ScreenerResult, recommendation: InstrumentRecommendation | None = None) -> None:
        emitted.append((result, recommendation))

    monkeypatch.setattr(UniverseScreenerService, "screen", fake_screen)
    monkeypatch.setattr(UniverseScreenerService, "recommend_from_result", fake_recommend)
    monkeypatch.setattr(UniverseScreenerService, "persist_latest", fake_persist)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-screener",
            workflows=[UniverseScreenerWorkflow],
            activities=[run_universe_screen, emit_screener_result],
        ):
            handle = await env.client.start_workflow(
                UniverseScreenerWorkflow.run,
                {"cadence_minutes": 1, "max_iterations": 1, "timeframe": "1h", "lookback_bars": 50},
                id="test-universe-screener",
                task_queue="test-screener",
            )
            await handle.result()

    assert emitted
    saved_result, saved_recommendation = emitted[0]
    assert saved_result.top_candidates[0].symbol == "BTC-USD"
    assert saved_recommendation is not None
    assert saved_recommendation.selected_symbol == "BTC-USD"

