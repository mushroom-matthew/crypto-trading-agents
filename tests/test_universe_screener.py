from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncio
import json
import numpy as np
import pandas as pd
import pytest

from schemas.screener import InstrumentRecommendation
from services.universe_screener_service import ScreenerStateStore, UniverseScreenerService


def _ohlcv_frame(
    *,
    n: int = 60,
    start_price: float = 100.0,
    trend_per_bar: float = 0.2,
    base_volume: float = 1000.0,
    last_volume: float | None = None,
    flat: bool = False,
) -> pd.DataFrame:
    ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, float | datetime]] = []
    price = start_price
    for i in range(n):
        if flat:
            close = start_price
            spread = 0.1
        else:
            drift = trend_per_bar * i
            wiggle = 0.2 * np.sin(i / 5.0)
            close = start_price + drift + wiggle
            spread = 0.6 + (0.03 * (i % 7))
        open_ = close - 0.2
        high = close + spread
        low = close - spread
        volume = base_volume + (25 * (i % 5))
        if last_volume is not None and i == n - 1:
            volume = last_volume
            if not flat:
                high = close + spread * 2.5
                low = close - spread * 2.5
        rows.append(
            {
                "timestamp": ts0 + timedelta(hours=i),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(max(0.0, volume)),
            }
        )
        price = close
    return pd.DataFrame(rows)


@pytest.mark.asyncio
async def test_screen_ranks_candidates_and_records_config(monkeypatch):
    monkeypatch.setenv("SCREENER_TOP_N", "2")
    monkeypatch.setenv("SCREENER_COMPRESSION_WEIGHT", "0.3")
    monkeypatch.setenv("SCREENER_EXPANSION_WEIGHT", "0.7")

    data = {
        "AAA-USD": _ohlcv_frame(last_volume=4000.0, trend_per_bar=0.35),
        "BBB-USD": _ohlcv_frame(last_volume=900.0, trend_per_bar=0.05),
        "CCC-USD": _ohlcv_frame(flat=True, last_volume=0.0),
    }

    service = UniverseScreenerService(
        universe=list(data),
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: data[symbol],
    )
    result = await service.screen(timeframe="1h", lookback_bars=50)

    assert result.universe_size == 3
    assert len(result.top_candidates) == 2
    assert result.top_candidates[0].composite_score >= result.top_candidates[1].composite_score
    assert result.screener_config["weights"]["compression"] == pytest.approx(0.3)
    assert result.screener_config["weights"]["expansion"] == pytest.approx(0.7)
    assert result.screener_config["top_n"] == 2
    assert "compression_score" in result.top_candidates[0].score_components
    assert "expansion_score" in result.top_candidates[0].score_components


@pytest.mark.asyncio
async def test_screen_skips_insufficient_data():
    good = _ohlcv_frame(n=60, last_volume=3500.0)
    short = _ohlcv_frame(n=10)
    frames = {"GOOD-USD": good, "SHORT-USD": short}
    service = UniverseScreenerService(
        universe=list(frames),
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: frames[symbol],
    )

    result = await service.screen()

    assert result.universe_size == 1
    assert [c.symbol for c in result.top_candidates] == ["GOOD-USD"]


@pytest.mark.asyncio
async def test_zero_volume_and_flat_prices_produce_finite_scores():
    flat = _ohlcv_frame(flat=True, last_volume=0.0, base_volume=0.0)
    service = UniverseScreenerService(
        universe=["FLAT-USD"],
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: flat,
    )
    result = await service.screen()
    score = result.top_candidates[0]

    assert np.isfinite(score.volume_z)
    assert np.isfinite(score.atr_expansion)
    assert np.isfinite(score.range_expansion_z)
    assert np.isfinite(score.composite_score)
    assert 0.0 <= score.composite_score <= 1.0


@pytest.mark.asyncio
async def test_recommendation_uses_template_suggestion_for_compression_candidate():
    compressed = _ohlcv_frame(flat=True, base_volume=1000.0, last_volume=1200.0)
    service = UniverseScreenerService(
        universe=["COMP-USD"],
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: compressed,
    )
    result = await service.screen()
    rec = service.recommend_from_result(result)

    assert isinstance(rec, InstrumentRecommendation)
    assert rec.selected_symbol == "COMP-USD"
    assert rec.template_id in {"compression_breakout", "uncertain_wait", None}
    assert rec.key_levels is not None
    assert "support" in rec.key_levels


@pytest.mark.asyncio
async def test_grouped_batch_uses_supported_hypotheses_and_caps_per_group():
    frames = {
        f"SYM{i}-USD": _ohlcv_frame(
            trend_per_bar=0.25 if i % 2 == 0 else 0.05,
            last_volume=3500.0 if i < 4 else 1100.0,
            flat=(i >= 4),
        )
        for i in range(8)
    }
    service = UniverseScreenerService(
        universe=list(frames),
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: frames[symbol],
    )
    result = await service.screen()
    batch = service.build_recommendation_batch(result, max_per_group=2)

    assert batch.max_per_group == 2
    assert batch.total_candidates_considered == len(result.top_candidates)
    assert set(batch.supported_hypotheses) >= {"compression_breakout", "volatile_breakout", "uncertain_wait"}
    assert batch.groups
    for group in batch.groups:
        assert group.hypothesis in batch.supported_hypotheses
        assert len(group.recommendations) <= 2
        for idx, item in enumerate(group.recommendations, start=1):
            assert item.hypothesis == group.hypothesis
            assert item.expected_hold_timeframe == group.timeframe
            assert item.rank_in_group == idx


@pytest.mark.asyncio
async def test_session_preflight_payload_is_ready_for_session_start():
    service = UniverseScreenerService(
        universe=["AAA-USD", "BBB-USD"],
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: _ohlcv_frame(
            last_volume=4000.0 if symbol == "AAA-USD" else 1200.0,
            trend_per_bar=0.30 if symbol == "AAA-USD" else 0.02,
        ),
    )
    result = await service.screen()
    preflight = service.build_session_preflight(result, mode="paper")

    assert preflight.mode == "paper"
    assert preflight.screener_run_id == result.run_id
    assert preflight.shortlist.groups
    assert preflight.suggested_default_symbol is not None
    assert any("grouped by supported strategy hypotheses" in note for note in preflight.notes)


@pytest.mark.asyncio
async def test_llm_annotation_reranks_batch_with_stub_transport():
    frames = {
        "AAA-USD": _ohlcv_frame(last_volume=4000.0, trend_per_bar=0.35),
        "BBB-USD": _ohlcv_frame(last_volume=3500.0, trend_per_bar=0.34),
        "CCC-USD": _ohlcv_frame(last_volume=1200.0, flat=True),
    }

    holder: dict[str, any] = {}

    def transport(_system_prompt: str, user_payload_json: str) -> str:
        payload = json.loads(user_payload_json)
        batch = payload["deterministic_batch"]
        holder["baseline"] = batch
        if batch["groups"]:
            first_group = batch["groups"][0]
            first_group["rationale"] = "LLM reranked for clearer thesis grouping."
            if len(first_group["recommendations"]) > 1:
                first_group["recommendations"] = list(reversed(first_group["recommendations"]))
        batch["source"] = "llm_annotated_screener_grouping"
        return json.dumps(batch)

    service = UniverseScreenerService(
        universe=list(frames),
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: frames[symbol],
        batch_annotation_transport=transport,
    )
    result = await service.screen()
    batch = service.build_recommendation_batch(result, annotate_with_llm=True)

    assert batch.annotation_meta is not None
    assert batch.annotation_meta["applied"] is True
    assert batch.source == "llm_annotated_screener_grouping"
    assert batch.groups
    assert "LLM reranked" in batch.groups[0].rationale
    for group in batch.groups:
        assert [item.rank_in_group for item in group.recommendations] == list(range(1, len(group.recommendations) + 1))


@pytest.mark.asyncio
async def test_llm_annotation_invalid_output_falls_back_to_deterministic():
    service = UniverseScreenerService(
        universe=["AAA-USD"],
        ohlcv_fetcher=lambda symbol, timeframe, lookback_bars: _ohlcv_frame(last_volume=3000.0),
        batch_annotation_transport=lambda _system, _payload: "not valid json",
    )
    result = await service.screen()
    batch = service.build_recommendation_batch(result, annotate_with_llm=True)

    assert batch.groups
    assert batch.annotation_meta is not None
    assert batch.annotation_meta["applied"] is False
    assert batch.annotation_meta["mode"] == "deterministic_fallback"


def test_screener_state_store_round_trip(tmp_path):
    store = ScreenerStateStore(
        result_path=tmp_path / "result.json",
        recommendation_path=tmp_path / "recommendation.json",
    )
    df = _ohlcv_frame()
    service = UniverseScreenerService(universe=["AAA-USD"], ohlcv_fetcher=lambda *_args: df, store=store)

    # Use a real screening pass in a small async shim to keep the test deterministic.
    async def _build():
        r = await service.screen()
        return r, service.recommend_from_result(r)

    screener_result, recommendation = asyncio.run(_build())
    service.persist_latest(screener_result, recommendation)

    loaded_result = store.load_result()
    loaded_recommendation = store.load_recommendation()
    assert loaded_result is not None
    assert loaded_recommendation is not None
    assert loaded_result.top_candidates[0].symbol == "AAA-USD"
    assert loaded_recommendation.selected_symbol == "AAA-USD"
