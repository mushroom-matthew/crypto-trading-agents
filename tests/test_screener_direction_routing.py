"""Tests for R59: directional bias computation in universe screener.

Covers:
- _compute_price_position_in_range() — Donchian metric
- _candidate_direction() — deterministic routing rules per hypothesis
- direction_bias field on SymbolAnomalyScore
- Directional template_id returned from _candidate_template_id()
- build_recommendation_batch() direction-aware grouping
- range_mean_revert neutral → uncertain_wait rerouting
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from services.universe_screener_service import UniverseScreenerService
from schemas.screener import (
    SymbolAnomalyScore,
    InstrumentRecommendationItem,
    InstrumentRecommendationGroup,
    ScreenerResult,
)


_NOW = datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)
_RUN_ID = "test-direction-run"


def _make_df(close: float, high: float | None = None, low: float | None = None, bars: int = 30) -> pd.DataFrame:
    """Minimal OHLCV dataframe with fixed price_position_in_range."""
    h = high if high is not None else close * 1.02
    l = low if low is not None else close * 0.98
    rows = [{
        "open": close, "high": h, "low": l, "close": close, "volume": 1000.0,
        "timestamp": _NOW,
    }] * bars
    return pd.DataFrame(rows)


def _make_score(
    symbol: str = "BTC-USD",
    price_position: float = 0.5,
    direction_bias: str = "neutral",
    template_id_suggestion: str | None = "compression_breakout",
    trend_state: str = "range",
    vol_state: str = "normal",
) -> SymbolAnomalyScore:
    return SymbolAnomalyScore(
        symbol=symbol,
        as_of=_NOW,
        volume_z=0.0,
        atr_expansion=0.0,
        range_expansion_z=0.0,
        bb_bandwidth_pct_rank=0.1,
        close=50000.0,
        trend_state=trend_state,  # type: ignore[arg-type]
        vol_state=vol_state,  # type: ignore[arg-type]
        dist_to_prior_high_pct=2.0,
        dist_to_prior_low_pct=-1.0,
        composite_score=0.7,
        price_position_in_range=price_position,
        direction_bias=direction_bias,  # type: ignore[arg-type]
        score_components={
            "template_id_suggestion": template_id_suggestion,
            "compression_score": 0.8,
            "expansion_score": 0.1,
            "price_position_in_range": price_position,
            "direction_bias": direction_bias,
        },
    )


# ---------------------------------------------------------------------------
# _compute_price_position_in_range
# ---------------------------------------------------------------------------

class TestComputePricePositionInRange:
    def _make_range_df(self, close: float, range_high: float, range_low: float) -> pd.DataFrame:
        rows = []
        for _ in range(25):
            rows.append({"open": close, "high": range_high, "low": range_low, "close": close, "volume": 1000.0})
        return pd.DataFrame(rows)

    def test_close_at_midpoint_returns_05(self):
        df = self._make_range_df(close=100.0, range_high=110.0, range_low=90.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 100.0)
        assert abs(pos - 0.5) < 0.01

    def test_close_at_range_low_returns_0(self):
        df = self._make_range_df(close=90.0, range_high=110.0, range_low=90.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 90.0)
        assert pos == pytest.approx(0.0, abs=0.01)

    def test_close_at_range_high_returns_1(self):
        df = self._make_range_df(close=110.0, range_high=110.0, range_low=90.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 110.0)
        assert pos == pytest.approx(1.0, abs=0.01)

    def test_zero_span_returns_05(self):
        """Flat market — Donchian span is 0, default to 0.5."""
        df = self._make_range_df(close=100.0, range_high=100.0, range_low=100.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 100.0)
        assert pos == 0.5

    def test_clamped_above_1(self):
        """Close above range high is clamped to 1.0."""
        df = self._make_range_df(close=120.0, range_high=110.0, range_low=90.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 120.0)
        assert pos == pytest.approx(1.0)

    def test_clamped_below_0(self):
        """Close below range low is clamped to 0.0."""
        df = self._make_range_df(close=80.0, range_high=110.0, range_low=90.0)
        pos = UniverseScreenerService._compute_price_position_in_range(df, 80.0)
        assert pos == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _candidate_direction — routing rules
# ---------------------------------------------------------------------------

class TestCandidateDirection:
    def _dir(self, hypothesis: str, pos: float) -> str:
        return UniverseScreenerService._candidate_direction(hypothesis, pos)

    # compression_breakout
    def test_compression_low_position_is_long(self):
        assert self._dir("compression_breakout", 0.20) == "long"

    def test_compression_boundary_low_is_long(self):
        assert self._dir("compression_breakout", 0.35) == "long"

    def test_compression_midrange_is_neutral(self):
        assert self._dir("compression_breakout", 0.50) == "neutral"

    def test_compression_boundary_high_is_short(self):
        assert self._dir("compression_breakout", 0.65) == "short"

    def test_compression_high_position_is_short(self):
        assert self._dir("compression_breakout", 0.80) == "short"

    # volatile_breakout
    def test_volatile_high_position_is_long(self):
        assert self._dir("volatile_breakout", 0.70) == "long"

    def test_volatile_low_position_is_short(self):
        assert self._dir("volatile_breakout", 0.30) == "short"

    def test_volatile_midrange_is_neutral(self):
        assert self._dir("volatile_breakout", 0.50) == "neutral"

    # range_mean_revert
    def test_range_high_extreme_is_short(self):
        assert self._dir("range_mean_revert", 0.75) == "short"

    def test_range_boundary_high_is_short(self):
        assert self._dir("range_mean_revert", 0.70) == "short"

    def test_range_low_extreme_is_long(self):
        assert self._dir("range_mean_revert", 0.15) == "long"

    def test_range_boundary_low_is_long(self):
        assert self._dir("range_mean_revert", 0.30) == "long"

    def test_range_midrange_is_neutral(self):
        assert self._dir("range_mean_revert", 0.50) == "neutral"

    # trend hypotheses — always directional
    def test_bull_trending_always_long(self):
        for pos in [0.1, 0.5, 0.9]:
            assert self._dir("bull_trending", pos) == "long"

    def test_bear_defensive_always_short(self):
        for pos in [0.1, 0.5, 0.9]:
            assert self._dir("bear_defensive", pos) == "short"

    # uncertain_wait — always neutral
    def test_uncertain_wait_always_neutral(self):
        assert self._dir("uncertain_wait", 0.50) == "neutral"


# ---------------------------------------------------------------------------
# _candidate_template_id — directional template resolution
# ---------------------------------------------------------------------------

class TestCandidateTemplateId:
    def _service(self) -> UniverseScreenerService:
        return UniverseScreenerService.__new__(UniverseScreenerService)

    def test_compression_long(self):
        score = _make_score(direction_bias="long", template_id_suggestion="compression_breakout")
        assert self._service()._candidate_template_id(score) == "compression_breakout_long"

    def test_compression_short(self):
        score = _make_score(direction_bias="short", template_id_suggestion="compression_breakout")
        assert self._service()._candidate_template_id(score) == "compression_breakout_short"

    def test_compression_neutral_returns_base(self):
        score = _make_score(direction_bias="neutral", template_id_suggestion="compression_breakout")
        assert self._service()._candidate_template_id(score) == "compression_breakout"

    def test_volatile_long(self):
        score = _make_score(direction_bias="long", template_id_suggestion="volatile_breakout")
        assert self._service()._candidate_template_id(score) == "volatile_breakout_long"

    def test_volatile_short(self):
        score = _make_score(direction_bias="short", template_id_suggestion="volatile_breakout")
        assert self._service()._candidate_template_id(score) == "volatile_breakout_short"

    def test_range_long(self):
        score = _make_score(direction_bias="long", template_id_suggestion="range_mean_revert")
        assert self._service()._candidate_template_id(score) == "range_long"

    def test_range_short(self):
        score = _make_score(direction_bias="short", template_id_suggestion="range_mean_revert")
        assert self._service()._candidate_template_id(score) == "range_short"

    def test_bull_trending_unchanged(self):
        score = _make_score(direction_bias="long", template_id_suggestion="bull_trending")
        assert self._service()._candidate_template_id(score) == "bull_trending"

    def test_bear_defensive_unchanged(self):
        score = _make_score(direction_bias="short", template_id_suggestion="bear_defensive")
        assert self._service()._candidate_template_id(score) == "bear_defensive"

    def test_none_suggestion_returns_none(self):
        score = _make_score(direction_bias="neutral", template_id_suggestion=None)
        assert self._service()._candidate_template_id(score) is None


# ---------------------------------------------------------------------------
# build_recommendation_batch — direction-aware grouping
# ---------------------------------------------------------------------------

def _make_screener_result(candidates: list[SymbolAnomalyScore]) -> ScreenerResult:
    return ScreenerResult(
        run_id=_RUN_ID,
        as_of=_NOW,
        universe_size=len(candidates),
        top_candidates=candidates,
    )


class TestBatchDirectionGrouping:
    def _service(self) -> UniverseScreenerService:
        svc = UniverseScreenerService.__new__(UniverseScreenerService)
        svc.ohlcv_fetcher = None
        return svc

    def test_long_and_short_compression_form_separate_groups(self):
        candidates = [
            _make_score("BTC-USD", 0.20, "long", "compression_breakout"),
            _make_score("ETH-USD", 0.80, "short", "compression_breakout"),
        ]
        result = _make_screener_result(candidates)
        batch = self._service().build_recommendation_batch(result)
        labels = [g.label for g in batch.groups]
        assert any("Long" in l for l in labels), f"Expected Long group in {labels}"
        assert any("Short" in l for l in labels), f"Expected Short group in {labels}"

    def test_direction_bias_propagated_to_group(self):
        candidates = [_make_score("BTC-USD", 0.20, "long", "compression_breakout")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        group = batch.groups[0]
        assert group.direction_bias == "long"

    def test_direction_bias_propagated_to_item(self):
        candidates = [_make_score("BTC-USD", 0.80, "short", "compression_breakout")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        item = batch.groups[0].recommendations[0]
        assert item.direction_bias == "short"

    def test_range_mean_revert_neutral_rerouted_to_uncertain_wait(self):
        # price_position 0.50 → range_mean_revert neutral → uncertain_wait
        candidates = [_make_score("BTC-USD", 0.50, "neutral", "range_mean_revert", trend_state="range")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        hypotheses = [g.hypothesis for g in batch.groups]
        assert "uncertain_wait" in hypotheses, f"Expected uncertain_wait, got {hypotheses}"
        assert "range_mean_revert" not in hypotheses

    def test_group_label_includes_direction(self):
        candidates = [_make_score("BTC-USD", 0.20, "long", "compression_breakout")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        label = batch.groups[0].label
        assert "↑" in label or "Long" in label, f"Expected direction indicator in label: {label}"

    def test_group_label_short_has_down_arrow(self):
        candidates = [_make_score("BTC-USD", 0.80, "short", "compression_breakout")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        label = batch.groups[0].label
        assert "↓" in label or "Short" in label, f"Expected direction indicator in label: {label}"

    def test_neutral_group_label_has_no_direction_indicator(self):
        candidates = [_make_score("BTC-USD", 0.50, "neutral", "compression_breakout")]
        batch = self._service().build_recommendation_batch(_make_screener_result(candidates))
        label = batch.groups[0].label
        assert "↑" not in label and "↓" not in label and "Long" not in label and "Short" not in label


# ---------------------------------------------------------------------------
# Schema field validation
# ---------------------------------------------------------------------------

class TestSchemaDirectionFields:
    def test_symbol_anomaly_score_defaults(self):
        score = _make_score()
        assert score.price_position_in_range == 0.5
        assert score.direction_bias == "neutral"

    def test_recommendation_item_defaults(self):
        item = InstrumentRecommendationItem(
            symbol="BTC-USD",
            hypothesis="compression_breakout",
            expected_hold_timeframe="1h",
            thesis="test",
            confidence="high",
            composite_score=0.7,
            rank_global=1,
            rank_in_group=1,
        )
        assert item.direction_bias == "neutral"

    def test_recommendation_group_defaults(self):
        group = InstrumentRecommendationGroup(
            hypothesis="compression_breakout",
            timeframe="1h",
            label="test",
            rationale="test",
        )
        assert group.direction_bias == "neutral"

    def test_price_position_in_range_clamped(self):
        import pytest
        with pytest.raises(Exception):
            _make_score(price_position=1.5)  # ge=0.0, le=1.0
