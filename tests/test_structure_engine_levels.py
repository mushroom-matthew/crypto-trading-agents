"""Tests for services/structure_engine.py — level extraction and ladder ranking.

Validates:
- S1 anchor levels from IndicatorSnapshot htf_* fields
- Role classification (support/resistance/neutral)
- Distance computation (abs, pct, atr)
- Level deduplication by level_id
- Ladder construction: near/mid/far buckets
- Determinism: same input → same output
- Ranking stability: sorted by distance_abs
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.llm_strategist import IndicatorSnapshot
from schemas.structure_engine import LevelLadder, StructureLevel, StructureSnapshot
from services.structure_engine import (
    _classify_role,
    _compute_distances,
    _make_level_id,
    _bucket_by_atr,
    _build_ladder,
    _extract_s1_from_indicator,
    build_structure_snapshot,
    get_stop_candidates,
    get_target_candidates,
    get_entry_candidates,
)

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)


def _indicator(
    symbol: str = "BTC-USD",
    close: float = 50000.0,
    atr_14: float = 750.0,
    htf_daily_high: float = 51000.0,
    htf_daily_low: float = 49000.0,
    htf_daily_open: float = 49500.0,
    htf_prev_daily_high: float = 52000.0,
    htf_prev_daily_low: float = 48000.0,
    htf_5d_high: float = 53000.0,
    htf_5d_low: float = 47000.0,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe="1h",
        as_of=NOW,
        close=close,
        atr_14=atr_14,
        htf_daily_high=htf_daily_high,
        htf_daily_low=htf_daily_low,
        htf_daily_open=htf_daily_open,
        htf_prev_daily_high=htf_prev_daily_high,
        htf_prev_daily_low=htf_prev_daily_low,
        htf_5d_high=htf_5d_high,
        htf_5d_low=htf_5d_low,
    )


# ---------------------------------------------------------------------------
# _classify_role
# ---------------------------------------------------------------------------

class TestClassifyRole:
    def test_above_reference_is_resistance(self):
        assert _classify_role(51000.0, 50000.0) == "resistance"

    def test_below_reference_is_support(self):
        assert _classify_role(49000.0, 50000.0) == "support"

    def test_near_reference_is_neutral(self):
        # Within 0.05% of reference
        assert _classify_role(50020.0, 50000.0) == "neutral"
        assert _classify_role(49985.0, 50000.0) == "neutral"

    def test_exactly_at_reference_is_neutral(self):
        assert _classify_role(50000.0, 50000.0) == "neutral"

    def test_just_outside_neutral_band_resistance(self):
        # 0.05% above → resistance (price at 50025.1...)
        assert _classify_role(50026.0, 50000.0) == "resistance"

    def test_just_outside_neutral_band_support(self):
        assert _classify_role(49974.0, 50000.0) == "support"

    def test_zero_reference_price_returns_neutral(self):
        assert _classify_role(50000.0, 0.0) == "neutral"


# ---------------------------------------------------------------------------
# _compute_distances
# ---------------------------------------------------------------------------

class TestComputeDistances:
    def test_basic_computation(self):
        dist_abs, dist_pct, dist_atr = _compute_distances(51000.0, 50000.0, 1000.0)
        assert dist_abs == pytest.approx(1000.0)
        assert dist_pct == pytest.approx(2.0)
        assert dist_atr == pytest.approx(1.0)

    def test_below_reference(self):
        dist_abs, dist_pct, dist_atr = _compute_distances(49000.0, 50000.0, 1000.0)
        assert dist_abs == pytest.approx(1000.0)  # abs(49000 - 50000) = 1000
        assert dist_pct == pytest.approx(2.0)

    def test_no_atr_returns_none(self):
        _, _, dist_atr = _compute_distances(51000.0, 50000.0, None)
        assert dist_atr is None

    def test_zero_atr_returns_none(self):
        _, _, dist_atr = _compute_distances(51000.0, 50000.0, 0.0)
        assert dist_atr is None


# ---------------------------------------------------------------------------
# _make_level_id
# ---------------------------------------------------------------------------

class TestMakeLevelId:
    def test_deterministic(self):
        id1 = _make_level_id("BTC-USD", "prior_session_high", "1d", 50000.0)
        id2 = _make_level_id("BTC-USD", "prior_session_high", "1d", 50000.0)
        assert id1 == id2

    def test_different_price_different_id(self):
        id1 = _make_level_id("BTC-USD", "prior_session_high", "1d", 50000.0)
        id2 = _make_level_id("BTC-USD", "prior_session_high", "1d", 51000.0)
        assert id1 != id2

    def test_format(self):
        level_id = _make_level_id("BTC-USD", "prior_session_high", "1d", 50000.1234)
        assert level_id == "BTC-USD|prior_session_high|1d|50000.1234"


# ---------------------------------------------------------------------------
# S1 extraction from IndicatorSnapshot
# ---------------------------------------------------------------------------

class TestExtractS1FromIndicator:
    def test_extracts_d1_high_and_low(self):
        ind = _indicator(close=50000.0, htf_daily_high=51000.0, htf_daily_low=49000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        prices = {l.price for l in levels}
        assert 51000.0 in prices
        assert 49000.0 in prices

    def test_d1_high_classified_as_resistance(self):
        ind = _indicator(close=50000.0, htf_daily_high=51000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        d1_high = next(l for l in levels if l.price == 51000.0 and l.kind == "prior_session_high" and l.source_timeframe == "1d")
        assert d1_high.role_now == "resistance"

    def test_d1_low_classified_as_support(self):
        ind = _indicator(close=50000.0, htf_daily_low=49000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        d1_low = next(l for l in levels if l.price == 49000.0 and l.kind == "prior_session_low" and l.source_timeframe == "1d")
        assert d1_low.role_now == "support"

    def test_5d_high_extracted(self):
        ind = _indicator(close=50000.0, htf_5d_high=53000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        five_d = next((l for l in levels if l.source_timeframe == "5d" and l.kind == "rolling_window_high"), None)
        assert five_d is not None
        assert five_d.price == 53000.0

    def test_5d_low_extracted(self):
        ind = _indicator(close=50000.0, htf_5d_low=47000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        five_d_low = next((l for l in levels if l.source_timeframe == "5d" and l.kind == "rolling_window_low"), None)
        assert five_d_low is not None
        assert five_d_low.price == 47000.0

    def test_d2_anchors_extracted(self):
        ind = _indicator(htf_prev_daily_high=52000.0, htf_prev_daily_low=48000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        d2_prices = {l.price for l in levels if l.source_label.startswith("D-2")}
        assert 52000.0 in d2_prices
        assert 48000.0 in d2_prices

    def test_midpoint_level_created(self):
        # D-1 mid = (51000 + 49000) / 2 = 50000
        ind = _indicator(close=50000.0, htf_daily_high=51000.0, htf_daily_low=49000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        mids = [l for l in levels if l.kind == "prior_session_mid"]
        assert len(mids) == 1
        assert mids[0].price == 50000.0  # (51000 + 49000) / 2

    def test_empty_indicator_returns_no_levels(self):
        ind = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=NOW, close=50000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        assert levels == []

    def test_distance_atr_computed(self):
        ind = _indicator(close=50000.0, htf_daily_high=51500.0, atr_14=1000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 1000.0)
        d1_high = next(l for l in levels if l.price == 51500.0 and l.kind == "prior_session_high")
        # distance = |51500 - 50000| / 1000 = 1.5
        assert d1_high.distance_atr == pytest.approx(1.5)

    def test_level_ids_are_deterministic(self):
        ind = _indicator()
        levels1 = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        levels2 = _extract_s1_from_indicator("snap2", "BTC-USD", NOW, ind, 50000.0, 750.0)
        ids1 = {l.level_id for l in levels1}
        ids2 = {l.level_id for l in levels2}
        # Level IDs should be content-addressed (same prices → same IDs regardless of snapshot_id)
        assert ids1 == ids2

    def test_stop_anchor_eligibility_set(self):
        ind = _indicator(htf_daily_low=49000.0, htf_5d_low=47000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        stop_eligible = [l for l in levels if l.eligible_for_stop_anchor]
        assert len(stop_eligible) > 0

    def test_target_anchor_eligibility_set(self):
        ind = _indicator(htf_daily_high=51000.0, htf_5d_high=53000.0)
        levels = _extract_s1_from_indicator("snap1", "BTC-USD", NOW, ind, 50000.0, 750.0)
        target_eligible = [l for l in levels if l.eligible_for_target_anchor]
        assert len(target_eligible) > 0


# ---------------------------------------------------------------------------
# _build_ladder
# ---------------------------------------------------------------------------

class TestBuildLadder:
    def _make_level(self, price: float, role: str, dist_atr: float) -> StructureLevel:
        ref = 50000.0
        dist_abs = abs(price - ref)
        dist_pct = dist_abs / ref * 100
        return StructureLevel(
            level_id=f"BTC-USD|prior_session_high|1d|{price:.4f}",
            snapshot_id="snap1",
            symbol="BTC-USD",
            as_of_ts=NOW,
            price=price,
            source_timeframe="1d",
            kind="prior_session_high",
            source_label="D-1",
            role_now=role,
            distance_abs=dist_abs,
            distance_pct=dist_pct,
            distance_atr=dist_atr,
        )

    def test_near_support_bucket(self):
        level = self._make_level(49500.0, "support", dist_atr=0.7)
        ladder = _build_ladder([level], "1d")
        assert level in ladder.near_supports
        assert ladder.mid_supports == []
        assert ladder.far_supports == []

    def test_mid_support_bucket(self):
        level = self._make_level(48500.0, "support", dist_atr=2.0)
        ladder = _build_ladder([level], "1d")
        assert level in ladder.mid_supports

    def test_far_support_bucket(self):
        level = self._make_level(46000.0, "support", dist_atr=5.0)
        ladder = _build_ladder([level], "1d")
        assert level in ladder.far_supports

    def test_near_resistance_bucket(self):
        level = self._make_level(50500.0, "resistance", dist_atr=0.7)
        ladder = _build_ladder([level], "1d")
        assert level in ladder.near_resistances

    def test_no_atr_goes_to_mid_bucket(self):
        level = self._make_level(49000.0, "support", dist_atr=None)
        level = level.model_copy(update={"distance_atr": None})
        ladder = _build_ladder([level], "1d")
        assert level in ladder.mid_supports

    def test_neutral_levels_excluded_from_ladder(self):
        # Midpoint D-1 might be neutral — should not appear in any ladder bucket
        ref = 50000.0
        level = StructureLevel(
            level_id="BTC-USD|prior_session_mid|1d|50000.0000",
            snapshot_id="snap1",
            symbol="BTC-USD",
            as_of_ts=NOW,
            price=50000.0,
            source_timeframe="1d",
            kind="prior_session_mid",
            source_label="D-1 Mid",
            role_now="neutral",
            distance_abs=0.0,
            distance_pct=0.0,
            distance_atr=0.0,
        )
        ladder = _build_ladder([level], "1d")
        all_levels = (ladder.near_supports + ladder.mid_supports + ladder.far_supports +
                      ladder.near_resistances + ladder.mid_resistances + ladder.far_resistances)
        assert level not in all_levels

    def test_supports_sorted_by_distance(self):
        far = self._make_level(48000.0, "support", dist_atr=2.5)
        close = self._make_level(49500.0, "support", dist_atr=0.8)
        ladder = _build_ladder([far, close], "1d")
        all_sup = ladder.near_supports + ladder.mid_supports
        # near first
        assert all_sup[0] == close or all_sup[0].distance_abs <= all_sup[-1].distance_abs

    def test_ladder_only_includes_matching_timeframe(self):
        level_1d = self._make_level(51000.0, "resistance", dist_atr=1.5)
        level_5d = StructureLevel(
            level_id="BTC-USD|rolling_window_high|5d|53000.0000",
            snapshot_id="snap1",
            symbol="BTC-USD",
            as_of_ts=NOW,
            price=53000.0,
            source_timeframe="5d",  # different timeframe
            kind="rolling_window_high",
            source_label="5D High",
            role_now="resistance",
            distance_abs=3000.0,
            distance_pct=6.0,
            distance_atr=4.0,
        )
        ladder = _build_ladder([level_1d, level_5d], "1d")
        # Only the 1d level should be in the 1d ladder
        all_levels = (ladder.near_resistances + ladder.mid_resistances + ladder.far_resistances)
        assert level_1d in all_levels
        assert level_5d not in all_levels


# ---------------------------------------------------------------------------
# build_structure_snapshot — full S1 integration
# ---------------------------------------------------------------------------

class TestBuildStructureSnapshot:
    def test_returns_structure_snapshot_type(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        assert isinstance(snap, StructureSnapshot)

    def test_snapshot_has_symbol_and_price(self):
        ind = _indicator(symbol="ETH-USD", close=3000.0)
        snap = build_structure_snapshot(ind)
        assert snap.symbol == "ETH-USD"
        assert snap.reference_price == 3000.0

    def test_s1_levels_populated_from_htf_fields(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        assert len(snap.levels) > 0

    def test_levels_include_d1_high_low(self):
        ind = _indicator(htf_daily_high=51000.0, htf_daily_low=49000.0)
        snap = build_structure_snapshot(ind)
        prices = {l.price for l in snap.levels}
        assert 51000.0 in prices
        assert 49000.0 in prices

    def test_ladders_keyed_by_timeframe(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        assert "1d" in snap.ladders
        assert isinstance(snap.ladders["1d"], LevelLadder)

    def test_snapshot_hash_is_hex_string(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        assert len(snap.snapshot_hash) == 64
        assert all(c in "0123456789abcdef" for c in snap.snapshot_hash)

    def test_deterministic_hash_for_same_input(self):
        ind = _indicator()
        snap1 = build_structure_snapshot(ind)
        snap2 = build_structure_snapshot(ind)
        # Hash covers content (level ids + prices + as_of_ts + reference_price)
        # Since both compute over same content, hashes should match
        # Note: snapshot_id is uuid4, so they differ — hash covers content only
        assert snap1.snapshot_hash == snap2.snapshot_hash

    def test_different_price_different_hash(self):
        ind1 = _indicator(close=50000.0)
        ind2 = _indicator(close=51000.0)
        snap1 = build_structure_snapshot(ind1)
        snap2 = build_structure_snapshot(ind2)
        assert snap1.snapshot_hash != snap2.snapshot_hash

    def test_no_events_without_prior_snapshot(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        # No prior snapshot → no events can be detected
        assert snap.events == []

    def test_quality_flags_set(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        assert "1d" in snap.quality.available_timeframes
        assert "s2_swings" in snap.quality.missing_timeframes  # no ohlcv_df provided

    def test_empty_indicator_produces_partial_snapshot(self):
        ind = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=NOW, close=50000.0)
        snap = build_structure_snapshot(ind)
        assert snap.quality.is_partial is True
        assert "htf_anchors" in snap.quality.missing_timeframes

    def test_no_duplicate_levels(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        ids = [l.level_id for l in snap.levels]
        assert len(ids) == len(set(ids))

    def test_reference_atr_propagated(self):
        ind = _indicator(atr_14=1000.0)
        snap = build_structure_snapshot(ind)
        assert snap.reference_atr == 1000.0

    def test_policy_priority_none_without_events(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind, prior_snapshot=None)
        assert snap.policy_event_priority is None

    def test_stop_candidates_from_snapshot(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        stops = get_stop_candidates(snap)
        assert all(l.eligible_for_stop_anchor for l in stops)
        # Sorted by distance
        if len(stops) > 1:
            assert stops[0].distance_abs <= stops[1].distance_abs

    def test_target_candidates_from_snapshot(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        targets = get_target_candidates(snap)
        assert all(l.eligible_for_target_anchor for l in targets)

    def test_entry_candidates_from_snapshot(self):
        ind = _indicator()
        snap = build_structure_snapshot(ind)
        entries = get_entry_candidates(snap)
        assert all(l.eligible_for_entry_trigger for l in entries)
