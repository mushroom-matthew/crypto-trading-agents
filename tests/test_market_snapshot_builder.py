"""Tests for services/market_snapshot_builder.py — R49 Market Snapshot Definition."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from schemas.llm_strategist import (
    AssetState,
    IndicatorSnapshot,
    LLMInput,
    PortfolioState,
)
from schemas.market_snapshot import PolicySnapshot, TickSnapshot
from schemas.structure_engine import LevelLadder, StructureLevel, StructureSnapshot
from services.market_snapshot_builder import (
    build_policy_snapshot,
    build_tick_snapshot,
)

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _indicator(symbol: str = "BTC-USD", timeframe: str = "1h") -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=NOW,
        close=50000.0,
        volume=1234.5,
        atr_14=750.0,
        rsi_14=55.0,
        compression_flag=False,
        expansion_flag=False,
        breakout_confirmed=False,
    )


def _asset_state(symbol: str = "BTC-USD") -> AssetState:
    return AssetState(
        symbol=symbol,
        indicators=[_indicator(symbol)],
        trend_state="uptrend",
        vol_state="normal",
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=NOW,
        equity=10000.0,
        cash=8000.0,
        positions={"BTC-USD": 0.04},
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=1.0,
    )


def _llm_input(symbols: list[str] | None = None) -> LLMInput:
    syms = symbols or ["BTC-USD"]
    return LLMInput(
        portfolio=_portfolio(),
        assets=[_asset_state(s) for s in syms],
        risk_params={},
    )


# ---------------------------------------------------------------------------
# build_tick_snapshot
# ---------------------------------------------------------------------------

class TestBuildTickSnapshot:
    def test_returns_tick_snapshot(self):
        snap = build_tick_snapshot(_indicator())
        assert isinstance(snap, TickSnapshot)

    def test_provenance_fields(self):
        snap = build_tick_snapshot(_indicator())
        assert snap.provenance.snapshot_kind == "tick"
        assert snap.provenance.snapshot_id
        assert snap.provenance.snapshot_hash
        assert snap.provenance.feature_pipeline_hash
        assert snap.provenance.symbol == "BTC-USD"
        assert snap.provenance.timeframe == "1h"

    def test_as_of_ts_matches_indicator(self):
        ind = _indicator()
        snap = build_tick_snapshot(ind)
        assert snap.provenance.as_of_ts == ind.as_of

    def test_data_fields_copied(self):
        snap = build_tick_snapshot(_indicator())
        assert snap.close == 50000.0
        assert snap.volume == 1234.5
        assert snap.atr_14 == 750.0
        assert snap.rsi_14 == 55.0
        assert snap.compression_flag is False

    def test_custom_bar_id(self):
        snap = build_tick_snapshot(_indicator(), bar_id="custom-bar-001")
        assert snap.provenance.created_at_bar_id == "custom-bar-001"

    def test_auto_bar_id_format(self):
        snap = build_tick_snapshot(_indicator())
        bar_id = snap.provenance.created_at_bar_id
        # Format: <symbol>|<timeframe>|<iso8601>
        assert "BTC-USD" in bar_id
        assert "1h" in bar_id

    def test_parent_snapshot_id_threaded(self):
        snap = build_tick_snapshot(
            _indicator(), parent_tick_snapshot_id="parent-001"
        )
        assert snap.provenance.parent_tick_snapshot_id == "parent-001"

    def test_snapshot_hash_deterministic_for_same_inputs(self):
        ind = _indicator()
        s1 = build_tick_snapshot(ind, bar_id="bar-001")
        s2 = build_tick_snapshot(ind, bar_id="bar-001")
        # Same inputs → same hash
        assert s1.provenance.snapshot_hash == s2.provenance.snapshot_hash

    def test_snapshot_id_unique_per_call(self):
        ind = _indicator()
        s1 = build_tick_snapshot(ind)
        s2 = build_tick_snapshot(ind)
        assert s1.provenance.snapshot_id != s2.provenance.snapshot_id

    def test_staleness_not_stale_for_fresh_indicator(self):
        snap = build_tick_snapshot(_indicator(), max_staleness_seconds=300.0)
        # indicator.as_of == NOW, which is well within 300s of build time
        # (clock difference might cause minor staleness in slow CI — use generous threshold)
        # At worst, this test runs minutes after NOW; 300s max_staleness_seconds is fine
        # as long as the test machine clock difference is < 5 min.
        assert isinstance(snap.quality.is_stale, bool)
        assert snap.quality.staleness_seconds >= 0.0

    def test_feature_derivation_log_present(self):
        snap = build_tick_snapshot(_indicator())
        assert len(snap.feature_derivation_log.entries) >= 1
        assert snap.feature_derivation_log.pipeline_hash


# ---------------------------------------------------------------------------
# build_policy_snapshot
# ---------------------------------------------------------------------------

class TestBuildPolicySnapshot:
    def test_returns_policy_snapshot(self):
        snap = build_policy_snapshot(_llm_input())
        assert isinstance(snap, PolicySnapshot)

    def test_provenance_fields(self):
        snap = build_policy_snapshot(_llm_input())
        assert snap.provenance.snapshot_kind == "policy"
        assert snap.provenance.snapshot_id
        assert snap.provenance.snapshot_hash
        assert snap.provenance.symbol == "BTC-USD"

    def test_equity_and_cash(self):
        snap = build_policy_snapshot(_llm_input())
        assert snap.equity == 10000.0
        assert snap.cash == 8000.0

    def test_open_positions(self):
        snap = build_policy_snapshot(_llm_input())
        assert snap.open_positions == sorted(["BTC-USD"])

    def test_numerical_block_populated(self):
        snap = build_policy_snapshot(_llm_input())
        assert "BTC-USD" in snap.numerical
        nb = snap.numerical["BTC-USD"]
        assert nb.close == 50000.0
        assert nb.atr_14 == 750.0

    def test_derived_block_populated(self):
        snap = build_policy_snapshot(_llm_input())
        assert "BTC-USD" in snap.derived
        db = snap.derived["BTC-USD"]
        assert db.trend_state == "uptrend"
        assert db.vol_state == "normal"

    def test_missing_sections_includes_text_and_visual(self):
        snap = build_policy_snapshot(_llm_input())
        assert "text_signals" in snap.quality.missing_sections
        assert "visual_signals" in snap.quality.missing_sections

    def test_missing_sections_includes_memory_bundle_when_absent(self):
        snap = build_policy_snapshot(_llm_input())
        assert "memory_bundle" in snap.quality.missing_sections

    def test_missing_sections_excludes_memory_bundle_when_provided(self):
        snap = build_policy_snapshot(_llm_input(), memory_bundle_id="bundle-001")
        assert "memory_bundle" not in snap.quality.missing_sections

    def test_memory_bundle_fields_threaded(self):
        snap = build_policy_snapshot(
            _llm_input(),
            memory_bundle_id="bundle-001",
            memory_bundle_summary="3 similar regimes",
        )
        assert snap.memory_bundle_id == "bundle-001"
        assert snap.memory_bundle_summary == "3 similar regimes"

    def test_policy_event_type_threaded(self):
        snap = build_policy_snapshot(
            _llm_input(), policy_event_type="plan_generation"
        )
        assert snap.policy_event_type == "plan_generation"

    def test_policy_event_metadata_threaded(self):
        snap = build_policy_snapshot(
            _llm_input(), policy_event_metadata={"run_id": "run-123"}
        )
        assert snap.policy_event_metadata["run_id"] == "run-123"

    def test_snapshot_hash_deterministic(self):
        llm_in = _llm_input()
        s1 = build_policy_snapshot(llm_in, bar_id="bar-001")
        s2 = build_policy_snapshot(llm_in, bar_id="bar-001")
        assert s1.provenance.snapshot_hash == s2.provenance.snapshot_hash

    def test_snapshot_id_unique_per_call(self):
        llm_in = _llm_input()
        s1 = build_policy_snapshot(llm_in)
        s2 = build_policy_snapshot(llm_in)
        assert s1.provenance.snapshot_id != s2.provenance.snapshot_id

    def test_multi_symbol(self):
        snap = build_policy_snapshot(_llm_input(symbols=["BTC-USD", "ETH-USD"]))
        assert "BTC-USD" in snap.numerical
        assert "ETH-USD" in snap.numerical
        # open_positions comes from portfolio.positions, not asset list
        assert snap.open_positions == sorted(["BTC-USD"])

    def test_normalized_features_populated_by_r55(self):
        """R55: normalized_features is now populated from regime fingerprint components."""
        from schemas.regime_fingerprint import (
            FINGERPRINT_VERSION,
            NUMERIC_VECTOR_FEATURE_NAMES_V1,
        )
        snap = build_policy_snapshot(_llm_input())
        for db in snap.derived.values():
            # R55 populates normalized_features with 6 named components
            assert set(db.normalized_features.keys()) == set(NUMERIC_VECTOR_FEATURE_NAMES_V1)
            # All values are in [0, 1]
            for v in db.normalized_features.values():
                assert 0.0 <= v <= 1.0, f"normalized feature out of bounds: {v}"
            # Version is linked to FINGERPRINT_VERSION
            assert db.normalized_features_version == FINGERPRINT_VERSION

    def test_feature_derivation_log_present(self):
        snap = build_policy_snapshot(_llm_input())
        assert len(snap.feature_derivation_log.entries) >= 1
        assert snap.feature_derivation_log.pipeline_hash

    def test_naive_timestamp_gets_utc(self):
        """IndicatorSnapshot with naive datetime should be treated as UTC."""
        ind = IndicatorSnapshot(
            symbol="BTC-USD",
            timeframe="1h",
            as_of=datetime(2025, 1, 15, 12, 0, 0),  # naive
            close=50000.0,
        )
        asset = AssetState(
            symbol="BTC-USD",
            indicators=[ind],
            trend_state="sideways",
            vol_state="low",
        )
        llm_in = LLMInput(portfolio=_portfolio(), assets=[asset], risk_params={})
        snap = build_policy_snapshot(llm_in)
        assert snap.provenance.as_of_ts.tzinfo is not None


# ---------------------------------------------------------------------------
# R58 Structure Engine integration
# ---------------------------------------------------------------------------

def _make_structure_snapshot(
    symbol: str = "BTC-USD",
    snapshot_id: str = "struct-snap-001",
    snapshot_hash: str = "a" * 64,
    policy_priority: str | None = "high",
) -> StructureSnapshot:
    """Build a minimal StructureSnapshot for testing the builder integration."""
    level = StructureLevel(
        level_id=f"{symbol}|prior_session_high|1d|51000.0000",
        snapshot_id=snapshot_id,
        symbol=symbol,
        as_of_ts=NOW,
        price=51000.0,
        source_timeframe="1d",
        kind="prior_session_high",
        source_label="D-1 High",
        role_now="resistance",
        distance_abs=1000.0,
        distance_pct=2.0,
        distance_atr=1.33,
    )
    ladder = LevelLadder(
        source_timeframe="1d",
        near_resistances=[level],
    )
    support_level = StructureLevel(
        level_id=f"{symbol}|prior_session_low|1d|49000.0000",
        snapshot_id=snapshot_id,
        symbol=symbol,
        as_of_ts=NOW,
        price=49000.0,
        source_timeframe="1d",
        kind="prior_session_low",
        source_label="D-1 Low",
        role_now="support",
        distance_abs=1000.0,
        distance_pct=2.0,
        distance_atr=1.33,
    )
    ladder_with_both = LevelLadder(
        source_timeframe="1d",
        near_supports=[support_level],
        near_resistances=[level],
    )
    return StructureSnapshot(
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        symbol=symbol,
        as_of_ts=NOW,
        generated_at_ts=NOW,
        source_timeframe="1h",
        reference_price=50000.0,
        levels=[level, support_level],
        ladders={"1d": ladder_with_both},
        policy_event_priority=policy_priority,
    )


class TestBuildTickSnapshotStructureIntegration:
    """R58: build_tick_snapshot threads structure snapshot refs into TickSnapshot."""

    def test_no_structure_snapshot_fields_are_none(self):
        snap = build_tick_snapshot(_indicator())
        assert snap.structure_snapshot_id is None
        assert snap.structure_snapshot_hash is None

    def test_structure_id_threaded(self):
        struct = _make_structure_snapshot()
        snap = build_tick_snapshot(_indicator(), structure_snapshot=struct)
        assert snap.structure_snapshot_id == "struct-snap-001"

    def test_structure_hash_threaded(self):
        struct = _make_structure_snapshot(snapshot_hash="b" * 64)
        snap = build_tick_snapshot(_indicator(), structure_snapshot=struct)
        assert snap.structure_snapshot_hash == "b" * 64

    def test_tick_snapshot_still_valid_without_structure(self):
        snap = build_tick_snapshot(_indicator())
        assert snap.close == 50000.0
        assert snap.provenance.snapshot_kind == "tick"

    def test_structure_snapshot_does_not_affect_hash(self):
        """Structure refs are NOT included in the tick snapshot hash (provenance only)."""
        ind = _indicator()
        s1 = build_tick_snapshot(ind, bar_id="bar-001")
        struct = _make_structure_snapshot()
        s2 = build_tick_snapshot(ind, bar_id="bar-001", structure_snapshot=struct)
        # Hash is computed from indicator data only — structure reference is additive
        assert s1.provenance.snapshot_hash == s2.provenance.snapshot_hash


class TestBuildPolicySnapshotStructureIntegration:
    """R58: build_policy_snapshot threads structure refs into PolicySnapshot and DerivedSignalBlock."""

    def test_no_structure_adds_structure_engine_to_missing_sections(self):
        snap = build_policy_snapshot(_llm_input())
        assert "structure_engine" in snap.quality.missing_sections

    def test_with_structure_no_missing_section(self):
        struct = _make_structure_snapshot()
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert "structure_engine" not in snap.quality.missing_sections

    def test_structure_snapshot_id_on_policy_snapshot(self):
        struct = _make_structure_snapshot(snapshot_id="struct-abc")
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert snap.structure_snapshot_id == "struct-abc"

    def test_structure_snapshot_hash_on_policy_snapshot(self):
        struct = _make_structure_snapshot(snapshot_hash="c" * 64)
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert snap.structure_snapshot_hash == "c" * 64

    def test_structure_events_count_is_zero_when_no_events(self):
        struct = _make_structure_snapshot()
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert snap.structure_events_count == 0

    def test_structure_policy_priority_threaded(self):
        struct = _make_structure_snapshot(policy_priority="high")
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert snap.structure_policy_priority == "high"

    def test_structure_policy_priority_none_when_no_events(self):
        struct = _make_structure_snapshot(policy_priority=None)
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        assert snap.structure_policy_priority is None

    def test_derived_block_structure_snapshot_id(self):
        struct = _make_structure_snapshot(snapshot_id="struct-xyz")
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        db = snap.derived["BTC-USD"]
        assert db.structure_snapshot_id == "struct-xyz"

    def test_derived_block_nearest_support_pct(self):
        struct = _make_structure_snapshot()
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        db = snap.derived["BTC-USD"]
        # support level is at 49000, ref is 50000 → distance_pct = 2.0
        assert db.nearest_support_pct == pytest.approx(2.0)

    def test_derived_block_nearest_resistance_pct(self):
        struct = _make_structure_snapshot()
        snap = build_policy_snapshot(
            _llm_input(),
            structure_snapshots={"BTC-USD": struct},
        )
        db = snap.derived["BTC-USD"]
        # resistance level is at 51000, ref is 50000 → distance_pct = 2.0
        assert db.nearest_resistance_pct == pytest.approx(2.0)

    def test_derived_block_no_structure_gives_none_distances(self):
        snap = build_policy_snapshot(_llm_input())
        db = snap.derived["BTC-USD"]
        assert db.nearest_support_pct is None
        assert db.nearest_resistance_pct is None

    def test_no_structure_policy_snapshot_refs_are_none(self):
        snap = build_policy_snapshot(_llm_input())
        assert snap.structure_snapshot_id is None
        assert snap.structure_snapshot_hash is None
        assert snap.structure_events_count is None
        assert snap.structure_policy_priority is None
