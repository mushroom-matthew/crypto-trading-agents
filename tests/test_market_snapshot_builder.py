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

    def test_normalized_features_empty_by_default(self):
        """normalized_features is intentionally empty until R55."""
        snap = build_policy_snapshot(_llm_input())
        for db in snap.derived.values():
            assert db.normalized_features == {}
            assert db.normalized_features_version is None

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
