"""Tests for schemas/market_snapshot.py — R49 Market Snapshot Definition."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from schemas.market_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    DerivedSignalBlock,
    FeatureDerivationEntry,
    FeatureDerivationLog,
    NumericalSignalBlock,
    PolicySnapshot,
    SnapshotProvenance,
    SnapshotQuality,
    TickSnapshot,
    TextSignalDigest,
    VisualSignalFingerprint,
    compute_snapshot_hash,
)

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _provenance(kind: str = "tick") -> SnapshotProvenance:
    return SnapshotProvenance(
        snapshot_kind=kind,  # type: ignore[arg-type]
        snapshot_id=str(uuid4()),
        snapshot_hash="abc123",
        feature_pipeline_hash="def456",
        as_of_ts=NOW,
        generated_at_ts=NOW,
        created_at_bar_id="BTC-USD|1h|2025-01-15T12:00:00+00:00",
        symbol="BTC-USD",
        timeframe="1h",
    )


# ---------------------------------------------------------------------------
# SnapshotProvenance
# ---------------------------------------------------------------------------

class TestSnapshotProvenance:
    def test_defaults(self):
        p = _provenance("tick")
        assert p.snapshot_version == SNAPSHOT_SCHEMA_VERSION
        assert p.snapshot_kind == "tick"
        assert p.policy_event_id is None
        assert p.parent_tick_snapshot_id is None

    def test_policy_kind(self):
        p = _provenance("policy")
        assert p.snapshot_kind == "policy"

    def test_extra_forbid(self):
        with pytest.raises(Exception):
            SnapshotProvenance(
                snapshot_kind="tick",
                snapshot_id="id",
                snapshot_hash="h",
                feature_pipeline_hash="fph",
                as_of_ts=NOW,
                generated_at_ts=NOW,
                created_at_bar_id="bar",
                symbol="BTC-USD",
                timeframe="1h",
                unknown_field="oops",
            )


# ---------------------------------------------------------------------------
# SnapshotQuality
# ---------------------------------------------------------------------------

class TestSnapshotQuality:
    def test_defaults(self):
        q = SnapshotQuality()
        assert q.is_stale is False
        assert q.staleness_seconds == 0.0
        assert q.missing_sections == []
        assert q.quality_warnings == []

    def test_stale_with_warnings(self):
        q = SnapshotQuality(
            is_stale=True,
            staleness_seconds=600.0,
            missing_sections=["text_signals"],
            quality_warnings=["Snapshot is 600s old"],
        )
        assert q.is_stale
        assert q.staleness_seconds == 600.0
        assert "text_signals" in q.missing_sections


# ---------------------------------------------------------------------------
# NumericalSignalBlock
# ---------------------------------------------------------------------------

class TestNumericalSignalBlock:
    def test_all_optional(self):
        b = NumericalSignalBlock()
        assert b.close is None
        assert b.volume is None
        assert b.atr_14 is None

    def test_with_values(self):
        b = NumericalSignalBlock(close=50000.0, volume=1234.5, atr_14=750.0, rsi_14=55.0)
        assert b.close == 50000.0
        assert b.rsi_14 == 55.0

    def test_extra_forbid(self):
        with pytest.raises(Exception):
            NumericalSignalBlock(close=100.0, bad_field=True)


# ---------------------------------------------------------------------------
# DerivedSignalBlock
# ---------------------------------------------------------------------------

class TestDerivedSignalBlock:
    def test_defaults(self):
        d = DerivedSignalBlock()
        assert d.regime is None
        assert d.normalized_features == {}
        assert d.normalized_features_version is None
        assert d.template_id is None

    def test_with_values(self):
        d = DerivedSignalBlock(
            regime="bull_trending",
            trend_state="uptrend",
            vol_state="normal",
            compression_flag=False,
            expansion_flag=True,
            breakout_confirmed=True,
        )
        assert d.trend_state == "uptrend"
        assert d.breakout_confirmed is True

    def test_trend_state_literal_validated(self):
        with pytest.raises(Exception):
            DerivedSignalBlock(trend_state="sideways_up")  # invalid literal

    def test_vol_state_literal_validated(self):
        with pytest.raises(Exception):
            DerivedSignalBlock(vol_state="very_high")  # invalid literal


# ---------------------------------------------------------------------------
# FeatureDerivationEntry / FeatureDerivationLog
# ---------------------------------------------------------------------------

class TestFeatureDerivation:
    def test_entry_defaults(self):
        e = FeatureDerivationEntry(transform="indicator_snapshot")
        assert e.version == "1.0"
        assert e.input_window_bars is None
        assert e.params == {}
        assert e.output_fields == []

    def test_log_defaults(self):
        log = FeatureDerivationLog()
        assert log.entries == []
        assert log.pipeline_hash == ""

    def test_log_with_entries(self):
        e = FeatureDerivationEntry(
            transform="test_transform",
            params={"symbol": "BTC-USD"},
            output_fields=["close", "rsi_14"],
        )
        log = FeatureDerivationLog(entries=[e], pipeline_hash="hash123")
        assert len(log.entries) == 1
        assert log.pipeline_hash == "hash123"


# ---------------------------------------------------------------------------
# TickSnapshot
# ---------------------------------------------------------------------------

class TestTickSnapshot:
    def _make(self, **kwargs) -> TickSnapshot:
        return TickSnapshot(provenance=_provenance("tick"), **kwargs)

    def test_minimal(self):
        snap = self._make()
        assert snap.close is None
        assert snap.compression_flag is None

    def test_with_fields(self):
        snap = self._make(close=50000.0, rsi_14=60.0, compression_flag=False)
        assert snap.close == 50000.0
        assert snap.rsi_14 == 60.0

    def test_extra_forbid(self):
        with pytest.raises(Exception):
            self._make(unknown="x")

    def test_no_policy_only_fields(self):
        # TickSnapshot must NOT have per-symbol dicts (policy-only)
        snap = self._make()
        assert not hasattr(snap, "numerical")
        assert not hasattr(snap, "derived")
        assert not hasattr(snap, "memory_bundle_id")


# ---------------------------------------------------------------------------
# PolicySnapshot
# ---------------------------------------------------------------------------

class TestPolicySnapshot:
    def _make(self, **kwargs) -> PolicySnapshot:
        return PolicySnapshot(provenance=_provenance("policy"), **kwargs)

    def test_minimal(self):
        snap = self._make()
        assert snap.numerical == {}
        assert snap.derived == {}
        assert snap.text_digest is None
        assert snap.memory_bundle_id is None
        assert snap.open_positions == []

    def test_with_numerical_and_derived(self):
        num = {"BTC-USD": NumericalSignalBlock(close=50000.0, atr_14=750.0)}
        der = {"BTC-USD": DerivedSignalBlock(regime="bull_trending", trend_state="uptrend")}
        snap = self._make(numerical=num, derived=der)
        assert snap.numerical["BTC-USD"].close == 50000.0
        assert snap.derived["BTC-USD"].regime == "bull_trending"

    def test_with_memory_bundle(self):
        snap = self._make(
            memory_bundle_id="bundle-001",
            memory_bundle_summary="3 similar regimes found",
        )
        assert snap.memory_bundle_id == "bundle-001"

    def test_with_text_digest(self):
        td = TextSignalDigest(headline_summary="BTC rallies", sentiment="bullish")
        snap = self._make(text_digest=td)
        assert snap.text_digest.sentiment == "bullish"

    def test_with_visual_fingerprint(self):
        vf = VisualSignalFingerprint(
            pattern_tags=["bull_flag"],
            structural_tags=["compression_box"],
        )
        snap = self._make(visual_fingerprint=vf)
        assert "bull_flag" in snap.visual_fingerprint.pattern_tags

    def test_policy_event_fields(self):
        snap = self._make(
            policy_event_type="plan_generation",
            policy_event_metadata={"run_id": "run-123"},
            equity=10000.0,
            cash=8000.0,
            open_positions=["BTC-USD"],
        )
        assert snap.policy_event_type == "plan_generation"
        assert snap.equity == 10000.0
        assert "BTC-USD" in snap.open_positions

    def test_extra_forbid(self):
        with pytest.raises(Exception):
            self._make(bad_field="x")


# ---------------------------------------------------------------------------
# compute_snapshot_hash
# ---------------------------------------------------------------------------

class TestComputeSnapshotHash:
    def test_deterministic(self):
        data = {"symbol": "BTC-USD", "close": 50000.0, "ts": "2025-01-15"}
        h1 = compute_snapshot_hash(data)
        h2 = compute_snapshot_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_snapshot_hash({"a": 1})
        h2 = compute_snapshot_hash({"a": 2})
        assert h1 != h2

    def test_key_order_invariant(self):
        h1 = compute_snapshot_hash({"a": 1, "b": 2})
        h2 = compute_snapshot_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_returns_sha256_hex(self):
        h = compute_snapshot_hash({"x": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_nested_dict_stable(self):
        data = {"symbols": ["BTC-USD", "ETH-USD"], "nested": {"close": 1.0}}
        h = compute_snapshot_hash(data)
        assert len(h) == 64
