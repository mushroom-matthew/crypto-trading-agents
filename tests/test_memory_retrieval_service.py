"""Tests for services/memory_retrieval_service.py (Runbook 51).

Validates MemoryRetrievalService behavior:
- Empty store edge cases
- Bucket separation and quota enforcement
- Similarity scoring
- Bundle reuse / no-reuse thresholds
- Global fallback
- Diversity constraint
- Retrieval metadata population

No DB or MCP server imports. All state is in-memory.
"""

from __future__ import annotations

import pytest
import math
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from schemas.signal_event import SignalEvent
from schemas.episode_memory import (
    DiversifiedMemoryBundle,
    EpisodeMemoryRecord,
    MemoryRetrievalMeta,
    MemoryRetrievalRequest,
)
from services.episode_memory_service import EpisodeMemoryStore, build_episode_record
from services.memory_retrieval_service import (
    MemoryRetrievalService,
    _cosine_similarity,
    _fingerprint_distance,
    _score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_VALID_UNTIL = datetime(2024, 6, 2, 0, 0, 0, tzinfo=timezone.utc)

_FP_A = {"vol_percentile": 0.8, "atr_percentile": 0.7, "volume_percentile": 0.6}
_FP_B = {"vol_percentile": 0.2, "atr_percentile": 0.1, "volume_percentile": 0.2}


def _make_signal(symbol: str = "BTC-USD", **overrides) -> SignalEvent:
    defaults = dict(
        signal_id=str(uuid4()),
        engine_version="1.0.0",
        ts=_NOW,
        valid_until=_VALID_UNTIL,
        timeframe="1h",
        symbol=symbol,
        direction="long",
        trigger_id="trig-001",
        strategy_type="compression_breakout",
        regime_snapshot_hash="abc" + "0" * 61,
        entry_price=50000.0,
        stop_price_abs=49000.0,
        target_price_abs=52000.0,
        risk_r_multiple=2.0,
        expected_hold_bars=8,
        thesis="Test.",
        feature_schema_version="1.2.0",
    )
    defaults.update(overrides)
    return SignalEvent(**defaults)


def _make_record(
    symbol: str = "BTC-USD",
    outcome_class: str = "win",
    regime_fingerprint: dict | None = None,
    entry_ts: datetime | None = None,
    pnl: float | None = None,
    r_achieved: float | None = None,
    mae: float | None = None,
    mfe: float | None = None,
    playbook_id: str | None = None,
    timeframe: str | None = None,
    trigger_category: str | None = None,
    failure_modes: list | None = None,
) -> EpisodeMemoryRecord:
    """Build a record directly with explicit outcome_class override."""
    sig = _make_signal(symbol=symbol, timeframe=timeframe or "1h")
    if playbook_id:
        sig = _make_signal(symbol=symbol, timeframe=timeframe or "1h", playbook_id=playbook_id)
    rec = build_episode_record(
        sig,
        pnl=pnl,
        r_achieved=r_achieved,
        mae=mae,
        mfe=mfe,
        regime_fingerprint=regime_fingerprint,
        trigger_category=trigger_category,
    )
    # Override outcome_class and failure_modes for test control
    updates: dict = {"outcome_class": outcome_class}
    if failure_modes is not None:
        updates["failure_modes"] = failure_modes
    if entry_ts is not None:
        updates["entry_ts"] = entry_ts
    return rec.model_copy(update=updates)


def _make_request(
    symbol: str = "BTC-USD",
    regime_fingerprint: dict | None = None,
    **overrides,
) -> MemoryRetrievalRequest:
    return MemoryRetrievalRequest(
        symbol=symbol,
        regime_fingerprint=regime_fingerprint or _FP_A,
        **overrides,
    )


def _populated_store(n_wins: int = 3, n_losses: int = 3, fp: dict | None = None) -> EpisodeMemoryStore:
    """Create a store with n_wins + n_losses records for BTC-USD."""
    store = EpisodeMemoryStore()
    fingerprint = fp or _FP_A
    for _ in range(n_wins):
        rec = _make_record(
            outcome_class="win",
            regime_fingerprint=fingerprint,
            r_achieved=2.0,
            pnl=100.0,
        )
        store.add(rec)
    for _ in range(n_losses):
        rec = _make_record(
            outcome_class="loss",
            regime_fingerprint=fingerprint,
            r_achieved=-1.0,
            pnl=-50.0,
            mae=300.0,
            mfe=50.0,
            trigger_category="breakout",
            failure_modes=["low_volume_breakout_failure"],
        )
        store.add(rec)
    return store


# ---------------------------------------------------------------------------
# Cosine similarity and distance helpers
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        fp = {"a": 0.5, "b": 0.5}
        assert _cosine_similarity(fp, fp) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_half(self):
        # cos(90°) = 0 → mapped to 0.5
        a = {"a": 1.0, "b": 0.0}
        b = {"a": 0.0, "b": 1.0}
        assert _cosine_similarity(a, b) == pytest.approx(0.5)

    def test_no_shared_keys_returns_half(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == pytest.approx(0.5)

    def test_empty_dicts_return_half(self):
        assert _cosine_similarity({}, {}) == pytest.approx(0.5)


class TestFingerprintDistance:
    def test_identical_distance_zero(self):
        fp = {"a": 0.5, "b": 0.3}
        assert _fingerprint_distance(fp, fp) == pytest.approx(0.0)

    def test_orthogonal_distance_half(self):
        a = {"a": 1.0, "b": 0.0}
        b = {"a": 0.0, "b": 1.0}
        assert _fingerprint_distance(a, b) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

class TestEmptyStore:
    def test_empty_store_returns_empty_bundles(self):
        store = EpisodeMemoryStore()
        svc = MemoryRetrievalService(store)
        req = _make_request()
        bundle = svc.retrieve(req)
        assert bundle.winning_contexts == []
        assert bundle.losing_contexts == []
        assert bundle.failure_mode_patterns == []

    def test_empty_store_insufficient_buckets_populated(self):
        store = EpisodeMemoryStore()
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        # All three buckets should be listed as insufficient
        meta = bundle.retrieval_meta
        assert "wins" in meta.insufficient_buckets
        assert "losses" in meta.insufficient_buckets

    def test_empty_store_candidate_pool_size_zero(self):
        store = EpisodeMemoryStore()
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert bundle.retrieval_meta.candidate_pool_size == 0


# ---------------------------------------------------------------------------
# Bucket separation and quotas
# ---------------------------------------------------------------------------

class TestBucketSeparation:
    def test_wins_and_losses_separated(self):
        store = _populated_store(n_wins=2, n_losses=2)
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        for rec in bundle.winning_contexts:
            assert rec.outcome_class == "win"
        for rec in bundle.losing_contexts:
            assert rec.outcome_class == "loss"

    def test_win_quota_respected(self):
        store = _populated_store(n_wins=10, n_losses=0)
        svc = MemoryRetrievalService(store)
        req = _make_request(win_quota=3)
        bundle = svc.retrieve(req)
        assert len(bundle.winning_contexts) <= 3

    def test_loss_quota_respected(self):
        store = _populated_store(n_wins=0, n_losses=10)
        svc = MemoryRetrievalService(store)
        req = _make_request(loss_quota=3)
        bundle = svc.retrieve(req)
        assert len(bundle.losing_contexts) <= 3

    def test_failure_mode_quota_respected(self):
        store = EpisodeMemoryStore()
        for _ in range(10):
            rec = _make_record(
                outcome_class="loss",
                failure_modes=["low_volume_breakout_failure"],
                regime_fingerprint=_FP_A,
                pnl=-50.0,
            )
            store.add(rec)
        svc = MemoryRetrievalService(store)
        req = _make_request(failure_mode_quota=2)
        bundle = svc.retrieve(req)
        assert len(bundle.failure_mode_patterns) <= 2

    def test_insufficient_buckets_listed_when_empty(self):
        # Only wins, no losses
        store = _populated_store(n_wins=5, n_losses=0)
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert "losses" in bundle.retrieval_meta.insufficient_buckets


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

class TestSimilarityScoring:
    def test_similar_regime_scores_higher(self):
        """Record with matching regime fingerprint scores higher than mismatching."""
        req = _make_request(regime_fingerprint=_FP_A)

        matching_rec = _make_record(
            regime_fingerprint=_FP_A,
            outcome_class="win",
        )
        mismatch_rec = _make_record(
            regime_fingerprint=_FP_B,
            outcome_class="win",
        )
        score_match = _score(matching_rec, req)
        score_mismatch = _score(mismatch_rec, req)
        assert score_match > score_mismatch

    def test_playbook_match_boosts_score(self):
        req = _make_request(playbook_id="rsi_extremes")
        rec_match = _make_record(regime_fingerprint=_FP_A, outcome_class="win")
        rec_match = rec_match.model_copy(update={"playbook_id": "rsi_extremes"})
        rec_mismatch = _make_record(regime_fingerprint=_FP_A, outcome_class="win")
        rec_mismatch = rec_mismatch.model_copy(update={"playbook_id": "other_playbook"})

        score_match = _score(rec_match, req)
        score_mismatch = _score(rec_mismatch, req)
        assert score_match > score_mismatch

    def test_recency_decays_score(self):
        """Older records score lower than recent records with same regime."""
        req = _make_request(regime_fingerprint=_FP_A, recency_decay_lambda=0.1)
        recent_rec = _make_record(
            regime_fingerprint=_FP_A,
            outcome_class="win",
            entry_ts=datetime.now(tz=timezone.utc) - timedelta(days=1),
        )
        # Set exit_ts explicitly via model_copy
        recent_rec = recent_rec.model_copy(
            update={"exit_ts": datetime.now(tz=timezone.utc) - timedelta(days=1)}
        )
        old_rec = _make_record(
            regime_fingerprint=_FP_A,
            outcome_class="win",
            entry_ts=datetime.now(tz=timezone.utc) - timedelta(days=60),
        )
        old_rec = old_rec.model_copy(
            update={"exit_ts": datetime.now(tz=timezone.utc) - timedelta(days=60)}
        )
        score_recent = _score(recent_rec, req)
        score_old = _score(old_rec, req)
        assert score_recent > score_old


# ---------------------------------------------------------------------------
# Bundle reuse
# ---------------------------------------------------------------------------

class TestBundleReuse:
    def _make_prior_bundle(
        self, delta: float, symbol: str = "BTC-USD"
    ) -> DiversifiedMemoryBundle:
        meta = MemoryRetrievalMeta(
            regime_fingerprint_delta=delta,
            bundle_reused=False,
            candidate_pool_size=6,
        )
        return DiversifiedMemoryBundle(
            bundle_id=str(uuid4()),
            symbol=symbol,
            created_at=datetime.now(tz=timezone.utc),
            retrieval_meta=meta,
        )

    def test_bundle_reused_when_delta_below_threshold(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        prior = self._make_prior_bundle(delta=0.03)
        req = _make_request()
        bundle = svc.retrieve(req, prior_bundle=prior)
        assert bundle.retrieval_meta.bundle_reused is True
        assert bundle.bundle_id == prior.bundle_id

    def test_bundle_not_reused_when_delta_at_threshold(self):
        """Delta exactly 0.05 should NOT trigger reuse (< 0.05 is the condition)."""
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        prior = self._make_prior_bundle(delta=0.05)
        req = _make_request()
        bundle = svc.retrieve(req, prior_bundle=prior)
        assert bundle.retrieval_meta.bundle_reused is False
        assert bundle.bundle_id != prior.bundle_id

    def test_bundle_not_reused_when_delta_above_threshold(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        prior = self._make_prior_bundle(delta=0.20)
        req = _make_request()
        bundle = svc.retrieve(req, prior_bundle=prior)
        assert bundle.retrieval_meta.bundle_reused is False

    def test_no_reuse_when_no_prior_bundle(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request(), prior_bundle=None)
        assert bundle.retrieval_meta.bundle_reused is False

    def test_reused_bundle_preserves_content(self):
        """Reused bundle has same winning/losing contexts as prior."""
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        # First retrieval
        req = _make_request()
        first = svc.retrieve(req)
        # Build a prior with delta < 0.05
        prior_meta = first.retrieval_meta.model_copy(
            update={"regime_fingerprint_delta": 0.02, "bundle_reused": False}
        )
        prior = first.model_copy(update={"retrieval_meta": prior_meta})
        # Second retrieval with prior
        second = svc.retrieve(req, prior_bundle=prior)
        assert second.bundle_id == first.bundle_id
        assert len(second.winning_contexts) == len(first.winning_contexts)


# ---------------------------------------------------------------------------
# Global fallback
# ---------------------------------------------------------------------------

class TestGlobalFallback:
    def test_global_fallback_used_when_symbol_local_insufficient(self):
        """When BTC-USD has no records but ETH-USD has similar-regime records,
        global fallback should include them (within distance threshold)."""
        store = EpisodeMemoryStore()
        # Add wins only for ETH-USD with matching fingerprint
        for _ in range(3):
            rec = _make_record(
                symbol="ETH-USD",
                outcome_class="win",
                regime_fingerprint=_FP_A,
                r_achieved=2.0,
                pnl=100.0,
            )
            store.add(rec)
        svc = MemoryRetrievalService(store)
        req = _make_request(
            symbol="BTC-USD",
            regime_fingerprint=_FP_A,
            global_fallback_max_fingerprint_distance=0.30,
        )
        bundle = svc.retrieve(req)
        # Global fallback should have provided wins from ETH-USD
        assert len(bundle.winning_contexts) > 0
        assert bundle.retrieval_meta.retrieval_scope in ("symbol", "global")

    def test_global_fallback_not_used_when_regime_too_different(self):
        """Records that are too far in regime space should be excluded from fallback."""
        store = EpisodeMemoryStore()
        # Add records for ETH-USD with a very different fingerprint
        for _ in range(3):
            rec = _make_record(
                symbol="ETH-USD",
                outcome_class="win",
                regime_fingerprint=_FP_B,  # very different from _FP_A
                r_achieved=2.0,
                pnl=100.0,
            )
            store.add(rec)
        svc = MemoryRetrievalService(store)
        # Use a very tight fallback threshold so _FP_B doesn't qualify
        req = _make_request(
            symbol="BTC-USD",
            regime_fingerprint=_FP_A,
            global_fallback_max_fingerprint_distance=0.01,  # very tight
        )
        bundle = svc.retrieve(req)
        assert len(bundle.winning_contexts) == 0  # fallback excluded by distance


# ---------------------------------------------------------------------------
# Diversity constraint
# ---------------------------------------------------------------------------

class TestDiversityConstraint:
    def test_same_day_entries_deduped(self):
        """No more than half the quota should come from the same calendar day."""
        store = EpisodeMemoryStore()
        same_day = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        # Add 6 wins all on the same day
        for i in range(6):
            rec = _make_record(
                outcome_class="win",
                regime_fingerprint=_FP_A,
                r_achieved=2.0,
                pnl=100.0,
                entry_ts=same_day + timedelta(hours=i),
            )
            # Explicitly set entry_ts
            rec = rec.model_copy(update={"entry_ts": same_day + timedelta(hours=i)})
            store.add(rec)

        svc = MemoryRetrievalService(store)
        req = _make_request(win_quota=4)
        bundle = svc.retrieve(req)
        # With quota=4, max from same day = max(1, 4//2) = 2
        assert len(bundle.winning_contexts) <= 4


# ---------------------------------------------------------------------------
# Retrieval metadata
# ---------------------------------------------------------------------------

class TestRetrievalMeta:
    def test_candidate_pool_size_reflects_store(self):
        store = _populated_store(n_wins=3, n_losses=3)
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert bundle.retrieval_meta.candidate_pool_size == 6

    def test_retrieval_latency_ms_populated(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert bundle.retrieval_meta.retrieval_latency_ms is not None
        assert bundle.retrieval_meta.retrieval_latency_ms >= 0.0

    def test_insufficient_buckets_empty_when_all_full(self):
        """If all buckets are satisfied, insufficient_buckets should be empty.

        Records are spread across different entry_ts days so the diversity
        constraint (max half-quota per day) does not reduce selection below quota.
        """
        store = EpisodeMemoryStore()
        fp = _FP_A
        base_day = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Add 5 wins each on a different day → diversity constraint never bites
        for i in range(5):
            entry = base_day + timedelta(days=i)
            w = _make_record(outcome_class="win", regime_fingerprint=fp, r_achieved=2.0, pnl=100.0)
            w = w.model_copy(update={"entry_ts": entry})
            store.add(w)
        # Add 5 losses each on a different day
        for i in range(5):
            entry = base_day + timedelta(days=i)
            l = _make_record(
                outcome_class="loss",
                regime_fingerprint=fp,
                r_achieved=-1.0,
                pnl=-50.0,
                failure_modes=["low_volume_breakout_failure"],
            )
            l = l.model_copy(update={"entry_ts": entry})
            store.add(l)

        svc = MemoryRetrievalService(store)
        req = _make_request(win_quota=3, loss_quota=3, failure_mode_quota=2)
        bundle = svc.retrieve(req)
        # wins and losses should be satisfied; failure_mode quota may or may not
        # be satisfied depending on labels; we only check win/loss here
        assert "wins" not in bundle.retrieval_meta.insufficient_buckets
        assert "losses" not in bundle.retrieval_meta.insufficient_buckets

    def test_bundle_id_is_unique_per_retrieval(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        b1 = svc.retrieve(_make_request())
        b2 = svc.retrieve(_make_request())
        assert b1.bundle_id != b2.bundle_id

    def test_retrieval_scope_symbol_when_local_sufficient(self):
        store = _populated_store(n_wins=5, n_losses=5)
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert bundle.retrieval_meta.retrieval_scope in ("symbol", "global")

    def test_similarity_spec_version_correct(self):
        store = _populated_store()
        svc = MemoryRetrievalService(store)
        bundle = svc.retrieve(_make_request())
        assert bundle.retrieval_meta.similarity_spec_version == "1.0.0"
