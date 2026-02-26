"""Tests for schemas/episode_memory.py (Runbook 51).

Validates that all Pydantic models enforce their contracts:
- Required fields present
- Literal constraints enforced
- Default values correct
- extra="forbid" active on all models
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from schemas.episode_memory import (
    FAILURE_MODE_TAXONOMY,
    DiversifiedMemoryBundle,
    EpisodeMemoryRecord,
    MemoryRetrievalMeta,
    MemoryRetrievalRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(**overrides) -> EpisodeMemoryRecord:
    defaults = dict(
        episode_id=str(uuid4()),
        symbol="BTC-USD",
        outcome_class="win",
    )
    defaults.update(overrides)
    return EpisodeMemoryRecord(**defaults)


def _make_bundle(**overrides) -> DiversifiedMemoryBundle:
    defaults = dict(
        bundle_id=str(uuid4()),
        symbol="BTC-USD",
        created_at=datetime.now(tz=timezone.utc),
        retrieval_meta=MemoryRetrievalMeta(),
    )
    defaults.update(overrides)
    return DiversifiedMemoryBundle(**defaults)


def _make_request(**overrides) -> MemoryRetrievalRequest:
    defaults = dict(
        symbol="BTC-USD",
        regime_fingerprint={"vol_percentile": 0.5, "atr_percentile": 0.3},
    )
    defaults.update(overrides)
    return MemoryRetrievalRequest(**defaults)


# ---------------------------------------------------------------------------
# EpisodeMemoryRecord
# ---------------------------------------------------------------------------

class TestEpisodeMemoryRecord:
    def test_minimal_valid_record(self):
        """EpisodeMemoryRecord accepts minimal required fields."""
        rec = _make_episode()
        assert rec.symbol == "BTC-USD"
        assert rec.outcome_class == "win"
        assert rec.failure_modes == []

    def test_outcome_class_win(self):
        rec = _make_episode(outcome_class="win")
        assert rec.outcome_class == "win"

    def test_outcome_class_loss(self):
        rec = _make_episode(outcome_class="loss")
        assert rec.outcome_class == "loss"

    def test_outcome_class_neutral(self):
        rec = _make_episode(outcome_class="neutral")
        assert rec.outcome_class == "neutral"

    def test_outcome_class_invalid_rejected(self):
        """Invalid outcome_class must raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_episode(outcome_class="breakeven")

    def test_failure_modes_defaults_empty(self):
        """failure_modes field defaults to an empty list."""
        rec = _make_episode()
        assert isinstance(rec.failure_modes, list)
        assert len(rec.failure_modes) == 0

    def test_failure_modes_populated(self):
        rec = _make_episode(failure_modes=["stop_too_tight_noise_out"])
        assert "stop_too_tight_noise_out" in rec.failure_modes

    def test_direction_literal_long(self):
        rec = _make_episode(direction="long")
        assert rec.direction == "long"

    def test_direction_literal_short(self):
        rec = _make_episode(direction="short")
        assert rec.direction == "short"

    def test_direction_invalid_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_episode(direction="flat")

    def test_extra_fields_rejected(self):
        """extra='forbid' must reject unknown fields."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EpisodeMemoryRecord(
                episode_id=str(uuid4()),
                symbol="BTC-USD",
                outcome_class="win",
                unknown_field="bad",
            )

    def test_optional_fields_default_none(self):
        rec = _make_episode()
        assert rec.signal_id is None
        assert rec.trade_id is None
        assert rec.pnl is None
        assert rec.r_achieved is None


# ---------------------------------------------------------------------------
# MemoryRetrievalMeta
# ---------------------------------------------------------------------------

class TestMemoryRetrievalMeta:
    def test_default_bundle_reused_false(self):
        meta = MemoryRetrievalMeta()
        assert meta.bundle_reused is False

    def test_default_candidate_pool_size_zero(self):
        meta = MemoryRetrievalMeta()
        assert meta.candidate_pool_size == 0

    def test_default_insufficient_buckets_empty(self):
        meta = MemoryRetrievalMeta()
        assert meta.insufficient_buckets == []

    def test_default_retrieval_scope_symbol(self):
        meta = MemoryRetrievalMeta()
        assert meta.retrieval_scope == "symbol"

    def test_similarity_spec_version_default(self):
        meta = MemoryRetrievalMeta()
        assert meta.similarity_spec_version == "1.0.0"

    def test_extra_fields_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryRetrievalMeta(unknown="x")


# ---------------------------------------------------------------------------
# DiversifiedMemoryBundle
# ---------------------------------------------------------------------------

class TestDiversifiedMemoryBundle:
    def test_empty_bundle_valid(self):
        bundle = _make_bundle()
        assert bundle.winning_contexts == []
        assert bundle.losing_contexts == []
        assert bundle.failure_mode_patterns == []

    def test_bundle_with_records(self):
        rec = _make_episode(outcome_class="win")
        bundle = _make_bundle(winning_contexts=[rec])
        assert len(bundle.winning_contexts) == 1

    def test_bundle_extra_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DiversifiedMemoryBundle(
                bundle_id=str(uuid4()),
                symbol="BTC-USD",
                created_at=datetime.now(tz=timezone.utc),
                retrieval_meta=MemoryRetrievalMeta(),
                unexpected_key="oops",
            )


# ---------------------------------------------------------------------------
# MemoryRetrievalRequest
# ---------------------------------------------------------------------------

class TestMemoryRetrievalRequest:
    def test_defaults_sensible(self):
        req = _make_request()
        assert req.win_quota == 3
        assert req.loss_quota == 3
        assert req.failure_mode_quota == 2
        assert req.recency_decay_lambda == pytest.approx(0.01)
        assert req.regime_weight == pytest.approx(0.40)
        assert req.playbook_weight == pytest.approx(0.30)
        assert req.timeframe_weight == pytest.approx(0.15)
        assert req.feature_vector_weight == pytest.approx(0.15)
        assert req.global_fallback_max_fingerprint_distance == pytest.approx(0.30)

    def test_fingerprint_stored(self):
        fp = {"vol_percentile": 0.7, "atr_percentile": 0.4}
        req = _make_request(regime_fingerprint=fp)
        assert req.regime_fingerprint == fp

    def test_extra_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryRetrievalRequest(
                symbol="BTC-USD",
                regime_fingerprint={"vol_percentile": 0.5},
                unexpected="bad",
            )

    def test_optional_fields_default_none(self):
        req = _make_request()
        assert req.playbook_id is None
        assert req.template_id is None
        assert req.direction is None
        assert req.timeframe is None
        assert req.policy_event_type is None


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class TestFailureModeTaxonomy:
    def test_taxonomy_not_empty(self):
        assert len(FAILURE_MODE_TAXONOMY) > 0

    def test_known_modes_present(self):
        assert "false_breakout_reversion" in FAILURE_MODE_TAXONOMY
        assert "stop_too_tight_noise_out" in FAILURE_MODE_TAXONOMY
        assert "late_entry_poor_r_multiple" in FAILURE_MODE_TAXONOMY
