"""Tests for services/episode_memory_service.py (Runbook 51).

Validates build_episode_record() and EpisodeMemoryStore behavior.

Uses minimal SignalEvent fixtures built inline — no MCP server imports.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from schemas.signal_event import SignalEvent
from schemas.episode_memory import EpisodeMemoryRecord
from services.episode_memory_service import (
    EpisodeMemoryStore,
    build_episode_record,
    _classify_outcome,
    _detect_failure_modes,
)


# ---------------------------------------------------------------------------
# Helpers: minimal SignalEvent factory
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_VALID_UNTIL = datetime(2024, 6, 2, 0, 0, 0, tzinfo=timezone.utc)


def _make_signal(**overrides) -> SignalEvent:
    """Build a minimal valid SignalEvent for testing."""
    defaults = dict(
        signal_id=str(uuid4()),
        engine_version="1.0.0",
        ts=_NOW,
        valid_until=_VALID_UNTIL,
        timeframe="1h",
        symbol="BTC-USD",
        direction="long",
        trigger_id="trig-001",
        strategy_type="compression_breakout",
        regime_snapshot_hash="abc123" + "0" * 58,
        entry_price=50000.0,
        stop_price_abs=49000.0,
        target_price_abs=52000.0,
        risk_r_multiple=2.0,
        expected_hold_bars=8,
        thesis="Compression breakout above resistance.",
        feature_schema_version="1.2.0",
    )
    defaults.update(overrides)
    return SignalEvent(**defaults)


# ---------------------------------------------------------------------------
# _classify_outcome
# ---------------------------------------------------------------------------

class TestClassifyOutcome:
    def test_win_by_r_achieved(self):
        assert _classify_outcome(pnl=None, r_achieved=1.0) == "win"

    def test_win_by_pnl(self):
        assert _classify_outcome(pnl=100.0, r_achieved=None) == "win"

    def test_loss_by_r_achieved(self):
        assert _classify_outcome(pnl=None, r_achieved=-1.0) == "loss"

    def test_loss_by_pnl(self):
        assert _classify_outcome(pnl=-50.0, r_achieved=None) == "loss"

    def test_neutral_zero_pnl(self):
        assert _classify_outcome(pnl=0.0, r_achieved=None) == "neutral"

    def test_neutral_borderline_r(self):
        # 0.0 r_achieved: not > 0.5, not < -0.3, pnl=None → neutral
        assert _classify_outcome(pnl=None, r_achieved=0.0) == "neutral"

    def test_neutral_both_none(self):
        assert _classify_outcome(pnl=None, r_achieved=None) == "neutral"

    def test_r_achieved_takes_precedence_over_pnl(self):
        # r_achieved > 0.5 wins even if pnl is negative (scale mismatch is caller's problem)
        assert _classify_outcome(pnl=-1.0, r_achieved=1.0) == "win"


# ---------------------------------------------------------------------------
# _detect_failure_modes
# ---------------------------------------------------------------------------

class TestDetectFailureModes:
    def test_win_has_no_failure_modes(self):
        modes = _detect_failure_modes(
            outcome_class="win",
            trigger_category="breakout",
            r_achieved=2.0,
            mae=100.0,
            mfe=400.0,
            mae_pct=0.2,
        )
        assert modes == []

    def test_stop_too_tight_noise_out(self):
        # mae/mfe > 3 → stop_too_tight_noise_out
        modes = _detect_failure_modes(
            outcome_class="loss",
            trigger_category="momentum",
            r_achieved=-0.8,
            mae=600.0,
            mfe=150.0,  # ratio = 4 → > 3
            mae_pct=0.6,
        )
        assert "stop_too_tight_noise_out" in modes

    def test_late_entry_poor_r_multiple(self):
        modes = _detect_failure_modes(
            outcome_class="loss",
            trigger_category="momentum",
            r_achieved=-0.8,
            mae=200.0,
            mfe=0.3,  # mfe < 0.5 AND r < -0.5
            mae_pct=None,
        )
        assert "late_entry_poor_r_multiple" in modes

    def test_low_volume_breakout_failure(self):
        modes = _detect_failure_modes(
            outcome_class="loss",
            trigger_category="breakout",
            r_achieved=-0.2,
            mae=100.0,
            mfe=200.0,
            mae_pct=None,
        )
        assert "low_volume_breakout_failure" in modes

    def test_signal_conflict_chop_for_neutral(self):
        modes = _detect_failure_modes(
            outcome_class="neutral",
            trigger_category="momentum",
            r_achieved=0.1,
            mae=50.0,
            mfe=60.0,  # ratio < 3 → no stop_too_tight; not a breakout
            mae_pct=None,
        )
        assert "signal_conflict_chop" in modes


# ---------------------------------------------------------------------------
# build_episode_record
# ---------------------------------------------------------------------------

class TestBuildEpisodeRecord:
    def test_minimal_build_succeeds(self):
        sig = _make_signal()
        rec = build_episode_record(sig)
        assert isinstance(rec, EpisodeMemoryRecord)
        assert rec.signal_id == sig.signal_id
        assert rec.symbol == "BTC-USD"

    def test_outcome_class_win_when_pnl_positive(self):
        sig = _make_signal()
        rec = build_episode_record(sig, pnl=250.0)
        assert rec.outcome_class == "win"

    def test_outcome_class_loss_when_pnl_negative(self):
        sig = _make_signal()
        rec = build_episode_record(sig, pnl=-100.0)
        assert rec.outcome_class == "loss"

    def test_outcome_class_neutral_when_pnl_zero(self):
        sig = _make_signal()
        rec = build_episode_record(sig, pnl=0.0)
        assert rec.outcome_class == "neutral"

    def test_failure_modes_populated_for_breakout_loss(self):
        sig = _make_signal(strategy_type="compression_breakout")
        rec = build_episode_record(sig, pnl=-50.0, trigger_category="breakout")
        assert "low_volume_breakout_failure" in rec.failure_modes

    def test_failure_modes_empty_for_win(self):
        sig = _make_signal()
        rec = build_episode_record(sig, pnl=300.0, r_achieved=2.0)
        assert rec.failure_modes == []

    def test_regime_fingerprint_stored(self):
        sig = _make_signal()
        fp = {"vol_percentile": 0.6, "atr_percentile": 0.4}
        rec = build_episode_record(sig, regime_fingerprint=fp)
        assert rec.regime_fingerprint == fp

    def test_entry_ts_from_signal(self):
        sig = _make_signal()
        rec = build_episode_record(sig)
        assert rec.entry_ts == sig.ts

    def test_resolution_ts_set_when_not_provided(self):
        sig = _make_signal()
        rec = build_episode_record(sig)
        assert rec.resolution_ts is not None

    def test_timeframe_from_signal(self):
        sig = _make_signal(timeframe="15m")
        rec = build_episode_record(sig)
        assert rec.timeframe == "15m"

    def test_playbook_id_from_signal(self):
        sig = _make_signal(playbook_id="rsi_extremes")
        rec = build_episode_record(sig)
        assert rec.playbook_id == "rsi_extremes"

    def test_direction_from_signal(self):
        sig = _make_signal(direction="short")
        rec = build_episode_record(sig)
        assert rec.direction == "short"


# ---------------------------------------------------------------------------
# EpisodeMemoryStore
# ---------------------------------------------------------------------------

class TestEpisodeMemoryStore:
    def _make_record(self, symbol: str = "BTC-USD", **kw) -> EpisodeMemoryRecord:
        sig = _make_signal(symbol=symbol)
        return build_episode_record(sig, **kw)

    def test_empty_store_size_zero(self):
        store = EpisodeMemoryStore()
        assert store.size() == 0

    def test_add_increments_size(self):
        store = EpisodeMemoryStore()
        rec = self._make_record()
        store.add(rec)
        assert store.size() == 1

    def test_add_multiple(self):
        store = EpisodeMemoryStore()
        store.add(self._make_record())
        store.add(self._make_record())
        assert store.size() == 2

    def test_get_by_symbol_returns_matching(self):
        store = EpisodeMemoryStore()
        btc = self._make_record(symbol="BTC-USD")
        eth = self._make_record(symbol="ETH-USD")
        store.add(btc)
        store.add(eth)
        results = store.get_by_symbol("BTC-USD")
        assert len(results) == 1
        assert results[0].symbol == "BTC-USD"

    def test_get_by_symbol_empty_for_unknown(self):
        store = EpisodeMemoryStore()
        store.add(self._make_record(symbol="BTC-USD"))
        assert store.get_by_symbol("SOL-USD") == []

    def test_get_all_returns_all(self):
        store = EpisodeMemoryStore()
        store.add(self._make_record(symbol="BTC-USD"))
        store.add(self._make_record(symbol="ETH-USD"))
        all_recs = store.get_all()
        assert len(all_recs) == 2

    def test_add_overwrite_same_episode_id(self):
        """Adding a record with the same episode_id replaces the existing entry."""
        store = EpisodeMemoryStore()
        sig = _make_signal()
        rec = build_episode_record(sig, pnl=100.0)
        store.add(rec)

        # Overwrite with a mutated copy (different outcome_class)
        updated = rec.model_copy(update={"pnl": -50.0, "outcome_class": "loss"})
        store.add(updated)

        assert store.size() == 1
        retrieved = store.get_by_symbol("BTC-USD")
        assert retrieved[0].outcome_class == "loss"
