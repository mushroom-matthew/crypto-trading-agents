"""Tests for Setup Event Generator (Runbook 44).

Tests use lightweight IndicatorSnapshot fixtures to avoid requiring a live DB.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from schemas.feature_version import FEATURE_SCHEMA_VERSION
from schemas.model_score import ModelScorePacket
from schemas.setup_event import SetupEvent, SessionContext
from services.model_scorer import ModelScorer, NullModelScorer
from agents.analytics.setup_event_generator import (
    SetupEventGenerator,
    _session_context_for,
    _hash_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(**overrides) -> "IndicatorSnapshot":
    """Create a minimal IndicatorSnapshot for testing."""
    from datetime import datetime, timezone
    from schemas.llm_strategist import IndicatorSnapshot

    defaults: Dict[str, Any] = dict(
        symbol="BTC-USD",
        timeframe="1h",
        as_of=datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
        close=50000.0,
        high=50200.0,
        low=49800.0,
        open=50000.0,
        volume=1000.0,
        rsi_14=50.0,
        sma_short=50000.0,
        sma_medium=49900.0,
        ema_short=50050.0,
        ema_medium=49980.0,
        atr_14=200.0,
        macd=70.0,
        macd_signal=60.0,
        macd_hist=10.0,
        bollinger_upper=51000.0,
        bollinger_lower=49000.0,
        bollinger_middle=50000.0,
        donchian_upper_short=50300.0,
        donchian_lower_short=49700.0,
        # Candlestick fields
        candle_body_pct=0.2,  # low conviction
        is_inside_bar=1.0,    # inside bar
        is_impulse_candle=0.0,
        vol_burst=False,
        is_bullish=1.0,
        is_bearish=0.0,
        candle_strength=0.5,
        # HTF fields
        htf_daily_high=51000.0,
        htf_daily_low=49000.0,
        htf_daily_close=50100.0,
        htf_daily_atr=500.0,
        htf_price_vs_daily_mid=0.0,  # near daily midpoint
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def _make_generator(**kwargs) -> SetupEventGenerator:
    return SetupEventGenerator(engine_semver="0.5.0", **kwargs)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSetupEventSchema:
    def test_feature_schema_version_constant(self):
        """FEATURE_SCHEMA_VERSION is '1.3.0' (bumped for R40 compression/breakout indicators)."""
        assert FEATURE_SCHEMA_VERSION == "1.3.0"

    def test_setup_event_schema_valid(self):
        """SetupEvent accepts all required fields."""
        now = datetime.now(timezone.utc)
        event = SetupEvent(
            state="compression_candidate",
            engine_semver="0.5.0",
            feature_schema_version="1.2.0",
            ts=now,
            symbol="BTC-USD",
            timeframe="1h",
            session=SessionContext(
                session_type="crypto_us",
                time_in_session_sin=0.5,
                time_in_session_cos=0.866,
                is_weekend=False,
            ),
            feature_snapshot={"rsi_14": 55.0},
            feature_snapshot_hash="abc123",
        )
        assert event.state == "compression_candidate"
        assert event.outcome is None

    def test_setup_event_extra_forbidden(self):
        """Extra fields are rejected."""
        with pytest.raises(Exception):
            SetupEvent(
                state="break_attempt",
                engine_semver="0.5.0",
                feature_schema_version="1.2.0",
                ts=datetime.now(timezone.utc),
                symbol="BTC-USD",
                timeframe="1h",
                session=SessionContext(
                    session_type="crypto_us",
                    time_in_session_sin=0.5,
                    time_in_session_cos=0.866,
                    is_weekend=False,
                ),
                feature_snapshot={"rsi_14": 55.0},
                feature_snapshot_hash="abc123",
                unknown_key="bad",
            )


# ---------------------------------------------------------------------------
# NullModelScorer
# ---------------------------------------------------------------------------

class TestNullModelScorer:
    def test_null_scorer_returns_none_scores(self):
        """NullModelScorer returns ModelScorePacket with all None scores."""
        scorer = NullModelScorer()
        packet = scorer.score({"rsi_14": 55.0, "close": 50000.0})
        assert packet.model_quality_score is None
        assert packet.p_false_breakout is None
        assert packet.p_cont_1R is None

    def test_null_scorer_is_entry_blocked_false(self):
        """NullModelScorer never blocks entry."""
        scorer = NullModelScorer()
        packet = scorer.score({})
        assert scorer.is_entry_blocked(packet) is False

    def test_null_scorer_size_multiplier_one(self):
        """NullModelScorer returns size_multiplier = 1.0."""
        scorer = NullModelScorer()
        packet = scorer.score({})
        assert scorer.size_multiplier(packet) == 1.0


# ---------------------------------------------------------------------------
# ModelScorer: gate + sizing
# ---------------------------------------------------------------------------

class TestModelScorerGating:
    def _make_packet(self, p_fb=None, quality=None) -> ModelScorePacket:
        return ModelScorePacket(p_false_breakout=p_fb, model_quality_score=quality)

    def test_model_gate_blocks_on_high_p_false_breakout(self):
        """Score with p_false_breakout=0.55 → is_entry_blocked True."""
        scorer = NullModelScorer()
        packet = self._make_packet(p_fb=0.55)
        assert scorer.is_entry_blocked(packet) is True

    def test_model_gate_no_block_below_threshold(self):
        """Score with p_false_breakout=0.39 → is_entry_blocked False."""
        scorer = NullModelScorer()
        packet = self._make_packet(p_fb=0.39)
        assert scorer.is_entry_blocked(packet) is False

    def test_model_gate_none_p_fb_no_block(self):
        """p_false_breakout=None → is_entry_blocked False."""
        scorer = NullModelScorer()
        packet = self._make_packet(p_fb=None)
        assert scorer.is_entry_blocked(packet) is False

    def test_size_multiplier_low_quality(self):
        """quality=0.0 → mult = clamp(0.5 + (0.0-0.5), 0.5, 1.25) = 0.5."""
        scorer = NullModelScorer()
        packet = self._make_packet(quality=0.0)
        assert scorer.size_multiplier(packet) == pytest.approx(0.5)

    def test_size_multiplier_high_quality(self):
        """quality=1.0 → mult = clamp(0.5 + (1.0-0.5), 0.5, 1.25) = 1.0 + 0.5 = 1.0+0.5 = 1.25?
        0.5 + 1.0*(1.0-0.5) = 0.5 + 0.5 = 1.0, then clamped to [0.5, 1.25] = 1.0
        Wait: formula is 0.5 + 1.0*(quality-0.5)
        quality=1.0: 0.5 + 1.0*(0.5) = 1.0, clamped to 1.0
        Actually: max(0.5, min(1.25, 0.5 + 1.0*(1.0-0.5))) = max(0.5, min(1.25, 1.0)) = 1.0
        """
        scorer = NullModelScorer()
        packet = self._make_packet(quality=1.0)
        assert scorer.size_multiplier(packet) == pytest.approx(1.0)

    def test_size_multiplier_mid_quality(self):
        """quality=0.5 → mult = 0.5 + 1.0*(0.0) = 0.5, clamped = 0.5."""
        scorer = NullModelScorer()
        packet = self._make_packet(quality=0.5)
        # 0.5 + 1.0*(0.5-0.5) = 0.5 + 0 = 0.5
        assert scorer.size_multiplier(packet) == pytest.approx(0.5)

    def test_size_multiplier_none_quality(self):
        """quality=None → size_multiplier = 1.0."""
        scorer = NullModelScorer()
        packet = self._make_packet(quality=None)
        assert scorer.size_multiplier(packet) == 1.0

    def test_size_multiplier_clamps_high(self):
        """Very high quality → clamped to 1.25."""
        scorer = NullModelScorer()
        # quality=2.0: 0.5 + 1.0*(2.0-0.5)=0.5+1.5=2.0 → clamp to 1.25
        packet = self._make_packet(quality=2.0)
        assert scorer.size_multiplier(packet) == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# SetupEventGenerator: state machine
# ---------------------------------------------------------------------------

class TestSetupEventGenerator:
    def test_no_event_on_normal_bar(self):
        """Normal bar with large body → no SetupEvent (not inside bar + high body pct)."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)
        snapshot = _make_snapshot(
            is_inside_bar=0.0,
            candle_body_pct=0.8,  # high conviction bar
        )
        events = gen.on_bar("BTC-USD", "1h", now, snapshot)
        assert events == []

    def test_compression_candidate_detected(self):
        """Inside bar + low body + near daily mid → CompressionCandidate emitted."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)
        snapshot = _make_snapshot(
            is_inside_bar=1.0,
            candle_body_pct=0.2,
            htf_price_vs_daily_mid=0.1,  # near mid
        )
        events = gen.on_bar("BTC-USD", "1h", now, snapshot)
        assert len(events) == 1
        assert events[0].state == "compression_candidate"
        assert events[0].symbol == "BTC-USD"
        assert events[0].engine_semver == "0.5.0"
        assert events[0].feature_schema_version == FEATURE_SCHEMA_VERSION

    def test_no_compression_when_impulse_and_vol_burst(self):
        """vol_burst=True + candle_strength > 2 (extreme vol proxy) → no compression candidate."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)
        snapshot = _make_snapshot(
            is_inside_bar=1.0,
            candle_body_pct=0.2,
            vol_burst=True,
            candle_strength=3.0,  # extreme impulse
        )
        events = gen.on_bar("BTC-USD", "1h", now, snapshot)
        assert events == []

    def test_break_attempt_emitted_after_compression(self):
        """Compression detected, then close breaks above range high with impulse → BreakAttempt."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)

        # Bar 1: compression candidate
        snap1 = _make_snapshot(
            is_inside_bar=1.0,
            candle_body_pct=0.2,
            donchian_upper_short=50300.0,
            donchian_lower_short=49700.0,
            close=50000.0,
        )
        events = gen.on_bar("BTC-USD", "1h", now, snap1)
        assert len(events) == 1
        assert events[0].state == "compression_candidate"

        # Bar 2: break above range with impulse
        # 50300 * 1.001 = 50350.3, so close=50450 is above the threshold
        snap2 = _make_snapshot(
            close=50450.0,
            is_impulse_candle=1.0,
            donchian_upper_short=50300.0,
            donchian_lower_short=49700.0,
        )
        events2 = gen.on_bar("BTC-USD", "1h", now, snap2)
        assert len(events2) == 1
        assert events2[0].state == "break_attempt"
        assert events2[0].setup_chain_id == events[0].setup_chain_id

    def test_no_break_without_impulse(self):
        """Close breaks range but no impulse AND no vol burst → stays in compression."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)

        # Bar 1: compression
        snap1 = _make_snapshot(
            is_inside_bar=1.0,
            candle_body_pct=0.2,
            donchian_upper_short=50300.0,
            donchian_lower_short=49700.0,
            close=50000.0,
        )
        gen.on_bar("BTC-USD", "1h", now, snap1)

        # Bar 2: break above range BUT no impulse, no vol burst
        snap2 = _make_snapshot(
            close=50450.0,
            is_impulse_candle=0.0,  # no impulse
            vol_burst=False,        # no vol burst
        )
        events2 = gen.on_bar("BTC-USD", "1h", now, snap2)
        assert events2 == []

    def test_compression_ttl_resets_to_idle(self):
        """After compression_ttl_bars without break, state returns to idle."""
        gen = _make_generator(compression_ttl_bars=3)
        now = datetime.now(timezone.utc)

        # Compression detected
        snap = _make_snapshot(is_inside_bar=1.0, candle_body_pct=0.2, close=50000.0)
        gen.on_bar("BTC-USD", "1h", now, snap)
        assert gen._states["BTC-USD:1h"].state == "compression_candidate"

        # 3 bars of non-breaking (inside bars to keep compressed)
        for _ in range(3):
            gen.on_bar("BTC-USD", "1h", now, snap)

        # Next bar should reset to idle (compression_bars_elapsed > ttl=3)
        snap_normal = _make_snapshot(is_inside_bar=0.0, candle_body_pct=0.5, close=50000.0)
        gen.on_bar("BTC-USD", "1h", now, snap_normal)
        # State was reset to idle (compression_bars_elapsed > ttl on bar 4, then idle)
        # After the TTL bar the state is already "idle", next non-compression bar stays idle
        assert gen._states["BTC-USD:1h"].state == "idle"

    def test_feature_snapshot_immutable_hash(self):
        """feature_snapshot_hash matches SHA-256 of the snapshot dict."""
        gen = _make_generator()
        now = datetime.now(timezone.utc)
        snap = _make_snapshot(is_inside_bar=1.0, candle_body_pct=0.2, close=50000.0)
        events = gen.on_bar("BTC-USD", "1h", now, snap)
        assert len(events) == 1
        evt = events[0]

        expected_hash = _hash_snapshot(evt.feature_snapshot)
        assert evt.feature_snapshot_hash == expected_hash
        assert len(evt.feature_snapshot_hash) == 64  # SHA-256 hex

    def test_null_scorer_in_setup_event(self):
        """SetupEvent with NullModelScorer has None model scores."""
        gen = _make_generator(scorer=NullModelScorer())
        now = datetime.now(timezone.utc)
        snap = _make_snapshot(is_inside_bar=1.0, candle_body_pct=0.2)
        events = gen.on_bar("BTC-USD", "1h", now, snap)
        assert len(events) == 1
        assert events[0].model_quality_score is None
        assert events[0].p_false_breakout is None


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------

class TestSessionContext:
    def test_session_context_crypto_us(self):
        """Bar at 15:00 UTC → session_type = 'crypto_us', is_weekend = False."""
        ts = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)  # Monday
        ctx = _session_context_for(ts, "BTC-USD")
        assert ctx.session_type == "crypto_us"
        assert ctx.is_weekend is False
        assert ctx.asset_class == "crypto"

    def test_session_context_weekend(self):
        """Bar on Saturday → is_weekend = True."""
        ts = datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc)  # Saturday
        ctx = _session_context_for(ts, "BTC-USD")
        assert ctx.is_weekend is True

    def test_session_context_cyclic_encoding(self):
        """sin/cos encoding is cyclic: at start of session sin=0, cos=1."""
        # At exactly hour 0 (start of Asia = start of day)
        ts = datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc)
        ctx = _session_context_for(ts, "BTC-USD")
        assert ctx.session_type == "crypto_asia"
        # frac = 0, angle = 0, sin=0, cos=1
        assert ctx.time_in_session_sin == pytest.approx(0.0, abs=1e-9)
        assert ctx.time_in_session_cos == pytest.approx(1.0, abs=1e-9)

    def test_session_context_asia(self):
        """Bar at 04:00 UTC → session_type = 'crypto_asia'."""
        ts = datetime(2026, 2, 17, 4, 0, tzinfo=timezone.utc)
        ctx = _session_context_for(ts, "BTC-USD")
        assert ctx.session_type == "crypto_asia"

    def test_session_context_london(self):
        """Bar at 09:00 UTC → session_type = 'crypto_london'."""
        ts = datetime(2026, 2, 17, 9, 0, tzinfo=timezone.utc)
        ctx = _session_context_for(ts, "BTC-USD")
        assert ctx.session_type == "crypto_london"


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

class TestSnapshotHash:
    def test_hash_deterministic(self):
        """Same dict always produces same hash."""
        d = {"rsi_14": 55.2, "atr_14": 100.5, "close": 50000.0}
        assert _hash_snapshot(d) == _hash_snapshot(d)

    def test_hash_independent_of_key_order(self):
        """Dict with same key-values but different insertion order → same hash."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert _hash_snapshot(d1) == _hash_snapshot(d2)

    def test_hash_is_sha256(self):
        """Hash is 64-char hex (SHA-256)."""
        h = _hash_snapshot({"x": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
