"""Per-symbol state machine that detects compression → break attempt transitions
and emits SetupEvents for the training data pipeline.

State lifecycle:
    IDLE
      ↓  _is_compression_candidate()
    COMPRESSION_CANDIDATE  (emits SetupEvent, freezes range)
      ↓  _is_break_attempt()  (using locked range from candidate state)
    BREAK_ATTEMPT          (emits SetupEvent, triggers model scoring)
      ↓  outcome reconciler fills later
    CONFIRMED | FAILED | TTL_EXPIRED
      ↓  → back to IDLE

Detection criteria (intentionally simple — tune after first 200 labeled events):

CompressionCandidate (all required):
  - is_inside_bar == 1.0  (current range inside prior bar)
  - candle_body_pct < 0.40  (weak directional conviction)
  - vol_state not in ('extreme',)
  - htf_price_vs_daily_mid: abs value < 0.75  (near daily midpoint, not at structural extreme)
    (condition skipped if htf_price_vs_daily_mid is None — insufficient history)

BreakAttempt (from COMPRESSION_CANDIDATE state):
  - close > compression_range_high * 1.001  (upside break, 0.1% noise filter)
    OR close < compression_range_low * 0.999  (downside break)
  - is_impulse_candle == 1.0 OR vol_burst == 1  (impulse confirmation)
  - Within compression_ttl_bars of CompressionCandidate (default: 24 bars)

TTL expiry from BreakAttempt: outcome_ttl_bars (default: 48 bars).

DISCLAIMER: Setup events are research observations only. Not investment advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from schemas.feature_version import FEATURE_SCHEMA_VERSION
from schemas.setup_event import SessionContext, SetupEvent, SetupState
from schemas.llm_strategist import IndicatorSnapshot
from services.model_scorer import ModelScorer, NullModelScorer


# Compression detection thresholds (tune after ≥200 labeled events)
_COMPRESSION_BODY_PCT_MAX = 0.40
_COMPRESSION_HTF_MID_ABS_MAX = 0.75
_BREAK_NOISE_FILTER = 0.001   # 0.1% — prevents false breaks from spread noise
_COMPRESSION_TTL_BARS = 24   # Max bars to wait for break after compression
_OUTCOME_TTL_BARS = 48       # Max bars to wait for outcome after break attempt


def _hash_snapshot(snapshot_dict: dict) -> str:
    canonical = json.dumps(snapshot_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _session_context_for(ts: datetime, symbol: str) -> SessionContext:
    """Compute session context tokens for a bar timestamp.

    For crypto, defines four synthetic liquidity windows:
      Asia:   00:00–08:00 UTC
      London: 08:00–13:00 UTC
      US:     13:00–20:00 UTC
      Close:  20:00–24:00 UTC
    """
    hour = ts.hour + ts.minute / 60.0
    session_boundaries = [
        ("crypto_asia",   0.0,  8.0),
        ("crypto_london", 8.0, 13.0),
        ("crypto_us",    13.0, 20.0),
        ("crypto_close", 20.0, 24.0),
    ]
    session_type = "crypto_utc_day"
    session_start, session_end = 0.0, 24.0
    for name, start, end in session_boundaries:
        if start <= hour < end:
            session_type = name
            session_start, session_end = start, end
            break

    session_len = session_end - session_start
    frac = (hour - session_start) / session_len if session_len > 0 else 0.0
    angle = 2 * math.pi * frac
    is_weekend = ts.weekday() >= 5  # Saturday=5, Sunday=6

    return SessionContext(
        session_type=session_type,
        time_in_session_sin=math.sin(angle),
        time_in_session_cos=math.cos(angle),
        minutes_to_session_close=max(0.0, (session_end - hour) * 60),
        is_weekend=is_weekend,
        asset_class="crypto",
    )


@dataclass
class _SymbolState:
    """In-memory state machine state for one symbol."""
    state: str = "idle"
    setup_chain_id: str = ""
    compression_range_high: Optional[float] = None
    compression_range_low: Optional[float] = None
    compression_atr: Optional[float] = None
    compression_ts: Optional[datetime] = None
    compression_bars_elapsed: int = 0
    break_ts: Optional[datetime] = None
    break_bars_elapsed: int = 0


class SetupEventGenerator:
    """Per-symbol state machine that detects and emits SetupEvents.

    Usage:
        gen = SetupEventGenerator(engine_semver="0.5.0", scorer=NullModelScorer())
        for bar_ts, snapshot in per_bar_loop:
            events = gen.on_bar(symbol, timeframe, bar_ts, snapshot)
            for evt in events:
                ledger_service.write(evt)
    """

    def __init__(
        self,
        engine_semver: str,
        scorer: Optional[ModelScorer] = None,
        strategy_template_version: Optional[str] = None,
        compression_ttl_bars: int = _COMPRESSION_TTL_BARS,
        outcome_ttl_bars: int = _OUTCOME_TTL_BARS,
    ) -> None:
        self._engine_semver = engine_semver
        self._scorer = scorer or NullModelScorer()
        self._template_version = strategy_template_version
        self._compression_ttl = compression_ttl_bars
        self._outcome_ttl = outcome_ttl_bars
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
    ) -> List[SetupEvent]:
        """Process one bar. Returns 0 or more SetupEvents emitted this bar."""
        key = f"{symbol}:{timeframe}"
        if key not in self._states:
            self._states[key] = _SymbolState()
        state = self._states[key]
        events: List[SetupEvent] = []

        if state.state == "idle":
            if self._is_compression_candidate(snapshot):
                evt = self._emit_compression_candidate(symbol, timeframe, bar_ts, snapshot, state)
                events.append(evt)

        elif state.state == "compression_candidate":
            state.compression_bars_elapsed += 1
            if state.compression_bars_elapsed > self._compression_ttl:
                state.state = "idle"
            elif self._is_break_attempt(snapshot, state):
                evt = self._emit_break_attempt(symbol, timeframe, bar_ts, snapshot, state)
                events.append(evt)
            elif not self._is_still_compressed(snapshot):
                state.state = "idle"

        elif state.state == "break_attempt":
            state.break_bars_elapsed += 1
            if state.break_bars_elapsed > self._outcome_ttl:
                state.state = "idle"

        return events

    def reset(self, symbol: str, timeframe: str) -> None:
        """Reset state machine for a symbol/timeframe (e.g., at day boundaries)."""
        key = f"{symbol}:{timeframe}"
        if key in self._states:
            self._states[key] = _SymbolState()

    # ------------------------------------------------------------------
    # Detection predicates
    # ------------------------------------------------------------------

    def _is_compression_candidate(self, snapshot: IndicatorSnapshot) -> bool:
        has_inside_bar = (snapshot.is_inside_bar or 0.0) >= 1.0
        low_conviction = (snapshot.candle_body_pct or 1.0) < _COMPRESSION_BODY_PCT_MAX
        htf_near_mid = True
        if snapshot.htf_price_vs_daily_mid is not None:
            htf_near_mid = abs(snapshot.htf_price_vs_daily_mid) < _COMPRESSION_HTF_MID_ABS_MAX
        # Proxy for vol_state == "extreme": treat as extreme if vol_burst is True AND
        # candle_strength (body/ATR) > 2.0 (a very large impulse bar).
        # This avoids flagging compression in the middle of an expansion.
        if snapshot.vol_burst and (snapshot.candle_strength or 0.0) > 2.0:
            return False
        return has_inside_bar and low_conviction and htf_near_mid

    def _is_break_attempt(self, snapshot: IndicatorSnapshot, state: _SymbolState) -> bool:
        if state.compression_range_high is None or state.compression_range_low is None:
            return False
        close = snapshot.close
        upside_break = close > state.compression_range_high * (1 + _BREAK_NOISE_FILTER)
        downside_break = close < state.compression_range_low * (1 - _BREAK_NOISE_FILTER)
        if not (upside_break or downside_break):
            return False
        impulse = (snapshot.is_impulse_candle or 0.0) >= 1.0
        vol_burst = bool(snapshot.vol_burst)
        return impulse or vol_burst

    def _is_still_compressed(self, snapshot: IndicatorSnapshot) -> bool:
        """Check if compression conditions still hold."""
        if (snapshot.is_impulse_candle or 0.0) >= 1.0:
            return False
        return True

    # ------------------------------------------------------------------
    # Event factories
    # ------------------------------------------------------------------

    def _freeze_snapshot(self, snapshot: IndicatorSnapshot) -> tuple:
        snap_dict = snapshot.model_dump()
        for k, v in snap_dict.items():
            if isinstance(v, datetime):
                snap_dict[k] = v.isoformat()
        snap_hash = _hash_snapshot(snap_dict)
        return snap_dict, snap_hash

    def _emit_compression_candidate(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
        state: _SymbolState,
    ) -> SetupEvent:
        chain_id = str(uuid.uuid4())
        state.state = "compression_candidate"
        state.setup_chain_id = chain_id
        state.compression_range_high = snapshot.donchian_upper_short
        state.compression_range_low = snapshot.donchian_lower_short
        state.compression_atr = snapshot.atr_14
        state.compression_ts = bar_ts
        state.compression_bars_elapsed = 0

        snap_dict, snap_hash = self._freeze_snapshot(snapshot)
        score = self._scorer.score(snap_dict)

        return SetupEvent(
            setup_chain_id=chain_id,
            state="compression_candidate",
            engine_semver=self._engine_semver,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            strategy_template_version=self._template_version,
            ts=bar_ts,
            symbol=symbol,
            timeframe=timeframe,
            session=_session_context_for(bar_ts, symbol),
            feature_snapshot=snap_dict,
            feature_snapshot_hash=snap_hash,
            compression_range_high=state.compression_range_high,
            compression_range_low=state.compression_range_low,
            compression_atr_at_detection=state.compression_atr,
            model_quality_score=score.model_quality_score,
            p_cont_1R=score.p_cont_1R,
            p_false_breakout=score.p_false_breakout,
            p_atr_expand=score.p_atr_expand,
            model_version=score.model_version,
            ttl_bars=self._outcome_ttl,
        )

    def _emit_break_attempt(
        self,
        symbol: str,
        timeframe: str,
        bar_ts: datetime,
        snapshot: IndicatorSnapshot,
        state: _SymbolState,
    ) -> SetupEvent:
        state.state = "break_attempt"
        state.break_ts = bar_ts
        state.break_bars_elapsed = 0

        snap_dict, snap_hash = self._freeze_snapshot(snapshot)
        score = self._scorer.score(snap_dict)

        return SetupEvent(
            setup_chain_id=state.setup_chain_id,
            state="break_attempt",
            engine_semver=self._engine_semver,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            strategy_template_version=self._template_version,
            ts=bar_ts,
            symbol=symbol,
            timeframe=timeframe,
            session=_session_context_for(bar_ts, symbol),
            feature_snapshot=snap_dict,
            feature_snapshot_hash=snap_hash,
            compression_range_high=state.compression_range_high,
            compression_range_low=state.compression_range_low,
            compression_atr_at_detection=state.compression_atr,
            model_quality_score=score.model_quality_score,
            p_cont_1R=score.p_cont_1R,
            p_false_breakout=score.p_false_breakout,
            p_atr_expand=score.p_atr_expand,
            model_version=score.model_version,
            ttl_bars=self._outcome_ttl,
        )
