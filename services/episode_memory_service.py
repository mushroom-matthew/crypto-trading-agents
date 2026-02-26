"""Episode memory service: in-memory store and record builder (Runbook 51).

Converts resolved SignalEvents into EpisodeMemoryRecord objects and provides
a simple in-memory store keyed by episode_id.

No database I/O — the store lives in process memory. DB persistence is deferred
to a future runbook.

DISCLAIMER: Records are research telemetry. No trading decisions are derived
automatically from memory content.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from schemas.episode_memory import (
    FAILURE_MODE_TAXONOMY,
    EpisodeMemoryRecord,
)
from schemas.signal_event import SignalEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def _classify_outcome(
    pnl: Optional[float],
    r_achieved: Optional[float],
) -> str:
    """Return 'win', 'loss', or 'neutral' for an episode.

    Precedence:
    1. r_achieved if available (preferred — scale-independent)
    2. pnl sign if r_achieved is absent
    3. 'neutral' if both are absent or zero
    """
    if r_achieved is not None:
        if r_achieved > 0.5:
            return "win"
        if r_achieved < -0.3:
            return "loss"
    if pnl is not None:
        if pnl > 0:
            return "win"
        if pnl < 0:
            return "loss"
    return "neutral"


# ---------------------------------------------------------------------------
# Failure mode detection
# ---------------------------------------------------------------------------

def _detect_failure_modes(
    outcome_class: str,
    trigger_category: Optional[str],
    r_achieved: Optional[float],
    mae: Optional[float],
    mfe: Optional[float],
    mae_pct: Optional[float],
) -> List[str]:
    """Deterministically label failure modes from resolved episode fields.

    Rules applied in priority order; a single episode may carry multiple labels.
    """
    if outcome_class == "win":
        # Winners have no failure modes by definition
        return []

    modes: List[str] = []

    # stop_too_tight_noise_out: loss where MAE dominated the excursion.
    # Proxy: mae/mfe ratio > 3 (the market moved strongly against us relative
    # to any favourable excursion — tight stop, noise-out pattern).
    if outcome_class in ("loss", "neutral"):
        if mae is not None and mfe is not None and mfe > 0:
            if (mae / mfe) > 3.0:
                modes.append("stop_too_tight_noise_out")
        elif mae_pct is not None:
            # Alternative: mae_pct dominates — use if mfe unavailable
            pass  # primary rule above takes precedence

    # late_entry_poor_r_multiple: loss with a bad r_achieved and limited MFE
    # (price barely moved in our favour — we entered late).
    if outcome_class == "loss":
        if (
            r_achieved is not None
            and r_achieved < -0.5
            and mfe is not None
            and mfe < 0.5
        ):
            modes.append("late_entry_poor_r_multiple")

    # low_volume_breakout_failure: breakout-category signals that lost.
    if outcome_class == "loss" and trigger_category == "breakout":
        modes.append("low_volume_breakout_failure")

    # signal_conflict_chop: neutral losses with no clearer label — the market
    # was choppy and the signal conflicted with itself.
    if outcome_class == "neutral" and not modes:
        modes.append("signal_conflict_chop")

    return modes


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_episode_record(
    signal_event: SignalEvent,
    regime_fingerprint: Optional[Dict[str, float]] = None,
    regime_version: Optional[str] = None,
    trade_id: Optional[str] = None,
    exit_ts: Optional[datetime] = None,
    resolution_ts: Optional[datetime] = None,
    pnl: Optional[float] = None,
    r_achieved: Optional[float] = None,
    hold_bars: Optional[int] = None,
    hold_minutes: Optional[float] = None,
    mae: Optional[float] = None,
    mfe: Optional[float] = None,
    mae_pct: Optional[float] = None,
    mfe_pct: Optional[float] = None,
    snapshot_id: Optional[str] = None,
    snapshot_hash: Optional[str] = None,
    stance: Optional[str] = None,
    trigger_category: Optional[str] = None,
) -> EpisodeMemoryRecord:
    """Build an EpisodeMemoryRecord from a resolved SignalEvent.

    The signal_event provides identity and strategy metadata; outcome fields
    (pnl, r_achieved, mae, mfe, …) are supplied by the caller after reconciliation.

    Args:
        signal_event: The emitted signal (immutable after emission).
        regime_fingerprint: Normalized feature dict at signal time (from R55).
        regime_version: FINGERPRINT_VERSION string for the fingerprint.
        trade_id: Optional trade/fill ID linking to the execution record.
        exit_ts: When the position was closed.
        resolution_ts: When this record was reconciled (default: now).
        pnl: Realized P&L for the episode.
        r_achieved: Achieved R-multiple (positive = win, negative = loss).
        hold_bars: Number of bars held.
        hold_minutes: Approximate hold duration in minutes.
        mae: Maximum Adverse Excursion (price units).
        mfe: Maximum Favourable Excursion (price units).
        mae_pct: MAE as a percentage of entry price.
        mfe_pct: MFE as a percentage of entry price.
        snapshot_id: PolicySnapshot ID at signal time (from R49).
        snapshot_hash: PolicySnapshot hash for audit.
        stance: Agent stance string ("long_bias", "neutral", etc.).
        trigger_category: TriggerCondition category (e.g. "breakout", "mean_reversion").

    Returns:
        EpisodeMemoryRecord with outcome_class and failure_modes populated.
    """
    outcome_class = _classify_outcome(pnl=pnl, r_achieved=r_achieved)
    failure_modes = _detect_failure_modes(
        outcome_class=outcome_class,
        trigger_category=trigger_category or getattr(signal_event, "strategy_type", None),
        r_achieved=r_achieved,
        mae=mae,
        mfe=mfe,
        mae_pct=mae_pct,
    )

    return EpisodeMemoryRecord(
        episode_id=str(uuid4()),
        signal_id=signal_event.signal_id,
        trade_id=trade_id,
        entry_ts=signal_event.ts,
        exit_ts=exit_ts,
        resolution_ts=resolution_ts or datetime.now(tz=timezone.utc),
        snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        regime_fingerprint=regime_fingerprint,
        regime_version=regime_version,
        symbol=signal_event.symbol,
        timeframe=signal_event.timeframe,
        playbook_id=signal_event.playbook_id,
        template_id=signal_event.strategy_template_version,
        trigger_category=trigger_category,
        direction=signal_event.direction,
        pnl=pnl,
        r_achieved=r_achieved,
        hold_bars=hold_bars,
        hold_minutes=hold_minutes,
        mae=mae,
        mfe=mfe,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        stance=stance,
        outcome_class=outcome_class,
        failure_modes=failure_modes,
        retrieval_scope=None,
    )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class EpisodeMemoryStore:
    """Thread-unsafe in-memory store for EpisodeMemoryRecord objects.

    Keyed by episode_id. Thread safety is the caller's responsibility.
    DB persistence is deferred — this is intentionally pure-memory for R51.
    """

    def __init__(self) -> None:
        self._records: Dict[str, EpisodeMemoryRecord] = {}

    def add(self, record: EpisodeMemoryRecord) -> None:
        """Insert or overwrite a record by episode_id."""
        self._records[record.episode_id] = record

    def get_by_symbol(self, symbol: str) -> List[EpisodeMemoryRecord]:
        """Return all records for a given symbol, unsorted."""
        return [r for r in self._records.values() if r.symbol == symbol]

    def get_all(self) -> List[EpisodeMemoryRecord]:
        """Return all records across all symbols, unsorted."""
        return list(self._records.values())

    def size(self) -> int:
        """Return total number of records in the store."""
        return len(self._records)
