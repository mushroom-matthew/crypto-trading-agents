"""Episode memory service: in-memory store and record builder (Runbook 51/A3).

Converts resolved SignalEvents into EpisodeMemoryRecord objects and provides
a store keyed by episode_id.  In-memory for session lifetime; optional DB
persistence via persist_episode() / load_recent() (Runbook A3).

DISCLAIMER: Records are research telemetry. No trading decisions are derived
automatically from memory content.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import sqlalchemy as sa

from schemas.episode_memory import (
    FAILURE_MODE_TAXONOMY,
    EpisodeMemoryRecord,
    EpisodeSource,
)
from schemas.signal_event import SignalEvent

logger = logging.getLogger(__name__)

_EPISODE_MEMORY_ENABLED = os.environ.get("EPISODE_MEMORY_DB_ENABLED", "1") == "1"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episode_memory (
    episode_id      TEXT PRIMARY KEY,
    signal_id       TEXT,
    symbol          TEXT NOT NULL,
    timeframe       TEXT,
    playbook_id     TEXT,
    template_id     TEXT,
    direction       TEXT,
    outcome_class   TEXT NOT NULL,
    r_achieved      REAL,
    mfe_pct         REAL,
    mae_pct         REAL,
    hold_bars       INTEGER,
    regime_fingerprint_hash TEXT,
    regime_fingerprint_json TEXT,
    failure_modes_json      TEXT,
    entry_ts        TIMESTAMP WITH TIME ZONE,
    exit_ts         TIMESTAMP WITH TIME ZONE,
    episode_source  TEXT DEFAULT 'live',
    session_id      TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
"""

# Add columns to existing tables that pre-date them.
_MIGRATE_EPISODE_SOURCE_SQL = """
ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS episode_source TEXT DEFAULT 'live'
"""
_MIGRATE_REGIME_FP_JSON_SQL = """
ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS regime_fingerprint_json TEXT
"""
_MIGRATE_SESSION_ID_SQL = """
ALTER TABLE episode_memory ADD COLUMN IF NOT EXISTS session_id TEXT
"""

_INSERT_EPISODE_SQL = """
INSERT INTO episode_memory (
    episode_id, signal_id, symbol, timeframe, playbook_id, template_id,
    direction, outcome_class, r_achieved, mfe_pct, mae_pct, hold_bars,
    regime_fingerprint_hash, regime_fingerprint_json, failure_modes_json,
    entry_ts, exit_ts, episode_source, session_id
) VALUES (
    :episode_id, :signal_id, :symbol, :timeframe, :playbook_id, :template_id,
    :direction, :outcome_class, :r_achieved, :mfe_pct, :mae_pct, :hold_bars,
    :regime_fingerprint_hash, :regime_fingerprint_json, :failure_modes_json,
    :entry_ts, :exit_ts, :episode_source, :session_id
)
ON CONFLICT (episode_id) DO NOTHING
"""


def _make_episode_engine(db_url: Optional[str] = None):
    """Create a synchronous SQLAlchemy engine for episode_memory."""
    url = db_url or os.environ.get("DB_DSN", "")
    if not url:
        return None
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("asyncpg://", "postgresql://")
    try:
        engine = sa.create_engine(url, pool_pre_ping=True, pool_size=2)
        return engine
    except Exception as exc:
        logger.warning("episode_memory: could not create engine: %s", exc)
        return None


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
    episode_source: "EpisodeSource" = "live",
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
        episode_source=episode_source,
        retrieval_scope=None,
    )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class EpisodeMemoryStore:
    """In-memory store for EpisodeMemoryRecord objects with optional DB persistence.

    Keyed by episode_id. Thread safety is the caller's responsibility.
    DB persistence is enabled when DB_DSN is set and EPISODE_MEMORY_DB_ENABLED=1.
    """

    def __init__(self, engine=None, db_url: Optional[str] = None) -> None:
        self._records: Dict[str, EpisodeMemoryRecord] = {}
        self._engine = engine
        self._table_ensured = False
        if self._engine is None and _EPISODE_MEMORY_ENABLED:
            self._engine = _make_episode_engine(db_url)

    def _ensure_table(self) -> None:
        """Create the episode_memory table if it doesn't exist and apply column migrations."""
        if self._table_ensured or self._engine is None:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(sa.text(_CREATE_TABLE_SQL))
                # Apply column migrations for tables created before these columns existed.
                conn.execute(sa.text(_MIGRATE_EPISODE_SOURCE_SQL))
                conn.execute(sa.text(_MIGRATE_REGIME_FP_JSON_SQL))
                conn.execute(sa.text(_MIGRATE_SESSION_ID_SQL))
            self._table_ensured = True
        except Exception as exc:
            logger.warning("episode_memory: table creation failed (non-fatal): %s", exc)

    def add(self, record: EpisodeMemoryRecord) -> None:
        """Insert or overwrite a record by episode_id."""
        self._records[record.episode_id] = record

    def persist_episode(
        self,
        record: EpisodeMemoryRecord,
        session_id: Optional[str] = None,
    ) -> None:
        """Write an episode record to the DB (non-fatal if DB unavailable).

        R82: accepts session_id for cross-session indexing.
        Also stores regime_fingerprint_json so retrieval service can score
        episodes by regime distance across sessions.
        """
        if self._engine is None:
            return
        self._ensure_table()
        try:
            import hashlib
            # Compute a short hash of regime_fingerprint for indexed lookup
            fp_hash: Optional[str] = None
            fp_json: Optional[str] = None
            if record.regime_fingerprint:
                canonical = json.dumps(record.regime_fingerprint, sort_keys=True, default=str)
                fp_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                fp_json = canonical  # store full JSON for retrieval scoring

            with self._engine.begin() as conn:
                conn.execute(
                    sa.text(_INSERT_EPISODE_SQL),
                    {
                        "episode_id": record.episode_id,
                        "signal_id": record.signal_id,
                        "symbol": record.symbol,
                        "timeframe": record.timeframe,
                        "playbook_id": record.playbook_id,
                        "template_id": record.template_id,
                        "direction": record.direction,
                        "outcome_class": record.outcome_class,
                        "r_achieved": float(record.r_achieved) if record.r_achieved is not None else None,
                        "mfe_pct": float(record.mfe_pct) if record.mfe_pct is not None else None,
                        "mae_pct": float(record.mae_pct) if record.mae_pct is not None else None,
                        "hold_bars": record.hold_bars,
                        "regime_fingerprint_hash": fp_hash,
                        "regime_fingerprint_json": fp_json,
                        "failure_modes_json": json.dumps(record.failure_modes) if record.failure_modes else None,
                        "entry_ts": record.entry_ts,
                        "exit_ts": record.exit_ts,
                        "episode_source": record.episode_source,
                        "session_id": session_id,
                    },
                )
            logger.info(
                "episode_memory: persisted episode_id=%s symbol=%s outcome=%s r=%.2f",
                record.episode_id,
                record.symbol,
                record.outcome_class,
                record.r_achieved or 0.0,
            )
        except Exception as exc:
            logger.warning("episode_memory: persist failed (non-fatal): %s", exc)

    def load_recent(self, symbol: str, limit: int = 50) -> List[EpisodeMemoryRecord]:
        """Load the most recent episode records for a symbol from DB.

        Returns an empty list if DB is unavailable or table doesn't exist.
        Only loads the core fields needed for retrieval scoring.
        """
        if self._engine is None:
            return []
        self._ensure_table()
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sa.text("""
                        SELECT episode_id, signal_id, symbol, timeframe, playbook_id,
                               template_id, direction, outcome_class, r_achieved,
                               mfe_pct, mae_pct, hold_bars, failure_modes_json,
                               regime_fingerprint_json,
                               entry_ts, exit_ts,
                               COALESCE(episode_source, 'live') AS episode_source
                        FROM episode_memory
                        WHERE symbol = :symbol
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """),
                    {"symbol": symbol, "limit": limit},
                ).fetchall()
        except Exception as exc:
            logger.debug("episode_memory: load_recent failed (non-fatal): %s", exc)
            return []

        records: List[EpisodeMemoryRecord] = []
        for row in rows:
            try:
                failure_modes = json.loads(row.failure_modes_json) if row.failure_modes_json else []
                # R82: restore regime_fingerprint from JSON for retrieval scoring
                regime_fp: Optional[Dict[str, float]] = None
                _fp_json = getattr(row, "regime_fingerprint_json", None)
                if _fp_json:
                    try:
                        regime_fp = {k: float(v) for k, v in json.loads(_fp_json).items()}
                    except Exception:
                        regime_fp = None
                records.append(EpisodeMemoryRecord(
                    episode_id=row.episode_id,
                    signal_id=row.signal_id,
                    symbol=row.symbol,
                    timeframe=row.timeframe,
                    playbook_id=row.playbook_id,
                    template_id=row.template_id,
                    direction=row.direction,
                    outcome_class=row.outcome_class,
                    r_achieved=float(row.r_achieved) if row.r_achieved is not None else None,
                    mfe_pct=float(row.mfe_pct) if row.mfe_pct is not None else None,
                    mae_pct=float(row.mae_pct) if row.mae_pct is not None else None,
                    hold_bars=row.hold_bars,
                    failure_modes=failure_modes,
                    regime_fingerprint=regime_fp,
                    entry_ts=row.entry_ts,
                    exit_ts=row.exit_ts,
                    episode_source=getattr(row, "episode_source", None) or "live",
                ))
            except Exception as exc:
                logger.debug("episode_memory: skipping malformed row: %s", exc)
        return records

    def get_by_symbol(self, symbol: str) -> List[EpisodeMemoryRecord]:
        """Return all in-memory records for a given symbol, unsorted."""
        return [r for r in self._records.values() if r.symbol == symbol]

    def get_all(self) -> List[EpisodeMemoryRecord]:
        """Return all in-memory records across all symbols, unsorted."""
        return list(self._records.values())

    def size(self) -> int:
        """Return total number of in-memory records in the store."""
        return len(self._records)
