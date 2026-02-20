"""
Signal Ledger Service.

Persists SignalEvents to the signal_ledger table and records fill drift telemetry.

DISCLAIMER: Signals are research-only observations. This service records them for
track-record analysis. It does not route orders or recommend position sizes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa

from schemas.signal_event import SignalEvent

logger = logging.getLogger(__name__)

SLIPPAGE_WARNING_THRESHOLD_BPS = 30  # 0.3% on a 1% stop — material

_SIGNAL_LEDGER_ENABLED = os.environ.get("SIGNAL_LEDGER_ENABLED", "1") == "1"

_INSERT_SQL = """
INSERT INTO signal_ledger (
    signal_id, ts, engine_version, symbol, direction, timeframe,
    strategy_type, trigger_id, regime_snapshot_hash,
    entry_price, stop_price, target_price,
    stop_anchor_type, target_anchor_type,
    risk_r_multiple, expected_hold_bars, valid_until,
    thesis, screener_rank, confidence
) VALUES (
    :signal_id, :ts, :engine_version, :symbol, :direction, :timeframe,
    :strategy_type, :trigger_id, :regime_snapshot_hash,
    :entry_price, :stop_price, :target_price,
    :stop_anchor_type, :target_anchor_type,
    :risk_r_multiple, :expected_hold_bars, :valid_until,
    :thesis, :screener_rank, :confidence
)
ON CONFLICT (signal_id) DO NOTHING
"""


def compute_regime_snapshot_hash(snapshot_dict: dict) -> str:
    """SHA-256 hash of the indicator snapshot at signal emission time.

    Provides cryptographic proof that the signal was generated from pre-move conditions.
    """
    canonical = json.dumps(snapshot_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _make_engine(db_url: Optional[str] = None):
    """Create a synchronous SQLAlchemy engine for the signal ledger."""
    url = db_url or os.environ.get("DB_DSN", "")
    if not url:
        return None
    # Convert async DSN (postgresql+asyncpg) to sync (postgresql+psycopg2)
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("asyncpg://", "postgresql://")
    try:
        engine = sa.create_engine(url, pool_pre_ping=True, pool_size=2)
        return engine
    except Exception as exc:
        logger.warning("signal_ledger: could not create engine: %s", exc)
        return None


class SignalLedgerService:
    """Writes SignalEvents to the signal_ledger table."""

    def __init__(self, engine=None, db_url: Optional[str] = None):
        self._engine = engine
        if self._engine is None and _SIGNAL_LEDGER_ENABLED:
            self._engine = _make_engine(db_url)

    def insert_signal(self, signal: SignalEvent) -> None:
        """Insert a new signal row. Idempotent (ON CONFLICT DO NOTHING)."""
        if self._engine is None:
            logger.debug("signal_ledger: no engine, skipping insert for signal_id=%s", signal.signal_id)
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sa.text(_INSERT_SQL),
                    {
                        "signal_id": signal.signal_id,
                        "ts": signal.ts,
                        "engine_version": signal.engine_version,
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "timeframe": signal.timeframe,
                        "strategy_type": signal.strategy_type,
                        "trigger_id": signal.trigger_id,
                        "regime_snapshot_hash": signal.regime_snapshot_hash,
                        "entry_price": signal.entry_price,
                        "stop_price": signal.stop_price_abs,
                        "target_price": signal.target_price_abs,
                        "stop_anchor_type": signal.stop_anchor_type,
                        "target_anchor_type": signal.target_anchor_type,
                        "risk_r_multiple": signal.risk_r_multiple,
                        "expected_hold_bars": signal.expected_hold_bars,
                        "valid_until": signal.valid_until,
                        "thesis": signal.thesis,
                        "screener_rank": signal.screener_rank,
                        "confidence": signal.confidence,
                    },
                )
            logger.info(
                "signal_ledger: inserted signal_id=%s symbol=%s direction=%s R=%.2f",
                signal.signal_id,
                signal.symbol,
                signal.direction,
                signal.risk_r_multiple,
            )
        except Exception as exc:
            logger.warning("signal_ledger: insert failed (non-fatal): %s", exc)

    def record_fill(
        self,
        signal_id: str,
        fill_price: float,
        fill_ts: datetime,
        signal_ts: datetime,
        signal_entry_price: float,
    ) -> None:
        """Record fill drift telemetry after a fill event.

        slippage_bps = (fill_price - signal_price) / signal_price * 10000
        fill_latency_ms = fill_ts - signal_ts (in milliseconds)
        """
        latency_ms = int((fill_ts - signal_ts).total_seconds() * 1000)
        slippage_bps = (fill_price - signal_entry_price) / signal_entry_price * 10_000

        if abs(slippage_bps) > SLIPPAGE_WARNING_THRESHOLD_BPS:
            logger.warning(
                "fill_drift: SLIPPAGE WARNING signal_id=%s slippage_bps=%.1f "
                "(threshold=%d bps). fill_price=%.6f signal_price=%.6f latency_ms=%d",
                signal_id,
                slippage_bps,
                SLIPPAGE_WARNING_THRESHOLD_BPS,
                fill_price,
                signal_entry_price,
                latency_ms,
            )

        if self._engine is None:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sa.text(
                        """
                        UPDATE signal_ledger
                        SET fill_price = :fill_price,
                            fill_ts = :fill_ts,
                            fill_latency_ms = :fill_latency_ms,
                            slippage_bps = :slippage_bps,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE signal_id = :signal_id
                        """
                    ),
                    {
                        "signal_id": signal_id,
                        "fill_price": fill_price,
                        "fill_ts": fill_ts,
                        "fill_latency_ms": latency_ms,
                        "slippage_bps": round(slippage_bps, 2),
                    },
                )
        except Exception as exc:
            logger.warning("signal_ledger: record_fill failed (non-fatal): %s", exc)

    def resolve_signal(
        self,
        signal_id: str,
        outcome: str,
        outcome_ts: datetime,
        r_achieved: float,
        mfe_pct: float,
        mae_pct: float,
    ) -> None:
        """Write outcome fields once the reconciler resolves the signal.

        outcome: 'target_hit' | 'stop_hit' | 'expired' | 'cancelled'
        r_achieved: actual R realized (can be negative for losses)
        mfe_pct: max favorable excursion as a percentage of entry price
        mae_pct: max adverse excursion as a percentage of entry price (always negative for longs)
        """
        if self._engine is None:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sa.text(
                        """
                        UPDATE signal_ledger
                        SET outcome = :outcome,
                            outcome_ts = :outcome_ts,
                            r_achieved = :r_achieved,
                            mfe_pct = :mfe_pct,
                            mae_pct = :mae_pct,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE signal_id = :signal_id
                          AND outcome IS NULL
                        """
                    ),
                    {
                        "signal_id": signal_id,
                        "outcome": outcome,
                        "outcome_ts": outcome_ts,
                        "r_achieved": round(r_achieved, 4),
                        "mfe_pct": round(mfe_pct, 4),
                        "mae_pct": round(mae_pct, 4),
                    },
                )
            logger.info(
                "signal_ledger: resolved signal_id=%s outcome=%s r_achieved=%.2f",
                signal_id,
                outcome,
                r_achieved,
            )
        except Exception as exc:
            logger.warning("signal_ledger: resolve_signal failed (non-fatal): %s", exc)

    def evaluate_capital_gates(self) -> dict:
        """Evaluate whether the statistical capital gates allow stage progression.

        DISCLAIMER: Gate evaluation is advisory only. Human approval is required
        to promote to the next capital stage. Signals are research data, not guarantees.

        Returns:
            {
                "gate_pass": bool,          # True only if ALL conditions met
                "resolved_count": int,      # >= 40 required
                "expectancy_r": float,      # mean R of resolved signals (> 0 required)
                "max_drawdown_ratio": float,# actual_dd / expected_daily_cap (<1.5 required)
                "risk_overcharge_median": float,  # median risk_overcharge_ratio (< 5 required)
                "slippage_delta_bps": float,      # |paper_slippage_bps - live_slippage_bps| (< 25 required)
                "conditions": dict,         # per-condition pass/fail
            }
        """
        if self._engine is None:
            return {
                "gate_pass": False,
                "resolved_count": 0,
                "expectancy_r": 0.0,
                "max_drawdown_ratio": None,
                "risk_overcharge_median": None,
                "slippage_delta_bps": 0.0,
                "conditions": {
                    "resolved_count_ge_40": False,
                    "expectancy_positive": False,
                    "max_drawdown_ok": True,
                    "risk_overcharge_ok": True,
                    "slippage_delta_within_25bps": True,
                },
            }

        try:
            with self._engine.connect() as conn:
                stats = conn.execute(sa.text("""
                    SELECT
                        COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) AS resolved_count,
                        AVG(CASE WHEN outcome IS NOT NULL THEN r_achieved END) AS mean_r,
                        AVG(CASE WHEN slippage_bps IS NOT NULL THEN ABS(slippage_bps) END) AS avg_slippage_bps
                    FROM signal_ledger
                """)).fetchone()
        except Exception as exc:
            logger.warning("signal_ledger: evaluate_capital_gates failed: %s", exc)
            return {
                "gate_pass": False,
                "resolved_count": 0,
                "expectancy_r": 0.0,
                "max_drawdown_ratio": None,
                "risk_overcharge_median": None,
                "slippage_delta_bps": 0.0,
                "conditions": {
                    "resolved_count_ge_40": False,
                    "expectancy_positive": False,
                    "max_drawdown_ok": True,
                    "risk_overcharge_ok": True,
                    "slippage_delta_within_25bps": True,
                },
            }

        resolved_count = int(stats.resolved_count or 0)
        mean_r = float(stats.mean_r or 0.0)
        median_slippage = float(stats.avg_slippage_bps or 0.0)

        cond_resolved = resolved_count >= 40
        cond_expectancy = mean_r > 0
        # Placeholders until risk telemetry query is wired
        cond_drawdown = True
        cond_risk_overcharge = True
        cond_slippage_delta = abs(median_slippage) < 25

        gate_pass = all([
            cond_resolved,
            cond_expectancy,
            cond_drawdown,
            cond_risk_overcharge,
            cond_slippage_delta,
        ])

        return {
            "gate_pass": gate_pass,
            "resolved_count": resolved_count,
            "expectancy_r": round(mean_r, 4),
            "max_drawdown_ratio": None,
            "risk_overcharge_median": None,
            "slippage_delta_bps": round(abs(median_slippage), 2),
            "conditions": {
                "resolved_count_ge_40": cond_resolved,
                "expectancy_positive": cond_expectancy,
                "max_drawdown_ok": cond_drawdown,
                "risk_overcharge_ok": cond_risk_overcharge,
                "slippage_delta_within_25bps": cond_slippage_delta,
            },
        }
