"""
Signal Outcome Reconciler.

Periodically scans unresolved rows in signal_ledger and resolves them using
subsequent OHLCV candles. Updates each row with outcome, R achieved, MFE, MAE.

DISCLAIMER: Reconciler output is research telemetry only. No trading decisions
are made automatically from reconciler results.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa

from services.signal_ledger_service import SignalLedgerService

logger = logging.getLogger(__name__)

_SELECT_UNRESOLVED = """
SELECT signal_id, ts, valid_until, symbol, direction, timeframe,
       entry_price, stop_price, target_price
FROM signal_ledger
WHERE outcome IS NULL
  AND ts < NOW() - INTERVAL '1 minute'
ORDER BY ts
LIMIT :limit
"""


class SignalOutcomeReconciler:
    """Resolves open signal rows using OHLCV data fetched after signal emission."""

    def __init__(
        self,
        ledger_service: Optional[SignalLedgerService] = None,
        engine=None,
        cadence_seconds: int = 300,
    ):
        self._ledger = ledger_service or SignalLedgerService(engine=engine)
        self._engine = engine or (self._ledger._engine)
        self._cadence = cadence_seconds

    async def run_forever(self) -> None:
        """Run reconciliation loop until cancelled."""
        logger.info("SignalOutcomeReconciler: starting, cadence=%ds", self._cadence)
        while True:
            try:
                resolved = await asyncio.to_thread(self._reconcile_batch, limit=200)
                logger.info("SignalOutcomeReconciler: resolved %d signals", resolved)
            except Exception as exc:
                logger.error("SignalOutcomeReconciler: batch error: %s", exc)
            await asyncio.sleep(self._cadence)

    def _reconcile_batch(self, limit: int = 200) -> int:
        """Reconcile up to `limit` unresolved signals. Returns count resolved."""
        if self._engine is None:
            return 0
        resolved = 0
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sa.text(_SELECT_UNRESOLVED), {"limit": limit}
                ).fetchall()
        except Exception as exc:
            logger.warning("SignalOutcomeReconciler: DB query failed: %s", exc)
            return 0

        for row in rows:
            try:
                outcome = self._resolve_one(row)
                if outcome is not None:
                    resolved += 1
            except Exception as exc:
                logger.warning(
                    "SignalOutcomeReconciler: failed to resolve signal_id=%s: %s",
                    row.signal_id,
                    exc,
                )
        return resolved

    def _resolve_one(self, row) -> Optional[str]:
        """Attempt to resolve a single signal row.

        Returns the outcome string if resolved, None if insufficient data.
        """
        from backtesting.dataset import load_ohlcv

        now_utc = datetime.now(timezone.utc)
        signal_ts = row.ts if row.ts.tzinfo else row.ts.replace(tzinfo=timezone.utc)
        valid_until = row.valid_until if row.valid_until.tzinfo else row.valid_until.replace(tzinfo=timezone.utc)

        fetch_end = min(now_utc, valid_until)
        try:
            df = load_ohlcv(row.symbol, signal_ts, fetch_end, row.timeframe)
        except Exception as exc:
            logger.debug("SignalOutcomeReconciler: OHLCV fetch failed for %s: %s", row.symbol, exc)
            return None

        if df is None or len(df) < 2:
            if now_utc >= valid_until:
                self._ledger.resolve_signal(
                    signal_id=row.signal_id,
                    outcome="expired",
                    outcome_ts=valid_until,
                    r_achieved=0.0,
                    mfe_pct=0.0,
                    mae_pct=0.0,
                )
                return "expired"
            return None

        entry = float(row.entry_price)
        stop = float(row.stop_price)
        target = float(row.target_price)
        direction = row.direction
        risk = abs(entry - stop)

        outcome: Optional[str] = None
        outcome_ts: Optional[datetime] = None
        r_achieved: float = 0.0

        highs = df["high"].values
        lows = df["low"].values
        bar_times = df.index

        mfe = 0.0
        mae = 0.0

        for high, low, bar_ts in zip(highs, lows, bar_times):
            if direction == "long":
                bar_mfe = (high - entry) / entry * 100
                bar_mae = (low - entry) / entry * 100
                mfe = max(mfe, bar_mfe)
                mae = min(mae, bar_mae)

                if high >= target:
                    outcome = "target_hit"
                    outcome_ts = bar_ts
                    r_achieved = (target - entry) / max(risk, 1e-9)
                    break
                if low <= stop:
                    outcome = "stop_hit"
                    outcome_ts = bar_ts
                    r_achieved = (stop - entry) / max(risk, 1e-9)
                    break
            else:  # short
                bar_mfe = (entry - low) / entry * 100
                bar_mae = (high - entry) / entry * 100
                mfe = max(mfe, bar_mfe)
                mae = min(mae, bar_mae)

                if low <= target:
                    outcome = "target_hit"
                    outcome_ts = bar_ts
                    r_achieved = (entry - target) / max(risk, 1e-9)
                    break
                if high >= stop:
                    outcome = "stop_hit"
                    outcome_ts = bar_ts
                    r_achieved = (entry - stop) / max(risk, 1e-9)
                    break

        if outcome is None:
            if now_utc >= valid_until:
                close = float(df["close"].iloc[-1])
                if direction == "long":
                    r_achieved = (close - entry) / max(risk, 1e-9)
                else:
                    r_achieved = (entry - close) / max(risk, 1e-9)
                outcome = "expired"
                outcome_ts = valid_until
            else:
                return None

        self._ledger.resolve_signal(
            signal_id=row.signal_id,
            outcome=outcome,
            outcome_ts=outcome_ts,
            r_achieved=r_achieved,
            mfe_pct=round(mfe, 4),
            mae_pct=round(mae, 4),
        )
        return outcome
