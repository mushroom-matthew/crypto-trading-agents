"""
SetupOutcomeReconciler — labels open SetupEvent rows after TTL.

Outcome rules (evaluated in order, first match wins):
1. hit_1r: Within ttl_bars after break_attempt ts, price reaches entry + 1R.
2. hit_stop: Within ttl_bars, price reaches stop level.
3. ttl_expired: Neither occurred within ttl_bars.

Labels are immutable once written (no updates after first resolution).

DISCLAIMER: Reconciler output is research telemetry only. Not investment advice.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import sqlalchemy as sa

logger = logging.getLogger(__name__)

_SELECT_OPEN_BREAK_ATTEMPTS = """
SELECT
    setup_event_id, setup_chain_id, ts, symbol, timeframe,
    compression_range_high, compression_range_low,
    compression_atr_at_detection, ttl_bars
FROM setup_event_ledger
WHERE state = 'break_attempt'
  AND outcome IS NULL
ORDER BY ts
LIMIT :limit
"""


class SetupOutcomeReconciler:
    """Labels open SetupEvent rows (break_attempt state) after TTL using OHLCV."""

    def __init__(
        self,
        engine=None,
        cadence_seconds: int = 300,
    ):
        self._engine = engine
        self._cadence = cadence_seconds

    async def run_forever(self) -> None:
        """Run reconciliation loop until cancelled."""
        logger.info("SetupOutcomeReconciler: starting, cadence=%ds", self._cadence)
        while True:
            try:
                if self._engine is not None:
                    from backtesting.dataset import load_ohlcv
                    resolved = await asyncio.to_thread(
                        self.reconcile_open_events,
                        datetime.now(timezone.utc),
                        load_ohlcv,
                    )
                    logger.info("SetupOutcomeReconciler: resolved %d events", resolved)
            except Exception as exc:
                logger.error("SetupOutcomeReconciler: batch error: %s", exc)
            await asyncio.sleep(self._cadence)

    def reconcile_open_events(
        self,
        cutoff_ts: datetime,
        ohlcv_provider: Callable,
        limit: int = 200,
    ) -> int:
        """Reconcile all open break_attempt rows.

        Args:
            cutoff_ts: Only process rows where the TTL has elapsed as of this time.
            ohlcv_provider: Function matching load_ohlcv signature.
            limit: Max rows per batch.

        Returns:
            Count of rows labeled.
        """
        if self._engine is None:
            return 0

        resolved = 0
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sa.text(_SELECT_OPEN_BREAK_ATTEMPTS), {"limit": limit}
                ).fetchall()
        except Exception as exc:
            logger.warning("SetupOutcomeReconciler: DB query failed: %s", exc)
            return 0

        for row in rows:
            try:
                ts = row.ts if row.ts.tzinfo else row.ts.replace(tzinfo=timezone.utc)
                ttl_bars = row.ttl_bars or 48
                # Use 1h as proxy timeframe if needed; real TTL is bar-based
                ttl_end = ts + timedelta(hours=ttl_bars)
                if cutoff_ts < ttl_end:
                    continue  # TTL not elapsed yet

                try:
                    df = ohlcv_provider(row.symbol, ts, cutoff_ts, row.timeframe)
                except Exception as exc:
                    logger.debug("SetupOutcomeReconciler: OHLCV failed for %s: %s", row.symbol, exc)
                    continue

                if df is None or len(df) < 2:
                    outcome = "ttl_expired"
                    outcome_ts = ttl_end
                    mfe_pct = mae_pct = r_achieved = 0.0
                    bars_to_outcome = ttl_bars
                else:
                    outcome, mfe_pct, mae_pct, r_achieved, bars_to_outcome = self._determine_outcome(
                        row, df, ttl_bars
                    )
                    outcome_ts = ts + timedelta(hours=bars_to_outcome)

                self._write_outcome(
                    row.setup_event_id, outcome, outcome_ts,
                    mfe_pct, mae_pct, r_achieved, bars_to_outcome
                )
                resolved += 1
            except Exception as exc:
                logger.warning(
                    "SetupOutcomeReconciler: failed for setup_event_id=%s: %s",
                    row.setup_event_id, exc
                )
        return resolved

    def _determine_outcome(
        self,
        row,
        ohlcv: pd.DataFrame,
        ttl_bars: int,
    ) -> tuple:
        """Returns (outcome, mfe_pct, mae_pct, r_achieved, bars_to_outcome)."""
        # Use compression range as entry/stop proxy
        if row.compression_range_high is None or row.compression_range_low is None:
            return "ttl_expired", 0.0, 0.0, 0.0, ttl_bars

        range_high = float(row.compression_range_high)
        range_low = float(row.compression_range_low)
        atr = float(row.compression_atr_at_detection or (range_high - range_low))

        # Determine direction from break
        close = ohlcv["close"].iloc[0] if len(ohlcv) > 0 else range_high
        is_upside = close > range_high

        if is_upside:
            entry = range_high
            stop = range_low  # stop at compression low
            target_1r = entry + atr  # 1R above
        else:
            entry = range_low
            stop = range_high
            target_1r = entry - atr

        risk = abs(entry - stop)
        mfe = 0.0
        mae = 0.0

        highs = ohlcv["high"].values[:ttl_bars]
        lows = ohlcv["low"].values[:ttl_bars]

        for i, (high, low) in enumerate(zip(highs, lows)):
            if is_upside:
                mfe = max(mfe, (high - entry) / max(entry, 1e-9) * 100)
                mae = min(mae, (low - entry) / max(entry, 1e-9) * 100)
                if high >= target_1r:
                    r = (target_1r - entry) / max(risk, 1e-9)
                    return "hit_1r", round(mfe, 4), round(mae, 4), round(r, 4), i + 1
                if low <= stop:
                    r = (stop - entry) / max(risk, 1e-9)
                    return "hit_stop", round(mfe, 4), round(mae, 4), round(r, 4), i + 1
            else:
                mfe = max(mfe, (entry - low) / max(entry, 1e-9) * 100)
                mae = min(mae, (high - entry) / max(entry, 1e-9) * 100)
                if low <= target_1r:
                    r = (entry - target_1r) / max(risk, 1e-9)
                    return "hit_1r", round(mfe, 4), round(mae, 4), round(r, 4), i + 1
                if high >= stop:
                    r = (entry - stop) / max(risk, 1e-9)
                    return "hit_stop", round(mfe, 4), round(mae, 4), round(r, 4), i + 1

        return "ttl_expired", round(mfe, 4), round(mae, 4), 0.0, ttl_bars

    def _write_outcome(
        self,
        setup_event_id: str,
        outcome: str,
        outcome_ts: datetime,
        mfe_pct: float,
        mae_pct: float,
        r_achieved: float,
        bars_to_outcome: int,
    ) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(sa.text("""
                    UPDATE setup_event_ledger
                    SET outcome = :outcome,
                        outcome_ts = :outcome_ts,
                        mfe_pct = :mfe_pct,
                        mae_pct = :mae_pct,
                        r_achieved = :r_achieved,
                        bars_to_outcome = :bars_to_outcome
                    WHERE setup_event_id = :id
                      AND outcome IS NULL
                """), {
                    "id": setup_event_id,
                    "outcome": outcome,
                    "outcome_ts": outcome_ts,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "r_achieved": r_achieved,
                    "bars_to_outcome": bars_to_outcome,
                })
        except Exception as exc:
            logger.warning("SetupOutcomeReconciler: write failed for %s: %s", setup_event_id, exc)
