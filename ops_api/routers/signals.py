"""
Signals router — read-only endpoints for signal ledger data.

All responses include the disclaimer that signals are research data only.

DISCLAIMER: Signals are research observations, not personalized investment advice.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/signals", tags=["signals"])

_DISCLAIMER = (
    "RESEARCH ONLY. Not personalized investment advice. "
    "Signals are quantitative strategy observations; no sizing is implied."
)


class SignalRow(BaseModel):
    signal_id: str
    ts: datetime
    symbol: str
    direction: str
    timeframe: str
    strategy_type: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_r_multiple: float
    outcome: Optional[str] = None
    r_achieved: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    fill_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    fill_latency_ms: Optional[int] = None
    engine_version: str
    disclaimer: str = _DISCLAIMER


class PerformanceSummary(BaseModel):
    resolved_count: int
    win_rate: float
    mean_r_achieved: float
    capital_gate_status: dict
    disclaimer: str = _DISCLAIMER


def _get_service():
    from services.signal_ledger_service import SignalLedgerService
    return SignalLedgerService()


@router.get("/latest", response_model=List[SignalRow])
async def get_latest_signals(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=500),
) -> List[SignalRow]:
    """Return the most recent signals from the signal ledger.

    Signals are research observations. Not personalized investment advice.
    """
    import sqlalchemy as sa

    svc = _get_service()
    if svc._engine is None:
        return []
    try:
        q = "SELECT * FROM signal_ledger ORDER BY ts DESC LIMIT :limit"
        params: dict = {"limit": limit}
        with svc._engine.connect() as conn:
            rows = conn.execute(sa.text(q), params).fetchall()
        result = []
        for row in rows:
            if symbol and row.symbol != symbol:
                continue
            result.append(SignalRow(
                signal_id=row.signal_id,
                ts=row.ts,
                symbol=row.symbol,
                direction=row.direction,
                timeframe=row.timeframe,
                strategy_type=row.strategy_type,
                entry_price=float(row.entry_price),
                stop_price=float(row.stop_price),
                target_price=float(row.target_price),
                risk_r_multiple=float(row.risk_r_multiple),
                outcome=row.outcome,
                r_achieved=float(row.r_achieved) if row.r_achieved is not None else None,
                mfe_pct=float(row.mfe_pct) if row.mfe_pct is not None else None,
                mae_pct=float(row.mae_pct) if row.mae_pct is not None else None,
                fill_price=float(row.fill_price) if row.fill_price is not None else None,
                slippage_bps=float(row.slippage_bps) if row.slippage_bps is not None else None,
                fill_latency_ms=row.fill_latency_ms,
                engine_version=row.engine_version,
            ))
        return result
    except Exception:
        return []


@router.get("/history", response_model=List[SignalRow])
async def get_signal_history(
    symbol: Optional[str] = Query(default=None),
    strategy_type: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    limit: int = Query(default=200, le=1000),
) -> List[SignalRow]:
    """Return historical resolved signals with optional filters.

    Use this endpoint for track-record analysis.
    Signals are research data. Not personalized investment advice.
    """
    import sqlalchemy as sa

    svc = _get_service()
    if svc._engine is None:
        return []
    try:
        conditions = ["outcome IS NOT NULL"]
        params: dict = {"limit": limit}
        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        if strategy_type:
            conditions.append("strategy_type = :strategy_type")
            params["strategy_type"] = strategy_type
        if outcome:
            conditions.append("outcome = :outcome")
            params["outcome"] = outcome
        if since:
            conditions.append("ts >= :since")
            params["since"] = since
        where = " AND ".join(conditions)
        q = f"SELECT * FROM signal_ledger WHERE {where} ORDER BY ts DESC LIMIT :limit"
        with svc._engine.connect() as conn:
            rows = conn.execute(sa.text(q), params).fetchall()
        result = []
        for row in rows:
            result.append(SignalRow(
                signal_id=row.signal_id,
                ts=row.ts,
                symbol=row.symbol,
                direction=row.direction,
                timeframe=row.timeframe,
                strategy_type=row.strategy_type,
                entry_price=float(row.entry_price),
                stop_price=float(row.stop_price),
                target_price=float(row.target_price),
                risk_r_multiple=float(row.risk_r_multiple),
                outcome=row.outcome,
                r_achieved=float(row.r_achieved) if row.r_achieved is not None else None,
                mfe_pct=float(row.mfe_pct) if row.mfe_pct is not None else None,
                mae_pct=float(row.mae_pct) if row.mae_pct is not None else None,
                fill_price=float(row.fill_price) if row.fill_price is not None else None,
                slippage_bps=float(row.slippage_bps) if row.slippage_bps is not None else None,
                fill_latency_ms=row.fill_latency_ms,
                engine_version=row.engine_version,
            ))
        return result
    except Exception:
        return []


@router.get("/performance", response_model=PerformanceSummary)
async def get_signal_performance() -> PerformanceSummary:
    """Return aggregate performance statistics for the signal engine.

    Includes capital gate evaluation (advisory; human approval required for stage promotion).
    All metrics are research-only. Not personalized investment advice.
    """
    import sqlalchemy as sa

    svc = _get_service()
    gates = svc.evaluate_capital_gates()
    resolved_count = gates["resolved_count"]
    mean_r = gates["expectancy_r"]

    win_rate = 0.0
    if svc._engine is not None and resolved_count > 0:
        try:
            with svc._engine.connect() as conn:
                row = conn.execute(sa.text("""
                    SELECT
                        COUNT(*) FILTER (WHERE outcome = 'target_hit') AS wins
                    FROM signal_ledger
                    WHERE outcome IS NOT NULL
                """)).fetchone()
            wins = int(row.wins or 0) if row else 0
            win_rate = wins / resolved_count if resolved_count > 0 else 0.0
        except Exception:
            win_rate = 0.0

    return PerformanceSummary(
        resolved_count=resolved_count,
        win_rate=round(win_rate, 4),
        mean_r_achieved=mean_r,
        capital_gate_status=gates,
    )
