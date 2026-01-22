"""Wallet and reconciliation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func

from app.db.repo import Database
from app.db.models import Wallet as WalletModel, Balance, LedgerEntry
from app.ledger.reconciliation import Reconciler, DriftRecord as ReconDriftRecord
from app.ledger.engine import LedgerEngine
from app.coinbase.client import CoinbaseClient
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wallets", tags=["wallets"])

# Initialize database
_db = Database()
_settings = get_settings()


# Response Schemas
class Wallet(BaseModel):
    """Wallet information."""

    wallet_id: int
    name: str
    currency: Optional[str] = None
    ledger_balance: Decimal
    coinbase_balance: Optional[Decimal] = None
    drift: Optional[Decimal] = None
    tradeable_fraction: Decimal
    type: str


class DriftRecord(BaseModel):
    """Drift between ledger and Coinbase."""

    wallet_id: int
    wallet_name: str
    currency: str
    ledger_balance: Decimal
    coinbase_balance: Decimal
    drift: Decimal
    within_threshold: bool


class ReconciliationReport(BaseModel):
    """Reconciliation report."""

    timestamp: datetime
    total_wallets: int
    drifts_detected: int
    drifts_within_threshold: int
    drifts_exceeding_threshold: int
    records: List[DriftRecord]


class ReconcileRequest(BaseModel):
    """Request to trigger reconciliation."""

    threshold: Optional[Decimal] = Decimal("0.0001")


@router.get("", response_model=List[Wallet])
async def list_wallets():
    """
    List all wallets.

    Returns wallet information including balances and drift status.
    """
    try:
        logger.info("List wallets requested")

        async with _db.session() as session:
            # Query all wallets
            result = await session.execute(select(WalletModel))
            wallets = result.scalars().all()

            wallet_list = []
            for w in wallets:
                # Query balances for this wallet
                balance_result = await session.execute(
                    select(Balance).where(Balance.wallet_id == w.wallet_id)
                )
                balances = balance_result.scalars().all()

                # Get primary currency balance (USD or first available)
                primary_balance = Decimal("0")
                currency = None
                for b in balances:
                    if b.currency == "USD" or currency is None:
                        primary_balance = b.available
                        currency = b.currency

                wallet_list.append(
                    Wallet(
                        wallet_id=w.wallet_id,
                        name=w.name,
                        currency=currency,
                        ledger_balance=primary_balance,
                        coinbase_balance=None,  # Filled during reconciliation
                        drift=None,  # Calculated during reconciliation
                        tradeable_fraction=w.tradeable_fraction,
                        type=w.type.value
                    )
                )

        logger.info(f"Retrieved {len(wallet_list)} wallets")
        return wallet_list

    except Exception as e:
        logger.error("Failed to list wallets: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{wallet_id}", response_model=Wallet)
async def get_wallet(wallet_id: int):
    """
    Get wallet details.

    Returns detailed information for a specific wallet.
    """
    try:
        # TODO: Query specific wallet
        logger.info("Get wallet requested: %d", wallet_id)

        raise HTTPException(status_code=501, detail="Not implemented")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get wallet: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{wallet_id}/transactions")
async def get_wallet_transactions(wallet_id: int, limit: int = 100):
    """
    Get wallet transaction history.

    Returns recent debits/credits for a wallet.
    """
    try:
        # TODO: Query LedgerEntry table
        logger.info("Get wallet transactions requested: %d (limit=%d)", wallet_id, limit)

        # In the future:
        # from app.db.models import LedgerEntry
        # from sqlalchemy import select
        #
        # async with db.session() as session:
        #     result = await session.execute(
        #         select(LedgerEntry)
        #         .where(LedgerEntry.wallet_id == wallet_id)
        #         .order_by(LedgerEntry.posted_at.desc())
        #         .limit(limit)
        #     )
        #     entries = result.scalars().all()
        #     return [
        #         {
        #             "entry_id": e.entry_id,
        #             "currency": e.currency,
        #             "amount": float(e.amount),
        #             "side": e.side.value,
        #             "source": e.source,
        #             "posted_at": e.posted_at
        #         }
        #         for e in entries
        #     ]

        return []

    except Exception as e:
        logger.error("Failed to get wallet transactions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reconcile", response_model=ReconciliationReport)
async def trigger_reconciliation(request: ReconcileRequest):
    """
    Trigger wallet reconciliation.

    Compares ledger balances with Coinbase balances and reports drift.
    """
    try:
        logger.info("Reconciliation requested with threshold: %s", request.threshold)

        # Initialize ledger engine and reconciler
        ledger = LedgerEngine(_db)
        reconciler = Reconciler(_db, ledger)

        # Run reconciliation
        async with CoinbaseClient(_settings) as client:
            recon_report = await reconciler.reconcile(client, threshold=request.threshold)

        # Convert to API response format
        records = [
            DriftRecord(
                wallet_id=d.wallet_id,
                wallet_name=d.wallet_name,
                currency=d.currency,
                ledger_balance=d.ledger_balance,
                coinbase_balance=d.coinbase_balance,
                drift=d.drift,
                within_threshold=d.within_threshold
            )
            for d in recon_report.entries
        ]

        # Calculate summary statistics
        total_wallets = len(set(d.wallet_id for d in recon_report.entries))
        drifts_detected = sum(1 for d in recon_report.entries if abs(d.drift) > 0)
        drifts_within = sum(1 for d in recon_report.entries if d.within_threshold)
        drifts_exceeding = sum(1 for d in recon_report.entries if not d.within_threshold)

        response = ReconciliationReport(
            timestamp=datetime.utcnow(),
            total_wallets=total_wallets,
            drifts_detected=drifts_detected,
            drifts_within_threshold=drifts_within,
            drifts_exceeding_threshold=drifts_exceeding,
            records=records
        )

        logger.info(
            f"Reconciliation complete: {total_wallets} wallets, "
            f"{drifts_detected} drifts detected, {drifts_exceeding} exceeding threshold"
        )

        return response

    except Exception as e:
        logger.error("Failed to trigger reconciliation: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reconcile/history")
async def get_reconciliation_history(limit: int = 50):
    """
    Get reconciliation history.

    Returns past reconciliation reports with drift trends over time.
    """
    try:
        # TODO: Store reconciliation reports in database and query here
        logger.info("Reconciliation history requested (limit=%d)", limit)

        # Placeholder
        return []

    except Exception as e:
        logger.error("Failed to get reconciliation history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
