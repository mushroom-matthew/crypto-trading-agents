"""Temporal activities for persisting fills into the real ledger."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from temporalio import activity

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import Wallet, WalletType
from app.db.repo import Database
from app.ledger.engine import LedgerEngine, Posting


LOG = get_logger(__name__)


class PersistFillParams(BaseModel):
    """Payload passed from the ExecutionLedgerWorkflow when a fill is recorded."""

    fill: dict[str, Any]
    workflow_id: str
    sequence: int
    recorded_at: float
    trading_wallet_id: Optional[int] = Field(default=None)
    trading_wallet_name: Optional[str] = Field(default=None)
    equity_wallet_name: str = Field(default="system_equity")


_STATE: Optional[tuple[str, Settings, Database, LedgerEngine]] = None


def _get_state() -> tuple[Settings, Database, LedgerEngine]:
    global _STATE
    settings = get_settings()
    db_dsn = settings.ledger_db_dsn or settings.db_dsn

    if _STATE is None or _STATE[0] != db_dsn:
        effective_settings = settings.model_copy(update={"db_dsn": db_dsn})
        database = Database(effective_settings)
        ledger = LedgerEngine(database)
        _STATE = (db_dsn, effective_settings, database, ledger)

    _, state_settings, database, ledger = _STATE
    return state_settings, database, ledger


async def _get_wallet_by_id_or_name(
    wallet_id: Optional[int],
    wallet_name: Optional[str],
    *,
    default_type: WalletType,
) -> Wallet:
    settings, database, _ = _get_state()

    async with database.session() as session:
        wallet: Optional[Wallet] = None
        if wallet_id:
            wallet = await session.get(Wallet, wallet_id)
        if not wallet and wallet_name:
            result = await session.execute(select(Wallet).where(Wallet.name == wallet_name))
            wallet = result.scalar_one_or_none()
        if wallet:
            return wallet

    async with database.session() as session:
        wallet = Wallet(
            name=wallet_name or f"{default_type.value.lower()}-{settings.coinbase_portfolio_id or 'default'}",
            coinbase_account_id=None,
            portfolio_id=settings.coinbase_portfolio_id,
            type=default_type,
            tradeable_fraction=Decimal("1.0") if default_type == WalletType.COINBASE_TRADING else Decimal("0"),
        )
        session.add(wallet)
        await session.flush()
        return wallet


async def _ensure_equity_wallet(name: str) -> Wallet:
    _, database, _ = _get_state()

    async with database.session() as session:
        result = await session.execute(select(Wallet).where(Wallet.name == name))
        wallet = result.scalar_one_or_none()
        if wallet:
            return wallet

    async with database.session() as session:
        wallet = Wallet(
            name=name,
            coinbase_account_id=None,
            portfolio_id=None,
            type=WalletType.EXTERNAL,
            tradeable_fraction=Decimal("0"),
        )
        session.add(wallet)
        await session.flush()
        return wallet


@activity.defn
async def persist_fill_activity(payload: dict[str, Any]) -> None:
    """Write fill data into the double-entry ledger using real database tables."""

    params = PersistFillParams.model_validate(payload)
    fill = params.fill

    settings, database, ledger = _get_state()

    base_currency, quote_currency = fill["symbol"].split("/")
    qty = Decimal(str(fill["qty"]))
    fill_price = Decimal(str(fill.get("fill_price", fill.get("price", 0))))
    notional = Decimal(str(fill.get("cost", qty * fill_price)))

    trading_wallet = await _get_wallet_by_id_or_name(
        params.trading_wallet_id or settings.ledger_trading_wallet_id,
        params.trading_wallet_name or settings.ledger_trading_wallet_name,
        default_type=WalletType.COINBASE_TRADING,
    )
    equity_wallet = await _ensure_equity_wallet(params.equity_wallet_name or settings.ledger_equity_wallet_name)

    fill_key = f"{params.workflow_id}:fill:{params.sequence}"
    postings: list[Posting] = []

    if fill["side"].upper() == "BUY":
        postings.extend(
            [
                Posting(
                    wallet_id=trading_wallet.wallet_id,
                    currency=base_currency,
                    amount=qty,
                    side=_LEDGER_SIDE_DEBIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:base:wallet",
                ),
                Posting(
                    wallet_id=equity_wallet.wallet_id,
                    currency=base_currency,
                    amount=qty,
                    side=_LEDGER_SIDE_CREDIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:base:equity",
                ),
                Posting(
                    wallet_id=equity_wallet.wallet_id,
                    currency=quote_currency,
                    amount=notional,
                    side=_LEDGER_SIDE_DEBIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:quote:equity",
                ),
                Posting(
                    wallet_id=trading_wallet.wallet_id,
                    currency=quote_currency,
                    amount=notional,
                    side=_LEDGER_SIDE_CREDIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:quote:wallet",
                ),
            ]
        )
    else:
        postings.extend(
            [
                Posting(
                    wallet_id=trading_wallet.wallet_id,
                    currency=base_currency,
                    amount=qty,
                    side=_LEDGER_SIDE_CREDIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:base:wallet",
                ),
                Posting(
                    wallet_id=equity_wallet.wallet_id,
                    currency=base_currency,
                    amount=qty,
                    side=_LEDGER_SIDE_DEBIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:base:equity",
                ),
                Posting(
                    wallet_id=equity_wallet.wallet_id,
                    currency=quote_currency,
                    amount=notional,
                    side=_LEDGER_SIDE_CREDIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:quote:equity",
                ),
                Posting(
                    wallet_id=trading_wallet.wallet_id,
                    currency=quote_currency,
                    amount=notional,
                    side=_LEDGER_SIDE_DEBIT,
                    source="workflow_fill",
                    idempotency_key=f"{fill_key}:quote:wallet",
                ),
            ]
        )

    try:
        await ledger.post_double_entry(postings)
    except Exception as exc:  # pragma: no cover - logged for operational visibility
        LOG.error("Failed to post ledger entries", fill=fill, error=str(exc))
        raise


# Lazy import to avoid circular dependency
from app.db.models import LedgerSide  # noqa: E402

_LEDGER_SIDE_DEBIT = LedgerSide.debit
_LEDGER_SIDE_CREDIT = LedgerSide.credit


__all__ = ["persist_fill_activity", "PersistFillParams"]
