"""Ledger reconciliation and seeding routines."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.coinbase import accounts
from app.coinbase.client import CoinbaseClient
from app.core.errors import LedgerError
from app.core.logging import get_logger
from app.db.models import (
    Balance,
    LedgerEntry,
    LedgerSide,
    Wallet,
    WalletType,
)
from app.db.repo import Database
from app.ledger.engine import LedgerEngine, Posting


LOG = get_logger(__name__)


@dataclass(slots=True)
class DriftRecord:
    wallet_id: int
    wallet_name: str
    currency: str
    ledger_balance: Decimal
    coinbase_balance: Decimal
    drift: Decimal
    within_threshold: bool


class ReconciliationReport(BaseModel):
    """Structured reconciliation output."""

    entries: list[DriftRecord]

    @property
    def has_drift(self) -> bool:
        return any(not entry.within_threshold for entry in self.entries)


class Reconciler:
    """Synchronise ledger snapshots with Coinbase account balances."""

    def __init__(self, db: Database, ledger: LedgerEngine) -> None:
        self._db = db
        self._ledger = ledger

    async def seed_from_coinbase(self, client: CoinbaseClient) -> None:
        """Create wallets and ledger entries for Coinbase accounts."""

        all_accounts = await accounts.list_accounts(client)
        async with self._db.session() as session:
            equity_wallet = await self._ensure_equity_wallet(session)

        for account in all_accounts:
            balances = await accounts.get_balances(client, account.uuid)
            async with self._db.session() as session:
                wallet = await self._ensure_wallet(session, account)
                for balance in balances.balances:
                    if balance.value <= 0:
                        continue
                    session.add(
                        Balance(
                            wallet_id=wallet.wallet_id,
                            currency=balance.currency,
                            available=balance.value,
                            hold=Decimal("0"),
                        )
                    )
                await session.flush()

            for balance in balances.balances:
                if balance.value <= 0:
                    continue
                seed_key = f"seed:{account.uuid}:{balance.currency}"
                postings = [
                    Posting(
                        wallet_id=wallet.wallet_id,
                        currency=balance.currency,
                        amount=balance.value,
                        side=LedgerSide.debit,
                        source="coinbase_seed",
                        idempotency_key=f"{seed_key}:asset",
                    ),
                    Posting(
                        wallet_id=equity_wallet.wallet_id,
                        currency=balance.currency,
                        amount=balance.value,
                        side=LedgerSide.credit,
                        source="coinbase_seed",
                        idempotency_key=f"{seed_key}:equity",
                    ),
                ]
                await self._ledger.post_double_entry(postings)

    async def reconcile(
        self,
        client: CoinbaseClient,
        *,
        threshold: Decimal = Decimal("0.0001"),
    ) -> ReconciliationReport:
        """Compare Coinbase balances with ledger state."""

        report_entries: list[DriftRecord] = []
        wallets = await self._fetch_wallets()

        for wallet in wallets:
            account_id = wallet.coinbase_account_id
            if not account_id:
                continue
            balances = await accounts.get_balances(client, account_id)
            ledger_balances = await self._ledger_balances(wallet.wallet_id)
            coinbase_map = {bal.currency: bal.value for bal in balances.balances}
            currencies = set(ledger_balances.keys()) | set(coinbase_map.keys())
            for currency in currencies:
                ledger_total = ledger_balances.get(currency, Decimal("0"))
                coinbase_total = coinbase_map.get(currency, Decimal("0"))
                drift = coinbase_total - ledger_total
                within_threshold = abs(drift) <= threshold
                report_entries.append(
                    DriftRecord(
                        wallet_id=wallet.wallet_id,
                        wallet_name=wallet.name,
                        currency=currency,
                        ledger_balance=ledger_total,
                        coinbase_balance=coinbase_total,
                        drift=drift,
                        within_threshold=within_threshold,
                    )
                )
                if not within_threshold:
                    LOG.warning(
                        "Ledger drift detected",
                        wallet=wallet.name,
                        currency=currency,
                        ledger=str(ledger_total),
                        coinbase=str(coinbase_total),
                        drift=str(drift),
                    )
        return ReconciliationReport(entries=report_entries)

    async def _ensure_wallet(self, session: AsyncSession, account: accounts.Account) -> Wallet:
        wallet = await session.scalar(
            select(Wallet).where(Wallet.coinbase_account_id == account.uuid).with_for_update(of=Wallet)
        )
        if wallet:
            return wallet

        wallet_type = WalletType.COINBASE_TRADING if (account.type or "").upper().endswith("TRADING") else WalletType.COINBASE_SPOT
        wallet = Wallet(
            name=account.name or f"coinbase-{account.uuid[:8]}",
            coinbase_account_id=account.uuid,
            portfolio_id=account.portfolio_uuid,
            type=wallet_type,
            tradeable_fraction=Decimal("0"),
        )
        session.add(wallet)
        await session.flush()
        return wallet

    async def _ensure_equity_wallet(self, session: AsyncSession) -> Wallet:
        wallet = await session.scalar(select(Wallet).where(Wallet.name == "system_equity").with_for_update(of=Wallet))
        if wallet:
            return wallet
        wallet = Wallet(
            name="system_equity",
            coinbase_account_id=None,
            portfolio_id=None,
            type=WalletType.EXTERNAL,
            tradeable_fraction=Decimal("0"),
        )
        session.add(wallet)
        await session.flush()
        return wallet

    async def _fetch_wallets(self) -> list[Wallet]:
        async with self._db.session() as session:
            result = await session.execute(select(Wallet))
            return list(result.scalars())

    async def _ledger_balances(self, wallet_id: int) -> dict[str, Decimal]:
        stmt: Select[tuple[str, Decimal]] = (
            select(LedgerEntry.currency, LedgerEntry.balance_after)
            .where(LedgerEntry.wallet_id == wallet_id)
            .order_by(LedgerEntry.currency, LedgerEntry.created_at.desc())
        )
        async with self._db.session() as session:
            result = await session.execute(stmt)
            latest: dict[str, Decimal] = {}
            for currency, balance in result.all():
                if currency not in latest:
                    latest[currency] = Decimal(balance)
            return latest


__all__ = ["Reconciler", "ReconciliationReport"]
