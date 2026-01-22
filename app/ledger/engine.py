"""Double-entry ledger engine and reservation management."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import IdempotencyError, LedgerError
from app.core.logging import get_logger
from app.db.models import Balance, LedgerEntry, LedgerSide, Reservation, ReservationState, Wallet
from app.db.repo import Database


LOG = get_logger(__name__)


@dataclass(slots=True)
class Posting:
    """Representation of a single ledger posting."""

    wallet_id: int
    currency: str
    amount: Decimal
    side: LedgerSide
    source: str
    idempotency_key: str
    external_tx_id: Optional[str] = None


class LedgerEngine:
    """High-level ledger APIs for posting and reservations."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def post_double_entry(self, postings: Sequence[Posting]) -> list[LedgerEntry]:
        """Persist a double-entry transaction.

        Raises:
            LedgerError: if the postings do not balance per currency.
            IdempotencyError: when attempting to reuse an existing key with mismatched payload.
        """

        if not postings:
            raise LedgerError("Cannot post empty transaction")

        self._validate_balance(postings)

        async with self._db.session() as session:
            materialized: list[LedgerEntry] = []
            for posting in postings:
                persisted = await self._get_existing_entry(session, posting.idempotency_key)
                if persisted:
                    LOG.info("Ledger posting already exists", entry_id=persisted.entry_id, key=posting.idempotency_key)
                    materialized.append(persisted)
                    continue
                entry = await self._create_entry(session, posting)
                materialized.append(entry)
        return materialized

    async def _create_entry(self, session: AsyncSession, posting: Posting) -> LedgerEntry:
        """Insert a new ledger entry and compute balance_after."""

        current = await self._current_balance(session, posting.wallet_id, posting.currency)
        if posting.side == LedgerSide.debit:
            new_balance = current + posting.amount
        else:
            new_balance = current - posting.amount

        entry = LedgerEntry(
            wallet_id=posting.wallet_id,
            currency=posting.currency,
            amount=posting.amount,
            side=posting.side,
            balance_after=new_balance,
            source=posting.source,
            external_tx_id=posting.external_tx_id,
            idempotency_key=posting.idempotency_key,
        )
        session.add(entry)
        try:
            await session.flush()
        except IntegrityError as exc:  # pragma: no cover - triggered on race; covered via unique constraint
            raise IdempotencyError(f"Ledger entry idempotency conflict for {posting.idempotency_key}") from exc
        return entry

    async def _current_balance(self, session: AsyncSession, wallet_id: int, currency: str) -> Decimal:
        stmt: Select[tuple[Decimal]] = (
            select(LedgerEntry.balance_after)
            .where(LedgerEntry.wallet_id == wallet_id, LedgerEntry.currency == currency)
            .order_by(LedgerEntry.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return Decimal("0")
        return Decimal(row)

    async def _get_existing_entry(self, session: AsyncSession, key: str) -> Optional[LedgerEntry]:
        stmt = select(LedgerEntry).where(LedgerEntry.idempotency_key == key)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    def _validate_balance(self, postings: Sequence[Posting]) -> None:
        per_currency: dict[str, Decimal] = {}
        for posting in postings:
            currency_key = posting.currency
            if posting.amount <= 0:
                raise LedgerError("Ledger postings must use positive amounts")

            sign = Decimal("1") if posting.side == LedgerSide.debit else Decimal("-1")
            per_currency[currency_key] = per_currency.get(currency_key, Decimal("0")) + sign * posting.amount

        for key, net in per_currency.items():
            if net != 0:
                LOG.error("Unbalanced ledger transaction", currency=key, net=str(net))
                raise LedgerError(f"Ledger postings are not balanced for currency {key}")

    async def acquire_tradable_lock(
        self,
        *,
        wallet_id: int,
        currency: str,
        amount: Decimal,
        idempotency_key: str,
    ) -> Reservation:
        """Reserve tradable funds enforcing fraction and active reservations."""

        async with self._db.session() as session:
            existing = await self._get_reservation_by_key(session, idempotency_key)
            if existing:
                if existing.state != ReservationState.active:
                    raise LedgerError("Reservation already finalized")
                return existing

            wallet = await session.scalar(select(Wallet).where(Wallet.wallet_id == wallet_id).with_for_update())
            if wallet is None:
                raise LedgerError(f"Wallet {wallet_id} not found")

            available = await self._latest_balance_snapshot(session, wallet_id, currency)
            fraction = wallet.tradeable_fraction or Decimal("0")
            active_reserved = await self._active_reservations_sum(session, wallet_id, currency)
            allowed = (available * fraction) - active_reserved

            if amount > allowed:
                raise LedgerError(
                    f"Insufficient tradable balance. amount={amount} allowed={allowed} "
                    f"fraction={fraction}"
                )

            reservation = Reservation(
                wallet_id=wallet_id,
                currency=currency,
                amount=amount,
                state=ReservationState.active,
                idempotency_key=idempotency_key,
            )
            session.add(reservation)
            try:
                await session.flush()
            except IntegrityError as exc:  # pragma: no cover
                raise IdempotencyError(f"Reservation idempotency conflict {idempotency_key}") from exc
            return reservation

    async def release_reservation(
        self,
        *,
        reservation_id: int,
        state: ReservationState,
    ) -> Reservation:
        """Transition reservation state to consumed or canceled."""

        if state not in {ReservationState.consumed, ReservationState.canceled}:
            raise LedgerError("Reservation can only transition to consumed or canceled")

        async with self._db.session() as session:
            reservation = await session.scalar(
                select(Reservation).where(Reservation.res_id == reservation_id).with_for_update()
            )
            if reservation is None:
                raise LedgerError(f"Reservation {reservation_id} not found")
            reservation.state = state
            await session.flush()
            return reservation

    async def _latest_balance_snapshot(self, session: AsyncSession, wallet_id: int, currency: str) -> Decimal:
        stmt = (
            select(Balance.available)
            .where(Balance.wallet_id == wallet_id, Balance.currency == currency)
            .order_by(Balance.fetched_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        value = result.scalar_one_or_none()
        return Decimal(value or 0)

    async def _active_reservations_sum(self, session: AsyncSession, wallet_id: int, currency: str) -> Decimal:
        stmt = (
            select(func.coalesce(func.sum(Reservation.amount), 0))
            .where(
                Reservation.wallet_id == wallet_id,
                Reservation.currency == currency,
                Reservation.state == ReservationState.active,
            )
        )
        result = await session.execute(stmt)
        total = result.scalar_one()
        return Decimal(total or 0)

    async def _get_reservation_by_key(self, session: AsyncSession, key: str) -> Optional[Reservation]:
        stmt = select(Reservation).where(Reservation.idempotency_key == key)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


__all__ = ["LedgerEngine", "Posting"]
