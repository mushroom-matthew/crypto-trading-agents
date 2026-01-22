from decimal import Decimal

import pytest

from app.db.models import Balance, ReservationState, Wallet, WalletType
from app.ledger.engine import LedgerEngine
from app.db.repo import Database
from app.core.errors import LedgerError


@pytest.mark.asyncio
async def test_acquire_tradable_lock_prevents_double_spend(db: Database):
    ledger = LedgerEngine(db)

    async with db.session() as session:
        wallet = Wallet(
            name="spot",
            coinbase_account_id="acc-1",
            portfolio_id="port-1",
            type=WalletType.COINBASE_SPOT,
            tradeable_fraction=Decimal("0.5"),
        )
        session.add(wallet)
        await session.flush()
        session.add(
            Balance(
                wallet_id=wallet.wallet_id,
                currency="USD",
                available=Decimal("100"),
                hold=Decimal("0"),
            )
        )
        await session.flush()
        wallet_id = wallet.wallet_id

    reservation = await ledger.acquire_tradable_lock(
        wallet_id=wallet_id,
        currency="USD",
        amount=Decimal("40"),
        idempotency_key="reserve-1",
    )
    assert reservation.state == ReservationState.active

    reservation_second = await ledger.acquire_tradable_lock(
        wallet_id=wallet_id,
        currency="USD",
        amount=Decimal("10"),
        idempotency_key="reserve-2",
    )
    assert reservation_second.amount == Decimal("10")

    with pytest.raises(LedgerError):
        await ledger.acquire_tradable_lock(
            wallet_id=wallet_id,
            currency="USD",
            amount=Decimal("10"),
            idempotency_key="reserve-3",
        )
