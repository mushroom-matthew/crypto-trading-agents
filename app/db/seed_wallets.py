"""Utility for seeding demo wallets/balances so the dashboard has data."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Sequence

from sqlalchemy import select

from app.core.logging import get_logger, setup_logging
from app.db.models import Balance, Wallet, WalletType
from app.db.repo import Database


LOG = get_logger(__name__)

# Update these entries to match your Coinbase portfolio/account IDs.
SEED_WALLETS: Sequence[dict[str, object]] = [
    {
        "name": "mock_trading",
        "coinbase_account_id": "acct-demo-trading",
        "portfolio_id": "portfolio-demo-trading",
        "type": WalletType.COINBASE_TRADING,
        "tradeable_fraction": Decimal("0.750"),
        "balances": [
            {"currency": "USD", "available": Decimal("25000"), "hold": Decimal("0")},
        ],
    },
    {
        "name": "system_equity",
        "coinbase_account_id": "acct-demo-equity",
        "portfolio_id": "portfolio-demo-equity",
        "type": WalletType.EXTERNAL,
        "tradeable_fraction": Decimal("1.000"),
        "balances": [
            {"currency": "USD", "available": Decimal("50000"), "hold": Decimal("0")},
            {"currency": "USDC", "available": Decimal("1000"), "hold": Decimal("0")},
        ],
    },
]


async def seed_wallets(entries: Sequence[dict[str, object]] = SEED_WALLETS) -> None:
    setup_logging()
    database = Database()
    created = 0
    updated = 0

    async with database.session() as session:
        for entry in entries:
            name = entry["name"]
            wallet = await session.scalar(select(Wallet).where(Wallet.name == name))

            if wallet is None:
                wallet = Wallet(
                    name=name,
                    coinbase_account_id=entry.get("coinbase_account_id"),
                    portfolio_id=entry.get("portfolio_id"),
                    type=entry.get("type") or WalletType.COINBASE_SPOT,
                    tradeable_fraction=entry.get("tradeable_fraction", Decimal("0")),
                )
                session.add(wallet)
                await session.flush()
                created += 1
                LOG.info("Created wallet", wallet=name, wallet_id=wallet.wallet_id)
            else:
                wallet.coinbase_account_id = entry.get("coinbase_account_id")
                wallet.portfolio_id = entry.get("portfolio_id")
                wallet.type = entry.get("type") or wallet.type
                wallet.tradeable_fraction = entry.get("tradeable_fraction", wallet.tradeable_fraction)
                updated += 1
                LOG.info("Updated wallet", wallet=name, wallet_id=wallet.wallet_id)

            for bal in entry.get("balances", []):
                currency = bal["currency"]
                balance_row = await session.scalar(
                    select(Balance).where(Balance.wallet_id == wallet.wallet_id, Balance.currency == currency)
                )
                if balance_row is None:
                    balance_row = Balance(
                        wallet_id=wallet.wallet_id,
                        currency=currency,
                        available=bal.get("available", Decimal("0")),
                        hold=bal.get("hold", Decimal("0")),
                    )
                    session.add(balance_row)
                else:
                    balance_row.available = bal.get("available", balance_row.available)
                    balance_row.hold = bal.get("hold", balance_row.hold)

    await database.dispose()
    LOG.info("Seed complete", created_wallets=created, updated_wallets=updated)


if __name__ == "__main__":
    asyncio.run(seed_wallets())
