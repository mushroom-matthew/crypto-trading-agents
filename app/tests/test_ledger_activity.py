import pytest
from sqlalchemy import select

from agents.activities.ledger import persist_fill_activity
from app.db.models import LedgerEntry, Wallet


@pytest.mark.asyncio
async def test_persist_fill_activity_creates_ledger_entries(db, settings):
    payload = {
        "fill": {
            "symbol": "BTC/USD",
            "side": "BUY",
            "qty": "0.1",
            "fill_price": "30000",
            "cost": "3000",
        },
        "workflow_id": "wf-test",
        "sequence": 1,
        "recorded_at": 0.0,
        "trading_wallet_name": "mock_trading",
        "equity_wallet_name": "system_equity",
    }

    await persist_fill_activity(payload)

    async with db.session() as session:
        entries = (await session.execute(select(LedgerEntry))).scalars().all()
        assert len(entries) == 4
        currencies = {entry.currency for entry in entries}
        assert currencies == {"BTC", "USD"}

        wallets = (await session.execute(select(Wallet))).scalars().all()
        wallet_names = {wallet.name for wallet in wallets}
        assert {"mock_trading", "system_equity"}.issubset(wallet_names)
