from decimal import Decimal

import pytest
import respx
from httpx import Response
from sqlalchemy import select

from app.coinbase.client import CoinbaseClient
from app.costing.fees import FeeService
from app.costing.gate import CostGate
from app.db.models import Balance, LedgerEntry, Reservation, ReservationState, Wallet, WalletType
from app.db.repo import Database
from app.ledger.engine import LedgerEngine
from app.strategy.trade_executor import TradeExecutor, TradeParams
from app.core.config import Settings


@pytest.mark.asyncio
@respx.mock
async def test_trade_executor_places_order_and_updates_ledger(db: Database, settings: Settings):
    ledger = LedgerEngine(db)

    async with db.session() as session:
        trading_wallet = Wallet(
            name="trading",
            coinbase_account_id="acct-trade",
            portfolio_id="portfolio-1",
            type=WalletType.COINBASE_TRADING,
            tradeable_fraction=Decimal("1"),
        )
        equity_wallet = Wallet(
            name="system_equity",
            coinbase_account_id=None,
            portfolio_id=None,
            type=WalletType.EXTERNAL,
            tradeable_fraction=Decimal("0"),
        )
        session.add_all([trading_wallet, equity_wallet])
        await session.flush()
        session.add(
            Balance(
                wallet_id=trading_wallet.wallet_id,
                currency="USD",
                available=Decimal("1000"),
                hold=Decimal("0"),
            )
        )
        await session.flush()
        wallet_id = trading_wallet.wallet_id

    respx.get("https://example.com/api/v3/brokerage/transaction_summary").mock(
        return_value=Response(
            200,
            json={"fee_tier": {"maker_fee_rate": "0.0005", "taker_fee_rate": "0.0005", "pricing_tier": "test"}},
        )
    )
    respx.get("https://example.com/api/v3/brokerage/product/BTC-USD/book", params={"level": 2}).mock(
        return_value=Response(
            200,
            json={
                "bids": [["10000", "1"]],
                "asks": [["10001", "1"]],
            },
        )
    )
    respx.post("https://example.com/api/v3/brokerage/orders").mock(
        return_value=Response(
            200,
            json={
                "order": {
                    "id": "order-1",
                    "product_id": "BTC-USD",
                    "side": "BUY",
                    "status": "FILLED",
                    "filled_size": "0.01",
                    "average_filled_price": "10000",
                }
            },
        )
    )

    async with CoinbaseClient(settings) as client:
        fee_service = FeeService(db, client, ttl=0)
        cost_gate = CostGate(db, fee_service, client)
        executor = TradeExecutor(db, ledger, cost_gate, client)
        params = TradeParams(
            wallet_id=wallet_id,
            product_id="BTC-USD",
            side="buy",
            qty=Decimal("0.01"),
            order_type="market",
            notional_estimate_usd=Decimal("100"),
            expected_edge=Decimal("10"),
            is_marketable=True,
            will_rest=False,
            idempotency_key="trade-1",
        )
        result = await executor.execute_trade(params)

    assert result.status == "placed"

    async with db.session() as session:
        reservations = await session.execute(select(Reservation))
        reservation_rows = reservations.scalars().all()
        assert reservation_rows
        assert reservation_rows[0].state == ReservationState.consumed

        ledger_entries = await session.execute(select(LedgerEntry))
        rows = ledger_entries.scalars().all()
        assert len(rows) == 4
        currencies = {entry.currency for entry in rows}
        assert currencies == {"BTC", "USD"}
