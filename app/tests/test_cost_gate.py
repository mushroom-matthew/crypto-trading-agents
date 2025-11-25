from decimal import Decimal

import pytest
import respx
from httpx import Response

from app.coinbase.client import CoinbaseClient
from app.costing.fees import FeeService
from app.costing.gate import CostGate
from app.db.models import CostEstimate
from sqlalchemy import select
from app.db.repo import Database
from app.core.config import Settings


@pytest.mark.asyncio
@respx.mock
async def test_cost_gate_declines_when_costs_exceed_edge(db: Database, settings: Settings):
    respx.get("https://example.com/api/v3/brokerage/transaction_summary").mock(
        return_value=Response(
            200,
            json={
                "fee_tier": {
                    "maker_fee_rate": "0.01",
                    "taker_fee_rate": "0.02",
                    "pricing_tier": "test",
                }
            },
        )
    )
    respx.get("https://example.com/api/v3/brokerage/product/BTC-USD/book", params={"level": 2}).mock(
        return_value=Response(
            200,
            json={
                "bids": [["100", "1"]],
                "asks": [["101", "1"]],
            },
        )
    )

    async with CoinbaseClient(settings) as client:
        fee_service = FeeService(db, client, ttl=0)
        gate = CostGate(db, fee_service, client)
        decision = await gate.evaluate(
            product_id="BTC-USD",
            side="buy",
            qty=Decimal("1"),
            notional_estimate_usd=Decimal("100"),
            is_marketable=True,
            will_rest=False,
            expected_edge=Decimal("1"),
            override=False,
        )

    assert decision.proceed is False

    async with db.session() as session:
        result = await session.execute(select(CostEstimate))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].decision is False
