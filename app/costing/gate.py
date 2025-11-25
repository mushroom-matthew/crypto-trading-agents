"""Cost-aware pre-trade gate logic."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from app.coinbase.advanced_trade import OrderBook, get_orderbook
from app.coinbase.client import CoinbaseClient
from app.costing import slippage as slippage_mod
from app.costing.fees import FeeService
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import CostEstimate
from app.db.repo import Database


LOG = get_logger(__name__)


@dataclass(slots=True)
class CostDecision:
    proceed: bool
    total_cost: Decimal
    orderbook: OrderBook
    cost_estimate_id: int


class CostGate:
    """Evaluate total trade cost and persist the audit trail."""

    def __init__(self, db: Database, fee_service: FeeService, client: CoinbaseClient) -> None:
        self._db = db
        self._fee_service = fee_service
        self._client = client
        self._settings = get_settings()

    async def evaluate(
        self,
        *,
        product_id: str,
        side: str,
        qty: Decimal,
        notional_estimate_usd: Decimal,
        is_marketable: bool,
        will_rest: bool,
        expected_edge: Decimal,
        override: bool = False,
        safety_buffer: Optional[Decimal] = None,
        include_network_fees: bool = False,
        order_id: Optional[int] = None,
    ) -> CostDecision:
        """Evaluate whether a trade should proceed."""

        safety_buffer = safety_buffer if safety_buffer is not None else Decimal(str(self._settings.default_safety_buffer))

        fee_rate = await self._fee_service.lookup_current_rate(will_rest=will_rest)
        notional = notional_estimate_usd
        ex_fee = notional * fee_rate

        orderbook = await get_orderbook(self._client, product_id, level=2)
        spread = _calculate_spread(orderbook) if is_marketable else Decimal("0")
        slippage = slippage_mod.simulate(orderbook, qty, side, is_marketable)
        transfer_fee = Decimal("0")
        if include_network_fees:
            transfer_fee = _estimate_network_fee(currency=product_id.split("-")[0], qty=qty)

        total_cost = ex_fee + slippage + spread + transfer_fee
        threshold = total_cost * (Decimal("1") + safety_buffer)
        proceed = override or expected_edge >= threshold

        async with self._db.session() as session:
            record = CostEstimate(
                order_id=order_id,
                ex_fee=ex_fee,
                spread=spread,
                slippage=slippage,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                decision=proceed,
                override_flag=override,
            )
            session.add(record)
            await session.flush()
            cost_estimate_id = record.id

        LOG.info(
            "Cost gate evaluated",
            product_id=product_id,
            side=side,
            qty=str(qty),
            ex_fee=str(ex_fee),
            slippage=str(slippage),
            spread=str(spread),
            transfer_fee=str(transfer_fee),
            total_cost=str(total_cost),
            proceed=proceed,
            override=override,
        )

        return CostDecision(
            proceed=proceed,
            total_cost=total_cost,
            orderbook=orderbook,
            cost_estimate_id=cost_estimate_id,
        )


def _calculate_spread(orderbook: OrderBook) -> Decimal:
    if not orderbook.asks or not orderbook.bids:
        return Decimal("0")
    return orderbook.asks[0].price - orderbook.bids[0].price


def _estimate_network_fee(*, currency: str, qty: Decimal) -> Decimal:
    # Placeholder implementation; extend using actual network fee estimators.
    return Decimal("0")


__all__ = ["CostGate", "CostDecision"]
