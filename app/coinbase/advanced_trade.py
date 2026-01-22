"""Coinbase Advanced Trade order helpers."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.coinbase.client import CoinbaseClient
from app.core.errors import CoinbaseAPIError

logger = logging.getLogger(__name__)


class OrderConfig(BaseModel):
    """Normalized order configuration payload."""

    product_id: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT"]
    base_size: Decimal
    limit_price: Optional[Decimal] = None
    time_in_force: Literal["GOOD_TIL_CANCELLED", "GOOD_TIL_TIME", "FILL_OR_KILL", "IMMEDIATE_OR_CANCEL"] = "GOOD_TIL_CANCELLED"
    post_only: bool = False

    def to_payload(self, portfolio_id: Optional[str]) -> dict[str, Any]:
        """Convert to Coinbase request payload."""

        configuration: dict[str, Any] = {
            "side": self.side,
        }
        if self.order_type == "MARKET":
            configuration["order_configuration"] = {"market_market_ioc": {"base_size": str(self.base_size)}}
        else:
            configuration["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(self.base_size),
                    "limit_price": str(self.limit_price),
                    "post_only": self.post_only,
                }
            }

        payload = {
            "client_order_id": None,
            "product_id": self.product_id,
            **configuration,
        }
        if portfolio_id:
            payload["portfolio_id"] = portfolio_id
        return payload


class OrderResponse(BaseModel):
    """Canonical representation of an order after submission."""

    model_config = ConfigDict(extra="ignore")

    order_id: str = Field(..., alias="id")
    product_id: str
    side: Literal["BUY", "SELL"]
    status: str
    filled_size: Decimal = Field(default=Decimal("0"), alias="filled_size")
    average_filled_price: Optional[Decimal] = Field(default=None, alias="average_filled_price")


class OrderBookLevel(BaseModel):
    """Single order book level entry."""

    price: Decimal
    size: Decimal


class OrderBook(BaseModel):
    """Simplified order book snapshot."""

    model_config = ConfigDict(extra="ignore")

    product_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]


async def place_order(
    client: CoinbaseClient,
    *,
    portfolio_id: Optional[str],
    product_id: str,
    side: Literal["buy", "sell"],
    order_type: Literal["limit", "market"],
    qty: Decimal,
    price: Optional[Decimal] = None,
    idempotency_key: Optional[str] = None,
) -> OrderResponse:
    """Submit an order via Coinbase Advanced Trade.

    SAFETY: This function places REAL orders with REAL money on Coinbase.
    It will be blocked unless LIVE_TRADING_ACK=true is set in the environment.
    """
    # SAFETY CHECK: Verify runtime mode before submitting to Coinbase
    from agents.runtime_mode import get_runtime_mode

    runtime = get_runtime_mode()

    # This function ALWAYS places real orders, so we check runtime mode
    if runtime.is_live:
        logger.critical(
            "COINBASE LIVE ORDER: %s %s %.8f %s at %s, ack=%s",
            side.upper(), product_id, qty, order_type.upper(),
            price or "MARKET", runtime.live_trading_ack
        )
        if not runtime.live_trading_ack:
            error_msg = (
                "COINBASE ORDER BLOCKED: Cannot place real Coinbase order without explicit "
                "LIVE_TRADING_ACK=true environment variable. This would execute a real trade "
                "with real money. Set LIVE_TRADING_ACK=true to acknowledge and proceed."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        # Log acknowledgment for audit trail
        logger.critical(
            "COINBASE ORDER ACKNOWLEDGED: Proceeding with REAL order (LIVE_TRADING_ACK=true)"
        )
    else:
        # Even in paper mode, warn about Coinbase API calls
        logger.warning(
            "COINBASE API CALL IN PAPER MODE: %s %s %.8f %s - "
            "This should not happen in paper trading!",
            side.upper(), product_id, qty, order_type.upper()
        )

    config = OrderConfig(
        product_id=product_id,
        side=side.upper(),  # type: ignore[arg-type]
        order_type=order_type.upper(),  # type: ignore[arg-type]
        base_size=qty,
        limit_price=price,
    )
    payload = config.to_payload(portfolio_id)
    raw = await client.post("/api/v3/brokerage/orders", json_payload=payload, idempotency_key=idempotency_key)
    order = raw.get("order")
    if not order:
        raise CoinbaseAPIError("Unexpected order response payload", 500, payload=raw)
    return OrderResponse.model_validate(order)


async def get_order(client: CoinbaseClient, order_id: str) -> OrderResponse:
    """Fetch order status by Coinbase order id."""

    raw = await client.get(f"/api/v3/brokerage/orders/{order_id}")
    order = raw.get("order") or raw
    return OrderResponse.model_validate(order)


async def get_orderbook(client: CoinbaseClient, product_id: str, level: Literal[1, 2, 3] = 2) -> OrderBook:
    """Fetch order book snapshot at requested depth."""

    raw = await client.get(f"/api/v3/brokerage/product/{product_id}/book", params={"level": level})
    bids = _parse_levels(raw.get("bids", []))
    asks = _parse_levels(raw.get("asks", []))
    return OrderBook(product_id=product_id, bids=bids, asks=asks)


def _parse_levels(levels: Iterable[list[str]]) -> list[OrderBookLevel]:
    normalized: list[OrderBookLevel] = []
    for level in levels:
        if len(level) < 2:
            continue
        price, size = level[0], level[1]
        normalized.append(OrderBookLevel(price=Decimal(price), size=Decimal(size)))
    return normalized


__all__ = ["OrderResponse", "OrderBook", "place_order", "get_order", "get_orderbook"]
