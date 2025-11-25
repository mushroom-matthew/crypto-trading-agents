"""Orderbook-based slippage estimation."""

from __future__ import annotations

from decimal import Decimal

from app.coinbase.advanced_trade import OrderBook


def simulate(orderbook: OrderBook, qty: Decimal, side: str, is_marketable: bool) -> Decimal:
    """Estimate slippage given an orderbook snapshot.

    Returns:
        Decimal: positive cost representing expected slippage in quote currency.
    """

    if qty <= 0 or not is_marketable:
        return Decimal("0")

    levels = orderbook.asks if side.lower() == "buy" else orderbook.bids
    if not levels:
        return Decimal("0")

    best_price = levels[0].price
    remaining = qty
    notional = Decimal("0")

    for level in levels:
        if remaining <= 0:
            break
        trade_size = min(level.size, remaining)
        notional += trade_size * level.price
        remaining -= trade_size

    if remaining > 0:
        # Not enough depth; assume final chunk fills at worst available price observed.
        notional += remaining * levels[-1].price

    avg_fill_price = notional / qty
    slippage = (avg_fill_price - best_price) * qty if side.lower() == "buy" else (best_price - avg_fill_price) * qty
    return abs(slippage)


__all__ = ["simulate"]
