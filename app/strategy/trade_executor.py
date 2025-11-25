"""Trade execution orchestration tying together reservation, cost gate, and Coinbase APIs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from sqlalchemy import select

from app.coinbase import advanced_trade, transfers
from app.coinbase.advanced_trade import OrderResponse
from app.coinbase.client import CoinbaseClient
from app.costing.gate import CostGate
from app.core.errors import LedgerError
from app.core.idempotency import from_value
from app.core.logging import get_logger
from app.db.models import CostEstimate, LedgerSide, Order, OrderSide, OrderType, ReservationState, Wallet, WalletType
from app.db.repo import Database
from app.ledger.engine import LedgerEngine, Posting


LOG = get_logger(__name__)


@dataclass(slots=True)
class TradeParams:
    wallet_id: int
    product_id: str
    side: str
    qty: Decimal
    order_type: str
    notional_estimate_usd: Decimal
    expected_edge: Decimal
    is_marketable: bool
    will_rest: bool
    idempotency_key: str
    price: Optional[Decimal] = None
    override: bool = False


@dataclass(slots=True)
class TradeResult:
    status: str
    reservation_id: Optional[int]
    coinbase_order_id: Optional[str]
    details: str


class TradeExecutor:
    """Top-level trade execution pipeline."""

    def __init__(self, db: Database, ledger: LedgerEngine, gate: CostGate, client: CoinbaseClient) -> None:
        self._db = db
        self._ledger = ledger
        self._gate = gate
        self._client = client

    async def execute_trade(self, params: TradeParams) -> TradeResult:
        LOG.info("Executing trade", product=params.product_id, side=params.side, qty=str(params.qty))

        wallet = await self._load_wallet(params.wallet_id)
        reserve_currency = _reserve_currency(params.product_id, params.side)
        reserve_amount = params.notional_estimate_usd if params.side.lower() == "buy" else params.qty

        reservation = await self._ledger.acquire_tradable_lock(
            wallet_id=params.wallet_id,
            currency=reserve_currency,
            amount=reserve_amount,
            idempotency_key=f"{params.idempotency_key}:reserve",
        )

        decision = await self._gate.evaluate(
            product_id=params.product_id,
            side=params.side,
            qty=params.qty,
            notional_estimate_usd=params.notional_estimate_usd,
            is_marketable=params.is_marketable,
            will_rest=params.will_rest,
            expected_edge=params.expected_edge,
            override=params.override,
        )

        if not decision.proceed:
            await self._ledger.release_reservation(reservation_id=reservation.res_id, state=ReservationState.canceled)
            LOG.info("Trade skipped by cost gate", product_id=params.product_id)
            return TradeResult(
                status="skipped_cost_gate",
                reservation_id=reservation.res_id,
                coinbase_order_id=None,
                details="Cost gate decline",
            )

        trading_wallet = wallet
        if wallet.type != WalletType.COINBASE_TRADING:
            trading_wallet = await self._ensure_trading_funds(
                wallet=wallet,
                currency=reserve_currency,
                amount=reserve_amount,
                idempotency_key=params.idempotency_key,
            )

        order_response = await advanced_trade.place_order(
            self._client,
            portfolio_id=trading_wallet.portfolio_id,
            product_id=params.product_id,
            side=params.side,
            order_type=params.order_type,
            qty=params.qty,
            price=params.price,
            idempotency_key=params.idempotency_key,
        )

        order_row = await self._persist_order(trading_wallet.wallet_id, order_response, params)

        await self._link_cost_estimate(decision.cost_estimate_id, order_row.order_id)

        await self._handle_fills(trading_wallet, order_response, params.idempotency_key)

        await self._ledger.release_reservation(reservation_id=reservation.res_id, state=ReservationState.consumed)

        return TradeResult(
            status="placed",
            reservation_id=reservation.res_id,
            coinbase_order_id=order_response.order_id,
            details=order_response.status,
        )

    async def _load_wallet(self, wallet_id: int) -> Wallet:
        async with self._db.session() as session:
            wallet = await session.scalar(select(Wallet).where(Wallet.wallet_id == wallet_id))
            if wallet is None:
                raise LedgerError(f"Wallet {wallet_id} not found")
            return wallet

    async def _ensure_trading_funds(
        self, *, wallet: Wallet, currency: str, amount: Decimal, idempotency_key: str
    ) -> Wallet:
        async with self._db.session() as session:
            trading_wallet = await session.scalar(
                select(Wallet).where(
                    Wallet.portfolio_id == wallet.portfolio_id,
                    Wallet.type == WalletType.COINBASE_TRADING,
                )
            )
        if trading_wallet is None:
            raise LedgerError("No trading wallet linked to funding wallet")
        if not wallet.coinbase_account_id or not trading_wallet.coinbase_account_id:
            raise LedgerError("Wallets lack Coinbase account identifiers")

        idem = from_value(f"{idempotency_key}:transfer")
        await transfers.create_internal(
            self._client,
            from_account=wallet.coinbase_account_id,
            to_account=trading_wallet.coinbase_account_id,
            currency=currency,
            amount=amount,
            idempotency_key=idem,
        )
        return trading_wallet

    async def _persist_order(self, wallet_id: int, response: OrderResponse, params: TradeParams) -> Order:
        async with self._db.session() as session:
            order = Order(
                wallet_id=wallet_id,
                coinbase_order_id=response.order_id,
                product_id=response.product_id,
                side=OrderSide[params.side.lower()],
                order_type=OrderType[params.order_type.lower()],
                price=params.price,
                qty=params.qty,
                status=response.status,
                filled_qty=response.filled_size,
            )
            session.add(order)
            await session.flush()
            return order

    async def _handle_fills(self, wallet: Wallet, response: OrderResponse, base_idempotency: str) -> None:
        if response.filled_size <= 0 or response.average_filled_price is None:
            return
        equity_wallet = await self._load_equity_wallet()
        base_currency, quote_currency = response.product_id.split("-")
        base_amount = response.filled_size
        quote_amount = response.filled_size * response.average_filled_price
        fill_key = f"{base_idempotency}:fill:{response.order_id}"
        postings = [
            Posting(
                wallet_id=wallet.wallet_id,
                currency=base_currency,
                amount=base_amount,
                side=LedgerSide.debit,
                source="trade_fill",
                idempotency_key=f"{fill_key}:base:asset",
            ),
            Posting(
                wallet_id=equity_wallet.wallet_id,
                currency=base_currency,
                amount=base_amount,
                side=LedgerSide.credit,
                source="trade_fill",
                idempotency_key=f"{fill_key}:base:equity",
            ),
            Posting(
                wallet_id=equity_wallet.wallet_id,
                currency=quote_currency,
                amount=quote_amount,
                side=LedgerSide.debit,
                source="trade_fill",
                idempotency_key=f"{fill_key}:quote:equity",
            ),
            Posting(
                wallet_id=wallet.wallet_id,
                currency=quote_currency,
                amount=quote_amount,
                side=LedgerSide.credit,
                source="trade_fill",
                idempotency_key=f"{fill_key}:quote:asset",
            ),
        ]
        await self._ledger.post_double_entry(postings)

    async def _load_equity_wallet(self) -> Wallet:
        async with self._db.session() as session:
            wallet = await session.scalar(select(Wallet).where(Wallet.name == "system_equity"))
            if wallet is None:
                raise LedgerError("System equity wallet missing; run seed process first")
            return wallet

    async def _link_cost_estimate(self, estimate_id: int, order_id: int) -> None:
        async with self._db.session() as session:
            estimate = await session.get(CostEstimate, estimate_id)
            if estimate:
                estimate.order_id = order_id
                await session.flush()


def _reserve_currency(product_id: str, side: str) -> str:
    base, quote = product_id.split("-")
    return quote if side.lower() == "buy" else base


__all__ = ["TradeExecutor", "TradeParams", "TradeResult"]
