"""Typer CLI entrypoint for ledger and trading operations."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Optional

import typer
from sqlalchemy import select

from app.coinbase.client import CoinbaseClient
from app.costing.fees import FeeService
from app.costing.gate import CostGate
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.models import Balance, Wallet
from app.db.repo import Database
from app.ledger.engine import LedgerEngine
from app.ledger.reconciliation import Reconciler
from app.strategy.trade_executor import TradeExecutor, TradeParams


app = typer.Typer(help="Coinbase-enabled trading toolkit.")
ledger_cli = typer.Typer(help="Ledger lifecycle commands.")
wallet_cli = typer.Typer(help="Wallet configuration commands.")
trade_cli = typer.Typer(help="Trading commands.")
reconcile_cli = typer.Typer(help="Reconciliation tools.")

app.add_typer(ledger_cli, name="ledger")
app.add_typer(wallet_cli, name="wallet")
app.add_typer(trade_cli, name="trade")
app.add_typer(reconcile_cli, name="reconcile")


@ledger_cli.command("seed-from-coinbase")
def seed_from_coinbase() -> None:
    """Seed wallets and balances from Coinbase accounts."""

    asyncio.run(_seed_from_coinbase())


async def _seed_from_coinbase() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    db = Database(settings)
    ledger = LedgerEngine(db)
    reconciler = Reconciler(db, ledger)
    async with CoinbaseClient(settings) as client:
        await reconciler.seed_from_coinbase(client)
    typer.echo("Seeded wallets from Coinbase successfully.")


@wallet_cli.command("set-tradeable-fraction")
def set_tradeable_fraction(wallet: int, frac: float) -> None:
    """Update tradeable fraction for a wallet."""

    asyncio.run(_set_tradeable_fraction(wallet, Decimal(str(frac))))


async def _set_tradeable_fraction(wallet_id: int, fraction: Decimal) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    if not Decimal("0") <= fraction <= Decimal("1"):
        raise typer.BadParameter("Fraction must be between 0 and 1")
    db = Database(settings)
    async with db.session() as session:
        wallet = await session.get(Wallet, wallet_id)
        if wallet is None:
            raise typer.BadParameter(f"Wallet {wallet_id} not found")
        wallet.tradeable_fraction = fraction
        await session.flush()
    typer.echo(f"Updated wallet {wallet_id} tradeable fraction to {fraction}")


@wallet_cli.command("list")
def list_wallets(currency: str = typer.Option(None, "--currency", help="Filter balances by currency")) -> None:
    """List wallets, Coinbase account ids, and cached balances."""

    asyncio.run(_list_wallets(currency=currency))


async def _list_wallets(currency: Optional[str]) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    db = Database(settings)
    async with db.session() as session:
        stmt = (
            select(
                Wallet.wallet_id,
                Wallet.name,
                Wallet.type,
                Wallet.coinbase_account_id,
                Wallet.tradeable_fraction,
                Balance.currency,
                Balance.available,
                Balance.hold,
            )
            .outerjoin(Balance, Balance.wallet_id == Wallet.wallet_id)
            .order_by(Wallet.wallet_id, Balance.currency)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()

    summaries: dict[int, dict[str, Any]] = {}
    for row in rows:
        existing = summaries.setdefault(
            row.wallet_id,
            {
                "wallet_id": row.wallet_id,
                "name": row.name,
                "type": row.type.value if row.type else None,
                "coinbase_account": row.coinbase_account_id,
                "tradeable_fraction": str(row.tradeable_fraction),
                "balances": [],
            },
        )
        if row.currency:
            if currency and row.currency.lower() != currency.lower():
                continue
            existing["balances"].append(
                {
                    "currency": row.currency,
                    "available": str(row.available),
                    "hold": str(row.hold),
                }
            )

    for summary in summaries.values():
        balances = summary["balances"]
        balance_str = ", ".join(
            f"{bal['currency']}:avail={bal['available']},hold={bal['hold']}" for bal in balances
        ) or "no balances"
        typer.echo(
            f"wallet_id={summary['wallet_id']} name={summary['name']} type={summary['type']} "
            f"tradeable_fraction={summary['tradeable_fraction']} coinbase_account={summary['coinbase_account']} "
            f"balances=[{balance_str}]"
        )


@trade_cli.command("place")
def place_trade(
    wallet: int = typer.Argument(..., help="Source wallet id"),
    product: str = typer.Option(..., "--product", help="Product pair e.g. BTC-USD"),
    side: str = typer.Option(..., "--side", help="buy or sell"),
    qty: float = typer.Option(..., "--qty", help="Quantity to trade"),
    order_type: str = typer.Option("limit", "--order-type", help="limit or market"),
    price: Optional[float] = typer.Option(None, "--price", help="Limit price if applicable"),
    override: bool = typer.Option(False, "--override", help="Override cost gate"),
    expected_edge: float = typer.Option(0.0, "--expected-edge", help="Expected edge in USD"),
    notional: float = typer.Option(..., "--notional", help="Notional estimate in USD"),
    is_marketable: bool = typer.Option(False, "--marketable", help="Whether order is marketable"),
    will_rest: bool = typer.Option(True, "--will-rest", help="Whether order will rest on book"),
    idem: str = typer.Option(..., "--idempotency-key", help="Unique id for the trade"),
) -> None:
    """Place a trade via Coinbase Advanced Trade APIs."""

    asyncio.run(
        _place_trade(
            wallet_id=wallet,
            product_id=product,
            side=side,
            qty=Decimal(str(qty)),
            order_type=order_type,
            price=Decimal(str(price)) if price is not None else None,
            override=override,
            expected_edge=Decimal(str(expected_edge)),
            notional=Decimal(str(notional)),
            is_marketable=is_marketable,
            will_rest=will_rest,
            idempotency_key=idem,
        )
    )


async def _place_trade(
    *,
    wallet_id: int,
    product_id: str,
    side: str,
    qty: Decimal,
    order_type: str,
    price: Optional[Decimal],
    override: bool,
    expected_edge: Decimal,
    notional: Decimal,
    is_marketable: bool,
    will_rest: bool,
    idempotency_key: str,
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    db = Database(settings)
    ledger = LedgerEngine(db)
    async with CoinbaseClient(settings) as client:
        fee_service = FeeService(db, client)
        cost_gate = CostGate(db, fee_service, client)
        executor = TradeExecutor(db, ledger, cost_gate, client)
        params = TradeParams(
            wallet_id=wallet_id,
            product_id=product_id,
            side=side,
            qty=qty,
            order_type=order_type,
            notional_estimate_usd=notional,
            expected_edge=expected_edge,
            is_marketable=is_marketable,
            will_rest=will_rest,
            idempotency_key=idempotency_key,
            price=price,
            override=override,
        )
        result = await executor.execute_trade(params)
    typer.echo(f"Trade result: {result.status} - {result.details}")


@reconcile_cli.command("run")
def run_reconciliation(threshold: float = typer.Option(0.0001, "--threshold", help="Drift tolerance")) -> None:
    """Run reconciliation against Coinbase balances."""

    asyncio.run(_run_reconciliation(Decimal(str(threshold))))


async def _run_reconciliation(threshold: Decimal) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    db = Database(settings)
    ledger = LedgerEngine(db)
    reconciler = Reconciler(db, ledger)
    async with CoinbaseClient(settings) as client:
        report = await reconciler.reconcile(client, threshold=threshold)
    for entry in report.entries:
        status = "OK" if entry.within_threshold else "DRIFT"
        typer.echo(
            f"[{status}] wallet={entry.wallet_name} currency={entry.currency} "
            f"ledger={entry.ledger_balance} coinbase={entry.coinbase_balance} drift={entry.drift}"
        )


if __name__ == "__main__":
    app()
