"""Initial ledger schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    wallet_type = sa.Enum("COINBASE_SPOT", "COINBASE_TRADING", "EXTERNAL", name="wallet_type")
    ledger_side = sa.Enum("debit", "credit", name="ledger_side")
    order_side = sa.Enum("buy", "sell", name="order_side")
    order_type = sa.Enum("limit", "market", name="order_type")
    reservation_state = sa.Enum("active", "consumed", "canceled", name="reservation_state")

    op.create_table(
        "wallets",
        sa.Column("wallet_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("coinbase_account_id", sa.String(length=255)),
        sa.Column("portfolio_id", sa.String(length=255)),
        sa.Column("type", wallet_type, nullable=False),
        sa.Column("tradeable_fraction", sa.Numeric(4, 3), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "fees_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("portfolio_id", sa.String(length=255), nullable=False),
        sa.Column("maker_rate", sa.Numeric(12, 8), nullable=False),
        sa.Column("taker_rate", sa.Numeric(12, 8), nullable=False),
        sa.Column("tier_name", sa.String(length=128)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_fees_snapshots_portfolio", "fees_snapshots", ["portfolio_id"])

    op.create_table(
        "balances",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("available", sa.Numeric(24, 12), nullable=False),
        sa.Column("hold", sa.Numeric(24, 12), nullable=False, server_default="0"),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_balances_wallet_currency", "balances", ["wallet_id", "currency"])

    op.create_table(
        "ledger_entries",
        sa.Column("entry_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Numeric(24, 12), nullable=False),
        sa.Column("side", ledger_side, nullable=False),
        sa.Column("balance_after", sa.Numeric(24, 12), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("external_tx_id", sa.String(length=255)),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ledger_entries_wallet_created_at", "ledger_entries", ["wallet_id", "created_at"])

    op.create_table(
        "orders",
        sa.Column("order_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("coinbase_order_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column("product_id", sa.String(length=64), nullable=False),
        sa.Column("side", order_side, nullable=False),
        sa.Column("order_type", order_type, nullable=False),
        sa.Column("price", sa.Numeric(24, 12)),
        sa.Column("qty", sa.Numeric(24, 12), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("filled_qty", sa.Numeric(24, 12), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_orders_wallet_created_at", "orders", ["wallet_id", "created_at"])

    op.create_table(
        "transfers",
        sa.Column("transfer_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id_from", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("wallet_id_to", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Numeric(24, 12), nullable=False),
        sa.Column("coinbase_tx_id", sa.String(length=255)),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_transfers_wallet_created_at", "transfers", ["wallet_id_from", "created_at"])

    op.create_table(
        "reservations",
        sa.Column("res_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("wallet_id", sa.BigInteger(), sa.ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Numeric(24, 12), nullable=False),
        sa.Column("state", reservation_state, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False, unique=True),
    )
    op.create_index("ix_reservations_wallet_state", "reservations", ["wallet_id", "state"])

    op.create_table(
        "cost_estimates",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.order_id", ondelete="SET NULL")),
        sa.Column("ex_fee", sa.Numeric(24, 12), nullable=False),
        sa.Column("spread", sa.Numeric(24, 12), nullable=False),
        sa.Column("slippage", sa.Numeric(24, 12), nullable=False),
        sa.Column("transfer_fee", sa.Numeric(24, 12), nullable=False),
        sa.Column("total_cost", sa.Numeric(24, 12), nullable=False),
        sa.Column("decision", sa.Boolean(), nullable=False),
        sa.Column("override_flag", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_cost_estimates_created_at", "cost_estimates", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_cost_estimates_created_at", table_name="cost_estimates")
    op.drop_table("cost_estimates")

    op.drop_index("ix_reservations_wallet_state", table_name="reservations")
    op.drop_table("reservations")

    op.drop_index("ix_transfers_wallet_created_at", table_name="transfers")
    op.drop_table("transfers")

    op.drop_index("ix_orders_wallet_created_at", table_name="orders")
    op.drop_table("orders")

    op.drop_index("ix_ledger_entries_wallet_created_at", table_name="ledger_entries")
    op.drop_table("ledger_entries")

    op.drop_index("ix_balances_wallet_currency", table_name="balances")
    op.drop_table("balances")

    op.drop_index("ix_fees_snapshots_portfolio", table_name="fees_snapshots")
    op.drop_table("fees_snapshots")

    op.drop_table("wallets")

    sa.Enum(name="reservation_state").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="order_type").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="order_side").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="ledger_side").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="wallet_type").drop(op.get_bind(), checkfirst=True)
