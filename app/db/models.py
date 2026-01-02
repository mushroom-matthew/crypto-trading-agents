"""SQLAlchemy ORM models for the internal ledger."""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all models."""


class WalletType(enum.Enum):
    """Wallet types we track internally."""

    COINBASE_SPOT = "COINBASE_SPOT"
    COINBASE_TRADING = "COINBASE_TRADING"
    EXTERNAL = "EXTERNAL"


class LedgerSide(enum.Enum):
    """Debit or credit postings."""

    debit = "debit"
    credit = "credit"


class OrderSide(enum.Enum):
    """Order direction."""

    buy = "buy"
    sell = "sell"


class OrderType(enum.Enum):
    """Order type."""

    limit = "limit"
    market = "market"


class ReservationState(enum.Enum):
    """Reservation lifecycle states."""

    active = "active"
    consumed = "consumed"
    canceled = "canceled"


class BacktestStatus(enum.Enum):
    """Backtest run status."""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class RiskAllocationStatus(enum.Enum):
    """Risk allocation lifecycle states."""

    claimed = "claimed"
    used = "used"
    released = "released"
    expired = "expired"


PK_BIGINT = BigInteger().with_variant(Integer, "sqlite")


class Wallet(Base):
    """Internal wallet mapped to a Coinbase account or portfolio."""

    __tablename__ = "wallets"

    wallet_id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    coinbase_account_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    portfolio_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    type: Mapped[WalletType] = mapped_column(Enum(WalletType, name="wallet_type"))
    tradeable_fraction: Mapped[Decimal] = mapped_column(Numeric(4, 3), default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    balances: Mapped[list["Balance"]] = relationship(back_populates="wallet")
    ledger_entries: Mapped[list["LedgerEntry"]] = relationship(back_populates="wallet")
    orders: Mapped[list["Order"]] = relationship(back_populates="wallet")
    reservations: Mapped[list["Reservation"]] = relationship(back_populates="wallet")


class Balance(Base):
    """Snapshot of balances fetched from Coinbase."""

    __tablename__ = "balances"
    __table_args__ = (
        Index("ix_balances_wallet_currency", "wallet_id", "currency"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False)
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    available: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    hold: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False, default=Decimal("0"))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    wallet: Mapped["Wallet"] = relationship(back_populates="balances")


class LedgerEntry(Base):
    """Double-entry ledger postings."""

    __tablename__ = "ledger_entries"
    __table_args__ = (
        Index("ix_ledger_entries_wallet_created_at", "wallet_id", "created_at"),
        UniqueConstraint("idempotency_key"),
    )

    entry_id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False)
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    side: Mapped[LedgerSide] = mapped_column(Enum(LedgerSide, name="ledger_side"), nullable=False)
    balance_after: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    external_tx_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    wallet: Mapped["Wallet"] = relationship(back_populates="ledger_entries")


class Order(Base):
    """Orders placed via Coinbase Advanced Trade APIs."""

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("coinbase_order_id"),
        Index("ix_orders_wallet_created_at", "wallet_id", "created_at"),
    )

    order_id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False)
    coinbase_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide, name="order_side"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType, name="order_type"), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(24, 12), nullable=True)
    qty: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    filled_qty: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False, default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    wallet: Mapped["Wallet"] = relationship(back_populates="orders")


class Transfer(Base):
    """Transfers initiated to move funds into trading portfolios."""

    __tablename__ = "transfers"
    __table_args__ = (
        UniqueConstraint("idempotency_key"),
        Index("ix_transfers_wallet_created_at", "wallet_id_from", "created_at"),
    )

    transfer_id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    wallet_id_from: Mapped[int] = mapped_column(
        ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False
    )
    wallet_id_to: Mapped[int] = mapped_column(ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False)
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    coinbase_tx_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Reservation(Base):
    """Currency reservations to guard against double-spend."""

    __tablename__ = "reservations"
    __table_args__ = (
        UniqueConstraint("idempotency_key"),
        Index("ix_reservations_wallet_state", "wallet_id", "state"),
    )

    res_id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("wallets.wallet_id", ondelete="CASCADE"), nullable=False)
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    state: Mapped[ReservationState] = mapped_column(
        Enum(ReservationState, name="reservation_state"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    wallet: Mapped["Wallet"] = relationship(back_populates="reservations")


class FeesSnapshot(Base):
    """Cached maker/taker fee information per portfolio."""

    __tablename__ = "fees_snapshots"
    __table_args__ = (
        Index("ix_fees_snapshots_portfolio", "portfolio_id"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(String(255), nullable=False)
    maker_rate: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    taker_rate: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    tier_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CostEstimate(Base):
    """Persisted cost estimation decisions for auditability."""

    __tablename__ = "cost_estimates"
    __table_args__ = (
        Index("ix_cost_estimates_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    order_id: Mapped[Optional[int]] = mapped_column(ForeignKey("orders.order_id", ondelete="SET NULL"), nullable=True)
    ex_fee: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    spread: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    slippage: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    transfer_fee: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    total_cost: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    decision: Mapped[bool] = mapped_column(nullable=False)
    override_flag: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BlockEvent(Base):
    """Individual trade block events with full context for UI visibility."""

    __tablename__ = "block_events"
    __table_args__ = (
        Index("ix_block_events_ts_reason", "timestamp", "reason"),
        Index("ix_block_events_run_id", "run_id"),
        Index("ix_block_events_correlation_id", "correlation_id"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    trigger_id: Mapped[str] = mapped_column(String(255), nullable=False)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    reason: Mapped[str] = mapped_column(String(50), nullable=False)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class RiskAllocation(Base):
    """Risk budget tracking (claimed → used → released)."""

    __tablename__ = "risk_allocations"
    __table_args__ = (
        Index("ix_risk_allocations_run_id", "run_id"),
        Index("ix_risk_allocations_correlation_id", "correlation_id"),
        Index("ix_risk_allocations_claim_ts", "claim_timestamp"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    trigger_id: Mapped[str] = mapped_column(String(255), nullable=False)
    claim_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claim_amount: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    release_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    release_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8), nullable=True)
    status: Mapped[RiskAllocationStatus] = mapped_column(Enum(RiskAllocationStatus, name="risk_allocation_status"), nullable=False)


class PositionSnapshot(Base):
    """Point-in-time position state for live trading."""

    __tablename__ = "position_snapshots"
    __table_args__ = (
        Index("ix_position_snapshots_ts_symbol", "timestamp", "symbol"),
        Index("ix_position_snapshots_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    mark_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8), nullable=True)
    unrealized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8), nullable=True)


class BacktestRun(Base):
    """Backtest metadata and configuration."""

    __tablename__ = "backtest_runs"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_backtest_runs_run_id"),
        Index("ix_backtest_runs_status", "status"),
    )

    id: Mapped[int] = mapped_column(PK_BIGINT, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    config: Mapped[str] = mapped_column(Text, nullable=False)  # JSON config
    status: Mapped[BacktestStatus] = mapped_column(Enum(BacktestStatus, name="backtest_status"), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    candles_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    candles_processed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON results summary


__all__ = [
    "Base",
    "Wallet",
    "Balance",
    "LedgerEntry",
    "Order",
    "Transfer",
    "Reservation",
    "FeesSnapshot",
    "CostEstimate",
    "BlockEvent",
    "RiskAllocation",
    "PositionSnapshot",
    "BacktestRun",
    "WalletType",
    "LedgerSide",
    "OrderSide",
    "OrderType",
    "ReservationState",
    "BacktestStatus",
    "RiskAllocationStatus",
]
