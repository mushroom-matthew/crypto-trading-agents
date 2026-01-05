"""Add week1 tables: BlockEvent, RiskAllocation, PositionSnapshot, BacktestRun."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_add_week1_tables"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enums for new tables
    backtest_status = sa.Enum("queued", "running", "completed", "failed", name="backtest_status")
    risk_allocation_status = sa.Enum("claimed", "used", "released", "expired", name="risk_allocation_status")

    # BlockEvent table
    op.create_table(
        "block_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("correlation_id", sa.String(length=255), nullable=True),
        sa.Column("trigger_id", sa.String(length=255), nullable=False),
        sa.Column("symbol", sa.String(length=50), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("qty", sa.Numeric(18, 8), nullable=False),
        sa.Column("reason", sa.String(length=50), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
    )
    op.create_index("ix_block_events_ts_reason", "block_events", ["timestamp", "reason"])
    op.create_index("ix_block_events_run_id", "block_events", ["run_id"])
    op.create_index("ix_block_events_correlation_id", "block_events", ["correlation_id"])

    # RiskAllocation table
    op.create_table(
        "risk_allocations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("correlation_id", sa.String(length=255), nullable=True),
        sa.Column("trigger_id", sa.String(length=255), nullable=False),
        sa.Column("claim_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claim_amount", sa.Numeric(18, 8), nullable=False),
        sa.Column("release_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("release_amount", sa.Numeric(18, 8), nullable=True),
        sa.Column("status", risk_allocation_status, nullable=False),
    )
    op.create_index("ix_risk_allocations_run_id", "risk_allocations", ["run_id"])
    op.create_index("ix_risk_allocations_correlation_id", "risk_allocations", ["correlation_id"])
    op.create_index("ix_risk_allocations_claim_ts", "risk_allocations", ["claim_timestamp"])

    # PositionSnapshot table
    op.create_table(
        "position_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("symbol", sa.String(length=50), nullable=False),
        sa.Column("qty", sa.Numeric(18, 8), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("mark_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("unrealized_pnl", sa.Numeric(18, 8), nullable=True),
    )
    op.create_index("ix_position_snapshots_ts_symbol", "position_snapshots", ["timestamp", "symbol"])
    op.create_index("ix_position_snapshots_run_id", "position_snapshots", ["run_id"])

    # BacktestRun table
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=255), unique=True, nullable=False),
        sa.Column("config", sa.Text(), nullable=False),  # JSON config
        sa.Column("status", backtest_status, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("candles_total", sa.Integer(), nullable=True),
        sa.Column("candles_processed", sa.Integer(), nullable=True),
        sa.Column("results", sa.Text(), nullable=True),  # JSON results summary
    )
    op.create_index("ix_backtest_runs_status", "backtest_runs", ["status"])


def downgrade() -> None:
    # Drop BacktestRun
    op.drop_index("ix_backtest_runs_status", table_name="backtest_runs")
    op.drop_table("backtest_runs")

    # Drop PositionSnapshot
    op.drop_index("ix_position_snapshots_run_id", table_name="position_snapshots")
    op.drop_index("ix_position_snapshots_ts_symbol", table_name="position_snapshots")
    op.drop_table("position_snapshots")

    # Drop RiskAllocation
    op.drop_index("ix_risk_allocations_claim_ts", table_name="risk_allocations")
    op.drop_index("ix_risk_allocations_correlation_id", table_name="risk_allocations")
    op.drop_index("ix_risk_allocations_run_id", table_name="risk_allocations")
    op.drop_table("risk_allocations")

    # Drop BlockEvent
    op.drop_index("ix_block_events_correlation_id", table_name="block_events")
    op.drop_index("ix_block_events_run_id", table_name="block_events")
    op.drop_index("ix_block_events_ts_reason", table_name="block_events")
    op.drop_table("block_events")

    # Drop enums
    sa.Enum(name="risk_allocation_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="backtest_status").drop(op.get_bind(), checkfirst=True)
