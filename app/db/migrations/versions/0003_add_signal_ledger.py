"""add signal_ledger table

Revision ID: 0003_add_signal_ledger
Revises: 0002_add_week1_tables
Create Date: 2026-02-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_add_signal_ledger"
down_revision = "0002_add_week1_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signal_ledger",
        sa.Column("signal_id", sa.Text, primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("strategy_type", sa.Text, nullable=False),
        sa.Column("trigger_id", sa.Text, nullable=False),
        sa.Column("regime_snapshot_hash", sa.Text, nullable=False),
        sa.Column("entry_price", sa.Numeric(24, 12), nullable=False),
        sa.Column("stop_price", sa.Numeric(24, 12), nullable=False),
        sa.Column("target_price", sa.Numeric(24, 12), nullable=False),
        sa.Column("stop_anchor_type", sa.Text, nullable=True),
        sa.Column("target_anchor_type", sa.Text, nullable=True),
        sa.Column("risk_r_multiple", sa.Numeric(10, 4), nullable=False),
        sa.Column("expected_hold_bars", sa.Integer, nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("thesis", sa.Text, nullable=True),
        sa.Column("screener_rank", sa.Integer, nullable=True),
        sa.Column("confidence", sa.Text, nullable=True),
        # Outcome fields — null until resolved
        sa.Column("outcome", sa.Text, nullable=True),
        sa.Column("outcome_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("r_achieved", sa.Numeric(10, 4), nullable=True),
        sa.Column("mfe_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column("mae_pct", sa.Numeric(10, 4), nullable=True),
        # Fill fields — null until filled
        sa.Column("fill_price", sa.Numeric(24, 12), nullable=True),
        sa.Column("fill_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fill_latency_ms", sa.Integer, nullable=True),
        sa.Column("slippage_bps", sa.Numeric(10, 2), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_signal_ledger_ts", "signal_ledger", ["ts"])
    op.create_index("ix_signal_ledger_symbol_ts", "signal_ledger", ["symbol", "ts"])
    op.create_index("ix_signal_ledger_outcome", "signal_ledger", ["outcome"])
    op.create_index("ix_signal_ledger_engine_version", "signal_ledger", ["engine_version"])


def downgrade() -> None:
    op.drop_index("ix_signal_ledger_engine_version", table_name="signal_ledger")
    op.drop_index("ix_signal_ledger_outcome", table_name="signal_ledger")
    op.drop_index("ix_signal_ledger_symbol_ts", table_name="signal_ledger")
    op.drop_index("ix_signal_ledger_ts", table_name="signal_ledger")
    op.drop_table("signal_ledger")
