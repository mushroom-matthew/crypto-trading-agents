"""add setup_event_ledger table

Revision ID: 0004_add_setup_event_ledger
Revises: 0003_add_signal_ledger
Create Date: 2026-02-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_add_setup_event_ledger"
down_revision = "0003_add_signal_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "setup_event_ledger",
        sa.Column("setup_event_id", sa.Text, primary_key=True),
        sa.Column("setup_chain_id", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("engine_semver", sa.Text, nullable=False),
        sa.Column("feature_schema_version", sa.Text, nullable=False),
        sa.Column("strategy_template_version", sa.Text, nullable=True),
        sa.Column("feature_snapshot", sa.JSON, nullable=False),
        sa.Column("feature_snapshot_hash", sa.Text, nullable=False),
        sa.Column("compression_range_high", sa.Numeric(24, 12), nullable=True),
        sa.Column("compression_range_low", sa.Numeric(24, 12), nullable=True),
        sa.Column("compression_atr_at_detection", sa.Numeric(24, 12), nullable=True),
        sa.Column("session_type", sa.Text, nullable=False),
        sa.Column("time_in_session_sin", sa.Float, nullable=False),
        sa.Column("time_in_session_cos", sa.Float, nullable=False),
        sa.Column("is_weekend", sa.Boolean, nullable=False),
        sa.Column("asset_class", sa.Text, nullable=False, server_default="crypto"),
        # Model scores (null until model is trained)
        sa.Column("model_quality_score", sa.Float, nullable=True),
        sa.Column("p_cont_1r", sa.Float, nullable=True),
        sa.Column("p_false_breakout", sa.Float, nullable=True),
        sa.Column("p_atr_expand", sa.Float, nullable=True),
        sa.Column("model_version", sa.Text, nullable=True),
        # Outcome (null until reconciled by SetupOutcomeReconciler)
        sa.Column("outcome", sa.Text, nullable=True),
        sa.Column("outcome_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("bars_to_outcome", sa.Integer, nullable=True),
        sa.Column("mfe_pct", sa.Float, nullable=True),
        sa.Column("mae_pct", sa.Float, nullable=True),
        sa.Column("r_achieved", sa.Float, nullable=True),
        sa.Column("ttl_bars", sa.Integer, nullable=False, server_default="48"),
        # Links
        sa.Column("signal_event_id", sa.Text, nullable=True),
    )
    op.create_index("idx_sel_symbol_ts", "setup_event_ledger", ["symbol", "ts"])
    op.create_index("idx_sel_chain", "setup_event_ledger", ["setup_chain_id"])
    op.create_index("idx_sel_state", "setup_event_ledger", ["state"])
    op.create_index("idx_sel_fschema", "setup_event_ledger", ["feature_schema_version"])


def downgrade() -> None:
    op.drop_index("idx_sel_fschema", table_name="setup_event_ledger")
    op.drop_index("idx_sel_state", table_name="setup_event_ledger")
    op.drop_index("idx_sel_chain", table_name="setup_event_ledger")
    op.drop_index("idx_sel_symbol_ts", table_name="setup_event_ledger")
    op.drop_table("setup_event_ledger")
