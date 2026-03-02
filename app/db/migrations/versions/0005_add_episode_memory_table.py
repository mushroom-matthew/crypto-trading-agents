"""add episode_memory table

Revision ID: 0005_add_episode_memory_table
Revises: 0004_add_setup_event_ledger
Create Date: 2026-03-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_add_episode_memory_table"
down_revision = "0004_add_setup_event_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "episode_memory",
        sa.Column("episode_id", sa.String(64), primary_key=True),
        sa.Column("signal_id", sa.String(64), nullable=True),
        sa.Column("symbol", sa.String(50), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=True),
        sa.Column("playbook_id", sa.String(128), nullable=True),
        sa.Column("template_id", sa.String(128), nullable=True),
        sa.Column("direction", sa.String(8), nullable=True),
        sa.Column("outcome_class", sa.String(16), nullable=False),
        sa.Column("r_achieved", sa.Numeric(10, 4), nullable=True),
        sa.Column("mfe_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column("mae_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column("hold_bars", sa.Integer, nullable=True),
        sa.Column("regime_fingerprint_hash", sa.String(64), nullable=True),
        sa.Column("failure_modes_json", sa.Text, nullable=True),
        sa.Column("entry_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_episode_memory_symbol_created",
        "episode_memory",
        ["symbol", "created_at"],
    )
    op.create_index(
        "ix_episode_memory_signal_id",
        "episode_memory",
        ["signal_id"],
    )
    op.create_index(
        "ix_episode_memory_playbook",
        "episode_memory",
        ["playbook_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_episode_memory_playbook", table_name="episode_memory")
    op.drop_index("ix_episode_memory_signal_id", table_name="episode_memory")
    op.drop_index("ix_episode_memory_symbol_created", table_name="episode_memory")
    op.drop_table("episode_memory")
