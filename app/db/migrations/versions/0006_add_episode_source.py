"""add episode_source column to episode_memory

Revision ID: 0006_add_episode_source
Revises: 0005_add_episode_memory_table
Create Date: 2026-03-03

Distinguishes execution environment of each episode so the retrieval service
can apply source-fidelity weight multipliers:
  live    → 1.0 (full weight — highest fidelity)
  paper   → 0.7 (medium weight — realistic fills but no real capital)
  backtest → 0.4 (lower weight — simulated fills, look-ahead-free but no slippage)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_add_episode_source"
down_revision = "0005_add_episode_memory_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "episode_memory",
        sa.Column(
            "episode_source",
            sa.String(16),
            nullable=False,
            server_default="live",
        ),
    )
    op.create_index(
        "ix_episode_memory_source",
        "episode_memory",
        ["episode_source"],
    )


def downgrade() -> None:
    op.drop_index("ix_episode_memory_source", table_name="episode_memory")
    op.drop_column("episode_memory", "episode_source")
