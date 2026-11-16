"""Persist Meegle input evaluation review history.

Revision ID: 0010_meegle_evaluation
Revises: 0009_meegle_attempts
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_meegle_evaluation"
down_revision = "0009_meegle_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "meegle_ai_node_tasks",
        sa.Column("evaluation_rounds", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("meegle_ai_node_tasks", "evaluation_rounds")
