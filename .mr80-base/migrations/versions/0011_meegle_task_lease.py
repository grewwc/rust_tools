"""Add processing leases for Meegle AI tasks."""

import sqlalchemy as sa
from alembic import op

revision = "0011_meegle_task_lease"
down_revision = "0010_meegle_evaluation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meegle_ai_node_tasks", sa.Column("lease_owner", sa.String(64)))
    op.add_column("meegle_ai_node_tasks", sa.Column("lease_expires_at", sa.DateTime()))


def downgrade() -> None:
    op.drop_column("meegle_ai_node_tasks", "lease_expires_at")
    op.drop_column("meegle_ai_node_tasks", "lease_owner")
