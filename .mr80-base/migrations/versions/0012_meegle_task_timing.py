"""Track automatic processing and human waiting time for new Meegle tasks."""

import sqlalchemy as sa
from alembic import op

revision = "0012_meegle_task_timing"
down_revision = "0011_meegle_task_lease"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meegle_ai_node_tasks", sa.Column("timing", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("meegle_ai_node_tasks", "timing")
