"""Retain Meegle work item type for Coco session name queries."""

import sqlalchemy as sa
from alembic import op

revision = "0024_coco_session_context"
down_revision = "0023_deployment_primary_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "meegle_ai_node_tasks",
        sa.Column("work_item_type_key", sa.String(128), nullable=True, comment="工作项类型Key"),
    )


def downgrade() -> None:
    op.drop_column("meegle_ai_node_tasks", "work_item_type_key")
