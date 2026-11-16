"""Persist coding blockers and failure codes for human review."""

import sqlalchemy as sa
from alembic import op

revision = "0017_coding_blocked_results"
down_revision = "0016_lark_token_user_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meegle_ai_node_rounds", sa.Column("unresolved_items", sa.JSON(), nullable=True))
    op.add_column("meegle_ai_node_rounds", sa.Column("failure_code", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("meegle_ai_node_rounds", "failure_code")
    op.drop_column("meegle_ai_node_rounds", "unresolved_items")
