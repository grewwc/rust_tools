"""Persist which worker owns answering a comment, with an expiry lease."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0021_comment_answer_claim"
down_revision = "0020_pending_comment_feedback"
branch_labels = None
depends_on = None

# This revision only creates a table and its index, so the exported DDL can use
# IF NOT EXISTS and stay safe when the operator reruns it.
IDEMPOTENT_DDL = True


def upgrade() -> None:
    op.create_table(
        "meegle_ai_node_comment_claim",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            primary_key=True,
            autoincrement=True,
            comment="主键ID",
        ),
        sa.Column("task_id", sa.String(128), nullable=False, comment="关联任务ID"),
        sa.Column("request_id", sa.String(160), nullable=False, comment="评论请求ID"),
        sa.Column(
            "expires_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="声明过期时间",
        ),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="创建时间",
        ),
        sa.UniqueConstraint(
            "task_id", "request_id", name="uk_meegle_ai_claim_task_request"
        ),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点评论处理声明",
    )
    op.create_index(
        "idx_meegle_ai_claim_expiry",
        "meegle_ai_node_comment_claim",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_meegle_ai_claim_expiry",
        table_name="meegle_ai_node_comment_claim",
    )
    op.drop_table("meegle_ai_node_comment_claim")
