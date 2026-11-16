"""Persist comment feedback acknowledged before a coding round can start."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0020_pending_comment_feedback"
down_revision = "0019_ppe_deployments"
branch_labels = None
depends_on = None

# This revision only creates a table and its index, so the exported DDL can use
# IF NOT EXISTS and stay safe when the operator reruns it.
IDEMPOTENT_DDL = True


def upgrade() -> None:
    op.create_table(
        "meegle_ai_node_pending_feedback",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            primary_key=True,
            autoincrement=True,
            comment="主键ID",
        ),
        sa.Column("task_id", sa.String(128), nullable=False, comment="关联任务ID"),
        sa.Column("comment_id", sa.String(128), nullable=False, comment="平台评论ID"),
        sa.Column("feedback", sa.Text(), nullable=False, comment="待提交反馈"),
        sa.Column(
            "available_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="聚合窗口结束时间",
        ),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="创建时间",
        ),
        sa.UniqueConstraint(
            "task_id", "comment_id", name="uk_meegle_ai_pending_task_comment"
        ),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点待处理评论反馈",
    )
    op.create_index(
        "idx_meegle_ai_pending_task",
        "meegle_ai_node_pending_feedback",
        ["task_id", "available_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_meegle_ai_pending_task",
        table_name="meegle_ai_node_pending_feedback",
    )
    op.drop_table("meegle_ai_node_pending_feedback")
