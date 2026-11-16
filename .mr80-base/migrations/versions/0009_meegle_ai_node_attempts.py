"""Separate Meegle coding rounds from Coco execution attempts.

Revision ID: 0009_meegle_attempts
Revises: 0008_meegle_ai_node
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0009_meegle_attempts"
down_revision = "0008_meegle_ai_node"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "meegle_ai_node_rounds",
        sa.Column(
            "trigger_type",
            sa.String(32),
            nullable=False,
            server_default="initial",
            comment="轮次触发类型",
        ),
    )
    op.add_column(
        "meegle_ai_node_rounds",
        sa.Column(
            "reviewed_round_no",
            sa.Integer(),
            nullable=True,
            comment="反馈针对的轮次",
        ),
    )
    op.add_column(
        "meegle_ai_node_rounds",
        sa.Column(
            "base_commit_sha",
            sa.String(64),
            nullable=False,
            server_default="",
            comment="本轮起始提交SHA",
        ),
    )
    op.execute(
        "UPDATE meegle_ai_node_rounds "
        "SET trigger_type = 'review_feedback', "
        "reviewed_round_no = round_no - 1 "
        "WHERE round_no > 1"
    )

    op.create_table(
        "meegle_ai_node_attempts",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            primary_key=True,
            autoincrement=True,
            comment="主键ID",
        ),
        sa.Column("task_id", sa.String(128), nullable=False, comment="关联任务ID"),
        sa.Column("round_no", sa.Integer(), nullable=False, comment="业务轮次序号"),
        sa.Column("attempt_no", sa.Integer(), nullable=False, comment="轮内执行序号"),
        sa.Column("status", sa.String(32), nullable=False, comment="执行状态"),
        sa.Column("session_id", sa.String(128), nullable=False, comment="Coco Session ID"),
        sa.Column("run_id", sa.String(128), nullable=False, comment="Coco Run ID"),
        sa.Column("event_id", sa.String(128), nullable=False, comment="Coco起始事件ID"),
        sa.Column("error", sa.Text(), nullable=True, comment="执行错误"),
        sa.Column(
            "started_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="开始时间",
        ),
        sa.Column(
            "finished_at",
            mysql.DATETIME(fsp=6),
            nullable=True,
            comment="结束时间",
        ),
        sa.UniqueConstraint(
            "task_id",
            "round_no",
            "attempt_no",
            name="uk_meegle_ai_attempts_task_round_no",
        ),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点轮次执行尝试",
    )
    op.create_index(
        "idx_meegle_ai_attempts_task_round",
        "meegle_ai_node_attempts",
        ["task_id", "round_no", "attempt_no"],
    )


def downgrade() -> None:
    op.drop_table("meegle_ai_node_attempts")
    op.drop_column("meegle_ai_node_rounds", "base_commit_sha")
    op.drop_column("meegle_ai_node_rounds", "reviewed_round_no")
    op.drop_column("meegle_ai_node_rounds", "trigger_type")
