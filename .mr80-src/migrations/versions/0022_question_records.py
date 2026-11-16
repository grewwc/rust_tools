"""Persist review questions so a failed answer can be recovered later."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0022_question_records"
down_revision = "0021_comment_answer_claim"
branch_labels = None
depends_on = None

# This revision only creates a table and its index, so the exported DDL can use
# IF NOT EXISTS and stay safe when the operator reruns it.
IDEMPOTENT_DDL = True


def upgrade() -> None:
    op.create_table(
        "meegle_ai_node_question_record",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            primary_key=True,
            autoincrement=True,
            comment="主键ID",
        ),
        sa.Column("task_id", sa.String(128), nullable=False, comment="关联任务ID"),
        sa.Column("comment_id", sa.String(160), nullable=False, comment="提问评论ID"),
        sa.Column("repo_id", sa.String(128), nullable=False, comment="仓库ID"),
        sa.Column("mr_iid", sa.Integer(), nullable=False, comment="MR编号"),
        sa.Column(
            "discussion_id",
            sa.String(160),
            nullable=True,
            comment="回复所在讨论ID",
        ),
        sa.Column("position", mysql.JSON(), nullable=True, comment="回复位置"),
        sa.Column("parent", mysql.JSON(), nullable=True, comment="被引用的父评论"),
        sa.Column("question_text", sa.Text(), nullable=False, comment="问题正文"),
        sa.Column("author_email", sa.String(160), nullable=False, comment="提问者邮箱"),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="pending",
            comment="处理状态",
        ),
        sa.Column("answer_text", sa.Text(), nullable=True, comment="已生成的回答正文"),
        sa.Column("session_id", sa.String(160), nullable=True, comment="Coco会话ID"),
        sa.Column("run_id", sa.String(160), nullable=True, comment="Coco问答运行ID"),
        sa.Column("event_id", sa.String(160), nullable=True, comment="Coco问答事件ID"),
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="回答尝试次数",
        ),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            comment="更新时间",
        ),
        sa.UniqueConstraint(
            "task_id",
            "comment_id",
            name="uk_meegle_ai_question_task_comment",
        ),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点评论提问记录",
    )
    op.create_index(
        "idx_meegle_ai_question_status",
        "meegle_ai_node_question_record",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_meegle_ai_question_status",
        table_name="meegle_ai_node_question_record",
    )
    op.drop_table("meegle_ai_node_question_record")
