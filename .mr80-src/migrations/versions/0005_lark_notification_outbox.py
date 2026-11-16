"""persist all-approved Lark notification delivery

Revision ID: 0005_lark_notification_outbox
Revises: 0004_resolve_approval_contract
Create Date: 2026-09-04
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0005_lark_notification_outbox"
down_revision = "0004_resolve_approval_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resolve_tasks",
        sa.Column(
            "new_mr_url",
            sa.String(512),
            nullable=True,
            comment="NextCode回读的解冲突新MR URL",
        ),
    )
    op.add_column(
        "resolve_tasks",
        sa.Column(
            "trigger_author_email",
            sa.String(255),
            nullable=True,
            comment="COMMENT触发时输入/resolve的用户邮箱",
        ),
    )
    op.create_table(
        "lark_notification_outbox",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            autoincrement=True,
            nullable=False,
            comment="主键ID",
        ),
        sa.Column(
            "task_id",
            sa.BigInteger(),
            nullable=False,
            comment="关联resolve_tasks.id",
        ),
        sa.Column(
            "notification_type",
            sa.String(64),
            nullable=False,
            comment="通知类型",
        ),
        sa.Column(
            "recipient_email",
            sa.String(255),
            nullable=False,
            comment="收件人邮箱",
        ),
        sa.Column(
            "recipient_open_id",
            sa.String(128),
            nullable=True,
            comment="邮箱解析得到的飞书open_id",
        ),
        sa.Column(
            "mr_url",
            sa.String(512),
            nullable=False,
            comment="通知中的解冲突MR URL",
        ),
        sa.Column(
            "request_uuid",
            sa.String(50),
            nullable=False,
            comment="飞书消息发送幂等键",
        ),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="PENDING",
            comment="投递状态(PENDING/SENDING/RETRY/SENT/FAILED)",
        ),
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="已领取投递次数",
        ),
        sa.Column(
            "next_retry_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="下次可重试时间",
        ),
        sa.Column(
            "message_id",
            sa.String(128),
            nullable=True,
            comment="飞书发送成功返回的message_id",
        ),
        sa.Column(
            "last_error",
            sa.Text(),
            nullable=True,
            comment="最后投递错误",
        ),
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="乐观锁版本号",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="更新时间",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_uuid",
            name="uk_lark_notification_request_uuid",
        ),
        sa.UniqueConstraint(
            "task_id",
            "notification_type",
            name="uk_lark_notification_task_type",
        ),
        comment="飞书通知可靠投递箱",
    )
    op.create_index(
        "idx_lark_notification_status_due",
        "lark_notification_outbox",
        ["status", "next_retry_at", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_lark_notification_status_due",
        table_name="lark_notification_outbox",
    )
    op.drop_table("lark_notification_outbox")
    op.drop_column("resolve_tasks", "trigger_author_email")
    op.drop_column("resolve_tasks", "new_mr_url")
