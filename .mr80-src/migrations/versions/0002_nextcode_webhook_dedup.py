"""add persistent NextCode webhook delivery deduplication

Revision ID: 0002_webhook_dedup
Revises: 0001_initial
Create Date: 2026-08-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0002_webhook_dedup"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nextcode_webhook_deliveries",
        sa.Column(
            "id",
            mysql.BIGINT(unsigned=True),
            primary_key=True,
            autoincrement=True,
            comment="主键ID",
        ),
        sa.Column(
            "delivery_key",
            sa.String(255),
            nullable=False,
            comment="Delivery/Event去重键",
        ),
        sa.Column(
            "event_id",
            sa.String(128),
            nullable=True,
            comment="X-Vecode-Event-Id",
        ),
        sa.Column(
            "delivery_id",
            sa.String(128),
            nullable=True,
            comment="X-Vecode-Delivery",
        ),
        sa.Column(
            "event_type",
            sa.String(64),
            nullable=False,
            comment="X-Vecode-Event",
        ),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="PROCESSING",
            comment="处理状态",
        ),
        sa.Column(
            "processed_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="处理条目数",
        ),
        sa.Column("last_error", sa.Text(), nullable=True, comment="最后错误"),
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
        sa.UniqueConstraint("delivery_key", name="uk_nextcode_delivery_key"),
        comment="NextCode Webhook投递去重记录",
    )
    op.create_index(
        "idx_nextcode_delivery_event",
        "nextcode_webhook_deliveries",
        ["event_type", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_nextcode_delivery_event",
        table_name="nextcode_webhook_deliveries",
    )
    op.drop_table("nextcode_webhook_deliveries")
