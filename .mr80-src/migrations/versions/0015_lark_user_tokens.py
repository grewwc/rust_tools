"""Store per-user Feishu OAuth tokens for reading requirement documents."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BIGINT, DATETIME

revision = "0015_lark_user_tokens"
down_revision = "0014_optional_worker_vmid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lark_user_tokens",
        sa.Column(
            "id",
            BIGINT(unsigned=True).with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("open_id", sa.String(64), nullable=False),
        sa.Column("user_key", sa.String(128), nullable=False, server_default=""),
        sa.Column("refresh_token", sa.Text(), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=False),
        sa.Column("access_expires_at", sa.DateTime(timezone=True)),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True).with_variant(DATETIME(fsp=6), "mysql"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True).with_variant(DATETIME(fsp=6), "mysql"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("open_id", name="uk_lark_user_tokens_open_id"),
    )
    op.create_index("idx_lark_user_tokens_user_key", "lark_user_tokens", ["user_key"])


def downgrade() -> None:
    op.drop_index("idx_lark_user_tokens_user_key", table_name="lark_user_tokens")
    op.drop_table("lark_user_tokens")
