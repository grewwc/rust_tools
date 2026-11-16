"""Enforce one Feishu OAuth token binding per Meegle user."""

from alembic import op

revision = "0016_lark_token_user_unique"
down_revision = "0015_lark_user_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing duplicate user_keys must be reviewed by an operator first.
    # Reject them via the constraint rather than choosing or deleting tokens.
    op.create_unique_constraint(
        "uk_lark_user_tokens_user_key", "lark_user_tokens", ["user_key"]
    )


def downgrade() -> None:
    op.drop_constraint(
        "uk_lark_user_tokens_user_key", "lark_user_tokens", type_="unique"
    )
