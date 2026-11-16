"""Persist task-owned PPE environments and resumable deployment requests."""

import sqlalchemy as sa
from alembic import op

revision = "0019_ppe_deployments"
down_revision = "0018_v2_only_deployments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "aeolus_ppe_environments",
        sa.Column("environment_id", sa.String(32), primary_key=True),
        sa.Column("task_id", sa.String(128), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(30), nullable=False, unique=True),
        sa.Column("platform_id", sa.String(64)),
        sa.Column("resource_state", sa.String(32), nullable=False),
        sa.Column("active_deployment_id", sa.String(32)),
        sa.Column("current_deployment_id", sa.String(32)),
        sa.Column("profile_revision", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.UniqueConstraint("task_id", "generation", name="uk_aeolus_ppe_generation"),
    )
    op.create_index("ix_aeolus_ppe_environments_task_id", "aeolus_ppe_environments", ["task_id"])
    op.create_table(
        "aeolus_ppe_deployments",
        sa.Column("deployment_id", sa.String(32), primary_key=True),
        sa.Column("task_id", sa.String(128), nullable=False),
        sa.Column("environment_id", sa.String(32), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("round_no", sa.Integer(), nullable=False),
        sa.Column("commit_sha", sa.String(40), nullable=False),
        sa.Column("profile_revision", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("next_poll_at", sa.DateTime(), nullable=False),
        sa.Column("lease_owner", sa.String(32)),
        sa.Column("lease_until", sa.DateTime()),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.UniqueConstraint("task_id", "request_id", name="uk_aeolus_ppe_request"),
        sa.UniqueConstraint("task_id", "sequence", name="uk_aeolus_ppe_sequence"),
    )
    for column in ("task_id", "environment_id"):
        op.create_index(f"ix_aeolus_ppe_deployments_{column}", "aeolus_ppe_deployments", [column])
    op.create_index("idx_aeolus_ppe_poll", "aeolus_ppe_deployments", ["status", "next_poll_at"])


def downgrade() -> None:
    op.drop_table("aeolus_ppe_deployments")
    op.drop_table("aeolus_ppe_environments")
