"""Persistent optional deployment POC; leaves the code workflow unchanged."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import DATETIME

revision = "0013_aeolus_deployments"
down_revision = "0012_meegle_task_timing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "meegle_ai_node_tasks",
        sa.Column(
            "completion_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_table(
        "aeolus_deployment_workers",
        sa.Column("worker_id", sa.String(64), primary_key=True),
        sa.Column("vmid", sa.String(128), nullable=False, unique=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("ready", sa.Boolean(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("reserved_task_id", sa.String(128)),
        sa.Column("active_deployment_id", sa.String(64)),
        sa.Column("releasing", sa.Boolean(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_table(
        "aeolus_deployments",
        sa.Column("deployment_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(128), nullable=False),
        sa.Column("worker_id", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("round_no", sa.Integer(), nullable=False),
        sa.Column("commit_sha", sa.String(40), nullable=False),
        sa.Column("generation", sa.Integer()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime().with_variant(DATETIME(fsp=6), "mysql"), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.UniqueConstraint("task_id", "request_id", name="uk_aeolus_deploy_request"),
    )
    op.create_index("idx_aeolus_deploy_queue", "aeolus_deployments", ["worker_id", "status", "created_at"])
    op.create_index("idx_aeolus_deploy_task", "aeolus_deployments", ["task_id", "created_at"])


def downgrade() -> None:
    op.drop_table("aeolus_deployments")
    op.drop_table("aeolus_deployment_workers")
    op.drop_column("meegle_ai_node_tasks", "completion_requested")
