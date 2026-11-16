"""Remove single-slot worker state; retain worker and deployment history."""

import sqlalchemy as sa
from alembic import op

revision = "0018_v2_only_deployments"
down_revision = "0017_coding_blocked_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Stop deployment workers before upgrading. Existing service processes are not modified.
    op.drop_index("idx_aeolus_deploy_queue", table_name="aeolus_deployments")
    op.create_index("idx_aeolus_deploy_worker_status", "aeolus_deployments", ["worker_id", "status", "created_at"])
    for name in ("reserved_task_id", "active_deployment_id", "releasing", "generation"):
        op.drop_column("aeolus_deployment_workers", name)


def downgrade() -> None:
    # Restore an empty slot, never resurrect a historical process reservation.
    op.add_column(
        "aeolus_deployment_workers", sa.Column("generation", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "aeolus_deployment_workers", sa.Column("releasing", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("aeolus_deployment_workers", sa.Column("reserved_task_id", sa.String(128), nullable=True))
    op.add_column("aeolus_deployment_workers", sa.Column("active_deployment_id", sa.String(64), nullable=True))
    op.drop_index("idx_aeolus_deploy_worker_status", table_name="aeolus_deployments")
    op.create_index("idx_aeolus_deploy_queue", "aeolus_deployments", ["worker_id", "status", "created_at"])
