"""Allow workers to register by stable worker_id without a VMID."""

import sqlalchemy as sa
from alembic import op

revision = "0014_optional_worker_vmid"
down_revision = "0013_aeolus_deployments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("aeolus_deployment_workers", "vmid", existing_type=sa.String(128), nullable=True)


def downgrade() -> None:
    # Fill missing VMIDs before downgrading; do not invent machine identifiers.
    op.alter_column("aeolus_deployment_workers", "vmid", existing_type=sa.String(128), nullable=False)
