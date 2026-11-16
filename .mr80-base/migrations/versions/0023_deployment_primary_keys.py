"""Add numeric deployment primary keys while retaining unique business IDs."""

from alembic import op

revision = "0023_deployment_primary_keys"
down_revision = "0022_question_records"
branch_labels = None
depends_on = None

# Replace each primary key atomically so AUTO_INCREMENT is always indexed.
BUSINESS_KEYS = {
    "aeolus_deployment_workers": ("worker_id", "uk_aeolus_worker_id"),
    "aeolus_deployments": ("deployment_id", "uk_aeolus_deployment_id"),
    "aeolus_ppe_environments": ("environment_id", "uk_aeolus_ppe_environment_id"),
    "aeolus_ppe_deployments": ("deployment_id", "uk_aeolus_ppe_deployment_id"),
}


def upgrade() -> None:
    for table, (business_id, unique_name) in BUSINESS_KEYS.items():
        op.execute(
            f"ALTER TABLE `{table}` DROP PRIMARY KEY, "
            "ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID' FIRST, "
            f"ADD PRIMARY KEY (id), ADD UNIQUE KEY `{unique_name}` (`{business_id}`)"
        )


def downgrade() -> None:
    for table, (business_id, unique_name) in reversed(list(BUSINESS_KEYS.items())):
        op.execute(
            f"ALTER TABLE `{table}` DROP PRIMARY KEY, DROP INDEX `{unique_name}`, "
            f"DROP COLUMN id, ADD PRIMARY KEY (`{business_id}`)"
        )
