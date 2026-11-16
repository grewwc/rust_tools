"""persist resolver anchors and enforce one active task per MR

Revision ID: 0004_resolve_approval_contract
Revises: 0003_resolve_contract
Create Date: 2026-09-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_resolve_approval_contract"
down_revision = "0003_resolve_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resolve_tasks",
        sa.Column(
            "active_key",
            sa.String(191),
            nullable=True,
            comment="非终态任务唯一键(repo_id:original_mr_id)",
        ),
    )
    op.execute(
        "UPDATE resolve_tasks "
        "SET active_key = CONCAT(repo_id, ':', original_mr_id) "
        "WHERE status NOT IN ('ALL_APPROVED', 'CANCELLED', 'FAILED', 'COCO_TIMEOUT')"
    )
    op.create_unique_constraint(
        "uk_tasks_active_key",
        "resolve_tasks",
        ["active_key"],
    )
    op.create_index(
        "idx_tasks_repo_new_mr_status",
        "resolve_tasks",
        ["repo_id", "new_mr_id", "status"],
    )

    op.add_column(
        "conflict_blocks",
        sa.Column(
            "resolved_path",
            sa.String(512),
            nullable=True,
            comment="resolve commit中的文件路径",
        ),
    )
    op.add_column(
        "conflict_blocks",
        sa.Column(
            "resolved_start_line",
            sa.Integer(),
            nullable=True,
            comment="resolve commit中的起始行",
        ),
    )
    op.add_column(
        "conflict_blocks",
        sa.Column(
            "resolved_end_line",
            sa.Integer(),
            nullable=True,
            comment="resolve commit中的结束行",
        ),
    )


def downgrade() -> None:
    op.drop_column("conflict_blocks", "resolved_end_line")
    op.drop_column("conflict_blocks", "resolved_start_line")
    op.drop_column("conflict_blocks", "resolved_path")
    op.drop_index(
        "idx_tasks_repo_new_mr_status",
        table_name="resolve_tasks",
    )
    op.drop_constraint(
        "uk_tasks_active_key",
        "resolve_tasks",
        type_="unique",
    )
    op.drop_column("resolve_tasks", "active_key")
