"""freeze conflict inputs and deduplicate active processing versions

Revision ID: 0006_processing_version
Revises: 0005_lark_notification_outbox
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_processing_version"
down_revision = "0005_lark_notification_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resolve_tasks",
        sa.Column(
            "conflict_files_snapshot",
            sa.JSON(),
            nullable=True,
            comment="触发时冻结的冲突文件内容",
        ),
    )
    op.execute("""
        UPDATE resolve_tasks
        SET active_key = CONCAT('v1:', SHA2(CONCAT(
            repo_id, ':', original_mr_id, ':', source_commit_sha, ':', target_commit_sha
        ), 256))
        WHERE active_key IS NOT NULL
          AND source_commit_sha IS NOT NULL AND source_commit_sha <> ''
          AND target_commit_sha IS NOT NULL AND target_commit_sha <> ''
    """)
    op.alter_column(
        "resolve_tasks",
        "active_key",
        existing_type=sa.String(191),
        existing_nullable=True,
        comment="非终态处理版本唯一键",
    )


def downgrade() -> None:
    # With parallel versions, restoring one active key per MR would collide.
    # Operators must finish/cancel extra versions before attempting downgrade.
    op.execute("""
        UPDATE resolve_tasks SET active_key = CONCAT(repo_id, ':', original_mr_id)
        WHERE active_key IS NOT NULL
    """)
    op.alter_column(
        "resolve_tasks",
        "active_key",
        existing_type=sa.String(191),
        existing_nullable=True,
        comment="非终态任务唯一键(repo_id:original_mr_id)",
    )
    op.drop_column("resolve_tasks", "conflict_files_snapshot")
