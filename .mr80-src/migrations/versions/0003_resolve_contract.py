"""persist frozen MR and structured Coco resolve contract

Revision ID: 0003_resolve_contract
Revises: 0002_webhook_dedup
Create Date: 2026-08-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0003_resolve_contract"
down_revision = "0002_webhook_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    task_columns = (
        sa.Column("mr_version", sa.Integer(), nullable=True, comment="任务冻结的MR Version Number"),
        sa.Column("source_commit_sha", sa.String(64), nullable=True, comment="任务冻结的MR源分支commit"),
        sa.Column("target_commit_sha", sa.String(64), nullable=True, comment="任务冻结的MR目标分支commit"),
        sa.Column("resolve_base_sha", sa.String(64), nullable=True, comment="resolve分支创建基线commit"),
        sa.Column("resolve_commit_sha", sa.String(64), nullable=True, comment="Coco推送且服务端验收通过的commit"),
        sa.Column("coco_workspace_id", sa.String(128), nullable=True, comment="Coco Workspace ID"),
        sa.Column("coco_agent_id", sa.String(128), nullable=True, comment="Coco Agent ID"),
        sa.Column("coco_session_id", sa.String(128), nullable=True, comment="Coco Session ID"),
        sa.Column("coco_run_id", sa.String(128), nullable=True, comment="Coco当前Run ID"),
        sa.Column("coco_last_event_id", sa.String(128), nullable=True, comment="Coco最后消费的Event ID"),
        sa.Column(
            "coco_result_schema_version",
            sa.String(64),
            nullable=True,
            comment="Coco结构化结果schema版本",
        ),
        sa.Column(
            "coco_result_json",
            mysql.JSON(),
            nullable=True,
            comment="通过schema校验的Coco结构化结果",
        ),
    )
    for column in task_columns:
        op.add_column("resolve_tasks", column)

    block_columns = (
        sa.Column("conflict_id", sa.String(32), nullable=True, comment="任务内稳定冲突块ID"),
        sa.Column("source_author_email", sa.String(255), nullable=True, comment="经服务端复核的源侧author邮箱"),
        sa.Column("source_author_commit_sha", sa.String(64), nullable=True, comment="源侧最后修改commit"),
        sa.Column("source_path", sa.String(512), nullable=True, comment="源侧归因文件路径"),
        sa.Column("source_start_line", sa.Integer(), nullable=True, comment="源侧归因起始行"),
        sa.Column("source_end_line", sa.Integer(), nullable=True, comment="源侧归因结束行"),
        sa.Column("target_author_email", sa.String(255), nullable=True, comment="经服务端复核的目标侧author邮箱"),
        sa.Column("target_author_commit_sha", sa.String(64), nullable=True, comment="目标侧最后修改commit"),
        sa.Column("target_path", sa.String(512), nullable=True, comment="目标侧归因文件路径"),
        sa.Column("target_start_line", sa.Integer(), nullable=True, comment="目标侧归因起始行"),
        sa.Column("target_end_line", sa.Integer(), nullable=True, comment="目标侧归因结束行"),
        sa.Column("attribution_status", sa.String(32), nullable=True, comment="归因复核状态"),
        sa.Column("decision_summary", sa.Text(), nullable=True, comment="Agent给出的决策摘要"),
        sa.Column("introduced_fix", sa.Text(), nullable=True, comment="组合两侧后额外引入的修复"),
    )
    for column in block_columns:
        op.add_column("conflict_blocks", column)

    op.execute(
        "UPDATE conflict_blocks SET conflict_id = CONCAT('C', LPAD(block_idx + 1, 4, '0')) WHERE conflict_id IS NULL"
    )
    op.alter_column(
        "conflict_blocks",
        "conflict_id",
        existing_type=sa.String(32),
        nullable=False,
        existing_comment="任务内稳定冲突块ID",
    )
    op.create_unique_constraint(
        "uk_blocks_task_conflict",
        "conflict_blocks",
        ["task_id", "conflict_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uk_blocks_task_conflict",
        "conflict_blocks",
        type_="unique",
    )
    for name in (
        "introduced_fix",
        "decision_summary",
        "attribution_status",
        "target_end_line",
        "target_start_line",
        "target_path",
        "target_author_commit_sha",
        "target_author_email",
        "source_end_line",
        "source_start_line",
        "source_path",
        "source_author_commit_sha",
        "source_author_email",
        "conflict_id",
    ):
        op.drop_column("conflict_blocks", name)
    for name in (
        "coco_result_json",
        "coco_result_schema_version",
        "coco_last_event_id",
        "coco_run_id",
        "coco_session_id",
        "coco_agent_id",
        "coco_workspace_id",
        "resolve_commit_sha",
        "resolve_base_sha",
        "target_commit_sha",
        "source_commit_sha",
        "mr_version",
    ):
        op.drop_column("resolve_tasks", name)
