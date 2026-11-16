"""initial schema (unsigned PK, comments, idx_/uk_ naming)

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-04

SQL 审核规则适配:
  - 主键 BIGINT UNSIGNED
  - 所有列和表带 COMMENT
  - 索引命名:idx_ 普通索引,uk_ 唯一索引
  - TEXT 类型允许 NULL(避免 NOT NULL 无默认值写入失败)
  - 无 ForeignKey(目标 RDS 不支持 REFERENCES)
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # resolve_tasks
    op.create_table(
        "resolve_tasks",
        sa.Column(
            "id", mysql.BIGINT(unsigned=True),
            primary_key=True, autoincrement=True, comment="主键ID",
        ),
        sa.Column("repo_id", sa.String(128), nullable=False, comment="仓库ID"),
        sa.Column("source_branch", sa.String(255), nullable=False, comment="MR源分支"),
        sa.Column("target_branch", sa.String(255), nullable=False, comment="MR目标分支"),
        sa.Column("resolve_branch", sa.String(255), nullable=False, comment="解冲突分支"),
        sa.Column("original_mr_id", sa.BigInteger, nullable=False, comment="原始MR的IID"),
        sa.Column("trigger_type", sa.String(32), nullable=False, comment="触发类型(COMMENT/AUTO_CONFLICT)"),
        sa.Column("trigger_payload", mysql.JSON(), nullable=False, comment="触发webhook原始payload"),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING", comment="任务状态"),
        sa.Column("coco_task_id", sa.String(128), nullable=True, comment="Coco Agent任务ID"),
        sa.Column("lark_doc_url", sa.String(512), nullable=True, comment="飞书文档URL"),
        sa.Column("new_mr_id", sa.BigInteger, nullable=True, comment="解冲突新MR的IID"),
        sa.Column("mr_author_email", sa.String(255), nullable=True, comment="原始MR作者邮箱"),
        sa.Column("last_error", sa.Text(), nullable=True, comment="最后错误信息"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0", comment="重试次数"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0", comment="乐观锁版本号"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        comment="解冲突任务主表",
    )
    op.create_index(
        "idx_tasks_repo_mr_status", "resolve_tasks",
        ["repo_id", "original_mr_id", "status"],
    )
    op.create_index(
        "idx_tasks_status_updated", "resolve_tasks",
        ["status", "updated_at"],
    )

    # conflict_blocks
    op.create_table(
        "conflict_blocks",
        sa.Column(
            "id", mysql.BIGINT(unsigned=True),
            primary_key=True, autoincrement=True, comment="主键ID",
        ),
        sa.Column("task_id", sa.BigInteger, nullable=False, comment="关联resolve_tasks.id"),
        sa.Column("block_idx", sa.Integer(), nullable=False, comment="块序号(0-based)"),
        sa.Column("file_path", sa.String(512), nullable=False, comment="冲突文件路径"),
        sa.Column("start_line", sa.Integer(), nullable=False, comment="起始行号"),
        sa.Column("end_line", sa.Integer(), nullable=False, comment="结束行号"),
        sa.Column("ours_text", sa.Text(), nullable=True, comment="源分支冲突内容"),
        sa.Column("theirs_text", sa.Text(), nullable=True, comment="目标分支冲突内容"),
        sa.Column("ai_resolution", sa.Text(), nullable=True, comment="AI解冲突方案"),
        sa.Column("author_email", sa.String(255), nullable=True, comment="冲突代码author邮箱(git blame)"),
        sa.Column("discussion_id", sa.String(128), nullable=True, comment="MR Discussion ID"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.UniqueConstraint("task_id", "block_idx", name="uk_blocks_task_idx"),
        sa.UniqueConstraint("discussion_id", name="uk_blocks_discussion"),
        comment="冲突块详情",
    )
    op.create_index("idx_conflict_blocks_task_id", "conflict_blocks", ["task_id"])

    # approvals
    op.create_table(
        "approvals",
        sa.Column(
            "id", mysql.BIGINT(unsigned=True),
            primary_key=True, autoincrement=True, comment="主键ID",
        ),
        sa.Column("block_id", sa.BigInteger, nullable=False, comment="关联conflict_blocks.id"),
        sa.Column("reviewer_email", sa.String(255), nullable=True, comment="审批人邮箱"),
        sa.Column("status", sa.String(16), nullable=False, server_default="PENDING", comment="审批状态"),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True, comment="审批时间"),
        sa.Column("comment", sa.Text(), nullable=True, comment="审批备注(reject原因)"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0", comment="乐观锁版本号"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.UniqueConstraint("block_id", name="uk_approvals_block_id"),
        comment="逐块审批状态",
    )

    # task_events
    op.create_table(
        "task_events",
        sa.Column(
            "id", mysql.BIGINT(unsigned=True),
            primary_key=True, autoincrement=True, comment="主键ID",
        ),
        sa.Column("task_id", sa.BigInteger, nullable=False, comment="关联resolve_tasks.id"),
        sa.Column("event_type", sa.String(64), nullable=False, comment="事件类型"),
        sa.Column("payload", mysql.JSON(), nullable=False, comment="事件原始payload"),
        sa.Column("coco_event_id", sa.String(128), nullable=True, comment="Coco事件ID(去重用)"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, comment="事件发生时间"),
        sa.UniqueConstraint("task_id", "coco_event_id", name="uk_events_coco"),
        comment="任务事件审计日志",
    )
    op.create_index("idx_events_task_time", "task_events", ["task_id", "occurred_at"])
    op.create_index("idx_task_events_task_id", "task_events", ["task_id"])

    # repo_configs(BIGINT UNSIGNED 自增主键 + repo_id 唯一索引)
    op.create_table(
        "repo_configs",
        sa.Column(
            "id", mysql.BIGINT(unsigned=True),
            primary_key=True, autoincrement=True, comment="主键ID",
        ),
        sa.Column("repo_id", sa.String(128), nullable=False, comment="仓库ID"),
        sa.Column(
            "auto_resolve_enabled", sa.Boolean(),
            nullable=False, server_default=sa.text("0"), comment="是否开启自动解冲突",
        ),
        sa.Column("allowed_target_branches", mysql.JSON(), nullable=False, comment="允许自动解冲突的目标分支白名单"),
        sa.Column("agent_id", sa.String(128), nullable=True, comment="Coco Agent ID"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.UniqueConstraint("repo_id", name="uk_repo_configs_repo_id"),
        comment="仓库配置(自动解冲突开关等)",
    )


def downgrade() -> None:
    op.drop_table("repo_configs")
    op.drop_index("idx_task_events_task_id", table_name="task_events")
    op.drop_index("idx_events_task_time", table_name="task_events")
    op.drop_table("task_events")
    op.drop_table("approvals")
    op.drop_index("idx_conflict_blocks_task_id", table_name="conflict_blocks")
    op.drop_table("conflict_blocks")
    op.drop_index("idx_tasks_status_updated", table_name="resolve_tasks")
    op.drop_index("idx_tasks_repo_mr_status", table_name="resolve_tasks")
    op.drop_table("resolve_tasks")
