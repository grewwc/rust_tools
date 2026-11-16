"""Persist Meegle AI node tasks and coding rounds.

Revision ID: 0008_meegle_ai_node
Revises: 0007_task_trace
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0008_meegle_ai_node"
down_revision = "0007_task_trace"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meegle_ai_node_tasks",
        sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键ID"),
        sa.Column("task_id", sa.String(128), nullable=False, comment="Webhook幂等任务ID"),
        sa.Column("project_key", sa.String(128), nullable=False, comment="空间Key"),
        sa.Column("work_item_id", sa.BigInteger(), nullable=False, comment="工作项ID"),
        sa.Column("state_key", sa.String(128), nullable=False, comment="节点状态Key"),
        sa.Column("node_uuid", sa.String(128), nullable=False, comment="AI节点UUID"),
        sa.Column("user_key", sa.String(128), nullable=False, comment="触发用户Key"),
        sa.Column("props", mysql.JSON(), nullable=False, comment="AI节点属性"),
        sa.Column("action_token", sa.String(128), nullable=False, comment="卡片操作令牌"),
        sa.Column("status", sa.String(32), nullable=False, comment="任务状态"),
        sa.Column("requirement_url", sa.String(1024), nullable=False, comment="需求文档链接"),
        sa.Column("requirement_text", mysql.MEDIUMTEXT(), nullable=True, comment="需求文档正文"),
        sa.Column("evaluation", mysql.JSON(), nullable=True, comment="输入评估结果"),
        sa.Column("repo_id", sa.String(128), nullable=False, comment="代码仓库ID"),
        sa.Column("work_branch", sa.String(255), nullable=False, comment="编码分支"),
        sa.Column("coco_session_id", sa.String(128), nullable=False, comment="Coco Session ID"),
        sa.Column("coco_run_id", sa.String(128), nullable=False, comment="Coco Run ID"),
        sa.Column("coco_last_event_id", sa.String(128), nullable=False, comment="Coco最后事件ID"),
        sa.Column("mr_id", sa.String(128), nullable=False, comment="代码评审ID"),
        sa.Column("mr_iid", sa.BigInteger(), nullable=True, comment="代码评审IID"),
        sa.Column("mr_url", sa.String(1024), nullable=False, comment="代码评审链接"),
        sa.Column(
            "platform_status_done",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
            comment="Meegle状态是否已完成",
        ),
        sa.Column("claimed_request_ids", mysql.JSON(), nullable=False, comment="已处理的卡片请求ID"),
        sa.Column("last_error", sa.Text(), nullable=True, comment="最后错误"),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False, comment="创建时间"),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False, comment="更新时间"),
        sa.UniqueConstraint("task_id", name="uk_meegle_ai_tasks_task_id"),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点任务",
    )
    op.create_index(
        "idx_meegle_ai_tasks_state_node",
        "meegle_ai_node_tasks",
        ["project_key", "work_item_id", "state_key", "id"],
    )
    op.create_index(
        "idx_meegle_ai_tasks_uuid_node",
        "meegle_ai_node_tasks",
        ["project_key", "work_item_id", "node_uuid", "id"],
    )
    op.create_index(
        "idx_meegle_ai_tasks_status_updated",
        "meegle_ai_node_tasks",
        ["status", "updated_at"],
    )

    op.create_table(
        "meegle_ai_node_rounds",
        sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键ID"),
        sa.Column("task_id", sa.String(128), nullable=False, comment="关联任务ID"),
        sa.Column("round_no", sa.Integer(), nullable=False, comment="轮次序号"),
        sa.Column("status", sa.String(32), nullable=False, comment="轮次状态"),
        sa.Column("feedback", sa.Text(), nullable=True, comment="用户反馈"),
        sa.Column("summary", sa.Text(), nullable=True, comment="本轮摘要"),
        sa.Column("commit_sha", sa.String(64), nullable=False, comment="提交SHA"),
        sa.Column("code_url", sa.String(1024), nullable=False, comment="代码链接"),
        sa.Column("tests", mysql.JSON(), nullable=False, comment="测试记录"),
        sa.Column("error", sa.Text(), nullable=True, comment="本轮错误"),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=False, comment="开始时间"),
        sa.Column("finished_at", mysql.DATETIME(fsp=6), nullable=True, comment="结束时间"),
        sa.UniqueConstraint("task_id", "round_no", name="uk_meegle_ai_rounds_task_no"),
        mysql_charset="utf8mb4",
        comment="Meegle AI节点编码交互轮次",
    )
    op.create_index(
        "idx_meegle_ai_rounds_task",
        "meegle_ai_node_rounds",
        ["task_id", "round_no"],
    )


def downgrade() -> None:
    op.drop_table("meegle_ai_node_rounds")
    op.drop_table("meegle_ai_node_tasks")
