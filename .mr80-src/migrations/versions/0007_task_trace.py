"""Persist new-task timelines and external-call bodies.

Revision ID: 0007_task_trace
Revises: 0006_processing_version
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "0007_task_trace"
down_revision = "0006_processing_version"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "resolve_tasks", sa.Column("original_mr_url", sa.String(512), nullable=True, comment="original_mr_url")
    )
    op.add_column(
        "resolve_tasks",
        sa.Column(
            "trace_schema_version", sa.Integer(), nullable=False, server_default="0", comment="trace_schema_version"
        ),
    )
    columns = [
        sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="id"),
        sa.Column("event_uid", sa.String(64), nullable=False, comment="event_uid"),
        sa.Column("trace_id", sa.String(128), nullable=False, comment="trace_id"),
        sa.Column("occurred_at", mysql.DATETIME(fsp=6), nullable=False, comment="occurred_at"),
        sa.Column("recorded_at", sa.DateTime(), nullable=False, comment="recorded_at"),
        sa.Column("event_type", sa.String(64), nullable=False, comment="event_type"),
        sa.Column("summary", sa.String(1024), nullable=False, comment="summary"),
        sa.Column("details", sa.JSON(), nullable=False, comment="details"),
        sa.Column("body_state", sa.String(32), nullable=False, comment="body_state"),
        sa.Column("body_bytes", sa.BigInteger(), nullable=False, comment="body_bytes"),
    ]
    for name in ("task_id", "source_task_event_id", "duration_ms"):
        columns.append(sa.Column(name, sa.BigInteger(), nullable=True, comment=name))
    for name in ("task_version", "attempt", "retry_round", "http_status"):
        columns.append(sa.Column(name, sa.Integer(), nullable=True, comment=name))
    for name, size in {
        "task_status": 32,
        "from_status": 32,
        "to_status": 32,
        "provider": 16,
        "operation": 128,
        "call_id": 64,
        "operation_id": 64,
        "execution_id": 64,
        "outcome": 32,
        "external_request_id": 128,
    }.items():
        columns.append(sa.Column(name, sa.String(size), nullable=True, comment=name))
    op.create_table(
        "task_trace_events",
        *columns,
        sa.UniqueConstraint("event_uid", name="uk_trace_uid"),
        sa.UniqueConstraint("source_task_event_id", name="uk_trace_business_event"),
        mysql_charset="utf8mb4",
        comment="Task trace storage",
    )
    for name, fields in {
        "idx_trace_time": ["occurred_at", "id"],
        "idx_trace_task_time": ["task_id", "occurred_at", "id"],
        "idx_trace_call": ["call_id", "id"],
        "idx_trace_context": ["trace_id", "id"],
        "idx_trace_type_time": ["event_type", "occurred_at", "id"],
    }.items():
        op.create_index(name, "task_trace_events", fields)
    op.create_table(
        "task_trace_event_bodies",
        sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="id"),
        sa.Column("event_id", mysql.BIGINT(unsigned=True), nullable=False, comment="event_id"),
        sa.Column("body_kind", sa.String(32), nullable=False, comment="body_kind"),
        sa.Column("chunk_index", sa.Integer(), nullable=False, comment="chunk_index"),
        sa.UniqueConstraint("event_id", "body_kind", "chunk_index", name="uk_trace_body_chunk"),
        sa.Column("content", mysql.MEDIUMTEXT(), nullable=True, comment="content"),
        sa.Column("byte_count", sa.Integer(), nullable=False, comment="byte_count"),
        mysql_charset="utf8mb4",
        comment="Task trace storage",
    )
    op.create_table(
        "task_trace_links",
        sa.Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="id"),
        sa.Column("trace_id", sa.String(128), nullable=False, comment="trace_id"),
        sa.Column("task_id", sa.BigInteger(), nullable=False, comment="task_id"),
        sa.UniqueConstraint("trace_id", "task_id", name="uk_trace_link"),
        mysql_charset="utf8mb4",
        comment="Task trace storage",
    )
    op.create_index("idx_trace_link_task", "task_trace_links", ["task_id", "trace_id"])


def downgrade():
    op.drop_table("task_trace_links")
    op.drop_table("task_trace_event_bodies")
    op.drop_table("task_trace_events")
    op.drop_column("resolve_tasks", "trace_schema_version")
    op.drop_column("resolve_tasks", "original_mr_url")
