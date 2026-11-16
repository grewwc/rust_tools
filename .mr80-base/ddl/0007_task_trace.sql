-- ddl/0007_task_trace.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0007_task_trace.py 渲染
-- revision: 0007_task_trace
-- description: Persist new-task timelines and external-call bodies.
-- generated at: 2026-09-11T09:27:42.105865+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0007_task_trace 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0006_processing_version -> 0007_task_trace

ALTER TABLE resolve_tasks ADD COLUMN original_mr_url VARCHAR(512) COMMENT 'original_mr_url';

ALTER TABLE resolve_tasks ADD COLUMN trace_schema_version INTEGER NOT NULL COMMENT 'trace_schema_version' DEFAULT '0';

CREATE TABLE task_trace_events (
    id BIGINT UNSIGNED NOT NULL COMMENT 'id' AUTO_INCREMENT,
    event_uid VARCHAR(64) NOT NULL COMMENT 'event_uid',
    trace_id VARCHAR(128) NOT NULL COMMENT 'trace_id',
    occurred_at DATETIME(6) NOT NULL COMMENT 'occurred_at',
    recorded_at DATETIME NOT NULL COMMENT 'recorded_at',
    event_type VARCHAR(64) NOT NULL COMMENT 'event_type',
    summary VARCHAR(1024) NOT NULL COMMENT 'summary',
    details JSON NOT NULL COMMENT 'details',
    body_state VARCHAR(32) NOT NULL COMMENT 'body_state',
    body_bytes BIGINT NOT NULL COMMENT 'body_bytes',
    task_id BIGINT COMMENT 'task_id',
    source_task_event_id BIGINT COMMENT 'source_task_event_id',
    duration_ms BIGINT COMMENT 'duration_ms',
    task_version INTEGER COMMENT 'task_version',
    attempt INTEGER COMMENT 'attempt',
    retry_round INTEGER COMMENT 'retry_round',
    http_status INTEGER COMMENT 'http_status',
    task_status VARCHAR(32) COMMENT 'task_status',
    from_status VARCHAR(32) COMMENT 'from_status',
    to_status VARCHAR(32) COMMENT 'to_status',
    provider VARCHAR(16) COMMENT 'provider',
    operation VARCHAR(128) COMMENT 'operation',
    call_id VARCHAR(64) COMMENT 'call_id',
    operation_id VARCHAR(64) COMMENT 'operation_id',
    execution_id VARCHAR(64) COMMENT 'execution_id',
    outcome VARCHAR(32) COMMENT 'outcome',
    external_request_id VARCHAR(128) COMMENT 'external_request_id',
    PRIMARY KEY (id),
    CONSTRAINT uk_trace_uid UNIQUE (event_uid),
    CONSTRAINT uk_trace_business_event UNIQUE (source_task_event_id),
  KEY idx_trace_time (occurred_at, id),
  KEY idx_trace_task_time (task_id, occurred_at, id),
  KEY idx_trace_call (call_id, id),
  KEY idx_trace_context (trace_id, id),
  KEY idx_trace_type_time (event_type, occurred_at, id)
)CHARSET=utf8mb4 COMMENT='Task trace storage';

CREATE TABLE task_trace_event_bodies (
    id BIGINT UNSIGNED NOT NULL COMMENT 'id' AUTO_INCREMENT,
    event_id BIGINT UNSIGNED NOT NULL COMMENT 'event_id',
    body_kind VARCHAR(32) NOT NULL COMMENT 'body_kind',
    chunk_index INTEGER NOT NULL COMMENT 'chunk_index',
    content MEDIUMTEXT COMMENT 'content',
    byte_count INTEGER NOT NULL COMMENT 'byte_count',
    PRIMARY KEY (id),
    CONSTRAINT uk_trace_body_chunk UNIQUE (event_id, body_kind, chunk_index)
)CHARSET=utf8mb4 COMMENT='Task trace storage';

CREATE TABLE task_trace_links (
    id BIGINT UNSIGNED NOT NULL COMMENT 'id' AUTO_INCREMENT,
    trace_id VARCHAR(128) NOT NULL COMMENT 'trace_id',
    task_id BIGINT NOT NULL COMMENT 'task_id',
    PRIMARY KEY (id),
    CONSTRAINT uk_trace_link UNIQUE (trace_id, task_id),
  KEY idx_trace_link_task (task_id, trace_id)
)CHARSET=utf8mb4 COMMENT='Task trace storage';

UPDATE alembic_version SET version_num='0007_task_trace' WHERE alembic_version.version_num = '0006_processing_version';
