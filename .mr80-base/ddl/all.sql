-- ddl/all.sql — 全量 schema 快照
-- 由 scripts/export_ddl.py 渲染,对应 head revision: 0007_task_trace
-- generated at: 2026-09-11T09:27:42.773979+00:00
--
-- 用途:
--   - 空库初始化(新机房 / 本地 docker / 测试环境)
--   - 不用于增量!增量走 ddl/<rev>.sql
--
-- 仅限空库执行一次:CREATE 语句带 IF NOT EXISTS,后续 ALTER 不保证可重跑

CREATE TABLE IF NOT EXISTS alembic_version (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    version_num VARCHAR(32) NOT NULL COMMENT '当前迁移版本号',
    PRIMARY KEY (id),
    UNIQUE KEY uk_alembic_version_version_num (version_num)
) COMMENT='alembic迁移版本跟踪表';

-- Running upgrade  -> 0001_initial

CREATE TABLE IF NOT EXISTS resolve_tasks (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    repo_id VARCHAR(128) NOT NULL COMMENT '仓库ID',
    source_branch VARCHAR(255) NOT NULL COMMENT 'MR源分支',
    target_branch VARCHAR(255) NOT NULL COMMENT 'MR目标分支',
    resolve_branch VARCHAR(255) NOT NULL COMMENT '解冲突分支',
    original_mr_id BIGINT NOT NULL COMMENT '原始MR的IID',
    trigger_type VARCHAR(32) NOT NULL COMMENT '触发类型(COMMENT/AUTO_CONFLICT)',
    trigger_payload JSON NOT NULL COMMENT '触发webhook原始payload',
    status VARCHAR(32) NOT NULL COMMENT '任务状态' DEFAULT 'PENDING',
    coco_task_id VARCHAR(128) COMMENT 'Coco Agent任务ID',
    lark_doc_url VARCHAR(512) COMMENT '飞书文档URL',
    new_mr_id BIGINT COMMENT '解冲突新MR的IID',
    mr_author_email VARCHAR(255) COMMENT '原始MR作者邮箱',
    last_error TEXT COMMENT '最后错误信息',
    retry_count INTEGER NOT NULL COMMENT '重试次数' DEFAULT '0',
    version INTEGER NOT NULL COMMENT '乐观锁版本号' DEFAULT '0',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
  KEY idx_tasks_repo_mr_status (repo_id, original_mr_id, status),
  KEY idx_tasks_status_updated (status, updated_at),
  KEY idx_tasks_repo_new_mr_status (repo_id, new_mr_id, status)
)COMMENT='解冲突任务主表';

CREATE TABLE IF NOT EXISTS conflict_blocks (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id BIGINT NOT NULL COMMENT '关联resolve_tasks.id',
    block_idx INTEGER NOT NULL COMMENT '块序号(0-based)',
    file_path VARCHAR(512) NOT NULL COMMENT '冲突文件路径',
    start_line INTEGER NOT NULL COMMENT '起始行号',
    end_line INTEGER NOT NULL COMMENT '结束行号',
    ours_text TEXT COMMENT '源分支冲突内容',
    theirs_text TEXT COMMENT '目标分支冲突内容',
    ai_resolution TEXT COMMENT 'AI解冲突方案',
    author_email VARCHAR(255) COMMENT '冲突代码author邮箱(git blame)',
    discussion_id VARCHAR(128) COMMENT 'MR Discussion ID',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_blocks_task_idx UNIQUE (task_id, block_idx),
    CONSTRAINT uk_blocks_discussion UNIQUE (discussion_id),
  KEY idx_conflict_blocks_task_id (task_id)
)COMMENT='冲突块详情';

CREATE TABLE IF NOT EXISTS approvals (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    block_id BIGINT NOT NULL COMMENT '关联conflict_blocks.id',
    reviewer_email VARCHAR(255) COMMENT '审批人邮箱',
    status VARCHAR(16) NOT NULL COMMENT '审批状态' DEFAULT 'PENDING',
    reviewed_at DATETIME COMMENT '审批时间',
    comment TEXT COMMENT '审批备注(reject原因)',
    version INTEGER NOT NULL COMMENT '乐观锁版本号' DEFAULT '0',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_approvals_block_id UNIQUE (block_id)
)COMMENT='逐块审批状态';

CREATE TABLE IF NOT EXISTS task_events (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id BIGINT NOT NULL COMMENT '关联resolve_tasks.id',
    event_type VARCHAR(64) NOT NULL COMMENT '事件类型',
    payload JSON NOT NULL COMMENT '事件原始payload',
    coco_event_id VARCHAR(128) COMMENT 'Coco事件ID(去重用)',
    occurred_at DATETIME NOT NULL COMMENT '事件发生时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_events_coco UNIQUE (task_id, coco_event_id),
  KEY idx_events_task_time (task_id, occurred_at),
  KEY idx_task_events_task_id (task_id)
)COMMENT='任务事件审计日志';

CREATE TABLE IF NOT EXISTS repo_configs (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    repo_id VARCHAR(128) NOT NULL COMMENT '仓库ID',
    auto_resolve_enabled BOOL NOT NULL COMMENT '是否开启自动解冲突' DEFAULT 0,
    allowed_target_branches JSON NOT NULL COMMENT '允许自动解冲突的目标分支白名单',
    agent_id VARCHAR(128) COMMENT 'Coco Agent ID',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_repo_configs_repo_id UNIQUE (repo_id)
)COMMENT='仓库配置(自动解冲突开关等)';

INSERT INTO alembic_version (version_num) SELECT '0001_initial' WHERE NOT EXISTS (SELECT 1 FROM alembic_version);

-- Running upgrade 0001_initial -> 0002_webhook_dedup

CREATE TABLE IF NOT EXISTS nextcode_webhook_deliveries (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    delivery_key VARCHAR(255) NOT NULL COMMENT 'Delivery/Event去重键',
    event_id VARCHAR(128) COMMENT 'X-Vecode-Event-Id',
    delivery_id VARCHAR(128) COMMENT 'X-Vecode-Delivery',
    event_type VARCHAR(64) NOT NULL COMMENT 'X-Vecode-Event',
    status VARCHAR(16) NOT NULL COMMENT '处理状态' DEFAULT 'PROCESSING',
    processed_count INTEGER NOT NULL COMMENT '处理条目数' DEFAULT '0',
    last_error TEXT COMMENT '最后错误',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_nextcode_delivery_key UNIQUE (delivery_key),
  KEY idx_nextcode_delivery_event (event_type, created_at)
)COMMENT='NextCode Webhook投递去重记录';

UPDATE alembic_version SET version_num='0002_webhook_dedup' WHERE alembic_version.version_num = '0001_initial';

-- Running upgrade 0002_webhook_dedup -> 0003_resolve_contract

ALTER TABLE resolve_tasks ADD COLUMN mr_version INTEGER COMMENT '任务冻结的MR Version Number';

ALTER TABLE resolve_tasks ADD COLUMN source_commit_sha VARCHAR(64) COMMENT '任务冻结的MR源分支commit';

ALTER TABLE resolve_tasks ADD COLUMN target_commit_sha VARCHAR(64) COMMENT '任务冻结的MR目标分支commit';

ALTER TABLE resolve_tasks ADD COLUMN resolve_base_sha VARCHAR(64) COMMENT 'resolve分支创建基线commit';

ALTER TABLE resolve_tasks ADD COLUMN resolve_commit_sha VARCHAR(64) COMMENT 'Coco推送且服务端验收通过的commit';

ALTER TABLE resolve_tasks ADD COLUMN coco_workspace_id VARCHAR(128) COMMENT 'Coco Workspace ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_agent_id VARCHAR(128) COMMENT 'Coco Agent ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_session_id VARCHAR(128) COMMENT 'Coco Session ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_run_id VARCHAR(128) COMMENT 'Coco当前Run ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_last_event_id VARCHAR(128) COMMENT 'Coco最后消费的Event ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_result_schema_version VARCHAR(64) COMMENT 'Coco结构化结果schema版本';

ALTER TABLE resolve_tasks ADD COLUMN coco_result_json JSON COMMENT '通过schema校验的Coco结构化结果';

ALTER TABLE conflict_blocks ADD COLUMN conflict_id VARCHAR(32) COMMENT '任务内稳定冲突块ID';

ALTER TABLE conflict_blocks ADD COLUMN source_author_email VARCHAR(255) COMMENT '经服务端复核的源侧author邮箱';

ALTER TABLE conflict_blocks ADD COLUMN source_author_commit_sha VARCHAR(64) COMMENT '源侧最后修改commit';

ALTER TABLE conflict_blocks ADD COLUMN source_path VARCHAR(512) COMMENT '源侧归因文件路径';

ALTER TABLE conflict_blocks ADD COLUMN source_start_line INTEGER COMMENT '源侧归因起始行';

ALTER TABLE conflict_blocks ADD COLUMN source_end_line INTEGER COMMENT '源侧归因结束行';

ALTER TABLE conflict_blocks ADD COLUMN target_author_email VARCHAR(255) COMMENT '经服务端复核的目标侧author邮箱';

ALTER TABLE conflict_blocks ADD COLUMN target_author_commit_sha VARCHAR(64) COMMENT '目标侧最后修改commit';

ALTER TABLE conflict_blocks ADD COLUMN target_path VARCHAR(512) COMMENT '目标侧归因文件路径';

ALTER TABLE conflict_blocks ADD COLUMN target_start_line INTEGER COMMENT '目标侧归因起始行';

ALTER TABLE conflict_blocks ADD COLUMN target_end_line INTEGER COMMENT '目标侧归因结束行';

ALTER TABLE conflict_blocks ADD COLUMN attribution_status VARCHAR(32) COMMENT '归因复核状态';

ALTER TABLE conflict_blocks ADD COLUMN decision_summary TEXT COMMENT 'Agent给出的决策摘要';

ALTER TABLE conflict_blocks ADD COLUMN introduced_fix TEXT COMMENT '组合两侧后额外引入的修复';

UPDATE conflict_blocks SET conflict_id = CONCAT('C', LPAD(block_idx + 1, 4, '0')) WHERE conflict_id IS NULL;

ALTER TABLE conflict_blocks MODIFY conflict_id VARCHAR(32) NOT NULL COMMENT '任务内稳定冲突块ID';

ALTER TABLE conflict_blocks ADD CONSTRAINT uk_blocks_task_conflict UNIQUE (task_id, conflict_id);

UPDATE alembic_version SET version_num='0003_resolve_contract' WHERE alembic_version.version_num = '0002_webhook_dedup';

-- Running upgrade 0003_resolve_contract -> 0004_resolve_approval_contract

ALTER TABLE resolve_tasks ADD COLUMN active_key VARCHAR(191) COMMENT '非终态任务唯一键(repo_id:original_mr_id)';

UPDATE resolve_tasks SET active_key = CONCAT(repo_id, ':', original_mr_id) WHERE status NOT IN ('ALL_APPROVED', 'CANCELLED', 'FAILED', 'COCO_TIMEOUT');

ALTER TABLE resolve_tasks ADD CONSTRAINT uk_tasks_active_key UNIQUE (active_key);

ALTER TABLE conflict_blocks ADD COLUMN resolved_path VARCHAR(512) COMMENT 'resolve commit中的文件路径';

ALTER TABLE conflict_blocks ADD COLUMN resolved_start_line INTEGER COMMENT 'resolve commit中的起始行';

ALTER TABLE conflict_blocks ADD COLUMN resolved_end_line INTEGER COMMENT 'resolve commit中的结束行';

UPDATE alembic_version SET version_num='0004_resolve_approval_contract' WHERE alembic_version.version_num = '0003_resolve_contract';

-- Running upgrade 0004_resolve_approval_contract -> 0005_lark_notification_outbox

ALTER TABLE resolve_tasks ADD COLUMN new_mr_url VARCHAR(512) COMMENT 'NextCode回读的解冲突新MR URL';

ALTER TABLE resolve_tasks ADD COLUMN trigger_author_email VARCHAR(255) COMMENT 'COMMENT触发时输入/resolve的用户邮箱';

CREATE TABLE IF NOT EXISTS lark_notification_outbox (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id BIGINT NOT NULL COMMENT '关联resolve_tasks.id',
    notification_type VARCHAR(64) NOT NULL COMMENT '通知类型',
    recipient_email VARCHAR(255) NOT NULL COMMENT '收件人邮箱',
    recipient_open_id VARCHAR(128) COMMENT '邮箱解析得到的飞书open_id',
    mr_url VARCHAR(512) NOT NULL COMMENT '通知中的解冲突MR URL',
    request_uuid VARCHAR(50) NOT NULL COMMENT '飞书消息发送幂等键',
    status VARCHAR(16) NOT NULL COMMENT '投递状态(PENDING/SENDING/RETRY/SENT/FAILED)' DEFAULT 'PENDING',
    attempts INTEGER NOT NULL COMMENT '已领取投递次数' DEFAULT '0',
    next_retry_at DATETIME COMMENT '下次可重试时间',
    message_id VARCHAR(128) COMMENT '飞书发送成功返回的message_id',
    last_error TEXT COMMENT '最后投递错误',
    version INTEGER NOT NULL COMMENT '乐观锁版本号' DEFAULT '0',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_lark_notification_request_uuid UNIQUE (request_uuid),
    CONSTRAINT uk_lark_notification_task_type UNIQUE (task_id, notification_type),
  KEY idx_lark_notification_status_due (status, next_retry_at, updated_at)
)COMMENT='飞书通知可靠投递箱';

UPDATE alembic_version SET version_num='0005_lark_notification_outbox' WHERE alembic_version.version_num = '0004_resolve_approval_contract';

-- Running upgrade 0005_lark_notification_outbox -> 0006_processing_version

ALTER TABLE resolve_tasks ADD COLUMN conflict_files_snapshot JSON COMMENT '触发时冻结的冲突文件内容';

UPDATE resolve_tasks
        SET active_key = CONCAT('v1:', SHA2(CONCAT(
            repo_id, ':', original_mr_id, ':', source_commit_sha, ':', target_commit_sha
        ), 256))
        WHERE active_key IS NOT NULL
          AND source_commit_sha IS NOT NULL AND source_commit_sha <> ''
          AND target_commit_sha IS NOT NULL AND target_commit_sha <> '';

ALTER TABLE resolve_tasks MODIFY active_key VARCHAR(191) NULL COMMENT '非终态处理版本唯一键';

UPDATE alembic_version SET version_num='0006_processing_version' WHERE alembic_version.version_num = '0005_lark_notification_outbox';

-- Running upgrade 0006_processing_version -> 0007_task_trace

ALTER TABLE resolve_tasks ADD COLUMN original_mr_url VARCHAR(512) COMMENT 'original_mr_url';

ALTER TABLE resolve_tasks ADD COLUMN trace_schema_version INTEGER NOT NULL COMMENT 'trace_schema_version' DEFAULT '0';

CREATE TABLE IF NOT EXISTS task_trace_events (
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

CREATE TABLE IF NOT EXISTS task_trace_event_bodies (
    id BIGINT UNSIGNED NOT NULL COMMENT 'id' AUTO_INCREMENT,
    event_id BIGINT UNSIGNED NOT NULL COMMENT 'event_id',
    body_kind VARCHAR(32) NOT NULL COMMENT 'body_kind',
    chunk_index INTEGER NOT NULL COMMENT 'chunk_index',
    content MEDIUMTEXT COMMENT 'content',
    byte_count INTEGER NOT NULL COMMENT 'byte_count',
    PRIMARY KEY (id),
    CONSTRAINT uk_trace_body_chunk UNIQUE (event_id, body_kind, chunk_index)
)CHARSET=utf8mb4 COMMENT='Task trace storage';

CREATE TABLE IF NOT EXISTS task_trace_links (
    id BIGINT UNSIGNED NOT NULL COMMENT 'id' AUTO_INCREMENT,
    trace_id VARCHAR(128) NOT NULL COMMENT 'trace_id',
    task_id BIGINT NOT NULL COMMENT 'task_id',
    PRIMARY KEY (id),
    CONSTRAINT uk_trace_link UNIQUE (trace_id, task_id),
  KEY idx_trace_link_task (task_id, trace_id)
)CHARSET=utf8mb4 COMMENT='Task trace storage';

UPDATE alembic_version SET version_num='0007_task_trace' WHERE alembic_version.version_num = '0006_processing_version';
