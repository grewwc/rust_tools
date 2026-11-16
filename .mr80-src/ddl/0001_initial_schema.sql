-- ddl/0001_initial_schema.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0001_initial_schema.py 渲染
-- revision: 0001_initial
-- description: initial schema (unsigned PK, comments, idx_/uk_ naming)
-- generated at: 2026-08-28T11:46:26.858903+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0001_initial 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

CREATE TABLE alembic_version (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    version_num VARCHAR(32) NOT NULL COMMENT '当前迁移版本号',
    PRIMARY KEY (id),
    UNIQUE KEY uk_alembic_version_version_num (version_num)
) COMMENT='alembic迁移版本跟踪表';

-- Running upgrade  -> 0001_initial

CREATE TABLE resolve_tasks (
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
    PRIMARY KEY (id)
)COMMENT='解冲突任务主表';

CREATE INDEX idx_tasks_repo_mr_status ON resolve_tasks (repo_id, original_mr_id, status);

CREATE INDEX idx_tasks_status_updated ON resolve_tasks (status, updated_at);

CREATE TABLE conflict_blocks (
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
    CONSTRAINT uk_blocks_discussion UNIQUE (discussion_id)
)COMMENT='冲突块详情';

CREATE INDEX idx_conflict_blocks_task_id ON conflict_blocks (task_id);

CREATE TABLE approvals (
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

CREATE TABLE task_events (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id BIGINT NOT NULL COMMENT '关联resolve_tasks.id',
    event_type VARCHAR(64) NOT NULL COMMENT '事件类型',
    payload JSON NOT NULL COMMENT '事件原始payload',
    coco_event_id VARCHAR(128) COMMENT 'Coco事件ID(去重用)',
    occurred_at DATETIME NOT NULL COMMENT '事件发生时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_events_coco UNIQUE (task_id, coco_event_id)
)COMMENT='任务事件审计日志';

CREATE INDEX idx_events_task_time ON task_events (task_id, occurred_at);

CREATE INDEX idx_task_events_task_id ON task_events (task_id);

CREATE TABLE repo_configs (
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

INSERT INTO alembic_version (version_num) VALUES ('0001_initial');
