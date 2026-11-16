-- ddl/0008_meegle_ai_node_tasks.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0008_meegle_ai_node_tasks.py 渲染
-- revision: 0008_meegle_ai_node
-- description: Persist Meegle AI node tasks and coding rounds.
-- generated at: 2026-10-14T08:00:28.464682+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0008_meegle_ai_node 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0007_task_trace -> 0008_meegle_ai_node

CREATE TABLE meegle_ai_node_tasks (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT 'Webhook幂等任务ID',
    project_key VARCHAR(128) NOT NULL COMMENT '空间Key',
    work_item_id BIGINT NOT NULL COMMENT '工作项ID',
    state_key VARCHAR(128) NOT NULL COMMENT '节点状态Key',
    node_uuid VARCHAR(128) NOT NULL COMMENT 'AI节点UUID',
    user_key VARCHAR(128) NOT NULL COMMENT '触发用户Key',
    props JSON NOT NULL COMMENT 'AI节点属性',
    action_token VARCHAR(128) NOT NULL COMMENT '卡片操作令牌',
    status VARCHAR(32) NOT NULL COMMENT '任务状态',
    requirement_url VARCHAR(1024) NOT NULL COMMENT '需求文档链接',
    requirement_text MEDIUMTEXT COMMENT '需求文档正文',
    evaluation JSON COMMENT '输入评估结果',
    repo_id VARCHAR(128) NOT NULL COMMENT '代码仓库ID',
    work_branch VARCHAR(255) NOT NULL COMMENT '编码分支',
    coco_session_id VARCHAR(128) NOT NULL COMMENT 'Coco Session ID',
    coco_run_id VARCHAR(128) NOT NULL COMMENT 'Coco Run ID',
    coco_last_event_id VARCHAR(128) NOT NULL COMMENT 'Coco最后事件ID',
    mr_id VARCHAR(128) NOT NULL COMMENT '代码评审ID',
    mr_iid BIGINT COMMENT '代码评审IID',
    mr_url VARCHAR(1024) NOT NULL COMMENT '代码评审链接',
    platform_status_done BOOL NOT NULL COMMENT 'Meegle状态是否已完成' DEFAULT 0,
    claimed_request_ids JSON NOT NULL COMMENT '已处理的卡片请求ID',
    last_error TEXT COMMENT '最后错误',
    created_at DATETIME(6) NOT NULL COMMENT '创建时间',
    updated_at DATETIME(6) NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_tasks_task_id UNIQUE (task_id)
)COMMENT='Meegle AI节点任务' CHARSET=utf8mb4;

CREATE INDEX idx_meegle_ai_tasks_state_node ON meegle_ai_node_tasks (project_key, work_item_id, state_key, id);

CREATE INDEX idx_meegle_ai_tasks_uuid_node ON meegle_ai_node_tasks (project_key, work_item_id, node_uuid, id);

CREATE INDEX idx_meegle_ai_tasks_status_updated ON meegle_ai_node_tasks (status, updated_at);

CREATE TABLE meegle_ai_node_rounds (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT '关联任务ID',
    round_no INTEGER NOT NULL COMMENT '轮次序号',
    status VARCHAR(32) NOT NULL COMMENT '轮次状态',
    feedback TEXT COMMENT '用户反馈',
    summary TEXT COMMENT '本轮摘要',
    commit_sha VARCHAR(64) NOT NULL COMMENT '提交SHA',
    code_url VARCHAR(1024) NOT NULL COMMENT '代码链接',
    tests JSON NOT NULL COMMENT '测试记录',
    error TEXT COMMENT '本轮错误',
    started_at DATETIME(6) NOT NULL COMMENT '开始时间',
    finished_at DATETIME(6) COMMENT '结束时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_rounds_task_no UNIQUE (task_id, round_no)
)COMMENT='Meegle AI节点编码交互轮次' CHARSET=utf8mb4;

CREATE INDEX idx_meegle_ai_rounds_task ON meegle_ai_node_rounds (task_id, round_no);

UPDATE alembic_version SET version_num='0008_meegle_ai_node' WHERE alembic_version.version_num = '0007_task_trace';
