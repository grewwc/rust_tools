-- ddl/0009_meegle_ai_node_attempts.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0009_meegle_ai_node_attempts.py 渲染
-- revision: 0009_meegle_attempts
-- description: Separate Meegle coding rounds from Coco execution attempts.
-- generated at: 2026-10-14T08:00:29.171434+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0009_meegle_attempts 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0008_meegle_ai_node -> 0009_meegle_attempts

ALTER TABLE meegle_ai_node_rounds ADD COLUMN trigger_type VARCHAR(32) NOT NULL COMMENT '轮次触发类型' DEFAULT 'initial';

ALTER TABLE meegle_ai_node_rounds ADD COLUMN reviewed_round_no INTEGER COMMENT '反馈针对的轮次';

ALTER TABLE meegle_ai_node_rounds ADD COLUMN base_commit_sha VARCHAR(64) NOT NULL COMMENT '本轮起始提交SHA' DEFAULT '';

UPDATE meegle_ai_node_rounds SET trigger_type = 'review_feedback', reviewed_round_no = round_no - 1 WHERE round_no > 1;

CREATE TABLE meegle_ai_node_attempts (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT '关联任务ID',
    round_no INTEGER NOT NULL COMMENT '业务轮次序号',
    attempt_no INTEGER NOT NULL COMMENT '轮内执行序号',
    status VARCHAR(32) NOT NULL COMMENT '执行状态',
    session_id VARCHAR(128) NOT NULL COMMENT 'Coco Session ID',
    run_id VARCHAR(128) NOT NULL COMMENT 'Coco Run ID',
    event_id VARCHAR(128) NOT NULL COMMENT 'Coco起始事件ID',
    error TEXT COMMENT '执行错误',
    started_at DATETIME(6) NOT NULL COMMENT '开始时间',
    finished_at DATETIME(6) COMMENT '结束时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_attempts_task_round_no UNIQUE (task_id, round_no, attempt_no)
)COMMENT='Meegle AI节点轮次执行尝试' CHARSET=utf8mb4;

CREATE INDEX idx_meegle_ai_attempts_task_round ON meegle_ai_node_attempts (task_id, round_no, attempt_no);

UPDATE alembic_version SET version_num='0009_meegle_attempts' WHERE alembic_version.version_num = '0008_meegle_ai_node';
