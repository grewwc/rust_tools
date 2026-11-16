-- ddl/0013_aeolus_deployments.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0013_aeolus_deployments.py 渲染
-- revision: 0013_aeolus_deployments
-- description: Persistent optional deployment POC; leaves the code workflow unchanged.
-- generated at: 2026-10-14T08:00:32.075583+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0013_aeolus_deployments 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0012_meegle_task_timing -> 0013_aeolus_deployments

ALTER TABLE meegle_ai_node_tasks ADD COLUMN completion_requested BOOL NOT NULL DEFAULT false;

CREATE TABLE aeolus_deployment_workers (
    worker_id VARCHAR(64) NOT NULL,
    vmid VARCHAR(128) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    last_seen_at DATETIME NOT NULL,
    ready BOOL NOT NULL,
    generation INTEGER NOT NULL,
    reserved_task_id VARCHAR(128),
    active_deployment_id VARCHAR(64),
    releasing BOOL NOT NULL,
    data JSON NOT NULL,
    PRIMARY KEY (worker_id),
    UNIQUE (vmid)
);

CREATE TABLE aeolus_deployments (
    deployment_id VARCHAR(64) NOT NULL,
    task_id VARCHAR(128) NOT NULL,
    worker_id VARCHAR(64) NOT NULL,
    request_id VARCHAR(128) NOT NULL,
    round_no INTEGER NOT NULL,
    commit_sha VARCHAR(40) NOT NULL,
    generation INTEGER,
    status VARCHAR(32) NOT NULL,
    stage VARCHAR(32) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    data JSON NOT NULL,
    PRIMARY KEY (deployment_id),
    CONSTRAINT uk_aeolus_deploy_request UNIQUE (task_id, request_id)
);

CREATE INDEX idx_aeolus_deploy_queue ON aeolus_deployments (worker_id, status, created_at);

CREATE INDEX idx_aeolus_deploy_task ON aeolus_deployments (task_id, created_at);

UPDATE alembic_version SET version_num='0013_aeolus_deployments' WHERE alembic_version.version_num = '0012_meegle_task_timing';
