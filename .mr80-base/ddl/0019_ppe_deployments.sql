-- ddl/0019_ppe_deployments.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0019_ppe_deployments.py 渲染
-- revision: 0019_ppe_deployments
-- description: Persist task-owned PPE environments and resumable deployment requests.
-- generated at: 2026-11-04T07:47:19.211933+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0019_ppe_deployments 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0018_v2_only_deployments -> 0019_ppe_deployments

CREATE TABLE aeolus_ppe_environments (
    environment_id VARCHAR(32) NOT NULL,
    task_id VARCHAR(128) NOT NULL,
    generation INTEGER NOT NULL,
    name VARCHAR(30) NOT NULL,
    platform_id VARCHAR(64),
    resource_state VARCHAR(32) NOT NULL,
    active_deployment_id VARCHAR(32),
    current_deployment_id VARCHAR(32),
    profile_revision VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL,
    data JSON NOT NULL,
    PRIMARY KEY (environment_id),
    CONSTRAINT uk_aeolus_ppe_generation UNIQUE (task_id, generation),
    UNIQUE (name)
);

CREATE INDEX ix_aeolus_ppe_environments_task_id ON aeolus_ppe_environments (task_id);

CREATE TABLE aeolus_ppe_deployments (
    deployment_id VARCHAR(32) NOT NULL,
    task_id VARCHAR(128) NOT NULL,
    environment_id VARCHAR(32) NOT NULL,
    request_id VARCHAR(128) NOT NULL,
    sequence INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    commit_sha VARCHAR(40) NOT NULL,
    profile_revision VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    stage VARCHAR(32) NOT NULL,
    created_at DATETIME NOT NULL,
    next_poll_at DATETIME NOT NULL,
    lease_owner VARCHAR(32),
    lease_until DATETIME,
    error TEXT NOT NULL,
    data JSON NOT NULL,
    PRIMARY KEY (deployment_id),
    CONSTRAINT uk_aeolus_ppe_request UNIQUE (task_id, request_id),
    CONSTRAINT uk_aeolus_ppe_sequence UNIQUE (task_id, sequence)
);

CREATE INDEX ix_aeolus_ppe_deployments_task_id ON aeolus_ppe_deployments (task_id);

CREATE INDEX ix_aeolus_ppe_deployments_environment_id ON aeolus_ppe_deployments (environment_id);

CREATE INDEX idx_aeolus_ppe_poll ON aeolus_ppe_deployments (status, next_poll_at);

UPDATE alembic_version SET version_num='0019_ppe_deployments' WHERE alembic_version.version_num = '0018_v2_only_deployments';
