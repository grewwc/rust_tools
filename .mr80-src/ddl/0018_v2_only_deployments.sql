-- ddl/0018_v2_only_deployments.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0018_v2_only_deployments.py 渲染
-- revision: 0018_v2_only_deployments
-- description: Remove single-slot worker state; retain worker and deployment history.
-- generated at: 2026-11-04T02:17:16.812822+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0018_v2_only_deployments 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0017_coding_blocked_results -> 0018_v2_only_deployments

DROP INDEX idx_aeolus_deploy_queue ON aeolus_deployments;

CREATE INDEX idx_aeolus_deploy_worker_status ON aeolus_deployments (worker_id, status, created_at);

ALTER TABLE aeolus_deployment_workers DROP COLUMN reserved_task_id;

ALTER TABLE aeolus_deployment_workers DROP COLUMN active_deployment_id;

ALTER TABLE aeolus_deployment_workers DROP COLUMN releasing;

ALTER TABLE aeolus_deployment_workers DROP COLUMN generation;

UPDATE alembic_version SET version_num='0018_v2_only_deployments' WHERE alembic_version.version_num = '0017_coding_blocked_results';
