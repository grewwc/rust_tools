-- ddl/0011_meegle_task_lease.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0011_meegle_task_lease.py 渲染
-- revision: 0011_meegle_task_lease
-- description: Add processing leases for Meegle AI tasks.
-- generated at: 2026-10-14T08:00:30.634442+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0011_meegle_task_lease 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0010_meegle_evaluation -> 0011_meegle_task_lease

ALTER TABLE meegle_ai_node_tasks ADD COLUMN lease_owner VARCHAR(64);

ALTER TABLE meegle_ai_node_tasks ADD COLUMN lease_expires_at DATETIME;

UPDATE alembic_version SET version_num='0011_meegle_task_lease' WHERE alembic_version.version_num = '0010_meegle_evaluation';
