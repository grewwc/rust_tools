-- ddl/0012_meegle_task_timing.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0012_meegle_task_timing.py 渲染
-- revision: 0012_meegle_task_timing
-- description: Track automatic processing and human waiting time for new Meegle tasks.
-- generated at: 2026-10-14T08:00:31.368882+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0012_meegle_task_timing 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0011_meegle_task_lease -> 0012_meegle_task_timing

ALTER TABLE meegle_ai_node_tasks ADD COLUMN timing JSON;

UPDATE alembic_version SET version_num='0012_meegle_task_timing' WHERE alembic_version.version_num = '0011_meegle_task_lease';
