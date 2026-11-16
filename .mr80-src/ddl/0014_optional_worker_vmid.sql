-- ddl/0014_optional_worker_vmid.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0014_optional_worker_vmid.py 渲染
-- revision: 0014_optional_worker_vmid
-- description: Allow workers to register by stable worker_id without a VMID.
-- generated at: 2026-10-13T12:34:12.404513+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0014_optional_worker_vmid 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0013_aeolus_deployments -> 0014_optional_worker_vmid

ALTER TABLE aeolus_deployment_workers MODIFY vmid VARCHAR(128) NULL;

UPDATE alembic_version SET version_num='0014_optional_worker_vmid' WHERE alembic_version.version_num = '0013_aeolus_deployments';
