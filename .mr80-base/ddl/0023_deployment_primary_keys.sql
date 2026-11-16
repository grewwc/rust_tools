-- ddl/0023_deployment_primary_keys.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0023_deployment_primary_keys.py 渲染
-- revision: 0023_deployment_primary_keys
-- description: Add numeric deployment primary keys while retaining unique business IDs.
-- generated at: 2026-11-11T04:56:04.482027+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0023_deployment_primary_keys 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0022_question_records -> 0023_deployment_primary_keys

ALTER TABLE `aeolus_deployment_workers` DROP PRIMARY KEY, ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID' FIRST, ADD PRIMARY KEY (id), ADD UNIQUE KEY `uk_aeolus_worker_id` (`worker_id`);

ALTER TABLE `aeolus_deployments` DROP PRIMARY KEY, ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID' FIRST, ADD PRIMARY KEY (id), ADD UNIQUE KEY `uk_aeolus_deployment_id` (`deployment_id`);

ALTER TABLE `aeolus_ppe_environments` DROP PRIMARY KEY, ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID' FIRST, ADD PRIMARY KEY (id), ADD UNIQUE KEY `uk_aeolus_ppe_environment_id` (`environment_id`);

ALTER TABLE `aeolus_ppe_deployments` DROP PRIMARY KEY, ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID' FIRST, ADD PRIMARY KEY (id), ADD UNIQUE KEY `uk_aeolus_ppe_deployment_id` (`deployment_id`);

UPDATE alembic_version SET version_num='0023_deployment_primary_keys' WHERE alembic_version.version_num = '0022_question_records';
