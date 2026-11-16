-- ddl/0024_coco_session_context.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0024_coco_session_context.py 渲染
-- revision: 0024_coco_session_context
-- description: Retain Meegle work item type for Coco session name queries.
-- generated at: 2026-11-12T09:08:13.989874+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0024_coco_session_context 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0023_deployment_primary_keys -> 0024_coco_session_context

ALTER TABLE meegle_ai_node_tasks ADD COLUMN work_item_type_key VARCHAR(128) COMMENT '工作项类型Key';

UPDATE alembic_version SET version_num='0024_coco_session_context' WHERE alembic_version.version_num = '0023_deployment_primary_keys';
