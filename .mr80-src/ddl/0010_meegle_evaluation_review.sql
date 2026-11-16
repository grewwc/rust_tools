-- ddl/0010_meegle_evaluation_review.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0010_meegle_evaluation_review.py 渲染
-- revision: 0010_meegle_evaluation
-- description: Persist Meegle input evaluation review history.
-- generated at: 2026-10-14T08:00:29.902714+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0010_meegle_evaluation 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0009_meegle_attempts -> 0010_meegle_evaluation

ALTER TABLE meegle_ai_node_tasks ADD COLUMN evaluation_rounds JSON;

UPDATE alembic_version SET version_num='0010_meegle_evaluation' WHERE alembic_version.version_num = '0009_meegle_attempts';
