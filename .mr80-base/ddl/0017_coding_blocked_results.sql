-- ddl/0017_coding_blocked_results.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0017_coding_blocked_results.py 渲染
-- revision: 0017_coding_blocked_results
-- description: Persist coding blockers and failure codes for human review.
-- generated at: 2026-10-27T02:10:48.434507+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0017_coding_blocked_results 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0016_lark_token_user_unique -> 0017_coding_blocked_results

ALTER TABLE meegle_ai_node_rounds ADD COLUMN unresolved_items JSON;

ALTER TABLE meegle_ai_node_rounds ADD COLUMN failure_code TEXT;

UPDATE alembic_version SET version_num='0017_coding_blocked_results' WHERE alembic_version.version_num = '0016_lark_token_user_unique';
