-- ddl/0006_processing_version.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0006_processing_version.py 渲染
-- revision: 0006_processing_version
-- description: freeze conflict inputs and deduplicate active processing versions
-- generated at: 2026-09-10T12:04:31.898050+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0006_processing_version 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0005_lark_notification_outbox -> 0006_processing_version

ALTER TABLE resolve_tasks ADD COLUMN conflict_files_snapshot JSON COMMENT '触发时冻结的冲突文件内容';

UPDATE resolve_tasks
        SET active_key = CONCAT('v1:', SHA2(CONCAT(
            repo_id, ':', original_mr_id, ':', source_commit_sha, ':', target_commit_sha
        ), 256))
        WHERE active_key IS NOT NULL
          AND source_commit_sha IS NOT NULL AND source_commit_sha <> ''
          AND target_commit_sha IS NOT NULL AND target_commit_sha <> '';

ALTER TABLE resolve_tasks MODIFY active_key VARCHAR(191) NULL COMMENT '非终态处理版本唯一键';

UPDATE alembic_version SET version_num='0006_processing_version' WHERE alembic_version.version_num = '0005_lark_notification_outbox';
