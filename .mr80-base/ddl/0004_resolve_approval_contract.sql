-- ddl/0004_resolve_approval_contract.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0004_resolve_approval_contract.py 渲染
-- revision: 0004_resolve_approval_contract
-- description: persist resolver anchors and enforce one active task per MR
-- generated at: 2026-09-03T08:25:34.931065+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0004_resolve_approval_contract 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0003_resolve_contract -> 0004_resolve_approval_contract

ALTER TABLE resolve_tasks ADD COLUMN active_key VARCHAR(191) COMMENT '非终态任务唯一键(repo_id:original_mr_id)';

UPDATE resolve_tasks SET active_key = CONCAT(repo_id, ':', original_mr_id) WHERE status NOT IN ('ALL_APPROVED', 'CANCELLED', 'FAILED', 'COCO_TIMEOUT');

ALTER TABLE resolve_tasks ADD CONSTRAINT uk_tasks_active_key UNIQUE (active_key);

CREATE INDEX idx_tasks_repo_new_mr_status ON resolve_tasks (repo_id, new_mr_id, status);

ALTER TABLE conflict_blocks ADD COLUMN resolved_path VARCHAR(512) COMMENT 'resolve commit中的文件路径';

ALTER TABLE conflict_blocks ADD COLUMN resolved_start_line INTEGER COMMENT 'resolve commit中的起始行';

ALTER TABLE conflict_blocks ADD COLUMN resolved_end_line INTEGER COMMENT 'resolve commit中的结束行';

UPDATE alembic_version SET version_num='0004_resolve_approval_contract' WHERE alembic_version.version_num = '0003_resolve_contract';
