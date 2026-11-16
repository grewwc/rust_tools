-- ddl/0003_resolve_contract.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0003_resolve_contract.py 渲染
-- revision: 0003_resolve_contract
-- description: persist frozen MR and structured Coco resolve contract
-- generated at: 2026-08-28T11:46:28.155103+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0003_resolve_contract 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0002_webhook_dedup -> 0003_resolve_contract

ALTER TABLE resolve_tasks ADD COLUMN mr_version INTEGER COMMENT '任务冻结的MR Version Number';

ALTER TABLE resolve_tasks ADD COLUMN source_commit_sha VARCHAR(64) COMMENT '任务冻结的MR源分支commit';

ALTER TABLE resolve_tasks ADD COLUMN target_commit_sha VARCHAR(64) COMMENT '任务冻结的MR目标分支commit';

ALTER TABLE resolve_tasks ADD COLUMN resolve_base_sha VARCHAR(64) COMMENT 'resolve分支创建基线commit';

ALTER TABLE resolve_tasks ADD COLUMN resolve_commit_sha VARCHAR(64) COMMENT 'Coco推送且服务端验收通过的commit';

ALTER TABLE resolve_tasks ADD COLUMN coco_workspace_id VARCHAR(128) COMMENT 'Coco Workspace ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_agent_id VARCHAR(128) COMMENT 'Coco Agent ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_session_id VARCHAR(128) COMMENT 'Coco Session ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_run_id VARCHAR(128) COMMENT 'Coco当前Run ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_last_event_id VARCHAR(128) COMMENT 'Coco最后消费的Event ID';

ALTER TABLE resolve_tasks ADD COLUMN coco_result_schema_version VARCHAR(64) COMMENT 'Coco结构化结果schema版本';

ALTER TABLE resolve_tasks ADD COLUMN coco_result_json JSON COMMENT '通过schema校验的Coco结构化结果';

ALTER TABLE conflict_blocks ADD COLUMN conflict_id VARCHAR(32) COMMENT '任务内稳定冲突块ID';

ALTER TABLE conflict_blocks ADD COLUMN source_author_email VARCHAR(255) COMMENT '经服务端复核的源侧author邮箱';

ALTER TABLE conflict_blocks ADD COLUMN source_author_commit_sha VARCHAR(64) COMMENT '源侧最后修改commit';

ALTER TABLE conflict_blocks ADD COLUMN source_path VARCHAR(512) COMMENT '源侧归因文件路径';

ALTER TABLE conflict_blocks ADD COLUMN source_start_line INTEGER COMMENT '源侧归因起始行';

ALTER TABLE conflict_blocks ADD COLUMN source_end_line INTEGER COMMENT '源侧归因结束行';

ALTER TABLE conflict_blocks ADD COLUMN target_author_email VARCHAR(255) COMMENT '经服务端复核的目标侧author邮箱';

ALTER TABLE conflict_blocks ADD COLUMN target_author_commit_sha VARCHAR(64) COMMENT '目标侧最后修改commit';

ALTER TABLE conflict_blocks ADD COLUMN target_path VARCHAR(512) COMMENT '目标侧归因文件路径';

ALTER TABLE conflict_blocks ADD COLUMN target_start_line INTEGER COMMENT '目标侧归因起始行';

ALTER TABLE conflict_blocks ADD COLUMN target_end_line INTEGER COMMENT '目标侧归因结束行';

ALTER TABLE conflict_blocks ADD COLUMN attribution_status VARCHAR(32) COMMENT '归因复核状态';

ALTER TABLE conflict_blocks ADD COLUMN decision_summary TEXT COMMENT 'Agent给出的决策摘要';

ALTER TABLE conflict_blocks ADD COLUMN introduced_fix TEXT COMMENT '组合两侧后额外引入的修复';

UPDATE conflict_blocks SET conflict_id = CONCAT('C', LPAD(block_idx + 1, 4, '0')) WHERE conflict_id IS NULL;

ALTER TABLE conflict_blocks MODIFY conflict_id VARCHAR(32) NOT NULL COMMENT '任务内稳定冲突块ID';

ALTER TABLE conflict_blocks ADD CONSTRAINT uk_blocks_task_conflict UNIQUE (task_id, conflict_id);

UPDATE alembic_version SET version_num='0003_resolve_contract' WHERE alembic_version.version_num = '0002_webhook_dedup';
