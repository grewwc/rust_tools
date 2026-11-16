-- ddl/0020_pending_comment_feedback.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0020_pending_comment_feedback.py 渲染
-- revision: 0020_pending_comment_feedback
-- description: Persist comment feedback acknowledged before a coding round can start.
-- generated at: 2026-11-06T11:15:49.613011+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0020_pending_comment_feedback 或更新 → 跳过本文件
--
-- ✅ 本文件幂等,可重复执行。

-- Running upgrade 0019_ppe_deployments -> 0020_pending_comment_feedback

CREATE TABLE IF NOT EXISTS meegle_ai_node_pending_feedback (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT '关联任务ID',
    comment_id VARCHAR(128) NOT NULL COMMENT '平台评论ID',
    feedback TEXT NOT NULL COMMENT '待提交反馈',
    available_at DATETIME(6) NOT NULL COMMENT '聚合窗口结束时间',
    created_at DATETIME(6) NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_pending_task_comment UNIQUE (task_id, comment_id),
  KEY idx_meegle_ai_pending_task (task_id, available_at)
)CHARSET=utf8mb4 COMMENT='Meegle AI节点待处理评论反馈';

UPDATE alembic_version SET version_num='0020_pending_comment_feedback' WHERE alembic_version.version_num = '0019_ppe_deployments';
