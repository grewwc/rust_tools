-- ddl/0021_comment_answer_claim.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0021_comment_answer_claim.py 渲染
-- revision: 0021_comment_answer_claim
-- description: Persist which worker owns answering a comment, with an expiry lease.
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0021_comment_answer_claim 或更新 → 跳过本文件
--
-- ✅ 本文件幂等,可重复执行。

-- Running upgrade 0020_pending_comment_feedback -> 0021_comment_answer_claim

CREATE TABLE IF NOT EXISTS meegle_ai_node_comment_claim (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT '关联任务ID',
    request_id VARCHAR(160) NOT NULL COMMENT '评论请求ID',
    expires_at DATETIME(6) NOT NULL COMMENT '声明过期时间',
    created_at DATETIME(6) NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_claim_task_request UNIQUE (task_id, request_id),
  KEY idx_meegle_ai_claim_expiry (expires_at)
)CHARSET=utf8mb4 COMMENT='Meegle AI节点评论处理声明';

UPDATE alembic_version SET version_num='0021_comment_answer_claim' WHERE alembic_version.version_num = '0020_pending_comment_feedback';
