-- ddl/0022_question_records.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0022_question_records.py 渲染
-- revision: 0022_question_records
-- description: Persist review questions so a failed answer can be recovered later.
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0022_question_records 或更新 → 跳过本文件
--
-- ✅ 本文件幂等,可重复执行。

-- Running upgrade 0021_comment_answer_claim -> 0022_question_records

CREATE TABLE IF NOT EXISTS meegle_ai_node_question_record (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id VARCHAR(128) NOT NULL COMMENT '关联任务ID',
    comment_id VARCHAR(160) NOT NULL COMMENT '提问评论ID',
    repo_id VARCHAR(128) NOT NULL COMMENT '仓库ID',
    mr_iid INTEGER NOT NULL COMMENT 'MR编号',
    discussion_id VARCHAR(160) COMMENT '回复所在讨论ID',
    position JSON COMMENT '回复位置',
    parent JSON COMMENT '被引用的父评论',
    question_text TEXT NOT NULL COMMENT '问题正文',
    author_email VARCHAR(160) NOT NULL COMMENT '提问者邮箱',
    status VARCHAR(32) NOT NULL COMMENT '处理状态' DEFAULT 'pending',
    answer_text TEXT COMMENT '已生成的回答正文',
    session_id VARCHAR(160) COMMENT 'Coco会话ID',
    run_id VARCHAR(160) COMMENT 'Coco问答运行ID',
    event_id VARCHAR(160) COMMENT 'Coco问答事件ID',
    attempts INTEGER NOT NULL COMMENT '回答尝试次数' DEFAULT '0',
    created_at DATETIME(6) NOT NULL COMMENT '创建时间',
    updated_at DATETIME(6) NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_meegle_ai_question_task_comment UNIQUE (task_id, comment_id),
  KEY idx_meegle_ai_question_status (status)
)CHARSET=utf8mb4 COMMENT='Meegle AI节点评论提问记录';

UPDATE alembic_version SET version_num='0022_question_records' WHERE alembic_version.version_num = '0021_comment_answer_claim';
