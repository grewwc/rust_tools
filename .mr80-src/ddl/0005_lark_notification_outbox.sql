-- ddl/0005_lark_notification_outbox.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0005_lark_notification_outbox.py 渲染
-- revision: 0005_lark_notification_outbox
-- description: persist all-approved Lark notification delivery
-- generated at: 2026-09-04T07:49:08.640110+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0005_lark_notification_outbox 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0004_resolve_approval_contract -> 0005_lark_notification_outbox

ALTER TABLE resolve_tasks ADD COLUMN new_mr_url VARCHAR(512) COMMENT 'NextCode回读的解冲突新MR URL';

ALTER TABLE resolve_tasks ADD COLUMN trigger_author_email VARCHAR(255) COMMENT 'COMMENT触发时输入/resolve的用户邮箱';

CREATE TABLE lark_notification_outbox (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    task_id BIGINT NOT NULL COMMENT '关联resolve_tasks.id',
    notification_type VARCHAR(64) NOT NULL COMMENT '通知类型',
    recipient_email VARCHAR(255) NOT NULL COMMENT '收件人邮箱',
    recipient_open_id VARCHAR(128) COMMENT '邮箱解析得到的飞书open_id',
    mr_url VARCHAR(512) NOT NULL COMMENT '通知中的解冲突MR URL',
    request_uuid VARCHAR(50) NOT NULL COMMENT '飞书消息发送幂等键',
    status VARCHAR(16) NOT NULL COMMENT '投递状态(PENDING/SENDING/RETRY/SENT/FAILED)' DEFAULT 'PENDING',
    attempts INTEGER NOT NULL COMMENT '已领取投递次数' DEFAULT '0',
    next_retry_at DATETIME COMMENT '下次可重试时间',
    message_id VARCHAR(128) COMMENT '飞书发送成功返回的message_id',
    last_error TEXT COMMENT '最后投递错误',
    version INTEGER NOT NULL COMMENT '乐观锁版本号' DEFAULT '0',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_lark_notification_request_uuid UNIQUE (request_uuid),
    CONSTRAINT uk_lark_notification_task_type UNIQUE (task_id, notification_type)
)COMMENT='飞书通知可靠投递箱';

CREATE INDEX idx_lark_notification_status_due ON lark_notification_outbox (status, next_retry_at, updated_at);

UPDATE alembic_version SET version_num='0005_lark_notification_outbox' WHERE alembic_version.version_num = '0004_resolve_approval_contract';
