-- ddl/0002_nextcode_webhook_dedup.sql
-- 由 scripts/export_ddl.py 从 migrations/versions/0002_nextcode_webhook_dedup.py 渲染
-- revision: 0002_webhook_dedup
-- description: add persistent NextCode webhook delivery deduplication
-- generated at: 2026-08-28T11:46:27.504399+00:00
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 若 version_num 已是 0002_webhook_dedup 或更新 → 跳过本文件
--
-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。

-- Running upgrade 0001_initial -> 0002_webhook_dedup

CREATE TABLE nextcode_webhook_deliveries (
    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
    delivery_key VARCHAR(255) NOT NULL COMMENT 'Delivery/Event去重键',
    event_id VARCHAR(128) COMMENT 'X-Vecode-Event-Id',
    delivery_id VARCHAR(128) COMMENT 'X-Vecode-Delivery',
    event_type VARCHAR(64) NOT NULL COMMENT 'X-Vecode-Event',
    status VARCHAR(16) NOT NULL COMMENT '处理状态' DEFAULT 'PROCESSING',
    processed_count INTEGER NOT NULL COMMENT '处理条目数' DEFAULT '0',
    last_error TEXT COMMENT '最后错误',
    created_at DATETIME NOT NULL COMMENT '创建时间',
    updated_at DATETIME NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    CONSTRAINT uk_nextcode_delivery_key UNIQUE (delivery_key)
)COMMENT='NextCode Webhook投递去重记录';

CREATE INDEX idx_nextcode_delivery_event ON nextcode_webhook_deliveries (event_type, created_at);

UPDATE alembic_version SET version_num='0002_webhook_dedup' WHERE alembic_version.version_num = '0001_initial';
