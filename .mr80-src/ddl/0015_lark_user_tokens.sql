-- ddl/0015_lark_user_tokens.sql
-- 对应 migrations/versions/0015_lark_user_tokens.py
-- revision: 0015_lark_user_tokens
-- description: Store per-user Feishu OAuth tokens for reading requirement documents.
--
-- 运维执行前先查:
--   SELECT version_num FROM alembic_version;
-- 仅在 version_num 为 0014_optional_worker_vmid 时执行。
-- 本文件非幂等；已执行 0015 或后续迁移时不要重跑。

CREATE TABLE lark_user_tokens (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    open_id VARCHAR(64) NOT NULL,
    user_key VARCHAR(128) NOT NULL DEFAULT '',
    refresh_token TEXT NOT NULL,
    access_token TEXT NOT NULL,
    access_expires_at DATETIME,
    refresh_expires_at DATETIME,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uk_lark_user_tokens_open_id UNIQUE (open_id)
);

CREATE INDEX idx_lark_user_tokens_user_key ON lark_user_tokens (user_key);

UPDATE alembic_version SET version_num='0015_lark_user_tokens'
WHERE alembic_version.version_num = '0014_optional_worker_vmid';