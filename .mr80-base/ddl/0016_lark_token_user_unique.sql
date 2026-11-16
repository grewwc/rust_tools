-- ddl/0016_lark_token_user_unique.sql
-- 对应 migrations/versions/0016_lark_token_user_unique.py
-- revision: 0016_lark_token_user_unique
-- description: Enforce one Feishu OAuth token binding per Meegle user.
--
-- 仅在 alembic_version.version_num 为 0015_lark_user_tokens 时执行。
-- 本文件非幂等；已执行 0016 或后续迁移时不要重跑。
-- 重复 user_key 会导致添加唯一约束失败；由运维核验归属并制定修复方案，
-- 不自动选择或删除任何用户记录。失败后不得继续推进 alembic_version。

ALTER TABLE lark_user_tokens ADD CONSTRAINT uk_lark_user_tokens_user_key UNIQUE (user_key);

UPDATE alembic_version SET version_num='0016_lark_token_user_unique'
WHERE alembic_version.version_num = '0015_lark_user_tokens';