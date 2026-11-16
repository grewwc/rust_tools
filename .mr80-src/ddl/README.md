# DDL 执行说明(运维)

本目录的 `.sql` 文件由 `scripts/export_ddl.py` 从 `migrations/versions/*.py` 渲染,用于 DB schema 变更。

> **注意**:`alembic_version` 表的 PK 改成了 `UNIQUE KEY`(原始是 VARCHAR PK,过不了审核)。功能等价 —— version_num 仍然保证唯一。

## 文件说明

| 文件 | 用途 |
|------|------|
| `all.sql` | 空库初始化快照,新环境一次跑完 |
| `<rev>_<desc>.sql` | 从该 revision 的 `down_revision` 升级到该 revision 的增量 SQL |

## 发版前流程(增量迁移)

### 1. 查当前 DB 版本

```sql
SELECT version_num FROM alembic_version;
-- 例:返回 '0001_initial'
```

### 2. 看 MR 描述里的 DDL Impact

找到本次需要执行的 `ddl/<rev>_<desc>.sql`,看文件头注释的目标版本。

### 3. 比对版本,决定是否执行

按 `migrations/versions/*.py` 的 `down_revision` 依赖链确定顺序，不按字符串大小比较版本。
当前版本为目标的前置 revision 时执行对应增量；已到目标版本则跳过。
落后多个版本时依次执行缺失增量；版本不在预期链上时先对账，不直接重跑。

### 4. 执行

任选一种方式:

```bash
# 方式 A:跳板机 mysql CLI
mysql -h <rds_host> -P <port> -u <user> -p <db_name> < ddl/<rev>_<desc>.sql

# 方式 B:贴到内部 SQL 审核平台,审批后执行
```

### 5. 验证

```sql
SELECT version_num FROM alembic_version;
-- 应该返回新 revision id
```

## 全新环境(全量部署)

新机房、本地 docker、测试环境等空库，一次跑 `all.sql`:

```bash
mysql -h <host> -u <user> -p <db_name> < ddl/all.sql
```

`all.sql` 的 `CREATE TABLE`/`INDEX` 带 `IF NOT EXISTS`，但后续 migration 可能包含 `ALTER TABLE`，因此整个文件不能重复执行。已有数据库必须按当前 `alembic_version` 执行对应增量文件。

当前 `all.sql` 快照仅到 `0007_task_trace`，不能直接视为最新 schema；初始化后仍须沿迁移依赖链逐个执行后续增量文件，直到目标版本。

## 飞书授权唯一绑定（0015 → 0016）

- `0015_lark_user_tokens.sql` 建表并保留 `open_id` 唯一约束，与原始 0015 迁移一致；已部署 0015 的库不要重新建表。
- `0016_lark_token_user_unique.sql` 在 0015 上追加 `user_key` 唯一约束；保存逻辑按 `user_key` 更新，不通过 `open_id` 转移其他用户的绑定。先完成 schema 升级，再部署依赖该约束的保存逻辑。
- 执行 0016 前只检查重复键（不要导出 token）：

  ```sql
  SELECT user_key, COUNT(*) AS binding_count
  FROM lark_user_tokens
  GROUP BY user_key HAVING COUNT(*) > 1;
  ```

- 若返回记录（含重复空字符串），停止发布，由运维核验账号归属、确认数据修复方案后另行审批；迁移不会自动删除或选择任何 token。未预先处理的重复键会使添加唯一约束失败。
- SQL 执行器必须遇错即停；不要在添加约束失败后执行版本推进语句。MySQL DDL 可能自动提交，不能依靠外层事务撤销。

## 编码阻塞结果（0016 → 0017）

`0017_coding_blocked_results.sql` 仅给 `meegle_ai_node_rounds` 增加两个可空字段：
`unresolved_items JSON`、`failure_code TEXT`。历史行保持不变，读取时空值按空列表和空字符串处理。
任务和轮次状态列本身为字符串，无需修改类型。执行前确认已完成 0015、0016，不能跳过版本链；
先迁移数据库再启用新后端。旧 failed 任务不自动重分类为 blocked。

## ⚠️ 重要约束

- **增量 `.sql` 文件非幂等**(无 `IF NOT EXISTS`),**不要直接重跑**!重跑前必须查 `alembic_version` 表
- **不要手改 `.sql` 文件**!它们由 `scripts/export_ddl.py` 单向渲染,改动会被 CI 检查挡住
- 如果 SQL 执行失败,检查失败位置,联系开发者修复 `migrations/versions/*.py`,重新生成 `.sql`,不要在 DB 里手工修补

## 生成工具

开发者本地生成:

```bash
# 1. 起临时 MySQL(或用现有 docker)
docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=conciliator_ddl -p 3306:3306 mysql:8.0
export MYSQL_DSN="mysql+pymysql://root:root@127.0.0.1:3306/conciliator_ddl"
export ENV=dev

# 2. 渲染
uv run python scripts/export_ddl.py

# 3. commit 生成的文件
git add ddl/
git commit -m "chore(ddl): regenerate DDL"
```

CI 通过 `scripts/check_schema.py` 检查迁移依赖、DDL 文件对应关系和版本推进语句；
生成结果的实际 SQL 内容仍需 MR 审核。
