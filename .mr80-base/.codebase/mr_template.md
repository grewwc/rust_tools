# MR 模板 — Codebase 平台配置

> 在 Codebase 仓库 Settings → Merge Requests → Default description template 里配置以下内容。

## 变更说明

<!-- 简述本次改动:做了什么、为什么 -->

## DDL Impact

<!-- 本次包含 DB schema 变更必填;无 schema 变更可删本节 -->

- [ ] 本次包含 DB schema 变更
- 新增/修改的迁移文件:`migrations/versions/<hash>_*.py`
- 运维需执行的 DDL:`ddl/<hash>_*.sql`
- 目标 revision:`<revision_id>`
- 锁表风险评估:
  - [ ] 不锁表(online DDL:加列 / 加索引 / MySQL 8 默认 INPLACE)
  - [ ] 锁表(改列类型 / 删列 / 改字符集)— 需安排维护窗口
- 数据回填:
  - [ ] 无(纯 schema 变更)
  - [ ] 有(在 MR 描述里附数据回填脚本)
- 回滚方案:
  - 对应 `.py` 的 `downgrade` 段
  - 或 `alembic downgrade -1`

## Reviewer 检查清单

- [ ] `models.py` 改动合理(字段类型、可空性、索引必要性)
- [ ] `migrations/versions/*.py` 的 upgrade + downgrade 都正确
- [ ] `ddl/*.sql` 已重新生成并 commit(`scripts/export_ddl.py`)
- [ ] 锁表风险可接受(或已与运维约定维护窗口)
- [ ] CI schema-check job 全绿

## Test Plan

- [ ] 本地 `alembic upgrade head` 通过
- [ ] 本地 `alembic downgrade -1` + 再 `upgrade head` 通过(对称验证)
- [ ] 单元/集成测试通过
- [ ] (如适用)本地起服务,实际跑通涉及 schema 改动的业务路径

## 运维提示(发版前)

- 查 `SELECT version_num FROM alembic_version;`
- 看 `ddl/<rev>.sql` 的目标版本,若 DB 落后则执行
- 详见 [`ddl/README.md`](../ddl/README.md)
