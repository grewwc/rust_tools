# `/resolve` 全部冲突块逐块 `/approval` 端到端验证方案

## 结论

本阶段可以做真实 TCE 端到端验证。审计基线中有四个会阻断验收的缺口：

1. 生产流水线仍调用旧的 Docx/XML 写入逻辑，并要求空的 `LARK_DOC_FOLDER_TOKEN`；此前已经验证通过的应用身份 Markdown 上传还没有接进 `ResolveService`。
2. `/approval` 评论发生在新建的 resolve MR 上，但 `ApprovalService` 只按 `original_mr_id` 查活跃任务。真实评论会被当成“没有活跃任务”忽略。
3. `ApprovalService` 更新数据库后不会调用 `UpdateMergeRequest`，因此 MR 描述中的确认进度不会随逐块确认刷新。
4. 当前 Discussion 使用的是 NextCode 冲突 marker 视图的 `start_line`。这个行号既不是 source/target 各自的物理行号，也不一定是解冲突后文件的行号，无法可靠地把评论挂到 resolve MR 的正确代码行。

这些缺口已在 `feat/resolve-block-approval-e2e` 实现。本轮顺序固定为：先提交代码和 `0004` DDL；再构建 SCM、执行 DDL、更新 TCE；基础预检通过后打开 pipeline；最后只跑一条“全部逐块 `/approval`”的真实链路。

本计划基于以下仓库快照完成审计：

- 分支：`feat/runtime-config-defaults`
- commit：`43878c8`
- 实现分支：`feat/resolve-block-approval-e2e`
- `agentbuddy-lock.yaml` 已加入 `.gitignore`，不会提交。

## 本分支实现结果

- 正式飞书步骤已切换为应用身份上传 Markdown，文件名包含原 MR 和 task ID，并保存可打开的 `/file/{file_token}` 链接。
- Coco 输出升级为 `conciliator.resolve.result.v2`。每块必须返回 resolved 文件、行范围和真实文本，服务端再用 resolve commit 文件内容逐行复核。
- resolve MR 增加 mergeability 门禁；Discussion 使用 resolved 坐标，创建后按当前 MR Version 回读位置。
- `/approval` 改为按 resolve MR 查任务，每次确认都会更新并回读 MR 描述，回复 `X/Y`，全部完成后再进入 `ALL_APPROVED`。
- 新增数据库原子活跃任务键、resolved 坐标、业务事件，以及 `0004_resolve_approval_contract` migration/DDL。
- 新增两分支 fixture 脚本和 TCE/RDS/NextCode 只读校验脚本。fixture 的本地 dry-run 已验证为两个文件、四个冲突块，四块的 source/target 行范围均不同。
- 本地门禁结果：Ruff 通过；pytest 为 `156 passed, 10 skipped`，10 个 skip 均因本机 Docker 不可用；schema 静态检查通过；mypy 检查 54 个源码文件通过；fixture dry-run、脚本语法检查和 `git diff --check` 通过。MySQL testcontainers 缺少的动态证据必须由 RDS/TCE 阶段补齐。

## 本阶段范围

### 这轮要验证的内容

- 创建两个隔离的测试分支：`test1` 为 source，`test2` 为 target。
- 两个分支包含两个冲突文件、四个冲突块。
- 每个冲突块在 source 和 target 中的原始行范围不同。
- 使用 Chrome 创建 `test1 -> test2` 的原始冲突 MR，并由真实用户在 MR 主线评论 `/resolve`。
- Conciliator 验签、去重、冻结 MR、创建 resolve 分支、获取冲突、调用 Coco、校验结果、上传 Markdown、创建新 MR 和逐块 Discussion。
- 新 MR 为 `resolve branch -> test2`，包含双 parent 解冲突 commit，且没有合入冲突。
- 新 MR 描述符合 spec 7.7，并在每次逐块 `/approval` 后刷新进度。
- 每个冲突块有且只有一条 Conciliator Discussion，内容符合 spec 7.8，并能落到解冲突后文件的正确行。
- 四个 Discussion 全部由 Chrome 逐个回复 `/approval`，不使用 approve all。
- 最终任务进入 `ALL_APPROVED`，四条 approval 记录均为 `APPROVED`。
- RDS、NextCode、Coco、飞书和页面观察形成同一条可关联的证据链。

### 这轮明确不验证的内容

- `/reject <reason>`、Coco 重解、`/resolve retry`。
- `/resolve approve all`、`/resolve status`、`/resolve cancel`。
- 自动合并 resolve MR。
- 飞书下载、覆盖、内容 API 回读、SHA-256、L2 密级。
- 用户 OAuth、多用户 token 表、团队知识库归档。
- rename/delete、二进制文件、submodule 等冲突类型。
- 仓库管理员权限。当前只验证“原始 MR 作者可以确认所有块”；管理员能力另行补契约。

## 依据和证据边界

本轮以以下资料为准：

- 产品文档：<https://bytedance.larkoffice.com/wiki/WYhowUp2DiwbttkT1ENcZdN8nZb>
- 本地设计规范：`docs/superpowers/specs/2026-07-21-conciliator-design.md`，重点是 7.5、7.7、7.8。
- Coco 已验证方案：`docs/superpowers/plans/2026-08-28-coco-resolve-scope-and-integration-plan.md`。
- 飞书 Markdown 上传已验证方案：`docs/superpowers/plans/2026-09-02-lark-tenant-upload-validation.md`。

证据分四层，不能互相替代：

1. 单元测试只能证明本地函数和请求体契约。
2. TCE WebShell 脚本只能证明当前 Pod、当前镜像和当前身份能调用真实 API。
3. Chrome 页面用于确认真实 MR、Discussion、描述刷新和飞书文件的人类可见效果。
4. RDS 用于确认状态、关联 ID、冲突块、审批和审计事件持久化正确。

只有四层证据都通过，才能把本阶段记为通过。

## 审计基线能力

| 能力 | 当前实现 | 本轮判断 |
|---|---|---|
| Webhook HMAC 验签 | `NextCodeClient.verify_webhook_signature()` 已按原始 body 做 HMAC-SHA256 | 已实现 |
| Webhook 去重 | `nextcode_webhook_deliveries.delivery_key` 唯一；优先 Event ID、回退 Delivery ID | 部分实现；两个 ID 都缺失时仍继续处理，不能声称完整防重放 |
| 活跃任务检查 | `/resolve` 前调用 `find_active_for_mr()` | 部分实现；“先查再插”不是原子操作，并发评论可能创建两个活跃任务 |
| MR 冻结 | 保存 MR Version、source/target commit，并从冻结 source 创建 resolve 分支 | 已实现，仍需本轮真实回归 |
| 冲突解析 | 按文件排序并生成 `C0001...`，保存 marker 视图范围和两侧文本 | 已实现 |
| Coco 执行与回读 | 校验 Session 结果、branch HEAD、双 parent、残留 marker、测试和双侧 blame | 已实现，已有 TCE 证据；缺少解冲突后逐块锚点 |
| 飞书应用身份上传 | `LarkClient` 可从 KMS 取 App Secret、换 tenant token；`upload_markdown()` 已真实上传成功 | 独立能力已验证，但没有接入业务流水线 |
| 生产飞书步骤 | `_step4_document()` 仍用 `build_lark_doc_xml()` 和 `create_doc()` | 阻断项 |
| resolve MR 创建 | 已有创建和 MR 描述回读；失败后可按分支、标题和描述恢复 | 基本实现；还缺“新 MR 无冲突”的 mergeability 门禁 |
| 逐块 Discussion | 已有 `CreateComment` 和 `GetThread` | 部分实现；使用错误的 marker 行号，且逐块创建没有可靠恢复 |
| `/approval` 解析 | `parse_comment()` 已把 `/approval` 识别为 `ApproveBlock` | 已实现 |
| 新 MR 上查任务 | 只查 `original_mr_id`，不查 `new_mr_id` | 阻断项 |
| 单块审批持久化 | approval 行加锁，写入 reviewer 和 reviewed_at | 已实现 |
| 审批回复进度 | 当前只回复“Block #N 已 approve” | 不符合 7.5；缺少 `X/Y` |
| MR 描述动态刷新 | 模板能渲染勾选项，但 ApprovalService 从未调用 `update_mr_description()` | 阻断项 |
| 全部完成状态 | `_check_all_approved()` 可进入 `ALL_APPROVED` 并通知 | 代码存在，但被新 MR 查询问题挡住，且最终描述不会刷新 |
| 任务事件审计 | `task_events` 表和 `add_event()` 已存在 | 未接线；生产代码没有写事件 |
| 现有测试 | 本轮审计时目标单测 `55 passed` | 不能证明链路可用；审批集成测试错误地把原 MR 号传给了新 MR 评论路径 |
| TCE pipeline | 当前配置为 `PIPELINE_ENABLED=false` | 在代码、DDL 和部署完成前应继续保持关闭 |

## 验收前必须完成的代码改动

### 把正式飞书步骤切到 Markdown 上传

修改：

- `src/conciliator/config.py`
- `src/conciliator/services/resolve_service.py`
- `src/conciliator/integrations/lark/drive.py`
- `src/conciliator/services/templates.py`
- `tests/unit/test_config.py`
- `tests/unit/test_lark_drive.py`
- `tests/unit/test_resolve_service.py`

具体要求：

1. `PIPELINE_ENABLED=true` 时校验 `LARK_MARKDOWN_FOLDER_TOKEN`，不再要求旧的 `LARK_DOC_FOLDER_TOKEN`。
2. `_step4_document()` 使用服务端已复核的数据调用 `build_lark_doc_markdown()`，然后调用 `upload_markdown()`。
3. 文件名固定带 task 和原 MR，便于恢复和页面核对，例如：

   ```text
   conciliator-mr-<original_mr_iid>-task-<task_id>.md
   ```

4. 使用返回的 `file_token` 构造可打开的飞书文件链接，写入 `resolve_tasks.lark_doc_url`。链接必须在 Chrome 中真实打开后才能冻结为契约。
5. tenant token 仍由服务使用 App ID 和 KMS 中的 App Secret 运行时换取并缓存；不增加用户 OAuth，不持久化 tenant token。
6. 本轮不下载文件、不查 SHA、不设 L2，也不做覆盖写。
7. `upload_all` 没有业务幂等键，响应结果不明确时不得自动重试。任务应保留确定的文件名和错误后转为 `FAILED`；本阶段不为了恢复上传而新增文件夹列表查询。

### 给每个冲突块增加“解冲突后锚点”

marker 视图行号、source 行号、target 行号和 resolved 行号是四套不同坐标。必须显式区分，不能继续复用 `conflict_blocks.start_line`。

建议把 Coco 结果升级为 `conciliator.resolve.result.v2`。每个 conflict 除现有字段外，必须返回：

```json
{
  "resolved": {
    "path": "validation/resolve_approval_e2e/policy.py",
    "start_line": 21,
    "end_line": 26,
    "text": "解冲突后该块的真实代码"
  }
}
```

Conciliator 必须做以下服务端复核：

- `resolved.path` 属于冻结冲突文件集合。
- `resolved.start_line..end_line` 合法。
- 用 `GetFile(resolve_commit)` 读取最终文件，对应行切片与 `resolved.text` 完全一致。
- 最终文件无冲突 marker。
- source/target attribution 继续分别用冻结 commit 上的 `GetBlame` 校验。
- 只有以上检查全部通过，才允许上传飞书和创建 resolve MR。

持久化规则：

- `conflict_blocks.ai_resolution` 改为保存真实的 `resolved.text`，不再拿决策摘要冒充代码方案。
- 新增 `resolved_path`、`resolved_start_line`、`resolved_end_line`。
- `decision_summary` 继续只保存可公开的简洁决策说明；不保存原始 AgentThinking。
- `coco_result_json` 保存完整 v2 结构化结果。文档由 task、conflict blocks、`coco_result_json` 和模板重新生成，不需要另建“文件全文”表。

### 修正 resolve MR 和 Discussion 创建

修改：

- `src/conciliator/integrations/nextcode/discussions.py`
- `src/conciliator/integrations/nextcode/mrs.py`
- `src/conciliator/services/resolve_service.py`
- `src/conciliator/services/templates.py`

要求：

1. resolve MR 创建并回读 source、target、description 后，轮询 `GetMergeRequestMergeability`；必须明确得到 `mergeable=true`，否则任务失败，不能进入 `AWAITING_APPROVAL`。
2. Discussion 的 `Position.Path/StartLine/EndLine` 使用 `resolved_path/resolved_start_line/resolved_end_line`，不是 marker 行号。
3. `CreateComment` 后立即 `GetThread` 回读，校验 Thread ID、MR Version、Path、Side、StartLine、EndLine。
4. Discussion body 加稳定但不影响页面阅读的 task/block 标识，支持外部写成功但本地响应丢失后的恢复，例如：

   ```html
   <!-- conciliator:task=123:block=C0001 -->
   ```

5. 不能永久假设 `thread.positions[0]` 是最新位置。按当前 MR 的 source/target/base commit ID 选择对应 Position；找不到时失败并保留证据。
6. 每个 `conflict_id` 最终必须对应一个非空、唯一的 `discussion_id`。
7. 本分支没有猜测尚未验证的 `ListThreads` 请求契约。`CreateComment` 只发送一次，成功后立即 `GetThread` 回读；响应不明确时任务转为 `FAILED`，禁止盲目重发。隐藏的 task/block 标识已经保留，待后续单独验证列表契约后再补原地恢复。

Discussion 展示内容在 spec 7.8 基础上补充双侧归因和真实 resolved 范围：

```markdown
**冲突块 #1** · `path/to/file.py:resolved_start-resolved_end`
**Source author**：source@example.com · `source_path:source_start-source_end`
**Target author**：target@example.com · `target_path:target_start-target_end`

**Ours（源分支）**：...
**Theirs（目标分支）**：...
**🤖 AI 解冲突方案**：真实 resolved 代码片段
**决策摘要**：不含原始 AgentThinking 的公开说明

---
👤 回复 `/approval`、`approve` 或 `reject <reason>` 即可确认
```

### 修正新 MR 上的任务查找和审批闭环

修改：

- `src/conciliator/repositories/_sync_repo.py`
- `src/conciliator/repositories/task_repo.py`
- `src/conciliator/services/approval_service.py`
- `src/conciliator/services/templates.py`
- `tests/integration/test_task_repo.py`
- `tests/integration/test_approval_service.py`

要求：

1. `/resolve` 触发仍按 `original_mr_id` 查活跃任务。
2. 审批命令按 `repo_id + new_mr_id` 查任务；不能把两个语义混在同一个含糊方法里。
3. `discussion_id` 反查到的 task 必须与评论所在的 `new_mr_id` 相同，避免跨 MR Thread 误操作。
4. 每次 `/approval` 都要按以下顺序执行：
   - 锁定 approval 行并幂等写 `APPROVED`；
   - 重新读取所有 block/approval，计算 `X/Y`；
   - 重新渲染完整 MR 描述并调用 `UpdateMergeRequest`；
   - `GetMergeRequest` 回读 description，必须逐字一致；
   - 在原 Discussion 回复 `✅ Block #N 已 approve（X/Y）`；
   - 若 `X == Y`，再把 task 转为 `ALL_APPROVED` 并通知 MR 作者。
5. “已经 APPROVED”的重复事件仍要执行描述对账，不能提前 return。这样第一次描述更新失败后，Webhook 重投可以修复外部状态。
6. 最后一个块确认时，先把最终描述同步成功，再进入终态，避免任务进入 `ALL_APPROVED` 后无法通过活跃任务查询重试描述更新。

MR 描述模板按 spec 7.7 固定，允许增加冻结 commit、resolve commit 和测试结果，但以下字段和动态行为不能缺：

- 标题 `## 🤖 自动解冲突报告`。
- source、target、resolve branch、冲突块数和状态。
- 可打开的飞书 Markdown 链接。
- `#1...#N` 的确认清单。
- 每次确认后只新增一个勾选；最后全部为 `[x]`，状态变为“全部确认完成”。
- 指令区保留完整产品指令，但本轮只实际执行 `/approval`。

### 补齐防重放、原子活跃任务和审计事件

修改：

- `src/conciliator/api/webhooks.py`
- `src/conciliator/repositories/models.py`
- `src/conciliator/repositories/_sync_repo.py`
- `src/conciliator/services/resolve_service.py`
- `src/conciliator/services/approval_service.py`

要求：

1. pipeline 开启时，能产生业务写入的 Webhook 如果同时缺少 Event ID 和 Delivery ID，应拒绝处理，不能只打 warning 后继续。
2. 用数据库唯一 `active_key=<repo_id>:<original_mr_id>` 原子约束同一个原 MR 只能有一个非终态任务。任务进入终态时把 `active_key` 清空。
3. `task_events` 至少记录以下可关联事件：
   - `TASK_CREATED`
   - `MR_FROZEN`
   - `RESOLVE_BRANCH_CREATED`
   - `CONFLICTS_PERSISTED`
   - `COCO_SUBMITTED`
   - `COCO_VERIFIED`
   - `LARK_MARKDOWN_UPLOADED`
   - `RESOLVE_MR_CREATED`
   - 每块一个 `DISCUSSION_CREATED`
   - 每块一个 `BLOCK_APPROVED`
   - `ALL_APPROVED`
4. 事件 payload 只保存 task/MR/branch/commit/file token/Discussion/Webhook Event ID 等非密钥证据；不得保存 App Secret、tenant token、ZTI token 或 Webhook Secret。

## DDL 方案

当前 `0003_resolve_contract` 已有任务、双侧归因、Coco 结果、飞书 URL、Discussion 和 approval 所需主体字段。无需新增表，但建议新增 `0004_resolve_approval_contract`：

### `resolve_tasks` 改动

- 新增 `active_key VARCHAR(191) NULL`。
- 新增唯一约束 `uk_tasks_active_key(active_key)`。
- 新增索引 `idx_tasks_repo_new_mr_status(repo_id, new_mr_id, status)`。

DDL 执行前先检查当前是否已有重复的非终态任务：

```sql
SELECT repo_id, original_mr_id, COUNT(*) AS active_count
FROM resolve_tasks
WHERE status NOT IN ('ALL_APPROVED', 'CANCELLED', 'FAILED', 'COCO_TIMEOUT')
GROUP BY repo_id, original_mr_id
HAVING COUNT(*) > 1;
```

结果必须为空，才能回填 `active_key` 并创建唯一约束。

### `conflict_blocks` 改动

- 新增 `resolved_path VARCHAR(512) NULL`。
- 新增 `resolved_start_line INTEGER NULL`。
- 新增 `resolved_end_line INTEGER NULL`。

不新增 `resolved_text` 列：真实代码片段写入现有 `ai_resolution`，完整结构同时保留在 `resolve_tasks.coco_result_json`。

### 不需要的数据库改动

- 不建用户 OAuth/token 表。
- 不存 tenant access token。
- 不建飞书文件内容表、SHA 表或 L2 表。
- 不新增 reject/retry 版本表；这些留到后续阶段统一设计。

迁移必须先改 Alembic，再由 `scripts/export_ddl.py` 生成：

- `migrations/versions/0004_resolve_approval_contract.py`
- `ddl/0004_resolve_approval_contract.sql`
- `ddl/all.sql`

不要手改生成后的 SQL。

## 测试夹具设计

新增 `scripts/create_resolve_approval_fixture.py`。脚本只创建并推送两个分支，不创建 MR，MR 仍由 Chrome 创建。

分支命名：

```text
verify/resolve-approval-<run-id>-test1
verify/resolve-approval-<run-id>-test2
```

只保留两个远端测试分支的做法：

1. 从指定基线创建 `test2`。
2. 在 `test2` 上提交公共 fixture 基线，得到 `fixture_base_sha`。
3. 从 `fixture_base_sha` 创建 `test1`。
4. `test1` 提交 source 侧改动；`test2` 提交 target 侧改动。
5. 在临时 clone 中执行真实 merge，只做校验后 abort。
6. 只有满足“两文件、四块、每块两侧行范围不同”才 push 两个分支。

建议文件：

```text
validation/resolve_approval_e2e/policy.py
validation/resolve_approval_e2e/formatter.py
validation/resolve_approval_e2e/tests/test_source_requirements.py
validation/resolve_approval_e2e/tests/test_target_requirements.py
validation/resolve_approval_e2e/EXPECTED.md
```

fixture 在每个冲突点之前放置互不冲突、长度不同的 source/target padding，让同一语义块在两个分支上的物理行号不同。不要只检查 marker 视图行号；脚本输出的 manifest 必须明确记录：

```json
{
  "conflict_id": "C0001",
  "path": ".../formatter.py",
  "source_range": [12, 14],
  "target_range": [17, 20]
}
```

脚本退出前校验：

- 恰好两个冲突文件、四个 marker block。
- 四个稳定 conflict ID。
- 每个 block 的 source/target range 不完全相同。
- source、target 各自测试在各自分支通过。
- 预期的组合方案在临时 resolved tree 中同时通过两侧测试。
- 输出 `/tmp/conciliator-resolve-approval-<run-id>.json`，文件权限 `0600`。

现有 `create_coco_complex_fixture.py` 不直接复用：它会创建 base/source/target 三个远端分支，而且当前四个块中只有 `formatter.py` 的两侧修改行号天然不同，不能覆盖本轮行号错位要求。

## 实现后的本地门禁

至少增加以下测试：

1. `test_config.py`：pipeline 要求 Markdown folder token，不要求旧 Docx token。
2. `test_coco_client.py`：v2 resolved 锚点严格解析；漏字段、越界、额外字段均 fail closed。
3. `test_resolve_service.py`：最终文件行切片、Markdown 上传、file URL 持久化、resolve MR mergeability、Discussion resolved position 和 GetThread 回读。
4. `test_task_repo.py`：按 `new_mr_id` 查活跃任务；`active_key` 并发唯一；终态释放 key。
5. `test_approval_service.py`：用 `mr_iid=new_mr_id` 逐块 `/approval`，验证 `1/4...4/4`、MR 描述逐次更新、最终 `ALL_APPROVED`。
6. `test_webhooks.py` 和 `test_nextcode_contracts.py`：无 Event/Delivery ID 拒绝业务写、同 Event 重投无重复副作用、最新 Thread Position 选择正确。
7. `test_templates.py`：7.7/7.8 黄金文本，确保 Markdown、链接、勾选、双侧归因和 `/approval` 提示不漂移。

本地命令：

```bash
.venv/bin/ruff check src scripts tests
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest tests/integration/test_task_repo.py tests/integration/test_approval_service.py -q
.venv/bin/python scripts/check_schema.py
.venv/bin/mypy src scripts/create_resolve_approval_fixture.py scripts/verify_resolve_approval_state.py
.venv/bin/python scripts/create_resolve_approval_fixture.py --dry-run --run-id local-check
git diff --check
```

如果本机没有 Docker，数据库集成测试必须在有 MySQL 的环境补跑，不能用 skip 当作通过。

## TCE 上线前置条件

按以下顺序执行，不能先开 pipeline 再补 DDL：

1. 从当前分支创建新的功能分支，实现代码、测试、migration 和计划更新，提交并推送。
2. 构建新的 SCM 版本，记录 Git commit、SCM 版本和镜像版本。
3. 在 RDS AutoSQL 查询当前 `alembic_version`，执行并复核 `0004_resolve_approval_contract.sql`。
4. 确认 `repo_configs` 中 repo `1106267` 的 `auto_resolve_enabled=false`。本轮只允许显式 `/resolve`，避免打开 pipeline 后旧冲突 MR 被自动触发。
5. 更新 TCE 版本，但先保持 `PIPELINE_ENABLED=false`，检查 Pod 启动和基础契约。
6. 通过后再把 `PIPELINE_ENABLED=true`，完成一次滚动发布。
7. 从实例页进入目标 Pod WebShell。不要假定容器监听 8000；读取实际 `PORT0`，请求 `127.0.0.1:${PORT0}`。

TCE 预检：

```bash
curl -fsS "http://127.0.0.1:${PORT0}/healthz"
curl -fsS "http://127.0.0.1:${PORT0}/readyz"
```

同时复跑现有只读/受控验证：

- RDS 连接。
- KMS 两个 NextCode secret alias 和飞书 App Secret alias 可读且非空。
- NextCode App OpenAPI 只读查询。
- Coco Workspace、Agent 回读和最小 Session/消息契约。
- 飞书 tenant token 获取。正式链路会自行上传报告，不在预检阶段额外上传测试文件。

任何一项失败都不要创建测试 MR，也不要打开真实 `/resolve` 写链路。

## 真实端到端执行步骤

### 创建两个测试分支

在本地运行新 fixture 脚本，记录 run ID、两个分支、公共基线、source commit、target commit、文件和四个块的两侧预期范围。

```bash
.venv/bin/python scripts/create_resolve_approval_fixture.py --run-id <run-id>
```

不要复用旧 MR、旧 task 或已经前进过的 resolve branch。

### 用 Chrome 创建原始冲突 MR

打开：

<https://code.byted.org/xiahaitao/Conciliator/merge_requests>

创建：

```text
source: verify/resolve-approval-<run-id>-test1
target: verify/resolve-approval-<run-id>-test2
```

标题建议：

```text
[verification] resolve all blocks approval <run-id>
```

创建后先确认：

- 页面明确显示合入冲突。
- NextCode `GetMergeRequestMergeability` 为 `mergeable=false`、reason 为 `conflict`。
- 冲突文件恰好两个、marker block 恰好四个。
- MR 作者是执行 `/approval` 的当前用户。

记录 `original_mr_iid`。形状不符时立即停，不评论 `/resolve`。

### 在原 MR 评论 `/resolve`

用 Chrome 在 MR 主线只发一条：

```text
/resolve
```

随后核对：

- NextCode Webhook Delivery 为 `comment/created`，TCE 返回成功。
- `nextcode_webhook_deliveries` 对应记录为 `COMPLETED`，没有同 Event ID 的第二次业务处理。
- RDS 只创建一个活跃 task，`trigger_type=COMMENT`。
- task 从 `PENDING` 向 `COCO_RUNNING` 推进；若失败，保留 `last_error`、Session/Run/Event ID，不在原 task 上盲目重跑。

### 等待 Coco 和外部产物完成

等待 task 到 `AWAITING_APPROVAL`，期间核对：

- 冻结的 source/target HEAD 没有变化。
- resolve commit 的 parents 精确为 `[source_commit, target_commit]`。
- resolve branch HEAD 等于 `resolve_commit_sha`。
- 四个 conflict ID、path、双侧 attribution 和 resolved 锚点均通过服务端回读。
- 测试 exit code 全为 0。
- 飞书上传成功并持久化 URL。
- 新 MR 已创建，source 为 resolve branch，target 为原 test2。
- 新 MR `GetMergeRequestMergeability` 明确为可合入，没有冲突。

### 用 Chrome 验收新 MR 描述

初始描述至少满足：

- 标题和章节结构符合 spec 7.7。
- source、target、resolve branch、原 MR、冲突数、状态正确。
- 飞书链接可点击。
- 四个确认项均为 `[ ]`。
- 四个编号、路径和 resolved 行范围与 RDS 相同。
- 指令区存在，但本轮不执行 approve all/status/cancel/retry。

### 用 Chrome 验收飞书 Markdown

打开共享文件夹：

<https://bytedance.larkoffice.com/drive/folder/K0zefVjqUlsicAdu4TjccNhjnjh>

确认：

- 出现名称带原 MR 和 task ID 的新 `.md` 文件。
- 作者为 Conciliator 应用。
- 从新 MR 描述打开的链接指向同一个文件。
- 页面可见内容包含原 MR、source/target/resolve、四个块、双侧原文和归因、真实 AI 解决片段、决策摘要、测试结果。

本轮不下载，不做 API 内容回读，不检查 SHA 和 L2。

### 用 Chrome 验收四个 Discussion

新 MR 必须有四个 Conciliator 创建的代码行 Discussion：

- 每个 `C0001...C0004` 恰好一条。
- 每条挂在 resolved 文件和 resolved 行范围上。
- source/target 两侧原始行范围与 fixture manifest 对应，并且两侧不相同。
- Ours、Theirs、AI 方案、决策摘要完整。
- 不包含原始 AgentThinking。
- 提示可回复 `/approval`。

任一 Discussion 缺失、重复或落错行，本轮失败，不开始审批。

### 逐块回复 `/approval`

按 `C0001 -> C0004` 顺序逐个操作。每次只在对应 Discussion 下回复：

```text
/approval
```

每次回复后都等待这一块完整闭环，再操作下一块：

| 完成的块数 | RDS approval | Discussion 系统回复 | MR 描述 |
|---:|---|---|---|
| 1 | 1 APPROVED / 3 PENDING | `1/4` | 仅 #1 为 `[x]` |
| 2 | 2 APPROVED / 2 PENDING | `2/4` | #1、#2 为 `[x]` |
| 3 | 3 APPROVED / 1 PENDING | `3/4` | #1 至 #3 为 `[x]` |
| 4 | 4 APPROVED / 0 PENDING | `4/4` | 四项全为 `[x]`，状态为全部确认完成 |

每一步同时检查：

- comment Webhook 被接收且去重。
- reviewer email 为当前用户。
- `reviewed_at` 非空。
- 其他块的状态不变。
- MR description 回读与服务端模板逐字一致。

最后一块完成后，task 必须进入 `ALL_APPROVED`，并在新 MR 主线看到“所有冲突块已 approve，可以合入”的通知。

在 TCE Pod WebShell 中，审批前后分别运行只读校验脚本。脚本读取当前 Pod 的配置、RDS 和 NextCode，不会创建或修改外部资源：

```bash
python scripts/verify_resolve_approval_state.py \
  --repo-id 1106267 \
  --original-mr-number <original_mr_iid> \
  --expect-status AWAITING_APPROVAL

python scripts/verify_resolve_approval_state.py \
  --repo-id 1106267 \
  --original-mr-number <original_mr_iid> \
  --expect-status ALL_APPROVED
```

## RDS 最终验收 SQL

在 RDS AutoSQL 打开：

<https://cloud.bytedance.net/rds/detail/db/global/aeolus_govern/autoSQL?x-global-site=cn&x-global-vregion=China-North&x-global-vdc=cn&x-global-detail-tab=China-North&x-resource-account=public&x-bc-region-id=bytedance&x-bc-vregion=ChinaSinf-North&x-bc-vdc=multicloud>

先按原 MR 找本轮 task：

```sql
SELECT
  id, repo_id, original_mr_id, new_mr_id,
  source_branch, target_branch, resolve_branch,
  mr_version, source_commit_sha, target_commit_sha,
  resolve_base_sha, resolve_commit_sha,
  coco_workspace_id, coco_agent_id, coco_session_id, coco_run_id,
  coco_result_schema_version, lark_doc_url,
  status, retry_count, last_error, active_key,
  created_at, updated_at
FROM resolve_tasks
WHERE repo_id = '1106267'
  AND original_mr_id = <original_mr_iid>
ORDER BY id DESC
LIMIT 1;
```

期望：

- `status='ALL_APPROVED'`
- `retry_count=0`
- `last_error IS NULL`
- `active_key IS NULL`
- `new_mr_id`、所有冻结/resolve commit、Coco ID、`lark_doc_url` 均非空
- `coco_result_schema_version='conciliator.resolve.result.v2'`

查询冲突块：

```sql
SELECT
  conflict_id, block_idx,
  file_path, start_line AS marker_start, end_line AS marker_end,
  source_path, source_start_line, source_end_line,
  source_author_email, source_author_commit_sha,
  target_path, target_start_line, target_end_line,
  target_author_email, target_author_commit_sha,
  resolved_path, resolved_start_line, resolved_end_line,
  attribution_status, discussion_id,
  CHAR_LENGTH(ai_resolution) AS resolution_length,
  CHAR_LENGTH(decision_summary) AS decision_length
FROM conflict_blocks
WHERE task_id = <task_id>
ORDER BY block_idx;
```

期望恰好四行：

- `conflict_id` 为 `C0001...C0004`，`block_idx` 为 `0...3`。
- 两个不同的 `file_path`。
- 每块 source/target 的 path、range、author、commit 均非空。
- 每块 source range 与 target range 不完全相同。
- resolved path/range 和 `discussion_id` 均非空。
- `attribution_status='VERIFIED'`。
- AI 代码片段和决策摘要长度均大于 0。

查询 approval：

```sql
SELECT
  cb.conflict_id,
  a.status,
  a.reviewer_email,
  a.reviewed_at,
  a.comment,
  a.version
FROM approvals a
JOIN conflict_blocks cb ON cb.id = a.block_id
WHERE cb.task_id = <task_id>
ORDER BY cb.block_idx;
```

期望四行全部：

- `status='APPROVED'`
- `reviewer_email` 为当前用户
- `reviewed_at IS NOT NULL`
- `comment IS NULL`

汇总校验：

```sql
SELECT
  COUNT(*) AS total_blocks,
  SUM(a.status = 'APPROVED') AS approved_blocks,
  SUM(a.status = 'PENDING') AS pending_blocks,
  COUNT(DISTINCT cb.discussion_id) AS distinct_discussions,
  COUNT(DISTINCT cb.file_path) AS conflict_files,
  SUM(
    cb.source_start_line = cb.target_start_line
    AND cb.source_end_line = cb.target_end_line
  ) AS equal_side_ranges
FROM conflict_blocks cb
JOIN approvals a ON a.block_id = cb.id
WHERE cb.task_id = <task_id>;
```

期望：`4, 4, 0, 4, 2, 0`。

查询审计事件：

```sql
SELECT event_type, coco_event_id, occurred_at, payload
FROM task_events
WHERE task_id = <task_id>
ORDER BY occurred_at, id;
```

期望能看到“补齐防重放、原子活跃任务和审计事件”一节列出的主干事件，其中 `DISCUSSION_CREATED` 和 `BLOCK_APPROVED` 各四条，且没有 `REJECTED`、`RETRYING`、`CANCELLED` 事件。

## 最终通过标准

本阶段只有同时满足以下条件才算通过：

1. 原 MR 为两文件、四块冲突，source/target 对应块行范围均不相同。
2. `/resolve` 只创建一个 task、一个 resolve branch、一个 resolve MR 和一个飞书 Markdown。
3. resolve commit 为冻结 source/target 的双 parent commit，source/target HEAD 未变化。
4. 新 MR 无合入冲突，描述符合 7.7，飞书链接可打开。
5. 四个冲突块各有且只有一个符合 7.8 的 Discussion，并位于正确 resolved 行。
6. 四次 `/approval` 分别得到 `1/4...4/4`，描述每次只推进一格。
7. 最终 RDS 为一个 `ALL_APPROVED` task、四个 `APPROVED`、四个唯一 Discussion、零 rejected/retry。
8. Webhook 没有重复业务副作用；日志、RDS 和产物中没有任何 secret/token。
9. 本轮没有执行 reject/retry、approve all、status、cancel，也没有自动合并。

## 失败处理和证据保留

- 任一上线或验收门禁失败，立即把 `PIPELINE_ENABLED` 恢复为 `false`，不要在同一个 task 上反复评论。
- 保留 run ID、原 MR、新 MR、task ID、Webhook Event/Delivery ID、Coco Session/Run/Event ID、resolve commit、飞书 file token 和 `last_error`。
- Coco 已经 push 过的 FAILED task 不复用；修复后创建新的 run ID、测试分支、原 MR 和 task。
- 不删除有诊断价值的 MR、分支或飞书文件。验收完成后是否清理，由用户单独确认。
- 不把 WebShell URL、临时 token、cookie 或 secret 写进计划、日志、commit 或 MR 描述。

## 推荐交付拆分

建议用一个新功能分支完成，但按以下 commit 拆分，方便回滚和审查：

1. `feat: add resolve approval fixture and resolved anchors`
2. `feat: wire Markdown reports into resolve pipeline`
3. `fix: route block approvals by resolver MR`
4. `feat: refresh resolver MR approval progress`
5. `feat: persist active task and approval audit contract`
6. `test: add resolve approval end-to-end verifier`

所有代码、DDL和部署门禁通过后，再由用户确认 SCM/TCE 版本并开始“真实端到端执行步骤”中的写入验证。
