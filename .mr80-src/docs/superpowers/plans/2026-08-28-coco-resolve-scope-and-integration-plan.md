# Conciliator × Coco 联调与 `/resolve` 任务范围

> 日期：2026-08-28
>
> 状态：App 身份、双 Pod 基础任务和复杂 MR 的 1—4 步均已在 TCE 通过；真实 resolver commit、四个冲突块、服务端全部门禁和本地文档预览已经独立复核
>
> 前置结论：[NextCode OpenAPI 与 Webhook 契约验证](./2026-08-27-nextcode-openapi-webhook-contract-validation-revised.md)

NextCode 主契约已经在 TCE 验收，本轮不重复跑整套 NextCode fixture。后续 Pod 验证只把 NextCode 当作已知服务端能力，用它创建隔离资源并独立回读 Coco 的写入结果；新增回读调用若与 revised 文档的通过清单不一致，必须单独留证，不能扩张原结论。

## 本轮要回答的问题

本轮先把三件事说清楚，再做仓库级写操作。

1. `/resolve` 完整链路中，哪些工作已经能由 NextCode 和 Conciliator 稳定完成。
2. 哪些工作必须交给 Coco，Coco 的输入、输出和写权限边界是什么。
3. 如何分层验证 Coco 接入和解冲突效果，确保下次开新会话时可以重复执行。

本轮不直接打开完整业务 Pipeline。`PIPELINE_ENABLED` 必须继续保持 `false`，直到本文列出的生产身份验证、结构化结果回读、仓库写隔离和数据库状态迁移全部通过。

## 结论

### Coco 只负责受控执行面

Coco 的职责应收敛为：

- 在 Coco 托管沙箱中挂载指定仓库和指定 revision。
- 挂载从冻结 source commit 创建的 resolver 分支，合入冻结 target commit。
- 在编辑前保存不可变的冲突清单和两侧归因证据。
- 按语义解决冲突，运行测试，创建 merge commit，只 push resolver 分支。
- 返回小而稳定的结构化执行结果，包括 commit、测试、逐块决策摘要和归因证据。

Coco 不负责：

- 查询 MR 是否应该触发任务。
- 创建或更新飞书文档。
- 创建 MR、写 MR Description 或评论。
- 判断谁有审批权限。
- 推进 approve、reject、retry、cancel 等业务状态。
- 修改原 MR 的源分支、目标分支或直接合入目标分支。

这沿用已经人工验证成功的 `/Users/bytedance/Downloads/conflict-resolver` Skill 中关于冻结 commit、隔离写入、冲突审计和测试的要求。Skill 默认使用 target-based 拓扑；Conciliator 的产品契约以原 MR 的 source 分支为 resolver 起点，因此不沿用该拓扑方向。

### 飞书文档采用方案一

采用“Conciliator 生成飞书文档”，不让 Coco 调 `lark-cli`。

原因如下：

- Conciliator 已经掌握原 MR、冲突块、作者、resolver MR 和审批状态，能生成稳定内容。
- 飞书凭证和文档目录不需要进入 Agent 沙箱。
- 文档创建和更新可以按 task ID 幂等，retry 时可以更新原文档。
- Agent 只返回简短决策摘要，不采集或展示原始思维链。
- 后续即使替换 Agent 或模型，文档格式和权限模型也不受影响。

所谓“Agent 思考摘要”应改成 Agent 主动输出的“决策摘要”和“验证摘要”。不要把 `AgentThinking` 原样写入数据库或飞书文档。

### resolver 分支从 source commit 创建

resolver 分支是原 MR source 分支的替代分支。Conciliator 应先冻结原 MR 的 source commit 和 target commit，再从 source commit 创建 resolver 分支，由 Coco 把 target commit 合入 resolver 分支。目标分支的运行时和 API 语义仍是解冲突时的语义基线，但不决定 resolver 分支的第一个 parent。

拓扑如下：

```text
source commit ──创建 resolver branch──> resolver branch
                                         │
target commit ───────merge───────────────┘
                                         │
                                 resolve + test + push
                                         │
                              resolver MR -> target branch
```

最终提交应是保留两个 parent 的 merge commit。第一个 parent 是冻结的 source commit，第二个 parent 是冻结的 target commit。不能 squash 成单 parent 提交，也不能只提交冲突文件快照。

## 对原八项范围的修订

| 能力 | 负责人 | 说明 |
|---|---|---|
| 获取 MR、仓库、源分支、目标分支、MR 作者和冲突块 | Conciliator + NextCode | 已验证；任务开始时还要冻结 MR Version、source commit 和 target commit |
| 创建 resolver 分支 | Conciliator + NextCode | 从冻结的 source commit 创建；创建后回读 HEAD，必须等于该 commit |
| 分析冲突、测试、提交并 push | Coco | 这是 Coco 的核心任务；只允许写 resolver 分支 |
| 获取 source 和 target 两侧最后修改作者 | Coco 采集，Conciliator 校验 | NextCode 的 `GetBlame` 已可调用，但冲突 marker 行号不是原分支行号；Coco 在 Git merge index 中更容易得到两侧原始位置，再由服务按返回的 path、revision 和行范围调用 `GetBlame` 复核 |
| 组织飞书文档内容 | Conciliator | 原始两侧内容来自 NextCode，最终内容通过 resolver commit 回读，决策摘要来自 Coco 结构化结果 |
| 创建或更新飞书文档 | Conciliator + Lark OpenAPI | Coco 不接触飞书凭证 |
| 创建 resolver MR、写 Description 和行级评论 | Conciliator + NextCode | 写后必须回读；评论位置使用新 MR Version 和实际 resolver diff 行号 |
| approve、reject、retry、cancel 和权限校验 | Conciliator | Coco 只接收 retry 或 interrupt 指令，不做权限决策 |

### 为什么作者归因不是纯 NextCode 任务

现有验证只证明 `GetBlame` 在已知 branch、path 和行号时可返回 commit 与 author email。`ListMergeRequestConflicts` 返回的是带 marker 的合并结果，marker 中的行号不是 source 或 target 文件的原始行号。

因此不能直接把 marker 的 `start_line` 传给 `GetBlame`。也不能只按文本搜索原文件，因为重复片段、重命名、删除和空块都会产生歧义。

首选流程是：

1. Coco 在执行 merge 后读取 Git stage 1、2、3 和冲突索引。
2. Coco 为每个冲突块返回 source 与 target 的 path、revision、原始行范围、commit ID 和 author email。
3. Conciliator 对可映射的范围调用 NextCode `GetBlame` 复核。
4. 无法复核的块标记为 `attribution_unverified`，不能只凭 Agent 输出授予审批权限。

## Coco OpenAPI Next 契约

### 长期资源与任务资源

长期复用：

- 一个专用 Workspace。
- 一个专用 Agent。
- 一个固定版本的 conflict resolver Skill。
- 可选的 Environment，用来设置工作目录和无敏感信息的运行时配置。

每个 `/resolve` 任务新建：

- 一个 Session。
- 第一轮 Run。
- reject retry 时优先复用同一个 Session，形成新的 Run。

不要为每个 MR 重建 Workspace 和 Agent。Session 结束后归档；失败排查时可以短期保留 SandboxId 和 WebshellUrl。

### 已确认的 Action

统一 Endpoint：

```text
POST https://codebase-api.byted.org/v2/?Action=<ActionName>
```

官方文档明确区分两类调用身份：

```text
普通用户：X-Code-User-JWT: <codebase-user-jwt>
平台服务账号：X-Code-App-Id: <app-id>
              X-Code-App-Secret: <app-secret>
```

Conciliator 属于平台服务账号，直接使用 App headers 访问由同一 App 身份创建的
Workspace、Agent 和 Session。`SignUserJWT` 只用于平台代某个目标用户调用的场景，
需要 `CloudUserJWT`、Scopes 和过期时间；本阶段不需要代用户执行，因此不引入个人 JWT
或身份代理。Coco 不需要独立的 Coco API Token。

本链路使用以下 Action：

- `CreateManagedAgentWorkspace`
- `ListManagedAgentWorkspaceDevices`
- `CreateManagedAgentSkill`
- `CreateManagedAgent`
- `CreateManagedAgentSession`
- `SendManagedAgentSessionEvents`
- `SubscribeManagedAgentSessionEvents`
- `ListManagedAgentSessionEvents`
- `ListManagedAgentSessionRuns`
- `ListManagedAgentSessionResources`
- `GetManagedAgentSession`
- `GetManagedAgentDevice`
- `CancelManagedAgentSessionRun`
- `ArchiveManagedAgentSession`

官方主路径没有 `/v1/agents/tasks`，也没有需要 Conciliator 暴露的 `/hooks/coco`。执行进度由 SSE 和事件回读获得。

### 仓库挂载

创建 Session 时传：

```json
{
  "WorkspaceId": "<workspace-id>",
  "AgentId": "<agent-id>",
  "Resources": [
    {
      "Repository": {
        "RepoId": "1106267",
        "Revision": "<resolver-branch-or-frozen-commit>"
      }
    }
  ]
}
```

不传 clone URL，不在 prompt 中要求 Agent 自己 clone，也不在 TCE 容器里执行 git clone。

首轮仓库写验证使用 resolver branch 作为 `Revision`，随后在沙箱内校验该分支 HEAD 等于预期 source commit。若平台允许直接按 commit 挂载且仍能向预创建 resolver branch push，再切换为冻结 commit；在验证前不猜测其写分支行为。

### Session 事件消费

发送任务后保存 `Result.Events[0]` 中的：

- Event ID。
- Session ID。
- Run ID。

使用 `SubscribeManagedAgentSessionEvents` 订阅过程事件。断线时用 `AfterEventId` 从最后已持久化的事件继续，不要把 Run ID 当成事件游标。

需要处理：

- `AgentThinking`：默认不持久化原文，只记录事件存在、长度和时间。
- `AgentMessage`：按 `Delta=true` 拼接；`Delta=false` 视为该消息的完整快照。
- `AgentToolUse`：保存工具名、结果状态和脱敏参数摘要，用于审计。
- `SessionStatusIdle`：表示本轮结束，或正在等待危险操作确认。
- `SessionStatusTerminated`：表示失败终止。
- `ping`：只用于保活。

收到 Idle 后必须检查 `StopReason.RequiresAction`。非空时不能当作成功；本项目固定使用
`bypass_permissions` Agent，因此该事件表示生产 Agent 配置错误，任务立即进入 `FAILED`。

### PermissionMode

仓库写任务固定使用专用 `bypass_permissions` Agent。2026-08-28 已从 Agent 详情和 Session 元数据回读 `permission_mode=bypass_permissions`，并完成一次真实文件修改、commit 和普通 push。执行过程中没有 `UserToolConfirmation`，文件编辑事件为 `auto_approved=true`，source 和 target 分支均未变化。

生产链路不实现 `UserToolConfirmation`，也不引入 `COCO_ACTION_REQUIRED`。如果 Session 返回 `StopReason.RequiresAction`，说明专用 Agent 的权限配置不符合预期，任务应立即进入 `FAILED`，`last_error` 写为 `Coco Agent PermissionMode 配置错误`。

当前供生产服务使用、由 Conciliator App 身份创建的长期资源：

- Workspace：`Conciliator Service`
- Workspace ID：`793550816067727`
- 名称：`conciliator-resolver`
- Agent ID：`793550816045691`
- Runtime：`traex`
- Device：Cloud system device
- PermissionMode：`bypass_permissions`

旧 Workspace `792639127465978` 和 Agent `793541586477315` 由用户身份创建。App headers
访问旧 Agent 时返回 403，这是资源所有权隔离，不是 App headers 不受支持；生产配置
不能继续引用这组旧 ID。

## Coco resolver 任务契约

### 输入由 Conciliator 固化

每次任务输入至少包含：

```json
{
  "schema_version": "conciliator.resolve.input.v1",
  "task_id": 123,
  "repo_id": "1106267",
  "original_mr_number": 10,
  "mr_version": 1,
  "source_branch": "feat/example",
  "source_commit": "<40-char-sha>",
  "target_branch": "main",
  "target_commit": "<40-char-sha>",
  "attempt": 1,
  "resolve_branch": "fix/feat-example-to-main-conflict-20260828-t123",
  "expected_resolve_head": "<source-commit>",
  "conflicts": [
    {
      "conflict_id": "<stable-task-local-id>",
      "path": "src/example.py",
      "source_text": "...",
      "target_text": "...",
      "context_before": "...",
      "context_after": "..."
    }
  ],
  "locked_conflict_ids": [],
  "retry_reasons": []
}
```

外部契约统一使用 `source_text` 和 `target_text`，不要使用容易随 merge 方向变化的 `ours` 和 `theirs`。Agent 内部执行 source ← target merge 时，Git 的 ours 是 source，theirs 是 target。

输入中的 commit 和 conflict manifest 一经创建不能随分支漂移。若 MR 在任务期间发生 push，旧任务标记为 stale，重新创建任务或明确执行 rebase 流程，不能静默改用新 HEAD。

### Agent 必须执行的步骤

1. 检查挂载仓库、当前 branch、HEAD 和 remote，发现与输入不一致立即停止。
2. 确认原 source branch、target branch 和原 MR 没有写操作。
3. 确认 resolver HEAD 等于冻结 source commit，再合入冻结 target commit。
4. 编辑前保存 Git conflict index、冲突类型和每侧原始内容。
5. 为每个冲突块分析 source 意图、target 意图和最终组合方案。
6. 记录两侧 path、原始行范围、最后修改 commit 和 author email。
7. 解决内容冲突和 rename、delete、mode 等结构冲突。
8. 运行目标分支基线可运行的 focused tests，再运行必要的 broader tests。
9. 检查没有冲突 marker、没有未合并 index、没有越界文件改动。
10. 创建保留两个 parent 的 merge commit。
11. 只 push resolver branch。
12. 输出结构化结果，不创建 MR、评论或飞书文档。

### 结构化输出

最终 `AgentMessage` 只输出一份带明显边界的 JSON，不输出大段源代码：

```text
CONCILIATOR_RESULT_BEGIN
{...}
CONCILIATOR_RESULT_END
```

JSON 至少包含：

```json
{
  "schema_version": "conciliator.resolve.result.v1",
  "task_id": 123,
  "status": "succeeded",
  "attempt": 1,
  "resolve_branch": "fix/feat-example-to-main-conflict-20260828-t123",
  "commit_sha": "<40-char-sha>",
  "merge_parents": ["<source-sha>", "<target-sha>"],
  "conflicts": [
    {
      "conflict_id": "<id>",
      "path": "src/example.py",
      "decision_summary": "保留 target 的新校验，并接入 source 的新数据字段",
      "introduced_fix": "补充两侧组合后缺失的空值处理",
      "source_attribution": {
        "path": "src/example.py",
        "start_line": 10,
        "end_line": 15,
        "commit_sha": "<sha>",
        "author_email": "<email>"
      },
      "target_attribution": {
        "path": "src/example.py",
        "start_line": 8,
        "end_line": 13,
        "commit_sha": "<sha>",
        "author_email": "<email>"
      }
    }
  ],
  "tests": [
    {
      "command": "pytest tests/example_test.py -q",
      "exit_code": 0,
      "summary": "8 passed"
    }
  ],
  "unresolved_items": []
}
```

原始两侧代码和最终代码不依赖 Agent 消息传输。Conciliator 已有原始冲突内容，并应通过 NextCode 回读 resolver commit 的文件内容和 diff。

### 服务端验收不能只信 Agent 文本

收到结构化结果后，Conciliator 至少复核：

- resolver branch HEAD 等于 `commit_sha`。
- commit 有且只有预期的两个 merge parent。
- 第一个 parent 是冻结的 source commit，第二个 parent 是冻结的 target commit。
- source branch、target branch HEAD 没被任务修改。
- 所有原冲突文件不再含冲突 marker。
- Git diff 不包含输入范围外的异常文件。
- 对可映射的 source 与 target 行范围执行 `GetBlame`，author 与 commit 能对上。
- 测试摘要存在；关键测试失败时不能进入文档和 MR 创建阶段。

任何一项失败都进入 `COCO_RESULT_INVALID`，保留 Session、Run、最后 Event ID 和 readback 证据，不创建 resolver MR。

## approve、reject 和 retry

### 权限来源

当前 App OpenAPI 不能获取仓库 admin、owner 或 master。第一版权限不能依赖这些角色。

建议先采用可验证的最小规则：

- 行级 `approve` 和 `reject <reason>`：评论者邮箱必须命中该块经复核的 source author、target author 或原 MR 作者。
- `/resolve approve all`：仅允许原 MR 作者，或仓库配置中的显式维护者 allowlist。
- `/resolve cancel` 和 `/resolve retry`：仅允许原 MR 作者或显式维护者 allowlist。
- author 归因未复核时：只允许原 MR 作者处理，不能信任 Agent 单方面返回的邮箱。

如果后续必须支持仓库管理员，需新增独立、已验证的成员权限数据源；不能用仓库 creator、MR reviewer 或应用安装人冒充 admin。

### retry 使用同一 Session

reject 后优先复用原 Session，发送新的 UserMessage，内容只包含：

- 被拒绝的 conflict ID。
- 拒绝原因。
- 当前 resolver branch HEAD。
- 已批准、禁止改动的 conflict ID。
- 需要重新运行的测试。

每次 retry 产生新 Run 和新 commit，不改写旧提交历史。Conciliator 更新原飞书文档和原 resolver MR，不重复创建新 MR。

若原 Session 已归档或沙箱不可恢复，再创建新 Session，挂载当前 resolver branch，并完整传入已批准块、被拒绝块和历史决策摘要。

当前 `fetch_task_result` 只支持单 Run Session。复用 Session 之前，必须先持久化当前
`run_id`，并让事件状态和消息聚合只读取该 Run；否则旧 Run 的 Idle 或 AgentMessage
可能污染 retry 结果。

### cancel 必须取消外部执行

`/resolve cancel` 不能只改数据库状态。若 Run 仍在执行，应调用 `CancelManagedAgentSessionRun` 或发送 `UserInterrupt`，回读 Run 进入 cancelled 或 Session 回到 Idle 后，再归档 Session。

## 状态与数据库调整

`coco_task_id` 暂时作为兼容字段保留，值等于 Session ID。代码和 `0003_resolve_contract` 迁移已经新增：

- `coco_workspace_id`
- `coco_agent_id`
- `coco_session_id`
- `coco_run_id`
- `coco_last_event_id`
- `coco_result_schema_version`
- `coco_result_json`
- `source_commit_sha`
- `target_commit_sha`
- `resolve_base_sha`
- `resolve_commit_sha`

冲突块的 `author_email` 也作为兼容字段保留。代码和迁移已经拆出：

- `source_author_email`
- `source_author_commit_sha`
- `source_path`
- `source_start_line` 和 `source_end_line`
- `target_author_email`
- `target_author_commit_sha`
- `target_path`
- `target_start_line` 和 `target_end_line`
- `attribution_status`
- `decision_summary`
- `introduced_fix`

本阶段沿用现有状态，不增加 `COCO_ACTION_REQUIRED`。`RequiresAction` 直接进入 `FAILED`。更细的 Session、结果校验和取消状态等完整链路稳定后再评估，避免为了验证链路提前扩张状态机。

对应增量 DDL 已生成到 `ddl/0003_resolve_contract.sql`，静态 schema 检查通过。2026-08-28
数据库工单已经执行完成；TCE 实测回读到 `resolve_tasks` 和 `conflict_blocks` 的新增字段、
索引，并成功写入真实任务和 4 个冲突块。

## 已完成的实现修正

本轮已经在本地工作区完成：

- Coco 客户端改为 Codebase v2 `Action` 调用。
- 平台服务账号鉴权复用 `NEXTCODE_APP_ID` 和 NextCode App Secret KMS alias，并使用官方规定的 App headers。
- 删除独立 `KMS_SECRET_COCO_API_TOKEN_ALIAS` 配置。
- 删除假想的 `/hooks/coco` receiver 和 Bearer token 校验。
- 增加 Workspace、Device、Agent、Session、Message、SSE、Event、Run、Resource、Interrupt 和清理 Action 的客户端封装。
- Session 仓库输入改为 `Resources.Repository{RepoId, Revision}`。
- SSE 支持 `AfterEventId`，并区分正常 Idle 与 `RequiresAction`。
- resolver 分支按产品契约从 source branch 创建，Coco 合入 target branch。
- resolver 分支名已实现为 `fix/<source-slug>-to-<target-slug>-conflict-<YYYYMMDD>-t<task_id>`；日期取任务创建日期，重试复用原分支，不移动已有分支。名称过长时截断 source/target 部分，完整保留日期和 task ID。
- 任务开始时回读原 MR，冻结 MR Version、source commit 和 target commit；`CreateBranch` 直接使用冻结 source commit，并用 `GetBranch` 做幂等回读。
- 冲突文件按路径排序，冲突块生成 `C0001` 形式的稳定 ID；重复保存时校验冻结 manifest 一致。
- Coco 输入改为 `conciliator.resolve.input.v1` JSON；最终消息只接受边界内的 `conciliator.resolve.result.v1` JSON，不持久化 `AgentThinking`。
- Coco 完成后由服务端回读 resolver/source/target HEAD、merge commit parents、冲突文件 marker 和两侧 `GetBlame`；任一不一致都停止写文档和 MR。
- 飞书文档、MR Description 和 Discussion 模板改用 source/target author、决策摘要、额外修复和测试结果。
- 新建 resolver MR 前按稳定 source/target/title 恢复已有 MR，创建后再回读 source、target 和 Description。
- 增加 `0003_resolve_contract` 数据库迁移和对应增量 DDL。
- 增加可复用脚本 `scripts/verify_coco_basic_task.py`。
- 增加安全门禁：结构化结果或 NextCode 回读未通过时，不允许继续创建飞书文档和 resolver MR。

本轮最终回归结果：

- `.venv/bin/ruff check src scripts tests`：通过。
- `uv run pytest -q`：138 passed，8 skipped。
- `uv run mypy src`：51 个源文件通过。
- `git diff --check`：通过。

仍未完成：

- `CancelManagedAgentSessionRun` 的业务接入。
- Run 级事件过滤；当前客户端只能安全处理单 Run Session。
- resolver commit 的完整 diff 范围和 rename/delete 等结构冲突复核。
- 飞书文档写后回读，以及完整 approve、reject、retry 状态机联调。

## 已完成的真实验证

### 用户态基础任务通过

2026-08-28 使用 `bytedcli coco` 创建了临时 Workspace、Agent 和 Session，发送：

```text
Reply with exactly COCO_BASIC_SMOKE_OK and nothing else.
```

SSE 实际收到：

1. `SessionStatusRunning`
2. `AgentMessage`，`Delta=true`
3. `AgentMessage`，`Delta=false`
4. `SessionStatusIdle`，`StopReason.EndTurn`

最终消息精确为 `COCO_BASIC_SMOKE_OK`。Session 已归档，临时 Agent 和 Workspace 已删除。

该结果证明：

- Coco Managed Agent 控制面可访问。
- Workspace → Device → Agent → Session 资源关系符合文档。
- Message 能创建 Run。
- SSE 的 Event、Run、Delta 和 Idle 语义符合文档。
- 清理 Action 可用。

该结果在当时只证明用户态链路；后续 App headers 双 Pod smoke 和真实复杂任务已经补齐
生产身份证据。

### App headers 双 Pod smoke 通过

2026-08-28 在两个 TCE Pod 内，使用 KMS 读取的 NextCode App Secret 和官方 App headers，
复用 App 身份创建的 Workspace `793550816067727`、Agent `793550816045691` 运行基础任务。

完成方式：

```bash
cd /opt/tiger/conciliator
PYTHONPATH=/opt/tiger/conciliator/src \
  /opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_coco_basic_task.py \
  --workspace-id 793550816067727 \
  --agent-id 793550816045691 \
  --timeout 240
```

预期输出：

```text
coco_connectivity=ok
agent_message_seen=true
session_idle_seen=true
expected_reply_seen=true
run_list_nonempty=true
derived_status=succeeded
coco_basic_task=ok
```

两侧结果均为 `coco_basic_task=ok`：

- 小流量 Pod：Session `793550864317475`，Run `793550864313963`。
- 单机房 Pod：Session `793551057210735`，Run `793551057235515`。

这证明 App headers、KMS、Workspace/Agent 所有权、Session、Event、Run 和基础 Agent 回复
在 TCE 都能联通。`--provision` 路径另发现运行时 Workspace ID 未传给最终回读的问题，
客户端现已支持给 `fetch_session_result` 显式传入 Workspace ID，并补了回归测试。

### 专用 Agent 的 Git 写入通过

2026-08-28 先在用户态 Workspace `conciliator` 中创建了专用 Agent
`conciliator-resolver`，Agent ID 为 `793541586477315`。真实 Session
`793541838133278` 挂载仓库后切换到隔离 resolver 分支，新增
`coco-write-smoke-20260828.txt`，创建 commit
`adc4dbe2c8c69f3cfd45a60e3467bc9c07d63578` 并普通 push 成功。

验证前后分支状态：

- resolver：`2719174a97e348ce05a38cdcb0863c2979da3410` → `adc4dbe2c8c69f3cfd45a60e3467bc9c07d63578`
- source：保持 `2719174a97e348ce05a38cdcb0863c2979da3410`
- target：保持 `6218f10ccc6c147bd4abf1712615ee01b42f80ff`

这个结果证明 `bypass_permissions` Agent 可以无人值守修改文件、commit 并普通 push，也证明
现有 instruction 能约束它不改 source 和 target。由于用户态资源不能由 App 直接复用，
之后已用 App headers 创建同名长期 Agent `793550816045691`，并在 TCE 双 Pod 完成基础任务；
真正的复杂冲突任务也由这个 App-owned Agent 执行。

## 分阶段验证计划

### 客户端契约和无仓库 smoke

状态：完成。

- [x] 单元测试验证 Action URL、App headers、Session resource、Message event、Delta 拼接、Idle 和 RequiresAction。
- [x] 用户态真实基础任务通过。
- [x] 构建并部署基础验证版本 SCM `1.0.0.10`，随后部署严格结果契约修正版 `1.0.0.11`。
- [x] 在两个 TCE Pod 分别运行 App headers 基础 smoke，均得到 `coco_basic_task=ok`。
- [x] 确认两个 Pod 都能通过 KMS 使用 App Secret，脚本和日志不输出 secret。

退出条件已经满足。旧用户态 Agent 的 403 只说明资源所有权不同；App-owned Agent 已成功。

### 创建长期 Workspace、Skill 和 Agent

- [x] 创建并验证 App-owned 长期 Workspace `Conciliator Service`，ID `793550816067727`。
- [x] 使用 Cloud system device 和 `traex` Runtime 创建专用 Agent。
- [x] 将 resolver 执行边界写入 Agent Instruction。
- [x] 创建 App-owned 长期 Agent `conciliator-resolver`，Agent ID 为 `793550816045691`。
- [x] 回读 Agent 和 Session 元数据，确认 `PermissionMode=bypass_permissions`。
- [x] 把 `COCO_DEFAULT_AGENT_ID=793550816045691` 配到 TCE，替换旧用户态 ID。
- [x] 回读 App-owned 数字 Workspace ID：`793550816067727`。
- [x] 配置 `COCO_WORKSPACE_ID=793550816067727`。没有填写 Workspace slug。

退出条件：同一 Agent 连续创建两个 Session，都能证明 Skill 被加载并按要求输出版本标记。

### 只读仓库挂载

- [ ] 创建 Session，挂载 `RepoId=1106267` 和一个冻结 commit。
- [ ] Agent 只执行只读命令，输出 repo path、HEAD、branch 和指定文件摘要。
- [ ] `ListManagedAgentSessionResources` 回读一个 repository，target ID 为 `1106267`。
- [ ] 通过 NextCode 独立回读 commit 和文件摘要，结果一致。
- [ ] 确认没有 branch、commit、MR 和 comment 写入。

退出条件：挂载路径、revision 和文件内容与 NextCode 回读一致。

### 隔离写操作

在 `xiahaitao/Conciliator` 中创建新的 source、target 和 resolver 测试分支。不要复用 main 或业务分支。

- [x] source 与 target 从同一 base commit 创建。
- [x] 两侧构造确定性冲突，保存冻结 source SHA 和 target SHA。
- [x] resolver branch 从 source SHA 创建，并回读确认 HEAD 等于 source SHA。
- [x] 创建测试 MR：source → target，并确认 mergeability 为 conflict。
- [x] Coco Session 挂载 resolver branch，合入冻结 target SHA。
- [x] Coco 解决冲突、运行指定测试、创建 merge commit 并只 push resolver branch。
- [x] NextCode 回读 resolver HEAD、两个 parent、diff 和 marker。
- [x] 确认 source、target 和原 MR 未被 Agent 修改。
- [x] 测试分支、resolver 分支和 MR !11 保留为可重复证据。

退出条件：所有硬门禁通过，且不存在越界写操作。

### 效果验证

状态：通过。四个语义冲突块全部符合 `EXPECTED.md`，五个 source/target focused tests
全部通过；决策摘要和归因已写入文档预览。

本阶段保留现有单块 MR !10 作为链路 smoke，再新增一个复杂 MR 作为效果和文档质量验收。复杂 MR 使用独立的 base、source 和 target 测试分支，目录固定为 `validation/coco_complex_conflict/`，包含不冲突的 `EXPECTED.md` 和两侧独立测试。

复杂 MR 预设四个冲突块：

| 冲突块 | 文件 | 预期处理 |
|---|---|---|
| `C0001` | `formatter.py` 的报告格式 | 保留 target 的固定前缀和 source 的 author email |
| `C0002` | `policy.py` 的路由规则 | 综合两侧逻辑，security review 的优先级高于 owner approval |
| `C0003` | `policy.py` 的超时规则 | 同时保留 source 的 urgent 和 target 的 bulk 行为 |
| `C0004` | `policy.py` 的标签 | 同时保留 source label 和 target label |

source 和 target 分别新增互不冲突的 `unittest` 文件。单侧分支各自测试通过，resolver 合并后必须同时通过：

```bash
python3 -m unittest discover -s validation/coco_complex_conflict/tests -v
```

fixture 记录：

- 人工期望的行为，不只记录期望文本。
- source 和 target 的意图。
- 必须通过的 focused tests。
- 允许变化的文件集合。
- 两侧 author email 和 commit 证据。

硬门禁：

- 无未解决冲突和 marker。
- merge parent 正确。
- 只写 resolver branch。
- 必须测试通过。
- 结构化结果完整且能被服务端回读验证。

质量评分：

- 是否同时保留 source 与 target 的有效意图。
- 是否识别组合后才出现的新问题。
- 是否只做必要修改。
- 决策摘要能否解释每个冲突块。
- attribution 是否能复核。

复杂 MR 先检查 Git 实际产生两个冲突文件和四个 marker block，再交给 Coco。rename、delete、二进制和 submodule 冲突留到后续边界验证，不混入本轮验收。

### 服务端状态和对外写操作

- [x] 增加数据库迁移和 Session/Run/Event cursor 持久化字段。
- [x] 在 TCE 执行并核对数据库迁移；真实任务 2 成功写入任务和四个冲突块。
- [ ] 实现 SSE 断线重连和进程重启恢复。
- [x] 实现结构化结果 parser、schema 校验和 NextCode HEAD/parent/marker/blame readback。
- [x] 已有服务端飞书文档 XML 模板。
- [x] 已实现由 Conciliator 创建 resolver MR、Description 和逐块行级评论的代码路径；本阶段未做真实写操作联调。
- [x] 同一份报告数据已渲染为 Markdown，写入 TCE Pod 的 `/tmp/conciliator-validation/task-2/`，并完成逐块人工核对。
- [ ] 下一阶段再验证飞书文档写后回读、完整 diff 和真实 resolver MR。
- [ ] 验证 `approve`、`reject <reason>`、兼容别名 `/approval` 和 `/reject <reason>`，以及 `/resolve retry`、`/resolve cancel`、`/resolve status`。
- [ ] 验证重复 Webhook、重复 Event 和重复命令不会产生重复副作用。

退出条件：完整链路可以运行，但仍不自动 merge resolver MR。

## 可重复验证入口

### 本地契约测试

```bash
.venv/bin/ruff check src scripts tests
.venv/bin/pytest tests/unit/test_coco_client.py tests/unit/test_config.py -q
```

### TCE App headers 基础 smoke

```bash
cd /opt/tiger/conciliator
PYTHONPATH=/opt/tiger/conciliator/src \
  /opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_coco_basic_task.py \
  --workspace-id 793550816067727 \
  --agent-id 793550816045691 \
  --timeout 240
```

默认复用 App-owned 长期 Workspace 和 Agent，只新建 Session。需要验证资源创建权限时才加
`--provision`；临时 Workspace 和 Agent 在验证结束后会自动清理，排查时才加
`--keep-resources`。

### 用户态独立交叉验证

当需要区分“Coco 平台故障”和“Conciliator App 权限故障”时，可用 `bytedcli coco workspace/device/agent/session` 重跑无仓库任务。

- 用户态成功、App headers 失败：先检查 Workspace/Agent 是否由同一 App 身份创建，再检查 Managed Agents scopes、App 安装范围和 KMS App Secret。只有需要代用户调用时才检查 `SignUserJWT`。
- 两者都失败：检查 Coco 控制面、Runtime 或 Session 事件。
- Session 创建成功、仓库挂载失败：检查 App 对目标 repo 的安装和下载权限。
- 仓库可读、push 失败：检查 resolver branch 权限、Agent git credential 和保护分支规则。

## 当前阶段在 TCE 验证 1—4 步

验证期间保持：

```text
INTEGRATIONS_ENABLED=false
NEXTCODE_ENABLED=true
PIPELINE_ENABLED=false
COCO_DEFAULT_AGENT_ID=793550816045691
COCO_WORKSPACE_ID=793550816067727
```

先完成代码和环境门禁：

1. 实现正式 resolver 分支名 `fix/<source-slug>-to-<target-slug>-conflict-<YYYYMMDD>-t<task_id>`。日期取任务创建日期，source/target 中的 `/` 等字符转成 `-`，超长时保留日期和 task ID；分支仍从冻结 source commit 创建。
2. 将 `RequiresAction` 改为立即 `FAILED`，不实现人工工具确认。
3. 增加复杂 fixture 创建脚本，创建 base/source/target 分支、四个冲突块和 source → target MR，回读 mergeability 与冲突块数量。
4. 增加 `verify_coco_resolve_flow.py`。脚本复用正式 Service、Repository、NextCodeClient 和 CocoClient，允许新建任务或按 task ID 续跑，执行到本地报告生成后停止。
5. 把报告渲染与飞书写入解耦。TCE 验证写入 `conflicts.json`、`coco-result.json`、`lark-document-preview.md` 和 `validation-summary.json`，文件权限为 `0600`，内容不得包含 secret 或 token。
6. 确认生产数据库已经执行 `0003_resolve_contract`，并把数字 `COCO_WORKSPACE_ID` 配到 TCE。
7. 提交并推送精确 commit，构建 SCM，更新 TCE 后再运行真实验证。

2026-08-28 的门禁回读结果：

- App headers 创建的 Workspace ID 为 `793550816067727`。
- Agent `793550816045691` 的 `WorkspaceId`、Runtime 和
  `PermissionMode=bypass_permissions` 回读一致。
- 生产库已经执行 `ddl/0003_resolve_contract.sql`；实测可写入新增字段和冲突块唯一索引。
- SCM `1.0.0.10` 的双 Pod 基础 smoke 通过；严格结果契约修正版 `1.0.0.11`
  已部署，两个 App-owned 资源 ID 已从 TCE 集群配置回读。

实际执行顺序：

1. 两个 Pod 分别运行无仓库 Coco smoke，确认 TCE 应用身份能创建 Session、发送消息、读取 Event 和 Run，且没有 `RequiresAction`。
2. 只在单机房 Pod 创建复杂 fixture，记录原 MR、source commit、target commit、冲突文件和冲突块。
3. Conciliator 冻结 MR Version、source commit 和 target commit，从冻结 source commit 创建 resolver 分支。创建后回读 HEAD；重试沿用任务中已经保存的分支名。
4. Conciliator 获取冲突文件，按路径和块顺序生成 `C0001` 至 `C0004`，把 marker 视图行号、source/target 内容写入数据库。这里的行号明确表示 NextCode 冲突视图中的 marker 行号，不冒充两侧文件的物理行号。
5. Conciliator 创建 Coco Session，挂载 resolver 分支。Coco 合入冻结 target commit，按 `EXPECTED.md` 解决冲突，运行两侧测试，创建双 parent merge commit，只 push resolver 分支，并返回逐块决策摘要和归因。
6. 服务端回读 resolver/source/target HEAD、merge parents、最终文件、残留 marker 和两侧 blame。任一检查失败都停止，不生成报告。
7. 服务端生成本地 Markdown 报告和 JSON 证据。人工逐块对照 `EXPECTED.md`、最终代码、测试结果、author email 与 commit SHA。

本阶段的通过标准：

- resolver 分支名和基点符合规则，source、target 没有变化。
- 数据库保存两个冲突文件和四个稳定编号的冲突块；重复保存不会新增记录。
- resolver commit 的 parent 顺序为 `[source, target]`，最终文件没有 marker，两侧测试全部通过。
- Coco 结果中的 conflict ID、path、归因和测试均通过 NextCode 回读校验。
- 文档预览完整包含 MR、分支、冻结 commit、每个冲突块的两侧代码和归因、解决决策、最终文件内容以及测试结果。
- 脚本退出码为 `0`，`validation-summary.json` 中所有检查为 `PASS`。

这些条件全部满足后，本轮结束。飞书 OpenAPI、MR Description、resolver MR、Discussion、approve、reject、retry 和 cancel 放到下一阶段。

## 2026-08-28 TCE 真实执行记录

### 复杂 fixture 与原 MR

本地脚本使用 Git 凭证创建并推送三个隔离分支；创建前会检查远端同名分支不存在，
创建后先在临时 clone 中执行真实 merge，只有得到两个冲突文件、四个 marker block 才会
继续 push：

```bash
uv run python scripts/create_coco_complex_fixture.py \
  --git-provision \
  --run-id 20260828-stage14 \
  --output /tmp/coco-complex-fixture.json
```

固定证据：

- base：`verify/coco-complex-20260828-stage14-base`，commit
  `52058c19362df2cdf7d0a1582475fb712868c340`。
- source：`verify/coco-complex-20260828-stage14-source`，commit
  `b3e925690ab453e627bff0b0d3e41f7418f1f51e`。
- target：`verify/coco-complex-20260828-stage14-target`，commit
  `956fcd9c47c0e10f6e2049c1e2f11b002da7be6f`。
- 原 MR：[!11](https://code.byted.org/xiahaitao/Conciliator/merge_requests/11)，页面和
  `GetMergeRequestMergeability` 均确认 conflict。
- 实际冲突顺序：`formatter.py` 一个块，`policy.py` 三个块，即 `C0001` 至 `C0004`。

source、target 分别带独立测试，合并后的验收命令固定为：

```bash
python3 -m unittest discover -s validation/coco_complex_conflict/tests -v
```

### 第一次 1—4 步执行：Git 成功，结果契约被正确拦截

在单机房 Pod 使用正式 `ResolveService`、数据库 Repository、NextCodeClient 和 CocoClient
运行：

```bash
cd /opt/tiger/conciliator
PYTHONPATH=/opt/tiger/conciliator/src \
  /opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_coco_resolve_flow.py \
  --repo-id 1106267 \
  --mr-number 11 \
  --output-dir /tmp/conciliator-validation \
  --timeout 1200 \
  --poll-interval 10
```

运行产生任务 `task_id=1`，冻结 source/target commit 与上文一致，并从冻结 source 创建：

```text
fix/verify-coco-complex-20260828-stage14-source-to-verify-coco-complex-20260828-stage14-target-conflict-20260828-t1
```

数据库保存了两个文件、四个稳定编号冲突块。Coco Session `793551151598796`、Run
`793551151605608` 完成后 push commit
`75fd783afbdd5833840ad6e9f616ff0b3af9cc29`。独立回读确认：

- resolver commit parents 精确为 `[source, target]`。
- source、target HEAD 均未变化。
- 最终文件没有 marker。
- 五个 focused tests 全部通过。
- 最终逻辑同时保留 security review、owner approval、urgent/bulk timeout、两侧 labels，
  报告格式同时保留 `[conflict]` 前缀与 author email，语义效果符合预期。

但 Coco 最终 JSON 漏了顶层 `status`、`attempt`、`resolve_branch`。服务端没有放宽 parser，
而是把任务标为 `FAILED`，错误为：

```text
result keys mismatch: missing=['attempt', 'resolve_branch', 'status'], extra=[]
```

因此第一次运行只证明步骤 1、2 和步骤 3 的 Git/代码效果；步骤 3 的结构化结果验收未过，
步骤 4 按安全门禁没有生成报告。这是符合预期的 fail-closed 行为，不能记为 1—4 全通过。

后续修正保留严格 parser，在发送给 Agent 的 prompt 中加入完整的
`conciliator.resolve.result.v1` 模板，明确十个顶层键必须完全一致，并填入确定的
task ID、attempt、resolve branch 和 merge parents。同时修正基础 smoke 的运行时
Workspace ID 回读。修正包含在 commit
`8ec3da5dab4ec424fa2b33603fe96f740bc80d18` 和 SCM `1.0.0.11` 中。部署后创建新任务
和新 resolver 分支复跑，没有复用已经 FAILED 且 resolver 已前进的 task 1。

### 第二次复跑：1—4 步全部通过

2026-08-28 在单机房 Pod `fed-dp-6b9e96188f-7dcddc4449-gxrfv` 复跑 MR !11，脚本
退出码为 0，数据库任务 `task_id=2` 进入 `DOC_CREATED`：

- resolver 分支：
  `fix/verify-coco-complex-20260828-stage14-source-to-verify-coco-complex-20260828-stage14-target-conflict-20260828-t2`。
- Coco Session：`793553926132743`；Run：`793553926128943`。
- resolver commit：`d2444a4ad58310db49d6b5ca1a19fad159837501`。
- parents：`b3e925690ab453e627bff0b0d3e41f7418f1f51e`、
  `956fcd9c47c0e10f6e2049c1e2f11b002da7be6f`，顺序精确为 `[source, target]`。
- source、target 远端 HEAD 与冻结值一致，没有被 Agent 修改。
- `coco-result.json` 恰好包含十个顶层键，`status=succeeded`、`attempt=1`、
  `unresolved_items=[]`。
- 测试结果为 exit 0，五个测试全部通过。

数据库和报告中的稳定冲突块为：

| ID | 文件 | NextCode marker 行 | 解决结果 |
|---|---|---:|---|
| `C0001` | `formatter.py` | 8–13 | 保留 target 的 `[conflict]` 固定格式，并追加 source author email |
| `C0002` | `policy.py` | 14–20 | security review 优先，随后保留 owner approval 自动路由 |
| `C0003` | `policy.py` | 35–41 | 同时保留 urgent=10、bulk=60，默认 30 |
| `C0004` | `policy.py` | 55–59 | 合并 `conciliator`、`source-change`、`target-runtime` labels |

`validation-summary.json` 的八项门禁全部为 `PASS`：

```text
conflict_ids_and_paths
conflict_markers_removed
merge_parents
resolve_branch_head
source_branch_unchanged
source_target_blame
target_branch_unchanged
tests
```

随后从本地直接读取三个远端 branch，再次确认 resolver/source/target HEAD 和两个 parent，
并在 resolver commit 的独立临时 worktree 重跑五个测试，结果仍为 `OK`；`rg` 没有发现
残留 conflict marker。这是独立于服务端 `validation-summary.json` 的交叉证据。

### 本地文档预览与证据目录

任务 2 在 Pod 中生成：

```text
/tmp/conciliator-validation/task-2/conflicts.json
/tmp/conciliator-validation/task-2/coco-result.json
/tmp/conciliator-validation/task-2/lark-document-preview.md
/tmp/conciliator-validation/task-2/validation-summary.json
```

四个文件权限均为 `0600`。为通过 TCE 文件下载白名单做后续人工取证，另复制到
`/opt/tiger/toutiao/log/conciliator-validation/task-2/`；原始 `/tmp` 文件保留不动。

人工检查 `lark-document-preview.md` 已确认：MR/分支/冻结 commit/resolve commit 完整；
四个块均包含 marker 行、source/target 原文、author email、author commit、原分支行范围、
解决决策和额外修复；最终两个文件全文及测试命令、exit 0 和五项通过摘要完整。

产物 SHA-256：

- `coco-result.json`：`8962cf3f3120fe172f0a14f91b5ae3d103b8a537e3767c104a1b867fe9d26cf2`
- `conflicts.json`：`c9eb3d2da26152bfe1daa91f82c8e8c1c2f3f9b875767afd4d3e68771e91f5fe`
- `lark-document-preview.md`：`e6a02f7e8dd0aca12a4898fd6fd838b686212e278c98466db760f45d7558a878`

### 产物生成边界与来源复核

本阶段没有创建真实飞书文档。Coco 最终 `AgentMessage` 提供
`conciliator.resolve.result.v1` 结构化结果；Conciliator 完成 schema、Git、NextCode 和
blame 回读后，把冲突块原文、Coco 决策、归因、最终文件内容和测试结果交给服务端模板，
生成 `lark-document-preview.md`。因此：

- `coco-result.json` 是 Coco 结构化结果的规范化落盘版本。
- `lark-document-preview.md` 由 Conciliator 服务端模板生成，Coco 不直接写文档。
- 人工只读取和原样复制产物，没有修改文件内容。

2026-08-28 再次从 Coco Session `793553926132743` 的现存事件读取最终
`AgentMessage`，重新执行服务端 parser 和同一 JSON 序列化。复算出的 SHA-256 与 Pod 中
`coco-result.json` 完全一致，比较结果为 `match=true`。`validation-summary.json` 记录的
三个产物哈希也与当前文件一致，四个原始文件的修改时间都在
`2026-08-28T14:02:45Z` 这一秒。现有证据能够证明产物来自已记录的 Coco Session 和
Conciliator 生成流程，生成后没有发生内容变化。

为方便人工查看，已在下载白名单目录生成：

```text
/opt/tiger/toutiao/log/conciliator-validation/task-2.tar.gz
```

压缩包 SHA-256 为
`baf1dffcc35ea9d72c7f431f6e85bc3bb9037f6d82d6733bc37c841511bfca45`。通过 Chrome
WebShell 下载后，本地哈希一致；解压目录为
`/Users/bytedance/Downloads/conciliator-validation-task-2/`。原始 `/tmp` 目录和日志目录下
的四个文件没有改动。

### App-owned Agent 的控制台边界

TCE 当前使用的 App-owned 资源为：

```text
Workspace ID: 793550816067727
Workspace slug: conciliator-service
Agent ID: 793550816045691
Agent name: conciliator-resolver
Session ID: 793553926132743
Run ID: 793553926128943
```

生产库任务 2 的 `coco_workspace_id`、`coco_agent_id`、`coco_session_id` 和
`coco_run_id` 与上述值一致。`ListManagedAgentSessionRuns` 也回读到同一个 Agent ID，
Run 在 2026-08-28 22:00:00（UTC+8）进入队列，22:00:23 开始，22:02:40 完成，最终
状态为 `idle`。

个人登录态下的 `https://code.byted.org/managed-agents/conciliator/agents` 只显示旧的
用户态 Agent `793541586477315`。以下 App-owned 候选地址均已用 Chrome 验证，页面返回
`Workspace not found`：

```text
https://code.byted.org/managed-agents/conciliator-service/agents
https://code.byted.org/managed-agents/conciliator-service/agents/793550816045691
https://code.byted.org/managed-agents/793550816067727/agents/793550816045691
```

个人身份执行 Workspace 查询也返回 `AccessDenied`。当前没有可由个人账号打开的
App-owned Agent 控制台链接，运行证据需要从 TCE 内使用 App 身份调用 Coco OpenAPI，
再与生产库任务记录交叉核对。Agent 列表里的 `Last updated` 是 Agent 资源更新时间，
不会随 Session Run 刷新，不能用来判断最近一次运行时间。

结论：本阶段定义的 1—4 步已经全部完成。飞书 OpenAPI、MR Description 和创建
resolver MR 留到下一阶段。

## 参考资料

- [Conciliator 设计文档](../specs/2026-07-21-conciliator-design.md)
- [NextCode OpenAPI 与 Webhook 契约验证](./2026-08-27-nextcode-openapi-webhook-contract-validation-revised.md)
- [Coco OpenAPI Next — 云端 Agent 能力，开箱即用](https://bytedance.larkoffice.com/docx/Tx9zdKjJPo3SsGxNRwBcrKQcnNk)
- [Coco OpenAPI Next 用户文档](https://bytedance.larkoffice.com/docx/UoCCdwEYBoFRYGxrm2dcbKq8nPe)
- [Coco OpenAPI Next 快速开始与实战示例](https://bytedance.larkoffice.com/docx/HFs2dKivWouV2Mx0V2DcGK26n3b)
- 已人工验证 Skill：`/Users/bytedance/Downloads/conflict-resolver`
