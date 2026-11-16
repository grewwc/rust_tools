# Conciliator `rds-tce-runtime` 后续实施计划

- 日期：2026-08-20
- 运行验证更新：2026-08-25，分支 `feat/infra-only-mode`
- 原始评审基线：开发机 `/data00/work/Conciliator`，提交 `1e7b2c4672e1c23fcd876ea452da413efe182b25`
- 对照设计：[`2026-07-21-conciliator-design.md`](../specs/2026-07-21-conciliator-design.md) 与 [`2026-07-21-conciliator.md`](2026-07-21-conciliator.md)
- 外部依据：
  - [NextCode documentation](https://bytedance.larkoffice.com/wiki/HwJFwyuqIixwFykXM71csXMZnpc)
  - [Coco OpenAPI Next — 云端 Agent 能力，开箱即用](https://bytedance.larkoffice.com/docx/Tx9zdKjJPo3SsGxNRwBcrKQcnNk)

## 1. 结论

当前分支已经完成 RDS/TCE 运行时、数据库模型、状态机、worker、审批服务和三类外部系统适配层的代码骨架。2026-08-25 已在生产 TCE 完成 infra-only 验证：1 个集群的 2 个实例均运行，`/readyz` 返回 `status=ok`，RDS 可连接；但还没有形成可调用真实 NextCode/Coco OpenAPI 的端到端实现。

最关键的问题不是三个 README 中已经列出的 v2 TODO，而是 NextCode 与 Coco 客户端建立在假定的 REST 契约上：当前 mock 测试会通过，但请求 URL、鉴权 header、响应 envelope、Webhook 格式和 Coco 任务模型均与最新文档不一致。真实环境下，主流程会在创建 resolve 分支之前或提交 Coco 任务时失败。

后续工作必须按以下顺序推进：

1. 先锁定 NextCode/Coco 的真实 API 契约并重写适配层。
2. 再重塑 Coco Session/SSE 驱动的 pipeline 和持久化状态。
3. 然后补冲突块、blame、审批 retry 和 MR 动态更新。
4. 最后以真实契约测试和标杆仓库 E2E 作为上线门禁。

## 2. 当前已完成的部分

| 能力 | 当前状态 | 说明 |
|---|---|---|
| TCE/Atum 运行时 | 基础运行已验证 | `atum.yaml`、`run/prod_run.sh`、内部依赖安装、`PORT0`、多 worker 启动已具备；生产集群 `219250886` 的 2 个实例均运行 |
| RDS 接入 | 生产连接已验证 | `bytedmysql`、同步 SQLAlchemy + threadpool、RDS PSM 配置、初始 DDL/ORM 已具备；TCE `/readyz` 数据库 ping 返回 `status=ok` |
| 配置与密钥 | 基本完成 | KMS provider、开发环境 fallback、必填配置已具备；Coco 配置项需按新契约调整 |
| 领域状态机 | 基本完成 | Task/Approval 状态与乐观锁已实现，但 retry、超时、恢复语义仍有错误 |
| Web 服务 | 基础运行已验证 | FastAPI、health/readiness 已在 TCE 验证；NextCode/Coco webhook 路由和 worker lifecycle 已具备但尚未完成真实契约 E2E |
| 审批交互 | 部分完成 | 指令解析、approve/reject/status/cancel 和 DB 更新已实现；真实 Thread/Comment 契约、admin、retry 未完成 |
| 可观测性 | 部分完成 | structlog 与 metrics wrapper 已有；审计事件、阶段耗时、外部调用指标和告警未闭环 |
| 自动化测试 | 骨架通过 | 在同一提交的隔离副本上为 `82 passed, 7 skipped`，`mypy` 通过；但外部 API mock 使用了错误契约，7 个集成测试因无 Docker 跳过 |

### 2.1 TCE infra-only 验证证据（2026-08-25）

- TCE 服务：[data.aeolus.conciliator（服务 ID `210017612`）](https://cloud.bytedance.net/tce/services/210017612?control_plane=cn&module=cluster&x-resource-account=public&x-bc-region-id=bytedance)。
- 生产集群：`default`，集群 ID `219250886`，状态 `running`。
- 实例：单机房 1 个、小流量 1 个；TCE 显示实例数 `2/2`、服务健康分 100。
- 制品：主仓库 `data/dp/aeolus_conciliator:1.0.0.4`，镜像版本 `1.0.0.7`。
- 环境：`ENV=prod`、`INTEGRATIONS_ENABLED=false`、
  `MYSQL_PSM=toutiao.mysql.aeolus_govern_write`、`MYSQL_DB_NAME=aeolus_govern`。
- 结果：`/readyz` 返回 `status=ok`；该检查包含数据库 ping，证明 TCE HTTP
  服务和 RDS 连接均正常。
- 范围：本次只通过 infra-only 门禁；KMS、ZTI、NextCode Action、Coco Session/SSE、
  飞书写入和后台 Worker 尚未验证，不能据此宣称核心 pipeline 已完成。

## 3. 最新文档带来的契约修正

### 3.1 NextCode

  - [Codebase(NextCode) OpenAPI 接入指南](https://bytedance.larkoffice.com/wiki/PKoiwOfVniSnO6kEomWcett0nIc)
  - [NextCode Webhook 接入指南](https://bytedance.larkoffice.com/wiki/DOOqwgVBWiJ5Zukjjf2cdRJJnOg)

NextCode OpenAPI 的统一调用方式是：

```text
POST https://codebase-api.byted.org/v2/?Action=<Action>
X-Code-App-ID: <app-id>
X-Code-App-Secret: <app-secret>
X-Code-ZTI-Token: <dynamic-zti-token>
Content-Type: application/json
```

成功响应为 `ResponseMetadata + Result`，失败响应在 `ResponseMetadata.Error` 中返回 `Code/Message`。所有 API 都通过 `Action` 区分，而不是当前实现中的 `/v1/repos/...`、`/v1/projects/...` REST path。

与本项目相关的已确认 Action 包括：

| 领域 | Action |
|---|---|
| 分支 | `CreateBranch`、`GetBranch` |
| MR | `CreateMergeRequest`、`GetMergeRequest`、`UpdateMergeRequest`、`GetMergeRequestMergeability` |
| Diff | `ListDiffFiles`、`ListDiffFileContents`、`ListRangeDiffFileContents` |
| 评论/Thread | `CreateComment`、`GetComment`、`GetThread`、`ListThreads`、`UpdateThread` |
| 仓库内容 | `GetFile`、`ListFiles`、`ListDirectoryEntries` |

Webhook 也不是当前假定的 GitLab 兼容格式：

- 事件类型来自 `X-Vecode-Event`，例如 `merge_request`、`comment`、`merge_request_review_thread`。
- delivery/event 标识来自 `X-Vecode-Delivery`、`X-Vecode-Event-Id`。
- 签名来自 `X-Vecode-Signature`，计算方式为 `sha256=` + `HMAC-SHA256(secret, raw_body)`。
- 最新文档没有 `X-Codebase-Timestamp`，也没有 `timestamp + "." + body` 的签名规则。
- body 使用 `Action`、`Sender`、`Repository`、`MergeRequest`、`Comment` 等 CamelCase 字段。
- App 除了 API scope，还必须安装到目标仓库；仓库写入/clone 权限需要单独审批。

### 3.2 Coco OpenAPI Next

Coco OpenAPI Next 与 NextCode 共用 `codebase-api.byted.org/v2/` 的 Action 风格接口。平台方可以使用 Codebase App ID/Secret；当前实现中的 Bearer `coco_api_token` 和 `/v1/agents/tasks` 不在最新主链路中。

官方主路径是：

```text
一次性准备：Workspace -> system Device -> Agent
每个任务：CreateManagedAgentSession
       -> SendManagedAgentSessionEvents(UserMessage)
       -> SubscribeManagedAgentSessionEvents(SSE)
       -> SessionStatusIdle / SessionStatusTerminated
```

对 Conciliator 最重要的能力是：

- `CreateManagedAgentSession` 用 `Resources.Repository[{RepoId, Revision}]` 挂载仓库；不传 clone URL。
- `SubscribeManagedAgentSessionEvents` 返回 SSE；`AfterEventId` 用于断线续传。
- 过程事件包括 `AgentThinking`、`AgentMessage`、`AgentToolUse`、`SessionStatusIdle` 和 `ping`。
- `SessionStatusIdle.StopReason.RequiresAction` 表示等待工具确认；可通过 `UserToolConfirmation` allow/deny。
- `ListManagedAgentSessionRuns`、`GetManagedAgentSessionStats`、`CancelManagedAgentSessionRun` 用于状态、成本、超时与取消。
- 最新文档没有保证返回当前代码假定的 `file_diffs`、`thinking`、`commit_sha` 聚合结果。

### 3.3 TCE 内 clone 的架构约束

 [Codebase 拉代码方案 (分场景)](https://bytedance.larkoffice.com/wiki/NuzEwex31iyVErko1gkcIA0XnHh) 明确说明：在 TCE 或其他普通容器内直接 clone 代码不合规，在线/离线自动化拉代码应走 CI/Bits 等受控环境。Conciliator 已选用 Coco 云端沙箱，因此运行时应删除“在 TCE 本地 clone + merge + blame”的主路径，改为：

1. 通过 Coco Session 的 `Resources.Repository` 挂载 `resolve_branch`。
2. 在受控沙箱中执行 merge、冲突采集、blame、测试、commit 和 push。
3. Conciliator 只接收结构化冲突报告和执行事件，不在 TCE 落仓库代码。

## 4. 缺口清单与优先级

### P0：不补就无法调用真实系统

| 缺口 | 当前实现 | 必须补齐 |
|---|---|---|
| NextCode transport | 任意 HTTP method + `/v1/...` path | 统一 `POST /v2/?Action=...`，解析 `Result/ResponseMetadata.Error`，记录 `RequestId` |
| NextCode 鉴权 | `X-Codebase-App-*` + `X-Bytedance-DPS-Token` | 改为 `X-Code-App-ID/Secret` + 每次请求动态获取的 `X-Code-ZTI-Token` |
| NextCode 领域适配 | GitLab 风格 snake_case response | 按 BAM/SDK v2 的 Action request/response 类型重写 branch/MR/diff/comment/thread/member adapter |
| Webhook 安全与路由 | `X-Codebase-*`、时间戳、body 内 `event_type` | 按 `X-Vecode-*` 校验；事件类型从 header 读；解析 CamelCase payload；按 Event ID 去重 |
| MR 冲突触发 | 假定 webhook body 有 `conflicts > 0` | 收到 `opened/pushed/edited` 后调用 `GetMergeRequestMergeability` 再判断；只对 repo config 白名单入队 |
| Coco 调用模型 | `/v1/agents/tasks` submit/get + Bearer token | 改成 Session + UserMessage + SSE；使用 WorkspaceId/AgentId 和 Codebase App 凭证 |
| Coco 仓库输入 | prompt 传 `repo_url`，Agent 自行 clone | Session 创建时传 `Resources.Repository{RepoId, Revision=resolve_branch}` |
| Coco 完成判定 | fake `status=succeeded` 和假想 `/hooks/coco` | 消费 SSE，以 `SessionStatusIdle/Terminated`、Run 状态和 Stats 判定；cursor 持久化后断点续传 |
| Coco/本地任务映射 | 把外部 `task_id` 强转成本地 task PK | 持久化 `coco_session_id/run_id/last_event_id`，所有事件先按 session id 反查 task |
| 结果提取 | 假定返回 `file_diffs/thinking/commit_sha` | thinking/message 从 SSE Delta 聚合；commit/diff 用 `GetBranch` + `ListDiffFileContents` 确认；禁止信任未验证字段 |
| 冲突采集 | 普通 compare diff 当作带冲突标记的文件 | 先验证 BAM 是否提供冲突明细；若没有，使用 Coco 沙箱内的确定性 collector Skill 生成结构化 manifest |
| 飞书写入 | 假定 create document 可直接接 XML content | 对照飞书 OpenAPI 验证创建 + 批量 block 写入，并加入大文档分批、重试、降级 |

### P1：骨架存在但业务语义未完成

| 缺口 | 代码证据 | 目标行为 |
|---|---|---|
| `git blame` 未接入 | `_step2_identify_conflicts` 固定写 `author_email=None` | collector 对 ours 侧运行 blame/mailmap，保存 author；失败时明确降级策略 |
| retry 实际不可用 | `RETRYING` 仍调用完整 `run_pipeline`，第一步尝试 `RETRYING -> PREPARING`，状态机不允许 | 只把 REJECTED 块、原因和原方案发送到同一或新 Session；更新原 Thread，approval 重置为 PENDING |
| admin 权限未接 | `_is_mr_author_or_admin` 只比较 MR author | 接真实仓库成员/权限 Action，缓存 5 分钟；权限 API 不可用时 fail closed |
| MR 描述不动态更新 | 初次创建后不再调用 `update_mr_description` | approve/reject/retry/cancel/完成时刷新状态和 checklist |
| Discussion 语义未验证 | 使用 GitLab `/discussions` + `new_line` | 基于 `CreateComment`/Thread 契约验证行级 position、ThreadId 和 reply 方式；无行级能力时退化为普通 Thread 并保留 block marker |
| Coco 超时错误 | poll 60 秒看到非成功就直接 `COCO_TIMEOUT -> FAILED` | 按 15 分钟硬超时；运行中继续订阅；超时调用 cancel；区分 cancelled/terminated/failed |
| cancel 不取消外部 Run | 只更新 DB 状态 | 调 `CancelManagedAgentSessionRun` 或发送 `UserInterrupt`，再落终态 |
| 失败通知不完整 | pipeline catch 只改 DB | 对可定位的原 MR/resolve MR 发失败原因和重试建议；错误正文脱敏 |
| 幂等不完整 | 只查活跃任务；comment webhook 未按 event id 去重 | 增加 webhook event 唯一键；分支/MR/Thread 创建使用可恢复的业务幂等键 |
| 多进程并发超配 | Uvicorn 4 进程 × 每进程 4 worker | 明确总并发；默认单进程 + 4 协程，或实现 DB claim/lease 后再扩多进程 |
| repo 并发限制未使用 | `repo_max_concurrent_tasks` 仅配置 | worker claim 时按 repo 限流，避免同仓库多次 merge/push 互相影响 |
| 审计链不完整 | `task_events` 主要只写假想 Coco webhook | 所有状态迁移、外部请求结果、Webhook、命令、重试和取消写审计事件 |
| 指标不完整 | 只在少数成功/失败点 emit | 补阶段耗时、SSE 重连、外部 API outcome、queue depth、超时和 action-required 指标 |

### P2：上线可信度与运维闭环

- 契约测试：从 BAM/Codebase SDK v2 固定 request/response fixture，禁止继续手写猜测字段。
- Webhook fixture：保存脱敏后的真实 `merge_request`、`comment`、`merge_request_review_thread` 样本。
- E2E：标杆冲突仓库完成 approve 路径和 reject/retry 路径。
- 权限验收：App scope、仓库安装、写权限、Managed Agents scope、ZTI、飞书权限逐项验证。
- 灰度：先仅开放 `/resolve` 手动触发和单仓库白名单；稳定后再开启 auto conflict。
- 告警：失败率、Coco terminated/timeout、SSE 重连、Webhook 签名失败、队列积压。
- 数据保留：task events 和冲突代码的 90 天清理任务，以及飞书文档访问范围检查。
- 文档同步：修订旧 spec 中失效的 REST/Webhook/Coco webhook/TCE clone 设计，README 不再宣称核心 pipeline 已完成。

## 5. 建议的目标链路

```text
NextCode X-Vecode Webhook
  -> 校验 raw-body HMAC + EventId 去重
  -> GetMergeRequest / GetMergeRequestMergeability
  -> CreateBranch(resolve/...)
  -> CreateManagedAgentSession(
       WorkspaceId,
       AgentId,
       Resources.Repository{RepoId, Revision=resolve_branch}
     )
  -> SendManagedAgentSessionEvents(UserMessage)
  -> SubscribeManagedAgentSessionEvents(AfterEventId 可恢复)
       -> 聚合 Thinking/Message
       -> 处理 action-required 或采用受控 dont_ask Agent
       -> Idle: GetBranch + ListDiffFileContents 校验 push 结果
       -> Terminated/timeout: cancel + 失败通知
  -> 结构化 conflict manifest 入库
  -> 创建飞书报告
  -> CreateMergeRequest + CreateComment/Thread
  -> comment/thread webhook 驱动逐块审批
  -> 全 approve 后通知，不自动合入
```

## 6. 分阶段实施安排

### Phase 0：契约冻结与设计校正

目标：在写业务代码前消除所有未验证假设。

- [ ] 从 BAM API 页面或 `codebase/sdk` v2 提取下列 Action 的 request/response schema：`CreateBranch`、`GetBranch`、`GetMergeRequest`、`GetMergeRequestMergeability`、`CreateMergeRequest`、`UpdateMergeRequest`、`ListDiffFileContents`、`CreateComment`、`GetThread`、成员权限查询。
- [ ] 用一个测试 App 在测试仓库验证 scope、安装、分支写入和行级 Thread 能力；所有样本脱敏后保存到 `tests/fixtures/nextcode/`。
- [ ] 验证 `GetMergeRequestMergeability` 是否返回冲突文件/冲突块内容。若没有，确认采用“Coco collector Skill”方案，不再实现 TCE 本地 clone fallback。
- [ ] 用测试 Workspace/Agent 验证 app identity、repo resource mount、push resolve branch、SSE、`AfterEventId`、Stats、cancel 和 action-required；样本保存到 `tests/fixtures/coco/`。
- [ ] 更新设计 spec 的第 3、5、9、11、12、13 节，移除失效协议和 TCE clone 方案。

验收：每个外部调用都能指向一个官方 schema 或真实脱敏 fixture；代码中不再存在“假设返回”的字段。

### Phase 1：重写统一的 Codebase Action 客户端

主要文件：

- `src/conciliator/integrations/nextcode/client.py`
- `src/conciliator/integrations/nextcode/{branches,mrs,merges,discussions,repos}.py`
- `src/conciliator/config.py`
- `tests/unit/test_nextcode_*.py`

任务：

- [ ] 把 `_request(method, path, ...)` 改为 `_call(action, body, ...)`。
- [ ] 使用 `X-Code-App-ID/Secret`，接入动态 ZTI provider，不能缓存固定 ZTI 字符串。
- [ ] 正确处理 HTTP 429/5xx、`ResponseMetadata.Error`、RequestId 和指数退避；写接口只在明确幂等时自动重试。
- [ ] 按官方 CamelCase schema 建 typed DTO，再映射成领域内 snake_case 对象。
- [ ] 为每个 Action 增加基于官方 fixture 的 contract test。

验收：测试不再出现 `/v1/repos`、`/v1/projects`、`code/data/msg` 等旧假定；测试断言实际 Action、headers、body 和 envelope。

### Phase 2：重写 NextCode Webhook 入口

主要文件：

- `src/conciliator/api/webhooks.py`
- `src/conciliator/api/comment_router.py`
- `src/conciliator/integrations/nextcode/client.py`
- `src/conciliator/repositories/{models,task_repo,_sync_repo}.py`

任务：

- [ ] 按 `X-Vecode-Signature` 校验 `sha256=HMAC(secret, raw_body)`；删除文档未定义的 timestamp 依赖。
- [ ] 从 `X-Vecode-Event` 路由事件，从 `X-Vecode-Event-Id`/delivery 建 webhook 幂等键。
- [ ] 为 `merge_request`、`comment`、`merge_request_review_thread` 建 Pydantic payload 模型。
- [ ] MR opened/pushed/edited 时异步调用 mergeability 再决定是否触发，确保 handler 只做验证、去重、入队并快速返回。
- [ ] 解析 Comment/Thread 的真实关系，不用 GitLab `object_attributes/discussion_id` 猜测。

验收：真实脱敏 webhook fixture 全部通过；重复 EventId 不重复创建任务或执行审批命令。

### Phase 3：实现 Coco Session/SSE 适配器

主要文件：

- `src/conciliator/integrations/coco/client.py`
- `src/conciliator/integrations/coco/events.py`
- `src/conciliator/workers/task_runner.py`
- `src/conciliator/config.py`

任务：

- [ ] 增加 `coco_workspace_id` 配置，保留前置创建的 `coco_default_agent_id`；删除独立 Coco Bearer token 主路径。
- [ ] 实现 create session、send user message、subscribe SSE、list events/runs、stats、cancel/interrupt。
- [ ] 增量拼接 `AgentThinking`/`AgentMessage`；把 event id 持久化为 cursor，断线后使用 `AfterEventId` 恢复。
- [ ] 正确处理 `SessionStatusIdle.StopReason.RequiresAction`。MVP 建议使用专用、最小权限、仅挂载 resolve branch 的 `dont_ask` Agent；若安全评审不接受，再实现基于 allowlist 的 `UserToolConfirmation`。
- [ ] 以 `SessionStatusIdle` 结束一轮，以 `SessionStatusTerminated`/Run status 识别失败，不再维护 `/hooks/coco` 假想主路径。

验收：模拟 SSE 断线重连不会丢事件或重复推进；timeout/cancel 会终止外部 Run；无本地 task id 与 Coco session id 混用。

### Phase 4：重塑 resolve pipeline 与持久化

主要文件：

- `src/conciliator/services/resolve_service.py`
- `src/conciliator/repositories/{models,task_repo,_sync_repo}.py`
- `src/conciliator/workers/task_runner.py`
- `migrations/versions/0002_*.py`
- `ddl/0002_*.sql`

建议新增字段：

- `coco_session_id`
- `coco_run_id`
- `coco_last_event_id`
- `source_commit_sha`、`target_commit_sha`、`resolve_commit_sha`
- `coco_final_message` 或单独的 run/result 表

任务：

- [ ] 把 submit/poll 状态改成 session created/running/action-required/idle/terminated 的可恢复编排。
- [ ] Session 挂载 `RepoId + resolve_branch`；prompt 要求 merge 指定 target revision，只允许 push 到 resolve branch。
- [ ] 若 NextCode 无冲突明细 API，实现 Coco collector Skill：在沙箱中确定性执行 merge、解析 conflict markers、对 ours 侧 blame/mailmap，并输出 JSON schema 的 manifest。
- [ ] 对 manifest 做严格 JSON schema 校验；缺块、重复块、非法路径、空 author 等情况有显式降级或失败策略。
- [ ] Coco 完成后调用 `GetBranch` 确认新 commit，再用 `ListDiffFileContents` 获取可信 diff；确认目标分支不被修改。
- [ ] 飞书报告和 MR/Thread 创建改为可恢复步骤：每个外部资源 ID 落库后再进入下一状态。
- [ ] 实现真正的 15 分钟 Coco 超时、20 分钟总超时和失败通知。
- [ ] worker 引入 DB claim/lease 或限定单进程，保证多副本不会重复执行外部写操作。

验收：任意一步进程重启后都能从已落库的外部 ID/cursor 继续；不会重复创建分支、MR、文档或 Thread。

### Phase 5：补齐审批闭环

主要文件：

- `src/conciliator/services/approval_service.py`
- `src/conciliator/services/resolve_service.py`
- `src/conciliator/services/templates.py`
- `src/conciliator/integrations/nextcode/{discussions,mrs,repos}.py`

任务：

- [ ] 接真实成员权限查询，实现 MR author/admin 权限并做 5 分钟缓存。
- [ ] approve/reject 后更新 MR description checklist 和 Thread 回复中的 X/Y 进度。
- [ ] 实现仅重解 REJECTED 块：携带 block、reject reason、旧方案，在原 Session 多轮追问或创建明确关联的新 Session。
- [ ] retry 完成后只更新原 Thread，保留已 APPROVED 块，把目标 Approval 行从 REJECTED 重置为 PENDING 并保留审计事件。
- [ ] cancel 时同时终止 Coco Run；终态任务忽略后续事件。
- [ ] 明确 Thread 行定位失效时的 fallback，避免 MR 新 commit 后因 position 过期导致整个任务失败。

验收：标杆仓库的 approve-all、单块 approve、reject -> retry -> approve、status、cancel 均与 DB/MR/Thread 状态一致。

### Phase 6：契约、E2E、灰度和文档收口

- [ ] NextCode/Coco/Lark contract tests 全部使用官方 schema 或真实脱敏 fixture。
- [ ] 增加完整 service integration test，覆盖 `PENDING -> ALL_APPROVED` 和 reject/retry 分支。
- [ ] 在测试仓库手动跑两条 E2E，保存 task event、MR、飞书报告、Coco session/run 证据。
- [x] TCE infra-only smoke test：2 个实例运行，health/ready 正常，RDS 可连接。
- [ ] TCE 集成 smoke test：KMS、ZTI、NextCode Action、Coco session、飞书写入逐项验证。
- [ ] 先以单仓库白名单 + 手动 `/resolve` 灰度；自动触发保持关闭。
- [ ] 补 dashboards/alerts 和 runbook；修复现有 ruff 的两个 import-order 错误。
- [ ] 更新 README 的“已完成/待办”，只有真实 E2E 通过后才宣称核心 pipeline 已完成。

## 7. 测试与上线门禁

### 合并门禁

- `uv run pytest`：单元 + mock integration 全绿，无无理由 skip。
- `uv run ruff check .`：零错误。
- `uv run mypy src/`：零错误。
- NextCode/Coco contract fixture 与代码 DTO 一致。
- schema migration、DDL 导出和 downgrade/upgrade 对称性通过。

### 上线门禁

1. 手动触发 E2E：冲突 MR -> resolve branch -> Coco push -> 飞书报告 -> resolve MR -> Thread approve -> ALL_APPROVED。
2. retry E2E：至少一个块 reject -> retry -> 原 Thread 新方案 -> approve，已通过块不变化。
3. 故障恢复：SSE 中断、进程重启、重复 webhook、NextCode 429、Coco timeout 各验证一次。
4. 安全验证：App 最小 scope、仓库安装、ZTI、resolve branch 限制、敏感日志扫描。
5. 观察期内无重复 MR/Thread、无目标分支意外写入、无 TCE 本地 clone。

## 8. 建议拆分的 MR

1. `fix/nextcode-action-contract`：NextCode transport、DTO、contract tests。
2. `fix/nextcode-webhook-contract`：X-Vecode webhook、安全、去重、真实 fixtures。
3. `feat/coco-session-sse`：Session、SSE、cursor、Run timeout/cancel。
4. `feat/resolve-pipeline-v2`：repo resource、collector、commit/diff 校验、DB migration。
5. `feat/approval-retry-v2`：真实 Thread、权限、动态描述、partial retry。
6. `test/conciliator-e2e`：契约/E2E/灰度门禁、README/spec 收口。

每个 MR 都应保持可回滚，不混入 RDS/TCE 基础设施重构；数据库变更必须同时提交 Alembic migration 和 `ddl/*.sql`。

## 9. 明确不做

- 不自动合入 resolve MR。
- 不在 TCE 容器直接 clone 或长期保存仓库代码。
- 不在未验证 BAM/SDK schema 前继续猜测 NextCode 字段。
- 不依赖文档未定义的 Coco webhook、`file_diffs` 或聚合 task result。
- 不在自动 conflict 模式开启前跳过手动触发灰度。
