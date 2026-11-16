# Conciliator 设计文档

- **日期**：2026-07-21
- **作者**：夏海涛
- **状态**：Draft
- **参考**：
  - [创建 App（codebase app 开发文档）](https://bytedance.larkoffice.com/wiki/EYfEwivTbiNmBhkF4Mpcfjqvnhb)
  - [Coco OpenAPI Next](https://bytedance.larkoffice.com/docx/Tx9zdKjJPo3SsGxNRwBcrKQcnNk)

---

## 1. 目的与范围

Conciliator 是一个 codebase app（独立后端服务），在 MR 存在冲突时自动解冲突，把“AI 解冲突方案 + 冲突代码 author 邮箱”汇总到飞书文档，在新 MR 上发起逐块确认流程，owner 全部 approve 后通知合入（**不自动合入**）。

### 核心功能

1. 基于当前 MR 的源分支，创建解冲突分支
2. 自动完成解冲突，把所有冲突块与解冲突方案汇总到飞书文档，标记每个冲突块的 author（email）
3. 以解冲突分支为源分支、原 MR 目标分支为目标分支创建新 MR，描述里带飞书文档链接
4. 冲突代码逐块确认机制（owner 在 Discussion 里 approve/reject）

### 范围切分

本设计覆盖完整 pipeline 与确认机制。**不在范围**：

- 自动合入（通知但由人合入）
- 外部 LLM 接入（全部委托给 Coco OpenAPI Next）
- 代码安全扫描、构建验证（交给 Coco Agent 沙箱负责）
- 多租户管理（单实例服务多仓库）

---

## 2. 关键决策（已确认）

| 决策项           | 选择                                                                   |
| ---------------- | ---------------------------------------------------------------------- |
| 技术栈           | Python + FastAPI                                                       |
| 触发方式         | 评论指令 `/resolve` + 冲突状态自动（按仓库配置开关）                   |
| 解冲突引擎       | LLM 全部委托给 Coco OpenAPI Next（云端 Agent 沙箱）                    |
| Author 定位      | `git blame` 最后修改者（支持 mailmap）                                 |
| 确认机制         | 在 MR Discussion 里逐块 approve/reject（每条 Discussion = 一个冲突块） |
| 持久化           | 字节云 RDS（MySQL）                                                    |
| 飞书文档         | 服务从 Coco 事件流拾取 + 自己生成                                      |
| 部署             | 字节云应用平台                                                         |
| 分支 commit 落地 | Coco Agent 直接 push 到解冲突分支                                      |
| 闭环动作         | 全 approve 后通知，不自动合入                                          |

---

## 3. 整体架构

### 3.1 角色边界

| 模块                                     | 职责                                                                                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Conciliator 服务（Python + FastAPI）** | webhook 接收/校验、分支编排、冲突块解析、`git blame`、Coco 事件流拾取、飞书文档生成、MR Discussion 创建、审批状态机、MySQL 持久化 |
| **NextCode OpenAPI**                     | MR/分支/commit/Discussion 增删查、webhook 事件源                                                                                  |
| **Coco OpenAPI Next**                    | 在云端沙箱读冲突、生成解冲突代码、commit、push 到解冲突分支                                                                       |
| **飞书 OpenAPI**                         | 上传 Markdown 报告、按邮箱解析 `open_id`、以应用机器人身份发送完成私聊                                                           |
| **MySQL（字节云 RDS）**                  | 任务、冲突块、approval、事件流持久化                                                                                              |

核心原则：**确定性逻辑在服务，AI 能力全部委托给 Coco**。

### 3.2 数据流总览

```
[NextCode MR]
   │ webhook(comment /resolve 或 merge_request 冲突状态变更)
   ▼
[Conciliator FastAPI]
   │ 1. 校验 webhook 签名 + 解析 MR/分支
   │ 2. NextCode API:从源分支创建 resolve/<source>-<ts> 分支
   │ 3. 读冲突文件列表 + 解析冲突块 + git blame 定位 author email
   │ 4. Coco OpenAPI:提交"解冲突"任务(Agent 在沙箱改码 + push 到 resolve 分支)
   │ 5. 订阅 Coco 事件流 → 拾取 diff / thinking
   │ 6. 飞书 OpenAPI:创建文档,写入"冲突块 + 方案 + author email"
   │ 7. NextCode API:创建 MR(resolve 分支 → 目标分支),描述带文档链接
   │ 8. NextCode API:每个冲突块创建一条 Discussion
   ▼
[审批闭环]
   owner 在 Discussion 里回复 approve/reject → webhook comment 事件
   → 服务更新 MySQL 审批状态 → 全 approve → 在 MR 评论通知，并私聊 `/resolve` 发起人
```

### 3.3 部署形态

- 单服务，字节云应用平台
- 环境变量注入 App ID/Secret、Coco Token、飞书 App 凭证、MySQL DSN
- webhook URL 暴露为 `https://<app>/hooks/codebase`，在 codebase app 配置中注册
- 同时暴露 `/hooks/coco` 接收 Coco 事件回调

---

## 4. 项目结构

```
Conciliator/
├── pyproject.toml                # uv 依赖管理
├── .env.example
├── README.md
├── src/
│   └── conciliator/
│       ├── main.py               # FastAPI app 入口 + lifespan
│       ├── config.py             # pydantic-settings
│       ├── api/                  # HTTP 入口层(协议转换)
│       │   ├── webhooks.py       # /hooks/codebase、/hooks/coco
│       │   └── health.py         # /healthz、/readyz、/metrics
│       ├── domain/               # 领域模型(纯数据 + 业务规则,无 IO)
│       │   ├── task.py           # ResolveTask 聚合根
│       │   ├── conflict.py       # ConflictBlock / ConflictFile
│       │   ├── approval.py       # Approval 状态机
│       │   └── events.py         # 领域事件
│       ├── services/             # 应用服务层(编排 + 事务边界)
│       │   ├── resolve_service.py     # 主编排 pipeline
│       │   ├── approval_service.py    # 处理 approve/reject/retry
│       │   └── notification_service.py# MR 评论 @ 通知
│       ├── integrations/         # 外部系统适配层(每系统一子包)
│       │   ├── nextcode/
│       │   │   ├── client.py     # HTTP + 鉴权(App Secret 签名)
│       │   │   ├── branches.py
│       │   │   ├── merges.py
│       │   │   ├── mrs.py
│       │   │   └── discussions.py
│       │   ├── coco/
│       │   │   ├── client.py     # 提交 Agent 任务
│       │   │   └── events.py     # 事件流订阅(webhook + poll 兜底)
│       │   ├── lark/
│       │   │   ├── client.py     # app_access_token 缓存
│       │   │   ├── docs.py
│       │   │   └── contacts.py   # open_id → email
│       │   └── git/
│       │       ├── repo.py       # clone 到临时目录 / cleanup
│       │       ├── conflicts.py  # 解析 <<<<<<< ======= >>>>>>>>
│       │       └── blame.py
│       ├── repositories/         # MySQL 持久化(SQLAlchemy)
│       │   ├── db.py
│       │   ├── models.py
│       │   └── task_repo.py
│       ├── workers/
│       │   └── task_runner.py    # 轮询 PENDING/RETRYING 任务
│       └── utils/
│           ├── logging.py
│           ├── retry.py
│           └── text.py
├── migrations/                   # Alembic
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/superpowers/specs/
```

### 模块设计原则

每个模块满足“单一职责 + 清晰接口 + 可独立测试”。换 LLM 只动 `integrations/coco/`；换存储只动 `repositories/`。`domain/` 纯函数式，不依赖 IO。

---

## 5. 核心 Pipeline

### 5.1 触发阶段（webhook handler，只入队）

```
NextCode webhook 到达
   │
   ▼
api/webhooks.py:
   1. 校验签名(X-Codebase-Signature,HMAC-SHA256 + App Secret)
   2. 时间戳校验(防重放,超过 5 min 拒绝)
   3. 解析事件类型:
      - comment.created,body 含 "/resolve"  → Trigger.COMMENT
      - merge_request.updated,conflicts > 0  → Trigger.AUTO_CONFLICT
        (auto 模式需仓库配置 auto_resolve_enabled=true)
   4. 幂等:同 MR + 源分支 + 目标分支已有 running 任务,丢弃
   5. repositories.task_repo.create(status=PENDING, payload=Trigger)
   6. 立即返回 200(< 100ms)
```

### 5.2 执行阶段（resolve_service.run_resolve）

worker.task_runner 轮询 PENDING/RETRYING 任务 → 调 resolve_service：

**Step 1 任务预处理 `_prepare`**

- 从 Trigger 解析 repo_id、source_branch、target_branch、original_mr_id
- 从评论 webhook 记录 `trigger_author_email`；它表示输入 `/resolve` 的用户，可能与 MR 作者相同，也可能不同
- 从 `nextcode.mrs.get(original_mr_id)` 获取 `mr_author_email`，不要用触发人覆盖 MR 作者
- 生成 resolve_branch = `resolve/{source_branch}/{ts}`
- `nextcode.branches.create(repo, source_branch, resolve_branch)`
- task.status = BRANCH_CREATED

**Step 2 冲突识别 + author 定位 `_identify_conflicts`**

- `nextcode.merges.list_conflict_files(repo, resolve_branch, target_branch)` → List[ConflictFile]
- [fallback] 若 API 不返回冲突内容：`git.repo.clone` + `git merge --no-commit target_branch` 触发冲突
- 对每个冲突文件：
  - `git.conflicts.parse(text)` → List[ConflictBlock(file_path, start_line, end_line, ours_text, theirs_text)]
  - `git.blame.blame_lines(file_path, start_line..end_line)` → author_email（ours 侧最后修改者，经 mailmap 映射）
- 持久化所有 ConflictBlock

**Step 3 委托 Coco Agent `_dispatch_to_coco`**

- 构造 Agent 任务 prompt（仓库 + resolve_branch + target_branch + 冲突文件列表 + 已识别冲突块行号；要求沙箱 merge target、解所有冲突、commit、push 到 resolve_branch）
- `coco.client.submit_resolve_task(agent_id, prompt)` → coco_task_id
- task.coco_task_id = ...,status = COCO_RUNNING

**Step 4 事件流拾取 + 飞书文档 `_collect_and_document`** （Coco 任务完成后触发）

- `coco.events.fetch_run(coco_task_id)` → 所有 diff/thinking 事件
- 对每个 ConflictBlock，从 diff 事件匹配该 file 的变更，抽出 AI 解冲突方案
- `lark.docs.create(标题=MR 标题 + resolve_branch)`
- 写入内容（见 5.4 飞书文档模板）
- task.lark_doc_url = ...,status = DOC_CREATED

**Step 5 创建 MR + Discussion `_create_mr_and_discussions`**

- `nextcode.mrs.create(source=resolve_branch, target=target_branch, title="[resolve] {original}", description=MR_DESC_TEMPLATE(...))`
- 回读新 MR，持久化 `task.new_mr_id` 和 NextCode 返回的 `task.new_mr_url`，status = MR_CREATED
- 对每个 ConflictBlock：
  - `nextcode.discussions.create(mr_id, file_path, start_line, body=DISCUSSION_TEMPLATE(...))`
  - 持久化 Approval(block_idx, discussion_id, PENDING)
- `notification_service.notify(mr_id, "已创建 N 个冲突块待确认,@{author_emails}")`
- task.status = AWAITING_APPROVAL

**Coco 任务完成的双保险接收**：

- 主路径：Coco webhook 推送到 `/hooks/coco`
- 兜底路径：worker 每 30s poll Coco 任务状态
- 任一到达即推进 Step 4

### 5.3 全部确认后的飞书私聊

最后一个冲突块确认完成时，服务在同一个数据库事务中完成三件事：

- `resolve_tasks.status` 转为 `ALL_APPROVED` 并释放 `active_key`
- 写入 `ALL_APPROVED` 审计事件
- 为 `COMMENT` 触发的任务写入唯一的 `lark_notification_outbox` 记录

通知 worker 从 outbox 原子领取任务，通过
`POST /open-apis/contact/v3/users/batch_get_id?user_id_type=open_id`
把 `trigger_author_email` 解析成 `open_id`，再通过
`POST /open-apis/im/v1/messages?receive_id_type=open_id` 发送应用机器人私聊。
消息使用固定的 `uuid=conciliator-all-approved-{task_id}`，内容如下：

```text
所有冲突块已 approve，可以合入。MR链接：<new_mr_url>
```

其中链接使用 NextCode `GetMergeRequest` 回读的 `URL`，不由服务拼接。发送成功后保存
`message_id`。网络、服务端和限流错误进入指数退避重试；找不到用户、权限不足等确定性错误
直接记为 `FAILED`。通知失败不回滚 `ALL_APPROVED`。自动冲突任务没有明确的 `/resolve`
发起人，因此不发这条私聊，也不回退给 MR 作者。

### 5.4 飞书文档模板

````markdown
# MR 自动解冲突报告

- 源分支 → 目标分支：`feature-x` → `main`
- resolve 分支：`resolve/feature-x/<ts>`
- 冲突文件数 / 冲突块数
- resolve MR 链接

## 冲突块详情

### 1. path/to/file.py:12-45

- Author： alice@example.com
- Ours（源分支）：
  ```python
  ...
  ```
- Theirs（目标分支）：
  ```python
  ...
  ```
- 🤖 AI 解冲突方案：
  ```python
  ...
  ```
- Agent 思考（摘要，最多 500 字）：...

### 2. ...
````

### 5.5 审批闭环（approval_service）

详见第 7 节“确认机制”。

### 5.6 异常处理

| 阶段   | 异常                    | 处理                                            |
| ------ | ----------------------- | ----------------------------------------------- |
| Step 1 | 分支已存在 / 权限不足   | task.status = FAILED，MR 评论原因               |
| Step 2 | NextCode 冲突 API 失败  | fallback 本地 git clone；两次都失败 → FAILED    |
| Step 3 | Coco 提交失败           | 重试 3 次（指数退避），失败后 FAILED            |
| Step 3 | Coco 任务超时（15 min） | status = COCO_TIMEOUT，MR 评论通知              |
| Step 4 | 飞书 API 限流           | SDK 自带重试；超 5 次 → 跳过文档步骤，MR 仍创建 |
| Step 5 | MR 创建失败             | 重试 3 次，失败后 FAILED                        |
| 全流程 | 任何未捕获异常          | task.status = FAILED + stacktrace + MR 通知     |

### 5.7 超时与 SLA

- webhook 响应：< 1s（只入队）
- Coco 任务超时：15 分钟
- 单任务总超时：20 分钟
- worker 并发：每实例 4 个（可配置）

---

## 6. 数据模型与状态机

### 6.1 ER 图

```
┌─────────────────────┐         ┌──────────────────────────┐
│   resolve_tasks     │ 1     N │   conflict_blocks        │
│─────────────────────│────────<│──────────────────────────│
│ id (PK, BIGINT)     │         │ id (PK, BIGINT)          │
│ repo_id             │         │ task_id (FK)             │
│ source_branch       │         │ block_idx (INT)          │
│ target_branch       │         │ file_path                │
│ resolve_branch      │         │ start_line               │
│ original_mr_id      │         │ end_line                 │
│ trigger_type        │         │ ours_text (TEXT)         │
│ trigger_payload(JSON)│        │ theirs_text (TEXT)       │
│ status (ENUM)       │         │ ai_resolution (TEXT)     │
│ coco_task_id        │         │ author_email             │
│ lark_doc_url        │         │ discussion_id            │
│ new_mr_id           │         │ created_at               │
│ mr_author_email     │         └───────────┬──────────────┘
│ last_error          │                     │ 1
│ retry_count         │                     ▼ 1
│ created_at          │             ┌──────────────────────┐
│ updated_at          │             │   approvals          │
│ version (optimistic)│             │──────────────────────│
└─────────────────────┘             │ id (PK, BIGINT)     │
                                    │ block_id (FK,UNIQUE)│
                                    │ reviewer_email      │
                                    │ status (ENUM)       │
                                    │ reviewed_at         │
                                    │ comment (TEXT)      │
                                    │ created_at          │
                                    │ updated_at          │
                                    └──────────────────────┘

┌─────────────────────────────┐   ┌─────────────────────────┐
│   task_events (审计)        │   │   repo_configs          │
│─────────────────────────────│   │─────────────────────────│
│ id (PK)                     │   │ repo_id (PK)            │
│ task_id (FK)                │   │ auto_resolve_enabled    │
│ event_type                  │   │ allowed_target_branches │
│ payload (JSON)              │   │   (JSON)                │
│ occurred_at                 │   │ agent_id                │
│                             │   │ created_at / updated_at │
│ UNIQUE(task_id, coco_event_id) │ └─────────────────────────┘
└─────────────────────────────┘
```

`resolve_tasks` 另存 NextCode 回读的 `new_mr_url` 和 `/resolve` 发起人的
`trigger_author_email`。`lark_notification_outbox` 保存通知类型、收件人邮箱和 `open_id`、MR URL、请求 `uuid`、
投递状态、重试次数、下次重试时间、飞书 `message_id` 与最后错误。每个 task 的
`ALL_APPROVED` 通知只能有一条。

### 6.2 关键索引

- `resolve_tasks(repo_id, original_mr_id, status)` — 查 MR 当前任务
- `resolve_tasks(status, updated_at)` — worker 轮询待处理任务
- `conflict_blocks(task_id, block_idx)` UNIQUE
- `conflict_blocks(discussion_id)` UNIQUE — 反查 block
- `approvals(block_id)` UNIQUE — 一块一 approval
- `task_events(task_id, occurred_at)` — 审计回放
- `task_events(task_id, coco_event_id)` UNIQUE — Coco 事件去重
- `lark_notification_outbox(task_id, notification_type)` UNIQUE — 最终通知去重
- `lark_notification_outbox(status, next_retry_at, updated_at)` — 通知 worker 领取与故障恢复

### 6.3 ResolveTask 状态机

```
PENDING
  │ worker 拉起
  ▼
PREPARING(解析 trigger)
  ▼
BRANCH_CREATED
  ▼
CONFLICTS_FETCHED(blocks + blame)
  ▼
COCO_SUBMITTED
  ▼
COCO_RUNNING ──超时──→ COCO_TIMEOUT ──→ FAILED
  │
  │ 完成
  ▼
DOC_CREATED
  ▼
MR_CREATED
  ▼
AWAITING_APPROVAL
  │
  ├─全 approve────→ ALL_APPROVED(终态,通知合入)
  │
  ├─任一 reject───→ PARTIAL_REJECTED
  │                    │
  │                    │ /resolve retry
  │                    ▼
  │                  RETRYING（排队态）
  │                     │ worker 原子认领
  │                     ▼
  │                  RETRY_PREPARING
  │                     ▼
  │                  RETRY_COCO_SUBMITTED(仅重解 REJECTED 块)
  │                                 │
  │                                 ▼
  │                              RETRY_COCO_RUNNING(等待 Coco poll)
  │                                 │
  │                                 ▼ 完成
  │                              RETRY_RESULT_VERIFIED
  │                                 ▼
  │                              RETRY_DOC_UPLOADED
  │                                 ▼
  │                              RETRY_PUBLISHING
  │                                 │ 更新原 MR、原 Discussion，
  │                                 │ 仅将本轮 REJECTED 回退为 PENDING
  │                                 │
  │                                 ▼
  │                              AWAITING_APPROVAL(只待 owner 重审 reject 过的块)
  │
  └─/resolve cancel→ CANCELLED(终态)

任何阶段未捕获异常 → FAILED(终态)
```

**状态分类（给 worker 调度用）：**

| 类别                                | 状态                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| 可调度（worker 拉起/继续）          | PENDING、RETRYING                                                                     |
| 等待外部事件（webhook/poller 推动） | COCO_RUNNING、RETRY_COCO_RUNNING、AWAITING_APPROVAL、PARTIAL_REJECTED                 |
| 中间态（worker 继续）               | PREPARING、BRANCH_CREATED、CONFLICTS_FETCHED、COCO_SUBMITTED、DOC_CREATED、MR_CREATED、RETRY_PREPARING、RETRY_COCO_SUBMITTED、RETRY_RESULT_VERIFIED、RETRY_DOC_UPLOADED、RETRY_PUBLISHING |
| 终态                                | ALL_APPROVED、CANCELLED、COCO_TIMEOUT、FAILED                                         |

### 6.4 Approval 状态机

```
PENDING ──approve──→ APPROVED
   │
   └──reject────────→ REJECTED
                         │
                         └──/resolve retry 触发整 task RETRYING
                            → 该块走 retry 子流程,Approval 行 UPDATE 回 PENDING(保留行,不 DELETE,便于审计)
```

### 6.5 并发与幂等

| 问题                     | 方案                                                           |
| ------------------------ | -------------------------------------------------------------- |
| 同一 MR 短时间多次触发   | webhook handler 查活跃任务，有则丢弃                           |
| 同一 task 被多实例抢占   | 乐观锁 `version` 字段 + `UPDATE ... WHERE id=? AND version=?`  |
| Coco 事件重复推送        | `task_events(task_id, coco_event_id)` UNIQUE                   |
| 重复 approve 同一 block  | approval 已 APPROVED 时，重复 approve 幂等返回成功；outbox 唯一键保证不重复通知 |
| comment webhook 重复推送 | 按 webhook payload 里 `event_id` 进 `task_events` UNIQUE 去重  |

---

## 7. 确认机制（逐块 approve/reject）

### 7.1 用户交互全景

每条 Discussion = 一个冲突块。owner 在对应 Discussion 里用**简短回复**触发 approve/reject，服务用 `discussion_id` 反查 block，无需带 idx。

```
resolve MR #1234  [resolve/feature-x → main]

描述:
  自动解冲突报告 → 飞书文档链接
  待确认冲突块:3 个
  全局指令:approve all / retry / status / cancel

Discussions:
┌── #1 path/to/file.py:12-45  (Author: alice@x.com) ────┐
│ 系统:[冲突块详情 + AI 方案 + 思考摘要]                │
│ alice: approve                                         │
│ 系统:✅ Block #1 已 approve(1/3)                     │
└────────────────────────────────────────────────────────┘
┌── #2 path/file.py:60-80  (Author: bob@x.com) ─────────┐
│ 系统:[冲突块详情 + AI 方案]                           │
│ bob: reject 这里应该保留 theirs                       │
│ 系统:❌ Block #2 已 reject(待 /resolve retry 重解)  │
└────────────────────────────────────────────────────────┘
```

### 7.2 指令协议

**单块指令 — 在对应 Discussion 里回复**（用 discussion_id 反查 block）：

| 评论内容                                          | 识别为                     | 权限                   |
| ------------------------------------------------- | -------------------------- | ---------------------- |
| `approve` / `lgtm` / `✅` / `+1`                  | approve 该块               | 块 author 或 `/resolve` 发起人 |
| `reject <reason>` / `❌ <reason>` / `-1 <reason>` | reject 该块（reason 推荐） | 块 author 或 `/resolve` 发起人 |
| _（空内容/无关评论）_                             | 忽略                       | —                      |

**全局指令 — 在 MR 主线评论或 Discussion 里都行：**

| 指令                   | 作用                 | 权限      |
| ---------------------- | -------------------- | --------- |
| `/resolve approve all` | 全部 PENDING 块通过  | `/resolve` 发起人 |
| `/resolve retry`       | 重解所有 REJECTED 块 | `/resolve` 发起人 |
| `/resolve status`      | 回复进度汇总         | 任何人    |
| `/resolve cancel`      | 取消任务（终态）     | `/resolve` 发起人 |

### 7.3 评论位置识别逻辑

```
comment webhook 到达
   │
   ▼
1. 提取 comment.body、comment.discussion_id、comment.author_email
   │
   ▼
2. 匹配顺序:
   a. 先匹配全局指令正则 ^/resolve\s+(approve all|retry|status|cancel)
      → 走全局处理
   b. 否则,若 comment.discussion_id 存在且属于某 conflict_block:
      → 匹配单块指令(approve/lgtm/✅/+1 | reject/❌/-1 [reason])
      → 反查 block_id,执行 approve/reject
   c. 都不匹配 → 忽略
```

### 7.4 权限模型

```
单块 approve/reject:
  comment.author_email == block.author_email ? 允许
  : comment.author_email == task.trigger_author_email ? 允许
  : 拒绝,回复"无权限，仅 block author 或 /resolve 发起人可操作"

全局指令(approve all / retry / cancel):
  comment.author_email == task.trigger_author_email ? 允许
  : 拒绝

/status:对任何人都开放
```

**author email 获取顺序：**

1. `git blame` 的 `author mailmap` 结果（优先 mailmap 映射后）
2. mailmap 缺失 → blame 原始 email
3. 仓库配置 `trust_codebase_account=true` → 用 codebase 账号 email（从 NextCode user API 反查）

### 7.5 approval_service 处理流程

**单块 approve（approve/lgtm/✅/+1 + discussion_id）：**

1. `SELECT block JOIN task WHERE block.discussion_id = ?`
2. 找不到 → 回复“该 Discussion 不属于活跃任务”，结束
3. task.status 不是 AWAITING_APPROVAL / PARTIAL_REJECTED → 回复当前状态，结束
4. 权限校验（author email 比对）
5. `FOR UPDATE` 锁 approval 行
6. 幂等：已 APPROVED → 回复“已 approve”，结束
7. status PENDING → APPROVED,reviewer_email,reviewed_at = now
8. 回复“✅ Block #N 已 approve （X/Y）”
9. 检查整 task：全 APPROVED → 刷新 MR 描述检查项；事务内写入 `ALL_APPROVED` 和发起人飞书通知 outbox

**单块 reject（reject/❌/-1 `<reason>` + discussion_id）：**
1~5. 同上 6. reason 必填，去除非法控制字符且最多 1000 字符 7. 在一个数据库事务内完成 approval `PENDING → REJECTED`、保存 reason、task 首次转为 `PARTIAL_REJECTED` 和写 `BLOCK_REJECTED` 事件 8. 更新 MR 描述 9. 回复“❌ Block #N 已 reject：<reason>”并提示在 MR 主线使用 `/resolve retry`

**approve all：**

1. 权限校验（`/resolve` 发起人）
2. `UPDATE approvals SET status=APPROVED WHERE task_id=? AND status=PENDING`
3. 回复“✅ 全部 N 块已 approve”
4. task.status = ALL_APPROVED + 通知

**retry：**

1. 权限校验
2. 选出所有 approval.status = REJECTED 的 block_ids
3. 无 REJECTED 块 → 回复“无可重解的块”
4. 对这些块走 retry 子流程（见 7.6）

**status：** 回复表格（block idx、file:line、author、status、进度）

**cancel：**

1. task.status = CANCELLED（终态）
2. 回复“任务已取消，分支与 MR 保留，不再响应”
3. 后续该 task 的 webhook 一律忽略

### 7.6 retry 子流程（只重解 REJECTED 块）

1. `/resolve retry` 只允许从 `PARTIAL_REJECTED` 发起。事务内冻结所有 REJECTED block、reason 和其他 locked block，`retry_count + 1`，task 进入 `RETRYING`。
2. worker 以条件更新把 `RETRYING` 原子认领为 `RETRY_PREPARING`。其他 worker 未认领成功就退出，不把任务标为 FAILED。
3. 回读 source、target、resolve 三个分支和原/新 MR。任一冻结基线漂移，不调用 Coco，任务回到 `PARTIAL_REJECTED`。
4. 创建新的 Coco Session，只传 REJECTED block、上次方案和 reject reason；approved/pending ID 作为 `locked_conflict_ids`。Agent 不得再次 merge target，必须在旧 resolve commit 上创建单 parent 普通 commit。
5. 服务端校验 retry schema、rejected ID 精确集合、单 parent、分支 HEAD、文件内容、无冲突 marker、locked 方案仍存在、测试成功及两侧 blame。
6. 用最新数据库结果重写完整 Markdown，上传为 `...-attempt-N.md` 新文件；旧文件保留，task 和 MR 改用新链接。
7. 不创建新 MR 或新顶层 Discussion。在每个被拒块原 Discussion 追加带 attempt marker、原 reject reason 和新方案的回复。
8. 全部外部发布成功后，只把本轮 REJECTED approval 重置为 `PENDING`；原 APPROVED 和原 PENDING 块不变。task 回到 `AWAITING_APPROVAL`。

重解输出 schema 为 `conciliator.resolve.retry.result.v1`。Coco 明确失败且 resolve HEAD 未移动时回到 `PARTIAL_REJECTED`；若发现未知 push、结果校验失败或发布后的状态不一致，则 fail closed 为 `FAILED`，不 force push。

**好处：** 已 approve 块完全不动，Coco 调用次数最小化；历史可追溯；author 只需重审 reject 过的块。

### 7.7 MR 描述模板（系统写入 + 动态更新）

```markdown
## 🤖 自动解冲突报告

- **源分支** → **目标分支**：`feature-x` → `main`
- **resolve 分支**：`resolve/feature-x/<ts>`
- **冲突块数**：3
- **状态**：等待 owner 确认（1/3）
- **任务状态**：`PARTIAL_REJECTED`
- **重解次数**：0
- **确认分布**：1 approved / 1 rejected / 1 pending

### 📄 详细报告

[飞书文档链接](https://xxx)

### ✅ 确认进度

- [x] #1 `path/to/file.py:12-45` — alice@x.com ✓
- [ ] #2 `path/to/file.py:60-80` — bob@x.com ✗ REJECTED：keep target guard
- [ ] #3 `util.py:8-15` — carol@x.com · PENDING

### 📜 指令

- 在对应冲突块 Discussion 里回复 `approve` 或 `reject <reason>`
- MR 主线回复 `/resolve approve all` — 全部同意（限 `/resolve` 发起人）
- MR 主线回复 `/resolve retry` — 重解被拒块
- MR 主线回复 `/resolve status` — 查看进度
```

每次状态变化，resolve_service 调 `nextcode.mrs.update_description` 刷新检查项。

### 7.8 Discussion body 模板

````markdown
**冲突块 #1** · `path/to/file.py:12-45`
**Author**： alice@example.com

**Ours（源分支）**：

```python
def foo():
    return old_logic()
```

**Theirs（目标分支）**：

```python
def foo():
    return new_logic()
```

**🤖 AI 解冲突方案**：

```python
def foo():
    return new_logic()  # 保留目标分支语义
```

**Agent 思考**（摘要）：目标分支已将 `old_logic` 重构为 `new_logic`，源分支对该函数无实质修改，采用 theirs。

---

👤 回复 `approve` 或 `reject <reason>` 即可确认
````

重解不新建 Discussion，而是在原 Thread 追加：

````markdown
<!-- conciliator:task=123:block=C0002:attempt=2 -->
**第 2 轮重解 · 冲突块 #2** · `path/to/file.py:60-80`
**上次 reject 原因**：keep target guard

**🤖 AI 重解方案**：

```
<new resolution>
```

**决策摘要**：按用户反馈保留 target 校验
**额外修复**：(无)

---
👤 本块已重置为 PENDING，请重新回复 `/approval` 或 `reject <reason>`
````

---

## 8. 测试策略

### 8.1 测试分层

| 层级         | 工具                                                 | 范围                                                        | 目标                        |
| ------------ | ---------------------------------------------------- | ----------------------------------------------------------- | --------------------------- |
| **单元测试** | pytest + pytest-mock                                 | `domain/`（状态机、指令解析、blame 处理）、`utils/`、纯函数 | 90%+ 覆盖                   |
| **集成测试** | pytest + respx（HTTP mock）+ testcontainers（MySQL） | `services/`、`repositories/`                                | mock 外部系统，真实验持久化 |
| **契约测试** | schemathesis 或 pytest + OpenAPI                     | webhook payload、NextCode API 响应 schema                   | 防外部变更破坏              |
| **E2E**      | 手动 + 标杆仓库                                      | 完整 pipeline                                               | 上线前/重大变更手动跑       |

### 8.2 必覆盖用例

**单元：**

- 冲突块解析：`<<<<<<<` 嵌套、单文件多块、空 ours/theirs、格式异常
- blame 输出：多行连续 blame、mailmap 映射、author ≠ committer
- 指令解析：`approve`、`LGTM`、`✅`、`+1`、`reject xxx`、`❌ xxx`、`/resolve approve all`、非法指令、大小写、首尾空格
- 状态机迁移：合法/非法迁移
- 权限模型：块 author、`/resolve` 发起人、陌生用户 3 角色 × 每种指令

**集成：**

- webhook 签名校验：正确/错误签名、缺失 header、超时重放
- 幂等：同 MR 重复触发、重复 approve、Coco 事件重复
- resolve_service 流程：每步成功/失败，验证状态推进与回滚
- approval_service:approve/reject/retry/status/cancel 各指令 DB 变化
- 并发：同 task 两 worker 抢占（乐观锁）

**E2E 标杆场景：**

- 仓库 A → `/resolve` → 完整跑通 → owner Discussion 回 `approve` → 全 approve → MR 描述更新 → 通知
- 仓库 A → `/resolve` → 某 owner 回 `reject xxx` → `/resolve retry` → 重解 → approve

---

## 9. 字节内部组件集成

参考 [python.byted.org 基础库索引](https://python.byted.org/library/index.html)。所有内部 SDK 走包管理从字节内部 pip 源安装。

### 集成的 SDK 与替换关系

| 内部 SDK    | 包名                        | 用途                                                                | 替换了原来的                          |
| ----------- | --------------------------- | ------------------------------------------------------------------- | ------------------------------------- |
| MySQL SDK   | `bytedmysql`                | DB 连接，自带服务发现（consul/db_psm）、auth、SQL 注入检测、metrics | `pymysql` + `aiomysql` 自管连接池     |
| Metrics SDK | `bytedance.metrics`         | 上报 counter/meter/histogram/timer 到内部 metrics 平台              | `prometheus-client` 自暴露 `/metrics` |
| KMS         | `bytedkms` （`import kms`） | 从 KMS 拉敏感配置（App Secret、Coco Token、飞书凭证）               | env 明文密钥                          |
| DPS token   | `byteddps`                  | 服务身份认证 token，注入到所有外部 HTTP 调用                        | 自管 Authorization header             |

### MySQL 集成方式（bytedmysql 同步 + 线程池）

bytedmysql 只提供同步接口（基于 `pymysql`）。FastAPI 异步上下文下的集成：

```python
from bytedmysql import sqlalchemy_init
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sqlalchemy_init()  # 注册 pymysql 方言 + 服务发现钩子

engine = create_engine(
    "mysql+pymysql://user:password@/conciliator"
    "?charset=utf8mb4&db_psm=<mysql_psm>",  # host 失效,走 consul
    pool_size=10, max_overflow=20, pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
```

**异步上下文下的访问**：用 `sqlalchemy.ext.asyncio` 不行（bytedmysql 无 async driver）。改用 `starlette.concurrency.run_in_threadpool` 把同步 DB 调用包到线程池：

```python
from starlette.concurrency import run_in_threadpool

async def get_task(task_id: int) -> ResolveTask | None:
    def _query():
        with SessionLocal() as session:
            return session.get(ResolveTask, task_id)
    return await run_in_threadpool(_query)
```

> **Trade-off**：同步 driver 失去原生 async 性能优势，但本项目是 IO 编排服务（LLM 调用、git clone 都是数十秒到分钟级），DB 调用占比极小，线程池开销可忽略。换来的是 bytedmysql 的服务发现/auth/metrics/SQL 注入检测全部免费。

### KMS 集成方式

```python
import kms

kms_client = kms.KMSClient(PSM="<service_psm>")  # 服务自身 PSM
nextcode_app_secret = kms_client.get_config_by_name("nextcode_app_secret")
coco_api_token = kms_client.get_config_by_name("coco_api_token")
lark_app_secret = kms_client.get_config_by_name("lark_app_secret")
```

启动时一次性拉取并缓存（配置变更需重启或主动 refresh）。env 变量里只保留非敏感项 + KMS PSM + 配置名。

### Metrics 集成方式

```python
from bytedance import metrics

metrics.emit_counter("conciliator.tasks", 1, tags={"trigger": "COMMENT", "status": "PENDING"})
metrics.emit_timer("conciliator.task_duration", duration_seconds, tags={"stage": "coco"})
```

不再暴露 `/metrics` endpoint，内部 metrics 平台直接查。本地调试可设 `BYTEDMETRICS_DEBUG=True` 在 stderr 看打点。

### DPS token 注入

所有外部 HTTP 调用（NextCode、Coco、飞书）的 client 在构造时注入 DPS token：

```python
import byteddps

class NextCodeClient:
    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-Codebase-App-Id": self._app_id,
            "X-Codebase-App-Secret": self._secret,
            "X-Bytedance-DPS-Token": byteddps.get_token(),
        }
```

> NextCode/Coco/飞书 是否实际接收 DPS token 待对接确认（spec 第 13 节开放问题）。若不接收，该 header 自动忽略，不影响功能。

---

## 10. 可观测性

### 10.1 Prometheus 指标（`/metrics`）

| 指标                                        | 类型      | 维度                     |
| ------------------------------------------- | --------- | ------------------------ |
| `conciliator_tasks_total`                   | Counter   | `trigger_type`、`status` |
| `conciliator_task_duration_seconds`         | Histogram | `stage`                  |
| `conciliator_coco_task_duration_seconds`    | Histogram | —                        |
| `conciliator_conflict_blocks_total`         | Counter   | `repo_id`                |
| `conciliator_approval_total`                | Counter   | `action`                 |
| `conciliator_external_api_calls_total`      | Counter   | `system`、`outcome`      |
| `conciliator_external_api_duration_seconds` | Histogram | `system`、`operation`    |
| `conciliator_webhook_processed_total`       | Counter   | `event_type`、`outcome`  |
| `conciliator_queue_depth`                   | Gauge     | `status`                 |

### 10.2 结构化日志

JSON 格式，接字节云日志平台。每条日志带 `task_id`、`trace_id`、`repo_id`、`mr_id`。关键节点（状态机迁移、外部 API 调用、指令处理）必打。敏感字段（email、App Secret）脱敏。

### 10.3 任务可追溯（`task_events`）

每次状态迁移、外部调用、指令处理都写 event。出问题按 `task_id` 拉全事件流回放，等于自带 audit log。暴露 `GET /admin/tasks/{id}/events`（限内网 + admin token）。

### 10.4 告警

- 任务失败率 > 10%（5 min 窗口）→ 飞书机器人通知 oncall
- Coco 任务超时率 > 20% → 通知（疑似 Coco 平台问题）
- webhook 处理失败率 > 5% → 通知（疑似签名/解析问题）

---

## 11. 安全

### 11.1 密钥管理

- 所有密钥（App Secret、Coco Token、飞书 App 凭证、MySQL DSN）走字节云 KMS / 应用平台环境变量注入，不落代码/不进 git
- `.env.example` 只列字段名，不留真实值
- 启动时校验必填配置，缺失即拒启动

### 11.2 webhook 签名校验

- 每个 webhook 必校验 `X-Codebase-Signature`（HMAC-SHA256 + App Secret，算 body）
- 校验失败返回 401，记录告警
- 时间戳防重放（超过 5 min 拒绝）

### 11.3 输入校验

- webhook payload 全部走 pydantic 模型校验
- 文件路径、分支名、email 经 SQLAlchemy 参数化（防 SQL 注入）
- 飞书文档内容 XML 转义（防 XSS，飞书本身有过滤，双保险）

### 11.4 权限隔离

- 服务账号（App）只装最小权限仓库（按需申请，走工单）
- MySQL 账号只给 DDL + DML on `conciliator.*`，不给 SUPER/PROCESS
- 内部 admin 接口（`/admin/*`）走单独 token + IP 白名单

### 11.5 数据最小化与保留

- 冲突代码内容进 MySQL + 飞书文档（均内网，合规）
- `task_events` 保留 90 天后归档清理
- 不向 Coco 发送不必要的仓库上下文（只发冲突文件 + resolve_branch）

### 11.6 速率与成本控制

- 单仓库并发任务上限（默认 3），超限排队
- Coco 任务超时硬上限 15 min
- 飞书 API 走 SDK 自带限流 + 重试

---

## 12. 配置项汇总（`config.py`）

```python
class Settings(BaseSettings):
    # ── 服务 ──
    app_name: str = "conciliator"
    env: Literal["dev", "staging", "prod"] = "prod"
    log_level: str = "INFO"

    # ── NextCode(codebase app)──
    nextcode_app_id: str
    nextcode_app_secret: SecretStr
    nextcode_api_base: str = "https://code.byted.org/api"
    nextcode_webhook_path: str = "/hooks/codebase"

    # ── Coco OpenAPI Next ──
    coco_api_base: str
    coco_api_token: SecretStr
    coco_default_agent_id: str
    coco_task_timeout_seconds: int = 900       # 15 min
    coco_event_webhook_path: str = "/hooks/coco"

    # ── 飞书 ──
    lark_app_id: str
    lark_app_secret: SecretStr
    lark_doc_folder_token: str

    # ── MySQL ──
    mysql_dsn: SecretStr
    mysql_pool_size: int = 10
    mysql_pool_max_overflow: int = 20

    # ── 运行时 ──
    worker_concurrency: int = 4
    worker_poll_interval_seconds: int = 5
    task_total_timeout_seconds: int = 1200     # 20 min
    max_retries_per_task: int = 3
    repo_max_concurrent_tasks: int = 3

    # ── Git(本地 fallback)──
    git_clone_base_dir: str = "/tmp/conciliator"
    git_clone_timeout_seconds: int = 120
    git_use_local_clone: bool = true
```

---

## 13. 待确认事项 / 开放问题

（自审后无 TBD，此节为后续实现阶段可能需要再确认的点）

1. **NextCode API 实际能力**：文档列了接口列表但未逐个验证。实现阶段需确认：
   - `list_conflict_files` 是否直接返回冲突内容（决定是否一定需要本地 clone）
   - Discussion 创建是否支持指定行号
   - webhook payload 实际字段命名（签名 header、event_id、discussion_id）
2. **Coco OpenAPI Next 实际接口**：事件流推送格式、diff 事件字段命名需对接时确认
3. **飞书文档 API**：folder_token 权限、批量写入限流
4. **代码仓库访问凭证**：Coco Agent 推送 resolve_branch 需要仓库 write 权限，需确认 codebase app 权限模型是否覆盖
