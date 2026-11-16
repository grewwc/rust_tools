# Conciliator

> 自动解 MR 冲突的 Codebase App。当 MR 存在冲突时，创建 resolve 分支、委托 Coco Agent 解冲突、把方案与 author 邮箱汇总到飞书文档，在新 MR 上发起逐块确认流程，全部 approve 后通知合入。

## ✨ 目标功能

1. **创建 resolve 分支** — 基于冻结的 MR 源分支 commit
2. **自动解冲突** — 委托 [Coco OpenAPI Next](https://code.byted.org/managed-agents) 在云端沙箱把冻结的目标分支 commit 合入 resolve 分支，解冲突并 commit + push
3. **飞书文档汇总** — 把所有冲突块（source/target/AI 方案）、author email、Agent 决策摘要写入飞书文档
4. **resolve MR** — 以 resolve 分支为源、原目标分支为目标创建新 MR，描述里带飞书文档链接
5. **逐块确认** — 每个冲突块在 MR 里发起一条 Discussion，owner 回复 `approve` / `reject <reason>` 确认
6. **飞书私信通知** — 进入待审批状态时通知 `/resolve` 发起人查看 resolve MR；全部 approve 后再次通知发起人确认是否合入（**不自动合入**）

## 🏗 架构

```
NextCode MR ──webhook──▶ Conciliator（FastAPI）
                          │
                          ├─ 1. NextCode API：创建 resolve 分支
                          ├─ 2. 解析冲突块 + git blame 定位 author
                          ├─ 3. Coco OpenAPI：提交解冲突任务（Agent 在沙箱改码 + push）
                          ├─ 4. Coco Event：读取结构化任务结果 → 飞书文档
                          ├─ 5. NextCode API：创建 MR + 每个冲突块发起 Discussion
                          │
                          └─ 审批闭环：Discussion approve/reject → 状态机推进 → 通知合入
```

**角色边界**：确定性逻辑（分支编排、blame、文档、审批流）在服务，AI 能力（代码生成）全部委托给 Coco Agent。

## 📁 项目结构

仓库包含三个独立工程：

| 工程 | 安装、开发与部署说明 |
| --- | --- |
| Conciliator 服务 | 根目录 `pyproject.toml`、`src/conciliator/`，见[本地后端](docs/knowledge/meegle-ai-node/local-backend.md) |
| Meegle 前端 | [frontend/README.md](frontend/README.md)，独立 npm/LPM 工程 |
| Devbox deployment worker | [deployment-worker/README.md](deployment-worker/README.md)，独立 Python 包与 CLI |

前端和 worker 通过现有 HTTP API 与服务通信。迁移目录不改变服务接口、数据库或后台任务启动方式。

```
src/conciliator/
├── api/                 # FastAPI 路由（webhook、health）
├── domain/              # 领域模型（状态机、指令解析，零 IO）
├── services/            # 应用服务（resolve / approval / notification）
├── integrations/        # 外部系统适配层
│   ├── nextcode/        # NextCode OpenAPI
│   ├── coco/            # Coco OpenAPI Next
│   ├── lark/            # 飞书 OpenAPI
│   └── git/             # 本地 git 操作（clone、blame、冲突解析）
├── repositories/        # MySQL 持久层（bytedmysql 同步 + threadpool async 包装）
├── workers/             # 服务内部任务调度（Coco poller + stuck recovery）
└── utils/               # logging、retry、metrics、secrets（KMS）
migrations/              # Alembic 迁移
tests/                   # unit + integration
docs/                    # design spec + implementation plan + deployment
```

## 🛠 技术栈

| 层 | 选型 |
|----|------|
| 语言 / 框架 | Python 3.11+、FastAPI、pydantic v2 |
| ORM / 迁移 | SQLAlchemy 2.x（同步）、Alembic |
| HTTP | httpx（async） |
| Git | GitPython |
| DB | 字节云 RDS（MySQL 8.0） |
| 内部 SDK | `bytedmysql`（DB + 服务发现）、`bytedance.metrics`（打点）、`bytedkmsv2`（密钥）、`byteddps`（服务身份） |
| LLM | Coco OpenAPI Next（全部委托，不自管 LLM） |
| 包管理 | uv |

## 🚀 快速开始（开发）

按[本地后端](docs/knowledge/meegle-ai-node/local-backend.md)安装依赖、填写 `.env`、迁移开发库并启动服务。
需要 Python 3.11+、uv、字节内网和 MySQL；完整联调使用 doas 注入 ZTI 身份。
业务部署 worker 单独安装，解释器和依赖路径统一保存在个人 worker JSON 的 runtime profile 中；接入示例使用 Aeolus Python 3.10、VQS Python 3.7，见[worker README](deployment-worker/README.md)。

## 🧪 测试

```bash
# 全部测试（集成测试需要 Docker 起 testcontainer MySQL）
uv run pytest

# 仅单元测试（不需要 Docker）
uv run pytest tests/unit/

# 单个模块
uv run pytest tests/unit/test_command_parsing.py -v

# 覆盖率
uv run pytest --cov=src/conciliator --cov-report=term-missing
```

## 🔧 代码质量

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

## 📜 用户指令协议

冲突块审批评论（含重解回复）在发送前检查完整 Markdown 的长度。超过 6000 字符时，改为显示冲突块编号、文件位置、双方作者、飞书报告链接和审批指令，完整代码与决策保留在本轮报告中。

MR Discussion 里 owner 与系统交互：

| 评论 | 识别为 | 权限 |
|------|--------|------|
| `approve` / `lgtm` / `✅` / `+1` | 通过该块 | 块 author 或 `/resolve` 发起人 |
| `reject <reason>` / `❌ <reason>` / `-1 <reason>` | 拒绝该块 | 块 author 或 `/resolve` 发起人 |

MR 主线全局指令：

| 指令 | 作用 | 权限 |
|------|------|------|
| `/resolve approve all` | 全部 PENDING 块通过 | `/resolve` 发起人 |
| `/resolve retry` | 重解 REJECTED 块 | `/resolve` 发起人 |
| `/resolve status` | 原 MR 返回任务状态；resolve MR 返回冲突块审批进度 | 任何人 |
| `/resolve cancel` | 取消任务 | `/resolve` 发起人 |

`/resolve status` 回复在本次查询评论的讨论串下。原 MR 有多个活动任务时，会列出各任务的状态及处理版本；没有活动任务时返回最近一次任务的最终状态，没有任务记录时提示使用 `/resolve` 发起。resolve MR 可查询活动或已结束任务的审批明细。状态查询不触发任务执行，也不要求查询者是任务发起人或超级管理员。

### 纯空行冲突免人工审批

只有 source、target 两侧都仅含空格、制表符、CR、LF（或零行）的冲突块才免审批；代码删除、注释和缩进变化仍需人工审批。分类由服务端根据冻结输入决定，记录为 `NOT_REQUIRED`，保留冲突 ID，不创建审批 Discussion，也不计入人工审批总数。

Coco 仍处理全部冲突。纯空行块必须原样保留 source 侧空白，服务端结合冻结文件上下文和最终提交原文进行严格核对；校验不通过会进入现有的一次自动修复流程。重解时这些块保持锁定。MR 描述、报告和 `/resolve status` 会分别展示人工审批进度及免审批块数。

全部冲突都免审批时，在 resolve MR 创建、回读验证后进入 `ALL_APPROVED`，私信发起人“纯空行冲突已处理，无需人工审批”，不发送等待审批通知，也不自动合并 MR。完成状态与通知入队保持事务一致，Worker 可恢复发布后中断的收尾操作。

无需数据库迁移。所有实例更新完成后再发起新任务；已发布任务不会自动重新分类，已有评论不会被删除。

### 待审批飞书通知

任务首次解冲突完成，以及每轮重解完成后，进入 `AWAITING_APPROVAL` 时，向原 `/resolve` 评论作者发送飞书私信，包含任务编号、处理轮次和 resolve MR 链接。自动触发、缺少发起人邮箱或 resolve MR 链接的任务不发送这条通知，不回退到 MR 作者。

状态变更和通知入队在同一数据库事务中完成。复用 `lark_notification_outbox` 和现有发送 worker，按任务及重解轮次去重；投递失败沿用现有重试策略，不改变任务审批状态。通知类型为 `AWAITING_APPROVAL:<retry_count>`，首次处理为 `0`，每轮使用固定发送 UUID。原有 `ALL_APPROVED` 通知独立保留。

本功能不需要数据库迁移。更新服务后，新进入待审批状态的任务生效；已经处于该状态的任务不补发通知。

### Aeolus 飞书用户授权

- 配置 `LARK_OAUTH_REDIRECT_URI`（如 `https://<后端域名>/oauth/lark/callback`），并在飞书应用中注册同一回调地址后，主文档、引用文档预检查和 agent 文档代理先尝试应用身份，读取失败后再尝试操作者身份；任一身份可读即可继续，应用可读时无需用户 OAuth。未配置时保持原有文档读取路径，启动日志会告警，且不执行引用文档预检查。
- 应用没有文档访问权且用户缺少有效授权时才引导 OAuth，也可给应用添加文档阅读权限后重试。已有用户授权不等于拥有该文档的访问权；双方 ACL 均不足时应申请文档权限，应用 API 权限缺失或临时故障不应提示反复授权。`lark_authorized` 仅表示用户 token 状态，不能单独用来判定是否需要授权；卡片应依据当前评估/编码的授权失败标志。此策略允许任务使用应用可读的文档，但文档代理仍校验任务、操作者绑定和引用文档白名单。
- 授权回调会通过 Meegle 用户目录取得邮箱，再查询同一飞书应用下的 `open_id`，与实际授权者比对；需确保应用具备相应目录查询权限。无法核验或身份不匹配时拒绝保存 token，不回退为信任 URL 中的 `user_key`。授权需在发起流程的同一浏览器完成。
- 上线前执行 `0015_lark_user_tokens` 及 `0016_lark_token_user_unique` 迁移。后者仅增加 `user_key` 唯一约束；若已有重复绑定，迁移会失败，应先由管理员核验处理，不会自动删除或任选一条。SQL 部署参见 [DDL 说明](ddl/README.md)。
- 重新授权成功后仅自动恢复因授权中断的**评估**；编码失败仍需显式重试。编码以已确认共识为准，不注入主文档正文；共识缺少实现细节时应先重新确认。
- `offline_access` 的实际 refresh token 下发、刷新及轮换行为仍需在目标租户验证；模拟测试不替代真实授权验收。若未下发 refresh token，access token 过期后需要重新授权。并发刷新合并仅限单个服务实例。
- 当前 token 仍明文存储于数据库，部署时应限制表访问权限；应用层加密需另行设计密钥管理及存量迁移。文档 capability 暂与 webhook 共用签名密钥，轮换会使在途能力凭证失效，应避开活跃任务并重试受影响任务。

## 📚 文档

- [设计 spec](docs/superpowers/specs/2026-07-21-conciliator-design.md) — 架构、数据模型、状态机、确认机制
- [NextCode 契约验证](docs/superpowers/plans/2026-08-27-nextcode-openapi-webhook-contract-validation-revised.md) — 已验证的 NextCode 能力、契约和边界
- [Coco 联调与 `/resolve` 收口计划](docs/superpowers/plans/2026-08-28-coco-resolve-scope-and-integration-plan.md) — Coco 任务范围、OpenAPI 契约、门禁和验收步骤
- [`/resolve` 逐块确认验收计划](docs/superpowers/plans/2026-09-03-resolve-all-blocks-approval-e2e-validation.md) — 两文件四冲突块、Markdown 报告、新 MR、Discussion 和逐块 `/approval` 验收步骤
- [本地后端](docs/knowledge/meegle-ai-node/local-backend.md) — 依赖、开发配置、MySQL、ZTI、启动与检查
- [Devbox 部署](docs/knowledge/meegle-ai-node/devbox-deployment.md) — worker 接入、部署、释放与恢复
- [生产部署](docs/DEPLOYMENT.md)及[生产配置清单](docs/production-configuration-checklist.md) — TCE、KMS、DDL 与上线门槛
- [Meegle AI 节点 Coding 知识库](docs/knowledge/meegle-ai-node/README.md) — Agent 路由、本地后端、卡片 API/webhook、LPM skills、发布与测试空间验收

## 📈 可观测性

| 维度 | 接入点 |
|------|--------|
| 日志 | structlog JSON，应用日志自动附带 `trace_id`（trace scope 内）和业务 `task_id`；生产经 `bytedlogger` 上报 Argos StreamLog（华北） |
| Metrics | `bytedance.metrics` 上报：`conciliator.tasks`、`conciliator.task_duration`、`conciliator.approval`、`conciliator.queue_depth` 等 |
| 任务审计 | `task_events` 表记录每次状态迁移 + 外部调用 + 指令处理 |
| 健康检查 | `GET /healthz`（liveness）、`GET /readyz`（含 DB ping） |

## 🗺 路线图

### 已完成

- NextCode OpenAPI 与 webhook 契约验证，包含冻结 SHA、分支方向和去重语义
- Coco OpenAPI Next 基础客户端、Codebase App 鉴权头、Session Event 轮询与 SSE 解析
- Coco 用户身份基本任务冒烟验证
- MR Version / source SHA / target SHA 冻结、稳定 conflict ID 和确定性 resolver 分支
- Coco 结构化输入输出、NextCode 分支/commit/marker/blame 回读门禁
- `/resolve` 状态机、Session/Run/result 持久化和 DDL 骨架

### 待办

- 完成 Coco App-only 接入探针，并落定、验证生产身份方案
- 在 TCE 实测并固化 Coco Agent Instruction/Skill 与结构化结果契约
- 验证 Coco 在沙箱创建 merge commit、push resolve 分支并回传可核验结果
- 用隔离冲突 MR 验证 source/target `blame` 归因和结构冲突降级规则
- 在 TCE 验证 Markdown 报告、resolve MR、逐块 Discussion 和逐块 `/approval` 闭环
- 验证 reject/retry、approve all、status 和 cancel 链路
- 发布前完成 `/resolve` 端到端验收

> 正式后端始终初始化全部业务能力。完整上线配置与尚需完成的真实环境验收见 [生产配置清单](docs/production-configuration-checklist.md)。

## 📄 License

Internal use only.
