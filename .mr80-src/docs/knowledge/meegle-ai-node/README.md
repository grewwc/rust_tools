---
title: Meegle AI node Coding knowledge index
description: Routing index for local backend deployment, card development, webhook configuration, release, and end-to-end validation.
keywords:
  - Meegle
  - Meego
  - 飞书项目
  - AI 节点
  - aeolus 编码助手
  - coding card
  - webhook
  - API_BASE
  - LPM
  - Coco Agent
last_verified: 2026-11-06
---

# Meegle AI 节点 Coding 知识库

本知识库用于在新的开发机或 DevBox 上拉起 Conciliator 后端、修改并发布 `aeolus 编码助手` 卡片，以及在 Meegle 测试空间验证完整 Coding 流程。

## Agent 读取路由

先识别任务，再读取对应页面；不要默认加载全部页面。

| 当前任务或检索词 | 必读页面 |
| --- | --- |
| Devbox worker、动态端口、MAAT 入口、部署释放与恢复 | [Devbox 部署](devbox-deployment.md)、[worker 安装](../../../deployment-worker/README.md) |
| PPE/Devbox 运行时推荐、Python 2/3 识别、手动选择运行配置 | [运行时选择](runtime-selection.md) |
| Aeolus PPE、SCM/ICM 构建、ENV/TCE 部署、PPE 应用鉴权 | [PPE 平台契约与验证结论](ppe-environment-conclusions.md)、[实现与运行约定](ppe-deployment-implementation.md) |
| Python 2.7 PPE、py27、旧版 Aeolus PPE、release-aeolus-2.104 | [Python 2.7 PPE](ppe-py27.md)、[实现与运行约定](ppe-deployment-implementation.md) |
| VQS PPE、VizQueryService、完整 Aeolus + VQS 环境、AEOLUS_SERVICE_DISCOVERY、VQS SCM 流水线 | [VQS PPE 运行约定](vqs-ppe.md)、[实现与运行约定](ppe-deployment-implementation.md) |
| 安装 LPM、Meegle skill、MCP 文档服务、登录 | [工具与知识服务](tooling.md) |
| `.env`、MySQL、后端启动、`dev_run.sh`、KMS/ZTI、飞书长连接 | [本地后端](local-backend.md) |
| TCE、生产配置、KMS ACL、生产数据库迁移 | [生产部署](../../DEPLOYMENT.md)、[配置清单](../../production-configuration-checklist.md) |
| `API_BASE`、卡片修改、webhook、点位配置、build/release/publish | [卡片、webhook 与发布](card-webhook-publish.md) |
| 正式 AI 节点、Aeolus 研发助手、测试与生产插件隔离 | [正式 AI 节点](production-node.md) |
| 飞书消息 / Meego 卡片同步、状态或交互功能调整 | [双入口同步规则](card-layout-verification.md#双入口同步规则)、[飞书全流程消息约定](feishu-card-visual-review.md) |
| 卡片布局、配色、可读性、视觉评审 | [卡片可读性约定](card-visual-review.md) |
| 卡片全屏滚动、响应式布局、飞书输入上限、表单层级 | [布局与平台约束](card-layout-verification.md) |
| 飞书消息展示、折叠详情、全流程消息布局 | [飞书全流程消息约定](feishu-card-visual-review.md) |
| PPE 卡片部署、业务路由、第二轮升级、验证确认与释放验收 | [PPE 产品端到端验收](ppe-product-e2e-validation.md) |
| 平台测试、流程验收、Failed to fetch、401/404、权限错误 | [端到端验证与排障](e2e-validation.md) |

跨层故障按调用方向读取：卡片请求失败先读“端到端验证”，确认地址或发布问题后再读“卡片、webhook 与发布”；后端启动失败先读“本地后端”。

## 稳定事实

```text
Meegle AI 节点 ──webhook──> Conciliator 后端
卡片（浏览器） ──API_BASE──> Conciliator 后端
Conciliator ──> Meegle OpenAPI / 飞书文档 / Coco Agent / Codebase / MySQL
```

- 输入评估从需求文档提取 `code_context`（仓库、base 分支/commit、MR 目标分支）；未指定时默认 `dp/aeolus`、`master` 最新提交，MR 目标默认同 base 分支（显式指定优先）。后端核验后把完整 SHA 固定在评估 JSON 与需求共识中，Coding/重试复用。
- 输入评估使用 Agent `798403950855499`，通过后需要用户确认才开始 Coding；评估结果和反馈保存在独立的可折叠历史记录中。
- 同一任务的后续输入评估复用评估 Session；每轮 JSON 保存 Session/Run/Event 游标（含格式修复），与 Coding 共用租约和重启恢复扫描。恢复只收取原 Run；旧任务缺少游标时转为可重试失败，不自动重复发送。
- 卡片 API origin 与 AI 节点 webhook 是两项独立配置，修改一项不会自动修改另一项。
- 共享状态、展示信息和交互功能调整必须同步检查 Meego 网页卡片与飞书消息卡片，分别修改和验证两处；具体范围与完成条件见[双入口同步规则](card-layout-verification.md#双入口同步规则)。
- `ENV=dev` 从 `.env` 或进程环境变量读取凭据；staging/prod 从 KMS 读取。测试插件 ID 在 `.env` 显式指定。
- 后端入口为 `/hooks/meegle/ai-node`，卡片查询入口为 `/api/aeolus/tasks/by-node`。
- 点位配置是全量替换；更新前必须拉取远端完整基线并检查 diff。
- 前端发布是 `release` 后再用真实 artifact version 执行 `publish`。

## 权威来源

知识条目与实现冲突时，按以下顺序核实：

1. 当前源代码、迁移、测试和 `run/dev_run.sh`。
2. 当前 `lpm --help`、`lpm <command> --help` 与 `lpm ai docs` 返回的官方资料。
3. 本知识库。
4. `docs/superpowers/` 下的历史计划和设计记录。

关键实现位置：

| 内容 | 路径 |
| --- | --- |
| 后端配置 | `src/conciliator/config.py`、`src/conciliator/utils/secrets.py` |
| 后端启动 | `run/dev_run.sh` |
| AI 节点 API | `src/conciliator/api/aeolus.py` |
| Coding 流程 | `src/conciliator/services/aeolus_service.py` |
| 卡片入口 | `frontend/src/features/ai_node_web_aeolus_card/App.tsx` |
| 飞书消息生成与回调 | `src/conciliator/services/aeolus_lark_service.py` |
| 插件工程 | `frontend/plugin.config.json` |

## 安全边界

- 不读取、不打印、不提交 `.env`、Developer Token、插件 Secret、webhook token 或 LPM 本地凭据。
- 不从 `plugin.config.json` 提取 Secret 写入前端或文档。
- 任务查询响应含卡片 action token，不把完整响应粘贴到公开聊天、工单或日志。
- `local-config set`、`update --source-type=local` 和 `publish` 会影响远端，执行前必须获得用户明确确认。
