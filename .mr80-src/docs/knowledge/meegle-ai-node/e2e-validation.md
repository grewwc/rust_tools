---
title: Meegle AI node end-to-end validation and troubleshooting
description: Validate the Coding workflow in Meegle and diagnose card, webhook, document, Coco, and database failures.
keywords: [E2E, testing, Failed to fetch, by-node, wiki:node:read, Unauthorized]
last_verified: 2026-11-09
---

# 端到端验证与排障

## 测试前检查

- 后端 `/healthz` 和 `/readyz` 均返回 HTTP 200、`status=ok`；数据库不可用时 `/readyz` 返回 HTTP 503。
- 卡片 `API_BASE` 与 AI 节点 webhook 都指向当前后端。
- AI 节点属性 `design_doc` 映射到需求文档对应的工作项字段。
- 测试工作项该字段是有效的飞书文档或知识库链接。
- 飞书应用能访问该文档；知识库链接还需要相应的 wiki 节点读取权限。
- 前置节点已完成，测试空间正在使用新发布的插件/节点版本。

## 最短验证闭环

1. 在测试工作项触发 AI 节点。
2. 卡片进入“输入校验”，后端读取需求文档并评估输入。
3. 输入评估使用 Agent `798403950855499`；结果进入待确认，不自动 Coding。
4. 查看结论、缺失项与风险，提交一次评估反馈，确认新一轮携带最新文档和历史反馈；评估记录可折叠展开。
5. 评估通过后点击“确认评估并开始 Coding”，后端才创建编码分支和 Coco session/run。评估未通过不能开始 Coding。
6. 进入 Review 后检查代码链接、摘要、测试结果和本轮执行记录。
7. 提交一次代码反馈，确认下一轮使用最新代码状态和反馈，历史轮次仍可查看。
8. 确认结果，节点进入完成状态。

联调时观察：

```bash
tail -f /path/to/Conciliator/logs/app.log

curl "http://<reachable-host>:8000/api/aeolus/tasks/by-node?space_id=<space_id>&work_item_id=<work_item_id>&node_id=<node_id>&work_object_id=story"
```

查询响应包含卡片 action token，不要公开完整响应。

## 故障路由

| 现象 | 优先检查 | 下一篇 |
| --- | --- | --- |
| 卡片 `Failed to fetch` | `API_BASE` 可达性、浏览器 Network、CORS、mixed content | [卡片地址](card-webhook-publish.md#区分两个地址) |
| `by-node` 返回 404 | 节点是否真实触发；四个查询标识是否与回调一致 | 本页“日志与状态” |
| 卡片仍是旧版或 mock | 是否发布新 artifact；测试空间是否升级节点版本 | [发布](card-webhook-publish.md#发布到限定测试空间) |
| webhook 验签失败 | 点位 token 与 `MEEGLE_WEBHOOK_TOKEN`（dev）或 KMS 中的 Meegle Webhook Token（staging/prod）；plugin ID | [webhook](card-webhook-publish.md#按需修改-webhook) |
| `wiki:node:read` 或文档读取失败 | 飞书应用权限、知识空间成员权限、文档可见范围 | [开发配置](local-backend.md#开发配置) |
| Coco `Unauthorized` | NextCode App ID/Secret、workspace/agent 所有权、ZTI/App-only 条件 | [开发配置](local-backend.md#开发配置) |
| `Environment preparation failed` | 对应 Coco attempt/session 事件和 Agent 环境安装步骤 | 本页“日志与状态” |
| MySQL 表不存在 | DSN、目标 database、Alembic `upgrade head` | [数据库与启动](local-backend.md#数据库与启动) |
| 进程存活但服务不可用 | `/readyz`、`logs/app.log`、端口占用、外部依赖 | [数据库与启动](local-backend.md#数据库与启动) |

## 日志与状态

获取任务 ID：

- 飞书全流程消息、Devbox/PPE 部署结果通知底部显示完整“任务 ID”，可选中复制。
- Meego 卡片在交互记录之后显示完整 ID，点击“复制”；浏览器不允许复制时可手动选中复制。
- 任务 ID 用于串联同一任务的评估、Coding 和部署日志，与工作项 ID、部署 ID 不同。
- 网页卡片需要发布包含此改动的新版本；已发送的旧飞书消息不会自动补上 ID。

排障顺序：

1. 浏览器 Network 确认实际 URL、状态码和 CORS。
2. `tail -f logs/app.log` 按 `task_id` 找第一处业务错误。
3. 查询 `by-node` 确认任务状态、轮次和 `last_error`。
4. Coco 失败时，用当前 attempt 的 session/run 定位远端事件。
5. 不要把 HTTP 200 直接视为业务成功；部分远端 Action API 会在响应体的 `ResponseMetadata.Error` 中返回鉴权错误。

后端新增的链路日志使用 `workflow=meegle`，统一采用单行结构化字段。按 `task_id`
查询可以串起 webhook、需求及引用文档读取、输入评估、Coding、MR、部署、飞书通知和节点完成；
授权尚未关联任务时，使用 `user_key` 和 `request_id`。

| 字段 / 事件 | 排障用途 |
| --- | --- |
| `task_id`、`execution_id`、`step`、`step_id` | 同一任务、一次处理及其嵌套步骤；开始/结束事件携带耗时 `duration_ms` |
| `meegle_task_state_changed` | 数据库提交后的 `previous_state` → `state`；拒绝的操作和失败提交不会报成功 |
| `evaluation_round_no`、`round_no`、`attempt_no`、`session_id`、`run_id` | 关联评估历史、编码轮次、替换执行和 Coco 平台记录 |
| `meegle_webhook_accepted`、`meegle_card_action_*` | 区分新任务、重复 webhook、卡片过期、操作接受和拒绝 |
| `deployment_id`、`environment_id`、`worker_id`、`generation` | 串起 PPE、Devbox、Worker 回调和 MAAT 入口及释放 |
| `meegle_ppe_state_changed`、`meegle_maat_state_changed` | 查看部署阶段、外部资源标识和旧错误清除；`unknown` 需核对原写入结果 |
| `meegle_step_failed`、`meegle_external_failed` / `finished`、`meegle_api_finished` | 错误类型、来源位置、HTTP 状态、平台业务码及 `external_request_id` |
| `meegle_devbox_stopped` | `port_released` 和 `cleanup_state` 表示 Worker 回执确认的释放结果 |

默认 INFO 保留关键业务步骤、已提交状态变化和写操作；正常卡片查询、Worker 心跳/领取、
PPE/MAAT 轮询和外部读请求使用 DEBUG。错误使用 WARNING/ERROR。步骤 `finished`
表示调用已返回，最终业务结果需查看任务 `state`、评估 `valid` 或 Coding `result_status`；
一个返回了校验失败结果的步骤仍可以正常结束。

可在日志平台先过滤 `workflow=meegle` 和 `task_id`，再按 `deployment_id` 或 `run_id`
缩小范围。本地 JSON 日志例如：

```bash
tail -f logs/app.log | jq -Rrc --arg task "<task_id>" \
  'fromjson? | select(.workflow == "meegle" and .task_id == $task)'
```

新事件不记录请求体、响应体、URL 查询参数、文档正文、提示词、反馈、卡片签名或 Token；
异常只记录类型、业务码及最后一个栈帧的位置，不输出原异常文本或完整栈。
日志仅覆盖到达后端的请求；浏览器网络/CORS 故障仍需从 Network 检查。

## 变更后的回归

后端：

```bash
cd /path/to/Conciliator
uv run pytest tests/unit/test_aeolus_api.py tests/unit/test_aeolus_service.py
uv run ruff check src/conciliator/api/aeolus.py src/conciliator/services/aeolus_service.py
```

卡片：

```bash
cd /path/to/Conciliator/frontend
npm test
npm run build:dev
```

提交前确认 `git status --short` 不包含 `.env`、日志、`.lpm-cache`、构建产物或凭据文件。

## Coding Agent 与阻塞结果

Coding 根据已确认的 `code_context.repository` 选择 Agent，配置以 Settings 为准：

| 仓库 | 配置 |
| --- | --- |
| `dp/aeolus` 及未单独配置的仓库 | `AEOLUS_CODING_AGENT_ID` |
| `dp/VizQueryService` | `AEOLUS_CODING_AGENTS` 中的默认条目 |
| 其他仓库 | 通过 `AEOLUS_CODING_AGENTS` 的仓库路径到 Agent ID 映射扩展；环境覆盖会替换整份映射 |

首次编码和会话失效后的重建使用该映射，已有 Session/Run 恢复不切换 Agent。
运行环境和依赖须在对应 Agent 侧准备，输入评估仍使用独立 Agent。

`coding_blocked` 保留部分提交、阻塞项及测试结果，不自动重试、完成节点或提供新部署；
可提交处理意见继续编码，已有环境可独立释放。Review MR 仅展示已核验的提交或代码链接。
未知失败码作为说明保留；不要将所有失败归因于授权，也不要自动重分类历史任务。
