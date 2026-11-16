# 评估后仓库核验失败及重试问题

- 记录日期：2026-10-22。
- 状态：暂缓处理；未修改代码、权限或任务状态。

## 现象与证据

Agent 返回需求评估通过后，后端调用 NextCode `GetRepository` 返回
HTTP 403、`AccessDenied: no permission to perform this action`。
界面只提供重新评估入口，无法单独重试仓库核验。

- 发生时间：2026-10-22 10:25:29（北京时间）。
- request_id：`dbcn86chmjlkbe23oh60-20261022102529B8BCF4B6455E194AC867`。
- 本地日志：检查时位于 `logs/app.log:307`，事件为 `aeolus_task_failed`；日志可能随应用重启被覆盖。
- 实际调用栈：`_run_validation → _accept_evaluation_result → resolve_context → get_repository → GetRepository`。

已确认错误发生在 Agent 返回评估结果后的仓库访问及代码基线核验阶段，
尚未进入创建编码分支或 Coding。403 表示本次应用调用身份的仓库读取请求被拒绝；
具体缺少哪层授权尚未核实，不能据此判断用户个人没有仓库权限。

## 当前实现问题

`_accept_evaluation_result` 在持久化评估结果前调用 `resolve_context`。
仓库核验异常被统一归为评估失败，异常处理清空当前 `evaluation`，
因此权限恢复后只能重新运行评估，无法复用本次 Agent 已返回的结果。

## 后续处理方向（待实施）

1. 核实本次 NextCode 应用调用身份为何无法读取目标仓库。
2. 保留已返回的需求评估结果，将仓库核验失败与需求评估失败区分展示。
3. 提供“重试仓库核验”，成功固定基线后进入评估待确认，不重复调用评估 Agent。
4. 保留确认后分支准备失败时已有的“重试编码”路径。

验收应覆盖：Agent 评估通过但 `GetRepository` 返回 403 时保留评估结果；
修复权限后仅重试仓库核验；核验成功前不创建分支、不启动 Coding，成功后仍等待用户确认。
