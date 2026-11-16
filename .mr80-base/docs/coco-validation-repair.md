# Coco 结果校验与一次自动修复

本改动基于 `feat/resolve-version-dedup`。当前没有运行中的任务，因此只支持新结果协议，不兼容旧协议输出。部署时需要已有的 0006 数据结构；本改动不新增数据库列或工单。

## 输入、输出和报告

首次输入为 `conciliator.resolve.input.v2`，结果为 `conciliator.resolve.result.v3`；用户拒绝后的重解输入为 `conciliator.resolve.retry.input.v2`，结果为 `conciliator.resolve.retry.result.v2`。

Coco 始终使用冻结的 source/target SHA，分支后续提交不改变任务版本。首次 resolver 从冻结 source 创建，合入冻结 target，生成 parent 依次为 source、target 的 merge commit。代码方案可以选择一侧或组合修改。只推送 resolver，不修改原分支、MR、评论或飞书文档。用户 `/resolve retry` 仍创建新 Session；校验修复复用本轮 Session。

结果顶层只包含 `schema_version`、`status`、`commit_sha`、`conflicts`、`unresolved_items`。任务编号、业务轮次、resolver 分支由服务端填写；实际 parent 从提交回读并持久化，不让模型抄写固定值。

每块返回 `conflict_id`、`path`、`decision_summary`、`source_attribution`、`target_attribution`、`resolved`。删除 `tests` 和 `introduced_fix`，不再要求 Coco 运行测试。数据库原有 nullable `introduced_fix` 列暂时保留，不再写入或展示。

两侧行号和作者继续由 Coco 获取。Prompt 要求在冻结版本中对真实行范围执行 blame，从涉及的提交中选 committer date 最新者，取其 author；同时间按 SHA 字典序确定。每侧一位作者，空侧作者为 null、行号为 0、commit_sha 为空。服务端保留既有 blame 作者/commit 匹配复核，不新增原文定位或按时间选择作者的算法。文件重命名不在支持范围，Coco 应报告 unresolved_items。

`resolved.text` 允许空字符串表示删除；非空文件中的删除位置使用相邻现存行作为 Discussion 锚点。整文件删除/空文件没有可靠行内锚点，现有发布链路不扩展支持。

报告仅展示任务版本与分支、两侧作者和文件位置、两侧冲突块原文、解决代码和解决决策。每块不展示归因 commit hash，不展示 marker 行、额外修复、完整文件和测试结果；Discussion 和 MR 描述同步精简。任务级冻结 SHA 和实际解决提交仍保留，便于辨认处理版本。

## 校验处理

- 报告位置或文本与实际文件有差异：记录 `COCO_REPORT_WARNING`，继续处理，使用 Coco 返回的解决文本。不修正行号，不重新定位，不生成降级报告。记录失败也不阻断。
- 提交不存在、resolver HEAD 不符、提交关系错误、冲突标记残留、冲突 ID 缺失/重复、JSON 不符合协议、两侧作者无法通过既有 blame 复核：反馈具体错误，在同一 Session 自动修复一次。
- 依赖接口暂时不可用：保留运行态，后续轮询重试，不消耗模型修复次数。

位置问题放行只取消 Conciliator 的阻断校验；NextCode 创建行内 Discussion 时仍可能拒绝无效位置。本改动不增加用户已排除的位置修正或普通评论降级流程。

## 自动修复 Prompt

实现位于 `src/conciliator/integrations/coco/prompts.py` 的 `repair_prompt()`，包含唯一消息标记、具体阻断原因、冻结版本、实际 resolver HEAD、期望 parent、服务端确认的 repair_base_sha、冲突清单和锁定块。

核心要求：继续本 Session 上下文，逐项修复；仅 JSON/作者/SHA 输出有误时只修正输出，不重新提交代码。若必须改代码，只在服务端确认的本任务提交上追加修复，不覆盖他人提交、不 force push、不追随原分支新 HEAD。无法安全修复则返回具体 unresolved_items。

为允许修复已经推送的代码，最终提交可以保留原期望 parent，也可以是服务端验证过的 repair_base_sha 的一个普通子提交。初次 merge commit 仍保留在历史中。repair_base_sha 仅在分支 HEAD 的真实 parent 等于该轮预期时设置，不能采用未知历史。持久化的是修复后真实 parent，而非原来期望的 parent。

## Session、Run 和恢复

同一 Session 每次 UserMessage 产生新 Run。发送后保存新 Run ID 和本轮 UserMessage Event ID，从该事件之后分页读取；按 Run 隔离回复和结束状态。`AfterEventId` 使用 Event ID，不使用 Run ID。Run 状态投影已结束但最终事件未出现时继续等待，避免提前认定 JSON 损坏。

每页保留接口返回的 `NextAfterEventId`，作为下一次请求的 `AfterEventId`；一页不足 200 条仍可能有后续事件，不能据此结束。无后续游标，或空页返回当前游标时结束读取。按 Event ID 去重，游标循环、停滞及超过 100 页时抛出读取错误，不把已读取的部分结果作为完整结果。最终 JSON 和当前 Run 的结束事件都读到后，才进入结果校验。

使用现有 `trigger_payload` JSON 保存服务端控制字段，并在接收 webhook 时清除外部输入的同名控制字段：

- `_coco_round`：Session、当前 Run、本轮消息事件边界。
- `_coco_validation_repair`：一次修复发送状态、错误、标记、开始时间、可信修复基线。
- `_failure_notice`：最终失败原因和评论投递状态。

先通过任务 version CAS 占用修复机会，再发送外部消息。并发轮询只能占用一次。发送超时或发送后进程退出时，恢复流程先按唯一消息标记查询已发送事件，不盲目再次发送。两分钟后仍无法确认发送结果则失败并通知用户。取消导致 CAS 失败时，不覆盖取消状态。

自动修复不增加业务 attempt/retry_count；每个业务 Session 最多一次。用户之后发起 `/resolve retry` 是新的业务轮次，有自己的 Session 和一次修复机会。

## 最终失败通知

自动修复后仍有阻断错误，或修复 Run 失败，任务设为 `FAILED` 并释放 active_key。失败状态与待通知标记通过同一次 CAS 写入。

机器人回复到原 MR 的 `_trigger_discussion_id`，即触发该任务的 `/resolve` 评论线程，包含任务编号、失败原因及重新触发提示，不发到 resolver MR 或别的同版本重复触发线程。

后台轮询会补发未完成通知。评论包含任务唯一标记，发送前回读线程；如果上次写入后响应丢失，查到已有评论便完成投递记录。发送状态预留五分钟恢复窗口，避免另一个 worker 在正常请求期间重复发送。没有触发评论的自动任务记录 no_trigger_thread，不猜测评论目标。

## MR 发布与合并条件

首次发布及重试发布不再等待 `GetMergeRequestMergeability` 返回可合并。CI、审批、仓库规则或目标分支后续变化造成的 merge blocked，不作为本次解冲突任务失败的依据。服务仍验证冻结提交、结果内容、MR 回读及 Discussion，发布完成后进入 `AWAITING_APPROVAL`；是否允许最终合并由 Codebase 的规则决定。

Webhook 自动触发入口仍只在原 MR 的不可合并原因是 `conflict` 时启动解冲突，避免把等待检查或审批的 MR 当作代码冲突任务。

## 验证边界

回归测试覆盖新协议、空侧作者、报告精简、非阻断位置差异、提交关系和冲突标记、同 Session 多轮隔离/分页、一次修复的并发 CAS、发送不确定时恢复、取消保护、失败线程回复及补发防重。

测试使用真实 SQLite 仓储验证状态持久化与并发，外部 Coco/NextCode 调用使用受控替身。它们不等于真实 Coco 多轮执行、NextCode 行内评论和飞书发布的端到端验证；该验证需部署后另行执行。
