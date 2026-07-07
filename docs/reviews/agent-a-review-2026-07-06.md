# Agent `a` 深度评审报告

> **评审日期**: 2026-07-06  
> **评审标准**: 现代化通用 Agent（对标 Claude Code / Cursor / Aider）  
> **评审方法**: 16 个并行子 agent 分模块审查 + 3 轮源码交叉验证  
> **验证状态**: 所有 P0/P1 发现均经源码逐行确认；2 项发现经验证后**推翻**

---

## 目录

- [1. 执行摘要](#1-执行摘要)
- [2. Bug 清单](#2-bug-清单)
  - [2.1 P0 — 安全/数据丢失](#21-p0--安全数据丢失)
  - [2.2 P1 — 功能缺陷](#22-p1--功能缺陷)
  - [2.3 P2 — 次要缺陷](#23-p2--次要缺陷)
- [3. 效果与 Prompt 优化](#3-效果与-prompt-优化)
- [4. 架构优化建议](#4-架构优化建议)
- [5. 验证记录](#5-验证记录)
- [6. 优先级排序](#6-优先级排序)

---

## 1. 执行摘要

Agent `a` 是一个功能丰富的 LLM agent runtime，具备进程调度、工具注册、MCP 集成、技能路由、思维子系统（ToT/验证/目标分解）、反思进化等能力。整体架构成熟度较高，但在**安全性**、**流处理健壮性**、**思维子系统有效性**、**prompt 工程**四个方面存在显著差距。

**关键数据**：
- 确认 P0 安全漏洞：3 个（SSRF × 2 + reasoning_content 丢失）
- 确认 P1 功能缺陷：22 个
- 确认 P2 次要缺陷：20+ 个
- 推翻发现：2 个（Memory GC 腐败、Patch 半写入）
- **重大发现**：思维子系统（Verification/ToT/Goal Decomposition）全部是**死代码**——每轮注入 prompt 但从不解析 LLM 回复，永久卡在初始状态

**最紧急的五件事**：
1. 修复 `web_fetch` SSRF（P0 × 2，可访问内网/云元数据）
2. 修复流处理 `reasoning_content` 丢失 + SSE 末事件丢失（P0/P1）
3. 修复命令沙箱的多重绕过路径（P1 × 4）
4. **移除或修复思维子系统的死循环**（P1 × 3，每轮浪费 token 且永不进展）
5. 修复 `task_tools` 的 channel/futex 泄漏和 `wait_policy=all` 不生效（P1 × 3）

---

## 2. Bug 清单

### 2.1 P0 — 安全/数据丢失

#### P0-1: `web_fetch` SSRF — HTTP 重定向绕过

**文件**: `src/bin/ai/tools/web_tools.rs:413-418`  
**验证**: ✅ 已确认

`reqwest::blocking::Client::builder()` 未设置 `.redirect(Policy::none())`，reqwest 默认跟随最多 10 次重定向。host/IP 黑名单（398-411 行）只校验**初始 URL**。

```
攻击者 URL: http://attacker.example/
→ 302 Location: http://169.254.169.254/latest/meta-data/
→ 静默绕过所有 SSRF 防护，读取云元数据
```

**修复**: 设置 `.redirect(Policy::none())`，手动处理 `Location` 头并重新校验每个跳转目标。

---

#### P0-2: `web_fetch` SSRF — DNS 重绑定绕过

**文件**: `src/bin/ai/tools/web_tools.rs:401-411`  
**验证**: ✅ 已确认

IP 检查仅在 `host.parse::<IpAddr>()` 成功时触发（即 URL 中直接包含字面 IP）。域名主机完全跳过检查。此外：
- `::ffff:127.0.0.1`（IPv4-mapped IPv6）的 `is_loopback()` 返回 false
- `0.0.0.0` 的 `is_unspecified()` 未被检查

**修复**: 解析域名并检查所有 A/AAAA 记录；补全 IPv6-mapped 和 unspecified 检查。

---

#### P0-3: `extract.rs` — `reasoning_content` 在同 chunk 中被丢弃

**文件**: `src/bin/ai/stream/extract.rs:48, 109`  
**验证**: ✅ 已确认

```rust
if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
    // 仅当 content 为空时才处理 reasoning
}
```

当 provider（DeepSeek、Qwen、OpenRouter 多模型）在同一 SSE chunk 中同时发送 `content` 和 `reasoning_content` 时，reasoning 分支被跳过，**推理内容静默丢失**。持久化的 `reasoning_text` 也不完整，影响下一轮请求的 thinking 上下文。

**修复**: 移除 `content.is_empty()` 前置条件，在同 chunk 中先处理 `reasoning_content` 再处理 `content`。

---

### 2.2 P1 — 功能缺陷

#### P1-1: 命令沙箱 — 绝对路径绕过程序黑名单

**文件**: `src/bin/ai/tools/service/command.rs:947, 1081`  
**验证**: ✅ 已确认

`program` 取自命令的第一个 token（已小写但**未做 basename 归一化**）。`rm -rf x` 被拦截，但 `/bin/rm -rf x` 不被拦截。

**修复**: 对 `program` 取 basename 并解析符号链接后再比较。

---

#### P1-2: 命令沙箱 — 子 shell / 花括号分组绕过

**文件**: `src/bin/ai/tools/service/command.rs:141`  
**验证**: ✅ 已确认

`split_unquoted_segments` 不分割 `(` `)` `{` `}`。`(rm -rf /tmp)` 作为单段通过验证。

**修复**: 拒绝未引用的 `(` / `)` / `{` / `}`，或递归验证内部命令。

---

#### P1-3: 命令沙箱 — 解释器 `-c`/`-e` 标志未覆盖

**文件**: `src/bin/ai/tools/service/command.rs:722-741`  
**验证**: ✅ 已确认

仅检查 `bash | sh | zsh | ksh | dash`。`python3 -c "import os; os.system('rm -rf x')"` 完全绕过。

**修复**: 扩展内联代码检测至 `python*`、`perl*`、`ruby*`、`node*`、`php*`、`awk` 等。

---

#### P1-4: 命令沙箱 — `source` / `.` / `nc` 等未拦截

**文件**: `src/bin/ai/tools/service/command.rs:1071-1079`  
**验证**: ✅ 已确认

`source` 和 `.` 在代码中被注释掉。`nc`/`ncat`/`netcat`/`telnet`/`socat` 也被注释掉。允许反向 shell 和数据外泄。

**修复**: 取消注释，拦截 `source`、`.`、网络工具。

---

#### P1-5: 被拦截命令返回 `Ok` 而非 `Err`

**文件**: `src/bin/ai/tools/service/command.rs:1188-1190`  
**验证**: ✅ 已确认

策略拒绝被报告为**成功**的工具调用。Agent 无法区分策略拒绝和运行时消息。

**修复**: 返回 `Err(reason)` 或定义 `ToolError::Policy` 类型。

---

#### P1-6: `enable_tools` UTF-8 切片 panic

**文件**: `src/bin/ai/tools/enable_tools.rs:223-224`  
**验证**: ✅ 已确认

`desc[..80]` 字节切片在多字节 UTF-8 边界处 panic。

**修复**: 使用 `desc.chars().take(80).collect::<String>()`。

---

#### P1-7: `cargo_check` / `cargo_test` 无超时、无输出上限

**文件**: `src/bin/ai/tools/cargo_tools.rs:90-104`  
**验证**: ✅ 已确认

`Command::new("cargo").output()` 无超时包装，stdout/stderr 全量读入内存。挂起测试或大量输出可无限阻塞或 OOM。

**修复**: 添加超时 + 流式截断。

---

#### P1-8: `web_search` 缓存错误结果

**文件**: `src/bin/ai/tools/web_tools.rs:164-166`  
**验证**: ✅ 已确认

`Err` 结果也被缓存 5 分钟，毒化后续相同查询。

**修复**: 仅缓存 `Ok(非空)` 结果。

---

#### P1-9: SSE 流 — 末事件在无空行终止时丢失

**文件**: `src/bin/ai/stream/runtime.rs:272-294`  
**验证**: ✅ 已确认

`process_pending_tail` 在 `pending` 缓冲区为空时提前返回，不检查 `sse_event_data` 是否仍有未 flush 的事件。部分 provider 在关闭连接前不发送最终空行，最后一个 SSE 事件被静默丢弃。

**修复**: 在 `pending.is_empty()` 分支中也检查 `sse_event_data`。

---

#### P1-10: `normalize.rs` — 缺失 `output_index` 导致并行工具调用碰撞

**文件**: `src/bin/ai/stream/normalize.rs:297-302`  
**验证**: ✅ 已确认

`output_index` 缺失时默认为 0，多个并行工具调用全部碰撞到索引 0，互相覆盖。

**修复**: 使用 `call_id`/`item_id` 派生的稳定键。

---

#### P1-11: Memory 变更无所有权校验

**文件**: `src/bin/ai/tools/service/memory.rs:1071-1181`  
**验证**: ✅ 已确认

`execute_memory_update` 和 `execute_memory_delete` 仅按 `id` 查找，不检查 `owner_pid`。Agent A 可以修改/删除 Agent B 的 owned 条目。

**修复**: 在 update/delete 前校验 owner。

---

#### P1-12: 429 重试忽略 `Retry-After` 头

**文件**: `src/bin/ai/request.rs:1266-1310`  
**验证**: ✅ 已确认

`Retry-After` 头从未被读取（grep 零匹配）。

**修复**: 读取 `Retry-After` 头，与计算退避取较大值。

---

#### P1-13: OpenAI 适配器优先使用 OpenRouter API Key

**文件**: `src/bin/ai/provider/adapter/openai.rs:22-28`  
**验证**: ✅ 已确认

`api_key_candidates` 列表中 `MODEL_OPENROUTER_API_KEY` 排在 `MODEL_OPENAI_API_KEY` 之前，取第一个非空值。

**修复**: 调整候选顺序或按 provider 分离。

---

#### P1-14: `knowledge_consolidate` 更新 JSONL 但不更新 RAG 向量索引

**文件**: `src/bin/ai/tools/knowledge_tools.rs:554-637`  
**验证**: ✅ 已确认

已删除条目的嵌入成为孤儿向量，新合并条目无嵌入。

**修复**: 在 `apply_batch_update` 后同步 RAG 向量存储。

---

#### P1-15: 🚨 思维验证工作流永久卡在 `GenerateHypothesis`

**文件**: `src/bin/ai/driver/thinking/verification.rs:122-130` + `orchestrator.rs:130-144, 194-198`  
**验证**: ✅ 已确认（代码搜索证实 `advance_step` 和 `complete_cycle` 无外部调用者）

`VerificationWorkflow::new(initial_hypothesis)` 设置 `current_step = GenerateHypothesis`，但：
- `advance_step()` 仅在 `current_step == ExecuteTest` 时才前进，而初始状态是 `GenerateHypothesis`
- `complete_verification_cycle` 从未被任何代码调用

一旦模型发出 `<meta:begin_verification>H</meta:begin_verification>`，工作流**永久卡住**，每轮重新注入 `generate_hypothesis_prompt`，要求 LLM 生成被忽略的 JSON。唯一退出是 `<meta:reset_thinking/>`。

**影响**: 每轮浪费 token，零验证价值，直到手动重置。

**修复**: 在 `on_finalize` 中解析 LLM 的 JSON 回复并调用 `advance_step()`/`complete_cycle()`，或移除整个子系统。

---

#### P1-16: 🚨 Tree-of-Thoughts 永不生长；`add_branch` 从未被调用

**文件**: `src/bin/ai/driver/thinking/engine.rs:111-135` + `orchestrator.rs:529-537`  
**验证**: ✅ 已确认（`add_branch` 无外部调用者）

`on_prepare_rich` 调用 `ucb_select`（总是返回根节点）和 `generate_thinking_prompt`（要求 LLM 生成替代假设 JSON），但 orchestrator **从不解析该 JSON、从不调用 `add_branch`**。树永远只有根节点。`ucb_select` 每次选节点 0，同一 prompt 每轮重复注入。

**修复**: 在 `on_finalize` 中解析 JSON 并调用 `add_branch`，或移除。

---

#### P1-17: 🚨 目标分解输出从未被消费；`decompose_active` 从未被调用

**文件**: `src/bin/ai/driver/thinking/goals.rs:339-345` + `orchestrator.rs:601-608`  
**验证**: ✅ 已确认（`decompose_active` 无外部调用者）

`on_prepare_rich` 每轮注入 `generate_decomposition_prompt()`，但 `decompose_active`、`update_sub_goal_state`、`complete_sub_goal` 从未被 orchestrator 调用。目标永远没有子目标，状态永远是 "none yet - decompose first"。

**修复**: 在 `on_finalize` 中解析分解 JSON 并调用 `decompose_active`，或移除。

---

#### P1-18: `task_wait` 的 `wait_policy="all"` 在单次调用中不生效

**文件**: `src/bin/ai/tools/task_tools.rs:1585, 1632-1670`  
**验证**: ✅ 已确认

`epoll_wait_many` 总是以 `WaitPolicy::Any` 调用。当非挂起唤醒（一个任务完成）后，代码重新扫描一次，如果仍有 pending 任务，直接返回而不是循环等待。`wait_policy="all"` 实际退化为 `any` 语义，模型被迫反复调用 `task_wait`，每次看到误导性的 `BUDGET ELAPSED` 标签。

**修复**: 非挂起唤醒下，若仍有 pending 任务，重新调用 `epoll_wait_many` 而非返回。

---

#### P1-19: `task_tools` Channel/futex 泄漏 on 失败路径

**文件**: `src/bin/ai/tools/task_tools.rs:1556-1571, 1646-1658`  
**验证**: ✅ 已确认

子 agent 进程终止但未发布结果时，清理代码只释放 `task_result.consumer`，不释放 `task_result.producer`（因为子 agent 未运行其释放代码）。`channel_destroy` 因 `ref_count != 0` 失败，错误被 `let _ =` 丢弃。channel + event source ref + futex 永久泄漏。对比 `cleanup_launched_agent_team_members`（1077-1084）正确释放两个 holder。

**修复**: 在失败路径中同时释放 `task_result.producer`。

---

#### P1-20: `prune_completed_tasks` 驱逐注册条目但不释放内核资源

**文件**: `src/bin/ai/tools/task_tools.rs:217-232`  
**验证**: ✅ 已确认

注册表超过 100 条时，按 `started_at` 驱逐最旧条目，但不执行 `channel_close`/`channel_destroy`/`futex_destroy`/`kill_process`。且按 spawn 时间驱逐可能赶走仍在运行的任务。

**修复**: 仅驱逐已终止的条目，并执行完整清理序列。

---

#### P1-21: `execute_tool_spawn` 竞态 — worker 可能在 Running 条目插入前完成

**文件**: `src/bin/ai/driver/tools/mod.rs:1365 vs 1432`  
**验证**: ✅ 已确认

worker 线程在 1365 行 spawn，`Running` 条目在 1432 行才插入。快速完成的工具在父线程到达 1432 行前就尝试写入 `Completed` 状态，`get_mut` 返回 `None`，完成状态被静默丢弃。任务永远卡在 `Running`。

**修复**: 在 `thread::spawn` 前插入 `Running` 条目。

---

#### P1-22: 并行工具 batch panic 产生空 `tool_call_id`

**文件**: `src/bin/ai/driver/tools/mod.rs:2321-2334`  
**验证**: ✅ 已确认

并行 batch 中一个线程 panic 时，fallback 的 `ToolResult` 使用空 `tool_call_id`。Anthropic/OpenAI API 要求 `tool_use_id` 匹配，空 id 导致 API 400 或错误归因，可能**中止整个对话轮次**。

**修复**: 在闭包中捕获 `tool_call.id.clone()`，在 panic fallback 中使用。

---

### 2.3 P2 — 次要缺陷

#### P2-1: Prompt XML 标签注入

**文件**: `src/bin/ai/driver/skill_runtime.rs:49-82`  
**验证**: ✅ 已确认

Persona prompt 和 AGENTS.md 内容未转义直接插入 `<identity>` 等标签。`</identity>` 字面量可破坏 prompt 结构。

---

#### P2-2: `std::mem::take(&mut messages)` 后 await 丢失对话（取消时）

**文件**: `src/bin/ai/driver/turn_runtime/orchestrator.rs:849-858`  
**验证**: ⚠️ 需确认取消面

`messages` 在 await 前被 `mem::take` 清空。若 `mid_turn_llm_summarize` 的 future 被取消（运行时关闭、panic），`drained` 在 future 内被 drop，`messages` 保持空——**整轮对话丢失**。

**修复**: 不要在 await 前 `mem::take`；传 `&mut` 或用 guard 恢复。

---

#### P2-3: 空 `final_assistant_text` 跳过所有收尾（持久化/压缩/标题/观察者）

**文件**: `src/bin/ai/driver/turn_runtime/finalize.rs:240`  
**验证**: ✅ 已确认

`Break` 路径的空 `final_assistant_text` 跳过 `persist_pending_turn_messages`、`compact_session_history`、`maybe_generate_session_title`、所有 `on_finalize`。未持久化的工具消息丢失，下一轮继承膨胀的历史。

**修复**: 解耦持久化/压缩与"有最终文本"。

---

#### P2-4: 全局静态 `drain_pending_enable` 与并发子 agent 竞态

**文件**: `src/bin/ai/driver/turn_runtime/orchestrator.rs:769-770`  
**验证**: ⚠️ 需确认并发模型

进程全局 drain。父 turn 和子 subagent 并发运行时，一方的 `drain_pending_enable()` 可窃取另一方的工具启用请求。

---

#### P2-5: `BUDGET ELAPSED` 误标（非挂起唤醒时）

**文件**: `src/bin/ai/tools/task_tools.rs:1670-1699`  
**验证**: ✅ 已确认

非挂起唤醒（事件已就绪）但仍有 pending 任务时，代码发出 `BUDGET ELAPSED`，但预算**并未**耗尽。模型误判任务健康，可能放弃子 agent。

---

#### P2-6: `task_status` 无限泄漏已完成任务

**文件**: `src/bin/ai/tools/task_tools.rs:1726-1767`  
**验证**: ✅ 已确认

`task_status` 用 `consume=false` 查看结果，不移除注册条目或销毁内核资源。模型通过 `task_status` 消费结果但不调用 `task_wait` 时，channel/futex 永久存活。

---

#### P2-7: 思维子系统字节切片 panic 风险

**文件**: `src/bin/ai/driver/thinking/engine.rs:369-373`, `goals.rs:180-184`  
**验证**: ✅ 已确认

`&tool_results[..2000]` 和 `&self.context[..1000]` 不检查字符边界。目前是死路径（未被调用），但一旦接入会 panic，且 panic 被 `catch_unwind` 捕获后会**永久毒化观察者**，静默禁用所有思维功能。

---

#### P2-8: `is_work_turn` 守卫与文档意图矛盾

**文件**: `src/bin/ai/driver/thinking/orchestrator.rs:637-646`  
**验证**: ✅ 已确认

注释说自学习不应在非工作轮（纯 Q&A）运行，但守卫是 `ctx.had_tool_calls || !self.active_modes.is_empty()`。`apply_meta_tags` 在检查前运行并插入 `active_modes`，所以仅发出 meta-tag（无工具）的轮次也被视为 "work turn"。

---

#### P2-9: 反思进化 — 计数器丢失更新竞态

**文件**: `src/bin/ai/driver/reflection/background.rs:475-507`  
**验证**: ✅ 已确认

`apply_evolution_feedback` 做非原子 read-modify-write：读取 `target.tags` 快照 → 计算新计数器 → 写回完整 tags 数组。两个并发反思可都读 `pass=2`、都写 `pass=3`，丢失一次增量。

---

#### P2-10: 反思进化 — 重复 canary 提升

**文件**: `src/bin/ai/driver/reflection/background.rs:399-473`  
**验证**: ✅ 已确认

`maybe_promote_stable_self_note` 检查 `has_canary` 和 `append` 是两次独立锁获取。两个并发反思可都观察到"无 canary"并都 append，违反"仅一个 canary"不变量。

---

#### P2-11: 反思 — writeback 去重在后台任务成功前记录

**文件**: `src/bin/ai/driver/reflection/writeback.rs:33-47, 107-111`  
**验证**: ✅ 已确认

`writeback_recently_seen` 在 `tokio::spawn` 前同步插入去重 key。后台任务失败时（LLM 超时、网络错误），去重条目持续 5 分钟，阻止重试。

---

#### P2-12: 反思 — daemon 条目永不清理

**文件**: `crates/aios_kernel/src/local.rs:2888-2942`  
**验证**: ✅ 已确认

`daemon_exit` 只翻状态，不 `remove`。每轮至少 spawn 一个 writeback daemon，条目无限累积。长会话内存增长与轮次成正比。

---

#### P2-13: 反思 — `evaluate_turn_feedback` 误报"讨论错误"的答案为失败

**文件**: `src/bin/ai/driver/reflection/background.rs:536-567`  
**验证**: ✅ 已确认

返回 `Fail` 如果最终回答包含 "error"/"failed"/"失败" 等词。正确诊断错误的答案（如"构建错误由缺少 use 导入引起"）被误判为 `Fail`，驱动错误的回滚。

---

#### P2-14: `MAX_INPUT_CHARS = 4000` 可能截断粘贴的代码/错误

**文件**: `src/bin/ai/prompt.rs:20`  
**验证**: ⚠️ 常量存在，强制执行路径未确认

---

#### P2-15: `mv` 路径限制是词法的，非符号链接解析

**文件**: `src/bin/ai/tools/service/command.rs:949-1037`  
**验证**: ✅ 已确认

---

#### P2-16: `try_wait` 错误导致资源泄漏

**文件**: `src/cmd/run.rs:436`  
**验证**: ✅ 已确认

---

#### P2-17: `knowledge_search` category 过滤无效

**文件**: `src/bin/ai/tools/knowledge_tools.rs:301-307`  
**验证**: ✅ 已确认

category 被拼接到搜索字符串而非作为字段过滤。

---

#### P2-18: `knowledge_save` 静默丢弃 RAG 索引更新失败

**文件**: `src/bin/ai/tools/knowledge_tools.rs:125-145`  
**验证**: ✅ 已确认

---

#### P2-19: `compact_context` 是空操作 stub

**文件**: `src/bin/ai/tools/context_tools.rs:50-70`  
**验证**: ✅ 已确认

返回硬编码字符串，实际不执行任何压缩。系统提示中却宣传为真实能力。

---

#### P2-20: 项目指令文档无长度上限

**文件**: `src/bin/ai/driver/skill_runtime.rs:688-703`  
**验证**: ✅ 已确认

本仓库有 6+ 个 scoped AGENTS.md，每轮全量拼接，破坏 prompt-cache 稳定性。

---

#### P2-21: `cargo_test` 默认 `--workspace` 违反项目规则

**文件**: `src/bin/ai/tools/cargo_tools.rs:80, 92-94`  
**验证**: ✅ 已确认

---

#### P2-22: `execute_tool_cancel`/`execute_tool_wait` 遇到首个未知 id 即中止整个调用

**文件**: `src/bin/ai/driver/tools/mod.rs:1587-1590, 1651/1757`  
**验证**: ✅ 已确认

与 `execute_tool_status`（跳过未知 id 返回部分结果）不一致。一个过期 id 导致所有有效 id 的结果丢失。

---

#### P2-23: `mcp_client.lock().unwrap()` 在 mutex 中毒时 panic

**文件**: `src/bin/ai/driver/turn_runtime/orchestrator.rs:713, 744`  
**验证**: ✅ 已确认

与 `finalize.rs:191,215` 使用 `.into_inner()` 恢复不一致。

---

## 3. 效果与 Prompt 优化

### E1: 🚨 思维子系统全部是死循环（最高优先级效果问题）

**文件**: `src/bin/ai/driver/thinking/orchestrator.rs`

三个"思维模式"（Verification、ToT、Goal Decomposition）共享同一失败模式：每轮注入 `STRICT JSON` prompt，然后**忽略 LLM 的 JSON 回复**。`advance_step`、`complete_cycle`、`add_branch`、`decompose_active` 均无外部调用者（已验证）。

**净效果**：任何被 LLM 通过 `<meta:begin_*>` 激活的模式，将对话锁定为每轮重复注入同一 prompt，直到 `<meta:reset_thinking/>`。这是**每轮 token 税**，零状态进展。

**此外**：`build_system_prompt_injection` 和 `on_prepare_rich` 对同一模式做**双重注入**——先一个通用状态块，再一个详细 prompt 块。LLM 每轮收到两份重叠指令。

**建议**: 
1. 短期：移除 `<meta:begin_*>` 解析，禁用死循环
2. 长期：在 `on_finalize` 中实现 JSON 回复解析 + 状态推进，或用更简单的内联推理替代

---

### E2: 反思子系统默认禁用

**文件**: `src/bin/ai/driver/turn_runtime/finalize.rs:136-151, 170-178`

`maybe_append_self_reflection` 的门控是 `!ne("false")`，即仅在值为 `"false"` 时运行。默认值 `"true"` → **跳过**。`maybe_spawn_critic_revise_background` 同理（`if integrated { return; }`）。整个进化/canary/回滚机制（~600 行）在默认配置下**永不执行**。

**建议**: 确认预期默认值。如果"集成"反思在其他地方处理，删除后台路径；否则翻转门控。

---

### E3: 通用 fallback identity 与 persona 冲突

**文件**: `skill_runtime.rs:762-766 + 1062-1078`

无 skill/agent 激活时，identity = "You are a highly capable general-purpose AI assistant…"。非默认 persona 追加另一个 Identity 段。模型看到两个矛盾的角色声明。

**建议**: persona 激活时省略通用 fallback。

---

### E4: Agent + persona 共存优先级未定义

**文件**: `skill_runtime.rs:744-759, 1062-1077`

persona 的自我贬低句是唯一的优先级声明，无结构化强制。

**建议**: 定义显式优先级，persona 仅作为"文体风味"注入。

---

### E5: 无输出格式 / 代码块约定

**文件**: `skill_runtime.rs:784-786`

未指定代码围栏语言标记、文件路径头、diff 格式、何时用 `apply_patch` vs 内联片段。Aider/Cursor 有明确输出契约。

---

### E6: 工具名硬编码字符串造成静默过时风险

**文件**: `skill_runtime.rs:788-1029`

每个工具指导段用 `has_tool(available_tools, "<name>")` 门控，名称硬编码。重命名后指导段静默消失。

**建议**: 集中化工具名常量。

---

### E7: `plan` 工具纯粹装饰性

**文件**: `src/bin/ai/tools/plan_tools.rs:45-97`

`execute_plan` 格式化步骤为字符串并返回 — 不持久化、不跟踪、不强制执行。

**建议**: 持久化到 scratchpad 或移除。

---

### E8: 命令输出截断过小且不透明

`MAX_COMMAND_OUTPUT_CHARS = 16_000`，截断仅追加 `\n... (truncated)`，无字节数、无头尾保留。`cargo build` 失败时常丢失实际错误行。

---

### E9: `task_wait` PARKED 往返主导 `wait_policy=all` 场景

每次协作挂起返回冗长 `PARKED` 消息，强制模型重新调用。N 任务 join = N 往返，每次发出多行消息。结合 P1-18（`all` 不生效），"fan out 5 subagents and join" 的 UX 很差。

**建议**: 挂起后让 driver 自动重入 `task_wait`，返回单一合并结果。

---

### E10: `task_spawn` 标记为 `SyncOnly` 阻止并行批量

**文件**: `src/bin/ai/tools/task_tools.rs:552, 596, 659, 1393`

`task`、`task_spawn`、`agent_team`、`task_wait` 全部 `SyncOnly`，无法在单次并行工具调用 batch 中发出。文档描述的"fan out several in parallel"要求模型跨多轮顺序发出 `task_spawn`。

---

### E11: 无结果截断 — subagent 大输出可爆父 context

**文件**: `src/bin/ai/tools/task_tools.rs:1852-1874`

`format_task_result` 全量内联 `result.output`，无长度上限。

---

### E12: 无持久权限模型

策略纯黑名单。Claude Code/Cursor 使用白名单 + 交互式批准 + "always allow X" 持久化。这里每个被拦截命令是硬停止无升级路径，而每个非拦截命令自由运行。

---

### E13: Persona 注入泄漏实现框架

persona 块暴露内部词汇（"Persistent persona"、"Avatar"、"higher-priority agent, skill, policy"）。现代 persona 应呈现为自然角色文本。

---

### E14: `web_search` 空结果返回 `Err`

空结果集是合法搜索结果，但返回 `Err("No results found for: ...")`。模型视为失败并重试。

---

### E15: `eprintln!` debug spam 在生产路径

**文件**: `src/bin/ai/driver/turn_runtime/finalize.rs:352`

```rust
eprintln!("[session-title] sessions_dir={}", sessions_dir.display());
```

每轮收尾无条件打印到 stderr。

---

### E16: 空 `final_assistant_text` 跳过标题生成

**文件**: `src/bin/ai/driver/turn_runtime/finalize.rs:240, 387`

标题写入失败被静默忽略，可能导致每轮重新生成标题（浪费 LLM 调用）。

---

## 4. 架构优化建议

### A1: `SystemPromptBuilder` 应提取为独立模块

**文件**: `src/bin/ai/driver/skill_runtime.rs`（1923 行）

系统 prompt 组装、工具选择、MCP 过滤、能力目录、测试全部埋在一个文件中。

---

### A2: `validate_single_segment` 是 220 行过程式策略

**文件**: `src/bin/ai/tools/service/command.rs:926-1149`

两个重叠列表（`DANGEROUS_PROGRAM_NAMES` 和 `denied_programs`）分别维护。

**建议**: 表驱动规则 `{ program, arg_index_extractor, deny_set }`，单一真相源。

---

### A3: 验证和执行使用不同分词器

`validate_execute_command` 用 `tokenize_shell_words`，执行用 `split_space_keep_symbol`。不一致时命令在一种分词下验证通过、在另一种下执行。

**建议**: 共享一个分词器 + property test。

---

### A4: `runtime.rs` 2590 行，隐式状态机

SSE framing、内容提取、渲染、工具调用恢复、usage 跟踪、取消全部在一个文件中。状态机是过程式循环 + 标志位。

**建议**: 拆分模块 + 显式 `StreamState` 枚举。

---

### A5: 两个命令执行入口有策略分歧风险

`execute_command`（有验证）和 `os_tools::spawn_process`（策略未验证）是独立路径。

**建议**: 单一命令执行网关。

---

### A6: 思维子系统与 driver 通过 meta-tag 字符串协议紧耦合

**文件**: `src/bin/ai/driver/thinking/orchestrator.rs:208-244, 445-485`

整个思维状态机由解析字面子串（`<meta:begin_verification>` 等）驱动。无结构化协议、无类型安全、无版本化。

**建议**: 定义 `enum ThinkingCommand { BeginVerification(Hypothesis), Reset, ... }`，在 driver 层统一解析。

---

### A7: `agents.rs`/`skills.rs` 的 `build_system_prompt` 是朴素拼接

`system_prompt + "\n\n" + prompt`，无角色契约与过程指令的结构化分离。

---

### A8: `context_reminder` 和 `system_prompt` 缓存生命周期不一致

两缓存共享 builder 但在 skill 切换时只 patch `system_prompt`，`context_reminder` 无等价 patch 路径。

**建议**: 单一 `render()` 返回两者，按内容哈希版本化。

---

### A9: Channel ref-count 责任分散，无 RAII

**文件**: `src/bin/ai/tools/task_tools.rs:779-782`, `driver/mod.rs:830, 2268`

`task_result.producer` 由子 agent 释放，`task_result.consumer` 由 `task_wait` 释放。任一方崩溃 → `channel_destroy` 失败 → 泄漏。同一约定跨 `task_tools.rs`、`os_tools.rs`、`driver/mod.rs` 重复。

**建议**: `TaskResultChannel` RAII handle 拥有两个 holder slot，drop 时释放两者。

---

### A10: `TASK_REGISTRY` 是进程全局 static，非 session 隔离

**文件**: `src/bin/ai/tools/task_tools.rs:198-199`

多 session 或测试间任务互不可见，`task_status` 返回外部任务。

**建议**: 隔离到 `DriverContext` 或按 session_id 键控。

---

### A11: 锁顺序不变量仅靠注释维护

**文件**: `src/bin/ai/driver/turn_runtime/finalize.rs:190-226`

`app.os` / `GLOBAL_OS`（共享 `Arc<Mutex<Kernel>>`）与 futex helpers 间的自死锁风险仅以注释形式存在。

**建议**: 封装 register→alloc→spawn→exit→destroy 序列为单一 helper。

---

### A12: `is_precision_tool` 硬编码 match 列表

**文件**: `src/bin/ai/driver/turn_runtime/context_budget.rs:402-405`

违反 AGENTS.md 规则 #7（"避免硬编码字符串规则"）。MCP/skill 工具无法被分类为 precision。

**建议**: 在工具注册元数据中添加 `precision: bool` 字段。

---

### A13: Undo 状态仅内存

**文件**: `src/bin/ai/tools/undo_tools.rs:5`

进程崩溃后历史丢失。

**建议**: 持久化快照到 `.a/undo/`。

---

### A14: 反思 `active_modes` 是易失缓存

**文件**: `src/bin/ai/driver/thinking/orchestrator.rs:509-517, 692-693`

每轮 finalize 清空、prepare_rich 重建。若代码路径设置 mode 但未创建 backing object，缓存静默失步。

**建议**: 改为按需计算的方法。

---

## 5. 验证记录

### 验证轮次 1（7 项）

| # | 发现 | 结论 | 证据 |
|---|------|------|------|
| 1 | `web_fetch` SSRF 重定向绕过 | ✅ 确认 | reqwest builder 无 `redirect(Policy::none())` |
| 2 | `web_fetch` SSRF DNS 重绑定 | ✅ 确认 | IP 检查仅对字面 IP 触发 |
| 3 | `enable_tools` UTF-8 panic | ✅ 确认 | `desc[..80]` 字节切片 |
| 4 | `cargo_check/test` 无超时 | ✅ 确认 | `Command::output()` 无超时/截断 |
| 5 | `web_search` 缓存错误 | ✅ 确认 | `cache.put(key, result.clone())` 含 Err |
| 6 | 被拦截命令返回 Ok | ✅ 确认 | `return Ok(format!("Command blocked: ..."))` |
| 7 | 绝对路径绕过黑名单 | ✅ 确认 | `program` 未 basename 归一化 |

### 验证轮次 2（7 项）

| # | 发现 | 结论 | 证据 |
|---|------|------|------|
| 1 | Memory GC 腐败 | ❌ **推翻** | 无 `apply_batch_update`/tombstone 机制；GC 用 temp-file+rename |
| 2 | Memory 变更无所有权校验 | ✅ 确认 | update/delete 仅按 `id` 查找，不检查 owner |
| 3 | `reasoning_content` 丢弃 | ✅ 确认 | `content.is_empty()` 前置条件 |
| 4 | `output_index` 默认 0 碰撞 | ✅ 确认 | `unwrap_or(0)` 确凿 |
| 5 | DeepSeek thinking 开关丢弃 | ✅ 确认（设计意图） | 文档注释明确要求省略 |
| 6 | 429 忽略 Retry-After | ✅ 确认 | grep 零匹配 |
| 7 | OpenAI 用 OpenRouter key | ✅ 确认 | 候选列表顺序 OpenRouter 优先 |

### 验证轮次 3（6 项）

| # | 发现 | 结论 | 证据 |
|---|------|------|------|
| 1 | SSE 末事件丢失 | ✅ 确认 | `pending.is_empty()` 提前返回，不检查 `sse_event_data` |
| 2 | Prompt XML 标签注入 | ✅ 确认 | 无转义，persona/AGENTS.md 直接插入 |
| 3 | Patch 半写入 | ❌ **推翻** | 全内存计算，成功才写入；无 backup 但不需要 |
| 4 | Consolidate 不更新 RAG | ✅ 确认 | `apply_batch_update` 不触碰向量存储 |
| 5 | `compact_context` 空操作 | ✅ 确认 | 返回硬编码字符串 |
| 6 | 项目指令无长度上限 | ✅ 确认 | `doc.content.trim()` 全量拼接 |

### 推翻的发现

1. **Memory GC 腐败**（原 P0）：子 agent 报告的 `apply_batch_update` + tombstone 机制**不存在**。GC 使用标准 temp-file + atomic rename。代码路径安全。

2. **Patch 半写入**（原 P2）：`apply_unified_patch` 在内存 `Vec<String>` 中计算完整结果，任何 hunk 失败立即返回 `Err`，写入仅在全部成功后执行。不存在半写入状态。

---

## 6. 优先级排序

### 立即修复（P0 安全/数据丢失）

| # | 问题 | 修复复杂度 |
|---|------|-----------|
| 1 | `web_fetch` SSRF 重定向绕过 | 低 |
| 2 | `web_fetch` SSRF DNS 重绑定 | 中 |
| 3 | `reasoning_content` 同 chunk 丢弃 | 低 |

### 尽快修复（P1 功能）

| # | 问题 | 修复复杂度 |
|---|------|-----------|
| 4 | 🚨 思维验证工作流永久卡住 | 中 — 实现 JSON 解析或移除 |
| 5 | 🚨 ToT 永不生长 | 中 — 实现 JSON 解析或移除 |
| 6 | 🚨 目标分解从未消费 | 中 — 实现 JSON 解析或移除 |
| 7 | 命令沙箱绝对路径绕过 | 低 |
| 8 | 命令沙箱子 shell 绕过 | 低 |
| 9 | 命令沙箱解释器绕过 | 低 |
| 10 | 命令沙箱 `source`/`nc` 未拦截 | 低 |
| 11 | `enable_tools` UTF-8 panic | 低 |
| 12 | SSE 末事件丢失 | 低 |
| 13 | `output_index` 默认 0 碰撞 | 中 |
| 14 | `cargo_check/test` 无超时 | 中 |
| 15 | `web_search` 缓存错误 | 低 |
| 16 | Memory 变更无所有权校验 | 低 |
| 17 | 429 忽略 Retry-After | 低 |
| 18 | OpenAI 用 OpenRouter key | 低 |
| 19 | Consolidate 不更新 RAG | 中 |
| 20 | 被拦截命令返回 Ok | 低 |
| 21 | `task_wait` wait_policy=all 不生效 | 中 |
| 22 | task_tools channel/futex 泄漏 | 中 |
| 23 | `prune_completed_tasks` 不释放资源 | 中 |
| 24 | `execute_tool_spawn` 竞态 | 低 — 调换顺序 |
| 25 | 并行 batch panic 空 tool_call_id | 低 |

### 计划修复（P2 + 效果）

- Prompt XML 标签注入
- `mem::take` + await 丢失对话
- 空 final_assistant_text 跳过收尾
- 全局 drain_pending_enable 竞态
- 思维字节切片 panic
- 反思计数器/daemon/writeback 问题
- 项目指令文档无上限
- `compact_context` 空操作
- `knowledge_search` category 过滤无效
- 命令输出截断过小
- Persona/fallback identity 冲突
- `plan` 工具装饰性
- `cargo_test` 默认 `--workspace`
- `task_status` 泄漏已完成任务
- `eprintln!` debug spam

### 架构演进

- `SystemPromptBuilder` 提取独立模块
- 命令验证/执行统一分词器
- `runtime.rs` 拆分 + 显式状态机
- 命令执行单一网关
- 思维子系统结构化协议替代 meta-tag 字符串
- Channel ref-count RAII
- `TASK_REGISTRY` session 隔离
- 锁顺序封装为 helper
- `is_precision_tool` 元数据驱动

---

## 附录：审查覆盖范围

### 已完整审查的模块（16 个子 agent）

| 模块 | 结果质量 |
|------|---------|
| Prompt 组装 (`skill_runtime.rs`) | 详尽 |
| Service 层 (`memory.rs`, `command.rs`) | 详尽（1 项推翻） |
| Provider 适配器 (`openai.rs`, `thinking.rs` 等) | 详尽 |
| 流运行时 (`stream/runtime.rs`) | 详尽 |
| 流提取/归一化 (`extract.rs`, `normalize.rs`, `splitter.rs`) | 详尽 |
| 命令/OS/Git/Patch 工具 | 详尽（1 项推翻） |
| Web/Knowledge/RAG/Skill/Cargo/Plan 工具 | 详尽 |
| MCP/Knowledge/Skill 路由 | 部分（路径问题） |
| Reflection 子系统 | 详尽 |
| Task tools + 子 agent 管理 | 详尽 |
| Turn runtime 编排 | 详尽 |
| Driver/tools 层 | 详尽 |
| Thinking 子系统 | 详尽（重大发现） |
| Config/CLI/History | 有内容 |
| Code search/LSP/AST 工具 | 部分（未完成最终答案） |
| Request 层 | 部分（与 prompt assembly 重叠） |

### 未完整审查的模块

| 模块 | 原因 |
|------|------|
| `models.rs` 模型注册 | 子 agent 仅返回 thinking |
| Hooks/Signal/Observer | 子 agent 仅返回 thinking |
| Code search/LSP/AST | 子 agent 未完成最终答案 |

**建议**: 对未完整审查的模块进行补充审查，特别是 `models.rs`（模型路由逻辑）和 code search/LSP 工具。

---

*报告结束*
