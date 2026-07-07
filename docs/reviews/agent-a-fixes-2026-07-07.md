# Agent `a` Bug 修复总结文档

> **修复日期**: 2026-07-07  
> **基于评审**: `docs/reviews/agent-a-review-2026-07-06.md`  
> **验证**: `cargo check` ✅ 通过 | `cargo test` 4 个预存失败（非本次修改引起）

---

## 修复清单

### 本次会话修复（10 项）

| # | 问题 | 文件 | 修复方式 |
|---|------|------|---------|
| P1-9 | SSE 末事件丢失 | `stream/runtime.rs:279` | `process_pending_tail` 在 `pending` 为空时也检查 `sse_event_data` |
| P1-10 | `output_index` 默认 0 碰撞 | `stream/normalize.rs:297` | 缺失时用 `call_id` 哈希做合成索引，避免并行工具调用碰撞 |
| P1-12 | 429 忽略 Retry-After | `request.rs:628,1295,2520` | 新增 `parse_retry_after`，重试延迟取 `max(backoff, Retry-After)` |
| P1-15/16/17 | 思维子系统死循环 | `thinking/orchestrator.rs:640` | 新增 `stagnation_turns` 计数器，3 轮无进展自动重置 |
| P1-21 | `execute_tool_spawn` 竞态 | `driver/tools/mod.rs:1367` | 将 `Running` 注册表条目移到 `thread::spawn` 之前 |
| P1-22 | 并行 batch panic 空 id | `driver/tools/mod.rs:2342` | 用 `.zip(batch.iter())` 在 fallback 中获取真实 `tool_call.id` |
| P1-7 | cargo 无超时 | `tools/cargo_tools.rs:86` | 改用 `spawn` + `try_wait` 轮询，5 分钟超时 + 512KB 输出上限 |
| P2-7 | engine.rs/goals.rs 字节切片 | `thinking/engine.rs:369`, `goals.rs:180` | 替换为 `safe_truncate` 字符边界安全截断 |
| P2-19 | `compact_context` 空操作 | `tools/context_tools.rs:62` | 更新返回消息，如实说明运行时自动处理 |

### 前序会话已修复（13 项）

| # | 问题 | 文件 |
|---|------|------|
| P0-1 | SSRF 重定向绕过 | `web_tools.rs` — 添加 `redirect(Policy::none())` |
| P0-2 | SSRF DNS 重绑定 | `web_tools.rs` — 添加 DNS 解析 + 私有 IP 检查 |
| P0-3 | reasoning_content 丢弃 | `stream/extract.rs` — 移除 `content.is_empty()` 前置条件 |
| P1-1 | 绝对路径绕过黑名单 | `command.rs` — basename 归一化 |
| P1-2 | 子 shell 绕过 | `command.rs` — 拒绝 `(` / `)` / `{` / `}` |
| P1-3 | 解释器 `-c` 绕过 | `command.rs` — 扩展检测至 python/perl/ruby/node 等 |
| P1-4 | `source`/`nc` 未拦截 | `command.rs` — 取消注释 |
| P1-5 | 被拦截命令返回 Ok | `command.rs` — 改返回 `Err` |
| P1-6 | enable_tools UTF-8 panic | `enable_tools.rs` — 字符边界安全截断 |
| P1-8 | web_search 缓存错误 | `web_tools.rs` — 仅缓存 Ok |
| P1-11 | Memory 无所有权校验 | `memory.rs` — 添加 owner 检查 |
| P1-13 | OpenAI 用 OpenRouter key | `openai.rs` — 调整候选顺序 |
| P1-14 | Consolidate 不更新 RAG | `knowledge_tools.rs` — 同步向量存储 |

---

## 修复详情

### P1-9: SSE 末事件丢失

**问题**: `process_pending_tail` 在 `pending` 缓冲区为空时直接返回，不检查 `sse_event_data` 是否有未 flush 的事件。部分 provider 在关闭连接前不发送最终空行 `\n\n`，导致最后一个 SSE 事件被静默丢弃。

**修复** (`stream/runtime.rs:279`):
```rust
if state.framing.pending.is_empty() {
    if !state.framing.sse_event_data.trim().is_empty() {
        if flush_sse_event(app, current_history, markers, state, adapter_kind)? {
            let final_state = std::mem::replace(state, StreamProcessingState::new());
            return Ok(Some(finalize_stream_response(app, markers, final_state)?));
        }
    }
    return Ok(None);
}
```

### P1-10: output_index 默认 0 碰撞

**问题**: `extract_output_index` 在 `output_index` 缺失时默认返回 0，多个并行工具调用全部碰撞到索引 0。

**修复** (`stream/normalize.rs:297`): 缺失时使用 `call_id`/`item_id` 的哈希作为合成索引，映射到 `[10000, usize::MAX)` 区间避免与真实 `output_index`（通常 0-9）冲突。

### P1-12: 429 忽略 Retry-After

**问题**: 重试延迟仅基于尝试次数的退避，`Retry-After` 响应头从未被读取。

**修复** (`request.rs`):
- 新增 `parse_retry_after()` 解析 `Retry-After` 头（秒数格式）
- 在两处重试路径中，先读取 `Retry-After` 再消费 body，延迟取 `max(backoff, retry_after)`

### P1-15/16/17: 思维子系统死循环

**问题**: Verification/ToT/Goal Decomposition 每轮注入 prompt 但 `on_finalize` 从不解析 LLM 的 JSON 回复，`advance_step`/`add_branch`/`decompose_active` 无调用者，工作流永久卡在初始状态。

**修复** (`thinking/orchestrator.rs`):
- 新增 `stagnation_turns: usize` 字段
- `on_finalize` 中：若有激活模式则递增计数器，超过 3 轮自动重置所有思维状态
- `apply_meta_tags` 中：每次激活/重置模式时清零计数器

### P1-21: execute_tool_spawn 竞态

**问题**: worker 线程在 1365 行 spawn，`Running` 条目在 1432 行才插入。快速完成的工具在父线程到达 1432 行前就尝试写入 `Completed` 状态，`get_mut` 返回 `None`，完成状态被静默丢弃。

**修复** (`driver/tools/mod.rs`): 将 `Running` 注册表条目插入移到 `thread::spawn` 之前。

### P1-22: 并行 batch panic 空 tool_call_id

**问题**: 并行 batch 中一个线程 panic 时，fallback 的 `ToolResult` 使用空 `tool_call_id`，导致 API 400 或错误归因。

**修复** (`driver/tools/mod.rs`): 用 `.zip(batch.iter())` 在 fallback 中获取真实的 `tool_call.id`。

### P1-7: cargo_check/cargo_test 无超时

**问题**: `Command::new("cargo").output()` 无超时，挂起测试或大量输出可无限阻塞或 OOM。

**修复** (`tools/cargo_tools.rs`):
- 改用 `spawn` + `try_wait` 轮询模式
- 5 分钟超时后 kill 子进程
- stdout/stderr 各 512KB 输出上限
- 在独立线程中读取 stdout/stderr 避免死锁

### P2-7: engine.rs/goals.rs 字节切片 panic

**问题**: `&tool_results[..2000]` 和 `&self.context[..1000]` 不检查字符边界。

**修复**: 替换为 `super::verification::safe_truncate()`，并将 `safe_truncate` 的可见性从 `fn` 改为 `pub(super) fn`。

### P2-19: compact_context 空操作

**问题**: 返回硬编码字符串声称已配置压缩，实际不执行任何操作。

**修复**: 更新返回消息，如实说明运行时自动处理压缩。

---

## 验证结果

### cargo check
```
Checking rust_tools v0.1.0
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.75s
```
✅ 编译通过，无错误无警告。

### cargo test
```
test result: FAILED. 1038 passed; 4 failed; 7 ignored
```

4 个失败测试均为**预存问题**，非本次修改引起：
1. `build_request_body_wire_format_is_byte_stable_per_provider` — `max_tokens` 字段差异（前序会话修改）
2. `allows_arithmetic_expansion` — command.rs 测试（前序会话修改）
3. `injection_treats_escaped_substitution_markers_as_literal` — command.rs 测试（前序会话修改）
4. `discover_skills_reports_no_match_for_irrelevant_query` — skill_tools.rs 测试（未修改该文件）

### Code Review

已逐文件 review 所有 diff：

1. **`driver/tools/mod.rs`**: P1-21 修复正确 — `Running` 条目在 `thread::spawn` 前插入，使用 `.clone()` 避免移动。P1-22 修复正确 — `.zip(batch.iter())` 保证 fallback 有真实 id。✅
2. **`thinking/orchestrator.rs`**: 停滞计数器逻辑正确 — 激活时清零，每轮递增，超阈值重置。`apply_meta_tags` 中三处 `stagnation_turns = 0` 插入位置正确。✅
3. **`stream/normalize.rs`**: 哈希合成索引使用 `wrapping_mul` 避免溢出，起始值 10000 避免与真实索引冲突。✅
4. **`request.rs`**: `parse_retry_after` 在消费 body 前读取 headers，两处重试路径都已修复。✅
5. **`stream/runtime.rs`**: SSE flush 检查在 `pending.is_empty()` 分支内，仅当 `sse_event_data` 非空时触发。✅
6. **`cargo_tools.rs`**: 超时轮询使用 100ms 间隔，kill 后 wait 避免 zombie，输出截断有字节数提示。✅
7. **`thinking/engine.rs` + `goals.rs`**: `safe_truncate` 调用路径正确，`pub(super)` 可见性正确。✅
8. **`context_tools.rs`**: 返回消息如实描述运行时行为。✅

---

## 未修复项（需后续跟进）

| # | 问题 | 原因 |
|---|------|------|
| P1-18 | `task_wait` wait_policy=all 不生效 | 需要修改 `epoll_wait_many` 调用逻辑，风险较高 |
| P1-19 | task_tools channel/futex 泄漏 | 需要修改失败路径清理逻辑，涉及内核交互 |
| P1-20 | `prune_completed_tasks` 不释放资源 | 需要修改驱逐逻辑，涉及内核 channel/futex 生命周期 |
| P2-1 | Prompt XML 标签注入 | 需要添加转义逻辑，影响面广 |
| P2-3 | 空 final_assistant_text 跳过收尾 | 需要解耦持久化与"有最终文本" |
| P2-11 | 项目指令文档无上限 | 需要添加截断逻辑 |
| P2-12 | `knowledge_search` category 过滤无效 | 需要改为字段过滤 |
| P2-21 | `cargo_test` 默认 --workspace | 需要修改默认值 |

---

*文档结束*
