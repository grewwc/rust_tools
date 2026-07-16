# a.rs 子 Agent 协作功能 Review

## 结论

当前子 Agent 协作功能整体可用，但协作语义还不够硬，主要风险是：

1. 子 Agent 默认继承的上下文偏宽。
2. 异步 `task_spawn` 不保证主 Agent 最终回答前一定等待子 Agent 返回。
3. 子 Agent 报错主要作为普通结果返回，缺少强制处理机制。
4. 当前允许子 Agent 继续派生孙 Agent，只用最大深度兜底；如果产品预期是禁止嵌套，这个设计不够直接。

相关实现主要不在 `src/bin/a.rs` 单文件内，而在：

- `src/bin/ai/tools/task_tools.rs`
- `src/bin/ai/driver/tools/sync_task.rs`
- `src/bin/ai/driver/background_dispatch.rs`
- `src/bin/ai/driver/runtime_ctx.rs`
- `src/bin/ai/driver/turn_runtime/*`

## 1. 子 Agent 上下文是否有问题？

有潜在问题：默认上下文继承偏宽。

当前默认 `InheritOptions` 为：

- `history: true`
- `memory: false`
- `cwd: true`
- `skills: true`

见 `src/bin/ai/tools/task_tools.rs:51-77`。

也就是说，默认子 Agent 会继承父 Agent 的历史、当前工作目录、skills/tools，但 memory 默认私有。

### 合理点

- `memory` 默认私有，避免子 Agent 把大量临时推理、任务日志、事件写入主 memory。
- `inherit` 支持显式配置：`"all"`、`"none"`、`"history,memory,cwd,skills"` 等。

### 风险点

1. 默认继承 `history` 容易让子 Agent 上下文过大。
   - 对 leaf task 来说，子 Agent 通常只需要一段聚焦 prompt 和必要文件上下文。
   - 默认带完整历史容易造成 token 膨胀、注意力偏移，也可能继承主 Agent 的协作策略，而不是专注完成子任务。

2. 默认继承 `cwd` 可能带来并发写冲突。
   - `task_spawn` 支持并行启动多个子 Agent。
   - 如果多个子 Agent 默认共享项目 cwd，又都拥有写文件能力，可能并发修改同一文件。
   - 当前代码支持 `inherit.cwd=false` 时创建 per-task scratch cwd，但默认不是 scratch。

3. 默认继承 `skills/tools` 让嵌套委派变得自然。
   - 子 Agent 如果继续拥有 `task` / `task_spawn` / `agent_team`，模型会倾向继续开子 Agent。
   - 当前主要依赖最大深度限制兜底，而不是从工具暴露层主动禁止。

### 建议

- 默认改为更窄的继承策略，例如 `inherit="cwd,skills"` 或 `inherit="cwd"`；`history` 改为 opt-in。
- 对异步子 Agent 默认只读；写操作需要显式授权。
- 子 Agent 的工具集应可裁剪，默认隐藏 `task` / `task_spawn` / `agent_team`，除非 manifest 或 spawn 参数显式允许。

## 2. 主 Agent 是否一定等待子 Agent 返回？

不是。需要区分同步和异步两种模式。

### `task`：同步等待

`task` 是同步子 Agent 调用。工具描述明确说明当前 Agent 会阻塞，直到子 Agent 完成。

相关实现：

- 启动子 Agent turn：`src/bin/ai/driver/tools/sync_task.rs:170-190`
- 安装 result slot / phase slot / depth scope：`src/bin/ai/driver/tools/sync_task.rs:197-204`
- `block_in_place` 等待 oneshot / cancel / hard timeout：`src/bin/ai/driver/tools/sync_task.rs:247-257`

所以 `task` 基本会等待到完成、失败、取消或 hard timeout。

### `task_spawn`：异步，不强制等待

`task_spawn` 是异步接口，只会立即返回 `task_id`。主 Agent 必须之后主动调用 `task_wait` 或 `task_status`。

`task_wait` 默认 `wait_policy=all`，但这是“调用了 `task_wait` 后”的等待语义，并不意味着主 Agent 一定会调用。

当前设计的优点是：

- `task_wait` 超时只表示本次等待预算耗尽。
- 超时不会杀掉子 Agent。
- 超时不会销毁 channel/futex。
- 主 Agent 可以继续用相同 `task_id` 调 `task_wait` 续等。

风险是：

- 未看到 runtime 在主 Agent final answer 前强制检查 pending task。
- 如果主 Agent `task_spawn` 后直接最终回答，子 Agent 仍可能在后台跑。
- 当前主要依赖 prompt/tool description 约束模型，而不是 runtime invariant。

### 建议

- 增加 turn-finalization barrier：如果当前 turn 有未收集的 `task_spawn`，默认禁止 final answer。
- 或给 `task_spawn` 增加参数：
  - `join_required: true` 默认值。
  - `detached: true` 必须显式声明才允许不等待。
- final 前自动注入 pending task 状态，提醒模型必须 `task_wait` / `task_status` 或显式 detach。

## 3. 子 Agent 报错怎么办？

当前有错误通路，但主要是把错误作为结果返回给主 Agent，而不是强制失败整个主 Agent turn。

### 同步 `task`

同步子 Agent 内部通过 `run_turn(...).map_err(...)` 捕获错误。

等待结束后会把结果归类为：

- `COMPLETED`
- `FAILED`
- timeout / cancel 类状态

相关实现见 `src/bin/ai/driver/tools/sync_task.rs:170-190`、`src/bin/ai/driver/tools/sync_task.rs:271-274`。

### 异步 `task_spawn`

后台 dispatch 会把结果写成 JSON：

- `status: completed` 或 `failed`
- `error: ...`

相关实现见 `src/bin/ai/driver/background_dispatch.rs:260-262`。

如果后台执行失败，还会记录 scheduler outcome 并 terminate/cleanup，见 `src/bin/ai/driver/background_dispatch.rs:297-298`。

### `task_wait` 收集阶段

`task_wait` 读取 result channel 后：

- 有结果：格式化返回并清理 channel/futex。
- 进程仍在跑：返回 pending。
- 进程不 pending 且没有结果：作为 failed-without-output 处理并清理资源。

相关实现见 `src/bin/ai/tools/task_tools.rs:1205-1220`。

### 问题

- 对主 Agent 来说，子 Agent failed 通常只是 `task_wait` 输出中的一个失败结果。
- 主 Agent 可以忽略这个失败，继续给用户看似成功的答案。
- 没有强制“所有子任务成功才允许继续”的语义。
- 缺少统一错误分类、retry 策略和 degraded mode 策略。

### 建议

- `task_wait` 返回更明确的结构化摘要：`completed_count` / `failed_count` / `pending_count`。
- 如果 `failed_count > 0`，工具输出应高亮提示主 Agent 必须处理。
- 增加 `require_success` 参数：
  - `require_success=true` 时，只要有子 Agent failed，当前工具调用以 Err 返回。
  - `require_success=false` 时保留现在的 batch 汇总语义。
- 对 retryable 错误做自动 retry 或至少分类，例如 model timeout、tool error、panic、cancel、quota exceeded。

## 4. 子 Agent 是否会继续生成嵌套子 Agent？如何禁止深度嵌套？

会。当前设计默认允许嵌套，但有最大深度上限。

代码中定义：

```rust
pub(crate) const MAX_SUBAGENT_SPAWN_DEPTH: usize = 2;
```

见 `src/bin/ai/tools/task_tools.rs:33-35`。

当前语义是：

- `depth=1`：顶层 Agent 直接 spawn 子 Agent。
- `depth=2`：子 Agent 再 spawn 孙 Agent。
- 超过该深度后拒绝继续委派。

所以当前行为是：

- 主 Agent 可以 spawn 子 Agent。
- 子 Agent 可以继续 spawn 孙 Agent。
- 孙 Agent 再 spawn 会被拒绝。
- `agent_team` 复用 `task_spawn/task_wait` substrate，因此也应受同一深度机制约束。

如果产品预期是“完全禁止子 Agent 再开子 Agent”，当前 `MAX_SUBAGENT_SPAWN_DEPTH=2` 不满足，需要更硬的策略。

### 简单方案

把：

```rust
MAX_SUBAGENT_SPAWN_DEPTH = 2
```

改成：

```rust
MAX_SUBAGENT_SPAWN_DEPTH = 1
```

这样只允许顶层 Agent spawn 一层子 Agent。

### 更清晰的方案

在准备子任务时直接判断：

```rust
if current_depth > 0 {
    return Err("Nested subagents are disabled".to_string());
}
```

这种方式语义更直接：不是“最大深度为 1”，而是“子 Agent 不允许再委派”。

### 更完整的方案

同时做两层防护：

1. 工具层隐藏。
   - 当 `SUBAGENT_DEPTH > 0` 时，不给子 Agent 暴露 `task` / `task_spawn` / `agent_team`。
   - 这样模型不会被诱导去调用不存在的工具。

2. runtime 层拒绝。
   - 即使工具被错误暴露，`prepare_subagent_task` 也拒绝 nested spawn。
   - prompt/tool exposure 不是安全边界，runtime 拒绝仍然必须保留。

推荐采用“工具层隐藏 + runtime 层拒绝”的双保险。

## 风险优先级

1. 异步 `task_spawn` 没有 final 前强制 join。
   - 主 Agent 可以忘等，后台子任务继续跑，结果无人消费。

2. 默认继承 `history` / `cwd` 太宽。
   - 子 Agent 容易拿到过多上下文。
   - 并发写项目 cwd 存在冲突风险。

3. 错误只是普通输出，缺少强制处理机制。
   - 子 Agent failed 后，主 Agent 可能继续给用户一个看似成功的答案。

4. 嵌套允许到 depth=2。
   - 如果产品预期是“子 Agent 不能再派生”，当前实现不满足。

## 建议改造优先级

1. 为 `task_spawn` 增加 pending-task finalization barrier。
2. 默认 `inherit.history=false`，由主 Agent 显式传必要上下文。
3. 子 Agent 默认隐藏 `task` / `task_spawn` / `agent_team`，除非显式允许。
4. 把 nested policy 做成配置：
   - `max_subagent_depth = 1`
   - 或 per-agent/per-task `allow_nested_subagents = false`
5. `task_wait` 增加强结构化失败摘要和 `require_success` 参数。
6. 对共享 cwd 的并发写操作加锁，或默认使用 scratch cwd。

## 验证记录

本 review 没有修改 Rust 代码，只新增文档。

此前已运行聚焦测试：

```bash
cargo test --bin a task_tools -- --nocapture
```

结果：

```text
test result: ok. 19 passed; 0 failed; 0 ignored
```

未运行全量 `cargo test`，因为本次是 review 文档整理，且项目指引要求只验证相关范围。