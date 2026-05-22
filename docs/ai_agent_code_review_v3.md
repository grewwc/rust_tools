# AI Agent (a.rs) Code Review - V3 Report

**Review 日期**: 2026-05-22  
**Review 范围**: Agent 层架构审视 — 识别与 AIOS Kernel 功能重复、逻辑问题及优化机会  
**对比基准**: `docs/ai_agent_code_review_v2.md` (前一轮修复验证)

---

## 📋 问题总览

| 严重度 | # | 问题 | 类别 |
|---|---|---|---|
| 🔴 高 | 1 | `GLOBAL_OS` 与 `App.os` 双重引用同一 Kernel | 架构/一致性 |
| 🔴 高 | 2 | `TASK_REGISTRY` 与 kernel process table 状态重复 | AIOS 功能重复 |
| 🟡 中 | 3 | `prune_completed_tasks` O(n²) 性能问题 | 性能 |
| 🟡 中 | 4 | 文本相似度函数存在两套独立实现 | 代码重复 |
| 🟡 中 | 5 | `quota_turns` 扣减逻辑三处重复 | 代码重复 / 应下沉 kernel |
| 🟡 中 | 6 | `epoll_wait_many` 在 agent 层重写了 kernel 等待原语 | AIOS 功能重复 |
| 🟡 中 | 7 | `with_os_kernel` 每次调用多余锁操作 | 性能 |
| 🟢 低 | 8 | foreground/background 执行逻辑大量重复 | 代码重复 |
| 🟢 低 | 9 | `select_subagent` 每次调用重新加载所有 agents | 性能 |
| 🟢 低 | 10 | wake-up prompt 模板在两个地方硬编码重复 | 代码重复 |

---

## 🔴 P0 — 高严重度

### 问题 1：`GLOBAL_OS` 与 `App.os` 双重引用同一 Kernel — 数据竞争风险

**文件**: `driver/mod.rs:597-598` + `tools/os_tools.rs:7`

**现状**:

```rust
// driver/mod.rs — run() 函数中
let os_arc = new_local_kernel();
crate::ai::tools::os_tools::init_os_tools_globals(os_arc.clone()); // 存到全局 static
// ...
app.os = os_arc;  // 同时存到 App 结构体
```

`App.os` 和 `GLOBAL_OS` 持有 **同一个 `SharedKernel`（`Arc<Mutex<...>>`）** 的两份引用，导致：

- **锁竞争路径不一致**：driver 层通过 `app.os.lock()` 拿锁，tool 层通过 `GLOBAL_OS.lock()` 拿锁，两条代码路径可能在同一个 async 上下文中交叉使用
- **维护困惑**：新增代码不知道应该用 `app.os` 还是 `GLOBAL_OS`
- **潜在死锁风险**：如果某条路径在持有 `App.os` 锁时调用 tool，而 tool 又试图获取 `GLOBAL_OS` 锁

**建议**:

统一为一个访问入口。两种方案：

1. **方案 A**: 把 `GLOBAL_OS` 改为从 `DRIVER_CTX` task_local 中获取，消除全局 static
2. **方案 B**: 让 `os_tools` 通过 kernel 的 `current_pid_provider` 回调机制拿到引用，消除双重全局状态

---

### 问题 2：`TASK_REGISTRY` 与 AIOS Kernel process table 状态重复

**文件**: `tools/task_tools.rs:154`

**现状**:

```rust
static TASK_REGISTRY: LazyLock<Mutex<FastMap<String, AsyncTaskEntry>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));
```

AIOS Kernel **已经维护了完整的进程表**（`os.list_processes()`、`os.get_process(pid)`），而 `TASK_REGISTRY` 又在 agent 层面维护了一套平行的 task 注册表。两套系统存储了大量重叠信息：

| TASK_REGISTRY 字段 | AIOS Kernel 已有的等价信息 |
|---|---|
| `pid` | kernel process table |
| `started_at` | kernel `created_at_tick` |
| `agent_name` | kernel process `name` |
| `description` | kernel process `goal` |
| state (running/completed) | kernel `ProcessState` |

**唯一真正 kernel 没有的字段**: `result_channel_id`、`completion_futex_addr`、`inherit`、`selection_explanation`。

**建议**:

把这些 agent 特有字段作为 kernel process 的 metadata/extension 存储（比如通过 `shm` 或 process annotation），而不是维护一个完整的并行注册表。这样可以：

- 消除 `prune_completed_tasks` 与 kernel 进程清理之间的同步问题
- 简化 `task_status` 实现（直接查 kernel process table）
- 避免 registry 与 kernel 状态不一致的风险

---

## 🟡 P1 — 中严重度

### 问题 3：`prune_completed_tasks` O(n²) 性能问题

**文件**: `tools/task_tools.rs:169-187`

**现状**:

```rust
fn prune_completed_tasks(registry: &mut FastMap<String, AsyncTaskEntry>) {
    if registry.len() <= MAX_TASK_REGISTRY_SIZE { return; }

    let mut oldest = registry.iter()
        .min_by_key(|(_, entry)| entry.started_at)
        .map(|(key, _)| key.clone());

    while registry.len() > MAX_TASK_REGISTRY_SIZE {
        // 每次循环都要全量遍历找 oldest → O(n²)
        let Some(key) = oldest.take() else { break; };
        registry.remove(&key);
        oldest = registry.iter()
            .min_by_key(|(_, entry)| entry.started_at)
            .map(|(next_key, _)| next_key.clone());
    }
}
```

每删一个 entry 都要 `O(n)` 全量遍历找最老的，删除 k 个就是 `O(n*k)`。虽然 `MAX_TASK_REGISTRY_SIZE=100` 不太大，但写法不优雅。

**建议**: 收集所有 key + started_at，排序后一次删除最老的 k 个：

```rust
fn prune_completed_tasks(registry: &mut FastMap<String, AsyncTaskEntry>) {
    if registry.len() <= MAX_TASK_REGISTRY_SIZE { return; }
    let mut entries: Vec<(String, Instant)> = registry
        .iter().map(|(k, v)| (k.clone(), v.started_at)).collect();
    entries.sort_by_key(|(_, t)| *t);
    let to_remove = registry.len() - MAX_TASK_REGISTRY_SIZE;
    for (key, _) in entries.into_iter().take(to_remove) {
        registry.remove(&key);
    }
}
```

---

### 问题 4：文本相似度函数存在两套独立实现

**位置对比**:

| 功能 | `driver/text_similarity.rs` | `knowledge/indexing/similarity.rs` |
|---|---|---|
| 文本归一化 | `normalize_text_for_similarity` | `norm_text`（不同算法：去空格 vs 保留空格） |
| Jaccard 相似度 | `jaccard_similarity_for_sets`（基于 SkipSet） | `jaccard`（基于 Vec/HashSet） |
| 字符 bigram | `char_ngram_set` (返回 SkipSet) | `bigrams` (返回 Vec<(char, char)>) |
| Dice 系数 | ❌ 不存在 | `dice_coefficient` |
| Cosine 相似度 | `cosine_tfidf_similarity`（基于 TF-IDF） | `cosine_similarity`（基于 f32 向量） |

两套实现用了 **不同的数据结构和算法**，但解决的是同一类问题。`text_similarity.rs` 用于 skill/agent routing，`knowledge/indexing/similarity.rs` 用于 knowledge/memory 检索。

**建议**:

- 统一到一个 `similarity` crate/module，提供不同精度层次的 API（fast approximate vs exact）
- 对于 `norm_text` 和 `normalize_text_for_similarity` 这种功能重叠但语义不同的函数，至少统一命名并注释区别
- `char_ngram_set` 和 `bigrams` 应该合并为一个通用的 n-gram 生成函数

---

### 问题 5：`quota_turns` 扣减逻辑在三处重复

**文件**: `driver/mod.rs` 的三个位置 (L751-768, L1046-1062, L1290-1305)

**现状**: 三段代码几乎一模一样：

```rust
if p.quota_turns > 0 {
    p.quota_turns -= 1;
}
if p.quota_turns == 0 {
    termination_result = "Terminated: Max LLM quota reached.".to_string();
}
if matches!(p.state, ProcessState::Waiting { .. }
    | ProcessState::Sleeping { .. }
    | ProcessState::Stopped)
{
    should_terminate = false;
}
```

**建议**:

提取为 kernel 层的方法：

```rust
impl Kernel {
    fn consume_quota(&mut self, pid: u32) -> QuotaOutcome {
        // QuotaOutcome { Exhausted, Pending, Ok }
    }
}
```

消除重复，且 quota 扣减本应是 AIOS kernel 的职责——agent 不应该自己管理 quota。

---

### 问题 6：`epoll_wait_many` 在 agent 层重写了 kernel 等待原语

**文件**: `tools/task_tools.rs:281-403`

**现状**: `epoll_wait_many` 函数实现了一个 **通用的多源等待机制**（支持 Channel + Event + Futex 混合等待），这本质上是 `epoll_wait` 的高级封装。

AIOS kernel 已经提供了：
- `epoll_create` / `epoll_ctl_add` / `epoll_wait`
- `wait_on_events` with `WaitPolicy`
- `futex_try_wait` / `futex_event_id`

但 agent 层的 `epoll_wait_many` 把这些低级原语重新组合成了一个高级 API。

**建议**:

这个函数应该下沉到 AIOS kernel 中作为 `epoll_wait_many` 或 `wait_on_sources` 系统调用，而不是在 agent tool 实现中自己编写。kernel 层实现还可以：
- 更好地与调度器集成
- 支持更高效的唤醒机制
- 统一的错误处理和超时管理

---

### 问题 7：`with_os_kernel` 每次调用多余锁操作

**文件**: `tools/task_tools.rs:204-220`

**现状**:

```rust
fn with_os_kernel<T>(f: impl FnOnce(&mut dyn Kernel) -> Result<T, String>) -> Result<T, String> {
    let shared = {
        let guard = GLOBAL_OS.lock()...; // 第一次锁
        guard.as_ref().cloned()...        // clone Arc
    };                                   // 释放第一次锁
    let mut kernel = shared.lock()...;   // 第二次锁
    f(kernel.as_mut())
}
```

每次调用都经历 **两次锁获取**：先锁 `GLOBAL_OS` 拿到 `Arc`，再锁 `Arc` 内部的 `Mutex` 拿到 kernel 引用。在高频率的 `task_wait` 场景中（每次 poll 多个 task），这是不必要的开销。

**建议**:

直接用 `DRIVER_CTX` task_local 中的 `App.os` 引用，避免全局 static 的间接寻址：

```rust
DRIVER_CTX.with(|ctx| {
    let app = ctx.borrow();
    let mut kernel = app.os.lock().unwrap();
    f(kernel.as_mut())
})
```

---

## 🟢 P2 — 低严重度

### 问题 8：`run_foreground_resume` 与 background task 执行逻辑大量重复

**文件**: `driver/mod.rs:676-775` vs `993-1083`

**现状**: `run_foreground_resume` 和 background task 的 `inner_fut` 逻辑几乎相同：

1. 构造 wake-up prompt
2. 设置 current pid
3. 调用 `run_turn`
4. 处理 quota 扣减
5. 处理终止/等待状态

**建议**:

提取共用函数：

```rust
async fn execute_process_turn(
    app: &mut App,
    mcp: &McpContext,
    skills: &[Skill],
    pid: u32,
    question: &str,
    termination_handler: impl FnOnce(&str, bool),
) -> TurnResult { ... }
```

---

### 问题 9：`select_subagent` 每次调用重新加载所有 agents

**文件**: `tools/task_tools.rs:542`

**现状**:

```rust
let all_agents = agents::load_all_agents();
```

`load_all_agents()` 每次都从磁盘重新读取所有 `.agent` 文件。在高频 `task_spawn` 场景下（比如并行启动 5 个子任务），这是 5 次磁盘 I/O。

**建议**:

从 `DRIVER_CTX` 的 `agent_manifests`（已在 `ensure_runtime_manifests_loaded` 中加载并缓存）获取，而不是每次重新从磁盘读。

---

### 问题 10：wake-up prompt 模板在两个地方硬编码重复

**文件**: `driver/mod.rs:692-697` 和 `911-916`

**现状**: 完全相同的 wake-up prompt 模板出现了两次：

```
"[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n...\n\nWake-up handling rules:\n- If the mailbox indicates async tool wake-up..."
```

**建议**:

提取为公共函数：

```rust
fn format_wakeup_prompt(pid: u32, goal: &str, messages: &str) -> String {
    format!(
        "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:...",
        pid, goal, messages
    )
}
```

---

## 📊 核心架构建议

### 问题本质

当前 agent 层承担了太多 **本应属于操作系统（AIOS kernel）的职责**，导致两个系统之间出现状态同步和逻辑重复：

```
┌─────────────────────────────────────┐
│          Agent Layer (a.rs)          │
│  ┌─────────────────────────────────┐│
│  │ TASK_REGISTRY (重复进程表)       ││  ← 应该用 kernel process table
│  │ quota_turns 扣减 (3处重复)      ││  ← 应该由 kernel 管理
│  │ epoll_wait_many (重写等待原语)   ││  ← 应该下沉到 kernel
│  │ GLOBAL_OS + App.os (双重引用)   ││  ← 应该统一访问入口
│  └─────────────────────────────────┘│
├─────────────────────────────────────┤
│        AIOS Kernel Layer             │
│  ┌─────────────────────────────────┐│
│  │ Process Table                   ││
│  │ Quota Management                ││
│  │ Epoll / Wait Primitives         ││
│  │ SharedKernel (Arc<Mutex<...>>)  ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### 推荐改进方向

1. **Agent 层做薄**：Agent 只负责 LLM 交互、prompt 构造、tool dispatch；进程管理、quota、等待原语全部下沉到 kernel
2. **消除全局 static**：`GLOBAL_OS` 和 `TASK_REGISTRY` 都应通过 `DRIVER_CTX` task_local 访问
3. **统一相似度库**：合并 `driver/text_similarity.rs` 和 `knowledge/indexing/similarity.rs`
4. **提取公共逻辑**：foreground/background 执行流程、wake-up prompt、quota 扣减等重复逻辑统一封装

---

## 🔄 与 V2 报告的关系

V2 报告主要验证了 V1 中 P0/P1/P2 问题的修复情况（均已修复）。本轮 V3 是 **新一轮架构审查**，聚焦于 agent 层与 AIOS kernel 层的职责边界问题，不涉及 V1/V2 中已修复的 bug。
