# `a.rs` Agent 深度 Review：长期记忆 & 自我学习/进步

> Review 日期: 2026-05-22

## 整体架构评价

架构设计相当清晰，采用了分层解耦的思路：

```
Knowledge (面向用户的知识)  ←→  Memory (Agent 内部行为记忆)
         ↓                            ↓
    JsonlStore  ←—KnowledgeSync—→  VectorStore (SQLite)
         ↑                            ↑
    retrieval/ (BM25+语义混合检索)    indexing/ (embedding+similarity)
         ↑
    reflection/ (自我反思 + Critic&Revise + Project Writeback)
    thinking/   (思维树 + 目标管理 + 验证循环 + 经验泛化)
```

代码质量整体较高，注释充分，错误处理也做得比较细致。但以下几个问题值得关注：

---

## 🔴 严重问题 (P0)

### 1. `project_memory` 分类断裂 — Writeback 存了但 Auto-Recall 召回不了

**位置**: `knowledge/retrieval/recall.rs:357-362` vs `driver/reflection/writeback.rs:160`

**问题**: Writeback 模块把所有项目知识以 `category: "project_memory"` 写入存储，但知识系统的 `Category` 枚举里根本没有 `"project_memory"` 这个变体（只有 `ProjectInfo`、`ProjectWriteback` 等）。这意味着：

```rust
// knowledge/types.rs - from_str 里没有 "project_memory" 匹配
"project_writeback" => Self::ProjectWriteback,
_ => Self::Other,  // "project_memory" 会落到这里!
```

而 `should_include_in_auto_recall()` 只允许四个类别：

```rust
// knowledge/retrieval/recall.rs:357
fn should_include_in_auto_recall(entry: &KnowledgeEntry) -> bool {
    matches!(
        entry.category_enum(),
        Category::UserMemory | Category::ProjectInfo | Category::Architecture | Category::DecisionLog
    )
}
```

**后果**: 所有 writeback 保存的项目知识虽然持久化了，但**永远不会被自动召回到 system prompt 中**。这是长期记忆功能最核心的一个断裂——学了但记不起来。

**修复建议**: 在 `Category::from_str` 里加上 `"project_memory" => Self::ProjectWriteback`（或新增一个 `ProjectMemory` 变体），并在 `should_include_in_auto_recall` 中加入该类别。

---

### 2. `delete_by_id` 使用非原子写入 — 崩溃时可能丢失整个记忆文件

**位置**: `tools/storage/memory_store.rs:985`

```rust
// 这里用的是直接 write，而非 atomic_write_file
std::fs::write(&self.path, output)
    .map_err(|e| format!("Failed to write memory file: {}", e))?;
```

而同文件的 `enforce_max_entries`（line 924）已经修复为使用 `atomic_write_file`，注释里也明确指出了这个问题：

```rust
// 修复点 P1-1：原实现 `fs::write(&path, output)` 是 truncate-then-write，
// 中途崩溃会留下不完整的主文件。改成 tmp+rename，文件系统层面原子。
```

`delete_by_id` 漏掉了这个修复，同样 `JsonlStore::delete_by_id`（jsonl_store.rs:153）也有同样的问题。

---

### 3. 经验缓冲区 (Experience Buffer) 仅存在于内存 — 重启后泛化学习丢失

**位置**: `driver/thinking/generalization.rs:39`

```rust
pub struct ExperienceGeneralizer {
    principles: Vec<GeneralizedPrinciple>,
    pub(crate) experience_buffer: Vec<RawExperience>,  // ← 只活在内存中!
    max_buffer_size: usize,
    min_experiences_for_generalization: usize,  // 默认 3
}
```

`principles` 在 `new()` 时从存储加载（`load_principles_from_store`），但 `experience_buffer` 不会持久化。每次 Agent 重启后，buffer 清空，必须重新积累 ≥3 条经验才能触发下一次泛化。对于交互不太频繁的场景，**泛化机制可能永远无法触发**。

**修复建议**: 将 `experience_buffer` 持久化到一个独立的 JSONL 文件或存储 category 中。

---

### 4. 所有 self_note 优先级硬编码为 255 (永久) — 反思噪声无限累积

**位置**: `driver/reflection/background.rs:351`

```rust
let entry = AgentMemoryEntry {
    // ...
    category: "self_note".to_string(),
    priority: Some(255),  // ← 所有反思笔记都是"永久"
    // ...
};
```

而在 `types.rs:97`，`SelfNote` 被定义为 `ShortLived`：

```rust
Self::SelfNote => KnowledgeType::ShortLived,
```

同时在 `is_permanent_memory` 中，所有 guideline 类别（包括 self_note）都豁免 GC。

**后果**: 每一轮对话的自动反思笔记都以最高优先级永久保存，永远不会被淘汰。长时间使用后，记忆文件会被大量低质量的 self_note 淹没，挤占真正有价值的记忆空间。

**修复建议**: 将 self_note 的默认优先级降为 150-200，或者在 `maintain_after_append` 中对 self_note 做专门的衰减/淘汰策略。

---

## 🟡 中等问题 (P1)

### 5. Writeback 条目不同步到向量索引

**位置**: `driver/reflection/writeback.rs:168`

```rust
store.append(&entry)?;  // 只写 JSONL
// ← 没有调用 sync_entry_to_vector 或类似函数
```

`sync_entry_to_vector` 函数存在（`knowledge/sync/knowledge_sync.rs:31`）但 writeback 流程没有调用它。这意味着通过向量搜索（semantic search）找不到 writeback 保存的项目知识。

---

### 6. 泛化模块的"语义分组"其实只是 category+tags 拼接

**位置**: `driver/thinking/generalization.rs:378-399`

```rust
fn semantic_group_key(&self, exp: &RawExperience) -> String {
    let category_key = exp.category.to_lowercase();
    let tag_key = exp.tags.iter()
        .map(|t| t.to_lowercase())
        // ...
    if tag_key.is_empty() { category_key } else {
        format!("{}:{}", category_key, tag_key)
    }
}
```

函数名暗示语义相似性，但实际只是字符串拼接。两条内容相似但 tag 不同的经验会被分到不同组，无法被泛化。这严重削弱了"从经验中学习通用规律"的能力。

---

### 7. `AgentMemoryEntry` 存在两份定义，有漂移风险

- `knowledge/entry.rs` 中的 `KnowledgeEntry`（alias `AgentMemoryEntry`）— **没有** `owner_pid`/`owner_pgid`
- `tools/storage/memory_store.rs:156` 中的 `AgentMemoryEntry` — **有** `owner_pid`/`owner_pgid`

两者结构不同，通过 `pub type` 别名试图兼容，但实际上是两个独立的 struct。如果序列化/反序列化交叉使用，会导致字段丢失。

---

### 8. `generalized_principle` 重复追加导致文件膨胀

**位置**: `driver/thinking/generalization.rs:215-238`

每次 principle 被 reinforce（加强），`persist_principle` 都会 `store.append()` 一条新记录（同样的 ID）。JSONL 文件会积累大量同一 principle 的旧版本。虽然 `load_principles_from_store` 有去重逻辑，但文件本身不断膨胀。

---

## 🟢 小问题 (P2)

### 9. `sync_entry_to_vector` 参数 `_jsonl_store` 未使用

```rust
pub fn sync_entry_to_vector(
    _jsonl_store: &JsonlStore,  // ← 未使用
    vector_store: &dyn VectorStoreSync,
    entry: &KnowledgeEntry,
) -> Result<(), String> {
```

属于接口设计不干净，虽然不影响功能。

### 10. RecallBundle 缓存的 `.tap()` 使用了自定义 trait

**位置**: `driver/reflection/recall.rs:230-236`

自定义了一个 `Tap` trait 来实现 `.tap()` 方法，而不是用 `itertools` 或标准库。虽然功能正确，但属于不必要的重复造轮子。

### 11. Vector Store 的 `hybrid_search` 中 `entry_map` 构建方式

```rust
let entry_map: FastMap<String, VectorEntry> = all_entries
    .drain(..)
    .map(|(id, entry, _)| (id, entry))
    .collect();
```

这里 `drain` 之后再遍历 `all_ids`，但 `all_ids` 包含了 BM25 结果中可能有而向量结果中没有的 ID。如果 BM25 返回了一个向量库中不存在的 ID，`entry_map.get(&id)` 会返回 `None`，被静默跳过。这不是 bug 但可能导致结果数量少于 `limit`。

---

## ✅ 做得好的地方

1. **原子写入 (`atomic_write_file`)**: `enforce_max_entries` 和 `rotate_if_exceeds` 都用了 tmp+rename 的原子写入策略
2. **文件锁 (`with_memory_file_lock`)**: 所有写操作都经过文件锁保护
3. **SQLite WAL 模式**: VectorStore 使用了 WAL + NORMAL sync，并发性能不错
4. **FTS5 加速搜索**: `MemoryIndex` 用 SQLite FTS5 做快速候选过滤，避免全文件扫描
5. **混合检索策略**: BM25 + embedding cosine similarity 的混合方案比较成熟
6. **Rotate 时保留永久条目**: 归档时主动把 permanent 条目提取回主文件
7. **Critic & Revise 机制**: 带 gate 模型的自我修正流程设计合理
8. **ThinkingOrchestrator**: Tree of Thoughts + Verification + Goal-directed 三种模式的编排比较完整

---

## 修复优先级建议

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **P0-1** | #1 project_memory 分类断裂 | 长期记忆核心功能失效 |
| **P0-2** | #4 self_note 全 255 优先级 | 记忆文件无限膨胀 |
| **P0-3** | #2 delete_by_id 非原子写 | 数据丢失风险 |
| **P1-1** | #3 experience buffer 不持久化 | 跨会话学习断裂 |
| **P1-2** | #5 writeback 不同步向量索引 | 语义搜索召回不全 |
| **P1-3** | #6 语义分组只是字符串拼接 | 泛化学习效果差 |
| **P1-4** | #7 两份 AgentMemoryEntry | 维护风险 |
