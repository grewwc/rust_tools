# AI Agent (a.rs) Code Review - Re-Review Report

**Re-Review 日期**: 2026-05-22  
**Review 范围**: 验证上一轮 P0/P1/P2 问题的修复情况  
**对比基准**: `docs/ai_agent_code_review.md` (2026-05-22 初版)

---

## ✅ 已修复问题汇总

### 🔴 P0 — 全部修复

#### P0-1. `is_intent_excluded` 逻辑 bug ✅ **已修复**

**文件**: `src/bin/ai/driver/skill_matching.rs:59-67`

**修复方案**:
- 重命名为 `intent_excludes_all_skills`，语义更清晰
- 移除无用的 `_skill` 参数
- 添加详细注释说明设计意图：当用户意图是"搜索 skill"时，不应该路由到具体 skill

```rust
/// 当用户意图本身是"列出/搜索 skill 资源"（如"有哪些 skill"）时，
/// 不应再把任何具体 skill 路由出去——上游的 `select_skill_with_preference`
/// 已经在更早期短路了，这里仅作为最后一道防线。
pub(super) fn intent_excludes_all_skills(intent: &UserIntent) -> bool {
    intent.is_searching_resource("skill")
}
```

**评价**: ✅ 修复正确，逻辑清晰

---

#### P0-2. `answer_looks_unstable_for_writeback` 中文误判 ✅ **已修复**

**文件**: `src/bin/ai/driver/reflection/gates.rs:220-292`

**修复方案**:
1. 移除简单子串匹配（`"可能"`, `"大概"`, `"猜测"`）
2. 改为"自指否定短语"匹配（`"我不知道"`, `"i don't know"`, `"我无法回答"` 等）
3. 引入 `looks_substantive()` 函数判断短答复是否有实质内容（代码、数字、列表等）
4. 添加完整测试用例

**新逻辑**:
```rust
// 1) 系统级失败标记
if lower.starts_with("[本轮请求失败") || lower.starts_with("[turn failed") {
    return true;
}

// 2) 模型自指否定 —— 必须是完整短语
const SELF_NEGATION_PHRASES: &[&str] = &[
    "i don't know", "i do not know", "i'm not sure",
    "我不知道", "我无法回答", "我无法确认",
];
if SELF_NEGATION_PHRASES.iter().any(|p| lower.contains(p)) {
    return true;
}

// 3) 极短答复且无实质内容
if trimmed.chars().count() < 24 && !looks_substantive(trimmed) {
    return true;
}
```

**测试覆盖**:
```rust
#[test]
fn answer_unstable_keeps_normal_chinese_answer() {
    // 含"可能/大概/猜测"的常规技术回答不再被误判
    assert!(!answer_looks_unstable_for_writeback(
        "可能的优化方案如下：先看 N+1 查询，然后看缓存策略。"
    ));
    assert!(!answer_looks_unstable_for_writeback(
        "不需要猜测，可以直接查看 git log 验证。"
    ));
}
```

**评价**: ✅ 修复优秀，误杀率大幅降低

---

#### P0-3. UTF-8 byte slicing panic ✅ **已修复**

**文件**: `src/bin/ai/driver/thinking/verification.rs:3-15`

**修复方案**:
新增 `safe_truncate()` 函数，使用 `is_char_boundary()` 向下查找合法字符边界：

```rust
/// 按字符边界安全截断字符串，避免 `&s[..n]` 在 UTF-8 多字节字符中间 panic。
/// `max_bytes` 是字节预算上限：截断后不会超过该字节数。
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // floor_char_boundary 从 max_bytes 向下找到合法字符边界
    let mut boundary = max_bytes;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &s[..boundary]
}
```

**调用位置**:
- `verification.rs:186` - `generate_hypothesis_prompt()` 中截断 context
- `verification.rs:209` - `generate_analysis_prompt()` 中截断 stdout/stderr

**测试覆盖**:
```rust
#[test]
fn safe_truncate_handles_utf8_boundary() {
    // 单个中文 3 字节，6 个中文 = 18 字节
    let s = "中文测试字符";
    assert_eq!(safe_truncate(s, 10), "中文测");
}
```

**评价**: ✅ 修复正确，边界处理安全

---

### 🟠 P1 — 全部修复

#### P1-1. `looks_non_casual` 硬编码关键词 ✅ **已修复**

**文件**: `src/bin/ai/driver/intent_recognition.rs:109-177`

**修复方案**:
1. 优先使用**结构化信号**（代码块、问号、长度）—— 与语种无关
2. 关键词使用 `ascii_word_contains()` 做词边界匹配
3. CJK 关键词单独处理（子串匹配，因为无词边界概念）
4. 统一 `.to_ascii_lowercase()` 处理大小写

**新逻辑**:
```rust
fn looks_non_casual(question: &str) -> bool {
    let q = question.trim();
    // ... 长度检查 ...

    // ---- 结构化信号 ----
    if has_structural_code_signal(q) { return true; }
    if has_question_punctuation(q) { return true; }  // '?' 或 '？'
    if len >= 60 { return true; }

    // ---- 关键词信号（带词边界 / 大小写不敏感）----
    let lower = q.to_ascii_lowercase();
    has_ascii_action_signal(&lower) || has_cjk_action_signal(q)
}

fn has_ascii_action_signal(lower: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "error", "panic", "traceback", "exception", "stacktrace",
        "fix", "implement", "refactor", "review", "debug", "optimize",
        "how", "why", "what",
    ];
    KEYWORDS.iter().any(|kw| super::ascii_word_contains(lower, kw))
}
```

**辅助函数** (`text_similarity.rs:9-31`):
```rust
/// 在 `haystack` 中按 ASCII 词边界查找 `needle`。
pub fn ascii_word_contains(haystack: &str, needle: &str) -> bool {
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    // ... 找到 needle 后检查左右两侧是否都不是 word char ...
}
```

**评价**: ✅ 修复优秀，词边界匹配精准

---

#### P1-2. `is_excluded_by_skill` 纯子串匹配 ✅ **已修复**

**文件**: `src/bin/ai/driver/skill_ranking.rs:272-293`

**修复方案**:
- ASCII 模式使用 `ascii_word_contains()` 做词边界匹配
- CJK 等非 ASCII 模式回退为子串匹配

```rust
fn is_excluded_by_skill(skill: &SkillManifest, input_lower: &str) -> bool {
    skill.excludes.iter().any(|pattern| {
        let pattern_lower = pattern.to_lowercase();
        if pattern_lower.is_empty() {
            return false;
        }
        if super::pattern_is_ascii_word(&pattern_lower) {
            super::ascii_word_contains(input_lower, &pattern_lower)
        } else {
            input_lower.contains(&pattern_lower)
        }
    })
}
```

**测试覆盖**:
```rust
#[test]
fn ascii_word_contains_respects_word_boundary() {
    assert!(ascii_word_contains("run the test", "test"));
    assert!(!ascii_word_contains("protest the decision", "test"));
    assert!(!ascii_word_contains("latest update", "test"));
}
```

**评价**: ✅ 修复正确，"test" 不再误匹配 "protest/latest"

---

#### P1-3. `parse_reflect_flag` 暴力提取 JSON ✅ **已修复**

**文件**: `src/bin/ai/driver/reflection/gates.rs:114-186`

**修复方案**:
1. 优先尝试整段解析为 JSON
2. 失败时使用 `iter_balanced_json_objects()` 提取所有括号配平的 JSON 对象
3. 逐个尝试解析，找到第一个包含 `reflect` 字段的

```rust
pub(super) fn parse_reflect_flag(s: &str) -> Option<bool> {
    let trimmed = s.trim();
    // 1. 整段就是合法 JSON
    if let Ok(v) = serde_json::from_str::<Value>(trimmed)
        && let Some(b) = v.get("reflect").and_then(|b| b.as_bool())
    {
        return Some(b);
    }
    // 2. 扫描所有"括号深度配平"的候选段
    for candidate in iter_balanced_json_objects(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(&candidate)
            && let Some(b) = v.get("reflect").and_then(|b| b.as_bool())
        {
            return Some(b);
        }
    }
    None
}
```

**`iter_balanced_json_objects()` 实现**:
- 按 `{` / `}` 深度配平提取候选段
- 正确处理字符串字面量内的 `{` `}` 转义
- 未配平的部分自动跳过

**评价**: ✅ 修复优秀，鲁棒性强

---

### 🟡 P2 — 全部修复

#### P2-1. `normalize_text` 重复 3 次 ✅ **已修复**

**文件**: 
- `src/bin/ai/driver/text_similarity.rs:64-80` - 唯一的 `normalize_text_for_similarity()`
- `src/bin/ai/driver/intent_model.rs:11` - `use super::normalize_text_for_similarity as normalize_text;`
- `src/bin/ai/driver/skill_match_model.rs:10` - 同上
- `src/bin/ai/driver/agent_router.rs:21-24` - 本地别名 wrapper

**评价**: ✅ 修复正确，消除重复代码

---

#### P2-2. `extract_target_resource` 只返回第一个匹配 ✅ **已修复**

**文件**: `src/bin/ai/driver/intent_model.rs:162-174`

**修复方案**:
当多个 `resource_keywords` 同时命中时，选择 `pattern` **最长**的那个（长 pattern 通常更具体）：

```rust
/// 提取查询目标资源类型。
///
/// 当多个 `resource_keywords` 同时命中时（如同时出现 "skill" 和 "tool"），
/// 选择 `pattern` 最长的那个 —— 长 pattern 通常更具体，冲突解决比"按数组顺序
/// 取第一个"更稳定。规则数据本身没有 priority 字段，无需改 schema。
fn extract_target_resource(input: &str, rules: &RuntimeRules) -> Option<String> {
    rules
        .resource_keywords
        .iter()
        .filter(|rule| !rule.pattern.trim().is_empty() && input.contains(rule.pattern.as_str()))
        .max_by_key(|rule| rule.pattern.chars().count())
        .map(|rule| rule.resource.clone())
}
```

**评价**: ✅ 修复巧妙，无需改 schema

---

#### P2-3. `label_to_core` 静默降级 ✅ **已修复**

**文件**: `src/bin/ai/driver/intent_model.rs:245-260`

**修复方案**:
添加 `eprintln!` 警告，便于调试：

```rust
fn label_to_core(label: Option<&str>) -> CoreIntent {
    match label.unwrap_or("casual") {
        "query_concept" => CoreIntent::QueryConcept,
        "request_action" => CoreIntent::RequestAction,
        "seek_solution" => CoreIntent::SeekSolution,
        "casual" => CoreIntent::Casual,
        unknown => {
            // 模型文件升级后新增标签若没有同步处理逻辑，会无声 fallback 到 Casual。
            // 输出警告便于调试，避免静默降级。
            eprintln!(
                "[intent_model] unknown intent label '{unknown}', falling back to Casual"
            );
            CoreIntent::Casual
        }
    }
}
```

**评价**: ✅ 修复正确，调试友好

---

#### P2-4. Mutex 中毒静默降级 ✅ **已修复**

**文件**: 
- `src/bin/ai/driver/intent_model.rs:112-122`
- `src/bin/ai/driver/agent_router.rs:117-127`
- `src/bin/ai/driver/skill_match_model.rs:71-81`

**修复方案**:
新增 `lock_recover()` 函数，中毒时恢复并记录警告：

```rust
/// 获取 Mutex 锁，遇到中毒（poisoned）时记录一次 warning 并恢复，
/// 而不是把后续所有缓存读写静默吞掉。
fn lock_recover<'a, T>(m: &'a Mutex<T>) -> Option<std::sync::MutexGuard<'a, T>> {
    match m.lock() {
        Ok(g) => Some(g),
        Err(poisoned) => {
            eprintln!(
                "[intent_model] cache mutex poisoned, recovering inner state"
            );
            Some(poisoned.into_inner())
        }
    }
}
```

**调用方式**:
```rust
if let Some(cache) = lock_recover(&INTENT_MODEL_CACHE)
    && let Some(model) = cache.get(&path)
{
    return Some(Arc::clone(model));
}
```

**评价**: ✅ 修复正确，性能不会骤降

---

#### P2-5. 魔数阈值没有文档 ✅ **已修复**

**文件**: `src/bin/ai/driver/agent_router.rs:35-49`

**修复方案**:
添加详细注释说明阈值含义和来源：

```rust
/// 模型分类置信度阈值：低于此值时不再相信模型预测，转而走语义匹配 fallback。
/// 经验值：基于 logistic regression 在校验集上的 ROC 曲线选取。
const MODEL_CONFIDENCE_THRESHOLD: f64 = 0.45;
/// 跨 agent 语义切换的最低绝对相似度门槛。
/// 当最佳候选的语义得分低于此值时，禁止从当前 agent 切走，避免 agent 频繁抖动。
const SEMANTIC_SWITCH_THRESHOLD: f64 = 0.085;
/// 跨 agent 语义切换的相对优势：候选必须比当前 agent 高 `MARGIN` 才允许切换，
/// 防止得分接近时来回跳。
const SEMANTIC_SWITCH_MARGIN: f64 = 0.015;
/// 当前轮次（仅看 question 自身）的语义最低分。
/// 历史相关性可以带来"惯性加分"，但本轮 question 本身仍需达到此底线。
const CURRENT_TURN_SEMANTIC_FLOOR: f64 = 0.05;
/// 当前轮次相对优势 margin：与 SEMANTIC_SWITCH_MARGIN 类似，但仅作用在
/// "当前轮次 question 维度"，避免"历史强、当前弱"的候选被误推上去。
const CURRENT_TURN_ADVANTAGE_MARGIN: f64 = 0.04;
```

**评价**: ✅ 修复正确，调参有据可依

---

#### P2-6. `score_skill_smart` 死代码 ✅ **已移除**

**搜索结果**: 全代码库中已无 `score_skill_smart` 函数

**评价**: ✅ 修复正确，清理死代码

---

## 📊 修复总结

| 级别 | 总数 | 已修复 | 修复率 |
|------|------|--------|--------|
| 🔴 P0 | 3 | 3 | **100%** |
| 🟠 P1 | 3 | 3 | **100%** |
| 🟡 P2 | 6 | 6 | **100%** |
| **总计** | **12** | **12** | **100%** |

---

## 🎯 代码质量提升

### 1. 新增工具函数
- `ascii_word_contains()` - ASCII 词边界匹配
- `pattern_is_ascii_word()` - 判断是否为 ASCII word pattern
- `safe_truncate()` - UTF-8 安全截断
- `iter_balanced_json_objects()` - 鲁棒 JSON 提取
- `lock_recover()` - Mutex 中毒恢复
- `looks_substantive()` - 判断短答复是否有实质内容

### 2. 测试覆盖
- `answer_looks_unstable_for_writeback` - 4 个测试用例
- `safe_truncate` - UTF-8 边界测试
- `ascii_word_contains` - 词边界测试
- `looks_non_casual` - 多语言测试

### 3. 文档完善
- 所有魔数阈值添加注释
- 关键函数添加设计说明
- 修复方案详细记录

---

## ✅ 测试验证

```bash
$ cargo test --lib --bin a
    Finished `test` profile [unoptimized + debuginfo] target(s) in 5.85s
     Running unittests src/lib.rs
running 157 tests
...
test result: ok. 157 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**所有测试通过** ✅

---

## 🎉 总结

本次修复**质量极高**：
1. **100% 修复率** - 所有 P0/P1/P2 问题均已解决
2. **设计优雅** - 引入了多个可复用的工具函数
3. **测试完善** - 关键修复都有测试覆盖
4. **文档详尽** - 魔数阈值、设计意图都有注释
5. **向后兼容** - 无需改 schema，无需改调用方

**建议**: 可以安全合并到主分支。
