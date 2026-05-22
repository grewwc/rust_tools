# AI Agent (a.rs) Code Review

**Review 日期**: 2026-05-22  
**Review 范围**: `src/bin/a.rs` 及其依赖的整个 `src/bin/ai/` 模块  
**Review 重点**: 逻辑正确性、硬编码字符串匹配、生产环境可靠性

---

## 概览

| 级别 | 数量 | 主要类型 |
|------|------|----------|
| 🔴 P0 | 3 | 逻辑 bug、中文误判、UTF-8 panic |
| 🟠 P1 | 3 | 硬编码子串匹配做意图/排除判断 |
| 🟡 P2 | 6 | 重复代码、静默降级、魔数、死代码 |

---

## 🔴 P0 — 必须修复

### P0-1. `is_intent_excluded` 忽略了 `_skill` 参数

**文件**: `src/bin/ai/driver/skill_matching.rs:60-62`

```rust
pub(super) fn is_intent_excluded(_skill: &SkillManifest, intent: &UserIntent) -> bool {
    intent.is_searching_resource("skill")
}
```

**问题**:
- 函数签名接收了 `_skill` 参数但完全没用
- 当用户意图是"搜索 skill 资源"时，**所有** skill 都会被排除
- 包括专门用来列出/搜索 skill 的那个 skill 也会被排除
- 用户在问"有哪些可用的 skill"时，系统反而会拒绝路由到任何 skill

**建议**: 应该根据具体 skill 的 manifest 信息判断是否应该排除该 skill，而不是一刀切。

---

### P0-2. `answer_looks_unstable_for_writeback` — 中文子串匹配导致大面积误判

**文件**: `src/bin/ai/driver/reflection/gates.rs:148-166`

```rust
[
    "[本轮请求失败", "i'm sorry", "不确定", "可能", "猜测",
    "大概", "无法确认", "need to verify", "might be",
]
.iter().any(|needle| lower.contains(&needle.to_lowercase()))
```

**问题**:
- `"可能"` 会匹配到 **"可能的优化方案如下"**、**"这个函数可能有问题"** 这类完全正常的回答。技术回答里"可能"是个极高频词
- `"大概"` 会匹配 **"大概率可以编译通过"**、**"概要设计如下"**
- `"猜测"` 会匹配 **"不需要猜测，可以直接验证"**（否定语境也被误判）
- `answer.chars().count() < 40` 就判 unstable —— "好的，已修复" 这种稳定的短回答也会被拦截

**影响**: 这个函数的目的是防止把"不确定的回答"写入记忆/知识库。但当前的子串匹配方式误杀率太高，基本等于"中文技术回答里有不确定性词汇就不写入记忆"。

**建议**:
1. 使用分词或正则表达式做词边界匹配
2. 考虑否定语境（"不需要猜测" vs "只能猜测"）
3. 短回答阈值应该更宽松，或结合其他信号判断

---

### P0-3. UTF-8 byte slicing 会 panic

**文件**: `src/bin/ai/driver/thinking/verification.rs:172, 195-196`

```rust
// line 172
if context.len() > 2000 { &context[..2000] } else { context }

// line 195-196
if result.stdout_preview.len() > 600 { &result.stdout_preview[..600] } else { &result.stdout_preview }
if result.stderr_preview.len() > 600 { &result.stderr_preview[..600] } else { &result.stderr_preview }
```

**问题**:
- Rust 的 `&str[..N]` 是按字节索引的
- 如果第 2000 字节恰好在一个中文字符（3字节）或 emoji（4字节）中间，**直接 panic**
- 在 agent 处理中文代码注释、中文测试输出时尤其容易触发

**修复方案**:
```rust
// 方案 1: 用 chars() 迭代器
if context.len() > 2000 {
    &context.chars().take(2000).collect::<String>()
} else {
    context
}

// 方案 2: 用 get() 安全切片（如果字节边界不在字符中间）
context.get(..2000).unwrap_or(context)

// 方案 3: 用 char_indices 找到最近的字符边界
if context.len() > 2000 {
    let boundary = context.char_indices()
        .nth(2000)
        .map(|(i, _)| i)
        .unwrap_or(context.len());
    &context[..boundary]
} else {
    context
}
```

---

## 🟠 P1 — 硬编码字符串匹配做意图/路由判断

### P1-1. `looks_non_casual` — 手工关键词列表做意图升级

**文件**: `src/bin/ai/driver/intent_recognition.rs:103-130`

```rust
let has_code = q.contains("```") || q.contains("::") || q.contains("fn ");
let has_error_words = q.contains("error")
    || q.contains("Error")       // 大小写不一致
    || q.contains("panic")
    || q.contains("traceback");

let has_action_verbs = q.contains("fix")
    || q.contains("implement")
    || q.contains("how ")        // 尾部空格，漏了 "how." "how?" "how\n"
    || q.contains("why ");
```

**问题**:
- `"fn "` 会误匹配 `"fun "`、`"often "`（虽然后者不含空格，但容易遗漏边界情况）
- `"Error"` vs `"error"` 大小写不一致 —— 漏了 `"ERROR"`（全大写日志里很常见）
- `"how "` 和 `"why "` 要求后面跟空格，但 `"how?"`, `"how."`, `"why\n"` 都不会命中
- `"implement"` 会匹配 `"reimplementation"` 的子串
- 整体是一个 **关键词白名单**，无法扩展。用户说"重构"、"优化"、"review" 这些 action verb 都不会触发升级

**建议**:
1. 统一 `.to_lowercase()` 后再匹配
2. 使用正则表达式做词边界匹配：`\b(fix|implement|how|why)\b`
3. 考虑用 LLM 做意图分类，而不是硬编码关键词

---

### P1-2. `is_excluded_by_skill` — 纯子串匹配做排除

**文件**: `src/bin/ai/driver/skill_ranking.rs:272-280`

```rust
fn is_excluded_by_skill(skill: &SkillManifest, input_lower: &str) -> bool {
    skill.excludes.iter().any(|pattern| {
        let pattern_lower = pattern.to_lowercase();
        input_lower.contains(&pattern_lower)
    })
}
```

**问题**:
- 如果某个 skill 的 `excludes` 配了 `"test"`，那么 `"protest"`, `"latest"`, `"contest"`, `"attest"` 全部会被误排除
- 没有任何词边界检查

**修复方案**:
```rust
use regex::Regex;

fn is_excluded_by_skill(skill: &SkillManifest, input_lower: &str) -> bool {
    skill.excludes.iter().any(|pattern| {
        let pattern_lower = pattern.to_lowercase();
        let re = Regex::new(&format!(r"\b{}\b", regex::escape(&pattern_lower))).unwrap();
        re.is_match(input_lower)
    })
}
```

---

### P1-3. `parse_reflect_flag` — 暴力提取 JSON

**文件**: `src/bin/ai/driver/reflection/gates.rs:107-121`

```rust
let l = trimmed.find('{')?;
let r = trimmed.rfind('}')?;
let sub = &trimmed[l..=r];
```

**问题**:
- 如果 LLM 返回了 `Here is my analysis: {"context": "some stuff"} and the answer is {"reflect": true}`
- 会取从第一个 `{` 到最后一个 `}` 的整段子串
- 拼出一个不合法的 JSON，解析失败

**建议**:
1. 要求 LLM 只返回 JSON，不要附加说明
2. 或者用更鲁棒的 JSON 提取逻辑（比如尝试解析多个 `{}` 段）

---

## 🟡 P2 — 逻辑/设计问题

### P2-1. `normalize_text` 在 3 个文件里各 copy-paste 了一份

完全相同的函数分别出现在：
- `src/bin/ai/driver/intent_model.rs:223-243`
- `src/bin/ai/driver/agent_router.rs:178-198`
- `src/bin/ai/driver/skill_match_model.rs:95-115`

**问题**:
- 如果 normalize 逻辑有 bug（比如对 CJK 标点处理不当），需要改 3 个地方
- 违反 DRY 原则

**建议**: 提取到共享模块，如 `src/bin/ai/driver/text_utils.rs`

---

### P2-2. `extract_target_resource` 只返回第一个匹配的关键词

**文件**: `src/bin/ai/driver/intent_model.rs:146-152`

```rust
rules.resource_keywords.iter()
    .find(|rule| !rule.pattern.trim().is_empty() && input.contains(rule.pattern.as_str()))
    .map(|rule| rule.resource.clone())
```

**问题**:
- 如果用户输入 "帮我找个 review skill 的 tool"，`"skill"` 和 `"tool"` 都在文本中出现
- 但 `find()` 只返回在 `resource_keywords` 数组中先定义的那个
- 没有优先级或冲突解决机制

**建议**:
1. 在 `resource_keywords` 配置中增加 `priority` 字段
2. 或者返回所有匹配的资源类型，让上层决策

---

### P2-3. `label_to_core` 对未知标签静默降级

**文件**: `src/bin/ai/driver/intent_model.rs:245-252`

```rust
fn label_to_core(label: Option<&str>) -> CoreIntent {
    match label.unwrap_or("casual") {
        "query_concept" => CoreIntent::QueryConcept,
        "request_action" => CoreIntent::RequestAction,
        "seek_solution" => CoreIntent::SeekSolution,
        _ => CoreIntent::Casual,
    }
}
```

**问题**:
- 如果模型文件更新后新增了标签（比如 `"code_review"`），这里会无声地 fallback 到 `Casual`
- 没有任何日志，调试时很难发现

**修复方案**:
```rust
fn label_to_core(label: Option<&str>) -> CoreIntent {
    let label_str = label.unwrap_or("casual");
    match label_str {
        "query_concept" => CoreIntent::QueryConcept,
        "request_action" => CoreIntent::RequestAction,
        "seek_solution" => CoreIntent::SeekSolution,
        "casual" => CoreIntent::Casual,
        unknown => {
            tracing::warn!("Unknown intent label '{}', falling back to Casual", unknown);
            CoreIntent::Casual
        }
    }
}
```

---

### P2-4. Mutex 中毒后静默降级

**文件**: 
- `src/bin/ai/driver/intent_model.rs`
- `src/bin/ai/driver/agent_router.rs`
- `src/bin/ai/driver/skill_match_model.rs`

```rust
if let Ok(cache) = INTENT_MODEL_CACHE.lock()
```

**问题**:
- 如果持锁线程 panic 导致 Mutex 中毒，后续**所有**请求都拿不到缓存
- 会反复重新加载模型文件，性能骤降
- 但没有任何日志提示

**修复方案**:
```rust
match INTENT_MODEL_CACHE.lock() {
    Ok(cache) => { /* use cache */ },
    Err(poisoned) => {
        tracing::error!("Intent model cache mutex poisoned, recovering...");
        let mut cache = poisoned.into_inner();
        // 继续使用，但记录错误
    }
}
```

---

### P2-5. 魔数阈值没有文档

**文件**: `src/bin/ai/driver/agent_router.rs`

```rust
const MODEL_CONFIDENCE_THRESHOLD: f64 = 0.45;
const SEMANTIC_SWITCH_THRESHOLD: f64 = 0.085;
const SEMANTIC_SWITCH_MARGIN: f64 = 0.015;
const CURRENT_TURN_SEMANTIC_FLOOR: f64 = 0.05;
const CURRENT_TURN_ADVANTAGE_MARGIN: f64 = 0.04;
```

**问题**:
- 这些阈值是怎么来的？没有注释说明调整依据
- 调参时只能靠试

**建议**: 添加注释说明来源，例如：
```rust
/// 模型置信度阈值：低于此值时触发 LLM 升级
/// 来源：2025-12 月在 500 条测试集上调优得出
const MODEL_CONFIDENCE_THRESHOLD: f64 = 0.45;
```

---

### P2-6. `score_skill_smart` 永远返回 0.0 — 死代码

**文件**: `src/bin/ai/driver/skill_matching.rs:65-71`

```rust
pub(super) fn score_skill_smart(
    _skill: &SkillManifest, _input_lower: &str, _intent: Option<&UserIntent>,
) -> f64 {
    0.0
}
```

**问题**:
- 函数签名存在但实现是空的
- 如果有调用方依赖它做语义评分，会得到全 0

**建议**: 要么实现它，要么删除它。

---

## 架构建议

### 1. 提取共享工具模块

把重复的 `normalize_text` 和其他文本处理函数提取到：
```
src/bin/ai/driver/text_utils.rs
```

### 2. 改进意图识别策略

当前的问题是用硬编码关键词做意图判断。建议分阶段改进：
- **短期**: 用正则表达式 + 词边界匹配替换 `.contains()`
- **中期**: 扩展关键词白名单，支持多语言（中文 action verbs）
- **长期**: 用 LLM 做意图分类，关键词只作为快速路径

### 3. 改进排除逻辑

当前的 `excludes` 是纯子串匹配，误杀率高。建议：
- 使用正则表达式词边界匹配
- 或者在 skill manifest 中支持更丰富的排除规则（如 `"test" AND NOT "unittest"`）

### 4. 改进 JSON 提取

当前的 `parse_reflect_flag` 用 `{` 和 `}` 做简单提取，容易失败。建议：
- 在 prompt 中强制要求 LLM 只返回 JSON
- 或者用更鲁棒的提取逻辑（尝试解析多个 `{}` 段）

---

## 测试建议

### 必须补充的测试用例

1. **UTF-8 边界测试**
   ```rust
   #[test]
   fn test_utf8_slicing() {
       let chinese = "这是一个测试".repeat(1000); // 远超 2000 字节
       let truncated = safe_truncate(&chinese, 2000);
       assert!(truncated.len() <= 2000);
       assert!(truncated.chars().all(|c| c != '\u{FFFD}')); // 没有替换字符
   }
   ```

2. **中文误判测试**
   ```rust
   #[test]
   fn test_answer_stability_chinese() {
       assert!(!answer_looks_unstable_for_writeback("可能的优化方案如下："));
       assert!(!answer_looks_unstable_for_writeback("不需要猜测，可以直接验证"));
       assert!(answer_looks_unstable_for_writeback("我不确定，可能需要进一步验证"));
   }
   ```

3. **词边界匹配测试**
   ```rust
   #[test]
   fn test_skill_exclude_word_boundary() {
       let skill = SkillManifest { excludes: vec!["test".to_string()], .. };
       assert!(is_excluded_by_skill(&skill, "run the test"));
       assert!(!is_excluded_by_skill(&skill, "protest the decision"));
       assert!(!is_excluded_by_skill(&skill, "latest update"));
   }
   ```

---

## 总结

**最需要优先修的**:
1. `answer_looks_unstable_for_writeback` 的中文子串匹配 — 误杀率极高，直接影响经验记忆的回写质量
2. `verification.rs` 的 UTF-8 byte slicing — 处理中文时会 panic
3. `is_intent_excluded` 忽略 skill 参数 — 当前逻辑是错的

**核心改进方向**:
- 把子串匹配替换为分词后的精确匹配或正则表达式（带 `\b` 词边界）
- 抽取公共的 `normalize_text` 到共享模块
- 修复 UTF-8 切片 panic 风险
- 给 `label_to_core` 的 `_` 分支加日志 warning
- 给魔数阈值加注释说明来源
