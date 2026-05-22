use serde::{Deserialize, Serialize};

/// 核心意图类型 - 只关注用户的交流目的，不关注具体对象
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreIntent {
    /// 询问概念/定义（"这是什么"、"是什么意思"）
    QueryConcept,
    /// 请求操作（"帮我做 X"）
    RequestAction,
    /// 寻求解决方案（"怎么处理"、"如何解决"）
    SeekSolution,
    /// 闲聊/其他
    Casual,
}

/// 意图修饰符 - 可扩展的元数据，用于描述意图的特殊属性
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentModifiers {
    /// 是否是搜索/查找类查询（"找几个"、"收集"、"有哪些"）
    pub is_search_query: bool,
    /// 目标资源类型（"skills", "tools", "docs" 等）
    pub target_resource: Option<String>,
    /// 是否包含否定词
    pub negation: bool,
}

/// 完整的用户意图，包含核心意图和修饰符
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserIntent {
    pub core: CoreIntent,
    pub modifiers: IntentModifiers,
}

impl UserIntent {
    /// 创建一个简单的用户意图（无修饰符）
    pub fn new(core: CoreIntent) -> Self {
        Self {
            core,
            modifiers: IntentModifiers::default(),
        }
    }

    /// 检查是否是搜索资源类查询
    pub fn is_search_query(&self) -> bool {
        self.modifiers.is_search_query
    }

    /// 检查是否是针对特定资源类型的搜索
    pub fn is_searching_resource(&self, resource_type: &str) -> bool {
        self.modifiers.is_search_query
            && self
                .modifiers
                .target_resource
                .as_ref()
                .map(|r| r == resource_type)
                .unwrap_or(false)
    }
}

pub fn detect_intent(input: &str) -> UserIntent {
    super::intent_model::detect_intent(input, None)
}

pub fn detect_intent_with_model_path(input: &str, model_path: &std::path::Path) -> UserIntent {
    super::intent_model::detect_intent(input, Some(model_path))
}

pub fn detect_intent_fallback(input: &str) -> UserIntent {
    detect_intent(input)
}

/// 对"本地 TF-IDF 给出 Casual 但内容不像闲聊"的请求做 LLM 二次判定。
///
/// 调用时机（caller 决定）：在 prepare_turn / skill_runtime 这种异步路径上，
/// 当 `local.core == CoreIntent::Casual` 且 `looks_non_casual(question)` 时，
/// 调用本函数升级。本地分类失败时 LLM 也失败，那就保留 `local`，避免阻塞。
///
/// 本函数会在 stderr 打印 `[intent:llm]` 标记（在 request.rs 内部完成），
/// 终端用户能直接看到"这一轮意图识别用了大模型"。
pub async fn upgrade_intent_via_model(
    app: &crate::ai::types::App,
    question: &str,
    local: UserIntent,
) -> UserIntent {
    if local.core != CoreIntent::Casual {
        return local;
    }
    if !looks_non_casual(question) {
        return local;
    }
    match crate::ai::request::classify_intent_via_model(app, question).await {
        Some(core) => UserIntent {
            core,
            modifiers: local.modifiers,
        },
        None => local,
    }
}

/// 触发 LLM fallback 的启发式：本地分到 Casual，但问题本身明显
/// 是个非平凡请求（带代码、长问题、含明显操作动词或问题信号）。
///
/// 实现策略：
/// 1. 优先依赖结构化信号（代码块、错误堆栈痕迹、问号、长度），它们与语种无关。
/// 2. 仅在结构化信号不足时，退化到关键词匹配；此时 ASCII 关键词使用 ASCII
///    词边界（避免 `"fn "` 漏掉 `"fn("`、`"how"` 误匹配 `"however"`），
///    CJK 关键词只做子串匹配。
fn looks_non_casual(question: &str) -> bool {
    let q = question.trim();
    if q.is_empty() {
        return false;
    }
    let len = q.chars().count();
    if len < 8 {
        return false;
    }
    // ---- 结构化信号 ----
    if has_structural_code_signal(q) {
        return true;
    }
    if has_question_punctuation(q) {
        return true;
    }
    if len >= 60 {
        return true;
    }

    // ---- 关键词信号（带词边界 / 大小写不敏感）----
    let lower = q.to_ascii_lowercase();
    has_ascii_action_signal(&lower) || has_cjk_action_signal(q)
}

/// 是否包含明显的"代码 / 错误堆栈"结构信号。
fn has_structural_code_signal(q: &str) -> bool {
    if q.contains("```") {
        return true;
    }
    // Rust path / namespace 形式：`module::item`
    if q.contains("::") {
        return true;
    }
    // 各语言常见函数声明前缀（带词边界）
    let lower = q.to_ascii_lowercase();
    for kw in ["fn ", "fn(", "def ", "func ", "function "] {
        if lower.contains(kw) {
            return true;
        }
    }
    false
}

fn has_question_punctuation(q: &str) -> bool {
    q.contains('?') || q.contains('？')
}

fn has_ascii_action_signal(lower: &str) -> bool {
    // 一律按 ASCII 词边界匹配，避免 "implement" 误匹配 "reimplementation"、
    // "how" 误匹配 "however" 等情况。
    const KEYWORDS: &[&str] = &[
        "error", "panic", "traceback", "exception", "stacktrace",
        "fix", "implement", "refactor", "review", "debug", "optimize",
        "how", "why", "what",
    ];
    KEYWORDS
        .iter()
        .any(|kw| super::ascii_word_contains(lower, kw))
}

fn has_cjk_action_signal(q: &str) -> bool {
    // CJK 没有词边界概念，但这些词足够具体，子串匹配误杀率低。
    const PATTERNS: &[&str] = &[
        "帮我", "修复", "实现", "重构", "优化", "怎么", "如何", "为什么",
        "报错", "失败", "异常",
    ];
    PATTERNS.iter().any(|p| q.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_exists() {
        let path = super::super::intent_model::default_model_path();
        assert!(path.exists(), "missing bundled intent model: {}", path.display());
    }

    #[test]
    fn test_default_model_loads() {
        let path = super::super::intent_model::default_model_path();
        let model = super::super::intent_model::load_model_file(&path);
        assert!(model.is_ok(), "failed to load bundled intent model");
    }

    #[test]
    fn test_fallback_query_concept() {
        let intent = detect_intent_fallback("什么是 trait object");
        assert_eq!(intent.core, CoreIntent::QueryConcept);
        assert!(!intent.modifiers.is_search_query);
    }

    #[test]
    fn test_fallback_request_action() {
        let intent = detect_intent_fallback("帮我添加错误处理");
        assert_eq!(intent.core, CoreIntent::RequestAction);
        assert!(!intent.modifiers.is_search_query);
    }

    #[test]
    fn test_fallback_seek_solution() {
        let intent = detect_intent_fallback("怎么处理这个报错？");
        assert_eq!(intent.core, CoreIntent::SeekSolution);
        assert!(!intent.modifiers.is_search_query);
    }

    #[test]
    fn test_fallback_search_skill() {
        let intent = detect_intent_fallback("帮我找几个 review skill");
        assert_eq!(intent.core, CoreIntent::RequestAction);
        assert!(intent.modifiers.is_search_query);
        assert_eq!(intent.modifiers.target_resource, Some("skill".to_string()));
    }

    #[test]
    fn test_fallback_search_tool() {
        let intent = detect_intent_fallback("有什么工具可以调试？");
        assert_eq!(intent.core, CoreIntent::RequestAction);
        assert!(intent.modifiers.is_search_query);
        assert_eq!(intent.modifiers.target_resource, Some("tool".to_string()));
    }

    #[test]
    fn test_fallback_negation() {
        let intent = detect_intent_fallback("不要执行这个");
        assert!(intent.modifiers.negation);
    }

    #[test]
    fn test_detect_intent_casual_greeting() {
        let intent = detect_intent("hello");
        assert_eq!(intent.core, CoreIntent::Casual);
    }

    #[test]
    fn test_detect_intent_english_solution() {
        let intent = detect_intent("how to fix this panic");
        assert_eq!(intent.core, CoreIntent::SeekSolution);
    }

    #[test]
    fn test_detect_intent_request_action_beats_search_modifier() {
        let intent = detect_intent("帮我找几个 review skill");
        assert_eq!(intent.core, CoreIntent::RequestAction);
        assert!(intent.is_searching_resource("skill"));
    }
}
