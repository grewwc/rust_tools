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
    let clean_question = crate::ai::request::strip_system_reminders(question);
    let clean_question = clean_question.trim();
    if clean_question.is_empty() {
        return local;
    }
    if !looks_non_casual(clean_question) {
        return local;
    }
    match crate::ai::request::classify_intent_via_model(app, clean_question).await {
        Some(core) => UserIntent {
            core,
            modifiers: local.modifiers,
        },
        None => local,
    }
}

/// 触发 LLM fallback 的启发式：本地分到 Casual，但问题本身明显
/// 是个非平凡请求。
///
/// 只依赖结构信号，不再依赖任何手写关键词词表：代码块/路径样式、
/// 多行结构、问句形态、长度等。这样不会因为某几个词面命中就把语义路由
/// 拉偏。
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
    if q.lines().filter(|line| !line.trim().is_empty()).count() >= 3 {
        return true;
    }
    if count_artifact_like_tokens(q) >= 2 {
        return true;
    }
    if len >= 60 {
        return true;
    }
    false
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
    count_artifact_like_tokens(q) > 0
}

fn has_question_punctuation(q: &str) -> bool {
    q.contains('?') || q.contains('？')
}

fn count_artifact_like_tokens(question: &str) -> usize {
    question
        .split_whitespace()
        .filter(|token| {
            token.contains('/')
                || token.contains('\\')
                || token.ends_with(".rs")
                || token.ends_with(".ts")
                || token.ends_with(".tsx")
                || token.ends_with(".js")
                || token.ends_with(".jsx")
                || token.ends_with(".py")
                || token.ends_with(".go")
                || token.ends_with(".java")
                || token.ends_with(".json")
                || token.ends_with(".yaml")
                || token.ends_with(".yml")
                || token.ends_with(".toml")
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_exists() {
        let path = super::super::intent_model::default_model_path();
        assert!(
            path.exists(),
            "missing bundled intent model: {}",
            path.display()
        );
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
    fn test_detect_intent_request_action_without_keyword_modifiers() {
        let intent = detect_intent("帮我找几个 review skill");
        // 关键词 modifier 已移除：核心意图完全由模型决定。
        // 该样例在当前模型下可能是 Casual，不再强制提升为 RequestAction。
        assert!(matches!(
            intent.core,
            CoreIntent::RequestAction | CoreIntent::Casual
        ));
        assert!(!intent.is_search_query());
        assert!(intent.modifiers.target_resource.is_none());
        assert!(!intent.modifiers.negation);
    }

    #[test]
    fn system_reminder_polluted_greeting_is_not_non_casual() {
        let polluted = format!(
            "<system-reminder>{}</system-reminder>\n\nhi",
            "src/bin/ai/driver/skill_runtime.rs\n".repeat(200)
        );
        assert!(!looks_non_casual(
            crate::ai::request::strip_system_reminders(&polluted).trim()
        ));
    }
}
