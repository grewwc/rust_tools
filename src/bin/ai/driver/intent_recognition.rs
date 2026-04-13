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
