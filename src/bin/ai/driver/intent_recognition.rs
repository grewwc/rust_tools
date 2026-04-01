use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::{request, types::App};

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

/// 使用 LLM 进行意图识别
pub async fn detect_intent_with_llm(app: &App, input: &str) -> Result<UserIntent, Box<dyn std::error::Error>> {
    let system_prompt = r#"你是一个意图识别助手。你的任务是分析用户的输入，识别其意图。

请按照以下规则分析：

1. **核心意图（core）** - 判断用户的交流目的：
   - `query_concept`: 询问概念、定义、含义（"这是什么"、"是什么意思"、"what is"）
   - `request_action`: 请求执行某个操作（"帮我做 X"、"请执行"、"help me"）
   - `seek_solution`: 寻求解决方案、询问方法（"怎么处理"、"如何解决"、"how to"）
   - `casual`: 闲聊、问候、或其他不属于以上类别

2. **意图修饰符（modifiers）** - 提取额外信息：
   - `is_search_query`: 是否是搜索/查找类查询（包含"找几个"、"收集"、"有哪些"、"推荐"、"搜索"等）
   - `target_resource`: 如果在搜索，目标资源是什么？可能的值："skill"、"tool"、"doc"、"file"，如果不是搜索则为 null
   - `negation`: 是否包含否定词（"不"、"别"、"不要"、"not"、"don't" 等）

**重要规则**：
- 核心意图和修饰符是正交的。例如"帮我找几个工具"的核心意图是 `request_action`（因为说"帮我"），同时修饰符中 `is_search_query=true`，`target_resource="tool"`
- 不要过度解读，基于字面意思判断
- 如果不确定，选择最接近的类别

请以 JSON 格式返回，格式如下：
{
  "core": "query_concept|request_action|seek_solution|casual",
  "modifiers": {
    "is_search_query": true|false,
    "target_resource": "skill|tool|doc|file|null",
    "negation": true|false
  }
}

**示例**：
- "什么是 Rust 的所有权？" → {"core": "query_concept", "modifiers": {"is_search_query": false, "target_resource": null, "negation": false}}
- "帮我审查这段代码" → {"core": "request_action", "modifiers": {"is_search_query": false, "target_resource": null, "negation": false}}
- "怎么处理这个报错？" → {"core": "seek_solution", "modifiers": {"is_search_query": false, "target_resource": null, "negation": false}}
- "帮我找几个 review skill" → {"core": "request_action", "modifiers": {"is_search_query": true, "target_resource": "skill", "negation": false}}
- "有什么工具可以调试？" → {"core": "query_concept", "modifiers": {"is_search_query": true, "target_resource": "tool", "negation": false}}
- "不要执行这个" → {"core": "request_action", "modifiers": {"is_search_query": false, "target_resource": null, "negation": true}}
"#;

    let user_message = format!("请分析以下用户输入的意图：\n\n用户输入：{}", input);

    // 构建请求消息
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": system_prompt
        }),
        serde_json::json!({
            "role": "user",
            "content": user_message
        }),
    ];

    // 使用一个小而快的模型进行意图识别
    // 可以选择专门的轻量级模型，如 DeepSeek-V3 或其他快速模型
    let intent_model = app.config.intent_model.as_deref().unwrap_or("deepseek-chat");
    
    let response = request::do_request_json(app, intent_model, &messages, false).await?;
    
    // 解析响应
    let intent_json = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|msg| msg.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .ok_or("Failed to extract intent from LLM response")?;

    // 尝试从响应中提取 JSON
    let intent_value = extract_json_from_response(intent_json)?;
    let user_intent: UserIntent = serde_json::from_value(intent_value)?;

    Ok(user_intent)
}

/// 从 LLM 响应中提取 JSON
fn extract_json_from_response(content: &str) -> Result<Value, Box<dyn std::error::Error>> {
    // 尝试直接解析
    if let Ok(value) = serde_json::from_str::<Value>(content.trim()) {
        return Ok(value);
    }

    // 尝试提取代码块中的 JSON
    let json_block_markers = [
        ("```json", "```"),
        ("```JSON", "```"),
        ("```", "```"),
    ];

    for (start, end) in &json_block_markers {
        if let Some(start_idx) = content.find(start) {
            let content_after_start = &content[start_idx + start.len()..];
            if let Some(end_idx) = content_after_start.find(end) {
                let json_str = content_after_start[..end_idx].trim();
                if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                    return Ok(value);
                }
            }
        }
    }

    // 尝试找到第一个 { 和最后一个 }
    if let Some(first_brace) = content.find('{') {
        if let Some(last_brace) = content.rfind('}') {
            if first_brace < last_brace {
                let json_str = &content[first_brace..=last_brace];
                if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                    return Ok(value);
                }
            }
        }
    }

    Err(format!("Failed to parse JSON from response: {}", content).into())
}

/// 回退到基于规则的意图识别（当 LLM 不可用时）
pub fn detect_intent_fallback(input: &str) -> UserIntent {
    let mut modifiers = IntentModifiers::default();

    // 询问概念的关键词
    let concept_patterns = [
        "是什么", "什么意思", "含义", "定义", "解释", "说明",
        "what is", "what's", "meaning", "define", "explain",
        "是啥", "啥意思", "咋回事", "指的是什么", "代表什么",
    ];

    // 寻求解决方案的关键词
    let seek_solution_patterns = [
        "怎么", "如何", "怎么办", "怎么处理", "如何解决",
        "how to", "how do i", "what should i do",
        "为什么", "为啥", "原因",
    ];

    // 请求行动的关键词
    let request_action_patterns = [
        "帮我", "给我", "请", "帮我做", "帮我写", "帮我改",
        "帮我检查", "帮我调试", "帮我重构", "帮我审查",
        "help me", "please", "do this", "fix", "review", "refactor",
        "优化", "改进", "整理", "重写", "运行", "执行",
    ];

    // 检测搜索/查找类查询的模式
    let search_patterns = [
        "找几个", "找一些", "找些", "收集",
        "有什么", "有哪些", "推荐几个", "推荐一些",
        "搜索", "查找",
    ];
    
    // 检查是否是搜索查询
    if search_patterns.iter().any(|p| input.contains(p)) {
        modifiers.is_search_query = true;
        modifiers.target_resource = extract_target_resource(input);
    }

    // 检查否定词
    let negation_patterns = ["不", "别", "不要", "无需", "不需要", "not", "don't", "no"];
    modifiers.negation = negation_patterns.iter().any(|p| input.contains(p));

    // 优先匹配最具体的模式
    if concept_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent {
            core: CoreIntent::QueryConcept,
            modifiers,
        };
    }

    if seek_solution_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent {
            core: CoreIntent::SeekSolution,
            modifiers,
        };
    }

    if request_action_patterns.iter().any(|p| input.contains(p)) {
        return UserIntent {
            core: CoreIntent::RequestAction,
            modifiers,
        };
    }

    UserIntent {
        core: CoreIntent::Casual,
        modifiers,
    }
}

/// 从输入中提取目标资源类型
fn extract_target_resource(input: &str) -> Option<String> {
    let resource_keywords = [
        ("技能", "skill"),
        ("skill", "skill"),
        ("skills", "skill"),
        ("工具", "tool"),
        ("tool", "tool"),
        ("tools", "tool"),
        ("文档", "doc"),
        ("doc", "doc"),
        ("docs", "doc"),
        ("文件", "file"),
        ("file", "file"),
    ];

    for (cn, en) in resource_keywords {
        if input.contains(cn) || input.contains(en) {
            return Some(en.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_query_concept() {
        let intent = detect_intent_fallback("Rust 的所有权是什么？");
        assert_eq!(intent.core, CoreIntent::QueryConcept);
        assert!(!intent.modifiers.is_search_query);
    }

    #[test]
    fn test_fallback_request_action() {
        let intent = detect_intent_fallback("帮我审查这段代码");
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
        // "有什么" 触发搜索查询，核心意图为 Casual（因为没有明确的概念询问词）
        let intent = detect_intent_fallback("有什么工具可以调试？");
        assert_eq!(intent.core, CoreIntent::Casual);
        assert!(intent.modifiers.is_search_query);
        assert_eq!(intent.modifiers.target_resource, Some("tool".to_string()));
    }

    #[test]
    fn test_fallback_negation() {
        let intent = detect_intent_fallback("不要执行这个");
        assert!(intent.modifiers.negation);
    }

    #[test]
    fn test_extract_json_from_response() {
        // 直接 JSON
        let json = r#"{"core": "request_action", "modifiers": {"is_search_query": true, "target_resource": "skill", "negation": false}}"#;
        let result = extract_json_from_response(json).unwrap();
        assert!(result.get("core").is_some());

        // 带代码块
        let json_block = r#"```json
{"core": "query_concept", "modifiers": {"is_search_query": false, "target_resource": null, "negation": false}}
```"#;
        let result = extract_json_from_response(json_block).unwrap();
        assert!(result.get("core").is_some());

        // 带额外文本
        let extra_text = r#"好的，我来分析：
{"core": "seek_solution", "modifiers": {"is_search_query": false, "target_resource": null, "negation": false}}
希望这能帮到你。"#;
        let result = extract_json_from_response(extra_text).unwrap();
        assert!(result.get("core").is_some());
    }
}
