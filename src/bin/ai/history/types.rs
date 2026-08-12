use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::types::ToolCall;

pub(in crate::ai) const MAX_HISTORY_TURNS: usize = 200;
pub(in crate::ai) const COLON: char = '\0';
pub(in crate::ai) const NEWLINE: char = '\x01';
pub(crate) const ROLE_SYSTEM: &str = "system";
pub(crate) const ROLE_INTERNAL_NOTE: &str = "internal_note";

/// 工具执行的结构化结果旁路。正文仍原样保存在 `messages`，该记录只用于构造
/// 模型请求时判断旧失败是否已被同执行签名的后续成功解决。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct ToolExecutionOutcome {
    pub(in crate::ai) tool_call_id: String,
    pub(in crate::ai) execution_signature: String,
    pub(in crate::ai) succeeded: bool,
}

/// 显式 skill 选择在 turn 准备阶段的实际注入结果。该旁路记录不进入模型消息，
/// 用于区分命令解析、状态传递与 skill 注入三个阶段的问题。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct SkillActivationEvent {
    pub(in crate::ai) requested_skill: String,
    pub(in crate::ai) injected_skill: Option<String>,
    pub(in crate::ai) source: String,
    pub(in crate::ai) outcome: String,
}

/// 运行时合成 user 消息的内部来源标记（非真实用户输入，不构成用户轮次边界）。
///
/// `Message` 暂无独立 metadata 字段，因此把仅供运行时使用的来源旁路存进 user
/// 消息不会使用的 `reasoning_content`。该字段会随 canonical history 持久化和重建，
/// 并在 request normalization 的第一步清除，绝不能进入 provider payload。
/// 不得改用 content 前缀：真实用户可以输入任意正文，按正文识别会伪造轮次边界。
const RUNTIME_SYNTHETIC_USER_ORIGIN: &str = "runtime-origin:synthetic-user:v1";

/// 构造运行时合成的 user 消息。所有轮中途注入的 user 消息都必须走此入口。
pub(in crate::ai) fn runtime_synthetic_user_message(content: Value) -> Message {
    Message {
        role: "user".to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: Some(RUNTIME_SYNTHETIC_USER_ORIGIN.to_string()),
    }
}

/// 判断消息是否为运行时合成的 user 消息（非真实用户输入的轮次边界）。
pub(in crate::ai) fn is_runtime_synthetic_user_message(message: &Message) -> bool {
    message.role == "user"
        && message.reasoning_content.as_deref() == Some(RUNTIME_SYNTHETIC_USER_ORIGIN)
}

/// 清除仅供运行时使用的消息来源旁路，避免内部标记泄漏给 provider。
pub(in crate::ai) fn clear_runtime_message_metadata(message: &mut Message) {
    if is_runtime_synthetic_user_message(message) {
        message.reasoning_content = None;
    }
}

/// messages 中最后一个**真实** user 消息的索引（跳过运行时合成的 user 消息）。
pub(in crate::ai) fn last_real_user_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(in crate::ai) struct Message {
    pub(in crate::ai) role: String,
    pub(in crate::ai) content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_call_id: Option<String>,
    /// 模型在 thinking/reasoning 模式下返回的 reasoning_content。
    /// 部分服务端（如 DeepSeek thinking-mode）要求把上一轮 assistant 的
    /// reasoning_content 原样回传，否则会返回 400 invalid_request_error。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) reasoning_content: Option<String>,
}

pub(crate) fn is_internal_note_role(role: &str) -> bool {
    role == ROLE_INTERNAL_NOTE
}

pub(crate) fn is_system_like_role(role: &str) -> bool {
    role == ROLE_SYSTEM || is_internal_note_role(role)
}

pub(in crate::ai) fn retained_turn_start(messages: &[Message], max_user_turns: usize) -> usize {
    if max_user_turns == 0 || messages.is_empty() {
        return messages.len();
    }

    let user_indices = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.role == "user" && !is_runtime_synthetic_user_message(message)).then_some(index)
        })
        .collect::<Vec<_>>();

    if user_indices.len() <= max_user_turns {
        return 0;
    }

    user_indices[user_indices.len() - max_user_turns]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn assistant() -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String("ok".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn synthetic_user_origin_does_not_collide_with_user_content() {
        let synthetic = runtime_synthetic_user_message(Value::String(
            "[Runtime task-evidence handoff, not a new end-user request.]".to_string(),
        ));
        assert!(is_runtime_synthetic_user_message(&synthetic));

        // 真实用户可以原样输入旧 marker 前缀，仍必须被视为真实轮次边界。
        assert!(!is_runtime_synthetic_user_message(&user(
            "[runtime-synthetic-user] 请把这段文本当作普通输入"
        )));
        assert!(!is_runtime_synthetic_user_message(&user("请修复这个 bug")));

        // 非 user 角色即使携带内部旁路也不命中。
        let mut assistant = assistant();
        assistant.reasoning_content = synthetic.reasoning_content.clone();
        assert!(!is_runtime_synthetic_user_message(&assistant));

        // 来源旁路与 content 形态无关，多模态消息同样可可靠识别。
        let multimodal = runtime_synthetic_user_message(Value::Array(vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "x.png"}}),
            serde_json::json!({"type": "text", "text": "分析这张图"}),
        ]));
        assert!(is_runtime_synthetic_user_message(&multimodal));

        // canonical history 序列化/恢复必须保留来源旁路。
        let encoded = serde_json::to_string(&multimodal).unwrap();
        let restored: Message = serde_json::from_str(&encoded).unwrap();
        assert!(is_runtime_synthetic_user_message(&restored));
        assert_eq!(restored.content, multimodal.content);
    }

    #[test]
    fn last_real_user_index_skips_synthetic_pairs() {
        let messages = vec![
            user("旧问题"),
            assistant(),
            user("当前真实问题"),
            runtime_synthetic_user_message(Value::String("证据交接".to_string())),
            assistant(),
        ];
        // 边界必须落在真实问题，而不是合成消息。
        assert_eq!(last_real_user_index(&messages), Some(2));
        // 无合成消息时等价于 rposition(role == "user")。
        let plain = vec![user("a"), assistant(), user("b")];
        assert_eq!(last_real_user_index(&plain), Some(2));
        // 空列表。
        assert_eq!(last_real_user_index(&[]), None);
    }

    #[test]
    fn retained_turn_start_ignores_synthetic_users() {
        let messages = vec![
            user("第 1 轮"),
            assistant(),
            user("第 2 轮"),
            assistant(),
            runtime_synthetic_user_message(Value::String("图片 followup".to_string())),
            assistant(),
        ];
        // 只有 2 个真实轮次：max=2 时从第 1 轮开始（合成消息不占轮次数）。
        assert_eq!(retained_turn_start(&messages, 2), 0);
        // max=1 时从第 2 轮开始。
        assert_eq!(retained_turn_start(&messages, 1), 2);
    }
}
