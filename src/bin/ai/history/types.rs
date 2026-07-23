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
        .filter_map(|(index, message)| (message.role == "user").then_some(index))
        .collect::<Vec<_>>();

    if user_indices.len() <= max_user_turns {
        return 0;
    }

    user_indices[user_indices.len() - max_user_turns]
}
