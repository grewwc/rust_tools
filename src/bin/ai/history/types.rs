use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::types::ToolCall;

pub(in crate::ai) const MAX_HISTORY_TURNS: usize = 200;
pub(in crate::ai) const COLON: char = '\0';
pub(in crate::ai) const NEWLINE: char = '\x01';
pub(crate) const ROLE_SYSTEM: &str = "system";
pub(crate) const ROLE_INTERNAL_NOTE: &str = "internal_note";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(in crate::ai) struct Message {
    pub(in crate::ai) role: String,
    pub(in crate::ai) content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_call_id: Option<String>,
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
