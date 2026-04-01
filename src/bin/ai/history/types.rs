use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::types::ToolCall;

pub(in crate::ai) const MAX_HISTORY_LINES: usize = 200;
pub(in crate::ai) const COLON: char = '\0';
pub(in crate::ai) const NEWLINE: char = '\x01';

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(in crate::ai) struct Message {
    pub(in crate::ai) role: String,
    pub(in crate::ai) content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::ai) tool_call_id: Option<String>,
}
