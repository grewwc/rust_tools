use std::{
    collections::HashMap,
    fs::File,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{cli::ParsedCli, prompt::PromptEditor};

#[derive(Clone)]
pub(super) struct AppConfig {
    pub(super) api_key: String,
    pub(super) history_file: PathBuf,
    pub(super) endpoint: String,
    pub(super) vl_default_model: String,
    pub(super) history_max_chars: usize,
    pub(super) history_keep_last: usize,
    pub(super) history_summary_max_chars: usize,
}

pub(super) struct App {
    pub(super) cli: ParsedCli,
    pub(super) config: AppConfig,
    pub(super) session_id: String,
    pub(super) session_history_file: PathBuf,
    pub(super) client: Client,
    pub(super) current_model: String,
    pub(super) pending_files: Option<String>,
    pub(super) pending_clipboard: bool,
    pub(super) pending_short_output: bool,
    pub(super) attached_image_files: Vec<String>,
    pub(super) shutdown: Arc<AtomicBool>,
    pub(super) streaming: Arc<AtomicBool>,
    pub(super) cancel_stream: Arc<AtomicBool>,
    pub(super) writer: Option<File>,
    pub(super) prompt_editor: Option<PromptEditor>,
    pub(super) agent_context: Option<AgentContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolDefinition {
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FunctionDefinition {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ToolCall {
    pub(super) id: String,
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct FunctionCall {
    pub(super) name: String,
    pub(super) arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolResult {
    pub(super) tool_call_id: String,
    pub(super) content: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct AgentContext {
    pub(super) tools: Vec<ToolDefinition>,
    pub(super) mcp_servers: HashMap<String, McpServerConfig>,
    pub(super) max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct McpServerConfig {
    pub(super) command: String,
    #[serde(default)]
    pub(super) args: Vec<String>,
    #[serde(default)]
    pub(super) env: HashMap<String, String>,
    #[serde(default = "default_mcp_request_timeout_ms")]
    pub(super) request_timeout_ms: u64,
    #[serde(default)]
    pub(super) disabled: bool,
}

fn default_mcp_request_timeout_ms() -> u64 {
    30000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct McpTool {
    pub(super) name: String,
    pub(super) description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub(super) input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct McpResource {
    pub(super) uri: String,
    pub(super) name: String,
    #[serde(default)]
    pub(super) description: Option<String>,
    #[serde(default, rename = "mimeType")]
    pub(super) mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct McpPrompt {
    pub(super) name: String,
    #[serde(default)]
    pub(super) description: Option<String>,
    #[serde(default)]
    pub(super) arguments: Vec<McpPromptArgument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct McpPromptArgument {
    pub(super) name: String,
    #[serde(default)]
    pub(super) description: Option<String>,
    #[serde(default)]
    pub(super) required: bool,
}

pub(super) fn take_stream_cancelled(app: &App) -> bool {
    app.cancel_stream.swap(false, Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum StreamOutcome {
    #[default]
    Completed,
    Cancelled,
    ToolCall,
}

#[derive(Debug, Clone, Default)]
pub(super) struct StreamResult {
    pub(super) outcome: StreamOutcome,
    pub(super) tool_calls: Vec<ToolCall>,
    pub(super) assistant_text: String,
    pub(super) hidden_meta: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionContext {
    pub(super) question: String,
    pub(super) history_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct LoopOverrides {
    pub(super) short_output: bool,
    pub(super) history_count: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct FileParseResult {
    pub(super) text_files: Vec<String>,
    pub(super) image_files: Vec<String>,
    pub(super) binary_files: Vec<String>,
}
