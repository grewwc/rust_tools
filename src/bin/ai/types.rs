use std::{
    fs::File,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use reqwest::Client;
use rust_tools::commonw::FastMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{agents::AgentManifest, cli::ParsedCli, prompt::PromptEditor};

/// Configuration for the AI application, including API credentials,
/// endpoint, model settings, and conversation history parameters.
#[derive(Clone)]
pub(super) struct AppConfig {
    pub(super) api_key: String,
    pub(super) history_file: PathBuf,
    pub(super) endpoint: String,
    pub(super) vl_default_model: String,
    pub(super) history_max_chars: usize,
    pub(super) history_keep_last: usize,
    pub(super) history_summary_max_chars: usize,
    pub(super) intent_model: Option<String>,
    pub(super) intent_model_path: PathBuf,
    pub(super) agent_route_model_path: PathBuf,
    pub(super) skill_match_model_path: PathBuf,
}

/// Main application state holding CLI arguments, configuration,
/// HTTP client, session data, and streaming control flags.
impl Clone for App {
    fn clone(&self) -> Self {
        Self {
            cli: self.cli.clone(),
            config: self.config.clone(),
            session_id: self.session_id.clone(),
            session_history_file: self.session_history_file.clone(),
            client: self.client.clone(),
            current_model: self.current_model.clone(),
            current_agent: self.current_agent.clone(),
            current_agent_manifest: self.current_agent_manifest.clone(),
            pending_files: self.pending_files.clone(),
            pending_short_output: self.pending_short_output,
            attached_image_files: self.attached_image_files.clone(),
            shutdown: self.shutdown.clone(),
            streaming: self.streaming.clone(),
            cancel_stream: self.cancel_stream.clone(),
            ignore_next_prompt_interrupt: self.ignore_next_prompt_interrupt,
            writer: self.writer.clone(),
            prompt_editor: None,
            agent_context: self.agent_context.clone(),
            last_skill_bias: self.last_skill_bias.clone(),
            os: self.os.clone(),
            agent_reload_counter: self.agent_reload_counter,
            observers: Vec::new(),
        }
    }
}

impl App {
    #[allow(dead_code)]
    pub(super) fn clone_without_observers_intentionally(&self) -> Self {
        self.clone()
    }
}

pub(super) struct App {
    pub(super) cli: ParsedCli,
    pub(super) config: AppConfig,
    pub(super) session_id: String,
    pub(super) session_history_file: PathBuf,
    pub(super) client: Client,
    pub(super) current_model: String,
    pub(super) current_agent: String,
    pub(super) current_agent_manifest: Option<AgentManifest>,
    pub(super) pending_files: Option<String>,
    pub(super) pending_short_output: bool,
    pub(super) attached_image_files: Vec<String>,
    pub(super) shutdown: Arc<AtomicBool>,
    pub(super) streaming: Arc<AtomicBool>,
    pub(super) cancel_stream: Arc<AtomicBool>,
    pub(super) ignore_next_prompt_interrupt: bool,
    pub(super) writer: Option<Arc<std::sync::Mutex<File>>>,
    pub(super) prompt_editor: Option<PromptEditor>,
    pub(super) agent_context: Option<AgentContext>,
    pub(super) last_skill_bias: Option<SkillBiasMemory>,
    pub(super) os: crate::ai::os::kernel::SharedKernel,
    pub(super) agent_reload_counter: Option<usize>,
    pub(super) observers: Vec<Box<dyn crate::ai::driver::observer::TurnObserver>>,
}

impl App {
    #[allow(dead_code)]
    pub(super) fn register_observer(
        &mut self,
        observer: Box<dyn crate::ai::driver::observer::TurnObserver>,
    ) {
        let new_name = observer.name().to_string();
        // Only dedup by name when the observer provides a non-default name.
        // "anonymous" is the trait's default fallback — multiple anonymous
        // observers are legitimate and must NOT be collapsed into one.
        if new_name != "anonymous"
            && self.observers.iter().any(|o| o.name() == new_name)
        {
            return;
        }
        self.observers.push(observer);
    }

    #[allow(dead_code)]
    pub(super) fn unregister_observer(&mut self, name: &str) -> bool {
        if name == "anonymous" {
            // Refuse to mass-remove anonymous observers; must use typed handle.
            return false;
        }
        let len_before = self.observers.len();
        self.observers.retain(|o| o.name() != name);
        self.observers.len() != len_before
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SkillBiasMemory {
    pub(super) skill_name: String,
    pub(super) question: String,
}

/// Schema definition for a tool that can be offered to the AI model,
/// wrapping a function definition with a type discriminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolDefinition {
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: FunctionDefinition,
}

/// Describes a callable function: its name, human-readable description,
/// and JSON Schema for parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FunctionDefinition {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) parameters: Value,
}

/// A request from the AI model to invoke a specific tool,
/// identified by a unique call ID and containing the function name and arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ToolCall {
    pub(super) id: String,
    #[serde(rename = "type")]
    pub(super) tool_type: String,
    pub(super) function: FunctionCall,
}

/// The function invocation details within a `ToolCall`,
/// containing the function name and a JSON-encoded argument string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct FunctionCall {
    pub(super) name: String,
    pub(super) arguments: String,
}

/// The output produced after executing a tool call,
/// linking back to the original call ID with the result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolResult {
    pub(super) tool_call_id: String,
    pub(super) content: String,
}

/// Runtime context for an agent, containing its available tools,
/// MCP server configurations, and iteration limits.
#[derive(Debug, Clone, Default)]
pub(super) struct AgentContext {
    pub(super) tools: Vec<ToolDefinition>,
    pub(super) mcp_servers: FastMap<String, McpServerConfig>,
    pub(super) max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::ai) struct McpServerConfig {
    pub(in crate::ai) command: String,
    #[serde(default)]
    pub(in crate::ai) args: Vec<String>,
    #[serde(default)]
    pub(in crate::ai) env: FastMap<String, String>,
    #[serde(default = "default_mcp_request_timeout_ms")]
    pub(in crate::ai) request_timeout_ms: u64,
    #[serde(default)]
    pub(in crate::ai) disabled: bool,
}

fn default_mcp_request_timeout_ms() -> u64 {
    30000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::ai) struct McpTool {
    pub(in crate::ai) name: String,
    pub(in crate::ai) description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub(in crate::ai) input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::ai) struct McpResource {
    pub(in crate::ai) uri: String,
    pub(in crate::ai) name: String,
    #[serde(default)]
    pub(in crate::ai) description: Option<String>,
    #[serde(default, rename = "mimeType")]
    pub(in crate::ai) mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::ai) struct McpPrompt {
    pub(in crate::ai) name: String,
    #[serde(default)]
    pub(in crate::ai) description: Option<String>,
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
