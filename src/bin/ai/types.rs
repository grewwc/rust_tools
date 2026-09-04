use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use reqwest::Client;
use rust_tools::cw::SkipMap;
use rustc_hash::{FxHashMap, FxHashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use aios_kernel::kernel::SharedKernel;

use super::{
    agents::AgentManifest,
    cli::ParsedCli,
    middleware::{RequestMiddleware, ToolMiddleware},
    persona::PersonaProfile,
    pipeline::HookRegistry,
    prompt::PromptEditor,
};

/// Configuration for the AI application, including API credentials,
/// endpoint, model settings, and conversation history parameters.
#[derive(Clone)]
pub(super) struct AppConfig {
    pub(super) api_key: String,
    pub(super) base_history_file: PathBuf,
    pub(super) history_file: PathBuf,
    pub(super) endpoint: String,
    pub(super) vl_default_model: String,
    pub(super) history_max_chars: usize,
    pub(super) history_keep_last: usize,
    pub(super) history_summary_max_chars: usize,
    pub(super) intent_model: Option<String>,
}

impl App {
    /// Create the common isolated baseline used by driver snapshots and child work.
    ///
    /// Process-local extensions and one-shot foreground state are intentionally reset,
    /// while shared runtime handles keep their existing `Arc` identity. One exception:
    /// the stream filter chain is inherited (Arc-shared) so that forked turns apply the
    /// same content-transformation policy as the foreground turn.
    fn fork_baseline(&self) -> Self {
        Self {
            cli: self.cli.clone(),
            config: self.config.clone(),
            session_id: self.session_id.clone(),
            session_history_file: self.session_history_file.clone(),
            active_persona: self.active_persona.clone(),
            client: self.client.clone(),
            current_model: self.current_model.clone(),
            current_agent: self.current_agent.clone(),
            current_agent_manifest: self.current_agent_manifest.clone(),
            pending_files: self.pending_files.clone(),
            forced_skills: self.forced_skills.clone(),
            forced_skill_source: self.forced_skill_source,
            // This state belongs only to the next user message in the foreground;
            // `App` clones are also used for DriverContext, subagents, and
            // background tasks, which must not inherit this continuation.
            pending_skill_continuation: None,
            forced_question: self.forced_question.clone(),
            attached_image_files: self.attached_image_files.clone(),
            shutdown: self.shutdown.clone(),
            streaming: self.streaming.clone(),
            cancel_stream: self.cancel_stream.clone(),
            ignore_next_prompt_interrupt: self.ignore_next_prompt_interrupt,
            prompt_editor: None,
            agent_context: self.agent_context.clone(),
            last_skill_bias: self.last_skill_bias.clone(),
            os: self.os.clone(),
            agent_reload_counter: self.agent_reload_counter,
            observers: Vec::new(),
            // Same policy as `observers`: process-level tool middleware strategy
            // does not propagate through clone (subagents/background are independent).
            tool_middlewares: Vec::new(),
            // Same policy as `tool_middlewares`: process-level LLM request
            // middleware strategy does not propagate through clone.
            llm_middlewares: Vec::new(),
            // Same policy as `observers`/`tool_middlewares`: stage/global hook
            // callbacks do not propagate through clone (subagents/background are
            // independent). The stream filter chain is the exception: it is a
            // content-transformation policy whose output lands in parent history and
            // subagent payloads, so forked turns must apply the parent's chain too
            // (see HookRegistry::inherit_stream_filters).
            hooks: {
                let mut hooks = HookRegistry::new();
                hooks.inherit_stream_filters(&self.hooks);
                hooks
            },
            last_known_prompt_tokens: self.last_known_prompt_tokens,
            last_known_cached_prompt_tokens: self.last_known_cached_prompt_tokens,
            goal_mode: self.goal_mode.clone(),
            last_turn_had_tool_calls: self.last_turn_had_tool_calls,
            last_turn_interrupted: self.last_turn_interrupted,
            prune_marks: self.prune_marks.clone(),
            turn_reasoning_items: self.turn_reasoning_items.clone(),
            stale_patch_targets: self.stale_patch_targets.clone(),
        }
    }

    pub(super) fn snapshot_for_driver_context(&self) -> Self {
        self.fork_baseline()
    }

    pub(super) fn fork_for_subagent(&self) -> Self {
        self.fork_baseline()
    }

    pub(super) fn snapshot_for_detached_helper(&self) -> Self {
        self.fork_baseline()
    }
}

#[cfg(test)]
impl Clone for App {
    fn clone(&self) -> Self {
        self.fork_baseline()
    }
}

pub(super) struct App {
    pub(super) cli: ParsedCli,
    pub(super) config: AppConfig,
    pub(super) session_id: String,
    pub(super) session_history_file: PathBuf,
    pub(super) active_persona: PersonaProfile,
    pub(super) client: Client,
    pub(super) current_model: String,
    pub(super) current_agent: String,
    pub(super) current_agent_manifest: Option<AgentManifest>,
    pub(super) pending_files: Option<String>,
    /// Forced skill list explicitly selected by the user via `@skills:<name>` or
    /// `/skills <name>...` in the input box, effective for **this turn only**
    /// (selection order preserved; multiple skills treated equally).
    /// Turn preparation reads it, force-injects these skills, and clears it at the
    /// end of the turn; the next turn is not forced.
    pub(super) forced_skills: Vec<String>,
    /// Source of `forced_skills`. Only explicit user selection carries this value,
    /// for per-turn persistence auditing.
    pub(super) forced_skill_source: Option<ForcedSkillSource>,
    /// One-shot continuation saved after the current skill explicitly requested
    /// user input via `request_user_input`. Consumed by the next ordinary user
    /// message; an explicit skill selection or session switch overwrites/clears it.
    pub(super) pending_skill_continuation: Option<PendingSkillContinuation>,
    /// When /skills <name>... <rest>, <rest> is used as this turn's question.
    pub(super) forced_question: Option<String>,
    pub(super) attached_image_files: Vec<String>,
    pub(super) shutdown: Arc<AtomicBool>,
    pub(super) streaming: Arc<AtomicBool>,
    pub(super) cancel_stream: Arc<AtomicBool>,
    pub(super) ignore_next_prompt_interrupt: bool,
    pub(super) prompt_editor: Option<PromptEditor>,
    pub(super) agent_context: Option<AgentContext>,
    pub(super) last_skill_bias: Option<SkillBiasMemory>,
    pub(super) os: SharedKernel,
    pub(super) agent_reload_counter: Option<usize>,
    pub(super) observers: Vec<Box<dyn crate::ai::driver::observer::TurnObserver>>,
    /// Tool execution middleware chain (Step 5: built per turn via
    /// `build_tool_executor_chain`; empty chain = zero behavior change).
    /// Shared via `Arc`: `dyn` middleware is not Clone, and the Arc refcount can be
    /// safely copied when building the chain each turn.
    pub(super) tool_middlewares: Vec<Arc<dyn ToolMiddleware>>,
    /// Process-level LLM request middleware chain (same policy as
    /// `tool_middlewares`: `request_model_response` builds the chain via
    /// `build_llm_client_chain` per request; empty chain = zero behavior change).
    /// Shared via `Arc`: like `tool_middlewares`, safely copyable per request
    /// chain build.
    pub(super) llm_middlewares: Vec<Arc<dyn RequestMiddleware>>,
    /// Process-level hook registry (Step 3: driver turn lifecycle hooks).
    /// Empty registry = zero behavior change; turn start/end fire in `run_turn`.
    pub(super) hooks: HookRegistry,
    /// Actual prompt_tokens returned by the server on the last request (from usage
    /// stats). Used to replace character estimation in the next request's
    /// max_tokens clamp for better precision.
    pub(super) last_known_prompt_tokens: Option<u64>,
    /// Prompt cache hit token count returned by the server on the last request.
    /// Used to subtract the reusable prefix from the next request's TPM budget
    /// estimate, so a 100% cache hit is not billed as the full prompt and falsely
    /// triggers waiting.
    pub(super) last_known_cached_prompt_tokens: Option<u64>,
    /// Goal mode state. `None` = not enabled; `Some("")` = waiting for the user's
    /// goal; `Some(goal)` = goal set, the agent keeps driving automatically until
    /// completion.
    pub(super) goal_mode: Option<String>,
    /// Whether the previous turn called tools. In goal mode this decides whether
    /// the goal is complete: if a turn ends with no tool calls, the agent is
    /// considered to have delivered its final result.
    pub(super) last_turn_had_tool_calls: bool,
    /// Whether the previous turn was interrupted by Ctrl+C. Both interruption and
    /// "natural no-tool completion" set `last_turn_had_tool_calls` to false, but
    /// with opposite semantics: the former does not mean the goal was achieved.
    /// Goal mode distinguishes them — when interrupted, keep goal_mode and fall
    /// back to waiting for user input, without printing the misleading
    /// "Goal achieved" message.
    pub(super) last_turn_interrupted: bool,

    /// LLM-guided context pruning counters (tool_call_id → consecutive mark count)
    pub(super) prune_marks: FxHashMap<String, u8>,

    /// Current-turn in-memory side channel: full `reasoning` output items captured
    /// via the Responses protocol (including encrypted_content), keyed by the
    /// "first tool_call id" of the assistant message carrying tool_calls. On
    /// replay they are spliced verbatim before the corresponding function_call so
    /// the model keeps the previous hop's reasoning context.
    ///
    /// Deliberately not placed in the `Message` struct: reasoning items are only
    /// useful for same-turn tool-chain replay and are stripped across turns by
    /// sqlite; putting them in the shared `Message` would pollute 295 literals and
    /// require `#[serde(skip)]` to keep them off disk. A turn-level bypass map
    /// (same pattern as `prune_marks`) is naturally turn-aligned, purely
    /// in-memory, and physically impossible to persist.
    pub(super) turn_reasoning_items: FxHashMap<String, Vec<Value>>,

    /// Runtime ledger of stale `apply_patch` targets: the set of target file paths
    /// whose last patch failed with `context mismatch` / `ambiguous patch` and for
    /// which no successful `read_file` / `write_file` / `apply_patch` has happened
    /// on the same path since.
    ///
    /// Semantics: a failed patch means the model's view of the file is stale;
    /// re-patching before re-fetching ground truth (successful read/write/patch)
    /// will just fail again, which is why guard [`patch_retry_requires_fresh_read`]
    /// rejects it.
    ///
    /// Why a dedicated ledger instead of scanning `messages`: history compression
    /// folds failed apply_patch groups into `internal_note` stubs (dropping the
    /// `role=tool` results and `assistant.tool_calls`), so a message-scan-based
    /// guard loses the stale state and cannot intercept retries. The ledger is
    /// maintained directly from tool execution results and synced to the current
    /// session's SQLite meta; it is reloaded on session switch/restore, so it is
    /// unaffected by message compression and cannot leak across sessions. Legacy
    /// databases replay it from still-visible structured messages on first load.
    pub(super) stale_patch_targets: FxHashSet<PathBuf>,
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
        if new_name != "anonymous" && self.observers.iter().any(|o| o.name() == new_name) {
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

    pub(super) fn refresh_prompt_editor_for_current_session(&mut self) {
        self.prompt_editor = Some(PromptEditor::new(
            &self.session_id,
            self.config.history_file.as_path(),
        ));
    }

    pub(super) fn sync_persona_session_binding(&mut self) {
        self.refresh_prompt_editor_for_current_session();
        let _ = crate::ai::persona::PersonaStore::new()
            .remember_session(&self.active_persona.id, &self.session_id);
    }

    pub(super) fn current_persona_memory_file(&self) -> PathBuf {
        crate::ai::persona::memory_file_for_persona(&self.active_persona.id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SkillBiasMemory {
    pub(super) skill_name: String,
    pub(super) question: String,
}

/// A skill that has explicitly requested user input. It only allows continuing
/// with the next ordinary user message and must not be treated as a vague
/// cross-turn skill preference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PendingSkillContinuation {
    /// Currently active skill list (when multiple skills stack, keep all of them,
    /// treated equally with no precedence).
    pub(super) skill_names: Vec<String>,
}

/// Entry point where the user explicitly specifies forced skills. Keeping the
/// source distinguishes command-parsing issues from injection issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForcedSkillSource {
    SkillsCommandInline,
    SkillsCommandNextTurn,
    InlineReference,
}

impl ForcedSkillSource {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::SkillsCommandInline => "/skills-inline",
            Self::SkillsCommandNextTurn => "/skills-next-turn",
            Self::InlineReference => "@skills",
        }
    }
}

/// Schema definition for a tool that can be offered to the AI model,
/// wrapping a function definition with a type discriminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolDefinition {
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
    pub(crate) function: FunctionDefinition,
}

/// Describes a callable function: its name, human-readable description,
/// and JSON Schema for parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

/// A request from the AI model to invoke a specific tool,
/// identified by a unique call ID and containing the function name and arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
    pub(crate) function: FunctionCall,
}

/// The function invocation details within a `ToolCall`,
/// containing the function name and a JSON-encoded argument string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

/// The output produced after executing a tool call,
/// linking back to the original call ID with the result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolResult {
    pub(crate) tool_call_id: String,
    pub(crate) content: String,
}

/// Runtime context for an agent, containing its available tools,
/// MCP server configurations, and iteration limits.
#[derive(Debug, Clone, Default)]
pub(super) struct AgentContext {
    pub(super) tools: Vec<ToolDefinition>,
    pub(super) mcp_servers: SkipMap<String, McpServerConfig>,
    pub(super) max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::ai) struct McpServerConfig {
    pub(in crate::ai) command: String,
    #[serde(default)]
    pub(in crate::ai) args: Vec<String>,
    #[serde(default)]
    pub(in crate::ai) env: SkipMap<String, String>,
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
    let cancelled = app.cancel_stream.swap(false, Ordering::Relaxed);
    if cancelled {
        crate::ai::driver::signal::clear_request_interrupt();
    }
    cancelled
}

pub(super) fn clear_stream_cancel(app: &App) {
    app.cancel_stream.store(false, Ordering::Relaxed);
    crate::ai::driver::signal::clear_request_interrupt();
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum StreamOutcome {
    #[default]
    Completed,
    EmptyResponse,
    /// This turn's response was truncated: the server returned
    /// `finish_reason=length` (output cap hit), or a tool call was dropped for
    /// incomplete arguments JSON, leaving the turn with no valid tool call. Neither
    /// should silently count as normal completion; retry automatically and prompt
    /// the model to shrink each output (e.g. chunked file writes).
    Truncated,
    Cancelled,
    ToolCall,
}

#[derive(Debug, Clone, Default)]
pub(super) struct StreamResult {
    pub(super) outcome: StreamOutcome,
    pub(super) tool_calls: Vec<ToolCall>,
    pub(super) assistant_text: String,
    pub(super) hidden_meta: String,
    /// Raw reasoning_content emitted by the model in thinking mode,
    /// passed back to the backend on multi-turn requests as the server requires.
    pub(super) reasoning_text: String,
    /// Full `reasoning` output items (including encrypted_content) captured from
    /// the Responses stream this turn. Only for same-turn tool-chain replay, passed
    /// through to the in-memory assistant message; never persisted to history.
    pub(super) reasoning_items: Vec<serde_json::Value>,
    pub(super) skip_response_drain: bool,
    /// Server returned finish_reason=length (output cap hit). Even when the outcome
    /// is Completed (with visible text), the flag is kept so upper layers can
    /// decide whether to inject an "output may be incomplete" hint.
    pub(super) truncated_by_length: bool,
    /// Truncation was caused by a stream read error (network jitter / server
    /// disconnect), not by the model hitting the output cap. The outcome is
    /// Truncated, but lowering reasoning_effort or injecting shrink hints is
    /// pointless — the model did not output too much; the server dropped the
    /// stream. Upper layers should do a plain retry rather than shrink-and-rewrite.
    pub(super) stream_error: bool,
    /// Raw finish_reason returned by the server (e.g. `stop` / `length` /
    /// `tool_calls`). Used for truncation diagnostics, distinguishing "output cap
    /// hit" from other causes.
    pub(super) finish_reason_value: Option<String>,
    /// Server usage stats: prompt tokens (normalized).
    pub(super) usage_prompt_tokens: u64,
    /// Server usage stats: cached_tokens hit within this prompt.
    pub(super) usage_cached_prompt_tokens: u64,
    /// Server usage stats: completion tokens (normalized, includes reasoning).
    pub(super) usage_completion_tokens: u64,
    /// Server usage stats: reasoning tokens (from completion_tokens_details).
    /// Some providers (GLM thinking mode) report reasoning tokens separately;
    /// this is the key metric for diagnosing "reasoning exhausting the budget
    /// causing zero-output truncation".
    pub(super) usage_reasoning_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionContext {
    /// The user's pure prompt input (excluding @file-injected content).
    /// Used for routing / intent detection / skill matching scenarios that need to
    /// see "user intent".
    pub(super) question: String,
    /// Additional text injected via @file, binaries (pdf), pending_files, etc.
    /// Only concatenated into the final user_message sent to the LLM; not used for
    /// routing/feature extraction.
    pub(super) attachments_text: String,
    pub(super) history_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct FileParseResult {
    pub(super) text_files: Vec<String>,
    pub(super) image_files: Vec<String>,
    pub(super) binary_files: Vec<String>,
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc};

    use super::PendingSkillContinuation;
    use crate::ai::{
        driver::observer::TurnObserver,
        middleware::{RequestMiddleware, ToolMiddleware, test_util::test_app},
        ports::{llm::LlmClient, stream::PassthroughFilter, tool::ToolExecutor},
        prompt::PromptEditor,
    };

    struct MarkerObserver;

    impl TurnObserver for MarkerObserver {}

    struct PassthroughToolMiddleware;

    impl ToolMiddleware for PassthroughToolMiddleware {
        fn name(&self) -> &'static str {
            "passthrough-tool"
        }

        fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
            inner
        }
    }

    struct PassthroughRequestMiddleware;

    impl RequestMiddleware for PassthroughRequestMiddleware {
        fn name(&self) -> &'static str {
            "passthrough-request"
        }

        fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            inner
        }
    }

    #[test]
    fn app_forks_reset_process_local_state_and_preserve_shared_handles() {
        let mut app = test_app();
        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec!["continuation".to_string()],
        });
        app.prompt_editor = Some(PromptEditor::new(
            "clone-characterization",
            Path::new("clone-characterization.sqlite"),
        ));
        app.observers = vec![Box::new(MarkerObserver)];
        app.tool_middlewares = vec![Arc::new(PassthroughToolMiddleware)];
        app.llm_middlewares = vec![Arc::new(PassthroughRequestMiddleware)];
        app.hooks
            .register_global_before("clone-characterization", |_| Ok(()));
        app.hooks.register_stream_filter(PassthroughFilter);
        app.forced_skills = vec!["source-skill".to_string()];
        app.prune_marks.insert("source-call".to_string(), 1);

        let mut cloned = app.fork_for_subagent();

        let driver_snapshot = app.snapshot_for_driver_context();
        let detached_snapshot = app.snapshot_for_detached_helper();
        assert!(driver_snapshot.pending_skill_continuation.is_none());
        assert!(driver_snapshot.tool_middlewares.is_empty());
        assert!(detached_snapshot.pending_skill_continuation.is_none());
        assert!(detached_snapshot.tool_middlewares.is_empty());

        assert!(cloned.pending_skill_continuation.is_none());
        assert!(cloned.prompt_editor.is_none());
        assert!(cloned.observers.is_empty());
        assert!(cloned.tool_middlewares.is_empty());
        assert!(cloned.llm_middlewares.is_empty());
        // Stream filters are content-transformation policy: forked turns must apply
        // the parent's registered filters (their output lands in parent history and
        // subagent payloads), while stage/global hook callbacks stay process-local.
        assert_eq!(cloned.hooks.stream_filters().len(), 1);
        assert_eq!(driver_snapshot.hooks.stream_filters().len(), 1);
        assert_eq!(detached_snapshot.hooks.stream_filters().len(), 1);
        // len() counts only the inherited filter: the registered global hook did not
        // propagate into any fork.
        assert_eq!(cloned.hooks.len(), 1);

        assert!(Arc::ptr_eq(&app.shutdown, &cloned.shutdown));
        assert!(Arc::ptr_eq(&app.streaming, &cloned.streaming));
        assert!(Arc::ptr_eq(&app.cancel_stream, &cloned.cancel_stream));
        assert!(Arc::ptr_eq(&app.os, &cloned.os));

        cloned.forced_skills.push("clone-only-skill".to_string());
        cloned.prune_marks.insert("clone-only-call".to_string(), 2);
        assert_eq!(app.forced_skills, ["source-skill"]);
        assert_eq!(app.prune_marks.len(), 1);
        assert_eq!(app.prune_marks.get("source-call"), Some(&1));

        assert!(app.pending_skill_continuation.is_some());
        assert!(app.prompt_editor.is_some());
        assert_eq!(app.observers.len(), 1);
        assert_eq!(app.tool_middlewares.len(), 1);
        assert_eq!(app.llm_middlewares.len(), 1);
        assert_eq!(app.hooks.len(), 2);
    }
}
