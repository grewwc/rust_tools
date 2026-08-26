use crate::ai::history::Message;
use rust_tools::commonw::FastSet;
use std::path::PathBuf;

pub(super) struct PreparedToolResult {
    pub(super) content_for_model: String,
    pub(super) content_for_terminal: String,
}

pub(super) struct LargeToolSummary {
    pub(super) body: String,
    pub(super) summary: String,
    pub(super) top_level_keys: Vec<String>,
    pub(super) field_samples: Vec<String>,
    /// Structured key lines of the text content (function/type definitions, error lines, etc.),
    /// providing recall anchors for the first overflow stub so the model can decide
    /// whether it needs to re-read the file with read_file.
    pub(super) key_lines: Vec<String>,
}

pub(super) struct TurnPreparation {
    pub(super) skill_turn: super::super::skill_runtime::SkillTurnGuard,
    pub(super) messages: Vec<Message>,
    pub(super) turn_messages: Vec<Message>,
    pub(super) persisted_turn_messages: usize,
    pub(super) max_iterations: usize,
}

pub(super) struct ToolCallExecution {
    pub(super) stream_result: crate::ai::types::StreamResult,
    pub(super) allowed_tool_names: FastSet<String>,
}

pub(super) enum IterationExecution {
    Exit(TurnOutcome),
    RequestFailed(String),
    EmptyResponse,
    /// The response for this round was truncated (server finish_reason=length, or a tool-call
    /// arguments JSON was dropped as incomplete). Retry automatically after injecting a
    /// "shrink a single output" hint, rather than finishing silently.
    Truncated(crate::ai::types::StreamResult),
    /// The pre-timeout wrap-up signal fires mid model request: abandon the current request and
    /// have the orchestrator immediately enter a forced wrap-up (no-tool) iteration instead of
    /// waiting for the current iteration to end naturally.
    WrapUpFinal,
    FinalResponse(crate::ai::types::StreamResult),
    ToolCall(ToolCallExecution),
}

pub(super) enum TurnLoopStep {
    Continue,
    /// This round only ran the target-scoped instruction preflight and produced no file side
    /// effects. The orchestrator may grant it one retry that does not count against the normal
    /// tool-iteration budget.
    ScopedPreflightContinue(Vec<PathBuf>),
    Break,
    Return(TurnOutcome),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver) enum TurnOutcome {
    Continue,
    Quit,
}
