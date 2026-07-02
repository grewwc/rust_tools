use crate::ai::history::Message;
use rust_tools::commonw::FastSet;

pub(super) struct PreparedToolResult {
    pub(super) content_for_model: String,
    pub(super) content_for_terminal: String,
}

pub(super) struct LargeToolSummary {
    pub(super) body: String,
    pub(super) summary: String,
    pub(super) top_level_keys: Vec<String>,
    pub(super) field_samples: Vec<String>,
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
    FinalResponse(crate::ai::types::StreamResult),
    ToolCall(ToolCallExecution),
}

pub(super) enum TurnLoopStep {
    Continue,
    Break,
    Return(TurnOutcome),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver) enum TurnOutcome {
    Continue,
    Quit,
}
