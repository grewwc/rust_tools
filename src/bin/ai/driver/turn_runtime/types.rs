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
    /// 文本内容的结构化关键行（函数/类型定义、错误行等），
    /// 为首次 overflow stub 提供召回锚点，让模型判断是否需要重新 read_file。
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
    /// 本轮响应被截断（服务端 finish_reason=length，或工具调用 arguments JSON
    /// 不完整被丢弃）。应注入"收缩单次输出"提示后自动重试，而非静默完成。
    Truncated(crate::ai::types::StreamResult),
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
