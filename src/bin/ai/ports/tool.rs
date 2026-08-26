// =============================================================================
// ToolExecutor - tool execution port (dependency inversion)
// =============================================================================
use std::future::Future;
use std::pin::Pin;
use crate::ai::{history::{Message, ToolExecutionOutcome}, types::{App, ToolCall, ToolResult}};

#[derive(Debug, Default)]
pub(crate) struct ToolExecOutput {
    pub tool_results: Vec<ToolResult>,
    /// Assistant messages that middlewares may inject (e.g. an auth-denial explanation); the current empty chain always yields none, and the driver consumes them on demand.
    pub assistant_messages: Vec<Message>,
    /// These fields pass through the full results of the actual dispatch so the driver can consume them losslessly (empty chain = identity, zero behavior change).
    pub executed_tool_calls: Vec<ToolCall>,
    pub cached_hits: Vec<bool>,
    pub execution_outcomes: Vec<Option<ToolExecutionOutcome>>,
    pub had_error: bool,
}

pub(crate) trait ToolExecutor: Send + Sync {
    fn execute<'a>(
        &'a self,
        app: &'a mut App,
        tool_calls: Vec<ToolCall>,
    ) -> Pin<Box<dyn Future<Output = Result<ToolExecOutput, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

