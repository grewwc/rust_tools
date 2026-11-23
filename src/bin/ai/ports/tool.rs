// =============================================================================
// ToolExecutor - tool execution port (dependency inversion)
// =============================================================================
use crate::ai::{
    history::{Message, ToolExecutionOutcome},
    types::{App, ToolCall, ToolResult},
};
use rust_tools::commonw::FastSet;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Default)]
pub(crate) struct ToolExecOutput {
    pub tool_results: Vec<ToolResult>,
    /// Assistant messages injected by middleware, such as an auth-denial explanation.
    pub assistant_messages: Vec<Message>,
    /// Dispatch details retained so the driver can consume results losslessly.
    pub executed_tool_calls: Vec<ToolCall>,
    pub cached_hits: Vec<bool>,
    pub execution_outcomes: Vec<Option<ToolExecutionOutcome>>,
    /// Model-visible failures that were never executed and have no durable outcome.
    pub pre_dispatch_error_ids: FastSet<String>,
    pub had_error: bool,
}

pub(crate) trait ToolExecutor: Send + Sync {
    fn execute<'a>(
        &'a self,
        app: &'a mut App,
        tool_calls: Vec<ToolCall>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<ToolExecOutput, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + 'a,
        >,
    >;
}
