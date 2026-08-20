// =============================================================================
// ToolExecutor - 工具执行端口（依赖倒置）
// =============================================================================
use std::future::Future;
use std::pin::Pin;
use crate::ai::{history::{Message, ToolExecutionOutcome}, types::{App, ToolCall, ToolResult}};

#[derive(Debug, Default)]
pub(crate) struct ToolExecOutput {
    pub tool_results: Vec<ToolResult>,
    /// 中间件可注入的 assistant 消息（如鉴权拒绝说明）；当前空链恒为空，驱动按需消费。
    pub assistant_messages: Vec<Message>,
    /// 以下字段透传真实派发的完整结果，保证驱动无损失消费（空链 = 恒等，零行为变化）。
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

