// =============================================================================
// PromptBuilder - Prompt 组装端口（依赖倒置）
// =============================================================================
// 之前 `prompt/*` 的组装过程被 driver 直接硬编码，无法插入项目级/用户级
// 中间件（如敏感词过滤、压缩策略替换）。现通过 trait 解耦。
use crate::ai::{history::Message, types::App};

#[derive(Debug, Default, Clone)]
pub(crate) struct PromptBuildRequest {
    pub messages: Vec<Message>,
    pub tool_names_hint: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct PromptBuildOutput {
    pub messages: Vec<Message>,
}

pub(crate) trait PromptBuilder: Send + Sync {
    fn build(&self, app: &App, req: PromptBuildRequest) -> PromptBuildOutput;
}

/// 默认实现：透传（保留现有 `prompt::build_system_prompt` 等逻辑，
//  后续在 `prompt` crate 中实现真实组装时仅替换此 impl）
pub(crate) struct DefaultPromptBuilder;

impl PromptBuilder for DefaultPromptBuilder {
    fn build(&self, _app: &App, req: PromptBuildRequest) -> PromptBuildOutput {
        PromptBuildOutput { messages: req.messages }
    }
}
