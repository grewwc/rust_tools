use serde_json::Value;

pub struct PrepareContext {
    pub question: String,
    /// 当前 turn 在 session 内的序号（单调递增），
    /// 用 history_count 填充，作为上下文膨胀程度的代理指标。
    /// observer 可据此决定是否注入"上下文预算"提醒。
    /// 默认 0 表示首次调用（无历史消息）。
    pub turn_index: usize,
    /// 当前 agent 可用的工具名列表，observer 可据此决定是否注入委派提示。
    pub available_tool_names: Vec<String>,
}

pub struct ToolResultContext<'a> {
    pub tool_name: String,
    /// 工具结果原文：用借用而非 clone，避免 N 个 observer × 256K markdown
    /// 时把内存复制成 O(N·M)。observer 内部如需保存可自行 to_string。
    pub result_content: &'a str,
    pub success: bool,
}

pub struct FinalizeContext {
    pub question: String,
    pub final_text: String,
    pub had_tool_calls: bool,
}

#[derive(Debug, Clone)]
pub struct SuggestedToolCall {
    pub tool_name: String,
    pub arguments: Value,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionKind {
    Behavior,
    Fact,
}

pub struct PrepareOutput {
    pub sections: Vec<(SectionKind, String, String)>,
    pub suggested_tool_calls: Vec<SuggestedToolCall>,
}

impl PrepareOutput {
    pub fn empty() -> Self {
        Self {
            sections: Vec::new(),
            suggested_tool_calls: Vec::new(),
        }
    }
}

pub struct ObserverOutput {
    pub display_lines: Vec<String>,
}

pub trait TurnObserver: Send + Sync {
    fn on_prepare(&mut self, ctx: &PrepareContext) -> Vec<(String, String)> {
        let _ = ctx;
        Vec::new()
    }

    fn on_prepare_rich(&mut self, ctx: &PrepareContext) -> PrepareOutput {
        let legacy = self.on_prepare(ctx);
        PrepareOutput {
            sections: legacy
                .into_iter()
                .map(|(kind, content)| {
                    let kind_enum = if kind == "Behavior" {
                        SectionKind::Behavior
                    } else {
                        SectionKind::Fact
                    };
                    (kind_enum, kind, content)
                })
                .collect(),
            suggested_tool_calls: Vec::new(),
        }
    }

    fn on_tool_result(&mut self, ctx: &ToolResultContext<'_>) {
        let _ = ctx;
    }

    fn on_finalize(&mut self, ctx: &FinalizeContext) -> ObserverOutput {
        let _ = ctx;
        ObserverOutput {
            display_lines: Vec::new(),
        }
    }

    fn on_conversation_end(&mut self) {}

    fn name(&self) -> &str {
        "anonymous"
    }

    fn is_poisoned(&self) -> bool {
        false
    }

    fn mark_poisoned(&mut self) {}
}
