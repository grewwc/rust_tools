use serde_json::Value;

pub struct PrepareContext {
    pub question: String,
}

pub struct ToolResultContext {
    pub tool_name: String,
    pub result_content: String,
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

    fn on_tool_result(&mut self, ctx: &ToolResultContext) {
        let _ = ctx;
    }

    fn on_finalize(&mut self, ctx: &FinalizeContext) -> ObserverOutput {
        let _ = ctx;
        ObserverOutput {
            display_lines: Vec::new(),
        }
    }

    fn on_conversation_end(&mut self) {
    }

    fn name(&self) -> &str {
        "anonymous"
    }

    fn is_poisoned(&self) -> bool {
        false
    }

    fn mark_poisoned(&mut self) {
    }
}
