use serde_json::Value;

pub struct PrepareContext {
    pub question: String,
    /// Global, monotonically increasing turn index persisted within the session.
    /// Normal, resume, and internal turns each consume their own index.
    /// Observers can use it to decide whether to inject a "context budget" reminder.
    /// 0 means this is the session's first call.
    pub turn_index: usize,
    /// Names of the tools currently available to the agent; observers can use
    /// this to decide whether to inject a delegation hint.
    pub available_tool_names: Vec<String>,
}

pub struct ToolResultContext<'a> {
    pub tool_name: String,
    /// Raw tool result text. Borrowed rather than cloned to avoid copying
    /// N observers x 256K of markdown into O(N*M) memory. Observers that need
    /// to keep it can call to_string themselves.
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
