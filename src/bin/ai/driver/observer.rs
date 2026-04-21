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

pub struct ObserverOutput {
    pub display_lines: Vec<String>,
    pub memory_entries: Vec<MemoryEntry>,
}

pub struct MemoryEntry {
    pub category: String,
    pub note: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub priority: u8,
}

impl MemoryEntry {
    pub fn to_agent_memory_entry(&self) -> crate::ai::tools::storage::memory_store::AgentMemoryEntry {
        crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: None,
            timestamp: chrono::Local::now().to_rfc3339(),
            category: self.category.clone(),
            note: self.note.clone(),
            tags: self.tags.clone(),
            source: self.source.clone(),
            priority: Some(self.priority),
            owner_pid: None,
            owner_pgid: None,
        }
    }
}

pub trait TurnObserver: Send + Sync {
    fn on_prepare(&mut self, ctx: &PrepareContext) -> Vec<(String, String)> {
        let _ = ctx;
        Vec::new()
    }

    fn on_tool_result(&mut self, ctx: &ToolResultContext) {
        let _ = ctx;
    }

    fn on_finalize(&mut self, ctx: &FinalizeContext) -> ObserverOutput {
        let _ = ctx;
        ObserverOutput {
            display_lines: Vec::new(),
            memory_entries: Vec::new(),
        }
    }

    fn on_conversation_end(&mut self) {
    }

    fn name(&self) -> &str {
        "anonymous"
    }
}
