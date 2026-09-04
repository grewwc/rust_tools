// =============================================================================
// PipelineContext - pipeline context (replaces the driver's ad-hoc 7+ parameter passing)
// =============================================================================
use crate::ai::{history::Message, types::App};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageKind {
    Prepare,
    BudgetCheck,
    BuildRequest,
    SendRequest,
    ParseStream,
    /// Placeholder for request-side history decoding / stream filtering (used by the inner
    /// compression pipeline); semantically distinct from ParseStream (response stream parsing),
    /// with its own kind so it does not collide with the ParseStream.after trigger of on_after_stream.
    Decode,
    ExecuteTools,
    Persist,
    Finalize,
}

/// Mutable context shared across the entire Pipeline lifecycle
pub struct PipelineContext<'a> {
    pub app: &'a mut App,
    pub messages: Vec<Message>,
    pub turn_index: usize,
    pub stage: StageKind,
    /// Tags that middleware can read and write
    pub tags: Vec<String>,
}

impl<'a> PipelineContext<'a> {
    pub fn new(app: &'a mut App, messages: Vec<Message>, turn_index: usize) -> Self {
        Self {
            app,
            messages,
            turn_index,
            stage: StageKind::Prepare,
            tags: vec![],
        }
    }
    pub fn advance(&mut self, kind: StageKind) {
        self.stage = kind;
    }
}
