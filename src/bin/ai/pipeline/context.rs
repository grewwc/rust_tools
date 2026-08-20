// =============================================================================
// PipelineContext - 流水线上下文（替代 driver 散传 7+ 参数）
// =============================================================================
use crate::ai::{history::Message, types::App};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageKind {
    Prepare,
    BudgetCheck,
    BuildRequest,
    SendRequest,
    ParseStream,
    /// 请求侧历史解码/流过滤占位（内层压缩 pipeline 用）；与 ParseStream（响应流解析）语义不同，
    /// 独立 kind 避免与 on_after_stream 的 ParseStream.after 触发点重复。
    Decode,
    ExecuteTools,
    Persist,
    Finalize,
}

/// 在整个 Pipeline 生命周期内共享的可变上下文
pub struct PipelineContext<'a> {
    pub app: &'a mut App,
    pub messages: Vec<Message>,
    pub turn_index: usize,
    pub stage: StageKind,
    /// 中间件可读写标签
    pub tags: Vec<String>,
}

impl<'a> PipelineContext<'a> {
    pub fn new(app: &'a mut App, messages: Vec<Message>, turn_index: usize) -> Self {
        Self { app, messages, turn_index, stage: StageKind::Prepare, tags: vec![] }
    }
    pub fn advance(&mut self, kind: StageKind) { self.stage = kind; }
}
