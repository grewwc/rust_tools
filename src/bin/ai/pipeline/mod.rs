// =============================================================================
// Pipeline - Turn 流水线抽象
// =============================================================================
// 将原来 `driver/turn_runtime/orchestrator.rs` 中 400+ 行的扁平过程拆为
// 可组合的 Stage 链，每一 Stage 只依赖 `ports::*` 抽象，便于中间件插入。

pub(crate) mod context;
pub(crate) mod stage;
pub(crate) mod hook;
pub(crate) mod turn_pipeline;
pub(crate) mod stages;

#[allow(unused_imports)]
pub(crate) use stages::{CompressStage, DecodeStage};

#[allow(unused_imports)]
pub(crate) use context::{PipelineContext, StageKind};
#[allow(unused_imports)]
pub(crate) use stage::{BoxStage, Pipeline, Stage};
#[allow(unused_imports)]
pub(crate) use hook::{HookRegistry, HookEntry, HookFn};
#[allow(unused_imports)]
pub(crate) use turn_pipeline::{DefaultTurnPipeline, TurnPipeline};
