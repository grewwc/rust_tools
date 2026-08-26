// =============================================================================
// Pipeline - turn pipeline abstraction
// =============================================================================
// Splits the former 400+ line flat process in `driver/turn_runtime/orchestrator.rs`
// into a composable chain of Stages, each depending only on the `ports::*`
// abstractions so middleware can be inserted easily.

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
