// =============================================================================
// Pipeline - turn pipeline abstraction
// =============================================================================
// Splits the former 400+ line flat process in `driver/turn_runtime/orchestrator.rs`
// into a composable chain of Stages, each depending only on the `ports::*`
// abstractions so middleware can be inserted easily.

pub(crate) mod context;
pub(crate) mod hook;
pub(crate) mod stage;
pub(crate) mod stages;
pub(crate) mod turn_pipeline;

#[allow(unused_imports)]
pub(crate) use stages::{CompressStage, DecodeStage};

#[allow(unused_imports)]
pub(crate) use context::{PipelineContext, StageKind};
#[allow(unused_imports)]
pub(crate) use hook::{HookEntry, HookFn, HookRegistry};
#[allow(unused_imports)]
pub(crate) use stage::{BoxStage, Pipeline, Stage};
#[allow(unused_imports)]
pub(crate) use turn_pipeline::{DefaultTurnPipeline, TurnPipeline};
