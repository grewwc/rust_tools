// =============================================================================
// Stage - 流水线阶段 trait
// =============================================================================
use super::context::{PipelineContext, StageKind};
use super::hook::HookRegistry;
use std::{future::Future, pin::Pin};

pub trait Stage: Send + Sync {
    fn name(&self) -> &'static str;
    /// 本阶段对应的语义 StageKind，用于在 execute 前后触发 per-stage 钩子。
    fn kind(&self) -> StageKind;
    fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
    ) -> Pin<
        Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>,
    >;
}

pub type BoxStage = Box<dyn Stage>;

/// 线性流水线：按注册顺序迭代执行 Stage 链，每个 stage 前后触发 per-stage 钩子。
pub struct Pipeline {
    stages: Vec<BoxStage>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }
    pub fn push<S: Stage + 'static>(mut self, s: S) -> Self {
        self.stages.push(Box::new(s));
        self
    }
    pub fn push_boxed(mut self, s: BoxStage) -> Self {
        self.stages.push(s);
        self
    }

    pub fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
        hooks: &'a HookRegistry,
    ) -> Pin<
        Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>,
    > {
        // A failed before hook or stage skips that stage's after hooks;
        // any error stops the remaining stages.
        Box::pin(async move {
            for stage in &self.stages {
                let kind = stage.kind();
                hooks.fire_stage_before(ctx, kind)?;
                stage.execute(ctx).await?;
                hooks.fire_stage_after(ctx, kind)?;
            }
            Ok(())
        })
    }
}
