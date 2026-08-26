// =============================================================================
// TurnPipeline - turn-level pipeline trait (abstracts the orchestrator's flat
// flow into a replaceable object)
// =============================================================================
use std::{future::Future, pin::Pin};
use super::context::{PipelineContext, StageKind};
use super::hook::HookRegistry;
use super::stage::Pipeline;

/// Turn pipeline abstraction: the driver depends only on this trait, not on the
/// concrete stage composition or hook implementations. This eases unit-test
/// mocking, middleware decoration, and multi-tenant/multi-model branch replacement.
pub trait TurnPipeline: Send + Sync {
    fn name(&self) -> &'static str;
    /// Execute the whole turn (may trigger HookRegistry before/after hooks internally).
    fn run<'a>(
        &'a self,
        ctx: &'a mut PipelineContext<'_>,
        hooks: &'a HookRegistry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

/// Default implementation: wraps `Pipeline` + hook firing (before/after per StageKind order).
pub struct DefaultTurnPipeline {
    name: &'static str,
    pipeline: Pipeline,
}

impl DefaultTurnPipeline {
    pub fn new(name: &'static str, pipeline: Pipeline) -> Self { Self { name, pipeline } }
    pub fn pipeline(&self) -> &Pipeline { &self.pipeline }
}

impl TurnPipeline for DefaultTurnPipeline {
    fn name(&self) -> &'static str { self.name }
    fn run<'a>(
        &'a self,
        ctx: &'a mut PipelineContext<'_>,
        hooks: &'a HookRegistry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        // Fire global before hooks, run the Pipeline (executing stages in
        // sequence, including per-stage hooks), then fire global after hooks.
        // `Stage::execute` / `Pipeline::execute` decouple ctx's borrow lifetime
        // from `PipelineContext`'s internal lifetime (separate `'a`/`'b`), so we
        // can execute each segment with safe sequential `&mut` reborrows, no raw
        // pointers; the future also stays Send.
        let pipeline = &self.pipeline;
        Box::pin(async move {
            hooks.fire_before(ctx, StageKind::Prepare)?;
            pipeline.execute(ctx, hooks).await?;
            hooks.fire_after(ctx, StageKind::Finalize)?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::pipeline::context::{PipelineContext, StageKind};
    use crate::ai::pipeline::hook::HookRegistry;
    use crate::ai::pipeline::stage::Stage;
    use crate::ai::types::App;

    struct TagStage(&'static str);
    impl Stage for TagStage {
        fn name(&self) -> &'static str { self.0 }
        fn kind(&self) -> StageKind { StageKind::BuildRequest }
        fn execute<'a, 'b>(
            &'a self,
            ctx: &'a mut PipelineContext<'b>,
        ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
            Box::pin(async move {
                ctx.tags.push(self.0.to_string());
                Ok(())
            })
        }
    }

    fn leak_app() -> &'static mut App {
        let app = crate::ai::middleware::test_util::test_app();
        Box::leak(Box::new(app))
    }

    #[tokio::test]
    async fn turn_pipeline_runs_with_hooks() {
        let pipeline = Pipeline::new().push(TagStage("s1")).push(TagStage("s2"));
        let tp = DefaultTurnPipeline::new("test", pipeline);
        let mut hooks = HookRegistry::new();
        hooks.register_global_before("gb", |ctx| { ctx.tags.push("before".into()); Ok(()) });
        hooks.register_global_after("ga", |ctx| { ctx.tags.push("after".into()); Ok(()) });
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert_eq!(ctx.tags, vec!["before", "s1", "s2", "after"]);
        let _ = StageKind::Prepare;
    }

    #[tokio::test]
    async fn per_stage_hooks_fire() {
        // Each stage's kind() is BuildRequest, so registered per-stage hooks
        // should fire around every stage, and global hooks (before/after) must
        // not be re-fired per stage.
        let pipeline = Pipeline::new().push(TagStage("s1")).push(TagStage("s2"));
        let tp = DefaultTurnPipeline::new("test-per-stage", pipeline);
        let mut hooks = HookRegistry::new();
        hooks.register_before(StageKind::BuildRequest, "b", |ctx| { ctx.tags.push("b".into()); Ok(()) });
        hooks.register_after(StageKind::BuildRequest, "a", |ctx| { ctx.tags.push("a".into()); Ok(()) });
        hooks.register_global_before("gb", |ctx| { ctx.tags.push("before".into()); Ok(()) });
        hooks.register_global_after("ga", |ctx| { ctx.tags.push("after".into()); Ok(()) });
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        tp.run(&mut ctx, &hooks).await.unwrap();
        // Per-stage hooks fire once around each stage; global hooks fire once at
        // the very beginning and end of the turn.
        assert_eq!(ctx.tags, vec!["before", "b", "s1", "a", "b", "s2", "a", "after"]);
    }
}
