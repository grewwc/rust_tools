// =============================================================================
// TurnPipeline - Turn 级流水线 Trait（将 orchestrator 扁平流程抽象为可替换对象）
// =============================================================================
use std::{future::Future, pin::Pin};
use super::context::{PipelineContext, StageKind};
use super::hook::HookRegistry;
use super::stage::Pipeline;

/// Turn 流水线抽象：driver 只依赖此 trait，不依赖具体 stage 组合与钩子实现。
/// 便于单测 mock、中间件装饰、以及多租户/多模型分支替换。
pub trait TurnPipeline: Send + Sync {
    fn name(&self) -> &'static str;
    /// 执行整个 turn（内部可按需触发 HookRegistry 的 before/after）。
    fn run<'a>(
        &'a self,
        ctx: &'a mut PipelineContext<'_>,
        hooks: &'a HookRegistry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

/// 默认实现：包裹 `Pipeline` + 钩子触发（按 StageKind 顺序触发 before/after）。
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
        // 触发全局 before → Pipeline（迭代执行 stage，含 per-stage 钩子）→ 全局 after。
        // `Stage::execute` / `Pipeline::execute` 将 ctx 的借用生命周期与
        // `PipelineContext` 的内部生命周期解耦（`'a`/`'b` 分离），因此这里可以
        // 用安全顺序 `&mut` 重借逐段执行，无需裸指针；future 也可保持 Send。
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
        // 每个 stage 的 kind() 对应 BuildRequest，注册的 per-stage 钩子应环绕每个 stage 触发，
        // 且全局钩子（before/after）不被 stage 级触发重复调用。
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
        // 每个 stage 前后各触发一次 per-stage 钩子，全局钩子仅在 turn 首尾各一次。
        assert_eq!(ctx.tags, vec!["before", "b", "s1", "a", "b", "s2", "a", "after"]);
    }
}
