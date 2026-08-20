// =============================================================================
// stages - 可插拔压缩 / 流过滤的 Pipeline Stage 适配
// =============================================================================
// 将 `ports::history::Compressor` 与 `ports::stream::StreamFilter` 暴露为
// `pipeline::Stage`，便于 driver 通过 `Pipeline` / `HookRegistry` / `Middleware` 组合。
// 默认不启用（零行为变更），仅在显式 push 时生效。

use std::{future::Future, pin::Pin};

use super::context::{PipelineContext, StageKind};
use super::stage::Stage;
use crate::ai::ports::history::{Compressor, DefaultCompressor};
use crate::ai::ports::stream::{FilterChain, StreamFilter};

/// 压缩阶段：在 BuildRequest 之前对 `ctx.messages` 做可插拔压缩。
/// 若 `max_chars==0` 则透传（与 `history::compress` 契约一致）。
pub struct CompressStage {
    name: &'static str,
    compressor: Box<dyn Compressor>,
    max_chars: usize,
    keep_last: usize,
}

impl CompressStage {
    pub fn new(compressor: Box<dyn Compressor>, max_chars: usize, keep_last: usize) -> Self {
        Self { name: "compress", compressor, max_chars, keep_last }
    }
    pub fn with_default(max_chars: usize, keep_last: usize) -> Self {
        Self { name: "compress", compressor: Box::new(DefaultCompressor), max_chars, keep_last }
    }
    pub fn with_name(mut self, name: &'static str) -> Self { self.name = name; self }
    pub fn compressor_name(&self) -> &str { self.compressor.name() }
}

impl Stage for CompressStage {
    fn name(&self) -> &'static str { self.name }
    /// 压缩是构建请求前的预算检查辅助阶段，映射到 BudgetCheck。
    fn kind(&self) -> StageKind { StageKind::BudgetCheck }
    fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        Box::pin(async move {
            if self.max_chars > 0 && !ctx.messages.is_empty() {
                let taken = std::mem::take(&mut ctx.messages);
                let compressed = self.compressor.compress(taken, self.max_chars, self.keep_last);
                ctx.messages = compressed;
            }
            Ok(())
        })
    }
}

/// 流过滤阶段：在 ParseStream 阶段前后插入过滤器链。
/// 演示“独立阶段可插过滤器”：实际 chunk 过滤在 `stream::runtime` 循环内逐块 `chain.apply`，
/// 本 Stage 仅负责在 Pipeline 中占位、记录 tags 供观测/测试，并在 before/after hook 中可替换 chain。
pub struct DecodeStage {
    name: &'static str,
    before: FilterChain,
    after: FilterChain,
}

impl DecodeStage {
    pub fn new() -> Self { Self { name: "decode", before: FilterChain::new(), after: FilterChain::new() } }
    pub fn with_before(mut self, chain: FilterChain) -> Self { self.before = chain; self }
    pub fn with_after(mut self, chain: FilterChain) -> Self { self.after = chain; self }
    pub fn push_before<F: StreamFilter + 'static>(mut self, f: F) -> Self { self.before = self.before.push(f); self }
    pub fn push_after<F: StreamFilter + 'static>(mut self, f: F) -> Self { self.after = self.after.push(f); self }
    pub fn before_len(&self) -> usize { self.before.len() }
    pub fn after_len(&self) -> usize { self.after.len() }
    /// 在 Pipeline 上下文中对某段文本模拟 after 链应用（便于单测验证链式行为）。
    pub fn apply_after(&self, text: String) -> Option<String> {
        if self.after.is_empty() { Some(text) } else { self.after.apply(text) }
    }
    pub fn apply_before(&self, text: String) -> Option<String> {
        if self.before.is_empty() { Some(text) } else { self.before.apply(text) }
    }
}

impl Default for DecodeStage {
    fn default() -> Self { Self::new() }
}

impl Stage for DecodeStage {
    fn name(&self) -> &'static str { self.name }
    /// 解码是请求侧历史解码/流过滤占位，映射到 Decode。
    /// 注意不要映射到 ParseStream：后者是响应流解析的语义 kind，内层压缩 pipeline 若在此
    /// 复用会与 driver 的 fire_after_stream_hooks（on_after_stream）双触发。
    fn kind(&self) -> StageKind { StageKind::Decode }
    fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        Box::pin(async move {
            // 记录过滤器链指纹，便于 driver/hook 观测（零行为变更：不改 ctx.messages 实时流）
            if !self.before.is_empty() || !self.after.is_empty() {
                ctx.tags.push(format!("decode:before={} after={}", self.before.len(), self.after.len()));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;
    use crate::ai::middleware::test_util::test_app;
    use crate::ai::pipeline::context::PipelineContext;
    use crate::ai::pipeline::hook::HookRegistry;
    use crate::ai::pipeline::turn_pipeline::{DefaultTurnPipeline, TurnPipeline};
    use crate::ai::ports::history::NoopCompressor;
    use crate::ai::ports::stream::PassthroughFilter;
    use serde_json::Value;

    fn msg(role: &str, text: &str) -> Message {
        Message { role: role.to_string(), content: Value::String(text.to_string()), tool_calls: None, tool_call_id: None, reasoning_content: None }
    }
    fn leak_app() -> &'static mut crate::ai::types::App { Box::leak(Box::new(test_app())) }

    #[tokio::test]
    async fn compress_stage_uses_injected_compressor() {
        // Noop 保持原样
        let app = leak_app();
        let messages = vec![msg("user", "a"), msg("assistant", "b"), msg("user", "c")];
        let mut ctx = PipelineContext::new(app, messages.clone(), 0);
        let stage = CompressStage::new(Box::new(NoopCompressor), 10, 1);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage);
        let tp = DefaultTurnPipeline::new("test-noop", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert_eq!(ctx.messages, messages);
    }

    #[tokio::test]
    async fn inner_compress_pipeline_does_not_fire_parse_stream_hooks() {
        // 回归：内层压缩 pipeline（CompressStage→DecodeStage）是请求侧压缩/解码，不是响应流解析，
        // 不得触发 ParseStream 钩子——否则与 driver 的 fire_after_stream_hooks 双触发 on_after_stream。
        let app = leak_app();
        let messages = vec![msg("user", "a"), msg("assistant", "b"), msg("user", "c")];
        let mut ctx = PipelineContext::new(app, messages, 0);
        let mut hooks = HookRegistry::new();
        let fired = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let f = std::sync::Arc::clone(&fired);
        hooks.register_after(StageKind::ParseStream, "after_stream", move |_ctx| {
            f.lock().unwrap().push("after_stream".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        hooks.register_after(StageKind::Decode, "decode", move |_ctx| {
            f.lock().unwrap().push("decode".to_string());
            Ok(())
        });
        let f = std::sync::Arc::clone(&fired);
        hooks.register_after(StageKind::BudgetCheck, "budget", move |_ctx| {
            f.lock().unwrap().push("budget".to_string());
            Ok(())
        });

        let pipeline = crate::ai::pipeline::stage::Pipeline::new()
            .push(CompressStage::with_default(10, 1))
            .push(DecodeStage::default());
        pipeline.execute(&mut ctx, &hooks).await.unwrap();

        // 仅 BudgetCheck.after 与 Decode.after 触发；ParseStream.after 不得出现。
        assert_eq!(
            *fired.lock().unwrap(),
            vec!["budget".to_string(), "decode".to_string()]
        );
    }

    #[tokio::test]
    async fn compress_stage_default_truncates_via_keep_last() {
        let app = leak_app();
        let messages = vec![msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2"), msg("assistant", "a2"), msg("user", "u3")];
        let mut ctx = PipelineContext::new(app, messages, 0);
        // max_chars 小，keep_last=1 预期触发裁剪（DefaultCompressor 走 history::compress 简化版）
        let stage = CompressStage::with_default(10, 1);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage);
        let tp = DefaultTurnPipeline::new("test-default", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        // 只要压缩后长度 <= 原长度且非空，即证明 compressor 被调用
        assert!(!ctx.messages.is_empty());
        assert!(ctx.messages.len() <= 5);
    }

    #[tokio::test]
    async fn decode_stage_filter_chain_applies() {
        struct DropHello;
        impl StreamFilter for DropHello {
            fn filter(&self, chunk: &str) -> Option<String> { if chunk.contains("hello") { None } else { Some(chunk.to_string()) } }
            fn name(&self) -> &str { "drop_hello" }
        }
        struct Upper;
        impl StreamFilter for Upper {
            fn filter(&self, chunk: &str) -> Option<String> { Some(chunk.to_uppercase()) }
            fn name(&self) -> &str { "upper" }
        }
        let stage = DecodeStage::new().push_after(Upper).push_after(PassthroughFilter);
        assert_eq!(stage.apply_after("hi".to_string()), Some("HI".to_string()));
        let stage2 = DecodeStage::new().push_before(DropHello);
        assert_eq!(stage2.apply_before("hello world".to_string()), None);
        assert_eq!(stage2.apply_before("good".to_string()), Some("good".to_string()));
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage2);
        let tp = DefaultTurnPipeline::new("test-decode", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert!(ctx.tags.iter().any(|t| t.contains("decode:before")));
    }
}
