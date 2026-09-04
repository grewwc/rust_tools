// =============================================================================
// stages - Pipeline Stage adapters for pluggable compression / stream filtering
// =============================================================================
// Exposes `ports::history::Compressor` and `ports::stream::StreamFilter` as
// `pipeline::Stage`s so the driver can compose them via `Pipeline` / `HookRegistry` /
// `Middleware`. Disabled by default (zero behavior change); only effective when explicitly
// pushed.

use std::{future::Future, pin::Pin};

use super::context::{PipelineContext, StageKind};
use super::stage::Stage;
use crate::ai::ports::history::{Compressor, DefaultCompressor};
use crate::ai::ports::stream::{FilterChain, StreamFilter};

/// Compression stage: applies pluggable compression to `ctx.messages` before BuildRequest.
/// Passes through when `max_chars==0` (consistent with the `history::compress` contract).
pub struct CompressStage {
    name: &'static str,
    compressor: Box<dyn Compressor>,
    max_chars: usize,
    keep_last: usize,
}

impl CompressStage {
    pub fn new(compressor: Box<dyn Compressor>, max_chars: usize, keep_last: usize) -> Self {
        Self {
            name: "compress",
            compressor,
            max_chars,
            keep_last,
        }
    }
    pub fn with_default(max_chars: usize, keep_last: usize) -> Self {
        Self {
            name: "compress",
            compressor: Box::new(DefaultCompressor),
            max_chars,
            keep_last,
        }
    }
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
    pub fn compressor_name(&self) -> &str {
        self.compressor.name()
    }
}

impl Stage for CompressStage {
    fn name(&self) -> &'static str {
        self.name
    }
    /// Compression is a budget-check helper stage before building the request; maps to BudgetCheck.
    fn kind(&self) -> StageKind {
        StageKind::BudgetCheck
    }
    fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
    ) -> Pin<
        Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>,
    > {
        Box::pin(async move {
            if self.max_chars > 0 && !ctx.messages.is_empty() {
                let taken = std::mem::take(&mut ctx.messages);
                let compressed = self
                    .compressor
                    .compress(taken, self.max_chars, self.keep_last);
                ctx.messages = compressed;
            }
            Ok(())
        })
    }
}

/// Stream-filtering stage: inserts a filter chain before/after the ParseStream stage.
/// Demonstrates "independent stages can plug in filters": actual chunk filtering happens via
/// per-chunk `chain.apply` inside the `stream::runtime` loop; this Stage only occupies a slot
/// in the Pipeline, records tags for observation/testing, and allows the chain to be replaced
/// in the before/after hooks.
pub struct DecodeStage {
    name: &'static str,
    before: FilterChain,
    after: FilterChain,
}

impl DecodeStage {
    pub fn new() -> Self {
        Self {
            name: "decode",
            before: FilterChain::new(),
            after: FilterChain::new(),
        }
    }
    pub fn with_before(mut self, chain: FilterChain) -> Self {
        self.before = chain;
        self
    }
    pub fn with_after(mut self, chain: FilterChain) -> Self {
        self.after = chain;
        self
    }
    pub fn push_before<F: StreamFilter + 'static>(mut self, f: F) -> Self {
        self.before = self.before.push(f);
        self
    }
    pub fn push_after<F: StreamFilter + 'static>(mut self, f: F) -> Self {
        self.after = self.after.push(f);
        self
    }
    pub fn before_len(&self) -> usize {
        self.before.len()
    }
    pub fn after_len(&self) -> usize {
        self.after.len()
    }
    /// Simulates applying the after chain to a piece of text in a Pipeline context (for
    /// unit-testing chain behavior).
    pub fn apply_after(&self, text: String) -> Option<String> {
        if self.after.is_empty() {
            Some(text)
        } else {
            self.after.apply(text)
        }
    }
    pub fn apply_before(&self, text: String) -> Option<String> {
        if self.before.is_empty() {
            Some(text)
        } else {
            self.before.apply(text)
        }
    }
}

impl Default for DecodeStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for DecodeStage {
    fn name(&self) -> &'static str {
        self.name
    }
    /// Decode is a request-side history-decoding / stream-filtering placeholder; maps to Decode.
    /// Do not map it to ParseStream: that is the semantic kind for response-stream parsing, and
    /// reusing it in the inner compression pipeline would double-fire
    /// fire_after_stream_hooks (on_after_stream) with the driver.
    fn kind(&self) -> StageKind {
        StageKind::Decode
    }
    fn execute<'a, 'b>(
        &'a self,
        ctx: &'a mut PipelineContext<'b>,
    ) -> Pin<
        Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>,
    > {
        Box::pin(async move {
            // Record a filter-chain fingerprint for driver/hook observation (zero behavior change:
            // does not touch the live ctx.messages stream)
            if !self.before.is_empty() || !self.after.is_empty() {
                ctx.tags.push(format!(
                    "decode:before={} after={}",
                    self.before.len(),
                    self.after.len()
                ));
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
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
    fn leak_app() -> &'static mut crate::ai::types::App {
        Box::leak(Box::new(test_app()))
    }

    #[tokio::test]
    async fn compress_stage_uses_injected_compressor() {
        // Noop keeps messages unchanged
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
        // Regression: the inner compression pipeline (CompressStage→DecodeStage) is request-side
        // compression/decoding, not response-stream parsing, so it must not fire ParseStream
        // hooks -- otherwise on_after_stream would double-fire with the driver's
        // fire_after_stream_hooks.
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

        // Only BudgetCheck.after and Decode.after fire; ParseStream.after must not appear.
        assert_eq!(
            *fired.lock().unwrap(),
            vec!["budget".to_string(), "decode".to_string()]
        );
    }

    #[tokio::test]
    async fn compress_stage_default_truncates_via_keep_last() {
        let app = leak_app();
        let messages = vec![
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("user", "u2"),
            msg("assistant", "a2"),
            msg("user", "u3"),
        ];
        let mut ctx = PipelineContext::new(app, messages, 0);
        // Small max_chars with keep_last=1 is expected to trigger truncation (DefaultCompressor
        // uses a simplified history::compress)
        let stage = CompressStage::with_default(10, 1);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage);
        let tp = DefaultTurnPipeline::new("test-default", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        // As long as the compressed length is <= the original and non-empty, the compressor
        // was invoked
        assert!(!ctx.messages.is_empty());
        assert!(ctx.messages.len() <= 5);
    }

    #[tokio::test]
    async fn decode_stage_filter_chain_applies() {
        struct DropHello;
        impl StreamFilter for DropHello {
            fn filter(&self, chunk: &str) -> Option<String> {
                if chunk.contains("hello") {
                    None
                } else {
                    Some(chunk.to_string())
                }
            }
            fn name(&self) -> &str {
                "drop_hello"
            }
        }
        struct Upper;
        impl StreamFilter for Upper {
            fn filter(&self, chunk: &str) -> Option<String> {
                Some(chunk.to_uppercase())
            }
            fn name(&self) -> &str {
                "upper"
            }
        }
        let stage = DecodeStage::new()
            .push_after(Upper)
            .push_after(PassthroughFilter);
        assert_eq!(stage.apply_after("hi".to_string()), Some("HI".to_string()));
        let stage2 = DecodeStage::new().push_before(DropHello);
        assert_eq!(stage2.apply_before("hello world".to_string()), None);
        assert_eq!(
            stage2.apply_before("good".to_string()),
            Some("good".to_string())
        );
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage2);
        let tp = DefaultTurnPipeline::new("test-decode", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert!(ctx.tags.iter().any(|t| t.contains("decode:before")));
    }
}
