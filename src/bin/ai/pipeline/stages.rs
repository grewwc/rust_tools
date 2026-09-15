// =============================================================================
// stages - Pipeline Stage adapters for pluggable compression / stream filtering
// =============================================================================
// Exposes `ports::history::Compressor` and `ports::stream::StreamFilter` as
// `pipeline::Stage`s so the driver can compose them via `Pipeline` / `HookRegistry` /
// `Middleware`. Disabled by default (zero behavior change); only effective when explicitly
// pushed.

use std::{future::Future, path::PathBuf, pin::Pin};

use super::context::{PipelineContext, StageKind};
use super::stage::Stage;
use crate::ai::history::{
    ContextCompressionOutcome, ContextCompressionStatus, messages_total_chars_pub,
};
use crate::ai::ports::history::{Compressor, DefaultCompressor};
use crate::ai::ports::stream::{FilterChain, StreamFilter};

/// Compression stage: applies pluggable compression to `ctx.messages` before BuildRequest.
/// Passes through when `max_chars==0` (consistent with the `history::compress` contract).
pub struct CompressStage {
    name: &'static str,
    compressor: Box<dyn Compressor>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
}

impl CompressStage {
    pub fn new(compressor: Box<dyn Compressor>, max_chars: usize, keep_last: usize) -> Self {
        Self {
            name: "compress",
            compressor,
            max_chars,
            keep_last,
            summary_max_chars: 0,
            overflow_dir: None,
            cwd: None,
        }
    }
    pub fn with_default(max_chars: usize, keep_last: usize) -> Self {
        Self::new(Box::new(DefaultCompressor), max_chars, keep_last)
    }
    /// Configures the archive sink explicitly; without it, old dialogue is retained and the
    /// outcome tags report why compaction could not proceed.
    pub fn with_context_options(
        mut self,
        summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<PathBuf>,
    ) -> Self {
        self.summary_max_chars = summary_max_chars;
        self.overflow_dir = overflow_dir;
        self.cwd = cwd;
        self
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
            let messages = std::mem::take(&mut ctx.messages);
            let outcome = if self.max_chars == 0 || messages.is_empty() {
                let status = if self.max_chars == 0 {
                    ContextCompressionStatus::Disabled
                } else {
                    ContextCompressionStatus::Complete
                };
                let before_chars = messages_total_chars_pub(&messages);
                ContextCompressionOutcome::new(messages, before_chars, self.max_chars, status)
            } else {
                self.compressor.compress(
                    messages,
                    self.max_chars,
                    self.keep_last,
                    self.summary_max_chars,
                    self.overflow_dir.clone(),
                    self.cwd.as_deref(),
                )
            };
            ctx.tags.push(format!(
                "compress:status={:?} budget_met={} before_chars={} after_chars={} max_chars={}",
                outcome.status,
                outcome.budget_met(),
                outcome.before_chars,
                outcome.after_chars,
                outcome.max_chars,
            ));
            // The outcome always owns the retained messages, including missing/failed archives.
            // Restore them before returning so an unsuccessful compaction cannot empty context.
            ctx.messages = outcome.into_messages();
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
        let stage = CompressStage::new(Box::new(NoopCompressor), 1, 1);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage);
        let tp = DefaultTurnPipeline::new("test-noop", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert_eq!(ctx.messages, messages);
        assert!(ctx.tags.iter().any(|tag| {
            tag.starts_with("compress:status=Disabled budget_met=false ")
        }));
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
    async fn compress_stage_default_archives_with_explicit_sink() {
        let dir = std::env::temp_dir().join(format!("ai-stage-archive-{}", uuid::Uuid::new_v4()));
        let app = leak_app();
        let messages = vec![
            msg("user", &"old request\n".repeat(1_000)),
            msg("assistant", &"old answer\n".repeat(1_000)),
            msg("user", "current request"),
        ];
        let mut ctx = PipelineContext::new(app, messages.clone(), 0);
        let stage = CompressStage::with_default(2_048, 1)
            .with_context_options(0, Some(dir.clone()), None);
        let pipeline = crate::ai::pipeline::stage::Pipeline::new().push(stage);
        let tp = DefaultTurnPipeline::new("test-default", pipeline);
        let hooks = HookRegistry::new();
        tp.run(&mut ctx, &hooks).await.unwrap();
        assert_eq!(ctx.messages.last(), messages.last());
        assert!(!ctx.messages.contains(&messages[0]));
        assert!(!ctx.messages.contains(&messages[1]));
        assert!(ctx.tags.iter().any(|tag| {
            tag.starts_with("compress:status=Complete budget_met=true ")
        }));
        let archived = std::fs::read_to_string(dir.join("overflow-history.md")).unwrap();
        assert!(archived.contains(messages[0].content.as_str().unwrap()));
        assert!(archived.contains(messages[1].content.as_str().unwrap()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn compress_stage_missing_sink_retains_messages_and_reports_unmet_budget() {
        let app = leak_app();
        let messages = vec![
            msg("user", &"old request".repeat(1_000)),
            msg("assistant", &"old answer".repeat(1_000)),
            msg("user", "current request"),
        ];
        let mut ctx = PipelineContext::new(app, messages.clone(), 0);
        CompressStage::with_default(2_048, 1)
            .execute(&mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.messages, messages);
        assert!(ctx.tags.iter().any(|tag| {
            tag.starts_with("compress:status=MissingArchiveSink budget_met=false ")
        }));
    }

    #[tokio::test]
    async fn compress_stage_failed_archive_restores_messages_and_reports_failure() {
        let dir = std::env::temp_dir().join(format!("ai-stage-blocked-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let blocked_sink = dir.join("not-a-directory");
        std::fs::write(&blocked_sink, "archive blocker").unwrap();
        let app = leak_app();
        let messages = vec![
            msg("user", &"old request".repeat(1_000)),
            msg("assistant", &"old answer".repeat(1_000)),
            msg("user", "current request"),
        ];
        let mut ctx = PipelineContext::new(app, messages.clone(), 0);
        CompressStage::with_default(2_048, 1)
            .with_context_options(0, Some(blocked_sink.clone()), None)
            .execute(&mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.messages, messages);
        assert!(ctx.tags.iter().any(|tag| {
            tag.starts_with("compress:status=ArchiveCommitFailed budget_met=false ")
        }));
        assert_eq!(std::fs::read_to_string(blocked_sink).unwrap(), "archive blocker");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn compress_stage_forwards_options_and_skips_disabled_or_empty_context() {
        struct AssertOptions;
        impl Compressor for AssertOptions {
            fn compress(
                &self,
                messages: Vec<Message>,
                max_chars: usize,
                keep_last: usize,
                summary_max_chars: usize,
                overflow_dir: Option<PathBuf>,
                cwd: Option<&std::path::Path>,
            ) -> ContextCompressionOutcome {
                assert_eq!((max_chars, keep_last, summary_max_chars), (2_048, 1, 777));
                assert!(!messages.is_empty());
                assert_eq!(overflow_dir, Some(PathBuf::from("explicit-archive")));
                assert_eq!(cwd, Some(std::path::Path::new("explicit-cwd")));
                NoopCompressor.compress(messages, max_chars, keep_last, summary_max_chars, None, None)
            }
            fn name(&self) -> &str {
                "assert-options"
            }
        }
        for (max_chars, messages, expected) in [
            (
                2_048,
                vec![msg("user", &"x".repeat(4_096))],
                "compress:status=Disabled budget_met=false ",
            ),
            (
                0,
                vec![msg("user", "unmodified")],
                "compress:status=Disabled budget_met=true ",
            ),
            (
                2_048,
                Vec::new(),
                "compress:status=Complete budget_met=true ",
            ),
        ] {
            let mut ctx = PipelineContext::new(leak_app(), messages.clone(), 0);
            CompressStage::new(Box::new(AssertOptions), max_chars, 1)
                .with_context_options(
                    777,
                    Some(PathBuf::from("explicit-archive")),
                    Some(PathBuf::from("explicit-cwd")),
                )
                .execute(&mut ctx)
                .await
                .unwrap();
            assert_eq!(ctx.messages, messages);
            assert!(ctx.tags.iter().any(|tag| tag.starts_with(expected)));
        }
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
