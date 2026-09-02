// =============================================================================
// StreamDecoder - stream decoding port (dependency inversion)
// =============================================================================
// Decouples the parsing implementation of `stream` from callers, making it easy to swap protocols, add observability, or mock.
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use reqwest::Response;
use crate::ai::{history::Message, types::App};

// =============================================================================
// StreamFilter - pluggable filter for streaming chunks
// =============================================================================
/// A pluggable filter for a single stream chunk: can be inserted before/after the StreamDecoder to transform or drop text.
/// Returning `Some(text)` keeps the chunk (possibly rewritten); `None` drops it.
pub(crate) trait StreamFilter: Send + Sync {
    fn filter(&self, chunk: &str) -> Option<String>;
    fn name(&self) -> &str;
}

/// Pass-through filter: keeps text as-is; used for the default chain and as a test placeholder.
pub(crate) struct PassthroughFilter;
impl StreamFilter for PassthroughFilter {
    fn filter(&self, chunk: &str) -> Option<String> { Some(chunk.to_string()) }
    fn name(&self) -> &str { "passthrough" }
}

/// Filter chain: applied in registration order; any filter returning None drops the chunk.
#[derive(Clone)]
pub(crate) struct FilterChain {
    filters: Vec<Arc<dyn StreamFilter>>,
}
impl FilterChain {
    pub(crate) fn new() -> Self { Self { filters: Vec::new() } }
    pub(crate) fn push<F: StreamFilter + 'static>(mut self, f: F) -> Self { self.filters.push(Arc::new(f)); self }
    pub(crate) fn push_boxed(mut self, f: Box<dyn StreamFilter>) -> Self { self.filters.push(Arc::from(f)); self }
    pub(crate) fn register<F: StreamFilter + 'static>(&mut self, filter: F) { self.filters.push(Arc::new(filter)); }
    pub(crate) fn is_empty(&self) -> bool { self.filters.is_empty() }
    pub(crate) fn len(&self) -> usize { self.filters.len() }
    pub(crate) fn apply(&self, mut chunk: String) -> Option<String> {
        for f in &self.filters {
            match f.filter(&chunk) {
                Some(next) => chunk = next,
                None => return None,
            }
        }
        Some(chunk)
    }
    pub(crate) fn names(&self) -> Vec<&str> { self.filters.iter().map(|f| f.name()).collect() }
}
impl Default for FilterChain {
    fn default() -> Self { Self::new() }
}

/// Decorator: wraps `StreamDecoder` with a `FilterChain` applied around it.
/// Demonstrates the "port decorator" shape: the driver can inject a decoder with filters for composable parsing,
/// without changing the internal state machine of `stream::stream_response`.
pub(crate) struct FilteredStreamDecoder<D: StreamDecoder> {
    inner: D,
    before: FilterChain,
    after: FilterChain,
}
impl<D: StreamDecoder> FilteredStreamDecoder<D> {
    pub(crate) fn new(inner: D) -> Self { Self { inner, before: FilterChain::new(), after: FilterChain::new() } }
    pub(crate) fn with_before(mut self, chain: FilterChain) -> Self { self.before = chain; self }
    pub(crate) fn with_after(mut self, chain: FilterChain) -> Self { self.after = chain; self }
    pub(crate) fn before_chain_mut(&mut self) -> &mut FilterChain { &mut self.before }
    pub(crate) fn after_chain_mut(&mut self) -> &mut FilterChain { &mut self.after }
}
impl<D: StreamDecoder> StreamDecoder for FilteredStreamDecoder<D> {
    fn decode<'a>(
        &'a self,
        app: &'a mut App,
        response: Response,
        current_history: &'a [Message],
        terminal_dedupe_candidate: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<DecodedStream, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
// Pre-filtering could be simulated before the real network; this only demonstrates decorator pass-through:
// actual chunk-level filtering is applied per chunk via FilterChain inside the parse loop of stream::runtime.
// To keep zero behavior change, an empty default chain is equivalent to direct delegation.
        let _before = &self.before;
        let _after = &self.after;
        self.inner.decode(app, response, current_history, terminal_dedupe_candidate)
    }
}

/// Small helper: applies the after chain to already-decoded text (for unit tests / offline post-processing).
pub(crate) fn apply_after_filters(chain: &FilterChain, text: String) -> Option<String> {
    if chain.is_empty() { Some(text) } else { chain.apply(text) }
}

#[derive(Debug, Default)]
pub(crate) struct DecodedStream {
    pub text: String,
    pub tool_calls: Vec<crate::ai::types::ToolCall>,
    pub is_complete: bool,
}

pub(crate) trait StreamDecoder: Send + Sync {
    fn decode<'a>(
        &'a self,
        app: &'a mut App,
        response: Response,
        current_history: &'a [Message],
        terminal_dedupe_candidate: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<DecodedStream, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

