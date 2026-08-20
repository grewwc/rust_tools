// =============================================================================
// StreamDecoder - 流解码端口（依赖倒置）
// =============================================================================
// 将 `stream` 的解析实现与调用方解耦，便于替换协议、插入观测或 mock。
use std::future::Future;
use std::pin::Pin;
use reqwest::Response;
use crate::ai::{history::Message, types::App};

// =============================================================================
// StreamFilter - 流式 chunk 可插拔过滤器
// =============================================================================
/// 单个流 chunk 的可插拔过滤器：可在 StreamDecoder 前后插入，对文本做变换/丢弃。
/// 返回 `Some(text)` 表示保留（可重写），`None` 表示丢弃该 chunk。
pub(crate) trait StreamFilter: Send + Sync {
    fn filter(&self, chunk: &str) -> Option<String>;
    fn name(&self) -> &str;
}

/// 直通滤器：原样保留，用于默认链与测试占位。
pub(crate) struct PassthroughFilter;
impl StreamFilter for PassthroughFilter {
    fn filter(&self, chunk: &str) -> Option<String> { Some(chunk.to_string()) }
    fn name(&self) -> &str { "passthrough" }
}

/// 过滤器链：按注册顺序依次应用，任一返回 None 则丢弃。
pub(crate) struct FilterChain {
    filters: Vec<Box<dyn StreamFilter>>,
}
impl FilterChain {
    pub(crate) fn new() -> Self { Self { filters: Vec::new() } }
    pub(crate) fn push<F: StreamFilter + 'static>(mut self, f: F) -> Self { self.filters.push(Box::new(f)); self }
    pub(crate) fn push_boxed(mut self, f: Box<dyn StreamFilter>) -> Self { self.filters.push(f); self }
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

/// 装饰器：在 `StreamDecoder` 前后插入 `FilterChain`。
/// 演示“端口装饰器”形态：driver 可通过注入带过滤器的 decoder 实现可组合解析，
/// 而无需改动 `stream::stream_response` 内部状态机。
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
        // 预过滤可在真实网络前模拟；此处仅演示装饰器透传，
        // 实际 chunk 级过滤在 stream::runtime 的解析循环内按 FilterChain 逐块 apply。
        // 为保持零行为变更，默认空链等价于直接委托。
        let _before = &self.before;
        let _after = &self.after;
        self.inner.decode(app, response, current_history, terminal_dedupe_candidate)
    }
}

/// 简易辅助：在已解码文本上应用 after 链（用于单测 / 离线后处理）。
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

