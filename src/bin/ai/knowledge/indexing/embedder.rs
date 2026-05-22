//! 嵌入向量提供方（fastembed 已移除）
//!
//! 历史上这里挂的是 `fastembed::TextEmbedding` (MultilingualE5Small)，需要静态
//! 链接 ONNX Runtime + XNNPACK，占整个 binary ~25MB / 23.6%——属于体积膨胀
//! 的最大单一原因。
//!
//! 现在统一降级为"嵌入不可用"：
//! - `embed_text` / `embed_texts` 返回 `None`
//! - 所有调用方（`keyword_search.rs`、`memory_store.rs::search` 的 vector 重排、
//!   `memory.rs::execute_memory_dedup` 的 cosine 去重等）都已经处理过 `None`
//!   场景，会自动回退到 BM25 + lexical similarity（dice / jaccard / char_overlap）。
//! - 语义重排不再生效，但召回仍然是可用的。
//!
//! 如果将来想再启用嵌入，建议挂一个外部 embedding HTTP 服务（OpenAI / 自部署
//! BGE 等）通过 `set_provider` 注入；这里保留 trait + GLOBAL_PROVIDER 的形状
//! 就是为了未来切换不动调用方。
//!
//! 意图识别走 `intent_model.rs` 的本地 TF-IDF，复杂场景再调 LLM——
//! 见 `intent_model::detect_intent_async`。

use std::path::PathBuf;
use std::sync::OnceLock;

use dirs::cache_dir;

pub trait EmbeddingProvider: Sync + Send {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>>;
    fn is_ready(&self) -> bool;
    fn try_load(&self);
}

/// 默认 provider：永远返回 None，让调用方自动走 BM25 / lexical 降级路径。
pub struct NullEmbeddingProvider;

impl EmbeddingProvider for NullEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
    fn embed_batch(&self, _texts: &[String]) -> Option<Vec<Vec<f32>>> {
        None
    }
    fn is_ready(&self) -> bool {
        false
    }
    fn try_load(&self) {
        // no-op
    }
}

static GLOBAL_PROVIDER: OnceLock<Box<dyn EmbeddingProvider>> = OnceLock::new();

/// 旧 API 兼容：返回默认 cache 路径。fastembed 已移除，但保留路径以便外部
/// embedding provider（如挂 HTTP 服务时本地落盘缓存）继续复用。
#[allow(dead_code)]
fn default_cache_dir() -> PathBuf {
    cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rust_tools")
        .join("embedding_cache")
}

fn default_provider() -> Box<dyn EmbeddingProvider> {
    Box::new(NullEmbeddingProvider)
}

fn global_provider() -> &'static dyn EmbeddingProvider {
    GLOBAL_PROVIDER.get_or_init(default_provider).as_ref()
}

pub fn set_provider(provider: Box<dyn EmbeddingProvider>) {
    let _ = GLOBAL_PROVIDER.set(provider);
}

pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    global_provider().embed(text)
}

pub fn embed_texts(texts: &[String]) -> Option<Vec<Vec<f32>>> {
    global_provider().embed_batch(texts)
}

pub fn is_ready() -> bool {
    global_provider().is_ready()
}

/// no-op：保留旧调用点（driver/mod.rs 启动 warm_up）兼容性，无副作用。
pub fn warm_up() {
    // fastembed 已移除，没有需要预热的本地模型。
}
