/// Embedding provider trait and implementations.
/// Single source for text embedding — replaces scattered EMBEDDING_PROVIDER globals.
use std::sync::{Mutex, OnceLock};

/// Trait for embedding providers.
pub trait EmbeddingProvider: Sync + Send {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>>;
}

/// No-op provider (used when no model is loaded).
struct NoopEmbeddingProvider;
impl EmbeddingProvider for NoopEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
    fn embed_batch(&self, _texts: &[String]) -> Option<Vec<Vec<f32>>> {
        None
    }
}

/// FastEmbed-based provider (lazy-loaded).
pub struct FastEmbedProvider {
    inner: Mutex<Option<fastembed::TextEmbedding>>,
    cache_dir: std::path::PathBuf,
}

impl FastEmbedProvider {
    pub fn new(cache_dir: std::path::PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            cache_dir,
        }
    }

    fn get_embedder(&self) -> Result<&fastembed::TextEmbedding, String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        if guard.is_none() {
            let embedder = fastembed::TextEmbedding::try_new(
                fastembed::InitOptions::default()
                    .with_cache_dir(self.cache_dir.clone())
                    .with_show_download_progress(true),
            )
            .map_err(|e| format!("Failed to load embedding model: {}", e))?;
            *guard = Some(embedder);
        }
        // Safety: we just ensured Some exists
        Ok(unsafe {
            std::mem::transmute::<&fastembed::TextEmbedding, &fastembed::TextEmbedding>(
                guard.as_ref().unwrap(),
            )
        })
    }
}

impl EmbeddingProvider for FastEmbedProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.get_embedder().ok()?;
        let embeddings = embedder.embed(vec![text], None).ok()?;
        embeddings.into_iter().next()
    }

    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Some(Vec::new());
        }
        let embedder = self.get_embedder().ok()?;
        embedder.embed(texts.to_vec(), None).ok()
    }
}

/// Global embedding provider — set once at startup.
static GLOBAL_PROVIDER: OnceLock<Box<dyn EmbeddingProvider>> = OnceLock::new();

/// Set the global embedding provider. Must be called before first use.
pub fn set_provider(provider: Box<dyn EmbeddingProvider>) {
    let _ = GLOBAL_PROVIDER.set(provider);
}

/// Embed text using the global provider.
pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    if let Some(p) = GLOBAL_PROVIDER.get() {
        p.embed(text)
    } else {
        None
    }
}

/// Embed multiple texts using the global provider.
pub fn embed_texts(texts: &[String]) -> Option<Vec<Vec<f32>>> {
    if let Some(p) = GLOBAL_PROVIDER.get() {
        p.embed_batch(texts)
    } else {
        None
    }
}

/// Check if an embedding provider is available.
pub fn has_provider() -> bool {
    GLOBAL_PROVIDER.get().is_some()
}
