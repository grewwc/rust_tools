use std::sync::atomic::{AtomicBool, Ordering};
use std::{path::PathBuf, sync::Mutex};

use dirs::cache_dir;

pub trait EmbeddingProvider: Sync + Send {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>>;
    fn is_ready(&self) -> bool;
    fn try_load(&self);
}

enum FastEmbedState {
    NotLoaded,
    Ready(fastembed::TextEmbedding),
    Failed,
}

pub struct FastEmbedProvider {
    inner: Mutex<FastEmbedState>,
    cache_dir: PathBuf,
    ready_flag: AtomicBool,
}

impl FastEmbedProvider {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            inner: Mutex::new(FastEmbedState::NotLoaded),
            cache_dir,
            ready_flag: AtomicBool::new(false),
        }
    }
}

impl EmbeddingProvider for FastEmbedProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        if !self.ready_flag.load(Ordering::Acquire) {
            return None;
        }
        let guard = self.inner.lock().ok()?;
        match &*guard {
            FastEmbedState::Ready(e) => e.embed(vec![text], None).ok().and_then(|v| v.into_iter().next()),
            _ => None,
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Some(Vec::new());
        }
        if !self.ready_flag.load(Ordering::Acquire) {
            return None;
        }
        let guard = self.inner.lock().ok()?;
        match &*guard {
            FastEmbedState::Ready(e) => e.embed(texts.to_vec(), None).ok(),
            _ => None,
        }
    }

    fn is_ready(&self) -> bool {
        self.ready_flag.load(Ordering::Acquire)
    }

    fn try_load(&self) {
        {
            let guard = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if !matches!(*guard, FastEmbedState::NotLoaded) {
                return;
            }
        }

        let result = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(fastembed::EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(self.cache_dir.clone())
                .with_show_download_progress(false),
        );

        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        match result {
            Ok(embedder) => {
                *guard = FastEmbedState::Ready(embedder);
                self.ready_flag.store(true, Ordering::Release);
            }
            Err(e) => {
                eprintln!("[embedding] model load failed: {e}, falling back to TF-IDF");
                *guard = FastEmbedState::Failed;
            }
        }
    }
}

static GLOBAL_PROVIDER: std::sync::OnceLock<Box<dyn EmbeddingProvider>> =
    std::sync::OnceLock::new();

fn default_cache_dir() -> PathBuf {
    cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rust_tools")
        .join("fastembed_cache")
}

fn default_provider() -> Box<dyn EmbeddingProvider> {
    Box::new(FastEmbedProvider::new(default_cache_dir()))
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

pub fn warm_up() {
    std::thread::spawn(|| {
        global_provider().try_load();
    });
}
