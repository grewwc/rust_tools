use crate::ai::knowledge::config::KnowledgeConfig;
use crate::ai::knowledge::retrieval::recall;
use crate::ai::knowledge::storage::jsonl_store::JsonlStore;
use crate::ai::tools::storage::memory_store::MemoryStore;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

const RECALL_BUNDLE_CACHE_TTL: Duration = Duration::from_secs(15);
const RECALL_BUNDLE_CACHE_LIMIT: usize = 12;

static RECALL_BUNDLE_CACHE: LazyLock<Mutex<Vec<RecallBundleCacheEntry>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone)]
pub(crate) struct AutoRecalledKnowledge {
    pub(crate) content: String,
    pub(crate) high_confidence_project_memory: bool,
    pub(crate) entry_count: usize,
    pub(crate) project_hint: Option<String>,
    pub(crate) categories: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct RecallBundle {
    pub(crate) guidelines: Option<String>,
    pub(crate) recalled: Option<AutoRecalledKnowledge>,
}

#[derive(Clone, PartialEq, Eq)]
struct RecallBundleCacheKey {
    question: String,
    guideline_max_chars: usize,
    recall_max_chars: usize,
    project_hint: Option<String>,
    file_len: Option<u64>,
    modified_unix_ms: Option<u128>,
}

struct RecallBundleCacheEntry {
    created_at: Instant,
    key: RecallBundleCacheKey,
    value: RecallBundle,
}

fn current_project_hint() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let name = cwd.file_name()?.to_str()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

pub(crate) fn build_persistent_guidelines(question: &str, max_chars: usize) -> Option<String> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_persistent_guidelines_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(crate) fn build_auto_recalled_knowledge(
    question: &str,
    max_chars: usize,
) -> Option<AutoRecalledKnowledge> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_auto_recalled_knowledge_from_parts(&jsonl_store, question, max_chars, &config)
}

pub(crate) fn build_auto_recalled_knowledge_with_project(
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
) -> Option<AutoRecalledKnowledge> {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let config = KnowledgeConfig::from_config_file();
    build_auto_recalled_knowledge_with_project_from_parts(
        &jsonl_store,
        question,
        max_chars,
        project_hint,
        &config,
    )
}

pub(crate) fn build_recall_bundle(
    question: &str,
    guideline_max_chars: usize,
    recall_max_chars: usize,
) -> RecallBundle {
    let store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(store.path().to_path_buf());
    let cache_key = recall_bundle_cache_key(
        question,
        guideline_max_chars,
        recall_max_chars,
        jsonl_store.path(),
    );
    if let Some(cached) = try_get_cached_recall_bundle(&cache_key) {
        return cached;
    }

    let config = KnowledgeConfig::from_config_file();
    RecallBundle {
        guidelines: build_persistent_guidelines_from_parts(
            &jsonl_store,
            question,
            guideline_max_chars,
            &config,
        ),
        recalled: build_auto_recalled_knowledge_from_parts(
            &jsonl_store,
            question,
            recall_max_chars,
            &config,
        ),
    }
    .tap(|bundle| store_cached_recall_bundle(cache_key, bundle.clone()))
}

fn build_persistent_guidelines_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<String> {
    recall::build_persistent_guidelines(jsonl_store, question, max_chars, config)
}

fn build_auto_recalled_knowledge_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge(jsonl_store, question, max_chars, config)
        .map(map_auto_recalled_knowledge)
}

fn build_auto_recalled_knowledge_with_project_from_parts(
    jsonl_store: &JsonlStore,
    question: &str,
    max_chars: usize,
    project_hint: Option<&str>,
    config: &KnowledgeConfig,
) -> Option<AutoRecalledKnowledge> {
    recall::build_auto_recalled_knowledge_with_project(
        jsonl_store,
        question,
        max_chars,
        project_hint,
        config,
    )
    .map(map_auto_recalled_knowledge)
}

fn map_auto_recalled_knowledge(
    r: crate::ai::knowledge::retrieval::recall::AutoRecalledKnowledge,
) -> AutoRecalledKnowledge {
    AutoRecalledKnowledge {
        content: r.content,
        high_confidence_project_memory: r.high_confidence_project_memory,
        entry_count: r.entry_count,
        project_hint: r.project_hint,
        categories: r.categories,
    }
}

pub(super) fn current_project_name_hint() -> Option<String> {
    current_project_hint()
}

fn recall_bundle_cache_key(
    question: &str,
    guideline_max_chars: usize,
    recall_max_chars: usize,
    path: &Path,
) -> RecallBundleCacheKey {
    let metadata = std::fs::metadata(path).ok();
    RecallBundleCacheKey {
        question: question.trim().to_string(),
        guideline_max_chars,
        recall_max_chars,
        project_hint: current_project_hint(),
        file_len: metadata.as_ref().map(|m| m.len()),
        modified_unix_ms: metadata
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(system_time_millis),
    }
}

fn system_time_millis(value: SystemTime) -> Option<u128> {
    value
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn try_get_cached_recall_bundle(key: &RecallBundleCacheKey) -> Option<RecallBundle> {
    let Ok(mut cache) = RECALL_BUNDLE_CACHE.lock() else {
        return None;
    };
    cache.retain(|entry| entry.created_at.elapsed() < RECALL_BUNDLE_CACHE_TTL);
    cache.iter().find(|entry| &entry.key == key).map(|entry| entry.value.clone())
}

fn store_cached_recall_bundle(key: RecallBundleCacheKey, value: RecallBundle) {
    let Ok(mut cache) = RECALL_BUNDLE_CACHE.lock() else {
        return;
    };
    cache.retain(|entry| entry.created_at.elapsed() < RECALL_BUNDLE_CACHE_TTL && entry.key != key);
    cache.insert(
        0,
        RecallBundleCacheEntry {
            created_at: Instant::now(),
            key,
            value,
        },
    );
    if cache.len() > RECALL_BUNDLE_CACHE_LIMIT {
        cache.truncate(RECALL_BUNDLE_CACHE_LIMIT);
    }
}

trait Tap: Sized {
    fn tap<F: FnOnce(&Self)>(self, f: F) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}
