use rustc_hash::FxHashMap;
/// Agent memory store — the underlying persistent storage system
///
/// ## Architecture
/// `MemoryStore` is the low-level storage for all knowledge/memory, shared by two upper layers:
///
/// 1. **Knowledge** - user-facing knowledge management
///    - Exposed to users via `knowledge_tools.rs`
///    - Stores factual knowledge such as project facts, decision records, and user preferences
///    - Categories: `user_memory`, `project_info`, `architecture`, `decision_log`
///
/// 2. **Memory** - explicitly saved long-term rules and session-scoped internal records
///    - Managed by the `memory.rs` service layer
///    - Stores behavioral guidance such as safety rules, coding guidelines, and self-reflections
///    - Categories: `safety_rules`, `coding_guideline`, `self_note`, `common_sense`
///
/// ## Category distinction
/// - **Guidance categories**:
///   `safety_rules`, `user_preference`, `preference`, `coding_guideline`,
///   `best_practice`, `common_sense`, `self_note`
///
/// - **Knowledge categories**:
///   `user_memory`, `project_info`, `architecture`, `decision_log`
///   and other non-guidance categories
///
/// ## Search mechanism
/// - BM25 keyword search + text similarity (lexical level)
/// - Archive file search support (configurable)
/// - Automatic dedup and GC
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use super::memory_index::MemoryIndex;
use super::with_memory_file_lock;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

/// Global (source_path, MemoryIndex) registry: lazily loaded and reused per path.
/// The first access to a source path opens / rebuilds the SQLite index; all later calls receive the
/// same `Arc<MemoryIndex>`, sharing LFU counts and the FTS index across calls.
fn memory_index_for(source_path: &Path) -> Option<Arc<MemoryIndex>> {
    use std::sync::Mutex;
    static REG: OnceLock<Mutex<Vec<(PathBuf, Arc<MemoryIndex>)>>> = OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = reg.lock().ok()?;
    if let Some((_, idx)) = guard.iter().find(|(p, _)| p == source_path) {
        return Some(idx.clone());
    }
    let db_path = derive_db_path(source_path)?;
    match MemoryIndex::open_or_init(db_path.clone(), source_path.to_path_buf()) {
        Ok(idx) => {
            let arc = Arc::new(idx);
            guard.push((source_path.to_path_buf(), arc.clone()));
            Some(arc)
        }
        Err(e) => {
            // Fall back to the BM25 path when the index is unavailable, without blocking the main store.
            trace_memory_event(
                "memory.index.open_failed",
                "MemoryIndex unavailable; falling back to BM25",
                &[
                    ("source", source_path.display().to_string()),
                    ("db", db_path.display().to_string()),
                    ("error", e),
                ],
            );
            None
        }
    }
}

pub(crate) fn rebuild_index_for_path(path: &Path) {
    if let Some(idx) = memory_index_for(path)
        && let Err(err) = idx.rebuild_from_source()
    {
        trace_memory_event(
            "memory.index.rebuild_failed",
            "MemoryIndex rebuild failed after explicit rewrite; index may drift",
            &[("path", path.display().to_string()), ("error", err)],
        );
    }
}

/// Derive the corresponding sqlite path from a source jsonl path:
/// `agent_memory.jsonl` -> `agent_memory.db`
/// `agent_memory.subagent-xxx.jsonl` -> `agent_memory.subagent-xxx.db`
fn derive_db_path(source: &Path) -> Option<PathBuf> {
    let stem = source.file_stem()?.to_str()?;
    let parent = source.parent()?;
    Some(parent.join(format!("{stem}.db")))
}

/// Mirror key operational events of the memory subsystem into the AIOS kernel trace ring,
/// making data-mutating actions (rotate / enforce / GC) observable on the AIOS side.
/// Silently return when the kernel is unavailable or locking fails — this must never affect the main flow.
pub(crate) fn trace_memory_event(location: &'static str, msg: &str, fields: &[(&str, String)]) {
    use aios_kernel::{FastMap, primitives::TraceLevel};

    let g = match crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let kernel = match g.as_ref() {
        Some(k) => k.clone(),
        None => return,
    };
    drop(g);

    let mut map: FastMap<String, String> = FastMap::default();
    for (k, v) in fields {
        map.insert((*k).to_string(), v.clone());
    }
    if let Ok(mut guard) = kernel.lock() {
        guard.trace_event(
            location.to_string(),
            TraceLevel::Info,
            None,
            map,
            Some(msg.to_string()),
        );
    }
}

/// Atomically write content to `path`: write to a tmp file in the same directory, fsync, then rename.
/// If the process crashes midway, only the pre-rename old file remains; no half-written JSONL is ever produced.
fn atomic_write_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("memory");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{}.tmp.{}.{}", file_name, pid, nanos));
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(contents)?;
        f.flush()?;
        let _ = f.sync_all();
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        // Best-effort cleanup of the tmp file if rename fails, to avoid leaving a partial artifact
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentMemoryEntry {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) timestamp: String,
    pub(crate) category: String,
    pub(crate) note: String,
    pub(crate) tags: Vec<String>,
    pub(crate) source: Option<String>,
    /// Priority level: 0-255. Higher = more important. 255 = permanent (never delete).
    /// Default: 100 (normal priority). Low: 0-49, Normal: 50-99, High: 100-200, Permanent: 255
    #[serde(default = "default_priority")]
    pub(crate) priority: Option<u8>,
    #[serde(default)]
    pub(crate) owner_pid: Option<u64>,
    #[serde(default)]
    pub(crate) owner_pgid: Option<u64>,

    /// Optional image path (for memo entries that include screenshots/images).
    /// When set, OCR text is extracted and stored in `note` for search indexing.
    #[serde(default)]
    pub(crate) image_path: Option<String>,
    /// Verified provenance and historical revisions are not searchable note text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) distilled: Option<crate::ai::knowledge::distilled::DistilledMetadata>,
}

fn default_priority() -> Option<u8> {
    Some(100)
}

impl Default for AgentMemoryEntry {
    fn default() -> Self {
        Self {
            id: None,
            timestamp: String::new(),
            category: String::new(),
            note: String::new(),
            tags: Vec::new(),
            source: None,
            distilled: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }
}

pub(crate) struct MemoryStore {
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryBatchUpdateReport {
    pub(crate) deleted: usize,
    pub(crate) appended: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DistilledUpsertReport {
    pub(crate) inserted: bool,
    pub(crate) updated: bool,
    pub(crate) duplicate: bool,
    pub(crate) entry_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum KnowledgeAppendOutcome {
    Appended,
    Duplicate { existing_id: Option<String> },
}

/// In-process cache for knowledge_save idempotent dedup: invalidated by file fingerprint (len, mtime).
/// When the file is rewritten by another process / rotation / GC, the fingerprint changes -> the cache clears and falls back to a full-file scan,
/// so correctness is identical to the no-cache case; repeated saves just drop from O(full file) to O(1).
struct KnowledgeDedupCache {
    fingerprint: (u64, SystemTime),
    /// Normalized equivalence key of (category, note, source, tags) -> existing entry id.
    seen: FxHashMap<(String, String, String, Vec<String>), Option<String>>,
}

impl KnowledgeDedupCache {
    fn empty() -> Self {
        Self {
            fingerprint: (0, SystemTime::UNIX_EPOCH),
            seen: FxHashMap::default(),
        }
    }
}

static KNOWLEDGE_DEDUP_CACHE: LazyLock<Mutex<FxHashMap<PathBuf, KnowledgeDedupCache>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

/// Remove the dedup cache entry when a private memory file ends its lifecycle, so deleted sub-agent
/// paths do not keep accumulating in a long-lived parent process. Removing the cache does not touch persisted data; if the path is
/// reused later, the cache is rebuilt from the current file fingerprint.
pub(crate) fn remove_knowledge_dedup_cache_entry(path: &Path) {
    if let Ok(mut cache) = KNOWLEDGE_DEDUP_CACHE.lock() {
        cache.remove(path);
    }
}

fn memory_file_fingerprint(path: &Path) -> (u64, SystemTime) {
    std::fs::metadata(path)
        .map(|m| (m.len(), m.modified().unwrap_or(SystemTime::UNIX_EPOCH)))
        .unwrap_or((0, SystemTime::UNIX_EPOCH))
}

fn equivalent_knowledge_key(entry: &AgentMemoryEntry) -> (String, String, String, Vec<String>) {
    (
        entry.category.trim().to_lowercase(),
        normalize_learning_note(&entry.note),
        entry.source.as_deref().unwrap_or("").trim().to_lowercase(),
        normalized_knowledge_tags(&entry.tags),
    )
}

fn should_dedup_learning_entry(entry: &AgentMemoryEntry) -> bool {
    matches!(
        entry.category.as_str(),
        "self_note"
            | "project_memory"
            | "coding_guideline"
            | "common_sense"
            | "best_practice"
            | "safety_rules"
    )
}

fn cap_memory_entry(entry: &AgentMemoryEntry) -> AgentMemoryEntry {
    const MAX_NOTE_BYTES: usize = 4_096;
    if entry.note.len() <= MAX_NOTE_BYTES {
        return entry.clone();
    }

    let mut capped = entry.clone();
    let mut truncated = String::with_capacity(MAX_NOTE_BYTES + 64);
    let mut used = 0usize;
    for ch in capped.note.chars() {
        let extra = ch.len_utf8();
        if used + extra > MAX_NOTE_BYTES {
            break;
        }
        truncated.push(ch);
        used += extra;
    }
    truncated.push_str("\n…[note truncated to fit memory store cap]");
    capped.note = truncated;
    capped
}

fn equivalent_knowledge_entry(left: &AgentMemoryEntry, right: &AgentMemoryEntry) -> bool {
    normalize_knowledge_field(&left.category) == normalize_knowledge_field(&right.category)
        && normalize_learning_note(&left.note) == normalize_learning_note(&right.note)
        && normalize_knowledge_field(left.source.as_deref().unwrap_or(""))
            == normalize_knowledge_field(right.source.as_deref().unwrap_or(""))
        && normalized_knowledge_tags(&left.tags) == normalized_knowledge_tags(&right.tags)
}

fn normalize_knowledge_field(value: &str) -> String {
    value.trim().to_lowercase()
}

fn normalized_knowledge_tags(tags: &[String]) -> Vec<String> {
    let mut normalized = tags
        .iter()
        .map(|tag| normalize_knowledge_field(tag))
        .filter(|tag| !tag.is_empty())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn normalize_learning_note(note: &str) -> String {
    note.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
#[cfg(test)]
impl MemoryStore {
    pub(crate) fn for_tests_with_path(path: PathBuf) -> Self {
        Self { path }
    }
}
/// Build a store directly from an explicit path, bypassing task_local override / env / config.
/// Used by the parent task to write whitelist entries back into the main memory file after sub-agent finalize.
pub(crate) fn store_for_path(path: PathBuf) -> MemoryStore {
    MemoryStore { path }
}
fn resolve_memory_file() -> PathBuf {
    if let Some(path) = crate::ai::driver::runtime_ctx::override_memory_path() {
        return path;
    }
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(crate::commonw::utils::expanduser(path).as_ref());
        }
    }
    let cfg = crate::commonw::configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    PathBuf::from(crate::commonw::utils::expanduser(&raw).as_ref())
}
mod distilled;
mod evict;
mod importance;
mod prune;
mod query;

pub(crate) use importance::MemoryImportance;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod importance_tests;
#[cfg(test)]
mod retention_tests;
