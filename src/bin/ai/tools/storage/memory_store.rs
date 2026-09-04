use rustc_hash::{FxHashMap, FxHashSet};
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
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use super::memory_index::MemoryIndex;
use super::with_memory_file_lock;
use crate::ai::knowledge::indexing::similarity;
use crate::ai::tools::service::memory::{execute_memory_dedup, execute_memory_gc};
use crate::commonw::configw;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::ffi::OsStr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

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

impl MemoryStore {
    pub(crate) fn from_env_or_config() -> Self {
        Self {
            path: resolve_memory_file(),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        let entry = cap_memory_entry(entry);
        super::with_memory_file_lock(&self.path, || {
            if should_dedup_learning_entry(&entry)
                && self.has_recent_duplicate(&entry, 200).unwrap_or(false)
            {
                return Ok(());
            }
            self.append_entry_while_locked(&entry)
        })
    }

    /// Perform an atomic idempotent write of user-visible long-term knowledge. `knowledge_save` retries and duplicate
    /// tool calls must not create duplicate JSONL records or repeated RAG upserts; other
    /// MemoryStore callers keep their original semantics.
    pub(crate) fn append_idempotent_knowledge(
        &self,
        entry: &AgentMemoryEntry,
    ) -> Result<KnowledgeAppendOutcome, String> {
        let entry = cap_memory_entry(entry);
        self.ensure_memory_file_for_lock()?;
        let key = equivalent_knowledge_key(&entry);
        // Fingerprint validation and cache-hit checks run under the file lock, ensuring the same view of the
        // file state as scanning/appending/consolidation, avoiding the race where an external write changed the file but the cache wrongly returned Duplicate.
        super::with_memory_file_lock(&self.path, || {
            let fingerprint = memory_file_fingerprint(&self.path);
            let cache_hit = {
                let mut cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
                let path_cache = cache.entry(self.path.clone()).or_insert_with(|| {
                    let mut c = KnowledgeDedupCache::empty();
                    c.fingerprint = fingerprint;
                    c
                });
                if path_cache.fingerprint != fingerprint {
                    path_cache.fingerprint = fingerprint;
                    path_cache.seen.clear();
                }
                path_cache.seen.get(&key).cloned()
            };
            if let Some(existing_id) = cache_hit {
                return Ok(KnowledgeAppendOutcome::Duplicate { existing_id });
            }
            // Cache miss: scan the file under the file lock
            if let Some(existing) = self.find_equivalent_knowledge(&entry)? {
                let mut cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
                cache
                    .entry(self.path.clone())
                    .or_insert_with(KnowledgeDedupCache::empty)
                    .seen
                    .insert(key.clone(), existing.id.clone());
                return Ok(KnowledgeAppendOutcome::Duplicate {
                    existing_id: existing.id,
                });
            }
            self.append_entry_while_locked(&entry)?;
            let new_fp = memory_file_fingerprint(&self.path);
            let mut cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
            let path_cache = cache
                .entry(self.path.clone())
                .or_insert_with(KnowledgeDedupCache::empty);
            path_cache.seen.insert(key.clone(), entry.id.clone());
            path_cache.fingerprint = new_fp;
            Ok(KnowledgeAppendOutcome::Appended)
        })
    }

    fn ensure_memory_file_for_lock(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
        }
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to initialize memory file: {e}"))?;
        Ok(())
    }

    fn append_entry_while_locked(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
        }
        let serialized = serde_json::to_string(entry)
            .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open memory file: {e}"))?;

        let needs_newline = file
            .metadata()
            .map_err(|e| format!("Failed to read memory file metadata: {e}"))?
            .len()
            > 0
            && {
                file.seek(SeekFrom::End(-1))
                    .map_err(|e| format!("Failed to seek memory file: {e}"))?;
                let mut last = [0u8; 1];
                file.read_exact(&mut last)
                    .map_err(|e| format!("Failed to read memory file: {e}"))?;
                last[0] != b'\n'
            };

        if needs_newline {
            file.write_all(b"\n")
                .map_err(|e| format!("Failed to write memory file: {e}"))?;
        }
        file.write_all(serialized.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| format!("Failed to write memory file: {e}"))?;

        // JSONL is the source of truth; the SQLite index sync below is best-effort,
        // failures are traced but not propagated, so rusqlite problems never block the main store.
        if let Some(idx) = memory_index_for(&self.path) {
            if let Err(e) = idx.upsert_entry(entry) {
                trace_memory_event(
                    "memory.index.upsert_failed",
                    "MemoryIndex upsert failed; index may drift",
                    &[
                        ("path", self.path.display().to_string()),
                        ("entry_id", entry.id.clone().unwrap_or_default()),
                        ("error", e),
                    ],
                );
            } else {
                let _ = idx.refresh_signature();
            }
        }
        Ok(())
    }

    fn find_equivalent_knowledge(
        &self,
        target: &AgentMemoryEntry,
    ) -> Result<Option<AgentMemoryEntry>, String> {
        let file = match fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("Failed to read memory file: {error}")),
        };
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.map_err(|error| format!("Failed to read memory file: {error}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(existing) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                continue;
            };
            if equivalent_knowledge_entry(&existing, target) {
                return Ok(Some(existing));
            }
        }
        Ok(None)
    }

    fn has_recent_duplicate(
        &self,
        target: &AgentMemoryEntry,
        recent_limit: usize,
    ) -> Result<bool, String> {
        let target_norm = normalize_learning_note(&target.note);
        let target_source = target.source.as_deref().unwrap_or("");
        let recent = self.recent_tail_window(recent_limit)?;
        Ok(recent.into_iter().any(|entry| {
            if entry.category != target.category {
                return false;
            }
            if entry.source.as_deref().unwrap_or("") != target_source {
                return false;
            }
            normalize_learning_note(&entry.note) == target_norm
        }))
    }

    /// Same entry window as `recent(limit)` for the append duplicate check,
    /// obtained by scanning the file backwards from the end and parsing at
    /// most `limit` entries instead of reading and parsing the whole file.
    ///
    /// Equivalence to `recent(limit)`:
    /// - Line model: `\n`-separated byte slices, with the final unterminated
    ///   segment counting as a line, matching `BufRead::lines`. Splitting on
    ///   the `\n` byte is UTF-8 safe because continuation bytes are never
    ///   0x0A, so every slice contains whole characters.
    /// - Window identity: both implementations keep the last `limit`
    ///   successfully-parsed entries. The predicate in
    ///   `has_recent_duplicate` depends only on individual entries, so the
    ///   newest-first ordering of `recent` is not needed here.
    /// - Skipped lines: whitespace-only lines and lines that fail JSON
    ///   parsing are ignored in both implementations and do not count
    ///   toward `limit`.
    /// - Errors: `recent` fails the whole scan when any line in the file is
    ///   not valid UTF-8, and `append` maps that error to "no duplicate".
    ///   This scanner fails identically for invalid lines inside the scanned
    ///   tail, and once `limit` entries are collected it validates the
    ///   remaining prefix as raw UTF-8 (`ensure_range_is_utf8`, a cheap
    ///   byte-level check without JSON parsing), so corruption in the
    ///   unscanned prefix produces the same error.
    fn recent_tail_window(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        if limit == 0 || !self.path.exists() {
            return Ok(Vec::new());
        }
        let mut file =
            fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("Failed to read memory file: {e}"))?
            .len();

        const CHUNK: usize = 32 * 1024;
        let mut chunk: Vec<u8> = Vec::new();
        // Exclusive end of the not-yet-scanned region; bytes >= end have
        // already been scanned as complete lines.
        let mut end = file_len;
        // Byte offset where scanning stopped; the prefix below it is only
        // validated by `ensure_range_is_utf8` after the loop.
        let mut scan_stop = 0u64;
        let mut window = CHUNK as u64;
        let mut collected: Vec<AgentMemoryEntry> = Vec::new();

        while end > 0 && collected.len() < limit {
            let start = end.saturating_sub(window);
            let len = (end - start) as usize;
            chunk.resize(len, 0);
            file.seek(SeekFrom::Start(start))
                .map_err(|e| format!("Failed to read memory file: {e}"))?;
            file.read_exact(&mut chunk[..len])
                .map_err(|e| format!("Failed to read memory file: {e}"))?;

            match chunk.iter().rposition(|b| *b == b'\n') {
                Some(i) => {
                    // The line is the segment between the last '\n' in the
                    // chunk and `end`; everything at or after that '\n' is
                    // scanned once the line is consumed.
                    Self::parse_tail_window_line(&chunk[i + 1..len], &mut collected, limit)?;
                    end = start + i as u64;
                    window = CHUNK as u64;
                }
                None if start == 0 => {
                    // No '\n' left before the region start: the rest of the
                    // file is a single line.
                    Self::parse_tail_window_line(&chunk[..len], &mut collected, limit)?;
                    end = 0;
                }
                None => {
                    // No '\n' inside the window but the file continues
                    // further back: the line spans the whole window, so grow
                    // it and retry. Terminates once start reaches 0.
                    window = window.saturating_mul(2).min(end);
                }
            }
            scan_stop = end;
        }

        Self::ensure_range_is_utf8(&mut file, scan_stop)?;
        Ok(collected)
    }

    /// Decode one raw `\n`-delimited line exactly like `recent` does via
    /// `BufRead::lines` + `trim`: invalid UTF-8 fails the whole scan, blank
    /// lines and lines that fail JSON parsing are skipped and do not count
    /// toward the window limit.
    fn parse_tail_window_line(
        raw: &[u8],
        collected: &mut Vec<AgentMemoryEntry>,
        limit: usize,
    ) -> Result<(), String> {
        let line = std::str::from_utf8(raw)
            .map_err(|_| "Failed to read memory file: stream did not contain valid UTF-8")?;
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        if collected.len() < limit {
            if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                collected.push(entry);
            }
        }
        Ok(())
    }

    /// Validate that the bytes `[0, upto)` are valid UTF-8 without parsing
    /// them, so the backward scan in `recent_tail_window` keeps `recent`'s
    /// behavior of failing the whole read when any part of the file is not
    /// valid UTF-8.
    fn ensure_range_is_utf8(file: &mut fs::File, upto: u64) -> Result<(), String> {
        if upto == 0 {
            return Ok(());
        }
        const BLOCK: usize = 64 * 1024;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to read memory file: {e}"))?;
        let mut block = vec![0u8; BLOCK];
        // Trailing bytes of a multi-byte character split by the block
        // boundary; prepended to the next block before decoding.
        let mut carry: Vec<u8> = Vec::new();
        let mut done = 0u64;
        while done < upto {
            let want = ((upto - done) as usize).min(BLOCK);
            file.read_exact(&mut block[..want])
                .map_err(|e| format!("Failed to read memory file: {e}"))?;
            carry.extend_from_slice(&block[..want]);
            match std::str::from_utf8(&carry) {
                Ok(_) => carry.clear(),
                Err(e) => {
                    if e.error_len().is_none() && done as usize + want < upto as usize {
                        // Incomplete only because the character may continue
                        // in the next block; keep it and retry.
                        carry.drain(..e.valid_up_to());
                    } else {
                        return Err(
                            "Failed to read memory file: stream did not contain valid UTF-8"
                                .to_string(),
                        );
                    }
                }
            }
            done += want as u64;
        }
        Ok(())
    }

    /// Apply "delete + add" changes in batch. Prepare all target content first, then commit under the same main-file lock;
    /// on mid-way failure the already-committed files are restored, so the current file and rotated archives are never partially updated.
    /// JSONL remains the source of truth; the SQLite index is fully rebuilt best-effort after a successful write-back.
    pub(crate) fn apply_batch_update(
        &self,
        delete_ids: &[&str],
        new_entries: &[AgentMemoryEntry],
    ) -> Result<MemoryBatchUpdateReport, String> {
        if delete_ids.is_empty() && new_entries.is_empty() {
            return Ok(MemoryBatchUpdateReport {
                deleted: 0,
                appended: 0,
            });
        }
        let id_set: FxHashSet<&str> = delete_ids.iter().copied().collect();
        super::with_memory_file_lock(&self.path, || {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("Failed to create memory dir: {err}"))?;
            }
            let mut paths = if id_set.is_empty() {
                vec![self.path.clone()]
            } else {
                self.memory_files_to_scan_consolidate()?
            };
            if !paths.contains(&self.path) {
                paths.insert(0, self.path.clone());
            }

            let mut rewrites = Vec::new();
            let mut deleted_total = 0usize;
            for path in paths {
                let is_current = path == self.path;
                let original = match fs::read(&path) {
                    Ok(bytes) => Some(bytes),
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound && is_current => None,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(err) => {
                        return Err(format!(
                            "Failed to read memory file {}: {err}",
                            path.display()
                        ));
                    }
                };
                let content = original
                    .as_deref()
                    .map(std::str::from_utf8)
                    .transpose()
                    .map_err(|err| {
                        format!("Invalid UTF-8 in memory file {}: {err}", path.display())
                    })?
                    .unwrap_or_default();
                let mut kept = Vec::new();
                let mut deleted = 0usize;
                for line in content
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                {
                    if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                        if entry.id.as_deref().is_some_and(|id| id_set.contains(id)) {
                            deleted += 1;
                        } else {
                            kept.push(entry);
                        }
                    }
                }
                if is_current {
                    kept.extend(new_entries.iter().cloned());
                }
                if is_current || deleted > 0 {
                    deleted_total += deleted;
                    rewrites.push((path, original, kept));
                }
            }

            let mut committed: Vec<usize> = Vec::new();
            for (path, _, entries) in &rewrites {
                if let Err(commit_err) = Self::write_all_entries(path, entries) {
                    let mut rollback_errors = Vec::new();
                    for &index in committed.iter().rev() {
                        let (written_path, written_original, _) = &rewrites[index];
                        let result = match written_original {
                            Some(bytes) => atomic_write_file(written_path, bytes),
                            None => match fs::remove_file(written_path) {
                                Ok(()) => Ok(()),
                                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
                                Err(err) => Err(err),
                            },
                        };
                        if let Err(err) = result {
                            rollback_errors.push(format!("{}: {err}", written_path.display()));
                        }
                    }
                    let rollback_suffix = if rollback_errors.is_empty() {
                        String::new()
                    } else {
                        format!("; rollback also failed for {}", rollback_errors.join(", "))
                    };
                    return Err(format!(
                        "Failed to commit memory batch at {}: {commit_err}{rollback_suffix}",
                        path.display()
                    ));
                }
                committed.push(committed.len());
            }

            for (path, _, _) in &rewrites {
                if path == &self.path || derive_db_path(path).is_some_and(|db| db.exists()) {
                    rebuild_index_for_path(path);
                }
            }

            Ok(MemoryBatchUpdateReport {
                deleted: deleted_total,
                appended: new_entries.len(),
            })
        })
    }

    fn memory_files_to_scan(&self, include_archives: bool) -> Result<Vec<PathBuf>, String> {
        let cfg = configw::get_all_config();
        let search_archives = cfg
            .get_opt("ai.memory.search_archives.enable")
            .unwrap_or_else(|| "false".to_string())
            .trim()
            .eq_ignore_ascii_case("true");
        let keep_last_archives = cfg
            .get_opt("ai.memory.search_archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        let mut files: Vec<PathBuf> = Vec::new();
        let archives = self.collect_archive_files(include_archives)?;
        if include_archives {
            // Explicit archive scan request (e.g. -ns memo retrieval): no truncation,
            // ensuring historical memos moved into old archives by rotation stay retrievable.
            // keep_last_archives truncation applies only to the global
            // full-search performance optimization when search_archives.enable is on.
            files.extend(archives.into_iter().map(|(path, _)| path));
        } else if search_archives {
            let take_from = archives.len().saturating_sub(keep_last_archives);
            files.extend(archives.into_iter().skip(take_from).map(|(path, _)| path));
        }
        files.push(self.path.clone());
        Ok(files)
    }

    /// Collect archive files (rotated archives + optional legacy migration backups), returned in ascending mtime order.
    fn collect_archive_files(
        &self,
        include_legacy_backups: bool,
    ) -> Result<Vec<(PathBuf, SystemTime)>, String> {
        let Some(parent) = self.path.parent() else {
            return Ok(Vec::new());
        };
        let base = self
            .path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();
        let legacy_base = self
            .path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();
        let archive_prefix = format!("{base}.");
        let legacy_migration_prefix = format!("{legacy_base}.legacy-migrate-");
        let mut archives = Vec::new();
        for entry in fs::read_dir(parent).map_err(|e| format!("{}", e))? {
            let entry = entry.map_err(|e| format!("{}", e))?;
            let file_name = entry.file_name().to_str().unwrap_or("").to_string();
            let is_rotation_archive = file_name.starts_with(&archive_prefix);
            // Legacy migration used to leave the original JSONL as
            // `agent_memory.legacy-migrate-<timestamp>.jsonl.bak`. It does not match
            // the current rotation naming `<base>.{timestamp}`; include it only when an explicit
            // archive query (-ns etc.) asks for it, so ordinary current-file searches never read a stale migration snapshot.
            let is_legacy_migration_backup = include_legacy_backups
                && file_name.starts_with(&legacy_migration_prefix)
                && file_name.ends_with(".jsonl.bak");
            if !is_rotation_archive && !is_legacy_migration_backup {
                continue;
            }
            let meta = entry.metadata().map_err(|e| format!("{}", e))?;
            if !meta.is_file() {
                continue;
            }
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            archives.push((entry.path(), modified));
        }
        archives.sort_by_key(|(_, modified)| *modified);
        Ok(archives)
    }

    /// Scan dedicated to --consolidate-knowledge: includes all rotated archives, excludes legacy migration backups.
    /// Consolidation must see historical entries moved into archives by rotation; migration backups are read-only historical snapshots,
    /// not part of the consolidation view (and never rewritten).
    fn memory_files_to_scan_consolidate(&self) -> Result<Vec<PathBuf>, String> {
        let mut files: Vec<PathBuf> = self
            .collect_archive_files(false)?
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        files.push(self.path.clone());
        Ok(files)
    }

    pub(crate) fn entries_by_category(
        &self,
        category: &str,
        limit: usize,
        include_archives: bool,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        self.entries_by_category_from_paths(category, limit, || {
            self.memory_files_to_scan(include_archives)
        })
    }

    pub(crate) fn entries_by_category_current_file(
        &self,
        category: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        self.entries_by_category_from_paths(category, limit, || Ok(vec![self.path.clone()]))
    }

    fn entries_by_category_from_paths<F>(
        &self,
        category: &str,
        limit: usize,
        files: F,
    ) -> Result<Vec<AgentMemoryEntry>, String>
    where
        F: FnOnce() -> Result<Vec<PathBuf>, String>,
    {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::new();
        for path in files()? {
            if !path.exists() {
                continue;
            }
            let file =
                fs::File::open(&path).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                    continue;
                };
                if entry.category != category {
                    continue;
                }
                window.push_back(entry);
                if window.len() > limit {
                    window.pop_front();
                }
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }

    pub(crate) fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(AgentMemoryEntry, f64)>, String> {
        let query_lc = query.to_lowercase();

        // Fast path: first use SQLite FTS5 to get the candidate id set (O(log N) MATCH),
        // then go back to the JSONL to load the candidate entries exactly and run the existing BM25 + text-similarity scoring.
        // This reduces search from "full-file scan + full-file tokenize" to "hit lines + tokenize",
        // with output format / score weights / ordering logic unchanged.
        // When FTS is unavailable or candidates are too few, fall back to the original scan path with fully equivalent behavior.
        let fts_candidate_cap = limit.saturating_mul(20).max(60).min(400);
        let fts_ids: Option<std::collections::HashSet<String>> = memory_index_for(&self.path)
            .and_then(|idx| match idx.search_ids(&query_lc, fts_candidate_cap) {
                Ok(v) if v.len() >= limit => Some(v.into_iter().collect()),
                _ => None,
            });

        let mut docs: Vec<(AgentMemoryEntry, String, Vec<String>)> = Vec::new();
        for p in self.memory_files_to_scan(false)? {
            if !p.exists() {
                continue;
            }
            let file =
                fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                    continue;
                };
                if let Some(ids) = &fts_ids {
                    // Fast path: keep only FTS-hit entries
                    if let Some(id) = entry.id.as_deref() {
                        if !ids.contains(id) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                let mut full = String::new();
                full.push_str(&entry.category);
                full.push(' ');
                full.push_str(&entry.note);
                if let Some(s) = &entry.source {
                    full.push(' ');
                    full.push_str(s);
                }
                if !entry.tags.is_empty() {
                    full.push(' ');
                    full.push_str(&entry.tags.join(" "));
                }
                let tokens = similarity::expand_tokens(&similarity::tokenize(&full.to_lowercase()));
                docs.push((entry, full, tokens));
            }
        }
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let nq_tokens = similarity::expand_tokens(&similarity::tokenize(&query_lc));
        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        let mut avgdl = 0.0f64;
        for (_, _, toks) in &docs {
            avgdl += toks.len() as f64;
            let mut set: FxHashSet<&str> = FxHashSet::default();
            for t in toks {
                if set.insert(t.as_str()) {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        avgdl /= docs.len() as f64;
        let n_docs = docs.len() as f64;
        let k1 = 1.2f64;
        let b = 0.75f64;
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(docs.len());
        let mut bm25_vals: Vec<f64> = Vec::with_capacity(docs.len());
        for (idx, (_entry, _full, toks)) in docs.iter().enumerate() {
            let mut tf: FxHashMap<&str, usize> = FxHashMap::default();
            for t in toks {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut bm25 = 0.0f64;
            let dl = toks.len() as f64;
            let mut seenq: FxHashSet<&str> = FxHashSet::default();
            for qt in &nq_tokens {
                if !seenq.insert(qt.as_str()) {
                    continue;
                }
                let dfv = *df.get(qt.as_str()).unwrap_or(&0) as f64;
                if dfv <= 0.0 {
                    continue;
                }
                let idf = ((n_docs - dfv + 0.5) / (dfv + 0.5) + 1.0).ln();
                let tfv = *tf.get(qt.as_str()).unwrap_or(&0) as f64;
                if tfv <= 0.0 {
                    continue;
                }
                let denom = tfv + k1 * (1.0 - b + b * (dl / avgdl.max(1e-6)));
                bm25 += idf * (tfv * (k1 + 1.0)) / denom;
            }
            bm25_vals.push(bm25);
            // Ranking is BM25-only here: the character-similarity re-scoring
            // (compute_similarity) was removed because it re-weighted the same
            // token space BM25 already scores. The priority boost applies below.
            scored.push((bm25, idx));
        }
        let max_bm25 = bm25_vals.iter().cloned().fold(0.0f64, f64::max);
        for i in 0..scored.len() {
            scored[i].0 = if max_bm25 > 0.0 {
                scored[i].0 / max_bm25
            } else {
                0.0
            };
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let cap = limit.saturating_mul(10).min(200).max(limit);
        let mut top_idx: Vec<(f64, usize)> =
            scored.iter().take(cap).map(|(s, i)| (*s, *i)).collect();
        // Priority boost: scale the blended score by entry priority so that
        // explicitly high-value knowledge (High 100-200, Permanent 255) ranks
        // above same-relevance low-priority entries. Default 100 → ×1.0;
        // permanent 255 → ~×1.8; low 0 → ~×0.5.
        for i in 0..top_idx.len() {
            let pri = docs[top_idx[i].1].0.priority.unwrap_or(100) as f64;
            top_idx[i].0 *= 1.0 + (pri - 100.0) / 200.0;
        }
        top_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        top_idx.truncate(limit);
        let mut out = Vec::with_capacity(top_idx.len());
        for (s, i) in top_idx {
            out.push((docs[i].0.clone(), s));
        }
        // Count LFU for hit entries; failures are traced only. Count only the top-N (already truncated to limit),
        // not the cap=200 intermediate set, so low-scoring fringe entries do not inflate their hits.
        if let Some(idx) = memory_index_for(&self.path) {
            let ids: Vec<String> = out.iter().filter_map(|(e, _)| e.id.clone()).collect();
            if !ids.is_empty() {
                if let Err(e) = idx.record_hits(&ids) {
                    trace_memory_event(
                        "memory.index.hits_failed",
                        "MemoryIndex record_hits failed",
                        &[("path", self.path.display().to_string()), ("error", e)],
                    );
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file =
            fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let reader = BufReader::new(file);

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::with_capacity(limit + 1);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                continue;
            };
            window.push_back(entry);
            if window.len() > limit {
                window.pop_front();
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn test_search_recall_ngram() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e1 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "log".to_string(),
            note: "parsing login error occurred".to_string(),
            tags: vec!["auth".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let e2 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-02T00:00:00Z".to_string(),
            category: "info".to_string(),
            note: "user profile updated".to_string(),
            tags: vec!["user".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e1).unwrap();
        store.append(&e2).unwrap();
        let out = store.search("parse login", 5).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("parsing login")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_synonym_login() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_syn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-03T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "user login failed due to authentication error".to_string(),
            tags: vec!["login".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("signin failure", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("login failed")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_chinese_login_variants() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_cn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-04T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "登录失败，密码错误".to_string(),
            tags: vec!["登录".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("登陆失败", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("登录失败")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_dedup_scans_tail_window_across_bad_lines() {
        let path = std::env::temp_dir().join(format!(
            "rt_mem_tail_window_{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let mk = |note: &str| AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "self_note".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: Some("session:test".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        // Write the JSONL directly with trailing corrupt/blank lines to cover
        // the backward scan's tolerance for bad tail lines.
        let mut buf = String::new();
        for i in 0..5 {
            buf.push_str(&serde_json::to_string(&mk(&format!("note-{i}"))).unwrap());
            buf.push('\n');
        }
        buf.push_str("not-json\n\n");
        std::fs::write(&path, buf).unwrap();

        // Duplicate of the newest good entry: must hit the tail-window dedup.
        store.append(&mk("note-4")).unwrap();
        // Duplicate of an older-but-still-in-window entry: deduped too.
        store.append(&mk("note-0")).unwrap();
        // A genuinely new note must be appended.
        store.append(&mk("brand-new")).unwrap();

        let recent = store.recent(20).unwrap();
        assert_eq!(recent.iter().filter(|e| e.note == "note-4").count(), 1);
        assert_eq!(recent.iter().filter(|e| e.note == "note-0").count(), 1);
        assert!(recent.iter().any(|e| e.note == "brand-new"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn learning_entries_deduplicate_recent_exact_writes() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_dedup_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());

        let entry = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "self_note".to_string(),
            note: "Do: verify before write".to_string(),
            tags: vec!["agent".to_string()],
            source: Some("session:test".to_string()),
            priority: Some(120),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        store.append(&entry).unwrap();
        store.append(&entry).unwrap();

        let recent = store.recent(10).unwrap();
        assert_eq!(
            recent
                .iter()
                .filter(|e| e.category == "self_note" && e.note == "Do: verify before write")
                .count(),
            1
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn idempotent_knowledge_write_returns_existing_entry_without_appending() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_knowledge_dedup_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let first = AgentMemoryEntry {
            id: Some("mem_existing".to_string()),
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "user_memory".to_string(),
            note: "Keep project decisions in the architecture log.".to_string(),
            tags: vec!["architecture".to_string(), "decision".to_string()],
            source: Some("project:demo".to_string()),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let retry = AgentMemoryEntry {
            id: Some("mem_retry".to_string()),
            timestamp: "2025-01-06T00:00:00Z".to_string(),
            category: " USER_MEMORY ".to_string(),
            note: "  keep project decisions in the architecture log.  ".to_string(),
            tags: vec!["decision".to_string(), "ARCHITECTURE".to_string()],
            source: Some(" PROJECT:DEMO ".to_string()),
            priority: Some(200),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        assert_eq!(
            store.append_idempotent_knowledge(&first).unwrap(),
            KnowledgeAppendOutcome::Appended
        );
        assert_eq!(
            store.append_idempotent_knowledge(&retry).unwrap(),
            KnowledgeAppendOutcome::Duplicate {
                existing_id: Some("mem_existing".to_string())
            }
        );
        assert_eq!(store.recent(10).unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db"));
    }

    #[test]
    fn delete_subagent_memory_removes_only_its_knowledge_dedup_cache_entry() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("rt_mem_cache_lifecycle_{ts}"));
        std::fs::create_dir_all(&dir).unwrap();
        let removed_path = dir.join("agent_memory.subagent-removed.jsonl");
        let retained_path = dir.join("agent_memory.subagent-retained.jsonl");
        let entry = AgentMemoryEntry {
            id: Some("mem_cache_lifecycle".to_string()),
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "user_memory".to_string(),
            note: "Cache lifecycle test entry.".to_string(),
            tags: vec!["test".to_string()],
            source: Some("test".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        for path in [&removed_path, &retained_path] {
            let store = MemoryStore::for_tests_with_path(path.clone());
            assert_eq!(
                store.append_idempotent_knowledge(&entry).unwrap(),
                KnowledgeAppendOutcome::Appended
            );
        }
        {
            let cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
            assert!(cache.contains_key(&removed_path));
            assert!(cache.contains_key(&retained_path));
        }

        crate::ai::history::delete_subagent_memory(&removed_path).unwrap();

        {
            let cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
            assert!(!cache.contains_key(&removed_path));
            assert!(cache.contains_key(&retained_path));
        }

        crate::ai::history::delete_subagent_memory(&retained_path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_batch_update_rewrites_delete_and_append_in_one_pass() {
        let path = std::env::temp_dir().join(format!(
            "rt_mem_batch_update_{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let entry_with_id = |id: &str, note: &str, ts: &str| AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: ts.to_string(),
            category: "user_memory".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let write_lines = |entries: &[AgentMemoryEntry]| {
            let mut buf = String::new();
            for entry in entries {
                buf.push_str(&serde_json::to_string(entry).unwrap());
                buf.push('\n');
            }
            std::fs::write(&path, buf).unwrap();
        };
        let read_entries = || -> Vec<AgentMemoryEntry> {
            std::fs::read_to_string(&path)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
                .collect()
        };
        write_lines(&[
            entry_with_id("mem_1", "keep me", "2025-01-01T00:00:00Z"),
            entry_with_id("mem_2", "drop me", "2025-01-01T00:00:01Z"),
            entry_with_id("mem_3", "merge me", "2025-01-01T00:00:02Z"),
        ]);

        let store = MemoryStore::for_tests_with_path(path.clone());
        let merged = entry_with_id("mem_merged", "merged note", "2025-01-02T00:00:00Z");
        let report = store
            .apply_batch_update(&["mem_2", "mem_3"], &[merged.clone()])
            .unwrap();

        assert_eq!(
            report,
            MemoryBatchUpdateReport {
                deleted: 2,
                appended: 1
            }
        );

        let kept = read_entries();
        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter()
                .any(|entry| entry.id.as_deref() == Some("mem_1"))
        );
        assert!(
            kept.iter()
                .any(|entry| entry.id.as_deref() == Some("mem_merged"))
        );
        assert!(
            !kept
                .iter()
                .any(|entry| entry.id.as_deref() == Some("mem_2"))
        );
        assert!(
            !kept
                .iter()
                .any(|entry| entry.id.as_deref() == Some("mem_3"))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_batch_update_deletes_across_rotation_archives() {
        let dir = std::env::temp_dir().join(format!(
            "rt_mem_archive_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let current = dir.join("agent_memory.jsonl");
        let archive = dir.join("agent_memory.jsonl.20260101000000");
        let entry_with_id = |id: &str, note: &str, ts: &str| AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: ts.to_string(),
            category: "user_memory".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let write_lines = |path: &std::path::Path, entries: &[AgentMemoryEntry]| {
            let mut buf = String::new();
            for entry in entries {
                buf.push_str(&serde_json::to_string(entry).unwrap());
                buf.push('\n');
            }
            std::fs::write(path, buf).unwrap();
        };
        let read_ids = |path: &std::path::Path| -> Vec<String> {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
                .filter_map(|entry| entry.id)
                .collect()
        };
        // Current file: keep cur_a, delete cur_b; archives: delete arch_c, keep arch_d.
        write_lines(
            &current,
            &[
                entry_with_id("cur_a", "keep me", "2025-01-01T00:00:00Z"),
                entry_with_id("cur_b", "drop me", "2025-01-01T00:00:01Z"),
            ],
        );
        write_lines(
            &archive,
            &[
                entry_with_id("arch_c", "drop in archive", "2025-01-01T00:00:02Z"),
                entry_with_id("arch_d", "keep in archive", "2025-01-01T00:00:03Z"),
            ],
        );

        let store = MemoryStore::for_tests_with_path(current.clone());
        let merged = entry_with_id("merged_1", "merged note", "2025-01-02T00:00:00Z");
        let report = store
            .apply_batch_update(&["cur_b", "arch_c"], &[merged.clone()])
            .unwrap();

        assert_eq!(
            report,
            MemoryBatchUpdateReport {
                deleted: 2,
                appended: 1
            }
        );
        assert_eq!(
            read_ids(&current),
            vec!["cur_a".to_string(), "merged_1".to_string()]
        );
        assert_eq!(read_ids(&archive), vec!["arch_d".to_string()]);

        // all_with_archives must see entries from both the main file and the rotated archives.
        let all_ids: Vec<String> = store
            .all_with_archives()
            .unwrap()
            .into_iter()
            .filter_map(|entry| entry.id)
            .collect();
        assert_eq!(all_ids.len(), 3);
        assert!(all_ids.contains(&"cur_a".to_string()));
        assert!(all_ids.contains(&"arch_d".to_string()));
        assert!(all_ids.contains(&"merged_1".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }
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

impl MemoryStore {
    pub(crate) fn rotate_if_exceeds(&self, max_bytes: u64) -> Result<bool, String> {
        let path = self.path().to_path_buf();
        with_memory_file_lock(&path, || {
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => return Ok(false),
            };
            if meta.len() <= max_bytes {
                return Ok(false);
            }

            // Fix P0-2: the original implementation used `rename` + `File::create` directly, freezing even entries in the
            // category whitelist (permanent entries: safety_rules / reflection self_note / coding_guideline /
            // user_preference / project_memory ...) into the archive, after which they default
            // out of recall — effectively discarding the core rules of "long-term memory".
            //
            // Now all entries are read out first: whitelist entries stay in the new main file, the rest go to the archive:
            //   - new main file = original content ∩ {is_permanent_memory}
            //   - archive file = original content (unchanged, same as the old implementation)
            // That way long-term assets are never lost, regardless of whether the recall layer enables search_archives.
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read memory file before rotate: {}", e))?;

            let entries: Vec<AgentMemoryEntry> = content
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    serde_json::from_str::<AgentMemoryEntry>(line).ok()
                })
                .collect();
            let permanent: Vec<&AgentMemoryEntry> = entries
                .iter()
                .filter(|e| crate::ai::tools::service::memory::is_permanent_memory(e))
                .collect();
            let preserved_total = permanent.len();
            let archived_total = entries.len();

            let ts = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
            let mut new_name = path.clone();
            let ext = new_name
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("jsonl")
                .to_string();
            new_name.set_extension(format!("{ext}.{}", ts));
            std::fs::rename(&path, &new_name)
                .map_err(|e| format!("Failed to rotate file: {}", e))?;

            // Rebuild the main file: write back all priority=255 entries (original timestamp order preserved)
            let mut head = String::new();
            for entry in &permanent {
                if let Ok(s) = serde_json::to_string(*entry) {
                    head.push_str(&s);
                    head.push('\n');
                }
            }
            atomic_write_file(&path, head.as_bytes()).map_err(|e| {
                format!(
                    "Failed to recreate memory file with permanent entries after rotate: {}",
                    e
                )
            })?;

            trace_memory_event(
                "memory.rotate",
                "memory file rotated; permanent entries preserved in head",
                &[
                    ("path", path.display().to_string()),
                    ("archive", new_name.display().to_string()),
                    ("archived_total", archived_total.to_string()),
                    ("preserved_permanent", preserved_total.to_string()),
                    ("max_bytes", max_bytes.to_string()),
                    ("file_size", meta.len().to_string()),
                ],
            );
            // Rotation moves the vast majority of entries to the archive, leaving the index content badly stale.
            // Trigger a rebuild directly here — the main file now holds only permanent entries, so the rebuild is cheap.
            if let Some(idx) = memory_index_for(&path) {
                let _ = idx.rebuild_from_source();
            }
            Ok(true)
        })
    }

    pub(crate) fn maintain_after_append(&self) {
        let cfg = configw::get_all_config();
        let max_bytes = cfg
            .get_opt("ai.memory.auto_rotate.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(8 * 1024 * 1024);
        let gc_days = cfg
            .get_opt("ai.memory.auto_gc.days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(30);
        let min_keep = cfg
            .get_opt("ai.memory.auto_gc.min_keep")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(200);
        let prob = cfg
            .get_opt("ai.memory.auto_maintain.probability")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.05);
        let max_entries = cfg
            .get_opt("ai.memory.quota.max_entries")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10000);
        let rotated = self.rotate_if_exceeds(max_bytes).unwrap_or(false);
        let _ = if rotated {
            self.cleanup_archives_auto()
        } else {
            Ok(())
        };
        let _ = self.enforce_max_entries(max_entries, min_keep);
        let roll = rand::random::<f64>();
        if roll < prob {
            let _ = execute_memory_dedup(&json!({}));
            let _ = execute_memory_gc(&json!({ "max_days": gc_days, "min_keep": min_keep }));
            let _ = self.cleanup_archives_auto();
        }
    }

    fn enforce_max_entries(&self, max_entries: usize, min_keep: usize) -> Result<(), String> {
        super::with_memory_file_lock(&self.path, || {
            let content = std::fs::read_to_string(&self.path)
                .map_err(|e| format!("Failed to read memory file: {}", e))?;

            let mut entries: Vec<AgentMemoryEntry> = content
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return None;
                    }
                    serde_json::from_str::<AgentMemoryEntry>(line).ok()
                })
                .collect();

            let original_total = entries.len();
            if original_total <= max_entries {
                return Ok(());
            }

            // Sort key: permanent entries (whitelist: safety/preference/coding_guideline/
            // project_memory/...) always last; the rest by ascending priority then ascending ts.
            // Deletion then cuts the lowest-priority, oldest entries from the front.
            entries.sort_by(|a, b| {
                let perm_a = crate::ai::tools::service::memory::is_permanent_memory(a);
                let perm_b = crate::ai::tools::service::memory::is_permanent_memory(b);
                if perm_a && !perm_b {
                    return std::cmp::Ordering::Greater;
                }
                if perm_b && !perm_a {
                    return std::cmp::Ordering::Less;
                }
                let pa = a.priority.unwrap_or(100);
                let pb = b.priority.unwrap_or(100);
                pa.cmp(&pb).then_with(|| a.timestamp.cmp(&b.timestamp))
            });

            // Fix P0-1: the original implementation used `while … { remove(i); if … { remove(i); } }`
            // deleting twice at the same index — the second remove actually deleted the next entry that had already "shifted up",
            // and it never checked the permanent-entry skip, so whitelist entries could be hit by mistake. Changed to a single remove +
            // leaving i unchanged (after remove(i) the next entry lands at i), with permanent whitelist entries skipped.
            //
            // The target field now only serves "stop as soon as the quota is met"; it no longer triggers a second deletion.
            let target = max_entries.saturating_sub(min_keep);
            let mut removed = 0usize;
            let mut skipped_permanent = 0usize;
            let mut i = 0usize;
            while i < entries.len() && entries.len() > max_entries {
                if crate::ai::tools::service::memory::is_permanent_memory(&entries[i]) {
                    // Permanent entries are always skipped and must not be removed.
                    skipped_permanent += 1;
                    i += 1;
                    continue;
                }
                entries.remove(i);
                removed += 1;
                if target > 0 && removed >= target && entries.len() <= max_entries {
                    break;
                }
            }

            let mut output = String::new();
            for entry in &entries {
                if let Ok(s) = serde_json::to_string(entry) {
                    output.push_str(&s);
                    output.push('\n');
                }
            }

            // Fix P1-1: the original `fs::write(&path, output)` was truncate-then-write,
            // leaving an incomplete main file on a mid-way crash. Switched to tmp+rename for filesystem-level atomicity.
            atomic_write_file(&self.path, output.as_bytes()).map_err(|e| {
                format!("Failed to write memory file after quota enforcement: {}", e)
            })?;

            // The file was fully rewritten: trigger an index rebuild to stay consistent; the rebuild wraps a transaction internally and failures are only traced.
            if let Some(idx) = memory_index_for(&self.path) {
                let _ = idx.rebuild_from_source();
            }

            trace_memory_event(
                "memory.enforce_max_entries",
                "memory quota enforced",
                &[
                    ("path", self.path.display().to_string()),
                    ("before", original_total.to_string()),
                    ("after", entries.len().to_string()),
                    ("removed", removed.to_string()),
                    ("skipped_permanent", skipped_permanent.to_string()),
                    ("max_entries", max_entries.to_string()),
                    ("min_keep", min_keep.to_string()),
                ],
            );
            Ok(())
        })
    }

    /// Batch-delete memory entries (atomic: read once → filter → write back).
    /// Returns the number of entries actually deleted.
    pub(crate) fn delete_by_ids(&self, ids: &[&str]) -> Result<usize, String> {
        if ids.is_empty() {
            return Ok(0);
        }
        self.apply_batch_update(ids, &[])
            .map(|report| report.deleted)
    }

    /// Delete an entry by id (returns the deleted entry)
    pub(crate) fn delete_by_id(&self, id: &str) -> Result<Option<AgentMemoryEntry>, String> {
        super::with_memory_file_lock(&self.path, || {
            let content = std::fs::read_to_string(&self.path)
                .map_err(|e| format!("Failed to read memory file: {}", e))?;

            let mut entries: Vec<AgentMemoryEntry> = Vec::new();
            let mut deleted_entry: Option<AgentMemoryEntry> = None;

            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                    let entry_id = entry.id.as_deref().unwrap_or("");
                    if entry_id == id {
                        deleted_entry = Some(entry);
                        continue;
                    }
                    entries.push(entry);
                }
            }

            if deleted_entry.is_none() {
                return Ok(None);
            }

            let mut output = String::new();
            for entry in &entries {
                if let Ok(s) = serde_json::to_string(entry) {
                    output.push_str(&s);
                    output.push('\n');
                }
            }

            // Consistent with enforce_max_entries: tmp + rename atomic write, avoiding an incomplete main file after a crash.
            atomic_write_file(&self.path, output.as_bytes())
                .map_err(|e| format!("Failed to write memory file: {}", e))?;

            Ok(deleted_entry)
        })
    }

    /// Batch-append memory entries; internally reuses `apply_batch_update([], entries)`,
    /// using a single atomic rewrite to avoid intermediate states like "half appended".
    pub(crate) fn append_batch(&self, entries: &[AgentMemoryEntry]) -> Result<usize, String> {
        if entries.is_empty() {
            return Ok(0);
        }
        self.apply_batch_update(&[], entries)
            .map(|report| report.appended)
    }

    fn cleanup_archives_auto(&self) -> Result<(), String> {
        let cfg = configw::get_all_config();
        let retain_days = cfg
            .get_opt("ai.memory.archives.retain_days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(60);
        let keep_last = cfg
            .get_opt("ai.memory.archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10);
        let max_total = cfg
            .get_opt("ai.memory.archives.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(64 * 1024 * 1024);
        self.cleanup_archives(retain_days, keep_last, max_total)
    }

    pub(crate) fn cleanup_archives(
        &self,
        retain_days: i64,
        keep_last: usize,
        max_total_bytes: u64,
    ) -> Result<(), String> {
        let path = self.path().to_path_buf();
        let parent = match path.parent() {
            Some(p) => p.to_path_buf(),
            None => return Ok(()),
        };
        let base = match path.file_name().and_then(OsStr::to_str) {
            Some(s) => s.to_string(),
            None => return Ok(()),
        };

        let mut archives = Vec::new();
        for entry in std::fs::read_dir(&parent).map_err(|e| format!("{}", e))? {
            let entry = entry.map_err(|e| format!("{}", e))?;
            let file_name = entry.file_name().to_str().unwrap_or("").to_string();
            if !file_name.starts_with(&(base.clone() + ".")) {
                continue;
            }
            let meta = entry.metadata().map_err(|e| format!("{}", e))?;
            if !meta.is_file() {
                continue;
            }
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let size = meta.len();
            archives.push((entry.path(), modified, size));
        }

        if archives.is_empty() {
            return Ok(());
        }

        archives.sort_by_key(|(_, modified, _)| *modified);

        // Age-based cleanup
        if retain_days > 0 {
            let cutoff = SystemTime::now()
                .checked_sub(Duration::from_secs((retain_days as u64) * 86400))
                .unwrap_or(SystemTime::UNIX_EPOCH);
            for (p, m, _) in archives.clone() {
                if m < cutoff {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }

        // Refresh list after potential deletions
        let mut archives2 = Vec::new();
        for (p, m, s) in archives.into_iter() {
            if p.exists() {
                archives2.push((p, m, s));
            }
        }
        if archives2.is_empty() {
            return Ok(());
        }
        archives2.sort_by_key(|(_, modified, _)| *modified);

        // Keep last N
        if archives2.len() > keep_last {
            let to_delete = archives2.len() - keep_last;
            for i in 0..to_delete {
                let (p, _, _) = &archives2[i];
                let _ = std::fs::remove_file(p);
            }
        }

        // Size cap
        let mut archives3: Vec<(std::path::PathBuf, SystemTime, u64)> = archives2
            .into_iter()
            .filter(|(p, _, _)| p.exists())
            .collect();
        archives3.sort_by_key(|(_, m, _)| *m);
        let mut total: u64 = archives3.iter().map(|(_, _, s)| *s).sum();
        let mut idx = 0usize;
        while total > max_total_bytes && idx < archives3.len() {
            let (p, _, s) = &archives3[idx];
            if std::fs::remove_file(p).is_ok() {
                total = total.saturating_sub(*s);
            }
            idx += 1;
        }
        Ok(())
    }
}

/// Memory importance scoring — used for active learning and automatic forgetting
#[derive(Debug, Clone)]
pub struct MemoryImportance {
    /// Reference count
    pub frequency: u32,
    /// Time-decay factor (0.0 - 1.0, higher when more recent)
    pub recency: f64,
    /// Breadth of applicability (0.0 - 1.0)
    pub generality: f64,
    /// Whether the user has confirmed it
    pub user_validated: bool,
}

impl MemoryImportance {
    pub fn new() -> Self {
        Self {
            frequency: 0,
            recency: 1.0,
            generality: 0.5,
            user_validated: false,
        }
    }

    /// Compute the composite importance score (0.0 - 1.0)
    pub fn score(&self) -> f64 {
        let freq_score = (self.frequency as f64).min(10.0) / 10.0; // 0-1
        let recency_score = self.recency.clamp(0.0, 1.0);
        let generality_score = self.generality.clamp(0.0, 1.0);
        let validation_bonus = if self.user_validated { 0.2 } else { 0.0 };

        // Weights: frequency 30%, recency 30%, generality 20%, user confirmation 20%
        (freq_score * 0.3 + recency_score * 0.3 + generality_score * 0.2 + validation_bonus)
            .min(1.0)
    }

    /// Increment the reference count
    pub fn increment_frequency(&mut self) {
        self.frequency += 1;
    }

    /// Update the time decay
    pub fn update_recency(&mut self, created_at: &str) {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
            let now = chrono::Utc::now();
            let age_days = (now - dt.with_timezone(&chrono::Utc)).num_seconds() as f64 / 86400.0;
            // Exponential decay: 30-day half-life
            self.recency = (-age_days * std::f64::consts::LN_2 / 30.0).exp();
        }
    }

    /// Evaluate generality (based on category and tags)
    pub fn evaluate_generality(&mut self, category: &str, tags: &[String]) {
        let general_categories = [
            "common_sense",
            "best_practice",
            "coding_guideline",
            "safety_rules",
        ];

        let general_tags = ["general", "universal", "fundamental", "core"];

        let category_score = if general_categories.contains(&category.as_ref()) {
            1.0
        } else {
            0.5
        };

        let tag_score = tags
            .iter()
            .filter(|t| general_tags.contains(&t.as_str()))
            .count() as f64
            / tags.len().max(1) as f64;

        self.generality = (category_score * 0.7 + tag_score * 0.3).clamp(0.0, 1.0);
    }

    /// Mark as user-confirmed
    pub fn mark_user_validated(&mut self) {
        self.user_validated = true;
    }

    /// Decide whether this should be forgotten (low-value memory)
    pub fn should_prune(&self, min_score: f64) -> bool {
        self.score() < min_score
    }
}

impl Default for MemoryImportance {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    /// Prune low-value memories.
    ///
    /// Removes memories that satisfy all of:
    /// - importance score < min_score
    /// - priority < 200 (not high priority)
    /// - older than max_age_days
    pub fn prune_low_value_memories(
        &self,
        min_score: f64,
        max_age_days: i64,
    ) -> Result<usize, String> {
        let entries = self.all()?;
        let mut to_remove = Vec::new();
        let now = chrono::Utc::now();

        for entry in entries {
            // Permanent and high-priority memories are not deleted
            if entry.priority.unwrap_or(100) >= 200 {
                continue;
            }

            // Compute importance
            let mut importance = MemoryImportance::new();
            importance.update_recency(&entry.timestamp);
            importance.evaluate_generality(&entry.category, &entry.tags);

            // Check age
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&entry.timestamp) {
                let age_days = (now - dt.with_timezone(&chrono::Utc)).num_days();
                if age_days > max_age_days && importance.should_prune(min_score) {
                    to_remove.push(entry.id.clone());
                }
            }
        }

        let removed_count = to_remove.len();

        // Batch removal: one load, one filter pass against the id set, one
        // rewrite. Sequentially filtering per id (the old behavior) yields
        // the same surviving set as this single pass, and a rewrite with no
        // matching id still replaces the file with its parseable entries, so
        // missing ids are handled the same way as before.
        let ids_to_remove: Vec<String> = to_remove.into_iter().flatten().collect();
        if !ids_to_remove.is_empty() {
            let id_set: FxHashSet<String> = ids_to_remove.iter().cloned().collect();
            super::with_memory_file_lock(&self.path, || {
                // Same enumeration as all(): the old per-id removal reloaded
                // through all(), so entries scanned from archives were folded
                // back into the main file on rewrite. Keep that behavior.
                let entries = self.load_entries_search_order()?;
                let surviving: Vec<AgentMemoryEntry> = entries
                    .into_iter()
                    .filter(|e| e.id.as_deref().map_or(true, |id| !id_set.contains(id)))
                    .collect();
                let mut output = String::new();
                for entry in &surviving {
                    let serialized = serde_json::to_string(entry)
                        .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;
                    output.push_str(&serialized);
                    output.push('\n');
                }
                // Atomic tmp + rename write: a failed rewrite leaves the
                // original file intact instead of the half-written state the
                // old truncate-then-write loop could produce.
                atomic_write_file(&self.path, output.as_bytes())
                    .map_err(|e| format!("Failed to write memory file: {e}"))?;

                // Same index handling as before: delete each removed id and
                // refresh the signature; failures only affect drift detection
                // on the next index open.
                if let Some(idx) = memory_index_for(&self.path) {
                    for id in &ids_to_remove {
                        let _ = idx.delete_id(id);
                    }
                    let _ = idx.refresh_signature();
                }
                Ok(())
            })?;
        }

        Ok(removed_count)
    }

    /// Rewrite the whole JSONL file (atomic write: tmp → rename).
    fn write_all_entries(path: &Path, entries: &[AgentMemoryEntry]) -> Result<(), String> {
        let mut output = String::new();
        for entry in entries {
            if let Ok(s) = serde_json::to_string(entry) {
                output.push_str(&s);
                output.push('\n');
            }
        }
        atomic_write_file(path, output.as_bytes())
            .map_err(|e| format!("Failed to write memory file: {}", e))
    }

    /// Get all memories.
    pub fn all(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        // Direct collection path, equivalent to the old
        // `search("", usize::MAX)`: an empty query tokenizes to no query
        // tokens, so BM25 scores exactly 0.0 for every document; score
        // normalization keeps 0.0 (max is 0) and the priority boost keeps
        // +0.0 (its factor is >= 0.5), so both stable sorts preserve document
        // order and `search` degenerates to "all parseable entries in scan
        // order". The FTS fast path can never trigger there because it
        // requires `candidates.len() >= usize::MAX`; skipping it only omits
        // one best-effort index probe. `record_hits` below mirrors the LFU
        // accounting `search` performs on the entries it returns.
        let entries = self.load_entries_search_order()?;
        if let Some(idx) = memory_index_for(&self.path) {
            let ids: Vec<String> = entries.iter().filter_map(|e| e.id.clone()).collect();
            if !ids.is_empty() {
                if let Err(e) = idx.record_hits(&ids) {
                    trace_memory_event(
                        "memory.index.hits_failed",
                        "MemoryIndex record_hits failed",
                        &[("path", self.path.display().to_string()), ("error", e)],
                    );
                }
            }
        }
        Ok(entries)
    }

    /// Collect every parseable entry in the exact order `search` scans:
    /// `memory_files_to_scan(false)` order, each file front to back,
    /// skipping blank and unparseable lines. Used by `all` and by batch
    /// rewrites that must observe exactly what `all` used to observe.
    fn load_entries_search_order(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        let mut entries = Vec::new();
        for p in self.memory_files_to_scan(false)? {
            if !p.exists() {
                continue;
            }
            let file =
                fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                    continue;
                };
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Get all memories (including all rotated archives, excluding legacy migration backups).
    /// Used by --consolidate-knowledge: consolidation must see historical entries moved
    /// into archives by rotation, otherwise they never enter the consolidation view.
    pub(crate) fn all_with_archives(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        let mut entries = Vec::new();
        for p in self.memory_files_to_scan_consolidate()? {
            if !p.exists() {
                continue;
            }
            let file =
                fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }

    /// Record that a memory was used (increment the reference count).
    /// The actual write target is the `hits` column of the SQLite index; JSONL cannot be updated in place.
    /// Silently return Ok when the index is unavailable, for backward-compatible behavior.
    pub fn record_usage(&self, entry_id: &str) -> Result<(), String> {
        if entry_id.is_empty() {
            return Ok(());
        }
        if let Some(idx) = memory_index_for(&self.path) {
            let _ = idx.record_hits(&[entry_id.to_string()]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod importance_tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_memory_importance_score() {
        let mut importance = MemoryImportance::new();
        assert_eq!(importance.frequency, 0);
        assert_eq!(importance.recency, 1.0);
        assert_eq!(importance.generality, 0.5);
        assert!(!importance.user_validated);

        // Initial score
        let initial_score = importance.score();
        assert!(initial_score > 0.0 && initial_score < 1.0);

        // Add a reference
        for _ in 0..10 {
            importance.increment_frequency();
        }
        assert_eq!(importance.frequency, 10);

        // User confirmation
        importance.mark_user_validated();
        assert!(importance.user_validated);

        // Score should increase
        let new_score = importance.score();
        assert!(new_score > initial_score);
    }

    #[test]
    fn test_memory_importance_recency_decay() {
        let mut importance = MemoryImportance::new();

        // Memory from 30 days ago
        let old_timestamp = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        importance.update_recency(&old_timestamp);

        // Recency should have decayed to about 0.5 (half-life)
        assert!(importance.recency > 0.4 && importance.recency < 0.6);

        // Memory from 90 days ago
        let very_old_timestamp = (Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        importance.update_recency(&very_old_timestamp);

        // Recency should be very low
        assert!(importance.recency < 0.2);
    }

    #[test]
    fn test_memory_importance_generality() {
        let mut importance = MemoryImportance::new();

        // General category
        importance.evaluate_generality("common_sense", &vec![]);
        assert!(importance.generality >= 0.7);

        // Specific category
        importance.evaluate_generality("user_specific", &vec![]);
        assert!(importance.generality <= 0.5);

        // With general tags
        importance.evaluate_generality(
            "user_specific",
            &vec!["general".to_string(), "core".to_string()],
        );
        assert!(importance.generality >= 0.5);
    }

    #[test]
    fn test_should_prune() {
        let mut importance = MemoryImportance::new();

        // High-value memories must not be pruned
        importance.frequency = 10;
        importance.user_validated = true;
        assert!(!importance.should_prune(0.3));

        // Low-value memories should be pruned
        let mut low_importance = MemoryImportance::new();
        low_importance.frequency = 0;
        low_importance.recency = 0.1;
        low_importance.generality = 0.2;
        assert!(low_importance.should_prune(0.3));
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rt_mem_retention_{tag}_{nanos}.jsonl"))
    }

    fn entry(category: &str, note: &str, ts: &str, priority: u8) -> AgentMemoryEntry {
        AgentMemoryEntry {
            id: None,
            timestamp: ts.to_string(),
            category: category.to_string(),
            note: note.to_string(),
            tags: vec![],
            source: None,
            priority: Some(priority),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }

    fn entry_with_id(
        id: &str,
        category: &str,
        note: &str,
        ts: &str,
        priority: u8,
    ) -> AgentMemoryEntry {
        let mut entry = entry(category, note, ts, priority);
        entry.id = Some(id.to_string());
        entry
    }

    fn write_lines(path: &Path, entries: &[AgentMemoryEntry]) {
        let mut buf = String::new();
        for e in entries {
            buf.push_str(&serde_json::to_string(e).unwrap());
            buf.push('\n');
        }
        std::fs::write(path, buf).unwrap();
    }

    fn read_entries(path: &Path) -> Vec<AgentMemoryEntry> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<AgentMemoryEntry>(l.trim()).ok())
            .collect()
    }

    /// P0-1 regression: the original double-delete bug wrongly removed the entry right after a priority=255 one once the quota was full;
    /// build a "low-priority + permanent" mix here and assert that after enforce all priority=255
    /// entries survive and the quota is pressed back under max_entries.
    #[test]
    fn prune_low_value_removes_matching_ids_in_one_pass() {
        let path = unique_path("prune_batch");
        let mut all = Vec::new();
        // Old low-priority entries: must be pruned.
        for i in 0..3 {
            all.push(entry_with_id(
                &format!("old-{i}"),
                "tool_stat",
                &format!("old-{i}"),
                "2025-01-01T00:00:00Z",
                50,
            ));
        }
        // Old but permanent: must survive regardless of age.
        all.push(entry_with_id(
            "perm",
            "safety_rules",
            "keep",
            "2025-01-01T00:00:00Z",
            255,
        ));
        // Low priority but too recent: must survive.
        all.push(entry_with_id(
            "fresh",
            "tool_stat",
            "fresh",
            "2099-01-01T00:00:00Z",
            50,
        ));
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        // tool_stat with no general tags scores 0.1 < 0.2; safety_rules and
        // priority >= 200 are exempt, so exactly the three old-* entries go.
        let removed = store.prune_low_value_memories(0.2, 365).unwrap();
        assert_eq!(removed, 3);

        let kept = read_entries(&path);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|e| e.id.as_deref() == Some("perm")));
        assert!(kept.iter().any(|e| e.id.as_deref() == Some("fresh")));
        assert!(
            kept.iter()
                .all(|e| !e.id.as_deref().unwrap_or("").starts_with("old-"))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn enforce_max_entries_keeps_all_permanent_entries() {
        let path = unique_path("enforce_perm");
        let mut all = Vec::new();
        // 100 ordinary low-priority entries, ordered old to new
        for i in 0..100 {
            all.push(entry(
                "tool_stat",
                &format!("note-{i}"),
                &format!("2025-01-01T00:00:{:02}Z", i % 60),
                50,
            ));
        }
        // 5 permanent entries (safety_rules)
        for i in 0..5 {
            all.push(entry(
                "safety_rules",
                &format!("perm-{i}"),
                &format!("2025-02-01T00:00:{:02}Z", i),
                255,
            ));
        }
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        store.enforce_max_entries(50, 10).unwrap();

        let kept = read_entries(&path);
        assert!(
            kept.len() <= 50,
            "expected <=50 entries, got {}",
            kept.len()
        );
        let perm_kept = kept.iter().filter(|e| e.priority == Some(255)).count();
        assert_eq!(perm_kept, 5, "all permanent entries must survive");

        let _ = std::fs::remove_file(&path);
    }

    /// P0-2 regression: after rotate the new main file may contain only priority=255 entries,
    /// while the archive file contains all original entries.
    #[test]
    fn rotate_preserves_permanent_entries_in_main_file() {
        let path = unique_path("rotate_perm");
        let mut all = Vec::new();
        // Inflate the file to a few KB with a large enough note
        let big = "x".repeat(2048);
        for i in 0..20 {
            all.push(entry(
                "tool_cache",
                &format!("{}-{}", big, i),
                &format!("2025-01-01T00:00:{:02}Z", i),
                80,
            ));
        }
        all.push(entry(
            "safety_rules",
            "do not run rm -rf /",
            "2025-02-02T00:00:00Z",
            255,
        ));
        all.push(entry(
            "self_note",
            "always read before edit",
            "2025-02-02T00:00:01Z",
            255,
        ));
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        // Threshold deliberately smaller than the current size to force a rotate
        let rotated = store.rotate_if_exceeds(1024).unwrap();
        assert!(rotated, "expected rotate to happen");

        // Main file keeps only permanent entries
        let head = read_entries(&path);
        assert_eq!(head.len(), 2);
        assert!(head.iter().all(|e| e.priority == Some(255)));

        // Archive file exists and contains all original entries
        let parent = path.parent().unwrap();
        let archives: Vec<_> = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let head_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with(head_name) && p != &path
            })
            .collect();
        assert_eq!(archives.len(), 1, "expected exactly one archive");
        let archived = read_entries(&archives[0]);
        assert_eq!(archived.len(), all.len());

        let _ = std::fs::remove_file(&path);
        for a in archives {
            let _ = std::fs::remove_file(a);
        }
    }

    /// P0 regression: -ns memo retrieval (include_archives=true) must scan all archives,
    /// not truncated by keep_last_archives; it must also handle `.jsonl.bak` files left by legacy migration.
    /// Otherwise historical memos moved out by rotation or migration become permanently unretrievable.
    #[test]
    fn entries_by_category_include_archives_scans_all_archives() {
        let path = unique_path("memo_arch_scan");
        let parent = path.parent().unwrap();
        std::fs::create_dir_all(parent).unwrap();
        let base = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();
        let legacy_base = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();

        // Main file: 1 memo
        write_lines(
            &path,
            &[entry("memo", "main memo", "2026-07-16T00:00:00Z", 100)],
        );

        // Create 5 archive files (beyond the default keep_last_archives=3 window).
        // Put "二次分析" in the oldest archive, simulating a user record moved out by rotation.
        let archive_notes = [
            "oldest: 二次分析问题排查",
            "archive2 memo",
            "archive3 memo",
            "archive4 memo",
            "archive5 memo",
        ];
        let mut archive_paths = Vec::new();
        for (i, note) in archive_notes.iter().enumerate() {
            let ap = parent.join(format!("{base}.2026070{}170000", i + 1));
            write_lines(&ap, &[entry("memo", note, "2026-07-01T00:00:00Z", 100)]);
            // Set increasing mtimes so ordering is deterministic (oldest -> newest)
            let times = std::fs::FileTimes::new();
            let _ = times.set_modified(
                UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000 + i as u64),
            );
            std::fs::File::open(&ap).unwrap().set_times(times).unwrap();
            archive_paths.push(ap);
        }

        // Older versions left pre-migration data in this naming format; it is not an ordinary rotation archive.
        let legacy_path = parent.join(format!(
            "{legacy_base}.legacy-migrate-20260701180745.jsonl.bak"
        ));
        write_lines(
            &legacy_path,
            &[entry(
                "memo",
                "legacy: 二次分析问题排查 mysql",
                "2026-06-30T00:00:00Z",
                100,
            )],
        );

        let store = MemoryStore::for_tests_with_path(path.clone());
        let memos = store.entries_by_category("memo", 100_000, true).unwrap();

        // Main file 1 + normal archives 5 + legacy migration backup 1 = 7 entries.
        assert_eq!(
            memos.len(),
            7,
            "include_archives=true 必须扫描全部归档及旧迁移备份，不应被 keep_last 截断"
        );
        assert!(
            memos.iter().any(|m| m.note.contains("二次分析")),
            "最旧归档和旧迁移备份中的 memo 都必须可检索"
        );

        // Cleanup
        let _ = std::fs::remove_file(&path);
        for p in archive_paths {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(legacy_path);
    }

    #[test]
    fn entries_by_category_current_file_ignores_global_archive_search() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let cfg_path = unique_path("current_only_config");
        std::fs::write(
            &cfg_path,
            "ai.memory.search_archives.enable = true\nai.memory.search_archives.keep_last = 10\n",
        )
        .unwrap();
        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
        crate::commonw::configw::refresh();

        let path = unique_path("current_only");
        let parent = path.parent().unwrap();
        std::fs::create_dir_all(parent).unwrap();
        let base = path.file_name().and_then(|n| n.to_str()).unwrap();
        let archive_path = parent.join(format!("{base}.20260729120000"));

        write_lines(
            &path,
            &[entry("memo", "main memo", "2026-07-29T00:00:00Z", 100)],
        );
        write_lines(
            &archive_path,
            &[entry("memo", "archive memo", "2026-07-28T00:00:00Z", 100)],
        );

        let store = MemoryStore::for_tests_with_path(path.clone());
        let configured_scan = store.entries_by_category("memo", 10, false).unwrap();
        let current_only = store.entries_by_category_current_file("memo", 10).unwrap();

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();

        let _ = std::fs::remove_file(cfg_path);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(archive_path);

        assert_eq!(configured_scan.len(), 2);
        assert_eq!(current_only.len(), 1);
        assert_eq!(current_only[0].note, "main memo");
    }
}
