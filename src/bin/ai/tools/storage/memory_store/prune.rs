use super::*;
use rustc_hash::FxHashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
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
            super::super::with_memory_file_lock(&self.path, || {
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
    pub(super) fn write_all_entries(path: &Path, entries: &[AgentMemoryEntry]) -> Result<(), String> {
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
