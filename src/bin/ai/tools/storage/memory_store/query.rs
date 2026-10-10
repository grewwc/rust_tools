use super::*;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::SystemTime;
use crate::ai::knowledge::indexing::similarity;
use crate::commonw::configw;
impl MemoryStore {

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
        super::super::with_memory_file_lock(&self.path, || {
            self.apply_batch_update_while_locked(delete_ids, new_entries, true)
        })
    }

    /// The caller holds the canonical memory lock across validation and commit.
    pub(super) fn apply_batch_update_while_locked(
        &self,
        delete_ids: &[&str],
        new_entries: &[AgentMemoryEntry],
        include_archives: bool,
    ) -> Result<MemoryBatchUpdateReport, String> {
            let id_set: FxHashSet<&str> = delete_ids.iter().copied().collect();
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("Failed to create memory dir: {err}"))?;
            }
            let mut paths = if id_set.is_empty() || !include_archives {
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
    }

    pub(super) fn memory_files_to_scan(&self, include_archives: bool) -> Result<Vec<PathBuf>, String> {
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
    pub(super) fn memory_files_to_scan_consolidate(&self) -> Result<Vec<PathBuf>, String> {
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
