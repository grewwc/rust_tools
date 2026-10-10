use super::*;
use std::ffi::OsStr;
use std::time::{Duration, SystemTime};
use crate::ai::tools::service::memory::{execute_memory_dedup, execute_memory_gc};
use crate::commonw::configw;
use serde_json::json;
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

    pub(super) fn enforce_max_entries(&self, max_entries: usize, min_keep: usize) -> Result<(), String> {
        super::super::with_memory_file_lock(&self.path, || {
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
        super::super::with_memory_file_lock(&self.path, || {
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
