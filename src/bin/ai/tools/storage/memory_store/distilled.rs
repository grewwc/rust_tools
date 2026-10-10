use super::*;
use rustc_hash::FxHashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
impl MemoryStore {
    pub(crate) fn from_env_or_config() -> Self {
        Self {
            path: resolve_memory_file(),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Commit one verified revision after comparing its canonical ID and revision.
    /// Archived entries are never candidates for updates or duplicate detection.
    #[cfg(test)]
    pub(crate) fn upsert_distilled(
        &self,
        entry: AgentMemoryEntry,
        expected: Option<(String, u32)>,
    ) -> Result<DistilledUpsertReport, String> {
        self.upsert_distilled_batch(vec![(entry, expected)], false)?
            .pop()
            .ok_or_else(|| "Missing distilled batch result".to_string())
    }

    /// Validate the complete batch against one canonical snapshot, then commit once.
    /// Preview uses the same staging logic, without creating or writing any store files.
    pub(crate) fn upsert_distilled_batch(
        &self,
        pending: Vec<(AgentMemoryEntry, Option<(String, u32)>)>,
        dry_run: bool,
    ) -> Result<Vec<DistilledUpsertReport>, String> {
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        if !dry_run {
            self.ensure_memory_file_for_lock()?;
        }
        super::super::with_memory_file_lock(&self.path, || {
            let mut entries = self.current_entries_while_locked()?;
            let mut reports = Vec::with_capacity(pending.len());
            let mut touched_ids = FxHashSet::default();
            for (entry, expected) in pending {
                let report = Self::stage_distilled_upsert(&mut entries, entry, expected)?;
                if !touched_ids.insert(report.entry_id.clone()) {
                    return Err("Distilled batch contains a repeated canonical ID".to_string());
                }
                reports.push(report);
            }
            if !dry_run {
                let delete_ids: Vec<_> = reports.iter()
                    .filter(|report| report.updated)
                    .map(|report| report.entry_id.as_str())
                    .collect();
                let changed_ids: FxHashSet<_> = reports.iter()
                    .filter(|report| !report.duplicate)
                    .map(|report| report.entry_id.as_str())
                    .collect();
                let new_entries: Vec<_> = entries.into_iter()
                    .filter(|entry| entry.id.as_deref().is_some_and(|id| changed_ids.contains(id)))
                    .collect();
                if !new_entries.is_empty() {
                    self.apply_batch_update_while_locked(&delete_ids, &new_entries, false)?;
                }
            }
            Ok(reports)
        })
    }

    /// Stage an upsert in memory only; callers persist only after every item passes.
    fn stage_distilled_upsert(
        entries: &mut Vec<AgentMemoryEntry>,
        mut entry: AgentMemoryEntry,
        expected: Option<(String, u32)>,
    ) -> Result<DistilledUpsertReport, String> {
        use crate::ai::knowledge::distilled::{DistilledRevision, active_distilled_metadata};

        let mut metadata = active_distilled_metadata(&entry)
            .ok_or_else(|| "Invalid distilled metadata or content digest".to_string())?;
        let viewer = crate::ai::tools::service::memory::ViewerContext::current();
        if !viewer.can_see(&entry) {
            return Err("Distilled entry is outside the current owner scope".to_string());
        }
        {
            let same_key = |existing: &&AgentMemoryEntry| {
                existing.owner_pid == entry.owner_pid
                    && existing.owner_pgid == entry.owner_pgid
                    && existing.distilled.as_ref().is_some_and(|old| {
                        old.scope == metadata.scope && old.topic_key == metadata.topic_key
                    })
            };
            let collisions: Vec<_> = entries.iter().filter(same_key).collect();
            let previous = if let Some((id, revision)) = &expected {
                let matching: Vec<_> = entries
                    .iter()
                    .filter(|existing| existing.id.as_deref() == Some(id.as_str()))
                    .collect();
                if matching.len() != 1 {
                    return Err("Stale or ambiguous distilled entry ID".to_string());
                }
                let existing = matching[0];
                let old = active_distilled_metadata(existing)
                    .ok_or_else(|| "Expected entry is not active distilled memory".to_string())?;
                if old.revision != *revision {
                    return Err("Stale distilled revision".to_string());
                }
                if !viewer.can_see(existing) || !same_key(&existing) {
                    return Err("Distilled scope, topic or owner cannot change".to_string());
                }
                if collisions.len() != 1 {
                    return Err("Distilled scope/topic/owner collision".to_string());
                }
                Some((existing, old))
            } else {
                if collisions.len() > 1 {
                    return Err("Distilled scope/topic/owner collision".to_string());
                }
                collisions.first().and_then(|existing| {
                    active_distilled_metadata(existing).map(|old| (*existing, old))
                })
            };
            if let Some((existing, old)) = &previous {
                // A consumed archive is not new evidence. Re-running extraction
                // must never roll back a newer revision, even with fresh CAS data.
                if old.content_digest != metadata.content_digest
                    && metadata.source_digests.iter().all(|digest| old.source_digests.contains(digest))
                {
                    return Err("Previously consumed source cannot replace current distilled knowledge".to_string());
                }
                if old.content_digest == metadata.content_digest
                    && metadata.source_digests.iter().all(|digest| old.source_digests.contains(digest))
                {
                    let entry_id = existing.id.as_ref().filter(|id| !id.is_empty())
                        .ok_or_else(|| "Distilled entry is missing its canonical ID".to_string())?;
                    return Ok(DistilledUpsertReport {
                        inserted: false, updated: false, duplicate: true, entry_id: entry_id.clone(),
                    });
                }
            }
            if expected.is_none() && !collisions.is_empty() {
                return Err("Distilled topic already exists; an expected revision is required".to_string());
            }
            let entry_id = if let Some((existing, old)) = previous {
                metadata.revision = old.revision.checked_add(1)
                    .ok_or_else(|| "Distilled revision overflow".to_string())?;
                metadata.previous_revisions = old.previous_revisions;
                metadata.previous_revisions.push(DistilledRevision {
                    revision: old.revision,
                    note: existing.note.clone(),
                    content_digest: old.content_digest,
                    evidence: old.evidence,
                });
                for digest in old.source_digests {
                    if !metadata.source_digests.contains(&digest) {
                        metadata.source_digests.push(digest);
                    }
                }
                expected.as_ref().expect("updates require an expected revision").0.clone()
            } else {
                metadata.revision = 1;
                metadata.previous_revisions.clear();
                let id = entry.id.clone().filter(|id| !id.trim().is_empty())
                    .unwrap_or_else(crate::ai::tools::service::memory::next_memory_id);
                if entries.iter().any(|existing| existing.id.as_ref() == Some(&id)) {
                    return Err("Distilled entry ID collides with an existing entry".to_string());
                }
                id
            };
            entry.id = Some(entry_id.clone());
            entry.distilled = Some(metadata);
            if expected.is_some() {
                entries.retain(|existing| existing.id.as_ref() != Some(&entry_id));
            }
            entries.push(entry);
            Ok(DistilledUpsertReport {
                inserted: expected.is_none(), updated: expected.is_some(), duplicate: false, entry_id,
            })
        }
    }

    /// Read verified canonical rows using the same owner visibility as explicit memory tools.
    pub(crate) fn active_distilled_entries(&self, scope: &str) -> Result<Vec<AgentMemoryEntry>, String> {
        let viewer = crate::ai::tools::service::memory::ViewerContext::current();
        super::super::with_memory_file_lock(&self.path, || {
            Ok(self.current_entries_while_locked()?.into_iter().filter(|entry| {
                viewer.can_see(entry)
                    && crate::ai::knowledge::distilled::active_distilled_metadata(entry)
                        .is_some_and(|metadata| metadata.scope == scope)
            }).collect())
        })
    }

    /// Fail closed on malformed rows so revision commits cannot discard unrelated data.
    pub(super) fn current_entries_while_locked(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        let content = match fs::read_to_string(&self.path) {
            Ok(content) => content,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(format!("Failed to read canonical memory: {err}")),
        };
        content.lines().enumerate().filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| serde_json::from_str(line)
                .map_err(|err| format!("Invalid canonical memory row {}: {err}", index + 1)))
            .collect()
    }

    pub(crate) fn append(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        let entry = cap_memory_entry(entry);
        super::super::with_memory_file_lock(&self.path, || {
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
        super::super::with_memory_file_lock(&self.path, || {
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
}
