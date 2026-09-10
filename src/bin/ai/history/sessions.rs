#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
    fs::File,
    fs::{self},
    io::{self, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

use chrono::{DateTime, Local, Utc};
use rust_tools::commonw::FastMap;
use rust_tools::cw::SkipMap;
use serde_json::json;

use super::{
    blob::{delete_assets_dir, delete_history_artifacts},
    markdown::messages_to_markdown,
    sqlite::{
        SessionListMetadata, backup_sqlite, read_all_messages_sqlite,
        read_first_user_prompt_sqlite, read_session_list_metadata_sqlite,
        read_session_mark_message_sqlite, read_session_marked_sqlite,
        read_session_title_origin_sqlite, read_session_title_sqlite,
        remap_context_checkpoint_paths_sqlite, with_session_state_lock,
        write_session_mark_sqlite, write_session_title_sqlite, MarkMessageUpdate,
    },
    types::Message,
};

const MAX_SESSION_ID_BYTES: usize = 128;
const SESSIONS_LIFECYCLE_LOCK: &str = ".sessions-lifecycle.lock";
const SESSION_SIZE_CACHE_FILE: &str = ".sizes-cache.json";

pub(in crate::ai) fn with_sessions_lifecycle_lock<T>(
    sessions_root: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    fs::create_dir_all(sessions_root)?;
    let lock_path = sessions_root.join(SESSIONS_LIFECYCLE_LOCK);
    let lock_file = File::options()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    #[cfg(unix)]
    unsafe {
        if libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN);
    }
    result
}

/// Recursively copy the directory tree `src` -> `dst` (`dst` must not exist
/// yet). Used by fork_session to copy the assets directory completely:
/// checkpoint bodies live in nested directories, and a shallow copy would
/// leave forked markers pointing at missing files.
pub(super) fn copy_dir_recursively(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursively(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionInfo {
    pub(in crate::ai) id: String,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) history_revision: i64,
    pub(in crate::ai) size_bytes: u64,
    pub(in crate::ai) first_user_prompt: Option<String>,
    pub(in crate::ai) summary: Option<String>,
    /// Whether the user marked this session as important via `/mark`.
    pub(in crate::ai) marked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai) enum PruneSessionDeleteResult {
    Deleted,
    Missing,
    Changed,
    NotExpired,
    Active,
}

/// The persisted origin of a title. Old databases have no origin marker and
/// must be treated conservatively so existing good titles are not overwritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai) enum SessionTitleOrigin {
    Model,
    Fallback,
    Legacy,
}

impl SessionTitleOrigin {
    fn from_persisted(value: Option<&str>) -> Self {
        match value {
            Some("model") => Self::Model,
            Some("fallback") => Self::Fallback,
            _ => Self::Legacy,
        }
    }

    fn persisted_value(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Fallback => "fallback",
            // Legacy is only for reading old data; new writes must explicitly
            // mark the real origin.
            Self::Legacy => "legacy",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct SessionTitle {
    pub(in crate::ai) text: String,
    pub(in crate::ai) origin: SessionTitleOrigin,
}

impl SessionStore {
    pub(in crate::ai) fn new(history_file: &Path) -> Self {
        Self {
            root: sessions_root_from_history_file(history_file),
        }
    }

    pub(in crate::ai) fn ensure_root_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub(in crate::ai) fn sessions_root(&self) -> &Path {
        &self.root
    }

    /// The session ID becomes part of persisted paths; callers must reject
    /// invalid input first and never silently rewrite it.
    pub(in crate::ai) fn validate_session_id(session_id: &str) -> io::Result<()> {
        if session_id.is_empty()
            || session_id.len() > MAX_SESSION_ID_BYTES
            || !session_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "session id must contain 1-128 ASCII letters, digits, '-' or '_'",
            ));
        }
        Ok(())
    }

    pub(in crate::ai) fn session_exists(&self, session_id: &str) -> io::Result<bool> {
        Self::validate_session_id(session_id)?;
        Ok(self.session_history_file(session_id).is_file())
    }

    pub(in crate::ai) fn session_history_file(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.sqlite"))
    }

    pub(in crate::ai) fn session_assets_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.assets"))
    }

    /// This session's checkpoint directory:
    /// `<sessions_root>/checkpoints/<id>/`.
    pub(in crate::ai) fn checkpoints_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join("checkpoints").join(id)
    }

    pub(in crate::ai) fn list_sessions(&self) -> io::Result<Vec<SessionInfo>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        // First filter valid sessions on this thread (file-name check + stat),
        // then dispatch each database's metadata read to parallel threads:
        // opening dozens or hundreds of sqlite files serially is one of `/ss`'s
        // main costs.
        let mut jobs: Vec<(String, PathBuf, Option<SystemTime>)> = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let id = stem.to_string();
            // Ignore invalid file names left by old versions or external
            // writers so they cannot re-enter the selectable session set.
            if Self::validate_session_id(&id).is_err() {
                continue;
            }
            let file_modified = match entry.metadata() {
                Ok(v) => v.modified().ok(),
                Err(_) => continue,
            };
            jobs.push((id, path, file_modified));
        }
        // Parallel reads are gathered back on the main thread by index and
        // inserted into the SkipMap in original order, keeping the same
        // ordering as the serial version.
        let mut metadata_results = Self::read_session_list_metadata_parallel(&jobs);

        let mut sessions: Box<SkipMap<(u64, String), SessionInfo>> =
            SkipMap::new(16, |a: &(u64, String), b: &(u64, String)| {
                match b.0.cmp(&a.0) {
                    std::cmp::Ordering::Equal => a.1.cmp(&b.1) as i32 * -1,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Greater => -1,
                }
            });
        for (idx, (id, _path, file_modified)) in jobs.into_iter().enumerate() {
            let (
                first_user_prompt,
                generated_title,
                last_activity_unix_ms,
                history_revision,
                marked,
            ) = match metadata_results[idx].take() {
                Some(metadata) => (
                    metadata.first_user_prompt,
                    metadata.session_title,
                    metadata.last_activity_unix_ms,
                    metadata.history_revision,
                    metadata.marked,
                ),
                None => (None, None, None, 0, false),
            };
            // New databases use the logical activity time maintained inside
            // transactions; for old databases the metadata reader falls back
            // to the last canonical message's created_at. The main DB mtime
            // is only used when no logical time is readable. -shm/-wal
            // mtimes must not be used: read-only connections and SQLite
            // housekeeping can also refresh them.
            let modified_local = last_activity_unix_ms
                .and_then(DateTime::<Utc>::from_timestamp_millis)
                .map(|time| time.with_timezone(&Local))
                .or_else(|| file_modified.map(DateTime::<Local>::from));
            // Prefer the LLM-generated title (stored in the meta table),
            // falling back to the first message's summary.
            let summary = generated_title
                .as_deref()
                .map(normalize_generated_session_title)
                .filter(|title| !title.is_empty())
                .or_else(|| {
                    first_user_prompt
                        .as_deref()
                        .map(generate_session_summary)
                        .map(|summary| normalize_generated_session_title(&summary))
                        .filter(|summary| !summary.is_empty())
                });
            let timestamp = modified_local
                .map(|dt| dt.timestamp_millis() as u64)
                .unwrap_or(0);
            sessions.insert(
                (timestamp, id.clone()),
                SessionInfo {
                    id,
                    modified_local,
                    history_revision,
                    // Sizes are computed on demand by `attach_session_sizes`
                    // / `session_total_size`; list_sessions no longer walks
                    // every session's assets directory.
                    size_bytes: 0,
                    first_user_prompt,
                    summary,
                    marked,
                },
            );
        }
        Ok(sessions.into_iter().map(|(_, v)| v).collect())
    }

    /// Read list metadata (title, first request, activity time) of many
    /// session databases in parallel.
    ///
    /// Keeps `list_sessions`'s original fault-tolerance semantics: a failed
    /// read of one database yields `None` and the caller falls back to the
    /// file mtime; one corrupt or old-format session never blocks the whole
    /// list. Static striding avoids atomic-counter contention, and
    /// `std::thread::scope` lets threads borrow `jobs` without copying.
    fn read_session_list_metadata_parallel(
        jobs: &[(String, PathBuf, Option<SystemTime>)],
    ) -> Vec<Option<SessionListMetadata>> {
        let count = jobs.len();
        let mut results: Vec<Option<SessionListMetadata>> = Vec::new();
        results.resize_with(count, || None);
        if count == 0 {
            return results;
        }
        const MAX_WORKERS: usize = 16;
        let workers = count.min(MAX_WORKERS);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    scope.spawn(move || {
                        let mut chunk: Vec<(usize, Option<SessionListMetadata>)> = Vec::new();
                        let mut idx = worker;
                        while idx < count {
                            let (_, path, _) = &jobs[idx];
                            chunk.push((idx, read_session_list_metadata_sqlite(path).ok()));
                            idx += workers;
                        }
                        chunk
                    })
                })
                .collect();
            for handle in handles {
                for (idx, metadata) in handle.join().unwrap_or_default() {
                    results[idx] = metadata;
                }
            }
        });
        results
    }

    /// Read the recovery preview of a single session only, so startup
    /// recovery does not scan and stat every session.
    pub(in crate::ai) fn read_session_preview(
        &self,
        session_id: &str,
    ) -> io::Result<Option<(Option<String>, Option<DateTime<Local>>)>> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }

        let file_modified = path
            .metadata()
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        let (first_user_prompt, generated_title, last_activity_unix_ms) =
            match read_session_list_metadata_sqlite(&path) {
                Ok(metadata) => (
                    metadata.first_user_prompt,
                    metadata.session_title,
                    metadata.last_activity_unix_ms,
                ),
                Err(_) => (None, None, None),
            };
        let modified_local = last_activity_unix_ms
            .and_then(DateTime::<Utc>::from_timestamp_millis)
            .map(|time| time.with_timezone(&Local))
            .or_else(|| file_modified.map(DateTime::<Local>::from));
        let summary = generated_title
            .as_deref()
            .map(normalize_generated_session_title)
            .filter(|title| !title.is_empty())
            .or_else(|| {
                first_user_prompt
                    .as_deref()
                    .map(generate_session_summary)
                    .map(|summary| normalize_generated_session_title(&summary))
                    .filter(|summary| !summary.is_empty())
            });

        Ok(Some((summary, modified_local)))
    }

    fn session_size_bytes(
        &self,
        sqlite_path: &Path,
        session_id: &str,
        derived_history_size: u64,
    ) -> io::Result<u64> {
        let mut total = file_size_if_exists(sqlite_path)?;
        for suffix in ["-wal", "-shm", "-journal"] {
            total = total.saturating_add(file_size_if_exists(&PathBuf::from(format!(
                "{}{}",
                sqlite_path.display(),
                suffix
            )))?);
        }
        total = total.saturating_add(derived_history_size);
        total = total.saturating_add(directory_size(&self.session_assets_dir(session_id))?);
        total = total.saturating_add(directory_size(&self.checkpoints_dir(session_id))?);
        Ok(total)
    }

    /// Compute the total bytes of a single session, for commands like
    /// `/ss current` that only show one session's size, without scanning and
    /// stat-ing all sessions.
    pub(in crate::ai) fn session_total_size(&self, session_id: &str) -> io::Result<u64> {
        let derived_history_size = *self
            .derived_session_history_artifact_sizes()?
            .get(session_id)
            .unwrap_or(&0);
        self.session_size_bytes(
            &self.session_history_file(session_id),
            session_id,
            derived_history_size,
        )
    }

    /// Fill in `size_bytes` for the listed sessions in parallel.
    ///
    /// `list_sessions` no longer computes sizes for performance: recursively
    /// stat-ing every session's assets is the main bottleneck (~2.6s on this
    /// machine). Only commands that display sizes (e.g. `/ss list`) call this
    /// method, dispatching each session's stat to its own thread and using
    /// all cores to bring wall-clock time down to a few hundred milliseconds.
    /// Results are cached in `<root>/.sizes-cache.json` keyed by a top-level
    /// directory fingerprint: the fingerprint reads only the direct children
    /// of assets / checkpoints (name, type, length, mtime), so added,
    /// deleted, or rewritten files all show up in it; when the fingerprint is
    /// unchanged the cache is reused and later `/ss` calls do not re-walk
    /// multi-hundred-MB overflow directories.
    pub(in crate::ai) fn attach_session_sizes(
        &self,
        sessions: &mut [SessionInfo],
    ) -> io::Result<()> {
        if sessions.is_empty() {
            return Ok(());
        }
        let derived_history_sizes = self.derived_session_history_artifact_sizes()?;
        let cache_path = self.root.join(SESSION_SIZE_CACHE_FILE);
        let mut cache = Self::load_session_size_cache(&cache_path);
        // Pre-collect (index, session_id, derived size); threads only borrow
        // self, not sessions.
        let jobs: Vec<(usize, String, u64)> = sessions
            .iter()
            .enumerate()
            .map(|(idx, session)| {
                (
                    idx,
                    session.id.clone(),
                    *derived_history_sizes.get(&session.id).unwrap_or(&0),
                )
            })
            .collect();
        let (sizes, recomputed): (Vec<(usize, u64)>, Vec<(String, String, u64, u64)>) =
            std::thread::scope(|scope| {
                let handles: Vec<_> = jobs
                    .iter()
                    .map(|(idx, id, derived_history_size)| {
                        let (idx, id, derived_history_size) =
                            (*idx, id.clone(), *derived_history_size);
                        let cache = &cache;
                        scope.spawn(move || {
                            let assets_dir = self.session_assets_dir(&id);
                            let checkpoints_dir = self.checkpoints_dir(&id);
                            // Skip the recursive walk when the top-level
                            // fingerprint hits the cache; on walk failure
                            // record 0 and do not write the cache, as before.
                            let (assets_size, checkpoints_size, recompute) = match (
                                Self::dir_two_level_fingerprint(&assets_dir),
                                Self::dir_two_level_fingerprint(&checkpoints_dir),
                            ) {
                                (Ok(assets_fp), Ok(checkpoints_fp)) => {
                                    let fingerprint = format!("{assets_fp}\u{1f}{checkpoints_fp}");
                                    match cache.get(&id) {
                                        Some((cached_fp, cached_assets, cached_checkpoints))
                                            if *cached_fp == fingerprint =>
                                        {
                                            (*cached_assets, *cached_checkpoints, None)
                                        }
                                        _ => {
                                            match directory_size(&assets_dir).and_then(|assets| {
                                                directory_size(&checkpoints_dir)
                                                    .map(|checkpoints| (assets, checkpoints))
                                            }) {
                                                Ok((assets, checkpoints)) => (
                                                    assets,
                                                    checkpoints,
                                                    Some((
                                                        id.clone(),
                                                        fingerprint,
                                                        assets,
                                                        checkpoints,
                                                    )),
                                                ),
                                                Err(_) => (0, 0, None),
                                            }
                                        }
                                    }
                                }
                                _ => (0, 0, None),
                            };
                            let sqlite_path = self.session_history_file(&id);
                            let mut total = file_size_if_exists(&sqlite_path)
                                .unwrap_or(0)
                                .saturating_add(derived_history_size);
                            for suffix in ["-wal", "-shm", "-journal"] {
                                total = total.saturating_add(
                                    file_size_if_exists(&PathBuf::from(format!(
                                        "{}{}",
                                        sqlite_path.display(),
                                        suffix
                                    )))
                                    .unwrap_or(0),
                                );
                            }
                            total = total
                                .saturating_add(assets_size)
                                .saturating_add(checkpoints_size);
                            ((idx, total), recompute)
                        })
                    })
                    .collect();
                let mut sizes = Vec::with_capacity(jobs.len());
                let mut recomputed = Vec::new();
                for handle in handles {
                    if let Ok((size, recompute)) = handle.join() {
                        sizes.push(size);
                        if let Some(entry) = recompute {
                            recomputed.push(entry);
                        }
                    }
                }
                (sizes, recomputed)
            });
        for (idx, size) in sizes {
            if let Some(session) = sessions.get_mut(idx) {
                session.size_bytes = size;
            }
        }
        if !recomputed.is_empty() {
            for (id, fingerprint, assets_size, checkpoints_size) in recomputed {
                cache.insert(id, (fingerprint, assets_size, checkpoints_size));
            }
            // Also drop leftover cache entries of deleted sessions so the
            // cache file cannot grow without bound.
            let current_ids: std::collections::HashSet<&str> =
                sessions.iter().map(|session| session.id.as_str()).collect();
            cache.retain(|id, _| current_ids.contains(id.as_str()));
            Self::save_session_size_cache(&cache_path, &cache)?;
        }
        Ok(())
    }

    /// Two-level directory fingerprint: reads direct children (name, type,
    /// length, mtime) and, for each direct subdirectory, its direct children,
    /// to detect tree changes. Adding/deleting/rewriting changes either the
    /// file's own size/mtime or its directory's mtime, which changes the
    /// fingerprint; a missing directory yields `Ok("-")`. Compared with a
    /// recursive walk, the fingerprint reads are proportional to "top-level
    /// entry count + direct-subdirectory entry count".
    /// Known edge: rewriting a file at depth >= 3 changes no ancestor
    /// directory mtime, which the two-level fingerprint cannot see; this
    /// repository's actual layout (assets two levels deep, checkpoints
    /// published atomically via rename) is unaffected, and any add/delete
    /// still heals the cache through directory mtime signals.
    fn dir_two_level_fingerprint(path: &Path) -> io::Result<String> {
        let entries = match fs::read_dir(path) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok("-".to_string()),
            Err(error) => return Err(error),
        };
        let mut parts: Vec<String> = Vec::new();
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let (kind, size, mtime_ns) = if file_type.is_dir() {
                // Subdirectories do not count toward the byte total, but
                // their mtime changes with content add/delete and serves as a
                // change signal.
                let mtime = fs::metadata(&entry.path())
                    .ok()
                    .and_then(|m| Self::metadata_mtime_nanos(&m))
                    .unwrap_or(0);
                ("d", 0u64, mtime)
            } else if file_type.is_file() {
                let metadata = entry.metadata()?;
                (
                    "f",
                    metadata.len(),
                    Self::metadata_mtime_nanos(&metadata).unwrap_or(0),
                )
            } else {
                // Symlinks etc. are neither recursed into nor counted,
                // matching directory_size's semantics.
                ("o", 0u64, 0)
            };
            parts.push(format!("{kind}|{name}|{size}|{mtime_ns}"));
            // Second level: rewrites (size/mtime changes) and add/delete of
            // files inside direct subdirectories are detected too. Entries
            // carry the parent directory name as a prefix so same-named
            // children of different directories cannot be confused; a failed
            // second-level read does not block the whole fingerprint.
            if file_type.is_dir() {
                let Ok(sub_entries) = fs::read_dir(entry.path()) else {
                    continue;
                };
                for sub in sub_entries {
                    let Ok(sub) = sub else { continue };
                    let Ok(sub_type) = sub.file_type() else {
                        continue;
                    };
                    let sub_name = sub.file_name().to_string_lossy().into_owned();
                    let (sub_kind, sub_size, sub_mtime) = if sub_type.is_dir() {
                        let mtime = fs::metadata(sub.path())
                            .ok()
                            .and_then(|m| Self::metadata_mtime_nanos(&m))
                            .unwrap_or(0);
                        ("d", 0u64, mtime)
                    } else if sub_type.is_file() {
                        let Ok(metadata) = sub.metadata() else {
                            continue;
                        };
                        (
                            "f",
                            metadata.len(),
                            Self::metadata_mtime_nanos(&metadata).unwrap_or(0),
                        )
                    } else {
                        ("o", 0u64, 0)
                    };
                    parts.push(format!(
                        "{name}/{sub_kind}|{sub_name}|{sub_size}|{sub_mtime}"
                    ));
                }
            }
        }
        parts.sort();
        Ok(parts.join(";"))
    }

    fn metadata_mtime_nanos(metadata: &fs::Metadata) -> Option<u128> {
        metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
    }

    /// Persisted format of the session size cache:
    /// `Vec<(session_id, fingerprint, assets_size, checkpoints_size)>`.
    /// Serializes the tuple sequence directly to avoid a serde derive for the
    /// cache structure; a missing or corrupt file invalidates the whole cache
    /// and triggers recomputation.
    fn load_session_size_cache(path: &Path) -> FastMap<String, (String, u64, u64)> {
        let Ok(bytes) = fs::read(path) else {
            return FastMap::default();
        };
        let Ok(entries) = serde_json::from_slice::<Vec<(String, String, u64, u64)>>(&bytes) else {
            return FastMap::default();
        };
        entries
            .into_iter()
            .map(|(id, fingerprint, assets_size, checkpoints_size)| {
                (id, (fingerprint, assets_size, checkpoints_size))
            })
            .collect()
    }

    fn save_session_size_cache(
        path: &Path,
        cache: &FastMap<String, (String, u64, u64)>,
    ) -> io::Result<()> {
        let entries: Vec<(String, String, u64, u64)> = cache
            .iter()
            .map(|(id, (fingerprint, assets_size, checkpoints_size))| {
                (
                    id.clone(),
                    fingerprint.clone(),
                    *assets_size,
                    *checkpoints_size,
                )
            })
            .collect();
        let payload =
            serde_json::to_vec(&entries).map_err(|error| io::Error::other(error.to_string()))?;
        // Atomic publish via temporary file + rename, same as sqlite backup
        // publishing; concurrent /ss calls each write their own temporary
        // file.
        let temporary = path.with_file_name(format!(
            ".{}.tmp-{}",
            SESSION_SIZE_CACHE_FILE,
            uuid::Uuid::new_v4()
        ));
        fs::write(&temporary, &payload)?;
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Actively remove a session's size-cache entry after deletion, so the
    /// leftover entry does not dangle until the next /ss that recomputes.
    /// A failed write does not block deletion: attach_session_sizes's retain
    /// cleanup heals it.
    fn remove_session_size_cache_entry(&self, session_id: &str) -> io::Result<()> {
        let cache_path = self.root.join(SESSION_SIZE_CACHE_FILE);
        let mut cache = Self::load_session_size_cache(&cache_path);
        if cache.remove(session_id).is_none() {
            return Ok(());
        }
        Self::save_session_size_cache(&cache_path, &cache)
    }

    fn derived_session_history_artifact_sizes(&self) -> io::Result<FastMap<String, u64>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(FastMap::default());
            }
            Err(error) => return Err(error),
        };
        let mut sizes: FastMap<String, u64> = FastMap::default();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            let Some(session_id) = derived_session_history_artifact_session_id(file_name) else {
                continue;
            };
            let size = entry.metadata()?.len();
            let total = sizes.entry(session_id).or_insert(0);
            *total = total.saturating_add(size);
        }
        Ok(sizes)
    }

    fn derived_session_history_artifact_paths(&self, session_id: &str) -> io::Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            if derived_session_history_artifact_name_matches(file_name, session_id) {
                paths.push(entry.path());
            }
        }
        Ok(paths)
    }

    fn delete_derived_session_history_artifacts(&self, session_id: &str) -> io::Result<bool> {
        let paths = self.derived_session_history_artifact_paths(session_id)?;
        let existed = !paths.is_empty();
        for path in paths {
            remove_file_if_exists(&path)?;
        }
        Ok(existed)
    }

    fn delete_all_derived_session_history_artifacts(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            if is_any_derived_session_history_artifact_name(file_name) {
                remove_file_if_exists(&entry.path())?;
            }
        }
        Ok(())
    }

    pub(in crate::ai) fn delete_session(&self, session_id: &str) -> io::Result<bool> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let checkpoints = self.checkpoints_dir(session_id);
        let deleted = with_sessions_lifecycle_lock(&self.root, || {
            super::checkpoint::with_checkpoint_lock(&checkpoints, || {
                // Shares the same cross-process lock as the canonical writer /
                // rollback; the lock file itself must be kept, otherwise a
                // writer during or right after deletion could flock different
                // inodes for the same path.
                with_session_state_lock(&path, || {
                    self.delete_session_artifacts_unlocked(session_id, &path, &assets, &checkpoints)
                })
            })
        })?;
        if deleted {
            // Actively clean the cache entry after a successful delete; a
            // failed cleanup does not affect the delete result, and leftover
            // entries are healed by attach_session_sizes's retain.
            let _ = self.remove_session_size_cache_entry(session_id);
            // Reclaim the in-process state-lock map entry: subagent paths reclaim it in
            // `delete_subagent_history`, but a deleted main session would otherwise leave its
            // entry (keyed by the never-reused session path) in the global map for the rest of
            // the process lifetime — unbounded growth across create/delete cycles. The on-disk
            // `.sqlite.state.lock` file is deliberately kept: unlinking it after releasing the
            // flock would let a concurrent waiter that already opened the old inode lock a
            // different inode than the next writer at the same path.
            super::sqlite::remove_session_state_lock_entry(&path);
        }
        Ok(deleted)
    }

    /// Prune's final validation and deletion must jointly hold the lifecycle
    /// + checkpoint + session-state locks. `is_active` runs after all metadata
    /// checks pass and before the actual unlink, so the caller can do a last
    /// PID liveness check; the state-lock file itself is not part of the
    /// deletion set and survives this critical section.
    pub(in crate::ai) fn delete_session_if_unchanged(
        &self,
        candidate: &SessionInfo,
        cutoff: DateTime<Local>,
        is_active: impl FnOnce() -> io::Result<bool>,
    ) -> io::Result<PruneSessionDeleteResult> {
        Self::validate_session_id(&candidate.id)?;
        let path = self.session_history_file(&candidate.id);
        let assets = self.session_assets_dir(&candidate.id);
        let checkpoints = self.checkpoints_dir(&candidate.id);
        let result = with_sessions_lifecycle_lock(&self.root, || {
            super::checkpoint::with_checkpoint_lock(&checkpoints, || {
                with_session_state_lock(&path, || {
                    let current = self
                        .list_sessions()?
                        .into_iter()
                        .find(|session| session.id == candidate.id);
                    let Some(current) = current else {
                        return if path.exists() {
                            Err(io::Error::other(format!(
                                "failed to re-read session '{}' before prune",
                                candidate.id
                            )))
                        } else {
                            Ok(PruneSessionDeleteResult::Missing)
                        };
                    };
                    if !current.modified_local.is_some_and(|time| time < cutoff) {
                        return Ok(PruneSessionDeleteResult::NotExpired);
                    }
                    if current.modified_local != candidate.modified_local
                        || current.history_revision != candidate.history_revision
                    {
                        return Ok(PruneSessionDeleteResult::Changed);
                    }
                    if is_active()? {
                        return Ok(PruneSessionDeleteResult::Active);
                    }
                    self.delete_session_artifacts_unlocked(
                        &candidate.id,
                        &path,
                        &assets,
                        &checkpoints,
                    )?;
                    // Actively clean the cache entry after a successful
                    // delete; a failure does not block the prune result, and
                    // the cache self-heals.
                    let _ = self.remove_session_size_cache_entry(&candidate.id);
                    Ok(PruneSessionDeleteResult::Deleted)
                })
            })
        })?;
        if matches!(result, PruneSessionDeleteResult::Deleted) {
            // Same in-process lock-map entry reclaim as `delete_session`; must run after the
            // `with_session_state_lock` scope above so its Arc clone is dropped (strong_count 1).
            super::sqlite::remove_session_state_lock_entry(&path);
        }
        Ok(result)
    }

    fn delete_session_artifacts_unlocked(
        &self,
        session_id: &str,
        path: &Path,
        assets: &Path,
        checkpoints: &Path,
    ) -> io::Result<bool> {
        let existed = path.exists();
        delete_history_artifacts(path)?;
        let derived_existed = self.delete_derived_session_history_artifacts(session_id)?;
        delete_assets_dir(assets)?;
        remove_dir_if_exists(checkpoints)?;
        Ok(existed || derived_existed)
    }

    pub(in crate::ai) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let checkpoints = self.checkpoints_dir(session_id);
        super::checkpoint::with_checkpoint_lock(&checkpoints, || {
            delete_history_artifacts(&path)?;
            self.delete_derived_session_history_artifacts(session_id)?;
            delete_assets_dir(&assets)?;
            remove_dir_if_exists(&checkpoints)?;
            let _ = self.remove_session_size_cache_entry(session_id);
            Ok(())
        })
    }

    pub(in crate::ai) fn clear_session_history(&self, session_id: &str) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let checkpoints = self.checkpoints_dir(session_id);
        super::checkpoint::with_checkpoint_lock(&checkpoints, || {
            if path.exists() {
                super::sqlite::clear_session_history_sqlite(&path)?;
            }
            self.delete_derived_session_history_artifacts(session_id)?;
            delete_assets_dir(&assets)?;
            remove_dir_if_exists(&checkpoints)?;
            let _ = self.remove_session_size_cache_entry(session_id);
            Ok(())
        })
    }

    /// Recover interrupted checkpoint rollback transactions before reading a
    /// live session.
    pub(in crate::ai) fn recover_checkpoint_state(&self, session_id: &str) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        super::checkpoint::CheckpointStore::from_session_paths(
            self.session_history_file(session_id),
            self.session_assets_dir(session_id),
            self.checkpoints_dir(session_id),
        )
        .recover()
    }

    pub(in crate::ai) fn clear_all_sessions(&self) -> io::Result<usize> {
        let checkpoints_root = self.root.join("checkpoints");
        super::checkpoint::with_checkpoint_root_exclusive_lock(&checkpoints_root, || {
            let sessions = self.list_sessions()?;
            let mut deleted = 0usize;
            for session in sessions {
                let path = self.session_history_file(&session.id);
                let assets = self.session_assets_dir(&session.id);
                let checkpoints = self.checkpoints_dir(&session.id);
                let existed = path.exists();
                let result = (|| {
                    delete_history_artifacts(&path)?;
                    delete_assets_dir(&assets)?;
                    remove_dir_if_exists(&checkpoints)
                })();
                if result.is_ok() && existed {
                    deleted += 1;
                }
            }
            self.delete_all_derived_session_history_artifacts()?;
            // Tolerate orphan checkpoint directories left by old versions:
            // they have no matching `.sqlite` and are not enumerated by
            // `list_sessions`, but clear-all must still wipe all session data.
            remove_dir_if_exists(&checkpoints_root)?;
            // All sessions are cleared, so the whole cache is stale; just
            // remove the cache file.
            remove_file_if_exists(&self.root.join(SESSION_SIZE_CACHE_FILE))?;
            Ok(deleted)
        })
    }

    pub(in crate::ai) fn first_user_prompt(&self, session_id: &str) -> io::Result<Option<String>> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        read_first_user_prompt_sqlite(&path)
    }

    /// Whether the session is empty (no user messages at all).
    /// Used to clean up empty sessions when the user exits with Ctrl+C in
    /// interactive mode. A missing file or no role='user' rows in the
    /// messages table both count as empty.
    pub(in crate::ai) fn is_empty_session(&self, session_id: &str) -> io::Result<bool> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(true);
        }
        let count = super::sqlite::count_user_turns_sqlite(&path)?;
        Ok(count == 0)
    }

    /// Read the session title.
    pub(in crate::ai) fn read_session_title(&self, session_id: &str) -> io::Result<Option<String>> {
        Ok(self
            .read_session_title_with_origin(session_id)?
            .map(|title| title.text))
    }

    /// Read the session title and its origin. Old data without an origin
    /// marker is labeled `Legacy`.
    pub(in crate::ai) fn read_session_title_with_origin(
        &self,
        session_id: &str,
    ) -> io::Result<Option<SessionTitle>> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        let Some(text) = read_session_title_sqlite(&path)? else {
            return Ok(None);
        };
        let origin =
            SessionTitleOrigin::from_persisted(read_session_title_origin_sqlite(&path)?.as_deref());
        Ok(Some(SessionTitle { text, origin }))
    }

    /// Write a model-generated session title.
    pub(in crate::ai) fn write_session_title(
        &self,
        session_id: &str,
        title: &str,
    ) -> io::Result<()> {
        self.write_session_title_with_origin(session_id, title, SessionTitleOrigin::Model)
    }

    /// Write a session title together with its origin.
    pub(in crate::ai) fn write_session_title_with_origin(
        &self,
        session_id: &str,
        title: &str,
        origin: SessionTitleOrigin,
    ) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        write_session_title_sqlite(
            &self.session_history_file(session_id),
            title,
            origin.persisted_value(),
        )
    }

    /// Whether the user marked the session as important via `/mark`.
    /// A missing or unreadable session is treated as unmarked.
    pub(in crate::ai) fn read_session_marked(&self, session_id: &str) -> io::Result<bool> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(false);
        }
        read_session_marked_sqlite(&path)
    }

    /// Read the `/mark` message. A missing session or a session without a
    /// message yields an empty string.
    pub(in crate::ai) fn read_session_mark_message(&self, session_id: &str) -> io::Result<String> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(String::new());
        }
        Ok(read_session_mark_message_sqlite(&path)?.unwrap_or_default())
    }

    /// Atomically persist the session "important" mark flag and its optional
    /// mark message in one lock + one transaction (see `write_session_mark_sqlite`), so
    /// `/mark`/`/unmark` never persist a half state.
    pub(in crate::ai) fn write_session_mark(
        &self,
        session_id: &str,
        marked: bool,
        message: MarkMessageUpdate<'_>,
    ) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        write_session_mark_sqlite(&self.session_history_file(session_id), marked, message)
    }

    /// Whether an LLM-generated title already exists.
    pub(in crate::ai) fn has_generated_title(&self, session_id: &str) -> bool {
        self.read_session_title_with_origin(session_id)
            .ok()
            .flatten()
            .is_some_and(|title| title.origin == SessionTitleOrigin::Model)
    }

    pub(in crate::ai) fn read_all_messages(&self, session_id: &str) -> io::Result<Vec<Message>> {
        Self::validate_session_id(session_id)?;
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_all_messages_sqlite(&path)
    }

    /// Copy the full state while holding the source and destination session
    /// locks, and let the caller keep modifying the new branch afterwards.
    /// Multi-session locks are acquired in directory order so two forks in
    /// opposite directions cannot wait on each other.
    fn fork_session_with<T>(
        &self,
        src: &str,
        dst: &str,
        after_fork: impl FnOnce(&Path) -> io::Result<T>,
    ) -> io::Result<T> {
        Self::validate_session_id(src)?;
        Self::validate_session_id(dst)?;
        let src_path = self.session_history_file(src);
        let dst_path = self.session_history_file(dst);
        let src_assets = self.session_assets_dir(src);
        let dst_assets = self.session_assets_dir(dst);
        let src_checkpoints = self.checkpoints_dir(src);
        let dst_checkpoints = self.checkpoints_dir(dst);
        self.ensure_root_dir()?;
        super::checkpoint::with_checkpoint_locks(
            &[src_checkpoints.as_path(), dst_checkpoints.as_path()],
            || {
                if !src_path.exists() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("source session '{src}' not found"),
                    ));
                }
                if dst_path.exists() || dst_assets.exists() || dst_checkpoints.exists() {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!("destination session '{dst}' already exists"),
                    ));
                }
                backup_sqlite(&src_path, &dst_path)?;

                // The assets directory is optional; if present it must be
                // copied completely. Checkpoint bodies live in nested
                // directories, and a shallow copy would leave forked markers
                // pointing at missing files.
                if src_assets.is_dir() {
                    if let Err(error) = copy_dir_recursively(&src_assets, &dst_assets) {
                        let _ = delete_history_artifacts(&dst_path);
                        let _ = fs::remove_dir_all(&dst_assets);
                        return Err(error);
                    }
                    if let Err(error) = remap_context_checkpoint_paths_sqlite(
                        &dst_path,
                        Some(&src_assets),
                        &dst_assets,
                    ) {
                        let _ = delete_history_artifacts(&dst_path);
                        let _ = fs::remove_dir_all(&dst_assets);
                        return Err(error);
                    }
                }
                after_fork(&dst_path)
            },
        )
    }

    /// Copy the `src` session wholesale to `dst` as a new branch. The source
    /// session is untouched. Refuses to overwrite an existing dst (to avoid
    /// accidental clobbering). The assets directory is copied recursively if
    /// present.
    pub(in crate::ai) fn fork_session(&self, src: &str, dst: &str) -> io::Result<()> {
        self.fork_session_with(src, dst, |_| Ok(()))
    }

    /// Branch on top of `src`, keeping the first `keep_turns` complete user
    /// turns. Fits the "I want to roll back to a turn and continue in a
    /// different direction" scenario.
    pub(in crate::ai) fn branch_session(
        &self,
        src: &str,
        dst: &str,
        keep_turns: usize,
    ) -> io::Result<()> {
        self.fork_session_with(src, dst, |dst_path| {
            super::sqlite::truncate_messages_to_user_turns_sqlite(dst_path, keep_turns)
        })
    }

    pub(in crate::ai) fn export_session_to_markdown(
        &self,
        session_id: &str,
        output_path: &Path,
    ) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        let messages = self.read_all_messages(session_id)?;
        if messages.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Session '{}' not found or empty", session_id),
            ));
        }

        let markdown = messages_to_markdown(&messages, session_id);

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(output_path)?;
        use std::io::Write;
        file.write_all(markdown.as_bytes())?;

        Ok(())
    }

    /// Package a session wholesale into a zip archive (SQLite + assets) for
    /// cross-machine migration.
    /// Archive layout:
    ///   manifest.json   - version + original session id + creation time
    ///   session.sqlite  - full SQLite database (checkpointed; all messages /
    ///                      titles / summaries)
    ///   assets/...      - assets directory contents (if present)
    pub(in crate::ai) fn export_session_archive(
        &self,
        session_id: &str,
        output_path: &Path,
    ) -> io::Result<()> {
        Self::validate_session_id(session_id)?;
        let sqlite_path = self.session_history_file(session_id);
        let checkpoints = self.checkpoints_dir(session_id);
        super::checkpoint::with_checkpoint_lock(&checkpoints, || {
            if !sqlite_path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("session '{session_id}' not found"),
                ));
            }

            let snapshot = self
                .root
                .join(format!(".session-archive-{}.sqlite", uuid::Uuid::new_v4()));
            backup_sqlite(&sqlite_path, &snapshot)?;

            let result = (|| {
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                let file = File::create(output_path)?;
                let mut zip = zip::ZipWriter::new(file);
                let options = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated);

                // manifest.json
                let manifest = json!({
                    "version": 1u32,
                    "session_id": session_id,
                    "created_at": Local::now().to_rfc3339(),
                });
                zip.start_file("manifest.json", options)
                    .map_err(|e| io::Error::other(e.to_string()))?;
                zip.write_all(serde_json::to_vec_pretty(&manifest)?.as_slice())?;

                // session.sqlite
                zip.start_file("session.sqlite", options)
                    .map_err(|e| io::Error::other(e.to_string()))?;
                let mut sqlite_file = File::open(&snapshot)?;
                std::io::copy(&mut sqlite_file, &mut zip)?;

                // assets/ (optional)
                let assets_dir = self.session_assets_dir(session_id);
                if assets_dir.is_dir() {
                    add_dir_to_zip(&mut zip, &assets_dir, "assets", options)?;
                }

                zip.finish().map_err(|e| io::Error::other(e.to_string()))?;
                Ok(())
            })();
            let _ = delete_history_artifacts(&snapshot);
            result
        })
    }

    /// Import a session from a zip archive.
    /// `dst_id` is the session id to import into (errors if it already
    /// exists). Returns the imported session id.
    pub(in crate::ai) fn import_session_archive(
        &self,
        archive_path: &Path,
        dst_id: &str,
    ) -> io::Result<String> {
        Self::validate_session_id(dst_id)?;
        let file = File::open(archive_path)?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| io::Error::other(e.to_string()))?;

        validate_archive_entries(&mut archive)?;

        // Read the manifest (optional, only for validation)
        let manifest = {
            let mut buf = Vec::new();
            match archive.by_name("manifest.json") {
                Ok(mut entry) => {
                    std::io::copy(&mut entry, &mut buf)?;
                    serde_json::from_slice::<serde_json::Value>(&buf).ok()
                }
                Err(_) => None,
            }
        };
        let _ = manifest; // Only for validation; the original id is not enforced

        let dst_sqlite = self.session_history_file(dst_id);
        let dst_assets = self.session_assets_dir(dst_id);
        let dst_checkpoints = self.checkpoints_dir(dst_id);
        self.ensure_root_dir()?;
        super::checkpoint::with_checkpoint_lock(&dst_checkpoints, || {
            if dst_sqlite.exists() || dst_assets.exists() || dst_checkpoints.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("destination session '{dst_id}' already exists"),
                ));
            }

            let result = (|| {
                // Extract session.sqlite
                {
                    let mut entry = archive.by_name("session.sqlite").map_err(|e| {
                        io::Error::other(format!("session.sqlite not found in archive: {e}"))
                    })?;
                    let mut out = File::create(&dst_sqlite)?;
                    std::io::copy(&mut entry, &mut out)?;
                }

                // Extract assets/ (if present)
                for i in 0..archive.len() {
                    let mut entry = archive
                        .by_index(i)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    let name = entry.name().to_string();
                    if name == "manifest.json" || name == "session.sqlite" {
                        continue;
                    }
                    let rel = entry.enclosed_name().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsafe archive entry: {name}"),
                        )
                    })?;
                    let rel = rel.strip_prefix("assets").map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unexpected archive entry: {name}"),
                        )
                    })?;
                    if rel.as_os_str().is_empty() {
                        continue;
                    }
                    let out_path = dst_assets.join(rel);
                    if entry.is_dir() {
                        fs::create_dir_all(&out_path)?;
                        continue;
                    }
                    if let Some(parent) = out_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let mut out = File::create(&out_path)?;
                    std::io::copy(&mut entry, &mut out)?;
                }

                remap_context_checkpoint_paths_sqlite(&dst_sqlite, None, &dst_assets)?;
                Ok(dst_id.to_string())
            })();
            if result.is_err() {
                let _ = delete_history_artifacts(&dst_sqlite);
                let _ = delete_assets_dir(&dst_assets);
            }
            result
        })
    }
}

/// Validate the archive layout, rejecting Zip Slip, duplicate SQLite files,
/// and unknown entries before any destination file is created.
fn validate_archive_entries(archive: &mut zip::ZipArchive<File>) -> io::Result<()> {
    let mut session_sqlite_count = 0usize;
    let mut manifest_count = 0usize;
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|error| io::Error::other(error.to_string()))?;
        let name = entry.name().to_string();
        match name.as_str() {
            "session.sqlite" => {
                if entry.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "session.sqlite must be a file",
                    ));
                }
                session_sqlite_count += 1;
            }
            "manifest.json" => {
                if entry.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest.json must be a file",
                    ));
                }
                manifest_count += 1;
            }
            _ => {
                let enclosed_name = entry.enclosed_name().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsafe archive entry: {name}"),
                    )
                })?;
                let relative = enclosed_name.strip_prefix("assets").map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected archive entry: {name}"),
                    )
                })?;
                if relative.as_os_str().is_empty() && !entry.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "assets entry must be a directory",
                    ));
                }
            }
        }
    }
    if session_sqlite_count != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "archive must contain exactly one session.sqlite",
        ));
    }
    if manifest_count > 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "archive must not contain duplicate manifest.json entries",
        ));
    }
    Ok(())
}

fn file_size_if_exists(path: &Path) -> io::Result<u64> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error),
    }
}

fn remove_file_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn derived_session_history_artifact_name_matches(file_name: &str, session_id: &str) -> bool {
    let file_name = history_artifact_name_from_state_lock(file_name).unwrap_or(file_name);
    let current_proc_prefix = format!("{session_id}.proc-");
    let current_subagent_prefix = format!("{session_id}.subagent-");
    let legacy_proc_prefix = format!("{session_id}.sqlite.proc-");
    let legacy_subagent_prefix = format!("{session_id}.sqlite.subagent-");

    ((file_name.starts_with(&current_proc_prefix)
        || file_name.starts_with(&current_subagent_prefix))
        && is_sqlite_history_artifact_name(file_name))
        || file_name.starts_with(&legacy_proc_prefix)
        || file_name.starts_with(&legacy_subagent_prefix)
}

fn derived_session_history_artifact_session_id(file_name: &str) -> Option<String> {
    let file_name = history_artifact_name_from_state_lock(file_name).unwrap_or(file_name);
    let (raw_session_id, _) = file_name
        .split_once(".proc-")
        .or_else(|| file_name.split_once(".subagent-"))?;
    if !raw_session_id.ends_with(".sqlite") && !is_sqlite_history_artifact_name(file_name) {
        return None;
    }
    let session_id = raw_session_id
        .strip_suffix(".sqlite")
        .unwrap_or(raw_session_id);
    if SessionStore::validate_session_id(session_id).is_err() {
        return None;
    }
    Some(session_id.to_string())
}

fn is_any_derived_session_history_artifact_name(file_name: &str) -> bool {
    let file_name = history_artifact_name_from_state_lock(file_name).unwrap_or(file_name);
    ((file_name.contains(".proc-") || file_name.contains(".subagent-"))
        && is_sqlite_history_artifact_name(file_name))
        || file_name.contains(".sqlite.proc-")
        || file_name.contains(".sqlite.subagent-")
}

fn history_artifact_name_from_state_lock(file_name: &str) -> Option<&str> {
    file_name
        .strip_prefix('.')
        .and_then(|name| name.strip_suffix(".state.lock"))
}

fn is_sqlite_history_artifact_name(file_name: &str) -> bool {
    file_name.ends_with(".sqlite")
        || file_name.ends_with(".sqlite-wal")
        || file_name.ends_with(".sqlite-shm")
        || file_name.ends_with(".sqlite-journal")
}

/// Count bytes of regular files in a directory; symlinks are not followed,
/// avoiding cycles or out-of-tree reads when displaying sizes.
fn directory_size(path: &Path) -> io::Result<u64> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut total = 0u64;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            total = total.saturating_add(directory_size(&entry.path())?);
        } else if file_type.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn remove_dir_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Recursively add a directory's contents to a zip archive.
fn add_dir_to_zip(
    zip: &mut zip::ZipWriter<File>,
    dir: &Path,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
) -> io::Result<()> {
    let entries = fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let zip_name = format!("{prefix}/{name}");
        if path.is_dir() {
            add_dir_to_zip(zip, &path, &zip_name, options)?;
        } else {
            zip.start_file(&zip_name, options)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let mut f = File::open(&path)?;
            std::io::copy(&mut f, zip)?;
        }
    }
    Ok(())
}

fn sessions_root_from_history_file(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let name = history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("history");
    parent.join(format!("{name}.sessions"))
}

fn sanitize_session_id(session_id: &str) -> String {
    let mut out = String::new();
    for ch in session_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "session".to_string()
    } else {
        out
    }
}

/// Generate a concise session title/summary from the first user message.
/// Handles JSON content (e.g. image data), extracting key information into a
/// summarizing title. Unlike a plain truncation, this function:
/// 1. strips agent/command prefixes (e.g. "a ", "/")
/// 2. extracts the first sentence (up to a sentence-ending mark or newline)
/// 3. strips common filler prefixes (e.g. "帮我", "请", "我想")
/// 4. caps the result at a reasonable length
pub(in crate::ai) fn generate_session_summary(first_prompt: &str) -> String {
    let text = first_prompt.trim();
    if text.is_empty() {
        return "(空会话)".to_string();
    }

    // Strip the agent prefix (e.g. "a ", "a:", "agent:", etc.)
    let text = strip_agent_prefix(text);

    // Handle merged multi-message input (separated by \n---\n)
    let messages: Vec<&str> = text.split("\n---\n").collect();
    let mut all_text_parts = Vec::new();
    let mut has_any_image = false;

    for msg in &messages {
        let msg = msg.trim();
        if msg.is_empty() {
            continue;
        }

        // Try parsing as a JSON array (multimodal messages)
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(msg) {
            let (parts, has_image) = extract_from_json_array(&arr);
            all_text_parts.extend(parts);
            if has_image {
                has_any_image = true;
            }
        }
        // Try parsing as a single JSON object
        else if let Ok(obj) = serde_json::from_str::<serde_json::Value>(msg) {
            if let Some(obj) = obj.as_object() {
                let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match item_type {
                    "text" => {
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            let cleaned = t.trim();
                            if !cleaned.is_empty() {
                                all_text_parts.push(cleaned.to_string());
                            }
                        }
                    }
                    "image_url" => has_any_image = true,
                    _ => {}
                }
            }
        }
        // Plain text
        else {
            // Extract the first sentence (up to a sentence-ending mark or
            // newline)
            let first_sentence = extract_first_sentence(msg);
            if !first_sentence.is_empty() {
                all_text_parts.push(first_sentence);
            }
        }
    }

    if all_text_parts.is_empty() && has_any_image {
        return "[图片]".to_string();
    }
    if all_text_parts.is_empty() {
        return "(无文本内容)".to_string();
    }

    let combined = all_text_parts.join(" ");
    // Strip common filler prefixes so the title is more concise
    let cleaned = strip_filler_prefixes(&combined);
    truncate_summary(&cleaned, 40)
}

/// Extract text parts and the image flag from a JSON array.
fn extract_from_json_array(arr: &[serde_json::Value]) -> (Vec<String>, bool) {
    let mut parts = Vec::new();
    let mut has_image = false;
    for item in arr {
        if let Some(obj) = item.as_object() {
            let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match item_type {
                "text" => {
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        let cleaned = t.trim();
                        if !cleaned.is_empty() {
                            parts.push(cleaned.to_string());
                        }
                    }
                }
                "image_url" => has_image = true,
                _ => {}
            }
        }
    }
    (parts, has_image)
}

/// Truncate a summary to the given length, appending an ellipsis.
fn truncate_summary(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_len).collect();
    out.push_str("…");
    out
}

/// Shared filler-prefix list for session title cleaning (request shells /
/// question words).
///
/// Historically `strip_request_filler_prefixes` and `strip_filler_prefixes`
/// each maintained a nearly identical array, and the former was missing
/// 如何/怎么/怎样, so "如何实现 X" was cleaned differently on the LLM-title
/// path versus the fallback-summary path. This single source of truth is
/// shared by both.
///
/// Order is match priority: the strip loops greedily match in array order,
/// so long compound prefixes must come before their shorter substrings (e.g.
/// "你帮我看一下" before "帮我"), otherwise the short prefix is stripped
/// first, leaving a broken shell. New entries: long prefixes first, short
/// words last.
const SESSION_TITLE_FILLER_PREFIXES: &[&str] = &[
    "你帮我看一下",
    "你帮我给",
    "请帮我给",
    "麻烦帮我给",
    "帮我看一下",
    "帮我给",
    "能不能帮我",
    "可以帮我",
    "麻烦帮我",
    "请帮我",
    "你帮我",
    "帮我",
    "请",
    "麻烦",
    "拜托",
    "求",
    "我想",
    "我想要",
    "我需要",
    "希望",
    "希望能",
    "想问一下",
    "问一下",
    "请问",
    "想知道",
    "看一下",
    "帮看看",
    "看看",
    "如何",
    "怎么",
    "怎样",
];

/// Remove the `<think>...</think>` chain of thought from model output,
/// returning clean text usable as a title.
///
/// Background: some models (thinking mode) wrap the chain of thought in a
/// `<think>` tag and return it together with the answer. Without stripping,
/// `.lines().next()` would cut at the `<think>` first line and
/// `is_low_quality_session_title("<think>")` would judge it acceptable, so
/// a reasoning fragment could be written as the title.
///
/// Rules: match `<think>` / `</think>` case-insensitively; remove a complete
/// pair as a whole; for an unclosed `<think>` (open without close), truncate
/// to the content before that `<think>` (the chain of thought usually trails,
/// and the text before it is the answer). Non-think text is kept verbatim.
pub(in crate::ai) fn strip_think_tags(text: &str) -> String {
    // Locate tag bytes directly on the original string with ASCII
    // case-insensitive matching, avoiding a `to_lowercase()` copy that would
    // then be sliced back into `text` by index — some Unicode characters
    // change byte length when lowercased, which would misalign the indices.
    fn find_ci(haystack: &str, needle_lower: &str, from: usize) -> Option<usize> {
        let bytes = haystack.as_bytes();
        let nlen = needle_lower.len();
        if nlen == 0 || from + nlen > bytes.len() {
            return None;
        }
        (from..=bytes.len() - nlen).find(|&i| {
            bytes[i..i + nlen]
                .iter()
                .zip(needle_lower.bytes())
                .all(|(b, n)| b.to_ascii_lowercase() == n)
        })
    }

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(open) = find_ci(text, "<think>", cursor) {
        out.push_str(&text[cursor..open]);
        let after_open = open + "<think>".len();
        match find_ci(text, "</think>", after_open) {
            Some(close) => {
                // Skip the whole `<think>...</think>` segment; continue after
                // the closing tag.
                cursor = close + "</think>".len();
            }
            None => {
                // Unclosed: drop everything after the `<think>` start.
                return out.trim().to_string();
            }
        }
    }
    out.push_str(&text[cursor..]);
    out.trim().to_string()
}

/// Clean a model-generated session title so request shells like "帮我/请问"
/// do not end up in the title.
pub(in crate::ai) fn normalize_generated_session_title(title: &str) -> String {
    // Strip the chain of thought first as defense in depth, covering paths
    // where finalize hands over the raw LLM text directly.
    let title = strip_think_tags(title);
    let first_line = title.lines().next().unwrap_or("").trim();
    if is_preserved_content_message(first_line) {
        return String::new();
    }
    let without_agent = strip_agent_prefix(first_line);
    let without_request = strip_request_filler_prefixes(without_agent).0;
    truncate_summary(without_request.trim(), 30)
}

/// Returns whether the existing title looks like a raw user-request fragment;
/// such old titles are allowed to be regenerated and overwritten by later turns.
pub(in crate::ai) fn is_low_quality_session_title(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty()
        || trimmed.contains('\n')
        || trimmed.contains('\r')
        || is_preserved_content_message(trimmed)
    {
        return true;
    }
    let without_agent = strip_agent_prefix(trimmed);
    let (_, stripped_request_prefix) = strip_request_filler_prefixes(without_agent);
    stripped_request_prefix || trimmed.chars().count() > 40
}

/// The archive protocol is internal context; it must not be shown as a session
/// title or treated as a valid title.
pub(super) fn is_preserved_content_message(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("[[PRESERVED_CONTENT_STUB_V1]]")
        || text.starts_with("较早的用户图片内容已归档")
        || text.starts_with("较早的用户文本内容已归档")
}

fn strip_request_filler_prefixes(mut text: &str) -> (&str, bool) {
    let fillers = SESSION_TITLE_FILLER_PREFIXES;
    let mut stripped_any = false;
    loop {
        let mut stripped = false;
        let trimmed = text.trim_start();
        for filler in fillers {
            if let Some(rest) = trimmed.strip_prefix(filler) {
                text = rest.trim_start();
                stripped = true;
                stripped_any = true;
                break;
            }
        }
        if !stripped {
            return (trimmed, stripped_any);
        }
    }
}

/// Strip the agent/command prefix (e.g. "a ", "a:", "a：", "/", etc.).
fn strip_agent_prefix(text: &str) -> &str {
    let t = text.trim_start();
    // Match agent prefixes such as "a ", "a:", "a：".
    if let Some(rest) = t.strip_prefix("a ") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("a:") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("a：") {
        return rest.trim_start();
    }
    // Match command prefixes starting with "/" (drop the command name, keep
    // the arguments).
    if let Some(rest) = t.strip_prefix('/') {
        // Skip the command name (up to the first whitespace).
        if let Some(space_pos) = rest.find(|c: char| c.is_whitespace()) {
            return rest[space_pos..].trim_start();
        }
        // Only a command name with no arguments: return empty.
        return "";
    }
    t
}

/// Extract the first sentence (up to a period, question mark, exclamation
/// mark, or newline).
fn extract_first_sentence(text: &str) -> String {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut end = text.len();
    for (idx, (i, ch)) in chars.iter().enumerate() {
        match ch {
            // Chinese period / question mark / exclamation mark / newline:
            // cut here.
            '。' | '？' | '！' | '\n' => {
                end = *i;
                break;
            }
            // ASCII period: decide whether it is a sentence boundary or part
            // of a filename / identifier.
            '.' => {
                let prev_is_alnum = idx > 0 && chars[idx - 1].1.is_alphanumeric();
                let next_is_alnum = idx + 1 < chars.len() && chars[idx + 1].1.is_alphanumeric();
                // Alphanumeric on both sides (e.g. a.rs, file.txt, v2.0):
                // not a sentence boundary.
                if prev_is_alnum && next_is_alnum {
                    continue;
                }
                // Followed by whitespace or end of string: treat as a sentence
                // boundary.
                let next_is_space = idx + 1 < chars.len() && chars[idx + 1].1.is_whitespace();
                let is_last = idx + 1 >= chars.len();
                if next_is_space || is_last {
                    end = *i;
                    break;
                }
            }
            // ASCII question mark / exclamation mark: cut here.
            '?' | '!' => {
                end = *i;
                break;
            }
            _ => {}
        }
    }
    text[..end].trim().to_string()
}

/// Strip common redundant prefixes so the title is shorter and more concise.
fn strip_filler_prefixes(text: &str) -> String {
    let fillers = SESSION_TITLE_FILLER_PREFIXES;
    let mut t = text.trim();
    loop {
        let mut stripped = false;
        for filler in fillers {
            if let Some(rest) = t.strip_prefix(filler) {
                t = rest.trim_start();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        SESSION_SIZE_CACHE_FILE, SessionStore, SessionTitleOrigin, generate_session_summary,
        is_low_quality_session_title, normalize_generated_session_title, strip_think_tags,
    };
    use crate::ai::history::{Message, append_history_messages};
    use serde_json::Value;
    use std::{fs, path::PathBuf};

    fn temp_history_file() -> PathBuf {
        std::env::temp_dir().join(format!(
            "ai-session-test-{}-{}.sqlite",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    fn checkpoint_marker(path: &std::path::Path) -> Message {
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "[context_checkpoint path={}] durable state",
                path.display()
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn session_title_fallback_strips_compound_filler_prefixes() {
        assert_eq!(
            generate_session_summary("你帮我给a.rs这个agent的system prompt加一个限制吧"),
            "a.rs这个agent的system prompt加一个限制吧"
        );
        assert_eq!(
            generate_session_summary("帮我给 session title 问题排查一下"),
            "session title 问题排查一下"
        );
    }

    #[test]
    fn low_quality_session_titles_are_normalized_or_regenerated() {
        let bad_title = "你帮我给a.rs这个agent的system prompt加一个限制吧";
        assert!(is_low_quality_session_title(bad_title));
        assert_eq!(
            normalize_generated_session_title(bad_title),
            "a.rs这个agent的system prompt加一个限制…"
        );

        assert!(!is_low_quality_session_title("session title 问题排查"));
    }

    #[test]
    fn preserved_content_stub_is_not_a_session_title() {
        let stub = r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image","file_path":"/tmp/x"}"#;

        assert!(is_low_quality_session_title(stub));
        assert!(normalize_generated_session_title(stub).is_empty());
    }

    #[test]
    fn strip_think_tags_removes_reasoning_and_keeps_real_title() {
        // Paired tags: the whole chain of thought is removed, leaving only
        // the real title after it.
        assert_eq!(
            strip_think_tags("<think>让我想想用户到底要什么</think>\n优化上下文压缩逻辑"),
            "优化上下文压缩逻辑"
        );
        // Case-insensitive.
        assert_eq!(
            strip_think_tags("<Think>reasoning</THINK>Session 标题修复"),
            "Session 标题修复"
        );
        // Unclosed: truncated before `<think>`; the answer that precedes it
        // is still kept.
        assert_eq!(
            strip_think_tags("真正的标题<think>后面是没写完的思维链"),
            "真正的标题"
        );
        // No tags: returned as-is (only trimmed).
        assert_eq!(strip_think_tags("普通标题"), "普通标题");
        // After normalize, a pure chain-of-thought input is not treated as a
        // valid title.
        assert!(normalize_generated_session_title("<think>only reasoning</think>").is_empty());
    }

    #[test]
    fn preserved_content_notice_is_not_a_fallback_session_title() {
        let notice = "较早的用户图片内容已归档，原文未丢失。\n归档文件: /tmp/image.json";
        let fallback = normalize_generated_session_title(&generate_session_summary(notice));

        assert!(fallback.is_empty());
    }

    #[test]
    fn session_size_cache_reuses_top_level_fingerprint_hits() {
        // Temporary sessions root: SessionStore derives its root from the
        // history file path.
        let session_id = format!("size-cache-{}", uuid::Uuid::new_v4());
        let root = std::env::temp_dir().join(format!(
            "ai-session-size-cache-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let store = SessionStore::new(&root.join("unused.sqlite"));
        store.ensure_root_dir().unwrap();

        // Create a session: sqlite main DB + two levels of files inside the
        // assets dir.
        let sqlite_path = store.session_history_file(&session_id);
        fs::write(&sqlite_path, vec![0u8; 100]).unwrap();
        let assets_dir = store.session_assets_dir(&session_id);
        fs::create_dir_all(assets_dir.join("sub")).unwrap();
        fs::write(assets_dir.join("a.txt"), vec![0u8; 30]).unwrap();
        fs::write(assets_dir.join("sub").join("b.txt"), vec![0u8; 40]).unwrap();

        let mut sessions = store.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        // Expected: main DB 100 + assets 70 + checkpoints 0 (absent from the
        // top-level fingerprint).
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 170);

        // Second call hits the cache; the size stays unchanged.
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 170);
        // The cache file is persisted.
        assert!(store.root.join(SESSION_SIZE_CACHE_FILE).is_file());

        // A new top-level file changes the fingerprint, invalidating the
        // cache and forcing a recompute.
        fs::write(assets_dir.join("c.txt"), vec![0u8; 20]).unwrap();
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 190);

        // New deep file: the sub dir mtime changes, invalidating the
        // fingerprint and forcing a recompute.
        fs::write(assets_dir.join("sub").join("c2.txt"), vec![0u8; 15]).unwrap();
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 205);

        // Deep file overwrite: the two-level fingerprint picks up size/mtime
        // changes inside the subdir, invalidating the cache and forcing a
        // recompute.
        fs::write(assets_dir.join("sub").join("b.txt"), vec![0u8; 10]).unwrap();
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 175);

        // Self-healing: recompute after a new top-level file; the deep
        // overwrite is reflected too.
        fs::write(assets_dir.join("d.txt"), vec![0u8; 5]).unwrap();
        store.attach_session_sizes(&mut sessions).unwrap();
        assert_eq!(sessions[0].size_bytes, 180);

        // Clean up the temp directory.
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_removes_size_cache_entry() {
        let session_id = format!("del-cache-{}", uuid::Uuid::new_v4());
        let root = std::env::temp_dir().join(format!(
            "ai-session-del-cache-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let store = SessionStore::new(&root.join("unused.sqlite"));
        store.ensure_root_dir().unwrap();

        // Create a session and trigger a cache write.
        let sqlite_path = store.session_history_file(&session_id);
        fs::write(&sqlite_path, vec![0u8; 100]).unwrap();
        let assets_dir = store.session_assets_dir(&session_id);
        fs::create_dir_all(assets_dir.join("sub")).unwrap();
        fs::write(assets_dir.join("a.txt"), vec![0u8; 30]).unwrap();
        fs::write(assets_dir.join("sub").join("b.txt"), vec![0u8; 40]).unwrap();

        let mut sessions = store.list_sessions().unwrap();
        store.attach_session_sizes(&mut sessions).unwrap();
        assert!(store.root.join(SESSION_SIZE_CACHE_FILE).is_file());

        // After deleting the session, its cache entry must be removed
        // synchronously (not relying on the next attach's retain
        // self-healing).
        assert!(store.delete_session(&session_id).unwrap());
        let cache =
            SessionStore::load_session_size_cache(&store.root.join(SESSION_SIZE_CACHE_FILE));
        assert!(!cache.contains_key(&session_id));

        // Clean up the temp directory.
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn session_title_origin_persists_with_the_title() {
        let history_file = temp_history_file();
        let store = SessionStore::new(&history_file);
        let session_id = "current";
        store.ensure_root_dir().unwrap();

        store
            .write_session_title_with_origin(
                session_id,
                "修复 session title",
                SessionTitleOrigin::Fallback,
            )
            .unwrap();
        assert_eq!(
            store.read_session_title_with_origin(session_id).unwrap(),
            Some(super::SessionTitle {
                text: "修复 session title".to_string(),
                origin: SessionTitleOrigin::Fallback,
            })
        );

        store
            .write_session_title(session_id, "会话标题生成修复")
            .unwrap();
        assert_eq!(
            store.read_session_title_with_origin(session_id).unwrap(),
            Some(super::SessionTitle {
                text: "会话标题生成修复".to_string(),
                origin: SessionTitleOrigin::Model,
            })
        );

        let _ = fs::remove_dir_all(store.sessions_root());
    }

    #[test]
    fn fork_copies_checkpoint_assets_and_remaps_marker_paths() {
        let history_file = temp_history_file();
        let store = SessionStore::new(&history_file);
        let source_id = "source";
        let target_id = "target";
        let source_db = store.session_history_file(source_id);
        let source_asset = store
            .session_assets_dir(source_id)
            .join("context-checkpoints")
            .join("durable.md");
        fs::create_dir_all(source_asset.parent().unwrap()).unwrap();
        fs::write(&source_asset, "source checkpoint body").unwrap();
        append_history_messages(&source_db, &[checkpoint_marker(&source_asset)]).unwrap();

        store.fork_session(source_id, target_id).unwrap();

        let target_asset = store
            .session_assets_dir(target_id)
            .join("context-checkpoints")
            .join("durable.md");
        let target_messages = store.read_all_messages(target_id).unwrap();
        assert_eq!(target_messages.len(), 1);
        assert!(
            target_messages[0]
                .content
                .as_str()
                .unwrap_or_default()
                .contains(target_asset.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(store.session_assets_dir(source_id)).unwrap();
        assert_eq!(
            fs::read_to_string(&target_asset).unwrap(),
            "source checkpoint body"
        );

        let _ = fs::remove_dir_all(store.sessions_root());
    }

    #[test]
    fn archive_import_remaps_checkpoint_marker_paths() {
        let history_file = temp_history_file();
        let store = SessionStore::new(&history_file);
        let source_id = "source";
        let target_id = "imported";
        let source_db = store.session_history_file(source_id);
        let source_asset = store
            .session_assets_dir(source_id)
            .join("context-checkpoints")
            .join("durable.md");
        fs::create_dir_all(source_asset.parent().unwrap()).unwrap();
        fs::write(&source_asset, "archived checkpoint body").unwrap();
        append_history_messages(&source_db, &[checkpoint_marker(&source_asset)]).unwrap();

        let archive = std::env::temp_dir().join(format!(
            "ai-session-archive-test-{}.zip",
            uuid::Uuid::new_v4()
        ));
        store.export_session_archive(source_id, &archive).unwrap();
        store.import_session_archive(&archive, target_id).unwrap();

        let target_asset = store
            .session_assets_dir(target_id)
            .join("context-checkpoints")
            .join("durable.md");
        let target_messages = store.read_all_messages(target_id).unwrap();
        assert_eq!(target_messages.len(), 1);
        assert!(
            target_messages[0]
                .content
                .as_str()
                .unwrap_or_default()
                .contains(target_asset.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(store.session_assets_dir(source_id)).unwrap();
        assert_eq!(
            fs::read_to_string(&target_asset).unwrap(),
            "archived checkpoint body"
        );

        let _ = fs::remove_file(archive);
        let _ = fs::remove_dir_all(store.sessions_root());
    }

    #[test]
    fn session_ids_reject_path_separators_and_silent_normalization() {
        for session_id in ["", "../other", "has space", "name/slash", "名字"] {
            assert!(SessionStore::validate_session_id(session_id).is_err());
        }
        assert!(SessionStore::validate_session_id("safe_session-123").is_ok());
    }

    #[test]
    fn archive_import_rejects_zip_slip_before_creating_session_files() {
        use std::io::Write;

        let history_file = temp_history_file();
        let store = SessionStore::new(&history_file);
        let archive_path = std::env::temp_dir().join(format!(
            "ai-session-malicious-archive-{}.zip",
            uuid::Uuid::new_v4()
        ));
        let file = std::fs::File::create(&archive_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("session.sqlite", options).unwrap();
        zip.write_all(b"not reached: archive validation must fail first")
            .unwrap();
        zip.start_file("assets/../../escaped.txt", options).unwrap();
        zip.write_all(b"malicious").unwrap();
        zip.finish().unwrap();

        let error = store
            .import_session_archive(&archive_path, "imported")
            .expect_err("Zip Slip archive must be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(!store.session_history_file("imported").exists());
        assert!(!store.session_assets_dir("imported").exists());

        let _ = fs::remove_file(archive_path);
        let _ = fs::remove_dir_all(store.sessions_root());
    }

    #[test]
    fn listed_session_size_includes_assets_and_checkpoints() {
        let history_file = temp_history_file();
        let store = SessionStore::new(&history_file);
        let session_id = "sized";
        let sqlite_path = store.session_history_file(session_id);
        append_history_messages(
            &sqlite_path,
            &[Message {
                role: "user".to_string(),
                content: Value::String("size me".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let assets_dir = store.session_assets_dir(session_id);
        let checkpoints_dir = store.checkpoints_dir(session_id);
        fs::create_dir_all(&assets_dir).unwrap();
        fs::create_dir_all(&checkpoints_dir).unwrap();
        fs::write(assets_dir.join("asset.bin"), b"assets").unwrap();
        fs::write(checkpoints_dir.join("checkpoint.bin"), b"checkpoints").unwrap();
        fs::write(
            store.sessions_root().join("sized.proc-42.sqlite"),
            b"derived",
        )
        .unwrap();

        let mut listed = store.list_sessions().unwrap();
        store.attach_session_sizes(&mut listed).unwrap();
        assert!(!listed.iter().any(|session| session.id == "sized.proc-42"));
        let session = listed
            .iter()
            .find(|session| session.id == session_id)
            .unwrap();
        let expected_minimum = fs::metadata(&sqlite_path).unwrap().len()
            + b"assets".len() as u64
            + b"checkpoints".len() as u64
            + b"derived".len() as u64;
        assert!(session.size_bytes >= expected_minimum);

        let _ = fs::remove_dir_all(store.sessions_root());
    }
}
