//! Session checkpoints and movable branch pointers.
//!
//! Each session's conversation history lives in its own SQLite WAL file.
//! Checkpoints create a consistent snapshot via the SQLite Online Backup API
//! and also save the session assets, so that copying the main database cannot
//! miss WAL pages or context-checkpoint bodies.
//!
//! The context-history cache keys invalidation on file len/mtime and
//! `meta.history_revision`; copying files changes those values, which
//! invalidates the cache automatically, so the correct history is readable
//! after a rollback.
//!
//! Each generation also records its parent (the live base at save time) in a
//! `checkpoint-v2` manifest, and the checkpoints directory tracks the live
//! base in `BASE.ref` (updated on every save and checkout). Branch files are
//! movable pointers to immutable generations; `HEAD.ref` either names a
//! detached generation or an attached branch. `list` fills missing parents
//! (pre-tracking `checkpoint-v1` snapshots, legacy layouts)
//! by chaining in save order so the tree still reads top-down.

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(target_os = "macos")]
use std::{ffi::CString, os::unix::ffi::OsStrExt};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, RwLock},
};

use base64::Engine as _;
use chrono::{DateTime, Local};
#[cfg(unix)]
use libc::{LOCK_EX, LOCK_SH, LOCK_UN, flock};
use rust_tools::commonw::FastMap;

use crate::ai::history::SessionStore;

#[cfg(test)]
use super::sqlite::restore_sqlite_after_rollback_with_transaction;
use super::{
    sqlite::{
        backup_sqlite, clear_session_history_sqlite, live_rollback_transaction_is_published,
        message_count_sqlite_read_only, read_all_messages_sqlite_read_only,
        restore_sqlite_after_rollback,
        restore_staged_sqlite_after_rollback,
    },
    types::Message,
};
pub(in crate::ai) const MAX_CHECKPOINTS_PER_SESSION: usize = 50;
pub(in crate::ai) const MAX_CHECKPOINT_STORAGE_BYTES: u64 = 512 * 1024 * 1024;
const SQLITE_SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];
const GENERATION_SQLITE_FILE: &str = "history.sqlite";
const GENERATION_ASSETS_DIR: &str = "assets";
const GENERATION_MANIFEST_FILE: &str = "checkpoint.manifest";
const GENERATION_STAGE_MARKER: &str = ".stage-";
const GENERATION_PREVIOUS_MARKER: &str = ".previous-";
const GENERATION_PUBLISH_READY_MARKER: &str = ".publish-ready";
/// Git-style pointer to the most recently preserved checkpoint. It is a plain
/// file (not a checkpoint directory), so it never shows up in listings; it
/// only records a name for the `HEAD` (`@`) alias to resolve. `-` is the
/// previous checkout instead (see `PREV_HEAD_REF_FILE`).
const HEAD_REF_FILE: &str = "HEAD.ref";
const HEAD_BRANCH_PREFIX: &str = "ref: branches/";
const BRANCHES_DIR: &str = "branches";
const DEFAULT_BRANCH_NAME: &str = "main";
/// Live-base pointer: which checkpoint the live history was last saved from
/// or rolled back to. The next `save` records it as the new checkpoint's
/// parent, so `tree` shows real lineage. Like `HEAD.ref` it is a plain file,
/// never a listing entry.
const BASE_REF_FILE: &str = "BASE.ref";
/// Previous-HEAD pointer for Git-style `switch -` / `checkout -`: the raw
/// `HEAD.ref` content before the last successful branch/checkout navigation.
/// A plain file, never a listing entry; absent until the first navigation.
const PREV_HEAD_REF_FILE: &str = "PREV_HEAD.ref";
const LIVE_ROLLBACK_PREFIX: &str = ".live-rollback-";
const LIVE_ROLLBACK_MANIFEST: &str = "rollback.manifest";

/// In-process mutexes combined with `flock` cover the checkpoint lifecycle of
/// one session. Each session holds its own Mutex so snapshot operations of
/// different sessions do not serialize against each other.
static CHECKPOINT_SESSION_LOCKS: LazyLock<Mutex<FastMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

/// Shared/exclusive gate for the checkpoint root directory. Normal operations
/// hold the shared lock, `clear-all` holds the exclusive lock, so the latter
/// cannot interleave with a just-created session checkpoint.
static CHECKPOINT_ROOT_LOCKS: LazyLock<Mutex<FastMap<PathBuf, Arc<RwLock<()>>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointInfo {
    pub(in crate::ai) name: String,
    pub(in crate::ai) message: Option<String>,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) size_bytes: u64,
    /// Bytes charged to the per-session checkpoint quota. COW-cloned file data
    /// is shared with the live assets and therefore does not consume quota
    /// until a later snapshot has to materialize its own copy.
    pub(in crate::ai) storage_bytes: u64,
    /// Parent checkpoint name (`None` for roots). A parent that no longer
    /// exists (deleted after the child was saved) is kept as-is so the tree
    /// can mark the child as an orphan instead of a false root.
    pub(in crate::ai) parent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) struct CheckpointBranch {
    pub(in crate::ai) name: String,
    /// `None` means the branch exists but has not received its first
    /// checkpoint yet.
    pub(in crate::ai) target: Option<String>,
    /// The branch names a checkpoint object that is no longer present.
    pub(in crate::ai) is_dangling: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai) enum CheckpointHead {
    Attached {
        branch: String,
        target: Option<String>,
    },
    Detached(Option<String>),
}

const CHECKPOINT_STATUS_PART_CHARS: usize = 10;

fn compact_checkpoint_status_part(value: &str) -> String {
    if value.chars().count() <= CHECKPOINT_STATUS_PART_CHARS {
        return value.to_string();
    }
    value
        .chars()
        .take(CHECKPOINT_STATUS_PART_CHARS - 1)
        .chain(std::iter::once('…'))
        .collect()
}

pub(in crate::ai) fn checkpoint_status_label(head: &CheckpointHead) -> Option<String> {
    match head {
        CheckpointHead::Attached { branch, target } => Some(format!(
            "{}@{}",
            compact_checkpoint_status_part(branch),
            target
                .as_deref()
                .map(compact_checkpoint_status_part)
                .unwrap_or_else(|| "-".to_string())
        )),
        CheckpointHead::Detached(Some(target)) => {
            Some(format!("det@{}", compact_checkpoint_status_part(target)))
        }
        CheckpointHead::Detached(None) => None,
    }
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointStore {
    /// The live sqlite file of the current session.
    session_file: PathBuf,
    /// The assets directory of the current session.
    session_assets: PathBuf,
    /// The checkpoint directory of this session.
    dir: PathBuf,
}

#[derive(Debug, Clone)]
enum CheckpointSource {
    Generation { sqlite: PathBuf, assets: PathBuf },
    Legacy { sqlite: PathBuf },
}

#[derive(Debug, Clone)]
struct RefSnapshot {
    head: Option<String>,
    base: Option<String>,
    branch: Option<(PathBuf, Option<String>)>,
}

impl CheckpointStore {
    pub(in crate::ai) fn new(history_file: &Path, session_id: &str) -> Self {
        let store = SessionStore::new(history_file);
        Self::from_session_store(&store, session_id)
    }

    pub(in crate::ai) fn from_session_store(store: &SessionStore, session_id: &str) -> Self {
        Self::from_session_paths(
            store.session_history_file(session_id),
            store.session_assets_dir(session_id),
            store.checkpoints_dir(session_id),
        )
    }

    pub(super) fn from_session_paths(
        session_file: PathBuf,
        session_assets: PathBuf,
        dir: PathBuf,
    ) -> Self {
        Self {
            session_file,
            session_assets,
            dir,
        }
    }

    fn checkpoint_path(&self, name: &str) -> PathBuf {
        self.generation_dir(name).join(GENERATION_SQLITE_FILE)
    }

    fn checkpoint_assets_path(&self, name: &str) -> PathBuf {
        self.generation_dir(name).join(GENERATION_ASSETS_DIR)
    }

    fn generation_dir(&self, name: &str) -> PathBuf {
        self.dir.join(sanitize_name(name))
    }

    fn legacy_checkpoint_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}.sqlite", sanitize_name(name)))
    }

    fn legacy_checkpoint_assets_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}.assets", sanitize_name(name)))
    }

    fn branches_dir(&self) -> PathBuf {
        self.dir.join(BRANCHES_DIR)
    }

    fn branch_path(&self, name: &str) -> PathBuf {
        self.branches_dir().join(sanitize_name(name))
    }

    fn aliases_default_branch_unlocked(&self, branch: &str) -> io::Result<bool> {
        if branch == DEFAULT_BRANCH_NAME {
            return Ok(true);
        }
        if !self.branch_path(branch).exists() || !self.branch_path(DEFAULT_BRANCH_NAME).exists() {
            return Ok(false);
        }
        let exact_entry_exists = fs::read_dir(self.branches_dir())?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name == branch)
        });
        Ok(!exact_entry_exists)
    }

    fn initialize_default_branch_unlocked(&self) -> io::Result<()> {
        if self.dir.exists() {
            let head_path = self.dir.join(HEAD_REF_FILE);
            if !head_path.exists() && self.list_unlocked()?.is_empty() {
                self.reset_to_unborn_default_branch_unlocked()?;
                self.clear_base_unlocked()?;
            }
            return Ok(());
        }
        self.reset_to_unborn_default_branch_unlocked()
    }

    fn reset_to_unborn_default_branch_unlocked(&self) -> io::Result<()> {
        fs::create_dir_all(self.branches_dir())?;
        self.write_ref_unlocked(&self.branch_path(DEFAULT_BRANCH_NAME), "")?;
        if let Err(error) = self.set_attached_head_unlocked(DEFAULT_BRANCH_NAME) {
            let _ = fs::remove_file(self.branch_path(DEFAULT_BRANCH_NAME));
            return Err(error);
        }
        Ok(())
    }

    /// Save the current session history as a checkpoint named `name`.
    /// Returns the checkpoint file path.
    pub(in crate::ai) fn save(&self, name: &str) -> io::Result<PathBuf> {
        self.save_with_message(name, None)
    }

    /// Save the current session history with an optional commit message.
    pub(in crate::ai) fn save_with_message(
        &self,
        name: &str,
        message: Option<&str>,
    ) -> io::Result<PathBuf> {
        let normalized_name = sanitize_name(name);
        if is_reserved_checkpoint_name(name) || is_reserved_checkpoint_name(&normalized_name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "'{normalized_name}' is reserved for a checkpoint pointer; \
                     pick another checkpoint name"
                ),
            ));
        }
        self.with_locked(|store| {
            if !store.session_file.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "current session has no history yet; nothing to checkpoint",
                ));
            }

            let checkpoints = store.list_unlocked()?;
            let branches = store.list_branches_unlocked()?;
            if store
                .checkpoint_source_unlocked(&normalized_name)?
                .is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "checkpoint '{normalized_name}' already exists; checkpoint commits are immutable"
                    ),
                ));
            }
            let parent = store.base_target_unlocked()?;
            let refs = store.capture_ref_snapshot_unlocked()?;
            let branch_growth = store.active_branch_growth_unlocked(&normalized_name)?;
            let current_size = checkpoints
                .iter()
                .fold(0u64, |total, checkpoint| {
                    total.saturating_add(checkpoint.storage_bytes)
                })
                .saturating_add(store.branch_storage_size_unlocked(&branches)?);
            let staged =
                store.stage_generation(&normalized_name, parent.as_deref(), message)?;
            let sqlite = staged.path.join(GENERATION_SQLITE_FILE);
            let logical_base = checkpoint_storage_size(
                &sqlite,
                &staged.path.join(GENERATION_ASSETS_DIR),
            )?;
            let storage_base = sqlite_storage_size(&sqlite)?
                .saturating_add(staged.copied_asset_bytes);
            // Cache the size and message count in the staged manifest so later
            // `list` skips the recursive walk and the dirty check can skip
            // opening the snapshot SQLite on a count mismatch. Logical size is
            // displayed; storage size charges only bytes materialized instead
            // of COW-cloned from the live assets.
            let staged_messages = message_count_sqlite_read_only(&sqlite).ok();
            let saved_at = read_generation_manifest(&staged.path)
                .and_then(|manifest| manifest.saved_at)
                .unwrap_or_else(unix_now_secs);
            let (manifest, _staged_size, staged_storage) = generation_manifest_with_sizes(
                parent.as_deref(),
                saved_at,
                logical_base,
                storage_base,
                staged_messages,
                message,
            );
            fs::write(staged.path.join(GENERATION_MANIFEST_FILE), manifest)?;
            let mut result = validate_checkpoint_budget(
                checkpoints.len(),
                false,
                current_size,
                0,
                staged_storage.saturating_add(branch_growth),
                MAX_CHECKPOINTS_PER_SESSION,
                MAX_CHECKPOINT_STORAGE_BYTES,
            )
            .and_then(|()| store.mark_generation_publish_ready(&staged.path))
            .and_then(|()| store.publish_generation(&normalized_name, &staged.path));
            if result.is_ok()
                && let Err(pointer_error) = store.advance_after_save_unlocked(&normalized_name)
            {
                result = match store.restore_ref_snapshot_unlocked(&refs) {
                    Ok(()) => {
                        let _ = fs::remove_dir_all(store.generation_dir(&normalized_name));
                        Err(pointer_error)
                    }
                    Err(restore_error) => Err(io::Error::other(format!(
                        "saved checkpoint '{normalized_name}' but failed to advance its ref ({pointer_error}) and failed to restore the previous refs ({restore_error})"
                    ))),
                };
            }
            if result.is_err() {
                let _ = fs::remove_dir_all(&staged.path);
            }
            result.map(|()| store.checkpoint_path(&normalized_name))
        })
    }

    /// List all checkpoints of this session (newest first by modification
    /// time; size includes assets).
    pub(in crate::ai) fn list(&self) -> io::Result<Vec<CheckpointInfo>> {
        self.with_locked(|store| store.list_unlocked())
    }

    /// Roll the checkpoint named `name` back into the current session
    /// (overwrites the live history).
    pub(in crate::ai) fn rollback(&self, name: &str) -> io::Result<()> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            let refs = store.capture_ref_snapshot_unlocked()?;
            let old_head_raw = store.head_raw_unlocked()?;
            if let Err(error) = store.set_detached_current_unlocked(&normalized_name) {
                return match store.restore_ref_snapshot_unlocked(&refs) {
                    Ok(()) => Err(error),
                    Err(restore_error) => Err(io::Error::other(format!(
                        "failed to move HEAD to '{normalized_name}' ({error}) and failed to restore the previous refs ({restore_error})"
                    ))),
                };
            }
            if let Err(error) = store.restore_checkpoint_unlocked(&normalized_name) {
                return match store.restore_ref_snapshot_unlocked(&refs) {
                    Ok(()) => Err(error),
                    Err(restore_error) => Err(io::Error::other(format!(
                        "failed to restore checkpoint '{normalized_name}' ({error}) and failed to restore the previous refs ({restore_error})"
                    ))),
                };
            }
            store.record_previous_head_unlocked(old_head_raw.as_deref(), &normalized_name);
            Ok(())
        })
    }

    /// Create a movable branch pointer at `start`. When omitted, use the
    /// current HEAD state; only a detached/unset HEAD falls back to the live
    /// base. An attached unborn branch creates another unborn branch. Creating
    /// a branch never moves HEAD; checkout/switch owns that transition.
    pub(in crate::ai) fn create_branch(&self, name: &str, start: Option<&str>) -> io::Result<()> {
        let branch = sanitize_name(name);
        self.with_locked(|store| {
            let target = store.prepare_branch_unlocked(name, &branch, start)?;
            store.write_ref_unlocked(
                &store.branch_path(&branch),
                target.as_deref().unwrap_or_default(),
            )
        })
    }

    /// Validate a branch creation and resolve its start without mutating refs.
    pub(in crate::ai) fn prepare_branch(
        &self,
        name: &str,
        start: Option<&str>,
    ) -> io::Result<Option<String>> {
        let branch = sanitize_name(name);
        self.with_locked(|store| store.prepare_branch_unlocked(name, &branch, start))
    }

    pub(in crate::ai) fn list_branches(&self) -> io::Result<Vec<CheckpointBranch>> {
        self.with_locked(|store| store.list_branches_unlocked())
    }

    /// Whether this session has checkpoint state that must survive empty-live
    /// session cleanup. The default unborn `main` scaffold alone is disposable.
    pub(in crate::ai) fn has_persisted_state(&self) -> io::Result<bool> {
        self.with_locked(|store| {
            if !store.list_unlocked()?.is_empty() {
                return Ok(true);
            }
            if store
                .list_branches_unlocked()?
                .iter()
                .any(|branch| branch.name != DEFAULT_BRANCH_NAME || branch.target.is_some())
            {
                return Ok(true);
            }

            let default_head = format!("{HEAD_BRANCH_PREFIX}{DEFAULT_BRANCH_NAME}");
            let head = Self::read_optional_ref_unlocked(&store.dir.join(HEAD_REF_FILE))?;
            if head.as_deref().is_some_and(|head| head != default_head) {
                return Ok(true);
            }
            Ok(
                Self::read_optional_ref_unlocked(&store.dir.join(BASE_REF_FILE))?.is_some()
                    || Self::read_optional_ref_unlocked(
                        &store.dir.join(PREV_HEAD_REF_FILE),
                    )?
                    .is_some(),
            )
        })
    }

    /// Return a compact status label only when every displayed checkpoint ref
    /// resolves to a stored checkpoint object.
    pub(in crate::ai) fn current_status_label(&self) -> io::Result<Option<String>> {
        self.with_locked(|store| {
            let head = match store.head_state_unlocked()? {
                CheckpointHead::Detached(None) => match store.base_target_unlocked()? {
                    Some(base) if store.checkpoint_source_unlocked(&base)?.is_some() => {
                        CheckpointHead::Detached(Some(base))
                    }
                    _ => CheckpointHead::Detached(None),
                },
                head => head,
            };
            let target = match &head {
                CheckpointHead::Attached { target, .. } | CheckpointHead::Detached(target) => {
                    target.as_deref()
                }
            };
            if let Some(target) = target
                && store.checkpoint_source_unlocked(target)?.is_none()
            {
                return Ok(None);
            }
            Ok(checkpoint_status_label(&head))
        })
    }

    /// Delete a branch pointer without deleting its checkpoint history.
    pub(in crate::ai) fn delete_branch(&self, name: &str) -> io::Result<bool> {
        let branch = sanitize_name(name);
        self.with_locked(|store| {
            if store.aliases_default_branch_unlocked(&branch)? {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("cannot delete protected default branch '{DEFAULT_BRANCH_NAME}'"),
                ));
            }
            if store.head_branch_unlocked()?.as_deref() == Some(branch.as_str()) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("cannot delete the checked-out branch '{branch}'"),
                ));
            }
            match fs::remove_file(store.branch_path(&branch)) {
                Ok(()) => Ok(true),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
                Err(error) => Err(error),
            }
        })
    }

    /// Attach HEAD to `name` without restoring its target. The caller must
    /// already have verified that the live history matches the branch target.
    pub(in crate::ai) fn attach_branch(&self, name: &str) -> io::Result<()> {
        let branch = sanitize_name(name);
        self.with_locked(|store| {
            store.branch_target_unlocked(&branch)?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("branch '{branch}' not found"),
                )
            })?;
            let old_head_raw = store.head_raw_unlocked()?;
            store.set_attached_head_unlocked(&branch)?;
            let new_raw = format!("{HEAD_BRANCH_PREFIX}{branch}");
            store.record_previous_head_unlocked(old_head_raw.as_deref(), &new_raw);
            Ok(())
        })
    }

    /// Restore a branch target into the live session and attach HEAD to it.
    pub(in crate::ai) fn switch_branch(&self, name: &str) -> io::Result<()> {
        let branch = sanitize_name(name);
        self.with_locked(|store| {
            let target = store.branch_target_unlocked(&branch)?.ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("branch '{branch}' not found"))
            })?;
            let Some(target) = target else {
                if matches!(
                    store.head_state_unlocked()?,
                    CheckpointHead::Attached {
                        branch: current,
                        target: None,
                    } if current == branch
                ) {
                    return store.clear_base_unlocked();
                }
                let old_head_raw = store.head_raw_unlocked()?;
                let transaction = store.stage_empty_live_rollback()?;
                store.complete_live_rollback(&transaction)?;
                store.set_attached_head_unlocked(&branch)?;
                store.clear_base_unlocked()?;
                let new_raw = format!("{HEAD_BRANCH_PREFIX}{branch}");
                store.record_previous_head_unlocked(old_head_raw.as_deref(), &new_raw);
                return Ok(());
            };
            let previous = store.base_target_unlocked()?;
            let refs = store.capture_ref_snapshot_unlocked()?;
            let old_head_raw = store.head_raw_unlocked()?;
            store.restore_checkpoint_unlocked(&target)?;
            if let Err(error) = store
                .set_attached_head_unlocked(&branch)
                .and_then(|()| store.set_base_unlocked(&target))
            {
                let refs_error = store.restore_ref_snapshot_unlocked(&refs).err();
                let mut live_error = None;
                if let Some(previous) = previous {
                    live_error = store.restore_checkpoint_unlocked(&previous).err();
                }
                if refs_error.is_some() || live_error.is_some() {
                    return Err(io::Error::other(format!(
                        "failed to switch branch '{branch}' ({error}); ref rollback: {}; live rollback: {}",
                        refs_error
                            .map(|error| error.to_string())
                            .unwrap_or_else(|| "ok".to_string()),
                        live_error
                            .map(|error| error.to_string())
                            .unwrap_or_else(|| "ok".to_string())
                    )));
                }
                return Err(error);
            }
            let new_raw = format!("{HEAD_BRANCH_PREFIX}{branch}");
            store.record_previous_head_unlocked(old_head_raw.as_deref(), &new_raw);
            Ok(())
        })
    }

    /// Delete the checkpoint named `name`. Returns whether any file was
    /// actually deleted.
    pub(in crate::ai) fn delete(&self, name: &str) -> io::Result<bool> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            let branches = store.branches_pointing_to_unlocked(&normalized_name)?;
            if !branches.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "checkpoint '{normalized_name}' is referenced by branch{} {}; delete the branch first",
                        if branches.len() == 1 { "" } else { "es" },
                        branches.join(", ")
                    ),
                ));
            }
            let generation = store.generation_dir(&normalized_name);
            let mut deleted = false;
            match fs::remove_dir_all(generation) {
                Ok(()) => deleted = true,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            let deleted = store.remove_legacy_checkpoint(&normalized_name)? || deleted;
            if deleted {
                // Dropping history is only allowed through explicit delete:
                // when the deleted checkpoint was the HEAD target, drop the
                // pointer as well so it never dangles.
                store.clear_head_if_points_to_unlocked(&normalized_name)?;
                store.clear_previous_if_points_to_unlocked(&normalized_name)?;
                // The live base is intentionally left alone: the live history
                // still derives from the deleted checkpoint, so the next save
                // shows up as an orphan instead of a false root.
            }
            Ok(deleted)
        })
    }

    /// Delete every checkpoint of this session (generation dirs, legacy
    /// files, and the HEAD pointer). The live history is untouched.
    /// Returns the number of checkpoints removed. The live-base pointer is
    /// kept on purpose, mirroring `delete`: the live history still derives
    /// from a now-deleted checkpoint, so the next save shows up as an
    /// orphan instead of a false root.
    pub(in crate::ai) fn clear(&self) -> io::Result<usize> {
        self.with_locked(|store| {
            let count = store.list_unlocked()?.len();
            for entry in fs::read_dir(&store.dir)? {
                let path = entry?.path();
                if path.file_name().and_then(|name| name.to_str()) == Some(BASE_REF_FILE) {
                    continue;
                }
                let remove_result = if path.is_dir() {
                    fs::remove_dir_all(&path)
                } else {
                    fs::remove_file(&path)
                };
                match remove_result {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
            }
            store.reset_to_unborn_default_branch_unlocked()?;
            Ok(count)
        })
    }

    /// Read the resolved HEAD checkpoint target.
    /// Returns `None` only when no checkpoint has been saved or restored yet.
    pub(in crate::ai) fn head_target(&self) -> io::Result<Option<String>> {
        self.with_locked(|store| store.head_target_unlocked())
    }

    pub(in crate::ai) fn head_state(&self) -> io::Result<CheckpointHead> {
        self.with_locked(|store| store.head_state_unlocked())
    }

    /// Read the previous HEAD state for Git-style `switch -` / `checkout -`.
    /// Returns `None` until the first successful branch/checkout navigation.
    pub(in crate::ai) fn previous_head_state(&self) -> io::Result<Option<CheckpointHead>> {
        self.with_locked(|store| store.previous_head_state_unlocked())
    }

    /// Point HEAD at `name` (sanitized). Used to heal legacy or dangling
    /// pointers; normal save and restore operations move BASE and HEAD together.
    pub(in crate::ai) fn set_head(&self, name: &str) -> io::Result<()> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            fs::create_dir_all(&store.dir)?;
            fs::write(
                store.dir.join(HEAD_REF_FILE),
                format!("{normalized_name}\n"),
            )?;
            Ok(())
        })
    }

    fn head_target_unlocked(&self) -> io::Result<Option<String>> {
        match fs::read_to_string(self.dir.join(HEAD_REF_FILE)) {
            Ok(content) => {
                let target = content.trim().to_string();
                if target.is_empty() {
                    Ok(None)
                } else if let Some(branch) = target.strip_prefix(HEAD_BRANCH_PREFIX) {
                    Ok(self.branch_target_unlocked(branch)?.flatten())
                } else {
                    Ok(Some(target))
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn head_branch_unlocked(&self) -> io::Result<Option<String>> {
        match fs::read_to_string(self.dir.join(HEAD_REF_FILE)) {
            Ok(content) => Ok(content
                .trim()
                .strip_prefix(HEAD_BRANCH_PREFIX)
                .filter(|branch| !branch.is_empty())
                .map(str::to_string)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn head_state_unlocked(&self) -> io::Result<CheckpointHead> {
        if let Some(branch) = self.head_branch_unlocked()? {
            let target = self.branch_target_unlocked(&branch)?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("HEAD points to missing branch '{branch}'"),
                )
            })?;
            Ok(CheckpointHead::Attached { branch, target })
        } else {
            Ok(CheckpointHead::Detached(self.head_target_unlocked()?))
        }
    }

    fn clear_head_if_points_to_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        match self.head_target_unlocked()? {
            Some(target) if target == normalized_name => {
                match fs::remove_file(self.dir.join(HEAD_REF_FILE)) {
                    Ok(()) => Ok(()),
                    Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
                    Err(err) => Err(err),
                }
            }
            _ => Ok(()),
        }
    }

    /// Read the live-base pointer (which checkpoint the live history was last
    /// saved from or rolled back to). Returns `None` when nothing has been
    /// saved or rolled back yet.
    pub(in crate::ai) fn base_target(&self) -> io::Result<Option<String>> {
        self.with_locked(|store| store.base_target_unlocked())
    }

    fn base_target_unlocked(&self) -> io::Result<Option<String>> {
        if let Some(target) = self.head_target_unlocked()? {
            return Ok(Some(target));
        }
        match fs::read_to_string(self.dir.join(BASE_REF_FILE)) {
            Ok(content) => {
                let target = content.trim().to_string();
                if target.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(target))
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn set_base_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        self.write_ref_unlocked(&self.dir.join(BASE_REF_FILE), normalized_name)
    }

    fn clear_base_unlocked(&self) -> io::Result<()> {
        match fs::remove_file(self.dir.join(BASE_REF_FILE)) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }

    fn set_detached_current_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        self.write_ref_unlocked(&self.dir.join(HEAD_REF_FILE), normalized_name)?;
        self.set_base_unlocked(normalized_name)
    }

    fn set_attached_head_unlocked(&self, branch: &str) -> io::Result<()> {
        self.write_ref_unlocked(
            &self.dir.join(HEAD_REF_FILE),
            &format!("{HEAD_BRANCH_PREFIX}{branch}"),
        )
    }

    fn write_ref_unlocked(&self, path: &Path, value: &str) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temp = path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(&temp, format!("{value}\n"))?;
        match fs::rename(&temp, path) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_file(temp);
                Err(error)
            }
        }
    }

    fn read_optional_ref_unlocked(path: &Path) -> io::Result<Option<String>> {
        match fs::read_to_string(path) {
            Ok(content) => Ok(Some(content.trim().to_string())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Raw `HEAD.ref` content (`ref: branches/<name>` when attached).
    fn head_raw_unlocked(&self) -> io::Result<Option<String>> {
        match Self::read_optional_ref_unlocked(&self.dir.join(HEAD_REF_FILE))? {
            Some(raw) if !raw.is_empty() => Ok(Some(raw)),
            _ => Ok(None),
        }
    }

    fn previous_head_state_unlocked(&self) -> io::Result<Option<CheckpointHead>> {
        let raw = match Self::read_optional_ref_unlocked(&self.dir.join(PREV_HEAD_REF_FILE))? {
            Some(raw) if !raw.is_empty() => raw,
            _ => return Ok(None),
        };
        if let Some(branch) = raw.strip_prefix(HEAD_BRANCH_PREFIX) {
            // A deleted branch still reports its name so the caller can say
            // where `-` used to point instead of pretending nothing happened.
            let target = match self.branch_target_unlocked(branch)? {
                Some(target) => target,
                None => None,
            };
            return Ok(Some(CheckpointHead::Attached {
                branch: branch.to_string(),
                target,
            }));
        }
        Ok(Some(CheckpointHead::Detached(Some(raw))))
    }

    /// Remember `old_raw` as the previous HEAD, but only when it differs from
    /// the new raw HEAD value: no-op navigations must not clobber `-`.
    /// A missing old value means there is nothing to go back to, so the
    /// previous pointer is left untouched.
    /// This auxiliary navigation record is best-effort: HEAD and live history
    /// have already committed, so failure must not report a failed checkout.
    /// Atomic ref writes preserve the old pointer on failure; `-` may stay stale.
    fn record_previous_head_unlocked(&self, old_raw: Option<&str>, new_raw: &str) {
        if let Some(old) = old_raw.filter(|old| !old.is_empty() && *old != new_raw) {
            let _ = self.write_ref_unlocked(&self.dir.join(PREV_HEAD_REF_FILE), old);
        }
    }

    /// Drop the previous pointer when it resolves to the deleted checkpoint.
    /// A previous attached branch is left alone: the caller reports the
    /// dangling branch by name instead.
    fn clear_previous_if_points_to_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        let previous_points_here =
            matches!(self.previous_head_state_unlocked()?, Some(CheckpointHead::Detached(Some(target))) if target == normalized_name);
        if previous_points_here {
            match fs::remove_file(self.dir.join(PREV_HEAD_REF_FILE)) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    fn capture_ref_snapshot_unlocked(&self) -> io::Result<RefSnapshot> {
        let head_path = self.dir.join(HEAD_REF_FILE);
        let head = Self::read_optional_ref_unlocked(&head_path)?;
        let base = Self::read_optional_ref_unlocked(&self.dir.join(BASE_REF_FILE))?;
        let branch = head
            .as_deref()
            .and_then(|value| value.strip_prefix(HEAD_BRANCH_PREFIX))
            .map(|name| {
                let path = self.branch_path(name);
                Self::read_optional_ref_unlocked(&path).map(|value| (path, value))
            })
            .transpose()?;
        Ok(RefSnapshot { head, base, branch })
    }

    fn restore_optional_ref_unlocked(&self, path: &Path, value: Option<&str>) -> io::Result<()> {
        match value {
            Some(value) => self.write_ref_unlocked(path, value),
            None => match fs::remove_file(path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error),
            },
        }
    }

    fn restore_ref_snapshot_unlocked(&self, refs: &RefSnapshot) -> io::Result<()> {
        if let Some((path, value)) = &refs.branch {
            self.restore_optional_ref_unlocked(path, value.as_deref())?;
        }
        self.restore_optional_ref_unlocked(&self.dir.join(HEAD_REF_FILE), refs.head.as_deref())?;
        self.restore_optional_ref_unlocked(&self.dir.join(BASE_REF_FILE), refs.base.as_deref())
    }

    fn active_branch_growth_unlocked(&self, target: &str) -> io::Result<u64> {
        let Some(branch) = self.head_branch_unlocked()? else {
            return Ok(0);
        };
        let old_size = fs::metadata(self.branch_path(&branch))
            .map(|metadata| metadata.len())
            .or_else(|error| {
                if error.kind() == io::ErrorKind::NotFound {
                    Ok(0)
                } else {
                    Err(error)
                }
            })?;
        Ok((target.len().saturating_add(1) as u64).saturating_sub(old_size))
    }

    fn advance_after_save_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        if let Some(branch) = self.head_branch_unlocked()? {
            self.write_ref_unlocked(&self.branch_path(&branch), normalized_name)?;
        } else {
            self.write_ref_unlocked(&self.dir.join(HEAD_REF_FILE), normalized_name)?;
        }
        self.set_base_unlocked(normalized_name)
    }

    /// The outer option reports whether the branch exists; the inner option
    /// is its checkpoint target (`None` for an unborn branch).
    fn branch_target_unlocked(&self, branch: &str) -> io::Result<Option<Option<String>>> {
        match fs::read_to_string(self.branch_path(branch)) {
            Ok(content) => {
                let target = content.trim().to_string();
                if target.is_empty() {
                    Ok(Some(None))
                } else {
                    Ok(Some(Some(target)))
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn list_branches_unlocked(&self) -> io::Result<Vec<CheckpointBranch>> {
        let mut branches = Vec::new();
        let entries = match fs::read_dir(self.branches_dir()) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(branches),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let target = fs::read_to_string(entry.path())?.trim().to_string();
            let target = (!target.is_empty()).then_some(target);
            let is_dangling = match target.as_deref() {
                Some(target) => self.checkpoint_source_unlocked(target)?.is_none(),
                None => false,
            };
            branches.push(CheckpointBranch {
                name,
                target,
                is_dangling,
            });
        }
        branches.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(branches)
    }

    fn branches_pointing_to_unlocked(&self, target: &str) -> io::Result<Vec<String>> {
        Ok(self
            .list_branches_unlocked()?
            .into_iter()
            .filter(|branch| branch.target.as_deref() == Some(target))
            .map(|branch| branch.name)
            .collect())
    }

    fn branch_storage_size_unlocked(&self, branches: &[CheckpointBranch]) -> io::Result<u64> {
        branches.iter().try_fold(0u64, |total, branch| {
            Ok(total.saturating_add(fs::metadata(self.branch_path(&branch.name))?.len()))
        })
    }

    fn prepare_branch_unlocked(
        &self,
        name: &str,
        branch: &str,
        start: Option<&str>,
    ) -> io::Result<Option<String>> {
        if is_reserved_checkpoint_name(name) || is_reserved_checkpoint_name(branch) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("'{branch}' is reserved and cannot be used as a branch name"),
            ));
        }
        if self.branch_path(branch).exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("branch '{branch}' already exists"),
            ));
        }
        let target = match start {
            Some(start) => Some(self.resolve_revision_unlocked(start)?),
            None => match self.head_state_unlocked()? {
                CheckpointHead::Attached { target, .. } => target,
                CheckpointHead::Detached(Some(target)) => Some(target),
                CheckpointHead::Detached(None) => self.base_target_unlocked()?,
            },
        };
        if let Some(target) = target.as_deref()
            && self.checkpoint_source_unlocked(target)?.is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("cannot create branch '{branch}': checkpoint '{target}' does not exist"),
            ));
        }
        let checkpoints = self.list_unlocked()?;
        let branches = self.list_branches_unlocked()?;
        validate_checkpoint_budget(
            checkpoints.len().saturating_add(branches.len()),
            false,
            checkpoints
                .iter()
                .fold(0u64, |total, checkpoint| {
                    total.saturating_add(checkpoint.storage_bytes)
                })
                .saturating_add(self.branch_storage_size_unlocked(&branches)?),
            0,
            target
                .as_ref()
                .map_or(1, |target| target.len().saturating_add(1)) as u64,
            MAX_CHECKPOINTS_PER_SESSION,
            MAX_CHECKPOINT_STORAGE_BYTES,
        )?;
        Ok(target)
    }

    fn resolve_revision_unlocked(&self, name: &str) -> io::Result<String> {
        let normalized = sanitize_name(name);
        if let Some(target) = self.branch_target_unlocked(&normalized)? {
            let target = target.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("branch '{normalized}' has no checkpoint yet"),
                )
            })?;
            if self.checkpoint_source_unlocked(&target)?.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("branch '{normalized}' points to missing checkpoint '{target}'"),
                ));
            }
            return Ok(target);
        }
        if self.checkpoint_source_unlocked(&normalized)?.is_some() {
            return Ok(normalized);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("checkpoint or branch '{normalized}' not found"),
        ))
    }

    fn restore_checkpoint_unlocked(&self, normalized_name: &str) -> io::Result<()> {
        let Some(source) = self.checkpoint_source_unlocked(normalized_name)? else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("checkpoint '{normalized_name}' not found"),
            ));
        };
        match source {
            CheckpointSource::Legacy { sqlite } => {
                if let Some(parent) = self.session_file.parent() {
                    fs::create_dir_all(parent)?;
                }
                restore_sqlite_after_rollback(&sqlite, &self.session_file)
            }
            CheckpointSource::Generation { sqlite, .. } => {
                let transaction = self.stage_live_rollback(&sqlite, Some(normalized_name))?;
                self.complete_live_rollback(&transaction)
            }
        }
    }

    /// Read the recorded parent of one generation directory: `None` when the
    /// checkpoint predates parent tracking (v1 manifest) or the manifest is
    /// missing/unparseable; otherwise the v2 parent (`None` for a root).
    fn generation_parent_unlocked(
        &self,
        normalized_name: &str,
    ) -> io::Result<Option<Option<String>>> {
        Ok(read_generation_parent(
            &self.generation_dir(normalized_name),
        ))
    }

    /// Recover unfinished live rollbacks at startup and before every
    /// checkpoint operation.
    pub(super) fn recover(&self) -> io::Result<()> {
        self.with_locked(|_| Ok(()))
    }

    fn with_locked<T>(&self, operation: impl FnOnce(&Self) -> io::Result<T>) -> io::Result<T> {
        with_checkpoint_lock(&self.dir, || {
            self.recover_live_rollbacks_unlocked()?;
            self.recover_incomplete_generations_unlocked()?;
            self.initialize_default_branch_unlocked()?;
            operation(self)
        })
    }

    /// Whether the live canonical messages match the named checkpoint.
    /// Runtime-owned assets may be compacted, archived, or regenerated without
    /// adding a turn, so asset-directory drift must not mint an auto checkpoint.
    /// The checkpoint database opens read-only, so comparing never bumps the
    /// directory mtime that `SAVED` displays. Returns `Ok(false)` when the
    /// checkpoint does not exist; I/O errors pass through so the caller can
    /// fall back to preserving (fail safe, never fail lossy).
    pub(in crate::ai) fn live_matches_checkpoint(
        &self,
        live: &[Message],
        name: &str,
    ) -> io::Result<bool> {
        self.with_locked(|store| store.live_matches_checkpoint_unlocked(live, name))
    }

    /// Whether live canonical messages still match the last restored or saved
    /// checkpoint. An unborn history is clean only while it remains empty.
    pub(in crate::ai) fn live_matches_base(&self, live: &[Message]) -> io::Result<bool> {
        self.with_locked(|store| {
            if matches!(
                store.head_state_unlocked()?,
                CheckpointHead::Attached { target: None, .. }
            ) {
                return Ok(live.is_empty());
            }
            let Some(base) = store.base_target_unlocked()? else {
                return Ok(live.is_empty());
            };
            store.live_matches_checkpoint_unlocked(live, &base)
        })
    }

    /// Summary of what switching to `name` would replace.
    /// Returns `(target_len, common_prefix_len)` where the prefix is the
    /// leading run of identical canonical messages shared by live and the
    /// target. Callers derive dropped as `live.len() - common` so prompts
    /// report messages actually lost instead of the live total. Returns
    /// `Ok(None)` when the target does not exist.
    pub(in crate::ai) fn checkpoint_live_diff(
        &self,
        live: &[Message],
        name: &str,
    ) -> io::Result<Option<(usize, usize)>> {
        self.with_locked(|store| store.checkpoint_live_diff_unlocked(live, name))
    }

    fn checkpoint_live_diff_unlocked(
        &self,
        live: &[Message],
        name: &str,
    ) -> io::Result<Option<(usize, usize)>> {
        let normalized = sanitize_name(name);
        let Some(source) = self.checkpoint_source_unlocked(&normalized)? else {
            return Ok(None);
        };
        let sqlite = match &source {
            CheckpointSource::Generation { sqlite, .. } | CheckpointSource::Legacy { sqlite } => {
                sqlite
            }
        };
        let target = read_all_messages_sqlite_read_only(sqlite)?;
        let common = live
            .iter()
            .zip(target.iter())
            .take_while(|(a, b)| a == b)
            .count();
        Ok(Some((target.len(), common)))
    }

    fn live_matches_checkpoint_unlocked(&self, live: &[Message], name: &str) -> io::Result<bool> {
        let normalized = sanitize_name(name);
        // Manifest count first: avoids opening the snapshot SQLite entirely
        // when a new turn already proves dirty. Falls back below for
        // pre-caching manifests and legacy layouts.
        if generation_is_ready(&self.generation_dir(&normalized))
            && let Some(cached) =
                read_generation_manifest(&self.generation_dir(&normalized))
                    .and_then(|manifest| manifest.message_count)
            && cached != live.len()
        {
            return Ok(false);
        }
        let Some(source) = self.checkpoint_source_unlocked(&normalized)? else {
            return Ok(false);
        };
        let sqlite = match &source {
            CheckpointSource::Generation { sqlite, .. } | CheckpointSource::Legacy { sqlite } => {
                sqlite
            }
        };
        // Fast path like `git diff --quiet`: a count mismatch already proves
        // dirty without deserializing every row. Fail open to the full compare
        // when the count query itself fails (missing table, corrupt DB).
        if let Ok(count) = message_count_sqlite_read_only(sqlite)
            && count != live.len()
        {
            return Ok(false);
        }
        Ok(read_all_messages_sqlite_read_only(sqlite)?.as_slice() == live)
    }

    fn list_unlocked(&self) -> io::Result<Vec<CheckpointInfo>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut out: Vec<(CheckpointInfo, Option<Option<String>>)> = Vec::new();
        let mut legacy_entries = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if metadata.is_dir() && generation_is_ready(&path) {
                let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                    continue;
                };
                // `None` here means pre-tracking (v1); the save-order fallback
                // below fills it in after sorting. `SAVED` shows when the
                // snapshot was taken, not the directory mtime: reads used to
                // open the snapshot sqlite read-write and SQLite drops
                // transient sidecars inside its directory, which bumps the
                // mtime long after the save and used to steal the `[latest]`
                // tag. Manifests without a timestamp fall back to the manifest
                // file mtime (creating sidecars never rewrites that file),
                // then to the directory mtime.
                let known = read_generation_manifest(&path);
                let manifest_mtime = fs::metadata(path.join(GENERATION_MANIFEST_FILE))
                    .and_then(|m| m.modified())
                    .ok()
                    .map(DateTime::<Local>::from);
                let mtime = metadata.modified().ok().map(DateTime::<Local>::from);
                // Prefer the size cached at save time; only pre-caching
                // generations pay the recursive walk.
                let size_bytes = match known.as_ref().and_then(|manifest| manifest.size_bytes) {
                    Some(cached) => cached,
                    None => directory_storage_size(&path)?,
                };
                let storage_bytes = known
                    .as_ref()
                    .and_then(|manifest| manifest.storage_bytes)
                    .unwrap_or(size_bytes);
                let message = known.as_ref().and_then(|manifest| manifest.message.clone());
                out.push((
                    CheckpointInfo {
                        name: name.to_string(),
                        message,
                        modified_local: manifest_saved_time(known.as_ref())
                            .or(manifest_mtime)
                            .or(mtime),
                        size_bytes,
                        storage_bytes,
                        parent: None,
                    },
                    known.map(|manifest| manifest.parent),
                ));
            } else if path.extension().and_then(|s| s.to_str()) == Some("sqlite") {
                legacy_entries.push((path, metadata));
            }
        }
        for (path, metadata) in legacy_entries {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if out.iter().any(|checkpoint| checkpoint.0.name == stem) {
                continue;
            }
            let size_bytes =
                checkpoint_storage_size(&path, &self.legacy_checkpoint_assets_path(stem))?;
            out.push((
                CheckpointInfo {
                    name: stem.to_string(),
                    message: None,
                    modified_local: metadata.modified().ok().map(DateTime::<Local>::from),
                    size_bytes,
                    storage_bytes: size_bytes,
                    parent: None,
                },
                None,
            ));
        }
        // Newest-first; break mtime ties by name so the order stays
        // deterministic (and matches timestamped `auto-*` names) instead of
        // depending on directory read order.
        out.sort_by(|a, b| {
            b.0.modified_local
                .cmp(&a.0.modified_local)
                .then_with(|| b.0.name.cmp(&a.0.name))
        });
        // Fill in parents: recorded v2 links win, with self-links (a replaced
        // checkpoint re-saved from itself) demoted to roots so the forest can
        // never cycle. Pre-tracking snapshots have no lineage on disk, so they
        // chain in save order and the tree still reads top-down.
        let names: Vec<String> = out.iter().map(|(cp, _)| cp.name.clone()).collect();
        for (index, (cp, known)) in out.iter_mut().enumerate() {
            cp.parent = match known {
                Some(Some(parent)) if parent == &cp.name => None,
                Some(Some(parent)) => Some(parent.clone()),
                Some(None) => None,
                None => names.get(index + 1).cloned(),
            };
        }
        Ok(out.into_iter().map(|(cp, _)| cp).collect())
    }

    fn checkpoint_source_unlocked(&self, name: &str) -> io::Result<Option<CheckpointSource>> {
        let generation = self.generation_dir(name);
        if generation_is_ready(&generation) {
            return Ok(Some(CheckpointSource::Generation {
                sqlite: generation.join(GENERATION_SQLITE_FILE),
                assets: generation.join(GENERATION_ASSETS_DIR),
            }));
        }
        let legacy = self.legacy_checkpoint_path(name);
        if legacy.exists() {
            return Ok(Some(CheckpointSource::Legacy { sqlite: legacy }));
        }
        Ok(None)
    }

    /// Lightweight existence check for checkout validation. Unlike `list()`,
    /// this skips the recursive size walk, so switching stays O(1) instead of
    /// O(total checkpoint bytes) like a full `git status` would.
    pub(in crate::ai) fn checkpoint_exists(&self, name: &str) -> io::Result<bool> {
        let normalized = sanitize_name(name);
        self.with_locked(|store| Ok(store.checkpoint_source_unlocked(&normalized)?.is_some()))
    }

    fn checkpoint_storage_size_for_name(&self, name: &str) -> io::Result<u64> {
        let generation = self.generation_dir(name);
        let mut size = if generation_is_ready(&generation) {
            directory_storage_size(&generation)?
        } else {
            0
        };
        size = size.saturating_add(checkpoint_storage_size(
            &self.legacy_checkpoint_path(name),
            &self.legacy_checkpoint_assets_path(name),
        )?);
        Ok(size)
    }

    fn stage_generation(
        &self,
        name: &str,
        parent: Option<&str>,
        message: Option<&str>,
    ) -> io::Result<StagedGeneration> {
        fs::create_dir_all(&self.dir)?;
        let staged = self.dir.join(format!(
            ".{name}{GENERATION_STAGE_MARKER}{}",
            uuid::Uuid::new_v4()
        ));
        let result = (|| {
            fs::create_dir_all(&staged)?;
            backup_sqlite(&self.session_file, &staged.join(GENERATION_SQLITE_FILE))?;
            let copied_asset_bytes =
                copy_directory_snapshot(&self.session_assets, &staged.join(GENERATION_ASSETS_DIR))?;
            fs::write(
                staged.join(GENERATION_MANIFEST_FILE),
                generation_manifest_body(parent, unix_now_secs(), None, None, None, message),
            )?;
            Ok(copied_asset_bytes)
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staged);
        }
        result.map(|copied_asset_bytes| StagedGeneration {
            path: staged,
            copied_asset_bytes,
        })
    }

    fn publish_generation(&self, name: &str, staged: &Path) -> io::Result<()> {
        let destination = self.generation_dir(name);
        let previous = self.dir.join(format!(
            ".{name}{GENERATION_PREVIOUS_MARKER}{}",
            uuid::Uuid::new_v4()
        ));
        let had_previous = destination.exists();
        if had_previous {
            fs::rename(&destination, &previous)?;
        }
        if let Err(error) = fs::rename(staged, &destination) {
            if had_previous {
                let _ = fs::rename(&previous, &destination);
            }
            return Err(error);
        }
        if had_previous {
            let _ = fs::remove_dir_all(previous);
        }
        let _ = fs::remove_dir(destination.join(GENERATION_PUBLISH_READY_MARKER));
        // The new generation is now the single source of truth; the legacy
        // layout is only kept for compatible reads. A later cleanup failure
        // cannot break the published checkpoint; the next lock-held operation
        // reaps it again.
        let _ = self.remove_legacy_checkpoint(name);
        Ok(())
    }

    fn mark_generation_publish_ready(&self, staged: &Path) -> io::Result<()> {
        // Directory creation atomically publishes the approval after quota
        // validation. Recovery can distinguish it from an earlier interrupted
        // stage without adding bytes to the checkpoint's quota accounting.
        fs::create_dir(staged.join(GENERATION_PUBLISH_READY_MARKER))
    }

    fn remove_legacy_checkpoint(&self, name: &str) -> io::Result<bool> {
        let path = self.legacy_checkpoint_path(name);
        let assets = self.legacy_checkpoint_assets_path(name);
        let mut deleted = false;
        match fs::remove_file(&path) {
            Ok(()) => deleted = true,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        for suffix in SQLITE_SIDECAR_SUFFIXES {
            let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
            match fs::remove_file(sidecar) {
                Ok(()) => deleted = true,
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
        match fs::remove_dir_all(&assets) {
            Ok(()) => deleted = true,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        Ok(deleted)
    }

    fn stage_empty_live_rollback(&self) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let transaction = self
            .dir
            .join(format!("{LIVE_ROLLBACK_PREFIX}{}", uuid::Uuid::new_v4()));
        let result = (|| {
            let new_state = transaction.join("new");
            fs::create_dir_all(new_state.join(GENERATION_ASSETS_DIR))?;
            let sqlite = new_state.join(GENERATION_SQLITE_FILE);
            if self.session_file.is_file() {
                backup_sqlite(&self.session_file, &sqlite)?;
            }
            clear_session_history_sqlite(&sqlite)?;
            fs::write(
                transaction.join(LIVE_ROLLBACK_MANIFEST),
                b"commit\nsource: -\n",
            )?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&transaction);
        }
        result.map(|()| transaction)
    }

    fn stage_live_rollback(
        &self,
        sqlite: &Path,
        source_name: Option<&str>,
    ) -> io::Result<PathBuf> {
        // Only the SQLite snapshot is staged: generation assets are immutable,
        // so `complete` restores them directly from the source generation
        // (recorded in the manifest) instead of paying a second full-tree copy
        // checkpoint -> txn. This halves the assets I/O on every checkout.
        fs::create_dir_all(&self.dir)?;
        let transaction = self
            .dir
            .join(format!("{LIVE_ROLLBACK_PREFIX}{}", uuid::Uuid::new_v4()));
        let result = (|| {
            let new_state = transaction.join("new");
            fs::create_dir_all(&new_state)?;
            backup_sqlite(sqlite, &new_state.join(GENERATION_SQLITE_FILE))?;
            let source_line = match source_name {
                Some(name) => format!("commit\nsource: {name}\n"),
                None => "commit\nsource: -\n".to_string(),
            };
            fs::write(transaction.join(LIVE_ROLLBACK_MANIFEST), source_line)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&transaction);
        }
        result.map(|()| transaction)
    }

    fn complete_live_rollback(&self, transaction: &Path) -> io::Result<()> {
        let new_state = transaction.join("new");
        let sqlite = new_state.join(GENERATION_SQLITE_FILE);
        let staged_assets = new_state.join(GENERATION_ASSETS_DIR);
        let transaction_id = live_rollback_transaction_id(transaction)?;
        if !live_rollback_is_ready(transaction) {
            return Err(io::Error::other(format!(
                "incomplete checkpoint rollback transaction: {}",
                transaction.display()
            )));
        }
        if live_rollback_transaction_is_published(&self.session_file, &transaction_id)? {
            // Assets are restored before the marker-bearing SQLite DB is
            // published, so a matching marker means only cleanup remains.
            // Replaying either state would clobber post-rollback writes.
            cleanup_live_rollback_transaction(transaction);
            return Ok(());
        }
        if let Some(parent) = self.session_file.parent() {
            fs::create_dir_all(parent)?;
        }
        // Assets come straight from the source generation (immutable), not via
        // the transaction copy. Old transactions without a `source:` line keep
        // restoring from their staged copy; `source: -` means an empty target
        // and skips the restore (same no-op as restoring an empty dir).
        match live_rollback_source_name(transaction)? {
            Some(name) => {
                let source_assets = self.generation_dir(&name).join(GENERATION_ASSETS_DIR);
                if generation_is_ready(&self.generation_dir(&name)) {
                    restore_checkpoint_assets(&source_assets, &self.session_assets)?;
                } else if staged_assets.is_dir() {
                    restore_checkpoint_assets(&staged_assets, &self.session_assets)?;
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("checkpoint '{name}' not found"),
                    ));
                }
            }
            None if staged_assets.is_dir() => {
                restore_checkpoint_assets(&staged_assets, &self.session_assets)?;
            }
            None => {}
        }
        restore_staged_sqlite_after_rollback(
            &sqlite,
            &self.session_file,
            &transaction_id,
        )?;
        cleanup_live_rollback_transaction(transaction);
        Ok(())
    }

    fn recover_live_rollbacks_unlocked(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !entry.file_type()?.is_dir() || !name.starts_with(LIVE_ROLLBACK_PREFIX) {
                continue;
            }
            if live_rollback_is_ready(&path) {
                self.complete_live_rollback(&path)?;
            } else {
                // Staging writes the manifest last and cannot begin a SQLite
                // replacement until the staged source is complete. A malformed
                // directory therefore has no rollback to resume and must not
                // block every later checkpoint operation.
                fs::remove_dir_all(path)?;
            }
        }
        Ok(())
    }

    fn recover_incomplete_generations_unlocked(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries.collect::<Result<Vec<_>, _>>()?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut staged = Vec::new();
        let mut previous = Vec::new();
        for entry in &entries {
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if let Some(base) = staged_generation_base(name, GENERATION_STAGE_MARKER) {
                staged.push((base.to_string(), entry.path()));
            } else if let Some(base) = staged_generation_base(name, GENERATION_PREVIOUS_MARKER) {
                previous.push((base.to_string(), entry.path()));
            }
        }

        for (name, stage) in staged {
            if !generation_is_ready(&stage) || !generation_is_publish_ready(&stage) {
                fs::remove_dir_all(stage)?;
                continue;
            }
            let destination = self.generation_dir(&name);
            if destination.exists() {
                fs::remove_dir_all(stage)?;
                continue;
            }
            fs::rename(&stage, &destination)?;
            let _ = fs::remove_dir(destination.join(GENERATION_PUBLISH_READY_MARKER));
            if let Some(index) = previous
                .iter()
                .position(|(previous_name, _)| previous_name == &name)
            {
                let (_, old) = previous.swap_remove(index);
                let _ = fs::remove_dir_all(old);
            }
        }

        for (name, old) in previous {
            let destination = self.generation_dir(&name);
            if destination.exists() {
                fs::remove_dir_all(old)?;
            } else {
                fs::rename(old, destination)?;
            }
        }

        // Once a generation is published, leftover legacy `.sqlite` / `.assets`
        // are no longer recoverable state and must be reclaimed so they cannot
        // bypass quota accounting.
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            if generation_is_ready(&self.generation_dir(name)) {
                self.remove_legacy_checkpoint(name)?;
            }
        }
        Ok(())
    }
}

/// Hold the in-process and cross-process exclusive lock for one session's
/// checkpoint transactions.
pub(super) fn with_checkpoint_lock<T>(
    checkpoint_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_dir = checkpoint_lock_dir(checkpoint_dir);
    with_checkpoint_root_shared_lock(&lock_dir, || {
        with_session_checkpoint_lock(&lock_dir, checkpoint_dir, operation)
    })
}

/// Exclusively lock the checkpoint root while clearing all sessions, so
/// clearing cannot interleave with save, rollback, fork, or archive.
pub(super) fn with_checkpoint_root_exclusive_lock<T>(
    checkpoints_root: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    with_checkpoint_root_exclusive_lock_impl(
        &checkpoint_lock_dir_from_root(checkpoints_root),
        operation,
    )
}

/// Lock files live next to the checkpoint root; cleaning up snapshot
/// directories never unlinks a lock inode that is currently held.
fn checkpoint_lock_dir(checkpoint_dir: &Path) -> PathBuf {
    let checkpoints_root = checkpoint_dir.parent().unwrap_or_else(|| Path::new("."));
    checkpoint_lock_dir_from_root(checkpoints_root)
}

fn checkpoint_lock_dir_from_root(checkpoints_root: &Path) -> PathBuf {
    checkpoints_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".checkpoint-locks")
}

fn with_session_checkpoint_lock<T>(
    lock_dir: &Path,
    checkpoint_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    fs::create_dir_all(lock_dir)?;
    let name = checkpoint_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    let lock_path = lock_dir.join(format!("{name}.lock"));
    let local_lock = {
        let mut locks = CHECKPOINT_SESSION_LOCKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(
            locks
                .entry(lock_path.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    };
    let guard = local_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;

    #[cfg(unix)]
    unsafe {
        if flock(file.as_raw_fd(), LOCK_EX) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = flock(file.as_raw_fd(), LOCK_UN);
    }
    drop(file);
    drop(guard);
    result
}

/// When accessing multiple sessions at once, acquire locks in normalized
/// directory order so two forks in opposite directions cannot wait on each
/// other. Duplicate directories are deduplicated, so degrading to a single
/// session never re-enters the Mutex.
pub(super) fn with_checkpoint_locks<T>(
    checkpoint_dirs: &[&Path],
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let mut dirs = checkpoint_dirs
        .iter()
        .map(|dir| (*dir).to_path_buf())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs.dedup();
    let Some(first) = dirs.first() else {
        return operation();
    };
    let lock_dir = checkpoint_lock_dir(first);
    if dirs
        .iter()
        .any(|checkpoint_dir| checkpoint_lock_dir(checkpoint_dir) != lock_dir)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint sessions must share one root directory",
        ));
    }
    with_checkpoint_root_shared_lock(&lock_dir, || {
        with_checkpoint_locks_inner(&dirs, &lock_dir, &mut Some(operation))
    })
}

fn with_checkpoint_locks_inner<T, F>(
    checkpoint_dirs: &[PathBuf],
    lock_dir: &Path,
    operation: &mut Option<F>,
) -> io::Result<T>
where
    F: FnOnce() -> io::Result<T>,
{
    let Some((checkpoint_dir, remaining)) = checkpoint_dirs.split_first() else {
        return operation
            .take()
            .expect("checkpoint lock operation must run once")();
    };
    with_session_checkpoint_lock(lock_dir, checkpoint_dir, || {
        with_checkpoint_locks_inner(remaining, lock_dir, operation)
    })
}

fn with_checkpoint_root_shared_lock<T>(
    lock_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_path = lock_dir.join(".all.lock");
    let local_lock = checkpoint_root_local_lock(&lock_path);
    let guard = local_lock
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result = with_checkpoint_root_file_lock(&lock_path, false, operation);
    drop(guard);
    result
}

fn with_checkpoint_root_exclusive_lock_impl<T>(
    lock_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_path = lock_dir.join(".all.lock");
    let local_lock = checkpoint_root_local_lock(&lock_path);
    let guard = local_lock
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result = with_checkpoint_root_file_lock(&lock_path, true, operation);
    drop(guard);
    result
}

fn checkpoint_root_local_lock(lock_path: &Path) -> Arc<RwLock<()>> {
    let mut locks = CHECKPOINT_ROOT_LOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        locks
            .entry(lock_path.to_path_buf())
            .or_insert_with(|| Arc::new(RwLock::new(()))),
    )
}

fn with_checkpoint_root_file_lock<T>(
    lock_path: &Path,
    exclusive: bool,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let parent = lock_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    unsafe {
        let lock_mode = if exclusive { LOCK_EX } else { LOCK_SH };
        if flock(file.as_raw_fd(), lock_mode) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = flock(file.as_raw_fd(), LOCK_UN);
    }
    result
}

fn validate_checkpoint_budget(
    current_count: usize,
    replacing_existing: bool,
    current_size: u64,
    replaced_size: u64,
    snapshot_size: u64,
    max_count: usize,
    max_size: u64,
) -> io::Result<()> {
    if !replacing_existing && current_count >= max_count {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "checkpoint limit reached ({max_count} per session); delete an old checkpoint or save with an existing name to replace it"
            ),
        ));
    }

    let projected_size = current_size
        .saturating_sub(replaced_size)
        .saturating_add(snapshot_size);
    if projected_size > max_size {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "checkpoint storage limit reached ({} MiB per session; projected {} bytes); delete old checkpoints or reduce session assets before saving",
                max_size / (1024 * 1024),
                projected_size,
            ),
        ));
    }
    Ok(())
}

fn checkpoint_storage_size(sqlite: &Path, assets: &Path) -> io::Result<u64> {
    Ok(sqlite_storage_size(sqlite)?.saturating_add(directory_storage_size(assets)?))
}

fn sqlite_storage_size(path: &Path) -> io::Result<u64> {
    let mut size = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error),
    };
    for suffix in SQLITE_SIDECAR_SUFFIXES {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::metadata(sidecar) {
            Ok(metadata) => size = size.saturating_add(metadata.len()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(size)
}

fn directory_storage_size(path: &Path) -> io::Result<u64> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut size = 0u64;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            size = size.saturating_add(directory_storage_size(&entry.path())?);
        } else {
            size = size.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(size)
}

fn generation_is_ready(path: &Path) -> bool {
    path.join(GENERATION_MANIFEST_FILE).is_file()
        && path.join(GENERATION_SQLITE_FILE).is_file()
        && path.join(GENERATION_ASSETS_DIR).is_dir()
}

fn generation_is_publish_ready(path: &Path) -> bool {
    path.join(GENERATION_PUBLISH_READY_MARKER).is_dir()
}

fn cleanup_live_rollback_transaction(transaction: &Path) {
    if let Err(err) = fs::remove_dir_all(transaction) {
        eprintln!(
            "[checkpoint] rollback was published, but failed to remove transaction '{}': {err}",
            transaction.display()
        );
    }
}

fn live_rollback_is_ready(path: &Path) -> bool {
    // New transactions stage only SQLite (assets restore straight from the
    // source generation); old ones also staged an assets copy. Ready means the
    // manifest plus the staged SQLite exist.
    path.join(LIVE_ROLLBACK_MANIFEST).is_file()
        && path.join("new").join(GENERATION_SQLITE_FILE).is_file()
}

/// Source generation recorded in a live-rollback manifest (`source: <name>`).
/// Returns `Ok(None)` for the empty target (`source: -`) and for manifests
/// written before sources were tracked (those transactions carry their own
/// assets copy); `Err` only when the manifest itself is unreadable.
fn live_rollback_source_name(transaction: &Path) -> io::Result<Option<String>> {
    let content = fs::read_to_string(transaction.join(LIVE_ROLLBACK_MANIFEST))?;
    for line in content.lines().skip(1) {
        if let Some(name) = line.strip_prefix("source: ") {
            return Ok(if name == "-" || name.is_empty() {
                None
            } else {
                Some(name.to_string())
            });
        }
    }
    Ok(None)
}

fn live_rollback_transaction_id(transaction: &Path) -> io::Result<String> {
    let Some(name) = transaction.file_name().and_then(|name| name.to_str()) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "rollback transaction has no valid name: {}",
                transaction.display()
            ),
        ));
    };
    let Some(transaction_id) = name.strip_prefix(LIVE_ROLLBACK_PREFIX) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid rollback transaction name: {}",
                transaction.display()
            ),
        ));
    };
    if transaction_id.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "rollback transaction has an empty identifier: {}",
                transaction.display()
            ),
        ));
    }
    Ok(transaction_id.to_string())
}

fn staged_generation_base<'a>(name: &'a str, marker: &str) -> Option<&'a str> {
    name.strip_prefix('.')
        .and_then(|trimmed| trimmed.split_once(marker).map(|(base, _)| base))
        .filter(|base| !base.is_empty())
}

#[derive(Debug)]
struct StagedGeneration {
    path: PathBuf,
    copied_asset_bytes: u64,
}

fn copy_directory_snapshot(source: &Path, destination: &Path) -> io::Result<u64> {
    if destination.exists() {
        fs::remove_dir_all(destination)?;
    }
    if !source.is_dir() {
        fs::create_dir_all(destination)?;
        return Ok(0);
    }
    copy_directory_snapshot_impl(source, destination)
}

#[cfg(target_os = "macos")]
fn copy_directory_snapshot_impl(source: &Path, destination: &Path) -> io::Result<u64> {
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "asset path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "asset path contains NUL"))?;
    // COPYFILE_CLONE recursively creates independent COW files on APFS. If a
    // file cannot be cloned, copyfile safely falls back to a byte copy and
    // reports those materialized bytes through COPYFILE_STATE_COPIED.
    let state = unsafe { libc::copyfile_state_alloc() };
    if state.is_null() {
        return Err(io::Error::last_os_error());
    }
    let result = unsafe {
        libc::copyfile(
            source.as_ptr(),
            destination.as_ptr(),
            state,
            libc::COPYFILE_METADATA
                | libc::COPYFILE_DATA
                | libc::COPYFILE_RECURSIVE
                | libc::COPYFILE_CLONE,
        )
    };
    let copy_error = (result != 0).then(io::Error::last_os_error);
    let mut copied: libc::off_t = 0;
    let state_result = unsafe {
        libc::copyfile_state_get(
            state,
            libc::COPYFILE_STATE_COPIED as u32,
            (&mut copied as *mut libc::off_t).cast(),
        )
    };
    unsafe {
        libc::copyfile_state_free(state);
    }
    if let Some(error) = copy_error {
        return Err(error);
    }
    if state_result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::try_from(copied).unwrap_or(0))
}

#[cfg(not(target_os = "macos"))]
fn copy_directory_snapshot_impl(source: &Path, destination: &Path) -> io::Result<u64> {
    super::sessions::copy_dir_recursively(source, destination)?;
    directory_storage_size(destination)
}

/// Overflow assets are UUID-named and immutable after writing. Rollback only
/// restores files referenced by the checkpoint; it must not replace the live
/// directory wholesale: a concurrent writer may have created a new asset and
/// committed its stub to the canonical DB after the rollback, and deleting
/// that file would leave a `file_path` that can never be read back.
fn restore_checkpoint_assets(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    if source.is_dir() {
        copy_dir_skip_identical(source, destination)?;
    }
    Ok(())
}

/// Restore every snapshotted path. Session scratch files are mutable, so equal
/// length does not imply equal content. On APFS each replacement is another
/// COW clone rather than a full byte copy.
fn copy_dir_skip_identical(source: &Path, destination: &Path) -> io::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            fs::create_dir_all(&to)?;
            copy_dir_skip_identical(&from, &to)?;
        } else {
            replace_snapshot_file(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn replace_snapshot_file(source: &Path, destination: &Path) -> io::Result<()> {
    let temporary = destination.with_extension(format!("restore-{}", uuid::Uuid::new_v4()));
    let source_c = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "asset path contains NUL"))?;
    let temporary_c = CString::new(temporary.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "asset path contains NUL"))?;
    let cloned = unsafe { libc::clonefile(source_c.as_ptr(), temporary_c.as_ptr(), 0) } == 0;
    if cloned {
        if let Err(error) = fs::rename(&temporary, destination) {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        return Ok(());
    }
    let _ = fs::remove_file(&temporary);
    fs::copy(source, destination).map(|_| ())
}

#[cfg(not(target_os = "macos"))]
fn replace_snapshot_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::copy(source, destination).map(|_| ())
}

/// Normalize a checkpoint name into a safe file name (alphanumeric, `-`, `_`;
/// everything else becomes `_`).
fn sanitize_name(name: &str) -> String {
    // Keep in sync with `is_reserved_checkpoint_name` below.
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' || ch == '/' {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "checkpoint".to_string()
    } else {
        out
    }
}

/// A parsed `checkpoint-v2` manifest: the recorded parent link plus the save
/// timestamp. `saved_at` is `None` for manifests written before timestamps
/// were tracked; callers fall back to the manifest file mtime, then the directory mtime.
/// `size_bytes`, `storage_bytes`, and `message_count` are optional fast paths
/// cached at save time. `size_bytes` is the logical snapshot size while
/// `storage_bytes` is the newly allocated data charged to the quota. Like git
/// storing tree sizes in a commit, these caches let `list` skip the recursive
/// size walk and the dirty check skip opening the snapshot SQLite on a count
/// mismatch. `None` means a pre-caching manifest and falls back to the existing
/// walk/query behavior.
struct GenerationManifest {
    parent: Option<String>,
    message: Option<String>,
    saved_at: Option<u64>,
    size_bytes: Option<u64>,
    storage_bytes: Option<u64>,
    message_count: Option<usize>,
}

/// Parse a generation manifest into its parent link and save timestamp.
/// Returns `None` when the checkpoint predates parent tracking
/// (`checkpoint-v1`), has no manifest, or the manifest is unparseable.
fn read_generation_manifest(generation: &Path) -> Option<GenerationManifest> {
    let content = fs::read_to_string(generation.join(GENERATION_MANIFEST_FILE)).ok()?;
    let mut lines = content.lines();
    if lines.next()? != "checkpoint-v2" {
        return None;
    }
    let parent = lines.next()?.strip_prefix("parent: ")?;
    let parent = if parent == "-" {
        None
    } else {
        Some(parent.to_string())
    };
    // Third line is optional: manifests written before timestamps existed
    // have no `saved_at` line and fall back to mtime. Remaining lines are
    // optional caches or encoded commit metadata; missing storage falls back
    // to logical size for pre-COW snapshots.
    let saved_at = match lines.next() {
        None => None,
        Some(line) => Some(line.strip_prefix("saved_at: ")?.parse::<u64>().ok()?),
    };
    let mut size_bytes = None;
    let mut storage_bytes = None;
    let mut message_count = None;
    let mut message = None;
    for line in lines {
        if let Some(value) = line.strip_prefix("size: ")
            && size_bytes.is_none()
        {
            size_bytes = value.parse::<u64>().ok();
            continue;
        }
        if let Some(value) = line.strip_prefix("storage: ")
            && storage_bytes.is_none()
        {
            storage_bytes = value.parse::<u64>().ok();
            continue;
        }
        if let Some(value) = line.strip_prefix("messages: ")
            && message_count.is_none()
        {
            message_count = value.parse::<usize>().ok();
            continue;
        }
        if let Some(value) = line.strip_prefix("message-base64: ")
            && message.is_none()
        {
            message = base64::engine::general_purpose::STANDARD
                .decode(value)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok());
        }
    }
    Some(GenerationManifest {
        parent,
        message,
        saved_at,
        size_bytes,
        storage_bytes,
        message_count,
    })
}

/// Parse a generation manifest into its parent link. Returns `None` when the
/// checkpoint predates parent tracking (`checkpoint-v1`), has no manifest, or
/// the manifest is unparseable; otherwise the v2 parent (`None` = root).
fn read_generation_parent(generation: &Path) -> Option<Option<String>> {
    read_generation_manifest(generation).map(|manifest| manifest.parent)
}

/// Wall-clock seconds for stamping a new manifest. Degenerate clocks fall
/// back to zero rather than failing the save.
fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Display time for a manifest timestamp: unix seconds rendered in local
/// time. Returns `None` when the manifest carries no timestamp.
fn manifest_saved_time(manifest: Option<&GenerationManifest>) -> Option<DateTime<Local>> {
    let secs = manifest?.saved_at?;
    let utc = chrono::DateTime::<chrono::Utc>::from(
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs),
    );
    Some(utc.with_timezone(&Local))
}

/// Render a v2 manifest body for `parent` (`None` = root) stamped with
/// `saved_at` (unix seconds): `SAVED` and newest-first order read this, not
/// the directory mtime. Size, storage, and message count are optional caches;
/// `None` omits the line so old readers (which only require the first three
/// lines) keep parsing.
fn generation_manifest_body(
    parent: Option<&str>,
    saved_at: u64,
    size_bytes: Option<u64>,
    storage_bytes: Option<u64>,
    message_count: Option<usize>,
    message: Option<&str>,
) -> String {
    let parent_line = match parent {
        Some(name) => format!("parent: {name}\n"),
        None => "parent: -\n".to_string(),
    };
    let mut body = format!("checkpoint-v2\n{parent_line}saved_at: {saved_at}\n");
    if let Some(size) = size_bytes {
        body.push_str(&format!("size: {size}\n"));
    }
    if let Some(storage) = storage_bytes {
        body.push_str(&format!("storage: {storage}\n"));
    }
    if let Some(count) = message_count {
        body.push_str(&format!("messages: {count}\n"));
    }
    if let Some(message) = message {
        let encoded = base64::engine::general_purpose::STANDARD.encode(message);
        body.push_str(&format!("message-base64: {encoded}\n"));
    }
    body
}

fn generation_manifest_with_sizes(
    parent: Option<&str>,
    saved_at: u64,
    logical_base: u64,
    storage_base: u64,
    message_count: Option<usize>,
    message: Option<&str>,
) -> (String, u64, u64) {
    let mut logical_size = logical_base;
    let mut storage_size = storage_base;
    loop {
        let body = generation_manifest_body(
            parent,
            saved_at,
            Some(logical_size),
            Some(storage_size),
            message_count,
            message,
        );
        let body_size = body.len() as u64;
        let next_logical = logical_base.saturating_add(body_size);
        let next_storage = storage_base.saturating_add(body_size);
        if next_logical == logical_size && next_storage == storage_size {
            return (body, logical_size, storage_size);
        }
        logical_size = next_logical;
        storage_size = next_storage;
    }
}

/// Whether `name` is reserved for pointer aliases: the HEAD pointer (`HEAD`
/// case-insensitively, plus the git-style aliases `@` and `-`) and the
/// newest-checkpoint alias `latest` (case-insensitive). Reserved names can
/// never become checkpoint directories: on case-insensitive filesystems
/// `head` and `HEAD` would collide, `@` / `-` sanitize to misleading
/// directory names, and `latest` must keep resolving to the newest entry.
pub(in crate::ai) fn is_reserved_checkpoint_name(name: &str) -> bool {
    let trimmed = name.trim();
    trimmed.eq_ignore_ascii_case("head")
        || trimmed.eq_ignore_ascii_case("latest")
        || trimmed == "@"
        || trimmed == "-"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::{Message, append_history_messages, build_message_arr};
    use serde_json::Value;
    use std::sync::{Arc, Barrier};

    fn temp_history_file() -> PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "ai_ckpt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p.push("history.sqlite");
        p
    }

    #[test]
    fn sanitize_name_handles_unsafe_chars() {
        assert_eq!(sanitize_name("my checkpoint"), "my_checkpoint");
        assert_eq!(sanitize_name("../../etc/passwd"), "etc_passwd");
        assert_eq!(sanitize_name(""), "checkpoint");
        assert_eq!(sanitize_name("v1.2"), "v1_2");
    }

    #[test]
    fn normalized_reserved_names_are_rejected() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-reserved");

        let save_error = store.save("la💥test").unwrap_err();
        assert_eq!(save_error.kind(), io::ErrorKind::InvalidInput);
        let branch_error = store.create_branch("he💥ad", None).unwrap_err();
        assert_eq!(branch_error.kind(), io::ErrorKind::InvalidInput);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn checkpoint_budget_rejects_over_limit_without_replacing_existing_entry() {
        let count_error = validate_checkpoint_budget(2, false, 10, 0, 1, 2, 100).unwrap_err();
        assert_eq!(count_error.kind(), io::ErrorKind::StorageFull);

        validate_checkpoint_budget(2, true, 10, 5, 6, 2, 100).unwrap();

        let storage_error = validate_checkpoint_budget(1, false, 90, 10, 21, 2, 100).unwrap_err();
        assert_eq!(storage_error.kind(), io::ErrorKind::StorageFull);
    }

    #[test]
    fn checkpoint_recovery_publishes_validated_staged_generation() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-stage");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("staged history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let staged = store.dir.join(".stable.stage-interrupted");
        fs::create_dir_all(staged.join(GENERATION_ASSETS_DIR)).unwrap();
        backup_sqlite(&store.session_file, &staged.join(GENERATION_SQLITE_FILE)).unwrap();
        fs::write(staged.join(GENERATION_MANIFEST_FILE), b"checkpoint-v1\n").unwrap();
        store.mark_generation_publish_ready(&staged).unwrap();

        let listed = store.list().unwrap();

        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "stable");
        assert!(!staged.exists());
        assert!(store.checkpoint_path("stable").exists());
        assert!(
            !store
                .generation_dir("stable")
                .join(GENERATION_PUBLISH_READY_MARKER)
                .exists()
        );
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn checkpoint_recovery_discards_complete_unvalidated_staged_generation() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-stage-unvalidated");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("unvalidated history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let staged = store.stage_generation("unvalidated", None, None).unwrap();
        assert!(generation_is_ready(&staged.path));

        assert!(store.list().unwrap().is_empty());
        assert!(!staged.path.exists());
        assert!(!store.generation_dir("unvalidated").exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn checkpoint_list_reclaims_incomplete_staged_generation() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-stage-cleanup");
        let staged = store.dir.join(".orphan.stage-interrupted");
        fs::create_dir_all(&staged).unwrap();
        fs::write(staged.join("partial-assets.bin"), vec![0u8; 64 * 1024]).unwrap();

        assert!(store.list().unwrap().is_empty());
        assert!(!staged.exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn checkpoint_clear_removes_everything_but_live_history_and_base() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-clear");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("live history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        store.save("a").unwrap();
        store.save("b").unwrap();
        store.set_head("b").unwrap();
        assert_eq!(store.list().unwrap().len(), 2);

        assert_eq!(store.clear().unwrap(), 2);
        assert!(store.list().unwrap().is_empty());
        assert_eq!(store.head_target().unwrap(), None);
        // Live history survives; the base pointer is kept so the next save
        // shows up as an orphan, not a false root.
        assert!(store.session_file.exists());
        assert_eq!(store.base_target().unwrap(), Some("b".to_string()));
        store.save("c").unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].parent.as_deref(), Some("b"));

        // Clearing again removes the one resaved checkpoint; an empty store
        // is a no-op returning zero.
        assert_eq!(store.clear().unwrap(), 1);
        assert!(store.list().unwrap().is_empty());
        assert_eq!(store.clear().unwrap(), 0);
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn concurrent_checkpoint_saves_respect_session_limit() {
        let history_file = temp_history_file();
        let store = Arc::new(CheckpointStore::new(&history_file, "sess-concurrent"));
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("source".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(MAX_CHECKPOINTS_PER_SESSION + 1));
        let mut tasks = Vec::new();
        for index in 0..=MAX_CHECKPOINTS_PER_SESSION {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            tasks.push(std::thread::spawn(move || {
                barrier.wait();
                store.save(&format!("checkpoint-{index}")).is_ok()
            }));
        }
        let saved = tasks
            .into_iter()
            .filter_map(|task| task.join().ok())
            .filter(|saved| *saved)
            .count();

        assert_eq!(saved, MAX_CHECKPOINTS_PER_SESSION);
        assert_eq!(store.list().unwrap().len(), MAX_CHECKPOINTS_PER_SESSION);
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn save_list_rollback_delete_roundtrip() {
        let history_file = temp_history_file();
        let session_id = "sess-abc";
        let store = CheckpointStore::new(&history_file, session_id);

        // save must fail before any session file exists.
        assert!(store.save("c1").is_err());

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("v1".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        // Save checkpoint c1.
        let ckpt = store.save("c1").unwrap();
        assert!(ckpt.exists());

        // Change the live history to "v2".
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("v2".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        // list can see c1.
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "c1");

        // Roll back to c1; the live history should be restored to only "v1".
        store.rollback("c1").unwrap();
        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, Value::String("v1".to_string()));

        // Rolling back a nonexistent checkpoint must error.
        assert!(store.rollback("nope").is_err());

        // c1 is main's tip, so deleting it while a branch references it is
        // refused (Git-style). Re-attach to main and move its tip past c1 with
        // a new save; then c1 is unreferenced and becomes deletable.
        let err = store.delete("c1").unwrap_err();
        assert!(
            err.to_string().contains("referenced by branch main"),
            "unexpected error: {err}"
        );
        store.attach_branch("main").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("v3")]).unwrap();
        store.save("c2").unwrap();
        assert!(store.delete("c1").unwrap());
        assert!(!store.delete("c1").unwrap());
        let remaining = store.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].name, "c2");

        // Cleanup.
        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn commit_message_roundtrips_without_changing_legacy_saves() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-message");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();

        let message = "Fix login flow\n\nKeep the old checkpoint format readable. 🚀";
        store
            .save_with_message("with-message", Some(message))
            .unwrap();
        store.save("legacy").unwrap();

        let listed = store.list().unwrap();
        assert_eq!(
            listed
                .iter()
                .find(|checkpoint| checkpoint.name == "with-message")
                .and_then(|checkpoint| checkpoint.message.as_deref()),
            Some(message)
        );
        assert_eq!(
            listed
                .iter()
                .find(|checkpoint| checkpoint.name == "legacy")
                .and_then(|checkpoint| checkpoint.message.as_deref()),
            None
        );

        let manifest = read_generation_manifest(&store.generation_dir("with-message")).unwrap();
        assert_eq!(manifest.message.as_deref(), Some(message));
        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn head_pointer_tracks_current_checkpoint_and_clears_on_delete() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-head");
        // No HEAD before anything is preserved.
        assert_eq!(store.head_target().unwrap(), None);

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("v1".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        store.save("c1").unwrap();
        assert_eq!(store.head_target().unwrap().as_deref(), Some("c1"));
        // The pointer file must not show up as a checkpoint.
        assert_eq!(
            store
                .list()
                .unwrap()
                .iter()
                .map(|cp| cp.name.clone())
                .collect::<Vec<_>>(),
            vec!["c1".to_string()]
        );

        // Reserved names can never become checkpoint directories.
        assert!(store.save("HEAD").is_err());
        assert!(store.save("@").is_err());
        assert!(store.save("-").is_err());

        // c1 is main's tip, so deleting it while a branch references it is
        // refused. Move the branch tip away, check c1 out detached, then
        // deleting the HEAD target clears the pointer without touching history.
        let err = store.delete("c1").unwrap_err();
        assert!(
            err.to_string().contains("referenced by branch"),
            "unexpected error: {err}"
        );
        append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
        store.save("c2").unwrap();
        store.rollback("c1").unwrap();
        assert_eq!(store.head_target().unwrap().as_deref(), Some("c1"));
        assert!(store.delete("c1").unwrap());
        assert_eq!(store.head_target().unwrap(), None);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    fn lineage_msg(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn lineage_parent_of(store: &CheckpointStore, name: &str) -> Option<String> {
        store
            .list()
            .unwrap()
            .into_iter()
            .find(|cp| cp.name == name)
            .and_then(|cp| cp.parent)
    }

    #[test]
    fn save_chains_parents_linearly_through_live_base() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-lineage");
        assert_eq!(store.base_target().unwrap(), None);

        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("a").unwrap();
        assert_eq!(lineage_parent_of(&store, "a"), None);
        assert_eq!(store.base_target().unwrap().as_deref(), Some("a"));

        append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
        store.save("b").unwrap();
        // Newest-first: b records the live base (a) as its parent.
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].name, "b");
        assert_eq!(listed[0].parent.as_deref(), Some("a"));
        assert_eq!(listed[1].name, "a");
        assert_eq!(listed[1].parent, None);
        assert_eq!(store.base_target().unwrap().as_deref(), Some("b"));
        // The base pointer file must not show up as a checkpoint.
        assert_eq!(listed.len(), 2);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn rollback_branch_records_branch_parent() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-branch");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("a").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
        store.save("b").unwrap();

        // Roll back to a and continue there: the next save must be a child of
        // a, not of the newest checkpoint b.
        store.rollback("a").unwrap();
        assert_eq!(store.base_target().unwrap().as_deref(), Some("a"));
        append_history_messages(&store.session_file, &[lineage_msg("v3")]).unwrap();
        store.save("c").unwrap();

        assert_eq!(lineage_parent_of(&store, "a"), None);
        assert_eq!(lineage_parent_of(&store, "b"), Some("a".to_string()));
        assert_eq!(lineage_parent_of(&store, "c"), Some("a".to_string()));
        assert_eq!(store.base_target().unwrap().as_deref(), Some("c"));

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn checkpoint_names_are_immutable() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-resave");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("a").unwrap();
        store.save("b").unwrap();
        let error = store.save("b").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(lineage_parent_of(&store, "b"), Some("a".to_string()));
        assert_eq!(lineage_parent_of(&store, "a"), None);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn new_store_starts_on_unborn_main_and_advances_it() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-unborn-branch");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();

        assert_eq!(
            store.list_branches().unwrap(),
            vec![CheckpointBranch {
                name: "main".to_string(),
                target: None,
                is_dangling: false,
            }]
        );
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "main".to_string(),
                target: None,
            }
        );
        assert_eq!(
            store.delete_branch("main").unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );

        store.save("first").unwrap();
        assert_eq!(lineage_parent_of(&store, "first"), None);
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "main".to_string(),
                target: Some("first".to_string()),
            }
        );
        assert_eq!(
            store.list_branches().unwrap()[0].target.as_deref(),
            Some("first")
        );
        store.create_branch("topic", Some("first")).unwrap();
        store.switch_branch("topic").unwrap();
        let error = store.delete_branch("main").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(
            error
                .to_string()
                .contains("cannot delete protected default branch 'main'")
        );
        // Case-insensitive filesystems must not let a differently-cased path
        // bypass the default-branch protection.
        if store.branch_path("MAIN").exists() {
            assert_eq!(
                store.delete_branch("MAIN").unwrap_err().kind(),
                io::ErrorKind::PermissionDenied
            );
        }

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn clear_recreates_unborn_main() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-clear-main");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("first").unwrap();

        assert_eq!(store.clear().unwrap(), 1);
        assert!(store.list().unwrap().is_empty());
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "main".to_string(),
                target: None,
            }
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn previous_checkout_toggles_between_branches_and_detached() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-previous-checkout");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();

        // No navigation yet: nothing to go back to.
        assert_eq!(store.previous_head_state().unwrap(), None);

        store.save("c1").unwrap();
        // Saving moves HEAD with the branch; it is not a navigation.
        assert_eq!(store.previous_head_state().unwrap(), None);

        store.create_branch("b2", None).unwrap();
        store.switch_branch("b2").unwrap();
        assert_eq!(
            store.previous_head_state().unwrap(),
            Some(CheckpointHead::Attached {
                branch: "main".to_string(),
                target: Some("c1".to_string()),
            })
        );

        // A no-op re-attach must not clobber `-`.
        store.attach_branch("b2").unwrap();
        assert_eq!(
            store.previous_head_state().unwrap(),
            Some(CheckpointHead::Attached {
                branch: "main".to_string(),
                target: Some("c1".to_string()),
            })
        );

        // Toggling back records the branch we just left.
        store.switch_branch("main").unwrap();
        assert_eq!(
            store.previous_head_state().unwrap(),
            Some(CheckpointHead::Attached {
                branch: "b2".to_string(),
                target: Some("c1".to_string()),
            })
        );

        // Detaching records the attached branch we left ...
        store.rollback("c1").unwrap();
        assert_eq!(
            store.previous_head_state().unwrap(),
            Some(CheckpointHead::Attached {
                branch: "main".to_string(),
                target: Some("c1".to_string()),
            })
        );

        // ... and re-attaching records the detached checkout we left.
        store.attach_branch("main").unwrap();
        assert_eq!(
            store.previous_head_state().unwrap(),
            Some(CheckpointHead::Detached(Some("c1".to_string())))
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn repaired_unborn_main_can_create_an_unborn_branch() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-repair-unborn-main");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        assert_eq!(store.list_branches().unwrap().len(), 1);
        store.set_base_unlocked("next_1").unwrap();
        store
            .write_ref_unlocked(&store.branch_path(DEFAULT_BRANCH_NAME), "next_1")
            .unwrap();
        fs::remove_file(store.dir.join(HEAD_REF_FILE)).unwrap();

        assert_eq!(
            store.current_status_label().unwrap().as_deref(),
            Some("main@-")
        );
        assert_eq!(store.base_target().unwrap(), None);
        assert_eq!(
            store.branch_target_unlocked(DEFAULT_BRANCH_NAME).unwrap(),
            Some(None)
        );
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "main".to_string(),
                target: None,
            }
        );

        store.create_branch("check", None).unwrap();
        assert_eq!(
            store.current_status_label().unwrap().as_deref(),
            Some("main@-")
        );
        assert_eq!(store.branch_target_unlocked("main").unwrap(), Some(None));
        assert_eq!(store.branch_target_unlocked("check").unwrap(), Some(None));
        assert!(
            !store
                .live_matches_base(&build_message_arr(10, &store.session_file).unwrap())
                .unwrap()
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn dangling_branch_targets_are_reported_and_not_reused() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-dangling-branch");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("first").unwrap();
        fs::remove_dir_all(store.generation_dir("first")).unwrap();

        let branches = store.list_branches().unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].name, "main");
        assert_eq!(branches[0].target.as_deref(), Some("first"));
        assert!(branches[0].is_dangling);
        assert_eq!(store.current_status_label().unwrap(), None);

        let error = store.create_branch("topic", None).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(
            error
                .to_string()
                .contains("checkpoint 'first' does not exist")
        );

        let error = store.prepare_branch("topic", Some("main")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(
            error
                .to_string()
                .contains("branch 'main' points to missing checkpoint 'first'")
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn switching_to_unborn_branch_clears_other_branch_live_state() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-unborn-switch");
        store.create_branch("b2", None).unwrap();
        store.create_branch("b1", None).unwrap();
        store.switch_branch("b1").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("b1-unsaved")]).unwrap();
        assert_eq!(build_message_arr(10, &store.session_file).unwrap().len(), 1);

        store.switch_branch("b2").unwrap();

        assert!(
            build_message_arr(10, &store.session_file)
                .unwrap()
                .is_empty()
        );
        assert!(store.live_matches_base(&[]).unwrap());
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "b2".to_string(),
                target: None,
            }
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn empty_live_session_with_checkpoint_state_is_not_disposable() {
        let history_file = temp_history_file();
        let session_id = "sess-empty-live-with-checkpoint";
        let store = CheckpointStore::new(&history_file, session_id);
        store.create_branch("check", None).unwrap();
        store.switch_branch("check").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("branch-history")]).unwrap();
        store.save("verified_tag").unwrap();
        store.switch_branch(DEFAULT_BRANCH_NAME).unwrap();

        assert!(build_message_arr(10, &store.session_file).unwrap().is_empty());
        assert!(!SessionStore::new(&history_file).is_empty_session(session_id).unwrap());

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn attached_branch_advances_without_losing_alternate_history() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-branch");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("c1").unwrap();
        store.create_branch("c2", Some("c1")).unwrap();
        store.switch_branch("c2").unwrap();

        append_history_messages(&store.session_file, &[lineage_msg("branch-v2")]).unwrap();
        store.save("c2-1").unwrap();
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "c2".to_string(),
                target: Some("c2-1".to_string()),
            }
        );

        store.rollback("c1").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("alternate-v2")]).unwrap();
        store.save("alternate").unwrap();
        store.switch_branch("c2").unwrap();
        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[1].content, Value::String("branch-v2".to_string()));
        assert_eq!(lineage_parent_of(&store, "c2-1"), Some("c1".to_string()));
        assert_eq!(
            lineage_parent_of(&store, "alternate"),
            Some("c1".to_string())
        );
        assert_eq!(
            store.delete("c2-1").unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );

        store.rollback("alternate").unwrap();
        assert!(store.delete_branch("c2").unwrap());
        assert!(store.delete("c2-1").unwrap());

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn attach_branch_does_not_restore_the_session_database() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-attach-only");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("c1").unwrap();
        store.create_branch("topic", Some("c1")).unwrap();

        let connection = rusqlite::Connection::open(&store.session_file).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE attach_marker(value INTEGER); \
                 INSERT INTO attach_marker VALUES (7);",
            )
            .unwrap();
        drop(connection);

        store.attach_branch("topic").unwrap();
        assert_eq!(
            store.head_state().unwrap(),
            CheckpointHead::Attached {
                branch: "topic".to_string(),
                target: Some("c1".to_string()),
            }
        );
        let connection = rusqlite::Connection::open(&store.session_file).unwrap();
        let marker: i64 = connection
            .query_row("SELECT value FROM attach_marker", [], |row| row.get(0))
            .unwrap();
        assert_eq!(marker, 7);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn previous_head_write_failure_does_not_fail_committed_navigation() {
        for operation in ["rollback", "attach", "switch", "unborn"] {
            for block_rename in [false, true] {
                let history_file = temp_history_file();
                let store = CheckpointStore::new(&history_file, "sess-prev-failure");
                store.create_branch("empty", None).unwrap();
                append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
                store.save("c1").unwrap();
                append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
                store.save("c2").unwrap();
                let target = if operation == "attach" { "c2" } else { "c1" };
                store.create_branch("topic", Some(target)).unwrap();
                let before = build_message_arr(10, &store.session_file).unwrap();
                let previous = store.dir.join(PREV_HEAD_REF_FILE);
                let temp = previous.with_extension(format!("tmp-{}", std::process::id()));
                if block_rename {
                    fs::create_dir(&previous).unwrap();
                } else {
                    fs::write(&previous, "c1\n").unwrap();
                    fs::create_dir(&temp).unwrap();
                }

                let (result, head, base, count) = match operation {
                    "rollback" => (store.rollback("c1"), "c1", Some("c1"), 1),
                    "attach" => (
                        store.attach_branch("topic"),
                        "ref: branches/topic",
                        Some("c2"),
                        2,
                    ),
                    "switch" => (
                        store.switch_branch("topic"),
                        "ref: branches/topic",
                        Some("c1"),
                        1,
                    ),
                    _ => (store.switch_branch("empty"), "ref: branches/empty", None, 0),
                };
                assert!(
                    result.is_ok(),
                    "{operation}, block_rename={block_rename}: {result:?}"
                );
                assert_eq!(store.head_raw_unlocked().unwrap().as_deref(), Some(head));
                assert_eq!(store.base_target().unwrap().as_deref(), base);
                assert_eq!(
                    CheckpointStore::read_optional_ref_unlocked(&store.dir.join(BASE_REF_FILE))
                        .unwrap()
                        .as_deref(),
                    base
                );
                assert_eq!(
                    build_message_arr(10, &store.session_file).unwrap(),
                    before[..count]
                );
                if block_rename {
                    assert!(previous.is_dir());
                    assert!(!temp.exists());
                } else {
                    assert_eq!(fs::read_to_string(previous).unwrap(), "c1\n");
                }
                fs::remove_dir_all(history_file.parent().unwrap()).unwrap();
            }
        }
    }

    #[test]
    fn rollback_ref_write_failure_leaves_live_history_unchanged() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-ref-failure");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("c1").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
        store.save("c2").unwrap();
        append_history_messages(&store.session_file, &[lineage_msg("unsaved")]).unwrap();
        let before = build_message_arr(10, &store.session_file).unwrap();

        let blocked_temp = store
            .dir
            .join(BASE_REF_FILE)
            .with_extension(format!("tmp-{}", std::process::id()));
        fs::create_dir(&blocked_temp).unwrap();
        assert!(store.rollback("c1").is_err());

        assert_eq!(build_message_arr(10, &store.session_file).unwrap(), before);
        assert_eq!(store.head_target().unwrap().as_deref(), Some("c2"));
        assert_eq!(
            fs::read_to_string(store.dir.join(BASE_REF_FILE))
                .unwrap()
                .trim(),
            "c2"
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn deleting_parent_keeps_child_as_orphan() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-orphan");
        append_history_messages(&store.session_file, &[lineage_msg("v1")]).unwrap();
        store.save("a").unwrap();
        store.save("b").unwrap();
        assert!(store.delete("a").unwrap());
        // The child keeps its (now dangling) parent so the tree can mark it.
        assert_eq!(lineage_parent_of(&store, "b"), Some("a".to_string()));
        // b is main's tip, so deleting it while a branch references it is
        // refused. Move the branch tip past b, then b is unreferenced and
        // deletable; the live history still derives from the new tip.
        let err = store.delete("b").unwrap_err();
        assert!(
            err.to_string().contains("referenced by branch"),
            "unexpected error: {err}"
        );
        append_history_messages(&store.session_file, &[lineage_msg("v2")]).unwrap();
        store.save("c").unwrap();
        assert!(store.delete("b").unwrap());
        assert_eq!(store.base_target().unwrap().as_deref(), Some("c"));

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn legacy_v1_checkpoints_chain_in_save_order() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-legacy-chain");
        // Creation order matches name-descending tie-break, so the fallback
        // chain is deterministic whatever the filesystem timestamp granularity.
        for name in ["chain-aa", "chain-bb"] {
            let dir = store.generation_dir(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(GENERATION_MANIFEST_FILE), b"checkpoint-v1\n").unwrap();
            // `generation_is_ready` also gates on these; `list` never opens them.
            fs::write(dir.join(GENERATION_SQLITE_FILE), b"").unwrap();
            fs::create_dir_all(dir.join(GENERATION_ASSETS_DIR)).unwrap();
        }
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].name, "chain-bb");
        assert_eq!(listed[0].parent.as_deref(), Some("chain-aa"));
        assert_eq!(listed[1].name, "chain-aa");
        assert_eq!(listed[1].parent, None);

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn manifest_parent_parse_covers_v1_v2_and_garbage() {
        let history_file = temp_history_file();
        let root = history_file.parent().unwrap().join("manifests");
        let write = |name: &str, body: &[u8]| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(GENERATION_MANIFEST_FILE), body).unwrap();
            dir
        };
        assert_eq!(
            read_generation_parent(&write("v1", b"checkpoint-v1\n")),
            None
        );
        assert_eq!(
            read_generation_parent(&write("root", b"checkpoint-v2\nparent: -\n")),
            Some(None)
        );
        assert_eq!(
            read_generation_parent(&write("child", b"checkpoint-v2\nparent: a\n")),
            Some(Some("a".to_string()))
        );
        assert_eq!(read_generation_parent(&write("garbage", b"nope\n")), None);
        assert_eq!(read_generation_parent(&root.join("missing")), None);
        assert_eq!(
            generation_manifest_body(None, 0, None, None, None, None),
            "checkpoint-v2\nparent: -\nsaved_at: 0\n"
        );
        assert_eq!(
            generation_manifest_body(Some("a"), 7, None, None, None, None),
            "checkpoint-v2\nparent: a\nsaved_at: 7\n"
        );
        // Pre-timestamp manifests still parse (timestamp falls back to mtime).
        assert_eq!(
            read_generation_manifest(&write("old", b"checkpoint-v2\nparent: a\n"))
                .map(|manifest| (manifest.parent, manifest.saved_at)),
            Some((Some("a".to_string()), None))
        );
        assert_eq!(
            read_generation_manifest(&write("new", b"checkpoint-v2\nparent: a\nsaved_at: 7\n"))
                .map(|manifest| (manifest.parent, manifest.saved_at)),
            Some((Some("a".to_string()), Some(7)))
        );

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn live_matches_checkpoint_tracks_live_edits() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-match");
        let message = |content: &str| Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };

        assert!(store.live_matches_base(&[]).unwrap());
        append_history_messages(&store.session_file, &[message("v1")]).unwrap();
        let unsaved = build_message_arr(10, &store.session_file).unwrap();
        assert!(!store.live_matches_base(&unsaved).unwrap());
        store.save("c1").unwrap();
        assert_eq!(store.base_target().unwrap().as_deref(), Some("c1"));
        assert_eq!(store.head_target().unwrap().as_deref(), Some("c1"));
        let live = build_message_arr(10, &store.session_file).unwrap();
        // Just saved: live matches the snapshot.
        assert!(store.live_matches_checkpoint(&live, "c1").unwrap());
        assert!(store.live_matches_base(&live).unwrap());
        // Runtime asset churn without a new canonical turn is still clean.
        fs::create_dir_all(&store.session_assets).unwrap();
        fs::write(store.session_assets.join("runtime-cache"), b"derived").unwrap();
        assert!(store.live_matches_checkpoint(&live, "c1").unwrap());
        // Unknown names never match.
        assert!(!store.live_matches_checkpoint(&live, "nope").unwrap());

        // A new turn diverges live from the snapshot.
        append_history_messages(&store.session_file, &[message("v2")]).unwrap();
        let live = build_message_arr(10, &store.session_file).unwrap();
        assert!(!store.live_matches_checkpoint(&live, "c1").unwrap());
        assert!(!store.live_matches_base(&live).unwrap());

        // Rolling back restores the match.
        store.rollback("c1").unwrap();
        let live = build_message_arr(10, &store.session_file).unwrap();
        assert!(store.live_matches_checkpoint(&live, "c1").unwrap());
        assert!(store.live_matches_base(&live).unwrap());
        assert_eq!(store.base_target().unwrap().as_deref(), Some("c1"));
        assert_eq!(store.head_target().unwrap().as_deref(), Some("c1"));

        // Rewinding all canonical messages is still dirty against a non-empty BASE.
        clear_session_history_sqlite(&store.session_file).unwrap();
        let live = build_message_arr(10, &store.session_file).unwrap();
        assert!(live.is_empty());
        assert!(!store.live_matches_base(&live).unwrap());

        // Saved listings carry the manifest timestamp, not the dir mtime.
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].modified_local.is_some());

        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn checkpoint_restores_wal_history_and_context_assets() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-assets");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("before checkpoint".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let checkpoint_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("durable.md");
        fs::create_dir_all(checkpoint_asset.parent().unwrap()).unwrap();
        fs::write(&checkpoint_asset, "before asset").unwrap();
        let scratch_asset = store.session_assets.join("tmp").join("workspace.bin");
        fs::create_dir_all(scratch_asset.parent().unwrap()).unwrap();
        fs::write(&scratch_asset, "before scratch").unwrap();

        store.save("stable").unwrap();
        assert!(store.checkpoint_assets_path("stable").exists());
        assert!(
            store
                .checkpoint_assets_path("stable")
                .join("tmp/workspace.bin")
                .exists()
        );
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(
            listed[0].size_bytes
                >= fs::metadata(store.checkpoint_path("stable")).unwrap().len()
                    + "before asset".len() as u64
        );
        #[cfg(target_os = "macos")]
        assert!(listed[0].storage_bytes < listed[0].size_bytes);

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after checkpoint".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&checkpoint_asset, "after asset").unwrap();
        fs::write(&scratch_asset, "after_ scratch").unwrap();

        store.rollback("stable").unwrap();
        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored[0].content,
            Value::String("before checkpoint".to_string())
        );
        assert_eq!(
            fs::read_to_string(checkpoint_asset).unwrap(),
            "before asset"
        );
        assert_eq!(fs::read_to_string(scratch_asset).unwrap(), "before scratch");

        assert_eq!(store.clear().unwrap(), 1);
        assert!(!store.checkpoint_assets_path("stable").exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn cow_snapshot_charges_allocated_copy_not_logical_asset_size() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-large-cow");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("large workspace".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let sparse = store.session_assets.join("tmp/large-workspace.bin");
        fs::create_dir_all(sparse.parent().unwrap()).unwrap();
        fs::File::create(&sparse)
            .unwrap()
            .set_len(MAX_CHECKPOINT_STORAGE_BYTES + 1)
            .unwrap();

        store.save("large-cow").unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].size_bytes > MAX_CHECKPOINT_STORAGE_BYTES);
        assert!(listed[0].storage_bytes < MAX_CHECKPOINT_STORAGE_BYTES);

        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn legacy_sqlite_only_checkpoint_keeps_current_assets() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-legacy");
        let legacy_path = store.legacy_checkpoint_path("legacy");
        append_history_messages(
            &legacy_path,
            &[Message {
                role: "user".to_string(),
                content: Value::String("legacy history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let live_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("live.md");
        fs::create_dir_all(live_asset.parent().unwrap()).unwrap();
        fs::write(&live_asset, "current asset").unwrap();

        store.rollback("legacy").unwrap();

        assert_eq!(
            build_message_arr(10, &store.session_file).unwrap()[0].content,
            Value::String("legacy history".to_string())
        );
        assert_eq!(fs::read_to_string(live_asset).unwrap(), "current asset");
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn startup_recovery_completes_interrupted_live_rollback() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-rollback-recovery");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("before".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let live_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("state.md");
        fs::create_dir_all(live_asset.parent().unwrap()).unwrap();
        fs::write(&live_asset, "before asset").unwrap();
        store.save("stable").unwrap();

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&live_asset, "after asset").unwrap();

        let transaction = store
            .stage_live_rollback(&store.checkpoint_path("stable"), Some("stable"))
            .unwrap();
        assert!(transaction.exists());

        store.recover().unwrap();

        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, Value::String("before".to_string()));
        assert_eq!(fs::read_to_string(live_asset).unwrap(), "before asset");
        assert!(!transaction.exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn startup_recovery_does_not_replay_published_live_rollback() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-rollback-idempotent-recovery");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("before".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let live_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("working-checkpoint.md");
        fs::create_dir_all(live_asset.parent().unwrap()).unwrap();
        fs::write(&live_asset, "before working checkpoint").unwrap();
        store.save("stable").unwrap();

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&live_asset, "after working checkpoint").unwrap();

        let transaction = store
            .stage_live_rollback(&store.checkpoint_path("stable"), Some("stable"))
            .unwrap();
        let transaction_id = live_rollback_transaction_id(&transaction).unwrap();
        restore_checkpoint_assets(&store.checkpoint_assets_path("stable"), &store.session_assets)
            .unwrap();
        restore_sqlite_after_rollback_with_transaction(
            &transaction.join("new").join(GENERATION_SQLITE_FILE),
            &store.session_file,
            Some(&transaction_id),
        )
        .unwrap();
        assert!(
            live_rollback_transaction_is_published(&store.session_file, &transaction_id).unwrap()
        );

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after published rollback".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&live_asset, "after published rollback working checkpoint").unwrap();

        store.recover().unwrap();

        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].content, Value::String("before".to_string()));
        assert_eq!(
            restored[1].content,
            Value::String("after published rollback".to_string())
        );
        assert_eq!(
            fs::read_to_string(live_asset).unwrap(),
            "after published rollback working checkpoint"
        );
        assert!(!transaction.exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn rollback_publish_removes_sidecars_without_a_live_main_db() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-rollback-sidecar-publish");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("checkpoint history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        store.save("stable").unwrap();
        let transaction = store
            .stage_live_rollback(&store.checkpoint_path("stable"), Some("stable"))
            .unwrap();
        let transaction_id = live_rollback_transaction_id(&transaction).unwrap();

        fs::remove_file(&store.session_file).unwrap();
        let sidecars = ["-wal", "-shm", "-journal"]
            .map(|suffix| PathBuf::from(format!("{}{}", store.session_file.display(), suffix)));
        for sidecar in &sidecars {
            fs::write(sidecar, b"stale sidecar").unwrap();
        }

        restore_sqlite_after_rollback_with_transaction(
            &transaction.join("new").join(GENERATION_SQLITE_FILE),
            &store.session_file,
            Some(&transaction_id),
        )
        .unwrap();

        assert!(store.session_file.is_file());
        assert!(sidecars.iter().all(|sidecar| !sidecar.exists()));
        assert_eq!(
            build_message_arr(10, &store.session_file).unwrap()[0].content,
            Value::String("checkpoint history".to_string())
        );
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn startup_recovery_reclaims_incomplete_live_rollback_transaction() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-rollback-stage-cleanup");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("live history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let transaction = store
            .dir
            .join(format!("{LIVE_ROLLBACK_PREFIX}{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&transaction).unwrap();
        fs::write(transaction.join(LIVE_ROLLBACK_MANIFEST), b"commit\n").unwrap();

        store.recover().unwrap();

        assert!(!transaction.exists());
        assert_eq!(
            build_message_arr(10, &store.session_file).unwrap()[0].content,
            Value::String("live history".to_string())
        );
        assert!(store.save("stable").unwrap().exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn restoring_checkpoint_assets_preserves_concurrent_unique_files() {
        let root = std::env::temp_dir().join(format!(
            "ai-checkpoint-assets-restore-{}",
            uuid::Uuid::new_v4()
        ));
        let checkpoint_assets = root.join("checkpoint");
        let live_assets = root.join("live");
        fs::create_dir_all(&checkpoint_assets).unwrap();
        fs::create_dir_all(&live_assets).unwrap();
        fs::write(checkpoint_assets.join("existing.md"), "checkpoint").unwrap();
        fs::write(live_assets.join("existing.md"), "newer").unwrap();
        fs::write(live_assets.join("concurrent.md"), "concurrent output").unwrap();

        restore_checkpoint_assets(&checkpoint_assets, &live_assets).unwrap();

        assert_eq!(
            fs::read_to_string(live_assets.join("existing.md")).unwrap(),
            "checkpoint"
        );
        assert_eq!(
            fs::read_to_string(live_assets.join("concurrent.md")).unwrap(),
            "concurrent output"
        );
        let _ = fs::remove_dir_all(root);
    }
}
