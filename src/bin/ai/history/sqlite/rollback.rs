use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};

use super::{
    connection::open_history_db, lock::with_session_state_lock, migrations::init_history_schema,
    revision::touch_session_activity,
};

const CHECKPOINT_LIVE_ROLLBACK_META_KEY: &str = "checkpoint_live_rollback_transaction_v1";

/// Reads the live DB's three monotonic counters before a rollback:
/// `history_generation`, `history_revision`, `turn_seq`. A rollback uses
/// `backup_sqlite` to overwrite the live path with the whole checkpoint DB, which
/// would restore all three counters to their old checkpoint-time values and break
/// the "monotonic across rollbacks" invariant. This function reads the live values
/// before the overwrite so that
/// `rebase_metadata_after_rollback` raises them back above the live values after the overwrite.
/// When the database is missing or the meta row is absent, each counter reads 0, matching the “never modified” baseline.
pub(in crate::ai) struct LiveRollbackMetadata {
    pub(in crate::ai) generation: i64,
    pub(in crate::ai) revision: i64,
    pub(in crate::ai) turn_seq: i64,
}

impl LiveRollbackMetadata {
    fn zero() -> Self {
        Self {
            generation: 0,
            revision: 0,
            turn_seq: 0,
        }
    }
}

/// Returns whether a rollback transaction has already atomically published its
/// replacement SQLite database. The marker is written into the replacement DB
/// before it is renamed over the live path, so a later recovery can avoid
/// replaying that replacement over messages written after the rollback.
pub(in crate::ai) fn live_rollback_transaction_is_published(
    path: &Path,
    transaction_id: &str,
) -> io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|error| io::Error::other(error.to_string()))?;
    let meta_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .is_some();
    if !meta_exists {
        return Ok(false);
    }
    let marker = conn
        .query_row(
            "SELECT value FROM meta WHERE key = ?1 LIMIT 1",
            params![CHECKPOINT_LIVE_ROLLBACK_META_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(marker.as_deref() == Some(transaction_id))
}

pub(in crate::ai) fn read_live_rollback_metadata(path: &Path) -> io::Result<LiveRollbackMetadata> {
    // `Connection::open` would create the file; here we only need to read an existing live DB, falling back to the 0 baseline when it is missing.
    if !path.exists() {
        return Ok(LiveRollbackMetadata::zero());
    }
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|e| io::Error::other(e.to_string()))?;
    let meta_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .is_some();
    if !meta_exists {
        return Ok(LiveRollbackMetadata::zero());
    }
    let read = |key: &str| -> io::Result<i64> {
        let value = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1 LIMIT 1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| io::Error::other(e.to_string()))?;
        value.map_or(Ok(0), |value| {
            value.parse::<i64>().map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid SQLite meta value for {key}: {error}"),
                )
            })
        })
    };
    Ok(LiveRollbackMetadata {
        generation: read("history_generation")?,
        revision: read("history_revision")?,
        turn_seq: read("turn_seq")?,
    })
}

/// Called after `backup_sqlite` overwrites the live DB with a checkpoint: clear the derived snapshots
/// (canonical has rolled back to the checkpoint, so the old snapshots no longer match), and raise the three monotonic counters
/// above their pre-rollback live values, guaranteeing:
/// - `history_generation` is strictly greater than the live value -> any
///     stale snapshot write still holding an old generation is rejected by fencing (the
///     generation comparison in `write_context_snapshot_sqlite` returns false).
/// - `turn_seq` is not lower than the live value -> turn numbers allocated after the rollback never reuse already-used numbers.
/// - `history_revision` is strictly greater than the live value -> cross-connection file caches observe the change and reload.
/// Everything commits in a single Immediate transaction, avoiding an inconsistent window of
/// “rolled back but counters not raised” after the overwrite.
pub(super) fn rebase_metadata_after_rollback(
    path: &Path,
    live_generation: i64,
    live_revision: i64,
    live_turn_seq: i64,
    transaction_id: Option<&str>,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    let result = (|| {
        init_history_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| io::Error::other(e.to_string()))?;
        // canonical has rolled back, so the old derived snapshots (possibly tagged with the generation at checkpoint time)
        // no longer match; clear them to force the next read to recompute from canonical.
        tx.execute("DELETE FROM context_messages", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute("DELETE FROM context_snapshot", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        // The rewind ring is derived state too; a checkpoint restore must not
        // leave a projection that a later truncation could promote.
        tx.execute("DELETE FROM context_snapshot_history", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        // Each counter takes MAX(checkpoint value, live value) then +1 (generation/revision must strictly
        // increase; turn_seq can simply take the live value, since it is the “next number to allocate”).
        let bump_gen = live_generation.saturating_add(1).max(0);
        let bump_rev = live_revision.saturating_add(1).max(0);
        let bump_turn = live_turn_seq.max(0);
        tx.execute(
            "INSERT INTO meta (key, value) VALUES ('history_generation', ?1)
             ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
            params![bump_gen.to_string()],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT INTO meta (key, value) VALUES ('turn_seq', ?1)
             ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
            params![bump_turn.to_string()],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT INTO meta (key, value) VALUES ('history_revision', ?1)
             ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
            params![bump_rev.to_string()],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        if let Some(transaction_id) = transaction_id {
            tx.execute(
                "INSERT INTO meta (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![CHECKPOINT_LIVE_ROLLBACK_META_KEY, transaction_id],
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        }
        touch_session_activity(&tx)?;
        tx.commit().map_err(|e| io::Error::other(e.to_string()))
    })();
    // The caller renames this working DB over the live path immediately after
    // this returns. Returning the connection to the process-wide write pool
    // would keep the renamed inode's WAL index (`-shm`) alive in this process
    // after publish deletes and recreates the sidecars: later connections to
    // the live path would then attach to that stale, unlinked WAL index, so
    // commits from other processes stay invisible here and our next write can
    // overwrite them. Close it on every path so publication starts with no
    // open users of the source file.
    conn.close();
    result
}

/// Prepare a rebased temporary DB inside the session state lock, then publish it to the live path
/// via atomic rename. A crash before the final publish leaves the live DB unchanged; after the publish, metadata
/// is fully raised, so there is no intermediate “DB rolled back but counters still stale” state.
pub(in crate::ai) fn restore_sqlite_after_rollback(
    checkpoint: &Path,
    live_path: &Path,
) -> io::Result<()> {
    restore_sqlite_after_rollback_with_transaction(checkpoint, live_path, None)
}

/// Like [`restore_sqlite_after_rollback`], but atomically stamps the published
/// replacement DB with a rollback transaction identifier. The stamp is part of
/// the same replacement DB that is renamed into place, making crash recovery
/// idempotent even when later asset restoration or cleanup fails.
pub(in crate::ai) fn restore_sqlite_after_rollback_with_transaction(
    checkpoint: &Path,
    live_path: &Path,
    transaction_id: Option<&str>,
) -> io::Result<()> {
    restore_rollback_database(checkpoint, live_path, transaction_id, false)
}

/// Restore a private, closed transaction snapshot. The caller holds the
/// checkpoint lock so the source cannot change while cloning. Preserve it for
/// crash replay; only the disposable working copy is rebased and published.
pub(in crate::ai) fn restore_staged_sqlite_after_rollback(
    staged: &Path,
    live_path: &Path,
    transaction_id: &str,
) -> io::Result<()> {
    restore_rollback_database(staged, live_path, Some(transaction_id), true)
}

fn restore_rollback_database(
    checkpoint: &Path,
    live_path: &Path,
    transaction_id: Option<&str>,
    private_staged_source: bool,
) -> io::Result<()> {
    with_session_state_lock(live_path, || {
        let live = read_live_rollback_metadata(live_path)?;
        let parent = live_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let working = parent.join(format!(".rollback-rebased-{}.sqlite", uuid::Uuid::new_v4()));

        let result = (|| {
            if !private_staged_source || !try_clone_staged_sqlite(checkpoint, &working) {
                backup_sqlite(checkpoint, &working)?;
            }
            rebase_metadata_after_rollback(
                &working,
                live.generation,
                live.revision,
                live.turn_seq,
                transaction_id,
            )?;
            // Materialize the working WAL before renaming, without copying the
            // entire DB again. The marker and rebased counters publish together.
            publish_rebased_sqlite(&working, live_path)
        })();
        let _ = fs::remove_file(&working);
        let _ = remove_sqlite_sidecars(&working);
        result
    })
}

/// Clone only private staged DBs with no sidecars. Live/WAL sources always use
/// Online Backup, including old transaction snapshots left with a WAL. A clone
/// has an independent inode and copy-on-write blocks, unlike an unsafe hardlink.
#[cfg(target_os = "macos")]
fn try_clone_staged_sqlite(source: &Path, target: &Path) -> bool {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};

    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar = source.as_os_str().to_os_string();
        sidecar.push(suffix);
        if !matches!(Path::new(&sidecar).try_exists(), Ok(false)) {
            return false;
        }
    }
    let (Ok(source), Ok(target)) = (
        CString::new(source.as_os_str().as_bytes()),
        CString::new(target.as_os_str().as_bytes()),
    ) else {
        return false;
    };
    // SAFETY: Both paths are NUL-terminated and remain alive for this call.
    // Unsupported filesystems and cross-volume clones fall back to Online Backup.
    unsafe { libc::clonefile(source.as_ptr(), target.as_ptr(), 0) == 0 }
}

#[cfg(not(target_os = "macos"))]
fn try_clone_staged_sqlite(_source: &Path, _target: &Path) -> bool {
    false
}

/// Create a consistent snapshot with the SQLite Online Backup API and write it to the target via atomic replacement.
/// Copying the WAL main file directly would miss pages not yet checkpointed; the backup API reads both the main DB and the WAL
/// from the same SQLite snapshot. The source opens read-only so a rollback never drops transient sidecars into the checkpoint
/// directory (which used to bump its mtime and steal the `[latest]` tag). After the main DB replacement succeeds, the old
/// sidecar files are removed so a stale WAL/SHM cannot mix with the new main DB.
pub(in crate::ai) fn backup_sqlite(source: &Path, target: &Path) -> io::Result<()> {
    if !source.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("SQLite source does not exist: {}", source.display()),
        ));
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    let temporary = parent.join(format!(".{file_name}.backup-{}.tmp", uuid::Uuid::new_v4()));

    let result = (|| {
        let source_conn =
            Connection::open_with_flags(source, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| io::Error::other(e.to_string()))?;
        source_conn
            .backup(rusqlite::MAIN_DB, &temporary, None)
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(source_conn);

        fs::rename(&temporary, target)?;
        remove_sqlite_sidecars(target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = remove_sqlite_sidecars(&temporary);
    }
    result
}

/// Consume the private rebased working DB without another Online Backup.
/// Both DBs are closed and their WALs checkpointed before the atomic rename;
/// failures before publication leave the old live DB intact and retryable.
fn publish_rebased_sqlite(source: &Path, target: &Path) -> io::Result<()> {
    if !source.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("SQLite source does not exist: {}", source.display()),
        ));
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    quiesce_and_remove_sqlite_sidecars(source)?;
    quiesce_and_remove_sqlite_sidecars(target)?;
    fs::rename(source, target)
}

#[cfg(test)]
mod copy_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn staged_publish_renames_working_inode_without_copying() {
        use std::os::unix::fs::MetadataExt;
        let dir = std::env::temp_dir().join(format!("rollback_publish_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("working.sqlite");
        let target = dir.join("live.sqlite");
        let conn = Connection::open(&source).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; CREATE TABLE proof(value); INSERT INTO proof VALUES(42);",
        )
        .unwrap();
        drop(conn);
        let inode = fs::metadata(&source).unwrap().ino();
        publish_rebased_sqlite(&source, &target).unwrap();
        assert!(!source.exists());
        assert_eq!(fs::metadata(&target).unwrap().ino(), inode);
        let conn = Connection::open(&target).unwrap();
        assert_eq!(
            conn.query_row("SELECT value FROM proof", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            42
        );
        drop(conn);
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn staged_clone_is_copy_on_write_not_a_hardlink() {
        use std::os::unix::fs::MetadataExt;
        let dir = std::env::temp_dir().join(format!("rollback_clone_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.sqlite");
        let target = dir.join("clone.sqlite");
        fs::write(&source, b"snapshot bytes").unwrap();
        if try_clone_staged_sqlite(&source, &target) {
            assert_ne!(
                fs::metadata(&source).unwrap().ino(),
                fs::metadata(&target).unwrap().ino()
            );
            assert_eq!(fs::read(&target).unwrap(), b"snapshot bytes");
            fs::write(&target, b"modified bytes").unwrap();
            assert_eq!(fs::read(&source).unwrap(), b"snapshot bytes");
            eprintln!(
                "clonefile succeeded; independent inode and in-place write isolation verified"
            );
        } else {
            eprintln!("clonefile unsupported here; restore uses Online Backup fallback");
        }
        let _ = fs::remove_dir_all(dir);
    }
}

fn quiesce_and_remove_sqlite_sidecars(path: &Path) -> io::Result<()> {
    if path.exists() {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .map_err(|e| io::Error::other(e.to_string()))?;
        conn.busy_timeout(Duration::from_secs(5))
            .map_err(|e| io::Error::other(e.to_string()))?;
        let busy = conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(conn);
        if busy != 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("SQLite WAL checkpoint is busy: {}", path.display()),
            ));
        }
    }
    remove_sqlite_sidecars(path)
}

/// Copy the parent session's history DB to a standalone per-process file for a sub-agent.
///
/// The `inherit.history` semantics are “the sub-agent may read the parent session's context, but writes must not flow back into the parent DB”.
/// Reusing the parent DB's canonical file directly would let the sub-agent's internal prompt/tool traces pollute the parent session
/// and interleave writes to the same session across concurrent sub-agents. Here we use SQLite Online Backup to make one
/// consistent snapshot copy; the sub-agent only reads and writes its own fork file afterward, keeping the parent DB isolated.
///
/// When the parent session has no history file yet (first session), publish a brand-new empty DB instead of reusing a leftover child DB.
pub(in crate::ai) fn fork_history_for_subagent(parent: &Path, child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || match fs::metadata(parent) {
        Ok(_metadata) => {
            // Ensure the child file's parent directory exists, or the temporary file for `backup_sqlite` would fail to be created.
            if let Some(dir) = child.parent() {
                fs::create_dir_all(dir)?;
            }
            backup_sqlite(parent, child)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // The parent session has no history file, so the sub-agent starts from a blank history.
            reset_history_for_subagent_unlocked(child)
        }
        Err(error) => Err(error),
    })
}

/// When dispatching a sub-agent for the first time without history inheritance, publish a brand-new empty DB instead of reusing a leftover DB for the same pid.
pub(in crate::ai) fn reset_history_for_subagent(child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || reset_history_for_subagent_unlocked(child))
}

fn reset_history_for_subagent_unlocked(child: &Path) -> io::Result<()> {
    let temporary = child.with_extension(format!(
        "sqlite.empty-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    if let Some(parent) = temporary.parent() {
        fs::create_dir_all(parent)?;
    }
    let result = (|| {
        let conn = open_history_db(&temporary)?;
        let initialized = init_history_schema(&conn);
        // The staged file is renamed into place below, so it must be fully
        // closed rather than returned to the connection cache — including when
        // schema initialization fails, or the cached handle would keep the
        // temp file's inode open after the cleanup below unlinks it.
        conn.close();
        initialized?;
        backup_sqlite(&temporary, child)
    })();
    let _ = fs::remove_file(&temporary);
    let _ = remove_sqlite_sidecars(&temporary);
    result
}

fn remove_sqlite_sidecars(path: &Path) -> io::Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::remove_file(sidecar) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
