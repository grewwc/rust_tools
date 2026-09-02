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
fn rebase_metadata_after_rollback(
    path: &Path,
    live_generation: i64,
    live_revision: i64,
    live_turn_seq: i64,
    transaction_id: Option<&str>,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
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
}

/// Prepare a rebased temporary DB inside the session state lock, then publish it to the live path
/// via Online Backup's atomic rename. A crash before the final publish leaves the live DB unchanged; after the publish, metadata
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
    with_session_state_lock(live_path, || {
        let live = read_live_rollback_metadata(live_path)?;
        let parent = live_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let working = parent.join(format!(".rollback-rebased-{}.sqlite", uuid::Uuid::new_v4()));

        let result = (|| {
            backup_sqlite(checkpoint, &working)?;
            rebase_metadata_after_rollback(
                &working,
                live.generation,
                live.revision,
                live.turn_seq,
                transaction_id,
            )?;
            // The second backup materializes the working WAL into the final
            // temporary main DB. Publishing it has no fallible cleanup after
            // the rename, so the embedded transaction marker is sufficient to
            // distinguish an already-published rollback during recovery.
            publish_rebased_sqlite(&working, live_path)
        })();
        let _ = fs::remove_file(&working);
        let _ = remove_sqlite_sidecars(&working);
        result
    })
}

/// Create a consistent snapshot with the SQLite Online Backup API and write it to the target via atomic replacement.
/// Copying the WAL main file directly would miss pages not yet checkpointed; the backup API reads both the main DB and the WAL
/// from the same SQLite snapshot. After the main DB replacement succeeds, the old sidecar files are removed so a stale WAL/SHM
/// cannot mix with the new main DB.
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
        let source_conn = Connection::open(source).map_err(|e| io::Error::other(e.to_string()))?;
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

/// Publish a rebased rollback database without any fallible work after the
/// target rename. Existing target sidecars are checkpointed and removed before
/// publication, so a later failure leaves the old main DB intact and retryable.
fn publish_rebased_sqlite(source: &Path, target: &Path) -> io::Result<()> {
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
    let temporary = parent.join(format!(
        ".{file_name}.rollback-{}.tmp",
        uuid::Uuid::new_v4()
    ));

    let result = (|| {
        let source_conn = Connection::open(source).map_err(|e| io::Error::other(e.to_string()))?;
        source_conn
            .backup(rusqlite::MAIN_DB, &temporary, None)
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(source_conn);

        quiesce_and_remove_sqlite_sidecars(target)?;
        fs::rename(&temporary, target)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = remove_sqlite_sidecars(&temporary);
    }
    result
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
        init_history_schema(&conn)?;
        drop(conn);
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
