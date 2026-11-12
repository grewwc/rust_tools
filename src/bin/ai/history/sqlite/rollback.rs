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

/// Rebase a whole checkpoint database before publishing it over the live path.
/// Its canonical messages and singleton context projection come from the same
/// SQLite snapshot, so retain the projection and stamp it with the rebased
/// generation. Clear orphan context rows and the archived rewind ring.
///
/// Raise the three monotonic counters above their pre-rollback live values,
/// guaranteeing:
/// - `history_generation` is strictly greater than the live value -> any
///     stale snapshot write still holding an old generation is rejected by fencing (the
///     generation comparison in `write_context_snapshot_sqlite` returns false).
/// - `turn_seq` is not lower than the live value -> turn numbers allocated after the rollback never reuse already-used numbers.
/// - `history_revision` is strictly greater than the live value -> cross-connection file caches observe the change and reload.
/// Re-stamping the retained projection does not change the writer fence: a
/// writer holding a pre-rollback generation still fails the generation check.
/// Everything commits in one Immediate transaction before atomic publication.
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
        // Rewind fallbacks are not carried across a checkpoint restore.
        tx.execute("DELETE FROM context_snapshot_history", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        // Each counter takes MAX(checkpoint value, live value) then +1 (generation/revision must strictly
        // increase; turn_seq can simply take the live value, since it is the “next number to allocate”).
        let bump_gen = live_generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("history generation overflow"))?
            .max(0);
        let bump_rev = live_revision.saturating_add(1).max(0);
        let bump_turn = live_turn_seq.max(0);
        tx.execute(
            "INSERT INTO meta (key, value) VALUES ('history_generation', ?1)
             ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
            params![bump_gen.to_string()],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        // Use the stored generation because the checkpoint's counter can be
        // greater than the pre-rollback live counter. Keep its watermark and
        // fingerprint so readers still merge the canonical tail and reject a
        // projection produced under a different compression policy.
        let retained_snapshot = tx
            .execute(
                "UPDATE context_snapshot
                 SET source_generation = (
                     SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'history_generation'
                 ),
                     compression_state = NULL
                 WHERE singleton = 1",
                [],
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        if retained_snapshot == 0 {
            tx.execute("DELETE FROM context_messages", [])
                .map_err(|e| io::Error::other(e.to_string()))?;
        }
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
        let live_roots = if live_path.exists() {
            let conn = super::connection::open_history_db_read_only(live_path)?;
            super::metadata::read_write_roots(&conn)?
        } else {
            Vec::new()
        };
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
            // Grants are live session authorization, never restored from imported checkpoint data.
            // The lock keeps grants appended after the checkpoint in this published replacement.
            store_live_write_roots_before_publish(&working, &live_roots)?;
            // Materialize the working WAL before renaming, without copying the
            // entire DB again. The marker and rebased counters publish together.
            publish_rebased_sqlite(&working, live_path)
        })();
        let _ = fs::remove_file(&working);
        let _ = remove_sqlite_sidecars(&working);
        result
    })
}

fn store_live_write_roots_before_publish(path: &Path, roots: &[PathBuf]) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    let result = (|| {
        let tx = conn.transaction().map_err(io::Error::other)?;
        super::metadata::store_write_roots(&tx, roots)?;
        tx.commit().map_err(io::Error::other)
    })();
    // Publication and error cleanup both require the working DB to be closed.
    conn.close();
    result
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

#[cfg(target_os = "linux")]
fn try_clone_staged_sqlite(source: &Path, target: &Path) -> bool {
    use std::{
        fs::{File, OpenOptions},
        os::unix::io::AsRawFd,
    };

    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar = source.as_os_str().to_os_string();
        sidecar.push(suffix);
        if !matches!(Path::new(&sidecar).try_exists(), Ok(false)) {
            return false;
        }
    }
    let result = (|| -> io::Result<()> {
        let source = File::open(source)?;
        let target = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(target)?;
        // SAFETY: Both descriptors remain open for the call. FICLONE
        // (_IOW(0x94, 9, int)) creates independent copy-on-write file contents.
        if unsafe { libc::ioctl(target.as_raw_fd(), 0x40049409, source.as_raw_fd()) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(target);
    }
    result.is_ok()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
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
        let source_conn = Connection::open_with_flags(source, OpenFlags::SQLITE_OPEN_READ_ONLY)
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
mod projection_tests {
    use super::super::{
        append_history_sqlite, read_all_messages_sqlite, read_context_history_sqlite,
        write_context_snapshot_sqlite,
    };
    use super::*;
    use crate::ai::history::Message;

    const PROJECTION: &str = "checkpoint-restore-projection";

    struct Fixture {
        dir: PathBuf,
        checkpoint: PathBuf,
        live: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir =
                std::env::temp_dir().join(format!("rollback_projection_{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            Self {
                checkpoint: dir.join("checkpoint.sqlite"),
                live: dir.join("live.sqlite"),
                dir,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[cfg(unix)]
    #[test]
    fn rollback_write_roots_close_connections_on_success_and_error() {
        for reject_write in [false, true] {
            let fixture = Fixture::new();
            append_history_sqlite(&fixture.live, vec![msg("user", "live")]).unwrap();
            let conn = open_history_db(&fixture.live).unwrap();
            // This connection-local setting survives a cache return, but not a close.
            conn.pragma_update(None, "cache_size", 12345).unwrap();
            if reject_write {
                conn.execute_batch(
                    "CREATE TRIGGER reject_roots BEFORE INSERT ON meta
                     WHEN NEW.key='session_write_roots'
                     BEGIN SELECT RAISE(ABORT, 'reject roots'); END;",
                )
                .unwrap();
            }
            drop(conn);

            let result = store_live_write_roots_before_publish(
                &fixture.live,
                std::slice::from_ref(&fixture.dir),
            );
            assert_eq!(result.is_err(), reject_write);
            // Check before any rename or deletion can hide a cached source handle.
            for suffix in ["-wal", "-shm"] {
                assert!(!PathBuf::from(format!("{}{suffix}", fixture.live.display())).exists());
            }
            let conn = open_history_db(&fixture.live).unwrap();
            let cache_size: i64 = conn
                .pragma_query_value(None, "cache_size", |row| row.get(0))
                .unwrap();
            assert_ne!(cache_size, 12345, "grant writer must not remain cached");
            let roots = super::super::metadata::read_write_roots(&conn).unwrap();
            assert_eq!(
                roots,
                if reject_write {
                    vec![]
                } else {
                    vec![fixture.dir.clone()]
                }
            );
            conn.close();
        }
    }

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: serde_json::json!(content),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn initialize_history(path: &Path, generation: i64) -> Vec<Message> {
        let messages = vec![
            msg("user", "canonical question"),
            msg("assistant", "canonical answer"),
        ];
        append_history_sqlite(path, messages.clone()).unwrap();
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('history_generation', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![generation.to_string()],
        )
        .unwrap();
        messages
    }

    fn save_projection(path: &Path, content: &str) -> i64 {
        let source = read_context_history_sqlite(path, PROJECTION).unwrap();
        assert!(
            write_context_snapshot_sqlite(
                path,
                &[msg("assistant", content)],
                source.source_message_id,
                source.canonical_generation,
                PROJECTION,
            )
            .unwrap()
        );
        source.source_message_id
    }

    #[test]
    fn checkpoint_restore_preserves_context_projection_and_writer_fence() {
        for (checkpoint_generation, live_generation) in [(2, 7), (11, 2)] {
            let fixture = Fixture::new();
            let canonical = initialize_history(&fixture.checkpoint, checkpoint_generation);
            initialize_history(&fixture.live, live_generation);
            save_projection(&fixture.checkpoint, "older projection");
            let watermark = save_projection(&fixture.checkpoint, "saved projection");
            let stale = read_context_history_sqlite(&fixture.live, PROJECTION).unwrap();

            restore_sqlite_after_rollback(&fixture.checkpoint, &fixture.live).unwrap();

            let conn = Connection::open(&fixture.live).unwrap();
            let generation = super::super::revision::history_generation(&conn).unwrap();
            assert_eq!(generation, checkpoint_generation.max(live_generation + 1));
            assert!(generation > stale.canonical_generation);
            let snapshot: (i64, i64, String) = conn
                .query_row(
                    "SELECT source_message_id, source_generation, projection_fingerprint
                     FROM context_snapshot WHERE singleton = 1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            assert_eq!(snapshot, (watermark, generation, PROJECTION.to_string()));
            let row_counts: (i64, i64) = conn
                .query_row(
                    "SELECT (SELECT COUNT(*) FROM context_messages),
                            (SELECT COUNT(*) FROM context_snapshot_history)",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(row_counts, (1, 0));
            drop(conn);

            let restored = read_context_history_sqlite(&fixture.live, PROJECTION).unwrap();
            assert!(restored.snapshot_is_current);
            assert_eq!(restored.source_message_id, watermark);
            assert_eq!(restored.canonical_generation, generation);
            assert_eq!(
                restored.messages,
                vec![msg("assistant", "saved projection")]
            );
            assert_eq!(read_all_messages_sqlite(&fixture.live).unwrap(), canonical);
            assert!(
                !write_context_snapshot_sqlite(
                    &fixture.live,
                    &[msg("assistant", "stale live projection")],
                    stale.source_message_id,
                    stale.canonical_generation,
                    PROJECTION,
                )
                .unwrap()
            );
            assert_eq!(
                read_context_history_sqlite(&fixture.live, PROJECTION)
                    .unwrap()
                    .messages,
                restored.messages
            );
        }
    }

    #[test]
    fn checkpoint_restore_without_snapshot_clears_orphan_context_messages() {
        let fixture = Fixture::new();
        let canonical = initialize_history(&fixture.checkpoint, 3);
        initialize_history(&fixture.live, 1);
        save_projection(&fixture.checkpoint, "older projection");
        save_projection(&fixture.checkpoint, "orphan projection");
        let conn = Connection::open(&fixture.checkpoint).unwrap();
        conn.execute("DELETE FROM context_snapshot", []).unwrap();
        drop(conn);

        restore_sqlite_after_rollback(&fixture.checkpoint, &fixture.live).unwrap();

        let conn = Connection::open(&fixture.live).unwrap();
        let row_counts: (i64, i64, i64) = conn
            .query_row(
                "SELECT (SELECT COUNT(*) FROM context_messages),
                        (SELECT COUNT(*) FROM context_snapshot),
                        (SELECT COUNT(*) FROM context_snapshot_history)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row_counts, (0, 0, 0));
        drop(conn);
        let restored = read_context_history_sqlite(&fixture.live, PROJECTION).unwrap();
        assert!(!restored.snapshot_is_current);
        assert_eq!(restored.messages, canonical);
    }

    #[test]
    fn checkpoint_restore_reader_reuses_projection_and_canonical_tail() {
        let fixture = Fixture::new();
        let mut canonical = initialize_history(&fixture.checkpoint, 2);
        initialize_history(&fixture.live, 4);
        let watermark = save_projection(&fixture.checkpoint, "saved projection");
        let tail = msg("user", "uncompacted tail");
        append_history_sqlite(&fixture.checkpoint, vec![tail.clone()]).unwrap();
        canonical.push(tail.clone());

        restore_sqlite_after_rollback(&fixture.checkpoint, &fixture.live).unwrap();

        let conn = Connection::open(&fixture.live).unwrap();
        let stored_watermark: i64 = conn
            .query_row(
                "SELECT source_message_id FROM context_snapshot WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_watermark, watermark);
        drop(conn);
        let restored = read_context_history_sqlite(&fixture.live, PROJECTION).unwrap();
        assert!(!restored.snapshot_is_current);
        assert_eq!(restored.source_message_id, watermark + 1);
        assert_eq!(
            restored.messages,
            vec![msg("assistant", "saved projection"), tail]
        );

        let changed_policy = read_context_history_sqlite(&fixture.live, "changed-policy").unwrap();
        assert!(!changed_policy.snapshot_is_current);
        assert_eq!(changed_policy.messages, canonical);
    }
}

#[cfg(test)]
mod copy_tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn rollback_staged_clone_keeps_source_isolated() {
        use std::os::unix::fs::MetadataExt;

        let dir = std::env::temp_dir().join(format!("rollback_clone_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.sqlite");
        let target = dir.join("clone.sqlite");
        let conn = Connection::open(&source).unwrap();
        conn.execute_batch("CREATE TABLE proof(value); INSERT INTO proof VALUES(42);")
            .unwrap();
        drop(conn);
        let original = fs::read(&source).unwrap();

        if try_clone_staged_sqlite(&source, &target) {
            assert_eq!(fs::read(&target).unwrap(), original);
        } else {
            assert!(!target.exists(), "failed clones must leave no working file");
            backup_sqlite(&source, &target).unwrap();
        }
        assert_ne!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(&target).unwrap().ino()
        );
        let conn = Connection::open(&target).unwrap();
        assert_eq!(
            conn.query_row("SELECT value FROM proof", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            42
        );
        conn.execute("UPDATE proof SET value = 99", []).unwrap();
        drop(conn);
        assert_eq!(fs::read(&source).unwrap(), original);
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rollback_staged_clone_rejects_each_sidecar() {
        let dir = std::env::temp_dir().join(format!("rollback_sidecars_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.sqlite");
        let target = dir.join("clone.sqlite");
        fs::write(&source, b"snapshot bytes").unwrap();
        for suffix in ["-wal", "-shm", "-journal"] {
            let mut sidecar = source.as_os_str().to_os_string();
            sidecar.push(suffix);
            let sidecar = PathBuf::from(sidecar);
            fs::write(&sidecar, b"sidecar bytes").unwrap();
            assert!(!try_clone_staged_sqlite(&source, &target), "{suffix}");
            assert!(!target.exists());
            fs::remove_file(sidecar).unwrap();
        }
        assert_eq!(fs::read(&source).unwrap(), b"snapshot bytes");
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rollback_staged_clone_cleans_up_open_and_ioctl_errors() {
        let dir =
            std::env::temp_dir().join(format!("rollback_clone_errors_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.sqlite");
        let target = dir.join("clone.sqlite");

        fs::write(&target, b"stale working file").unwrap();
        assert!(!try_clone_staged_sqlite(&source, &target));
        assert!(
            !target.exists(),
            "source open errors must remove the target"
        );

        fs::write(&source, b"snapshot bytes").unwrap();
        fs::write(&target, b"stale working file").unwrap();
        assert!(!try_clone_staged_sqlite(&source, &target));
        assert!(
            !target.exists(),
            "target open errors must remove the target"
        );

        // A directory opens read-only on Linux, but cannot be reflinked.
        assert!(!try_clone_staged_sqlite(&dir, &target));
        assert!(!target.exists(), "ioctl errors must remove the target");
        assert_eq!(fs::read(&source).unwrap(), b"snapshot bytes");
        let _ = fs::remove_dir_all(dir);
    }

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
