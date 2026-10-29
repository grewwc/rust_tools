use super::*;
use rusqlite::Connection;
use serde_json::Value;
use std::io;

#[test]
fn sqlite_error_classifies_transient_failures_as_retryable() {
    // BUSY/LOCKED and SQLITE_IOERR system I/O failures (e.g. FSTAT/SHMOPEN when
    // concurrently opening the WAL for the first time) are transient and must map to WouldBlock so the caller can retry with a short backoff.
    for result_code in [
        rusqlite::ffi::SQLITE_BUSY,
        rusqlite::ffi::SQLITE_LOCKED,
        rusqlite::ffi::SQLITE_IOERR,
        rusqlite::ffi::SQLITE_IOERR_FSTAT,
    ] {
        let error = rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(result_code), None);
        assert_eq!(
            sqlite_error_kind(&error),
            io::ErrorKind::WouldBlock,
            "schema and transaction paths must share retry classification"
        );
        let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
        assert_eq!(io_error.kind(), io::ErrorKind::WouldBlock);
    }

    // Non-transient failures (e.g. a read-only filesystem) must not be retried.
    let error = rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_READONLY),
        None,
    );
    let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
    assert_eq!(io_error.kind(), io::ErrorKind::Other);
}

#[test]
fn context_snapshot_busy_timeout_includes_session_state_lock_wait() {
    let dir = std::env::temp_dir().join(format!(
        "snapshot_state_lock_timeout_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let holder_path = path.clone();
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        with_session_state_lock(&holder_path, || {
            locked_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(250));
            Ok(())
        })
        .unwrap();
    });
    locked_rx.recv().unwrap();

    let started = Instant::now();
    let error = write_context_snapshot_sqlite_with_busy_timeout(
        &path,
        &[],
        0,
        0,
        "fingerprint",
        Duration::from_millis(30),
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert!(started.elapsed() < Duration::from_millis(200));

    holder.join().unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

fn msg(role: &str, text: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn tool_msg(id: &str, text: &str) -> Message {
    let mut message = msg("tool", text);
    message.tool_call_id = Some(id.to_string());
    message
}

fn outcome(id: &str) -> ToolExecutionOutcome {
    ToolExecutionOutcome {
        tool_call_id: id.to_string(),
        execution_signature: format!("signature-{id}"),
        succeeded: true,
    }
}

#[test]
fn cached_outcomes_track_appends_and_prunes_without_stale_hits() {
    let dir = std::env::temp_dir().join(format!(
        "tool_outcome_cache_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![tool_msg("call-1", "first")]).unwrap();

    // Miss then hit: the second read serves the same rows from the cache.
    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1")]).unwrap();
    let first = read_tool_execution_outcomes_cached(&path).unwrap();
    assert_eq!(*first, vec![outcome("call-1")]);
    let second = read_tool_execution_outcomes_cached(&path).unwrap();
    assert_eq!(*second, vec![outcome("call-1")]);
    // A cache hit must share the stored snapshot instead of copying the table.
    assert!(std::sync::Arc::ptr_eq(&first, &second));

    // A new append bumps the revision, so the next read re-scans instead of
    // returning the cached rows.
    append_history_sqlite(&path, vec![tool_msg("call-2", "second")]).unwrap();
    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-2")]).unwrap();
    assert_eq!(
        *read_tool_execution_outcomes_cached(&path).unwrap(),
        vec![outcome("call-1"), outcome("call-2")]
    );

    // History rewrites prune orphaned rows and bump the revision as well.
    replace_all_messages_sqlite(&path, &[tool_msg("call-2", "second")]).unwrap();
    assert_eq!(
        *read_tool_execution_outcomes_cached(&path).unwrap(),
        vec![outcome("call-2")]
    );

    // Eviction (delete/rename path) drops cached rows even if a later file
    // reuses the same path with a restarted revision counter.
    remove_history_revision_cache_entry(&path);
    let _ = std::fs::remove_file(&path);
    append_history_sqlite(&path, vec![tool_msg("call-9", "ninth")]).unwrap();
    assert!(
        read_tool_execution_outcomes_cached(&path)
            .unwrap()
            .is_empty()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn turn_sequence_is_atomic_and_survives_history_clear() {
    let dir = std::env::temp_dir().join(format!(
        "turn_sequence_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "first"), msg("user", "second")]).unwrap();

    // When an older session is upgraded for the first time, numbering continues from the persisted user-turn count.
    assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 2);
    assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 3);

    clear_session_history_sqlite(&path).unwrap();
    assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 4);

    // Each thread opens its own connection, covering the same transaction-contention paths across runtimes and processes.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let path = path.clone();
            std::thread::spawn(move || reserve_turn_index_sqlite(&path).unwrap())
        })
        .collect();
    let mut indexes: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    indexes.sort_unstable();
    assert_eq!(indexes, (5..13).collect::<Vec<_>>());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rollback_metadata_read_error_does_not_replace_live_history() {
    let dir = std::env::temp_dir().join(format!(
        "rollback_metadata_error_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let live = dir.join("live.sqlite");
    let checkpoint = dir.join("checkpoint.sqlite");
    append_history_sqlite(&live, vec![msg("user", "live message")]).unwrap();
    append_history_sqlite(&checkpoint, vec![msg("user", "checkpoint message")]).unwrap();
    let conn = Connection::open(&live).unwrap();
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('turn_seq', 'invalid')
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [],
    )
    .unwrap();
    drop(conn);

    let error = restore_sqlite_after_rollback(&checkpoint, &live).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    let messages = read_all_messages_sqlite(&live).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(value_to_string(&messages[0].content), "live message");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn restored_live_db_opens_with_a_fresh_wal_index() {
    let dir = std::env::temp_dir().join(format!(
        "rollback_fresh_wal_index_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let live = dir.join("live.sqlite");
    let checkpoint = dir.join("checkpoint.sqlite");
    append_history_sqlite(&live, vec![msg("user", "live")]).unwrap();
    append_history_sqlite(&checkpoint, vec![msg("user", "restored")]).unwrap();

    restore_sqlite_after_rollback(&checkpoint, &live).unwrap();

    // The working copy used to build the replacement must be fully closed
    // before it is renamed over the live path. A connection left in the
    // thread-local cache keeps the old, already-unlinked WAL index (shm)
    // mapping alive for the new live inode; a later connection in this
    // process then reuses that ghost index instead of creating the live
    // sidecar, so commits from other processes stay invisible here.
    let restored = Connection::open(&live).unwrap();
    let count: i64 = restored
        .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);
    assert!(
        dir.join("live.sqlite-shm").exists(),
        "opening the published DB must create a fresh WAL index for its inode"
    );
    drop(restored);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn staged_rollback_repeated_restore_preserves_snapshots_and_counters() {
    let dir = std::env::temp_dir().join(format!("staged_repeat_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let live = dir.join("live.sqlite");
    let mut stages = Vec::new();
    for name in ["b1", "b2"] {
        let source = dir.join(format!("{name}.sqlite"));
        let staged = dir.join(format!("{name}.stage.sqlite"));
        append_history_sqlite(&source, vec![msg("user", name)]).unwrap();
        backup_sqlite(&source, &staged).unwrap();
        let bytes = std::fs::read(&staged).unwrap();
        stages.push((staged, bytes, name));
    }
    let mut previous = (0, 0, 0);
    for i in 0..8 {
        let (staged, original, name) = &stages[i % 2];
        let transaction = format!("switch-{i}");
        restore_staged_sqlite_after_rollback(staged, &live, &transaction).unwrap();
        assert!(live_rollback_transaction_is_published(&live, &transaction).unwrap());
        assert_eq!(
            read_all_messages_sqlite(&live).unwrap(),
            vec![msg("user", name)]
        );
        let conn = Connection::open(&live).unwrap();
        let counter = |key: &str| -> i64 {
            conn.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key=?1",
                [key],
                |r| r.get(0),
            )
            .unwrap()
        };
        let current = (
            counter("history_generation"),
            counter("history_revision"),
            counter("turn_seq"),
        );
        assert!(current.0 > previous.0 && current.1 > previous.1);
        assert!(current.2 >= previous.2);
        previous = current;
        drop(conn);
        append_history_sqlite(&live, vec![msg("assistant", "live-only change")]).unwrap();
        reserve_turn_index_sqlite(&live).unwrap();
        assert_eq!(&std::fs::read(staged).unwrap(), original);
        assert_eq!(
            read_all_messages_sqlite_read_only(staged).unwrap(),
            vec![msg("user", name)]
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn staged_rollback_falls_back_for_uncheckpointed_wal() {
    let dir = std::env::temp_dir().join(format!("staged_wal_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let staged = dir.join("stage.sqlite");
    let live = dir.join("live.sqlite");
    append_history_sqlite(&staged, vec![msg("user", "checkpoint")]).unwrap();
    let writer = Connection::open(&staged).unwrap();
    writer
        .execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;")
        .unwrap();
    writer
        .execute(
            "INSERT INTO meta(key,value) VALUES('wal_only','committed')",
            [],
        )
        .unwrap();
    assert!(
        std::fs::metadata(dir.join("stage.sqlite-wal"))
            .unwrap()
            .len()
            > 0
    );

    restore_staged_sqlite_after_rollback(&staged, &live, "wal-fallback").unwrap();
    let restored = Connection::open(&live).unwrap();
    let value: String = restored
        .query_row("SELECT value FROM meta WHERE key='wal_only'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(value, "committed");
    assert!(live_rollback_transaction_is_published(&live, "wal-fallback").unwrap());
    assert_eq!(
        read_all_messages_sqlite(&live).unwrap(),
        vec![msg("user", "checkpoint")]
    );
    drop(restored);
    drop(writer);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn staged_rollback_busy_live_wal_is_retryable_without_publication() {
    let dir = std::env::temp_dir().join(format!("staged_busy_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let staged = dir.join("stage.sqlite");
    let live = dir.join("live.sqlite");
    append_history_sqlite(&staged, vec![msg("user", "checkpoint")]).unwrap();
    append_history_sqlite(&live, vec![msg("user", "live")]).unwrap();
    let reader = Connection::open(&live).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM messages;")
        .unwrap();
    append_history_sqlite(&live, vec![msg("assistant", "new live WAL")]).unwrap();
    let before = read_all_messages_sqlite(&live).unwrap();
    let error = restore_staged_sqlite_after_rollback(&staged, &live, "busy").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(read_all_messages_sqlite(&live).unwrap(), before);
    assert!(!live_rollback_transaction_is_published(&live, "busy").unwrap());
    drop(reader);
    restore_staged_sqlite_after_rollback(&staged, &live, "busy").unwrap();
    assert_eq!(
        read_all_messages_sqlite(&live).unwrap(),
        vec![msg("user", "checkpoint")]
    );
    assert!(live_rollback_transaction_is_published(&live, "busy").unwrap());
    let _ = std::fs::remove_dir_all(dir);
}

/// P1 regression: `read_history_revision` must observe the write increment **across connections**.
/// Every write path opens a new connection to read the version, mimicking build_context_history's cache-invalidation check.
/// The old implementation used the connection-local `PRAGMA data_version`, which always returns a fixed value on new connections and cannot invalidate the cache.
#[test]
fn history_revision_increments_across_fresh_connections() {
    let dir = std::env::temp_dir().join(format!(
        "hist_rev_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    // A brand-new DB has never written a revision: treat it as 0 (“never modified”).
    assert_eq!(read_history_revision(&path), Some(0));

    append_history_sqlite(&path, vec![msg("user", "hi")]).unwrap();
    let r1 = read_history_revision(&path).unwrap();
    assert!(r1 > 0, "append should bump revision, got {r1}");

    append_history_sqlite(&path, vec![msg("assistant", "hello")]).unwrap();
    let r2 = read_history_revision(&path).unwrap();
    assert!(r2 > r1, "second append should bump again: {r1} -> {r2}");

    replace_all_messages_sqlite(&path, &[msg("user", "reset")]).unwrap();
    let r3 = read_history_revision(&path).unwrap();
    assert!(r3 > r2, "replace should bump: {r2} -> {r3}");

    truncate_messages_sqlite(&path, 0).unwrap();
    let r4 = read_history_revision(&path).unwrap();
    assert!(r4 > r3, "truncate should bump: {r3} -> {r4}");

    // clear deletes the meta first and then bumps, a structure unlike the other write paths, so it needs its own coverage.
    append_history_sqlite(&path, vec![msg("user", "again")]).unwrap();
    let r5 = read_history_revision(&path).unwrap();
    clear_session_history_sqlite(&path).unwrap();
    let r6 = read_history_revision(&path).unwrap();
    assert!(
        r6 > r5,
        "clear should bump even after wiping meta: {r5} -> {r6}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn history_revision_cache_invalidates_on_wal_write_with_live_connection() {
    // Verify: with a live connection, WAL writes do not checkpoint the main DB, but the `-wal` sidecar changes
    // must invalidate the revision cache, or the cache would return a stale value.
    let dir = std::env::temp_dir().join(format!(
        "hist_rev_wal_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    append_history_sqlite(&path, vec![msg("user", "first")]).unwrap();
    let r1 = read_history_revision(&path).unwrap();
    assert!(r1 > 0);

    // Keep the connection alive to prevent the WAL checkpoint from writing back to the main DB
    let guard = rusqlite::Connection::open(&path).unwrap();

    // Short-lived connection writes: the WAL grows but the main DB mtime may stay unchanged
    append_history_sqlite(&path, vec![msg("user", "second")]).unwrap();

    // The cache must invalidate (WAL sidecar metadata changed) and return the new revision
    let r2 = read_history_revision(&path).unwrap();
    assert!(
        r2 > r1,
        "revision must increment even with live connection: {r1} -> {r2}"
    );

    drop(guard);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn history_connection_caps_wal_size_after_checkpoint() {
    let dir = std::env::temp_dir().join(format!(
        "hist_wal_limit_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let mut conn = open_history_db(&path).unwrap();
    assert_eq!(
        conn.query_row("PRAGMA journal_size_limit", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        super::connection::HISTORY_WAL_SIZE_LIMIT_BYTES,
        "every history connection must cap the retained WAL"
    );

    conn.execute_batch(
        "PRAGMA wal_autocheckpoint=0;
         CREATE TABLE payload(data BLOB);",
    )
    .unwrap();
    let tx = conn.transaction().unwrap();
    {
        let mut insert = tx
            .prepare("INSERT INTO payload(data) VALUES(randomblob(4096))")
            .unwrap();
        for _ in 0..2000 {
            insert.execute([]).unwrap();
        }
    }
    tx.commit().unwrap();

    let wal_path = dir.join("history.db-wal");
    let wal_size = std::fs::metadata(&wal_path).unwrap().len();
    assert!(wal_size > super::connection::HISTORY_WAL_SIZE_LIMIT_BYTES as u64);

    conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)").unwrap();
    assert_eq!(
        std::fs::metadata(&wal_path).unwrap().len(),
        wal_size,
        "a checkpoint resets the WAL but does not shrink the file by itself"
    );

    conn.execute("INSERT INTO payload(data) VALUES(randomblob(1))", [])
        .unwrap();
    assert!(
        std::fs::metadata(&wal_path).unwrap().len()
            <= super::connection::HISTORY_WAL_SIZE_LIMIT_BYTES as u64,
        "the first commit after a checkpoint reset must truncate the retained WAL"
    );
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM payload", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 2001, "truncation must not discard committed rows");

    drop(conn);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn llm_prune_marks_round_trip_and_clear_on_history_truncate() {
    let dir = std::env::temp_dir().join(format!(
        "llm_prune_marks_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.sqlite");
    append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

    let marks = [("call_b".to_string(), 2_u8), ("call_a".to_string(), 1_u8)]
        .into_iter()
        .collect::<FxHashMap<_, _>>();
    write_llm_prune_marks_sqlite(&path, &marks).unwrap();
    assert_eq!(read_llm_prune_marks_sqlite(&path).unwrap(), marks);

    truncate_messages_sqlite(&path, 0).unwrap();
    assert!(read_llm_prune_marks_sqlite(&path).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn prompt_residency_clears_when_truncation_removes_turns() {
    let dir = std::env::temp_dir().join(format!(
        "prompt_residency_truncate_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.sqlite");
    append_history_sqlite(
        &path,
        vec![
            msg("user", "first"),
            msg("assistant", "one"),
            msg("user", "second"),
            msg("assistant", "two"),
        ],
    )
    .unwrap();
    let residency = outcomes::PromptResidency {
        scoped_instruction_dirs: vec![PathBuf::from("/tmp/project/src")],
        session_context_guidance: true,
    };
    let write = || write_prompt_residency_sqlite(&path, &residency).unwrap();

    // Keeping every message removes nothing, so the residency still applies.
    write();
    truncate_messages_sqlite(&path, 4).unwrap();
    assert_eq!(read_prompt_residency_sqlite(&path).unwrap(), Some(residency.clone()));

    truncate_messages_to_user_turns_sqlite(&path, 1).unwrap();
    assert_eq!(read_prompt_residency_sqlite(&path).unwrap(), None);

    write();
    truncate_messages_sqlite(&path, 1).unwrap();
    assert_eq!(read_prompt_residency_sqlite(&path).unwrap(), None);

    write();
    truncate_messages_sqlite(&path, 0).unwrap();
    assert_eq!(read_prompt_residency_sqlite(&path).unwrap(), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stale_patch_targets_survive_history_replacement_and_clear_with_session() {
    let dir = std::env::temp_dir().join(format!(
        "stale_patch_meta_test_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.sqlite");
    append_history_sqlite(&path, vec![msg("user", "before compression")]).unwrap();

    let targets = FxHashSet::from_iter([PathBuf::from("/tmp/a.rs"), PathBuf::from("/tmp/b.rs")]);
    write_stale_patch_targets_sqlite(&path, &targets).unwrap();
    assert_eq!(
        read_stale_patch_targets_sqlite(&path).unwrap(),
        Some(targets.clone())
    );

    // replace_all_messages is the persisted-history compaction/rewrite path; the ledger must not be lost with the message shape.
    replace_all_messages_sqlite(&path, &[msg("user", "after compression")]).unwrap();
    assert_eq!(
        read_stale_patch_targets_sqlite(&path).unwrap(),
        Some(targets)
    );

    // An explicitly empty set must also be distinguished from “old DB with no meta yet”, so recovery never wrongly takes the legacy replay path.
    write_stale_patch_targets_sqlite(&path, &FxHashSet::default()).unwrap();
    assert_eq!(
        read_stale_patch_targets_sqlite(&path).unwrap(),
        Some(FxHashSet::default())
    );

    clear_session_history_sqlite(&path).unwrap();
    assert_eq!(read_stale_patch_targets_sqlite(&path).unwrap(), None);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn first_user_prompt_skips_preserved_content_notices() {
    let dir = std::env::temp_dir().join(format!(
        "first_prompt_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    append_history_sqlite(
        &path,
        vec![
            msg("user", "较早的用户图片内容已归档，原文未丢失。"),
            msg("user", r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image"}"#),
            msg("user", "较早的用户图片内容已归档，归档文件: /tmp/2"),
            msg("user", "较早的用户图片内容已归档，归档文件: /tmp/3"),
            msg("user", "较早的用户图片内容已归档，归档文件: /tmp/4"),
            msg("user", "这是实际用户请求"),
            msg("user", "这是后续用户请求"),
        ],
    )
    .unwrap();

    assert_eq!(
        read_first_user_prompt_sqlite(&path).unwrap().as_deref(),
        Some("这是实际用户请求\n---\n这是后续用户请求")
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn session_list_metadata_reads_fields_and_legacy_activity_without_creating_database() {
    let dir = std::env::temp_dir().join(format!(
        "session_list_metadata_test_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    let missing_path = dir.join("missing.db");
    assert!(read_session_list_metadata_sqlite(&missing_path).is_err());
    assert!(!missing_path.exists());

    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "first prompt")]).unwrap();
    write_session_title_sqlite(&path, "generated title", "model").unwrap();

    // Simulate an older session that has not written last_activity_unix_ms. The list must use the message time,
    // not the SQLite -shm file time, which a read-only connection creates/refreshes.
    let conn = open_history_db(&path).unwrap();
    conn.execute("UPDATE messages SET created_at = ?1", [1_700_000_000_i64])
        .unwrap();
    conn.execute("DELETE FROM meta WHERE key = 'last_activity_unix_ms'", [])
        .unwrap();
    drop(conn);

    let metadata = read_session_list_metadata_sqlite(&path).unwrap();
    assert_eq!(metadata.first_user_prompt.as_deref(), Some("first prompt"));
    assert_eq!(metadata.session_title.as_deref(), Some("generated title"));
    assert_eq!(metadata.last_activity_unix_ms, Some(1_700_000_000_000));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn structured_tool_outcomes_follow_message_lifecycle() {
    let dir = std::env::temp_dir().join(format!(
        "tool_outcome_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let original_messages = vec![tool_msg("call-1", "first"), tool_msg("call-2", "second")];
    append_history_sqlite(&path, original_messages.clone()).unwrap();
    let expected = vec![outcome("call-1"), outcome("call-2")];
    append_tool_execution_outcomes_sqlite(&path, &expected).unwrap();

    assert_eq!(
        read_tool_execution_outcomes_sqlite(&path).unwrap(),
        expected
    );
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), original_messages);

    replace_all_messages_sqlite(&path, &[tool_msg("call-2", "second")]).unwrap();
    assert_eq!(
        read_tool_execution_outcomes_sqlite(&path).unwrap(),
        vec![outcome("call-2")]
    );

    append_history_sqlite(&path, vec![tool_msg("call-3", "third")]).unwrap();
    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-3")]).unwrap();
    truncate_messages_sqlite(&path, 1).unwrap();
    assert_eq!(
        read_tool_execution_outcomes_sqlite(&path).unwrap(),
        vec![outcome("call-2")]
    );

    clear_session_history_sqlite(&path).unwrap();
    assert!(
        read_tool_execution_outcomes_sqlite(&path)
            .unwrap()
            .is_empty()
    );

    // Older history may have reused the same ID; before the message set changes, its ambiguous outcome must be permanently discarded,
    // otherwise deleting the newer occurrence would wrongly bind its state to the retained older message.
    append_history_sqlite(
        &path,
        vec![
            tool_msg("legacy-reused", "older"),
            tool_msg("legacy-reused", "newer"),
        ],
    )
    .unwrap();
    append_tool_execution_outcomes_sqlite(&path, &[outcome("legacy-reused")]).unwrap();
    replace_all_messages_sqlite(&path, &[tool_msg("legacy-reused", "older")]).unwrap();
    assert!(
        read_tool_execution_outcomes_sqlite(&path)
            .unwrap()
            .is_empty()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn skill_activation_events_preserve_injection_audit_without_messages() {
    let dir = std::env::temp_dir().join(format!(
        "skill_activation_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let event = SkillActivationEvent {
        requested_skill: "bytedcli".to_string(),
        injected_skill: Some("bytedcli".to_string()),
        source: "/skills-inline".to_string(),
        outcome: "injected".to_string(),
    };

    append_skill_activation_event_sqlite(&path, &event).unwrap();

    assert_eq!(
        read_skill_activation_events_sqlite(&path).unwrap(),
        vec![event]
    );
    assert!(read_all_messages_sqlite(&path).unwrap().is_empty());

    clear_session_history_sqlite(&path).unwrap();
    assert!(
        read_skill_activation_events_sqlite(&path)
            .unwrap()
            .is_empty()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn interrupted_stream_diagnostics_are_isolated_from_model_history() {
    let dir = std::env::temp_dir().join(format!(
        "interrupted_stream_diagnostic_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    append_interrupted_stream_diagnostic_sqlite(
        &path,
        "test-model",
        "partial body",
        "partial reasoning",
    )
    .unwrap();

    assert!(read_all_messages_sqlite(&path).unwrap().is_empty());
    let conn = Connection::open(&path).unwrap();
    let row: (String, String, String) = conn
        .query_row(
            "SELECT assistant_text, reasoning_text, source_model
             FROM interrupted_stream_diagnostics",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        row,
        (
            "partial body".to_string(),
            "partial reasoning".to_string(),
            "test-model".to_string(),
        )
    );

    clear_session_history_sqlite(&path).unwrap();
    let remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM interrupted_stream_diagnostics",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(remaining, 0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn structured_tool_outcomes_ignore_non_sqlite_history_files() {
    let dir = std::env::temp_dir().join(format!(
        "tool_outcome_text_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.txt");
    std::fs::write(&path, "plain text history\n").unwrap();

    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1")]).unwrap();

    assert!(
        read_tool_execution_outcomes_sqlite(&path)
            .unwrap()
            .is_empty()
    );
    assert!(read_tool_message_ids_sqlite(&path).unwrap().is_empty());
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "plain text history\n"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn truncating_by_user_turn_prunes_removed_tool_outcomes() {
    let dir = std::env::temp_dir().join(format!(
        "tool_outcome_turn_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(
        &path,
        vec![
            msg("user", "first turn"),
            tool_msg("call-1", "first result"),
            msg("user", "second turn"),
            tool_msg("call-2", "second result"),
        ],
    )
    .unwrap();
    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1"), outcome("call-2")]).unwrap();

    truncate_messages_to_user_turns_sqlite(&path, 1).unwrap();

    assert_eq!(
        read_tool_execution_outcomes_sqlite(&path).unwrap(),
        vec![outcome("call-1")]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn canonical_messages_are_unchanged_by_context_snapshots() {
    const PROJECTION: &str = "test-policy-a";
    let dir = std::env::temp_dir().join(format!(
        "context_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String("准备读取文件".to_string()),
        tool_calls: Some(vec![ToolCall {
            id: "call-1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some("provider 原始 continuation state".to_string()),
    };
    let original = vec![msg("user", "读取文件"), assistant.clone()];
    append_history_sqlite_for_model(&path, original.clone(), Some("glm-5.2-opencode")).unwrap();

    let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);
    assert_ne!(
        source.messages[1].reasoning_content,
        assistant.reasoning_content
    );

    let snapshot = vec![msg(ROLE_INTERNAL_NOTE, "[History summary]\n已读取文件")];
    write_context_snapshot_sqlite(
        &path,
        &snapshot,
        source.source_message_id,
        source.canonical_generation,
        PROJECTION,
    )
    .unwrap();
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);

    let tail = msg("user", "继续分析");
    append_history_sqlite_for_model(&path, vec![tail.clone()], Some("gpt-5.5")).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![snapshot[0].clone(), tail.clone()]);
    assert!(!projected.snapshot_is_current);

    let mut expected_canonical = original;
    expected_canonical.push(tail);
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), expected_canonical);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn context_snapshot_guidance_cleanup_preserves_canonical_and_protocol() {
    const POLICY: &str = "guidance-projection";
    let dir = std::env::temp_dir().join(format!("guidance-snapshot-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let guidance = msg(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nRead serially.",
    );
    let revised = msg(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nKeep full evidence.",
    );
    let assistant = Message {
        tool_calls: Some(vec![ToolCall {
            id: "call-guidance".into(),
            tool_type: "function".into(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".into(),
                arguments: "{}".into(),
            },
        }]),
        ..msg("assistant", "Read the evidence")
    };
    let tool = Message {
        tool_call_id: Some("call-guidance".into()),
        ..msg("tool", "Exact evidence")
    };
    let original = vec![
        msg("user", "old"),
        guidance.clone(),
        msg("user", "current"),
        guidance.clone(),
        guidance.clone(),
        revised.clone(),
        assistant.clone(),
        tool.clone(),
    ];
    append_history_sqlite_for_model(&path, original.clone(), None).unwrap();
    let source = read_context_history_sqlite(&path, POLICY).unwrap();
    // Pass unclean input to prove the write boundary itself filters it.
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &original,
            source.source_message_id,
            source.canonical_generation,
            POLICY
        )
        .unwrap()
    );
    let conn = Connection::open(&path).unwrap();
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM context_messages", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 6);
    let restored = read_context_history_sqlite(&path, POLICY).unwrap();
    assert!(restored.snapshot_is_current);
    assert_eq!(
        restored.messages,
        vec![
            msg("user", "old"),
            msg("user", "current"),
            guidance,
            revised,
            assistant,
            tool
        ]
    );
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn context_snapshot_guidance_cleanup_handles_legacy_snapshot_and_new_tail() {
    const POLICY: &str = "guidance-legacy";
    let dir = std::env::temp_dir().join(format!("guidance-legacy-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let user = msg("user", "first request");
    let guidance = msg(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nRead serially.",
    );
    let original = vec![user.clone(), guidance.clone(), guidance.clone()];
    append_history_sqlite_for_model(&path, original.clone(), None).unwrap();
    let source = read_context_history_sqlite(&path, POLICY).unwrap();
    write_context_snapshot_sqlite(
        &path,
        &source.messages,
        source.source_message_id,
        source.canonical_generation,
        POLICY,
    )
    .unwrap();
    // Reproduce an older snapshot containing exact duplicate guidance.
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "INSERT INTO context_messages
        SELECT 2, role, content, tool_calls, tool_call_id, reasoning_content
        FROM context_messages WHERE position = 1",
        [],
    )
    .unwrap();
    let legacy = read_context_history_sqlite(&path, POLICY).unwrap();
    assert_eq!(legacy.messages, vec![user.clone(), guidance.clone()]);
    assert!(!legacy.snapshot_is_current);
    let synthetic = crate::ai::history::runtime_synthetic_user_message("task completed".into());
    append_history_sqlite_for_model(&path, vec![synthetic.clone()], None).unwrap();
    assert_eq!(
        read_context_history_sqlite(&path, POLICY).unwrap().messages,
        vec![user.clone(), guidance.clone(), synthetic.clone()]
    );
    let next_user = msg("user", "next request");
    append_history_sqlite_for_model(&path, vec![next_user.clone(), guidance.clone()], None)
        .unwrap();
    let merged = read_context_history_sqlite(&path, POLICY).unwrap();
    assert_eq!(
        merged.messages,
        vec![user, synthetic.clone(), next_user.clone(), guidance.clone()]
    );
    let mut expected = original;
    expected.extend([synthetic, next_user, guidance]);
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), expected);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn stale_context_snapshot_cannot_resurrect_truncated_history() {
    const PROJECTION: &str = "test-policy-a";
    let dir = std::env::temp_dir().join(format!(
        "stale_context_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "保留"), msg("assistant", "删除")]).unwrap();
    let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

    truncate_messages_sqlite(&path, 1).unwrap();
    let written = write_context_snapshot_sqlite(
        &path,
        &[msg(ROLE_INTERNAL_NOTE, "包含已删除内容的旧快照")],
        stale_source.source_message_id,
        stale_source.canonical_generation,
        PROJECTION,
    )
    .unwrap();
    assert!(!written);
    assert_eq!(
        read_context_history_sqlite(&path, PROJECTION)
            .unwrap()
            .messages,
        vec![msg("user", "保留")]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewind_after_snapshot_watermark_reuses_covered_projection() {
    const PROJECTION: &str = "rewind-covered";
    let dir = std::env::temp_dir().join(format!(
        "rewind_covered_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let first_turn = vec![msg("user", "first"), msg("assistant", "first answer")];
    append_history_sqlite(&path, first_turn.clone()).unwrap();
    let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let summary = msg(ROLE_INTERNAL_NOTE, "[summary] first turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&summary),
            source.source_message_id,
            source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );
    // The next turn is appended after the snapshot watermark; rewinding it
    // leaves every canonical message covered by the snapshot intact.
    append_history_sqlite(
        &path,
        vec![msg("user", "second"), msg("assistant", "second answer")],
    )
    .unwrap();
    truncate_messages_sqlite(&path, first_turn.len()).unwrap();

    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![summary]);
    assert!(projected.snapshot_is_current);
    assert_eq!(projected.source_message_id, source.source_message_id);
    assert!(projected.canonical_generation > source.canonical_generation);
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), first_turn);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewind_last_compacted_turn_restores_archived_projection() {
    const PROJECTION: &str = "rewind-previous";
    let dir = std::env::temp_dir().join(format!(
        "rewind_previous_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let first_turn = vec![msg("user", "first"), msg("assistant", "first answer")];
    append_history_sqlite(&path, first_turn.clone()).unwrap();
    let first_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let first_summary = msg(ROLE_INTERNAL_NOTE, "[summary] first turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&first_summary),
            first_source.source_message_id,
            first_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // Completing and compacting the second turn replaces the first-turn
    // snapshot, which is retained as the rewind fallback.
    append_history_sqlite(
        &path,
        vec![msg("user", "second"), msg("assistant", "second answer")],
    )
    .unwrap();
    let second_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let second_summary = msg(ROLE_INTERNAL_NOTE, "[summary] second turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&second_summary),
            second_source.source_message_id,
            second_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // Rewinding the second turn must promote the retained first-turn
    // projection instead of rebuilding from canonical history.
    truncate_messages_sqlite(&path, first_turn.len()).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![first_summary]);
    assert!(projected.snapshot_is_current);
    assert_eq!(projected.source_message_id, first_source.source_message_id);
    assert!(projected.canonical_generation > second_source.canonical_generation);
    assert_eq!(read_all_messages_sqlite(&path).unwrap(), first_turn);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewind_absorbs_surviving_tail_into_retained_projection() {
    const PROJECTION: &str = "rewind-tail";
    let dir = std::env::temp_dir().join(format!(
        "rewind_tail_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let first_turn = vec![msg("user", "first"), msg("assistant", "first answer")];
    append_history_sqlite(&path, first_turn.clone()).unwrap();
    let first_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let first_summary = msg(ROLE_INTERNAL_NOTE, "[summary] first turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&first_summary),
            first_source.source_message_id,
            first_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    let second_turn = vec![msg("user", "second"), msg("assistant", "second answer")];
    append_history_sqlite(&path, second_turn.clone()).unwrap();
    let second_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let second_summary = msg(ROLE_INTERNAL_NOTE, "[summary] second turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&second_summary),
            second_source.source_message_id,
            second_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // The third turn starts past the snapshot watermark; rewinding only its
    // assistant reply leaves the user message as a surviving tail.
    append_history_sqlite(
        &path,
        vec![msg("user", "third"), msg("assistant", "third answer")],
    )
    .unwrap();
    let kept = first_turn.len() + second_turn.len() + 1;
    truncate_messages_sqlite(&path, kept).unwrap();

    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![second_summary, msg("user", "third")]
    );
    assert!(projected.snapshot_is_current);
    assert_eq!(
        projected.source_message_id,
        second_source.source_message_id + 1
    );
    assert!(projected.canonical_generation > second_source.canonical_generation);
    assert_eq!(read_all_messages_sqlite(&path).unwrap().len(), kept);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewind_promotes_archived_projection_and_absorbs_tail() {
    const PROJECTION: &str = "rewind-promoted-tail";
    let dir = std::env::temp_dir().join(format!(
        "rewind_promoted_tail_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let first_turn = vec![msg("user", "first"), msg("assistant", "first answer")];
    append_history_sqlite(&path, first_turn.clone()).unwrap();
    let first_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let first_summary = msg(ROLE_INTERNAL_NOTE, "[summary] first turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&first_summary),
            first_source.source_message_id,
            first_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    let second_turn = vec![msg("user", "second"), msg("assistant", "second answer")];
    append_history_sqlite(&path, second_turn).unwrap();
    let second_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let second_summary = msg(ROLE_INTERNAL_NOTE, "[summary] second turn");
    assert!(
        write_context_snapshot_sqlite(
            &path,
            std::slice::from_ref(&second_summary),
            second_source.source_message_id,
            second_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // Keep the first turn plus the second turn's user message. The current
    // snapshot covers the removed reply, so the rewind falls back to the
    // previous snapshot and must absorb that user message into it.
    let kept = first_turn.len() + 1;
    truncate_messages_sqlite(&path, kept).unwrap();

    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![first_summary, msg("user", "second")]);
    assert!(projected.snapshot_is_current);
    assert_eq!(
        projected.source_message_id,
        second_source.source_message_id - 1
    );
    assert_eq!(read_all_messages_sqlite(&path).unwrap().len(), kept);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn consecutive_rewinds_reuse_archived_snapshots() {
    const PROJECTION: &str = "rewind-ring";
    let dir = std::env::temp_dir().join(format!(
        "consecutive_rewind_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    let mut summaries = Vec::new();
    for turn in 1..=4 {
        append_history_sqlite(
            &path,
            vec![
                msg("user", &format!("user {turn}")),
                msg("assistant", &format!("answer {turn}")),
            ],
        )
        .unwrap();
        let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
        let summary = msg(ROLE_INTERNAL_NOTE, &format!("[summary] turn {turn}"));
        assert!(
            write_context_snapshot_sqlite(
                &path,
                std::slice::from_ref(&summary),
                source.source_message_id,
                source.canonical_generation,
                PROJECTION,
            )
            .unwrap()
        );
        summaries.push(summary);
    }

    // The first rewind cuts inside the newest snapshot: the ring promotes turn
    // 3's projection and absorbs the surviving user message.
    truncate_messages_sqlite(&path, 7).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![summaries[2].clone(), msg("user", "user 4")]
    );
    assert!(projected.snapshot_is_current);

    // The next rewind moves below the promoted watermark; turn 2's boundary is
    // still archived, so the projection is reused again.
    truncate_messages_sqlite(&path, 5).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![summaries[1].clone(), msg("user", "user 3")]
    );
    assert!(projected.snapshot_is_current);

    // The oldest archived boundary still covers the cutoff.
    truncate_messages_sqlite(&path, 3).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![summaries[0].clone(), msg("user", "user 2")]
    );
    assert!(projected.snapshot_is_current);

    // Past the oldest boundary no archived projection can cover the cutoff.
    truncate_messages_sqlite(&path, 1).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![msg("user", "user 1")]);
    assert!(!projected.snapshot_is_current);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewind_keeps_promoted_projection_for_a_deeper_rewind() {
    const PROJECTION: &str = "rewind-promoted-ring";
    let dir = std::env::temp_dir().join(format!(
        "rewind_promoted_ring_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let mut summaries = Vec::new();
    for turn in 1..=4 {
        append_history_sqlite(
            &path,
            vec![
                msg("user", &format!("user {turn}")),
                msg("assistant", &format!("answer {turn}")),
            ],
        )
        .unwrap();
        // Snapshot only the first and the last boundary so the ring holds one
        // entry after the second write.
        if turn == 1 || turn == 4 {
            let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
            let summary = msg(ROLE_INTERNAL_NOTE, &format!("[summary] turn {turn}"));
            assert!(
                write_context_snapshot_sqlite(
                    &path,
                    std::slice::from_ref(&summary),
                    source.source_message_id,
                    source.canonical_generation,
                    PROJECTION,
                )
                .unwrap()
            );
            summaries.push(summary);
        }
    }

    // The first rewind promotes the only archived boundary and absorbs the
    // surviving tail. The promoted row must stay archived for the next cut.
    truncate_messages_sqlite(&path, 5).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![
            summaries[0].clone(),
            msg("user", "user 2"),
            msg("assistant", "answer 2"),
            msg("user", "user 3"),
        ]
    );
    assert!(projected.snapshot_is_current);

    // Cutting below the extended watermark still finds the promoted boundary.
    truncate_messages_sqlite(&path, 3).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(
        projected.messages,
        vec![summaries[0].clone(), msg("user", "user 2")]
    );
    assert!(projected.snapshot_is_current);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rollback_clears_context_snapshot_history() {
    const PROJECTION: &str = "rollback-previous";
    let dir = std::env::temp_dir().join(format!(
        "rollback_previous_snapshot_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(
        &path,
        vec![msg("user", "first"), msg("assistant", "first answer")],
    )
    .unwrap();
    let first_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "[summary] first turn")],
            first_source.source_message_id,
            first_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );
    append_history_sqlite(
        &path,
        vec![msg("user", "second"), msg("assistant", "second answer")],
    )
    .unwrap();
    let second_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "[summary] second turn")],
            second_source.source_message_id,
            second_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // A checkpoint restore rewrites canonical history; every derived snapshot
    // must go, including the archived rewind ring.
    let conn = open_history_db(&path).unwrap();
    let generation_before = super::revision::history_generation(&conn).unwrap();
    let archived: i64 = conn
        .query_row("SELECT COUNT(*) FROM context_snapshot_history", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        archived, 1,
        "context_snapshot_history should hold an archive before the rebase"
    );
    drop(conn);
    super::rollback::rebase_metadata_after_rollback(
        &path,
        second_source.canonical_generation,
        0,
        0,
        None,
    )
    .unwrap();
    let conn = open_history_db(&path).unwrap();
    assert!(super::revision::history_generation(&conn).unwrap() > generation_before);
    for table in [
        "context_snapshot",
        "context_messages",
        "context_snapshot_history",
    ] {
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0, "{table} should be cleared by the rollback rebase");
    }
    drop(conn);
    // With no derived snapshot left, the next read rebuilds from canonical
    // history instead of serving the pre-rollback projection.
    let rebuilt = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(rebuilt.messages.len(), 4);
    assert!(
        rebuilt
            .messages
            .iter()
            .all(|message| message.role != ROLE_INTERNAL_NOTE)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn context_snapshot_is_ignored_when_projection_policy_changes() {
    let dir = std::env::temp_dir().join(format!(
        "context_snapshot_policy_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let canonical = vec![msg("user", "必须从 canonical 重建")];
    append_history_sqlite(&path, canonical.clone()).unwrap();

    let source = read_context_history_sqlite(&path, "policy-a").unwrap();
    write_context_snapshot_sqlite(
        &path,
        &[msg(ROLE_INTERNAL_NOTE, "旧策略摘要")],
        source.source_message_id,
        source.canonical_generation,
        "policy-a",
    )
    .unwrap();
    assert!(
        read_context_history_sqlite(&path, "policy-a")
            .unwrap()
            .snapshot_is_current
    );

    let rebuilt = read_context_history_sqlite(&path, "policy-b").unwrap();
    assert_eq!(rebuilt.messages, canonical);
    assert!(!rebuilt.snapshot_is_current);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stale_context_snapshot_cannot_resurrect_cleared_history() {
    const PROJECTION: &str = "test-policy-a";
    let dir = std::env::temp_dir().join(format!(
        "stale_context_snapshot_clear_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "已经清除")]).unwrap();
    let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

    clear_session_history_sqlite(&path).unwrap();
    let current = msg("user", "清除后的新会话");
    append_history_sqlite(&path, vec![current.clone()]).unwrap();
    let written = write_context_snapshot_sqlite(
        &path,
        &[msg(ROLE_INTERNAL_NOTE, "包含已清除内容的旧快照")],
        stale_source.source_message_id,
        stale_source.canonical_generation,
        PROJECTION,
    )
    .unwrap();

    assert!(!written);
    assert_eq!(
        read_context_history_sqlite(&path, PROJECTION)
            .unwrap()
            .messages,
        vec![current]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn truncate_after_clear_cannot_promote_pre_clear_ring_entry() {
    const PROJECTION: &str = "test-policy-a";
    let dir = std::env::temp_dir().join(format!(
        "truncate_after_clear_ring_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "before clear")]).unwrap();
    let cleared_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "[summary] cleared turn")],
            cleared_source.source_message_id,
            cleared_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    clear_session_history_sqlite(&path).unwrap();
    let surviving = msg("user", "after clear");
    append_history_sqlite(&path, vec![surviving.clone()]).unwrap();
    // AUTOINCREMENT keeps the new row's id above the cleared snapshot's
    // watermark, so a watermark-only promotion would treat the pre-clear ring
    // entry as a projection covering the surviving prefix.
    truncate_messages_sqlite(&path, 1).unwrap();

    let conn = open_history_db(&path).unwrap();
    let archived: i64 = conn
        .query_row("SELECT COUNT(*) FROM context_snapshot_history", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(archived, 0, "the ring must stay empty after a clear");
    drop(conn);
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, vec![surviving]);
    assert!(!projected.snapshot_is_current);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ring_entries_store_one_compressed_payload_and_promote_exactly() {
    use crate::ai::types::{FunctionCall, ToolCall};

    const PROJECTION: &str = "ring-payload";
    let dir = std::env::temp_dir().join(format!(
        "ring_payload_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let first_turn = vec![msg("user", "first"), msg("assistant", "first answer")];
    append_history_sqlite(&path, first_turn).unwrap();
    let first_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    let long_summary = format!("[summary] {}", "first turn detail ".repeat(200));
    let first_summary = msg(ROLE_INTERNAL_NOTE, &long_summary);
    let mut tool_turn = msg("assistant", "read the file");
    tool_turn.tool_calls = Some(vec![ToolCall {
        id: "call-1".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{\"path\":\"a.txt\"}".to_string(),
        },
    }]);
    tool_turn.reasoning_content = Some("the file holds the answer".to_string());
    let tool_result = tool_msg("call-1", "file body");
    let first_projection = vec![first_summary.clone(), tool_turn.clone(), tool_result.clone()];
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &first_projection,
            first_source.source_message_id,
            first_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );
    let archived_rows: i64 = open_history_db(&path)
        .unwrap()
        .query_row(
            "SELECT COALESCE(SUM(LENGTH(role) + LENGTH(content) + LENGTH(IFNULL(tool_calls, ''))
                + LENGTH(IFNULL(tool_call_id, '')) + LENGTH(IFNULL(reasoning_content, ''))), 0)
             FROM context_messages",
            [],
            |row| row.get(0),
        )
        .unwrap();

    append_history_sqlite(&path, vec![msg("user", "second")]).unwrap();
    let second_source = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert!(
        write_context_snapshot_sqlite(
            &path,
            &[
                first_summary,
                tool_turn,
                tool_result,
                msg("user", "second"),
            ],
            second_source.source_message_id,
            second_source.canonical_generation,
            PROJECTION,
        )
        .unwrap()
    );

    // The replaced projection is one payload on its ring row, and gzip keeps it
    // smaller than the row text it covers.
    let conn = open_history_db(&path).unwrap();
    let payload: Vec<u8> = conn
        .query_row("SELECT payload FROM context_snapshot_history", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(
        (payload.len() as i64) < archived_rows,
        "payload {} should be smaller than {archived_rows} bytes of archived rows",
        payload.len()
    );
    let legacy_table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'context_messages_history'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(legacy_table, 0, "the per-row ring table is retired");
    drop(conn);

    // Promotion restores the archived projection exactly.
    truncate_messages_sqlite(&path, 2).unwrap();
    let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
    assert_eq!(projected.messages, first_projection);
    assert!(projected.snapshot_is_current);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v1_ring_rows_are_dropped_when_payloads_replace_the_row_table() {
    let dir = std::env::temp_dir().join(format!(
        "ring_payload_migration_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE context_snapshot_history (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_message_id INTEGER NOT NULL,
            projection_fingerprint TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE context_messages_history (
            snapshot_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT,
            PRIMARY KEY (snapshot_id, position)
        );
        INSERT INTO context_snapshot_history (source_message_id, projection_fingerprint)
            VALUES (7, 'v1');
        INSERT INTO context_messages_history (snapshot_id, position, role, content)
            VALUES (1, 0, 'assistant', 'old');
        PRAGMA user_version = 1;",
    )
    .unwrap();

    super::migrations::init_history_schema(&conn).unwrap();

    let legacy_table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'context_messages_history'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(legacy_table, 0);
    let ring: i64 = conn
        .query_row("SELECT COUNT(*) FROM context_snapshot_history", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        ring, 0,
        "an entry without a payload cannot be promoted and must not linger"
    );
    let payload_column: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('context_snapshot_history')
             WHERE name = 'payload'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(payload_column, 1);
    let version: i32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, super::migrations::HISTORY_SCHEMA_VERSION);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ring_prune_keeps_only_the_newest_eight_payloads() {
    const PROJECTION: &str = "ring-prune";
    let dir = std::env::temp_dir().join(format!(
        "ring_prune_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "start")]).unwrap();
    for turn in 0..10 {
        let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
        append_history_sqlite(&path, vec![msg("user", &format!("turn {turn}"))]).unwrap();
        assert!(
            write_context_snapshot_sqlite(
                &path,
                &[msg(ROLE_INTERNAL_NOTE, &format!("[summary] turn {turn}"))],
                source.source_message_id,
                source.canonical_generation,
                PROJECTION,
            )
            .unwrap()
        );
    }

    let conn = open_history_db(&path).unwrap();
    let (rows, newest, unpromotable): (i64, i64, i64) = conn
        .query_row(
            "SELECT COUNT(*), COALESCE(MAX(source_message_id), 0),
                    COALESCE(SUM(payload IS NULL), 0)
             FROM context_snapshot_history",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(rows, 8, "the ring keeps eight fallbacks");
    assert_eq!(newest, 9, "the live projection is not a ring entry");
    assert_eq!(unpromotable, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rollback_state_lock_serializes_every_public_side_state_writer() {
    let dir = std::env::temp_dir().join(format!(
        "history_side_writer_lock_test_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

    type Writer = Box<dyn FnOnce(&Path) -> io::Result<()> + Send>;
    let writers: Vec<(&str, Writer)> = vec![
        (
            "tool outcomes",
            Box::new(|path| append_tool_execution_outcomes_sqlite(path, &[outcome("call-1")])),
        ),
        (
            "stale patch targets",
            Box::new(|path| {
                let mut targets = FxHashSet::default();
                targets.insert(PathBuf::from("src/main.rs"));
                write_stale_patch_targets_sqlite(path, &targets)
            }),
        ),
        (
            "context snapshot",
            Box::new(|path| {
                let source = read_context_history_sqlite(path, "lock-test")?;
                write_context_snapshot_sqlite(
                    path,
                    &[msg(ROLE_INTERNAL_NOTE, "snapshot")],
                    source.source_message_id,
                    source.canonical_generation,
                    "lock-test",
                )
                .map(|_| ())
            }),
        ),
        (
            "session title",
            Box::new(|path| write_session_title_sqlite(path, "title", "test")),
        ),
    ];

    for (name, writer) in writers {
        let (lock_held_tx, lock_held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let lock_path = path.clone();
        let lock_thread = std::thread::spawn(move || {
            with_session_state_lock(&lock_path, || {
                lock_held_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        lock_held_rx.recv().unwrap();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let writer_path = path.clone();
        let writer_thread = std::thread::spawn(move || {
            let result = writer(&writer_path);
            done_tx.send(()).unwrap();
            result
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "{name} bypassed the rollback state lock"
        );

        release_tx.send(()).unwrap();
        lock_thread.join().unwrap().unwrap();
        writer_thread.join().unwrap().unwrap();
    }

    let _ = std::fs::remove_dir_all(&dir);
}

fn wake_note_text(pid: u64, ids: &[&str], checkpoint: &str) -> String {
    format!(
        "[Process {pid} Woke Up] Original goal: test goal\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after 30s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]\nProgress: {checkpoint}\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
        ids.join(", ")
    )
}

#[test]
fn wait_wake_notes_coalesce_keeps_latest_in_sqlite_history() {
    let dir = std::env::temp_dir().join(format!(
        "wait_wake_coalesce_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    append_history_sqlite(
        &path,
        vec![
            msg("user", "goal"),
            msg(
                ROLE_INTERNAL_NOTE,
                &wake_note_text(6, &["task_a", "task_b"], "checkpoint-1"),
            ),
            msg(
                ROLE_INTERNAL_NOTE,
                &wake_note_text(6, &["task_a", "task_b"], "checkpoint-2"),
            ),
            msg(
                ROLE_INTERNAL_NOTE,
                &wake_note_text(7, &["task_x"], "checkpoint-3"),
            ),
        ],
    )
    .unwrap();

    let latest = msg(
        ROLE_INTERNAL_NOTE,
        &wake_note_text(6, &["task_a", "task_b"], "checkpoint-4"),
    );
    assert!(coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());
    // The caller then appends the latest one at the tail.
    append_history_sqlite(&path, vec![latest]).unwrap();

    let notes: Vec<_> = read_all_messages_sqlite(&path)
        .unwrap()
        .into_iter()
        .filter(|m| m.role == ROLE_INTERNAL_NOTE)
        .collect();
    assert_eq!(notes.len(), 2);
    assert!(notes[0].content.as_str().unwrap().contains("checkpoint-3"));
    assert!(notes[1].content.as_str().unwrap().contains("checkpoint-4"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wait_wake_coalesce_sqlite_window_is_last_total_messages() {
    let dir = std::env::temp_dir().join(format!(
        "sqlite_wake_dedup_window_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    // Window semantics consistent with the blob backend: scan the last WAKE_NOTE_DEDUP_SCAN messages of the history (any role),
    // not “the most recent WAKE_NOTE_DEDUP_SCAN internal_notes”.
    // The old waiting note is at position 1, followed by WAKE_NOTE_DEDUP_SCAN+1 user messages, so it is outside the window
    // and must not be deleted (an internal_note-window scan would have hit and removed it).
    let mut history = vec![msg(
        ROLE_INTERNAL_NOTE,
        &wake_note_text(6, &["task_a"], "checkpoint-old"),
    )];
    history.extend(
        (0..crate::ai::history::types::WAKE_NOTE_DEDUP_SCAN as usize + 1)
            .map(|i| msg("user", &format!("filler {i}"))),
    );
    append_history_sqlite(&path, history).unwrap();

    let latest = msg(
        ROLE_INTERNAL_NOTE,
        &wake_note_text(6, &["task_a"], "checkpoint-new"),
    );
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());

    let notes: Vec<_> = read_all_messages_sqlite(&path)
        .unwrap()
        .into_iter()
        .filter(|m| m.role == ROLE_INTERNAL_NOTE)
        .collect();
    // The old waiting note outside the window is retained and not wrongly deleted.
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0]
            .content
            .as_str()
            .unwrap()
            .contains("checkpoint-old")
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wait_wake_coalesce_sqlite_is_noop_when_nothing_matches() {
    let dir = std::env::temp_dir().join(format!(
        "wait_wake_coalesce_noop_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    append_history_sqlite(
        &path,
        vec![
            msg("user", "goal"),
            msg(
                ROLE_INTERNAL_NOTE,
                &wake_note_text(6, &["task_a"], "checkpoint-1"),
            ),
        ],
    )
    .unwrap();

    // Same pid but a different task set: the identity differs, so no dedup.
    let other = msg(
        ROLE_INTERNAL_NOTE,
        &wake_note_text(6, &["task_z"], "checkpoint-x"),
    );
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &other).unwrap());

    // Non-internal_note message: the fast path does no IO.
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &msg("user", "hello")).unwrap());

    // A real-result wake (parse returns None): no dedup.
    let result_wake = msg(
        ROLE_INTERNAL_NOTE,
        "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
    );
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &result_wake).unwrap());

    // Missing database: best-effort returns false without error.
    // Note: only a valid wait note gets past the fast path to actually reach the open_history_db branch.
    let missing = dir.join("missing.db");
    let wait_note = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "c"));
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&missing, &wait_note).unwrap());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn read_all_messages_with_models_self_heals_fresh_and_pre_migration_dbs() {
    let dir = std::env::temp_dir().join(format!(
        "model_read_self_heal_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    // Brand-new path: no schema yet -> the reader must return empty, not error
    // (the messages table only exists after init_history_schema runs).
    let fresh = dir.join("fresh.sqlite");
    assert_eq!(
        read_all_messages_with_models_sqlite(&fresh).unwrap(),
        vec![]
    );

    // Pre-migration DB: messages table without the source_model column -> the
    // reader self-heals via the column migration instead of failing.
    let legacy = dir.join("legacy.sqlite");
    let conn = Connection::open(&legacy).unwrap();
    conn.execute_batch(
        "CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        INSERT INTO messages (role, content) VALUES ('assistant', '\"hello\"');",
    )
    .unwrap();
    drop(conn);

    let pairs = read_all_messages_with_models_sqlite(&legacy).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0.role, "assistant");
    assert_eq!(pairs[0].0.content, Value::String("hello".to_string()));
    assert_eq!(pairs[0].1, None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn session_origin_reads_first_real_exchange_verbatim_past_synthetic_and_tool_rows() {
    let dir = std::env::temp_dir().join(format!(
        "session_origin_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let mut tool_call_assistant = msg("assistant", "");
    tool_call_assistant.tool_calls = Some(vec![crate::ai::types::ToolCall {
        id: "call-1".to_string(),
        tool_type: "function".to_string(),
        function: crate::ai::types::FunctionCall {
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        },
    }]);
    let mut messages = vec![
        crate::ai::history::runtime_synthetic_user_message(Value::String("handoff".into())),
        msg("user", "第一轮：帮我看下 src/main.rs"),
        tool_call_assistant,
        tool_msg("call-1", "fn main() {}"),
        msg("assistant", "main 只有一个空函数。"),
    ];
    for i in 0..1_000 {
        messages.push(msg("user", &format!("turn {i}")));
        messages.push(msg("assistant", &format!("reply {i}")));
    }
    append_history_sqlite(&path, messages).unwrap();

    let (index, prompt, reply) = read_session_origin_sqlite(&path).unwrap().unwrap();
    assert_eq!(index, 1);
    assert_eq!(prompt, "第一轮：帮我看下 src/main.rs");
    assert_eq!(reply.as_deref(), Some("main 只有一个空函数。"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_connection_is_reused_within_a_thread() {
    let dir = std::env::temp_dir().join(format!(
        "write_conn_reuse_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");

    {
        let conn = open_history_db(&path).unwrap();
        super::migrations::init_history_schema(&conn).unwrap();
        // TEMP tables are connection-local: observing this row after the first
        // guard is dropped proves the same SQLite connection came back.
        conn.execute_batch(
            "CREATE TEMP TABLE reuse_probe(value INTEGER);
             INSERT INTO reuse_probe VALUES (42);",
        )
        .unwrap();
    }

    {
        let conn = open_history_db(&path).unwrap();
        let value: i64 = conn
            .query_row("SELECT value FROM reuse_probe", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, 42);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deleted_history_is_not_reused_for_a_new_database_at_the_same_path() {
    let dir = std::env::temp_dir().join(format!(
        "write_conn_delete_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let wal_path = dir.join("history.db-wal");

    {
        let conn = open_history_db(&path).unwrap();
        super::migrations::init_history_schema(&conn).unwrap();
        conn.execute_batch(
            "CREATE TEMP TABLE reuse_probe(value INTEGER);
             INSERT INTO reuse_probe VALUES (42);",
        )
        .unwrap();
    }

    // Session deletion removes the database and its sidecars while the pooled
    // handle still references the unlinked inode.
    crate::ai::history::delete_history_artifacts(&path).unwrap();
    assert!(!path.exists());

    {
        let conn = open_history_db(&path).unwrap();
        super::migrations::init_history_schema(&conn).unwrap();
        // TEMP state is connection-local, so its absence proves the deleted
        // database's connection was not handed back.
        let probes: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'reuse_probe'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(probes, 0);
        let messages: i64 = conn
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .unwrap();
        assert_eq!(messages, 0);
    }

    // Closing the deleted database's connection must not unlink the new file's
    // sidecars; the guard is pooled here, so the WAL has to survive.
    assert!(wal_path.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn init_history_schema_skips_ddl_after_version_marker() {
    let dir = std::env::temp_dir().join(format!(
        "schema_version_fast_path_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.db");
    let conn = Connection::open(&path).unwrap();

    super::migrations::init_history_schema(&conn).unwrap();
    assert_eq!(
        conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i32>(0))
            .unwrap(),
        super::migrations::HISTORY_SCHEMA_VERSION
    );

    // Dropping a table created by init and observing that the next call does
    // not recreate it proves the versioned fast path skipped the CREATE batch.
    // The marker is only written after a complete init, so this tradeoff is
    // what keeps the per-operation path cheap.
    conn.execute_batch("DROP TABLE skill_activation_events;")
        .unwrap();
    super::migrations::init_history_schema(&conn).unwrap();
    let table_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'skill_activation_events'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(table_count, 0, "second init replayed schema DDL");

    drop(conn);
    let _ = std::fs::remove_dir_all(&dir);
}
