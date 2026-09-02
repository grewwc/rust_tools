use super::*;
use serde_json::Value;
use std::io;
use rusqlite::Connection;

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
        let error =
            rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(result_code), None);
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
fn stale_patch_targets_survive_history_replacement_and_clear_with_session() {
    let dir = std::env::temp_dir().join(format!(
        "stale_patch_meta_test_{}_{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("history.sqlite");
    append_history_sqlite(&path, vec![msg("user", "before compression")]).unwrap();

    let targets =
        FxHashSet::from_iter([PathBuf::from("/tmp/a.rs"), PathBuf::from("/tmp/b.rs")]);
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
    append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1"), outcome("call-2")])
        .unwrap();

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
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-1")),
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-2")),
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(7, &["task_x"], "checkpoint-3")),
        ],
    )
    .unwrap();

    let latest =
        msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-4"));
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

    let latest =
        msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-new"));
    assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());

    let notes: Vec<_> = read_all_messages_sqlite(&path)
        .unwrap()
        .into_iter()
        .filter(|m| m.role == ROLE_INTERNAL_NOTE)
        .collect();
    // The old waiting note outside the window is retained and not wrongly deleted.
    assert_eq!(notes.len(), 1);
    assert!(notes[0].content.as_str().unwrap().contains("checkpoint-old"));

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
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-1")),
        ],
    )
    .unwrap();

    // Same pid but a different task set: the identity differs, so no dedup.
    let other = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_z"], "checkpoint-x"));
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
