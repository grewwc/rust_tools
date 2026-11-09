use std::{fs, io, path::PathBuf, time::Duration};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use super::super::{
    ContextHistory, SnapshotStamp, append_history_sqlite, backup_sqlite,
    clear_session_history_sqlite, close_cached_write_conn,
    coalesce_repeated_wait_wake_notes_sqlite, context, migrations, read_all_messages_sqlite,
    read_context_history_sqlite, remap_session_asset_paths_sqlite,
    remove_history_revision_cache_entry, remove_session_state_lock_entry,
    replace_all_messages_sqlite, restore_sqlite_after_rollback, revision, truncate_messages_sqlite,
    truncate_messages_to_user_turns_sqlite, write_context_snapshot_sqlite_if_stamp,
};
use super::{Message, ROLE_INTERNAL_NOTE, msg, open_history_db, wake_note_text};

const POLICY: &str = "b2-prefix-state";
const STATE: &str = "{\n  \"version\": 1, \"prefix\": \"certified\"\n}";
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("b2-prefix-state-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        Self {
            path: dir.join("history.db"),
            dir,
        }
    }

    fn append_turn(&self, name: &str) -> Vec<Message> {
        let messages = vec![
            msg("user", name),
            msg("assistant", &format!("{name} answer")),
        ];
        append_history_sqlite(&self.path, messages.clone()).unwrap();
        messages
    }

    fn read(&self) -> ContextHistory {
        read_context_history_sqlite(&self.path, POLICY).unwrap()
    }

    fn write_from(
        &self,
        basis: &ContextHistory,
        messages: &[Message],
        state: Option<&str>,
    ) -> io::Result<Option<SnapshotStamp>> {
        write_context_snapshot_sqlite_if_stamp(
            &self.path,
            messages,
            basis.source_message_id,
            POLICY,
            basis.snapshot_stamp,
            state,
            BUSY_TIMEOUT,
        )
    }

    fn save(&self, messages: &[Message], state: &str) -> SnapshotStamp {
        self.write_from(&self.read(), messages, Some(state))
            .unwrap()
            .unwrap()
    }

    fn state(&self) -> Option<String> {
        Connection::open(&self.path)
            .unwrap()
            .query_row(
                "SELECT compression_state FROM context_snapshot WHERE singleton = 1",
                [],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .unwrap()
            .flatten()
    }

    fn mutation_counters(&self) -> (i64, i64) {
        Connection::open(&self.path)
            .unwrap()
            .query_row(
                "SELECT
                    COALESCE((SELECT CAST(value AS INTEGER) FROM meta
                              WHERE key = 'history_revision'), 0),
                    (SELECT COUNT(*) FROM context_snapshot_history)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap()
    }

    fn assert_invalidated(&self, old: SnapshotStamp) {
        assert_eq!(self.state(), None);
        let current = self.read();
        assert_ne!(current.snapshot_stamp, old);
        assert!(
            current
                .snapshot_meta
                .as_ref()
                .is_none_or(|meta| meta.compression_state.is_none())
        );
        assert_eq!(
            write_context_snapshot_sqlite_if_stamp(
                &self.path,
                &[msg(ROLE_INTERNAL_NOTE, "[summary] stale writer")],
                current.source_message_id,
                POLICY,
                old,
                Some(STATE),
                BUSY_TIMEOUT,
            )
            .unwrap(),
            None
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        close_cached_write_conn(&self.path);
        remove_history_revision_cache_entry(&self.path);
        remove_session_state_lock_entry(&self.path);
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn b2_prefix_state_cas_rejects_stale_writer() {
    let fixture = Fixture::new();
    let first_turn = fixture.append_turn("first");
    assert_eq!(fixture.read().snapshot_stamp, SnapshotStamp::default());
    let first = fixture.save(&[msg(ROLE_INTERNAL_NOTE, "[summary] first")], STATE);
    assert_eq!(first.snapshot_revision, 1);
    let stale = fixture.read();
    let newest_rows = vec![msg(ROLE_INTERNAL_NOTE, "[summary] newest")];
    let newest = fixture
        .write_from(&stale, &newest_rows, Some("newest certificate"))
        .unwrap()
        .unwrap();
    assert_eq!(newest.canonical_generation, first.canonical_generation);
    assert_eq!(newest.snapshot_revision, first.snapshot_revision + 1);

    let counters = fixture.mutation_counters();
    assert_eq!(
        fixture
            .write_from(
                &stale,
                &[msg(ROLE_INTERNAL_NOTE, "[summary] stale")],
                Some("stale certificate"),
            )
            .unwrap(),
        None
    );
    let after_rejection = fixture.read();
    assert_eq!(after_rejection.messages, newest_rows);
    assert_eq!(after_rejection.snapshot_stamp, newest);
    assert_eq!(fixture.state().as_deref(), Some("newest certificate"));
    assert_eq!(fixture.mutation_counters(), counters);

    let basis = fixture.read();
    let tail = vec![msg("user", "second"), msg("assistant", "second answer")];
    let append_path = fixture.path.clone();
    let appended_tail = tail.clone();
    std::thread::spawn(move || append_history_sqlite(&append_path, appended_tail))
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(fixture.read().snapshot_stamp, basis.snapshot_stamp);
    assert!(fixture.mutation_counters().0 > counters.0);
    let merged_prefix = vec![msg(ROLE_INTERNAL_NOTE, "[summary] mergeable prefix")];
    let merged_stamp = fixture
        .write_from(&basis, &merged_prefix, Some("mergeable certificate"))
        .unwrap()
        .unwrap();
    let merged = fixture.read();
    let mut expected = merged_prefix;
    expected.extend(tail.clone());
    assert_eq!(merged.messages, expected);
    assert_eq!(merged.snapshot_stamp, merged_stamp);
    let meta = merged.snapshot_meta.unwrap();
    assert_eq!(meta.source_message_id, basis.source_message_id);
    assert_eq!(meta.message_count, 1);
    assert_eq!(
        meta.compression_state.as_deref(),
        Some("mergeable certificate")
    );
    assert!(meta.guidance_unchanged);
    assert!(merged.source_message_id > meta.source_message_id);
    assert!(!merged.snapshot_is_current);
    let mut expected_canonical = first_turn;
    expected_canonical.extend(tail);
    assert_eq!(
        read_all_messages_sqlite(&fixture.path).unwrap(),
        expected_canonical
    );
}

#[test]
fn b2_prefix_state_rewind_invalidates_retained_and_promoted() {
    let retained = Fixture::new();
    retained.append_turn("first");
    let first_summary = msg(ROLE_INTERNAL_NOTE, "[summary] first");
    let stamp = retained.save(std::slice::from_ref(&first_summary), STATE);
    retained.append_turn("second");
    // The surviving history ends exactly at the retained snapshot watermark.
    truncate_messages_to_user_turns_sqlite(&retained.path, 1).unwrap();
    retained.assert_invalidated(stamp);
    let current = retained.read();
    assert_eq!(current.messages, vec![first_summary.clone()]);
    assert!(current.snapshot_is_current);
    assert_eq!(current.snapshot_meta.unwrap().source_message_id, 2);

    let retained_tail = Fixture::new();
    retained_tail.append_turn("first");
    let stamp = retained_tail.save(std::slice::from_ref(&first_summary), STATE);
    retained_tail.append_turn("second");
    truncate_messages_sqlite(&retained_tail.path, 3).unwrap();
    retained_tail.assert_invalidated(stamp);
    let current = retained_tail.read();
    assert_eq!(
        current.messages,
        vec![first_summary.clone(), msg("user", "second")]
    );
    assert!(current.snapshot_is_current);
    assert_eq!(current.snapshot_meta.unwrap().message_count, 2);

    for keep in [2, 3] {
        let promoted = Fixture::new();
        promoted.append_turn("first");
        promoted.save(std::slice::from_ref(&first_summary), "archived certificate");
        promoted.append_turn("second");
        let stamp = promoted.save(
            &[msg(ROLE_INTERNAL_NOTE, "[summary] first and second")],
            STATE,
        );
        assert_eq!(promoted.mutation_counters().1, 1);
        truncate_messages_sqlite(&promoted.path, keep).unwrap();
        promoted.assert_invalidated(stamp);
        let current = promoted.read();
        let mut expected = vec![first_summary.clone()];
        if keep == 3 {
            expected.push(msg("user", "second"));
        }
        assert_eq!(current.messages, expected);
        assert!(current.snapshot_is_current);
        assert_eq!(
            current.snapshot_meta.unwrap().source_message_id,
            keep as i64
        );
        assert!(current.canonical_generation > stamp.canonical_generation);
    }

    // Exercise absorption directly so its own invalidation is required even
    // without the caller re-stamping the generation first.
    let absorbed = Fixture::new();
    absorbed.append_turn("first");
    let stamp = absorbed.save(std::slice::from_ref(&first_summary), STATE);
    let tail = absorbed.append_turn("second");
    {
        let mut conn = open_history_db(&absorbed.path).unwrap();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        context::extend_context_snapshot_through_latest(&tx).unwrap();
        revision::bump_history_revision(&tx).unwrap();
        tx.commit().unwrap();
    }
    absorbed.assert_invalidated(stamp);
    let current = absorbed.read();
    let mut expected = vec![first_summary];
    expected.extend(tail);
    assert_eq!(current.messages, expected);
    assert_eq!(current.canonical_generation, stamp.canonical_generation);
    assert!(current.snapshot_revision > stamp.snapshot_revision);
    assert!(current.snapshot_is_current);
}

#[test]
fn b2_prefix_state_clear_trim_restore_invalidate() {
    for operation in [
        "clear",
        "replace",
        "interior_delete",
        "trim_rows",
        "trim_turns",
        "asset_remap",
        "asset_remap_unchanged",
        "rollback_restore",
    ] {
        let fixture = Fixture::new();
        let wake = msg(
            ROLE_INTERNAL_NOTE,
            &wake_note_text(6, &["task_a"], "old checkpoint"),
        );
        let source_assets = fixture.dir.join("source.assets");
        let target_assets = fixture.dir.join("target.assets");
        let mut canonical = vec![
            msg("user", "first"),
            msg("assistant", "first answer"),
            wake,
            msg("user", "second"),
            msg("assistant", "second answer"),
        ];
        if operation == "asset_remap" {
            canonical.insert(
                2,
                msg(
                    ROLE_INTERNAL_NOTE,
                    &format!(
                        "[context_checkpoint path={}]",
                        source_assets
                            .join("context-checkpoints/source.txt")
                            .display()
                    ),
                ),
            );
        }
        append_history_sqlite(&fixture.path, canonical.clone()).unwrap();
        let summary = vec![msg(ROLE_INTERNAL_NOTE, "[summary] original projection")];
        let stamp = fixture.save(&summary, STATE);
        assert_eq!(fixture.state().as_deref(), Some(STATE), "{operation}");

        match operation {
            "clear" => {
                clear_session_history_sqlite(&fixture.path).unwrap();
                assert!(read_all_messages_sqlite(&fixture.path).unwrap().is_empty());
            }
            "replace" => {
                let replacement = vec![msg("user", "replacement")];
                replace_all_messages_sqlite(&fixture.path, &replacement).unwrap();
                assert_eq!(
                    read_all_messages_sqlite(&fixture.path).unwrap(),
                    replacement
                );
            }
            "interior_delete" => {
                let latest = msg(
                    ROLE_INTERNAL_NOTE,
                    &wake_note_text(6, &["task_a"], "new checkpoint"),
                );
                assert!(coalesce_repeated_wait_wake_notes_sqlite(&fixture.path, &latest).unwrap());
                assert_eq!(
                    read_all_messages_sqlite(&fixture.path).unwrap().len(),
                    canonical.len() - 1
                );
            }
            "trim_rows" => {
                truncate_messages_sqlite(&fixture.path, 1).unwrap();
                assert_eq!(read_all_messages_sqlite(&fixture.path).unwrap().len(), 1);
            }
            "trim_turns" => {
                truncate_messages_to_user_turns_sqlite(&fixture.path, 1).unwrap();
                assert_eq!(read_all_messages_sqlite(&fixture.path).unwrap().len(), 3);
            }
            "asset_remap" | "asset_remap_unchanged" => {
                let changed = remap_session_asset_paths_sqlite(
                    &fixture.path,
                    Some(&source_assets),
                    &target_assets,
                )
                .unwrap();
                assert_eq!(changed, usize::from(operation == "asset_remap"));
                if changed > 0 {
                    let remapped = read_all_messages_sqlite(&fixture.path).unwrap();
                    assert!(
                        remapped[2].content.as_str().unwrap().contains(
                            target_assets
                                .join("context-checkpoints/source.txt")
                                .to_str()
                                .unwrap()
                        )
                    );
                }
            }
            "rollback_restore" => {
                let checkpoint = fixture.dir.join("checkpoint.sqlite");
                backup_sqlite(&fixture.path, &checkpoint).unwrap();
                fixture.append_turn("after checkpoint");
                let live_stamp = fixture.save(
                    &[msg(ROLE_INTERNAL_NOTE, "[summary] after checkpoint")],
                    "live certificate",
                );
                restore_sqlite_after_rollback(&checkpoint, &fixture.path).unwrap();
                fixture.assert_invalidated(live_stamp);
                let restored = fixture.read();
                assert_eq!(restored.messages, summary);
                assert!(restored.snapshot_is_current);
                assert!(restored.snapshot_meta.is_some());
                assert_eq!(read_all_messages_sqlite(&fixture.path).unwrap(), canonical);
            }
            _ => unreachable!(),
        }
        fixture.assert_invalidated(stamp);
    }
}

#[test]
fn b2_prefix_state_policy_mismatch_keeps_physical_cas_token() {
    let fixture = Fixture::new();
    let canonical = fixture.append_turn("first");
    let stamp = fixture.save(&[msg(ROLE_INTERNAL_NOTE, "[summary] old policy")], STATE);
    let ignored = read_context_history_sqlite(&fixture.path, "new policy").unwrap();
    assert!(ignored.snapshot_meta.is_none());
    assert!(!ignored.snapshot_is_current);
    assert_eq!(ignored.messages, canonical);
    assert_eq!(ignored.snapshot_stamp, stamp);
    assert_eq!(ignored.snapshot_revision, stamp.snapshot_revision);
    let new_rows = vec![msg(ROLE_INTERNAL_NOTE, "[summary] new policy")];
    let new_stamp = write_context_snapshot_sqlite_if_stamp(
        &fixture.path,
        &new_rows,
        ignored.source_message_id,
        "new policy",
        ignored.snapshot_stamp,
        Some("new policy certificate"),
        BUSY_TIMEOUT,
    )
    .unwrap()
    .unwrap();
    let new_policy = read_context_history_sqlite(&fixture.path, "new policy").unwrap();
    assert_eq!(new_policy.snapshot_stamp, new_stamp);
    assert_eq!(new_policy.messages, new_rows);
    let meta = new_policy.snapshot_meta.unwrap();
    assert_eq!(meta.projection_fingerprint, "new policy");
    assert_eq!(
        meta.compression_state.as_deref(),
        Some("new policy certificate")
    );

    // A rejected source-generation stamp must also preserve the physical epoch;
    // the write fence uses the current canonical generation from metadata.
    {
        let conn = Connection::open(&fixture.path).unwrap();
        conn.execute(
            "UPDATE context_snapshot SET source_generation = source_generation - 1",
            [],
        )
        .unwrap();
    }
    let ignored_generation = read_context_history_sqlite(&fixture.path, "new policy").unwrap();
    assert!(ignored_generation.snapshot_meta.is_none());
    assert_eq!(ignored_generation.snapshot_stamp, new_stamp);
    assert_eq!(ignored_generation.messages, canonical);
    assert!(
        write_context_snapshot_sqlite_if_stamp(
            &fixture.path,
            &new_rows,
            ignored_generation.source_message_id,
            "new policy",
            ignored_generation.snapshot_stamp,
            Some("rebuilt certificate"),
            BUSY_TIMEOUT,
        )
        .unwrap()
        .is_some()
    );

    let legacy = Fixture::new();
    {
        let conn = Connection::open(&legacy.path).unwrap();
        conn.execute_batch(
            "CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL, content TEXT NOT NULL,
                tool_calls TEXT, tool_call_id TEXT, reasoning_content TEXT,
                source_model TEXT, created_at INTEGER NOT NULL DEFAULT (unixepoch())
             );
             CREATE TABLE context_messages (
                position INTEGER PRIMARY KEY,
                role TEXT NOT NULL, content TEXT NOT NULL,
                tool_calls TEXT, tool_call_id TEXT, reasoning_content TEXT
             );
             CREATE TABLE context_snapshot (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                source_message_id INTEGER NOT NULL,
                source_generation INTEGER NOT NULL,
                projection_fingerprint TEXT NOT NULL,
                snapshot_revision INTEGER NOT NULL DEFAULT 1
             );
             PRAGMA user_version = 4;",
        )
        .unwrap();
        for message in &canonical {
            conn.execute(
                "INSERT INTO messages (role, content) VALUES (?1, ?2)",
                params![
                    message.role,
                    serde_json::to_string(&message.content).unwrap()
                ],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO context_messages (position, role, content) VALUES (0, ?1, ?2)",
            params![
                ROLE_INTERNAL_NOTE,
                serde_json::to_string("[summary] v4 snapshot").unwrap()
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO context_snapshot VALUES (1, 2, 0, ?1, 7)",
            [POLICY],
        )
        .unwrap();
    }
    let migrated = legacy.read();
    assert_eq!(
        migrated.messages,
        vec![msg(ROLE_INTERNAL_NOTE, "[summary] v4 snapshot")]
    );
    assert_eq!(migrated.source_message_id, 2);
    assert_eq!(migrated.snapshot_stamp.snapshot_revision, 7);
    assert!(migrated.snapshot_is_current);
    let meta = migrated.snapshot_meta.unwrap();
    assert_eq!(meta.source_message_id, 2);
    assert_eq!(meta.message_count, 1);
    assert_eq!(meta.compression_state, None);
    let conn = Connection::open(&legacy.path).unwrap();
    assert_eq!(
        conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i32>(0))
            .unwrap(),
        migrations::HISTORY_SCHEMA_VERSION
    );
    assert_eq!(migrations::HISTORY_SCHEMA_VERSION, 5);
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('context_snapshot')
             WHERE name = 'snapshot_epoch'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap(),
        0
    );
}

#[test]
fn b2_prefix_state_guidance_filter_clears_certificate() {
    let fixture = Fixture::new();
    let guidance = msg(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nKeep full evidence.",
    );
    let original = vec![
        msg("user", "old"),
        guidance.clone(),
        msg("user", "current"),
        guidance.clone(),
        guidance.clone(),
        msg("assistant", "answer"),
    ];
    append_history_sqlite(&fixture.path, original.clone()).unwrap();
    let stamp = fixture.save(&original, STATE);
    assert_eq!(fixture.state(), None);
    let cleaned = vec![
        msg("user", "old"),
        msg("user", "current"),
        guidance,
        msg("assistant", "answer"),
    ];
    let read = fixture.read();
    assert_eq!(read.snapshot_stamp, stamp);
    assert_eq!(read.messages, cleaned);
    assert!(read.snapshot_is_current);
    let meta = read.snapshot_meta.unwrap();
    assert_eq!(meta.message_count, cleaned.len());
    assert!(meta.guidance_unchanged);
    assert_eq!(meta.compression_state, None);
    assert_eq!(read_all_messages_sqlite(&fixture.path).unwrap(), original);

    fixture.save(&cleaned, STATE);
    assert_eq!(fixture.state().as_deref(), Some(STATE));
    append_history_sqlite(&fixture.path, vec![msg("user", "next")]).unwrap();
    let expired = fixture.read();
    let meta = expired.snapshot_meta.unwrap();
    assert!(!meta.guidance_unchanged);
    assert_eq!(meta.message_count, cleaned.len());
    assert!(!expired.snapshot_is_current);
}

#[test]
fn b2_prefix_state_failed_write_is_atomic() {
    let fixture = Fixture::new();
    fixture.append_turn("first");
    let rows = vec![msg(ROLE_INTERNAL_NOTE, "[summary] committed")];
    fixture.save(&rows, STATE);
    let basis = fixture.read();
    let counters = fixture.mutation_counters();
    {
        let conn = Connection::open(&fixture.path).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER b2_reject_snapshot BEFORE INSERT ON context_messages
             WHEN NEW.position = 1
             BEGIN SELECT RAISE(ABORT, 'reject test snapshot'); END;",
        )
        .unwrap();
    }
    let replacement = vec![
        msg(ROLE_INTERNAL_NOTE, "[summary] replacement"),
        msg("assistant", "extra row"),
    ];
    assert!(
        fixture
            .write_from(&basis, &replacement, Some("uncommitted certificate"))
            .is_err()
    );
    assert_eq!(fixture.read().messages, rows);
    assert_eq!(fixture.read().snapshot_stamp, basis.snapshot_stamp);
    assert_eq!(fixture.state().as_deref(), Some(STATE));
    assert_eq!(fixture.mutation_counters(), counters);
    Connection::open(&fixture.path)
        .unwrap()
        .execute_batch("DROP TRIGGER b2_reject_snapshot;")
        .unwrap();
    assert!(
        fixture
            .write_from(&basis, &replacement, Some("committed certificate"))
            .unwrap()
            .is_some()
    );
    assert_eq!(fixture.read().messages, replacement);
    assert_eq!(fixture.state().as_deref(), Some("committed certificate"));
}

#[test]
fn b2_prefix_state_revision_overflow_preserves_snapshot() {
    let fixture = Fixture::new();
    let first_turn = fixture.append_turn("first");
    let rows = vec![msg(ROLE_INTERNAL_NOTE, "[summary] committed")];
    fixture.save(&rows, STATE);
    Connection::open(&fixture.path)
        .unwrap()
        .execute(
            "UPDATE context_snapshot SET snapshot_revision = ?1",
            [i64::MAX],
        )
        .unwrap();
    let basis = fixture.read();
    let counters = fixture.mutation_counters();
    assert_eq!(basis.snapshot_revision, i64::MAX);
    let error = fixture
        .write_from(&basis, &[msg("assistant", "overflow")], Some("overflow"))
        .unwrap_err();
    assert!(error.to_string().contains("snapshot revision overflow"));
    assert_eq!(fixture.read().messages, rows);
    assert_eq!(fixture.read().snapshot_stamp, basis.snapshot_stamp);
    assert_eq!(fixture.state().as_deref(), Some(STATE));
    assert_eq!(fixture.mutation_counters(), counters);

    let tail = fixture.append_turn("second");
    let counters = fixture.mutation_counters();
    // Retention re-stamps the generation before absorption; overflow must
    // roll back the entire truncation, including certificate invalidation.
    let error = truncate_messages_sqlite(&fixture.path, 3).unwrap_err();
    assert!(error.to_string().contains("snapshot revision overflow"));
    assert_eq!(fixture.read().snapshot_stamp, basis.snapshot_stamp);
    assert_eq!(fixture.state().as_deref(), Some(STATE));
    assert_eq!(fixture.mutation_counters(), counters);
    let mut canonical = first_turn;
    canonical.extend(tail);
    assert_eq!(read_all_messages_sqlite(&fixture.path).unwrap(), canonical);
}
