use super::*;
use rusqlite::{ffi, types::Value as SqlValue};
use std::{
    cell::RefCell,
    ffi::{CStr, c_int, c_void},
    path::PathBuf,
};

const GUIDANCE: &str = "[tool_first_use_guidance name=read_file]\nRead serially.";
const REVISED_GUIDANCE: &str = "[tool_first_use_guidance name=read_file]\nKeep all evidence.";
const OLD_CONTENT: &str = r#""old\u0020summary""#;
const CONCURRENT_CONTENT: &str = r#""concurrent projection""#;
const ACTIVITY: &str = "8000000000000000000";
const ARCHIVE_JSON: &[u8] = br#"[{"position":0,"role":"internal_note","content":"\"old\\u0020summary\"","tool_calls":null,"tool_call_id":null,"reasoning_content":null},{"position":7,"role":"assistant","content":"{\"z\":1, \"a\":[true,null]}","tool_calls":" [ ] ","tool_call_id":"","reasoning_content":"old\nreasoning"}]"#;
const CONCURRENT_ARCHIVE_JSON: &[u8] = br#"[{"position":0,"role":"internal_note","content":"\"concurrent projection\"","tool_calls":null,"tool_call_id":null,"reasoning_content":null},{"position":7,"role":"assistant","content":"{\"z\":1, \"a\":[true,null]}","tool_calls":" [ ] ","tool_call_id":"","reasoning_content":"old\nreasoning"}]"#;

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Cpu(&'static str),
    BeforeBegin,
    Begin,
    Commit,
    Write(c_int, String, i64),
}

#[derive(Default)]
struct Trace {
    events: Vec<Event>,
    force_legacy: bool,
    before_begin: Option<Box<dyn FnOnce()>>,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

struct TraceGuard;

impl TraceGuard {
    fn new(force_legacy: bool, before_begin: Option<Box<dyn FnOnce()>>) -> Self {
        TRACE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(Trace {
                force_legacy,
                before_begin,
                ..Trace::default()
            });
        });
        Self
    }

    fn finish(self) -> Vec<Event> {
        TRACE.with(|slot| slot.borrow_mut().take().unwrap().events)
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        TRACE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

fn record(event: Event) {
    TRACE.with(|slot| {
        if let Some(trace) = slot.borrow_mut().as_mut() {
            trace.events.push(event);
        }
    });
}

pub(super) fn force_legacy() -> bool {
    TRACE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|trace| trace.force_legacy)
    })
}

pub(super) fn record_cpu_work(work: &'static str) {
    record(Event::Cpu(work));
}

pub(super) fn before_begin<'a>(
    conn: &Connection,
    prepared: Option<PreparedContextSnapshot<'a>>,
) -> Option<PreparedContextSnapshot<'a>> {
    assert!(conn.is_autocommit());
    record(Event::BeforeBegin);
    let callback = TRACE.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .and_then(|trace| trace.before_begin.take())
    });
    if let Some(callback) = callback {
        callback();
    }
    prepared
}

pub(super) fn record_begin(conn: &Connection) {
    assert!(!conn.is_autocommit());
    record(Event::Begin);
}

pub(super) fn record_commit(conn: &Connection) {
    assert!(conn.is_autocommit());
    record(Event::Commit);
}

unsafe extern "C" fn fixed_unixepoch(
    context: *mut ffi::sqlite3_context,
    _argc: c_int,
    _argv: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: SQLite supplies a live function context for this synchronous call.
    unsafe { ffi::sqlite3_result_int64(context, 42) };
}

unsafe extern "C" fn observe_write(
    _data: *mut c_void,
    operation: c_int,
    _database: *const std::ffi::c_char,
    table: *const std::ffi::c_char,
    rowid: i64,
) {
    // SAFETY: SQLite owns a NUL-terminated table name for the callback duration.
    let table = unsafe { CStr::from_ptr(table) }
        .to_string_lossy()
        .into_owned();
    record(Event::Write(operation, table, rowid));
}

fn freeze_sql_time(conn: &Connection) {
    // SAFETY: The connection is live and exclusively borrowed. The callback
    // captures no Rust state and remains valid until SQLite closes the handle.
    let status = unsafe {
        ffi::sqlite3_create_function_v2(
            conn.handle(),
            c"unixepoch".as_ptr(),
            -1,
            ffi::SQLITE_UTF8 | ffi::SQLITE_DETERMINISTIC,
            std::ptr::null_mut(),
            Some(fixed_unixepoch),
            None,
            None,
            None,
        )
    };
    assert_eq!(status, ffi::SQLITE_OK);
}

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("context-prepare-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let conn = open_history_db(&path).unwrap();
        freeze_sql_time(&conn);
        init_history_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO meta (key, value) VALUES
                 ('history_generation', '2'),
                 ('history_revision', '7'),
                 ('last_activity_unix_ms', '8000000000000000000');
             INSERT INTO messages (role, content) VALUES
                 ('user', '\"canonical first\"'),
                 ('assistant', '\"canonical reply\"'),
                 ('user', '\"canonical second\"'),
                 ('assistant', '\"canonical second reply\"'),
                 ('user', '\"canonical tail\"');
             INSERT INTO context_snapshot VALUES (1, 4, 2, 'old-policy');
             INSERT INTO context_snapshot_history
                 (source_message_id, projection_fingerprint, payload) VALUES
                 (1, 'older-policy', X'010203'),
                 (99, 'legacy-null', NULL);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO context_messages (position, role, content) VALUES (0, ?1, ?2)",
            params![ROLE_INTERNAL_NOTE, OLD_CONTENT],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO context_messages VALUES (7, 'assistant', ?1, ' [ ] ', '', ?2)",
            params![r#"{"z":1, "a":[true,null]}"#, "old\nreasoning"],
        )
        .unwrap();
        // SAFETY: The callback captures no pointers and SQLite invokes it only
        // synchronously on this exclusively checked-out connection's thread.
        unsafe {
            ffi::sqlite3_update_hook(conn.handle(), Some(observe_write), std::ptr::null_mut());
        }
        drop(conn);
        Self { dir, path }
    }

    fn execute(&self, sql: &str) {
        open_history_db(&self.path)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
    }

    fn rows(&self, sql: &str) -> Vec<Vec<SqlValue>> {
        let conn = Connection::open(&self.path).unwrap();
        let mut stmt = conn.prepare(sql).unwrap();
        let columns = stmt.column_count();
        stmt.query_map([], |row| {
            (0..columns)
                .map(|column| row.get(column))
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap()
    }

    fn all_rows(&self) -> Vec<(&'static str, Vec<Vec<SqlValue>>)> {
        [
            "messages",
            "context_messages",
            "context_snapshot",
            "context_snapshot_history",
            "meta",
            "sqlite_sequence",
        ]
        .map(|table| {
            (
                table,
                self.rows(&format!("SELECT rowid, * FROM {table} ORDER BY rowid")),
            )
        })
        .to_vec()
    }

    fn context_rows(&self) -> Vec<Vec<SqlValue>> {
        self.rows("SELECT * FROM context_messages ORDER BY position")
    }

    fn archive_json(&self) -> Vec<u8> {
        let rows =
            self.rows("SELECT payload FROM context_snapshot_history WHERE source_message_id = 4");
        let SqlValue::Blob(payload) = &rows[0][0] else {
            panic!("archive payload must remain a BLOB");
        };
        let mut json = Vec::new();
        GzDecoder::new(payload.as_slice())
            .read_to_end(&mut json)
            .unwrap();
        json
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        super::super::connection::close_cached_write_conn(&self.path);
        let _ = std::fs::remove_dir_all(&self.dir);
        super::super::lock::remove_session_state_lock_entry(&self.path);
    }
}

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.into(),
        content: serde_json::Value::String(text.into()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn input() -> Vec<Message> {
    let guidance = message(ROLE_INTERNAL_NOTE, GUIDANCE);
    let assistant = Message {
        content: serde_json::from_str(r#"{"z":1, "a":["quote\"",true,null]}"#).unwrap(),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "new-call".into(),
            tool_type: "function".into(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".into(),
                arguments: "{\"path\":\"a\\nb\"}".into(),
            },
        }]),
        reasoning_content: Some("reason\nexact".into()),
        ..message("assistant", "")
    };
    let tool = Message {
        content: serde_json::from_str(r#"[{"type":"text","text":"tool\nresult"}]"#).unwrap(),
        tool_call_id: Some("new-call".into()),
        reasoning_content: Some(String::new()),
        ..message("tool", "")
    };
    vec![
        message("user", "old request"),
        guidance.clone(),
        message("user", "new\nrequest \"exact\""),
        guidance.clone(),
        guidance,
        message(ROLE_INTERNAL_NOTE, REVISED_GUIDANCE),
        assistant,
        tool,
    ]
}

fn text(value: &str) -> SqlValue {
    SqlValue::Text(value.into())
}

fn expected_context_rows() -> Vec<Vec<SqlValue>> {
    use SqlValue::{Integer as I, Null as N};
    vec![
        vec![I(0), text("user"), text(r#""old request""#), N, N, N],
        vec![
            I(1),
            text("user"),
            text(r#""new\nrequest \"exact\"""#),
            N,
            N,
            N,
        ],
        vec![
            I(2),
            text(ROLE_INTERNAL_NOTE),
            text(r#""[tool_first_use_guidance name=read_file]\nRead serially.""#),
            N,
            N,
            N,
        ],
        vec![
            I(3),
            text(ROLE_INTERNAL_NOTE),
            text(r#""[tool_first_use_guidance name=read_file]\nKeep all evidence.""#),
            N,
            N,
            N,
        ],
        vec![
            I(4),
            text("assistant"),
            text(r#"{"z":1,"a":["quote\"",true,null]}"#),
            text(
                r#"[{"id":"new-call","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\\nb\"}"}}]"#,
            ),
            N,
            text("reason\nexact"),
        ],
        vec![
            I(5),
            text("tool"),
            text(r#"[{"type":"text","text":"tool\nresult"}]"#),
            N,
            text("new-call"),
            text(""),
        ],
    ]
}

fn write(
    fixture: &Fixture,
    force_legacy: bool,
    before_begin: Option<Box<dyn FnOnce()>>,
) -> (io::Result<bool>, Vec<Event>) {
    let guard = TraceGuard::new(force_legacy, before_begin);
    let result = write_context_snapshot_sqlite(&fixture.path, &input(), 5, 2, "new-policy");
    (result, guard.finish())
}

fn cpu_events(events: &[Event]) -> Vec<&'static str> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Cpu(work) => Some(*work),
            _ => None,
        })
        .collect()
}

fn writes(events: &[Event]) -> Vec<&Event> {
    events
        .iter()
        .filter(|event| matches!(event, Event::Write(..)))
        .collect()
}

#[test]
fn context_snapshot_prepared_write_matches_legacy_bytes_order_and_cpu_boundary() {
    let prepared = Fixture::new();
    let legacy = Fixture::new();
    let canonical = prepared.rows("SELECT * FROM messages ORDER BY id");
    let (result, events) = write(&prepared, false, None);
    assert!(result.unwrap());
    let (legacy_result, legacy_events) = write(&legacy, true, None);
    assert!(legacy_result.unwrap());

    assert_eq!(prepared.all_rows(), legacy.all_rows());
    assert_eq!(prepared.context_rows(), expected_context_rows());
    assert_eq!(legacy.context_rows(), expected_context_rows());
    assert_eq!(
        prepared.rows("SELECT * FROM messages ORDER BY id"),
        canonical
    );
    assert_eq!(prepared.archive_json(), ARCHIVE_JSON);
    assert_eq!(legacy.archive_json(), ARCHIVE_JSON);
    assert_eq!(
        prepared.rows("SELECT * FROM context_snapshot"),
        vec![vec![
            SqlValue::Integer(1),
            SqlValue::Integer(5),
            SqlValue::Integer(2),
            text("new-policy")
        ]]
    );
    assert_eq!(
        prepared.rows("SELECT key, value, created_at FROM meta ORDER BY key"),
        vec![
            vec![text("history_generation"), text("2"), SqlValue::Integer(42)],
            vec![text("history_revision"), text("8"), SqlValue::Integer(42)],
            vec![
                text("last_activity_unix_ms"),
                text("8000000000000000001"),
                SqlValue::Integer(42)
            ],
        ]
    );
    assert_eq!(
        prepared.rows("SELECT snapshot_id, source_message_id, projection_fingerprint, created_at FROM context_snapshot_history ORDER BY snapshot_id"),
        vec![
            vec![SqlValue::Integer(1), SqlValue::Integer(1), text("older-policy"), SqlValue::Integer(42)],
            vec![SqlValue::Integer(3), SqlValue::Integer(4), text("old-policy"), SqlValue::Integer(42)],
        ]
    );
    assert_eq!(writes(&events), writes(&legacy_events));
    let begin = events
        .iter()
        .position(|event| *event == Event::Begin)
        .unwrap();
    let commit = events
        .iter()
        .position(|event| *event == Event::Commit)
        .unwrap();
    assert_eq!(
        cpu_events(&events[..begin]),
        [
            "archive gzip",
            "message JSON",
            "message JSON",
            "message JSON",
            "message JSON",
            "message JSON",
            "message JSON"
        ]
    );
    assert!(cpu_events(&events[begin..=commit]).is_empty());
    assert!(writes(&events[..begin]).is_empty());
    let legacy_begin = legacy_events
        .iter()
        .position(|event| *event == Event::Begin)
        .unwrap();
    assert!(cpu_events(&legacy_events[..legacy_begin]).is_empty());
    assert_eq!(
        cpu_events(&legacy_events[legacy_begin..]),
        cpu_events(&events[..begin])
    );
}

#[test]
fn context_snapshot_identity_mismatch_falls_back_without_losing_writes() {
    for bump_revision in [true, false] {
        let prepared = Fixture::new();
        let legacy = Fixture::new();
        let before_canonical = prepared.rows("SELECT * FROM messages ORDER BY id");
        let make_callback = |path: PathBuf| -> Box<dyn FnOnce()> {
            Box::new(move || {
                let mut other = Connection::open(path).unwrap();
                freeze_sql_time(&other);
                let tx = other
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .unwrap();
                tx.execute(
                    "UPDATE context_messages SET content = ?1 WHERE position = 0",
                    [CONCURRENT_CONTENT],
                )
                .unwrap();
                tx.execute(
                    "INSERT INTO messages (role, content) VALUES ('user', '\"concurrent tail\"')",
                    [],
                )
                .unwrap();
                if bump_revision {
                    tx.execute(
                        "UPDATE meta SET value = '8' WHERE key = 'history_revision'",
                        [],
                    )
                    .unwrap();
                }
                tx.commit().unwrap();
            })
        };
        let (result, events) = write(&prepared, false, Some(make_callback(prepared.path.clone())));
        assert!(result.unwrap());
        let (legacy_result, legacy_events) =
            write(&legacy, true, Some(make_callback(legacy.path.clone())));
        assert!(legacy_result.unwrap());
        assert_eq!(prepared.all_rows(), legacy.all_rows());
        assert_eq!(prepared.context_rows(), expected_context_rows());
        assert_eq!(prepared.archive_json(), CONCURRENT_ARCHIVE_JSON);
        let mut canonical = before_canonical;
        canonical.push(vec![
            SqlValue::Integer(6),
            text("user"),
            text(r#""concurrent tail""#),
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Integer(42),
        ]);
        assert_eq!(
            prepared.rows("SELECT * FROM messages ORDER BY id"),
            canonical
        );
        assert_eq!(writes(&events), writes(&legacy_events));
        let begin = events
            .iter()
            .position(|event| *event == Event::Begin)
            .unwrap();
        assert_eq!(cpu_events(&events[..begin]).first(), Some(&"archive gzip"));
        assert_eq!(
            cpu_events(&events[begin..]),
            [
                "archive gzip",
                "message JSON",
                "message JSON",
                "message JSON",
                "message JSON",
                "message JSON",
                "message JSON"
            ]
        );
        let conn = Connection::open(&prepared.path).unwrap();
        let mut conn = conn;
        let projected = read_context_history_on_conn(&mut conn, "new-policy").unwrap();
        let mut expected = input();
        let mut guidance = ToolGuidanceFilter::new(&expected);
        expected.retain(|message| guidance.keep(message));
        expected.push(message("user", "concurrent tail"));
        // The new real user expires guidance from the snapshot, as before.
        let mut guidance = ToolGuidanceFilter::new(&expected);
        expected.retain(|message| guidance.keep(message));
        assert_eq!(projected.messages, expected);
    }
}

#[test]
fn context_snapshot_prepared_failures_roll_back_all_rows() {
    for trigger in [
        "CREATE TRIGGER fail_row BEFORE INSERT ON context_messages
         WHEN NEW.position = 1 BEGIN SELECT RAISE(ABORT, 'forced row failure'); END;",
        "CREATE TRIGGER fail_activity BEFORE INSERT ON meta
         WHEN NEW.key = 'last_activity_unix_ms'
         BEGIN SELECT RAISE(ABORT, 'forced activity failure'); END;",
    ] {
        let mut errors = Vec::new();
        for force_legacy in [false, true] {
            let fixture = Fixture::new();
            fixture.execute(trigger);
            let before = fixture.all_rows();
            let (result, events) = write(&fixture, force_legacy, None);
            let error = result.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert!(error.to_string().contains("forced"));
            errors.push(
                error
                    .to_string()
                    .replace(fixture.path.to_str().unwrap(), "<history>"),
            );
            assert_eq!(fixture.all_rows(), before);
            assert!(!events.contains(&Event::Commit));
            assert!(open_history_db(&fixture.path).unwrap().is_autocommit());
            if !force_legacy {
                let begin = events
                    .iter()
                    .position(|event| *event == Event::Begin)
                    .unwrap();
                assert_eq!(cpu_events(&events[..begin]).len(), 7);
                assert!(cpu_events(&events[begin..]).is_empty());
            }
        }
        assert_eq!(errors[0], errors[1]);
    }
}

#[test]
fn context_snapshot_preparation_read_failure_uses_original_rollback_path() {
    for force_legacy in [false, true] {
        let fixture = Fixture::new();
        fixture.execute("UPDATE context_messages SET content = X'01' WHERE position = 0");
        let before = fixture.all_rows();
        let (result, events) = write(&fixture, force_legacy, None);
        let error = result.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(
            error
                .to_string()
                .contains("read context_messages for archive")
        );
        assert_eq!(fixture.all_rows(), before);
        assert!(events.contains(&Event::Begin));
        assert!(!events.contains(&Event::Commit));
    }
}

#[test]
fn context_snapshot_archive_trigger_keeps_transaction_read_semantics() {
    for (kind, table) in [
        ("TRIGGER", "context_snapshot_history"),
        ("TRIGGER", "CONTEXT_SNAPSHOT_HISTORY"),
        ("TEMP TRIGGER", "main.context_snapshot_history"),
    ] {
        let prepared = Fixture::new();
        let legacy = Fixture::new();
        let sql = format!(
            r#"CREATE {kind} archive_changes_projection AFTER DELETE ON {table}
               WHEN OLD.payload IS NULL BEGIN
                   UPDATE context_messages SET content = '"concurrent projection"' WHERE position = 0;
               END;"#
        );
        for fixture in [&prepared, &legacy] {
            fixture.execute(&sql);
        }
        let (result, events) = write(&prepared, false, None);
        assert!(result.unwrap());
        assert!(write(&legacy, true, None).0.unwrap());
        assert_eq!(prepared.all_rows(), legacy.all_rows());
        assert_eq!(prepared.archive_json(), CONCURRENT_ARCHIVE_JSON);
        assert_eq!(prepared.context_rows(), expected_context_rows());
        let begin = events
            .iter()
            .position(|event| *event == Event::Begin)
            .unwrap();
        assert!(cpu_events(&events[..begin]).is_empty());
        assert_eq!(cpu_events(&events[begin..]).first(), Some(&"archive gzip"));
    }
}

#[test]
fn context_snapshot_prepared_archive_branches_match_legacy_rows() {
    for (sql, expected_watermarks) in [
        ("DELETE FROM context_snapshot", vec![1, 99]),
        ("UPDATE context_snapshot SET source_generation = 1", vec![]),
        ("UPDATE context_snapshot SET source_message_id = 1", vec![1]),
        ("INSERT INTO context_snapshot_history (source_message_id, projection_fingerprint, payload)
          VALUES (5, 'newer-policy', X'04')", vec![1, 5]),
        ("INSERT INTO context_snapshot_history (source_message_id, projection_fingerprint, payload)
          VALUES (2, 'ring-policy', X'04'), (3, 'ring-policy', X'04'),
                 (4, 'ring-policy', X'04'), (5, 'ring-policy', X'04'),
                 (6, 'ring-policy', X'04'), (7, 'ring-policy', X'04'),
                 (8, 'ring-policy', X'04'), (9, 'ring-policy', X'04'),
                 (10, 'ring-policy', X'04'), (11, 'ring-policy', X'04')",
          vec![4, 5, 6, 7, 8, 9, 10, 11]),
    ] {
        let prepared = Fixture::new();
        let legacy = Fixture::new();
        prepared.execute(sql);
        legacy.execute(sql);
        assert!(write(&prepared, false, None).0.unwrap());
        assert!(write(&legacy, true, None).0.unwrap());
        assert_eq!(prepared.all_rows(), legacy.all_rows());
        assert_eq!(prepared.context_rows(), expected_context_rows());
        assert_eq!(
            prepared.rows("SELECT source_message_id FROM context_snapshot_history ORDER BY source_message_id"),
            expected_watermarks.into_iter().map(|watermark| vec![SqlValue::Integer(watermark)]).collect::<Vec<_>>()
        );
    }
}

#[test]
fn context_snapshot_generation_change_after_preparation_rejects_stale_write() {
    let fixture = Fixture::new();
    let path = fixture.path.clone();
    let callback = Box::new(move || {
        Connection::open(path)
            .unwrap()
            .execute(
                "UPDATE meta SET value = '3' WHERE key = 'history_generation'",
                [],
            )
            .unwrap();
    });
    let mut before = fixture.all_rows();
    let meta = before
        .iter_mut()
        .find(|(table, _)| *table == "meta")
        .unwrap();
    meta.1[0][2] = text("3");
    let (result, events) = write(&fixture, false, Some(callback));
    assert!(!result.unwrap());
    assert_eq!(fixture.all_rows(), before);
    assert!(writes(&events).is_empty());
    assert!(events.contains(&Event::Begin));
    assert!(!events.contains(&Event::Commit));
    assert_eq!(
        fixture.rows("SELECT value FROM meta WHERE key = 'last_activity_unix_ms'"),
        vec![vec![text(ACTIVITY)]]
    );
}
