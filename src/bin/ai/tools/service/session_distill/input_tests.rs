use super::*;
use std::{fs, io::Write};
use rusqlite::{Connection, params};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("distill-input-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }

    fn store(&self) -> SessionStore {
        SessionStore::new(&self.0.join("history.sqlite"))
    }

    fn session(&self, id: &str) -> (PathBuf, Connection) {
        let store = self.store();
        store.ensure_root_dir().unwrap();
        SessionStore::validate_session_id(id).unwrap();
        let path = store.sessions_root().join(format!("{id}.sqlite"));
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE messages (
                id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                tool_calls TEXT, tool_call_id TEXT, reasoning_content TEXT);
             INSERT INTO messages (id, role, content) VALUES (30, 'user', '\"Use SQLite instead.\"');
             INSERT INTO messages (id, role, content) VALUES (10, 'user', '\"Use Redis.\"');"
        ).unwrap();
        conn.execute(
            "INSERT INTO messages VALUES (20, 'assistant', ?1, ?2, NULL, 'inspect the source')",
            params![r#"[{"type":"text","text":"Inspecting."}]"#,
                r#"[{"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{}"}}]"#],
        ).unwrap();
        (path, conn)
    }

    fn archive(&self, name: &str, id: &str, path: &Path) -> PathBuf {
        let target = self.0.join(name);
        let mut archive = zip::ZipWriter::new(fs::File::create(&target).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        archive.start_file("manifest.json", options).unwrap();
        archive.write_all(serde_json::json!({"session_id": id}).to_string().as_bytes()).unwrap();
        archive.start_file("session.sqlite", options).unwrap();
        archive.write_all(&fs::read(path).unwrap()).unwrap();
        archive.finish().unwrap();
        target
    }

    fn resolve(&self, value: &str) -> Result<DistillInput, String> {
        DistillInput::resolve(value, &self.0, &self.store())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn session_distill_input_zip_and_local_id_preserve_identical_evidence() {
    let fixture = Fixture::new();
    let (path, conn) = fixture.session("source-session");
    drop(conn);
    let before = fs::read(&path).unwrap();
    let zip = fixture.archive("source.zip", "source-session", &path);
    let local = fixture.resolve("source-session").unwrap();
    let (id, messages) = local.read_messages().unwrap();
    assert_eq!(id, "source-session");
    assert_eq!(local.path(), path);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content.as_str(), Some("Use Redis."));
    assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
    assert_eq!(messages[1].reasoning_content.as_deref(), Some("inspect the source"));
    assert_eq!(messages[2].content.as_str(), Some("Use SQLite instead."));
    let segments = super::super::source_segments(&messages).unwrap();
    for input in ["source.zip", "./source.zip", zip.to_str().unwrap()] {
        let archive = fixture.resolve(input).unwrap();
        assert!(matches!(archive, DistillInput::Archive(_)));
        let (archive_id, archived) = archive.read_messages().unwrap();
        assert_eq!(archive_id, id);
        assert_eq!(serde_json::to_value(&archived).unwrap(), serde_json::to_value(&messages).unwrap());
        assert_eq!(serde_json::to_value(super::super::source_segments(&archived).unwrap()).unwrap(),
            serde_json::to_value(&segments).unwrap());
    }
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!PathBuf::from(format!("{}-wal", path.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", path.display())).exists());
}

#[test]
fn session_distill_input_existing_path_wins_over_identical_session_id() {
    let fixture = Fixture::new();
    let (path, conn) = fixture.session("ambiguous");
    drop(conn);
    fixture.archive("ambiguous", "archived-id", &path);
    let input = fixture.resolve("ambiguous").unwrap();
    assert!(matches!(input, DistillInput::Archive(_)));
    assert_eq!(input.read_messages().unwrap().0, "archived-id");
    fs::write(fixture.0.join("ambiguous"), "not a ZIP").unwrap();
    assert!(fixture.resolve("ambiguous").unwrap().read_messages().is_err());
}

#[test]
fn session_distill_input_invalid_and_missing_inputs_do_not_create_sessions() {
    let fixture = Fixture::new();
    for value in ["", " ", "missing", "missing.zip", "../outside", "a/b", "a.b", "会话"] {
        assert!(fixture.resolve(value).is_err(), "unexpectedly accepted {value:?}");
    }
    assert!(fixture.resolve(&"a".repeat(129)).is_err());
    assert!(fixture.resolve(".").unwrap_err().contains("not a file"));
    assert!(!fixture.store().sessions_root().exists());
    let missing = fixture.store().session_history_file("missing");
    assert!(read_all_messages_sqlite_read_only(&missing).is_err());
    assert!(!fixture.store().sessions_root().exists());
}

#[test]
fn session_distill_input_requires_full_id_and_does_not_migrate_source() {
    let fixture = Fixture::new();
    let (path, conn) = fixture.session("full-id_123");
    conn.execute_batch("PRAGMA user_version = 7;").unwrap();
    drop(conn);
    let before = fs::read(&path).unwrap();
    assert!(fixture.resolve("full-id").unwrap_err().contains("not found"));
    assert_eq!(fixture.resolve("full-id_123").unwrap().read_messages().unwrap().1.len(), 3);
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn session_distill_input_underscore_ids_never_alias_other_sessions() {
    let fixture = Fixture::new();
    for (id, legacy_alias) in [("_abc_", "abc"), ("_", "session")] {
        let (_, conn) = fixture.session(legacy_alias);
        drop(conn);
        assert!(fixture.resolve(id).unwrap_err().contains("not found"));

        let (path, conn) = fixture.session(id);
        drop(conn);
        let input = fixture.resolve(id).unwrap();
        assert_eq!(input.path(), path);
        let (source_id, messages) = input.read_messages().unwrap();
        assert_eq!(source_id, id);
        assert_eq!(messages.len(), 3);
    }
}

#[test]
fn session_distill_input_live_wal_reads_latest_commits_not_uncommitted_rows() {
    let fixture = Fixture::new();
    let (_, conn) = fixture.session("live-session");
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA wal_autocheckpoint = 0;").unwrap();
    let input = fixture.resolve("live-session").unwrap();
    assert_eq!(input.read_messages().unwrap().1.len(), 3);
    conn.execute_batch("INSERT INTO messages (id, role, content) VALUES (40, 'user', '\"New decision\"');").unwrap();
    assert_eq!(input.read_messages().unwrap().1.len(), 4);
    conn.execute_batch("BEGIN; INSERT INTO messages (id, role, content) VALUES (50, 'user', '\"Uncommitted\"');").unwrap();
    assert_eq!(input.read_messages().unwrap().1.len(), 4);
    conn.execute_batch("ROLLBACK;").unwrap();
}

#[test]
fn session_distill_input_corrupt_local_session_fails_without_rewriting_it() {
    let fixture = Fixture::new();
    let store = fixture.store();
    store.ensure_root_dir().unwrap();
    let path = store.session_history_file("corrupt");
    fs::write(&path, "not a database").unwrap();
    let error = fixture.resolve("corrupt").unwrap().read_messages().unwrap_err();
    assert!(error.contains("Failed to read local session 'corrupt'"));
    assert_eq!(fs::read(&path).unwrap(), b"not a database");
}
