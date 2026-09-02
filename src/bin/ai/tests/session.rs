//! Session lifecycle and cleanup tests.

use std::path::PathBuf;
use serde_json::Value;

use super::super::history::{
    Message, SessionStore, append_history_messages, build_message_arr,
};

#[test]
fn session_delete_cleans_up_overflow_history_file() {
    let session_id = format!("test-{}", uuid::Uuid::new_v4());
    let history_file = std::env::temp_dir().join(format!(
        "ai-session-cleanup-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file(&session_id);
    let assets = store.session_assets_dir(&session_id);
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(
        assets.join("overflow-history.md"),
        "# test overflow content",
    )
    .unwrap();
    let preserved_tool_dir = assets.join("tool-overflow-compressed");
    std::fs::create_dir_all(&preserved_tool_dir).unwrap();
    std::fs::write(
        preserved_tool_dir.join("read_file-test.txt"),
        "temporary preserved tool output",
    )
    .unwrap();
    std::fs::write(&db, b"test").unwrap();

    assert!(assets.join("overflow-history.md").exists());

    store.delete_session(&session_id).unwrap();

    assert!(
        !assets.exists(),
        "assets dir (including overflow file) should be deleted"
    );
    assert!(!db.exists(), "sqlite file should be deleted");

    let _ = std::fs::remove_dir_all(store.session_assets_dir("__cleanup__"));
}

/// `temp_dir()` now shares its root with tool-overflow, landing under `session_assets_dir/tmp/`.
/// Verifies that `delete_session` recursively cleans up `tmp/temp_registry.json` along with it.
#[test]
fn session_delete_cleans_up_temp_registry() {
    let session_id = format!("test-{}", uuid::Uuid::new_v4());
    let history_file =
        std::env::temp_dir().join(format!("ai-temp-cleanup-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file(&session_id);
    let assets = store.session_assets_dir(&session_id);
    let tmp_dir = assets.join("tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    // Simulate temp files written by write_file(temp=true) plus the registry
    std::fs::write(tmp_dir.join("scratch.rs"), "fn main() {}").unwrap();
    std::fs::write(tmp_dir.join("temp_registry.json"), r#"["scratch.rs"]"#).unwrap();
    std::fs::write(&db, b"test").unwrap();

    assert!(tmp_dir.join("temp_registry.json").exists());
    assert!(tmp_dir.join("scratch.rs").exists());

    store.delete_session(&session_id).unwrap();

    assert!(
        !assets.exists(),
        "assets dir (including tmp/) should be deleted"
    );
    assert!(!db.exists(), "sqlite file should be deleted");

    let _ = std::fs::remove_dir_all(store.session_assets_dir("__cleanup__"));
}

#[test]
fn session_delete_removes_sqlite_sidecars() {
    let history_file =
        std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(history_file.as_path());
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file("abc");
    std::fs::write(&db, b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-wal", db.display())), b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-shm", db.display())), b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-journal", db.display())), b"test").unwrap();
    let derived = store.sessions_root().join("abc.proc-42.sqlite");
    std::fs::write(&derived, b"derived").unwrap();
    std::fs::write(
        PathBuf::from(format!("{}-wal", derived.display())),
        b"derived",
    )
    .unwrap();
    let derived_state_lock = store.sessions_root().join(".abc.proc-42.sqlite.state.lock");
    std::fs::write(&derived_state_lock, b"lock").unwrap();
    let legacy_derived = store.sessions_root().join("abc.sqlite.subagent-legacy");
    std::fs::write(&legacy_derived, b"legacy").unwrap();
    let assets = store.session_assets_dir("abc");
    let checkpoints = store.checkpoints_dir("abc");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(assets.join("paste.png"), b"test").unwrap();
    std::fs::create_dir_all(&checkpoints).unwrap();
    std::fs::write(checkpoints.join("saved.sqlite"), b"test").unwrap();

    assert!(db.exists());
    assert!(PathBuf::from(format!("{}-wal", db.display())).exists());
    assert!(PathBuf::from(format!("{}-shm", db.display())).exists());
    assert!(PathBuf::from(format!("{}-journal", db.display())).exists());
    assert!(derived.exists());
    assert!(PathBuf::from(format!("{}-wal", derived.display())).exists());
    assert!(derived_state_lock.exists());
    assert!(legacy_derived.exists());
    assert!(assets.exists());
    assert!(checkpoints.exists());

    assert!(store.delete_session("abc").unwrap());

    assert!(!db.exists());
    assert!(!PathBuf::from(format!("{}-wal", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-journal", db.display())).exists());
    assert!(!derived.exists());
    assert!(!PathBuf::from(format!("{}-wal", derived.display())).exists());
    assert!(!derived_state_lock.exists());
    assert!(!legacy_derived.exists());
    assert!(!assets.exists());
    assert!(!checkpoints.exists());
}

#[test]
fn session_clear_history_removes_checkpoint_snapshots() {
    let history_file =
        std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(history_file.as_path());
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file("abc");
    append_history_messages(
        &db,
        &[Message {
            role: "user".to_string(),
            content: Value::String("clear this".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
    )
    .unwrap();
    let assets = store.session_assets_dir("abc");
    let checkpoints = store.checkpoints_dir("abc");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(assets.join("paste.png"), b"test").unwrap();
    std::fs::create_dir_all(&checkpoints).unwrap();
    std::fs::write(checkpoints.join("saved.sqlite"), b"test").unwrap();

    store.clear_session_history("abc").unwrap();

    assert!(build_message_arr(10, &db).unwrap().is_empty());
    assert!(!assets.exists());
    assert!(!checkpoints.exists());
}

#[test]
fn session_clear_all_removes_all_sqlite_sidecars() {
    let history_file =
        std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(history_file.as_path());
    store.ensure_root_dir().unwrap();

    for id in ["a", "b", "c"] {
        let db = store.session_history_file(id);
        std::fs::write(&db, b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-wal", db.display())), b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-shm", db.display())), b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-journal", db.display())), b"test").unwrap();
        let derived = store
            .sessions_root()
            .join(format!("{id}.subagent-child.sqlite"));
        std::fs::write(&derived, b"derived").unwrap();
        std::fs::write(
            PathBuf::from(format!("{}-shm", derived.display())),
            b"derived",
        )
        .unwrap();
        let assets = store.session_assets_dir(id);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("paste.png"), b"test").unwrap();
        let checkpoints = store.checkpoints_dir(id);
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::write(checkpoints.join("saved.sqlite"), b"test").unwrap();
    }
    let orphan_checkpoints = store.checkpoints_dir("orphan");
    std::fs::create_dir_all(&orphan_checkpoints).unwrap();
    std::fs::write(orphan_checkpoints.join("saved.sqlite"), b"test").unwrap();
    let orphan_derived = store.sessions_root().join("orphan.proc-1.sqlite");
    std::fs::write(&orphan_derived, b"orphan").unwrap();

    let deleted = store.clear_all_sessions().unwrap();
    assert_eq!(deleted, 3);

    for id in ["a", "b", "c"] {
        let db = store.session_history_file(id);
        assert!(!db.exists());
        assert!(!PathBuf::from(format!("{}-wal", db.display())).exists());
        assert!(!PathBuf::from(format!("{}-shm", db.display())).exists());
        assert!(!PathBuf::from(format!("{}-journal", db.display())).exists());
        let derived = store
            .sessions_root()
            .join(format!("{id}.subagent-child.sqlite"));
        assert!(!derived.exists());
        assert!(!PathBuf::from(format!("{}-shm", derived.display())).exists());
        let assets = store.session_assets_dir(id);
        assert!(!assets.exists());
        assert!(!store.checkpoints_dir(id).exists());
    }
    assert!(!orphan_checkpoints.exists());
    assert!(!orphan_derived.exists());
}
