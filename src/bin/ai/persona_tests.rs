use super::*;

fn temp_store() -> PersonaStore {
    let dir = std::env::temp_dir().join(format!("rust-tools-personas-{}", Uuid::new_v4()));
    PersonaStore::for_tests_with_path(dir.join("personas.json"))
}

#[test]
fn history_path_for_default_persona_preserves_base() {
    let base = PathBuf::from("/tmp/history.sqlite");
    assert_eq!(history_file_for_persona(&base, DEFAULT_PERSONA_ID), base);
}

#[test]
fn history_path_for_named_persona_gets_isolated_suffix() {
    let base = PathBuf::from("/tmp/history.sqlite");
    let path = history_file_for_persona(&base, "persona-123");
    assert_eq!(
        path,
        PathBuf::from("/tmp/history.persona-persona-123.sqlite")
    );
}

#[test]
fn create_persona_rejects_duplicate_name() {
    let store = temp_store();
    store
        .create_persona("Alice", Some("A"), "You are Alice.")
        .unwrap();
    let err = store
        .create_persona("alice", Some("B"), "You are another Alice.")
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
}

#[test]
fn delete_current_persona_falls_back_to_default() {
    let store = temp_store();
    let persona = store
        .create_persona("Reviewer", Some("R"), "You are a reviewer.")
        .unwrap();
    store.set_active_persona("Reviewer").unwrap();
    let result = store.delete_persona(&persona.id).unwrap();
    assert_eq!(result.removed.name, "Reviewer");
    assert!(result.active_after.is_default());
}

#[test]
fn remember_session_updates_persona_binding() {
    let store = temp_store();
    let persona = store
        .create_persona("Planner", None, "You are a planner.")
        .unwrap();
    store.remember_session(&persona.id, "session-1").unwrap();
    let all = store.list_personas().unwrap();
    let planner = all
        .into_iter()
        .find(|item| item.name == "Planner")
        .expect("planner persona should exist");
    assert_eq!(planner.last_session_id.as_deref(), Some("session-1"));
}
