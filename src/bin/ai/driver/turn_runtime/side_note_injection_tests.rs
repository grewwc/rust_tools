use super::*;
use crate::ai::driver::side_note::{current_target_id, push_side_note, side_note_file};

fn canonical_user(text: &str) -> history::Message {
    history::Message {
        role: "user".to_string(),
        content: serde_json::json!(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

#[test]
fn side_note_injection_commits_canonical_order_without_duplicate_append() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!(
        "side-note-canonical-order-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = crate::ai::driver::tests::test_app("foreground");
    app.session_history_file = dir.join("fixture.sqlite");
    let target = current_target_id();
    push_side_note(
        &app.session_history_file,
        "new constraint",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let mut turn_messages = vec![
        canonical_user("original request"),
        history::Message {
            role: "assistant".to_string(),
            ..canonical_user("preceding assistant text")
        },
    ];
    let mut messages = turn_messages.clone();
    let mut persisted = 0;
    drain_side_notes_and_inject(&app, &mut messages, &mut turn_messages, &mut persisted).unwrap();
    assert_eq!(persisted, 3);
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file).unwrap(),
        turn_messages
    );
    assert_eq!(messages, turn_messages);
    assert!(history::is_side_note_message(&turn_messages[2]));
    assert_eq!(
        std::fs::metadata(side_note_file(&app.session_history_file, target.as_deref()))
            .unwrap()
            .len(),
        0
    );
    assert!(persist_pending_turn_messages(
        &app,
        false,
        &turn_messages,
        &mut persisted
    ));
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file)
            .unwrap()
            .len(),
        3
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn side_note_injection_append_failure_keeps_queue_and_memory_order() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!(
        "side-note-canonical-failure-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = crate::ai::driver::tests::test_app("foreground");
    app.session_history_file = dir.join("blocked.sqlite");
    std::fs::create_dir_all(&app.session_history_file).unwrap();
    let target = current_target_id();
    let queue = push_side_note(
        &app.session_history_file,
        "must retry",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let original_queue = std::fs::read_to_string(&queue).unwrap();
    let original_messages = vec![canonical_user("original request")];
    let mut messages = original_messages.clone();
    let mut turn_messages = original_messages.clone();
    let mut persisted = 0;
    assert!(
        drain_side_notes_and_inject(&app, &mut messages, &mut turn_messages, &mut persisted)
            .is_err()
    );
    assert_eq!(messages, original_messages);
    assert_eq!(turn_messages, original_messages);
    assert_eq!(persisted, 0);
    assert_eq!(std::fs::read_to_string(queue).unwrap(), original_queue);
    let _ = std::fs::remove_dir_all(dir);
}
