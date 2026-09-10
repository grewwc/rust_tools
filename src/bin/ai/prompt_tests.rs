use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn alternate_tail_source_lines_keeps_the_newest_non_blank_lines() {
    let text = "first\r\n\n  \nsecond\nthird\n";
    assert_eq!(
        alternate_tail_source_lines(text),
        vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string()
        ]
    );

    // Only the tail is retained: the mirror shows at most a screenful, and a
    // long answer must not stay alive in the editor just because it was the
    // last thing printed.
    let long = (0..ALTERNATE_TAIL_SOURCE_LINES + 5)
        .map(|index| format!("line {index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let lines = alternate_tail_source_lines(&long);
    assert_eq!(lines.len(), ALTERNATE_TAIL_SOURCE_LINES);
    assert_eq!(lines.first().unwrap(), "line 5");
    assert_eq!(lines.last().unwrap(), "line 44");
}

#[test]
fn session_title_update_refreshes_active_prompt_without_next_turn() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "rust_tools_prompt_topic_{}_{}",
        std::process::id(),
        unique
    ));
    let history_file = root.join("history.jsonl");
    let session_id = format!("topic-test-{unique}");

    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();
    store
        .write_session_title(&session_id, "生成后的标题")
        .unwrap();

    let mut editor = PromptEditor::new(&session_id, &history_file);
    editor.set_session_topic(Some("首条消息摘要".to_string()));

    notify_session_title_updated(&session_id, "生成后的标题");

    assert!(editor.apply_pending_session_title_updates());
    assert_eq!(editor.session_topic.as_deref(), Some("生成后的标题"));

    let _ = fs::remove_dir_all(root);
}
