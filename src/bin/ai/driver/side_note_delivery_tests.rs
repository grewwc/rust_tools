use super::*;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("side-note-delivery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn history(&self) -> PathBuf {
        self.0.join("fixture.sqlite")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn side_note_commit_failure_and_panic_retain_exact_queue() {
    let dir = TestDir::new();
    let history = dir.history();
    let queue = push_side_note(&history, "must survive", "user", None).unwrap();
    let original = fs::read_to_string(&queue).unwrap();
    assert!(consume_side_notes(&history, None, |_| Err("append failed".to_string())).is_err());
    assert_eq!(fs::read_to_string(&queue).unwrap(), original);
    let panic = std::panic::catch_unwind(|| {
        let _ = consume_side_notes(&history, None, |_| panic!("exit before canonical commit"));
    });
    assert!(panic.is_err());
    assert_eq!(fs::read_to_string(&queue).unwrap(), original);
    let count = consume_side_notes(&history, None, |notes| {
        assert_eq!(fs::read_to_string(&queue).unwrap(), original);
        crate::ai::history::append_history_messages(
            &history,
            &side_notes_to_messages(notes.to_vec()),
        )
        .map_err(|error| error.to_string())
    })
    .unwrap();
    assert_eq!(count, 1);
    assert!(!has_pending_side_notes(&history, None));
    assert_eq!(
        crate::ai::history::read_all_messages_sqlite(&history)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn side_note_snapshot_acknowledgement_preserves_new_appends() {
    let dir = TestDir::new();
    let history = dir.history();
    let queue = push_side_note(&history, "first", "user", None).unwrap();
    let abandoned = read_pending_side_notes(&history, None).unwrap().unwrap();
    drop(abandoned);
    assert!(has_pending_side_notes(&history, None));
    let pending = read_pending_side_notes(&history, None).unwrap().unwrap();
    let first_input = pending.notes()[0].content.clone();
    push_side_note(&history, "second", "user", None).unwrap();
    assert!(fs::read_to_string(&queue).unwrap().contains("first"));
    crate::ai::history::append_history_messages(
        &history,
        &[Message {
            role: "user".to_string(),
            content: serde_json::json!(first_input),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
    )
    .unwrap();
    pending.acknowledge().unwrap();
    let remaining = read_pending_side_notes(&history, None).unwrap().unwrap();
    assert_eq!(remaining.notes().len(), 1);
    assert_eq!(remaining.notes()[0].content, "second");
    assert!(!fs::read_to_string(queue).unwrap().contains("first"));
}

#[test]
fn side_note_next_input_archives_before_dequeue_and_retains_on_failure() {
    use crate::ai::history::compress::{ARCHIVE_FILE_PATH_PREFIX, ARCHIVE_NOTE_PREFIX};
    let dir = TestDir::new();
    let history = dir.history();
    let queue = push_side_note(&history, "NEXT_INPUT_GUIDANCE", "user", None).unwrap();
    let archive = assets_dir_for_history(&history).join("overflow-history.md");
    fs::create_dir_all(&archive).unwrap();
    let original = fs::read_to_string(&queue).unwrap();
    assert!(drain_side_notes_to_archive(&history, None).is_err());
    assert_eq!(fs::read_to_string(&queue).unwrap(), original);
    fs::remove_dir(&archive).unwrap();
    let notes = drain_side_notes_to_archive(&history, None).unwrap();
    assert_eq!(notes.notes.len(), 1);
    assert_eq!(notes.question, "NEXT_INPUT_GUIDANCE");
    assert!(notes.image_files.is_empty());
    assert!(archive.is_file());
    assert!(!has_pending_side_notes(&history, None));
    assert!(
        !history.exists(),
        "next-input canonical framing remains owned by the original input path"
    );

    crate::ai::history::append_history_messages(
        &history,
        &[Message {
            role: "internal_note".to_string(),
            content: serde_json::json!(format!(
                "{ARCHIVE_NOTE_PREFIX}\n{ARCHIVE_FILE_PATH_PREFIX}{}\n",
                archive.display()
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
    )
    .unwrap();
    let restored = crate::ai::history::build_message_arr_for_history_view(&history).unwrap();
    assert_eq!(
        restored
            .iter()
            .map(|(message, _)| message.clone())
            .collect::<Vec<_>>(),
        side_notes_to_messages(notes.notes)
    );
}

#[test]
fn side_note_next_input_snapshots_images_before_dequeue() {
    let dir = TestDir::new();
    let history = dir.history();
    let assets = assets_dir_for_history(&history);
    let queue = push_side_note(&history, "[[image:shot.png]]", "user", None).unwrap();
    let original_queue = fs::read_to_string(&queue).unwrap();
    assert!(drain_side_notes_to_archive(&history, None).is_err());
    assert_eq!(fs::read_to_string(&queue).unwrap(), original_queue);
    let source = assets.join("shot.png");
    fs::write(&source, b"first screenshot").unwrap();
    let input = drain_side_notes_to_archive(&history, None).unwrap();
    assert_eq!(input.notes.len(), 1);
    assert!(input.question.is_empty());
    assert_eq!(input.image_files.len(), 1);
    let snapshot = PathBuf::from(&input.image_files[0]);
    assert!(snapshot.starts_with(assets.join("attachments")));
    fs::write(source, b"replacement").unwrap();
    assert_eq!(fs::read(snapshot).unwrap(), b"first screenshot");
    assert!(!has_pending_side_notes(&history, None));
}

#[test]
fn side_note_snapshot_commit_failure_and_replaced_prefix_preserve_queue() {
    let dir = TestDir::new();
    let history = dir.history();
    let queue = push_side_note(&history, "[[image:shot.png]]", "user", None).unwrap();
    let original = fs::read_to_string(&queue).unwrap();
    let batch = read_pending_side_notes(&history, None).unwrap().unwrap();
    assert!(commit_side_note_snapshot(&batch, None, |_| Err("append failed".to_string())).is_err());
    assert_eq!(fs::read_to_string(&queue).unwrap(), original);
    assert_eq!(
        commit_side_note_snapshot(&batch, None, |notes| {
            assert_eq!(notes.len(), 1);
            Ok(())
        })
        .unwrap(),
        1
    );
    assert!(!has_pending_side_notes(&history, None));
    push_side_note(&history, "new queue prefix", "user", None).unwrap();
    let replacement = fs::read_to_string(&queue).unwrap();
    assert_eq!(
        commit_side_note_snapshot(&batch, None, |_| panic!(
            "a replaced snapshot must not commit twice"
        ))
        .unwrap(),
        0
    );
    assert_eq!(fs::read_to_string(&queue).unwrap(), replacement);
}
