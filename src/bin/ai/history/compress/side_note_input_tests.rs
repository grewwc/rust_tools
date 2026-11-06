use super::*;
use crate::ai::history::{is_side_note_message, is_user_input_message};

fn message(role: &str, text: impl Into<String>) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text.into()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn side_note(text: &str) -> Message {
    crate::ai::driver::side_note::side_notes_to_messages(vec![
        crate::ai::driver::side_note::SideNote {
            from: "user".to_string(),
            content: text.to_string(),
            ts: 42,
            target: None,
        },
    ])
    .remove(0)
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("side-note-compress-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn side_note_is_compression_protected_without_new_turn_boundaries() {
    let note = side_note("Keep the original constraints.");
    assert!(is_side_note_message(&note));
    assert!(is_runtime_synthetic_user_message(&note));
    assert!(!is_user_input_message(&note));
    let lookalike = message("user", value_to_string(&note.content));
    assert!(is_user_input_message(&lookalike));
    assert!(!is_side_note_message(&lookalike));
    assert!(!is_user_input_message(
        &crate::ai::history::runtime_synthetic_user_message(Value::String(
            "runtime image follow-up".to_string()
        ))
    ));

    let mut history = Vec::new();
    for turn in 0..200 {
        history.push(message("user", format!("input {turn}")));
        history.extend(std::iter::repeat_n(note.clone(), 5));
    }
    assert_eq!(user_input_turn_count(&history), 200);
    assert_eq!(retained_turn_start(&history, 2), 198 * 6);
    assert_eq!(last_real_user_index(&history), Some(199 * 6));
    assert_eq!(summary_delta_messages(&[note.clone()]), vec![note]);
}

#[test]
fn side_note_survives_summary_after_thousand_turns() {
    let dir = TestDir::new();
    let content = format!(
        "OPENING_CONSTRAINT\n{}\nMIDDLE_CORRECTION\n{}\nFINAL_CONDITION",
        "a".repeat(3_400),
        "b".repeat(3_400)
    );
    let note = side_note(&content);
    let mut canonical = vec![message("user", "opening request"), note.clone()];
    canonical.push(message("assistant", "opening response"));
    for turn in 1..1_000 {
        canonical.push(message("user", format!("request {turn}")));
        canonical.push(message(
            "assistant",
            format!("response {turn}: {}", "detail ".repeat(20)),
        ));
    }
    let original = canonical.clone();
    let result = compress_messages_for_context_with_outcome(
        canonical,
        14_000,
        2,
        4_000,
        Some(dir.0.clone()),
        None,
    );
    assert!(result.messages.len() < original.len());
    assert!(result.messages.iter().any(is_incremental_summary));
    assert_eq!(
        result
            .messages
            .iter()
            .filter(|m| is_side_note_message(m))
            .collect::<Vec<_>>(),
        vec![&note]
    );
    assert_eq!(original[1], note);

    let projected = crate::ai::request::normalize_messages_for_request_for_test(&result.messages);
    let visible = projected
        .iter()
        .find(|m| m.content == note.content)
        .expect("complete guidance in request projection");
    assert!(visible.reasoning_content.is_none());

    let result = compress_messages_for_context_with_outcome(
        result.messages,
        1_000,
        1,
        4_000,
        Some(dir.0.clone()),
        None,
    );
    assert!(
        !result.budget_met(),
        "protected guidance alone exceeds the budget"
    );
    assert!(result.messages.contains(&note));
}

#[test]
fn side_note_is_not_replaced_by_offload_or_mutable_truncation() {
    let dir = TestDir::new();
    let note = side_note(&format!(
        "KEEP_ALL\n{}\nKEEP_END",
        "exact guidance ".repeat(800)
    ));
    let mut messages = vec![message("user", "opening"), note.clone()];
    for turn in 1..6 {
        messages.push(message("user", format!("next {turn}")));
        messages.push(message("assistant", "noise ".repeat(400)));
    }
    let shrunk = shrink_messages_to_fit(
        messages,
        1_000,
        Some(&dir.0),
        None,
        &rustc_hash::FxHashSet::default(),
    );
    assert!(shrunk.contains(&note));
    let mut shrunk = shrunk;
    truncate_mutable_messages_to_fit(
        &mut shrunk,
        100,
        Some(&dir.0),
        &rustc_hash::FxHashSet::default(),
    );
    emergency_cap_messages_to_fit(
        &mut shrunk,
        100,
        80,
        Some(&dir.0),
        &rustc_hash::FxHashSet::default(),
    );
    assert!(shrunk.contains(&note));
}
