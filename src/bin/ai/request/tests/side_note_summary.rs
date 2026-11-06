use super::*;

fn side_note(text: &str, ts: u64) -> Message {
    crate::ai::driver::side_note::side_notes_to_messages(vec![
        crate::ai::driver::side_note::SideNote {
            from: "user".to_string(),
            content: text.to_string(),
            ts,
            target: None,
        },
    ])
    .remove(0)
}

#[test]
fn summary_input_keeps_complete_early_side_note() {
    let content = format!(
        "EARLY_GUIDANCE\n{}\nMIDDLE_CONDITION\n{}\nFINAL_CONDITION",
        "中".repeat(3_500),
        "x".repeat(3_500)
    );
    let note = side_note(&content, 42);
    let mut messages = vec![note.clone()];
    for index in 0..30 {
        messages.push(Message {
            role: "user".to_string(),
            content: Value::String(format!("later {index} {}", "noise ".repeat(180))),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    let transcript = build_summary_history_input(&messages, 10_000);
    assert!(transcript.contains(note.content.as_str().unwrap()));
    assert!(transcript.starts_with("### SIDE-NOTE\n\n[side-note from=user ts=42]"));
    assert!(!crate::ai::history::is_user_input_message(&note));
    assert!(crate::ai::history::is_runtime_synthetic_user_message(&note));
    assert!(transcript.chars().count() <= 10_000);
    assert!(!transcript.contains("runtime-origin:synthetic-user"));
}

#[test]
fn summary_input_refuses_partial_side_note_batch() {
    let messages = vec![
        side_note(&"first ".repeat(120), 1),
        side_note(&"second ".repeat(120), 2),
    ];
    assert!(build_summary_history_input(&messages, 1_000).is_empty());
    assert!(render_summary_message(&messages[0], 80).is_empty());
}
