//! Pin the sequential spill policy, exact archive bytes, and cached budget totals.

use super::*;
use crate::ai::types::{FunctionCall, ToolCall};

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn image() -> Message {
    Message {
        role: "assistant".to_string(),
        content: serde_json::json!([
            {"type": "text", "text": "figure α"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", "A".repeat(768))}}
        ]),
        tool_calls: Some(Vec::new()),
        tool_call_id: Some("image-id".to_string()),
        reasoning_content: Some("image reasoning 🎈".to_string()),
    }
}

/// Independent copy of the original selector, including its per-round rescans.
fn sequential_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep = keep_recent_user_turns_when_trimming(messages, budget);
    let protected_tail_start = retained_turn_start(messages, keep);
    for (idx, message) in messages.iter().enumerate() {
        if idx >= protected_tail_start {
            break;
        }
        if is_system_like_role(&message.role)
            || message.role == "tool"
            || super::super::is_side_note_message(message)
        {
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            continue;
        }
        let text = value_to_string(&message.content);
        if is_preserved_user_or_image_stub(&text) {
            continue;
        }
        let char_count = if message_contains_image(&message.content) {
            message.content.to_string().chars().count()
        } else {
            text.chars().count()
        };
        if message_contains_image(&message.content) && char_count >= IMAGE_OVERFLOW_SPILL_MIN_CHARS
        {
            return Some(idx);
        }
        if message.role == "user" && char_count >= USER_OVERFLOW_SPILL_MIN_CHARS {
            return Some(idx);
        }
    }
    None
}

fn assert_cached_totals(cursor: &PreservedContentSpillCursor, messages: &[Message], budget: usize) {
    assert_eq!(
        cursor.total_chars,
        super::super::messages_total_chars(messages)
    );
    for (index, window) in cursor.windows.iter().enumerate() {
        let start = retained_turn_start(messages, index + 1);
        assert_eq!(window.start, start);
        assert_eq!(
            window.chars,
            super::super::messages_total_chars(&messages[start..])
        );
    }
    let keep = keep_recent_user_turns_when_trimming(messages, budget);
    assert_eq!(
        cursor.protected_tail_start(budget),
        retained_turn_start(messages, keep)
    );
}

fn archive_snapshot(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn collect(root: &Path, dir: &Path, files: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(root, &path, files);
            } else {
                files.push((
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(path).unwrap(),
                ));
            }
        }
    }
    let mut files = Vec::new();
    collect(root, root, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));
    files
}

fn assert_sequential_parity(
    fixture: Vec<Message>,
    budget: usize,
    expected_spills: &[usize],
    block_user_archive: bool,
) {
    let expect_image_archive = expected_spills.iter().any(|idx| fixture[*idx] == image());
    let root = std::env::temp_dir().join(format!("ai-preserved-spill-{}", uuid::Uuid::new_v4()));
    let sequential_dir = root.join("sequential");
    let actual_dir = root.join("actual");
    for dir in [&sequential_dir, &actual_dir] {
        std::fs::create_dir_all(dir).unwrap();
        if block_user_archive {
            std::fs::write(dir.join(PRESERVED_USER_OVERFLOW_DIR), "blocked").unwrap();
        }
    }

    let mut expected = fixture.clone();
    let mut cursor = PreservedContentSpillCursor::new(&expected);
    let mut sequence = Vec::new();
    loop {
        assert_cached_totals(&cursor, &expected, budget);
        let candidate = sequential_candidate(&expected, budget);
        assert_eq!(cursor.next_candidate(&expected, budget), candidate);
        let Some(idx) = candidate else {
            break;
        };
        let kind = if message_contains_image(&expected[idx].content) {
            "image"
        } else {
            "user"
        };
        let before = super::super::message_billable_chars(&expected[idx]);
        // Keep the original commit-before-replacement and stop-on-first-failure order.
        let Some(path) =
            write_preserved_message_overflow_file(&sequential_dir, &expected[idx], kind)
        else {
            break;
        };
        expected[idx].content = Value::String(build_preserved_message_overflow_stub(&path, kind));
        drop_request_reminder(&mut expected[idx]);
        cursor.record_replacement(
            idx,
            before,
            super::super::message_billable_chars(&expected[idx]),
        );
        sequence.push(idx);
    }
    assert_eq!(sequence, expected_spills);
    assert_cached_totals(&cursor, &expected, budget);

    let mut actual = fixture;
    spill_oversized_preserved_messages(&mut actual, &actual_dir, budget);
    assert_eq!(
        serde_json::to_vec(&actual).unwrap(),
        serde_json::to_vec(&expected).unwrap()
    );
    let snapshot = archive_snapshot(&actual_dir);
    assert_eq!(snapshot, archive_snapshot(&sequential_dir));

    if expect_image_archive {
        let (_, bytes) = snapshot
            .iter()
            .find(|(path, _)| {
                path == &Path::new(PRESERVED_IMAGE_OVERFLOW_DIR)
                    .join("image-8e5b963cc9aceb18eb45caeb.json")
            })
            .expect("the image archive must retain its original digest");
        let golden = format!(
            "{{\n  \"role\": \"assistant\",\n  \"kind\": \"image\",\n  \"content\": [\n    {{\n      \"type\": \"text\",\n      \"text\": \"figure α\"\n    }},\n    {{\n      \"type\": \"image_url\",\n      \"image_url\": {{\n        \"url\": \"data:image/png;base64,{}\"\n      }}\n    }}\n  ],\n  \"tool_calls\": [],\n  \"tool_call_id\": \"image-id\"\n}}",
            "A".repeat(768)
        );
        assert_eq!(bytes.as_slice(), golden.as_bytes());
    }
    // Rebuilding from already-spilled messages must neither change stubs nor rewrite bytes.
    spill_oversized_preserved_messages(&mut actual, &actual_dir, budget);
    assert_eq!(actual, expected);
    assert_eq!(archive_snapshot(&actual_dir), snapshot);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn preserved_spill_matches_sequential_sequence_and_archive_bytes() {
    let mut first = message("user", &"ü".repeat(USER_OVERFLOW_SPILL_MIN_CHARS));
    crate::ai::history::attach_request_reminder(&mut first, "<system-reminder>α</system-reminder>");
    let mut tool_image = image();
    tool_image.role = "tool".to_string();
    let mut call_image = image();
    call_image.tool_calls = Some(vec![ToolCall {
        id: "call".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        },
    }]);
    let side_note = crate::ai::history::runtime_synthetic_user_message(Value::String(format!(
        "[side-note from=reviewer ts=123] Live guidance (side-note) from reviewer: {}",
        "S".repeat(2_000)
    )));
    assert_sequential_parity(
        vec![
            message("system", &"S".repeat(2_048)),
            first,
            message("user", &"ü".repeat(USER_OVERFLOW_SPILL_MIN_CHARS)),
            image(),
            tool_image,
            call_image,
            side_note,
            crate::ai::history::runtime_synthetic_user_message(Value::String("N".repeat(1_024))),
            message("user", "third protected turn"),
            image(),
            message("user", "second protected turn"),
            message("assistant", "a"),
            message("user", &"Z".repeat(1_024)),
        ],
        128_000,
        &[1, 2, 3, 7],
        false,
    );

    // A spill crosses 48K: the third-most-recent turn becomes protected mid-pass.
    assert_sequential_parity(
        vec![
            message("system", "sys"),
            message("user", &"A".repeat(32_000)),
            message("user", &"B".repeat(10_000)),
            message("user", &"C".repeat(10_000)),
            message("user", &"D".repeat(2_000)),
            message("user", "second"),
            message("user", "newest"),
        ],
        128_000,
        &[1, 2, 3],
        false,
    );

    // A message larger than the entire budget spills; the oversized newest turn stays raw.
    assert_sequential_parity(
        vec![
            message("system", "sys"),
            message("user", &"A".repeat(20_000)),
            image(),
            message("user", &"M".repeat(3_000)),
            message("assistant", &"a".repeat(2_500)),
            message("user", &"R".repeat(6_000)),
            message("assistant", "ok"),
        ],
        4_000,
        &[1, 2, 3],
        false,
    );

    // Updating a formerly oversized tail can protect an image that was initially eligible.
    assert_sequential_parity(
        vec![
            message("system", "sys"),
            message("user", &"A".repeat(5_000)),
            message("user", &"M".repeat(3_000)),
            image(),
            message("user", &"R".repeat(500)),
        ],
        4_000,
        &[1, 2],
        false,
    );

    // A later failed archive retains its original and prevents every subsequent spill.
    assert_sequential_parity(
        vec![
            message("system", "sys"),
            image(),
            message("user", &"A".repeat(2_000)),
            image(),
            message("user", "u2"),
            message("user", "u3"),
            message("user", "u4"),
            message("user", "u5"),
        ],
        128_000,
        &[1],
        true,
    );
}

#[test]
fn preserved_spill_cursor_matches_sequential_across_budget_boundaries() {
    for user_turns in 0..=8 {
        for budget in [0, 1, 511, 1_024, 4_000, 48_000, 128_000] {
            let mut messages = vec![message("system", "sys"), image()];
            for turn in 0..user_turns {
                let chars = [1_023, 1_024, 17_000, 30_000, 4_000][turn % 5];
                let mut user = message("user", &"界".repeat(chars));
                if turn % 2 == 0 {
                    crate::ai::history::attach_request_reminder(&mut user, "saved reminder 🧭");
                }
                messages.push(user);
                messages.push(image());
                messages.push(message("assistant", &"n".repeat(1_025)));
                messages.push(crate::ai::history::runtime_synthetic_user_message(
                    Value::String("N".repeat(1_024)),
                ));
            }
            let mut cursor = PreservedContentSpillCursor::new(&messages);
            loop {
                assert_cached_totals(&cursor, &messages, budget);
                let previous = cursor.next;
                let candidate = cursor.next_candidate(&messages, budget);
                assert_eq!(candidate, sequential_candidate(&messages, budget));
                let Some(idx) = candidate else {
                    break;
                };
                assert!(idx >= previous && cursor.next == idx + 1);
                let before = super::super::message_billable_chars(&messages[idx]);
                let kind = if message_contains_image(&messages[idx].content) {
                    "image"
                } else {
                    "user"
                };
                messages[idx].content = Value::String(build_preserved_message_overflow_stub(
                    &Path::new("/test")
                        .join(PRESERVED_USER_OVERFLOW_DIR)
                        .join("fixture.json"),
                    kind,
                ));
                drop_request_reminder(&mut messages[idx]);
                cursor.record_replacement(
                    idx,
                    before,
                    super::super::message_billable_chars(&messages[idx]),
                );
            }
            assert_cached_totals(&cursor, &messages, budget);
        }
    }
}
