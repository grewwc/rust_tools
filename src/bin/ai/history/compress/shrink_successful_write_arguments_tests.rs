use super::truncate::{
    build_context_overflow_tool_arguments_stub, minimize_truncated_previews_for_hard_budget,
};
use super::*;
use crate::ai::types::{FunctionCall, ToolCall};
use rustc_hash::FxHashSet;

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn assistant_call_args(id: &str, name: &str, args: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn tool_result(id: &str, content: &str) -> Message {
    Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    }
}

fn big_write_args() -> String {
    format!(
        r#"{{"file_path": "/tmp/out.txt", "content": "{}"}}"#,
        "x".repeat(5000)
    )
}

fn big_patch_args() -> String {
    serde_json::json!({
        "patch": format!("*** Begin Patch\n*** Update File: /tmp/a.rs\n@@\n-{}\n+{}\n*** End Patch",
            "old line".repeat(200), "new line".repeat(200)),
    })
    .to_string()
}

fn temp_overflow_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ai-shrink-write-{}", uuid::Uuid::new_v4()))
}

fn append_recent_groups(messages: &mut Vec<Message>, count: usize) {
    for i in 0..count {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }
}

fn folded_write_note(messages: &[Message]) -> &Message {
    let notes = messages
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .collect::<Vec<_>>();
    assert_eq!(notes.len(), 1, "one evidence note replaces the write pair");
    let note = notes[0];
    assert!(has_valid_folded_tool_provenance(note));
    assert!(note.tool_calls.is_none());
    assert!(note.tool_call_id.is_none());
    note
}

fn assert_call_pair_removed(messages: &[Message], id: &str) {
    assert!(
        !messages.iter().any(|message| {
            message.tool_call_id.as_deref() == Some(id)
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| calls.iter().any(|call| call.id == id))
        }),
        "folded calls and results must disappear together"
    );
}

fn assert_messages_archived(archive_path: &std::path::Path, original_messages: &[Message]) {
    let archived = std::fs::read_to_string(archive_path).expect("verbatim archive exists");
    for original in original_messages {
        assert!(
            archived.contains(&serde_json::to_string_pretty(original).unwrap()),
            "archive must retain each complete original message JSON"
        );
    }
}

/// A successful write outside the recent window becomes an evidence note;
/// both original messages, including arguments and reasoning, are archived verbatim.
#[test]
fn shrink_successful_write_outside_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    let mut assistant = assistant_call_args("call-w", "write_file", &big_args);
    assistant.content = Value::String("I wrote the requested file.".to_string());
    assistant.reasoning_content = Some("original assistant reasoning".to_string());
    messages.push(assistant);
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    let originals = messages[2..4].to_vec();
    // Fill the four-group protection window so the write group ages out.
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    let note = folded_write_note(&messages);
    let text = note.content.as_str().unwrap();
    assert!(text.contains("write_file => success"));
    assert!(text.contains("/tmp/out.txt"));
    assert!(text.contains(&format!("- archive_file_path: {}", archive_path.display())));
    assert!(!text.contains("_context_overflow_truncated"));
    assert_call_pair_removed(&messages, "call-w");
    assert_messages_archived(&archive_path, &originals);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Recent successful writes retain their full arguments for follow-up edits.
#[test]
fn keep_write_arguments_inside_recent_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args, "recent window keeps full arguments");
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Failed writes keep full arguments so the model can fix or retry them.
#[test]
fn failed_write_keeps_arguments() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result(
        "call-w",
        "Error: write_file failed: permission denied",
    ));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args, "failed write keeps full arguments");
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Old successful patches fold into evidence with both messages archived verbatim.
#[test]
fn shrink_successful_apply_patch_outside_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = format!(
        r#"{{"patch": "*** Begin Patch\n*** Update File: /tmp/a.rs\n@@\n-{}\n+{}\n*** End Patch"}}"#,
        "old line".repeat(200),
        "new line".repeat(200)
    );
    messages.push(assistant_call_args("call-p", "apply_patch", &big_args));
    messages.push(tool_result(
        "call-p",
        "Successfully patched /tmp/a.rs; +1 -1 (1 lines)",
    ));
    let originals = messages[2..4].to_vec();
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let note = folded_write_note(&messages);
    let text = note.content.as_str().unwrap();
    assert!(text.contains("apply_patch => success"));
    assert!(text.contains("/tmp/a.rs"));
    assert_call_pair_removed(&messages, "call-p");
    assert_messages_archived(&overflow_dir.join(OVERFLOW_HISTORY_FILENAME), &originals);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Arguments at or below 160 chars stay intact when there is no useful saving.
#[test]
fn short_arguments_unchanged() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let small_args = r#"{"file_path": "/tmp/a.txt", "content": "hi"}"#.to_string();
    messages.push(assistant_call_args("call-s", "write_file", &small_args));
    messages.push(tool_result("call-s", "Successfully wrote to /tmp/a.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &small_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Folding is idempotent: a repeated pass neither changes notes nor grows the archive.
#[test]
fn idempotent_when_already_replaced() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);
    let messages_after_first = messages.clone();
    let archive_after_first = std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME))
        .unwrap()
        .len();

    // A second pass preserves the note and does not append another archive batch.
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);
    assert_eq!(messages, messages_after_first);
    let archive_after_second =
        std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME))
            .unwrap()
            .len();
    assert_eq!(
        archive_after_second, archive_after_first,
        "no duplicate archive"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Current-turn protected call IDs retain full arguments even outside the window.
#[test]
fn protected_call_ids_never_shrunk() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let mut protected = FxHashSet::default();
    protected.insert("call-w".to_string());
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Non-write tools are outside the successful write folding policy.
#[test]
fn non_write_tools_ignored() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = format!(r#"{{"file_path": "/tmp/a.rs", "limit": 1000}}"#);
    messages.push(assistant_call_args("call-r", "read_file", &big_args));
    messages.push(tool_result("call-r", "some content"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Without an overflow directory, both original messages must remain intact.
#[test]
fn no_overflow_dir_still_works() {
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let originals = messages.clone();
    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, None, &protected);
    assert_eq!(
        messages, originals,
        "no archive means no replacement or removal"
    );
}

#[test]
fn tool_argument_pointer_keeps_original_chars_and_drops_legacy_preview() {
    let legacy = serde_json::json!({
        "_context_overflow_truncated": true,
        "original_chars": 5_000,
        "archive_file_path": "/tmp/archive.md",
        "preview": "{\"file_path\":\"/tmp/out.txt\",\"content\":\"partial payload\"}",
    })
    .to_string();
    let pointer = build_context_overflow_tool_arguments_pointer(&legacy, 160)
        .expect("legacy preview should minimize to a metadata-only pointer");
    assert_eq!(
        serde_json::from_str::<Value>(&pointer).unwrap(),
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": 5_000,
            "archive_file_path": "/tmp/archive.md",
        })
    );
    assert!(is_context_overflow_truncated_tool_arguments(&pointer));
    assert!(build_context_overflow_tool_arguments_pointer(&pointer, usize::MAX).is_none());
}

/// Medium arguments must stay intact when the evidence note would be larger.
/// Reject before archiving so repeated compression cannot grow unused archive batches.
#[test]
fn medium_arguments_not_shrunk_and_not_archived() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    // About 200 chars exceeds the trigger, but cannot pay for the evidence note.
    let medium_args = format!(
        r#"{{"file_path": "/tmp/med.txt", "content": "{}"}}"#,
        "m".repeat(160)
    );
    assert!(
        medium_args.chars().count() > 160 && medium_args.chars().count() <= 240,
        "medium_args chars = {}",
        medium_args.chars().count()
    );
    messages.push(assistant_call_args("call-m", "write_file", &medium_args));
    messages.push(tool_result("call-m", "Successfully wrote to /tmp/med.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    // Locate by ID so the assertion does not depend on archive note placement.
    let write_idx = messages
        .iter()
        .position(|m| {
            m.tool_calls
                .as_ref()
                .is_some_and(|calls| calls[0].id == "call-m")
        })
        .unwrap();
    let args = &messages[write_idx].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(
        args, &medium_args,
        "medium args stay intact when folding cannot shrink"
    );
    // No archive batch should be written for a rejected size candidate.
    assert!(
        !overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists(),
        "no archive write when truncation is impossible"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Reject argument truncation when a long archive path leaves no preview budget.
/// Empty previews previously encouraged the model to copy recovery metadata as
/// real tool parameters. Keep the original arguments and write no archive batch.
#[test]
fn empty_preview_stub_rejected_when_path_eats_budget() {
    // Use a long directory name so the archive locator exhausts the 160-char target.
    let long_segment = "d".repeat(140);
    let overflow_dir = std::env::temp_dir().join(long_segment);
    let original_args = format!(
        r#"{{"file_path":"/tmp/x.txt","content":"{}"}}"#,
        "c".repeat(400)
    );
    let mut message = assistant_call_args("call-x", "write_file", &original_args);
    // Original arguments are about 460 chars; a 300-char cut targets 160 chars.
    let reduced = truncate_mutable_field(
        &mut message,
        MutableMessageField::ToolArguments(0),
        300,
        Some(&overflow_dir),
        FieldArchivePolicy::BestEffort,
    );
    assert!(
        !reduced,
        "must reject truncation when preview budget is exhausted"
    );
    let args = &message.tool_calls.as_ref().unwrap()[0].function.arguments;
    assert_eq!(args, &original_args, "arguments must be left intact");
    assert!(
        !overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists(),
        "no archive write when truncation is rejected"
    );
}

#[test]
fn metadata_only_tool_argument_stub_rejects_long_archive_path() {
    let target = 160;
    let original_args = big_write_args();
    let original_chars = original_args.chars().count();
    let overflow_dir = temp_overflow_dir().join("d".repeat(180));
    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    let archive_path = archive_path.to_string_lossy();
    assert!(archive_path.chars().count() > target);
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, Some(&archive_path), target)
            .is_none(),
        "the final metadata builder must reject an oversized archive locator"
    );

    let mut message = assistant_call_args("call-long-path", "write_file", &original_args);
    assert!(!truncate_mutable_field(
        &mut message,
        MutableMessageField::ToolArguments(0),
        original_chars - target,
        Some(&overflow_dir),
        FieldArchivePolicy::BestEffort,
    ));
    assert_eq!(
        message.tool_calls.as_ref().unwrap()[0].function.arguments,
        original_args
    );
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());

    let no_path = build_context_overflow_tool_arguments_stub(original_chars, None, target)
        .expect("the existing no-path shape should still fit");
    assert_eq!(
        serde_json::from_str::<Value>(&no_path).unwrap(),
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": original_chars,
            "archive_file_path": Value::Null,
        })
    );
    assert!(is_context_overflow_truncated_tool_arguments(&no_path));
    let pointer_chars = no_path.chars().count();
    assert!(pointer_chars <= target && pointer_chars < original_chars);
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, None, pointer_chars).is_some()
    );
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, None, pointer_chars - 1)
            .is_none()
    );
    assert!(build_context_overflow_tool_arguments_stub(1, None, target).is_none());
}

#[test]
fn successful_write_protection_preserves_complete_pairs_for_both_tools() {
    for tool_name in ["write_file", "apply_patch"] {
        for case in [
            "recent",
            "failed",
            "exit_error",
            "protected",
            "no_sink",
            "missing_result",
        ] {
            let overflow_dir = temp_overflow_dir();
            let args = if tool_name == "write_file" {
                big_write_args()
            } else {
                big_patch_args()
            };
            let result = match case {
                "failed" => "Error: write failed; preserve original diagnostic text",
                "exit_error" => "Exit code: 1\nwrite failed",
                _ if tool_name == "write_file" => "Successfully wrote to /tmp/out.txt",
                _ => "Successfully patched /tmp/a.rs; +1 -1 (1 lines)",
            };
            let mut messages = vec![
                msg("system", "s"),
                msg("user", "do"),
                assistant_call_args("call-write", tool_name, &args),
            ];
            if case != "missing_result" {
                messages.push(tool_result("call-write", result));
            }
            let recent_count = if case == "recent" {
                KEEP_RECENT_TOOL_GROUPS - 1
            } else {
                KEEP_RECENT_TOOL_GROUPS
            };
            append_recent_groups(&mut messages, recent_count);
            let originals = messages.clone();
            let mut protected = FxHashSet::default();
            if case == "protected" {
                protected.insert("call-write".to_string());
            }
            let sink = (case != "no_sink").then_some(overflow_dir.as_path());
            shrink_successful_write_arguments(&mut messages, sink, &protected);
            assert_eq!(
                messages, originals,
                "{tool_name} {case} must preserve all fields"
            );
            assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
            let _ = std::fs::remove_dir_all(overflow_dir);
        }
    }
}

#[test]
fn successful_write_recent_window_counts_parallel_groups() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![
        msg("system", "s"),
        msg("user", "do"),
        assistant_call_args("call-w", "write_file", &big_write_args()),
        tool_result("call-w", "Successfully wrote to /tmp/out.txt"),
    ];
    for i in 0..KEEP_RECENT_TOOL_GROUPS - 1 {
        let first_id = format!("parallel-{i}-a");
        let second_id = format!("parallel-{i}-b");
        let mut assistant = assistant_call_args(&first_id, "text_grep", r#"{"pattern":"x"}"#);
        assistant.tool_calls.as_mut().unwrap().push(
            assistant_call_args(&second_id, "text_grep", r#"{"pattern":"y"}"#)
                .tool_calls
                .unwrap()
                .remove(0),
        );
        messages.extend([
            assistant,
            tool_result(&first_id, "no match"),
            tool_result(&second_id, "no match"),
        ]);
    }
    let originals = messages.clone();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(
        messages, originals,
        "six results in three groups cannot age out the write"
    );

    messages.extend([
        assistant_call_args("fourth-group", "text_grep", r#"{"pattern":"z"}"#),
        tool_result("fourth-group", "no match"),
    ]);
    let recent_messages = messages[4..].to_vec();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    assert_call_pair_removed(&messages, "call-w");
    folded_write_note(&messages);
    let retained_raw = messages
        .iter()
        .filter(|message| message.role == "assistant" || message.role == "tool")
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        retained_raw, recent_messages,
        "the four protected groups remain verbatim"
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn successful_write_archive_failure_keeps_original_messages() {
    let dir = temp_overflow_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let blocker = dir.join("not-a-directory");
    std::fs::write(&blocker, "block archive creation").unwrap();
    let overflow_dir = blocker.join("overflow");
    let mut messages = vec![
        msg("system", "s"),
        msg("user", "do"),
        assistant_call_args("call-w", "write_file", &big_write_args()),
        tool_result("call-w", "Successfully wrote to /tmp/out.txt"),
    ];
    append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    let originals = messages.clone();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(
        messages, originals,
        "archive failure must not mutate any projection field"
    );
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn successful_write_mixed_batch_keeps_other_call_result_pairs() {
    let overflow_dir = temp_overflow_dir();
    let mut assistant = assistant_call_args("call-w", "write_file", &big_write_args());
    for (id, name, args) in [
        ("call-failed", "apply_patch", big_patch_args()),
        ("call-protected", "write_file", big_write_args()),
        (
            "call-read",
            "read_file",
            r#"{"file_path":"/tmp/a.rs"}"#.to_string(),
        ),
    ] {
        assistant.tool_calls.as_mut().unwrap().push(
            assistant_call_args(id, name, &args)
                .tool_calls
                .unwrap()
                .remove(0),
        );
    }
    assistant.content = Value::String("original mixed-batch narration".to_string());
    assistant.reasoning_content = Some("original mixed-batch reasoning".to_string());
    let successful_result = tool_result("call-w", "Successfully wrote to /tmp/out.txt");
    let surviving_results = vec![
        tool_result("call-protected", "Successfully wrote to /tmp/out.txt"),
        tool_result("call-read", "exact file contents"),
        tool_result(
            "call-failed",
            "Error: exact patch diagnostic\noriginal failure body",
        ),
    ];
    let mut remaining_assistant = assistant.clone();
    remaining_assistant.tool_calls.as_mut().unwrap().remove(0);
    let originals = vec![assistant.clone(), successful_result.clone()];
    let mut messages = vec![
        msg("system", "s"),
        msg("user", "do"),
        assistant,
        surviving_results[0].clone(),
        successful_result,
        surviving_results[1].clone(),
        surviving_results[2].clone(),
    ];
    append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    let mut protected = FxHashSet::default();
    protected.insert("call-protected".to_string());
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);
    assert_call_pair_removed(&messages, "call-w");
    assert!(
        messages.contains(&remaining_assistant),
        "surviving calls keep narration and reasoning"
    );
    for result in &surviving_results {
        assert!(
            messages.contains(result),
            "failed/protected/non-write results remain verbatim"
        );
    }
    let note = folded_write_note(&messages);
    let provenance = super::folded_prune::provenance(note).unwrap();
    assert_eq!(provenance.source_ids().collect::<Vec<_>>(), vec!["call-w"]);
    assert_messages_archived(&overflow_dir.join(OVERFLOW_HISTORY_FILENAME), &originals);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn successful_write_batch_folds_all_successful_calls_with_complete_provenance() {
    let overflow_dir = temp_overflow_dir();
    let mut assistant = assistant_call_args("call-w", "write_file", &big_write_args());
    assistant.tool_calls.as_mut().unwrap().push(
        assistant_call_args("call-p", "apply_patch", &big_patch_args())
            .tool_calls
            .unwrap()
            .remove(0),
    );
    let originals = vec![
        assistant,
        tool_result("call-p", "Successfully patched /tmp/a.rs; +1 -1 (1 lines)"),
        tool_result("call-w", "Successfully wrote to /tmp/out.txt"),
    ];
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    messages.extend(originals.clone());
    append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    let note = folded_write_note(&messages);
    assert!(
        note.content
            .as_str()
            .unwrap()
            .starts_with("compressed_tool_round: 2 tool calls")
    );
    let provenance = super::folded_prune::provenance(note).unwrap();
    assert_eq!(
        provenance.source_ids().collect::<Vec<_>>(),
        vec!["call-w", "call-p"]
    );
    assert_call_pair_removed(&messages, "call-w");
    assert_call_pair_removed(&messages, "call-p");
    assert_messages_archived(&overflow_dir.join(OVERFLOW_HISTORY_FILENAME), &originals);
    let archived = std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME)).unwrap();
    let patch_position = archived
        .find(&serde_json::to_string_pretty(&originals[1]).unwrap())
        .unwrap();
    let write_position = archived
        .find(&serde_json::to_string_pretty(&originals[2]).unwrap())
        .unwrap();
    assert!(
        patch_position < write_position,
        "archive preserves the observed result order"
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn successful_write_note_uses_resolved_targets_and_archives_full_patch_result() {
    for tool_name in ["write_file", "apply_patch"] {
        let overflow_dir = temp_overflow_dir();
        let (args, result, targets) = if tool_name == "write_file" {
            (
                big_write_args(),
                "Successfully wrote to /actual/resolved output.txt\nnote: exact hint".to_string(),
                vec!["/actual/resolved output.txt"],
            )
        } else {
            (
                big_patch_args(),
                format!(
                    "Successfully patched 2 files:\n- /actual/first; file.rs; +1 -1 (1 lines)\n- /actual/second.rs; +2 -0 (2 lines)\nnote: {}",
                    "exact precision evidence\n".repeat(200)
                ),
                vec!["/actual/first; file.rs", "/actual/second.rs"],
            )
        };
        let originals = vec![
            assistant_call_args("call-write", tool_name, &args),
            tool_result("call-write", &result),
        ];
        let mut messages = vec![msg("system", "s"), msg("user", "do")];
        messages.extend(originals.clone());
        append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
        shrink_successful_write_arguments(
            &mut messages,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        let text = folded_write_note(&messages).content.as_str().unwrap();
        assert!(
            text.contains(&format!("target_files: {}", serde_json::json!(targets))),
            "target files must come from success evidence, including every patched file"
        );
        assert_messages_archived(&overflow_dir.join(OVERFLOW_HISTORY_FILENAME), &originals);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }
}

#[test]
fn successful_write_reused_ids_do_not_borrow_later_success() {
    let overflow_dir = temp_overflow_dir();
    let failed_pair = vec![
        assistant_call_args("reused-id", "write_file", &big_write_args()),
        tool_result("reused-id", "Error: original write failed"),
    ];
    let successful_pair = vec![
        assistant_call_args("reused-id", "write_file", &big_write_args()),
        tool_result("reused-id", "Successfully wrote to /tmp/out.txt"),
    ];
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    messages.extend(failed_pair.clone());
    messages.extend(successful_pair.clone());
    append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    for message in failed_pair {
        assert!(
            messages.contains(&message),
            "an earlier failure must remain intact despite ID reuse"
        );
    }
    folded_write_note(&messages);
    assert_messages_archived(
        &overflow_dir.join(OVERFLOW_HISTORY_FILENAME),
        &successful_pair,
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn successful_write_duplicate_results_leave_group_unchanged() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![
        msg("system", "s"),
        msg("user", "do"),
        assistant_call_args("call-w", "write_file", &big_write_args()),
        tool_result("call-w", "Successfully wrote to /tmp/out.txt"),
        tool_result("call-w", "Error: conflicting duplicate result"),
    ];
    append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
    let originals = messages.clone();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &FxHashSet::default());
    assert_eq!(messages, originals, "ambiguous pairing must fail closed");
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn successful_write_legacy_pointers_fold_during_hard_budget_minimization() {
    for preview in [None, Some("legacy partial arguments ".repeat(200))] {
        let overflow_dir = temp_overflow_dir();
        let legacy_dir = overflow_dir.join("legacy");
        let original_pair = vec![
            assistant_call_args("call-w", "write_file", &big_write_args()),
            tool_result("call-w", "Successfully wrote to /tmp/out.txt"),
        ];
        let legacy_path = archive_messages_to_overflow(&original_pair, Some(&legacy_dir)).unwrap();
        let legacy_before = std::fs::read_to_string(&legacy_path).unwrap();
        let mut pointer = serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": big_write_args().chars().count(),
            "archive_file_path": legacy_path,
        });
        if let Some(preview) = preview {
            pointer["preview"] = Value::String(preview);
        }
        let archived_pair = vec![
            assistant_call_args("call-w", "write_file", &pointer.to_string()),
            original_pair[1].clone(),
        ];
        let mut messages = vec![msg("system", "s"), msg("user", "do")];
        messages.extend(archived_pair.clone());
        append_recent_groups(&mut messages, KEEP_RECENT_TOOL_GROUPS);
        minimize_truncated_previews_for_hard_budget(
            &mut messages,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        let text = folded_write_note(&messages).content.as_str().unwrap();
        assert!(text.contains(&format!(
            "original_arguments_archive_file_path: {legacy_path}"
        )));
        assert!(text.contains("/tmp/out.txt"));
        assert_call_pair_removed(&messages, "call-w");
        assert_messages_archived(
            &overflow_dir.join(OVERFLOW_HISTORY_FILENAME),
            &archived_pair,
        );
        assert_eq!(
            std::fs::read_to_string(&legacy_path).unwrap(),
            legacy_before
        );
        let _ = std::fs::remove_dir_all(overflow_dir);
    }
}

#[test]
fn successful_write_legacy_pointer_protection_keeps_pairs_unfolded() {
    for case in [
        "recent",
        "failed",
        "protected",
        "no_sink",
        "missing_archive",
    ] {
        let overflow_dir = temp_overflow_dir();
        let legacy_dir = overflow_dir.join("legacy");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_path = legacy_dir.join(OVERFLOW_HISTORY_FILENAME);
        if case != "missing_archive" {
            std::fs::write(&legacy_path, "original archived arguments").unwrap();
        }
        let pointer = serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": 5_000,
            "archive_file_path": legacy_path,
        })
        .to_string();
        let result = if case == "failed" {
            "Error: original failure"
        } else {
            "Successfully wrote to /tmp/out.txt"
        };
        let mut messages = vec![
            msg("system", "s"),
            msg("user", "do"),
            assistant_call_args("call-w", "write_file", &pointer),
            tool_result("call-w", result),
        ];
        append_recent_groups(
            &mut messages,
            if case == "recent" {
                KEEP_RECENT_TOOL_GROUPS - 1
            } else {
                KEEP_RECENT_TOOL_GROUPS
            },
        );
        let originals = messages.clone();
        let mut protected = FxHashSet::default();
        if case == "protected" {
            protected.insert("call-w".to_string());
        }
        minimize_truncated_previews_for_hard_budget(
            &mut messages,
            (case != "no_sink").then_some(overflow_dir.as_path()),
            &protected,
        );
        assert_eq!(messages, originals, "{case} legacy pairs must not fold");
        assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
        let _ = std::fs::remove_dir_all(overflow_dir);
    }
}
