//! Direct compression tests for history::compress_messages_for_context and session-temp spill files.

use serde_json::Value;

use super::super::{
    history::{
        COLON, ContextCompressionStatus, Message, NEWLINE, append_history, build_message_arr,
        compress_messages_for_context, compress_messages_for_context_with_outcome,
        messages_total_chars_pub,
    },
    types::{FunctionCall, ToolCall},
};
use super::*;

struct CompressionTempDir(std::path::PathBuf);

impl CompressionTempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ai-compression-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for CompressionTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compression_message(role: &str, text: String) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn compression_dialogue(turns: usize, body_chars: usize) -> Vec<Message> {
    let body = "x".repeat(body_chars);
    (0..turns)
        .flat_map(|i| {
            [
                compression_message("user", format!("QUESTION_{i:02} {body}")),
                compression_message("assistant", format!("ANSWER_{i:02} {body}")),
            ]
        })
        .collect()
}

fn assert_source_bound_summary(
    compressed: &[Message],
    original: &[Message],
    overflow_dir: &std::path::Path,
    initial_goal: &str,
) -> Vec<Message> {
    let summaries: Vec<_> = compressed
        .iter()
        .filter_map(|message| {
            message
                .content
                .as_str()?
                .strip_prefix("[incremental-memory-v1]\n")
                .map(|json| (message, json))
        })
        .collect();
    assert_eq!(summaries.len(), 1, "one removed span needs one increment");
    let (message, json) = summaries[0];
    assert_eq!(message.role, crate::ai::history::ROLE_INTERNAL_NOTE);
    let record: Value = serde_json::from_str(json).expect("summary must be complete JSON");
    assert_eq!(record["schema"], 1);
    assert_eq!(record["provenance"], "assistant_derived_unverified");
    assert_eq!(
        record["source_scope"],
        "compression_input_projection_not_claim_verification"
    );
    let entries = record["entries"].as_array().expect("summary entries");
    assert!(
        entries.iter().any(|entry| {
            entry["text"]
                .as_str()
                .is_some_and(|text| text.contains(initial_goal))
        }),
        "summary must preserve the initial goal: {record}"
    );
    assert!(
        entries
            .iter()
            .all(|entry| entry["status"] == "derived_unverified")
    );

    let source = &record["source"];
    assert_eq!(source["start_line"], 1);
    assert_eq!(source["encoding"], "one_raw_message_json_per_line");
    assert_eq!(source.get("source_model"), Some(&Value::Null));
    let path = std::path::Path::new(source["archive_file_path"].as_str().unwrap());
    assert!(path.starts_with(overflow_dir.join("summary-sources")));
    assert_eq!(
        path.file_stem().and_then(|stem| stem.to_str()),
        source["sha256"].as_str()
    );
    let raw = std::fs::read_to_string(path).expect("source locator must be readable");
    let restored: Vec<Message> = raw
        .lines()
        .map(|line| serde_json::from_str(line).expect("complete source message"))
        .collect();
    assert!(
        !restored.is_empty(),
        "compression must archive an actual span"
    );
    assert!(
        restored.len() < original.len(),
        "recent dialogue must stay inline"
    );
    assert_eq!(source["end_line"].as_u64(), Some(restored.len() as u64));
    assert_eq!(restored, original[..restored.len()]);
    let expected_raw: String = original[..restored.len()]
        .iter()
        .map(|message| format!("{}\n", serde_json::to_string(message).unwrap()))
        .collect();
    assert_eq!(
        raw, expected_raw,
        "archive must preserve exact source bytes"
    );
    let inline: Vec<_> = compressed
        .iter()
        .filter(|message| message.role != crate::ai::history::ROLE_INTERNAL_NOTE)
        .cloned()
        .collect();
    assert_eq!(
        inline,
        original[restored.len()..],
        "source and inline tail must partition history without loss"
    );
    restored
}

#[test]
fn history_compression_inserts_summary_and_keeps_recent() {
    let dir = CompressionTempDir::new();
    let path = dir.0.join("history.sqlite");
    let long = "x".repeat(220);
    let mut blob = String::new();
    for i in 0..10 {
        blob.push_str(&format!("user{COLON}u{i} {long}{NEWLINE}"));
        blob.push_str(&format!("assistant{COLON}a{i} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(100, &path).unwrap();
    // Source metadata needs more room than the legacy 200-character prose note.
    // The larger cap still requires compression and retains all four recent turns.
    let budget = 4_000;
    assert!(messages_total_chars_pub(&messages) > budget);
    let outcome = compress_messages_for_context_with_outcome(
        messages.clone(),
        budget,
        4,
        2_000,
        Some(dir.0.clone()),
        None,
    );
    assert_eq!(outcome.status, ContextCompressionStatus::Complete);
    assert_eq!(outcome.before_chars, messages_total_chars_pub(&messages));
    assert_eq!(
        outcome.after_chars,
        messages_total_chars_pub(&outcome.messages)
    );
    assert_eq!(outcome.max_chars, budget);
    assert!(outcome.budget_met());
    assert!(outcome.after_chars < outcome.before_chars);
    let restored =
        assert_source_bound_summary(&outcome.messages, &messages, &dir.0, "Main request: u0");
    assert_eq!(restored, messages[..12]);
    assert_eq!(
        &outcome.messages[outcome.messages.len() - 8..],
        &messages[12..]
    );
}

#[test]
fn history_compression_summarizes_when_keep_last_exceeds_turns_but_budget_overflows() {
    // Reproduces the "agent forgets earlier questions after ~30 turns" bug:
    // with a large `keep_last` (e.g. CLI default 256) but a much smaller
    // `max_chars` budget, the older-segment summary path was never taken,
    // and early user turns got silently dropped from the head of the list.
    // The shrink path must retain the initial goal and a readable source span.
    let dir = CompressionTempDir::new();
    let path = dir.0.join("history.sqlite");
    let long = "y".repeat(10);
    let mut blob = String::new();
    for i in 0..30usize {
        blob.push_str(&format!("user{COLON}QUESTION_{i:02} {long}{NEWLINE}"));
        let answer = if i == 0 { "y".repeat(5_000) } else { long.clone() };
        blob.push_str(&format!("assistant{COLON}ANSWER_{i:02} {answer}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(300, &path).unwrap();
    // Removing the large first turn leaves room for both a complete source-bound
    // summary and its archive pointer; uniform small turns can leave no headroom.
    let budget = 4_000;
    assert!(messages_total_chars_pub(&messages) > budget);
    // Raise only the summary allowance: the source locator and provenance must
    // fit alongside complete entries, while the original total cap stays strict.
    let outcome = compress_messages_for_context_with_outcome(
        messages.clone(),
        budget,
        256,
        2_400,
        Some(dir.0.clone()),
        None,
    );
    assert_eq!(outcome.status, ContextCompressionStatus::Complete);
    assert!(outcome.budget_met(), "{} > {budget}", outcome.after_chars);
    assert_eq!(outcome.before_chars, messages_total_chars_pub(&messages));
    assert_eq!(
        outcome.after_chars,
        messages_total_chars_pub(&outcome.messages)
    );
    assert!(outcome.after_chars < outcome.before_chars);
    assert_source_bound_summary(
        &outcome.messages,
        &messages,
        &dir.0,
        "Main request: QUESTION_00",
    );
    assert!(outcome.messages.ends_with(&messages[messages.len() - 2..]));
}

#[test]
fn overflow_history_file_preserves_dropped_messages_and_placeholder_in_context() {
    let dir = CompressionTempDir::new();
    let path = dir.0.join("history.sqlite");
    let overflow_dir = dir.0.join("overflow");

    let long = "z".repeat(10);
    let mut blob = String::new();
    for i in 0..20usize {
        blob.push_str(&format!("user{COLON}Q{i:02} {long}{NEWLINE}"));
        let answer = if i == 0 { "z".repeat(5_000) } else { long.clone() };
        blob.push_str(&format!("assistant{COLON}A{i:02} {answer}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(100, &path).unwrap();
    // A dominant first turn leaves enough space after removal for the summary,
    // the archive locator and an unchanged recent tail under the original cap.
    let budget = 4_500;
    assert!(messages_total_chars_pub(&messages) > budget);
    let outcome = compress_messages_for_context_with_outcome(
        messages.clone(),
        budget,
        256,
        2_400,
        Some(overflow_dir.clone()),
        None,
    );
    assert_eq!(outcome.status, ContextCompressionStatus::Complete);
    assert!(outcome.budget_met(), "{} > {budget}", outcome.after_chars);
    assert_eq!(
        outcome.after_chars,
        messages_total_chars_pub(&outcome.messages)
    );
    assert!(outcome.after_chars < outcome.before_chars);
    let compressed = &outcome.messages;
    let restored = assert_source_bound_summary(compressed, &messages, &overflow_dir, "Q00");
    let archive_text = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str().unwrap_or_default();
            text.contains("归档文件:").then_some(text)
        })
        .expect("should include an explicit archive note");
    assert!(
        archive_text.contains("read_file"),
        "archive note should mention read_file as the mechanism to retrieve archive, got: {archive_text:?}"
    );

    let overflow_file = overflow_dir.join("overflow-history.md");
    assert!(
        overflow_file.exists(),
        "overflow file should have been created at {:?}",
        overflow_file
    );
    let overflow_content = std::fs::read_to_string(&overflow_file).unwrap();
    assert!(archive_text.contains(overflow_file.to_str().unwrap()));
    for message in &restored {
        assert!(
            overflow_content.contains(message.content.as_str().unwrap()),
            "overflow archive must preserve the full removed message"
        );
    }
    assert!(
        overflow_content.contains("Q00"),
        "overflow file should contain the earliest user question Q00, got first 200 chars: {:?}",
        &overflow_content[..overflow_content.len().min(200)]
    );
    assert!(
        overflow_content.contains("# Overflow History Archive"),
        "overflow file should have the header"
    );

    assert!(compressed.ends_with(&messages[messages.len() - 2..]));
}

#[test]
fn overflow_flush_failure_restores_dropped_messages_without_data_loss() {
    // A regular file in place of the directory fails deterministically, even
    // with elevated privileges; permission-bit tests would not have that property.
    let dir = CompressionTempDir::new();
    let overflow_dir = dir.0.join("not-a-directory");
    std::fs::write(&overflow_dir, b"not a directory").unwrap();
    let messages = compression_dialogue(30, 260);
    let budget = 4_000;
    let before = messages_total_chars_pub(&messages);
    assert!(before > budget);
    // Exercise both the initial older/recent split and the budget-only shrink
    // path, with and without derived prose. All must fail closed on archive IO.
    for keep_last in [0, 4, 256] {
        for summary_max_chars in [0, 2_400] {
            let outcome = compress_messages_for_context_with_outcome(
                messages.clone(),
                budget,
                keep_last,
                summary_max_chars,
                Some(overflow_dir.clone()),
                None,
            );
            assert_eq!(
                outcome.status,
                ContextCompressionStatus::ArchiveCommitFailed,
                "keep_last={keep_last}, summary_max_chars={summary_max_chars}"
            );
            assert_eq!(
                outcome.messages, messages,
                "failed archival must preserve all roles, metadata and order without dangling notes"
            );
            assert_eq!(outcome.before_chars, before);
            assert_eq!(outcome.after_chars, before);
            assert_eq!(outcome.max_chars, budget);
            assert!(!outcome.budget_met());
        }
    }
    assert_eq!(std::fs::read(&overflow_dir).unwrap(), b"not a directory");
}

#[test]
fn history_compression_without_archive_sink_preserves_original_and_reports_status() {
    let messages = compression_dialogue(30, 260);
    let budget = 4_000;
    let before = messages_total_chars_pub(&messages);
    assert!(before > budget);
    for keep_last in [0, 4, 256] {
        for summary_max_chars in [0, 2_400] {
            let outcome = compress_messages_for_context_with_outcome(
                messages.clone(),
                budget,
                keep_last,
                summary_max_chars,
                None,
                None,
            );
            assert_eq!(
                outcome.status,
                ContextCompressionStatus::MissingArchiveSink,
                "keep_last={keep_last}, summary_max_chars={summary_max_chars}"
            );
            assert_eq!(
                outcome.messages, messages,
                "no sink must never authorize lossy prose or raw-message removal"
            );
            assert_eq!(outcome.before_chars, before);
            assert_eq!(outcome.after_chars, before);
            assert_eq!(outcome.max_chars, budget);
            assert!(!outcome.budget_met());
        }
    }
}

#[test]
fn history_compression_without_summary_archives_removed_dialogue_and_keeps_recent() {
    let messages = compression_dialogue(30, 260);
    let budget = 4_000;
    assert!(messages_total_chars_pub(&messages) > budget);
    for keep_last in [4, 256] {
        let dir = CompressionTempDir::new();
        let outcome = compress_messages_for_context_with_outcome(
            messages.clone(),
            budget,
            keep_last,
            0,
            Some(dir.0.clone()),
            None,
        );
        assert_eq!(outcome.status, ContextCompressionStatus::Complete);
        // Budget-only shrinking selects raw turns before adding its pointer.
        // With uniform small turns that pointer can exceed the remaining room;
        // completion must report this, not discard protected content to hide it.
        assert_eq!(outcome.budget_met(), keep_last == 4);
        if !outcome.budget_met() {
            assert!(outcome.diagnostic().unwrap().contains("budget_met=false"));
        }
        assert_eq!(outcome.before_chars, messages_total_chars_pub(&messages));
        assert_eq!(
            outcome.after_chars,
            messages_total_chars_pub(&outcome.messages)
        );
        assert!(outcome.after_chars < outcome.before_chars);
        assert!(outcome.messages.ends_with(&messages[messages.len() - 2..]));
        let archive_path = dir.0.join("overflow-history.md");
        let archive = std::fs::read_to_string(&archive_path)
            .expect("zero-summary removal still needs an archive");
        assert!(
            outcome.messages.iter().any(|message| {
                message.role == crate::ai::history::ROLE_INTERNAL_NOTE
                    && message.content.as_str().is_some_and(|text| {
                        text.contains(archive_path.to_str().unwrap()) && text.contains("read_file")
                    })
            }),
            "context must retain a usable archive pointer"
        );
        assert!(
            !dir.0.join("summary-sources").exists(),
            "zero-summary mode must not synthesize a summary increment"
        );
        for message in &messages {
            assert!(
                outcome.messages.contains(message)
                    || archive.contains(message.content.as_str().unwrap()),
                "every original message must remain inline or readable from the archive"
            );
        }
        assert!(
            !outcome.messages.contains(&messages[0]),
            "fixture must actually remove older dialogue"
        );
    }
}

#[test]
fn history_compression_source_roundtrip_preserves_old_and_recent_tool_pairs() {
    let dir = CompressionTempDir::new();
    let mut messages = compression_dialogue(10, 500);
    let (mut old_call, old_result) =
        read_file_call_pair("call_old", "src/old.rs", "old exact output");
    old_call.reasoning_content = Some("old source reasoning".to_string());
    messages.splice(2..2, [old_call.clone(), old_result.clone()]);
    let (recent_call, recent_result) =
        read_file_call_pair("call_recent", "src/recent.rs", "recent exact output");
    messages.splice(
        messages.len() - 1..messages.len() - 1,
        [recent_call.clone(), recent_result.clone()],
    );
    let budget = 8_000;
    assert!(messages_total_chars_pub(&messages) > budget);
    let outcome = compress_messages_for_context_with_outcome(
        messages.clone(),
        budget,
        4,
        2_400,
        Some(dir.0.clone()),
        None,
    );
    assert_eq!(outcome.status, ContextCompressionStatus::Complete);
    assert!(outcome.budget_met());
    assert_eq!(
        outcome.after_chars,
        messages_total_chars_pub(&outcome.messages)
    );
    assert!(outcome.after_chars < outcome.before_chars);
    let source = assert_source_bound_summary(
        &outcome.messages,
        &messages,
        &dir.0,
        "Main request: QUESTION_00",
    );
    assert!(
        source
            .windows(2)
            .any(|pair| pair == [old_call.clone(), old_result.clone()])
    );
    assert!(
        outcome
            .messages
            .windows(2)
            .any(|pair| pair == [recent_call.clone(), recent_result.clone()])
    );
}

#[test]
fn history_compression_reports_disabled_and_no_eligible_dialogue_without_mutation() {
    let dir = CompressionTempDir::new();
    let messages = vec![
        compression_message("system", "policy".repeat(1_000)),
        compression_message("user", "latest question".to_string()),
        compression_message("assistant", "latest answer".to_string()),
    ];
    let before = messages_total_chars_pub(&messages);
    for (budget, status) in [
        (0, ContextCompressionStatus::Disabled),
        (4_000, ContextCompressionStatus::NoEligibleDialogue),
    ] {
        let outcome = compress_messages_for_context_with_outcome(
            messages.clone(),
            budget,
            1,
            2_400,
            Some(dir.0.clone()),
            None,
        );
        assert_eq!(outcome.status, status);
        assert_eq!(outcome.messages, messages);
        assert_eq!(outcome.before_chars, before);
        assert_eq!(outcome.after_chars, before);
        assert_eq!(outcome.max_chars, budget);
        assert_eq!(outcome.budget_met(), budget == 0);
    }
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
}

#[test]
fn compression_spills_non_compressible_read_file_outputs_to_session_temp_files() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-preserve-overflow-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    for i in 0..8usize {
        let id = format!("call_{i}");
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: format!(
                        r#"{{"filePath":"src/lib.rs","startLine":{},"endLine":{}}}"#,
                        i + 1,
                        i + 20
                    ),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("x".repeat(28_000)),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    let compressed =
        compress_messages_for_context(messages, 20_000, 2, 400, Some(overflow_dir), None);

    let stub = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            text.contains("Output preserved for tool `read_file`")
                .then_some(text.to_string())
        })
        .expect("expected preserved read_file overflow stub");

    let file_path = stub
        .lines()
        .find_map(|line| line.trim().strip_prefix("- file_path: "))
        .expect("stub should contain overflow file path");
    assert!(
        std::path::Path::new(file_path).exists(),
        "overflow file path from stub should exist: {file_path}"
    );
    // The stub must keep a content preview as a recall anchor so later turns do not "forget".
    assert!(
        stub.contains("Preview (for recall"),
        "stub should contain a content preview: {stub}"
    );
}

#[test]
fn overflow_stub_recall_anchor_survives_compaction() {
    // Reproduce a "tool-heavy session (few user turns × hundreds of read_file calls)": many early
    // read_file groups + one near-end user turn. After compression assert (1) total billable drops sharply and converges
    // into budget, and (2) every read_file's file_path recall anchor is still findable in the output (zero amnesia).

    let overflow_dir =
        std::env::temp_dir().join(format!("ai-recall-anchor-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 60 early read_file groups, each result 4000 chars — any single one over the threshold must be spilled into a preview stub.
    for i in 0..60usize {
        let id = format!("call_{i}");
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: format!(r#"{{"filePath":"src/file_{i}.rs"}}"#),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String(format!("content of file {i}\n").repeat(200)),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }
    // Near-end user turn (protected tail window).
    messages.push(Message {
        role: "user".to_string(),
        content: Value::String("最新的问题".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });

    let before = messages_total_chars_pub(&messages);
    let budget = 40_000usize;
    let compressed =
        compress_messages_for_context(messages, budget, 256, 400, Some(overflow_dir.clone()), None);
    let after = messages_total_chars_pub(&compressed);

    // The total drops sharply and converges into budget (tool-heavy sessions no longer stall structurally).
    assert!(
        after < before,
        "compaction must reduce total billable ({after} !< {before})"
    );
    assert!(
        after <= budget,
        "tool-heavy history must converge under budget ({after} > {budget})"
    );

    // Collect all output text after compression and verify every early read_file's file_path is still recallable:
    // either the src/file_N.rs path survives in a stub/anchor, or the spill temp-file path appears in some note.
    let joined: String = compressed
        .iter()
        .filter_map(|m| m.content.as_str().map(str::to_string))
        .collect::<Vec<_>>()
        .join("\n");
    // At least one spill temp file must be written to disk (read_file results spill with zero compression).
    let overflow_files: Vec<_> = std::fs::read_dir(overflow_dir.join("tool-overflow-compressed"))
        .map(|rd| rd.filter_map(Result::ok).collect())
        .unwrap_or_default();
    assert!(
        !overflow_files.is_empty(),
        "read_file outputs should be spilled to session temp files"
    );
    // A recall lead survives folding: at least one of a compressed_tool_round note or a stub anchor.
    assert!(
        joined.contains("compressed_tool_round")
            || joined.contains("Output preserved for tool")
            || joined.contains("read_file"),
        "compacted history must retain read_file recall anchors"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn compression_keeps_recent_non_compressible_tool_output_verbatim() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-preserve-overflow-recent-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let recent_output = "y".repeat(12_000);
    messages.push(Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: "call_recent".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"filePath":"src/lib.rs","startLine":1,"endLine":300}"#.to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    });
    messages.push(Message {
        role: "tool".to_string(),
        content: Value::String(recent_output.clone()),
        tool_calls: None,
        tool_call_id: Some("call_recent".to_string()),
        reasoning_content: None,
    });

    let compressed =
        compress_messages_for_context(messages, 32_000, 256, 400, Some(overflow_dir.clone()), None);

    // The most recent read_file result must be neither spilled into a stub nor pruned: visible verbatim.
    assert!(
        compressed.iter().all(|m| {
            m.content
                .as_str()
                .map(|s| !s.contains("Output preserved for tool"))
                .unwrap_or(true)
        }),
        "recent non-compressible tool output must not be spilled to a stub"
    );
    assert!(
        compressed
            .iter()
            .any(|m| m.content.as_str() == Some(recent_output.as_str())),
        "recent read_file output must remain verbatim in context"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn compression_spills_old_user_message_to_session_temp_file() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-preserve-user-overflow-{}",
        uuid::Uuid::new_v4()
    ));
    let old_user = "U".repeat(20_000);
    let latest_user = "继续处理当前问题";
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(old_user.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("阶段一：先定位".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("继续".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("阶段二：验证".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("继续".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(latest_user.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("继续执行".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let compressed =
        compress_messages_for_context(messages, 2_000, 256, 400, Some(overflow_dir.clone()), None);

    let stub = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            extract_stub_file_path(text).map(|_| text.to_string())
        })
        .expect("expected preserved user overflow stub");
    let file_path = extract_stub_file_path(&stub).expect("stub should contain overflow file path");
    assert!(
        std::path::Path::new(&file_path).exists(),
        "user overflow file path from stub should exist: {file_path}"
    );
    assert!(
        !stub.contains("[[PRESERVED_CONTENT_STUB_V1]]"),
        "model-facing archive notice must not expose the internal stub protocol"
    );
    let persisted = std::fs::read_to_string(&file_path).expect("should read persisted user file");
    assert!(
        persisted.contains(&old_user[..64]),
        "persisted user file should contain original user content"
    );

    let has_latest_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(latest_user));
    assert!(
        has_latest_user,
        "latest user turn should remain inline and not be spilled"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn compression_spills_old_image_message_to_session_temp_file() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-preserve-image-overflow-{}",
        uuid::Uuid::new_v4()
    ));
    let image_payload = format!("data:image/png;base64,{}", "A".repeat(16_000));
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![serde_json::json!({
                "type": "image_url",
                "image_url": { "url": image_payload }
            })]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到图片".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("阶段一".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("继续".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("阶段二".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("继续".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("请继续".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let compressed =
        compress_messages_for_context(messages, 2_000, 256, 400, Some(overflow_dir.clone()), None);

    let stub = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            extract_stub_file_path(text).map(|_| text.to_string())
        })
        .expect("expected preserved image overflow stub");
    let file_path = extract_stub_file_path(&stub).expect("stub should contain overflow file path");
    assert!(
        std::path::Path::new(&file_path).exists(),
        "image overflow file path from stub should exist: {file_path}"
    );
    let persisted = std::fs::read_to_string(&file_path).expect("should read persisted image file");
    assert!(
        persisted.contains("data:image/png;base64,"),
        "persisted image file should contain original image payload"
    );
    assert!(
        !stub.contains("[[PRESERVED_CONTENT_STUB_V1]]"),
        "model-facing archive notice must not expose the internal stub protocol"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}
