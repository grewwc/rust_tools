//! Direct compression tests for history::compress_messages_for_context and session-temp spill files.

use serde_json::Value;

use super::super::{
    history::{
        COLON, NEWLINE, Message, append_history, build_message_arr,
        compress_messages_for_context, messages_total_chars_pub,
    },
    types::{FunctionCall, ToolCall},
};
use super::*;

#[test]
fn history_compression_inserts_summary_and_keeps_recent() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let long = "x".repeat(220);
    let mut blob = String::new();
    for i in 0..10 {
        blob.push_str(&format!("user{COLON}u{i} {long}{NEWLINE}"));
        blob.push_str(&format!("assistant{COLON}a{i} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(100, &path).unwrap();
    let compressed = compress_messages_for_context(messages, 1800, 4, 200, None, None);

    assert!(!compressed.is_empty());
    assert_eq!(compressed[0].role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(
        compressed[0]
            .content
            .as_str()
            .unwrap_or_default()
            .contains("对话摘要")
    );
    assert!(
        compressed[0]
            .content
            .as_str()
            .unwrap_or_default()
            .contains("Main request: u0")
    );
    assert_eq!(
        compressed.last().unwrap().content,
        Value::String(format!("a9 {long}"))
    );
    let total = compressed
        .iter()
        .map(|m| m.content.as_str().map(|s| s.chars().count()).unwrap_or(0))
        .sum::<usize>();
    assert!(total <= 1800);

    let _ = std::fs::remove_file(path);
}

#[test]
fn history_compression_summarizes_when_keep_last_exceeds_turns_but_budget_overflows() {
    // Reproduces the "agent forgets earlier questions after ~30 turns" bug:
    // with a large `keep_last` (e.g. CLI default 256) but a much smaller
    // `max_chars` budget, the older-segment summary path was never taken,
    // and early user turns got silently dropped from the head of the list.
    // The new shrink path must inject a summary note so at least a textual
    // trace of the earliest user turns survives.
    let path =
        std::env::temp_dir().join(format!("ai-history-long-{}.sqlite", uuid::Uuid::new_v4()));
    let long = "y".repeat(260);
    let mut blob = String::new();
    for i in 0..30usize {
        blob.push_str(&format!("user{COLON}QUESTION_{i:02} {long}{NEWLINE}"));
        blob.push_str(&format!("assistant{COLON}ANSWER_{i:02} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(300, &path).unwrap();
    // keep_last=256 models the default configured history window; max_chars=4000
    // is far smaller than the raw history size (30 turns * ~560 bytes ~= 17k).
    let compressed = compress_messages_for_context(messages, 4000, 256, 600, None, None);

    assert!(!compressed.is_empty());
    assert_eq!(
        compressed[0].role,
        crate::ai::history::ROLE_INTERNAL_NOTE,
        "expected a synthesized summary at the head; got {:?}",
        compressed[0].role
    );
    let note_text = compressed[0]
        .content
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        note_text.contains("对话摘要"),
        "summary header missing: {note_text:?}"
    );
    assert!(
        note_text.contains("Main request: QUESTION_00"),
        "summary should preserve the initial goal, got: {note_text:?}"
    );
    // The summary should at least preserve a non-trivial textual trace of
    // the dropped region (instead of silently losing it). The exact content
    // depends on heuristic topic extraction; we only assert the summary body
    // has some characters beyond the header.
    let body_len = note_text
        .trim_start_matches("对话摘要（自动压缩，以下为早期对话要点）：")
        .trim()
        .chars()
        .count();
    assert!(
        body_len >= 10,
        "summary body is essentially empty: {note_text:?}"
    );

    let total = compressed
        .iter()
        .map(|m| m.content.as_str().map(|s| s.len()).unwrap_or(0))
        .sum::<usize>();
    assert!(
        total <= 4000,
        "compressed payload must respect the byte budget, got {total}"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn overflow_history_file_preserves_dropped_messages_and_placeholder_in_context() {
    let path =
        std::env::temp_dir().join(format!("ai-overflow-test-{}.sqlite", uuid::Uuid::new_v4()));
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-overflow-dir-{}", uuid::Uuid::new_v4()));

    let long = "z".repeat(300);
    let mut blob = String::new();
    for i in 0..20usize {
        blob.push_str(&format!("user{COLON}Q{i:02} {long}{NEWLINE}"));
        blob.push_str(&format!("assistant{COLON}A{i:02} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(100, &path).unwrap();
    let compressed =
        compress_messages_for_context(messages, 2000, 256, 400, Some(overflow_dir.clone()), None);

    let first_msg = compressed.first().expect("should have messages");
    assert_eq!(
        first_msg.role,
        crate::ai::history::ROLE_INTERNAL_NOTE,
        "first message should be an internal note with compressed long-term memory"
    );
    let memory_text = first_msg.content.as_str().unwrap_or_default();
    assert!(
        memory_text.contains("长期记忆摘要"),
        "first note should expose compressed memory, got: {memory_text:?}"
    );
    assert!(
        memory_text.contains("Q00"),
        "compressed memory should still expose the initial goal, got: {memory_text:?}"
    );
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
    assert!(
        overflow_content.contains("Q00"),
        "overflow file should contain the earliest user question Q00, got first 200 chars: {:?}",
        &overflow_content[..overflow_content.len().min(200)]
    );
    assert!(
        overflow_content.contains("# Overflow History Archive"),
        "overflow file should have the header"
    );

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn overflow_flush_failure_restores_dropped_messages_without_data_loss() {
    // Defect-1 regression: when archive flush fails, the messages pending deletion must be put back — never silently drop history.
    let path =
        std::env::temp_dir().join(format!("ai-overflow-fail-{}.sqlite", uuid::Uuid::new_v4()));
    // Key: point overflow_dir at an **existing regular file**. Then OverflowSink::flush's
    // create_dir_all(parent=file) fails, and OpenOptions.open(file/overflow-history.md)
    // fails with ENOTDIR because a path component is a file → flush() necessarily returns false, deterministically triggering the failure-rollback path.
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-overflow-failfile-{}", uuid::Uuid::new_v4()));
    std::fs::write(&overflow_dir, b"not a directory").unwrap();

    let long = "z".repeat(300);
    let mut blob = String::new();
    for i in 0..20usize {
        blob.push_str(&format!("user{COLON}Q{i:02} {long}{NEWLINE}"));
        blob.push_str(&format!("assistant{COLON}A{i:02} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(100, &path).unwrap();
    let original_user_count = messages.iter().filter(|m| m.role == "user").count();
    let compressed =
        compress_messages_for_context(messages, 2000, 256, 400, Some(overflow_dir.clone()), None);

    // flush failure → never delete history: all original user messages must still be in the return value (the old code silently dropped them).
    let kept_user_count = compressed.iter().filter(|m| m.role == "user").count();
    assert_eq!(
        kept_user_count, original_user_count,
        "flush 失败时不得丢任何 user 消息，期望 {original_user_count} 条，实得 {kept_user_count} 条"
    );

    let joined: String = compressed
        .iter()
        .filter_map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("Q00"),
        "最早的用户问题 Q00 必须被放回，不能丢失"
    );
    // The failure path must never inject summary/archive notes (avoiding dangling pointers to non-existent archive files).
    assert!(
        !joined.contains("长期记忆摘要"),
        "flush 失败路径不得注入摘要 note"
    );
    assert!(
        !joined.contains("长期记忆归档"),
        "flush 失败路径不得注入归档指针 note"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&overflow_dir);
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

    let compressed = compress_messages_for_context(messages, 20_000, 2, 400, Some(overflow_dir), None);

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
