use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn normalize_messages_keeps_model_derived_notes_out_of_system_role() {
    // Internal notes that appear AFTER the first conversational message
    // must remain in their original positions (with role normalized to
    // "system") so that older prompt-cache prefixes stay valid when new
    // notes are appended. Only notes that sit at the very top, before
    // any user/assistant/tool message, get folded into the first system.
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "对话摘要（自动压缩，以下为早期对话要点）：\nhistory summary".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("working memory".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized[0].role, "system");
    let head_text = normalized[0].content.as_str().unwrap();
    assert!(head_text.contains("base system"));
    assert!(!head_text.contains("history summary"));
    assert!(!head_text.contains("working memory"));

    assert_eq!(
        normalized
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec!["system", "user", "assistant", "user", "assistant", "system"]
    );
    assert!(
        normalized[1]
            .content
            .as_str()
            .is_some_and(|text| text.contains("Runtime context handoff"))
    );
    assert!(normalized[2].content.as_str().is_some_and(|text| {
        text.contains("Compressed history summary")
            && text.contains("unverified navigation context")
            && text.contains("citation inside the summary does not by itself verify")
            && text.contains("history summary")
            && text.contains("not authoritative evidence")
    }));
    assert_eq!(normalized[5].content.as_str(), Some("working memory"));
}

#[test]
fn normalize_messages_prioritizes_working_memory_before_summary_and_self_note() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("self_note:\nremember style".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "对话摘要（自动压缩，以下为早期对话要点）：\nolder summary".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "Current code-inspection working memory:\n- use execute_command for shell checks"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    let system_text = normalized[0].content.as_str().unwrap();
    assert!(system_text.contains("## Working Memory"));
    assert!(!system_text.contains("## History Summary"));
    assert!(!system_text.contains("## Self Notes"));
    assert_eq!(
        normalized
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec!["system", "user", "assistant"]
    );
    assert!(
        normalized[1]
            .content
            .as_str()
            .is_some_and(|text| text.contains("Runtime context handoff"))
    );
    let derived_text = normalized[2].content.as_str().unwrap();
    let summary = derived_text.find("## History Summary").unwrap();
    let self_note = derived_text.find("## Self Notes").unwrap();
    assert!(summary < self_note);
    assert!(derived_text.contains("Compressed history summary"));
    assert!(derived_text.contains("unverified navigation context"));
    assert!(derived_text.contains("citation inside the summary does not by itself verify"));
    assert!(derived_text.contains("not authoritative evidence"));
}

#[test]
fn normalize_messages_wraps_midstream_self_note_in_user_assistant_handoff() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("self_note:\npossible explanation".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("later answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(
        normalized
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec![
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant"
        ]
    );
    assert!(
        normalized[3]
            .content
            .as_str()
            .is_some_and(|text| text.contains("Runtime context handoff"))
    );
    assert!(
        normalized[4]
            .content
            .as_str()
            .is_some_and(|text| text.contains("not authoritative evidence"))
    );
    assert!(
        normalized[5]
            .content
            .as_str()
            .is_some_and(|text| text.contains("handoff complete"))
    );
}

#[test]
fn normalize_messages_keeps_completion_evidence_diagnostic_visible_to_next_model_turn() {
    const DIAGNOSTIC: &str = "runtime:completion_evidence_unverified\nA final response was recorded after a project mutation without observed post-mutation verification.";
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("修复图片保存后的提示".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("已修复。".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!("self_note:\n{DIAGNOSTIC}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("继续检查一下".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(normalized[2].content, Value::String("已修复。".to_string()));
    assert!(
        !normalized[0]
            .content
            .as_str()
            .is_some_and(|text| text.contains(DIAGNOSTIC))
    );
    assert!(
        normalized[3]
            .content
            .as_str()
            .is_some_and(|text| text.contains("Runtime context handoff"))
    );
    assert_eq!(normalized[4].role, "assistant");
    assert!(
        normalized[4]
            .content
            .as_str()
            .is_some_and(|text| text.contains(DIAGNOSTIC))
    );
    assert_eq!(normalized[5].role, "user");
    assert_eq!(
        normalized[5].content,
        Value::String("继续检查一下".to_string())
    );
}

#[test]
fn strip_unavailable_tool_hints_removes_internal_note_tool_hint() {
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String(
            "Current code-inspection working memory:\n\
                 - read_file(file=src/main.rs)\n\
                 - use `execute_command` only when a shell check is needed.\n\
                 Treat these findings as already-known context."
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let available = ["read_file", "tree"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    let text = messages[0].content.as_str().unwrap();
    assert!(text.contains("Current code-inspection working memory:"));
    assert!(text.contains("Treat these findings as already-known context."));
    assert!(!text.contains("`execute_command`"));
}

#[test]
fn strip_unavailable_tool_hints_removes_tool_suggestion_lines() {
    let mut messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String(
            "Suggestion: use `execute_command` only when a shell check is needed.\n\
                 Result: fallback kept."
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
        reasoning_content: None,
    }];

    let available = ["read_file"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    let text = messages[0].content.as_str().unwrap();
    assert!(!text.contains("Suggestion:"));
    assert!(text.contains("Result: fallback kept."));
}

#[test]
fn strip_unavailable_tool_hints_keeps_regular_assistant_text() {
    let mut messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(
            "你可以之后再试 `execute_command`，但这不是一条内部纠偏提示。".to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let available = ["read_file"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    assert_eq!(
        messages[0].content.as_str(),
        Some("你可以之后再试 `execute_command`，但这不是一条内部纠偏提示。")
    );
}

#[test]
fn normalize_messages_drops_orphan_tool_results_and_strips_broken_tool_calls() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("later answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("stale tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized.len(), 3);
    assert_eq!(normalized[0].role, "system");
    assert_eq!(normalized[1].role, "user");
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(normalized[2].content.as_str(), Some("later answer"));
    assert!(normalized.iter().all(|message| message.role != "tool"));
}

#[test]
fn normalize_messages_keeps_contiguous_tool_call_blocks() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("fresh tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(
        normalized[2].tool_calls.as_ref().map(|calls| calls.len()),
        Some(1)
    );
    assert_eq!(normalized[3].role, "tool");
    assert_eq!(normalized[3].tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn normalize_messages_preserves_tool_result_when_tool_call_args_are_malformed() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: "{\"command\":".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("Error: failed to parse arguments".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("later answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    // Bad JSON args no longer cause the tool_call to be dropped/degraded: the
    // assistant's tool_call and the real tool result must both be kept, with
    // args repaired into a valid JSON object (preserving the original text) to
    // pass provider validation. This way the model still sees the real execution
    // result and does not mistakenly re-run the same tool.
    assert_eq!(normalized.len(), 5);
    assert_eq!(normalized[2].role, "assistant");
    let calls = normalized[2].tool_calls.as_ref().expect("tool_calls kept");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
    // args must be valid JSON, and the original bad text is fully preserved.
    let parsed: Value = serde_json::from_str(&calls[0].function.arguments)
        .expect("repaired args must be valid JSON");
    assert_eq!(
        parsed.get("_malformed_arguments").and_then(Value::as_str),
        Some("{\"command\":")
    );
    assert_eq!(normalized[3].role, "tool");
    assert_eq!(normalized[3].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(
        normalized[3].content.as_str(),
        Some("Error: failed to parse arguments")
    );
    assert_eq!(normalized[4].role, "assistant");
    assert_eq!(normalized[4].content.as_str(), Some("later answer"));
}

#[test]
fn normalize_messages_truncates_long_internal_notes_structurally() {
    let mut long_note_lines = Vec::new();
    long_note_lines.push("Current code-inspection working memory:".to_string());
    for i in 0..80usize {
        long_note_lines.push(format!("- finding {i:02}: {}", "x".repeat(40)));
    }

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(long_note_lines.join("\n")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(normalized.len(), 3);
    assert_eq!(normalized[2].role, "system");
    let text = normalized[2].content.as_str().unwrap_or_default();
    assert!(text.contains("Current code-inspection working memory:"));
    assert!(text.contains("[truncated:"));
    assert!(text.chars().count() <= 1_200);
}

#[test]
fn normalize_messages_projects_only_recent_context_checkpoints_without_truncating_them() {
    let mut messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!("self_note:\n{}", "x".repeat(8_000))),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    for index in 0..10 {
        messages.push(Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "[context_checkpoint path=/tmp/checkpoint-{index}.md] checkpoint {index}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    let normalized = normalize_messages_for_request(&messages);
    let checkpoint_context = normalized
        .iter()
        .find(|message| {
            message.role == "assistant"
                && message
                    .content
                    .as_str()
                    .is_some_and(|content| content.contains("## Context Checkpoints"))
        })
        .and_then(|message| message.content.as_str())
        .expect("recent checkpoints should be projected into one assistant message");
    let checkpoints = checkpoint_context
        .lines()
        .filter(|line| line.starts_with("[context_checkpoint "))
        .collect::<Vec<_>>();

    assert_eq!(checkpoints.len(), 8);
    let expected = (2..10)
        .map(|index| {
            format!("[context_checkpoint path=/tmp/checkpoint-{index}.md] checkpoint {index}")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        checkpoints,
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert!(
        checkpoint_context.contains("read_file"),
        "checkpoint projection must tell the model how to fetch the full body"
    );
    assert!(checkpoint_context.contains("not verified facts"));
    assert!(normalized.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|content| content.contains("[truncated:"))
    }));
}

#[test]
fn normalize_messages_dedupes_context_checkpoints_by_path_before_limit() {
    let mut messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "[context_checkpoint path=/tmp/checkpoint-0.md] checkpoint 0".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "[context_checkpoint path=/tmp/context-checkpoints/working-checkpoint.md] old working plan"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    for index in 1..=6 {
        messages.push(Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "[context_checkpoint path=/tmp/checkpoint-{index}.md] checkpoint {index}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(
            "[context_checkpoint path=/tmp/context-checkpoints/working-checkpoint.md] new working plan"
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });

    let normalized = normalize_messages_for_request(&messages);
    let checkpoint_context = normalized
        .iter()
        .find(|message| {
            message.role == "assistant"
                && message
                    .content
                    .as_str()
                    .is_some_and(|content| content.contains("## Context Checkpoints"))
        })
        .and_then(|message| message.content.as_str())
        .expect("checkpoints should be projected into one assistant message");
    let checkpoints = checkpoint_context
        .lines()
        .filter(|line| line.starts_with("[context_checkpoint "))
        .collect::<Vec<_>>();

    assert_eq!(checkpoints.len(), 8);
    assert!(checkpoints[0].contains("checkpoint-0.md"));
    assert!(checkpoint_context.contains("new working plan"));
    assert!(!checkpoint_context.contains("old working plan"));
}
