use super::super::*;
use serde_json::Value;

fn normalization_text_message(role: &str, text: impl Into<String>) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text.into()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

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
fn normalize_messages_keeps_the_working_memory_tail_note_as_user_context() {
    let note = "Current code-inspection working memory:\n- read_file(file=src/lib.rs) => ok";
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
        crate::ai::history::runtime_synthetic_user_message(Value::String(note.to_string())),
    ];

    let normalized = normalize_messages_for_request(&messages);
    let last = normalized.last().expect("normalized messages");
    assert_eq!(last.role, "user");
    assert_eq!(last.content.as_str(), Some(note));
    assert_eq!(normalized[0].role, "system");
    assert_eq!(normalized[0].content.as_str(), Some("base system"));
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
        // The prefix derived block must not be the tail: upstream models
        // without prefill support reject a request ending with an assistant
        // message, so a closing user message is projected after it.
        vec!["system", "user", "assistant", "user"]
    );
    assert_eq!(
        normalized.last().unwrap().content,
        Value::String(super::super::normalize::DERIVED_CONTEXT_RETURN.to_string())
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
fn normalize_messages_preserves_orphan_tool_evidence_and_strips_broken_tool_calls() {
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

    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[0].role, "system");
    assert_eq!(normalized[1].role, "user");
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(normalized[2].content.as_str(), Some("later answer"));
    assert_eq!(normalized[3].role, "user");
    assert!(normalized[3].content.as_str().is_some_and(|text| {
        text.starts_with(super::super::normalize::HISTORICAL_TOOL_EVIDENCE_CONTEXT_HEADER)
            && text.contains("- tool_call_id=call_1: stale tool output")
    }));
    assert!(
        normalized
            .iter()
            .all(|message| message.tool_calls.is_none())
    );
    assert_eq!(normalized[3].reasoning_content, None);
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
    assert_eq!(
        serde_json::to_vec(&normalized).unwrap(),
        serde_json::to_vec(&messages).unwrap()
    );
}

#[test]
fn normalize_messages_preserves_standalone_tool_runs_in_place_with_bounded_evidence() {
    let mut messages = vec![
        normalization_text_message("system", "base system"),
        normalization_text_message("user", "question"),
    ];
    let long_preview = "界".repeat(300);
    for index in 0..10 {
        let mut message = normalization_text_message(
            "tool",
            match index {
                1 => "   ".to_string(),
                3 => long_preview.clone(),
                _ => format!("evidence-{index}"),
            },
        );
        message.tool_call_id = match index {
            0 => None,
            2 => Some(" ".to_string()),
            _ => Some(format!("orphan-{index}")),
        };
        messages.push(message);
    }
    messages.push(normalization_text_message("assistant", "between runs"));
    for index in 0..2 {
        let mut message = normalization_text_message("tool", format!("tail-evidence-{index}"));
        message.tool_call_id = Some(format!("tail-{index}"));
        messages.push(message);
    }
    let original = messages.clone();

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(messages, original);
    assert_eq!(
        normalized
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec!["system", "user", "user", "assistant", "user"]
    );
    assert_eq!(normalized[3], original[12]);
    let first = normalized[2].content.as_str().unwrap();
    assert!(first.starts_with(super::super::normalize::HISTORICAL_TOOL_EVIDENCE_CONTEXT_HEADER));
    assert!(first.contains("- tool_call_id=unknown: evidence-0"));
    assert!(first.contains("- tool_call_id=unknown: evidence-2"));
    assert!(first.contains(&format!("- tool_call_id=orphan-3: {}…", "界".repeat(239))));
    let evidence_lines = first
        .lines()
        .filter(|line| line.starts_with("- tool_call_id="))
        .collect::<Vec<_>>();
    // A blank item still occupies its original slot in the eight-item window.
    assert_eq!(evidence_lines.len(), 7);
    assert!(
        evidence_lines
            .iter()
            .all(|line| line.split_once(": ").unwrap().1.chars().count() <= 240)
    );
    assert!(!first.contains("evidence-8"));
    assert!(!first.contains("evidence-9"));
    let tail = normalized[4].content.as_str().unwrap();
    assert!(tail.contains("- tool_call_id=tail-0: tail-evidence-0"));
    assert!(tail.contains("- tool_call_id=tail-1: tail-evidence-1"));
    assert!(
        normalized
            .iter()
            .all(|message| message.reasoning_content.is_none())
    );
}

#[test]
fn normalize_messages_keeps_one_user_note_for_a_blank_standalone_tool_run() {
    let messages = vec![
        normalization_text_message("user", "question"),
        normalization_text_message("tool", "   "),
        normalization_text_message("tool", ""),
    ];
    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(normalized.len(), 2);
    assert_eq!(normalized[1].role, "user");
    let text = normalized[1].content.as_str().unwrap();
    assert!(text.starts_with(super::super::normalize::HISTORICAL_TOOL_EVIDENCE_CONTEXT_HEADER));
    assert!(text.contains("reason: tool outputs had no adjacent assistant tool_calls"));
    assert!(!text.contains("- tool_call_id="));
    assert_eq!(normalized[1].reasoning_content, None);
}

#[test]
fn normalize_messages_preserves_adjacent_unmatched_evidence_after_matched_results() {
    let mut assistant = normalization_text_message("assistant", "calling tools");
    assistant.tool_calls = Some(
        ["paired", "unanswered"]
            .into_iter()
            .map(|id| crate::ai::types::ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            })
            .collect(),
    );
    let mut messages = vec![
        normalization_text_message("system", "base system"),
        normalization_text_message("user", "question"),
        assistant,
    ];
    for (id, output) in [
        ("extra", "unmatched evidence"),
        ("paired", "matched evidence"),
        ("paired", "duplicate evidence"),
    ] {
        let mut result = normalization_text_message("tool", output);
        result.tool_call_id = Some(id.to_string());
        messages.push(result);
    }
    messages.push(normalization_text_message("user", "continue"));
    let original = messages.clone();

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(messages, original);
    assert_eq!(normalized.len(), 6);
    assert_eq!(
        normalized[2]
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|call| call.id.as_str())
            .collect::<Vec<_>>(),
        vec!["paired"]
    );
    assert_eq!(normalized[3], original[4]);
    assert_eq!(normalized[4].role, "user");
    let note = normalized[4].content.as_str().unwrap();
    assert!(note.starts_with(super::super::normalize::HISTORICAL_TOOL_EVIDENCE_CONTEXT_HEADER));
    assert!(note.contains("- tool_call_id=extra: unmatched evidence"));
    assert!(note.contains("- tool_call_id=paired: duplicate evidence"));
    assert!(!note.contains("- tool_call_id=paired: matched evidence"));
    assert_eq!(normalized[5], original[6]);
    assert_eq!(
        normalized
            .iter()
            .filter(|message| message.role == "tool")
            .count(),
        1
    );
    assert!(
        normalized
            .iter()
            .all(|message| message.reasoning_content.is_none())
    );
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
fn normalize_messages_prioritizes_folded_tool_evidence_over_boilerplate() {
    let evidence = (0..8)
        .map(|index| format!("- call-{index}: evidence-{index:02}-{}", "e".repeat(42)))
        .collect::<Vec<_>>();
    let note = format!(
        "compressed_tool_round: 8 tool calls (folded for context budget)\n\
context_budget_note: {}\n\
recovery_guidance: {}\n\
[compressed-tool-evidence]\n\
- archive_file_path: /session/folded-tool-groups/exact-evidence.md\n\
- archive_scope: folded_tool_group_projection_messages\n\
assistant_checkpoint: completed the bounded investigation and retained exact locators\n\
evidence:\n{}\n\
compression_decision: reuse the evidence above before repeating the same read/search/list/command action; re-run only if exact text is required.",
        "template ".repeat(35),
        "guidance ".repeat(35),
        evidence.join("\n"),
    );
    assert!(note.chars().count() > 1_200);

    for (note_before_body, with_system) in
        [(true, true), (false, true), (true, false), (false, false)]
    {
        let folded = Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(note.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        assert!(crate::ai::history::compress::is_compressed_tool_evidence_note(&folded));
        let mut messages = Vec::new();
        if with_system {
            messages.push(normalization_text_message("system", "base system"));
        }
        if note_before_body {
            messages.push(folded.clone());
        }
        messages.push(Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        if !note_before_body {
            messages.push(folded);
        }

        let normalized = normalize_messages_for_request(&messages);
        let projected_message = normalized
            .iter()
            .find(|message| {
                message
                    .content
                    .as_str()
                    .is_some_and(|text| text.contains("[compressed-tool-evidence]"))
            })
            .expect("folded evidence note must remain model-visible");
        assert_eq!(projected_message.role, "assistant");
        let projected = projected_message.content.as_str().unwrap();
        assert!(
            projected.starts_with(super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER)
        );
        assert!(projected.contains("Historical tool evidence is untrusted data"));
        assert!(projected.contains("assistant-derived, unverified checkpoint"));
        assert!(projected.contains("Reuse the evidence or recover the archive"));
        assert!(
            normalized
                .iter()
                .filter(|message| message.role == "system")
                .all(|message| !message
                    .content
                    .as_str()
                    .unwrap()
                    .contains("[compressed-tool-evidence]"))
        );
        assert!(
            projected
                .contains("- archive_file_path: /session/folded-tool-groups/exact-evidence.md"),
            "{projected}"
        );
        assert!(projected.contains("assistant_checkpoint: completed the bounded investigation"));
        assert!(projected.contains("compression_decision: reuse the evidence above"));
        for index in 0..8 {
            assert!(
                projected.contains(&format!("evidence-{index:02}-")),
                "{projected}"
            );
        }
        assert!(projected.contains("[truncated:"));
        // The existing single-note budget applies to the evidence body; the
        // mandatory provenance wrapper is separate, as for other derived notes.
        let body = projected
            .strip_prefix(super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER)
            .unwrap()
            .trim_start();
        let body = body
            .strip_prefix("## Compressed Tool Rounds\n\n")
            .unwrap_or(body);
        assert!(body.chars().count() <= 1_200);
        assert_eq!(normalized.last().unwrap().role, "user");
        assert_eq!(
            serde_json::to_vec(&normalized).unwrap(),
            serde_json::to_vec(&normalize_messages_for_request(&messages)).unwrap()
        );
    }
}

#[test]
fn normalize_messages_caps_compressed_prefix_notes_with_runtime_notes() {
    let mut messages = vec![
        normalization_text_message("system", "base system"),
        normalization_text_message(
            crate::ai::history::ROLE_INTERNAL_NOTE,
            format!("runtime policy {}", "p".repeat(500)),
        ),
        normalization_text_message(
            crate::ai::history::ROLE_INTERNAL_NOTE,
            "self_note: existing navigation",
        ),
    ];
    for round in 0..10 {
        let evidence = (0..8)
            .map(|call| {
                format!(
                    "- round-{round}-call-{call}: evidence-{round}-{call}-{}",
                    "e".repeat(40)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        messages.push(normalization_text_message(crate::ai::history::ROLE_INTERNAL_NOTE, format!(
            "compressed_tool_round: 8 tool calls\n[compressed-tool-evidence]\n- archive_file_path: /session/round-{round}.md\nassistant_checkpoint: unverified round-{round}\nevidence:\n{evidence}\ncompression_decision: recover the archive before repeating tools"
        )));
        if round == 0 {
            messages.push(normalization_text_message(
                crate::ai::history::ROLE_INTERNAL_NOTE,
                "runtime policy between compressed notes",
            ));
        }
    }
    messages.push(normalization_text_message("user", "question"));

    let normalized = normalize_messages_for_request(&messages);

    let system = normalized[0].content.as_str().unwrap();
    assert!(system.contains("runtime policy between compressed notes"));
    assert!(!system.contains("compressed_tool_round:"));
    assert!(!system.contains("archive_file_path:"));
    let runtime_payload = system
        .strip_prefix("base system\n\n[Merged system notes from history/runtime]\n")
        .unwrap();
    let derived = normalized
        .iter()
        .find(|message| message.role == "assistant")
        .unwrap()
        .content
        .as_str()
        .unwrap();
    assert!(derived.contains("self_note: existing navigation"));
    let header = super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER;
    let compressed = &derived[derived.find(header).unwrap()..];
    assert_eq!(derived.matches(header).count(), 1);
    let compressed_payload = compressed
        .strip_prefix(header)
        .unwrap()
        .strip_prefix("\n\n")
        .unwrap();
    assert!(runtime_payload.chars().count() + compressed_payload.chars().count() <= 4_000);
    assert!(compressed.contains("- archive_file_path: /session/round-0.md"));
    for call in 0..8 {
        assert!(compressed.contains(&format!("evidence-0-{call}-")));
    }
    assert!(!compressed.contains("/session/round-9.md"));
    assert_eq!(
        normalized.last().unwrap().content.as_str(),
        Some("question")
    );
}

#[test]
fn normalize_messages_preserves_compressed_prefix_locators_when_header_exceeds_remaining_budget() {
    let runtime_notes = (0..3)
        .map(|index| format!("runtime policy {index}: {}", "p".repeat(1182)))
        .collect::<Vec<_>>();
    let mut messages = vec![normalization_text_message("system", "base system")];
    for note in &runtime_notes {
        messages.push(normalization_text_message(
            crate::ai::history::ROLE_INTERNAL_NOTE,
            note.clone(),
        ));
    }
    messages.push(normalization_text_message(
        crate::ai::history::ROLE_INTERNAL_NOTE,
        format!(
            "compressed_tool_round: 1 tool call\n\
             [compressed-tool-evidence]\n\
             - archive_file_path: /session/budget-boundary.md\n\
             - tool_call_id: call-boundary\n\
             assistant_checkpoint: {}\n\
             evidence:\n\
             - call-boundary: observed output\n\
             compression_decision: recover the archive before repeating tools",
            "unverified ".repeat(50)
        ),
    ));
    messages.push(normalization_text_message("user", "question"));

    let normalized = normalize_messages_for_request(&messages);

    let runtime_payload = normalized[0]
        .content
        .as_str()
        .unwrap()
        .strip_prefix("base system\n\n[Merged system notes from history/runtime]\n")
        .unwrap();
    assert_eq!(
        runtime_payload,
        format!("## Additional Notes\n\n{}", runtime_notes.join("\n\n"))
    );
    let header = super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER;
    assert!(4_000 - runtime_payload.chars().count() < header.chars().count());
    let compressed = normalized
        .iter()
        .find(|message| message.role == "assistant")
        .unwrap()
        .content
        .as_str()
        .unwrap();
    assert!(compressed.starts_with(header));
    assert_eq!(compressed.matches(header).count(), 1);
    for line in [
        "- archive_file_path: /session/budget-boundary.md",
        "- tool_call_id: call-boundary",
        "evidence:",
        "- call-boundary: observed output",
    ] {
        assert!(compressed.lines().any(|actual| actual == line), "{line}");
    }
    assert!(compressed.contains("[truncated:"));
}

#[test]
fn normalize_messages_preserves_small_compressed_prefix_bytes() {
    let note = "compressed_tool_round: 1 tool call\n\
                [compressed-tool-evidence]\n\
                - archive_file_path: /session/small-round.md\n\
                - tool_call_id: call-small\n\
                evidence:\n\
                - call-small: observed output";
    let messages = vec![
        normalization_text_message("system", "base system"),
        normalization_text_message(crate::ai::history::ROLE_INTERNAL_NOTE, "runtime policy"),
        normalization_text_message(crate::ai::history::ROLE_INTERNAL_NOTE, note),
        normalization_text_message("user", "question"),
    ];

    let normalized = normalize_messages_for_request(&messages);

    let compressed = normalized
        .iter()
        .find(|message| message.role == "assistant")
        .unwrap()
        .content
        .as_str()
        .unwrap();
    assert_eq!(
        compressed,
        format!(
            "{}\n\n## Compressed Tool Rounds\n\n{note}",
            super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER
        )
    );
}

#[test]
fn normalize_messages_keeps_cached_prefix_bytes_when_compressed_tail_notes_are_added() {
    let note = "compressed_tool_round: 1 tool call\nassistant_checkpoint: prior answer\nevidence:\n- call-1: observed output";
    for compressed_prefix in [false, true] {
        let mut messages = vec![
            normalization_text_message("system", "base system"),
            normalization_text_message(crate::ai::history::ROLE_INTERNAL_NOTE, "runtime policy"),
            normalization_text_message(
                crate::ai::history::ROLE_INTERNAL_NOTE,
                "Context note: reused prior tool results\ncached source",
            ),
        ];
        if compressed_prefix {
            messages.push(normalization_text_message(
                crate::ai::history::ROLE_INTERNAL_NOTE,
                note,
            ));
        }
        messages.push(normalization_text_message("user", "question"));
        messages.push(normalization_text_message("assistant", "answer"));
        let before = normalize_messages_for_request(&messages);
        assert_eq!(
            before[0].content.as_str(),
            Some(
                "base system\n\n[Merged system notes from history/runtime]\n## Cached Tool Results\n\nContext note: reused prior tool results\ncached source\n\n## Additional Notes\n\nruntime policy"
            )
        );
        messages.push(normalization_text_message(
            crate::ai::history::ROLE_INTERNAL_NOTE,
            note,
        ));
        messages.push(normalization_text_message("user", "continue"));

        let after = normalize_messages_for_request(&messages);

        assert_eq!(
            serde_json::to_vec(&after[..before.len()]).unwrap(),
            serde_json::to_vec(&before).unwrap()
        );
        assert_eq!(
            after[before.len()].content.as_str(),
            Some(super::super::normalize::DERIVED_CONTEXT_HANDOFF)
        );
        assert_eq!(after[before.len() + 1].role, "assistant");
        assert!(
            after[before.len() + 1]
                .content
                .as_str()
                .unwrap()
                .starts_with(super::super::normalize::COMPRESSED_TOOL_ROUND_CONTEXT_HEADER)
        );
        assert_eq!(after.last().unwrap().content.as_str(), Some("continue"));
    }
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
    // Fixture assets do not exist on disk, so every projected marker carries
    // the fail-closed stale annotation; recency selection is unaffected.
    let expected = (2..10)
        .map(|index| {
            format!(
                "[context_checkpoint path=/tmp/checkpoint-{index}.md] checkpoint {index} \
                 (stale: checkpoint asset missing or unreadable — content unverifiable)"
            )
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

#[test]
fn normalize_messages_without_system_dedupes_checkpoint_paths() {
    let checkpoint = |text: &str| Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        checkpoint("[context_checkpoint path=/tmp/working-checkpoint.md] stale plan"),
        checkpoint("[context_checkpoint path=/tmp/other-checkpoint.md] independent memory"),
        checkpoint("[context_checkpoint path=/tmp/working-checkpoint.md] current plan"),
    ];

    let normalized = normalize_messages_for_request(&messages);
    let projected = normalized
        .iter()
        .filter_map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(projected.contains("current plan"));
    assert!(projected.contains("independent memory"));
    assert!(!projected.contains("stale plan"));
    assert_eq!(
        messages.len(),
        4,
        "normalization must not mutate canonical input"
    );
}

#[test]
fn normalize_messages_preserves_ambiguous_checkpoint_paths() {
    let checkpoint = |text: &str| Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    let messages = vec![
        checkpoint("[context_checkpoint path=/tmp/root] shared/first.md] first memory"),
        checkpoint("[context_checkpoint path=/tmp/root] shared/second.md] second memory"),
    ];

    let normalized = normalize_messages_for_request(&messages);
    let projected = normalized
        .iter()
        .filter_map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(projected.contains("first memory"));
    assert!(projected.contains("second memory"));
}
