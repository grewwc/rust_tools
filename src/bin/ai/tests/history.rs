//! History persistence, canonical-history, and context-projection tests.

use serde_json::Value;

use super::super::{
    history::{
        COLON, MAX_HISTORY_TURNS, Message, NEWLINE, append_history,
        append_history_messages, build_context_history, build_message_arr,
    },
    types::{FunctionCall, ToolCall},
};
use super::*;

#[test]
fn history_file_parsing_txt_matches_go_format() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.txt", uuid::Uuid::new_v4()));
    std::fs::write(
        &path,
        format!("user{COLON}hi{NEWLINE}assistant{COLON}hello{NEWLINE}"),
    )
    .unwrap();

    let messages = build_message_arr(4, &path).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, Value::String("hi".to_string()));
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, Value::String("hello".to_string()));

    let _ = std::fs::remove_file(path);
}

#[test]
fn history_file_parsing_sqlite_matches_go_format() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    append_history(
        &path,
        &format!("user{COLON}hi{NEWLINE}assistant{COLON}hello{NEWLINE}"),
    )
    .unwrap();

    let messages = build_message_arr(4, &path).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, Value::String("hi".to_string()));
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, Value::String("hello".to_string()));

    let _ = std::fs::remove_file(path);
}

#[test]
fn history_file_parsing_txt_round_trips_structured_messages() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.txt", uuid::Uuid::new_v4()));
    let messages = structured_history_messages();

    append_history_messages(&path, &messages).unwrap();

    let loaded = build_message_arr(10, &path).unwrap();
    assert_eq!(loaded, messages);

    let _ = std::fs::remove_file(path);
}

#[test]
fn history_file_parsing_sqlite_round_trips_structured_messages() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let messages = structured_history_messages();

    append_history_messages(&path, &messages).unwrap();

    let loaded = build_message_arr(10, &path).unwrap();
    assert_eq!(loaded, messages);

    let _ = std::fs::remove_file(path);
}

#[test]
fn canonical_history_keeps_reasoning_without_promoting_it_to_visible_content() {
    for ext in ["txt", "sqlite"] {
        let path =
            std::env::temp_dir().join(format!("ai-history-{}.{}", uuid::Uuid::new_v4(), ext));
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("inspect this".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("我先看一下这个文件。".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"file_path":"src/main.rs"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: Some("step by step".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("fn main() {}".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::Null,
                tool_calls: Some(vec![ToolCall {
                    id: "call_2".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"file_path":"Cargo.toml"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: Some("需要查看 Cargo.toml 里的 crate 配置。".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("[package]\nname = \"rust_tools\"".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_2".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("done".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: Some("final hidden reasoning".to_string()),
            },
        ];

        append_history_messages(&path, &messages).unwrap();

        let loaded = build_message_arr(10, &path).unwrap();
        assert_eq!(loaded.len(), 6);
        assert_eq!(
            loaded[1].content,
            Value::String("我先看一下这个文件。".to_string())
        );
        assert_eq!(loaded[1].reasoning_content.as_deref(), Some("step by step"));
        assert_eq!(
            loaded[1].tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
        assert_eq!(loaded[2].content, Value::String("fn main() {}".to_string()));
        assert_eq!(loaded[3].content, Value::Null);
        assert_eq!(
            loaded[3].reasoning_content.as_deref(),
            Some("需要查看 Cargo.toml 里的 crate 配置。")
        );
        assert!(
            loaded
                .iter()
                .all(|message| message.content.as_str().is_none_or(
                    |content| !content.contains("需要查看 Cargo.toml 里的 crate 配置。")
                )),
            "hidden reasoning must not become visible history content"
        );
        assert_eq!(
            loaded[3].tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
        assert_eq!(
            loaded[4].content,
            Value::String("[package]\nname = \"rust_tools\"".to_string())
        );
        assert_eq!(loaded[5].content, Value::String("done".to_string()));
        assert_eq!(
            loaded[5].reasoning_content.as_deref(),
            Some("final hidden reasoning")
        );

        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn declared_model_keeps_raw_reasoning_and_builds_tagged_context_projection() {
    let path = std::env::temp_dir().join(format!(
        "ai-history-glm-reasoning-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String("准备调用工具".to_string()),
        tool_calls: Some(vec![ToolCall {
            id: "call_glm_history".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some("GLM continuation state".to_string()),
    };
    crate::ai::history::append_history_messages_for_model(
        &path,
        std::slice::from_ref(&assistant),
        "glm-5.3",
    )
    .unwrap();

    let loaded = build_message_arr(10, &path).unwrap();
    assert_eq!(loaded, vec![assistant]);
    let context =
        crate::ai::history::read_context_history_sqlite(&path, "reasoning-projection-test")
            .unwrap();
    assert!(
        context.messages[0]
            .reasoning_content
            .as_deref()
            .is_some_and(|reasoning| reasoning
                .starts_with(crate::ai::history::compress::PERSISTED_REASONING_REPLAY_PREFIX)),
        "模型 continuation state 应只在可重建上下文投影里携带来源标记"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn history_retains_turns_under_cap() {
    let turns = MAX_HISTORY_TURNS.saturating_sub(50).max(1);
    for ext in ["txt", "sqlite"] {
        let path =
            std::env::temp_dir().join(format!("ai-history-{}.{}", uuid::Uuid::new_v4(), ext));
        for i in 0..turns {
            append_history_messages_retry_transient(
                &path,
                &[
                    Message {
                        role: "user".to_string(),
                        content: Value::String(format!("u{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    },
                ],
            )
            .unwrap();
        }
        let loaded = build_message_arr(10_000, &path).unwrap();
        assert_eq!(
            loaded.first().unwrap().content,
            Value::String("u0".to_string())
        );
        assert_eq!(
            loaded.last().unwrap().content,
            Value::String(format!("a{}", turns - 1))
        );
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn canonical_history_never_compacts_old_turns() {
    let turns = MAX_HISTORY_TURNS + 50;
    for ext in ["txt", "sqlite"] {
        let path =
            std::env::temp_dir().join(format!("ai-history-{}.{}", uuid::Uuid::new_v4(), ext));
        for i in 0..turns {
            append_history_messages_retry_transient(
                &path,
                &[
                    Message {
                        role: "user".to_string(),
                        content: Value::String(format!("u{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    },
                    Message {
                        role: "tool".to_string(),
                        content: Value::String(format!("t{i}")),
                        tool_calls: None,
                        tool_call_id: Some(format!("call_{i}")),
                        reasoning_content: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}_final")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    },
                ],
            )
            .unwrap();
        }
        let loaded = build_message_arr(10_000, &path).unwrap();
        assert_eq!(loaded.len(), turns * 4);
        let first_user = loaded.iter().find(|m| m.role == "user").unwrap();
        assert_eq!(first_user.content, Value::String("u0".to_string()));
        let user_count = loaded.iter().filter(|m| m.role == "user").count();
        assert_eq!(user_count, turns);
        assert_eq!(
            loaded.last().unwrap().content,
            Value::String(format!("a{}_final", turns - 1))
        );
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn context_history_summarizes_beyond_history_count_instead_of_dropping() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));

    for i in 0..240 {
        append_history_messages_retry_transient(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("question-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("answer-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(32, &path, 6000, 32, 2000, None, None).unwrap();

    assert!(!context.is_empty());
    assert_eq!(
        context.first().unwrap().role,
        crate::ai::history::ROLE_INTERNAL_NOTE
    );
    assert!(
        context
            .first()
            .and_then(|m| m.content.as_str())
            .unwrap_or_default()
            .contains("摘要")
    );
    assert_eq!(context.iter().filter(|m| m.role == "user").count(), 32);
    assert_eq!(
        context.last().unwrap().content,
        Value::String("answer-239".to_string())
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn context_history_keep_last_counts_user_turns_not_raw_messages() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));

    for i in 0..6 {
        append_history_messages_retry_transient(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("question-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: format!("call_{i}"),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "demo_tool".to_string(),
                            arguments: format!(r#"{{"i":{i}}}"#),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "tool".to_string(),
                    content: Value::String(format!("tool-output-{i}")),
                    tool_calls: None,
                    tool_call_id: Some(format!("call_{i}")),
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("answer-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(2, &path, 100_000, 2, 2_000, None, None).unwrap();

    let user_questions = context
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        user_questions,
        vec!["question-4".to_string(), "question-5".to_string()]
    );
    assert!(context.iter().any(|m| {
        crate::ai::history::is_system_like_role(&m.role)
            && m.content
                .as_str()
                .unwrap_or_default()
                .contains("question-0")
    }));

    let _ = std::fs::remove_file(path);
}

#[test]
fn context_history_summary_keeps_tool_names_and_results() {
    let path = std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));

    for i in 0..8 {
        append_history_messages_retry_transient(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("请分析 issue-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: format!("call_{i}"),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "tree".to_string(),
                            arguments: format!(r#"{{"path":"issue-{i}"}}"#),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "tool".to_string(),
                    content: Value::String(format!(
                        "ERROR: repeated failure for issue-{i}\nfull stack trace {}",
                        "x".repeat(400)
                    )),
                    tool_calls: None,
                    tool_call_id: Some(format!("call_{i}")),
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("结论 issue-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(2, &path, 1_800, 2, 1_000, None, None).unwrap();
    let summary = context
        .first()
        .and_then(|m| m.content.as_str())
        .unwrap_or_default()
        .to_string();
    assert!(summary.contains("Verified facts and sources"));
    assert!(summary.contains("Assistant's previous answer (not independently verified)"));
    assert!(summary.contains("tree"));
    assert!(summary.contains("issue-0"));
    assert!(summary.contains("ERROR") || summary.contains("repeated failure"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn context_history_cache_invalidates_after_history_changes() {
    let path =
        std::env::temp_dir().join(format!("ai-history-cache-{}.sqlite", uuid::Uuid::new_v4()));
    append_history(
        &path,
        &format!("user{COLON}first{NEWLINE}assistant{COLON}one{NEWLINE}"),
    )
    .unwrap();

    let first = build_context_history(8, &path, 10_000, 8, 2_000, None, None).unwrap();
    assert_eq!(first.len(), 2);

    std::thread::sleep(std::time::Duration::from_millis(2));
    append_history(
        &path,
        &format!("user{COLON}second{NEWLINE}assistant{COLON}two{NEWLINE}"),
    )
    .unwrap();

    let second = build_context_history(8, &path, 10_000, 8, 2_000, None, None).unwrap();
    assert_eq!(second.len(), 4);
    assert_eq!(
        second.last().unwrap().content,
        serde_json::Value::String("two".to_string())
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn context_history_caps_oversized_canonical_tail_without_mutating_canonical_data() {
    let id = uuid::Uuid::new_v4();
    let path = std::env::temp_dir().join(format!("ai-history-hard-cap-{id}.sqlite"));
    let overflow_dir = std::env::temp_dir().join(format!("ai-history-hard-cap-assets-{id}"));
    let raw_tool_result = "precise-tool-output\n".repeat(4_000);
    assert!(raw_tool_result.chars().count() > 64_000);

    append_history_messages(
        &path,
        &[
            Message {
                role: "user".to_string(),
                content: Value::String("inspect the output".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_oversized".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "inspect_output".to_string(),
                        arguments: r#"{"path":"large.log"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String(raw_tool_result.clone()),
                tool_calls: None,
                tool_call_id: Some("call_oversized".to_string()),
                reasoning_content: None,
            },
        ],
    )
    .unwrap();

    // Even with regular history compression off, the absolute safety cap must still project an oversized canonical tail.
    let context = build_context_history(8, &path, 0, 8, 2_000, Some(overflow_dir.clone()), None).unwrap();
    let projected = context
        .iter()
        .find(|message| message.role == "tool")
        .and_then(|message| message.content.as_str())
        .unwrap();
    assert_ne!(projected, raw_tool_result);
    assert!(projected.starts_with("[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]"));
    assert!(projected.contains("file_path"));
    assert!(projected.chars().count() < 64_000);

    let canonical = build_message_arr(8, &path).unwrap();
    assert_eq!(
        canonical
            .iter()
            .find(|message| message.role == "tool")
            .and_then(|message| message.content.as_str()),
        Some(raw_tool_result.as_str())
    );

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn sqlite_recent_turn_window_reads_only_recent_user_turns() {
    let path =
        std::env::temp_dir().join(format!("ai-history-window-{}.sqlite", uuid::Uuid::new_v4()));
    let mut messages = Vec::new();
    for i in 0..5 {
        messages.push(Message {
            role: "user".to_string(),
            content: serde_json::Value::String(format!("u{i}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(format!("a{i}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    append_history_messages(&path, &messages).unwrap();

    let recent = crate::ai::history::read_recent_turn_window_sqlite(&path, 2).unwrap();
    let texts = recent
        .messages
        .iter()
        .filter_map(|m| m.content.as_str())
        .collect::<Vec<_>>();

    assert_eq!(texts, vec!["u3", "a3", "u4", "a4"]);
    assert!(recent.has_older_messages);

    let _ = std::fs::remove_file(path);
}

#[test]
fn sqlite_context_fastpath_keeps_existing_history_summary() {
    let path = std::env::temp_dir().join(format!(
        "ai-history-fastpath-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let messages = vec![
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\nolder summary".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(
                "[context_checkpoint path=/tmp/older-checkpoint.md] durable earlier finding"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: serde_json::Value::String("u1".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String("a1".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: serde_json::Value::String("u2".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String("a2".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    append_history_messages(&path, &messages).unwrap();

    let context = build_context_history(2, &path, 10_000, 2, 2_000, None, None).unwrap();
    assert_eq!(context[0].role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(
        context[0]
            .content
            .as_str()
            .unwrap_or_default()
            .contains("older summary")
    );
    assert!(context.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|content| content.contains("durable earlier finding"))
    }));

    let _ = std::fs::remove_file(path);
}
