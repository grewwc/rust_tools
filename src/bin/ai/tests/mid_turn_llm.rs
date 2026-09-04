//! Mid-turn LLM-summary compression tests (history::mid_turn_llm_summarize).

use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};

use super::super::{
    history::{Message, SessionStore, messages_total_chars_pub, mid_turn_llm_summarize},
    types::{FunctionCall, ToolCall},
};
use super::*;

/// Regression: when the older tool groups already saved more than 4K, we must not return early while still above hard_target.
/// The latest full tool group (especially parallel results and large arguments) must keep its paired structure and converge within the total budget.
#[tokio::test]
async fn mid_turn_llm_summary_reaches_hard_target_after_effective_early_folding() {
    let root =
        std::env::temp_dir().join(format!("ai-mid-turn-hard-target-{}", uuid::Uuid::new_v4()));
    let mut app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    app.config.history_file = root.join("history.sqlite");
    app.session_id = "hard-target-regression".to_string();

    let mut messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("继续完成当前任务".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    for index in 0..4 {
        let id = format!("hard-target-call-{index}");
        let result_chars = if index == 3 { 20_000 } else { 3_000 };
        let arguments = if index == 3 {
            serde_json::json!({ "query": "q".repeat(12_000) }).to_string()
        } else {
            serde_json::json!({ "query": format!("old-{index}") }).to_string()
        };
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "text_grep".to_string(),
                    arguments,
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("x".repeat(result_chars)),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    let hard_target = 5_000;
    let (compressed, before, after, did_summarize, _llm_summary_inserted) =
        mid_turn_llm_summarize(&app, messages, 4, 2_000, hard_target, None).await;

    assert!(before > hard_target + 20_000);
    assert!(did_summarize);
    assert!(
        after <= hard_target,
        "after={after}, payload={compressed:?}"
    );
    assert_eq!(after, messages_total_chars_pub(&compressed));
    let latest_call = compressed
        .iter()
        .find_map(|message| {
            message
                .tool_calls
                .as_ref()?
                .iter()
                .find(|call| call.id == "hard-target-call-3")
        })
        .expect("latest assistant tool call must remain structurally present");
    assert!(serde_json::from_str::<Value>(&latest_call.function.arguments).is_ok());
    assert!(compressed.iter().any(|message| {
        message.role == "tool" && message.tool_call_id.as_deref() == Some("hard-target-call-3")
    }));

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn mid_turn_llm_summary_path_a_preserves_raw_archive_pointer() {
    let mut app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    let root = std::env::temp_dir().join(format!("ai-mid-turn-path-a-{}", uuid::Uuid::new_v4()));
    app.config.history_file = root.join("history.sqlite");
    app.session_id = "path-a-archive-regression".to_string();

    let old_user = format!("早期目标: 修复无损压缩回指 {}", "u".repeat(4_000));
    let old_assistant = format!("早期结论: 需要归档 earlier 原文 {}", "a".repeat(4_000));
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
            content: Value::String(old_assistant.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("继续当前任务".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, before, after, _, _) =
        mid_turn_llm_summarize(&app, messages, 1, 1_000, 20_000, None).await;

    assert!(after < before, "Path A should reduce earlier history");
    assert!(compressed.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.starts_with("[mid-turn-summary]"))
    }));
    assert!(compressed.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.contains("归档文件:"))
    }));

    let archive_file = SessionStore::new(app.config.history_file.as_path())
        .session_assets_dir(&app.session_id)
        .join("overflow-history.md");
    let archived =
        std::fs::read_to_string(&archive_file).expect("Path A raw archive should be readable");
    assert!(
        archived.contains("早期目标: 修复无损压缩回指"),
        "{archived}"
    );
    assert!(
        archived.contains("早期结论: 需要归档 earlier 原文"),
        "{archived}"
    );
    assert!(archived.contains("raw_message_json"), "{archived}");

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn mid_turn_llm_summary_path_a_runs_when_old_user_turns_folded_away() {
    // Regression test: after persistent compression, earlier user messages are replaced by internal_note summaries,
    // and the visible role=="user" boundary in the projection is fewer than keep_recent_turns (=2). retained_turn_start
    // returning 0 makes Path A get skipped wholesale, so leftover assistant(tool_calls)/tool records (protected by protocol
    // pairing, impossible to delete one by one) can never be reclaimed by the LLM semantic summary and the context only grows.
    // After the fix: split_at falls back to the first user message position, system-like summary/archive markers are still
    // preserved by preserved_system_end, and the old conversation segment between them can be summarized by Path A normally.
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    let big_tool_output = "x".repeat(12_000);
    let messages = vec![
        Message {
            role: "system".into(),
            content: Value::String("system prompt".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        // Summary + archive markers produced by the earlier compression (internal_note, system-like, protected)
        Message {
            role: "internal_note".into(),
            content: Value::String("长期记忆摘要（压缩保留）：之前的对话已完成。".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "internal_note".into(),
            content: Value::String("归档：早期轮次已存档于 overflow 文件。".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        // Leftover old assistant(tool_calls)+tool (protected by protocol pairing, cannot be deleted one by one)
        Message {
            role: "assistant".into(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: "call_old_1".into(),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: "{\"path\":\"old.rs\"}".into(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".into(),
            content: Value::String(big_tool_output),
            tool_calls: None,
            tool_call_id: Some("call_old_1".into()),
            reasoning_content: None,
        },
        // The most recent 2 user turns (protected tail)
        Message {
            role: "user".into(),
            content: Value::String("请继续修改这个文件".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".into(),
            content: Value::String("好的，我来处理。".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".into(),
            content: Value::String("完成了吗".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".into(),
            content: Value::String("已完成。".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let before = messages_total_chars_pub(&messages);
    let (compressed, _before, after, _, _) =
        mid_turn_llm_summarize(&app, messages, 2, 1_000, 20_000, None).await;
    assert!(
        after < before,
        "应发生压缩：after={} before={}",
        after,
        before
    );
    // Path A should generate a [mid-turn-summary] for the old prefix; before the fix split_at==0 made Path A get skipped
    let has_mid_turn_summary = compressed.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.starts_with("[mid-turn-summary]"))
    });
    assert!(
        has_mid_turn_summary,
        "Path A 应对旧前缀生成 [mid-turn-summary] 摘要，实际 roles: {:?}",
        compressed
            .iter()
            .map(|m| m.role.as_str())
            .collect::<Vec<_>>()
    );
    // The old large tool output should be reclaimed by the summary and no longer appear verbatim in the result
    let still_has_raw_tool_output = compressed.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.len() > 5_000)
    });
    assert!(!still_has_raw_tool_output, "旧的大块 tool 输出应被摘要回收");
}
