//! Mid-turn LLM-summary compression tests (history::mid_turn_llm_summarize).

use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};

use super::super::{
    history::{Message, SessionStore, messages_total_chars_pub, mid_turn_llm_summarize},
    types::{FunctionCall, ToolCall},
};
use super::*;

struct SummaryFixture {
    app: types::App,
    root: std::path::PathBuf,
    server: tokio::task::JoinHandle<Value>,
}

impl SummaryFixture {
    async fn new(status: &str, content: &str) -> Self {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": content}}]
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let server = tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(10), async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (body_start, body_len) = loop {
                    let mut buf = [0; 4096];
                    let read = socket.read(&mut buf).await.unwrap();
                    assert!(read > 0, "incomplete summary request headers");
                    bytes.extend_from_slice(&buf[..read]);
                    assert!(bytes.len() <= 128_000, "unexpectedly large request headers");
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                        assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
                        let len: usize = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse().unwrap())
                            })
                            .expect("summary request Content-Length");
                        assert!(len <= 128_000, "unexpectedly large summary request");
                        break (end + 4, len);
                    }
                };
                while bytes.len() < body_start + body_len {
                    let mut buf = [0; 4096];
                    let read = socket.read(&mut buf).await.unwrap();
                    assert!(read > 0, "incomplete summary request body");
                    bytes.extend_from_slice(&buf[..read]);
                }
                let request: Value =
                    serde_json::from_slice(&bytes[body_start..body_start + body_len]).unwrap();
                socket.write_all(response.as_bytes()).await.unwrap();
                request
            })
            .await
            .expect("local summary server timed out")
        });
        let root = std::env::temp_dir().join(format!("ai-mid-turn-{}", uuid::Uuid::new_v4()));
        let mut app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
        app.config.history_file = root.join("history.sqlite");
        app.session_id = "summary-fixture".to_string();
        // A registered model overrides config.endpoint. An unregistered, unique
        // model exercises the real request path using only this loopback server.
        app.current_model = format!("summary-fixture-{}", uuid::Uuid::new_v4());
        app.config.endpoint = endpoint.clone();
        assert_eq!(
            crate::ai::models::endpoint_for_model(&app.current_model, &endpoint),
            endpoint
        );
        app.client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        Self { app, root, server }
    }

    async fn request(&mut self) -> Value {
        let request = (&mut self.server)
            .await
            .expect("local summary server panicked");
        assert_eq!(request["model"], self.app.current_model);
        assert_eq!(request["stream"], false);
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["role"], "user");
        request
    }
}

impl Drop for SummaryFixture {
    fn drop(&mut self) {
        self.server.abort();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn assert_summary_source(messages: &[Message], expected: &[Message], app: &types::App) -> Value {
    let record: Value = messages
        .iter()
        .find_map(|message| {
            let json = message
                .content
                .as_str()?
                .strip_prefix("[incremental-memory-v1]")?;
            assert_eq!(message.role, "internal_note");
            Some(serde_json::from_str(json).expect("complete summary record"))
        })
        .expect("Path A must insert a source-bound increment");
    assert_eq!(record["provenance"], "assistant_derived_unverified");
    let entries = record["entries"].as_array().unwrap();
    assert!(!entries.is_empty());
    assert!(
        entries
            .iter()
            .all(|entry| entry["status"] == "derived_unverified")
    );
    let path = std::path::Path::new(record["source"]["archive_file_path"].as_str().unwrap());
    let assets = SessionStore::new(&app.config.history_file).session_assets_dir(&app.session_id);
    assert!(path.starts_with(assets.join("summary-sources")));
    let source = std::fs::read_to_string(path).expect("summary source must be readable");
    let restored: Vec<Message> = source
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        restored, expected,
        "source must preserve full content and tool metadata"
    );
    assert_eq!(record["source"]["start_line"], 1);
    assert_eq!(record["source"]["end_line"], expected.len());
    record
}

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
    assert_path_a_preserves_raw_archive_pointer("200 OK").await;
}

#[tokio::test]
async fn mid_turn_llm_summary_path_a_http_400_fallback_preserves_raw_archive_pointer() {
    assert_path_a_preserves_raw_archive_pointer("400 Bad Request").await;
}

async fn assert_path_a_preserves_raw_archive_pointer(status: &str) {
    let draft = "Goals:\n- Preserve raw history via a readable archive pointer.";
    let mut fixture = SummaryFixture::new(status, draft).await;
    let app = &fixture.app;

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

    let earlier = messages[1..3].to_vec();
    let tail = messages[3..].to_vec();
    // The budget covers both the draft and its complete source/provenance record.
    let (compressed, before, after, effective, inserted) =
        mid_turn_llm_summarize(app, messages, 1, 4_000, 20_000, None).await;

    assert!(after < before, "Path A should reduce earlier history");
    assert!(effective && inserted);
    assert_eq!(after, messages_total_chars_pub(&compressed));
    assert!(compressed.ends_with(&tail));
    let record = assert_summary_source(&compressed, &earlier, app);
    if status == "200 OK" {
        assert_eq!(record["entries"][0]["text"], draft.lines().nth(1).unwrap());
    } else {
        // A valid JSON body on a rejected response must never be used as a
        // successful model summary; the deterministic fallback preserves intent.
        assert!(!record.to_string().contains(draft.lines().nth(1).unwrap()));
        assert!(record.to_string().contains("早期目标"));
    }
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

    assert!(archived.contains(&old_user));
    assert!(archived.contains(&old_assistant));
    let request = fixture.request().await;
    assert!(
        request["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("早期目标")
    );
}

#[tokio::test]
async fn mid_turn_llm_summary_path_a_runs_when_old_user_turns_folded_away() {
    // Regression test: after persistent compression, earlier user messages are replaced by internal_note summaries,
    // and the visible role=="user" boundary in the projection is fewer than keep_recent_turns (=2). retained_turn_start
    // returning 0 makes Path A get skipped wholesale, so leftover assistant(tool_calls)/tool records (protected by protocol
    // pairing, impossible to delete one by one) can never be reclaimed by the LLM semantic summary and the context only grows.
    // After the fix: split_at falls back to the first user message position, system-like summary/archive markers are still
    // preserved by preserved_system_end, and the old conversation segment between them can be summarized by Path A normally.
    let draft = "Goals:\n- Reclaim the earlier tool output without resummarizing prior memory.";
    let mut fixture = SummaryFixture::new("200 OK", draft).await;
    let app = &fixture.app;
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
    let prefix = messages[..3].to_vec();
    let earlier = messages[3..5].to_vec();
    let tail = messages[5..].to_vec();
    let (compressed, measured_before, after, effective, inserted) =
        mid_turn_llm_summarize(app, messages, 2, 4_000, 20_000, None).await;
    assert!(
        after < before,
        "应发生压缩：after={} before={}",
        after,
        before
    );
    assert_eq!(measured_before, before);
    assert_eq!(after, messages_total_chars_pub(&compressed));
    assert!(effective && inserted);
    // A new archive locator is inserted after the system message. Existing
    // prefix messages must stay byte-for-byte intact and in their original order.
    assert_eq!(compressed.first(), prefix.first());
    let retained_prefix: Vec<_> = compressed
        .iter()
        .filter(|message| prefix.contains(message))
        .cloned()
        .collect();
    assert_eq!(retained_prefix, prefix);
    assert!(compressed.ends_with(&tail));
    // The old tool pair must be replaced by a source-bound increment even when
    // earlier user-turn boundaries were already folded out of the projection.
    let record = assert_summary_source(&compressed, &earlier, app);
    assert_eq!(record["entries"][0]["text"], draft.lines().nth(1).unwrap());
    // The old large tool output should be reclaimed by the summary and no longer appear verbatim in the result
    let still_has_raw_tool_output = compressed.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.len() > 5_000)
    });
    assert!(!still_has_raw_tool_output, "旧的大块 tool 输出应被摘要回收");
    assert!(
        !compressed
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("call_old_1"))
    );
    let request = fixture.request().await;
    let transcript = request["messages"][1]["content"].as_str().unwrap();
    assert!(!transcript.contains("之前的对话已完成"));
    assert!(!transcript.contains("请继续修改这个文件"));
}
