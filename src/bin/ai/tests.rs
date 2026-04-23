use std::path::PathBuf;
use std::sync::{atomic::AtomicBool, Arc};

use serde_json::Value;

use super::{
    files,
    history::{
        append_history, append_history_messages, build_context_history, build_message_arr,
        compress_messages_for_context, Message, SessionStore, COLON, MAX_HISTORY_TURNS, NEWLINE,
    },
    models,
    prompt::MultilineHistoryState,
    request::{StreamChoice, StreamChunk, StreamDelta},
    stream, tools,
    types::{FunctionCall, ToolCall},
};

fn any_model_name() -> String {
    super::model_names::all()
        .first()
        .map(|m| m.name.clone())
        .expect("models.json is empty")
}

fn vl_model_name_at(index: usize) -> Option<String> {
    super::model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .nth(index)
        .map(|m| m.name.clone())
}

fn any_vl_model_name() -> String {
    vl_model_name_at(0).unwrap_or_else(any_model_name)
}

#[test]
fn default_model_names_exist() {
    assert!(!super::model_names::all().is_empty());
}

#[test]
fn cli_parse_args_basic() {
    let cli = super::cli::parse_cli_args(
        ["a", "hello", "-m", "minimax"]
            .into_iter()
            .map(|s| s.to_string()),
    );
    assert_eq!(cli.model.as_deref(), Some("minimax"));
    assert_eq!(cli.args, vec!["hello".to_string()]);
}

#[test]
fn resolve_model_is_unicode_safe() {
    use std::path::PathBuf;

    let cli = super::cli::ParsedCli::default();
    let config = super::types::AppConfig {
        api_key: String::new(),
        history_file: PathBuf::new(),
        endpoint: String::new(),
        vl_default_model: any_vl_model_name(),
        history_max_chars: 12000,
        history_keep_last: 8,
        history_summary_max_chars: 4000,
        intent_model: None,
        intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/ai/config/intent/intent_model.json"),
        agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/ai/config/agent_route/agent_route_model.json"),
        skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/ai/config/skill_match/skill_match_model.json"),
    };
    let client = reqwest::Client::builder().build().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let app = super::types::App {
        cli,
        config,
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        client,
        current_model: any_model_name(),
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        pending_files: None,
        pending_short_output: false,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        writer: None,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
    };

    let mut question = "a 什么是rust的一个crate？".to_string();
    let model = super::driver::resolve_model_for_input(&app, false, &mut question);
    assert_eq!(model, app.current_model);
    assert_eq!(question, "a 什么是rust的一个crate？");
}

#[test]
fn image_files_auto_route_to_vl() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str(), false);
    assert_eq!(model, Some(vl));
}

#[test]
fn configured_vl_model_is_used_for_images() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str(), false);
    assert_eq!(model, Some(vl));
}

#[test]
fn successful_ocr_keeps_text_model_for_images() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str(), true);
    assert_eq!(model, None);
}

#[test]
fn determine_vl_model_supports_selector_and_fuzzy_name() {
    assert_eq!(models::determine_vl_model(""), any_vl_model_name());
    assert_eq!(models::determine_vl_model("0"), any_vl_model_name());
    if let Some(vl1) = vl_model_name_at(1) {
        assert_eq!(models::determine_vl_model("1"), vl1);
    } else {
        assert_eq!(models::determine_vl_model("1"), any_vl_model_name());
    }

    let vl = any_vl_model_name();
    let canonical = models::determine_vl_model(&vl);
    assert_eq!(canonical, vl);
}

#[test]
fn tools_are_disabled_for_qwen_flash() {
    assert!(!models::tools_enabled("qwen3.5-flash"));
    assert!(models::tools_enabled("qwen3-max"));
}

#[test]
fn image_file_detection_by_suffix() {
    assert!(files::is_image_path("/tmp/hello.png"));
    assert!(files::is_image_path("/tmp/hello.JPEG"));
    assert!(!files::is_image_path("/tmp/hello.pdf"));
}

#[test]
fn image_mime_type_matches_suffix() {
    assert_eq!(files::image_mime_type("a.png"), "image/png");
    assert_eq!(files::image_mime_type("a.jpg"), "image/jpeg");
    assert_eq!(files::image_mime_type("a.unknown"), "image/jpeg");
}

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
    let compressed = compress_messages_for_context(messages, 1200, 4, 200, None);

    assert!(!compressed.is_empty());
    assert_eq!(compressed[0].role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(compressed[0]
        .content
        .as_str()
        .unwrap_or_default()
        .contains("对话摘要"));
    assert!(compressed[0]
        .content
        .as_str()
        .unwrap_or_default()
        .contains("初始目标: u0"));
    assert_eq!(
        compressed.last().unwrap().content,
        Value::String(format!("a9 {long}"))
    );
    let total = compressed
        .iter()
        .map(|m| m.content.as_str().map(|s| s.len()).unwrap_or(0))
        .sum::<usize>();
    assert!(total <= 1200);

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
        blob.push_str(&format!(
            "user{COLON}QUESTION_{i:02} {long}{NEWLINE}"
        ));
        blob.push_str(&format!("assistant{COLON}ANSWER_{i:02} {long}{NEWLINE}"));
    }
    append_history(&path, &blob).unwrap();

    let messages = build_message_arr(300, &path).unwrap();
    // keep_last=256 models the CLI default `--history 256`; max_chars=4000
    // is far smaller than the raw history size (30 turns * ~560 bytes ~= 17k).
    let compressed = compress_messages_for_context(messages, 4000, 256, 600, None);

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
        note_text.contains("初始目标: QUESTION_00"),
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
    let path = std::env::temp_dir().join(format!(
        "ai-overflow-test-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
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
        compress_messages_for_context(messages, 2000, 256, 400, Some(overflow_dir.clone()));

    let first_msg = compressed.first().expect("should have messages");
    assert_eq!(
        first_msg.role, crate::ai::history::ROLE_INTERNAL_NOTE,
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
            text.contains("原始归档文件").then_some(text)
        })
        .expect("should include an explicit archive note");
    assert!(
        archive_text.contains("read_file_lines"),
        "archive note should give an explicit file-read action, got: {archive_text:?}"
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
        overflow_content.contains("溢出对话历史"),
        "overflow file should have the header"
    );

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn session_delete_cleans_up_overflow_history_file() {
    let session_id = format!("test-{}", uuid::Uuid::new_v4());
    let history_file = std::env::temp_dir().join(format!("ai-session-cleanup-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file(&session_id);
    let assets = store.session_assets_dir(&session_id);
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(assets.join("overflow-history.md"), "# test overflow content").unwrap();
    std::fs::write(&db, b"test").unwrap();

    assert!(assets.join("overflow-history.md").exists());

    store.delete_session(&session_id).unwrap();

    assert!(!assets.exists(), "assets dir (including overflow file) should be deleted");
    assert!(!db.exists(), "sqlite file should be deleted");

    let _ = std::fs::remove_dir_all(store.session_assets_dir("__cleanup__"));
}

fn structured_history_messages() -> Vec<Message> {
    vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": "hello"
            })]),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "demo".to_string(),
                    arguments: r#"{"x":1}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ]
}

#[test]
fn history_retains_turns_under_cap() {
    let turns = MAX_HISTORY_TURNS.saturating_sub(50).max(1);
    for ext in ["txt", "sqlite"] {
        let path =
            std::env::temp_dir().join(format!("ai-history-{}.{}", uuid::Uuid::new_v4(), ext));
        for i in 0..turns {
            append_history_messages(
                &path,
                &[
                    Message {
                        role: "user".to_string(),
                        content: Value::String(format!("u{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}")),
                        tool_calls: None,
                        tool_call_id: None,
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
fn history_compacts_old_turns_into_summary() {
    let turns = MAX_HISTORY_TURNS + 50;
    for ext in ["txt", "sqlite"] {
        let path =
            std::env::temp_dir().join(format!("ai-history-{}.{}", uuid::Uuid::new_v4(), ext));
        for i in 0..turns {
            append_history_messages(
                &path,
                &[
                    Message {
                        role: "user".to_string(),
                        content: Value::String(format!("u{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}")),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    Message {
                        role: "tool".to_string(),
                        content: Value::String(format!("t{i}")),
                        tool_calls: None,
                        tool_call_id: Some(format!("call_{i}")),
                    },
                    Message {
                        role: "assistant".to_string(),
                        content: Value::String(format!("a{i}_final")),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ],
            )
            .unwrap();
        }
        let loaded = build_message_arr(10_000, &path).unwrap();
        assert_eq!(loaded.first().unwrap().role, crate::ai::history::ROLE_INTERNAL_NOTE);
        assert!(loaded
            .first()
            .and_then(|m| m.content.as_str())
            .unwrap_or_default()
            .contains("历史摘要"));
        let first_user = loaded.iter().find(|m| m.role == "user").unwrap();
        assert_ne!(first_user.content, Value::String("u0".to_string()));
        let user_count = loaded.iter().filter(|m| m.role == "user").count();
        assert!(user_count <= MAX_HISTORY_TURNS);
        assert!(user_count < turns);
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
        append_history_messages(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("question-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("answer-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(32, &path, 6000, 32, 2000, None).unwrap();

    assert!(!context.is_empty());
    assert_eq!(context.first().unwrap().role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(context
        .first()
        .and_then(|m| m.content.as_str())
        .unwrap_or_default()
        .contains("摘要"));
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
        append_history_messages(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("question-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
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
                },
                Message {
                    role: "tool".to_string(),
                    content: Value::String(format!("tool-output-{i}")),
                    tool_calls: None,
                    tool_call_id: Some(format!("call_{i}")),
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("answer-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(2, &path, 100_000, 2, 2_000, None).unwrap();

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
        append_history_messages(
            &path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String(format!("请分析 issue-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: format!("call_{i}"),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "grep_search".to_string(),
                            arguments: format!(r#"{{"query":"issue-{i}"}}"#),
                        },
                    }]),
                    tool_call_id: None,
                },
                Message {
                    role: "tool".to_string(),
                    content: Value::String(format!(
                        "ERROR: repeated failure for issue-{i}\nfull stack trace {}",
                        "x".repeat(400)
                    )),
                    tool_calls: None,
                    tool_call_id: Some(format!("call_{i}")),
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(format!("结论 issue-{i}")),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
        )
        .unwrap();
    }

    let context = build_context_history(2, &path, 1_800, 2, 1_000, None).unwrap();
    let summary = context
        .first()
        .and_then(|m| m.content.as_str())
        .unwrap_or_default()
        .to_string();
    assert!(summary.contains("已知工具结论"));
    assert!(summary.contains("grep_search"));
    assert!(summary.contains("issue-0"));
    assert!(summary.contains("ERROR") || summary.contains("repeated failure"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn context_history_cache_invalidates_after_history_changes() {
    let path = std::env::temp_dir().join(format!("ai-history-cache-{}.sqlite", uuid::Uuid::new_v4()));
    append_history(
        &path,
        &format!("user{COLON}first{NEWLINE}assistant{COLON}one{NEWLINE}"),
    )
    .unwrap();

    let first = build_context_history(8, &path, 10_000, 8, 2_000, None).unwrap();
    assert_eq!(first.len(), 2);

    std::thread::sleep(std::time::Duration::from_millis(2));
    append_history(
        &path,
        &format!("user{COLON}second{NEWLINE}assistant{COLON}two{NEWLINE}"),
    )
    .unwrap();

    let second = build_context_history(8, &path, 10_000, 8, 2_000, None).unwrap();
    assert_eq!(second.len(), 4);
    assert_eq!(second.last().unwrap().content, serde_json::Value::String("two".to_string()));

    let _ = std::fs::remove_file(path);
}

#[test]
fn sqlite_recent_turn_window_reads_only_recent_user_turns() {
    let path = std::env::temp_dir().join(format!("ai-history-window-{}.sqlite", uuid::Uuid::new_v4()));
    let mut messages = Vec::new();
    for i in 0..5 {
        messages.push(Message {
            role: "user".to_string(),
            content: serde_json::Value::String(format!("u{i}")),
            tool_calls: None,
            tool_call_id: None,
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(format!("a{i}")),
            tool_calls: None,
            tool_call_id: None,
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
    let path = std::env::temp_dir().join(format!("ai-history-fastpath-{}.sqlite", uuid::Uuid::new_v4()));
    let messages = vec![
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\nolder summary".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: serde_json::Value::String("u1".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String("a1".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: serde_json::Value::String("u2".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String("a2".to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    append_history_messages(&path, &messages).unwrap();

    let context = build_context_history(2, &path, 10_000, 2, 2_000, None).unwrap();
    assert_eq!(context[0].role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(context[0]
        .content
        .as_str()
        .unwrap_or_default()
        .contains("older summary"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn session_delete_removes_sqlite_sidecars() {
    let history_file =
        std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(history_file.as_path());
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file("abc");
    std::fs::write(&db, b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-wal", db.display())), b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-shm", db.display())), b"test").unwrap();
    std::fs::write(PathBuf::from(format!("{}-journal", db.display())), b"test").unwrap();
    let assets = store.session_assets_dir("abc");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(assets.join("paste.png"), b"test").unwrap();

    assert!(db.exists());
    assert!(PathBuf::from(format!("{}-wal", db.display())).exists());
    assert!(PathBuf::from(format!("{}-shm", db.display())).exists());
    assert!(PathBuf::from(format!("{}-journal", db.display())).exists());
    assert!(assets.exists());

    assert!(store.delete_session("abc").unwrap());

    assert!(!db.exists());
    assert!(!PathBuf::from(format!("{}-wal", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", db.display())).exists());
    assert!(!PathBuf::from(format!("{}-journal", db.display())).exists());
    assert!(!assets.exists());
}

#[test]
fn session_clear_all_removes_all_sqlite_sidecars() {
    let history_file =
        std::env::temp_dir().join(format!("ai-history-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(history_file.as_path());
    store.ensure_root_dir().unwrap();

    for id in ["a", "b", "c"] {
        let db = store.session_history_file(id);
        std::fs::write(&db, b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-wal", db.display())), b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-shm", db.display())), b"test").unwrap();
        std::fs::write(PathBuf::from(format!("{}-journal", db.display())), b"test").unwrap();
        let assets = store.session_assets_dir(id);
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("paste.png"), b"test").unwrap();
    }

    let deleted = store.clear_all_sessions().unwrap();
    assert_eq!(deleted, 3);

    for id in ["a", "b", "c"] {
        let db = store.session_history_file(id);
        assert!(!db.exists());
        assert!(!PathBuf::from(format!("{}-wal", db.display())).exists());
        assert!(!PathBuf::from(format!("{}-shm", db.display())).exists());
        assert!(!PathBuf::from(format!("{}-journal", db.display())).exists());
        let assets = store.session_assets_dir(id);
        assert!(!assets.exists());
    }
}

#[test]
fn thinking_chunks_are_wrapped_once() {
    colored::control::set_override(false);
    let chunk = StreamChunk {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: String::new(),
                reasoning_content: "step one".to_string(),
                reasoning_details: String::new(),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }],
    };
    let mut thinking_open = false;
    let text =
        stream::extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
    assert_eq!(text, "\n<thinking>\nstep one");
    assert!(thinking_open);

    let chunk = StreamChunk {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: "final".to_string(),
                reasoning_content: String::new(),
                reasoning_details: String::new(),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }],
    };
    let text =
        stream::extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
    assert_eq!(text, "\n<end thinking>\nfinal");
    assert!(!thinking_open);
}

#[test]
fn multiline_history_navigation_restores_draft() {
    let mut history =
        MultilineHistoryState::new(vec!["first".to_string(), "second\nline".to_string()]);

    assert_eq!(history.previous("draft"), Some("second\nline".to_string()));
    assert_eq!(history.previous("ignored"), Some("first".to_string()));
    assert_eq!(history.previous("ignored"), None);
    assert_eq!(history.next(), Some("second\nline".to_string()));
    assert_eq!(history.next(), Some("draft".to_string()));
    assert_eq!(history.next(), None);
}

#[test]
fn table_preview_lines_are_not_double_printed_after_live_emit() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);

    let header_out = renderer.consume_line("| name | value |", false);
    assert!(header_out.contains("| name | value |\n"));

    let sep_out = renderer.consume_line("| --- | --- |", true);
    assert_eq!(sep_out, "");

    let row_out = renderer.consume_line("| foo | bar |", true);
    assert_eq!(row_out, "");

    let end_out = renderer.consume_line("done", false);
    assert!(end_out.contains("\x1b["));
    assert!(end_out.contains("done"));
}

#[test]
fn table_live_preview_detection_requires_table_like_content() {
    assert!(stream::line_looks_like_table_preview("| col1 | col2"));
    assert!(stream::line_looks_like_table_preview("  col1 | col2"));
    assert!(!stream::line_looks_like_table_preview("plain text"));
    assert!(!stream::line_looks_like_table_preview("```| not table"));
}

#[test]
fn math_frac_renders_with_nested_braces() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    assert_eq!(renderer.consume_line("$$", false), "\n");

    let out = renderer.consume_line(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", false);
    assert!(out.contains("x ="));
    assert!(out.contains("(-b ± √(b² - 4ac))/2a"));
    assert!(!out.contains("\\frac"));

    let out2 = renderer.consume_line(r"y = \frac{1}{\frac{2}{3}}", false);
    assert!(out2.contains("y ="));
    assert!(out2.contains("1/(2/3)"));
    assert!(!out2.contains("\\frac"));
}

#[test]
fn math_renderer_preserves_longer_commands_and_literal_braces() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    assert_eq!(renderer.consume_line("$$", false), "\n");

    let out = renderer.consume_line(
        r"\leftarrow \rightarrow \leftrightarrow \subseteq \supseteq \sqrt[3]{x} \sqrt[5]{y} \left\{a\right\}",
        false,
    );
    assert!(out.contains("←"));
    assert!(out.contains("→"));
    assert!(out.contains("↔"));
    assert!(out.contains("⊆"));
    assert!(out.contains("⊇"));
    assert!(out.contains("∛(x)"));
    assert!(out.contains("√[5](y)"));
    assert!(out.contains("{a}"));
    assert!(!out.contains("arrow"));
    assert!(!out.contains("⊂eq"));
    assert!(!out.contains("⊃eq"));
}

#[test]
fn math_renderer_maps_mathbb_and_preserves_unknown_commands() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    assert_eq!(renderer.consume_line("$$", false), "\n");

    let out = renderer.consume_line(r"\mathbb{R} \customcmd \alpha", false);
    assert!(out.contains("ℝ"));
    assert!(out.contains(r"\customcmd"));
    assert!(out.contains("α"));
}

#[test]
fn execute_command_blocks_dangerous_programs() {
    assert!(tools::validate_execute_command("rm -rf /").is_err());
    // assert!(tools::validate_execute_command("mv a b").is_err());
    assert!(tools::validate_execute_command("sudo ls").is_err());
}

#[test]
fn execute_command_allows_common_shell_syntax() {
    assert!(tools::validate_execute_command("ls | wc").is_ok());
    assert!(tools::validate_execute_command("ls && pwd").is_ok());
    assert!(tools::validate_execute_command("echo hi > /tmp/a").is_ok());
}

#[test]
fn execute_command_allows_readonly_commands() {
    assert!(tools::validate_execute_command("ls").is_ok());
    assert!(tools::validate_execute_command("pwd").is_ok());
    assert!(tools::validate_execute_command("cat Cargo.toml").is_ok());
    assert!(tools::validate_execute_command("rg main src").is_ok());
}

#[test]
fn execute_command_captures_stdout() {
    let tool_call = ToolCall {
        id: "call_1".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "execute_command".to_string(),
            arguments: r#"{"command":"echo hello"}"#.to_string(),
        },
    };
    let res = tools::execute_tool_call(&tool_call).unwrap();
    assert_eq!(res.content.trim(), "hello");
}

#[test]
fn stream_chunk_accepts_null_content() {
    let payload = r#"{"choices":[{"delta":{"content":null,"reasoning_content":null}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_accepts_reasoning_alias() {
    // OpenCode/OpenRouter providers often stream reasoning under `delta.reasoning`.
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":"step by step"}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_accepts_reasoning_text_alias() {
    // Some provider shims expose the same field as `delta.reasoning_text`.
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning_text":"step by step"}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_ignores_structured_reasoning_object_without_text() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"confidence":0.9,"thinking":true}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_extracts_text_from_reasoning_object() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"thinking","text":"step by step"}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_extracts_nested_reasoning_delta_text() {
    let payload =
        r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"reasoning_text","delta":"No"}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "No");
}

#[test]
fn stream_chunk_ignores_bool_and_number_reasoning() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":42}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_merges_reasoning_details_into_reasoning_content() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning_details":[{"text":"step 1"},{"text":" step 2"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
    parsed.merge_reasoning();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step 1 step 2");
}

#[test]
fn stream_chunk_merges_reasoning_details_prefix_with_punctuation_continuation() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":", that's not really necessary.","reasoning_details":[{"delta":"No"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        ", that's not really necessary."
    );
    parsed.merge_reasoning();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "No, that's not really necessary."
    );
}

#[test]
fn stream_chunk_reasoning_content_takes_priority_over_details() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":"from reasoning field","reasoning_details":[{"text":"from details"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "from reasoning field");
    parsed.merge_reasoning();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "from reasoning field");
}

#[test]
fn merge_reasoning_fragments_stripped_overlap() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("I think", " think this is right"),
        "I think this is right"
    );
}

#[test]
fn merge_reasoning_fragments_cjk_punctuation_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("是的", "，这很重要"),
        "是的，这很重要"
    );
    assert_eq!(
        merge_reasoning_fragments("注意", "！危险"),
        "注意！危险"
    );
}

#[test]
fn merge_reasoning_fragments_english_contraction_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("It is", "n't necessary"),
        "It isn't necessary"
    );
    assert_eq!(
        merge_reasoning_fragments("I", "'ve already checked"),
        "I've already checked"
    );
    assert_eq!(
        merge_reasoning_fragments("They", "'re coming"),
        "They're coming"
    );
}

#[test]
fn merge_reasoning_fragments_no_false_positive_on_independent_sentence() {
    use crate::ai::request::merge_reasoning_fragments;
    let result = merge_reasoning_fragments("First step done", "Second step begins");
    assert_eq!(result, "Second step begins");
}

#[test]
fn merge_reasoning_fragments_ellipsis_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("等等", "…还有更多"),
        "等等…还有更多"
    );
}

#[test]
fn stream_chunk_opencode_structured_content_extracts_text() {
    let payload = r#"{"choices":[{"delta":{"content":[{"type":"output_text","text":"hi"}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.content, "hi");
}

#[test]
fn stream_tool_call_maps_type_field() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"mcp_demo_get_time","arguments":""}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let call = &parsed.choices[0].delta.tool_calls[0];
    assert_eq!(call.id, "call_1");
    assert_eq!(call.tool_type, "function");
    assert_eq!(call.function.name, "mcp_demo_get_time");
}

#[test]
fn stream_tool_call_defaults_when_nulls_present() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":null,"type":null,"function":{"name":null,"arguments":null}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let call = &parsed.choices[0].delta.tool_calls[0];
    assert_eq!(call.id, "");
    assert_eq!(call.tool_type, "");
    assert_eq!(call.function.name, "");
    assert_eq!(call.function.arguments, "");
}

#[test]
fn stream_chunk_accepts_structured_content_arrays() {
    let payload = r#"{"choices":[{"delta":{"content":[{"type":"output_text","text":"hel"},{"type":"output_text","text":"lo"}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.content, "hello");
}

#[test]
fn stream_tool_call_accepts_object_arguments() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":{"file":"a.rs","patch":"..."}}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let args: Value = serde_json::from_str(&parsed.choices[0].delta.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["file"], "a.rs");
    assert_eq!(args["patch"], "...");
}
