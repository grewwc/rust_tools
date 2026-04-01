use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

use serde_json::Value;

use super::{
    files,
    history::{
        COLON, Message, NEWLINE, SessionStore, append_history, append_history_messages,
        build_message_arr, compress_messages_for_context,
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
        pending_files: None,
        pending_clipboard: false,
        pending_short_output: false,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        writer: None,
        prompt_editor: None,
        agent_context: None,
    };

    let mut question = "a 什么是rust的一个crate？".to_string();
    let model = super::driver::resolve_model_for_input(&app, &mut question);
    assert_eq!(model, app.current_model);
    assert_eq!(question, "a 什么是rust的一个crate？");
}

#[test]
fn image_files_auto_route_to_vl() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str());
    assert_eq!(model, Some(vl));
}

#[test]
fn configured_vl_model_is_used_for_images() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str());
    assert_eq!(model, Some(vl));
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
    let compressed = compress_messages_for_context(messages, 1200, 4, 200);

    assert!(!compressed.is_empty());
    assert_eq!(compressed[0].role, "system");
    assert!(
        compressed[0]
            .content
            .as_str()
            .unwrap_or_default()
            .contains("对话摘要")
    );
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
fn sigint_during_stream_only_cancels_current_reply() {
    let shutdown = AtomicBool::new(false);
    let streaming = AtomicBool::new(true);
    let cancel_stream = AtomicBool::new(false);

    assert_eq!(
        super::driver::sigint_action(&shutdown, &streaming, &cancel_stream),
        super::driver::SigintAction::CancelStream
    );
}

#[test]
fn second_sigint_during_stream_requests_shutdown() {
    let shutdown = AtomicBool::new(false);
    let streaming = AtomicBool::new(true);
    let cancel_stream = AtomicBool::new(true);

    assert_eq!(
        super::driver::sigint_action(&shutdown, &streaming, &cancel_stream),
        super::driver::SigintAction::Shutdown
    );
}

#[test]
fn sigint_while_idle_requests_graceful_shutdown() {
    let shutdown = AtomicBool::new(false);
    let streaming = AtomicBool::new(false);
    let cancel_stream = AtomicBool::new(false);

    assert_eq!(
        super::driver::sigint_action(&shutdown, &streaming, &cancel_stream),
        super::driver::SigintAction::Shutdown
    );
}

#[test]
fn sigint_while_shutdown_pending_exits() {
    let shutdown = AtomicBool::new(true);
    let streaming = AtomicBool::new(false);
    let cancel_stream = AtomicBool::new(false);

    assert_eq!(
        super::driver::sigint_action(&shutdown, &streaming, &cancel_stream),
        super::driver::SigintAction::Exit
    );
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
fn execute_command_blocks_dangerous_programs() {
    assert!(tools::validate_execute_command("rm -rf /").is_err());
    assert!(tools::validate_execute_command("mv a b").is_err());
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
