use std::sync::{Arc, atomic::AtomicBool};

use serde_json::Value;

use super::{
    files,
    history::{COLON, NEWLINE, build_message_arr},
    models,
    prompt::MultilineHistoryState,
    request::{StreamChoice, StreamChunk, StreamDelta},
    stream, tools,
};

#[test]
fn selector_mapping_matches_go() {
    assert_eq!(
        models::model_from_selector(0, false).as_str(),
        "qwq-plus-latest"
    );
    assert_eq!(
        models::model_from_selector(1, false).as_str(),
        "qwen3.5-plus"
    );
    assert_eq!(models::model_from_selector(2, false).as_str(), "qwen-max");
    assert_eq!(models::model_from_selector(3, false).as_str(), "qwen3-max");
    assert_eq!(
        models::model_from_selector(4, false).as_str(),
        "qwen3-coder-plus"
    );
    assert_eq!(
        models::model_from_selector(5, false).as_str(),
        "deepseek-v3.1"
    );
    assert_eq!(models::model_from_selector(5, true).as_str(), "deepseek-r1");
    assert_eq!(models::model_from_selector(6, false).as_str(), "qwen-flash");
}

#[test]
fn trailing_selector_is_detected() {
    assert_eq!(super::driver::trailing_model_selector("hello -3"), Some(3));
    assert_eq!(super::driver::trailing_model_selector("hello-3"), None);
    assert_eq!(super::driver::trailing_model_selector("hello -33"), None);
}

#[test]
fn resolve_model_is_unicode_safe() {
    use clap::Parser;
    use std::path::PathBuf;

    let cli = super::cli::Cli::parse_from(["a"]);
    let config = super::types::AppConfig {
        api_key: String::new(),
        history_file: PathBuf::new(),
        endpoint: String::new(),
        vl_default_model: models::qwen_vl_flash().to_string(),
    };
    let client = reqwest::blocking::Client::builder().build().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let app = super::types::App {
        cli,
        config,
        client,
        current_model: models::qwen3_max().to_string(),
        pending_files: None,
        pending_clipboard: false,
        pending_short_output: false,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        raw_args: String::new(),
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
    let model = super::driver::attachment_forced_model("qwen-flash", true, models::qwen_vl_flash());
    assert_eq!(model, Some(models::qwen_vl_flash().to_string()));
}

#[test]
fn configured_vl_model_is_used_for_images() {
    let model = super::driver::attachment_forced_model("qwen-flash", true, models::qwen_vl_max());
    assert_eq!(model, Some(models::qwen_vl_max().to_string()));
}

#[test]
fn determine_vl_model_supports_selector_and_fuzzy_name() {
    assert_eq!(models::determine_vl_model(""), models::qwen_vl_flash());
    assert_eq!(models::determine_vl_model("1"), models::qwen_vl_max());
    assert_eq!(
        models::determine_vl_model("qwen-vl-ocr-latst"),
        models::qwen_vl_ocr()
    );
}

#[test]
fn tools_are_disabled_for_qwen_flash() {
    assert!(!models::tools_enabled("qwen-flash"));
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
fn history_file_parsing_matches_go_format() {
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
fn thinking_chunks_are_wrapped_once() {
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
    let streaming = AtomicBool::new(true);
    let cancel_stream = AtomicBool::new(false);

    assert_eq!(
        super::driver::sigint_action(&streaming, &cancel_stream),
        super::driver::SigintAction::CancelStream
    );
}

#[test]
fn second_sigint_during_stream_requests_shutdown() {
    let streaming = AtomicBool::new(true);
    let cancel_stream = AtomicBool::new(true);

    assert_eq!(
        super::driver::sigint_action(&streaming, &cancel_stream),
        super::driver::SigintAction::Shutdown
    );
}

#[test]
fn sigint_while_idle_exits_immediately() {
    let streaming = AtomicBool::new(false);
    let cancel_stream = AtomicBool::new(false);

    assert_eq!(
        super::driver::sigint_action(&streaming, &cancel_stream),
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
fn execute_command_blocks_dangerous_programs() {
    assert!(tools::validate_execute_command("rm -rf /").is_err());
    assert!(tools::validate_execute_command("mv a b").is_err());
    assert!(tools::validate_execute_command("cp a b").is_err());
    assert!(tools::validate_execute_command("sudo ls").is_err());
}

#[test]
fn execute_command_blocks_shell_metacharacters() {
    assert!(tools::validate_execute_command("ls; rm -rf /").is_err());
    assert!(tools::validate_execute_command("ls | wc").is_err());
    assert!(tools::validate_execute_command("ls && pwd").is_err());
    assert!(tools::validate_execute_command("echo hi > /tmp/a").is_err());
    assert!(tools::validate_execute_command("find . -exec ls {} \\;").is_err());
}

#[test]
fn execute_command_allows_readonly_commands() {
    assert!(tools::validate_execute_command("ls").is_ok());
    assert!(tools::validate_execute_command("pwd").is_ok());
    assert!(tools::validate_execute_command("cat Cargo.toml").is_ok());
    assert!(tools::validate_execute_command("rg main src").is_ok());
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
