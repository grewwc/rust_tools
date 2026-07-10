use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

use rust_tools::cw::SkipSet;
use serde_json::Value;

use super::{
    files,
    history::{
        COLON, MAX_HISTORY_TURNS, Message, NEWLINE, SessionStore, append_history,
        append_history_messages, build_context_history, build_message_arr,
        compress_messages_for_context, mid_turn_compress,
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

fn vl_model_handle_at(index: usize) -> Option<String> {
    super::model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .nth(index)
        .map(|m| super::model_names::model_handle(m))
}

fn any_vl_model_handle() -> String {
    vl_model_handle_at(0).unwrap_or_else(any_model_name)
}

fn test_app_with_cancel_stream(cancel_stream: Arc<AtomicBool>) -> super::types::App {
    super::types::App {
        cli: super::cli::ParsedCli::default(),
        config: super::types::AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: any_vl_model_name(),
            history_max_chars: 12000,
            history_keep_last: 8,
            history_summary_max_chars: 4000,
            intent_model: None,
            agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/agent_route/agent_route_model.json"),
            skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/skill_match/skill_match_model.json"),
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: super::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: any_model_name(),
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skill: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown: Arc::new(AtomicBool::new(false)),
        streaming: Arc::new(AtomicBool::new(false)),
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
    }
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
fn cli_parse_note_search_interactive_mode() {
    let cli = super::cli::parse_cli_args(
        ["a", "-ns", "-i", "帮我找之前记过的 trait object"]
            .into_iter()
            .map(|s| s.to_string()),
    );
    assert!(cli.note_search);
    assert!(cli.interactive);
    assert_eq!(cli.args, vec!["帮我找之前记过的 trait object".to_string()]);
}

#[test]
fn resolve_model_is_unicode_safe() {
    use std::path::PathBuf;

    let cli = super::cli::ParsedCli::default();
    let config = super::types::AppConfig {
        api_key: String::new(),
        base_history_file: PathBuf::new(),
        history_file: PathBuf::new(),
        endpoint: String::new(),
        vl_default_model: any_vl_model_name(),
        history_max_chars: 12000,
        history_keep_last: 8,
        history_summary_max_chars: 4000,
        intent_model: None,
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
        active_persona: super::persona::default_persona(),
        client,
        current_model: any_model_name(),
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skill: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
    };

    let mut question = "a 什么是rust的一个crate？".to_string();
    let model = super::driver::resolve_model_for_input(&app, false, &mut question);
    assert_eq!(model, app.current_model);
    assert_eq!(question, "a 什么是rust的一个crate？");
}

#[test]
fn image_files_keep_text_model() {
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, "any", false);
    assert_eq!(model, None);
}

#[test]
fn take_stream_cancelled_clears_request_interrupt_futex() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let cancel_stream = Arc::new(AtomicBool::new(true));
    let app = test_app_with_cancel_stream(cancel_stream.clone());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    crate::ai::driver::signal::signal_request_interrupt();

    let futex = crate::ai::driver::signal::request_interrupt_futex().unwrap();
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(futex), Some(1));
    }

    assert!(super::types::take_stream_cancelled(&app));
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(futex), Some(0));
    }
}

#[test]
fn request_shutdown_sets_request_interrupt_futex() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());

    assert!(app.shutdown.load(std::sync::atomic::Ordering::Relaxed));
    let futex = crate::ai::driver::signal::request_interrupt_futex().unwrap();
    let os = app.os.lock().unwrap();
    assert_eq!(os.futex_load(futex), Some(1));
    drop(os);
    crate::ai::driver::signal::clear_request_interrupt();
}

#[tokio::test]
async fn wait_for_interrupt_sources_returns_after_shutdown_request() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let shutdown = app.shutdown.clone();
    let waiter = tokio::spawn(async move {
        crate::ai::driver::signal::wait_for_interrupt_sources(None, None).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    crate::ai::driver::signal::request_shutdown(shutdown.as_ref());

    tokio::time::timeout(std::time::Duration::from_millis(200), waiter)
        .await
        .expect("shutdown should wake interrupt waiter")
        .expect("waiter should complete cleanly");
    crate::ai::driver::signal::clear_request_interrupt();
}

#[tokio::test]
async fn wait_for_interrupt_sources_returns_after_daemon_cancel() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let app = test_app_with_cancel_stream(Arc::new(AtomicBool::new(false)));
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let local_interrupt =
        crate::ai::driver::signal::alloc_interrupt_futex("background_cancel_test")
            .expect("local interrupt futex");
    let (handle, cancel_token) = {
        let mut os = app.os.lock().unwrap();
        os.daemon_register(
            "background_cancel_test".to_string(),
            aios_kernel::primitives::DaemonKind::Reflection,
            None,
        )
    };

    let waiter = tokio::spawn(async move {
        crate::ai::driver::signal::wait_for_interrupt_sources(
            Some(cancel_token),
            Some(local_interrupt),
        )
        .await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    {
        let mut os = app.os.lock().unwrap();
        assert!(os.cancel_daemon(handle));
    }

    tokio::time::timeout(std::time::Duration::from_millis(200), waiter)
        .await
        .expect("daemon cancel should wake interrupt waiter")
        .expect("waiter should complete cleanly");

    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.futex_load(local_interrupt), Some(1));
    }
    crate::ai::driver::signal::destroy_interrupt_futex(local_interrupt);
}

#[test]
fn successful_ocr_keeps_text_model_for_images() {
    let vl = any_vl_model_name();
    let model = super::driver::attachment_forced_model("qwen3.5-flash", true, vl.as_str(), true);
    assert_eq!(model, None);
}

#[test]
fn partial_ocr_success_still_counts_as_usable_for_text_models() {
    let ocr = super::driver::model::OcrExtraction {
        tool_name: "mcp_ocr_ocr_image".to_string(),
        content: "ok".to_string(),
        images: vec![
            super::driver::model::OcrImageSummary {
                file_name: "ok.png".to_string(),
                extracted_chars: 2,
                error: None,
            },
            super::driver::model::OcrImageSummary {
                file_name: "bad.png".to_string(),
                extracted_chars: 0,
                error: Some("failed".to_string()),
            },
        ],
    };
    assert!(ocr.has_usable_text());
}

#[test]
fn all_failed_ocr_does_not_keep_text_model() {
    let ocr = super::driver::model::OcrExtraction {
        tool_name: "mcp_ocr_ocr_image".to_string(),
        content: String::new(),
        images: vec![super::driver::model::OcrImageSummary {
            file_name: "bad.png".to_string(),
            extracted_chars: 0,
            error: Some("failed".to_string()),
        }],
    };
    assert!(!ocr.has_usable_text());
}

#[test]
fn determine_vl_model_supports_selector_and_fuzzy_name() {
    // 空输入走 default_vl_model（quality_tier 优先）；数字索引走"按 is_vl 过滤后的第 N 个"。
    // 两条路径并不等价，这里分别按各自的不变量来断言，避免硬编码具体模型名。
    let empty = models::determine_vl_model("");
    let zero = models::determine_vl_model("0");
    let first_vl = any_vl_model_handle();
    assert_eq!(
        zero, first_vl,
        "selector \"0\" should pick first VL in models.json"
    );
    // empty 仅要求是 VL 模型即可（best-by-tier 可能与 first_vl 不同）。
    assert!(
        super::model_names::find_by_identifier(&empty)
            .map(|m| m.is_vl)
            .unwrap_or(false)
    );

    if let Some(vl1) = vl_model_handle_at(1) {
        assert_eq!(models::determine_vl_model("1"), vl1);
    } else {
        // 越界时回退到 default_vl_model
        assert_eq!(models::determine_vl_model("1"), empty);
    }

    // 直接以已知 VL 模型名作输入时，应返回原名（exact match）。
    let canonical = models::determine_vl_model(&first_vl);
    assert_eq!(canonical, first_vl);
}

#[test]
fn tools_default_flag_is_respected_per_model_entry() {
    // 以前这里硬编码 qwen3.5-flash / qwen3-max 的 tools_enabled 行为；这两个模型
    // 已经从 models.json 中移除。改成扫描真实条目，校验 models::tools_enabled
    // 与 ModelDef.tools_default_enabled 的对齐关系，仍能守住"配置即真相"的不变量。
    for def in super::model_names::all() {
        assert_eq!(
            models::tools_enabled(&def.name),
            def.tools_default_enabled,
            "model {} tools_enabled should match its tools_default_enabled flag",
            def.name
        );
    }
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
    let compressed = compress_messages_for_context(messages, 1800, 4, 200, None);

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
            .contains("初始目标: u0")
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
        compress_messages_for_context(messages, 2000, 256, 400, Some(overflow_dir.clone()));

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
            text.contains("原始归档文件").then_some(text)
        })
        .expect("should include an explicit archive note");
    assert!(
        archive_text.contains("read_file"),
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
            content: Value::String("x".repeat(4000)),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    let compressed = compress_messages_for_context(messages, 20_000, 256, 400, Some(overflow_dir));

    let stub = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            text.contains("Output preserved for non-compressible tool `read_file`")
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
    // stub 内应保留一段内容预览作为召回锚点，避免后续 turn "失忆"。
    assert!(
        stub.contains("Preview (for recall"),
        "stub should contain a content preview: {stub}"
    );
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
        compress_messages_for_context(messages, 32_000, 256, 400, Some(overflow_dir.clone()));

    // 最近这条 read_file 结果既不应被外溢成 stub，也不应被裁剪：逐字可见。
    assert!(
        compressed.iter().all(|m| {
            m.content
                .as_str()
                .map(|s| !s.contains("Output preserved for non-compressible tool"))
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

fn extract_stub_file_path(stub: &str) -> Option<String> {
    const PREFIX: &str = "[[PRESERVED_CONTENT_STUB_V1]]";
    let payload = stub.strip_prefix(PREFIX)?;
    let value = serde_json::from_str::<Value>(payload).ok()?;
    value.get("file_path")?.as_str().map(str::to_string)
}

fn read_file_call_pair(id: &str, path: &str, content: &str) -> (Message, Message) {
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: format!(r#"{{"filePath":"{path}"}}"#),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };
    let tool = Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    };
    (assistant, tool)
}

#[test]
fn compression_collapses_byte_identical_repeated_read_file_but_keeps_changed_versions() {
    // 回归测试：断开"重复整篇重读"失忆环。
    // 同一文件被反复 read_file 且**内容逐字节相同**时，只应保留一份全文，
    // 其余冗余副本折叠为回指 stub（无损）。而内容确实变化的版本（文件被编辑）
    // 必须每个都完整保留——绝不能因签名相同而误折叠成 stub 丢失真实差异。
    let identical = format!("// agent_adapter.py\n{}", "A".repeat(6_000));
    let changed_v1 = format!("// controller.py v1\n{}", "B".repeat(6_000));
    let changed_v2 = format!("// controller.py v2 EDITED\n{}", "C".repeat(6_000));

    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 6 次读取同一文件、内容完全相同（模拟失忆环里的重复整篇重读）。
    for i in 0..6 {
        let (a, t) = read_file_call_pair(
            &format!("call_same_{i}"),
            "/repo/agent_adapter.py",
            &identical,
        );
        messages.push(a);
        messages.push(t);
    }
    // 同一文件被编辑，产生两个内容不同的版本，各读一次。
    let (a1, t1) = read_file_call_pair("call_ctrl_1", "/repo/controller.py", &changed_v1);
    messages.push(a1);
    messages.push(t1);
    let (a2, t2) = read_file_call_pair("call_ctrl_2", "/repo/controller.py", &changed_v2);
    messages.push(a2);
    messages.push(t2);

    // 末尾追加足够多的"近端"tool 消息，把上面所有读取推出 KEEP_RECENT 保护窗，
    // 确保 dedup 真正作用到它们身上。每条内容/路径都唯一，避免自身被 dedup 折叠。
    for i in 0..8 {
        let (a, t) = read_file_call_pair(
            &format!("call_pad_{i}"),
            &format!("/repo/pad_{i}.py"),
            &format!("padding-{i}"),
        );
        messages.push(a);
        messages.push(t);
    }

    // overflow_dir=None：隔离 dedup 行为，避免保留下来的那一份全文再被 offload
    // 到磁盘 stub（offload 阈值 480 字符是另一条正交路径，已有专门测试覆盖）。
    let compressed = compress_messages_for_context(messages, 200_000, 256, 400, None);

    let full_identical = compressed
        .iter()
        .filter(|m| m.content.as_str() == Some(identical.as_str()))
        .count();
    assert_eq!(
        full_identical, 1,
        "byte-identical repeated read_file must collapse to exactly one full copy"
    );

    let dedup_stubs = compressed
        .iter()
        .filter(|m| {
            m.content
                .as_str()
                .map(|s| s.contains("byte-identical") && s.contains("No need to re-read"))
                .unwrap_or(false)
        })
        .count();
    assert_eq!(
        dedup_stubs, 5,
        "the other five identical reads must become re-read-suppressing dedup stubs"
    );

    // 两个内容不同的版本都必须完整保留，绝不因签名相同而被折叠。
    assert!(
        compressed
            .iter()
            .any(|m| m.content.as_str() == Some(changed_v1.as_str())),
        "changed file version 1 must be preserved verbatim"
    );
    assert!(
        compressed
            .iter()
            .any(|m| m.content.as_str() == Some(changed_v2.as_str())),
        "changed file version 2 must be preserved verbatim"
    );
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
        compress_messages_for_context(messages, 2_000, 256, 400, Some(overflow_dir.clone()));

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
        compress_messages_for_context(messages, 2_000, 256, 400, Some(overflow_dir.clone()));

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

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn mid_turn_compress_preserves_latest_user_message() {
    let latest_user = "请继续修复 request.rs 的流式中断问题";
    let filler = "x".repeat(12000);
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
            content: Value::String("早期需求：实现 streaming".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
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
            content: Value::String("收到，我继续处理".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, before, after) = mid_turn_compress(messages, 4000, None);
    assert!(after <= before, "compression should not expand payload");

    let has_latest_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(latest_user));
    assert!(
        has_latest_user,
        "mid-turn compression must preserve the latest user message"
    );
}

#[test]
fn mid_turn_compress_spills_non_compressible_outputs_when_overflow_dir_present() {
    // 回归测试：mid-turn 压缩传入 overflow_dir 后，read_file 等「不可压缩」
    // 工具的大输出应被零压缩外溢到会话文件 + 留预览 stub，从而真正降低字符数。
    // 历史 bug：mid-turn 走 None overflow_dir，这类输出既不能裁剪也不能外溢，
    // 只能原样堆在上下文里，导致每轮只压掉零星几 K（用户报告"压不动"）。
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-midturn-overflow-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    // 10 组 read_file 调用，每条结果 8000 字符，远端组（非最近 6 条）应被外溢。
    for i in 0..10usize {
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
                        i + 40
                    ),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("y".repeat(8000)),
            tool_calls: None,
            tool_call_id: Some(id),
            reasoning_content: None,
        });
    }

    let before = messages
        .iter()
        .map(|m| m.content.as_str().map(|s| s.chars().count()).unwrap_or(0))
        .sum::<usize>();
    let (compressed, reported_before, reported_after) =
        mid_turn_compress(messages, 36_000, Some(overflow_dir.as_path()));

    assert!(reported_before >= before);
    assert!(
        reported_after < reported_before,
        "mid-turn compression with overflow_dir must shrink payload \
         (before={reported_before}, after={reported_after})"
    );
    // 应出现 read_file 外溢 stub，且其指向的文件真实存在（全文零压缩保存）。
    let stub = compressed
        .iter()
        .find_map(|m| {
            let text = m.content.as_str()?;
            text.contains("Output preserved for non-compressible tool `read_file`")
                .then_some(text.to_string())
        })
        .expect("expected preserved read_file overflow stub in mid-turn compression");
    let file_path = stub
        .lines()
        .find_map(|line| line.trim().strip_prefix("- file_path: "))
        .expect("stub should contain overflow file path");
    assert!(
        std::path::Path::new(file_path).exists(),
        "overflow file from mid-turn stub should exist: {file_path}"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn large_image_does_not_evict_tool_history_from_budget() {
    // 回归测试：一张大 base64 图片不应把 agent 的工具结果（工作记忆）
    // 挤出上下文。历史 bug：value_len_chars 按 base64 文本长度计费，
    // 一张 ~900K 字符的图片让 messages_total_chars 暴涨，压缩管线每轮
    // 都把工具结果删掉 -> agent 失忆 -> 反复重复同样的探索。
    let huge_base64 = "A".repeat(900_000);
    let image_content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": { "url": format!("data:image/png;base64,{huge_base64}") }
        }
    ]);

    let messages = vec![
        Message {
            role: "user".to_string(),
            content: image_content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("我先探索代码结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String(
                "code_search 结果：found memo.rs at src/bin/re/memo".to_string(),
            ),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("继续实现".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    // soft_threshold 36K：若图片仍按 base64 长度计费（900K），会判为超额并
    // 触发压缩，把 tool 结果删掉。修复后图片仅计 ~1K，总预算远低于阈值。
    let (compressed, before, after) = mid_turn_compress(messages, 36_000, None);
    assert!(
        before <= 36_000,
        "image must not dominate the char budget (got {before})"
    );
    assert_eq!(
        before, after,
        "no compression should trigger for image-only payload"
    );

    let kept_tool_result = compressed.iter().any(|m| {
        m.role == "tool"
            && m.content.as_str() == Some("code_search 结果：found memo.rs at src/bin/re/memo")
    });
    assert!(
        kept_tool_result,
        "tool result (agent working memory) must survive; otherwise the agent re-explores"
    );

    let image_intact = compressed.iter().any(|m| {
        m.content
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("image_url"))
            .and_then(|iu| iu.get("url"))
            .and_then(|u| u.as_str())
            .map(|u| u.len() > 100_000)
            .unwrap_or(false)
    });
    assert!(
        image_intact,
        "image content itself must remain zero-compressed"
    );
}

#[test]
fn mid_turn_compress_preserves_recent_two_user_messages() {
    let previous_user = "先定位 streaming 中断的根因";
    let latest_user = "再补上修复并验证回归";
    let filler = "x".repeat(14_000);
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
            content: Value::String("更早需求：梳理模块结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(previous_user.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
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
            content: Value::String("收到，我会按顺序处理".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, before, after) = mid_turn_compress(messages, 4_000, None);
    assert!(after <= before, "compression should not expand payload");

    let has_previous_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(previous_user));
    let has_latest_user = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(latest_user));

    assert!(
        has_previous_user,
        "mid-turn compression must preserve the previous user turn"
    );
    assert!(
        has_latest_user,
        "mid-turn compression must preserve the latest user turn"
    );
}

#[test]
fn mid_turn_compress_prefers_three_recent_user_turns_when_context_is_small_enough() {
    let user2 = "第二阶段：定位流式卡住点";
    let user3 = "第三阶段：修复压缩策略";
    let user4 = "第四阶段：补测试并复盘";
    let filler = "x".repeat(8_000);

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
            content: Value::String("第一阶段：读取代码结构".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user2.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user3.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(filler),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(user4.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("收到，按 2->3->4 顺序执行".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, _before, _after) = mid_turn_compress(messages, 4_000, None);

    let has_user2 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user2));
    let has_user3 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user3));
    let has_user4 = compressed
        .iter()
        .any(|m| m.role == "user" && m.content.as_str() == Some(user4));

    assert!(
        has_user2,
        "should preserve previous-2 user turn when context is moderate"
    );
    assert!(
        has_user3,
        "should preserve previous-1 user turn when context is moderate"
    );
    assert!(
        has_user4,
        "should preserve latest user turn when context is moderate"
    );
}

#[test]
fn mid_turn_compress_keeps_tool_pairs_consistent() {
    let huge = "x".repeat(18_000);
    let tool_call = ToolCall {
        id: "call_pair_1".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{\"file_path\":\"src/main.rs\"}".to_string(),
        },
    };

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
            content: Value::String("先分析历史错误".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![tool_call.clone()]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String(huge.clone()),
            tool_calls: None,
            tool_call_id: Some(tool_call.id.clone()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("我继续排查".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("请继续修复并验证".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(huge),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let (compressed, _before, _after) = mid_turn_compress(messages, 4_000, None);

    let mut assistant_tool_ids = SkipSet::default();
    for message in &compressed {
        if message.role == "assistant" {
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    assistant_tool_ids.insert(call.id.clone());
                }
            }
        }
    }

    let mut tool_message_ids = SkipSet::default();
    for message in &compressed {
        if message.role == "tool" {
            if let Some(id) = &message.tool_call_id {
                tool_message_ids.insert(id.clone());
            }
        }
    }

    for id in &assistant_tool_ids {
        assert!(
            tool_message_ids.contains(id),
            "assistant tool_call '{id}' must have a paired tool message"
        );
    }

    for id in &tool_message_ids {
        assert!(
            assistant_tool_ids.contains(id),
            "tool message '{id}' must be referenced by an assistant tool_call"
        );
    }
}

#[test]
fn session_delete_cleans_up_overflow_history_file() {
    let session_id = format!("test-{}", uuid::Uuid::new_v4());
    let history_file = std::env::temp_dir().join(format!(
        "ai-session-cleanup-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file(&session_id);
    let assets = store.session_assets_dir(&session_id);
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(
        assets.join("overflow-history.md"),
        "# test overflow content",
    )
    .unwrap();
    let preserved_tool_dir = assets.join("tool-overflow-compressed");
    std::fs::create_dir_all(&preserved_tool_dir).unwrap();
    std::fs::write(
        preserved_tool_dir.join("read_file-test.txt"),
        "temporary preserved tool output",
    )
    .unwrap();
    std::fs::write(&db, b"test").unwrap();

    assert!(assets.join("overflow-history.md").exists());

    store.delete_session(&session_id).unwrap();

    assert!(
        !assets.exists(),
        "assets dir (including overflow file) should be deleted"
    );
    assert!(!db.exists(), "sqlite file should be deleted");

    let _ = std::fs::remove_dir_all(store.session_assets_dir("__cleanup__"));
}

/// `temp_dir()` 现在与 tool-overflow 同源，落在 `session_assets_dir/tmp/` 下。
/// 验证 `delete_session` 会连同 `tmp/temp_registry.json` 一起递归清理。
#[test]
fn session_delete_cleans_up_temp_registry() {
    let session_id = format!("test-{}", uuid::Uuid::new_v4());
    let history_file =
        std::env::temp_dir().join(format!("ai-temp-cleanup-{}.sqlite", uuid::Uuid::new_v4()));
    let store = SessionStore::new(&history_file);
    store.ensure_root_dir().unwrap();

    let db = store.session_history_file(&session_id);
    let assets = store.session_assets_dir(&session_id);
    let tmp_dir = assets.join("tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    // 模拟 write_file(temp=true) 写入的临时文件 + 注册表
    std::fs::write(tmp_dir.join("scratch.rs"), "fn main() {}").unwrap();
    std::fs::write(tmp_dir.join("temp_registry.json"), r#"["scratch.rs"]"#).unwrap();
    std::fs::write(&db, b"test").unwrap();

    assert!(tmp_dir.join("temp_registry.json").exists());
    assert!(tmp_dir.join("scratch.rs").exists());

    store.delete_session(&session_id).unwrap();

    assert!(
        !assets.exists(),
        "assets dir (including tmp/) should be deleted"
    );
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
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": "hello"
            })]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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
        assert_eq!(
            loaded.first().unwrap().role,
            crate::ai::history::ROLE_INTERNAL_NOTE
        );
        assert!(
            loaded
                .first()
                .and_then(|m| m.content.as_str())
                .unwrap_or_default()
                .contains("历史摘要")
        );
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

    let context = build_context_history(32, &path, 6000, 32, 2000, None).unwrap();

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
        append_history_messages(
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
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: format!("call_{i}"),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "find_path".to_string(),
                            arguments: format!(r#"{{"query":"issue-{i}"}}"#),
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

    let context = build_context_history(2, &path, 1_800, 2, 1_000, None).unwrap();
    let summary = context
        .first()
        .and_then(|m| m.content.as_str())
        .unwrap_or_default()
        .to_string();
    assert!(summary.contains("已知工具结论"));
    assert!(summary.contains("find_path"));
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
    assert_eq!(
        second.last().unwrap().content,
        serde_json::Value::String("two".to_string())
    );

    let _ = std::fs::remove_file(path);
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

    let context = build_context_history(2, &path, 10_000, 2, 2_000, None).unwrap();
    assert_eq!(context[0].role, crate::ai::history::ROLE_INTERNAL_NOTE);
    assert!(
        context[0]
            .content
            .as_str()
            .unwrap_or_default()
            .contains("older summary")
    );

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
            ..Default::default()
        }],
        ..Default::default()
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
            ..Default::default()
        }],
        ..Default::default()
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

    // 流式表格渲染：表头行进入静默缓冲（等待分隔行确认是否成表），缓冲期间
    // 输出"生成表格中"占位预览，不直接 echo 原始文本——否则最终成表时会与
    // 渲染后的表头重复打印。
    let header_out = renderer.consume_line("| name | value |", false);
    assert!(header_out.contains("\x1b["));
    assert!(!header_out.contains("| name | value |"));

    // 分隔行确认成表后，表内容继续静默缓冲（不再 echo 原始行）。
    let sep_out = renderer.consume_line("| --- | --- |", true);
    assert_eq!(sep_out, "");
    let row_out = renderer.consume_line("| foo | bar |", true);
    assert_eq!(row_out, "");

    // 表格结束（非表格行 "done"）时，先清除占位预览，再一次性渲染完整表格。
    let end_out = renderer.consume_line("done", false);
    // 清除占位预览的 ANSI 序列
    assert!(end_out.contains("\x1b[1A"));
    // 渲染后的表格包含表头与数据，但原始 markdown 文本不单独出现
    assert!(end_out.contains("name"));
    assert!(end_out.contains("value"));
    assert!(end_out.contains("foo"));
    assert!(end_out.contains("bar"));
    assert!(!end_out.contains("| name | value |"));
    assert!(!end_out.contains("| --- | --- |"));
    assert!(!end_out.contains("| foo | bar |"));
    // 表格后的普通文本正常输出
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
fn execute_command_blocks_git_destructive_to_uncommitted() {
    // checkout：会丢弃工作树改动
    assert!(tools::validate_execute_command("git checkout -- src/main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout -- .").is_err());
    assert!(tools::validate_execute_command("git checkout .").is_err());
    assert!(tools::validate_execute_command("git checkout -f main").is_err());
    assert!(tools::validate_execute_command("git checkout --force main").is_err());
    assert!(tools::validate_execute_command("git -C /repo checkout -- x").is_err());
    // checkout：纯分支切换，git 会保护未提交改动，放行
    assert!(tools::validate_execute_command("git checkout main").is_ok());
    assert!(tools::validate_execute_command("git checkout -b feature/x").is_ok());
    // checkout：-B 会强制重置并切分支，丢弃未提交改动
    assert!(tools::validate_execute_command("git checkout -B main").is_err());
    assert!(tools::validate_execute_command("git checkout --force-create main").is_err());
    // checkout：无 -- 但路径启发式判定为文件（含扩展名）
    assert!(tools::validate_execute_command("git checkout src/main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout README.md").is_err());
    assert!(tools::validate_execute_command("git checkout main.rs").is_err());
    assert!(tools::validate_execute_command("git checkout archive.tar.gz").is_err());
    // checkout：不含扩展名的参数不误拦（分支/tag 形态）
    assert!(tools::validate_execute_command("git checkout main").is_ok());
    assert!(tools::validate_execute_command("git checkout v1.2.3").is_ok());
    assert!(tools::validate_execute_command("git checkout feature/x").is_ok());

    // restore：默认恢复工作树会丢弃未提交改动
    assert!(tools::validate_execute_command("git restore src/main.rs").is_err());
    assert!(tools::validate_execute_command("git restore --worktree src/main.rs").is_err());
    assert!(tools::validate_execute_command("git restore --source=HEAD~1 src/main.rs").is_err());
    // restore：仅取消暂存，工作树不动，放行
    assert!(tools::validate_execute_command("git restore --staged src/main.rs").is_ok());
    assert!(tools::validate_execute_command("git restore --staged --source=HEAD src/main.rs").is_ok());

    // reset：--hard/--merge/--keep 丢弃未提交改动
    assert!(tools::validate_execute_command("git reset --hard").is_err());
    assert!(tools::validate_execute_command("git reset --hard HEAD~1").is_err());
    assert!(tools::validate_execute_command("git reset --merge").is_err());
    assert!(tools::validate_execute_command("git reset --keep").is_err());
    // reset：--soft / 默认(mixed) 保留工作树，放行
    assert!(tools::validate_execute_command("git reset --soft HEAD~1").is_ok());
    assert!(tools::validate_execute_command("git reset").is_ok());
    assert!(tools::validate_execute_command("git reset HEAD~1").is_ok());

    // clean：-f 删除未跟踪文件，不可回滚
    assert!(tools::validate_execute_command("git clean -f").is_err());
    assert!(tools::validate_execute_command("git clean -fd").is_err());
    assert!(tools::validate_execute_command("git clean --force").is_err());
    // clean：dry-run 等不实际删除，放行
    assert!(tools::validate_execute_command("git clean -n").is_ok());

    // switch：force 切分支会丢弃未提交改动
    assert!(tools::validate_execute_command("git switch -f main").is_err());
    assert!(tools::validate_execute_command("git switch --force main").is_err());
    assert!(tools::validate_execute_command("git switch --discard-changes main").is_err());
    assert!(tools::validate_execute_command("git switch -C fix").is_err());
    assert!(tools::validate_execute_command("git switch --force-create fix").is_err());
    // switch：纯创建/切换分支，git 保护未提交改动，放行
    assert!(tools::validate_execute_command("git switch main").is_ok());
    assert!(tools::validate_execute_command("git switch -c feature/x").is_ok());

    // 经间接包装器（env/xargs）也需拦截
    assert!(tools::validate_execute_command("env git checkout -- x").is_err());
    assert!(tools::validate_execute_command("xargs git reset --hard").is_err());
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
    let payload =
        r#"{"choices":[{"delta":{"content":"","reasoning":{"confidence":0.9,"thinking":true}}}]}"#;
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
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"reasoning_text","delta":"No"}}}]}"#;
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
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "from reasoning field"
    );
    parsed.merge_reasoning();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "from reasoning field"
    );
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
    assert_eq!(merge_reasoning_fragments("注意", "！危险"), "注意！危险");
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
    let args: Value =
        serde_json::from_str(&parsed.choices[0].delta.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["file"], "a.rs");
    assert_eq!(args["patch"], "...");
}
