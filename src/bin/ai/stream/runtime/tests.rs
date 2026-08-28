use super::super::inline_recovery::normalize_tool_call_arguments;
use super::*;
use crate::ai::{
    cli::ParsedCli,
    tools::os_tools::{GLOBAL_OS, init_os_tools_globals},
    types::{App, AppConfig},
};
use std::io::Read as _;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool, mpsc};

const REPORTED_FULLWIDTH_DSML_TOOL_CALL: &str = r#"<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="read_file">
<｜｜DSML｜｜parameter name="file_path" string="true">/Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/iteration.rs</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="limit" string="false">80</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="offset" string="false">110</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>"#;

#[test]
fn prompt_cache_metrics_none_without_hit() {
    assert_eq!(format_prompt_cache_metrics(1000, 0), None);
    assert_eq!(format_prompt_cache_metrics(0, 0), None);
}

#[test]
fn prompt_cache_metrics_reports_hit_rate() {
    let line = format_prompt_cache_metrics(1000, 750).unwrap();
    assert_eq!(line, "↳ cache · 750/1.0k tokens · 75% hit");

    let large = format_prompt_cache_metrics(59_798, 59_648).unwrap();
    assert_eq!(large, "↳ cache · 59.6k/59.8k tokens · 100% hit");
}

#[test]
fn terminal_dedupe_recognizes_exact_replayed_tool_round_narration() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "结论已经在工具调用前展示。".to_string(),
        buffered_terminal_output: "结论已经在工具调用前".to_string(),
    });

    assert!(terminal_dedupe_still_matches(&state));
    assert!(!terminal_dedupe_buffer_is_complete_match(&state));

    let dedupe = state.render.terminal_dedupe.as_mut().unwrap();
    dedupe.buffered_terminal_output.push_str("展示。");
    state.content.assistant_text = dedupe.buffered_terminal_output.clone();

    assert!(terminal_dedupe_buffer_is_complete_match(&state));
    assert!(final_assistant_matches_terminal_dedupe(&state));
}

#[test]
fn terminal_dedupe_ignores_digest_blocks_in_final_assistant_text() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "结论已经展示。".to_string(),
        buffered_terminal_output: "结论已经展示。".to_string(),
    });
    state.content.assistant_text = format!(
        "结论已经展示。{}内部图片摘要{}",
        crate::ai::request::DIGEST_BEGIN,
        crate::ai::request::DIGEST_END
    );

    assert!(final_assistant_matches_terminal_dedupe(&state));
}

#[test]
fn terminal_dedupe_releases_content_after_visible_divergence() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "旧结论".to_string(),
        buffered_terminal_output: "新结论".to_string(),
    });

    assert!(!terminal_dedupe_still_matches(&state));
    assert!(!terminal_dedupe_buffer_is_complete_match(&state));
}

#[test]
fn waiting_hint_tool_name_is_single_line_and_terminal_safe() {
    assert_eq!(
        sanitize_waiting_hint_tool_name("apply_\x1b[31mpatch\n next\tstep"),
        "apply_patch next step"
    );
    assert_eq!(sanitize_waiting_hint_tool_name("\n\t"), "tool");
}

#[test]
fn idle_timeout_discards_unconfirmed_tool_call_and_marks_stream_error() {
    let mut app = test_app();
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut current_history = String::new();
    state.content.stream_idle_timed_out = true;
    state.content.tool_calls_map.insert(
        0,
        ToolCallBuilder {
            id: "call-timeout".to_string(),
            tool_type: "function".to_string(),
            function_name: "apply_patch".to_string(),
            arguments: r#"{"patch":"partial but currently valid"}"#.to_string(),
            printed_arguments_len: 0,
        },
    );

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.stream_error);
    assert!(result.tool_calls.is_empty());
}

#[test]
fn tool_arg_cap_discards_unconfirmed_tool_call_and_marks_truncated() {
    // When accumulated tool arguments exceed the cap and get cut off, the half-finished tool call
    // must not reach the execution layer even if the JSON happens to be valid at the cutoff instant
    // (same principle as idle timeout): drop it and take degenerate_repetition's retryable Truncated path instead of executing partial args.
    let mut app = test_app();
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut current_history = String::new();
    state.content.finish_reason_seen = true;
    state.content.finish_reason_value = Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
    state.content.tool_args_cap_exceeded = true;
    state.content.tool_calls_map.insert(
        0,
        ToolCallBuilder {
            id: "call-cap".to_string(),
            tool_type: "function".to_string(),
            function_name: "apply_patch".to_string(),
            // Even when the JSON is valid at the cutoff instant, it must not be executed.
            arguments: r#"{"patch":"partial but currently valid"}"#.to_string(),
            printed_arguments_len: 0,
        },
    );
    // JSON in assistant_text that looks like an inline tool call (recognized by recover_inline_tool_calls)
    // is equally untrusted in this over-limit scenario and must not be recovered for execution, otherwise it would bypass the drop logic above.
    state.content.assistant_text =
        r#"{"function":{"name":"apply_patch","arguments":"{}"},"id":"call-recover"}"#.to_string();

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.tool_calls.is_empty());
}

#[test]
fn degenerate_reasoning_repetition_requires_three_long_contentful_copies() {
    let phrase = "需要先确认当前上下文是否仍然有效，然后再继续执行。";
    assert!(!has_degenerate_repetition(&phrase.repeat(2)));
    assert!(has_degenerate_repetition(&phrase.repeat(3)));
    assert!(!has_degenerate_repetition(&"----------------".repeat(3)));
}

#[test]
fn degenerate_reasoning_repetition_detects_suffix_after_normal_progress() {
    let phrase = "I need to inspect the existing implementation before changing it. ";
    let reasoning = format!(
        "First I will locate the relevant module. {}",
        phrase.repeat(3)
    );
    assert!(has_degenerate_repetition(&reasoning));
}

#[test]
fn degenerate_repetition_catches_visible_content_runaway() {
    // Reproduces an incident: the model repeated the same phrase verbatim in its **visible output**
    // until the budget was full, producing a giant junk message on disk and triggering a provider 400
    // on the next turn. The degenerate guard must also apply to visible assistant text (previously it only covered reasoning_content).
    let phrase = "我再重新读一遍修复区域，以确保我掌握的是当前状态。";
    assert!(has_degenerate_repetition(&phrase.repeat(3)));
    // Single-character repetition (e.g. the 80,000 repetitions of one character in the incident) must also match.
    assert!(has_degenerate_repetition(&"再".repeat(64)));
}

#[test]
fn thinking_fold_defaults_to_configured_lines_for_tty() {
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, None),
        DEFAULT_THINKING_MAX_VISIBLE_LINES
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("12")),
        12
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("0")),
        usize::MAX
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(true, Some("oops")),
        DEFAULT_THINKING_MAX_VISIBLE_LINES
    );
    assert_eq!(
        resolve_thinking_fold_max_visible_lines(false, Some("12")),
        usize::MAX
    );
}

#[test]
fn stream_text_event_to_content_ignores_thinking_events() {
    let mut markers = StreamMarkers::new();
    markers.enable_subagent_preview("build");

    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::OpenThinking,
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::AppendThinking("step one".to_string()),
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::AppendContent("final answer".to_string()),
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        Some("final answer".to_string())
    );
    assert_eq!(
        stream_text_event_to_content(
            &StreamTextEvent::CloseThinking,
            &markers,
            StreamEventMergeMode::Append,
            "",
        ),
        None
    );
}

fn test_app() -> App {
    App {
        cli: ParsedCli::default(),
        hooks: Default::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 0,
            history_keep_last: 0,
            history_summary_max_chars: 0,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: String::new(),
        current_agent: String::new(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skills: Vec::new(),
        forced_skill_source: None,
        pending_skill_continuation: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown: Arc::new(AtomicBool::new(false)),
        streaming: Arc::new(AtomicBool::new(false)),
        cancel_stream: Arc::new(AtomicBool::new(false)),
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
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares: Vec::new(),
        llm_middlewares: Vec::new(),
    }
}

#[tokio::test]
async fn wait_for_interrupt_observes_request_interrupt_source() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = wait_for_interrupt(&app);
    let trigger = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        crate::ai::driver::signal::signal_request_interrupt();
    };

    tokio::join!(waiter, trigger);
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn wait_for_interrupt_or_timeout_returns_true_on_request_interrupt() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = tokio::spawn(async move {
        wait_for_interrupt_or_timeout(&app, Some(Duration::from_secs(5))).await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    crate::ai::driver::signal::signal_request_interrupt();

    let interrupted = tokio::time::timeout(Duration::from_millis(200), waiter)
        .await
        .expect("stream retry wait should wake on interrupt")
        .expect("waiter should complete");
    assert!(interrupted);

    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn closing_thinking_marker_starts_on_new_line_when_reasoning_line_is_open() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking", true)
        .unwrap();
    let mut content = format!("{}\nfinal", markers.end_thinking_tag);
    if content.starts_with(&markers.end_thinking_tag) && state.render.markdown.has_unfinished_line()
    {
        content.insert(0, '\n');
    }

    assert_eq!(content, format!("\n{}\nfinal", markers.end_thinking_tag));
}

#[test]
fn closing_thinking_marker_keeps_compact_spacing_when_already_at_line_start() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking\n", true)
        .unwrap();
    let mut content = format!("{}\nfinal", markers.end_thinking_tag);
    normalize_end_thinking_boundary(&mut content, &markers, &state.render.markdown);

    assert_eq!(content, format!("{}\nfinal", markers.end_thinking_tag));
}

#[test]
fn tool_call_boundary_closes_thinking_on_a_fresh_line() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking", true)
        .unwrap();

    assert_eq!(
        format_end_thinking_line(&markers, &state.render.markdown),
        format!("\n{}\n", markers.end_thinking_tag)
    );
}

#[test]
fn snapshot_content_only_appends_missing_suffix() {
    assert_eq!(unseen_suffix("hello wor", "hello world"), "ld");
    assert_eq!(unseen_suffix("hello world", "hello world"), "");
    assert_eq!(unseen_suffix("hello world", "\n\nhello world"), "");
    assert_eq!(unseen_suffix("hello world", "\n\nhello world!"), "!");
    assert_eq!(unseen_suffix("prefix", "suffix"), "suffix");
}

#[test]
fn tool_call_render_chunk_only_streams_unprinted_suffix() {
    let mut builder = ToolCallBuilder::default();

    builder.arguments.push_str("{\"patch\":\"a");
    assert!(take_tool_call_render_chunk(None, 0, &mut builder).is_none());

    builder.function_name = "apply_patch".to_string();
    let first = take_tool_call_render_chunk(None, 0, &mut builder).unwrap();
    assert!(first.open_line);
    assert_eq!(first.function_name, "apply_patch");
    assert_eq!(first.arguments, "{\"patch\":\"a");

    builder.arguments.push('你');
    let second = take_tool_call_render_chunk(Some(0), 0, &mut builder).unwrap();
    assert!(!second.open_line);
    assert_eq!(second.arguments, "你");
}

#[test]
fn normalize_tool_call_arguments_rejects_incomplete_json_and_canonicalizes_empty() {
    assert_eq!(normalize_tool_call_arguments(""), Some("{}".to_string()));
    assert_eq!(
        normalize_tool_call_arguments(" {\"command\":\"pwd\"} "),
        Some("{\"command\":\"pwd\"}".to_string())
    );
    assert_eq!(normalize_tool_call_arguments("{\"command\":"), None);
}

#[test]
fn collect_valid_tool_calls_reports_drop_on_incomplete_arguments() {
    let mut builders: rust_tools::cw::SkipMap<usize, ToolCallBuilder> =
        rust_tools::cw::SkipMap::default();
    // Simulate a large write_file hitting the output cap: arguments JSON cut in half and unrepairable.
    builders.insert(
        0,
        ToolCallBuilder {
            function_name: "write_file".to_string(),
            arguments: "{\"path\":\"/tmp/x\",\"content\":\"aaa".to_string(),
            ..Default::default()
        },
    );
    let (calls, dropped) = collect_valid_tool_calls(&mut builders);
    assert!(calls.is_empty(), "半截 JSON 应被丢弃");
    assert!(dropped, "发生丢弃时应返回 dropped=true");
}

#[test]
fn collect_valid_tool_calls_no_drop_on_valid_arguments() {
    let mut builders: rust_tools::cw::SkipMap<usize, ToolCallBuilder> =
        rust_tools::cw::SkipMap::default();
    builders.insert(
        0,
        ToolCallBuilder {
            function_name: "read_file".to_string(),
            arguments: "{\"path\":\"/tmp/x\"}".to_string(),
            ..Default::default()
        },
    );
    let (calls, dropped) = collect_valid_tool_calls(&mut builders);
    assert_eq!(calls.len(), 1);
    assert!(!dropped, "合法 JSON 不应触发 dropped");
}

#[test]
fn recover_inline_tool_calls_handles_bare_object() {
    // Simulate qwen3.7-max emitting a tool call as content.
    let raw = r#"{"name":"read_file","arguments":{"path":"/tmp/x"}}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
    assert_eq!(calls[0].tool_type, "function");
}

#[test]
fn recover_inline_tool_calls_handles_arguments_as_json_string() {
    let raw = r#"{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_fenced_code_block() {
    let raw = "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp/x\"}}\n```";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].function.name, "read_file");
}

#[test]
fn recover_inline_tool_calls_handles_tool_call_xml_wrapper() {
    let raw = r#"<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].function.name, "read_file");
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_json_body() {
    // The Hermes/Qwen XML shape the model actually emitted in the screenshot (body is JSON).
    let raw = "<tool_call>\n<function=read_file>\n{\"path\":\"/tmp/x\"}\n</function>\n</tool_call>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_parameter_tags() {
    let raw = "<function=read_file><parameter=path>/tmp/x</parameter><parameter=limit>200</parameter></function>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    // Numeric arguments must be recognized as JSON numbers, not strings.
    assert_eq!(args["limit"], 200);
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_no_args() {
    let raw = "<tool_call><function=list_agents></function></tool_call>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_agents");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn recover_inline_tool_calls_handles_hermes_xml_parallel_calls() {
    let raw = "<function=read_file>{\"path\":\"/a\"}</function><function=read_file>{\"path\":\"/b\"}</function>";
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.arguments, r#"{"path":"/a"}"#);
    assert_eq!(calls[1].function.arguments, r#"{"path":"/b"}"#);
}

#[test]
fn recover_inline_tool_calls_handles_array_of_calls() {
    let raw = r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "a");
    assert_eq!(calls[1].function.name, "b");
    assert_eq!(calls[1].function.arguments, r#"{"x":1}"#);
}

#[test]
fn recover_inline_tool_calls_handles_openai_function_wrapper() {
    let raw = r#"{"id":"call_123","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls[0].id, "call_123");
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
}

#[test]
fn recover_inline_tool_calls_rejects_plain_text() {
    // Plain text answers must never be misidentified as a tool call.
    assert!(recover_inline_tool_calls("Hello world").is_none());
    assert!(recover_inline_tool_calls("").is_none());
    // name without arguments, and name not in the known object set — strictly this should not parse,
    // but for compatibility we still recognize a bare name on its own. Below are the true negative samples:
    assert!(recover_inline_tool_calls("{\"foo\":\"bar\"}").is_none());
    assert!(recover_inline_tool_calls("12345").is_none());
    // String-form args must themselves be valid JSON, otherwise reject.
    assert!(recover_inline_tool_calls(r#"{"name":"x","arguments":"not-json"}"#).is_none());
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_parameter_tags() {
    // The Anthropic style deepseek-v4-flash actually emits: <invoke name=...>/<parameter name=...>.
    let raw = r#"<function_calls><invoke name="read_file"><parameter name="path">/tmp/x</parameter><parameter name="limit">200</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    assert_eq!(args["limit"], 200);
}

#[test]
fn recover_inline_tool_calls_respects_anthropic_string_attr() {
    // string="true" must keep values that look like JSON scalars (true/123/null) as strings,
    // instead of auto-parsing them into bool/number. string="false" parses as JSON as usual.
    let raw = r#"<function_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="dry_run" string="true">true</parameter><parameter name="count" string="true">123</parameter><parameter name="tools" string="false">["a","b"]</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["operation"], "enable");
    assert_eq!(
        args["dry_run"], "true",
        "string=\"true\" must keep 'true' as string"
    );
    assert_eq!(
        args["count"], "123",
        "string=\"true\" must keep '123' as string"
    );
    assert_eq!(args["tools"][0], "a");
    assert_eq!(args["tools"][1], "b");
}

#[test]
fn recover_inline_tool_calls_matches_anthropic_string_attr_by_exact_name() {
    let raw = r#"<function_calls><invoke name="enable_tools"><parameter name="string_value" string="true">123</parameter><parameter name="count" notstring="true">456</parameter></invoke></function_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("expected recovered tool call");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["string_value"], "123");
    assert_eq!(args["count"], 456);
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_namespaced_tags() {
    // With a namespace prefix (antml:) and no outer wrapper.
    let raw = r#"<invoke name="list_agents"></invoke>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_agents");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn recover_inline_tool_calls_respects_anthropic_xml_string_attr() {
    // DSML `string="true"`: the value must stay a string even when it looks like a JSON scalar.
    // Covers the output shape the user reported from deepseek with MCP tools enabled.
    let raw = r#"<tool_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="tools" string="false">["mcp_excel_open_workbook"]</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "enable_tools");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // string="true" -> "enable" must remain a string, never treated as an identifier.
    assert_eq!(args["operation"], "enable");
    assert!(
        args["operation"].is_string(),
        "string=\"true\" 必须保持字符串"
    );
    // string="false" -> the array JSON parses normally.
    assert_eq!(args["tools"][0], "mcp_excel_open_workbook");
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_parallel_calls() {
    let raw = r#"<tool_calls><invoke name="read_file"><parameter name="path">/a</parameter></invoke><invoke name="read_file"><parameter name="path">/b</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
    assert_eq!(calls.len(), 2);
    let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let b: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a["path"], "/a");
    assert_eq!(b["path"], "/b");
}

#[test]
fn recover_inline_tool_calls_handles_anthropic_xml_string_true_attr() {
    // DSML `string="true"`: the value stays a string even when it looks like a JSON scalar.
    let raw = r#"<tool_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="tools" string="false">["read_file","write_file"]</parameter><parameter name="verbose" string="true">true</parameter><parameter name="count" string="true">123</parameter></invoke></tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "enable_tools");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["operation"], "enable", "string=true -> 字符串");
    assert!(args["operation"].is_string());
    assert_eq!(
        args["tools"],
        serde_json::json!(["read_file", "write_file"]),
        "string=false -> 原生 JSON 数组"
    );
    assert_eq!(
        args["verbose"], "true",
        "看起来像 bool 但 string=true -> 字符串 \"true\""
    );
    assert!(args["verbose"].is_string());
    assert_eq!(
        args["count"], "123",
        "看起来像数字但 string=true -> 字符串 \"123\""
    );
    assert!(args["count"].is_string());
}

#[test]
fn recover_inline_tool_calls_handles_bare_registered_xml_with_raw_string_body() {
    let raw = r#"<execute_command>cd /tmp && pwd</execute_command>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover bare xml tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "execute_command");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "cd /tmp && pwd");
}

#[test]
fn anthropic_xml_streamer_suppresses_markup_and_emits_events() {
    let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let (cleaned, events) = streamer.push(
            r#"Let me check.<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>"#,
        );
    // The invoke markers stay hidden; only the leading prose is kept.
    assert_eq!(cleaned, "Let me check.");
    // Emits Begin/Args/End events, consistent with the internal tool_call pipeline.
    assert_eq!(events.len(), 3);
    match (&events[0], &events[1], &events[2]) {
        (
            InternalToolCallStreamEvent::Begin(name),
            InternalToolCallStreamEvent::Args(args),
            InternalToolCallStreamEvent::End,
        ) => {
            assert_eq!(name, "read_file");
            let v: serde_json::Value = serde_json::from_str(args).unwrap();
            assert_eq!(v["path"], "/tmp/x");
        }
        _ => panic!("unexpected events: {events:?}"),
    }
}

#[test]
fn anthropic_xml_streamer_handles_split_chunks() {
    let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let mut all_events = Vec::new();
    let mut all_cleaned = String::new();
    for chunk in [
        "pre <inv",
        "oke name=\"read_file\"><parameter name=\"pa",
        "th\">/tmp/x</parameter></in",
        "voke> post",
    ] {
        let (cleaned, events) = streamer.push(chunk);
        all_cleaned.push_str(&cleaned);
        all_events.extend(events);
    }
    assert_eq!(all_cleaned, "pre  post");
    assert_eq!(all_events.len(), 3);
    match &all_events[0] {
        InternalToolCallStreamEvent::Begin(name) => assert_eq!(name, "read_file"),
        other => panic!("unexpected first event: {other:?}"),
    }
}

#[test]
fn anthropic_xml_streamer_leaves_prose_angle_brackets_intact() {
    let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
    let (cleaned, events) = streamer.push("a < b and c > d, also <div> here");
    assert_eq!(cleaned, "a < b and c > d, also <div> here");
    assert!(events.is_empty());
}

#[test]
fn response_completed_event_does_not_block_late_snapshot_text() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.completed"),
        r#"{"status":"completed"}"#,
    )
    .unwrap();
    assert!(!outcome.should_stop);
    assert!(!outcome.meaningful_progress);

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"hello world"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "hello world");
    assert_eq!(state.content.assistant_text, "hello world");
}

#[test]
fn replayed_content_part_added_does_not_duplicate_visible_text() {
    // User-visible "conclusion printed twice": a compatibility gateway re-delivers the full text of
    // content_part.added (output_text) after the output_text.delta increments. Rendering as-is in
    // Append mode duplicates the body; ReplayedChunk must compute the unseen suffix for content and render only the new part.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    // 1) delta increments render part of the body first
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"修复完成"}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "修复完成");

    // 2) content_part.added re-sends the part's full text (multi-path delivery by the protocol)
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"修复完成，验证通过。"}}"#,
    )
    .unwrap();
    // The seen prefix is swallowed; only the unseen suffix is appended
    assert_eq!(state.content.assistant_text, "修复完成，验证通过。");
    assert_eq!(current_history, "修复完成，验证通过。");

    // 3) Re-sending the exact same text: full overlap, nothing more is appended
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"修复完成，验证通过。"}}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "修复完成，验证通过。");
    assert_eq!(current_history, "修复完成，验证通过。");
}

#[test]
fn stream_payload_meaningful_progress_includes_new_reasoning_chunks() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let usage_only = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#,
    )
    .unwrap();
    assert!(!usage_only.should_stop);
    assert!(!usage_only.meaningful_progress);

    let reasoning_only = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.delta"),
        r#"{"delta":"thinking step"}"#,
    )
    .unwrap();
    assert!(!reasoning_only.should_stop);
    assert!(reasoning_only.meaningful_progress);
    assert_eq!(state.content.reasoning_text, "thinking step");
    assert!(current_history.is_empty());

    let duplicate_reasoning_snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.done"),
        r#"{"text":"thinking step"}"#,
    )
    .unwrap();
    assert!(!duplicate_reasoning_snapshot.meaningful_progress);

    let content = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"answer"}"#,
    )
    .unwrap();
    assert!(content.meaningful_progress);
    assert_eq!(current_history, "answer");

    let duplicate_snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"answer"}"#,
    )
    .unwrap();
    assert!(!duplicate_snapshot.meaningful_progress);

    let tool_call = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}]}}]}"#,
    )
    .unwrap();
    assert!(tool_call.meaningful_progress);

    let finish_reason = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    )
    .unwrap();
    assert!(finish_reason.meaningful_progress);
    assert_eq!(state.content.finish_reason_value.as_deref(), Some("stop"));
}

#[test]
fn repeated_reasoning_deltas_preserve_model_output() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    for _ in 0..2 {
        let outcome = process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.reasoning_summary_text.delta"),
            r#"{"delta":"same step"}"#,
        )
        .unwrap();
        assert!(outcome.meaningful_progress);
    }

    assert_eq!(state.content.reasoning_text, "same stepsame step");
    assert!(current_history.is_empty());
}

#[tokio::test]
async fn process_chunk_result_marks_empty_sse_as_no_meaningful_progress() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let empty_step = process_chunk_result(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Ok(Some(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":0,\"total_tokens\":1}}\n\n"
                .as_slice(),
        )),
    )
    .await
    .unwrap();
    match empty_step {
        StreamChunkStep::Continue {
            meaningful_progress,
        } => assert!(!meaningful_progress),
        _ => panic!("empty SSE should keep streaming without refreshing watchdog"),
    }

    let content_step = process_chunk_result(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Ok(Some(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n".as_slice(),
        )),
    )
    .await
    .unwrap();
    match content_step {
        StreamChunkStep::Continue {
            meaningful_progress,
        } => assert!(meaningful_progress),
        _ => panic!("content SSE should keep streaming and refresh watchdog"),
    }
    assert_eq!(current_history, "hello");
}

#[test]
fn suppressed_terminal_output_still_collects_subagent_response() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, || {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.output_text.done"),
            r#"{"text":"subagent result"}"#,
        )
        .unwrap();
    });

    assert_eq!(current_history, "subagent result");
    assert_eq!(state.content.assistant_text, "subagent result");
}

#[test]
fn output_text_done_snapshot_with_leading_whitespace_does_not_duplicate_answer() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"结论：history 文件结构本身正常。"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"\n\n结论：history 文件结构本身正常。"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "结论：history 文件结构本身正常。");
    assert_eq!(
        state.content.assistant_text,
        "结论：history 文件结构本身正常。"
    );
}

#[test]
fn output_text_stream_preserves_inter_paragraph_newlines_without_snapshot_duplication() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    // The responses protocol emits in segments: body -> bare newline -> next segment. A bare newline is part of the body format,
    // must go into assistant_text verbatim; the final .done snapshot is only for dedup and must not append the whole content again.
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"有问题，而且问题不在 a.rs 本身。"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"\n\n"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"核心是 Agent 的收敛机制过于宽松。"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"}"#,
    )
    .unwrap();

    assert_eq!(
        current_history,
        "有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"
    );
    assert_eq!(
        state.content.assistant_text,
        "有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"
    );
}

#[test]
fn process_stream_payload_suppresses_bare_registered_xml_tool_markup() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"先确认一下。<execute_command>pwd</execute_command>"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "先确认一下。");
    assert_eq!(state.content.assistant_text, "先确认一下。");
    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.function_name, "execute_command");
    assert_eq!(builder.arguments, r#"{"command":"pwd"}"#);
}

#[test]
fn opencode_message_snapshot_recovers_reported_fullwidth_dsml_before_rendering() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();
    let payload = serde_json::json!({
        "choices": [{
            "message": {
                "content": REPORTED_FULLWIDTH_DSML_TOOL_CALL
            }
        }]
    })
    .to_string();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        None,
        &payload,
    )
    .unwrap();

    assert!(current_history.is_empty());
    assert!(state.content.assistant_text.is_empty());
    assert!(state.content.hidden_meta.is_empty());
    assert_eq!(state.content.tool_calls_map.len(), 1);
    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.function_name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&builder.arguments).unwrap();
    assert_eq!(
        args["file_path"],
        "/Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/iteration.rs"
    );
    assert_eq!(args["limit"], 80);
    assert_eq!(args["offset"], 110);
}

#[test]
fn fullwidth_dsml_done_snapshot_does_not_duplicate_delta_tool_call() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    for event_type in ["response.output_text.delta", "response.output_text.done"] {
        let payload = if event_type.ends_with(".delta") {
            serde_json::json!({ "delta": REPORTED_FULLWIDTH_DSML_TOOL_CALL }).to_string()
        } else {
            serde_json::json!({ "text": REPORTED_FULLWIDTH_DSML_TOOL_CALL }).to_string()
        };
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some(event_type),
            &payload,
        )
        .unwrap();
    }

    assert!(current_history.is_empty());
    assert!(state.content.assistant_text.is_empty());
    assert_eq!(state.content.tool_calls_map.len(), 1);
    assert_eq!(state.content.internal_tool_call_idx, 1);
}

#[test]
fn inline_tool_call_fallback_does_not_persist_protocol_as_hidden_meta() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.assistant_text = REPORTED_FULLWIDTH_DSML_TOOL_CALL.to_string();
    let mut app = test_app();
    let mut current_history = String::new();

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::ToolCall);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].function.name, "read_file");
    assert!(result.assistant_text.is_empty());
    assert!(
        result.hidden_meta.is_empty(),
        "工具协议不是 self_note，不得进入 hidden_meta"
    );
}

#[test]
fn process_stream_payload_halts_and_downshifts_on_hallucinated_result_marker() {
    // Reproduces this incident: the model fabricated a "tool call -> tool result" sequence in its
    // visible body, emitting `<function_results>` protocol markers the system never generates. Requirements:
    // (1) the hallucinated result block is stripped whole and never persisted; (2) the stream stops
    // (should_stop=true); (3) degenerate_repetition finish_reason is set to take the downgrade-retry path, keeping hallucinated body text from poisoning the next request.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"我再读一遍<function_results>File: a.rs\n3 matches found</function_results>"}"#,
    )
    .unwrap();

    assert!(outcome.should_stop, "检出幻觉标记必须停流");
    assert!(
        outcome.meaningful_progress,
        "退化停流已设置 finish_reason，应视为语义进展"
    );
    assert_eq!(
        state.content.finish_reason_value.as_deref(),
        Some("degenerate_repetition"),
        "必须走 degenerate_repetition 降档重试路径"
    );
    assert!(
        !state.content.assistant_text.contains("function_results"),
        "幻觉协议标记不得落入 assistant_text：{}",
        state.content.assistant_text
    );
    assert!(
        !state.content.assistant_text.contains("matches found"),
        "幻觉结果文本不得落盘：{}",
        state.content.assistant_text
    );
}

#[test]
fn thinking_fold_keeps_reasoning_buffer_intact() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.render.thinking_fold.max_visible_lines = 2;
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_text.delta"),
        r#"{"delta":"step 1\nstep 2\nstep 3"}"#,
    )
    .unwrap();

    assert_eq!(state.content.reasoning_text, "step 1\nstep 2\nstep 3");
    assert!(state.content.thinking_open);
    assert!(current_history.is_empty());
    assert!(state.content.assistant_text.is_empty());

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"final answer"}"#,
    )
    .unwrap();

    assert_eq!(state.content.reasoning_text, "step 1\nstep 2\nstep 3");
    assert_eq!(current_history, "final answer");
    assert_eq!(state.content.assistant_text, "final answer");
    assert!(!state.content.thinking_open);
    assert!(!state.render.thinking_fold.active);
}

#[test]
fn reasoning_summary_done_snapshot_does_not_duplicate_thinking() {
    // The Responses protocol first streams `.delta` increments, then re-sends the whole reasoning
    // summary via a `.done` event (SnapshotChunk). Without unseen-suffix dedup on the snapshot, thinking prints twice.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    for delta in ["I'm considering ", "inspecting the task_tools."] {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.reasoning_summary_text.delta"),
            &format!(
                r#"{{"delta":{}}}"#,
                serde_json::Value::String(delta.to_string())
            ),
        )
        .unwrap();
    }

    assert_eq!(
        state.content.reasoning_text,
        "I'm considering inspecting the task_tools."
    );

    // The `.done` event carries the complete summary (full) — after dedup nothing more may be appended.
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.done"),
        r#"{"text":"I'm considering inspecting the task_tools."}"#,
    )
    .unwrap();

    assert_eq!(
        state.content.reasoning_text, "I'm considering inspecting the task_tools.",
        "snapshot 不应重复追加已流式过的推理摘要"
    );
}

#[test]
fn content_part_summary_text_events_never_replay_streamed_reasoning() {
    // gpt-5.5/5.6's Responses API re-sends already-streamed reasoning summaries via the summary_text
    // of content_part.added / content_part.done. These events are snapshot re-sends rather than model
    // increments and must be deduped by unseen suffix, otherwise reasoning_text accumulates twice,
    // polluting degenerate detection and possibly triggering duplicate thinking rendering.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    // First summary: stream via delta first, then re-send the same segment via content_part.added/done
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.delta"),
        r#"{"delta":"Analyzing task cancellation"}"#,
    )
    .unwrap();
    for ev in ["response.content_part.added", "response.content_part.done"] {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some(ev),
            r#"{"part":{"type":"summary_text","text":"Analyzing task cancellation"}}"#,
        )
        .unwrap();
    }
    assert_eq!(
        state.content.reasoning_text, "Analyzing task cancellation",
        "content_part 的 summary_text 重发不应重复累积 reasoning_text"
    );

    // Second summary: likewise verify that the delta + content_part re-send does not pollute
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.delta"),
        r#"{"delta":"Collecting and inspecting tasks"}"#,
    )
    .unwrap();
    for ev in ["response.content_part.added", "response.content_part.done"] {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some(ev),
            r#"{"part":{"type":"summary_text","text":"Collecting and inspecting tasks"}}"#,
        )
        .unwrap();
    }
    assert_eq!(
        state.content.reasoning_text, "Analyzing task cancellationCollecting and inspecting tasks",
        "多段摘要的 content_part 重发仍不应重复累积"
    );
}

#[test]
fn thinking_fold_drops_interior_blank_lines() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.render.thinking_fold.max_visible_lines = 8;
    state.render.thinking_fold.active = true;

    // Models often separate paragraphs with blank lines: blank lines between segments must not consume visible rows of the fold window.
    write_thinking_content_folded("para 1\n\npara 2\n", &mut state, &markers).unwrap();

    let fold = &state.render.thinking_fold;
    assert_eq!(
        fold.recent_lines.iter().collect::<Vec<_>>(),
        vec!["para 1", "para 2"]
    );
    assert_eq!(fold.total_lines, 2);
}

#[test]
fn thinking_fold_window_counts_current_line_inside_visible_budget() {
    // Lock and widen COLUMNS: this case asserts the body/markers exist verbatim, so it must avoid
    // reading a leaked narrow column width while the COLUMNS=12 wrapping case runs concurrently and triggering clamp truncation.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 3;
    fold.total_lines = 3;
    fold.recent_lines.push_back("line-1".to_string());
    fold.recent_lines.push_back("line-2".to_string());
    fold.recent_lines.push_back("line-3".to_string());
    fold.current_line = "line-4".to_string();

    assert_eq!(thinking_fold_hidden_count(fold), 1);
    assert_eq!(
        thinking_fold_visible_lines(fold),
        vec!["line-2", "line-3", "line-4"]
    );

    let (window, _) = render_thinking_fold_window(fold);
    assert_eq!(window.matches("earlier lines").count(), 1);
    assert!(!window.contains("line-1"));
    assert!(window.contains("line-2"));
    assert!(window.contains("line-3"));
    assert!(window.contains("line-4"));

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_zero_window_is_pure_summary() {
    // 0-row window = summary only: neither completed body lines nor the in-flight line enter the visible window.
    // `finalize_fold` temporarily uses this semantics when thinking wraps up, so trailing recap
    // conclusions/questions do not duplicate the final answer on screen (streaming still shows max_visible_lines).
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 0;
    fold.total_lines = 3;
    fold.recent_lines.push_back("line-1".to_string());
    fold.recent_lines.push_back("line-2".to_string());
    fold.recent_lines.push_back("line-3".to_string());
    fold.current_line = "conclusion? 需要我帮你吗".to_string();

    assert!(thinking_fold_visible_lines(fold).is_empty());

    let (window, _) = render_thinking_fold_window(fold);
    assert!(window.contains("earlier lines"));
    assert!(!window.contains("line-1"));
    assert!(!window.contains("line-2"));
    assert!(!window.contains("line-3"));
    assert!(!window.contains("conclusion"));

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_window_wraps_long_lines_to_terminal_width() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "12");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 4;
    fold.total_lines = 1;
    fold.recent_lines
        .push_back("12345678901234567890".to_string());
    fold.current_line = "abcdef".to_string();

    let (window, rows) = render_thinking_fold_window(fold);

    let plain_lines = window
        .lines()
        .map(crate::ai::stream::extract::strip_ansi_codes)
        .collect::<Vec<_>>();
    // COLUMNS=12, reserve = indent 4 -> effective width 8 columns. Long lines wrap naturally at 8 columns,
    // each wrapped segment being one physical line, and all fit within the 4-physical-line visible budget.
    assert_eq!(
        plain_lines,
        vec!["    12345678", "    90123456", "    7890", "    abcdef",]
    );
    assert_eq!(rows, 4);
    for visible in &plain_lines {
        assert!(
            visible.starts_with(THINKING_FOLD_BODY_INDENT),
            "thinking body should stay nested under header: {visible:?}"
        );
        assert!(
            unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
            "line exceeds terminal width: {visible:?}"
        );
    }

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_window_caps_wrapped_content_to_physical_row_budget() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "12");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 2;
    fold.total_lines = 1;
    fold.recent_lines
        .push_back("12345678901234567890".to_string());
    fold.current_line = "abcdef".to_string();

    let (window, rows) = render_thinking_fold_window(fold);
    let plain_lines = window
        .lines()
        .map(crate::ai::stream::extract::strip_ansi_codes)
        .collect::<Vec<_>>();

    // Even when logical lines are within budget, wrapping can exceed the physical-line budget; keep the
    // latest two lines plus a one-line notice so the cursor-up erase range stays constantly bounded.
    assert_eq!(plain_lines, vec!["    … more", "    7890", "    abcdef"]);
    assert_eq!(rows, 3);
    assert!(rows <= fold.max_visible_lines + 1);
    for visible in &plain_lines {
        assert!(
            unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
            "line exceeds terminal width: {visible:?}"
        );
    }

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn one_column_fold_content_keeps_row_accounting_safe() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "5");
    }

    // With a 4-column indent, body and fold notice have one column left. The truncation notice must still
    // be one column, wide chars render as a single-column placeholder, and the terminal must not wrap on its own or the erase count under-counts.
    assert_eq!(
        crate::ai::stream::clamp_line_to_terminal_row_with_reserve("marker", 4),
        "…"
    );

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 2;
    fold.current_line = "中a".to_string();

    let (window, rows) = render_thinking_fold_window(fold);
    let plain_lines = window
        .lines()
        .map(crate::ai::stream::extract::strip_ansi_codes)
        .collect::<Vec<_>>();

    assert_eq!(plain_lines, vec!["    ?", "    a"]);
    assert_eq!(rows, 2);
    for visible in &plain_lines {
        assert!(
            unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 5,
            "line exceeds terminal width: {visible:?}"
        );
    }

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn fold_window_keeps_last_terminal_columns_unused_without_terminal_detection() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "12");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 5;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    fold.total_lines = 1;
    fold.recent_lines
        .push_back("12345678901234567890".to_string());
    fold.current_line = "abcdef".to_string();

    let (window, rows) = render_thinking_fold_window(fold);
    let plain_lines = window
        .lines()
        .map(crate::ai::stream::extract::strip_ansi_codes)
        .collect::<Vec<_>>();

    // COLUMNS=12, reserve = indent 4 + generic right margin 2 = 6 -> effective width 6 columns.
    // Independent of TERM_PROGRAM detection, long lines always avoid the delayed-wrap column.
    assert_eq!(
        plain_lines,
        vec![
            "    123456",
            "    789012",
            "    345678",
            "    90",
            "    abcdef",
        ]
    );
    assert_eq!(rows, 5);
    assert!(
        !window.ends_with('\n'),
        "live fold body must keep the cursor on its last row"
    );
    for visible in &plain_lines {
        assert!(
            unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 10,
            "fold rewrite line reaches delayed-wrap column: {visible:?}"
        );
    }

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn fold_body_erase_starts_from_the_last_rendered_row() {
    let mut one_row = Vec::new();
    erase_fold_body(&mut one_row, 1).expect("erase one-row fold body");
    assert_eq!(one_row, b"\r\r\x1b[2K\r");

    let mut four_rows = Vec::new();
    erase_fold_body(&mut four_rows, 4).expect("erase four-row fold body");
    assert_eq!(
        four_rows,
        b"\r\x1b[3A\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[3A\r"
    );
    assert!(
        !four_rows
            .windows(b"\x1b[0J".len())
            .any(|window| window == b"\x1b[0J"),
        "bounded erase must not clear the side-note footer"
    );
}

#[test]
fn thinking_fold_window_indents_body_under_header() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 2;
    fold.total_lines = 2;
    fold.recent_lines.push_back("line-1".to_string());
    fold.current_line = "line-2".to_string();

    let (window, rows) = render_thinking_fold_window(fold);
    let plain_lines = window
        .lines()
        .map(crate::ai::stream::extract::strip_ansi_codes)
        .collect::<Vec<_>>();

    assert_eq!(
        plain_lines,
        vec!["    … 1 earlier lines", "    line-1", "    line-2"]
    );
    assert_eq!(rows, 3);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_window_without_hidden_lines_has_no_fold_marker() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.max_visible_lines = 4;
    fold.total_lines = 2;
    fold.recent_lines.push_back("line-1".to_string());
    fold.recent_lines.push_back("line-2".to_string());
    fold.current_line = "line-3".to_string();

    let (window, rows) = render_thinking_fold_window(fold);

    // No hidden lines, no active header: window physical rows == visible logical lines (3).
    assert!(!window.contains("earlier lines"));
    assert!(window.contains("line-1"));
    assert!(window.contains("line-2"));
    assert!(window.contains("line-3"));
    assert_eq!(rows, 3);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_window_body_excludes_anchored_header() {
    // The header is stripped from body rendering and anchored separately: `render_thinking_fold_window`
    // only produces body content (fold summary + visible lines), never the header. This is the core
    // invariant of the "orphan header stacking" fix — body may be erased and redrawn repeatedly via cursor-up, while the header lands once and is never erased.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.total_lines = 2;
    fold.recent_lines.push_back("line-1".to_string());
    fold.current_line = "line-2".to_string();

    let (window, rows) = render_thinking_fold_window(fold);

    // Fold marker(1) + visible line(1) + current(1) = 3 physical rows, header not included.
    assert!(!window.contains("thinking"));
    assert!(window.contains("earlier lines"));
    assert!(window.contains("line-1"));
    assert!(window.contains("line-2"));
    assert_eq!(rows, 3);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn completed_thinking_fold_replaces_anchored_header_in_place() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = super::super::state::ThinkingFoldState::new();
    fold.active = true;
    fold.header_drawn = true;
    fold.max_visible_lines = 2;
    fold.total_lines = 3;
    fold.window_rows = 2;
    fold.rendered_body_lines = vec!["    second line".to_string(), "    third line".to_string()];
    let mut out = Vec::new();

    finalize_fold_to(&mut out, &mut fold, true).unwrap();

    // Assert per-line clear sequences (not CSI 0J): 0J clears from the first body row to the physical
    // screen bottom, crossing the DECSTBM scroll region and wiping the bottom side-note editor, so the
    // window must be cleared row by row with \x1b[2K and the anchored header rewritten in place.
    assert_eq!(
        String::from_utf8(out).unwrap(),
        format!(
            "\r\x1b[1A\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1A\r\r\x1b[1A\r\x1b[2K  {ACCENT_MUTED}✓ thinking · 3 lines\x1b[0m\r\n{ACCENT_MUTED}    … 3 earlier lines\x1b[0m"
        )
    );
    assert!(!fold.active);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_erase_rows_follow_current_terminal_reflow_of_previous_body() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.window_rows = 2;
    fold.rendered_body_lines = vec![
        "    … 103 earlier lines".to_string(),
        "    Actually, looking more carefully:".to_string(),
    ];

    unsafe {
        std::env::set_var("COLUMNS", "80");
    }
    assert_eq!(thinking_fold_rendered_body_rows(fold), 2);

    unsafe {
        std::env::set_var("COLUMNS", "12");
    }
    assert!(
        thinking_fold_rendered_body_rows(fold) > fold.window_rows,
        "narrow terminal should reflow previous body beyond cached window_rows"
    );

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_header_anchored_once_and_window_rows_track_body_only() {
    // "Orphan header stacking" regression: the header lands only on the first redraw (header_drawn=true)
    // and is never printed again no matter how many redraws follow; window_rows counts only body physical
    // rows (excluding the header), so cursor-up erasure always targets the visible body area and never drifts when the window scrolls into scrollback.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.render.thinking_fold.max_visible_lines = 2;
    state.render.thinking_fold.active = true;

    write_thinking_content_folded("line-1\n", &mut state, &markers).unwrap();
    assert!(state.render.thinking_fold.header_drawn);
    // 1 visible line, no fold -> body 1 row.
    assert_eq!(state.render.thinking_fold.window_rows, 1);
    assert_eq!(state.render.thinking_fold.rendered_body_lines.len(), 1);

    write_thinking_content_folded("line-2\nline-3\nline-4\n", &mut state, &markers).unwrap();
    // The header still lands only once and is never reprinted.
    assert!(state.render.thinking_fold.header_drawn);
    // 4 completed, 2 visible -> fold marker(1) + visible(2) = body 3 rows, header not counted.
    assert_eq!(state.render.thinking_fold.window_rows, 3);
    assert_eq!(state.render.thinking_fold.rendered_body_lines.len(), 3);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn cancelled_stream_result_finalizes_active_thinking_fold() {
    // On cancel with an active fold window, it must be finalized (finalize→reset), preventing a partial
    // thinking remnant with a new header stacked under it on the next retry (cross-turn root cause of duplicated headers + large blank areas).
    let mut state = StreamProcessingState::new();
    {
        let fold = &mut state.render.thinking_fold;
        fold.active = true;
        fold.max_visible_lines = 2;
        fold.total_lines = 1;
        fold.recent_lines.push_back("partial".to_string());
        fold.window_rows = 2;
    }

    let result = cancelled_stream_result(&mut state);

    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert!(result.skip_response_drain);
    // After finalize the fold state is reset: no longer active, window rows zeroed, no orphan window left behind.
    assert!(!state.render.thinking_fold.active);
    assert_eq!(state.render.thinking_fold.window_rows, 0);
    assert!(state.render.thinking_fold.recent_lines.is_empty());
}

#[test]
fn cancelled_stream_result_finalizes_active_subagent_fold() {
    let mut state = StreamProcessingState::new();
    {
        let fold = &mut state.render.subagent_fold;
        fold.active = true;
        fold.max_visible_lines = 2;
        fold.total_lines = 1;
        fold.recent_lines.push_back("partial answer".to_string());
        fold.window_rows = 2;
    }

    let result = cancelled_stream_result(&mut state);

    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert!(!state.render.subagent_fold.active);
    assert_eq!(state.render.subagent_fold.window_rows, 0);
    assert!(state.render.subagent_fold.recent_lines.is_empty());
}

#[test]
fn standalone_stream_marker_requires_exact_control_line() {
    assert!(is_standalone_stream_marker(
        "\n╭─ thinking\n",
        "╭─ thinking"
    ));
    assert!(is_standalone_stream_marker(
        "\n╰─ done thinking\n",
        "╰─ done thinking"
    ));
    assert!(!is_standalone_stream_marker(
        "reasoning mentions ╭─ thinking literally",
        "╭─ thinking"
    ));
    assert!(!is_standalone_stream_marker(
        "prefix\n╰─ done thinking\nsuffix",
        "╰─ done thinking"
    ));
}

#[test]
fn snapshot_done_chunk_does_not_duplicate_already_streamed_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"hello wor"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"hello world"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "hello world");
    assert_eq!(state.content.assistant_text, "hello world");
}

#[test]
fn reasoning_item_added_done_same_id_keeps_full_payload() {
    // The gateway re-delivers .added (partial payload) and .done (complete payload) for the same
    // reasoning resource: same id, different encrypted_content lengths (a real partial payload of
    // >=256 delivered via .added is captured too). The accumulator must converge by id and keep the
    // longest payload, otherwise the same resource id appearing twice triggers modelhub 400 (-4003 Duplicate item found).
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let added_payload = format!(
        r#"{{"output_index":0,"item":{{"type":"reasoning","id":"rs_same","encrypted_content":"{}"}}}}"#,
        "A".repeat(300)
    );
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.added"),
        &added_payload,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.done"),
        r#"{"output_index":0,"item":{"type":"reasoning","id":"rs_same","encrypted_content":"FULL_LONGER_PAYLOAD"}}"#,
    )
    .unwrap();

    assert_eq!(
        state.content.reasoning_items.len(),
        1,
        "同 id 的 reasoning item 必须收敛为一项"
    );
    assert_eq!(
        state.content.reasoning_items[0]
            .get("encrypted_content")
            .and_then(serde_json::Value::as_str),
        Some("FULL_LONGER_PAYLOAD"),
        "必须保留最长（完整）载荷"
    );
}

#[test]
fn tool_call_snapshot_done_does_not_duplicate_already_streamed_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":""}}"#,
        )
        .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.function_call_arguments.delta"),
        r#"{"output_index":0,"delta":"{\"path\":\"a"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.function_call_arguments.done"),
        r#"{"output_index":0,"arguments":"{\"path\":\"abc\"}"}"#,
    )
    .unwrap();

    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.id, "call_1");
    assert_eq!(builder.function_name, "write_file");
    assert_eq!(builder.arguments, "{\"path\":\"abc\"}");
}

fn write_http_chunk(stream: &mut std::net::TcpStream, payload: &str) -> std::io::Result<()> {
    write!(stream, "{:X}\r\n", payload.len())?;
    stream.write_all(payload.as_bytes())?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

#[tokio::test]
async fn stream_response_returns_after_finish_reason_without_eof() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return after the configured finish_reason grace window")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "hello");
    assert_eq!(current_history, "hello");
    assert!(result.skip_response_drain);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_marks_length_finish_reason_as_truncated() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        // Visible text present but the server truncated at the output cap: finish_reason=length.
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial output\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return after finish_reason grace window")
    .unwrap();

    // Key assertion: text present but finish_reason=length is treated as Completed. Reasoning models
    // often exhaust the output budget on reasoning tokens, yielding finish_reason=length while the
    // visible assistant_text is actually complete; retrying would only truncate again for nothing. Only
    // truncated tool call arguments JSON (dropped_malformed_tool_call) or no visible output at all
    // should escalate to Truncated and trigger a retry.
    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "partial output");

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_marks_reasoning_only_early_stop_as_truncated() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        // Only reasoning was emitted, no visible content was ever produced, no finish_reason was ever
        // sent, and then the connection closes outright (early EOF) — simulating the early-stop of GLM-style
        // enable_thinking models cut off mid chain-of-thought.
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Hmm\"}}]}\n\n",
        )
        .unwrap();
        // Close the chunked body (0-size chunk) then drop the stream to produce an EOF.
        let _ = stream.write_all(b"0\r\n\r\n");
        let _ = stream.flush();
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return promptly on reasoning-only early stop")
    .unwrap();

    // Key assertion: an early stop with only reasoning, no visible text and no finish_reason must
    // escalate to Truncated so the upper layer retries with a downgraded model / thinking off, not silently Completed.
    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.assistant_text.trim().is_empty());
    assert_eq!(result.reasoning_text, "Hmm");
    assert!(!result.truncated_by_length);
    assert!(!result.stream_error);

    drop(response);
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_keeps_reading_delayed_chunks_after_finish_reason() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(&mut stream, "data: [DONE]\n\n").unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should keep reading delayed chunks after finish_reason")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "hello world");
    assert_eq!(current_history, "hello world");
    assert!(result.skip_response_drain);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn recover_inline_tool_calls_normalizes_namespaced_xml_prefix() {
    // Some frontends/models wrap Anthropic-style invokes in the <|DSML|> protocol.
    // After normalization the Anthropic XML parser should recognize them; no per-<|PREFIX|> parser needed.
    let raw = r#"<|DSML|tool_calls><|DSML|invoke name="apply_patch"><|DSML|parameter name="file_path">/tmp/x</|DSML|parameter><|DSML|parameter name="patch">---</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover DSML-wrapped tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "apply_patch");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x");
    assert_eq!(args["patch"], "---");
}

#[test]
fn recover_inline_tool_calls_normalizes_fullwidth_dsml_prefix() {
    // Per debug.md, DeepSeek actually emits the fullwidth-vertical-bar variant: <｜｜DSML｜｜...>.
    let raw = r#"<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="apply_patch"><｜｜DSML｜｜parameter name="file_path">/tmp/x</｜｜DSML｜｜parameter><｜｜DSML｜｜parameter name="patch">---</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover fullwidth-DSML tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "apply_patch");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x");
    assert_eq!(args["patch"], "---");
}

/// Reproduces the session bc1f2e88 failure: a reasoner with a prefilled `<think>` template writes
/// the chain of thought into the content channel, ending only with a dangling `</think>`. With the
/// splitter armed, leaked reasoning before `</think>` must go into reasoning_text (rendered in the
/// thinking fold); only the real answer after `</think>` enters assistant_text, so the final answer is never output twice.
#[test]
fn armed_demuxer_splits_leaked_reasoning_from_visible_content() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    // The chain of thought arrives across multiple content chunks, with `</think>` split at a chunk boundary.
    for payload in [
        r#"{"choices":[{"delta":{"content":"Let me consolidate. "}}]}"#,
        r#"{"choices":[{"delta":{"content":"I have enough evidence.</thi"}}]}"#,
        r#"{"choices":[{"delta":{"content":"nk>## 结论\n不是 bug。"}}]}"#,
    ] {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            None,
            payload,
        )
        .unwrap();
    }

    // The real answer after `</think>` is the only visible body; the chain of thought must not leak into assistant_text.
    assert_eq!(state.content.assistant_text, "## 结论\n不是 bug。");
    assert!(!state.content.assistant_text.contains("Let me consolidate"));
    assert!(!state.content.assistant_text.contains("</think>"));
    // The leaked reasoning is split back into the reasoning channel.
    assert!(
        state
            .content
            .reasoning_text
            .contains("Let me consolidate. I have enough evidence.")
    );
}

#[test]
fn demuxer_buffered_content_counts_as_stream_progress() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"long reasoning without close yet"}}]}"#,
    )
    .unwrap();

    assert!(outcome.meaningful_progress);
    assert!(state.content.assistant_text.is_empty());
    assert!(state.content.reasoning_text.is_empty());
}

#[test]
fn demuxer_flush_without_close_tag_commits_visible_content() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"visible fallback without close"}}]}"#,
    )
    .unwrap();
    assert!(state.content.assistant_text.is_empty());

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.assistant_text, "visible fallback without close");
    assert_eq!(current_history, "visible fallback without close");
}

#[test]
fn replayed_content_part_after_demux_close_does_not_replay_reasoning_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"reasoning</think>answer"}}]}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "answer");

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"reasoning</think>answer"}}"#,
    )
    .unwrap();

    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

#[test]
fn output_text_snapshot_after_demux_close_does_not_replay_reasoning_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"reasoning</think>answer"}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "answer");

    let snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"reasoning</think>answer"}"#,
    )
    .unwrap();

    assert!(!snapshot.meaningful_progress);
    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

#[test]
fn output_text_snapshot_can_finish_a_partially_streamed_demux_capture() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"reasoning"}"#,
    )
    .unwrap();
    assert!(state.content.assistant_text.is_empty());

    let snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"reasoning</think>answer"}"#,
    )
    .unwrap();

    assert!(snapshot.meaningful_progress);
    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

/// Reverse assertion: for a normal model without the splitter armed (using the separate reasoning_content
/// field), behavior is unchanged — a literal `</think>` in content lands verbatim in the visible body and is never swallowed.
#[test]
fn stream_filters_rewrite_visible_content_before_commit() {
    // Filter: rewrite "secret" into "[REDACTED]".
    struct RedactFilter;
    impl crate::ai::ports::stream::StreamFilter for RedactFilter {
        fn filter(&self, chunk: &str) -> Option<String> {
            Some(chunk.replace("secret", "[REDACTED]"))
        }
        fn name(&self) -> &'static str {
            "redact"
        }
    }
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    // Step 6: inject the filter chain and verify that `process_stream_payload`'s visible-content commit
    // point applies the filters (the rewrite lands in assistant_text / history, the original text does not appear).
    state.filters = crate::ai::ports::stream::FilterChain::new().push(RedactFilter);
    let mut app = test_app();
    let mut current_history = String::new();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"hello secret world"}}]}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "hello [REDACTED] world");
    assert_eq!(current_history, "hello [REDACTED] world");
    assert!(!state.content.assistant_text.contains("secret"));
}

fn unarmed_demuxer_leaves_content_untouched() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"see </think> literal"}}]}"#,
    )
    .unwrap();

    assert_eq!(state.content.assistant_text, "see </think> literal");
    assert!(state.content.reasoning_text.is_empty());
}
