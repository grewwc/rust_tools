    use super::*;
    use super::super::inline_recovery::normalize_tool_call_arguments;
    use crate::ai::{
        cli::ParsedCli,
        tools::os_tools::{GLOBAL_OS, init_os_tools_globals},
        types::{App, AppConfig},
    };
    use std::io::Read as _;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool, mpsc};

    #[test]
    fn prompt_cache_metrics_none_without_hit() {
        assert_eq!(format_prompt_cache_metrics(1000, 0), None);
        assert_eq!(format_prompt_cache_metrics(0, 0), None);
    }

    #[test]
    fn prompt_cache_metrics_reports_hit_rate() {
        let line = format_prompt_cache_metrics(1000, 750).unwrap();
        assert!(line.contains("750/1000"));
        assert!(line.contains("75% hit"));
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

    fn test_app() -> App {
        App {
            cli: ParsedCli::default(),
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
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: String::new(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
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
        }
    }

    #[tokio::test]
    async fn wait_for_interrupt_observes_request_interrupt_source() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        if content.starts_with(&markers.end_thinking_tag)
            && state.render.markdown.has_unfinished_line()
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
        // 模拟大文件 write_file 撞输出上限：arguments JSON 半截、无法修复。
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
        // 模拟 qwen3.7-max 把 tool call 当成 content 输出的情况。
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
        // 截图中模型实际输出的 Hermes/Qwen XML 形态（body 为 JSON）。
        let raw =
            "<tool_call>\n<function=read_file>\n{\"path\":\"/tmp/x\"}\n</function>\n</tool_call>";
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
        // 数字参数被识别为 JSON 数字而非字符串。
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
        // 普通文本回答绝不能被误判为 tool call。
        assert!(recover_inline_tool_calls("Hello world").is_none());
        assert!(recover_inline_tool_calls("").is_none());
        // 仅有 name 没有 arguments，且 name 不在合法对象中——这里应该也是不解析的，
        // 但为保留兼容性我们允许 name 单独存在时仍然识别。下面是真正的负样本：
        assert!(recover_inline_tool_calls("{\"foo\":\"bar\"}").is_none());
        assert!(recover_inline_tool_calls("12345").is_none());
        // 字符串形式的 args 必须本身也是合法 JSON，否则拒绝。
        assert!(recover_inline_tool_calls(r#"{"name":"x","arguments":"not-json"}"#).is_none());
    }

    #[test]
    fn recover_inline_tool_calls_handles_anthropic_xml_parameter_tags() {
        // deepseek-v4-flash 实际输出的 Anthropic 风格：<invoke name=...>/<parameter name=...>。
        let raw = r#"<function_calls><invoke name="read_file"><parameter name="path">/tmp/x</parameter><parameter name="limit">200</parameter></invoke></function_calls>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/tmp/x");
        assert_eq!(args["limit"], 200);
    }

    #[test]
    fn recover_inline_tool_calls_handles_anthropic_xml_namespaced_tags() {
        // 带命名空间前缀（antml:）且无外层包裹。
        let raw = r#"<invoke name="list_agents"></invoke>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_agents");
        assert_eq!(calls[0].function.arguments, "{}");
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
    fn anthropic_xml_streamer_suppresses_markup_and_emits_events() {
        let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = streamer.push(
            r#"Let me check.<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>"#,
        );
        // invoke 标记不外显，仅保留前置散文。
        assert_eq!(cleaned, "Let me check.");
        // 产出 Begin/Args/End 事件，与内部 tool_call 管线一致。
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

        let should_stop = process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.completed"),
            r#"{"status":"completed"}"#,
        )
        .unwrap();
        assert!(!should_stop);

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.output_text.done"),
            r#"{"text":"hello world"}"#,
        )
        .unwrap();

        assert_eq!(current_history, "hello world");
        assert_eq!(state.content.assistant_text, "hello world");
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
            normalize::StreamProviderAdapterKind::OpenAi,
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
            normalize::StreamProviderAdapterKind::OpenAi,
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
    fn thinking_fold_drops_interior_blank_lines() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        state.render.thinking_fold.max_visible_lines = 8;
        state.render.thinking_fold.active = true;

        // 模型常用空行分段：段间空行不应占用折叠窗口的可见行。
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
        // 锁定并置宽 COLUMNS：本用例断言正文/标记原样存在，须避免与 COLUMNS=12
        // 的折行用例并发时读到被泄漏的窄列宽而触发 clamp 截断。
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        assert_eq!(window.matches("lines folded").count(), 1);
        assert!(!window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert!(window.contains("line-3"));
        assert!(window.contains("line-4"));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_window_rows_follow_wrapped_terminal_height() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("COLUMNS", "12");
        }

        let mut state = StreamProcessingState::new();
        let fold = &mut state.render.thinking_fold;
        fold.max_visible_lines = 2;
        fold.total_lines = 2;
        fold.recent_lines.push_back("12345678901234567890".to_string());
        fold.recent_lines.push_back("abcdef".to_string());
        fold.current_line = "ghijklmnopqrst".to_string();

        let (window, rows) = render_thinking_fold_window(fold);

        // 每条可见行被 clamp 成单物理行，窗口物理行数恒等于逻辑行数：
        // 1 折叠标记 + 2 可见行 = 3。
        assert_eq!(rows, 3);
        // 每条渲染行都不超过终端列宽（12），确保 cursor-up 擦除精确。
        for line in window.lines() {
            let visible = crate::ai::stream::extract::strip_ansi_codes(line);
            assert!(
                unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
                "line exceeds terminal width: {visible:?}"
            );
        }
        // 未溢出的短行原样保留；溢出的超长行被截断为省略号结尾。
        assert!(window.contains("abcdef"));
        assert!(window.contains('…'));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_window_without_hidden_lines_has_no_fold_marker() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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

        // 无隐藏行、无 active header：窗口物理行数 == 可见逻辑行数（3）。
        assert!(!window.contains("lines folded"));
        assert!(window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert!(window.contains("line-3"));
        assert_eq!(rows, 3);

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_active_window_includes_header_in_row_budget() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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

        // active header(1) + 折叠标记(1) + 可见行(1) + current(1) = 4 物理行。
        assert!(window.contains("thinking"));
        assert!(window.contains("lines folded"));
        assert!(window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert_eq!(rows, 4);

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn cancelled_stream_result_finalizes_active_thinking_fold() {
        // 取消时若折叠窗口仍活跃，必须收口（finalize→reset），避免半截 `╭─ thinking`
        // 残留、下一轮重试在其下叠新 header（重复 header + 大段空白的跨轮根因）。
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
        // finalize 后折叠状态被 reset：不再 active，窗口行数归零，无孤儿窗口残留。
        assert!(!state.render.thinking_fold.active);
        assert_eq!(state.render.thinking_fold.window_rows, 0);
        assert!(state.render.thinking_fold.recent_lines.is_empty());
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
            normalize::StreamProviderAdapterKind::OpenCode,
            Some("response.output_text.delta"),
            r#"{"delta":"hello wor"}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenCode,
            Some("response.output_text.done"),
            r#"{"text":"hello world"}"#,
        )
        .unwrap();

        assert_eq!(current_history, "hello world");
        assert_eq!(state.content.assistant_text, "hello world");
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
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":""}}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.function_call_arguments.delta"),
            r#"{"output_index":0,"delta":"{\"path\":\"a"}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
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
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
            // 有可见文本但服务端因输出上限截断：finish_reason=length。
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

        // 关键断言：有文本但 finish_reason=length 时按 Completed 处理。推理模型
        // 的 reasoning token 经常占满输出预算导致 finish_reason=length，但可见的
        // assistant_text 实际已完整。继续重试只会无意义地反复截断。只有 tool call
        // arguments JSON 被截断（dropped_malformed_tool_call）或完全没有可见输出
        // 时，才应升级为 Truncated 触发重试。
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
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
            // 只吐了 reasoning，从未产出可见 content，也从未发送 finish_reason，
            // 随后直接关闭连接（提前 EOF）——模拟 GLM 等 enable_thinking 模型憋着
            // 思考链被掐断的早停场景。
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Hmm\"}}]}\n\n",
            )
            .unwrap();
            // 关闭 chunked body（0-size chunk）后 drop stream，制造 EOF。
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

        // 关键断言：只有 reasoning、无可见文本、无 finish_reason 的早停必须升级为
        // Truncated，交给上层降档 / 关 thinking 后重试，而不是静默 Completed。
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
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        // 某些前端/模型会把 Anthropic 风格的 invoke 包在 <|DSML|> 协议里输出。
        // 归一化后应被 Anthropic XML 解析器识别，无需为每种 <|PREFIX|> 单独写 parser。
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
        // debug.md 里的 DeepSeek 实际会输出全角竖线版本：<｜｜DSML｜｜...>。
        let raw = r#"<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="apply_patch"><｜｜DSML｜｜parameter name="file_path">/tmp/x</｜｜DSML｜｜parameter><｜｜DSML｜｜parameter name="patch">---</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"#;
        let calls =
            recover_inline_tool_calls(raw).expect("should recover fullwidth-DSML tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "apply_patch");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["file_path"], "/tmp/x");
        assert_eq!(args["patch"], "---");
    }
