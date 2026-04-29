use std::io::{self, IsTerminal, Write};
use std::sync::Arc;
use std::time::Duration;

use crate::ai::{
    driver::print::format_section_header,
    models,
    provider::ApiProvider,
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, ToolCall, take_stream_cancelled},
};

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_with_tools, strip_ansi_codes},
    framing, normalize,
    splitter::StreamSplitSegment,
    state::{StreamChunkStep, StreamMarkers, StreamProcessingState, ToolCallBuilder},
};

/// Maximum number of decode errors before giving up and returning partial content
const MAX_DECODE_ERRORS: usize = 3;
/// Delay in milliseconds between retry attempts on transient errors
const DECODE_ERROR_RETRY_DELAY_MS: u64 = 100;

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    _terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let adapter_kind = normalize::resolve_adapter_kind(
        models::model_provider(&app.current_model),
        &models::endpoint_for_model(&app.current_model, &app.config.endpoint),
    );

    if should_show_opencode_waiting_hint(app) {
        print_waiting_hint(&mut state)?;
    }

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, state.content.thinking_open) {
            return Ok(result);
        }

        let chunk_result = tokio::select! {
            chunk = response.chunk() => chunk,
            _ = wait_for_interrupt(app) => {
                return Ok(cancelled_stream_result(state.content.thinking_open));
            }
        };

        match process_chunk_result(
            app,
            current_history,
            &markers,
            &mut state,
            adapter_kind,
            chunk_result,
        )
        .await?
        {
            StreamChunkStep::Continue => {}
            StreamChunkStep::Stop => break,
            StreamChunkStep::Return(result) => return Ok(result),
        }
    }

    if let Some(result) =
        process_pending_tail(app, current_history, &markers, &mut state, adapter_kind).await?
    {
        return Ok(result);
    }

    finalize_stream_response(app, &markers, state)
}

fn should_show_opencode_waiting_hint(app: &App) -> bool {
    io::stdout().is_terminal() && models::model_provider(&app.current_model) == ApiProvider::OpenCode
}

fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    println!(
        "{}",
        format_section_header("stream", Some("waiting for first visible chunk from provider..."))
    );
    io::stdout().flush()?;
    state.render.waiting_hint_active = true;
    Ok(())
}

fn upgrade_waiting_hint_for_buffering(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active || state.render.waiting_hint_buffering {
        return Ok(());
    }
    print!("\x1b[1A\r\x1b[2K");
    println!(
        "{}",
        format_section_header(
            "stream",
            Some("provider is alive but buffering visible output; text may arrive in one block...")
        )
    );
    io::stdout().flush()?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    print!("\x1b[1A\r\x1b[2K");
    io::stdout().flush()?;
    state.render.waiting_hint_active = false;
    state.render.waiting_hint_buffering = false;
    Ok(())
}

fn immediate_cancel_result(app: &App, thinking_open: bool) -> Option<StreamResult> {
    app.cancel_stream
        .load(std::sync::atomic::Ordering::Relaxed)
        .then(|| cancelled_stream_result(thinking_open))
}

fn cancelled_stream_result(thinking_open: bool) -> StreamResult {
    if thinking_open {
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }
    StreamResult {
        outcome: StreamOutcome::Cancelled,
        tool_calls: Vec::new(),
        assistant_text: String::new(),
        hidden_meta: String::new(),
    }
}

async fn process_chunk_result<T: AsRef<[u8]>>(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
    chunk_result: Result<Option<T>, reqwest::Error>,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    match chunk_result {
        Ok(Some(chunk)) => {
            framing::push_chunk(&mut state.framing, chunk.as_ref());
            state.framing.decode_error_count = 0;
            consume_pending_complete_lines(app, current_history, markers, state, adapter_kind).await
        }
        Ok(None) => Ok(StreamChunkStep::Stop),
        Err(err) => {
            if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                Ok(StreamChunkStep::Return(result))
            } else {
                Ok(StreamChunkStep::Continue)
            }
        }
    }
}

async fn consume_pending_complete_lines(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    // Move the pending buffer out so line slices can borrow from it while `state`
    // remains available for mutation inside `process_stream_line()`.
    let lines = match framing::take_complete_lines(&mut state.framing) {
        Ok(lines) => lines,
        Err(err) => {
            if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                return Ok(StreamChunkStep::Return(result));
            }
            return Ok(StreamChunkStep::Continue);
        }
    };
    let mut should_stop = false;
    for line in lines {
        if process_stream_line(app, current_history, markers, state, adapter_kind, &line)? {
            should_stop = true;
            break;
        }
    }
    Ok(if should_stop {
        StreamChunkStep::Stop
    } else {
        StreamChunkStep::Continue
    })
}

async fn process_pending_tail(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<Option<StreamResult>, Box<dyn std::error::Error>> {
    if state.framing.pending.is_empty() {
        return Ok(None);
    }

    let line = match framing::take_pending_tail(&mut state.framing) {
        Ok(Some(line)) => line,
        Ok(None) => return Ok(None),
        Err(err) => {
            if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                return Ok(Some(result));
            }
            return Ok(None);
        }
    };
    if !line.is_empty() {
        let _ = process_stream_line(app, current_history, markers, state, adapter_kind, &line)?;
    }
    if flush_sse_event(app, current_history, markers, state, adapter_kind)? {
        let final_state = std::mem::replace(state, StreamProcessingState::new());
        return Ok(Some(finalize_stream_response(app, markers, final_state)?));
    }
    Ok(None)
}

fn finalize_stream_response(
    app: &mut App,
    markers: &StreamMarkers,
    mut state: StreamProcessingState,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    clear_waiting_hint(&mut state)?;

    if state.content.thinking_open {
        write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_ref(),
            &mut state.render.markdown,
            false,
        )?;
    }

    flush_terminal_splitter(&mut state, markers)?;

    state.render.markdown.flush_pending()?;

    if state.render.current_printing_index.is_some() {
        println!("\x1b[0m)");
    }

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(false));
    }

    // AIOS: flush any pending LLM usage to kernel `/dev/llm` before returning.
    // Prefer the model echoed by the provider; fall back to what we requested.
    if let Some((echoed_model, usage)) = state.pending_llm_usage.take() {
        let model_for_pricing = if echoed_model.is_empty() {
            app.current_model.clone()
        } else {
            echoed_model
        };
        let _ = crate::ai::request::charge_llm_usage_to_kernel(
            app,
            &model_for_pricing,
            &usage,
            0,
        );
    }

    let tool_calls = collect_valid_tool_calls(&mut state.content.tool_calls_map);

    let outcome = if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        StreamOutcome::Completed
    };

    Ok(StreamResult {
        outcome,
        tool_calls,
        assistant_text: state.content.assistant_text,
        hidden_meta: state.content.hidden_meta,
    })
}

async fn wait_for_interrupt(app: &App) {
    loop {
        if app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
            || app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn handle_stream_decode_error<E: std::fmt::Display>(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    err: E,
) -> Option<StreamResult> {
    state.framing.decode_error_count += 1;
    eprintln!(
        "[Warning] 读取响应流时出错：{} (错误次数：{}/{})",
        err, state.framing.decode_error_count, MAX_DECODE_ERRORS
    );

    if take_stream_cancelled(app) {
        return Some(cancelled_stream_result(state.content.thinking_open));
    }

    if state.framing.decode_error_count <= MAX_DECODE_ERRORS {
        eprintln!("[Warning] 尝试继续读取...");
        tokio::time::sleep(Duration::from_millis(DECODE_ERROR_RETRY_DELAY_MS)).await;
        return None;
    }

    eprintln!("[Error] 响应流读取失败，返回已收集的内容");

    if state.content.thinking_open {
        let _ = write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_ref(),
            &mut state.render.markdown,
            false,
        );
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    let _ = state.render.markdown.flush_pending();

    Some(StreamResult {
        outcome: StreamOutcome::Completed,
        tool_calls: collect_valid_tool_calls(&mut state.content.tool_calls_map),
        assistant_text: std::mem::take(&mut state.content.assistant_text),
        hidden_meta: String::new(),
    })
}

fn normalize_tool_call_arguments(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }
    serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
    Some(trimmed.to_string())
}

fn collect_valid_tool_calls(
    builders: &mut rust_tools::commonw::FastMap<usize, ToolCallBuilder>,
) -> Vec<ToolCall> {
    builders
        .drain()
        .filter_map(|(_, mut builder)| {
            let Some(arguments) = normalize_tool_call_arguments(&builder.arguments) else {
                eprintln!(
                    "[Warning] dropping malformed tool call '{}' due to incomplete JSON arguments",
                    builder.function_name
                );
                return None;
            };
            builder.arguments = arguments;
            Some(builder.build())
        })
        .collect()
}

fn ensure_tool_calls_section_open(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
) {
    if state.render.printed_tool_calls_header {
        return;
    }

    let _ = clear_waiting_hint(state);

    if state.content.thinking_open {
        let _ = write_stream_content(
            &format_end_thinking_line(markers, &state.render.markdown),
            app.writer.as_ref(),
            &mut state.render.markdown,
            false,
        );
        state.content.thinking_open = false;
    }
    let _ = state.render.markdown.flush_pending();
    println!("{}", format_section_header("tool calls", None));
    state.render.printed_tool_calls_header = true;
}

struct ToolCallRenderChunk {
    function_name: String,
    arguments: String,
    open_line: bool,
}

fn take_tool_call_render_chunk(
    current_printing_index: Option<usize>,
    index: usize,
    builder: &mut ToolCallBuilder,
) -> Option<ToolCallRenderChunk> {
    if builder.function_name.is_empty() {
        return None;
    }

    let start = builder.printed_arguments_len.min(builder.arguments.len());
    let arguments = builder.arguments[start..].to_string();
    builder.printed_arguments_len = builder.arguments.len();

    Some(ToolCallRenderChunk {
        function_name: builder.function_name.clone(),
        arguments,
        open_line: current_printing_index != Some(index),
    })
}

fn open_tool_call_line(
    state: &mut StreamProcessingState,
    index: usize,
    function_name: &str,
) -> io::Result<()> {
    if state.render.current_printing_index.is_some() {
        println!("\x1b[0m)");
    }
    state.render.current_printing_index = Some(index);
    print!(
        "  {}│{} {}{}{}({}",
        ACCENT_RULE, RESET, ACCENT_MUTED, function_name, RESET, DIM
    );
    io::stdout().flush()
}

fn write_tool_call_arguments_stream(arguments: &str) -> io::Result<()> {
    if arguments.is_empty() {
        return Ok(());
    }

    let mut stdout = io::stdout();
    if stdout.is_terminal() {
        for ch in arguments.chars() {
            let mut buf = [0u8; 4];
            stdout.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
            stdout.flush()?;
        }
        return Ok(());
    }

    stdout.write_all(arguments.as_bytes())?;
    stdout.flush()
}

fn process_external_tool_calls_delta(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    chunk: &StreamChunk,
) {
    let Some(choice) = chunk.choices.first() else {
        return;
    };

    for stream_tool_call in &choice.delta.tool_calls {
        let index = stream_tool_call.index;
        ensure_tool_calls_section_open(app, markers, state);

        let render_chunk = {
            let builder = state.content.tool_calls_map.entry(index).or_default();
            if !stream_tool_call.id.is_empty() {
                builder.id.clone_from(&stream_tool_call.id);
            }
            if !stream_tool_call.tool_type.is_empty() {
                builder.tool_type.clone_from(&stream_tool_call.tool_type);
            }
            if !stream_tool_call.function.name.is_empty() {
                builder
                    .function_name
                    .clone_from(&stream_tool_call.function.name);
            }
            builder
                .arguments
                .push_str(&stream_tool_call.function.arguments);
            take_tool_call_render_chunk(state.render.current_printing_index, index, builder)
        };

        if let Some(render_chunk) = render_chunk {
            if render_chunk.open_line {
                let _ = open_tool_call_line(state, index, &render_chunk.function_name);
            }
            let _ = write_tool_call_arguments_stream(&render_chunk.arguments);
        }
    }
}

fn process_internal_tool_calls(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    internal_tool_calls: Vec<super::state::InternalToolCall>,
) {
    for tc in internal_tool_calls {
        let super::state::InternalToolCall {
            id,
            tool_type,
            function_name,
            arguments,
        } = tc;
        ensure_tool_calls_section_open(app, markers, state);

        let builder = state
            .content
            .tool_calls_map
            .entry(state.content.internal_tool_call_idx)
            .or_default();
        builder.id = id;
        builder.tool_type = tool_type;
        builder.function_name = function_name;
        builder.arguments = arguments;

        if state.render.current_printing_index.is_some() {
            println!("\x1b[0m)");
            state.render.current_printing_index = None;
        }

        print!(
            "  {}│{} {}{}{}({}{}\x1b[0m)",
            ACCENT_RULE,
            RESET,
            ACCENT_MUTED,
            builder.function_name,
            RESET,
            DIM,
            builder.arguments
        );
        println!();
        let _ = io::stdout().flush();

        state.content.internal_tool_call_idx += 1;
    }
}

fn commit_visible_content(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    mut content: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if content.is_empty() {
        return Ok(());
    }

    normalize_end_thinking_boundary(&mut content, markers, &state.render.markdown);

    clear_waiting_hint(state)?;

    maybe_write_stream_content(
        content.as_str(),
        app.writer.as_ref(),
        state,
        markers,
        state.content.thinking_open,
    )?;
    if state.content.thinking_open {
        return Ok(());
    }

    let text = if content.contains(&markers.end_thinking_tag) {
        content.replace(&markers.end_thinking_tag, "")
    } else {
        content
    };
    let text = text.trim_matches('\n');
    current_history.reserve(text.len());
    state.content.assistant_text.reserve(text.len());
    current_history.push_str(text);
    state.content.assistant_text.push_str(text);

    Ok(())
}

fn format_end_thinking_line(markers: &StreamMarkers, markdown: &MarkdownStreamRenderer) -> String {
    let mut content = format!("{}\n", markers.end_thinking_tag);
    normalize_end_thinking_boundary(&mut content, markers, markdown);
    content
}

fn normalize_end_thinking_boundary(
    content: &mut String,
    markers: &StreamMarkers,
    markdown: &MarkdownStreamRenderer,
) {
    if content.starts_with(&markers.end_thinking_tag) && markdown.has_unfinished_line() {
        content.insert(0, '\n');
    }
}

fn flush_terminal_splitter(
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
) -> io::Result<()> {
    let marker_line = format!("{}\n", markers.end_thinking_tag);
    let marker_line_with_prefix = format!("\n{}\n", markers.end_thinking_tag);
    let segments = state
        .render
        .terminal_splitter
        .flush(&[marker_line_with_prefix.as_str(), marker_line.as_str()]);
    for segment in segments {
        write_stream_split_segment(segment, state)?;
    }
    Ok(())
}

fn write_stream_split_segment(
    segment: StreamSplitSegment,
    state: &mut StreamProcessingState,
) -> io::Result<()> {
    match segment {
        StreamSplitSegment::Text(text) => maybe_write_plain_stream_text(&text, state, false),
        StreamSplitSegment::Marker { marker_index: _, text } => {
            write_stream_content_to_terminal(&text, &mut state.render.markdown, false)
        }
    }
}

fn maybe_write_plain_stream_text(
    content: &str,
    state: &mut StreamProcessingState,
    dimmed: bool,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }
    if dimmed {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    write_stream_content_to_terminal(content, &mut state.render.markdown, false)
}

fn maybe_write_stream_content(
    content: &str,
    writer: Option<&Arc<std::sync::Mutex<std::fs::File>>>,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
    dimmed: bool,
) -> io::Result<()> {
    if let Some(w) = writer {
        let mut guard = w.lock().unwrap();
        let clean = strip_ansi_codes(content);
        guard.write_all(clean.as_bytes())?;
        guard.flush()?;
    }

    if dimmed {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    let marker_line = format!("{}\n", markers.end_thinking_tag);
    let marker_line_with_prefix = format!("\n{}\n", markers.end_thinking_tag);
    let segments = state
        .render
        .terminal_splitter
        .push(content, &[marker_line_with_prefix.as_str(), marker_line.as_str()]);
    for segment in segments {
        write_stream_split_segment(segment, state)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        types::{App, AppConfig},
    };
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, atomic::AtomicBool};

    fn test_app() -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 0,
                history_keep_last: 0,
                history_summary_max_chars: 0,
                intent_model: None,
                intent_model_path: PathBuf::new(),
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: String::new(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
        }
    }

    #[test]
    fn closing_thinking_marker_starts_on_new_line_when_reasoning_line_is_open() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state.render.markdown.write_chunk("still thinking", true).unwrap();
        let mut content = format!("{}\nfinal", markers.end_thinking_tag);
        if content.starts_with(&markers.end_thinking_tag) && state.render.markdown.has_unfinished_line() {
            content.insert(0, '\n');
        }

        assert_eq!(content, format!("\n{}\nfinal", markers.end_thinking_tag));
    }

    #[test]
    fn closing_thinking_marker_keeps_compact_spacing_when_already_at_line_start() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state.render.markdown.write_chunk("still thinking\n", true).unwrap();
        let mut content = format!("{}\nfinal", markers.end_thinking_tag);
        normalize_end_thinking_boundary(&mut content, &markers, &state.render.markdown);

        assert_eq!(content, format!("{}\nfinal", markers.end_thinking_tag));
    }

    #[test]
    fn tool_call_boundary_closes_thinking_on_a_fresh_line() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state.render.markdown.write_chunk("still thinking", true).unwrap();

        assert_eq!(
            format_end_thinking_line(&markers, &state.render.markdown),
            format!("\n{}\n", markers.end_thinking_tag)
        );
    }

    #[test]
    fn first_reasoning_chunk_keeps_open_marker_and_first_text() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut app = test_app();
        let mut current_history = String::new();
        let path = std::env::temp_dir().join(format!(
            "ai-stream-thinking-{}.log",
            uuid::Uuid::new_v4()
        ));
        let file = File::create(&path).unwrap();
        app.writer = Some(Arc::new(Mutex::new(file)));
        let payload = r#"{"choices":[{"delta":{"reasoning_content":"我判断这是首段"}}]}"#;

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::Compatible,
            None,
            payload,
        )
        .unwrap();

        assert!(state.content.thinking_open);
        assert!(current_history.is_empty());
        drop(app.writer.take());
        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(written.contains(&markers.thinking_tag));
        assert!(written.contains("我判断这是首段"));
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

}

fn process_stream_payload(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
    event_type: Option<&str>,
    payload: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (chunk, merge_mode) = match normalize::parse_stream_payload(adapter_kind, payload, event_type) {
        super::state::ParsedStreamPayload::Ignore => return Ok(false),
        super::state::ParsedStreamPayload::Done => return Ok(true),
        super::state::ParsedStreamPayload::Chunk(chunk) => (chunk, StreamEventMergeMode::Append),
        super::state::ParsedStreamPayload::SnapshotChunk(chunk) => {
            (chunk, StreamEventMergeMode::AppendMissingSuffix)
        }
    };

    // AIOS: capture usage block from whichever chunk carries it. OpenAI emits
    // the final `usage` on a chunk with `choices: []`, so we must pull it *before*
    // the empty-choices early return below.
    if let Some(ref usage) = chunk.usage {
        state.pending_llm_usage = Some((chunk.model.clone(), usage.clone()));
    }

    if chunk.choices.is_empty() {
        state.content.empty_choice_chunks += 1;
        if should_show_opencode_waiting_hint(app) && state.content.empty_choice_chunks >= 3 {
            let _ = upgrade_waiting_hint_for_buffering(state);
        }
        return Ok(false);
    }

    state.content.empty_choice_chunks = 0;

    if let Some(choice) = chunk.choices.first() {
        if !choice.delta.reasoning_content.is_empty() {
            state.content.saw_reasoning_output = true;
        }
    }

    process_external_tool_calls_delta(app, markers, state, &chunk);

    let (events, internal_tool_calls) = extract_chunk_events_with_tools(
        &chunk,
        markers.hidden_begin,
        markers.hidden_end,
        &mut state.content.thinking_open,
        &mut state.content.hidden_meta_parse,
    );
    process_internal_tool_calls(app, markers, state, internal_tool_calls);

    if events.is_empty() {
        return Ok(false);
    }
    for event in events {
        match event {
            StreamTextEvent::AppendHiddenMeta(text) => {
                state.content.hidden_meta.push_str(&text);
            }
            other => {
                let Some(content) = stream_text_event_to_content(
                    &other,
                    markers,
                    merge_mode,
                    &state.content.assistant_text,
                ) else {
                    continue;
                };
                if content.is_empty() {
                    continue;
                }
                commit_visible_content(app, current_history, markers, state, content)?;
            }
        }
    }

    // Keep streaming until explicit stream end ([DONE] or EOF). Some providers can
    // set finish_reason before all visible content chunks are delivered.
    Ok(false)
}

#[derive(Clone, Copy)]
enum StreamEventMergeMode {
    Append,
    AppendMissingSuffix,
}

fn stream_text_event_to_content(
    event: &StreamTextEvent,
    markers: &StreamMarkers,
    merge_mode: StreamEventMergeMode,
    assistant_text: &str,
) -> Option<String> {
    match event {
        StreamTextEvent::OpenThinking => Some(format!("\n{}\n", markers.thinking_tag)),
        StreamTextEvent::AppendThinking(text) => (!text.is_empty()).then(|| text.clone()),
        StreamTextEvent::AppendContent(text) => match merge_mode {
            StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
            StreamEventMergeMode::AppendMissingSuffix => {
                let suffix = unseen_suffix(assistant_text, text);
                (!suffix.is_empty()).then_some(suffix)
            }
        },
        StreamTextEvent::AppendHiddenMeta(_) => None,
        StreamTextEvent::CloseThinking => Some(format!("{}\n", markers.end_thinking_tag)),
    }
}

fn unseen_suffix(existing: &str, incoming: &str) -> String {
    if incoming.is_empty() || existing.ends_with(incoming) {
        return String::new();
    }

    let boundaries = incoming
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(incoming.len()))
        .collect::<Vec<_>>();

    for overlap_chars in (1..boundaries.len()).rev() {
        let split_idx = boundaries[overlap_chars];
        if existing.ends_with(&incoming[..split_idx]) {
            return incoming[split_idx..].to_string();
        }
    }

    incoming.to_string()
}

fn flush_sse_event(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(event) = framing::flush_sse_event(&mut state.framing) else {
        return Ok(false);
    };
    process_stream_payload(
        app,
        current_history,
        markers,
        state,
        adapter_kind,
        event.event_type.as_deref(),
        &event.payload,
    )
}

fn process_stream_line(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
    line: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if let Some(event) = framing::consume_sse_line(&mut state.framing, line) {
        return process_stream_payload(
            app,
            current_history,
            markers,
            state,
            adapter_kind,
            event.event_type.as_deref(),
            &event.payload,
        );
    }

    Ok(false)
}

fn write_stream_content(
    content: &str,
    writer: Option<&Arc<std::sync::Mutex<std::fs::File>>>,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if let Some(w) = writer {
        let mut guard = w.lock().unwrap();
        let clean = strip_ansi_codes(content);
        guard.write_all(clean.as_bytes())?;
        guard.flush()?;
    }
    write_stream_content_to_terminal(content, markdown, dimmed)
}

fn write_stream_content_to_file(
    content: &str,
    mut writer: Option<&mut std::fs::File>,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        let clean = strip_ansi_codes(content);
        file.write_all(clean.as_bytes())?;
        file.flush()?;
    }
    Ok(())
}

fn write_stream_content_to_terminal(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if markdown.should_render(content) {
        markdown.write_chunk(content, dimmed)?;
        io::stdout().flush()?;
    } else {
        if dimmed {
            print!("{DIM}{content}{RESET}");
        } else {
            print!("{content}");
        }
        io::stdout().flush()?;
    }
    Ok(())
}
