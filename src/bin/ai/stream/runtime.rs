use std::io::{self, Write};
use std::time::Duration;

use colored::Colorize;

use crate::ai::{
    request::StreamChunk,
    types::{App, StreamOutcome, StreamResult, ToolCall, take_stream_cancelled},
};

use super::{
    MarkdownStreamRenderer,
    extract::{extract_chunk_text_with_tools, strip_ansi_codes},
    state::{
        ParsedStreamPayload, StreamChunkStep, StreamMarkers, StreamProcessingState,
    },
};

/// Maximum number of decode errors before giving up and returning partial content
const MAX_DECODE_ERRORS: usize = 3;
/// Delay in milliseconds between retry attempts on transient errors
const DECODE_ERROR_RETRY_DELAY_MS: u64 = 100;

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, state.thinking_open) {
            return Ok(result);
        }

        let chunk_result = tokio::select! {
            chunk = response.chunk() => chunk,
            _ = wait_for_interrupt(app) => {
                return Ok(cancelled_stream_result(state.thinking_open));
            }
        };

        match process_chunk_result(app, current_history, &markers, &mut state, chunk_result).await?
        {
            StreamChunkStep::Continue => {}
            StreamChunkStep::Stop => break,
            StreamChunkStep::Return(result) => return Ok(result),
        }
    }

    if let Some(result) = process_pending_tail(app, current_history, &markers, &mut state).await? {
        return Ok(result);
    }

    finalize_stream_response(app, &markers, state)
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
    chunk_result: Result<Option<T>, reqwest::Error>,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    match chunk_result {
        Ok(Some(chunk)) => {
            state.pending.extend_from_slice(chunk.as_ref());
            state.decode_error_count = 0;
            consume_pending_complete_lines(app, current_history, markers, state).await
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
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    // Move the pending buffer out so line slices can borrow from it while `state`
    // remains available for mutation inside `process_stream_line()`.
    let mut pending = std::mem::take(&mut state.pending);
    let mut should_stop = false;
    let mut consumed = 0usize;
    while let Some(line_end_rel) = pending[consumed..].iter().position(|b| *b == b'\n') {
        let line_end = consumed + line_end_rel + 1;
        let line = match std::str::from_utf8(&pending[consumed..line_end]) {
            Ok(line) => line,
            Err(err) => {
                if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                    state.pending = pending;
                    return Ok(StreamChunkStep::Return(result));
                }
                consumed = line_end;
                continue;
            }
        };
        if process_stream_line(app, current_history, markers, state, line)? {
            should_stop = true;
            consumed = line_end;
            break;
        }
        consumed = line_end;
    }
    if consumed != 0 {
        pending.drain(..consumed);
    }
    state.pending = pending;
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
) -> Result<Option<StreamResult>, Box<dyn std::error::Error>> {
    if state.pending.is_empty() {
        return Ok(None);
    }

    let pending = std::mem::take(&mut state.pending);
    let line = match std::str::from_utf8(&pending) {
        Ok(line) => line,
        Err(err) => {
            if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                return Ok(Some(result));
            }
            state.pending = pending;
            return Ok(None);
        }
    };
    if !line.is_empty() {
        let _ = process_stream_line(app, current_history, markers, state, line)?;
    }
    state.pending = pending;
    Ok(None)
}

fn finalize_stream_response(
    app: &mut App,
    markers: &StreamMarkers,
    mut state: StreamProcessingState,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    if state.thinking_open {
        write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_mut(),
            &mut state.markdown,
            true,
        )?;
    }

    state.markdown.flush_pending()?;

    if state.current_printing_index.is_some() {
        println!("\x1b[0m)");
    }

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(false));
    }

    let tool_calls: Vec<ToolCall> = state
        .tool_calls_map
        .into_values()
        .map(|builder| builder.build())
        .collect();

    let outcome = if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        StreamOutcome::Completed
    };

    Ok(StreamResult {
        outcome,
        tool_calls,
        assistant_text: state.assistant_text,
        hidden_meta: state.hidden_meta,
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
    state.decode_error_count += 1;
    eprintln!(
        "[Warning] 读取响应流时出错：{} (错误次数：{}/{})",
        err, state.decode_error_count, MAX_DECODE_ERRORS
    );

    if take_stream_cancelled(app) {
        return Some(cancelled_stream_result(state.thinking_open));
    }

    if state.decode_error_count <= MAX_DECODE_ERRORS {
        eprintln!("[Warning] 尝试继续读取...");
        tokio::time::sleep(Duration::from_millis(DECODE_ERROR_RETRY_DELAY_MS)).await;
        return None;
    }

    eprintln!("[Error] 响应流读取失败，返回已收集的内容");

    if state.thinking_open {
        let _ = write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_mut(),
            &mut state.markdown,
            true,
        );
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    let _ = state.markdown.flush_pending();

    Some(StreamResult {
        outcome: StreamOutcome::Completed,
        tool_calls: state.tool_calls_map.drain().map(|(_, b)| b.build()).collect(),
        assistant_text: std::mem::take(&mut state.assistant_text),
        hidden_meta: String::new(),
    })
}

fn ensure_tool_calls_section_open(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
) {
    if state.printed_tool_calls_header {
        return;
    }

    if state.thinking_open {
        let _ = write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_mut(),
            &mut state.markdown,
            true,
        );
        state.thinking_open = false;
    }
    let _ = state.markdown.flush_pending();
    println!("\n{}", "[Tool Calls]".yellow());
    state.printed_tool_calls_header = true;
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

        let builder = state.tool_calls_map.entry(index).or_default();
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

        if !builder.function_name.is_empty() {
            if state.current_printing_index != Some(index) {
                if state.current_printing_index.is_some() {
                    println!("\x1b[0m)");
                }
                state.current_printing_index = Some(index);
                print!("  - {}(\x1b[2m", builder.function_name.cyan());
                let _ = io::stdout().flush();
                print!("{}", builder.arguments);
                let _ = io::stdout().flush();
            } else if !stream_tool_call.function.arguments.is_empty() {
                print!("{}", stream_tool_call.function.arguments);
                let _ = io::stdout().flush();
            }
        }
    }
}

fn process_hidden_meta_and_visible_content(
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    mut content: String,
) -> String {
    if content.is_empty() {
        return content;
    }

    let mut visible = String::with_capacity(content.len());
    let hb: Vec<char> = markers.hidden_begin.chars().collect();
    let he: Vec<char> = markers.hidden_end.chars().collect();
    for ch in content.chars() {
        if !state.hidden_open {
            if state.hidden_begin_match < hb.len() && ch == hb[state.hidden_begin_match] {
                state.hidden_begin_match += 1;
                if state.hidden_begin_match == hb.len() {
                    state.hidden_open = true;
                    state.hidden_begin_match = 0;
                }
            } else {
                if state.hidden_begin_match > 0 {
                    for k in 0..state.hidden_begin_match {
                        visible.push(hb[k]);
                    }
                    state.hidden_begin_match = 0;
                }
                visible.push(ch);
            }
        } else if state.hidden_end_match < he.len() && ch == he[state.hidden_end_match] {
            state.hidden_end_match += 1;
            if state.hidden_end_match == he.len() {
                state.hidden_open = false;
                state.hidden_end_match = 0;
            }
        } else {
            if state.hidden_end_match > 0 {
                for k in 0..state.hidden_end_match {
                    state.hidden_meta.push(he[k]);
                }
                state.hidden_end_match = 0;
            }
            state.hidden_meta.push(ch);
        }
    }
    content = visible;
    content
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

        let builder = state.tool_calls_map.entry(state.internal_tool_call_idx).or_default();
        builder.id = id;
        builder.tool_type = tool_type;
        builder.function_name = function_name;
        builder.arguments = arguments;

        if state.current_printing_index.is_some() {
            println!("\x1b[0m)");
            state.current_printing_index = None;
        }

        print!(
            "  - {}(\x1b[2m{}\x1b[0m)",
            builder.function_name.cyan(),
            builder.arguments
        );
        println!();
        let _ = io::stdout().flush();

        state.internal_tool_call_idx += 1;
    }
}

fn commit_visible_content(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    content: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if content.is_empty() {
        return Ok(());
    }

    write_stream_content(
        content.as_str(),
        app.writer.as_mut(),
        &mut state.markdown,
        state.thinking_open,
    )?;
    if state.thinking_open {
        return Ok(());
    }

    let text = if content.contains(&markers.end_thinking_tag) {
        content.replace(&markers.end_thinking_tag, "")
    } else {
        content
    };
    let text = text.trim_matches('\n');
    current_history.reserve(text.len());
    state.assistant_text.reserve(text.len());
    current_history.push_str(text);
    state.assistant_text.push_str(text);

    Ok(())
}

fn parse_stream_payload(line: &str) -> ParsedStreamPayload {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return ParsedStreamPayload::Ignore;
    }

    let payload = trimmed.trim_start_matches("data:").trim();
    if payload.is_empty() {
        return ParsedStreamPayload::Ignore;
    }
    if payload == "[DONE]" {
        return ParsedStreamPayload::Done;
    }

    match serde_json::from_str(payload) {
        Ok(chunk) => ParsedStreamPayload::Chunk(chunk),
        Err(err) => {
            eprintln!("handleResponse error {err}");
            eprintln!("======> response: ");
            eprintln!("{payload}");
            eprintln!("<======");
            ParsedStreamPayload::Ignore
        }
    }
}

fn process_stream_line(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    line: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let chunk = match parse_stream_payload(line) {
        ParsedStreamPayload::Ignore => return Ok(false),
        ParsedStreamPayload::Done => return Ok(true),
        ParsedStreamPayload::Chunk(chunk) => chunk,
    };

    let mut reached_finish_reason = false;
    if let Some(choice) = chunk.choices.first() {
        reached_finish_reason = choice.finish_reason.is_some();
    }

    process_external_tool_calls_delta(app, markers, state, &chunk);

    let (mut content, internal_tool_calls) = extract_chunk_text_with_tools(
        &chunk,
        &markers.thinking_tag,
        &markers.end_thinking_tag,
        &mut state.thinking_open,
    );

    content = process_hidden_meta_and_visible_content(markers, state, content);
    process_internal_tool_calls(app, markers, state, internal_tool_calls);

    if content.is_empty() {
        return Ok(reached_finish_reason);
    }
    commit_visible_content(app, current_history, markers, state, content)?;
    if state.thinking_open {
        return Ok(false);
    }

    Ok(reached_finish_reason)
}

fn write_stream_content(
    content: &str,
    mut writer: Option<&mut std::fs::File>,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        let clean = strip_ansi_codes(content);
        file.write_all(clean.as_bytes())?;
        file.flush()?;
    }

    if markdown.should_render(content) {
        markdown.write_chunk(content, dimmed)?;
        io::stdout().flush()?;
    } else {
        if dimmed {
            print!("{}", content.dimmed());
        } else {
            print!("{content}");
        }
        io::stdout().flush()?;
    }
    Ok(())
}
