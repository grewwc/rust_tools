use super::*;
/// Update the tool-argument stall timer: starts it when a tool call opens, resets it once no
/// tool call is open or the provider already declared how the response ended. See
/// `STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS`.
pub(super) fn update_tool_args_open_at(
    state: &StreamContentState,
    open_at: Option<Instant>,
) -> Option<Instant> {
    if state.finish_reason_seen
        || state.response_completed
        || state.response_incomplete
        || state.tool_calls_map.is_empty()
    {
        None
    } else {
        Some(open_at.unwrap_or_else(Instant::now))
    }
}

/// Whether an open tool call has been receiving arguments for at least `stall_timeout` without finishing.
pub(super) fn tool_args_stream_stalled(
    open_at: Option<Instant>,
    now: Instant,
    stall_timeout: Duration,
) -> bool {
    open_at.is_some_and(|open_at| now.duration_since(open_at) >= stall_timeout)
}

/// Track an open `function_call` output item from the Responses item events.
///
/// The provider declaring a tool call in progress means that call's arguments are being generated server-side, and
/// this wire can deliver the whole payload as one late delta (a 20KB `write_file` argument) with no event in
/// between, so that silence must not be read as a stalled connection. Reasoning items carry the same open/closed
/// state in their own parse result, which also resolves an `.added` line whose item status is already terminal;
/// only chunks can come from a `function_call` item here, so the two paths cannot double count.
pub(super) fn note_function_call_item_lifecycle(
    event_type: Option<&str>,
    parsed: &super::state::ParsedStreamPayload,
    content: &mut StreamContentState,
) {
    if !matches!(
        parsed,
        super::state::ParsedStreamPayload::Chunk(_)
            | super::state::ParsedStreamPayload::SnapshotChunk(_)
    ) {
        return;
    }
    let Some(name) = event_type else { return };
    if name.eq_ignore_ascii_case("response.output_item.added") {
        content.open_output_items = content.open_output_items.saturating_add(1);
    } else if name.eq_ignore_ascii_case("response.output_item.done") {
        content.open_output_items = content.open_output_items.saturating_sub(1);
    }
}

pub(super) struct ToolCallRenderChunk {
    pub(crate) function_name: String,
    pub(crate) arguments: String,
    pub(crate) open_line: bool,
}

pub(super) fn take_tool_call_render_chunk(
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

pub(super) fn open_tool_call_line(
    state: &mut StreamProcessingState,
    index: usize,
    function_name: &str,
) -> io::Result<()> {
    // The hint is about to name this call, so its argument-throughput window restarts here:
    // whatever the row later reports belongs to the tool call shown on that row.
    state.content.reset_tool_args_metrics();
    state.render.current_printing_index = Some(index);
    if runtime_ctx::terminal_output_enabled() && io::stdout().is_terminal() {
        print_tool_call_waiting_hint(state, function_name)?;
    }
    Ok(())
}

/// The terminal does not print tool-call arguments; the streaming receive phase only shows an erasable
/// tool-name status line, while the actual execution lines are printed uniformly by the tool execution layer.
pub(super) fn write_tool_call_arguments_stream(_arguments: &str) -> io::Result<()> {
    Ok(())
}

pub(super) fn process_external_tool_calls_delta(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    chunk: &StreamChunk,
    merge_mode: StreamEventMergeMode,
) -> bool {
    let Some(choice) = chunk.choices.first() else {
        return false;
    };

    let mut meaningful_progress = false;
    for stream_tool_call in &choice.delta.tool_calls {
        if stream_tool_call.id.is_empty()
            && stream_tool_call.tool_type.is_empty()
            && stream_tool_call.function.name.is_empty()
            && stream_tool_call.function.arguments.is_empty()
        {
            continue;
        }
        let index = match stream_tool_call.index {
            Some(idx) => idx,
            None => resolve_indexless_tool_call_key(&mut state.content, &stream_tool_call.id),
        };
        ensure_tool_calls_section_open(app, markers, state);

        let render_chunk = {
            let builder = state.content.tool_calls_map.entry(index).or_default();
            let before = (
                builder.id.len(),
                builder.tool_type.len(),
                builder.function_name.len(),
                builder.arguments.len(),
            );
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
            append_tool_call_arguments(
                &mut builder.arguments,
                &stream_tool_call.function.arguments,
                merge_mode,
            );
            let after = (
                builder.id.len(),
                builder.tool_type.len(),
                builder.function_name.len(),
                builder.arguments.len(),
            );
            meaningful_progress |= after != before;
            take_tool_call_render_chunk(state.render.current_printing_index, index, builder)
        };

        if let Some(render_chunk) = render_chunk {
            if render_chunk.open_line {
                let _ = open_tool_call_line(state, index, &render_chunk.function_name);
            }
            state
                .content
                .count_tool_arg_delta(estimate_stream_tokens(&render_chunk.arguments));
            let _ = write_tool_call_arguments_stream(&render_chunk.arguments);
            let _ = refresh_tool_call_rate_hint(state, &render_chunk.function_name);
        }
    }
    meaningful_progress
}

/// Incrementally resolves cumulative keys for chat-completions tool calls whose
/// `index` is missing. If everything fell onto the default key 0, parallel tool
/// calls would merge into one; instead group by id: reuse the key of an existing
/// builder with the same id, otherwise synthesize a stable key in
/// [10000, usize::MAX) from a hash of the id (real provider indexes are single
/// digits, so no collision). Parameter-continuation deltas with neither id nor
/// index attach to the most recent call without an index; if there is none, fall
/// back to the old-behavior key 0.
pub(super) fn resolve_indexless_tool_call_key(state: &mut StreamContentState, id: &str) -> usize {
    if !id.is_empty() {
        for (key, builder) in state.tool_calls_map.iter() {
            if !builder.id.is_empty() && builder.id == id {
                return *key;
            }
        }
        let mut hash = 10000u64;
        for byte in id.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        let key = hash as usize;
        state.last_indexless_tool_call_key = Some(key);
        return key;
    }
    state.last_indexless_tool_call_key.unwrap_or(0)
}

pub(super) fn append_tool_call_arguments(
    existing: &mut String,
    incoming: &str,
    merge_mode: StreamEventMergeMode,
) {
    if incoming.is_empty() {
        return;
    }

    match merge_mode {
        StreamEventMergeMode::Append => existing.push_str(incoming),
        StreamEventMergeMode::AppendMissingSuffix => {
            let suffix = unseen_suffix(existing, incoming);
            existing.push_str(&suffix);
        }
    }
}

/// Consume internal tool-call stream events. The return value indicates whether this batch detected a hallucinated
/// internal tool-protocol marker (`HallucinatedProtocolMarker`) — the caller then stops the stream and retries downgraded.
pub(super) fn process_internal_tool_calls(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    internal_tool_call_events: Vec<InternalToolCallStreamEvent>,
) -> (bool, bool) {
    let mut saw_hallucinated_marker = false;
    let mut meaningful_progress = false;
    for event in internal_tool_call_events {
        match event {
            InternalToolCallStreamEvent::Begin(function_name) => {
                if function_name.trim().is_empty() {
                    continue;
                }
                meaningful_progress = true;
                ensure_tool_calls_section_open(app, markers, state);

                let index = state.content.internal_tool_call_idx;
                let builder = state.content.tool_calls_map.entry(index).or_default();
                builder.id = format!("internal_{index}");
                builder.tool_type = "function".to_string();
                builder.function_name = function_name.clone();

                let _ = open_tool_call_line(state, index, &function_name);
            }
            InternalToolCallStreamEvent::Args(chunk) => {
                if chunk.is_empty() {
                    continue;
                }
                meaningful_progress = true;
                let index = state.content.internal_tool_call_idx;
                let function_name = {
                    let builder = state.content.tool_calls_map.entry(index).or_default();
                    if builder.function_name.is_empty() {
                        builder.id = format!("internal_{index}");
                        builder.tool_type = "function".to_string();
                    }
                    builder.arguments.push_str(&chunk);
                    builder.printed_arguments_len = builder.arguments.len();
                    builder.function_name.clone()
                };

                let _ = write_tool_call_arguments_stream(&chunk);
                state
                    .content
                    .count_tool_arg_delta(estimate_stream_tokens(&chunk));
                let _ = refresh_tool_call_rate_hint(state, &function_name);
            }
            InternalToolCallStreamEvent::End => {
                if state.render.current_printing_index == Some(state.content.internal_tool_call_idx)
                {
                    // The streaming phase no longer prints tool name/arguments (open_tool_call_line and
                    // write_tool_call_arguments_stream are no-ops), so only the color needs resetting here.
                    // Never use println! — it would insert a blank line between the `✓` and the following output
                    // (the external delta tool path never prints this line anyway).
                    if runtime_ctx::terminal_output_enabled() {
                        print!("\x1b[0m");
                    }
                    state.render.current_printing_index = None;
                    if runtime_ctx::terminal_output_enabled() {
                        let _ = io::stdout().flush();
                    }
                }
                state.content.internal_tool_call_idx += 1;
            }
            InternalToolCallStreamEvent::HallucinatedProtocolMarker => {
                // The streamer already strips the whole hallucinated "tool result" so nothing is shown; only the signal
                // is recorded here, and the caller stops the stream and takes the degenerate_repetition downgrade-retry path.
                saw_hallucinated_marker = true;
            }
        }
    }
    (saw_hallucinated_marker, meaningful_progress)
}
