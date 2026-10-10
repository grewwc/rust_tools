use super::*;
/// Compatible gateways like OpenCode sometimes return the complete DSML tool protocol as a single content snapshot.
/// Recognize such a full wrapper before body submission and terminal rendering, so we do not only recover via a stream-end fallback.
pub(super) fn recover_protocol_only_inline_tool_call_snapshot(
    chunk: &mut StreamChunk,
    merge_mode: StreamEventMergeMode,
    state: &StreamProcessingState,
) -> Vec<InternalToolCallStreamEvent> {
    let Some(choice) = chunk.choices.first_mut() else {
        return Vec::new();
    };
    if !choice.delta.tool_calls.is_empty() || choice.delta.content.trim().is_empty() {
        return Vec::new();
    }

    let normalized = normalize_inline_tool_call_markup(&choice.delta.content);
    let normalized = normalized.trim();
    if !normalized.starts_with("<tool_calls>") || !normalized.ends_with("</tool_calls>") {
        return Vec::new();
    }
    let Some(mut tool_calls) = recover_inline_tool_calls(normalized) else {
        return Vec::new();
    };

    choice.delta.content.clear();
    if matches!(merge_mode, StreamEventMergeMode::AppendMissingSuffix) {
        // A `.done` snapshot re-sends the full protocol already parsed from earlier deltas; filter only semantically equal calls,
        // so genuinely new parallel calls are not swallowed just because other tool calls already exist.
        tool_calls.retain(|tool_call| {
            !state
                .content
                .tool_calls_map
                .iter()
                .any(|(_, builder)| collected_tool_call_matches(builder, tool_call))
        });
    }

    let mut events = Vec::with_capacity(tool_calls.len().saturating_mul(3));
    for tool_call in tool_calls {
        events.push(InternalToolCallStreamEvent::Begin(tool_call.function.name));
        events.push(InternalToolCallStreamEvent::Args(
            tool_call.function.arguments,
        ));
        events.push(InternalToolCallStreamEvent::End);
    }
    events
}

pub(super) fn collected_tool_call_matches(
    builder: &ToolCallBuilder,
    tool_call: &crate::ai::types::ToolCall,
) -> bool {
    if builder.function_name != tool_call.function.name {
        return false;
    }
    match (
        serde_json::from_str::<serde_json::Value>(&builder.arguments),
        serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments),
    ) {
        (Ok(existing), Ok(incoming)) => existing == incoming,
        _ => builder.arguments.trim() == tool_call.function.arguments.trim(),
    }
}

/// Detect three consecutive, exactly-identical long fragments at the tail of the text (degenerate repetition loop).
///
/// Used for both reasoning_content and visible assistant output: under long tool-chain contexts the model may
/// verbatim-repeat one sentence in the chain or body; continuing to read only drains the output budget and persists junk.
/// Compares characters rather than bytes to handle Chinese correctly; the fragment must contain enough real content —
/// letters, digits or Chinese characters — so separator lines, whitespace or Markdown punctuation are not misjudged as a degeneration loop.
/// Number of trailing chars forming a degenerate verbatim repetition (one pattern repeated
/// REASONING_REPEAT_COUNT times at the tail), or None when no such tail exists.
///
/// Same detection rules as the former boolean detector, but also returns the repeated tail length so
/// callers can strip the looped junk before the partial text flows to retry/finalize. Runs on every
/// stream chunk, so it only keeps a tail large enough to cover the largest candidate fragment,
/// avoiding a long reasoning that degrades into repeatedly scanning the whole text as context grows.
pub(super) fn degenerate_repetition_strip_len(text: &str) -> Option<usize> {
    let mut chars = text
        .chars()
        .rev()
        .take(MAX_REASONING_REPEAT_CHARS * REASONING_REPEAT_COUNT)
        .collect::<Vec<_>>();
    chars.reverse();
    let max_pattern_len = (chars.len() / REASONING_REPEAT_COUNT).min(MAX_REASONING_REPEAT_CHARS);
    if max_pattern_len < MIN_REASONING_REPEAT_CHARS {
        return None;
    }

    for pattern_len in MIN_REASONING_REPEAT_CHARS..=max_pattern_len {
        let repeated_len = pattern_len * REASONING_REPEAT_COUNT;
        let repeated = &chars[chars.len() - repeated_len..];
        let pattern = &repeated[..pattern_len];
        if pattern.iter().filter(|ch| ch.is_alphanumeric()).count() < MIN_REASONING_REPEAT_CHARS / 2
        {
            continue;
        }
        if repeated[pattern_len..pattern_len * 2] == *pattern
            && repeated[pattern_len * 2..] == *pattern
        {
            if is_fold_placeholder_repeat_pattern(pattern) {
                continue;
            }
            return Some(repeated_len);
        }
    }
    None
}

pub(super) fn is_fold_placeholder_repeat_pattern(pattern: &[char]) -> bool {
    let pattern = pattern.iter().collect::<String>();
    let trimmed = pattern.trim();
    let Some(rest) = trimmed
        .strip_prefix('…')
        .or_else(|| trimmed.strip_prefix("..."))
    else {
        return false;
    };
    let rest = rest.trim();
    if rest == "more" {
        return true;
    }
    let mut words = rest.split_whitespace();
    let Some(count) = words.next() else {
        return false;
    };
    if !count.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    matches!(
        (words.next(), words.next(), words.next()),
        (Some("earlier"), Some("line" | "lines"), None)
    )
}

pub(super) fn has_degenerate_repetition(text: &str) -> bool {
    degenerate_repetition_strip_len(text).is_some()
}

#[derive(Clone, Copy)]
pub(super) enum StreamEventMergeMode {
    Append,
    AppendMissingSuffix,
}

pub(super) fn render_thinking_event(
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    event: &StreamTextEvent,
) -> Result<(), Box<dyn std::error::Error>> {
    if !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }

    match event {
        StreamTextEvent::OpenThinking => {
            flush_digest_filter_to_terminal(markers, state, false)?;
            if markers.subagent_preview_enabled() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            maybe_write_stream_content(
                &format!("\n{}\n", markers.thinking_tag),
                state,
                markers,
                true,
            )?;
        }
        StreamTextEvent::AppendThinking(text) => {
            if text.is_empty() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            // digest is extra image-understanding content meant for the model; the thinking channel's terminal display strips it too
            let terminal_text = state.render.digest_filter.push(text);
            if terminal_text.is_empty() {
                return Ok(());
            }
            if markers.subagent_preview_enabled() {
                write_subagent_content_folded(terminal_text.as_str(), state)?;
            } else {
                maybe_write_stream_content(terminal_text.as_str(), state, markers, true)?;
            }
        }
        StreamTextEvent::CloseThinking => {
            flush_digest_filter_to_terminal(markers, state, true)?;
            if markers.subagent_preview_enabled() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            if state.render.thinking_fold.active {
                finalize_thinking_fold(state)?;
            } else {
                maybe_write_stream_content(
                    &format!("{}\n", markers.end_thinking_tag),
                    state,
                    markers,
                    true,
                )?;
            }
        }
        StreamTextEvent::AppendContent(_) | StreamTextEvent::AppendHiddenMeta(_) => {}
    }

    Ok(())
}

pub(super) fn flush_digest_filter_to_terminal(
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    dimmed: bool,
) -> io::Result<()> {
    let residual = state.render.digest_filter.flush();
    if residual.is_empty() || !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }
    clear_waiting_hint(state)?;
    if markers.subagent_preview_enabled() {
        write_subagent_content_folded(&residual, state)
    } else {
        maybe_write_stream_content(&residual, state, markers, dimmed)
    }
}

pub(super) fn stream_text_event_to_content(
    event: &StreamTextEvent,
    markers: &StreamMarkers,
    merge_mode: StreamEventMergeMode,
    assistant_text: &str,
) -> Option<String> {
    // Thinking events may only go through render_thinking_event()'s terminal display path; they must not enter
    // assistant_text/current_history. Deliberately only visible body is returned here.
    if markers.subagent_preview_enabled() {
        return match event {
            StreamTextEvent::AppendContent(text) => match merge_mode {
                StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
                StreamEventMergeMode::AppendMissingSuffix => {
                    let suffix = unseen_suffix(assistant_text, text);
                    (!suffix.is_empty()).then_some(suffix)
                }
            },
            StreamTextEvent::OpenThinking
            | StreamTextEvent::AppendThinking(_)
            | StreamTextEvent::CloseThinking
            | StreamTextEvent::AppendHiddenMeta(_) => None,
        };
    }

    match event {
        StreamTextEvent::AppendContent(text) => match merge_mode {
            StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
            StreamEventMergeMode::AppendMissingSuffix => {
                let suffix = unseen_suffix(assistant_text, text);
                (!suffix.is_empty()).then_some(suffix)
            }
        },
        StreamTextEvent::OpenThinking
        | StreamTextEvent::AppendThinking(_)
        | StreamTextEvent::CloseThinking
        | StreamTextEvent::AppendHiddenMeta(_) => None,
    }
}

pub(super) fn unseen_suffix(existing: &str, incoming: &str) -> String {
    if incoming.is_empty() || existing.ends_with(incoming) {
        return String::new();
    }

    let leading_ws_len = incoming
        .char_indices()
        .find_map(|(idx, c)| (!c.is_whitespace()).then_some(idx))
        .unwrap_or(incoming.len());
    if leading_ws_len > 0 {
        let trimmed = &incoming[leading_ws_len..];
        if trimmed.is_empty() || existing.ends_with(trimmed) {
            return String::new();
        }
        if let Some(suffix) = unseen_suffix_after_visible_overlap(existing, trimmed) {
            return suffix;
        }
    }

    if let Some(suffix) = unseen_suffix_after_visible_overlap(existing, incoming) {
        return suffix;
    }

    if let Some(suffix) = unseen_suffix_whitespace_tolerant(existing, incoming) {
        return suffix;
    }
    incoming.to_string()
}

pub(super) fn unseen_suffix_after_visible_overlap(existing: &str, incoming: &str) -> Option<String> {
    let boundaries = incoming
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(incoming.len()))
        .collect::<Vec<_>>();

    for overlap_chars in (1..boundaries.len()).rev() {
        let split_idx = boundaries[overlap_chars];
        let overlap = &incoming[..split_idx];
        // Overlaps of pure whitespace (e.g. \n) are almost always false matches — models often
        // start a new paragraph with \n, and assistant_text often ends with \n. Only overlaps
        // containing visible characters count as real repetition.
        if existing.ends_with(overlap) && overlap.chars().any(|c| !c.is_whitespace()) {
            return Some(incoming[split_idx..].to_string());
        }
    }

    None
}

/// Whitespace-tolerant suffix dedup.
///
/// Tolerates whitespace differences from old history/abnormal providers: when `assistant_text` and the final
/// `response.output_text.done` snapshot differ only in whitespace, still avoid re-appending the whole already-streamed
/// snapshot as new content.
///
/// Aligns existing and incoming character by character with "whitespace skippable" to find the incoming prefix already
/// covered by existing, and returns the remaining incoming tail (original whitespace preserved). If all of incoming's visible
/// characters are covered, returns `Some("")`; if the visible characters cannot align, returns `None`.
pub(super) fn unseen_suffix_whitespace_tolerant(existing: &str, incoming: &str) -> Option<String> {
    let e: Vec<(usize, char)> = existing.char_indices().collect();
    let i: Vec<(usize, char)> = incoming.char_indices().collect();
    let (mut ei, mut ii) = (0usize, 0usize);

    // Skip leading whitespace of incoming (snapshots often start with a newline while assistant_text does not)
    while ii < i.len() && i[ii].1.is_whitespace() {
        ii += 1;
    }

    // last_matched_ii records the Vec index just after the last matched visible character in incoming,
    // used to locate the byte start of the "remaining uncovered tail" in incoming at the end.
    let mut last_matched_ii = ii;

    while ei < e.len() && ii < i.len() {
        let (ec, ic) = (e[ei].1, i[ii].1);
        if ec.is_whitespace() && ic.is_whitespace() {
            while ei < e.len() && e[ei].1.is_whitespace() {
                ei += 1;
            }
            while ii < i.len() && i[ii].1.is_whitespace() {
                ii += 1;
            }
            continue;
        }
        if ec.is_whitespace() {
            ei += 1;
            continue;
        }
        if ic.is_whitespace() {
            ii += 1;
            continue;
        }
        // Both sides are visible characters: they must be equal to count as aligned
        if ec == ic {
            last_matched_ii = ii + 1;
            ei += 1;
            ii += 1;
        } else {
            return None;
        }
    }

    // existing is exhausted; the bytes after last_matched_ii in incoming are the remaining (uncovered) tail.
    // If incoming is fully matched too, start_byte == incoming.len(), returning an empty string.
    let start_byte = i
        .get(last_matched_ii)
        .map(|(b, _)| *b)
        .unwrap_or(incoming.len());
    Some(incoming[start_byte..].to_string())
}

pub(super) fn flush_sse_event(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
) -> Result<StreamPayloadOutcome, Box<dyn std::error::Error>> {
    let Some(event) = framing::flush_sse_event(&mut state.framing) else {
        return Ok(StreamPayloadOutcome::default());
    };
    process_stream_payload(
        app,
        current_history,
        markers,
        state,
        adapter,
        event.event_type.as_deref(),
        &event.payload,
    )
}

pub(super) fn process_stream_line(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
    line: &str,
) -> Result<StreamPayloadOutcome, Box<dyn std::error::Error>> {
    if let Some(event) = framing::consume_sse_line(&mut state.framing, line) {
        return process_stream_payload(
            app,
            current_history,
            markers,
            state,
            adapter,
            event.event_type.as_deref(),
            &event.payload,
        );
    }

    Ok(StreamPayloadOutcome::default())
}

pub(crate) fn write_stream_content(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }
    write_stream_content_to_terminal(content, markdown, dimmed)
}

pub(super) fn write_stream_content_to_terminal(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if markdown.should_render(content) {
        markdown.write_chunk(content, dimmed)?;
        io::stdout().flush()?;
    } else {
        if dimmed {
            print!("{}{content}{RESET}", theme::current().accent_muted);
        } else {
            print!("{content}");
        }
        io::stdout().flush()?;
    }
    Ok(())
}
