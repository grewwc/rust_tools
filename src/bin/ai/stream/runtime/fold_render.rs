use super::*;
/// Physical-row budget the live fold body may occupy so the whole window (header + body) stays inside the visible
/// viewport. Returning `usize::MAX` disables clamping (non-tty / unknown height), preserving the previous behavior
/// exactly. On a real terminal this is `viewport_rows - FOLD_VIEWPORT_RESERVED_ROWS`, floored at 1 so an extremely
/// short pane still shows one body row. In normal terminals this budget dwarfs `max_visible_lines`, so the effective
/// window size — and every rendered byte — is unchanged; it only tightens when the viewport genuinely cannot hold
/// the configured window, which is exactly the case that used to leak stacked markers.
pub(super) fn fold_body_viewport_row_budget() -> usize {
    let viewport = io::stdout().is_terminal().then(raw_terminal_rows);
    fold_body_row_budget_for(viewport)
}

/// Pure viewport-budget arithmetic, split out so the clamp can be unit-tested without a real tty. `None` means the
/// viewport height is unknown (non-tty / pipe / test) and clamping is disabled.
pub(super) fn fold_body_row_budget_for(viewport_rows: Option<usize>) -> usize {
    match viewport_rows {
        Some(rows) => rows.saturating_sub(FOLD_VIEWPORT_RESERVED_ROWS).max(1),
        None => usize::MAX,
    }
}

/// Folded rendering of thinking content: maintain a rewritable window starting at the first line,
/// always showing only the most recent N lines in the terminal and folding the rest into one summary line.
pub(super) fn write_thinking_content_folded(
    content: &str,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }
    if crate::ai::background::serve_live_streaming() {
        // Serve children stream to a chat client, not a terminal: cursor
        // rewrites are meaningless across the process boundary, and unfolded
        // thinking lines would be indistinguishable from the answer. Frame
        // the thinking lifecycle instead and keep it off stdout (the SSE
        // line pump would otherwise forward it as plain message events).
        return write_thinking_content_serve(content, markers);
    }
    let fold = &mut state.render.thinking_fold;

    if fold.max_visible_lines == usize::MAX {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    // Control lines must be exclusive markers; they must not be interleaved with body text.
    if is_standalone_stream_marker(content, &markers.thinking_tag) {
        if !fold.active {
            if state.render.markdown.has_unfinished_line() {
                write_stream_content_to_terminal("\n", &mut state.render.markdown, false)?;
            }
            fold.active = true;
        }
        return thinking_fold_redraw(
            fold_header_rate(&state.content, Instant::now()).as_deref(),
            fold,
        );
    }

    if !fold.active {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    append_fold_content(fold, content);

    thinking_fold_redraw(
        fold_header_rate(&state.content, Instant::now()).as_deref(),
        fold,
    )
}

/// Classify one serve-mode thinking chunk into an ordered frame sequence.
///
/// The common cases are a standalone open/close marker or a plain body
/// chunk. Chunk granularity is model-driven rather than line-driven, so the
/// close marker can also arrive glued to body text; a standalone-only check
/// would miss it and leave the remote fold open until end-of-stream, letting
/// the answer interleave with the chat client's thinking status line.
/// Splitting here keeps the close on time. Text after the marker is already
/// answer text (the content path never sees this chunk), so it is returned
/// as a delta instead of being swallowed by the fold.
pub(super) fn split_serve_thinking_chunk<'a>(
    content: &'a str,
    thinking_tag: &str,
    end_thinking_tag: &str,
) -> Vec<(crate::ai::background::ServeLiveKind, &'a str)> {
    use crate::ai::background::ServeLiveKind;
    if is_standalone_stream_marker(content, thinking_tag) {
        return vec![(ServeLiveKind::ThinkingStart, "")];
    }
    if is_standalone_stream_marker(content, end_thinking_tag) {
        return vec![(ServeLiveKind::ThinkingDone, "")];
    }
    if !end_thinking_tag.is_empty() {
        if let Some((head, tail)) = content.split_once(end_thinking_tag) {
            let mut out = Vec::with_capacity(3);
            if !head.is_empty() {
                out.push((ServeLiveKind::Thinking, head));
            }
            out.push((ServeLiveKind::ThinkingDone, ""));
            if !tail.trim().is_empty() {
                out.push((ServeLiveKind::Delta, tail));
            }
            return out;
        }
    }
    vec![(ServeLiveKind::Thinking, content)]
}

/// Serve-mode thinking sink: publish the classified frames for the chat
/// client to fold remotely. Everything returns here, so nothing thinking
/// related reaches child stdout.
pub(super) fn write_thinking_content_serve(content: &str, markers: &StreamMarkers) -> io::Result<()> {
    use crate::ai::background::publish_serve_frame;
    for (kind, text) in
        split_serve_thinking_chunk(content, &markers.thinking_tag, &markers.end_thinking_tag)
    {
        publish_serve_frame(kind, text);
    }
    Ok(())
}

pub(super) fn write_subagent_content_folded(
    content: &str,
    state: &mut StreamProcessingState,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }

    let fold = &mut state.render.subagent_fold;
    if fold.max_visible_lines == usize::MAX {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, false);
    }

    if !fold.active {
        if state.render.markdown.has_unfinished_line() {
            write_stream_content_to_terminal("\n", &mut state.render.markdown, false)?;
        }
        fold.active = true;
    }

    append_fold_content(fold, content);
    thinking_fold_redraw(None, fold)
}

pub(super) fn append_fold_content(fold: &mut super::state::ThinkingFoldState, content: &str) {
    for ch in content.chars() {
        if ch == '\n' {
            let completed_line = std::mem::take(&mut fold.current_line);
            if fold.skip_blank_lines && completed_line.trim().is_empty() {
                continue;
            }
            fold.total_lines += 1;
            fold.recent_lines.push_back(completed_line);
            while fold.recent_lines.len() > fold.max_visible_lines {
                fold.recent_lines.pop_front();
            }
        } else {
            fold.current_line.push(ch);
        }
    }
}

/// Redraw the fold header and body within the viewport; cached row counts cover only the body.
pub(super) fn thinking_fold_redraw(
    rate: Option<&str>,
    fold: &mut super::state::ThinkingFoldState,
) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    thinking_fold_redraw_to(&mut out, rate, fold)
}

pub(super) fn thinking_fold_redraw_to(
    out: &mut impl Write,
    rate: Option<&str>,
    fold: &mut super::state::ThinkingFoldState,
) -> io::Result<()> {
    // Measure against the emulator's reflowed width before any counting: inside the
    // window where xterm.js has already rewrapped the drawn rows but the PTY winsize
    // has not arrived, ioctl reports the old width and every span below would land
    // short, stranding the previous header. A no-op when no listener can answer.
    super::side_note_input::refresh_true_width();
    // Erase the region currently on screen — header rows plus body rows — and redraw the header on the
    // region's first row on every redraw, so the window stays anchored where the fold started.
    //
    // Both counts are recomputed from the text that was actually written, at the width in force now.
    // When the refresh answered, that width is the reflowed one; when it could not, the ioctl value
    // still matches the screen between resizes, and live rows are wrapped at the same width (see
    // `render::markdown`), so a wide terminal shows them in full.
    //
    // The first redraw (activation) has nothing on screen above the cursor, so it erases nothing and the
    // header lands at the fold's start position as before; every later redraw has a header (and possibly
    // an empty body) above the cursor that must be cleared and reprinted.
    let body_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    let erase_rows = if fold.header_drawn {
        // The header ends with CRLF. With no body, the cursor is on the blank
        // row below it, so the erase span must include that row as well.
        body_rows
            .max(1)
            .saturating_add(thinking_fold_header_rendered_rows(fold))
    } else {
        body_rows
    };
    erase_fold_body(out, erase_rows)?;
    if fold.active {
        fold.header_rendered_line = write_fold_header(out, rate, fold)?;
        fold.header_drawn = true;
    }

    // Bound the live window to the viewport so header + body never scroll past the top; otherwise the relative-cursor
    // erase above cannot reach the previous window and each redraw leaks a stacked marker line. Restored right after so
    // the configured `max_visible_lines` (and the model-facing reasoning buffer) is untouched.
    let saved_max_visible_lines = fold.max_visible_lines;
    fold.max_visible_lines = fold.max_visible_lines.min(fold_body_viewport_row_budget());
    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let clamped_max_visible_lines = fold.max_visible_lines;
    THINKING_FOLD_BODY_BUF.with(|buf| -> io::Result<()> {
        let mut buf = buf.borrow_mut();
        let (body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
            &body_lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            clamped_max_visible_lines,
            &mut buf,
        );
        if !buf.is_empty() {
            out.write_all(buf.as_bytes())?;
        }
        fold.window_rows = body_rows;
        fold.rendered_body_lines = rendered_body_lines;
        Ok(())
    })?;
    fold.max_visible_lines = saved_max_visible_lines;
    out.flush()?;
    Ok(())
}

/// Print the fold header, leaving the cursor at the start of the first body row.
///
/// The written plain text is returned so the caller can keep it on the fold: later erases recompute the
/// header's physical row count from it, at the width current then (see
/// `thinking_fold_header_rendered_rows`).
pub(super) fn write_fold_header(
    out: &mut impl Write,
    rate: Option<&str>,
    fold: &super::state::ThinkingFoldState,
) -> io::Result<String> {
    let mut label = fold.header_label.clone();
    // Live reasoning throughput rides on the fold header: the header is the
    // renderer's own redraw target (unlike the one-shot model/session status
    // line printed by request/transport.rs), so appending here needs no extra
    // cursor movement and stays inside the fold's erase span.
    if let Some(rate) = rate {
        label.push_str(" · ");
        label.push_str(rate);
    }
    let reserve_cols = fold.rewrite_right_margin_cols;
    write_fold_status_line(out, &label, reserve_cols)
}

/// Both redraw and completion erase the header by moving up its row count. Bound the entire decorated
/// line, including indentation and live metrics, to a single live-region row and return the plain text
/// that was written; the erase steps recompute the row count from that text, because a narrowed terminal
/// re-wraps the header row that is already on screen. This only clips the status display, never the
/// underlying content.
pub(super) fn write_fold_status_line(
    out: &mut impl Write,
    label: &str,
    reserve_cols: usize,
) -> io::Result<String> {
    let line = clamp_line_to_terminal_row_with_reserve(&format!("  {label}"), reserve_cols);
    write!(out, "{}{line}\x1b[0m\r\n", theme::current().accent_muted)?;
    Ok(line)
}

/// Write the final header directly when thinking ends; used for an empty fold that never wrote an in-progress header.
pub(super) fn write_thinking_fold_completion_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
    line_count: usize,
) -> io::Result<()> {
    write_fold_status_line(
        out,
        &format!("{} · {line_count} lines", fold.footer_label),
        fold.rewrite_right_margin_cols,
    )?;
    Ok(())
}

/// Rewrite the anchored `○ thinking` in place to the completed state instead of printing a separate `✓ thinking` below the body.
///
/// `erase_fold_body` moved the cursor back to the first body line under the header, so the header sits
/// directly above the cursor and is cleared row by row before being rewritten. Its row count comes from
/// the text that was written rather than a fixed one row: after a narrowing resize the terminal re-wraps
/// the old header, and a one-row erase would leave the header's first row behind.
pub(super) fn replace_thinking_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
    line_count: usize,
) -> io::Result<()> {
    erase_rows_above_cursor(out, thinking_fold_header_rendered_rows(fold))?;
    write_thinking_fold_completion_header(out, fold, line_count)
}

/// After rendering the body the cursor rests on the last physical line, not on an extra blank line; redraws therefore only need to move up
/// `rows - 1`; returning to the line start before erasing to the screen bottom covers reflow lines produced by a narrowed window.
pub(super) fn erase_fold_body(out: &mut impl Write, rows: usize) -> io::Result<()> {
    if rows == 0 {
        return Ok(());
    }
    write!(out, "\r")?;
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    // CSI 0J cannot be used: it clears from the first body line to the end of the physical screen, crossing the DECSTBM scroll
    // region and wiping out the side-note composer at the bottom. Clear only the rendered body window line by line, then restore
    // the cursor to the first body line so relative-cursor semantics of later redraws stay unchanged.
    for row in 0..rows {
        write!(out, "\r\x1b[2K")?;
        if row + 1 < rows {
            write!(out, "\x1b[1B")?;
        }
    }
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    write!(out, "\r")
}

/// Final rendering when thinking ends: overwrite the body window and turn the anchored `○` into `✓` in place.
pub(crate) fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    finalize_fold(&mut state.render.thinking_fold, true)
}

pub(super) fn finalize_subagent_preview_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    // Subagent preview keeps the window body at the end: the subagent's final output should stay visible in the terminal,
    // so the "forced 0-line pure summary" ending used by thinking (which would fold the key conclusions too) is not applied.
    finalize_fold(&mut state.render.subagent_fold, false)
}

pub(super) fn finalize_fold(
    fold: &mut super::state::ThinkingFoldState,
    collapse_body: bool,
) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    finalize_fold_to(&mut out, fold, collapse_body)
}

/// Fold-finalize writer implementation; extracted behind a writer so regression tests can verify terminal cursor sequences precisely.
pub(super) fn finalize_fold_to(
    mut out: &mut impl Write,
    fold: &mut super::state::ThinkingFoldState,
    collapse_body: bool,
) -> io::Result<()> {
    // Same reflowed-width discipline as `thinking_fold_redraw_to`: the header rewrite
    // below moves the cursor up by rows recomputed at the width in force now.
    super::side_note_input::refresh_true_width();
    if !fold.active {
        return Ok(());
    }

    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    erase_fold_body(&mut out, erase_rows)?;
    // The thinking completed state replaces the in-progress header instead of printing another footer below the body.
    // If the fold never actually landed, write the completed state directly to avoid a brief `○ thinking`.
    let line_count = fold
        .total_lines
        .saturating_add(usize::from(!fold.current_line.is_empty()));
    if collapse_body {
        if fold.header_drawn {
            replace_thinking_fold_header(&mut out, fold, line_count)?;
        } else {
            write_thinking_fold_completion_header(&mut out, fold, line_count)?;
            fold.header_drawn = true;
        }
    } else if !fold.header_drawn {
        // Subagent preview keeps the existing two-line header/footer layout.
        fold.header_rendered_line = write_fold_header(&mut out, None, fold)?;
        fold.header_drawn = true;
    }

    // Thinking finalize folds to a pure summary (0-line window) with no body lines: the tail of thinking often restates
    // conclusions/questions, and keeping visible lines would duplicate the final answer that follows in the terminal. Subagent
    // preview does not fold — it keeps the recent visible window lines so the subagent's final output stays visible.
    let saved_max_visible_lines = fold.max_visible_lines;
    if collapse_body {
        fold.max_visible_lines = 0;
    }
    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let max_visible_rows = fold.max_visible_lines;
    let mut final_body_rows = 0usize;
    THINKING_FOLD_BODY_BUF.with(|buf| -> io::Result<()> {
        let mut buf = buf.borrow_mut();
        let (body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
            &body_lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            max_visible_rows,
            &mut buf,
        );
        fold.max_visible_lines = saved_max_visible_lines;
        if !buf.is_empty() {
            out.write_all(buf.as_bytes())?;
        }
        fold.window_rows = body_rows;
        fold.rendered_body_lines = rendered_body_lines;
        final_body_rows = body_rows;
        Ok(())
    })?;

    if !collapse_body {
        // Subagent preview keeps its footer; thinking's scale info was already written into the in-place-replaced header.
        if final_body_rows > 0 {
            out.write_all(b"\r\n")?;
        }
        write!(
            out,
            "  {}{} · {line_count} lines\x1b[0m\r\n",
            theme::current().accent_muted,
            fold.footer_label,
        )?;
    } else if final_body_rows > 0 {
        // The collapsed window ends on the one-line summary marker (the body
        // is rendered without a trailing newline so streaming redraws do not
        // scroll). Finalize is terminal: release the cursor onto a fresh line,
        // otherwise the next output — the deferred-body "generating…" hint or
        // the final answer echoed by the driver — concatenates onto the marker
        // row ("… N earlier lines   ⠋ generating…"). With zero body rows the
        // completion header already ends with CRLF, so no extra blank line.
        out.write_all(b"\r\n")?;
    }
    out.flush()?;

    // Reset fold state
    fold.reset();
    Ok(())
}

pub(super) fn thinking_fold_hidden_count(fold: &super::state::ThinkingFoldState) -> usize {
    let current_line = usize::from(!fold.current_line.is_empty());
    fold.total_lines
        .saturating_add(current_line)
        .saturating_sub(fold.max_visible_lines)
}

pub(super) fn thinking_fold_visible_lines(fold: &super::state::ThinkingFoldState) -> Vec<&str> {
    // 0-line window = pure summary mode: even the current incomplete line is hidden, so restated conclusions cannot leak to the terminal.
    if fold.max_visible_lines == 0 {
        return Vec::new();
    }
    let current_line = usize::from(!fold.current_line.is_empty());
    let visible_completed = fold.max_visible_lines.saturating_sub(current_line);
    let completed_skip = fold.recent_lines.len().saturating_sub(visible_completed);
    let mut visible = fold
        .recent_lines
        .iter()
        .skip(completed_skip)
        .map(String::as_str)
        .collect::<Vec<_>>();
    if current_line > 0 {
        visible.push(fold.current_line.as_str());
    }
    visible
}

/// Physical rows the written rows of a fold region occupy on a terminal `cols` columns wide.
pub(super) fn thinking_fold_rendered_body_rows(fold: &super::state::ThinkingFoldState) -> usize {
    fold.rendered_body_lines
        .iter()
        .map(|line| live_preview_cursor_rows(line))
        .sum()
}

/// Physical rows the anchored fold header currently occupies. The header is written clamped to one
/// live-region row, but a terminal that narrowed re-wraps a row that is already on screen, so the erase
/// recomputes the footprint from the stored text (at the live width) instead of assuming a single row.
pub(super) fn thinking_fold_header_rendered_rows(fold: &super::state::ThinkingFoldState) -> usize {
    if fold.header_rendered_line.is_empty() {
        1
    } else {
        live_preview_cursor_rows(&fold.header_rendered_line)
    }
}

/// Erase `rows` rows whose last row sits directly **above** the cursor — the shape written by
/// `write_fold_status_line` and `write_waiting_hint_line`, both of which end with CRLF — then park the
/// cursor on the first of the erased rows so the next write lands where that region started.
pub(super) fn erase_rows_above_cursor(out: &mut impl Write, rows: usize) -> io::Result<()> {
    if rows == 0 {
        return Ok(());
    }
    write!(out, "\r\x1b[{rows}A")?;
    for row in 0..rows {
        write!(out, "\r\x1b[2K")?;
        if row + 1 < rows {
            write!(out, "\x1b[1B")?;
        }
    }
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    Ok(())
}

/// Park a live fold frame before an out-of-band terminal line (a mid-stream warning the
/// stream then continues past). Fold redraws erase relative to the current cursor, so any
/// line printed while a frame is live desyncs the cursor: every later redraw then misses the
/// old header and stacks another full `○ thinking` row below it. Erasing the frame here keeps
/// buffers and `active`, so the next redraw draws one fresh frame below the intruder line.
/// Mirrors the erase half of `finalize_fold_to`; the resume half is the normal redraw.
pub(super) fn suspend_fold_frame(
    out: &mut impl Write,
    fold: &mut super::state::ThinkingFoldState,
) -> io::Result<()> {
    if !fold.active {
        return Ok(());
    }
    super::side_note_input::refresh_true_width();
    let body_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    let erase_rows = if fold.header_drawn {
        body_rows
            .max(1)
            .saturating_add(thinking_fold_header_rendered_rows(fold))
    } else {
        body_rows
    };
    erase_fold_body(out, erase_rows)?;
    out.flush()?;
    fold.header_drawn = false;
    fold.window_rows = 0;
    fold.rendered_body_lines.clear();
    Ok(())
}

/// Park every live terminal region immediately before a mid-stream diagnostic line, while the
/// cursor is still where the live regions left it. Folds are suspended (see
/// `suspend_fold_frame`); the waiting hint is cleared.
pub(super) fn suspend_live_terminal_regions(state: &mut StreamProcessingState) -> io::Result<()> {
    if !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }
    let stdout = io::stdout();
    let mut out = stdout.lock();
    suspend_fold_frame(&mut out, &mut state.render.thinking_fold)?;
    suspend_fold_frame(&mut out, &mut state.render.subagent_fold)?;
    drop(out);
    clear_waiting_hint(state)
}

/// Settle every live terminal region before a stream-stopping diagnostic line: same
/// cursor-desync hazard as `suspend_live_terminal_regions`, but the stream ends here, so close
/// the frames with their completion headers instead of parking them for a redraw that never
/// comes. Returns whether the thinking fold was closed, so the caller can skip the plain
/// (non-fold) end-of-thinking marker it would otherwise print below the completion header.
pub(super) fn finalize_live_folds_before_diagnostic(state: &mut StreamProcessingState) -> bool {
    if !runtime_ctx::terminal_output_enabled() {
        return false;
    }
    let mut settled = false;
    if state.render.thinking_fold.active {
        let _ = finalize_thinking_fold(state);
        settled = true;
    }
    if state.render.subagent_fold.active {
        let _ = finalize_subagent_preview_fold(state);
    }
    let _ = clear_waiting_hint(state);
    settled
}

pub(super) fn thinking_fold_window_lines(fold: &super::state::ThinkingFoldState) -> (Vec<String>, usize) {
    let hidden_count = thinking_fold_hidden_count(fold);
    let visible_lines = thinking_fold_visible_lines(fold);
    // Nothing to show below the header: a `… N earlier lines` marker alone would
    // repeat the header's own `· N lines` count and point at lines that are
    // nowhere on screen, so the fold renders header-only in that state.
    if visible_lines.is_empty() {
        return (Vec::new(), 0);
    }

    let mut lines = Vec::with_capacity(visible_lines.len() + usize::from(hidden_count > 0));
    let marker_lines = usize::from(hidden_count > 0);
    if hidden_count > 0 {
        lines.push(format!("… {hidden_count} earlier lines"));
    }
    for line in visible_lines {
        lines.push(line.to_string());
    }
    (lines, marker_lines)
}

/// Render the **body** of the fold window (fold summary + recent visible lines), without the header.
/// The header is anchored and printed separately by `write_fold_header`. Returns the number of physical body lines; the body does not
/// end with a newline and the cursor always stays on the last line, so xterm.js does not interpret a trailing LF as extra scrolling.
pub(super) fn render_thinking_fold_window(fold: &super::state::ThinkingFoldState) -> (String, usize) {
    let (lines, marker_lines) = thinking_fold_window_lines(fold);
    THINKING_FOLD_BODY_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        let (rows, _) = render_thinking_fold_window_lines(
            &lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            fold.max_visible_lines,
            &mut buf,
        );
        (buf.clone(), rows)
    })
}

// Scratch buffer reused across thinking-fold body renders so each redraw does not
// rebuild a zero-capacity String. Purely an allocation-reuse optimization: the
// renderer clears the buffer before every rebuild, so the bytes handed to the
// terminal are identical to building a fresh String each time.
thread_local! {
    static THINKING_FOLD_BODY_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Make control characters literal before wrapping a rewritable fold row. Raw tabs
/// move to terminal-defined tab stops, while CR/ESC can move the cursor independently
/// of cell widths; either invalidates the erase footprint and leaks previous frames.
/// Use fixed four-space tab indentation and visible escapes for other controls.
/// This is display-only: the fold buffers and canonical model content stay unchanged.
pub(super) fn fold_display_line(line: &str) -> String {
    let mut display = String::with_capacity(line.len());
    for ch in line.chars() {
        match ch {
            '\t' => display.push_str("    "),
            ch if ch.is_control() => display.extend(ch.escape_default()),
            ch => display.push(ch),
        }
    }
    display
}

/// Render the **body** of the fold window (fold summary + recent visible lines), without the header.
/// The header is anchored and printed separately by `write_fold_header`. Writes the body into
/// `out` (cleared first) and returns the number of physical body lines plus the plain-text rows
/// kept for later width recomputation; the body does not end with a newline and the cursor always
/// stays on the last line, so xterm.js does not interpret a trailing LF as extra scrolling.
pub(super) fn render_thinking_fold_window_lines(
    lines: &[String],
    marker_lines: usize,
    rewrite_right_margin_cols: usize,
    max_visible_rows: usize,
    out: &mut String,
) -> (usize, Vec<String>) {
    out.clear();
    if lines.is_empty() {
        return (0, Vec::new());
    }

    let reserve_cols = THINKING_FOLD_BODY_INDENT_WIDTH + rewrite_right_margin_cols;
    let marker_lines = marker_lines.min(lines.len());
    let mut wrapped_content_rows = Vec::new();
    for line in lines.iter().skip(marker_lines) {
        wrapped_content_rows.extend(wrap_line_to_terminal_rows_with_reserve(
            &fold_display_line(line),
            reserve_cols,
        ));
    }
    let hidden_wrapped_rows = wrapped_content_rows.len().saturating_sub(max_visible_rows);
    let marker = if hidden_wrapped_rows > 0 {
        // When truncated at a physical line, the first hidden content may come from a still-streaming logical line, so the
        // imprecise "earlier lines" count can no longer be reported.
        Some("… more".to_string())
    } else if marker_lines > 0 {
        Some(lines[0].clone())
    } else {
        None
    };
    let mut rows_to_render = Vec::with_capacity(
        wrapped_content_rows
            .len()
            .saturating_sub(hidden_wrapped_rows)
            .saturating_add(usize::from(marker.is_some())),
    );
    if let Some(marker) = marker {
        // The fold hint must always occupy exactly one physical line; only the body is allowed to wrap.
        rows_to_render.push((
            clamp_line_to_terminal_row_with_reserve(&marker, reserve_cols),
            true,
        ));
    }
    rows_to_render.extend(
        wrapped_content_rows
            .into_iter()
            .skip(hidden_wrapped_rows)
            .map(|line| (line, false)),
    );

    let mut rendered_lines = Vec::with_capacity(rows_to_render.len());
    // Folded body has fixed indentation. The body keeps at most max_visible_rows wrapped physical lines; if more
    // content must be hidden, the single-line fold hint does not count against the body budget. Each wrapped segment
    // occupies exactly one physical line; extra right margin for the xterm.js integrated terminal.
    let mut rows = 0usize;
    let mut first_rendered_row = true;

    for (wrapped_row, is_marker) in rows_to_render {
        if !first_rendered_row {
            out.push_str("\r\n");
        }
        first_rendered_row = false;
        let rendered_line = format!("{THINKING_FOLD_BODY_INDENT}{wrapped_row}");
        rows += 1;
        if is_marker {
            out.push_str(theme::current().accent_muted);
            out.push_str(&rendered_line);
            out.push_str("\x1b[0m");
        } else {
            // Thinking body uses the theme's muted color (same family as the marker) instead of
            // SGR-dim on the terminal default foreground: dim support varies by terminal, and the
            // result was nearly indistinguishable from the answer body text on some setups.
            out.push_str(&theme::current().accent_muted);
            out.push_str(&rendered_line);
            out.push_str(RESET);
        }
        rendered_lines.push(rendered_line);
    }

    (rows, rendered_lines)
}
