use super::*;
/// Whether to show a compact "waiting for model output" status hint in the terminal.
/// Applies to all TTY sessions. Written and flushed on its own line so it appears
/// immediately; once the first visible chunk arrives it is erased by moving the cursor back up over
/// the rows the hint occupies (see `erase_waiting_hint`), leaving no extra lines behind.
pub(super) fn should_show_waiting_hint(app: &App) -> bool {
    runtime_ctx::terminal_output_enabled()
        && io::stdout().is_terminal()
        && !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
}

pub(super) fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    // Waiting hint on its own line: erased with a cursor-up plus row clears when the first chunk arrives.
    write_waiting_hint_line(state, "waiting…")?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_tool_call = false;
    Ok(())
}

/// Write the hint as exactly one live-region row and remember its plain text: the hint owns its own line
/// and is erased by moving the cursor back up over it, and a terminal that narrowed re-wraps the row that
/// is already on screen, so the erase recomputes the physical row count from the text actually written.
pub(super) fn write_waiting_hint_line(state: &mut StreamProcessingState, label: &str) -> io::Result<()> {
    let row = clamp_line_to_terminal_row(&format!("  ⠋ {label}"));
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}{row}{RESET}", theme::current().accent_muted)?;
    out.flush()?;
    state.render.waiting_hint_line = row;
    Ok(())
}

/// Erase the hint row(s) currently on screen and park the cursor where the hint started, so the next
/// output prints in its place. Clears `waiting_hint_line` but leaves the hint flags to the caller:
/// `clear_waiting_hint` resets them, while the in-place rewrites (`upgrade_waiting_hint_for_buffering`,
/// `show_deferred_body_buffering_hint`, `refresh_deferred_body_rate_hint`,
/// `refresh_tool_call_rate_hint`) keep the hint active.
pub(super) fn erase_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active || state.render.waiting_hint_line.is_empty() {
        return Ok(());
    }
    let rows = live_preview_cursor_rows(&state.render.waiting_hint_line);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    erase_rows_above_cursor(&mut out, rows)?;
    out.flush()?;
    state.render.waiting_hint_line.clear();
    Ok(())
}

pub(super) fn sanitize_waiting_hint_tool_name(function_name: &str) -> String {
    let sanitized = sanitize_for_terminal(function_name);
    let single_line = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.is_empty() {
        "tool".to_string()
    } else {
        single_line
    }
}

/// Label for the tool-call receiving hint. `rate_text` is the optional live
/// argument-throughput suffix: it stays `None` until a measurable argument window
/// exists, so the row keeps its original wording in the first moments of a call.
pub(super) fn tool_call_hint_label(function_name: &str, rate_text: Option<&str>) -> String {
    let function_name = sanitize_waiting_hint_tool_name(function_name);
    match rate_text {
        Some(rate_text) => format!("receiving `{function_name}` arguments… · {rate_text}"),
        None => format!("receiving `{function_name}` arguments…"),
    }
}

pub(super) fn print_tool_call_waiting_hint(
    state: &mut StreamProcessingState,
    function_name: &str,
) -> io::Result<()> {
    if state.render.waiting_hint_active {
        clear_waiting_hint(state)?;
    }
    write_waiting_hint_line(state, &tool_call_hint_label(function_name, None))?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_tool_call = true;
    Ok(())
}

pub(super) fn configure_thinking_fold(state: &mut StreamProcessingState) {
    state.render.thinking_fold.max_visible_lines = resolve_thinking_fold_max_visible_lines(
        io::stdout().is_terminal(),
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
    state.render.thinking_fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
}

pub(super) fn configure_subagent_preview_fold(
    app: &App,
    state: &mut StreamProcessingState,
    markers: &mut StreamMarkers,
) {
    state.render.subagent_fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    if !io::stdout().is_terminal() || runtime_ctx::current_subagent_depth() == 0 {
        state.render.subagent_fold.max_visible_lines = usize::MAX;
        return;
    }

    state.render.subagent_fold.max_visible_lines = resolve_thinking_fold_max_visible_lines(
        true,
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
    markers.enable_subagent_preview(&app.current_agent);
    if let (Some(header), Some(footer)) = (
        markers.subagent_fold_header.as_deref(),
        markers.subagent_fold_footer.as_deref(),
    ) {
        state.render.subagent_fold.set_labels(header, footer);
    }
}

pub(super) fn resolve_thinking_fold_max_visible_lines(is_tty: bool, raw: Option<&str>) -> usize {
    if !is_tty {
        return usize::MAX;
    }

    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return DEFAULT_THINKING_MAX_VISIBLE_LINES;
    };

    match raw.parse::<usize>() {
        Ok(0) => usize::MAX,
        Ok(lines) => lines,
        Err(_) => DEFAULT_THINKING_MAX_VISIBLE_LINES,
    }
}

pub(super) fn upgrade_waiting_hint_for_buffering(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active
        || state.render.waiting_hint_buffering
        || state.render.waiting_hint_tool_call
    {
        return Ok(());
    }
    // Erase the hint and rewrite it in place so the buffering state stays visible.
    erase_waiting_hint(state)?;
    write_waiting_hint_line(state, "buffering…")?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

/// Show a compact "generating…" hint while the assistant body is being withheld
/// (defer_assistant_body) and thinking is closed. The final answer is streamed but
/// not rendered until the completion/citation gates accept it, so without this the
/// terminal stays blank for the whole generation. Upgrades the initial "waiting…"
/// line in place (cursor up + clear + rewrite) and stays put until
/// `clear_waiting_hint` fires — at the next renderable chunk or at stream end.
pub(super) fn show_deferred_body_buffering_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !io::stdout().is_terminal()
        || state.render.waiting_hint_buffering
        || state.render.waiting_hint_tool_call
    {
        return Ok(());
    }
    if state.render.waiting_hint_active {
        erase_waiting_hint(state)?;
    }
    write_waiting_hint_line(state, "generating…")?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

/// Minimum interval between in-place refreshes of a live rate hint, shared by the
/// deferred-body "generating…" hint and the tool-call "receiving `X` arguments…" hint
/// (only one of them owns the row at a time). The refresh is chunk-driven (no timer in
/// the stream loop), so this bounds terminal repaints to ~2 Hz while tokens keep
/// flowing; a stalled stream keeps the last written rate.
pub(super) const LIVE_RATE_HINT_REFRESH_MS: u64 = 500;

/// Real-time output-throughput text for the deferred-body "generating…" hint.
/// While `defer_assistant_body` withholds the final answer from the terminal until
/// the completion/citation gates accept it, this in-place rewrite is the only
/// place the live rate is visible. Called on each committed output chunk and
/// throttled by `DEFERRED_HINT_RATE_REFRESH_MS`; rows whose text is unchanged are
/// not repainted, so there is no flicker. The `~` prefix marks the estimate as
/// heuristic; the exact server-reported numbers are printed at stream end by
/// `maybe_print_token_throughput_metrics`.
pub(super) fn refresh_deferred_body_rate_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    // Never clobber the tool-call hint, and never draw a fresh row under a line
    // that was already erased (the erase here is a no-op, so a blank line would
    // duplicate the hint below the cursor).
    if state.render.waiting_hint_tool_call || state.render.waiting_hint_line.is_empty() {
        return Ok(());
    }
    let Some(started_at) = state.content.output_started_at else {
        return Ok(());
    };
    let tokens = state.content.live_output_tokens;
    let elapsed = started_at.elapsed();
    if tokens == 0 || elapsed < MIN_RATE_WINDOW {
        return Ok(());
    }
    if state
        .render
        .waiting_hint_rate_refreshed_at
        .is_some_and(|at| at.elapsed() < Duration::from_millis(LIVE_RATE_HINT_REFRESH_MS))
    {
        return Ok(());
    }
    let Some(rate_text) = format_live_rate_text(tokens, Some(elapsed)) else {
        return Ok(());
    };
    let label = format!("generating… · {rate_text}");
    let row = clamp_line_to_terminal_row(&format!("  ⠋ {label}"));
    if row == state.render.waiting_hint_line {
        return Ok(());
    }
    // In-place rewrite with the same geometry as `upgrade_waiting_hint_for_buffering`:
    // the erase draws the cursor back to where the hint started, then the rewrite
    // lands exactly on the same row(s).
    erase_waiting_hint(state)?;
    write_waiting_hint_line(state, &label)?;
    state.render.waiting_hint_rate_refreshed_at = Some(Instant::now());
    Ok(())
}

/// Live argument-throughput text for the tool-call receiving hint. Tool-call arguments
/// are never printed to the terminal (`write_tool_call_arguments_stream` is a no-op), so
/// this in-place rewrite is the only sign that a large payload — apply_patch,
/// execute_command, task, … — is still flowing. Throttled and gated exactly like the
/// deferred-body hint; the `~` prefix marks the count as a text-based estimate.
pub(super) fn refresh_tool_call_rate_hint(
    state: &mut StreamProcessingState,
    function_name: &str,
) -> io::Result<()> {
    // Only the tool-call hint may rewrite this row: never clobber the "waiting…" /
    // "generating…" hints, and never draw a fresh row under one already erased.
    if !state.render.waiting_hint_tool_call || state.render.waiting_hint_line.is_empty() {
        return Ok(());
    }
    let Some(started_at) = state.content.tool_args_started_at else {
        return Ok(());
    };
    let tokens = state.content.live_tool_arg_tokens;
    let elapsed = started_at.elapsed();
    if tokens == 0 || elapsed < MIN_RATE_WINDOW {
        return Ok(());
    }
    if state
        .render
        .waiting_hint_rate_refreshed_at
        .is_some_and(|at| at.elapsed() < Duration::from_millis(LIVE_RATE_HINT_REFRESH_MS))
    {
        return Ok(());
    }
    let Some(rate_text) = format_live_rate_text(tokens, Some(elapsed)) else {
        return Ok(());
    };
    let label = tool_call_hint_label(function_name, Some(&rate_text));
    let row = clamp_line_to_terminal_row(&format!("  ⠋ {label}"));
    if row == state.render.waiting_hint_line {
        return Ok(());
    }
    // In-place rewrite with the same geometry as `refresh_deferred_body_rate_hint`:
    // the erase draws the cursor back to where the hint started, then the rewrite
    // lands exactly on the same row(s).
    erase_waiting_hint(state)?;
    write_waiting_hint_line(state, &label)?;
    state.render.waiting_hint_rate_refreshed_at = Some(Instant::now());
    Ok(())
}

/// Pure formatter for a waiting hint's live rate text (`~N tok @ R tok/s`), shared by the
/// deferred-body "generating…" hint and the tool-call "receiving `X` arguments…" hint.
/// `None` until the first token of that window and until `MIN_RATE_WINDOW` has elapsed,
/// mirroring `format_token_rate`'s "—" gate so the very first chunk is not reported as a
/// misleading burst rate.
pub(super) fn format_live_rate_text(tokens: u64, elapsed: Option<Duration>) -> Option<String> {
    if tokens == 0 {
        return None;
    }
    let rate = format_token_rate(tokens, elapsed);
    if rate == "—" {
        return None;
    }
    Some(format!(
        "~{} tok @ {rate} tok/s",
        format_compact_token_count(tokens),
    ))
}

pub(crate) fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    // Erase the standalone hint row(s) so following content prints in place.
    erase_waiting_hint(state)?;
    state.render.waiting_hint_active = false;
    state.render.waiting_hint_buffering = false;
    state.render.waiting_hint_tool_call = false;
    Ok(())
}
