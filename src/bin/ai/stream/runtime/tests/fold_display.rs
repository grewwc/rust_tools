use super::*;

#[test]
fn thinking_fold_display_controls_preserve_source_and_row_footprint() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let previous_columns = std::env::var_os("COLUMNS");
    unsafe {
        std::env::set_var("COLUMNS", "32");
    }

    assert_eq!(
        fold_display_line("\t中文\r\x08\x1b[2J\u{85}"),
        "    中文\\r\\u{8}\\u{1b}[2J\\u{85}"
    );
    let mut fold = super::super::super::state::ThinkingFoldState::new();
    fold.active = true;
    fold.max_visible_lines = 20;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    // Keep split control sequences as source text; only the display projection
    // may turn them into inert, countable glyphs.
    append_fold_content(&mut fold, "\t中文\r\nbody\tone\ttwo\tthree\x1b");
    append_fold_content(&mut fold, "[2J\x08");
    let source_line = fold.current_line.clone();
    let source_recent = fold.recent_lines.clone();
    let mut out = Vec::new();
    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    let display = String::from_utf8(out).unwrap();
    assert!(!display.contains('\t'));
    assert!(!display.contains("\x1b[2J"));
    assert_eq!(fold.current_line, source_line);
    assert_eq!(fold.recent_lines, source_recent);
    assert_eq!(fold.total_lines, 1);
    assert_eq!(fold.window_rows, fold.rendered_body_lines.len());
    assert_eq!(thinking_fold_rendered_body_rows(&fold), fold.window_rows);
    for line in &fold.rendered_body_lines {
        assert!(!line.chars().any(char::is_control));
        assert_eq!(live_preview_cursor_rows(line), 1);
    }

    unsafe {
        match previous_columns {
            Some(value) => std::env::set_var("COLUMNS", value),
            None => std::env::remove_var("COLUMNS"),
        }
    }
}

#[test]
fn thinking_fold_redraw_reuses_header_after_empty_body() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = super::super::super::state::ThinkingFoldState::new();
    fold.active = true;
    fold.max_visible_lines = 2;
    let mut header = Vec::new();
    write_fold_header(&mut header, None, &fold).unwrap();
    let mut out = Vec::new();

    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    assert_eq!(out, header, "activation must not erase preceding output");
    assert!(fold.header_drawn);
    assert_eq!(fold.window_rows, 0);

    // With no body the cursor is on the blank row below the header. Clear both
    // rows before reprinting, just as for a one-row body, without moving above
    // the fold. Repeated open markers and skipped blank lines keep this state.
    let mut redraw_prefix = Vec::new();
    erase_fold_body(&mut redraw_prefix, 2).unwrap();
    redraw_prefix.extend_from_slice(&header);
    for content in ["", "\n \n", "first line"] {
        append_fold_content(&mut fold, content);
        out.clear();
        thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
        assert!(
            out.starts_with(&redraw_prefix),
            "redraw must replace the existing header after {content:?}: {out:?}"
        );
        if fold.window_rows == 0 {
            assert_eq!(out, redraw_prefix);
        }
    }
    assert_eq!(fold.window_rows, 1);

    // Once content exists, the cursor already rests on its last row: the
    // normal one-row body must still erase exactly two rows including the header.
    out.clear();
    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    assert!(out.starts_with(&redraw_prefix));

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_empty_body_completion_replaces_header() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let previous_columns = std::env::var_os("COLUMNS");
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = super::super::super::state::ThinkingFoldState::new();
    fold.active = true;
    fold.max_visible_lines = 2;
    let mut out = Vec::new();

    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    assert_eq!(fold.window_rows, 0);
    out.clear();
    finalize_fold_to(&mut out, &mut fold, true).unwrap();

    assert_eq!(
        String::from_utf8(out).unwrap(),
        format!(
            "\r\x1b[1A\r\x1b[2K{}  ✓ thinking · 0 lines\x1b[0m\r\n",
            crate::ai::theme::current().accent_muted,
        )
    );
    assert!(!fold.active);
    assert!(!fold.header_drawn);
    assert_eq!(fold.window_rows, 0);

    unsafe {
        match previous_columns {
            Some(columns) => std::env::set_var("COLUMNS", columns),
            None => std::env::remove_var("COLUMNS"),
        }
    }
}

#[test]
fn subagent_fold_redraw_preserves_body_and_footer() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = super::super::super::state::ThinkingFoldState::new_with_labels(
        "subagent explore",
        "done subagent explore",
        false,
    );
    fold.active = true;
    fold.max_visible_lines = 2;
    append_fold_content(&mut fold, "first line\nsecond\tline");
    let mut out = Vec::new();
    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    let mut header = Vec::new();
    write_fold_header(&mut header, None, &fold).unwrap();
    assert!(out.starts_with(&header));
    assert_eq!(fold.window_rows, 2);

    append_fold_content(&mut fold, "\nthird line");
    out.clear();
    thinking_fold_redraw_to(&mut out, None, &mut fold).unwrap();
    let mut redraw_prefix = Vec::new();
    erase_fold_body(&mut redraw_prefix, 3).unwrap();
    redraw_prefix.extend_from_slice(&header);
    assert!(out.starts_with(&redraw_prefix));
    assert_eq!(fold.window_rows, 3);

    out.clear();
    finalize_fold_to(&mut out, &mut fold, false).unwrap();
    let rendered = String::from_utf8(out).unwrap();
    assert!(rendered.contains("second    line"));
    assert!(!rendered.contains('\t'));
    assert!(rendered.contains("third line"));
    assert!(!rendered.contains("first line"));
    assert!(rendered.ends_with("done subagent explore · 3 lines\x1b[0m\r\n"));
    assert!(!rendered.contains("\x1b[0J"));
    assert!(!fold.active);
    assert!(!fold.header_drawn);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn thinking_fold_header_anchored_once_and_window_rows_track_body_only() {
    // The header is reprinted on every redraw (each redraw erases one extra
    // row above the body and redraws the header there, so a resize reflow that
    // outruns the app's view of the width cannot leave permanent stacked
    // markers); window_rows still counts only body physical rows (excluding
    // the header), so cursor-up erasure targets the visible body area and
    // never drifts when the window scrolls into scrollback.
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
    // header_drawn stays true across redraws (the header is reprinted each redraw).
    assert!(state.render.thinking_fold.header_drawn);
    // 4 completed, 2 visible -> fold marker(1) + visible(2) = body 3 rows, header not counted.
    assert_eq!(state.render.thinking_fold.window_rows, 3);
    assert_eq!(state.render.thinking_fold.rendered_body_lines.len(), 3);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn fold_body_row_budget_disables_clamp_when_viewport_unknown() {
    // Non-tty / pipe / test: viewport height is unknown, so clamping must be disabled (usize::MAX) to
    // preserve the exact previous behavior — no rendered byte changes off a real terminal.
    assert_eq!(fold_body_row_budget_for(None), usize::MAX);
}

#[test]
fn fold_body_row_budget_leaves_headroom_for_header_marker_and_cursor() {
    // On a real viewport the body budget is `rows - reserved` (header + marker + one cursor-headroom row),
    // floored at 1 so even a pathologically short pane still renders a body row.
    assert_eq!(
        fold_body_row_budget_for(Some(50)),
        50 - FOLD_VIEWPORT_RESERVED_ROWS
    );
    // Tall terminals dwarf the default window, so the effective size (and rendered output) is unchanged.
    assert!(fold_body_row_budget_for(Some(50)) >= DEFAULT_THINKING_MAX_VISIBLE_LINES);
    // A viewport too short to hold header+marker+headroom still yields at least one visible body row.
    assert_eq!(fold_body_row_budget_for(Some(2)), 1);
    assert_eq!(fold_body_row_budget_for(Some(1)), 1);
}

#[test]
fn short_viewport_clamps_fold_window_so_header_plus_body_stay_visible() {
    // Regression: with a configured window larger than the viewport can hold, the live window must be clamped
    // to `viewport - reserved` body rows so `header + window_rows` never exceeds the viewport. Otherwise the
    // window top scrolls into scrollback, the relative-cursor erase can no longer reach it, and every redraw
    // leaks a stacked `… more` / `… N earlier lines` marker (the screenshot failure).
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    // Emulate a short pane: viewport of 5 rows -> body budget = 5 - 3 = 2 rows.
    let viewport_rows = 5usize;
    let budget = fold_body_row_budget_for(Some(viewport_rows));
    assert_eq!(budget, 2);

    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    // Configured window (10) is far taller than the short viewport can hold.
    fold.max_visible_lines = 10;
    fold.total_lines = 100;
    for i in 0..10 {
        fold.recent_lines.push_back(format!("line-{i}"));
    }
    fold.current_line = "current".to_string();

    // Apply the same clamp the redraw path applies, then render.
    let configured = fold.max_visible_lines;
    fold.max_visible_lines = configured.min(budget);
    let (window, rows) = render_thinking_fold_window(fold);
    fold.max_visible_lines = configured; // model-facing config must be restored untouched

    // Body rows are bounded by the budget, and the whole window (header + body) fits the viewport with headroom.
    assert!(
        rows <= budget + 1, // budget body rows + at most one fold-marker row
        "clamped window body rows {rows} exceed budget {budget}"
    );
    assert!(
        1 + rows <= viewport_rows,
        "header(1) + body {rows} must stay within viewport {viewport_rows}"
    );
    // The marker is still present (content was folded), just not stacked.
    assert_eq!(
        window.matches("more").count() + window.matches("earlier lines").count(),
        1
    );
    // Clamping is display-only: the ring buffer / configured window are untouched.
    assert_eq!(fold.max_visible_lines, 10);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}
