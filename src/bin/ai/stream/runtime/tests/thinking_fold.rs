use super::*;

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
    assert!(window.is_empty());
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

    let mut fold = super::super::super::state::ThinkingFoldState::new();
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
            "\r\x1b[1A\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1A\r\r\x1b[1A\r\x1b[2K{}  ✓ thinking · 3 lines\x1b[0m\r\n",
            crate::ai::theme::current().accent_muted,
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

/// Models only what the fold's own writer emits: SGR is ignored, and `\r`, `\n`, CSI A/B/K plus
/// DECAWM auto-wrap cover the rest. The writer ends every row with CRLF, so `resize` re-wraps each
/// row on its own, exactly as a reflowing terminal does with hard-wrapped rows.
struct ReflowGrid {
    cols: usize,
    cells: Vec<Vec<char>>,
    row: usize,
    col: usize,
}

impl ReflowGrid {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            cells: vec![vec![' '; cols]; rows],
            row: 0,
            col: 0,
        }
    }

    fn newline(&mut self) {
        assert!(self.row + 1 < self.cells.len(), "test grid ran out of rows");
        self.row += 1;
    }

    fn feed(&mut self, text: &str) {
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '\r' => self.col = 0,
                '\n' => self.newline(),
                '\x1b' => {
                    assert_eq!(chars.next(), Some('['), "only CSI sequences are modelled");
                    let mut args = String::new();
                    let command = loop {
                        let next = chars.next().expect("unterminated CSI sequence");
                        if ('@'..='~').contains(&next) {
                            break next;
                        }
                        args.push(next);
                    };
                    match command {
                        'm' => {}
                        'K' => {
                            assert_eq!(args, "2");
                            self.cells[self.row].fill(' ');
                        }
                        'A' => {
                            let rows = args.parse::<usize>().unwrap_or(1).max(1);
                            self.row = self.row.saturating_sub(rows);
                        }
                        'B' => {
                            let rows = args.parse::<usize>().unwrap_or(1).max(1);
                            self.row = (self.row + rows).min(self.cells.len() - 1);
                        }
                        other => panic!("unsupported CSI {args}{other}"),
                    }
                }
                _ => {
                    if self.col >= self.cols {
                        self.col = 0;
                        self.newline();
                    }
                    self.cells[self.row][self.col] = ch;
                    self.col += 1;
                }
            }
        }
    }

    /// Terminal reflow: every row on screen is re-wrapped at `cols`, rows keep their order, and the
    /// cursor stays at the end of the row it was on.
    fn resize(&mut self, cols: usize) {
        let cursor_row = self.row;
        // A terminal reflows the rows it has drawn; the blank tail below stays blank.
        let drawn_through = self
            .cells
            .iter()
            .rposition(|row| row.iter().any(|cell| *cell != ' '))
            .map_or(0, |last| last.max(cursor_row));
        let mut reflowed: Vec<Vec<char>> = Vec::new();
        for (index, row) in self.cells.iter().take(drawn_through + 1).enumerate() {
            let mut content = row.clone();
            while content.last() == Some(&' ') {
                content.pop();
            }
            let mut fragments: Vec<Vec<char>> = content.chunks(cols).map(<[char]>::to_vec).collect();
            if fragments.is_empty() {
                fragments.push(Vec::new());
            }
            if index == cursor_row {
                self.row = reflowed.len() + fragments.len() - 1;
                let remainder = content.len() % cols;
                self.col = if remainder == 0 && !content.is_empty() {
                    cols
                } else {
                    remainder
                };
            }
            // Every row of a terminal's screen is `cols` cells wide, blanks included.
            for fragment in fragments.iter_mut() {
                fragment.resize(cols, ' ');
            }
            reflowed.extend(fragments);
        }
        assert!(reflowed.len() <= self.cells.len(), "test grid ran out of rows");
        while reflowed.len() < self.cells.len() {
            reflowed.push(vec![' '; cols]);
        }
        self.cells = reflowed;
        self.cols = cols;
    }

    fn text(&self) -> String {
        self.cells
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }
}

/// A terminal that reflows the rows it has already drawn when it narrows and reports the new width only
/// afterwards — the VS Code/xterm.js ordering that strands fold rows. Once the new width is known, the
/// fold must erase the whole reflowed region on every frame, stranding nothing, and must not eat the
/// transcript row above it.
#[test]
fn thinking_fold_reclaims_rows_stranded_by_a_resize_reported_after_the_reflow() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);

    unsafe {
        std::env::set_var("COLUMNS", "140");
    }
    let mut grid = ReflowGrid::new(140, 64);
    grid.feed("transcript\r\n");
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    // One long reasoning line, so a narrowed terminal re-wraps the frame rows rather than leaving them.
    append_fold_content(fold, &"reasoning ".repeat(20));
    for rate in ["~1000 tok @ 100 tok/s", "~1001 tok @ 101 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    assert_eq!(grid.text().matches("○ thinking").count(), 1, "steady state");

    // The panel narrows: the terminal reflows what it has drawn, and the new winsize reaches this
    // process only after the next frame has already been written at the old width.
    grid.resize(60);
    // The frame queries the emulator before measuring: it sees the reflowed width while ioctl still
    // reports the old one, so its erase must cover the region outright.
    crate::ai::stream::side_note_input::set_scripted_true_width(Some(60));
    let mut bytes = Vec::new();
    thinking_fold_redraw_to(&mut bytes, Some("~1002 tok @ 102 tok/s"), fold).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());
    let stranded = grid.text();
    assert_eq!(
        stranded.matches("○ thinking").count(),
        1,
        "the lag frame must erase against the reflowed width and leave no stranded header:\n{stranded}"
    );

    unsafe {
        std::env::set_var("COLUMNS", "60");
    }
    let mut bytes = Vec::new();
    thinking_fold_redraw_to(&mut bytes, Some("~1003 tok @ 103 tok/s"), fold).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());

    let screen = grid.text();
    assert_eq!(
        screen.matches("○ thinking").count(),
        1,
        "rows stranded by the resize stayed on screen:\n{screen}"
    );
    assert_eq!(
        screen.matches("~1002").count(),
        0,
        "the header written inside the resize gap was never reclaimed:\n{screen}"
    );
    assert_eq!(
        screen.matches("transcript").count(),
        1,
        "reclaiming the stranded rows ate the transcript row above the fold:\n{screen}"
    );
    // Reset the thread-local scripted width: it outlives this test on its worker
    // thread, and a later width-sensitive test on the same thread would otherwise
    // inherit 60 columns and truncate its assertions.
    crate::ai::stream::side_note_input::set_scripted_true_width(None);
}

/// L=0 timing: the reflow and the winsize update both land between frames, so no frame ever
/// renders against a stale width. The next frame already knows the new width, its own span
/// covers the whole region, and without any stale-width correction nothing may reach above it.
#[test]
fn thinking_fold_resize_reported_between_frames_does_not_erase_the_transcript() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);

    unsafe {
        std::env::set_var("COLUMNS", "140");
    }
    let mut grid = ReflowGrid::new(140, 64);
    grid.feed("transcript\r\n");
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    append_fold_content(fold, &"reasoning ".repeat(20));
    for rate in ["~1000 tok @ 100 tok/s", "~1001 tok @ 101 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    assert_eq!(grid.text().matches("○ thinking").count(), 1, "steady state");

    grid.resize(60);
    unsafe {
        std::env::set_var("COLUMNS", "60");
    }
    let mut bytes = Vec::new();
    thinking_fold_redraw_to(&mut bytes, Some("~1002 tok @ 102 tok/s"), fold).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());

    let screen = grid.text();
    assert_eq!(
        screen.matches("transcript").count(),
        1,
        "the no-lag resize path erased the transcript row above the fold:\n{screen}"
    );
    assert_eq!(
        screen.matches("○ thinking").count(),
        1,
        "a lag-free frame should keep exactly one header:\n{screen}"
    );
}

/// L>=2 timing: the winsize update arrives only after TWO frames have already rendered against
/// the stale width. Each frame queries the emulator's reflowed width first, so neither may strand a row.
#[test]
fn thinking_fold_reclaims_every_header_stranded_by_multiple_lag_frames() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);

    unsafe {
        std::env::set_var("COLUMNS", "140");
    }
    let mut grid = ReflowGrid::new(140, 64);
    grid.feed("transcript\r\n");
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    append_fold_content(fold, &"reasoning ".repeat(20));
    for rate in ["~1000 tok @ 100 tok/s", "~1001 tok @ 101 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    assert_eq!(grid.text().matches("○ thinking").count(), 1, "steady state");

    // Reflow first; the winsize update lands only after a second stale frame, so two frames
    // render against the old width and each of them under-erases.
    grid.resize(60);
    crate::ai::stream::side_note_input::set_scripted_true_width(Some(60));
    for rate in ["~1002 tok @ 102 tok/s", "~1003 tok @ 103 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    unsafe {
        std::env::set_var("COLUMNS", "60");
    }
    let mut bytes = Vec::new();
    thinking_fold_redraw_to(&mut bytes, Some("~1004 tok @ 104 tok/s"), fold).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());

    let screen = grid.text();
    assert_eq!(
        screen.matches("○ thinking").count(),
        1,
        "headers stranded by two lag frames survived the recovery frame:\n{screen}"
    );
    assert_eq!(
        screen.matches("transcript").count(),
        1,
        "reclaiming the stranded rows ate the transcript row above the fold:\n{screen}"
    );
    // Same thread-local cleanup as above: do not leak the scripted 60-column width
    // into width-sensitive tests that later reuse this worker thread.
    crate::ai::stream::side_note_input::set_scripted_true_width(None);
}

/// A mid-stream warning printed while the fold is live must not strand the header: parking
/// the frame before the intruder line lets the next redraw draw one fresh header below it
/// instead of stacking a stranded `○ thinking` per frame.
#[test]
fn thinking_fold_survives_mid_stream_warning_without_stacking_headers() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);

    unsafe {
        std::env::set_var("COLUMNS", "100");
    }
    let mut grid = ReflowGrid::new(100, 30);
    grid.feed("transcript\r\n");
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    append_fold_content(fold, &"reasoning ".repeat(20));
    for rate in ["~1.8k tok @ 147 tok/s", "~1.8k tok @ 146 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    assert_eq!(grid.text().matches("○ thinking").count(), 1, "steady state");

    // The warning path: park the live frame, then print the intruder line below it.
    let mut bytes = Vec::new();
    suspend_fold_frame(&mut bytes, fold).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());
    grid.feed("  ⚠ warning\r\n");

    append_fold_content(fold, &"more reasoning ".repeat(20));
    for rate in ["~1.9k tok @ 147 tok/s", "~3.8k tok @ 142 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }

    let screen = grid.text();
    assert_eq!(
        screen.matches("○ thinking").count(),
        1,
        "the header stranded by the warning survived later redraws:\n{screen}"
    );
    assert_eq!(
        screen.matches("transcript").count(),
        1,
        "parking the frame ate the transcript row above the fold:\n{screen}"
    );
    assert!(
        screen.contains("⚠ warning"),
        "the intruder line did not survive the resume:\n{screen}"
    );
}

/// A stream-stopping warning closes the live fold first, so the completion header replaces
/// the frame in place and the diagnostic prints below it: no stranded `○ thinking` rows.
#[test]
fn thinking_fold_finalized_before_stop_warning_leaves_no_live_header() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _columns = SavedColumns(std::env::var_os("COLUMNS"));
    crate::ai::stream::side_note_input::set_scripted_true_width(None);

    unsafe {
        std::env::set_var("COLUMNS", "100");
    }
    let mut grid = ReflowGrid::new(100, 30);
    grid.feed("transcript\r\n");
    let mut state = StreamProcessingState::new();
    let fold = &mut state.render.thinking_fold;
    fold.active = true;
    fold.max_visible_lines = 2;
    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    append_fold_content(fold, &"reasoning ".repeat(20));
    for rate in ["~1.8k tok @ 147 tok/s", "~1.8k tok @ 146 tok/s"] {
        let mut bytes = Vec::new();
        thinking_fold_redraw_to(&mut bytes, Some(rate), fold).unwrap();
        grid.feed(&String::from_utf8(bytes).unwrap());
    }
    assert_eq!(grid.text().matches("○ thinking").count(), 1, "steady state");

    // The stop path: close the fold (as `finalize_live_folds_before_diagnostic` does),
    // then print the warning below the completion header.
    let mut bytes = Vec::new();
    finalize_fold_to(&mut bytes, fold, true).unwrap();
    grid.feed(&String::from_utf8(bytes).unwrap());
    grid.feed("  ⚠ 检测到模型推理重复循环，停止当前响应并自动重试…\r\n");

    let screen = grid.text();
    assert_eq!(
        screen.matches("○ thinking").count(),
        0,
        "finalizing the fold left a live header above the warning:\n{screen}"
    );
    assert_eq!(
        screen.matches("✓ thinking").count(),
        1,
        "the completion header did not replace the frame:\n{screen}"
    );
    assert_eq!(
        screen.matches("transcript").count(),
        1,
        "closing the fold ate the transcript row above it:\n{screen}"
    );
    assert!(
        screen.contains("重复循环"),
        "the warning line did not survive below the completion header:\n{screen}"
    );
}

#[test]
fn thinking_fold_terminal_grid_refresh_does_not_accumulate_headers() {
    // Accept only the fold/footer control sequences, including scroll margins,
    // saved cursors and IND/RI. Scrollback counts towards the orphan check.
    struct Grid {
        cols: usize,
        cells: Vec<Vec<char>>,
        history: Vec<Vec<char>>,
        row: usize,
        col: usize,
        bottom: usize,
        saved: (usize, usize),
    }
    impl Grid {
        fn newline(&mut self) {
            if self.row == self.bottom {
                self.history.push(self.cells.remove(0));
                self.cells.insert(self.bottom, vec![' '; self.cols]);
            } else {
                self.row = (self.row + 1).min(self.cells.len() - 1);
            }
        }
        fn feed(&mut self, text: &str) {
            let mut chars = text.chars();
            while let Some(ch) = chars.next() {
                match ch {
                    '\r' => self.col = 0,
                    '\n' => self.newline(),
                    '\t' => self.col = ((self.col / 8 + 1) * 8).min(self.cols - 1),
                    '\x1b' => {
                        match chars.next().expect("complete escape") {
                            '7' => {
                                self.saved = (self.row, self.col.min(self.cols - 1));
                                continue;
                            }
                            '8' => {
                                (self.row, self.col) = self.saved;
                                continue;
                            }
                            'D' => {
                                self.col = self.col.min(self.cols - 1);
                                self.newline();
                                continue;
                            }
                            'M' => {
                                assert!(self.row > 0, "footer RI must follow IND");
                                self.row -= 1;
                                self.col = self.col.min(self.cols - 1);
                                continue;
                            }
                            '[' => {}
                            c => panic!("unsupported escape {c}"),
                        }
                        let mut args = String::new();
                        let command = loop {
                            let c = chars.next().expect("complete CSI");
                            if ('@'..='~').contains(&c) {
                                break c;
                            }
                            args.push(c);
                        };
                        match command {
                            'm' => {}
                            'h' | 'l' => assert_eq!(args, "?25"),
                            'H' => {
                                let (row, col) = args.split_once(';').expect("CUP coordinates");
                                self.row = row.parse::<usize>().unwrap() - 1;
                                self.col = col.parse::<usize>().unwrap() - 1;
                            }
                            'r' => {
                                self.bottom = if args.is_empty() {
                                    self.cells.len() - 1
                                } else {
                                    let (top, bottom) = args.split_once(';').unwrap();
                                    assert_eq!(top, "1");
                                    bottom.parse::<usize>().unwrap() - 1
                                };
                                assert!(self.bottom > 0);
                                self.row = 0;
                                self.col = 0;
                            }
                            'A' => {
                                let n = args.parse::<usize>().unwrap_or(1).max(1);
                                self.row = self.row.saturating_sub(n);
                                self.col = self.col.min(self.cols - 1);
                            }
                            'B' => {
                                let n = args.parse::<usize>().unwrap_or(1).max(1);
                                self.row = (self.row + n).min(self.cells.len() - 1);
                                self.col = self.col.min(self.cols - 1);
                            }
                            'K' => {
                                assert_eq!(args, "2");
                                self.cells[self.row].fill(' ');
                                self.col = self.col.min(self.cols - 1);
                            }
                            _ => panic!("unsupported CSI {args}{command}"),
                        }
                    }
                    _ => {
                        // Test fixtures use only single-cell glyphs; fail rather
                        // than silently pretending to support other Unicode.
                        assert!(ch.is_ascii_graphic() || matches!(ch, ' ' | '○' | '✓' | '·' | '…' | '▌'));
                        if self.col == self.cols {
                            self.col = 0;
                            self.newline();
                        }
                        self.cells[self.row][self.col] = ch;
                        self.col += 1;
                    }
                }
            }
        }
        fn text(&self) -> String {
            self.history
                .iter()
                .chain(&self.cells)
                .map(|row| row.iter().collect::<String>().trim_end().to_string())
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // Restore dimensions even when a frame assertion fails, while ENV_LOCK is held.
    struct SavedDimensions([(&'static str, Option<std::ffi::OsString>); 2]);
    impl Drop for SavedDimensions {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }
    let _dimensions = SavedDimensions([
        ("COLUMNS", std::env::var_os("COLUMNS")),
        ("LINES", std::env::var_os("LINES")),
    ]);
    let rates = [
        "~973 tok @ 192 tok/s",
        "~975 tok @ 193 tok/s",
        "~976 tok @ 193 tok/s",
        "~978 tok @ 193 tok/s",
        "~980 tok @ 192 tok/s",
        "~981 tok @ 192 tok/s",
        "~982 tok @ 192 tok/s",
        "~984 tok @ 193 tok/s",
        "~986 tok @ 191 tok/s",
    ];
    let header_cols = format!("  ○ thinking · {}", rates[0]).chars().count();
    let mut failures = Vec::new();
    for cols in [200, header_cols, header_cols - 1, 32] {
        unsafe {
            std::env::set_var("COLUMNS", cols.to_string());
        }
        for start_row in [0, 5, 10] {
            for body in ["", "body", "body\tone\ttwo\tthree\tfour\tfive"] {
              for with_footer in [false, true] {
                let mut grid = Grid {
                    cols,
                    cells: vec![vec![' '; cols]; 12],
                    history: Vec::new(),
                    row: start_row,
                    col: 0,
                    bottom: 11,
                    saved: (0, 0),
                };
                let mut footer = super::super::super::side_note_input::FooterReservation::for_test(cols as u16, 12);
                let draft: Vec<char> = "keep-draft".chars().collect();
                grid.feed("transcript\r\n");
                let mut fold = super::super::super::state::ThinkingFoldState::new();
                fold.active = true;
                fold.max_visible_lines = 2;
                fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
                let mut counts = Vec::new();
                for (frame, rate) in rates.iter().enumerate() {
                    if frame == 2 {
                        append_fold_content(&mut fold, body);
                    }
                    let mut bytes = Vec::new();
                    thinking_fold_redraw_to(&mut bytes, Some(rate), &mut fold).unwrap();
                    grid.feed(&String::from_utf8(bytes).unwrap());
                    if with_footer && frame >= 3 {
                        let cursor = (grid.row, grid.col);
                        let mut bytes = Vec::new();
                        if frame == 3 {
                            footer.apply_reservation_to(&mut bytes).unwrap();
                        }
                        footer.draw_to(&mut bytes, &draft).unwrap();
                        grid.feed(&String::from_utf8(bytes).unwrap());
                        let expected = (cursor.0.min(10), cursor.1);
                        if (grid.row, grid.col) != expected {
                            failures.push(format!("footer moved cursor: {cursor:?} -> {:?}, expected {expected:?}", (grid.row, grid.col)));
                        }
                        assert!(grid.cells[11].iter().collect::<String>().contains("keep-draft"));
                    }
                    let screen = grid.text();
                    assert_eq!(screen.matches("transcript").count(), 1);
                    if !body.is_empty() && frame >= 2 {
                        assert_eq!(screen.matches("body").count(), 1);
                    }
                    counts.push(screen.matches("○ thinking").count());
                }
                let mut bytes = Vec::new();
                finalize_fold_to(&mut bytes, &mut fold, true).unwrap();
                grid.feed(&String::from_utf8(bytes).unwrap());
                let remaining = grid.text().matches("○ thinking").count();
                assert_eq!(grid.text().matches("✓ thinking").count(), 1);
                if with_footer {
                    assert!(grid.cells[11].iter().collect::<String>().contains("keep-draft"));
                    let cursor = (grid.row, grid.col);
                    let mut bytes = Vec::new();
                    footer.leave_to(&mut bytes).unwrap();
                    grid.feed(&String::from_utf8(bytes).unwrap());
                    assert_eq!((grid.row, grid.col), cursor);
                    assert!(!grid.text().contains("keep-draft"));
                    assert_eq!(grid.bottom, 11);
                }
                grid.feed("answer\r\n");
                assert_eq!(grid.text().matches("answer").count(), 1);
                if counts.iter().any(|&n| n != 1) || remaining != 0 {
                    failures.push(format!(
                        "cols={cols} start={start_row} footer={with_footer} body={body:?}: live={counts:?}, stale={remaining}"
                    ));
                }
              }
            }
        }
    }
    assert!(failures.is_empty(), "orphan headers: {}", failures.join(", "));

    // Fixed viewport: no resize and no unrelated writes inside the live fold.
    // Run with non-TTY stdout so COLUMNS/LINES, rather than an inherited ioctl
    // size, drive the production wrapping and viewport-budget functions.
    const ROWS: usize = 15;
    const COLS: usize = 140;
    unsafe {
        std::env::set_var("COLUMNS", COLS.to_string());
        std::env::set_var("LINES", ROWS.to_string());
    }
    assert_eq!(raw_terminal_rows(), ROWS, "run this fixture with piped stdout");
    assert_eq!(
        clamp_line_to_terminal_row_with_reserve(&"x".repeat(COLS * 2), 2)
            .chars()
            .count(),
        COLS - 2,
        "run this fixture with piped stdout"
    );

    // A timeout warning is ordinary output, not part of the fold footprint.
    // Model both orderings with real renderer bytes: the old warning-first
    // order must reproduce an orphan, while finalize-first preserves output.
    for warning_first in [true, false] {
        for body in ["", "short reasoning\nsecond line"] {
            for start_row in [0, ROWS - 2] {
                let mut grid = Grid {
                    cols: COLS,
                    cells: vec![vec![' '; COLS]; ROWS],
                    history: Vec::new(),
                    row: start_row,
                    col: 0,
                    bottom: ROWS - 1,
                    saved: (0, 0),
                };
                grid.feed("transcript-before-timeout\r\n");
                let mut fold = super::super::super::state::ThinkingFoldState::new();
                fold.active = true;
                fold.max_visible_lines = 2;
                fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
                append_fold_content(&mut fold, body);
                let mut bytes = Vec::new();
                thinking_fold_redraw_to(&mut bytes, Some(rates[0]), &mut fold).unwrap();
                grid.feed(&String::from_utf8(bytes).unwrap());
                if warning_first {
                    grid.feed("timeout-warning\r\n");
                }
                let mut bytes = Vec::new();
                finalize_fold_to(&mut bytes, &mut fold, true).unwrap();
                grid.feed(&String::from_utf8(bytes).unwrap());
                if !warning_first {
                    grid.feed("timeout-warning\r\n");
                }
                let screen = grid.text();
                assert_eq!(
                    screen.matches("○ thinking").count(),
                    usize::from(warning_first),
                    "warning_first={warning_first} body={body:?} start={start_row}\n{screen}"
                );
                assert_eq!(screen.matches("transcript-before-timeout").count(), 1);
                assert_eq!(screen.matches("✓ thinking").count(), 1);
                if !warning_first {
                    assert_eq!(screen.matches("timeout-warning").count(), 1);
                }
            }
        }
    }

    let mut continuous = vec![String::new(), "ascii-start ".to_string()];
    let mut multiline = vec![String::new(), "first line\n".to_string()];
    for frame in 0..24 {
        let chunk = format!("chunk-{frame:02} {}", "abcdefghij ".repeat(13));
        continuous.push(chunk.clone());
        multiline.push(format!("{chunk}{}", if frame % 3 == 2 { "\n" } else { "" }));
    }
    let mut trailing_newlines = multiline.clone();
    // Replace the long tail with short logical lines, then empty lines and an
    // idle frame: exercise both expanding and shrinking the physical footprint.
    trailing_newlines.extend(
        ["short-a\n", "short-b\n", "short-c\n", "\n", "\n", ""]
            .map(str::to_string),
    );
    for (scenario, chunks) in [
        ("continuous-ascii", continuous),
        ("multiple-logical-lines", multiline),
        ("trailing-newlines", trailing_newlines),
    ] {
        let chunk_lengths: Vec<_> = chunks.iter().map(String::len).collect();
        for start_row in [0, 1, 6, 10, 13, 14] {
            for max_visible_lines in [1, 2, 3] {
                // Cover disabled, already active before the first header, and
                // entry while a long body is being refreshed.
                for footer_at in [None, Some(0), Some(6)] {
                    let case = format!(
                        "15x140 scenario={scenario} start_row={start_row} max_visible_lines={max_visible_lines} footer_at={footer_at:?}"
                    );
                    let mut grid = Grid {
                        cols: COLS,
                        cells: vec![vec![' '; COLS]; ROWS],
                        history: Vec::new(),
                        row: start_row,
                        col: 0,
                        bottom: ROWS - 1,
                        saved: (0, 0),
                    };
                    grid.feed("transcript-before-fold\r\n");
                    let mut footer = super::super::super::side_note_input::FooterReservation::for_test(
                        COLS as u16,
                        ROWS as u16,
                    );
                    let draft: Vec<char> = "keep-draft".chars().collect();
                    let mut fold = super::super::super::state::ThinkingFoldState::new();
                    fold.active = true;
                    fold.max_visible_lines = max_visible_lines;
                    fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
                    let mut saw_more = false;

                    for (frame, chunk) in chunks.iter().enumerate() {
                        let check = |grid: &Grid, phase: &str, ansi: &str, live_headers| {
                            let screen = grid.text();
                            assert_eq!(
                                (
                                    screen.matches("○ thinking").count(),
                                    screen.matches("transcript-before-fold").count(),
                                ),
                                (live_headers, 1),
                                "{case} frame={frame} phase={phase} chunk_lengths={:?} chunk={chunk:?} cursor=({}, {}) ANSI={ansi:?}\n{screen}",
                                &chunk_lengths[..=frame], grid.row, grid.col,
                            );
                        };
                        let footer_active = footer_at.is_some_and(|at| frame >= at);
                        if footer_at == Some(frame) {
                            let cursor = (grid.row, grid.col);
                            let mut bytes = Vec::new();
                            footer.apply_reservation_to(&mut bytes).unwrap();
                            footer.draw_to(&mut bytes, &draft).unwrap();
                            let ansi = String::from_utf8(bytes).unwrap();
                            grid.feed(&ansi);
                            check(&grid, "footer-enter", &ansi, usize::from(frame > 0));
                            assert_eq!(
                                (grid.row, grid.col),
                                (cursor.0.min(ROWS - 2), cursor.1),
                                "{case} frame={frame} footer-enter ANSI={ansi:?}"
                            );
                        }
                        append_fold_content(&mut fold, chunk);
                        let mut bytes = Vec::new();
                        thinking_fold_redraw_to(
                            &mut bytes,
                            Some(rates[frame % rates.len()]),
                            &mut fold,
                        )
                        .unwrap();
                        let ansi = String::from_utf8(bytes).unwrap();
                        grid.feed(&ansi);
                        check(&grid, "redraw", &ansi, 1);
                        saw_more |= ansi.contains("… more");
                        if footer_active {
                            // Check before repainting; a repaint must not hide
                            // accidental footer erasure by the fold renderer.
                            assert!(
                                grid.cells[ROWS - 1].iter().collect::<String>().contains("keep-draft"),
                                "{case} frame={frame} redraw erased footer ANSI={ansi:?}\n{}", grid.text()
                            );
                            let cursor = (grid.row, grid.col);
                            let mut bytes = Vec::new();
                            footer.draw_to(&mut bytes, &draft).unwrap();
                            let ansi = String::from_utf8(bytes).unwrap();
                            grid.feed(&ansi);
                            check(&grid, "footer-redraw", &ansi, 1);
                            assert_eq!((grid.row, grid.col), cursor, "{case} frame={frame} footer-redraw ANSI={ansi:?}");
                        }
                    }
                    assert!(saw_more, "{case}: long ASCII must exercise the physical-row … more marker");
                    let mut bytes = Vec::new();
                    finalize_fold_to(&mut bytes, &mut fold, true).unwrap();
                    let ansi = String::from_utf8(bytes).unwrap();
                    grid.feed(&ansi);
                    let screen = grid.text();
                    assert_eq!(
                        (
                            screen.matches("○ thinking").count(),
                            screen.matches("✓ thinking").count(),
                            screen.matches("transcript-before-fold").count(),
                        ),
                        (0, 1, 1),
                        "{case} finalize chunk_lengths={chunk_lengths:?} ANSI={ansi:?}\n{screen}"
                    );
                    if footer_at.is_some() {
                        assert!(grid.cells[ROWS - 1].iter().collect::<String>().contains("keep-draft"), "{case} finalize erased footer ANSI={ansi:?}");
                        let cursor = (grid.row, grid.col);
                        let mut bytes = Vec::new();
                        footer.leave_to(&mut bytes).unwrap();
                        let ansi = String::from_utf8(bytes).unwrap();
                        grid.feed(&ansi);
                        assert_eq!((grid.row, grid.col), cursor, "{case} footer-leave ANSI={ansi:?}");
                        assert_eq!(grid.bottom, ROWS - 1, "{case}");
                        assert!(!grid.text().contains("keep-draft"), "{case}");
                    }
                    grid.feed("answer-after-fold\r\n");
                    let screen = grid.text();
                    assert_eq!(screen.matches("○ thinking").count(), 0, "{case}\n{screen}");
                    assert_eq!(screen.matches("✓ thinking").count(), 1, "{case}\n{screen}");
                    assert_eq!(screen.matches("transcript-before-fold").count(), 1, "{case}\n{screen}");
                    assert_eq!(screen.matches("answer-after-fold").count(), 1, "{case}\n{screen}");
                }
            }
        }
    }
}
