//! Tests for the `output_format` cluster.

use super::super::*;

#[test]
fn tty_tool_output_fold_window_keeps_latest_visible_lines() {
    // Assert the body/marker exists verbatim; widen COLUMNS so it does not run
    // concurrently with the COLUMNS=12 clamp case and read a leaked narrow width,
    // truncating the output.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = TtyToolOutputFoldState::default();
    fold.total_lines = TOOL_OUTPUT_FOLD_MAX_VISIBLE;
    for idx in 1..=TOOL_OUTPUT_FOLD_MAX_VISIBLE {
        fold.recent_lines.push_back(format!("line-{idx}"));
    }
    fold.current_line = format!("line-{}", TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1);

    let expected_owned = (2..=TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1)
        .map(|idx| format!("line-{idx}"))
        .collect::<Vec<_>>();
    assert_eq!(tty_tool_output_hidden_count(&fold), 1);
    assert_eq!(
        tty_tool_output_visible_lines(&fold),
        expected_owned
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );

    let (window, _) = render_tty_tool_output_fold_window(&fold);
    assert_eq!(window.matches("lines folded").count(), 1);
    // Compare the **exact** body sequence after stripping ANSI and the `  │ ` prefix
    // per line, rather than `contains("line-1")`: visible lines like line-10..line-19
    // all contain "line-1" as a substring, so substring assertions would falsely fail
    // (test fragility exposed after raising MAX_VISIBLE from 8 to 64). The exact
    // sequence simultaneously proves line-1 was folded and the rest kept in order.
    let body_tokens = window
        .lines()
        .map(|line| crate::ai::driver::print::sanitize_for_terminal(line))
        .filter_map(|line| line.rsplit("│ ").next().map(str::to_string))
        .filter(|body| !body.contains("lines folded"))
        .collect::<Vec<_>>();
    assert_eq!(body_tokens, expected_owned);

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn tty_tool_output_fold_window_preserves_mock_qr_output() {
    // Simulate a QR-login command's output: QR codes are typically 30–50 lines and
    // must not be truncated by the generic log-folding strategy.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mock_qr = (0..41)
        .map(|row| format!("mock-qr-{row:02} ██  ██  ██  ██"))
        .collect::<Vec<_>>();
    let mut fold = TtyToolOutputFoldState::default();
    fold.total_lines = mock_qr.len();
    fold.recent_lines.extend(mock_qr.iter().cloned());

    let (window, rows) = render_tty_tool_output_fold_window(&fold);
    assert_eq!(tty_tool_output_hidden_count(&fold), 0);
    assert_eq!(rows, mock_qr.len());
    assert!(!window.contains("lines folded"));
    for row in &mock_qr {
        assert!(window.contains(row), "missing QR row: {row}");
    }

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn terminal_visual_grid_detection_requires_a_block_glyph_grid() {
    // Ordinary command output (e.g. git diff) must not be rendered to the terminal
    // even when it has many lines.
    let git_diff = "diff --git a/file.rs b/file.rs\n@@ -1,3 +1,4 @@\n-old line\n+new line\n";
    assert!(!contains_terminal_visual_grid(git_diff));

    let mock_qr = (0..VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS)
        .map(|row| format!("mock-qr-{row:02} ██  ██  ██  ██"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(contains_terminal_visual_grid(&mock_qr));
}

#[test]
fn tty_tool_output_fold_window_clamps_each_line_to_single_row() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "12");
    }

    let mut fold = TtyToolOutputFoldState::default();
    fold.total_lines = TOOL_OUTPUT_FOLD_MAX_VISIBLE;
    fold.recent_lines
        .push_back("12345678901234567890".to_string());
    for idx in 0..(TOOL_OUTPUT_FOLD_MAX_VISIBLE - 2) {
        fold.recent_lines.push_back(format!("pad-{idx}"));
    }
    fold.recent_lines.push_back("abcdef".to_string());
    fold.current_line = "ghijklmnopqrst".to_string();

    let (window, rows) = render_tty_tool_output_fold_window(&fold);
    let visible_lines = tty_tool_output_visible_lines(&fold);

    // Every rendered line is clamped to a single physical row: the window's physical
    // row count equals 1 fold marker + visible logical lines.
    assert_eq!(rows, 1 + visible_lines.len());
    // Each rendered line (after stripping the `  │ ` prefix and ANSI) does not exceed
    // the terminal width (12), so cursor-up is exact.
    for line in window.lines() {
        let visible = crate::ai::driver::print::sanitize_for_terminal(line);
        assert!(
            unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
            "line exceeds terminal width: {visible:?}"
        );
    }
    assert!(!window.contains("12345678901234567890"));
    assert!(window.contains("abcdef"));
    // Overwide lines are truncated with an ellipsis ending instead of lingering
    // verbatim and undercounting rows for cursor-up.
    assert!(window.contains('…'));

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

#[test]
fn tty_tool_output_fold_redraw_clears_only_its_previous_rows() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    unsafe {
        std::env::set_var("COLUMNS", "200");
    }

    let mut fold = TtyToolOutputFoldState::default();
    fold.total_lines = 1;
    fold.recent_lines.push_back("completed line".to_owned());
    fold.current_line = "initial partial line".to_owned();
    let mut initial = Vec::new();
    fold.redraw_to(&mut initial).unwrap();

    fold.current_line = "updated partial line".to_owned();
    let mut redraw = Vec::new();
    fold.redraw_to(&mut redraw).unwrap();
    let redraw = String::from_utf8(redraw).unwrap();

    // The previous two-row window is erased in place. A screen-to-end erase
    // (`CSI 0J`) would also erase the DECSTBM-reserved side-note composer.
    assert!(
        redraw.starts_with("\r\x1b[2A\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1A\r"),
        "unexpected redraw sequence: {redraw:?}"
    );
    assert!(
        !redraw.contains("\x1b[0J"),
        "unexpected full-screen erase: {redraw:?}"
    );
    assert!(redraw.contains("updated partial line"));

    unsafe {
        std::env::remove_var("COLUMNS");
    }
}
