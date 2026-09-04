//! Terminal output presentation: visual-grid detection, TTY fold
//! window state, and tool output clamping.

use super::*;

pub(in crate::ai::driver::turn_runtime) const TOOL_OUTPUT_FOLD_MAX_VISIBLE: usize = 64;
// Regular command logs should not appear in the terminal; non-PTY streamed output is
// shown only when it forms a continuous block-glyph grid. This cap covers common terminal
// QR codes while keeping long ordinary logs from growing the probe buffer without bound.
pub(in crate::ai::driver::turn_runtime) const VISUAL_OUTPUT_PROBE_MAX_BYTES: usize = 16 * 1024;
pub(in crate::ai::driver::turn_runtime) const VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS: usize = 3;
pub(in crate::ai::driver::turn_runtime) const VISUAL_OUTPUT_MIN_BLOCK_GLYPHS_PER_ROW: usize = 8;

/// Decide whether a line looks like terminal visual output drawn with Unicode block
/// glyphs (e.g. a QR code). No command-name allowlist, so no CLI's behavior is hardcoded
/// into the generic executor.
pub(in crate::ai::driver::turn_runtime) fn is_terminal_visual_grid_line(line: &str) -> bool {
    line.chars()
        .filter(|ch| {
            matches!(
                ch,
                '█' | '▀' | '▄' | '▌' | '▐' | '▖' | '▗' | '▘' | '▝' | '▚' | '▞' | '■'
            )
        })
        .count()
        >= VISUAL_OUTPUT_MIN_BLOCK_GLYPHS_PER_ROW
}

/// Only at least three consecutive block-glyph grid rows count as visual output, so
/// progress bars or plain text cannot trigger a false positive.
pub(in crate::ai::driver::turn_runtime) fn contains_terminal_visual_grid(text: &str) -> bool {
    let mut consecutive_rows = 0;
    for line in text.lines() {
        if is_terminal_visual_grid_line(line) {
            consecutive_rows += 1;
            if consecutive_rows >= VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS {
                return true;
            }
        } else {
            consecutive_rows = 0;
        }
    }
    false
}

pub(in crate::ai::driver::turn_runtime) fn trim_visual_output_probe(probe: &mut String) {
    if probe.len() <= VISUAL_OUTPUT_PROBE_MAX_BYTES {
        return;
    }

    let excess = probe.len() - VISUAL_OUTPUT_PROBE_MAX_BYTES;
    let trim_at = probe
        .char_indices()
        .find_map(|(offset, _)| (offset >= excess).then_some(offset))
        .unwrap_or(probe.len());
    probe.drain(..trim_at);
}

#[derive(Debug, Default)]
pub(in crate::ai::driver::turn_runtime) struct TtyToolOutputFoldState {
    pub(super) recent_lines: VecDeque<String>,
    pub(super) current_line: String,
    pub(super) total_lines: usize,
    window_rows: usize,
}

impl TtyToolOutputFoldState {
    pub(super) fn reset(&mut self) {
        self.recent_lines.clear();
        self.current_line.clear();
        self.total_lines = 0;
        self.window_rows = 0;
    }

    pub(super) fn push_text(&mut self, text: &str) -> std::io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        for ch in text.chars() {
            if ch == '\n' {
                self.total_lines += 1;
                self.recent_lines
                    .push_back(std::mem::take(&mut self.current_line));
                while self.recent_lines.len() > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    self.recent_lines.pop_front();
                }
            } else {
                self.current_line.push(ch);
            }
        }
        self.redraw()
    }

    pub(super) fn finish(&mut self) -> std::io::Result<()> {
        self.redraw()
    }

    fn redraw(&mut self) -> std::io::Result<()> {
        let mut out = std::io::stdout();
        if self.window_rows > 0 {
            write!(out, "\x1b[{}A\r\x1b[0J", self.window_rows)?;
        }

        let (window, window_rows) = render_tty_tool_output_fold_window(self);
        if !window.is_empty() {
            out.write_all(window.as_bytes())?;
            out.flush()?;
        }
        self.window_rows = window_rows;
        Ok(())
    }
}

pub(in crate::ai::driver::turn_runtime) fn tty_tool_output_hidden_count(
    fold: &TtyToolOutputFoldState,
) -> usize {
    let current_line = usize::from(!fold.current_line.is_empty());
    fold.total_lines
        .saturating_add(current_line)
        .saturating_sub(TOOL_OUTPUT_FOLD_MAX_VISIBLE)
}

pub(in crate::ai::driver::turn_runtime) fn tty_tool_output_visible_lines(
    fold: &TtyToolOutputFoldState,
) -> Vec<&str> {
    let current_line = usize::from(!fold.current_line.is_empty());
    let visible_completed = TOOL_OUTPUT_FOLD_MAX_VISIBLE.saturating_sub(current_line);
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

pub(in crate::ai::driver::turn_runtime) fn render_tty_tool_output_fold_window(
    fold: &TtyToolOutputFoldState,
) -> (String, usize) {
    let hidden_count = tty_tool_output_hidden_count(fold);
    let visible_lines = tty_tool_output_visible_lines(fold);
    if hidden_count == 0 && visible_lines.is_empty() {
        return (String::new(), 0);
    }

    let mut out = String::new();
    // Every line is clamped to “at most one physical row”, so the window's physical row
    // count always equals its logical line count and cursor-up erasure is exact; auto-wrapped
    // overlong/wide lines no longer leave residue from an undercounted erase.
    let mut rows = 0usize;

    if hidden_count > 0 {
        let marker = format!(
            "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}{}{RESET}",
            clamp_tool_output_body(&format!("··· {hidden_count} lines folded ···"))
        );
        rows += 1;
        out.push_str(&marker);
        out.push('\n');
    }

    for line in visible_lines {
        let rendered = format_tool_output_line(&clamp_tool_output_body(line));
        rows += 1;
        out.push_str(&rendered);
        out.push('\n');
    }

    (out, rows)
}

/// Folded tool-output lines uniformly carry a `  │ ` prefix (4 columns); the body is
/// clamped to a single physical row using the terminal width minus 4.
pub(in crate::ai::driver::turn_runtime) fn clamp_tool_output_body(body: &str) -> String {
    const PREFIX_COLS: usize = 4;
    clamp_line_to_terminal_row_with_reserve(body, PREFIX_COLS)
}
