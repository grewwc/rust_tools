use std::io::{self, Write};
use std::time::Duration;

use crossterm::{
    cursor,
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size as terminal_size},
};
use ratatui::{
    Terminal,
    backend::{Backend, ClearType as BackendClearType, CrosstermBackend},
    buffer::CellDiffOption,
    layout::Position,
};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

/// Maximum viewport height (textarea + chrome); scales with the terminal size,
/// capped at 11 lines.
const MAX_VIEWPORT_HEIGHT: u16 = 11;
/// Upper bound on textarea lines (a comfortable value on large terminals).
const MAX_TEXTAREA_LINES: u16 = 7;
/// Fixed chrome line count for the normal editing state: model(1) + help(1).
/// No decorative divider is drawn anymore, so stray horizontal lines do not
/// pile up after terminal resizes.
const VIEWPORT_CHROME_LINES: u16 = 2;
/// Minimum textarea line count, used for clamping.
const MIN_TEXTAREA_LINES: u16 = 2;
/// With empty input, reserve 3 lines for the textarea plus the fixed chrome.
const EMPTY_VIEWPORT_HEIGHT: u16 = 3 + VIEWPORT_CHROME_LINES;

/// Max candidate lines shown at once in the completion panel; aligned with
/// `render::COMPLETION_WINDOW`.
const PANEL_COMPLETION_WINDOW: u16 = 12;
/// Fallback chrome while the completion panel is active: minimum textarea
/// lines(1) + compressed help line(1) = 2.
/// The completion state hides model/session info, giving height priority to the
/// candidate list.
const PANEL_CHROME_LINES: u16 = 2;
/// The completion state allows a taller inline viewport than normal editing, so
/// large terminals can show more candidates at once.
const MAX_COMPLETION_VIEWPORT_HEIGHT: u16 = PANEL_CHROME_LINES + PANEL_COMPLETION_WINDOW + 2;

fn multiline_viewport_height(terminal_rows: u16, prefill: Option<&str>) -> u16 {
    let available_rows = terminal_rows.saturating_sub(2).max(1);
    // With empty input, keep 3 textarea lines by default so the input area is
    // not too narrow.
    if prefill.is_none_or(str::is_empty) {
        return EMPTY_VIEWPORT_HEIGHT
            .min(available_rows)
            .min(MAX_VIEWPORT_HEIGHT);
    }
    // Base textarea lines: 1/4 of the terminal's available lines, at least MIN,
    // at most MAX
    let base_textarea = (available_rows / 4).clamp(MIN_TEXTAREA_LINES, MAX_TEXTAREA_LINES);
    // With prefilled content, textarea lines are at least base and fit the
    // content, capped at MAX_TEXTAREA_LINES
    let content_rows = prefill.map(|text| text.lines().count().max(1)).unwrap_or(1) as u16;
    let textarea = content_rows.clamp(base_textarea, MAX_TEXTAREA_LINES);
    let viewport = textarea.saturating_add(VIEWPORT_CHROME_LINES);
    viewport.min(available_rows).min(MAX_VIEWPORT_HEIGHT)
}

/// Viewport height needed while the completion panel is active: extra space for
/// the panel while keeping the textarea's line count unchanged. Desired panel
/// lines = min(candidates, PANEL_COMPLETION_WINDOW) + top/bottom borders(2),
/// plus PANEL_CHROME_LINES (minimum textarea lines + compressed help line).
/// When this does not exceed base_height (height without a panel), use
/// base_height directly so a tiny panel never shrinks the viewport.
fn viewport_height_with_completion(
    terminal_rows: u16,
    base_height: u16,
    completion_items: Option<usize>,
) -> u16 {
    let available_rows = terminal_rows.saturating_sub(2).max(1);
    let base = base_height.min(available_rows);
    let Some(items) = completion_items else {
        return base;
    };
    let visible = (items.min(PANEL_COMPLETION_WINDOW as usize) as u16).max(1);
    let panel_lines = visible.saturating_add(2); // top/bottom borders
    let desired = panel_lines.saturating_add(PANEL_CHROME_LINES);
    desired
        .max(base)
        .min(MAX_COMPLETION_VIEWPORT_HEIGHT)
        .min(available_rows)
}

type MultilineTerminal = Terminal<CrosstermBackend<io::Stdout>>;

fn build_inline_terminal(height: u16) -> io::Result<MultilineTerminal> {
    let backend = CrosstermBackend::new(io::stdout());
    Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height.max(1)),
        },
    )
    .map_err(|err| io::Error::other(err.to_string()))
}

/// Deletes the real terminal lines previously reserved by the inline viewport.
///
/// Ratatui reserves `Viewport::Inline` height by emitting newlines; a regular
/// clear only erases cells and cannot reclaim those lines. The standard ANSI
/// `CSI Ps M` deletes lines starting at the cursor, letting the post-submit
/// preview follow the previous round's output directly. Some terminals handle
/// leftover cells or cursor position after line deletion inconsistently, so we
/// finally return to the top line and clear the area after it, ensuring the
/// next regular output starts at that top line. The return value says whether
/// enough last-frame state exists to perform the line deletion.
fn delete_inline_viewport_rows<W: Write>(
    output: &mut W,
    viewport_top_row: Option<u16>,
    viewport_height: Option<u16>,
) -> io::Result<bool> {
    let (Some(top_row), Some(height)) = (viewport_top_row, viewport_height) else {
        return Ok(false);
    };
    if height == 0 {
        return Ok(false);
    }

    execute!(output, cursor::MoveTo(0, top_row))?;
    output.write_all(format!("\x1b[{height}M").as_bytes())?;
    execute!(
        output,
        cursor::MoveTo(0, top_row),
        Clear(ClearType::FromCursorDown),
    )?;
    output.flush()?;
    Ok(true)
}

/// After clearing the inline viewport, re-anchors the cursor to the old
/// viewport's top.
///
/// `Terminal::clear()` restores the cursor from before the call, while a new
/// `Viewport::Inline` takes the row of the backend cursor at creation time as
/// its top edge. Rebuilding directly would treat the cursor row inside the
/// input box as the new top edge, leaving the original viewport's top content
/// stuck in scrollback. When the height changed between two frames, trust the
/// viewport top row recorded in the previous frame; on some terminals the live
/// cursor read after input drifts to the top-left corner temporarily.
/// Only when the first frame has not been drawn yet, fall back to computing the
/// top row from the cursor's offset relative to the viewport.
fn clear_and_reanchor_inline_viewport<B: Backend>(
    terminal: &mut Terminal<B>,
    last_viewport_top_row: Option<u16>,
    last_cursor_offset_row: Option<u16>,
) -> Result<(), B::Error> {
    let viewport_top_row = match last_viewport_top_row {
        Some(top_row) => top_row,
        None => {
            let cursor_position = terminal.backend_mut().get_cursor_position()?;
            last_cursor_offset_row
                .map(|offset| cursor_position.y.saturating_sub(offset))
                .unwrap_or(cursor_position.y)
        }
    };

    terminal.clear()?;
    terminal
        .backend_mut()
        .set_cursor_position(Position::new(0, viewport_top_row))
}

/// When the completion panel opens/closes, the inline viewport's required height
/// changes, and ratatui's inline viewport height is fixed at creation and cannot
/// be modified in place. After clearing, put the cursor back at the old
/// viewport's top, then rebuild the Terminal with the new height; ratatui then
/// expands downward via `append_lines` from the same top anchor, so neither
/// growing nor shrinking leaves the old frame in scrollback. The textarea's
/// line count is unaffected — the added/reclaimed height only affects the panel
/// area.
fn resize_inline_viewport(
    terminal: &mut MultilineTerminal,
    new_height: u16,
    last_viewport_top_row: Option<u16>,
    last_cursor_offset_row: Option<u16>,
) -> io::Result<()> {
    let _ = terminal.hide_cursor();
    clear_and_reanchor_inline_viewport(terminal, last_viewport_top_row, last_cursor_offset_row)?;
    *terminal = build_inline_terminal(new_height)?;
    Ok(())
}

/// After a horizontal resize the terminal first reflows existing content, so
/// ratatui's saved viewport top row still holds pre-resize coordinates.
/// Re-locate the top row from the real cursor and the previous frame's relative
/// row offset, and erase the old viewport first so it is not pushed into
/// scrollback on the next `autoresize` re-anchor.
fn clear_reflowed_inline_viewport<B: Backend>(
    terminal: &mut Terminal<B>,
    cursor_offset_row: u16,
) -> Result<(), B::Error> {
    let cursor_position = terminal.backend_mut().get_cursor_position()?;
    let viewport_top = cursor_position.y.saturating_sub(cursor_offset_row);
    terminal
        .backend_mut()
        .set_cursor_position(Position::new(0, viewport_top))?;
    terminal
        .backend_mut()
        .clear_region(BackendClearType::CurrentLine)?;
    terminal
        .backend_mut()
        .clear_region(BackendClearType::AfterCursor)?;
    terminal
        .backend_mut()
        .set_cursor_position(cursor_position)?;
    terminal.backend_mut().flush()
}

/// Forces every cell of the current frame to be written back to the terminal.
///
/// In some inline viewport scenarios the terminal may still show a deleted
/// character, while ratatui's previous-frame buffer already considers that
/// position blank, so a regular diff would not emit a space there again. Use
/// `AlwaysUpdate` only on the frame after the input got shorter: it wipes such
/// ghosts and avoids a full redraw every frame.
fn force_frame_repaint(frame: &mut ratatui::Frame<'_>) {
    let area = frame.area();
    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            buffer[(x, y)].set_diff_option(CellDiffOption::AlwaysUpdate);
        }
    }
}

fn textarea_logical_char_count(textarea: &TextArea<'_>) -> usize {
    textarea
        .lines()
        .iter()
        .map(|line| line.chars().count())
        .sum::<usize>()
        .saturating_add(textarea.lines().len().saturating_sub(1))
}

fn submitted_input_preview_lines(content: &str) -> Vec<String> {
    let mut rendered = Vec::new();
    let mut lines = content.lines();
    let marker = crate::ai::theme::ACCENT_SUCCESS;
    // The post-submit preview uses a soft mid-purple, distinct from the editing
    // state's warm gray, readable on both light and dark backgrounds.
    let body = crate::ai::theme::ACCENT_SUBMITTED;
    if let Some(first) = lines.next() {
        // Bold green `>` marker + low-saturation warm-gray body, matching the
        // textarea editing-state colors.
        rendered.push(format!("\x1b[1m{marker}❯\x1b[0m {body}{first}\x1b[0m"));
        for line in lines {
            rendered.push(format!("  {body}{line}\x1b[0m"));
        }
    }
    rendered
}

fn print_submitted_input_preview(content: &str) {
    for line in submitted_input_preview_lines(content) {
        println!("{line}");
    }
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        enable_raw_mode()?;

        // Disable bracketed paste under SSH: after the terminal intercepts
        // Ctrl+V, clipboard content that is an image (binary) cannot travel
        // through bracketed paste, leaving the paste event empty or never fired.
        // With it disabled, Ctrl+V produces Event::Key(Ctrl+V) directly, and the
        // handler reads the clipboard through the OSC52 path.
        let is_ssh = std::env::var("SSH_CONNECTION").is_ok()
            || std::env::var("SSH_CLIENT").is_ok()
            || std::env::var("SSH_TTY").is_ok();
        if is_ssh {
            let _ = execute!(io::stdout(), DisableBracketedPaste);
        } else {
            let _ = execute!(io::stdout(), EnableBracketedPaste);
        }

        // Inline viewport initialization really expands the terminal area via
        // append_lines(). Empty input keeps 3 textarea lines by default; when
        // editing existing content it grows by the prefilled line count, leaving
        // the textarea enough space.
        // The fallback must match the empty-input reserved height: on some
        // terminals (e.g. the VS Code integrated terminal) ioctl(TIOCGWINSZ) can
        // fail briefly at specific timings; falling back to a larger value then
        // would push extra blank lines above the textarea, showing up as a large
        // blank gap between the body and model/help. Falling back to
        // EMPTY_VIEWPORT_HEIGHT guarantees the needed input space even when the
        // size is unknown.
        let mut base_viewport_height = terminal_size()
            .map(|(_, h)| multiline_viewport_height(h, self.pending_prefill.as_deref()))
            .unwrap_or(EMPTY_VIEWPORT_HEIGHT);

        let mut terminal = match build_inline_terminal(base_viewport_height) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(err);
            }
        };

        // On exit, clean up the viewport based on "where the last actual render
        // ended"; do not rely on the cursor position at Terminal creation, because
        // completion-panel/textarea growth rebuilds the inline viewport.
        let mut last_viewport_top_row: Option<u16> = None;
        // `Viewport::Inline` reserves real terminal lines; on exit you must delete
        // the height actually rendered last, not a value derived from the base
        // height (the completion panel and short terminals both change the actual
        // height).
        let mut last_viewport_height: Option<u16> = None;
        // After a resize reflow the viewport's absolute coordinates change, but the
        // cursor's relative row inside the viewport does not; record that offset
        // to clear the old frame before the next autoresize.
        let mut last_cursor_offset_row: Option<u16> = None;

        let result: io::Result<Option<String>> = (|| {
            // Prefilled content (editing an existing memo): load into the textarea
            // line by line, then clear the source.
            let mut textarea: TextArea = match self.pending_prefill.take() {
                Some(text) => TextArea::from(text.lines().map(|l| l.to_string())),
                None => TextArea::default(),
            };
            let mut history = MultilineHistoryState::new(self.multiline_history_entries());
            let mut status_msg: Option<String> = self.pending_status_msg.take();
            let mut pending_tab_completion: Option<PendingTabCompletion> = None;
            let mut completion_panel: Option<CompletionPanel> = None;
            let mut recent_text_input: Option<RecentTextInput> = None;
            // Record how many completion candidates the current viewport already
            // accommodates: None means no panel (base height).
            // When the panel appears/disappears or the candidate count changes,
            // rebuild the viewport accordingly so the panel gets enough height
            // while the textarea's line count stays unchanged.
            let mut fitted_completion_items: Option<usize> = None;
            // When the input gets shorter, force one frame write-back to wipe
            // characters left by desync between the ratatui buffer and the real
            // terminal.
            let mut force_repaint_next_frame = false;

            loop {
                // The background only publishes title updates; the terminal is
                // still redrawn by the foreground input loop at this safe draw
                // point.
                self.apply_pending_session_title_updates();

                // When the panel state changes, rebuild the inline viewport to
                // match the height the panel needs.
                let current_items = completion_panel.as_ref().map(|p| p.items.len());
                if current_items != fitted_completion_items {
                    let terminal_rows = terminal_size().map(|(_, h)| h).unwrap_or(0);
                    let new_height = viewport_height_with_completion(
                        terminal_rows,
                        base_viewport_height,
                        current_items,
                    );
                    resize_inline_viewport(
                        &mut terminal,
                        new_height,
                        last_viewport_top_row,
                        last_cursor_offset_row,
                    )?;
                    fitted_completion_items = current_items;
                }

                // Auto-grow the viewport when content exceeds the textarea capacity
                // (grow only, never shrink, to avoid frequent flicker).
                let content_lines = textarea.lines().len() as u16;
                let textarea_capacity = base_viewport_height.saturating_sub(VIEWPORT_CHROME_LINES);
                if content_lines > textarea_capacity && base_viewport_height < MAX_VIEWPORT_HEIGHT {
                    let terminal_rows = terminal_size().map(|(_, h)| h).unwrap_or(0);
                    let available = terminal_rows.saturating_sub(2).max(1);
                    let new_height = content_lines
                        .saturating_add(VIEWPORT_CHROME_LINES)
                        .min(MAX_VIEWPORT_HEIGHT)
                        .min(available);
                    if new_height > base_viewport_height {
                        resize_inline_viewport(
                            &mut terminal,
                            new_height,
                            last_viewport_top_row,
                            last_cursor_offset_row,
                        )?;
                        base_viewport_height = new_height;
                    }
                }

                let force_repaint = force_repaint_next_frame;
                terminal
                    .draw(|f| {
                        let area = f.area();
                        last_viewport_top_row = Some(area.y);
                        last_viewport_height = Some(area.height);
                        last_cursor_offset_row = render_multiline_popup(
                            f,
                            &mut textarea,
                            status_msg.as_deref(),
                            completion_panel.as_ref(),
                            &self.current_model_label,
                            &self.current_reasoning_effort_label,
                            self.session_topic.as_deref(),
                        );
                        if force_repaint {
                            force_frame_repaint(f);
                        }
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;
                self.notify_first_render();
                force_repaint_next_frame = false;

                if !event::poll(Duration::from_millis(250))
                    .map_err(|e| io::Error::other(e.to_string()))?
                {
                    continue;
                }
                let event = event::read().map_err(|e| io::Error::other(e.to_string()))?;
                if matches!(event, Event::Resize(_, _)) {
                    // VS Code has finished its horizontal reflow; first recompute
                    // from the real cursor and clean up the old viewport, then let
                    // the next `Terminal::draw` call `autoresize` to re-anchor and
                    // redraw.
                    if let Some(cursor_offset_row) = last_cursor_offset_row {
                        clear_reflowed_inline_viewport(&mut terminal, cursor_offset_row)
                            .map_err(|e| io::Error::other(e.to_string()))?;
                    }
                    continue;
                }

                let previous_input_len = textarea_logical_char_count(&textarea);
                match handle_multiline_event(
                    event,
                    &mut textarea,
                    &mut history,
                    &mut status_msg,
                    &mut pending_tab_completion,
                    &mut completion_panel,
                    &mut recent_text_input,
                    &self.session_image_dir,
                )? {
                    EventLoopAction::Continue => {
                        force_repaint_next_frame =
                            textarea_logical_char_count(&textarea) < previous_input_len;
                    }
                    EventLoopAction::Submit(result) => break Ok(result),
                }
            }
        })();

        // Exiting the TUI: Ratatui's clear only erases characters; the lines
        // actually appended by the inline viewport must be deleted separately,
        // otherwise the next submit preview or model output appears after those
        // blank lines.
        let _ = terminal.hide_cursor();
        let can_delete_rows = last_viewport_top_row
            .is_some_and(|_| last_viewport_height.is_some_and(|height| height > 0));
        if !can_delete_rows {
            let _ = terminal.clear();
        }
        drop(terminal);
        if can_delete_rows {
            let _ = delete_inline_viewport_rows(
                &mut io::stdout(),
                last_viewport_top_row,
                last_viewport_height,
            );
        } else if let Some(top_row) = last_viewport_top_row {
            let _ = execute!(
                io::stdout(),
                cursor::MoveTo(0, top_row),
                Clear(ClearType::FromCursorDown),
            );
        } else {
            let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        }
        let _ = execute!(io::stdout(), cursor::Show);
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = disable_raw_mode();

        let result = match result {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                return interrupted_error();
            }
            Err(err) => return Err(err),
            Ok(result) => result,
        };
        if let Some(content) = &result {
            self.save_history_entry(content);
            print_submitted_input_preview(content);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{
        Terminal,
        backend::{Backend, TestBackend},
        buffer::Cell,
        layout::Position,
        widgets::Paragraph,
    };

    use super::{
        clear_and_reanchor_inline_viewport, clear_reflowed_inline_viewport,
        delete_inline_viewport_rows, force_frame_repaint, multiline_viewport_height,
        submitted_input_preview_lines, viewport_height_with_completion,
    };

    #[test]
    fn inline_viewport_row_deletion_moves_to_top_and_deletes_rendered_height() {
        let mut output = Vec::new();

        assert!(delete_inline_viewport_rows(&mut output, Some(4), Some(3)).unwrap());
        assert_eq!(output, b"\x1b[5;1H\x1b[3M\x1b[5;1H\x1b[J");
    }

    #[test]
    fn inline_viewport_row_deletion_requires_a_nonempty_rendered_viewport() {
        for (top_row, height) in [(None, Some(3)), (Some(4), None), (Some(4), Some(0))] {
            let mut output = Vec::new();
            assert!(!delete_inline_viewport_rows(&mut output, top_row, height).unwrap());
            assert!(output.is_empty());
        }
    }

    #[test]
    fn forced_repaint_clears_character_missing_from_ratatui_back_buffer() {
        let backend = TestBackend::new(4, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|_| {}).unwrap();

        // Simulate the real terminal still showing a character while ratatui's
        // previous-frame buffer already considers that position blank.
        let slash = Cell::new("/");
        terminal
            .backend_mut()
            .draw(std::iter::once((0, 0, &slash)))
            .unwrap();
        terminal.draw(|frame| force_frame_repaint(frame)).unwrap();

        assert_eq!(terminal.backend().buffer()[(0, 0)].symbol(), " ");
    }

    #[test]
    fn clear_reflowed_viewport_uses_cursor_relative_top_and_restores_cursor() {
        let backend = TestBackend::new(8, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(Paragraph::new("row0\nrow1\nrow2\nrow3\nrow4"), frame.area());
            })
            .unwrap();
        let cursor = Position::new(3, 4);
        terminal.backend_mut().set_cursor_position(cursor).unwrap();

        clear_reflowed_inline_viewport(&mut terminal, 2).unwrap();

        assert_eq!(
            terminal.backend_mut().get_cursor_position().unwrap(),
            cursor
        );
        assert_eq!(terminal.backend().buffer()[(0, 1)].symbol(), "r");
        assert_eq!(terminal.backend().buffer()[(0, 2)].symbol(), " ");
        assert_eq!(terminal.backend().buffer()[(0, 4)].symbol(), " ");
    }

    #[test]
    fn clearing_and_reanchoring_inline_viewport_uses_previous_viewport_top() {
        let backend = TestBackend::new(8, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .backend_mut()
            .set_cursor_position(Position::new(3, 4))
            .unwrap();

        clear_and_reanchor_inline_viewport(&mut terminal, Some(1), Some(2)).unwrap();

        assert_eq!(
            terminal.backend_mut().get_cursor_position().unwrap(),
            Position::new(0, 1)
        );
    }

    #[test]
    fn multiline_viewport_height_scales_with_terminal() {
        // Empty input: viewport = 3 input lines + chrome(2) = 5
        assert_eq!(multiline_viewport_height(30, None), 5);
        assert_eq!(multiline_viewport_height(30, Some("")), 5);
        // Prefilled but shorter than base: keep the base size
        assert_eq!(multiline_viewport_height(30, Some("one line")), 9);
        // Small terminal: terminal=12, available=10; empty input still keeps 3
        // input lines
        assert_eq!(multiline_viewport_height(12, None), 5);
        // On large terminals empty input still keeps 3 input lines
        assert_eq!(multiline_viewport_height(40, None), 5);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        // terminal=40: available=38, base_textarea=7, content=20→clamp(7,7)=7, viewport=9
        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 9);
        assert_eq!(multiline_viewport_height(10, Some(&prefill)), 8);
        assert_eq!(multiline_viewport_height(4, Some(&prefill)), 2);
        assert_eq!(multiline_viewport_height(4, None), 2); // available=2; still bounded by available lines
    }

    #[test]
    fn completion_viewport_grows_with_candidates_without_shrinking_base() {
        // No panel: keep the empty-input base height (5).
        assert_eq!(viewport_height_with_completion(30, 5, None), 5);
        // 1 candidate: panel needs 1+2(borders)=3 + 2(chrome)=5, same as base.
        assert_eq!(viewport_height_with_completion(30, 5, Some(1)), 5);
        // 3 candidates: 3+2+2=7 > base; the viewport grows to 7, the extra 3
        // lines go to the panel.
        assert_eq!(viewport_height_with_completion(30, 5, Some(3)), 7);
        // Many candidates: the completion-state cap is separately raised to 16,
        // fitting 12 candidate lines + borders + compressed chrome.
        assert_eq!(viewport_height_with_completion(30, 5, Some(50)), 16);
    }

    #[test]
    fn completion_viewport_capped_by_available_terminal_rows() {
        // With a 12-line terminal available=10; even if the panel wants more it
        // cannot exceed 10.
        assert_eq!(viewport_height_with_completion(12, 4, Some(50)), 10);
        // The base itself is also bounded by available.
        assert_eq!(viewport_height_with_completion(6, 8, None), 4);
    }

    #[test]
    fn submitted_input_preview_formats_single_and_multi_line_content() {
        let marker = crate::ai::theme::ACCENT_SUCCESS;
        let body = crate::ai::theme::ACCENT_SUBMITTED;
        let reset = "\x1b[0m";
        assert_eq!(
            submitted_input_preview_lines("hello"),
            vec![format!("\x1b[1m{marker}❯{reset} {body}hello{reset}")]
        );
        assert_eq!(
            submitted_input_preview_lines("hello\nworld"),
            vec![
                format!("\x1b[1m{marker}❯{reset} {body}hello{reset}"),
                format!("  {body}world{reset}"),
            ]
        );
    }
}
