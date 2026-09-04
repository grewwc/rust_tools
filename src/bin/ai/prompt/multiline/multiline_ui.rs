use std::io;
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
    layout::{Position, Rect, Size},
};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};
use crate::commonw::prompt::acquire_foreground_stdin;

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
/// Rebuild only after a resize burst has been quiet for this long. Rebuilding
/// every intermediate size paints the fixed viewport on a new terminal row on
/// each frame in terminals such as VS Code, leaving stacked border/caret rows.
const RESIZE_SETTLE_DELAY: Duration = Duration::from_millis(150000);

/// Max candidate lines shown at once in the completion panel; aligned with
/// `render::COMPLETION_WINDOW`.
const PANEL_COMPLETION_WINDOW: u16 = 12;
/// Fallback chrome while the completion panel is active: the minimum textarea
/// lines(1) plus the compressed help line(1) = 2. The input area's left-edge
/// marker bar occupies a column, not a row, so it adds nothing here.
/// The completion state hides model/session info, giving height priority to the
/// candidate list.
const PANEL_CHROME_LINES: u16 = 1 + 1;
/// The completion state allows a taller prompt viewport than normal editing, so
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewportRebuildMode {
    ReserveMissingRows,
    ReflowOnly,
}

fn fixed_viewport_area(
    terminal_size: Size,
    cursor_position: Position,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
) -> (Rect, u16) {
    let height = requested_height.max(1).min(terminal_size.height.max(1));
    let live_top = cursor_position
        .y
        .saturating_sub(cursor_offset_row)
        .min(terminal_size.height.saturating_sub(1));
    let missing_rows = live_top
        .saturating_add(height)
        .saturating_sub(terminal_size.height);
    let lines_to_scroll = match mode {
        ViewportRebuildMode::ReserveMissingRows => missing_rows,
        ViewportRebuildMode::ReflowOnly => 0,
    };
    let viewport_top = live_top
        .saturating_sub(lines_to_scroll)
        .min(terminal_size.height.saturating_sub(height));
    (
        Rect::new(0, viewport_top, terminal_size.width, height),
        lines_to_scroll,
    )
}

fn prepare_fixed_viewport<B: Backend>(
    backend: &mut B,
    terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
) -> Result<Rect, B::Error> {
    // One synchronous DSR query is the authoritative cursor position for this
    // rebuild. Issuing an extra query cannot identify a stale reply (both use
    // the same untagged terminal response) and can instead leave another reply
    // in the input stream when a resize interrupts the round-trip.
    let cursor_position = backend.get_cursor_position()?;
    let (area, lines_to_scroll) = fixed_viewport_area(
        terminal_size,
        cursor_position,
        requested_height,
        cursor_offset_row,
        mode,
    );
    if lines_to_scroll > 0 {
        backend.set_cursor_position(Position::new(0, terminal_size.height.saturating_sub(1)))?;
        backend.append_lines(lines_to_scroll)?;
    }
    if clear_existing_viewport {
        // The box is always the last thing drawn, so every transcript row sits
        // ABOVE `area.y`: clearing from there to the end of the screen cannot
        // erase conversation history. This range is also the only one that
        // removes both ghost directions — a widening reflow moves the box up and
        // strands its old rows below the new bottom, while a narrowing reflow
        // pushes them further down. Bounding the clear to `area.height` rows
        // left exactly those stranded rows on screen.
        //
        // The transcript-loss this used to risk came from a wrong `area.y`
        // (collapsing to row 0 when the reflow anchor offset fell back to zero),
        // not from the clear itself; `parked_anchor_offset` fixes that input.
        backend.set_cursor_position(Position::new(0, area.y))?;
        backend.clear_region(BackendClearType::AfterCursor)?;
    }
    backend.flush()?;
    Ok(area)
}

fn terminal_with_fixed_viewport<B: Backend>(
    backend: B,
    area: Rect,
) -> Result<Terminal<B>, B::Error> {
    Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Fixed(area),
        },
    )
}

/// Row offset of the parked hardware cursor within the box, counted from the
/// box's top row.
///
/// `park_reflow_anchor` parks the hidden cursor on the box's bottom row after
/// every draw, so a DSR query returns that bottom row and subtracting this
/// offset recovers the box top. The offset must come from the box that is
/// actually on screen, not from the height the rebuild is about to apply: a
/// rebuild that changes the height (completion panel, content growth) still
/// reads a position parked by the previous box, and defaulting the offset to
/// zero in that case placed the recovered top `height - 1` rows too low —
/// which scrolled the box off-screen and cleared the transcript above it.
fn parked_anchor_offset(last_drawn_area: Option<Rect>, new_height: u16) -> u16 {
    last_drawn_area
        .map(|area| area.height.saturating_sub(1))
        .unwrap_or_else(|| new_height.saturating_sub(1))
}

/// Builds a fixed viewport immediately below the preceding output.
///
/// Ratatui's inline viewport calls `append_lines` on every real terminal resize,
/// even when the screen already has enough rows for the input area. IDE terminal
/// size notifications can oscillate, turning those reservations into a growing
/// blank gap. A fixed viewport avoids that automatic behavior; this bootstrap
/// scrolls only the rows that are actually missing at the bottom of the screen.
fn build_fixed_terminal(height: u16) -> io::Result<MultilineTerminal> {
    let mut backend = CrosstermBackend::new(io::stdout());
    let terminal_size = backend.size()?;
    let area = prepare_fixed_viewport(
        &mut backend,
        terminal_size,
        height,
        0,
        ViewportRebuildMode::ReserveMissingRows,
        false,
    )?;
    terminal_with_fixed_viewport(backend, area).map_err(|err| io::Error::other(err.to_string()))
}

/// Blanks the rows in `[from_row, to_row)` one line at a time, leaving every
/// row outside that range untouched.
fn clear_row_range<B: Backend>(
    backend: &mut B,
    from_row: u16,
    to_row: u16,
) -> Result<(), B::Error> {
    if to_row > from_row {
        for row in from_row..to_row {
            backend.set_cursor_position(Position::new(0, row))?;
            backend.clear_region(BackendClearType::CurrentLine)?;
        }
        backend.flush()?;
    }
    Ok(())
}

/// Re-anchors a fixed viewport after terminal reflow or a requested height
/// change without reserving another full block of terminal lines.
///
/// `clear_previous_extent` is reserved for explicit viewport-height changes,
/// where the box keeps its top row and may shrink; the rows a shorter box no
/// longer covers must be blanked. Width reflow moves the whole box together
/// with the transcript above it, so resize handling must leave it false —
/// clearing the previous extent there would erase re-wrapped transcript.
fn rebuild_fixed_viewport(
    terminal: &mut MultilineTerminal,
    terminal_size: Size,
    new_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    previous_top_row: Option<u16>,
    clear_previous_extent: bool,
) -> io::Result<Rect> {
    let area = prepare_fixed_viewport(
        terminal.backend_mut(),
        terminal_size,
        new_height,
        cursor_offset_row,
        mode,
        true,
    )?;
    *terminal = terminal_with_fixed_viewport(CrosstermBackend::new(io::stdout()), area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    // A height change keeps the box top fixed, so a SHRINKING box leaves its
    // former bottom rows on screen as ghosts. Blank exactly that remainder;
    // rows above the box hold transcript and must never be touched.
    if clear_previous_extent {
        if let Some(previous_top) = previous_top_row {
            // The parked anchor sits on the previous box's bottom row, so its
            // offset identifies that box's height.
            let previous_bottom =
                previous_top.saturating_add(cursor_offset_row.saturating_add(1));
            let new_bottom = area.y.saturating_add(area.height);
            clear_row_range(terminal.backend_mut(), new_bottom, previous_bottom)?;
        }
    }
    Ok(area)
}

/// Parks the hardware cursor at the viewport's bottom row, hidden.
///
/// The visible editing caret is drawn into the buffer as a reverse-video cell
/// (see render.rs), so it reflows with the text and needs no tracking. The
/// hardware cursor is parked at a FIXED row of the box instead — its bottom
/// row — because emulators preserve a cursor's logical line through width
/// reflow, and the box height never changes on a width reflow. A rebuild
/// recovers the reflowed top as `bottom - (height - 1)`, taking `height` from
/// the viewport that is actually on screen rather than from a stored offset
/// that a burst of resizes can leave stale.
fn park_reflow_anchor<B: Backend>(
    terminal: &mut Terminal<B>,
    viewport_area: Rect,
) -> Result<(), B::Error> {
    let anchor = Position::new(
        viewport_area.x,
        viewport_area.y.saturating_add(viewport_area.height.saturating_sub(1)),
    );
    // The hardware cursor stays hidden: a hidden cursor still tracks its
    // logical line through reflow, so DSR queries keep returning a valid
    // anchor, and the visible caret is the drawn reverse-video cell.
    terminal.backend_mut().hide_cursor()?;
    terminal.backend_mut().set_cursor_position(anchor)?;
    terminal.backend_mut().flush()?;
    Ok(())
}

fn clear_fixed_viewport<B: Backend>(
    terminal: &mut Terminal<B>,
    viewport_top_row: Option<u16>,
) -> Result<bool, B::Error> {
    let Some(top_row) = viewport_top_row else {
        return Ok(false);
    };
    terminal
        .backend_mut()
        .set_cursor_position(Position::new(0, top_row))?;
    terminal
        .backend_mut()
        .clear_region(BackendClearType::AfterCursor)?;
    terminal.backend_mut().flush()?;
    Ok(true)
}

/// Forces every cell of the current frame to be written back to the terminal.
///
/// After terminal reflow the terminal may still show a deleted
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

fn take_redraw_request(redraw_requested: &mut bool, external_change: bool) -> bool {
    *redraw_requested |= external_change;
    std::mem::take(redraw_requested)
}

/// Consumes consecutive resize notifications after the first one. The caller
/// rebuilds the viewport once this returns, using the final terminal size. A
/// non-resize event is returned rather than discarded so keyboard, paste,
/// focus, and mouse input keep their original ordering.
fn drain_resize_burst(
    mut poll_event: impl FnMut(Duration) -> io::Result<bool>,
    mut read_event: impl FnMut() -> io::Result<Event>,
) -> io::Result<Option<Event>> {
    loop {
        if !poll_event(RESIZE_SETTLE_DELAY)? {
            return Ok(None);
        }
        let event = read_event()?;
        if !matches!(event, Event::Resize(_, _)) {
            return Ok(Some(event));
        }
    }
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
        // The streaming side-note listener (Ctrl+G composer) owns stdin in
        // cbreak mode for the whole turn. Preempt it BEFORE touching stdin or
        // termios: the foreground flag makes it stop poll/read, restore termios
        // and release its stdin lease, and only then does this function return.
        // Without the handshake, the cursor-position query (\x1b[6n) below races
        // the listener for stdin, its DSR response gets consumed, the query
        // blocks until timeout, and the whole input box falls back to a
        // prompt-less line read — the terminal appears frozen after the final
        // answer with no input prompt.
        let _stdin_owner = acquire_foreground_stdin();
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

        // Empty input keeps 3 textarea lines by default; when editing existing
        // content the viewport grows by the prefilled line count, leaving the
        // textarea enough space.
        // The fallback must match the empty-input reserved height: on some
        // terminals (e.g. the VS Code integrated terminal) ioctl(TIOCGWINSZ) can
        // fail briefly at specific timings; falling back to a larger value then
        // would push extra blank lines above the textarea, showing up as a large
        // blank gap between the body and model/help. Falling back to
        // EMPTY_VIEWPORT_HEIGHT guarantees the needed input space even when the
        // size is unknown.
        let mut base_viewport_height = terminal_size()
            .ok()
            .map(|(_, h)| multiline_viewport_height(h, self.pending_prefill.as_deref()))
            .unwrap_or(EMPTY_VIEWPORT_HEIGHT);

        let mut terminal = match build_fixed_terminal(base_viewport_height) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(err);
            }
        };

        let initial_viewport_area = terminal.get_frame().area();
        // The viewport currently on screen. Its top row is where exit cleanup
        // starts, and its height is where the parked anchor's offset comes
        // from, so both stay consistent with what is actually drawn.
        let mut last_drawn_area: Option<Rect> = Some(initial_viewport_area);

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
            // resize the viewport accordingly so the panel gets enough height
            // while the textarea's line count stays unchanged.
            let mut fitted_completion_items: Option<usize> = None;
            // When the input gets shorter, force one frame write-back to wipe
            // characters left by desync between the ratatui buffer and the real
            // terminal.
            let mut force_repaint_next_frame = false;
            // Consume this flag after each frame so poll timeouts do not redraw
            // an unchanged input screen. Resize and title events explicitly
            // request the next frame.
            let mut redraw_requested = true;

            loop {
                // The background only publishes title updates; the terminal is
                // still redrawn by the foreground input loop at this safe draw
                // point.
                let title_changed = self.apply_pending_session_title_updates();

                if take_redraw_request(&mut redraw_requested, title_changed) {
                    // When the panel state changes, resize the fixed viewport to
                    // match the height the panel needs.
                    let current_items = completion_panel.as_ref().map(|p| p.items.len());
                    if current_items != fitted_completion_items {
                        let terminal_size = terminal.backend().size()?;
                        let new_height = viewport_height_with_completion(
                            terminal_size.height,
                            base_viewport_height,
                            current_items,
                        );
                        let rebuilt_area = rebuild_fixed_viewport(
                            &mut terminal,
                            terminal_size,
                            new_height,
                            parked_anchor_offset(last_drawn_area, new_height),
                            ViewportRebuildMode::ReserveMissingRows,
                            last_drawn_area.map(|area| area.y),
                            true,
                        )?;
                        // Re-park immediately: the rebuild moved the hardware
                        // cursor, and a resize arriving before the next draw
                        // would otherwise anchor to a stale row.
                        park_reflow_anchor(&mut terminal, rebuilt_area)?;
                        last_drawn_area = Some(rebuilt_area);
                        fitted_completion_items = current_items;
                        force_repaint_next_frame = false;
                    }

                    // Auto-grow the viewport when content exceeds the textarea capacity
                    // (grow only, never shrink, to avoid frequent flicker).
                    let content_lines = textarea.lines().len() as u16;
                    // Content rows that fit = viewport height minus the
                    // model/help chrome rows (the left-edge marker bar occupies
                    // a column, not a row).
                    let textarea_capacity = base_viewport_height.saturating_sub(VIEWPORT_CHROME_LINES);
                    if content_lines > textarea_capacity
                        && base_viewport_height < MAX_VIEWPORT_HEIGHT
                    {
                        let terminal_size = terminal.backend().size()?;
                        let available = terminal_size.height.saturating_sub(2).max(1);
                        let new_height = content_lines
                            .saturating_add(VIEWPORT_CHROME_LINES)
                            .min(MAX_VIEWPORT_HEIGHT)
                            .min(available);
                        if new_height > base_viewport_height {
                            let rebuilt_area = rebuild_fixed_viewport(
                                &mut terminal,
                                terminal_size,
                                new_height,
                                parked_anchor_offset(last_drawn_area, new_height),
                                ViewportRebuildMode::ReserveMissingRows,
                                last_drawn_area.map(|area| area.y),
                                true,
                            )?;
                            park_reflow_anchor(&mut terminal, rebuilt_area)?;
                            last_drawn_area = Some(rebuilt_area);
                            base_viewport_height = new_height;
                            force_repaint_next_frame = false;
                        }
                    }

                    let force_repaint = force_repaint_next_frame;
                    let mut drawn_viewport_area = Rect::ZERO;
                    // The visible editing caret is drawn into the buffer (see
                    // render.rs), so it cannot jump mid-draw. The hardware
                    // cursor is hidden while ratatui applies the frame diff and
                    // then parked at the box bottom by `park_reflow_anchor`.
                    terminal.hide_cursor()?;
                    terminal
                        .draw(|f| {
                            let area = f.area();
                            drawn_viewport_area = area;
                            last_drawn_area = Some(area);
                            let _ = render_multiline_popup(
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
                    park_reflow_anchor(&mut terminal, drawn_viewport_area)?;
                    self.notify_first_render();
                    force_repaint_next_frame = false;
                }

                if !event::poll(Duration::from_millis(250))
                    .map_err(|e| io::Error::other(e.to_string()))?
                {
                    continue;
                }
                let mut event = event::read().map_err(|e| io::Error::other(e.to_string()))?;
                if let Event::Resize(_, _) = event {
                    let deferred_event = drain_resize_burst(
                        |timeout| {
                            event::poll(timeout).map_err(|e| io::Error::other(e.to_string()))
                        },
                        || event::read().map_err(|e| io::Error::other(e.to_string())),
                    )?;
                    let terminal_size = terminal.backend().size()?;
                    let requested_height = viewport_height_with_completion(
                        terminal_size.height,
                        base_viewport_height,
                        fitted_completion_items,
                    );
                    // The hardware cursor is parked (hidden) at the viewport's
                    // bottom row, and the emulator preserves that logical line
                    // through the width reflow. A live DSR query returns where
                    // the bottom row landed; subtracting `height - 1` (the box
                    // height never changes on a width reflow) recovers the
                    // reflowed viewport top, so the box tracks the re-wrapped
                    // transcript exactly. Clearing from that top down removes
                    // the reflowed copy of the old viewport (no ghost rows), and
                    // the rows above stay untouched because they now hold the
                    // re-wrapped transcript (hence clear_gap_above stays false).
                    let rebuilt_area = rebuild_fixed_viewport(
                        &mut terminal,
                        terminal_size,
                        requested_height,
                        parked_anchor_offset(last_drawn_area, requested_height),
                        ViewportRebuildMode::ReserveMissingRows,
                        last_drawn_area.map(|area| area.y),
                        false,
                    )?;
                    // Store the rebuilt box and re-park immediately rather than
                    // waiting for the next draw: resize notifications arrive in
                    // a burst, and every rebuild must leave the anchor valid for
                    // the one that follows.
                    park_reflow_anchor(&mut terminal, rebuilt_area)?;
                    last_drawn_area = Some(rebuilt_area);
                    // Rebuilding starts with empty ratatui buffers and the viewport
                    // was explicitly cleared, so a forced all-cell repaint would
                    // only increase autowrap risk while the user is still resizing.
                    force_repaint_next_frame = false;
                    redraw_requested = true;
                    let Some(deferred_event) = deferred_event else {
                        continue;
                    };
                    event = deferred_event;
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
                        redraw_requested = true;
                    }
                    EventLoopAction::Submit(result) => break Ok(result),
                }
            }
        })();

        // Exiting the TUI: clear from the fixed viewport's reflowed top and leave
        // the cursor there, so the submit preview follows the previous output.
        let _ = terminal.hide_cursor();
        let cleared_viewport =
            clear_fixed_viewport(&mut terminal, last_drawn_area.map(|area| area.y)).unwrap_or(false);
        drop(terminal);
        if !cleared_viewport {
            let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        }
    let _ = execute!(io::stdout(), cursor::Show);
    // Restore the default cursor shape: the editor switched it to a thin bar
    // via DECSCUSR while active.
        let _ = execute!(io::stdout(), cursor::SetCursorStyle::DefaultUserShape);
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
    use std::collections::VecDeque;

    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use ratatui::{
        Terminal,
        backend::{Backend, TestBackend},
        buffer::Cell,
        layout::{Position, Rect},
        widgets::Paragraph,
    };

    use super::{
        RESIZE_SETTLE_DELAY, ViewportRebuildMode, clear_fixed_viewport,
        clear_row_range, drain_resize_burst, fixed_viewport_area,
        force_frame_repaint, multiline_viewport_height, park_reflow_anchor,
        parked_anchor_offset,
        prepare_fixed_viewport, submitted_input_preview_lines, take_redraw_request,
        terminal_with_fixed_viewport, viewport_height_with_completion,
    };

    #[test]
    fn idle_poll_timeouts_do_not_request_more_frames() {
        let mut redraw_requested = true;

        assert!(take_redraw_request(&mut redraw_requested, false));
        assert!(!take_redraw_request(&mut redraw_requested, false));
        assert!(!take_redraw_request(&mut redraw_requested, false));

        assert!(take_redraw_request(&mut redraw_requested, true));
        assert!(!take_redraw_request(&mut redraw_requested, false));
    }

    #[test]
    fn resize_burst_waits_for_quiet_before_rebuilding() {
        let mut readiness = VecDeque::from([true, true, false]);
        let mut events = VecDeque::from([Event::Resize(100, 30), Event::Resize(120, 30)]);
        let mut poll_timeouts = Vec::new();

        let deferred = drain_resize_burst(
            |timeout| {
                poll_timeouts.push(timeout);
                Ok(readiness.pop_front().unwrap())
            },
            || Ok(events.pop_front().unwrap()),
        )
        .unwrap();

        assert!(deferred.is_none());
        assert!(events.is_empty());
        assert_eq!(poll_timeouts, vec![RESIZE_SETTLE_DELAY; 3]);
    }

    #[test]
    fn resize_burst_preserves_first_non_resize_event() {
        let mut readiness = VecDeque::from([true, true]);
        let mut events = VecDeque::from([
            Event::Resize(100, 30),
            Event::Key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE)),
        ]);

        let deferred = drain_resize_burst(
            |_| Ok(readiness.pop_front().unwrap()),
            || Ok(events.pop_front().unwrap()),
        )
        .unwrap();

        assert!(matches!(
            deferred,
            Some(Event::Key(key)) if key.code == KeyCode::Char('x')
        ));
        assert!(events.is_empty());
    }

    #[test]
    fn fixed_viewport_scrolls_only_rows_missing_below_cursor() {
        let size = ratatui::layout::Size {
            width: 80,
            height: 24,
        };
        let (area, lines_to_scroll) = fixed_viewport_area(
            size,
            Position::new(0, 10),
            5,
            0,
            ViewportRebuildMode::ReserveMissingRows,
        );
        assert_eq!(area, ratatui::layout::Rect::new(0, 10, 80, 5));
        assert_eq!(lines_to_scroll, 0);

        let (area, lines_to_scroll) = fixed_viewport_area(
            size,
            Position::new(0, 22),
            5,
            0,
            ViewportRebuildMode::ReserveMissingRows,
        );
        assert_eq!(area, ratatui::layout::Rect::new(0, 19, 80, 5));
        assert_eq!(lines_to_scroll, 3);

        let (area, lines_to_scroll) = fixed_viewport_area(
            size,
            Position::new(0, 22),
            5,
            0,
            ViewportRebuildMode::ReflowOnly,
        );
        assert_eq!(area, ratatui::layout::Rect::new(0, 19, 80, 5));
        assert_eq!(lines_to_scroll, 0);
    }

    #[test]
    fn height_only_resize_reanchors_from_live_cursor_without_losing_output() {
        // Screen fully filled: the box occupies the bottom rows, simulated
        // model output sits above it. Shrinking the height re-anchors the box
        // around the parked bottom-row anchor without appending rows, so the output above
        // stays on screen instead of being pushed into scrollback.
        let mut backend = TestBackend::new(10, 17);
        backend.set_cursor_position(Position::new(0, 12)).unwrap();
        let initial_area = prepare_fixed_viewport(
            &mut backend,
            ratatui::layout::Size {
                width: 10,
                height: 17,
            },
            5,
            0,
            ViewportRebuildMode::ReserveMissingRows,
            false,
        )
        .unwrap();
        assert_eq!(initial_area, Rect::new(0, 12, 10, 5));

        // Height shrink 17 -> 10: the terminal keeps the editing caret visible
        // at the bottom, which puts the box top at row 5.
        backend.resize(10, 10);
        backend.set_cursor_position(Position::new(0, 9)).unwrap();
        for row in 0..5 {
            let output = Cell::new("K");
            backend.draw(std::iter::once((0, row, &output))).unwrap();
        }
        let scrollback_before = backend.scrollback().area.height;

        let rebuilt_area = prepare_fixed_viewport(
            &mut backend,
            ratatui::layout::Size {
                width: 10,
                height: 10,
            },
            5,
            4,
            ViewportRebuildMode::ReflowOnly,
            true,
        )
        .unwrap();

        assert_eq!(rebuilt_area, Rect::new(0, 5, 10, 5));
        assert_eq!(backend.scrollback().area.height, scrollback_before);
        for row in 0..5 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
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
    fn fixed_viewport_rebuilds_without_growing_scrollback() {
        let mut backend = TestBackend::new(10, 6);
        backend.set_cursor_position(Position::new(0, 5)).unwrap();
        let initial_area = prepare_fixed_viewport(
            &mut backend,
            ratatui::layout::Size {
                width: 10,
                height: 6,
            },
            3,
            0,
            ViewportRebuildMode::ReserveMissingRows,
            false,
        )
        .unwrap();
        assert_eq!(initial_area, ratatui::layout::Rect::new(0, 3, 10, 3));
        let reserved_scrollback_height = backend.scrollback().area.height;
        assert_eq!(reserved_scrollback_height, 2);

        let mut final_area = initial_area;
        for (width, cursor_row) in [(9, 5), (10, 4), (9, 5), (10, 5)] {
            backend.resize(width, 6);
            backend
                .set_cursor_position(Position::new(0, cursor_row))
                .unwrap();
            final_area = prepare_fixed_viewport(
                &mut backend,
                ratatui::layout::Size { width, height: 6 },
                3,
                2,
                ViewportRebuildMode::ReflowOnly,
                true,
            )
            .unwrap();
            assert!(final_area.bottom() <= 6);
            assert_eq!(backend.scrollback().area.height, reserved_scrollback_height);
        }

        let mut terminal = terminal_with_fixed_viewport(backend, final_area).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(Paragraph::new("INPUT\nSTATUS"), frame.area());
                frame.set_cursor_position(Position::new(0, frame.area().y));
            })
            .unwrap();
        assert_eq!(terminal.backend().buffer()[(0, 3)].symbol(), "I");
        assert_eq!(terminal.backend().buffer()[(0, 4)].symbol(), "S");
        assert!(clear_fixed_viewport(&mut terminal, Some(3)).unwrap());
        assert_eq!(terminal.backend().buffer()[(0, 3)].symbol(), " ");
    }

    #[test]
    fn bottom_cursor_anchor_tracks_width_reflow_without_growing_scrollback() {
        let mut backend = TestBackend::new(12, 10);
        let scrollback_height = backend.scrollback().area.height;

        // A narrow terminal wraps the three-row transcript into five rows and
        // moves the viewport's parked bottom-row cursor with it. Widening
        // restores the original three rows. Rebuilding from that anchor must preserve every
        // transcript row without appending permanent blank lines.
        for (width, transcript_rows) in [(8, 5), (12, 3), (8, 5), (12, 3)] {
            backend.resize(width, 10);
            for row in 0..transcript_rows {
                let output = Cell::new("K");
                backend.draw(std::iter::once((0, row, &output))).unwrap();
            }
            backend
                .set_cursor_position(Position::new(0, transcript_rows + 3))
                .unwrap();
            let area = prepare_fixed_viewport(
                &mut backend,
                ratatui::layout::Size { width, height: 10 },
                4,
                3,
                ViewportRebuildMode::ReflowOnly,
                true,
            )
            .unwrap();

            assert_eq!(area, Rect::new(0, transcript_rows, width, 4));
            assert_eq!(backend.scrollback().area.height, scrollback_height);
            for row in 0..transcript_rows {
                assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
            }
        }
    }

    #[test]
    fn width_reflow_rebuild_clears_old_viewport_copy_and_keeps_transcript() {
        // Narrowing reflow: the transcript re-wraps and grows, and the old
        // viewport (with the parked caret) is pushed down by the same amount.
        // Rebuilding from the DSR-reported caret position must place the box
        // exactly below the taller transcript, wipe the reflowed copy of the
        // old viewport (no ghost rows), and keep every transcript row on screen
        // without growing scrollback.
        let mut backend = TestBackend::new(12, 10);

        // Old layout before narrowing: transcript rows [0,3), viewport [3,7).
        for row in 0..3 {
            let output = Cell::new("K");
            backend.draw(std::iter::once((0, row, &output))).unwrap();
        }
        for row in 3..7 {
            let ghost = Cell::new("X");
            backend.draw(std::iter::once((0, row, &ghost))).unwrap();
        }
        let scrollback_before = backend.scrollback().area.height;

        // Emulator applied the narrowing reflow: the transcript now wraps to
        // five rows, the old viewport content moved to rows [5,9), and the
        // caret (parked at viewport offset 3) landed on row 8.
        backend.resize(12, 10);
        for row in 0..5 {
            let output = Cell::new("K");
            backend.draw(std::iter::once((0, row, &output))).unwrap();
        }
        for row in 5..9 {
            let ghost = Cell::new("X");
            backend.draw(std::iter::once((0, row, &ghost))).unwrap();
        }
        backend.set_cursor_position(Position::new(0, 8)).unwrap();

        let area = prepare_fixed_viewport(
            &mut backend,
            ratatui::layout::Size {
                width: 12,
                height: 10,
            },
            4,
            3,
            ViewportRebuildMode::ReflowOnly,
            true,
        )
        .unwrap();

        assert_eq!(area, Rect::new(0, 5, 12, 4));
        assert_eq!(backend.scrollback().area.height, scrollback_before);
        // The taller transcript stays visible above the box.
        for row in 0..5 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
        }
        // The reflowed copy of the old viewport is wiped, not left as a ghost.
        for row in 5..9 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), " ");
        }
    }

    #[test]
    fn width_reflow_widening_moves_box_up_and_clears_old_rows() {
        // Widening reflow: the transcript re-wraps and shrinks, so the box moves
        // up. Clearing from the new top down must wipe the old (now lower)
        // viewport copy while keeping the shorter transcript visible.
        let mut backend = TestBackend::new(12, 10);

        // Old layout before widening: transcript rows [0,5), viewport [5,9).
        for row in 0..5 {
            let output = Cell::new("K");
            backend.draw(std::iter::once((0, row, &output))).unwrap();
        }
        for row in 5..9 {
            let ghost = Cell::new("X");
            backend.draw(std::iter::once((0, row, &ghost))).unwrap();
        }
        let scrollback_before = backend.scrollback().area.height;

        // Emulator applied the widening reflow: the transcript now wraps to
        // three rows, the old viewport content moved up to rows [3,7), and the
        // caret (offset 3) landed on row 6.
        backend.resize(12, 10);
        for row in 0..3 {
            let output = Cell::new("K");
            backend.draw(std::iter::once((0, row, &output))).unwrap();
        }
        for row in 3..7 {
            let ghost = Cell::new("X");
            backend.draw(std::iter::once((0, row, &ghost))).unwrap();
        }
        backend.set_cursor_position(Position::new(0, 6)).unwrap();

        let area = prepare_fixed_viewport(
            &mut backend,
            ratatui::layout::Size {
                width: 12,
                height: 10,
            },
            4,
            3,
            ViewportRebuildMode::ReflowOnly,
            true,
        )
        .unwrap();

        assert_eq!(area, Rect::new(0, 3, 12, 4));
        assert_eq!(backend.scrollback().area.height, scrollback_before);
        for row in 0..3 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
        }
        // The old viewport copy below the new top is wiped.
        for row in 3..9 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), " ");
        }
    }

    #[test]
    fn clear_row_range_blanks_only_the_requested_rows() {
        // The viewport clear is bounded to rows the box will actually repaint.
        // Clearing beyond that range is what used to erase the transcript above
        // a mis-anchored box, turning a positioning error into data loss.
        let mut backend = TestBackend::new(10, 8);
        for row in 1..7 {
            let filler = Cell::new("K");
            backend.draw(std::iter::once((0, row, &filler))).unwrap();
        }

        clear_row_range(&mut backend, 3, 5).unwrap();

        for row in 3..5 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), " ");
        }
        // Rows above and below the range are untouched.
        for row in [1, 2, 5, 6] {
            assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
        }
        // An empty or inverted range clears nothing.
        clear_row_range(&mut backend, 5, 5).unwrap();
        clear_row_range(&mut backend, 6, 5).unwrap();
        assert_eq!(backend.buffer()[(0, 6)].symbol(), "K");
    }

    #[test]
    fn parked_anchor_offset_comes_from_the_box_on_screen() {
        // The anchor is parked on the bottom row of the box that is currently
        // drawn. A rebuild that changes the height must still derive the offset
        // from that on-screen box; falling back to zero here is what placed the
        // recovered top `height - 1` rows too low and scrolled the transcript
        // away.
        let drawn = Rect::new(0, 2, 20, 4);
        assert_eq!(parked_anchor_offset(Some(drawn), 6), 3);
        // With no drawn box yet (first rebuild) fall back to the new height.
        assert_eq!(parked_anchor_offset(None, 6), 5);
    }

    #[test]
    fn reflow_anchor_is_parked_at_viewport_bottom_row_and_hidden() {
        let backend = TestBackend::new(10, 10);
        let area = Rect::new(0, 3, 10, 4);
        let mut terminal = terminal_with_fixed_viewport(backend, area).unwrap();

        park_reflow_anchor(&mut terminal, area).unwrap();

        // The hidden hardware cursor is parked at the box's bottom row on screen,
        // which is the fixed row every rebuild reads back through DSR.
        terminal
            .backend_mut()
            .assert_cursor_position(Position::new(0, 6));
        assert!(!terminal.backend().cursor_visible());
    }

    #[test]
    fn multiline_viewport_height_scales_with_terminal() {
        // Empty input: viewport = 3 input lines + chrome(2) = 5. The left-edge
        // marker bar occupies a column, not a row, so it adds no height.
        assert_eq!(multiline_viewport_height(30, None), 5);
        assert_eq!(multiline_viewport_height(30, Some("")), 5);
        // Prefilled but shorter than base: keep the base size. base textarea is
        // 7 content lines; 7 + chrome(2) = 9.
        assert_eq!(multiline_viewport_height(30, Some("one line")), 9);
        // Small terminal: terminal=12, available=10; empty input still keeps its
        // 5-line box (3 content + chrome).
        assert_eq!(multiline_viewport_height(12, None), 5);
        // On large terminals empty input still keeps the same compact box.
        assert_eq!(multiline_viewport_height(40, None), 5);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        // terminal=40: available=38, base_textarea=7, content=20→clamp(7,7)=7,
        // viewport = 7 + chrome(2) = 9.
        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 9);
        assert_eq!(multiline_viewport_height(10, Some(&prefill)), 8);
        assert_eq!(multiline_viewport_height(4, Some(&prefill)), 2);
        assert_eq!(multiline_viewport_height(4, None), 2); // available=2; still bounded by available lines
    }

    #[test]
    fn completion_viewport_grows_with_candidates_without_shrinking_base() {
        // No panel: return the passed-in base height unchanged (here 5).
        assert_eq!(viewport_height_with_completion(30, 5, None), 5);
        // Panel chrome = 1 content line + help(1) = 2 (the left marker bar adds
        // no row). 1 candidate: panel 1+2(borders)=3 + chrome(2)=5, equal to base.
        assert_eq!(viewport_height_with_completion(30, 5, Some(1)), 5);
        // 3 candidates: panel 3+2=5 + chrome(2)=7; the viewport grows and the
        // extra lines go to the panel.
        assert_eq!(viewport_height_with_completion(30, 5, Some(3)), 7);
        // Many candidates: the completion-state cap is 16 = chrome(2) + 12
        // candidate lines + panel borders(2).
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
