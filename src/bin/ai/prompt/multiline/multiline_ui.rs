use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};

use crossterm::{
    cursor,
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode, size as terminal_size,
    },
};
use ratatui::{
    Terminal,
    backend::{Backend, ClearType as BackendClearType, CrosstermBackend},
    buffer::{Cell, CellDiffOption},
    layout::{Position, Rect, Size},
    style::{Modifier, Style},
};
use tui_textarea::TextArea;
use unicode_width::UnicodeWidthChar;

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

/// How long a single ambiguous Escape is deferred so a fragmented CPR cannot
/// submit the textarea.
const CPR_ESCAPE_GRACE: Duration = Duration::from_millis(250);

/// How long a timed-out DSR reply may still be expected in the input stream.
/// Past this the reply was lost with the connection (an SSH drop discards it in
/// transit), so the guard stops deferring Escapes for it and recovery stops
/// waiting for it.
const CPR_REPLY_WINDOW: Duration = Duration::from_secs(10);

/// Upper bound on unaccounted replies kept in the guard queue. Every timed-out
/// query pushes one, so the queue only grows while the link stays down.
const MAX_PENDING_CPR_REPLIES: usize = 8;

/// Delays before successive recovery probes: the first runs while a one-off
/// stall is likely to have cleared, later ones back off to a steady 30 s so a
/// dead link costs one query per interval instead of one per frame.
const RECOVERY_PROBE_BACKOFFS: [Duration; 4] = [
    Duration::from_secs(1),
    Duration::from_secs(5),
    Duration::from_secs(15),
    Duration::from_secs(30),
];

/// A main-screen probe is only safe once no *recent* reply is outstanding: an
/// older reply would already have been delivered, so an answer arriving now is
/// the probe's own.
const CPR_PROBE_QUIET_PERIOD: Duration = Duration::from_secs(2);

fn recovery_probe_delay(failures: u32) -> Duration {
    let index = (failures as usize).min(RECOVERY_PROBE_BACKOFFS.len() - 1);
    RECOVERY_PROBE_BACKOFFS[index]
}

/// Crossterm emits a key if ESC arrives in its own read. After a query timeout,
/// briefly defer that ambiguous key so a fragmented CPR cannot submit the
/// textarea. `pending_cpr_replies` holds one entry per timed-out query whose
/// reply may still be in the input stream: the first inline query, its retry,
/// and every failed recovery probe. Each entry is removed when its reply is
/// actually observed, so every orphan gets swallowed and a real Escape only
/// pays the grace once per outstanding reply. Entries expire after
/// `CPR_REPLY_WINDOW`, because a reply that late was lost with the connection
/// and deferring forever would tax every later Escape submission.
fn read_prompt_event(
    pending: &mut VecDeque<Event>,
    pending_cpr_replies: &mut VecDeque<Instant>,
    mut read: impl FnMut(Duration) -> io::Result<Option<Event>>,
) -> io::Result<Option<Event>> {
    if let Some(event) = pending.pop_front() {
        return Ok(Some(event));
    }
    let Some(first) = read(Duration::from_millis(250))? else {
        return Ok(None);
    };
    prune_expired_cpr_replies(pending_cpr_replies, Instant::now());
    if pending_cpr_replies.is_empty()
        || !matches!(&first, Event::Key(key) if key.code == KeyCode::Esc
            && key.modifiers.is_empty() && key.kind == KeyEventKind::Press)
    {
        return Ok(Some(first));
    }

    let deadline = Instant::now() + CPR_ESCAPE_GRACE;
    let mut tail = String::new();
    let mut lookahead = VecDeque::new();
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        let Some(event) = read(remaining)? else { break };
        lookahead.push_back(event.clone());
        if matches!(event, Event::Resize(_, _)) {
            continue;
        }
        let Event::Key(key) = event else { break };
        let KeyCode::Char(ch) = key.code else { break };
        if key.kind != KeyEventKind::Press
            || !(key.modifiers.is_empty() || (ch == 'R' && key.modifiers == KeyModifiers::SHIFT))
        {
            break;
        }
        tail.push(ch);
        match cursor_reply_tail(&tail) {
            Some(true) => {
                // Resize notifications are real input, not part of the reply.
                pending.extend(
                    lookahead
                        .into_iter()
                        .filter(|event| matches!(event, Event::Resize(_, _))),
                );
                // One orphan ESC from a timed-out query is consumed.
                pending_cpr_replies.pop_front();
                return Ok(None);
            }
            Some(false) => {}
            None => break,
        }
    }
    // Replay every event on a mismatch or timeout, including across submissions.
    pending.extend(lookahead);
    // The Escape was a real key (no CPR tail), but the outstanding replies stay
    // queued: a reply that is merely slow must still be swallowed when it
    // arrives, otherwise its trailing `[row;colR` would be typed into the box.
    Ok(Some(first))
}

/// Drops replies that can no longer arrive, oldest first.
fn prune_expired_cpr_replies(pending: &mut VecDeque<Instant>, now: Instant) {
    while pending
        .front()
        .is_some_and(|query| now.duration_since(*query) >= CPR_REPLY_WINDOW)
    {
        pending.pop_front();
    }
}

/// Some(false) is a CPR prefix, Some(true) a complete reply, None ordinary input.
fn cursor_reply_tail(tail: &str) -> Option<bool> {
    let tail = tail.strip_prefix('[')?;
    let (coordinates, complete) = tail.strip_suffix('R').map_or((tail, false), |s| (s, true));
    let mut parts = coordinates.split(';');
    let row = parts.next()?;
    let column = parts.next();
    let digits = |s: &str| s.len() <= 5 && s.bytes().all(|byte| byte.is_ascii_digit());
    if parts.next().is_some() || !digits(row) || column.is_some_and(|s| !digits(s)) {
        return None;
    }
    if column.is_some() && row.is_empty() {
        return None;
    }
    if complete && (row.is_empty() || column.is_none_or(str::is_empty)) {
        return None;
    }
    Some(complete)
}

/// A timed-out DSR reply must remain inside the event parser, not cooked stdin.
/// The alternate screen provides known coordinates without another query and
/// keeps the main transcript intact, but it also hides that transcript, so the
/// editor keeps probing the main screen and returns to it as soon as a query
/// works again. Query disabling survives prompt sessions: a late reply has no
/// request id and must not anchor a later inline viewport.
#[derive(Default)]
struct PromptScreen {
    queries_disabled: bool,
    alternate: bool,
    /// Send time of every query whose reply is still unaccounted for, oldest
    /// first. Feeds the orphan-Escape guard in `read_prompt_event`, and holds
    /// recovery probes back while a fresh reply could be mistaken for theirs.
    pending_cpr_replies: VecDeque<Instant>,
    /// One inline retry per prompt session: a rebuild must not block on two
    /// two-second waits per frame once the link really is down.
    retry_used: bool,
    /// Failed query/probe attempts so far, used to space out the next probe.
    recovery_failures: u32,
    /// Earliest instant for the next main-screen probe.
    next_recovery_probe: Option<Instant>,
    /// Tail of the last model output, repainted above the box on the alternate
    /// screen (`PromptEditor::set_alternate_screen_tail`).
    tail: Vec<String>,
}

impl PromptScreen {
    /// Records `timed_out` queries whose replies may still arrive, and schedules
    /// the next attempt to get back onto the main screen.
    fn note_timed_out_queries(&mut self, timed_out: u8, now: Instant) {
        self.note_pending_cpr(timed_out, now);
        self.queries_disabled = true;
        // The first probe runs while a one-off stall is likely to have cleared;
        // later ones back off so a dead link costs one query per interval instead
        // of one per frame.
        self.next_recovery_probe = Some(now + recovery_probe_delay(self.recovery_failures));
        self.recovery_failures = self.recovery_failures.saturating_add(1);
    }

    /// Retain the number of DSR replies that may still arrive without changing
    /// whether future inline queries are enabled.
    fn note_pending_cpr(&mut self, pending: u8, now: Instant) {
        for _ in 0..pending {
            self.pending_cpr_replies.push_back(now);
        }
        while self.pending_cpr_replies.len() > MAX_PENDING_CPR_REPLIES {
            self.pending_cpr_replies.pop_front();
        }
        prune_expired_cpr_replies(&mut self.pending_cpr_replies, now);
    }

    /// True when the main screen may be probed again: only once no *recent*
    /// reply is outstanding (an older one would already have been delivered, so
    /// an answer arriving now is the probe's own) and the backoff has elapsed.
    fn recovery_probe_due(&mut self, now: Instant) -> bool {
        self.replies_settled(now)
            && self.queries_disabled
            && self.next_recovery_probe.is_none_or(|at| now >= at)
    }

    /// True while no *recent* reply is outstanding, i.e. a query answer arriving
    /// now belongs to that query. An older outstanding reply was either already
    /// delivered or lost with the link, so it no longer competes.
    fn replies_settled(&mut self, now: Instant) -> bool {
        prune_expired_cpr_replies(&mut self.pending_cpr_replies, now);
        self
            .pending_cpr_replies
            .back()
            .is_none_or(|query| now.duration_since(*query) >= CPR_PROBE_QUIET_PERIOD)
    }

    /// Clears the probe schedule after a probe put the inline box back.
    fn note_recovered(&mut self) {
        self.recovery_failures = 0;
        self.next_recovery_probe = None;
    }

    fn prepare_viewport(
        &mut self,
        backend: &mut CrosstermBackend<io::Stdout>,
        terminal_size: Size,
        requested_height: u16,
        cursor_offset_row: u16,
        mode: ViewportRebuildMode,
        clear_existing_viewport: bool,
        previous_top_row: Option<u16>,
    ) -> io::Result<Rect> {
        let now = Instant::now();
        if !self.queries_disabled && self.replies_settled(now) {
            match prepare_fixed_viewport(
                backend,
                terminal_size,
                requested_height,
                cursor_offset_row,
                mode,
                clear_existing_viewport,
            ) {
                Ok(area) => return Ok(area),
                Err(err) if PromptEditor::is_cursor_position_timeout(&err) => {
                    // A stalled round-trip is usually a transient link hiccup, so
                    // retry once before paying for the alternate screen. Both
                    // attempts read the same parked cursor row — nothing is drawn
                    // in between — so even the answer to the first query anchors
                    // the second one correctly. A reflow invalidates that (the
                    // anchor moves with the re-wrapped text), which is what the
                    // size comparison detects. Reply timestamps are taken when an
                    // attempt gives up, so the retry's own two seconds do not
                    // count against the reply window or the probe backoff.
                    if !self.retry_used && backend.size().ok() == Some(terminal_size) {
                        self.retry_used = true;
                        match prepare_fixed_viewport(
                            backend,
                            terminal_size,
                            requested_height,
                            cursor_offset_row,
                            mode,
                            clear_existing_viewport,
                        ) {
                            Ok(area) => {
                                // One reply is still unaccounted for: the first
                                // timed-out query or, if that reply was consumed
                                // by the retry, the retry's own response. Keep
                                // later queries from treating it as authoritative.
                                self.note_pending_cpr(1, Instant::now());
                                return Ok(area);
                            }
                            Err(err) if PromptEditor::is_cursor_position_timeout(&err) => {
                                self.note_timed_out_queries(2, Instant::now());
                            }
                            Err(err) => return Err(err),
                        }
                    } else {
                        self.note_timed_out_queries(1, Instant::now());
                    }
                }
                Err(err) => return Err(err),
            }
        }
        if !self.alternate {
            // The inline box is already drawn on the main screen when a
            // mid-editing rebuild falls back here. The emulator saves the main
            // screen on EnterAlternateScreen and restores it verbatim on
            // LeaveAlternateScreen at exit, so the old box (including the
            // drawn caret cell) would come back as a permanent ghost over the
            // transcript. Clear it first: transcript rows sit ABOVE the box
            // top, so this touches nothing but the box. (If a width reflow
            // has already moved the box down, a few re-wrapped transcript
            // tail rows may also be blanked; a DSR query is the only way to
            // tell, and it just timed out.)
            if let Some(top_row) = previous_top_row {
                backend.set_cursor_position(Position::new(0, top_row))?;
                backend.clear_region(BackendClearType::AfterCursor)?;
                backend.flush()?;
            }
            // Set this before writing so cleanup also runs on a partial write.
            self.alternate = true;
            execute!(io::stdout(), EnterAlternateScreen)?;
        }
        if !self.queries_disabled {
            // A recent reply, not a dead link, kept the inline path out: query
            // again as soon as that reply can no longer answer the query, or the
            // inline probe would never run and the box would stay on the
            // alternate screen for the rest of the prompt.
            self.queries_disabled = true;
            self.next_recovery_probe = Some(now + CPR_PROBE_QUIET_PERIOD);
        }
        prepare_query_free_viewport(backend, terminal_size, requested_height, &self.tail)
    }
}

impl Drop for PromptScreen {
    fn drop(&mut self) {
        if self.alternate {
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
        }
    }
}

/// Rows of mirrored model output kept above the box on the alternate screen.
/// Bounded so a viewport rebuild never repaints more than a few rows over a
/// slow link.
const MAX_ALTERNATE_TAIL_ROWS: u16 = 12;

/// Only used on the alternate screen, where clearing cannot erase transcript.
///
/// The box is anchored to the bottom of the screen with the tail of the last
/// model output mirrored directly above it: this fallback has no working cursor
/// query left to place an inline box under the real transcript (the emulator
/// keeps that one saved and hidden), so without the mirror the answer the user
/// wants to read is the one thing missing from the screen.
fn prepare_query_free_viewport<B: Backend>(
    backend: &mut B,
    terminal_size: Size,
    requested_height: u16,
    tail: &[String],
) -> Result<Rect, B::Error> {
    let height = requested_height.max(1).min(terminal_size.height.max(1));
    let area = Rect::new(
        0,
        terminal_size.height.saturating_sub(height),
        terminal_size.width,
        height,
    );
    backend.set_cursor_position(Position::new(0, 0))?;
    backend.clear_region(BackendClearType::All)?;
    draw_alternate_tail(backend, terminal_size.width, area.y, tail)?;
    // Park the cursor at the box top: that is the position the following frame
    // (and any redraw) starts from.
    backend.set_cursor_position(Position::new(area.x, area.y))?;
    backend.flush()?;
    Ok(area)
}

/// Mirror the newest rows of the model output tail directly above the box.
///
/// These rows sit outside the ratatui viewport, which `Viewport::Fixed` never
/// draws into or clears, so the text stays until the next rebuild.
fn draw_alternate_tail<B: Backend>(
    backend: &mut B,
    width: u16,
    bottom_row: u16,
    tail: &[String],
) -> Result<(), B::Error> {
    if tail.is_empty() || width == 0 || bottom_row == 0 {
        return Ok(());
    }
    let rows = alternate_tail_rows(tail, width, bottom_row.min(MAX_ALTERNATE_TAIL_ROWS) as usize);
    if rows.is_empty() {
        return Ok(());
    }
    let style = Style::default().add_modifier(Modifier::DIM);
    let first_row = bottom_row - rows.len() as u16;
    // One cell per grapheme column: trailing columns of a wide glyph are left
    // out of the iterator (the terminal advances over them), because the
    // backend prints every cell it is handed.
    let mut cells: Vec<(u16, u16, Cell)> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let y = first_row + index as u16;
        let mut x = 0u16;
        for ch in row.chars() {
            let mut cell = Cell::from(ch);
            cell.set_style(style);
            cells.push((x, y, cell));
            x += UnicodeWidthChar::width_cjk(ch).unwrap_or(1).max(1) as u16;
        }
    }
    backend.draw(cells.iter().map(|(x, y, cell)| (*x, *y, cell)))?;
    Ok(())
}

/// Wrap the mirrored tail to `width` columns and keep the newest `max_rows`
/// rows. Wrapping rather than truncating keeps the closing sentence readable,
/// which is the part the user is trying to recover.
fn alternate_tail_rows(tail: &[String], width: u16, max_rows: usize) -> Vec<String> {
    let width = width as usize;
    if width == 0 || max_rows == 0 {
        return Vec::new();
    }
    let mut rows: VecDeque<String> = VecDeque::new();
    for line in tail {
        let mut row = String::new();
        let mut row_width = 0usize;
        for ch in line.chars() {
            let ch_width = UnicodeWidthChar::width_cjk(ch).unwrap_or(1).max(1);
            if row_width + ch_width > width && !row.is_empty() {
                rows.push_back(std::mem::take(&mut row));
                row_width = 0;
                // Drop rows that scrolled past the top of the mirror.
                while rows.len() > max_rows {
                    rows.pop_front();
                }
            }
            row.push(ch);
            row_width += ch_width;
        }
        rows.push_back(row);
        while rows.len() > max_rows {
            rows.pop_front();
        }
    }
    rows.into_iter().collect()
}

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
/// Rebuilt viewport after a recovery probe: the caller adopts this terminal and
/// treats `area` as the viewport already in place, because the probe restored a
/// screen without drawing a box frame on it.
struct RecoveredViewport {
    terminal: MultilineTerminal,
    area: Rect,
    /// Screen size the probe anchored for. Callers must record this instead of
    /// re-reading the backend size: a resize that landed during the probe's
    /// blocking cursor query would otherwise look already applied, and its
    /// queued `Event::Resize` would be dropped as a duplicate, leaving a box
    /// anchored for the pre-resize screen (and a stale row for exit cleanup).
    size: Size,
}

/// Leaves the alternate screen and re-places the box on the main screen once a
/// DSR round-trip works again.
///
/// The alternate screen hides the real transcript until the editor exits, so it
/// is held only while queries keep failing: every prompt start and every idle
/// tick retries the inline path through here. Returns `None` while the next
/// probe is not due yet.
fn try_recover_inline_viewport(
    screen: &mut PromptScreen,
    new_height: u16,
) -> io::Result<Option<RecoveredViewport>> {
    if !screen.recovery_probe_due(Instant::now()) {
        return Ok(None);
    }
    let mut backend = CrosstermBackend::new(io::stdout());
    let terminal_size = backend.size()?;
    if screen.alternate {
        // Leaving restores the main screen together with the cursor row where the
        // inline box was cleared before entering, which is exactly the box top —
        // so the query answer needs no parked-bottom-row offset.
        execute!(io::stdout(), LeaveAlternateScreen)?;
        screen.alternate = false;
    }
    screen.queries_disabled = false;
    // The probe already is a second chance for the query that degraded the
    // screen, so it must not also spend the prompt's one inline retry.
    screen.retry_used = true;
    let area = screen.prepare_viewport(
        &mut backend,
        terminal_size,
        new_height,
        0,
        ViewportRebuildMode::ReserveMissingRows,
        false,
        None,
    )?;
    if screen.queries_disabled {
        // The probe timed out as well: `prepare_viewport` already re-entered the
        // alternate screen and armed the next backoff.
    } else {
        screen.note_recovered();
    }
    let terminal = terminal_with_fixed_viewport(backend, area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    Ok(Some(RecoveredViewport {
        terminal,
        area,
        size: terminal_size,
    }))
}

/// Returns the terminal together with the screen size its anchor was computed
/// for; the caller records that size instead of a fresh `size()` read, so a
/// resize that landed during the build is not mistaken for one already applied.
fn build_fixed_terminal(
    height: u16,
    screen: &mut PromptScreen,
) -> io::Result<(MultilineTerminal, Size)> {
    // A previous timeout must not pin the whole session to the alternate screen:
    // try the inline path again, and fall through to the query-free one if it
    // still fails.
    if let Some(recovered) = try_recover_inline_viewport(screen, height)? {
        return Ok((recovered.terminal, recovered.size));
    }
    let mut backend = CrosstermBackend::new(io::stdout());
    let terminal_size = backend.size()?;
    let area = screen.prepare_viewport(
        &mut backend,
        terminal_size,
        height,
        0,
        ViewportRebuildMode::ReserveMissingRows,
        false,
        None,
    )?;
    let terminal = terminal_with_fixed_viewport(backend, area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    Ok((terminal, terminal_size))
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
    screen: &mut PromptScreen,
    terminal_size: Size,
    new_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    previous_top_row: Option<u16>,
    clear_previous_extent: bool,
) -> io::Result<Rect> {
    let area = screen.prepare_viewport(
        terminal.backend_mut(),
        terminal_size,
        new_height,
        cursor_offset_row,
        mode,
        true,
        previous_top_row,
    )?;
    *terminal = terminal_with_fixed_viewport(CrosstermBackend::new(io::stdout()), area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    // A height change keeps the box top fixed, so a SHRINKING box leaves its
    // former bottom rows on screen as ghosts. Blank exactly that remainder;
    // rows above the box hold transcript and must never be touched.
    if clear_previous_extent && !screen.alternate {
        if let Some(previous_top) = previous_top_row {
            // The parked anchor sits on the previous box's bottom row, so its
            // offset identifies that box's height.
            let previous_bottom = previous_top.saturating_add(cursor_offset_row.saturating_add(1));
            let new_bottom = area.y.saturating_add(area.height);
            clear_row_range(terminal.backend_mut(), new_bottom, previous_bottom)?;
        }
    }
    Ok(area)
}

/// Re-anchor the existing fixed viewport after terminal width/height reflow.
///
/// A resize is normally consumed before the next keyboard event, but a
/// background session-title update can request the first foreground redraw
/// after a long idle period. That redraw must re-anchor first rather than paint
/// the old fixed coordinates over the reflowed terminal content.
fn rebuild_after_terminal_reflow(
    terminal: &mut MultilineTerminal,
    screen: &mut PromptScreen,
    base_viewport_height: u16,
    fitted_completion_items: Option<usize>,
    last_drawn_area: Option<Rect>,
) -> io::Result<Rect> {
    let terminal_size = terminal.backend().size()?;
    let requested_height = viewport_height_with_completion(
        terminal_size.height,
        base_viewport_height,
        fitted_completion_items,
    );
    let rebuilt_area = rebuild_fixed_viewport(
        terminal,
        screen,
        terminal_size,
        requested_height,
        parked_anchor_offset(last_drawn_area, requested_height),
        // Reflow moves an already-reserved viewport; it must never append more
        // terminal rows. Appending here makes every delayed resize notification
        // scroll the inline editor and strands the previously drawn caret.
        ViewportRebuildMode::ReflowOnly,
        last_drawn_area.map(|area| area.y),
        false,
    )?;
    park_reflow_anchor(terminal, rebuilt_area)?;
    Ok(rebuilt_area)
}

/// Parks the hardware cursor at the viewport's bottom row, hidden.
///
/// The visible editing caret is drawn into the buffer as a styled cell (see
/// render.rs), so it reflows with the text and needs no tracking. The hardware
/// cursor is parked at a FIXED row of the box instead — its bottom
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
        viewport_area
            .y
            .saturating_add(viewport_area.height.saturating_sub(1)),
    );
    // The hardware cursor stays hidden: a hidden cursor still tracks its
    // logical line through reflow, so DSR queries keep returning a valid
    // anchor, and the visible caret is the drawn styled cell.
    terminal.backend_mut().hide_cursor()?;
    terminal.backend_mut().set_cursor_position(anchor)?;
    terminal.backend_mut().flush()?;
    Ok(())
}

fn update_pending_resize_rebuild(
    pending_resize_rebuild: &mut bool,
    last_applied_terminal_size: Size,
    notified_terminal_size: Size,
) {
    // VS Code/xterm.js can deliver a delayed duplicate Resize notification.
    // Re-querying and clearing the viewport for an unchanged size can move the
    // anchor without any real reflow and leave the drawn caret behind. If a
    // resize burst returns to the applied size before rebuilding, cancel the
    // pending rebuild for the same reason.
    *pending_resize_rebuild = notified_terminal_size != last_applied_terminal_size;
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

/// Consume a deferred resize at a redraw safe point.
///
/// A viewport-height rebuild already reads and re-anchors the live cursor, so
/// no second reflow rebuild is needed in that case. Otherwise the caller must
/// rebuild before it draws the frame.
fn take_standalone_resize_rebuild(
    pending_resize_rebuild: &mut bool,
    viewport_rebuilt: bool,
) -> bool {
    let needs_rebuild = *pending_resize_rebuild && !viewport_rebuilt;
    *pending_resize_rebuild = false;
    needs_rebuild
}

fn submitted_input_preview_lines(content: &str) -> Vec<String> {
    let mut rendered = Vec::new();
    let mut lines = content.lines();
    let marker = crate::ai::theme::current().accent_success;
    // The post-submit preview body uses the theme's `accent.submitted`, a bright
    // signature warm hue (amber/yellow on dark themes, deep purple on light).
    // It is deliberately NOT a white/gray: assistant output is full of near-white
    // body/strong text, so the echoed user input must carry its own hue to stay
    // visible and recognizable as the user's own text. The bold green marker
    // keeps the submit boundary distinct.
    let body = crate::ai::theme::current().accent_submitted;
    if let Some(first) = lines.next() {
        // Bold `❯` marker marks the submit boundary; the body color is theme-driven.
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
        // the listener for stdin: a stolen response would force an unnecessary
        // timeout and a switch to the query-free editing screen.
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

        let mut screen = PromptScreen {
            queries_disabled: self.cursor_position_queries_disabled,
            pending_cpr_replies: self.pending_cpr_replies.clone(),
            tail: self.alternate_tail_lines.clone(),
            ..PromptScreen::default()
        };
        let (mut terminal, anchored_terminal_size) =
            match build_fixed_terminal(base_viewport_height, &mut screen) {
                Ok(built) => built,
                Err(err) => {
                    self.cursor_position_queries_disabled = screen.queries_disabled;
                    self.pending_cpr_replies = std::mem::take(&mut screen.pending_cpr_replies);
                    drop(screen);
                    let _ = execute!(io::stdout(), DisableBracketedPaste, cursor::Show);
                    let _ = disable_raw_mode();
                    return Err(err);
                }
            };

        let initial_viewport_area = terminal.get_frame().area();
        let mut last_applied_terminal_size = anchored_terminal_size;
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
            // Set only when a resize notification changes the terminal size. The
            // rebuild runs before the next foreground redraw (including a
            // background title update) or before a non-resize input event, giving
            // the emulator time to finish its asynchronous scrollback reflow.
            let mut pending_resize_rebuild = false;

            loop {
                // The background only publishes title updates; the terminal is
                // still redrawn by the foreground input loop at this safe draw
                // point.
                let title_changed = self.apply_pending_session_title_updates();

                if take_redraw_request(&mut redraw_requested, title_changed) {
                    let mut viewport_rebuilt = false;
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
                            &mut screen,
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
                        last_applied_terminal_size = terminal_size;
                        fitted_completion_items = current_items;
                        force_repaint_next_frame = false;
                        viewport_rebuilt = true;
                    }

                    // Auto-grow the viewport when content exceeds the textarea capacity
                    // (grow only, never shrink, to avoid frequent flicker).
                    let content_lines = textarea.lines().len() as u16;
                    // Content rows that fit = viewport height minus the
                    // model/help chrome rows (the left-edge marker bar occupies
                    // a column, not a row).
                    let textarea_capacity =
                        base_viewport_height.saturating_sub(VIEWPORT_CHROME_LINES);
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
                                &mut screen,
                                terminal_size,
                                new_height,
                                parked_anchor_offset(last_drawn_area, new_height),
                                ViewportRebuildMode::ReserveMissingRows,
                                last_drawn_area.map(|area| area.y),
                                true,
                            )?;
                            park_reflow_anchor(&mut terminal, rebuilt_area)?;
                            last_drawn_area = Some(rebuilt_area);
                            last_applied_terminal_size = terminal_size;
                            base_viewport_height = new_height;
                            force_repaint_next_frame = false;
                            viewport_rebuilt = true;
                        }
                    }

                    if take_standalone_resize_rebuild(&mut pending_resize_rebuild, viewport_rebuilt)
                    {
                        let rebuilt_area = rebuild_after_terminal_reflow(
                            &mut terminal,
                            &mut screen,
                            base_viewport_height,
                            fitted_completion_items,
                            last_drawn_area,
                        )?;
                        last_drawn_area = Some(rebuilt_area);
                        last_applied_terminal_size = terminal.backend().size()?;
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

                let prompt_event = read_prompt_event(
                    &mut self.pending_terminal_events,
                    &mut screen.pending_cpr_replies,
                    |timeout| {
                        if event::poll(timeout)? {
                            event::read().map(Some)
                        } else {
                            Ok(None)
                        }
                    },
                )?;
                let Some(event) = prompt_event else {
                    // Idle tick: the alternate screen is a fallback, not a
                    // destination. Probe the main screen for a working query and
                    // re-anchor the inline box there.
                    let probe_height = viewport_height_with_completion(
                        terminal.backend().size()?.height,
                        base_viewport_height,
                        fitted_completion_items,
                    );
                    if let Some(recovered) = try_recover_inline_viewport(&mut screen, probe_height)? {
                        terminal = recovered.terminal;
                        last_drawn_area = Some(recovered.area);
                        // The probe's own size, not a fresh read: a resize that
                        // landed during its blocking query must still queue the
                        // reflow rebuild (see `build_fixed_terminal`).
                        last_applied_terminal_size = recovered.size;
                        redraw_requested = true;
                    }
                    continue;
                };
                if let Event::Resize(width, height) = event {
                    // Do not rebuild here: the emulator (VS Code / xterm.js)
                    // rewraps the scrollback asynchronously after the resize
                    // events, so a DSR cursor query issued now can return the
                    // pre-reflow row. Ignore delayed same-size notifications and
                    // defer a real resize until foreground work needs a redraw.
                    update_pending_resize_rebuild(
                        &mut pending_resize_rebuild,
                        last_applied_terminal_size,
                        Size::new(width, height),
                    );
                    continue;
                }
                if pending_resize_rebuild {
                    pending_resize_rebuild = false;
                    let rebuilt_area = rebuild_after_terminal_reflow(
                        &mut terminal,
                        &mut screen,
                        base_viewport_height,
                        fitted_completion_items,
                        last_drawn_area,
                    )?;
                    last_drawn_area = Some(rebuilt_area);
                    last_applied_terminal_size = terminal.backend().size()?;
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
            clear_fixed_viewport(&mut terminal, last_drawn_area.map(|area| area.y))
                .unwrap_or(false);
        drop(terminal);
        if !cleared_viewport {
            let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        }
        self.cursor_position_queries_disabled = screen.queries_disabled;
        // Carry the unaccounted replies into later prompts: an orphan ESC only
        // surfaces on the next standalone Escape, which a non-Escape submission
        // (F2 / Alt+Enter) can precede.
        self.pending_cpr_replies = std::mem::take(&mut screen.pending_cpr_replies);
        drop(screen);
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
        ViewportRebuildMode, clear_fixed_viewport, clear_row_range, fixed_viewport_area,
        force_frame_repaint, multiline_viewport_height, park_reflow_anchor, parked_anchor_offset,
        prepare_fixed_viewport, submitted_input_preview_lines, take_redraw_request,
        take_standalone_resize_rebuild, terminal_with_fixed_viewport,
        update_pending_resize_rebuild, viewport_height_with_completion,
    };

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn split_cursor_reply_is_consumed_without_losing_resize_or_following_text() {
        let mut input = VecDeque::from([key(KeyCode::Esc), Event::Resize(90, 30)]);
        input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        input.push_back(key(KeyCode::Char('x')));
        let mut pending = VecDeque::new();
        let mut read = |_| Ok(input.pop_front());
        let mut replies = VecDeque::from([std::time::Instant::now()]);
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            None
        );
        assert!(
            replies.is_empty(),
            "consuming the orphan reply must account for the timed-out query"
        );
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            Some(Event::Resize(90, 30))
        );
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            Some(key(KeyCode::Char('x')))
        );
    }

    #[test]
    fn consumed_orphan_reply_keeps_later_escape_immediate() {
        // The orphan ESC plus its fragmented late CPR is consumed first ...
        let mut input = VecDeque::new();
        input.push_back(key(KeyCode::Esc));
        input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        // ... then a real standalone Escape (the submit key) must go through.
        input.push_back(key(KeyCode::Esc));
        let mut pending = VecDeque::new();
        let mut replies = VecDeque::from([std::time::Instant::now()]);
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, |_| Ok(input.pop_front()))
                .unwrap(),
            None
        );
        assert!(
            replies.is_empty(),
            "the orphan reply is accounted for exactly once"
        );
        // No 250 ms lookahead may delay this real submission.
        let mut calls = 0;
        let event = super::read_prompt_event(&mut pending, &mut replies, |_| {
            calls += 1;
            Ok(input.pop_front())
        })
        .unwrap();
        assert_eq!(event, Some(key(KeyCode::Esc)));
        assert_eq!(calls, 1);
    }

    #[test]
    fn cursor_reply_guard_preserves_standalone_escape_and_invalid_lookahead() {
        for tail in ["", "x", "[1x", "[13;", "[;1R", "[123456;1R"] {
            let original = std::iter::once(key(KeyCode::Esc))
                .chain(tail.chars().map(|ch| key(KeyCode::Char(ch))))
                .collect::<VecDeque<_>>();
            let mut input = original.clone();
            let mut pending = VecDeque::new();
            let mut output = VecDeque::new();
            let mut replies = VecDeque::from([std::time::Instant::now()]);
            while !input.is_empty() || !pending.is_empty() {
                if let Some(event) =
                    super::read_prompt_event(&mut pending, &mut replies, |_| Ok(input.pop_front()))
                        .unwrap()
                {
                    output.push_back(event);
                }
            }
            assert_eq!(output, original, "tail={tail:?}");
        }
    }

    #[test]
    fn cursor_reply_guard_leaves_literal_text_paste_and_control_keys_unchanged() {
        let original = "[13;1R"
            .chars()
            .map(|ch| key(KeyCode::Char(ch)))
            .chain([
                Event::Paste("\x1b[13;1R".into()),
                Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
                key(KeyCode::Backspace),
                key(KeyCode::Enter),
                key(KeyCode::Left),
            ])
            .collect::<VecDeque<_>>();
        let mut input = original.clone();
        let mut pending = VecDeque::new();
        let mut output = VecDeque::new();
        let mut replies = VecDeque::from([std::time::Instant::now()]);
        while !input.is_empty() {
            output.push_back(
                super::read_prompt_event(&mut pending, &mut replies, |_| Ok(input.pop_front()))
                    .unwrap()
                    .unwrap(),
            );
        }
        assert_eq!(output, original);
        assert!(pending.is_empty());
        assert_eq!(
            replies.len(),
            1,
            "ordinary input must not consume an unaccounted reply"
        );
    }

    #[test]
    fn healthy_terminal_escape_is_not_delayed_or_read_ahead() {
        let mut calls = 0;
        let mut replies = VecDeque::new();
        let event = super::read_prompt_event(&mut VecDeque::new(), &mut replies, |_| {
            calls += 1;
            Ok(Some(key(KeyCode::Esc)))
        })
        .unwrap();
        assert_eq!(event, Some(key(KeyCode::Esc)));
        assert_eq!(calls, 1);
    }

    #[test]
    fn every_timed_out_query_swallows_its_own_orphan_reply() {
        // A retried query (or a failed recovery probe) leaves its own orphan in
        // the input stream; each must be swallowed, and the real Escape that
        // follows them must still submit without another grace.
        let mut input = VecDeque::new();
        for _ in 0..2 {
            input.push_back(key(KeyCode::Esc));
            input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        }
        input.push_back(key(KeyCode::Esc));
        let mut pending = VecDeque::new();
        let mut replies = VecDeque::from([std::time::Instant::now(), std::time::Instant::now()]);
        for _ in 0..2 {
            assert_eq!(
                super::read_prompt_event(&mut pending, &mut replies, |_| Ok(input.pop_front()))
                    .unwrap(),
                None
            );
        }
        assert!(replies.is_empty(), "both orphans were accounted for");
        let mut calls = 0;
        let event = super::read_prompt_event(&mut pending, &mut replies, |_| {
            calls += 1;
            Ok(input.pop_front())
        })
        .unwrap();
        assert_eq!(event, Some(key(KeyCode::Esc)));
        assert_eq!(calls, 1, "later Escape submissions are not deferred again");
    }

    #[test]
    fn a_slow_reply_is_still_swallowed_after_a_real_escape_was_replayed() {
        // The grace can expire before a fragmented reply's tail arrives (the gap
        // is what the empty second read simulates). The entry stays queued, so
        // the orphan cannot land in the box as literal text.
        let mut replies = VecDeque::from([std::time::Instant::now()]);
        let mut input = VecDeque::from([key(KeyCode::Esc), key(KeyCode::Esc)]);
        input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        let mut reads = 0usize;
        let mut read = |_: std::time::Duration| {
            reads += 1;
            // The second read is the lookahead behind the real Escape: it finds
            // no tail within the grace.
            if reads == 2 {
                return Ok(None);
            }
            Ok(input.pop_front())
        };
        assert_eq!(
            super::read_prompt_event(&mut VecDeque::new(), &mut replies, &mut read).unwrap(),
            Some(key(KeyCode::Esc))
        );
        assert_eq!(replies.len(), 1, "an unaccounted reply stays queued");
        assert_eq!(
            super::read_prompt_event(&mut VecDeque::new(), &mut replies, &mut read).unwrap(),
            None,
            "the late reply is still consumed"
        );
        assert!(replies.is_empty());
    }

    #[test]
    fn a_lost_reply_stops_deferring_escapes_after_the_window() {
        // A reply the connection dropped must not tax every later Escape: the
        // entry expires and the Escape goes through immediately.
        let expired = std::time::Instant::now()
            .checked_sub(super::CPR_REPLY_WINDOW)
            .expect("the test process must be older than the reply window");
        let mut replies = VecDeque::from([expired]);
        let mut calls = 0;
        let event = super::read_prompt_event(&mut VecDeque::new(), &mut replies, |_| {
            calls += 1;
            Ok(Some(key(KeyCode::Esc)))
        })
        .unwrap();
        assert_eq!(event, Some(key(KeyCode::Esc)));
        assert_eq!(calls, 1, "an expired reply must not delay the Escape");
        assert!(replies.is_empty(), "expired replies are dropped");
    }

    #[test]
    fn recovery_probe_backoff_grows_and_saturates() {
        let seconds: Vec<u64> = (0..5)
            .map(|failures| super::recovery_probe_delay(failures).as_secs())
            .collect();
        assert_eq!(
            seconds,
            vec![1, 5, 15, 30, 30],
            "a dead link is probed at a steady interval, not on every frame"
        );
    }

    #[test]
    fn recovery_probe_waits_out_fresh_replies_and_its_backoff() {
        let start = std::time::Instant::now();
        let mut screen = super::PromptScreen::default();
        assert!(
            !screen.recovery_probe_due(start),
            "an inline screen must not probe"
        );
        screen.note_timed_out_queries(1, start);
        assert!(screen.queries_disabled);
        assert_eq!(screen.pending_cpr_replies.len(), 1);
        // The reply may still be in flight and would answer the probe instead.
        assert!(!screen.recovery_probe_due(start + std::time::Duration::from_millis(500)));
        assert!(screen.recovery_probe_due(start + std::time::Duration::from_secs(2)));
        // A failed probe backs off for five seconds before the next attempt.
        screen.note_timed_out_queries(1, start + std::time::Duration::from_secs(2));
        assert!(!screen.recovery_probe_due(start + std::time::Duration::from_secs(5)));
        assert!(screen.recovery_probe_due(start + std::time::Duration::from_secs(7)));
        screen.note_recovered();
        assert_eq!(screen.recovery_failures, 0);
        assert!(screen.next_recovery_probe.is_none());
    }

    #[test]
    fn cursor_reply_tail_accepts_only_bounded_cpr_coordinates() {
        for prefix in ["[", "[1", "[13;", "[13;1"] {
            assert_eq!(super::cursor_reply_tail(prefix), Some(false));
        }
        for reply in ["[13;1R", "[65535;65535R"] {
            assert_eq!(super::cursor_reply_tail(reply), Some(true));
        }
        for invalid in [
            "",
            "13;1R",
            "[R",
            "[1R",
            "[;1R",
            "[1;R",
            "[1;2;3R",
            "[1;2x",
            "[123456;1R",
        ] {
            assert_eq!(super::cursor_reply_tail(invalid), None, "{invalid}");
        }
    }

    #[test]
    fn query_free_viewport_bootstrap_and_resize_never_query_or_scroll() {
        let mut output = Vec::new();
        {
            let mut backend = ratatui::backend::CrosstermBackend::new(&mut output);
            let area = super::prepare_query_free_viewport(
                &mut backend,
                ratatui::layout::Size::new(80, 24),
                5,
                &[],
            )
            .unwrap();
            // Bottom-anchored: the mirrored output tail sits directly above.
            assert_eq!(area, Rect::new(0, 19, 80, 5));
            let mut terminal = terminal_with_fixed_viewport(backend, area).unwrap();

            // An alternate-screen resize has no transcript to re-anchor and
            // must not send another DSR after a previous query timed out.
            let resized = super::prepare_query_free_viewport(
                terminal.backend_mut(),
                ratatui::layout::Size::new(60, 4),
                7,
                &[],
            )
            .unwrap();
            assert_eq!(resized, Rect::new(0, 0, 60, 4));
            terminal.resize(resized).unwrap();
            terminal
                .draw(|frame| frame.render_widget(Paragraph::new("ready"), frame.area()))
                .unwrap();
        }
        assert!(!output.windows(4).any(|bytes| bytes == b"\x1b[6n"));
        assert!(!output.contains(&b'\n'));
    }

    #[test]
    fn query_free_viewport_clamps_height_on_small_terminals() {
        let mut backend = TestBackend::new(10, 3);
        for (requested_height, expected_height) in [(0, 1), (2, 2), (8, 3)] {
            let area = super::prepare_query_free_viewport(
                &mut backend,
                ratatui::layout::Size::new(10, 3),
                requested_height,
                &[],
            )
            .unwrap();
            let expected_top = 3 - expected_height;
            assert_eq!(area, Rect::new(0, expected_top, 10, expected_height));
            assert_eq!(
                backend.get_cursor_position().unwrap(),
                Position::new(0, expected_top)
            );
        }
    }

    #[test]
    fn query_free_viewport_mirrors_output_tail_above_the_box() {
        // A timed-out cursor query drops the editor onto the alternate screen
        // with the transcript hidden; the newest rows of the previous answer are
        // mirrored directly above the box instead.
        let mut backend = TestBackend::new(20, 10);
        let tail = vec![
            "first line".to_string(),
            "second line".to_string(),
            "third line".to_string(),
        ];
        let area = super::prepare_query_free_viewport(
            &mut backend,
            ratatui::layout::Size::new(20, 10),
            4,
            &tail,
        )
        .unwrap();

        assert_eq!(area, Rect::new(0, 6, 20, 4));
        let buffer = backend.buffer();
        let row_text = |y: u16| {
            (0..20u16)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        assert_eq!(row_text(3), "first line");
        assert_eq!(row_text(4), "second line");
        assert_eq!(row_text(5), "third line");
        // Box rows stay empty: the mirror never draws inside the viewport.
        assert!(row_text(6).is_empty());
    }

    #[test]
    fn alternate_tail_rows_wrap_and_keep_the_newest_rows() {
        let tail = vec!["abcdefghij".to_string(), "中文测试".to_string()];
        // "abcd"/"efgh"/"ij" then "中文"/"测试": only the newest three rows fit.
        assert_eq!(
            super::alternate_tail_rows(&tail, 4, 3),
            vec!["ij".to_string(), "中文".to_string(), "测试".to_string()]
        );
        assert!(super::alternate_tail_rows(&tail, 0, 3).is_empty());
        assert!(super::alternate_tail_rows(&tail, 4, 0).is_empty());
    }

    #[test]
    fn pending_resize_waits_for_foreground_redraw() {
        let mut redraw_requested = true;

        assert!(take_redraw_request(&mut redraw_requested, false));
        assert!(!take_redraw_request(&mut redraw_requested, false));

        let mut pending_resize_rebuild = true;
        // An idle poll timeout leaves both states unchanged. A later foreground
        // redraw consumes the pending resize exactly once.
        assert!(!take_redraw_request(&mut redraw_requested, false));
        assert!(pending_resize_rebuild);
        redraw_requested = true;
        assert!(take_redraw_request(&mut redraw_requested, false));
        assert!(take_standalone_resize_rebuild(
            &mut pending_resize_rebuild,
            false
        ));
        assert!(!pending_resize_rebuild);
        assert!(!take_redraw_request(&mut redraw_requested, false));
    }

    #[test]
    fn duplicate_resize_notification_does_not_schedule_rebuild() {
        let applied = ratatui::layout::Size::new(80, 24);
        let mut pending = false;

        update_pending_resize_rebuild(&mut pending, applied, applied);
        assert!(!pending);

        update_pending_resize_rebuild(&mut pending, applied, ratatui::layout::Size::new(79, 24));
        assert!(pending);

        // A resize burst that returns to the applied dimensions before the
        // rebuild must cancel the stale intermediate notification.
        update_pending_resize_rebuild(&mut pending, applied, applied);
        assert!(!pending);
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
        let marker = crate::ai::theme::current().accent_success;
        let body = crate::ai::theme::current().accent_submitted;
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
