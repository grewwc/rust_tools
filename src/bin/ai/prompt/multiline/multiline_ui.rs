use std::{
    cell::RefCell,
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
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
        size as terminal_size,
    },
};
use ratatui::{
    Terminal,
    backend::{Backend, ClearType as BackendClearType, CrosstermBackend},
    buffer::{Cell, CellDiffOption},
    layout::{Position, Rect, Size},
    style::{Color, Modifier, Style},
};
use tui_textarea::TextArea;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

#[cfg(test)]
use super::render::{preserve_prompt_cursor_group, render_multiline_popup};
use super::{
    MultilineHistoryState, SingleRowFrame,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup_with_checkpoint,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};
use crate::commonw::prompt::acquire_foreground_stdin;

/// Maximum viewport height (textarea + chrome); scales with the terminal size,
/// capped at 11 lines.
const MAX_VIEWPORT_HEIGHT: u16 = 11;
/// Upper bound on textarea lines (a comfortable value on large terminals).
const MAX_TEXTAREA_LINES: u16 = 7;
/// Fixed chrome line count for the normal editing state: model(1) + title(1)
/// + help(1).
/// No decorative divider is drawn anymore, so stray horizontal lines do not
/// pile up after terminal resizes.
const VIEWPORT_CHROME_LINES: u16 = 3;
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

/// Keep growth while a draft remains, but release its extra rows when it is
/// completely deleted. Blank logical lines still count as draft content.
fn content_viewport_height(terminal_rows: u16, current_height: u16, lines: &[String]) -> u16 {
    if lines.len() <= 1 && lines.first().is_none_or(String::is_empty) {
        return multiline_viewport_height(terminal_rows, None);
    }
    let available = terminal_rows.saturating_sub(2).max(1);
    let needed = u16::try_from(lines.len())
        .unwrap_or(u16::MAX)
        .saturating_add(VIEWPORT_CHROME_LINES)
        .min(MAX_VIEWPORT_HEIGHT)
        .min(available);
    current_height.max(needed)
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
/// Small idle window used to collect text bytes that an SSH terminal exposes as
/// individual key events after bracketed paste is disabled.
const SSH_INPUT_BATCH_IDLE: Duration = Duration::from_millis(2);
/// Bound one batch so ordinary typing and control events stay responsive.
const SSH_INPUT_BATCH_WINDOW: Duration = Duration::from_millis(12);
const MAX_SSH_INPUT_BATCH_EVENTS: usize = 1024;

/// A main-screen probe is only safe once no *recent* reply is outstanding: an
/// older reply would already have been delivered, so an answer arriving now is
/// the probe's own.
const CPR_PROBE_QUIET_PERIOD: Duration = Duration::from_secs(2);

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

fn is_ssh_text_input_event(event: &Event) -> bool {
    // Enter and Tab can submit or change completion state, so they are boundaries.
    matches!(
        event,
        Event::Key(key)
            if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                && (key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT)
                && matches!(key.code, KeyCode::Char(_))
    )
}

fn read_ssh_input_batch(
    first: Event,
    is_ssh: bool,
    pending: &mut VecDeque<Event>,
    pending_cpr_replies: &mut VecDeque<Instant>,
    mut read: impl FnMut(Duration) -> io::Result<Option<Event>>,
) -> io::Result<Vec<Event>> {
    let mut batch = vec![first];
    if !is_ssh || !is_ssh_text_input_event(&batch[0]) {
        return Ok(batch);
    }
    prune_expired_cpr_replies(pending_cpr_replies, Instant::now());
    // Leave ambiguous Escapes to the regular reader while CPRs are outstanding.
    // Its fragmented-reply grace must never extend the batch deadline.
    if !pending_cpr_replies.is_empty() {
        return Ok(batch);
    }

    let deadline = Instant::now() + SSH_INPUT_BATCH_WINDOW;
    while batch.len() < MAX_SSH_INPUT_BATCH_EVENTS {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        let timeout = remaining.min(SSH_INPUT_BATCH_IDLE);
        // No cursor query is issued during batching, so no CPR lookahead is needed.
        let event = match pending.pop_front() {
            Some(event) => Some(event),
            None => read(timeout)?,
        };
        let Some(event) = event else {
            break;
        };
        if !is_ssh_text_input_event(&event) {
            pending.push_front(event);
            break;
        }
        batch.push(event);
    }
    Ok(batch)
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
/// keeps the main transcript intact. Once degraded, the active editor stays on
/// that query-free screen; recovery is attempted only at the next prompt start,
/// after cleanup has restored the main screen. Query disabling survives prompt
/// sessions: a late reply has no request id and must not anchor a later inline
/// viewport.
#[derive(Default)]
struct PromptScreen {
    queries_disabled: bool,
    alternate: bool,
    /// Completion time of every query whose reply is still unaccounted for,
    /// oldest first. Feeds the orphan-Escape guard in `read_prompt_event`, and holds
    /// recovery probes back while a fresh reply could be mistaken for theirs.
    pending_cpr_replies: VecDeque<Instant>,
    /// One inline retry per prompt session: a rebuild must not block on two
    /// two-second waits per frame once the link really is down.
    retry_used: bool,
    /// Tail of the last model output, repainted above the box on the alternate
    /// screen (`PromptEditor::set_alternate_screen_tail`).
    tail: Vec<String>,
    /// Width of every drawn box row, top row first, as the terminal saw it when
    /// the row was written; empty means nothing has been recorded.
    ///
    /// A width reflow splits each screen row by its written content, so the rows
    /// the re-wrap inserts above the box depend on how far each drawn row was
    /// painted and not on the box width: rows the app never wrote (blank padding
    /// rows inside the viewport) re-wrap into exactly one row. Widths accumulate
    /// because the terminal keeps cells written by earlier frames, and
    /// `prepare_viewport` drops the record whenever the box's rows are cleared.
    /// While the record is empty, the cursor remains at the cleared box top;
    /// only a painted frame parks it at the bottom.
    box_row_widths: Vec<u16>,
    /// A painted prompt linked into the cursor's logical line group does not
    /// gain terminal-owned wraps. Keep its original bottom offset separately
    /// from the width record used to erase shortened input.
    cursor_group_height: Option<u16>,
    /// Geometry of the last cleared or painted viewport. Stored tops may only
    /// clear the main screen while this geometry still matches the terminal.
    viewport_geometry: Option<(Rect, Size)>,
}

impl PromptScreen {
    /// Recovery may re-enable queries only before the next prompt is drawn.
    /// Never leave a usable alternate-screen editor for a blocking DSR probe.
    fn can_recover_at_prompt_start(&mut self, now: Instant) -> bool {
        self.queries_disabled && !self.alternate && self.replies_settled(now)
    }

    /// True while no *recent* reply is outstanding, i.e. a query answer arriving
    /// now belongs to that query. An older outstanding reply was either already
    /// delivered or lost with the link, so it no longer competes.
    fn replies_settled(&mut self, now: Instant) -> bool {
        prune_expired_cpr_replies(&mut self.pending_cpr_replies, now);
        self.pending_cpr_replies
            .back()
            .is_none_or(|query| now.duration_since(*query) >= CPR_PROBE_QUIET_PERIOD)
    }

    fn prepare_viewport(
        &mut self,
        backend: &mut CrosstermBackend<io::Stdout>,
        terminal_size: Size,
        requested_height: u16,
        cursor_offset_row: u16,
        mode: ViewportRebuildMode,
        clear_existing_viewport: bool,
        previous_area: Option<Rect>,
        verify_anchor: bool,
    ) -> io::Result<(Rect, Size)> {
        let now = Instant::now();
        if !self.queries_disabled && self.replies_settled(now) {
            match prepare_anchored_viewport(
                backend,
                terminal_size,
                requested_height,
                cursor_offset_row,
                mode,
                clear_existing_viewport,
                previous_area,
                &self.box_row_widths,
                self.cursor_group_height,
                verify_anchor,
                &mut self.pending_cpr_replies,
            ) {
                Ok(prepared) => {
                    if clear_existing_viewport {
                        self.box_row_widths.clear();
                        self.cursor_group_height = None;
                    }
                    self.viewport_geometry = Some(prepared);
                    return Ok(prepared);
                }
                Err(err) if PromptEditor::is_cursor_position_timeout(&err) => {
                    // A stalled round-trip is usually a transient link hiccup, so
                    // retry once before paying for the alternate screen. Fresh
                    // anchors and deferred rebuilds verify the reply stream
                    // again; the other rebuilds query the same parked row. A
                    // reflow invalidates the parked row, so retry only while the
                    // terminal size is unchanged. Both attempts share the same
                    // outstanding-reply accounting.
                    if !self.retry_used && backend.size().ok() == Some(terminal_size) {
                        self.retry_used = true;
                        match prepare_anchored_viewport(
                            backend,
                            terminal_size,
                            requested_height,
                            cursor_offset_row,
                            mode,
                            clear_existing_viewport,
                            previous_area,
                            &self.box_row_widths,
                            self.cursor_group_height,
                            verify_anchor,
                            &mut self.pending_cpr_replies,
                        ) {
                            Ok(prepared) => {
                                if clear_existing_viewport {
                                    self.box_row_widths.clear();
                                    self.cursor_group_height = None;
                                }
                                self.viewport_geometry = Some(prepared);
                                return Ok(prepared);
                            }
                            Err(err) if PromptEditor::is_cursor_position_timeout(&err) => {
                                self.queries_disabled = true;
                            }
                            Err(err) => return Err(err),
                        }
                    } else {
                        self.queries_disabled = true;
                    }
                }
                Err(err) if previous_area.is_none() && is_unstable_prompt_geometry(&err) => {
                    // A new prompt has no inline frame to preserve. Bootstrap
                    // on the known alternate-screen coordinates after the limit.
                    self.queries_disabled = true;
                }
                Err(err) => return Err(err),
            }
        }
        if !self.alternate {
            // Clear the old inline box only while its stored geometry is still
            // valid. A failed query cannot locate its top after another resize.
            let live_size = backend.size()?;
            if let Some((area, size)) = self.viewport_geometry
                && Some(area) == previous_area
                && size == live_size
            {
                backend.set_cursor_position(Position::new(0, area.y))?;
                backend.clear_region(BackendClearType::AfterCursor)?;
                backend.flush()?;
            }
            // Set this before writing so cleanup also runs on a partial write.
            self.alternate = true;
            execute!(io::stdout(), EnterAlternateScreen)?;
        }
        // A recent reply can also keep the inline path out without a timeout.
        // Keep every rebuild query-free for the rest of this prompt in either
        // case; the next prompt start can reconsider recovery after cleanup.
        self.queries_disabled = true;
        let terminal_size = backend.size()?;
        let area =
            prepare_query_free_viewport(backend, terminal_size, requested_height, &self.tail)?;
        self.box_row_widths.clear();
        self.cursor_group_height = None;
        self.viewport_geometry = Some((area, terminal_size));
        Ok((area, terminal_size))
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
    let rows = alternate_tail_rows(
        tail,
        width,
        bottom_row.min(MAX_ALTERNATE_TAIL_ROWS) as usize,
    );
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

/// Each query emits one CPR request and a successful read consumes one reply,
/// possibly an older one. Preserve the net outstanding count across retries,
/// replacing consumed old entries with the new query's timestamp. A failed
/// query conservatively retains its entry even if writing the request failed.
fn tracked_cursor_query<T, E>(
    pending: &mut VecDeque<Instant>,
    query: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let result = query();
    pending.push_back(Instant::now());
    if result.is_ok() {
        pending.pop_front();
    }
    while pending.len() > MAX_PENDING_CPR_REPLIES {
        pending.pop_front();
    }
    result
}

/// Probe round trips a fresh anchor may spend before giving up.
const VERIFIED_ANCHOR_ATTEMPTS: u8 = 3;

/// The two rows the next anchor probe commands for itself.
///
/// Both rows slide by one per probe and do not repeat within a screen height,
/// because the probe accepts a pair of replies as its own: a pair left unread by
/// an earlier probe commanded the same two rows under a fixed pair, and the
/// reply read after it would then be that probe's pinned answer — a row the
/// cursor was moved to, not the row it lives on. Anchoring the box on that row
/// is what leaves blank rows above the box, or clears printed output when the
/// stale row is higher up the screen.
///
/// The counter is a function-local static because it belongs to this one probe;
/// no other module state should have to know about it.
fn anchor_probe_rows(last_screen_row: u16) -> (u16, u16) {
    use std::sync::atomic::{AtomicU16, Ordering};

    static ANCHOR_PROBE_SEQUENCE: AtomicU16 = AtomicU16::new(0);

    // `low` stays below `last_screen_row` so `high = low + 1` never leaves the
    // screen; `last_screen_row == 0` is handled by the caller before this runs.
    let span = u32::from(last_screen_row);
    let low = (u32::from(ANCHOR_PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed)) % span) as u16;
    (low, low + 1)
}

/// Reads the cursor row in a way that a reply left over from an earlier query
/// cannot spoof.
///
/// `ESC[6n` replies carry no tag, so a single answer cannot be told apart from
/// a stale one. This protocol instead pins the cursor to two rows it commands
/// itself — rows no earlier probe commanded — and requires both replies to land
/// exactly on those rows; only then has every earlier reply been read, and the
/// following query is trusted.
/// The caller's cursor is put back with `restore` on every exit path, so the
/// probes never move where the box is drawn afterwards.
fn verified_cursor_row(
    query_row: &mut impl FnMut() -> io::Result<u16>,
    save: &mut impl FnMut() -> io::Result<()>,
    restore: &mut impl FnMut() -> io::Result<()>,
    set_row: &mut impl FnMut(u16) -> io::Result<()>,
    last_screen_row: u16,
    pending: &mut VecDeque<Instant>,
) -> io::Result<u16> {
    if last_screen_row == 0 {
        // Both probes would command the same row, so every reply matches and
        // proves nothing; a single-row screen also has no row a stale reply
        // could get wrong.
        return tracked_cursor_query(pending, query_row);
    }
    for _ in 0..VERIFIED_ANCHOR_ATTEMPTS {
        save()?;
        let (low_row, high_row) = anchor_probe_rows(last_screen_row);
        // Both probe answers are collected before anything is propagated, so a
        // query that times out still reaches `restore` below instead of leaving
        // the caller's cursor parked on a probe row.
        let probe = (|| -> io::Result<(u16, u16)> {
            set_row(low_row)?;
            let low = tracked_cursor_query(pending, &mut *query_row)?;
            set_row(high_row)?;
            let high = tracked_cursor_query(pending, &mut *query_row)?;
            Ok((low, high))
        })();
        let restored = restore();
        let (low, high) = match probe {
            Ok(rows) => {
                restored?;
                rows
            }
            // The probe already failed with the error the caller knows how to
            // handle (a DSR timeout). Reporting a restore error instead would
            // turn that into a hard failure.
            Err(err) => return Err(err),
        };
        if low == low_row && high == high_row {
            // Nothing older than the two probes answered them, so the reply
            // stream is drained. Retire any old accounting before the final
            // query, whose own reply must remain guarded if it times out.
            pending.clear();
            return tracked_cursor_query(pending, query_row);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "The cursor position could not be read within a normal duration",
    ))
}

/// Verify replies without taking the cursor outside its protected prompt group.
/// An absolute probe elsewhere on the screen would expose the whole group to
/// reflow if a resize landed during the round trip. Relative probes stay inside
/// the old prompt even when transcript reflow moves its screen coordinates.
fn verified_prompt_cursor_row(
    query_row: &mut impl FnMut() -> io::Result<u16>,
    save: &mut impl FnMut() -> io::Result<()>,
    restore: &mut impl FnMut() -> io::Result<()>,
    move_up: &mut impl FnMut(u16) -> io::Result<()>,
    move_down: &mut impl FnMut(u16) -> io::Result<()>,
    height: u16,
    pending: &mut VecDeque<Instant>,
) -> io::Result<u16> {
    use std::sync::atomic::{AtomicU16, Ordering};

    static PROMPT_PROBE_SEQUENCE: AtomicU16 = AtomicU16::new(0);
    if height < 2 {
        return tracked_cursor_query(pending, query_row);
    }
    for _ in 0..VERIFIED_ANCHOR_ATTEMPTS {
        save()?;
        let offset = 1 + PROMPT_PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed) % (height - 1);
        let probe = (|| -> io::Result<(u16, u16, u16)> {
            let bottom = tracked_cursor_query(pending, &mut *query_row)?;
            move_up(offset)?;
            let low = tracked_cursor_query(pending, &mut *query_row)?;
            move_down(1)?;
            let high = tracked_cursor_query(pending, &mut *query_row)?;
            Ok((bottom, low, high))
        })();
        let restored = restore();
        let (bottom, low, high) = match probe {
            Ok(rows) => {
                restored?;
                rows
            }
            Err(err) => return Err(err),
        };
        if bottom >= offset && low == bottom - offset && high == low + 1 {
            pending.clear();
            return tracked_cursor_query(pending, query_row);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "The cursor position could not be read within a normal duration",
    ))
}

#[cfg(test)]
fn prepare_fixed_viewport<B: Backend>(
    backend: &mut B,
    terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
    pending: &mut VecDeque<Instant>,
) -> Result<Rect, B::Error> {
    // One synchronous DSR query is the authoritative cursor position for this
    // rebuild. A fresh anchor does not trust a single reply and verifies the
    // reply stream first (see `prepare_anchored_viewport`); height rebuilds and
    // the keystroke-driven reflow rebuild keep the one round trip because the
    // frame that follows is already on its way to the user.
    let cursor_position = tracked_cursor_query(pending, || backend.get_cursor_position())?;
    prepare_fixed_viewport_at(
        backend,
        terminal_size,
        requested_height,
        cursor_offset_row,
        mode,
        clear_existing_viewport,
        cursor_position,
    )
}

/// Bound cursor-query and pre-paint retries so input polling remains reachable.
const PROMPT_GEOMETRY_RETRIES: usize = 3;

#[derive(Debug)]
struct UnstablePromptGeometry;

impl std::fmt::Display for UnstablePromptGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Terminal geometry changed while preparing the prompt")
    }
}

impl std::error::Error for UnstablePromptGeometry {}

fn unstable_prompt_geometry() -> io::Error {
    io::Error::new(io::ErrorKind::WouldBlock, UnstablePromptGeometry)
}

fn is_unstable_prompt_geometry(err: &io::Error) -> bool {
    err.get_ref()
        .is_some_and(|source| source.is::<UnstablePromptGeometry>())
}

fn defer_unstable_prompt_geometry<T>(result: io::Result<T>) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) if is_unstable_prompt_geometry(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Validate the cursor reply's geometry before computing an offset or clearing.
/// A resize during the query keeps the old frame intact and queries again at
/// the new size. Both resize passes use this seam and the same old width record.
#[cfg(test)]
fn prepare_fixed_viewport_with_query<B: Backend>(
    backend: &mut B,
    terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
    previous_area: Option<Rect>,
    row_widths: &[u16],
    pending: &mut VecDeque<Instant>,
    query: impl FnMut(&mut B, Size, &mut VecDeque<Instant>) -> io::Result<Position>,
) -> io::Result<(Rect, Size)> {
    prepare_cursor_group_viewport_with_query(
        backend,
        terminal_size,
        requested_height,
        cursor_offset_row,
        mode,
        clear_existing_viewport,
        previous_area,
        row_widths,
        None,
        pending,
        query,
    )
}

fn prepare_cursor_group_viewport_with_query<B: Backend>(
    backend: &mut B,
    mut terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
    previous_area: Option<Rect>,
    row_widths: &[u16],
    cursor_group_height: Option<u16>,
    pending: &mut VecDeque<Instant>,
    mut query: impl FnMut(&mut B, Size, &mut VecDeque<Instant>) -> io::Result<Position>,
) -> io::Result<(Rect, Size)> {
    for _ in 0..PROMPT_GEOMETRY_RETRIES {
        let cursor_position = query(backend, terminal_size, pending)?;
        let live_size = backend
            .size()
            .map_err(|err| io::Error::other(err.to_string()))?;
        if live_size != terminal_size {
            terminal_size = live_size;
            continue;
        }
        // A painted frame parks at its bottom. A cleared frame keeps the cursor
        // at its top until paint, so another resize can move that top naturally.
        let cursor_offset_row = if previous_area.is_some() && row_widths.is_empty() {
            0
        } else if let Some(height) = cursor_group_height {
            height.saturating_sub(1)
        } else {
            previous_area.map_or(cursor_offset_row, |area| {
                reflowed_anchor_offset(
                    Some(area),
                    requested_height,
                    row_widths,
                    terminal_size.width,
                )
            })
        };
        let area = prepare_fixed_viewport_at(
            backend,
            terminal_size,
            requested_height,
            cursor_offset_row,
            mode,
            clear_existing_viewport,
            cursor_position,
        )
        .map_err(|err| io::Error::other(err.to_string()))?;
        return Ok((area, terminal_size));
    }
    Err(unstable_prompt_geometry())
}

/// Check again after clearing and immediately before painting fixed coordinates.
/// Each successful re-anchor commits its new size. Exhaustion defers this frame
/// to the input poll, keeping its last valid anchor and width record intact.
fn prepare_prompt_paint<T>(
    state: &mut T,
    applied_size: &mut Size,
    mut read_size: impl FnMut(&mut T) -> io::Result<Size>,
    mut reanchor: impl FnMut(&mut T) -> io::Result<Size>,
) -> io::Result<Size> {
    for _ in 0..PROMPT_GEOMETRY_RETRIES {
        let live_size = read_size(state)?;
        if live_size_matches_applied(live_size, *applied_size) {
            return Ok(live_size);
        }
        *applied_size = reanchor(state)?;
    }
    let live_size = read_size(state)?;
    if live_size_matches_applied(live_size, *applied_size) {
        return Ok(live_size);
    }
    Err(unstable_prompt_geometry())
}

/// Builds the fixed viewport for a cursor position the caller already read,
/// without querying the terminal itself.
fn prepare_fixed_viewport_at<B: Backend>(
    backend: &mut B,
    terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
    cursor_position: Position,
) -> Result<Rect, B::Error> {
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
    backend.set_cursor_position(Position::new(area.x, area.y))?;
    if clear_existing_viewport {
        // The box is always the last thing drawn, so every transcript row sits
        // ABOVE `area.y`: clearing from there to the end of the screen cannot
        // erase conversation history. This range is also the only one that
        // removes both ghost directions — a widening reflow moves the box up and
        // strands its old rows below the new bottom, while a narrowing reflow
        // pushes them further down. Bounding the clear to `area.height` rows
        // left exactly those stranded rows on screen.
        //
        // Width reflow callers include the old rows' extra wraps in the parked
        // offset. Without that correction this clear starts inside the old box
        // and leaves its first rows between the transcript and the new frame.
        backend.clear_region(BackendClearType::AfterCursor)?;
    }
    backend.flush()?;
    Ok(area)
}

/// Whether a rebuild must verify the DSR reply stream before trusting its row.
///
/// A fresh anchor is the box top itself, and the deferred reflow rebuild has no
/// frame waiting on it. Height rebuilds (which run on every panel change) and the
/// keystroke-driven reflow rebuild accept the single round trip.
fn reply_stream_is_verified(
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    verify_anchor: bool,
) -> bool {
    verify_anchor || (cursor_offset_row == 0 && mode == ViewportRebuildMode::ReserveMissingRows)
}

/// Runs the inline-anchor build, verifying the DSR reply stream first when this
/// is a fresh anchor, unless the caller accepts a single round trip.
///
/// A fresh anchor (`cursor_offset_row == 0`) is taken as the box top directly,
/// so a reply left over from an earlier query anchors the box on the row that
/// earlier query reported — in practice the previous prompt's parked bottom row
/// — and the exit clear then strands blank rows above the submitted preview. A
/// rebuild instead recovers its top as `parked row - offset`, where the same
/// stale reply only shifts the box inside rows that are already reserved, which
/// is why a rebuild normally keeps one round trip. `verify_anchor` buys the same
/// protection for a rebuild whose reply was parked by the box the terminal is
/// still re-wrapping.
fn prepare_anchored_viewport(
    backend: &mut CrosstermBackend<io::Stdout>,
    terminal_size: Size,
    requested_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    clear_existing_viewport: bool,
    previous_area: Option<Rect>,
    row_widths: &[u16],
    cursor_group_height: Option<u16>,
    verify_anchor: bool,
    pending: &mut VecDeque<Instant>,
) -> io::Result<(Rect, Size)> {
    let verify_reply = reply_stream_is_verified(cursor_offset_row, mode, verify_anchor);
    prepare_cursor_group_viewport_with_query(
        backend,
        terminal_size,
        requested_height,
        cursor_offset_row,
        mode,
        clear_existing_viewport,
        previous_area,
        row_widths,
        cursor_group_height,
        pending,
        |backend, size, pending| {
            query_anchored_cursor_position(
                backend,
                size,
                cursor_group_height,
                verify_reply,
                pending,
            )
        },
    )
}

fn query_anchored_cursor_position(
    backend: &mut CrosstermBackend<io::Stdout>,
    terminal_size: Size,
    cursor_group_height: Option<u16>,
    verify_reply: bool,
    pending: &mut VecDeque<Instant>,
) -> io::Result<Position> {
    if !verify_reply {
        return tracked_cursor_query(pending, || backend.get_cursor_position());
    }
    let last_screen_row = terminal_size.height.saturating_sub(1);
    // The verification closures outlive each other, so they share the caller's
    // backend through a `RefCell`; the reborrow inside the block keeps the
    // `&mut` usable for the draw that follows.
    let row = {
        let shared = RefCell::new(&mut *backend);
        let mut save = || -> io::Result<()> {
            let mut guard = shared.borrow_mut();
            execute!(&mut *guard, cursor::SavePosition)
        };
        let mut restore = || -> io::Result<()> {
            let mut guard = shared.borrow_mut();
            execute!(&mut *guard, cursor::RestorePosition)
        };
        let mut set_row = |row: u16| -> io::Result<()> {
            let mut guard = shared.borrow_mut();
            execute!(&mut *guard, cursor::MoveTo(0, row))
        };
        let mut query_row = || -> io::Result<u16> {
            let mut guard = shared.borrow_mut();
            let position = guard.get_cursor_position()?;
            Ok(position.y)
        };
        if let Some(height) = cursor_group_height {
            let mut move_up = |rows: u16| -> io::Result<()> {
                let mut guard = shared.borrow_mut();
                execute!(&mut *guard, cursor::MoveUp(rows))
            };
            let mut move_down = |rows: u16| -> io::Result<()> {
                let mut guard = shared.borrow_mut();
                execute!(&mut *guard, cursor::MoveDown(rows))
            };
            verified_prompt_cursor_row(
                &mut query_row,
                &mut save,
                &mut restore,
                &mut move_up,
                &mut move_down,
                height.min(terminal_size.height),
                pending,
            )?
        } else {
            verified_cursor_row(
                &mut query_row,
                &mut save,
                &mut restore,
                &mut set_row,
                last_screen_row,
                pending,
            )?
        }
    };
    Ok(Position::new(0, row))
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
///
/// Returns the terminal together with the screen size its anchor was computed
/// for; the caller records that size instead of a fresh `size()` read, so a
/// resize that landed during the build is not mistaken for one already applied.
fn build_fixed_terminal(
    height: u16,
    screen: &mut PromptScreen,
) -> io::Result<(MultilineTerminal, Size)> {
    // Cleanup has left the previous prompt's alternate screen. Retry only here,
    // before a new editor is visible, and only after recent replies settle.
    if screen.can_recover_at_prompt_start(Instant::now()) {
        screen.queries_disabled = false;
        // Recovery is already a second chance; keep its existing single attempt
        // instead of also spending the ordinary prompt's one inline retry.
        screen.retry_used = true;
    }
    let mut backend = CrosstermBackend::new(io::stdout());
    let terminal_size = backend.size()?;
    let (area, terminal_size) = screen.prepare_viewport(
        &mut backend,
        terminal_size,
        height,
        0,
        ViewportRebuildMode::ReserveMissingRows,
        false,
        None,
        false,
    )?;
    let terminal = terminal_with_fixed_viewport(backend, area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    Ok((terminal, terminal_size))
}

/// Blanks the rows in `[from_row, to_row)` one line at a time, leaving every
/// row outside that range untouched.
#[cfg(test)]
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
/// Clearing from the recovered top already blanks every released row. Keep the
/// cursor at that top until paint so a subsequent reflow cannot lose the anchor.
fn rebuild_fixed_viewport(
    terminal: &mut MultilineTerminal,
    screen: &mut PromptScreen,
    terminal_size: Size,
    new_height: u16,
    cursor_offset_row: u16,
    mode: ViewportRebuildMode,
    previous_area: Option<Rect>,
    verify_anchor: bool,
) -> io::Result<(Rect, Size)> {
    let (area, terminal_size) = screen.prepare_viewport(
        terminal.backend_mut(),
        terminal_size,
        new_height,
        cursor_offset_row,
        mode,
        true,
        previous_area,
        verify_anchor,
    )?;
    *terminal = terminal_with_fixed_viewport(CrosstermBackend::new(io::stdout()), area)
        .map_err(|err| io::Error::other(err.to_string()))?;
    Ok((area, terminal_size))
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
    verify_anchor: bool,
) -> io::Result<(Rect, Size)> {
    let terminal_size = terminal.backend().size()?;
    let requested_height = viewport_height_with_completion(
        terminal_size.height,
        base_viewport_height,
        fitted_completion_items,
    );
    // A painted prompt is one protected cursor group: xterm clips it at resize
    // and lets this replacement supply the ellipses. Its bottom offset remains
    // height - 1, so clearing cannot turn transient UI wraps into scrollback.
    // Preserve the old phase and width record until the query validates geometry;
    // an unprotected old frame still needs its full wrap correction before clear.
    let (rebuilt_area, terminal_size) = rebuild_fixed_viewport(
        terminal,
        screen,
        terminal_size,
        requested_height,
        parked_anchor_offset(last_drawn_area, requested_height),
        // Reflow moves an already-reserved viewport; it must never append more
        // terminal rows. Appending here makes every delayed resize notification
        // scroll the inline editor and strands the previously drawn caret.
        ViewportRebuildMode::ReflowOnly,
        last_drawn_area,
        verify_anchor,
    )?;
    // The cleared top follows transcript reflow until paint re-establishes the
    // protected bottom anchor. Record only the size validated for this phase.
    Ok((rebuilt_area, terminal_size))
}

/// Recover the first physical row of the old prompt from its parked cursor,
/// regardless of how many rows its replacement will paint.
fn reflowed_anchor_offset(
    last_drawn_area: Option<Rect>,
    requested_height: u16,
    row_widths: &[u16],
    new_width: u16,
) -> u16 {
    let extra_rows = last_drawn_area
        .map(|area| reflowed_extra_rows(area, row_widths, new_width))
        .unwrap_or(0);
    parked_anchor_offset(last_drawn_area, requested_height).saturating_add(extra_rows)
}

/// Extra rows between the old box's first row and its parked bottom-row anchor
/// after the terminal re-wraps it at `new_width`.
///
/// Every drawn row re-wraps into `ceil(painted_width / new_width)` rows, where
/// `painted_width` is how far the app actually wrote into that row
/// (`box_row_widths`) — a row the app never wrote stays exactly one row. The
/// bottom row is the exception: the anchor is at column zero, so that row's
/// trailing wraps are below the anchor and do not contribute to its offset.
///
/// Add this count to `height - 1` BEFORE the rebuild clears the old width
/// record. If the old top has already entered scrollback, the visible top
/// saturates at row zero; ordinary screen erasure cannot recover those hidden
/// input rows without also mutating the transcript's scrollback.
fn reflowed_extra_rows(previous_area: Rect, row_widths: &[u16], new_width: u16) -> u16 {
    // A row cannot be painted wider than the box it was drawn in, so nothing can
    // re-wrap once the box fits; a zero width would also divide by zero.
    if new_width == 0 || previous_area.width <= new_width {
        return 0;
    }
    let mut extra_rows = 0u16;
    // The bottom row's trailing wraps are below the column-zero anchor.
    for index in 0..previous_area.height.saturating_sub(1) {
        // An unknown row (never recorded, or blank) re-wraps into one row.
        let width = row_widths.get(index as usize).copied().unwrap_or(0);
        extra_rows = extra_rows.saturating_add(width.div_ceil(new_width).saturating_sub(1));
    }
    // Saturating: a 65535-row box whose rows nearly all re-wrap must not wrap the
    // counter around into an incorrect anchor offset.
    extra_rows
}

/// Width of each box row as the terminal sees it: the furthest column the frame
/// wrote in that row, counted from `area.left()`.
///
/// The furthest cell that counts as written wins — a symbol other than a space,
/// or any non-default style attribute. Styled blank cells count: the backend
/// emits them and the terminal extends the line's used length with them, so the
/// write boundary rather than the last visible glyph is what re-wraps. A row the
/// app never wrote stays 0, which `reflowed_extra_rows` treats as a single row.
fn painted_row_widths(buffer: &ratatui::buffer::Buffer, area: Rect) -> Vec<u16> {
    let mut widths = Vec::with_capacity(area.height as usize);
    for y in area.top()..area.bottom() {
        let mut width = 0;
        for x in (area.left()..area.right()).rev() {
            let Some(cell) = buffer.cell((x, y)) else {
                continue;
            };
            let written = cell.symbol() != " "
                || cell.fg != Color::Reset
                || cell.bg != Color::Reset
                || cell.modifier != Modifier::empty();
            if written {
                width = x.saturating_sub(area.left()).saturating_add(1);
                break;
            }
        }
        widths.push(width);
    }
    widths
}

/// Columns a forced repaint must rewrite in each row: every column that can
/// still carry ink there — the row's accumulated written width from earlier
/// frames, or this frame's own content, whichever reaches further.
///
/// `previous` is the widest write per row since the box's rows were last
/// erased, so cells beyond it are blank on the terminal: a reflow only moves
/// ink inside rows that a rebuild erases before the next forced repaint.
/// Writing the whole row instead makes the terminal count a short row as a
/// full-width line, and the next narrowing re-wraps rows that hold no ink
/// (see `reflowed_extra_rows`).
fn forced_repaint_widths(
    buffer: &ratatui::buffer::Buffer,
    area: Rect,
    previous: &[u16],
) -> Vec<u16> {
    painted_row_widths(buffer, area)
        .into_iter()
        .enumerate()
        .map(|(row, width)| {
            width
                .max(previous.get(row).copied().unwrap_or(0))
                .min(area.width)
        })
        .collect()
}

#[cfg(test)]
mod width_reflow_geometry_tests {
    use super::*;

    /// Four rows before the parked bottom row each gain one wrap. The old top
    /// is therefore eight rows above the anchor rather than four.
    #[test]
    fn narrowing_reflow_extends_the_old_anchor_offset() {
        let previous_area = Rect::new(0, 7, 40, 5);
        assert_eq!(reflowed_extra_rows(previous_area, &[40; 5], 20), 4);
    }

    /// Widening re-wraps nothing, so the box keeps the top the rebuild recovered.
    #[test]
    fn widening_reflow_adds_no_anchor_offset() {
        let previous_area = Rect::new(0, 3, 20, 5);
        assert_eq!(reflowed_extra_rows(previous_area, &[20; 5], 40), 0);
    }

    /// A height-only change keeps the width, so no drawn row can have been split.
    #[test]
    fn height_only_change_adds_no_anchor_offset() {
        let previous_area = Rect::new(0, 3, 40, 5);
        assert_eq!(reflowed_extra_rows(previous_area, &[40; 5], 40), 0);
    }

    /// Rows the app never wrote are blank on screen and re-wrap into ONE row.
    ///
    /// Three 40-column content rows and three blank rows become nine rows at
    /// width 20. Counting blank padding as full-width would clear transcript.
    #[test]
    fn blank_padding_rows_do_not_extend_the_anchor_offset() {
        let previous_area = Rect::new(0, 7, 40, 6);
        assert_eq!(
            reflowed_extra_rows(previous_area, &[40, 40, 40, 0, 0, 0], 20),
            3
        );
        // The 3 content rows alone re-wrap into 6 rows, two per drawn row, so the
        // rows above the bottom one insert `4 - 2 = 2` rows above the box.
        assert_eq!(
            reflowed_extra_rows(Rect::new(0, 7, 40, 3), &[40, 40, 40], 20),
            2
        );
    }

    /// An unknown record must not over-clear: every drawn row is then assumed to
    /// be a single re-wrapped row, which is what the box of blank rows already is.
    #[test]
    fn unknown_row_widths_add_no_anchor_offset() {
        let previous_area = Rect::new(0, 5, 40, 5);
        assert_eq!(reflowed_extra_rows(previous_area, &[], 20), 0);
    }

    /// The bottom row's trailing wraps are below the anchor. Counting them in
    /// its offset would clear one transcript row on every narrowing reflow.
    #[test]
    fn bottom_row_trailing_rows_do_not_extend_the_anchor_offset() {
        let previous_area = Rect::new(0, 7, 40, 5);
        let widths = [40; 5];
        let reflowed_rows = 10; // 5 drawn rows at 40 columns -> 2 screen rows each
        let last = 2; // the bottom drawn row itself re-wraps into 2 screen rows
        let rows_above_bottom_row = reflowed_rows - last;
        assert_eq!(
            reflowed_extra_rows(previous_area, &widths, 20),
            rows_above_bottom_row - (previous_area.height - 1)
        );
        assert_eq!(reflowed_extra_rows(previous_area, &widths, 20), 4);
    }

    /// Chrome wraps move the parked cursor even though the replacement frame
    /// truncates those rows. Count them to recover the first old prompt row;
    /// the former textarea-only offset left one blank row above the prompt.
    #[test]
    fn chrome_wraps_count_for_anchor_placement() {
        // Five-row box narrowed 40 -> 20: rows 1-2 are the (empty) textarea,
        // rows 3-4 are full-width chrome.
        let previous_area = Rect::new(0, 7, 40, 5);
        let widths = [0, 0, 0, 40, 40];
        assert_eq!(reflowed_extra_rows(previous_area, &widths, 20), 1);
        let offset = reflowed_anchor_offset(Some(previous_area), 5, &widths, 20);
        assert_eq!(offset, 5);
        let (area, scroll) = fixed_viewport_area(
            Size::new(20, 20),
            Position::new(0, 12),
            5,
            offset,
            ViewportRebuildMode::ReflowOnly,
        );
        assert_eq!(area, Rect::new(0, 7, 20, 5));
        assert_eq!(scroll, 0);
    }

    /// Textarea and chrome wraps both move the same parked anchor.
    #[test]
    fn textarea_wraps_still_place_the_box() {
        let previous_area = Rect::new(0, 7, 40, 5);
        let widths = [0, 40, 40, 40, 0];
        assert_eq!(
            reflowed_anchor_offset(Some(previous_area), 5, &widths, 20),
            7
        );
    }

    /// Changing the requested height does not change the old anchor offset.
    #[test]
    fn reflow_placement_uses_the_old_height_and_all_row_widths() {
        let previous_area = Rect::new(0, 7, 40, 5);
        let widths = [40, 0, 40, 0, 40];
        assert_eq!(
            reflowed_anchor_offset(Some(previous_area), 3, &widths, 20),
            previous_area.height - 1 + reflowed_extra_rows(previous_area, &widths, 20)
        );
    }

    /// A box extending into scrollback can only have its visible suffix erased.
    #[test]
    fn reflowed_top_saturates_at_the_visible_screen_without_scrolling() {
        let previous = Rect::new(0, 7, 40, 5);
        let offset =
            parked_anchor_offset(Some(previous), 5) + reflowed_extra_rows(previous, &[40; 5], 20);
        let (area, scroll) = fixed_viewport_area(
            Size::new(20, 12),
            Position::new(0, 6),
            5,
            offset,
            ViewportRebuildMode::ReflowOnly,
        );
        assert_eq!(area, Rect::new(0, 0, 20, 5));
        assert_eq!(scroll, 0);
    }

    #[test]
    fn narrowing_rebuild_clears_the_old_box_without_moving_transcript() {
        use ratatui::backend::TestBackend;

        let mut backend = TestBackend::new(20, 14);
        let body = Cell::new("K");
        for row in 0..14 {
            backend.draw(std::iter::once((0, row, &body))).unwrap();
        }
        backend.set_cursor_position(Position::new(0, 13)).unwrap();
        backend.append_lines(2).unwrap();
        let scrollback = backend.scrollback().clone();
        // The old five-row frame has reflowed into ten rows starting at row 2.
        // Its last original row starts at row 10; its trailing wrap is row 11.
        let ghost = Cell::new("X");
        for row in 2..12 {
            backend.draw(std::iter::once((0, row, &ghost))).unwrap();
        }
        backend.set_cursor_position(Position::new(0, 10)).unwrap();
        let previous = Rect::new(0, 7, 40, 5);
        let offset =
            parked_anchor_offset(Some(previous), 5) + reflowed_extra_rows(previous, &[40; 5], 20);
        let area = prepare_fixed_viewport(
            &mut backend,
            Size::new(20, 14),
            5,
            offset,
            ViewportRebuildMode::ReflowOnly,
            true,
            &mut VecDeque::new(),
        )
        .unwrap();
        assert_eq!(area, Rect::new(0, 2, 20, 5));
        assert_eq!(backend.scrollback(), &scrollback);
        for row in 0..2 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), "K");
        }
        for row in 2..14 {
            assert_eq!(backend.buffer()[(0, row)].symbol(), " ");
        }
    }

    fn assert_immediate_resize_frames(
        widths: &[u16],
        has_status: bool,
        settle_after_cycles: bool,
        notify_resize: bool,
    ) {
        use ratatui::backend::TestBackend;

        const HEIGHT: u16 = 24;
        const PROMPT_HEIGHT: u16 = 6;
        let transcript = ["model response stays directly above the prompt", "END"];
        let mut backend = TestBackend::new(80, HEIGHT);
        // Keep real scrollback as well as visible output: neither may change
        // during a prompt rebuild.
        backend
            .draw(std::iter::once((0, 0, &Cell::new("H"))))
            .unwrap();
        backend
            .set_cursor_position(Position::new(0, HEIGHT - 1))
            .unwrap();
        backend.append_lines(1).unwrap();
        let scrollback_height = backend.scrollback().area.height;
        let mut area = Rect::new(0, 2, 80, PROMPT_HEIGHT);
        let mut terminal = terminal_with_fixed_viewport(backend, area).unwrap();
        let mut textarea = TextArea::default();
        let frames_drawn = std::cell::Cell::new(0usize);
        let mut paint_prompt = |terminal: &mut Terminal<TestBackend>| {
            let mut widths = Vec::new();
            terminal
                .draw(|frame| {
                    let area = frame.area();
                    render_multiline_popup_with_checkpoint(
                        frame,
                        &mut textarea,
                        has_status.then_some("a long status message for this prompt"),
                        None,
                        "agent",
                        "fixture-model-with-a-long-name",
                        false,
                        "high",
                        Some("a long session title for this prompt"),
                        None,
                        area.width,
                        "",
                    );
                    widths = painted_row_widths(frame.buffer_mut(), area);
                })
                .unwrap();
            frames_drawn.set(frames_drawn.get() + 1);
            let area = terminal.get_frame().area();
            park_reflow_anchor(terminal, area).unwrap();
            widths
        };
        let mut row_widths = paint_prompt(&mut terminal);
        if has_status {
            for y in area.bottom() - 3..area.bottom() {
                let row = (0..area.width)
                    .map(|x| terminal.backend().buffer()[(x, y)].symbol())
                    .collect::<String>();
                assert!(
                    !row.contains('…'),
                    "wide chrome should fit before resizing: {row:?}"
                );
            }
        }
        let (input_start, input_len) = super::super::render::textarea_row_range(
            PROMPT_HEIGHT,
            &[String::new()],
            None,
            has_status,
            true,
        );
        let mut resize_state = PromptResizeState::default();
        let mut applied_size = Size::new(80, HEIGHT);
        let now = Instant::now();

        for &width in widths {
            let narrowing = width < area.width;
            // TestBackend clips on resize rather than reflowing. Split the
            // actual previous frame into physical rows to model the emulator,
            // including the bottom status row's fragments BELOW its anchor.
            let mut physical_rows = Vec::new();
            let mut anchor_row = 0;
            for (index, written_width) in row_widths.iter().copied().enumerate() {
                if index + 1 == area.height as usize {
                    anchor_row = physical_rows.len() as u16;
                }
                let row = (0..written_width)
                    .map(|x| terminal.backend().buffer()[(x, area.y + index as u16)].clone())
                    .collect::<Vec<_>>();
                if row.is_empty() {
                    physical_rows.push(Vec::new());
                } else {
                    physical_rows.extend(row.chunks(width as usize).map(|chunk| chunk.to_vec()));
                }
            }
            if narrowing {
                assert!(
                    row_widths[input_start as usize..(input_start + input_len) as usize]
                        .iter()
                        .all(|written| *written <= width),
                    "textarea must not wrap"
                );
                assert!(
                    row_widths[area.height as usize - 2] > width,
                    "title must wrap"
                );
                assert!(
                    row_widths[area.height as usize - 1] > width,
                    "status must wrap"
                );
            }
            terminal.backend_mut().resize(width, HEIGHT);
            terminal
                .backend_mut()
                .clear_region(BackendClearType::All)
                .unwrap();
            let transcript_rows = transcript
                .iter()
                .flat_map(|line| line.as_bytes().chunks(width as usize))
                .collect::<Vec<_>>();
            let top = transcript_rows.len() as u16;
            assert!(top + physical_rows.len() as u16 <= HEIGHT);
            for (y, row) in transcript_rows.iter().enumerate() {
                let cells = row
                    .iter()
                    .map(|byte| Cell::from(*byte as char))
                    .collect::<Vec<_>>();
                terminal
                    .backend_mut()
                    .draw(
                        cells
                            .iter()
                            .enumerate()
                            .map(|(x, cell)| (x as u16, y as u16, cell)),
                    )
                    .unwrap();
            }
            for (y, row) in physical_rows.iter().enumerate() {
                terminal
                    .backend_mut()
                    .draw(
                        row.iter()
                            .enumerate()
                            .map(|(x, cell)| (x as u16, top + y as u16, cell)),
                    )
                    .unwrap();
            }
            let before = terminal.backend().buffer().clone();
            let scrollback = terminal.backend().scrollback().clone();
            let parked = Position::new(0, top + anchor_row);
            terminal.backend_mut().set_cursor_position(parked).unwrap();
            let offset = reflowed_anchor_offset(Some(area), PROMPT_HEIGHT, &row_widths, width);
            let size = Size::new(width, HEIGHT);
            // Feed the same event and draw gate as the input loop. No clock
            // advancement or idle poll is allowed before the first frame.
            let mut redraw_requested = if notify_resize {
                resize_state.notify_resize(&Event::Resize(width, HEIGHT), applied_size)
            } else {
                // A foreground title/input update can observe new geometry
                // before its Resize notification is consumed.
                true
            };
            assert!(take_redraw_request(&mut redraw_requested, false));
            assert!(!resize_state.rebuild_is_due(now, size, applied_size));
            let kind = resize_state
                .take_frame_rebuild(now, size, applied_size)
                .unwrap();
            assert_eq!(kind, ResizeFrameRebuild::Immediate);
            let frames_before = frames_drawn.get();
            let (expected, appended) = fixed_viewport_area(
                size,
                parked,
                PROMPT_HEIGHT,
                offset,
                ViewportRebuildMode::ReflowOnly,
            );
            assert_eq!(
                expected.y, top,
                "gap after resizing {} -> {width}",
                area.width
            );
            assert_eq!(appended, 0);
            area = prepare_fixed_viewport(
                terminal.backend_mut(),
                size,
                PROMPT_HEIGHT,
                offset,
                ViewportRebuildMode::ReflowOnly,
                true,
                &mut VecDeque::new(),
            )
            .unwrap();
            assert_eq!(area, expected);
            resize_state.finish_rebuild(kind, now, size);
            applied_size = size;
            // All old fragments, including rows released below the replacement
            // frame, are cleared before painting. No prefix is left as a gap.
            for y in top..HEIGHT {
                for x in 0..width {
                    assert_eq!(terminal.backend().buffer()[(x, y)].symbol(), " ");
                }
            }
            terminal = terminal_with_fixed_viewport(terminal.backend().clone(), area).unwrap();
            park_reflow_anchor(&mut terminal, area).unwrap();
            terminal
                .backend_mut()
                .assert_cursor_position(Position::new(0, area.bottom() - 1));
            row_widths = paint_prompt(&mut terminal);
            assert_eq!(frames_drawn.get(), frames_before + 1);
            assert!(
                row_widths[PROMPT_HEIGHT as usize - 3..]
                    .iter()
                    .all(|written| *written < width),
                "repainted chrome must stay inside its single row"
            );
            if narrowing {
                for y in area.bottom() - 3..area.bottom() {
                    let row = (0..width)
                        .map(|x| terminal.backend().buffer()[(x, y)].symbol())
                        .collect::<String>();
                    assert!(row.trim_end().ends_with('…'), "first resize frame: {row:?}");
                }
            }
            terminal
                .backend_mut()
                .assert_cursor_position(Position::new(0, area.bottom() - 1));
            assert!(!terminal.backend().cursor_visible());
            assert_eq!(terminal.backend().scrollback(), &scrollback);
            assert_eq!(
                terminal.backend().scrollback().area.height,
                scrollback_height
            );
            for y in 0..top {
                for x in 0..width {
                    assert_eq!(terminal.backend().buffer()[(x, y)], before[(x, y)]);
                }
            }
            for y in area.bottom()..HEIGHT {
                for x in 0..width {
                    assert_eq!(terminal.backend().buffer()[(x, y)].symbol(), " ");
                }
            }
            assert!(resize_state.pending());
            assert!(!resize_state.notify_resize(&Event::Resize(width, HEIGHT), applied_size));
            assert!(
                resize_state.pending(),
                "a duplicate must retain reconciliation"
            );
            assert_eq!(resize_state.take_frame_rebuild(now, size, size), None);
        }

        if !settle_after_cycles {
            return;
        }
        // A single same-size settling pass consumes the new frame's widths.
        // It must not count the wide chrome again or move any transcript cell.
        let before = terminal.backend().buffer().clone();
        let scrollback = terminal.backend().scrollback().clone();
        let settled = now + RESIZE_SETTLE_DELAY;
        let kind = resize_state
            .take_frame_rebuild(settled, applied_size, applied_size)
            .unwrap();
        assert_eq!(kind, ResizeFrameRebuild::Reconcile);
        let offset = reflowed_anchor_offset(Some(area), PROMPT_HEIGHT, &row_widths, area.width);
        let rebuilt = prepare_fixed_viewport(
            terminal.backend_mut(),
            applied_size,
            PROMPT_HEIGHT,
            offset,
            ViewportRebuildMode::ReflowOnly,
            true,
            &mut VecDeque::new(),
        )
        .unwrap();
        assert_eq!(rebuilt, area);
        terminal = terminal_with_fixed_viewport(terminal.backend().clone(), rebuilt).unwrap();
        paint_prompt(&mut terminal);
        resize_state.finish_rebuild(kind, settled, applied_size);
        assert_eq!(terminal.backend().buffer(), &before);
        assert_eq!(terminal.backend().scrollback(), &scrollback);
        assert!(!resize_state.pending());
        assert_eq!(
            resize_state.take_frame_rebuild(
                settled + RESIZE_SETTLE_DELAY,
                applied_size,
                applied_size
            ),
            None
        );
    }

    #[test]
    fn resize_event_draws_ellipsis_before_the_settle_timer() {
        assert_immediate_resize_frames(&[20], true, false, true);
        assert_immediate_resize_frames(&[20], false, false, true);
    }

    #[test]
    fn live_resize_rebuilds_before_a_queued_notification() {
        assert_immediate_resize_frames(&[20], true, false, false);
    }

    #[test]
    fn chrome_reflow_cycles_keep_prompt_adjacent_and_transcript_unchanged() {
        assert_immediate_resize_frames(&[20, 40, 12, 40, 20, 40, 12, 40], true, true, true);
    }

    #[test]
    fn width_reflow_resize_during_cursor_query_keeps_both_frames_adjacent() {
        use ratatui::{backend::TestBackend, buffer::Buffer};

        const HEIGHT: u16 = 24;
        const PROMPT_HEIGHT: u16 = 6;
        let transcript = ["model response stays directly above the prompt", "END"];
        let mut backend = TestBackend::new(80, HEIGHT);
        backend
            .draw(std::iter::once((0, 0, &Cell::new("H"))))
            .unwrap();
        backend
            .set_cursor_position(Position::new(0, HEIGHT - 1))
            .unwrap();
        backend.append_lines(1).unwrap();
        let scrollback = backend.scrollback().clone();
        let mut area = Rect::new(0, 2, 80, PROMPT_HEIGHT);
        let mut terminal = terminal_with_fixed_viewport(backend, area).unwrap();
        let mut textarea = TextArea::default();
        let mut paint_prompt = |terminal: &mut Terminal<TestBackend>| {
            let mut widths = Vec::new();
            terminal
                .draw(|frame| {
                    let frame_area = frame.area();
                    render_multiline_popup_with_checkpoint(
                        frame,
                        &mut textarea,
                        Some("a long status message for this prompt"),
                        None,
                        "agent",
                        "fixture-model-with-a-long-name",
                        false,
                        "high",
                        Some("a long session title for this prompt"),
                        None,
                        frame_area.width,
                        "",
                    );
                    widths = painted_row_widths(frame.buffer_mut(), frame_area);
                })
                .unwrap();
            let frame_area = terminal.get_frame().area();
            park_reflow_anchor(terminal, frame_area).unwrap();
            widths
        };
        let mut row_widths = paint_prompt(&mut terminal);
        let mut old_frame = terminal.backend().buffer().clone();
        // TestBackend clips instead of reflowing. Model a resize from the actual
        // old frame, including the status fragments below its column-zero anchor.
        let reflow = |backend: &mut TestBackend,
                      width: u16,
                      previous: Rect,
                      old_frame: &Buffer,
                      row_widths: &[u16]| {
            let mut rows = Vec::new();
            let mut anchor_row = 0;
            for (index, written) in row_widths.iter().copied().enumerate() {
                if index + 1 == previous.height as usize {
                    anchor_row = rows.len() as u16;
                }
                let row = (0..written)
                    .map(|x| old_frame[(x, previous.y + index as u16)].clone())
                    .collect::<Vec<_>>();
                if row.is_empty() {
                    rows.push(Vec::new());
                } else {
                    rows.extend(row.chunks(width as usize).map(|chunk| chunk.to_vec()));
                }
            }
            backend.resize(width, HEIGHT);
            backend.clear_region(BackendClearType::All).unwrap();
            let transcript_rows = transcript
                .iter()
                .flat_map(|line| line.as_bytes().chunks(width as usize))
                .collect::<Vec<_>>();
            let top = transcript_rows.len() as u16;
            assert!(top + rows.len() as u16 <= HEIGHT);
            for (y, row) in transcript_rows.iter().enumerate() {
                let cells = row
                    .iter()
                    .map(|byte| Cell::from(*byte as char))
                    .collect::<Vec<_>>();
                backend
                    .draw(
                        cells
                            .iter()
                            .enumerate()
                            .map(|(x, cell)| (x as u16, y as u16, cell)),
                    )
                    .unwrap();
            }
            for (y, row) in rows.iter().enumerate() {
                backend
                    .draw(
                        row.iter()
                            .enumerate()
                            .map(|(x, cell)| (x as u16, top + y as u16, cell)),
                    )
                    .unwrap();
            }
            backend
                .set_cursor_position(Position::new(0, top + anchor_row))
                .unwrap();
            top
        };
        reflow(terminal.backend_mut(), 40, area, &old_frame, &row_widths);
        let now = Instant::now();
        let mut applied_size = Size::new(80, HEIGHT);
        let mut resize_state = PromptResizeState::default();
        assert!(resize_state.notify_resize(&Event::Resize(40, HEIGHT), applied_size));

        for (at, sampled_width, reply_width, expected_kind) in [
            (now, 40, 20, ResizeFrameRebuild::Immediate),
            (
                now + RESIZE_SETTLE_DELAY,
                20,
                12,
                ResizeFrameRebuild::Reconcile,
            ),
        ] {
            let sampled_size = terminal.backend().size().unwrap();
            assert_eq!(sampled_size.width, sampled_width);
            let kind = resize_state
                .take_frame_rebuild(at, sampled_size, applied_size)
                .unwrap();
            assert_eq!(kind, expected_kind);
            assert!(
                reflowed_anchor_offset(Some(area), PROMPT_HEIGHT, &row_widths, reply_width)
                    > reflowed_anchor_offset(Some(area), PROMPT_HEIGHT, &row_widths, sampled_width)
            );
            let mut queries = 0;
            let mut reflowed_frame = None;
            let mut reflowed_scrollback = None;
            let mut expected_top = 0;
            let (rebuilt, validated_size) = prepare_fixed_viewport_with_query(
                terminal.backend_mut(),
                sampled_size,
                PROMPT_HEIGHT,
                parked_anchor_offset(Some(area), PROMPT_HEIGHT),
                ViewportRebuildMode::ReflowOnly,
                true,
                Some(area),
                &row_widths,
                &mut VecDeque::new(),
                |backend, query_size, pending| {
                    if queries == 0 {
                        assert_eq!(query_size.width, sampled_width);
                        expected_top = reflow(backend, reply_width, area, &old_frame, &row_widths);
                        reflowed_frame = Some(backend.buffer().clone());
                        reflowed_scrollback = Some(backend.scrollback().clone());
                    } else {
                        assert_eq!(query_size.width, reply_width);
                        assert_eq!(
                            backend.buffer(),
                            reflowed_frame.as_ref().unwrap(),
                            "a changed-size reply must not clear any old prompt row"
                        );
                    }
                    queries += 1;
                    Ok(tracked_cursor_query(pending, || backend.get_cursor_position()).unwrap())
                },
            )
            .unwrap();
            assert_eq!(
                queries, 2,
                "query again before clearing at the changed width"
            );
            assert_eq!(validated_size, Size::new(reply_width, HEIGHT));
            assert_eq!(
                rebuilt,
                Rect::new(0, expected_top, reply_width, PROMPT_HEIGHT)
            );
            let before_clear = reflowed_frame.unwrap();
            // TestBackend also resizes scrollback columns. Compare the rebuild
            // against the emulator's new-width state, while its row count stays fixed.
            let resized_scrollback = reflowed_scrollback.unwrap();
            assert_eq!(resized_scrollback.area.height, scrollback.area.height);
            for y in 0..expected_top {
                for x in 0..reply_width {
                    assert_eq!(terminal.backend().buffer()[(x, y)], before_clear[(x, y)]);
                }
            }
            for y in expected_top..HEIGHT {
                for x in 0..reply_width {
                    assert_eq!(terminal.backend().buffer()[(x, y)].symbol(), " ");
                }
            }
            assert_eq!(
                terminal.backend().scrollback(),
                &resized_scrollback,
                "no rows may be appended"
            );
            area = rebuilt;
            terminal = terminal_with_fixed_viewport(terminal.backend().clone(), area).unwrap();
            row_widths = paint_prompt(&mut terminal);
            terminal
                .backend_mut()
                .assert_cursor_position(Position::new(0, area.bottom() - 1));
            assert!(!terminal.backend().cursor_visible());
            assert_eq!(terminal.backend().scrollback(), &resized_scrollback);
            for y in 0..expected_top {
                for x in 0..reply_width {
                    assert_eq!(terminal.backend().buffer()[(x, y)], before_clear[(x, y)]);
                }
            }
            for y in area.bottom()..HEIGHT {
                for x in 0..reply_width {
                    assert_eq!(terminal.backend().buffer()[(x, y)].symbol(), " ");
                }
            }
            for y in area.bottom() - 3..area.bottom() {
                let row = (0..reply_width)
                    .map(|x| terminal.backend().buffer()[(x, y)].symbol())
                    .collect::<String>();
                assert!(row.trim_end().ends_with('…'), "resize frame: {row:?}");
            }
            old_frame = terminal.backend().buffer().clone();
            resize_state.finish_rebuild(kind, at, validated_size);
            applied_size = validated_size;
        }
        assert!(!resize_state.pending());
    }

    #[test]
    fn same_size_reconciliation_recovers_an_immediate_anchor_that_moved() {
        use ratatui::backend::TestBackend;

        let now = Instant::now();
        let size = Size::new(20, 18);
        let mut resize_state = PromptResizeState::default();
        assert!(resize_state.notify_resize(&Event::Resize(20, 18), Size::new(40, 18)));
        let kind = resize_state
            .take_frame_rebuild(now, size, Size::new(40, 18))
            .unwrap();
        assert_eq!(kind, ResizeFrameRebuild::Immediate);
        resize_state.finish_rebuild(kind, now, size);
        let kind = resize_state
            .take_frame_rebuild(now + RESIZE_SETTLE_DELAY, size, size)
            .unwrap();
        assert_eq!(kind, ResizeFrameRebuild::Reconcile);

        // A late terminal reflow moved both the transcript and the immediate
        // prompt down two rows without another size notification.
        let previous = Rect::new(0, 4, 20, 6);
        let mut backend = TestBackend::new(20, 18);
        for y in 0..6 {
            backend
                .draw(std::iter::once((0, y, &Cell::new("K"))))
                .unwrap();
        }
        for y in 6..12 {
            backend
                .draw(std::iter::once((0, y, &Cell::new("X"))))
                .unwrap();
        }
        backend.set_cursor_position(Position::new(0, 11)).unwrap();
        let scrollback = backend.scrollback().clone();
        let offset = reflowed_anchor_offset(Some(previous), 6, &[3, 1, 1, 18, 18, 18], 20);
        let area = prepare_fixed_viewport(
            &mut backend,
            size,
            6,
            offset,
            ViewportRebuildMode::ReflowOnly,
            true,
            &mut VecDeque::new(),
        )
        .unwrap();
        assert_eq!(area, Rect::new(0, 6, 20, 6));
        assert_eq!(backend.scrollback(), &scrollback);
        for y in 0..6 {
            assert_eq!(backend.buffer()[(0, y)].symbol(), "K");
        }
        for y in 6..18 {
            assert_eq!(backend.buffer()[(0, y)].symbol(), " ");
        }
        let mut terminal = terminal_with_fixed_viewport(backend, area).unwrap();
        park_reflow_anchor(&mut terminal, area).unwrap();
        terminal
            .backend_mut()
            .assert_cursor_position(Position::new(0, 11));
        resize_state.finish_rebuild(kind, now + RESIZE_SETTLE_DELAY, size);
        assert!(!resize_state.pending());
    }

    #[test]
    fn width_reflow_anchor_probe_stays_inside_prompt_group() {
        let row = std::cell::Cell::new(23_u16);
        let saved = std::cell::Cell::new(23_u16);
        let visited = RefCell::new(Vec::new());
        let mut query = || {
            visited.borrow_mut().push(row.get());
            Ok(row.get())
        };
        let result = verified_prompt_cursor_row(
            &mut query,
            &mut || {
                saved.set(row.get());
                Ok(())
            },
            &mut || {
                row.set(saved.get());
                Ok(())
            },
            &mut |offset| {
                row.set(row.get() - offset);
                Ok(())
            },
            &mut |offset| {
                row.set(row.get() + offset);
                Ok(())
            },
            6,
            &mut VecDeque::new(),
        )
        .unwrap();
        assert_eq!(result, 23);
        assert_eq!(row.get(), 23);
        assert_eq!(visited.borrow().len(), 4);
        assert!(visited.borrow().iter().all(|row| (18..24).contains(row)));
    }

    #[test]
    fn width_reflow_bottom_scrollback_round_trips_keep_prompt_and_caret() {
        use ratatui::backend::WindowSize;

        // Model the xterm buffer contract, including its cursor-line exemption.
        // With scrollback, reflow inserts rows by advancing baseY rather than
        // advancing the cursor's screen row, even when blank rows follow it.
        #[derive(Clone)]
        struct Line {
            cells: Vec<Cell>,
            written: usize,
            wrapped: bool,
        }
        impl Line {
            fn blank(width: u16) -> Self {
                Self {
                    cells: vec![Cell::default(); width as usize],
                    written: 0,
                    wrapped: false,
                }
            }
        }
        #[derive(Clone)]
        struct ReflowBackend {
            size: Size,
            lines: Vec<Line>,
            base_y: usize,
            cursor: Position,
            cursor_visible: bool,
        }
        impl ReflowBackend {
            fn new() -> Self {
                let size = Size::new(80, 24);
                let base_y = 32;
                let mut lines = vec![Line::blank(size.width); base_y + size.height as usize];
                for line in &mut lines[..base_y + 18] {
                    line.cells[0] = Cell::new("T");
                    line.written = 1;
                }
                Self {
                    size,
                    lines,
                    base_y,
                    cursor: Position::new(0, 18),
                    cursor_visible: false,
                }
            }
            fn resize(&mut self, width: u16) {
                let absolute_cursor = self.base_y + self.cursor.y as usize;
                let old_len = self.lines.len();
                let old_width = self.size.width;
                let mut reflowed = Vec::new();
                let mut index = 0;
                while index < self.lines.len() {
                    let start = index;
                    index += 1;
                    while index < self.lines.len() && self.lines[index].wrapped {
                        index += 1;
                    }
                    if (start..index).contains(&absolute_cursor) {
                        // xterm leaves the active logical line to the application.
                        for line in &self.lines[start..index] {
                            let mut line = line.clone();
                            line.cells.resize(width as usize, Cell::default());
                            line.written = line.written.min(width as usize);
                            reflowed.push(line);
                        }
                        continue;
                    }
                    let mut content = Vec::new();
                    for (offset, line) in self.lines[start..index].iter().enumerate() {
                        let used = if offset + 1 < index - start {
                            old_width as usize
                        } else {
                            line.written
                        };
                        content.extend_from_slice(&line.cells[..used]);
                    }
                    if content.is_empty() {
                        reflowed.push(Line::blank(width));
                    } else {
                        for (part, cells) in content.chunks(width as usize).enumerate() {
                            let mut line = Line::blank(width);
                            line.cells[..cells.len()].clone_from_slice(cells);
                            line.written = cells.len();
                            line.wrapped = part > 0;
                            reflowed.push(line);
                        }
                    }
                }
                if reflowed.len() >= old_len {
                    self.base_y += reflowed.len() - old_len;
                } else {
                    self.base_y = self.base_y.saturating_sub(old_len - reflowed.len());
                }
                self.lines = reflowed;
                self.size.width = width;
                self.cursor.x = self.cursor.x.min(width - 1);
            }
            fn assert_caret(&self, area: Rect) {
                let carets = (0..self.size.height)
                    .flat_map(|y| {
                        self.lines[self.base_y + y as usize]
                            .cells
                            .iter()
                            .enumerate()
                            .filter(|(_, cell)| {
                                cell.symbol() == " "
                                    && cell.modifier.contains(Modifier::REVERSED | Modifier::BOLD)
                            })
                            .map(move |(x, _)| Position::new(x as u16, y))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(carets.len(), 1, "carets={carets:?}");
                assert_eq!(carets[0].y, area.y);
                assert!(!self.cursor_visible);
                for line in &self.lines[self.base_y + area.bottom() as usize..] {
                    assert!(
                        line.cells
                            .iter()
                            .all(|cell| cell.symbol().trim().is_empty())
                    );
                }
            }
        }
        impl Backend for ReflowBackend {
            type Error = io::Error;

            fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
            where
                I: Iterator<Item = (u16, u16, &'a Cell)>,
            {
                for (x, y, cell) in content {
                    let row = self.base_y + y as usize;
                    if x + 1 == self.size.width && cell.symbol() == "  " {
                        // Two contiguous spaces at the right edge naturally wrap
                        // onto the following row, as Crossterm's Print does.
                        self.lines[row].cells[x as usize] = Cell::default();
                        self.lines[row].written = self.size.width as usize;
                        self.lines[row + 1].cells[0] = Cell::default();
                        self.lines[row + 1].wrapped = true;
                    } else {
                        self.lines[row].cells[x as usize] = cell.clone();
                        self.lines[row].written = self.lines[row].written.max(x as usize + 1);
                    }
                }
                Ok(())
            }
            fn hide_cursor(&mut self) -> io::Result<()> {
                self.cursor_visible = false;
                Ok(())
            }
            fn show_cursor(&mut self) -> io::Result<()> {
                self.cursor_visible = true;
                Ok(())
            }
            fn get_cursor_position(&mut self) -> io::Result<Position> {
                Ok(self.cursor)
            }
            fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
                self.cursor = position.into();
                Ok(())
            }
            fn clear(&mut self) -> io::Result<()> {
                self.clear_region(BackendClearType::All)
            }
            fn clear_region(&mut self, clear_type: BackendClearType) -> io::Result<()> {
                let from = match clear_type {
                    BackendClearType::All => self.base_y,
                    BackendClearType::AfterCursor => self.base_y + self.cursor.y as usize,
                    _ => return Err(io::Error::other("unsupported reflow-fixture clear")),
                };
                for line in &mut self.lines[from..] {
                    *line = Line::blank(self.size.width);
                }
                Ok(())
            }
            fn append_lines(&mut self, count: u16) -> io::Result<()> {
                self.base_y += count as usize;
                self.lines
                    .extend((0..count).map(|_| Line::blank(self.size.width)));
                Ok(())
            }
            fn size(&self) -> io::Result<Size> {
                Ok(self.size)
            }
            fn window_size(&mut self) -> io::Result<WindowSize> {
                Ok(WindowSize {
                    columns_rows: self.size,
                    pixels: Size::ZERO,
                })
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        fn paint(terminal: &mut Terminal<ReflowBackend>, widths: &mut Vec<u16>) {
            let mut textarea = TextArea::default();
            terminal
                .draw(|frame| {
                    let area = frame.area();
                    render_multiline_popup_with_checkpoint(
                        frame,
                        &mut textarea,
                        Some("status message with a long explanatory label that needs an ellipsis"),
                        None,
                        "fixture-agent",
                        "fixture-model-with-a-long-name",
                        false,
                        "high",
                        Some("session topic with a long descriptive label that also needs an ellipsis"),
                        None,
                        area.width,
                        "fixture-backend",
                    );
                    *widths = painted_row_widths(frame.buffer_mut(), area);
                })
                .unwrap();
            let area = terminal.get_frame().area();
            assert!(
                preserve_prompt_cursor_group(terminal.backend_mut(), area, area.width).unwrap()
            );
            park_reflow_anchor(terminal, area).unwrap();
            terminal.backend().assert_caret(area);
            if area.width == 40 {
                for y in area.bottom() - 3..area.bottom() {
                    let row = &terminal.backend().lines[terminal.backend().base_y + y as usize];
                    let text = row.cells.iter().map(Cell::symbol).collect::<String>();
                    assert!(text.trim_end().ends_with('…'), "chrome={text:?}");
                }
            }
        }

        let mut terminal =
            terminal_with_fixed_viewport(ReflowBackend::new(), Rect::new(0, 18, 80, 6)).unwrap();
        let mut area = terminal.get_frame().area();
        let mut widths = Vec::new();
        paint(&mut terminal, &mut widths);
        let initial_top = area.y;
        let initial_cursor = terminal.backend().cursor.y;
        let initial_base = terminal.backend().base_y;
        let mut tops = vec![initial_top];
        let mut cursors = vec![initial_cursor];
        let mut bases = vec![initial_base];
        let mut applied = Size::new(80, 24);
        let mut resize = PromptResizeState::default();
        let mut now = Instant::now();
        for _ in 0..4 {
            for width in [40, 80] {
                terminal.backend_mut().resize(width);
                let size = terminal.backend().size().unwrap();
                assert!(resize.notify_resize(&Event::Resize(width, size.height), applied));
                let kind = resize.take_frame_rebuild(now, size, applied).unwrap();
                let (next_area, size) = prepare_cursor_group_viewport_with_query(
                    terminal.backend_mut(),
                    size,
                    6,
                    parked_anchor_offset(Some(area), 6),
                    ViewportRebuildMode::ReflowOnly,
                    true,
                    Some(area),
                    &widths,
                    Some(area.height),
                    &mut VecDeque::new(),
                    |backend, _, _| backend.get_cursor_position(),
                )
                .unwrap();
                area = next_area;
                applied = size;
                resize.finish_rebuild(kind, now, size);
                widths.clear();
                terminal = terminal_with_fixed_viewport(terminal.backend().clone(), area).unwrap();
                paint(&mut terminal, &mut widths);
                if width == 80 {
                    tops.push(area.y);
                    cursors.push(terminal.backend().cursor.y);
                    bases.push(terminal.backend().base_y);
                }
                now += Duration::from_millis(25);
            }
        }
        assert_eq!(
            tops,
            vec![initial_top; 5],
            "cursor rows={cursors:?}, baseY={bases:?}"
        );
        assert_eq!(cursors, vec![initial_cursor; 5]);
        assert_eq!(bases, vec![initial_base; 5]);
        eprintln!("width round trips: top={tops:?}, cursor={cursors:?}, baseY={bases:?}");

        now += RESIZE_SETTLE_DELAY;
        let kind = resize.take_frame_rebuild(now, applied, applied).unwrap();
        assert_eq!(kind, ResizeFrameRebuild::Reconcile);
        let (area, size) = prepare_cursor_group_viewport_with_query(
            terminal.backend_mut(),
            applied,
            6,
            5,
            ViewportRebuildMode::ReflowOnly,
            true,
            Some(area),
            &widths,
            Some(area.height),
            &mut VecDeque::new(),
            |backend, _, _| backend.get_cursor_position(),
        )
        .unwrap();
        resize.finish_rebuild(kind, now, size);
        terminal = terminal_with_fixed_viewport(terminal.backend().clone(), area).unwrap();
        paint(&mut terminal, &mut widths);
        assert_eq!(area.y, initial_top);
        assert_eq!(terminal.backend().cursor.y, initial_cursor);
        assert_eq!(terminal.backend().base_y, initial_base);
        assert!(!resize.pending());
    }

    #[test]
    fn width_reflow_post_clear_resize_reanchors_before_paint() {
        use ratatui::backend::TestBackend;

        const HEIGHT: u16 = 32;
        const PROMPT_HEIGHT: u16 = 6;
        struct Fixture {
            terminal: Terminal<TestBackend>,
            area: Rect,
            size: Size,
            widths: Vec<u16>,
            transcript: Vec<String>,
            scrollback_rows: u16,
        }
        impl Fixture {
            fn new() -> Self {
                let mut backend = TestBackend::new(40, HEIGHT);
                backend
                    .draw(std::iter::once((0, 0, &Cell::new("H"))))
                    .unwrap();
                backend
                    .set_cursor_position(Position::new(0, HEIGHT - 1))
                    .unwrap();
                backend.append_lines(1).unwrap();
                let scrollback_rows = backend.scrollback().area.height;
                let transcript = (0..12)
                    .map(|index| format!("output {index}"))
                    .chain(std::iter::once(
                        "last model response wraps 0123456789 abcdef".to_owned(),
                    ))
                    .collect::<Vec<_>>();
                let top = transcript
                    .iter()
                    .map(|line| line.len().div_ceil(40))
                    .sum::<usize>() as u16;
                let area = Rect::new(0, top, 40, PROMPT_HEIGHT);
                let mut fixture = Self {
                    terminal: terminal_with_fixed_viewport(backend, area).unwrap(),
                    area,
                    size: Size::new(40, HEIGHT),
                    widths: Vec::new(),
                    transcript,
                    scrollback_rows,
                };
                fixture.resize(40);
                fixture.paint();
                fixture
            }
            fn top(&self) -> u16 {
                let width = self.terminal.backend().size().unwrap().width as usize;
                self.transcript
                    .iter()
                    .map(|line| line.len().div_ceil(width))
                    .sum::<usize>() as u16
            }
            fn assert_transcript(&self) {
                let width = self.terminal.backend().size().unwrap().width;
                let rows = self
                    .transcript
                    .iter()
                    .flat_map(|line| line.as_bytes().chunks(width as usize));
                for (y, row) in rows.enumerate() {
                    for x in 0..width {
                        let expected = row
                            .get(x as usize)
                            .map_or(Cell::new(" "), |byte| Cell::from(*byte as char));
                        assert_eq!(self.terminal.backend().buffer()[(x, y as u16)], expected);
                    }
                }
            }
            fn resize(&mut self, width: u16) {
                let old_frame = self.terminal.backend().buffer().clone();
                let mut rows = Vec::new();
                let mut anchor_row = 0;
                for index in 0..self.area.height {
                    if index + 1 == self.area.height && !self.widths.is_empty() {
                        anchor_row = rows.len() as u16;
                    }
                    let written = self.widths.get(index as usize).copied().unwrap_or(0);
                    let row = (0..written)
                        .map(|x| old_frame[(x, self.area.y + index)].clone())
                        .collect::<Vec<_>>();
                    if row.is_empty() {
                        rows.push(Vec::new());
                    } else {
                        rows.extend(row.chunks(width as usize).map(|chunk| chunk.to_vec()));
                    }
                }
                self.terminal.backend_mut().resize(width, HEIGHT);
                self.terminal
                    .backend_mut()
                    .clear_region(BackendClearType::All)
                    .unwrap();
                let output = self
                    .transcript
                    .iter()
                    .flat_map(|line| line.as_bytes().chunks(width as usize))
                    .collect::<Vec<_>>();
                let top = output.len() as u16;
                assert!(top + rows.len() as u16 <= HEIGHT);
                for (y, row) in output.iter().enumerate() {
                    let cells = row
                        .iter()
                        .map(|byte| Cell::from(*byte as char))
                        .collect::<Vec<_>>();
                    self.terminal
                        .backend_mut()
                        .draw(
                            cells
                                .iter()
                                .enumerate()
                                .map(|(x, cell)| (x as u16, y as u16, cell)),
                        )
                        .unwrap();
                }
                for (y, row) in rows.iter().enumerate() {
                    self.terminal
                        .backend_mut()
                        .draw(
                            row.iter()
                                .enumerate()
                                .map(|(x, cell)| (x as u16, top + y as u16, cell)),
                        )
                        .unwrap();
                }
                self.terminal
                    .backend_mut()
                    .set_cursor_position(Position::new(0, top + anchor_row))
                    .unwrap();
            }
            fn clear(&mut self) -> io::Result<Size> {
                let before = self.terminal.backend().scrollback().clone();
                let size = self.terminal.backend().size().unwrap();
                let (area, size) = prepare_fixed_viewport_with_query(
                    self.terminal.backend_mut(),
                    size,
                    PROMPT_HEIGHT,
                    parked_anchor_offset(Some(self.area), PROMPT_HEIGHT),
                    ViewportRebuildMode::ReflowOnly,
                    true,
                    Some(self.area),
                    &self.widths,
                    &mut VecDeque::new(),
                    |backend, _, pending| {
                        Ok(
                            tracked_cursor_query(pending, || backend.get_cursor_position())
                                .unwrap(),
                        )
                    },
                )?;
                self.area = area;
                self.size = size;
                self.widths.clear();
                self.terminal =
                    terminal_with_fixed_viewport(self.terminal.backend().clone(), area).unwrap();
                assert_eq!(
                    self.area.y,
                    self.top(),
                    "clear must recover the first old prompt row"
                );
                self.terminal
                    .backend_mut()
                    .assert_cursor_position(Position::new(0, self.area.y));
                self.assert_transcript();
                for y in self.top()..HEIGHT {
                    for x in 0..size.width {
                        assert_eq!(self.terminal.backend().buffer()[(x, y)].symbol(), " ");
                    }
                }
                assert_eq!(
                    self.terminal.backend().scrollback(),
                    &before,
                    "no appended rows"
                );
                Ok(size)
            }
            fn paint(&mut self) {
                let mut textarea = TextArea::default();
                let mut widths = Vec::new();
                self.terminal
                    .draw(|frame| {
                        let area = frame.area();
                        render_multiline_popup_with_checkpoint(
                            frame,
                            &mut textarea,
                            Some("status text that must truncate directly at the new width"),
                            None,
                            "agent",
                            "fixture-model-with-a-long-name",
                            false,
                            "high",
                            Some("session title that must truncate directly at the new width"),
                            None,
                            area.width,
                            "",
                        );
                        widths = painted_row_widths(frame.buffer_mut(), area);
                    })
                    .unwrap();
                self.widths = widths;
                park_reflow_anchor(&mut self.terminal, self.area).unwrap();
                assert_eq!(
                    self.area.y,
                    self.top(),
                    "a painted prompt must be adjacent to output"
                );
                assert_eq!(self.area.width, self.size.width);
                self.terminal
                    .backend_mut()
                    .assert_cursor_position(Position::new(0, self.area.bottom() - 1));
                assert!(!self.terminal.backend().cursor_visible());
                self.assert_transcript();
                assert_eq!(
                    self.terminal.backend().scrollback().area.height,
                    self.scrollback_rows
                );
                for y in self.area.bottom()..HEIGHT {
                    for x in 0..self.size.width {
                        assert_eq!(self.terminal.backend().buffer()[(x, y)].symbol(), " ");
                    }
                }
                for y in self.area.bottom() - 3..self.area.bottom() {
                    let row = (0..self.size.width)
                        .map(|x| self.terminal.backend().buffer()[(x, y)].symbol())
                        .collect::<String>();
                    assert!(
                        row.trim_end().ends_with('…'),
                        "first paint at new width: {row:?}"
                    );
                }
            }
        }
        let mut fixture = Fixture::new();
        let mut resize_state = PromptResizeState::default();
        let mut now = Instant::now();
        for _ in 0..4 {
            let painted_width = fixture.size.width;
            let notified_width = if painted_width == 40 { 20 } else { 40 };
            fixture.resize(notified_width);
            assert!(
                resize_state.notify_resize(&Event::Resize(notified_width, HEIGHT), fixture.size)
            );
            for kind in [ResizeFrameRebuild::Immediate, ResizeFrameRebuild::Reconcile] {
                let live = fixture.terminal.backend().size().unwrap();
                assert_eq!(
                    resize_state.take_frame_rebuild(now, live, fixture.size),
                    Some(kind)
                );
                let size = fixture.clear().unwrap();
                resize_state.finish_rebuild(kind, now, size);
                let target_width = if kind == ResizeFrameRebuild::Immediate {
                    painted_width
                } else {
                    notified_width
                };
                let mut applied = fixture.size;
                let mut injected = false;
                prepare_prompt_paint(
                    &mut fixture,
                    &mut applied,
                    |fixture| {
                        if !injected {
                            assert!(
                                fixture.widths.is_empty(),
                                "inject only after the validated clear"
                            );
                            fixture.resize(target_width);
                            injected = true;
                        }
                        Ok(fixture.terminal.backend().size().unwrap())
                    },
                    |fixture| {
                        let size = fixture.clear()?;
                        resize_state.finish_rebuild(ResizeFrameRebuild::Immediate, now, size);
                        Ok(size)
                    },
                )
                .unwrap();
                assert_eq!(applied, fixture.size);
                assert_eq!(applied.width, target_width);
                fixture.paint();
                assert!(
                    resize_state.pending(),
                    "a new pre-paint resize owes reconciliation"
                );
                now += RESIZE_SETTLE_DELAY;
            }
        }
        assert_eq!(
            resize_state.take_frame_rebuild(now, fixture.size, fixture.size),
            Some(ResizeFrameRebuild::Reconcile)
        );
        let size = fixture.clear().unwrap();
        resize_state.finish_rebuild(ResizeFrameRebuild::Reconcile, now, size);
        let mut applied = fixture.size;
        prepare_prompt_paint(
            &mut fixture,
            &mut applied,
            |fixture| Ok(fixture.terminal.backend().size().unwrap()),
            |fixture| fixture.clear(),
        )
        .unwrap();
        fixture.paint();
        assert!(!resize_state.pending());
    }

    #[test]
    fn width_reflow_continuous_post_clear_resize_defers_paint() {
        let mut live = Size::new(20, 24);
        let mut applied = live;
        let mut reanchors = 0;
        let result = prepare_prompt_paint(
            &mut live,
            &mut applied,
            |live| {
                live.width = if live.width == 20 { 40 } else { 20 };
                Ok(*live)
            },
            |live| {
                reanchors += 1;
                Ok(*live)
            },
        );
        assert!(defer_unstable_prompt_geometry(result).unwrap().is_none());
        assert_eq!(reanchors, PROMPT_GEOMETRY_RETRIES);
    }

    #[test]
    fn width_reflow_continuous_query_resize_keeps_old_prompt_intact() {
        use ratatui::backend::TestBackend;
        let mut backend = TestBackend::new(20, 10);
        backend
            .draw(std::iter::once((0, 9, &Cell::new("X"))))
            .unwrap();
        backend.set_cursor_position(Position::new(0, 9)).unwrap();
        let mut queries = 0;
        let mut last_resized_frame = None;
        let result = prepare_fixed_viewport_with_query(
            &mut backend,
            Size::new(20, 10),
            6,
            5,
            ViewportRebuildMode::ReflowOnly,
            true,
            Some(Rect::new(0, 4, 20, 6)),
            &[1; 6],
            &mut VecDeque::new(),
            |backend, size, pending| {
                queries += 1;
                backend.resize(if size.width == 20 { 40 } else { 20 }, 10);
                last_resized_frame = Some(backend.buffer().clone());
                Ok(tracked_cursor_query(pending, || backend.get_cursor_position()).unwrap())
            },
        );
        assert!(defer_unstable_prompt_geometry(result).unwrap().is_none());
        assert_eq!(queries, PROMPT_GEOMETRY_RETRIES);
        // TestBackend reshapes its buffer on resize. Every old cell must remain
        // intact after the final resize, including the prompt sentinel.
        let expected = last_resized_frame.unwrap();
        assert!(expected.content.iter().any(|cell| cell.symbol() == "X"));
        assert_eq!(backend.buffer(), &expected);
        assert_eq!(backend.scrollback().area.height, 0);
    }

    /// `painted_row_widths` reports where the frame stopped writing: text, a
    /// styled blank cell count, blank padding does not.
    #[test]
    fn painted_row_widths_follow_the_furthest_written_column() {
        use ratatui::buffer::Buffer;

        let area = Rect::new(0, 0, 10, 4);
        let mut buffer = Buffer::empty(area);
        buffer[(2, 0)].set_symbol("x");
        // Row 1 is never written: its cells stay blank.
        // Row 2 holds only a space, but a styled one, which the backend emits.
        buffer[(4, 2)].set_style(Style::default().fg(Color::Red));
        assert_eq!(painted_row_widths(&buffer, area), vec![3, 0, 5, 0]);
    }

    /// A forced repaint reaches as far as ink can exist in a row — its own
    /// content or an earlier, wider write — and no further, so a row that holds
    /// nothing is not written out to the full width the terminal would count.
    #[test]
    fn forced_repaint_widths_stop_at_the_last_cell_that_can_carry_ink() {
        use ratatui::buffer::Buffer;

        let area = Rect::new(0, 0, 10, 4);
        let mut buffer = Buffer::empty(area);
        buffer[(2, 0)].set_symbol("x");
        buffer[(4, 2)].set_style(Style::default().fg(Color::Red));
        // Row 0 held ten columns before it shrank to three; row 1 is blank in
        // this frame but an earlier one wrote five columns there.
        assert_eq!(
            forced_repaint_widths(&buffer, area, &[10, 5, 0, 0]),
            vec![10, 5, 5, 0]
        );
        // With no record of earlier writes only the content is rewritten, so the
        // blank rows stay untouched instead of becoming full-width lines.
        assert_eq!(forced_repaint_widths(&buffer, area, &[]), vec![3, 0, 5, 0]);
    }
}

/// Parks the hardware cursor at the viewport's bottom row, hidden.
///
/// The visible editing caret is drawn into the buffer as a styled cell (see
/// render.rs), so it reflows with the text and needs no tracking. The hardware
/// cursor is parked at column zero on the box's bottom row. The live anchor
/// follows that row through terminal reflow. Recovering the old top subtracts
/// `height - 1` plus the extra wraps of all rows BEFORE the bottom row; the
/// bottom row's own trailing wraps are below the anchor. The resulting local
/// clear removes the entire visible old frame without scrolling the transcript.
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
    *pending_resize_rebuild =
        !live_size_matches_applied(notified_terminal_size, last_applied_terminal_size);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResizeFrameRebuild {
    Immediate,
    Reconcile,
}

#[derive(Default)]
struct PromptResizeState {
    immediate: bool,
    reconcile: bool,
    settle: ResizeSettle,
}

impl PromptResizeState {
    /// Explicit size notifications repaint before polling for another event.
    /// A duplicate must neither redraw nor cancel the owed settling pass.
    fn notify_resize(&mut self, event: &Event, applied_size: Size) -> bool {
        let Event::Resize(width, height) = event else {
            return false;
        };
        update_pending_resize_rebuild(
            &mut self.immediate,
            applied_size,
            Size::new(*width, *height),
        );
        if self.immediate {
            self.settle.restart();
        }
        self.immediate
    }

    fn pending(&self) -> bool {
        self.immediate || self.reconcile
    }

    fn rebuild_is_due(&mut self, now: Instant, live_size: Size, applied_size: Size) -> bool {
        if self.reconcile && live_size_matches_applied(live_size, applied_size) {
            // The immediate frame already applied this width. Its final
            // cursor check must not be canceled just because the size matches.
            self.settle.observe(now, live_size)
        } else {
            resize_rebuild_is_due(&mut self.settle, now, live_size, applied_size)
        }
    }

    fn take_frame_rebuild(
        &mut self,
        now: Instant,
        live_size: Size,
        applied_size: Size,
    ) -> Option<ResizeFrameRebuild> {
        if take_standalone_resize_rebuild(&mut self.immediate, false)
            || !live_size_matches_applied(live_size, applied_size)
        {
            // A title/input redraw can beat the queued Resize event. It must
            // not paint fixed coordinates from the old terminal geometry.
            Some(ResizeFrameRebuild::Immediate)
        } else if self.rebuild_is_due(now, live_size, applied_size) {
            Some(ResizeFrameRebuild::Reconcile)
        } else {
            None
        }
    }

    fn finish_rebuild(&mut self, kind: ResizeFrameRebuild, now: Instant, sampled_size: Size) {
        self.reconcile = kind == ResizeFrameRebuild::Immediate;
        if self.reconcile {
            self.settle.observe(now, sampled_size);
        }
    }
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
/// Forces every cell that can carry ink in the current frame to be written back
/// to the terminal, and returns the per-row width record that follows from it.
///
/// After terminal reflow the terminal may still show a deleted character, while
/// ratatui's previous-frame buffer already considers that position blank, so a
/// regular diff would not emit a space there again. Use `AlwaysUpdate` after
/// input shortens or an emulator-width change clears the protected rows. The
/// wipe reaches only as far as `forced_repaint_widths`
/// says ink can exist, so it does not stretch a short row to the full width.
fn force_frame_repaint(frame: &mut ratatui::Frame<'_>, previous: &[u16]) -> Vec<u16> {
    let area = frame.area();
    let row_widths = forced_repaint_widths(frame.buffer_mut(), area, previous);
    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        let width = row_widths
            .get(y.saturating_sub(area.top()) as usize)
            .copied()
            .unwrap_or(0);
        for x in area.left()..area.left().saturating_add(width) {
            buffer[(x, y)].set_diff_option(CellDiffOption::AlwaysUpdate);
        }
    }
    row_widths
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

/// Schedule the deferred resize once the input poll goes idle. A consumed CPR
/// also returns no event, but can leave resize notifications queued: drain them
/// before rebuilding so an intermediate size cannot become the new anchor.
fn resize_redraw_after_idle(
    pending_resize_rebuild: bool,
    pending_events: &VecDeque<Event>,
) -> bool {
    pending_resize_rebuild && pending_events.is_empty()
}

/// Whether a changed live geometry may rebuild the viewport now.
///
/// Every sighting counts, including one that reads the already-applied size
/// again: geometry that moved back is a move, and it restarts the window instead
/// of leaving the earlier sightings standing. A geometry that matches the applied
/// one still asks for no rebuild. The idle poll uses this window for the final
/// reconciliation. `PromptResizeState` bypasses it for explicit resize events
/// and foreground redraws that observe changed geometry. A live width within
/// one column of the applied width still
/// asks for no rebuild either (see `live_size_matches_applied`): that is the
/// emulator's async re-wrap timing, not a real resize.
fn resize_rebuild_is_due(
    settle: &mut ResizeSettle,
    now: Instant,
    live_size: Size,
    applied_size: Size,
) -> bool {
    settle.observe(now, live_size) && !live_size_matches_applied(live_size, applied_size)
}

/// How long live geometry must hold still before the final anchor reconciliation.
///
/// If further reflow moves the immediate frame before the display settles,
/// re-query its parked anchor with the immediate frame's width record. This also
/// guards live geometry changes that arrive before an explicit notification.
/// Wall clock rather than poll count, because the draw path and idle poll
/// observe this window at unrelated rates.
const RESIZE_SETTLE_DELAY: Duration = Duration::from_millis(500);

/// Live geometry seen for a pending resize, and since when it has held still.
#[derive(Default)]
struct ResizeSettle {
    size: Option<Size>,
    since: Option<Instant>,
}

impl ResizeSettle {
    /// Records one sighting of `size` and reports whether the geometry has held
    /// still for `RESIZE_SETTLE_DELAY`.
    fn observe(&mut self, now: Instant, size: Size) -> bool {
        if self.size != Some(size) {
            self.size = Some(size);
            self.since = Some(now);
        }
        self.since
            .is_some_and(|since| now.saturating_duration_since(since) >= RESIZE_SETTLE_DELAY)
    }

    /// Restarts the window after a notification announced a new geometry.
    fn restart(&mut self) {
        self.size = None;
        self.since = None;
    }
}

/// Live-geometry jitter tolerance for rebuild decisions.
///
/// VS Code/xterm.js can briefly report a width one column off the applied
/// geometry without any real reflow (async re-wrap timing). Re-anchoring for
/// that phantom size moves the box without moving the transcript and strands
/// the drawn caret; repeated phantom rebuilds accumulate into the slow upward
/// idle drift. A real one-column resize re-wraps at most one column of ink,
/// so painting it at the old width until a larger move arrives is harmless,
/// while a spurious re-anchor is not. Height differences always reflect panel
/// changes and are never ignored.
fn live_size_matches_applied(live: Size, applied: Size) -> bool {
    live == applied || (live.height == applied.height && live.width.abs_diff(applied.width) <= 1)
}

/// Consume a deferred resize at a redraw safe point.
///
/// This must run before height or completion changes clear the old painted-row
/// record. Only an earlier reflow rebuild can consume that geometry; a height
/// rebuild is not a substitute for applying the old width's anchor correction.
fn take_standalone_resize_rebuild(pending_resize_rebuild: &mut bool, reflow_rebuilt: bool) -> bool {
    let needs_rebuild = *pending_resize_rebuild && !reflow_rebuilt;
    *pending_resize_rebuild = false;
    needs_rebuild
}

/// Greedy display-cell wrap for one echo line. Empty input yields one empty row
/// so blank input lines keep their (empty) box row instead of vanishing.
fn wrap_echo_line(line: &str, max_width: usize) -> Vec<String> {
    let mut rows = vec![String::new()];
    let mut row_width = 0;
    for ch in line.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if row_width + ch_width > max_width && row_width > 0 {
            rows.push(String::new());
            row_width = 0;
        }
        rows.last_mut().expect("wrap rows are never empty").push(ch);
        row_width += ch_width;
    }
    rows
}

fn submitted_input_preview_lines(content: &str) -> Vec<String> {
    // The echo prints synchronously to the live terminal (submit path only),
    // so a direct winsize query is the right width source, matching the other
    // geometry reads in this module; the streaming renderer's override chain
    // targets serve children that never reach the interactive submit path.
    let cols = crossterm::terminal::size()
        .map(|(cols, _)| cols as usize)
        .unwrap_or(80);
    submitted_input_preview_lines_with_width(content, cols)
}

/// Post-submit echo of the user input, framed in a rounded box so a user turn
/// stands out in a long transcript at a glance:
///
/// ```text
/// ╭─ ❯ ───────╮
/// │ /history  │
/// ╰───────────╯
/// ```
///
/// The box hugs the content (one padding cell per side) and never exceeds the
/// terminal width: overlong lines wrap inside the box instead of being cut.
/// Widths are display cells (`unicode_width`, ambiguous counted as one like
/// the box-drawing characters themselves), so CJK input keeps the borders
/// aligned; tabs are expanded because a raw tab would break the padding math.
/// The echo body uses the theme's `accent.submitted`, a bright signature warm
/// hue (amber/orange on dark themes, deep purple on light). It is deliberately
/// NOT a white/gray: assistant output is full of near-white body/strong text,
/// so the echoed user input must carry its own hue to stay recognizable as
/// the user's own text. The `❯` title on the top border reuses `accent.prompt`
/// rather than `accent.success` so green keeps meaning "done" everywhere.
fn submitted_input_preview_lines_with_width(content: &str, term_cols: usize) -> Vec<String> {
    // Two borders plus one padding cell per side.
    const OVERHEAD: usize = 4;
    // Keeps at least one dash after the `❯` title on the top border. The floor
    // applies even on absurdly narrow terminals so the box stays aligned (it
    // may overflow the visible width there, where everything is broken anyway).
    const MIN_INNER: usize = 3;
    let max_inner = term_cols.saturating_sub(OVERHEAD).max(MIN_INNER);
    let mut rows: Vec<String> = Vec::new();
    for line in content.lines() {
        rows.extend(wrap_echo_line(&line.replace('\t', "    "), max_inner));
    }
    if rows.is_empty() {
        return Vec::new();
    }
    let inner = rows
        .iter()
        .map(|row| UnicodeWidthStr::width(row.as_str()))
        .max()
        .unwrap_or(0)
        .clamp(MIN_INNER, max_inner);
    let theme = crate::ai::theme::current();
    let marker = theme.accent_prompt;
    let bar = theme.accent_submitted_bar;
    let body = theme.accent_submitted;
    let total = inner + OVERHEAD;
    let mut rendered = Vec::with_capacity(rows.len() + 2);
    rendered.push(format!(
        "\x1b[1m{bar}╭─\x1b[0m \x1b[1m{marker}❯\x1b[0m \x1b[1m{bar}{}╮\x1b[0m",
        "─".repeat(total - 6)
    ));
    for row in &rows {
        let pad = inner - UnicodeWidthStr::width(row.as_str());
        rendered.push(format!(
            "\x1b[1m{bar}│\x1b[0m \x1b[1m{body}{row}{}\x1b[0m \x1b[1m{bar}│\x1b[0m",
            " ".repeat(pad)
        ));
    }
    rendered.push(format!("\x1b[1m{bar}╰{}╯\x1b[0m", "─".repeat(total - 2)));
    rendered
}

fn print_submitted_input_preview(content: &str) {
    for line in submitted_input_preview_lines(content) {
        println!("{line}");
    }
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        let checkpoint_label = self.current_checkpoint_status_label();
        // The streaming side-note listener (Ctrl+G composer) owns stdin in
        // cbreak mode for the whole turn. Preempt it BEFORE touching stdin or
        // termios: the foreground flag makes it stop poll/read, restore termios
        // and release its stdin lease. The handshake is bounded (see
        // `acquire_foreground_stdin`): on timeout the prompt proceeds without
        // the lease rather than wedging the UI forever.
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
            box_row_widths: Vec::new(),
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
            let mut last_painted_terminal_width: Option<u16> = None;
            // Consume this flag after each frame so poll timeouts do not redraw
            // an unchanged input screen. Resize and title events explicitly
            // request the next frame.
            let mut redraw_requested = true;
            // Resize paints immediately, then checks the parked anchor once
            // more after the geometry settles. The second pass uses the new
            // frame's widths, so it cannot count old chrome wraps twice.
            let mut resize_state = PromptResizeState::default();
            // A background title update that arrives while a resize is pending
            // waits for settled geometry instead of painting immediately (see
            // the deferral at the top of the loop); this carries that owed
            // repaint across poll iterations until it may paint.
            let mut title_repaint_owed = false;
            // Idle-drift field log: while the box drifts in a real terminal,
            // record what the loop saw so the cause can be read off the file
            // instead of guessed. File only, never stdout (display-safe); a
            // no-op unless RUST_TOOLS_EDITOR_DRIFT_LOG points at a path.
            let drift_log = |event: &str| {
                let path = std::env::var("RUST_TOOLS_EDITOR_DRIFT_LOG").unwrap_or_default();
                if path.is_empty() {
                    return;
                }
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let mut file = match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    Ok(file) => file,
                    Err(_) => return,
                };
                let _ = <std::fs::File as std::io::Write>::write_fmt(
                    &mut file,
                    format_args!("[{now_ms}] {event}\n"),
                );
            };
            #[cfg(test)]
            let mut fixture_frame_sequence = 0_u64;

            'input: loop {
                // The background only publishes title updates; the terminal is
                // still redrawn by the foreground input loop at this safe draw
                // point.
                let title_changed = self.apply_pending_session_title_updates();
                if title_changed {
                    title_repaint_owed = true;
                    drift_log(&format!(
                        "title update arrived; repaint owed; topic={:?}",
                        self.session_topic
                    ));
                }
                // A title-only update yields to the final resize anchor check.
                // Explicit resize events already owe an immediate frame; input
                // also re-anchors before painting if reconciliation is pending.
                let mut paint_title_now = false;
                if title_repaint_owed && !redraw_requested && resize_state.pending() {
                    let live_size = terminal.backend().size()?;
                    if resize_state.rebuild_is_due(
                        Instant::now(),
                        live_size,
                        last_applied_terminal_size,
                    ) {
                        // Settled: the draw block below rebuilds first.
                        paint_title_now = true;
                        title_repaint_owed = false;
                    }
                    // Else still re-wrapping: keep the request owed and wait
                    // one poll; the next loop top retries this gate.
                } else if title_repaint_owed {
                    // No resize pending, or a redraw is already owed: fold the
                    // stored topic into the normal frame below.
                    paint_title_now = true;
                    title_repaint_owed = false;
                }
                if title_repaint_owed {
                    // Still re-wrapping: note each wait so the settle window is
                    // visible in the log (bounded: clears on paint).
                    if let Ok(live_size) = terminal.backend().size() {
                        drift_log(&format!(
                            "title repaint deferred: live={live_size:?} applied={last_applied_terminal_size:?}"
                        ));
                    }
                }

                if take_redraw_request(&mut redraw_requested, paint_title_now) {
                    let frame_result = (|| -> io::Result<()> {
                        // Consume old-screen width geometry before a completion or
                        // content-height rebuild clears its painted-width record.
                        // A foreground redraw can beat the queued Resize event.
                        let mut terminal_size = last_applied_terminal_size;
                        // Explicit Resize events bypass the settling delay. Query a
                        // fresh anchor and rebuild before the first paint, using the
                        // live width in case a burst already advanced past the event.
                        let live_size = terminal.backend().size()?;
                        if let Some(kind) = resize_state.take_frame_rebuild(
                            Instant::now(),
                            live_size,
                            last_applied_terminal_size,
                        ) {
                            let (rebuilt_area, sampled_size) = rebuild_after_terminal_reflow(
                                &mut terminal,
                                &mut screen,
                                base_viewport_height,
                                fitted_completion_items,
                                last_drawn_area,
                                true,
                            )?;
                            resize_state.finish_rebuild(kind, Instant::now(), sampled_size);
                            last_drawn_area = Some(rebuilt_area);
                            last_applied_terminal_size = sampled_size;
                            terminal_size = sampled_size;
                            force_repaint_next_frame = false;
                        }
                        // When the panel state changes, resize the fixed viewport to
                        // match the height the panel needs.
                        let current_items = completion_panel.as_ref().map(|p| p.items.len());
                        if current_items != fitted_completion_items {
                            let new_height = viewport_height_with_completion(
                                terminal_size.height,
                                base_viewport_height,
                                current_items,
                            );
                            let (rebuilt_area, validated_size) = rebuild_fixed_viewport(
                                &mut terminal,
                                &mut screen,
                                terminal_size,
                                new_height,
                                parked_anchor_offset(last_drawn_area, new_height),
                                ViewportRebuildMode::ReserveMissingRows,
                                last_drawn_area,
                                false,
                            )?;
                            // The cleared top remains the anchor until paint.
                            last_drawn_area = Some(rebuilt_area);
                            terminal_size = validated_size;
                            last_applied_terminal_size = validated_size;
                            fitted_completion_items = current_items;
                            force_repaint_next_frame = false;
                        }

                        let content_height = content_viewport_height(
                            terminal_size.height,
                            base_viewport_height,
                            textarea.lines(),
                        );
                        if content_height != base_viewport_height {
                            let new_height = viewport_height_with_completion(
                                terminal_size.height,
                                content_height,
                                current_items,
                            );
                            let (rebuilt_area, validated_size) = rebuild_fixed_viewport(
                                &mut terminal,
                                &mut screen,
                                terminal_size,
                                new_height,
                                parked_anchor_offset(last_drawn_area, new_height),
                                ViewportRebuildMode::ReserveMissingRows,
                                last_drawn_area,
                                false,
                            )?;
                            last_drawn_area = Some(rebuilt_area);
                            terminal_size = validated_size;
                            last_applied_terminal_size = validated_size;
                            base_viewport_height = content_height;
                            force_repaint_next_frame = false;
                        }

                        let mut drawn_viewport_area = Rect::ZERO;
                        let mut drawn_row_widths: Vec<u16> = Vec::new();
                        // The visible editing caret is drawn into the buffer (see
                        // render.rs), so it cannot jump mid-draw. The hardware
                        // cursor is hidden while ratatui applies the frame diff and
                        // then parked at the box bottom by `park_reflow_anchor`.
                        // Room badge suffix for the title row. The badge was
                        // computed once when the input box opened, so combining
                        // here costs one short format per frame and survives
                        // background title updates that replace the topic
                        // mid-typing.
                        let topic_with_badge: Option<String> =
                            match (self.session_topic.as_deref(), self.room_badge.as_deref()) {
                                (Some(topic), Some(badge)) => Some(format!("{topic} {badge}")),
                                (Some(topic), None) => Some(topic.to_owned()),
                                (None, Some(badge)) => Some(badge.to_owned()),
                                (None, None) => None,
                            };
                        terminal.hide_cursor()?;
                        let prepared_paint_size = prepare_prompt_paint(
                            &mut terminal,
                            &mut terminal_size,
                            |terminal| terminal.backend().size(),
                            |terminal| {
                                let (area, size) = rebuild_after_terminal_reflow(
                                    terminal,
                                    &mut screen,
                                    base_viewport_height,
                                    fitted_completion_items,
                                    last_drawn_area,
                                    true,
                                )?;
                                last_drawn_area = Some(area);
                                last_applied_terminal_size = size;
                                resize_state.finish_rebuild(
                                    ResizeFrameRebuild::Immediate,
                                    Instant::now(),
                                    size,
                                );
                                force_repaint_next_frame = false;
                                Ok(size)
                            },
                        )?;
                        let live_paint_width = if !screen.queries_disabled
                            && screen.replies_settled(Instant::now())
                        {
                            let measured =
                                tracked_cursor_query(&mut screen.pending_cpr_replies, || {
                                    SingleRowFrame::measure_columns(terminal.backend_mut(), || {
                                        cursor::position().map(|(column, _)| column)
                                    })
                                });
                            match measured {
                                Ok(width) => width,
                                Err(err) if PromptEditor::is_cursor_position_timeout(&err) => {
                                    screen.queries_disabled = true;
                                    prepared_paint_size.width.max(1)
                                }
                                Err(err) => return Err(err),
                            }
                        } else {
                            prepared_paint_size.width.max(1)
                        };
                        let paint_width_changed = last_painted_terminal_width
                            .is_some_and(|width| width != live_paint_width);
                        let force_repaint = force_repaint_next_frame || paint_width_changed;
                        let mut output_frame = SingleRowFrame::begin(io::stdout())?;
                        // Establish the cursor group before any visible ink is
                        // written. A resize during the first frame must leave
                        // these rows to the application as well.
                        if !screen.alternate || paint_width_changed {
                            let group_area = terminal.get_frame().area();
                            terminal
                                .backend_mut()
                                .set_cursor_position(Position::new(group_area.x, group_area.y))?;
                            if !screen.alternate {
                                output_frame.prepare_cursor_group(group_area.height as usize)?;
                            }
                            if paint_width_changed {
                                // Clear before drawing the new caret: a diff
                                // can otherwise paint it before clearing the
                                // old position when the PTY width still lags.
                                output_frame.clear_cursor_group(group_area.height as usize)?;
                            }
                        }
                        terminal
                            .draw(|f| {
                                let area = f.area();
                                drawn_viewport_area = area;
                                last_drawn_area = Some(area);
                                let _ = render_multiline_popup_with_checkpoint(
                                    f,
                                    &mut textarea,
                                    status_msg.as_deref(),
                                    completion_panel.as_ref(),
                                    &self.current_agent_label,
                                    &self.current_model_label,
                                    self.model_remote,
                                    &self.current_reasoning_effort_label,
                                    topic_with_badge.as_deref(),
                                    checkpoint_label.as_deref(),
                                    live_paint_width,
                                    &self.current_backend_label,
                                );
                                drawn_row_widths = if force_repaint {
                                    force_frame_repaint(f, &screen.box_row_widths)
                                } else {
                                    painted_row_widths(f.buffer_mut(), area)
                                };
                            })
                            .map_err(|e| io::Error::other(e.to_string()))?;
                        // Accumulate, never shrink: the terminal keeps cells written
                        // by earlier frames, so the widest row content seen since the
                        // rows were last erased is what re-wraps.
                        if screen.box_row_widths.len() < drawn_row_widths.len() {
                            screen.box_row_widths.resize(drawn_row_widths.len(), 0);
                        }
                        for (slot, width) in screen.box_row_widths.iter_mut().zip(&drawn_row_widths)
                        {
                            *slot = (*slot).max(*width);
                        }
                        screen.cursor_group_height = if !screen.alternate {
                            Some(drawn_viewport_area.height)
                        } else {
                            None
                        };
                        park_reflow_anchor(&mut terminal, drawn_viewport_area)?;
                        output_frame.finish()?;
                        last_painted_terminal_width = Some(live_paint_width);
                        screen.viewport_geometry = Some((drawn_viewport_area, terminal_size));
                        let resize_pending = resize_state.pending();
                        drift_log(&format!(
                            "frame drawn: area={drawn_viewport_area:?} title={paint_title_now} resize_pending={resize_pending} applied={last_applied_terminal_size:?}"
                        ));
                        self.notify_first_render();
                        // Emit only after the real draw and anchor writes complete,
                        // so an external PTY driver can synchronize without keys.
                        #[cfg(test)]
                        if std::env::var("RUST_TOOLS_EDITOR_FIXTURE").as_deref() == Ok("1") {
                            use std::io::Write;

                            fixture_frame_sequence += 1;
                            let area = drawn_viewport_area;
                            let metadata = serde_json::json!({
                                "sequence": fixture_frame_sequence,
                                "size": [last_applied_terminal_size.width, last_applied_terminal_size.height],
                                "frame": {"x": area.x, "y": area.y, "width": area.width, "height": area.height},
                                "alternate": screen.alternate,
                                "cursor": textarea.cursor(),
                                "content": textarea.lines().join("\n"),
                            });
                            let mut out = io::stdout().lock();
                            write!(out, "\x1b]777;editor;frame;{metadata}\x07")?;
                            if fixture_frame_sequence == 1 {
                                write!(out, "\x1b]777;editor;ready\x07")?;
                            }
                            out.flush()?;
                        }
                        force_repaint_next_frame = false;
                        Ok(())
                    })();
                    if defer_unstable_prompt_geometry(frame_result)?.is_none() {
                        // Yield to input before trying another bounded frame.
                        redraw_requested = true;
                        resize_state.immediate = true;
                        resize_state.settle.restart();
                    }
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
                    if resize_redraw_after_idle(
                        resize_state.pending(),
                        &self.pending_terminal_events,
                    ) {
                        // One settling pass also reconciles a parked anchor
                        // that moved after the immediate frame, even when the
                        // terminal width now matches that frame's applied size.
                        let live_size = terminal.backend().size()?;
                        if resize_state.rebuild_is_due(
                            Instant::now(),
                            live_size,
                            last_applied_terminal_size,
                        ) {
                            redraw_requested = true;
                            drift_log(&format!(
                                "idle poll: resize due, redraw requested: live={live_size:?} applied={last_applied_terminal_size:?}"
                            ));
                        }
                    } else {
                        // Nothing left to settle: either no rebuild is pending or
                        // a foreground path already consumed the notification.
                        resize_state.settle.restart();
                    }
                    // Keep the active fallback visible and responsive. A cursor
                    // query may block, so recovery waits for the next prompt.
                    continue;
                };
                if let Event::Resize(width, height) = event {
                    // Return directly to the draw block before polling again:
                    // the first frame uses the new width and synchronous chrome
                    // ellipses. Keep one later pass to reconcile cursor movement.
                    redraw_requested |=
                        resize_state.notify_resize(&event, last_applied_terminal_size);
                    drift_log(&format!(
                        "resize notified: {width}x{height} applied={last_applied_terminal_size:?}"
                    ));
                    continue;
                }
                if resize_state.pending() {
                    let live_size = terminal.backend().size()?;
                    let settled = resize_state.rebuild_is_due(
                        Instant::now(),
                        live_size,
                        last_applied_terminal_size,
                    );
                    // Input stays responsive while the final pass is pending,
                    // but its frame must use the live parked row too.
                    if (resize_state.reconcile
                        && live_size_matches_applied(live_size, last_applied_terminal_size))
                        || !live_size_matches_applied(live_size, last_applied_terminal_size)
                    {
                        let prepared =
                            defer_unstable_prompt_geometry(rebuild_after_terminal_reflow(
                                &mut terminal,
                                &mut screen,
                                base_viewport_height,
                                fitted_completion_items,
                                last_drawn_area,
                                false,
                            ))?;
                        if let Some((rebuilt_area, sampled_size)) = prepared {
                            last_drawn_area = Some(rebuilt_area);
                            last_applied_terminal_size = sampled_size;
                            if settled {
                                resize_state.finish_rebuild(
                                    ResizeFrameRebuild::Reconcile,
                                    Instant::now(),
                                    sampled_size,
                                );
                            }
                            drift_log(&format!(
                                "reflow rebuild (pre-input): area={rebuilt_area:?} sampled={sampled_size:?}"
                            ));
                        } else {
                            resize_state.immediate = true;
                        }
                    }
                }

                let input_batch = read_ssh_input_batch(
                    event,
                    is_ssh,
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
                for event in input_batch {
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
                            force_repaint_next_frame |=
                                textarea_logical_char_count(&textarea) < previous_input_len;
                            redraw_requested = true;
                        }
                        EventLoopAction::Submit(result) => break 'input Ok(result),
                    }
                }
            }
        })();

        // A final resize may have arrived after input handling. Clear only a
        // current stored top, or recover it with a bounded query first.
        let _ = terminal.hide_cursor();
        if screen.alternate {
            let _ = terminal.backend_mut().clear_region(BackendClearType::All);
        } else if let Some(area) = last_drawn_area {
            let live_size = terminal.backend().size().ok();
            if screen.viewport_geometry == Some((area, last_applied_terminal_size))
                && live_size == Some(last_applied_terminal_size)
            {
                let _ = clear_fixed_viewport(&mut terminal, Some(area.y));
            } else if !screen.queries_disabled && screen.replies_settled(Instant::now()) {
                if let Some(size) = live_size {
                    // This path clears from the queried top and leaves the
                    // cursor there; it never parks again at stored coordinates.
                    let _ = prepare_anchored_viewport(
                        terminal.backend_mut(),
                        size,
                        area.height,
                        parked_anchor_offset(Some(area), area.height),
                        ViewportRebuildMode::ReflowOnly,
                        true,
                        Some(area),
                        &screen.box_row_widths,
                        screen.cursor_group_height,
                        true,
                        &mut screen.pending_cpr_replies,
                    );
                }
            }
        }
        drop(terminal);
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
    use std::{cell::RefCell, collections::VecDeque, io};

    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use ratatui::{
        Terminal,
        backend::{Backend, TestBackend},
        buffer::Cell,
        layout::{Position, Rect, Size},
        widgets::Paragraph,
    };

    use super::{
        VERIFIED_ANCHOR_ATTEMPTS, ViewportRebuildMode, clear_fixed_viewport, clear_row_range,
        fixed_viewport_area, force_frame_repaint, multiline_viewport_height, park_reflow_anchor,
        parked_anchor_offset, prepare_fixed_viewport, reflowed_anchor_offset,
        submitted_input_preview_lines, submitted_input_preview_lines_with_width,
        take_redraw_request, take_standalone_resize_rebuild, terminal_with_fixed_viewport,
        update_pending_resize_rebuild, verified_cursor_row, viewport_height_with_completion,
    };

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn ssh_text_input_batch_collects_text_and_preserves_resize_boundary() {
        let first = key(KeyCode::Char('a'));
        let second = key(KeyCode::Char('b'));
        let resize = Event::Resize(120, 40);
        let mut input = VecDeque::from([second.clone(), resize.clone()]);
        let mut pending = VecDeque::new();
        let mut replies = VecDeque::new();

        let batch =
            super::read_ssh_input_batch(first.clone(), true, &mut pending, &mut replies, |_| {
                Ok(input.pop_front())
            })
            .unwrap();

        assert_eq!(batch, vec![first, second]);
        assert_eq!(pending, VecDeque::from([resize]));
        assert!(input.is_empty());
    }

    #[test]
    fn ssh_text_input_batch_preserves_enter_and_tab_boundaries() {
        for code in [KeyCode::Enter, KeyCode::Tab] {
            for buffered in [false, true] {
                let first = key(KeyCode::Char('a'));
                let second = key(KeyCode::Char('b'));
                let boundary = key(code);
                let tail = key(KeyCode::Char('x'));
                let mut input = VecDeque::from([second.clone(), boundary.clone(), tail.clone()]);
                let mut pending = if buffered {
                    std::mem::take(&mut input)
                } else {
                    VecDeque::new()
                };
                let mut replies = VecDeque::new();

                let batch = super::read_ssh_input_batch(
                    first.clone(),
                    true,
                    &mut pending,
                    &mut replies,
                    |_| Ok(input.pop_front()),
                )
                .unwrap();

                assert_eq!(batch, vec![first, second]);
                assert_eq!(pending.pop_front(), Some(boundary.clone()));
                let batch = super::read_ssh_input_batch(
                    boundary.clone(),
                    true,
                    &mut pending,
                    &mut replies,
                    |_| panic!("Enter/Tab must not pre-read input after a possible submission"),
                )
                .unwrap();
                assert_eq!(batch, vec![boundary]);
                assert_eq!(
                    pending.into_iter().chain(input).collect::<Vec<_>>(),
                    vec![tail],
                    "code={code:?}, buffered={buffered}"
                );
            }
        }
    }

    #[test]
    fn ssh_text_input_batch_preserves_paste_and_control_boundaries() {
        for boundary in [
            Event::Paste("clipboard text".into()),
            key(KeyCode::Esc),
            key(KeyCode::F(2)),
            key(KeyCode::Left),
            Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Event::Key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)),
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('a'),
                KeyModifiers::NONE,
                crossterm::event::KeyEventKind::Release,
            )),
        ] {
            let first = key(KeyCode::Char('a'));
            let tail = key(KeyCode::Char('x'));
            let mut input = VecDeque::from([boundary.clone(), tail.clone()]);
            let mut pending = VecDeque::new();
            let mut replies = VecDeque::new();
            let batch = super::read_ssh_input_batch(
                first.clone(),
                true,
                &mut pending,
                &mut replies,
                |_| Ok(input.pop_front()),
            )
            .unwrap();

            assert_eq!(batch, vec![first]);
            assert_eq!(pending, VecDeque::from([boundary]));
            assert_eq!(input, VecDeque::from([tail]));
        }
    }

    #[test]
    fn ssh_text_input_batch_with_pending_cpr_never_reads_ahead_or_starts_grace() {
        let first = key(KeyCode::Char('a'));
        let second = key(KeyCode::Char('b'));
        let resize = Event::Resize(90, 30);
        let tail = key(KeyCode::Char('x'));
        let mut input = VecDeque::from([key(KeyCode::Esc), resize.clone()]);
        input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        input.push_back(tail.clone());
        let mut pending = VecDeque::from([second.clone()]);
        let outstanding = std::time::Instant::now();
        let mut replies = VecDeque::from([outstanding]);
        let original_input = input.clone();

        let batch =
            super::read_ssh_input_batch(first.clone(), true, &mut pending, &mut replies, |_| {
                panic!("pending CPR must disable batching before any read or Escape grace")
            })
            .unwrap();

        assert_eq!(batch, vec![first]);
        assert_eq!(pending, VecDeque::from([second.clone()]));
        assert_eq!(replies, VecDeque::from([outstanding]));
        assert_eq!(input, original_input);
        let mut read = |_| Ok(input.pop_front());
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            Some(second)
        );
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            None,
            "the normal reader must still swallow the fragmented CPR"
        );
        assert!(replies.is_empty());
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            Some(resize)
        );
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut replies, &mut read).unwrap(),
            Some(tail)
        );
        assert!(pending.is_empty());
        assert!(input.is_empty());
    }

    #[test]
    fn ssh_text_input_batch_resumes_after_cpr_expires() {
        let first = key(KeyCode::Char('a'));
        let second = key(KeyCode::Char('b'));
        let expired = std::time::Instant::now() - super::CPR_REPLY_WINDOW;
        let mut input = VecDeque::from([second.clone()]);
        let mut pending = VecDeque::new();
        let mut replies = VecDeque::from([expired]);
        let batch = super::read_ssh_input_batch(
            first.clone(),
            true,
            &mut pending,
            &mut replies,
            |timeout| {
                assert!(timeout <= super::SSH_INPUT_BATCH_IDLE);
                assert!(timeout <= super::SSH_INPUT_BATCH_WINDOW);
                Ok(input.pop_front())
            },
        )
        .unwrap();

        assert_eq!(batch, vec![first, second]);
        assert!(replies.is_empty());
        assert!(pending.is_empty());
        assert!(input.is_empty());
    }

    #[test]
    fn local_text_input_is_not_batched() {
        let first = key(KeyCode::Char('a'));
        let mut pending = VecDeque::new();
        let mut replies = VecDeque::new();
        let mut reads = 0;

        let batch =
            super::read_ssh_input_batch(first.clone(), false, &mut pending, &mut replies, |_| {
                reads += 1;
                Ok(Some(key(KeyCode::Char('b'))))
            })
            .unwrap();

        assert_eq!(batch, vec![first]);
        assert_eq!(reads, 0);
        assert!(pending.is_empty());
    }

    /// Exercise the production input loop in an external PTY/emulator. Run this
    /// test alone with --ignored --nocapture --test-threads=1 and
    /// RUST_TOOLS_EDITOR_FIXTURE=1; answer CPR queries and submit with F2.
    /// RUST_TOOLS_REFLOW_DRAFT selects empty, multiline or wide. Optional
    /// RUST_TOOLS_EDITOR_EXPECTED checks the exact submitted text (empty for None).
    /// RUST_TOOLS_EDITOR_REPEAT=1 opens a second prompt on the same editor to
    /// exercise query-disable and orphan-reply state across prompt cleanup.
    /// OSC 777 editor;frame JSON precedes editor;ready on the first draw;
    /// editor;submitted JSON contains the returned content, including null.
    #[test]
    #[ignore]
    fn multiline_editor_pty_fixture() -> io::Result<()> {
        use std::io::{IsTerminal, Write};

        if std::env::var("RUST_TOOLS_EDITOR_FIXTURE").as_deref() != Ok("1") {
            return Ok(());
        }
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Err(io::Error::other("editor fixture requires a PTY"));
        }
        let draft = match std::env::var("RUST_TOOLS_REFLOW_DRAFT").as_deref() {
            Ok("empty") | Err(std::env::VarError::NotPresent) => "",
            Ok("multiline") => "DRAFT_LINE_1 alpha\nDRAFT_LINE_2 beta\nDRAFT_LINE_3 gamma",
            Ok("wide") => {
                "DRAFT_WIDE 中文宽字符测试 mixed ASCII 0123456789 中文尾部\n第二行 abcdefghijklmnopqrstuvwxyz"
            }
            _ => return Err(io::Error::other("unknown RUST_TOOLS_REFLOW_DRAFT")),
        };
        struct RestoreFixture(std::path::PathBuf);
        impl Drop for RestoreFixture {
            fn drop(&mut self) {
                let _ = crossterm::execute!(
                    io::stdout(),
                    crossterm::cursor::Show,
                    crossterm::cursor::SetCursorStyle::DefaultUserShape,
                );
                let _ = crossterm::terminal::disable_raw_mode();
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let root = std::env::temp_dir().join(format!("prompt-editor-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root)?;
        let _restore = RestoreFixture(root.clone());
        let session_store = crate::ai::history::SessionStore::new(&root.join("history"));
        let (subscription, updates) = crate::ai::prompt::subscribe_session_title_updates();
        // Construct an empty REPL history rather than calling PromptEditor::new,
        // which loads the user's ~/.liner_history before paths can be replaced.
        let mut editor = super::PromptEditor {
            editor: Some(
                crate::ai::prompt::LineEditor::with_config(
                    rustyline::Config::builder()
                        .completion_type(rustyline::CompletionType::List)
                        .build(),
                )
                .map_err(|error| io::Error::other(error.to_string()))?,
            ),
            history_path: root.join("repl-history"),
            session_id: "editor-pty-test".to_string(),
            session_store,
            session_image_dir: root.join("disabled-image-directory"),
            pending_prefill: None,
            pending_status_msg: None,
            current_model_label: "fixture-model".to_string(),
            model_remote: false,
            current_agent_label: "fixture-agent".to_string(),
            current_reasoning_effort_label: "high".to_string(),
            current_backend_label: String::new(),
            session_topic: Some("fixture-topic".to_string()),
            room_badge: None,
            remote_checkpoint_status_label: None,
            session_title_update_subscription: subscription,
            session_title_updates: std::sync::Mutex::new(updates),
            first_render_notifier: None,
            cursor_position_queries_disabled: false,
            pending_cpr_replies: VecDeque::new(),
            pending_terminal_events: VecDeque::new(),
            alternate_tail_lines: Vec::new(),
        };
        // Match the existing PTY child: fail image-directory creation before
        // text paste can access the system clipboard.
        std::fs::write(&editor.session_image_dir, b"")?;
        if !draft.is_empty() {
            editor.set_prefill(draft);
        }
        let mut body = String::from("BODY_BEGIN\n");
        for index in 0..96 {
            body.push_str(&format!(
                "MARK{index:02}:English transcript 中文正文保持连续 abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ another-long-ASCII-segment-0123456789 END{index:02}\n"
            ));
        }
        body.push_str("BODY_END\n");
        editor.set_alternate_screen_tail(&body);
        crossterm::terminal::enable_raw_mode()?;
        {
            let mut out = io::stdout().lock();
            write!(out, "{}", body.replace('\n', "\r\n"))?;
            out.flush()?;
        }
        let repeat = std::env::var("RUST_TOOLS_EDITOR_REPEAT").as_deref() == Ok("1");
        for prompt in 0..=usize::from(repeat) {
            if prompt > 0 && !draft.is_empty() {
                editor.set_prefill(draft);
            }
            let content = editor.read_multi_line_tui()?;
            assert!(!crossterm::terminal::is_raw_mode_enabled()?);
            let submitted = serde_json::json!({"content": content, "prompt": prompt});
            {
                let mut out = io::stdout().lock();
                write!(out, "\x1b]777;editor;submitted;{submitted}\x07")?;
                out.flush()?;
            }
            if let Ok(expected) = std::env::var("RUST_TOOLS_EDITOR_EXPECTED") {
                assert_eq!(content.as_deref().unwrap_or(""), expected);
            }
        }
        Ok(())
    }

    /// Real-output fixture for an external PTY/emulator driver. Run only this
    /// ignored test with --nocapture --test-threads=1 and answer its CPR queries.
    /// Resize the PTY and emulator before sending `r`; `q` clears the last frame.
    #[test]
    #[ignore]
    fn multiline_reflow_pty_fixture() -> io::Result<()> {
        use std::io::Write;
        use tui_textarea::{CursorMove, TextArea};

        if std::env::var("RUST_TOOLS_REFLOW_FIXTURE").as_deref() != Ok("1") {
            return Ok(());
        }
        struct RestoreTerminal;
        impl Drop for RestoreTerminal {
            fn drop(&mut self) {
                let _ = crossterm::execute!(
                    io::stdout(),
                    crossterm::cursor::Show,
                    crossterm::cursor::SetCursorStyle::DefaultUserShape,
                );
                let _ = crossterm::terminal::disable_raw_mode();
            }
        }
        crossterm::terminal::enable_raw_mode()?;
        let _restore = RestoreTerminal;
        {
            let mut out = io::stdout().lock();
            write!(out, "BODY_BEGIN\r\n")?;
            for index in 0..96 {
                write!(out, "MARK{index:02}:English transcript 中文正文保持连续 ")?;
                write!(
                    out,
                    "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
                )?;
                write!(
                    out,
                    "another-long-ASCII-segment-0123456789 END{index:02}\r\n"
                )?;
            }
            write!(out, "BODY_END\r\n")?;
            out.flush()?;
        }
        let draft = match std::env::var("RUST_TOOLS_REFLOW_DRAFT").as_deref() {
            Ok("multiline") => "DRAFT_LINE_1 alpha\nDRAFT_LINE_2 beta\nDRAFT_LINE_3 gamma",
            Ok("wide") => {
                "DRAFT_WIDE 中文宽字符测试 mixed ASCII 0123456789 中文尾部\n第二行 abcdefghijklmnopqrstuvwxyz"
            }
            _ => "",
        };
        let mut textarea = if draft.is_empty() {
            TextArea::default()
        } else {
            TextArea::from(draft.lines())
        };
        textarea.move_cursor(CursorMove::Bottom);
        textarea.move_cursor(CursorMove::End);
        let original_lines = textarea.lines().to_vec();
        let original_cursor = textarea.cursor();
        let (_, rows) = crossterm::terminal::size()?;
        let base_height = multiline_viewport_height(rows, Some(draft));
        let mut screen = super::PromptScreen::default();
        let (mut terminal, mut sampled_size) =
            super::build_fixed_terminal(base_height, &mut screen)?;
        let mut area = terminal.get_frame().area();
        let mut phase = "ready";
        let mut pending = VecDeque::new();
        loop {
            let mut caret = None;
            terminal.hide_cursor()?;
            terminal.draw(|frame| {
                area = frame.area();
                caret = super::render_multiline_popup(
                    frame,
                    &mut textarea,
                    None,
                    None,
                    "",
                    "fixture-model",
                    false,
                    "high",
                    Some("fixture-topic"),
                );
                screen.box_row_widths = super::painted_row_widths(frame.buffer_mut(), area);
            })?;
            screen.cursor_group_height = super::preserve_prompt_cursor_group(
                terminal.backend_mut(),
                area,
                sampled_size.width,
            )?
            .then_some(area.height);
            park_reflow_anchor(&mut terminal, area)?;
            assert_eq!(textarea.lines(), original_lines.as_slice());
            assert_eq!(textarea.cursor(), original_cursor);
            let metadata = serde_json::json!({
                "phase": phase,
                "size": [sampled_size.width, sampled_size.height],
                "frame": {"x": area.x, "y": area.y, "width": area.width, "height": area.height},
                "top": area.y,
                "cursor": [original_cursor.0, original_cursor.1],
                "caret": caret.map(|position| [position.x, position.y]),
                "painted_widths": screen.box_row_widths,
                "alternate": screen.alternate,
            });
            {
                let mut out = io::stdout().lock();
                write!(
                    out,
                    "\x1b]777;reflow;meta;{metadata}\x07\x1b]777;reflow;{phase}\x07"
                )?;
                out.flush()?;
            }
            loop {
                let event = super::read_prompt_event(
                    &mut pending,
                    &mut screen.pending_cpr_replies,
                    |timeout| {
                        if crossterm::event::poll(timeout)? {
                            crossterm::event::read().map(Some)
                        } else {
                            Ok(None)
                        }
                    },
                )?;
                match event {
                    Some(Event::Key(key)) if key.kind == crossterm::event::KeyEventKind::Press => {
                        match key.code {
                            KeyCode::Char('r') => {
                                (area, sampled_size) = super::rebuild_after_terminal_reflow(
                                    &mut terminal,
                                    &mut screen,
                                    base_height,
                                    None,
                                    Some(area),
                                    false,
                                )?;
                                phase = "resized";
                                break;
                            }
                            KeyCode::Char('q') => {
                                assert!(clear_fixed_viewport(&mut terminal, Some(area.y))?);
                                assert_eq!(textarea.lines(), original_lines.as_slice());
                                assert_eq!(textarea.cursor(), original_cursor);
                                drop(terminal);
                                drop(screen);
                                let mut out = io::stdout().lock();
                                write!(out, "\x1b]777;reflow;finished\x07")?;
                                out.flush()?;
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                    // The driver coalesces Resize notifications and explicitly
                    // requests exactly one rebuild with `r` after each resize.
                    _ => {}
                }
            }
        }
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
    fn active_alternate_screen_never_allows_recovery_even_after_long_idle() {
        let start = std::time::Instant::now();
        let mut screen = super::PromptScreen::default();
        screen.queries_disabled = true;
        screen.alternate = true;
        let eligible = [start, start + std::time::Duration::from_secs(3600)]
            .map(|now| screen.can_recover_at_prompt_start(now));
        let still_disabled = screen.queries_disabled;
        // No alternate screen was actually entered; keep Drop query-free too.
        screen.alternate = false;
        assert_eq!(eligible, [false, false]);
        assert!(still_disabled);
    }

    #[test]
    fn next_prompt_allows_recovery_after_replies_settle() {
        let start = std::time::Instant::now();
        let mut screen = super::PromptScreen::default();
        assert!(
            !screen.can_recover_at_prompt_start(start),
            "a healthy inline screen does not need recovery"
        );
        // The next prompt inherits query disabling and outstanding replies, but
        // cleanup has already left the previous prompt's alternate screen.
        screen.queries_disabled = true;
        screen.pending_cpr_replies.push_back(start);
        assert!(screen.can_recover_at_prompt_start(start + super::CPR_PROBE_QUIET_PERIOD));
        // Eligibility alone must not retire orphan accounting; the verified
        // anchor or event parser must still account for this reply.
        assert_eq!(screen.pending_cpr_replies.len(), 1);
    }

    #[test]
    fn next_prompt_does_not_recover_while_recent_cpr_is_pending() {
        let start = std::time::Instant::now();
        let mut screen = super::PromptScreen::default();
        screen.queries_disabled = true;
        screen.pending_cpr_replies.push_back(start);
        assert!(!screen.can_recover_at_prompt_start(start));
        assert!(!screen.can_recover_at_prompt_start(
            start + super::CPR_PROBE_QUIET_PERIOD - std::time::Duration::from_millis(1)
        ));
        // The newest outstanding reply controls eligibility, not the oldest.
        screen
            .pending_cpr_replies
            .push_back(start + super::CPR_PROBE_QUIET_PERIOD);
        assert!(!screen.can_recover_at_prompt_start(start + super::CPR_PROBE_QUIET_PERIOD));
        assert!(screen.queries_disabled);
        assert_eq!(screen.pending_cpr_replies.len(), 2);
    }

    #[test]
    fn next_prompt_can_recover_when_orphan_replies_are_consumed_or_expired() {
        let start = std::time::Instant::now();
        let mut screen = super::PromptScreen::default();
        screen.queries_disabled = true;
        screen.pending_cpr_replies.push_back(start);
        let mut input = VecDeque::from([key(KeyCode::Esc)]);
        input.extend("[13;1R".chars().map(|ch| key(KeyCode::Char(ch))));
        assert_eq!(
            super::read_prompt_event(
                &mut VecDeque::new(),
                &mut screen.pending_cpr_replies,
                |_| { Ok(input.pop_front()) }
            )
            .unwrap(),
            None
        );
        assert!(screen.can_recover_at_prompt_start(start));
        screen.pending_cpr_replies.push_back(start);
        assert!(screen.can_recover_at_prompt_start(start + super::CPR_REPLY_WINDOW));
        assert!(screen.pending_cpr_replies.is_empty());
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
    fn pending_resize_is_due_for_an_idle_redraw_without_keyboard_input() {
        let mut redraw_requested = true;

        assert!(take_redraw_request(&mut redraw_requested, false));
        assert!(!take_redraw_request(&mut redraw_requested, false));

        let mut pending_resize_rebuild = true;
        redraw_requested |=
            super::resize_redraw_after_idle(pending_resize_rebuild, &VecDeque::new());
        assert!(take_redraw_request(&mut redraw_requested, false));
        assert!(take_standalone_resize_rebuild(
            &mut pending_resize_rebuild,
            false
        ));
        assert!(!pending_resize_rebuild);
        redraw_requested |=
            super::resize_redraw_after_idle(pending_resize_rebuild, &VecDeque::new());
        assert!(!take_redraw_request(&mut redraw_requested, false));
    }

    #[test]
    fn idle_resize_redraw_drains_notifications_buffered_by_cursor_reply() {
        let mut pending_events = VecDeque::from([Event::Resize(60, 24)]);
        assert!(!super::resize_redraw_after_idle(true, &pending_events));
        assert_eq!(pending_events.pop_front(), Some(Event::Resize(60, 24)));
        assert!(super::resize_redraw_after_idle(true, &pending_events));
        assert!(!super::resize_redraw_after_idle(false, &pending_events));
    }

    #[test]
    fn resize_burst_rebuilds_once_after_idle_and_ignores_duplicates() {
        let applied = ratatui::layout::Size::new(80, 24);
        let mut pending = false;
        let mut notifications = VecDeque::from([
            Event::Resize(70, 24),
            Event::Resize(60, 24),
            Event::Resize(60, 24),
        ]);
        while let Some(Event::Resize(width, height)) = notifications.pop_front() {
            update_pending_resize_rebuild(
                &mut pending,
                applied,
                ratatui::layout::Size::new(width, height),
            );
            if !notifications.is_empty() {
                assert!(!super::resize_redraw_after_idle(pending, &notifications));
            }
        }
        assert!(super::resize_redraw_after_idle(pending, &notifications));
        assert!(take_standalone_resize_rebuild(&mut pending, false));
        assert!(!take_standalone_resize_rebuild(&mut pending, false));

        let applied = ratatui::layout::Size::new(60, 24);
        update_pending_resize_rebuild(&mut pending, applied, applied);
        assert!(!super::resize_redraw_after_idle(pending, &notifications));
    }

    #[test]
    fn duplicate_resize_notification_does_not_schedule_rebuild() {
        let applied = ratatui::layout::Size::new(80, 24);
        let mut pending = false;

        update_pending_resize_rebuild(&mut pending, applied, applied);
        assert!(!pending);

        update_pending_resize_rebuild(&mut pending, applied, ratatui::layout::Size::new(79, 24));
        assert!(!pending);

        update_pending_resize_rebuild(&mut pending, applied, ratatui::layout::Size::new(78, 24));
        assert!(pending);

        // A resize burst that returns to the applied dimensions before the
        // rebuild must cancel the stale intermediate notification.
        update_pending_resize_rebuild(&mut pending, applied, applied);
        assert!(!pending);
    }

    #[test]
    fn only_fresh_anchors_and_deferred_rebuilds_verify_the_reply_stream() {
        use super::{ViewportRebuildMode, reply_stream_is_verified};

        // A fresh anchor is taken as the box top, so one reply is never trusted.
        assert!(reply_stream_is_verified(
            0,
            ViewportRebuildMode::ReserveMissingRows,
            false
        ));
        // Height rebuilds run on every panel change and keep one round trip.
        assert!(!reply_stream_is_verified(
            3,
            ViewportRebuildMode::ReserveMissingRows,
            false
        ));
        // The keystroke-driven reflow rebuild keeps it too: the user is waiting.
        assert!(!reply_stream_is_verified(
            3,
            ViewportRebuildMode::ReflowOnly,
            false
        ));
        // A rebuild at offset zero is still a rebuild, not a fresh anchor.
        assert!(!reply_stream_is_verified(
            0,
            ViewportRebuildMode::ReflowOnly,
            false
        ));
        // The deferred reflow rebuild opts in: nobody waits for that frame.
        assert!(reply_stream_is_verified(
            3,
            ViewportRebuildMode::ReflowOnly,
            true
        ));
        // The flag forces verification whatever the mode would have chosen.
        assert!(reply_stream_is_verified(
            3,
            ViewportRebuildMode::ReserveMissingRows,
            true
        ));
    }

    #[test]
    fn idle_resize_waits_for_a_settled_geometry_before_rebuilding() {
        use ratatui::layout::Size;
        use std::time::{Duration, Instant};

        let size = Size::new(80, 24);
        let start = Instant::now();
        let mut settle = super::ResizeSettle::default();
        // Final reconciliation waits for stable geometry; the immediate resize
        // frame bypasses this window.
        assert!(!settle.observe(start, size));
        assert!(!settle.observe(start + Duration::from_millis(250), size));
        assert!(settle.observe(start + Duration::from_millis(600), size));
        // Geometry that moves again restarts the window.
        let moved = start + Duration::from_millis(700);
        assert!(!settle.observe(moved, Size::new(80, 25)));
        assert!(!settle.observe(moved + Duration::from_millis(250), Size::new(80, 25)));
        assert!(settle.observe(moved + Duration::from_millis(600), Size::new(80, 25)));
        // A fresh notification must not inherit sightings of the same size.
        settle.restart();
        assert!(!settle.observe(moved + Duration::from_millis(600), Size::new(80, 25)));
    }

    #[test]
    fn foreground_redraw_waits_for_a_still_geometry_before_rebuilding() {
        use ratatui::layout::Size;
        use std::time::{Duration, Instant};

        let applied = Size::new(80, 24);
        let narrowed = Size::new(60, 24);
        let start = Instant::now();
        let mut settle = super::ResizeSettle::default();
        // A foreground redraw can beat the queued Resize event, and the emulator
        // re-wraps its scrollback after the size it announces. Rebuilding on that
        // first sight computes the clear from the pre-reflow layout and paints the
        // box where the coming re-wrap then moves it, leaving the previous frame
        // on screen above the new one.
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start,
            narrowed,
            applied
        ));
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(250),
            narrowed,
            applied
        ));
        assert!(super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(600),
            narrowed,
            applied
        ));
        // A geometry that matches the applied one never owes a rebuild, and seeing
        // it again restarts the window for the next move.
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(900),
            applied,
            applied
        ));
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(1000),
            narrowed,
            applied
        ));
        assert!(super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(1600),
            narrowed,
            applied
        ));
    }

    #[test]
    fn one_column_live_jitter_never_owes_a_rebuild() {
        use ratatui::layout::Size;
        use std::time::{Duration, Instant};

        let applied = Size::new(80, 24);
        assert!(super::live_size_matches_applied(applied, applied));
        assert!(super::live_size_matches_applied(Size::new(79, 24), applied));
        assert!(super::live_size_matches_applied(Size::new(81, 24), applied));
        assert!(!super::live_size_matches_applied(
            Size::new(78, 24),
            applied
        ));
        // Height moves are always real.
        assert!(!super::live_size_matches_applied(
            Size::new(80, 25),
            applied
        ));
        assert!(!super::live_size_matches_applied(
            Size::new(79, 25),
            applied
        ));

        // Even a settled 1-column sighting owes no rebuild.
        let start = Instant::now();
        let mut settle = super::ResizeSettle::default();
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start,
            Size::new(79, 24),
            applied
        ));
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(600),
            Size::new(79, 24),
            applied
        ));
        // A real move observed through the same window still rebuilds.
        assert!(!super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(700),
            Size::new(70, 24),
            applied
        ));
        assert!(super::resize_rebuild_is_due(
            &mut settle,
            start + Duration::from_millis(1300),
            Size::new(70, 24),
            applied
        ));
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
            &mut VecDeque::new(),
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
            &mut VecDeque::new(),
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
        // The record says the row was written four columns wide, so the wipe
        // reaches the ghost.
        terminal
            .draw(|frame| {
                force_frame_repaint(frame, &[4]);
            })
            .unwrap();

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
            &mut VecDeque::new(),
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
                &mut VecDeque::new(),
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
    fn width_reflow_keeps_the_first_old_prompt_row_even_after_bottom_flush() {
        // The old bottom-pinning policy moved a widening prompt below its
        // recovered first row and opened a gap. Its replacement always keeps
        // the live anchor's top, whether the old prompt touched the bottom or not.
        for (previous, size, cursor_row, expected_top) in [
            (Rect::new(0, 20, 80, 4), Size::new(120, 24), 13, 10),
            (Rect::new(0, 12, 80, 4), Size::new(120, 24), 8, 5),
            (Rect::new(0, 20, 120, 4), Size::new(80, 24), 20, 14),
            (Rect::new(0, 20, 80, 4), Size::new(80, 30), 23, 20),
        ] {
            let offset =
                reflowed_anchor_offset(Some(previous), 4, &[previous.width; 4], size.width);
            let (area, appended) = fixed_viewport_area(
                size,
                Position::new(0, cursor_row),
                4,
                offset,
                ViewportRebuildMode::ReflowOnly,
            );
            assert_eq!(area, Rect::new(0, expected_top, size.width, 4));
            assert_eq!(appended, 0);
        }
    }

    #[test]
    fn reflow_only_viewport_clamps_an_oversized_box_without_scrolling() {
        // Bottom gluing no longer determines placement. The existing fixed
        // viewport bounds still keep an oversized frame inside the screen.
        let (area, appended) = fixed_viewport_area(
            Size::new(20, 24),
            Position::new(0, 23),
            30,
            29,
            ViewportRebuildMode::ReflowOnly,
        );
        assert_eq!(area, Rect::new(0, 0, 20, 24));
        assert_eq!(appended, 0);
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
                &mut VecDeque::new(),
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
            &mut VecDeque::new(),
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
        // Keep the row the parked anchor recovers even if the old box was
        // flush with the bottom: pinning it down again would open a gap.
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
            &mut VecDeque::new(),
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
    fn prepaint_guard_tolerates_one_column_jitter() {
        let original = Size::new(80, 24);
        let mut applied = original;
        let mut reanchors = 0;
        for width in [79, 81, 80, 79] {
            let mut live = Size::new(width, 24);
            let paint_size = super::prepare_prompt_paint(
                &mut live,
                &mut applied,
                |live| Ok(*live),
                |live| {
                    reanchors += 1;
                    Ok(*live)
                },
            )
            .unwrap();
            assert_eq!(paint_size, live);
            assert_eq!(applied, original);
        }
        assert_eq!(reanchors, 0);

        for live in [Size::new(78, 24), Size::new(80, 25)] {
            let mut live = live;
            super::prepare_prompt_paint(
                &mut live,
                &mut applied,
                |live| Ok(*live),
                |live| {
                    reanchors += 1;
                    Ok(*live)
                },
            )
            .unwrap();
            assert_eq!(applied, live);
        }
        assert_eq!(reanchors, 2);

        let mut resize = super::PromptResizeState::default();
        assert!(!resize.notify_resize(&Event::Resize(79, 24), original));
        assert!(!resize.notify_resize(&Event::Resize(81, 24), original));
        assert!(!resize.pending());
        let now = std::time::Instant::now();
        resize.finish_rebuild(super::ResizeFrameRebuild::Immediate, now, original);
        assert!(!resize.rebuild_is_due(now, Size::new(79, 24), original));
        assert_eq!(
            resize.take_frame_rebuild(
                now + super::RESIZE_SETTLE_DELAY,
                Size::new(79, 24),
                original,
            ),
            Some(super::ResizeFrameRebuild::Reconcile),
        );
    }

    #[test]
    fn empty_input_collapses_grown_viewport() {
        let lines = (0..8)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>();
        let grown = super::content_viewport_height(24, 6, &lines);
        assert_eq!(grown, 11);
        assert_eq!(
            super::content_viewport_height(24, grown, &["remaining".into()]),
            grown
        );
        assert_eq!(
            super::content_viewport_height(24, grown, &[String::new(), String::new()]),
            grown
        );

        let mut textarea = tui_textarea::TextArea::from(lines);
        textarea.select_all();
        textarea.delete_char();
        assert_eq!(textarea.lines(), &[String::new()]);
        let collapsed = super::content_viewport_height(24, grown, textarea.lines());
        assert_eq!(collapsed, 6);
        assert_eq!(
            super::content_viewport_height(4, grown, textarea.lines()),
            2
        );

        let mut backend = TestBackend::new(80, 24);
        backend.set_cursor_position(Position::new(0, 23)).unwrap();
        backend
            .draw(std::iter::once((3, 22, &Cell::new("ghost"))))
            .unwrap();
        let area = prepare_fixed_viewport(
            &mut backend,
            Size::new(80, 24),
            collapsed,
            parked_anchor_offset(Some(Rect::new(0, 13, 80, grown)), collapsed),
            ViewportRebuildMode::ReserveMissingRows,
            true,
            &mut VecDeque::new(),
        )
        .unwrap();
        assert_eq!(area, Rect::new(0, 13, 80, 6));
        assert!(
            backend
                .buffer()
                .content
                .iter()
                .all(|cell| cell.symbol() == " ")
        );
        assert_eq!(backend.scrollback().area.height, 0);
    }

    #[test]
    fn multiline_viewport_height_scales_with_terminal() {
        // Empty input: viewport = 3 input lines + chrome(3) = 6. The left-edge
        // marker bar occupies a column, not a row, so it adds no height.
        assert_eq!(multiline_viewport_height(30, None), 6);
        assert_eq!(multiline_viewport_height(30, Some("")), 6);
        // Prefilled but shorter than base: keep the base size. base textarea is
        // 7 content lines; 7 + chrome(3) = 10.
        assert_eq!(multiline_viewport_height(30, Some("one line")), 10);
        // Small terminal: terminal=12, available=10; empty input still keeps its
        // 6-line box (3 content + chrome).
        assert_eq!(multiline_viewport_height(12, None), 6);
        // On large terminals empty input still keeps the same compact box.
        assert_eq!(multiline_viewport_height(40, None), 6);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        // terminal=40: available=38, base_textarea=7, content=20→clamp(7,7)=7,
        // viewport = 7 + chrome(3) = 10.
        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 10);
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
    fn submitted_input_preview_draws_box_around_single_line() {
        let marker = crate::ai::theme::current().accent_prompt;
        let bar = crate::ai::theme::current().accent_submitted_bar;
        let body = crate::ai::theme::current().accent_submitted;
        let bold = "\x1b[1m";
        let reset = "\x1b[0m";
        // "hello" is 5 cells wide: inner 5, total 9, 3 top dashes, 7 bottom.
        assert_eq!(
            submitted_input_preview_lines_with_width("hello", 80),
            vec![
                format!("{bold}{bar}╭─{reset} {bold}{marker}❯{reset} {bold}{bar}───╮{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}hello{reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}╰───────╯{reset}"),
            ]
        );
    }

    #[test]
    fn submitted_input_preview_pads_short_rows_to_content_width() {
        let marker = crate::ai::theme::current().accent_prompt;
        let bar = crate::ai::theme::current().accent_submitted_bar;
        let body = crate::ai::theme::current().accent_submitted;
        let bold = "\x1b[1m";
        let reset = "\x1b[0m";
        // Inner width follows the longest row ("world", 5); "hi" gets 3 pads.
        assert_eq!(
            submitted_input_preview_lines_with_width("hi\nworld", 80),
            vec![
                format!("{bold}{bar}╭─{reset} {bold}{marker}❯{reset} {bold}{bar}───╮{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}hi   {reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}world{reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}╰───────╯{reset}"),
            ]
        );
    }

    #[test]
    fn submitted_input_preview_aligns_cjk_borders() {
        let marker = crate::ai::theme::current().accent_prompt;
        let bar = crate::ai::theme::current().accent_submitted_bar;
        let body = crate::ai::theme::current().accent_submitted;
        let bold = "\x1b[1m";
        let reset = "\x1b[0m";
        // "你好" spans 4 cells, so inner is 4 and "hi" pads with 2 spaces.
        assert_eq!(
            submitted_input_preview_lines_with_width("hi\n你好", 80),
            vec![
                format!("{bold}{bar}╭─{reset} {bold}{marker}❯{reset} {bold}{bar}──╮{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}hi  {reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}你好{reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}╰──────╯{reset}"),
            ]
        );
    }

    #[test]
    fn submitted_input_preview_wraps_overlong_lines_inside_box() {
        let marker = crate::ai::theme::current().accent_prompt;
        let bar = crate::ai::theme::current().accent_submitted_bar;
        let body = crate::ai::theme::current().accent_submitted;
        let bold = "\x1b[1m";
        let reset = "\x1b[0m";
        // 10 columns leave 6 inner cells, so 10 ASCII cells wrap into 6 + 4.
        assert_eq!(
            submitted_input_preview_lines_with_width("abcdefghij", 10),
            vec![
                format!("{bold}{bar}╭─{reset} {bold}{marker}❯{reset} {bold}{bar}────╮{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}abcdef{reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}│{reset} {bold}{body}ghij  {reset} {bold}{bar}│{reset}"),
                format!("{bold}{bar}╰────────╯{reset}"),
            ]
        );
    }

    #[test]
    fn submitted_input_preview_renders_nothing_for_empty_content() {
        assert!(submitted_input_preview_lines_with_width("", 80).is_empty());
        assert!(submitted_input_preview_lines("").is_empty());
    }

    #[test]
    fn submitted_input_preview_expands_tabs_before_measuring() {
        // "a\tb" becomes "a    b" (7 cells) so the borders stay aligned.
        let out = submitted_input_preview_lines_with_width("a\tb", 80);
        assert_eq!(out.len(), 3);
        assert!(
            out[1].contains("a    b"),
            "tab must expand to spaces: {}",
            out[1]
        );
    }

    /// Scripted stand-in for a terminal answering DSR queries: replies left over
    /// from earlier queries are handed out first, then the row the protocol last
    /// commanded (the saved row again once a restore runs).
    struct ScriptedAnchor {
        stale: VecDeque<u16>,
        queued: Option<VecDeque<u16>>,
        commanded: u16,
        true_row: u16,
        failing: bool,
        timeout_at: Option<u32>,
        queries: u32,
        calls: Vec<String>,
    }

    impl ScriptedAnchor {
        fn new(stale: impl IntoIterator<Item = u16>, true_row: u16) -> Self {
            Self {
                stale: stale.into_iter().collect(),
                queued: None,
                commanded: true_row,
                true_row,
                failing: false,
                timeout_at: None,
                queries: 0,
                calls: Vec::new(),
            }
        }

        /// Makes every query fail the way crossterm reports a DSR timeout.
        fn fail_queries(&mut self) {
            self.failing = true;
        }

        fn record(&mut self, call: impl Into<String>) {
            self.calls.push(call.into());
        }

        fn answer(&mut self) -> io::Result<u16> {
            self.queries += 1;
            self.record("query");
            // FIFO mode retains every issued query's reply, including replies
            // left unread when a query times out or consumes an older answer.
            if let Some(queued) = &mut self.queued {
                queued.push_back(self.commanded);
            }
            if self.failing || self.timeout_at == Some(self.queries) {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "The cursor position could not be read within a normal duration",
                ));
            }
            Ok(self
                .stale
                .pop_front()
                .or_else(|| self.queued.as_mut().and_then(VecDeque::pop_front))
                .unwrap_or(self.commanded))
        }
    }

    /// Drives `verified_cursor_row` against `terminal`, returning its result
    /// together with the terminal's call log.
    fn run_verified_anchor(
        terminal: &RefCell<ScriptedAnchor>,
        last_screen_row: u16,
    ) -> (io::Result<u16>, Vec<String>) {
        run_verified_anchor_with_pending(terminal, last_screen_row, &mut VecDeque::new())
    }

    fn run_verified_anchor_with_pending(
        terminal: &RefCell<ScriptedAnchor>,
        last_screen_row: u16,
        pending: &mut VecDeque<std::time::Instant>,
    ) -> (io::Result<u16>, Vec<String>) {
        let mut save = || -> io::Result<()> {
            terminal.borrow_mut().record("save");
            Ok(())
        };
        let mut restore = || -> io::Result<()> {
            let mut terminal = terminal.borrow_mut();
            terminal.record("restore");
            terminal.commanded = terminal.true_row;
            Ok(())
        };
        let mut set_row = |row: u16| -> io::Result<()> {
            let mut terminal = terminal.borrow_mut();
            terminal.record(format!("set:{row}"));
            terminal.commanded = row;
            Ok(())
        };
        let mut query_row = || terminal.borrow_mut().answer();
        let result = verified_cursor_row(
            &mut query_row,
            &mut save,
            &mut restore,
            &mut set_row,
            last_screen_row,
            pending,
        );
        (result, terminal.borrow().calls.clone())
    }

    #[test]
    fn verified_anchor_rejects_stale_replies_left_by_an_earlier_query() {
        // Two replies reporting the previous prompt's parked bottom row (9) are
        // still in the stream when the fresh anchor queries, while the cursor the
        // box must anchor to really sits on row 17.
        let terminal = RefCell::new(ScriptedAnchor::new([9, 9], 17));
        let (row, calls) = run_verified_anchor(&terminal, 23);

        // The first attempt mismatched (9 where it commanded row 0), so the row
        // comes from the verified second attempt — never the stale 9.
        assert_eq!(row.unwrap(), 17);
        assert_eq!(terminal.borrow().queries, 5);
        // Both attempts put the cursor back, including the rejected one.
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.as_str() == "restore")
                .count(),
            2
        );
    }

    #[test]
    fn verified_anchor_answers_immediately_when_the_link_is_clean() {
        let terminal = RefCell::new(ScriptedAnchor::new(VecDeque::new(), 17));
        let (row, calls) = run_verified_anchor(&terminal, 23);

        assert_eq!(row.unwrap(), 17);
        // One save, the two probes, one restore, then the trusted query. The
        // rows the probes command are consecutive rows chosen for the probe,
        // covered by `anchor_probe_rows_only_probe_on_rows_inside_the_screen`;
        // only the call order is pinned here.
        assert_eq!(calls.len(), 7);
        assert_eq!(calls[0], "save");
        assert_eq!(calls[2], "query");
        assert_eq!(calls[4], "query");
        assert_eq!(calls[5], "restore");
        assert_eq!(calls[6], "query");
        assert!(calls[1].starts_with("set:") && calls[3].starts_with("set:"));
    }

    #[test]
    fn verified_anchor_refreshes_pending_replies_before_old_entries_expire() {
        let count = 2 * usize::from(VERIFIED_ANCHOR_ATTEMPTS);
        let now = std::time::Instant::now();
        let old = now - super::CPR_REPLY_WINDOW + std::time::Duration::from_millis(500);
        let mut screen = super::PromptScreen::default();
        screen.pending_cpr_replies = VecDeque::from(vec![old; count]);
        assert!(screen.replies_settled(now));

        // Each query consumes an old response but leaves its own response in
        // flight. The net count stays constant; its timestamps must be renewed.
        let terminal = RefCell::new(ScriptedAnchor::new(vec![8; count], 17));
        terminal.borrow_mut().queued = Some(VecDeque::new());
        let (row, _) =
            run_verified_anchor_with_pending(&terminal, 23, &mut screen.pending_cpr_replies);
        assert!(row.is_err());
        assert_eq!(terminal.borrow().queries as usize, count);
        screen.queries_disabled = true;
        assert_eq!(screen.pending_cpr_replies.len(), count);
        assert!(screen.pending_cpr_replies.iter().all(|at| *at >= now));
        super::prune_expired_cpr_replies(
            &mut screen.pending_cpr_replies,
            now + std::time::Duration::from_secs(1),
        );
        assert_eq!(screen.pending_cpr_replies.len(), count);

        // Deliver the actual newly queued replies as fragmented CPRs after the
        // old timestamps expired, followed by ordinary typing and submission.
        let queued = terminal.borrow_mut().queued.take().unwrap();
        assert_eq!(queued.len(), count);
        let mut input = VecDeque::new();
        for row in queued {
            input.push_back(key(KeyCode::Esc));
            input.extend(
                format!("[{};1R", row + 1)
                    .chars()
                    .map(|ch| key(KeyCode::Char(ch))),
            );
        }
        input.extend([key(KeyCode::Char('x')), key(KeyCode::Esc)]);
        let mut pending = VecDeque::new();
        for _ in 0..count {
            assert_eq!(
                super::read_prompt_event(&mut pending, &mut screen.pending_cpr_replies, |_| {
                    Ok(input.pop_front())
                })
                .unwrap(),
                None
            );
        }
        assert!(screen.pending_cpr_replies.is_empty());
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut screen.pending_cpr_replies, |_| {
                Ok(input.pop_front())
            })
            .unwrap(),
            Some(key(KeyCode::Char('x')))
        );
        let mut reads = 0;
        assert_eq!(
            super::read_prompt_event(&mut pending, &mut screen.pending_cpr_replies, |_| {
                reads += 1;
                Ok(input.pop_front())
            })
            .unwrap(),
            Some(key(KeyCode::Esc))
        );
        assert_eq!(reads, 1);
        assert!(input.is_empty());
    }

    #[test]
    fn verified_anchor_timeout_counts_each_unanswered_query_across_retries() {
        let terminal = RefCell::new(ScriptedAnchor::new([], 17));
        terminal.borrow_mut().fail_queries();
        let mut pending = VecDeque::new();
        for expected in 1..=2 {
            let (row, calls) = run_verified_anchor_with_pending(&terminal, 23, &mut pending);
            assert!(row.is_err());
            assert_eq!(pending.len(), expected);
            assert_eq!(calls.last().map(String::as_str), Some("restore"));
        }
    }

    #[test]
    fn verified_anchor_successful_retry_clears_prior_pending_replies() {
        let terminal = RefCell::new(ScriptedAnchor::new([], 17));
        terminal.borrow_mut().timeout_at = Some(1);
        let mut pending = VecDeque::new();
        assert!(
            run_verified_anchor_with_pending(&terminal, 23, &mut pending)
                .0
                .is_err()
        );
        assert_eq!(pending.len(), 1);

        // The timed-out reply was lost. A matching probe pair on the retry
        // establishes a clean stream and retires the old conservative entry.
        assert_eq!(
            run_verified_anchor_with_pending(&terminal, 23, &mut pending)
                .0
                .unwrap(),
            17
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn verified_anchor_final_query_timeout_keeps_only_its_reply() {
        let terminal = RefCell::new(ScriptedAnchor::new([], 17));
        terminal.borrow_mut().timeout_at = Some(3);
        let mut pending = VecDeque::from([std::time::Instant::now(); 2]);
        let (row, calls) = run_verified_anchor_with_pending(&terminal, 23, &mut pending);
        assert!(row.is_err());
        assert_eq!(terminal.borrow().queries, 3);
        assert_eq!(pending.len(), 1);
        assert_eq!(calls[5], "restore");
    }

    #[test]
    fn verified_anchor_single_row_query_does_not_claim_a_verified_drain() {
        let terminal = RefCell::new(ScriptedAnchor::new([], 0));
        let mut pending = VecDeque::from([std::time::Instant::now(); 2]);
        let (row, _) = run_verified_anchor_with_pending(&terminal, 0, &mut pending);
        assert_eq!(row.unwrap(), 0);
        assert_eq!(terminal.borrow().queries, 1);
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn tracked_cursor_query_at_capacity_consumes_only_one_reply() {
        let old = std::time::Instant::now() - std::time::Duration::from_secs(5);
        let mut pending = VecDeque::from(vec![old; super::MAX_PENDING_CPR_REPLIES]);
        super::tracked_cursor_query(&mut pending, || Ok::<_, io::Error>(17)).unwrap();
        assert_eq!(pending.len(), super::MAX_PENDING_CPR_REPLIES);
        assert_eq!(
            pending.iter().filter(|at| **at == old).count(),
            pending.len() - 1
        );
        assert!(*pending.back().unwrap() > old);

        let result = super::tracked_cursor_query(&mut pending, || {
            Err::<u16, _>(io::Error::from(io::ErrorKind::TimedOut))
        });
        assert!(result.is_err());
        assert_eq!(pending.len(), super::MAX_PENDING_CPR_REPLIES);
    }

    #[test]
    fn verified_anchor_gives_up_on_a_link_that_never_answers() {
        // Every probe is answered with row 8 whatever row was commanded, so no
        // attempt can show that the reply stream was drained.
        let stale = std::iter::repeat(8u16).take(2 * usize::from(VERIFIED_ANCHOR_ATTEMPTS));
        let terminal = RefCell::new(ScriptedAnchor::new(stale, 17));
        let (row, calls) = run_verified_anchor(&terminal, 23);

        assert!(crate::ai::prompt::PromptEditor::is_cursor_position_timeout(
            &row.unwrap_err()
        ));
        assert_eq!(
            terminal.borrow().queries,
            2 * u32::from(VERIFIED_ANCHOR_ATTEMPTS)
        );
        // Every attempt put the cursor back before the next probe ran.
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.as_str() == "restore")
                .count(),
            usize::from(VERIFIED_ANCHOR_ATTEMPTS)
        );

        // A link that does not answer at all makes the query itself time out: the
        // attempt still restores the cursor before that error escapes.
        let terminal = RefCell::new(ScriptedAnchor::new(VecDeque::new(), 17));
        terminal.borrow_mut().fail_queries();
        let (row, calls) = run_verified_anchor(&terminal, 23);

        assert!(crate::ai::prompt::PromptEditor::is_cursor_position_timeout(
            &row.unwrap_err()
        ));
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], "save");
        assert!(calls[1].starts_with("set:"));
        assert_eq!(calls[2], "query");
        assert_eq!(calls[3], "restore");
    }
}
