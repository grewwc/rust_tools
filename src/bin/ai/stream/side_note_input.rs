// side_note_input.rs — same-terminal side-note input listener during streaming output (Ctrl+G)
//
// While the main agent is streaming model output, stdin is idle (the interactive input
// box is not open). This module polls for Ctrl+G (0x07) in cbreak mode (ICANON/ECHO
// off, ISIG/OPOST kept): on hit it opens a one-line input; Esc / F2 / Alt+Enter submit
// the draft as a side-note with `from="user"` into the foreground queue, and the next
// iteration injects it into the LLM context (turn loop drains via
// `driver::side_note::drain_side_notes`).
// A `/bg`-family draft is handled by the listener itself after it releases terminal
// input ownership: the same process becomes a shell background job without cancelling
// the active request, then an English continuation note is queued for the next turn
// iteration. The literal command is never injected into the LLM context.
// Pressing Ctrl+G again while typing discards the current draft; Enter does not send
// (aligned with the main input's "Enter=newline, Esc/F2 submit" key semantics).
//
// Design notes:
// - ISIG kept: Ctrl+C still goes through the existing SIGINT interrupt path without
//   breaking the streaming interrupt semantics.
// - OPOST kept: the main agent's rendered output `\n` → `\r\n` conversion is unaffected.
// - ECHO off with self-managed echo: backspace can erase by character width, and
//   Ctrl+G itself produces no echo to interfere with rendering.
// - Short poll + stop flag: when stream_response ends (any return path) the guard
//   requests exit and waits for the terminal to be restored, avoiding leftover cbreak
//   state affecting the later input box.
use std::{
    collections::VecDeque,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::ai::{
    driver::{runtime_ctx, side_note::push_side_note},
    history::SuspendedSessionStore,
    theme::{self, RESET},
};
use crate::commonw::prompt::{acquire_background_stdin, foreground_stdin_requested};

/// Ctrl+G (BEL)
const CTRL_G: u8 = 0x07;
/// Poll interval in milliseconds; also bounds how quickly the stop flag is observed.
const POLL_MS: i32 = 50;

/// Upper bound for one `CSI 18 t` round trip inside the listener loop.
const WIDTH_SERVICE_WAIT_MS: u64 = 60;
/// Freshness window of the cached emulator answer before ioctl becomes authoritative again.
const WIDTH_REFRESH_TTL_MS: u64 = 250;
/// Consecutive failed terminal queries that disable probing for this stdin lease.
const WIDTH_REFRESH_TIMEOUT_BUDGET: u8 = 2;

struct WidthSample {
    cols: u16,
    measured_at: Instant,
    kernel_cols: Option<u16>,
}

/// Shared across the listener and render threads. Never hold this lock during
/// terminal I/O: rendering queues a coalesced request and consumes answers later.
struct WidthQueryState {
    pending: bool,
    sample: Option<WidthSample>,
    consecutive_failures: u8,
}

impl WidthQueryState {
    const fn new() -> Self {
        Self {
            pending: false,
            sample: None,
            consecutive_failures: 0,
        }
    }

    fn fresh_cols(&self, now: Instant, kernel_cols: Option<u16>) -> Option<u16> {
        let sample = self.sample.as_ref()?;
        // A kernel resize supersedes the cached answer immediately. Before the
        // kernel catches up, an emulator answer may legitimately differ from it.
        (now.saturating_duration_since(sample.measured_at)
            < Duration::from_millis(WIDTH_REFRESH_TTL_MS)
            && kernel_cols == sample.kernel_cols)
            .then_some(sample.cols)
    }

    fn request_refresh(&mut self, now: Instant, kernel_cols: Option<u16>) {
        if self.pending
            || self.consecutive_failures >= WIDTH_REFRESH_TIMEOUT_BUDGET
            || self.fresh_cols(now, kernel_cols).is_some()
        {
            return;
        }
        self.sample = None;
        self.pending = true;
    }

    fn complete(&mut self, cols: Option<u16>, now: Instant, kernel_cols: Option<u16>) {
        self.pending = false;
        self.sample = cols.filter(|&cols| cols > 0).map(|cols| WidthSample {
            cols,
            measured_at: now,
            kernel_cols,
        });
        if self.sample.is_some() {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        }
    }
}

static WIDTH_QUERY: Mutex<WidthQueryState> = Mutex::new(WidthQueryState::new());

fn width_query_state() -> std::sync::MutexGuard<'static, WidthQueryState> {
    WIDTH_QUERY
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

/// Set while the listener thread owns stdin and services asynchronous width requests.
static LISTENER_ALIVE: AtomicBool = AtomicBool::new(false);
/// Number of listener worker threads currently alive (0 or 1 in normal
/// operation). Incremented when the worker thread starts and decremented when
/// it exits, so the count is an exact thread census — tests use it to prove
/// that a run_loop wait window holds exactly one listener for its whole
/// duration rather than one per scheduler wakeup.
static LISTENER_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Set while the one-line side-note composer is open (Ctrl+G input mode). Transient
/// in-place status writers (subagent status line, task heartbeat, retry waits) must
/// not touch the TTY while this is set: their bare `\r\x1b[2K` redraws would displace
/// the output cursor and erase/misplace the DECSTBM-reserved composer row.
static COMPOSER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Returns true while the side-note composer is open and owns the footer row.
pub(crate) fn side_note_composer_active() -> bool {
    COMPOSER_ACTIVE.load(Ordering::Relaxed)
}

fn set_composer_active(active: bool) {
    COMPOSER_ACTIVE.store(active, Ordering::Relaxed);
}

/// The composer line the listener last painted.
///
/// Every footer-aware output write erases the reserved row before writing (a line
/// that starts on it would otherwise scroll half a composer line into scrollback).
/// The listener repaints the composer at poll cadence, so without this snapshot an
/// in-progress draft would stay invisible for up to one poll interval per output
/// line — and look lost while the model streams.
struct ComposerPaint {
    rows: u16,
    cols: u16,
    input: Vec<char>,
}

static COMPOSER_PAINT: Mutex<Option<ComposerPaint>> = Mutex::new(None);

/// An unsubmitted draft carried across listener-loop instances of one session.
///
/// The draft buffer belongs to the loop, while the driver drops the wait-window
/// listener right before a parked turn resumes (and a foreground stdin handoff
/// restarts the same loop). Keeping the draft only in the loop frame therefore
/// loses a note typed while subagents were still running as soon as new model
/// output arrives. The store is keyed by session so a draft can never leak into
/// another session's composer.
struct StoredDraft {
    session_key: String,
    input: Vec<char>,
    in_input: bool,
}

static DRAFT: Mutex<Option<StoredDraft>> = Mutex::new(None);

fn composer_paint() -> Option<(u16, u16, Vec<char>)> {
    let guard = COMPOSER_PAINT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    guard
        .as_ref()
        .map(|paint| (paint.rows, paint.cols, paint.input.clone()))
}

fn set_composer_paint(rows: u16, cols: u16, input: &[char]) {
    let mut guard = COMPOSER_PAINT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = Some(ComposerPaint {
        rows,
        cols,
        input: input.to_vec(),
    });
}

fn clear_composer_paint() {
    let mut guard = COMPOSER_PAINT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = None;
}

fn restore_draft(session_key: &str) -> Option<StoredDraft> {
    let mut slot = DRAFT.lock().unwrap_or_else(|error| error.into_inner());
    match slot.as_ref() {
        Some(stored) if stored.session_key == session_key => slot.take(),
        Some(_) => {
            // The previous session's listener is gone for good; never queue its
            // text for a later session.
            *slot = None;
            None
        }
        None => None,
    }
}

fn store_draft(session_key: &str, input: &[char], in_input: bool) {
    let mut slot = DRAFT.lock().unwrap_or_else(|error| error.into_inner());
    if input.is_empty() {
        if slot
            .as_ref()
            .is_some_and(|stored| stored.session_key == session_key)
        {
            *slot = None;
        }
        return;
    }
    *slot = Some(StoredDraft {
        session_key: session_key.to_owned(),
        input: input.to_vec(),
        in_input,
    });
}

#[cfg(test)]
thread_local! {
    static SCRIPTED_TRUE_COLS: std::cell::Cell<Option<u16>> = const { std::cell::Cell::new(None) };
}

/// Tests script the emulator's reflowed width directly instead of touching stdin.
#[cfg(test)]
pub(crate) fn set_scripted_true_width(cols: Option<u16>) {
    SCRIPTED_TRUE_COLS.with(|cell| cell.set(cols.filter(|&c| c > 0)));
}

/// Read the listener's last valid answer without waiting for a terminal query.
/// A miss leaves `raw_terminal_cols` on its live ioctl/COLUMNS path.
#[cfg(not(test))]
pub(crate) fn fresh_true_width_cols() -> Option<u16> {
    if !LISTENER_ALIVE.load(Ordering::Acquire) {
        return None;
    }
    let kernel_cols = terminal_size().ok().map(|(cols, _)| cols);
    width_query_state().fresh_cols(Instant::now(), kernel_cols)
}

#[cfg(test)]
pub(crate) fn fresh_true_width_cols() -> Option<u16> {
    SCRIPTED_TRUE_COLS.with(|cell| cell.get())
}

/// Queue at most one asynchronous query when the cached answer is no longer usable.
/// This render-path call never waits for the listener, even under the stdout lock.
pub(crate) fn refresh_true_width() {
    if !LISTENER_ALIVE.load(Ordering::Acquire) {
        return;
    }
    let kernel_cols = terminal_size().ok().map(|(cols, _)| cols);
    width_query_state().request_refresh(Instant::now(), kernel_cols);
}
/// Upper bound for how long Drop waits for the listener thread to confirm stdin /
/// cbreak release.
const SHUTDOWN_WAIT_MS: u64 = 250;
/// Upper bound for waiting at startup for the listener to take over the terminal;
/// on timeout the not-yet-started task is not allowed to modify the terminal.
const STARTUP_WAIT_MS: u64 = 250;
/// Number of physical rows permanently reserved for the bottom composer row. Model
/// output scrolls continuously in the scroll region above it.
const FOOTER_ROWS: u16 = 1;
const COMPOSER_PREFIX: &str = "  [side-note] Esc/F2 send · Ctrl+G cancel > ";
const COMPOSER_CURSOR: char = '▌';

fn should_yield_stdin(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Relaxed) || foreground_stdin_requested()
}

/// Bottom footer reserved via DECSTBM. All output stays on the main screen, avoiding
/// alternate-screen hiding of the transcript; each composer redraw saves/restores the
/// output cursor, so continuous model output is not affected.
pub(super) struct FooterReservation {
    cols: u16,
    rows: u16,
}

impl FooterReservation {
    #[cfg(test)]
    pub(super) fn for_test(cols: u16, rows: u16) -> Self {
        Self { cols, rows }
    }

    fn enter(stop: &AtomicBool) -> io::Result<Self> {
        if should_yield_stdin(stop) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "side-note stopped",
            ));
        }
        let (cols, rows) = terminal_size()?;
        if rows <= FOOTER_ROWS {
            return Err(io::Error::other(
                "terminal is too short for side-note footer",
            ));
        }
        let footer = Self {
            cols,
            rows,
        };
        footer.apply_reservation(stop)?;
        Ok(footer)
    }

    fn output_bottom(&self) -> u16 {
        self.rows - FOOTER_ROWS
    }

    fn apply_reservation(&self, stop: &AtomicBool) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        if should_yield_stdin(stop) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "side-note stopped",
            ));
        }
        self.apply_reservation_to(&mut out)?;
        out.flush()
    }

    pub(super) fn apply_reservation_to(&self, out: &mut impl Write) -> io::Result<()> {
        // IND followed by RI returns to the same output cell unless it was on the
        // bottom row: there, IND scrolls the output up once and RI follows it. This
        // leaves room for the footer without separating the output cursor from
        // the live fold that rewrites relative to it. Jumping to the bottom first
        // would strand that fold and leave old thinking headers behind.
        // Save after this adjustment because DECSTBM itself homes the cursor.
        // As with footer redraws, the stream's live regions reserve a right margin
        // and must not leave the cursor in a pending-wrap state.
        write!(
            out,
            "\x1bD\x1bM\x1b7\x1b[1;{}r\x1b8",
            self.output_bottom()
        )?;
        Ok(())
    }

    fn refresh(&mut self) -> io::Result<()> {
        let (cols, rows) = terminal_size()?;
        if cols == self.cols && rows == self.rows {
            return Ok(());
        }
        if rows <= FOOTER_ROWS {
            return Err(io::Error::other(
                "terminal is too short for side-note footer",
            ));
        }
        let old_rows = self.rows;
        self.cols = cols;
        self.rows = rows;
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // Clear the old footer, then re-assert the reservation with the output
        // cursor clamped inside it: a resize can leave the saved cursor on the
        // new footer row, where the next line write would scroll the composer
        // into the transcript.
        clear_footer_and_reserve(&mut out, old_rows, self.output_bottom())?;
        out.flush()
    }

    fn draw(&mut self, input: &[char]) -> io::Result<()> {
        self.refresh()?;
        let stdout = io::stdout();
        let mut out = stdout.lock();
        self.draw_to(&mut out, input)?;
        out.flush()
    }

    /// Build and write the composer line through a supplied writer so the control
    /// sequence can be regression-tested without a TTY. The caller must hold the
    /// stdout lock for the whole write: the lock is shared with the stream renderer,
    /// and interleaving would split the save/draw/restore sequence into model text.
    pub(super) fn draw_to(&mut self, out: &mut impl io::Write, input: &[char]) -> io::Result<()> {
        out.write_all(composer_line_bytes(self.rows, self.cols, input).as_bytes())
    }

    fn clear(&mut self) -> io::Result<()> {
        self.refresh()?;
        let footer_row = terminal_size()
            .map(|(_, rows)| rows)
            .unwrap_or(self.rows)
            .max(1);
        let stdout = io::stdout();
        let mut out = stdout.lock();
        write!(
            out,
            "\x1b7\x1b[?25l\x1b[{};1H\x1b[2K\x1b[0m\x1b8\x1b[?25h",
            footer_row
        )?;
        out.flush()
    }

    fn leave(&mut self) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // Release ownership while the lock is still held: a writer blocked on
        // stdout must observe the released footer, otherwise it could re-assert
        // the scroll region after `leave_to` reset it.
        set_composer_active(false);
        // Same reasoning for the paint snapshot: no writer that acquires the lock
        // after this point may repaint a composer the listener just closed.
        clear_composer_paint();
        self.leave_to(&mut out)?;
        out.flush()
    }

    pub(super) fn leave_to(&self, out: &mut impl Write) -> io::Result<()> {
        write!(out, "\x1b7\x1b[{};1H\x1b[2K\x1b[r\x1b8\x1b[0m", self.rows)?;
        Ok(())
    }
}

fn terminal_size() -> io::Result<(u16, u16)> {
    use std::os::unix::io::AsRawFd;

    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::ioctl(io::stdout().as_raw_fd(), libc::TIOCGWINSZ, &mut ws) };
    if rc == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
        Ok((ws.ws_col, ws.ws_row))
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Write one complete ordinary output line while the composer footer may be reserved.
///
/// The composer owns the last physical row and repaints it at poll cadence. Ordinary
/// output shares the same TTY cursor: when that cursor rests on the footer row, a
/// plain line write replaces only the leading part of the composer line, and the
/// following line feed pushes the hybrid row into scrollback, where the composer's
/// own `clear`/`leave` can never erase it. The erase below turns such a write into
/// one complete line; the cursor clamp repairs a cursor that a resize or a region
/// reset pushed onto the footer row, so later fragments also stay inside the
/// scrollable area. The next redraw restores the draft.
///
/// The caller passes an already-locked stdout handle so no composer redraw can run
/// between the erase and the line.
pub(crate) fn write_output_line(out: &mut impl Write, line: &str) -> io::Result<()> {
    write_output_line_to(out, line, active_footer_row())?;
    // Restore the draft in the same locked write: the erase above removed the
    // composer row, and waiting for the listener's poll redraw would leave the
    // draft invisible for up to one interval per output line.
    repaint_composer(out)?;
    out.flush()
}

/// Print one ordinary stdout line through the footer-aware path.
pub(crate) fn print_output_line(line: &str) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = write_output_line(&mut out, line);
}

/// Print one stderr diagnostic line through the same footer-aware path.
///
/// Both streams share the terminal's current row, so a diagnostic written while
/// the cursor rests on the footer row has the same hybrid-line failure mode as a
/// stdout line. The stdout lock serializes the erase/clamp with the composer
/// redraw; a failed erase must not drop the diagnostic.
pub(crate) fn print_output_stderr_line(line: &str) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let stderr = io::stderr();
    let mut err = stderr.lock();
    let _ = write_stderr_line_to(&mut out, &mut err, line, active_footer_row());
    let _ = repaint_composer(&mut out);
    let _ = out.flush();
}

/// Print one ordinary output fragment without a trailing newline.
///
/// Used for in-place status rows that a later `\r` overwrite completes; the
/// footer erase/clamp keeps the fragment out of the composer row in the same
/// way as a complete line.
pub(crate) fn print_output_fragment(text: &str) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = write_output_fragment_to(&mut out, text, active_footer_row());
    let _ = repaint_composer(&mut out);
    let _ = out.flush();
}

fn active_footer_row() -> Option<u16> {
    if side_note_composer_active() {
        terminal_size().ok().map(|(_, rows)| rows)
    } else {
        None
    }
}

/// Repaint the composer line the listener last drew, if any.
///
/// No-op in listening mode and after the composer closed: the paint snapshot is
/// cleared under the stdout lock by `clear_input`/`leave`, so a writer that runs
/// after the composer is gone cannot resurrect its row.
pub(super) fn repaint_composer(out: &mut impl Write) -> io::Result<()> {
    let Some((rows, cols, input)) = composer_paint() else {
        return Ok(());
    };
    out.write_all(composer_line_bytes(rows, cols, &input).as_bytes())
}

/// Repaint the active composer after a live writer that bypasses the footer-aware
/// helpers (the markdown streaming renderer rewrites its preview with `CSI 0J`,
/// which erases past the DECSTBM margin into the footer row).
///
/// No-op without a painted composer; the stdout lock serializes the repaint with
/// the listener's own draw.
pub(crate) fn repaint_stdout_composer() -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    repaint_composer(&mut out)?;
    out.flush()
}

fn write_output_fragment_to(
    out: &mut impl Write,
    text: &str,
    footer_row: Option<u16>,
) -> io::Result<()> {
    if let Some(row) = footer_row.filter(|row| *row > FOOTER_ROWS) {
        clear_footer_and_reserve(out, row, row - FOOTER_ROWS)?;
    }
    write!(out, "{text}")
}

fn write_output_line_to(
    out: &mut impl Write,
    line: &str,
    footer_row: Option<u16>,
) -> io::Result<()> {
    if let Some(row) = footer_row.filter(|row| *row > FOOTER_ROWS) {
        clear_footer_and_reserve(out, row, row - FOOTER_ROWS)?;
    }
    writeln!(out, "{line}")
}

fn write_stderr_line_to(
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    line: &str,
    footer_row: Option<u16>,
) -> io::Result<()> {
    if let Some(row) = footer_row.filter(|row| *row > FOOTER_ROWS) {
        // stdout is line-buffered and the erase carries no newline: flush it
        // before the diagnostic, or it would reach the terminal afterwards and
        // could clear the line it was meant to protect. A failed erase must not
        // drop the diagnostic.
        let _ = clear_footer_and_reserve(stdout, row, row - FOOTER_ROWS);
        let _ = stdout.flush();
    }
    writeln!(stderr, "{line}")
}

/// Erase the composer row, then re-assert the footer reservation on `bottom`
/// with the output cursor clamped inside it.
///
/// Resetting DECSTBM homes the cursor, so every region change is bracketed by
/// DECSC/DECRC. `IND`/`RI` under the full-screen region is a no-op unless the
/// cursor is on the last physical row: there it scrolls the already-cleared
/// footer row out and leaves the cursor one row above, inside the region.
fn clear_footer_and_reserve(out: &mut impl Write, row: u16, bottom: u16) -> io::Result<()> {
    debug_assert!(row > FOOTER_ROWS);
    write!(
        out,
        "\x1b7\x1b[{row};1H\x1b[2K\x1b8\x1b[r\x1b8\x1bD\x1bM\x1b7\x1b[1;{bottom}r\x1b8"
    )
}

/// Whether same-terminal side-note input listening is enabled: foreground main agent
/// + interactive stdin + terminal output only.
pub(crate) fn side_note_input_enabled() -> bool {
    let term_out = runtime_ctx::terminal_output_enabled();
    let stdin_tty = io::stdin().is_terminal();
    let depth = runtime_ctx::current_subagent_depth();
    let enabled =
        term_out && stdin_tty && depth == 0 && !crate::ai::background::live_background_active();
    // Temporary diagnostics (prints the real value of each enable condition when
    // RUST_TOOLS_SIDE_NOTE_DEBUG=1), for locating why the listener fails to start
    // when "Ctrl+G does nothing"; remove after diagnosis.
    if std::env::var_os("RUST_TOOLS_SIDE_NOTE_DEBUG").is_some() {
        eprintln!(
            "[side-note-debug] enabled={enabled} term_out={term_out} stdin_tty={stdin_tty} depth={depth}"
        );
    }
    enabled
}

/// Number of listener worker threads currently alive (0 or 1 in normal
/// operation). Exposed for tests: a wait window must hold exactly one listener
/// thread for its whole duration.
pub(crate) fn live_listener_threads() -> usize {
    LISTENER_COUNT.load(Ordering::Acquire)
}

/// RAII guard: holds the background listener task, requests exit on drop and waits
/// for the terminal to be restored.
pub(crate) struct SideNoteInputGuard {
    stop: Arc<AtomicBool>,
    terminal_released: mpsc::Receiver<()>,
    task: Option<JoinHandle<()>>,
}

/// Destination of a submitted side-note draft.
///
/// Local turns append to the session's file queue; the serve-chat client has
/// no local queue (its turn runs in a server-side child), so its drafts are
/// POSTed to the serve session instead. The listener owns no terminal duties
/// and only waits on the sink at poll cadence, so a blocking HTTP POST with
/// its own short timeout composes the same way a filesystem write does.
#[derive(Clone)]
pub(crate) enum SideNoteSink {
    /// `push_side_note` into this history file's queue.
    Local(PathBuf),
    /// POST the draft text; returns true when the server queued it.
    Remote(Arc<dyn Fn(String) -> bool + Send + Sync>),
}

impl SideNoteSink {
    fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }

    /// Stable identity of the session whose draft the composer edits.
    ///
    /// The serve client keeps one listener for its whole session, so remote sinks
    /// never carry a draft across instances: restoring one there could inject a
    /// previous connection's text into another session.
    fn draft_key(&self) -> Option<String> {
        match self {
            Self::Local(history_file) => Some(format!("local:{}", history_file.to_string_lossy())),
            Self::Remote(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SideNoteLoopExit {
    Stop,
    ResumeAfterForeground,
    BackgroundRequested,
}

enum BackgroundRequestAction {
    LiveHandoff {
        session_id: String,
        persona_id: String,
        model: String,
    },
    DetachAttach {
        shutdown: Arc<AtomicBool>,
    },
}

impl SideNoteInputGuard {
    /// The session metadata is used only for a live `/bg` handoff after cbreak mode
    /// and the stdin lease have both been released by this worker.
    pub(crate) fn spawn(
        history_file: PathBuf,
        session_id: String,
        persona_id: String,
        model: String,
    ) -> Self {
        Self::spawn_with_background_action(
            SideNoteSink::Local(history_file),
            BackgroundRequestAction::LiveHandoff {
                session_id,
                persona_id,
                model,
            },
        )
    }

    /// Attach-mode side-note listener. Normal side-notes are still persisted to
    /// the backgrounded session's queue; a `/bg` draft detaches only the current
    /// attach view and is never injected into the model context.
    pub(crate) fn spawn_for_attach(history_file: PathBuf, shutdown: Arc<AtomicBool>) -> Self {
        Self::spawn_with_background_action(
            SideNoteSink::Local(history_file),
            BackgroundRequestAction::DetachAttach { shutdown },
        )
    }

    /// Remote counterpart of [`spawn`](Self::spawn) for the serve-chat client:
    /// the same footer composer, but drafts are POSTed to the serve session.
    /// `/bg`-family drafts stay ordinary note text here: there is no local
    /// attach view to detach, and the draft must never be silently dropped.
    pub(crate) fn spawn_remote(
        post: Arc<dyn Fn(String) -> bool + Send + Sync>,
    ) -> Self {
        // The action never fires: the loop only reports BackgroundRequested
        // for local sinks (see `submit_draft`).
        Self::spawn_with_background_action(
            SideNoteSink::Remote(post),
            BackgroundRequestAction::DetachAttach {
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        )
    }

    fn spawn_with_background_action(
        sink: SideNoteSink,
        background_action: BackgroundRequestAction,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let task_stop = stop.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (terminal_released_tx, terminal_released) = mpsc::sync_channel(1);
        let task = thread::Builder::new()
            .name("a-side-note-input".to_owned())
            .spawn(move || {
                // This thread is the listener; count it for its whole lifetime so
                // tests can assert "exactly one listener per wait window".
                LISTENER_COUNT.fetch_add(1, Ordering::AcqRel);
                let mut ready_tx = Some(ready_tx);
                loop {
                    if task_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // Foreground prompts such as tool confirmation may share a turn with
                    // this listener. The lease guarantees a single stdin reader / termios
                    // owner at any moment; after the foreground ends the listener retakes it.
                    let Some(stdin_owner) = acquire_background_stdin() else {
                        thread::sleep(Duration::from_millis(POLL_MS as u64));
                        continue;
                    };
                    // If the guard was dropped before this thread got scheduled, we must
                    // not switch cbreak or set the scroll region afterwards; otherwise
                    // the next prompt turn would inherit orphaned terminal state.
                    if should_yield_stdin(&task_stop) {
                        drop(stdin_owner);
                        continue;
                    }
                    let term = match CbreakTerm::enter() {
                        Ok(term) => term,
                        Err(_) => break,
                    };
                    if should_yield_stdin(&task_stop) {
                        drop(term);
                        drop(stdin_owner);
                        continue;
                    }
                    // The listener only takes over stdin; before Ctrl+G is pressed it
                    // never sets the scroll region or moves stdout, so a normal turn's
                    // state/body is not unconditionally pushed to the terminal bottom.
                    // Reserving the footer here would pin the cursor to the last screen
                    // row and strand the turn's text at the top with a blank gap between
                    // the two.
                    if let Some(ready_tx) = ready_tx.take() {
                        let _ = ready_tx.send(true);
                    }
                    *width_query_state() = WidthQueryState::new();
                    LISTENER_ALIVE.store(true, Ordering::Release);
                    let exit = side_note_input_loop(&sink, &task_stop, term);
                    LISTENER_ALIVE.store(false, Ordering::Release);
                    *width_query_state() = WidthQueryState::new();
                    drop(stdin_owner);
                    match exit {
                        SideNoteLoopExit::ResumeAfterForeground => continue,
                        SideNoteLoopExit::BackgroundRequested => {
                            let SideNoteSink::Local(history_file) = &sink else {
                                // Unreachable: remote sinks never report
                                // BackgroundRequested (see `submit_draft`).
                                // Break defensively without touching the terminal.
                                break;
                            };
                            if crate::ai::terminal_session::is_worker() {
                                // Only the client detaches. Keep polling this same PTY
                                // so Ctrl+G works immediately after every reattach,
                                // without cancelling or steering the active model turn.
                                if let Err(error) = crate::ai::terminal_session::detach() {
                                    eprintln!("[background] detach failed: {error}");
                                }
                                continue;
                            }
                            match &background_action {
                                BackgroundRequestAction::LiveHandoff {
                                    session_id,
                                    persona_id,
                                    model,
                                } => {
                                    let suspended_store = SuspendedSessionStore::new();
                                    let suspended_entry = match suspended_store
                                        .suspend_current_terminal(
                                            session_id,
                                            history_file.as_path(),
                                            persona_id,
                                            model,
                                        ) {
                                        Ok(entry) => entry,
                                        Err(error) => {
                                            eprintln!(
                                                "[background] live handoff failed: cannot save suspended session binding: {error}"
                                            );
                                            break;
                                        }
                                    };
                                    match crate::ai::background::detach_live_session(session_id) {
                                        Ok(()) => {
                                            // The handoff only pauses the in-flight request while the
                                            // shell retakes the terminal; it does not cancel it. Queue
                                            // guidance after the handoff so a failed handoff cannot
                                            // falsely tell the model that interaction is unavailable.
                                            let _ = push_side_note(
                                                &history_file,
                                                crate::ai::background::BACKGROUND_CONTINUATION_NOTE,
                                                "user",
                                                None,
                                            );
                                        }
                                        Err(error) => {
                                            let _ = suspended_store.take_selected_for_terminal_key(
                                                &suspended_entry.terminal_key,
                                                &suspended_entry,
                                            );
                                            eprintln!("[background] live handoff failed: {error}");
                                        }
                                    }
                                }
                                BackgroundRequestAction::DetachAttach { shutdown } => {
                                    shutdown.store(true, Ordering::Relaxed);
                                }
                            }
                            break;
                        }
                        SideNoteLoopExit::Stop => break,
                    }
                }
                if let Some(ready_tx) = ready_tx {
                    let _ = ready_tx.send(false);
                }
                // Can only be confirmed after the input loop returns and CbreakTerm has
                // been dropped; receiving this message guarantees this thread will no
                // longer poll/read stdin nor hold cbreak terminal state.
                LISTENER_ALIVE.store(false, Ordering::Release);
                let _ = terminal_released_tx.send(());
                LISTENER_COUNT.fetch_sub(1, Ordering::AcqRel);
            })
            .ok();
        if task.is_none()
            || !ready_rx
                .recv_timeout(Duration::from_millis(STARTUP_WAIT_MS))
                .unwrap_or(false)
        {
            stop.store(true, Ordering::Relaxed);
        }
        SideNoteInputGuard {
            stop,
            terminal_released,
            task,
        }
    }
}

impl Drop for SideNoteInputGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        LISTENER_ALIVE.store(false, Ordering::Release);
        let _ = self
            .terminal_released
            .recv_timeout(Duration::from_millis(SHUTDOWN_WAIT_MS));
        if self.task.as_ref().is_some_and(JoinHandle::is_finished) {
            // Only reap finished threads; never join a still-running worker after the
            // timeout, keeping Drop's wait hard-bounded.
            let _ = self.task.take().expect("checked above").join();
        }
    }
}

/// cbreak terminal mode: ICANON off (single keys delivered immediately) and ECHO off
/// (self-managed echo so erasing by character width works), ISIG kept (Ctrl+C still
/// raises SIGINT) and OPOST kept (output newline conversion unaffected).
///
/// Reads are also non-blocking: every read site polls for readability first,
/// but a byte can vanish between poll and read (competing reader, pty quirk),
/// and a blocking read would then stall forever holding the stdin lease while
/// ignoring the stop flag — wedging the next input box, which waits on that
/// lease. With a non-blocking fd the race degrades to an `EAGAIN` retry that
/// re-checks the stop flag on the next poll.
///
/// `O_NONBLOCK` is a file-status flag of the open file description, and a
/// terminal usually hands the process fds 0-2 that share one description of
/// the same tty. Setting it on stdin therefore also makes stdout non-blocking,
/// and a filled pty output buffer then turns every render write into `EAGAIN`
/// ("Resource temporarily unavailable (os error 35)"), which the stream path
/// reports as a failed request. The listener therefore reads through its own
/// open of the same terminal device — a new description, so the flag cannot
/// leak — and leaves stdin's flags untouched. Only when no private handle can
/// be opened does it fall back to the old stdin-based behavior.
struct CbreakTerm {
    saved: libc::termios,
    /// Handle the listener polls and reads; non-blocking on entry.
    read_fd: libc::c_int,
    /// True when `read_fd` is our own open and is closed on drop.
    owns_read_fd: bool,
    /// stdin `fcntl` flags before the fallback `O_NONBLOCK`; `None` when stdin
    /// was left untouched or its flags could not be queried.
    stdin_flags: Option<libc::c_int>,
}

/// Warned-once guard for the fallback `O_NONBLOCK` setup below.
static NONBLOCK_WARNED: AtomicBool = AtomicBool::new(false);
/// Warned-once guard for the degraded fallback where stdin's `O_NONBLOCK`
/// (shared with stdout) is actually applied.
static NONBLOCK_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

impl CbreakTerm {
    fn enter() -> io::Result<Self> {
        // SAFETY: only reads/writes stdin's termios; no mutable access shared across
        // threads.
        unsafe {
            let mut t = std::mem::zeroed::<libc::termios>();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
                return Err(io::Error::last_os_error());
            }
            let saved = t;
            t.c_lflag &= !(libc::ICANON | libc::ECHO);
            t.c_cc[libc::VMIN] = 1;
            t.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) != 0 {
                return Err(io::Error::last_os_error());
            }
            if let Some(read_fd) = open_private_tty() {
                return Ok(CbreakTerm {
                    saved,
                    read_fd,
                    owns_read_fd: true,
                    stdin_flags: None,
                });
            }
            // Fallback: no private handle available. Keep reading stdin itself and
            // make it non-blocking there; this is the one branch whose `O_NONBLOCK`
            // is shared with stdout. On failure keep blocking reads (legacy
            // behavior) rather than disabling the listener.
            let mut stdin_flags = None;
            let current = libc::fcntl(libc::STDIN_FILENO, libc::F_GETFL);
            if current >= 0
                && libc::fcntl(
                    libc::STDIN_FILENO,
                    libc::F_SETFL,
                    current | libc::O_NONBLOCK,
                ) == 0
            {
                stdin_flags = Some(current);
                // Degraded mode: the flag just set is shared with stdout, so
                // render writes can now fail with EAGAIN. Say so once, with a
                // non-panicking write (the same EAGAIN can hit stderr too).
                if !NONBLOCK_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
                    let _ = writeln!(
                        io::stderr(),
                        "[Warning] side-note listener: no private tty handle; stdin O_NONBLOCK is shared with stdout, so terminal writes may fail with EAGAIN (os error 35)."
                    );
                }
            } else if !NONBLOCK_WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[Warning] side-note listener: no private tty handle and stdin O_NONBLOCK failed ({}); a poll/read race may stall the next input box.",
                    io::Error::last_os_error()
                );
            }
            Ok(CbreakTerm {
                saved,
                read_fd: libc::STDIN_FILENO,
                owns_read_fd: false,
                stdin_flags,
            })
        }
    }

    /// Handle the listener polls and reads; non-blocking for this listener.
    fn read_fd(&self) -> libc::c_int {
        self.read_fd
    }
}

/// Open a private handle on the terminal stdin is attached to.
///
/// A fresh `open` creates a new open file description, so `O_NONBLOCK` set on
/// this handle cannot leak into stdin/stdout (see [`CbreakTerm`]). `ttyname(0)`
/// names exactly the device stdin is attached to; `/dev/tty` covers a tty that
/// reports no usable name.
fn open_private_tty() -> Option<libc::c_int> {
    open_private_tty_for(libc::STDIN_FILENO)
}

/// [`open_private_tty`] for an arbitrary tty fd (tests drive it with a pty slave).
fn open_private_tty_for(tty_fd: libc::c_int) -> Option<libc::c_int> {
    // `O_NOCTTY`: never acquire a controlling terminal as a side effect of opening
    // the device path.
    const FLAGS: libc::c_int =
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOCTTY;
    let mut buf = vec![libc::c_char::default(); 4096];
    // SAFETY: `buf` is writable for its full length.
    let mut fd = if unsafe { libc::ttyname_r(tty_fd, buf.as_mut_ptr(), buf.len()) } == 0 {
        // SAFETY: success means `buf` holds a NUL-terminated path.
        unsafe { libc::open(buf.as_ptr(), FLAGS) }
    } else {
        -1
    };
    if fd >= 0 && !same_terminal_device(fd, tty_fd) {
        // SAFETY: `fd` is an open handle owned here.
        unsafe { libc::close(fd) };
        fd = -1;
    }
    if fd < 0 {
        // SAFETY: static NUL-terminated path.
        fd = unsafe { libc::open(c"/dev/tty".as_ptr(), FLAGS) };
        // `/dev/tty` names the controlling terminal, which need not be the
        // terminal stdin is attached to; reading a different device would steal
        // keystrokes from it.
        if fd >= 0 && !same_terminal_device(fd, tty_fd) {
            // SAFETY: `fd` is an open handle owned here.
            unsafe { libc::close(fd) };
            fd = -1;
        }
    }
    (fd >= 0).then_some(fd)
}

/// True when `fd` and `tty_fd` refer to the same terminal device.
///
/// A private handle only substitutes for stdin's terminal when the device
/// matches: `/dev/tty` can name a different controlling terminal, and the
/// listener must never consume input from that one.
fn same_terminal_device(fd: libc::c_int, tty_fd: libc::c_int) -> bool {
    // SAFETY: `fstat`/`isatty` only inspect the two open fds; a zeroed `stat`
    // is a valid output buffer for `fstat`.
    unsafe {
        let mut opened = std::mem::zeroed::<libc::stat>();
        let mut expected = std::mem::zeroed::<libc::stat>();
        if libc::fstat(fd, &mut opened) != 0 || libc::fstat(tty_fd, &mut expected) != 0 {
            return false;
        }
        opened.st_rdev == expected.st_rdev && libc::isatty(fd) == 1
    }
}

impl Drop for CbreakTerm {
    fn drop(&mut self) {
        // SAFETY: restores the original terminal state captured on entry.
        unsafe {
            if let Some(flags) = self.stdin_flags {
                libc::fcntl(libc::STDIN_FILENO, libc::F_SETFL, flags);
            }
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
            if self.owns_read_fd {
                libc::close(self.read_fd);
            }
        }
    }
}

/// Number of terminal columns occupied by a character (East Asian Wide / Fullwidth
/// count as 2 columns, everything else 1). Used for exact column alignment of input
/// echo and backspace erasure.
fn char_width(c: char) -> usize {
    let cp = c as u32;
    if (0x1100..=0x115F).contains(&cp)
        || (0x2E80..=0xA4CF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp) // CJK Extension A
        || (0xAC00..=0xD7A3).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0xFE30..=0xFE4F).contains(&cp)
        || (0xFF00..=0xFF60).contains(&cp)
        || (0xFFE0..=0xFFE6).contains(&cp)
        || (0x1F300..=0x1F9FF).contains(&cp)
        || (0x20000..=0x2FFFD).contains(&cp)
    // CJK Extension B+
    {
        2
    } else {
        1
    }
}

fn display_width(chars: impl IntoIterator<Item = char>) -> usize {
    chars.into_iter().map(char_width).sum()
}

fn composer_layout(cols: usize) -> (&'static str, Option<char>) {
    let full_width = display_width(COMPOSER_PREFIX.chars()) + char_width(COMPOSER_CURSOR) + 1;
    if cols >= full_width {
        (COMPOSER_PREFIX, Some(COMPOSER_CURSOR))
    } else if cols >= 4 {
        ("> ", Some(COMPOSER_CURSOR))
    } else {
        ("", None)
    }
}

/// Trim long input to a single-line tail view, always leaving one column for the
/// composer's visible caret so the bottom line never wraps.
fn input_viewport_with_layout(
    input: &[char],
    cols: usize,
    prefix: &str,
    caret: Option<char>,
) -> String {
    let fixed_width = display_width(prefix.chars()) + caret.map(char_width).unwrap_or(0) + 1;
    let available = cols.saturating_sub(fixed_width);
    if available == 0 {
        return String::new();
    }

    let mut tail = Vec::new();
    let mut width = 0;
    for &ch in input.iter().rev() {
        let char_width = char_width(ch);
        if width + char_width > available {
            break;
        }
        width += char_width;
        tail.push(ch);
    }
    tail.reverse();
    if tail.len() == input.len() {
        return tail.into_iter().collect();
    }

    while width > available.saturating_sub(1) {
        let Some(ch) = tail.first().copied() else {
            break;
        };
        width -= char_width(ch);
        tail.remove(0);
    }
    let mut visible = String::from('…');
    visible.extend(tail);
    visible
}

fn input_viewport(input: &[char], cols: usize) -> String {
    let (prefix, caret) = composer_layout(cols);
    input_viewport_with_layout(input, cols, prefix, caret)
}

fn composer_line_parts(input: &[char], cols: usize) -> (&'static str, String, String) {
    let (prefix, caret) = composer_layout(cols);
    let visible = input_viewport_with_layout(input, cols, prefix, caret);
    (prefix, visible, caret.map(String::from).unwrap_or_default())
}

/// The exact bytes of one composer paint. Shared by the listener's own redraw and
/// by output writers that repaint the draft after erasing the footer row.
///
/// The whole control sequence must land in one locked write: interleaving it with
/// model text would split the save/draw/restore. Do not leave the real cursor in
/// the footer, so the next model/token output does not start from the input row;
/// the tail uses a visible block caret to mark the edit point.
///
/// Argument order is load-bearing: the first positional `{}` is the absolute CUP
/// row; an SGR string in that slot would terminate the CSI early and draw the
/// composer at the model-output cursor instead of the footer row.
fn composer_line_bytes(rows: u16, cols: u16, input: &[char]) -> String {
    let (prefix, visible, caret) = composer_line_parts(input, cols as usize);
    format!(
        "\x1b7\x1b[?25l\x1b[{rows};1H\x1b[2K{}{prefix}{RESET}{visible}{}{caret}{RESET}\x1b8\x1b[?25h",
        theme::current().accent_muted,
        theme::current().accent_muted
    )
}

fn redraw_input(footer: &mut FooterReservation, input: &[char]) -> io::Result<()> {
    footer.draw(input)?;
    set_composer_paint(footer.rows, footer.cols, input);
    Ok(())
}

fn clear_input(footer: &mut FooterReservation) -> io::Result<()> {
    // Drop the snapshot before the erase: a writer that runs between the two must
    // not repaint a draft the composer is already closing.
    clear_composer_paint();
    footer.clear()
}

fn refresh_footer(footer: &mut FooterReservation, input: Option<&[char]>) -> io::Result<()> {
    footer.refresh()?;
    if let Some(input) = input {
        // Route through `redraw_input` so the snapshot follows a resize.
        redraw_input(footer, input)?;
    }
    Ok(())
}

/// Write the side-note from a separate thread that owns no terminal duties; the
/// listener thread only waits for the result at poll cadence, gives up waiting as
/// soon as stop arrives and enters terminal cleanup, and must not get stuck on
/// filesystem I/O.
fn persist_side_note_interruptibly(
    sink: &SideNoteSink,
    content: &str,
    stop: &AtomicBool,
) -> Option<bool> {
    let sink = sink.clone();
    let content = content.to_owned();
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    if thread::Builder::new()
        .name("a-side-note-persist".to_owned())
        .spawn(move || {
            let persisted = match &sink {
                SideNoteSink::Local(history_file) => {
                    push_side_note(history_file, &content, "user", None).is_ok()
                }
                SideNoteSink::Remote(post) => post(content),
            };
            let _ = result_tx.send(persisted);
        })
        .is_err()
    {
        return Some(false);
    }

    loop {
        if should_yield_stdin(stop) {
            return None;
        }
        match result_rx.recv_timeout(Duration::from_millis(POLL_MS as u64)) {
            Ok(persisted) => {
                return (!should_yield_stdin(stop)).then_some(persisted);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Some(false),
        }
    }
}

/// Submit the current draft (shared by the Esc/F2/Alt+Enter submit keys): with empty
/// content just exit input mode; otherwise write to the foreground queue. On write
/// failure the draft is kept and the composer stays, never silently losing the
/// instruction.
fn submit_draft(
    sink: &SideNoteSink,
    stop: &AtomicBool,
    footer: &mut FooterReservation,
    input: &mut Vec<char>,
    pending: &mut Vec<u8>,
    in_input: &mut bool,
) -> io::Result<bool> {
    let content: String = input.iter().collect();
    pending.clear();
    let content = content.trim().to_string();
    if content.is_empty() {
        input.clear();
        *in_input = false;
        let cleared = clear_input(footer);
        set_composer_active(false);
        cleared?;
        return Ok(false);
    }
    // Remote sinks have no local attach view to detach, so `/bg`-family
    // drafts fall through as ordinary note text (see `spawn_remote`).
    if sink.is_local()
        && crate::ai::driver::commands::session::is_real_time_background_command(&content)
    {
        input.clear();
        *in_input = false;
        let cleared = clear_input(footer);
        set_composer_active(false);
        cleared?;
        return Ok(true);
    }
    // Do not insert a confirmation line into the transcript: the model may be mid-way
    // through streaming a Markdown paragraph and an extra newline would change its
    // semantic layout. Clearing the footer is the send feedback.
    match persist_side_note_interruptibly(sink, &content, stop) {
        Some(true) => {
            input.clear();
            *in_input = false;
            let cleared = clear_input(footer);
            set_composer_active(false);
            cleared?;
            Ok(false)
        }
        Some(false) => {
            // Keep the draft and keep showing the composer so the user's instruction
            // is not silently lost; press Esc/F2 again to retry, or Ctrl+G to give up.
            redraw_input(footer, input)?;
            Ok(false)
        }
        None => Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "side-note listener stopped while persisting",
        )),
    }
}

/// Single-byte read with timeout (only used to decide whether a post-Esc key is a
/// submit-type function-key sequence).
/// None means timeout, stop received, or the listener handle closed/errored.
fn read_byte_timeout(fd: libc::c_int, timeout_ms: i32, stop: &AtomicBool) -> Option<u8> {
    let deadline = std::time::Instant::now()
        + Duration::from_millis(u64::try_from(timeout_ms.max(0)).unwrap_or_default());
    loop {
        if should_yield_stdin(stop) || std::time::Instant::now() >= deadline {
            return None;
        }
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let wait_ms = i32::try_from(remaining.as_millis().max(1)).unwrap_or(i32::MAX);
        // SAFETY: the pollfd is a stack-local, exclusively mutable reference.
        let ret = unsafe { libc::poll(&mut pfd, 1, wait_ms) };
        if ret < 0 {
            // EINTR (SIGINT/SIGWINCH etc.) retries, still bounded by the deadline /
            // stop, so a listener thread that is exiting cannot hang forever inside
            // escape sequence parsing.
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return None;
        }
        if ret == 0 {
            return None; // timeout: no follow-up byte, i.e. a bare Esc
        }
        if pfd.revents & libc::POLLIN == 0 {
            return None;
        }
        // Input for the next round may have arrived between poll returning and the
        // guard requesting exit; do not swallow it.
        if should_yield_stdin(stop) {
            return None;
        }
        let mut byte = [0u8; 1];
        // SAFETY: one-byte stack buffer. The listener handle is non-blocking, so a
        // byte that vanished after poll surfaces as `EAGAIN` instead of a stall.
        let n = unsafe { libc::read(fd, byte.as_mut_ptr().cast(), 1) };
        if n == 1 {
            return Some(byte[0]);
        }
        if n < 0 {
            let kind = io::Error::last_os_error().kind();
            if kind == io::ErrorKind::Interrupted || kind == io::ErrorKind::WouldBlock {
                continue; // signal, or the byte vanished after poll; re-poll within the deadline
            }
        }
        return None; // EOF or error
    }
}

/// Decide whether the keys following ESC (0x1b) form a submit key: bare Esc, F2
/// (`ESC O Q` / `ESC [ 1 2 ~`), Alt+Enter (`ESC \r` / `ESC \n`). Terminals emit the
/// full function-key sequence in a single write burst, so follow-up bytes are almost
/// immediately available; whether or not it submits, this function consumes every
/// follow-up byte after ESC so bytes from other sequences such as arrow keys cannot
/// leak into the draft.
fn is_submit_escape(fd: libc::c_int, stop: &AtomicBool) -> bool {
    const ESCAPE_FOLLOWUP_MS: i32 = 30;
    const MAX_CSI_BYTES: usize = 16;
    match read_byte_timeout(fd, ESCAPE_FOLLOWUP_MS, stop) {
        None => !should_yield_stdin(stop), // bare Esc; never submit the draft while stopping / foreground-preempted
        Some(0x0d) | Some(0x0a) => true,   // Alt+Enter
        Some(0x4f) => {
            // SS3 form: `ESC O Q` → F2; the rest (F1/F3/F4 etc.) are swallowed and ignored.
            read_byte_timeout(fd, ESCAPE_FOLLOWUP_MS, stop) == Some(0x51)
        }
        Some(0x5b) => {
            // CSI form: read until the `~` terminator; `ESC [ 1 2 ~` → F2, anything else ignored.
            let mut seq = Vec::new();
            loop {
                match read_byte_timeout(fd, ESCAPE_FOLLOWUP_MS, stop) {
                    Some(0x7e) => return seq == b"12",
                    Some(b) if seq.len() < MAX_CSI_BYTES => seq.push(b),
                    Some(_) => return false,
                    None => return false,
                }
            }
        }
        Some(_) => false,
    }
}

/// Returns why the listener released terminal input ownership.
/// Scans an accumulated buffer for a `CSI 8 ; rows ; cols t` text-area report.
/// `Partial` keeps the read going, `Invalid` replays the bytes as ordinary input.
#[derive(Debug, PartialEq, Eq)]
enum ReportScan {
    Partial,
    Invalid,
    Complete(u16),
}

fn scan_text_area_report(buf: &[u8]) -> ReportScan {
    const HEAD: &[u8] = b"\x1b[8;";
    if buf.len() <= HEAD.len() {
        return if HEAD.starts_with(buf) {
            ReportScan::Partial
        } else {
            ReportScan::Invalid
        };
    }
    if !buf.starts_with(HEAD) {
        return ReportScan::Invalid;
    }
    let rest = &buf[HEAD.len()..];
    let mut rows_end = 0;
    while rows_end < rest.len() && rest[rows_end].is_ascii_digit() {
        rows_end += 1;
    }
    if rows_end == rest.len() {
        return ReportScan::Partial;
    }
    if rows_end == 0 || rest[rows_end] != b';' {
        return ReportScan::Invalid;
    }
    let cols_start = rows_end + 1;
    let mut cols_end = cols_start;
    while cols_end < rest.len() && rest[cols_end].is_ascii_digit() {
        cols_end += 1;
    }
    if cols_end == rest.len() {
        return ReportScan::Partial;
    }
    if cols_end == cols_start || rest[cols_end] != b't' {
        return ReportScan::Invalid;
    }
    match std::str::from_utf8(&rest[cols_start..cols_end])
        .ok()
        .and_then(|cols| cols.parse::<u16>().ok())
    {
        Some(cols) if cols > 0 => ReportScan::Complete(cols),
        _ => ReportScan::Invalid,
    }
}

/// Service one coalesced request on the stdin owner. The shared state remains
/// unlocked during I/O, so a slow or unsupported terminal cannot stall rendering.
fn service_pending_width_query(fd: libc::c_int, replay: &mut VecDeque<u8>) {
    if !width_query_state().pending {
        return;
    }
    let kernel_cols = terminal_size().ok().map(|(cols, _)| cols);
    let cols = service_width_query(fd, replay);
    width_query_state().complete(cols, Instant::now(), kernel_cols);
}

/// One `CSI 18 t` round trip on the listener's own tty handle. Bytes read while
/// hunting for the reply that turn out to be ordinary input are pushed to `replay`, which
/// the loop's byte source drains first, so a query can never eat a keystroke.
fn service_width_query(fd: libc::c_int, replay: &mut VecDeque<u8>) -> Option<u16> {
    let mut out = io::stdout();
    if out.write_all(b"\x1b[18t").is_err() || out.flush().is_err() {
        return None;
    }
    let deadline = Instant::now() + Duration::from_millis(WIDTH_SERVICE_WAIT_MS);
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            replay.extend(buf.drain(..));
            return None;
        }
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: the pollfd is a stack-local, exclusively mutable reference.
        let ret = unsafe { libc::poll(&mut pfd, 1, remaining.as_millis() as i32) };
        if ret < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            replay.extend(buf.drain(..));
            return None;
        }
        if ret == 0 {
            replay.extend(buf.drain(..));
            return None;
        }
        if pfd.revents & libc::POLLIN == 0 {
            if pfd.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0 {
                replay.extend(buf.drain(..));
                return None;
            }
            continue;
        }
        let mut byte = [0u8; 1];
        // SAFETY: one-byte stack buffer. The listener handle is non-blocking, so a
        // byte that vanished after poll surfaces as `EAGAIN` instead of a stall.
        let n = unsafe { libc::read(fd, byte.as_mut_ptr().cast(), 1) };
        if n <= 0 {
            if n < 0 {
                let kind = io::Error::last_os_error().kind();
                if kind == io::ErrorKind::Interrupted || kind == io::ErrorKind::WouldBlock {
                    continue;
                }
            }
            replay.extend(buf.drain(..));
            return None;
        }
        if buf.len() >= 64 {
            replay.extend(buf.drain(..));
            return None;
        }
        buf.push(byte[0]);
        match scan_text_area_report(&buf) {
            ReportScan::Partial => {}
            ReportScan::Complete(cols) => return Some(cols),
            ReportScan::Invalid => replay.extend(buf.drain(..)),
        }
    }
}

fn side_note_input_loop(
    sink: &SideNoteSink,
    stop: &AtomicBool,
    term: CbreakTerm,
) -> SideNoteLoopExit {
    // Input mode: UTF-8 byte accumulation buffer + parsed characters + echoed column count
    let mut pending: Vec<u8> = Vec::new();
    // A draft typed in an earlier listener of this session survives the driver's
    // guard swap (wait window -> turn) and the stdin handoff to a foreground
    // prompt; without the store it would vanish at the next model output.
    let draft_key = sink.draft_key();
    let restored = draft_key.as_deref().and_then(restore_draft);
    let mut input: Vec<char> = restored
        .as_ref()
        .map(|draft| draft.input.clone())
        .unwrap_or_default();
    let mut in_input = restored.as_ref().is_some_and(|draft| draft.in_input);
    // Created lazily on the first Ctrl+G: a turn that never enters input mode must not
    // touch the scroll region or the cursor position.
    let mut footer: Option<FooterReservation> = None;
    if in_input && !should_yield_stdin(stop) {
        // The draft was visible when the previous listener stopped: reopen the
        // composer so it does not look lost while the model resumes.
        match FooterReservation::enter(stop) {
            Ok(active_footer) => {
                footer = Some(active_footer);
                set_composer_active(true);
                let redrawn = match footer.as_mut() {
                    Some(active_footer) => redraw_input(active_footer, &input),
                    None => Ok(()),
                };
                if redrawn.is_err() {
                    // Keep the draft in `input` (and in the store on exit); only
                    // the visible composer failed, and Ctrl+G reopens it.
                    set_composer_active(false);
                    if let Some(mut active_footer) = footer.take() {
                        let _ = active_footer.leave();
                    }
                    in_input = false;
                }
            }
            Err(_) => in_input = false,
        }
    }
    let mut background_requested = false;
    // Bytes the width-query service read while hunting for its `CSI 18 t` reply but that
    // turned out to be ordinary input; the byte source below drains them first so a query
    // never eats a keystroke.
    let mut replay: VecDeque<u8> = VecDeque::new();
    // Kept alive for the whole loop: cbreak termios and the non-blocking read handle.
    let read_fd = term.read_fd();

    loop {
        if should_yield_stdin(stop) {
            break;
        }
        // Publish a pending width answer before polling again. Fold rendering
        // continues independently while this thread waits for the terminal.
        service_pending_width_query(read_fd, &mut replay);
        let b = if let Some(queued) = replay.pop_front() {
            queued
        } else {
            let mut pfd = libc::pollfd {
                fd: read_fd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: the pollfd is a stack-local, exclusively mutable reference.
            let ret = unsafe { libc::poll(&mut pfd, 1, POLL_MS) };
            if should_yield_stdin(stop) {
                break;
            }
            if ret < 0 {
                // EINTR: signals such as terminal resize (SIGWINCH) interrupt poll — a
                // normal occurrence; recompute the footer position. Only other errors
                // (invalid fd etc.) exit the listener.
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    if let Some(active_footer) = footer.as_mut() {
                        if refresh_footer(active_footer, in_input.then_some(input.as_slice()))
                            .is_err()
                        {
                            break;
                        }
                    }
                    continue;
                }
                break;
            }
            if ret == 0 {
                // The renderer still contains a few legacy full-screen erase sequences
                // (CSI 0J). DECSTBM keeps them from scrolling into the footer but does
                // not bound their erase range; redraw the active composer at poll cadence
                // so an in-progress side-note draft survives async output.
                if in_input {
                    let Some(active_footer) = footer.as_mut() else {
                        continue;
                    };
                    if redraw_input(active_footer, &input).is_err() {
                        break;
                    }
                }
                continue;
            }
            if pfd.revents & libc::POLLIN == 0 {
                if pfd.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0 {
                    break; // stdin closed/errored, exit the listener
                }
                continue;
            }
            // When exit and stdin readiness happen together, leave the byte for the
            // prompt that opens next.
            if should_yield_stdin(stop) {
                break;
            }
            let mut byte = [0u8; 1];
            // SAFETY: one-byte stack buffer. The listener handle is non-blocking, so
            // a byte that vanished after poll surfaces as `EAGAIN` (handled below)
            // instead of stalling the lease holder.
            let n = unsafe { libc::read(read_fd, byte.as_mut_ptr().cast(), 1) };
            if n == 0 {
                break; // EOF
            }
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::WouldBlock
                    || err.kind() == io::ErrorKind::Interrupted
                {
                    continue;
                }
                break;
            }
            byte[0]
        };

        if !in_input {
            // Listening state: only Ctrl+G enters input mode; Ctrl+C is handled by the
            // SIGINT path (exit as soon as it is read)
            if b == CTRL_G {
                // Do not clear `input`: a draft restored from a previous listener of
                // this session waits here until the composer is opened again. Submit
                // and cancel already empty it.
                pending.clear();
                if footer.is_none() {
                    match FooterReservation::enter(stop) {
                        Ok(active_footer) => footer = Some(active_footer),
                        Err(_) => continue,
                    }
                }
                if let Some(active_footer) = footer.as_mut() {
                    // Claim ownership before the first draw: a concurrent line write
                    // must erase the composer row as soon as the composer is visible.
                    set_composer_active(true);
                    if redraw_input(active_footer, &input).is_err() {
                        break;
                    }
                    in_input = true;
                }
            } else if b == 0x03 {
                break; // Ctrl+C during streaming: main loop handles the interrupt, listener task exits
            }
            continue;
        }

        // Input mode: control bytes handled directly, everything else decoded as
        // accumulated UTF-8
        match b {
            0x0d | 0x0a => {
                // Enter no longer sends: aligned with the main input's "Enter=newline,
                // Esc/F2 submit" key semantics. In the single-line composer Enter is
                // ignored outright to avoid accidental submits.
            }
            0x7f | 0x08 => {
                // backspace: redraw the single-line viewport; long input never
                // auto-wraps in the footer.
                if input.pop().is_some() {
                    pending.clear(); // avoid a leftover half character forming an illegal sequence with later bytes
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    if redraw_input(active_footer, &input).is_err() {
                        break;
                    }
                }
            }
            0x1b => {
                // Esc / F2 / Alt+Enter: submit the draft. is_submit_escape consumes
                // the full escape sequence of F2 and other function keys so their
                // follow-up bytes cannot leak into the draft; other sequences such as
                // arrow keys are swallowed without submitting.
                if is_submit_escape(read_fd, stop) {
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    match submit_draft(
                        sink,
                        stop,
                        active_footer,
                        &mut input,
                        &mut pending,
                        &mut in_input,
                    ) {
                        Ok(true) => {
                            background_requested = true;
                            break;
                        }
                        Ok(false) => {}
                        Err(_) => break,
                    }
                }
            }
            0x07 => {
                // Ctrl+G again: discard the current draft and exit input mode (cancel).
                input.clear();
                pending.clear();
                in_input = false;
                let cleared = match footer.as_mut() {
                    Some(active_footer) => clear_input(active_footer),
                    None => Ok(()),
                };
                set_composer_active(false);
                if cleared.is_err() {
                    break;
                }
            }
            0x03 => {
                // Ctrl+C while typing: SIGINT has already raised the main interrupt, discard this draft
                input.clear();
                pending.clear();
                if let Some(active_footer) = footer.as_mut() {
                    let _ = clear_input(active_footer);
                }
                break;
            }
            _ => {
                pending.push(b);
                match std::str::from_utf8(&pending) {
                    Ok(s) => {
                        for ch in s.chars() {
                            if ch.is_control() {
                                continue;
                            }
                            input.push(ch);
                        }
                        let Some(active_footer) = footer.as_mut() else {
                            break;
                        };
                        if redraw_input(active_footer, &input).is_err() {
                            break;
                        }
                        pending.clear();
                    }
                    Err(e) if e.error_len().is_none() => {
                        // Incomplete multi-byte sequence, keep accumulating
                    }
                    Err(_) => {
                        pending.clear(); // illegal byte, discard
                    }
                }
            }
        }
    }
    // Loop exit (stop / EOF / error / Ctrl+C): the listener thread alone performs
    // the terminal cleanup, avoiding the guard resetting the scroll region first
    // while a blocking poll has not yet returned.
    // Persist an unsubmitted draft for the next listener of this session. Submit
    // and cancel already emptied `input`, which clears the slot instead.
    if let Some(session_key) = draft_key.as_deref() {
        store_draft(session_key, &input, in_input);
    }
    if let Some(mut active_footer) = footer {
        if in_input {
            let _ = clear_input(&mut active_footer);
        }
        let _ = active_footer.leave();
    }
    // Ownership is released inside `leave` while it still holds the stdout lock,
    // so a writer blocked there observes the released footer and cannot re-assert
    // the scroll region after the reset.
    if background_requested {
        SideNoteLoopExit::BackgroundRequested
    } else if foreground_stdin_requested() && !stop.load(Ordering::Relaxed) {
        SideNoteLoopExit::ResumeAfterForeground
    } else {
        SideNoteLoopExit::Stop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_query_refresh_does_not_wait_for_a_listener_answer() {
        let _serial = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = std::mem::replace(&mut *width_query_state(), WidthQueryState::new());
        let was_alive = LISTENER_ALIVE.swap(true, Ordering::AcqRel);
        let (done_tx, done_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            // No listener services the request. Even while stdout is locked,
            // a burst of render frames must return instead of waiting on it.
            let stdout = io::stdout();
            let _out = stdout.lock();
            for _ in 0..80 {
                refresh_true_width();
            }
            let _ = done_tx.send(());
        });
        let completed = done_rx.recv_timeout(Duration::from_millis(100));
        let pending = width_query_state().pending;
        LISTENER_ALIVE.store(was_alive, Ordering::Release);
        let joined = worker.join();
        *width_query_state() = previous;
        assert!(joined.is_ok());
        assert!(completed.is_ok(), "render frames waited for a width answer");
        assert!(
            pending,
            "the frames must queue a query without a listener answer"
        );
    }

    #[test]
    fn width_query_coalesces_frames_and_expires_answers() {
        let now = Instant::now();
        let mut query = WidthQueryState::new();
        for _ in 0..80 {
            query.request_refresh(now, Some(120));
        }
        assert!(query.pending);
        query.complete(Some(60), now, Some(120));
        // The emulator may reflow before the kernel reports its new width.
        assert_eq!(query.fresh_cols(now, Some(120)), Some(60));
        query.request_refresh(now, Some(120));
        assert!(!query.pending, "a fresh answer must be reused");

        let expired = now + Duration::from_millis(WIDTH_REFRESH_TTL_MS);
        assert_eq!(query.fresh_cols(expired, Some(120)), None);
        query.request_refresh(expired, Some(120));
        assert!(query.pending, "an expired answer needs a new query");
    }

    #[test]
    fn width_query_failed_answers_clear_cache_and_disable_probing() {
        let now = Instant::now();
        let mut query = WidthQueryState::new();
        query.complete(Some(120), now, Some(120));
        query.complete(None, now, Some(120));
        assert_eq!(query.fresh_cols(now, Some(120)), None);
        assert_eq!(query.consecutive_failures, 1);
        query.request_refresh(now, Some(120));
        assert!(query.pending);
        query.complete(Some(0), now, Some(120));
        assert_eq!(query.fresh_cols(now, Some(120)), None);
        query.request_refresh(now, Some(120));
        assert!(
            !query.pending,
            "unsupported terminals must stop being probed"
        );
    }

    #[test]
    fn width_query_kernel_resize_supersedes_cached_emulator_width() {
        let now = Instant::now();
        let mut query = WidthQueryState::new();
        query.complete(Some(60), now, Some(120));
        assert_eq!(query.fresh_cols(now, Some(120)), Some(60));
        // A later kernel resize is authoritative even inside the cache TTL.
        assert_eq!(query.fresh_cols(now, Some(80)), None);
        query.request_refresh(now, Some(80));
        assert!(query.pending);
        query.complete(Some(80), now, Some(80));
        assert_eq!(query.fresh_cols(now, Some(80)), Some(80));
        assert_eq!(query.consecutive_failures, 0);
    }

    #[test]
    fn scan_text_area_report_accepts_reports_and_rejects_input() {
        assert_eq!(scan_text_area_report(b"\x1b[8;24;120t"), ReportScan::Complete(120));
        assert_eq!(scan_text_area_report(b"\x1b[8;24;12"), ReportScan::Partial);
        assert_eq!(scan_text_area_report(b"\x1b[8;"), ReportScan::Partial);
        assert_eq!(scan_text_area_report(b""), ReportScan::Partial);
        assert_eq!(scan_text_area_report(b"\x1b[24;120R"), ReportScan::Invalid);
        assert_eq!(scan_text_area_report(b"abc"), ReportScan::Invalid);
        assert_eq!(scan_text_area_report(b"\x1b[8;;120t"), ReportScan::Invalid);
        assert_eq!(scan_text_area_report(b"\x1b[8;24;120X"), ReportScan::Invalid);
    }

    #[test]
    fn char_width_ascii_and_wide() {
        assert_eq!(char_width('a'), 1);
        assert_eq!(char_width(' '), 1);
        assert_eq!(char_width('中'), 2);
        assert_eq!(char_width('。'), 2);
        assert_eq!(char_width('🚀'), 2);
    }

    #[test]
    fn enabled_check_does_not_panic() {
        // Must never be enabled with non-interactive stdin (pipe/CI), so the listener
        // cannot pollute redirected input. The query must not panic regardless of
        // whether the local stdin is a tty.
        let _ = side_note_input_enabled();
    }

    /// The listener's non-blocking read handle must be its own open file
    /// description: `O_NONBLOCK` is a file-status flag of the description (not of
    /// the fd number), so a handle sharing the tty's description — as `dup` or
    /// `/dev/fd/0` would — also makes stdout non-blocking, and a filled pty buffer
    /// then turns render writes into `EAGAIN` ("os error 35").
    #[test]
    fn listener_read_handle_does_not_share_tty_flags() {
        let mut master: libc::c_int = -1;
        let mut slave: libc::c_int = -1;
        // SAFETY: openpty initializes both fds on success.
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            eprintln!("skipping: no pty available");
            return;
        }
        let Some(private) = open_private_tty_for(slave) else {
            // SAFETY: both fds are owned here and closed exactly once.
            unsafe {
                libc::close(master);
                libc::close(slave);
            }
            eprintln!("skipping: no private tty handle");
            return;
        };
        // SAFETY: plain fcntl flag query/update on fds owned here.
        unsafe {
            let private_flags = libc::fcntl(private, libc::F_GETFL);
            assert!(private_flags >= 0);
            assert_ne!(
                private_flags & libc::O_NONBLOCK,
                0,
                "listener handle must be opened non-blocking"
            );
            assert_ne!(
                libc::fcntl(private, libc::F_GETFD) & libc::FD_CLOEXEC,
                0,
                "listener handle must not leak into exec'd children"
            );
            // The handle must stay on the same device as the tty it reads for; the
            // `/dev/tty` fallback could otherwise silently read another terminal.
            let mut private_stat = std::mem::zeroed::<libc::stat>();
            let mut tty_stat = std::mem::zeroed::<libc::stat>();
            assert_eq!(libc::fstat(private, &mut private_stat), 0);
            assert_eq!(libc::fstat(slave, &mut tty_stat), 0);
            assert_eq!(
                private_stat.st_rdev, tty_stat.st_rdev,
                "listener handle must reference the same terminal device"
            );
            let tty_flags = libc::fcntl(slave, libc::F_GETFL);
            assert!(tty_flags >= 0);
            assert_eq!(
                tty_flags & libc::O_NONBLOCK,
                0,
                "listener handle must not share the terminal's file-status flags"
            );
            libc::close(private);
            libc::close(master);
            libc::close(slave);
        }
    }

    #[test]
    fn guard_drop_is_bounded_when_non_terminal_worker_blocks() {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let (terminal_released_tx, terminal_released) = mpsc::sync_channel(1);
        let (worker_release_tx, worker_release_rx) = mpsc::channel();
        let task = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                thread::yield_now();
            }
            let _ = terminal_released_tx.send(());
            // Simulates a worker whose terminal duty has ended but whose later
            // non-terminal work blocks forever.
            let _ = worker_release_rx.recv();
        });
        let guard = SideNoteInputGuard {
            stop,
            terminal_released,
            task: Some(task),
        };

        let (drop_done_tx, drop_done_rx) = mpsc::channel();
        let dropper = thread::spawn(move || {
            drop(guard);
            let _ = drop_done_tx.send(());
        });

        if drop_done_rx.recv_timeout(Duration::from_secs(1)).is_err() {
            let _ = worker_release_tx.send(());
            let _ = dropper.join();
            panic!("SideNoteInputGuard::drop exceeded its bounded wait");
        }
        let _ = worker_release_tx.send(());
        let _ = dropper.join();
    }

    #[test]
    fn guard_drop_waits_for_normal_listener_cleanup_ack() {
        let stop = Arc::new(AtomicBool::new(false));
        let listener_stop = stop.clone();
        let terminal_owned = Arc::new(AtomicBool::new(true));
        let listener_terminal_owned = terminal_owned.clone();
        let (terminal_released_tx, terminal_released) = mpsc::sync_channel(1);
        let (cleanup_tx, cleanup_rx) = mpsc::channel();
        let task = thread::spawn(move || {
            while !listener_stop.load(Ordering::Relaxed) {
                thread::yield_now();
            }
            let _ = cleanup_rx.recv();
            listener_terminal_owned.store(false, Ordering::Relaxed);
            let _ = terminal_released_tx.send(());
        });
        let guard = SideNoteInputGuard {
            stop,
            terminal_released,
            task: Some(task),
        };

        let (drop_done_tx, drop_done_rx) = mpsc::channel();
        let dropper = thread::spawn(move || {
            drop(guard);
            let _ = drop_done_tx.send(());
        });

        assert!(
            drop_done_rx
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
        let _ = cleanup_tx.send(());
        assert!(drop_done_rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(!terminal_owned.load(Ordering::Relaxed));
        let _ = dropper.join();
    }

    #[test]
    fn input_viewport_keeps_the_tail_on_one_line() {
        let input: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        // cols is wide enough for the full COMPOSER_PREFIX, verifying the single-line
        // constraint under the full prompt prefix.
        let visible = input_viewport(&input, 60);
        assert!(visible.starts_with('…'));
        assert!(visible.ends_with("vwxyz"));
        let total_width = display_width(COMPOSER_PREFIX.chars())
            + display_width(visible.chars())
            + char_width(COMPOSER_CURSOR)
            + 1;
        assert!(total_width <= 60);
    }

    #[test]
    fn input_viewport_keeps_wide_characters_intact() {
        let input: Vec<char> = "前缀-修复这些问题-🚀".chars().collect();
        // Same as above: verify wide characters (CJK/emoji) are not truncated using
        // a width that fits the full prefix.
        let visible = input_viewport(&input, 64);
        assert!(visible.contains('🚀'));
        let total_width = display_width(COMPOSER_PREFIX.chars())
            + display_width(visible.chars())
            + char_width(COMPOSER_CURSOR)
            + 1;
        assert!(total_width <= 64);
    }

    #[test]
    fn composer_line_never_wraps_in_narrow_terminals() {
        let input: Vec<char> = "输入很长的 side note 🚀".chars().collect();
        for cols in 1..=12 {
            let (prefix, visible, caret) = composer_line_parts(&input, cols);
            assert!(
                display_width(prefix.chars())
                    + display_width(visible.chars())
                    + display_width(caret.chars())
                    < cols
            );
        }
    }

    #[test]
    fn composer_draw_targets_footer_row_before_any_color() {
        let mut footer = FooterReservation {
            cols: 80,
            rows: 24,
        };
        let mut out = Vec::new();
        footer.draw_to(&mut out, &[]).unwrap();
        let seq = String::from_utf8(out).unwrap();
        // The absolute CUP to the reserved footer row must precede every SGR color:
        // an SGR string in the CUP row slot terminates the CSI early, the rest of
        // the address leaks as literal text, and the composer is drawn at the
        // model-output cursor instead of the footer row (duplicated-composer bug).
        let expected = format!("\x1b7\x1b[?25l\x1b[{};1H\x1b[2K", footer.rows);
        assert!(
            seq.starts_with(&expected),
            "composer must address the footer row first: {seq:?}"
        );
        // Both positional color slots must receive SGR sequences, never the row number.
        assert_eq!(
            seq.matches(theme::current().accent_muted).count(),
            2,
            "composer color slots misfilled: {seq:?}"
        );
    }

    #[test]
    fn empty_submit_releases_composer_ownership() {
        // Transient status writers consult this flag to defer their bare
        // `\r\x1b[2K` redraws; every path that leaves input mode must clear it,
        // even when the footer redraw itself fails (no TTY in tests).
        set_composer_active(true);
        let sink = SideNoteSink::Local(std::path::PathBuf::from("/nonexistent-test-queue"));
        let stop = std::sync::atomic::AtomicBool::new(false);
        let mut footer = FooterReservation { cols: 80, rows: 24 };
        let mut input: Vec<char> = vec![' ', ' '];
        let mut pending = Vec::new();
        let mut in_input = true;
        let _ = submit_draft(&sink, &stop, &mut footer, &mut input, &mut pending, &mut in_input);
        assert!(!in_input);
        assert!(!side_note_composer_active());
    }

    #[test]
    fn output_line_clears_reserved_footer_before_writing() {
        // A plain line written while the output cursor rests on the footer row must
        // not leave half of the composer line behind: the row is cleared first, the
        // cursor is clamped back into the reserved region, and the write becomes one
        // complete line. Without a reserved footer the sequence is unchanged.
        let mut out = Vec::new();
        write_output_line_to(&mut out, "  │ side-note: injected 1 note(s)", Some(24)).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\x1b7\x1b[24;1H\x1b[2K\x1b8\x1b[r\x1b8\x1bD\x1bM\x1b7\x1b[1;23r\x1b8  │ side-note: injected 1 note(s)\n"
        );

        let mut out = Vec::new();
        write_output_line_to(&mut out, "plain", None).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "plain\n");
    }

    #[test]
    fn stderr_line_flushes_footer_erase_before_diagnostic() {
        // stderr shares the terminal row with stdout: a diagnostic while the
        // composer owns the footer must apply the same erase/clamp, but the
        // diagnostic text is written even when the erase fails. stdout is
        // line-buffered and the erase has no newline, so it must be flushed
        // before the diagnostic instead of trailing it.
        use std::cell::RefCell;
        use std::rc::Rc;

        struct Log(Rc<RefCell<Vec<&'static str>>>);
        struct Out<'a> {
            log: &'a Log,
            bytes: Vec<u8>,
        }
        impl std::io::Write for Out<'_> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.log.0.borrow_mut().push("stdout-write");
                self.bytes.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.log.0.borrow_mut().push("stdout-flush");
                Ok(())
            }
        }
        struct ErrOut<'a> {
            log: &'a Log,
            bytes: Vec<u8>,
        }
        impl std::io::Write for ErrOut<'_> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.log.0.borrow_mut().push("stderr-write");
                self.bytes.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let log = Log(Rc::new(RefCell::new(Vec::new())));
        let mut out = Out { log: &log, bytes: Vec::new() };
        let mut err = ErrOut { log: &log, bytes: Vec::new() };
        write_stderr_line_to(&mut out, &mut err, "[Warning] retry", Some(24)).unwrap();

        assert_eq!(
            String::from_utf8(out.bytes).unwrap(),
            "\x1b7\x1b[24;1H\x1b[2K\x1b8\x1b[r\x1b8\x1bD\x1bM\x1b7\x1b[1;23r\x1b8"
        );
        assert_eq!(String::from_utf8(err.bytes).unwrap(), "[Warning] retry\n");
        let events = log.0.borrow();
        let flush_at = events.iter().position(|e| *e == "stdout-flush").unwrap();
        let stderr_at = events.iter().position(|e| *e == "stderr-write").unwrap();
        assert!(flush_at < stderr_at, "erase must be flushed first: {events:?}");
        assert!(events[..flush_at].iter().all(|e| *e == "stdout-write"));
    }

    #[test]
    fn leave_releases_composer_ownership() {
        // The listener clears the flag inside `leave` while it still holds the
        // stdout lock; clearing it only after `leave` returned left a window
        // where a blocked writer could re-assert the released scroll region.
        set_composer_active(true);
        let mut footer = FooterReservation { cols: 80, rows: 24 };
        let _ = footer.leave();
        assert!(!side_note_composer_active());
    }

    #[test]
    fn fragment_clears_reserved_footer_before_writing() {
        let mut out = Vec::new();
        write_output_fragment_to(&mut out, "running", Some(24)).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\x1b7\x1b[24;1H\x1b[2K\x1b8\x1b[r\x1b8\x1bD\x1bM\x1b7\x1b[1;23r\x1b8running"
        );

        let mut out = Vec::new();
        write_output_fragment_to(&mut out, "running", None).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "running");
    }

    /// The draft and paint stores are process-wide; serialize the tests that use them.
    static STORE_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn store_test_guard() -> std::sync::MutexGuard<'static, ()> {
        STORE_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    #[test]
    fn draft_survives_listener_reentry_for_the_same_session() {
        let _serial = store_test_guard();
        let key = "local:/tmp/a-side-note-draft-reentry.sqlite";
        let draft: Vec<char> = "waiting for the subagent".chars().collect();
        store_draft(key, &draft, true);
        let restored = restore_draft(key).expect("same session must restore the draft");
        assert_eq!(restored.input, draft);
        assert!(restored.in_input, "the composer must reopen with the draft");
        assert!(
            restore_draft(key).is_none(),
            "a restored draft is consumed once"
        );
    }

    #[test]
    fn draft_is_never_carried_into_another_session() {
        let _serial = store_test_guard();
        let old = "local:/tmp/a-side-note-old.sqlite";
        let new = "local:/tmp/a-side-note-new.sqlite";
        store_draft(old, &"secret".chars().collect::<Vec<_>>(), true);
        assert!(restore_draft(new).is_none());
        assert!(
            restore_draft(old).is_none(),
            "the stale draft is dropped, not queued"
        );
    }

    #[test]
    fn empty_draft_clears_the_store() {
        let _serial = store_test_guard();
        let key = "local:/tmp/a-side-note-empty.sqlite";
        store_draft(key, &"typed then submitted".chars().collect::<Vec<_>>(), true);
        store_draft(key, &[], false);
        assert!(restore_draft(key).is_none());
    }

    #[test]
    fn output_line_repaints_the_draft_after_erasing_the_footer() {
        let _serial = store_test_guard();
        set_composer_paint(24, 80, &"keep me".chars().collect::<Vec<_>>());
        let mut out = Vec::new();
        write_output_line_to(&mut out, "  │ note", Some(24)).unwrap();
        repaint_composer(&mut out).unwrap();
        clear_composer_paint();

        let bytes = String::from_utf8(out).unwrap();
        let line_at = bytes.find("  │ note\n").expect("line written");
        let repaint_at = line_at
            + bytes[line_at..]
                .find("\x1b[24;1H\x1b[2K")
                .expect("composer repainted after the line");
        assert!(line_at < repaint_at, "repaint must follow the line: {bytes:?}");
        assert!(bytes.contains("keep me"), "draft text repainted: {bytes:?}");
    }

    #[test]
    fn submitted_draft_lands_in_the_foreground_side_note_queue() {
        // A draft submitted through the composer must persist to the session's
        // foreground side-note queue so the next turn's
        // drain_side_notes_and_inject (orchestrator.rs) picks it up — the contract
        // the wait-window listener relies on while the loop waits for subagents.
        let dir = std::env::temp_dir().join(format!("a-side-note-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let history_file = dir.join("test-session.sqlite");
        let sink = SideNoteSink::Local(history_file.clone());
        let stop = std::sync::atomic::AtomicBool::new(false);
        let mut footer = FooterReservation { cols: 80, rows: 24 };
        let mut input: Vec<char> = "deliver this note".chars().collect();
        let mut pending = Vec::new();
        let mut in_input = true;
        // The draft must be persisted before the footer redraw; in tests stdout is
        // not a tty, so the footer clear can legitimately fail with ENOTTY. Any
        // other error would mean the persist step itself failed.
        match submit_draft(&sink, &stop, &mut footer, &mut input, &mut pending, &mut in_input) {
            Ok(_) => {}
            Err(err) => assert_eq!(
                err.raw_os_error(),
                Some(libc::ENOTTY),
                "non-footer submit error: {err}"
            ),
        }
        assert!(!in_input);
        assert!(!side_note_composer_active());

        let notes = crate::ai::driver::side_note::drain_side_notes(&history_file, None);
        assert_eq!(notes.len(), 1, "exactly one note must land in the foreground queue");
        assert_eq!(notes[0].content, "deliver this note");
        assert_eq!(notes[0].from, "user");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_window_holds_exactly_one_listener_thread() {
        // Mirrors the run_loop wait window: ONE guard created before the wait,
        // held across every scheduler wakeup, dropped when the window ends. The
        // thread census must show exactly one listener for the whole multi-second
        // wait (not one per wakeup) and zero after the guard drops. Where no
        // terminal is available (CI), the worker exits during startup and the
        // census stays at zero; the invariant checked there is "never more than
        // one" and "returns to zero on drop".
        let guard = SideNoteInputGuard::spawn(
            std::path::PathBuf::from("/tmp/a-side-note-wait-window-test"),
            "test-session".to_owned(),
            "test-persona".to_owned(),
            "test-model".to_owned(),
        );
        let census_deadline = Instant::now() + Duration::from_millis(STARTUP_WAIT_MS + 250);
        while live_listener_threads() == 0 && Instant::now() < census_deadline {
            thread::sleep(Duration::from_millis(25));
        }
        if live_listener_threads() == 1 {
            // A terminal is present and the listener took it: it must stay alive
            // for the entire wait window — sample every 500ms over 3s.
            for _ in 0..6 {
                assert_eq!(live_listener_threads(), 1, "listener died mid-wait");
                thread::sleep(Duration::from_millis(500));
            }
        } else {
            // Non-terminal environment: the worker cannot take stdin, but it must
            // never leave a thread behind while the guard is held.
            for _ in 0..3 {
                assert_eq!(live_listener_threads(), 0, "unexpected listener thread");
                thread::sleep(Duration::from_millis(100));
            }
        }
        drop(guard);
        let drop_deadline = Instant::now() + Duration::from_millis(SHUTDOWN_WAIT_MS + 250);
        while live_listener_threads() != 0 && Instant::now() < drop_deadline {
            thread::sleep(Duration::from_millis(25));
        }
        assert_eq!(live_listener_threads(), 0, "listener thread outlived its guard");
    }
}
