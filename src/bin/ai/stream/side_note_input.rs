// side_note_input.rs — same-terminal side-note input listener during streaming output (Ctrl+G)
//
// While the main agent is streaming model output, stdin is idle (the interactive input
// box is not open). This module polls for Ctrl+G (0x07) in cbreak mode (ICANON/ECHO
// off, ISIG/OPOST kept): on hit it opens a one-line input; Esc / F2 / Alt+Enter submit
// the draft as a side-note with `from="user"` into the foreground queue, and the next
// iteration injects it into the LLM context via `driver::side_note::poll_and_inject`.
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
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::ai::{
    driver::{runtime_ctx, side_note::push_side_note},
    theme::{ACCENT_MUTED, RESET},
};
use crate::commonw::prompt::{acquire_background_stdin, foreground_stdin_requested};

/// Ctrl+G (BEL)
const CTRL_G: u8 = 0x07;
/// Poll interval in milliseconds; also bounds how quickly the stop flag is observed.
const POLL_MS: i32 = 50;
/// Upper bound for how long Drop waits for the listener thread to confirm stdin /
/// cbreak release.
const SHUTDOWN_WAIT_MS: u64 = 250;
/// Upper bound for waiting at startup for the listener to take over the terminal;
/// on timeout the not-yet-started task is not allowed to modify the terminal.
const STARTUP_WAIT_MS: u64 = 250;
/// Number of physical rows permanently reserved for the bottom composer. Model
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
struct FooterReservation {
    cols: u16,
    rows: u16,
}

impl FooterReservation {
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
        let footer = Self { cols, rows };
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
        // Unconditionally scroll the whole visible screen by two rows to explicitly
        // create two blank lines: "last output row + footer". We cannot assume the
        // physical last row has no stale content after the cursor; only by first
        // scrolling it into scrollback can leave() be guaranteed to clean up only
        // the blank lines this footer itself created.
        write!(
            out,
            "\x1b[{};1H\n\n\x1b[1;{}r\x1b[{};1H",
            self.rows,
            self.output_bottom(),
            self.output_bottom()
        )?;
        out.flush()
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
        // Clear the old footer before re-establishing the scroll region; the
        // save/restore keeps the model-output cursor from being moved by the resize
        // redraw.
        write!(
            out,
            "\x1b7\x1b[{};1H\x1b[2K\x1b[r\x1b[1;{}r\x1b8",
            old_rows,
            self.output_bottom()
        )?;
        out.flush()
    }

    fn draw(&mut self, input: &[char]) -> io::Result<()> {
        self.refresh()?;
        let (prefix, visible, caret) = composer_line_parts(input, self.cols as usize);
        let stdout = io::stdout();
        let mut out = stdout.lock();
        // The stdout lock is shared with the stream renderer: the whole control
        // sequence cannot be interleaved into model text. Do not leave the real
        // cursor in the footer, so the next model/token output does not start from
        // the input row; the tail uses a visible block caret to mark the edit point.
        write!(
            out,
            "\x1b7\x1b[?25l\x1b[{};1H\x1b[2K{ACCENT_MUTED}{prefix}{RESET}{visible}{ACCENT_MUTED}{caret}{RESET}\x1b8\x1b[?25h",
            self.rows
        )?;
        out.flush()
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
        write!(out, "\x1b7\x1b[{};1H\x1b[2K\x1b[r\x1b8\x1b[0m", self.rows)?;
        out.flush()
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

/// Whether same-terminal side-note input listening is enabled: foreground main agent
/// + interactive stdin + terminal output only.
pub(crate) fn side_note_input_enabled() -> bool {
    let term_out = runtime_ctx::terminal_output_enabled();
    let stdin_tty = io::stdin().is_terminal();
    let depth = runtime_ctx::current_subagent_depth();
    let enabled = term_out && stdin_tty && depth == 0;
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

/// RAII guard: holds the background listener task, requests exit on drop and waits
/// for the terminal to be restored.
pub(crate) struct SideNoteInputGuard {
    stop: Arc<AtomicBool>,
    terminal_released: mpsc::Receiver<()>,
    task: Option<JoinHandle<()>>,
}

impl SideNoteInputGuard {
    pub(crate) fn spawn(history_file: PathBuf) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let task_stop = stop.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (terminal_released_tx, terminal_released) = mpsc::sync_channel(1);
        let task = thread::Builder::new()
            .name("a-side-note-input".to_owned())
            .spawn(move || {
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
                    if let Some(ready_tx) = ready_tx.take() {
                        let _ = ready_tx.send(true);
                    }
                    let resume_after_foreground =
                        side_note_input_loop(&history_file, &task_stop, term);
                    drop(stdin_owner);
                    if !resume_after_foreground {
                        break;
                    }
                }
                if let Some(ready_tx) = ready_tx {
                    let _ = ready_tx.send(false);
                }
                // Can only be confirmed after the input loop returns and CbreakTerm has
                // been dropped; receiving this message guarantees this thread will no
                // longer poll/read stdin nor hold cbreak terminal state.
                let _ = terminal_released_tx.send(());
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
struct CbreakTerm {
    saved: libc::termios,
}

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
            Ok(CbreakTerm { saved })
        }
    }
}

impl Drop for CbreakTerm {
    fn drop(&mut self) {
        // SAFETY: restores the original terminal state captured on entry.
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
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

fn redraw_input(footer: &mut FooterReservation, input: &[char]) -> io::Result<()> {
    footer.draw(input)
}

fn clear_input(footer: &mut FooterReservation) -> io::Result<()> {
    footer.clear()
}

fn refresh_footer(footer: &mut FooterReservation, input: Option<&[char]>) -> io::Result<()> {
    footer.refresh()?;
    if let Some(input) = input {
        footer.draw(input)?;
    }
    Ok(())
}

/// Write the side-note from a separate thread that owns no terminal duties; the
/// listener thread only waits for the result at poll cadence, gives up waiting as
/// soon as stop arrives and enters terminal cleanup, and must not get stuck on
/// filesystem I/O.
fn persist_side_note_interruptibly(
    history_file: &Path,
    content: &str,
    stop: &AtomicBool,
) -> Option<bool> {
    let history_file = history_file.to_path_buf();
    let content = content.to_owned();
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    if thread::Builder::new()
        .name("a-side-note-persist".to_owned())
        .spawn(move || {
            let persisted = push_side_note(&history_file, &content, "user", None).is_ok();
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
    history_file: &Path,
    stop: &AtomicBool,
    footer: &mut FooterReservation,
    input: &mut Vec<char>,
    pending: &mut Vec<u8>,
    in_input: &mut bool,
) -> io::Result<()> {
    let content: String = input.iter().collect();
    pending.clear();
    let content = content.trim().to_string();
    if content.is_empty() {
        input.clear();
        *in_input = false;
        return clear_input(footer);
    }
    // Do not insert a confirmation line into the transcript: the model may be mid-way
    // through streaming a Markdown paragraph and an extra newline would change its
    // semantic layout. Clearing the footer is the send feedback.
    match persist_side_note_interruptibly(history_file, &content, stop) {
        Some(true) => {
            input.clear();
            *in_input = false;
            clear_input(footer)
        }
        Some(false) => {
            // Keep the draft and keep showing the composer so the user's instruction
            // is not silently lost; press Esc/F2 again to retry, or Ctrl+G to give up.
            redraw_input(footer, input)
        }
        None => Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "side-note listener stopped while persisting",
        )),
    }
}

/// Single-byte read with timeout (only used to decide whether a post-Esc key is a
/// submit-type function-key sequence).
/// None means timeout, stop received, or stdin closed/errored.
fn read_byte_timeout(timeout_ms: i32, stop: &AtomicBool) -> Option<u8> {
    let deadline = std::time::Instant::now()
        + Duration::from_millis(u64::try_from(timeout_ms.max(0)).unwrap_or_default());
    loop {
        if should_yield_stdin(stop) || std::time::Instant::now() >= deadline {
            return None;
        }
        let mut pfd = libc::pollfd {
            fd: libc::STDIN_FILENO,
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
        // SAFETY: one-byte stack buffer; poll confirmed readability so the blocking
        // read returns immediately.
        let n = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), 1) };
        if n == 1 {
            return Some(byte[0]);
        }
        if n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue; // read interrupted by a signal, retry
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
fn is_submit_escape(stop: &AtomicBool) -> bool {
    const ESCAPE_FOLLOWUP_MS: i32 = 30;
    const MAX_CSI_BYTES: usize = 16;
    match read_byte_timeout(ESCAPE_FOLLOWUP_MS, stop) {
        None => !should_yield_stdin(stop), // bare Esc; never submit the draft while stopping / foreground-preempted
        Some(0x0d) | Some(0x0a) => true,   // Alt+Enter
        Some(0x4f) => {
            // SS3 form: `ESC O Q` → F2; the rest (F1/F3/F4 etc.) are swallowed and ignored.
            read_byte_timeout(ESCAPE_FOLLOWUP_MS, stop) == Some(0x51)
        }
        Some(0x5b) => {
            // CSI form: read until the `~` terminator; `ESC [ 1 2 ~` → F2, anything else ignored.
            let mut seq = Vec::new();
            loop {
                match read_byte_timeout(ESCAPE_FOLLOWUP_MS, stop) {
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

/// Returns true when the exit was caused by a foreground prompt taking over; the
/// caller should retake stdin after the prompt finishes.
fn side_note_input_loop(history_file: &PathBuf, stop: &AtomicBool, _term: CbreakTerm) -> bool {
    // Input mode: UTF-8 byte accumulation buffer + parsed characters + echoed column count
    let mut pending: Vec<u8> = Vec::new();
    let mut input: Vec<char> = Vec::new();
    let mut in_input = false;
    let mut footer: Option<FooterReservation> = None;

    loop {
        if should_yield_stdin(stop) {
            break;
        }
        let b = {
            let mut pfd = libc::pollfd {
                fd: libc::STDIN_FILENO,
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
                // not bound their erase range; while input is active we redraw at poll
                // cadence so a draft briefly cleared by an async redraw recovers within
                // 50ms, without forcing every renderer's mature rewrite state machine
                // onto a different cursor protocol.
                if in_input {
                    let Some(active_footer) = footer.as_mut() else {
                        break;
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
            // SAFETY: one-byte stack buffer; poll confirmed readability so the blocking
            // read returns immediately.
            let n = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), 1) };
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
                input.clear();
                pending.clear();
                match FooterReservation::enter(stop) {
                    Ok(mut active_footer) => {
                        if redraw_input(&mut active_footer, &input).is_err() {
                            let _ = active_footer.leave();
                            break;
                        }
                        footer = Some(active_footer);
                        in_input = true;
                    }
                    Err(_) => {
                        // Ignore this Ctrl+G when the current size cannot safely
                        // establish a footer; listening continues and the user can
                        // retry after resizing the terminal.
                    }
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
                if is_submit_escape(stop) {
                    let Some(active_footer) = footer.as_mut() else {
                        break;
                    };
                    if submit_draft(
                        history_file,
                        stop,
                        active_footer,
                        &mut input,
                        &mut pending,
                        &mut in_input,
                    )
                    .is_err()
                    {
                        break;
                    }
                    if !in_input {
                        if let Some(mut finished_footer) = footer.take() {
                            let _ = finished_footer.leave();
                        }
                    }
                }
            }
            0x07 => {
                // Ctrl+G again: discard the current draft and exit input mode (cancel).
                input.clear();
                pending.clear();
                in_input = false;
                if let Some(mut finished_footer) = footer.take() {
                    if clear_input(&mut finished_footer).is_err() {
                        let _ = finished_footer.leave();
                        break;
                    }
                    let _ = finished_footer.leave();
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
    if let Some(mut active_footer) = footer {
        if in_input {
            let _ = clear_input(&mut active_footer);
        }
        let _ = active_footer.leave();
    }
    foreground_stdin_requested() && !stop.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
