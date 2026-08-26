//! Request-layer diagnostics that respect terminal ownership.

use std::fmt;
use std::io::IsTerminal;
use std::io::Write;

/// Whether request diagnostics may be written to the live terminal.
pub(in crate::ai) fn request_diagnostics_enabled() -> bool {
    crate::ai::driver::runtime_ctx::terminal_output_enabled()
}

/// Emit a request diagnostic to stderr only when the current task owns the
/// terminal. Background subagents publish progress through task IPC/status lines
/// instead of writing directly to the foreground TTY.
pub(in crate::ai) fn emit_request_diagnostic(args: fmt::Arguments<'_>) -> bool {
    if !request_diagnostics_enabled() {
        return false;
    }
    eprintln!("{args}");
    true
}

/// Single-line transient status line for progress-style messages that repeat
/// frequently, such as TPM rate-limit waits and retry waits.
///
/// - TTY + foreground: redraws in place with `\r\x1b[2K` and is finally removed
///   by `clear()` leaving no trace, avoiding screen spam.
/// - Non-TTY (pipe / logs) / background subagent: falls back to a plain
///   `eprintln!`, emitted at most once per change and only when the last output
///   was more than a set interval ago, keeping the line count bounded.
///
/// The caller keeps an `Option<TransientStatusLine>`, creates it on the first
/// update, and calls `clear()` then drops it when done. All writes go to
/// stderr, so they never interfere with the main output on stdout.
pub(in crate::ai) struct TransientStatusLine {
    last_text: String,
    last_emitted: std::time::Instant,
    is_tty: bool,
    visible: bool,
}

impl TransientStatusLine {
    /// Only created when output is allowed; otherwise returns None so the
    /// caller can be a full no-op.
    pub(in crate::ai) fn new() -> Option<Self> {
        if !request_diagnostics_enabled() {
            return None;
        }
        let is_tty = std::io::stderr().is_terminal();
        // Initialize last_emitted to 1 hour ago so the first update always
        // produces output.
        let last_emitted = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or(std::time::Instant::now());
        Some(Self {
            last_text: String::new(),
            last_emitted,
            is_tty,
            visible: false,
        })
    }

    /// Updates the current line. On a TTY it redraws in place every time; on a
    /// non-TTY it only prints a line when the text changed and the last output
    /// was more than `MIN_NON_TTY_INTERVAL` ago, preventing log flooding.
    pub(in crate::ai) fn update(&mut self, text: &str) {
        if self.is_tty {
            // Show in dim color to avoid stealing attention; use \r to return
            // to the start of the line plus \x1b[2K to clear the whole line.
            // Note: no newline is written; the cursor stays at the end of the
            // current line.
            let _ = write!(std::io::stderr(), "\r\x1b[2K\x1b[2m{text}\x1b[0m");
            let _ = std::io::stderr().flush();
            self.visible = true;
        } else if text != self.last_text && self.last_emitted.elapsed() >= MIN_NON_TTY_INTERVAL {
            eprintln!("[Info] {text}");
            self.last_emitted = std::time::Instant::now();
        }
        self.last_text = text.to_string();
    }

    /// Clears the transient line (only meaningful on a TTY), ensuring
    /// heartbeat/wait hints do not linger into the next line of output.
    pub(in crate::ai) fn clear(&mut self) {
        if self.is_tty && self.visible {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.visible = false;
        }
    }
}

impl Drop for TransientStatusLine {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Minimum interval between two transient messages of the same kind in a
/// non-TTY environment, to avoid flooding the logs.
const MIN_NON_TTY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
