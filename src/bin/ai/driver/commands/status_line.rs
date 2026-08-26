// =============================================================================
// Status line utility: briefly show a hint on the current terminal line, then
// erase it automatically.
// =============================================================================
// Features:
//   - Writes only to stderr, never polluting stdout (assistant output)
//   - Uses ANSI escapes to erase the current line, adding no scrollback
//   - Fully self-contained, with no dependency on driver / prompt / TUI render
//     modules
// =============================================================================

use std::io::{self, Write};
use std::time::Duration;

/// Show a dim hint on the current terminal line and erase it automatically
/// after `duration`.
///
/// How it works:
/// 1. `\x1b[2K` - clear the entire current line (cursor position unchanged)
/// 2. `\x1b[2m` - enable the dim style (ANSI standard; terminals usually
///    render it in gray)
/// 3. Print the text + `\x1b[0m` to reset the style
/// 4. Wait for the given duration
/// 5. Clear the line again with `\x1b[2K`
///
/// What the user sees: a gray hint appears briefly and then disappears,
/// leaving no trace in the terminal history.
pub(crate) fn show_status(msg: &str) {
    show_status_with_duration(msg, Duration::from_secs(2));
}

pub(crate) fn show_status_with_duration(msg: &str, duration: Duration) {
    let mut stderr = io::stderr();

    // Clear the current line → write the dimmed message → flush
    let _ = write!(stderr, "\x1b[2K\x1b[2m{}\x1b[0m", msg);
    let _ = stderr.flush();

    std::thread::sleep(duration);

    // Clear the current line again → flush (the user sees the hint disappear)
    let _ = write!(stderr, "\x1b[2K");
    let _ = stderr.flush();
}

/// Only display the status line without waiting to erase it (for callers that
/// manage the lifecycle themselves).
pub(crate) fn print_status(msg: &str) {
    let mut stderr = io::stderr();
    let _ = write!(stderr, "\x1b[2K\x1b[2m{}\x1b[0m", msg);
    let _ = stderr.flush();
}

/// Erase the status line (paired with `print_status`).
pub(crate) fn clear_status() {
    let mut stderr = io::stderr();
    let _ = write!(stderr, "\x1b[2K");
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_line_module_compiles() {
        // Confirm the function signatures are correct; do not actually sleep in
        // the test
        let _ = show_status as fn(&str);
        let _ = show_status_with_duration as fn(&str, Duration);
        let _ = print_status as fn(&str);
        let _ = clear_status as fn();
    }
}
