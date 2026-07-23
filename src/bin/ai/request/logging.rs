//! Request-layer diagnostics that respect terminal ownership.

use std::fmt;

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
