use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigintAction {
    CancelStream,
    Shutdown,
    Exit,
}

pub(in crate::ai) fn handle_sigint(shutdown: &AtomicBool, streaming: &AtomicBool, cancel_stream: &AtomicBool) {
    match sigint_action(shutdown, streaming, cancel_stream) {
        SigintAction::CancelStream => {
            crate::ai::tools::registry::common::request_tool_cancel();
            cancel_stream.store(true, Ordering::Relaxed);
        }
        SigintAction::Shutdown => {
            crate::ai::tools::registry::common::request_tool_cancel();
            shutdown.store(true, Ordering::Relaxed);
        }
        SigintAction::Exit => {
            crate::ai::tools::registry::common::request_tool_cancel();
            shutdown.store(true, Ordering::Relaxed);
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
            #[cfg(not(test))]
            std::process::exit(130);
        }
    }
}

pub(in crate::ai) fn sigint_action(
    shutdown: &AtomicBool,
    _streaming: &AtomicBool,
    _cancel_stream: &AtomicBool,
) -> SigintAction {
    if shutdown.load(Ordering::Relaxed) {
        SigintAction::Exit
    } else {
        // Always cancel the current turn/output and keep the REPL session alive.
        // Repeated Ctrl+C should keep cancelling, not escalate to shutdown.
        SigintAction::CancelStream
    }
}
