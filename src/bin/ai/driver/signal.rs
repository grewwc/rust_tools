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
            cancel_stream.store(true, Ordering::Relaxed);
        }
        SigintAction::Shutdown => {
            shutdown.store(true, Ordering::Relaxed);
        }
        SigintAction::Exit => {
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
    streaming: &AtomicBool,
    cancel_stream: &AtomicBool,
) -> SigintAction {
    if streaming.load(Ordering::Relaxed) {
        if cancel_stream.load(Ordering::Relaxed) {
            SigintAction::Shutdown
        } else {
            SigintAction::CancelStream
        }
    } else if shutdown.load(Ordering::Relaxed) {
        SigintAction::Exit
    } else if cancel_stream.load(Ordering::Relaxed) {
        SigintAction::Shutdown
    } else {
        // Outside streaming (including tool execution), first Ctrl+C cancels the current
        // turn but keeps the session alive. A second Ctrl+C will shut down.
        SigintAction::CancelStream
    }
}
