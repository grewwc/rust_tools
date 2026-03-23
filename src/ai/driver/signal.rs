use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigintAction {
    CancelStream,
    Shutdown,
    Exit,
}

pub fn handle_sigint(
    shutdown: &AtomicBool,
    streaming: &AtomicBool,
    cancel_stream: &AtomicBool,
) {
    match sigint_action(streaming, cancel_stream) {
        SigintAction::CancelStream => {
            cancel_stream.store(true, Ordering::Release);
        }
        SigintAction::Shutdown => {
            shutdown.store(true, Ordering::Release);
        }
        SigintAction::Exit => {
            shutdown.store(true, Ordering::Release);
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
            #[cfg(not(test))]
            std::process::exit(130);
        }
    }
}

pub fn sigint_action(streaming: &AtomicBool, cancel_stream: &AtomicBool) -> SigintAction {
    if streaming.load(Ordering::Acquire) {
        if cancel_stream.load(Ordering::Acquire) {
            SigintAction::Shutdown
        } else {
            SigintAction::CancelStream
        }
    } else {
        SigintAction::Exit
    }
}
