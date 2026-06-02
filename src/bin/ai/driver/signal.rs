use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use aios_kernel::primitives::{DaemonCancelToken, FutexAddr};
use tokio::sync::Notify;

use crate::ai::tools::os_tools::GLOBAL_OS;

static REQUEST_INTERRUPT_FUTEX: LazyLock<Mutex<Option<(usize, FutexAddr)>>> =
    LazyLock::new(|| Mutex::new(None));
static REQUEST_INTERRUPT_FLAG: AtomicBool = AtomicBool::new(false);
static REQUEST_INTERRUPT_NOTIFY: LazyLock<Notify> = LazyLock::new(Notify::new);

pub(in crate::ai) fn request_interrupt_notify() -> &'static Notify {
    &REQUEST_INTERRUPT_NOTIFY
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigintAction {
    CancelStream,
    Shutdown,
    Exit,
}

pub(in crate::ai) fn handle_sigint(
    shutdown: &AtomicBool,
    streaming: &AtomicBool,
    cancel_stream: &AtomicBool,
) {
    match sigint_action(shutdown, streaming, cancel_stream) {
        SigintAction::CancelStream => {
            crate::ai::tools::registry::common::request_tool_cancel();
            cancel_stream.store(true, Ordering::Relaxed);
            signal_request_interrupt();
        }
        SigintAction::Shutdown => {
            crate::ai::tools::registry::common::request_tool_cancel();
            request_shutdown(shutdown);
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
        }
        SigintAction::Exit => {
            // 用户已二次 Ctrl+C，明确要求退出：必须无条件、优先退出。
            // 不能在此之前调用任何会锁 kernel 的函数（request_tool_cancel /
            // request_shutdown 都会 os.lock()）——若某个后台任务正持有 kernel 锁，
            // 这里会阻塞，导致 std::process::exit 永远执行不到，表现为"Ctrl+C 关不掉"。
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
            #[cfg(not(test))]
            std::process::exit(130);
            #[cfg(test)]
            {
                shutdown.store(true, Ordering::Relaxed);
            }
        }
    }
}

pub(in crate::ai) fn request_shutdown(shutdown: &AtomicBool) {
    shutdown.store(true, Ordering::Relaxed);
    signal_request_interrupt();
}

fn current_global_os() -> Option<aios_kernel::kernel::SharedKernel> {
    let guard = GLOBAL_OS.lock().ok()?;
    guard.as_ref().cloned()
}

fn shared_kernel_id(os: &aios_kernel::kernel::SharedKernel) -> usize {
    std::sync::Arc::as_ptr(os) as *const () as usize
}

pub(in crate::ai) fn request_interrupt_futex() -> Option<FutexAddr> {
    let os = current_global_os()?;
    let os_id = shared_kernel_id(&os);
    let mut os = os.lock().ok()?;
    let mut registry = REQUEST_INTERRUPT_FUTEX.lock().ok()?;
    if let Some((registered_os_id, addr)) = *registry {
        if registered_os_id == os_id && os.futex_load(addr).is_some() {
            return Some(addr);
        }
    }
    let addr = os.futex_create(0, "request_interrupt".to_string());
    *registry = Some((os_id, addr));
    Some(addr)
}

pub(in crate::ai) fn signal_request_interrupt() {
    REQUEST_INTERRUPT_FLAG.store(true, Ordering::Release);
    REQUEST_INTERRUPT_NOTIFY.notify_waiters();
    let Some(addr) = request_interrupt_futex() else {
        return;
    };
    let Some(os) = current_global_os() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let _ = os.futex_store(addr, 1);
}

pub(in crate::ai) fn clear_request_interrupt() {
    REQUEST_INTERRUPT_FLAG.store(false, Ordering::Release);
    let Some(addr) = request_interrupt_futex() else {
        return;
    };
    let Some(os) = current_global_os() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let _ = os.futex_store(addr, 0);
}

pub(in crate::ai) fn alloc_interrupt_futex(label: impl Into<String>) -> Option<FutexAddr> {
    let os = current_global_os()?;
    let mut os = os.lock().ok()?;
    Some(os.futex_create(0, label.into()))
}

pub(in crate::ai) fn signal_interrupt_futex(addr: FutexAddr) {
    let Some(os) = current_global_os() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let _ = os.futex_store(addr, 1);
}

pub(in crate::ai) fn clear_interrupt_futex(addr: FutexAddr) {
    let Some(os) = current_global_os() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let _ = os.futex_store(addr, 0);
}

pub(in crate::ai) fn destroy_interrupt_futex(addr: FutexAddr) {
    let Some(os) = current_global_os() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let _ = os.futex_destroy(addr);
}

pub(in crate::ai) fn interrupt_futex_ready(addr: FutexAddr) -> bool {
    let Some(os) = current_global_os() else {
        return false;
    };
    let Ok(os) = os.lock() else {
        return false;
    };
    os.futex_try_wait(addr, 0).is_some()
}

pub(in crate::ai) fn request_interrupt_ready() -> bool {
    if REQUEST_INTERRUPT_FLAG.load(Ordering::Acquire) {
        return true;
    }
    request_interrupt_futex()
        .map(interrupt_futex_ready)
        .unwrap_or(false)
}

pub(in crate::ai) fn interrupt_sources_ready(local_interrupt_futex: Option<FutexAddr>) -> bool {
    if request_interrupt_ready() {
        return true;
    }
    if let Some(addr) = local_interrupt_futex {
        return interrupt_futex_ready(addr);
    }
    false
}

pub(in crate::ai) async fn wait_for_interrupt_sources(
    cancel_token: Option<DaemonCancelToken>,
    local_interrupt_futex: Option<FutexAddr>,
) {
    loop {
        if interrupt_sources_ready(local_interrupt_futex) {
            return;
        }
        if let Some(token) = cancel_token.as_ref()
            && token.is_cancelled()
        {
            if let Some(addr) = local_interrupt_futex {
                signal_interrupt_futex(addr);
            }
            return;
        }
        // 主信号通过 Notify 唤醒；本地 futex 仍需短轮询兜底（无对应通知通道）。
        let notified = REQUEST_INTERRUPT_NOTIFY.notified();
        tokio::select! {
            _ = notified => {}
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }
}

pub(in crate::ai) fn sigint_action(
    shutdown: &AtomicBool,
    streaming: &AtomicBool,
    _cancel_stream: &AtomicBool,
) -> SigintAction {
    if shutdown.load(Ordering::Relaxed) {
        SigintAction::Exit
    } else if streaming.load(Ordering::Relaxed) {
        SigintAction::CancelStream
    } else {
        SigintAction::Shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::{SigintAction, sigint_action};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn sigint_cancels_streaming_turn() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(true);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream),
            SigintAction::CancelStream
        );
    }

    #[test]
    fn sigint_requests_shutdown_when_idle() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream),
            SigintAction::Shutdown
        );
    }

    #[test]
    fn stale_cancel_flag_does_not_block_idle_shutdown() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(true);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream),
            SigintAction::Shutdown
        );
    }

    #[test]
    fn second_sigint_exits_after_shutdown_requested() {
        let shutdown = AtomicBool::new(true);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream),
            SigintAction::Exit
        );

        streaming.store(true, Ordering::Relaxed);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream),
            SigintAction::Exit
        );
    }
}
