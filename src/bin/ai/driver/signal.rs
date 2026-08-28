use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use aios_kernel::primitives::{DaemonCancelToken, FutexAddr};
use tokio::sync::Notify;

use crate::ai::tools::os_tools::GLOBAL_OS;

static REQUEST_INTERRUPT_FUTEX: LazyLock<Mutex<Option<(usize, FutexAddr)>>> =
    LazyLock::new(|| Mutex::new(None));
static REQUEST_INTERRUPT_FLAG: AtomicBool = AtomicBool::new(false);
static REQUEST_INTERRUPT_NOTIFY: LazyLock<Notify> = LazyLock::new(Notify::new);
// Records only session exits triggered by Ctrl+C; proactive exits such as
// `/sessions close` must not be misdetected as a suspension.
static SIGINT_SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Registry of sub-agents currently executing in the foreground synchronously
/// (blocking the parent turn).
///
/// The sync `task` tool blocks the parent turn inside `execute_sync_task` while
/// waiting for the sub-agent to finish; during that window the parent agent
/// itself neither streams nor sits in its iteration loop — it is "stuck" inside
/// a tool call. Pressing Ctrl+C then, judging only by the global
/// shutdown/streaming flags, would shut the whole main agent down (the
/// sub-agent is still stuck in the silent prepare phase with streaming=false,
/// so the Shutdown branch wins).
///
/// This registry lets SIGINT first target-cancel the "most recent foreground
/// sub-agent": the first Ctrl+C only flips that sub-agent's own cancel flag
/// (never touching the global shutdown), and the parent turn survives after
/// receiving the sub-agent's cancellation error; if the sub-agent is stuck
/// (cancel requested but still on the stack), a second Ctrl+C goes straight to
/// the lock-free forced exit — soft cancellation has proven ineffective, and
/// the foreground turn activity flag would keep sigint_action permanently
/// latched on CancelStream (see probe_foreground_subagent).
///
/// A stack supports nested sub-agents: targeted cancellation always applies to
/// the top of the stack (deepest, most recently dispatched).
struct ForegroundSubagent {
    id: u64,
    /// The sub-agent's private cancel flag; once flipped, the process-level
    /// interrupt notification wakes its wait path.
    cancel: Arc<AtomicBool>,
    /// Whether cancellation has already been requested (used to recognize a
    /// "second Ctrl+C" as a stuck-state escalation).
    cancel_requested: bool,
}

/// The result of probing how one SIGINT should act on the foreground sub-agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForegroundSubagentProbe {
    /// No targetable sub-agent (stack empty); this interrupt was not consumed
    /// and normal sigint_action applies.
    None,
    /// Targeted cancellation of the top-of-stack sub-agent happened (its cancel
    /// flag was flipped and it was notified); this interrupt was consumed.
    Cancelled,
    /// The top-of-stack sub-agent was previously asked to cancel but still has
    /// not exited → stuck; soft cancellation is exhausted and a forced exit is due.
    Stuck,
}

static FOREGROUND_SUBAGENTS: LazyLock<Mutex<Vec<ForegroundSubagent>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static FOREGROUND_SUBAGENT_SEQ: AtomicU64 = AtomicU64::new(1);

/// Whether a foreground turn is currently executing. Raised by the foreground
/// `run_turn` via `ForegroundTurnGuard`, covering the whole lifecycle of
/// prepare / thinking / model streaming / tool execution / mid-turn compression
/// / phase switching.
///
/// `app.streaming` is raised only in the "model streaming" and "tool execution"
/// sub-phases, and is false between phases and during prepare / compression /
/// thinking. Relying on streaming alone would make Ctrl+C in those gaps fall
/// into the `Shutdown` branch and exit the whole interactive session. This flag
/// supplies the missing fact "a foreground turn is running", so Ctrl+C in the
/// gaps cancels the turn instead of quitting the session.
///
/// Sub-agents (sync / background) hold their own private flags, and their
/// `run_turn` does not raise this flag, so it reflects only the activity of the
/// "foreground main turn".
static FOREGROUND_TURN_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII guard: raises `FOREGROUND_TURN_ACTIVE` when the foreground `run_turn`
/// enters, and lowers it on drop (normal return / early return / panic) so the
/// flag never leaks stale state.
pub(in crate::ai) struct ForegroundTurnGuard;

impl ForegroundTurnGuard {
    pub(in crate::ai) fn enter() -> Self {
        FOREGROUND_TURN_ACTIVE.store(true, Ordering::Relaxed);
        Self
    }
}

impl Drop for ForegroundTurnGuard {
    fn drop(&mut self) {
        FOREGROUND_TURN_ACTIVE.store(false, Ordering::Relaxed);
    }
}

fn foreground_turn_active() -> bool {
    FOREGROUND_TURN_ACTIVE.load(Ordering::Relaxed)
}

/// RAII guard: registers a foreground sub-agent's cancel flag when
/// `execute_sync_task` dispatches it, and deregisters it automatically on drop
/// (including panic / early return) so the registry never leaks stale entries.
pub(in crate::ai) struct ForegroundSubagentGuard {
    id: u64,
}

impl ForegroundSubagentGuard {
    pub(in crate::ai) fn register(cancel: Arc<AtomicBool>) -> Self {
        let id = FOREGROUND_SUBAGENT_SEQ.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut stack) = FOREGROUND_SUBAGENTS.lock() {
            stack.push(ForegroundSubagent {
                id,
                cancel,
                cancel_requested: false,
            });
        }
        Self { id }
    }
}

impl Drop for ForegroundSubagentGuard {
    fn drop(&mut self) {
        if let Ok(mut stack) = FOREGROUND_SUBAGENTS.lock() {
            stack.retain(|entry| entry.id != self.id);
        }
    }
}

/// Probes how one SIGINT should act on the top-of-stack foreground sub-agent,
/// possibly canceling it in a targeted way.
///
/// See the semantics of the three [`ForegroundSubagentProbe`] variants. `Stuck`
/// is one of the two cases that `try_cancel_foreground_subagent() -> bool` used
/// to lump together as `false` (empty stack / stuck); callers must distinguish
/// them: an empty stack takes normal sigint_action, a stuck sub-agent goes
/// straight to forced exit, otherwise `foreground_turn_active()==true` keeps it
/// latched on CancelStream forever.
fn probe_foreground_subagent() -> ForegroundSubagentProbe {
    let cancel_flag = {
        let Ok(mut stack) = FOREGROUND_SUBAGENTS.lock() else {
            return ForegroundSubagentProbe::None;
        };
        let Some(top) = stack.last_mut() else {
            return ForegroundSubagentProbe::None;
        };
        if top.cancel_requested {
            // Cancellation was requested but the sub-agent is still on the stack →
            // stuck. Soft cancellation has proven ineffective (if it worked, this
            // guard would already have dropped when execute_sync_task returned), so
            // escalation to a forced exit is required.
            return ForegroundSubagentProbe::Stuck;
        }
        top.cancel_requested = true;
        top.cancel.clone()
    };
    cancel_flag.store(true, Ordering::Relaxed);
    // Targeted cancellation only flips the target sub-agent's private cancel flag
    // and wakes its waiters; it does not set the process-level
    // REQUEST_INTERRUPT_FLAG/futex, so concurrent background turns will not read
    // the global flag and mistake it for "this turn was canceled". The target
    // sub-agent's wait path re-checks its private cancel_stream after being woken,
    // via wait_for_interrupt_sources' cancel_flag parameter.
    notify_request_interrupt_waiters();
    let _ = crate::ai::tools::registry::common::try_request_tool_cancel();
    ForegroundSubagentProbe::Cancelled
}

/// Lock-free forced-exit fallback: used on paths where the user has clearly
/// pressed Ctrl+C multiple times (Exit branch) or a foreground sub-agent is
/// stuck (Stuck escalation). Never call anything that locks the kernel on this
/// path (request_tool_cancel / request_shutdown both take locks) — if some
/// background task holds the kernel lock, this would block and process::exit
/// would never run, manifesting as "Ctrl+C can't close the app".
/// Close stdin first so the input thread blocked on read returns.
#[inline]
#[cfg_attr(not(test), allow(unused_variables))] // shutdown is used by the test branch only; avoid an unused warning in non-test builds
fn force_exit_after_sigint(shutdown: &AtomicBool) {
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
    // If shutdown was already requested, the user's second Ctrl+C is an explicit
    // exit request: it must take priority and exit unconditionally, and must
    // never be intercepted by "targeted sub-agent cancellation" (otherwise the
    // app cannot be closed). Otherwise, first try directing this interrupt at
    // the foreground sub-agent (see the probe_foreground_subagent semantics).
    if !shutdown.load(Ordering::Relaxed) {
        match probe_foreground_subagent() {
            ForegroundSubagentProbe::Cancelled => return,
            ForegroundSubagentProbe::Stuck => {
                // The top-of-stack sub-agent was asked to cancel but still has not
                // exited (stuck). The foreground turn's activity flag is true the
                // whole time, so calling sigint_action as usual would stay on
                // CancelStream forever (it only sets a soft flag while the main
                // thread has long been parked with nobody polling) — exactly the
                // root cause of "Ctrl+C can't close the app". Soft cancellation has
                // proven ineffective, and the runtime may already be wedged (a
                // worker stuck on a lock leaves Notify/timers unpolled), so the
                // shutdown branch's reliance on runtime observation cannot help
                // either; the only way out is the lock-free forced-exit fallback
                // (see force_exit_after_sigint).
                force_exit_after_sigint(shutdown);
                return;
            }
            ForegroundSubagentProbe::None => {}
        }
    }
    match sigint_action(shutdown, streaming, cancel_stream, foreground_turn_active()) {
        SigintAction::CancelStream => {
            // Wake request/stream waiters first, then notify the tool layer to
            // cancel. The latter takes the kernel lock; if that lock contention
            // happens on a disconnected/stuck path, it must not block the real
            // Ctrl+C cancel signal.
            cancel_stream.store(true, Ordering::Relaxed);
            signal_request_interrupt();
            let _ = crate::ai::tools::registry::common::try_request_tool_cancel();
        }
        SigintAction::Shutdown => {
            request_sigint_shutdown(shutdown);
            #[cfg(unix)]
            unsafe {
                let _ = libc::close(libc::STDIN_FILENO);
            }
            let _ = crate::ai::tools::registry::common::try_request_tool_cancel();
        }
        SigintAction::Exit => {
            force_exit_after_sigint(shutdown);
        }
    }
}

pub(in crate::ai) fn request_shutdown(shutdown: &AtomicBool) {
    shutdown.store(true, Ordering::Relaxed);
    super::notify_scheduler();
    signal_request_interrupt();
}

/// Marks that this shutdown came from Ctrl+C, so the driver persists the session
/// in a safe event-loop context.
pub(in crate::ai) fn request_sigint_shutdown(shutdown: &AtomicBool) {
    SIGINT_SHUTDOWN_REQUESTED.store(true, Ordering::Release);
    request_shutdown(shutdown);
}

/// Reads and clears the Ctrl+C exit marker so later non-signal exits do not
/// suspend the session again.
pub(in crate::ai) fn take_sigint_shutdown_request() -> bool {
    SIGINT_SHUTDOWN_REQUESTED.swap(false, Ordering::AcqRel)
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

/// Wakes only the tasks waiting on `REQUEST_INTERRUPT_NOTIFY`, and does **not**
/// set the process-level interrupt flag/futex.
///
/// Used to cancel a single foreground sub-agent in a targeted way: the
/// cancellation condition lives in the target sub-agent's private `cancel_stream`
/// (its wait path calls `wait_for_interrupt_sources` with `Some(&cancel_stream)`
/// and, once woken, re-checks that flag to return). Using
/// `signal_request_interrupt` instead would write the process-level flag/futex;
/// concurrent background turns' `should_abort_retry_wait` /
/// `stream_interrupt_requested` would read that global flag and mistake it for
/// "this turn was canceled", and `clear_request_interrupt`, called when any
/// request enters, could also make the target sub-agent miss the cancellation.
pub(in crate::ai) fn notify_request_interrupt_waiters() {
    REQUEST_INTERRUPT_NOTIFY.notify_waiters();
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
    cancel_flag: Option<&AtomicBool>,
) {
    loop {
        if interrupt_sources_ready(local_interrupt_futex) {
            return;
        }
        // Targeted cancellation (foreground sub-agent) flips only its private
        // cancel_stream and sets no global flag; after being woken by the Notify,
        // re-check that flag here so the wait returns promptly.
        if let Some(flag) = cancel_flag
            && flag.load(Ordering::Relaxed)
        {
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
        // The main signal wakes via Notify; the local futex still needs a short
        // polling fallback (no matching notification channel).
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
    foreground_turn_active: bool,
) -> SigintAction {
    if shutdown.load(Ordering::Relaxed) {
        SigintAction::Exit
    } else if streaming.load(Ordering::Relaxed) || foreground_turn_active {
        // streaming: model streaming / tool execution sub-phases.
        // foreground_turn_active: covers the streaming=false gaps such as prepare
        // / thinking / phase switching / mid-turn compression — as long as a
        // foreground turn is running, Ctrl+C always cancels the turn and never
        // exits the session; exit semantics are left to Ctrl+C in the idle state
        // (no turn running).
        SigintAction::CancelStream
    } else {
        SigintAction::Shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ForegroundSubagentGuard, ForegroundTurnGuard, SigintAction, foreground_turn_active,
        request_shutdown, request_sigint_shutdown, sigint_action, take_sigint_shutdown_request,
        handle_sigint, probe_foreground_subagent, ForegroundSubagentProbe,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn sigint_cancels_streaming_turn() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(true);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, false),
            SigintAction::CancelStream
        );
    }

    #[test]
    fn sigint_requests_shutdown_when_idle() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, false),
            SigintAction::Shutdown
        );
    }

    #[test]
    fn sigint_shutdown_is_marked_separately_from_regular_shutdown() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let shutdown = AtomicBool::new(false);
        let _ = take_sigint_shutdown_request();

        request_shutdown(&shutdown);
        assert!(shutdown.load(Ordering::Relaxed));
        assert!(!take_sigint_shutdown_request());

        shutdown.store(false, Ordering::Relaxed);
        request_sigint_shutdown(&shutdown);
        assert!(shutdown.load(Ordering::Relaxed));
        assert!(take_sigint_shutdown_request());
        assert!(!take_sigint_shutdown_request());
    }

    #[test]
    fn stale_cancel_flag_does_not_block_idle_shutdown() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(true);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, false),
            SigintAction::Shutdown
        );
    }

    #[test]
    fn sigint_cancels_during_non_streaming_turn_gap() {
        // Gaps such as prepare / thinking / phase switching / mid-turn compression:
        // streaming=false but a foreground turn is still running. Ctrl+C here must
        // cancel the turn and never exit the session.
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, true),
            SigintAction::CancelStream
        );
    }

    #[test]
    fn second_sigint_exits_after_shutdown_requested() {
        let shutdown = AtomicBool::new(true);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, false),
            SigintAction::Exit
        );

        streaming.store(true, Ordering::Relaxed);
        assert_eq!(
            sigint_action(&shutdown, &streaming, &cancel_stream, true),
            SigintAction::Exit
        );
    }

    #[test]
    fn foreground_turn_guard_toggles_active_flag() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        assert!(!foreground_turn_active());
        {
            let _turn = ForegroundTurnGuard::enter();
            assert!(foreground_turn_active());
        }
        assert!(!foreground_turn_active());
    }

    #[test]
    fn first_sigint_cancels_foreground_subagent_without_shutdown() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        super::clear_request_interrupt();

        let cancel = Arc::new(AtomicBool::new(false));
        let _registration = ForegroundSubagentGuard::register(cancel.clone());

        // First SIGINT: targeted cancellation of the sub-agent — flip its own
        // cancel flag, leave global shutdown untouched.
        assert_eq!(
            probe_foreground_subagent(),
            ForegroundSubagentProbe::Cancelled
        );
        assert!(cancel.load(Ordering::Relaxed));

        // Second SIGINT: the sub-agent is still on the stack (stuck) → not
        // consumed; classified as stuck.
        assert_eq!(probe_foreground_subagent(), ForegroundSubagentProbe::Stuck);

        super::clear_request_interrupt();
    }

    #[test]
    fn stuck_foreground_subagent_escapes_cancel_stream() {
        // Regression test: when a foreground sub-agent is stuck (cancel requested
        // but still on the stack), the second Ctrl+C must exit directly even though
        // the foreground turn activity flag is true (the parent turn is blocked in
        // execute_sync_task without returning), instead of falling into
        // sigint_action's CancelStream (which only sets a soft flag while the
        // parked main thread polls nothing — "Ctrl+C can't close the app").
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        super::clear_request_interrupt();

        let cancel = Arc::new(AtomicBool::new(false));
        let _registration = ForegroundSubagentGuard::register(cancel.clone());

        // First SIGINT: targeted cancellation of the sub-agent, consumed.
        assert_eq!(
            probe_foreground_subagent(),
            ForegroundSubagentProbe::Cancelled
        );
        assert!(cancel.load(Ordering::Relaxed));

        // Simulate the parent turn still blocked in execute_sync_task: raise the
        // foreground turn activity flag — exactly the condition that used to latch
        // sigint_action on CancelStream forever.
        let _turn = ForegroundTurnGuard::enter();
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);

        // Second SIGINT: stuck + foreground turn active → must force-exit directly,
        // and must not be swallowed by CancelStream (in test mode
        // force_exit_after_sigint sets shutdown).
        handle_sigint(&shutdown, &streaming, &cancel_stream);
        assert!(shutdown.load(Ordering::Relaxed));
        assert!(!cancel_stream.load(Ordering::Relaxed));

        super::clear_request_interrupt();
    }

    #[test]
    fn sigint_with_no_foreground_subagent_falls_through() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // With an empty stack this interrupt must not be consumed; callers proceed
        // to normal shutdown/exit.
        assert_eq!(
            probe_foreground_subagent(),
            ForegroundSubagentProbe::None
        );
    }

    #[test]
    fn foreground_guard_unregisters_on_drop() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        super::clear_request_interrupt();

        let cancel = Arc::new(AtomicBool::new(false));
        {
            let _registration = ForegroundSubagentGuard::register(cancel.clone());
        }
        // The guard has dropped: the entry is no longer on the stack, so targeted
        // cancellation has nothing to consume.
        assert_eq!(
            probe_foreground_subagent(),
            ForegroundSubagentProbe::None
        );

        super::clear_request_interrupt();
    }
}
