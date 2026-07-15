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

/// 当前正在前台同步执行（阻塞父 turn）的子 agent 注册表。
///
/// 同步 `task` 工具在 `execute_sync_task` 里阻塞父 turn 等待子 agent 完成，
/// 期间父 agent 自身既不流式也不在迭代循环里——它“卡”在工具调用里。此时按
/// Ctrl+C，若只看全局 shutdown/streaming 标志会直接把整个主 agent 关掉
/// （子 agent 还卡在静默的 prepare 阶段、streaming=false，于是走 Shutdown 分支）。
///
/// 注册表让 SIGINT 先定向取消“最近一个前台子 agent”：第一次 Ctrl+C 只翻该子
/// agent 自己的 cancel 标志（绝不碰全局 shutdown），父 turn 拿到子 agent 的取消
/// 错误后继续存活；若子 agent 卡死、再次按 Ctrl+C 才落回正常的 shutdown/exit。
///
/// 用栈支持嵌套子 agent：定向取消总是作用于栈顶（最深、最新派发）的那个。
struct ForegroundSubagent {
    id: u64,
    cancel: Arc<AtomicBool>,
    cancel_requested: bool,
}

static FOREGROUND_SUBAGENTS: LazyLock<Mutex<Vec<ForegroundSubagent>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static FOREGROUND_SUBAGENT_SEQ: AtomicU64 = AtomicU64::new(1);

/// 前台 turn 是否正在执行。由前台 `run_turn` 通过 `ForegroundTurnGuard` 抬起，
/// 覆盖 prepare / 思考 / 模型流 / 工具执行 / mid-turn 压缩 / 阶段切换的整个生命周期。
///
/// `app.streaming` 只在「模型流」和「工具执行」两个子阶段抬起，阶段之间以及
/// prepare / 压缩 / 思考期间为 false。仅凭 streaming 判定会让这些空窗里的
/// Ctrl+C 落入 `Shutdown` 分支、直接退出整个交互式会话。该标志补齐「当前有前台
/// turn 在跑」这一事实，使空窗里的 Ctrl+C 也走「取消本轮」而非「退出会话」。
///
/// 子 agent（sync / background）各自持有私有标志，且其 `run_turn` 不抬此标志，
/// 因此该标志只反映「前台主 turn」的活动状态。
static FOREGROUND_TURN_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII 守卫：前台 `run_turn` 进入时抬起 `FOREGROUND_TURN_ACTIVE`，
/// drop（正常返回 / 提前 return / panic）时落下，保证标志不泄漏陈旧状态。
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

/// RAII 守卫：`execute_sync_task` 派发前台子 agent 时注册其 cancel 标志，
/// drop（含 panic / 提前 return）时自动注销，保证注册表不泄漏陈旧条目。
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

/// 尝试把一次 SIGINT 定向到栈顶前台子 agent。
///
/// 返回 `true` 表示“已消费”：翻了子 agent 的 cancel 标志、发了中断通知，调用方
/// 不应再走全局 shutdown/exit。返回 `false` 表示没有可定向的子 agent（栈空），
/// 或栈顶子 agent 此前已被请求取消却仍未退出（判定为卡死，应升级到全局 shutdown）。
fn try_cancel_foreground_subagent() -> bool {
    let cancel_flag = {
        let Ok(mut stack) = FOREGROUND_SUBAGENTS.lock() else {
            return false;
        };
        let Some(top) = stack.last_mut() else {
            return false;
        };
        if top.cancel_requested {
            // 已请求过取消但子 agent 还在栈里 → 视为卡死，升级到全局 shutdown。
            return false;
        }
        top.cancel_requested = true;
        top.cancel.clone()
    };
    cancel_flag.store(true, Ordering::Relaxed);
    crate::ai::tools::registry::common::request_tool_cancel();
    signal_request_interrupt();
    true
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
    // 若已请求过 shutdown，用户的二次 Ctrl+C 是明确的退出诉求：必须优先、无条件
    // 退出，绝不能被“定向取消子 agent”拦截（否则关不掉）。其余情况下先尝试把这次
    // 中断定向给前台子 agent（见 try_cancel_foreground_subagent 的语义）。
    if !shutdown.load(Ordering::Relaxed) && try_cancel_foreground_subagent() {
        return;
    }
    match sigint_action(shutdown, streaming, cancel_stream, foreground_turn_active()) {
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
    foreground_turn_active: bool,
) -> SigintAction {
    if shutdown.load(Ordering::Relaxed) {
        SigintAction::Exit
    } else if streaming.load(Ordering::Relaxed) || foreground_turn_active {
        // streaming：模型流 / 工具执行子阶段。
        // foreground_turn_active：补齐 prepare / 思考 / 阶段切换 / mid-turn 压缩
        // 等 streaming=false 的空窗——只要前台 turn 在跑，Ctrl+C 一律取消本轮，
        // 绝不退出会话；退出语义留给「空闲态」（无 turn 在跑）的 Ctrl+C。
        SigintAction::CancelStream
    } else {
        SigintAction::Shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ForegroundSubagentGuard, ForegroundTurnGuard, SigintAction, foreground_turn_active,
        sigint_action, try_cancel_foreground_subagent,
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
        // prepare / 思考 / 阶段切换 / mid-turn 压缩等空窗：streaming=false 但前台
        // turn 仍在跑。此时 Ctrl+C 必须取消本轮、绝不退出会话。
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        assert!(!foreground_turn_active());
        {
            let _turn = ForegroundTurnGuard::enter();
            assert!(foreground_turn_active());
        }
        assert!(!foreground_turn_active());
    }

    #[test]
    fn first_sigint_cancels_foreground_subagent_without_shutdown() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        super::clear_request_interrupt();

        let cancel = Arc::new(AtomicBool::new(false));
        let _registration = ForegroundSubagentGuard::register(cancel.clone());

        // 第一次 SIGINT：定向取消子 agent，翻它自己的 cancel 标志、不碰全局 shutdown。
        assert!(try_cancel_foreground_subagent());
        assert!(cancel.load(Ordering::Relaxed));

        // 第二次 SIGINT：子 agent 仍在栈里（卡死）→ 不再消费，升级到全局 shutdown。
        assert!(!try_cancel_foreground_subagent());

        super::clear_request_interrupt();
    }

    #[test]
    fn sigint_with_no_foreground_subagent_falls_through() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        // 栈空时不应消费这次中断，调用方据此走正常 shutdown/exit。
        assert!(!try_cancel_foreground_subagent());
    }

    #[test]
    fn foreground_guard_unregisters_on_drop() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        super::clear_request_interrupt();

        let cancel = Arc::new(AtomicBool::new(false));
        {
            let _registration = ForegroundSubagentGuard::register(cancel.clone());
        }
        // guard 已 drop：栈里不再有该条目，定向取消无对象可消费。
        assert!(!try_cancel_foreground_subagent());

        super::clear_request_interrupt();
    }
}
