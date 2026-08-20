//! 全局 fork 串行锁（macOS ObjC fork-safety）。
//!
//! macOS 上 `std::process::Command` 的 spawn 底层走 fork/posix_spawn；如果 fork 发生
//! 的瞬间另一个线程正在初始化 Objective-C 类（例如 pdfw 的 Vision OCR 会初始化
//! `__NSCFBoolean`），子进程会在 `objc_initializeAfterForkError` 处崩溃。这把锁把
//! 进程内所有子进程 spawn 与 ObjC 调用串行化，避免二者并发。
//!
//! - 同一线程可重入（例如 pdfw 持有锁时又 spawn tesseract），不同线程互斥。
//! - 辅助方法只覆盖 fork 临界区：spawn 返回后立即释放，不覆盖等待子进程，
//!   因此长命令/编辑器会话不会长时间占用锁。

use std::cell::Cell;
use std::io::{self, Write};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::thread;

static FORK_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

thread_local! {
    static HELD: Cell<bool> = const { Cell::new(false) };
}

/// 获取 fork 串行锁。同一线程可重入；返回的守卫 drop 时释放。
pub fn lock() -> ForkGuard {
    if HELD.with(|h| h.get()) {
        return ForkGuard { guard: None };
    }
    HELD.with(|h| h.set(true));
    let guard = FORK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ForkGuard { guard: Some(guard) }
}

/// fork 串行锁守卫。
pub struct ForkGuard {
    guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for ForkGuard {
    fn drop(&mut self) {
        if self.guard.is_some() {
            HELD.with(|h| h.set(false));
        }
    }
}

/// 在 fork 锁保护下 `spawn`，spawn 返回后立即释放锁（不覆盖等待子进程）。
pub fn spawn(cmd: &mut Command) -> io::Result<Child> {
    let _guard = lock();
    cmd.spawn()
}

/// 在 fork 锁保护下 spawn 并等待，语义同 [`Command::status`]（等待期间不持锁）。
pub fn status(cmd: &mut Command) -> io::Result<ExitStatus> {
    spawn(cmd)?.wait()
}

/// 在 fork 锁保护下 spawn 并等待输出，语义同 [`Command::output`]（等待期间不持锁）。
pub fn output(cmd: &mut Command) -> io::Result<Output> {
    // `Command::output` 语义：stdout/stderr 默认 piped，stdin 默认 null（立即 EOF）。
    // 本实现需等价复刻该行为：stdout/stderr 设为 piped，否则 `wait_with_output`
    // 将得到空输出；stdin 设为 null，避免继承父进程 stdin 在后台/交互路径卡住。
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    spawn(cmd)?.wait_with_output()
}

/// 在 fork 锁保护下 spawn 并等待输出，可向 stdin 喂数据，语义同 [`Command::output`] 但 stdin 为 piped。
///
/// `output` 为防后台/交互路径吞输入已固定为 `stdin(null)`；如需显式 stdin 请用本函数，
/// 而不是改 `output` 语义。本函数将 `stdin` 设为 `piped`，在后台线程写入 `input` 后关闭
/// （向子进程发送 EOF），同时父线程并发读取 stdout/stderr，避免管道缓冲导致的死锁。
/// 写入过程中的 `BrokenPipe`（子进程已提前退出）会被忽略，以已收集的输出为准。
pub fn output_with_input(cmd: &mut Command, input: &[u8]) -> io::Result<Output> {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = spawn(cmd)?;
    // stdin 已设为 piped，`spawn` 成功后必为 Some；take 后由本函数负责关闭/写入。
    let Some(stdin) = child.stdin.take() else {
        return child.wait_with_output();
    };
    if input.is_empty() {
        // 无输入：直接关闭 stdin（drop），子进程读到立即 EOF，语义等价于 null。
        drop(stdin);
        return child.wait_with_output();
    }
    let input = input.to_owned();
    // 并发写入 stdin，避免 stdin 写入与 stdout/stderr 读取相互阻塞。
    // 子进程可能边读 stdin 边写 stdout，单线程“先写完再读”会在 64KB 管道缓冲边界死锁。
    let handle = thread::spawn(move || {
        let mut stdin = stdin;
        let _ = stdin.write_all(&input);
        // `stdin` drop 时关闭写端，向子进程发送 EOF；flush 非必需但显式调用更清晰。
        let _ = stdin.flush();
    });
    let output = child.wait_with_output();
    let _ = handle.join();
    output
}
