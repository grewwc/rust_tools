//! 后台模式：把 agent 从终端 detach 出来，关闭终端后 agent 仍能继续运行。
//!
//! 关键点是 daemonize 必须发生在 tokio runtime 创建之前——
//! fork 只会拷贝调用线程，多线程 runtime 的其它 worker 线程不会进入子进程，
//! 在 runtime 启动后再 fork 会导致子进程里的 runtime 残缺、死锁。
//! 因此本模块的入口是同步函数，由 `ai::entry` 在构建 runtime 之前调用。

use std::path::Path;

use crate::ai::cli::ParsedCli;
use crate::ai::driver;

/// 后台模式下追加到用户问题后的"不要中途停止"指令。
const BACKGROUND_DIRECTIVE: &str = "\n\n[后台模式提示] 你正在后台模式运行，发起任务的终端可能已经关闭。\
请务必完整地完成上面交给你的任务，在任务真正完成之前不要停止；\
中途遇到问题就继续排查、调用工具解决，而不是请求人工输入或提前结束。\
完成后把最终结论/产出清晰输出即可。";

/// 后台模式入口（同步）：在创建 tokio runtime 之前完成 daemonize。
pub(super) fn run_background(mut cli: ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    // 后台模式必须有明确的任务（位置参数），否则没有 TTY 也无法交互输入。
    if cli.args.is_empty() {
        return Err(
            "background mode (-bg) 需要搭配位置参数（任务描述）使用，例如：a 修复编译错误 -bg".into(),
        );
    }

    // 强制新建 session：用一个确定的 session id 同时作为日志文件名，
    // run_with_cli 内部 resolve_startup_session_choice 会直接采用这个 cli.session。
    let session_id = uuid::Uuid::new_v4().to_string();
    cli.session = Some(session_id.clone());
    // 把"不要中途停止"指令拼到用户问题里（next_question 会 join cli.args）。
    cli.args.push(BACKGROUND_DIRECTIVE.to_string());

    let log_path = std::path::PathBuf::from(format!("{session_id}.log"));

    // detach 前先在原终端上提示用户日志位置，方便后续 tail 查看进度。
    eprintln!("[background] session id : {session_id}");
    eprintln!("[background] log file   : {}", log_path.display());
    eprintln!("[background] 正在脱离终端，关闭本终端不会影响 agent 运行。");

    daemonize(&log_path)?;

    // 已经在 daemon 子进程中：构建 runtime 并运行。
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(driver::run_with_cli(cli))
}

/// 经典 double-fork + setsid 把进程变成 daemon，并把 stdout/stderr 重定向到日志文件，
/// stdin 重定向到 /dev/null。父进程直接 exit(0)，让 shell 立刻返回。
///
/// 该函数只在子进程（daemon）中返回；两个父进程分支都会 `exit(0)`。
#[cfg(unix)]
fn daemonize(log_path: &Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;

    // 第一次 fork：父进程退出，子进程不再是进程组首领。
    match unsafe { libc::fork() } {
        -1 => return Err(std::io::Error::last_os_error()),
        0 => {}
        _ => std::process::exit(0),
    }

    // 成为新会话组长，脱离控制终端。
    if unsafe { libc::setsid() } == -1 {
        return Err(std::io::Error::last_os_error());
    }

    // 第二次 fork：确保 daemon 不会再意外获取控制终端。
    match unsafe { libc::fork() } {
        -1 => return Err(std::io::Error::last_os_error()),
        0 => {}
        _ => std::process::exit(0),
    }

    // 重定向标准流：stdin <- /dev/null，stdout/stderr -> 日志文件。
    let dev_null = std::fs::OpenOptions::new().read(true).open("/dev/null")?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    unsafe {
        libc::dup2(dev_null.as_raw_fd(), 0);
        libc::dup2(log.as_raw_fd(), 1);
        libc::dup2(log.as_raw_fd(), 2);
    }
    // dup2 之后 fd 0/1/2 已指向目标，可以关闭原始句柄。
    drop(dev_null);
    drop(log);

    Ok(())
}

#[cfg(not(unix))]
fn daemonize(_log_path: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "background mode (-bg) 仅支持 unix（double-fork daemonize）",
    ))
}
