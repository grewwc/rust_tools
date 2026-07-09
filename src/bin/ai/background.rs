//! 后台模式：把 agent 从终端 detach 出来，关闭终端后 agent 仍能继续运行。
//!
//! 关键点是 daemonize 必须发生在 tokio runtime 创建之前——
//! fork 只会拷贝调用线程，多线程 runtime 的其它 worker 线程不会进入子进程，
//! 在 runtime 启动后再 fork 会导致子进程里的 runtime 残缺、死锁。
//! 因此本模块的入口是同步函数，由 `ai::entry` 在构建 runtime 之前调用。

use std::path::{Path, PathBuf};
use std::process;

use crate::ai::cli::ParsedCli;
use crate::ai::driver;

/// 后台模式下追加到用户问题后的"不要中途停止"指令。
const BACKGROUND_DIRECTIVE: &str = "\n\n[后台模式提示] 你正在后台模式运行，发起任务的终端可能已经关闭。\
请务必完整地完成上面交给你的任务，在任务真正完成之前不要停止；\
中途遇到问题就继续排查、调用工具解决，而不是请求人工输入或提前结束。\
完成后把最终结论/产出清晰输出即可。";

/// 后台模式入口（同步）：在创建 tokio runtime 之前完成 daemonize。
pub(super) fn run_background(mut cli: ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    // 生成 session id（同时作为日志文件名）。
    let session_id = cli
        .session
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone();

    // 后台模式：优先使用位置参数作为任务描述；若未提供位置参数，
    // 则在 daemonize 之前（仍持有 TTY）交互式读取多行输入——
    // detach 后 stdin 会被重定向到 /dev/null，无法再交互输入。
    if cli.args.is_empty() {
        match read_task_interactively(&cli, &session_id)? {
            Some(s) if !s.trim().is_empty() => cli.args = vec![s],
            _ => {
                eprintln!("[background] 输入为空，已取消。");
                return Ok(());
            }
        }
    }

    // 把"不要中途停止"指令拼到用户问题里（next_question 会 join cli.args）。
    cli.args.push(BACKGROUND_DIRECTIVE.to_string());

    let log_path = std::path::PathBuf::from(format!("{session_id}.log"));

    // detach 前先在原终端上提示用户日志位置，方便后续 tail 查看进度。
    eprintln!("[background] session id : {session_id}");
    eprintln!("[background] log file   : {}", log_path.display());
    eprintln!("[background] 正在脱离终端，关闭本终端不会影响 agent 运行。");

    daemonize(&log_path)?;

    // ---------- 以下在 daemon 子进程中执行 ----------

    let pid_path = PathBuf::from(format!("{session_id}.pid"));
    write_pid_file(&pid_path)?;

    let result = {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(driver::run_with_cli(cli))
    };

    // 任务结束后清理 PID 文件（无论成功还是失败）。
    let _ = std::fs::remove_file(&pid_path);

    result
}

/// 在 daemonize 之前交互式读取后台任务描述。
///
/// 后台模式 detach 后 stdin 会被重定向到 /dev/null，无法再交互输入，
/// 因此必须在 daemonize 之前（仍持有 TTY 时）把任务描述读进来。
/// 复用 PromptEditor 提供与正常交互模式一致的多行编辑体验（补全/历史/粘贴）。
/// 非 TTY 环境（管道输入）时退化为读取 stdin 全部内容。
fn read_task_interactively(
    cli: &ParsedCli,
    session_id: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    // 读取 history_file 配置（与 config::load_config 一致，但不强制 api_key 校验），
    // 仅供 PromptEditor 构造 session 资产目录使用。
    let history_file = crate::commonw::configw::get_all_config()
        .get_opt("history_file")
        .unwrap_or_else(|| "~/.history_file.sqlite".to_string());
    let history_file = PathBuf::from(crate::commonw::utils::expanduser(&history_file).as_ref());

    let mut editor = crate::ai::prompt::PromptEditor::new(session_id, &history_file);
    let model = crate::ai::models::initial_model(cli);
    editor.set_current_model_label(crate::ai::models::model_display_label(&model));
    editor.set_session_topic(Some("后台任务".to_string()));

    match editor.read_multi_line() {
        Ok(input) => Ok(input),
        // Ctrl+C 取消输入，视为空输入。
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// 向 `--stop <session-id>` 指定后台进程发送 SIGTERM。
pub(super) fn stop_background(session_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let pid_path = PathBuf::from(format!("{session_id}.pid"));

    if !pid_path.exists() {
        return Err(format!(
            "PID 文件 {}.pid 不存在（session 可能已完成/从未启动）",
            session_id
        )
        .into());
    }

    let pid_str = std::fs::read_to_string(&pid_path)?;
    let pid: libc::pid_t = pid_str.trim().parse().map_err(|_| {
        format!(
            "PID 文件 {} 内容异常: {}",
            pid_path.display(),
            pid_str.trim()
        )
    })?;

    // 如果进程已不存在，清理 pid 文件并优雅退出。
    let alive = unsafe { libc::kill(pid, 0) } == 0;
    if !alive {
        let _ = std::fs::remove_file(&pid_path);
        return Err(format!(
            "进程 {pid}（session {session_id}）已经不在了（可能已完成），已清理 PID 文件"
        )
        .into());
    }

    // 发 SIGTERM（对应 ctrl+c）。
    eprintln!("[stop] 向 session {session_id}（PID {pid}）发送 SIGTERM...");
    let ret = unsafe { libc::kill(pid, libc::SIGTERM) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        return Err(format!("kill({pid}, SIGTERM) 失败: {err}").into());
    }

    // 等 3 秒让进程优雅退出。
    std::thread::sleep(std::time::Duration::from_secs(3));

    if unsafe { libc::kill(pid, 0) } == 0 {
        eprintln!("[stop] 进程 {pid} 还在运行，可能需要更强力的手段：");
        eprintln!("       kill -9 {pid}");
    } else {
        let _ = std::fs::remove_file(&pid_path);
        eprintln!("[stop] session {session_id}（PID {pid}）已停止。");
    }
    Ok(())
}

/// 把自己的 PID 写入 `.pid` 文件，以便 `--stop` 能找到进程。
fn write_pid_file(pid_path: &Path) -> std::io::Result<()> {
    let pid = process::id() as libc::pid_t;
    std::fs::write(pid_path, pid.to_string())
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
        _ => process::exit(0),
    }

    // 成为新会话组长，脱离控制终端。
    if unsafe { libc::setsid() } == -1 {
        return Err(std::io::Error::last_os_error());
    }

    // 第二次 fork：确保 daemon 不会再意外获取控制终端。
    match unsafe { libc::fork() } {
        -1 => return Err(std::io::Error::last_os_error()),
        0 => {}
        _ => process::exit(0),
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
