//! 后台模式：把 agent 从终端 detach 出来，关闭终端后 agent 仍能继续运行。
//!
//! 实现方式：不用 libc::fork（fork 会把父进程半初始化的 CF/os_log/objc 状态复制进子进程，
//! 后台任务一旦触发 getaddrinfo / Foundation 等路径就会 SIGBUS / abort，
//! 即此前 `-bg` 的 objc_initializeAfterForkError / "child side of fork pre-exec" 崩溃），
//! 而是用 posix_spawn 重新 exec 一个全新的 `a --daemon-child <session>` 进程作为 daemon
//!（见 [`spawn_daemon_child`]）。因此本模块的入口是同步函数：父进程只负责拉起子进程
//! 并重定向标准流；daemon 子进程会在解析 CLI 前调用 `setsid` 创建独立会话，再执行任务。

use std::path::{Path, PathBuf};
use std::process;

use crate::ai::cli::ParsedCli;
use crate::ai::driver;

/// 后台模式下追加到用户问题后的"不要中途停止"指令。
const BACKGROUND_DIRECTIVE: &str = "\n\n[后台模式提示] 你正在后台模式运行，发起任务的终端可能已经关闭。\
请务必完整地完成上面交给你的任务，在任务真正完成之前不要停止；\
中途遇到问题就继续排查、调用工具解决，而不是请求人工输入或提前结束。\
完成后把最终结论/产出清晰输出即可。";

/// 后台模式父进程入口（同步）：在创建 tokio runtime 之前把 daemon 子进程拉起来。
///
/// 先在本函数里完成交互式读任务描述（仍持有 TTY），随后用 posix_spawn 重新 exec
/// 一个全新进程作为 daemon（见 [`spawn_daemon_child`]），然后父进程 `exit(0)`
/// 让 shell 立刻返回。任务执行体在 [`run_background_child`] 中。
pub(super) fn run_background(mut cli: ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    // 生成 session id（同时作为日志文件名）。
    let session_id = cli
        .session
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone();

    // 后台模式：优先使用位置参数作为任务描述；若未提供位置参数，
    // 则在下发子进程之前（仍持有 TTY）交互式读取多行输入——
    // 子进程 stdin 会被重定向到 /dev/null，无法再交互输入。
    if cli.args.is_empty() {
        match read_task_interactively(&cli, &session_id)? {
            Some(s) if !s.trim().is_empty() => cli.args = vec![s],
            _ => {
                eprintln!("[background] 输入为空，已取消。");
                return Ok(());
            }
        }
    }

    let log_path = std::path::PathBuf::from(format!("{session_id}.log"));

    // 下发前先在原终端上提示用户日志位置，方便后续 tail 查看进度。
    eprintln!("[background] session id : {session_id}");
    eprintln!("[background] log file   : {}", log_path.display());
    eprintln!("[background] 正在脱离终端，关闭本终端不会影响 agent 运行。");

    spawn_daemon_child(&cli, &session_id, &log_path)?;

    // 父进程退出，shell 立刻返回。
    process::exit(0)
}

/// daemon 子进程入口：由 `ai::entry` 在识别到 `--daemon-child <session_id>` 时调用。
///
/// 该进程是全新 exec 出来的，不经过任何 fork，因此不会继承父进程半初始化的
/// CF/os_log/objc 状态，也就不会触发 `objc_initializeAfterForkError` 或
/// "child side of fork pre-exec" 这类崩溃。
pub(super) fn run_background_child(
    mut cli: ParsedCli,
    session_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    cli.session = Some(session_id.clone());

    // 把"不要中途停止"指令拼到用户问题里（next_question 会 join cli.args）。
    cli.args.push(BACKGROUND_DIRECTIVE.to_string());

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

/// 创建独立 session，脱离启动后台任务的控制终端。
///
/// 此函数只由新 exec 的 daemon 子进程在创建 runtime 前调用。不能用
/// `CommandExt::process_group(0)` 替代：后者只会调用 `setpgid`，并会让子进程
/// 成为进程组首领，从而使 `setsid` 失败。
#[cfg(unix)]
pub(super) fn detach_daemon_session() -> std::io::Result<()> {
    if unsafe { libc::setsid() } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn detach_daemon_session() -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "daemon child session detach 仅支持 unix",
    ))
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

/// 把已解析的 `ParsedCli` 序列化为 daemon 子进程的 argv（不含 argv[0]）。
///
/// 修复原有 `spawn_daemon_child` 直接 `std::env::args_os().skip(1)` 导致的
/// 「无位置参数 + 交互输入」场景丢失任务描述的 bug：父进程在 `run_background`
/// 中通过 `read_task_interactively` 把任务写入 `cli.args`，但子进程若仍用
/// 原始 `env::args` 则收不到该任务，只会执行空 prompt + 后台 directive。
/// 本函数改以 `cli` 为唯一真实来源重建参数，确保交互读到的任务被传递。
pub(crate) fn build_daemon_args(cli: &ParsedCli, session_id: &str) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;
    let mut args: Vec<OsString> = Vec::new();
    // 内部 daemon 标记，`ai::entry` 会剥离它并注入 session_id
    args.push(OsString::from("--daemon-child"));
    args.push(OsString::from(session_id));

    if let Some(ref v) = cli.model {
        args.push(OsString::from("--model"));
        args.push(OsString::from(v));
    }
    if let Some(ref v) = cli.agent {
        args.push(OsString::from("--agent"));
        args.push(OsString::from(v));
    }
    if cli.clear {
        args.push(OsString::from("--clear"));
    }
    if cli.new_session {
        args.push(OsString::from("--new-session"));
    }
    if cli.resume {
        args.push(OsString::from("--resume"));
    }
    if let Some(ref v) = cli.session {
        if v != session_id {
            args.push(OsString::from("--session"));
            args.push(OsString::from(v));
        }
    }
    if !cli.files.trim().is_empty() {
        args.push(OsString::from("--files"));
        args.push(OsString::from(&cli.files));
    }
    if cli.list_tools {
        args.push(OsString::from("--list-tools"));
    }
    if cli.list_mcp_tools {
        args.push(OsString::from("--list-mcp-tools"));
    }
    if cli.list_skills {
        args.push(OsString::from("--list-skills"));
    }
    if cli.list_agents {
        args.push(OsString::from("--list-agents"));
    }
    if cli.no_skills {
        args.push(OsString::from("--no-skills"));
    }
    if !cli.mcp_config.trim().is_empty() {
        args.push(OsString::from("--mcp-config"));
        args.push(OsString::from(&cli.mcp_config));
    }
    if cli.help {
        args.push(OsString::from("--help"));
    }
    if cli.interactive {
        args.push(OsString::from("--interactive"));
    }
    if let Some(ref eff) = cli.reasoning_effort_override {
        args.push(OsString::from("--reasoning-effort"));
        match eff {
            Some(level) => args.push(OsString::from(level.as_str())),
            None => args.push(OsString::from("off")),
        }
    }
    if cli.note_search {
        args.push(OsString::from("--note-search"));
    }
    if cli.note_flag {
        args.push(OsString::from("--note"));
        if let Some(ref v) = cli.note {
            args.push(OsString::from(v));
        }
    }
    if let Some(ref v) = cli.note_delete {
        args.push(OsString::from("--note-delete"));
        args.push(OsString::from(v));
    }
    if let Some(ref v) = cli.note_edit {
        args.push(OsString::from("--note-edit"));
        args.push(OsString::from(v));
    }
    if cli.consolidate_knowledge {
        args.push(OsString::from("--consolidate-knowledge"));
    }
    if cli.generate_completions {
        args.push(OsString::from("--generate-completions"));
    }
    for a in &cli.args {
        args.push(OsString::from(a));
    }
    args
}

/// 用 posix_spawn 拉一个全新 exec 的 `a --daemon-child <session>` 作为 daemon：
///
/// - stdin -> /dev/null，stdout/stderr -> 日志文件（与旧 double-fork 行为一致）；
/// - daemon 子进程会在解析 CLI 前调用 `setsid`，创建独立 session 并脱离控制终端；
/// - 走 `fork_guard::spawn`（macOS 上 std `Command` 用 posix_spawn，不做用户态 fork），
///   不执行 pthread_atfork / CF / objc 的 fork 安全检查，
///   从根上消除 `-bg` 模式下"fork 后继承损坏的 CF/os_log 状态"这一类崩溃。
///
/// 该函数在父进程中返回；父进程随后 `exit(0)`，让 shell 立刻返回。
#[cfg(unix)]
fn spawn_daemon_child(cli: &ParsedCli, session_id: &str, log_path: &Path) -> std::io::Result<()> {
    use std::ffi::OsString;

    let exe = std::env::current_exe()?;
    let args: Vec<OsString> = build_daemon_args(cli, session_id);

    let dev_null = std::fs::OpenOptions::new().read(true).open("/dev/null")?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let mut cmd = std::process::Command::new(exe);
    cmd.args(&args)
        .stdin(dev_null)
        .stdout(log.try_clone()?)
        .stderr(log);
    crate::fork_guard::spawn(&mut cmd)?;
    Ok(())
}

#[cfg(not(unix))]
fn spawn_daemon_child(_cli: &ParsedCli, _session_id: &str, _log_path: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "background mode (-bg) 仅支持 unix（posix_spawn daemonize）",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::cli::parse_cli_args;

    #[test]
    fn daemon_args_contains_interactively_entered_task() {
        // 回归：a -bg 无位置参数时，父进程在 read_task_interactively 中把任务写入 cli.args，
        // 子进程必须通过 ParsedCli 重新序列化收到同一任务，而不是用原始 env::args。
        let mut cli = parse_cli_args(["a".to_string(), "-bg".to_string()].into_iter());
        assert!(cli.args.is_empty());
        assert!(cli.background);
        // 模拟交互输入
        let interactive_task = "请帮我重构 auth 模块并补充单测".to_string();
        cli.args = vec![interactive_task.clone()];
        // run_background 会先生成 session_id 并写入 cli.session
        let session_id = "test-session-123".to_string();
        cli.session = Some(session_id.clone());

        let args = build_daemon_args(&cli, &session_id);
        // 必须包含 --daemon-child <session> 前缀
        assert_eq!(args[0].to_string_lossy(), "--daemon-child");
        assert_eq!(args[1].to_string_lossy(), session_id);
        // 必须包含交互输入的任务（作为位置参数在末尾）
        let args_str: Vec<String> = args.iter().map(|s| s.to_string_lossy().to_string()).collect();
        assert!(
            args_str.contains(&interactive_task),
            "daemon args 应包含交互输入的任务，实际 args={args_str:?}"
        );
        // 不应包含重复的 --background（子进程已是 daemon）
        assert!(
            !args_str.iter().any(|s| s == "--background" || s == "-bg"),
            "daemon args 不应包含 --background/-bg，实际 args={args_str:?}"
        );
    }

    #[test]
    fn daemon_args_preserves_cli_flags_and_positional() {
        let cli = parse_cli_args(
            [
                "a".to_string(),
                "--model".to_string(),
                "gpt-test".to_string(),
                "--files".to_string(),
                "a.txt,b.txt".to_string(),
                "-bg".to_string(),
                "fix the bug".to_string(),
            ]
            .into_iter(),
        );
        let session_id = "sess-xyz".to_string();
        let mut cli = cli;
        cli.session = Some(session_id.clone());
        let args = build_daemon_args(&cli, &session_id);
        let args_str: Vec<String> = args.iter().map(|s| s.to_string_lossy().to_string()).collect();
        assert!(args_str.contains(&"--model".to_string()));
        assert!(args_str.contains(&"gpt-test".to_string()));
        assert!(args_str.contains(&"--files".to_string()));
        assert!(args_str.contains(&"a.txt,b.txt".to_string()));
        assert!(args_str.contains(&"fix the bug".to_string()));
    }

    #[test]
    fn daemon_args_roundtrip_via_parse() {
        // 确保序列化后的 args 能被 parse_cli_args 正确还原
        let mut cli = parse_cli_args(
            [
                "a".to_string(),
                "--model".to_string(),
                "my-model".to_string(),
                "--reasoning-effort".to_string(),
                "high".to_string(),
                "-bg".to_string(),
                "do something important".to_string(),
            ]
            .into_iter(),
        );
        let session_id = "roundtrip-sess".to_string();
        cli.session = Some(session_id.clone());
        let daemon_args = build_daemon_args(&cli, &session_id);
        // 去掉前两项 --daemon-child <session>，剩余部分模拟子进程在 ai::entry 剥离后的 argv
        let child_argv: Vec<String> = std::iter::once("a".to_string())
            .chain(daemon_args.iter().skip(2).map(|s| s.to_string_lossy().to_string()))
            .collect();
        let reparsed = parse_cli_args(child_argv.into_iter());
        assert_eq!(reparsed.args, vec!["do something important".to_string()]);
        assert_eq!(reparsed.model.as_deref(), Some("my-model"));
        assert_eq!(
            reparsed.reasoning_effort_override,
            Some(Some(crate::ai::provider::ReasoningEffort::High))
        );
    }
}
