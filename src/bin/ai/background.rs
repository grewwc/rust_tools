//! Background mode: detaches the agent from the terminal so the agent keeps
//! running after the terminal closes.
//!
//! Implementation: do not use libc::fork (fork copies the parent's
//! half-initialized CF/os_log/objc state into the child; once the background
//! task hits paths like getaddrinfo / Foundation it SIGBUSes / aborts — the
//! objc_initializeAfterForkError / "child side of fork pre-exec" crashes
//! previously seen under `-bg`). Instead, use posix_spawn to re-exec a fresh
//! `a --daemon-child <session>` process as the daemon (see
//! [`spawn_daemon_child`]). The module entry points are therefore synchronous:
//! the parent only spawns the child and redirects the standard streams; the
//! daemon child calls `setsid` before parsing the CLI to create its own
//! session, then runs the task.

use std::path::{Path, PathBuf};
use std::process;

use crate::ai::cli::ParsedCli;
use crate::ai::driver;

/// The "do not stop midway" directive appended to the user's question in background mode.
const BACKGROUND_DIRECTIVE: &str = "\n\n[后台模式提示] 你正在后台模式运行，发起任务的终端可能已经关闭。\
请务必完整地完成上面交给你的任务，在任务真正完成之前不要停止；\
中途遇到问题就继续排查、调用工具解决，而不是请求人工输入或提前结束。\
完成后把最终结论/产出清晰输出即可。";

/// Parent-process entry for background mode (synchronous): spawns the daemon child before creating the tokio runtime.
///
/// This function first interactively reads the task description (while the TTY
/// is still held), then uses posix_spawn to re-exec a fresh process as the
/// daemon (see [`spawn_daemon_child`]); the parent then `exit(0)`s so the shell
/// returns immediately. The task body lives in [`run_background_child`].
pub(super) fn run_background(mut cli: ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    // Generate the session id (also used as the log file name).
    let session_id = cli
        .session
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone();

    // Background mode: prefer positional args as the task description; if none
    // is provided, read multi-line input interactively (while the TTY is still
    // held) before spawning the child — the child's stdin is redirected to
    // /dev/null, so interactive input is no longer possible there.
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

    // Before spawning, print the log location on the original terminal so the user can tail it for progress.
    eprintln!("[background] session id : {session_id}");
    eprintln!("[background] log file   : {}", log_path.display());
    eprintln!("[background] 正在脱离终端，关闭本终端不会影响 agent 运行。");

    spawn_daemon_child(&cli, &session_id, &log_path)?;

    // Parent exits; the shell returns immediately.
    process::exit(0)
}

/// Daemon child entry point: called by `ai::entry` when it detects `--daemon-child <session_id>`.
///
/// This process is freshly exec'd without any fork, so it does not inherit the
/// parent's half-initialized CF/os_log/objc state and cannot hit crashes like
/// `objc_initializeAfterForkError` or "child side of fork pre-exec".
pub(super) fn run_background_child(
    mut cli: ParsedCli,
    session_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    cli.session = Some(session_id.clone());

    // Append the "do not stop midway" directive to the user's question (next_question joins cli.args).
    cli.args.push(BACKGROUND_DIRECTIVE.to_string());

    let pid_path = PathBuf::from(format!("{session_id}.pid"));
    write_pid_file(&pid_path)?;

    let result = {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(driver::run_with_cli(cli))
    };

    // Clean up the PID file after the task finishes (regardless of success or failure).
    let _ = std::fs::remove_file(&pid_path);

    result
}

/// Create an independent session, detaching from the controlling terminal that launched the background task.
///
/// Only called by the freshly exec'd daemon child before creating the runtime.
/// `CommandExt::process_group(0)` is not a substitute: it only calls `setpgid`,
/// makes the child a process-group leader, and thereby makes `setsid` fail.
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

/// Interactively read the background task description before daemonizing.
///
/// After background-mode detach, stdin is redirected to /dev/null and
/// interactive input is no longer possible, so the task description must be
/// read before daemonizing (while the TTY is still held). Reuses PromptEditor
/// to provide the same multi-line editing experience (completion / history /
/// paste) as normal interactive mode. Falls back to reading all of stdin in
/// non-TTY (piped input) environments.
fn read_task_interactively(
    cli: &ParsedCli,
    session_id: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    // Read the history_file config (consistent with config::load_config, but
    // no api_key validation enforced); only used by PromptEditor to build the
    // session assets directory.
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
        // Ctrl+C cancels input and is treated as empty input.
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Send SIGTERM to the background process named by `--stop <session-id>`.
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

    // If the process no longer exists, clean up the pid file and exit gracefully.
    let alive = unsafe { libc::kill(pid, 0) } == 0;
    if !alive {
        let _ = std::fs::remove_file(&pid_path);
        return Err(format!(
            "进程 {pid}（session {session_id}）已经不在了（可能已完成），已清理 PID 文件"
        )
        .into());
    }

    // Send SIGTERM (equivalent of ctrl+c).
    eprintln!("[stop] sending SIGTERM to session {session_id} (PID {pid})...");
    let ret = unsafe { libc::kill(pid, libc::SIGTERM) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        return Err(format!("kill({pid}, SIGTERM) 失败: {err}").into());
    }

    // Wait 3 seconds to let the process exit gracefully.
    std::thread::sleep(std::time::Duration::from_secs(3));

    if unsafe { libc::kill(pid, 0) } == 0 {
        eprintln!("[stop] process {pid} is still running; a stronger measure may be needed:");
        eprintln!("       kill -9 {pid}");
    } else {
        let _ = std::fs::remove_file(&pid_path);
        eprintln!("[stop] session {session_id} (PID {pid}) stopped.");
    }
    Ok(())
}

/// Write our own PID into the `.pid` file so `--stop` can find the process.
fn write_pid_file(pid_path: &Path) -> std::io::Result<()> {
    let pid = process::id() as libc::pid_t;
    std::fs::write(pid_path, pid.to_string())
}

/// Serialize the parsed `ParsedCli` into the daemon child's argv (excluding argv[0]).
///
/// Fixes a bug in the original `spawn_daemon_child`, which used
/// `std::env::args_os().skip(1)` directly and lost the task description in the
/// "no positional args + interactive input" scenario: the parent writes the
/// task into `cli.args` via `read_task_interactively` in `run_background`, but
/// a child still reading the raw `env::args` would never receive it and would
/// only run an empty prompt plus the background directive. This function
/// rebuilds argv from `cli` as the single source of truth so interactively
/// read tasks are passed through.
pub(crate) fn build_daemon_args(cli: &ParsedCli, session_id: &str) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;
    let mut args: Vec<OsString> = Vec::new();
    // Internal daemon marker; `ai::entry` strips it and injects session_id
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

/// Use posix_spawn to launch a freshly exec'd `a --daemon-child <session>` as the daemon:
///
/// - stdin -> /dev/null, stdout/stderr -> log file (same as the old
///   double-fork behavior);
/// - the daemon child calls `setsid` before parsing the CLI, creating an
///   independent session and detaching from the controlling terminal;
/// - goes through `fork_guard::spawn` (on macOS std `Command` uses
///   posix_spawn, no user-space fork) and skips pthread_atfork / CF / objc
///   fork-safety checks, eliminating at the root the class of `-bg` crashes
///   caused by "inheriting corrupted CF/os_log state after fork".
///
/// Returns in the parent process; the parent then `exit(0)`s so the shell returns immediately.
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
        "background mode (-bg) is unix-only (posix_spawn daemonize)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::cli::parse_cli_args;

    #[test]
    fn daemon_args_contains_interactively_entered_task() {
        // Regression: when `a -bg` has no positional args, the parent writes the
        // task into cli.args in read_task_interactively; the child must receive
        // the same task re-serialized through ParsedCli, not the raw env::args.
        let mut cli = parse_cli_args(["a".to_string(), "-bg".to_string()].into_iter());
        assert!(cli.args.is_empty());
        assert!(cli.background);
        // Simulate interactive input
        let interactive_task = "请帮我重构 auth 模块并补充单测".to_string();
        cli.args = vec![interactive_task.clone()];
        // run_background generates a session_id first and writes it into cli.session
        let session_id = "test-session-123".to_string();
        cli.session = Some(session_id.clone());

        let args = build_daemon_args(&cli, &session_id);
        // Must contain the --daemon-child <session> prefix
        assert_eq!(args[0].to_string_lossy(), "--daemon-child");
        assert_eq!(args[1].to_string_lossy(), session_id);
        // Must contain the interactively entered task (as a trailing positional arg)
        let args_str: Vec<String> = args.iter().map(|s| s.to_string_lossy().to_string()).collect();
        assert!(
            args_str.contains(&interactive_task),
            "daemon args should contain the interactively entered task, actual args={args_str:?}"
        );
        // Must not contain a duplicate --background (the child is already the daemon)
        assert!(
            !args_str.iter().any(|s| s == "--background" || s == "-bg"),
            "daemon args must not contain --background/-bg, actual args={args_str:?}"
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
        // Ensure the serialized args can be correctly reconstructed by parse_cli_args
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
        // Drop the leading --daemon-child <session> pair; the remainder simulates the child's argv after ai::entry strips them
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
