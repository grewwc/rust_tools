//! Command execution.
//!
//! Executes system commands, with timeout control and working-directory
//! configuration.

use crate::{commonw::utils::expanduser, strw::split::split_space_keep_symbol};

use std::borrow::Cow;
use std::{
    ffi::OsString,
    fs::File,
    io::{self, Read},
    process::{Child, Command, ExitStatus, Output, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::{io::FromRawFd, process::CommandExt};

/// Command execution options.
///
/// Configures how a command is executed.
///
/// # Fields
///
/// * `cwd` - Optional working-directory path
///
/// # Examples
///
/// ```rust
/// use rust_tools::cmd::RunCmdOptions;
///
/// // Run in the current directory
/// let opts = RunCmdOptions::default();
///
/// // Run in a specific directory
/// let opts = RunCmdOptions {
///     cwd: Some("/tmp"),
/// };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RunCmdOptions<'a> {
    /// Working directory for command execution.
    ///
    /// If `None`, the command runs in the current process's working directory.
    pub cwd: Option<&'a str>,
}

/// Normalize a command string.
///
/// Trims leading/trailing whitespace and rejects empty commands.
///
/// # Arguments
///
/// * `command` - Raw command string
///
/// # Returns
///
/// - `Ok(&str)` - The normalized command
/// - `Err(io::Error)` - The command is empty
fn normalize_command(command: &str) -> io::Result<&str> {
    let command = command.trim();
    if command.is_empty() {
        return Err(io::Error::other("empty command"));
    }
    Ok(command)
}

fn is_shell_boundary(byte: Option<u8>) -> bool {
    byte.is_none_or(|b| {
        b.is_ascii_whitespace() || matches!(b, b'|' | b'&' | b';' | b'<' | b'>' | b'(' | b')')
    })
}

/// Determines whether a command must be executed through a shell.
///
/// Only syntax that genuinely needs shell interpretation outside quotes is
/// recognized, so literals inside double quotes such as `<` / `>` / `|` are not
/// misclassified as requiring `sh -c`.
///
/// Note: this function still conservatively treats single quotes, backslashes,
/// and `$` (variable expansion) as requiring a shell, because the current
/// `build_no_shell_command` only implements double-quote grouping and does not
/// fully replicate shell escaping/quoting/variable-expansion semantics.
fn should_use_shell(command: &str) -> bool {
    let bytes = command.as_bytes();
    let mut i = 0usize;
    let mut in_double = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_double {
            match b {
                b'\\' | b'`' => return true,
                b'"' => {
                    in_double = false;
                    i += 1;
                    continue;
                }
                b'$' => return true,
                _ => {
                    i += 1;
                    continue;
                }
            }
        }

        match b {
            b'"' => {
                in_double = true;
            }
            b'\'' | b'\\' | b'`' => return true,
            b'$' => return true,
            b'|' | b'>' | b'<' | b';' | b'&' | b'*' | b'?' => return true,
            b'#' if is_shell_boundary(i.checked_sub(1).map(|idx| bytes[idx])) => return true,
            // `(` is only treated as shell grouping/subshell syntax at a token
            // start boundary; `foo(bar)` in an ordinary argument must not force
            // the command into a shell.
            b'(' if is_shell_boundary(i.checked_sub(1).map(|idx| bytes[idx])) => {
                return true;
            }
            // `)` is only treated as closing shell grouping at a token end boundary.
            b')' if is_shell_boundary(bytes.get(i + 1).copied()) => {
                return true;
            }
            b'\n' => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

/// Returns whether the command takes the shell execution path.
///
/// Reused by upper-layer safety validation so that validation semantics stay
/// consistent with actual execution semantics.
pub fn command_requires_shell(command: &str) -> bool {
    should_use_shell(command)
}

/// Builds a shell-based `Command`.
///
/// Picks a suitable shell for the current OS:
/// - Windows: `cmd /C`
/// - Unix-like: `sh -c`
///
/// # Arguments
///
/// * `command` - The command to execute
/// * `opts` - Execution options
fn build_shell_command(command: &str, opts: RunCmdOptions<'_>) -> Command {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    cmd
}

/// Builds a `Command`.
///
/// Automatically decides whether a shell is required and builds the
/// corresponding `Command`.
///
/// # Arguments
///
/// * `command` - The command to execute
/// * `opts` - Execution options
///
/// # Returns
///
/// - `Ok(Command)` - The command was built successfully
/// - `Err(io::Error)` - Building failed
fn build_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;
    if should_use_shell(command) {
        Ok(build_shell_command(command, opts))
    } else {
        build_no_shell_command(command, opts)
    }
}

/// Builds a `Command` without a shell.
///
/// Parses the command and arguments directly to avoid shell-injection risk.
///
/// # Arguments
///
/// * `command` - The command to execute (including arguments)
/// * `opts` - Execution options
///
/// # Returns
///
/// - `Ok(Command)` - The command was built successfully
/// - `Err(io::Error)` - Building failed
fn build_no_shell_command(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Command> {
    let command = normalize_command(command)?;

    // Split the command into program and arguments.
    let mut iter = split_space_keep_symbol(command, r#"""#);
    let Some(program) = iter.next() else {
        return Err(io::Error::other("empty command"));
    };

    // The program name also supports leading `~` expansion (matching shell
    // semantics); quoted names are treated literally.
    let mut cmd = if program.starts_with('"') {
        Command::new(program)
    } else {
        match expanduser(program) {
            Cow::Borrowed(p) => Command::new(p),
            Cow::Owned(p) => Command::new(p),
        }
    };
    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    // Handle arguments, expanding an unquoted leading `~` (shell semantics).
    // On the non-shell path, double quotes only serve to group tokens and must
    // not be passed to the child process as literals. A leading `~` inside
    // quotes is a literal and is not expanded (e.g. `echo "~"` should print
    // `~`). This does not attempt to replicate full shell semantics; more
    // complex single-quote/backslash escaping is still conservatively routed to
    // the shell path by `should_use_shell`.
    iter.for_each(|arg| {
        let normalized_arg = if arg.contains('"') {
            arg.replace('"', "")
        } else {
            arg.to_string()
        };
        // Shell rule: only a word whose first character is an unquoted `~` is
        // expanded; a leading quote (`"~"`) or a quote right after `~`
        // (`~"x"`, `~"/foo"`) is a literal.
        let new_arg = if arg.starts_with('"') || arg.starts_with(r#"~""#) {
            Cow::Borrowed(normalized_arg.as_str())
        } else {
            expanduser(&normalized_arg)
        };
        if new_arg == normalized_arg {
            cmd.arg(OsString::from(new_arg.as_ref()));
        } else {
            cmd.arg(OsString::from(new_arg.into_owned()));
        }
    });
    Ok(cmd)
}

#[cfg(unix)]
fn configure_child_process_group(cmd: &mut Command) {
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_child_process_group(_cmd: &mut Command) {}

/// Makes the child process adopt the slave PTY as its controlling terminal.
/// Wiring only stdout to the PTY is not enough for the TTY detection of some
/// CLIs; they also check stdin/stderr or whether a controlling terminal exists.
#[cfg(unix)]
fn configure_child_process_group_with_controlling_terminal(cmd: &mut Command) {
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY.into(), 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Opens a PTY pair. The master is read only by the parent; the slave becomes
/// the child's stdin/stdout/stderr.
#[cfg(unix)]
fn open_pseudo_terminal() -> io::Result<(File, File)> {
    let mut master = -1;
    let mut slave = -1;
    let result = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }

    // On openpty success, ownership of both fds has moved into the `File`s;
    // they are closed automatically when they go out of scope.
    Ok(unsafe { (File::from_raw_fd(master), File::from_raw_fd(slave)) })
}

#[cfg(unix)]
fn terminate_child(child: &mut Child) {
    let pgid = child.id() as libc::pid_t;
    unsafe {
        let _ = libc::kill(-pgid, libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) {
    let _ = child.kill();
}

/// After the foreground command exits, detects whether its process group still
/// has live members (typically long-running services spawned with `&`, such as
/// `python app.py &`). `kill(-pgid, 0)` sends no signal; it only probes:
/// a return of 0 means the group still has members, `ESRCH` means none remain.
///
/// Because the child becomes the group leader via `setsid`, `child.id()` is the
/// pgid of that process group; even after the foreground leader has been
/// reaped, the pgid is not reclaimed as long as background members are alive,
/// so this probe is safe.
#[cfg(unix)]
fn background_group_alive(pgid: u32) -> bool {
    unsafe { libc::kill(-(pgid as libc::pid_t), 0) == 0 }
}

#[cfg(not(unix))]
fn background_group_alive(_pgid: u32) -> bool {
    false
}

/// Executes a command and returns its output.
///
/// Runs the command and captures stdout and stderr.
///
/// # Arguments
///
/// * `command` - The command to execute
/// * `opts` - Execution options
///
/// # Returns
///
/// - `Ok(Output)` - The command ran successfully; returns the output
/// - `Err(io::Error)` - The command failed
///
/// # Examples
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd_output;
///
/// let output = run_cmd_output("ls -la", Default::default())
///     .expect("命令执行失败");
///
/// println!("状态码：{}", output.status);
/// println!("输出：{}", String::from_utf8_lossy(&output.stdout));
/// ```
pub fn run_cmd_output(command: &str, opts: RunCmdOptions<'_>) -> io::Result<Output> {
    crate::fork_guard::output(&mut build_command(command, opts)?)
}

/// Executes a command with timeout control.
///
/// Runs the command and terminates it if it does not finish within the given
/// time.
///
/// # Arguments
///
/// * `command` - The command to execute
/// * `opts` - Execution options
/// * `timeout` - Timeout duration
///
/// # Returns
///
/// - `Ok(Output)` - The command completed before the timeout
/// - `Err(io::Error)` - The command failed or timed out
///   - `ErrorKind::TimedOut` - The command timed out
///
/// # Examples
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd_output_with_timeout;
/// use std::time::Duration;
///
/// match run_cmd_output_with_timeout(
///     "sleep 5",
///     Default::default(),
///     Duration::from_secs(2),
/// ) {
///     Ok(output) => println!("完成：{}", String::from_utf8_lossy(&output.stdout)),
///     Err(e) if e.kind() == std::io::ErrorKind::TimedOut => println!("超时"),
///     Err(e) => println!("错误：{}", e),
/// }
/// ```
///
/// # Notes
///
/// - The child process is terminated on timeout
/// - stdin is set to null
pub fn run_cmd_output_with_timeout(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
) -> io::Result<Output> {
    run_cmd_output_streaming_with_timeout(command, opts, timeout, |_| {}, || false)
}

pub fn run_cmd_output_with_timeout_non_interactive(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
) -> io::Result<Output> {
    let result = run_cmd_output_streaming_with_timeout_tracked_non_interactive(
        command,
        opts,
        timeout,
        |_| {},
        || false,
        |_| {},
    )?;
    result_to_output(result)
}

#[derive(Clone, Copy)]
enum StreamKind {
    Stdout,
    Stderr,
}

/// Reader thread: tags each chunk with its stream kind and sends it over a
/// channel back to the main thread for accumulation.
///
/// Key point: the thread no longer accumulates data itself and is not rejoined
/// via `join`. If a command spawns a long-lived process in the background
/// (e.g. Flask started by `python app.py &`), that child inherits the same
/// stdout/stderr pipe write-end fd, so `read()` never sees EOF and the thread
/// never exits. Joining such a thread would deadlock the main thread. With
/// pure channel reporting, the main thread can return as soon as the
/// foreground command exits, leaking the reader thread as a daemon (it blocks
/// on a harmless pipe read and is reclaimed when the process exits).
fn spawn_pipe_reader<R>(mut reader: R, kind: StreamKind, tx: Sender<(StreamKind, Vec<u8>)>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(read) => {
                    if tx.send((kind, buf[..read].to_vec())).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    let _ = err;
                    break;
                }
            }
        }
    });
}

/// After the foreground command exits, gives the reader thread a short grace
/// period to drain the tail output already buffered in the pipe, then returns.
/// If the command spawned a background process that inherited the pipe, the
/// channel never becomes `Disconnected`, so a fixed grace period is mandatory
/// and waiting must be bounded.
const DRAIN_GRACE: Duration = Duration::from_millis(100);

/// Stall detection for PTY interactive commands: when a command is alive on the
/// PTY but produces no output for a long time, it is almost certainly waiting
/// for human input (QR scan, password, menu choice) that the agent cannot
/// provide; when output is buffered through a pipe (e.g. `| tail`) it is
/// invisible from start to finish. In that case terminate early and return the
/// partial output plus an explicit diagnosis instead of hanging silently until
/// the 60-300s timeout backstop.
/// - `PTY_STALL_AFTER_OUTPUT`: once the command has produced output, continued
///   silence beyond this duration is judged a stall.
/// - `PTY_STALL_SILENT_START`: a command with no output at all since startup
///   that keeps running beyond this duration is judged a stall.
const PTY_STALL_AFTER_OUTPUT: Duration = Duration::from_secs(10);
const PTY_STALL_SILENT_START: Duration = Duration::from_secs(20);

fn drain_channel<F>(
    rx: &Receiver<(StreamKind, Vec<u8>)>,
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    on_chunk: &mut F,
    grace: Duration,
) where
    F: FnMut(&[u8]),
{
    let deadline = Instant::now() + grace;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // Still take any data that is ready, but do not wait for more.
            while let Ok((kind, chunk)) = rx.try_recv() {
                accumulate_chunk(kind, &chunk, stdout_buf, stderr_buf, on_chunk);
            }
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok((kind, chunk)) => accumulate_chunk(kind, &chunk, stdout_buf, stderr_buf, on_chunk),
            Err(RecvTimeoutError::Timeout) => break,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn accumulate_chunk<F>(
    kind: StreamKind,
    chunk: &[u8],
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
    on_chunk: &mut F,
) where
    F: FnMut(&[u8]),
{
    match kind {
        StreamKind::Stdout => stdout_buf.extend_from_slice(chunk),
        StreamKind::Stderr => stderr_buf.extend_from_slice(chunk),
    }
    on_chunk(chunk);
}

/// Command execution result: carries stdout/stderr and uses flags to
/// distinguish the "killed by timeout/cancel" cases.
///
/// On timeout or cancellation the partially captured output is still preserved
/// (`status` may be `None`) so the upper layer can hand the caller whatever was
/// produced before the kill, rather than an uninformative
/// timeout/cancelled-only error.
#[derive(Debug)]
pub struct CommandRunResult {
    pub status: Option<ExitStatus>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
    pub cancelled: bool,
    /// The PTY command was judged "stalled" (alive but silent for a long time,
    /// likely waiting for interactive input) and terminated early.
    pub stalled: bool,
}

#[derive(Clone, Copy)]
struct CommandEnvPolicy {
    suppress_pagers: bool,
    suppress_interaction: bool,
}

const INHERITED_ENV_POLICY: CommandEnvPolicy = CommandEnvPolicy {
    suppress_pagers: false,
    suppress_interaction: false,
};
const NON_INTERACTIVE_ENV_POLICY: CommandEnvPolicy = CommandEnvPolicy {
    suppress_pagers: true,
    suppress_interaction: true,
};
// PTY output is still captured by the parent, so pagers are disabled; but
// prompt, editor, terminal, and color environment are kept so explicitly
// interactive commands keep running in real-terminal mode.
const PSEUDO_TERMINAL_ENV_POLICY: CommandEnvPolicy = CommandEnvPolicy {
    suppress_pagers: true,
    suppress_interaction: false,
};

fn apply_pager_suppression_env(cmd: &mut Command) {
    cmd.env("PAGER", "cat").env("GIT_PAGER", "cat");
}

fn apply_interaction_suppression_env(cmd: &mut Command) {
    cmd.env("GIT_EDITOR", "true")
        .env("GIT_SEQUENCE_EDITOR", "true")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0");
}

fn apply_command_env_policy(cmd: &mut Command, policy: CommandEnvPolicy) {
    if policy.suppress_pagers {
        apply_pager_suppression_env(cmd);
    }
    if policy.suppress_interaction {
        apply_interaction_suppression_env(cmd);
    }
}

/// Converts a result carrying timeout/cancel markers back to the legacy
/// `io::Result<Output>` semantics:
/// timeout -> `Err(TimedOut)`, cancel -> `Err(Interrupted)`, normal ->
/// `Ok(Output)`. Legacy paths (hooks, non-streaming
/// `run_cmd_output_with_timeout`) do not need partial output and only care
/// about success/failure, so the signature is unchanged and this function
/// discards the partial output of killed cases.
fn result_to_output(r: CommandRunResult) -> io::Result<Output> {
    if r.timed_out {
        Err(io::Error::new(io::ErrorKind::TimedOut, "timeout"))
    } else if r.cancelled {
        Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
    } else if r.stalled {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "stalled (likely waiting for interactive input)",
        ))
    } else {
        Ok(Output {
            status: r.status.expect("non-killed result must carry a status"),
            stdout: r.stdout,
            stderr: r.stderr,
        })
    }
}

pub fn run_cmd_output_streaming_with_timeout<F, C>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    on_chunk: F,
    should_cancel: C,
) -> io::Result<Output>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
{
    let r = run_cmd_output_streaming_with_timeout_tracked(
        command,
        opts,
        timeout,
        on_chunk,
        should_cancel,
        |_| {},
    )?;
    result_to_output(r)
}

/// Same as [`run_cmd_output_streaming_with_timeout`], but additionally accepts
/// an `on_background_group` callback: after the foreground command exits
/// normally, if its process group still has live members (the command spawned a
/// long-running service with `&`, e.g. `python app.py &`), the callback is
/// invoked once with that group's pgid. The upper layer can use it to register
/// the pgid in a session-level registry and clean up uniformly when the session
/// ends, avoiding orphan processes.
///
/// Note: a pgid is only meaningful while the current process is alive and must
/// not be persisted to disk — after a restart the same value may be reused by
/// an unrelated process, and `killpg` would then kill the wrong target.
pub fn run_cmd_output_streaming_with_timeout_tracked<F, C, G>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    on_chunk: F,
    should_cancel: C,
    on_background_group: G,
) -> io::Result<CommandRunResult>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
    G: FnMut(u32),
{
    run_cmd_output_streaming_with_timeout_tracked_inner(
        command,
        opts,
        timeout,
        on_chunk,
        should_cancel,
        on_background_group,
        false,
        INHERITED_ENV_POLICY,
        None,
    )
}

pub fn run_cmd_output_streaming_with_timeout_tracked_non_interactive<F, C, G>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    on_chunk: F,
    should_cancel: C,
    on_background_group: G,
) -> io::Result<CommandRunResult>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
    G: FnMut(u32),
{
    run_cmd_output_streaming_with_timeout_tracked_inner(
        command,
        opts,
        timeout,
        on_chunk,
        should_cancel,
        on_background_group,
        false,
        NON_INTERACTIVE_ENV_POLICY,
        None,
    )
}

/// Same as [`run_cmd_output_streaming_with_timeout_tracked`], but runs the
/// child in a PTY. Intended only for explicit calls that need terminal
/// capabilities (e.g. QR-code login, full-screen interactive CLIs); this path
/// disables only pagers that could block captured output, not prompts, editors,
/// terminal capabilities, or color. Regular commands should still use pipes to
/// keep stdout/stderr separation and the non-interactive semantics of ordinary
/// logs.
pub fn run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal<F, C, G>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    on_chunk: F,
    should_cancel: C,
    on_background_group: G,
) -> io::Result<CommandRunResult>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
    G: FnMut(u32),
{
    run_cmd_output_streaming_with_timeout_tracked_inner(
        command,
        opts,
        timeout,
        on_chunk,
        should_cancel,
        on_background_group,
        true,
        PSEUDO_TERMINAL_ENV_POLICY,
        None,
    )
}

fn run_cmd_output_streaming_with_timeout_tracked_inner<F, C, G>(
    command: &str,
    opts: RunCmdOptions<'_>,
    timeout: Duration,
    mut on_chunk: F,
    should_cancel: C,
    mut on_background_group: G,
    pseudo_terminal: bool,
    env_policy: CommandEnvPolicy,
    // Stall-detection thresholds `(after_output, silent_start)` for PTY;
    // `None` uses the defaults. Tests may inject tiny values to exercise the
    // stall path quickly; production always passes `None`.
    pty_stall: Option<(Duration, Duration)>,
) -> io::Result<CommandRunResult>
where
    F: FnMut(&[u8]),
    C: Fn() -> bool,
    G: FnMut(u32),
{
    let mut cmd = build_command(command, opts)?;
    apply_command_env_policy(&mut cmd, env_policy);

    let (mut child, rx) = if pseudo_terminal {
        #[cfg(unix)]
        {
            let (master, slave) = open_pseudo_terminal()?;
            cmd.stdin(Stdio::from(slave.try_clone()?));
            cmd.stdout(Stdio::from(slave.try_clone()?));
            cmd.stderr(Stdio::from(slave));
            configure_child_process_group_with_controlling_terminal(&mut cmd);

            let child = crate::fork_guard::spawn(&mut cmd)?;
            let (tx, rx) = mpsc::channel::<(StreamKind, Vec<u8>)>();
            // The PTY merges stdout/stderr into the same master; store it in
            // the stdout slot to keep the existing "merged output" result
            // semantics for the upper layer.
            spawn_pipe_reader(master, StreamKind::Stdout, tx);
            (child, rx)
        }
        #[cfg(not(unix))]
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "pseudo terminal is only supported on Unix",
            ));
        }
    } else {
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_child_process_group(&mut cmd);

        let mut child = crate::fork_guard::spawn(&mut cmd)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing stderr pipe"))?;
        let (tx, rx) = mpsc::channel::<(StreamKind, Vec<u8>)>();
        spawn_pipe_reader(stdout, StreamKind::Stdout, tx.clone());
        spawn_pipe_reader(stderr, StreamKind::Stderr, tx);
        (child, rx)
    };
    let pgid = child.id();

    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();

    let deadline = Instant::now() + timeout;
    // For PTY stall detection: command start time and the time of the most
    // recent output received.
    let started_at = Instant::now();
    let mut last_output_at = started_at;
    let status = loop {
        while let Ok((kind, chunk)) = rx.try_recv() {
            accumulate_chunk(
                kind,
                &chunk,
                &mut stdout_buf,
                &mut stderr_buf,
                &mut on_chunk,
            );
            last_output_at = Instant::now();
        }

        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if should_cancel() {
                    // Cancel: kill the whole process group (including spawned
                    // background processes); the pipes close and reader threads exit.
                    terminate_child(&mut child);
                    let killed_status = child.wait().ok();
                    drain_channel(
                        &rx,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &mut on_chunk,
                        DRAIN_GRACE,
                    );
                    // On cancel, still return the partially captured output
                    // (status may be None if wait failed) so the upper layer
                    // can hand the model whatever was produced before the
                    // cancellation, not just a bare cancelled marker.
                    return Ok(CommandRunResult {
                        status: killed_status,
                        stdout: stdout_buf,
                        stderr: stderr_buf,
                        timed_out: false,
                        cancelled: true,
                        stalled: false,
                    });
                }
                // PTY interactive-command stall detection: the command is alive
                // but silent for a long time, almost certainly waiting for
                // human input (QR scan, password, menu choice) that the agent
                // cannot provide; output buffered through a pipe (e.g.
                // `| tail`) is invisible entirely. Terminate early and return
                // the captured partial output plus the stalled flag so the
                // upper layer can report an explicit "waiting for interactive
                // input" diagnosis instead of hanging silently into the
                // 60-300s timeout backstop.
                if pseudo_terminal {
                    let now = Instant::now();
                    let (after_output, silent_start) =
                        pty_stall.unwrap_or((PTY_STALL_AFTER_OUTPUT, PTY_STALL_SILENT_START));
                    let produced_output = !stdout_buf.is_empty() || !stderr_buf.is_empty();
                    let stalled = if produced_output {
                        now.duration_since(last_output_at) >= after_output
                    } else {
                        now.duration_since(started_at) >= silent_start
                    };
                    if stalled {
                        terminate_child(&mut child);
                        let killed_status = child.wait().ok();
                        drain_channel(
                            &rx,
                            &mut stdout_buf,
                            &mut stderr_buf,
                            &mut on_chunk,
                            DRAIN_GRACE,
                        );
                        return Ok(CommandRunResult {
                            status: killed_status,
                            stdout: stdout_buf,
                            stderr: stderr_buf,
                            timed_out: false,
                            cancelled: false,
                            stalled: true,
                        });
                    }
                }
                if Instant::now() >= deadline {
                    // Timeout: likewise kill the whole process group so
                    // background processes stop holding the pipes.
                    terminate_child(&mut child);
                    let killed_status = child.wait().ok();
                    drain_channel(
                        &rx,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &mut on_chunk,
                        DRAIN_GRACE,
                    );
                    // On timeout, still return the captured partial output
                    // (typically progress and errors a long build/test printed
                    // before being killed) so the model can decide the next
                    // step instead of seeing only an uninformative "timeout".
                    return Ok(CommandRunResult {
                        status: killed_status,
                        stdout: stdout_buf,
                        stderr: stderr_buf,
                        timed_out: true,
                        cancelled: false,
                        stalled: false,
                    });
                }
                match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok((kind, chunk)) => {
                        accumulate_chunk(
                            kind,
                            &chunk,
                            &mut stdout_buf,
                            &mut stderr_buf,
                            &mut on_chunk,
                        );
                        last_output_at = Instant::now();
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {}
                }
            }
            Err(err) => return Err(err),
        }
    };

    // The foreground command has exited. If it spawned a background process
    // that inherited the pipes (e.g. `flask &`), the channel never becomes
    // Disconnected; drain the tail output within the fixed grace period and
    // return, never joining the reader threads.
    drain_channel(
        &rx,
        &mut stdout_buf,
        &mut stderr_buf,
        &mut on_chunk,
        DRAIN_GRACE,
    );

    // The foreground leader has exited; report the pgid if background members
    // of the group are still alive, for session-level cleanup.
    if background_group_alive(pgid) {
        on_background_group(pgid);
    }

    Ok(CommandRunResult {
        status: Some(status),
        stdout: stdout_buf,
        stderr: stderr_buf,
        timed_out: false,
        cancelled: false,
        stalled: false,
    })
}

/// Executes a command and returns the output text.
///
/// Runs the command and returns stdout and stderr merged into a single string.
///
/// # Arguments
///
/// * `command` - The command to execute
///
/// # Returns
///
/// - `Ok(String)` - The command output (stdout + stderr)
/// - `Err(io::Error)` - The command failed
///
/// # Examples
///
/// ```rust,no_run
/// use rust_tools::cmd::run_cmd;
///
/// let output = run_cmd("echo Hello").expect("命令执行失败");
/// println!("输出：{}", output);
/// ```
///
/// # Notes
///
/// - An empty command returns an empty string
/// - stdout and stderr are merged
pub fn run_cmd(command: &str) -> io::Result<String> {
    if command.trim().is_empty() {
        return Ok("".to_owned());
    }

    let output = run_cmd_output(command, RunCmdOptions::default())?;
    let mut result = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !stderr.is_empty() {
        result.push_str(&stderr);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{
        RunCmdOptions, run_cmd, run_cmd_output, run_cmd_output_streaming_with_timeout,
        run_cmd_output_streaming_with_timeout_tracked, should_use_shell,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn test_run_cmd_basic() {
        #[cfg(unix)]
        {
            let output = run_cmd("echo test").unwrap();
            assert!(output.contains("test"));
        }
    }

    #[test]
    fn test_run_cmd_empty() {
        let output = run_cmd("").unwrap();
        assert_eq!(output, "");
    }

    #[test]
    fn test_run_cmd_output_basic() {
        #[cfg(unix)]
        {
            let output = run_cmd_output("echo hello", RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert!(output.stdout.contains(&b'h'));
        }
    }

    #[test]
    fn test_should_use_shell_ignores_double_quoted_literals() {
        assert!(!should_use_shell(r#"printf "%s" "<literal>|foo(bar)#bar""#));
        assert!(!should_use_shell(r#"printf "%s" ">(literal)""#));
    }

    #[test]
    fn test_should_use_shell_detects_real_shell_syntax_outside_quotes() {
        assert!(should_use_shell("cat < input.txt"));
        assert!(should_use_shell("echo hi | wc -c"));
        assert!(should_use_shell("diff <(echo a) <(echo b)"));
        assert!(should_use_shell("echo foo # comment"));
    }

    #[test]
    fn test_should_use_shell_does_not_treat_hash_in_word_as_comment() {
        assert!(!should_use_shell("printf %s foo#bar"));
    }

    #[test]
    fn test_should_use_shell_routes_variable_expansion_to_shell() {
        // A plain `$VAR` is shell variable expansion and must take the shell
        // path, otherwise it would be passed to the child as a literal.
        assert!(should_use_shell("echo $HOME"));
        assert!(should_use_shell(r#"echo "$HOME""#));
        assert!(should_use_shell("ls $DIR"));
        assert!(should_use_shell("echo $?"));
    }

    #[test]
    fn test_run_cmd_output_expands_plain_variable_in_shell() {
        #[cfg(unix)]
        {
            let home = std::env::var("HOME").unwrap();
            let output = run_cmd_output("echo $HOME", RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), home);
        }
    }

    #[test]
    fn test_run_cmd_output_keeps_quoted_tilde_literal() {
        // Shell semantics: a `~` inside quotes is a literal and is not expanded.
        #[cfg(unix)]
        {
            let output = run_cmd_output(r#"echo "~""#, RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim_end(), "~");
        }
    }

    #[test]
    fn test_run_cmd_output_keeps_tilde_followed_by_quote_literal() {
        // Shell semantics: a quote right after `~` likewise blocks expansion
        // (`~"/foo"` → literal `~/foo`, `~"x"` → literal `~x`), matching
        // observed `sh` behavior.
        #[cfg(unix)]
        {
            let output = run_cmd_output(r#"echo ~"/foo""#, RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(
                String::from_utf8_lossy(&output.stdout).trim_end(),
                r#"~/foo"#
            );

            let output = run_cmd_output(r#"echo ~"x""#, RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim_end(), r#"~x"#);
        }
    }

    #[test]
    fn test_run_cmd_output_expands_unquoted_tilde() {
        #[cfg(unix)]
        {
            let home = std::env::var("HOME").unwrap();
            let output = run_cmd_output("echo ~", RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), home);
        }
    }

    #[test]
    fn test_build_no_shell_command_expands_program_tilde() {
        // A leading `~` in the program position is expanded too, consistent
        // with the argument position and shell semantics.
        #[cfg(unix)]
        {
            let home = std::env::var("HOME").unwrap();
            let cmd = super::build_no_shell_command("~/bin/prog --flag", RunCmdOptions::default())
                .unwrap();
            let program = cmd.get_program().to_string_lossy().into_owned();
            assert!(program.starts_with(&home), "program={program}");
            assert!(program.ends_with("/bin/prog"), "program={program}");
        }
    }

    #[test]
    fn test_run_cmd_output_preserves_double_quoted_lt_literal_without_shell() {
        #[cfg(unix)]
        {
            let output =
                run_cmd_output(r#"printf "%s" "<literal>""#, RunCmdOptions::default()).unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout), "<literal>");
        }
    }

    #[test]
    fn test_run_cmd_output_streaming_collects_chunks() {
        #[cfg(unix)]
        {
            let mut streamed = Vec::new();
            let output = run_cmd_output_streaming_with_timeout(
                "printf 'hello\\nworld'",
                RunCmdOptions::default(),
                Duration::from_secs(2),
                |chunk| streamed.extend_from_slice(chunk),
                || false,
            )
            .unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&streamed), "hello\nworld");
        }
    }

    #[test]
    fn test_pseudo_terminal_makes_standard_streams_tty() {
        #[cfg(unix)]
        {
            let result = super::run_cmd_output_streaming_with_timeout_tracked_pseudo_terminal(
                "test -t 0 && test -t 1 && test -t 2 && printf pty-ready",
                RunCmdOptions::default(),
                Duration::from_secs(2),
                |_| {},
                || false,
                |_| {},
            )
            .expect("pseudo-terminal command should run");

            assert!(result.status.is_some_and(|status| status.success()));
            assert!(
                String::from_utf8_lossy(&result.stdout).contains("pty-ready"),
                "expected PTY output, got: {:?}",
                String::from_utf8_lossy(&result.stdout)
            );
            assert!(result.stderr.is_empty());
        }
    }

    #[test]
    fn test_pty_stall_terminates_interactive_command_and_preserves_partial_output() {
        // Interactive commands (e.g. QR-code login) print output and then wait
        // for human input: they should be judged "stalled" and terminated
        // quickly, preserving the captured partial output (the QR code) instead
        // of hanging silently until the timeout backstop.
        #[cfg(unix)]
        {
            let started = Instant::now();
            let result = super::run_cmd_output_streaming_with_timeout_tracked_inner(
                "printf 'qr-code-block'; sleep 30",
                RunCmdOptions::default(),
                Duration::from_secs(60),
                |_| {},
                || false,
                |_| {},
                true,
                super::PSEUDO_TERMINAL_ENV_POLICY,
                Some((Duration::from_millis(300), Duration::from_millis(600))),
            )
            .expect("PTY command should run");

            let elapsed = started.elapsed();
            assert!(result.stalled, "expected stall kill: {result:?}");
            assert!(!result.timed_out && !result.cancelled);
            assert!(
                String::from_utf8_lossy(&result.stdout).contains("qr-code-block"),
                "partial output must be preserved: {:?}",
                String::from_utf8_lossy(&result.stdout)
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "stall kill must be fast, took {elapsed:?}"
            );
        }
    }

    #[test]
    fn test_pty_stall_terminates_silent_command() {
        // A PTY command with no output at all since startup (e.g. output
        // swallowed by `| tail` pipe buffering): once the silent-start threshold
        // expires it is likewise judged stalled, avoiding an uninformative hang
        // for the full timeout.
        #[cfg(unix)]
        {
            let started = Instant::now();
            let result = super::run_cmd_output_streaming_with_timeout_tracked_inner(
                "sleep 30",
                RunCmdOptions::default(),
                Duration::from_secs(60),
                |_| {},
                || false,
                |_| {},
                true,
                super::PSEUDO_TERMINAL_ENV_POLICY,
                Some((Duration::from_millis(600), Duration::from_millis(300))),
            )
            .expect("PTY command should run");

            assert!(
                result.stalled,
                "silent PTY command must be stall-killed: {result:?}"
            );
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "silent stall kill must be fast"
            );
        }
    }

    #[test]
    fn test_stall_guard_only_applies_to_pty_runs() {
        // The non-PTY path is unaffected by stall detection: the regular
        // timeout remains the backstop (ordinary commands like compiles or
        // downloads may legitimately stay silent for a long time after
        // printing output).
        #[cfg(unix)]
        {
            let result = run_cmd_output_streaming_with_timeout(
                "printf 'output'; sleep 30",
                RunCmdOptions::default(),
                Duration::from_millis(300),
                |_| {},
                || false,
            );

            assert!(matches!(
                result.as_ref().map_err(|err| err.kind()),
                Err(std::io::ErrorKind::TimedOut)
            ));
        }
    }

    #[test]
    fn test_timeout_kills_shell_descendants_without_waiting_for_them() {
        #[cfg(unix)]
        {
            let started = Instant::now();
            let result = run_cmd_output_streaming_with_timeout(
                "sh -c 'sleep 5'",
                RunCmdOptions::default(),
                Duration::from_millis(200),
                |_| {},
                || false,
            );

            assert!(matches!(
                result.as_ref().map_err(|err| err.kind()),
                Err(std::io::ErrorKind::TimedOut)
            ));
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "timeout waited for shell descendant to exit"
            );
        }
    }

    #[test]
    fn test_tracked_timeout_preserves_partial_output() {
        #[cfg(unix)]
        {
            let result = run_cmd_output_streaming_with_timeout_tracked(
                "printf 'before-timeout\\n'; sleep 5",
                RunCmdOptions::default(),
                Duration::from_millis(200),
                |_| {},
                || false,
                |_| {},
            )
            .expect("tracked timeout should return its partial result");

            assert!(result.timed_out);
            assert!(!result.cancelled);
            assert!(String::from_utf8_lossy(&result.stdout).contains("before-timeout"));
        }
    }

    #[test]
    fn test_returns_promptly_when_command_backgrounds_a_long_lived_process() {
        // Regression: the foreground command finishes, but a background spawned
        // process inherited the stdout/stderr pipes. The old implementation
        // joined a reader thread that never exits and deadlocked; the new one
        // should return after the grace period.
        #[cfg(unix)]
        {
            let started = Instant::now();
            let mut streamed = Vec::new();
            let output = run_cmd_output_streaming_with_timeout(
                "sh -c 'sleep 30 & echo ready'",
                RunCmdOptions::default(),
                Duration::from_secs(10),
                |chunk| streamed.extend_from_slice(chunk),
                || false,
            )
            .expect("should return without hanging");

            assert!(output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("ready"),
                "expected foreground output to be captured"
            );
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "hung waiting for backgrounded descendant to close pipes"
            );
        }
    }

    #[test]
    fn test_reports_pgid_when_background_group_survives() {
        // After spawning a long-lived background process the foreground returns
        // immediately: the surviving group's pgid should be reported for the
        // upper layer to register session-level cleanup. The test kills the
        // group itself afterwards to avoid leaks.
        #[cfg(unix)]
        {
            let mut reported: Vec<u32> = Vec::new();
            let output = super::run_cmd_output_streaming_with_timeout_tracked(
                "sh -c 'sleep 30 & echo up'",
                RunCmdOptions::default(),
                Duration::from_secs(10),
                |_| {},
                || false,
                |pgid| reported.push(pgid),
            )
            .expect("should return without hanging");

            assert!(output.status.unwrap().success());
            assert_eq!(reported.len(), 1, "expected exactly one surviving group");
            let pgid = reported[0];
            assert!(pgid > 0);
            // The process group should indeed be alive.
            assert_eq!(unsafe { libc::kill(-(pgid as libc::pid_t), 0) }, 0);
            // Cleanup: kill the whole group.
            unsafe {
                let _ = libc::kill(-(pgid as libc::pid_t), libc::SIGKILL);
            }
        }
    }

    #[test]
    fn test_does_not_report_pgid_for_clean_foreground_command() {
        // After a pure foreground command exits, the group has no members; no
        // pgid should be reported.
        #[cfg(unix)]
        {
            let mut reported: Vec<u32> = Vec::new();
            let output = super::run_cmd_output_streaming_with_timeout_tracked(
                "echo hello",
                RunCmdOptions::default(),
                Duration::from_secs(5),
                |_| {},
                || false,
                |pgid| reported.push(pgid),
            )
            .expect("should succeed");

            assert!(output.status.unwrap().success());
            assert!(
                reported.is_empty(),
                "clean foreground command should not report a surviving group, got {reported:?}"
            );
        }
    }

    #[test]
    fn test_pseudo_terminal_env_only_disables_pagers() {
        fn configured_env<'a>(
            cmd: &'a std::process::Command,
            key: &str,
        ) -> Option<Option<&'a std::ffi::OsStr>> {
            cmd.get_envs()
                .find(|(name, _)| *name == std::ffi::OsStr::new(key))
                .map(|(_, value)| value)
        }

        let mut cmd = std::process::Command::new("env");
        super::apply_command_env_policy(&mut cmd, super::PSEUDO_TERMINAL_ENV_POLICY);

        for key in ["PAGER", "GIT_PAGER"] {
            assert_eq!(
                configured_env(&cmd, key),
                Some(Some(std::ffi::OsStr::new("cat"))),
                "PTY should disable captured-output pager {key}"
            );
        }
        for key in [
            "GIT_EDITOR",
            "GIT_SEQUENCE_EDITOR",
            "GIT_TERMINAL_PROMPT",
            "TERM",
            "NO_COLOR",
            "CLICOLOR",
        ] {
            assert_eq!(
                configured_env(&cmd, key),
                None,
                "PTY should preserve inherited interactive environment {key}"
            );
        }
    }

    #[test]
    fn test_non_interactive_runner_disables_pagers_and_prompts() {
        let output = super::run_cmd_output_with_timeout_non_interactive(
            "env",
            RunCmdOptions::default(),
            Duration::from_secs(5),
        )
        .expect("non-interactive command should succeed");
        let env = String::from_utf8_lossy(&output.stdout);
        for expected in [
            "PAGER=cat",
            "GIT_PAGER=cat",
            "GIT_EDITOR=true",
            "GIT_SEQUENCE_EDITOR=true",
            "GIT_TERMINAL_PROMPT=0",
            "TERM=dumb",
            "NO_COLOR=1",
            "CLICOLOR=0",
        ] {
            assert!(
                env.lines().any(|line| line == expected),
                "missing {expected}"
            );
        }
    }
}
