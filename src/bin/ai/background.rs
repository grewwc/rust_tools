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

use std::{
    ffi::CString,
    fs::File,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    path::{Path, PathBuf},
    process,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicBool, AtomicI32, Ordering},
    },
    time::Duration,
};

use crate::ai::cli::ParsedCli;
use crate::ai::driver;

/// Guidance injected into a turn after it has lost its interactive terminal.
///
/// Keep this English because it is model-visible runtime context.
pub(crate) const BACKGROUND_CONTINUATION_NOTE: &str = "[Background mode] The user moved this session to the background. Continue the current task autonomously until it is actually complete. Do not wait for interactive input; investigate and use available tools to resolve issues, then produce the final result.";

/// The "do not stop midway" directive appended to a newly started `a -bg` task.
const BACKGROUND_DIRECTIVE: &str = BACKGROUND_CONTINUATION_NOTE;

/// The foreground process registers this only after a live `/bg` handoff has
/// succeeded. `entry` removes the file after the normal driver loop exits, matching
/// the cleanup performed by `run_background_child` for a fresh `a -bg` daemon.
static LIVE_BACKGROUND_PID_FILE: LazyLock<Mutex<Option<PathBuf>>> =
    LazyLock::new(|| Mutex::new(None));

static LIVE_BACKGROUND_ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE_STDIO_RELAY_ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE_ATTACH_TARGET_PID: AtomicI32 = AtomicI32::new(0);

/// The helper must give an interactive shell enough time to observe the stopped job,
/// reclaim the foreground process group, and print its prompt before the agent resumes.
const LIVE_BACKGROUND_RESUME_DELAY: Duration = Duration::from_millis(250);

/// Live-output FIFO for a real-time `/bg` handoff (see [`detach_live_session`]).
///
/// After the handoff stdout/stderr are drained by an in-process relay and — by
/// the user's explicit requirement — no output log is written, so the only way
/// to observe a backgrounded session is a live channel. A named pipe is used
/// deliberately: FIFO data never persists on disk. A later `a` invocation on
/// the same session attaches to the pipe (see [`attach_live_session`]) and
/// replays the chunks until the background process exits.
static LIVE_OUTPUT_FIFO: LazyLock<Mutex<Option<LiveOutputFifo>>> =
    LazyLock::new(|| Mutex::new(None));

struct LiveOutputFifo {
    /// Writer fd, cached after the first successful open; dropped and reopened
    /// on demand when the attach reader goes away (EPIPE) — see
    /// [`publish_live_output`].
    fd: Option<OwnedFd>,
    path: PathBuf,
}

/// Session-scoped FIFO inode name (working-directory based, next to the
/// live-handoff `.pid` file, so a same-directory `a` re-entry can find both).
fn live_fifo_path_for_session(session_id: &str) -> PathBuf {
    PathBuf::from(format!("{session_id}.live"))
}

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

    let pid_path = pid_path_for_session(&session_id);
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

/// Move the already-running interactive process into the shell background without
/// replacing it or writing an output log.
///
/// A live turn has in-memory HTTP streams, scheduler state, and Tokio tasks, so it
/// cannot be transferred to the fresh child process used by `a -bg`. Forking that
/// state is unsafe on macOS because Objective-C/CoreFoundation may already be
/// initialized. Instead, a fresh minimal helper is spawned with `posix_spawn`; this
/// process redirects stdin to `/dev/null` and stdout/stderr to a relay pipe, ignores
/// terminal hangups, then stops itself with `SIGTSTP`. The shell observes the stopped
/// job and returns to its prompt; the helper sends `SIGCONT` shortly afterwards,
/// leaving this same process running as a background job with its active turn intact.
///
/// The caller must have released all terminal input ownership before invoking this
/// function. It is deliberately Unix-only, matching `a -bg`.
#[cfg(unix)]
pub(crate) fn detach_live_session(session_id: &str) -> std::io::Result<()> {
    let pid_path = pid_path_for_session(session_id);

    // All realistically fallible setup happens before standard descriptors change.
    // If any of it fails, the user remains in the foreground with a usable terminal.
    spawn_resume_helper()?;
    write_pid_file(&pid_path)?;

    // Best-effort: opening the live-output FIFO before the stop lets a later
    // `a` re-entry watch this session; a failure only disables watching. Do
    // this before stdout/stderr are redirected to the relay so the warning is visible.
    if let Err(err) = open_live_output_fifo(session_id) {
        eprintln!("[background] warning: cannot open live output pipe: {err}");
    }

    eprintln!(
        "[background] session {session_id} will continue in the shell background (no log file)."
    );
    if let Err(err) = redirect_standard_streams_for_live_background() {
        let _ = std::fs::remove_file(&pid_path);
        close_live_output_fifo();
        return Err(err);
    }
    ignore_terminal_hangups();

    {
        let mut registered = LIVE_BACKGROUND_PID_FILE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *registered = Some(pid_path);
    }
    LIVE_BACKGROUND_ACTIVE.store(true, Ordering::Relaxed);

    // `raise(SIGTSTP)` stops every thread in this process. It returns only after the
    // helper has resumed us with SIGCONT, so the caller can immediately enqueue the
    // background-continuation side note without interrupting the active model request.
    if unsafe { libc::raise(libc::SIGTSTP) } != 0 {
        cleanup_live_background_pid_file();
        LIVE_BACKGROUND_ACTIVE.store(false, Ordering::Relaxed);
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn detach_live_session(_session_id: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "live background handoff is unix-only",
    ))
}

/// Internal entry point for the freshly spawned helper. It deliberately obtains the
/// parent PID itself rather than accepting a PID argument, so this internal command
/// cannot be used to signal an arbitrary process.
#[cfg(unix)]
pub(super) fn resume_detached_parent() -> Result<(), Box<dyn std::error::Error>> {
    let parent = unsafe { libc::getppid() };
    if parent <= 1 {
        return Err("live background helper lost its parent before it could resume it".into());
    }
    std::thread::sleep(LIVE_BACKGROUND_RESUME_DELAY);
    if unsafe { libc::kill(parent, libc::SIGCONT) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn resume_detached_parent() -> Result<(), Box<dyn std::error::Error>> {
    Err("live background helper is unix-only".into())
}

/// Delete the current process's live-handoff PID file after the foreground driver
/// eventually finishes. Fresh `a -bg` children use their own local cleanup instead.
pub(super) fn cleanup_live_background_pid_file() {
    LIVE_BACKGROUND_ACTIVE.store(false, Ordering::Relaxed);
    LIVE_STDIO_RELAY_ACTIVE.store(false, Ordering::Relaxed);
    let pid_path = LIVE_BACKGROUND_PID_FILE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(pid_path) = pid_path {
        let _ = std::fs::remove_file(pid_path);
    }
    // Also close the live-output FIFO and remove its inode, so a subsequent
    // `a` re-entry never attaches to a stale pipe after this process exits.
    close_live_output_fifo();
}

/// Create (or reuse) the FIFO inode this process publishes live output into
/// after a real-time `/bg` handoff. The writer fd is deliberately NOT opened
/// here: on macOS an `O_WRONLY|O_NONBLOCK` FIFO open fails with ENXIO while no
/// reader holds the pipe, which `publish_live_output` uses as the exact
/// "nobody is attached" signal.
#[cfg(unix)]
fn open_live_output_fifo(session_id: &str) -> std::io::Result<()> {
    let path = live_fifo_path_for_session(session_id);
    let c_path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "fifo path contains NUL")
    })?;
    if unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) } != 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(err);
        }
    }
    let mut guard = LIVE_OUTPUT_FIFO.lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(LiveOutputFifo { fd: None, path });
    Ok(())
}

#[cfg(not(unix))]
fn open_live_output_fifo(_session_id: &str) -> std::io::Result<()> {
    Ok(())
}

/// Publish one committed assistant-text chunk to the live-output FIFO opened
/// by a real-time `/bg` handoff. Best-effort by contract: never blocks, never
/// fails the caller, and silently drops chunks while nobody is attached.
pub(crate) fn publish_live_output(text: &str) {
    if text.is_empty() {
        return;
    }
    if LIVE_STDIO_RELAY_ACTIVE.load(Ordering::Relaxed) {
        // After live `/bg` handoff stdout/stderr are already mirrored through the
        // relay pipe. Suppress legacy assistant-chunk publishing to avoid
        // duplicating visible assistant text on reattach.
        return;
    }
    publish_live_output_bytes(text.as_bytes());
}

fn publish_live_output_bytes(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let mut guard = LIVE_OUTPUT_FIFO.lock().unwrap_or_else(|p| p.into_inner());
    let Some(fifo) = guard.as_mut() else { return };
    if fifo.fd.is_none() {
        // Open the writer on demand. On macOS an O_WRONLY|O_NONBLOCK FIFO open
        // fails with ENXIO while no reader holds the pipe: exactly the
        // "nobody is attached" signal, so the chunk is dropped, never queued.
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&fifo.path)
        {
            Ok(fd) => fifo.fd = Some(fd.into()),
            Err(_) => return,
        }
    }
    let mut written = 0usize;
    while written < bytes.len() {
        // Rust's std ignores SIGPIPE, so EPIPE (reader gone) is a plain error.
        let n = unsafe {
            libc::write(
                fifo.fd.as_ref().expect("fd set above").as_raw_fd(),
                bytes[written..].as_ptr().cast(),
                bytes.len() - written,
            )
        };
        if n <= 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::EPIPE) {
                // The attach reader closed: drop the cached fd so the next
                // chunk re-probes (and finds ENXIO until a new reader attaches).
                fifo.fd = None;
            }
            break; // EAGAIN (no reader) / EPIPE / other: drop the rest
        }
        written += n as usize;
    }
}

/// PID recorded by a real-time `/bg` handoff, if that process is still alive.
/// Both the `.pid` file and the `.live` FIFO live in the working directory of
/// the original `a` invocation, so a same-directory re-entry can find them.
pub(super) fn live_session_pid(session_id: &str) -> Option<libc::pid_t> {
    let contents = std::fs::read_to_string(pid_path_for_session(session_id)).ok()?;
    let pid: libc::pid_t = contents.trim().parse().ok()?;
    if pid <= 0 || unsafe { libc::kill(pid, 0) } != 0 {
        return None;
    }
    Some(pid)
}

pub(super) fn is_session_live(session_id: &str) -> bool {
    live_session_pid(session_id).is_some_and(|pid| pid != process::id() as libc::pid_t)
}

pub(crate) fn live_background_active() -> bool {
    LIVE_BACKGROUND_ACTIVE.load(Ordering::Relaxed)
}

#[cfg(unix)]
struct LiveAttachSignalGuard {
    pid: libc::pid_t,
}

#[cfg(unix)]
impl LiveAttachSignalGuard {
    fn new(pid: libc::pid_t) -> Self {
        LIVE_ATTACH_TARGET_PID.store(pid, Ordering::Release);
        Self { pid }
    }
}

#[cfg(unix)]
impl Drop for LiveAttachSignalGuard {
    fn drop(&mut self) {
        let _ = LIVE_ATTACH_TARGET_PID.compare_exchange(
            self.pid,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// Forward Ctrl+C pressed in a reattach process to the backgrounded agent
/// process, so its normal signal handler applies the same interrupt semantics
/// it would have used before `/bg`.
#[cfg(unix)]
pub(crate) fn forward_sigint_to_live_attach_target() -> bool {
    let pid = LIVE_ATTACH_TARGET_PID.load(Ordering::Acquire);
    if pid <= 0 {
        return false;
    }
    if unsafe { libc::kill(pid, 0) } != 0 {
        if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            let _ = LIVE_ATTACH_TARGET_PID.compare_exchange(
                pid,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        return false;
    }
    unsafe { libc::kill(pid, libc::SIGINT) == 0 }
}

#[cfg(not(unix))]
pub(crate) fn forward_sigint_to_live_attach_target() -> bool {
    false
}

/// How [`attach_live_session`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttachOutcome {
    /// The backgrounded process finished (or was never live); the caller should
    /// proceed with the normal session flow.
    BackgroundCompleted,
    /// The attach process stopped watching while the backgrounded session is
    /// still running; the caller should exit without opening the session.
    UserDetached,
}

/// Attach to a session running in the shell background after a real-time `/bg`
/// handoff and replay its live output until the background process exits.
///
/// Returns [`AttachOutcome::BackgroundCompleted`] immediately when the session
/// is not live, so callers can use this as an unconditional pre-step. This is a
/// blocking function (short `poll(2)` waits); call it on a dedicated thread
/// from async contexts.
#[cfg(unix)]
pub(super) fn attach_live_session(
    session_id: &str,
    shutdown: &AtomicBool,
) -> std::io::Result<AttachOutcome> {
    use std::os::unix::fs::OpenOptionsExt;

    let Some(pid) = live_session_pid(session_id) else {
        return Ok(AttachOutcome::BackgroundCompleted);
    };
    if pid == process::id() as libc::pid_t {
        return Ok(AttachOutcome::BackgroundCompleted);
    }
    let _signal_guard = LiveAttachSignalGuard::new(pid);
    let fifo_path = live_fifo_path_for_session(session_id);
    if !fifo_path.exists() {
        // Live process but no FIFO (e.g. a plain `a -bg` daemon or a failed FIFO setup):
        // do not open the same session concurrently. There is no live stream to show,
        // so leave the existing process running and return to the shell.
        eprintln!(
            "[background] session {session_id} is already running in the background (pid {pid}); live output is unavailable, not opening a second copy."
        );
        return Ok(AttachOutcome::UserDetached);
    }
    let fifo = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&fifo_path)
    {
        Ok(fifo) => fifo,
        Err(err) => {
            if live_session_pid(session_id).is_some() {
                eprintln!(
                    "[background] session {session_id} is already running in the background (pid {pid}); cannot open live output pipe: {err}"
                );
                return Ok(AttachOutcome::UserDetached);
            }
            let _ = std::fs::remove_file(pid_path_for_session(session_id));
            return Ok(AttachOutcome::BackgroundCompleted);
        }
    };
    eprintln!(
        "[background] session {session_id} is running in the shell background (pid {pid}); live output follows, Ctrl+C to interrupt."
    );

    let mut reader = std::io::BufReader::new(fifo);
    let mut buf = vec![0u8; 16384];
    loop {
        if shutdown.load(Ordering::Relaxed) {
            eprintln!("[background] detached; session {session_id} is still running (pid {pid}).");
            return Ok(AttachOutcome::UserDetached);
        }
        // Short poll timeout keeps the loop responsive to Ctrl+C and to a
        // background process that exited without a clean FIFO close.
        let mut pfd = libc::pollfd {
            fd: reader.get_ref().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, 200) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if rc == 0 {
            // Timeout with no data: if the background process is gone, stop
            // watching (its FIFO writes already ended).
            if live_session_pid(session_id).is_none() {
                let _ = std::fs::remove_file(pid_path_for_session(session_id));
                break;
            }
            continue;
        }
        match reader.read(&mut buf) {
            Ok(0) => {
                // EOF: no writer currently holds the FIFO. While the background
                // process is still alive this is transient (its writer opens on
                // demand); only treat it as completion once the process is gone.
                if live_session_pid(session_id).is_none() {
                    let _ = std::fs::remove_file(pid_path_for_session(session_id));
                    break;
                }
                // Transient EOF: an O_NONBLOCK read end reports EOF (POLLHUP)
                // immediately and `poll` keeps returning instantly, so without
                // this sleep the loop would spin at 100% CPU until the
                // background process publishes its next chunk or exits.
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(n) => {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                if out.write_all(&buf[..n]).is_err() || out.flush().is_err() {
                    // Terminal gone; stop watching rather than failing the caller.
                    return Ok(AttachOutcome::UserDetached);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
    }
    eprintln!("[background] session {session_id} finished running in the background.");
    Ok(AttachOutcome::BackgroundCompleted)
}

#[cfg(not(unix))]
pub(super) fn attach_live_session(
    _session_id: &str,
    _shutdown: &AtomicBool,
) -> std::io::Result<AttachOutcome> {
    Ok(AttachOutcome::BackgroundCompleted)
}

/// Close the live-output FIFO (if open) and remove its inode. Called by
/// [`cleanup_live_background_pid_file`] when the backgrounded process exits.
fn close_live_output_fifo() {
    let fifo = LIVE_OUTPUT_FIFO
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(fifo) = fifo {
        let _ = std::fs::remove_file(&fifo.path);
    }
}

#[cfg(unix)]
fn spawn_resume_helper() -> std::io::Result<()> {
    let exe = std::env::current_exe()?;
    let dev_null = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;
    let mut command = std::process::Command::new(exe);
    command
        .arg("--resume-detached-parent")
        .stdin(dev_null.try_clone()?)
        .stdout(dev_null.try_clone()?)
        .stderr(dev_null);
    let child = crate::fork_guard::spawn(&mut command)?;
    reap_live_background_helper(child);
    Ok(())
}

/// Wait on the live-handoff helper from a detached thread so it cannot linger
/// as a zombie for the rest of this process's life.
///
/// The helper exits ~250 ms after sending SIGCONT (see
/// [`resume_detached_parent`]), while the parent keeps running as a background
/// job. Dropping its `Child` without waiting would leave one zombie per `/bg`
/// handoff until this process exits, so a reaper thread waits on it as soon as
/// it terminates. The child is handed to the thread over a channel: if thread
/// creation fails, the `Child` is dropped unwaited right here (reverting to the
/// old single-zombie behavior) rather than waiting inline — blocking would
/// swallow the helper's SIGCONT before the parent stops itself with SIGTSTP in
/// `detach_live_session`, leaving this process stopped forever.
#[cfg(unix)]
fn reap_live_background_helper(child: std::process::Child) {
    let (tx, rx) = std::sync::mpsc::channel::<std::process::Child>();
    let reaper = std::thread::Builder::new()
        .name("live-bg-helper-reaper".to_string())
        .spawn(move || {
            if let Ok(mut child) = rx.recv() {
                let _ = child.wait();
            }
        });
    if reaper.is_ok() {
        let _ = tx.send(child);
    }
}

#[cfg(unix)]
fn redirect_standard_streams_for_live_background() -> std::io::Result<()> {
    use std::os::fd::AsRawFd;

    let dev_null = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;
    if unsafe { libc::dup2(dev_null.as_raw_fd(), libc::STDIN_FILENO) } == -1 {
        return Err(std::io::Error::last_os_error());
    }

    let mut pipe_fds = [-1; 2];
    if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    let read_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[0]) };
    let write_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[1]) };
    spawn_live_stdio_relay(read_fd)?;

    if unsafe { libc::dup2(write_fd.as_raw_fd(), libc::STDOUT_FILENO) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::dup2(write_fd.as_raw_fd(), libc::STDERR_FILENO) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    LIVE_STDIO_RELAY_ACTIVE.store(true, Ordering::Relaxed);
    Ok(())
}

#[cfg(unix)]
fn spawn_live_stdio_relay(read_fd: OwnedFd) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("live-bg-stdio-relay".to_string())
        .spawn(move || {
            let mut input = File::from(read_fd);
            let mut buf = [0_u8; 8192];
            loop {
                match input.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => publish_live_output_bytes(&buf[..n]),
                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            LIVE_STDIO_RELAY_ACTIVE.store(false, Ordering::Relaxed);
        })
        .map(|_| ())
}

#[cfg(unix)]
fn ignore_terminal_hangups() {
    // After the shell has put this job in the background, closing the original
    // terminal can send SIGHUP. Standard streams are already `/dev/null`, so
    // ignoring the hangup keeps the live turn alive without further terminal I/O.
    unsafe {
        let _ = libc::signal(libc::SIGHUP, libc::SIG_IGN);
        let _ = libc::signal(libc::SIGTTIN, libc::SIG_IGN);
        let _ = libc::signal(libc::SIGTTOU, libc::SIG_IGN);
    }
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
    let pid_path = pid_path_for_session(session_id);

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

fn pid_path_for_session(session_id: &str) -> PathBuf {
    PathBuf::from(format!("{session_id}.pid"))
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
fn spawn_daemon_child(
    _cli: &ParsedCli,
    _session_id: &str,
    _log_path: &Path,
) -> std::io::Result<()> {
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
        let args_str: Vec<String> = args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect();
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
        let args_str: Vec<String> = args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect();
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
            .chain(
                daemon_args
                    .iter()
                    .skip(2)
                    .map(|s| s.to_string_lossy().to_string()),
            )
            .collect();
        let reparsed = parse_cli_args(child_argv.into_iter());
        assert_eq!(reparsed.args, vec!["do something important".to_string()]);
        assert_eq!(reparsed.model.as_deref(), Some("my-model"));
        assert_eq!(
            reparsed.reasoning_effort_override,
            Some(Some(crate::ai::provider::ReasoningEffort::High))
        );
    }

    #[cfg(unix)]
    #[test]
    fn live_background_helper_is_reaped_promptly() {
        // Regression: the live-handoff helper used to be dropped without
        // waiting, leaving a zombie child for the rest of the process
        // lifetime. The reaper thread must reap it as soon as it exits.
        let child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .expect("spawn helper process");
        let pid = child.id() as i32;
        reap_live_background_helper(child);

        // The child must no longer be a child of this process shortly after it
        // exits: `waitpid(pid, WNOHANG)` returns 0 while the child still exists
        // (running or zombie) and -1/ECHILD once the reaper thread reaped it.
        // If this test's own waitpid wins the race instead, the child is
        // reaped all the same and the reaper thread's `wait` simply fails with
        // ECHILD (ignored via `let _`).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let rc = unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) };
            if rc == -1 {
                let errno = std::io::Error::last_os_error().raw_os_error();
                if errno == Some(libc::EINTR) {
                    continue; // spurious wakeup; retry the poll
                }
                assert_eq!(
                    errno,
                    Some(libc::ECHILD),
                    "helper pid {pid} should be reaped by now"
                );
                break;
            }
            if rc == pid {
                break; // this test's waitpid reaped it; also fine
            }
            assert!(
                std::time::Instant::now() < deadline,
                "helper pid {pid} was never reaped (still running or zombie)"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    #[test]
    fn attach_live_session_refuses_live_pid_without_fifo() -> std::io::Result<()> {
        // A live pid without a live-output FIFO must not be treated as completed:
        // opening the session while that process is still alive would race two
        // processes against the same SQLite history.
        let _lock = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("a-bg-no-fifo-{}", process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = (|| -> std::io::Result<()> {
            let session = "no-fifo-session";
            let mut child = std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg("sleep 5")
                .spawn()?;
            std::fs::write(pid_path_for_session(session), child.id().to_string())?;
            let shutdown = AtomicBool::new(false);
            let outcome = attach_live_session(session, &shutdown);
            let _ = child.kill();
            let _ = child.wait();
            let outcome = outcome?;
            assert_eq!(outcome, AttachOutcome::UserDetached);
            assert!(
                pid_path_for_session(session).exists(),
                "live pid marker must stay so a later attach still refuses concurrent open"
            );
            Ok(())
        })();

        std::env::set_current_dir(&prev).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[cfg(unix)]
    #[test]
    fn attach_live_session_ignores_own_pid_file() -> std::io::Result<()> {
        // A fresh `a -bg` daemon writes its own pid file before entering
        // run_with_cli. Startup reattach must not mistake that marker for
        // another live process and exit before executing the daemon's task.
        let _lock = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("a-bg-self-pid-{}", process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = (|| -> std::io::Result<()> {
            let session = "self-pid-session";
            write_pid_file(&pid_path_for_session(session))?;
            let shutdown = AtomicBool::new(false);
            let outcome = attach_live_session(session, &shutdown)?;
            assert_eq!(outcome, AttachOutcome::BackgroundCompleted);
            Ok(())
        })();

        std::env::set_current_dir(&prev).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[cfg(unix)]
    #[test]
    fn live_attach_sigint_forwarding_signals_target_process() -> std::io::Result<()> {
        let mut child = std::process::Command::new("/bin/sleep").arg("30").spawn()?;
        let pid = child.id() as libc::pid_t;
        {
            let _guard = LiveAttachSignalGuard::new(pid);
            assert!(
                forward_sigint_to_live_attach_target(),
                "forwarding should signal the registered live attach target"
            );

            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            loop {
                if child.try_wait()?.is_some() {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("target process did not exit after forwarded SIGINT");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        assert!(!forward_sigint_to_live_attach_target());
        Ok(())
    }

    /// Round-trip the live-output FIFO: the backgrounded side publishes chunks
    /// and a detached attach reader replays them until the background exits.
    #[test]
    #[cfg(unix)]
    fn live_output_fifo_roundtrip() -> std::io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;
        use std::sync::atomic::AtomicBool;

        // Both the `.pid` file and the FIFO are working-directory based, so the
        // test runs in a throwaway directory guarded by the global env lock.
        let _lock = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("a-bg-fifo-{}", process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = (|| -> std::io::Result<()> {
            let session = "fifo-roundtrip-session";
            // This test process plays the backgrounded agent: register its own
            // pid so attach_live_session sees a live session.
            write_pid_file(&pid_path_for_session(session))?;
            open_live_output_fifo(session)?;

            // A detached reader attaches; on macOS an O_RDONLY|O_NONBLOCK FIFO
            // open succeeds even without a writer.
            let mut reader = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(live_fifo_path_for_session(session))?;
            // The writer opens on demand while a reader holds the pipe.
            publish_live_output("hello background");
            publish_live_output(" world");
            let mut got = Vec::new();
            let mut buf = [0u8; 64];
            let deadline = std::time::Instant::now() + Duration::from_millis(2000);
            while got.len() < "hello background world".len() && std::time::Instant::now() < deadline
            {
                let mut pfd = libc::pollfd {
                    fd: reader.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                if unsafe { libc::poll(&mut pfd, 1, 50) } > 0 {
                    let n = reader.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                }
            }
            assert_eq!(got, b"hello background world");

            // "Background process" exits: close the writer and drop the pid.
            close_live_output_fifo();
            let _ = std::fs::remove_file(pid_path_for_session(session));

            // A re-entry attach must now complete immediately (not live).
            let shutdown = AtomicBool::new(false);
            let outcome = attach_live_session(session, &shutdown)?;
            assert_eq!(outcome, AttachOutcome::BackgroundCompleted);
            Ok(())
        })();

        std::env::set_current_dir(&prev).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[test]
    #[cfg(unix)]
    fn live_stdio_relay_forwards_terminal_bytes_to_fifo() -> std::io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;

        let _lock = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("a-bg-stdio-relay-{}", process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = (|| -> std::io::Result<()> {
            let session = "stdio-relay-session";
            open_live_output_fifo(session)?;
            let mut reader = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(live_fifo_path_for_session(session))?;

            let mut pipe_fds = [-1; 2];
            if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } == -1 {
                return Err(std::io::Error::last_os_error());
            }
            let read_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[0]) };
            let write_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[1]) };
            spawn_live_stdio_relay(read_fd)?;

            let mut writer = File::from(write_fd);
            writer.write_all(b"running execute_command\n")?;
            writer.flush()?;
            drop(writer);

            let mut got = Vec::new();
            let mut buf = [0u8; 64];
            let deadline = std::time::Instant::now() + Duration::from_millis(2000);
            while got.len() < "running execute_command\n".len()
                && std::time::Instant::now() < deadline
            {
                let mut pfd = libc::pollfd {
                    fd: reader.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                if unsafe { libc::poll(&mut pfd, 1, 50) } > 0 {
                    let n = reader.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                }
            }
            assert_eq!(got, b"running execute_command\n");
            close_live_output_fifo();
            Ok(())
        })();

        std::env::set_current_dir(&prev).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        result
    }
}
