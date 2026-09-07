//! Terminal frontend for persistent PTY sessions.
//!
//! A bare interactive `a` (eligible CLI, real TTY, no live terminal binding)
//! starts a `--terminal-host` process that owns a PTY worker, then attaches to
//! it. `/bg` from the worker's side note detaches only this frontend; the
//! worker keeps running, and re-attaching replays sanitized output so the
//! session looks uninterrupted. The real terminal answers the worker's escape
//! queries through this relay (query -> OUTPUT -> real terminal -> reply on
//! stdin -> INPUT -> host -> PTY), exactly like tmux/ssh.
//!
//! Every exit path restores the original termios and resets terminal modes;
//! the real terminal is never left in raw mode. Ctrl+C is a raw byte forwarded
//! to the worker, never a client signal.

use super::{
    registry,
    wire::{self, Decoder, Queue, Reply, Request, Window},
};
use crate::ai::{cli::ParsedCli, history::physical_terminal_key};
use std::{
    io,
    os::unix::io::{AsRawFd, RawFd},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

type EntryResult = Result<Option<i32>, Box<dyn std::error::Error>>;

/// Restore worker-owned modes without moving subsequent output over history.
/// DECSTBM (`CSI r`) homes the cursor even when the margins were already full
/// screen. Save/restore it, then reset attributes and show the cursor. The worker
/// UI uses a fixed primary-screen viewport, not an alternate screen: do not emit
/// DECRST 1049, which can restore a stale saved cursor even on the primary screen.
const RESET: &[u8] = b"\x1b7\x1b[r\x1b8\x1b[?2004l\x1b[0m\x1b[?25h";
/// How long to wait for a freshly spawned host to bind its bootstrap socket.
const SPAWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll timeout; doubles as the resize-check cadence.
const POLL_MS: i32 = 150;

/// Set by SIGHUP/SIGTERM/SIGINT so the loop exits and the terminal is restored
/// even if the event loop never sees another event.
static SIGNALED: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
pub(super) struct AttachEnd {
    pub(super) code: i32,
    /// True when the host asked us to leave: the worker keeps running.
    pub(super) detached: bool,
}

pub(super) fn maybe_run(cli: &ParsedCli, original_args: &[String]) -> EntryResult {
    let root = registry::root()?;
    maybe_run_impl(cli, original_args, &root, libc::STDIN_FILENO, libc::STDOUT_FILENO)
}

/// Real client entry (fd injection exists only for offline tests).
pub(super) fn maybe_run_impl(
    cli: &ParsedCli,
    original_args: &[String],
    root: &Path,
    stdin_fd: RawFd,
    stdout_fd: RawFd,
) -> EntryResult {
    // Only a real interactive terminal frontend may attach or start a host.
    if unsafe { libc::isatty(stdin_fd) != 1 || libc::isatty(stdout_fd) != 1 } {
        return Ok(None);
    }
    if !eligible(cli) {
        return Ok(None);
    }
    let terminal = physical_terminal_key().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let mut stale_cleared = false;
    let stream = loop {
        let (target, spawned) = resolve_target(cli, original_args, root, &terminal)?;
        let deadline = spawned.then(|| Instant::now() + SPAWN_TIMEOUT);
        let Some(stream) = connect_with_retry(&target, deadline)? else {
            if spawned {
                return Err("persistent terminal host did not start".into());
            }
            if cli.session.is_some() {
                // `-ss <id>` with no live host: normal startup handles it.
                return Ok(None);
            }
            // The terminal binding is stale (its host died): clear it and retry
            // once with a fresh host so this terminal stays persistent.
            if !stale_cleared {
                if let Some(binding) = registry::binding(root, &terminal)? {
                    let _ = registry::clear_terminal(root, &terminal, &binding.socket);
                }
                stale_cleared = true;
                continue;
            }
            return Ok(None);
        };
        break stream;
    };
    let end = attach(stream, terminal, query_window(stdin_fd), stdin_fd, stdout_fd)?;
    if end.detached {
        eprintln!(
            "[bg] session still running in the background: run `a` in this terminal \
             (or `a -ss <session-id>` in any terminal) to re-attach"
        );
    }
    Ok(Some(end.code))
}

/// Where to attach: an explicit `-ss` session socket, the terminal binding, or
/// (no live target) a freshly spawned host for this terminal.
fn resolve_target(
    cli: &ParsedCli,
    original_args: &[String],
    root: &Path,
    terminal: &str,
) -> io::Result<(PathBuf, bool)> {
    if let Some(session) = cli.session.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return Ok((root.join(registry::session_name(session)?), false));
    }
    if let Some(binding) = registry::binding(root, terminal)? {
        return Ok((root.join(binding.socket), false));
    }
    let (name, _token) = spawn_host(original_args)?;
    Ok((root.join(registry::bootstrap_name(&name)?), true))
}

/// `registry::connect` with a bounded retry window. `None` means "no live host"
/// (stale or not yet started); the caller decides whether to retry.
fn connect_with_retry(target: &Path, deadline: Option<Instant>) -> io::Result<Option<UnixStream>> {
    loop {
        match registry::connect(target) {
            Ok(Some(stream)) => return Ok(Some(stream)),
            // The host holds the lock while binding its listener; the socket
            // file may not exist yet for a few milliseconds after spawn.
            Ok(None) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            // The host chmods the freshly bound socket to 0600 right after
            // bind; a client polling in that window sees 0755. Retry within
            // the deadline instead of treating a racing fresh socket as unsafe.
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {}
            Err(error) => return Err(error),
        }
        match deadline {
            Some(deadline) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            _ => return Ok(None),
        }
    }
}

/// Fresh `a --terminal-host <uuid> <token> <original argv>` with stdio null.
/// The host calls setsid itself, so it survives this client exiting (and has
/// no controlling terminal to receive SIGHUP from). `original_args` includes
/// argv[0], which the host skips when spawning the worker.
fn spawn_host(original_args: &[String]) -> io::Result<(String, String)> {
    let name = uuid::Uuid::new_v4().to_string();
    let token = uuid::Uuid::new_v4().to_string();
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("--terminal-host")
        .arg(&name)
        .arg(&token)
        .args(original_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn()?;
    Ok((name, token))
}

/// Only a bare interactive `a` (or `a -ss <id>`) is intercepted. Everything
/// else — background mode, one-shot tasks, utility/list/help modes, explicit
/// resume/new-session, model/skill overrides, the worker itself — keeps the
/// normal startup path.
pub(super) fn eligible(cli: &ParsedCli) -> bool {
    if super::is_worker() {
        return false;
    }
    if cli.background || cli.resume || cli.new_session || cli.clear {
        return false;
    }
    if cli.help
        || cli.version
        || cli.list_tools
        || cli.list_mcp_tools
        || cli.list_skills
        || cli.list_agents
        || cli.generate_completions
        || cli.stop_session.is_some()
    {
        return false;
    }
    if cli.note_search
        || cli.note_flag
        || cli.note.is_some()
        || cli.note_delete.is_some()
        || cli.note_edit.is_some()
        || cli.consolidate_knowledge
    {
        return false;
    }
    if !cli.args.is_empty() || !cli.files.is_empty() {
        return false;
    }
    if cli.model.is_some()
        || cli.agent.is_some()
        || cli.interactive
        || cli.no_skills
        || cli.reasoning_effort_override.is_some()
        || !cli.mcp_config.is_empty()
    {
        return false;
    }
    true
}

/// The attach loop. `stdin_fd`/`stdout_fd` are the real terminal (0/1 in the
/// client; a fresh PTY pair in tests). Returns when the host sends DETACHED
/// (worker keeps running), EXIT (worker exit code), the connection dies, or a
/// signal arrives — the TerminalGuard restores termios on every path.
pub(super) fn attach(
    mut stream: UnixStream,
    terminal: String,
    window: Window,
    stdin_fd: RawFd,
    stdout_fd: RawFd,
) -> io::Result<AttachEnd> {
    stream.set_nonblocking(true)?;
    SIGNALED.store(false, Ordering::Relaxed);
    let _signals = Signals::install(stdin_fd);
    let _terminal = TerminalGuard::new(stdin_fd, stdout_fd)?;
    // A previous client may have died without cleanup; reset modes before the
    // worker's output, preserving the shell's current output position. Install
    // the guard first so even a failed reset write restores the terminal.
    write_fd(stdout_fd, RESET)?;
    let mut queue = Queue::default();
    queue.json(wire::REQUEST, &Request::Attach { terminal, window })?;
    let mut decoder = Decoder::default();
    let mut last_window = window;
    loop {
        // Bounded input: a slow host must not make the frontend grow without
        // limit. Dropped input is preferable to a wedged session.
        if !queue.is_empty() && queue.flush(&mut stream).is_err() {
            queue = Queue::default();
        }
        let mut descriptors = [
            pollfd(stdin_fd, libc::POLLIN),
            pollfd(
                stream.as_raw_fd(),
                libc::POLLIN | if queue.is_empty() { 0 } else { libc::POLLOUT },
            ),
        ];
        let result = unsafe { libc::poll(descriptors.as_mut_ptr(), descriptors.len() as _, POLL_MS) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if SIGNALED.load(Ordering::Relaxed) {
            return Ok(AttachEnd { code: 0, detached: false });
        }
        if descriptors[0].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            let mut bytes = [0u8; 8192];
            loop {
                let count = unsafe { libc::read(stdin_fd, bytes.as_mut_ptr() as *mut _, bytes.len()) };
                if count > 0 {
                    if queue.framed(wire::INPUT, &bytes[..count as usize]).is_err() {
                        break; // queue full: drop new input, keep the session
                    }
                    if count as usize == bytes.len() {
                        continue;
                    }
                    break;
                }
                if count == 0 {
                    return Ok(AttachEnd { code: 0, detached: false }); // terminal closed
                }
                match io::Error::last_os_error().kind() {
                    io::ErrorKind::Interrupted => continue,
                    io::ErrorKind::WouldBlock => break,
                    _ => return Ok(AttachEnd { code: 0, detached: false }),
                }
            }
        }
        if descriptors[1].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            let mut bytes = [0u8; 8192];
            loop {
                let count = unsafe { libc::read(stream.as_raw_fd(), bytes.as_mut_ptr() as *mut _, bytes.len()) };
                if count > 0 {
                    decoder.push(&bytes[..count as usize])?;
                    while let Some((kind, payload)) = decoder.next()? {
                        match kind {
                            wire::REPLY => {
                                let reply: Reply = serde_json::from_slice(&payload)?;
                                if let Some(error) = reply.error {
                                    return Err(io::Error::other(error));
                                }
                            }
                            wire::OUTPUT => write_fd(stdout_fd, &payload)?,
                            wire::DETACHED => {
                                return Ok(AttachEnd { code: 0, detached: true });
                            }
                            wire::EXIT => {
                                let code = i32::from_be_bytes(payload.try_into().map_err(|_| {
                                    io::Error::other("invalid PTY exit frame")
                                })?);
                                return Ok(AttachEnd { code, detached: false });
                            }
                            // INPUT/RESIZE are client->host only.
                            _ => {}
                        }
                    }
                    if count as usize == bytes.len() {
                        continue;
                    }
                    break;
                }
                if count == 0 {
                    return Ok(AttachEnd { code: 1, detached: false }); // host died
                }
                match io::Error::last_os_error().kind() {
                    io::ErrorKind::Interrupted => continue,
                    io::ErrorKind::WouldBlock => break,
                    _ => return Ok(AttachEnd { code: 1, detached: false }),
                }
            }
        }
        let window = query_window(stdin_fd);
        if window.rows != last_window.rows || window.cols != last_window.cols {
            last_window = window;
            let _ = queue.json(wire::RESIZE, &window);
        }
    }
}

struct TerminalGuard {
    fd: RawFd,
    output_fd: RawFd,
    original: libc::termios,
}

impl TerminalGuard {
    fn new(fd: RawFd, output_fd: RawFd) -> io::Result<Self> {
        let mut original = std::mem::MaybeUninit::<libc::termios>::uninit();
        if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let original = unsafe { original.assume_init() };
        let mut raw = original;
        unsafe { libc::cfmakeraw(&mut raw) };
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd, output_fd, original })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // Reset the same output terminal used by attach, on both normal and
        // error returns. Restore termios even if the terminal write fails.
        let _ = write_fd(self.output_fd, RESET);
        if unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) } != 0 {
            eprintln!("[debug] termios restore failed: {}", io::Error::last_os_error());
        }
    }
}

struct Signals;

impl Signals {
    /// Installs handlers only for the real client (fd 0); offline tests use
    /// their own PTY and must not touch the test process's signal table.
    fn install(stdin_fd: RawFd) -> Self {
        if stdin_fd == libc::STDIN_FILENO {
            extern "C" fn handle(_: libc::c_int) {
                SIGNALED.store(true, Ordering::Relaxed);
            }
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = handle as *const () as usize;
            unsafe {
                libc::sigemptyset(&mut action.sa_mask);
                for signal in [libc::SIGHUP, libc::SIGTERM, libc::SIGINT] {
                    libc::sigaction(signal, &action, std::ptr::null_mut());
                }
                libc::signal(libc::SIGPIPE, libc::SIG_IGN);
            }
        }
        Signals
    }
}

fn query_window(fd: RawFd) -> Window {
    let mut size = libc::winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
    if unsafe { libc::ioctl(fd, libc::TIOCGWINSZ as _, &mut size) } == 0 && size.ws_row > 0 && size.ws_col > 0
    {
        Window { rows: size.ws_row, cols: size.ws_col }
    } else {
        Window { rows: 24, cols: 80 }
    }
}

fn write_fd(fd: RawFd, bytes: &[u8]) -> io::Result<()> {
    let mut offset = 0;
    while offset < bytes.len() {
        let count = unsafe { libc::write(fd, bytes[offset..].as_ptr() as *const _, bytes.len() - offset) };
        if count > 0 {
            offset += count as usize;
            continue;
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error);
    }
    Ok(())
}

fn pollfd(fd: RawFd, events: i16) -> libc::pollfd {
    libc::pollfd { fd, events, revents: 0 }
}
