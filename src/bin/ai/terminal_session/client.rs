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
/// Bounded display backlog: a stalled SSH link must not freeze the client.
/// Output beyond this cap is dropped (with a notice) instead of blocking;
/// the host's 256 KB replay buffer still lets a re-attach recover the tail.
pub(super) const MAX_DISPLAY_BACKLOG: usize = 4 * 1024 * 1024;
/// After EXIT/DETACHED, keep draining the backlog for this long so output
/// produced while the terminal was stalled is still shown once it recovers.
const EXIT_FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Set by SIGHUP/SIGTERM/SIGINT so the loop exits and the terminal is restored
/// even if the event loop never sees another event.
static SIGNALED: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
pub(super) struct AttachEnd {
    pub(super) code: i32,
    /// True when the host asked us to leave: the worker keeps running.
    pub(super) detached: bool,
    /// Session id from the attach reply; used for re-attach hints after the
    /// connection dies unexpectedly.
    pub(super) session: Option<String>,
    /// True only when the connection died unexpectedly (socket EOF or read
    /// error); never set for a normal EXIT/DETACHED. The worker may still be
    /// running, so a re-attach hint is only meaningful in this case.
    pub(super) connection_lost: bool,
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
        notice(
            "[bg] session still running in the background: run `a` in this terminal \
             (or `a -ss <session-id>` in any terminal) to re-attach\n",
        );
    } else if let Some(hint) = reattach_hint(&end) {
        notice(&hint);
    }
    Ok(Some(end.code))
}

/// Hint after a lost connection. Only an unexpected disconnect (socket EOF or
/// read error) can leave a live worker; a normal worker exit with code 1 must
/// not suggest re-attaching to a session that is already gone.
pub(super) fn reattach_hint(end: &AttachEnd) -> Option<String> {
    if end.connection_lost {
        if let Some(session) = &end.session {
            return Some(format!(
                "[session connection closed; if it is still running, re-attach with `a -ss {session}`]\n"
            ));
        }
    }
    None
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
    // Best-effort: stdout is O_NONBLOCK now, and a stalled terminal must not
    // abort the attach.
    let _ = write_fd(stdout_fd, RESET);
    let mut queue = Queue::default();
    queue.json(wire::REQUEST, &Request::Attach { terminal, window })?;
    let mut decoder = Decoder::default();
    let mut last_window = window;
    let mut backlog = DisplayBacklog::default();
    let mut session: Option<String> = None;
    // Set once EXIT/DETACHED arrives. The loop keeps draining the backlog for
    // a bounded window, so output produced while the terminal was stalled is
    // still shown once the terminal recovers, then attach returns.
    let mut ending: Option<AttachEnd> = None;
    let mut ending_since = Instant::now();
    // Set when the host socket hits EOF while `ending` is already set (the
    // host closes the connection right after EXIT/DETACHED). The fd is then
    // excluded from poll (`fd = -1`) so the lingering POLLHUP cannot turn the
    // bounded flush window into a busy loop.
    let mut socket_done = false;
    loop {
        if let Some(end) = ending.take() {
            if backlog.is_empty() || ending_since.elapsed() >= EXIT_FLUSH_TIMEOUT {
                // Timeout with a backlog that never drained: `pump` never
                // reached the point where it emits the drop notice, so queue
                // it behind the remaining data and pump once. A terminal that
                // recovers right now still sees it; one that stays stalled
                // loses it with the process, as before.
                if !backlog.is_empty() {
                    backlog.queue_notice();
                    let _ = backlog.pump(stdout_fd);
                }
                return Ok(end);
            }
            ending = Some(end);
        }
        // Bounded input: a slow host must not make the frontend grow without
        // limit. Dropped input is preferable to a wedged session. Once the
        // socket is gone (EOF after EXIT/DETACHED) there is nothing to flush
        // to; skip rather than hit the dead fd every poll.
        if !socket_done && !queue.is_empty() && queue.flush(&mut stream).is_err() {
            queue = Queue::default();
        }
        let mut descriptors = [
            // During the ending flush window input is meaningless (the worker
            // is gone or detached); drop stdin from poll instead of queueing
            // keystrokes that would be thrown away by the dead socket.
            pollfd(if ending.is_some() { -1 } else { stdin_fd }, libc::POLLIN),
            pollfd(
                if socket_done { -1 } else { stream.as_raw_fd() },
                if socket_done {
                    0
                } else {
                    libc::POLLIN | if queue.is_empty() { 0 } else { libc::POLLOUT }
                },
            ),
            pollfd(stdout_fd, if backlog.is_empty() { 0 } else { libc::POLLOUT }),
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
            return Ok(AttachEnd { code: 0, detached: false, session, connection_lost: false });
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
                    return Ok(AttachEnd { code: 0, detached: false, session, connection_lost: false }); // terminal closed
                }
                match io::Error::last_os_error().kind() {
                    io::ErrorKind::Interrupted => continue,
                    io::ErrorKind::WouldBlock => break,
                    _ => return Ok(AttachEnd { code: 0, detached: false, session, connection_lost: false }),
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
                                session = reply.session;
                            }
                            wire::OUTPUT => backlog.push_frame(&payload),
                            wire::DETACHED => {
                                ending = Some(AttachEnd { code: 0, detached: true, session: session.clone(), connection_lost: false });
                                ending_since = Instant::now();
                            }
                            wire::EXIT => {
                                let code = i32::from_be_bytes(payload.try_into().map_err(|_| {
                                    io::Error::other("invalid PTY exit frame")
                                })?);
                                ending = Some(AttachEnd { code, detached: false, session: session.clone(), connection_lost: false });
                                ending_since = Instant::now();
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
                    // The host closes the socket right after EXIT/DETACHED; the
                    // ending flush window must not be cut short by that EOF.
                    if ending.is_some() {
                        socket_done = true;
                        break;
                    }
                    // Host died without EXIT/DETACHED: a lost connection, but
                    // keep the same bounded flush window as the EXIT/DETACHED
                    // path so tail output buffered before the death is still
                    // shown on a healthy terminal (empty backlog returns now).
                    let end = AttachEnd { code: 1, detached: false, session: session.clone(), connection_lost: true };
                    if backlog.is_empty() {
                        return Ok(end);
                    }
                    ending = Some(end);
                    ending_since = Instant::now();
                    socket_done = true;
                    break;
                }
                match io::Error::last_os_error().kind() {
                    io::ErrorKind::Interrupted => continue,
                    io::ErrorKind::WouldBlock => break,
                    _ => {
                        if ending.is_some() {
                            socket_done = true;
                            break;
                        }
                        // Socket read error without EXIT/DETACHED: same lost
                        // connection handling as EOF above, including the
                        // bounded flush of already-buffered tail output.
                        let end = AttachEnd { code: 1, detached: false, session: session.clone(), connection_lost: true };
                        if backlog.is_empty() {
                            return Ok(end);
                        }
                        ending = Some(end);
                        ending_since = Instant::now();
                        socket_done = true;
                        break;
                    }
                }
            }
        }
        if descriptors[2].revents & (libc::POLLOUT | libc::POLLHUP | libc::POLLERR) != 0 {
            // Terminal writable (or gone): drain the display backlog. A real
            // write error here means the terminal itself disappeared, so
            // continuing to attach is pointless.
            if let Err(error) = backlog.pump(stdout_fd) {
                return Err(error);
            }
        }
        let window = query_window(stdin_fd);
        if window.rows != last_window.rows || window.cols != last_window.cols {
            last_window = window;
            let _ = queue.json(wire::RESIZE, &window);
        }
    }
}

/// Bounded, non-blocking display output for the real terminal. A stalled
/// terminal (e.g. an unresponsive SSH link) must never freeze the client:
/// frames beyond MAX_DISPLAY_BACKLOG are dropped and one notice is emitted
/// once the backlog drains. The host keeps its own 256 KB replay buffer, so
/// a re-attach still recovers the tail of what was dropped here.
#[derive(Default)]
pub(super) struct DisplayBacklog {
    bytes: Vec<u8>,
    offset: usize,
    truncated: bool,
    dropped: usize,
}

impl DisplayBacklog {
    pub(super) fn is_empty(&self) -> bool {
        self.offset >= self.bytes.len()
    }

    /// Queue a whole output frame. Frames that would exceed the cap are
    /// dropped whole (frame boundaries stay clean) and remembered for the
    /// notice emitted when the backlog drains.
    pub(super) fn push_frame(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }
        if self.bytes.len() - self.offset + payload.len() > MAX_DISPLAY_BACKLOG {
            self.truncated = true;
            self.dropped += payload.len();
        } else {
            self.bytes.extend_from_slice(payload);
        }
    }

    /// Write as much as the terminal accepts right now (`fd` is O_NONBLOCK).
    /// WouldBlock keeps the remainder for the next poll; a real error means
    /// the terminal is gone and surfaces as Err.
    pub(super) fn pump(&mut self, fd: RawFd) -> io::Result<()> {
        loop {
            if self.offset >= self.bytes.len() {
                if self.truncated {
                    // Drain completed: emit the notice and settle the round.
                    self.queue_notice();
                    continue;
                }
                self.bytes.clear();
                self.offset = 0;
                return Ok(());
            }
            let count = unsafe {
                libc::write(fd, self.bytes[self.offset..].as_ptr() as *const _, self.bytes.len() - self.offset)
            };
            if count > 0 {
                self.offset += count as usize;
                // Compact amortized so a long backlog never shifts repeatedly.
                if self.offset >= 64 * 1024 {
                    self.bytes.copy_within(self.offset.., 0);
                    self.bytes.truncate(self.bytes.len() - self.offset);
                    self.offset = 0;
                }
                continue;
            }
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            if error.kind() == io::ErrorKind::WouldBlock {
                return Ok(());
            }
            return Err(error);
        }
    }

    /// Append the drop notice behind whatever is still buffered (data first,
    /// notice last), settling the current stall round. Bypasses the cap: a
    /// notice must never be dropped by the same cap it reports. `pump` emits
    /// it once the backlog drains; the flush-timeout path calls this directly
    /// for a last attempt on a terminal that never caught up.
    pub(super) fn queue_notice(&mut self) {
        if self.truncated {
            self.truncated = false;
            let dropped = self.dropped;
            self.dropped = 0;
            self.bytes.extend_from_slice(&truncated_notice(dropped));
        }
    }
}

fn truncated_notice(dropped: usize) -> Vec<u8> {
    format!(
        "\r\n\x1b[2m[output truncated: terminal was unresponsive; {dropped} bytes dropped]\x1b[0m\r\n"
    )
    .into_bytes()
}

struct TerminalGuard {
    fd: RawFd,
    output_fd: RawFd,
    original: libc::termios,
    /// Original fcntl flags of `output_fd`, restored on drop so the shared
    /// terminal description is not left O_NONBLOCK for the shell.
    original_flags: libc::c_int,
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
        // Non-blocking output: the attach display loop must never be suspended
        // by a stalled terminal (e.g. an unresponsive SSH link).
        let flags = unsafe { libc::fcntl(output_fd, libc::F_GETFL) };
        if flags < 0 || unsafe { libc::fcntl(output_fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            let error = io::Error::last_os_error();
            let _ = unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original) }; // roll back raw mode
            return Err(error);
        }
        Ok(Self { fd, output_fd, original, original_flags: flags })
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
        let _ = unsafe { libc::fcntl(self.output_fd, libc::F_SETFL, self.original_flags) };
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

/// Best-effort stderr notice that never blocks: the terminal itself may be
/// wedged (e.g. a stalled SSH link), so give the fd a short writability
/// window and drop the notice if it does not fit.
fn notice(text: &str) {
    let fd = libc::STDERR_FILENO;
    let mut pfd = pollfd(fd, libc::POLLOUT);
    if unsafe { libc::poll(&mut pfd, 1, 500) } > 0 && pfd.revents & libc::POLLOUT != 0 {
        // Single write, no retry loop: `fd` is a blocking terminal and the
        // available space after POLLOUT may be smaller than the text. A short
        // write is fine for a notice; looping here could block on a wedged
        // terminal again.
        unsafe { libc::write(fd, text.as_ptr() as *const _, text.len()) };
    }
}

fn pollfd(fd: RawFd, events: i16) -> libc::pollfd {
    libc::pollfd { fd, events, revents: 0 }
}
