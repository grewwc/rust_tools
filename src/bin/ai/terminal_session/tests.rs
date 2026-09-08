//! Offline tests for the terminal client/host pair. Every test uses its own
//! PTY pair (host + client share it) and its own temp registry root, so no
//! real terminal, real `a` process, or the real registry is touched.

use super::client::{
    attach, eligible, maybe_run_impl, reattach_hint, AttachEnd, DisplayBacklog, MAX_DISPLAY_BACKLOG,
};
use super::{host, registry, wire};
use crate::ai::cli::ParsedCli;
use std::io::{self, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn window() -> wire::Window {
    wire::Window { rows: 24, cols: 80 }
}

fn pty_pair() -> (std::fs::File, std::fs::File) {
    host::open_pty(window()).unwrap()
}

/// Spawn the host fixture on a temp registry root and return (root, socket).
fn start_host(worker: Command) -> (PathBuf, PathBuf) {
    // macOS caps Unix socket paths at SUN_LEN (104 bytes); temp_dir() on macOS
    // is already ~70 bytes, so keep the test root short and under /tmp.
    let root = std::path::PathBuf::from("/tmp").join(format!(
        "a-pty-test-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let name = registry::bootstrap_name(&uuid::Uuid::new_v4().to_string()).unwrap();
    let token = "test-token".to_string();
    let socket = root.join(&name);
    let root2 = root.clone();
    let name2 = name.clone();
    thread::spawn(move || host::fixture(root2, name2, token, worker));
    (root, socket)
}

fn connect_socket(socket: &Path, timeout: Duration) -> UnixStream {
    let deadline = Instant::now() + timeout;
    loop {
        match registry::connect(socket) {
            Ok(Some(stream)) => return stream,
            // Same startup race as the production path: the host holds the
            // lock before its listener socket file exists.
            Ok(None) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            // Same 0600 chmod race as the production path (see client.rs).
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {}
            Err(error) => panic!("connect failed: {error}"),
        }
        if Instant::now() >= deadline {
            panic!("timed out connecting to {}", socket.display());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// Blocking read from a nonblocking PTY master until `needle` appears.
fn read_until(fd: RawFd, needle: &[u8], timeout: Duration) -> Vec<u8> {
    let deadline = Instant::now() + timeout;
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    while !buffer.windows(needle.len()).any(|w| w == needle) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!(
                "timed out waiting for {:?}; got {:?}",
                String::from_utf8_lossy(needle),
                String::from_utf8_lossy(&buffer)
            );
        }
        let mut pollfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut pollfd, 1, remaining.as_millis() as i32) } <= 0 {
            continue;
        }
        match unsafe { libc::read(fd, chunk.as_mut_ptr() as *mut _, chunk.len()) } {
            n if n > 0 => buffer.extend_from_slice(&chunk[..n as usize]),
            _ => thread::sleep(Duration::from_millis(10)),
        }
    }
    buffer
}

fn write_all(fd: RawFd, bytes: &[u8]) {
    let mut offset = 0;
    while offset < bytes.len() {
        let count =
            unsafe { libc::write(fd, bytes[offset..].as_ptr() as *const _, bytes.len() - offset) };
        assert!(count > 0, "PTY master write failed");
        offset += count as usize;
    }
}

/// Non-blocking drain of `fd` into `out`; stops at the first WouldBlock.
fn drain_fd(fd: RawFd, out: &mut Vec<u8>) {
    let mut chunk = [0u8; 8192];
    loop {
        match unsafe { libc::read(fd, chunk.as_mut_ptr() as *mut _, chunk.len()) } {
            n if n > 0 => out.extend_from_slice(&chunk[..n as usize]),
            _ => break, // WouldBlock: nothing more for now
        }
    }
}

fn termios_of(fd: RawFd) -> libc::termios {
    let mut termios = unsafe { std::mem::zeroed::<libc::termios>() };
    assert_eq!(unsafe { libc::tcgetattr(fd, &mut termios) }, 0);
    termios
}

fn same_termios(a: &libc::termios, b: &libc::termios) -> bool {
    a.c_iflag == b.c_iflag
        && a.c_oflag == b.c_oflag
        && a.c_cflag == b.c_cflag
        // PENDIN (macOS "pending input" flag) is a transient kernel state bit
        // set while unread PTY input sits in the buffer; it is not a mode we
        // ever set or restore, so ignore it when comparing.
        && (a.c_lflag & !libc::PENDIN) == (b.c_lflag & !libc::PENDIN)
        && a.c_cc == b.c_cc
        // c_ispeed/c_ospeed are excluded: openpty reports speed 0 initially,
        // and the kernel normalizes it after any tcsetattr, so they differ
        // even when the modes were restored exactly.
}

/// Capture the remaining output after attach returns, including its cleanup.
/// The marker is written only after the guard has restored terminal state.
fn assert_client_reset(master: RawFd, slave: RawFd, mut output: Vec<u8>) {
    const RESET: &[u8] = b"\x1b7\x1b[r\x1b8\x1b[?2004l\x1b[0m\x1b[?25h";
    const DONE: &[u8] = b"__CLIENT_RETURNED__";
    write_all(slave, DONE);
    output.extend(read_until(master, DONE, Duration::from_secs(5)));
    assert!(output.starts_with(RESET), "attach must preserve the output cursor");
    let mut suffix = RESET.to_vec();
    suffix.extend_from_slice(DONE);
    assert!(output.ends_with(&suffix), "exit must reset modes on the supplied output fd");
    assert_eq!(output.windows(RESET.len()).filter(|w| *w == RESET).count(), 2);
}

/// A `ParsedCli` with every field at its default value (bare `a`).
fn default_cli() -> ParsedCli {
    ParsedCli {
        model: None,
        agent: None,
        clear: false,
        new_session: false,
        resume: false,
        session: None,
        files: String::new(),
        args: Vec::new(),
        list_tools: false,
        list_mcp_tools: false,
        list_skills: false,
        list_agents: false,
        no_skills: false,
        mcp_config: String::new(),
        help: false,
        version: false,
        interactive: false,
        reasoning_effort_override: None,
        thinking_disabled_override: false,
        max_tokens_override: None,
        note_search: false,
        note: None,
        note_flag: false,
        note_delete: None,
        note_edit: None,
        consolidate_knowledge: false,
        generate_completions: false,
        background: false,
        stop_session: None,
    }
}

#[test]
fn attach_relays_io_and_restores_termios() {
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "printf 'hi\\n'; read line; printf 'got:%s\\n' \"$line\""]);
    let (root, socket) = start_host(worker);
    let (master, slave) = pty_pair();
    let before = termios_of(slave.as_raw_fd());
    let (tx, rx) = mpsc::channel();
    let slave_fd = slave.as_raw_fd();
    thread::spawn(move || {
        let stream = connect_socket(&socket, Duration::from_secs(5));
        let end = attach(stream, "test-terminal".into(), window(), slave_fd, slave_fd);
        tx.send(end).unwrap();
    });
    // The worker's first output arrives relayed through the client.
    // PTY line discipline translates \n to \r\n (ONLCR), like a real terminal.
    let mut output = read_until(master.as_raw_fd(), b"hi\r\n", Duration::from_secs(5));
    // Keystrokes on the client's terminal reach the worker and come back.
    write_all(master.as_raw_fd(), b"world\n");
    output.extend(read_until(master.as_raw_fd(), b"got:world\r\n", Duration::from_secs(5)));
    let end = rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(end.code, 0);
    assert!(!end.detached);
    // Raw mode must not outlive the client.
    let after = termios_of(slave.as_raw_fd());
    if !same_termios(&after, &before) {
        eprintln!(
            "[debug] iflag {:x}->{:x} oflag {:x}->{:x} cflag {:x}->{:x} lflag {:x}->{:x} cc {:?}->{:?}",
            before.c_iflag, after.c_iflag, before.c_oflag, after.c_oflag,
            before.c_cflag, after.c_cflag, before.c_lflag, after.c_lflag,
            before.c_cc, after.c_cc
        );
    }
    assert!(same_termios(&after, &before));
    assert_client_reset(master.as_raw_fd(), slave_fd, output);
    let _ = root;
}

#[test]
fn detach_keeps_worker_and_reattach_replays_output() {
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "printf 'first\\n'; read line"]);
    let (root, socket) = start_host(worker);
    // First client attaches, then the host detaches it (the `/bg` path).
    let (master1, slave1) = pty_pair();
    let before1 = termios_of(slave1.as_raw_fd());
    let (tx1, rx1) = mpsc::channel();
    let slave1_fd = slave1.as_raw_fd();
    let socket1 = socket.clone();
    thread::spawn(move || {
        let stream = connect_socket(&socket1, Duration::from_secs(5));
        let end = attach(stream, "t1".into(), window(), slave1_fd, slave1_fd);
        tx1.send(end).unwrap();
    });
    let output1 = read_until(master1.as_raw_fd(), b"first\r\n", Duration::from_secs(5));
    wire::rpc(&socket, wire::Request::Detach { token: "test-token".into() }).unwrap();
    let end1 = rx1.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert!(end1.detached);
    assert_eq!(end1.code, 0);
    assert!(same_termios(&termios_of(slave1_fd), &before1));
    assert_client_reset(master1.as_raw_fd(), slave1_fd, output1);
    // Second client re-attaches and receives the replayed output, including a
    // clear-screen marker, then ends the worker with a keystroke.
    let (master2, slave2) = pty_pair();
    let (tx2, rx2) = mpsc::channel();
    let slave2_fd = slave2.as_raw_fd();
    let socket2 = socket.clone();
    thread::spawn(move || {
        let stream = connect_socket(&socket2, Duration::from_secs(5));
        let end = attach(stream, "t2".into(), window(), slave2_fd, slave2_fd);
        tx2.send(end).unwrap();
    });
    let replay = read_until(master2.as_raw_fd(), b"first\r\n", Duration::from_secs(5));
    assert!(replay.windows(4).any(|w| w == b"\x1b[2J"));
    write_all(master2.as_raw_fd(), b"bye\n");
    let end2 = rx2.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(end2.code, 0);
    assert!(!end2.detached);
    assert_client_reset(master2.as_raw_fd(), slave2_fd, replay);
    let _ = root;
}

#[test]
fn second_attach_is_rejected_while_attached() {
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "printf 'hi\\n'; read line"]);
    let (root, socket) = start_host(worker);
    let (master1, slave1) = pty_pair();
    let (tx1, rx1) = mpsc::channel();
    let slave1_fd = slave1.as_raw_fd();
    let socket1 = socket.clone();
    thread::spawn(move || {
        let stream = connect_socket(&socket1, Duration::from_secs(5));
        let end = attach(stream, "t1".into(), window(), slave1_fd, slave1_fd);
        tx1.send(end).unwrap();
    });
    read_until(master1.as_raw_fd(), b"hi\r\n", Duration::from_secs(5));
    // A second attach attempt surfaces the error instead of competing.
    let (master2, slave2) = pty_pair();
    let slave2_fd = slave2.as_raw_fd();
    let before2 = termios_of(slave2_fd);
    let socket2 = socket.clone();
    let error = thread::spawn(move || {
        let stream = connect_socket(&socket2, Duration::from_secs(5));
        let result = attach(stream, "t2".into(), window(), slave2_fd, slave2_fd);
        result.err().map(|e| e.to_string())
    })
    .join()
    .unwrap();
    let message = error.expect("second attach should have failed");
    assert!(message.contains("already attached"), "unexpected error: {message}");
    assert!(same_termios(&termios_of(slave2_fd), &before2));
    assert_client_reset(master2.as_raw_fd(), slave2_fd, Vec::new());
    // End the worker so the host and first client exit cleanly.
    write_all(master1.as_raw_fd(), b"x\n");
    let end1 = rx1.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(end1.code, 0);
    let _ = (master2, root);
}

#[test]
fn client_eligibility_is_restricted_to_bare_interactive_invocations() {
    let cli = default_cli();
    assert!(eligible(&cli));
    let mut cli = default_cli();
    cli.background = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.resume = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.new_session = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.help = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.list_tools = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.args = vec!["summarize the repo".into()];
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.session = Some("abc".into());
    assert!(eligible(&cli));
    let mut cli = default_cli();
    cli.interactive = true;
    assert!(!eligible(&cli));
    let mut cli = default_cli();
    cli.model = Some("gpt-5".into());
    assert!(!eligible(&cli));
}

#[test]
fn maybe_run_attaches_to_live_terminal_binding() {
    let _lock = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "printf 'hi\\n'; sleep 2"]);
    let (root, socket) = start_host(worker);
    let name = socket.file_name().unwrap().to_string_lossy().into_owned();
    registry::bind_terminal(
        &root,
        "tmux:test-pane",
        &registry::Binding { socket: name, session: None },
    )
    .unwrap();
    let previous = std::env::var("TMUX_PANE").ok();
    unsafe { std::env::set_var("TMUX_PANE", "test-pane") };
    let (master, slave) = pty_pair();
    let slave_fd = slave.as_raw_fd();
    let (tx, rx) = mpsc::channel();
    let cli = default_cli();
    let args = vec!["a".to_string()];
    let root2 = root.clone();
    thread::spawn(move || {
        let result = maybe_run_impl(&cli, &args, &root2, slave_fd, slave_fd)
            .map_err(|error| error.to_string());
        tx.send(result).unwrap();
    });
    read_until(master.as_raw_fd(), b"hi\r\n", Duration::from_secs(5));
    let result = rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(result, Some(0));
    match previous {
        Some(value) => unsafe { std::env::set_var("TMUX_PANE", value) },
        None => unsafe { std::env::remove_var("TMUX_PANE") },
    }
}

#[test]
fn display_backlog_drops_overflow_and_notices_once() {
    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let (read_fd, write_fd) = (fds[0], fds[1]);
    for fd in [read_fd, write_fd] {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags >= 0, "F_GETFL failed");
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) },
            0,
            "F_SETFL failed"
        );
    }
    // Fill the pipe so writes fail with WouldBlock, like a stalled terminal.
    let filler = [b'x'; 4096];
    loop {
        let count = unsafe { libc::write(write_fd, filler.as_ptr() as *const _, filler.len()) };
        if count > 0 {
            continue;
        }
        assert_eq!(
            io::Error::last_os_error().kind(),
            io::ErrorKind::WouldBlock,
            "pipe fill must end at EAGAIN"
        );
        break;
    }
    let mut backlog = DisplayBacklog::default();
    let kept = vec![b'a'; 100 * 1024];
    backlog.push_frame(&kept);
    // A frame that would exceed the cap is dropped (with a notice) instead of
    // growing the backlog without bound.
    let dropped = vec![b'b'; 5 * 1024 * 1024];
    backlog.push_frame(&dropped);
    // Terminal still full: pump must not block and must keep the remainder.
    backlog.pump(write_fd).unwrap();
    assert!(!backlog.is_empty());
    let mut output = Vec::new();
    // Discard the filler (pre-existing terminal content), then start fresh.
    drain_fd(read_fd, &mut output);
    output.clear();
    // Terminal recovers: pump writes only what fits, so drain between pumps.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !backlog.is_empty() {
        assert!(Instant::now() < deadline, "backlog must drain within 5s");
        backlog.pump(write_fd).unwrap();
        drain_fd(read_fd, &mut output);
    }
    drain_fd(read_fd, &mut output);
    assert!(output.starts_with(&kept[..]), "kept frame must arrive first");
    let notice = String::from_utf8_lossy(&output[kept.len()..]);
    assert!(notice.contains("[output truncated: terminal was unresponsive"));
    assert!(notice.contains(&format!("{} bytes dropped", dropped.len())));
    assert_eq!(notice.matches("output truncated").count(), 1, "notice must appear exactly once");
}

#[test]
fn stalled_terminal_does_not_freeze_client() {
    // A worker that floods output while the client's terminal never reads
    // (the PTY master is left untouched) used to block attach forever in the
    // blocking stdout write. The bounded backlog must let attach finish once
    // the worker exits.
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "i=0; while [ $i -lt 20000 ]; do echo 'payload line'; i=$((i+1)); done"]);
    let (root, socket) = start_host(worker);
    let (master, slave) = pty_pair();
    let slave_fd = slave.as_raw_fd();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let stream = connect_socket(&socket, Duration::from_secs(5));
        let end = attach(stream, "test-terminal".into(), window(), slave_fd, slave_fd);
        tx.send(end).unwrap();
    });
    // Never read `master`: attach must still return (worker exit plus the
    // bounded flush window) instead of hanging on a full terminal buffer.
    let end = rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
    assert_eq!(end.code, 0);
    let _ = (master, root);
}

#[test]
fn second_stall_notice_reports_only_second_round_drops() {
    // A previous stall must not leak into the next notice: the dropped counter
    // resets when the notice is emitted, so a second stall reports only its
    // own bytes (before the fix, both rounds were summed into the second
    // notice, e.g. "7 MB dropped" when the second round only dropped 2 MB).
    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let (read_fd, write_fd) = (fds[0], fds[1]);
    for fd in [read_fd, write_fd] {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags >= 0, "F_GETFL failed");
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) },
            0,
            "F_SETFL failed"
        );
    }
    let mut backlog = DisplayBacklog::default();
    let drain_all = |backlog: &mut DisplayBacklog, write_fd: RawFd, read_fd: RawFd| {
        let mut output = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !backlog.is_empty() {
            assert!(Instant::now() < deadline, "backlog must drain within 5s");
            backlog.pump(write_fd).unwrap();
            drain_fd(read_fd, &mut output);
        }
        drain_fd(read_fd, &mut output);
        output
    };
    // Each round keeps a small frame so the backlog is non-empty and pump
    // runs to the truncation branch (a lone dropped frame leaves `bytes`
    // empty; the notice is still emitted because attach calls pump every
    // poll, but this test must drive it explicitly).
    let kept = vec![b'a'; 1024];
    // Round 1: one big frame dropped while the terminal is stalled.
    let first_dropped = vec![b'b'; 5 * 1024 * 1024];
    backlog.push_frame(&kept);
    backlog.push_frame(&first_dropped);
    let first = drain_all(&mut backlog, write_fd, read_fd);
    let first_text = String::from_utf8_lossy(&first);
    assert!(first_text.contains(&format!("{} bytes dropped", first_dropped.len())));
    // Round 2: a 6 MB frame also exceeds the cap; the notice must report
    // only this round's 6 MB, not the 5 MB from round 1.
    let second_dropped = vec![b'c'; 6 * 1024 * 1024];
    backlog.push_frame(&kept);
    backlog.push_frame(&second_dropped);
    let second = drain_all(&mut backlog, write_fd, read_fd);
    let second_text = String::from_utf8_lossy(&second);
    assert!(second_text.contains(&format!("{} bytes dropped", second_dropped.len())));
    assert!(
        !second_text.contains(&format!("{} bytes dropped", first_dropped.len() + second_dropped.len())),
        "second notice must not sum both stalls"
    );
}

#[test]
fn exit_code_one_is_not_reported_as_lost_connection() {
    // A worker that exits with status 1 must be returned verbatim, not
    // mislabeled as a lost connection (which would trigger a misleading
    // re-attach hint for an already-finished session).
    let mut worker = Command::new("/bin/sh");
    worker.args(["-c", "exit 1"]);
    let (root, socket) = start_host(worker);
    let (master, slave) = pty_pair();
    let slave_fd = slave.as_raw_fd();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let stream = connect_socket(&socket, Duration::from_secs(5));
        let end = attach(stream, "test-terminal".into(), window(), slave_fd, slave_fd);
        tx.send(end).unwrap();
    });
    let end = rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(end.code, 1, "worker exit status must be forwarded");
    assert!(!end.detached);
    assert!(!end.connection_lost, "normal exit 1 must not look like a lost connection");
    let _ = (master, root);
}

#[test]
fn socket_eof_is_reported_as_lost_connection() {
    // Host-side death: the socket dies without EXIT/DETACHED, so the client
    // must flag a lost connection (the worker may still be alive elsewhere).
    let (master, slave) = pty_pair();
    let slave_fd = slave.as_raw_fd();
    let (peer, held) = UnixStream::pair().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let end = attach(peer, "test-terminal".into(), window(), slave_fd, slave_fd);
        tx.send(end).unwrap();
    });
    // Let attach write its REQUEST frame first, then close the peer like a
    // dead host (socket EOF without any EXIT/DETACHED frame).
    thread::sleep(Duration::from_millis(300));
    drop(held);
    let end = rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(end.code, 1);
    assert!(!end.detached);
    assert!(end.connection_lost, "socket EOF without EXIT must be a lost connection");
    let _ = master;
}

#[test]
fn reattach_hint_only_for_lost_connections_with_session() {
    let with_session = || Some("sess-1".into());
    let lost = AttachEnd { code: 1, detached: false, session: with_session(), connection_lost: true };
    assert!(reattach_hint(&lost).is_some(), "lost connection with a session id must hint");
    assert!(
        reattach_hint(&lost).unwrap().contains("a -ss sess-1"),
        "hint must include the re-attach command"
    );
    // Normal worker exit with code 1: no hint, the session is already gone.
    let normal = AttachEnd { code: 1, detached: false, session: with_session(), connection_lost: false };
    assert!(reattach_hint(&normal).is_none(), "normal exit 1 must not hint");
    // Lost connection without a session id: nothing useful to hint.
    let anonymous = AttachEnd { code: 1, detached: false, session: None, connection_lost: true };
    assert!(reattach_hint(&anonymous).is_none(), "no session id means no hint");
}

#[test]
fn host_death_flushes_backlog_before_returning() {
    // Unexpected EOF must not throw away buffered tail output: like the
    // EXIT/DETACHED path, the client drains the display backlog for a
    // bounded window before returning the lost-connection result.
    let (master, slave) = pty_pair();
    let slave_fd = slave.as_raw_fd();
    let (peer, mut held) = UnixStream::pair().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let end = attach(peer, "test-terminal".into(), window(), slave_fd, slave_fd);
        tx.send(end).unwrap();
    });
    // REPLY, then one OUTPUT frame, then the host dies without EXIT/DETACHED.
    let reply = wire::frame(wire::REPLY, br#"{"session":"s1"}"#).unwrap();
    held.write_all(&reply).unwrap();
    let tail = b"tail output before host death";
    held.write_all(&wire::frame(wire::OUTPUT, tail).unwrap()).unwrap();
    thread::sleep(Duration::from_millis(300));
    drop(held); // host dies
    let end = rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(end.code, 1);
    assert!(end.connection_lost);
    // The buffered OUTPUT must have been flushed to the terminal, not lost.
    let out = read_until(master.as_raw_fd(), tail, Duration::from_secs(5));
    assert!(
        out.windows(tail.len()).any(|w| w == tail),
        "tail output was lost: {:?}",
        String::from_utf8_lossy(&out)
    );
    let _ = master;
}

#[test]
fn truncation_notice_survives_a_backlog_at_cap() {
    // The flush-timeout path queues the drop notice behind a backlog that
    // never drained. The notice must bypass the cap (otherwise it would be
    // dropped by the very cap it reports) and still drain out.
    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let (read_fd, write_fd) = (fds[0], fds[1]);
    for fd in [read_fd, write_fd] {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(flags >= 0, "F_GETFL failed");
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) },
            0,
            "F_SETFL failed"
        );
    }
    let mut backlog = DisplayBacklog::default();
    // Fill almost to the cap, then drop a frame on top: a stall round begins.
    backlog.push_frame(&vec![b'x'; MAX_DISPLAY_BACKLOG - 64]);
    backlog.push_frame(&vec![b'y'; 1024]); // exceeds the cap, dropped
    // Simulate the flush-timeout path: queue the notice while the backlog is
    // still full (with the notice appended it exceeds the cap).
    backlog.queue_notice();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut out = Vec::new();
    while !backlog.is_empty() {
        assert!(Instant::now() < deadline, "backlog must drain within 5s");
        backlog.pump(write_fd).unwrap();
        drain_fd(read_fd, &mut out);
    }
    drain_fd(read_fd, &mut out);
    let text = String::from_utf8_lossy(&out);
    assert!(
        text.contains("1024 bytes dropped"),
        "notice missing or wrong: {:?}",
        &text[text.len().saturating_sub(200)..]
    );
}
