use super::{registry::{self, Binding, Lease}, replay::Replay, wire::{self, Queue, Request, Reply, Window}};
use std::{
    fs::File,
    io::{self, Read},
    os::unix::{io::{AsRawFd, FromRawFd, RawFd}, net::UnixStream, process::ExitStatusExt},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const COMPLETED_RETENTION: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(PartialEq)]
enum Role { Pending, Attached, Closing, Exit }

struct Peer {
    stream: UnixStream,
    decoder: wire::Decoder,
    output: Queue,
    role: Role,
    alive: bool,
    since: Instant,
    alias: Option<String>,
}

struct Worker {
    child: Child,
    master: File,
    input: Queue,
    ended: Option<(i32, Instant)>,
}

struct Claim { session: String, lease: Option<Lease> }

struct Host {
    root: PathBuf,
    name: String,
    token: String,
    bootstrap: Lease,
    current: Option<Claim>,
    pending: Option<Claim>,
    terminal: Option<String>,
    peers: Vec<Peer>,
    worker: Option<Worker>,
    command: Option<Command>,
    worker_args: Vec<String>,
    replay: Replay,
    completed: Option<(i32, Instant)>,
    completion_delivered: bool,
    started: Instant,
}

pub(super) fn run(root: PathBuf, name: String, token: String, worker_args: Vec<String>) -> io::Result<i32> {
    Host::new(root, name, token, worker_args, None)?.run()
}

impl Host {
    fn new(root: PathBuf, name: String, token: String, worker_args: Vec<String>, command: Option<Command>) -> io::Result<Self> {
        let bootstrap = Lease::bind(root.join(&name))?;
        Ok(Self { root, name, token, bootstrap, current: None, pending: None, terminal: None,
            peers: Vec::new(), worker: None, command, worker_args, replay: Replay::default(),
            completed: None, completion_delivered: false, started: Instant::now() })
    }

    fn run(mut self) -> io::Result<i32> {
        loop {
            if let Some((code, ended)) = self.completed {
                if self.completion_delivered || ended.elapsed() >= COMPLETED_RETENTION { return Ok(code); }
            } else if self.worker.is_none() && self.started.elapsed() > Duration::from_secs(15) {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "terminal client did not attach"));
            }
            // Every descriptor in this event loop is nonblocking. A partial
            // handshake or a slow frontend must never suspend the worker.
            let mut descriptors = vec![pollfd(self.bootstrap.fd(), libc::POLLIN)];
            for claim in [&self.current, &self.pending].into_iter().flatten() {
                if let Some(lease) = &claim.lease { descriptors.push(pollfd(lease.fd(), libc::POLLIN)); }
            }
            for peer in &self.peers {
                descriptors.push(pollfd(peer.stream.as_raw_fd(), libc::POLLIN |
                    if peer.output.is_empty() { 0 } else { libc::POLLOUT }));
            }
            if let Some(worker) = &self.worker {
                descriptors.push(pollfd(worker.master.as_raw_fd(), libc::POLLIN |
                    if worker.input.is_empty() { 0 } else { libc::POLLOUT }));
            }
            let result = unsafe { libc::poll(descriptors.as_mut_ptr(), descriptors.len() as _, 100) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted { continue; }
                return Err(error);
            }
            self.accept();
            for index in 0..self.peers.len() { self.read_peer(index); }
            self.read_worker();
            for peer in &mut self.peers {
                if peer.alive && peer.output.flush(&mut peer.stream).is_err() { peer.alive = false; }
                if peer.alive && matches!(peer.role, Role::Closing | Role::Exit) && peer.output.is_empty() {
                    if peer.role == Role::Exit { self.completion_delivered = true; }
                    peer.alive = false;
                }
                if peer.role != Role::Attached && peer.since.elapsed() > CONNECT_TIMEOUT { peer.alive = false; }
            }
            self.peers.retain(|peer| peer.alive);
        }
    }

    fn accept(&mut self) {
        let mut listeners = vec![(&self.bootstrap, None)];
        for claim in [&self.current, &self.pending].into_iter().flatten() {
            if let Some(lease) = &claim.lease { listeners.push((lease, Some(claim.session.clone()))); }
        }
        for (lease, alias) in listeners {
            for _ in 0..16 {
                let Ok((stream, _)) = lease.listener.accept() else { break; };
                if self.peers.len() >= 32 || stream.set_nonblocking(true).is_err() { continue; }
                self.peers.push(Peer { stream, decoder: wire::Decoder::default(), output: Queue::default(),
                    role: Role::Pending, alive: true, since: Instant::now(), alias: alias.clone() });
            }
        }
    }

    fn read_peer(&mut self, index: usize) {
        if !self.peers[index].alive || matches!(self.peers[index].role, Role::Closing | Role::Exit) { return; }
        let mut bytes = [0; 8192];
        for _ in 0..16 {
            let result = self.peers[index].stream.read(&mut bytes);
            match result {
                Ok(0) => { self.peers[index].alive = false; break; }
                Ok(count) => {
                    if self.peers[index].decoder.push(&bytes[..count]).is_err() {
                        self.peers[index].alive = false; break;
                    }
                    loop {
                        match self.peers[index].decoder.next() {
                            Ok(Some((kind, payload))) => {
                                if let Err(error) = self.message(index, kind, &payload) {
                                    let peer = &mut self.peers[index];
                                    peer.role = Role::Closing;
                                    peer.since = Instant::now();
                                    if peer.output.json(wire::REPLY, &Reply { error: Some(error.to_string()), ..Reply::default() }).is_err() {
                                        peer.alive = false;
                                    }
                                }
                                if self.peers[index].role != Role::Attached { return; }
                            }
                            Ok(None) => break,
                            Err(_) => { self.peers[index].alive = false; return; }
                        }
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.peers[index].alive = false; break; }
            }
        }
    }

    fn message(&mut self, index: usize, kind: u8, payload: &[u8]) -> io::Result<()> {
        if self.peers[index].role == Role::Pending {
            if kind != wire::REQUEST { return Err(io::Error::other("expected PTY handshake")); }
            let request: Request = serde_json::from_slice(payload)?;
            if let Request::Attach { terminal, window } = request { return self.attach(index, terminal, window); }
            let token = match &request {
                Request::Register { token, .. } | Request::Claim { token, .. } |
                Request::Commit { token } | Request::Rollback { token } |
                Request::Detach { token } | Request::Terminal { token } => token,
                Request::Attach { .. } => unreachable!(),
            };
            if token != &self.token { return Err(io::Error::new(io::ErrorKind::PermissionDenied, "invalid PTY control token")); }
            match request {
                Request::Register { session, .. } => {
                    self.claim(session)?;
                    if let Err(error) = self.commit() { self.pending = None; return Err(error); }
                }
                Request::Claim { session, .. } => self.claim(session)?,
                Request::Commit { .. } => self.commit()?,
                Request::Rollback { .. } => self.pending = None,
                Request::Detach { .. } => {
                    for peer in &mut self.peers {
                        if peer.alive && peer.role == Role::Attached {
                            if peer.output.framed(wire::DETACHED, &[]).is_err() { peer.alive = false; }
                            peer.role = Role::Closing;
                            peer.since = Instant::now();
                        }
                    }
                }
                Request::Terminal { .. } => {},
                Request::Attach { .. } => unreachable!(),
            }
            let reply = self.reply();
            let peer = &mut self.peers[index];
            peer.role = Role::Closing;
            peer.since = Instant::now();
            return peer.output.json(wire::REPLY, &reply);
        }
        match kind {
            wire::INPUT => {
                if let Some(worker) = &mut self.worker { worker.input.push(payload.to_vec())?; }
            }
            wire::RESIZE => {
                let window = serde_json::from_slice(payload)?;
                if let Some(worker) = &self.worker { set_window(worker.master.as_raw_fd(), window)?; }
            }
            _ => return Err(io::Error::other("invalid PTY client frame")),
        }
        Ok(())
    }

    fn reply(&self) -> Reply {
        Reply { error: None, terminal: self.terminal.clone(), session: self.current.as_ref().map(|claim| claim.session.clone()) }
    }

    fn claim(&mut self, session: String) -> io::Result<()> {
        if self.pending.is_some() { return Err(io::Error::other("a terminal session switch is already pending")); }
        let path = self.root.join(registry::session_name(&session)?);
        let lease = if self.current.as_ref().is_some_and(|claim| claim.session == session) { None }
            else { Some(Lease::bind(path).map_err(|error| io::Error::other(format!("session {session} is already running or unavailable: {error}")))?) };
        self.pending = Some(Claim { session, lease });
        Ok(())
    }

    fn commit(&mut self) -> io::Result<()> {
        let pending = self.pending.as_ref().ok_or_else(|| io::Error::other("no pending terminal session switch"))?;
        if let Some(terminal) = &self.terminal {
            registry::bind_terminal(&self.root, terminal, &Binding { socket: self.name.clone(), session: Some(pending.session.clone()) })?;
        }
        let pending = self.pending.take().unwrap();
        if pending.lease.is_some() { self.current = Some(pending); }
        Ok(())
    }

    fn attach(&mut self, index: usize, terminal: String, window: Window) -> io::Result<()> {
        if terminal.is_empty() || terminal.len() > 1024 { return Err(io::Error::other("invalid terminal identity")); }
        if self.peers.iter().any(|peer| peer.alive && matches!(peer.role, Role::Attached | Role::Exit)) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "session is already attached; detach it with /bg first"));
        }
        if self.pending.is_some() || self.peers[index].alias.as_ref().is_some_and(|alias|
            !self.current.as_ref().is_some_and(|claim| &claim.session == alias)) {
            return Err(io::Error::other("session alias is changing; try attaching again"));
        }
        if let Some(worker) = &self.worker { set_window(worker.master.as_raw_fd(), window)?; }
        registry::bind_terminal(&self.root, &terminal, &Binding { socket: self.name.clone(), session: self.reply().session })?;
        if let Some(previous) = &self.terminal {
            if previous != &terminal { let _ = registry::clear_terminal(&self.root, previous, &self.name); }
        }
        self.terminal = Some(terminal.clone());
        if self.worker.is_none() && self.completed.is_none() { self.spawn_worker(window, &terminal)?; }
        let reply = self.reply();
        let replay = self.replay.snapshot();
        let peer = &mut self.peers[index];
        peer.output.json(wire::REPLY, &reply)?;
        if !replay.is_empty() {
            peer.output.framed(wire::OUTPUT, b"\x1b[0m\x1b[2J\x1b[H")?;
            for chunk in replay.chunks(wire::MAX_FRAME) { peer.output.framed(wire::OUTPUT, chunk)?; }
        }
        peer.role = Role::Attached;
        if let Some((code, _)) = self.completed {
            peer.output.framed(wire::EXIT, &code.to_be_bytes())?;
            peer.role = Role::Exit;
            peer.since = Instant::now();
        }
        Ok(())
    }

    fn spawn_worker(&mut self, window: Window, terminal: &str) -> io::Result<()> {
        let (master, slave) = open_pty(window)?;
        let mut command = if let Some(command) = self.command.take() { command } else {
            let mut command = Command::new(std::env::current_exe()?);
            let id = &self.name[2..self.name.len() - 5];
            command.args(["--terminal-worker", id, &self.token, terminal]);
            command.args(self.worker_args.iter().skip(1));
            command
        };
        command.stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?)).stderr(Stdio::from(slave));
        let child = command.spawn()?;
        self.worker = Some(Worker { child, master, input: Queue::default(), ended: None });
        Ok(())
    }

    fn read_worker(&mut self) {
        let Some(worker) = &mut self.worker else { return; };
        // Failed input only detaches that frontend; output is still drained and
        // the worker is never stopped or restarted on a transport error.
        if worker.input.flush(&mut worker.master).is_err() { worker.input = Queue::default(); }
        let mut chunks = Vec::new();
        let mut eof = false;
        let mut bytes = [0; 8192];
        for _ in 0..16 {
            match worker.master.read(&mut bytes) {
                Ok(0) => { eof = true; break; }
                Ok(count) => chunks.push(bytes[..count].to_vec()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => { eof = true; break; }
            }
        }
        if worker.ended.is_none() {
            if let Ok(Some(status)) = worker.child.try_wait() {
                worker.ended = Some((status.code().unwrap_or_else(|| 128 + status.signal().unwrap_or(1)), Instant::now()));
            }
        }
        let finished = worker.ended.filter(|(_, since)| eof || since.elapsed() > Duration::from_secs(2));
        for chunk in chunks {
            self.replay.push(&chunk);
            for peer in &mut self.peers {
                if peer.alive && peer.role == Role::Attached && peer.output.framed(wire::OUTPUT, &chunk).is_err() {
                    peer.alive = false;
                }
            }
        }
        if let Some((code, _)) = finished {
            self.worker = None;
            self.completed = Some((code, Instant::now()));
            for peer in &mut self.peers {
                if peer.alive && peer.role == Role::Attached {
                    if peer.output.framed(wire::EXIT, &code.to_be_bytes()).is_err() { peer.alive = false; }
                    peer.role = Role::Exit;
                    peer.since = Instant::now();
                }
            }
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        if let Some(terminal) = &self.terminal { let _ = registry::clear_terminal(&self.root, terminal, &self.name); }
    }
}

fn pollfd(fd: RawFd, events: i16) -> libc::pollfd { libc::pollfd { fd, events, revents: 0 } }

pub(super) fn set_window(fd: RawFd, window: Window) -> io::Result<()> {
    let size = libc::winsize { ws_row: window.rows.max(1), ws_col: window.cols.max(1), ws_xpixel: 0, ws_ypixel: 0 };
    if unsafe { libc::ioctl(fd, libc::TIOCSWINSZ as _, &size) } < 0 { return Err(io::Error::last_os_error()); }
    Ok(())
}

pub(super) fn open_pty(window: Window) -> io::Result<(File, File)> {
    let (mut master, mut slave) = (-1, -1);
    if unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut::<libc::c_char>(),
            std::ptr::null_mut::<libc::termios>(),
            std::ptr::null_mut::<libc::winsize>(),
        )
    } < 0 {
        return Err(io::Error::last_os_error());
    }
    let (master, slave) = unsafe { (File::from_raw_fd(master), File::from_raw_fd(slave)) };
    for fd in [master.as_raw_fd(), slave.as_raw_fd()] {
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0 { return Err(io::Error::last_os_error()); }
    }
    let flags = unsafe { libc::fcntl(master.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(master.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    set_window(master.as_raw_fd(), window)?;
    Ok((master, slave))
}

#[cfg(test)]
pub(super) fn fixture(root: PathBuf, name: String, token: String, command: Command) -> io::Result<i32> {
    Host::new(root, name, token, Vec::new(), Some(command))?.run()
}