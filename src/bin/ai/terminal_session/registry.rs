use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::{fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
        io::{AsRawFd, RawFd}, net::{UnixListener, UnixStream}},
    path::{Path, PathBuf},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(super) fn root() -> io::Result<PathBuf> {
    let path = PathBuf::from(format!("/tmp/a-pty-{}", unsafe { libc::geteuid() }));
    private_dir(&path)?;
    Ok(path)
}

pub(super) fn private_dir(path: &Path) -> io::Result<()> {
    match DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {},
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {},
        Err(error) => return Err(error),
    }
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_dir() || meta.uid() != unsafe { libc::geteuid() }
        || meta.mode() & 0o777 != 0o700
    {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "unsafe PTY registry directory"));
    }
    Ok(())
}

fn digest(value: &str) -> String {
    Sha256::digest(value.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn session_name(session: &str) -> io::Result<String> {
    if session.is_empty() || session.len() > 128
        || !session.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid session id"));
    }
    Ok(format!("s-{}.sock", digest(session)))
}

pub(super) fn bootstrap_name(id: &str) -> io::Result<String> {
    let id = uuid::Uuid::parse_str(id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid PTY bootstrap id"))?;
    Ok(format!("b-{}.sock", id.simple()))
}

fn secure_open(path: &Path, create: bool) -> io::Result<File> {
    let file = OpenOptions::new().read(true).write(true).create(create).mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.uid() != unsafe { libc::geteuid() }
        || meta.mode() & 0o777 != 0o600 || meta.nlink() != 1
    {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, "unsafe PTY registry file"));
    }
    Ok(file)
}

fn lock(path: &Path, create: bool, blocking: bool) -> io::Result<File> {
    let file = secure_open(path, create)?;
    let operation = libc::LOCK_EX | if blocking { 0 } else { libc::LOCK_NB };
    if unsafe { libc::flock(file.as_raw_fd(), operation) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

fn lock_path(path: &Path) -> PathBuf { path.with_extension("lock") }

pub(super) struct Lease {
    pub(super) listener: UnixListener,
    path: PathBuf,
    inode: u64,
    _lock: File,
}

impl Lease {
    pub(super) fn bind(path: PathBuf) -> io::Result<Self> {
        let guard = lock(&lock_path(&path), true, false)?;
        match fs::symlink_metadata(&path) {
            Ok(meta) => {
                if !meta.file_type().is_socket() || meta.uid() != unsafe { libc::geteuid() } {
                    return Err(io::Error::new(io::ErrorKind::PermissionDenied, "unsafe stale PTY socket"));
                }
                fs::remove_file(&path)?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {},
            Err(error) => return Err(error),
        }
        let listener = UnixListener::bind(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let inode = fs::symlink_metadata(&path)?.ino();
        Ok(Self { listener, path, inode, _lock: guard })
    }
    pub(super) fn fd(&self) -> RawFd { self.listener.as_raw_fd() }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path).is_ok_and(|meta| meta.ino() == self.inode) {
            let _ = fs::remove_file(&self.path);
        }
        // Never unlink the lock inode: a waiter could otherwise lock an old inode
        // while a new owner locks a newly-created one at the same path.
    }
}

pub(super) fn connect(path: &Path) -> io::Result<Option<UnixStream>> {
    match lock(&lock_path(path), false, false) {
        Ok(_stale) => Ok(None),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            let meta = fs::symlink_metadata(path)?;
            if !meta.file_type().is_socket() || meta.uid() != unsafe { libc::geteuid() }
                || meta.mode() & 0o777 != 0o600
            {
                return Err(io::Error::new(io::ErrorKind::PermissionDenied, "unsafe PTY socket"));
            }
            UnixStream::connect(path).map(Some)
        }
        Err(error) => Err(error),
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct Binding { pub(super) socket: String, pub(super) session: Option<String> }

fn terminal_path(root: &Path, terminal: &str) -> PathBuf {
    root.join(format!("t-{}.terminal", digest(terminal)))
}

pub(super) fn binding(root: &Path, terminal: &str) -> io::Result<Option<Binding>> {
    let path = terminal_path(root, terminal);
    let _guard = lock(&lock_path(&path), true, true)?;
    read_binding(&path)
}

fn read_binding(path: &Path) -> io::Result<Option<Binding>> {
    let file = match secure_open(path, false) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut data = Vec::new();
    file.take(4097).read_to_end(&mut data)?;
    let binding: Binding = serde_json::from_slice(&data)?;
    // A registry record may name only a bootstrap socket in the private root.
    if !binding.socket.starts_with("b-") || !binding.socket.ends_with(".sock")
        || bootstrap_name(&binding.socket[2..binding.socket.len()-5])? != binding.socket
    {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid terminal binding"));
    }
    Ok(Some(binding))
}

pub(super) fn bind_terminal(root: &Path, terminal: &str, binding: &Binding) -> io::Result<()> {
    let path = terminal_path(root, terminal);
    let _guard = lock(&lock_path(&path), true, true)?;
    let tmp = root.join(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(&tmp)?;
        file.write_all(&serde_json::to_vec(binding)?)?;
        fs::rename(&tmp, path)
    })();
    if result.is_err() { let _ = fs::remove_file(tmp); }
    result
}

pub(super) fn clear_terminal(root: &Path, terminal: &str, socket: &str) -> io::Result<()> {
    let path = terminal_path(root, terminal);
    let _guard = lock(&lock_path(&path), true, true)?;
    if read_binding(&path)?.is_some_and(|binding| binding.socket == socket) {
        fs::remove_file(path)?;
    }
    Ok(())
}
