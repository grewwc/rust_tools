//! Read-only Git changes in a serve session's cwd and eligible linked worktrees.

use std::{
    ffi::{CString, OsStr},
    fs::{self, DirBuilder, File, Metadata, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, symlink},
        },
    },
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, ExitStatus, Stdio},
    sync::LazyLock,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{Path as RoutePath, Query, State, rejection::QueryRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rust_tools::commonw::FastSet;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{ServeState, SessionStore, bad_request, check_auth, conflict};

const FILE_LIMIT: usize = 200;
const LIST_LIMIT: usize = 64 * 1024;
const DIFF_LINES: usize = 1000;
const DETAIL_LIMIT: usize = 128 * 1024;
const DIFF_INPUT_LIMIT: usize = 2 * 1024 * 1024;
const STATUS_LIMIT: usize = 4 * 1024 * 1024;
const INDEX_LIMIT: u64 = 32 * 1024 * 1024;
const PROBE_LIMIT: usize = 8192;
const TIME_LIMIT: Duration = Duration::from_secs(5);
const WORKTREE_LIMIT: usize = 32;
const WORKTREE_LIST_LIMIT: usize = 1024 * 1024;
const WORKTREE_TIME_LIMIT: Duration = Duration::from_millis(500);
const SCAN_RESPONSE_MARGIN: Duration = Duration::from_millis(250);
const CURRENT_SCOPE: &str = "current";
const SCRATCH_PREFIX: &str = ".a-serve-changes-";
const LIST_TRUNCATED: &str = "file list limit reached";
const DIFF_TRUNCATED: &str = "diff limit reached";
const SCOPES_TRUNCATED: &str = "worktree scope limit reached";
const SCOPES_TIMEOUT: &str = "worktree scan timed out";
static CAPACITY: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
static REVISION_KEY: LazyLock<Uuid> = LazyLock::new(Uuid::new_v4);

pub(super) fn routes(app: Router<ServeState>) -> Router<ServeState> {
    app.route("/sessions/{id}/changes", get(list_changes))
        .route("/sessions/{id}/changes/scopes", get(changes_scopes))
        .route("/sessions/{id}/changes/{file_id}", get(file_change))
}

type ParsedQuery<T> = std::result::Result<Query<T>, QueryRejection>;

#[derive(Deserialize)]
struct ScopeQuery {
    scope: Option<String>,
}

#[derive(Deserialize)]
struct RevisionQuery {
    revision: Option<String>,
}

struct DetailRequest {
    file_id: String,
    revision: Option<String>,
}

enum ChangesRequest {
    Scopes,
    Snapshot {
        scope_id: Result<Option<String>>,
        detail: Option<DetailRequest>,
    },
}

fn scope_selection(query: ParsedQuery<ScopeQuery>) -> Result<Option<String>> {
    let Query(query) = query.map_err(|_| Failure::UnknownScope)?;
    match query.scope {
        None => Ok(None),
        Some(id) if id == CURRENT_SCOPE => Ok(None),
        Some(id)
            if id.strip_prefix("wt-").is_some_and(|hash| {
                hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            }) =>
        {
            Ok(Some(id))
        }
        Some(_) => Err(Failure::UnknownScope),
    }
}

async fn list_changes(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath(id): RoutePath<String>,
    query: ParsedQuery<ScopeQuery>,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Snapshot {
            scope_id: scope_selection(query),
            detail: None,
        },
    )
    .await
}

async fn changes_scopes(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath(id): RoutePath<String>,
) -> Response {
    changes(state, headers, id, ChangesRequest::Scopes).await
}

async fn file_change(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath((id, file_id)): RoutePath<(String, String)>,
    scope: ParsedQuery<ScopeQuery>,
    revision: ParsedQuery<RevisionQuery>,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Snapshot {
            scope_id: scope_selection(scope),
            detail: Some(DetailRequest {
                file_id,
                revision: revision.ok().and_then(|Query(query)| query.revision),
            }),
        },
    )
    .await
}

async fn changes(
    state: ServeState,
    headers: HeaderMap,
    id: String,
    request: ChangesRequest,
) -> Response {
    if let Err(error) = check_auth(&state, &headers) {
        return error.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request("invalid session id".to_string()).into_response();
    }
    let Ok(permit) = CAPACITY.try_acquire() else {
        return Failure::Busy.into_response();
    };
    let deadline = Instant::now() + TIME_LIMIT;
    let job = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let store = SessionStore::new(&state.history_file);
        if !store.session_exists(&id)? {
            return Err(Failure::Session);
        }
        let mut current = Scope::new(&state, &store, &id)?;
        let git = Git { deadline };
        let response = match request {
            ChangesRequest::Scopes => list_scopes(git, &mut current, &id)?.into_response(),
            ChangesRequest::Snapshot { scope_id, detail } => {
                read_changes(git, &mut current, &id, scope_id?, detail)?
            }
        };
        current.verify(&state, &store, &id)?;
        Ok::<_, Failure>(response)
    });
    let mut response = match tokio::time::timeout_at(deadline.into(), job).await {
        Ok(Ok(Ok(response))) => response,
        Ok(Ok(Err(error))) => error.into_response(),
        Ok(Err(_)) => Failure::Git.into_response(),
        Err(_) => Failure::Timeout.into_response(),
    };
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response
}

#[derive(Debug)]
enum Failure {
    Session,
    File,
    Cwd,
    UnknownScope,
    Stale,
    Git,
    Busy,
    Timeout,
    InventoryLimit,
}

type Result<T> = std::result::Result<T, Failure>;

impl From<io::Error> for Failure {
    fn from(_: io::Error) -> Self {
        Self::Git
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Session => (StatusCode::NOT_FOUND, "session not found"),
            Self::File => (StatusCode::NOT_FOUND, "change not found"),
            Self::Cwd => {
                return conflict("session cwd is unavailable or not a Git working tree".into())
                    .into_response();
            }
            Self::UnknownScope => (StatusCode::BAD_REQUEST, "unknown changes scope"),
            Self::Stale => {
                return conflict("changes revision is missing or stale".into()).into_response();
            }
            Self::Git => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Git changes could not be read",
            ),
            Self::Busy => (StatusCode::SERVICE_UNAVAILABLE, "changes service is busy"),
            Self::Timeout => (StatusCode::SERVICE_UNAVAILABLE, "changes request timed out"),
            Self::InventoryLimit => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Git inventory exceeds the collection limit",
            ),
        };
        (status, Json(serde_json::json!({"error": message}))).into_response()
    }
}

struct Scope {
    id: String,
    cwd: PathBuf,
    directory: File,
    private: Vec<PathBuf>,
    sidecar_dir: PathBuf,
}

#[derive(Serialize)]
struct ScopeInfo {
    id: String,
    label: String,
    path: String,
    current: bool,
}

impl Scope {
    fn cwd(state: &ServeState, id: &str) -> Result<PathBuf> {
        // Unlike the turn cwd helper, this route must not fall back to the
        // daemon's workspace when the session cwd is unset or unavailable.
        let cwd = state
            .cwds
            .lock()
            .map_err(|_| Failure::Cwd)?
            .get(id)
            .ok_or(Failure::Cwd)?
            .current
            .clone();
        if !cwd.is_absolute() {
            return Err(Failure::Cwd);
        }
        cwd.canonicalize().map_err(|_| Failure::Cwd)
    }

    fn new(state: &ServeState, store: &SessionStore, id: &str) -> Result<Self> {
        let cwd = Self::cwd(state, id)?;
        let directory = open_directory(&cwd).map_err(|_| Failure::Cwd)?;
        let history = std::path::absolute(&state.history_file)?;
        let sidecar_dir = history.parent().ok_or(Failure::Cwd)?.to_path_buf();
        let mut scope = Self {
            id: CURRENT_SCOPE.to_string(),
            cwd,
            directory,
            private: Vec::new(),
            sidecar_dir: sidecar_dir.canonicalize().unwrap_or(sidecar_dir),
        };
        scope.protect(store.sessions_root())?;
        scope.protect(&history)?;
        for suffix in ["-journal", "-shm", "-wal"] {
            let mut sibling = history.as_os_str().to_os_string();
            sibling.push(suffix);
            scope.protect(Path::new(&sibling))?;
        }
        scope.protect(&std::env::temp_dir().join(".agent_tmp"))?;
        if let Ok(dir) = super::ctl::state_dir() {
            scope.protect(&dir)?;
        }
        if !scope.allows(&scope.cwd) {
            return Err(Failure::Cwd);
        }
        Ok(scope)
    }

    fn worktree(&self, cwd: PathBuf) -> Result<Self> {
        let directory = open_directory(&cwd).map_err(|_| Failure::UnknownScope)?;
        Ok(Self {
            id: worktree_scope_id(&cwd),
            cwd,
            directory,
            private: self.private.clone(),
            sidecar_dir: self.sidecar_dir.clone(),
        })
    }

    fn allows(&self, path: &Path) -> bool {
        !self.private.iter().any(|p| path.starts_with(p))
            && !path
                .components()
                .any(|part| reserved(part.as_os_str().as_bytes()))
    }

    fn info(&self) -> ScopeInfo {
        ScopeInfo {
            id: self.id.clone(),
            label: self
                .cwd
                .file_name()
                .unwrap_or(OsStr::new("/"))
                .to_string_lossy()
                .into_owned(),
            path: self.cwd.to_string_lossy().into_owned(),
            current: self.id == CURRENT_SCOPE,
        }
    }

    fn protect(&mut self, path: &Path) -> Result<()> {
        let absolute = std::path::absolute(path)?;
        if let Ok(canonical) = absolute.canonicalize()
            && canonical != absolute
        {
            self.private.push(canonical);
        }
        self.private.push(absolute);
        Ok(())
    }

    fn verify(&self, state: &ServeState, store: &SessionStore, id: &str) -> Result<()> {
        if !store.session_exists(id)? {
            return Err(Failure::Session);
        }
        if Self::cwd(state, id)? != self.cwd {
            return Err(Failure::Stale);
        }
        self.verify_directory()
    }

    fn verify_directory(&self) -> Result<()> {
        let opened = self.directory.metadata()?;
        let current = open_directory(&self.cwd)
            .and_then(|dir| dir.metadata())
            .map_err(|_| Failure::Cwd)?;
        if opened.dev() != current.dev() || opened.ino() != current.ino() {
            return Err(Failure::Stale);
        }
        Ok(())
    }

    fn relative(&self, root: &Path, git_path: &[u8]) -> Option<PathBuf> {
        if git_path
            .split(|b| *b == b'/')
            .any(|part| part.is_empty() || part == b"." || part == b".." || reserved(part))
        {
            return None;
        }
        let absolute = root.join(OsStr::from_bytes(git_path));
        let relative = absolute.strip_prefix(&self.cwd).ok()?;
        if relative.as_os_str().is_empty() || self.private.iter().any(|p| absolute.starts_with(p)) {
            return None;
        }
        if absolute.parent() == Some(&self.sidecar_dir)
            && absolute.file_name().is_some_and(|name| {
                let name = name.as_bytes();
                [b"serve-cwds.json".as_slice(), b"serve-groups.json"]
                    .iter()
                    .any(|sidecar| {
                        name.strip_prefix(*sidecar)
                            .is_some_and(|tail| tail.is_empty() || tail.starts_with(b".tmp"))
                    })
            })
        {
            return None;
        }
        Some(relative.to_path_buf())
    }

    fn inspect(&self, path: &Path) -> Result<Current> {
        let mut dir = self.directory.try_clone()?;
        let mut parts = path.components().peekable();
        while let Some(part) = parts.next() {
            let name = CString::new(part.as_os_str().as_bytes()).map_err(|_| Failure::Git)?;
            if parts.peek().is_some() {
                match open_at(&dir, &name, libc::O_DIRECTORY) {
                    Ok(next) => dir = next,
                    Err(error) => return classify_open_error(&error),
                }
                continue;
            }
            // Check types without opening devices, sockets or FIFOs. The
            // subsequent open still refuses symlinks and checks the inode.
            let stat = match stat_at(&dir, &name) {
                Ok(stat) => stat,
                Err(error) => return classify_open_error(&error),
            };
            return match stat.st_mode & libc::S_IFMT {
                libc::S_IFLNK => Ok(Current::Symlink),
                libc::S_IFREG if stat.st_nlink == 1 => {
                    let file = open_at(&dir, &name, 0).map_err(|_| Failure::Stale)?;
                    let meta = file.metadata()?;
                    #[allow(clippy::unnecessary_cast)]
                    if !meta.is_file()
                        || meta.nlink() != 1
                        || meta.dev() != stat.st_dev as u64
                        || meta.ino() != stat.st_ino as u64
                    {
                        return Err(Failure::Stale);
                    }
                    Ok(Current::Regular(Stamp::from(&meta), file))
                }
                _ => Ok(Current::Other),
            };
        }
        Ok(Current::Outside)
    }

    fn contents(&self, entry: &Entry, limit: usize) -> Result<Option<Bounded>> {
        let current = self.inspect(&entry.path)?;
        if current.kind() != entry.kind {
            return Err(Failure::Stale);
        }
        if let Current::Regular(stamp, mut file) = current {
            let mut bytes = Vec::new();
            Read::by_ref(&mut file)
                .take((limit + 1) as u64)
                .read_to_end(&mut bytes)?;
            if Stamp::from(&file.metadata()?) != stamp {
                return Err(Failure::Stale);
            }
            return Ok(Some(Bounded::new(bytes, limit)));
        }
        Ok(None)
    }
}

fn reserved(part: &[u8]) -> bool {
    part.eq_ignore_ascii_case(b".git")
        || part.eq_ignore_ascii_case(b".agent_tmp")
        || part.starts_with(SCRATCH_PREFIX.as_bytes())
}

fn open_directory(path: &Path) -> io::Result<File> {
    let mut dir = File::open("/")?;
    for part in path.components().skip(1) {
        let name = CString::new(part.as_os_str().as_bytes())?;
        dir = open_at(&dir, &name, libc::O_DIRECTORY)?;
    }
    Ok(dir)
}

fn open_at(dir: &File, name: &std::ffi::CStr, flags: libc::c_int) -> io::Result<File> {
    // SAFETY: the borrowed fd and NUL-terminated name stay live for openat.
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK | flags,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful openat transferred sole ownership of fd.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn stat_at(dir: &File, name: &std::ffi::CStr) -> io::Result<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: the directory and name stay live and stat is writable.
    if unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fstatat succeeded and initialized stat.
    Ok(unsafe { stat.assume_init() })
}

fn classify_open_error(error: &io::Error) -> Result<Current> {
    match error.raw_os_error() {
        Some(libc::ENOENT) => Ok(Current::Missing),
        Some(libc::ELOOP | libc::ENOTDIR | libc::EACCES) => Ok(Current::Outside),
        _ => Err(Failure::Git),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp([u64; 8]);

impl From<&Metadata> for Stamp {
    fn from(meta: &Metadata) -> Self {
        Self([
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mode().into(),
            meta.mtime() as u64,
            meta.mtime_nsec() as u64,
            meta.ctime() as u64,
            meta.ctime_nsec() as u64,
        ])
    }
}

enum Current {
    Regular(Stamp, File),
    Missing,
    Symlink,
    Other,
    Outside,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Regular(Stamp),
    Missing,
    Symlink,
    Other,
    Outside,
}

impl Current {
    fn kind(&self) -> Kind {
        match self {
            Self::Regular(stamp, _) => Kind::Regular(*stamp),
            Self::Missing => Kind::Missing,
            Self::Symlink => Kind::Symlink,
            Self::Other => Kind::Other,
            Self::Outside => Kind::Outside,
        }
    }
}

impl Kind {
    fn hash(self, digest: &mut Sha256) {
        let tag = match self {
            Self::Regular(_) => 0,
            Self::Missing => 1,
            Self::Symlink => 2,
            Self::Other => 3,
            Self::Outside => 4,
        };
        digest.update([tag]);
        if let Self::Regular(Stamp(values)) = self {
            for value in values {
                digest.update(value.to_le_bytes());
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Git {
    deadline: Instant,
}

struct GitOutput {
    status: Option<ExitStatus>,
    data: Bounded,
}

impl GitOutput {
    fn code(&self) -> Option<i32> {
        self.status.and_then(|s| s.code())
    }
}

impl Git {
    fn limited(self, limit: Duration) -> Self {
        Self {
            deadline: self.deadline.min(Instant::now() + limit),
        }
    }

    fn command(&self, cwd: &Path) -> Command {
        let mut command = Command::new("git");
        command
            .args(["--no-pager", "--literal-pathspecs", "-C"])
            .arg(cwd);
        for (key, _) in std::env::vars_os() {
            if key.as_bytes().starts_with(b"GIT_") {
                command.env_remove(key);
            }
        }
        command.envs([
            ("GIT_OPTIONAL_LOCKS", "0"),
            ("GIT_PAGER", "cat"),
            ("LC_ALL", "C"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_GLOBAL", "/dev/null"),
            ("GIT_ATTR_NOSYSTEM", "1"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_NO_LAZY_FETCH", "1"),
            ("GIT_ALLOW_PROTOCOL", ""),
        ]);
        command
    }

    fn remaining(&self) -> Result<Duration> {
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            Err(Failure::Timeout)
        } else {
            Ok(remaining)
        }
    }

    fn run(&self, command: &mut Command, limit: usize) -> Result<GitOutput> {
        self.remaining()?;
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = crate::fork_guard::spawn(command)?;
        let output = (|| {
            let data = self.read_stdout(child.stdout.take().ok_or(Failure::Git)?, limit)?;
            let status = if data.capped {
                None
            } else {
                Some(self.wait(&mut child)?)
            };
            Ok(GitOutput { status, data })
        })();
        if !matches!(
            &output,
            Ok(GitOutput {
                status: Some(_),
                ..
            })
        ) {
            let _ = child.kill();
            let _ = child.wait();
        }
        output
    }

    fn read_stdout(&self, mut pipe: ChildStdout, limit: usize) -> Result<Bounded> {
        let mut bytes = Vec::new();
        let mut buffer = [0; 8192];
        let mut descriptor = libc::pollfd {
            fd: pipe.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        while bytes.len() <= limit {
            let timeout = self.remaining()?.as_millis().max(1) as libc::c_int;
            // SAFETY: descriptor is a valid writable array of one pollfd.
            let ready = unsafe { libc::poll(&mut descriptor, 1, timeout) };
            if ready == 0 {
                continue;
            }
            if ready < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Failure::Git);
            }
            let room = buffer.len().min(limit + 1 - bytes.len());
            match pipe.read(&mut buffer[..room]) {
                Ok(0) => break,
                Ok(read) => bytes.extend_from_slice(&buffer[..read]),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Bounded::new(bytes, limit))
    }

    fn wait(&self, child: &mut Child) -> Result<ExitStatus> {
        loop {
            if let Some(status) = child.try_wait()? {
                return Ok(status);
            }
            std::thread::sleep(self.remaining()?.min(Duration::from_millis(2)));
        }
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(prefix: &str) -> io::Result<Self> {
        let path = std::env::temp_dir()
            .canonicalize()?
            .join(format!("{prefix}{}", Uuid::new_v4()));
        DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }

    fn file(&self, name: &str) -> io::Result<File> {
        OpenOptions::new()
            .write(true)
            .mode(0o600)
            .create_new(true)
            .open(self.0.join(name))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct RepoLayout {
    root: PathBuf,
    git_dir: PathBuf,
    common: PathBuf,
    index: PathBuf,
    objects: PathBuf,
    config: &'static [u8],
}

impl RepoLayout {
    fn read(git: Git, scope: &mut Scope) -> Result<Self> {
        let layout = git.run(
            git.command(&scope.cwd).args([
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--absolute-git-dir",
                "--git-common-dir",
                "--git-path",
                "index",
                "--git-path",
                "objects",
                "--show-object-format",
            ]),
            32 * 1024,
        )?;
        if layout.code() != Some(0) {
            return Err(Failure::Cwd);
        }
        let fields: Vec<_> = trimmed_line(&layout.data.bytes)
            .split(|b| *b == b'\n')
            .collect();
        let [root, git_dir, common, index, objects, object_format] = fields.as_slice() else {
            return Err(Failure::Cwd);
        };
        let root = absolute_git_path(root)?
            .canonicalize()
            .map_err(|_| Failure::Cwd)?;
        if !scope.cwd.starts_with(&root) {
            return Err(Failure::Cwd);
        }
        // Git recognizes a symlinked .git directory as another repository.
        // Allow a real directory or a normal worktree gitfile, not that alias.
        let root_dir = open_directory(&root).map_err(|_| Failure::Cwd)?;
        let git_entry = stat_at(&root_dir, c".git").map_err(|_| Failure::Cwd)?;
        if !matches!(
            git_entry.st_mode & libc::S_IFMT,
            libc::S_IFDIR | libc::S_IFREG
        ) {
            return Err(Failure::Cwd);
        }
        let (git_dir, common, index, objects) = (
            absolute_git_path(git_dir)?,
            absolute_git_path(common)?,
            absolute_git_path(index)?,
            absolute_git_path(objects)?,
        );
        open_directory(&objects).map_err(|_| Failure::Cwd)?;
        for private in [&git_dir, &common, &objects] {
            scope.protect(private)?;
            if scope.cwd.starts_with(private) {
                return Err(Failure::Cwd);
            }
        }
        let config = match *object_format {
            b"sha1" => b"[core]\nrepositoryformatversion = 0\n".as_slice(),
            b"sha256" => {
                b"[core]\nrepositoryformatversion = 1\n[extensions]\nobjectformat = sha256\n"
            }
            _ => return Err(Failure::Cwd),
        };
        Ok(Self {
            root,
            git_dir,
            common,
            index,
            objects,
            config,
        })
    }
}

/// `git status` can run repository-configured clean filters. Run it against a
/// private Git directory with a copied index and only non-executable settings.
/// The source repository supplies objects and ignore rules, never commands.
struct Repo {
    git: Git,
    root: PathBuf,
    scratch: Scratch,
    objects: PathBuf,
    head: Vec<u8>,
    settings: Vec<Vec<u8>>,
}

impl Repo {
    fn new(git: Git, scope: &Scope, layout: &RepoLayout) -> Result<Self> {
        let RepoLayout {
            root,
            git_dir,
            common,
            index,
            objects,
            config,
        } = layout;
        let head = git.run(
            git.command(&scope.cwd)
                .args(["rev-parse", "--verify", "--quiet", "HEAD"]),
            256,
        )?;
        let head = match head.code() {
            Some(0) if valid_oid(trimmed_line(&head.data.bytes)) => {
                trimmed_line(&head.data.bytes).to_vec()
            }
            Some(1) if head.data.bytes.is_empty() => Vec::new(),
            _ => return Err(Failure::Git),
        };
        const SAFE_SETTINGS: &str = "^core\\.(filemode|ignorecase|symlinks|autocrlf|eol|checkstat|trustctime|sparsecheckout|sparsecheckoutcone|excludesfile|attributesfile)$";
        let mut query = git.command(&scope.cwd);
        query
            .env_remove("GIT_CONFIG_NOSYSTEM")
            .env_remove("GIT_CONFIG_GLOBAL");
        let settings = git.run(
            query.args(["config", "--null", "--get-regexp", SAFE_SETTINGS]),
            32 * 1024,
        )?;
        if !matches!(settings.code(), Some(0 | 1)) {
            return Err(Failure::Git);
        }
        let settings = settings
            .data
            .bytes
            .split(|b| *b == 0)
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut line = line.to_vec();
                if let Some(separator) = line.iter().position(|b| *b == b'\n') {
                    line[separator] = b'=';
                } else {
                    line.extend_from_slice(b"=true");
                }
                Ok(line)
            })
            .collect::<Result<Vec<_>>>()?;

        let scratch = Scratch::new(SCRATCH_PREFIX)?;
        for dir in ["refs", "info", "objects"] {
            fs::create_dir(scratch.0.join(dir))?;
        }
        scratch.file("config")?.write_all(config)?;
        let mut head_file = scratch.file("HEAD")?;
        head_file.write_all(if head.is_empty() {
            b"ref: refs/heads/unborn"
        } else {
            &head
        })?;
        head_file.write_all(b"\n")?;
        let mut index_budget = INDEX_LIMIT;
        copy_index(index, &scratch, "index", true, &mut index_budget)?;
        // Even rev-parse --shared-index-path freshens the original shared
        // index's timestamp. Locate candidates without letting Git open it.
        for candidate in fs::read_dir(index.parent().ok_or(Failure::Git)?)? {
            git.remaining()?;
            let candidate = candidate?;
            let name = candidate.file_name();
            if let Some(name) = name.to_str()
                && name
                    .strip_prefix("sharedindex.")
                    .is_some_and(|oid| valid_oid(oid.as_bytes()))
            {
                copy_index(&candidate.path(), &scratch, name, false, &mut index_budget)?;
            }
        }
        for (base, info) in [
            (common, "info/exclude"),
            (common, "info/attributes"),
            (git_dir, "info/sparse-checkout"),
        ] {
            symlink(base.join(info), scratch.0.join(info))?;
        }
        Ok(Self {
            git,
            root: root.clone(),
            scratch,
            objects: objects.clone(),
            head,
            settings,
        })
    }

    fn command(&self, cwd: &Path) -> Command {
        let mut command = self.git.command(cwd);
        command
            .env("GIT_DIR", &self.scratch.0)
            .env("GIT_WORK_TREE", &self.root)
            .env("GIT_OBJECT_DIRECTORY", &self.objects);
        for setting in &self.settings {
            command.arg("-c").arg(OsStr::from_bytes(setting));
        }
        command
    }

    fn status(&self, cwd: &Path) -> Result<Bounded> {
        let output = self.git.run(
            self.command(cwd).args([
                "status",
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--renames",
                "--ignore-submodules=all",
                "--",
                ".",
            ]),
            STATUS_LIMIT,
        )?;
        if output.data.capped {
            return Err(Failure::InventoryLimit);
        }
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        Ok(output.data)
    }

    fn blob(&self, oid: &[u8], limit: usize) -> Result<Bounded> {
        if !valid_oid(oid) {
            return Err(Failure::Git);
        }
        let output = self.git.run(
            self.command(&self.root).args([
                OsStr::new("cat-file"),
                OsStr::new("blob"),
                OsStr::from_bytes(oid),
            ]),
            limit,
        )?;
        if !output.data.capped && output.code() != Some(0) {
            return Err(Failure::Git);
        }
        Ok(output.data)
    }

    fn hunks(&self, old: Option<&[u8]>, new: Option<&[u8]>) -> Result<Bounded> {
        let write = |name: &'static str, data: Option<&[u8]>| -> Result<&str> {
            if let Some(data) = data {
                self.scratch.file(name)?.write_all(data)?;
                Ok(name)
            } else {
                Ok("/dev/null")
            }
        };
        let before = write("before", old)?;
        let after = write("after", new)?;
        // Even a parent worktree of TMPDIR must not supply diff settings.
        let mut command = self.git.command(&self.scratch.0);
        command.env("GIT_DIR", self.scratch.0.join("no-git"));
        let output = self.git.run(
            command.args([
                "diff",
                "--no-index",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--",
                before,
                after,
            ]),
            DETAIL_LIMIT,
        )?;
        if !output.data.capped && !matches!(output.code(), Some(0 | 1)) {
            return Err(Failure::Git);
        }
        Ok(output.data)
    }
}

struct WorktreeList {
    main: Option<PathBuf>,
    common: PathBuf,
    data: Bounded,
}

impl WorktreeList {
    fn read(git: Git, current: &Scope, layout: &RepoLayout) -> Result<Self> {
        let output = git.run(
            git.command(&current.cwd)
                .args(["worktree", "list", "--porcelain", "-z"]),
            WORKTREE_LIST_LIMIT,
        )?;
        if !output.data.capped && output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let (main, _) = worktree_records(&output.data.bytes)
            .next()
            .ok_or(Failure::Git)?;
        Ok(Self {
            main: absolute_git_path(main)?.canonicalize().ok(),
            common: layout.common.canonicalize()?,
            data: output.data,
        })
    }

    fn paths<'a>(
        &'a self,
        git: Git,
        current: &'a Scope,
        layout: &'a RepoLayout,
    ) -> impl Iterator<Item = Result<PathBuf>> + 'a {
        let mut seen = FastSet::default();
        worktree_records(&self.data.bytes).filter_map(move |(path, available)| {
            if let Err(error) = git.remaining() {
                return Some(Err(error));
            }
            if !available {
                return None;
            }
            let main = self.main.as_ref()?;
            let path = absolute_git_path(path).ok()?.canonicalize().ok()?;
            // The main checkout bounds discovery but is not a linked scope.
            // The current scope represents its own worktree, even for a session
            // cwd below that worktree's root. Do not widen it to that root.
            (path != *main
                && path.starts_with(main)
                && path != layout.root
                && current.allows(&path)
                && seen.insert(path.clone()))
            .then_some(Ok(path))
        })
    }

    fn open(&self, git: Git, current: &Scope, path: PathBuf) -> Result<(Scope, RepoLayout)> {
        let mut scope = current.worktree(path)?;
        let layout = RepoLayout::read(git, &mut scope).map_err(|error| match error {
            Failure::Cwd => Failure::UnknownScope,
            _ => error,
        })?;
        // A stale registration can now point at a different checkout, or at a
        // directory where Git would fall back to an enclosing worktree.
        if layout.root != scope.cwd
            || layout.common.canonicalize().ok().as_deref() != Some(self.common.as_path())
        {
            return Err(Failure::UnknownScope);
        }
        Ok((scope, layout))
    }

    fn resolve(
        &self,
        git: Git,
        current: &Scope,
        layout: &RepoLayout,
        id: &str,
    ) -> Result<(Scope, RepoLayout)> {
        for path in self.paths(git, current, layout) {
            let path = path?;
            if worktree_scope_id(&path) == id {
                return self.open(git, current, path);
            }
        }
        Err(Failure::UnknownScope)
    }
}

fn worktree_records(bytes: &[u8]) -> impl Iterator<Item = (&[u8], bool)> {
    let mut fields = bytes.split_inclusive(|byte| *byte == 0);
    std::iter::from_fn(move || {
        let path = fields
            .next()?
            .strip_suffix(&[0])?
            .strip_prefix(b"worktree ")?;
        let mut available = true;
        loop {
            let field = fields.next()?.strip_suffix(&[0])?;
            if field.is_empty() {
                return Some((path, available));
            }
            if matches!(field, b"bare" | b"prunable") || field.starts_with(b"prunable ") {
                available = false;
            }
        }
    })
}

fn worktree_scope_id(path: &Path) -> String {
    let hash = Sha256::digest(path.as_os_str().as_bytes());
    let prefix = u64::from_be_bytes(std::array::from_fn(|index| hash[index]));
    format!("wt-{prefix:016x}")
}

fn absolute_git_path(bytes: &[u8]) -> Result<PathBuf> {
    let path = Path::new(OsStr::from_bytes(bytes));
    if !path.is_absolute() {
        return Err(Failure::Cwd);
    }
    Ok(path.to_path_buf())
}

fn trimmed_line(bytes: &[u8]) -> &[u8] {
    bytes.strip_suffix(b"\n").unwrap_or(bytes)
}

fn valid_oid(bytes: &[u8]) -> bool {
    matches!(bytes.len(), 40 | 64) && bytes.iter().all(u8::is_ascii_hexdigit)
}

fn copy_index(
    source: &Path,
    scratch: &Scratch,
    name: &str,
    optional: bool,
    budget: &mut u64,
) -> Result<()> {
    let source = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(source)
    {
        Ok(file) => file,
        Err(error) if optional && error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let meta = source.metadata()?;
    if !meta.is_file() || meta.len() > *budget {
        return Err(Failure::InventoryLimit);
    }
    let mut copy = scratch.file(name)?;
    let copied = io::copy(&mut source.take(*budget + 1), &mut copy)?;
    *budget = budget.checked_sub(copied).ok_or(Failure::InventoryLimit)?;
    Ok(())
}

struct Bounded {
    bytes: Vec<u8>,
    capped: bool,
}

impl Bounded {
    fn new(mut bytes: Vec<u8>, limit: usize) -> Self {
        let capped = bytes.len() > limit;
        bytes.truncate(limit);
        Self { bytes, capped }
    }

    fn text(&self) -> Option<&str> {
        if self.bytes.contains(&0) {
            return None;
        }
        match std::str::from_utf8(&self.bytes) {
            Ok(text) => Some(text),
            Err(error) if self.capped && error.error_len().is_none() => {
                std::str::from_utf8(&self.bytes[..error.valid_up_to()]).ok()
            }
            Err(_) => None,
        }
    }
}

struct Entry {
    path: PathBuf,
    old_path: Option<PathBuf>,
    display: String,
    before: Option<Vec<u8>>,
    old_mode: Vec<u8>,
    kind: Kind,
    tracked: bool,
    status: &'static str,
}

impl Entry {
    fn unavailable(&self) -> Option<&'static str> {
        if self.kind == Kind::Symlink || self.old_mode == b"120000" {
            Some("symlink")
        } else if self.kind == Kind::Other || self.old_mode == b"160000" {
            Some("not a regular file")
        } else {
            None
        }
    }
}

#[derive(Serialize)]
struct FileRow {
    id: String,
    path: String,
    old_path: Option<String>,
    status: &'static str,
    additions: Option<usize>,
    deletions: Option<usize>,
    binary: bool,
}

#[derive(Serialize)]
struct ScopeRow {
    #[serde(flatten)]
    scope: ScopeInfo,
    changed: bool,
    files: usize,
}

impl ScopeRow {
    fn read(git: Git, scope: &Scope, layout: &RepoLayout, session_id: &str) -> Result<Self> {
        // Count the eligible inventory, before the list's display limits or
        // blob reads. In particular, status-only nested directories do not count.
        let files = Snapshot::read(scope, layout, session_id, git)?
            .entries
            .len();
        git.remaining()?;
        Ok(Self {
            scope: scope.info(),
            changed: files != 0,
            files,
        })
    }
}

#[derive(Serialize)]
struct ScopesReply {
    source: &'static str,
    scopes: Vec<ScopeRow>,
    truncated: bool,
    reason: Option<&'static str>,
}

impl ScopesReply {
    fn truncate(&mut self, reason: &'static str) {
        self.truncated = true;
        self.reason.get_or_insert(reason);
    }
}

fn list_scopes(git: Git, current: &mut Scope, session_id: &str) -> Result<Json<ScopesReply>> {
    // Leave time to verify the session and return partial results before the
    // outer request deadline fires.
    let git = Git {
        deadline: git.deadline - SCAN_RESPONSE_MARGIN,
    };
    let layout = RepoLayout::read(git, current)?;
    let mut reply = ScopesReply {
        source: "git",
        scopes: vec![ScopeRow::read(git, current, &layout, session_id)?],
        truncated: false,
        reason: None,
    };
    let worktrees = match WorktreeList::read(git.limited(WORKTREE_TIME_LIMIT), current, &layout) {
        Ok(worktrees) => worktrees,
        Err(Failure::Timeout) => {
            reply.truncate(SCOPES_TIMEOUT);
            return Ok(Json(reply));
        }
        Err(error) => return Err(error),
    };
    if worktrees.data.capped {
        reply.truncate(SCOPES_TRUNCATED);
    }
    for path in worktrees.paths(git, current, &layout) {
        let Ok(path) = path else {
            reply.truncate(SCOPES_TIMEOUT);
            break;
        };
        let per_tree = git.limited(WORKTREE_TIME_LIMIT);
        let row = (|| {
            let (scope, layout) = worktrees.open(per_tree, current, path)?;
            let row = ScopeRow::read(per_tree, &scope, &layout, session_id)?;
            scope.verify_directory()?;
            Ok(row)
        })();
        match row {
            Ok(_) if reply.scopes.len() == WORKTREE_LIMIT => {
                reply.truncate(SCOPES_TRUNCATED);
                break;
            }
            Ok(row) => reply.scopes.push(row),
            Err(Failure::Timeout) => reply.truncate(SCOPES_TIMEOUT),
            Err(Failure::InventoryLimit) => {
                reply.truncate("worktree inventory exceeds the collection limit");
            }
            Err(_) => continue,
        }
    }
    Ok(Json(reply))
}

fn read_changes(
    git: Git,
    current: &mut Scope,
    session_id: &str,
    scope_id: Option<String>,
    detail: Option<DetailRequest>,
) -> Result<Response> {
    let missing_revision = detail
        .as_ref()
        .is_some_and(|detail| detail.revision.is_none());
    if scope_id.is_none() && missing_revision {
        return Err(Failure::Stale);
    }
    let layout = RepoLayout::read(git, current)?;
    if let Some(id) = scope_id {
        let worktrees = WorktreeList::read(git, current, &layout)?;
        let (scope, layout) = worktrees.resolve(git, current, &layout, &id)?;
        if missing_revision {
            return Err(Failure::Stale);
        }
        let response = Snapshot::read(&scope, &layout, session_id, git)?.respond(detail)?;
        scope.verify_directory()?;
        Ok(response)
    } else {
        Snapshot::read(current, &layout, session_id, git)?.respond(detail)
    }
}

#[derive(Serialize)]
struct ListReply {
    source: &'static str,
    root_label: String,
    scope: ScopeInfo,
    revision: String,
    files: Vec<FileRow>,
    truncated: bool,
    reason: Option<&'static str>,
}

#[derive(Serialize)]
struct DetailReply {
    revision: String,
    file_id: String,
    path: String,
    status: &'static str,
    format: &'static str,
    diff: String,
    truncated: bool,
    reason: Option<&'static str>,
}

struct Snapshot<'a> {
    scope: &'a Scope,
    repo: Repo,
    revision: String,
    entries: Vec<Entry>,
}

impl<'a> Snapshot<'a> {
    fn read(scope: &'a Scope, layout: &RepoLayout, session_id: &str, git: Git) -> Result<Self> {
        let repo = Repo::new(git, scope, layout)?;
        let status = repo.status(&scope.cwd)?;
        let mut digest = Sha256::new();
        digest.update(REVISION_KEY.as_bytes());
        hash_piece(&mut digest, session_id.as_bytes());
        hash_piece(&mut digest, scope.id.as_bytes());
        hash_piece(&mut digest, scope.cwd.as_os_str().as_bytes());
        hash_piece(&mut digest, &repo.head);
        let directory = scope.directory.metadata()?;
        digest.update(directory.dev().to_le_bytes());
        digest.update(directory.ino().to_le_bytes());
        let mut entries = Vec::new();
        let mut records = status.bytes.split_inclusive(|b| *b == 0);
        while let Some(record) = records.next() {
            let record = record.strip_suffix(&[0]).ok_or(Failure::Git)?;
            repo.git.remaining()?;
            let (path, old, xy, old_mode, oid, tracked) = match record.first() {
                Some(b'1') => {
                    let parts: Vec<_> = record.splitn(9, |b| *b == b' ').collect();
                    let [_, xy, _, mode, _, _, oid, _, path] = parts.as_slice() else {
                        return Err(Failure::Git);
                    };
                    (*path, None, *xy, *mode, *oid, true)
                }
                Some(b'2') => {
                    let parts: Vec<_> = record.splitn(10, |b| *b == b' ').collect();
                    let [_, xy, _, mode, _, _, oid, _, _, path] = parts.as_slice() else {
                        return Err(Failure::Git);
                    };
                    (
                        *path,
                        Some(
                            records
                                .next()
                                .and_then(|old| old.strip_suffix(&[0]))
                                .ok_or(Failure::Git)?,
                        ),
                        *xy,
                        *mode,
                        *oid,
                        true,
                    )
                }
                Some(b'u') => {
                    let parts: Vec<_> = record.splitn(11, |b| *b == b' ').collect();
                    let [_, _, _, _, _, _, _, _, _, _, path] = parts.as_slice() else {
                        return Err(Failure::Git);
                    };
                    (
                        *path,
                        None,
                        b"UU".as_slice(),
                        b"".as_slice(),
                        b"".as_slice(),
                        true,
                    )
                }
                Some(b'?') if record.starts_with(b"? ") => (
                    &record[2..],
                    None,
                    b"??".as_slice(),
                    b"".as_slice(),
                    b"".as_slice(),
                    false,
                ),
                _ => return Err(Failure::Git),
            };
            let Some(path) = scope.relative(&repo.root, path) else {
                continue;
            };
            let kind = scope.inspect(&path)?.kind();
            if kind == Kind::Outside || !tracked && kind == Kind::Missing {
                continue;
            }
            let old_path = match old.and_then(|path| scope.relative(&repo.root, path)) {
                Some(old_path)
                    if matches!(
                        scope.inspect(&old_path)?.kind(),
                        Kind::Regular(_) | Kind::Missing
                    ) =>
                {
                    Some(old_path)
                }
                _ => None,
            };
            let safe_baseline = old.is_none() || old_path.is_some();
            let before = (safe_baseline && valid_oid(oid) && oid.iter().any(|b| *b != b'0'))
                .then(|| oid.to_vec());
            let status = if xy == b"UU" {
                "unknown"
            } else if old.is_some() && xy.contains(&b'R') {
                "renamed"
            } else if kind == Kind::Missing && xy.contains(&b'D') {
                "deleted"
            } else if !tracked || xy.contains(&b'A') || xy.contains(&b'C') {
                "added"
            } else {
                "modified"
            };
            hash_piece(&mut digest, record);
            hash_piece(
                &mut digest,
                old_path
                    .as_deref()
                    .map_or(b"".as_slice(), |p| p.as_os_str().as_bytes()),
            );
            kind.hash(&mut digest);
            entries.push(Entry {
                display: path.to_string_lossy().into_owned(),
                path,
                old_path,
                before,
                old_mode: if safe_baseline {
                    old_mode.to_vec()
                } else {
                    Vec::new()
                },
                kind,
                tracked,
                status,
            });
        }
        entries.sort_by(|a, b| {
            a.display
                .cmp(&b.display)
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| b.tracked.cmp(&a.tracked))
        });
        entries.dedup_by(|a, b| a.path == b.path);
        Ok(Self {
            scope,
            repo,
            revision: URL_SAFE_NO_PAD.encode(digest.finalize()),
            entries,
        })
    }

    fn row(&self, entry: &Entry) -> Result<FileRow> {
        let mut binary = false;
        let mut additions = None;
        if entry.unavailable().is_none() {
            let current = self.scope.contents(entry, PROBE_LIMIT)?;
            binary = current.as_ref().is_some_and(|file| file.text().is_none());
            if !binary && let Some(oid) = &entry.before {
                binary = self.repo.blob(oid, PROBE_LIMIT)?.text().is_none();
            }
            if !entry.tracked
                && let Some(content) = current
                    .as_ref()
                    .filter(|c| !c.capped)
                    .and_then(Bounded::text)
            {
                additions = Some(content.lines().count());
            }
        }
        let mut hash = Sha256::new();
        hash_piece(&mut hash, self.revision.as_bytes());
        hash_piece(&mut hash, entry.path.as_os_str().as_bytes());
        Ok(FileRow {
            id: URL_SAFE_NO_PAD.encode(hash.finalize()),
            path: entry.display.clone(),
            old_path: entry
                .old_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            status: entry.status,
            additions,
            deletions: additions.map(|_| 0),
            binary,
        })
    }

    fn respond(&self, detail: Option<DetailRequest>) -> Result<Response> {
        match detail {
            Some(request) => Ok(self.detail(request)?.into_response()),
            None => Ok(Json(self.list()?.0).into_response()),
        }
    }

    fn list(&self) -> Result<(ListReply, Vec<&Entry>)> {
        let scope = self.scope.info();
        let mut list = ListReply {
            source: "working_tree",
            root_label: scope.label.clone(),
            scope,
            revision: self.revision.clone(),
            files: Vec::new(),
            truncated: true,
            reason: Some(LIST_TRUNCATED),
        };
        let mut size = wire_size(&list);
        let mut entries = Vec::new();
        for entry in self.entries.iter().take(FILE_LIMIT) {
            let row = self.row(entry)?;
            size += wire_size(&row) + usize::from(!entries.is_empty());
            if size > LIST_LIMIT {
                break;
            }
            entries.push(entry);
            list.files.push(row);
        }
        list.truncated = entries.len() < self.entries.len();
        list.reason = list.truncated.then_some(LIST_TRUNCATED);
        Ok((list, entries))
    }

    fn detail(&self, request: DetailRequest) -> Result<Json<DetailReply>> {
        if request.revision.as_deref() != Some(&self.revision) {
            return Err(Failure::Stale);
        }
        let (list, entries) = self.list()?;
        let Some((row, entry)) = list
            .files
            .into_iter()
            .zip(entries)
            .find(|(row, _)| row.id == request.file_id)
        else {
            return Err(Failure::File);
        };
        let mut reply = DetailReply {
            revision: self.revision.clone(),
            file_id: row.id,
            path: row.path,
            status: row.status,
            format: "none",
            diff: String::new(),
            truncated: false,
            reason: None,
        };
        if let Some(reason) = entry.unavailable() {
            reply.reason = Some(reason);
        } else if row.binary {
            reply.reason = Some("binary");
        } else if entry.status == "unknown" {
            reply.reason = Some("unmerged file");
        } else {
            let current = self.scope.contents(entry, DIFF_INPUT_LIMIT)?;
            let before = entry
                .before
                .as_ref()
                .map(|oid| self.repo.blob(oid, DIFF_INPUT_LIMIT))
                .transpose()?;
            if current.as_ref().is_some_and(|file| file.text().is_none())
                || before.as_ref().is_some_and(|file| file.text().is_none())
            {
                reply.reason = Some("binary");
            } else if [before.as_ref(), current.as_ref()]
                .into_iter()
                .flatten()
                .any(|file| file.capped)
            {
                reply.truncated = true;
                reply.reason = Some("file exceeds diff input limit");
            } else if entry.tracked {
                let before = before.as_ref().map(|b| b.bytes.as_slice());
                let current = current.as_ref().map(|b| b.bytes.as_slice());
                let mode_changed = matches!(entry.kind, Kind::Regular(Stamp(values)) if (values[3] & 0o111 != 0) != (entry.old_mode == b"100755"));
                let input = if before == current {
                    None
                } else {
                    Some(self.repo.hunks(before, current)?)
                };
                let has_headers = before != current || entry.old_path.is_some() || mode_changed;
                if has_headers {
                    let mut patch = patch_header(entry, before.is_some(), current.is_some());
                    let clipped = input.as_ref().is_some_and(|input| input.capped);
                    if let Some(input) = input {
                        let text = input.text().ok_or(Failure::Git)?;
                        let text = complete_lines(text, clipped);
                        if let Some(index) = text
                            .split_inclusive('\n')
                            .scan(0, |offset, line| {
                                let start = *offset;
                                *offset += line.len();
                                Some((start, line))
                            })
                            .find_map(|(start, line)| line.starts_with("@@ ").then_some(start))
                        {
                            patch.push_str(&text[index..]);
                        }
                    } else if mode_changed && before.is_some() && current.is_some() {
                        let new_mode = if entry.old_mode == b"100755" {
                            "100644"
                        } else {
                            "100755"
                        };
                        patch.push_str(&format!(
                            "old mode {}\nnew mode {new_mode}\n",
                            String::from_utf8_lossy(&entry.old_mode)
                        ));
                    }
                    reply.set_patch(&patch, clipped);
                } else {
                    reply.reason = Some("no content changes");
                }
            } else if let Some(current) = current {
                let text = complete_lines(current.text().ok_or(Failure::Git)?, current.capped);
                let mut patch = patch_header(entry, false, true);
                let lines = text.lines().count();
                if lines > 0 {
                    patch.push_str(&format!("@@ -0,0 +1,{lines} @@\n"));
                    for line in text.split_inclusive('\n').take(DIFF_LINES) {
                        patch.push('+');
                        patch.push_str(line);
                    }
                    if !text.ends_with('\n') {
                        patch.push_str("\n\\ No newline at end of file\n");
                    }
                }
                reply.set_patch(&patch, current.capped);
            } else {
                return Err(Failure::Stale);
            }
        }
        Ok(Json(reply))
    }
}

fn hash_piece(hash: &mut Sha256, piece: &[u8]) {
    hash.update((piece.len() as u64).to_le_bytes());
    hash.update(piece);
}

fn wire_size(value: &impl Serialize) -> usize {
    serde_json::to_vec(value)
        .expect("changes wire fields always serialize")
        .len()
}

fn complete_lines(text: &str, capped: bool) -> &str {
    if capped {
        &text[..text.rfind('\n').map_or(0, |i| i + 1)]
    } else {
        text
    }
}

fn quote_path(prefix: &str, path: &Path) -> String {
    let bytes = path.as_os_str().as_bytes();
    if bytes
        .iter()
        .all(|b| matches!(*b, b' '..=b'~') && !matches!(*b, b'"' | b'\\'))
    {
        return format!("{prefix}{}", path.display());
    }
    let mut quoted = format!("\"{prefix}");
    for byte in bytes {
        match *byte {
            b'"' | b'\\' => {
                quoted.push('\\');
                quoted.push(*byte as char);
            }
            b' '..=b'~' => quoted.push(*byte as char),
            _ => quoted.push_str(&format!("\\{byte:03o}")),
        }
    }
    quoted.push('"');
    quoted
}

fn patch_header(entry: &Entry, before: bool, after: bool) -> String {
    let old_path = entry.old_path.as_deref().unwrap_or(&entry.path);
    let a = quote_path("a/", old_path);
    let b = quote_path("b/", &entry.path);
    let mut patch = format!("diff --git {a} {b}\n");
    if let Some(old_path) = &entry.old_path {
        patch.push_str(&format!(
            "rename from {}\nrename to {}\n",
            quote_path("", old_path),
            quote_path("", &entry.path)
        ));
    }
    let old = if before { &a } else { "/dev/null" };
    let new = if after { &b } else { "/dev/null" };
    patch.push_str(&format!("--- {old}\n+++ {new}\n"));
    patch
}

impl DetailReply {
    fn set_patch(&mut self, patch: &str, clipped: bool) {
        self.format = "unified";
        self.truncated = true;
        self.reason = Some(DIFF_TRUNCATED);
        let mut size = wire_size(self);
        let mut truncated = clipped;
        for (index, line) in patch.split_inclusive('\n').enumerate() {
            size += wire_size(&line) - 2;
            if index >= DIFF_LINES || size > DETAIL_LIMIT {
                truncated = true;
                break;
            }
            self.diff.push_str(line);
        }
        self.truncated = truncated;
        self.reason = truncated.then_some(DIFF_TRUNCATED);
    }
}

#[cfg(test)]
#[path = "changes_tests.rs"]
mod tests;
