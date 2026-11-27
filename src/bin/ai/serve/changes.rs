//! Git changes, index commits, and plain file inventories in eligible serve scopes.

use std::{
    collections::BinaryHeap,
    ffi::{CStr, CString, OsStr},
    fs::{self, DirBuilder, File, Metadata, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, symlink},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::LazyLock,
    time::{Duration, Instant, SystemTime},
};

use axum::{
    Json, Router,
    extract::{
        Path as RoutePath, Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rust_tools::commonw::{FastMap, FastSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{ServeState, SessionStore, check_auth};

const FILE_LIMIT: usize = 200;
const LIST_LIMIT: usize = 64 * 1024;
const PLAIN_SCAN_LIMIT: usize = 16 * 1024;
const PLAIN_DEPTH_LIMIT: usize = 64;
const DIFF_LINES: usize = 1000;
const DETAIL_LIMIT: usize = 128 * 1024;
const DIFF_INPUT_LIMIT: usize = 2 * 1024 * 1024;
const STATUS_LIMIT: usize = 4 * 1024 * 1024;
const COMMIT_INVENTORY_LIMIT: usize = 4 * 1024 * 1024;
const COMMIT_SCAN_LIMIT: usize = 4 * 1024 * 1024;
const COMMIT_PAGE_DEFAULT: usize = 25;
const COMMIT_PAGE_LIMIT: usize = 100;
const COMMIT_SUBJECT_LIMIT: usize = 512;
const ALTERNATE_LIMIT: usize = 32;
const ALTERNATES_FILE_LIMIT: usize = 64 * 1024;
const INDEX_LIMIT: u64 = 32 * 1024 * 1024;
const PROBE_LIMIT: usize = 8192;
const UNTRACKED_STAT_LIMIT: usize = 4 * 1024 * 1024;
const TIME_LIMIT: Duration = Duration::from_secs(5);
const COMMIT_TIME_LIMIT: Duration = Duration::from_secs(120);
const COMMIT_STDERR_LIMIT: usize = 4096;
const WORKTREE_LIMIT: usize = 32;
const WORKTREE_LIST_LIMIT: usize = 1024 * 1024;
const WORKTREE_TIME_LIMIT: Duration = Duration::from_millis(500);
const WORKTREE_RECENCY_TIME_LIMIT: Duration = Duration::from_millis(50);
const RENAME_SCAN_TIME_LIMIT: Duration = Duration::from_millis(250);
const SCAN_RESPONSE_MARGIN: Duration = Duration::from_millis(250);
const CURRENT_SCOPE: &str = "current";
const SCRATCH_PREFIX: &str = ".a-serve-changes-";
const LIST_TRUNCATED: &str = "file list limit reached";
const PLAIN_INCOMPLETE: &str = "file inventory could not be read completely";
const DIFF_TRUNCATED: &str = "diff limit reached";
const SCOPES_TRUNCATED: &str = "worktree scope limit reached";
const SCOPES_TIMEOUT: &str = "worktree scan timed out";
const SCOPES_FAILED: &str = "worktree discovery failed";
static CAPACITY: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
static COMMIT_CAPACITY: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
static REVISION_KEY: LazyLock<Uuid> = LazyLock::new(Uuid::new_v4);

pub(super) fn routes(app: Router<ServeState>) -> Router<ServeState> {
    app.route("/sessions/{id}/changes", get(list_changes))
        .route("/sessions/{id}/changes/scopes", get(changes_scopes))
        .route("/sessions/{id}/changes/{file_id}", get(file_change))
        .route("/sessions/{id}/changes/commits", get(list_commits))
        .route(
            "/sessions/{id}/changes/commits/{commit_id}",
            get(commit_change),
        )
        .route("/sessions/{id}/commit", post(create_commit))
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

#[derive(Deserialize)]
struct SideQuery {
    side: Option<DiffSide>,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DiffSide {
    Worktree,
    Staged,
}

struct DetailRequest {
    file_id: String,
    revision: Option<String>,
    side: Result<Option<DiffSide>>,
}

#[derive(Deserialize)]
struct CreateCommit {
    message: String,
    scope: Option<String>,
}

enum ChangesRequest {
    Read(ReadChanges),
    CreateCommit(Result<CreateCommit>),
}

enum ReadChanges {
    Scopes,
    Snapshot {
        scope_id: Result<Option<String>>,
        detail: Option<DetailRequest>,
    },
    Commits(Result<(Option<String>, CommitRequest)>),
}

enum CommitRequest {
    History {
        limit: usize,
        before: Option<String>,
    },
    Detail {
        commit_id: String,
    },
}

type CommitQuery = ParsedQuery<Vec<(String, String)>>;

fn commit_selection(
    query: CommitQuery,
    commit_id: Option<String>,
) -> Result<(Option<String>, CommitRequest)> {
    let Query(query) = query.map_err(|_| Failure::CommitQuery)?;
    let mut scope = None;
    let mut limit = None;
    let mut before = None;
    for (key, value) in query {
        match key.as_str() {
            "scope" if scope.is_none() => scope = Some(value),
            "scope" => return Err(Failure::UnknownScope),
            "before" if commit_id.is_none() && before.is_none() => before = Some(value),
            "limit" if commit_id.is_none() && limit.is_none() => {
                limit = value
                    .parse::<usize>()
                    .ok()
                    .filter(|n| (1..=COMMIT_PAGE_LIMIT).contains(n))
                    .filter(|_| value.bytes().all(|b| b.is_ascii_digit()));
                if limit.is_none() {
                    return Err(Failure::CommitQuery);
                }
            }
            _ => return Err(Failure::CommitQuery),
        }
    }
    let request = match commit_id {
        Some(commit_id) => CommitRequest::Detail { commit_id },
        None => CommitRequest::History {
            limit: limit.unwrap_or(COMMIT_PAGE_DEFAULT),
            before,
        },
    };
    Ok((parse_scope(scope)?, request))
}

fn scope_selection(query: ParsedQuery<ScopeQuery>) -> Result<Option<String>> {
    let Query(query) = query.map_err(|_| Failure::UnknownScope)?;
    parse_scope(query.scope)
}

fn parse_scope(scope: Option<String>) -> Result<Option<String>> {
    match scope {
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
        ChangesRequest::Read(ReadChanges::Snapshot {
            scope_id: scope_selection(query),
            detail: None,
        }),
    )
    .await
}

async fn changes_scopes(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath(id): RoutePath<String>,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Read(ReadChanges::Scopes),
    )
    .await
}

async fn file_change(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath((id, file_id)): RoutePath<(String, String)>,
    scope: ParsedQuery<ScopeQuery>,
    revision: ParsedQuery<RevisionQuery>,
    side: ParsedQuery<SideQuery>,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Read(ReadChanges::Snapshot {
            scope_id: scope_selection(scope),
            detail: Some(DetailRequest {
                file_id,
                revision: revision.ok().and_then(|Query(query)| query.revision),
                side: side
                    .map(|Query(query)| query.side)
                    .map_err(|_| Failure::UnknownSide),
            }),
        }),
    )
    .await
}

async fn list_commits(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath(id): RoutePath<String>,
    query: CommitQuery,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Read(ReadChanges::Commits(commit_selection(query, None))),
    )
    .await
}

async fn commit_change(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath((id, commit_id)): RoutePath<(String, String)>,
    query: CommitQuery,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::Read(ReadChanges::Commits(commit_selection(
            query,
            Some(commit_id),
        ))),
    )
    .await
}

async fn create_commit(
    State(state): State<ServeState>,
    headers: HeaderMap,
    RoutePath(id): RoutePath<String>,
    body: std::result::Result<Json<CreateCommit>, JsonRejection>,
) -> Response {
    changes(
        state,
        headers,
        id,
        ChangesRequest::CreateCommit(
            body.map(|Json(body)| body)
                .map_err(|_| Failure::CommitInput),
        ),
    )
    .await
}

async fn changes(
    state: ServeState,
    headers: HeaderMap,
    id: String,
    request: ChangesRequest,
) -> Response {
    let writing = matches!(request, ChangesRequest::CreateCommit(_));
    if let Err((status, Json(body))) = check_auth(&state, &headers) {
        return if writing {
            error_response(
                status,
                Some("unauthorized"),
                body["error"].as_str().unwrap_or("authentication failed"),
                None,
                true,
            )
        } else {
            (status, Json(body)).into_response()
        };
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return Failure::InvalidSessionId.response(writing);
    }
    let Ok(permit) = CAPACITY.try_acquire() else {
        return Failure::Busy.response(writing);
    };
    let deadline = Instant::now()
        + if writing {
            COMMIT_TIME_LIMIT
        } else {
            TIME_LIMIT
        };
    let job = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let store = SessionStore::new(&state.history_file);
        if !store.session_exists(&id)? {
            return Err(Failure::Session);
        }
        let git = Git { deadline };
        match request {
            ChangesRequest::Read(request) => request.respond(git, &state, &store, &id),
            ChangesRequest::CreateCommit(input) => write_commit(git, &state, &store, &id, input?),
        }
    });
    // Let the bounded Git runner reap the commit before reporting a timeout.
    let response_deadline = if writing {
        deadline + SCAN_RESPONSE_MARGIN
    } else {
        deadline
    };
    let mut response = match tokio::time::timeout_at(response_deadline.into(), job).await {
        Ok(Ok(Ok(response))) => response,
        Ok(Ok(Err(error))) => error.response(writing),
        Ok(Err(_)) | Err(_) if writing => Failure::CommitOutcomeUnknown.response(writing),
        Ok(Err(_)) => Failure::Git.response(writing),
        Err(_) => Failure::Timeout.response(writing),
    };
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    response
}

impl ReadChanges {
    fn respond(
        self,
        git: Git,
        state: &ServeState,
        store: &SessionStore,
        id: &str,
    ) -> Result<Response> {
        let mut current = Scope::new(state, store, id, CwdPolicy::Effective)?;
        let response = match self {
            Self::Scopes => list_scopes(git, &mut current, id),
            Self::Snapshot { scope_id, detail } => {
                read_changes(git, &mut current, id, scope_id?, detail)
            }
            Self::Commits(request) => {
                let (scope_id, request) = request?;
                read_commit_browser(git, &mut current, scope_id.as_deref(), request)
            }
        }?;
        current.verify(state, store, id)?;
        Ok(response)
    }
}

const REPOSITORY_UNAVAILABLE: &str = "Git repository object store is unavailable";

#[derive(Debug)]
enum Unavailable {
    Git,
    Worktree,
}

impl Unavailable {
    fn reason(&self) -> &'static str {
        match self {
            Self::Git => "git_unavailable",
            Self::Worktree => "not_a_repository",
        }
    }
}

#[derive(Debug)]
enum Failure {
    InvalidSessionId,
    Session,
    File,
    Cwd,
    UnknownScope,
    NotGitRepository,
    Stale,
    UnknownSide,
    DiffSideUnavailable,
    CommitInput,
    CommitMessage,
    NothingStaged,
    UnmergedConflicts,
    CommitBusy,
    CommitFailed(String),
    CommitOutcomeUnknown,
    CommitSummaryUnavailable,
    CommitQuery,
    CommitId,
    CommitNotFound,
    Unavailable(Unavailable),
    RepositoryUnavailable,
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

impl Failure {
    fn response(self, writing: bool) -> Response {
        let coded = |status, code, message: &str| {
            error_response(status, Some(code), message, None, writing)
        };
        let (status, code, message) = match self {
            Self::InvalidSessionId => (
                StatusCode::BAD_REQUEST,
                "invalid_session_id",
                "invalid session id",
            ),
            Self::Session => (
                StatusCode::NOT_FOUND,
                "session_not_found",
                "session not found",
            ),
            Self::File => (
                StatusCode::NOT_FOUND,
                "change_not_found",
                "change not found",
            ),
            Self::Cwd => (
                StatusCode::CONFLICT,
                "cwd_unresolved",
                if writing {
                    "session working directory could not be resolved; use /cd to set it"
                } else {
                    "session cwd is unavailable or not a Git working tree"
                },
            ),
            Self::UnknownScope => {
                return coded(
                    StatusCode::BAD_REQUEST,
                    "unknown_scope",
                    "unknown changes scope",
                );
            }
            Self::NotGitRepository => {
                return coded(
                    StatusCode::CONFLICT,
                    "not_a_git_repository",
                    "not a Git repository",
                );
            }
            Self::Stale => (
                StatusCode::CONFLICT,
                "changes_stale",
                "changes revision is missing or stale",
            ),
            Self::UnknownSide => {
                return coded(StatusCode::BAD_REQUEST, "unknown_side", "unknown diff side");
            }
            Self::DiffSideUnavailable => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    Some("diff_side_unavailable"),
                    "no changes on the requested diff side",
                    None,
                    true,
                );
            }
            Self::CommitInput => (
                StatusCode::BAD_REQUEST,
                "invalid_commit_request",
                "expected a JSON commit message and optional scope",
            ),
            Self::CommitMessage => (
                StatusCode::BAD_REQUEST,
                "invalid_commit_message",
                "commit message must not be empty or contain NUL",
            ),
            Self::NothingStaged => (
                StatusCode::CONFLICT,
                "nothing_staged",
                "there are no staged changes",
            ),
            Self::UnmergedConflicts => (
                StatusCode::CONFLICT,
                "unmerged_conflicts",
                "resolve merge conflicts before committing",
            ),
            Self::CommitBusy => (
                StatusCode::CONFLICT,
                "commit_busy",
                "a commit is already in progress",
            ),
            Self::CommitFailed(stderr) => {
                return coded(StatusCode::CONFLICT, "commit_failed", &stderr);
            }
            Self::CommitOutcomeUnknown => (
                StatusCode::CONFLICT,
                "commit_outcome_unknown",
                "Git did not report a commit result; check HEAD before retrying",
            ),
            Self::CommitSummaryUnavailable => (
                StatusCode::CONFLICT,
                "commit_summary_unavailable",
                "Git committed successfully but the summary could not be read; check HEAD before retrying",
            ),
            Self::CommitQuery => (
                StatusCode::BAD_REQUEST,
                "invalid_commit_query",
                "invalid commit query",
            ),
            Self::CommitId => (
                StatusCode::BAD_REQUEST,
                "invalid_commit_id",
                "invalid commit object ID",
            ),
            Self::CommitNotFound => (
                StatusCode::NOT_FOUND,
                "commit_not_found",
                "commit not found",
            ),
            Self::Unavailable(unavailable) => {
                let message = match unavailable {
                    Unavailable::Git => "Git is unavailable",
                    Unavailable::Worktree => "session cwd is unavailable or not a Git working tree",
                };
                return error_response(
                    StatusCode::CONFLICT,
                    Some("git_unavailable"),
                    message,
                    Some(unavailable.reason()),
                    writing,
                );
            }
            Self::RepositoryUnavailable => {
                return coded(
                    StatusCode::CONFLICT,
                    "repository_unavailable",
                    REPOSITORY_UNAVAILABLE,
                );
            }
            Self::Git => (
                StatusCode::SERVICE_UNAVAILABLE,
                "changes_unavailable",
                "Git changes could not be read",
            ),
            Self::Busy => (
                StatusCode::SERVICE_UNAVAILABLE,
                "changes_busy",
                "changes service is busy",
            ),
            Self::Timeout => (
                StatusCode::SERVICE_UNAVAILABLE,
                "changes_timeout",
                "changes request timed out",
            ),
            Self::InventoryLimit => (
                StatusCode::SERVICE_UNAVAILABLE,
                "changes_inventory_limit",
                "Git inventory exceeds the collection limit",
            ),
        };
        error_response(status, writing.then_some(code), message, None, writing)
    }
}

fn error_response(
    status: StatusCode,
    code: Option<&str>,
    message: &str,
    reason: Option<&str>,
    include_message: bool,
) -> Response {
    let mut body = serde_json::json!({"error": message});
    if let Some(code) = code {
        body["code"] = code.into();
    }
    if let Some(reason) = reason {
        body["reason"] = reason.into();
    }
    if include_message {
        body["message"] = message.into();
    }
    (status, Json(body)).into_response()
}

#[derive(Clone, Copy)]
enum CwdPolicy {
    Effective,
    Recorded,
}

struct Scope {
    id: String,
    resolved_cwd: PathBuf,
    cwd: PathBuf,
    cwd_policy: CwdPolicy,
    session_root: PathBuf,
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
    fn cwd(state: &ServeState, id: &str, policy: CwdPolicy) -> Result<(PathBuf, PathBuf)> {
        // Reads follow `/pwd`. Writes use the same recorded path but cannot
        // fall back to the daemon's directory or use it to resolve a relative path.
        let resolved = match policy {
            CwdPolicy::Effective => super::session_cwd(state, id),
            CwdPolicy::Recorded => state
                .cwds
                .lock()
                .map_err(|_| Failure::Cwd)?
                .get(id)
                .map(|entry| entry.current.clone())
                .filter(|path| path.is_absolute())
                .ok_or(Failure::Cwd)?,
        };
        let canonical = resolved.canonicalize().map_err(|_| Failure::Cwd)?;
        Ok((resolved, canonical))
    }

    fn new(
        state: &ServeState,
        store: &SessionStore,
        id: &str,
        cwd_policy: CwdPolicy,
    ) -> Result<Self> {
        let (resolved_cwd, cwd) = Self::cwd(state, id, cwd_policy)?;
        let directory = open_directory(&cwd).map_err(|_| Failure::Cwd)?;
        let history = std::path::absolute(&state.history_file)?;
        let sidecar_dir = history.parent().ok_or(Failure::Cwd)?.to_path_buf();
        let mut scope = Self {
            id: CURRENT_SCOPE.to_string(),
            resolved_cwd,
            session_root: cwd.clone(),
            cwd,
            cwd_policy,
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
            resolved_cwd: cwd.clone(),
            cwd,
            session_root: self.session_root.clone(),
            cwd_policy: self.cwd_policy,
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
            path: self.resolved_cwd.display().to_string(),
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
        let (resolved_cwd, cwd) = Self::cwd(state, id, self.cwd_policy)?;
        if cwd != self.cwd || resolved_cwd.as_os_str() != self.resolved_cwd.as_os_str() {
            return Err(Failure::Stale);
        }
        self.verify_directory()
    }

    fn check_worktree_candidate(&self, git: Git) -> Result<()> {
        let mut dir = self.directory.try_clone()?;
        let mut identity = dir.metadata()?;
        loop {
            git.remaining()?;
            match stat_at(&dir, c".git") {
                Ok(stat) => match stat.st_mode & libc::S_IFMT {
                    libc::S_IFDIR => {
                        let marker =
                            open_at(&dir, c".git", libc::O_DIRECTORY).map_err(|_| Failure::Cwd)?;
                        match stat_at(&marker, c"HEAD") {
                            Ok(_) => return Ok(()),
                            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                            Err(_) => return Err(Failure::Cwd),
                        }
                    }
                    libc::S_IFREG => {
                        let mut marker = open_at(&dir, c".git", 0).map_err(|_| Failure::Cwd)?;
                        if !marker.metadata()?.is_file() {
                            return Err(Failure::Cwd);
                        }
                        let mut prefix = [0; b"gitdir: ".len()];
                        match marker.read_exact(&mut prefix) {
                            Ok(()) if &prefix == b"gitdir: " => return Ok(()),
                            Ok(()) => return Err(Failure::Unavailable(Unavailable::Worktree)),
                            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                                return Err(Failure::Unavailable(Unavailable::Worktree));
                            }
                            Err(_) => return Err(Failure::Cwd),
                        }
                    }
                    _ => return Err(Failure::Cwd),
                },
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(_) => return Err(Failure::Cwd),
            }
            // Match Git's default discovery boundary after inherited GIT_*
            // overrides are removed. A nested cwd can still use its own repo.
            let parent = open_at(&dir, c"..", libc::O_DIRECTORY).map_err(|_| Failure::Cwd)?;
            let parent_identity = parent.metadata()?;
            if parent_identity.dev() != identity.dev() || parent_identity.ino() == identity.ino() {
                return Err(Failure::Unavailable(Unavailable::Worktree));
            }
            dir = parent;
            identity = parent_identity;
        }
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

    fn untracked_stats(&self, entry: &Entry, git: Git) -> Result<LineStats> {
        git.remaining()?;
        let Current::Regular(stamp, file) = self.inspect(&entry.path)? else {
            return Err(Failure::Stale);
        };
        if entry.kind != Kind::Regular(stamp) {
            return Err(Failure::Stale);
        }
        untracked_line_stats(file, stamp, git)
    }

    fn contents(&self, entry: &Entry, limit: usize) -> Result<Option<Bounded>> {
        let current = self.inspect(&entry.path)?;
        if current.kind() != entry.kind {
            return Err(Failure::Stale);
        }
        if let Current::Regular(stamp, file) = current {
            return Ok(Some(Bounded::read_file(file, stamp, limit)?));
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

fn open_at(dir: &File, name: &CStr, flags: libc::c_int) -> io::Result<File> {
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

fn stat_at(dir: &File, name: &CStr) -> io::Result<libc::stat> {
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

struct DirStream(*mut libc::DIR);

impl DirStream {
    fn open(dir: &File) -> io::Result<Self> {
        let fd = dir.try_clone()?.into_raw_fd();
        // SAFETY: fd is an open directory and ownership transfers on success.
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            let error = io::Error::last_os_error();
            // SAFETY: on failure the fd is still owned here.
            unsafe { libc::close(fd) };
            return Err(error);
        }
        Ok(Self(stream))
    }

    fn next(&mut self) -> io::Result<Option<&CStr>> {
        // SAFETY: the stream is owned by self. A returned name is valid
        // until the next call, which needs an exclusive borrow of self.
        unsafe {
            #[cfg(target_os = "linux")]
            let errno = libc::__errno_location();
            #[cfg(target_vendor = "apple")]
            let errno = libc::__error();
            #[cfg(any(target_os = "linux", target_vendor = "apple"))]
            {
                *errno = 0;
            }
            let entry = libc::readdir(self.0);
            if entry.is_null() {
                #[cfg(any(target_os = "linux", target_vendor = "apple"))]
                if *errno != 0 {
                    return Err(io::Error::from_raw_os_error(*errno));
                }
                return Ok(None);
            }
            Ok(Some(CStr::from_ptr(
                std::ptr::addr_of!((*entry).d_name).cast(),
            )))
        }
    }
}

impl Drop for DirStream {
    fn drop(&mut self) {
        // SAFETY: this closes the owned stream and its descriptor once.
        unsafe { libc::closedir(self.0) };
    }
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
        // Snapshot the non-Git environment. Inherited overrides must never
        // affect repository, worktree, index, or object-store selection.
        command
            .env_clear()
            .envs(std::env::vars_os().filter(|(key, _)| !key.as_bytes().starts_with(b"GIT_")));
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
        self.run_with(command, limit, Stdio::null(), |_| false)
    }

    fn run_with(
        &self,
        command: &mut Command,
        limit: usize,
        stdin: Stdio,
        stop: impl FnMut(&[u8]) -> bool,
    ) -> Result<GitOutput> {
        self.remaining()?;
        command
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = Self::spawn(command)?;
        let output = (|| {
            let (data, stopped) =
                self.read_pipe(child.stdout.take().ok_or(Failure::Git)?, limit, false, stop)?;
            let status = if data.capped || stopped {
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

    fn spawn(command: &mut Command) -> Result<Child> {
        crate::fork_guard::spawn(command).map_err(|error| match error.kind() {
            io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => {
                Failure::Unavailable(Unavailable::Git)
            }
            _ => Failure::Git,
        })
    }

    fn run_with_stderr(&self, command: &mut Command) -> Result<GitOutput> {
        self.remaining()?;
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .process_group(0);
        let mut child = crate::fork_guard::spawn(command)
            .map_err(|error| Failure::CommitFailed(format!("could not start Git: {error}")))?;
        let output = (|| {
            let (data, _) = self.read_pipe(
                child.stderr.take().ok_or(Failure::CommitOutcomeUnknown)?,
                COMMIT_STDERR_LIMIT,
                true,
                |_| false,
            )?;
            let status = Some(self.wait(&mut child)?);
            Ok(GitOutput { status, data })
        })();
        if output.is_err() {
            // SAFETY: Git was started as its own process group. Kill its
            // hooks as well so they cannot keep the command alive after timeout.
            unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) };
            let _ = child.wait();
        }
        output
    }

    fn read_pipe(
        &self,
        mut pipe: impl Read + AsRawFd,
        limit: usize,
        drain: bool,
        mut stop: impl FnMut(&[u8]) -> bool,
    ) -> Result<(Bounded, bool)> {
        let mut bytes = Vec::new();
        let mut stopped = false;
        let mut buffer = [0; 8192];
        let mut descriptor = libc::pollfd {
            fd: pipe.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        while drain || bytes.len() <= limit {
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
            let available = limit + 1 - bytes.len();
            let room = if drain {
                buffer.len()
            } else {
                buffer.len().min(available)
            };
            match pipe.read(&mut buffer[..room]) {
                Ok(0) => break,
                Ok(read) => bytes.extend_from_slice(&buffer[..read.min(available)]),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
            stopped = stop(&bytes[..bytes.len().min(limit)]);
            if stopped {
                break;
            }
        }
        Ok((Bounded::new(bytes, limit), stopped))
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

    fn object_command(&self, git: Git, cwd: &Path, worktree: &Path) -> Command {
        let mut command = git.command(cwd);
        command
            .env("GIT_DIR", &self.0)
            .env("GIT_WORK_TREE", worktree)
            .env("GIT_NO_REPLACE_OBJECTS", "1");
        command
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ObjectStores(Vec<PathBuf>);

impl ObjectStores {
    fn read(git: Git, primary: PathBuf, session_root: &Path) -> Result<Self> {
        let mut seen = FastSet::default();
        seen.insert(primary.clone());
        let mut stores = vec![primary];
        let mut index = 0;
        while let Some(store) = stores.get(index).cloned() {
            git.remaining()?;
            let data = read_alternates(&store)?;
            for line in data
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty() && !line.starts_with(b"#"))
            {
                git.remaining()?;
                // Git resolves relative entries against the object store, not
                // its info directory or the checkout.
                let path = store
                    .join(alternate_path(line)?)
                    .canonicalize()
                    .map_err(|_| Failure::RepositoryUnavailable)?;
                if !path.starts_with(session_root) {
                    return Err(Failure::RepositoryUnavailable);
                }
                if seen.insert(path.clone()) {
                    if stores.len() == ALTERNATE_LIMIT + 1 {
                        return Err(Failure::RepositoryUnavailable);
                    }
                    stores.push(path);
                }
            }
            index += 1;
        }
        Ok(Self(stores))
    }

    fn install(&self, git: Git, scratch: &Scratch) -> Result<()> {
        let mut alternates = String::new();
        for (index, store) in self.0.iter().enumerate() {
            let name = if index == 0 {
                "objects".to_string()
            } else {
                format!("alternate-{index}")
            };
            let dest = scratch.0.join(&name);
            fs::create_dir(&dest)?;
            // Only packs and loose object prefixes are linked. Giving Git the
            // live object directories would let it reread the untrusted
            // alternates files after validation, including transitive entries.
            for entry in fs::read_dir(store).map_err(|_| Failure::RepositoryUnavailable)? {
                git.remaining()?;
                let entry = entry.map_err(|_| Failure::RepositoryUnavailable)?;
                let name = entry.file_name();
                let bytes = name.as_bytes();
                if bytes == b"pack"
                    || bytes.len() == 2
                        && bytes
                            .iter()
                            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
                {
                    if !entry
                        .file_type()
                        .map_err(|_| Failure::RepositoryUnavailable)?
                        .is_dir()
                    {
                        return Err(Failure::RepositoryUnavailable);
                    }
                    symlink(entry.path(), dest.join(name))?;
                }
            }
            if index > 0 {
                alternates.push_str(&format!("../{name}\n"));
            }
        }
        if !alternates.is_empty() {
            fs::create_dir(scratch.0.join("objects/info"))?;
            scratch
                .file("objects/info/alternates")?
                .write_all(alternates.as_bytes())?;
        }
        Ok(())
    }
}

fn read_alternates(store: &Path) -> Result<Vec<u8>> {
    let directory = open_directory(store).map_err(|_| Failure::RepositoryUnavailable)?;
    let info = match open_at(&directory, c"info", libc::O_DIRECTORY) {
        Ok(info) => info,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err(Failure::RepositoryUnavailable),
    };
    match stat_at(&info, c"alternates") {
        Ok(stat) if stat.st_mode & libc::S_IFMT == libc::S_IFREG => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        _ => return Err(Failure::RepositoryUnavailable),
    }
    let file = open_at(&info, c"alternates", 0).map_err(|_| Failure::RepositoryUnavailable)?;
    if !file
        .metadata()
        .map_err(|_| Failure::RepositoryUnavailable)?
        .is_file()
    {
        return Err(Failure::RepositoryUnavailable);
    }
    let mut data = Vec::new();
    file.take((ALTERNATES_FILE_LIMIT + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|_| Failure::RepositoryUnavailable)?;
    if data.len() > ALTERNATES_FILE_LIMIT {
        return Err(Failure::RepositoryUnavailable);
    }
    Ok(data)
}

fn alternate_path(line: &[u8]) -> Result<PathBuf> {
    // Git accepts C-style quoted paths as well as literal path bytes.
    let mut decoded = Vec::new();
    let bytes = if let Some(quoted) = line.strip_prefix(b"\"") {
        let mut quoted = quoted.iter().copied();
        loop {
            match quoted.next() {
                Some(b'"') if quoted.len() == 0 => break,
                Some(b'"') | None => return Err(Failure::RepositoryUnavailable),
                Some(b'\\') => decoded.push(match quoted.next() {
                    Some(b'a') => 7,
                    Some(b'b') => 8,
                    Some(b't') => b'\t',
                    Some(b'n') => b'\n',
                    Some(b'v') => 11,
                    Some(b'f') => 12,
                    Some(b'r') => b'\r',
                    Some(byte @ (b'\\' | b'"')) => byte,
                    Some(first @ b'0'..=b'3') => {
                        let Some(second @ b'0'..=b'7') = quoted.next() else {
                            return Err(Failure::RepositoryUnavailable);
                        };
                        let Some(third @ b'0'..=b'7') = quoted.next() else {
                            return Err(Failure::RepositoryUnavailable);
                        };
                        (first - b'0') * 64 + (second - b'0') * 8 + (third - b'0')
                    }
                    _ => return Err(Failure::RepositoryUnavailable),
                }),
                Some(byte) => decoded.push(byte),
            }
        }
        decoded.as_slice()
    } else {
        line
    };
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(Failure::RepositoryUnavailable);
    }
    Ok(PathBuf::from(OsStr::from_bytes(bytes)))
}

struct RepoLayout {
    root: PathBuf,
    git_dir: PathBuf,
    common: PathBuf,
    index: PathBuf,
    objects: ObjectStores,
    config: &'static [u8],
    oid_width: usize,
}

impl RepoLayout {
    fn read(git: Git, scope: &mut Scope) -> Result<Self> {
        // Only ask Git to validate a checkout candidate, not a plain cwd.
        scope.check_worktree_candidate(git)?;
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
        match layout.code() {
            Some(0) => {}
            Some(128) if layout.data.bytes.is_empty() => return Err(Failure::Cwd),
            _ => return Err(Failure::Git),
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
        let objects = ObjectStores::read(git, objects, &scope.session_root)?;
        for private in [&git_dir, &common].into_iter().chain(&objects.0) {
            scope.protect(private)?;
            if scope.cwd.starts_with(private) {
                return Err(Failure::Cwd);
            }
        }
        let (oid_width, config) = match *object_format {
            b"sha1" => (40, b"[core]\nrepositoryformatversion = 0\n".as_slice()),
            b"sha256" => (
                64,
                b"[core]\nrepositoryformatversion = 1\n[extensions]\nobjectformat = sha256\n"
                    .as_slice(),
            ),
            _ => return Err(Failure::Cwd),
        };
        Ok(Self {
            root,
            git_dir,
            common,
            index,
            objects,
            config,
            oid_width,
        })
    }

    fn scratch(&self, git: Git, head: &[u8]) -> Result<Scratch> {
        let scratch = Scratch::new(SCRATCH_PREFIX)?;
        for dir in ["refs", "info"] {
            fs::create_dir(scratch.0.join(dir))?;
        }
        self.objects.install(git, &scratch)?;
        scratch.file("config")?.write_all(self.config)?;
        let mut head_file = scratch.file("HEAD")?;
        head_file.write_all(if head.is_empty() {
            b"ref: refs/heads/unborn"
        } else {
            head
        })?;
        head_file.write_all(b"\n")?;
        Ok(scratch)
    }

    fn valid_commit_id(&self, oid: &[u8]) -> bool {
        oid.len() == self.oid_width && oid.iter().all(|b| matches!(*b, b'0'..=b'9' | b'a'..=b'f'))
    }
}

/// `git status` can run repository-configured clean filters. Run it against a
/// private Git directory with a copied index and only non-executable settings.
/// The source repository supplies objects and ignore rules, never commands.
struct Repo {
    git: Git,
    root: PathBuf,
    scratch: Scratch,
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
            ..
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

        let scratch = layout.scratch(git, &head)?;
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
            head,
            settings,
        })
    }

    fn command(&self, cwd: &Path) -> Command {
        let mut command = self.scratch.object_command(self.git, cwd, &self.root);
        for setting in &self.settings {
            command.arg("-c").arg(OsStr::from_bytes(setting));
        }
        command
    }

    fn head_commit_time(&self) -> Option<SystemTime> {
        // Read the inventory's pinned HEAD without loading worktree log settings.
        let output = self
            .git
            .limited(WORKTREE_RECENCY_TIME_LIMIT)
            .run(
                self.command(&self.root).args([
                    "log",
                    "-1",
                    "--no-walk",
                    "--format=%ct",
                    "HEAD",
                    "--",
                ]),
                32,
            )
            .ok()?;
        if output.code() != Some(0) {
            return None;
        }
        let seconds = std::str::from_utf8(trimmed_line(&output.data.bytes))
            .ok()?
            .parse()
            .ok()?;
        SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
    }

    fn status(&self, cwd: &Path) -> Result<Bounded> {
        let output = self.git.run(
            self.command(cwd).args([
                "status",
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--renames",
                "--ignore-submodules=dirty",
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

    fn incoming_staged_renames(&self, scope: &Scope) -> Result<FastSet<Vec<u8>>> {
        // A scoped status can report an inbound rename as an addition. Check
        // the pinned HEAD and copied index, without scanning worktree files
        // outside the selected directory, before using those index blobs.
        let output = self.git.limited(RENAME_SCAN_TIME_LIMIT).run(
            self.command(&self.root).args([
                "diff",
                "--cached",
                "--name-status",
                "-z",
                "--find-renames",
                "--diff-filter=RC",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=all",
                "--",
            ]),
            STATUS_LIMIT,
        )?;
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let mut incoming = FastSet::default();
        let mut parts = output
            .data
            .bytes
            .split_inclusive(|b| *b == 0)
            .map(|part| part.strip_suffix(b"\0").ok_or(Failure::Git));
        while let Some(status) = parts.next().transpose()? {
            if !status.starts_with(b"R") && !status.starts_with(b"C") {
                return Err(Failure::Git);
            }
            let old = parts.next().ok_or(Failure::Git)??;
            let new = parts.next().ok_or(Failure::Git)??;
            if scope.relative(&self.root, old).is_none()
                && scope.relative(&self.root, new).is_some()
            {
                incoming.insert(new.to_vec());
            }
        }
        Ok(incoming)
    }

    fn numstats<'a>(
        &self,
        git: Git,
        scope: &Scope,
        entries: &'a [Entry],
        side: DiffSide,
        blobs: &FastSet<&[u8]>,
    ) -> Result<FastMap<&'a Path, LineStats>> {
        let mut command = self.command(&scope.cwd);
        command.args([
            "diff",
            "--numstat",
            "-z",
            "--relative",
            "--find-renames",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
        ]);
        if side == DiffSide::Staged {
            command.arg("--cached");
        }
        command.arg("--");
        // Only include displayed candidates and their authorized rename sources.
        // A visible row's JSON uses more bytes than its literal path arguments.
        let mut path_budget = LIST_LIMIT;
        let mut rows = Vec::new();
        for entry in entries {
            if !entry.numstat_eligible(side)
                || entry
                    .blobs(side)
                    .any(|blob| !blobs.contains(blob.oid.as_slice()))
            {
                continue;
            }
            let paths = [Some(entry.path.as_path()), entry.rename_source(side)];
            let bytes = paths
                .iter()
                .flatten()
                .map(|path| path.as_os_str().len() + 1)
                .sum();
            let Some(remaining) = path_budget.checked_sub(bytes) else {
                break;
            };
            path_budget = remaining;
            command.args(paths.into_iter().flatten());
            rows.push(entry);
        }
        if rows.is_empty() {
            return Ok(FastMap::default());
        }
        let output = git.run(&mut command, STATUS_LIMIT)?;
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let stats = parse_numstats(&output.data.bytes)?;
        Ok(rows
            .into_iter()
            .filter_map(|entry| {
                let row = stats
                    .get(entry.path.as_os_str().as_bytes())
                    .copied()
                    .flatten()?;
                let old = entry
                    .rename_source(side)
                    .map(|path| path.as_os_str().as_bytes());
                if row.old_path != old
                    || git.remaining().is_err()
                    || side == DiffSide::Worktree
                        && scope.inspect(&entry.path).ok()?.kind() != entry.kind
                {
                    return None;
                }
                Some((entry.path.as_path(), row.line_stats()?))
            })
            .collect())
    }

    fn available_blobs<'a>(
        &self,
        git: Git,
        oids: impl Iterator<Item = &'a [u8]>,
    ) -> Result<FastSet<&'a [u8]>> {
        // One missing baseline would make Git discard the whole numstat batch.
        // Check object types in one pass without reading blob contents into Rust.
        let mut seen = FastSet::default();
        let oids: Vec<_> = oids.filter(|oid| seen.insert(*oid)).collect();
        let mut available = FastSet::default();
        if oids.is_empty() {
            return Ok(available);
        }
        let mut input = self.scratch.file("numstat-blobs")?;
        for oid in &oids {
            input.write_all(oid)?;
            input.write_all(b"\n")?;
        }
        let output = git.run_with(
            self.command(&self.root)
                .args(["cat-file", "--batch-check=%(objecttype)"]),
            oids.len() * (64 + b" missing\n".len()),
            Stdio::from(File::open(self.scratch.0.join("numstat-blobs"))?),
            |_| false,
        )?;
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let mut lines = output.data.bytes.split_inclusive(|byte| *byte == b'\n');
        for oid in oids {
            match lines.next() {
                Some(b"blob\n") => {
                    available.insert(oid);
                }
                Some(_) => {}
                None => return Err(Failure::Git),
            }
        }
        if lines.next().is_some() {
            return Err(Failure::Git);
        }
        Ok(available)
    }

    fn blob(&self, oid: &[u8], limit: usize) -> Result<Bounded> {
        read_blob(self.git, &mut self.command(&self.root), oid, limit)
    }

    fn hunks(&self, old: Option<&[u8]>, new: Option<&[u8]>) -> Result<Bounded> {
        diff_hunks(self.git, &self.scratch, old, new)
    }
}

fn read_blob(git: Git, command: &mut Command, oid: &[u8], limit: usize) -> Result<Bounded> {
    if !valid_oid(oid) {
        return Err(Failure::Git);
    }
    let output = git.run(
        command
            .args(["cat-file", "blob", "--end-of-options"])
            .arg(OsStr::from_bytes(oid)),
        limit,
    )?;
    if !output.data.capped && output.code() != Some(0) {
        return Err(Failure::Git);
    }
    Ok(output.data)
}

fn diff_hunks(
    git: Git,
    scratch: &Scratch,
    old: Option<&[u8]>,
    new: Option<&[u8]>,
) -> Result<Bounded> {
    let write = |name: &'static str, data: Option<&[u8]>| -> Result<&str> {
        if let Some(data) = data {
            scratch.file(name)?.write_all(data)?;
            Ok(name)
        } else {
            Ok("/dev/null")
        }
    };
    let before = write("before", old)?;
    let after = write("after", new)?;
    // Even a parent worktree of TMPDIR must not supply diff settings.
    let mut command = git.command(&scratch.0);
    command.env("GIT_DIR", scratch.0.join("no-git"));
    let output = git.run(
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
            Failure::Cwd | Failure::Unavailable(Unavailable::Worktree) => Failure::UnknownScope,
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

    fn read_file(mut file: File, stamp: Stamp, limit: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take((limit + 1) as u64)
            .read_to_end(&mut bytes)?;
        if Stamp::from(&file.metadata()?) != stamp {
            return Err(Failure::Stale);
        }
        Ok(Self::new(bytes, limit))
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
    head: Option<FileBlob>,
    index: Option<FileBlob>,
    worktree_mode: Option<&'static str>,
    kind: Kind,
    redacted_source: bool,
    unmerged: bool,
    untracked: bool,
    status: &'static str,
    staged_status: Option<&'static str>,
    worktree_status: Option<&'static str>,
}

struct FileBlob {
    mode: &'static str,
    oid: Vec<u8>,
}

impl FileBlob {
    fn parse(mode: &[u8], oid: &[u8]) -> Option<Self> {
        let mode = file_mode(mode)?;
        (valid_oid(oid) && oid.iter().any(|b| *b != b'0')).then(|| Self {
            mode,
            oid: oid.to_vec(),
        })
    }
}

fn file_mode(mode: &[u8]) -> Option<&'static str> {
    Some(match mode {
        b"100644" => "100644",
        b"100755" => "100755",
        b"120000" => "120000",
        b"160000" => "160000",
        _ => return None,
    })
}

fn file_status(code: u8) -> Result<Option<&'static str>> {
    Ok(Some(match code {
        b'M' => "M",
        b'A' => "A",
        b'D' => "D",
        b'R' => "R",
        b'C' => "C",
        b'T' => "T",
        b'U' => "U",
        b'.' => return Ok(None),
        _ => return Err(Failure::Git),
    }))
}

impl Entry {
    fn status(&self, side: DiffSide) -> Option<&'static str> {
        match side {
            DiffSide::Staged => self.staged_status,
            DiffSide::Worktree => self.worktree_status,
        }
    }

    fn rename_source(&self, side: DiffSide) -> Option<&Path> {
        if matches!(self.status(side), Some("R" | "C")) {
            self.old_path.as_deref()
        } else {
            None
        }
    }

    fn before(&self, side: DiffSide) -> Option<&FileBlob> {
        if self.redacted_source {
            return None;
        }
        match side {
            DiffSide::Staged => self.head.as_ref(),
            DiffSide::Worktree => self.index.as_ref(),
        }
    }

    fn blobs(&self, side: DiffSide) -> impl Iterator<Item = &FileBlob> {
        let after = (side == DiffSide::Staged)
            .then_some(self.index.as_ref())
            .flatten();
        self.before(side).into_iter().chain(after)
    }

    fn numstat_eligible(&self, side: DiffSide) -> bool {
        self.status(side).is_some()
            && !self.redacted_source
            && !(side == DiffSide::Worktree && self.untracked)
            && self.unavailable(side).is_none()
    }

    fn unavailable(&self, side: DiffSide) -> Option<&'static str> {
        let worktree = side == DiffSide::Worktree;
        if worktree && self.kind == Kind::Symlink
            || self.blobs(side).any(|file| file.mode == "120000")
        {
            Some("symlink")
        } else if worktree && self.kind == Kind::Other
            || self.blobs(side).any(|file| file.mode == "160000")
        {
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
    staged_status: Option<&'static str>,
    worktree_status: Option<&'static str>,
    additions: Option<usize>,
    deletions: Option<usize>,
    staged_additions: Option<usize>,
    staged_deletions: Option<usize>,
    binary: bool,
    untracked: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    unmerged: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
}

#[derive(Serialize)]
struct ScopeRow {
    #[serde(flatten)]
    scope: ScopeInfo,
    changed: bool,
    files: usize,
    #[serde(skip)]
    updated_at: Option<SystemTime>,
}

impl ScopeRow {
    fn read(git: Git, scope: &Scope, layout: &RepoLayout, session_id: &str) -> Result<Self> {
        // Count the eligible inventory, before the list's display limits or
        // blob reads. In particular, status-only nested directories do not count.
        let snapshot = Snapshot::read(scope, layout, session_id, git)?;
        let files = snapshot.entries.len();
        git.remaining()?;
        // A missing date does not invalidate a successfully read inventory.
        let updated_at = if snapshot.repo.head.is_empty() {
            scope
                .directory
                .metadata()
                .ok()
                .and_then(|meta| meta.modified().ok())
        } else {
            snapshot.repo.head_commit_time()
        };
        Ok(Self {
            scope: scope.info(),
            changed: files != 0,
            files,
            updated_at,
        })
    }

    fn sort_by_recency(scopes: &mut [Self]) {
        scopes.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| a.scope.label.cmp(&b.scope.label))
                .then_with(|| a.scope.id.cmp(&b.scope.id))
        });
    }
}

#[derive(Serialize)]
struct PlainScopeRow {
    id: String,
    label: String,
    path: String,
    is_current: bool,
    dirty: bool,
    file_count: usize,
}

#[derive(Serialize)]
struct ScopesReply<Row> {
    source: &'static str,
    git_available: bool,
    current_path: String,
    scopes: Vec<Row>,
    truncated: bool,
    reason: Option<&'static str>,
}

impl<Row> ScopesReply<Row> {
    fn truncate(&mut self, reason: &'static str) {
        self.truncated = true;
        self.reason.get_or_insert(reason);
    }
}

fn list_scopes(git: Git, current: &mut Scope, session_id: &str) -> Result<Response> {
    // Leave time to verify the session and return partial results before the
    // outer request deadline fires.
    let git = Git {
        deadline: git.deadline - SCAN_RESPONSE_MARGIN,
    };
    let layout = match RepoLayout::read(git, current) {
        Ok(layout) => layout,
        Err(Failure::Unavailable(unavailable)) => {
            let list = ListReply::plain(current, unavailable, git.deadline);
            let ScopeInfo {
                id,
                label,
                path,
                current,
            } = list.scope;
            return Ok(Json(ScopesReply {
                source: list.source,
                git_available: list.git_available,
                current_path: path.clone(),
                scopes: vec![PlainScopeRow {
                    id,
                    label,
                    path,
                    is_current: current,
                    dirty: false,
                    file_count: list.files.len(),
                }],
                truncated: list.truncated,
                reason: list.reason,
            })
            .into_response());
        }
        Err(error) => return Err(error),
    };
    let row = ScopeRow::read(git, current, &layout, session_id)?;
    let mut reply = ScopesReply {
        source: "git",
        git_available: true,
        current_path: row.scope.path.clone(),
        scopes: vec![row],
        truncated: false,
        reason: None,
    };
    let worktrees = match WorktreeList::read(git.limited(WORKTREE_TIME_LIMIT), current, &layout) {
        Ok(worktrees) => worktrees,
        Err(error) => {
            reply.truncate(match error {
                Failure::Timeout => SCOPES_TIMEOUT,
                _ => SCOPES_FAILED,
            });
            return Ok(Json(reply).into_response());
        }
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
    ScopeRow::sort_by_recency(&mut reply.scopes[1..]);
    Ok(Json(reply).into_response())
}

fn read_changes(
    git: Git,
    current: &mut Scope,
    session_id: &str,
    scope_id: Option<String>,
    detail: Option<DetailRequest>,
) -> Result<Response> {
    if detail.as_ref().is_some_and(|request| request.side.is_err()) {
        return Err(Failure::UnknownSide);
    }
    with_scope(
        git,
        current,
        scope_id.as_deref(),
        |scope, source| match source {
            ScopeSource::Git(layout) => {
                if detail
                    .as_ref()
                    .is_some_and(|request| request.revision.is_none())
                {
                    return Err(Failure::Stale);
                }
                Snapshot::read(scope, layout, session_id, git)?.respond(detail)
            }
            ScopeSource::Plain(unavailable) if detail.is_none() => Ok(Json(ListReply::plain(
                scope,
                unavailable,
                git.deadline - SCAN_RESPONSE_MARGIN,
            ))
            .into_response()),
            ScopeSource::Plain(_) => Err(Failure::NotGitRepository),
        },
    )
}

enum ScopeSource<'a> {
    Git(&'a RepoLayout),
    Plain(Unavailable),
}

fn repo_scope(
    git: Git,
    current: &Scope,
    layout: RepoLayout,
    scope_id: Option<&str>,
) -> Result<(Option<Scope>, RepoLayout)> {
    if let Some(id) = scope_id {
        let worktrees = WorktreeList::read(git, current, &layout)?;
        let (scope, layout) = worktrees.resolve(git, current, &layout, id)?;
        Ok((Some(scope), layout))
    } else {
        Ok((None, layout))
    }
}

fn with_scope<T>(
    git: Git,
    current: &mut Scope,
    scope_id: Option<&str>,
    read: impl FnOnce(&Scope, ScopeSource<'_>) -> Result<T>,
) -> Result<T> {
    let resolved = RepoLayout::read(git, current)
        .map_err(|error| match error {
            Failure::Unavailable(_) if scope_id.is_some() => Failure::UnknownScope,
            error => error,
        })
        .and_then(|layout| repo_scope(git, current, layout, scope_id));
    let (scope, layout) = match resolved {
        Ok(resolved) => resolved,
        Err(Failure::Unavailable(unavailable)) => {
            return read(current, ScopeSource::Plain(unavailable));
        }
        Err(error) => return Err(error),
    };
    let result = read(scope.as_ref().unwrap_or(current), ScopeSource::Git(&layout))?;
    if let Some(scope) = scope {
        scope.verify_directory()?;
    }
    Ok(result)
}

fn write_commit(
    git: Git,
    state: &ServeState,
    store: &SessionStore,
    session_id: &str,
    input: CreateCommit,
) -> Result<Response> {
    if input.message.trim().is_empty() || input.message.contains('\0') {
        return Err(Failure::CommitMessage);
    }
    let scope_id = parse_scope(input.scope)?;
    let mut current = Scope::new(state, store, session_id, CwdPolicy::Recorded)?;
    // Keep this service's writes from racing each other's HEAD readback.
    let _permit = COMMIT_CAPACITY
        .try_acquire()
        .map_err(|_| Failure::CommitBusy)?;
    let probe = git.limited(TIME_LIMIT);
    let layout = RepoLayout::read(probe, &mut current).map_err(|error| match error {
        Failure::Unavailable(Unavailable::Worktree) => Failure::NotGitRepository,
        error => error,
    })?;
    let (selected, layout) = repo_scope(probe, &current, layout, scope_id.as_deref())?;
    let scope = selected.as_ref().unwrap_or(&current);
    // The commit covers the entire selected index, even when the Changes
    // list is rooted in a subdirectory. Only a single output byte is needed.
    let unmerged = probe.run(
        probe
            .command(&layout.root)
            .args(["ls-files", "--unmerged", "-z", "--"]),
        1,
    )?;
    if !unmerged.data.bytes.is_empty() {
        return Err(Failure::UnmergedConflicts);
    }
    if unmerged.code() != Some(0) {
        return Err(Failure::Git);
    }
    let staged = probe.run_with_stderr(git.command(&scope.cwd).args([
        "diff",
        "--cached",
        "--quiet",
        "--no-relative",
        "--no-ext-diff",
        "--no-textconv",
        "--ignore-submodules=none",
        "--",
    ]))?;
    match staged.code() {
        Some(0) => return Err(Failure::NothingStaged),
        Some(1) => {}
        _ => return Err(commit_failure(staged)),
    }
    current.verify(state, store, session_id)?;
    if let Some(scope) = &selected {
        scope.verify_directory()?;
    }
    let output = git
        .run_with_stderr(
            git.command(&scope.cwd)
                .args(["commit", "-m", &input.message]),
        )
        .map_err(|error| match error {
            Failure::CommitFailed(_) => error,
            _ => Failure::CommitOutcomeUnknown,
        })?;
    if output.code() != Some(0) {
        return Err(commit_failure(output));
    }
    let reply = read_new_commit(git.limited(TIME_LIMIT), scope, &layout)
        .map_err(|_| Failure::CommitSummaryUnavailable)?;
    Ok(Json(reply).into_response())
}

fn commit_failure(output: GitOutput) -> Failure {
    let stderr = String::from_utf8_lossy(&output.data.bytes);
    let stderr = stderr.trim();
    let message = if stderr.is_empty() {
        match output.status {
            Some(status) => format!("Git failed: {status}"),
            None => "Git failed".to_string(),
        }
    } else {
        stderr[..stderr.floor_char_boundary(COMMIT_STDERR_LIMIT)].to_string()
    };
    Failure::CommitFailed(message)
}

#[derive(Serialize)]
struct NewCommit {
    id: String,
    short_id: String,
    branch: Option<String>,
}

#[derive(Serialize)]
struct NewCommitReply {
    commit: NewCommit,
    message: String,
    files: usize,
    additions: Option<usize>,
    deletions: Option<usize>,
}

fn read_new_commit(git: Git, scope: &Scope, layout: &RepoLayout) -> Result<NewCommitReply> {
    let show = || {
        let mut command = git.command(&scope.cwd);
        command.env("GIT_NO_REPLACE_OBJECTS", "1").args([
            "show",
            "--no-notes",
            "--no-decorate",
            "--no-show-signature",
            "--no-color",
        ]);
        command
    };
    let output = git.run(
        show().args([
            "--no-patch",
            "--encoding=UTF-8",
            "--format=format:%H%x00%h%x00%B",
            "--end-of-options",
            "HEAD",
            "--",
        ]),
        COMMIT_INVENTORY_LIMIT,
    )?;
    if output.code() != Some(0) {
        return Err(Failure::Git);
    }
    let mut fields = output.data.bytes.splitn(3, |b| *b == 0);
    let (Some(id), Some(short), Some(message)) = (fields.next(), fields.next(), fields.next())
    else {
        return Err(Failure::Git);
    };
    if !layout.valid_commit_id(id) || short.is_empty() || !id.starts_with(short) {
        return Err(Failure::Git);
    }
    let id = as_git_str(id)?;
    let branch = git.run(
        git.command(&scope.cwd)
            .args(["symbolic-ref", "--quiet", "HEAD"]),
        8192,
    )?;
    let branch = match branch.code() {
        Some(0) => trimmed_line(&branch.data.bytes)
            .strip_prefix(b"refs/heads/")
            .map(|name| String::from_utf8_lossy(name).into_owned()),
        Some(1) if branch.data.bytes.is_empty() => None,
        _ => return Err(Failure::Git),
    };
    let output = git.run(
        show().args([
            "--format=format:",
            "--diff-merges=first-parent",
            "--raw",
            "--numstat",
            "-z",
            "--find-renames",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--submodule=short",
            "--no-relative",
            "--end-of-options",
            id,
            "--",
        ]),
        COMMIT_INVENTORY_LIMIT,
    )?;
    if output.code() != Some(0) {
        return Err(Failure::Git);
    }
    let (files, totals) = committed_totals(&output.data.bytes)?;
    Ok(NewCommitReply {
        commit: NewCommit {
            id: id.to_owned(),
            short_id: as_git_str(short)?.to_owned(),
            branch,
        },
        message: String::from_utf8_lossy(message)
            .trim_end_matches('\n')
            .to_owned(),
        files,
        additions: totals.map(|(additions, _)| additions),
        deletions: totals.map(|(_, deletions)| deletions),
    })
}

fn committed_totals(bytes: &[u8]) -> Result<(usize, Option<(usize, usize)>)> {
    // Raw records precede numstat and identify gitlinks, whose numeric
    // representation is a synthetic change to "Subproject commit" lines.
    let mut raw = bytes.split_inclusive(|b| *b == 0).peekable();
    let mut size = 0;
    let mut files = 0;
    let mut gitlinks = false;
    while let Some(meta) = raw.next_if(|record| record.starts_with(b":")) {
        let mut fields = meta.strip_suffix(b"\0").ok_or(Failure::Git)?[1..].split(|b| *b == b' ');
        let (Some(before), Some(after), Some(status)) =
            (fields.next(), fields.next(), fields.nth(2))
        else {
            return Err(Failure::Git);
        };
        let paths = if status.starts_with(b"R") || status.starts_with(b"C") {
            2
        } else {
            1
        };
        size += meta.len();
        for _ in 0..paths {
            size += raw
                .next()
                .filter(|path| path.ends_with(b"\0"))
                .ok_or(Failure::Git)?
                .len();
        }
        gitlinks |= before == b"160000" || after == b"160000";
        files += 1;
    }
    let stats = parse_numstats(&bytes[size..])?;
    if stats.len() != files {
        return Err(Failure::Git);
    }
    let totals = (!gitlinks).then_some((0usize, 0usize)).and_then(|initial| {
        stats.values().try_fold(initial, |(added, deleted), row| {
            let (additions, deletions) = row.as_ref()?.text_counts()?;
            Some((
                added.checked_add(additions)?,
                deleted.checked_add(deletions)?,
            ))
        })
    });
    Ok((files, totals))
}

fn read_commit_browser(
    git: Git,
    current: &mut Scope,
    scope_id: Option<&str>,
    request: CommitRequest,
) -> Result<Response> {
    let history = matches!(request, CommitRequest::History { .. });
    let result = with_scope(git, current, scope_id, |scope, source| match source {
        ScopeSource::Git(layout) => read_commits(git, scope, layout, request),
        ScopeSource::Plain(unavailable) => Err(Failure::Unavailable(unavailable)),
    });
    if !history {
        return result;
    }
    let (code, reason) = match result {
        Err(Failure::Unavailable(unavailable)) => ("git_unavailable", unavailable.reason()),
        Err(Failure::RepositoryUnavailable) => ("repository_unavailable", REPOSITORY_UNAVAILABLE),
        result => return result,
    };
    Ok(CommitHistoryReply::empty(current)
        .conflict(code, reason)
        .into_response())
}

fn read_commits(
    git: Git,
    scope: &Scope,
    layout: &RepoLayout,
    request: CommitRequest,
) -> Result<Response> {
    let requested_id = match &request {
        CommitRequest::History { before, .. } => before.as_deref(),
        CommitRequest::Detail { commit_id } => Some(commit_id.as_str()),
    };
    if requested_id.is_some_and(|id| !layout.valid_commit_id(id.as_bytes())) {
        return Err(Failure::CommitId);
    }
    let repo = CommitRepo::open(git, scope, layout)?;
    match request {
        CommitRequest::History { limit, before } => Ok(repo
            .history(scope, limit, before.as_deref())?
            .into_response()),
        CommitRequest::Detail { commit_id } => {
            Ok(Json(repo.detail(scope, &commit_id)?).into_response())
        }
    }
}

/// Commit operations need the selected checkout's HEAD and object store, but
/// never the live index, working files, config, refs, or grafts.
struct CommitRepo<'a> {
    git: Git,
    layout: &'a RepoLayout,
    scratch: Scratch,
    branch: Option<String>,
    head: Option<String>,
    shallow: Vec<u8>,
}

impl<'a> CommitRepo<'a> {
    fn open(git: Git, scope: &Scope, layout: &'a RepoLayout) -> Result<Self> {
        let head_ref = || -> Result<Option<Vec<u8>>> {
            let output = git.run(
                git.command(&scope.cwd)
                    .args(["symbolic-ref", "--quiet", "HEAD"]),
                8192,
            )?;
            match output.code() {
                Some(0) => Ok(Some(trimmed_line(&output.data.bytes).to_vec())),
                Some(1) if output.data.bytes.is_empty() => Ok(None),
                _ => Err(Failure::Git),
            }
        };
        let reference = head_ref()?;
        let branch = reference
            .as_deref()
            .and_then(|name| name.strip_prefix(b"refs/heads/"))
            .map(|name| String::from_utf8_lossy(name).into_owned());
        let output = git.run(
            git.command(&scope.cwd).args([
                "rev-parse",
                "--verify",
                "--quiet",
                "--end-of-options",
                "HEAD",
            ]),
            256,
        )?;
        let oid = trimmed_line(&output.data.bytes);
        let head = match output.code() {
            Some(0) if layout.valid_commit_id(oid) => Some(as_git_str(oid)?.to_owned()),
            Some(1) if oid.is_empty() && branch.is_some() => None,
            _ => return Err(Failure::Git),
        };
        // Keep the symbolic name paired with the captured HEAD if a checkout
        // changes while the two fields are being read.
        if reference != head_ref()? {
            return Err(Failure::Git);
        }
        let scratch = layout.scratch(git, head.as_deref().unwrap_or_default().as_bytes())?;
        let shallow = read_shallow(layout)?;
        if !shallow.is_empty() {
            scratch.file("shallow")?.write_all(&shallow)?;
        }
        Ok(Self {
            git,
            layout,
            scratch,
            branch,
            head,
            shallow,
        })
    }

    fn command(&self) -> Command {
        let mut command = self
            .scratch
            .object_command(self.git, &self.scratch.0, &self.scratch.0);
        // Prove ancestry from commit objects, not the source's cached graph.
        command.args(["-c", "core.commitGraph=false"]);
        command
    }

    fn verify_commit(&self, id: &str) -> Result<bool> {
        let output = self.git.run(
            self.command()
                .args(["rev-parse", "--verify", "--quiet", "--end-of-options"])
                .arg(format!("{id}^{{commit}}")),
            256,
        )?;
        match output.code() {
            Some(0) => Ok(trimmed_line(&output.data.bytes) == id.as_bytes()),
            Some(1) if output.data.bytes.is_empty() => Ok(false),
            _ => Err(Failure::Git),
        }
    }

    fn on_first_parent_chain(&self, head: &str, id: &str) -> Result<bool> {
        if head == id {
            return Ok(true);
        }
        let mut scanned = 0;
        let mut found = false;
        let output = self.git.run_with(
            self.command()
                .args(["rev-list", "--first-parent", "--end-of-options", head, "--"]),
            COMMIT_SCAN_LIMIT,
            Stdio::null(),
            |bytes| {
                let tail = &bytes[scanned..];
                if let Some(last) = tail.iter().rposition(|b| *b == b'\n') {
                    found = tail[..last]
                        .split(|b| *b == b'\n')
                        .any(|oid| oid == id.as_bytes());
                    scanned += last + 1;
                }
                found
            },
        )?;
        if found {
            Ok(true)
        } else if output.code() == Some(0) && scanned == output.data.bytes.len() {
            Ok(false)
        } else {
            // A bounded or failed walk cannot establish a negative result.
            Err(Failure::Git)
        }
    }

    fn history(
        &self,
        scope: &Scope,
        limit: usize,
        before: Option<&str>,
    ) -> Result<CommitHistoryReply> {
        const CLIPPED: &str = "commit history limit reached";
        const STALE_CURSOR: &str = "commit cursor is not on the selected HEAD's first-parent chain";
        let mut reply = CommitHistoryReply::empty(scope);
        reply.branch = self.branch.clone();
        reply.head = self.head.clone();
        let Some(head) = self.head.as_deref() else {
            return Ok(if before.is_some() {
                reply.conflict("stale_cursor", STALE_CURSOR)
            } else {
                reply
            });
        };
        if let Some(cursor) = before {
            let valid = self.verify_commit(cursor)?;
            if !self.on_first_parent_chain(head, cursor)? {
                return Ok(reply.conflict("stale_cursor", STALE_CURSOR));
            }
            if !valid {
                return Err(Failure::Git);
            }
        }
        // Read one extra ID independently of subjects so even byte clipping
        // can return a cursor iff there are more local rows.
        let count = limit + 1;
        let start = before.unwrap_or(head);
        let ids = self.git.run(
            self.command()
                .args(["rev-list", "--first-parent"])
                .arg(format!("--max-count={count}"))
                .arg(if before.is_some() {
                    "--skip=1"
                } else {
                    "--skip=0"
                })
                .args(["--end-of-options", start, "--"]),
            (self.layout.oid_width + 1) * count,
        )?;
        if ids.code() != Some(0) {
            return Err(Failure::Git);
        }
        let ids: Vec<_> = as_git_str(&ids.data.bytes)?
            .split_terminator('\n')
            .collect();
        if ids.is_empty() {
            return Ok(reply);
        }
        if ids
            .iter()
            .any(|id| !self.layout.valid_commit_id(id.as_bytes()))
            || ids.len() > count
        {
            return Err(Failure::Git);
        }
        let page = &ids[..ids.len().min(limit)];
        let output = self.git.run(
            self.command()
                .args([
                    "log",
                    "--no-walk=unsorted",
                    "--no-patch",
                    "--no-notes",
                    "--no-decorate",
                    "--no-show-signature",
                    "--encoding=UTF-8",
                    "-z",
                    "--format=%H%x00%ct%x00%s",
                    "--end-of-options",
                ])
                .args(page)
                .arg("--"),
            LIST_LIMIT,
        )?;
        if !output.data.capped && output.code() != Some(0) {
            return Err(Failure::Git);
        }
        reply.next_cursor = Some(start.to_owned());
        reply.reason = Some(CLIPPED);
        let mut wire = wire_size(&reply);
        let mut fields = output.data.bytes.split_inclusive(|b| *b == 0);
        for expected in page {
            let id = fields.next().and_then(|field| field.strip_suffix(&[0]));
            let timestamp = fields.next().and_then(|field| field.strip_suffix(&[0]));
            let (Some(id), Some(timestamp), Some(subject)) = (id, timestamp, fields.next()) else {
                if output.data.capped {
                    break;
                }
                return Err(Failure::Git);
            };
            if id != expected.as_bytes() {
                return Err(Failure::Git);
            }
            let committed_at = as_git_str(timestamp)?.parse().map_err(|_| Failure::Git)?;
            let subject = String::from_utf8_lossy(match subject.strip_suffix(&[0]) {
                Some(subject) => subject,
                None if output.data.capped => subject,
                None => return Err(Failure::Git),
            });
            let row = CommitRow {
                id: (*expected).to_owned(),
                committed_at,
                subject: subject[..subject.floor_char_boundary(COMMIT_SUBJECT_LIMIT)].to_owned(),
            };
            wire += wire_size(&row) + usize::from(!reply.rows.is_empty());
            if wire > LIST_LIMIT {
                break;
            }
            reply.rows.push(row);
        }
        if reply.rows.is_empty() {
            return Err(Failure::InventoryLimit);
        }
        reply.truncated = output.data.capped || reply.rows.len() < page.len();
        reply.reason = reply.truncated.then_some(CLIPPED);
        reply.next_cursor = reply
            .rows
            .last()
            .filter(|_| reply.rows.len() < ids.len())
            .map(|row| row.id.clone());
        Ok(reply)
    }

    fn detail(&self, scope: &Scope, id: &str) -> Result<CommitDiffReply> {
        let head = self.head.as_deref().ok_or(Failure::CommitNotFound)?;
        if !self.verify_commit(id)? || !self.on_first_parent_chain(head, id)? {
            return Err(Failure::CommitNotFound);
        }
        let parent = self.first_parent(id)?;
        let mut reply = CommitDiffReply {
            source: "commit",
            scope: scope.info(),
            commit_id: id.to_owned(),
            base: "root",
            base_id: None,
            format: "none",
            diff: String::new(),
            truncated: false,
            reason: None,
        };
        if let Some(parent) = parent.as_deref() {
            match self.object_info(&[parent], "parent-input")?.as_slice() {
                [CommitObject::Commit] => {
                    reply.base = "first_parent";
                    reply.base_id = Some(parent.to_owned());
                }
                [CommitObject::Missing]
                    if self
                        .shallow
                        .split(|b| *b == b'\n')
                        .any(|oid| oid == id.as_bytes()) =>
                {
                    reply.base = "shallow_boundary";
                    reply.reason = Some("parent unavailable in shallow clone".to_owned());
                    return Ok(reply);
                }
                _ => return Err(Failure::Git),
            }
        }
        let inventory = self.inventory(id, parent.as_deref())?;
        let changes = self.authorized_changes(scope, &inventory.bytes)?;
        if changes.is_empty() {
            reply.reason = Some("no changes in this scope".to_string());
            return Ok(reply);
        }
        let oids: FastSet<_> = changes
            .iter()
            .flat_map(CommitDelta::sides)
            .filter(|side| side.mode != "160000")
            .map(|side| side.oid)
            .collect();
        let oids: Vec<_> = oids.into_iter().collect();
        let blobs = self
            .object_info(&oids, "blob-input")?
            .into_iter()
            .zip(oids)
            .map(|(info, oid)| match info {
                CommitObject::Blob(size) => Ok((oid, size)),
                _ => Err(Failure::Git),
            })
            .collect::<Result<FastMap<_, _>>>()?;
        reply.collect(self, changes, &blobs)?;
        Ok(reply)
    }

    fn first_parent(&self, id: &str) -> Result<Option<String>> {
        // Pretty-printed parents and revision expressions hide shallow parents.
        let output = self.git.run(
            self.command()
                .args(["cat-file", "commit", "--end-of-options", id]),
            PROBE_LIMIT,
        )?;
        if !output.data.capped && output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let mut lines = output.data.bytes.split_inclusive(|b| *b == b'\n');
        let tree = lines
            .next()
            .and_then(|line| line.strip_suffix(b"\n"))
            .and_then(|line| line.strip_prefix(b"tree "))
            .ok_or(Failure::Git)?;
        if !self.layout.valid_commit_id(tree) {
            return Err(Failure::Git);
        }
        let line = lines.next().ok_or(Failure::Git)?;
        if let Some(parent) = line.strip_prefix(b"parent ") {
            let parent = parent.strip_suffix(b"\n").ok_or(Failure::Git)?;
            if !self.layout.valid_commit_id(parent) {
                return Err(Failure::Git);
            }
            return Ok(Some(as_git_str(parent)?.to_owned()));
        }
        Ok(None)
    }

    fn object_info(&self, ids: &[&str], filename: &str) -> Result<Vec<CommitObject>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut input = Vec::with_capacity(ids.len() * (self.layout.oid_width + 1));
        for id in ids {
            if !self.layout.valid_commit_id(id.as_bytes()) {
                return Err(Failure::Git);
            }
            input.extend_from_slice(id.as_bytes());
            input.push(b'\n');
        }
        self.scratch.file(filename)?.write_all(&input)?;
        let output = self.git.run_with(
            self.command().args([
                "cat-file",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ]),
            ids.len() * (self.layout.oid_width + 32),
            Stdio::from(File::open(self.scratch.0.join(filename))?),
            |_| false,
        )?;
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        let mut lines = as_git_str(&output.data.bytes)?.split_terminator('\n');
        let info = ids
            .iter()
            .map(|id| {
                let data = lines
                    .next()
                    .and_then(|line| line.strip_prefix(id))
                    .and_then(|line| line.strip_prefix(' '))
                    .ok_or(Failure::Git)?;
                match data {
                    "missing" => Ok(CommitObject::Missing),
                    data => match data.split_once(' ') {
                        Some(("commit", _)) => Ok(CommitObject::Commit),
                        Some(("blob", size)) => {
                            Ok(CommitObject::Blob(size.parse().map_err(|_| Failure::Git)?))
                        }
                        Some(("tree" | "tag", _)) => Ok(CommitObject::Other),
                        _ => Err(Failure::Git),
                    },
                }
            })
            .collect::<Result<_>>()?;
        if lines.next().is_some() {
            return Err(Failure::Git);
        }
        Ok(info)
    }

    fn inventory(&self, id: &str, parent: Option<&str>) -> Result<Bounded> {
        let mut command = self.command();
        command.args([
            "diff-tree",
            "-r",
            "--root",
            "--no-commit-id",
            "--raw",
            "--no-abbrev",
            "-z",
            "--find-renames",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--end-of-options",
        ]);
        command.args(parent).args([id, "--"]);
        let output = self.git.run(&mut command, COMMIT_INVENTORY_LIMIT)?;
        if output.data.capped {
            return Err(Failure::InventoryLimit);
        }
        if output.code() != Some(0) {
            return Err(Failure::Git);
        }
        Ok(output.data)
    }

    fn authorized_changes<'b>(&self, scope: &Scope, raw: &'b [u8]) -> Result<Vec<CommitDelta<'b>>> {
        let mut changes = Vec::new();
        let mut records = raw.split_inclusive(|b| *b == 0);
        while let Some(meta) = records.next() {
            self.git.remaining()?;
            let mut fields = as_git_str(
                meta.strip_suffix(&[0])
                    .and_then(|meta| meta.strip_prefix(b":"))
                    .ok_or(Failure::Git)?,
            )?
            .split(' ');
            let (Some(old_mode), Some(new_mode), Some(old_id), Some(new_id), Some(status), None) = (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            ) else {
                return Err(Failure::Git);
            };
            let old_path = records
                .next()
                .and_then(|path| path.strip_suffix(&[0]))
                .ok_or(Failure::Git)?;
            let new_path = match status {
                "A" | "D" | "M" | "T" => old_path,
                status if status.starts_with('R') => records
                    .next()
                    .and_then(|path| path.strip_suffix(&[0]))
                    .ok_or(Failure::Git)?,
                _ => return Err(Failure::Git),
            };
            let side = |path, mode: &'b str, oid: &'b str| -> Result<Option<CommitSide<'b>>> {
                if !self.layout.valid_commit_id(oid.as_bytes()) {
                    return Err(Failure::Git);
                }
                match mode {
                    "000000" if oid.bytes().all(|b| b == b'0') => Ok(None),
                    "100644" | "100755" | "120000" | "160000" => Ok(scope
                        .relative(&self.layout.root, path)
                        .map(|path| CommitSide { path, mode, oid })),
                    _ => Err(Failure::Git),
                }
            };
            let before = side(old_path, old_mode, old_id)?;
            let after = side(new_path, new_mode, new_id)?;
            // An ineligible rename side contributes neither a name nor a blob.
            if before.is_some() || after.is_some() {
                changes.push(CommitDelta { before, after });
            }
        }
        Ok(changes)
    }
}

fn read_shallow(layout: &RepoLayout) -> Result<Vec<u8>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(layout.common.join("shallow"))
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    if !file.metadata()?.is_file() {
        return Err(Failure::Git);
    }
    let mut data = Vec::new();
    file.take((COMMIT_SCAN_LIMIT + 1) as u64)
        .read_to_end(&mut data)?;
    if data.len() > COMMIT_SCAN_LIMIT {
        return Err(Failure::InventoryLimit);
    }
    if data.split_inclusive(|b| *b == b'\n').any(|line| {
        line.strip_suffix(b"\n")
            .is_none_or(|id| !layout.valid_commit_id(id))
    }) {
        return Err(Failure::Git);
    }
    Ok(data)
}

fn as_git_str(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| Failure::Git)
}

enum CommitObject {
    Commit,
    Blob(u64),
    Missing,
    Other,
}

struct CommitSide<'a> {
    path: PathBuf,
    mode: &'a str,
    oid: &'a str,
}

struct CommitDelta<'a> {
    before: Option<CommitSide<'a>>,
    after: Option<CommitSide<'a>>,
}

#[derive(Serialize)]
struct CommitHistoryReply {
    source: &'static str,
    scope: ScopeInfo,
    branch: Option<String>,
    head: Option<String>,
    rows: Vec<CommitRow>,
    next_cursor: Option<String>,
    truncated: bool,
    reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'static str>,
}

impl CommitHistoryReply {
    fn empty(scope: &Scope) -> Self {
        Self {
            source: "commit_history",
            scope: scope.info(),
            branch: None,
            head: None,
            rows: Vec::new(),
            next_cursor: None,
            truncated: false,
            reason: None,
            code: None,
        }
    }

    fn conflict(mut self, code: &'static str, reason: &'static str) -> Self {
        self.code = Some(code);
        self.reason = Some(reason);
        self
    }

    fn into_response(self) -> Response {
        let status = if self.code.is_some() {
            StatusCode::CONFLICT
        } else {
            StatusCode::OK
        };
        (status, Json(self)).into_response()
    }
}

#[derive(Serialize)]
struct CommitRow {
    id: String,
    subject: String,
    committed_at: i64,
}

#[derive(Serialize)]
struct CommitDiffReply {
    source: &'static str,
    scope: ScopeInfo,
    commit_id: String,
    base: &'static str,
    base_id: Option<String>,
    format: &'static str,
    diff: String,
    truncated: bool,
    reason: Option<String>,
}

impl<'a> CommitDelta<'a> {
    fn sides(&self) -> impl Iterator<Item = &CommitSide<'a>> {
        self.before.iter().chain(&self.after)
    }

    fn notice(&self, reason: &str) -> String {
        let paths: Vec<_> = self.sides().map(|side| &side.path).collect();
        let first = quote_path("", paths[0]);
        if let [old, new] = paths.as_slice()
            && old != new
        {
            format!("{reason}: {first} -> {}\n", quote_path("", new))
        } else {
            format!("{reason}: {first}\n")
        }
    }

    fn patch(&self, repo: &CommitRepo<'_>, blobs: &FastMap<&str, u64>) -> Result<CommitPatch> {
        if let Some(kind) = self.sides().find_map(|side| match side.mode {
            "120000" => Some("symlink"),
            "160000" => Some("gitlink"),
            _ => None,
        }) {
            return Ok(CommitPatch::Notice(kind, false));
        }
        if self
            .sides()
            .any(|side| blobs[side.oid] > DIFF_INPUT_LIMIT as u64)
        {
            return Ok(CommitPatch::Notice("file exceeds diff input limit", true));
        }
        let load = |side: Option<&CommitSide<'_>>| -> Result<Option<Bounded>> {
            side.map(|side| {
                let blob = read_blob(
                    repo.git,
                    &mut repo.command(),
                    side.oid.as_bytes(),
                    DIFF_INPUT_LIMIT,
                )?;
                if blob.capped {
                    return Err(Failure::Git);
                }
                Ok(blob)
            })
            .transpose()
        };
        let same_blob = self
            .before
            .as_ref()
            .zip(self.after.as_ref())
            .is_some_and(|(a, b)| a.oid == b.oid);
        let before = load(self.before.as_ref())?;
        let after_blob = if same_blob {
            None
        } else {
            load(self.after.as_ref())?
        };
        let after = if same_blob {
            before.as_ref()
        } else {
            after_blob.as_ref()
        };
        if before.iter().chain(after).any(|blob| blob.text().is_none()) {
            return Ok(CommitPatch::Notice("binary", false));
        }
        let before_path = self.before.as_ref().map(|side| side.path.as_path());
        let after_path = self.after.as_ref().map(|side| side.path.as_path());
        let path = after_path.or(before_path).ok_or(Failure::Git)?;
        let renamed_from = before_path.filter(|old| *old != path);
        let (mut patch, a, b) = patch_preamble(path, renamed_from);
        match (self.before.as_ref(), self.after.as_ref()) {
            (None, Some(new)) => patch.push_str(&format!("new file mode {}\n", new.mode)),
            (Some(old), None) => patch.push_str(&format!("deleted file mode {}\n", old.mode)),
            (Some(old), Some(new)) if old.mode != new.mode => {
                patch.push_str(&format!("old mode {}\nnew mode {}\n", old.mode, new.mode));
            }
            _ => {}
        }
        let mut clipped = false;
        if !same_blob {
            let scratch = Scratch::new(SCRATCH_PREFIX)?;
            let input = diff_hunks(
                repo.git,
                &scratch,
                before.as_ref().map(|blob| blob.bytes.as_slice()),
                after.map(|blob| blob.bytes.as_slice()),
            )?;
            clipped = input.capped;
            if let Some(hunks) = hunk_text(&input)? {
                append_file_headers(
                    &mut patch,
                    before_path.map(|_| a.as_str()),
                    after_path.map(|_| b.as_str()),
                );
                patch.push_str(hunks);
            }
        }
        Ok(CommitPatch::Unified(patch, clipped))
    }
}

enum CommitPatch {
    Unified(String, bool),
    Notice(&'static str, bool),
}

impl CommitDiffReply {
    fn collect(
        &mut self,
        repo: &CommitRepo<'_>,
        changes: Vec<CommitDelta<'_>>,
        blobs: &FastMap<&str, u64>,
    ) -> Result<()> {
        // Allow escaped JSON strings, the final notice separator, and the
        // largest fixed fields when calculating space for complete lines.
        self.format = "unified";
        self.reason = Some(format!("{DIFF_TRUNCATED}\n"));
        let mut budget = DETAIL_LIMIT
            .checked_sub(wire_size(self))
            .ok_or(Failure::InventoryLimit)?;
        let mut lines = 0;
        let mut notices = String::new();
        'changes: for change in changes {
            repo.git.remaining()?;
            match change.patch(repo, blobs)? {
                CommitPatch::Notice(reason, oversized) => {
                    self.truncated |= oversized;
                    if !append_within(&mut notices, &change.notice(reason), &mut budget) {
                        self.truncated = true;
                        break;
                    }
                }
                CommitPatch::Unified(patch, clipped) => {
                    for line in patch.split_inclusive('\n') {
                        if lines == DIFF_LINES || !append_within(&mut self.diff, line, &mut budget)
                        {
                            self.truncated = true;
                            break 'changes;
                        }
                        lines += 1;
                    }
                    if clipped {
                        self.truncated = true;
                        break;
                    }
                }
            }
        }
        if self.diff.is_empty() {
            self.format = "none";
        }
        let notices = notices.trim_end_matches('\n');
        self.reason = match (self.truncated, notices.is_empty()) {
            (true, true) => Some(DIFF_TRUNCATED.to_string()),
            (true, false) => Some(format!("{DIFF_TRUNCATED}\n{notices}")),
            (false, false) => Some(notices.to_owned()),
            (false, true) => None,
        };
        Ok(())
    }
}

fn append_within(output: &mut String, line: &str, budget: &mut usize) -> bool {
    let size = wire_size(&line) - 2;
    if size > *budget {
        return false;
    }
    output.push_str(line);
    *budget -= size;
    true
}

#[derive(Serialize)]
struct ListReply {
    source: &'static str,
    root_label: String,
    scope: ScopeInfo,
    git_available: bool,
    revision: String,
    files: Vec<FileRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conflict_count: Option<usize>,
    truncated: bool,
    reason: Option<&'static str>,
}

impl ListReply {
    fn plain(scope: &Scope, unavailable: Unavailable, deadline: Instant) -> Self {
        let info = scope.info();
        let mut reply = Self {
            source: "plain",
            root_label: info.label.clone(),
            scope: info,
            git_available: false,
            // Reserve the maximum response overhead before adding rows.
            revision: format!("plain:{:064x}", 0),
            files: Vec::new(),
            conflict_count: None,
            truncated: true,
            reason: Some(PLAIN_INCOMPLETE),
        };
        let mut scan = PlainScan {
            paths: BinaryHeap::new(),
            remaining: PLAIN_SCAN_LIMIT,
            deadline: deadline - SCAN_RESPONSE_MARGIN,
            incomplete: false,
        };
        scan.walk(scope, &scope.directory, &mut PathBuf::new(), 0);
        let scope_key = scope.cwd.as_os_str().as_bytes();
        let mut digest = Sha256::new();
        hash_piece(&mut digest, scope_key);
        let mut size = wire_size(&reply);
        let mut clipped = false;
        for (display, path) in scan.paths.into_sorted_vec() {
            if reply.files.len() == FILE_LIMIT {
                clipped = true;
                break;
            }
            if Instant::now() >= deadline {
                scan.incomplete = true;
                break;
            }
            let Ok(Current::Regular(stamp, file)) = scope.inspect(&path) else {
                scan.incomplete = true;
                continue;
            };
            let Ok(probe) = Bounded::read_file(file, stamp, PROBE_LIMIT) else {
                scan.incomplete = true;
                continue;
            };
            let binary = probe.text().is_none();
            let row = FileRow {
                id: file_id(scope_key, &path),
                path: display,
                old_path: None,
                status: "file",
                staged_status: None,
                worktree_status: None,
                additions: None,
                deletions: None,
                staged_additions: None,
                staged_deletions: None,
                binary,
                untracked: false,
                unmerged: false,
                size: Some(stamp.0[2]),
            };
            size += wire_size(&row) + usize::from(!reply.files.is_empty());
            if size > LIST_LIMIT {
                clipped = true;
                break;
            }
            hash_piece(&mut digest, path.as_os_str().as_bytes());
            Kind::Regular(stamp).hash(&mut digest);
            digest.update([u8::from(binary)]);
            reply.files.push(row);
        }
        reply.truncated = scan.incomplete || clipped;
        let reason = if scan.incomplete {
            PLAIN_INCOMPLETE
        } else if clipped {
            LIST_TRUNCATED
        } else {
            unavailable.reason()
        };
        reply.reason = Some(reason);
        hash_piece(&mut digest, reason.as_bytes());
        reply.revision.truncate("plain:".len());
        use std::fmt::Write as _;
        for byte in digest.finalize() {
            write!(reply.revision, "{byte:02x}").expect("writing to a String cannot fail");
        }
        reply
    }
}

struct PlainScan {
    paths: BinaryHeap<(String, PathBuf)>,
    remaining: usize,
    deadline: Instant,
    incomplete: bool,
}

impl PlainScan {
    fn walk(&mut self, scope: &Scope, directory: &File, path: &mut PathBuf, depth: usize) {
        let Ok(mut entries) = DirStream::open(directory) else {
            self.incomplete = true;
            return;
        };
        loop {
            if self.remaining == 0 || Instant::now() >= self.deadline {
                self.incomplete = true;
                return;
            }
            let name = match entries.next() {
                Ok(Some(name)) => name,
                Ok(None) => return,
                Err(_) => {
                    self.incomplete = true;
                    return;
                }
            };
            self.remaining -= 1;
            let bytes = name.to_bytes();
            if matches!(bytes, b"." | b"..") || reserved(bytes) {
                continue;
            }
            path.push(OsStr::from_bytes(bytes));
            if scope
                .relative(&scope.cwd, path.as_os_str().as_bytes())
                .is_some()
            {
                match stat_at(directory, name) {
                    Ok(stat) => match stat.st_mode & libc::S_IFMT {
                        libc::S_IFDIR => {
                            if depth < PLAIN_DEPTH_LIMIT
                                && let Ok(dir) = open_at(directory, name, libc::O_DIRECTORY)
                            {
                                self.walk(scope, &dir, path, depth + 1);
                            } else {
                                self.incomplete = true;
                            }
                        }
                        libc::S_IFREG if stat.st_nlink == 1 => self.add(path),
                        _ => {}
                    },
                    Err(_) => self.incomplete = true,
                }
            }
            path.pop();
        }
    }

    fn add(&mut self, path: &Path) {
        // Keep the first sorted page regardless of the directory's entry order.
        let candidate = (path.to_string_lossy().into_owned(), path.to_path_buf());
        if self.paths.len() <= FILE_LIMIT {
            self.paths.push(candidate);
        } else if let Some(mut largest) = self.paths.peek_mut()
            && candidate < *largest
        {
            *largest = candidate;
        }
    }
}

#[derive(Serialize)]
struct DetailReply {
    revision: String,
    file_id: String,
    path: String,
    status: &'static str,
    side: DiffSide,
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
        let mut incoming = None;
        let mut records = status.bytes.split_inclusive(|b| *b == 0);
        while let Some(record) = records.next() {
            let record = record.strip_suffix(&[0]).ok_or(Failure::Git)?;
            repo.git.remaining()?;
            let (git_path, old, xy, modes, oids) = match record.first() {
                Some(b'1') => {
                    let parts: Vec<_> = record.splitn(9, |b| *b == b' ').collect();
                    let [_, xy, _, head, index, worktree, head_id, index_id, path] =
                        parts.as_slice()
                    else {
                        return Err(Failure::Git);
                    };
                    (
                        *path,
                        None,
                        *xy,
                        [*head, *index, *worktree],
                        [*head_id, *index_id],
                    )
                }
                Some(b'2') => {
                    let parts: Vec<_> = record.splitn(10, |b| *b == b' ').collect();
                    let [_, xy, _, head, index, worktree, head_id, index_id, _, path] =
                        parts.as_slice()
                    else {
                        return Err(Failure::Git);
                    };
                    let old = records
                        .next()
                        .and_then(|old| old.strip_suffix(&[0]))
                        .ok_or(Failure::Git)?;
                    (
                        *path,
                        Some(old),
                        *xy,
                        [*head, *index, *worktree],
                        [*head_id, *index_id],
                    )
                }
                Some(b'u') => {
                    let parts: Vec<_> = record.splitn(11, |b| *b == b' ').collect();
                    let [_, xy, _, _, _, _, worktree, _, _, _, path] = parts.as_slice() else {
                        return Err(Failure::Git);
                    };
                    (*path, None, *xy, [b"", b"", *worktree], [b"".as_slice(); 2])
                }
                Some(b'?') if record.starts_with(b"? ") => (
                    &record[2..],
                    None,
                    b".A".as_slice(),
                    [b"".as_slice(); 3],
                    [b"".as_slice(); 2],
                ),
                Some(b'!') => continue,
                _ => return Err(Failure::Git),
            };
            let [staged, worktree] = xy else {
                return Err(Failure::Git);
            };
            let unmerged = record[0] == b'u';
            // The XY pair on an unmerged record describes a conflict, not diffs against the index.
            let (staged_status, worktree_status) = if unmerged {
                (None, None)
            } else {
                (file_status(*staged)?, file_status(*worktree)?)
            };
            let untracked = record[0] == b'?';
            let Some(path) = scope.relative(&repo.root, git_path) else {
                continue;
            };
            let kind = scope.inspect(&path)?.kind();
            if kind == Kind::Outside || untracked && kind == Kind::Missing {
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
            let redacted_source = if old.is_some() {
                old_path.is_none()
            } else if staged_status == Some("A")
                && worktree_status.is_some()
                && scope.cwd != repo.root
            {
                // If rename detection cannot settle an inbound source, don't
                // expose the potentially out-of-scope index blob in this diff.
                incoming
                    .get_or_insert_with(|| repo.incoming_staged_renames(scope))
                    .as_ref()
                    .map_or(true, |paths| paths.contains(git_path))
            } else {
                false
            };
            let status = if unmerged {
                "unknown"
            } else if old.is_some() && xy.contains(&b'R') {
                "renamed"
            } else if kind == Kind::Missing && xy.contains(&b'D') {
                "deleted"
            } else if xy.contains(&b'A') || xy.contains(&b'C') {
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
            digest.update([u8::from(redacted_source)]);
            kind.hash(&mut digest);
            entries.push(Entry {
                display: path.to_string_lossy().into_owned(),
                path,
                redacted_source,
                old_path,
                head: FileBlob::parse(modes[0], oids[0]),
                index: FileBlob::parse(modes[1], oids[1]),
                worktree_mode: file_mode(modes[2]),
                kind,
                unmerged,
                untracked,
                status,
                staged_status,
                worktree_status,
            });
        }
        entries.sort_by(|a, b| {
            a.display
                .cmp(&b.display)
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| a.untracked.cmp(&b.untracked))
        });
        entries.dedup_by(|later, first| {
            if later.path != first.path {
                return false;
            }
            // A path deleted from the index can be recreated as untracked.
            if later.untracked && !first.unmerged {
                first.untracked = true;
                first.worktree_status = later.worktree_status;
            }
            true
        });
        // Keep conflicting paths visible even when the regular file list is clipped.
        if entries.iter().any(|entry| entry.unmerged) {
            entries.sort_by_key(|entry| !entry.unmerged);
        }
        Ok(Self {
            scope,
            repo,
            revision: URL_SAFE_NO_PAD.encode(digest.finalize()),
            entries,
        })
    }

    fn row(&self, entry: &Entry, stats: SideStats) -> FileRow {
        let (additions, deletions) = LineStats::counts(stats.worktree);
        let (staged_additions, staged_deletions) = LineStats::counts(stats.staged);
        FileRow {
            id: file_id(self.revision.as_bytes(), &entry.path),
            path: entry.display.clone(),
            old_path: entry
                .old_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            status: entry.status,
            staged_status: entry.status(DiffSide::Staged),
            worktree_status: entry.status(DiffSide::Worktree),
            additions,
            deletions,
            staged_additions,
            staged_deletions,
            binary: matches!(stats.worktree, Some(LineStats::Binary))
                || matches!(stats.staged, Some(LineStats::Binary)),
            untracked: entry.untracked,
            unmerged: entry.unmerged,
            size: None,
        }
    }

    fn respond(&self, detail: Option<DetailRequest>) -> Result<Response> {
        match detail {
            Some(request) => Ok(self.detail(request)?.into_response()),
            None => Ok(Json(self.list().0).into_response()),
        }
    }

    fn list(&self) -> (ListReply, Vec<(&Entry, SideStats)>) {
        let scope = self.scope.info();
        let conflict_count = self
            .entries
            .iter()
            .take_while(|entry| entry.unmerged)
            .count();
        let mut list = ListReply {
            source: "working_tree",
            root_label: scope.label.clone(),
            scope,
            git_available: true,
            revision: self.revision.clone(),
            files: Vec::new(),
            conflict_count: (conflict_count > 0).then_some(conflict_count),
            truncated: true,
            reason: Some(LIST_TRUNCATED),
        };
        // Stats are optional; reserve a margin to return the inventory if Git times out.
        let git = Git {
            deadline: self.repo.git.deadline - SCAN_RESPONSE_MARGIN,
        };
        let candidates = &self.entries[..self.entries.len().min(FILE_LIMIT)];
        let blobs = self
            .repo
            .available_blobs(
                git,
                candidates.iter().flat_map(|entry| {
                    [DiffSide::Staged, DiffSide::Worktree]
                        .into_iter()
                        .filter(|side| entry.numstat_eligible(*side))
                        .flat_map(|side| entry.blobs(side).map(|blob| blob.oid.as_slice()))
                }),
            )
            .unwrap_or_default();
        let stats_for = |side| {
            self.repo
                .numstats(git, self.scope, candidates, side, &blobs)
                .unwrap_or_default()
        };
        let staged = stats_for(DiffSide::Staged);
        let worktree = stats_for(DiffSide::Worktree);
        let mut size = wire_size(&list);
        let mut entries = Vec::new();
        for entry in candidates {
            let stats = SideStats {
                staged: staged.get(entry.path.as_path()).copied(),
                worktree: if entry.untracked && entry.unavailable(DiffSide::Worktree).is_none() {
                    // An untracked file is a diff from /dev/null, not an index entry.
                    self.scope.untracked_stats(entry, git).ok()
                } else {
                    worktree.get(entry.path.as_path()).copied()
                },
            };
            let row = self.row(entry, stats);
            size += wire_size(&row) + usize::from(!entries.is_empty());
            if size > LIST_LIMIT {
                break;
            }
            entries.push((entry, stats));
            list.files.push(row);
        }
        list.truncated = entries.len() < self.entries.len();
        list.reason = list.truncated.then_some(LIST_TRUNCATED);
        (list, entries)
    }

    fn detail(&self, request: DetailRequest) -> Result<Json<DetailReply>> {
        if request.revision.as_deref() != Some(&self.revision) {
            return Err(Failure::Stale);
        }
        let (list, entries) = self.list();
        let Some((row, (entry, stats))) = list
            .files
            .into_iter()
            .zip(entries)
            .find(|(row, _)| row.id == request.file_id)
        else {
            return Err(Failure::File);
        };
        let side = match request.side? {
            Some(side) if entry.unmerged || entry.status(side).is_some() => side,
            Some(_) => return Err(Failure::DiffSideUnavailable),
            None if entry.worktree_status.is_none() && entry.staged_status.is_some() => {
                DiffSide::Staged
            }
            None => DiffSide::Worktree,
        };
        let mut reply = DetailReply {
            revision: self.revision.clone(),
            file_id: row.id,
            path: row.path,
            status: row.status,
            side,
            format: "none",
            diff: String::new(),
            truncated: false,
            reason: None,
        };
        let untracked = side == DiffSide::Worktree && entry.untracked;
        if entry.unmerged {
            reply.reason = Some("file has unresolved merge conflicts");
        } else if entry.status(side).is_none() {
            reply.reason = Some("no content changes");
        } else if let Some(reason) = entry.unavailable(side) {
            reply.reason = Some(reason);
        } else if !untracked && matches!(stats.get(side), Some(LineStats::Binary)) {
            reply.reason = Some("binary");
        } else {
            let read_blob = |file: Option<&FileBlob>| {
                file.map(|file| self.repo.blob(&file.oid, DIFF_INPUT_LIMIT))
                    .transpose()
            };
            let before_file = entry.before(side);
            let before = read_blob(before_file)?;
            let (after, after_mode) = match side {
                DiffSide::Staged => (
                    read_blob(entry.index.as_ref())?,
                    entry.index.as_ref().map(|file| file.mode),
                ),
                DiffSide::Worktree => (
                    self.scope.contents(entry, DIFF_INPUT_LIMIT)?,
                    entry.worktree_mode,
                ),
            };
            let inputs = || before.iter().chain(&after);
            if inputs().any(|file| file.text().is_none()) {
                reply.reason = Some("binary");
            } else if inputs().any(|file| file.capped) {
                reply.truncated = true;
                reply.reason = Some("file exceeds diff input limit");
            } else if untracked {
                let after = after.ok_or(Failure::Stale)?;
                let text = after.text().ok_or(Failure::Git)?;
                let mut patch = patch_header(entry, side, false, true);
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
                reply.set_patch(&patch, false);
            } else {
                let before = before.as_ref().map(|b| b.bytes.as_slice());
                let after = after.as_ref().map(|b| b.bytes.as_slice());
                let modes = before_file
                    .map(|file| file.mode)
                    .zip(after_mode)
                    .filter(|(old, new)| old != new);
                if before != after || entry.rename_source(side).is_some() || modes.is_some() {
                    let mut patch = patch_header(entry, side, before.is_some(), after.is_some());
                    if let Some((old, new)) = modes {
                        patch.push_str(&format!("old mode {old}\nnew mode {new}\n"));
                    }
                    let mut clipped = false;
                    if before != after {
                        let input = self.repo.hunks(before, after)?;
                        clipped = input.capped;
                        if let Some(hunks) = hunk_text(&input)? {
                            patch.push_str(hunks);
                        }
                    }
                    reply.set_patch(&patch, clipped);
                } else {
                    reply.reason = Some("no content changes");
                }
            }
        }
        Ok(Json(reply))
    }
}

#[derive(Clone, Copy)]
enum LineStats {
    Text(usize, usize),
    Binary,
}

impl LineStats {
    fn counts(stats: Option<Self>) -> (Option<usize>, Option<usize>) {
        match stats {
            Some(Self::Text(additions, deletions)) => (Some(additions), Some(deletions)),
            _ => (None, None),
        }
    }
}

#[derive(Clone, Copy)]
struct SideStats {
    staged: Option<LineStats>,
    worktree: Option<LineStats>,
}

impl SideStats {
    fn get(self, side: DiffSide) -> Option<LineStats> {
        match side {
            DiffSide::Staged => self.staged,
            DiffSide::Worktree => self.worktree,
        }
    }
}

#[derive(Clone, Copy)]
struct Numstat<'a> {
    old_path: Option<&'a [u8]>,
    stats: LineStats,
}

impl Numstat<'_> {
    fn line_stats(self) -> Option<LineStats> {
        if self.old_path.is_some() && matches!(self.stats, LineStats::Text(0, 0)) {
            None
        } else {
            Some(self.stats)
        }
    }

    fn text_counts(self) -> Option<(usize, usize)> {
        if let LineStats::Text(additions, deletions) = self.line_stats()? {
            Some((additions, deletions))
        } else {
            None
        }
    }
}

fn parse_numstats(bytes: &[u8]) -> Result<FastMap<&[u8], Option<Numstat<'_>>>> {
    let mut records = bytes
        .split_inclusive(|byte| *byte == 0)
        .map(|record| record.strip_suffix(b"\0").ok_or(Failure::Git));
    let mut rows = FastMap::default();
    while let Some(record) = records.next() {
        let mut fields = record?.splitn(3, |byte| *byte == b'\t');
        let (Some(added), Some(deleted), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(Failure::Git);
        };
        let stats = match (added, deleted) {
            (b"-", b"-") => LineStats::Binary,
            (added, deleted) => LineStats::Text(
                as_git_str(added)?.parse().map_err(|_| Failure::Git)?,
                as_git_str(deleted)?.parse().map_err(|_| Failure::Git)?,
            ),
        };
        // With -z the empty path is a marker; the next two fields are raw paths.
        let (old_path, path) = if path.is_empty() {
            let mut next_path = || {
                records
                    .next()
                    .transpose()?
                    .filter(|path| !path.is_empty())
                    .ok_or(Failure::Git)
            };
            (Some(next_path()?), next_path()?)
        } else {
            (None, path)
        };
        // Multiple records for one destination are ambiguous.
        rows.entry(path)
            .and_modify(|row| *row = None)
            .or_insert(Some(Numstat { old_path, stats }));
    }
    Ok(rows)
}

fn untracked_line_stats(mut file: File, stamp: Stamp, git: Git) -> Result<LineStats> {
    if stamp.0[2] > UNTRACKED_STAT_LIMIT as u64 {
        return Ok(LineStats::Binary);
    }
    let mut buffer = [0; 8192];
    let mut pending = 0;
    let mut bytes = 0;
    let mut lines = 0;
    let mut last = b'\n';
    let stats = loop {
        git.remaining()?;
        let room = (UNTRACKED_STAT_LIMIT + 1 - bytes).min(buffer.len() - pending);
        let read = match file.read(&mut buffer[pending..pending + room]) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if read == 0 {
            break if pending == 0 {
                LineStats::Text(lines + usize::from(last != b'\n'), 0)
            } else {
                LineStats::Binary
            };
        }
        bytes += read;
        let len = pending + read;
        let chunk = &buffer[pending..len];
        if bytes > UNTRACKED_STAT_LIMIT || chunk.contains(&0) {
            break LineStats::Binary;
        }
        lines += chunk.iter().filter(|&&byte| byte == b'\n').count();
        last = buffer[len - 1];
        pending = match std::str::from_utf8(&buffer[..len]) {
            Ok(_) => 0,
            Err(error) if error.error_len().is_none() => {
                // An incomplete UTF-8 sequence may continue in the next chunk.
                let tail = error.valid_up_to();
                buffer.copy_within(tail..len, 0);
                len - tail
            }
            Err(_) => break LineStats::Binary,
        };
    };
    if Stamp::from(&file.metadata()?) != stamp {
        return Err(Failure::Stale);
    }
    Ok(stats)
}

fn file_id(scope_key: &[u8], path: &Path) -> String {
    let mut hash = Sha256::new();
    hash_piece(&mut hash, scope_key);
    hash_piece(&mut hash, path.as_os_str().as_bytes());
    URL_SAFE_NO_PAD.encode(hash.finalize())
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

fn hunk_text(input: &Bounded) -> Result<Option<&str>> {
    let text = complete_lines(input.text().ok_or(Failure::Git)?, input.capped);
    Ok(text
        .split_inclusive('\n')
        .scan(0, |offset, line| {
            let start = *offset;
            *offset += line.len();
            Some((start, line))
        })
        .find_map(|(start, line)| line.starts_with("@@ ").then(|| &text[start..])))
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

fn patch_preamble(path: &Path, old_path: Option<&Path>) -> (String, String, String) {
    let a = quote_path("a/", old_path.unwrap_or(path));
    let b = quote_path("b/", path);
    let mut patch = format!("diff --git {a} {b}\n");
    if let Some(old_path) = old_path {
        patch.push_str(&format!(
            "rename from {}\nrename to {}\n",
            quote_path("", old_path),
            quote_path("", path)
        ));
    }
    (patch, a, b)
}

fn append_file_headers(patch: &mut String, old: Option<&str>, new: Option<&str>) {
    let old = old.unwrap_or("/dev/null");
    let new = new.unwrap_or("/dev/null");
    patch.push_str(&format!("--- {old}\n+++ {new}\n"));
}

fn patch_header(entry: &Entry, side: DiffSide, before: bool, after: bool) -> String {
    let (mut patch, a, b) = patch_preamble(&entry.path, entry.rename_source(side));
    append_file_headers(
        &mut patch,
        before.then_some(a.as_str()),
        after.then_some(b.as_str()),
    );
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
