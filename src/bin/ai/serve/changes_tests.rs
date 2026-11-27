//! Route tests use isolated worktrees and synthetic session metadata.

use super::*;
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use hyper::service::Service;
use hyper_util::service::TowerToHyperService;
use serde_json::Value;

use crate::ai::serve::{SessionCwd, SessionTitleOrigin, run_session_cd, server_info};

struct Fixture {
    scratch: Scratch,
    repo: PathBuf,
    state: ServeState,
}

impl Fixture {
    fn new() -> Self {
        Self::with_object_format("sha1")
    }

    fn with_object_format(format: &str) -> Self {
        let fixture = Self::without_repository();
        fixture.git(&[
            "init",
            "--quiet",
            "--template=",
            &format!("--object-format={format}"),
        ]);
        fixture
    }

    fn without_repository() -> Self {
        let scratch = Scratch::new("a-serve-changes-test-").unwrap();
        let repo = scratch.0.join("repo");
        fs::create_dir(&repo).unwrap();
        let state = ServeState {
            history_file: scratch.0.join("history.sqlite"),
            token: "test-token".to_string(),
            default_add_dirs: Default::default(),
            cwds: Default::default(),
            groups: Default::default(),
            locks: Default::default(),
            open_sessions: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
            events: Default::default(),
        };
        Self {
            scratch,
            repo,
            state,
        }
    }

    fn git(&self, args: &[&str]) {
        Self::git_at(&self.repo, &[], args);
    }

    fn git_at(cwd: &Path, env: &[(&str, &str)], args: &[&str]) -> Vec<u8> {
        let git = Git {
            deadline: Instant::now() + TIME_LIMIT,
        };
        let out = crate::fork_guard::output(
            git.command(cwd)
                .envs(env.iter().copied())
                .args([
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                ])
                .args(args),
        )
        .unwrap();
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    fn oid(&self, revision: &str) -> String {
        let output = Self::git_at(
            &self.repo,
            &[],
            &["rev-parse", "--verify", "--end-of-options", revision],
        );
        as_git_str(trimmed_line(&output)).unwrap().to_owned()
    }

    fn commit_named(&self, subject: &str) -> String {
        self.git(&["add", "--all", "--", "."]);
        self.git(&["commit", "--quiet", "--allow-empty", "-m", subject]);
        self.oid("HEAD")
    }

    fn commit_at(cwd: &Path, seconds: u64) {
        Self::git_at(
            cwd,
            &[("GIT_COMMITTER_DATE", &format!("@{seconds} +0000"))],
            &[
                "commit",
                "--allow-empty",
                "--quiet",
                "--date=@1000000000 +0000",
                "-m",
                "recency fixture",
            ],
        );
    }

    fn file(&self, relative: &str, content: impl AsRef<[u8]>) {
        let path = self.repo.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    fn commit(&self) {
        self.git(&["add", "--", "."]);
        self.git(&["commit", "--quiet", "-m", "initial"]);
    }

    fn worktree(&self, path: PathBuf) -> PathBuf {
        self.git(&[
            "worktree",
            "add",
            "--quiet",
            "--detach",
            path.to_str().unwrap(),
        ]);
        path.canonicalize().unwrap()
    }

    fn create_session(&self, id: &str) {
        SessionStore::new(&self.state.history_file)
            .write_session_title_with_origin(id, "Changes test", SessionTitleOrigin::User)
            .unwrap();
    }

    fn session(&self, id: &str, cwd: PathBuf) {
        self.create_session(id);
        self.state.cwds.lock().unwrap().insert(
            id.to_string(),
            SessionCwd {
                current: cwd,
                previous: None,
            },
        );
    }

    async fn get(&self, uri: &str, authenticated: bool) -> (StatusCode, Value, usize) {
        self.request("GET", uri, authenticated, None).await
    }

    async fn post_commit(&self, input: Value, authenticated: bool) -> (StatusCode, Value, usize) {
        self.request("POST", "/sessions/one/commit", authenticated, Some(input))
            .await
    }

    async fn request(
        &self,
        method: &str,
        uri: &str,
        authenticated: bool,
        input: Option<Value>,
    ) -> (StatusCode, Value, usize) {
        // Concurrent fixtures share the route's process-wide request limit.
        static REQUEST: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let _request = REQUEST.lock().await;
        let app =
            routes(Router::new().route("/info", get(server_info))).with_state(self.state.clone());
        let mut request = Request::builder().method(method).uri(uri);
        if authenticated {
            request = request.header(header::AUTHORIZATION, "Bearer test-token");
        }
        let body = if let Some(input) = input {
            request = request.header(header::CONTENT_TYPE, "application/json");
            Body::from(input.to_string())
        } else {
            Body::empty()
        };
        let response = TowerToHyperService::new(app)
            .call(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        if status == StatusCode::OK && uri != "/info" {
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "private, no-store"
            );
        }
        let bytes = to_bytes(response.into_body(), DETAIL_LIMIT).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap(), bytes.len())
    }

    async fn commits(&self, suffix: &str) -> (Value, usize) {
        let (status, body, size) = self
            .get(&format!("/sessions/one/changes/commits{suffix}"), true)
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        (body, size)
    }

    async fn list(&self, id: &str) -> (Value, usize) {
        let (status, value, size) = self.get(&format!("/sessions/{id}/changes"), true).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        (value, size)
    }

    async fn detail(&self, id: &str, list: &Value, path: &str) -> (Value, usize) {
        self.detail_with_side(id, list, path, "").await
    }

    async fn detail_with_side(
        &self,
        id: &str,
        list: &Value,
        path: &str,
        query: &str,
    ) -> (Value, usize) {
        let (status, detail, size) = self
            .get(
                &format!(
                    "/sessions/{id}/changes/{}?revision={}{query}",
                    row(list, path)["id"].as_str().unwrap(),
                    list["revision"].as_str().unwrap(),
                ),
                true,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{detail}");
        assert_eq!(detail["revision"], list["revision"]);
        assert_eq!(detail["file_id"], row(list, path)["id"]);
        assert_eq!(detail["path"], path);
        (detail, size)
    }
}

fn rows(body: &Value) -> &[Value] {
    body["files"].as_array().unwrap()
}

fn row<'a>(body: &'a Value, path: &str) -> &'a Value {
    rows(body)
        .iter()
        .find(|row| row["path"] == path)
        .unwrap_or_else(|| panic!("missing {path}: {body}"))
}

#[tokio::test]
async fn changes_scope_discovers_filtered_counts() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "base\n");
    fixture.commit();
    let linked = fixture.worktree(fixture.repo.join(".worktrees/feature branch"));
    fs::write(linked.join("note.txt"), "edited\n").unwrap();
    fs::write(linked.join("new.txt"), "untracked\n").unwrap();
    fixture.session("one", fixture.repo.clone());

    let uri = "/sessions/one/changes/scopes";
    let (status, response, _) = fixture.get(uri, true).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let scopes = response["scopes"].as_array().unwrap();
    assert_eq!(response["source"], "git");
    assert_eq!(response["current_path"], scopes[0]["path"]);
    assert_eq!(response["truncated"], false);
    assert!(response["reason"].is_null());
    assert_eq!(scopes.len(), 2, "{response}");
    assert_eq!(
        scopes[0],
        serde_json::json!({
            "id": "current", "label": "repo", "path": fixture.repo,
            "current": true, "changed": false, "files": 0,
        })
    );
    let scope_id = scopes[1]["id"].as_str().unwrap();
    assert!(scope_id.strip_prefix("wt-").is_some_and(|hash| {
        hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    }));
    assert_eq!(
        scopes[1],
        serde_json::json!({
            "id": scope_id, "label": "feature branch", "path": linked,
            "current": false, "changed": true, "files": 2,
        })
    );
    assert_eq!(fixture.get(uri, false).await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn changes_scope_orders_by_committer_time_with_mtime_fallback() {
    let fixture = Fixture::new();
    Fixture::commit_at(&fixture.repo, 1_000_000_000);
    for (label, seconds) in [
        ("a-oldest", 1_100_000_000),
        ("z-newest", 1_500_000_000),
        ("m-tied", 1_500_000_000),
    ] {
        let worktree = fixture.worktree(fixture.repo.join(".worktrees").join(label));
        Fixture::commit_at(&worktree, seconds);
    }
    let unborn = fixture.worktree(fixture.repo.join(".worktrees/b-unborn"));
    Fixture::git_at(&unborn, &[], &["symbolic-ref", "HEAD", "refs/heads/unborn"]);
    let mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_400_000_000);
    File::open(&unborn)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(mtime))
        .unwrap();
    fixture.session("one", fixture.repo.clone());

    for _ in 0..2 {
        let (status, catalog, _) = fixture.get("/sessions/one/changes/scopes", true).await;
        assert_eq!(status, StatusCode::OK, "{catalog}");
        assert_eq!(catalog["truncated"], false, "{catalog}");
        let scopes = catalog["scopes"].as_array().unwrap();
        assert_eq!(scopes[0]["id"], CURRENT_SCOPE);
        let labels: Vec<_> = scopes
            .iter()
            .map(|scope| scope["label"].as_str().unwrap())
            .collect();
        assert_eq!(
            labels,
            ["repo", "m-tied", "z-newest", "b-unborn", "a-oldest"]
        );
    }
}

#[test]
fn changes_scope_orders_unknown_recency_last() {
    let row = |id: &str, label: &str, updated_at| ScopeRow {
        scope: ScopeInfo {
            id: id.to_string(),
            label: label.to_string(),
            path: label.to_string(),
            current: false,
        },
        changed: false,
        files: 0,
        updated_at,
    };
    let mut scopes = [
        row("wt-2", "unknown", None),
        row("wt-0", "unknown", None),
        row("wt-1", "z-known", Some(SystemTime::UNIX_EPOCH)),
    ];
    ScopeRow::sort_by_recency(&mut scopes);
    assert_eq!(
        scopes.each_ref().map(|row| row.scope.id.as_str()),
        ["wt-1", "wt-0", "wt-2"]
    );
}

#[tokio::test]
async fn changes_scope_excludes_external_and_unavailable_worktrees() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "base\n");
    fixture.commit();
    let outside = fixture.worktree(fixture.scratch.0.join("repo-outside"));
    let missing = fixture.worktree(fixture.repo.join(".worktrees/missing"));
    fs::remove_dir_all(&missing).unwrap();
    let escape = fixture.worktree(fixture.repo.join(".worktrees/escape"));
    let moved = fixture.scratch.0.join("moved-worktree");
    fs::rename(&escape, &moved).unwrap();
    symlink(&moved, &escape).unwrap();
    let replaced = fixture.worktree(fixture.repo.join(".worktrees/replaced"));
    fs::remove_file(replaced.join(".git")).unwrap();
    fixture.git(&["init", "--quiet", "--template=", replaced.to_str().unwrap()]);
    fixture.session("one", fixture.repo.clone());

    let (status, response, _) = fixture.get("/sessions/one/changes/scopes", true).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        response["scopes"].as_array().unwrap().len(),
        1,
        "{response}"
    );
    assert_eq!(response["scopes"][0]["id"], "current");
    assert_eq!(response["truncated"], false);
    for path in [outside, missing, escape, moved, replaced] {
        let uri = format!("/sessions/one/changes?scope={}", worktree_scope_id(&path));
        let (status, response, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        assert_eq!(
            response,
            serde_json::json!({"error": "unknown changes scope", "code": "unknown_scope"})
        );
    }
}

#[tokio::test]
async fn changes_scope_list_and_detail_are_bound_to_selection() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "before\n");
    fixture.commit();
    let linked = fixture.worktree(fixture.repo.join(".worktrees/feature"));
    fs::write(linked.join("note.txt"), "after\n").unwrap();
    fixture.session("one", fixture.repo.clone());

    let uri = "/sessions/one/changes";
    let (status, scopes, _) = fixture.get(&format!("{uri}/scopes"), true).await;
    assert_eq!(status, StatusCode::OK, "{scopes}");
    let scope_id = scopes["scopes"][1]["id"].as_str().unwrap();
    assert_eq!(scope_id, worktree_scope_id(&linked));
    let (status, list, _) = fixture.get(&format!("{uri}?scope={scope_id}"), true).await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(list["source"], "working_tree");
    assert_eq!(list["root_label"], "feature");
    assert_eq!(rows(&list).len(), 1);
    assert_eq!(
        list["scope"],
        serde_json::json!({
            "id": scope_id, "label": "feature", "path": linked, "current": false,
        })
    );
    let (current, _) = fixture.list("one").await;
    let (status, explicit, _) = fixture.get(&format!("{uri}?scope=current"), true).await;
    assert_eq!(status, StatusCode::OK, "{explicit}");
    assert_eq!(explicit, current);
    assert_eq!(current["scope"]["id"], "current");
    assert!(rows(&current).is_empty());

    let file_id = row(&list, "note.txt")["id"].as_str().unwrap();
    let revision = list["revision"].as_str().unwrap();
    let detail_uri = format!("{uri}/{file_id}?revision={revision}");
    let (status, detail, _) = fixture
        .get(&format!("{detail_uri}&scope={scope_id}"), true)
        .await;
    assert_eq!(status, StatusCode::OK, "{detail}");
    assert_eq!(detail["revision"], revision);
    assert_eq!(detail["file_id"], file_id);
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-before\n+after\n")
    );
    // The same file contents in another scope must not validate the token.
    fixture.file("note.txt", "after\n");
    for uri in [detail_uri.clone(), format!("{detail_uri}&scope=current")] {
        let (status, response, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {response}");
        assert_eq!(response["error"], "changes revision is missing or stale");
    }
    for uri in [
        format!("{uri}?scope=wt-0000000000000000"),
        format!("{uri}?scope=%2Ftmp"),
        format!("{uri}?scope=current&scope={scope_id}"),
        format!("{detail_uri}&scope=wt-0000000000000000"),
        format!("{detail_uri}&scope=current&scope={scope_id}"),
    ] {
        let (status, response, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        assert_eq!(
            response,
            serde_json::json!({"error": "unknown changes scope", "code": "unknown_scope"})
        );
    }
}

#[tokio::test]
async fn changes_scope_keeps_a_nested_session_cwd_narrow() {
    let fixture = Fixture::new();
    fixture.file("root.txt", "root\n");
    fixture.file("sub/note.txt", "base\n");
    fixture.commit();
    let linked = fixture.worktree(fixture.repo.join(".worktrees/feature"));
    fixture.file("root.txt", "edited main\n");
    fs::write(linked.join("root.txt"), "edited linked root\n").unwrap();
    fs::write(linked.join("sub/note.txt"), "edited linked subdir\n").unwrap();
    fixture.session("one", linked.join("sub"));

    let uri = "/sessions/one/changes";
    let (status, catalog, _) = fixture.get(&format!("{uri}/scopes"), true).await;
    assert_eq!(status, StatusCode::OK, "{catalog}");
    let scopes = catalog["scopes"].as_array().unwrap();
    let main_id = worktree_scope_id(&fixture.repo);
    let (main_status, main, _) = fixture.get(&format!("{uri}?scope={main_id}"), true).await;
    assert_eq!(
        (scopes.len(), main_status),
        (1, StatusCode::BAD_REQUEST),
        "catalog: {catalog}; main checkout selection: {main}"
    );
    let unknown_scope =
        serde_json::json!({"error": "unknown changes scope", "code": "unknown_scope"});
    assert_eq!(main, unknown_scope);
    assert_eq!(scopes[0]["id"], CURRENT_SCOPE);
    assert_eq!(scopes[0]["path"], linked.join("sub").to_str().unwrap());
    assert_eq!(scopes[0]["files"], 1);

    let (current, _) = fixture.list("one").await;
    assert_eq!(rows(&current).len(), 1);
    let (detail, _) = fixture.detail("one", &current, "note.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-base\n+edited linked subdir\n")
    );
    let file_id = detail["file_id"].as_str().unwrap();
    let revision = detail["revision"].as_str().unwrap();
    for uri in [
        format!("{uri}?scope={}", worktree_scope_id(&linked)),
        format!("{uri}/{file_id}?revision={revision}&scope={main_id}"),
    ] {
        let (status, response, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        assert_eq!(response, unknown_scope);
    }
}

#[tokio::test]
async fn changes_scope_caps_discovery() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "base\n");
    fixture.commit();
    for index in 0..WORKTREE_LIMIT {
        fixture.worktree(fixture.repo.join(format!(".worktrees/tree-{index:02}")));
    }
    fixture.session("one", fixture.repo.clone());

    let (status, response, _) = fixture.get("/sessions/one/changes/scopes", true).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let scopes = response["scopes"].as_array().unwrap();
    assert_eq!(scopes[0]["id"], "current");
    assert_eq!(scopes.len(), WORKTREE_LIMIT, "{response}");
    assert_eq!(response["truncated"], true);
    assert_eq!(response["reason"], SCOPES_TRUNCATED);
}

#[tokio::test]
async fn changes_scope_times_out_slow_worktrees_without_losing_others() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "base\n");
    fixture.commit();
    fixture.worktree(fixture.repo.join(".worktrees/a-slow"));
    let healthy = fixture.worktree(fixture.repo.join(".worktrees/z-healthy"));
    fixture.git(&["config", "extensions.worktreeConfig", "true"]);
    let fifo = fixture.repo.join(".git/worktrees/a-slow/config.worktree");
    let fifo = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: fifo is a live, NUL-terminated path inside the fixture.
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    fixture.session("one", fixture.repo.clone());

    let (status, response, _) = fixture.get("/sessions/one/changes/scopes", true).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let scopes = response["scopes"].as_array().unwrap();
    assert_eq!(scopes.len(), 2, "{response}");
    assert_eq!(scopes[0]["id"], "current");
    assert_eq!(scopes[1]["path"], healthy.to_str().unwrap());
    assert_eq!(response["truncated"], true);
    assert_eq!(response["reason"], SCOPES_TIMEOUT);
}

#[tokio::test]
async fn changes_route_uses_the_cwd_reported_by_commands() {
    const MODE: &str = "A_SERVE_CHANGES_CWD_TEST_MODE";
    if !matches!(std::env::var(MODE).as_deref(), Ok("child")) {
        // Set the daemon cwd in a child so parallel tests keep their own cwd.
        let daemon = Fixture::new();
        daemon.file("outside.txt", "outside the daemon cwd\n");
        daemon.file("sub/inside.txt", "inside the daemon cwd\n");
        let test = format!(
            "{}::changes_route_uses_the_cwd_reported_by_commands",
            module_path!().split_once("::").unwrap().1
        );
        let output = crate::fork_guard::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &test])
                .env(MODE, "child")
                .current_dir(daemon.repo.join("sub")),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let daemon_cwd = std::env::current_dir().unwrap();
    let fixture = Fixture::new();
    fixture.file("stored.txt", "inside the stored cwd\n");
    let alias = fixture.scratch.0.join("repo-alias");
    symlink(&fixture.repo, &alias).unwrap();
    fixture.create_session("unset");
    fixture.session("missing", fixture.repo.join("removed"));
    fixture.session("file", fixture.repo.join("stored.txt"));
    fixture.session("valid", fixture.repo.clone());
    fixture.session("alias", alias.clone());
    for (id, expected, file) in [
        ("unset", &daemon_cwd, "inside.txt"),
        ("missing", &daemon_cwd, "inside.txt"),
        ("file", &daemon_cwd, "inside.txt"),
        ("valid", &fixture.repo, "stored.txt"),
        ("alias", &alias, "stored.txt"),
    ] {
        let expected = expected.display().to_string();
        for command in ["/pwd", "/cd"] {
            let shown = run_session_cd(&fixture.state, id, command).unwrap();
            assert!(shown.stderr.is_empty(), "{}", shown.stderr);
            assert_eq!(
                shown.stdout,
                format!("Working directory: {expected}\n"),
                "{id}: {command}"
            );
        }
        let uri = format!("/sessions/{id}/changes");
        let (status, list, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {list}");
        assert_eq!(list["scope"]["path"], expected, "{id}: {list}");
        assert_eq!(rows(&list).len(), 1, "{id}: {list}");
        row(&list, file);
    }

    // A valid directory outside Git must not fall back to the daemon's repo.
    fixture.session("no-git", fixture.scratch.0.clone());
    let (status, body, _) = fixture
        .get("/sessions/no-git/changes/unused?revision=unused", true)
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(
        body,
        serde_json::json!({"error": "not a Git repository", "code": "not_a_git_repository"})
    );
}

#[tokio::test]
async fn changes_route_auth_and_revision_isolation() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "first\n");
    fixture.session("one", fixture.repo.clone());
    fixture.session("two", fixture.repo.clone());
    fixture.session("no-git", fixture.scratch.0.clone());
    let alias = fixture.scratch.0.join("alias");
    fs::create_dir(&alias).unwrap();
    symlink(fixture.repo.join(".git"), alias.join(".git")).unwrap();
    fixture.session("aliased", alias);
    for path in [
        "/sessions/one/changes",
        "/sessions/one/changes/no-id?revision=a&revision=b",
    ] {
        assert_eq!(fixture.get(path, false).await.0, StatusCode::UNAUTHORIZED);
    }
    assert_eq!(
        fixture.get("/sessions/absent/changes", true).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        fixture.get("/sessions/absent/changes/no-id", true).await.0,
        StatusCode::NOT_FOUND
    );
    let (status, unavailable, _) = fixture.get("/sessions/no-git/changes", true).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unavailable["source"], "plain");
    assert_eq!(unavailable["git_available"], false);
    assert_eq!(unavailable["reason"], "not_a_repository");
    assert_eq!(rows(&unavailable).len(), 1, "{unavailable}");
    row(&unavailable, "repo/note.txt");
    assert_eq!(
        fixture
            .get("/sessions/no-git/changes/no-id?revision=x", true)
            .await
            .0,
        StatusCode::CONFLICT
    );
    for uri in [
        "/sessions/aliased/changes",
        "/sessions/aliased/changes/no-id?revision=x",
    ] {
        assert_eq!(fixture.get(uri, true).await.0, StatusCode::CONFLICT);
    }
    let (info_status, info, _) = fixture.get("/info", true).await;
    assert_eq!(info_status, StatusCode::OK);
    assert_eq!(info["features"]["changes"], true);
    let (one, _) = fixture.list("one").await;
    let (two, _) = fixture.list("two").await;
    let first_revision = one["revision"].as_str().unwrap();
    let second_revision = two["revision"].as_str().unwrap();
    let first_file = row(&one, "note.txt")["id"].as_str().unwrap();
    assert_ne!(first_revision, second_revision);
    assert_ne!(first_file, row(&two, "note.txt")["id"]);
    assert_eq!(
        fixture
            .get(&format!("/sessions/one/changes/{first_file}"), true)
            .await
            .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/two/changes/{first_file}?revision={second_revision}"),
                true
            )
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/note.txt?revision={first_revision}"),
                true
            )
            .await
            .0,
        StatusCode::NOT_FOUND
    );

    fixture.file("note.txt", "second version\n");
    let (fresh, _) = fixture.list("one").await;
    assert_ne!(fresh["revision"], first_revision);
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/{first_file}?revision={first_revision}"),
                true
            )
            .await
            .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        fixture
            .get(
                &format!(
                    "/sessions/one/changes/{first_file}?revision={}",
                    fresh["revision"].as_str().unwrap()
                ),
                true
            )
            .await
            .0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn changes_route_scoped_worktree_patches() {
    let mut fixture = Fixture::new();
    fixture.state.history_file = fixture.repo.join("work/runtime.sqlite");
    fixture.state.default_add_dirs = std::sync::Arc::new(vec![fixture.repo.clone()]);
    fixture.file("work/modified.txt", "committed\n");
    fixture.file("work/deleted.txt", "delete me\n");
    fixture.file("work/old name.txt", "rename me\n");
    fixture.file("work/redirect/nested.txt", "original file\n");
    fixture.file("work/deleted.bin", [0, 255, 10]);
    fixture.file("sibling.txt", "before outside\n");
    fixture.file("outside.txt", "outside rename secret\n");
    fixture.file("offsite/nested.txt", "outside symlink secret\n");
    fixture.file(".gitignore", "work/ignored*\n");
    fixture.file("work/.agent_tmp/private.txt", "private rename secret\n");
    fixture.git(&["add", "-f", "--", "work/.agent_tmp/private.txt"]);
    fixture.commit();
    fixture.session("one", fixture.repo.join("work"));

    fixture.file("work/modified.txt", "staged\n");
    fixture.file("work/fresh.txt", "new and staged\n");
    fixture.git(&["add", "--", "work/modified.txt", "work/fresh.txt"]);
    fixture.file("work/modified.txt", "working\n");
    fixture.file("work/fresh.txt", "new and edited\n");
    fs::remove_file(fixture.repo.join("work/deleted.txt")).unwrap();
    fs::remove_file(fixture.repo.join("work/deleted.bin")).unwrap();
    fixture.git(&["mv", "--", "work/old name.txt", "work/new name.txt"]);
    fixture.git(&["mv", "--", "outside.txt", "work/incoming.txt"]);
    fixture.file("work/incoming.txt", "now inside\n");
    fixture.git(&[
        "mv",
        "--",
        "work/.agent_tmp/private.txt",
        "work/private-incoming.txt",
    ]);
    fixture.file("work/private-incoming.txt", "now public\n");
    fixture.file("work/new.txt", "new content");
    fixture.file("work/invalid.bin", [255, 10]);
    fixture.file("work/.agent_tmp/data", "temp secret\n");
    fixture.file("work/runtime.sqlite", "history secret\n");
    fixture.file("work/runtime.sqlite-wal", "journal secret\n");
    fixture.file("work/serve-cwds.json", "sidecar secret\n");
    fixture.file("work/ignored.txt", "ignored secret\n");
    fixture.file("sibling.txt", "outside sibling secret\n");
    symlink(
        fixture.repo.join("sibling.txt"),
        fixture.repo.join("work/link.txt"),
    )
    .unwrap();
    fs::remove_dir_all(fixture.repo.join("work/redirect")).unwrap();
    symlink(
        fixture.repo.join("offsite"),
        fixture.repo.join("work/redirect"),
    )
    .unwrap();

    let (list, _) = fixture.list("one").await;
    assert_eq!(list["source"], "working_tree");
    assert_eq!(list["root_label"], "work");
    assert_eq!(list["truncated"], false);
    assert!(list["reason"].is_null());
    assert_eq!(
        rows(&list)
            .iter()
            .map(|row| row["path"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "deleted.bin",
            "deleted.txt",
            "fresh.txt",
            "incoming.txt",
            "invalid.bin",
            "link.txt",
            "modified.txt",
            "new name.txt",
            "new.txt",
            "private-incoming.txt",
            "redirect",
        ]
    );
    for (path, status) in [
        ("deleted.txt", "deleted"),
        ("modified.txt", "modified"),
        ("new name.txt", "renamed"),
        ("new.txt", "added"),
        ("private-incoming.txt", "renamed"),
    ] {
        assert_eq!(row(&list, path)["status"], status);
    }
    let renamed = row(&list, "new name.txt");
    assert_eq!(renamed["old_path"], "old name.txt");
    assert_eq!(renamed["staged_status"], "R");
    for key in [
        "additions",
        "deletions",
        "staged_additions",
        "staged_deletions",
    ] {
        assert_eq!(renamed.get(key), Some(&Value::Null));
    }
    assert!(row(&list, "incoming.txt")["old_path"].is_null());
    let redacted = row(&list, "private-incoming.txt");
    assert!(redacted["old_path"].is_null());
    for key in [
        "additions",
        "deletions",
        "staged_additions",
        "staged_deletions",
    ] {
        assert_eq!(redacted.get(key), Some(&Value::Null));
    }

    let (detail, _) = fixture.detail("one", &list, "modified.txt").await;
    assert_eq!(detail["format"], "unified");
    let patch = detail["diff"].as_str().unwrap();
    assert!(patch.contains("--- a/modified.txt\n+++ b/modified.txt\n@@ "));
    assert_eq!(detail["side"], "worktree");
    assert!(patch.contains("-staged\n+working\n"));
    assert!(!patch.contains("committed"));
    let (detail, _) = fixture.detail("one", &list, "deleted.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("+++ /dev/null\n@@ ")
    );
    let (detail, _) = fixture.detail("one", &list, "new.txt").await;
    assert!(detail["diff"].as_str().unwrap().contains("--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,1 @@\n+new content\n\\ No newline at end of file\n"));
    let (detail, _) = fixture.detail("one", &list, "new name.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("rename from old name.txt\nrename to new name.txt\n")
    );
    let (detail, _) = fixture.detail("one", &list, "incoming.txt").await;
    let patch = detail["diff"].as_str().unwrap();
    assert!(patch.contains("+now inside"));
    assert!(!patch.contains("outside"));
    assert!(row(&list, "incoming.txt")["deletions"].is_null());
    let (detail, _) = fixture.detail("one", &list, "fresh.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-new and staged\n+new and edited\n")
    );

    for (path, reason) in [
        ("invalid.bin", "binary"),
        ("deleted.bin", "binary"),
        ("link.txt", "symlink"),
        ("redirect", "symlink"),
    ] {
        assert_eq!(row(&list, path)["binary"], reason == "binary");
        let (detail, _) = fixture.detail("one", &list, path).await;
        assert_eq!(detail["format"], "none");
        assert_eq!(detail["diff"], "");
        assert_eq!(detail["reason"], reason);
    }
}

#[tokio::test]
async fn changes_route_counts_line_deltas_for_reordered_lines() {
    let fixture = Fixture::new();
    let keep = "keep\n".repeat(130);
    let moved = "move\n".repeat(66);
    let before = format!("before\n{keep}{moved}remove\n");
    let after = format!("after\n{moved}{keep}insert\n");
    fixture.file("reordered.txt", before);
    fixture.commit();
    fixture.session("one", fixture.repo.clone());
    fixture.file("reordered.txt", after);
    let (list, _) = fixture.list("one").await;
    let file = row(&list, "reordered.txt");
    assert_eq!(
        (file["additions"].as_u64(), file["deletions"].as_u64()),
        (Some(68), Some(68))
    );
}

#[tokio::test]
async fn changes_route_reports_line_deltas_for_each_side() {
    let fixture = Fixture::new();
    fixture.file("modified.txt", "keep\r\nremove\r\nsame\r\nlast");
    fixture.file("deleted.txt", "removed\r\nlast\r");
    fixture.file("old name.txt", "one\ntwo\nthree\n");
    fixture.file("newline.txt", "last");
    fixture.commit();
    fixture.session("one", fixture.repo.clone());

    fixture.file("modified.txt", "staged\n");
    fixture.file("added.txt", "first\r\nsecond\rlast");
    fixture.git(&["add", "--", "modified.txt", "added.txt"]);
    fixture.file(
        "modified.txt",
        "keep\r\nreplace\r\nsame\r\ninserted\r\nlast",
    );
    fs::remove_file(fixture.repo.join("deleted.txt")).unwrap();
    fixture.git(&["mv", "--", "old name.txt", "new name.txt"]);
    fixture.file("new name.txt", "one\nfour\nthree\nfive\n");
    fixture.file("newline.txt", "last\n");
    fixture.file("untracked.txt", "first\r\nsecond\rlast");

    let (list, _) = fixture.list("one").await;
    let expected = [
        (
            "modified.txt",
            "modified",
            Some("M"),
            Some("M"),
            Some((1, 4)),
            Some((5, 1)),
        ),
        (
            "deleted.txt",
            "deleted",
            None,
            Some("D"),
            None,
            Some((0, 2)),
        ),
        (
            "new name.txt",
            "renamed",
            Some("R"),
            Some("M"),
            None,
            Some((2, 1)),
        ),
        (
            "newline.txt",
            "modified",
            None,
            Some("M"),
            None,
            Some((1, 1)),
        ),
        ("added.txt", "added", Some("A"), None, Some((2, 0)), None),
        (
            "untracked.txt",
            "added",
            None,
            Some("A"),
            None,
            Some((2, 0)),
        ),
    ];
    for (path, status, staged_status, worktree_status, staged, worktree) in expected {
        let entry = row(&list, path);
        assert_eq!(entry["binary"], false, "{entry}");
        assert_eq!(entry["status"], status, "{entry}");
        assert_eq!(
            entry["staged_status"],
            serde_json::json!(staged_status),
            "{entry}"
        );
        assert_eq!(
            entry["worktree_status"],
            serde_json::json!(worktree_status),
            "{entry}"
        );
        for (fields, counts) in [
            (["staged_additions", "staged_deletions"], staged),
            (["additions", "deletions"], worktree),
        ] {
            let expected = counts.map_or([Value::Null, Value::Null], |(a, d)| [a.into(), d.into()]);
            for (key, expected) in fields.into_iter().zip(expected) {
                assert_eq!(entry.get(key), Some(&expected), "{entry}");
            }
        }
        assert_eq!(entry["untracked"], path == "untracked.txt", "{entry}");
    }
    assert_eq!(row(&list, "new name.txt")["old_path"], "old name.txt");
}

#[tokio::test]
async fn changes_route_selects_staged_and_worktree_diffs() {
    let fixture = Fixture::new();
    for path in ["staged.txt", "worktree.txt", "both.txt"] {
        fixture.file(path, "original\n");
    }
    fixture.commit();
    fixture.session("one", fixture.repo.clone());
    fixture.file("staged.txt", "only staged\nextra line\n");
    fixture.file("both.txt", "in index\n");
    fixture.git(&["add", "--", "staged.txt", "both.txt"]);
    fixture.file("worktree.txt", "working copy\n");
    fixture.file("both.txt", "not staged\n");

    let (list, _) = fixture.list("one").await;
    let staged = row(&list, "staged.txt");
    assert_eq!(staged["staged_status"], "M");
    assert!(staged["worktree_status"].is_null());
    assert_eq!(staged["staged_additions"], 2);
    assert_eq!(staged["staged_deletions"], 1);
    assert!(staged["additions"].is_null() && staged["deletions"].is_null());
    let (default, _) = fixture.detail("one", &list, "staged.txt").await;
    let (explicit, _) = fixture
        .detail_with_side("one", &list, "staged.txt", "&side=staged")
        .await;
    assert_eq!(default, explicit);
    assert_eq!(default["side"], "staged");
    assert!(
        default["diff"]
            .as_str()
            .unwrap()
            .contains("-original\n+only staged\n+extra line\n")
    );

    for (query, side, hunk) in [
        ("&side=staged", "staged", "-original\n+in index\n"),
        ("&side=worktree", "worktree", "-in index\n+not staged\n"),
    ] {
        let (detail, _) = fixture
            .detail_with_side("one", &list, "both.txt", query)
            .await;
        assert_eq!(detail["side"], side);
        assert!(detail["diff"].as_str().unwrap().contains(hunk), "{detail}");
    }
    for (path, side) in [("staged.txt", "worktree"), ("worktree.txt", "staged")] {
        let uri = format!(
            "/sessions/one/changes/{}?revision={}&side={side}",
            row(&list, path)["id"].as_str().unwrap(),
            list["revision"].as_str().unwrap()
        );
        let (status, body, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert_eq!(body["code"], "diff_side_unavailable");
        assert_eq!(body["message"], "no changes on the requested diff side");
        assert_eq!(body["error"], body["message"]);
        assert!(body.get("diff").is_none(), "{body}");
    }

    let uri = format!(
        "/sessions/one/changes/{}?revision={}&side=head",
        staged["id"].as_str().unwrap(),
        list["revision"].as_str().unwrap()
    );
    let (status, body, _) = fixture.get(&uri, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "unknown_side");
    assert_eq!(fixture.get(&uri, false).await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn changes_route_keeps_binary_and_gitlink_numstats_unknown() {
    let fixture = Fixture::new();
    fixture.file("note.txt", b"old\0binary");
    fixture.commit();
    let base = fixture.oid("HEAD");
    fixture.file("note.txt", "index text\n");
    fixture.git(&["add", "--", "note.txt"]);
    fixture.git(&[
        "update-index",
        "--add",
        "--cacheinfo",
        &format!("160000,{base},module"),
    ]);
    fixture.file("note.txt", "worktree text\n");
    fixture.session("one", fixture.repo.clone());

    let (list, _) = fixture.list("one").await;
    let note = row(&list, "note.txt");
    assert_eq!(note["binary"], true);
    assert_eq!(note["additions"], 1);
    assert_eq!(note["deletions"], 1);
    assert!(note["staged_additions"].is_null() && note["staged_deletions"].is_null());
    let (detail, _) = fixture.detail("one", &list, "note.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-index text\n+worktree text\n")
    );
    let (detail, _) = fixture
        .detail_with_side("one", &list, "note.txt", "&side=staged")
        .await;
    assert_eq!(detail["reason"], "binary");

    let module = row(&list, "module");
    assert_eq!(module["staged_status"], "A");
    for key in [
        "additions",
        "deletions",
        "staged_additions",
        "staged_deletions",
    ] {
        assert_eq!(module.get(key), Some(&Value::Null));
    }
}

#[tokio::test]
async fn changes_route_reports_large_line_deltas() {
    let fixture = Fixture::new();
    let padding = "unchanged content\n".repeat(8192);
    fixture.file("large.txt", format!("{padding}replace me\n"));
    fixture.file("binary.bin", b"old\0binary");
    fixture.commit();
    fixture.session("one", fixture.repo.clone());

    fixture.file("large.txt", format!("{padding}replacement\ninserted\n"));
    fixture.file("binary.bin", b"new\0binary");
    fixture.file("new.txt", format!("{padding}no final newline"));

    let (list, _) = fixture.list("one").await;
    for (path, binary, additions, deletions) in [
        ("binary.bin", true, None, None),
        ("large.txt", false, Some(2), Some(1)),
        ("new.txt", false, Some(8193), Some(0)),
    ] {
        let entry = row(&list, path);
        assert_eq!(entry["binary"], binary, "{entry}");
        assert_eq!(entry["additions"].as_u64(), additions, "{entry}");
        assert_eq!(entry["deletions"].as_u64(), deletions, "{entry}");
    }
}

#[tokio::test]
async fn changes_route_reports_binary_and_deltas_at_old_probe_boundary() {
    let fixture = Fixture::new();
    fixture.file("before.bin", b"old\0binary");
    fixture.file("after.bin", "old text\n");
    fixture.file("before-capped.txt", vec![b'x'; PROBE_LIMIT + 1]);
    fixture.file("after-capped.txt", "old text\n");
    fixture.file("at-cap.txt", vec![b'x'; PROBE_LIMIT]);
    fixture.commit();
    fixture.session("one", fixture.repo.clone());

    fixture.file("before.bin", "new text\n");
    fixture.file("after.bin", b"new\0binary");
    fixture.file("before-capped.txt", "new text\n");
    fixture.file("after-capped.txt", vec![b'y'; PROBE_LIMIT + 1]);
    fixture.file("untracked-capped.txt", vec![b'y'; PROBE_LIMIT + 1]);
    fixture.file("at-cap.txt", vec![b'y'; PROBE_LIMIT]);
    // Keep this same-size edit visible to Git on coarse-mtime filesystems.
    File::open(fixture.repo.join("at-cap.txt"))
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))
        .unwrap();

    let (list, _) = fixture.list("one").await;
    for (path, binary) in [("before.bin", true), ("after.bin", true)] {
        let entry = row(&list, path);
        assert_eq!(entry["binary"], binary, "{entry}");
        assert_eq!(entry.get("additions"), Some(&Value::Null), "{entry}");
        assert_eq!(entry.get("deletions"), Some(&Value::Null), "{entry}");
    }
    for (path, deletions) in [
        ("before-capped.txt", 1),
        ("after-capped.txt", 1),
        ("untracked-capped.txt", 0),
    ] {
        let entry = row(&list, path);
        assert_eq!(entry["binary"], false, "{entry}");
        assert_eq!(entry["additions"], 1, "{entry}");
        assert_eq!(entry["deletions"], deletions, "{entry}");
    }
    let at_cap = row(&list, "at-cap.txt");
    assert_eq!(at_cap["additions"], 1);
    assert_eq!(at_cap["deletions"], 1);
}

#[tokio::test]
async fn changes_route_bounds_untracked_line_counts_and_validates_utf8() {
    let fixture = Fixture::new();
    fixture.session("one", fixture.repo.clone());
    fixture.file("split.txt", format!("{}é\r\nlast", "x".repeat(8191)));
    fixture.file("incomplete.txt", b"line\n\xc3");
    fixture.file("nul.txt", b"line\n\0");
    fixture.file("exact-cap.txt", vec![b'x'; UNTRACKED_STAT_LIMIT]);
    fixture.file("over-cap.txt", vec![b'x'; UNTRACKED_STAT_LIMIT + 1]);
    fixture.file("empty.txt", b"");

    let (list, _) = fixture.list("one").await;
    for (path, binary, additions, deletions) in [
        ("split.txt", false, Some(2), Some(0)),
        ("exact-cap.txt", false, Some(1), Some(0)),
        ("empty.txt", false, Some(0), Some(0)),
        ("incomplete.txt", true, None, None),
        ("nul.txt", true, None, None),
        ("over-cap.txt", true, None, None),
    ] {
        let entry = row(&list, path);
        assert_eq!(entry["binary"], binary, "{entry}");
        assert_eq!(entry["additions"].as_u64(), additions, "{entry}");
        assert_eq!(entry["deletions"].as_u64(), deletions, "{entry}");
    }
    let (detail, _) = fixture.detail("one", &list, "over-cap.txt").await;
    assert_eq!(detail["truncated"], true);
    assert_eq!(detail["reason"], "file exceeds diff input limit");
}

#[tokio::test]
async fn changes_route_counts_staged_and_working_lines_without_head() {
    for format in ["sha1", "sha256"] {
        let fixture = Fixture::with_object_format(format);
        let path = "--options\tand spaces.txt";
        fixture.file(path, "staged\n");
        fixture.git(&["add", "--", path]);
        fixture.file(path, "worktree\nadded\n");
        fixture.session("one", fixture.repo.clone());

        let (list, _) = fixture.list("one").await;
        let entry = row(&list, path);
        assert_eq!(entry["staged_status"], "A", "{entry}");
        assert_eq!(entry["worktree_status"], "M", "{entry}");
        assert_eq!(entry["staged_additions"], 1, "{entry}");
        assert_eq!(entry["staged_deletions"], 0, "{entry}");
        assert_eq!(entry["additions"], 2, "{entry}");
        assert_eq!(entry["deletions"], 1, "{entry}");
    }
}

#[tokio::test]
async fn changes_route_keeps_staged_rename_and_worktree_edits_separate() {
    let fixture = Fixture::new();
    fixture.file("first.txt", "first\n");
    fixture.file("second.txt", "second\n");
    fixture.commit();
    fixture.session("one", fixture.repo.clone());

    let path = "renamed\tfile.txt";
    fixture.git(&["mv", "--", "first.txt", path]);
    fixture.file(path, "second\n");
    fs::remove_file(fixture.repo.join("second.txt")).unwrap();

    let (list, _) = fixture.list("one").await;
    let entry = row(&list, path);
    assert_eq!(entry["status"], "renamed");
    assert_eq!(entry["old_path"], "first.txt");
    assert_eq!(entry["staged_status"], "R");
    assert_eq!(entry["worktree_status"], "M");
    assert_eq!(entry.get("staged_additions"), Some(&Value::Null));
    assert_eq!(entry.get("staged_deletions"), Some(&Value::Null));
    assert_eq!(entry["additions"], 1);
    assert_eq!(entry["deletions"], 1);
    let (stage, _) = fixture
        .detail_with_side("one", &list, path, "&side=staged")
        .await;
    assert!(
        stage["diff"]
            .as_str()
            .unwrap()
            .contains("rename from first.txt")
    );
    let (work, _) = fixture.detail("one", &list, path).await;
    let patch = work["diff"].as_str().unwrap();
    assert!(patch.contains("-first\n+second\n"));
    assert!(!patch.contains("rename from"));
}

#[tokio::test]
async fn changes_route_keeps_list_when_a_baseline_blob_is_missing() {
    let fixture = Fixture::new();
    fixture.file("missing.txt", "unavailable baseline\n");
    fixture.file("readable.txt", "before\n");
    fixture.commit();
    fixture.session("one", fixture.repo.clone());

    let oid = fixture.oid("HEAD:missing.txt");
    fs::remove_file(
        fixture
            .repo
            .join(".git/objects")
            .join(&oid[..2])
            .join(&oid[2..]),
    )
    .unwrap();
    fixture.file("missing.txt", "now text\n");
    fixture.file("readable.txt", "after\n");

    let (list, _) = fixture.list("one").await;
    let missing = row(&list, "missing.txt");
    assert_eq!(missing["binary"], false);
    assert_eq!(missing.get("additions"), Some(&Value::Null));
    assert_eq!(missing.get("deletions"), Some(&Value::Null));
    let readable = row(&list, "readable.txt");
    assert_eq!(readable["additions"], 1);
    assert_eq!(readable["deletions"], 1);
}

#[tokio::test]
async fn changes_route_caps_serialized_json_and_keeps_hidden_changes_in_revision() {
    let fixture = Fixture::new();
    fixture.file("b-bytes.txt", "before\n");
    fixture.commit();
    fixture.session("one", fixture.repo.clone());
    fixture.file("a-lines.txt", "line\n".repeat(DIFF_LINES + 10));
    fixture.file("b-bytes.txt", format!("{}\n", "\\".repeat(240)).repeat(400));
    for n in 0..FILE_LIMIT {
        fixture.file(&format!("z{n:03}.txt"), "row\n");
    }
    let (first, size) = fixture.list("one").await;
    assert!(size <= LIST_LIMIT);
    assert_eq!(rows(&first).len(), FILE_LIMIT);
    assert_eq!(first["truncated"], true);
    assert_eq!(first["reason"], LIST_TRUNCATED);

    fixture.file(
        &format!("z{:03}.txt", FILE_LIMIT - 1),
        "hidden row changed\n",
    );
    let (next, _) = fixture.list("one").await;
    assert_ne!(first["revision"], next["revision"]);
    let (line_detail, _) = fixture.detail("one", &next, "a-lines.txt").await;
    let line_diff = line_detail["diff"].as_str().unwrap();
    assert_eq!(line_detail["truncated"], true);
    assert_eq!(line_diff.lines().count(), DIFF_LINES);
    assert!(line_diff.ends_with('\n'));
    let (bytes_detail, size) = fixture.detail("one", &next, "b-bytes.txt").await;
    assert!(size <= DETAIL_LIMIT);
    assert_eq!(bytes_detail["truncated"], true);
    assert_eq!(bytes_detail["reason"], DIFF_TRUNCATED);
    assert!(bytes_detail["diff"].as_str().unwrap().ends_with('\n'));

    for n in 0..FILE_LIMIT {
        fixture.file(
            &format!("c{n:03}-{}.txt", "\"".repeat(190)),
            "escaped name\n",
        );
    }
    let (list, size) = fixture.list("one").await;
    assert!(size <= LIST_LIMIT);
    assert!(rows(&list).len() < FILE_LIMIT);
    assert_eq!(list["truncated"], true);
}

#[tokio::test]
async fn changes_route_does_not_run_repository_filters_or_write_index() {
    let fixture = Fixture::new();
    fixture.file(".gitattributes", "*.txt filter=preview\n");
    fixture.file("file.txt", "before\n");
    fixture.commit();
    fixture.git(&["update-index", "--split-index"]);
    let index = fixture.repo.join(".git/index");
    let index_before = fs::read(&index).unwrap();
    let shared = fs::read_dir(index.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .as_bytes()
                .starts_with(b"sharedindex.")
        })
        .unwrap();
    let old_mtime = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    File::open(&shared)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(old_mtime))
        .unwrap();
    let marker = fixture.scratch.0.join("filter-ran");
    fixture.git(&[
        "config",
        "filter.preview.clean",
        &format!("touch '{}'; cat", marker.display()),
    ]);
    fixture.git(&["config", "filter.preview.required", "true"]);
    fixture.file("file.txt", "after\n");
    fixture.session("one", fixture.repo.clone());
    let (list, _) = fixture.list("one").await;
    let (detail, _) = fixture.detail("one", &list, "file.txt").await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-before\n+after\n")
    );
    assert!(!marker.exists());
    assert_eq!(fs::read(index).unwrap(), index_before);
    assert_eq!(fs::metadata(shared).unwrap().modified().unwrap(), old_mtime);
}

#[tokio::test]
async fn changes_conflicts_have_no_diff_sides_and_block_the_selected_index() {
    let fixture = Fixture::new();
    fixture.git(&["symbolic-ref", "HEAD", "refs/heads/trunk"]);
    fixture.git(&["config", "user.name", "Test"]);
    fixture.git(&["config", "user.email", "test@example.invalid"]);
    fixture.file("shared.txt", "base\n");
    fixture.file("nested/staged.txt", "base\n");
    fixture.commit();
    fixture.git(&["checkout", "--quiet", "-b", "other"]);
    fixture.file("shared.txt", "other version\n");
    fixture.file("added.txt", "other addition\n");
    fixture.commit_named("theirs");
    fixture.git(&["checkout", "--quiet", "trunk"]);
    fixture.file("shared.txt", "our version\n");
    fixture.file("added.txt", "our addition\n");
    let head = fixture.commit_named("ours");
    let git = Git {
        deadline: Instant::now() + TIME_LIMIT,
    };
    let merged =
        crate::fork_guard::output(
            git.command(&fixture.repo)
                .args(["merge", "--no-edit", "other"]),
        )
        .unwrap();
    assert_eq!(merged.status.code(), Some(1), "{:?}", merged);
    assert_eq!(
        Fixture::git_at(
            &fixture.repo,
            &[],
            &["diff", "--name-only", "--diff-filter=U", "-z"]
        ),
        b"added.txt\0shared.txt\0"
    );
    fixture.file("nested/staged.txt", "only staged\n");
    fixture.git(&["add", "--", "nested/staged.txt"]);
    let index = fixture.repo.join(".git/index");
    let before = fs::read(&index).unwrap();
    fixture.session("one", fixture.repo.clone());
    fixture.session("nested", fixture.repo.join("nested"));

    let (list, _) = fixture.list("one").await;
    assert_eq!(list["conflict_count"], 2);
    assert_eq!(rows(&list).len(), 3);
    for path in ["added.txt", "shared.txt"] {
        let conflict = row(&list, path);
        assert_eq!(conflict["unmerged"], true);
        for field in [
            "staged_status",
            "worktree_status",
            "staged_additions",
            "staged_deletions",
            "additions",
            "deletions",
        ] {
            assert_eq!(
                conflict.get(field),
                Some(&Value::Null),
                "{path}: {conflict}"
            );
        }
        let (detail, _) = fixture.detail("one", &list, path).await;
        assert_eq!(detail["format"], "none");
        assert_eq!(detail["diff"], "");
        assert_eq!(detail["reason"], "file has unresolved merge conflicts");
    }
    let (nested, _) = fixture.list("nested").await;
    assert!(nested.get("conflict_count").is_none(), "{nested}");
    assert_eq!(rows(&nested).len(), 1);
    assert_eq!(row(&nested, "staged.txt")["staged_additions"], 1);
    for id in ["one", "nested"] {
        let (status, reply, _) = fixture
            .request(
                "POST",
                &format!("/sessions/{id}/commit"),
                true,
                Some(serde_json::json!({"message": "refuse this merge"})),
            )
            .await;
        assert_eq!(status, StatusCode::CONFLICT, "{id}: {reply}");
        assert_eq!(reply["code"], "unmerged_conflicts");
        assert_eq!(
            reply["message"],
            "resolve merge conflicts before committing"
        );
        assert_eq!(reply["error"], reply["message"]);
    }
    assert_eq!(fs::read(index).unwrap(), before);
    assert_eq!(fixture.oid("HEAD"), head);
}

#[tokio::test]
async fn changes_commit_writes_a_commit_of_staged_changes() {
    let fixture = Fixture::new();
    fixture.git(&["symbolic-ref", "HEAD", "refs/heads/trunk"]);
    fixture.git(&["config", "user.name", "Test"]);
    fixture.git(&["config", "user.email", "test@example.invalid"]);
    fixture.file("note.txt", "before\n");
    fixture.commit();
    let before = fixture.oid("HEAD");
    fixture.file("note.txt", "edited\nsecond line\n");
    fixture.git(&["add", "--", "note.txt"]);
    fixture.session("one", fixture.repo.clone());

    let message = "--amend is not an option here\n\nCommit body";
    let input = serde_json::json!({"message": message});
    assert_eq!(
        fixture.post_commit(input.clone(), false).await.0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(fixture.oid("HEAD"), before);
    let (status, reply, _) = fixture.post_commit(input, true).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let id = fixture.oid("HEAD");
    assert_ne!(id, before);
    assert_eq!(fixture.oid("HEAD^"), before);
    let short = Fixture::git_at(&fixture.repo, &[], &["rev-parse", "--short", "HEAD"]);
    let actual = Fixture::git_at(&fixture.repo, &[], &["log", "-1", "--format=%B"]);
    assert_eq!(
        reply,
        serde_json::json!({
            "commit": {"id": id, "short_id": as_git_str(trimmed_line(&short)).unwrap(), "branch": "trunk"},
            "message": as_git_str(&actual).unwrap().trim_end_matches('\n'),
            "files": 1, "additions": 2, "deletions": 1,
        })
    );
    assert!(reply["message"].as_str().unwrap().starts_with(message));
    assert!(Fixture::git_at(&fixture.repo, &[], &["status", "--porcelain=v2"]).is_empty());
    assert_eq!(
        fixture.get("/info", true).await.1["features"]["changes_commit"],
        true
    );
}

#[tokio::test]
async fn changes_commit_reports_unavailable_counts_for_an_unborn_sha256_head() {
    let fixture = Fixture::with_object_format("sha256");
    fixture.git(&["config", "user.name", "Test"]);
    fixture.git(&["config", "user.email", "test@example.invalid"]);
    fixture.git(&["config", "diff.relative", "true"]);
    fixture.file("artifact.bin", b"\0opaque bytes");
    fixture.file("nested/unstaged.txt", "not part of the index\n");
    fixture.git(&["add", "--", "artifact.bin"]);
    fixture.git(&[
        "update-index",
        "--add",
        "--cacheinfo",
        &format!("160000,{},module", "ab".repeat(32)),
    ]);
    fixture.session("one", fixture.repo.join("nested"));

    let (status, reply, _) = fixture
        .post_commit(
            serde_json::json!({"scope": "current", "message": "first commit"}),
            true,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let head = fixture.oid("HEAD");
    assert_eq!(head.len(), 64);
    assert_eq!(reply["commit"]["id"], head);
    assert_eq!(reply["files"], 2);
    assert_eq!(reply.get("additions"), Some(&Value::Null));
    assert_eq!(reply.get("deletions"), Some(&Value::Null));
    assert_eq!(
        Fixture::git_at(&fixture.repo, &[], &["ls-files", "-z"]),
        b"artifact.bin\0module\0"
    );
}

#[tokio::test]
async fn changes_commit_rejects_nothing_staged_and_empty_messages() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "before\n");
    fixture.commit();
    let before = fixture.oid("HEAD");
    fixture.file("note.txt", "unstaged edit\n");
    fixture.file("intent.txt", "intent to add\n");
    fixture.git(&["add", "-N", "--", "intent.txt"]);
    fixture.session("one", fixture.repo.clone());

    let (status, reply, _) = fixture
        .post_commit(serde_json::json!({"message": "nothing in the index"}), true)
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{reply}");
    assert_eq!(reply["code"], "nothing_staged");
    assert_eq!(reply["message"], "there are no staged changes", "{reply}");
    assert_eq!(reply["error"], reply["message"]);
    assert_eq!(fixture.oid("HEAD"), before);

    fixture.git(&["add", "--", "note.txt"]);
    for message in ["", "  \t\n", "starts\0ends"] {
        let (status, reply, _) = fixture
            .post_commit(serde_json::json!({"message": message}), true)
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(reply["code"], "invalid_commit_message");
        assert_eq!(
            reply["message"],
            "commit message must not be empty or contain NUL"
        );
        assert_eq!(reply["error"], reply["message"]);
        assert_eq!(fixture.oid("HEAD"), before);
    }

    fixture.session("no-git", fixture.scratch.0.clone());
    for (id, auth, input, expected_status, code) in [
        (
            "one",
            false,
            serde_json::json!({"message": "unauthenticated"}),
            StatusCode::UNAUTHORIZED,
            "unauthorized",
        ),
        (
            "bad.id",
            true,
            serde_json::json!({"message": "bad id"}),
            StatusCode::BAD_REQUEST,
            "invalid_session_id",
        ),
        (
            "absent",
            true,
            serde_json::json!({"message": "no session"}),
            StatusCode::NOT_FOUND,
            "session_not_found",
        ),
        (
            "one",
            true,
            serde_json::json!({"message": 42}),
            StatusCode::BAD_REQUEST,
            "invalid_commit_request",
        ),
        (
            "one",
            true,
            serde_json::json!({"message": "wrong scope", "scope": "/etc"}),
            StatusCode::BAD_REQUEST,
            "unknown_scope",
        ),
        (
            "no-git",
            true,
            serde_json::json!({"message": "not a repo"}),
            StatusCode::CONFLICT,
            "not_a_git_repository",
        ),
        (
            "no-git",
            true,
            serde_json::json!({"message": "no linked repo", "scope": "wt-0123456789abcdef"}),
            StatusCode::CONFLICT,
            "not_a_git_repository",
        ),
    ] {
        let (status, reply, _) = fixture
            .request("POST", &format!("/sessions/{id}/commit"), auth, Some(input))
            .await;
        assert_eq!(status, expected_status, "{id}: {reply}");
        assert_eq!(reply["code"], code, "{reply}");
        assert!(
            reply["message"]
                .as_str()
                .is_some_and(|message| !message.is_empty()),
            "{reply}"
        );
        assert_eq!(reply["error"], reply["message"]);
    }
    assert_eq!(fixture.oid("HEAD"), before);
}

#[tokio::test]
async fn changes_commit_rejects_unresolved_session_cwd_before_running_git() {
    const GIT_LOG: &str = "A_SERVE_CHANGES_CWD_GIT_LOG";
    const REAL_PATH: &str = "A_SERVE_CHANGES_CWD_REAL_PATH";
    if std::env::var_os(GIT_LOG).is_none() {
        // Isolate process cwd and PATH from all other tests. The daemon's
        // unrelated repository has a staged change an accidental commit can use.
        let daemon = Fixture::new();
        daemon.git(&["config", "user.name", "Test"]);
        daemon.git(&["config", "user.email", "test@example.invalid"]);
        daemon.file("note.txt", "before\n");
        daemon.commit();
        let head_before = daemon.oid("HEAD");
        daemon.file("note.txt", "staged in the daemon cwd\n");
        daemon.git(&["add", "--", "note.txt"]);
        let index = daemon.repo.join(".git/index");
        let index_before = fs::read(&index).unwrap();
        let marker = daemon.scratch.0.join("git-invocations");
        let fake_bin = daemon.scratch.0.join("fake-bin");
        fs::create_dir(&fake_bin).unwrap();
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o700)
            .open(fake_bin.join("git"))
            .unwrap()
            .write_all(format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"${GIT_LOG}\"\nPATH=\"${REAL_PATH}\" exec git \"$@\"\n").as_bytes())
            .unwrap();
        let test = format!(
            "{}::changes_commit_rejects_unresolved_session_cwd_before_running_git",
            module_path!().split_once("::").unwrap().1
        );
        let output = crate::fork_guard::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &test])
                .env(GIT_LOG, &marker)
                .env(REAL_PATH, std::env::var_os("PATH").unwrap())
                .env("PATH", &fake_bin)
                .current_dir(&daemon.repo),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{}\nGit invocations:\n{}",
            String::from_utf8_lossy(&output.stderr),
            fs::read_to_string(&marker).unwrap_or_default()
        );
        assert!(!marker.exists(), "a Git subprocess was started");
        assert_eq!(fs::read(&index).unwrap(), index_before);
        assert_eq!(daemon.oid("HEAD"), head_before);
        return;
    }

    let fixture = Fixture::without_repository();
    fixture.file("not-a-directory", "contents\n");
    fixture.session("missing", fixture.repo.join("removed"));
    fixture.session("file", fixture.repo.join("not-a-directory"));
    fixture.session("relative", PathBuf::from("."));
    fixture.create_session("unset");
    for id in ["missing", "file", "relative", "unset"] {
        let (status, reply, _) = fixture
            .request(
                "POST",
                &format!("/sessions/{id}/commit"),
                true,
                Some(serde_json::json!({"message": "commit the session index"})),
            )
            .await;
        assert_eq!(status, StatusCode::CONFLICT, "{id}: {reply}");
        assert_eq!(reply["code"], "cwd_unresolved", "{reply}");
        assert!(
            reply["message"]
                .as_str()
                .is_some_and(|message| message.contains("/cd")),
            "{reply}"
        );
        assert_eq!(reply["error"], reply["message"]);
    }
}

#[tokio::test]
async fn changes_commit_leaves_unstaged_edits_in_selected_worktree() {
    let fixture = Fixture::new();
    fixture.git(&["config", "user.name", "Test"]);
    fixture.git(&["config", "user.email", "test@example.invalid"]);
    fixture.file("note.txt", "before\n");
    fixture.commit();
    let before = fixture.oid("HEAD");
    let linked = fixture.worktree(fixture.repo.join(".worktrees/linked"));
    fs::write(linked.join("note.txt"), "in index\n").unwrap();
    Fixture::git_at(&linked, &[], &["add", "--", "note.txt"]);
    fs::write(linked.join("note.txt"), "in index\nnot staged\n").unwrap();
    fixture.file("note.txt", "staged in a different worktree\n");
    fixture.git(&["add", "--", "note.txt"]);
    fixture.session("one", fixture.repo.clone());

    let (status, reply, _) = fixture
        .post_commit(
            serde_json::json!({
                "scope": worktree_scope_id(&linked), "message": "commit the selected index",
            }),
            true,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert!(reply["commit"]["branch"].is_null());
    assert_eq!(reply["files"], 1);
    assert_eq!(reply["additions"], 1);
    assert_eq!(fixture.oid("HEAD"), before);
    let linked_id = Fixture::git_at(&linked, &[], &["rev-parse", "HEAD"]);
    assert_eq!(
        reply["commit"]["id"],
        as_git_str(trimmed_line(&linked_id)).unwrap()
    );
    assert_eq!(
        Fixture::git_at(&linked, &[], &["show", "HEAD:note.txt"]),
        b"in index\n"
    );
    assert_eq!(
        fs::read(linked.join("note.txt")).unwrap(),
        b"in index\nnot staged\n"
    );
    assert_eq!(
        Fixture::git_at(&linked, &[], &["status", "--porcelain", "--", "note.txt"]),
        b" M note.txt\n"
    );
    assert_eq!(
        Fixture::git_at(
            &fixture.repo,
            &[],
            &["status", "--porcelain", "--", "note.txt"]
        ),
        b"M  note.txt\n"
    );
}

#[tokio::test]
async fn changes_commit_bounds_non_utf8_git_stderr() {
    let fixture = Fixture::new();
    fixture.git(&["config", "user.name", "Test"]);
    fixture.git(&["config", "user.email", "test@example.invalid"]);
    fixture.file("note.txt", "before\n");
    fixture.commit();
    let before = fixture.oid("HEAD");
    let hooks = fixture.repo.join(".git/hooks");
    fs::create_dir(&hooks).unwrap();
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(hooks.join("pre-commit"))
        .unwrap()
        .write_all(
            b"#!/bin/sh\nprintf 'test hook refusal: \\377' >&2\nprintf '%05000d' 0 >&2\nexit 1\n",
        )
        .unwrap();
    fixture.file("note.txt", "staged edit\n");
    fixture.git(&["add", "--", "note.txt"]);
    fixture.session("one", fixture.repo.clone());

    let (status, reply, _) = fixture
        .post_commit(
            serde_json::json!({"message": "hooks can reject this"}),
            true,
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{reply}");
    assert_eq!(reply["code"], "commit_failed");
    assert_eq!(reply["error"], reply["message"]);
    let stderr = reply["message"].as_str().unwrap();
    assert!(
        stderr.starts_with("test hook refusal: \u{fffd}"),
        "{stderr}"
    );
    assert_eq!(stderr.len(), COMMIT_STDERR_LIMIT);
    assert_eq!(fixture.oid("HEAD"), before);
}

#[tokio::test]
async fn changes_commits_paginate_the_selected_head() {
    let fixture = Fixture::new();
    fixture.git(&["symbolic-ref", "HEAD", "refs/heads/trunk"]);
    fixture.session("one", fixture.repo.clone());
    let (unborn, _) = fixture.commits("").await;
    assert_eq!(
        unborn,
        serde_json::json!({
            "source": "commit_history", "scope": {"id": "current", "label": "repo", "path": fixture.repo, "current": true},
            "branch": "trunk", "head": null, "rows": [], "next_cursor": null, "truncated": false, "reason": null,
        })
    );

    fixture.file("note.txt", "first\n");
    let root = fixture.commit_named("root subject");
    Fixture::commit_at(&fixture.repo, 1_100_000_000);
    let middle = fixture.oid("HEAD");
    Fixture::commit_at(&fixture.repo, 1_000_000_000);
    let head = fixture.oid("HEAD");
    let (first, _) = fixture.commits("?scope=current&limit=1").await;
    assert_eq!(first["branch"], "trunk");
    assert_eq!(first["head"], head);
    assert_eq!(
        first["rows"],
        serde_json::json!([{
            "id": head, "subject": "recency fixture", "committed_at": 1_000_000_000,
        }])
    );
    assert_eq!(first["next_cursor"], head);
    assert_eq!(first["truncated"], false);
    assert!(first["reason"].is_null());

    // A new tip does not shift the exclusive cursor on this chain.
    let fresh = fixture.commit_named("new tip");
    let (next, _) = fixture.commits(&format!("?limit=1&before={head}")).await;
    assert_eq!(next["head"], fresh);
    assert_eq!(next["rows"][0]["id"], middle);
    assert_eq!(next["next_cursor"], middle);
    let (last, _) = fixture.commits(&format!("?limit=1&before={middle}")).await;
    assert_eq!(last["rows"][0]["id"], root);
    assert!(last["next_cursor"].is_null());
    let (end, _) = fixture.commits(&format!("?before={root}")).await;
    assert_eq!(end["rows"], serde_json::json!([]));
    assert!(end["next_cursor"].is_null());

    let linked = fixture.worktree(fixture.repo.join(".worktrees/historical"));
    Fixture::git_at(&linked, &[], &["checkout", "--quiet", "--detach", &root]);
    let scope_id = worktree_scope_id(&linked);
    let (selected, _) = fixture.commits(&format!("?scope={scope_id}")).await;
    assert_eq!(selected["head"], root);
    assert!(selected["branch"].is_null());
    assert_eq!(
        selected["scope"],
        serde_json::json!({
            "id": scope_id, "label": "historical", "path": linked, "current": false,
        })
    );
    assert_eq!(selected["rows"].as_array().unwrap().len(), 1);
    let (detail, _) = fixture.commits(&format!("/{root}?scope={scope_id}")).await;
    assert_eq!(detail["scope"], selected["scope"]);
    assert_eq!(detail["base"], "root");
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/commits?scope={scope_id}&before={head}"),
                true
            )
            .await
            .0,
        StatusCode::CONFLICT
    );
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/commits/{head}?scope={scope_id}"),
                true
            )
            .await
            .0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn changes_commits_bound_subjects_and_serialize_resumable_pages() {
    let fixture = Fixture::new();
    fixture.session("one", fixture.repo.clone());
    let escaped = "\u{1}".repeat(COMMIT_SUBJECT_LIMIT);
    for _ in 0..COMMIT_PAGE_DEFAULT + 2 {
        fixture.git(&[
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "ordinary subject",
        ]);
    }
    let (ordinary, _) = fixture.commits("").await;
    assert_eq!(
        ordinary["rows"].as_array().unwrap().len(),
        COMMIT_PAGE_DEFAULT
    );
    assert_eq!(ordinary["truncated"], false);
    assert!(ordinary["next_cursor"].is_string());
    for _ in 0..COMMIT_PAGE_DEFAULT {
        fixture.git(&["commit", "--quiet", "--allow-empty", "-m", &escaped]);
    }
    let unicode = format!("{}é", "x".repeat(COMMIT_SUBJECT_LIMIT - 1));
    let fresh = fixture.commit_named(&unicode);
    let (first, bytes) = fixture.commits("?limit=100").await;
    assert!(bytes <= LIST_LIMIT);
    assert_eq!(first["truncated"], true);
    let commits = first["rows"].as_array().unwrap();
    assert_eq!(commits[0]["id"], fresh);
    assert_eq!(
        commits[0]["subject"].as_str().unwrap().len(),
        COMMIT_SUBJECT_LIMIT - 1
    );
    assert_eq!(commits[1]["subject"], escaped);
    assert_eq!(first["next_cursor"], commits.last().unwrap()["id"]);
    let cursor = first["next_cursor"].as_str().unwrap();
    let (second, _) = fixture
        .commits(&format!("?limit=100&before={cursor}"))
        .await;
    assert!(second["next_cursor"].is_null());
    assert_eq!(second["truncated"], false);
    assert_eq!(
        second["rows"].as_array().unwrap().len() + commits.len(),
        COMMIT_PAGE_DEFAULT * 2 + 3
    );

    // A giant raw subject cannot consume the next row or strand the cursor.
    let huge = fixture.commit_named(&"x".repeat(LIST_LIMIT + 1024));
    let (raw, _) = fixture.commits("?limit=100").await;
    assert_eq!(raw["truncated"], true);
    assert_eq!(raw["next_cursor"], huge);
    assert_eq!(raw["rows"].as_array().unwrap().len(), 1);
    assert_eq!(
        raw["rows"][0]["subject"].as_str().unwrap().len(),
        COMMIT_SUBJECT_LIMIT
    );
    let (resumed, _) = fixture.commits(&format!("?limit=1&before={huge}")).await;
    assert_eq!(resumed["rows"][0]["id"], fresh);
}

#[tokio::test]
async fn changes_commits_reject_invalid_requests_and_advertise_support() {
    let fixture = Fixture::new();
    fixture.file("text.txt", "first\n");
    let commit = fixture.commit_named("first");
    let blob = fixture.oid("HEAD:text.txt");
    fixture.git(&["tag", "-a", "-m", "tag", "first", &commit]);
    let tag = fixture.oid("refs/tags/first");
    fixture.session("one", fixture.repo.clone());
    fixture.session("no-git", fixture.scratch.0.clone());
    for suffix in ["", &format!("/{commit}")] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits{suffix}"), false)
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            fixture
                .get(&format!("/sessions/absent/changes/commits{suffix}"), true)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        let (status, body, _) = fixture
            .get(&format!("/sessions/bad.id/changes/commits{suffix}"), true)
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "invalid session id");
        let (status, body, _) = fixture
            .get(&format!("/sessions/no-git/changes/commits{suffix}"), true)
            .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["code"], "git_unavailable");
        assert_eq!(body["reason"], "not_a_repository");
        for query in [
            "scope=wt-0000000000000000",
            "scope=%2Ftmp",
            "scope=current&scope=current",
        ] {
            let (status, body, _) = fixture
                .get(
                    &format!("/sessions/one/changes/commits{suffix}?{query}"),
                    true,
                )
                .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                body,
                serde_json::json!({"error": "unknown changes scope", "code": "unknown_scope"})
            );
        }
    }
    for query in [
        "limit=0",
        "limit=101",
        "limit=%2B2",
        "limit=1&limit=2",
        "before=&before=",
        "unknown=x",
    ] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits?{query}"), true)
                .await
                .0,
            StatusCode::BAD_REQUEST,
            "{query}"
        );
    }
    for invalid in [
        "",
        "HEAD",
        "--help",
        "HEAD%5E",
        &commit[..12],
        &"a".repeat(64),
        &format!("A{}", "a".repeat(39)),
    ] {
        assert_eq!(
            fixture
                .get(
                    &format!("/sessions/one/changes/commits?before={invalid}"),
                    true
                )
                .await
                .0,
            StatusCode::BAD_REQUEST,
            "{invalid}"
        );
        if !invalid.is_empty() {
            assert_eq!(
                fixture
                    .get(&format!("/sessions/one/changes/commits/{invalid}"), true)
                    .await
                    .0,
                StatusCode::BAD_REQUEST,
                "{invalid}"
            );
        }
    }
    for non_commit in [&blob, &tag, &"0".repeat(40)] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits/{non_commit}"), true)
                .await
                .0,
            StatusCode::NOT_FOUND,
            "{non_commit}"
        );
        assert_eq!(
            fixture
                .get(
                    &format!("/sessions/one/changes/commits?before={non_commit}"),
                    true
                )
                .await
                .0,
            StatusCode::CONFLICT,
            "{non_commit}"
        );
    }
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/commits/{commit}?limit=1"),
                true
            )
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    let (status, info, _) = fixture.get("/info", true).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(info["features"]["changes"], true);
    assert_eq!(info["features"]["changes_commits"], true);

    let git = Git {
        deadline: Instant::now() + TIME_LIMIT,
    };
    let error = git
        .run(
            &mut Command::new(fixture.scratch.0.join("missing-git")),
            256,
        )
        .err()
        .unwrap();
    assert!(matches!(error, Failure::Unavailable(Unavailable::Git)));
    assert_eq!(error.response(false).status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn changes_commits_use_the_repository_oid_width() {
    let fixture = Fixture::with_object_format("sha256");
    fixture.file("text.txt", "sha256 content\n");
    let root = fixture.commit_named("sha256 root");
    fixture.session("one", fixture.repo.clone());
    assert_eq!(root.len(), 64);
    let (history, _) = fixture.commits("").await;
    assert_eq!(history["head"], root);
    assert_eq!(history["rows"][0]["id"], root);
    let (detail, _) = fixture.commits(&format!("/{root}")).await;
    assert_eq!(detail["base"], "root");
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("+sha256 content\n")
    );
    for suffix in [
        format!("/{}", &root[..40]),
        format!("?before={}", &root[..40]),
    ] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits{suffix}"), true)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
    }
}

#[tokio::test]
async fn changes_commits_diff_root_and_merge_first_parent_only() {
    let fixture = Fixture::new();
    fixture.git(&["symbolic-ref", "HEAD", "refs/heads/trunk"]);
    fixture.file("root.txt", "root content\n");
    let root = fixture.commit_named("root");
    fixture.git(&["checkout", "--quiet", "-b", "side"]);
    fixture.file("side-only.txt", "merged content\n");
    let side = fixture.commit_named("side change");
    fixture.git(&["checkout", "--quiet", "-b", "elsewhere", &root]);
    let off_chain = fixture.commit_named("other branch");
    fixture.git(&["checkout", "--quiet", "trunk"]);
    fixture.file("trunk-only.txt", "trunk content\n");
    let trunk = fixture.commit_named("trunk change");
    fixture.git(&["merge", "--quiet", "--no-ff", "-m", "Merge side", "side"]);
    let merge = fixture.oid("HEAD");
    fixture.git(&["replace", &merge, &side]);
    fixture.session("one", fixture.repo.clone());

    let (history, _) = fixture.commits("").await;
    let ids: Vec<_> = history["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, [merge.as_str(), trunk.as_str(), root.as_str()]);
    assert_eq!(history["rows"][0]["subject"], "Merge side");
    for excluded in [&side, &off_chain] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits/{excluded}"), true)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            fixture
                .get(
                    &format!("/sessions/one/changes/commits?before={excluded}"),
                    true
                )
                .await
                .0,
            StatusCode::CONFLICT
        );
    }
    let (first, _) = fixture.commits(&format!("/{root}")).await;
    assert_eq!(first["source"], "commit");
    assert_eq!(first["base"], "root");
    assert_eq!(first["commit_id"], root);
    assert!(first["base_id"].is_null());
    assert_eq!(first["format"], "unified");
    assert!(
        first["diff"]
            .as_str()
            .unwrap()
            .contains("--- /dev/null\n+++ b/root.txt\n")
    );
    let (detail, _) = fixture.commits(&format!("/{merge}")).await;
    assert_eq!(detail["base"], "first_parent");
    assert_eq!(detail["base_id"], trunk);
    let diff = detail["diff"].as_str().unwrap();
    assert!(diff.contains("+++ b/side-only.txt\n") && diff.contains("+merged content\n"));
    assert!(!diff.contains("trunk-only.txt"));
}

#[tokio::test]
async fn changes_commits_distinguish_shallow_boundaries_from_missing_objects() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "before\n");
    let root = fixture.commit_named("root");
    fixture.file("note.txt", "after\n");
    let head = fixture.commit_named("head");
    let shallow = fixture.scratch.0.join("shallow");
    Fixture::git_at(
        &fixture.repo,
        &[("GIT_ALLOW_PROTOCOL", "file")],
        &[
            "clone",
            "--quiet",
            "--depth=1",
            &format!("file://{}", fixture.repo.display()),
            shallow.to_str().unwrap(),
        ],
    );
    fixture.session("one", shallow.clone());
    let (list, _) = fixture.commits("").await;
    assert_eq!(list["rows"].as_array().unwrap().len(), 1);
    assert_eq!(list["head"], head);
    assert!(list["next_cursor"].is_null());
    let (detail, _) = fixture.commits(&format!("/{head}")).await;
    assert_eq!(detail["base"], "shallow_boundary");
    assert!(detail["base_id"].is_null());
    assert_eq!(detail["format"], "none");
    assert_eq!(detail["diff"], "");
    assert_eq!(detail["reason"], "parent unavailable in shallow clone");
    assert_eq!(detail["truncated"], false);
    assert_eq!(
        fixture
            .get(&format!("/sessions/one/changes/commits/{root}"), true)
            .await
            .0,
        StatusCode::NOT_FOUND
    );

    fixture.session("one", fixture.repo.clone());
    let shallow_file = fixture.repo.join(".git/shallow");
    fs::write(&shallow_file, format!("{head}\n")).unwrap();
    let (detail, _) = fixture.commits(&format!("/{head}")).await;
    assert_eq!(detail["base"], "first_parent");
    assert_eq!(detail["base_id"], root);
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-before\n+after\n")
    );
    fs::remove_file(shallow_file).unwrap();

    let object = |oid: &str| {
        fixture
            .repo
            .join(".git/objects")
            .join(&oid[..2])
            .join(&oid[2..])
    };
    let blob = object(&fixture.oid("HEAD:note.txt"));
    let held = fixture.scratch.0.join("held-blob");
    fs::rename(&blob, &held).unwrap();
    let uri = format!("/sessions/one/changes/commits/{head}");
    assert_eq!(
        fixture.get(&uri, true).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    fs::rename(held, blob).unwrap();
    fixture.git(&["commit-graph", "write", "--reachable"]);
    fs::remove_file(object(&root)).unwrap();
    assert_eq!(
        fixture.get(&uri, true).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        fixture.get("/sessions/one/changes/commits", true).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        fixture
            .get(
                &format!("/sessions/one/changes/commits?before={root}"),
                true
            )
            .await
            .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn changes_commits_scope_filters_each_rename_side() {
    let fixture = Fixture::new();
    let common_a = "unchanged inbound line\n".repeat(12);
    let common_b = "unchanged outbound line\n".repeat(12);
    let common_c = "unchanged local line\n".repeat(12);
    fixture.file(
        "outside/original.txt",
        format!("{common_a}hidden old content\n"),
    );
    fixture.file(
        "work/outgoing.txt",
        format!("{common_b}eligible old content\n"),
    );
    fixture.file(
        "work/old name.txt",
        format!("{common_c}original local content\n"),
    );
    fixture.file("work/.agent_tmp/private.txt", "protected temporary data\n");
    fixture.file("outside/edit.txt", "before outside\n");
    let root = fixture.commit_named("base");
    fixture.git(&["mv", "--", "outside/original.txt", "work/incoming.txt"]);
    fixture.git(&["mv", "--", "work/outgoing.txt", "outside/destination.txt"]);
    fixture.git(&["mv", "--", "work/old name.txt", "work/new name.txt"]);
    fixture.file(
        "work/incoming.txt",
        format!("{common_a}eligible new content\n"),
    );
    fixture.file(
        "outside/destination.txt",
        format!("{common_b}hidden new content\n"),
    );
    fixture.file(
        "work/new name.txt",
        format!("{common_c}updated local content\n"),
    );
    fixture.file("work/.agent_tmp/private.txt", "new private data\n");
    fixture.file("work/odd\nname.txt", "safe odd path\n");
    let changed = fixture.commit_named("renames");
    fixture.file("outside/edit.txt", "external change only\n");
    let outside = fixture.commit_named("outside-only subject");
    let linked = fixture.worktree(fixture.repo.join(".worktrees/feature"));
    fixture.session("one", linked.join("work"));

    let (history, _) = fixture.commits("").await;
    assert_eq!(
        history["scope"]["path"],
        linked.join("work").to_str().unwrap()
    );
    assert_eq!(history["rows"][0]["subject"], "outside-only subject");
    let (empty, _) = fixture.commits(&format!("/{outside}")).await;
    assert_eq!(empty["format"], "none");
    assert_eq!(empty["diff"], "");
    assert_eq!(empty["reason"], "no changes in this scope");
    let (detail, _) = fixture.commits(&format!("/{changed}")).await;
    assert_eq!(detail["base_id"], root);
    assert_eq!(detail["truncated"], false);
    let diff = detail["diff"].as_str().unwrap();
    for expected in [
        "rename from old name.txt\nrename to new name.txt\n",
        "-original local content\n+updated local content\n",
        "--- /dev/null\n+++ b/incoming.txt\n",
        "+eligible new content\n",
        "--- a/outgoing.txt\n+++ /dev/null\n",
        "-eligible old content\n",
        "b/odd\\012name.txt",
    ] {
        assert!(diff.contains(expected), "missing {expected:?} in {diff}");
    }
    for excluded in [
        "a/work/",
        "b/work/",
        "outside/",
        "original.txt",
        "destination.txt",
        "hidden old content",
        "hidden new content",
        ".agent_tmp",
        "new private data",
    ] {
        assert!(!diff.contains(excluded), "included {excluded:?} in {diff}");
    }
    for path in [&fixture.repo, &linked] {
        let id = worktree_scope_id(path);
        assert_eq!(
            fixture
                .get(&format!("/sessions/one/changes/commits?scope={id}"), true)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            fixture
                .get(
                    &format!("/sessions/one/changes/commits/{changed}?scope={id}"),
                    true
                )
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
    }
}

#[tokio::test]
async fn changes_commits_omit_non_text_payload_without_using_the_index_or_drivers() {
    let fixture = Fixture::new();
    fixture.file(".gitattributes", "*.txt diff=preview\n");
    fixture.file("text.txt", "first\n");
    fixture.commit();
    fixture.file("artifact.bin", b"\0binary secret\n");
    symlink("symlink secret", fixture.repo.join("link")).unwrap();
    fixture.git(&["add", "--", "artifact.bin", "link"]);
    let gitlink = "ab".repeat(20);
    fixture.git(&[
        "update-index",
        "--add",
        "--cacheinfo",
        &format!("160000,{gitlink},module"),
    ]);
    fixture.git(&["commit", "--quiet", "-m", "opaque entries"]);
    let opaque = fixture.oid("HEAD");
    fixture.file("text.txt", "second\n");
    fixture.git(&["commit", "--quiet", "-m", "text edit", "--", "text.txt"]);
    let text = fixture.oid("HEAD");
    let marker = fixture.scratch.0.join("driver-ran");
    let script = fixture.scratch.0.join("external-driver");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(&script)
        .unwrap()
        .write_all(format!("#!/bin/sh\nprintf ran > '{}'\n", marker.display()).as_bytes())
        .unwrap();
    for config in [
        "diff.external",
        "diff.preview.command",
        "diff.preview.textconv",
    ] {
        fixture.git(&["config", config, script.to_str().unwrap()]);
    }
    let index = fixture.repo.join(".git/index");
    fs::write(&index, b"not an index").unwrap();
    fixture.session("one", fixture.repo.clone());
    let (detail, _) = fixture.commits(&format!("/{opaque}")).await;
    assert_eq!(detail["format"], "none");
    assert_eq!(detail["diff"], "");
    assert_eq!(detail["truncated"], false);
    let reason = detail["reason"].as_str().unwrap();
    for notice in ["binary: artifact.bin", "symlink: link", "gitlink: module"] {
        assert!(reason.contains(notice), "{reason}");
    }
    for secret in ["binary secret", "symlink secret", &gitlink] {
        assert!(!detail.to_string().contains(secret));
    }
    let (detail, _) = fixture.commits(&format!("/{text}")).await;
    assert!(
        detail["diff"]
            .as_str()
            .unwrap()
            .contains("-first\n+second\n")
    );
    assert!(!marker.exists());
    assert_eq!(fs::read(index).unwrap(), b"not an index");
}

#[tokio::test]
async fn changes_commits_cap_patch_lines_and_serialized_bytes() {
    let fixture = Fixture::new();
    fixture.session("one", fixture.repo.clone());
    fixture.file("lines.txt", "line\n".repeat(DIFF_LINES + 40));
    let lines = fixture.commit_named("many lines");
    fixture.file("escaped.txt", format!("{}\n", "\\".repeat(300)).repeat(400));
    let bytes = fixture.commit_named("escaped bytes");
    fixture.file("too-large.txt", vec![b'x'; DIFF_INPUT_LIMIT + 1]);
    fixture.file("trailing.txt", "keep this patch\n");
    let oversized = fixture.commit_named("oversized blob");
    let (line_detail, _) = fixture.commits(&format!("/{lines}")).await;
    assert_eq!(line_detail["truncated"], true);
    let diff = line_detail["diff"].as_str().unwrap();
    assert_eq!(diff.lines().count(), DIFF_LINES);
    assert!(diff.ends_with('\n'));
    let (byte_detail, size) = fixture.commits(&format!("/{bytes}")).await;
    assert!(size <= DETAIL_LIMIT && size > DETAIL_LIMIT - 800, "{size}");
    assert_eq!(byte_detail["truncated"], true);
    let diff = byte_detail["diff"].as_str().unwrap();
    assert!(diff.ends_with('\n') && diff.lines().count() < DIFF_LINES);
    let (large_detail, _) = fixture.commits(&format!("/{oversized}")).await;
    assert_eq!(large_detail["truncated"], true);
    assert!(
        large_detail["reason"]
            .as_str()
            .unwrap()
            .contains("file exceeds diff input limit: too-large.txt")
    );
    let diff = large_detail["diff"].as_str().unwrap();
    assert!(!diff.contains("too-large.txt"));
    assert!(diff.contains("+keep this patch\n"));
}

#[tokio::test]
async fn changes_commits_reject_tree_inventory_overflow() {
    let fixture = Fixture::new();
    fixture.file("blob.txt", "one blob\n");
    let blob = Fixture::git_at(&fixture.repo, &[], &["hash-object", "-w", "--", "blob.txt"]);
    let blob = as_git_str(trimmed_line(&blob)).unwrap();
    let mut inventory = Vec::new();
    let name = "x".repeat(220);
    for n in 0..14_000 {
        writeln!(inventory, "100644 blob {blob}\t{name}{n:05}").unwrap();
    }
    let input = fixture.scratch.0.join("tree-input");
    fs::write(&input, inventory).unwrap();
    let git = Git {
        deadline: Instant::now() + TIME_LIMIT,
    };
    let tree = git
        .run_with(
            git.command(&fixture.repo).arg("mktree"),
            256,
            File::open(input).unwrap().into(),
            |_| false,
        )
        .unwrap();
    assert_eq!(tree.code(), Some(0));
    let tree = as_git_str(trimmed_line(&tree.data.bytes)).unwrap();
    let head = Fixture::git_at(
        &fixture.repo,
        &[],
        &["commit-tree", tree, "-m", "large inventory"],
    );
    let head = as_git_str(trimmed_line(&head)).unwrap();
    fixture.git(&["update-ref", "HEAD", head]);
    fixture.session("one", fixture.repo.clone());
    let (status, body, _) = fixture
        .get(&format!("/sessions/one/changes/commits/{head}"), true)
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"], "Git inventory exceeds the collection limit");
}

fn shared_checkout(fixture: &Fixture) -> PathBuf {
    let checkout = fixture.scratch.0.join("shared checkout");
    Fixture::git_at(
        &fixture.repo,
        &[("GIT_ALLOW_PROTOCOL", "file")],
        &[
            "clone",
            "--quiet",
            "--shared",
            fixture.repo.to_str().unwrap(),
            checkout.to_str().unwrap(),
        ],
    );
    checkout
}

#[tokio::test]
async fn changes_verifier_rejects_external_alternate_objects() {
    let fixture = Fixture::new();
    fixture.file("sub/borrowed.txt", "synthetic external object content\n");
    let head = fixture.commit_named("external objects");
    let checkout = shared_checkout(&fixture);
    let alternate = fixture.repo.join(".git/objects");
    let from_git = Fixture::git_at(&checkout, &[], &["show", "HEAD:sub/borrowed.txt"]);
    assert_eq!(from_git, b"synthetic external object content\n");
    fixture.session("one", checkout.clone());
    let prefix = "/sessions/one/changes";
    for suffix in [&*format!("/commits/{head}"), "/commits", "/scopes", ""] {
        let (status, body, _) = fixture.get(&format!("{prefix}{suffix}"), true).await;
        assert_eq!(status, StatusCode::CONFLICT, "{suffix}: {body}");
        assert_eq!(body["code"], "repository_unavailable");
        assert!(body.get("diff").is_none());
        assert!(!body.to_string().contains(&*alternate.to_string_lossy()));
    }

    // Going through a local alternate (and C-style quoted path) must not make
    // a transitive target outside the canonical session cwd eligible.
    let objects = checkout.join(".git/objects");
    let local = objects.join("local store");
    fs::create_dir_all(local.join("info")).unwrap();
    fs::write(objects.join("info/alternates"), b"\"local\\040store\"\n").unwrap();
    fs::write(
        local.join("info/alternates"),
        format!("{}\n", alternate.display()),
    )
    .unwrap();
    let (status, body, _) = fixture.get(&format!("{prefix}/commits/{head}"), true).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "repository_unavailable");

    // The same checkout can read a packed relative alternate physically inside
    // its cwd, even though its primary object store is empty.
    fixture.git(&["repack", "-a", "-d"]);
    fs::remove_dir_all(&local).unwrap();
    fs::rename(&alternate, &local).unwrap();
    let (body, _) = fixture.commits(&format!("/{head}")).await;
    assert!(
        body["diff"]
            .as_str()
            .unwrap()
            .contains("+synthetic external object content\n")
    );
    fixture.session("one", checkout.join("sub"));
    let (status, body, _) = fixture.get(&format!("{prefix}/commits/{head}"), true).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "repository_unavailable");
}

async fn assert_plain_routes(fixture: &Fixture, cwd: &Path, reason: &str) {
    fixture.session("one", cwd.to_path_buf());
    let prefix = "/sessions/one/changes";
    for suffix in ["", "/scopes"] {
        let uri = format!("{prefix}{suffix}");
        assert_eq!(fixture.get(&uri, false).await.0, StatusCode::UNAUTHORIZED);
    }
    let (status, catalog, _) = fixture.get(&format!("{prefix}/scopes"), true).await;
    assert_eq!(status, StatusCode::OK, "{catalog}");
    assert_eq!(catalog["source"], "plain");
    assert_eq!(catalog["git_available"], false);
    assert_eq!(catalog["current_path"], cwd.to_str().unwrap());
    assert_eq!(catalog["truncated"], false);
    assert_eq!(catalog["reason"], reason);
    assert_eq!(
        catalog["scopes"],
        serde_json::json!([{
            "id": "current", "path": cwd, "label": cwd.file_name().unwrap().to_str().unwrap(),
            "is_current": true, "dirty": false, "file_count": 1,
        }])
    );
    let (status, list, _) = fixture.get(&format!("{prefix}?scope=current"), true).await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(list["source"], "plain");
    assert_eq!(list["git_available"], false);
    assert_eq!(list["reason"], reason);
    assert!(!list.as_object().unwrap().contains_key("diff"));
    assert_eq!(rows(&list).len(), 1, "{list}");
    let note = row(&list, "note.txt");
    assert_eq!(note.get("additions"), Some(&Value::Null));
    assert_eq!(note.get("deletions"), Some(&Value::Null));
    let id = note["id"].as_str().unwrap();
    let revision = list["revision"].as_str().unwrap();
    let detail = format!("{prefix}/{id}?revision={revision}&scope=current");
    let (status, body, _) = fixture.get(&detail, true).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(
        body,
        serde_json::json!({"error": "not a Git repository", "code": "not_a_git_repository"})
    );
    let (status, body, _) = fixture
        .get(&format!("{prefix}?scope=wt-0000000000000000"), true)
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body,
        serde_json::json!({"error": "unknown changes scope", "code": "unknown_scope"})
    );
    let (status, body, _) = fixture.get(&format!("{prefix}/commits"), true).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "git_unavailable");
    assert_eq!(body["scope"]["id"], "current");
    assert_eq!(body["head"], Value::Null);
    assert_eq!(body["rows"], serde_json::json!([]));
    assert_eq!(body["next_cursor"], Value::Null);
    assert_eq!(body["truncated"], false);
    assert_eq!(body["reason"], reason);
}

#[tokio::test]
async fn changes_plain_inventory_is_sorted_and_cwd_bounded() {
    let fixture = Fixture::without_repository();
    fixture.file("outside/secret.txt", "outside\n");
    fixture.file("work/z.bin", b"\0BIN");
    fixture.file("work/a/child.txt", "child\n");
    fixture.file("work/a.txt", "text\n");
    fixture.file("work/a/.git/secret.txt", "Git metadata\n");
    fixture.file("work/.agent_tmp/private.txt", "private\n");
    let cwd = fixture.repo.join("work");
    symlink(fixture.repo.join("outside"), cwd.join("linked-dir")).unwrap();
    symlink(
        fixture.repo.join("outside/secret.txt"),
        cwd.join("linked-file"),
    )
    .unwrap();
    fixture.session("one", cwd);

    let (list, _) = fixture.list("one").await;
    assert_eq!(list["source"], "plain");
    assert_eq!(list["root_label"], "work");
    assert_eq!(list["truncated"], false);
    let paths: Vec<_> = rows(&list)
        .iter()
        .map(|row| row["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["a.txt", "a/child.txt", "z.bin"]);
    let binary = row(&list, "z.bin");
    let id = binary["id"].as_str().unwrap();
    assert!(!id.is_empty());
    assert!(
        id.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    );
    assert_eq!(
        *binary,
        serde_json::json!({
            "id": id, "path": "z.bin", "old_path": null, "status": "file",
            "staged_status": null, "worktree_status": null,
            "additions": null, "deletions": null,
            "staged_additions": null, "staged_deletions": null,
            "binary": true, "untracked": false, "size": 4,
        })
    );
    assert_eq!(row(&list, "a.txt")["binary"], false);
    assert_eq!(row(&list, "a.txt")["size"], 5);
    assert!(
        list["revision"]
            .as_str()
            .unwrap()
            .strip_prefix("plain:")
            .is_some_and(|hash| {
                !hash.is_empty() && hash.bytes().all(|b| b.is_ascii_hexdigit())
            })
    );
    assert_eq!(fixture.list("one").await.0, list);
    fixture.file("work/z.bin", "now text\n");
    let refreshed = fixture.list("one").await.0;
    assert_ne!(refreshed["revision"], list["revision"]);
    assert_eq!(row(&refreshed, "z.bin")["id"], id);
    assert_eq!(row(&refreshed, "z.bin")["binary"], false);
}

#[tokio::test]
async fn changes_plain_inventory_caps_files_and_wire_bytes() {
    let fixture = Fixture::without_repository();
    for index in (0..=FILE_LIMIT).rev() {
        fixture.file(&format!("count/file-{index:03}"), "");
        fixture.file(
            &format!("bytes/{}/{index:03}-{}", "\\".repeat(200), "\\".repeat(200)),
            "",
        );
    }
    for (id, expected) in [("count", FILE_LIMIT), ("bytes", FILE_LIMIT - 1)] {
        fixture.session(id, fixture.repo.join(id));
        let (list, bytes) = fixture.list(id).await;
        assert_eq!(list["truncated"], true, "{id}: {list}");
        assert_eq!(list["reason"], LIST_TRUNCATED);
        assert!(bytes <= LIST_LIMIT, "{id}: {bytes}");
        assert!(rows(&list).len() <= expected);
        if id == "count" {
            assert_eq!(rows(&list).len(), FILE_LIMIT);
            assert_eq!(rows(&list)[0]["path"], "file-000");
            assert_eq!(
                rows(&list)[FILE_LIMIT - 1]["path"],
                format!("file-{:03}", FILE_LIMIT - 1)
            );
        } else {
            assert!(bytes > LIST_LIMIT - 1500, "{bytes}");
        }
    }
}

#[tokio::test]
async fn changes_plain_catalog_reports_the_cwd_after_leaving_a_repository() {
    let fixture = Fixture::new();
    fixture.file("git.txt", "before\n");
    fixture.commit();
    fixture.file("git.txt", "Git inventory\n");
    fixture.session("one", fixture.repo.clone());
    let (previous, _) = fixture.list("one").await;
    assert_eq!(previous["source"], "working_tree");
    let tracked = row(&previous, "git.txt");
    assert_eq!(tracked["additions"], 1);
    assert_eq!(tracked["deletions"], 1);

    let cwd = fixture.scratch.0.join("local");
    fs::create_dir(&cwd).unwrap();
    fs::write(cwd.join("note.txt"), "plain inventory\n").unwrap();
    let outcome = run_session_cd(&fixture.state, "one", &format!("/cd {}", cwd.display())).unwrap();
    assert!(outcome.stderr.is_empty(), "{}", outcome.stderr);
    let (status, catalog, _) = fixture.get("/sessions/one/changes/scopes", true).await;
    assert_eq!(status, StatusCode::OK, "{catalog}");
    assert_eq!(catalog["source"], "plain");
    assert_eq!(catalog["current_path"], cwd.to_str().unwrap());
    assert_ne!(catalog["current_path"], previous["scope"]["path"]);
    let pwd = run_session_cd(&fixture.state, "one", "/pwd").unwrap();
    assert_eq!(
        pwd.stdout,
        format!("Working directory: {}\n", cwd.display())
    );
    assert_plain_routes(&fixture, &cwd, "not_a_repository").await;
}

#[tokio::test]
async fn changes_verifier_gates_missing_git_and_ignores_inherited_overrides() {
    const MODE: &str = "A_SERVE_CHANGES_GIT_TEST_MODE";
    const CWD: &str = "A_SERVE_CHANGES_GIT_TEST_CWD";
    const REAL_PATH: &str = "A_SERVE_CHANGES_GIT_TEST_REAL_PATH";
    match std::env::var(MODE).as_deref() {
        Ok("missing") => {
            let cwd = PathBuf::from(std::env::var_os(CWD).unwrap());
            assert_plain_routes(&Fixture::without_repository(), &cwd, "git_unavailable").await;
            return;
        }
        Ok("non-repository") => {
            let fixture = Fixture::without_repository();
            fixture.file("note.txt", "plain inventory\n");
            fixture.file(".git/local.txt", "not Git metadata\n");
            assert_plain_routes(&fixture, &fixture.repo, "not_a_repository").await;
            return;
        }
        Ok("discovery-error") => {
            let fixture = Fixture::new();
            fixture.file("sub/note.txt", "current scope inventory\n");
            fixture.session("one", fixture.repo.join("sub"));
            let (inventory, _) = fixture.list("one").await;
            row(&inventory, "note.txt");
            let (status, catalog, _) = fixture.get("/sessions/one/changes/scopes", true).await;
            assert_eq!(status, StatusCode::OK, "{catalog}");
            assert_eq!(catalog["git_available"], true);
            assert_eq!(catalog["truncated"], true);
            assert_eq!(catalog["reason"], SCOPES_FAILED);
            assert_eq!(catalog["current_path"], inventory["scope"]["path"]);
            assert_eq!(catalog["scopes"].as_array().unwrap().len(), 1);
            assert_eq!(catalog["scopes"][0]["files"], 1);
            return;
        }
        Ok("overrides") => {
            let fixture = Fixture::new();
            fixture.file("note.txt", "working and committed diff\n");
            let head = fixture.commit_named("selected repo");
            fixture.session("one", fixture.repo.clone());
            let (list, _) = fixture.list("one").await;
            assert_eq!(list["git_available"], true);
            let (scopes_status, scopes, _) =
                fixture.get("/sessions/one/changes/scopes", true).await;
            assert_eq!(scopes_status, StatusCode::OK);
            assert_eq!(scopes["git_available"], true);
            let (detail, _) = fixture.commits(&format!("/{head}")).await;
            assert!(
                detail["diff"]
                    .as_str()
                    .unwrap()
                    .contains("+working and committed diff\n")
            );
            return;
        }
        _ => {}
    }
    // Re-execute this test with a private environment, without changing the
    // process-wide environment of the other parallel tests.
    let scratch = Scratch::new("a-serve-changes-environment-").unwrap();
    let fake_bin = scratch.0.join("fake-bin");
    fs::create_dir(&fake_bin).unwrap();
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(fake_bin.join("git"))
        .unwrap()
        .write_all(format!(
            "#!/bin/sh\n[ -n \"${REAL_PATH}\" ] && [ \"$5\" != worktree ] || exit 99\nPATH=\"${REAL_PATH}\" exec git \"$@\"\n"
        ).as_bytes())
        .unwrap();
    let missing = Fixture::new();
    missing.file("note.txt", "tracked file without Git on PATH\n");
    missing.commit();
    let executable = std::env::current_exe().unwrap();
    let test = format!(
        "{}::changes_verifier_gates_missing_git_and_ignores_inherited_overrides",
        module_path!().split_once("::").unwrap().1
    );
    for mode in ["missing", "non-repository", "discovery-error", "overrides"] {
        let mut command = Command::new(&executable);
        command.args(["--exact", &test]).env(MODE, mode);
        match mode {
            "missing" => {
                command.env("PATH", &scratch.0).env(CWD, &missing.repo);
            }
            "non-repository" => {
                // Any Git invocation here would exit with an unexpected error.
                command.env("PATH", &fake_bin).env_remove(REAL_PATH);
            }
            "discovery-error" => {
                command
                    .env("PATH", &fake_bin)
                    .env(REAL_PATH, std::env::var_os("PATH").unwrap());
            }
            _ => {
                for key in [
                    "GIT_DIR",
                    "GIT_WORK_TREE",
                    "GIT_OBJECT_DIRECTORY",
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                    "GIT_COMMON_DIR",
                ] {
                    command.env(key, scratch.0.join("inherited-override"));
                }
            }
        }
        let output = crate::fork_guard::output(&mut command).unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[tokio::test]
async fn changes_verifier_reports_stale_cursor_with_the_current_head() {
    let fixture = Fixture::new();
    let root = fixture.commit_named("common root");
    let before = fixture.commit_named("old checkout");
    fixture.git(&["checkout", "--quiet", "--detach", &root]);
    let head = fixture.commit_named("new checkout");
    fixture.session("one", fixture.repo.clone());
    let (status, body, _) = fixture
        .get(
            &format!("/sessions/one/changes/commits?before={before}"),
            true,
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "stale_cursor");
    assert_eq!(body["head"], head);
    assert_eq!(body["scope"]["id"], "current");
    assert_eq!(body["rows"], serde_json::json!([]));
    assert_eq!(body["next_cursor"], Value::Null);
    assert_eq!(body["truncated"], false);
    assert!(body["reason"].is_string());
}
