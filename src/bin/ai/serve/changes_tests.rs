//! Route tests use isolated worktrees and synthetic session metadata.

use super::*;
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use hyper::service::Service;
use hyper_util::service::TowerToHyperService;
use serde_json::Value;

use crate::ai::serve::{SessionCwd, SessionTitleOrigin, server_info};

struct Fixture {
    scratch: Scratch,
    repo: PathBuf,
    state: ServeState,
}

impl Fixture {
    fn new() -> Self {
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
        let fixture = Self {
            scratch,
            repo,
            state,
        };
        fixture.git(&["init", "--quiet", "--template="]);
        fixture
    }

    fn git(&self, args: &[&str]) {
        let git = Git {
            deadline: Instant::now() + TIME_LIMIT,
        };
        let out = crate::fork_guard::output(
            git.command(&self.repo)
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

    fn session(&self, id: &str, cwd: PathBuf) {
        SessionStore::new(&self.state.history_file)
            .write_session_title_with_origin(id, "Changes test", SessionTitleOrigin::User)
            .unwrap();
        self.state.cwds.lock().unwrap().insert(
            id.to_string(),
            SessionCwd {
                current: cwd,
                previous: None,
            },
        );
    }

    async fn get(&self, uri: &str, authenticated: bool) -> (StatusCode, Value, usize) {
        // Concurrent fixtures share the route's process-wide request limit.
        static REQUEST: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let _request = REQUEST.lock().await;
        let app =
            routes(Router::new().route("/info", get(server_info))).with_state(self.state.clone());
        let mut request = Request::builder().uri(uri);
        if authenticated {
            request = request.header(header::AUTHORIZATION, "Bearer test-token");
        }
        let response = TowerToHyperService::new(app)
            .call(request.body(Body::empty()).unwrap())
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

    async fn list(&self, id: &str) -> (Value, usize) {
        let (status, value, size) = self.get(&format!("/sessions/{id}/changes"), true).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        (value, size)
    }

    async fn detail(&self, id: &str, list: &Value, path: &str) -> (Value, usize) {
        let (status, detail, size) = self
            .get(
                &format!(
                    "/sessions/{id}/changes/{}?revision={}",
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
            serde_json::json!({"error": "unknown changes scope"})
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
            serde_json::json!({"error": "unknown changes scope"})
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
    let unknown_scope = serde_json::json!({"error": "unknown changes scope"});
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
async fn changes_route_rejects_unset_cwd() {
    let fixture = Fixture::new();
    let id = "unset-cwd";
    SessionStore::new(&fixture.state.history_file)
        .write_session_title_with_origin(id, "Changes test", SessionTitleOrigin::User)
        .unwrap();

    // Catch a fallback even if the test process itself is outside a worktree.
    assert!(matches!(Scope::cwd(&fixture.state, id), Err(Failure::Cwd)));
    for uri in [
        format!("/sessions/{id}/changes"),
        format!("/sessions/{id}/changes/scopes"),
        format!("/sessions/{id}/changes?scope=unknown"),
        format!("/sessions/{id}/changes/unknown?revision=unused"),
    ] {
        let (status, body, _) = fixture.get(&uri, true).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {body}");
        assert_eq!(
            body["error"],
            "session cwd is unavailable or not a Git working tree"
        );
    }
}

#[tokio::test]
async fn changes_route_auth_and_revision_isolation() {
    let fixture = Fixture::new();
    fixture.file("note.txt", "first\n");
    fixture.session("one", fixture.repo.clone());
    fixture.session("two", fixture.repo.clone());
    fixture.session("no-git", fixture.scratch.0.clone());
    fixture.session("gone", fixture.repo.join("removed"));
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
    for id in ["no-git", "gone", "aliased"] {
        assert_eq!(
            fixture
                .get(&format!("/sessions/{id}/changes"), true)
                .await
                .0,
            StatusCode::CONFLICT
        );
        assert_eq!(
            fixture
                .get(&format!("/sessions/{id}/changes/no-id?revision=x"), true)
                .await
                .0,
            StatusCode::CONFLICT
        );
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
    fixture.commit();
    fixture.session("one", fixture.repo.join("work"));

    fixture.file("work/modified.txt", "staged\n");
    fixture.git(&["add", "--", "work/modified.txt"]);
    fixture.file("work/modified.txt", "working\n");
    fs::remove_file(fixture.repo.join("work/deleted.txt")).unwrap();
    fs::remove_file(fixture.repo.join("work/deleted.bin")).unwrap();
    fixture.git(&["mv", "--", "work/old name.txt", "work/new name.txt"]);
    fixture.git(&["mv", "--", "outside.txt", "work/incoming.txt"]);
    fixture.file("work/incoming.txt", "now inside\n");
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
            "incoming.txt",
            "invalid.bin",
            "link.txt",
            "modified.txt",
            "new name.txt",
            "new.txt",
            "redirect",
        ]
    );
    for (path, status) in [
        ("deleted.txt", "deleted"),
        ("modified.txt", "modified"),
        ("new name.txt", "renamed"),
        ("new.txt", "added"),
    ] {
        assert_eq!(row(&list, path)["status"], status);
    }
    assert_eq!(row(&list, "new name.txt")["old_path"], "old name.txt");
    assert!(row(&list, "incoming.txt")["old_path"].is_null());

    let (detail, _) = fixture.detail("one", &list, "modified.txt").await;
    assert_eq!(detail["format"], "unified");
    let patch = detail["diff"].as_str().unwrap();
    assert!(patch.contains("--- a/modified.txt\n+++ b/modified.txt\n@@ "));
    assert!(patch.contains("-committed\n+working\n"));
    assert!(!patch.contains("staged"));
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
