use super::*;

pub(super) struct TurnWriteContext {
    pub(super) cwd: std::path::PathBuf,
    pub(super) warnings: Vec<String>,
}

/// The caller holds this session's async lock, including while reading cwd and grants.
pub(super) fn prepare_turn_write_context(
    state: &ServeState,
    session_id: &str,
    overrides: &mut TurnOverrides,
) -> Result<TurnWriteContext, String> {
    let stored = SessionStore::new(&state.history_file)
        .read_session_serve_config(session_id)
        .map_err(|error| format!("cannot read session writable directories: {error}"))?;
    merge_session_config(overrides, &stored);
    let (roots, mut warnings) = crate::ai::write_roots::available_roots(
        state
            .default_add_dirs
            .iter()
            .chain(&stored.additional_write_roots),
    );
    warnings.extend(stored.additional_write_root_warnings);
    overrides.add_dirs = roots;
    Ok(TurnWriteContext {
        cwd: session_cwd(state, session_id),
        warnings,
    })
}

/// Both JSON and streaming turns intercept the same user-owned directory command.
/// No child, model, image staging, or title generation is started for this branch.
pub(super) async fn dirs_command_response(
    state: &ServeState,
    session_id: &str,
    prompt: &str,
    streaming: bool,
) -> Option<axum::response::Response> {
    let command = crate::ai::write_roots::parse_dirs_command(prompt)?;
    let lock = session_lock(state, session_id).await;
    let _guard = lock.lock().await;
    let result = command.and_then(|command| {
        use crate::ai::write_roots::DirsCommand;
        let store = SessionStore::new(&state.history_file);
        let cwd = session_cwd(state, session_id);
        match command {
            DirsCommand::Add(value) => crate::ai::write_roots::add_session_directory(
                &store,
                session_id,
                &value,
                &cwd,
                &state.default_add_dirs,
            )
            .map(|(message, _)| message),
            DirsCommand::List => {
                let roots = if store.session_history_file(session_id).exists() {
                    store
                        .read_session_serve_config(session_id)
                        .map_err(|error| {
                            format!("cannot read session writable directories: {error}")
                        })?
                        .additional_write_roots
                } else {
                    Vec::new()
                };
                Ok(crate::ai::write_roots::writable_directory_report(
                    &cwd,
                    &state.default_add_dirs,
                    &roots,
                ))
            }
            DirsCommand::Delete(value) => {
                if !store.session_history_file(session_id).exists() {
                    return Err("this session has no writable directory grants".to_string());
                }
                crate::ai::write_roots::remove_session_directory(
                    &store,
                    session_id,
                    &value,
                    &cwd,
                    &state.default_add_dirs,
                )
                .map(|(message, _)| message)
            }
        }
    });
    if streaming {
        let event = match result {
            Ok(message) => {
                ServeSse::new("replace", serde_json::json!({"text": message}).to_string())
            }
            Err(error) => error_event(error),
        };
        let events = [event, done_event()]
            .into_iter()
            .map(|event| Ok::<_, std::convert::Infallible>(event.into_event()));
        Some(Sse::new(stream::iter(events)).into_response())
    } else {
        Some(match result {
            Ok(message) => Json(TurnResp {
                session_id: session_id.to_string(),
                output: message,
            })
            .into_response(),
            Err(error) => bad_request(error).into_response(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        root: std::path::PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::current_dir()
                .unwrap()
                .join(format!(".add-dir-serve-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            Self {
                root: root.canonicalize().unwrap(),
            }
        }
        fn directory(&self, name: &str) -> std::path::PathBuf {
            let path = self.root.join(name);
            std::fs::create_dir_all(&path).unwrap();
            path
        }
        fn state(&self) -> ServeState {
            ServeState {
                history_file: self.root.join("history.sqlite"),
                token: String::new(),
                default_add_dirs: Default::default(),
                cwds: Default::default(),
                locks: Default::default(),
                open_sessions: Default::default(),
                active_turns: Default::default(),
                confirms: Default::default(),
                events: Default::default(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
    async fn body(response: axum::response::Response) -> String {
        String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }
    fn request(prompt: String) -> TurnReq {
        serde_json::from_value(serde_json::json!({"prompt": prompt})).unwrap()
    }

    #[tokio::test]
    async fn add_dir_serve_json_and_sse_intercept_commands_and_persist_per_session() {
        let fixture = Fixture::new();
        let base = fixture.directory("base");
        let project = fixture.directory("base/project with spaces");
        let state = fixture.state();
        run_session_cd(&state, "a", &format!("/cd {}", base.display())).unwrap();
        let response = super::super::post_turn(
            State(state.clone()),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(" /add-dir 'project with spaces' ".into())),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body(response).await).unwrap();
        assert_eq!(json["session_id"], "a");
        assert!(
            json["output"]
                .as_str()
                .unwrap()
                .contains("Added session writable directory")
        );
        assert!(
            json["output"]
                .as_str()
                .unwrap()
                .contains(project.to_str().unwrap())
        );
        assert!(json.get("stdout").is_none());
        let response = super::super::post_turn_sse(
            State(state.clone()),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(format!("/add-dir \"{}\"", project.display()))),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let output = body(response).await;
        assert!(output.contains("event: replace"), "{output}");
        assert!(output.contains("already added"), "{output}");
        assert!(output.contains("event: done"), "{output}");
        assert!(state.active_turns.lock().unwrap().is_empty());
        let restarted = fixture.state();
        *restarted.cwds.lock().unwrap() = load_cwds(&restarted.history_file);
        assert_eq!(session_cwd(&restarted, "a"), base);
        let store = SessionStore::new(&restarted.history_file);
        assert_eq!(
            store
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots,
            [project]
        );
        assert!(
            store
                .read_session_serve_config("b")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
    }

    #[tokio::test]
    async fn add_dir_serve_invalid_commands_and_storage_failures_never_report_success() {
        let fixture = Fixture::new();
        let state = fixture.state();
        for prompt in [
            "/add-dir".to_string(),
            format!("/add-dir {}", fixture.root.join("missing").display()),
            "/add-dir a\nedit a".into(),
        ] {
            let response = super::super::post_turn(
                State(state.clone()),
                HeaderMap::new(),
                Path("a".into()),
                Json(request(prompt.clone())),
            )
            .await
            .into_response();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let response = super::super::post_turn_sse(
                State(state.clone()),
                HeaderMap::new(),
                Path("a".into()),
                Json(request(prompt)),
            )
            .await
            .into_response();
            let output = body(response).await;
            assert!(output.contains("event: error"), "{output}");
            assert!(output.contains("event: done"), "{output}");
            assert!(!output.contains("Added session writable directory"));
        }
        let project = fixture.directory("project");
        let mut failed = fixture.state();
        let blocker = fixture.root.join("not-a-directory");
        std::fs::write(&blocker, "file").unwrap();
        failed.history_file = blocker.join("history.sqlite");
        let response = super::super::post_turn(
            State(failed),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(format!("/add-dir {}", project.display()))),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(body(response).await.contains("cannot save"));
        assert!(!fixture.root.join("missing").exists());
    }

    #[tokio::test]
    async fn dirs_serve_lists_and_deletes_session_grants() {
        let fixture = Fixture::new();
        let project = fixture.directory("project");
        let state = fixture.state();
        let response = super::super::post_turn(
            State(state.clone()),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(format!("/dirs add {}", project.display()))),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let response = super::super::post_turn(
            State(state.clone()),
            HeaderMap::new(),
            Path("a".into()),
            Json(request("/dirs".into())),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let output = body(response).await;
        assert!(output.contains("session grants:"), "{output}");
        assert!(output.contains(project.to_str().unwrap()), "{output}");
        let response = super::super::post_turn(
            State(state.clone()),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(format!("/dirs del {}", project.display()))),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            body(response)
                .await
                .contains("Removed session writable directory")
        );
        assert!(
            SessionStore::new(&state.history_file)
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
        let response = super::super::post_turn(
            State(state),
            HeaderMap::new(),
            Path("a".into()),
            Json(request(format!("/dirs del {}", project.display()))),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn add_dir_serve_resolves_cwd_only_after_the_session_lock() {
        let fixture = Fixture::new();
        let old = fixture.directory("old");
        let new = fixture.directory("new");
        let project = fixture.directory("new/project");
        let state = fixture.state();
        run_session_cd(&state, "a", &format!("/cd {}", old.display())).unwrap();
        let lock = session_lock(&state, "a").await;
        let guard = lock.lock().await;
        let queued_state = state.clone();
        let queued = tokio::spawn(async move {
            super::super::post_turn(
                State(queued_state),
                HeaderMap::new(),
                Path("a".into()),
                Json(request("/add-dir project".into())),
            )
            .await
            .into_response()
        });
        tokio::task::yield_now().await;
        assert!(!queued.is_finished());
        run_session_cd(&state, "a", &format!("/cd {}", new.display())).unwrap();
        drop(guard);
        let response = queued.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            body(response)
                .await
                .contains(&project.display().to_string())
        );
    }

    #[tokio::test]
    async fn add_dir_serve_child_argv_and_actual_writes_are_isolated() {
        use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext, SUBAGENT_CWD};
        struct RestoreConfig(Option<std::ffi::OsString>);
        impl Drop for RestoreConfig {
            fn drop(&mut self) {
                match self.0.take() {
                    Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
                    None => unsafe { std::env::remove_var("CONFIGW_PATH") },
                }
                crate::commonw::configw::refresh();
            }
        }
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let fixture = Fixture::new();
        let cwd_a = fixture.directory("cwd-a");
        let cwd_b = fixture.directory("cwd-b");
        let shared = fixture.directory("shared");
        let extra = fixture.directory("extra-a");
        let mut state = fixture.state();
        state.default_add_dirs = Arc::new(vec![shared.clone()]);
        run_session_cd(&state, "a", &format!("/cd {}", cwd_a.display())).unwrap();
        run_session_cd(&state, "b", &format!("/cd {}", cwd_b.display())).unwrap();
        SessionStore::new(&state.history_file)
            .add_session_write_root("a", &extra)
            .unwrap();
        let config_path = fixture.root.join("empty.configw");
        std::fs::write(&config_path, "").unwrap();
        let restore = RestoreConfig(std::env::var_os("CONFIGW_PATH"));
        unsafe { std::env::set_var("CONFIGW_PATH", &config_path) };
        crate::commonw::configw::refresh();
        let target = extra.join("result.txt");
        for (id, allowed) in [("a", true), ("b", false)] {
            let lock = session_lock(&state, id).await;
            let _session_guard = lock.lock().await;
            let mut overrides = TurnOverrides::default();
            let context = prepare_turn_write_context(&state, id, &mut overrides).unwrap();
            assert!(overrides.add_dirs.contains(&shared));
            assert_eq!(overrides.add_dirs.contains(&extra), allowed);
            let mut cmd = std::process::Command::new("a");
            push_turn_args(&mut cmd, id, "--add-dir is prompt text", &overrides);
            let args: Vec<String> = cmd
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();
            let delimiter = args.iter().position(|arg| arg == "--").unwrap();
            assert!(args[..delimiter].iter().any(|arg| arg == "--add-dir"));
            assert_eq!(args[delimiter + 1], "--add-dir is prompt text");
            let mut app = crate::ai::tests::test_app_with_cancel_stream(Arc::new(
                std::sync::atomic::AtomicBool::new(false),
            ));
            app.cli = crate::ai::cli::try_parse_cli_args(std::iter::once("a".into()).chain(args))
                .unwrap();
            app.session_id = id.into();
            app.config.history_file = state.history_file.clone();
            app.session_history_file =
                SessionStore::new(&state.history_file).session_history_file(id);
            let driver = DriverContext::new(
                app,
                Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
                Arc::new(Vec::new()),
                Arc::new(Vec::new()),
            );
            let result = DRIVER_CTX.sync_scope(driver, || {
                SUBAGENT_CWD.sync_scope(context.cwd, || {
                    crate::ai::tools::service::file::execute_write_file(
                        &serde_json::json!({"file_path": target, "content": id}),
                    )
                })
            });
            assert_eq!(result.is_ok(), allowed, "{result:?}");
        }
        drop(restore);
        assert_eq!(std::fs::read_to_string(target).unwrap(), "a");
    }

    #[tokio::test]
    async fn add_dir_serve_default_root_recovers_when_mount_returns() {
        let fixture = Fixture::new();
        let mount = fixture.directory("mounted-project");
        let raw = serde_json::to_string(std::slice::from_ref(&mount)).unwrap();
        std::fs::remove_dir(&mount).unwrap();
        let mut state = fixture.state();
        state.default_add_dirs =
            Arc::new(super::super::ctl::parse_runtime_default_add_dirs(&raw).unwrap());
        let lock = session_lock(&state, "a").await;
        let _guard = lock.lock().await;
        let mut overrides = TurnOverrides::default();
        assert!(
            !prepare_turn_write_context(&state, "a", &mut overrides)
                .unwrap()
                .warnings
                .is_empty()
        );
        assert!(overrides.add_dirs.is_empty());
        std::fs::create_dir(&mount).unwrap();
        assert!(
            prepare_turn_write_context(&state, "a", &mut overrides)
                .unwrap()
                .warnings
                .is_empty()
        );
        assert_eq!(overrides.add_dirs, [mount]);
    }

    #[tokio::test]
    async fn add_dir_serve_request_authority_is_independent_of_daemon_origin() {
        if std::env::var_os(crate::ai::write_roots::AGENT_COMMAND_ENV).is_some() {
            let fixture = Fixture::new();
            let project = fixture.directory("project");
            let mut state = fixture.state();
            state.token = "root-grant-test-token".into();
            let mut headers = HeaderMap::new();
            headers.insert(
                axum::http::header::AUTHORIZATION,
                "Bearer root-grant-test-token".parse().unwrap(),
            );
            let response = super::super::post_turn(
                State(state.clone()),
                headers.clone(),
                Path("a".into()),
                Json(request(format!("/add-dir {}", project.display()))),
            )
            .await
            .into_response();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                body(response)
                    .await
                    .contains("Added session writable directory")
            );
            let response = super::super::post_turn_sse(
                State(state),
                headers,
                Path("a".into()),
                Json(request(format!("/add-dir {}", project.display()))),
            )
            .await
            .into_response();
            assert!(body(response).await.contains("already added"));
            return;
        }
        let output = tokio::task::spawn_blocking(|| {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "ai::serve::write_roots::tests::add_dir_serve_request_authority_is_independent_of_daemon_origin", "--nocapture"])
                .env(crate::ai::write_roots::AGENT_COMMAND_ENV, "1").output().unwrap()
        }).await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }

    #[tokio::test]
    async fn add_dir_serve_skips_missing_grants_without_forgetting_them() {
        let fixture = Fixture::new();
        let missing = fixture.directory("disconnected");
        let state = fixture.state();
        let store = SessionStore::new(&state.history_file);
        store.add_session_write_root("a", &missing).unwrap();
        std::fs::remove_dir(&missing).unwrap();
        let lock = session_lock(&state, "a").await;
        let _guard = lock.lock().await;
        let mut overrides = TurnOverrides::default();
        let context = prepare_turn_write_context(&state, "a", &mut overrides).unwrap();
        assert!(overrides.add_dirs.is_empty());
        assert!(
            context
                .warnings
                .iter()
                .any(|warning| warning.contains(&missing.display().to_string()))
        );
        assert_eq!(
            store
                .read_session_serve_config("a")
                .unwrap()
                .additional_write_roots,
            [missing]
        );
    }
}
