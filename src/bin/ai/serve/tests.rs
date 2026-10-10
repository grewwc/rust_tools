//! Serve-mode integration tests.

use super::*;

    use super::{ServeFrameDecoder, ServeImageUpload, ServeLiveEvent, setup_live_fifo};
    use super::{MAX_TURN_IMAGES, clean_sse_line, stage_turn_images};
    use super::{TurnReq, turn_overrides};
    #[cfg(unix)]
    use super::pump_live_fifo;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use uuid::Uuid;

    /// Handler-level coverage for the two session-lifecycle routes: a
    /// missing fork source 404s, a malformed id 400s, and deleting a
    /// missing session still succeeds (idempotent, so client retries after
    /// a dropped connection stay safe). Body shapes are covered by the
    /// client loopback tests in `chat.rs`; store behavior by the
    /// `SessionStore` fork/delete tests.
    fn lifecycle_test_state() -> super::ServeState {
        let root = std::env::temp_dir().join(format!(
            "a-serve-lifecycle-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        super::ServeState {
            history_file: root.join("history.sqlite"),
            workspace_root: root.join("workspace"),
            token: String::new(),
            locks: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
        }
    }

    /// Preview-path resolution is the security-bearing piece of
    /// `GET /sessions/{id}/file`: both roots accept images, and everything
    /// else (traversal, foreign paths, non-images, directories, oversized
    /// files) is refused with a distinguishable reason.
    #[test]
    fn resolve_preview_accepts_images_inside_the_roots_only() {
        use super::{MAX_PREVIEW_BYTES, PreviewReject, resolve_preview};
        let base = std::env::temp_dir().join(format!(
            "a-serve-preview-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let workspace = base.join("workspace");
        let assets = base.join("s1.assets");
        std::fs::create_dir_all(&workspace).expect("workspace");
        std::fs::create_dir_all(&assets).expect("assets");
        std::fs::write(workspace.join("chart.svg"), b"<svg/>").expect("chart");
        std::fs::write(workspace.join("shot.PNG"), b"png").expect("shot");
        std::fs::write(workspace.join("notes.txt"), b"nope").expect("notes");
        std::fs::write(assets.join("paste-1.webp"), b"webp").expect("paste");
        std::fs::write(base.join("outside.png"), b"outside").expect("outside");
        let roots = vec![workspace.clone(), assets.clone()];

        let got = resolve_preview("chart.svg", &roots).expect("relative workspace path");
        assert_eq!(got.content_type, "image/svg+xml");
        assert_eq!(
            got.path,
            workspace.join("chart.svg").canonicalize().expect("canon")
        );
        assert_eq!(
            resolve_preview("shot.PNG", &roots).expect("uppercase ext").content_type,
            "image/png",
            "the allowlist is case-insensitive"
        );
        // Absolute paths are checked against the same roots, so a file that
        // only lives in the session assets dir is still servable.
        let abs = assets.join("paste-1.webp");
        assert_eq!(
            resolve_preview(abs.to_str().expect("utf8"), &roots)
                .expect("assets path")
                .content_type,
            "image/webp"
        );

        assert_eq!(
            resolve_preview("", &roots).unwrap_err(),
            PreviewReject::BadRequest
        );
        assert_eq!(
            resolve_preview("notes.txt", &roots).unwrap_err(),
            PreviewReject::BadRequest,
            "non-image extensions stay invisible even inside the root"
        );
        assert_eq!(
            resolve_preview("missing.png", &roots).unwrap_err(),
            PreviewReject::Missing
        );
        assert_eq!(
            resolve_preview("../outside.png", &roots).unwrap_err(),
            PreviewReject::Outside,
            "a traversal out of the root must not resolve"
        );
        assert_eq!(
            resolve_preview(base.join("outside.png").to_str().expect("utf8"), &roots).unwrap_err(),
            PreviewReject::Outside,
            "an absolute path outside both roots is refused"
        );
        std::fs::create_dir_all(workspace.join("dir.png")).expect("dir");
        assert_eq!(
            resolve_preview("dir.png", &roots).unwrap_err(),
            PreviewReject::Missing,
            "a directory is not a previewable file"
        );
        let big = workspace.join("big.png");
        std::fs::File::create(&big)
            .expect("big")
            .set_len(MAX_PREVIEW_BYTES + 1)
            .expect("len");
        assert_eq!(
            resolve_preview("big.png", &roots).unwrap_err(),
            PreviewReject::TooLarge
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("outside.png"), workspace.join("link.png"))
                .expect("symlink");
            assert_eq!(
                resolve_preview("link.png", &roots).unwrap_err(),
                PreviewReject::Outside,
                "a symlink out of the root is refused like a traversal"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The preview route end to end: bearer auth, the bytes and content type
    /// the page needs, SVG hardening, and every request-level refusal.
    #[tokio::test]
    async fn file_route_serves_images_and_guards_access() {
        use axum::{http::StatusCode, response::IntoResponse as _};

        use super::SessionStore;

        async fn call(state: super::ServeState, auth: bool, path: &str) -> super::Response {
            use axum::response::IntoResponse as _;
            let mut headers = axum::http::HeaderMap::new();
            if auth {
                headers.insert(
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderValue::from_static("Bearer t"),
                );
            }
            super::get_session_file(
                axum::extract::State(state),
                headers,
                axum::extract::Path("s1".to_string()),
                axum::extract::Query(super::FileQuery {
                    path: path.to_string(),
                }),
            )
            .await
            .into_response()
        }
        let base = std::env::temp_dir().join(format!(
            "a-serve-file-route-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let workspace = base.join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace");
        std::fs::write(workspace.join("chart.svg"), b"<svg/>").expect("svg");
        std::fs::write(workspace.join("notes.txt"), b"nope").expect("notes");
        std::fs::write(base.join("outside.png"), b"outside").expect("outside");
        let history_file = base.join("history.sqlite");
        let assets = SessionStore::new(&history_file).session_assets_dir("s1");
        std::fs::create_dir_all(&assets).expect("assets");
        std::fs::write(assets.join("pasted.png"), b"png-bytes").expect("pasted");
        let state = super::ServeState {
            history_file,
            workspace_root: workspace,
            token: "t".to_string(),
            locks: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
        };

        let resp = call(state.clone(), false, "chart.svg").await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = call(state.clone(), true, "chart.svg").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/svg+xml")
        );
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_SECURITY_POLICY)
                .and_then(|v| v.to_str().ok()),
            Some("default-src 'none'; style-src 'unsafe-inline'; sandbox"),
            "an SVG preview must not be able to script this origin"
        );
        assert_eq!(
            resp.headers()
                .get(axum::http::header::X_CONTENT_TYPE_OPTIONS)
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .expect("body");
        assert_eq!(&body[..], b"<svg/>");

        let pasted = assets.join("pasted.png");
        let resp = call(state.clone(), true, pasted.to_str().expect("utf8")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/png"),
            "session assets are the second allowed root"
        );
        assert!(
            resp.headers()
                .get(axum::http::header::CONTENT_SECURITY_POLICY)
                .is_none(),
            "only SVG carries the sandbox header"
        );

        let resp = call(state.clone(), true, "notes.txt").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = call(state.clone(), true, "missing.png").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let resp = call(
            state.clone(),
            true,
            base.join("outside.png").to_str().expect("utf8"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        let mut bad_id_state = state.clone();
        bad_id_state.token = String::new();
        let resp = super::get_session_file(
            axum::extract::State(bad_id_state),
            axum::http::HeaderMap::new(),
            axum::extract::Path("bad id".to_string()),
            axum::extract::Query(super::FileQuery {
                path: "chart.svg".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Confirmation frames must survive the decoder like any other kind, with
    /// their JSON payload relayed verbatim (the client parses it as JSON).
    #[test]
    fn decoder_passes_confirmation_frames_through() {
        use crate::ai::background::ServeLiveKind;
        let wire = [
            test_frame(
                ServeLiveKind::ConfirmRequest,
                br#"{"id":7,"prompt":"proceed?"}"#,
            ),
            test_frame(ServeLiveKind::ConfirmDone, br#"{"id":7}"#),
        ]
        .concat();
        let mut decoder = ServeFrameDecoder::default();
        assert_eq!(
            decoder.push(&wire),
            vec![
                ServeLiveEvent::ConfirmRequest(r#"{"id":7,"prompt":"proceed?"}"#.to_string()),
                ServeLiveEvent::ConfirmDone(r#"{"id":7}"#.to_string()),
            ]
        );
    }

    /// The `/confirm` route pair: the pending question is readable, an answer
    /// reaches the turn child's stdin exactly once, and answering the same
    /// question again is refused instead of writing a second line.
    #[cfg(unix)]
    #[tokio::test]
    async fn confirm_answer_reaches_the_child_stdin_once() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        // Stand-in turn child: consumes one stdin line and echoes it, which is
        // what the real child's confirmation reader does.
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("read line; echo \"$line\"")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn stand-in child");
        let echo = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut text = String::new();
                let _ = out.read_to_string(&mut text);
                text
            })
        });
        let state = lifecycle_test_state();
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 7,
                    prompt: "proceed?".to_string(),
                    token: 11,
                }),
                answers: child.stdin.take(),
            })),
        );

        let read =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        assert_eq!(read.status(), StatusCode::OK);
        let body = axum::body::to_bytes(read.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["pending"]["id"], 7);
        assert_eq!(view["pending"]["prompt"], "proceed?");
        assert_eq!(view["pending"]["token"], 11);

        let answer = |allow| super::ConfirmAnswerReq { id: 7, token: 11, allow };
        let ok = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(true)),
        )
        .await
        .into_response();
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(echo.and_then(|handle| handle.join().ok()).as_deref(), Some("yes\n"));
        // The question is gone, so a second tap (or a second device) is
        // refused rather than writing another line.
        let again = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(false)),
        )
        .await
        .into_response();
        assert_eq!(again.status(), StatusCode::CONFLICT);
        // A later turn numbers its questions from 1 again, so the same id with
        // a fresh token is a different question: an answer carrying the stale
        // token must be refused instead of deciding text nobody read.
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 7,
                    prompt: "a different question".to_string(),
                    token: 12,
                }),
                answers: None,
            })),
        );
        let stale = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(true)),
        )
        .await
        .into_response();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        let body = axum::body::to_bytes(stale.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["error"], "no such confirmation is pending");
        let _ = child.wait();
    }

    #[tokio::test]
    async fn fork_missing_source_returns_not_found() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::fork_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("missing".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn fork_invalid_id_is_rejected() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::fork_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("../evil".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn side_note_queues_without_prior_history() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        // No history file exists yet: the note must still queue, since it can
        // arrive before the turn child persists its first message.
        let resp = super::post_side_note(
            State(state.clone()),
            HeaderMap::new(),
            Path("fresh".to_string()),
            Json(super::SideNoteReq {
                text: "  steer left  ".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["queued"], true);
        let store = super::SessionStore::new(state.history_file.as_path());
        let notes = super::super::driver::side_note::drain_side_notes(
            &store.session_history_file("fresh"),
            None,
        );
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].content, "steer left");
    }

    #[tokio::test]
    async fn side_note_rejects_empty_text_and_bad_id() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        let resp = super::post_side_note(
            State(state.clone()),
            HeaderMap::new(),
            Path("fresh".to_string()),
            Json(super::SideNoteReq {
                text: "   ".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = super::post_side_note(
            State(state.clone()),
            HeaderMap::new(),
            Path("../evil".to_string()),
            Json(super::SideNoteReq {
                text: "x".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_unknown_session_still_succeeds() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::delete_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("missing".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rename_session_persists_user_title() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        let resp = super::set_session_title(
            State(state.clone()),
            HeaderMap::new(),
            Path("rename-me".to_string()),
            Json(super::SetTitleReq {
                title: "  我的新标题  ".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.get("title").and_then(|t| t.as_str()), Some("我的新标题"));
        let store = super::SessionStore::new(state.history_file.as_path());
        let saved = store
            .read_session_title_with_origin("rename-me")
            .expect("read title");
        let saved = saved.expect("title persisted");
        assert_eq!(saved.text, "我的新标题");
        assert_eq!(
            saved.origin,
            super::SessionTitleOrigin::User,
            "renamed title must survive background auto-generation"
        );
    }

    #[tokio::test]
    async fn rename_session_rejects_bad_input() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let overlong = "x".repeat(super::MAX_SESSION_TITLE_CHARS + 1);
        for (id, title, want) in [
            ("ok-id", "", StatusCode::BAD_REQUEST),
            ("ok-id", "   ", StatusCode::BAD_REQUEST),
            ("ok-id", overlong.as_str(), StatusCode::BAD_REQUEST),
            ("../evil", "hi", StatusCode::BAD_REQUEST),
        ] {
            let resp = super::set_session_title(
                State(lifecycle_test_state()),
                HeaderMap::new(),
                Path(id.to_string()),
                Json(super::SetTitleReq {
                    title: title.to_string(),
                }),
            )
            .await
            .into_response();
            assert_eq!(resp.status(), want, "id={id:?} title_len={}", title.len());
        }
    }

    /// The per-session config routes: a fresh session reads as all-defaults,
    /// a POST persists exactly what was set, a partial POST merges with the
    /// stored values, and a blank field clears its key without touching the
    /// others.
    #[tokio::test]
    async fn session_config_roundtrips_and_merges_per_field() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        // Fresh session: all defaults.
        let resp = super::get_session_config(
            State(state.clone()),
            HeaderMap::new(),
            Path("cfg-me".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["model"], "");
        assert_eq!(v["agent"], "");
        assert_eq!(v["reasoning_effort"], "");
        // Set model + effort in one POST; agent stays default.
        let resp = super::set_session_config(
            State(state.clone()),
            HeaderMap::new(),
            Path("cfg-me".to_string()),
            Json(super::SessionConfigReq {
                model: Some("deepseek-r1".to_string()),
                agent: None,
                reasoning_effort: Some("high".to_string()),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        // Partial POST adds the agent without touching the other two.
        let resp = super::set_session_config(
            State(state.clone()),
            HeaderMap::new(),
            Path("cfg-me".to_string()),
            Json(super::SessionConfigReq {
                model: None,
                agent: Some("build".to_string()),
                reasoning_effort: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let store = super::SessionStore::new(state.history_file.as_path());
        let saved = store
            .read_session_serve_config("cfg-me")
            .expect("read config");
        assert_eq!(saved.model.as_deref(), Some("deepseek-r1"));
        assert_eq!(saved.agent.as_deref(), Some("build"));
        assert_eq!(saved.reasoning_effort.as_deref(), Some("high"));
        // A blank field clears it; the others survive.
        let resp = super::set_session_config(
            State(state.clone()),
            HeaderMap::new(),
            Path("cfg-me".to_string()),
            Json(super::SessionConfigReq {
                model: Some(String::new()),
                agent: None,
                reasoning_effort: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let saved = store
            .read_session_serve_config("cfg-me")
            .expect("read config");
        assert_eq!(saved.model, None);
        assert_eq!(saved.agent.as_deref(), Some("build"));
        assert_eq!(saved.reasoning_effort.as_deref(), Some("high"));
        // GET reflects the cleared field.
        let resp = super::get_session_config(
            State(state.clone()),
            HeaderMap::new(),
            Path("cfg-me".to_string()),
        )
        .await
        .into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["model"], "");
        assert_eq!(v["agent"], "build");
        assert_eq!(v["reasoning_effort"], "high");
    }

    /// Config updates are validated like per-turn overrides: unknown effort
    /// levels and flag-shaped or oversized names are refused with 400, and the
    /// rejected write changes nothing.
    #[tokio::test]
    async fn session_config_rejects_invalid_values() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        for (req, want) in [
            (
                super::SessionConfigReq {
                    model: Some("--evil".to_string()),
                    agent: None,
                    reasoning_effort: None,
                },
                StatusCode::BAD_REQUEST,
            ),
            (
                super::SessionConfigReq {
                    model: Some("x".repeat(super::MAX_TURN_OVERRIDE_CHARS + 1)),
                    agent: None,
                    reasoning_effort: None,
                },
                StatusCode::BAD_REQUEST,
            ),
            (
                super::SessionConfigReq {
                    model: None,
                    agent: None,
                    reasoning_effort: Some("turbo".to_string()),
                },
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let resp = super::set_session_config(
                State(state.clone()),
                HeaderMap::new(),
                Path("cfg-reject".to_string()),
                Json(req),
            )
            .await
            .into_response();
            assert_eq!(resp.status(), want);
        }
        let store = super::SessionStore::new(state.history_file.as_path());
        let saved = store
            .read_session_serve_config("cfg-reject")
            .expect("read config");
        assert_eq!(saved, super::SessionServeConfig::default());
    }

    /// The turn-override merge: an explicit per-turn pick wins, and a field
    /// the request left absent falls back to the session's stored config.
    #[test]
    fn merge_session_config_fills_absent_fields_only() {
        let stored = super::SessionServeConfig {
            model: Some("stored-model".to_string()),
            agent: Some("stored-agent".to_string()),
            reasoning_effort: Some("low".to_string()),
        };
        let mut overrides = super::TurnOverrides {
            model: Some("explicit".to_string()),
            agent: None,
            reasoning_effort: None,
        };
        super::merge_session_config(&mut overrides, &stored);
        assert_eq!(
            overrides.model.as_deref(),
            Some("explicit"),
            "an explicit per-turn override wins over the stored config"
        );
        assert_eq!(overrides.agent.as_deref(), Some("stored-agent"));
        assert_eq!(overrides.reasoning_effort.as_deref(), Some("low"));
        // All absent + an empty stored config keeps every field None, which
        // push_turn_override_args turns into "no flag, server default".
        let mut overrides = super::TurnOverrides::default();
        super::merge_session_config(&mut overrides, &super::SessionServeConfig::default());
        assert_eq!(overrides, super::TurnOverrides::default());
    }

    /// Seed helper for the rewind tests: a fixed-id session holding the
    /// given roles in order, like one built by real chat traffic.
    fn seed_rewind_session(state: &super::ServeState, id: &str, roles: &[&str]) {
        use crate::ai::history::{Message, append_history_messages};
        let store = super::SessionStore::new(state.history_file.as_path());
        store.ensure_root_dir().expect("root dir");
        let path = store.session_history_file(id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("session dir");
        }
        let messages: Vec<Message> = roles
            .iter()
            .map(|role| Message {
                role: role.to_string(),
                content: serde_json::Value::String(format!("{role} says hi")),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect();
        append_history_messages(&path, &messages).expect("seed messages");
    }

    #[tokio::test]
    async fn rewind_removes_anchor_and_everything_after_it() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant", "user", "assistant"]);
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 2 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.get("removed").and_then(|n| n.as_u64()), Some(2));
        assert_eq!(v.get("kept").and_then(|n| n.as_u64()), Some(2));
        let store = super::SessionStore::new(state.history_file.as_path());
        let rest = store.read_all_messages("s1").expect("read back");
        assert_eq!(rest.len(), 2);
        assert_eq!(rest[0].role, "user");
        assert_eq!(rest[1].role, "assistant");
    }

    #[tokio::test]
    async fn rewind_rejects_non_user_out_of_range_and_missing() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant"]);
        // Assistant anchor and past-the-end indexes must not truncate.
        for index in [1usize, 2, 99] {
            let resp = super::rewind_history(
                State(state.clone()),
                HeaderMap::new(),
                Path("s1".to_string()),
                Json(super::RewindReq {
                    message_index: index,
                }),
            )
            .await
            .into_response();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "index={index}");
        }
        // A missing session reads back as empty history (same as `read_history`),
        // so rewinding it is an out-of-range anchor, not a 404.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("missing".to_string()),
            Json(super::RewindReq { message_index: 0 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // Failed rewinds leave the session untouched.
        let store = super::SessionStore::new(state.history_file.as_path());
        assert_eq!(store.read_all_messages("s1").expect("read back").len(), 2);
    }

    #[tokio::test]
    async fn rewind_rejects_runtime_injected_user_anchor() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        use crate::ai::history::{append_history_messages, runtime_synthetic_user_message};
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user"]);
        let store = super::SessionStore::new(state.history_file.as_path());
        append_history_messages(
            &store.session_history_file("s1"),
            &[runtime_synthetic_user_message(serde_json::Value::String(
                "handoff".to_string(),
            ))],
        )
        .expect("seed synthetic");
        // The injected handoff is a user row but not a real turn boundary.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 1 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // The real user input next to it still rewinds.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 0 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(store.read_all_messages("s1").expect("read back").is_empty());
    }

    #[tokio::test]
    async fn serve_app_returns_mobile_client() {
        let resp = super::serve_app().await;
        assert!(resp
            .0
            .iter()
            .any(|(k, v)| *k == axum::http::header::CACHE_CONTROL && *v == "no-store"));
        let body = resp.1.0;
        assert!(body.contains("id=\"serve-app\""));
        assert!(body.contains("/sessions/"));
        assert!(body.contains("turns/stream"));
        assert!(body.contains("/rewind"));
        assert!(
            body.contains("/file?path="),
            "the page must fetch server-side images through the authed preview route"
        );
    }

    #[tokio::test]
    async fn history_total_header_reports_precut_count() {
        use axum::{
            extract::{Path, Query, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant", "user"]);
        let resp = super::read_history(
            State(state),
            HeaderMap::new(),
            Path("s1".to_string()),
            Query(super::HistoryQuery { limit: Some(1) }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        // The mobile client maps a tapped bubble to its canonical index as
        // `total - shown.length + bubble_index`, so the header must describe
        // the full session even when the body is a tail cut.
        assert_eq!(
            resp.headers()
                .get("x-history-total")
                .and_then(|v| v.to_str().ok()),
            Some("3")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: Vec<serde_json::Value> = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.len(), 1);
    }

    #[tokio::test]
    async fn server_info_reports_runtime_labels() {
        use axum::{
            extract::State,
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::server_info(State(lifecycle_test_state()), HeaderMap::new())
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let info: serde_json::Value = serde_json::from_slice(&body).expect("json");
        for key in ["model", "model_label", "agent", "reasoning_effort", "version"] {
            assert!(
                info.get(key).and_then(|v| v.as_str()).is_some(),
                "missing {key}"
            );
        }
        for key in ["models", "agents", "efforts"] {
            assert!(
                info.get(key).and_then(|v| v.as_array()).is_some(),
                "missing {key}"
            );
        }
        assert_eq!(
            info["efforts"],
            serde_json::json!(["minimal", "low", "medium", "high", "xhigh", "max", "off"])
        );
        assert!(
            info["models"].as_array().is_some_and(|options| !options.is_empty()
                && options.iter().all(|o| o.get("id").and_then(|v| v.as_str()).is_some()
                    && o.get("label").and_then(|v| v.as_str()).is_some())),
            "model options need id + label"
        );
        assert_eq!(info["agent"], "build");
        assert!(!info["version"].as_str().unwrap_or_default().is_empty());
    }

    fn turn_req(
        model: Option<&str>,
        agent: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> TurnReq {
        TurnReq {
            prompt: "hi".to_string(),
            model: model.map(str::to_string),
            agent: agent.map(str::to_string),
            reasoning_effort: reasoning_effort.map(str::to_string),
            images: Vec::new(),
            confirm: None,
        }
    }

    #[test]
    fn turn_overrides_default_to_absent() {
        let overrides = turn_overrides(&turn_req(None, None, None)).expect("valid");
        assert_eq!(overrides.model, None);
        assert_eq!(overrides.agent, None);
        assert_eq!(overrides.reasoning_effort, None);
        // Empty/blank strings also mean "server default", never a flag.
        let overrides =
            turn_overrides(&turn_req(Some("  "), Some(""), Some(""))).expect("valid");
        assert_eq!(overrides.model, None);
        assert_eq!(overrides.agent, None);
        assert_eq!(overrides.reasoning_effort, None);
    }

    #[test]
    fn turn_overrides_normalize_and_reject() {
        let overrides =
            turn_overrides(&turn_req(Some(" foo "), Some("build"), Some("LOW"))).expect("valid");
        assert_eq!(overrides.model.as_deref(), Some("foo"));
        assert_eq!(overrides.agent.as_deref(), Some("build"));
        assert_eq!(overrides.reasoning_effort.as_deref(), Some("low"));
        assert!(turn_overrides(&turn_req(Some("--evil"), None, None)).is_err());
        assert!(turn_overrides(&turn_req(None, Some("-x"), None)).is_err());
        assert!(turn_overrides(&turn_req(None, None, Some("ultra"))).is_err());
        let long = "m".repeat(super::MAX_TURN_OVERRIDE_CHARS + 1);
        assert!(turn_overrides(&turn_req(Some(&long), None, None)).is_err());
    }

    fn upload(name: &str, bytes: &[u8]) -> ServeImageUpload {
        use base64::Engine as _;
        ServeImageUpload {
            filename: name.to_string(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    }

    #[test]
    fn stage_turn_images_stages_multiple_files() {
        let dir = std::env::temp_dir().join(format!("a-serve-stage-{}", Uuid::new_v4()));
        let images = vec![upload("paste-a.png", b"aaa"), upload("paste-b.jpg", b"bb")];
        stage_turn_images(&dir, &images).expect("stage");
        assert_eq!(std::fs::read(dir.join("paste-a.png")).unwrap(), b"aaa");
        assert_eq!(std::fs::read(dir.join("paste-b.jpg")).unwrap(), b"bb");
        // An empty upload list is a no-op and creates nothing.
        let untouched = dir.join("untouched");
        stage_turn_images(&untouched, &[]).expect("empty ok");
        assert!(!untouched.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_sse_line_strips_the_overwrite_prefix() {
        // Completed/failed tool rows carry `\r\x1b[2K` to redraw the running
        // row on a live terminal; on the append-only SSE wire the prefix must
        // go, while the row indent and content stay byte-identical.
        assert_eq!(
            clean_sse_line("\r\x1b[2K  \u{1b}[32m✓\u{1b}[0m task_integrate"),
            "  \u{1b}[32m✓\u{1b}[0m task_integrate"
        );
        // A bare `\r` alone is still the overwrite control, not content.
        assert_eq!(clean_sse_line("\r[header]"), "[header]");
        // Ordinary rows (indented or not) pass through untouched.
        assert_eq!(clean_sse_line("  ● task_integrate"), "  ● task_integrate");
        assert_eq!(clean_sse_line("↳ speed · x"), "↳ speed · x");
        assert_eq!(clean_sse_line(""), "");
    }

    #[test]
    fn clean_sse_line_neutralizes_cursor_addressing_escapes() {
        // A glued running/completed pair keeps its same-row overwrite (the
        // client terminal resolves it like the local one) but loses the
        // mid-line erase.
        assert_eq!(
            clean_sse_line("  ● task_status\r\u{1b}[2K  \u{1b}[32m✓\u{1b}[0m task_status"),
            "  ● task_status\r  \u{1b}[32m✓\u{1b}[0m task_status"
        );
        // Cursor moves and region erases address the child's screen, never
        // the client's: strip them, keep the text and its SGR color.
        assert_eq!(
            clean_sse_line("\u{1b}[1A\r\u{1b}[2K  \u{1b}[32m✓\u{1b}[0m task_status"),
            "  \u{1b}[32m✓\u{1b}[0m task_status"
        );
        assert_eq!(
            clean_sse_line("↳ cache · 1k\u{1b}[0J"),
            "↳ cache · 1k"
        );
        // Unterminated introducers and non-CSI escapes never reach the wire.
        assert_eq!(clean_sse_line("ab\u{1b}[2"), "ab");
        assert_eq!(clean_sse_line("ab\u{1b}7cd"), "abcd");
    }

    #[test]
    fn stage_turn_images_rejects_unsafe_uploads() {
        let dir = std::env::temp_dir().join(format!("a-serve-stage-{}", Uuid::new_v4()));
        // Path traversal, nested names, non-image extensions and empty names.
        // Double extensions resolve by the final suffix; case is folded.
        for bad in [
            "../evil.png",
            "sub/dir.png",
            "note.txt",
            "x.png.exe",
            "",
            ".",
            "..",
        ] {
            let err =
                stage_turn_images(&dir, &[upload(bad, b"x")]).expect_err("must reject");
            assert!(!err.is_empty(), "empty error for {bad:?}");
        }
        // Uppercase image extensions are accepted (folders stay lowercase-safe).
        let upper = std::env::temp_dir().join(format!("a-serve-stage-{}", Uuid::new_v4()));
        stage_turn_images(&upper, &[upload("PHOTO.PNG", b"x")]).expect("uppercase ext ok");
        assert!(upper.join("PHOTO.PNG").is_file());
        let _ = std::fs::remove_dir_all(&upper);
        // Malformed base64.
        let bad_payload = ServeImageUpload {
            filename: "a.png".to_string(),
            data_base64: "!!!".to_string(),
        };
        stage_turn_images(&dir, std::slice::from_ref(&bad_payload))
            .expect_err("bad base64 must fail");
        // Image count cap (size caps share the same validated path).
        let many: Vec<_> = (0..MAX_TURN_IMAGES + 1)
            .map(|i| upload(&format!("f{i}.png"), b"x"))
            .collect();
        stage_turn_images(&dir, &many).expect_err("too many must fail");
        assert!(
            !dir.join("f0.png").exists(),
            "count rejection must stage nothing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_turn_images_is_atomic_and_enforces_size_caps() {
        use super::MAX_TURN_IMAGE_BYTES;

        // A late rejection must not leave the earlier valid file behind, nor
        // even create the assets dir.
        let dir = std::env::temp_dir().join(format!("a-serve-stage-{}", Uuid::new_v4()));
        let mixed = vec![upload("good.png", b"good"), upload("../evil.png", b"evil")];
        stage_turn_images(&dir, &mixed).expect_err("late entry must fail the turn");
        assert!(
            !dir.exists(),
            "validation must run before any filesystem write"
        );
        // Per-file cap: one byte over the limit is rejected.
        let big = vec![0u8; MAX_TURN_IMAGE_BYTES + 1];
        let dir2 = std::env::temp_dir().join(format!("a-serve-stage-{}", Uuid::new_v4()));
        stage_turn_images(&dir2, &[upload("big.png", &big)])
            .expect_err("oversized file must fail");
        assert!(!dir2.exists());
    }

    #[cfg(test)]
    fn test_frame(
        kind: crate::ai::background::ServeLiveKind,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut v = vec![kind as u8];
        v.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn frame_decoder_reassembles_split_frames() {
        use crate::ai::background::ServeLiveKind;
        // One delta (multibyte "é" inside) followed by a full thinking
        // lifecycle and the output-complete marker.
        let wire = [
            test_frame(ServeLiveKind::Delta, "héllo".as_bytes()),
            test_frame(ServeLiveKind::ThinkingStart, b""),
            test_frame(ServeLiveKind::Thinking, "带来".as_bytes()),
            test_frame(ServeLiveKind::ThinkingDone, b""),
            test_frame(ServeLiveKind::OutputComplete, b""),
        ]
        .concat();
        let mut decoder = ServeFrameDecoder::default();
        // Split inside the first header: nothing complete yet.
        assert_eq!(decoder.push(&wire[..2]), vec![]);
        // Header complete (5 + "h" + first byte of "é" = 9 bytes fed): the
        // delta frame still waits for its tail.
        assert_eq!(decoder.push(&wire[2..9]), vec![]);
        // The rest completes the delta plus all four trailing frames.
        assert_eq!(
            decoder.push(&wire[9..]),
            vec![
                ServeLiveEvent::Delta("héllo".to_string()),
                ServeLiveEvent::ThinkingStart,
                ServeLiveEvent::Thinking("带来".to_string()),
                ServeLiveEvent::ThinkingDone,
                ServeLiveEvent::OutputComplete,
            ]
        );
        assert_eq!(decoder.finish(), vec![]);
        // A truncated tail frame is dropped on EOF, never half-emitted.
        let mut cut = ServeFrameDecoder::default();
        assert_eq!(cut.push(&wire[..10]), vec![]);
        assert_eq!(cut.finish(), vec![]);
    }

    /// Offline pump wiring check with a fake writer (no model key needed):
    /// framed bytes written into the FIFO must surface as SSE events, an
    /// output-complete frame must end the turn while the child is still
    /// "alive", and the pump must stop once the child is reaped and the
    /// pipe is drained. `axum::Event` is opaque, so this asserts event flow
    /// and clean exit, not payload text (covered by the decoder test above).
    #[cfg(unix)]
    #[test]
    fn live_fifo_pump_forwards_frames_and_ends_turn_early() {
        use crate::ai::background::ServeLiveKind;
        let mut fifo = setup_live_fifo("test-pump").expect("fifo setup");
        let reader = fifo.reader.take().expect("reader");
        let (tx, mut rx) = tokio::sync::mpsc::channel(128);
        let child_done = Arc::new(AtomicBool::new(false));
        let turn_done = Arc::new(AtomicBool::new(false));
        let pump_done = Arc::clone(&child_done);
        let pump_turn = Arc::clone(&turn_done);
        let handle =
            std::thread::spawn(move || {
                pump_live_fifo(
                    reader,
                    &pump_done,
                    &pump_turn,
                    &tx,
                    super::TurnSendTiming::default(),
                    std::time::Instant::now(),
                    None,
                )
            });
        // Fake turn child: a delta frame split mid-header and inside a
        // multibyte character, then an output-complete frame; the writer
        // closes (EOF) while the child is still "alive".
        let path = fifo.path.clone();
        std::thread::spawn(move || {
            use std::io::Write;
            let mut writer = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("writer open");
            let delta = test_frame(ServeLiveKind::Delta, "héllo".as_bytes());
            writer.write_all(&delta[..2]).expect("split head");
            writer.write_all(&delta[2..]).expect("split tail");
            let done = test_frame(ServeLiveKind::OutputComplete, b"");
            writer.write_all(&done).expect("output complete");
        })
        .join()
        .expect("writer thread");
        // Both the delta and the early `done` must arrive while the child is
        // still running (child_done stays false throughout this wait).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut events = 0usize;
        while events < 2 && std::time::Instant::now() < deadline {
            while rx.try_recv().is_ok() {
                events += 1;
            }
            if events < 2 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        assert!(
            events >= 2,
            "pump must forward the delta and the early done while the child runs"
        );
        assert!(
            turn_done.load(Ordering::SeqCst),
            "output-complete frame must end the turn before child reap"
        );
        // Child reaped with a drained pipe: the pump must exit on its own.
        // (Plain `join`: a hang here is itself the failure signal.)
        child_done.store(true, Ordering::SeqCst);
        handle.join().expect("pump thread panicked");
        // The per-turn inode is still owned by `fifo` here and unlinked on drop.
        assert!(
            fifo.path.exists(),
            "fifo inode must outlive the pump for writer-drain ordering"
        );
    }

    /// Stopping a turn that is parked on a question: the interrupt path drops
    /// the answer pipe, which unblocks the child's stdin read (EOF) and clears
    /// the question, so a client's dialog disappears instead of waiting on an
    /// answer nobody can give any more.
    #[cfg(unix)]
    #[tokio::test]
    async fn interrupt_dismisses_the_pending_question_and_unblocks_the_child() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        // Stand-in turn child: it echoes what its first stdin line turned out
        // to be. Closing the pipe makes `read` return EOF, so an empty echo is
        // the proof that the blocked read did come back.
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("read line; echo \"read[$line]\"")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn stand-in child");
        let echo = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut text = String::new();
                let _ = out.read_to_string(&mut text);
                text
            })
        });
        let state = lifecycle_test_state();
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 1,
                    prompt: "commit?".to_string(),
                    token: 5,
                }),
                answers: child.stdin.take(),
            })),
        );
        // A registered turn child is what `running` reports. The entry is
        // removed before the interrupt below: this stand-in must not be
        // signalled, and the stop path is what the test is about, not the
        // signal.
        state
            .active_turns
            .lock()
            .expect("active turns")
            .insert("sid".to_string(), std::process::id());
        let busy =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        let body = axum::body::to_bytes(busy.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["running"], true);
        assert_eq!(view["pending"]["id"], 1);
        state.active_turns.lock().expect("active turns").remove("sid");

        let stopped = super::post_interrupt(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
        )
        .await
        .into_response();
        assert_eq!(stopped.status(), StatusCode::OK);
        assert_eq!(
            echo.and_then(|handle| handle.join().ok()).as_deref(),
            Some("read[]\n")
        );

        let after =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        let body = axum::body::to_bytes(after.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["pending"], serde_json::Value::Null);
        assert_eq!(view["running"], false);
        // The dismissed question can no longer be answered: the dialog a
        // client rebuilds from a stale view is refused, not silently accepted.
        let late = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(super::ConfirmAnswerReq {
                id: 1,
                token: 5,
                allow: true,
            }),
        )
        .await
        .into_response();
        assert_eq!(late.status(), StatusCode::CONFLICT);
    }

    /// The interrupt endpoint's core: the registered pid is what receives
    /// SIGINT, which is the local first-Ctrl+C semantics (cancel the streaming
    /// turn), not a hard kill, and only the addressed session is signalled.
    #[cfg(unix)]
    #[test]
    fn interrupt_active_turn_signals_the_registered_child() {
        use std::os::unix::process::ExitStatusExt;
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let active: std::sync::Mutex<std::collections::HashMap<String, u32>> = Default::default();
        active
            .lock()
            .expect("lock registry")
            .insert("s1".to_string(), child.id());
        assert!(
            super::interrupt_active_turn(&active, "s1"),
            "a registered child must be signalled"
        );
        let status = child.wait().expect("reap sleep");
        assert_eq!(status.signal(), Some(libc::SIGINT));
        assert!(
            !super::interrupt_active_turn(&active, "other"),
            "sessions without a registered child must report no interrupt"
        );
    }

    #[test]
    fn active_turn_guard_clears_its_registry_entry() {
        let active = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        {
            let _guard = super::ActiveTurnGuard::register(&active, "s1", 4242);
            assert_eq!(active.lock().expect("lock").get("s1").copied(), Some(4242));
        }
        assert!(
            active.lock().expect("lock").is_empty(),
            "a reaped child must not stay interruptible"
        );
    }

    /// Route coverage: an idle session answers `{"interrupted": false}` (an
    /// interrupt that raced the end of the stream), a malformed id is rejected
    /// before any signal, and a token-protected server rejects missing auth.
    #[tokio::test]
    async fn interrupt_route_reports_idle_sessions_and_guards_access() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::post_interrupt(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("s1".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["interrupted"], serde_json::Value::Bool(false));

        let mut authed = lifecycle_test_state();
        authed.token = "t".to_string();
        let resp = super::post_interrupt(State(authed), HeaderMap::new(), Path("s1".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = super::post_interrupt(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("bad id".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
