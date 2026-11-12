//! Session export/import routes (`GET /sessions/{id}/export`,
//! `GET /sessions/{id}/export-md`, `POST /sessions/import`).
//!
//! These mirror the local `/export`, `/export-md` and `/import` commands. A
//! browser cannot read the daemon's filesystem, so the archive bytes travel
//! over HTTP: exports are attachments fetched with the client's bearer token,
//! and an import uploads one archive that the daemon extracts into a new
//! session.

use super::*;
use base64::Engine as _;

/// One import upload is capped at 64 MiB of archive bytes. Base64 inflates the
/// JSON body by 4/3, so the route's body limit leaves headroom above this.
const MAX_IMPORT_BYTES: usize = 64 * 1024 * 1024;
/// Body limit for `POST /sessions/import`, including the base64 envelope.
pub(crate) const MAX_IMPORT_REQUEST_BYTES: usize = MAX_IMPORT_BYTES / 3 * 4 + 1024 * 1024;

/// `POST /sessions/import` request: one archive as base64 plus an optional
/// destination id. Without `id` the daemon allocates a fresh session id.
#[derive(Debug, Deserialize)]
pub(crate) struct ImportSessionReq {
    #[serde(default)]
    pub(crate) filename: String,
    pub(crate) data: String,
    #[serde(default)]
    pub(crate) id: String,
}

/// Short, ASCII-only attachment name derived from a validated session id.
fn attachment_name(session_id: &str, extension: &str) -> String {
    let short: String = session_id.chars().take(8).collect();
    format!("session-{short}.{extension}")
}

/// Build an attachment response. `filename` is server-controlled and ASCII.
fn attachment_response(bytes: Vec<u8>, content_type: &'static str, filename: &str) -> Response {
    let mut resp = Response::new(axum::body::Body::from(bytes));
    let h = resp.headers_mut();
    h.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    h.insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    // Session data sits behind the bearer token; no shared cache may keep it.
    h.insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    if let Ok(value) = HeaderValue::from_str(&format!("attachment; filename=\"{filename}\"")) {
        h.insert(axum::http::header::CONTENT_DISPOSITION, value);
    }
    resp
}

fn io_error_response(err: std::io::Error) -> Response {
    let status = match err.kind() {
        std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
        std::io::ErrorKind::AlreadyExists => StatusCode::CONFLICT,
        std::io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(serde_json::json!({"error": err.to_string()}))).into_response()
}

/// `GET /sessions/{id}/export` (authed): the session as a migration archive,
/// byte-compatible with the local `/export` zip.
pub(crate) async fn export_session_zip(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let history_file = state.history_file.clone();
    let session_id = id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let store = SessionStore::new(&history_file);
        let tmp = std::env::temp_dir().join(format!("a-serve-export-{}.zip", uuid::Uuid::new_v4()));
        let outcome = (|| {
            store.export_session_archive(&session_id, &tmp)?;
            std::fs::read(&tmp)
        })();
        let _ = std::fs::remove_file(&tmp);
        outcome
    })
    .await;
    match result {
        Ok(Ok(bytes)) => {
            attachment_response(bytes, "application/zip", &attachment_name(&id, "zip"))
        }
        Ok(Err(err)) => io_error_response(err),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Last assistant conclusion, matching the local `/export-md` selection: the
/// newest assistant message with text and no tool calls.
fn last_assistant_conclusion(messages: &[crate::ai::history::Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        if message.role != "assistant" {
            return None;
        }
        if message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            return None;
        }
        let text = match &message.content {
            serde_json::Value::String(text) => text.clone(),
            other => {
                let mut redacted = other.clone();
                redact_inline_image_data(&mut redacted);
                serde_json::to_string(&redacted).ok()?
            }
        };
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    })
}

/// `GET /sessions/{id}/export-md` (authed): the last model conclusion as a
/// Markdown attachment. Mirrors `/export-md`, which exports the final answer
/// rather than the whole transcript.
pub(crate) async fn export_session_markdown(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let history_file = state.history_file.clone();
    let session_id = id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let store = SessionStore::new(&history_file);
        let messages = store.read_all_messages(&session_id)?;
        Ok::<Option<String>, std::io::Error>(last_assistant_conclusion(&messages))
    })
    .await;
    match result {
        Ok(Ok(Some(text))) => attachment_response(
            text.into_bytes(),
            "text/markdown; charset=utf-8",
            &attachment_name(&id, "md"),
        ),
        Ok(Ok(None)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "no model conclusion to export"})),
        )
            .into_response(),
        Ok(Err(err)) => io_error_response(err),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// `POST /sessions/import` (authed): extract one uploaded archive into a new
/// session and return its id. The archive layout is the local `/export` zip.
pub(crate) async fn import_session(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Json(req): Json<ImportSessionReq>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if req.filename.chars().count() > 240 {
        return bad_request("oversized archive filename".to_string()).into_response();
    }
    let bytes = match base64::engine::general_purpose::STANDARD.decode(req.data.as_bytes()) {
        Ok(bytes) => bytes,
        Err(_) => {
            return bad_request("archive data is not valid base64".to_string()).into_response();
        }
    };
    if bytes.is_empty() || bytes.len() > MAX_IMPORT_BYTES {
        return bad_request("archive is empty or larger than the import cap".to_string())
            .into_response();
    }
    let dst_id = if req.id.trim().is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        let id = req.id.trim().to_string();
        if let Err(err) = SessionStore::validate_session_id(&id) {
            return bad_request(err.to_string()).into_response();
        }
        id
    };
    let lock = session_lock(&state, &dst_id).await;
    let session_guard = lock.lock_owned().await;
    let history_file = state.history_file.clone();
    let import_id = dst_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Keep the turn lock until import finishes, even if the HTTP request is cancelled.
        let _session_guard = session_guard;
        let store = SessionStore::new(&history_file);
        let tmp = std::env::temp_dir().join(format!("a-serve-import-{}.zip", uuid::Uuid::new_v4()));
        let outcome = (|| {
            std::fs::write(&tmp, &bytes)?;
            store.import_session_archive(&tmp, &import_id)
        })();
        let _ = std::fs::remove_file(&tmp);
        outcome
    })
    .await;
    match result {
        Ok(Ok(id)) => (StatusCode::OK, Json(serde_json::json!({ "id": id }))).into_response(),
        Ok(Err(err)) => io_error_response(err),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn import_session_waits_for_turn_lock_before_publishing() {
        let root = std::env::temp_dir().join(format!("serve-import-lock-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let history_file = root.join("history.sqlite");
        let store = SessionStore::new(&history_file);
        let source_db = store.session_history_file("source");
        crate::ai::history::append_history_messages(
            &source_db,
            &[crate::ai::history::Message {
                role: "user".into(),
                content: serde_json::json!("import me"),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        store.add_session_write_root("source", &root).unwrap();
        let archive = root.join("archive.zip");
        store.export_session_archive("source", &archive).unwrap();
        let state = ServeState {
            history_file,
            token: String::new(),
            default_add_dirs: Default::default(),
            cwds: Default::default(),
            locks: Default::default(),
            open_sessions: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
            events: Default::default(),
        };
        let lock = session_lock(&state, "imported").await;
        let guard = lock.lock().await;
        let req = ImportSessionReq {
            filename: "archive.zip".into(),
            data: base64::engine::general_purpose::STANDARD.encode(std::fs::read(archive).unwrap()),
            id: "imported".into(),
        };
        let mut import = tokio::spawn(import_session(State(state), HeaderMap::new(), Json(req)));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut import)
                .await
                .is_err()
        );
        assert!(!store.session_history_file("imported").exists());
        drop(guard);
        assert_eq!(import.await.unwrap().status(), StatusCode::OK);
        assert!(
            store
                .read_session_serve_config("imported")
                .unwrap()
                .additional_write_roots
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn attachment_names_stay_ascii_and_short() {
        assert_eq!(
            attachment_name("0d1866d8-1111-2222-3333-444455556666", "zip"),
            "session-0d1866d8.zip"
        );
        assert_eq!(attachment_name("abc", "md"), "session-abc.md");
    }

    #[test]
    fn conclusion_skips_tool_calls_and_empty_text() {
        let text = |value: &str| crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(value.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        let messages = vec![
            text("first"),
            crate::ai::history::Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String("tool round".to_string()),
                tool_calls: Some(vec![crate::ai::types::ToolCall {
                    id: "call".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "read".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            text("   "),
            text("final"),
        ];
        assert_eq!(
            last_assistant_conclusion(&messages).as_deref(),
            Some("final")
        );
    }
}
