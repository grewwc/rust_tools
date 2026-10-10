//! Session lifecycle routes: list/create/fork/delete, title, config.

use super::*;

pub(crate) async fn list_sessions(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Query(q): Query<SessionsQuery>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let mut items = match store.list_sessions() {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };
    // Newest first is already the store order; apply limit only.
    if let Some(limit) = q.limit {
        items.truncate(limit);
    }
    let out: Vec<SessionItem> = items
        .into_iter()
        .map(|s| SessionItem {
            id: s.id,
            size_bytes: s.size_bytes,
            summary: s.summary,
            first_user_prompt: s.first_user_prompt,
            marked: s.marked,
        })
        .collect();
    (StatusCode::OK, Json(serde_json::json!(out))).into_response()
}

pub(crate) async fn create_session(
    State(state): State<ServeState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    if let Err(err) = store.ensure_root_dir() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response();
    }
    let id = uuid::Uuid::new_v4().to_string();
    // Sessions are created lazily: the id is allocated here, the persisted
    // files appear on the first turn (`--session <id>` one-shot). Until then
    // GET history 404s, matching CLI one-shot semantics.
    debug_assert!(SessionStore::validate_session_id(&id).is_ok());
    (StatusCode::OK, Json(CreateSessionResp { id })).into_response()
}

/// Fork session `{id}` into a new branch, mirroring the local `/fork`:
/// wholesale copy (messages, assets, checkpoints) plus a depth-tagged fork
/// marker title, then the client switches to the new id. Refuses a missing
/// source with 404; a failed title write warns but never fails the fork.
pub(crate) async fn fork_session(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let dst = uuid::Uuid::new_v4().to_string();
    match store.fork_session(&id, &dst) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    }
    if let Err(err) =
        crate::ai::driver::commands::session::apply_fork_title(&store, &id, &dst)
    {
        eprintln!("[serve] fork title for session {dst}: {err}");
    }
    (StatusCode::OK, Json(CreateSessionResp { id: dst })).into_response()
}

/// Delete session `{id}`, mirroring the local `/close` (minus the local
/// suspended-binding cleanup and process exit, which stay client-side).
/// Idempotent: deleting a missing session still succeeds so client retries
/// after a dropped connection stay safe.
pub(crate) async fn delete_session(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    match store.delete_session(&id) {
        Ok(deleted) => (
            StatusCode::OK,
            Json(serde_json::json!({"deleted": deleted})),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Cap for a user-supplied session title (`POST /sessions/{id}/title`). The
/// local `/title` command accepts anything, but the network API caps input so
/// one bad request cannot stuff megabytes into session metadata.
pub(crate) const MAX_SESSION_TITLE_CHARS: usize = 200;

#[derive(Debug, Deserialize)]
pub(crate) struct SetTitleReq {
    #[serde(default)]
    pub(crate) title: String,
}

/// Rename session `{id}`, mirroring the local `/title <text>`: the title is
/// persisted with the `User` origin, so background auto-generation never
/// overwrites it. Like `/title`, renaming a not-yet-materialized (lazy)
/// session creates its store entry.
pub(crate) async fn set_session_title(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SetTitleReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let title = req.title.trim().to_string();
    if title.is_empty() {
        return bad_request("title must not be empty".to_string()).into_response();
    }
    if title.chars().count() > MAX_SESSION_TITLE_CHARS {
        return bad_request(format!("title exceeds {MAX_SESSION_TITLE_CHARS} chars"))
            .into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    match store.write_session_title_with_origin(&id, &title, SessionTitleOrigin::User) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({"title": title})),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Read the session's persisted serve-mode config (`GET /sessions/{id}/config`):
/// the model/agent/reasoning-effort picks the mobile client shows for this
/// session. An empty field means "server default"; a session that does not
/// exist yet reads as all-defaults.
pub(crate) async fn get_session_config(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    match store.read_session_serve_config(&id) {
        Ok(config) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "model": config.model.unwrap_or_default(),
                "agent": config.agent.unwrap_or_default(),
                "reasoning_effort": config.reasoning_effort.unwrap_or_default(),
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Update the session's persisted serve-mode config (`POST /sessions/{id}/config`):
/// a field absent from the body keeps its stored value, a blank field clears
/// it, and a non-blank field is validated like a per-turn override before it
/// is stored. The three keys are written in one transaction, so a crash cannot
/// leave a half-updated session config. Writing to a session that does not
/// exist yet materializes it (same as `POST /sessions/{id}/title`).
pub(crate) async fn set_session_config(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SessionConfigReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let model = match sanitize_session_config_name(req.model, "model") {
        Ok(update) => update,
        Err(e) => return bad_request(e).into_response(),
    };
    let agent = match sanitize_session_config_name(req.agent, "agent") {
        Ok(update) => update,
        Err(e) => return bad_request(e).into_response(),
    };
    let reasoning_effort = match sanitize_session_config_effort(req.reasoning_effort) {
        Ok(update) => update,
        Err(e) => return bad_request(e).into_response(),
    };
    let store = SessionStore::new(state.history_file.as_path());
    let mut config = match store.read_session_serve_config(&id) {
        Ok(config) => config,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };
    if let Some(value) = model {
        config.model = value;
    }
    if let Some(value) = agent {
        config.agent = value;
    }
    if let Some(value) = reasoning_effort {
        config.reasoning_effort = value;
    }
    match store.write_session_serve_config(&id, &config) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "model": config.model.unwrap_or_default(),
                "agent": config.agent.unwrap_or_default(),
                "reasoning_effort": config.reasoning_effort.unwrap_or_default(),
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}
