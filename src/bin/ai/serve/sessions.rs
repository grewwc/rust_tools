//! Session lifecycle routes: list/create/fork/delete, title, mark, config.

use super::*;
use crate::ai::driver::commands::use_backend::{
    agent_team_sticky, apply_subagent_preference, apply_subagent_use, effective_subagent_backend,
    format_use_state, reset_subagent_config,
};
use crate::ai::history::{MarkMessageUpdate, with_sessions_lifecycle_lock};
use crate::ai::tools::task_tools::BackendId;

/// Render the config response. `subagent_backend` carries an explicitly stored
/// canonical choice verbatim. Without one, an agent-team session reports the
/// TraeCLI default its picker uses (and that the runtime applies), while an
/// ordinary session reports empty so each client keeps its own initial picker
/// selection (the mobile sheet defaults to TraeCLI) and the runtime stays on
/// the kernel backend.
fn session_config_response(config: &SessionServeConfig) -> serde_json::Value {
    let kernel_model = config.model.clone().unwrap_or_else(default_turn_model);
    // A stored canonical choice is reported as-is. Without one, an agent-team
    // session reports the TraeCLI default its mobile picker shows; an ordinary
    // session reports empty so each client keeps its own initial picker
    // selection while the runtime uses the kernel default.
    let subagent_backend = match BackendId::canonical_stored(config.subagent_backend.as_deref()) {
        Some(backend) => backend.as_str(),
        None if agent_team_sticky(config) => effective_subagent_backend(config).as_str(),
        None => "",
    };
    serde_json::json!({
        "model": config.model.as_deref().unwrap_or_default(),
        "agent": config.agent.as_deref().unwrap_or_default(),
        "reasoning_effort": config.reasoning_effort.as_deref().unwrap_or_default(),
        "subagent_backend": subagent_backend,
        "subagent_model": config.subagent_model.as_deref().unwrap_or_default(),
        "subagent_kernel_model": config.subagent_kernel_model.as_deref().unwrap_or_default(),
        "default_skills": config.default_skills,
        "output": format_use_state(config, &kernel_model),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_reports_stored_choice_and_mode_aware_defaults() {
        for (stored, expected_wire) in [
            (None, ""),
            (Some(""), ""),
            (Some("kernel"), "kernel"),
            (Some("traecli"), "traecli"),
            (Some("unknown-backend"), ""),
            (Some("trae-cli"), ""),
        ] {
            let config = SessionServeConfig {
                subagent_backend: stored.map(str::to_string),
                ..Default::default()
            };
            let runtime_backend = BackendId::from_stored(stored);
            let response = session_config_response(&config);
            // An ordinary session reports no stored choice, so the mobile
            // picker keeps its own TraeCLI default while the runtime stays on
            // the kernel backend.
            assert_eq!(response["subagent_backend"], expected_wire);
            assert!(response["output"].as_str().unwrap().starts_with(&format!(
                "Session default subagent backend: {}\n",
                runtime_backend.as_str()
            )));
        }

        // An agent-team session without an explicit pick keeps the TraeCLI
        // default the mobile picker shows; resetting clears the team and the
        // session returns to the kernel default.
        let mut team = SessionServeConfig {
            default_skills: vec!["AGENT-TEAM".to_string()],
            ..Default::default()
        };
        let response = session_config_response(&team);
        assert_eq!(response["subagent_backend"], "traecli");
        assert!(
            response["output"]
                .as_str()
                .unwrap()
                .starts_with("Session default subagent backend: traecli\n")
        );
        reset_subagent_config(&mut team);
        let response = session_config_response(&team);
        assert_eq!(response["subagent_backend"], "");
        assert!(
            response["output"]
                .as_str()
                .unwrap()
                .starts_with("Session default subagent backend: kernel\n")
        );
    }
}

pub(crate) async fn list_sessions(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Query(q): Query<SessionsQuery>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let items = match store.list_sessions() {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };
    let mut all: Vec<SessionItem> = items
        .into_iter()
        .map(|s| SessionItem {
            id: s.id,
            size_bytes: s.size_bytes,
            summary: s.summary,
            first_user_prompt: s.first_user_prompt,
            marked: s.marked,
            archived: s.archived,
        })
        .collect();
    let total = all.len();
    if q.newest {
        all.reverse();
    }
    let included = q
        .include
        .as_deref()
        .and_then(|id| all.iter().find(|item| item.id == id).cloned());
    // Preserve newest-first store order while allowing clients to fetch
    // consecutive pages. Offset is applied before limit, and an explicitly
    // selected older session is appended without consuming that page.
    let offset = q.offset.unwrap_or(0).min(all.len());
    all.drain(..offset);
    if let Some(limit) = q.limit {
        all.truncate(limit);
    }
    let page_len = all.len();
    if offset == 0
        && let Some(item) = included
        && !all.iter().any(|shown| shown.id == item.id)
    {
        all.push(item);
    }
    let mut response = (StatusCode::OK, Json(serde_json::json!(all))).into_response();
    response.headers_mut().insert(
        "x-sessions-total",
        axum::http::HeaderValue::from_str(&total.to_string()).unwrap(),
    );
    response.headers_mut().insert(
        "x-sessions-page-size",
        axum::http::HeaderValue::from_str(&page_len.to_string()).unwrap(),
    );
    response
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
    if let Err(err) = crate::ai::driver::commands::session::apply_fork_title(&store, &id, &dst) {
        eprintln!("[serve] fork title for session {dst}: {err}");
    }
    // The branch keeps working where its parent was; otherwise the fork would
    // silently revert to the daemon cwd on its first turn.
    copy_session_cwd(&state, &id, &dst);
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
        Ok(deleted) => {
            // The session's virtual cwd dies with it; a later recreated
            // session of the same id starts from the daemon cwd again.
            drop_session_cwd(&state, &id);
            (
                StatusCode::OK,
                Json(serde_json::json!({"deleted": deleted})),
            )
                .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Set an existing session's important mark. Marking preserves its optional
/// mark message; unmarking clears it, mirroring local `/mark` and `/unmark`.
pub(crate) async fn set_session_mark(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SetSessionMarkReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let result = with_sessions_lifecycle_lock(store.sessions_root(), || {
        // The writer creates missing databases. Serialize this existence
        // check with deletion so marking cannot recreate a deleted session.
        if !store.session_exists(&id)? {
            return Ok(None);
        }
        let message = if req.marked {
            MarkMessageUpdate::Keep
        } else {
            MarkMessageUpdate::Clear
        };
        store.write_session_mark(&id, req.marked, message)?;
        store.read_session_marked(&id).map(Some)
    });
    match result {
        Ok(Some(marked)) => {
            (StatusCode::OK, Json(serde_json::json!({"marked": marked}))).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("session not found: {id}")})),
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
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"title": title}))).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

/// Read the session's persisted serve-mode config (`GET /sessions/{id}/config`):
/// the model/agent/reasoning-effort picks and subagent defaults for this session.
/// An empty field means "server default"; a session that does not exist yet
/// reads as all-defaults. `output` mirrors the local `/use` state text.
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
        Ok(config) => (StatusCode::OK, Json(session_config_response(&config))).into_response(),
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
/// is stored. Subagent defaults use the local `/use` and `/reset` rules. All
/// keys are written in one transaction, so a crash cannot leave a half-updated
/// session config. Writing to a session that does not
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
    // A turn child persists its runtime snapshot before exiting. Hold the same
    // session lock across this read and write so that snapshot cannot undo a patch.
    let lock = session_lock(&state, &id).await;
    let _guard = lock.lock().await;
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
    if let Some(subagents) = req.subagents {
        match subagents.action.as_str() {
            "use" => {
                let Some(backend) = subagents.backend.as_deref() else {
                    return bad_request("backend is required for subagents use".to_string())
                        .into_response();
                };
                if let Err(err) =
                    apply_subagent_use(&mut config, backend, subagents.model.as_deref())
                {
                    return bad_request(err).into_response();
                }
            }
            // `/subagent` shares the config channel but must not touch sticky
            // skills: it writes the same backend/model keys without the
            // agent-team rule, so the runtime mode stays as it was.
            "subagent" => {
                let Some(backend) = subagents.backend.as_deref() else {
                    return bad_request("backend is required for subagents subagent".to_string())
                        .into_response();
                };
                if let Err(err) =
                    apply_subagent_preference(&mut config, backend, subagents.model.as_deref())
                {
                    return bad_request(err).into_response();
                }
            }
            "reset" => reset_subagent_config(&mut config),
            _ => {
                return bad_request("subagents action must be use|subagent|reset".to_string())
                    .into_response();
            }
        }
    }
    match store.write_session_serve_config(&id, &config) {
        Ok(()) => (StatusCode::OK, Json(session_config_response(&config))).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}
