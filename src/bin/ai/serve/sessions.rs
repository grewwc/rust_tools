//! Session lifecycle routes: list/create/fork/delete, title, mark, config.

use super::*;
use crate::ai::driver::commands::use_backend::{
    agent_team_sticky, apply_subagent_preference, apply_subagent_use, effective_subagent_backend,
    format_use_state, reset_subagent_config,
};
use crate::ai::history::{MarkMessageUpdate, with_sessions_lifecycle_lock};
use crate::ai::tools::task_tools::BackendId;

pub(super) const MAX_SESSION_ACTIVITY_IDS: usize = 256;
pub(super) const MAX_SESSION_ACTIVITY_REQUEST_BYTES: usize = 64 * 1024;

/// Sample whether the session has a live foreground turn or parked work.
///
/// A foreground loop releases its turn lease when it yields to wait for
/// subagent results, but it keeps a work lease until it resumes or exits. Both
/// leases mean the session is still owned by live work; only their absence
/// means idle.
pub(super) fn session_activity(
    state: &ServeState,
    store: &SessionStore,
    id: &str,
) -> SessionActivity {
    use crate::ai::history::turn_lease::{
        TurnLeaseProbe, probe_session_work_lease, probe_turn_lease,
    };

    let local = state
        .active_turns
        .lock()
        .map(|turns| turns.contains_key(id));
    if matches!(local, Ok(true)) {
        return SessionActivity::Running;
    }
    let history = store.session_history_file(id);
    match (
        probe_turn_lease(&history),
        probe_session_work_lease(&history),
    ) {
        (TurnLeaseProbe::Held, _) | (_, TurnLeaseProbe::Held) => SessionActivity::Running,
        (TurnLeaseProbe::Free, TurnLeaseProbe::Free) if local.is_ok() => SessionActivity::Idle,
        _ => SessionActivity::Unknown,
    }
}

/// Cap for the published question text: enough to recognize the gate in a
/// tooltip, short enough to keep a full-page activity poll small.
const MAX_WAITING_PROMPT_CHARS: usize = 160;

/// The question a session's turn child is parked on, sampled from the live
/// confirm slot the answer path writes to. `None` while the turn merely runs:
/// only an unanswered question is a call for the user.
pub(super) fn session_waiting(state: &ServeState, id: &str) -> Option<(SessionWaiting, String)> {
    let slot = state.confirms.lock().ok()?.get(id).cloned()?;
    let pending = slot.lock().ok()?.pending.clone()?;
    let kind = match pending.kind {
        ConfirmKind::Confirm => SessionWaiting::Confirm,
        ConfirmKind::Input => SessionWaiting::Input,
    };
    // A gate prompt can be a multi-line block; the sidebar only has room for
    // its first line.
    let prompt = pending
        .prompt
        .lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(MAX_WAITING_PROMPT_CHARS)
        .collect();
    Some((kind, prompt))
}

/// Bulk, uncached turn samples for loaded IDs. This route deliberately does
/// not enumerate SQLite files, read metadata, or compute session totals.
pub(crate) async fn list_session_activity(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Json(req): Json<SessionActivityReq>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if req.ids.len() > MAX_SESSION_ACTIVITY_IDS {
        return bad_request(format!(
            "at most {MAX_SESSION_ACTIVITY_IDS} session ids are allowed"
        ))
        .into_response();
    }
    for id in &req.ids {
        if SessionStore::validate_session_id(id).is_err() {
            return bad_request(format!("invalid session id: {id}")).into_response();
        }
    }
    let store = SessionStore::new(state.history_file.as_path());
    let items: Vec<SessionActivityItem> = req
        .ids
        .into_iter()
        .map(|id| {
            let waiting = session_waiting(&state, &id);
            SessionActivityItem {
                activity: session_activity(&state, &store, &id),
                id,
                waiting: waiting.as_ref().map(|(kind, _)| *kind),
                waiting_prompt: waiting.map(|(_, prompt)| prompt),
            }
        })
        .collect();
    let mut response = Json(items).into_response();
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-store"));
    response
}

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
    let query = q.query.as_deref().unwrap_or_default().trim().to_lowercase();
    let keywords: Vec<&str> = query.split_whitespace().collect();
    // Snapshot organization once, then release its lock before reading cwd
    // or live state. All returned rows use the same grouping snapshot.
    let groups = groups_snapshot(&state);
    let mut all: Vec<SessionItem> = items
        .into_iter()
        .filter(|s| match q.archived.as_deref() {
            Some("exclude") => !s.archived,
            Some("only") => s.archived,
            _ => true,
        })
        .filter(|s| groups.matches_group(&s.id, q.group.as_deref()))
        .filter(|s| !q.marked || s.marked)
        .filter(|s| {
            if keywords.is_empty() {
                return true;
            }
            // Keep the same title precedence and placeholder as the WebUI.
            let title = s
                .summary
                .as_deref()
                .filter(|title| !title.is_empty())
                .or_else(|| {
                    s.first_user_prompt
                        .as_deref()
                        .filter(|prompt| !prompt.is_empty())
                })
                .map(str::to_owned)
                .unwrap_or_else(|| format!("new session · {}", &s.id[..s.id.len().min(8)]));
            let text = format!("{title} {}", s.id).to_lowercase();
            keywords.iter().all(|keyword| text.contains(keyword))
        })
        .map(|s| {
            // Resolve before moving `s.id`: the value is the session's live
            // `/cd` entry, falling back to the daemon cwd exactly like the
            // turn child will.
            let cwd = session_cwd(&state, &s.id).display().to_string();
            let group_id = groups.group_id(&s.id);
            SessionItem {
                id: s.id,
                size_bytes: s.size_bytes,
                summary: s.summary,
                first_user_prompt: s.first_user_prompt,
                marked: s.marked,
                archived: s.archived,
                group_id,
                cwd,
                activity: SessionActivity::Unknown,
                waiting: None,
                waiting_prompt: None,
            }
        })
        .collect();
    // The total describes the complete matching set, before paging or include.
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
    // Probe only the returned page, including an explicitly selected older ID.
    for item in &mut all {
        item.activity = session_activity(&state, &store, &item.id);
        let waiting = session_waiting(&state, &item.id);
        item.waiting = waiting.as_ref().map(|(kind, _)| *kind);
        item.waiting_prompt = waiting.map(|(_, prompt)| prompt);
    }
    let mut response = (StatusCode::OK, Json(serde_json::json!(all))).into_response();
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-store"));
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
    body: axum::body::Bytes,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let req: SetSessionGroupReq = if body.is_empty() {
        SetSessionGroupReq::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(req) => req,
            Err(err) => return bad_request(err.to_string()).into_response(),
        }
    };
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
    if let Some(group_id) = &req.group_id
        && let Err(err) = assign_session_group(&state, &id, Some(group_id))
    {
        return err;
    }
    let cwd = session_cwd(&state, &id).display().to_string();
    (
        StatusCode::OK,
        Json(CreateSessionResp {
            id,
            cwd,
            group_id: req.group_id,
        }),
    )
        .into_response()
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
    let group_id = copy_session_group(&state, &id, &dst);
    let cwd = session_cwd(&state, &dst).display().to_string();
    (
        StatusCode::OK,
        Json(CreateSessionResp {
            id: dst,
            cwd,
            group_id,
        }),
    )
        .into_response()
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
            drop_session_group(&state, &id);
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

#[derive(Deserialize)]
pub(crate) struct SetSessionArchiveReq {
    pub(crate) archived: bool,
}

/// Reuse formal archive metadata without changing titles or important marks.
pub(crate) async fn set_session_archive(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SetSessionArchiveReq>,
) -> Response {
    if let Err(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(&state.history_file);
    let result = with_sessions_lifecycle_lock(store.sessions_root(), || {
        if !store.session_exists(&id)? {
            return Ok(false);
        }
        store.write_session_archived(&id, req.archived)?;
        Ok(true)
    });
    match result {
        Ok(true) => Json(serde_json::json!({"id": id, "archived": req.archived})).into_response(),
        Ok(false) => (
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
