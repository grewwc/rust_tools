//! History routes: rewind and read.

use super::*;
use crate::ai::history::Message;

/// Server-side cap for the serve-chat `?count` view param. The client clamps
/// to its own max (20) before sending, so this only guards against a
/// malformed or oversized request turning a preview into a full dump.
const SERVE_HISTORY_VIEW_COUNT_CAP: usize = 100;

/// Same searchable-text rule as the serve-chat client (`String` verbatim,
/// anything else as compact JSON), used by the `/history last` conclusion
/// emptiness check so both sides agree on what counts as content.
fn serve_searchable_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        other => {
            serde_json::to_string(other).unwrap_or_else(|_| "<non-string content>".to_string())
        }
    }
}

fn is_serve_assistant_conclusion(message: &Message) -> bool {
    message.role == "assistant"
        && message
            .tool_calls
            .as_ref()
            .is_none_or(|calls| calls.is_empty())
        && !serve_searchable_text(&message.content).trim().is_empty()
}

/// Request body for `POST /sessions/{id}/rewind`: the canonical (0-based)
/// index into the session's full message list of the user message to rewind
/// to. The mobile client learns that index as
/// `X-History-Total - shown.length + bubble_index` (see `read_history`).
#[derive(Debug, Deserialize)]
pub(crate) struct RewindReq {
    pub(crate) message_index: usize,
}

/// Rewind session `{id}` to just before the anchored user message, mirroring
/// the local `/history rewind u<N>`: the anchored user message and everything
/// after it is removed, while titles and other metadata are untouched. The
/// anchor must be a real user turn boundary: assistant/tool messages and
/// runtime-injected user messages are rejected with 400, because rewinding to
/// those would split a turn in the middle. Holds the session lock, so a
/// rewind tapped during an active turn waits for the turn to finish instead
/// of racing it. Like the local rewind, the stale-patch ledger is rebuilt
/// from the surviving messages (keeping pre-rewind entries would let the next
/// patch bypass the fresh-read gate) and the context cache is invalidated.
pub(crate) async fn rewind_history(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<RewindReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let session_file = store.session_history_file(&id);
    // Serialize with turns (see `post_turn_sse`): the anchored index points
    // into the message prefix, which a running turn only appends to, so the
    // truncate below lands on a quiescent file.
    let lock = session_lock(&state, &id).await;
    let _guard = lock.lock().await;
    let messages = match store.read_all_messages(&id) {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };
    let Some(anchor) = messages.get(req.message_index) else {
        return bad_request(format!(
            "message_index {} out of range (session has {} messages)",
            req.message_index,
            messages.len()
        ))
        .into_response();
    };
    if anchor.role != "user" {
        return bad_request(format!(
            "message {} is '{}', not a user message: rewind anchors on a user input bubble",
            req.message_index, anchor.role
        ))
        .into_response();
    };
    if is_runtime_synthetic_user_message(anchor) {
        return bad_request(format!(
            "message {} is runtime-injected, not a real user input: pick a user input bubble",
            req.message_index
        ))
        .into_response();
    }
    let removed = messages.len() - req.message_index;
    if let Err(err) = truncate_history_messages(&session_file, req.message_index) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response();
    }
    let targets = stale_patch_targets_from_messages(&messages[..req.message_index]);
    if let Err(err) = write_stale_patch_targets_sqlite(&session_file, &targets) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response();
    }
    invalidate_context_history_cache_for(&session_file);
    (
        StatusCode::OK,
        Json(serde_json::json!({"removed": removed, "kept": req.message_index})),
    )
        .into_response()
}

pub(crate) async fn read_history(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<HistoryQuery>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let mut messages = match store.read_all_messages(&id) {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };
    // `?meta=roles`: bare array indexed by canonical position, so serve-chat
    // can resolve `u<N>` rewind ordinals without shipping the session payload.
    // Runtime-synthetic users use a distinct sentinel because they are not
    // real turn boundaries. `limit` is ignored
    // on purpose: a tail cut would renumber the rows and invalidate rewind
    // anchors, and the whole point of this shape is exactness at tiny size.
    if q.meta.as_deref() == Some("roles") {
        let roles: Vec<String> = messages
            .iter()
            .map(|message| {
                if is_runtime_synthetic_user_message(message) {
                    "runtime_synthetic_user".to_string()
                } else {
                    message.role.clone()
                }
            })
            .collect();
        let total = roles.len();
        let mut resp = (StatusCode::OK, Json(roles)).into_response();
        resp.headers_mut().insert(
            "x-history-total",
            axum::http::HeaderValue::from_str(&total.to_string())
                .unwrap_or(axum::http::HeaderValue::from_static("0")),
        );
        return resp;
    }
    // `?conclusion=N`: the N-th-most-recent assistant conclusion (1 =
    // latest), mirroring the serve-chat `/history last [N]` scan: assistant
    // rows with no tool calls and non-empty searchable text. The client
    // renders the one-row body like the local command; not found returns 400
    // carrying the exact local error wording.
    if let Some(nth) = q.conclusion {
        if nth == 0 {
            return bad_request(
                "invalid /history last argument: 0. try: /history last [N]".to_string(),
            )
            .into_response();
        }
        let mut seen = 0usize;
        let mut found: Option<&Message> = None;
        for message in messages.iter().rev() {
            if !is_serve_assistant_conclusion(message) {
                continue;
            }
            seen += 1;
            if seen == nth {
                found = Some(message);
                break;
            }
        }
        let Some(message) = found else {
            let detail = if seen == 0 {
                "[history] No assistant messages yet.".to_string()
            } else {
                format!("[history] Only {seen} assistant message(s) in history.")
            };
            return bad_request(detail).into_response();
        };
        let mut message = message.clone();
        redact_inline_image_data(&mut message.content);
        return (StatusCode::OK, Json(vec![message])).into_response();
    }
    // `?user_conclusion=N`: the final assistant conclusion after user row uN
    // and before the next user row. User ordinals match the `user_ordinal`
    // values attached to role-filtered `/history user` responses.
    if let Some(target_ordinal) = q.user_conclusion {
        if target_ordinal == 0 {
            return bad_request("user ordinal must be >= 1".to_string()).into_response();
        }
        let mut user_ordinal = 0usize;
        let mut target_found = false;
        let mut found: Option<&Message> = None;
        for message in &messages {
            if message.role == "user" && !is_runtime_synthetic_user_message(message) {
                if target_found {
                    break;
                }
                user_ordinal += 1;
                target_found = user_ordinal == target_ordinal;
                continue;
            }
            if target_found && is_serve_assistant_conclusion(message) {
                found = Some(message);
            }
        }
        let Some(message) = found else {
            let detail = if target_found {
                format!(
                    "[history] No assistant conclusion found for user message u{target_ordinal}."
                )
            } else {
                format!("[history] User message u{target_ordinal} was not found.")
            };
            return bad_request(detail).into_response();
        };
        let mut message = message.clone();
        redact_inline_image_data(&mut message.content);
        return (StatusCode::OK, Json(vec![message])).into_response();
    }
    // serve-chat preview views are bounded tails, so `?role` + `?count`
    // filter and tail-cut here and only the subset travels. Each user row
    // carries its global `user_ordinal` (counted before filtering, like the
    // client) so `u<N>` markers stay exact on a partial response. Absent
    // params (grep fetches, the mobile client) keep the unfiltered path
    // below.
    if q.role.is_some() || q.count.is_some() {
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut user_ordinal = 0usize;
        for message in &messages {
            let is_real_user =
                message.role == "user" && !is_runtime_synthetic_user_message(message);
            if is_real_user {
                user_ordinal += 1;
            }
            let role_matches = match q.role.as_deref() {
                None => true,
                Some("user") => is_real_user,
                Some("assistant") => message.role == "assistant",
                Some("tool") => message.role == "tool",
                // Same rule as the client system filter (system + internal notes).
                Some("system") => message.role == "system" || message.role == "internal_note",
                Some(other) => {
                    return bad_request(format!("invalid history role filter: {other}"))
                        .into_response();
                }
            };
            if !role_matches {
                continue;
            }
            let mut value = match serde_json::to_value(message) {
                Ok(v) => v,
                Err(err) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": err.to_string()})),
                    )
                        .into_response();
                }
            };
            if is_real_user {
                value["user_ordinal"] = serde_json::json!(user_ordinal);
            }
            rows.push(value);
        }
        if let Some(count) = q.count {
            let count = count.min(SERVE_HISTORY_VIEW_COUNT_CAP);
            if rows.len() > count {
                rows.drain(..rows.len() - count);
            }
        }
        for row in &mut rows {
            if let Some(content) = row.get_mut("content") {
                redact_inline_image_data(content);
            }
        }
        let mut resp = (StatusCode::OK, Json(rows)).into_response();
        resp.headers_mut().insert(
            "x-history-total",
            axum::http::HeaderValue::from_str(&messages.len().to_string())
                .unwrap_or(axum::http::HeaderValue::from_static("0")),
        );
        return resp;
    }
    // History is transferred over HTTP and the client renders text fields
    // only, so inline `data:` image payloads would ship megabytes of base64
    // for nothing. Redact the payload in place (keeping the block shape, the
    // media type, and every message index/role, so rewind anchors and role
    // filters stay valid) before the response is serialized.
    for message in &mut messages {
        redact_inline_image_data(&mut message.content);
    }
    // Total before the tail cut: the mobile client maps a tapped user bubble
    // back to its canonical index as `total - shown.length + bubble_index`
    // for `POST /rewind`. A header keeps the body shape (a bare message
    // array) unchanged, so older cached pages keep parsing.
    let total = messages.len();
    if let Some(limit) = q.limit {
        if messages.len() > limit {
            messages = messages.split_off(messages.len() - limit);
        }
    }
    let mut resp = (StatusCode::OK, Json(messages)).into_response();
    resp.headers_mut().insert(
        "x-history-total",
        axum::http::HeaderValue::from_str(&total.to_string())
            .unwrap_or(axum::http::HeaderValue::from_static("0")),
    );
    resp
}
