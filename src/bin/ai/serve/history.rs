//! History routes: rewind and read.

use super::*;

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
