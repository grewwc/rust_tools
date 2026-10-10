//! Turn lifecycle helpers and the `/interrupt` and `/confirm` routes.
//! The SSE stream bridge core lives in `mod.rs` and uses the guards and
//! parsers defined here.

use super::*;

/// Model a serve turn runs with when the client sent no per-turn override.
///
/// Mirrors `models::initial_model` minus its CLI-override branch: the child CLI
/// carries no `--model`, so the config default (resolved through the registry,
/// falling back to the registry default) is what the turn uses.
pub(crate) fn default_turn_model() -> String {
    let cfg = configw::get_all_config();
    cfg.get_opt(AiConfig::MODEL_DEFAULT)
        .filter(|v| !v.trim().is_empty())
        .map(|v| super::models::determine_model(&v))
        .unwrap_or_else(super::models::default_model)
}

/// Generate the session's model title for a served turn.
///
/// The daemon calls this twice per turn: at the start, in parallel with the
/// turn child (`pending_prompt` seeds the request so it does not wait for the
/// child's writes), and after the child was reaped, as a retry when the
/// parallel attempt failed. Turn children deliberately skip the title
/// round-trip: they hold the session's turn lock until they exit, so the extra
/// request would stall the session's next turn. Best-effort by construction:
/// it never delays or fails a turn, and it leaves a session that already has a
/// model title (or a user-set one) alone.
pub(crate) fn spawn_session_title_task(
    history_file: PathBuf,
    session_id: String,
    model: Option<String>,
    pending_prompt: Option<String>,
) {
    // A per-turn `--model` override is resolved through the registry in the
    // child too, so resolve it here to keep the title request on the same
    // model the turn actually ran with.
    let model = model
        .map(|model| super::models::determine_model(&model))
        .unwrap_or_else(default_turn_model);
    tokio::task::spawn(async move {
        crate::ai::driver::turn_runtime::generate_session_title_outside_turn(
            history_file.as_path(),
            &session_id,
            &model,
            pending_prompt.as_deref(),
        )
        .await;
    });
}

/// Ask the in-flight turn child for `session_id` to stop early by delivering
/// SIGINT to it. The child is the same one-shot binary the local REPL runs, so
/// it takes the local first-Ctrl+C path (cancel the stream, finalize the
/// partial turn) instead of being killed. `false` means no turn child is
/// registered: the turn already finished, which is a normal answer for an
/// interrupt that raced the end of the stream.
pub(crate) fn interrupt_active_turn(
    active_turns: &std::sync::Mutex<HashMap<String, u32>>,
    session_id: &str,
) -> bool {
    let pid = active_turns
        .lock()
        .ok()
        .and_then(|turns| turns.get(session_id).copied());
    #[cfg(unix)]
    if let Some(pid) = pid {
        // SAFETY: `kill` only delivers a signal. A child that exited between
        // the lookup above and here yields ESRCH, reported as "not
        // interrupted" rather than an error.
        return unsafe { libc::kill(pid as libc::pid_t, libc::SIGINT) == 0 };
    }
    #[cfg(not(unix))]
    let _ = pid;
    false
}

/// `POST /sessions/{id}/interrupt` (authed): stop this session's in-flight
/// turn, the remote counterpart of the local REPL's first Ctrl+C. Only this
/// session's turn child is signalled, so turns on other sessions keep running.
/// A turn parked on a confirmation question is unblocked as well, by closing
/// the pipe that its answer would travel on (see [`dismiss_pending_confirm`]).
/// `{"interrupted": false}` means no turn was running.
pub(crate) async fn post_interrupt(
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
    let interrupted = interrupt_active_turn(&state.active_turns, &id);
    // A turn can also be parked on a question. SIGINT reaches that child but
    // not its blocked stdin read, so stopping has to close the pipe as well;
    // otherwise the button would report success while the turn sits there.
    dismiss_pending_confirm(&state.confirms, &id);
    (
        StatusCode::OK,
        Json(serde_json::json!({"session_id": id, "interrupted": interrupted})),
    )
        .into_response()
}

/// Removes the session's entry from the active-turn registry when the turn
/// child is reaped, so an interrupt can only ever reach a live process.
pub(crate) struct ActiveTurnGuard {
    map: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    session_id: String,
    pid: u32,
}

impl ActiveTurnGuard {
    pub(crate) fn register(
        map: &Arc<std::sync::Mutex<HashMap<String, u32>>>,
        session_id: &str,
        pid: u32,
    ) -> Self {
        if let Ok(mut turns) = map.lock() {
            turns.insert(session_id.to_string(), pid);
        }
        Self {
            map: Arc::clone(map),
            session_id: session_id.to_string(),
            pid,
        }
    }
}

impl Drop for ActiveTurnGuard {
    fn drop(&mut self) {
        if let Ok(mut turns) = self.map.lock() {
            // Only clear our own entry: a stale guard must never hide a newer
            // turn on the same session.
            if turns.get(&self.session_id).copied() == Some(self.pid) {
                turns.remove(&self.session_id);
            }
        }
    }
}

/// One unanswered confirmation question, as published by a turn child.
#[derive(Debug, Clone)]
pub(crate) struct PendingConfirm {
    pub(crate) id: u64,
    pub(crate) prompt: String,
    /// Token handed to the client with the question; an answer must echo it.
    /// Child-side ids restart at 1 in every turn, so the id alone cannot tell
    /// one turn's question from another's.
    pub(crate) token: u64,
}

/// Hands out the [`PendingConfirm::token`] values. Comparing the echoed token
/// is what keeps a dialog left over from an earlier turn from deciding the
/// question that took its id.
pub(crate) static NEXT_CONFIRM_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Per-turn confirmation channel: the question the child is blocked on (if
/// any) and the pipe that answers it. The slot lives in [`ServeState`], not in
/// the SSE handler, so a question asked while the page is away (phone in the
/// background, connection dropped) is still there to be answered later.
#[derive(Debug, Default)]
pub(crate) struct ConfirmSlot {
    pub(crate) pending: Option<PendingConfirm>,
    /// Writing end of the turn child's stdin, taken from the spawned child.
    /// Held here so the answer path never depends on the client that started
    /// the turn.
    pub(crate) answers: Option<std::process::ChildStdin>,
}

/// Owns the session's confirmation entry for one turn: inserted before the
/// FIFO pump starts, removed when the turn child is reaped on any exit path.
pub(crate) struct ConfirmGuard {
    map: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
    session_id: String,
    slot: Arc<std::sync::Mutex<ConfirmSlot>>,
}

impl ConfirmGuard {
    pub(crate) fn register(
        map: &Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
        session_id: &str,
        slot: Arc<std::sync::Mutex<ConfirmSlot>>,
    ) -> Self {
        if let Ok(mut confirms) = map.lock() {
            confirms.insert(session_id.to_string(), Arc::clone(&slot));
        }
        Self {
            map: Arc::clone(map),
            session_id: session_id.to_string(),
            slot,
        }
    }

    pub(crate) fn slot(&self) -> &Arc<std::sync::Mutex<ConfirmSlot>> {
        &self.slot
    }
}

impl Drop for ConfirmGuard {
    fn drop(&mut self) {
        if let Ok(mut confirms) = self.map.lock() {
            // Only clear our own entry: a stale guard must never hide a newer
            // turn's channel on the same session.
            if confirms
                .get(&self.session_id)
                .is_some_and(|cur| Arc::ptr_eq(cur, &self.slot))
            {
                confirms.remove(&self.session_id);
            }
        }
    }
}

/// Parse `{"id":<u64>,"prompt":"..."}` as published by a turn child. A
/// malformed frame is dropped rather than stored: the child always publishes
/// both fields, and a half-parsed question could not be answered.
pub(crate) fn parse_confirm_request(payload: &str) -> Option<(u64, String)> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    Some((
        value.get("id")?.as_u64()?,
        value.get("prompt")?.as_str()?.to_string(),
    ))
}

/// Parse the `{"id":<u64>}` frame that closes a confirmation request.
pub(crate) fn parse_confirm_id(payload: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()?
        .get("id")?
        .as_u64()
}

/// The session's live confirmation channel, if a turn with the channel
/// enabled is running.
pub(crate) fn confirm_slot(state: &ServeState, id: &str) -> Option<Arc<std::sync::Mutex<ConfirmSlot>>> {
    state.confirms.lock().ok()?.get(id).map(Arc::clone)
}

/// Unblock a turn that is parked on a confirmation question.
///
/// The child is blocked reading its own stdin, and neither a delivered SIGINT
/// nor its own `close(STDIN_FILENO)` wakes that read: what ends it is the write
/// end going away, because a pipe with no writer reads as EOF. Dropping the
/// answer pipe therefore makes the child's reader return `None`, which its
/// gates already report as "canceled" — the same outcome as Ctrl+C at the
/// local prompt. Clearing the question on the way out is also what lets a
/// client stop showing it.
pub(crate) fn dismiss_pending_confirm(
    confirms: &std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>,
    session_id: &str,
) {
    let Some(slot) = confirms.lock().ok().and_then(|m| m.get(session_id).cloned()) else {
        return;
    };
    let Some(mut slot) = slot.lock().ok() else {
        return;
    };
    // Only a question in flight needs unblocking: dropping the pipe on a turn
    // that is merely streaming would also cancel the *next* question it asks.
    if slot.pending.take().is_some() {
        slot.answers = None;
    }
}

pub(crate) fn conflict(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({"error": msg})),
    )
}

/// `GET /sessions/{id}/confirm` (authed): the question this session's turn is
/// waiting on, so a client that missed the `confirm_request` event (page
/// reloaded, phone woke up) can still show and answer it. `{"pending": null}`
/// means there is nothing to answer right now. `running` travels with it
/// because clients already poll this endpoint: a page that never saw the
/// stream can still offer to stop a turn that is running here.
pub(crate) async fn get_confirm(
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
    let pending = confirm_slot(&state, &id).and_then(|slot| slot.lock().ok()?.pending.clone());
    let running = state
        .active_turns
        .lock()
        .ok()
        .is_some_and(|turns| turns.contains_key(&id));
    let body = match pending {
        Some(p) => {
            serde_json::json!({
                "pending": {"id": p.id, "prompt": p.prompt, "token": p.token},
                "running": running,
            })
        }
        None => serde_json::json!({"pending": null, "running": running}),
    };
    (StatusCode::OK, Json(body)).into_response()
}

/// Answer body for [`post_confirm`].
#[derive(Debug, Deserialize)]
pub(crate) struct ConfirmAnswerReq {
    pub(crate) id: u64,
    /// Echo of the token the question was shown with.
    pub(crate) token: u64,
    pub(crate) allow: bool,
}

/// `POST /sessions/{id}/confirm` (authed): answer the pending question, which
/// the child receives as one `yes`/`no` line on stdin. `409` means the id no
/// longer names a pending question (answered on another device, superseded, or
/// the turn ended); clients treat that as terminal, not as a retry.
pub(crate) async fn post_confirm(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<ConfirmAnswerReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let Some(slot) = confirm_slot(&state, &id) else {
        return conflict("no confirmation is pending for this session".to_string()).into_response();
    };
    // One lock for the whole answer: two clients answering the same question
    // must not both write a line into the child's stdin.
    let mut slot = match slot.lock() {
        Ok(slot) => slot,
        Err(_) => {
            return conflict("confirmation state is unavailable".to_string()).into_response();
        }
    };
    let Some(pending) = slot.pending.as_ref().map(|p| (p.id, p.token)) else {
        return conflict("no confirmation is pending for this session".to_string()).into_response();
    };
    if pending != (req.id, req.token) {
        // The answer names a question that is already gone, or one shown by a
        // dialog that outlived it; a newer question stays published for the
        // client to re-read instead of being decided by stale text.
        return conflict("no such confirmation is pending".to_string()).into_response();
    }
    let line = if req.allow { "yes\n" } else { "no\n" };
    let write = match slot.answers.as_mut() {
        Some(stdin) => {
            use std::io::Write;
            stdin.write_all(line.as_bytes()).and_then(|()| stdin.flush())
        }
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "confirmation channel is closed",
        )),
    };
    if let Err(err) = write {
        // A broken pipe means the question can never be answered (the child is
        // gone), so drop it instead of leaving the client to retry forever.
        slot.pending = None;
        return conflict(format!("confirmation channel closed: {err}")).into_response();
    }
    // Cleared only once the answer reached the child; the child's own
    // `confirm_done` frame is then a no-op.
    slot.pending = None;
    (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
}

/// Send-side millisecond clock for the per-turn diagnostic line at the end
/// of [`stream_child_turn`]. Shared by the stdout loop and the FIFO pump
/// thread; every stamp is first-writer-wins so concurrent forwards cannot
/// skew the spans.
#[derive(Clone, Default)]
pub(crate) struct TurnSendTiming {
    pub(crate) first_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub(crate) done_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl TurnSendTiming {
    pub(crate) fn stamp_ms(target: &std::sync::atomic::AtomicU64, t0: std::time::Instant) {
        let ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let _ = target.compare_exchange(
            0,
            ms,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    pub(crate) fn stamp_first(&self, t0: std::time::Instant) {
        Self::stamp_ms(&self.first_ms, t0);
    }

    pub(crate) fn stamp_done(&self, t0: std::time::Instant) {
        Self::stamp_ms(&self.done_ms, t0);
    }
}
