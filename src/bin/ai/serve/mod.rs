//! Serve mode: local HTTP API with parity to the CLI personal assistant.
//!
//! v1 scope: HTTP on loopback, Bearer auth (optional), session list/history,
//! one-shot turns via subprocess (same binary, `--session <id>`), session
//! fork/delete, skills/agents listing. Turn execution reuses the existing
//! one-shot path so behavior stays identical to `a --session <id> "<prompt>"`.

use std::{collections::HashMap, path::PathBuf, process::Stdio, sync::Arc};

use base64::Engine as _;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response, Sse, sse::Event},
    routing::{delete, get, post},
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};

use crate::ai::background;
use crate::ai::exe_path::runtime_exe;
use crate::commonw::configw;

use super::{
    agents,
    config_schema::AiConfig,
    history::{
        SessionServeConfig, SessionStore, SessionTitleOrigin,
        invalidate_context_history_cache_for, is_runtime_synthetic_user_message,
        truncate_history_messages, write_stale_patch_targets_sqlite,
    },
    model_names, models,
};
use super::driver::turn_runtime::stale_patch_targets_from_messages;
use super::driver::side_note::push_side_note;
pub(in crate::ai) mod chat;
pub(in crate::ai) mod ctl;


mod types;
mod info;
mod sessions;
mod history;
mod files;
mod turn;
#[cfg(test)]
mod tests;

pub(crate) use files::*;
pub(crate) use history::*;
pub(crate) use info::*;
pub(crate) use sessions::*;
pub(crate) use turn::*;
pub(crate) use types::*;

/// Run one turn by re-executing this binary in one-shot mode. This keeps serve
/// behavior identical to the local assistant without duplicating the driver.
///
/// Error detail is capped so a large dump (possibly containing local paths)
/// never floods the API response.
const MAX_TURN_ERROR_CHARS: usize = 4096;
/// Cap for one forwarded child-output line so a single tool dump cannot blow
/// up the SSE frame budget.
const MAX_SSE_LINE_CHARS: usize = 8192;

fn run_one_shot_turn(
    session_id: &str,
    prompt: &str,
    overrides: &TurnOverrides,
) -> std::io::Result<String> {
    let exe = runtime_exe()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--session")
        .arg(session_id)
        .arg(prompt);
    push_turn_override_args(&mut cmd, overrides);
    let out = cmd
        // Marks the child as serve-spawned (title skip); the FIFO env stays
        // unset here, so streaming paths keep seeing a plain child.
        .env(background::SERVE_CHILD_ENV, "1")
        .output()?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let mut combined = format!("turn failed: {stderr}{stdout}");
        if combined.len() > MAX_TURN_ERROR_CHARS {
            let skip = combined.len() - MAX_TURN_ERROR_CHARS;
            combined = format!("...[truncated]...{}", &combined[skip..]);
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            combined,
        ))
    }
}

async fn post_turn(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<TurnReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        return bad_request("prompt is empty".to_string()).into_response();
    }
    if let Err(err) = stage_turn_images(
        &SessionStore::new(&state.history_file).session_assets_dir(&id),
        &req.images,
    ) {
        return bad_request(err).into_response();
    }
    let mut overrides = match turn_overrides(&req) {
        Ok(overrides) => overrides,
        Err(err) => return bad_request(err).into_response(),
    };
    // A per-turn override wins; absent fields fall back to the session's
    // persisted serve config (then to the server default), so a turn spawned
    // without client flags still runs with the session's picks.
    let store = SessionStore::new(state.history_file.as_path());
    merge_session_config(&mut overrides, &read_session_config_or_default(&store, &id));
    let title_model = overrides.model.clone();
    let lock = session_lock(&state, &id).await;
    let guard = lock.lock().await;
    // Kick off the model title request in parallel with the turn child: the
    // title task never takes the session lock, and the pending prompt seeds it
    // before the child persists its first message.
    spawn_session_title_task(
        state.history_file.clone(),
        id.clone(),
        title_model.clone(),
        Some(prompt.clone()),
    );
    let session_id = id.clone();
    let output = tokio::task::spawn_blocking(move || {
        run_one_shot_turn(&session_id, &prompt, &overrides)
    })
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
        });
    let output = match output {
        Ok(Ok(text)) => text,
        Ok(Err(err)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
        Err(e) => return e.into_response(),
    };
    // The child is gone: retry the title request outside the lock it held for
    // its whole run, in case the parallel attempt above failed or raced an
    // empty history. No-ops when a title already settled.
    drop(guard);
    spawn_session_title_task(state.history_file.clone(), id.clone(), title_model, None);
    (
        StatusCode::OK,
        Json(TurnResp {
            session_id: id,
            output,
        }),
    )
        .into_response()
}

/// Request body for `POST /sessions/{id}/side-notes`: one guidance text for
/// the running turn, the remote counterpart of the local REPL's Ctrl+G
/// side-note. The text is appended to the session's side-note file queue,
/// which the turn child drains at the top of every iteration; queue appends
/// and the drain-side atomic rename compose without loss, so notes sent
/// mid-turn are seen by the next model request.
#[derive(Debug, Deserialize)]
struct SideNoteReq {
    text: String,
}

/// `POST /sessions/{id}/side-notes` (authed): queue one side-note for this
/// session's in-flight turn. Returns `{"queued": true}` even when no turn is
/// running; the note then waits in the queue for the next turn.
async fn post_side_note(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<SideNoteReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    if req.text.trim().is_empty() {
        return bad_request("side-note text is empty".to_string()).into_response();
    }
    // No session-file existence check: the queue file is independent of the
    // message history, and a note may arrive before the turn child persists
    // its first message. Queue appends and the drain-side atomic rename
    // compose without loss either way.
    let store = SessionStore::new(state.history_file.as_path());
    let session_file = store.session_history_file(&id);
    if let Err(err) = push_side_note(&session_file, req.text.trim(), "user", None) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({"session_id": id, "queued": true})),
    )
        .into_response()
}

async fn post_turn_sse(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<TurnReq>,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        return bad_request("prompt is empty".to_string()).into_response();
    }
    if let Err(err) = stage_turn_images(
        &SessionStore::new(&state.history_file).session_assets_dir(&id),
        &req.images,
    ) {
        return bad_request(err).into_response();
    }
    let mut overrides = match turn_overrides(&req) {
        Ok(overrides) => overrides,
        Err(err) => return bad_request(err).into_response(),
    };
    // A per-turn override wins; absent fields fall back to the session's
    // persisted serve config (then to the server default).
    let store = SessionStore::new(state.history_file.as_path());
    merge_session_config(&mut overrides, &read_session_config_or_default(&store, &id));
    let lock = session_lock(&state, &id).await;
    // Chunk-level streaming: the child publishes framed live events
    // (assistant-text deltas, thinking lifecycle, output-complete marker)
    // into a per-turn FIFO (see `background::SERVE_LIVE_FIFO_ENV`); the pump
    // forwards each frame as an SSE event while stdout lines (tool progress,
    // banners) keep flowing as message events. An output-complete frame ends
    // the turn early; the lock below is still held until the child is
    // reaped. Falls back to the line pump when the FIFO cannot be set up —
    // never fails the turn for streaming plumbing.
    let live_fifo = setup_live_fifo(&id);
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(128);
    let session_id = id.clone();
    let active_turns = Arc::clone(&state.active_turns);
    let confirms = Arc::clone(&state.confirms);
    // Remote confirmation needs both halves of the handoff: a live FIFO to
    // publish the question and a client that can answer it. Without either,
    // the child keeps the fail-closed path (a gated command reports to the
    // model) instead of blocking on a question nobody can see.
    let confirm_enabled = req.confirm == Some(true) && live_fifo.is_some();
    // The per-session lock moves into the pump task (not the handler scope):
    // the handler returns the SSE response immediately, so holding the guard
    // here would release it before the turn finishes and allow overlapping
    // writers on one session.
    let history_file = state.history_file.clone();
    let title_session_id = session_id.clone();
    let title_model = overrides.model.clone();
    // Kick off the model title request in parallel with the turn child; the
    // pending prompt seeds it before the child persists its first message.
    spawn_session_title_task(
        state.history_file.clone(),
        id.clone(),
        title_model.clone(),
        Some(prompt.clone()),
    );
    tokio::task::spawn(async move {
        let guard = lock.lock().await;
        let _ = tokio::task::spawn_blocking(move || {
            stream_child_turn(
                session_id,
                prompt,
                overrides,
                tx,
                live_fifo,
                active_turns,
                confirms,
                confirm_enabled,
            )
        })
        .await;
        // The child is gone: retry the title request outside the lock it held
        // for its whole run, in case the parallel attempt failed. No-ops when
        // a title already settled.
        drop(guard);
        spawn_session_title_task(history_file, title_session_id, title_model, None);
    });
    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (event, rx))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Blocking child-turn pump: runs the one-shot turn (same binary, same flags
/// as `run_one_shot_turn`) and forwards output as SSE events while the child
/// is still running: live-FIFO frames become `delta` / `thinking*` events
/// (one per arrival, chunk granularity), stdout lines become message events.
/// An output-complete frame sends `done` immediately while draining continues
/// until the child is reaped. Stderr is drained on a side thread so a chatty
/// child can never block on a full pipe.
fn stream_child_turn(
    session_id: String,
    prompt: String,
    overrides: TurnOverrides,
    tx: mpsc::Sender<Result<Event, std::convert::Infallible>>,
    mut live_fifo: Option<LiveFifo>,
    active_turns: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    confirms: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
    confirm_enabled: bool,
) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let t0 = std::time::Instant::now();
    let timing = TurnSendTiming::default();
    let send = |event: Event| {
        timing.stamp_first(t0);
        let _ = tx.blocking_send(Ok(event));
    };
    let send_done = || {
        timing.stamp_done(t0);
        send(done_event());
    };
    let child_done = Arc::new(AtomicBool::new(false));
    // Set when the turn observably ends: either the child's OutputComplete
    // frame (early `done`, silent bookkeeping still running) or the terminal
    // status below. Guards against a duplicated `done`; the per-session lock
    // stays held until the child is reaped either way.
    let turn_done = Arc::new(AtomicBool::new(false));
    let exe = match runtime_exe() {
        Ok(exe) => exe,
        Err(err) => {
            send(error_event(format!("cannot locate current binary: {err}")));
            send_done();
            return;
        }
    };
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--session")
        .arg(&session_id)
        .arg(&prompt);
    push_turn_override_args(&mut cmd, &overrides);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Marks the child as serve-spawned even when no FIFO is set up
        // (fallback line pump): the title skip must not depend on streaming.
        .env(background::SERVE_CHILD_ENV, "1");
    if confirm_enabled {
        // The child publishes its question on the live FIFO and reads the
        // answer as one stdin line: the daemon has no terminal for it to
        // inherit, and a closed stdin would read as an immediate "canceled".
        cmd.stdin(Stdio::piped())
            .env(background::SERVE_CONFIRM_ENV, "1");
    }
    if let Some(fifo) = live_fifo.as_ref() {
        cmd.env(background::SERVE_LIVE_FIFO_ENV, &fifo.path);
    }
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            // Every turn re-execs the daemon's own binary, so a spawn failure
            // here is most often that file having been replaced.
            send(error_event(format!(
                "cannot start turn process: {err}; if the daemon binary was replaced, restart it with `a --serve-restart`"
            )));
            send_done();
            return;
        }
    };
    // Registered for as long as the child runs, so an interrupt reaches
    // exactly this turn; the guard clears the entry when the child is reaped,
    // on every exit path below.
    let _active_turn = ActiveTurnGuard::register(&active_turns, &session_id, child.id());
    // Registered before the FIFO pump starts so an early question is never
    // dropped; the child's stdin moves into the slot, where the `/confirm`
    // endpoint writes the answer. A child without the pipe (spawn did not give
    // one) is left unpublished rather than published unanswerable.
    let confirm_guard = if confirm_enabled {
        child.stdin.take().map(|stdin| {
            ConfirmGuard::register(
                &confirms,
                &session_id,
                Arc::new(std::sync::Mutex::new(ConfirmSlot {
                    pending: None,
                    answers: Some(stdin),
                })),
            )
        })
    } else {
        None
    };
    let stderr_handle = child.stderr.take().map(|stderr| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut tail = String::new();
            let _ = std::io::BufReader::new(stderr).read_to_string(&mut tail);
            tail
        })
    });
    // The FIFO pump runs on its own thread: it owns the read end and stops
    // once the child is reaped and the pipe is drained (joined below).
    #[cfg(unix)]
    let fifo_handle = live_fifo
        .as_mut()
        .and_then(|fifo| fifo.reader.take())
        .map(|reader| {
            let tx = tx.clone();
            let pump_timing = timing.clone();
            let child_done = Arc::clone(&child_done);
            let turn_done = Arc::clone(&turn_done);
            let pump_confirm = confirm_guard.as_ref().map(|g| Arc::clone(g.slot()));
            std::thread::spawn(move || {
                pump_live_fifo(
                    reader,
                    &child_done,
                    &turn_done,
                    &tx,
                    pump_timing,
                    t0,
                    pump_confirm,
                )
            })
        });
    #[cfg(not(unix))]
    let fifo_handle: Option<std::thread::JoinHandle<()>> = None;
    if let Some(stdout) = child.stdout.take() {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
        {
            send(Event::default().data(clean_sse_line(&line)));
        }
    }
    let status = child.wait();
    child_done.store(true, Ordering::SeqCst);
    if let Some(handle) = fifo_handle {
        let _ = handle.join();
    }
    let stderr_tail = stderr_handle.and_then(|h| h.join().ok()).unwrap_or_default();
    // The pump may have ended the turn early via OutputComplete; only the
    // first `done` wins. A late failure still surfaces its detail as an
    // error event (harmless if the client already left on `done`).
    let already_done = turn_done.swap(true, Ordering::SeqCst);
    match status {
        Ok(status) if status.success() => {
            if !already_done {
                send_done();
            }
        }
        _ => {
            let mut detail = last_chars(&stderr_tail, MAX_TURN_ERROR_CHARS);
            if detail.trim().is_empty() {
                detail = "turn process failed".to_string();
            }
            send(error_event(detail));
            if !already_done {
                send_done();
            }
        }
    }
    // `live_fifo` drops here, removing the per-turn inode whatever happened.
    eprintln!(
        "[serve] turn session={} first_byte_ms={} done_ms={} total_ms={} early_done={}",
        session_id,
        timing.first_ms.load(std::sync::atomic::Ordering::SeqCst),
        timing.done_ms.load(std::sync::atomic::Ordering::SeqCst),
        t0.elapsed().as_millis(),
        already_done,
    );
}

/// Per-turn live-chunk FIFO: created by the SSE endpoint before spawning the
/// child, removed on drop (after the pump joins) whatever the outcome.
struct LiveFifo {
    path: PathBuf,
    reader: Option<std::fs::File>,
}

impl Drop for LiveFifo {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

static LIVE_FIFO_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Best-effort per-turn FIFO for chunk streaming. The read end opens here
/// (before the child spawns) so the child's on-demand writer open always
/// finds a reader. `None` means "fall back to the line pump".
/// Callers must have validated `session_id` (its charset is filename-safe).
#[cfg(unix)]
fn setup_live_fifo(session_id: &str) -> Option<LiveFifo> {
    use std::os::unix::ffi::OsStrExt;
    let n = LIVE_FIFO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "a-serve-{session_id}-{}-{n}.fifo",
        std::process::id()
    ));
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `mkfifo` only creates one inode at a fresh unique path with a
    // private mode; chunks may carry sensitive text, hence 0600.
    let mut made = unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) } == 0;
    if !made && std::io::Error::last_os_error().kind() == std::io::ErrorKind::AlreadyExists {
        // Stale inode from a killed server: reclaim the name and retry once.
        let _ = std::fs::remove_file(&path);
        made = unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) } == 0;
    }
    if !made {
        return None;
    }
    use std::os::unix::fs::OpenOptionsExt;
    let reader = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&path)
        .ok()?;
    Some(LiveFifo {
        path,
        reader: Some(reader),
    })
}

#[cfg(not(unix))]
fn setup_live_fifo(_session_id: &str) -> Option<LiveFifo> {
    None
}

/// Serve-live frame kinds, mirroring `background::ServeLiveKind`
/// discriminants (1 = delta, 2 = thinking start, 3 = thinking chunk,
/// 4 = thinking done, 5 = output complete, 6 = confirmation request,
/// 7 = confirmation resolved).
#[derive(Debug, PartialEq, Eq)]
enum ServeLiveEvent {
    Delta(String),
    ThinkingStart,
    Thinking(String),
    ThinkingDone,
    OutputComplete,
    /// Raw JSON payload of a remote confirmation request.
    ConfirmRequest(String),
    /// Raw JSON payload closing a confirmation request.
    ConfirmDone(String),
}

/// Incremental parser for serve-live frames (`[kind: u8][len: u32 BE][payload]`):
/// FIFO reads may split anywhere, so bytes accumulate until a whole frame is
/// present. Payloads are producer-guaranteed UTF-8; decoding is lossy only as
/// a defensive fallback. An unknown kind or absurd length resets the buffer
/// instead of desynchronizing the stream (impossible from our producer, only
/// from a foreign writer on the same inode).
#[derive(Default)]
struct ServeFrameDecoder {
    buf: Vec<u8>,
}

impl ServeFrameDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<ServeLiveEvent> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            if self.buf.len() < 5 {
                break;
            }
            let kind = self.buf[0];
            let len =
                u32::from_be_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]) as usize;
            let known = kind == background::ServeLiveKind::Delta as u8
                || kind == background::ServeLiveKind::ThinkingStart as u8
                || kind == background::ServeLiveKind::Thinking as u8
                || kind == background::ServeLiveKind::ThinkingDone as u8
                || kind == background::ServeLiveKind::OutputComplete as u8
                || kind == background::ServeLiveKind::ConfirmRequest as u8
                || kind == background::ServeLiveKind::ConfirmDone as u8;
            if !known || len > background::MAX_SERVE_FRAME_LEN as usize {
                self.buf.clear();
                break;
            }
            if self.buf.len() < 5 + len {
                break;
            }
            let payload = String::from_utf8_lossy(&self.buf[5..5 + len]).into_owned();
            self.buf.drain(..5 + len);
            out.push(match kind {
                k if k == background::ServeLiveKind::Delta as u8 => ServeLiveEvent::Delta(payload),
                k if k == background::ServeLiveKind::ThinkingStart as u8 => {
                    ServeLiveEvent::ThinkingStart
                }
                k if k == background::ServeLiveKind::Thinking as u8 => {
                    ServeLiveEvent::Thinking(payload)
                }
                k if k == background::ServeLiveKind::ThinkingDone as u8 => {
                    ServeLiveEvent::ThinkingDone
                }
                k if k == background::ServeLiveKind::ConfirmRequest as u8 => {
                    ServeLiveEvent::ConfirmRequest(payload)
                }
                k if k == background::ServeLiveKind::ConfirmDone as u8 => {
                    ServeLiveEvent::ConfirmDone(payload)
                }
                k if k == background::ServeLiveKind::OutputComplete as u8 => {
                    ServeLiveEvent::OutputComplete
                }
                // Every kind accepted by the `known` gate above must have an
                // arm here: falling through would end the turn for a frame
                // that is not an output-complete marker.
                _ => ServeLiveEvent::OutputComplete,
            });
        }
        out
    }

    /// Drain on EOF: parse whatever complete frames remain; a lone partial
    /// frame is a truncated write and is dropped.
    fn finish(&mut self) -> Vec<ServeLiveEvent> {
        let tail = std::mem::take(&mut self.buf);
        let events = self.push(&tail);
        self.buf.clear();
        events
    }
}

/// Map one parsed frame to its SSE event. `OutputComplete` has no event here:
/// the caller ends the turn with `done` instead (see `forward_live_event`).
fn live_event(event: ServeLiveEvent) -> Option<Event> {
    match event {
        ServeLiveEvent::Delta(text) => Some(delta_event(text)),
        ServeLiveEvent::ThinkingStart => Some(Event::default().event("thinking_start").data("")),
        ServeLiveEvent::Thinking(text) => Some(Event::default().event("thinking").data(
            serde_json::json!({"text": text}).to_string(),
        )),
        ServeLiveEvent::ThinkingDone => {
            Some(Event::default().event("thinking_done").data(""))
        }
        // Confirmation frames are built by `forward_live_event`, which holds
        // the pending state and adds the answer token; they never get here.
        ServeLiveEvent::ConfirmRequest(_) | ServeLiveEvent::ConfirmDone(_) => None,
        ServeLiveEvent::OutputComplete => None,
    }
}

/// Forward one parsed frame. An `OutputComplete` frame ends the user-visible
/// turn immediately (the child only runs silent bookkeeping after that
/// point); the first frame wins and later ones are no-ops, so the terminal
/// `done` below is never duplicated.
fn forward_live_event(
    event: ServeLiveEvent,
    turn_done: &std::sync::atomic::AtomicBool,
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    timing: &TurnSendTiming,
    t0: std::time::Instant,
    confirm: Option<&std::sync::Mutex<ConfirmSlot>>,
) {
    timing.stamp_first(t0);
    if matches!(event, ServeLiveEvent::OutputComplete) {
        if !turn_done.swap(true, std::sync::atomic::Ordering::SeqCst) {
            timing.stamp_done(t0);
            let _ = tx.blocking_send(Ok(done_event()));
        }
        return;
    }
    // Confirmation frames carry live state (the pending question) as well as a
    // client event, so they are built here; every other kind maps 1:1. The
    // slot is filled before the send: an answer can only arrive after a client
    // saw the event, and a client that lost the stream instead of receiving it
    // reads the slot through `GET .../confirm`.
    let sse = match event {
        ServeLiveEvent::ConfirmRequest(payload) => {
            let Some((id, prompt)) = parse_confirm_request(&payload) else {
                // A fragment of a split payload names no question; dropping it
                // must not end the turn, which the fallback mapping would do.
                return;
            };
            let token = NEXT_CONFIRM_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(slot) = confirm {
                if let Ok(mut slot) = slot.lock() {
                    slot.pending = Some(PendingConfirm {
                        id,
                        prompt: prompt.clone(),
                        token,
                    });
                }
            }
            Event::default()
                .event("confirm_request")
                .data(serde_json::json!({"id": id, "prompt": prompt, "token": token}).to_string())
        }
        ServeLiveEvent::ConfirmDone(payload) => {
            // A close that cannot name its question must not hide one: only a
            // matching id clears the slot.
            if let Some(id) = parse_confirm_id(&payload) {
                if let Some(slot) = confirm {
                    if let Ok(mut slot) = slot.lock() {
                        if slot.pending.as_ref().is_some_and(|p| p.id == id) {
                            slot.pending = None;
                        }
                    }
                }
            }
            Event::default().event("confirm_done").data(payload)
        }
        other => match live_event(other) {
            Some(e) => e,
            None => return,
        },
    };
    let _ = tx.blocking_send(Ok(sse));
}

/// Forward live-FIFO frames as SSE events until the child is reaped and the
/// pipe is drained. Mirrors the `attach_live_session` reader: an EOF while
/// the child is still alive is transient (its writer opens on demand at the
/// first chunk), so it sleeps instead of spinning; after the child exits EOF
/// is terminal.
#[cfg(unix)]
fn pump_live_fifo(
    reader: std::fs::File,
    child_done: &std::sync::atomic::AtomicBool,
    turn_done: &std::sync::atomic::AtomicBool,
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    timing: TurnSendTiming,
    t0: std::time::Instant,
    confirm: Option<Arc<std::sync::Mutex<ConfirmSlot>>>,
) {
    use std::os::fd::AsRawFd;
    let fd = reader.as_raw_fd();
    // Held for the whole pump: dropping it would close the read end early.
    let _owned = reader;
    let mut decoder = ServeFrameDecoder::default();
    let mut buf = [0u8; 8192];
    loop {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: polls one valid open fifo fd with a bounded timeout; the
        // fd outlives this loop via `_owned` above.
        let ready = unsafe { libc::poll(&mut pfd, 1, 200) };
        if ready < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        // SAFETY: reads at most `buf.len()` bytes from the fifo into `buf`.
        let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 {
            let kind = std::io::Error::last_os_error().kind();
            if kind != std::io::ErrorKind::Interrupted && kind != std::io::ErrorKind::WouldBlock {
                break;
            }
            continue;
        }
        if n == 0 {
            if child_done.load(std::sync::atomic::Ordering::SeqCst) {
                for event in decoder.finish() {
                    forward_live_event(event, turn_done, tx, &timing, t0, confirm.as_deref());
                }
                break;
            }
            // Transient EOF (writer not opened yet): back off, don't spin.
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }
        for event in decoder.push(&buf[..n as usize]) {
            forward_live_event(event, turn_done, tx, &timing, t0, confirm.as_deref());
        }
    }
}

fn delta_event(delta: String) -> Event {
    Event::default().event("delta").data(
        serde_json::json!({"delta": delta}).to_string(),
    )
}

fn truncate_line(line: &str) -> String {
    if line.chars().count() <= MAX_SSE_LINE_CHARS {
        line.to_string()
    } else {
        let kept: String = line.chars().take(MAX_SSE_LINE_CHARS).collect();
        format!("{kept}...[line truncated]")
    }
}

/// Normalize one child-stdout row for SSE transport: strip the live-terminal
/// overwrite prefix (`\r\x1b[2K`) that completed/failed tool rows carry to
/// redraw the `running` row in place, and neutralize every other
/// cursor-addressing escape inside the row. SSE rows are append-only, so the
/// prefix is meaningless on the wire; worse, the bare `\r` makes axum split
/// the row and re-prefix `data:`, injecting a literal `data: ` fragment that
/// only a terminal honoring the erase escape can hide again. Cursor moves
/// (`CSI A/B`), region erases (`CSI J/K`) and friends address screen rows
/// that only exist on the child's live terminal: replayed verbatim on the
/// chat client's screen they erase or overwrite unrelated rows (ragged
/// indents, duplicated-looking status lines). SGR color (`CSI ... m`) is
/// kept: it styles only the row itself and renders identically everywhere.
/// A bare mid-line `\r` (same-row overwrite, e.g. progress counters) passes
/// through: the client's terminal resolves it exactly like the local one,
/// and splitting on it would destroy echoed tool output that legitimately
/// contains carriage returns. `BufRead::lines` already removed the line
/// ending, so a leading `\r` here is always the overwrite control, never
/// content.
fn clean_sse_line(line: &str) -> String {
    // Neutralize cursor addressing first: stripping it can reveal a leading
    // `\r` (e.g. `\x1b[1A\r\x1b[2K...`) that the prefix wash below must see.
    let stripped = strip_non_sgr_csi(line);
    let line = stripped.strip_prefix('\r').unwrap_or(&stripped);
    let line = line.strip_prefix("\x1b[2K").unwrap_or(line);
    truncate_line(line)
}

/// Remove every ANSI CSI sequence except SGR color (`ESC [ ... m`).
/// Malformed (unterminated) introducers are dropped: a half-sequence can
/// never render as intended on the client.
fn strip_non_sgr_csi(line: &str) -> String {
    let mut kept = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            kept.push(ch);
            continue;
        }
        if chars.peek() == Some(&'[') {
            chars.next(); // consume '['
            let mut seq = String::from("[");
            for ch in chars.by_ref() {
                seq.push(ch);
                if ('\x40'..='\x7e').contains(&ch) {
                    break;
                }
            }
            // SGR color styles only the row itself: keep. Anything else
            // (erase, cursor moves, scroll) addresses foreign screen rows.
            if seq.ends_with('m') {
                kept.push('\x1b');
                kept.push_str(&seq);
            }
        } else {
            // Single-character non-CSI escape (cursor save/restore `7`/`8`,
            // `M`, ...): drop the introducer plus its target character.
            // Multi-character sequences (`(B`, OSC `]...`) have no emitters
            // on this path, so they are left for a future rule if one appears.
            chars.next();
        }
    }
    kept
}

fn last_chars(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        text.to_string()
    } else {
        let kept: String = chars[chars.len() - limit..].iter().collect();
        format!("...[truncated]...{kept}")
    }
}

fn error_event(detail: String) -> Event {
    Event::default().event("error").data(
        serde_json::json!({"error": detail}).to_string(),
    )
}

fn done_event() -> Event {
    Event::default().event("done").data("")
}

async fn list_skills(State(state): State<ServeState>, headers: HeaderMap) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let manifests = super::skills::load_all_skills();
    (StatusCode::OK, Json(manifests)).into_response()
}

async fn list_agents(State(state): State<ServeState>, headers: HeaderMap) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let manifests = super::agents::load_all_agents();
    (StatusCode::OK, Json(manifests)).into_response()
}

fn is_loopback_bind(bind: &str) -> bool {
    // Parse the host part instead of prefix-matching so that names like
    // "127.x.evil.com" cannot pass the loopback gate.
    let host = bind
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(bind)
        .trim();
    let host = host.trim_matches(|c| c == '[' || c == ']').to_ascii_lowercase();
    if host == "localhost" {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Effective serve bind: an explicit `--serve-bind` wins, then
/// `ai.serve.bind`, then [`DEFAULT_BIND`]. Shared by `run_serve` and the
/// `--serve-*` management commands so status/start always probe the same
/// address the server listens on.
pub(in crate::ai) fn resolve_serve_bind(cli_bind: &str, cfg_bind: &str) -> String {
    if !cli_bind.trim().is_empty() {
        cli_bind.trim().to_string()
    } else if !cfg_bind.trim().is_empty() {
        cfg_bind.trim().to_string()
    } else {
        DEFAULT_BIND.to_string()
    }
}

pub(in crate::ai) async fn run_serve(
    cli: super::cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the same config resolution as the CLI so serve sees the same
    // history file, models, and sandbox settings.
    let app_config = super::config::load_config()?;
    let history_file = app_config.history_file.clone();
    let cfg = configw::get_all_config();
    let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
    let bind = resolve_serve_bind(&cli.serve_bind, &from_cfg);
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    if token.is_empty() && !is_loopback_bind(&bind) {
        return Err(
            "refusing to serve without ai.serve.token on a non-loopback bind; set ai.serve.token or bind 127.0.0.1".into(),
        );
    }
    if token.is_empty() {
        eprintln!("[serve] warning: ai.serve.token is empty; listening on loopback without auth");
    } else {
        // The token is baked into ServeState at startup: a later
        // `a config set ai.serve.token` needs a restart to take effect.
        // Log the length (never the secret) so a 401 can be told apart
        // from "old process still running".
        eprintln!(
            "[serve] token auth enabled ({} chars); clients must send Authorization: Bearer <token>",
            token.len()
        );
    }
    let state = ServeState {
        history_file,
        // Same authority the turn child runs with (it inherits this process's
        // cwd at spawn), so a path the assistant printed resolves here.
        workspace_root: crate::ai::driver::runtime_ctx::effective_cwd()
            .unwrap_or_else(|_| PathBuf::from(".")),
        token,
        // One entry per touched session; bounded by session count in practice.
        // Idle eviction is deferred to a later multi-instance pass.
        locks: Arc::new(Mutex::new(HashMap::new())),
        active_turns: Arc::new(std::sync::Mutex::new(HashMap::new())),
        confirms: Arc::new(std::sync::Mutex::new(HashMap::new())),
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/app", get(serve_app))
        .route("/info", get(server_info))
        .route("/sessions", get(list_sessions).post(create_session))
        .route("/sessions/{id}/fork", post(fork_session))
        .route("/sessions/{id}/title", post(set_session_title))
        .route("/sessions/{id}/config", get(get_session_config).post(set_session_config))
        .route("/sessions/{id}", delete(delete_session))
        .route("/sessions/{id}/history", get(read_history)).route("/sessions/{id}/rewind", post(rewind_history))
        .route(
            "/sessions/{id}/turns",
            post(post_turn).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route(
            "/sessions/{id}/turns/stream",
            post(post_turn_sse).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route("/sessions/{id}/interrupt", post(post_interrupt))
        .route("/sessions/{id}/side-notes", post(post_side_note))
        .route("/sessions/{id}/file", get(get_session_file))
        .route(
            "/sessions/{id}/confirm",
            get(get_confirm).post(post_confirm),
        )
        .route("/skills", get(list_skills))
        .route("/agents", get(list_agents))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    // Record this process in the global serve state file so `--serve-status`
    // and `--serve-stop` see a foreground server too. Best-effort: a missing
    // state file only costs manageability, never availability.
    ctl::note_foreground_serve(&bind);
    eprintln!("[serve] listening on http://{bind}");
    axum::serve(listener, app).await?;
    // The listener shut down cleanly; drop our claim so a later status does
    // not report a stale pid. (A signal-killed server leaves a stale file
    // behind; status/start self-heal by reaping it.)
    ctl::clear_foreground_serve();
    Ok(())
}
