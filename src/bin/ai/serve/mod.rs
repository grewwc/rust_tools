//! Serve mode: HTTP(S) API for the CLI personal assistant.
//!
//! Turns re-execute this binary in one-shot mode to reuse the driver, with
//! JSON responses or SSE output. Session lifecycle, history, and runtime
//! catalogs are exposed through authenticated routes; authentication may be
//! omitted only for a loopback bind.

#[cfg(feature = "serve")]
use std::{collections::HashMap, path::PathBuf, process::Stdio, sync::Arc};

#[cfg(feature = "serve")]
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response, Sse, sse::Event},
    routing::{any, delete, get, post},
};
#[cfg(feature = "serve")]
use futures_util::stream;
#[cfg(feature = "serve")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serve")]
use tokio::sync::{Mutex, mpsc};

#[cfg(feature = "serve")]
use crate::ai::background;
#[cfg(feature = "serve")]
use crate::ai::exe_path::runtime_exe;
#[cfg(feature = "serve")]
use crate::commonw::configw;

#[cfg(feature = "serve")]
use super::driver::side_note::{list_pending_side_notes, push_side_note};
#[cfg(feature = "serve")]
use super::driver::turn_runtime::{
    SERVE_TITLE_SEED_ENV, SERVE_TITLE_SESSION_ENV, SessionTitleSeed,
    stale_patch_targets_from_messages,
};
#[cfg(feature = "serve")]
use super::{
    agents,
    config_schema::AiConfig,
    history::{
        SessionServeConfig, SessionStore, SessionTitleOrigin, invalidate_context_history_cache_for,
        is_runtime_synthetic_user_message, redact_inline_image_data, truncate_history_messages,
        write_stale_patch_targets_sqlite,
    },
    model_names, models,
};
pub(in crate::ai) mod chat;
#[cfg(feature = "serve")]
pub(in crate::ai) mod ctl;

#[cfg(all(feature = "serve", unix))]
mod changes;
#[cfg(feature = "serve")]
mod checkpoint;
#[cfg(feature = "serve")]
mod commands;
#[cfg(feature = "serve")]
mod compression;
#[cfg(feature = "serve")]
mod files;
#[cfg(feature = "serve")]
mod history;
#[cfg(feature = "serve")]
mod info;
#[cfg(feature = "serve")]
mod session_cwd;
#[cfg(feature = "serve")]
mod session_groups;
#[cfg(feature = "serve")]
mod sessions;
#[cfg(all(test, feature = "serve"))]
mod tests;
#[cfg(feature = "serve")]
mod tls;
#[cfg(feature = "serve")]
mod transfer;
#[cfg(feature = "serve")]
mod turn;
mod types;
#[cfg(feature = "serve")]
mod write_roots;

#[cfg(feature = "serve")]
pub(crate) use checkpoint::*;
#[cfg(feature = "serve")]
pub(crate) use commands::*;
#[cfg(feature = "serve")]
pub(crate) use files::*;
#[cfg(feature = "serve")]
pub(crate) use history::*;
#[cfg(feature = "serve")]
pub(crate) use info::*;
#[cfg(feature = "serve")]
pub(crate) use session_cwd::*;
#[cfg(feature = "serve")]
use session_groups::*;
#[cfg(feature = "serve")]
pub(crate) use sessions::*;
#[cfg(feature = "serve")]
pub(crate) use transfer::*;
#[cfg(feature = "serve")]
pub(crate) use turn::*;
pub(crate) use types::*;

/// Error detail is capped so a large dump (possibly containing local paths)
/// never floods the API response.
const MAX_TURN_ERROR_CHARS: usize = 4096;
const APP_ENTRY_PATHS: [&str; 3] = ["/", "/app", "/app/"];
/// Cap for one forwarded child-output line so a single tool dump cannot blow
/// up the SSE frame budget.
const MAX_SSE_LINE_CHARS: usize = 8192;

/// Point the turn child at the width its client draws at. The variable is
/// cleared first: a turn that reported no width must not inherit a stale one
/// from the daemon's environment, which would lay the child's rows out for a
/// screen nobody is looking at.
#[cfg(feature = "serve")]
fn apply_term_cols_env(cmd: &mut std::process::Command, term_cols: Option<u16>) {
    cmd.env_remove(background::SERVE_TERM_COLS_ENV);
    if let Some(cols) = term_cols {
        cmd.env(background::SERVE_TERM_COLS_ENV, cols.to_string());
    }
}

#[cfg(feature = "serve")]
fn apply_title_seed_env(
    cmd: &mut std::process::Command,
    session_id: &str,
    seed: &SessionTitleSeed,
) {
    cmd.env_remove(SERVE_TITLE_SEED_ENV);
    cmd.env_remove(SERVE_TITLE_SESSION_ENV);
    if let Some(value) = seed.child_env_value() {
        cmd.env(SERVE_TITLE_SEED_ENV, value);
        cmd.env(SERVE_TITLE_SESSION_ENV, session_id);
    }
}

#[cfg(feature = "serve")]
fn apply_session_history_env(cmd: &mut std::process::Command, history_file: &std::path::Path) {
    cmd.env(super::persona::SERVE_SESSION_HISTORY_ENV, history_file);
}

#[cfg(feature = "serve")]
fn run_one_shot_turn(
    session_id: &str,
    history_file: &std::path::Path,
    prompt: &str,
    overrides: &TurnOverrides,
    cwd: &std::path::Path,
    term_cols: Option<u16>,
    title_seed: &SessionTitleSeed,
) -> std::io::Result<String> {
    let exe = runtime_exe()?;
    let mut cmd = std::process::Command::new(exe);
    push_turn_args(&mut cmd, session_id, prompt, overrides);
    // The child renders the rows this client displays; its own stdout is a
    // pipe, so the client's width travels through the environment.
    apply_term_cols_env(&mut cmd, term_cols);
    apply_title_seed_env(&mut cmd, session_id, title_seed);
    apply_session_history_env(&mut cmd, history_file);
    let out = cmd
        // The turn runs in the session's virtual cwd, so relative paths in
        // every tool resolve per session rather than against the daemon's.
        .current_dir(cwd)
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
        combined = truncate_turn_error(&combined);
        Err(std::io::Error::new(std::io::ErrorKind::Other, combined))
    }
}

#[cfg(feature = "serve")]
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
    // Hand the client's bytes through unchanged and use the trimmed view only
    // for the blank check: leading indentation and trailing whitespace are
    // part of the prompt (code blocks, markdown hard breaks).
    let prompt = req.prompt.clone();
    if prompt.trim().is_empty() {
        return bad_request("prompt is empty".to_string()).into_response();
    }
    if let Some(response) = write_roots::dirs_command_response(&state, &id, &prompt, false).await {
        return response;
    }
    let title_seed = turn_title_seed(&req);
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
    // Display width the client reports for this turn: the child lays its rows
    // out for the screen the client draws on, not for its own piped stdout.
    let term_cols = turn_term_cols(&req);
    let lock = session_lock(&state, &id).await;
    let guard = lock.lock().await;
    // A per-turn override wins; absent fields fall back to the session's
    // persisted serve config (then to the server default), so a turn spawned
    // without client flags still runs with the session's picks.
    let write_roots::TurnWriteContext {
        cwd: session_cwd,
        warnings,
    } = match write_roots::prepare_turn_write_context(&state, &id, &mut overrides) {
        Ok(context) => context,
        Err(error) => return bad_request(error).into_response(),
    };
    // Forced skills ride in the prompt, not argv: the child has no `--skill`
    // flag, but its input pipeline extracts `@skills:` tokens into forced
    // skills exactly like the local REPL.
    let prompt = apply_turn_skills_prefix(&prompt, &overrides.skills);
    let title_model = overrides.model.clone();
    // Kick off the model title request in parallel with the turn child: the
    // title task never takes the session lock, and the pending prompt seeds it
    // before the child persists its first message.
    spawn_session_title_task(
        state.history_file.clone(),
        id.clone(),
        title_model.clone(),
        title_seed.clone(),
    );
    let session_id = id.clone();
    let child_title_seed = title_seed.clone();
    let child_history_file = state.history_file.clone();
    let output = tokio::task::spawn_blocking(move || {
        run_one_shot_turn(
            &session_id,
            &child_history_file,
            &prompt,
            &overrides,
            &session_cwd,
            term_cols,
            &child_title_seed,
        )
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
    let output = if warnings.is_empty() {
        output
    } else {
        format!("{}\n{output}", warnings.join("\n"))
    };
    // The child is gone: retry the title request outside the lock it held for
    // its whole run, in case the parallel attempt above failed or raced an
    // empty history. No-ops when a title already settled.
    drop(guard);
    spawn_session_title_task(
        state.history_file.clone(),
        id.clone(),
        title_model,
        title_seed,
    );
    (
        StatusCode::OK,
        Json(TurnResp {
            session_id: id,
            output,
        }),
    )
        .into_response()
}

/// Request body for `POST /sessions/{id}/side-notes`: guidance text and optional
/// images for the running turn, the remote counterpart of the REPL's Ctrl+G
/// side-note. Uploads are captured as immutable attachments before their paths
/// enter the session's queue. The turn child consumes guidance before its next
/// model request and acknowledges it only after a durable history commit.
#[derive(Debug, Deserialize)]
#[cfg(feature = "serve")]
struct SideNoteReq {
    #[serde(default)]
    text: String,
    #[serde(default)]
    images: Vec<ServeImageUpload>,
}

/// `POST /sessions/{id}/side-notes` (authed): queue one side-note for this
/// session's in-flight turn. Returns `{"queued": true}` even when no turn is
/// running; the note then waits in the queue for the next turn.
#[cfg(feature = "serve")]
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
    if req.text.trim().is_empty() && req.images.is_empty() {
        return bad_request("side-note text is empty".to_string()).into_response();
    }
    // No session-file existence check: the queue file is independent of the
    // message history, and a note may arrive before the turn child persists
    // its first message. Queue appends and the drain-side atomic rename
    // compose without loss either way.
    let store = SessionStore::new(state.history_file.as_path());
    let session_file = store.session_history_file(&id);
    let mut text = req.text.trim().to_string();
    if !req.images.is_empty() {
        // Each note owns its uploads. Reusing a client filename must not replace
        // an image already queued or used by the running turn.
        let images_dir = store
            .session_assets_dir(&id)
            .join("attachments")
            .join(uuid::Uuid::new_v4().to_string());
        let images = match stage_side_note_images(&images_dir, &req.images) {
            Ok(images) => images,
            Err(err) => return bad_request(err).into_response(),
        };
        for (name, path) in images {
            let captured = format!("[[image:{}]]", path.display());
            // Bind one bare marker per upload, including whitespace accepted
            // by filename validation. Equal names consume markers in order.
            let marker_range = text.match_indices("[[image:").find_map(|(start, _)| {
                let value_start = start + "[[image:".len();
                let end = value_start + text[value_start..].find("]]")?;
                (text[value_start..end].trim() == name).then_some(start..end + "]]".len())
            });
            if let Some(range) = marker_range {
                text.replace_range(range, &captured);
            } else {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&captured);
            }
        }
    }
    if let Err(err) = push_side_note(&session_file, &text, "user", None) {
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

/// `GET /sessions/{id}/side-notes` (authed): the pending queue in append
/// order. A missing queue file is an empty list; notes disappear only after
/// the turn child commits them to canonical input, so this is the set a UI
/// should present as waiting to send.
#[cfg(feature = "serve")]
async fn get_side_notes(
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
    let session_file = store.session_history_file(&id);
    match list_pending_side_notes(&session_file, None) {
        Ok(notes) => (
            StatusCode::OK,
            Json(serde_json::json!({"session_id": id, "notes": notes})),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

#[cfg(feature = "serve")]
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
    // The Android shell marks its requests; only those turns get the native
    // phone tool group, so web/CLI clients never see phone-only tools.
    let native_client = headers
        .get("x-aserve-native")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == "1");
    // Same byte-exact handoff as `post_turn`.
    let prompt = req.prompt.clone();
    if prompt.trim().is_empty() {
        return bad_request("prompt is empty".to_string()).into_response();
    }
    if let Some(response) = write_roots::dirs_command_response(&state, &id, &prompt, true).await {
        return response;
    }
    let title_seed = turn_title_seed(&req);
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
    // Display width the client reports for this turn: the child lays its rows
    // out for the screen the client draws on, not for its own piped stdout.
    let term_cols = turn_term_cols(&req);
    // Same `@skills:` prompt splice as the one-shot route above: the stream
    // child parses it through the local inline-reference path.
    let prompt = apply_turn_skills_prefix(&prompt, &overrides.skills);
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
    let (client_tx, rx) = mpsc::channel::<Result<ServeSse, std::convert::Infallible>>(128);
    // Publish every event to the session log before forwarding it to the
    // originating client, and keep draining after that client goes away so a
    // reconnecting page can replay what it missed. The log is bound only once
    // the per-session lock is held: a queued second turn must not replace the
    // running turn's log while that turn still owns the session.
    let bus = Arc::clone(&state.events);
    let bus_session = id.clone();
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
    // The turn child runs in the session's virtual cwd: every relative path
    // in its tools then resolves per session, not against the daemon's cwd.
    tokio::task::spawn(async move {
        let guard = lock.lock().await;
        // Read defaults only after the previous writer has finished, including
        // config patches queued behind a running turn.
        let write_roots::TurnWriteContext {
            cwd: session_cwd,
            warnings,
        } = match write_roots::prepare_turn_write_context(&state, &title_session_id, &mut overrides)
        {
            Ok(context) => context,
            Err(error) => {
                let sink = TurnEventSink::new(bus.begin(&bus_session), client_tx);
                for event in [error_event(error), done_event()] {
                    sink.send(event);
                }
                return;
            }
        };
        let title_model = overrides.model.clone();
        // The title seed stays separate from the skill-prefixed child input.
        spawn_session_title_task(
            history_file.clone(),
            title_session_id.clone(),
            title_model.clone(),
            title_seed.clone(),
        );
        let tx = TurnEventSink::new(bus.begin(&bus_session), client_tx);
        let turn_metadata =
            history::ServeTurnMetadata::new(SessionStore::new(&history_file), bus_session.clone());
        for warning in warnings {
            tx.send(ServeSse::new("", warning));
        }
        let child_title_seed = title_seed.clone();
        let child_history_file = history_file.clone();
        let _ = tokio::task::spawn_blocking(move || {
            stream_child_turn(
                session_id,
                child_history_file,
                prompt,
                overrides,
                tx,
                live_fifo,
                active_turns,
                confirms,
                confirm_enabled,
                native_client,
                session_cwd,
                term_cols,
                child_title_seed,
                turn_metadata,
            )
        })
        .await;
        // The child is gone: retry the title request outside the lock it held
        // for its whole run, in case the parallel attempt failed. No-ops when
        // a title already settled.
        drop(guard);
        spawn_session_title_task(history_file, title_session_id, title_model, title_seed);
    });
    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (event.map(ServeSse::into_event), rx))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Maximum post-reap drain time, even if descendants keep writing to a pipe.
#[cfg(feature = "serve")]
const TURN_PIPE_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Publish independently of client consumption. Serialize publication and
/// enqueueing so the live stream has the same order as the retained log.
#[cfg(feature = "serve")]
#[derive(Clone)]
struct TurnEventSink(Arc<TurnEventSinkInner>);

#[cfg(feature = "serve")]
struct TurnEventSinkInner {
    log: Arc<SessionEventLog>,
    client: std::sync::Mutex<Option<mpsc::Sender<Result<ServeSse, std::convert::Infallible>>>>,
}

#[cfg(feature = "serve")]
impl TurnEventSink {
    fn new(
        log: Arc<SessionEventLog>,
        client: mpsc::Sender<Result<ServeSse, std::convert::Infallible>>,
    ) -> Self {
        Self(Arc::new(TurnEventSinkInner {
            log,
            client: std::sync::Mutex::new(Some(client)),
        }))
    }

    fn send(&self, event: ServeSse) {
        let mut client = self.0.client.lock().unwrap_or_else(|err| err.into_inner());
        self.0.log.publish(&event.kind, &event.data);
        if let Some(sender) = client.as_ref() {
            if sender.try_send(Ok(event)).is_err() {
                // Once full or closed, detach permanently. Letting a slow
                // receiver repeatedly free one slot would reintroduce
                // backpressure. Reconnects use the bounded session log.
                *client = None;
            }
        }
    }
}

#[cfg(feature = "serve")]
impl Drop for TurnEventSinkInner {
    fn drop(&mut self) {
        // All pipe workers have joined before the last producer is dropped.
        // No forwarding task or network receiver can delay log completion.
        self.log.finish();
    }
}

/// Blocking child-turn pump: runs the one-shot turn (same binary, same flags
/// as `run_one_shot_turn`) and forwards output as SSE events while the child
/// is still running: live-FIFO frames become `delta` / `thinking*` events
/// (one per arrival, chunk granularity), stdout lines become message events.
/// An output-complete frame sends `done` immediately while draining continues
/// until the child is reaped. Both pipes are drained on side threads and the
/// child is reaped independently, so neither a chatty child nor a tool-spawned
/// descendant that inherited a pipe can hold the turn open.
#[cfg(feature = "serve")]
fn stream_child_turn(
    session_id: String,
    history_file: PathBuf,
    prompt: String,
    overrides: TurnOverrides,
    tx: TurnEventSink,
    mut live_fifo: Option<LiveFifo>,
    active_turns: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    confirms: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
    confirm_enabled: bool,
    native_client: bool,
    session_cwd: std::path::PathBuf,
    term_cols: Option<u16>,
    title_seed: SessionTitleSeed,
    turn_metadata: history::ServeTurnMetadata,
) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let t0 = std::time::Instant::now();
    let timing = TurnSendTiming::default();
    let send = |event: ServeSse| {
        timing.stamp_first(t0);
        tx.send(event);
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
    push_turn_args(&mut cmd, &session_id, &prompt, &overrides);
    apply_title_seed_env(&mut cmd, &session_id, &title_seed);
    apply_session_history_env(&mut cmd, &history_file);
    // Run in the session's virtual cwd (see `serve/session_cwd.rs`), so
    // tools, MCP children and the prompt all see that session's directory.
    cmd.current_dir(&session_cwd);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Marks the child as serve-spawned even when no FIFO is set up
        // (fallback line pump): the title skip must not depend on streaming.
        .env(background::SERVE_CHILD_ENV, "1");
    if native_client {
        cmd.env(background::SERVE_NATIVE_ENV, "1");
    }
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
    // The child renders the rows this client displays; its own stdout is a
    // pipe, so the client's width travels through the environment.
    apply_term_cols_env(&mut cmd, term_cols);
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            send(error_event(format!(
                "cannot start turn process: {err}; if the daemon binary was replaced, restart it with `a --serve-restart`"
            )));
            send_done();
            return;
        }
    };
    // Registered for as long as the child runs, so an interrupt reaches
    // exactly this turn; the guard clears the entry after reaping and draining,
    // on every exit path below.
    let mut active_turn = Some(ActiveTurnGuard::register(
        &active_turns,
        &session_id,
        child.id(),
    ));
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
    // A tool can leave a background process holding the child's pipe write
    // ends, so stdout EOF is not a reliable "child finished" signal. Drain
    // both pipes on side threads and reap the child independently below; the
    // readers get a bounded grace to finish after the reap.
    let pipe_workers = TurnPipeWorkers::start(
        &mut child,
        &tx,
        &child_done,
        timing.clone(),
        t0,
        TURN_PIPE_DRAIN_GRACE,
    );
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
                    Some(turn_metadata),
                )
            })
        });
    #[cfg(not(unix))]
    let fifo_handle: Option<std::thread::JoinHandle<()>> = None;
    let (status, stderr_tail) = reap_and_drain_turn_child(&mut child, &mut active_turn, || {
        child_done.store(true, Ordering::SeqCst);
        // Nonblocking readers observe the reap signal even without a newline
        // or EOF. Join them: no detached worker may retain a producer or fd.
        let stderr_tail = pipe_workers.join();
        if let Some(handle) = fifo_handle {
            let _ = handle.join();
        }
        stderr_tail
    });
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

#[cfg(feature = "serve")]
fn reap_and_drain_turn_child<T>(
    child: &mut std::process::Child,
    active_turn: &mut Option<ActiveTurnGuard>,
    drain: impl FnOnce() -> T,
) -> (std::io::Result<std::process::ExitStatus>, T) {
    let status = child.wait();
    let drained = drain();
    drop(active_turn.take());
    (status, drained)
}

/// Pipe read ends are exclusively owned by their workers. Nonblocking reads
/// and bounded polls make the reap signal observable without EOF; a deadline
/// also bounds draining a descendant that continuously writes.
#[cfg(all(feature = "serve", unix))]
fn drain_turn_pipe(
    mut reader: impl std::io::Read + std::os::fd::AsRawFd,
    child_done: &std::sync::atomic::AtomicBool,
    grace: std::time::Duration,
    transient_eof: bool,
    mut consume: impl FnMut(&[u8]),
) -> std::io::Result<()> {
    let fd = reader.as_raw_fd();
    // SAFETY: the owned reader keeps fd valid for both fcntl calls. Only this
    // worker reads the descriptor; O_NONBLOCK does not affect its write end.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut deadline = None;
    let mut buf = [0u8; 8192];
    loop {
        let reaping = child_done.load(std::sync::atomic::Ordering::SeqCst);
        if reaping {
            let end = deadline.get_or_insert_with(|| std::time::Instant::now() + grace);
            if std::time::Instant::now() >= *end {
                return Ok(());
            }
        }
        match reader.read(&mut buf) {
            Ok(0) if !transient_eof || reaping => return Ok(()),
            Ok(0) => {
                // A FIFO writer opens lazily; a pipe EOF is never transient.
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            Ok(n) => {
                consume(&buf[..n]);
                continue;
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if reaping {
                    // All currently buffered bytes are drained. Do not wait
                    // for a descendant to close its retained write end.
                    return Ok(());
                }
            }
            Err(err) => return Err(err),
        }
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: polls one valid owned descriptor, with a bounded wait.
        let ready = unsafe { libc::poll(&mut pfd, 1, 200) };
        if ready < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                return Err(err);
            }
        }
    }
}

#[cfg(all(feature = "serve", unix))]
struct TurnPipeWorkers {
    stdout: Option<std::thread::JoinHandle<()>>,
    stderr: Option<std::thread::JoinHandle<String>>,
}

#[cfg(all(feature = "serve", unix))]
impl TurnPipeWorkers {
    fn start(
        child: &mut std::process::Child,
        tx: &TurnEventSink,
        child_done: &Arc<std::sync::atomic::AtomicBool>,
        timing: TurnSendTiming,
        t0: std::time::Instant,
        grace: std::time::Duration,
    ) -> Self {
        let stdout = child.stdout.take().map(|reader| {
            let tx = tx.clone();
            let done = Arc::clone(child_done);
            std::thread::spawn(move || {
                let emit = |line: &[u8]| {
                    timing.stamp_first(t0);
                    tx.send(ServeSse::new(
                        "",
                        clean_sse_line(&String::from_utf8_lossy(line)),
                    ));
                };
                let mut pending = Vec::new();
                let result = drain_turn_pipe(reader, &done, grace, false, |bytes| {
                    for part in bytes.split_inclusive(|byte| *byte == b'\n') {
                        pending.extend_from_slice(part);
                        if pending.last() == Some(&b'\n') {
                            pending.pop();
                            if pending.last() == Some(&b'\r') {
                                pending.pop();
                            }
                            emit(&pending);
                            pending.clear();
                        }
                    }
                });
                // A final unterminated line belongs to the exited turn too.
                if !pending.is_empty() {
                    emit(&pending);
                }
                if let Err(err) = result {
                    tx.send(error_event(format!("cannot read turn stdout: {err}")));
                }
            })
        });
        let stderr = child.stderr.take().map(|reader| {
            let done = Arc::clone(child_done);
            std::thread::spawn(move || {
                // Keep enough bytes for the final character-based error tail,
                // including a boundary codepoint split across read chunks.
                let limit = MAX_TURN_ERROR_CHARS * 4 + 4;
                let mut tail = Vec::new();
                let result = drain_turn_pipe(reader, &done, grace, false, |bytes| {
                    tail.extend_from_slice(bytes);
                    if tail.len() > limit {
                        tail.drain(..tail.len() - limit);
                    }
                });
                let mut text = String::from_utf8_lossy(&tail).into_owned();
                if let Err(err) = result {
                    eprintln!("[serve] cannot read turn stderr: {err}");
                    if text.is_empty() {
                        text = format!("cannot read turn stderr: {err}");
                    }
                }
                text
            })
        });
        Self { stdout, stderr }
    }

    fn join(self) -> String {
        if let Some(handle) = self.stdout {
            let _ = handle.join();
        }
        self.stderr
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default()
    }
}

/// Per-turn live-chunk FIFO: created by the SSE endpoint before spawning the
/// child, removed on drop (after the pump joins) whatever the outcome.
#[cfg(feature = "serve")]
struct LiveFifo {
    path: PathBuf,
    reader: Option<std::fs::File>,
}

#[cfg(feature = "serve")]
impl Drop for LiveFifo {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(feature = "serve")]
static LIVE_FIFO_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Best-effort per-turn FIFO for chunk streaming. The read end opens here
/// (before the child spawns) so the child's on-demand writer open always
/// finds a reader. `None` means "fall back to the line pump".
/// Callers must have validated `session_id` (its charset is filename-safe).
#[cfg(unix)]
#[cfg(feature = "serve")]
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
#[cfg(feature = "serve")]
fn setup_live_fifo(_session_id: &str) -> Option<LiveFifo> {
    None
}

/// Decoded serve-live events. Wire kinds are defined by `ServeLiveKind` in
/// `background.rs`.
#[derive(Debug, PartialEq, Eq)]
#[cfg(feature = "serve")]
enum ServeLiveEvent {
    Delta(String),
    ThinkingStart,
    Thinking(String, Option<background::ServeTokenRate>),
    ThinkingDone(background::ServeThinkingDone),
    OutputComplete,
    /// Raw JSON payload of a remote confirmation request.
    ConfirmRequest(String),
    /// Raw JSON payload closing a confirmation request.
    ConfirmDone(String),
    /// Latest compact status-line snapshot for foreground-owned subagents.
    SubagentProgress(String),
    /// Raw JSON payload of a free-text input request (same shape as a
    /// confirmation request: `{"id","prompt"}`).
    InputRequest(String),
    /// Raw JSON payload closing an input request.
    InputDone(String),
    /// Assistant prose is about to stream; the final body follows as `Replace`.
    AssistantBodyPending,
    /// Final accepted assistant body; replaces any provisional delta text.
    Replace(String),
    /// Bounded JSON describing an actual tool dispatch or its completed result.
    Tool(String),
    /// Exact producer-owned rate for a completed output/reasoning phase.
    TokenRate(background::ServeTokenRate),
    CompressionStart,
    CompressionDone,
    /// Diagnostic text for a model request that ended the turn's round.
    RequestError(String),
}

/// Incremental parser for serve-live frames (`[kind: u8][len: u32 BE][payload]`):
/// FIFO reads may split anywhere, so bytes accumulate until a whole frame is
/// present. Payloads are producer-guaranteed UTF-8; decoding is lossy only as
/// a defensive fallback. An unknown kind or absurd length resets the buffer
/// instead of desynchronizing the stream (impossible from our producer, only
/// from a foreign writer on the same inode).
///
/// A committed body too large for one atomic frame arrives as a
/// `ReplaceStart`/`ReplaceChunk`/`ReplaceEnd` sequence and is buffered until it
/// completes: a client applies every `replace` it receives to the whole body,
/// so emitting fragments would leave only the last one visible.
#[derive(Default)]
#[cfg(feature = "serve")]
struct ServeFrameDecoder {
    buf: Vec<u8>,
    thinking_rate: Option<background::ServeTokenRate>,
    thinking: Option<ReceivedThinking>,
    last_thinking_phase: Option<u64>,
    /// In-flight chunked committed body, present between `ReplaceStart` and
    /// `ReplaceEnd`.
    pending_replace: Option<String>,
    pending_json: Option<PendingServeJson>,
}

#[cfg(feature = "serve")]
struct ReceivedThinking {
    phase: Option<u64>,
    bytes: u64,
    frame_loss: bool,
}

#[cfg(feature = "serve")]
struct PendingServeJson {
    kind: ServeJsonKind,
    expected_bytes: usize,
    text: String,
}

#[cfg(feature = "serve")]
enum ServeJsonKind {
    Tool,
    SubagentProgress,
    RequestError,
}

#[cfg(feature = "serve")]
impl ServeFrameDecoder {
    fn start_thinking(&mut self, phase: Option<u64>, out: &mut Vec<ServeLiveEvent>) {
        if phase.is_some()
            && self
                .thinking
                .as_ref()
                .is_some_and(|active| active.phase == phase)
        {
            return;
        }
        self.close_thinking(background::ServeThinkingStatus::Interrupted, out);
        self.thinking_rate = None;
        self.thinking = Some(ReceivedThinking {
            phase,
            bytes: 0,
            frame_loss: false,
        });
        out.push(ServeLiveEvent::ThinkingStart);
    }

    fn close_thinking(
        &mut self,
        status: background::ServeThinkingStatus,
        out: &mut Vec<ServeLiveEvent>,
    ) {
        if let Some(active) = self.thinking.as_ref() {
            self.thinking_done(
                background::ServeThinkingDone {
                    status,
                    phase: active.phase,
                    ..Default::default()
                },
                out,
            );
        }
    }

    fn thinking_done(
        &mut self,
        mut done: background::ServeThinkingDone,
        out: &mut Vec<ServeLiveEvent>,
    ) {
        use background::ServeThinkingStatus;
        // A failed close is repeated in the next start/output-complete frame.
        // It must update one phase, never close a newer phase a second time.
        if done.phase.is_some()
            && done.phase == self.last_thinking_phase
            && self
                .thinking
                .as_ref()
                .is_none_or(|active| active.phase != done.phase)
        {
            return;
        }
        if self.thinking.as_ref().is_some_and(|active| {
            active.phase.is_some() && done.phase.is_some() && active.phase != done.phase
        }) {
            self.close_thinking(ServeThinkingStatus::Interrupted, out);
        }
        let active = self.thinking.take();
        let received_bytes = active.as_ref().map_or(0, |active| active.bytes);
        let frame_loss = done.dropped_frames > 0
            || active.as_ref().is_some_and(|active| active.frame_loss)
            || done.bytes.is_some_and(|bytes| bytes != received_bytes);
        if frame_loss
            && matches!(
                done.status,
                ServeThinkingStatus::Completed | ServeThinkingStatus::Interrupted
            )
        {
            done.status = ServeThinkingStatus::FrameLoss;
        }
        done.phase = done
            .phase
            .or_else(|| active.and_then(|active| active.phase));
        self.last_thinking_phase = done.phase.or(self.last_thinking_phase);
        self.thinking_rate = None;
        out.push(ServeLiveEvent::ThinkingDone(done));
    }

    fn append_json_chunk(&mut self, chunk: &str, complete: bool) -> Option<ServeLiveEvent> {
        let mut pending = self.pending_json.take()?;
        if pending.text.len() + chunk.len() > pending.expected_bytes {
            return None;
        }
        pending.text.push_str(chunk);
        if !complete {
            self.pending_json = Some(pending);
            return None;
        }
        if pending.text.len() != pending.expected_bytes {
            return None;
        }
        Some(match pending.kind {
            ServeJsonKind::Tool => ServeLiveEvent::Tool(pending.text),
            ServeJsonKind::SubagentProgress => ServeLiveEvent::SubagentProgress(pending.text),
            ServeJsonKind::RequestError => ServeLiveEvent::RequestError(pending.text),
        })
    }

    fn start_json(&mut self, payload: &str) {
        self.pending_json = None;
        let Some((header, text)) = payload.split_once('\n') else {
            return;
        };
        let Some((kind, bytes)) = header.split_once(' ') else {
            return;
        };
        let Ok(expected_bytes) = bytes.parse::<usize>() else {
            return;
        };
        if expected_bytes > background::MAX_SERVE_FRAME_LEN as usize || text.len() > expected_bytes
        {
            return;
        }
        let kind = match kind {
            "tool" => ServeJsonKind::Tool,
            "subagent_progress" => ServeJsonKind::SubagentProgress,
            "request_error" => ServeJsonKind::RequestError,
            _ => return,
        };
        self.pending_json = Some(PendingServeJson {
            kind,
            expected_bytes,
            text: text.to_string(),
        });
    }

    /// Append one continuation chunk of a chunked committed body. A chunk with
    /// no start in flight names no body and is dropped; a body past the
    /// defensive cap is abandoned (dropped, never emitted) so a foreign writer
    /// cannot grow it without bound.
    fn append_replace_chunk(&mut self, chunk: &str) {
        let Some(mut body) = self.pending_replace.take() else {
            return;
        };
        if body.len() + chunk.len() > background::MAX_SERVE_FRAME_LEN as usize {
            return;
        }
        body.push_str(chunk);
        self.pending_replace = Some(body);
    }

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
                || kind == background::ServeLiveKind::ThinkingChunk as u8
                || kind == background::ServeLiveKind::ThinkingRate as u8
                || kind == background::ServeLiveKind::ThinkingDone as u8
                || kind == background::ServeLiveKind::OutputComplete as u8
                || kind == background::ServeLiveKind::ConfirmRequest as u8
                || kind == background::ServeLiveKind::ConfirmDone as u8
                || kind == background::ServeLiveKind::SubagentProgress as u8
                || kind == background::ServeLiveKind::InputRequest as u8
                || kind == background::ServeLiveKind::InputDone as u8
                || kind == background::ServeLiveKind::AssistantBodyPending as u8
                || kind == background::ServeLiveKind::Replace as u8
                || kind == background::ServeLiveKind::ReplaceStart as u8
                || kind == background::ServeLiveKind::ReplaceChunk as u8
                || kind == background::ServeLiveKind::ReplaceEnd as u8
                || kind == background::ServeLiveKind::Tool as u8
                || kind == background::ServeLiveKind::CompressionStart as u8
                || kind == background::ServeLiveKind::CompressionDone as u8
                || kind == background::ServeLiveKind::RequestError as u8;
            let known = known
                || kind == background::ServeLiveKind::JsonStart as u8
                || kind == background::ServeLiveKind::JsonChunk as u8
                || kind == background::ServeLiveKind::JsonEnd as u8;
            let known = known || kind == background::ServeLiveKind::TokenRate as u8;
            if !known || len > background::MAX_SERVE_FRAME_LEN as usize {
                self.buf.clear();
                self.close_thinking(background::ServeThinkingStatus::FrameLoss, &mut out);
                break;
            }
            if self.buf.len() < 5 + len {
                break;
            }
            let payload = String::from_utf8_lossy(&self.buf[5..5 + len]).into_owned();
            self.buf.drain(..5 + len);
            let event = match kind {
                k if k == background::ServeLiveKind::JsonStart as u8 => {
                    self.start_json(&payload);
                    None
                }
                k if k == background::ServeLiveKind::JsonChunk as u8 => {
                    self.append_json_chunk(&payload, false)
                }
                k if k == background::ServeLiveKind::JsonEnd as u8 => {
                    self.append_json_chunk(&payload, true)
                }
                k if k == background::ServeLiveKind::Delta as u8 => {
                    Some(ServeLiveEvent::Delta(payload))
                }
                k if k == background::ServeLiveKind::ThinkingStart as u8 => {
                    let metadata = serde_json::from_str::<serde_json::Value>(&payload).ok();
                    if let Some(previous) =
                        metadata.as_ref().and_then(|value| value.get("previous"))
                    {
                        if let Ok(done) = serde_json::from_value(previous.clone()) {
                            self.thinking_done(done, &mut out);
                        }
                    }
                    let phase = metadata
                        .as_ref()
                        .and_then(|value| value.get("phase")?.as_u64());
                    self.start_thinking(phase, &mut out);
                    None
                }
                k if k == background::ServeLiveKind::Thinking as u8 => {
                    let active = self.thinking.get_or_insert(ReceivedThinking {
                        phase: None,
                        bytes: 0,
                        frame_loss: false,
                    });
                    active.bytes = active.bytes.saturating_add(payload.len() as u64);
                    Some(ServeLiveEvent::Thinking(
                        payload,
                        self.thinking_rate.clone(),
                    ))
                }
                k if k == background::ServeLiveKind::ThinkingChunk as u8 => {
                    let chunk = payload.split_once('\n').and_then(|(header, text)| {
                        let (phase, offset) = header.split_once(' ')?;
                        Some((
                            phase.parse::<u64>().ok()?,
                            offset.parse::<u64>().ok()?,
                            text,
                        ))
                    });
                    if let Some((phase, offset, text)) = chunk {
                        self.start_thinking(Some(phase), &mut out);
                        let active = self.thinking.as_mut().expect("thinking phase installed");
                        active.frame_loss |= active.bytes != offset;
                        active.bytes = offset.saturating_add(text.len() as u64);
                        Some(ServeLiveEvent::Thinking(
                            text.to_string(),
                            self.thinking_rate.clone(),
                        ))
                    } else {
                        self.thinking_done(
                            background::ServeThinkingDone {
                                status: background::ServeThinkingStatus::FrameLoss,
                                ..Default::default()
                            },
                            &mut out,
                        );
                        None
                    }
                }
                k if k == background::ServeLiveKind::ThinkingRate as u8 => {
                    self.thinking_rate = serde_json::from_str(&payload).ok();
                    None
                }
                k if k == background::ServeLiveKind::ThinkingDone as u8 => {
                    let done = if payload.is_empty() {
                        background::ServeThinkingDone::default()
                    } else {
                        serde_json::from_str(&payload).unwrap_or(background::ServeThinkingDone {
                            status: background::ServeThinkingStatus::Interrupted,
                            ..Default::default()
                        })
                    };
                    self.thinking_done(done, &mut out);
                    None
                }
                k if k == background::ServeLiveKind::ConfirmRequest as u8 => {
                    Some(ServeLiveEvent::ConfirmRequest(payload))
                }
                k if k == background::ServeLiveKind::ConfirmDone as u8 => {
                    Some(ServeLiveEvent::ConfirmDone(payload))
                }
                k if k == background::ServeLiveKind::InputRequest as u8 => {
                    Some(ServeLiveEvent::InputRequest(payload))
                }
                k if k == background::ServeLiveKind::InputDone as u8 => {
                    Some(ServeLiveEvent::InputDone(payload))
                }
                k if k == background::ServeLiveKind::SubagentProgress as u8 => {
                    Some(ServeLiveEvent::SubagentProgress(payload))
                }
                k if k == background::ServeLiveKind::OutputComplete as u8 => {
                    if let Ok(done) = serde_json::from_str(&payload) {
                        self.thinking_done(done, &mut out);
                    }
                    Some(ServeLiveEvent::OutputComplete)
                }
                k if k == background::ServeLiveKind::AssistantBodyPending as u8 => {
                    Some(ServeLiveEvent::AssistantBodyPending)
                }
                k if k == background::ServeLiveKind::Replace as u8 => {
                    Some(ServeLiveEvent::Replace(payload))
                }
                k if k == background::ServeLiveKind::ReplaceStart as u8 => {
                    // A new body supersedes an unfinished one.
                    self.pending_replace = Some(payload);
                    None
                }
                k if k == background::ServeLiveKind::ReplaceChunk as u8 => {
                    self.append_replace_chunk(&payload);
                    None
                }
                k if k == background::ServeLiveKind::ReplaceEnd as u8 => {
                    match self.pending_replace.take() {
                        Some(mut body) => {
                            body.push_str(&payload);
                            Some(ServeLiveEvent::Replace(body))
                        }
                        // A tail whose start never arrived (a dropped frame, or
                        // an abandoned oversized body) emits nothing: a fragment
                        // must never become the authoritative body, so the client
                        // keeps whatever text it already streamed.
                        None => None,
                    }
                }
                k if k == background::ServeLiveKind::Tool as u8 => {
                    Some(ServeLiveEvent::Tool(payload))
                }
                k if k == background::ServeLiveKind::RequestError as u8 => {
                    Some(ServeLiveEvent::RequestError(payload))
                }
                k if k == background::ServeLiveKind::CompressionStart as u8 => {
                    Some(ServeLiveEvent::CompressionStart)
                }
                k if k == background::ServeLiveKind::CompressionDone as u8 => {
                    Some(ServeLiveEvent::CompressionDone)
                }
                k if k == background::ServeLiveKind::TokenRate as u8 => {
                    serde_json::from_str(&payload)
                        .ok()
                        .map(ServeLiveEvent::TokenRate)
                }
                // Every kind accepted by the `known` gate above must have an
                // arm here: falling through would end the turn for a frame
                // that is not an output-complete marker.
                _ => Some(ServeLiveEvent::OutputComplete),
            };
            if let Some(event) = event {
                if matches!(
                    event,
                    ServeLiveEvent::Delta(_)
                        | ServeLiveEvent::Replace(_)
                        | ServeLiveEvent::Tool(_)
                        | ServeLiveEvent::OutputComplete
                ) {
                    self.close_thinking(background::ServeThinkingStatus::Interrupted, &mut out);
                }
                out.push(event);
            }
        }
        out
    }

    /// Drain on EOF: parse whatever complete frames remain; a lone partial
    /// frame is a truncated write and is dropped.
    fn finish(&mut self) -> Vec<ServeLiveEvent> {
        let tail = std::mem::take(&mut self.buf);
        let mut events = self.push(&tail);
        self.buf.clear();
        self.pending_json = None;
        self.close_thinking(background::ServeThinkingStatus::Interrupted, &mut events);
        events
    }
}

/// One SSE event before it is turned into an axum `Event`. Keeping the parts
/// explicit lets the session log retain them for reconnects; axum's `Event`
/// exposes builders only, not getters.
#[cfg(feature = "serve")]
#[derive(Debug, Clone)]
struct ServeSse {
    kind: String,
    data: String,
}

#[cfg(feature = "serve")]
impl ServeSse {
    fn new(kind: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            data: data.into(),
        }
    }

    fn into_event(self) -> Event {
        let mut event = Event::default();
        if !self.kind.is_empty() {
            event = event.event(&self.kind);
        }
        event.data(self.data)
    }
}

/// Map one parsed frame to its SSE event. `OutputComplete` has no event here:
/// the caller ends the turn with `done` instead (see `forward_live_event`).
#[cfg(feature = "serve")]
fn live_event(event: ServeLiveEvent) -> Option<ServeSse> {
    match event {
        ServeLiveEvent::Delta(text) => Some(delta_event(text)),
        ServeLiveEvent::ThinkingStart => Some(ServeSse::new("thinking_start", "")),
        ServeLiveEvent::Thinking(text, rate) => {
            let mut payload = serde_json::json!({"text": text});
            if let Some(rate) = rate {
                payload["rate"] = serde_json::to_value(rate).ok()?;
            }
            Some(ServeSse::new("thinking", payload.to_string()))
        }
        ServeLiveEvent::ThinkingDone(done) => Some(ServeSse::new(
            "thinking_done",
            serde_json::to_string(&done).ok()?,
        )),
        ServeLiveEvent::AssistantBodyPending => Some(ServeSse::new("answer_pending", "")),
        ServeLiveEvent::Replace(text) => Some(ServeSse::new(
            "replace",
            serde_json::json!({"text": text}).to_string(),
        )),
        // Confirmation and input frames are built by `forward_live_event`,
        // which holds the pending state and adds the answer token; they never
        // get here.
        ServeLiveEvent::ConfirmRequest(_)
        | ServeLiveEvent::ConfirmDone(_)
        | ServeLiveEvent::InputRequest(_)
        | ServeLiveEvent::InputDone(_) => None,
        ServeLiveEvent::SubagentProgress(line) => Some(ServeSse::new("subagent_progress", line)),
        ServeLiveEvent::Tool(payload) => Some(ServeSse::new("tool", payload)),
        ServeLiveEvent::TokenRate(rate) => {
            Some(ServeSse::new("rate", serde_json::to_string(&rate).ok()?))
        }
        ServeLiveEvent::CompressionStart => Some(ServeSse::new("compression_start", "")),
        ServeLiveEvent::CompressionDone => Some(ServeSse::new("compression_done", "")),
        ServeLiveEvent::RequestError(detail) => Some(ServeSse::new(
            "request_error",
            serde_json::json!({"error": detail}).to_string(),
        )),
        ServeLiveEvent::OutputComplete => None,
    }
}

/// Forward one parsed frame. An `OutputComplete` frame ends the user-visible
/// turn immediately (the child only runs silent bookkeeping after that
/// point); the first frame wins and later ones are no-ops, so the terminal
/// `done` below is never duplicated.
#[cfg(feature = "serve")]
fn forward_live_event(
    event: ServeLiveEvent,
    turn_done: &std::sync::atomic::AtomicBool,
    tx: &TurnEventSink,
    timing: &TurnSendTiming,
    t0: std::time::Instant,
    confirm: Option<&std::sync::Mutex<ConfirmSlot>>,
    metadata: Option<&mut history::ServeTurnMetadata>,
) {
    timing.stamp_first(t0);
    if matches!(event, ServeLiveEvent::OutputComplete) {
        if !turn_done.swap(true, std::sync::atomic::Ordering::SeqCst) {
            timing.stamp_done(t0);
            let done = metadata.map_or_else(done_event, |metadata| metadata.done_event());
            tx.send(done);
        }
        return;
    }
    // Confirmation frames carry live state (the pending question) as well as a
    // client event, so they are built here; every other kind maps 1:1. The
    // slot is filled before the send: an answer can only arrive after a client
    // saw the event, and a client that lost the stream instead of receiving it
    // reads the slot through `GET .../confirm`.
    let mut sse = match event {
        ServeLiveEvent::ConfirmRequest(payload) => {
            let Some((id, prompt)) = parse_confirm_request(&payload) else {
                // A malformed payload cannot identify a question; drop it.
                return;
            };
            let token = NEXT_CONFIRM_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(slot) = confirm {
                if let Ok(mut slot) = slot.lock() {
                    slot.pending = Some(PendingConfirm {
                        id,
                        prompt: prompt.clone(),
                        kind: ConfirmKind::Confirm,
                        token,
                    });
                }
            }
            ServeSse::new(
                "confirm_request",
                serde_json::json!({"id": id, "prompt": prompt, "token": token}).to_string(),
            )
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
            ServeSse::new("confirm_done", payload)
        }
        ServeLiveEvent::InputRequest(payload) => {
            let Some((id, prompt)) = parse_confirm_request(&payload) else {
                return;
            };
            let token = NEXT_CONFIRM_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(slot) = confirm {
                if let Ok(mut slot) = slot.lock() {
                    slot.pending = Some(PendingConfirm {
                        id,
                        prompt: prompt.clone(),
                        kind: ConfirmKind::Input,
                        token,
                    });
                }
            }
            ServeSse::new(
                "input_request",
                serde_json::json!({"id": id, "prompt": prompt, "token": token}).to_string(),
            )
        }
        ServeLiveEvent::InputDone(payload) => {
            if let Some(id) = parse_confirm_id(&payload) {
                if let Some(slot) = confirm {
                    if let Ok(mut slot) = slot.lock() {
                        if slot.pending.as_ref().is_some_and(|p| p.id == id) {
                            slot.pending = None;
                        }
                    }
                }
            }
            ServeSse::new("input_done", payload)
        }
        other => match live_event(other) {
            Some(e) => e,
            None => return,
        },
    };
    if let Some(metadata) = metadata {
        if sse.kind == "replace" {
            metadata.record_committed_answer(&sse.data);
        }
        metadata.decorate_input_event(&mut sse);
    }
    tx.send(sse);
}

/// Forward live-FIFO frames as SSE events until the child is reaped and the
/// pipe is drained. Mirrors the `attach_live_session` reader: an EOF while
/// the child is still alive is transient (its writer opens on demand at the
/// first chunk), so it sleeps instead of spinning; after the child exits EOF
/// is terminal. Continuously writing descendants are bounded by the same
/// post-reap deadline as the stdout/stderr readers.
#[cfg(unix)]
#[cfg(feature = "serve")]
fn pump_live_fifo(
    reader: std::fs::File,
    child_done: &std::sync::atomic::AtomicBool,
    turn_done: &std::sync::atomic::AtomicBool,
    tx: &TurnEventSink,
    timing: TurnSendTiming,
    t0: std::time::Instant,
    confirm: Option<Arc<std::sync::Mutex<ConfirmSlot>>>,
    mut metadata: Option<history::ServeTurnMetadata>,
) {
    let mut decoder = ServeFrameDecoder::default();
    let result = drain_turn_pipe(reader, child_done, TURN_PIPE_DRAIN_GRACE, true, |bytes| {
        for event in decoder.push(bytes) {
            forward_live_event(
                event,
                turn_done,
                tx,
                &timing,
                t0,
                confirm.as_deref(),
                metadata.as_mut(),
            );
        }
    });
    if let Err(err) = result {
        eprintln!("[serve] cannot read turn FIFO: {err}");
    }
    // EOF and read/poll failures must both settle a partial reasoning phase.
    for event in decoder.finish() {
        forward_live_event(
            event,
            turn_done,
            tx,
            &timing,
            t0,
            confirm.as_deref(),
            metadata.as_mut(),
        );
    }
}

#[cfg(feature = "serve")]
fn delta_event(delta: String) -> ServeSse {
    ServeSse::new("delta", serde_json::json!({"delta": delta}).to_string())
}

#[cfg(feature = "serve")]
fn truncate_line(line: &str) -> String {
    if line.chars().count() <= MAX_SSE_LINE_CHARS {
        line.to_string()
    } else {
        let kept: String = line.chars().take(MAX_SSE_LINE_CHARS).collect();
        format!("{kept}...[line truncated]")
    }
}

/// Normalize one child-stdout row for SSE transport: keep the text a terminal
/// would end up showing for that row, then neutralize cursor-addressing
/// escapes. SSE rows are append-only, so a `\r` cannot rewind the client's
/// cursor; worse, axum re-prefixes `data: ` after every `\r`/`\n` byte while
/// the chat client only strips the first one, so any surviving `\r` renders as
/// a literal `data: ` fragment on the client's screen. `BufRead::lines` glues
/// redraw bytes into the row (completed/failed tool rows carry
/// `\r\x1b[2K` to redraw the `running` row in place, fold teardown emits
/// doubled `\r`s with no `\n` between), so the `\r` is routinely mid-line,
/// not just a leading prefix. `BufRead::lines` already removed the line
/// ending, so every `\r` left is an overwrite control, never content: only
/// the text after the last one is what the row shows, and replaying the
/// replaced text would print a stale running row right above its own
/// completion (`● plan_update` and `✓ plan_update` together — a pair no local
/// terminal ever shows). Cursor moves (`CSI A/B`), region erases (`CSI J/K`)
/// and friends address screen rows that only exist on the child's live
/// terminal: replayed verbatim on the chat client's screen they erase or
/// overwrite unrelated rows (ragged indents, duplicated-looking status
/// lines). SGR color (`CSI ... m`) is kept: it styles only the row itself and
/// renders identically everywhere.
#[cfg(feature = "serve")]
fn clean_sse_line(line: &str) -> String {
    // Collapse before escape stripping so an erase-only chunk (`\r\x1b[2K`)
    // still counts as the row's newest content: it clears the row instead of
    // leaving the text it replaced visible.
    truncate_line(&strip_non_sgr_csi(collapse_row_overwrites(line)))
}

/// Collapse same-row overwrite controls (`\r` redraws) into the one row of
/// text a terminal shows. Live rows are rewritten in place — `  ● tool`
/// becomes `\r\x1b[2K  ✓ tool` when the call returns, and the fold / metrics
/// rows reuse the pattern — so the newest chunk that draws on the row
/// replaces every chunk before it. An empty chunk (a bare `\r` with nothing
/// after it) only re-homes the cursor and leaves the row's text alone,
/// exactly like on a terminal.
#[cfg(feature = "serve")]
fn collapse_row_overwrites(line: &str) -> &str {
    line.rsplit('\r')
        .find(|chunk| !chunk.is_empty())
        .unwrap_or("")
}

/// Remove every ANSI CSI sequence except SGR color (`ESC [ ... m`).
/// Malformed (unterminated) introducers are dropped: a half-sequence can
/// never render as intended on the client.
#[cfg(feature = "serve")]
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

/// Trim a failed turn's combined output to the display cap.
///
/// The text carries the child's stderr, which can contain multi-byte
/// characters: a byte offset may land inside one and panic while reporting
/// the failure.
#[cfg(feature = "serve")]
fn truncate_turn_error(combined: &str) -> String {
    if combined.len() > MAX_TURN_ERROR_CHARS {
        last_chars(combined, MAX_TURN_ERROR_CHARS)
    } else {
        combined.to_string()
    }
}

#[cfg(feature = "serve")]
fn last_chars(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        text.to_string()
    } else {
        let kept: String = chars[chars.len() - limit..].iter().collect();
        format!("...[truncated]...{kept}")
    }
}

#[cfg(feature = "serve")]
fn error_event(detail: String) -> ServeSse {
    ServeSse::new("error", serde_json::json!({"error": detail}).to_string())
}

#[cfg(feature = "serve")]
fn done_event() -> ServeSse {
    ServeSse::new("done", "")
}

#[cfg(feature = "serve")]
async fn list_skills(State(state): State<ServeState>, headers: HeaderMap) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let manifests = super::skills::load_all_skills();
    (StatusCode::OK, Json(manifests)).into_response()
}

#[cfg(feature = "serve")]
async fn list_agents(State(state): State<ServeState>, headers: HeaderMap) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    let manifests = super::agents::load_all_agents();
    (StatusCode::OK, Json(manifests)).into_response()
}

/// Cap for a `/cd` command line and a `/cd-candidates` fragment. Long paths are
/// legitimate; unbounded input is not.
const MAX_CD_COMMAND_CHARS: usize = 4096;

/// Request body for `POST /sessions/{id}/cd`: the full `/cd` (or `:cd`)
/// command line as typed, executed verbatim by [`post_session_cd`].
#[derive(Debug, Deserialize)]
#[cfg(feature = "serve")]
struct CdCommandRequest {
    command: String,
}

/// Query for `GET /sessions/{id}/cd-candidates`: the fragment after `/cd`, as-is.
#[derive(Debug, Deserialize)]
#[cfg(feature = "serve")]
struct CdCandidatesQuery {
    frag: String,
}

/// Switch a session's virtual working directory (`POST /sessions/{id}/cd`).
///
/// The daemon hosts many sessions in one process, so the switch is recorded
/// per session in `ServeState.cwds` (see `serve/session_cwd.rs`) instead of
/// moving the shared process cwd; every turn child is spawned with
/// `.current_dir()` set to its session's entry. The captured message lines
/// come back over HTTP instead of printing into the server's terminal.
#[cfg(feature = "serve")]
async fn post_session_cd(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<CdCommandRequest>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    if req.command.trim().is_empty() || req.command.len() > MAX_CD_COMMAND_CHARS {
        return bad_request("empty or oversized cd command".to_string()).into_response();
    }
    match run_session_cd(&state, &id, &req.command) {
        Some(outcome) => {
            // `cwd` lets the web UI update the session bar immediately; the
            // printed `stdout` is the same value for clients that shell out.
            Json(serde_json::json!({
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "cwd": session_cwd(&state, &id).display().to_string(),
            }))
            .into_response()
        }
        None => bad_request("not a /cd command".to_string()).into_response(),
    }
}

/// Directory listing for serve-chat `/cd` completion
/// (`GET /sessions/{id}/cd-candidates`).
///
/// Reuses the local REPL's `/cd` completion logic against the session's
/// virtual cwd on this server's own filesystem — the server's disk *is* the
/// remote filesystem — so the client completes remote paths (directories
/// only, `dir/`-suffixed, nav entries and hidden toggle included) for the
/// session it is chatting with, not for the daemon's own cwd.
#[cfg(feature = "serve")]
async fn get_session_cd_candidates(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<CdCandidatesQuery>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    if q.frag.len() > MAX_CD_COMMAND_CHARS {
        return bad_request("oversized cd fragment".to_string()).into_response();
    }
    let base = session_cwd(&state, &id);
    let candidates = super::prompt::completion::cd_fragment_candidates_from(&q.frag, &base);
    Json(serde_json::json!({ "candidates": candidates })).into_response()
}

#[cfg(feature = "serve")]
fn is_loopback_bind(bind: &str) -> bool {
    // Parse the host part instead of prefix-matching so that names like
    // "127.x.evil.com" cannot pass the loopback gate.
    let host = bind.rsplit_once(':').map(|(h, _)| h).unwrap_or(bind).trim();
    let host = host
        .trim_matches(|c| c == '[' || c == ']')
        .to_ascii_lowercase();
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

#[cfg(feature = "serve")]
pub(in crate::ai) fn run_serve_tls_init(
    cli: &super::cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    if cli.serve
        || cli.serve_chat
        || cli.serve_sessions
        || cli.serve_start
        || cli.serve_stop
        || cli.serve_restart
        || cli.serve_status
    {
        return Err("--serve-tls-init must be run by itself (optionally with --serve-bind)".into());
    }
    let cfg = configw::get_all_config();
    let configured = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
    let bind = resolve_serve_bind(&cli.serve_bind, &configured);
    let result = tls::init_serve_tls(&bind)?;
    println!("[serve] TLS {:?}: {}", result.status, result.tls.cert);
    println!("[serve] private key: {}", result.tls.key);
    println!("[serve] CA certificate: {}", result.ca_cert.display());
    println!("[serve] CA SHA-256: {}", result.fingerprint);
    println!(
        "[serve] certificate names: {}",
        result.subject_alt_names.join(", ")
    );
    println!(
        "[serve] future `a --serve`/`a --serve-start` runs will use this managed certificate automatically"
    );
    Ok(())
}

#[cfg(feature = "serve")]
fn resolve_serve_history_file(
    base_history_file: &std::path::Path,
    persona_id: &str,
) -> std::io::Result<PathBuf> {
    // Anchor the root before any turn child changes to its session cwd.
    let absolute = if base_history_file.is_absolute() {
        base_history_file.to_path_buf()
    } else {
        std::env::current_dir()?.join(base_history_file)
    };
    Ok(super::persona::session_client_history_file(
        &absolute, persona_id, None,
    ))
}

#[cfg(feature = "serve")]
pub(in crate::ai) async fn run_serve(
    cli: super::cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the same config resolution as the CLI so serve sees the same
    // history file, models, and sandbox settings.
    let app_config = super::config::load_config()?;
    let persona = super::persona::PersonaStore::new()
        .active_persona()
        .unwrap_or_else(|_| super::persona::default_persona());
    let history_file = resolve_serve_history_file(&app_config.base_history_file, &persona.id)?;
    let cfg = configw::get_all_config();
    let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
    let bind = resolve_serve_bind(&cli.serve_bind, &from_cfg);
    let managed_tls = tls::default_serve_tls_config()?;
    let tls = tls::resolve_serve_tls(
        &cli.serve_tls_cert,
        &cli.serve_tls_key,
        managed_tls.as_ref(),
    )?;
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    let default_add_dirs = ctl::runtime_default_add_dirs(&cli)?;
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
    // Restore per-session `/cd` directories recorded by a previous daemon
    // process: without this a `--serve-restart` silently reverts every session
    // (and the subagents inheriting from its turns) to the daemon cwd.
    let restored_cwds = load_cwds(&history_file);
    let restored_groups = load_groups(&history_file);
    eprintln!(
        "[serve] restored {} persisted session working directories",
        restored_cwds.len()
    );
    let state = ServeState {
        history_file,
        token,
        default_add_dirs: Arc::new(default_add_dirs.clone()),
        cwds: Arc::new(std::sync::Mutex::new(restored_cwds)),
        groups: Arc::new(std::sync::Mutex::new(restored_groups)),
        // One entry per touched session; bounded by session count in practice.
        locks: Arc::new(Mutex::new(HashMap::new())),
        open_sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
        active_turns: Arc::new(std::sync::Mutex::new(HashMap::new())),
        confirms: Arc::new(std::sync::Mutex::new(HashMap::new())),
        events: Arc::new(SessionEventBus::default()),
    };
    // Browser open claims are renewed by the page; a tab that vanished without
    // releasing must not pin its session forever.
    let reaper_sessions = Arc::clone(&state.open_sessions);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            let mut sessions = reaper_sessions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let now = std::time::Instant::now();
            sessions.retain(|_, entry| entry.expires_at > now);
        }
    });
    let apk_download_route = format!("/{}", android_apk_filename());
    let app = APP_ENTRY_PATHS
        .into_iter()
        .fold(Router::new(), |router, path| {
            router.route(path, get(serve_app))
        })
        .route("/app/{*path}", any(redirect_legacy_app_api))
        .route("/healthz", get(healthz))
        .route("/app.webmanifest", get(serve_app_manifest))
        .route("/app-icon.svg", get(serve_app_icon))
        .route("/app-icon-192.png", get(serve_app_icon_192))
        .route("/app-icon-512.png", get(serve_app_icon_512))
        .route("/app-sw.js", get(serve_app_service_worker))
        .route("/a-serve.apk", get(serve_android_apk))
        .route("/a-serve-v1.1.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.2.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.3.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.4.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.5.2.apk", get(serve_android_apk))
        .route("/a-serve-v1.8.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.12.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.13.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.15.0.apk", get(serve_android_apk))
        .route("/a-serve-v1.15.1.apk", get(serve_android_apk))
        .route(&apk_download_route, get(serve_android_apk))
        .route("/info", get(server_info))
        .route("/subagent-models", get(subagent_models))
        .route("/sessions", get(list_sessions).post(create_session))
        .route("/session-group-activity", get(list_session_group_activity))
        .route(
            "/session-groups",
            get(list_session_groups).post(create_session_group),
        )
        .route(
            "/session-groups/{id}",
            axum::routing::patch(update_session_group).delete(delete_session_group),
        )
        .route(
            "/session-activity",
            post(list_session_activity)
                .layer(DefaultBodyLimit::max(MAX_SESSION_ACTIVITY_REQUEST_BYTES)),
        )
        .route(
            "/sessions/import",
            post(import_session).layer(DefaultBodyLimit::max(MAX_IMPORT_REQUEST_BYTES)),
        )
        .route("/sessions/{id}/fork", post(fork_session))
        .route("/sessions/{id}/title", post(set_session_title))
        .route("/sessions/{id}/mark", post(set_session_mark))
        .route("/sessions/{id}/archive", post(set_session_archive))
        .route("/sessions/{id}/group", post(set_session_group))
        .route(
            "/sessions/{id}/config",
            get(get_session_config).post(set_session_config),
        )
        .route("/sessions/{id}", delete(delete_session))
        .route("/sessions/{id}/history", get(read_history))
        .route("/sessions/{id}/export", get(export_session_zip))
        .route("/sessions/{id}/export-md", get(export_session_markdown))
        .route("/sessions/{id}/events", get(stream_session_events))
        .route("/sessions/{id}/open", post(open_session))
        .route("/sessions/{id}/heartbeat", post(heartbeat_session))
        .route("/sessions/{id}/release", post(release_session))
        .route("/sessions/{id}/rewind", post(rewind_history))
        .route(
            "/sessions/{id}/checkpoints",
            get(list_checkpoints).post(post_checkpoint_command),
        )
        .route(
            "/sessions/{id}/turns",
            post(post_turn).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route(
            "/sessions/{id}/turns/stream",
            post(post_turn_sse).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route("/sessions/{id}/interrupt", post(post_interrupt))
        .route(
            "/sessions/{id}/side-notes",
            get(get_side_notes)
                .post(post_side_note)
                .layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route("/sessions/{id}/file", get(get_session_file))
        .route(
            "/sessions/{id}/confirm",
            get(get_confirm).post(post_confirm),
        )
        .route("/skills", get(list_skills))
        .route("/agents", get(list_agents))
        .route("/commands", get(list_commands))
        .route("/sessions/{id}/cd", post(post_session_cd))
        .route(
            "/sessions/{id}/cd-candidates",
            get(get_session_cd_candidates),
        );
    #[cfg(unix)]
    let app = changes::routes(app);
    let app = app
        .with_state(state)
        .layer(axum::middleware::from_fn(compression::gzip));
    if let Some(tls) = tls.as_ref() {
        let server = tls::TlsServer::bind(&bind, tls).await?;
        // Record this process in the global serve state file so
        // `--serve-status` and `--serve-stop` see a foreground server too.
        ctl::note_foreground_serve(&bind, Some(tls), &default_add_dirs);
        eprintln!("[serve] listening on https://{bind}");
        server.serve(app).await?;
    } else {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        ctl::note_foreground_serve(&bind, None, &default_add_dirs);
        eprintln!("[serve] listening on http://{bind}");
        axum::serve(listener, app).await?;
    }
    // The listener shut down cleanly; drop our claim so a later status does
    // not report a stale pid. (A signal-killed server leaves a stale file
    // behind; status/start self-heal by reaping it.)
    ctl::clear_foreground_serve();
    Ok(())
}
