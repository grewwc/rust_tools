//! Serve mode: local HTTP API with parity to the CLI personal assistant.
//!
//! v1 scope: HTTP on loopback, Bearer auth (optional), session list/history,
//! one-shot turns via subprocess (same binary, `--session <id>`), skills/agents
//! listing. Turn execution reuses the existing one-shot path so behavior stays
//! identical to `a --session <id> "<prompt>"`.

use std::{collections::HashMap, path::PathBuf, process::Stdio, sync::Arc};

use base64::Engine as _;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, post},
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};

use crate::ai::background;
use crate::commonw::configw;

use super::{config_schema::AiConfig, history::SessionStore};

pub(in crate::ai) mod chat;

const DEFAULT_BIND: &str = "127.0.0.1:8080";

#[derive(Debug, Clone)]
struct ServeState {
    history_file: PathBuf,
    token: String,
    /// Per-session async mutex: at most one turn writer per session.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

#[derive(Debug, Serialize)]
struct Healthz {
    ok: bool,
    version: String,
}

#[derive(Debug, Serialize)]
struct SessionItem {
    id: String,
    size_bytes: u64,
    summary: Option<String>,
    first_user_prompt: Option<String>,
    marked: bool,
}

#[derive(Debug, Deserialize)]
struct SessionsQuery {
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct CreateSessionResp {
    id: String,
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TurnReq {
    prompt: String,
    /// Client-uploaded images for `[[image:name]]` placeholders in `prompt`.
    /// A remote serve-chat client shares no filesystem with the server, so
    /// pasted images travel inside the request (base64) and are staged under
    /// the same filenames before the turn runs. Absent for old clients and
    /// for text-only turns.
    #[serde(default)]
    images: Vec<ServeImageUpload>,
}

#[derive(Debug, Deserialize)]
struct ServeImageUpload {
    filename: String,
    data_base64: String,
}

/// Per-turn image upload caps: pasted screenshots are small, but the API
/// must not become an arbitrary file drop. The child resolves placeholders
/// against the session assets dir, so only bare filenames with an image
/// extension are accepted — never a path.
const MAX_TURN_IMAGES: usize = 10;
const MAX_TURN_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_TURN_IMAGES_TOTAL_BYTES: usize = 32 * 1024 * 1024;
const TURN_IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

/// Max JSON body for the two turn endpoints. Images travel base64 (~4/3
/// overhead) inside the request, so the framework default (2MB) would reject
/// legitimate multi-image turns with 413 before the handler caps run.
/// Handler-level caps (10MiB per file, 32MiB total decoded) still enforce the
/// real policy; this only lets those requests reach the handler.
const MAX_TURN_REQUEST_BYTES: usize = 48 * 1024 * 1024;

/// Stage uploaded turn images into the session assets dir under the uploaded
/// filenames, so the one-shot child's existing `[[image:name]]` resolution
/// finds them exactly as if they had been pasted locally. Pure filesystem
/// work; runs before the per-session lock is taken.
fn stage_turn_images(
    assets_dir: &std::path::Path,
    images: &[ServeImageUpload],
) -> Result<(), String> {
    if images.is_empty() {
        return Ok(());
    }
    if images.len() > MAX_TURN_IMAGES {
        return Err(format!(
            "too many images: {} (max {MAX_TURN_IMAGES})",
            images.len()
        ));
    }
    // Validate and decode everything before touching the filesystem: a late
    // rejection must not leave a directory or partially staged files behind.
    let mut staged: Vec<(&str, Vec<u8>)> = Vec::with_capacity(images.len());
    let mut total_bytes = 0usize;
    for image in images {
        let name = image.filename.trim();
        if name.is_empty()
            || name == "."
            || name == ".."
            || name.contains('/')
            || name.contains('\\')
        {
            return Err(format!("invalid image filename: {:?}", image.filename));
        }
        let ext = std::path::Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !TURN_IMAGE_EXTS.contains(&ext.as_str()) {
            return Err(format!("unsupported image type for {name:?}"));
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image.data_base64.trim())
            .map_err(|_| format!("image {name:?} is not valid base64"))?;
        if bytes.is_empty() || bytes.len() > MAX_TURN_IMAGE_BYTES {
            return Err(format!(
                "image {name:?} is {} bytes (max {MAX_TURN_IMAGE_BYTES})",
                bytes.len()
            ));
        }
        total_bytes += bytes.len();
        if total_bytes > MAX_TURN_IMAGES_TOTAL_BYTES {
            return Err(format!(
                "images exceed {MAX_TURN_IMAGES_TOTAL_BYTES} bytes in total"
            ));
        }
        staged.push((name, bytes));
    }
    std::fs::create_dir_all(assets_dir)
        .map_err(|err| format!("cannot prepare session assets dir: {err}"))?;
    for (name, bytes) in &staged {
        std::fs::write(assets_dir.join(name), bytes)
            .map_err(|err| format!("cannot stage image {name:?}: {err}"))?;
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct TurnResp {
    session_id: String,
    output: String,
}

fn unauthorized(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": msg})),
    )
}

fn bad_request(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg})),
    )
}

fn check_auth(state: &ServeState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if state.token.is_empty() {
        return Ok(());
    }
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let want = format!("Bearer {}", state.token);
    // Constant-time compare to avoid leaking prefix length via timing.
    let ok = got.len() == want.len()
        && got.bytes().zip(want.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0;
    if ok {
        Ok(())
    } else {
        Err(unauthorized("invalid bearer token"))
    }
}

async fn session_lock(state: &ServeState, session_id: &str) -> Arc<Mutex<()>> {
    let mut map = state.locks.lock().await;
    map.get(session_id)
        .cloned()
        .unwrap_or_else(|| {
            let lock = Arc::new(Mutex::new(()));
            map.insert(session_id.to_string(), lock.clone());
            lock
        })
}

async fn healthz() -> Json<Healthz> {
    Json(Healthz {
        ok: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
// Note: /healthz is intentionally public (liveness only, no session data).
// All session/skill/agent routes enforce check_auth.

async fn list_sessions(
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

async fn create_session(
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

async fn read_history(
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
    if let Some(limit) = q.limit {
        if messages.len() > limit {
            messages = messages.split_off(messages.len() - limit);
        }
    }
    (StatusCode::OK, Json(messages)).into_response()
}

/// Run one turn by re-executing this binary in one-shot mode. This keeps serve
/// behavior identical to the local assistant without duplicating the driver.
///
/// Error detail is capped so a large dump (possibly containing local paths)
/// never floods the API response.
const MAX_TURN_ERROR_CHARS: usize = 4096;
/// Cap for one forwarded child-output line so a single tool dump cannot blow
/// up the SSE frame budget.
const MAX_SSE_LINE_CHARS: usize = 8192;

fn run_one_shot_turn(session_id: &str, prompt: &str) -> std::io::Result<String> {
    let exe = std::env::current_exe()?;
    let out = std::process::Command::new(exe)
        .arg("--session")
        .arg(session_id)
        .arg(prompt)
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
    let lock = session_lock(&state, &id).await;
    let _guard = lock.lock().await;
    let session_id = id.clone();
    let output = tokio::task::spawn_blocking(move || run_one_shot_turn(&session_id, &prompt))
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
    (
        StatusCode::OK,
        Json(TurnResp {
            session_id: id,
            output,
        }),
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
    // The per-session lock moves into the pump task (not the handler scope):
    // the handler returns the SSE response immediately, so holding the guard
    // here would release it before the turn finishes and allow overlapping
    // writers on one session.
    tokio::task::spawn(async move {
        let _guard = lock.lock().await;
        let _ = tokio::task::spawn_blocking(move || {
            stream_child_turn(session_id, prompt, tx, live_fifo)
        })
        .await;
    });
    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (event, rx))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Send-side millisecond clock for the per-turn diagnostic line at the end
/// of [`stream_child_turn`]. Shared by the stdout loop and the FIFO pump
/// thread; every stamp is first-writer-wins so concurrent forwards cannot
/// skew the spans.
#[derive(Clone, Default)]
struct TurnSendTiming {
    first_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
    done_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl TurnSendTiming {
    fn stamp_ms(target: &std::sync::atomic::AtomicU64, t0: std::time::Instant) {
        let ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let _ = target.compare_exchange(
            0,
            ms,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    fn stamp_first(&self, t0: std::time::Instant) {
        Self::stamp_ms(&self.first_ms, t0);
    }

    fn stamp_done(&self, t0: std::time::Instant) {
        Self::stamp_ms(&self.done_ms, t0);
    }
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
    tx: mpsc::Sender<Result<Event, std::convert::Infallible>>,
    mut live_fifo: Option<LiveFifo>,
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
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(err) => {
            send(error_event(format!("cannot locate current binary: {err}")));
            send_done();
            return;
        }
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--session")
        .arg(&session_id)
        .arg(&prompt)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Marks the child as serve-spawned even when no FIFO is set up
        // (fallback line pump): the title skip must not depend on streaming.
        .env(background::SERVE_CHILD_ENV, "1");
    if let Some(fifo) = live_fifo.as_ref() {
        cmd.env(background::SERVE_LIVE_FIFO_ENV, &fifo.path);
    }
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            send(error_event(format!("cannot start turn process: {err}")));
            send_done();
            return;
        }
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
            std::thread::spawn(move || {
                pump_live_fifo(reader, &child_done, &turn_done, &tx, pump_timing, t0)
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
            send(Event::default().data(truncate_line(&line)));
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
/// 4 = thinking done, 5 = output complete).
#[derive(Debug, PartialEq, Eq)]
enum ServeLiveEvent {
    Delta(String),
    ThinkingStart,
    Thinking(String),
    ThinkingDone,
    OutputComplete,
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
                || kind == background::ServeLiveKind::OutputComplete as u8;
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
) {
    timing.stamp_first(t0);
    if matches!(event, ServeLiveEvent::OutputComplete) {
        if !turn_done.swap(true, std::sync::atomic::Ordering::SeqCst) {
            timing.stamp_done(t0);
            let _ = tx.blocking_send(Ok(done_event()));
        }
        return;
    }
    if let Some(e) = live_event(event) {
        let _ = tx.blocking_send(Ok(e));
    }
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
                    forward_live_event(event, turn_done, tx, &timing, t0);
                }
                break;
            }
            // Transient EOF (writer not opened yet): back off, don't spin.
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }
        for event in decoder.push(&buf[..n as usize]) {
            forward_live_event(event, turn_done, tx, &timing, t0);
        }
    }
}

fn delta_event(delta: String) -> Event {
    Event::default().event("delta").data(
        serde_json::json!({"delta": delta}).to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::{ServeFrameDecoder, ServeImageUpload, ServeLiveEvent, setup_live_fifo};
    use super::{MAX_TURN_IMAGES, stage_turn_images};
    #[cfg(unix)]
    use super::pump_live_fifo;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use uuid::Uuid;

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
}

fn truncate_line(line: &str) -> String {
    if line.chars().count() <= MAX_SSE_LINE_CHARS {
        line.to_string()
    } else {
        let kept: String = line.chars().take(MAX_SSE_LINE_CHARS).collect();
        format!("{kept}...[line truncated]")
    }
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

pub(in crate::ai) async fn run_serve(
    cli: super::cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the same config resolution as the CLI so serve sees the same
    // history file, models, and sandbox settings.
    let app_config = super::config::load_config()?;
    let history_file = app_config.history_file.clone();
    let cfg = configw::get_all_config();
    let bind = if !cli.serve_bind.trim().is_empty() {
        cli.serve_bind.trim().to_string()
    } else {
        let from_cfg = cfg.get_opt(AiConfig::SERVE_BIND).unwrap_or_default();
        if from_cfg.trim().is_empty() {
            DEFAULT_BIND.to_string()
        } else {
            from_cfg
        }
    };
    let token = cfg.get_opt(AiConfig::SERVE_TOKEN).unwrap_or_default();
    if token.is_empty() && !is_loopback_bind(&bind) {
        return Err(
            "refusing to serve without ai.serve.token on a non-loopback bind; set ai.serve.token or bind 127.0.0.1".into(),
        );
    }
    if token.is_empty() {
        eprintln!("[serve] warning: ai.serve.token is empty; listening on loopback without auth");
    }
    let state = ServeState {
        history_file,
        token,
        // One entry per touched session; bounded by session count in practice.
        // Idle eviction is deferred to a later multi-instance pass.
        locks: Arc::new(Mutex::new(HashMap::new())),
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/sessions", get(list_sessions).post(create_session))
        .route("/sessions/{id}/history", get(read_history))
        .route(
            "/sessions/{id}/turns",
            post(post_turn).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route(
            "/sessions/{id}/turns/stream",
            post(post_turn_sse).layer(DefaultBodyLimit::max(MAX_TURN_REQUEST_BYTES)),
        )
        .route("/skills", get(list_skills))
        .route("/agents", get(list_agents))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("[serve] listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
