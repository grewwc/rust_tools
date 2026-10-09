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
        SessionStore, SessionTitleOrigin, invalidate_context_history_cache_for,
        is_runtime_synthetic_user_message, truncate_history_messages,
        write_stale_patch_targets_sqlite,
    },
    model_names, models,
};
use super::driver::turn_runtime::stale_patch_targets_from_messages;
pub(in crate::ai) mod chat;
pub(in crate::ai) mod ctl;

const DEFAULT_BIND: &str = "127.0.0.1:8080";

#[derive(Debug, Clone)]
struct ServeState {
    history_file: PathBuf,
    /// Root for workspace-relative image paths on the read-only preview route
    /// (`GET /sessions/{id}/file`). Snapshotted at startup from the same
    /// authority a turn child runs with (`runtime_ctx::effective_cwd`), so a
    /// path the assistant prints in a reply resolves to the file it wrote.
    /// This session's assets dir is a second, per-request root.
    workspace_root: PathBuf,
    token: String,
    /// Per-session async mutex: at most one turn writer per session.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    /// Session id -> pid of the running turn child, for
    /// `POST /sessions/{id}/interrupt`. `locks` allows at most one live entry
    /// per session; the entry is removed when the child is reaped.
    active_turns: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    /// Session id -> the running turn's confirmation channel, for
    /// `GET`/`POST /sessions/{id}/confirm`. Registered before the turn can ask
    /// anything and removed when its child is reaped, so a question never
    /// outlives the process that asked it.
    confirms: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
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
    /// Per-turn model override for serve-chat `/model` switching. `None`
    /// (or absent, for old clients) keeps the server default reported by
    /// `GET /info`; `Some(id)` is forwarded as the child `--model` flag so
    /// the turn resolves through the same registry path as the local REPL.
    #[serde(default)]
    model: Option<String>,
    /// Per-turn agent override for serve-chat `/agent` switching. Forwarded
    /// as the child `--agent` flag; absent keeps the `"build"` fallback.
    #[serde(default)]
    agent: Option<String>,
    /// Per-turn reasoning-effort override for serve-chat `/effort`
    /// switching (`minimal|low|medium|high|xhigh|max|off`, case-insensitive).
    /// Absent (cleared) keeps the server default; forwarded as the child
    /// `--reasoning-effort` flag.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Client-uploaded images for `[[image:name]]` placeholders in `prompt`.
    /// A remote serve-chat client shares no filesystem with the server, so
    /// pasted images travel inside the request (base64) and are staged under
    /// the same filenames before the turn runs. Absent for old clients and
    /// for text-only turns.
    #[serde(default)]
    images: Vec<ServeImageUpload>,
    /// Whether this client renders and answers remote confirmation requests
    /// (`confirm_request` SSE events plus `GET`/`POST .../confirm`). Only an
    /// explicit `true` opens the child's channel: a client that cannot answer
    /// (serve-chat's REPL, older pages) keeps the previous fail-closed
    /// behavior instead of hanging the turn on a question nobody can see.
    #[serde(default)]
    confirm: Option<bool>,
}

/// Max length for one per-turn model/agent override: identifiers are short
/// registry names; the cap keeps a chatty client from bloating argv.
const MAX_TURN_OVERRIDE_CHARS: usize = 128;

/// Validated per-turn overrides extracted from [`TurnReq`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TurnOverrides {
    model: Option<String>,
    agent: Option<String>,
    reasoning_effort: Option<String>,
}

/// Validate one free-form model/agent override: non-empty after trim, short,
/// and never flag-shaped (a `--x` value would confuse the child CLI parser,
/// which only claims non-`-` tokens for `--model`/`--agent` values).
fn sanitize_turn_name_override(value: Option<String>, field: &str) -> Result<Option<String>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.len() > MAX_TURN_OVERRIDE_CHARS {
        return Err(format!("{field} override is too long"));
    }
    if trimmed.starts_with('-') {
        return Err(format!("{field} override must not start with '-'"));
    }
    Ok(Some(trimmed))
}

/// Effort levels a remote client may offer per turn. Single source of truth
/// for the `sanitize_turn_effort_override` whitelist below and the `efforts`
/// list published via `GET /info` (`off` included; clearing is expressed by
/// omitting the field, matching the local `/effort auto` semantics).
const SERVE_EFFORT_LEVELS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max", "off"];

/// Validate the `--reasoning-effort` override against the child CLI's
/// accepted values (see `SERVE_EFFORT_LEVELS`).
fn sanitize_turn_effort_override(value: Option<String>) -> Result<Option<String>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Ok(None);
    }
    if SERVE_EFFORT_LEVELS.contains(&normalized.as_str()) {
        Ok(Some(normalized))
    } else {
        Err(format!(
            "reasoning_effort override must be minimal|low|medium|high|xhigh|max|off, got {raw:?}"
        ))
    }
}

/// Validate the three per-turn overrides of a turn request.
fn turn_overrides(req: &TurnReq) -> Result<TurnOverrides, String> {
    Ok(TurnOverrides {
        model: sanitize_turn_name_override(req.model.clone(), "model")?,
        agent: sanitize_turn_name_override(req.agent.clone(), "agent")?,
        reasoning_effort: sanitize_turn_effort_override(req.reasoning_effort.clone())?,
    })
}

/// Append validated per-turn overrides as child CLI flags. Empty means
/// "server default", so no flag is emitted and old bare-`--session` behavior
/// is preserved.
fn push_turn_override_args(cmd: &mut std::process::Command, overrides: &TurnOverrides) {
    if let Some(model) = &overrides.model {
        cmd.arg("--model").arg(model);
    }
    if let Some(agent) = &overrides.agent {
        cmd.arg("--agent").arg(agent);
    }
    if let Some(effort) = &overrides.reasoning_effort {
        cmd.arg("--reasoning-effort").arg(effort);
    }
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
        // Log only lengths, never secret content: separates "client sent
        // nothing" (0) from "wrong value with the right length" (typo/case).
        eprintln!(
            "[serve] auth rejected: got {} chars, want {} chars",
            got.len(),
            want.len()
        );
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
/// Mobile web client (`GET /app`, public like `/healthz`): a single-file
/// chat UI compiled in via `include_str!`, so Android phones just open
/// `http://<host>:<port>/app` in Chrome with no app install. The page
/// itself carries no session data; the bearer token lives in the phone's
/// `localStorage` and every API call reuses the existing authed routes.
// `no-store` keeps phones from running a stale cached copy of this page
// after an upgrade (a stale page hides new diagnostics like the stored
// token length below the save row).
async fn serve_app() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    Html<&'static str>,
) {
    (
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Html(include_str!("app.html")),
    )
}
// Note: /healthz is intentionally public (liveness only, no session data).
// All session/skill/agent routes enforce check_auth.

/// Runtime info for remote clients (`GET /info`, authed): the exact
/// model/agent/reasoning-effort labels the next turn will run with, so a
/// remote input header can mirror the local REPL instead of guessing from
/// the client's own flags. The turn child is spawned as bare
/// `--session <id> <prompt>` with no `--model`/`--agent` passthrough, so
/// client-side flags never reach it — only server-side truth is displayed.
/// The `models`/`agents`/`efforts` option lists drive the mobile client's
/// dropdowns; an empty per-turn override keeps meaning "server default".
#[derive(Debug, Serialize)]
struct ServerModelOption {
    /// Value to send back as the per-turn override (registry handle).
    id: String,
    /// Human-readable label for the dropdown row.
    label: String,
}
#[derive(Debug, Serialize)]
struct ServerInfo {
    model: String,
    model_label: String,
    agent: String,
    reasoning_effort: String,
    version: String,
    models: Vec<ServerModelOption>,
    agents: Vec<String>,
    efforts: Vec<&'static str>,
}

async fn server_info(State(state): State<ServeState>, headers: HeaderMap) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    // Mirror `models::initial_model` minus its CLI-override branch: the
    // child CLI carries no `--model`, so the config default (resolved
    // through the registry, falling back to the registry default) is what
    // the turn uses.
    let model = default_turn_model();
    // Mirror `driver/mod.rs` App construction: the child CLI carries no
    // `--agent`, so the hardcoded `"build"` fallback is what the turn uses.
    let agent = "build".to_string();
    // Mirror `request::reasoning::reasoning_effort_display_label` for the
    // serve case: no CLI effort override exists server-side and the agent is
    // never `"sharp"`, so the registry default (or "server default") is the
    // displayed tier.
    let reasoning_effort = super::models::default_reasoning_effort(&model)
        .map(|e| e.as_str().to_string())
        .unwrap_or_else(|| "server default".to_string());
    let models = model_names::all()
        .into_iter()
        .map(|def| {
            let id = model_names::model_handle(def);
            let label = models::model_display_label(&id);
            ServerModelOption { id, label }
        })
        .collect();
    // Same switchable set as serve-chat `/agent list` (primary, enabled).
    let agents = agents::get_primary_agents(&agents::load_all_agents())
        .iter()
        .map(|manifest| manifest.name.clone())
        .collect();
    (
        StatusCode::OK,
        Json(ServerInfo {
            model_label: super::models::model_display_label(&model),
            model,
            agent,
            reasoning_effort,
            version: env!("CARGO_PKG_VERSION").to_string(),
            models,
            agents,
            efforts: SERVE_EFFORT_LEVELS.to_vec(),
        }),
    )
        .into_response()
}

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

/// Fork session `{id}` into a new branch, mirroring the local `/fork`:
/// wholesale copy (messages, assets, checkpoints) plus a depth-tagged fork
/// marker title, then the client switches to the new id. Refuses a missing
/// source with 404; a failed title write warns but never fails the fork.
async fn fork_session(
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
async fn delete_session(
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
const MAX_SESSION_TITLE_CHARS: usize = 200;

#[derive(Debug, Deserialize)]
struct SetTitleReq {
    #[serde(default)]
    title: String,
}

/// Rename session `{id}`, mirroring the local `/title <text>`: the title is
/// persisted with the `User` origin, so background auto-generation never
/// overwrites it. Like `/title`, renaming a not-yet-materialized (lazy)
/// session creates its store entry.
async fn set_session_title(
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

/// Request body for `POST /sessions/{id}/rewind`: the canonical (0-based)
/// index into the session's full message list of the user message to rewind
/// to. The mobile client learns that index as
/// `X-History-Total - shown.length + bubble_index` (see `read_history`).
#[derive(Debug, Deserialize)]
struct RewindReq {
    message_index: usize,
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
async fn rewind_history(
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

/// Query for the read-only preview route: one image path, either relative to
/// the workspace root or absolute inside an allowed root.
#[derive(Debug, Deserialize)]
struct FileQuery {
    path: String,
}

/// Image extensions the preview route serves. Everything else (sources,
/// configs, dotfiles, keys) stays invisible to remote clients even inside an
/// allowed root: the route exists so a phone can show pictures the assistant
/// produced, not to browse the server's filesystem.
const PREVIEW_IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "svg", "avif", "bmp"];
/// One preview is capped like an upload: a client allowed to send 10 MiB must
/// be able to fetch what it sent.
const MAX_PREVIEW_BYTES: u64 = MAX_TURN_IMAGE_BYTES as u64;
/// Long absolute paths are legitimate; unbounded input is not.
const MAX_PREVIEW_PATH_CHARS: usize = 4096;

/// A resolved, servable preview file.
#[derive(Debug)]
struct PreviewFile {
    path: PathBuf,
    content_type: &'static str,
}

/// Why a preview path was refused. Each variant maps to one status code, so a
/// client can tell "wrong kind of file" (400) from "outside the roots" (403)
/// and "nothing there" (404) without parsing prose.
#[derive(Debug, PartialEq, Eq)]
enum PreviewReject {
    BadRequest,
    Outside,
    Missing,
    TooLarge,
}

fn preview_content_type(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "avif" => "image/avif",
        _ => "image/bmp",
    }
}

/// Resolve one client-supplied path to a servable image inside `roots`.
///
/// `canonicalize` normalizes the path before the containment check, so `..`
/// segments, relative forms and symlinks all resolve together — a symlink
/// pointing out of the root is rejected exactly like a `..` traversal. The
/// extension allowlist is applied to the canonical target too, so
/// `chart.png -> secret.txt` is refused.
fn resolve_preview(raw: &str, roots: &[PathBuf]) -> Result<PreviewFile, PreviewReject> {
    let raw = raw.trim();
    if raw.is_empty() || raw.len() > MAX_PREVIEW_PATH_CHARS || raw.contains('\0') {
        return Err(PreviewReject::BadRequest);
    }
    let candidate = if std::path::Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        // Relative paths are workspace-relative: that is the directory the
        // turn child runs in, i.e. what the assistant means by `out/chart.svg`.
        let Some(base) = roots.first() else {
            return Err(PreviewReject::BadRequest);
        };
        base.join(raw)
    };
    let Ok(path) = candidate.canonicalize() else {
        return Err(PreviewReject::Missing);
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return Err(PreviewReject::Missing);
    };
    if !meta.is_file() {
        return Err(PreviewReject::Missing);
    }
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !PREVIEW_IMAGE_EXTS.contains(&ext.as_str()) {
        return Err(PreviewReject::BadRequest);
    }
    let inside = roots.iter().any(|root| {
        root.canonicalize()
            .map(|root| path.starts_with(&root))
            .unwrap_or(false)
    });
    if !inside {
        return Err(PreviewReject::Outside);
    }
    if meta.len() > MAX_PREVIEW_BYTES {
        return Err(PreviewReject::TooLarge);
    }
    Ok(PreviewFile {
        path,
        content_type: preview_content_type(&ext),
    })
}

/// `GET /sessions/{id}/file?path=...` (authed): one image from the workspace
/// root or this session's assets dir, fetched by the mobile page for inline
/// previews. Read-only and image-only by construction (see `resolve_preview`);
/// the client sends its bearer token in a `fetch` header, so the token never
/// appears in a URL.
async fn get_session_file(
    State(state): State<ServeState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<FileQuery>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }
    if SessionStore::validate_session_id(&id).is_err() {
        return bad_request(format!("invalid session id: {id}")).into_response();
    }
    let store = SessionStore::new(state.history_file.as_path());
    let roots = vec![state.workspace_root.clone(), store.session_assets_dir(&id)];
    let file = match resolve_preview(&q.path, &roots) {
        Ok(file) => file,
        Err(reject) => {
            let (status, msg) = match reject {
                PreviewReject::BadRequest => {
                    (StatusCode::BAD_REQUEST, "not a previewable image path")
                }
                PreviewReject::Outside => (
                    StatusCode::FORBIDDEN,
                    "path is outside the workspace and session assets",
                ),
                PreviewReject::Missing => (StatusCode::NOT_FOUND, "no such image"),
                PreviewReject::TooLarge => {
                    (StatusCode::PAYLOAD_TOO_LARGE, "image exceeds the preview size cap")
                }
            };
            return (status, Json(serde_json::json!({"error": msg}))).into_response();
        }
    };
    // The size cap bounds this to <= 10 MiB, so one blocking read on a
    // blocking thread keeps the runtime free without a streaming path.
    let path = file.path.clone();
    let bytes = match tokio::task::spawn_blocking(move || std::fs::read(path)).await {
        Ok(Ok(bytes)) => bytes,
        _ => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "no such image"})),
            )
                .into_response();
        }
    };
    let mut resp = Response::new(axum::body::Body::from(bytes));
    let h = resp.headers_mut();
    h.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(file.content_type),
    );
    h.insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    // Per-session data behind a bearer token: no shared cache may keep it.
    h.insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=60"),
    );
    // SVG is a scriptable document. `<img>` never runs its scripts, but a
    // direct navigation (opening the URL in a tab) would execute them in this
    // origin, where the page keeps its bearer token. `sandbox` disables
    // scripting while still rendering the picture.
    if file.content_type == "image/svg+xml" {
        h.insert(
            axum::http::header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static("default-src 'none'; style-src 'unsafe-inline'; sandbox"),
        );
    }
    resp
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

/// Model a serve turn runs with when the client sent no per-turn override.
///
/// Mirrors `models::initial_model` minus its CLI-override branch: the child CLI
/// carries no `--model`, so the config default (resolved through the registry,
/// falling back to the registry default) is what the turn uses.
fn default_turn_model() -> String {
    let cfg = configw::get_all_config();
    cfg.get_opt(AiConfig::MODEL_DEFAULT)
        .filter(|v| !v.trim().is_empty())
        .map(|v| super::models::determine_model(&v))
        .unwrap_or_else(super::models::default_model)
}

/// Generate the session's model title once its turn child has exited.
///
/// Turn children deliberately skip the title round-trip: they hold the
/// session's turn lock until they exit, so the extra request would stall the
/// session's next turn. Running it here — after the child was reaped and the
/// lock released — gives served sessions a real title while keeping the
/// request out of that critical section. Best-effort by construction: it never
/// delays or fails a turn, and it leaves a session that already has a model
/// title (or a user-set one) alone.
fn spawn_session_title_task(history_file: PathBuf, session_id: String, model: Option<String>) {
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
        )
        .await;
    });
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
    let overrides = match turn_overrides(&req) {
        Ok(overrides) => overrides,
        Err(err) => return bad_request(err).into_response(),
    };
    let title_model = overrides.model.clone();
    let lock = session_lock(&state, &id).await;
    let guard = lock.lock().await;
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
    // The child is gone: generate the title it skipped, outside the lock it
    // held for its whole run.
    drop(guard);
    spawn_session_title_task(state.history_file.clone(), id.clone(), title_model);
    (
        StatusCode::OK,
        Json(TurnResp {
            session_id: id,
            output,
        }),
    )
        .into_response()
}

/// Ask the in-flight turn child for `session_id` to stop early by delivering
/// SIGINT to it. The child is the same one-shot binary the local REPL runs, so
/// it takes the local first-Ctrl+C path (cancel the stream, finalize the
/// partial turn) instead of being killed. `false` means no turn child is
/// registered: the turn already finished, which is a normal answer for an
/// interrupt that raced the end of the stream.
fn interrupt_active_turn(
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
async fn post_interrupt(
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
    let overrides = match turn_overrides(&req) {
        Ok(overrides) => overrides,
        Err(err) => return bad_request(err).into_response(),
    };
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
        // The child is gone: generate the title it skipped, outside the lock it
        // held for its whole run. `done` already reached the client, and the
        // request must not keep this session's next turn waiting.
        drop(guard);
        spawn_session_title_task(history_file, title_session_id, title_model);
    });
    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (event, rx))
    });
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Removes the session's entry from the active-turn registry when the turn
/// child is reaped, so an interrupt can only ever reach a live process.
struct ActiveTurnGuard {
    map: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    session_id: String,
    pid: u32,
}

impl ActiveTurnGuard {
    fn register(
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
struct PendingConfirm {
    id: u64,
    prompt: String,
    /// Token handed to the client with the question; an answer must echo it.
    /// Child-side ids restart at 1 in every turn, so the id alone cannot tell
    /// one turn's question from another's.
    token: u64,
}

/// Hands out the [`PendingConfirm::token`] values. Comparing the echoed token
/// is what keeps a dialog left over from an earlier turn from deciding the
/// question that took its id.
static NEXT_CONFIRM_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Per-turn confirmation channel: the question the child is blocked on (if
/// any) and the pipe that answers it. The slot lives in [`ServeState`], not in
/// the SSE handler, so a question asked while the page is away (phone in the
/// background, connection dropped) is still there to be answered later.
#[derive(Debug, Default)]
struct ConfirmSlot {
    pending: Option<PendingConfirm>,
    /// Writing end of the turn child's stdin, taken from the spawned child.
    /// Held here so the answer path never depends on the client that started
    /// the turn.
    answers: Option<std::process::ChildStdin>,
}

/// Owns the session's confirmation entry for one turn: inserted before the
/// FIFO pump starts, removed when the turn child is reaped on any exit path.
struct ConfirmGuard {
    map: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
    session_id: String,
    slot: Arc<std::sync::Mutex<ConfirmSlot>>,
}

impl ConfirmGuard {
    fn register(
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

    fn slot(&self) -> &Arc<std::sync::Mutex<ConfirmSlot>> {
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
fn parse_confirm_request(payload: &str) -> Option<(u64, String)> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    Some((
        value.get("id")?.as_u64()?,
        value.get("prompt")?.as_str()?.to_string(),
    ))
}

/// Parse the `{"id":<u64>}` frame that closes a confirmation request.
fn parse_confirm_id(payload: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()?
        .get("id")?
        .as_u64()
}

/// The session's live confirmation channel, if a turn with the channel
/// enabled is running.
fn confirm_slot(state: &ServeState, id: &str) -> Option<Arc<std::sync::Mutex<ConfirmSlot>>> {
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
fn dismiss_pending_confirm(
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

fn conflict(msg: String) -> (StatusCode, Json<serde_json::Value>) {
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
async fn get_confirm(
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
struct ConfirmAnswerReq {
    id: u64,
    /// Echo of the token the question was shown with.
    token: u64,
    allow: bool,
}

/// `POST /sessions/{id}/confirm` (authed): answer the pending question, which
/// the child receives as one `yes`/`no` line on stdin. `409` means the id no
/// longer names a pending question (answered on another device, superseded, or
/// the turn ended); clients treat that as terminal, not as a retry.
async fn post_confirm(
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

#[cfg(test)]
mod tests {
    use super::{ServeFrameDecoder, ServeImageUpload, ServeLiveEvent, setup_live_fifo};
    use super::{MAX_TURN_IMAGES, clean_sse_line, stage_turn_images};
    use super::{TurnReq, turn_overrides};
    #[cfg(unix)]
    use super::pump_live_fifo;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use uuid::Uuid;

    /// Handler-level coverage for the two session-lifecycle routes: a
    /// missing fork source 404s, a malformed id 400s, and deleting a
    /// missing session still succeeds (idempotent, so client retries after
    /// a dropped connection stay safe). Body shapes are covered by the
    /// client loopback tests in `chat.rs`; store behavior by the
    /// `SessionStore` fork/delete tests.
    fn lifecycle_test_state() -> super::ServeState {
        let root = std::env::temp_dir().join(format!(
            "a-serve-lifecycle-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        super::ServeState {
            history_file: root.join("history.sqlite"),
            workspace_root: root.join("workspace"),
            token: String::new(),
            locks: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
        }
    }

    /// Preview-path resolution is the security-bearing piece of
    /// `GET /sessions/{id}/file`: both roots accept images, and everything
    /// else (traversal, foreign paths, non-images, directories, oversized
    /// files) is refused with a distinguishable reason.
    #[test]
    fn resolve_preview_accepts_images_inside_the_roots_only() {
        use super::{MAX_PREVIEW_BYTES, PreviewReject, resolve_preview};
        let base = std::env::temp_dir().join(format!(
            "a-serve-preview-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let workspace = base.join("workspace");
        let assets = base.join("s1.assets");
        std::fs::create_dir_all(&workspace).expect("workspace");
        std::fs::create_dir_all(&assets).expect("assets");
        std::fs::write(workspace.join("chart.svg"), b"<svg/>").expect("chart");
        std::fs::write(workspace.join("shot.PNG"), b"png").expect("shot");
        std::fs::write(workspace.join("notes.txt"), b"nope").expect("notes");
        std::fs::write(assets.join("paste-1.webp"), b"webp").expect("paste");
        std::fs::write(base.join("outside.png"), b"outside").expect("outside");
        let roots = vec![workspace.clone(), assets.clone()];

        let got = resolve_preview("chart.svg", &roots).expect("relative workspace path");
        assert_eq!(got.content_type, "image/svg+xml");
        assert_eq!(
            got.path,
            workspace.join("chart.svg").canonicalize().expect("canon")
        );
        assert_eq!(
            resolve_preview("shot.PNG", &roots).expect("uppercase ext").content_type,
            "image/png",
            "the allowlist is case-insensitive"
        );
        // Absolute paths are checked against the same roots, so a file that
        // only lives in the session assets dir is still servable.
        let abs = assets.join("paste-1.webp");
        assert_eq!(
            resolve_preview(abs.to_str().expect("utf8"), &roots)
                .expect("assets path")
                .content_type,
            "image/webp"
        );

        assert_eq!(
            resolve_preview("", &roots).unwrap_err(),
            PreviewReject::BadRequest
        );
        assert_eq!(
            resolve_preview("notes.txt", &roots).unwrap_err(),
            PreviewReject::BadRequest,
            "non-image extensions stay invisible even inside the root"
        );
        assert_eq!(
            resolve_preview("missing.png", &roots).unwrap_err(),
            PreviewReject::Missing
        );
        assert_eq!(
            resolve_preview("../outside.png", &roots).unwrap_err(),
            PreviewReject::Outside,
            "a traversal out of the root must not resolve"
        );
        assert_eq!(
            resolve_preview(base.join("outside.png").to_str().expect("utf8"), &roots).unwrap_err(),
            PreviewReject::Outside,
            "an absolute path outside both roots is refused"
        );
        std::fs::create_dir_all(workspace.join("dir.png")).expect("dir");
        assert_eq!(
            resolve_preview("dir.png", &roots).unwrap_err(),
            PreviewReject::Missing,
            "a directory is not a previewable file"
        );
        let big = workspace.join("big.png");
        std::fs::File::create(&big)
            .expect("big")
            .set_len(MAX_PREVIEW_BYTES + 1)
            .expect("len");
        assert_eq!(
            resolve_preview("big.png", &roots).unwrap_err(),
            PreviewReject::TooLarge
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("outside.png"), workspace.join("link.png"))
                .expect("symlink");
            assert_eq!(
                resolve_preview("link.png", &roots).unwrap_err(),
                PreviewReject::Outside,
                "a symlink out of the root is refused like a traversal"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The preview route end to end: bearer auth, the bytes and content type
    /// the page needs, SVG hardening, and every request-level refusal.
    #[tokio::test]
    async fn file_route_serves_images_and_guards_access() {
        use axum::{http::StatusCode, response::IntoResponse as _};

        use super::SessionStore;

        async fn call(state: super::ServeState, auth: bool, path: &str) -> super::Response {
            use axum::response::IntoResponse as _;
            let mut headers = axum::http::HeaderMap::new();
            if auth {
                headers.insert(
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderValue::from_static("Bearer t"),
                );
            }
            super::get_session_file(
                axum::extract::State(state),
                headers,
                axum::extract::Path("s1".to_string()),
                axum::extract::Query(super::FileQuery {
                    path: path.to_string(),
                }),
            )
            .await
            .into_response()
        }
        let base = std::env::temp_dir().join(format!(
            "a-serve-file-route-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let workspace = base.join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace");
        std::fs::write(workspace.join("chart.svg"), b"<svg/>").expect("svg");
        std::fs::write(workspace.join("notes.txt"), b"nope").expect("notes");
        std::fs::write(base.join("outside.png"), b"outside").expect("outside");
        let history_file = base.join("history.sqlite");
        let assets = SessionStore::new(&history_file).session_assets_dir("s1");
        std::fs::create_dir_all(&assets).expect("assets");
        std::fs::write(assets.join("pasted.png"), b"png-bytes").expect("pasted");
        let state = super::ServeState {
            history_file,
            workspace_root: workspace,
            token: "t".to_string(),
            locks: Default::default(),
            active_turns: Default::default(),
            confirms: Default::default(),
        };

        let resp = call(state.clone(), false, "chart.svg").await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = call(state.clone(), true, "chart.svg").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/svg+xml")
        );
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_SECURITY_POLICY)
                .and_then(|v| v.to_str().ok()),
            Some("default-src 'none'; style-src 'unsafe-inline'; sandbox"),
            "an SVG preview must not be able to script this origin"
        );
        assert_eq!(
            resp.headers()
                .get(axum::http::header::X_CONTENT_TYPE_OPTIONS)
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .expect("body");
        assert_eq!(&body[..], b"<svg/>");

        let pasted = assets.join("pasted.png");
        let resp = call(state.clone(), true, pasted.to_str().expect("utf8")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/png"),
            "session assets are the second allowed root"
        );
        assert!(
            resp.headers()
                .get(axum::http::header::CONTENT_SECURITY_POLICY)
                .is_none(),
            "only SVG carries the sandbox header"
        );

        let resp = call(state.clone(), true, "notes.txt").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = call(state.clone(), true, "missing.png").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let resp = call(
            state.clone(),
            true,
            base.join("outside.png").to_str().expect("utf8"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        let mut bad_id_state = state.clone();
        bad_id_state.token = String::new();
        let resp = super::get_session_file(
            axum::extract::State(bad_id_state),
            axum::http::HeaderMap::new(),
            axum::extract::Path("bad id".to_string()),
            axum::extract::Query(super::FileQuery {
                path: "chart.svg".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Confirmation frames must survive the decoder like any other kind, with
    /// their JSON payload relayed verbatim (the client parses it as JSON).
    #[test]
    fn decoder_passes_confirmation_frames_through() {
        use crate::ai::background::ServeLiveKind;
        let wire = [
            test_frame(
                ServeLiveKind::ConfirmRequest,
                br#"{"id":7,"prompt":"proceed?"}"#,
            ),
            test_frame(ServeLiveKind::ConfirmDone, br#"{"id":7}"#),
        ]
        .concat();
        let mut decoder = ServeFrameDecoder::default();
        assert_eq!(
            decoder.push(&wire),
            vec![
                ServeLiveEvent::ConfirmRequest(r#"{"id":7,"prompt":"proceed?"}"#.to_string()),
                ServeLiveEvent::ConfirmDone(r#"{"id":7}"#.to_string()),
            ]
        );
    }

    /// The `/confirm` route pair: the pending question is readable, an answer
    /// reaches the turn child's stdin exactly once, and answering the same
    /// question again is refused instead of writing a second line.
    #[cfg(unix)]
    #[tokio::test]
    async fn confirm_answer_reaches_the_child_stdin_once() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        // Stand-in turn child: consumes one stdin line and echoes it, which is
        // what the real child's confirmation reader does.
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("read line; echo \"$line\"")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn stand-in child");
        let echo = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut text = String::new();
                let _ = out.read_to_string(&mut text);
                text
            })
        });
        let state = lifecycle_test_state();
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 7,
                    prompt: "proceed?".to_string(),
                    token: 11,
                }),
                answers: child.stdin.take(),
            })),
        );

        let read =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        assert_eq!(read.status(), StatusCode::OK);
        let body = axum::body::to_bytes(read.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["pending"]["id"], 7);
        assert_eq!(view["pending"]["prompt"], "proceed?");
        assert_eq!(view["pending"]["token"], 11);

        let answer = |allow| super::ConfirmAnswerReq { id: 7, token: 11, allow };
        let ok = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(true)),
        )
        .await
        .into_response();
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(echo.and_then(|handle| handle.join().ok()).as_deref(), Some("yes\n"));
        // The question is gone, so a second tap (or a second device) is
        // refused rather than writing another line.
        let again = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(false)),
        )
        .await
        .into_response();
        assert_eq!(again.status(), StatusCode::CONFLICT);
        // A later turn numbers its questions from 1 again, so the same id with
        // a fresh token is a different question: an answer carrying the stale
        // token must be refused instead of deciding text nobody read.
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 7,
                    prompt: "a different question".to_string(),
                    token: 12,
                }),
                answers: None,
            })),
        );
        let stale = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(answer(true)),
        )
        .await
        .into_response();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        let body = axum::body::to_bytes(stale.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["error"], "no such confirmation is pending");
        let _ = child.wait();
    }

    #[tokio::test]
    async fn fork_missing_source_returns_not_found() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::fork_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("missing".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn fork_invalid_id_is_rejected() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::fork_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("../evil".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_unknown_session_still_succeeds() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::delete_session(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("missing".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rename_session_persists_user_title() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        let resp = super::set_session_title(
            State(state.clone()),
            HeaderMap::new(),
            Path("rename-me".to_string()),
            Json(super::SetTitleReq {
                title: "  我的新标题  ".to_string(),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.get("title").and_then(|t| t.as_str()), Some("我的新标题"));
        let store = super::SessionStore::new(state.history_file.as_path());
        let saved = store
            .read_session_title_with_origin("rename-me")
            .expect("read title");
        let saved = saved.expect("title persisted");
        assert_eq!(saved.text, "我的新标题");
        assert_eq!(
            saved.origin,
            super::SessionTitleOrigin::User,
            "renamed title must survive background auto-generation"
        );
    }

    #[tokio::test]
    async fn rename_session_rejects_bad_input() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let overlong = "x".repeat(super::MAX_SESSION_TITLE_CHARS + 1);
        for (id, title, want) in [
            ("ok-id", "", StatusCode::BAD_REQUEST),
            ("ok-id", "   ", StatusCode::BAD_REQUEST),
            ("ok-id", overlong.as_str(), StatusCode::BAD_REQUEST),
            ("../evil", "hi", StatusCode::BAD_REQUEST),
        ] {
            let resp = super::set_session_title(
                State(lifecycle_test_state()),
                HeaderMap::new(),
                Path(id.to_string()),
                Json(super::SetTitleReq {
                    title: title.to_string(),
                }),
            )
            .await
            .into_response();
            assert_eq!(resp.status(), want, "id={id:?} title_len={}", title.len());
        }
    }

    /// Seed helper for the rewind tests: a fixed-id session holding the
    /// given roles in order, like one built by real chat traffic.
    fn seed_rewind_session(state: &super::ServeState, id: &str, roles: &[&str]) {
        use crate::ai::history::{Message, append_history_messages};
        let store = super::SessionStore::new(state.history_file.as_path());
        store.ensure_root_dir().expect("root dir");
        let path = store.session_history_file(id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("session dir");
        }
        let messages: Vec<Message> = roles
            .iter()
            .map(|role| Message {
                role: role.to_string(),
                content: serde_json::Value::String(format!("{role} says hi")),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect();
        append_history_messages(&path, &messages).expect("seed messages");
    }

    #[tokio::test]
    async fn rewind_removes_anchor_and_everything_after_it() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant", "user", "assistant"]);
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 2 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.get("removed").and_then(|n| n.as_u64()), Some(2));
        assert_eq!(v.get("kept").and_then(|n| n.as_u64()), Some(2));
        let store = super::SessionStore::new(state.history_file.as_path());
        let rest = store.read_all_messages("s1").expect("read back");
        assert_eq!(rest.len(), 2);
        assert_eq!(rest[0].role, "user");
        assert_eq!(rest[1].role, "assistant");
    }

    #[tokio::test]
    async fn rewind_rejects_non_user_out_of_range_and_missing() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant"]);
        // Assistant anchor and past-the-end indexes must not truncate.
        for index in [1usize, 2, 99] {
            let resp = super::rewind_history(
                State(state.clone()),
                HeaderMap::new(),
                Path("s1".to_string()),
                Json(super::RewindReq {
                    message_index: index,
                }),
            )
            .await
            .into_response();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "index={index}");
        }
        // A missing session reads back as empty history (same as `read_history`),
        // so rewinding it is an out-of-range anchor, not a 404.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("missing".to_string()),
            Json(super::RewindReq { message_index: 0 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // Failed rewinds leave the session untouched.
        let store = super::SessionStore::new(state.history_file.as_path());
        assert_eq!(store.read_all_messages("s1").expect("read back").len(), 2);
    }

    #[tokio::test]
    async fn rewind_rejects_runtime_injected_user_anchor() {
        use axum::{
            Json,
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        use crate::ai::history::{append_history_messages, runtime_synthetic_user_message};
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user"]);
        let store = super::SessionStore::new(state.history_file.as_path());
        append_history_messages(
            &store.session_history_file("s1"),
            &[runtime_synthetic_user_message(serde_json::Value::String(
                "handoff".to_string(),
            ))],
        )
        .expect("seed synthetic");
        // The injected handoff is a user row but not a real turn boundary.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 1 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // The real user input next to it still rewinds.
        let resp = super::rewind_history(
            State(state.clone()),
            HeaderMap::new(),
            Path("s1".to_string()),
            Json(super::RewindReq { message_index: 0 }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(store.read_all_messages("s1").expect("read back").is_empty());
    }

    #[tokio::test]
    async fn serve_app_returns_mobile_client() {
        let resp = super::serve_app().await;
        assert!(resp
            .0
            .iter()
            .any(|(k, v)| *k == axum::http::header::CACHE_CONTROL && *v == "no-store"));
        let body = resp.1.0;
        assert!(body.contains("id=\"serve-app\""));
        assert!(body.contains("/sessions/"));
        assert!(body.contains("turns/stream"));
        assert!(body.contains("/rewind"));
        assert!(
            body.contains("/file?path="),
            "the page must fetch server-side images through the authed preview route"
        );
    }

    #[tokio::test]
    async fn history_total_header_reports_precut_count() {
        use axum::{
            extract::{Path, Query, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let state = lifecycle_test_state();
        seed_rewind_session(&state, "s1", &["user", "assistant", "user"]);
        let resp = super::read_history(
            State(state),
            HeaderMap::new(),
            Path("s1".to_string()),
            Query(super::HistoryQuery { limit: Some(1) }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        // The mobile client maps a tapped bubble to its canonical index as
        // `total - shown.length + bubble_index`, so the header must describe
        // the full session even when the body is a tail cut.
        assert_eq!(
            resp.headers()
                .get("x-history-total")
                .and_then(|v| v.to_str().ok()),
            Some("3")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: Vec<serde_json::Value> = serde_json::from_slice(&body).expect("json");
        assert_eq!(v.len(), 1);
    }

    #[tokio::test]
    async fn server_info_reports_runtime_labels() {
        use axum::{
            extract::State,
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::server_info(State(lifecycle_test_state()), HeaderMap::new())
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let info: serde_json::Value = serde_json::from_slice(&body).expect("json");
        for key in ["model", "model_label", "agent", "reasoning_effort", "version"] {
            assert!(
                info.get(key).and_then(|v| v.as_str()).is_some(),
                "missing {key}"
            );
        }
        for key in ["models", "agents", "efforts"] {
            assert!(
                info.get(key).and_then(|v| v.as_array()).is_some(),
                "missing {key}"
            );
        }
        assert_eq!(
            info["efforts"],
            serde_json::json!(["minimal", "low", "medium", "high", "xhigh", "max", "off"])
        );
        assert!(
            info["models"].as_array().is_some_and(|options| !options.is_empty()
                && options.iter().all(|o| o.get("id").and_then(|v| v.as_str()).is_some()
                    && o.get("label").and_then(|v| v.as_str()).is_some())),
            "model options need id + label"
        );
        assert_eq!(info["agent"], "build");
        assert!(!info["version"].as_str().unwrap_or_default().is_empty());
    }

    fn turn_req(
        model: Option<&str>,
        agent: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> TurnReq {
        TurnReq {
            prompt: "hi".to_string(),
            model: model.map(str::to_string),
            agent: agent.map(str::to_string),
            reasoning_effort: reasoning_effort.map(str::to_string),
            images: Vec::new(),
            confirm: None,
        }
    }

    #[test]
    fn turn_overrides_default_to_absent() {
        let overrides = turn_overrides(&turn_req(None, None, None)).expect("valid");
        assert_eq!(overrides.model, None);
        assert_eq!(overrides.agent, None);
        assert_eq!(overrides.reasoning_effort, None);
        // Empty/blank strings also mean "server default", never a flag.
        let overrides =
            turn_overrides(&turn_req(Some("  "), Some(""), Some(""))).expect("valid");
        assert_eq!(overrides.model, None);
        assert_eq!(overrides.agent, None);
        assert_eq!(overrides.reasoning_effort, None);
    }

    #[test]
    fn turn_overrides_normalize_and_reject() {
        let overrides =
            turn_overrides(&turn_req(Some(" foo "), Some("build"), Some("LOW"))).expect("valid");
        assert_eq!(overrides.model.as_deref(), Some("foo"));
        assert_eq!(overrides.agent.as_deref(), Some("build"));
        assert_eq!(overrides.reasoning_effort.as_deref(), Some("low"));
        assert!(turn_overrides(&turn_req(Some("--evil"), None, None)).is_err());
        assert!(turn_overrides(&turn_req(None, Some("-x"), None)).is_err());
        assert!(turn_overrides(&turn_req(None, None, Some("ultra"))).is_err());
        let long = "m".repeat(super::MAX_TURN_OVERRIDE_CHARS + 1);
        assert!(turn_overrides(&turn_req(Some(&long), None, None)).is_err());
    }

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
    fn clean_sse_line_strips_the_overwrite_prefix() {
        // Completed/failed tool rows carry `\r\x1b[2K` to redraw the running
        // row on a live terminal; on the append-only SSE wire the prefix must
        // go, while the row indent and content stay byte-identical.
        assert_eq!(
            clean_sse_line("\r\x1b[2K  \u{1b}[32m✓\u{1b}[0m task_integrate"),
            "  \u{1b}[32m✓\u{1b}[0m task_integrate"
        );
        // A bare `\r` alone is still the overwrite control, not content.
        assert_eq!(clean_sse_line("\r[header]"), "[header]");
        // Ordinary rows (indented or not) pass through untouched.
        assert_eq!(clean_sse_line("  ● task_integrate"), "  ● task_integrate");
        assert_eq!(clean_sse_line("↳ speed · x"), "↳ speed · x");
        assert_eq!(clean_sse_line(""), "");
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
                    None,
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

    /// Stopping a turn that is parked on a question: the interrupt path drops
    /// the answer pipe, which unblocks the child's stdin read (EOF) and clears
    /// the question, so a client's dialog disappears instead of waiting on an
    /// answer nobody can give any more.
    #[cfg(unix)]
    #[tokio::test]
    async fn interrupt_dismisses_the_pending_question_and_unblocks_the_child() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        // Stand-in turn child: it echoes what its first stdin line turned out
        // to be. Closing the pipe makes `read` return EOF, so an empty echo is
        // the proof that the blocked read did come back.
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("read line; echo \"read[$line]\"")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn stand-in child");
        let echo = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut text = String::new();
                let _ = out.read_to_string(&mut text);
                text
            })
        });
        let state = lifecycle_test_state();
        state.confirms.lock().expect("confirms").insert(
            "sid".to_string(),
            Arc::new(std::sync::Mutex::new(super::ConfirmSlot {
                pending: Some(super::PendingConfirm {
                    id: 1,
                    prompt: "commit?".to_string(),
                    token: 5,
                }),
                answers: child.stdin.take(),
            })),
        );
        // A registered turn child is what `running` reports. The entry is
        // removed before the interrupt below: this stand-in must not be
        // signalled, and the stop path is what the test is about, not the
        // signal.
        state
            .active_turns
            .lock()
            .expect("active turns")
            .insert("sid".to_string(), std::process::id());
        let busy =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        let body = axum::body::to_bytes(busy.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["running"], true);
        assert_eq!(view["pending"]["id"], 1);
        state.active_turns.lock().expect("active turns").remove("sid");

        let stopped = super::post_interrupt(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
        )
        .await
        .into_response();
        assert_eq!(stopped.status(), StatusCode::OK);
        assert_eq!(
            echo.and_then(|handle| handle.join().ok()).as_deref(),
            Some("read[]\n")
        );

        let after =
            super::get_confirm(State(state.clone()), HeaderMap::new(), Path("sid".to_string()))
                .await
                .into_response();
        let body = axum::body::to_bytes(after.into_body(), 4096)
            .await
            .expect("body");
        let view: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(view["pending"], serde_json::Value::Null);
        assert_eq!(view["running"], false);
        // The dismissed question can no longer be answered: the dialog a
        // client rebuilds from a stale view is refused, not silently accepted.
        let late = super::post_confirm(
            State(state.clone()),
            HeaderMap::new(),
            Path("sid".to_string()),
            axum::Json(super::ConfirmAnswerReq {
                id: 1,
                token: 5,
                allow: true,
            }),
        )
        .await
        .into_response();
        assert_eq!(late.status(), StatusCode::CONFLICT);
    }

    /// The interrupt endpoint's core: the registered pid is what receives
    /// SIGINT, which is the local first-Ctrl+C semantics (cancel the streaming
    /// turn), not a hard kill, and only the addressed session is signalled.
    #[cfg(unix)]
    #[test]
    fn interrupt_active_turn_signals_the_registered_child() {
        use std::os::unix::process::ExitStatusExt;
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let active: std::sync::Mutex<std::collections::HashMap<String, u32>> = Default::default();
        active
            .lock()
            .expect("lock registry")
            .insert("s1".to_string(), child.id());
        assert!(
            super::interrupt_active_turn(&active, "s1"),
            "a registered child must be signalled"
        );
        let status = child.wait().expect("reap sleep");
        assert_eq!(status.signal(), Some(libc::SIGINT));
        assert!(
            !super::interrupt_active_turn(&active, "other"),
            "sessions without a registered child must report no interrupt"
        );
    }

    #[test]
    fn active_turn_guard_clears_its_registry_entry() {
        let active = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        {
            let _guard = super::ActiveTurnGuard::register(&active, "s1", 4242);
            assert_eq!(active.lock().expect("lock").get("s1").copied(), Some(4242));
        }
        assert!(
            active.lock().expect("lock").is_empty(),
            "a reaped child must not stay interruptible"
        );
    }

    /// Route coverage: an idle session answers `{"interrupted": false}` (an
    /// interrupt that raced the end of the stream), a malformed id is rejected
    /// before any signal, and a token-protected server rejects missing auth.
    #[tokio::test]
    async fn interrupt_route_reports_idle_sessions_and_guards_access() {
        use axum::{
            extract::{Path, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
        };
        let resp = super::post_interrupt(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("s1".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["interrupted"], serde_json::Value::Bool(false));

        let mut authed = lifecycle_test_state();
        authed.token = "t".to_string();
        let resp = super::post_interrupt(State(authed), HeaderMap::new(), Path("s1".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let resp = super::post_interrupt(
            State(lifecycle_test_state()),
            HeaderMap::new(),
            Path("bad id".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
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

/// Normalize one child-stdout row for SSE transport: strip the live-terminal
/// overwrite prefix (`\r\x1b[2K`) that completed/failed tool rows carry to
/// redraw the `running` row in place. SSE rows are append-only, so the
/// prefix is meaningless on the wire; worse, the bare `\r` makes axum split
/// the row and re-prefix `data:`, injecting a literal `data: ` fragment that
/// only a terminal honoring the erase escape can hide again. `BufRead::lines`
/// already removed the line ending, so a leading `\r` here is always the
/// overwrite control, never content.
fn clean_sse_line(line: &str) -> String {
    let line = line.strip_prefix('\r').unwrap_or(line);
    let line = line.strip_prefix("\x1b[2K").unwrap_or(line);
    truncate_line(line)
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
