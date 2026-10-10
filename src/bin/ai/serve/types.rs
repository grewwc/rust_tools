//! Serve-mode HTTP request/response types and pure helpers shared by
//! the route modules (see `mod.rs`).

use super::*;

pub(crate) const DEFAULT_BIND: &str = "127.0.0.1:8080";

#[derive(Debug, Clone)]
pub(crate) struct ServeState {
    pub(crate) history_file: PathBuf,
    /// Root for workspace-relative image paths on the read-only preview route
    /// (`GET /sessions/{id}/file`). Snapshotted at startup from the same
    /// authority a turn child runs with (`runtime_ctx::effective_cwd`), so a
    /// path the assistant prints in a reply resolves to the file it wrote.
    /// This session's assets dir is a second, per-request root.
    pub(crate) workspace_root: PathBuf,
    pub(crate) token: String,
    /// Per-session async mutex: at most one turn writer per session.
    pub(crate) locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    /// Session id -> pid of the running turn child, for
    /// `POST /sessions/{id}/interrupt`. `locks` allows at most one live entry
    /// per session; the entry is removed when the child is reaped.
    pub(crate) active_turns: Arc<std::sync::Mutex<HashMap<String, u32>>>,
    /// Session id -> the running turn's confirmation channel, for
    /// `GET`/`POST /sessions/{id}/confirm`. Registered before the turn can ask
    /// anything and removed when its child is reaped, so a question never
    /// outlives the process that asked it.
    pub(crate) confirms: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConfirmSlot>>>>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Healthz {
    pub(crate) ok: bool,
    pub(crate) version: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionItem {
    pub(crate) id: String,
    pub(crate) size_bytes: u64,
    pub(crate) summary: Option<String>,
    pub(crate) first_user_prompt: Option<String>,
    pub(crate) marked: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SessionsQuery {
    pub(crate) limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CreateSessionResp {
    pub(crate) id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HistoryQuery {
    pub(crate) limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TurnReq {
    pub(crate) prompt: String,
    /// Per-turn model override for serve-chat `/model` switching. `None`
    /// (or absent, for old clients) keeps the server default reported by
    /// `GET /info`; `Some(id)` is forwarded as the child `--model` flag so
    /// the turn resolves through the same registry path as the local REPL.
    #[serde(default)]
    pub(crate) model: Option<String>,
    /// Per-turn agent override for serve-chat `/agent` switching. Forwarded
    /// as the child `--agent` flag; absent keeps the `"build"` fallback.
    #[serde(default)]
    pub(crate) agent: Option<String>,
    /// Per-turn reasoning-effort override for serve-chat `/effort`
    /// switching (`minimal|low|medium|high|xhigh|max|off`, case-insensitive).
    /// Absent (cleared) keeps the server default; forwarded as the child
    /// `--reasoning-effort` flag.
    #[serde(default)]
    pub(crate) reasoning_effort: Option<String>,
    /// Client-uploaded images for `[[image:name]]` placeholders in `prompt`.
    /// A remote serve-chat client shares no filesystem with the server, so
    /// pasted images travel inside the request (base64) and are staged under
    /// the same filenames before the turn runs. Absent for old clients and
    /// for text-only turns.
    #[serde(default)]
    pub(crate) images: Vec<ServeImageUpload>,
    /// Whether this client renders and answers remote confirmation requests
    /// (`confirm_request` SSE events plus `GET`/`POST .../confirm`). Only an
    /// explicit `true` opens the child's channel: a client that cannot answer
    /// (older pages, non-interactive terminals) keeps the previous fail-closed
    /// behavior instead of hanging the turn on a question nobody can see.
    #[serde(default)]
    pub(crate) confirm: Option<bool>,
}

/// Max length for one per-turn model/agent override: identifiers are short
/// registry names; the cap keeps a chatty client from bloating argv.
pub(crate) const MAX_TURN_OVERRIDE_CHARS: usize = 128;

/// Validated per-turn overrides extracted from [`TurnReq`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct TurnOverrides {
    pub(crate) model: Option<String>,
    pub(crate) agent: Option<String>,
    pub(crate) reasoning_effort: Option<String>,
}

/// Validate one free-form model/agent override: non-empty after trim, short,
/// and never flag-shaped (a `--x` value would confuse the child CLI parser,
/// which only claims non-`-` tokens for `--model`/`--agent` values).
pub(crate) fn sanitize_turn_name_override(value: Option<String>, field: &str) -> Result<Option<String>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(validate_turn_name_value(trimmed, field)?))
}

/// Validate one non-empty model/agent value: short, and never flag-shaped (a
/// `--x` value would confuse the child CLI parser, which only claims
/// non-`-` tokens for `--model`/`--agent` values).
pub(crate) fn validate_turn_name_value(trimmed: String, field: &str) -> Result<String, String> {
    if trimmed.len() > MAX_TURN_OVERRIDE_CHARS {
        return Err(format!("{field} override is too long"));
    }
    if trimmed.starts_with('-') {
        return Err(format!("{field} override must not start with '-'"));
    }
    Ok(trimmed)
}

/// Effort levels a remote client may offer per turn. Single source of truth
/// for the `sanitize_turn_effort_override` whitelist below and the `efforts`
/// list published via `GET /info` (`off` included; clearing is expressed by
/// omitting the field, matching the local `/effort auto` semantics).
pub(crate) const SERVE_EFFORT_LEVELS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max", "off"];

/// Validate the `--reasoning-effort` override against the child CLI's
/// accepted values (see `SERVE_EFFORT_LEVELS`).
pub(crate) fn sanitize_turn_effort_override(value: Option<String>) -> Result<Option<String>, String> {
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
pub(crate) fn turn_overrides(req: &TurnReq) -> Result<TurnOverrides, String> {
    Ok(TurnOverrides {
        model: sanitize_turn_name_override(req.model.clone(), "model")?,
        agent: sanitize_turn_name_override(req.agent.clone(), "agent")?,
        reasoning_effort: sanitize_turn_effort_override(req.reasoning_effort.clone())?,
    })
}

/// Append validated per-turn overrides as child CLI flags. Empty means
/// "server default", so no flag is emitted and old bare-`--session` behavior
/// is preserved.
pub(crate) fn push_turn_override_args(cmd: &mut std::process::Command, overrides: &TurnOverrides) {
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

/// One field of a `POST /sessions/{id}/config` update. `None` (field absent)
/// keeps the stored value; `Some(None)` (field present but blank) clears it;
/// `Some(Some(v))` sets it.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct SessionConfigReq {
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) agent: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_effort: Option<String>,
}

/// Normalize one model/agent field of a session-config update. A blank value
/// clears the stored pick; a non-blank one is validated like a per-turn
/// override.
pub(crate) fn sanitize_session_config_name(
    value: Option<String>,
    field: &str,
) -> Result<Option<Option<String>>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        return Ok(Some(None));
    }
    Ok(Some(Some(validate_turn_name_value(trimmed, field)?)))
}

/// Normalize the reasoning-effort field of a session-config update, whitelisted
/// against `SERVE_EFFORT_LEVELS` like a per-turn override.
pub(crate) fn sanitize_session_config_effort(
    value: Option<String>,
) -> Result<Option<Option<String>>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Ok(Some(None));
    }
    if SERVE_EFFORT_LEVELS.contains(&normalized.as_str()) {
        Ok(Some(Some(normalized)))
    } else {
        Err(format!(
            "reasoning_effort override must be minimal|low|medium|high|xhigh|max|off, got {raw:?}"
        ))
    }
}

/// Fold the session's persisted serve config into a turn's per-request
/// overrides: an explicit per-turn override wins; a field the request left
/// absent falls back to the session's stored model/agent/effort, so a turn
/// spawned without client flags still runs with the session's picks.
pub(crate) fn merge_session_config(overrides: &mut TurnOverrides, stored: &SessionServeConfig) {
    if overrides.model.is_none() {
        overrides.model = stored.model.clone();
    }
    if overrides.agent.is_none() {
        overrides.agent = stored.agent.clone();
    }
    if overrides.reasoning_effort.is_none() {
        overrides.reasoning_effort = stored.reasoning_effort.clone();
    }
}

/// Best-effort read of a session's stored serve config; a read failure must
/// never fail a turn, so it degrades to defaults.
pub(crate) fn read_session_config_or_default(store: &SessionStore, session_id: &str) -> SessionServeConfig {
    match store.read_session_serve_config(session_id) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("[serve] reading session config for {session_id} failed: {err}");
            SessionServeConfig::default()
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ServeImageUpload {
    pub(crate) filename: String,
    pub(crate) data_base64: String,
}

/// Per-turn image upload caps: pasted screenshots are small, but the API
/// must not become an arbitrary file drop. The child resolves placeholders
/// against the session assets dir, so only bare filenames with an image
/// extension are accepted — never a path.
pub(crate) const MAX_TURN_IMAGES: usize = 10;
pub(crate) const MAX_TURN_IMAGE_BYTES: usize = 10 * 1024 * 1024;
pub(crate) const MAX_TURN_IMAGES_TOTAL_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const TURN_IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

/// Max JSON body for the two turn endpoints. Images travel base64 (~4/3
/// overhead) inside the request, so the framework default (2MB) would reject
/// legitimate multi-image turns with 413 before the handler caps run.
/// Handler-level caps (10MiB per file, 32MiB total decoded) still enforce the
/// real policy; this only lets those requests reach the handler.
pub(crate) const MAX_TURN_REQUEST_BYTES: usize = 48 * 1024 * 1024;

/// Stage uploaded turn images into the session assets dir under the uploaded
/// filenames, so the one-shot child's existing `[[image:name]]` resolution
/// finds them exactly as if they had been pasted locally. Pure filesystem
/// work; runs before the per-session lock is taken.
pub(crate) fn stage_turn_images(
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
pub(crate) struct TurnResp {
    pub(crate) session_id: String,
    pub(crate) output: String,
}

pub(crate) fn unauthorized(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": msg})),
    )
}

pub(crate) fn bad_request(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg})),
    )
}

pub(crate) fn check_auth(state: &ServeState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
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

pub(crate) async fn session_lock(state: &ServeState, session_id: &str) -> Arc<Mutex<()>> {
    let mut map = state.locks.lock().await;
    map.get(session_id)
        .cloned()
        .unwrap_or_else(|| {
            let lock = Arc::new(Mutex::new(()));
            map.insert(session_id.to_string(), lock.clone());
            lock
        })
}
