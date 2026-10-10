//! Read-only session file preview route (`GET /sessions/{id}/file`).

use super::*;

/// Query for the read-only preview route: one image path, either relative to
/// the workspace root or absolute inside an allowed root.
#[derive(Debug, Deserialize)]
pub(crate) struct FileQuery {
    pub(crate) path: String,
}

/// Image extensions the preview route serves. Everything else (sources,
/// configs, dotfiles, keys) stays invisible to remote clients even inside an
/// allowed root: the route exists so a phone can show pictures the assistant
/// produced, not to browse the server's filesystem.
pub(crate) const PREVIEW_IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "svg", "avif", "bmp"];
/// One preview is capped like an upload: a client allowed to send 10 MiB must
/// be able to fetch what it sent.
pub(crate) const MAX_PREVIEW_BYTES: u64 = MAX_TURN_IMAGE_BYTES as u64;
/// Long absolute paths are legitimate; unbounded input is not.
pub(crate) const MAX_PREVIEW_PATH_CHARS: usize = 4096;

/// A resolved, servable preview file.
#[derive(Debug)]
pub(crate) struct PreviewFile {
    pub(crate) path: PathBuf,
    pub(crate) content_type: &'static str,
}

/// Why a preview path was refused. Each variant maps to one status code, so a
/// client can tell "wrong kind of file" (400) from "outside the roots" (403)
/// and "nothing there" (404) without parsing prose.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PreviewReject {
    BadRequest,
    Outside,
    Missing,
    TooLarge,
}

pub(crate) fn preview_content_type(ext: &str) -> &'static str {
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
pub(crate) fn resolve_preview(raw: &str, roots: &[PathBuf]) -> Result<PreviewFile, PreviewReject> {
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
pub(crate) async fn get_session_file(
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
