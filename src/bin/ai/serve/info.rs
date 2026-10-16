//! Public liveness/UI (`GET /healthz`, `GET /app`) and runtime info
//! (`GET /info`).

use super::*;
use axum::response::Redirect;

pub(crate) async fn healthz() -> Json<Healthz> {
    Json(Healthz {
        ok: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
/// Mobile web client (`GET /app`, public like `/healthz`): a single-file
/// chat UI compiled in via `include_str!`, available both in a browser and as
/// an installed PWA. The page
/// itself carries no session data; the bearer token lives in the phone's
/// `localStorage` and every API call reuses the existing authed routes.
// `no-store` keeps phones from running a stale cached copy of this page
// after an upgrade (a stale page hides new diagnostics like the stored
// token length below the save row).
pub(crate) async fn serve_app() -> (
    [(axum::http::header::HeaderName, &'static str); 1],
    Html<&'static str>,
) {
    (
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Html(include_str!("app.html")),
    )
}

/// Redirect API requests made by an already-open client that retained `/app`
/// as its server base. A temporary redirect preserves the method and body, so
/// stale POST/SSE callers recover as well as the initial health check.
pub(crate) async fn redirect_legacy_app_api(uri: axum::http::Uri) -> Redirect {
    let suffix = uri.path().strip_prefix("/app/").unwrap_or_default();
    let mut target = format!("/{}", suffix.trim_start_matches('/'));
    if let Some(query) = uri.query() {
        target.push('?');
        target.push_str(query);
    }
    Redirect::temporary(&target)
}

const APP_MANIFEST: &str = r##"{
  "id": "/app",
  "name": "a serve",
  "short_name": "a",
  "description": "Mobile client for a serve",
  "start_url": "/app",
  "scope": "/",
  "display": "standalone",
  "background_color": "#0d1117",
  "theme_color": "#0d1117",
  "launch_handler": { "client_mode": "navigate-existing" },
  "icons": [
    { "src": "/app-icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any" },
    { "src": "/app-icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any" },
    { "src": "/app-icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}"##;

const APP_ICON_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" rx="112" fill="#0d1117"/><circle cx="256" cy="256" r="160" fill="#2f81f7"/><text x="256" y="270" fill="white" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="300" font-weight="700" text-anchor="middle" dominant-baseline="central">a</text></svg>"##;

const APP_SERVICE_WORKER: &str = r##"self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});
self.addEventListener("fetch", (event) => {
  if (event.request.method === "GET") event.respondWith(fetch(event.request));
});
"##;

/// Public manifest: browsers cannot attach the API bearer token while
/// installing a web app, and this static identity contains no session data.
pub(crate) async fn serve_app_manifest() -> Response {
    public_app_asset(
        APP_MANIFEST,
        "application/manifest+json; charset=utf-8",
        "no-cache",
    )
}

/// Public maskable icon referenced by both the manifest and the document head.
pub(crate) async fn serve_app_icon() -> Response {
    public_app_asset(APP_ICON_SVG, "image/svg+xml", "public, max-age=86400")
}

pub(crate) async fn serve_app_icon_192() -> Response {
    public_app_png_icon(192)
}

pub(crate) async fn serve_app_icon_512() -> Response {
    public_app_png_icon(512)
}

fn public_app_png_icon(size: u32) -> Response {
    let mut image = image::RgbaImage::from_pixel(size, size, image::Rgba([13, 17, 23, 255]));
    let scale = size as f32;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let px = x as f32 + 0.5;
        let py = y as f32 + 0.5;
        let circle =
            (px - scale * 0.5).powi(2) + (py - scale * 0.5).powi(2) <= (scale * 0.36).powi(2);
        if circle {
            *pixel = image::Rgba([47, 129, 247, 255]);
        }
        let outer = ((px - scale * 0.46) / (scale * 0.14)).powi(2)
            + ((py - scale * 0.55) / (scale * 0.13)).powi(2)
            <= 1.0;
        let inner = ((px - scale * 0.46) / (scale * 0.075)).powi(2)
            + ((py - scale * 0.55) / (scale * 0.065)).powi(2)
            <= 1.0;
        let stem =
            px >= scale * 0.55 && px <= scale * 0.62 && py >= scale * 0.40 && py <= scale * 0.68;
        if (outer && !inner) || stem {
            *pixel = image::Rgba([255, 255, 255, 255]);
        }
    }
    let mut body = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut body, image::ImageFormat::Png)
        .expect("encoding the built-in PWA icon cannot fail");
    let mut response = body.into_inner().into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("image/png"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=86400"),
    );
    response
}

/// Network-only worker: it establishes installability without persisting
/// authenticated API responses, conversation data, or a stale application shell.
pub(crate) async fn serve_app_service_worker() -> Response {
    let mut response = public_app_asset(
        APP_SERVICE_WORKER,
        "application/javascript; charset=utf-8",
        "no-cache",
    );
    response.headers_mut().insert(
        axum::http::header::HeaderName::from_static("service-worker-allowed"),
        HeaderValue::from_static("/"),
    );
    response
}

fn public_app_asset(
    body: &'static str,
    content_type: &'static str,
    cache: &'static str,
) -> Response {
    let mut response = body.into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static(cache),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
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
/// Per-session picks are not global: they live in each session's persisted
/// config (`GET`/`POST /sessions/{id}/config`) and are folded into turns
/// server-side, so `/info` reports only the server defaults and option lists.
#[derive(Debug, Serialize)]
pub(crate) struct ServerModelOption {
    /// Value to send back as the per-turn override (registry handle).
    id: String,
    /// Human-readable label for the dropdown row.
    label: String,
}
#[derive(Debug, Serialize)]
pub(crate) struct ServerInfo {
    model: String,
    model_label: String,
    agent: String,
    reasoning_effort: String,
    version: String,
    models: Vec<ServerModelOption>,
    agents: Vec<String>,
    efforts: Vec<&'static str>,
}

pub(crate) async fn server_info(
    State(state): State<ServeState>,
    headers: HeaderMap,
) -> impl IntoResponse {
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
