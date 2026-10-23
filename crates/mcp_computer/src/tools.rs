//! Tool surface for the computer-use server: the schemas the model sees and the
//! dispatch that turns one `tools/call` into one platform-backend call.
//!
//! The surface is platform-neutral — it speaks in screen points, named buttons, key
//! chords and file paths, never in OS APIs — so a new platform is a new module under
//! `src/backend/` and never a change here (see `src/backend.rs`).
//!
//! Two behaviors decide whether the model can actually drive a desktop:
//!
//! - **`screenshot` returns a file path, not pixels.** The host reads only
//!   `content[0].text` from a tool result, so image bytes placed in the content
//!   envelope would be dropped. `a`'s `read_file` upgrades an image path to
//!   image-input semantics, which is the route that puts the frame in front of a
//!   vision model.
//! - **Backend calls run on a blocking thread under `with_timeout`.** Capture and
//!   injection block (helper subprocess, settled sleeps). Run inline they would hold
//!   the server's only executor thread, the per-operation cap could never fire, and a
//!   hung helper would instead surface to the host as its own request timeout — which
//!   kills and restarts this subprocess. `spawn_blocking` keeps the cap real.

use std::time::{Duration, Instant};

use mcp_stdio::{JsonRpcErr, McpServer, cap_text, text_content, with_timeout};
use rust_tools::pdfw::{OcrLine, OcrResult, ocr_image_with_regions};
use serde_json::{Value, json};

use crate::backend;
use crate::frame::{self, CaptureScope, CapturedFrame, FrameFingerprint};

const MAX_WAIT_MS: u32 = 30_000;
const MAX_CHANGE_WAIT_MS: u32 = 30_000;
const DEFAULT_CHANGE_WAIT_MS: u32 = 10_000;
const DEFAULT_POLL_MS: u32 = 500;
const MAX_POLL_MS: u32 = 5_000;

/// Per-operation cap in ms, overridable with `MCP_COMPUTER_OP_TIMEOUT_MS`.
///
/// The default 90s stays under the host's `request_timeout_ms` (120s) so a slow
/// helper is reported by this server, rather than reaching the host's transport
/// timeout — the host answers that by killing and restarting the subprocess.
pub(crate) fn op_timeout_ms() -> u64 {
    std::env::var("MCP_COMPUTER_OP_TIMEOUT_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(90_000)
}

/// The computer-use MCP server. Stateless: every call is one independent platform
/// operation, and the backend discovers its helper binaries and permissions per call.
pub(crate) struct ComputerServer;

impl McpServer for ComputerServer {
    fn initialize_result(&self) -> Value {
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "mcp-computer", "version": "0.1.0" }
        })
    }

    fn tools_list_result(&self) -> Value {
        json!({ "tools": tools() })
    }

    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr> {
        let params = params.unwrap_or(Value::Null);
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tools/call needs a tool `name`"))?
            .to_string();
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        let report = match name.as_str() {
            // Queries: the backend returns the model-facing report.
            "screen_info" => call(backend::screen_info).await?,
            "list_windows" => {
                let app = optional_string(&args, "app")?;
                call(move || backend::list_windows(app)).await?
            }
            "activate_app" => {
                let app = required_string(&args, "app")?;
                call(move || backend::activate_app(app)).await?
            }
            "activate_window" => {
                let window_id = required_u32(&args, "window_id")?;
                call(move || backend::activate_window(window_id)).await?
            }
            "screenshot" => {
                let scope = capture_scope(&args)?;
                let previous = optional_frame_id(&args, "previous_frame_id")?;
                if let Some(previous) = &previous
                    && previous.scope() != scope
                {
                    return Err(invalid(
                        "'previous_frame_id' belongs to a different screenshot scope",
                    ));
                }
                screenshot(scope, previous).await?
            }
            "find_text" => {
                let scope = capture_scope(&args)?;
                let query = required_string(&args, "text")?;
                let exact = optional_bool(&args, "exact")?.unwrap_or(false);
                let languages = string_list(&args, "languages")?;
                find_text(scope, query, exact, languages).await?
            }
            "click_text" => {
                let scope = capture_scope(&args)?;
                let query = required_string(&args, "text")?;
                let languages = string_list(&args, "languages")?;
                let button =
                    optional_string(&args, "button")?.unwrap_or_else(|| "left".to_string());
                click_text(scope, query, languages, button).await?
            }

            // Actions: the backend confirms success, so the reply states what was done
            // with which arguments — the model's only way to notice that a click landed
            // somewhere other than intended.
            "move_mouse" => {
                let x = required_f64(&args, "x")?;
                let y = required_f64(&args, "y")?;
                call(move || {
                    backend::move_mouse(x, y).map(|()| format!("pointer moved to ({x}, {y})"))
                })
                .await?
            }
            "click" => {
                let x = required_f64(&args, "x")?;
                let y = required_f64(&args, "y")?;
                let button =
                    optional_string(&args, "button")?.unwrap_or_else(|| "left".to_string());
                let count = optional_u32(&args, "count")?.unwrap_or(1).max(1);
                if count > 3 {
                    return Err(invalid("'count' must be between 1 and 3"));
                }
                let modifiers = string_list(&args, "modifiers")?;
                call(move || {
                    backend::click(x, y, &button, count, &modifiers).map(|()| {
                        let held = if modifiers.is_empty() {
                            String::new()
                        } else {
                            format!(" holding {}", modifiers.join("+"))
                        };
                        format!("{button} click ×{count} at ({x}, {y}){held}")
                    })
                })
                .await?
            }
            "drag" => {
                let from_x = required_f64(&args, "from_x")?;
                let from_y = required_f64(&args, "from_y")?;
                let to_x = required_f64(&args, "to_x")?;
                let to_y = required_f64(&args, "to_y")?;
                let button =
                    optional_string(&args, "button")?.unwrap_or_else(|| "left".to_string());
                let modifiers = string_list(&args, "modifiers")?;
                let steps = optional_u32(&args, "steps")?.unwrap_or(20);
                call(move || {
                    backend::drag(from_x, from_y, to_x, to_y, &button, &modifiers, steps).map(|()| {
                        format!(
                            "dragged {button} from ({from_x}, {from_y}) to ({to_x}, {to_y}) in {steps} steps"
                        )
                    })
                })
                .await?
            }
            "scroll" => {
                let x = optional_f64(&args, "x")?;
                let y = optional_f64(&args, "y")?;
                let dx = optional_f64(&args, "dx")?.unwrap_or(0.0);
                let dy = optional_f64(&args, "dy")?.unwrap_or(0.0);
                if dx == 0.0 && dy == 0.0 {
                    return Err(invalid("scroll needs a non-zero `dx` or `dy`"));
                }
                let unit = optional_string(&args, "unit")?.unwrap_or_else(|| "line".to_string());
                call(move || {
                    let at = match (x, y) {
                        (Some(x), Some(y)) => format!(" at ({x}, {y})"),
                        _ => " at the pointer".to_string(),
                    };
                    backend::scroll(x, y, dx, dy, &unit)
                        .map(|()| format!("scrolled dx={dx} dy={dy} in {unit} units{at}"))
                })
                .await?
            }
            "type_text" => {
                let text = required_string(&args, "text")?;
                let characters = text.chars().count();
                // The text itself is deliberately not echoed: typed content is often a
                // credential, and a tool reply is part of the transcript.
                call(move || {
                    backend::type_text(&text)
                        .map(|()| format!("typed {characters} characters into the focused field"))
                })
                .await?
            }
            "press_key" => {
                let key = required_string(&args, "key")?;
                call(move || backend::press_key(&key).map(|()| format!("pressed {key}"))).await?
            }
            "wait" => {
                let duration_ms = required_u32(&args, "duration_ms")?;
                if !(1..=MAX_WAIT_MS).contains(&duration_ms) {
                    return Err(invalid(&format!(
                        "'duration_ms' must be between 1 and {MAX_WAIT_MS}"
                    )));
                }
                wait(duration_ms).await?
            }
            "wait_for_change" => {
                let previous = required_frame_id(&args, "frame_id")?;
                let timeout_ms =
                    optional_u32(&args, "timeout_ms")?.unwrap_or(DEFAULT_CHANGE_WAIT_MS);
                if timeout_ms > MAX_CHANGE_WAIT_MS {
                    return Err(invalid(&format!(
                        "'timeout_ms' must be between 0 and {MAX_CHANGE_WAIT_MS}"
                    )));
                }
                if u64::from(timeout_ms) >= op_timeout_ms() {
                    return Err(invalid(
                        "'timeout_ms' must be below MCP_COMPUTER_OP_TIMEOUT_MS",
                    ));
                }
                let poll_ms = optional_u32(&args, "poll_interval_ms")?.unwrap_or(DEFAULT_POLL_MS);
                if !(100..=MAX_POLL_MS).contains(&poll_ms) {
                    return Err(invalid(&format!(
                        "'poll_interval_ms' must be between 100 and {MAX_POLL_MS}"
                    )));
                }
                wait_for_change(previous, timeout_ms, poll_ms).await?
            }

            other => {
                return Err(JsonRpcErr::new(
                    -32601,
                    &format!("unknown tool `{other}`"),
                    None,
                ));
            }
        };

        Ok(text_content(cap_text(&report)))
    }
}

/// Run one blocking backend call on a blocking thread, bounded by the per-operation
/// cap, and return its model-facing report.
///
/// On timeout the helper keeps running to completion in the background: the cap
/// bounds what the model waits for, not what the platform does — a drag or capture
/// already in flight is not interrupted mid-way.
async fn call<T, F>(job: F) -> Result<T, JsonRpcErr>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    with_timeout(op_timeout_ms(), run_blocking(job)).await
}

async fn run_blocking<T, F>(job: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    match tokio::task::spawn_blocking(job).await {
        Ok(result) => result,
        Err(error) => Err(format!(
            "the platform call did not run to completion: {error}"
        )),
    }
}

async fn wait(duration_ms: u32) -> Result<String, JsonRpcErr> {
    with_timeout(op_timeout_ms(), async move {
        tokio::time::sleep(Duration::from_millis(u64::from(duration_ms))).await;
        Ok(format!(
            "waited {duration_ms} ms; take a new screenshot before the next action"
        ))
    })
    .await
}

async fn screenshot(
    scope: CaptureScope,
    previous: Option<FrameFingerprint>,
) -> Result<String, JsonRpcErr> {
    let started = Instant::now();
    let captured = call(move || CapturedFrame::capture(scope)).await?;
    let status = if previous.is_some() {
        "changed"
    } else {
        "captured"
    };
    if let Some(previous) = &previous
        && captured.fingerprint() == previous
    {
        return Ok(frame::unchanged_report(
            previous,
            started.elapsed().as_millis(),
        ));
    }
    Ok(captured.into_report(status))
}

async fn find_text(
    scope: CaptureScope,
    query: String,
    exact: bool,
    languages: Vec<String>,
) -> Result<String, JsonRpcErr> {
    call(move || {
        let (captured, ocr) = capture_ocr(scope, &languages)?;
        Ok(format_matches(&captured, &ocr, &query, exact))
    })
    .await
}

async fn click_text(
    scope: CaptureScope,
    query: String,
    languages: Vec<String>,
    button: String,
) -> Result<String, JsonRpcErr> {
    call(move || {
        after_activating_window_target(scope, backend::activate_window, || {
            let (captured, ocr) = capture_ocr(scope, &languages)?;
            let matches = matching_lines(&ocr.lines, &query, true);
            let line = if matches.len() == 1 {
                matches[0]
            } else {
                return Err(format!(
                    "text query matched {} OCR lines; no click was sent\n{}",
                    matches.len(),
                    format_matches(&captured, &ocr, &query, true)
                ));
            };
            let mapping = captured.mapping().ok_or_else(|| {
                "the capture has no reliable image-to-screen mapping; no click was sent"
                    .to_string()
            })?;
            let center_x = f64::from(line.bounds.x) + f64::from(line.bounds.width) / 2.0;
            let center_y = f64::from(line.bounds.y) + f64::from(line.bounds.height) / 2.0;
            let (screen_x, screen_y) = mapping.pixel_to_screen(center_x, center_y);
            // OCR can take long enough for another application to steal focus. Recheck
            // the exact target after OCR so a window-scoped frame cannot click another app.
            activate_window_target(scope, backend::activate_window)?;
            backend::click(screen_x, screen_y, &button, 1, &[])?;
            Ok(format!(
                "status=clicked\ntext={}\nscreen_center={screen_x:.2},{screen_y:.2}\npixel_bounds={},{},{},{}",
                serde_json::to_string(&line.text).unwrap_or_else(|_| "\"\"".to_string()),
                line.bounds.x,
                line.bounds.y,
                line.bounds.width,
                line.bounds.height
            ))
        })
    })
    .await
}

fn after_activating_window_target<T>(
    scope: CaptureScope,
    activate_window: impl FnOnce(u32) -> Result<String, String>,
    action: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    activate_window_target(scope, activate_window)?;
    action()
}

fn activate_window_target(
    scope: CaptureScope,
    activate_window: impl FnOnce(u32) -> Result<String, String>,
) -> Result<(), String> {
    if let Some(window_id) = scope.window_id() {
        activate_window(window_id)?;
    }
    Ok(())
}

fn capture_ocr(
    scope: CaptureScope,
    languages: &[String],
) -> Result<(CapturedFrame, OcrResult), String> {
    let captured = CapturedFrame::capture(scope)?;
    let image = image::open(captured.path())
        .map_err(|error| format!("could not decode captured image for OCR: {error}"))?;
    let languages = languages.iter().map(String::as_str).collect::<Vec<_>>();
    let ocr = ocr_image_with_regions(&image, &languages)?;
    Ok((captured, ocr))
}

fn matching_lines<'a>(lines: &'a [OcrLine], query: &str, exact: bool) -> Vec<&'a OcrLine> {
    let query = query.to_lowercase();
    lines
        .iter()
        .filter(|line| {
            let text = line.text.to_lowercase();
            if exact {
                text == query
            } else {
                text.contains(&query)
            }
        })
        .collect()
}

fn format_matches(captured: &CapturedFrame, ocr: &OcrResult, query: &str, exact: bool) -> String {
    let matches = matching_lines(&ocr.lines, query, exact);
    let mut report = format!(
        "status=found\nquery={}\nmatch_count={}\nimage_size={}x{}",
        serde_json::to_string(query).unwrap_or_else(|_| "\"\"".to_string()),
        matches.len(),
        ocr.image_width,
        ocr.image_height
    );
    for (index, line) in matches.into_iter().enumerate() {
        use std::fmt::Write as _;

        let _ = write!(
            report,
            "\nmatch[{index}].text={}\nmatch[{index}].pixel_bounds={},{},{},{}",
            serde_json::to_string(&line.text).unwrap_or_else(|_| "\"\"".to_string()),
            line.bounds.x,
            line.bounds.y,
            line.bounds.width,
            line.bounds.height
        );
        if let Some(mapping) = captured.mapping() {
            let center_x = f64::from(line.bounds.x) + f64::from(line.bounds.width) / 2.0;
            let center_y = f64::from(line.bounds.y) + f64::from(line.bounds.height) / 2.0;
            let (screen_x, screen_y) = mapping.pixel_to_screen(center_x, center_y);
            let _ = write!(
                report,
                "\nmatch[{index}].screen_center={screen_x:.2},{screen_y:.2}"
            );
        } else {
            let _ = write!(report, "\nmatch[{index}].screen_center=unavailable");
        }
    }
    report
}

async fn wait_for_change(
    previous: FrameFingerprint,
    timeout_ms: u32,
    poll_ms: u32,
) -> Result<String, JsonRpcErr> {
    let scope = previous.scope();
    with_timeout(op_timeout_ms(), async move {
        let started = Instant::now();
        let deadline = started + Duration::from_millis(u64::from(timeout_ms));
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Ok(frame::unchanged_report(
                    &previous,
                    started.elapsed().as_millis(),
                ));
            }
            let captured = match tokio::time::timeout(
                deadline - now,
                run_blocking(move || CapturedFrame::capture(scope)),
            )
            .await
            {
                Ok(result) => result?,
                Err(_) => {
                    return Ok(frame::unchanged_report(
                        &previous,
                        started.elapsed().as_millis(),
                    ));
                }
            };
            if captured.fingerprint() != &previous {
                return Ok(captured.into_report("changed"));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(frame::unchanged_report(
                    &previous,
                    started.elapsed().as_millis(),
                ));
            }
            tokio::time::sleep(Duration::from_millis(u64::from(poll_ms)).min(deadline - now)).await;
        }
    })
    .await
}

fn tools() -> Value {
    json!([
        {
            "name": "screen_info",
            "description": "Start every desktop task here, including browser tasks. Report displays with their point size and scale, the cursor position, the frontmost application, on-screen windows, and required permissions. This is the coordinate reference for other tools — click/move_mouse take points in global display space, while screenshot files contain pixels.",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "list_windows",
            "description": "List the on-screen windows with their window id, owning application and frame in points. `app` filters by a case-insensitive substring of the owning application's name. A window id can be passed to `screenshot` to capture that window alone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": { "type": "string", "description": "Filter by owning application name (case-insensitive substring)." }
                },
                "required": []
            }
        },
        {
            "name": "activate_app",
            "description": "Bring an application to the front so it receives keyboard input. For browser work, activate the browser, use press_key to focus its address bar, then type_text and press Return. `app` is the application name from list_windows/screen_info; a missing application is an error.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": { "type": "string", "description": "Application name to bring to the front." }
                },
                "required": ["app"]
            }
        },
        {
            "name": "activate_window",
            "description": "Bring one exact on-screen window to the front and verify it before sending keyboard or pointer input. Use the window id from list_windows; a stale, minimized, unsupported, or unverified window is an error.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "window_id": { "type": "integer", "description": "Exact window id from list_windows." }
                },
                "required": ["window_id"]
            }
        },
        {
            "name": "screenshot",
            "description": "Observe any desktop or browser UI by capturing the desktop — or one region, window or display — to a PNG. Pass the returned path to read_file only when status is captured/changed. Reuse the returned frame_id as previous_frame_id after an action; an unchanged frame is discarded and must not be read again. `region` is [x, y, width, height] in points. Pass at most one of region, window_id, and display.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "[x, y, width, height] in global display points; omit for the whole desktop."
                    },
                    "window_id": { "type": "integer", "description": "Capture only this window (id from list_windows)." },
                    "display": { "type": "integer", "description": "Capture only this display (id from screen_info)." },
                    "include_cursor": { "type": "boolean", "description": "Draw the mouse pointer into the capture (default false)." },
                    "previous_frame_id": { "type": "string", "description": "frame_id from the preceding screenshot of the exact same scope; unchanged pixels avoid another read_file call." }
                },
                "required": []
            }
        },
        {
            "name": "find_text",
            "description": "Capture the current UI, run OCR, and return matching text lines with source-pixel bounds and actionable global screen-point centers. This avoids a screenshot/read_file round trip when the target is visible text. Pass at most one of region, window_id, and display.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Non-empty text to find (case-insensitive)." },
                    "exact": { "type": "boolean", "description": "Require the whole OCR line to equal text; default false uses substring matching." },
                    "languages": { "type": "array", "items": { "type": "string" }, "description": "Optional OCR language hints such as [\"zh-Hans\", \"en-US\"]." },
                    "region": { "type": "array", "items": { "type": "number" }, "description": "[x, y, width, height] in global display points." },
                    "window_id": { "type": "integer", "description": "Search only this window (id from list_windows)." },
                    "display": { "type": "integer", "description": "Search only this display (id from screen_info)." }
                },
                "required": ["text"]
            }
        },
        {
            "name": "click_text",
            "description": "Capture the current UI, OCR it, and click the center only when exactly one complete OCR line equals the query (case-insensitive). With window_id, the exact window is activated and verified before a fresh capture; activation failure sends no click. Zero or multiple matches are reported without clicking; use find_text for background observation and substring discovery.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Non-empty text to click (case-insensitive)." },
                    "languages": { "type": "array", "items": { "type": "string" }, "description": "Optional OCR language hints." },
                    "button": { "type": "string", "enum": ["left", "right"], "description": "Mouse button (default left)." },
                    "region": { "type": "array", "items": { "type": "number" }, "description": "[x, y, width, height] in global display points." },
                    "window_id": { "type": "integer", "description": "Activate, verify, then search only this window (id from list_windows)." },
                    "display": { "type": "integer", "description": "Search only this display (id from screen_info)." }
                },
                "required": ["text"]
            }
        },
        {
            "name": "move_mouse",
            "description": "Move the pointer to (x, y) in global display points without clicking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Global display point x." },
                    "y": { "type": "number", "description": "Global display point y." }
                },
                "required": ["x", "y"]
            }
        },
        {
            "name": "click",
            "description": "Click at (x, y) in global display points. `button` is `left` (default) or `right`; `count` 1 = single, 2 = double, 3 = triple click; `modifiers` are held down for the click, e.g. [\"cmd\"] or [\"shift\", \"alt\"].",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Global display point x." },
                    "y": { "type": "number", "description": "Global display point y." },
                    "button": { "type": "string", "enum": ["left", "right"], "description": "Mouse button (default left)." },
                    "count": { "type": "integer", "minimum": 1, "maximum": 3, "description": "Number of clicks in one gesture (default 1; 2 = double click)." },
                    "modifiers": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Modifiers held for the click, e.g. [\"cmd\"]."
                    }
                },
                "required": ["x", "y"]
            }
        },
        {
            "name": "drag",
            "description": "Press `button` at (from_x, from_y), move while held to (to_x, to_y) in `steps` increments, then release. It is delivered as a path because selection-tracking targets (text selections, sliders, canvas tools, drag-and-drop) ignore a single jump. All coordinates are global display points.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_x": { "type": "number", "description": "Global display point x to press at." },
                    "from_y": { "type": "number", "description": "Global display point y to press at." },
                    "to_x": { "type": "number", "description": "Global display point x to release at." },
                    "to_y": { "type": "number", "description": "Global display point y to release at." },
                    "button": { "type": "string", "enum": ["left", "right"], "description": "Mouse button (default left)." },
                    "modifiers": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Modifiers held for the drag, e.g. [\"shift\"] to extend a selection."
                    },
                    "steps": { "type": "integer", "description": "Intermediate positions between press and release (default 20)." }
                },
                "required": ["from_x", "from_y", "to_x", "to_y"]
            }
        },
        {
            "name": "scroll",
            "description": "Scroll the view under the pointer, optionally moving the pointer to (x, y) first — wheel events go to whatever is under the pointer, so giving both coordinates is what makes the target explicit. `dy` is vertical and `dx` horizontal, with positive `dy` scrolling up (content moves down); `unit` is `line` (default) or `pixel`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Move the pointer here first (global display point x)." },
                    "y": { "type": "number", "description": "Move the pointer here first (global display point y)." },
                    "dx": { "type": "number", "description": "Horizontal amount (default 0)." },
                    "dy": { "type": "number", "description": "Vertical amount; positive scrolls up." },
                    "unit": { "type": "string", "enum": ["line", "pixel"], "description": "Scroll unit (default line)." }
                },
                "required": []
            }
        },
        {
            "name": "type_text",
            "description": "Type `text` into whatever currently holds keyboard focus, exactly as written — Unicode-safe, with no keyboard layout involved. It does not press Return and does not choose a target, so activate_app and click the field first. The typed text is not echoed back in the reply, so a typed secret does not land in the transcript.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to type into the focused field." }
                },
                "required": ["text"]
            }
        },
        {
            "name": "press_key",
            "description": "Press a key or a `+`-joined chord, e.g. `escape`, `return`, `f5`, `cmd+shift+t`. Modifier names (cmd, shift, alt/option, ctrl) come first and the key last; an unknown key name is rejected with the accepted spellings instead of pressing nothing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "Key or chord, e.g. `escape` or `cmd+shift+t`." }
                },
                "required": ["key"]
            }
        },
        {
            "name": "wait",
            "description": "Pause briefly while an application or webpage updates, then take a fresh screenshot before acting. This is a platform-neutral UI settle, not a selector or DOM wait.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "duration_ms": { "type": "integer", "minimum": 1, "maximum": 30000, "description": "How long to pause in milliseconds (maximum 30000)." }
                },
                "required": ["duration_ms"]
            }
        },
        {
            "name": "wait_for_change",
            "description": "Poll the same capture scope encoded in frame_id until its visible pixels change or timeout_ms elapses. Probe images are deleted. On status=changed, read the returned path; on status=unchanged, reuse the prior visual state and do not call read_file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "frame_id": { "type": "string", "description": "Baseline frame_id returned by screenshot." },
                    "timeout_ms": { "type": "integer", "minimum": 0, "maximum": 30000, "description": "Observation budget in milliseconds (default 10000)." },
                    "poll_interval_ms": { "type": "integer", "minimum": 100, "maximum": 5000, "description": "Delay between probes in milliseconds (default 500)." }
                },
                "required": ["frame_id"]
            }
        }
    ])
}

fn invalid(message: &str) -> JsonRpcErr {
    JsonRpcErr::new(-32602, message, None)
}

fn required_string(args: &Value, key: &str) -> Result<String, JsonRpcErr> {
    optional_string(args, key)?.ok_or_else(|| invalid(&format!("missing or empty '{key}'")))
}

fn optional_string(args: &Value, key: &str) -> Result<Option<String>, JsonRpcErr> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(Some(text.clone())),
        Some(Value::String(_)) => Err(invalid(&format!("missing or empty '{key}'"))),
        Some(other) => Err(invalid(&format!("'{key}' must be a string, got {other}"))),
    }
}

fn required_frame_id(args: &Value, key: &str) -> Result<FrameFingerprint, JsonRpcErr> {
    optional_frame_id(args, key)?.ok_or_else(|| invalid(&format!("missing or empty '{key}'")))
}

fn optional_frame_id(args: &Value, key: &str) -> Result<Option<FrameFingerprint>, JsonRpcErr> {
    optional_string(args, key)?
        .map(|token| FrameFingerprint::from_token(&token).map_err(|error| invalid(&error)))
        .transpose()
}

fn required_f64(args: &Value, key: &str) -> Result<f64, JsonRpcErr> {
    optional_f64(args, key)?.ok_or_else(|| invalid(&format!("missing '{key}'")))
}

fn required_u32(args: &Value, key: &str) -> Result<u32, JsonRpcErr> {
    optional_u32(args, key)?.ok_or_else(|| invalid(&format!("missing '{key}'")))
}

fn optional_f64(args: &Value, key: &str) -> Result<Option<f64>, JsonRpcErr> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(&format!("'{key}' must be a number, got {value}"))),
    }
}

fn optional_u32(args: &Value, key: &str) -> Result<Option<u32>, JsonRpcErr> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|number| u32::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| {
                invalid(&format!(
                    "'{key}' must be a non-negative integer, got {value}"
                ))
            }),
    }
}

fn optional_bool(args: &Value, key: &str) -> Result<Option<bool>, JsonRpcErr> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| invalid(&format!("'{key}' must be true or false, got {value}"))),
    }
}

fn string_list(args: &Value, key: &str) -> Result<Vec<String>, JsonRpcErr> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid(&format!("'{key}' must be an array of strings")))
            })
            .collect(),
        Some(_) => Err(invalid(&format!("'{key}' must be an array of strings"))),
    }
}

/// Read `region` as `[x, y, width, height]` in points.
fn optional_region(args: &Value) -> Result<Option<(f64, f64, f64, f64)>, JsonRpcErr> {
    match args.get("region") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) if items.len() == 4 => {
            let mut numbers = [0.0_f64; 4];
            for (slot, item) in numbers.iter_mut().zip(items) {
                *slot = item
                    .as_f64()
                    .ok_or_else(|| invalid("'region' must be [x, y, width, height] in points"))?;
            }
            Ok(Some((numbers[0], numbers[1], numbers[2], numbers[3])))
        }
        Some(_) => Err(invalid("'region' must be [x, y, width, height] in points")),
    }
}

fn capture_scope(args: &Value) -> Result<CaptureScope, JsonRpcErr> {
    CaptureScope::from_arguments(
        optional_region(args)?,
        optional_u32(args, "window_id")?,
        optional_u32(args, "display")?,
        optional_bool(args, "include_cursor")?.unwrap_or(false),
    )
    .map_err(|error| invalid(&error))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;
    use rust_tools::pdfw::OcrRect;

    #[tokio::test]
    async fn zero_change_budget_returns_without_capturing() {
        let previous = FrameFingerprint::from_token(&format!(
            "mcp-computer-frame-v1|desktop|cursor=0|1x1|{}",
            "00".repeat(32)
        ))
        .unwrap();

        let report = match wait_for_change(previous, 0, 100).await {
            Ok(report) => report,
            Err(_) => panic!("a zero-budget wait should return without capturing"),
        };
        assert!(report.starts_with("status=unchanged\n"));
    }

    #[test]
    fn text_matching_is_case_insensitive_and_exposes_ambiguity() {
        let lines = vec![
            OcrLine {
                text: "Open settings".to_string(),
                bounds: OcrRect {
                    x: 0,
                    y: 0,
                    width: 20,
                    height: 10,
                },
            },
            OcrLine {
                text: "Settings help".to_string(),
                bounds: OcrRect {
                    x: 30,
                    y: 0,
                    width: 20,
                    height: 10,
                },
            },
        ];
        assert_eq!(matching_lines(&lines, "SETTINGS", false).len(), 2);
        assert_eq!(matching_lines(&lines, "open settings", true).len(), 1);
    }

    #[test]
    fn window_target_is_activated_before_action() {
        let scope = CaptureScope::from_arguments(None, Some(42), None, false).unwrap();
        let events = RefCell::new(Vec::new());

        let result = after_activating_window_target(
            scope,
            |window_id| {
                events.borrow_mut().push(format!("activate:{window_id}"));
                Ok(String::new())
            },
            || {
                events.borrow_mut().push("action".to_string());
                Ok(7)
            },
        )
        .unwrap();

        assert_eq!(result, 7);
        assert_eq!(&*events.borrow(), &["activate:42", "action"]);
    }

    #[test]
    fn activation_failure_prevents_window_action() {
        let scope = CaptureScope::from_arguments(None, Some(42), None, false).unwrap();
        let action_called = Cell::new(false);

        let error = after_activating_window_target(
            scope,
            |_| Err("exact window did not activate".to_string()),
            || {
                action_called.set(true);
                Ok(())
            },
        )
        .unwrap_err();

        assert_eq!(error, "exact window did not activate");
        assert!(!action_called.get());
    }

    #[test]
    fn desktop_action_does_not_activate_a_window() {
        let scope = CaptureScope::from_arguments(None, None, None, false).unwrap();
        let activation_called = Cell::new(false);

        let result = after_activating_window_target(
            scope,
            |_| {
                activation_called.set(true);
                Ok(String::new())
            },
            || Ok(9),
        )
        .unwrap();

        assert_eq!(result, 9);
        assert!(!activation_called.get());
    }
}
