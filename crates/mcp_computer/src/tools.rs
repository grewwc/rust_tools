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

use mcp_stdio::{JsonRpcErr, McpServer, cap_text, text_content, with_timeout};
use serde_json::{Value, json};

use crate::backend;

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
            "screenshot" => {
                let region = optional_region(&args)?;
                let window_id = optional_u32(&args, "window_id")?;
                let display = optional_u32(&args, "display")?;
                let cursor = optional_bool(&args, "include_cursor")?.unwrap_or(false);
                call(move || backend::screenshot(region, window_id, display, cursor)).await?
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
                let button = optional_string(&args, "button")?.unwrap_or_else(|| "left".to_string());
                let count = optional_u32(&args, "count")?.unwrap_or(1).max(1);
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
                let button = optional_string(&args, "button")?.unwrap_or_else(|| "left".to_string());
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
                    backend::scroll(x, y, dx, dy, &unit).map(|()| {
                        format!("scrolled dx={dx} dy={dy} in {unit} units{at}")
                    })
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
async fn call<F>(job: F) -> Result<String, JsonRpcErr>
where
    F: FnOnce() -> Result<String, String> + Send + 'static,
{
    with_timeout(op_timeout_ms(), async move {
        match tokio::task::spawn_blocking(job).await {
            Ok(report) => report,
            // The blocking pool lost the job. Reported as an ordinary tool error, in
            // wording clear of the host's transport trigger phrases.
            Err(error) => Err(format!("the platform call did not run to completion: {error}")),
        }
    })
    .await
}

fn tools() -> Value {
    json!([
        {
            "name": "screen_info",
            "description": "Report the current desktop state: displays with their point size and scale, the cursor position, the frontmost application, and the on-screen windows. This is the coordinate reference for the other tools — click/move_mouse take points in the global display space, while a screenshot file is `point size × scale` pixels, so divide pixel positions read off a capture by the display scale before clicking them.",
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
            "description": "Bring an application to the front so it receives keyboard input. `app` is the application name as it appears in list_windows/screen_info; a name that matches no application is reported as an error instead of being ignored.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": { "type": "string", "description": "Application name to bring to the front." }
                },
                "required": ["app"]
            }
        },
        {
            "name": "screenshot",
            "description": "Capture the desktop — or one region, window or display — to a PNG on disk and return its absolute path. Pass that path to read_file to actually look at it. `region` is [x, y, width, height] in points; `window_id` (from list_windows) and `display` (from screen_info) narrow the capture; `include_cursor` draws the pointer into the image. Positions read off the returned image are pixels, so divide them by the display scale from screen_info before using them as click coordinates.",
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
                    "include_cursor": { "type": "boolean", "description": "Draw the mouse pointer into the capture (default false)." }
                },
                "required": []
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
                    "count": { "type": "integer", "description": "Number of clicks in one gesture (default 1; 2 = double click)." },
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

fn required_f64(args: &Value, key: &str) -> Result<f64, JsonRpcErr> {
    optional_f64(args, key)?.ok_or_else(|| invalid(&format!("missing '{key}'")))
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
            .ok_or_else(|| invalid(&format!("'{key}' must be a non-negative integer, got {value}"))),
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
                *slot = item.as_f64().ok_or_else(|| {
                    invalid("'region' must be [x, y, width, height] in points")
                })?;
            }
            Ok(Some((numbers[0], numbers[1], numbers[2], numbers[3])))
        }
        Some(_) => Err(invalid("'region' must be [x, y, width, height] in points")),
    }
}