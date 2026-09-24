//! Linux backend: the session's own command-line helpers, one thin wrapper each.
//!
//! There is no kernel API for "move the pointer" that is stable across session
//! types, so the honest implementation is a driver per session type:
//!
//! - **X11** — `xdotool` (XTEST): pointer, buttons, scroll, typing, key chords,
//!   window search/activation. `wmctrl` adds window frames and titles.
//! - **Wayland** — `ydotool` (uinput) for input, `wtype` (virtual keyboard) for
//!   text; `grim` captures on wlroots, while window enumeration has no portable
//!   answer and is reported as unavailable rather than guessed.
//!
//! Follows the platform contract in `src/backend.rs`: query functions return a
//! model-facing report, actions return `Ok(())` or a model-facing error string, and
//! anything platform-specific that is missing (helper binary, `DISPLAY`, a capture
//! portal) is discovered at run time and named in the message. A parameter this
//! backend cannot honour — a window id without an X11 capture tool, the `pixel`
//! scroll unit — is reported as an error, never silently dropped.
//!
//! This module is self-contained (std only, no `use super::*`) so it can be
//! type-checked on its own with `rustc --edition 2024 --crate-type lib --emit=metadata`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Capture helpers, most capable first. `grim` is Wayland-only, the other two are
/// X11 (ImageMagick's `import` also works on XWayland).
const CAPTURE_TOOLS: [&str; 3] = ["grim", "scrot", "import"];
/// Input helpers, most capable first.
const INPUT_TOOLS: [&str; 2] = ["xdotool", "ydotool"];
/// Window list helpers, most capable first.
const WINDOW_TOOLS: [&str; 2] = ["wmctrl", "xdotool"];
/// Upper bound on rows returned by `list_windows`, so one call cannot flood the
/// model with a hundred windows.
const MAX_WINDOWS: usize = 40;
/// Upper bound on drag interpolation steps, matching the macOS backend's clamp.
const MAX_DRAG_STEPS: u32 = 200;

/// Phrases the host treats as transport failures (`src/bin/ai/mcp/client.rs`).
/// Helper stderr is forwarded to the model, so it is scanned first: forwarding one
/// of these words would make the host kill and restart this server mid-conversation.
/// Kept local, like the macOS backend's copy, so each platform module stands alone.
const TRANSPORT_TRIGGERS: [&str; 6] = [
    "mcp response timeout",
    "broken pipe",
    "closed the stream",
    "process exited",
    "failed to read response",
    "failed waiting for mcp response",
];

fn sanitize(text: &str) -> String {
    let lowered = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        match TRANSPORT_TRIGGERS
            .iter()
            .find(|trigger| lowered[index..].starts_with(**trigger))
        {
            Some(trigger) => {
                out.push_str("transport-error");
                index += trigger.len();
            }
            None => {
                let Some(character) = text[index..].chars().next() else {
                    break;
                };
                out.push(character);
                index += character.len_utf8();
            }
        }
    }
    out
}

/// First helper of `names` that is actually executable, in preference order.
fn find_helper(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|name| which(name))
}

fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path).map(|meta| meta.is_file()).unwrap_or(false)
}

/// The helper to use, or an error naming what to install. `hint` is the package
/// hint appended so the model can tell the user what a session would need.
fn helper(names: &[&str], hint: &str) -> Result<PathBuf, String> {
    find_helper(names).ok_or_else(|| {
        format!(
            "none of {} is available on PATH ({hint})",
            names.join(", ")
        )
    })
}

/// Run a helper and return its trimmed stdout. `what` names the operation in the
/// error, and stderr is passed through [`sanitize`] before it reaches the model.
fn run(program: &Path, args: &[String], what: &str) -> Result<String, String> {
    let output = Command::new(program).args(args).output().map_err(|error| {
        format!(
            "could not run {} for {what}: {error}",
            program.display()
        )
    })?;
    if !output.status.success() {
        let stderr = sanitize(String::from_utf8_lossy(&output.stderr).trim());
        let detail = if stderr.is_empty() {
            format!("status {:?}", output.status.code())
        } else {
            stderr
        };
        return Err(format!(
            "{} failed for {what}: {detail} — on X11 this normally means DISPLAY is \
             not set for this process; on Wayland the helper may need a running \
             input-remapper or the session's permission to use uinput",
            program.display()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn session_is_wayland() -> bool {
    env::var_os("WAYLAND_DISPLAY").is_some()
}

/// Coordinates are sent as integers: every helper here takes whole pixels.
fn whole(value: f64) -> i64 {
    value.round() as i64
}

/// `ydotool` argument for a pointer button, or an error for anything else the
/// helper cannot express.
fn ydotool_button(button: &str) -> Result<&'static str, String> {
    match button.to_ascii_lowercase().as_str() {
        "left" => Ok("0xC0"),
        "right" => Ok("0xC1"),
        "middle" => Ok("0xC2"),
        other => Err(format!(
            "unknown mouse button `{other}`; expected `left`, `right` or `middle`"
        )),
    }
}

/// `xdotool` button number for a pointer button.
fn xdotool_button(button: &str) -> Result<&'static str, String> {
    match button.to_ascii_lowercase().as_str() {
        "left" => Ok("1"),
        "middle" => Ok("2"),
        "right" => Ok("3"),
        other => Err(format!(
            "unknown mouse button `{other}`; expected `left`, `right` or `middle`"
        )),
    }
}

fn modifier_names(modifiers: &[String]) -> Vec<String> {
    modifiers.iter().map(|name| (*name).to_string()).collect()
}

/// Chord in the spelling `xdotool key` expects: `cmd` is `super` on Linux, and
/// `option` is `alt`.
fn xdotool_chord(chord: &str) -> Result<String, String> {
    let parts: Vec<String> = chord
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "meta" | "win" => "super".to_string(),
            "option" => "alt".to_string(),
            other => other.to_string(),
        })
        .collect();
    if parts.is_empty() {
        return Err("press_key needs a key, e.g. `escape` or `ctrl+shift+t`".to_string());
    }
    Ok(parts.join("+"))
}

pub(crate) fn move_mouse(x: f64, y: f64) -> Result<(), String> {
    if let Some(tool) = find_helper(&["xdotool"]) {
        return run(
            &tool,
            &["mousemove".to_string(), whole(x).to_string(), whole(y).to_string()],
            "moving the pointer",
        )
        .map(|_| ());
    }
    let tool = helper(&["ydotool"], "install ydotool for Wayland input")?;
    run(
        &tool,
        &[
            "mousemove".to_string(),
            "-a".to_string(),
            whole(x).to_string(),
            whole(y).to_string(),
        ],
        "moving the pointer",
    )
    .map(|_| ())
}

pub(crate) fn click(
    x: f64,
    y: f64,
    button: &str,
    count: u32,
    modifiers: &[String],
) -> Result<(), String> {
    move_mouse(x, y)?;
    std::thread::sleep(std::time::Duration::from_millis(30));
    let count = count.max(1);
    if let Some(tool) = find_helper(&["xdotool"]) {
        let mut args = vec![
            "click".to_string(),
            "--repeat".to_string(),
            count.to_string(),
            "--delay".to_string(),
            "60".to_string(),
        ];
        // `xdotool` has no "hold this modifier while clicking" flag, so the chord is
        // pressed around the click and released afterwards; a failure to release is
        // reported rather than left implicit.
        let held = modifier_names(modifiers);
        for modifier in &held {
            args.insert(0, format!("keydown+{}", xdotool_chord(modifier)?));
        }
        args.push(xdotool_button(button)?.to_string());
        run(&tool, &args, "clicking")?;
        for modifier in &held {
            run(
                &tool,
                &["keyup+".to_string(), xdotool_chord(modifier)?],
                "releasing a held modifier",
            )?;
        }
        return Ok(());
    }
    if !modifiers.is_empty() {
        return Err(
            "holding modifier keys during a click needs xdotool; ydotool cannot \
             express it"
                .to_string(),
        );
    }
    let tool = helper(&["ydotool"], "install ydotool for Wayland input")?;
    let button = ydotool_button(button)?;
    for _ in 0..count {
        run(&tool, &["click".to_string(), button.to_string()], "clicking")?;
    }
    Ok(())
}

pub(crate) fn drag(
    from_x: f64,
    from_y: f64,
    to_x: f64,
    to_y: f64,
    button: &str,
    modifiers: &[String],
    steps: u32,
) -> Result<(), String> {
    let tool = helper(
        &["xdotool", "ydotool"],
        "install xdotool (X11) or ydotool (Wayland) for pointer dragging",
    )?;
    let is_xdotool = tool
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("xdotool"))
        .unwrap_or(false);
    if !is_xdotool {
        return Err(
            "dragging needs xdotool: ydotool has no held-button drag that \
             interpolates between two points"
                .to_string(),
        );
    }
    let button = xdotool_button(button)?;
    for modifier in modifier_names(modifiers) {
        run(
            &tool,
            &["keydown".to_string(), xdotool_chord(&modifier)?],
            "holding a modifier for the drag",
        )?;
    }
    move_mouse(from_x, from_y)?;
    run(
        &tool,
        &["mousedown".to_string(), button.to_string()],
        "pressing the button",
    )?;
    // A drag has to be delivered as a path: selection-tracking targets ignore a
    // single jump from start to end.
    let steps = steps.clamp(2, MAX_DRAG_STEPS);
    for step in 1..=steps {
        let t = f64::from(step) / f64::from(steps);
        let x = from_x + (to_x - from_x) * t;
        let y = from_y + (to_y - from_y) * t;
        run(
            &tool,
            &["mousemove".to_string(), whole(x).to_string(), whole(y).to_string()],
            "dragging",
        )?;
    }
    run(
        &tool,
        &["mouseup".to_string(), button.to_string()],
        "releasing the button",
    )?;
    for modifier in modifier_names(modifiers) {
        run(
            &tool,
            &["keyup".to_string(), xdotool_chord(&modifier)?],
            "releasing a held modifier",
        )?;
    }
    Ok(())
}

pub(crate) fn scroll(
    x: Option<f64>,
    y: Option<f64>,
    dx: f64,
    dy: f64,
    unit: &str,
) -> Result<(), String> {
    if let (Some(x), Some(y)) = (x, y) {
        move_mouse(x, y)?;
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    if unit.eq_ignore_ascii_case("pixel") {
        return Err(
            "the `pixel` scroll unit is not available on Linux: xdotool and ydotool \
             both scroll in wheel clicks, so pass `line` and a click count"
                .to_string(),
        );
    }
    if !unit.eq_ignore_ascii_case("line") && !unit.eq_ignore_ascii_case("lines") {
        return Err(format!(
            "unknown scroll unit `{unit}`; expected `line` or `pixel`"
        ));
    }
    let tool = helper(
        &["xdotool", "ydotool"],
        "install xdotool (X11) or ydotool (Wayland) for scrolling",
    )?;
    let is_xdotool = tool
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("xdotool"))
        .unwrap_or(false);
    // X11 wheel buttons: 4 up, 5 down, 6 left, 7 right.
    let vertical = if dy > 0.0 { "4" } else { "5" };
    let horizontal = if dx > 0.0 { "6" } else { "7" };
    let mut clicks = Vec::new();
    if dy != 0.0 {
        clicks.push(vertical.to_string());
    }
    if dx != 0.0 {
        clicks.push(horizontal.to_string());
    }
    if clicks.is_empty() {
        return Err("scroll needs a non-zero `dx` or `dy`".to_string());
    }
    let repeat = whole(dy.abs().max(dx.abs())).max(1);
    if is_xdotool {
        for button in clicks {
            run(
                &tool,
                &[
                    "click".to_string(),
                    "--repeat".to_string(),
                    repeat.to_string(),
                    "--delay".to_string(),
                    "20".to_string(),
                    button,
                ],
                "scrolling",
            )?;
        }
        return Ok(());
    }
    // ydotool takes wheel events as key codes, which are not portable enough to
    // hardcode: report the limitation instead of sending the wrong event.
    Err(
        "scrolling needs xdotool; ydotool wheel events require machine-specific key \
         codes"
            .to_string(),
    )
}

pub(crate) fn type_text(text: &str) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    if let Some(tool) = find_helper(&["xdotool"]) {
        return run(
            &tool,
            &[
                "type".to_string(),
                "--delay".to_string(),
                "8".to_string(),
                "--clearmodifiers".to_string(),
                "--".to_string(),
                text.to_string(),
            ],
            "typing text",
        )
        .map(|_| ());
    }
    let tool = helper(
        &["wtype", "ydotool"],
        "install wtype or ydotool for Wayland text input",
    )?;
    if tool
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("wtype"))
        .unwrap_or(false)
    {
        return run(&tool, &["-".to_string(), text.to_string()], "typing text").map(|_| ());
    }
    run(&tool, &["type".to_string(), text.to_string()], "typing text").map(|_| ())
}

pub(crate) fn press_key(chord: &str) -> Result<(), String> {
    if let Some(tool) = find_helper(&["xdotool"]) {
        return run(
            &tool,
            &[
                "key".to_string(),
                "--clearmodifiers".to_string(),
                xdotool_chord(chord)?,
            ],
            "pressing a key",
        )
        .map(|_| ());
    }
    let tool = helper(
        &["wtype", "ydotool"],
        "install wtype or ydotool for Wayland key presses",
    )?;
    let is_wtype = tool
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("wtype"))
        .unwrap_or(false);
    if !is_wtype {
        return Err(
            "pressing named keys needs xdotool or wtype; ydotool takes raw key codes"
                .to_string(),
        );
    }
    // wtype spells modifiers as flags, and the key itself as the trailing argument.
    let parts: Vec<String> = chord
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.to_ascii_lowercase())
        .collect();
    let Some((key, modifiers)) = parts.split_last() else {
        return Err("press_key needs a key, e.g. `escape` or `ctrl+shift+t`".to_string());
    };
    let mut args = Vec::new();
    for modifier in modifiers {
        match modifier.as_str() {
            "ctrl" | "control" => args.push("-M".to_string()),
            "shift" => args.push("-M".to_string()),
            "alt" | "option" => args.push("-M".to_string()),
            "cmd" | "super" | "meta" | "win" => args.push("-M".to_string()),
            other => {
                return Err(format!(
                    "unknown modifier `{other}`; expected ctrl, shift, alt or cmd"
                ));
            }
        }
    }
    args.push(key.clone());
    run(&tool, &args, "pressing a key").map(|_| ())
}

fn screenshot_path() -> Result<PathBuf, String> {
    let dir = env::var("MCP_COMPUTER_SCREENSHOT_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".cache/mcp_computer"))
        })
        .ok_or_else(|| "HOME is not set, so there is nowhere to write the capture".to_string())?;
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("could not create {}: {error}", dir.display()))?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0);
    Ok(dir.join(format!("screen-{millis}.png")))
}

pub(crate) fn screenshot(
    region: Option<(f64, f64, f64, f64)>,
    window_id: Option<u32>,
    display: Option<u32>,
    include_cursor: bool,
) -> Result<String, String> {
    let path = screenshot_path()?;
    if let Some((_, _, width, height)) = region {
        if width <= 0.0 || height <= 0.0 {
            return Err("the screenshot region needs a positive width and height".to_string());
        }
    }
    // `display` has no meaning on Linux: the session exposes one screen (multi-monitor
    // setups differ per compositor), so it is refused rather than mapped to a guess.
    if display.is_some() {
        return Err(
            "the `display` parameter is not available on Linux; capture the whole \
             session or pass a `region`"
                .to_string(),
        );
    }
    let tool = helper(
        &CAPTURE_TOOLS,
        "install grim (Wayland), scrot or imagemagick (X11) to capture the screen",
    )?;
    let name = tool
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_string();
    let path_text = path.to_string_lossy().to_string();
    if name.starts_with("grim") {
        let mut args = Vec::new();
        if let Some((x, y, width, height)) = region {
            args.push("-g".to_string());
            args.push(format!(
                "{},{} {}x{}",
                whole(x),
                whole(y),
                whole(width),
                whole(height)
            ));
        }
        args.push(path_text);
        run(&tool, &args, "capturing the screen")?;
        return Ok(capture_report(&path, include_cursor));
    }
    if name.starts_with("scrot") {
        let mut args = Vec::new();
        if include_cursor {
            // scrot's `-p` asks for the pointer to be included.
            args.push("-p".to_string());
        }
        if let Some((x, y, width, height)) = region {
            args.push("-a".to_string());
            args.push(format!(
                "{},{},{},{}",
                whole(x),
                whole(y),
                whole(width),
                whole(height)
            ));
        }
        args.push(path_text);
        run(&tool, &args, "capturing the screen")?;
        return Ok(capture_report(&path, include_cursor));
    }
    // ImageMagick's `import`: `-window <id>` captures a window, `-window root` the
    // whole screen, and `-crop` a region. It has no pointer option.
    if include_cursor {
        return Err(
            "the `include_cursor` flag is not available with ImageMagick's import; \
             install grim or scrot for a capture that shows the pointer"
                .to_string(),
        );
    }
    let mut args = Vec::new();
    if let Some(window_id) = window_id {
        args.push("-window".to_string());
        args.push(window_id.to_string());
    } else {
        args.push("-window".to_string());
        args.push("root".to_string());
    }
    if let Some((x, y, width, height)) = region {
        args.push("-crop".to_string());
        args.push(format!(
            "{}x{}+{}+{}",
            whole(width),
            whole(height),
            whole(x),
            whole(y)
        ));
    }
    args.push(path_text);
    run(&tool, &args, "capturing the screen")?;
    Ok(capture_report(&path, include_cursor))
}

fn capture_report(path: &Path, include_cursor: bool) -> String {
    let mut report = format!("captured the screen to {}", path.display());
    if !include_cursor {
        report.push_str("\n(the pointer is not part of this capture)");
    }
    report
}

pub(crate) fn screen_info() -> Result<String, String> {
    let mut report = String::new();
    match find_helper(&["xrandr"]) {
        Some(tool) => {
            let output = run(&tool, &["--query".to_string()], "listing displays")?;
            report.push_str("displays (xrandr):\n");
            for line in output.lines().filter(|line| line.contains(" connected")) {
                report.push_str(&format!("  {}\n", line.trim()));
            }
        }
        None => report.push_str("displays: xrandr is not available on PATH\n"),
    }
    if let Some(tool) = find_helper(&["xdotool"]) {
        match run(
            &tool,
            &["getmouselocation".to_string(), "--shell".to_string()],
            "reading the pointer position",
        ) {
            Ok(output) => {
                let x = field(&output, "X").unwrap_or_else(|| "?".to_string());
                let y = field(&output, "Y").unwrap_or_else(|| "?".to_string());
                report.push_str(&format!("cursor: x={x} y={y}\n"));
            }
            Err(error) => report.push_str(&format!("cursor: unavailable ({error})\n")),
        }
        match active_window_name(&tool) {
            Ok(Some(app)) => report.push_str(&format!("frontmost app: {app}\n")),
            Ok(None) => report.push_str("frontmost app: unavailable\n"),
            Err(error) => report.push_str(&format!("frontmost app: unavailable ({error})\n")),
        }
    } else {
        report.push_str("cursor: unavailable (xdotool is not on PATH)\n");
        report.push_str("frontmost app: unavailable (xdotool is not on PATH)\n");
    }
    match window_rows() {
        Ok(rows) => report.push_str(&format!("on-screen windows: {}\n", rows.len())),
        Err(error) => report.push_str(&format!("windows: unavailable ({error})\n")),
    }
    if session_is_wayland() {
        report.push_str(
            "session: Wayland — window enumeration and key chords depend on xdotool \
             (XWayland) or wtype being installed\n",
        );
    }
    Ok(report)
}

/// One `key=value` line of `xdotool --shell` output.
fn field(output: &str, key: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .map(|value| value.trim().to_string())
}

fn active_window_name(xdotool: &Path) -> Result<Option<String>, String> {
    let id = run(
        xdotool,
        &["getactivewindow".to_string()],
        "reading the active window",
    )?;
    if id.is_empty() {
        return Ok(None);
    }
    let name = run(
        xdotool,
        &[
            "getwindowname".to_string(),
            id.split_whitespace().next().unwrap_or_default().to_string(),
        ],
        "reading the active window's name",
    )?;
    Ok((!name.is_empty()).then_some(name))
}

/// One on-screen window: id, owning application, title and frame in pixels.
struct WindowRow {
    id: String,
    app: String,
    title: String,
    x: i64,
    y: i64,
    width: i64,
    height: i64,
}

fn window_rows() -> Result<Vec<WindowRow>, String> {
    if let Some(tool) = find_helper(&["wmctrl"]) {
        // `-l -G -x`: one line per window with desktop, id, geometry, class and title.
        let output = run(
            &tool,
            &["-l".to_string(), "-G".to_string(), "-x".to_string()],
            "listing windows",
        )?;
        let mut rows = Vec::new();
        for line in output.lines().take(MAX_WINDOWS) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 8 {
                continue;
            }
            let (Ok(x), Ok(y), Ok(width), Ok(height)) = (
                fields[2].parse::<i64>(),
                fields[3].parse::<i64>(),
                fields[4].parse::<i64>(),
                fields[5].parse::<i64>(),
            ) else {
                continue;
            };
            rows.push(WindowRow {
                id: fields[0].to_string(),
                app: fields[6].to_string(),
                title: fields[7..].join(" "),
                x,
                y,
                width,
                height,
            });
        }
        return Ok(rows);
    }
    let tool = helper(
        &WINDOW_TOOLS,
        "install wmctrl or xdotool to enumerate windows",
    )?;
    let output = run(
        &tool,
        &[
            "search".to_string(),
            "--onlyvisible".to_string(),
            "--name".to_string(),
            String::new(),
        ],
        "listing windows",
    )?;
    let mut rows = Vec::new();
    for id in output.lines().take(MAX_WINDOWS) {
        let geometry = run(
            &tool,
            &[
                "getwindowgeometry".to_string(),
                "--shell".to_string(),
                id.to_string(),
            ],
            "reading a window's geometry",
        )
        .unwrap_or_default();
        let number = |key: &str| field(&geometry, key).and_then(|value| value.parse().ok());
        let Some((x, y, width, height)) = number("X")
            .zip(number("Y"))
            .zip(number("WIDTH").zip(number("HEIGHT")))
            .map(|((x, y), (width, height))| (x, y, width, height))
        else {
            continue;
        };
        let title = run(
            &tool,
            &["getwindowname".to_string(), id.to_string()],
            "reading a window's title",
        )
        .unwrap_or_default();
        rows.push(WindowRow {
            id: id.to_string(),
            app: String::new(),
            title,
            x,
            y,
            width,
            height,
        });
    }
    Ok(rows)
}

pub(crate) fn list_windows(app_filter: Option<String>) -> Result<String, String> {
    let rows = window_rows()?;
    let filter = app_filter.map(|value| value.to_ascii_lowercase());
    let mut report = String::new();
    let mut shown = 0usize;
    for row in rows {
        let owner = if row.app.is_empty() {
            row.title.clone()
        } else {
            row.app.clone()
        };
        if let Some(filter) = &filter {
            if !owner.to_ascii_lowercase().contains(filter)
                && !row.title.to_ascii_lowercase().contains(filter)
            {
                continue;
            }
        }
        shown += 1;
        report.push_str(&format!(
            "id={} app={} title=\"{}\" frame={}x{}+{}+{}\n",
            row.id,
            if row.app.is_empty() { "?" } else { &row.app },
            row.title,
            row.width,
            row.height,
            row.x,
            row.y
        ));
    }
    if shown == 0 {
        if filter.is_some() {
            report.push_str("no window matched that app filter\n");
        } else {
            report.push_str("no on-screen windows were reported\n");
        }
    }
    Ok(report)
}

pub(crate) fn activate_app(app: String) -> Result<String, String> {
    if let Some(tool) = find_helper(&["wmctrl"]) {
        // `-a` matches a substring of the window title or class, which is the same
        // matching the caller gets from `list_windows`.
        return run(
            &tool,
            &["-a".to_string(), app.clone()],
            "activating an application",
        )
        .map(|_| format!("activated {app}"));
    }
    let tool = helper(&WINDOW_TOOLS, "install wmctrl or xdotool")?;
    let ids = run(
        &tool,
        &[
            "search".to_string(),
            "--onlyvisible".to_string(),
            "--name".to_string(),
            app.clone(),
        ],
        "finding an application window",
    )?;
    let Some(id) = ids.lines().next() else {
        return Err(format!("no on-screen window matched `{app}`"));
    };
    run(
        &tool,
        &["windowactivate".to_string(), id.to_string()],
        "activating an application",
    )?;
    Ok(format!("activated {app}"))
}