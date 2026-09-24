//! Backend for operating systems this crate has no desktop control for.
//!
//! `backend.rs` selects this only when neither macOS nor Linux is the target. The
//! tool surface is platform-neutral, so every entry point still exists here and
//! each one returns the same actionable error. Reporting the absence is the point:
//! a silent no-op would read to the model as "the click landed".

fn unsupported<T>() -> Result<T, String> {
    Err(format!(
        "computer-use tools have no desktop-control backend for `{}`; \
         macOS (CoreGraphics) and Linux (session CLIs) are the implemented platforms",
        std::env::consts::OS
    ))
}

pub(crate) fn screen_info() -> Result<String, String> {
    unsupported()
}

pub(crate) fn screenshot(
    _region: Option<(f64, f64, f64, f64)>,
    _window_id: Option<u32>,
    _display: Option<u32>,
    _include_cursor: bool,
) -> Result<String, String> {
    unsupported()
}

pub(crate) fn list_windows(_app_filter: Option<String>) -> Result<String, String> {
    unsupported()
}

pub(crate) fn activate_app(_app: String) -> Result<String, String> {
    unsupported()
}

pub(crate) fn click(
    _x: f64,
    _y: f64,
    _button: &str,
    _count: u32,
    _modifiers: &[String],
) -> Result<(), String> {
    unsupported()
}

pub(crate) fn move_mouse(_x: f64, _y: f64) -> Result<(), String> {
    unsupported()
}

pub(crate) fn drag(
    _from_x: f64,
    _from_y: f64,
    _to_x: f64,
    _to_y: f64,
    _button: &str,
    _modifiers: &[String],
    _steps: u32,
) -> Result<(), String> {
    unsupported()
}

pub(crate) fn scroll(
    _x: Option<f64>,
    _y: Option<f64>,
    _dx: f64,
    _dy: f64,
    _unit: &str,
) -> Result<(), String> {
    unsupported()
}

pub(crate) fn type_text(_text: &str) -> Result<(), String> {
    unsupported()
}

pub(crate) fn press_key(_chord: &str) -> Result<(), String> {
    unsupported()
}