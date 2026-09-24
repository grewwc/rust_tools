//! Display, window and application queries for the macOS backend.
//!
//! Two CoreGraphics APIs are used, for different reasons:
//!
//! - `CGGetActiveDisplayList` / `CGDisplayBounds` / `CGDisplayPixelsWide` give the
//!   exact geometry needed to convert between points and pixels.
//! - `CGWindowListCopyWindowInfo` gives windows and, indirectly, the frontmost
//!   application: CoreGraphics returns on-screen windows front to back, so the
//!   first normal-layer window belongs to the app the user is looking at. This path
//!   needs no permission at all, whereas enumerating windows through
//!   `System Events` requires Accessibility access and fails with error -25211
//!   ("not allowed assistive access") when that access has not been granted.
//!
//! Screen *capture* is a different matter: it needs the Screen Recording
//! permission, which is why `screen_info` reports both permissions instead of
//! letting the model interpret an empty result as "there is nothing on screen".

use std::process::Command;
use std::time::Duration;

use core::ffi::c_void;

use objc2_core_foundation::{CFDictionary, CFNumber, CFString};
use objc2_core_graphics::{
    CGDisplayBounds, CGDisplayCopyDisplayMode, CGDisplayMode, CGDisplayPixelsHigh,
    CGDisplayPixelsWide, CGEvent, CGEventSource, CGEventSourceStateID, CGEventType,
    CGGetActiveDisplayList, CGMainDisplayID, CGMouseButton, CGWindowListCopyWindowInfo,
    CGWindowListOption, kCGWindowBounds, kCGWindowLayer, kCGWindowName, kCGWindowNumber,
    kCGWindowOwnerName,
};

use super::sanitize;

/// How many displays `CGGetActiveDisplayList` is asked about. A machine with more
/// than 16 active displays is not a case worth failing over: the extra ones would
/// simply not be listed.
const MAX_DISPLAYS: usize = 16;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    /// `Boolean AXIsProcessTrusted(void)` — true when this process may inject
    /// events, i.e. has Accessibility access.
    fn AXIsProcessTrusted() -> u8;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// `bool CGPreflightScreenCaptureAccess(void)` — true when this process may
    /// read the screen, i.e. has Screen Recording access. Unlike
    /// `CGRequestScreenCaptureAccess` it never prompts, which is what a tool call
    /// should do.
    fn CGPreflightScreenCaptureAccess() -> u8;
}

/// A rectangle in screen points plus the pixels-per-point scale of the display it
/// belongs to. `screenshot` reports this so the model can map image pixels back to
/// click coordinates.
#[derive(Clone, Copy)]
pub(crate) struct Frame {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) scale: f64,
}

struct Display {
    id: u32,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    pixel_width: usize,
    pixel_height: usize,
}

impl Display {
    /// Pixels per point: 2.0 on a Retina display, 1.0 on an external non-Retina
    /// one. A display that reports a zero width would divide by zero, so the value
    /// falls back to 1.0 and the model is told a usable scale either way.
    fn scale(&self) -> f64 {
        if self.width > 0.0 {
            self.pixel_width as f64 / self.width
        } else {
            1.0
        }
    }

    fn frame(&self) -> Frame {
        Frame {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            scale: self.scale(),
        }
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

fn displays() -> Result<Vec<Display>, String> {
    let mut ids = [0u32; MAX_DISPLAYS];
    let mut count = 0u32;
    let error =
        unsafe { CGGetActiveDisplayList(MAX_DISPLAYS as u32, ids.as_mut_ptr(), &mut count) };
    if error.0 != 0 {
        return Err(format!(
            "CoreGraphics could not list the active displays (CGError {})",
            error.0
        ));
    }
    let count = (count as usize).min(MAX_DISPLAYS);
    Ok(ids[..count]
        .iter()
        .map(|id| {
            let bounds = CGDisplayBounds(*id);
            // The backing store is not what `CGDisplayPixelsWide` reports: on a HiDPI
            // mode that call returns the display's *point* width, so a Retina screen
            // would be described as 1.00 px/point while its captures come back at 2x.
            // Only the current display mode carries the real pixel size; the point-sized
            // values remain as a fallback for a mode-less display.
            let mode = CGDisplayCopyDisplayMode(*id);
            let mode_width = CGDisplayMode::pixel_width(mode.as_deref());
            let mode_height = CGDisplayMode::pixel_height(mode.as_deref());
            Display {
                id: *id,
                x: bounds.origin.x,
                y: bounds.origin.y,
                width: bounds.size.width,
                height: bounds.size.height,
                pixel_width: if mode_width > 0 {
                    mode_width
                } else {
                    CGDisplayPixelsWide(*id)
                },
                pixel_height: if mode_height > 0 {
                    mode_height
                } else {
                    CGDisplayPixelsHigh(*id)
                },
            }
        })
        .collect())
}

/// Pixels per point at a screen position, falling back to the first display and
/// then to 1.0 — a wrong scale is worse than an approximate one only if it is
/// reported as exact, so the value is always shown to the model.
pub(crate) fn scale_at(x: f64, y: f64) -> f64 {
    let Ok(displays) = displays() else {
        return 1.0;
    };
    displays
        .iter()
        .find(|display| display.contains(x, y))
        .or_else(|| displays.first())
        .map(|display| display.scale())
        .unwrap_or(1.0)
}

pub(crate) fn main_frame() -> Result<Frame, String> {
    display_frame(CGMainDisplayID())
}

pub(crate) fn display_frame(display_id: u32) -> Result<Frame, String> {
    let displays = displays()?;
    displays
        .iter()
        .find(|display| display.id == display_id)
        .map(|display| display.frame())
        .ok_or_else(|| format!("display id {display_id} is not an active display"))
}

pub(crate) fn frame_at_point(x: f64, y: f64) -> Result<Frame, String> {
    let displays = displays()?;
    displays
        .iter()
        .find(|display| display.contains(x, y))
        .map(|display| display.frame())
        .ok_or_else(|| format!("the point ({x}, {y}) is outside every active display"))
}

pub(crate) fn cursor_position() -> Option<(f64, f64)> {
    // A null mouse event carries the current pointer position; there is no
    // standalone "get cursor" call in CoreGraphics.
    let event = CGEvent::new_mouse_event(
        CGEventSource::new(CGEventSourceStateID::HIDSystemState).as_deref(),
        CGEventType::Null,
        objc2_core_foundation::CGPoint::new(0.0, 0.0),
        CGMouseButton::Left,
    )?;
    let point = CGEvent::location(Some(&*event));
    Some((point.x, point.y))
}

/// One on-screen window, in the order CoreGraphics returned it (front to back).
pub(crate) struct WindowRow {
    id: u32,
    app: String,
    title: Option<String>,
    layer: i64,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

fn window_rows() -> Result<Vec<WindowRow>, String> {
    // On-screen only, desktop wallpaper excluded: a computer-use loop cares about
    // what can actually be seen and clicked.
    let option =
        CGWindowListOption::OptionOnScreenOnly | CGWindowListOption::ExcludeDesktopElements;
    let list = CGWindowListCopyWindowInfo(option, 0)
        .ok_or_else(|| "CoreGraphics returned no window list".to_string())?;

    // Entries are read through the C accessors rather than the typed generics: only
    // the opaque form of `CFArray`/`CFDictionary` implements `ConcreteType`, so a
    // typed `CFDictionary<CFString, CFType>` is not a legal downcast target.
    let mut rows = Vec::new();
    for index in 0..list.count() {
        let entry = unsafe { list.value_at_index(index) };
        if entry.is_null() {
            continue;
        }
        // Borrowed for this iteration only: the array owns the dictionary for as long
        // as `list` lives, and the dictionary owns every value read from it below.
        let dictionary = unsafe { &*(entry as *const CFDictionary) };
        let Some(id) = dictionary_number(dictionary, unsafe { kCGWindowNumber }) else {
            continue;
        };
        let (x, y, width, height) = dictionary_bounds(dictionary).unwrap_or((0.0, 0.0, 0.0, 0.0));
        rows.push(WindowRow {
            id: id as u32,
            app: dictionary_string(dictionary, unsafe { kCGWindowOwnerName })
                .unwrap_or_else(|| "unknown".to_string()),
            title: dictionary_string(dictionary, unsafe { kCGWindowName }),
            layer: dictionary_number(dictionary, unsafe { kCGWindowLayer }).unwrap_or(0.0) as i64,
            x,
            y,
            width,
            height,
        });
    }
    Ok(rows)
}

/// A dictionary key has to be handed over as the bare CFString pointer it already is.
fn key_pointer(key: &CFString) -> *const c_void {
    key as *const CFString as *const c_void
}

/// Read one attribute, returning `None` when the key is absent so the caller can
/// decide whether a missing attribute is fatal (the window id) or defaultable.
fn dictionary_value(dictionary: &CFDictionary, key: &CFString) -> Option<*const c_void> {
    let value = unsafe { dictionary.value(key_pointer(key)) };
    if value.is_null() { None } else { Some(value) }
}

fn dictionary_number(dictionary: &CFDictionary, key: &CFString) -> Option<f64> {
    let value = dictionary_value(dictionary, key)?;
    // Every numeric window attribute is integral, and `as_cgfloat` reads any
    // CFNumber width, so one accessor covers them all.
    unsafe { &*(value as *const CFNumber) }.as_cgfloat()
}

fn dictionary_string(dictionary: &CFDictionary, key: &CFString) -> Option<String> {
    let value = dictionary_value(dictionary, key)?;
    let string = unsafe { &*(value as *const CFString) };
    // `as_str_unchecked` only borrows when the internal storage is already UTF-8;
    // falling back to Display keeps non-ASCII names (for example a Chinese
    // application name) readable.
    match unsafe { string.as_str_unchecked() } {
        Some(text) => Some(text.to_string()),
        None => Some(string.to_string()),
    }
}

/// `kCGWindowBounds` is itself a CFDictionary of four CFNumbers, so the rectangle
/// needs a second lookup with the field names Apple documents for it.
fn dictionary_bounds(dictionary: &CFDictionary) -> Option<(f64, f64, f64, f64)> {
    let value = dictionary_value(dictionary, unsafe { kCGWindowBounds })?;
    let bounds = unsafe { &*(value as *const CFDictionary) };
    let x = CFString::from_str("X");
    let y = CFString::from_str("Y");
    let width = CFString::from_str("Width");
    let height = CFString::from_str("Height");
    Some((
        dictionary_number(bounds, &x)?,
        dictionary_number(bounds, &y)?,
        dictionary_number(bounds, &width)?,
        dictionary_number(bounds, &height)?,
    ))
}

pub(crate) fn window_frame(window_id: u32) -> Result<Frame, String> {
    let rows = window_rows()?;
    let row = rows
        .iter()
        .find(|row| row.id == window_id)
        .ok_or_else(|| format!("window id {window_id} is not an on-screen window"))?;
    Ok(Frame {
        x: row.x,
        y: row.y,
        width: row.width,
        height: row.height,
        scale: scale_at(row.x + row.width / 2.0, row.y + row.height / 2.0),
    })
}

fn frontmost_app(rows: &[WindowRow]) -> Option<String> {
    rows.iter()
        .find(|row| row.layer == 0)
        .map(|row| row.app.clone())
}

pub(crate) fn screen_info() -> Result<String, String> {
    let displays = displays()?;
    let main = CGMainDisplayID();
    let mut text = format!(
        "displays ({} active); all coordinates are points with the origin at the top-left of the main display:\n",
        displays.len()
    );
    for display in &displays {
        text.push_str(&format!(
            "  id={} bounds=x:{} y:{} {}x{} points, {}x{} px, scale {:.2}{}\n",
            display.id,
            display.x,
            display.y,
            display.width,
            display.height,
            display.pixel_width,
            display.pixel_height,
            display.scale(),
            if display.id == main { " (main)" } else { "" }
        ));
    }
    match cursor_position() {
        Some((x, y)) => text.push_str(&format!("cursor: x={x} y={y}\n")),
        None => text.push_str("cursor: unknown\n"),
    }
    match window_rows() {
        Ok(rows) => {
            match frontmost_app(&rows) {
                Some(app) => text.push_str(&format!("frontmost app: {app}\n")),
                None => text.push_str("frontmost app: unknown\n"),
            }
            text.push_str(&format!("on-screen windows: {}\n", rows.len()));
        }
        Err(error) => text.push_str(&format!("windows: unavailable ({error})\n")),
    }

    // SAFETY: both functions only read a process-level permission flag.
    let accessibility = unsafe { AXIsProcessTrusted() } != 0;
    let screen_recording = unsafe { CGPreflightScreenCaptureAccess() } != 0;
    text.push_str(&format!(
        "accessibility (mouse/keyboard injection): {}\n",
        if accessibility { "granted" } else { "NOT granted" }
    ));
    text.push_str(&format!(
        "screen recording (screenshot): {}\n",
        if screen_recording {
            "granted"
        } else {
            "NOT granted"
        }
    ));
    if !accessibility || !screen_recording {
        text.push_str(
            "grant the missing permissions to the process that starts this server in \
             System Settings > Privacy & Security; until then clicks and keystrokes are \
             delivered nowhere and captures come back empty\n",
        );
    }
    Ok(text)
}

pub(crate) fn list_windows(app_filter: Option<String>) -> Result<String, String> {
    let rows = window_rows()?;
    let frontmost = frontmost_app(&rows);
    let filter = app_filter
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty());

    let mut text = String::from(
        "on-screen windows, front to back:\n",
    );
    let mut shown = 0usize;
    for row in &rows {
        if let Some(filter) = &filter {
            let in_app = row.app.to_lowercase().contains(filter);
            let in_title = row
                .title
                .as_deref()
                .map(|title| title.to_lowercase().contains(filter))
                .unwrap_or(false);
            if !in_app && !in_title {
                continue;
            }
        }
        shown += 1;
        let front_mark = if Some(&row.app) == frontmost.as_ref() && row.layer == 0 {
            " (frontmost app)"
        } else {
            ""
        };
        text.push_str(&format!(
            "  window_id={} layer={} app={:?} title={:?} bounds=x:{} y:{} {}x{} points{}\n",
            row.id,
            row.layer,
            row.app,
            row.title.clone().unwrap_or_default(),
            row.x,
            row.y,
            row.width,
            row.height,
            front_mark
        ));
    }
    if shown == 0 {
        text.push_str("  (no on-screen window matched)\n");
    }
    text.push_str(
        "pass a window_id to `screenshot` to look at one window; the bounds are the \
         coordinate space `click`, `drag` and `scroll` use\n",
    );
    if all_titles_missing(&rows) {
        text.push_str(
            "titles are empty because macOS only reveals window titles to a process that \
             has the Screen Recording permission\n",
        );
    }
    Ok(text)
}

/// Enough rows to tell "this app has no title" from "titles are hidden": one
/// untitled window is normal (dialogs, palettes), every untitled row is not.
fn all_titles_missing(rows: &[WindowRow]) -> bool {
    !rows.is_empty() && rows.iter().all(|row| row.title.is_none())
}

pub(crate) fn activate_app(app: String) -> Result<String, String> {
    let name = app.trim();
    if name.is_empty() {
        return Err("activate_app needs an application name or bundle identifier".to_string());
    }
    // `open -b` takes a bundle identifier (`com.apple.Safari`), `open -a` an
    // application name or a path. A dotted name without spaces is the only shape
    // that is unambiguously an identifier.
    let bundle_identifier = name.contains('.') && !name.contains(' ');
    let mut command = Command::new("/usr/bin/open");
    command.arg(if bundle_identifier { "-b" } else { "-a" }).arg(name);
    let output = command
        .output()
        .map_err(|error| format!("could not run /usr/bin/open: {error}"))?;
    if !output.status.success() {
        let stderr = sanitize(String::from_utf8_lossy(&output.stderr).trim());
        return Err(format!(
            "/usr/bin/open could not activate {name:?} (status {:?}){}; check the name with \
             `list_windows`, which reports the application name of every on-screen window",
            output.status.code(),
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    // Activation is handed to the window server asynchronously, so report what is
    // actually in front afterwards rather than assuming the request took effect.
    std::thread::sleep(Duration::from_millis(400));
    let frontmost = window_rows()
        .ok()
        .and_then(|rows| frontmost_app(&rows))
        .unwrap_or_else(|| "unknown".to_string());
    Ok(format!("activation requested for {name:?}; frontmost app is now {frontmost:?}"))
}