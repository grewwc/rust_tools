//! Screen capture through `/usr/sbin/screencapture`.
//!
//! The CLI is used rather than `CGDisplayCreateImage` or ScreenCaptureKit because
//! it writes a PNG by itself and needs only the Screen Recording permission.
//!
//! One geometry trap is handled by reporting the captured frame in points: `-R`
//! takes screen *points*, while the PNG holds *pixels*, so a Retina display
//! produces an image twice the size of the captured region. `frame.rs` combines the
//! origin and point size returned here with the decoded image size, which is
//! everything needed to turn a position in the image back into a click coordinate.

use std::path::Path;
use std::process::Command;

use super::sanitize;
use super::screen::{self, Frame};

const SCREENCAPTURE: &str = "/usr/sbin/screencapture";

pub(crate) struct Screenshot {
    pub(crate) report: String,
    pub(crate) origin: Option<(f64, f64)>,
    pub(crate) point_size: Option<(f64, f64)>,
}

pub(crate) fn screenshot(
    region: Option<(f64, f64, f64, f64)>,
    window_id: Option<u32>,
    display: Option<u32>,
    include_cursor: bool,
    path: &Path,
) -> Result<Screenshot, String> {
    screen::ensure_screen_recording_permission()?;

    let mut command = Command::new(SCREENCAPTURE);
    // -x: no shutter sound. -o: no window shadow. -t png: one file, no clipboard.
    command.arg("-x").arg("-o").arg("-t").arg("png");
    if include_cursor {
        command.arg("-C");
    }
    if let Some((x, y, width, height)) = region {
        if width <= 0.0 || height <= 0.0 {
            return Err("the screenshot region needs a positive width and height".to_string());
        }
        command.arg(format!(
            "-R{},{},{},{}",
            x.round() as i64,
            y.round() as i64,
            width.round() as i64,
            height.round() as i64
        ));
    }
    if let Some(window_id) = window_id {
        command.arg(format!("-l{window_id}"));
    }
    if let Some(display) = display {
        command.arg(format!("-D{display}"));
    }
    command.arg(path);

    let output = command
        .output()
        .map_err(|error| format!("could not run {SCREENCAPTURE}: {error}"))?;
    if !output.status.success() {
        let stderr = sanitize(String::from_utf8_lossy(&output.stderr).trim());
        return Err(format!(
            "{SCREENCAPTURE} returned status {:?}{} — a missing or fully black capture \
             normally means the Screen Recording permission is not granted to the \
             process that started this server",
            output.status.code(),
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }

    // Which part of the screen this image covers. A window or display whose bounds
    // cannot be read still yields a usable image, just without the coordinate
    // mapping, so this stays best-effort.
    let frame = match (region, window_id, display) {
        (Some((x, y, width, height)), _, _) => {
            screen::frame_at_point(x + width / 2.0, y + height / 2.0)
                .ok()
                .map(|display| Frame {
                    x,
                    y,
                    width,
                    height,
                    scale: display.scale,
                })
        }
        (_, Some(window_id), _) => screen::window_frame(window_id).ok(),
        (_, _, Some(display)) => screen::display_frame(display).ok(),
        _ => screen::main_frame().ok(),
    };

    // The geometry lines are written by `frame.rs` against the stored image, which
    // may be downscaled from what screencapture wrote.
    Ok(Screenshot {
        report: String::new(),
        origin: frame.map(|frame| (frame.x, frame.y)),
        point_size: frame.map(|frame| (frame.width, frame.height)),
    })
}
