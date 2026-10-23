//! Screen capture through `/usr/sbin/screencapture`.
//!
//! The CLI is used rather than `CGDisplayCreateImage` or ScreenCaptureKit because
//! it writes a PNG by itself and needs only the Screen Recording permission. Its
//! output size is read from the PNG header for the coordinate mapping report.
//!
//! One geometry trap is handled here and reported to the model: `-R` takes screen
//! *points*, while the PNG holds *pixels*, so a Retina display produces an image
//! twice the size of the captured region. The report therefore carries the capture
//! origin, the size in points and the pixels-per-point scale, which is everything
//! needed to turn a position in the image back into a click coordinate.

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

    let (pixel_width, pixel_height) = png_dimensions(path).ok_or_else(|| {
        format!(
            "{SCREENCAPTURE} wrote {} but no PNG header could be read from it",
            path.display()
        )
    })?;

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

    Ok(Screenshot {
        report: report(pixel_width, pixel_height, frame),
        origin: frame.map(|frame| (frame.x, frame.y)),
        point_size: frame.map(|frame| (frame.width, frame.height)),
    })
}

/// Size in pixels from the PNG header: the IHDR chunk always starts at byte 16
/// with width then height as big-endian u32. Reading 24 bytes beats pulling in a
/// decoder just to report a size.
fn png_dimensions(path: &Path) -> Option<(u32, u32)> {
    let bytes = std::fs::read(path).ok()?;
    const PNG_MAGIC: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 24 || &bytes[..8] != PNG_MAGIC {
        return None;
    }
    let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((width, height))
}

/// Geometry details that let the model translate image pixels into screen points.
fn report(pixel_width: u32, pixel_height: u32, frame: Option<Frame>) -> String {
    let mut text = format!("image size: {pixel_width}x{pixel_height} px\n");
    match frame {
        Some(frame) => {
            // The scale is measured from the capture instead of read from the display:
            // `CGDisplayPixelsWide` reports the *point* size on a HiDPI mode (1.00 px/point
            // on a Retina screen), while the PNG holds the backing pixels. Measuring the
            // ratio against the frame the capture covers is also the only value that stays
            // true for a window capture, where the owning display need not be the one the
            // window is mostly on.
            let scale = if frame.width > 0.0 {
                pixel_width as f64 / frame.width
            } else {
                frame.scale
            };
            text.push_str(&format!(
                "capture frame: x={} y={} {}x{} points, scale {:.2} px/point\n",
                frame.x, frame.y, frame.width, frame.height, scale
            ));
            if scale > 0.0 {
                text.push_str(&format!(
                    "to click a feature at pixel (px, py) in this image: \
                     x = {} + px / {:.2}, y = {} + py / {:.2}\n",
                    frame.x, scale, frame.y, scale
                ));
            }
        }
        None => text.push_str(
            "capture frame: unknown (the window or display bounds could not be read); \
             take coordinates from `list_windows` and treat the image size as pixels\n",
        ),
    }
    text
}
