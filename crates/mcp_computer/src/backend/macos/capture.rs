//! Screen capture through `/usr/sbin/screencapture`.
//!
//! The CLI is used rather than `CGDisplayCreateImage` or ScreenCaptureKit because
//! it writes a PNG by itself, needs only the Screen Recording permission, and keeps
//! this crate free of image-encoding code. Its output is a plain file whose size is
//! read back from the PNG header, so nothing has to decode the pixels.
//!
//! One geometry trap is handled here and reported to the model: `-R` takes screen
//! *points*, while the PNG holds *pixels*, so a Retina display produces an image
//! twice the size of the captured region. The report therefore carries the capture
//! origin, the size in points and the pixels-per-point scale, which is everything
//! needed to turn a position in the image back into a click coordinate.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::screen::{self, Frame};
use super::sanitize;

const SCREENCAPTURE: &str = "/usr/sbin/screencapture";

/// Overrides where screenshots are written; the default keeps them out of the way
/// in the user's cache directory.
const DIR_ENV: &str = "MCP_COMPUTER_SCREENSHOT_DIR";
const DEFAULT_DIR: &str = ".cache/mcp_computer";

pub(crate) fn screenshot(
    region: Option<(f64, f64, f64, f64)>,
    window_id: Option<u32>,
    display: Option<u32>,
    include_cursor: bool,
) -> Result<String, String> {
    let path = output_path()?;

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
    command.arg(&path);

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

    let (pixel_width, pixel_height) = png_dimensions(&path).ok_or_else(|| {
        format!(
            "{SCREENCAPTURE} wrote {} but no PNG header could be read from it",
            path.display()
        )
    })?;

    // Which part of the screen this image covers. A window or display whose bounds
    // cannot be read still yields a usable image, just without the coordinate
    // mapping, so this stays best-effort.
    let frame = match (region, window_id, display) {
        (Some((x, y, width, height)), _, _) => screen::frame_at_point(x + width / 2.0, y + height / 2.0)
            .ok()
            .map(|display| Frame {
                x,
                y,
                width,
                height,
                scale: display.scale,
            }),
        (_, Some(window_id), _) => screen::window_frame(window_id).ok(),
        (_, _, Some(display)) => screen::display_frame(display).ok(),
        _ => screen::main_frame().ok(),
    };

    Ok(report(&path, pixel_width, pixel_height, frame))
}

/// A unique path per capture, so an earlier screenshot stays readable for the
/// model after a new one is taken.
fn output_path() -> Result<PathBuf, String> {
    let directory = output_directory()?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    Ok(directory.join(format!("shot-{stamp}.png")))
}

fn output_directory() -> Result<PathBuf, String> {
    let directory = match std::env::var(DIR_ENV) {
        Ok(value) if !value.trim().is_empty() => PathBuf::from(expand_home(value.trim())),
        _ => {
            let home = std::env::var("HOME").map_err(|_| {
                format!("HOME is not set; set {DIR_ENV} to choose where screenshots are written")
            })?;
            PathBuf::from(home).join(DEFAULT_DIR)
        }
    };
    std::fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "could not create the screenshot directory {}: {error}",
            directory.display()
        )
    })?;
    Ok(directory)
}

fn expand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => path.to_string(),
        },
        None => path.to_string(),
    }
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

/// The model-facing description of a fresh capture. It has to answer three
/// questions: where is the image, how big is it, and how does a position in the
/// image translate into a click.
fn report(path: &Path, pixel_width: u32, pixel_height: u32, frame: Option<Frame>) -> String {
    let mut text = format!("captured {pixel_width}x{pixel_height} px to {}\n", path.display());
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
    text.push_str("read this file with `read_file` to see the image before deciding where to click");
    text
}