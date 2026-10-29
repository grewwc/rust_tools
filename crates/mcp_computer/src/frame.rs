//! Shared screenshot scope, lifecycle, and visual fingerprinting.
//!
//! Backends only write a PNG to the path supplied here. This layer owns unique
//! paths, removes probe frames automatically, bounds the kept frames, and gives the
//! model a stable frame token based on decoded pixels rather than PNG encoding
//! details. A kept HiDPI frame is stored at one image pixel per screen point: the
//! vision model gets a quarter of the pixels, and an image coordinate is directly a
//! point offset from the capture origin. OCR reads the native pixels instead.

use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::imageops::FilterType;
use image::{DynamicImage, ImageEncoder, ImageReader, RgbaImage};
use sha2::{Digest, Sha256};

use crate::backend;

const FRAME_TOKEN_VERSION: &str = "mcp-computer-frame-v1";
const SCREENSHOT_DIR_ENV: &str = "MCP_COMPUTER_SCREENSHOT_DIR";
const CAPTURE_FILE_PREFIX: &str = "screen-";
/// Kept frames are scratch: `a`'s `read_file` copies a frame into its session before
/// showing it to the model, so the copy here only bridges the gap between the
/// screenshot reply and that read. Frames older than this are deleted when a new one
/// is kept. Age is the only rule because every `a` process runs its own server over
/// this one directory: a count cap would let one process delete another's unread frame.
const MAX_KEPT_FRAME_AGE: Duration = Duration::from_secs(30 * 60);
/// Downscale only when the capture holds at least this many pixels per point; below
/// it the saving does not pay for a resample that softens small text.
const MIN_DOWNSCALE_RATIO: f64 = 1.25;
static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CaptureTarget {
    Desktop,
    Region {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    },
    Window(u32),
    Display(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CaptureScope {
    target: CaptureTarget,
    include_cursor: bool,
}

impl CaptureScope {
    pub(crate) fn from_arguments(
        region: Option<(f64, f64, f64, f64)>,
        window_id: Option<u32>,
        display: Option<u32>,
        include_cursor: bool,
    ) -> Result<Self, String> {
        let selector_count = usize::from(region.is_some())
            + usize::from(window_id.is_some())
            + usize::from(display.is_some());
        if selector_count > 1 {
            return Err(
                "pass at most one of `region`, `window_id`, and `display` to screenshot"
                    .to_string(),
            );
        }

        let target = if let Some((x, y, width, height)) = region {
            let x = rounded_coordinate(x, "region.x")?;
            let y = rounded_coordinate(y, "region.y")?;
            let width = rounded_coordinate(width, "region.width")?;
            let height = rounded_coordinate(height, "region.height")?;
            if width <= 0 || height <= 0 {
                return Err(
                    "the screenshot region needs a positive width and height after rounding"
                        .to_string(),
                );
            }
            CaptureTarget::Region {
                x,
                y,
                width,
                height,
            }
        } else if let Some(window_id) = window_id {
            CaptureTarget::Window(window_id)
        } else if let Some(display) = display {
            CaptureTarget::Display(display)
        } else {
            CaptureTarget::Desktop
        };

        Ok(Self {
            target,
            include_cursor,
        })
    }

    pub(crate) fn include_cursor(self) -> bool {
        self.include_cursor
    }

    pub(crate) fn window_id(self) -> Option<u32> {
        match self.target {
            CaptureTarget::Window(window_id) => Some(window_id),
            _ => None,
        }
    }

    fn backend_arguments(self) -> (Option<(f64, f64, f64, f64)>, Option<u32>, Option<u32>) {
        match self.target {
            CaptureTarget::Desktop => (None, None, None),
            CaptureTarget::Region {
                x,
                y,
                width,
                height,
            } => (
                Some((x as f64, y as f64, width as f64, height as f64)),
                None,
                None,
            ),
            CaptureTarget::Window(id) => (None, Some(id), None),
            CaptureTarget::Display(id) => (None, None, Some(id)),
        }
    }

    fn encode(self) -> String {
        let target = match self.target {
            CaptureTarget::Desktop => "desktop".to_string(),
            CaptureTarget::Region {
                x,
                y,
                width,
                height,
            } => format!("region,{x},{y},{width},{height}"),
            CaptureTarget::Window(id) => format!("window,{id}"),
            CaptureTarget::Display(id) => format!("display,{id}"),
        };
        format!("{target}|cursor={}", u8::from(self.include_cursor))
    }

    fn decode(target: &str, cursor: &str) -> Result<Self, String> {
        let include_cursor = match cursor {
            "cursor=0" => false,
            "cursor=1" => true,
            _ => return Err("frame_id has an invalid cursor field".to_string()),
        };
        let fields: Vec<&str> = target.split(',').collect();
        let target = match fields.as_slice() {
            ["desktop"] => CaptureTarget::Desktop,
            ["region", x, y, width, height] => {
                let x = parse_field::<i32>(x, "region x")?;
                let y = parse_field::<i32>(y, "region y")?;
                let width = parse_field::<i32>(width, "region width")?;
                let height = parse_field::<i32>(height, "region height")?;
                if width <= 0 || height <= 0 {
                    return Err("frame_id has a non-positive region size".to_string());
                }
                CaptureTarget::Region {
                    x,
                    y,
                    width,
                    height,
                }
            }
            ["window", id] => CaptureTarget::Window(parse_field(id, "window id")?),
            ["display", id] => CaptureTarget::Display(parse_field(id, "display id")?),
            _ => return Err("frame_id has an invalid capture target".to_string()),
        };
        Ok(Self {
            target,
            include_cursor,
        })
    }
}

fn rounded_coordinate(value: f64, name: &str) -> Result<i32, String> {
    if !value.is_finite() {
        return Err(format!("{name} must be a finite number"));
    }
    let rounded = value.round();
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(format!("{name} is outside the supported coordinate range"));
    }
    Ok(rounded as i32)
}

fn parse_field<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("frame_id has an invalid {name}"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FrameFingerprint {
    scope: CaptureScope,
    width: u32,
    height: u32,
    pixel_digest: String,
}

impl FrameFingerprint {
    pub(crate) fn from_token(token: &str) -> Result<Self, String> {
        let fields: Vec<&str> = token.split('|').collect();
        let [version, target, cursor, dimensions, digest] = fields.as_slice() else {
            return Err("frame_id has an invalid field count".to_string());
        };
        if *version != FRAME_TOKEN_VERSION {
            return Err("frame_id uses an unsupported version".to_string());
        }
        let scope = CaptureScope::decode(target, cursor)?;
        let Some((width, height)) = dimensions.split_once('x') else {
            return Err("frame_id has invalid pixel dimensions".to_string());
        };
        let width = parse_field::<u32>(width, "pixel width")?;
        let height = parse_field::<u32>(height, "pixel height")?;
        if width == 0 || height == 0 {
            return Err("frame_id has zero pixel dimensions".to_string());
        }
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("frame_id has an invalid pixel digest".to_string());
        }
        Ok(Self {
            scope,
            width,
            height,
            pixel_digest: digest.to_ascii_lowercase(),
        })
    }

    pub(crate) fn scope(&self) -> CaptureScope {
        self.scope
    }

    pub(crate) fn to_token(&self) -> String {
        format!(
            "{FRAME_TOKEN_VERSION}|{}|{}x{}|{}",
            self.scope.encode(),
            self.width,
            self.height,
            self.pixel_digest
        )
    }

    fn from_pixels(image: &RgbaImage, scope: CaptureScope) -> Self {
        let (width, height) = image.dimensions();
        let mut hasher = Sha256::new();
        hasher.update(width.to_be_bytes());
        hasher.update(height.to_be_bytes());
        hasher.update(image.as_raw());
        let mut pixel_digest = String::with_capacity(64);
        for byte in hasher.finalize() {
            let _ = write!(pixel_digest, "{byte:02x}");
        }
        Self {
            scope,
            width,
            height,
            pixel_digest,
        }
    }
}

fn decode_png(path: &Path) -> Result<RgbaImage, String> {
    Ok(ImageReader::open(path)
        .map_err(|error| format!("could not open screenshot {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| {
            format!(
                "could not identify screenshot format for {}: {error}",
                path.display()
            )
        })?
        .decode()
        .map_err(|error| format!("could not decode screenshot {}: {error}", path.display()))?
        .into_rgba8())
}

/// One capture, decoded once: the native pixels serve the fingerprint and OCR, and
/// the file on disk is rewritten for the model only if the frame is kept.
pub(crate) struct CapturedFrame {
    file: CaptureFile,
    details: String,
    image: DynamicImage,
    fingerprint: FrameFingerprint,
    mapping: Option<ScreenMapping>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenMapping {
    origin_x: f64,
    origin_y: f64,
    point_width: f64,
    point_height: f64,
    pixel_width: u32,
    pixel_height: u32,
}

impl ScreenMapping {
    pub(crate) fn pixel_to_screen(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.origin_x + x * self.point_width / f64::from(self.pixel_width),
            self.origin_y + y * self.point_height / f64::from(self.pixel_height),
        )
    }

    /// Image size at one pixel per point, or `None` when the capture is not dense
    /// enough (a 1x display) for downscaling to pay off.
    fn point_sized(self) -> Option<(u32, u32)> {
        let ratio = f64::from(self.pixel_width) / self.point_width;
        if !(ratio >= MIN_DOWNSCALE_RATIO) {
            return None;
        }
        let width = self.point_width.round();
        let height = self.point_height.round();
        (width >= 1.0 && height >= 1.0).then_some((width as u32, height as u32))
    }
}

impl CapturedFrame {
    pub(crate) fn capture(scope: CaptureScope) -> Result<Self, String> {
        let file = CaptureFile::new()?;
        let (region, window_id, display) = scope.backend_arguments();
        let screenshot = backend::screenshot(
            region,
            window_id,
            display,
            scope.include_cursor(),
            file.path(),
        )?;
        let pixels = decode_png(file.path())?;
        let fingerprint = FrameFingerprint::from_pixels(&pixels, scope);
        let mapping = screenshot.origin.and_then(|(origin_x, origin_y)| {
            let (point_width, point_height) = screenshot
                .point_size
                .unwrap_or((f64::from(fingerprint.width), f64::from(fingerprint.height)));
            (fingerprint.width > 0
                && fingerprint.height > 0
                && point_width > 0.0
                && point_height > 0.0)
                .then_some(ScreenMapping {
                    origin_x,
                    origin_y,
                    point_width,
                    point_height,
                    pixel_width: fingerprint.width,
                    pixel_height: fingerprint.height,
                })
        });
        Ok(Self {
            file,
            details: screenshot.report,
            image: DynamicImage::ImageRgba8(pixels),
            fingerprint,
            mapping,
        })
    }

    pub(crate) fn fingerprint(&self) -> &FrameFingerprint {
        &self.fingerprint
    }

    /// The native capture pixels, which `mapping` describes.
    pub(crate) fn image(&self) -> &DynamicImage {
        &self.image
    }

    pub(crate) fn mapping(&self) -> Option<ScreenMapping> {
        self.mapping
    }

    /// Keep the frame for the model: store it at one pixel per point when the capture
    /// is HiDPI, prune old kept frames, and describe the stored file's geometry.
    pub(crate) fn into_report(mut self, status: &str) -> Result<String, String> {
        let (width, height) = self.store_for_model()?;
        self.file.keep = true;
        if let Some(directory) = self.file.path.parent() {
            prune_kept_frames(directory, &self.file.path, MAX_KEPT_FRAME_AGE);
        }
        let mut report = format!(
            "status={status}\nframe_id={}\nscreenshot saved: {}\n",
            self.fingerprint.to_token(),
            self.file.path.display()
        );
        report.push_str(&geometry_report(width, height, self.mapping));
        if !self.details.is_empty() {
            report.push_str(&self.details);
            if !self.details.ends_with('\n') {
                report.push('\n');
            }
        }
        report.push_str(
            "Next: call `read_file` on exactly this path to inspect the new frame before acting.",
        );
        Ok(report)
    }

    /// Rewrite a HiDPI capture at one pixel per point and return the stored size.
    fn store_for_model(&self) -> Result<(u32, u32), String> {
        let Some((width, height)) = self.mapping.and_then(ScreenMapping::point_sized) else {
            return Ok((self.fingerprint.width, self.fingerprint.height));
        };
        let resized = self.image.resize_exact(width, height, FilterType::Triangle);
        let path = &self.file.path;
        let file = File::create(path)
            .map_err(|error| format!("could not rewrite screenshot {}: {error}", path.display()))?;
        PngEncoder::new_with_quality(BufWriter::new(file), CompressionType::Fast, PngFilter::Adaptive)
            .write_image(resized.as_bytes(), width, height, resized.color().into())
            .map_err(|error| format!("could not encode screenshot {}: {error}", path.display()))?;
        Ok((width, height))
    }
}

/// Geometry of the stored image: everything needed to turn a position in it back
/// into a global screen point for `click`.
fn geometry_report(width: u32, height: u32, mapping: Option<ScreenMapping>) -> String {
    let mut text = format!("image size: {width}x{height} px\n");
    let Some(mapping) = mapping else {
        text.push_str(
            "capture frame: unknown (the window or display bounds could not be read); \
             take coordinates from `list_windows` and treat the image size as pixels\n",
        );
        return text;
    };
    let scale = f64::from(width) / mapping.point_width;
    let _ = writeln!(
        text,
        "capture frame: x={} y={} {}x{} points, scale {scale:.2} px/point",
        mapping.origin_x, mapping.origin_y, mapping.point_width, mapping.point_height
    );
    if (scale - 1.0).abs() < 0.01 {
        let _ = writeln!(
            text,
            "to click a feature at pixel (px, py) in this image: x = {} + px, y = {} + py",
            mapping.origin_x, mapping.origin_y
        );
    } else {
        let _ = writeln!(
            text,
            "to click a feature at pixel (px, py) in this image: \
             x = {} + px / {scale:.2}, y = {} + py / {scale:.2}",
            mapping.origin_x, mapping.origin_y
        );
    }
    text
}

/// Delete kept frames older than `max_age`. Only this server's own `screen-*.png`
/// files are considered, and `current` always survives.
fn prune_kept_frames(directory: &Path, current: &Path, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(CAPTURE_FILE_PREFIX) || !name.ends_with(".png") {
            continue;
        }
        let path = entry.path();
        if path == current {
            continue;
        }
        let expired = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| now.duration_since(modified).is_ok_and(|age| age > max_age));
        if expired {
            let _ = std::fs::remove_file(&path);
        }
    }
}

pub(crate) fn unchanged_report(fingerprint: &FrameFingerprint, elapsed_ms: u128) -> String {
    format!(
        "status=unchanged\nframe_id={}\nelapsed_ms={elapsed_ms}\n\
         No new screenshot was kept. Reuse the prior visual state and do not call `read_file`.",
        fingerprint.to_token()
    )
}

struct CaptureFile {
    path: PathBuf,
    keep: bool,
}

impl CaptureFile {
    fn new() -> Result<Self, String> {
        let directory = screenshot_directory()?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default();
        let sequence = NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "{CAPTURE_FILE_PREFIX}{}-{stamp}-{sequence}.png",
            std::process::id()
        ));
        Ok(Self { path, keep: false })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CaptureFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn screenshot_directory() -> Result<PathBuf, String> {
    let directory = match std::env::var(SCREENSHOT_DIR_ENV) {
        Ok(value) if !value.trim().is_empty() => expand_home(value.trim())?,
        _ => {
            let home = std::env::var_os("HOME").ok_or_else(|| {
                format!(
                    "HOME is not set; set {SCREENSHOT_DIR_ENV} to choose where screenshots are written"
                )
            })?;
            PathBuf::from(home).join(".cache/mcp_computer")
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

fn expand_home(path: &str) -> Result<PathBuf, String> {
    let Some(rest) = path.strip_prefix("~/") else {
        return Ok(PathBuf::from(path));
    };
    let home = std::env::var_os("HOME").ok_or_else(|| {
        format!("HOME is not set, so `{SCREENSHOT_DIR_ENV}={path}` cannot be expanded")
    })?;
    Ok(PathBuf::from(home).join(rest))
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use image::codecs::png::{CompressionType, FilterType, PngEncoder};
    use image::{ExtendedColorType, ImageEncoder};

    use super::*;

    fn write_png(path: &Path, pixels: &[u8], compression: CompressionType) {
        let file = File::create(path).unwrap();
        PngEncoder::new_with_quality(file, compression, FilterType::NoFilter)
            .write_image(pixels, 2, 1, ExtendedColorType::Rgba8)
            .unwrap();
    }

    #[test]
    fn frame_token_round_trips_scope() {
        let scopes = [
            CaptureScope::from_arguments(None, None, None, false).unwrap(),
            CaptureScope::from_arguments(Some((-4.4, 2.6, 80.2, 40.8)), None, None, true).unwrap(),
            CaptureScope::from_arguments(None, Some(42), None, false).unwrap(),
            CaptureScope::from_arguments(None, None, Some(3), true).unwrap(),
        ];
        for scope in scopes {
            let fingerprint = FrameFingerprint {
                scope,
                width: 120,
                height: 80,
                pixel_digest: "ab".repeat(32),
            };
            assert_eq!(
                FrameFingerprint::from_token(&fingerprint.to_token()).unwrap(),
                fingerprint
            );
        }
    }

    #[test]
    fn capture_scope_rejects_ambiguous_or_invalid_regions() {
        assert!(
            CaptureScope::from_arguments(Some((0.0, 0.0, 2.0, 2.0)), Some(1), None, false).is_err()
        );
        assert!(
            CaptureScope::from_arguments(Some((0.0, 0.0, 0.4, 2.0)), None, None, false).is_err()
        );
        assert!(
            CaptureScope::from_arguments(Some((f64::NAN, 0.0, 2.0, 2.0)), None, None, false)
                .is_err()
        );
    }

    #[test]
    fn fingerprint_hashes_decoded_pixels_not_png_encoding() {
        let first = CaptureFile::new().unwrap();
        let second = CaptureFile::new().unwrap();
        let changed = CaptureFile::new().unwrap();
        let pixels = [255, 0, 0, 255, 0, 0, 255, 255];
        write_png(first.path(), &pixels, CompressionType::Fast);
        write_png(second.path(), &pixels, CompressionType::Best);
        let mut changed_pixels = pixels;
        changed_pixels[0] = 254;
        write_png(changed.path(), &changed_pixels, CompressionType::Fast);
        assert_ne!(
            std::fs::read(first.path()).unwrap(),
            std::fs::read(second.path()).unwrap()
        );

        let scope = CaptureScope::from_arguments(None, None, None, false).unwrap();
        let fingerprint =
            |path: &Path| FrameFingerprint::from_pixels(&decode_png(path).unwrap(), scope);
        let first = fingerprint(first.path());
        let second = fingerprint(second.path());
        let changed = fingerprint(changed.path());
        assert_eq!(first, second);
        assert_ne!(first, changed);
    }

    #[test]
    fn hidpi_frames_are_stored_at_one_pixel_per_point() {
        let retina = ScreenMapping {
            origin_x: 72.0,
            origin_y: 38.0,
            point_width: 1314.0,
            point_height: 721.0,
            pixel_width: 2628,
            pixel_height: 1442,
        };
        assert_eq!(retina.point_sized(), Some((1314, 721)));
        let report = geometry_report(1314, 721, Some(retina));
        assert!(report.contains("image size: 1314x721 px"));
        assert!(report.contains("scale 1.00 px/point"));
        assert!(report.contains("x = 72 + px, y = 38 + py"));

        let standard = ScreenMapping {
            pixel_width: 1314,
            pixel_height: 721,
            ..retina
        };
        assert_eq!(standard.point_sized(), None);
    }

    #[test]
    fn pruning_removes_only_expired_frames_of_this_server() {
        let directory = std::env::temp_dir().join(format!(
            "mcp_computer_prune_{}_{}",
            std::process::id(),
            NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let now = SystemTime::now();
        let frame = |name: &str, age_secs: u64| {
            let path = directory.join(name);
            File::create(&path)
                .unwrap()
                .set_modified(now - Duration::from_secs(age_secs))
                .unwrap();
            path
        };
        let fresh = frame("screen-fresh.png", 60);
        let expired = frame("screen-expired.png", 7200);
        // The current frame survives even when its timestamp is old.
        let current = frame("screen-current.png", 7200);
        let foreign = frame("notes.png", 7200);

        prune_kept_frames(&directory, &current, Duration::from_secs(1800));
        assert!(fresh.exists());
        assert!(!expired.exists());
        assert!(current.exists());
        assert!(foreign.exists());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn malformed_frame_tokens_are_rejected() {
        assert!(FrameFingerprint::from_token("not-a-frame").is_err());
        assert!(
            FrameFingerprint::from_token("mcp-computer-frame-v1|desktop|cursor=0|1x1|xyz").is_err()
        );
    }

    #[test]
    fn screen_mapping_scales_pixel_centers_into_global_points() {
        let mapping = ScreenMapping {
            origin_x: 100.0,
            origin_y: 50.0,
            point_width: 200.0,
            point_height: 100.0,
            pixel_width: 400,
            pixel_height: 200,
        };
        assert_eq!(mapping.pixel_to_screen(200.0, 100.0), (200.0, 100.0));
    }
}
