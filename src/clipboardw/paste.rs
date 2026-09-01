//! Unified paste: read the clipboard -> detect the content type -> save to a file with
//! a suitable extension.
//!
//! This is the paste entry point for `oo -p`. Unlike the old binary -> string -> image
//! three-stage fallback retry, here the clipboard is read once and saved by type:
//!
//! - **SSH sessions**: query the terminal clipboard exactly once via OSC52, avoiding
//!   retransmitting the whole clipboard on every retry (large images can be several MB);
//! - **Extension**: detected images take the extension of their actual format
//!   (`output.png` / `output.jpg` etc.), text gets `.txt`, and other binary keeps its
//!   original file name.

use std::{fs, io};

use arboard::Clipboard;
use image::{ImageBuffer, Rgb, Rgba, buffer::ConvertBuffer};

use crate::clipboardw::string_content;
use crate::commonw::filename::add_suffix;

fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

/// Classification of the clipboard content
#[derive(Debug)]
enum SaveKind {
    /// Image bytes (native or base64-decoded), saved as `<fname>.<ext>`
    Image { data: Vec<u8>, ext: &'static str },
    /// Text, saved as `<fname>.txt`
    Text(String),
    /// Other binary (including arbitrary files wrapped in base64), saved as `<fname>`
    Binary(Vec<u8>),
}

/// Saves the clipboard content to a file and returns the actual saved path.
///
/// Read order:
/// 1. Local session: prefer arboard native images (does not trigger OSC52);
/// 2. Local session: arboard text; SSH session: read the terminal clipboard once via OSC52;
/// 3. Classify the content and write it with a suitable extension.
pub fn paste_to_file(fname: &str) -> Result<String, Box<dyn std::error::Error>> {
    // 1) Local native image
    if !is_ssh_session() {
        if let Some(path) = try_save_local_image(fname) {
            return Ok(path);
        }
    }

    // 2) Single read: arboard locally, OSC52 over SSH
    let raw: Vec<u8> = if is_ssh_session() {
        string_content::get_clipboard_raw_bytes_via_osc52()
            .ok_or_else(|| io::Error::other("no clipboard data via OSC52"))?
    } else {
        string_content::get_clipboard_content().into_bytes()
    };

    // 3) Classify and save
    save_classified(fname, classify(&raw))
}

/// Local arboard native image: save as `<fname>.jpg`.
fn try_save_local_image(fname: &str) -> Option<String> {
    let mut clipboard = Clipboard::new().ok()?;
    let image = clipboard.get_image().ok()?;
    let buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
        image.width as u32,
        image.height as u32,
        image.bytes.to_vec(),
    )?;
    let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = buf.convert();
    let path = image_save_path(fname, "jpg");
    rgb.save(&path).ok()?;
    Some(path)
}

/// Detects the type of the clipboard data from its content.
fn classify(raw: &[u8]) -> SaveKind {
    // Native image bytes (OSC52 returns base64(image bytes); decoding yields the image)
    if let Ok(format) = image::guess_format(raw) {
        if image::load_from_memory(raw).is_ok() {
            return SaveKind::Image {
                data: raw.to_vec(),
                ext: ext_of(format),
            };
        }
    }

    // Text (including base64-wrapped images / binaries)
    if let Ok(text) = std::str::from_utf8(raw) {
        if !text.is_empty() {
            let cleaned: String = text
                .chars()
                .filter(|c| !matches!(c, '\n' | '\r'))
                .collect();
            if !cleaned.is_empty() {
                if let Some(decoded) = crate::clipboardw::decode_base64_lenient(&cleaned) {
                    // An empty decode result (e.g. whitespace-only clipboard) is
                    // plain text, not a base64 payload.
                    if !decoded.is_empty() {
                        // base64-wrapped image (oo -B / oo -c bridge)
                        if let Ok(format) = image::guess_format(&decoded) {
                            if image::load_from_memory(&decoded).is_ok() {
                                return SaveKind::Image {
                                    data: decoded,
                                    ext: ext_of(format),
                                };
                            }
                        }
                        // Other base64-wrapped binary
                        return SaveKind::Binary(decoded);
                    }
                }
            }
            return SaveKind::Text(text.to_string());
        }
    }

    // Remaining raw bytes
    SaveKind::Binary(raw.to_vec())
}

fn ext_of(format: image::ImageFormat) -> &'static str {
    format.extensions_str().first().copied().unwrap_or("jpg")
}

/// Image save path: append `.ext` when there is no extension; keep an existing one.
fn image_save_path(fname: &str, ext: &str) -> String {
    add_suffix(fname, &format!(".{ext}"), || !fname.contains('.'))
}

fn save_classified(fname: &str, kind: SaveKind) -> Result<String, Box<dyn std::error::Error>> {
    let path = match kind {
        SaveKind::Image { data, ext } => {
            let path = image_save_path(fname, ext);
            fs::write(&path, data)?;
            path
        }
        SaveKind::Text(text) => {
            let path = add_suffix(fname, ".txt", || !fname.contains('.'));
            fs::write(&path, text)?;
            path
        }
        SaveKind::Binary(data) => {
            fs::write(fname, data)?;
            fname.to_string()
        }
    };
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use base64::engine::general_purpose;

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(1, 1, Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn classify_native_png_bytes() {
        let png = tiny_png();
        match classify(&png) {
            SaveKind::Image { data, ext } => {
                assert_eq!(ext, "png");
                assert_eq!(data, png);
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn classify_base64_wrapped_png() {
        let png = tiny_png();
        let b64 = general_purpose::STANDARD.encode(&png);
        match classify(b64.as_bytes()) {
            SaveKind::Image { data, ext } => {
                assert_eq!(ext, "png");
                assert_eq!(data, png);
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn classify_base64_wrapped_png_unpadded() {
        // Terminals may emit OSC52 responses without '=' padding; after one decode
        // the clipboard text can itself be unpadded base64. classify must still
        // recover the image.
        let png = tiny_png();
        let b64 = general_purpose::STANDARD.encode(&png);
        let b64 = b64.trim_end_matches('=');
        match classify(b64.as_bytes()) {
            SaveKind::Image { data, ext } => {
                assert_eq!(ext, "png");
                assert_eq!(data, png);
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn decode_base64_lenient_accepts_unpadded_and_wrapped() {
        use crate::clipboardw::decode_base64_lenient;

        let data = b"hello world";
        let b64 = general_purpose::STANDARD.encode(data);

        // Missing '=' padding must still decode.
        let unpadded = b64.trim_end_matches('=');
        assert_eq!(decode_base64_lenient(unpadded), Some(data.to_vec()));

        // Line-wrapped payloads (some terminals wrap at 76 chars) must still decode.
        let wrapped: String = b64
            .as_bytes()
            .chunks(4)
            .map(|c| String::from_utf8_lossy(c))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(decode_base64_lenient(&wrapped), Some(data.to_vec()));
    }

    #[test]
    fn classify_plain_text() {
        match classify(b"hello world") {
            SaveKind::Text(t) => assert_eq!(t, "hello world"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn classify_base64_binary() {
        let bin = vec![0u8, 1, 2, 3, 250];
        let b64 = general_purpose::STANDARD.encode(&bin);
        match classify(b64.as_bytes()) {
            SaveKind::Binary(data) => assert_eq!(data, bin),
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn classify_raw_non_utf8_binary() {
        match classify(&[0xFF, 0x00, 0x01]) {
            SaveKind::Binary(data) => assert_eq!(data, vec![0xFF, 0x00, 0x01]),
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn classify_whitespace_only_text() {
        // Matches the legacy string_content path: whitespace-only content is still saved as text
        match classify(b"\n  \n") {
            SaveKind::Text(t) => assert_eq!(t, "\n  \n"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn image_save_path_appends_extension_only_when_missing() {
        assert_eq!(image_save_path("output", "png"), "output.png");
        assert_eq!(image_save_path("out.txt", "png"), "out.txt");
    }

    #[test]
    fn save_classified_image_writes_with_extension() {
        let png = tiny_png();
        let base = std::env::temp_dir().join(format!("oo_paste_test_{}", std::process::id()));
        let base = base.to_string_lossy().into_owned();
        let _ = fs::remove_file(format!("{base}.png"));

        let path = save_classified(&base, SaveKind::Image { data: png.clone(), ext: "png" }).unwrap();
        assert_eq!(path, format!("{base}.png"));
        assert_eq!(fs::read(&path).unwrap(), png);

        let _ = fs::remove_file(&path);
    }
}
