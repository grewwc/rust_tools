//! 统一粘贴：读取剪贴板 → 识别内容类型 → 以合适的扩展名保存到文件。
//!
//! 这是 `oo -p` 的粘贴入口。相比旧的 binary → string → image 三级联重试，
//! 这里只读取一次剪贴板内容，并按类型自动选择保存方式：
//!
//! - **SSH 会话**：通过 OSC52 只查询一次终端剪贴板，避免每次重试都重新传输
//!   整份剪贴板内容（大图可达数 MB）；
//! - **扩展名**：识别出的图片按实际格式取扩展名（`output.png` / `output.jpg`
//!   等），文本加 `.txt`，其余二进制保留原文件名。

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

/// 剪贴板内容的分类结果
#[derive(Debug)]
enum SaveKind {
    /// 图片字节（原生或经 base64 解码），保存为 `<fname>.<ext>`
    Image { data: Vec<u8>, ext: &'static str },
    /// 文本，保存为 `<fname>.txt`
    Text(String),
    /// 其他二进制（含 base64 包裹的任意文件），保存为 `<fname>`
    Binary(Vec<u8>),
}

/// 将剪贴板内容保存为文件，返回实际保存的路径。
///
/// 读取顺序：
/// 1. 本地会话：优先 arboard 原生图片（不触发 OSC52）；
/// 2. 本地会话：arboard 文本；SSH 会话：一次 OSC52 读取终端剪贴板；
/// 3. 按内容分类后以合适的扩展名写入。
pub fn paste_to_file(fname: &str) -> Result<String, Box<dyn std::error::Error>> {
    // 1) 本地原生图片
    if !is_ssh_session() {
        if let Some(path) = try_save_local_image(fname) {
            return Ok(path);
        }
    }

    // 2) 一次读取：本地走 arboard，SSH 走 OSC52
    let raw: Vec<u8> = if is_ssh_session() {
        string_content::get_clipboard_raw_bytes_via_osc52()
            .ok_or_else(|| io::Error::other("no clipboard data via OSC52"))?
    } else {
        string_content::get_clipboard_content().into_bytes()
    };

    // 3) 分类并保存
    save_classified(fname, classify(&raw))
}

/// 本地 arboard 原生图片：保存为 `<fname>.jpg`。
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

/// 按内容识别剪贴板数据的类型。
fn classify(raw: &[u8]) -> SaveKind {
    // 原生图片字节（OSC52 返回 base64(图片字节)，解码后即图片）
    if let Ok(format) = image::guess_format(raw) {
        if image::load_from_memory(raw).is_ok() {
            return SaveKind::Image {
                data: raw.to_vec(),
                ext: ext_of(format),
            };
        }
    }

    // 文本（含 base64 包裹的图片 / 二进制）
    if let Ok(text) = std::str::from_utf8(raw) {
        if !text.is_empty() {
            let cleaned: String = text
                .chars()
                .filter(|c| !matches!(c, '\n' | '\r'))
                .collect();
            if !cleaned.is_empty() {
                use base64::Engine as _;
                use base64::engine::general_purpose;
                if let Ok(decoded) = general_purpose::STANDARD.decode(&cleaned) {
                    // base64 包裹的图片（oo -B / oo -c 桥接）
                    if let Ok(format) = image::guess_format(&decoded) {
                        if image::load_from_memory(&decoded).is_ok() {
                            return SaveKind::Image {
                                data: decoded,
                                ext: ext_of(format),
                            };
                        }
                    }
                    // 其他 base64 包裹的二进制
                    return SaveKind::Binary(decoded);
                }
            }
            return SaveKind::Text(text.to_string());
        }
    }

    // 其余原始字节
    SaveKind::Binary(raw.to_vec())
}

fn ext_of(format: image::ImageFormat) -> &'static str {
    format.extensions_str().first().copied().unwrap_or("jpg")
}

/// 图片保存路径：无扩展名时追加 `.ext`，已有扩展名则保留。
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
        // 与旧的 string_content 路径一致：纯空白仍按文本保存
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
