use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
};

use arboard::{Clipboard, ImageData};
use image::{ImageBuffer, ImageFormat, Rgb, Rgba, buffer::ConvertBuffer};

use crate::common::filename::add_suffix;

fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

fn stdout_is_tty() -> bool {
    // OSC52 only makes sense on interactive terminals.
    unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

fn prefer_osc52_image_bridge() -> bool {
    // Default to OSC52 transport so local copy can be pasted from remote SSH sessions.
    // Set OO_PREFER_NATIVE_IMAGE=1 to keep the previous native-image clipboard behavior.
    match std::env::var("OO_PREFER_NATIVE_IMAGE") {
        Ok(val) => {
            !matches!(
                val.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        }
        Err(_) => true,
    }
}

fn set_clipboard_via_osc52(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine as _;
    use base64::engine::general_purpose;

    let encoded = general_purpose::STANDARD.encode(content);
    let osc52 = format!("\x1b]52;c;{}\x07", encoded);

    let mut stdout = io::stdout();
    stdout.write_all(osc52.as_bytes())?;
    stdout.flush()?;

    Ok(())
}

fn image_to_base64(img: &image::DynamicImage) -> Result<String, Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    use base64::Engine as _;
    use base64::engine::general_purpose;
    Ok(general_purpose::STANDARD.encode(&buf))
}

/// Bridge: read image from native clipboard, re-encode as base64 TEXT in clipboard.
/// Run this on your LOCAL machine so that `oo -p` on a remote SSH server can
/// retrieve the image via OSC52 (which only carries text).
pub fn bridge_image_to_text_clipboard() -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = Clipboard::new()?;
    let image = clipboard.get_image()?;
    let data = image.bytes.to_vec();
    let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
        image.width as u32,
        image.height as u32,
        data,
    )
    .ok_or("failed to create image buffer")?;
    let dynamic = image::DynamicImage::ImageRgba8(img_buf);
    let b64 = image_to_base64(&dynamic)?;
    clipboard.set_text(b64)?;
    println!("Image encoded as text in clipboard — ready for `oo -p` on remote.");
    Ok(())
}

pub fn save_to_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let fname: String = add_suffix(fname, ".jpg", || !fname.contains('.'));

    // Helper function to try saving from clipboard via OSC 52.
    // Handles two cases:
    //   1. Native local image copy: terminal responds with base64(raw_image_bytes).
    //   2. oo -c image copy: terminal responds with base64(base64(image_bytes)) —
    //      the clipboard text is itself a base64 string that wraps the image.
    fn try_osc52_save(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
        let raw_bytes = crate::clipboard::string_content::get_clipboard_raw_bytes_via_osc52()
            .ok_or_else(|| io::Error::other("no clipboard data via OSC52"))?;

        // Case 1: raw_bytes are directly a valid image (native clipboard copy).
        if let Ok(img) = image::load_from_memory(&raw_bytes) {
            img.save(fname)?;
            println!("save to file: {fname}");
            return Ok(());
        }

        // Case 2: raw_bytes are a UTF-8 base64 string (from oo -c bridge).
        use base64::Engine as _;
        use base64::engine::general_purpose;
        let b64_str = std::str::from_utf8(&raw_bytes)
            .map(|s| s.replace(['\n', '\r'], ""))
            .map_err(|_| io::Error::other("clipboard data is not a valid image or base64 string"))?;
        let data = general_purpose::STANDARD.decode(&b64_str).map_err(|e| {
            let msg = format!("failed to decode clipboard base64: {}", e);
            io::Error::new(io::ErrorKind::InvalidData, msg)
        })?;
        let img = image::load_from_memory(&data).map_err(|e| {
            let msg = format!("failed to load image from clipboard data: {}", e);
            io::Error::new(io::ErrorKind::InvalidData, msg)
        })?;
        img.save(fname)?;
        println!("save to file: {fname}");
        Ok(())
    }

    match Clipboard::new() {
        Ok(mut clipboard) => {
            if let Ok(image) = clipboard.get_image() {
                let data = image.bytes;
                let image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                    image.width as u32,
                    image.height as u32,
                    data.to_vec(),
                )
                .ok_or("failed to create image")?;
                let image: ImageBuffer<Rgb<u8>, Vec<u8>> = image.convert();
                image.save(fname.as_str())?;
                println!("save to file: {fname}");
                Ok(())
            } else {
                // Try fallback if arboard works but has no image
                match try_osc52_save(&fname) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(Box::new(io::Error::other(
                        "no image found (local or remote)",
                    ))),
                }
            }
        }
        Err(_) => match try_osc52_save(&fname) {
            Ok(_) => Ok(()),
            Err(e) => {
                if is_ssh_session() {
                    Err(e)
                } else {
                    Err(Box::new(io::Error::other("no image")))
                }
            }
        },
    }
}

fn open_by_content(path: &str) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Peek at first 12 bytes to detect format (some formats need more bytes)
    let mut header = [0; 12];
    reader.read_exact(&mut header)?;

    let format = if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        // PNG magic bytes
        ImageFormat::Png
    } else if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        // JPEG magic bytes
        ImageFormat::Jpeg
    } else if header.starts_with(&[0x47, 0x49, 0x46, 0x38, 0x37, 0x61])
        || header.starts_with(&[0x47, 0x49, 0x46, 0x38, 0x39, 0x61])
    {
        // GIF magic bytes (GIF87a or GIF89a)
        ImageFormat::Gif
    } else if header.starts_with(&[0x52, 0x49, 0x46, 0x46])
        && header[8..12] == [0x57, 0x45, 0x42, 0x50]
    {
        // WebP magic bytes (RIFF....WEBP)
        ImageFormat::WebP
    } else if header.starts_with(&[0x42, 0x4D]) {
        // BMP magic bytes
        ImageFormat::Bmp
    } else if header.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || header.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        // TIFF magic bytes (little-endian or big-endian)
        ImageFormat::Tiff
    } else if header.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        // ICO magic bytes
        ImageFormat::Ico
    } else if header[0..4] == [0x71, 0x6F, 0x69, 0x66] {
        // QOI magic bytes "qoif"
        ImageFormat::Qoi
    } else {
        return Err(format!(
            "Unsupported image format. Supported formats: PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, QOI. Header bytes: {:02X?}", 
            &header[..8]
        ).into());
    };

    // Rewind by creating a new reader with header + rest
    use std::io::Cursor;
    let mut full_data = Vec::new();
    full_data.extend_from_slice(&header);
    reader.read_to_end(&mut full_data)?;

    let cursor = Cursor::new(full_data);
    Ok(image::load(cursor, format)?)
}

pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let img = open_by_content(fname)?;
    let base64_data = image_to_base64(&img)?;

    if stdout_is_tty() && prefer_osc52_image_bridge() {
        if set_clipboard_via_osc52(&base64_data).is_ok() {
            return Ok(());
        }
    }

    match Clipboard::new() {
        Ok(mut clipboard) => {
            let img_rgba = img.to_rgba8();
            let width = img_rgba.width();
            let height = img_rgba.height();
            let bytes = img_rgba.into_raw();

            let image_data = ImageData {
                width: width as usize,
                height: height as usize,
                bytes: std::borrow::Cow::Owned(bytes),
            };

            clipboard.set_image(image_data)?;
            Ok(())
        }
        Err(_) => {
            if is_ssh_session() {
                set_clipboard_via_osc52(&base64_data)
            } else {
                Err("failed to set clipboard content".into())
            }
        }
    }
}
