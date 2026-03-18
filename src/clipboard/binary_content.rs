use std::{
    fs,
    io::{self, Read, Write},
};

use arboard::Clipboard;
use base64::Engine as _;

fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
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

// Copy any file as binary to clipboard.
// Strategy:
// - Try to detect if it's an image and use arboard::ImageData if so (reuse image_content logic could be improved later).
// - Otherwise, copy as base64 text so it survives clipboard text-only implementations.

pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Try image path first by delegating to image_content::copy_from_file
    if let Ok(()) = crate::clipboard::image_content::copy_from_file(fname) {
        return Ok(());
    }

    // Fallback: read raw bytes and copy as base64 text
    let mut file = fs::File::open(fname)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    // encode as base64 to safely transport binary in clipboard text
    let encoded = base64::engine::general_purpose::STANDARD.encode(&buf);

    match Clipboard::new() {
        Ok(mut clipboard) => {
            clipboard.set_text(encoded)?;
            Ok(())
        }
        Err(_) => {
            if is_ssh_session() {
                set_clipboard_via_osc52(&encoded)
            } else {
                Err("failed to set clipboard content".into())
            }
        }
    }
}

pub fn save_to_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = crate::clipboard::string_content::get_clipboard_content();
    if content.is_empty() {
        return Err("clipboard empty".into());
    }
    let res = match base64::engine::general_purpose::STANDARD.decode(content) {
        Ok(data) => {
            fs::write(fname, data)?;
            Ok(())
        }
        Err(err) => Err(err),
    };
    Ok(res?)
}
