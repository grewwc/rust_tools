//! Text clipboard content handling.
//!
//! Provides reading, writing, and file operations for the text clipboard.
//! Supports the local clipboard and the OSC52 protocol over SSH sessions.

use std::{
    fmt::Display,
    fs,
    io::{self, Error, Write},
    time::Duration,
};

use crate::commonw::filename::add_suffix;

/// Error for non-text files.
#[derive(Debug)]
struct NonTextErr(String);

impl NonTextErr {
    fn new(msg: String) -> Self {
        Self(msg)
    }
}

impl Display for NonTextErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NonTextErr {}

/// Check whether the current session is an SSH session.
///
/// Detected by checking the following environment variables:
/// - `SSH_CONNECTION`
/// - `SSH_CLIENT`
/// - `SSH_TTY`
///
/// # Returns
///
/// Returns `true` when inside an SSH session, otherwise `false`.
fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("SSH_TTY").is_ok()
}

/// Set clipboard content via the OSC52 escape sequence.
///
/// OSC52 is a terminal escape sequence that lets an application set the
/// terminal's clipboard content. It is especially useful in SSH sessions,
/// where the local clipboard can be manipulated from the remote server.
///
/// # Arguments
///
/// * `content` - The text content to copy to the clipboard
///
/// # Returns
///
/// - `Ok(())` - Clipboard set successfully
/// - `Err(...)` - Failed to set the clipboard
///
/// # How it works
///
/// 1. Base64-encode the content
/// 2. Send the OSC52 escape sequence: `\x1b]52;c;<base64>\x07`
/// 3. The terminal decodes it and sets the clipboard
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

/// Check whether standard input is a TTY.
fn stdin_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Check whether standard output is a TTY.
fn stdout_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

/// Read raw bytes via an OSC52 terminal query.
///
/// Sends the OSC52 query sequence, waits for the terminal response, and
/// returns the raw clipboard bytes. The content is not required to be
/// valid UTF-8.
///
/// # Returns
///
/// - `Some(Vec<u8>)` - Clipboard bytes read successfully
/// - `None` - Read failed (not a TTY, timeout, etc.)
///
/// # Notes
///
/// - The terminal must support OSC52 queries
/// - Temporarily switches the terminal to non-canonical mode
/// - Adaptive read timeouts: first byte 3s, idle 1s, absolute cap 120s
fn read_osc52_bytes() -> Option<Vec<u8>> {
    if !stdin_is_tty() || !stdout_is_tty() {
        return None;
    }

    // Use raw file descriptors for I/O, bypassing the Rust stdio buffering
    // layer to avoid conflicts with crossterm's stdin/stdout management.

    // Save the original terminal settings.
    let fd = libc::STDIN_FILENO;
    let mut original_termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut original_termios) } != 0 {
        return None;
    }

    // Switch to non-canonical mode (return immediately).
    let mut new_termios = original_termios;
    new_termios.c_lflag &= !(libc::ICANON | libc::ECHO);
    new_termios.c_cc[libc::VMIN] = 0;
    new_termios.c_cc[libc::VTIME] = 1;

    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &new_termios) } != 0 {
        return None;
    }

    // Send the OSC52 query via libc::write (bypasses the stdout buffer).
    let query = b"\x1b]52;c;?\x07";
    let write_result = unsafe {
        libc::write(
            libc::STDOUT_FILENO,
            query.as_ptr() as *const libc::c_void,
            query.len(),
        )
    };
    if write_result < 0 {
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original_termios) };
        return None;
    }

    // Read the response via libc::read (bypasses the stdin buffer).
    //
    // Image base64 bridged by `oo -B` can push the OSC52 response into the
    // multi-MB range (response content = base64(base64(image))), and a fixed
    // 1.5s short timeout would give up mid-transfer and restore ECHO, so the
    // response tail gets echoed back by the remote pty onto the terminal
    // (the terminal visibly floods with base64). Therefore:
    //   - First-byte timeout: terminals that do not support OSC52 queries
    //     never respond, avoiding an infinite wait
    //   - Stop immediately once the end marker `\x07` / `\x1b\\` arrives
    //   - Idle timeout: data was received but nothing new for a long time,
    //     treat the transfer as finished
    //   - Absolute cap as a final backstop
    let result = (|| {
        let mut response = Vec::new();
        let mut buf = [0u8; 65536];

        let first_byte_timeout = Duration::from_secs(3);
        let idle_timeout = Duration::from_millis(1000);
        let absolute_timeout = Duration::from_secs(120);

        let start = std::time::Instant::now();
        let mut last_data = start;

        loop {
            let elapsed = start.elapsed();
            if elapsed >= absolute_timeout {
                break;
            }
            // No data at all (terminal does not support OSC52 queries).
            if response.is_empty() && elapsed >= first_byte_timeout {
                break;
            }
            // Data was received but nothing new for a long time → treat the
            // transfer as finished.
            if !response.is_empty() && last_data.elapsed() >= idle_timeout {
                break;
            }

            // Keep the poll timeout within the remaining first-byte/idle
            // budget to avoid busy-waiting.
            let mut poll_timeout = 200i32;
            if response.is_empty() {
                let remain = first_byte_timeout.saturating_sub(elapsed);
                poll_timeout = poll_timeout.min(remain.as_millis().max(1) as i32);
            } else {
                let remain = idle_timeout.saturating_sub(last_data.elapsed());
                poll_timeout = poll_timeout.min(remain.as_millis().max(1) as i32);
            }

            let mut pollfd = libc::pollfd {
                fd: libc::STDIN_FILENO,
                events: libc::POLLIN,
                revents: 0,
            };
            let n_ready = unsafe { libc::poll(&mut pollfd, 1, poll_timeout) };
            if n_ready <= 0 {
                continue;
            }
            let n = unsafe {
                libc::read(
                    libc::STDIN_FILENO,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                )
            };
            if n > 0 {
                let n = n as usize;
                response.extend_from_slice(&buf[..n]);
                last_data = std::time::Instant::now();
                // Stop as soon as the response end marker arrives.
                if response.contains(&b'\x07') || response.windows(2).any(|w| w == b"\x1b\\") {
                    break;
                }
            }
        }

        // Parse the response and extract the Base64 data.
        let response_str = String::from_utf8_lossy(&response);
        if let Some(start_idx) = response_str.find("]52;c;") {
            let data_start = start_idx + 6;
            use base64::Engine as _;
            use base64::engine::general_purpose;
            if let Some(end_idx) = response_str[data_start..].find('\x07') {
                let base64_data = &response_str[data_start..data_start + end_idx];
                return general_purpose::STANDARD.decode(base64_data).ok();
            }
            if let Some(end_idx) = response_str[data_start..].find("\x1b\\") {
                let base64_data = &response_str[data_start..data_start + end_idx];
                return general_purpose::STANDARD.decode(base64_data).ok();
            }
        }
        None
    })();

    // Flush any unread input before restoring the terminal settings (when
    // giving up on a timeout, a response tail may still be buffered) so that
    // leftover content is not echoed to the terminal once ECHO is restored.
    unsafe { libc::tcflush(fd, libc::TCIFLUSH) };
    // Restore the original terminal settings.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original_termios) };

    result
}

/// Read clipboard text via OSC52.
fn get_clipboard_via_osc52() -> Option<String> {
    read_osc52_bytes().and_then(|bytes| String::from_utf8(bytes).ok())
}

/// Read raw clipboard bytes via OSC52 (SSH sessions only).
///
/// # Returns
///
/// - `Some(Vec<u8>)` - Clipboard bytes read successfully in an SSH session
/// - `None` - Not an SSH session, or the read failed
///
/// # Example
///
/// ```rust,no_run
/// use rust_tools::clipboardw::get_clipboard_raw_bytes_via_osc52;
///
/// if let Some(bytes) = get_clipboard_raw_bytes_via_osc52() {
///     println!("读取到 {} 字节", bytes.len());
/// }
/// ```
pub fn get_clipboard_raw_bytes_via_osc52() -> Option<Vec<u8>> {
    if is_ssh_session() {
        read_osc52_bytes()
    } else {
        None
    }
}

/// Save clipboard content to a file.
///
/// Reads the current clipboard text content and saves it to the given file.
/// If the file name has no extension, `.txt` is appended automatically.
///
/// # Arguments
///
/// * `fname` - Target file name
///
/// # Returns
///
/// - `Ok(())` - Saved successfully
/// - `Err(io::Error)` - Save failed (empty clipboard or IO error)
///
/// # Example
///
/// ```rust,no_run
/// use rust_tools::clipboardw::save_to_file;
///
/// save_to_file("clipboard_content.txt").expect("保存失败");
/// ```
pub fn save_to_file(fname: &str) -> io::Result<()> {
    let fname = add_suffix(fname, ".txt", || !fname.contains('.'));
    let text = get_clipboard_content();
    if !text.is_empty() {
        fs::write(fname.as_str(), text)?;
        println!("save to file: {fname}");
        Ok(())
    } else {
        Err(Error::other("no text"))
    }
}

/// Copy file content into the clipboard.
///
/// Reads the file content and sets it as the clipboard content.
///
/// # Arguments
///
/// * `fname` - Source file name
///
/// # Returns
///
/// - `Ok(())` - Copied successfully
/// - `Err(...)` - Copy failed (file empty or not a text file)
///
/// # Example
///
/// ```rust,no_run
/// use rust_tools::clipboardw::copy_from_file;
///
/// copy_from_file("document.txt").expect("复制失败");
/// ```
pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text = match fs::read_to_string(fname) {
        Ok(text) => text,
        Err(_) => "".to_string(),
    };
    if text.is_empty() {
        return Err(Box::new(NonTextErr::new(format!(
            "{} is not text file.",
            fname
        ))));
    }
    set_clipboard_content(&text)?;
    Ok(())
}

/// Get the clipboard text content.
///
/// Auto-detects the environment and reads the clipboard with the appropriate
/// method:
/// - Local session: uses the `arboard` crate
/// - SSH session: uses the OSC52 protocol
///
/// # Returns
///
/// Returns the clipboard text content, or an empty string if reading fails.
///
/// # Example
///
/// ```rust,no_run
/// use rust_tools::clipboardw::get_clipboard_content;
///
/// let text = get_clipboard_content();
/// println!("剪贴板内容：{}", text);
/// ```
pub fn get_clipboard_content() -> String {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => clipboard.get_text().unwrap_or_default(),
        Err(_) => {
            if is_ssh_session() {
                get_clipboard_via_osc52().unwrap_or_default()
            } else {
                String::new()
            }
        }
    }
}

/// Set the clipboard text content.
///
/// Auto-detects the environment and sets the clipboard with the appropriate
/// method:
/// - Local session: uses the `arboard` crate
/// - SSH session: uses the OSC52 protocol
///
/// # Arguments
///
/// * `content` - The text content to set
///
/// # Returns
///
/// - `Ok(())` - Set successfully
/// - `Err(...)` - Failed to set the clipboard
///
/// # Example
///
/// ```rust,no_run
/// use rust_tools::clipboardw::set_clipboard_content;
///
/// set_clipboard_content("Hello, World!").expect("设置失败");
/// ```
pub fn set_clipboard_content(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            clipboard.set_text(content.to_string())?;
            Ok(())
        }
        Err(_) => {
            if is_ssh_session() {
                set_clipboard_via_osc52(content)
            } else {
                Err("failed to set clipboard content".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_ssh_session;

    #[test]
    fn test_ssh_session_detection() {
        // This test depends on the environment variables; it only verifies
        // that the function does not panic.
        let _ = is_ssh_session();
    }
}
