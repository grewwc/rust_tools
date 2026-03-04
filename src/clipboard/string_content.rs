use std::{
    fmt::Display,
    fs,
    io::{self, Error, Read, Write},
    time::Duration,
};

use crate::common::filename::add_suffix;

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

fn is_ssh_session() -> bool {
    std::env::var("SSH_CONNECTION").is_ok() || 
    std::env::var("SSH_CLIENT").is_ok() ||
    std::env::var("SSH_TTY").is_ok()
}

fn set_clipboard_via_osc52(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use base64::engine::general_purpose;
    use base64::Engine as _;
    
    let encoded = general_purpose::STANDARD.encode(content);
    let osc52 = format!("\x1b]52;c;{}\x07", encoded);
    
    let mut stdout = io::stdout();
    stdout.write_all(osc52.as_bytes())?;
    stdout.flush()?;
    
    Ok(())
}

fn get_clipboard_via_osc52() -> Option<String> {
    use std::os::unix::io::AsRawFd;
    
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    
    stdout.write_all(b"\x1b]52;c;?\x07").ok()?;
    stdout.flush().ok()?;
    
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    
    let fd = stdin.as_raw_fd();
    let mut original_termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(fd, &mut original_termios) } != 0 {
        return None;
    }
    
    let mut new_termios = original_termios;
    new_termios.c_lflag &= !(libc::ICANON | libc::ECHO);
    new_termios.c_cc[libc::VMIN] = 0;
    new_termios.c_cc[libc::VTIME] = 1;
    
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &new_termios) } != 0 {
        return None;
    }
    
    let result = (|| {
        let mut response = Vec::new();
        let mut buf = [0u8; 4096];
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(10);
        let mut found_start = false;
        let mut start_pos = 0;
        
        while start.elapsed() < timeout {
            match stdin.read(&mut buf) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(10));
                },
                Ok(n) => {
                    let old_len = response.len();
                    response.extend_from_slice(&buf[..n]);
                    
                    if !found_start {
                        // Search for start sequence, potentially spanning the last read boundary
                        // We search from old_len - 7 (to handle split start sequence)
                        let search_start = if old_len > 7 { old_len - 7 } else { 0 };
                        if let Some(pos) = response[search_start..].windows(7).position(|w| w == b"\x1b]52;c;") {
                            found_start = true;
                            start_pos = search_start + pos;
                        } else if response.len() > 1024 * 1024 * 10 { // 10MB limit without start
                            break;
                        }
                    }
                    
                    if found_start {
                        // Check for end sequence in the newly added part
                        // Be careful about \x1b\ splitting across read boundary
                        // Also ensure we don't check before the payload starts
                        let check_start = if old_len > 0 { old_len - 1 } else { start_pos + 7 };
                        let check_start = std::cmp::max(check_start, start_pos + 7);
                        
                        if check_start < response.len() {
                            let check_slice = &response[check_start..];
                            if check_slice.contains(&b'\x07') || check_slice.windows(2).any(|w| w == b"\x1b\\") {
                                break;
                            }
                        }
                    }
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        
        // Try to decode as much as possible even if truncated or slightly malformed
        let response_str = String::from_utf8_lossy(&response);
        // eprintln!("DEBUG: OSC52 raw response len: {}", response.len());
        // if response.len() < 100 {
        //    eprintln!("DEBUG: OSC52 raw response: {:?}", response_str);
        // }

        if let Some(start_idx) = response_str.find("]52;c;") {
            let data_start = start_idx + 6;
            
            // Find end index, checking both terminators
            let end_idx = response_str[data_start..].find('\x07')
                .or_else(|| response_str[data_start..].find("\x1b\\"));
                
            if let Some(len) = end_idx {
                let base64_data = &response_str[data_start..data_start + len];
                // Remove newlines if present (some terminals split output)
                let clean_base64 = base64_data.replace('\n', "").replace('\r', "");
                
                use base64::engine::general_purpose;
                use base64::Engine as _;
                match general_purpose::STANDARD.decode(&clean_base64) {
                    Ok(bytes) => {
                         // eprintln!("DEBUG: decoded {} bytes", bytes.len());
                         return String::from_utf8(bytes).ok();
                    },
                    Err(_e) => {
                        // eprintln!("DEBUG: base64 decode failed: {}", _e);
                        return None;
                    }
                }
            } else {
                // eprintln!("DEBUG: end terminator not found");
            }
        } else {
            // eprintln!("DEBUG: start sequence not found");
        }
        None
    })();
    
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original_termios) };
    
    result
}

pub fn save_to_file(fname: &str) -> io::Result<()> {
    let fname = add_suffix(fname, ".txt", || !fname.contains('.'));
    let text = get_clipboard_content();
    if !text.is_empty() {
        fs::write(fname.as_str(), text)?;
        println!("save to file: {fname}");
        Ok(())
    } else {
        Err(Error::new(io::ErrorKind::Other, "no text"))
    }
}

pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text = match fs::read_to_string(fname) {
        Ok(text) => text, 
        Err(_) => {
            "".to_string()
        }
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

pub fn set_clipboard_content(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            clipboard.set_text(content.to_string())?;
            Ok(())
        },
        Err(_) => {
            if is_ssh_session() {
                set_clipboard_via_osc52(content)
            } else {
                Err("failed to set clipboard content".into())
            }
        }
    }
}