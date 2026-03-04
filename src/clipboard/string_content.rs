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
        let mut buf = [0u8; 1024];
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(500);
        
        while start.elapsed() < timeout {
            match stdin.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    response.extend_from_slice(&buf[..n]);
                    if response.contains(&b'\x07') || response.windows(2).any(|w| w == b"\x1b\\") {
                        break;
                    }
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        
        let response_str = String::from_utf8_lossy(&response);
        if let Some(start_idx) = response_str.find("]52;c;") {
            let data_start = start_idx + 6;
            if let Some(end_idx) = response_str[data_start..].find('\x07') {
                let base64_data = &response_str[data_start..data_start + end_idx];
                use base64::engine::general_purpose;
                use base64::Engine as _;
                return general_purpose::STANDARD.decode(base64_data).ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok());
            }
            if let Some(end_idx) = response_str[data_start..].find("\x1b\\") {
                let base64_data = &response_str[data_start..data_start + end_idx];
                use base64::engine::general_purpose;
                use base64::Engine as _;
                return general_purpose::STANDARD.decode(base64_data).ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok());
            }
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