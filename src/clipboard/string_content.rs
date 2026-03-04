use std::{
    fmt::Display,
    fs,
    io::{self, Error, Write},
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
    if is_ssh_session() {
        "".to_string()
    } else {
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => clipboard.get_text().unwrap_or("".to_string()),
            Err(_) => "".to_string()
        }
    }
}

pub fn set_clipboard_content(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    if is_ssh_session() {
        set_clipboard_via_osc52(content)
    } else {
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                clipboard.set_text(content.to_string())?;
                Ok(())
            },
            Err(_) => {
                set_clipboard_via_osc52(content)
            }
        }
    }
}