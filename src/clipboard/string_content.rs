use std::{
    fmt::Display,
    fs,
    io::{self, Error},
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

pub fn save_to_file(fname: &str) -> io::Result<()> {
    let mut clipboard = arboard::Clipboard::new().unwrap();
    let fname = add_suffix(fname, ".txt", || !fname.contains('.'));
    if let Ok(text) = clipboard.get_text() {
        fs::write(fname.as_str(), text)?;
        println!("save to file: {fname}");
        Ok(())
    } else {
        Err(Error::new(io::ErrorKind::Other, "no text"))
    }
}

pub fn copy_from_file(fname: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = arboard::Clipboard::new().unwrap();
    let text = match fs::read_to_string(fname) {
        Ok(text) => text, 
        Err(_) => {
            // eprintln!("{err:?}");
            "".to_string()
        }
    };
    if text.is_empty() {
        return Err(Box::new(NonTextErr::new(format!(
            "{} is not text file.",
            fname
        ))));
    }
    // println!("text:{}, len:{}", text, text.len());
    clipboard.set_text(text)?;
    Ok(())
}

pub fn get_clipboard_content() -> String {
    let mut clipboard = arboard::Clipboard::new().unwrap();
    clipboard.get_text().unwrap_or("".to_string())
}

pub fn set_clipboard_content(content: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = arboard::Clipboard::new().unwrap();
    clipboard.set_text(content.to_string())?;
    Ok(())
}