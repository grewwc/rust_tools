use std::{
    fs,
    io::{self, Error},
};

use crate::common::filename::add_suffix;

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
    let text = fs::read_to_string(fname)?;
    clipboard.set_text(text)?;
    Ok(())
}
