use std::io;

mod storage {
    use rust_tools::common::utils::get_home_dir;
    use std::{fs, io, path::PathBuf};

    pub fn get_storage_path() -> io::Result<PathBuf> {
        let home = get_home_dir().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "Could not find home directory")
        })?;
        Ok(PathBuf::from(home).join(".rust_tools_p.txt"))
    }

    pub fn read_previous_content(path: &PathBuf) -> io::Result<Option<String>> {
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(path)?;
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        Ok(Some(trimmed.to_string()))
    }

    pub fn save_content(path: &PathBuf, content: &str) -> io::Result<()> {
        fs::write(path, content)
    }

    pub fn clear_content(path: &PathBuf) -> io::Result<()> {
        fs::write(path, "")
    }
}

mod clipboard_handler {
    use rust_tools::clipboard::string_content::{get_clipboard_content, set_clipboard_content};
    use std::io;

    pub fn restore_from_storage(content: &str) -> io::Result<()> {
        set_clipboard_content(content).map_err(|err| {
            let msg = format!("failed to copy to clipboard. err: {}", err);
            eprintln!("{}", msg);
            io::Error::new(io::ErrorKind::Other, msg)
        })?;

        println!("copied.");
        Ok(())
    }

    pub fn save_to_storage() -> String {
        get_clipboard_content()
    }
}

fn main() -> io::Result<()> {
    let path = storage::get_storage_path()?;

    if let Some(prev_content) = storage::read_previous_content(&path)? {
        clipboard_handler::restore_from_storage(&prev_content)?;
        storage::clear_content(&path)?;
    } else {
        let text = clipboard_handler::save_to_storage();
        storage::save_content(&path, &text)?;
        println!("saved.");
    }

    Ok(())
}
