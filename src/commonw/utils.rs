use std::borrow::Cow;
use std::{fs::File, io, path::Path};

pub fn get_home_dir() -> Option<String> {
    if let Ok(home) = std::env::var("HOME") {
        return Some(home);
    }
    None
}

pub fn expanduser(path_str: &str) -> Cow<'_, str> {
    if let Some(home) = get_home_dir() {
        return Cow::Owned(path_str.replace("~", &home));
    }
    Cow::Borrowed(path_str)
}

pub fn open_file_for_write_truncate(path: &Path, mode: u32) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(mode);
    }
    options.open(path)
}

pub fn open_file_for_append(path: &Path, mode: u32) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(mode);
    }
    options.open(path)
}
