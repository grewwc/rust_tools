use std::borrow::Cow;
use std::{fs::File, io, path::{Path, PathBuf}};

pub fn get_home_dir() -> Option<String> {
    if let Ok(home) = std::env::var("HOME") {
        return Some(home);
    }
    None
}

/// 获取配置目录（跨平台）
/// 
/// - macOS/Linux: `$HOME/.config`
/// - Windows: `%APPDATA%`
pub fn get_config_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        // macOS, Linux, 和其他 Unix 系统
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".config"))
    }
}

/// 获取缓存目录（跨平台）
/// 
/// - macOS: `$HOME/Library/Caches`
/// - Linux: `$HOME/.cache`
/// - Windows: `%LOCALAPPDATA%`
pub fn get_cache_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("Library").join("Caches"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        // Linux 和其他 Unix 系统
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".cache"))
    }
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
