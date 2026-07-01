use std::borrow::Cow;
use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
};

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
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
    }
}

/// 获取缓存目录（跨平台）
///
/// - macOS: `$HOME/Library/Caches`
/// - Linux: `$HOME/.cache`
/// - Windows: `%LOCALAPPDATA%`
pub fn get_cache_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library").join("Caches"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        // Linux 和其他 Unix 系统
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache"))
    }
}

/// 展开路径**开头**的 `~`，遵循 shell 语义：
/// - `~` 或 `~/rest` → `$HOME`[`/rest`]
/// - 中间/结尾出现的 `~` 原样保留，避免误伤合法文件名（如 `backup~`）
/// - `~user`（其他用户）需要查 passwd，超出范围，原样返回
pub fn expanduser(path_str: &str) -> Cow<'_, str> {
    if path_str == "~" || path_str.starts_with("~/") {
        if let Some(home) = get_home_dir() {
            // 开头的 `~` 一定是第一个字符，replacen 只替换它。
            return Cow::Owned(path_str.replacen('~', &home, 1));
        }
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

#[cfg(test)]
mod tests {
    use super::{expanduser, get_home_dir};

    #[test]
    fn expanduser_expands_leading_tilde_slash() {
        let Some(home) = get_home_dir() else {
            return;
        };
        let expanded = expanduser("~/.config/rust_tools/skills");
        assert_eq!(expanded, format!("{home}/.config/rust_tools/skills"));
    }

    #[test]
    fn expanduser_expands_bare_tilde() {
        let Some(home) = get_home_dir() else {
            return;
        };
        assert_eq!(expanduser("~"), home);
    }

    #[test]
    fn expanduser_leaves_non_leading_tilde_untouched() {
        // 绝对路径不变
        assert_eq!(expanduser("/etc/hosts"), "/etc/hosts");
        // 相对路径不变
        assert_eq!(expanduser("src/main.rs"), "src/main.rs");
        // 只展开开头的 `~`：中间/结尾的 `~` 不受影响（旧实现的全量 replace 会误伤）
        assert_eq!(expanduser("a/b~c/backup~"), "a/b~c/backup~");
        // `~user`（非当前用户）不处理
        assert_eq!(expanduser("~other/file"), "~other/file");
    }
}
