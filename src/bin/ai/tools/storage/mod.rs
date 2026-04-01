use std::path::Path;
use std::sync::{LazyLock, Mutex};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use libc::{flock, LOCK_EX, LOCK_UN};

pub(crate) mod command_runner;
pub(crate) mod file_store;
pub(crate) mod memory_store;

static MEMORY_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

pub(crate) fn with_memory_lock<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    if let Ok(_g) = MEMORY_FILE_LOCK.lock() {
        f()
    } else {
        f()
    }
}

pub(crate) fn with_memory_file_lock<F, T>(path: &Path, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let _g = MEMORY_FILE_LOCK.lock().map_err(|_| "lock poisoned".to_string())?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("Failed to open memory file for lock: {}", e))?;
    #[cfg(unix)]
    unsafe {
        if flock(file.as_raw_fd(), LOCK_EX) != 0 {
            return Err("Failed to acquire file lock".to_string());
        }
    }
    let result = f();
    #[cfg(unix)]
    unsafe {
        let _ = flock(file.as_raw_fd(), LOCK_UN);
    }
    drop(file);
    result
}
