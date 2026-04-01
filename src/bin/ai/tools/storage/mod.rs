use std::path::Path;
use std::sync::{LazyLock, Mutex};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use libc::{flock, LOCK_EX, LOCK_UN};
use std::fs;

pub(crate) mod command_runner;
pub(crate) mod file_store;
pub(crate) mod memory_store;
pub(crate) mod knowledge_cache;
pub(crate) mod knowledge_fingerprint;
pub(crate) mod knowledge_types;

/// 文件锁，用于并发访问 memory 文件
static MEMORY_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// 对 memory 文件执行操作，持有排他锁
pub(crate) fn with_memory_file_lock<F, T>(path: &Path, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let _guard = MEMORY_FILE_LOCK
        .lock()
        .map_err(|e| format!("Failed to acquire memory file lock: {}", e))?;
    
    #[cfg(unix)]
    {
        // Unix: 使用 flock 进行文件级锁
        if let Ok(file) = fs::OpenOptions::new().write(true).open(path) {
            unsafe {
                flock(file.as_raw_fd(), LOCK_EX);
            }
            let result = f();
            unsafe {
                flock(file.as_raw_fd(), LOCK_UN);
            }
            return result;
        }
    }
    
    // 回退：仅使用互斥锁
    f()
}
