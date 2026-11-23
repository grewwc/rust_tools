#[cfg(unix)]
use libc::{LOCK_EX, LOCK_UN, flock};
use std::fs;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

pub(crate) mod changes;
pub(crate) mod command_runner;
pub(crate) mod command_session;
pub(in crate::ai) mod decision_receipts;
pub(crate) mod file_store;
pub(crate) mod memory_index;
pub(crate) mod memory_store;
pub(crate) mod mutation_log;
pub(crate) mod overflow_cache;
pub(crate) mod process_registry;
pub(crate) mod rag_store;
pub(crate) mod read_registry;
pub(crate) mod temp_registry;
pub(crate) mod text_utils;
pub(crate) mod token_usage_store;

/// Serializes in-process operations across all memory files.
static MEMORY_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Runs a memory operation under the process mutex, with best-effort Unix flock.
pub(crate) fn with_memory_file_lock<F, T>(path: &Path, f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let _guard = MEMORY_FILE_LOCK
        .lock()
        .map_err(|e| format!("Failed to acquire memory file lock: {}", e))?;

    #[cfg(unix)]
    {
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

    // Non-Unix platforms and failed opens rely on the process mutex alone.
    f()
}
