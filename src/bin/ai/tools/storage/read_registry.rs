// =============================================================================
// Session read registry
// =============================================================================
// Every model-issued `read_file` call is recorded here: the path as passed, the
// resolved absolute path, and the last-read time. The `list_read_files` tool
// turns this into a recall list, so after several turns or context compression
// the model can answer "which files did I already read, and at what path"
// instead of re-locating files with find/execute_command.
//
// The dedup key is the resolved absolute path: `./x` and `x` count as one file,
// and the most recent original spelling and read time win. Symlinked aliases of
// the same file are NOT merged (paths are normalized lexically, not
// canonicalized), matching temp_registry behavior.
//
// Persists as JSON at `<temp_dir>/read_registry.json` (`temp_dir` prefers
// `~/.history_file.sessions/<session>.assets/tmp/`, same origin as
// temp_registry; fallback is the system temp dir isolated by session id), so it
// survives session suspension/resume in a new process.
// =============================================================================

use std::path::{Path, PathBuf};

/// Registry file name (relative to temp_dir).
const REGISTRY_FILENAME: &str = "read_registry.json";

/// Process-level mutex guaranteeing atomicity of the load-modify-save operations.
static REGISTRY_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

/// One recorded read.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct ReadEntry {
    /// Original path as passed to read_file (unresolved, e.g. a relative path).
    pub(crate) original: String,
    /// Resolved absolute path (dedup key).
    pub(crate) abs: String,
    /// UNIX epoch milliseconds of the most recent read (millisecond precision so
    /// the `changed` status in list_read_files cannot miss a modification that
    /// lands in the same wall-clock second as the read).
    pub(crate) last_read: u64,
}

/// Computes the registry file paths. Returns `(temp_dir, registry_file_path)`.
fn registry_paths() -> std::io::Result<(PathBuf, PathBuf)> {
    let temp_dir = crate::ai::driver::runtime_ctx::temp_dir()?;
    let registry_path = temp_dir.join(REGISTRY_FILENAME);
    Ok((temp_dir, registry_path))
}

/// Loads the registry from disk. Returns an empty list when the file does not exist.
fn load_entries(registry_path: &Path) -> Result<Vec<ReadEntry>, String> {
    if !registry_path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(registry_path)
        .map_err(|e| format!("Failed to read read registry: {e}"))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse read registry: {e}"))
}

/// Writes the registry back to disk atomically (tmp file in the same directory
/// + rename), so a crash mid-write cannot leave a truncated registry that would
/// silently kill list_read_files for the rest of the session.
fn save_entries(registry_path: &Path, entries: &[ReadEntry]) -> Result<(), String> {
    if let Some(parent) = registry_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create read registry dir: {e}"))?;
    }
    let content = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("Failed to serialize read registry: {e}"))?;
    let tmp_path = registry_path.with_extension("tmp");
    std::fs::write(&tmp_path, content)
        .map_err(|e| format!("Failed to write read registry tmp file: {e}"))?;
    std::fs::rename(&tmp_path, registry_path)
        .map_err(|e| format!("Failed to rename read registry into place: {e}"))
}

fn now_unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Records a successful read of `abs` (the resolved absolute path) under the
/// caller's original spelling `original`. Idempotent per resolved path: a
/// re-read updates the spelling and the last-read time. When nothing changed
/// (same path, same spelling, same millisecond), the disk save is skipped so
/// paging loops do not pay file IO per page.
pub(crate) fn register(original: &str, abs: &Path) -> Result<(), String> {
    let _guard = REGISTRY_LOCK
        .lock()
        .map_err(|e| format!("Failed to lock read registry: {e}"))?;
    let (_, registry_path) =
        registry_paths().map_err(|e| format!("Failed to get temp dir: {e}"))?;
    let abs_str = abs.to_string_lossy().into_owned();
    let now = now_unix_millis();
    let mut entries = load_entries(&registry_path)?;
    match entries.iter_mut().find(|e| e.abs == abs_str) {
        Some(e) if e.original == original && e.last_read == now => return Ok(()),
        Some(e) => {
            e.original = original.to_string();
            e.last_read = now;
        }
        None => entries.push(ReadEntry {
            original: original.to_string(),
            abs: abs_str,
            last_read: now,
        }),
    }
    save_entries(&registry_path, &entries)
}

/// Lists all recorded reads, newest first. Returns an empty list on any
/// registry failure (callers treat the list as best-effort).
pub(crate) fn list() -> Vec<ReadEntry> {
    let Ok(_guard) = REGISTRY_LOCK.lock() else {
        return Vec::new();
    };
    let Ok((_, registry_path)) = registry_paths() else {
        return Vec::new();
    };
    let Ok(mut entries) = load_entries(&registry_path) else {
        return Vec::new();
    };
    entries.sort_by(|a, b| b.last_read.cmp(&a.last_read));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;

    /// Isolated session id so temp_dir() lands in a private per-test directory
    /// (same pattern as temp_registry tests).
    fn isolated_scope<T>(f: impl FnOnce() -> T) -> T {
        let session_id = format!("read_reg_test_{}", uuid::Uuid::new_v4());
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope((session_id, 0usize), f)
    }

    #[test]
    fn register_dedups_by_resolved_path_and_keeps_latest_spelling() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        isolated_scope(|| {
            let abs = Path::new("/tmp/read_registry_test.txt");
            register("test.txt", abs).unwrap();
            register("./test.txt", abs).unwrap(); // same file, new spelling
            register("other.txt", Path::new("/tmp/other.txt")).unwrap();

            let entries = list();
            assert_eq!(entries.len(), 2);
            let mine = entries
                .iter()
                .find(|e| e.abs == "/tmp/read_registry_test.txt")
                .expect("first path must be present");
            assert_eq!(mine.original, "./test.txt"); // latest spelling wins
            assert!(mine.last_read > 0);
        });
    }

    #[test]
    fn list_orders_newest_first() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        isolated_scope(|| {
            // Fabricate a registry with known timestamps to make ordering
            // deterministic (register() uses wall-clock seconds).
            let (_, registry_path) = registry_paths().unwrap();
            let entries = vec![
                ReadEntry {
                    original: "old.txt".to_string(),
                    abs: "/tmp/old.txt".to_string(),
                    last_read: 100_000,
                },
                ReadEntry {
                    original: "new.txt".to_string(),
                    abs: "/tmp/new.txt".to_string(),
                    last_read: 200_000,
                },
            ];
            save_entries(&registry_path, &entries).unwrap();

            let listed = list();
            assert_eq!(listed.len(), 2);
            assert_eq!(listed[0].abs, "/tmp/new.txt");
            assert_eq!(listed[1].abs, "/tmp/old.txt");
        });
    }

    #[test]
    fn registry_persists_across_loads() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        isolated_scope(|| {
            register("/tmp/a.txt", Path::new("/tmp/a.txt")).unwrap();
            register("/tmp/b.txt", Path::new("/tmp/b.txt")).unwrap();
            // Reloading from disk (a fresh list()) sees both entries.
            let all = list();
            assert_eq!(all.len(), 2);
        });
    }
}
