// =============================================================================
// Persistent temp-file registry
// =============================================================================
// Temporary files created by the agent via `write_file(temp=true)` leave a record in this
// registry. The registry serves audit tracking; temp files are cleaned up by the runtime at
// session end.
//
// The registry persists as a JSON file at `<temp_dir>/temp_registry.json` (`temp_dir` prefers
// `~/.history_file.sessions/<session>.assets/tmp/`, same origin as tool-overflow and outside
// the project; fallback is the system temp dir), so it survives restarts after a session ends.
// =============================================================================

use std::path::PathBuf;

use rust_tools::commonw::FastSet;

/// Registry file name (relative to temp_dir).
const REGISTRY_FILENAME: &str = "temp_registry.json";

/// Process-level mutex guaranteeing atomicity of the load-modify-save operations.
static REGISTRY_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

/// Computes the registry file paths. Returns `(temp_dir, registry_file_path)`.
fn registry_paths() -> std::io::Result<(PathBuf, PathBuf)> {
    let temp_dir = crate::ai::driver::runtime_ctx::temp_dir()?;
    let registry_path = temp_dir.join(REGISTRY_FILENAME);
    Ok((temp_dir, registry_path))
}

/// Loads the registry from disk. Returns an empty set when the file does not exist.
fn load_paths(registry_path: &std::path::Path) -> Result<FastSet<String>, String> {
    if !registry_path.exists() {
        return Ok(FastSet::default());
    }
    let content = std::fs::read_to_string(registry_path)
        .map_err(|e| format!("Failed to read temp registry: {e}"))?;
    let paths: Vec<String> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse temp registry: {e}"))?;
    Ok(paths.into_iter().collect())
}

/// Writes the registry back to disk.
fn save_paths(registry_path: &std::path::Path, paths: &FastSet<String>) -> Result<(), String> {
    if let Some(parent) = registry_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create temp registry dir: {e}"))?;
    }
    let mut sorted: Vec<&String> = paths.iter().collect();
    sorted.sort();
    let content = serde_json::to_string_pretty(&sorted)
        .map_err(|e| format!("Failed to serialize temp registry: {e}"))?;
    std::fs::write(registry_path, content)
        .map_err(|e| format!("Failed to write temp registry: {e}"))?;
    Ok(())
}

/// Registers a temp file path (an already-resolved absolute path should be passed in).
/// Registering the same path twice is idempotent.
pub(crate) fn register(abs_path: &str) -> Result<(), String> {
    let _guard = REGISTRY_LOCK
        .lock()
        .map_err(|e| format!("Failed to lock temp registry: {e}"))?;
    let (_, registry_path) =
        registry_paths().map_err(|e| format!("Failed to get temp dir: {e}"))?;
    let mut paths = load_paths(&registry_path)?;
    paths.insert(abs_path.to_string());
    save_paths(&registry_path, &paths)
}

/// Checks whether a path is in the registry.
pub(crate) fn is_registered(abs_path: &str) -> bool {
    let Ok(_guard) = REGISTRY_LOCK.lock() else {
        return false;
    };
    let Ok((_, registry_path)) = registry_paths() else {
        return false;
    };
    load_paths(&registry_path)
        .map(|p| p.contains(abs_path))
        .unwrap_or(false)
}

/// Removes a path from the registry (call after a successful delete).
/// Succeeds silently when the path is absent.
pub(crate) fn unregister(abs_path: &str) -> Result<(), String> {
    let _guard = REGISTRY_LOCK
        .lock()
        .map_err(|e| format!("Failed to lock temp registry: {e}"))?;
    let (_, registry_path) =
        registry_paths().map_err(|e| format!("Failed to get temp dir: {e}"))?;
    let mut paths = load_paths(&registry_path)?;
    paths.remove(abs_path);
    save_paths(&registry_path, &paths)
}

/// Lists all currently registered paths (for debugging / audit).
#[allow(dead_code)]
pub(crate) fn list_registered() -> Vec<String> {
    let Ok(_guard) = REGISTRY_LOCK.lock() else {
        return Vec::new();
    };
    let Ok((_, registry_path)) = registry_paths() else {
        return Vec::new();
    };
    load_paths(&registry_path)
        .map(|p| {
            let mut v: Vec<String> = p.into_iter().collect();
            v.sort();
            v
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;

    #[test]
    fn register_and_check() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::temp_dir().join(format!("temp_reg_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            // Not registered before registering
            assert!(!is_registered("/nonexistent"));
            // Registered after registering
            register("/tmp/foo.txt").unwrap();
            assert!(is_registered("/tmp/foo.txt"));
            // Duplicate registration is idempotent
            register("/tmp/foo.txt").unwrap();
            assert!(is_registered("/tmp/foo.txt"));
            // Unregistered after unregistering
            unregister("/tmp/foo.txt").unwrap();
            assert!(!is_registered("/tmp/foo.txt"));
        });

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn registry_persists_across_loads() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // temp_dir()'s fallback path is isolated by session_id (TURN_IDENTITY); a unique session
        // owns a registry file exclusively, avoiding count pollution from sharing
        // `~/.agent_tmp/default/temp_registry.json` with other concurrent tests / legacy entries.
        let session_id = format!("temp_reg_persist_{}", uuid::Uuid::new_v4());
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope((session_id, 0usize), || {
            register("/tmp/a.txt").unwrap();
            register("/tmp/b.txt").unwrap();
            // Still present after reload
            assert!(is_registered("/tmp/a.txt"));
            assert!(is_registered("/tmp/b.txt"));
            let all = list_registered();
            assert_eq!(all.len(), 2);
        });
    }
}
