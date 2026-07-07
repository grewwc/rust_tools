// =============================================================================
// Persistent temp-file registry
// =============================================================================
// agent 通过 `write_file(temp=true)` 创建的临时文件会在此注册表留下记录。
// `delete_path` 只允许删除注册表中存在的路径——未经 agent 创建的文件一律
// 拒绝删除，从根上杜绝误删源码 / 配置 / 用户数据。
//
// 注册表以 JSON 文件持久化在 `<temp_dir>/temp_registry.json`
// （`temp_dir` 优先为 `~/.history_file.sessions/<session>.assets/tmp/`，
// 与 tool-overflow 同源，落在项目外；fallback 为系统临时目录），会话终止后重启仍可读取。
// =============================================================================

use std::path::PathBuf;

use rust_tools::commonw::FastSet;

/// 注册表文件名（相对于 temp_dir）。
const REGISTRY_FILENAME: &str = "temp_registry.json";

/// 进程级互斥锁：保证 load-modify-save 操作的原子性。
static REGISTRY_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

/// 计算注册表文件路径。返回 `(temp_dir, registry_file_path)`。
fn registry_paths() -> std::io::Result<(PathBuf, PathBuf)> {
    let temp_dir = crate::ai::driver::runtime_ctx::temp_dir()?;
    let registry_path = temp_dir.join(REGISTRY_FILENAME);
    Ok((temp_dir, registry_path))
}

/// 从磁盘加载注册表。文件不存在时返回空集合。
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

/// 将注册表写回磁盘。
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

/// 注册一个临时文件路径（应传入解析后的绝对路径）。
/// 重复注册同一路径是幂等的。
pub(crate) fn register(abs_path: &str) -> Result<(), String> {
    let _guard = REGISTRY_LOCK
        .lock()
        .map_err(|e| format!("Failed to lock temp registry: {e}"))?;
    let (_, registry_path) = registry_paths().map_err(|e| format!("Failed to get temp dir: {e}"))?;
    let mut paths = load_paths(&registry_path)?;
    paths.insert(abs_path.to_string());
    save_paths(&registry_path, &paths)
}

/// 检查路径是否在注册表中。
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

/// 从注册表中移除一个路径（删除成功后调用）。
/// 路径不存在时静默成功。
pub(crate) fn unregister(abs_path: &str) -> Result<(), String> {
    let _guard = REGISTRY_LOCK
        .lock()
        .map_err(|e| format!("Failed to lock temp registry: {e}"))?;
    let (_, registry_path) = registry_paths().map_err(|e| format!("Failed to get temp dir: {e}"))?;
    let mut paths = load_paths(&registry_path)?;
    paths.remove(abs_path);
    save_paths(&registry_path, &paths)
}

/// 列出当前注册的所有路径（供调试 / 审计）。
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
        let base = std::env::temp_dir().join(format!(
            "temp_reg_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            // 注册前不存在
            assert!(!is_registered("/nonexistent"));
            // 注册后存在
            register("/tmp/foo.txt").unwrap();
            assert!(is_registered("/tmp/foo.txt"));
            // 重复注册幂等
            register("/tmp/foo.txt").unwrap();
            assert!(is_registered("/tmp/foo.txt"));
            // 注销后不存在
            unregister("/tmp/foo.txt").unwrap();
            assert!(!is_registered("/tmp/foo.txt"));
        });

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn registry_persists_across_loads() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::temp_dir().join(format!(
            "temp_reg_persist_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            register("/tmp/a.txt").unwrap();
            register("/tmp/b.txt").unwrap();
            // 重新加载后仍在
            assert!(is_registered("/tmp/a.txt"));
            assert!(is_registered("/tmp/b.txt"));
            let all = list_registered();
            assert_eq!(all.len(), 2);
        });

        let _ = std::fs::remove_dir_all(&base);
    }
}
