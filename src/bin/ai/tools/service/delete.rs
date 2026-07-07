use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::storage::file_store::FileStore;
use crate::ai::tools::storage::temp_registry;

/// 删除 agent 自己通过 `write_file(temp=true)` 创建的临时文件。
///
/// **安全模型**：只有注册表（`temp_registry`）中记录的路径才允许删除。
/// 未经 agent 创建的文件（源码、配置、用户数据等）一律拒绝，从根上杜绝误删。
/// 注册表以 JSON 文件持久化，会话终止后重启仍可读取。
pub(crate) fn execute_delete_path(args: &Value) -> Result<String, String> {
    let raw_path = args
        .get("path")
        .or_else(|| args.get("file_path"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing path".to_string())?;
    let recursive = args["recursive"].as_bool().unwrap_or(false);

    let store = FileStore::new(PathBuf::from(raw_path));
    let resolved = store.path().to_path_buf();
    let abs_path_str = resolved.display().to_string();

    // 核心安全检查：路径必须在注册表中。不在注册表 → 一律拒绝。
    // 对相对路径，同时尝试解析到 temp_dir 下（write_file(temp=true) 的写入位置）。
    let temp_dir_resolved = if !std::path::Path::new(raw_path).is_absolute() {
        crate::ai::driver::runtime_ctx::temp_dir()
            .ok()
            .map(|d| d.join(raw_path))
            .map(|p| p.display().to_string())
    } else {
        None
    };

    // 优先用 effective_cwd 解析的路径查注册表，若未命中再尝试 temp_dir 解析。
    let registered_path = if temp_registry::is_registered(&abs_path_str) {
        abs_path_str.clone()
    } else if let Some(ref tp) = temp_dir_resolved
        && temp_registry::is_registered(tp)
    {
        tp.clone()
    } else {
        return Err(format!(
            "Refused: '{abs_path_str}' is not a registered temp file. \
             delete_path only removes files created via write_file(temp=true). \
             Source code, configs, and other project files cannot be deleted."
        ));
    };

    // 使用注册表中记录的路径（可能是 temp_dir 下的解析路径）。
    let resolved = PathBuf::from(&registered_path);

    if !resolved.exists() {
        // 文件已不存在（可能已被清理），从注册表移除残留条目。
        temp_registry::unregister(&registered_path)?;
        return Ok(format!(
            "Path already removed, cleaned registry entry: {abs_path_str}"
        ));
    }

    let meta = std::fs::symlink_metadata(&resolved)
        .map_err(|e| format!("Failed to read path metadata: {e}"))?;

    if meta.is_dir() {
        if !recursive {
            return Err(format!(
                "Path is a directory; set recursive=true to delete directories: {abs_path_str}"
            ));
        }
        std::fs::remove_dir_all(&resolved)
            .map_err(|e| format!("Failed to remove directory: {e}"))?;
        temp_registry::unregister(&registered_path)?;
        return Ok(format!("Removed directory (recursive): {abs_path_str}"));
    }

    // 删除普通文件 / 符号链接：先做 undo 快照，再删除。
    super::super::undo_tools::snapshot_file_before_write(&registered_path);
    std::fs::remove_file(&resolved)
        .map_err(|e| format!("Failed to remove file: {e}"))?;
    super::super::undo_tools::commit_change_set(&format!("delete_path: {abs_path_str}"));
    temp_registry::unregister(&registered_path)?;

    Ok(format!("Removed file: {abs_path_str}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;
    use std::fs;
    use std::path::PathBuf;

    fn make_temp_base() -> PathBuf {
        let path = std::env::temp_dir()
            .join(format!("ai_delete_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn with_cwd<F: FnOnce()>(base: &PathBuf, f: F) {
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), f);
    }

    #[test]
    fn deletes_registered_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        with_cwd(&base, || {
            // 用 write_file(temp=true) 创建并注册
            let write_args = serde_json::json!({
                "file_path": "scratch.txt", "content": "x", "temp": true
            });
            crate::ai::tools::service::file::execute_write_file(&write_args).unwrap();

            // 用相对路径删除（delete_path 应解析到 temp_dir）
            let args = serde_json::json!({ "path": "scratch.txt" });
            let result = execute_delete_path(&args);
            assert!(result.is_ok(), "delete failed: {:?}", result);
            let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
            assert!(!temp_dir.join("scratch.txt").exists());
        });
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn refuses_unregistered_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        with_cwd(&base, || {
            // 在 temp_dir 下手动创建文件（不经过 write_file(temp=true)）
            let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
            let unregistered = temp_dir.join("unregistered.txt");
            fs::write(&unregistered, "x").unwrap();

            let args = serde_json::json!({ "path": unregistered.to_string_lossy() });
            let err = execute_delete_path(&args).unwrap_err();
            assert!(err.contains("not a registered temp file"), "got: {err}");
            assert!(unregistered.exists());
        });
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn refuses_source_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        let src = base.join("main.rs");
        fs::write(&src, "fn main() {}").unwrap();

        with_cwd(&base, || {
            let args = serde_json::json!({ "path": "main.rs" });
            let err = execute_delete_path(&args).unwrap_err();
            assert!(err.contains("not a registered temp file"), "got: {err}");
        });
        assert!(src.exists(), "source file should not be deleted");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn refuses_non_recursive_directory() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        with_cwd(&base, || {
            let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
            let dir = temp_dir.join("mydir");
            fs::create_dir_all(&dir).unwrap();
            temp_registry::register(&dir.display().to_string()).unwrap();

            let args = serde_json::json!({ "path": dir.to_string_lossy() });
            let err = execute_delete_path(&args).unwrap_err();
            assert!(err.contains("recursive"), "got: {err}");
        });
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn deletes_registered_directory_recursive() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        with_cwd(&base, || {
            let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
            let dir = temp_dir.join("mydir");
            fs::create_dir_all(dir.join("sub")).unwrap();
            fs::write(dir.join("sub/file.txt"), "x").unwrap();
            temp_registry::register(&dir.display().to_string()).unwrap();

            let args = serde_json::json!({ "path": dir.to_string_lossy(), "recursive": true });
            let result = execute_delete_path(&args);
            assert!(result.is_ok(), "delete failed: {:?}", result);
            assert!(!dir.exists());
        });
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn cleans_registry_for_already_removed_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = make_temp_base();
        with_cwd(&base, || {
            let write_args = serde_json::json!({
                "file_path": "gone.txt", "content": "x", "temp": true
            });
            crate::ai::tools::service::file::execute_write_file(&write_args).unwrap();

            // 手动删除文件，模拟外部清理
            let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
            fs::remove_file(temp_dir.join("gone.txt")).unwrap();

            let args = serde_json::json!({ "path": "gone.txt" });
            let result = execute_delete_path(&args);
            assert!(result.is_ok(), "got: {:?}", result);
            assert!(result.unwrap().contains("already removed"));
        });
        let _ = fs::remove_dir_all(&base);
    }
}
