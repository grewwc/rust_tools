use std::fs;
use std::path::{Path, PathBuf};

use crate::ai::errors::AiError;
use aios_kernel::primitives::VfsError;

pub(crate) struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn validate_access(&self) -> Result<(), AiError> {
        if is_sensitive_fs_path(&self.path) {
            return Err(AiError::file(
                self.path.display().to_string(),
                "Access blocked: sensitive path",
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_exists(&self) -> Result<(), AiError> {
        if !self.path.exists() {
            return Err(AiError::file(
                self.path.display().to_string(),
                "File not found",
            ));
        }
        Ok(())
    }

    pub(crate) fn read_to_string(&self) -> Result<String, AiError> {
        // 优先路由到 AIOS VfsOps（带 trace + rusage_charge）；当内核未绑定时退回裸 std::fs。
        if let Some(result) = try_vfs_read(&self.path) {
            return result.map_err(|e| vfs_to_ai_err(&self.path, e));
        }
        fs::read_to_string(&self.path).map_err(|e| {
            AiError::file(
                self.path.display().to_string(),
                format!("Failed to read file: {}", e),
            )
        })
    }

    pub(crate) fn write_all(&self, content: &str) -> Result<(), AiError> {
        if let Some(result) = try_vfs_write(&self.path, content) {
            return result.map_err(|e| vfs_to_ai_err(&self.path, e));
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                AiError::file(
                    self.path.display().to_string(),
                    format!("Failed to create directory: {}", e),
                )
            })?;
        }
        fs::write(&self.path, content).map_err(|e| {
            AiError::file(
                self.path.display().to_string(),
                format!("Failed to write file: {}", e),
            )
        })
    }
}

fn vfs_to_ai_err(path: &Path, err: VfsError) -> AiError {
    AiError::file(path.display().to_string(), err.to_string())
}

/// 尝试走内核 VfsOps；返回 None 表示内核未就绪（e.g. 单元测试启动阶段），让调用方 fallback 到裸 std::fs。
fn try_vfs_read(path: &Path) -> Option<Result<String, VfsError>> {
    use crate::ai::tools::os_tools::GLOBAL_OS;

    let guard = GLOBAL_OS.lock().ok()?;
    let os_arc = guard.as_ref()?.clone();
    drop(guard);
    let mut os = os_arc.lock().ok()?;
    let pid = os.current_process_id();
    Some(os.vfs_read_to_string(pid, path))
}

fn try_vfs_write(path: &Path, content: &str) -> Option<Result<(), VfsError>> {
    use crate::ai::tools::os_tools::GLOBAL_OS;

    let guard = GLOBAL_OS.lock().ok()?;
    let os_arc = guard.as_ref()?.clone();
    drop(guard);
    let mut os = os_arc.lock().ok()?;
    let pid = os.current_process_id();
    Some(os.vfs_write_all(pid, path, content))
}

fn is_sensitive_fs_path(path: &Path) -> bool {
    let rendered = path.to_string_lossy();
    let rendered = rendered.as_ref();
    if rendered.contains("/.ssh/")
        || rendered.ends_with("/.ssh")
        || rendered.contains("/.gnupg/")
        || rendered.ends_with("/.gnupg")
        || rendered.contains("/.aws/")
        || rendered.ends_with("/.aws")
        || rendered.contains("/.kube/")
        || rendered.ends_with("/.kube")
        || rendered.contains("/.configW")
        || rendered.ends_with("/.configW")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        name,
        "id_rsa"
            | "id_rsa.pub"
            | "id_ed25519"
            | "id_ed25519.pub"
            | "authorized_keys"
            | "known_hosts"
            | ".netrc"
            | ".npmrc"
            | ".pypirc"
            | ".git-credentials"
            | "credentials"
            | "config.json"
    )
}
