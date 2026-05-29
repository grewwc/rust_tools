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
        if !path_within_allowed_roots(&self.path) {
            return Err(AiError::file(
                self.path.display().to_string(),
                "Access blocked: path is outside the sandbox allowed roots (ai.sandbox.allowed_roots)",
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
    // 用户在 `ai.sandbox.extra_sensitive_paths` 中追加的敏感子串。
    for needle in config_extra_sensitive_substrings() {
        if rendered.contains(&needle) {
            return true;
        }
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

/// 词法归一化路径：解析 `.`/`..` 而不触盘（路径可能尚不存在，如写入新文件）。
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// 读取 `ai.sandbox.extra_sensitive_paths`（逗号分隔，去空白）。
fn config_extra_sensitive_substrings() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config()
        .get(crate::ai::config_schema::AiConfig::SANDBOX_EXTRA_SENSITIVE_PATHS, "");
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 当 `ai.sandbox.allowed_roots` 非空时，文件路径必须位于其中某个根之下。
/// 为空（默认）时不施加额外限制，保持既有行为。
fn path_within_allowed_roots(path: &Path) -> bool {
    let raw = crate::commonw::configw::get_all_config()
        .get(crate::ai::config_schema::AiConfig::SANDBOX_ALLOWED_ROOTS, "");
    let roots: Vec<PathBuf> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| normalize_lexical(Path::new(s)))
        .collect();
    if roots.is_empty() {
        return true;
    }
    // 相对路径基于 effective_cwd 解析为绝对路径后再归一化。
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    path_within_roots(path, &base, &roots)
}

/// 纯函数：归一化 `path`（相对则基于 `base`）后判断是否落在任一 `roots` 之下。
fn path_within_roots(path: &Path, base: &Path, roots: &[PathBuf]) -> bool {
    if roots.is_empty() {
        return true;
    }
    let resolved = if path.is_absolute() {
        normalize_lexical(path)
    } else {
        normalize_lexical(&base.join(path))
    };
    roots.iter().any(|root| resolved.starts_with(root))
}

#[cfg(test)]
mod tests {
    use super::{is_sensitive_fs_path, normalize_lexical, path_within_roots};
    use std::path::{Path, PathBuf};

    #[test]
    fn normalize_lexical_resolves_dot_and_dotdot() {
        assert_eq!(
            normalize_lexical(Path::new("/home/user/proj/../proj/./src")),
            PathBuf::from("/home/user/proj/src")
        );
    }

    #[test]
    fn path_within_roots_empty_allows_everything() {
        assert!(path_within_roots(
            Path::new("/anywhere/file.txt"),
            Path::new("/base"),
            &[]
        ));
    }

    #[test]
    fn path_within_roots_accepts_inside_and_rejects_outside() {
        let roots = vec![PathBuf::from("/home/user/proj")];
        assert!(path_within_roots(
            Path::new("/home/user/proj/src/main.rs"),
            Path::new("/home/user/proj"),
            &roots
        ));
        // 越界绝对路径
        assert!(!path_within_roots(
            Path::new("/etc/passwd"),
            Path::new("/home/user/proj"),
            &roots
        ));
        // 通过 `..` 逃逸应被归一化后拦截
        assert!(!path_within_roots(
            Path::new("/home/user/proj/../secret"),
            Path::new("/home/user/proj"),
            &roots
        ));
    }

    #[test]
    fn path_within_roots_resolves_relative_against_base() {
        let roots = vec![PathBuf::from("/home/user/proj")];
        assert!(path_within_roots(
            Path::new("src/lib.rs"),
            Path::new("/home/user/proj"),
            &roots
        ));
        assert!(!path_within_roots(
            Path::new("../other/x"),
            Path::new("/home/user/proj"),
            &roots
        ));
    }

    #[test]
    fn sensitive_path_blocks_known_secrets() {
        assert!(is_sensitive_fs_path(Path::new("/home/u/.ssh/id_rsa")));
        assert!(is_sensitive_fs_path(Path::new("/home/u/.aws/credentials")));
        assert!(!is_sensitive_fs_path(Path::new("/home/u/proj/src/main.rs")));
    }
}


