use std::fs;
use std::path::{Path, PathBuf};

use crate::ai::errors::AiError;
use aios_kernel::primitives::VfsError;

pub(crate) struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            path: resolve_effective_path(path),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn validate_read_access(&self) -> Result<(), AiError> {
        // 封锁对压缩器内部 read_file / code_search 外溢产物的回读：这些文件是工具
        // 渲染结果的转储，模型读回只会拿到旧行号内容并触发无限重读循环。带原始锚点
        // 的 execute_command 日志、user/image 归档与 overflow-history.md 不在此列。
        if let Some(reason) = blocked_overflow_read_reason(&self.path) {
            return Err(AiError::file(self.path.display().to_string(), reason));
        }
        Ok(())
    }

    pub(crate) fn validate_write_access(&self) -> Result<(), AiError> {
        self.validate_read_access()?;
        if !path_within_allowed_roots(&self.path) {
            return Err(AiError::file(
                self.path.display().to_string(),
                "Access blocked: path is outside the sandbox write roots (defaults to effective_cwd when ai.sandbox.allowed_roots is unset)",
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

/// 若 `path` 落在会话压缩器生成的某个内部产物目录下，返回匹配到的目录名。
///
/// 这些目录里的文件是上下文压缩机制的中间产物：外溢的工具结果、折叠的归档等。
/// 只有锚定在 `*.assets/` 或 `.history_file.sessions/<id>/` 之下的同名目录才算数，
/// 避免误伤用户项目里恰好同名的普通目录。
fn session_overflow_dir_component(path: &Path) -> Option<&'static str> {
    let components = path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    components.iter().enumerate().find_map(|(idx, name)| {
        let dir = match *name {
            "tool-overflow-compressed" => "tool-overflow-compressed",
            "user-overflow-preserved" => "user-overflow-preserved",
            "image-overflow-preserved" => "image-overflow-preserved",
            _ => return None,
        };
        let anchored = components
            .get(idx.saturating_sub(1))
            .is_some_and(|parent| parent.ends_with(".assets"))
            || (idx >= 2 && components[idx - 2] == ".history_file.sessions");
        anchored.then_some(dir)
    })
}

/// 判断读取该路径是否应被拒绝，若拒绝则给出可执行的替代指引。
///
/// 仅拦截 `tool-overflow-compressed/` 下 read_file / code_search 的外溢产物：这些
/// 文件是"工具渲染结果"的内部转储，模型回读只会拿到带行号的旧输出，触发
/// "压缩→留 file_path→回读→再压缩→再留 file_path→再回读" 的无限重读循环；而它们
/// 都能通过 stub 里的 `original_file_path` / `original_query` 指回真正的原始来源。
///
/// 明确放行（这些文件对模型有独立价值，不能封锁）：
/// - `execute_command` 外溢日志——命令输出没有可替代的"原始源"，归档本身即证据；
/// - `user-overflow-preserved` / `image-overflow-preserved`——stub 主动引导模型按需回读；
/// - `overflow-history.md`——位于 assets 根目录，本就不在这些子目录内。
fn blocked_overflow_read_reason(path: &Path) -> Option<String> {
    if session_overflow_dir_component(path)? != "tool-overflow-compressed" {
        return None;
    }
    match overflow_artifact_tool_name(path)?.as_str() {
        tool @ ("read_file" | "code_search") => Some(format!(
            "Access blocked: this is an internal compression artifact (the archived render of a prior \
             `{tool}` result), not a source file. Re-reading it produces nested line numbers and \
             re-compression loops. Read the original target instead — for read_file use the stub's \
             `original_file_path` (+ `original_range`); for code_search re-run with the stub's \
             `original_query` / `original_path`."
        )),
        // execute_command / plan / 其它高精度工具的归档保持可读：它们没有更好的原始锚点。
        _ => None,
    }
}

/// 从外溢产物文件名 `{timestamp}-{tool}-{uuid}.txt` 中提取工具名。
///
/// 写入侧固定用 `%Y%m%dT%H%M%SZ`（无 `-`）时间戳与 uuid simple 格式（无 `-`），
/// 故首个 `-` 与末个 `-` 之间即工具名。解析失败返回 None（保守放行，不误封）。
fn overflow_artifact_tool_name(path: &Path) -> Option<String> {
    let stem = path.file_stem().and_then(|value| value.to_str())?;
    let first = stem.find('-')?;
    let last = stem.rfind('-')?;
    if last <= first + 1 {
        return None;
    }
    let tool = &stem[first + 1..last];
    (!tool.is_empty()).then(|| tool.to_string())
}

#[cfg(test)]
fn is_session_overflow_asset_path(path: &Path) -> bool {
    session_overflow_dir_component(path).is_some()
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

fn resolve_effective_path(path: PathBuf) -> PathBuf {
    // 先展开开头的 `~`：模型常按 shell 习惯给出 `~/.config/...`，但 `~` 只有
    // shell 才会展开，Rust 里它不是绝对路径，会被错误拼到 effective_cwd 之后。
    let path = match path.to_str() {
        Some(s) => PathBuf::from(crate::commonw::utils::expanduser(s).as_ref()),
        None => path,
    };
    if path.is_absolute() {
        return normalize_lexical(&path);
    }
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    normalize_lexical(&base.join(path))
}

/// 读取 `ai.sandbox.extra_sensitive_paths`（逗号分隔，去空白）。
fn config_extra_sensitive_substrings() -> Vec<String> {
    let raw = crate::commonw::configw::get_all_config().get(
        crate::ai::config_schema::AiConfig::SANDBOX_EXTRA_SENSITIVE_PATHS,
        "",
    );
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 当 `ai.sandbox.allowed_roots` 非空时，文件路径必须位于其中某个根之下。
/// 为空（默认）时，退回到 `effective_cwd()` 作为单一沙箱根目录。
fn path_within_allowed_roots(path: &Path) -> bool {
    let raw = crate::commonw::configw::get_all_config().get(
        crate::ai::config_schema::AiConfig::SANDBOX_ALLOWED_ROOTS,
        "",
    );
    // 相对路径基于 effective_cwd 解析为绝对路径后再归一化。
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    let mut roots: Vec<PathBuf> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| normalize_lexical(Path::new(s)))
        .collect();
    if roots.is_empty() {
        roots.push(normalize_lexical(&base));
    }
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
    use super::{
        FileStore, blocked_overflow_read_reason, is_sensitive_fs_path,
        is_session_overflow_asset_path, normalize_lexical, overflow_artifact_tool_name,
        path_within_allowed_roots, path_within_roots,
    };
    use crate::ai::test_support::ENV_LOCK;
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

    #[test]
    fn session_overflow_asset_path_detection_is_precise() {
        assert!(is_session_overflow_asset_path(Path::new(
            "/tmp/abc.assets/tool-overflow-compressed/result.txt"
        )));
        assert!(is_session_overflow_asset_path(Path::new(
            "/tmp/.history_file.sessions/abc/tool-overflow-compressed/result.txt"
        )));
        assert!(!is_session_overflow_asset_path(Path::new(
            "/tmp/project/tool-overflow-compressed/result.txt"
        )));
        assert!(!is_session_overflow_asset_path(Path::new(
            "/tmp/abc.assets/overflow-history.md"
        )));
    }

    #[test]
    fn overflow_artifact_tool_name_parses_write_side_filename() {
        // 写入侧格式：{%Y%m%dT%H%M%SZ}-{tool}-{uuid_simple}.txt
        assert_eq!(
            overflow_artifact_tool_name(Path::new(
                "/a.assets/tool-overflow-compressed/20260722T101112Z-read_file-abc123.txt"
            )),
            Some("read_file".to_string())
        );
        assert_eq!(
            overflow_artifact_tool_name(Path::new(
                "/a.assets/tool-overflow-compressed/20260722T101112Z-execute_command-def456.txt"
            )),
            Some("execute_command".to_string())
        );
        // 非预期命名保守返回 None（宁可放行也不误封普通文件）。
        assert_eq!(
            overflow_artifact_tool_name(Path::new("/a.assets/plain.txt")),
            None
        );
    }

    #[test]
    fn read_file_and_code_search_artifacts_are_blocked_with_redirect() {
        let read_artifact = Path::new(
            "/proj.assets/tool-overflow-compressed/20260722T101112Z-read_file-abc123.txt",
        );
        let reason = blocked_overflow_read_reason(read_artifact)
            .expect("read_file overflow artifact must be blocked");
        assert!(reason.contains("original_file_path"), "{reason}");

        let search_artifact = Path::new(
            "/proj.assets/tool-overflow-compressed/20260722T101112Z-code_search-abc123.txt",
        );
        let reason = blocked_overflow_read_reason(search_artifact)
            .expect("code_search overflow artifact must be blocked");
        assert!(reason.contains("original_query"), "{reason}");

        // 通过 FileStore 端到端确认拒绝（read_file 工具会走 validate_read_access）。
        assert!(FileStore::new(read_artifact.to_path_buf())
            .validate_read_access()
            .is_err());
    }

    #[test]
    fn execute_command_and_recall_archives_remain_readable() {
        // execute_command 日志没有可替代的原始来源，必须放行。
        assert!(blocked_overflow_read_reason(Path::new(
            "/proj.assets/tool-overflow-compressed/20260722T101112Z-execute_command-abc123.txt"
        ))
        .is_none());
        // user / image 归档 stub 主动引导模型按需回读，放行。
        assert!(blocked_overflow_read_reason(Path::new(
            "/proj.assets/user-overflow-preserved/20260722T101112Z-user-abc123.json"
        ))
        .is_none());
        assert!(blocked_overflow_read_reason(Path::new(
            "/proj.assets/image-overflow-preserved/20260722T101112Z-image-abc123.json"
        ))
        .is_none());
        // overflow-history.md 位于 assets 根目录，不在封锁子目录内。
        assert!(blocked_overflow_read_reason(Path::new("/proj.assets/overflow-history.md")).is_none());
        // 用户项目里恰好同名的目录不受影响（未锚定到 .assets / 会话目录）。
        assert!(blocked_overflow_read_reason(Path::new(
            "/proj/tool-overflow-compressed/20260722T101112Z-read_file-abc123.txt"
        ))
        .is_none());
    }

    #[test]
    fn default_write_root_falls_back_to_effective_cwd() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(temp_root.join("inside")).unwrap();
        let outside = temp_root
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .join("outside.txt");

        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", temp_root.join("empty.configw")) };
        crate::commonw::configw::refresh();

        let result =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp_root.clone(), || {
                (
                    path_within_allowed_roots(&temp_root.join("inside/file.txt")),
                    path_within_allowed_roots(&outside),
                )
            });

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        let _ = std::fs::remove_file(temp_root.join("empty.configw"));
        let _ = std::fs::remove_dir_all(&temp_root);

        assert!(result.0);
        assert!(!result.1);
    }

    #[test]
    fn read_access_is_not_limited_by_effective_cwd_when_not_sensitive() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-read-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).unwrap();
        let outside = temp_root
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .join(format!("outside-{}.txt", uuid::Uuid::new_v4()));

        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", temp_root.join("empty.configw")) };
        crate::commonw::configw::refresh();

        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(temp_root.clone(), || {
                FileStore::new(outside.clone()).validate_read_access()
            });

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        let _ = std::fs::remove_file(temp_root.join("empty.configw"));
        let _ = std::fs::remove_dir_all(&temp_root);

        assert!(
            result.is_ok(),
            "read access should ignore effective_cwd root"
        );
    }

    #[test]
    fn file_store_resolves_relative_paths_against_effective_cwd() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-relative-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(temp_root.join("nested")).unwrap();

        let resolved =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp_root.clone(), || {
                FileStore::new(PathBuf::from("nested/file.txt"))
                    .path()
                    .to_path_buf()
            });

        assert_eq!(resolved, temp_root.join("nested/file.txt"));
        let _ = std::fs::remove_dir_all(&temp_root);
    }
}
