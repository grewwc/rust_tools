use std::fs;
use std::path::{Path, PathBuf};

use crate::ai::errors::AiError;
use crate::ai::tools::storage::temp_registry;
use aios_kernel::primitives::VfsError;

pub(crate) struct FileStore {
    /// 调用方传入的原始路径（未经 resolve），用于错误提示中展示，避免模型
    /// 看到解析后的绝对路径时无法对应自己的输入。
    original: PathBuf,
    path: PathBuf,
}

impl FileStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        let original = path.clone();
        Self {
            original,
            path: resolve_effective_path(path),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn validate_read_access(&self) -> Result<(), AiError> {
        // 外溢归档是请求侧压缩后的唯一完整快照，必须可读；否则对模型而言等价于
        // 丢弃。read_file 归档中的历史行号由 service::file 在渲染前剥离，避免
        // 回读时产生嵌套行号。这里仍需把快照限制在当前会话，避免绝对路径把
        // 相邻会话的历史内容带入当前上下文。
        if let Some(reason) = blocked_overflow_read_reason(&self.path) {
            return Err(AiError::file(self.path.display().to_string(), reason));
        }
        Ok(())
    }

    pub(crate) fn validate_write_access(&self) -> Result<(), AiError> {
        self.validate_read_access()?;
        if path_within_allowed_roots(&self.path) {
            return Ok(());
        }
        // 同 session 的临时文件：write_file(temp=true) 已把解析后的绝对路径注册进
        // temp registry。即使该路径不在 effective_cwd / allowed_roots / 当前 temp_dir
        // 之下（例如子代理在隔离临时目录里创建的文件），write_file / apply_patch 也应
        // 允许继续操作，而不是被沙箱拦截。这是对「同 session 临时文件」的最权威判定。
        if temp_registry::is_registered(&self.path.display().to_string()) {
            return Ok(());
        }
        let resolved = self.path.display();
        // 明确告知模型「能写哪里」，而不仅是「这里不能写」。此前消息只提示
        // temp=true（scratch 语义），当模型的真实目标是把正式产物写到某个指定
        // 目录时无从下手，于是对同一越界路径反复重试。这里给出可写根目录 +
        // 两条具体可选路径（写进根目录内 / 用 temp=true），让模型一次纠偏。
        let roots = writable_roots_for_hint();
        let root_hint = match roots.first() {
            Some(root) => format!(
                "\nWritable root: '{}'. To fix, either (a) write to a path under \
                 that root, or (b) for scratch/intermediate files pass a relative \
                 filename with temp=true (e.g. file_path='script.py', temp=true). \
                 Do NOT retry the same absolute path — it will keep failing.",
                root.display()
            ),
            None => "\nHint: for temporary/scratch files, use temp=true with a \
                 relative filename to write under the session temp directory. \
                 Do NOT retry the same absolute path — it will keep failing."
                .to_string(),
        };
        Err(AiError::file(
            self.original.display().to_string(),
            format!(
                "Write blocked: path '{resolved}' is outside the allowed write \
                 directory (effective_cwd).{root_hint}"
            ),
        ))
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
        // 改动前快照（best-effort）：供 mutation log 的 before 字段使用。
        // 新文件 / 二进制 / 读取失败均为 None，不影响写盘。
        let before = self.read_to_string().ok();
        let result = self.write_all_inner(content);
        if result.is_ok() {
            // 记录到会话级 mutation log（best-effort，绝不影响写盘结果）。
            crate::ai::tools::storage::mutation_log::record(
                &self.path,
                "write",
                before.as_deref(),
                Some(content),
            );
        }
        result
    }

    fn write_all_inner(&self, content: &str) -> Result<(), AiError> {
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

/// 是否为会话压缩器保存的 `read_file` 结果快照。
///
/// 这类文件需要正常回读，但 service 层必须在重新渲染前剥离旧的行号前缀，防止
/// `read_file` 的展示格式在多次召回时不断嵌套。不能只依赖 `original_file_path`：
/// 原始文件可能已经变更、删除，或无法复现当时的截断结果。
pub(crate) fn is_read_file_overflow_artifact(path: &Path) -> bool {
    session_overflow_dir_component(path) == Some("tool-overflow-compressed")
        && overflow_artifact_tool_name(path).as_deref() == Some("read_file")
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

/// 当前 turn 对应的会话 asset 根目录。无活动 driver context 时保守返回 None，
/// 不能把测试/one-shot 环境误当成拥有任意历史会话的读取权限。
pub(crate) fn current_session_assets_dir() -> Option<PathBuf> {
    let ctx = crate::ai::driver::runtime_ctx::try_current()?;
    let turn_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let session_id = if turn_session_id.is_empty() {
        ctx.app_proto.session_id.as_str()
    } else {
        turn_session_id.as_str()
    };
    if session_id.is_empty() {
        return None;
    }
    Some(
        crate::ai::history::SessionStore::new(ctx.app_proto.config.history_file.as_path())
            .session_assets_dir(session_id),
    )
}

fn blocked_overflow_read_reason(path: &Path) -> Option<String> {
    blocked_overflow_read_reason_for_assets(path, current_session_assets_dir().as_deref())
}

fn blocked_overflow_read_reason_for_assets(
    path: &Path,
    current_session_assets: Option<&Path>,
) -> Option<String> {
    if !is_read_file_overflow_artifact(path) {
        return None;
    }

    let is_current_session_artifact = current_session_assets.is_some_and(|assets_dir| {
        let expected_dir = normalize_lexical(assets_dir).join("tool-overflow-compressed");
        normalize_lexical(path).parent() == Some(expected_dir.as_path())
    });
    if is_current_session_artifact {
        return None;
    }

    Some(
        "Access blocked: this read_file overflow artifact does not belong to the current session. \
         Use the stub's `original_file_path` (+ `original_range`) when available; only the current \
         session's exact `file_path` snapshot may be read when the original source changed or was removed."
            .to_string(),
    )
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

/// 计算当前生效的可写根目录集合：`ai.sandbox.allowed_roots` 非空时用其配置，
/// 为空（默认）时退回 `effective_cwd()`；session 临时目录始终追加为可写根。
/// 返回值同时用于写权限校验与「写被拒」错误提示中的可写路径建议，避免两处漂移。
fn configured_write_roots(base: &Path) -> Vec<PathBuf> {
    let raw = crate::commonw::configw::get_all_config().get(
        crate::ai::config_schema::AiConfig::SANDBOX_ALLOWED_ROOTS,
        "",
    );
    let mut roots: Vec<PathBuf> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| normalize_lexical(Path::new(s)))
        .collect();
    if roots.is_empty() {
        roots.push(normalize_lexical(base));
    }
    // session 临时目录始终可写：模型可能从之前工具输出中发现 temp 路径后
    // 用绝对路径写入（不带 temp=true），不应被沙箱拦截。
    if let Ok(temp) = crate::ai::driver::runtime_ctx::temp_dir() {
        let temp = normalize_lexical(&temp);
        if !roots.iter().any(|r| temp.starts_with(r)) {
            roots.push(temp);
        }
    }
    roots
}

/// 供「写被拒」错误提示使用的可写根目录：优先展示 effective_cwd 类的项目根，
/// 排在最前，便于模型据此纠偏（session 临时目录不适合承载正式产物，不作首选）。
fn writable_roots_for_hint() -> Vec<PathBuf> {
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    configured_write_roots(&base)
}

/// 当 `ai.sandbox.allowed_roots` 非空时，文件路径必须位于其中某个根之下。
/// 为空（默认）时，退回到 `effective_cwd()` 作为单一沙箱根目录。
fn path_within_allowed_roots(path: &Path) -> bool {
    // 相对路径基于 effective_cwd 解析为绝对路径后再归一化。
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    let roots = configured_write_roots(&base);
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
        FileStore, blocked_overflow_read_reason_for_assets, is_read_file_overflow_artifact,
        is_sensitive_fs_path, is_session_overflow_asset_path, normalize_lexical,
        overflow_artifact_tool_name, path_within_allowed_roots, path_within_roots, temp_registry,
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
    fn read_file_overflow_artifact_access_is_session_scoped() {
        let current_assets = Path::new("/sessions/current.assets");
        let current_artifact = Path::new(
            "/sessions/current.assets/tool-overflow-compressed/20260722T101112Z-read_file-abc123.txt",
        );
        let other_artifact = Path::new(
            "/sessions/other.assets/tool-overflow-compressed/20260722T101112Z-read_file-def456.txt",
        );

        assert!(is_read_file_overflow_artifact(current_artifact));
        assert!(
            blocked_overflow_read_reason_for_assets(current_artifact, Some(current_assets))
                .is_none()
        );
        assert!(
            blocked_overflow_read_reason_for_assets(other_artifact, Some(current_assets)).is_some()
        );
        assert!(blocked_overflow_read_reason_for_assets(current_artifact, None).is_some());
        assert!(!is_read_file_overflow_artifact(Path::new(
            "/proj/tool-overflow-compressed/20260722T101112Z-read_file-abc123.txt"
        )));

        // 无活动 driver context 时，FileStore 端到端必须拒绝历史快照。
        assert!(
            FileStore::new(current_artifact.to_path_buf())
                .validate_read_access()
                .is_err()
        );
    }

    #[test]
    fn read_file_overflow_artifact_cannot_escape_current_assets() {
        let current_assets = Path::new("/sessions/current.assets");
        let escaped = Path::new(
            "/sessions/current.assets/../other.assets/tool-overflow-compressed/20260722T101112Z-read_file-def456.txt",
        );
        assert!(blocked_overflow_read_reason_for_assets(escaped, Some(current_assets)).is_some());

        // 其它工具归档保持既有可读行为，不劣化其精确证据召回能力。
        let command_artifact = Path::new(
            "/sessions/other.assets/tool-overflow-compressed/20260722T101112Z-execute_command-def456.txt",
        );
        assert!(
            blocked_overflow_read_reason_for_assets(command_artifact, Some(current_assets))
                .is_none()
        );
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
    fn blocked_write_error_names_writable_root_and_warns_against_retry() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-hint-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).unwrap();
        let outside = temp_root
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .join(format!("outside-{}.json", uuid::Uuid::new_v4()));

        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", temp_root.join("empty.configw")) };
        crate::commonw::configw::refresh();

        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(temp_root.clone(), || {
                FileStore::new(outside.clone()).validate_write_access()
            })
            .expect_err("write outside effective_cwd must be blocked");
        let msg = err.to_string();

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        let _ = std::fs::remove_file(temp_root.join("empty.configw"));
        let _ = std::fs::remove_dir_all(&temp_root);

        // 保留稳定的分类前缀（供 orchestrator 的 write_blocked_outside_root_path 解析）。
        assert!(msg.contains("Write blocked: path '"), "msg: {msg}");
        // 必须明确告知可写根目录，而不仅是「这里不能写」。
        assert!(msg.contains("Writable root:"), "msg: {msg}");
        assert!(
            msg.contains(&temp_root.display().to_string()),
            "hint must name the effective_cwd root: {msg}"
        );
        // 必须劝阻对同一绝对路径的重试。
        assert!(
            msg.contains("Do NOT retry the same absolute path"),
            "msg: {msg}"
        );
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

    #[test]
    fn session_temp_dir_is_always_writable() {
        // session 临时目录（由 runtime_ctx::temp_dir() 返回）即使不在
        // effective_cwd 之下也应被允许写入：模型可能从先前工具输出中
        // 发现 temp 路径后用绝对路径写入（不带 temp=true）。
        let temp = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
        let target = temp.join(format!("sandbox-test-{}.txt", uuid::Uuid::new_v4()));
        assert!(
            path_within_allowed_roots(&target),
            "session temp dir path must be within allowed write roots: {}",
            target.display()
        );
    }

    #[test]
    fn temp_registry_registered_path_outside_roots_is_writable() {
        // 同 session 的临时文件（如子代理在隔离临时目录里创建、已注册进 temp
        // registry 的路径）即使不在 effective_cwd / allowed_roots 之下，也应被
        // write_file / apply_patch 允许继续操作，而不是被沙箱拦截。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-registry-{}", uuid::Uuid::new_v4()));
        // 一个在 effective_cwd 之外、也被 allowed_roots 排除的路径。
        let outside = temp_root
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .join(format!("outside-registered-{}.txt", uuid::Uuid::new_v4()));

        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", temp_root.join("empty.configw")) };
        crate::commonw::configw::refresh();

        // 未注册时，越界路径必须被拦截。
        let blocked = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(temp_root.clone(), || {
                FileStore::new(outside.clone()).validate_write_access()
            })
            .expect_err("unregistered outside path must be blocked");
        assert!(blocked.to_string().contains("Write blocked"));

        // 注册后（模拟 write_file(temp=true) 已注册该解析绝对路径），应放行。
        let abs = outside.display().to_string();
        temp_registry::register(&abs).unwrap();
        let allowed = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(temp_root.clone(), || {
                FileStore::new(outside.clone()).validate_write_access()
            });
        assert!(allowed.is_ok(), "registered temp path must be writable");
        let _ = temp_registry::unregister(&abs);

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        let _ = std::fs::remove_file(temp_root.join("empty.configw"));
        let _ = std::fs::remove_dir_all(&temp_root);
    }
}
