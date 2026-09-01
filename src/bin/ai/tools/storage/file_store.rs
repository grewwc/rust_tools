use std::fs;
use std::path::{Path, PathBuf};

use crate::ai::errors::AiError;
use crate::ai::tools::storage::temp_registry;
use aios_kernel::primitives::VfsError;

pub(crate) struct FileStore {
    /// Original path as passed by the caller (unresolved), shown in error messages so the model
    /// can map a resolved absolute path back to its own input.
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
        // The overflow archive is the only complete post-compression snapshot on the request side and must stay readable; otherwise, to the model, it is
        // equivalent to dropping it. Historical line numbers in read_file archives are stripped by service::file before rendering to avoid
        // nested line numbers on re-read. The snapshot must still be scoped to the current session so absolute paths cannot pull
        // neighboring sessions' history into the current context.
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
        // Same-session temp files: write_file(temp=true) has registered the resolved absolute path in the
        // temp registry. Even if the path is outside effective_cwd / allowed_roots / the current temp_dir
        // (e.g. a file created by a subagent in an isolated temp dir), write_file / apply_patch must still
        // be allowed to proceed rather than being blocked by the sandbox. This is the most authoritative check for a "same-session temp file".
        if temp_registry::is_registered(&self.path.display().to_string()) {
            return Ok(());
        }
        let resolved = self.path.display();
        // Tell the model explicitly "where it can write", not just "it cannot write here". The previous message only mentioned
        // temp=true (scratch semantics); when the model's real goal is to write a final artifact to some specified
        // directory it had no way to proceed and kept retrying the same out-of-bounds path. Provide the writable root +
        // two concrete options (write inside the root / use temp=true) so the model can correct course in one step.
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
        // Route through AIOS VfsOps first (with trace + rusage_charge); fall back to bare std::fs when the kernel is not bound.
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
        // Pre-change snapshot (best-effort): used for the before field of the mutation log.
        // New files / binaries / read failures all yield None and never affect the write itself.
        let before = self.read_to_string().ok();
        let result = self.write_all_inner(content);
        if result.is_ok() {
            // Record into the session-level mutation log (best-effort; never affects the write result).
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

/// Try the kernel VfsOps path; returning None means the kernel is not ready yet (e.g. during unit test startup) and the caller should fall back to bare std::fs.
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
    // Sensitive substrings appended by the user via `ai.sandbox.extra_sensitive_paths`.
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

/// If `path` lies under one of the internal artifact directories generated by the session compressor, return the matched directory name.
///
/// Files in these directories are intermediate products of the context compression mechanism: offloaded tool results, folded archives, etc.
/// Only same-named directories anchored under `*.assets/` or `.history_file.sessions/<id>/` count,
/// to avoid collateral damage to ordinary same-named directories in user projects.
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

/// Whether this is a `read_file` result snapshot saved by the session compressor.
///
/// Such files must be readable normally, but the service layer must preserve the original display form from the archive so that
/// `read_file` does not add asset-relative line numbers again. Relying on `original_file_path` alone is unsafe: the original
/// file may have changed, been deleted, or can no longer reproduce the original truncation result.
pub(crate) fn is_read_file_overflow_artifact(path: &Path) -> bool {
    session_overflow_dir_component(path) == Some("tool-overflow-compressed")
        && overflow_artifact_tool_name(path).as_deref() == Some("read_file")
}

/// Extract the tool name from an overflow artifact filename `{timestamp}-{tool}-{uuid}.txt`.
///
/// The writer side always uses the `%Y%m%dT%H%M%SZ` timestamp (no `-`) and uuid simple format (no `-`),
/// so the tool name is between the first and last `-`. Return None on parse failure (conservatively allow, never mis-block).
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

/// Session asset root for the current turn. Conservatively return None when there is no active driver context,
/// never mistaking test/one-shot environments as having read access to arbitrary historical sessions.
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

/// Lexically normalize the path: resolve `.`/`..` without touching the disk (the path may not exist yet, e.g. when writing a new file).
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
    // Expand a leading `~` first: models often follow shell habit and pass `~/.config/...`, but `~` is only
    // expanded by shells; in Rust it is not absolute and would be wrongly joined after effective_cwd.
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

/// Read `ai.sandbox.extra_sensitive_paths` (comma-separated, whitespace trimmed).
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

/// Compute the effective writable root set: use `ai.sandbox.allowed_roots` when non-empty;
/// fall back to `effective_cwd()` when empty (default). The session temp dir, user skills dir
/// (`ai.skills.dir`, default `~/.config/rust_tools/skills`), and the rust_tools user config dir
/// (default `~/.config/rust_tools`) are always appended as writable roots.
/// The result feeds both the write-permission check and the writable-path suggestions in "write denied" errors, keeping them from drifting apart.
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
    // The session temp dir is always writable: the model may discover a temp path in earlier tool output and then
    // write to it by absolute path (without temp=true); that must not be blocked by the sandbox.
    if let Ok(temp) = crate::ai::driver::runtime_ctx::temp_dir() {
        let temp = normalize_lexical(&temp);
        if !roots.iter().any(|r| temp.starts_with(r)) {
            roots.push(temp);
        }
    }
    // The user skills dir (ai.skills.dir, default ~/.config/rust_tools/skills) is always writable:
    // the model needs to create / edit user .skill files, and this dir sits outside effective_cwd / allowed_roots,
    // so it must not be blocked by the apply_patch / write_file sandbox (save_skill already writes there directly).
    let skills = crate::ai::skills::skills_dir();
    let skills = if skills.is_absolute() {
        normalize_lexical(&skills)
    } else {
        normalize_lexical(&base.join(skills))
    };
    if !roots.iter().any(|r| skills.starts_with(r)) {
        roots.push(skills);
    }
    // The rust_tools user config dir (default ~/.config/rust_tools) is always writable:
    // the model needs write_file / apply_patch to create and maintain user-supplied
    // per-project instruction files (~/.config/rust_tools/<project>/agents.md) and other
    // user config, so the sandbox must not block it.
    if let Some(config_dir) = crate::commonw::utils::get_config_dir().map(|d| d.join("rust_tools"))
    {
        let config_dir = normalize_lexical(&config_dir);
        if !roots.iter().any(|r| config_dir.starts_with(r)) {
            roots.push(config_dir);
        }
    }
    roots
}

/// Writable roots for the "write denied" error message: prefer effective_cwd-style project roots,
/// listed first so the model can correct course (the session temp dir is unsuitable for final artifacts and is not suggested first).
fn writable_roots_for_hint() -> Vec<PathBuf> {
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    configured_write_roots(&base)
}

/// When `ai.sandbox.allowed_roots` is non-empty, file paths must lie under one of its roots.
/// When empty (default), fall back to `effective_cwd()` as the single sandbox root.
pub(crate) fn path_within_allowed_roots(path: &Path) -> bool {
    // Relative paths are resolved against effective_cwd into absolute paths before normalization.
    let base =
        crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from("."));
    let roots = configured_write_roots(&base);
    path_within_roots(path, &base, &roots)
}

/// Pure function: normalize `path` (relative paths resolve against `base`) and check whether it lies under any of `roots`.
pub(crate) fn path_within_roots(path: &Path, base: &Path, roots: &[PathBuf]) -> bool {
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
        // Out-of-bounds absolute path
        assert!(!path_within_roots(
            Path::new("/etc/passwd"),
            Path::new("/home/user/proj"),
            &roots
        ));
        // Escaping via `..` should be blocked after normalization
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
        // Writer-side format: {%Y%m%dT%H%M%SZ}-{tool}-{uuid_simple}.txt
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
        // Conservatively return None on unexpected naming (better to allow than to mis-block an ordinary file).
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

        // Without an active driver context, the FileStore end-to-end path must reject historical snapshots.
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

        // Other tools' archives keep their existing readable behavior; their precise evidence recall must not degrade.
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

        // Keep the stable category prefix (parsed by the orchestrator's write_blocked_outside_root_path).
        assert!(msg.contains("Write blocked: path '"), "msg: {msg}");
        // Must state the writable roots explicitly, not just "cannot write here".
        assert!(msg.contains("Writable root:"), "msg: {msg}");
        assert!(
            msg.contains(&temp_root.display().to_string()),
            "hint must name the effective_cwd root: {msg}"
        );
        // Must discourage retrying the same absolute path.
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
        // The session temp dir (returned by runtime_ctx::temp_dir()) must remain writable even when it is not
        // under effective_cwd: the model may discover the temp path in earlier tool output
        // and write there by absolute path (without temp=true).
        let temp = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
        let target = temp.join(format!("sandbox-test-{}.txt", uuid::Uuid::new_v4()));
        assert!(
            path_within_allowed_roots(&target),
            "session temp dir path must be within allowed write roots: {}",
            target.display()
        );
    }

    #[test]
    fn skills_dir_is_always_writable() {
        // The user skills dir (ai.skills.dir, default ~/.config/rust_tools/skills) must stay writable even when
        // outside effective_cwd / allowed_roots: the model needs apply_patch / write_file to create and edit
        // user .skill files, so the sandbox must not block it.
        let skills = crate::ai::skills::skills_dir();
        let target = skills.join(format!("sandbox-test-{}.skill", uuid::Uuid::new_v4()));
        assert!(
            path_within_allowed_roots(&target),
            "skills dir path must be within allowed write roots: {}",
            target.display()
        );
    }

    #[test]
    fn rust_tools_config_dir_is_always_writable() {
        // The rust_tools user config dir (default ~/.config/rust_tools) must stay writable even when
        // outside effective_cwd / allowed_roots: the model needs write_file / apply_patch to create
        // and maintain per-project instruction files (~/.config/rust_tools/<project>/agents.md).
        let config_dir = crate::commonw::utils::get_config_dir()
            .expect("HOME must be set in the test env")
            .join("rust_tools");
        let target = config_dir.join(format!("sandbox-test-{}.txt", uuid::Uuid::new_v4()));
        assert!(
            path_within_allowed_roots(&target),
            "rust_tools config dir path must be within allowed write roots: {}",
            target.display()
        );
    }

    #[test]
    fn temp_registry_registered_path_outside_roots_is_writable() {
        // Same-session temp files (e.g. created by a subagent in an isolated temp dir and already registered in the
        // temp registry) must remain writable via write_file / apply_patch even when outside
        // effective_cwd / allowed_roots, instead of being blocked by the sandbox.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp_root =
            std::env::temp_dir().join(format!("file-store-registry-{}", uuid::Uuid::new_v4()));
        // A path outside effective_cwd and also excluded by allowed_roots.
        let outside = temp_root
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .join(format!("outside-registered-{}.txt", uuid::Uuid::new_v4()));

        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", temp_root.join("empty.configw")) };
        crate::commonw::configw::refresh();

        // When unregistered, the out-of-bounds path must be blocked.
        let blocked = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
            .sync_scope(temp_root.clone(), || {
                FileStore::new(outside.clone()).validate_write_access()
            })
            .expect_err("unregistered outside path must be blocked");
        assert!(blocked.to_string().contains("Write blocked"));

        // Once registered (simulating write_file(temp=true) having registered the resolved absolute path), it should be allowed.
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
