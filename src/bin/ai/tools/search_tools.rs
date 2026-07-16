use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
};
use crate::commonw::utils::expanduser;

fn params_list_directory() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (non-recursive). Absolute path recommended."
            }
        },
        "required": ["path"]
    })
}

fn params_find_path() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filename, path suffix, or glob pattern to match (e.g. \"Cargo.toml\", \"*.rs\", \"src/**\"). Exact filenames (no wildcards) use a fast BFS and return the first match."
            },
            "path": {
                "type": "string",
                "description": "Root directory to search in (default: \".\"). Returned paths are canonical absolute paths. Common build/cache/VCS dirs are skipped during recursion; pass such a dir explicitly as path to search inside it."
            }
        },
        "required": ["pattern"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_directory",
        description: "List direct children of a directory (non-recursive). Each line is a child name; directories are suffixed with '/'.",
        parameters: params_list_directory,
        execute: execute_list_directory,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "find_path",
        description: "Fast file-path finder under a root directory using filename, path-suffix, or glob match. This locates FILES by their path, not text inside files. For full-text matches, use an available content-search tool. Returns canonical absolute paths, one per line (empty output means no matches).",
        parameters: params_find_path,
        execute: execute_find_path,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

// find_path 是检索类结果：复现代价高，禁止有损压缩（只能零压缩
// 外溢）；但过时的旧检索结果允许被 LLM 裁剪释放上下文。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "find_path",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
    }
});

pub(crate) fn execute_list_directory(args: &Value) -> Result<String, String> {
    let path = args["path"].as_str().ok_or("Missing path")?;
    let dir_path = PathBuf::from(path);

    if !dir_path.exists() {
        return Err(format!("Directory not found: {}", path));
    }

    if !dir_path.is_dir() {
        return Err(format!("Not a directory: {}", path));
    }

    let entries: Vec<_> = fs::read_dir(&dir_path)
        .map_err(|e| format!("Failed to read directory: {}", e))?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir { format!("{}/", name) } else { name }
        })
        .collect();

    Ok(entries.join("\n"))
}

pub(crate) fn execute_find_path(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");

    let target = pattern.trim();
    if target.is_empty() {
        return Err("pattern is empty".to_string());
    }

    let root_pat = expanduser(path.trim()).to_string();
    let root_pat = if root_pat.trim().is_empty() {
        ".".to_string()
    } else {
        root_pat
    };
    let glob_mode = target.contains('*')
        || target.contains('?')
        || target.contains('[')
        || target.contains(']')
        || target.contains('{')
        || target.contains('}');

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let base_dir = {
        let p = PathBuf::from(&root_pat);
        if p.is_absolute() { p } else { cwd.join(p) }
    };

    // 快速路径：精确文件名（非 glob 模式）直接 BFS 命中即返回，
    // 避免每次都拉起 ff_embed 多线程 runtime 全量遍历
    if !glob_mode {
        if let Some(found) = find_first_file_by_name(&base_dir, target) {
            let abs = fs::canonicalize(&found).unwrap_or(found);
            return Ok(abs.to_string_lossy().trim().to_string());
        }
        return Ok(String::new());
    }

    // glob 模式：优先尝试 ff_embed 多线程搜索（大仓库更快），
    // 失败时回退到 terminalw 单线程 glob
    let result = run_glob_ff_embed(target, &root_pat, &base_dir);
    match result {
        Ok(output) if !output.trim().is_empty() => {
            return Ok(truncate_chars(output.trim(), 16_000));
        }
        Ok(_) => {
            // ff_embed 无结果，尝试 terminalw 回退
        }
        Err(_) => {
            // ff_embed 失败，尝试 terminalw 回退
        }
    }

    let fallback = run_glob_terminalw(target, path, &base_dir)?;
    Ok(truncate_chars(fallback.trim(), 16_000))
}

/// 使用 ff_embed 多线程搜索引擎进行 glob 匹配
fn run_glob_ff_embed(target: &str, root_pat: &str, base_dir: &Path) -> Result<String, String> {
    let wd = fs::canonicalize(base_dir).unwrap_or_else(|_| base_dir.to_path_buf());

    let opts = crate::ai::ff_embed::cli::Options {
        verbose: false,
        only_dir: false,
        print_md5: false,
        glob_mode: true,
        case_insensitive: false,
        relative: false,
        num_print: i64::MAX,
        thread_count: rust_tools::commonw::half_parallelism(),
        wd,
        root_pat: root_pat.to_string(),
        targets: vec![target.to_string()],
        excludes: Vec::new(),
    };

    crate::ai::ff_embed::output::begin_capture();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(rust_tools::commonw::half_parallelism())
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| format!("Failed to build async runtime: {e}"))?;
    let _ = rt.block_on(crate::ai::ff_embed::search::run_async(&opts));
    let results = crate::ai::ff_embed::output::finish_capture();

    let out: Vec<String> = results
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let p = PathBuf::from(s.trim());
            let abs = if p.is_absolute() { p } else { base_dir.join(p) };
            fs::canonicalize(&abs).unwrap_or(abs)
        })
        .filter(|abs| !rust_tools::commonw::path_contains_skip_dir(abs))
        .map(|abs| abs.to_string_lossy().to_string())
        .collect();
    Ok(out.join("\n"))
}

/// 使用 terminalw 单线程 glob 作为回退方案
fn run_glob_terminalw(target: &str, path: &str, base_dir: &Path) -> Result<String, String> {
    let matches =
        crate::terminalw::glob_paths(target, path).map_err(|e| format!("glob failed: {e}"))?;
    let out: Vec<String> = matches
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let p = PathBuf::from(s.trim());
            let abs = if p.is_absolute() { p } else { base_dir.join(p) };
            fs::canonicalize(&abs).unwrap_or(abs)
        })
        .filter(|abs| !rust_tools::commonw::path_contains_skip_dir(abs))
        .map(|abs| abs.to_string_lossy().to_string())
        .collect();
    Ok(out.join("\n"))
}

fn find_first_file_by_name(root: &Path, filename: &str) -> Option<PathBuf> {
    if filename.trim().is_empty() {
        return None;
    }

    if root.is_file() {
        let name = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
        return (name == filename).then_some(root.to_path_buf());
    }

    if !root.is_dir() {
        return None;
    }

    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    let mut scanned_dirs = 0usize;
    let max_dirs = 50_000usize;

    while let Some(dir) = queue.pop_front() {
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            return None;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            let file_name = file_name.as_ref();
            if file_name == filename {
                return Some(entry.path());
            }

            // 不能用 `?` —— 单个 entry 的 file_type() 失败（broken symlink / 权限不足）
            // 不应终止整个 BFS 搜索。
            let Some(ft) = entry.file_type().ok() else {
                continue;
            };
            if ft.is_dir() && !ft.is_symlink() && !rust_tools::commonw::is_skip_dir(file_name) {
                queue.push_back(entry.path());
            }
        }
    }

    None
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (truncated)");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_search_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn test_find_path_matches_by_filename() {
        let dir = make_temp_dir("findpath");
        fs::write(dir.join("target.txt"), "some text").unwrap();
        fs::write(dir.join("other.log"), "noise").unwrap();

        let args = serde_json::json!({
            "pattern": "target.txt",
            "path": dir.to_string_lossy()
        });
        let result = execute_find_path(&args);

        assert!(
            result.is_ok(),
            "find_path should not error, got: {:?}",
            result
        );
        let output = result.unwrap();
        assert!(
            output.contains("target.txt"),
            "should find target.txt by name, got: {}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_path_by_glob_pattern() {
        let dir = make_temp_dir("findpath_glob");
        fs::write(dir.join("foo.rs"), "").unwrap();
        fs::write(dir.join("bar.rs"), "").unwrap();
        fs::write(dir.join("baz.txt"), "").unwrap();

        let args = serde_json::json!({
            "pattern": "*.rs",
            "path": dir.to_string_lossy()
        });
        let result = execute_find_path(&args);
        assert!(result.is_ok(), "find_path glob failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("foo.rs"),
            "should find foo.rs, got: {}",
            output
        );
        assert!(
            output.contains("bar.rs"),
            "should find bar.rs, got: {}",
            output
        );
        assert!(
            !output.contains("baz.txt"),
            "should not find baz.txt, got: {}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_directory_returns_entries() {
        let dir = make_temp_dir("listdir");
        fs::write(dir.join("file1.txt"), "").unwrap();
        fs::write(dir.join("file2.txt"), "").unwrap();
        fs::create_dir(dir.join("subdir")).unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy()
        });
        let result = execute_list_directory(&args);
        assert!(result.is_ok(), "list_dir failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("file1.txt"),
            "should list file1.txt, got: {}",
            output
        );
        assert!(
            output.contains("file2.txt"),
            "should list file2.txt, got: {}",
            output
        );
        assert!(
            output.contains("subdir/"),
            "should list subdir with trailing /, got: {}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
