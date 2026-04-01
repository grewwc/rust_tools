use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
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

fn params_search_files() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Exact file name (preferred) or glob pattern to match. Examples: \"Cargo.toml\", \"*.rs\", \"**/*.md\""
            },
            "path": {
                "type": "string",
                "description": "Root directory to search in (default: \".\"). Returned paths are canonical absolute paths."
            }
        },
        "required": ["pattern"]
    })
}

fn params_grep_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filename or glob pattern to match (e.g. \"Cargo.toml\", \"*.rs\", \"src/**\")."
            },
            "path": {
                "type": "string",
                "description": "Root directory to search in (default: \".\")."
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob pattern override. If provided, it takes precedence over `pattern`."
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
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "search_files",
        description: "Find files under a directory by exact filename (fast) or glob pattern. Returns canonical absolute paths, one per line (empty output means no matches).",
        parameters: params_search_files,
        execute: execute_search_files,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "grep_search",
        description: "Fast file-path search under a root directory using filename match or glob. Returns paths relative to the current working directory (may include ANSI highlighting).",
        parameters: params_grep_search,
        execute: execute_grep_search,
        groups: &["builtin"],
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

pub(crate) fn execute_search_files(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");

    let cwd = std::env::current_dir().map_err(|e| format!("Failed to get cwd: {}", e))?;
    let base_dir = {
        let p = PathBuf::from(path);
        if p.is_absolute() { p } else { cwd.join(p) }
    };

    let is_exact_name = !pattern.contains('/')
        && !pattern.contains('\\')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('[')
        && !pattern.contains(']')
        && !pattern.contains('{')
        && !pattern.contains('}');

    if is_exact_name {
        if let Some(found) = find_first_file_by_name(Path::new(path), pattern) {
            let abs = if found.is_absolute() {
                found
            } else {
                base_dir.join(found)
            };
            let abs = fs::canonicalize(&abs).unwrap_or(abs);
            return Ok(abs.to_string_lossy().trim().to_string());
        }
        return Ok(String::new());
    }

    let matches =
        crate::terminalw::glob_paths(pattern, path).map_err(|e| format!("glob failed: {e}"))?;
    let out: Vec<String> = matches
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let p = PathBuf::from(s.trim());
            let abs = if p.is_absolute() { p } else { base_dir.join(p) };
            let abs = fs::canonicalize(&abs).unwrap_or(abs);
            abs.to_string_lossy().to_string()
        })
        .collect();
    Ok(out.join("\n").trim().to_string())
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

            let ft = entry.file_type().ok()?;
            if ft.is_dir() && !ft.is_symlink() {
                queue.push_back(entry.path());
            }
        }
    }

    None
}

pub(crate) fn execute_grep_search(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");
    let file_pattern = args["file_pattern"].as_str();

    let target = file_pattern.unwrap_or(pattern).trim();
    if target.is_empty() {
        return Err("pattern is empty".to_string());
    }

    let root_pat = expanduser(path.trim()).to_string();
    let root_pat = if root_pat.trim().is_empty() {
        ".".to_string()
    } else {
        root_pat
    };
    let glob_mode = file_pattern.is_some()
        || target.contains('*')
        || target.contains('?')
        || target.contains('[')
        || target.contains(']')
        || target.contains('{')
        || target.contains('}');

    let wd = std::env::current_dir()
        .ok()
        .and_then(|p| fs::canonicalize(&p).ok())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let opts = crate::ai::ff_embed::cli::Options {
        verbose: false,
        only_dir: false,
        print_md5: false,
        glob_mode,
        case_insensitive: false,
        relative: true,
        num_print: i64::MAX,
        thread_count: (num_cpus::get() / 2).max(1),
        wd,
        root_pat,
        targets: vec![target.to_string()],
        excludes: Vec::new(),
    };

    crate::ai::ff_embed::output::begin_capture();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads((num_cpus::get() / 2).max(1))
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| format!("Failed to build async runtime: {e}"))?;
    let _ = rt.block_on(crate::ai::ff_embed::search::run_async(&opts));
    let results = crate::ai::ff_embed::output::finish_capture();

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

    Ok(truncate_chars(results.join("\n").trim(), 16_000))
}
