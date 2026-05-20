use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".cargo",
];

const MAX_ENTRIES: usize = 2000;
const MAX_OUTPUT_CHARS: usize = 32_000;

fn params_tree() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root directory to display (default: \".\")."
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth to recurse (default: 3, max: 6)."
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden files/directories (default: false)."
            },
            "dirs_only": {
                "type": "boolean",
                "description": "Only show directories, not files (default: false)."
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob filter for files (e.g. \"*.rs\"). Directories are always shown for structure."
            }
        }
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "tree",
        description: "Display directory tree structure with depth control. Shows files and directories in a hierarchical format, respecting .gitignore-style skip directories. Use this to quickly understand project layout without reading individual files.",
        parameters: params_tree,
        execute: execute_tree,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_tree(args: &Value) -> Result<String, String> {
    let path = args["path"].as_str().unwrap_or(".");
    let max_depth = args["max_depth"].as_u64().unwrap_or(3).min(6) as usize;
    let show_hidden = args["show_hidden"].as_bool().unwrap_or(false);
    let dirs_only = args["dirs_only"].as_bool().unwrap_or(false);
    let file_pattern = args["file_pattern"].as_str();

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let root = {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            cwd.join(p)
        }
    };

    if !root.exists() {
        return Err(format!("Path not found: {}", root.display()));
    }
    if !root.is_dir() {
        return Err(format!("Not a directory: {}", root.display()));
    }

    let glob_matcher = file_pattern.map(|pat| GlobFilter::new(pat));

    let mut output = String::new();
    let display_root = root
        .strip_prefix(&cwd)
        .unwrap_or(&root)
        .to_string_lossy()
        .to_string();
    let display_root = if display_root.is_empty() {
        ".".to_string()
    } else {
        display_root
    };
    output.push_str(&display_root);
    output.push('\n');

    let mut stats = TreeStats::default();
    build_tree(
        &root,
        "",
        0,
        max_depth,
        show_hidden,
        dirs_only,
        &glob_matcher,
        &mut output,
        &mut stats,
    );

    output.push('\n');
    if dirs_only {
        output.push_str(&format!("{} directories", stats.dir_count));
    } else {
        output.push_str(&format!(
            "{} directories, {} files",
            stats.dir_count, stats.file_count
        ));
    }
    if stats.truncated {
        output.push_str(" (truncated, too many entries)");
    }

    Ok(truncate_output(&output, MAX_OUTPUT_CHARS))
}

#[derive(Default)]
struct TreeStats {
    dir_count: usize,
    file_count: usize,
    total_entries: usize,
    truncated: bool,
}

fn build_tree(
    dir: &Path,
    prefix: &str,
    depth: usize,
    max_depth: usize,
    show_hidden: bool,
    dirs_only: bool,
    glob_matcher: &Option<GlobFilter>,
    output: &mut String,
    stats: &mut TreeStats,
) {
    if depth >= max_depth || stats.total_entries >= MAX_ENTRIES {
        if stats.total_entries >= MAX_ENTRIES {
            stats.truncated = true;
        }
        return;
    }

    let mut entries: Vec<DirEntry> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let is_dir = e.path().is_dir();

                if !show_hidden && name.starts_with('.') {
                    return None;
                }
                if is_dir && SKIP_DIRS.contains(&name.as_str()) {
                    return None;
                }
                if dirs_only && !is_dir {
                    return None;
                }
                if !is_dir {
                    if let Some(matcher) = glob_matcher {
                        if !matcher.matches(&name) {
                            return None;
                        }
                    }
                }

                Some(DirEntry {
                    name,
                    is_dir,
                    path: e.path(),
                })
            })
            .collect(),
        Err(_) => return,
    };

    entries.sort_by(|a, b| {
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });

    let count = entries.len();

    for (i, entry) in entries.iter().enumerate() {
        if stats.total_entries >= MAX_ENTRIES {
            stats.truncated = true;
            break;
        }
        stats.total_entries += 1;

        let is_last = i == count - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let child_prefix = if is_last { "    " } else { "│   " };

        if entry.is_dir {
            stats.dir_count += 1;
            output.push_str(&format!("{}{}{}/\n", prefix, connector, entry.name));
            build_tree(
                &entry.path,
                &format!("{}{}", prefix, child_prefix),
                depth + 1,
                max_depth,
                show_hidden,
                dirs_only,
                glob_matcher,
                output,
                stats,
            );
        } else {
            stats.file_count += 1;
            output.push_str(&format!("{}{}{}\n", prefix, connector, entry.name));
        }
    }
}

struct DirEntry {
    name: String,
    is_dir: bool,
    path: PathBuf,
}

struct GlobFilter {
    patterns: Vec<String>,
}

impl GlobFilter {
    fn new(pattern: &str) -> Self {
        Self {
            patterns: pattern
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
        }
    }

    fn matches(&self, file_name: &str) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        for pat in &self.patterns {
            if glob_match_simple(pat, file_name) {
                return true;
            }
        }
        false
    }
}

fn glob_match_simple(pattern: &str, name: &str) -> bool {
    let pat = pattern.trim_start_matches("**/");
    if pat.starts_with("*.") {
        let ext = &pat[1..];
        return name.ends_with(ext);
    }
    if pat.contains('*') || pat.contains('?') {
        let parts: Vec<&str> = pat.split('*').collect();
        if parts.len() == 2 {
            return name.starts_with(parts[0]) && name.ends_with(parts[1]);
        }
    }
    name == pat || name.ends_with(pat)
}

fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (output truncated)");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_tree_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn test_tree_basic_structure() {
        let dir = make_temp_dir("basic");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("tests")).unwrap();
        fs::write(dir.join("Cargo.toml"), "").unwrap();
        fs::write(dir.join("src/main.rs"), "").unwrap();
        fs::write(dir.join("src/lib.rs"), "").unwrap();
        fs::write(dir.join("tests/test1.rs"), "").unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_tree(&args);
        assert!(result.is_ok(), "tree failed: {:?}", result);
        let output = result.unwrap();
        assert!(output.contains("src/"), "should show src dir");
        assert!(output.contains("main.rs"), "should show main.rs");
        assert!(output.contains("Cargo.toml"), "should show Cargo.toml");
        assert!(output.contains("2 directories"), "should count 2 dirs");
        assert!(output.contains("4 files"), "should count 4 files");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tree_dirs_only() {
        let dir = make_temp_dir("dirsonly");
        fs::create_dir_all(dir.join("src/models")).unwrap();
        fs::create_dir_all(dir.join("tests")).unwrap();
        fs::write(dir.join("README.md"), "").unwrap();
        fs::write(dir.join("src/main.rs"), "").unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy().to_string(),
            "dirs_only": true
        });
        let result = execute_tree(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("src/"), "should show src");
        assert!(output.contains("models/"), "should show models");
        assert!(!output.contains("README.md"), "should not show files");
        assert!(!output.contains("main.rs"), "should not show files");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tree_skips_hidden_and_gitignore_dirs() {
        let dir = make_temp_dir("skip");
        fs::create_dir_all(dir.join(".git/objects")).unwrap();
        fs::create_dir_all(dir.join("node_modules/pkg")).unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/app.ts"), "").unwrap();
        fs::write(dir.join(".hidden"), "").unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_tree(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("src/"), "should show src");
        assert!(!output.contains(".git"), "should skip .git");
        assert!(!output.contains("node_modules"), "should skip node_modules");
        assert!(!output.contains(".hidden"), "should skip hidden files");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tree_max_depth() {
        let dir = make_temp_dir("depth");
        fs::create_dir_all(dir.join("a/b/c/d/e")).unwrap();
        fs::write(dir.join("a/b/c/d/e/deep.txt"), "").unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy().to_string(),
            "max_depth": 2
        });
        let result = execute_tree(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("a/"), "depth 0 dir");
        assert!(output.contains("b/"), "depth 1 dir");
        assert!(!output.contains("c/"), "depth 2 should not recurse further");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tree_file_pattern_filter() {
        let dir = make_temp_dir("filefilter");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.rs"), "").unwrap();
        fs::write(dir.join("src/style.css"), "").unwrap();
        fs::write(dir.join("src/readme.md"), "").unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let result = execute_tree(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("main.rs"), "should show .rs files");
        assert!(!output.contains("style.css"), "should skip .css");
        assert!(!output.contains("readme.md"), "should skip .md");
        assert!(output.contains("src/"), "dirs always shown");

        let _ = fs::remove_dir_all(&dir);
    }
}
