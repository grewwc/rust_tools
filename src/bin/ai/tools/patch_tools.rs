use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_apply_patch() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to patch (some sensitive paths are blocked)."
            },
            "patch": {
                "type": "string",
                "description": "Unified diff patch text (expects @@ hunks with context/add/remove lines)."
            }
        },
        "required": ["file_path", "patch"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "Apply a unified-diff patch to a file (absolute path). Prefer this for updating an existing document or source file with the smallest localized change instead of rewriting the entire file. Creates missing parent directories; fails if context/removals do not match.",
        parameters: params_apply_patch,
        execute: execute_apply_patch,
        groups: &["openclaw", "builtin"],
    }
});

#[derive(Debug, Clone)]
struct UnifiedHunk {
    old_start: usize,
    lines: Vec<UnifiedLine>,
}

#[derive(Debug, Clone)]
enum UnifiedLine {
    Context(String),
    Remove(String),
    Add(String),
}

fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
    let mut hunks = Vec::new();
    let mut iter = patch.lines().peekable();
    while let Some(line) = iter.next() {
        let Some(rest) = line.strip_prefix("@@") else {
            continue;
        };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('-') else {
            return Err("invalid hunk header".to_string());
        };
        let mut parts = rest.split_whitespace();
        let old_part = parts.next().ok_or("invalid hunk header")?;
        let _new_part = parts.next().ok_or("invalid hunk header")?;

        let old_start = old_part
            .split(',')
            .next()
            .ok_or("invalid hunk header")?
            .parse::<isize>()
            .map_err(|_| "invalid hunk header")?;
        let old_start = if old_start <= 0 {
            0
        } else {
            old_start as usize
        };

        let mut lines = Vec::new();
        while let Some(next) = iter.peek().copied() {
            if next.starts_with("@@") {
                break;
            }
            let l = iter.next().unwrap_or_default();
            if l.starts_with("\\ No newline at end of file") {
                continue;
            }
            let mut chars = l.chars();
            let prefix = chars
                .next()
                .ok_or_else(|| "invalid hunk line: empty".to_string())?;
            let body = chars.as_str();
            match prefix {
                ' ' => lines.push(UnifiedLine::Context(body.to_string())),
                '-' => lines.push(UnifiedLine::Remove(body.to_string())),
                '+' => lines.push(UnifiedLine::Add(body.to_string())),
                _ => return Err(format!("invalid hunk line: {}", l)),
            }
        }
        hunks.push(UnifiedHunk { old_start, lines });
    }
    if hunks.is_empty() {
        return Err("no hunks found".to_string());
    }
    Ok(hunks)
}

fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks = parse_unified_hunks(patch)?;
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for hunk in hunks {
        let apply_at = hunk.old_start.saturating_sub(1);
        if apply_at > orig_lines.len() {
            return Err("hunk start out of range".to_string());
        }
        if apply_at < cursor {
            return Err("hunks out of order".to_string());
        }

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let mut idx = apply_at;

        for line in hunk.lines {
            match line {
                UnifiedLine::Context(s) => {
                    let cur = orig_lines.get(idx).ok_or("context out of range")?;
                    if cur != &s {
                        return Err("context mismatch".to_string());
                    }
                    out.push(s);
                    idx += 1;
                }
                UnifiedLine::Remove(s) => {
                    let cur = orig_lines.get(idx).ok_or("remove out of range")?;
                    if cur != &s {
                        return Err("remove mismatch".to_string());
                    }
                    idx += 1;
                }
                UnifiedLine::Add(s) => {
                    out.push(s);
                }
            }
        }

        cursor = idx;
    }

    out.extend_from_slice(&orig_lines[cursor..]);
    let mut s = out.join("\n");
    if had_trailing_newline {
        s.push('\n');
    }
    Ok(s)
}

fn is_sensitive_fs_path(path: &std::path::Path) -> bool {
    let s = path.to_string_lossy();
    let s = s.as_ref();
    if s.contains("/.ssh/")
        || s.ends_with("/.ssh")
        || s.contains("/.gnupg/")
        || s.ends_with("/.gnupg")
        || s.contains("/.aws/")
        || s.ends_with("/.aws")
        || s.contains("/.kube/")
        || s.ends_with("/.kube")
        || s.contains("/.configW")
        || s.ends_with("/.configW")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
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

pub(crate) fn execute_apply_patch(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let patch = args["patch"].as_str().ok_or("Missing patch")?;

    let path = PathBuf::from(file_path);
    if is_sensitive_fs_path(&path) {
        return Err("Access blocked: sensitive path".to_string());
    }
    let original = if path.exists() {
        fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?
    } else {
        String::new()
    };
    let next = apply_unified_patch(&original, patch)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }
    fs::write(&path, next).map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(format!("Successfully patched {}", file_path))
}

#[cfg(test)]
mod tests {
    use super::parse_unified_hunks;

    #[test]
    fn parse_unified_hunks_rejects_empty_hunk_line_instead_of_panicking() {
        let patch = "@@ -1,1 +1,1 @@\n\n-foo\n+bar\n";
        assert!(parse_unified_hunks(patch).is_err());
    }
}
