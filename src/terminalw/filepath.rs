use regex::Regex;
use std::path::Path;

fn build_case_insensitive_glob(pattern: &str) -> String {
    let mut out = String::new();
    for ch in pattern.chars() {
        if ch.is_ascii_alphabetic() {
            let lo = ch.to_ascii_lowercase();
            let up = ch.to_ascii_uppercase();
            out.push('[');
            out.push(lo);
            out.push(up);
            out.push(']');
        } else {
            out.push(ch);
        }
    }
    out
}

/// 将 glob 模式转为正则表达式。
/// 支持 `*`（匹配非 `/` 字符）、`?`、`[...]` 字符类。
fn glob_to_regex(pattern: &str) -> Result<Regex, String> {
    let mut re = String::from("^");
    let mut it = pattern.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '*' => {
                if it.peek() == Some(&'*') {
                    it.next(); // 消费第二个 *
                    if it.peek() == Some(&'/') {
                        it.next();
                        re.push_str("(.+/)?");
                    } else {
                        re.push_str(".*");
                    }
                } else {
                    re.push_str("[^/]*");
                }
            }
            '?' => re.push_str("[^/]"),
            '[' => {
                re.push('[');
                while let Some(cc) = it.next() {
                    re.push(cc);
                    if cc == ']' { break; }
                }
            }
            '.' | '+' | '^' | '$' | '{' | '}' | '(' | ')' | '|' | '\\' => {
                re.push('\\');
                re.push(c);
            }
            _ => re.push(c),
        }
    }
    re.push('$');
    Regex::new(&re).map_err(|e| format!("invalid glob pattern: {e}"))
}

/// 递归遍历 `dir`，将每个文件的完整路径与 `re` 匹配。
fn walk_and_match(dir: &Path, re: &Regex, out: &mut Vec<String>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{dir:?}: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let s = path.to_string_lossy().to_string();
        // 同时匹配文件和目录（glob `*` 经常用来匹配目录名）
        if re.is_match(&s) {
            out.push(s);
        }
        if path.is_dir() {
            walk_and_match(&path, re, out)?;
        }
    }
    Ok(())
}

pub fn glob_paths(pattern: &str, root_path: &str) -> Result<Vec<String>, String> {
    let p = Path::new(root_path).join(pattern);
    let pat = p.to_string_lossy().to_string();
    let re = glob_to_regex(&pat)?;
    // 找到通配符之前的目录前缀作为遍历起点
    let root = match pat.char_indices().find(|(_, c)| matches!(c, '*' | '?' | '[')) {
        Some((i, _)) => match pat[..i].rfind('/') {
            Some(pos) => Path::new(&pat[..=pos]),
            None => Path::new("."),
        },
        None => Path::new(&pat),
    };
    let mut out = Vec::new();
    if root.is_file() {
        let s = root.to_string_lossy().to_string();
        if re.is_match(&s) {
            out.push(s);
        }
    } else {
        walk_and_match(root, &re, &mut out)?;
    }
    Ok(out)
}

pub fn glob_case_insensitive(pattern: &str, root_path: &str) -> Result<Vec<String>, String> {
    let ci = build_case_insensitive_glob(pattern);
    glob_paths(&ci, root_path)
}
