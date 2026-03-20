use glob::glob;
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

pub fn glob_paths(pattern: &str, root_path: &str) -> Result<Vec<String>, String> {
    let p = Path::new(root_path).join(pattern);
    let mut out = Vec::new();
    let pat = p.to_string_lossy().to_string();
    for entry in glob(&pat).map_err(|e| e.to_string())? {
        if let Ok(path) = entry {
            out.push(path.to_string_lossy().to_string());
        }
    }
    Ok(out)
}

pub fn glob_case_insensitive(pattern: &str, root_path: &str) -> Result<Vec<String>, String> {
    let ci = build_case_insensitive_glob(pattern);
    glob_paths(&ci, root_path)
}
