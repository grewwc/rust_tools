/// Unified knowledge entry — replaces AgentMemoryEntry.
/// Single data structure for all knowledge/memory storage.
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};

use super::types::Category;

/// 环境本地路径属于一次性实现细节，不应作为可跨 session 自动召回的长期知识。
/// 这里不再靠零散 substring 黑名单，而是：
/// 1. 从 note 中抽取疑似文件系统路径；
/// 2. 判断它是否是 home-relative / absolute / windows absolute 路径。
pub(crate) fn note_has_local_env_path_leak(note: &str) -> bool {
    note.split_whitespace()
        .filter(|token| !is_web_url_token(token))
        .flat_map(path_candidates_from_token)
        .any(is_local_env_path_candidate)
}

fn is_web_url_token(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    [
        "http://", "https://", "ftp://", "ftps://", "ws://", "wss://",
    ]
    .iter()
    .any(|scheme| lower.contains(scheme))
}

fn path_candidates_from_token(token: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(start) = find_path_start(token, cursor) {
        let end = find_path_end(token, start);
        if end <= start {
            break;
        }
        let candidate = &token[start..end];
        let prefix = &token[..start];
        cursor = end;
        if candidate.starts_with("//") && is_web_scheme_prefix(prefix) {
            continue;
        }
        out.push(candidate);
    }
    out
}

fn find_path_start(token: &str, from: usize) -> Option<usize> {
    let chars = token.char_indices().collect::<Vec<_>>();
    for (idx, ch) in chars {
        if idx < from {
            continue;
        }
        let next = token[idx..].chars().nth(1);
        let prev = token[..idx].chars().next_back();
        if matches!(ch, '~') && matches!(next, Some('/' | '\\')) {
            return Some(idx);
        }
        if is_windows_drive_prefix(token, idx) {
            return Some(idx);
        }
        if matches!(ch, '/' | '\\')
            && (idx == 0 || prev.is_some_and(path_boundary_before_absolute_path))
        {
            return Some(idx);
        }
    }
    None
}

fn find_path_end(token: &str, start: usize) -> usize {
    let tail = &token[start..];
    for (offset, ch) in tail.char_indices() {
        if is_path_terminator(ch) {
            return start + offset;
        }
    }
    token.len()
}

fn is_path_terminator(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\''
            | '`'
            | ','
            | '，'
            | '。'
            | ';'
            | '；'
            | '!'
            | '！'
            | '?'
            | '？'
            | ')'
            | ']'
            | '}'
            | '>'
            | '》'
            | '”'
            | '’'
    )
}

fn path_boundary_before_absolute_path(ch: char) -> bool {
    matches!(
        ch,
        '=' | '(' | '[' | '{' | '<' | ':' | '：' | '"' | '\'' | '`'
    )
}

fn is_web_scheme_prefix(prefix: &str) -> bool {
    let lower = prefix.to_ascii_lowercase();
    ["http:", "https:", "ftp:", "ftps:", "ws:", "wss:"]
        .iter()
        .any(|scheme| lower.ends_with(scheme))
}

fn is_local_env_path_candidate(candidate: &str) -> bool {
    is_home_relative_path(candidate)
        || is_windows_absolute_path(candidate)
        || is_absolute_path(candidate)
}

fn is_home_relative_path(candidate: &str) -> bool {
    candidate.starts_with("~/") || candidate.starts_with("~\\")
}

fn is_windows_absolute_path(candidate: &str) -> bool {
    if candidate.starts_with("\\\\") {
        return candidate.chars().skip(2).any(|ch| ch != '\\' && ch != '/');
    }
    let bytes = candidate.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
        && candidate[3..].chars().any(|ch| ch != '\\' && ch != '/')
}

fn is_absolute_path(candidate: &str) -> bool {
    let path = Path::new(candidate);
    path.is_absolute()
        && path
            .components()
            .any(|component| matches!(component, Component::Normal(_)))
}

fn is_windows_drive_prefix(token: &str, idx: usize) -> bool {
    let bytes = token.as_bytes();
    idx + 2 < bytes.len()
        && bytes[idx].is_ascii_alphabetic()
        && bytes[idx + 1] == b':'
        && matches!(bytes[idx + 2], b'\\' | b'/')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    #[serde(default)]
    pub id: Option<String>,
    pub timestamp: String,
    pub category: String,
    pub note: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    /// Priority level: 0-255. Higher = more important. 255 = permanent.
    #[serde(default = "default_priority")]
    pub priority: Option<u8>,

    /// Optional image path (for memo entries that include screenshots/images).
    /// When set, OCR text is extracted and stored in `note` for search indexing.
    #[serde(default)]
    pub image_path: Option<String>,
}

fn default_priority() -> Option<u8> {
    Some(100)
}

impl KnowledgeEntry {
    pub fn category_enum(&self) -> Category {
        Category::from_str(&self.category)
    }

    pub fn priority_value(&self) -> u8 {
        self.priority.unwrap_or(100)
    }

    /// Create a search text from all fields
    pub fn search_text(&self) -> String {
        let mut text = format!("{}: {}", self.category, self.note);
        if !self.tags.is_empty() {
            text.push_str(&format!(" [tags: {}]", self.tags.join(", ")));
        }
        if let Some(src) = &self.source {
            text.push_str(&format!(" (source: {})", src));
        }
        if let Some(img) = &self.image_path {
            text.push_str(&format!(" [image: {}]", img));
        }
        text
    }

    /// Whether this entry mentions a project hint
    pub fn mentions_project(&self, project: &str) -> bool {
        let hint = project.trim().to_lowercase();
        if hint.is_empty() {
            return false;
        }
        self.category.to_lowercase().contains(&hint)
            || self.note.to_lowercase().contains(&hint)
            || self
                .source
                .as_deref()
                .unwrap_or("")
                .to_lowercase()
                .contains(&hint)
            || self
                .tags
                .iter()
                .any(|tag| tag.to_lowercase().contains(&hint))
    }

    /// Unique key for deduplication
    pub fn dedup_key(&self) -> String {
        format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}",
            self.id.as_deref().unwrap_or(""),
            self.timestamp,
            self.category,
            self.note
        )
    }

    pub fn has_local_env_path_leak(&self) -> bool {
        note_has_local_env_path_leak(&self.note)
    }
}

/// Alias for backward compatibility during migration
pub type AgentMemoryEntry = KnowledgeEntry;

#[cfg(test)]
mod tests {
    use super::note_has_local_env_path_leak;

    #[test]
    fn local_env_path_leak_detects_home_relative_and_absolute_paths() {
        assert!(note_has_local_env_path_leak(
            "Skill 文件位置：~/.config/rust_tools/skills/feishu-upload-md.skill，约182行"
        ));
        assert!(note_has_local_env_path_leak(
            "read_file(file=/Users/bytedance/.config/rust_tools/skills/feishu-upload-md.skill)"
        ));
        assert!(note_has_local_env_path_leak(
            r"配置文件位于 C:\Users\bytedance\AppData\Roaming\rust_tools\settings.json"
        ));
    }

    #[test]
    fn local_env_path_leak_ignores_relative_repo_paths_and_web_urls() {
        assert!(!note_has_local_env_path_leak(
            "实现位置在 src/bin/ai/knowledge/entry.rs，继续看这个模块。"
        ));
        assert!(!note_has_local_env_path_leak(
            "文档地址：https://example.com/.config/rust_tools/skills/feishu-upload-md.skill"
        ));
    }
}
