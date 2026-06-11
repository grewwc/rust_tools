use std::{
    fs::File,
    fs::{self},
    io,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Local};
use rust_tools::cw::SkipMap;

use super::{
    blob::{delete_assets_dir, delete_history_artifacts},
    markdown::messages_to_markdown,
    sqlite::{read_all_messages_sqlite, read_first_user_prompt_sqlite},
    types::Message,
};

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionInfo {
    pub(in crate::ai) id: String,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) size_bytes: u64,
    pub(in crate::ai) first_user_prompt: Option<String>,
    pub(in crate::ai) summary: Option<String>,
}

impl SessionStore {
    pub(in crate::ai) fn new(history_file: &Path) -> Self {
        Self {
            root: sessions_root_from_history_file(history_file),
        }
    }

    pub(in crate::ai) fn ensure_root_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub(in crate::ai) fn session_history_file(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.sqlite"))
    }

    pub(in crate::ai) fn session_assets_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.assets"))
    }

    /// 该 session 的 checkpoint 存放目录：`<sessions_root>/checkpoints/<id>/`。
    pub(in crate::ai) fn checkpoints_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join("checkpoints").join(id)
    }

    pub(in crate::ai) fn list_sessions(&self) -> io::Result<Vec<SessionInfo>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut sessions: Box<SkipMap<(u64, String), SessionInfo>> =
            SkipMap::new(16, |a: &(u64, String), b: &(u64, String)| {
                match b.0.cmp(&a.0) {
                    std::cmp::Ordering::Equal => a.1.cmp(&b.1) as i32 * -1,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Greater => -1,
                }
            });
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let modified_local = metadata.modified().ok().map(DateTime::<Local>::from);
            let first_user_prompt = read_first_user_prompt_sqlite(&path).unwrap_or(None);
            let id = stem.to_string();
            let summary = first_user_prompt.as_deref().map(generate_session_summary);
            let timestamp = modified_local
                .map(|dt| dt.timestamp_millis() as u64)
                .unwrap_or(0);
            sessions.insert(
                (timestamp, id.clone()),
                SessionInfo {
                    id,
                    modified_local,
                    size_bytes: metadata.len(),
                    first_user_prompt,
                    summary,
                },
            );
        }
        Ok(sessions.into_iter().map(|(_, v)| v).collect())
    }

    pub(in crate::ai) fn delete_session(&self, session_id: &str) -> io::Result<bool> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let existed = path.exists();
        delete_history_artifacts(&path)?;
        let _ = delete_assets_dir(&assets);
        Ok(existed)
    }

    pub(in crate::ai) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let _ = delete_history_artifacts(&path);
        let _ = delete_assets_dir(&assets);
        Ok(())
    }

    pub(in crate::ai) fn clear_session_history(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        if path.exists() {
            super::sqlite::clear_session_history_sqlite(&path)?;
        }
        let assets = self.session_assets_dir(session_id);
        let _ = delete_assets_dir(&assets);
        Ok(())
    }

    pub(in crate::ai) fn clear_all_sessions(&self) -> io::Result<usize> {
        let sessions = self.list_sessions()?;
        let mut deleted = 0usize;
        for s in sessions {
            if self.delete_session(&s.id).is_ok() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    pub(in crate::ai) fn first_user_prompt(&self, session_id: &str) -> io::Result<Option<String>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        read_first_user_prompt_sqlite(&path)
    }

    pub(in crate::ai) fn read_all_messages(&self, session_id: &str) -> io::Result<Vec<Message>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_all_messages_sqlite(&path)
    }

    /// 把 `src` session 整体复制到 `dst` 作为新分支。原 session 不动。
    /// 拒绝覆盖已有 dst（避免误覆盖）。assets 目录如果存在也一并复制。
    pub(in crate::ai) fn fork_session(&self, src: &str, dst: &str) -> io::Result<()> {
        let src_path = self.session_history_file(src);
        if !src_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("source session '{src}' not found"),
            ));
        }
        let dst_path = self.session_history_file(dst);
        if dst_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("destination session '{dst}' already exists"),
            ));
        }
        self.ensure_root_dir()?;
        fs::copy(&src_path, &dst_path)?;

        // assets 目录是可选的；存在则尽力浅复制
        let src_assets = self.session_assets_dir(src);
        if src_assets.is_dir() {
            let dst_assets = self.session_assets_dir(dst);
            let _ = fs::create_dir_all(&dst_assets);
            if let Ok(entries) = fs::read_dir(&src_assets) {
                for entry in entries.flatten() {
                    let from = entry.path();
                    if let Some(name) = entry.file_name().to_str() {
                        let to = dst_assets.join(name);
                        let _ = fs::copy(&from, &to);
                    }
                }
            }
        }
        Ok(())
    }

    /// 在 `src` 之上分支，并把分支保留到第 `keep_messages` 条消息（按 id 升序）。
    /// 适合"我想从某轮回滚后换个方向继续"的场景。
    pub(in crate::ai) fn branch_session(
        &self,
        src: &str,
        dst: &str,
        keep_messages: usize,
    ) -> io::Result<()> {
        self.fork_session(src, dst)?;
        let dst_path = self.session_history_file(dst);
        super::sqlite::truncate_messages_sqlite(&dst_path, keep_messages)?;
        Ok(())
    }

    pub(in crate::ai) fn export_session_to_markdown(
        &self,
        session_id: &str,
        output_path: &Path,
    ) -> io::Result<()> {
        let messages = self.read_all_messages(session_id)?;
        if messages.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Session '{}' not found or empty", session_id),
            ));
        }

        let markdown = messages_to_markdown(&messages, session_id);

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(output_path)?;
        use std::io::Write;
        file.write_all(markdown.as_bytes())?;

        Ok(())
    }
}

fn sessions_root_from_history_file(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let name = history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("history");
    parent.join(format!("{name}.sessions"))
}

fn sanitize_session_id(session_id: &str) -> String {
    let mut out = String::new();
    for ch in session_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "session".to_string()
    } else {
        out
    }
}

/// 从第一条用户消息生成一个简洁的 session 标题/摘要。
/// 处理 JSON 内容（如图片数据）、长文本截断、清理空白。
fn generate_session_summary(first_prompt: &str) -> String {
    let text = first_prompt.trim();
    if text.is_empty() {
        return "(空会话)".to_string();
    }

    // 处理多条消息合并的情况（用 \n---\n 分隔）
    let messages: Vec<&str> = text.split("\n---\n").collect();
    let mut all_text_parts = Vec::new();
    let mut has_any_image = false;

    for msg in &messages {
        let msg = msg.trim();
        if msg.is_empty() {
            continue;
        }

        // 尝试解析为 JSON 数组（多模态消息）
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(msg) {
            let (parts, has_image) = extract_from_json_array(&arr);
            all_text_parts.extend(parts);
            if has_image {
                has_any_image = true;
            }
        }
        // 尝试解析为单个 JSON 对象
        else if let Ok(obj) = serde_json::from_str::<serde_json::Value>(msg) {
            if let Some(obj) = obj.as_object() {
                let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match item_type {
                    "text" => {
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            let cleaned = t.trim();
                            if !cleaned.is_empty() {
                                all_text_parts.push(cleaned.to_string());
                            }
                        }
                    }
                    "image_url" => has_any_image = true,
                    _ => {}
                }
            }
        }
        // 普通文本
        else {
            let first_line = msg.lines().next().unwrap_or(msg).trim();
            if !first_line.is_empty() {
                all_text_parts.push(first_line.to_string());
            }
        }
    }

    if all_text_parts.is_empty() && has_any_image {
        return "[图片]".to_string();
    }
    if all_text_parts.is_empty() {
        return "(无文本内容)".to_string();
    }

    let combined = all_text_parts.join(" ");
    truncate_summary(&combined, 60)
}

/// 从 JSON 数组中提取文本部分和图片标记。
fn extract_from_json_array(arr: &[serde_json::Value]) -> (Vec<String>, bool) {
    let mut parts = Vec::new();
    let mut has_image = false;
    for item in arr {
        if let Some(obj) = item.as_object() {
            let item_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match item_type {
                "text" => {
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        let cleaned = t.trim();
                        if !cleaned.is_empty() {
                            parts.push(cleaned.to_string());
                        }
                    }
                }
                "image_url" => has_image = true,
                _ => {}
            }
        }
    }
    (parts, has_image)
}

/// 截断摘要到指定长度，添加省略号。
fn truncate_summary(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_len).collect();
    out.push_str("…");
    out
}
