use std::{
    fs::File,
    fs::{self},
    io::{self, Write},
    path::{Path, PathBuf},
};

use chrono::{DateTime, Local};
use rust_tools::cw::SkipMap;
use serde_json::json;

use super::{
    blob::{delete_assets_dir, delete_history_artifacts},
    markdown::messages_to_markdown,
    sqlite::{
        checkpoint_wal, read_all_messages_sqlite, read_first_user_prompt_sqlite,
        read_session_title_sqlite, write_session_title_sqlite,
    },
    types::Message,
};

/// 递归复制目录树 `src` -> `dst`（`dst` 不应已存在）。
/// fork_session 时用于完整复制 assets 目录：checkpoint 正文位于嵌套目录中，
/// 浅复制会让 fork 后的 marker 指向缺失文件。
fn copy_dir_recursively(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursively(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

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

    pub(in crate::ai) fn sessions_root(&self) -> &Path {
        &self.root
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
            // 优先使用 LLM 生成的标题（存储在 meta 表中），fallback 到首条消息摘要
            let generated_title = read_session_title_sqlite(&path).unwrap_or(None);
            let summary = generated_title
                .as_deref()
                .map(normalize_generated_session_title)
                .filter(|title| !title.is_empty())
                .or_else(|| first_user_prompt.as_deref().map(generate_session_summary));
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

    /// 判断 session 是否为空（没有任何用户消息）。
    /// 用于交互模式下用户直接 Ctrl+C 退出时清理空 session。
    /// 文件不存在或 messages 表中没有 role='user' 的记录均视为空。
    pub(in crate::ai) fn is_empty_session(&self, session_id: &str) -> io::Result<bool> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(true);
        }
        let count = super::sqlite::count_user_turns_sqlite(&path)?;
        Ok(count == 0)
    }

    /// 读取 LLM 生成的 session 标题。
    pub(in crate::ai) fn read_session_title(&self, session_id: &str) -> io::Result<Option<String>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        read_session_title_sqlite(&path)
    }

    /// 写入 LLM 生成的 session 标题。
    pub(in crate::ai) fn write_session_title(
        &self,
        session_id: &str,
        title: &str,
    ) -> io::Result<()> {
        write_session_title_sqlite(&self.session_history_file(session_id), title)
    }

    /// 检查是否已有 LLM 生成的标题。
    pub(in crate::ai) fn has_generated_title(&self, session_id: &str) -> bool {
        self.read_session_title(session_id).ok().flatten().is_some()
    }

    pub(in crate::ai) fn read_all_messages(&self, session_id: &str) -> io::Result<Vec<Message>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_all_messages_sqlite(&path)
    }

    /// 把 `src` session 整体复制到 `dst` 作为新分支。原 session 不动。
    /// 拒绝覆盖已有 dst（避免误覆盖）。assets 目录如果存在也递归复制。
    pub(in crate::ai) fn fork_session(&self, src: &str, dst: &str) -> io::Result<()> {
        let src_path = self.session_history_file(src);
        if !src_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("source session '{src}' not found"),
            ));
        }
        let dst_path = self.session_history_file(dst);
        let src_assets = self.session_assets_dir(src);
        let dst_assets = self.session_assets_dir(dst);
        if dst_path.exists() || dst_assets.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("destination session '{dst}' already exists"),
            ));
        }
        self.ensure_root_dir()?;
        fs::copy(&src_path, &dst_path)?;

        // assets 目录是可选的；若存在则必须完整复制。checkpoint 正文位于嵌套
        // 目录中，浅复制会让 fork 后的 marker 指向缺失文件。
        if src_assets.is_dir() {
            if let Err(error) = copy_dir_recursively(&src_assets, &dst_assets) {
                let _ = fs::remove_file(&dst_path);
                let _ = fs::remove_dir_all(&dst_assets);
                return Err(error);
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

    /// 将 session 完整打包为 zip 归档（SQLite + assets），用于跨机器迁移。
    /// 归档结构：
    ///   manifest.json   — 版本号 + 原始 session id + 创建时间
    ///   session.sqlite  — 完整的 SQLite 数据库（已 checkpoint，含全部消息/标题/摘要）
    ///   assets/...      — assets 目录内容（若存在）
    pub(in crate::ai) fn export_session_archive(
        &self,
        session_id: &str,
        output_path: &Path,
    ) -> io::Result<()> {
        let sqlite_path = self.session_history_file(session_id);
        if !sqlite_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("session '{session_id}' not found"),
            ));
        }

        // WAL checkpoint：把 -wal 中的数据合并进主库，确保 zip 里的 .sqlite 是完整的。
        let _ = checkpoint_wal(&sqlite_path);

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = File::create(output_path)?;
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // manifest.json
        let manifest = json!({
            "version": 1u32,
            "session_id": session_id,
            "created_at": Local::now().to_rfc3339(),
        });
        zip.start_file("manifest.json", options)
            .map_err(|e| io::Error::other(e.to_string()))?;
        zip.write_all(serde_json::to_vec_pretty(&manifest)?.as_slice())?;

        // session.sqlite
        zip.start_file("session.sqlite", options)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let mut sqlite_file = File::open(&sqlite_path)?;
        std::io::copy(&mut sqlite_file, &mut zip)?;

        // assets/（可选）
        let assets_dir = self.session_assets_dir(session_id);
        if assets_dir.is_dir() {
            add_dir_to_zip(&mut zip, &assets_dir, "assets", options)?;
        }

        zip.finish().map_err(|e| io::Error::other(e.to_string()))?;
        Ok(())
    }

    /// 从 zip 归档导入 session。
    /// `dst_id` 指定导入后的 session id（已存在则报错）。
    /// 返回导入后的 session id。
    pub(in crate::ai) fn import_session_archive(
        &self,
        archive_path: &Path,
        dst_id: &str,
    ) -> io::Result<String> {
        let file = File::open(archive_path)?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| io::Error::other(e.to_string()))?;

        // 读取 manifest（可选，仅用于校验）
        let manifest = {
            let mut buf = Vec::new();
            match archive.by_name("manifest.json") {
                Ok(mut entry) => {
                    std::io::copy(&mut entry, &mut buf)?;
                    serde_json::from_slice::<serde_json::Value>(&buf).ok()
                }
                Err(_) => None,
            }
        };
        let _ = manifest; // 仅用于校验，不强制使用原 id

        let dst_sqlite = self.session_history_file(dst_id);
        if dst_sqlite.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("destination session '{dst_id}' already exists"),
            ));
        }

        self.ensure_root_dir()?;

        // 解压 session.sqlite
        {
            let mut entry = archive.by_name("session.sqlite").map_err(|e| {
                io::Error::other(format!("session.sqlite not found in archive: {e}"))
            })?;
            let mut out = File::create(&dst_sqlite)?;
            std::io::copy(&mut entry, &mut out)?;
        }

        // 解压 assets/（如果存在）
        let dst_assets = self.session_assets_dir(dst_id);
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let name = entry.name().to_string();
            if name == "manifest.json" || name == "session.sqlite" {
                continue;
            }
            // 只处理 assets/ 前缀的条目
            let Some(rel) = name.strip_prefix("assets/") else {
                continue;
            };
            if rel.is_empty() || rel.ends_with('/') {
                continue;
            }
            let out_path = dst_assets.join(rel);
            if entry.is_dir() {
                fs::create_dir_all(&out_path)?;
                continue;
            }
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out = File::create(&out_path)?;
            std::io::copy(&mut entry, &mut out)?;
        }

        Ok(dst_id.to_string())
    }
}

/// 递归地把目录内容添加到 zip 归档中。
fn add_dir_to_zip(
    zip: &mut zip::ZipWriter<File>,
    dir: &Path,
    prefix: &str,
    options: zip::write::SimpleFileOptions,
) -> io::Result<()> {
    let entries = fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let zip_name = format!("{prefix}/{name}");
        if path.is_dir() {
            add_dir_to_zip(zip, &path, &zip_name, options)?;
        } else {
            zip.start_file(&zip_name, options)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let mut f = File::open(&path)?;
            std::io::copy(&mut f, zip)?;
        }
    }
    Ok(())
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
/// 处理 JSON 内容（如图片数据），提取关键信息并生成概括性标题。
/// 与简单截断不同，此函数会：
/// 1. 去掉 agent/命令前缀（如 "a "、"/"）
/// 2. 提取第一句话（到句号/问号/感叹号/换行）
/// 3. 去掉常见的冗余前缀（"帮我"、"请"、"我想"等）
/// 4. 控制在合理长度
pub(in crate::ai) fn generate_session_summary(first_prompt: &str) -> String {
    let text = first_prompt.trim();
    if text.is_empty() {
        return "(空会话)".to_string();
    }

    // 去掉 agent 前缀（如 "a "、"a:"、"agent:"等）
    let text = strip_agent_prefix(text);

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
            // 提取第一句话（到句号/问号/感叹号/换行）
            let first_sentence = extract_first_sentence(msg);
            if !first_sentence.is_empty() {
                all_text_parts.push(first_sentence);
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
    // 去掉常见的冗余前缀，使标题更简洁概括
    let cleaned = strip_filler_prefixes(&combined);
    truncate_summary(&cleaned, 40)
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

/// 清洗模型生成的 session 标题，避免把“帮我/请问”等请求壳写进标题。
pub(in crate::ai) fn normalize_generated_session_title(title: &str) -> String {
    let first_line = title.lines().next().unwrap_or("").trim();
    let without_agent = strip_agent_prefix(first_line);
    let without_request = strip_request_filler_prefixes(without_agent).0;
    truncate_summary(without_request.trim(), 30)
}

/// 判断已有标题是否像原始用户请求片段；这类旧标题允许后续 turn 重新生成覆盖。
pub(in crate::ai) fn is_low_quality_session_title(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() || trimmed.contains('\n') || trimmed.contains('\r') {
        return true;
    }
    let without_agent = strip_agent_prefix(trimmed);
    let (_, stripped_request_prefix) = strip_request_filler_prefixes(without_agent);
    stripped_request_prefix || trimmed.chars().count() > 40
}

fn strip_request_filler_prefixes(mut text: &str) -> (&str, bool) {
    let fillers = [
        "你帮我看一下",
        "你帮我给",
        "请帮我给",
        "麻烦帮我给",
        "帮我看一下",
        "帮我给",
        "能不能帮我",
        "可以帮我",
        "麻烦帮我",
        "请帮我",
        "你帮我",
        "帮我",
        "请",
        "麻烦",
        "拜托",
        "求",
        "我想",
        "我想要",
        "我需要",
        "希望",
        "希望能",
        "想问一下",
        "问一下",
        "请问",
        "想知道",
        "看一下",
        "帮看看",
        "看看",
    ];
    let mut stripped_any = false;
    loop {
        let mut stripped = false;
        let trimmed = text.trim_start();
        for filler in &fillers {
            if let Some(rest) = trimmed.strip_prefix(filler) {
                text = rest.trim_start();
                stripped = true;
                stripped_any = true;
                break;
            }
        }
        if !stripped {
            return (trimmed, stripped_any);
        }
    }
}

/// 去掉 agent/命令前缀（如 "a "、"a:"、"agent:"、"/" 等）。
fn strip_agent_prefix(text: &str) -> &str {
    let t = text.trim_start();
    // 匹配 "a "、"a:"、"a：" 等 agent 前缀
    if let Some(rest) = t.strip_prefix("a ") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("a:") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("a：") {
        return rest.trim_start();
    }
    // 匹配 "/" 开头的命令前缀（去掉命令名，保留参数）
    if let Some(rest) = t.strip_prefix('/') {
        // 跳过命令名（到第一个空白）
        if let Some(space_pos) = rest.find(|c: char| c.is_whitespace()) {
            return rest[space_pos..].trim_start();
        }
        // 只有命令名没有参数，返回空
        return "";
    }
    t
}

/// 提取第一句话（到句号、问号、感叹号或换行）。
fn extract_first_sentence(text: &str) -> String {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut end = text.len();
    for (idx, (i, ch)) in chars.iter().enumerate() {
        match ch {
            // 中文句号/问号/感叹号/换行 → 直接截断
            '。' | '？' | '！' | '\n' => {
                end = *i;
                break;
            }
            // 英文句号 → 需要判断是否是句子边界还是文件名/标识符的一部分
            '.' => {
                let prev_is_alnum = idx > 0 && chars[idx - 1].1.is_alphanumeric();
                let next_is_alnum = idx + 1 < chars.len() && chars[idx + 1].1.is_alphanumeric();
                // 如果前后都是字母/数字（如 a.rs、file.txt、v2.0），不视为句子边界
                if prev_is_alnum && next_is_alnum {
                    continue;
                }
                // 如果后面是空格或字符串结束，视为句子边界
                let next_is_space = idx + 1 < chars.len() && chars[idx + 1].1.is_whitespace();
                let is_last = idx + 1 >= chars.len();
                if next_is_space || is_last {
                    end = *i;
                    break;
                }
            }
            // 英文问号/感叹号 → 直接截断
            '?' | '!' => {
                end = *i;
                break;
            }
            _ => {}
        }
    }
    text[..end].trim().to_string()
}

/// 去掉常见的冗余前缀，使标题更简洁概括。
fn strip_filler_prefixes(text: &str) -> String {
    let fillers = [
        "你帮我看一下",
        "你帮我给",
        "请帮我给",
        "麻烦帮我给",
        "帮我看一下",
        "帮我给",
        "能不能帮我",
        "可以帮我",
        "麻烦帮我",
        "请帮我",
        "你帮我",
        "帮我",
        "请",
        "麻烦",
        "拜托",
        "求",
        "我想",
        "我想要",
        "我需要",
        "希望",
        "希望能",
        "想问一下",
        "问一下",
        "请问",
        "想知道",
        "看一下",
        "帮看看",
        "看看",
        "如何",
        "怎么",
        "怎样",
    ];
    let mut t = text.trim();
    loop {
        let mut stripped = false;
        for filler in &fillers {
            if let Some(rest) = t.strip_prefix(filler) {
                t = rest.trim_start();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        generate_session_summary, is_low_quality_session_title, normalize_generated_session_title,
    };

    #[test]
    fn session_title_fallback_strips_compound_filler_prefixes() {
        assert_eq!(
            generate_session_summary("你帮我给a.rs这个agent的system prompt加一个限制吧"),
            "a.rs这个agent的system prompt加一个限制吧"
        );
        assert_eq!(
            generate_session_summary("帮我给 session title 问题排查一下"),
            "session title 问题排查一下"
        );
    }

    #[test]
    fn low_quality_session_titles_are_normalized_or_regenerated() {
        let bad_title = "你帮我给a.rs这个agent的system prompt加一个限制吧";
        assert!(is_low_quality_session_title(bad_title));
        assert_eq!(
            normalize_generated_session_title(bad_title),
            "a.rs这个agent的system prompt加一个限制…"
        );

        assert!(!is_low_quality_session_title("session title 问题排查"));
    }
}
