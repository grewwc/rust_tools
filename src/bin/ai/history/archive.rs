use std::path::PathBuf;

use serde_json::Value;

use super::{compress::ARCHIVE_NOTE_PREFIX, types::Message};

const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";

/// 展开压缩器写入 internal_note 的 overflow 归档回指，供 `/history` 查看完整会话。
///
/// 同一路径可能因旧版重复注入而出现多次，只展开一次。归档不可读或格式不完整时
/// 保留原回指，避免把唯一的恢复线索从 `/history` 输出中隐藏掉。
pub(super) fn expand_overflow_archives(messages: Vec<Message>) -> Vec<Message> {
    let mut expanded = Vec::with_capacity(messages.len());
    let mut loaded_paths = Vec::<PathBuf>::new();

    for message in messages {
        let Some(path) = overflow_archive_path(&message) else {
            expanded.push(message);
            continue;
        };
        if loaded_paths.iter().any(|loaded| loaded == &path) {
            continue;
        }

        let archived = std::fs::read_to_string(&path)
            .ok()
            .map(|markdown| parse_overflow_history(&markdown))
            .unwrap_or_default();
        if archived.is_empty() {
            expanded.push(message);
            continue;
        }

        loaded_paths.push(path);
        expanded.extend(archived);
    }
    expanded
}

fn overflow_archive_path(message: &Message) -> Option<PathBuf> {
    if message.role != "internal_note" {
        return None;
    }
    let Value::String(text) = &message.content else {
        return None;
    };
    if !text.trim_start().starts_with(ARCHIVE_NOTE_PREFIX) {
        return None;
    }

    let path = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("归档文件: "))?
        .trim();
    if path.is_empty() {
        return None;
    }
    let path = PathBuf::from(path);
    (path.file_name().and_then(|name| name.to_str()) == Some(OVERFLOW_HISTORY_FILENAME))
        .then_some(path)
}

fn parse_overflow_history(markdown: &str) -> Vec<Message> {
    if let Some(messages) = parse_lossless_raw_messages(markdown) {
        return messages;
    }

    let mut messages = Vec::new();
    let mut current_role: Option<&str> = None;
    let mut content_lines = Vec::<&str>::new();

    for line in markdown.lines() {
        if let Some(role) = overflow_heading_role(line) {
            finish_message(&mut messages, current_role.take(), &mut content_lines);
            current_role = Some(role);
        } else if current_role.is_some() {
            content_lines.push(line);
        }
    }
    finish_message(&mut messages, current_role, &mut content_lines);
    messages
}

/// 优先读取归档里随展示文本一同保存的原始 `Message` JSON。
///
/// Markdown 正文只是给人和搜索工具看的投影；仅靠标题反解析会丢失
/// `tool_call_id` / `tool_calls` / `model` 等元数据，也会被正文中的同名标题干扰。
/// 若发现原始 JSON 但任一块损坏，则整体退回旧版 Markdown 解析，避免静默只恢复
/// 一部分消息。
fn parse_lossless_raw_messages(markdown: &str) -> Option<Vec<Message>> {
    const REMOVED_MESSAGES_BATCH: &str = "## 移出消息原文";
    const TRUNCATED_FIELD_BATCH: &str = "## 截断字段原文";
    const RAW_MARKER: &str = "raw_message_json:";

    let mut messages = Vec::new();
    let mut in_removed_messages_batch = false;
    let mut saw_raw_marker = false;
    let mut lines = markdown.lines();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == REMOVED_MESSAGES_BATCH {
            in_removed_messages_batch = true;
            continue;
        }
        if trimmed == TRUNCATED_FIELD_BATCH || trimmed.starts_with("# Overflow History Archive") {
            in_removed_messages_batch = false;
            continue;
        }
        if !in_removed_messages_batch || trimmed != RAW_MARKER {
            continue;
        }

        saw_raw_marker = true;
        if lines.next().map(str::trim) != Some("```json") {
            return None;
        }
        let mut json = String::new();
        let mut closed = false;
        for json_line in lines.by_ref() {
            if json_line.trim() == "```" {
                closed = true;
                break;
            }
            json.push_str(json_line);
            json.push('\n');
        }
        if !closed {
            return None;
        }
        let Ok(message) = serde_json::from_str::<Message>(json.trim_end()) else {
            return None;
        };
        messages.push(message);
    }

    saw_raw_marker.then_some(messages)
}

fn overflow_heading_role(line: &str) -> Option<&'static str> {
    match line.trim_end_matches('\r') {
        "## 用户" => Some("user"),
        "## 助手" => Some("assistant"),
        "### 工具结果" => Some("tool"),
        "### system" => Some("system"),
        "### internal_note" => Some("internal_note"),
        _ => None,
    }
}

fn finish_message(messages: &mut Vec<Message>, role: Option<&str>, content_lines: &mut Vec<&str>) {
    let Some(role) = role else {
        content_lines.clear();
        return;
    };

    while content_lines
        .last()
        .is_some_and(|line| line.trim().is_empty())
    {
        content_lines.pop();
    }
    // OverflowSink 在不同 append 批次之间写入 `---`；它不是消息正文。
    if content_lines
        .last()
        .is_some_and(|line| line.trim() == "---")
    {
        content_lines.pop();
        while content_lines
            .last()
            .is_some_and(|line| line.trim().is_empty())
        {
            content_lines.pop();
        }
    }
    let first_content = content_lines
        .iter()
        .position(|line| !line.trim().is_empty())
        .unwrap_or(content_lines.len());
    let content = content_lines[first_content..].join("\n");
    content_lines.clear();

    messages.push(Message {
        role: role.to_string(),
        content: Value::String(content),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

#[cfg(test)]
mod tests {
    use super::parse_overflow_history;

    #[test]
    fn parses_overflow_batches_in_original_order() {
        let markdown = "# 溢出对话历史\n\n---\n\n## 用户\n\nfirst user\n\n## 助手\n\nfirst answer\n\n---\n\n### 工具结果\n\ntool output\n\n### internal_note\n\nnote body\n";
        let messages = parse_overflow_history(markdown);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_str(), Some("first user"));
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content.as_str(), Some("first answer"));
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content.as_str(), Some("tool output"));
        assert_eq!(messages[3].role, "internal_note");
        assert_eq!(messages[3].content.as_str(), Some("note body"));
    }

    #[test]
    fn ignores_markdown_headings_inside_message_content() {
        let markdown = "## 用户\n\nquestion\n\n### Details\n\nstill user content\n";
        let messages = parse_overflow_history(markdown);

        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content.as_str(),
            Some("question\n\n### Details\n\nstill user content")
        );
    }

    #[test]
    fn lossless_raw_json_restores_tool_metadata_and_heading_content() {
        let raw = serde_json::json!({
            "role": "tool",
            "content": "before\n## 用户\nafter",
            "tool_calls": null,
            "tool_call_id": "call-42",
            "reasoning_content": "verified provenance"
        });
        let markdown = format!(
            "# Overflow History Archive\n\n## 移出消息原文\n\n## 工具结果\n\nbefore\n## 用户\nafter\n\nraw_message_json:\n```json\n{raw}\n```\n"
        );

        let messages = parse_overflow_history(&markdown);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "tool");
        assert_eq!(messages[0].content.as_str(), Some("before\n## 用户\nafter"));
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call-42"));
        assert_eq!(
            messages[0].reasoning_content.as_deref(),
            Some("verified provenance")
        );
    }

    #[test]
    fn lossless_raw_json_restores_multiple_messages_in_one_batch() {
        let user = serde_json::json!({
            "role": "user",
            "content": "first",
            "tool_calls": null,
            "tool_call_id": null,
            "reasoning_content": null
        });
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": "before\n## 用户\nafter",
            "tool_calls": null,
            "tool_call_id": null,
            "reasoning_content": null
        });
        let markdown = format!(
            "# Overflow History Archive\n\n## 移出消息原文\n\n## 用户\n\nfirst\n\nraw_message_json:\n```json\n{user}\n```\n\n## 助手\n\nbefore\n## 用户\nafter\n\nraw_message_json:\n```json\n{assistant}\n```\n"
        );

        let messages = parse_overflow_history(&markdown);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_str(), Some("first"));
        assert_eq!(messages[1].content.as_str(), Some("before\n## 用户\nafter"));
    }
}
