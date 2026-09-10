use std::path::PathBuf;

use serde_json::Value;

use super::{compress::ARCHIVE_NOTE_PREFIX, types::Message};

const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";

/// Expand overflow-archive back-references written by the compressor into
/// `internal_note` messages, so `/history` can show the full session.
///
/// The same path may appear multiple times due to duplicate injection by older
/// versions; expand it only once. If the archive is unreadable or the format is
/// incomplete, keep the original back-reference so the only recovery clue is
/// not hidden from the `/history` output.
///
/// Expanded messages report *unknown* source provenance (`None`): the archive
/// persists only the `Message` bodies and never a per-message model, and the
/// stub's own persisted model is the model active when the stub was written —
/// which can differ from the models that produced the archived content.
/// Inheriting the stub's model would therefore be a false attribution.
pub(super) fn expand_overflow_archives(
    messages: Vec<(Message, Option<String>)>,
) -> Vec<(Message, Option<String>)> {
    let mut expanded = Vec::with_capacity(messages.len());
    let mut loaded_paths = Vec::<PathBuf>::new();

    for (message, source_model) in messages {
        let Some(path) = overflow_archive_path(&message) else {
            expanded.push((message, source_model));
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
            expanded.push((message, source_model));
            continue;
        }

        loaded_paths.push(path);
        // Archived messages have no per-message model (see the doc comment);
        // report unknown rather than inheriting the stub's model.
        expanded.extend(archived.into_iter().map(|m| (m, None)));
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

/// Prefer the raw `Message` JSON stored alongside the display text.
///
/// The Markdown body is only a projection for humans and search tools; parsing
/// it back from headings alone would lose metadata such as `tool_call_id` /
/// `tool_calls` / `model` and could be confused by identical headings inside the
/// body. If raw JSON is found but any block is corrupt, fall back to the legacy
/// Markdown parser for the whole file instead of silently restoring only part
/// of the messages.
fn parse_lossless_raw_messages(markdown: &str) -> Option<Vec<Message>> {
    // Envelope text switched from Chinese to English; archives written by older
    // builds may still hold either spelling, so both must keep parsing.
    const REMOVED_MESSAGES_BATCH: &str = "## Removed messages (verbatim)";
    const REMOVED_MESSAGES_BATCH_LEGACY: &str = "## 移出消息原文";
    const TRUNCATED_FIELD_BATCH: &str = "## Truncated field original text";
    const TRUNCATED_FIELD_BATCH_LEGACY: &str = "## 截断字段原文";
    const RAW_MARKER: &str = "raw_message_json:";

    let mut messages = Vec::new();
    let mut in_removed_messages_batch = false;
    let mut saw_raw_marker = false;
    let mut lines = markdown.lines();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == REMOVED_MESSAGES_BATCH || trimmed == REMOVED_MESSAGES_BATCH_LEGACY {
            in_removed_messages_batch = true;
            continue;
        }
        if trimmed == TRUNCATED_FIELD_BATCH
            || trimmed == TRUNCATED_FIELD_BATCH_LEGACY
            || trimmed.starts_with("# Overflow History Archive")
        {
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
        "## 用户" | "## User" => Some("user"),
        "## 助手" | "## Assistant" => Some("assistant"),
        "### 工具结果" | "### Tool result" => Some("tool"),
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
    // OverflowSink writes `---` between different append batches; it is not
    // message content.
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
    use super::{expand_overflow_archives, parse_overflow_history, Message};
    use serde_json::Value;
    use uuid::Uuid;

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

    #[test]
    fn lossless_raw_json_parses_english_envelope() {
        // New archives write English envelope text; batch gating must behave
        // exactly like the legacy Chinese format.
        let user = serde_json::json!({
            "role": "user",
            "content": "first",
            "tool_calls": null,
            "tool_call_id": null,
            "reasoning_content": null
        });
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": "truncated original",
            "tool_calls": null,
            "tool_call_id": null,
            "reasoning_content": null
        });
        let markdown = format!(
            "# Overflow History Archive\n\n## Removed messages (verbatim)\n\n## User\n\nfirst\n\nraw_message_json:\n```json\n{user}\n```\n\n---\n\n## Truncated field original text\n\n### Field original text\n\n- role: assistant\n- field: content\n\nBegin original text\ntruncated original\nEnd original text\n\nraw_message_json:\n```json\n{assistant}\n```\n"
        );

        let messages = parse_overflow_history(&markdown);

        // Only the removed-messages batch is restored; the truncated-field
        // batch raw JSON must stay excluded.
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.as_str(), Some("first"));
    }

    #[test]
    fn expanded_archive_messages_do_not_inherit_stub_source_model() {
        // The stub recognizer only accepts files named `overflow-history.md`,
        // so host the temp archive in its own directory.
        let dir = std::env::temp_dir().join(format!("ai-archive-model-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let archive_path = dir.join("overflow-history.md");
        // Plain-Markdown archive body, the same shape the legacy write path
        // produced (no per-message raw JSON, hence no model either).
        std::fs::write(&archive_path, "## Assistant\n\narchived answer\n").unwrap();

        let stub = Message {
            role: "internal_note".to_string(),
            content: Value::String(format!(
                "长期记忆归档：更早的原始对话已移出上下文窗口，原文保存在会话归档文件中（零压缩）。\n归档文件: {}",
                archive_path.display()
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        let ordinary = Message {
            role: "user".to_string(),
            content: Value::String("still in canonical history".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };

        let expanded = expand_overflow_archives(vec![
            (ordinary, Some("gpt-5.5".to_string())),
            (stub, Some("glm-5.2-opencode".to_string())),
        ]);

        // Non-stub rows keep their own provenance; expanded archive rows must
        // report unknown instead of inheriting the stub's (potentially
        // unrelated) model.
        assert_eq!(expanded[0].0.role, "user");
        assert_eq!(expanded[0].1.as_deref(), Some("gpt-5.5"));
        assert_eq!(expanded[1].0.role, "assistant");
        assert_eq!(expanded[1].0.content.as_str(), Some("archived answer"));
        assert_eq!(expanded[1].1, None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
