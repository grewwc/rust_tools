use serde_json::Value;

use super::types::Message;

pub(in crate::ai) fn compress_messages_for_context(
    messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
) -> Vec<Message> {
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    let keep_last = keep_last.min(messages.len());
    if keep_last == 0 {
        return shrink_messages_to_fit(messages, max_chars);
    }

    let split_at = messages.len().saturating_sub(keep_last);
    let (older, recent) = messages.split_at(split_at);
    if older.is_empty() {
        return shrink_messages_to_fit(recent.to_vec(), max_chars);
    }

    let mut out = Vec::new();
    if summary_max_chars > 0 {
        let summary = build_summary_text(older, summary_max_chars);
        if !summary.trim().is_empty() {
            out.push(Message {
                role: "system".to_string(),
                content: Value::String(format!(
                    "对话摘要（自动压缩，以下为早期对话要点）：\n{summary}"
                )),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }
    out.extend_from_slice(recent);
    shrink_messages_to_fit(out, max_chars)
}

fn shrink_messages_to_fit(mut messages: Vec<Message>, max_chars: usize) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }

    if messages.is_empty() {
        return Vec::new();
    }

    truncate_tool_messages(&mut messages, 1200, 120);
    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        if let Some(idx) = first_index_of_role(&messages, "tool") {
            messages.remove(idx);
            continue;
        }
        break;
    }

    let mut start = 0usize;
    while start + 1 < messages.len() && messages_total_chars(&messages[start..]) > max_chars {
        start += 1;
    }
    if start > 0 {
        messages = messages[start..].to_vec();
    }

    if messages_total_chars(&messages) > max_chars {
        truncate_first_message_to_fit(&mut messages, max_chars);
    }

    messages
}

fn truncate_first_message_to_fit(messages: &mut [Message], max_chars: usize) {
    if messages.is_empty() {
        return;
    }

    let remaining_chars = max_chars
        .saturating_sub(messages_total_chars(&messages[1..]))
        .max(50);

    let first = &mut messages[0];
    let text = value_to_string(&first.content);
    let truncated = truncate_to_chars(&text, remaining_chars);
    first.content = Value::String(truncated);
}

fn messages_total_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| value_len_chars(&m.content))
        .sum::<usize>()
}

fn value_len_chars(v: &Value) -> usize {
    v.as_str()
        .map(|s| s.len())
        .unwrap_or_else(|| v.to_string().len())
}

pub(in crate::ai) fn value_to_string(v: &Value) -> String {
    v.as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| v.to_string())
}

fn build_summary_text(messages: &[Message], max_chars: usize) -> String {
    let mut lines = Vec::new();
    for m in messages {
        let role = match m.role.as_str() {
            "user" => "用户",
            "assistant" => "助手",
            "tool" => "工具",
            other => other,
        };
        let text = normalize_whitespace(&value_to_string(&m.content));

        let tool_info = if let Some(ref tool_calls) = m.tool_calls {
            let tool_names: Vec<&str> = tool_calls
                .iter()
                .map(|tc| tc.function.name.as_str())
                .collect();
            if !tool_names.is_empty() {
                format!(" [tools: {}]", tool_names.join(", "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if text.is_empty() && tool_info.is_empty() {
            continue;
        }

        let snippet = truncate_to_chars(&text, 200);
        lines.push(format!("{role}: {snippet}{tool_info}"));
        if lines.join("\n").len() >= max_chars {
            break;
        }
    }
    let joined = lines.join("\n");
    truncate_to_chars(&joined, max_chars)
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::new();
    let mut in_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(ch);
            in_ws = false;
        }
    }
    out.trim().to_string()
}

fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| s.len());
    let mut out = s[..end].to_string();
    out.push('…');
    out
}

fn first_index_of_role(messages: &[Message], role: &str) -> Option<usize> {
    for (i, m) in messages.iter().enumerate() {
        if m.role == role {
            return Some(i);
        }
    }
    None
}

fn truncate_tool_messages(messages: &mut [Message], max_chars_per_msg: usize, max_lines: usize) {
    for m in messages.iter_mut() {
        if m.role.as_str() != "tool" {
            continue;
        }
        let text = value_to_string(&m.content);
        if text.is_empty() {
            continue;
        }
        let mut out = String::new();
        let mut lines = 0usize;
        for line in text.lines() {
            if lines >= max_lines || out.chars().count() + line.chars().count() + 1 > max_chars_per_msg {
                break;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
            lines += 1;
        }
        if out.len() < text.len() {
            out.push_str("\n…");
            m.content = Value::String(out);
        }
    }
}

fn redact_images_except_last(messages: &mut [Message], keep_last: usize) {
    let mut indices = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let text = value_to_string(&m.content);
        if text.contains("data:image/") {
            indices.push(i);
        }
    }
    if indices.len() <= keep_last {
        return;
    }
    let cutoff = indices.len().saturating_sub(keep_last);
    for i in 0..cutoff {
        let idx = indices[i];
        if let Some(m) = messages.get_mut(idx) {
            m.content = Value::String("[[image omitted]]".to_string());
        }
    }
}

fn dedup_adjacent(messages: &mut Vec<Message>) {
    if messages.is_empty() {
        return;
    }
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut prev_role = String::new();
    let mut prev_content = String::new();
    for m in messages.drain(..) {
        let text = value_to_string(&m.content);
        if m.role == prev_role && text == prev_content {
            continue;
        }
        prev_role = m.role.clone();
        prev_content = text;
        out.push(m);
    }
    *messages = out;
}
