use serde_json::Value;

use super::types::{MAX_HISTORY_TURNS, Message, retained_turn_start};

const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;
const PERSISTED_HISTORY_SUMMARY_MAX_CHARS: usize = 8_000;

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

pub(in crate::ai) fn compact_persisted_history(messages: Vec<Message>) -> Vec<Message> {
    let user_turns = messages.iter().filter(|message| message.role == "user").count();
    if user_turns <= MAX_HISTORY_TURNS {
        return messages;
    }

    let keep_recent_turns = PERSISTED_HISTORY_KEEP_RECENT_TURNS.min(MAX_HISTORY_TURNS - 1);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return messages;
    }

    let summary = build_persisted_summary_text(&messages[..split_at], PERSISTED_HISTORY_SUMMARY_MAX_CHARS);
    let mut out = Vec::with_capacity(messages.len() - split_at + 1);
    if !summary.is_empty() {
        out.push(Message {
            role: "system".to_string(),
            content: Value::String(format!(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n{summary}"
            )),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    out.extend_from_slice(&messages[split_at..]);
    out
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

fn build_persisted_summary_text(messages: &[Message], max_chars: usize) -> String {
    #[derive(Default, Clone)]
    struct TurnSummary {
        topic_key: String,
        topic_label: String,
        user: String,
        user_key: String,
        assistant_final: String,
        tool_names: Vec<String>,
        tool_highlights: Vec<String>,
        count: usize,
    }

    fn strip_summary_header(text: &str) -> String {
        let trimmed = text.trim();
        for prefix in [
            "历史摘要（自动压缩，以下为更早对话的简短语义）：",
            "对话摘要（自动压缩，以下为早期对话要点）：",
        ] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                return rest.trim().to_string();
            }
        }
        trimmed.to_string()
    }

    fn normalize_semantic_key(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            let is_cjk = ('\u{4E00}'..='\u{9FFF}').contains(&ch);
            if is_cjk || ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
                continue;
            }
            if ch.is_whitespace() {
                out.push(' ');
            }
        }
        normalize_whitespace(&out)
    }

    fn extract_topic_from_text(text: &str) -> Option<(String, String)> {
        fn trim_punct(s: &str) -> &str {
            s.trim_matches(|ch: char| {
                ch.is_whitespace()
                    || matches!(
                        ch,
                        ',' | '.' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}'
                            | '<' | '>' | '"' | '\'' | '`'
                    )
            })
        }

        fn candidate_file_token(token: &str) -> Option<&str> {
            let token = trim_punct(token);
            if token.is_empty() || token.len() > 96 {
                return None;
            }
            if token.starts_with("http://") || token.starts_with("https://") {
                return None;
            }
            let token = token.split('#').next().unwrap_or(token);
            let token = token.split('?').next().unwrap_or(token);
            let token = token.split_once(':').map(|(a, _)| a).unwrap_or(token);
            let suffixes = [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift",
                ".c", ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ];
            if suffixes.iter().any(|suf| token.ends_with(suf)) {
                return Some(token);
            }
            None
        }

        fn basename(path: &str) -> &str {
            path.rsplit('/').next().unwrap_or(path)
        }

        fn find_error_code(text: &str) -> Option<String> {
            let bytes = text.as_bytes();
            let mut i = 0usize;
            while i + 5 <= bytes.len() {
                if bytes[i] == b'E'
                    && bytes[i + 1].is_ascii_digit()
                    && bytes[i + 2].is_ascii_digit()
                    && bytes[i + 3].is_ascii_digit()
                    && bytes[i + 4].is_ascii_digit()
                {
                    let code = &text[i..i + 5];
                    return Some(code.to_string());
                }
                i += 1;
            }
            None
        }

        if let Some(code) = find_error_code(text) {
            return Some((code.to_ascii_lowercase(), code));
        }

        for raw in text.split_whitespace() {
            if let Some(token) = candidate_file_token(raw) {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
            let token = trim_punct(raw);
            if token.contains('/')
                && token.len() <= 96
                && token.chars().any(|c| c == '.')
                && !token.starts_with("http://")
                && !token.starts_with("https://")
            {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
        }

        None
    }

    fn push_unique_limited(target: &mut Vec<String>, value: String, max_items: usize) {
        if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items {
            return;
        }
        target.push(value);
    }

    fn tool_highlight(text: &str) -> String {
        let text = normalize_whitespace(text);
        if text.is_empty() {
            return String::new();
        }
        let lowered = text.to_ascii_lowercase();
        let important = lowered.contains("error")
            || lowered.contains("failed")
            || lowered.contains("panic")
            || lowered.contains("exception")
            || lowered.contains("[error]");
        if important {
            return truncate_to_chars(&text, 120);
        }
        let first_line = text.lines().next().unwrap_or("").trim();
        truncate_to_chars(first_line, 90)
    }

    fn finalize_turn(turns: &mut Vec<TurnSummary>, current: &mut TurnSummary) {
        if current.user.trim().is_empty()
            && current.assistant_final.trim().is_empty()
            && current.tool_names.is_empty()
            && current.tool_highlights.is_empty()
        {
            *current = TurnSummary::default();
            return;
        }
        if current.count == 0 {
            current.count = 1;
        }
        turns.push(current.clone());
        *current = TurnSummary::default();
    }

    fn merge_turns(mut turns: Vec<TurnSummary>) -> Vec<TurnSummary> {
        let mut out: Vec<TurnSummary> = Vec::with_capacity(turns.len());
        for turn in turns.drain(..) {
            if let Some(last) = out.last_mut()
                && ((!turn.user_key.is_empty() && last.user_key == turn.user_key)
                    || (!turn.topic_key.is_empty() && last.topic_key == turn.topic_key))
            {
                last.count = last.count.saturating_add(turn.count.max(1));
                if last.topic_label.is_empty() && !turn.topic_label.is_empty() {
                    last.topic_label = turn.topic_label;
                    last.topic_key = turn.topic_key;
                }
                if !turn.assistant_final.is_empty()
                    && turn.assistant_final != last.assistant_final
                    && last.assistant_final.chars().count() < 160
                {
                    if last.assistant_final.is_empty() {
                        last.assistant_final = turn.assistant_final;
                    } else {
                        last.assistant_final = truncate_to_chars(
                            &format!("{} / {}", last.assistant_final, turn.assistant_final),
                            180,
                        );
                    }
                }
                for name in turn.tool_names {
                    push_unique_limited(&mut last.tool_names, name, 4);
                }
                for h in turn.tool_highlights {
                    push_unique_limited(&mut last.tool_highlights, h, 2);
                }
                continue;
            }
            out.push(turn);
        }
        out
    }

    fn render_line(turn: &TurnSummary) -> String {
        let mut line = String::new();
        if turn.count > 1 {
            line.push_str(&format!("重复×{} ", turn.count));
        }
        if !turn.topic_label.is_empty() {
            line.push_str("主题: ");
            line.push_str(&turn.topic_label);
            line.push_str(" | ");
        }
        if !turn.user.is_empty() {
            line.push_str("用户: ");
            line.push_str(&turn.user);
        }
        if !turn.assistant_final.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("结论: ");
            line.push_str(&turn.assistant_final);
        }
        if !turn.tool_names.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("工具: ");
            line.push_str(&turn.tool_names.join(", "));
        }
        if !turn.tool_highlights.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("关键: ");
            line.push_str(&turn.tool_highlights.join("；"));
        }
        truncate_to_chars(&line, 280)
    }

    let mut pre_summary_lines: Vec<String> = Vec::new();
    let mut turns: Vec<TurnSummary> = Vec::new();
    let mut current = TurnSummary::default();

    for message in messages {
        let text = normalize_whitespace(&value_to_string(&message.content));
        match message.role.as_str() {
            "system" => {
                let normalized = strip_summary_header(&text);
                if normalized.is_empty() || normalized.starts_with("self_note:") {
                    continue;
                }
                if normalized.contains("历史摘要") || normalized.contains("对话摘要") {
                    pre_summary_lines.push(format!(
                        "- 更早摘要: {}",
                        truncate_to_chars(&normalized, 240)
                    ));
                    continue;
                }
                let normalized = truncate_to_chars(&normalized, 240);
                if !normalized.is_empty() {
                    pre_summary_lines.push(format!("- 更早摘要: {normalized}"));
                }
            }
            "user" => {
                finalize_turn(&mut turns, &mut current);
                current.user = truncate_to_chars(&text, 120);
                current.user_key = truncate_to_chars(&normalize_semantic_key(&text), 160);
                if let Some((k, label)) = extract_topic_from_text(&text) {
                    current.topic_key = k;
                    current.topic_label = label;
                }
                if current.count == 0 {
                    current.count = 1;
                }
            }
            "assistant" => {
                if !text.is_empty() {
                    current.assistant_final = truncate_to_chars(&text, 180);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&text) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        push_unique_limited(&mut current.tool_names, tool_call.function.name.clone(), 4);
                        if current.topic_key.is_empty() {
                            current.topic_key = tool_call.function.name.to_ascii_lowercase();
                            current.topic_label = tool_call.function.name.clone();
                        }
                    }
                }
            }
            "tool" => {
                let h = tool_highlight(&text);
                if !h.is_empty() {
                    push_unique_limited(&mut current.tool_highlights, h.clone(), 2);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&h) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    finalize_turn(&mut turns, &mut current);

    let merged = merge_turns(turns);
    let mut lines: Vec<String> = Vec::new();
    for s in pre_summary_lines.into_iter().take(3) {
        lines.push(s);
    }
    for t in merged {
        lines.push(format!("- {}", render_line(&t)));
        if lines.join("\n").chars().count() >= max_chars {
            break;
        }
    }

    truncate_to_chars(&lines.join("\n"), max_chars)
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
