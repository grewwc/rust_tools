use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::{request, types::App};

use super::types::{
    MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start,
};

const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;
const PERSISTED_HISTORY_SUMMARY_MAX_CHARS: usize = 8_000;
const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";

struct OverflowSink {
    path: PathBuf,
    buffer: String,
}

impl OverflowSink {
    fn new(overflow_dir: &Path) -> Self {
        let path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
        Self {
            path,
            buffer: String::new(),
        }
    }

    fn push_messages(&mut self, messages: &[Message]) {
        if messages.is_empty() {
            return;
        }
        if self.buffer.is_empty() {
            self.buffer.push_str(
                "# 溢出对话历史\n\n以下内容因超出上下文窗口而被移出，模型可使用 read_file 工具读取此文件回顾。\n\n---\n\n",
            );
        }
        for msg in messages {
            let text = value_to_string(&msg.content);
            match msg.role.as_str() {
                "user" => {
                    self.buffer.push_str("## 用户\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
                "assistant" => {
                    self.buffer.push_str("## 助手\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
                "tool" => {
                    self.buffer.push_str("### 工具结果\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
                _ => {
                    self.buffer.push_str("### ");
                    self.buffer.push_str(&msg.role);
                    self.buffer.push_str("\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
            }
        }
    }

    fn flush(&mut self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        use std::io::Write;
        std::fs::File::create(&self.path)
            .and_then(|mut f| {
                f.write_all(self.buffer.as_bytes())?;
                f.sync_data()
            })
            .is_ok()
    }

    fn file_path(&self) -> &Path {
        &self.path
    }
}

fn build_overflow_placeholder(file_path: &str) -> String {
    let mut out = String::new();
    out.push_str("长期记忆归档：更早的原始对话未丢失。\n");
    out.push_str("原始归档文件: ");
    out.push_str(file_path);
    out.push('\n');
    out.push_str("先执行工具: read_file_lines\n参数: file_path=\"");
    out.push_str(file_path);
    out.push_str("\", offset=1, limit=200)\n");
    out.push_str("若当前问题依赖前文细节、最初目标、之前决定、旧报错或更早工具输出，请先读该文件；未读完时继续增大 offset 分段读取。\n");
    out
}

pub(in crate::ai) fn compress_messages_for_context(
    messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
) -> Vec<Message> {
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    let keep_last = keep_last.min(messages.len());
    if keep_last == 0 {
        return shrink_messages_to_fit_with_summary(messages, max_chars, summary_max_chars, overflow_dir.as_deref());
    }

    let split_at = retained_turn_start(&messages, keep_last);
    let (older, recent) = messages.split_at(split_at);
    if older.is_empty() {
        return shrink_messages_to_fit_with_summary(recent.to_vec(), max_chars, summary_max_chars, overflow_dir.as_deref());
    }

    let mut out = Vec::new();
    if summary_max_chars > 0 {
        let summary = build_persisted_summary_text(older, summary_max_chars);
        if !summary.trim().is_empty() {
            out.push(Message {
                role: ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(format!(
                    "对话摘要（自动压缩，以下为早期对话要点）：\n{summary}"
                )),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
    }
    out.extend_from_slice(recent);
    shrink_messages_to_fit_with_summary(out, max_chars, summary_max_chars, overflow_dir.as_deref())
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
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n{summary}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    out.extend_from_slice(&messages[split_at..]);
    out
}

pub(in crate::ai) async fn compact_persisted_history_with_app(
    app: &App,
    messages: Vec<Message>,
) -> Vec<Message> {
    compact_persisted_history_with_app_inner(app, messages, MAX_HISTORY_TURNS).await
}

/// 任务边界（一轮 turn 结束且没有再调工具，意味着 agent 给出了最终答案）触发的
/// 主动压缩。阈值从 `MAX_HISTORY_TURNS`(200) 下调到 `PERSISTED_HISTORY_KEEP_RECENT_TURNS`(160)，
/// 让"任务做完"这种自然分界点提前触发摘要，避免一直堆到硬上限才被动切。
/// 仍然不会摘出还没到 160 的对话，所以短对话不受影响。
pub(in crate::ai) async fn compact_persisted_history_at_boundary_with_app(
    app: &App,
    messages: Vec<Message>,
) -> Vec<Message> {
    compact_persisted_history_with_app_inner(app, messages, PERSISTED_HISTORY_KEEP_RECENT_TURNS)
        .await
}

async fn compact_persisted_history_with_app_inner(
    app: &App,
    messages: Vec<Message>,
    threshold_turns: usize,
) -> Vec<Message> {
    let user_turns = messages.iter().filter(|message| message.role == "user").count();
    if user_turns <= threshold_turns {
        return messages;
    }

    let keep_recent_turns = PERSISTED_HISTORY_KEEP_RECENT_TURNS.min(MAX_HISTORY_TURNS - 1);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return messages;
    }

    let summary = build_persisted_summary_text_with_app(
        app,
        &messages[..split_at],
        PERSISTED_HISTORY_SUMMARY_MAX_CHARS,
    )
    .await;
    let mut out = Vec::with_capacity(messages.len() - split_at + 1);
    if !summary.is_empty() {
        out.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n{summary}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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

    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_MESSAGES);
    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    dedup_repeated_tool_results(&mut messages);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        if let Some(group) = first_tool_call_group(&messages) {
            // 渐进式卸载：先尝试折叠为单行 stub 而不是整组删除，
            // 让模型仍能"看见"早期发生过哪些工具调用、以什么结果收尾，
            // 避免后续轮次因为完全失忆而重复工作。
            if let Some(stub) = fold_tool_call_group_to_stub(&messages, &group) {
                let stub_idx = group[0];
                for idx in group.iter().rev() {
                    messages.remove(*idx);
                }
                messages.insert(stub_idx, stub);
                if messages_total_chars(&messages) <= max_chars {
                    break;
                }
                continue;
            }
            // 兜底：极端情况（无法构造 stub）才整组删除
            for idx in group.into_iter().rev() {
                messages.remove(idx);
            }
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages) {
            messages.remove(idx);
            continue;
        }
        break;
    }

    if messages_total_chars(&messages) > max_chars {
        truncate_first_message_to_fit(&mut messages, max_chars);
    }

    keep_only_recent_reasoning_content(&mut messages);

    messages
}

/// Same as [`shrink_messages_to_fit`] but, before dropping early messages
/// outright, captures them into (or merges them with) a leading
/// `internal_note` summary so that long conversations still retain a
/// semantic memory of earlier user questions.
fn shrink_messages_to_fit_with_summary(
    mut messages: Vec<Message>,
    max_chars: usize,
    summary_max_chars: usize,
    overflow_dir: Option<&Path>,
) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }
    if messages.is_empty() {
        return Vec::new();
    }

    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_MESSAGES);
    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    dedup_repeated_tool_results(&mut messages);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    let had_leading_summary = messages.first().map(is_summary_message).unwrap_or(false);
    let mut dropped: Vec<Message> = Vec::new();

    while messages_total_chars(&messages) > max_chars {
        if let Some(group) = first_tool_call_group(&messages) {
            let mut removed_group = Vec::with_capacity(group.len());
            for idx in group.into_iter().rev() {
                removed_group.push(messages.remove(idx));
            }
            removed_group.reverse();
            dropped.extend(removed_group);
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages) {
            dropped.push(messages.remove(idx));
            continue;
        }
        break;
    }

    let dropped_has_user_turn = dropped.iter().any(|m| m.role == "user");
    let has_leading_summary_now = messages.first().map(is_summary_message).unwrap_or(false);

    if !dropped.is_empty() {
        if let Some(dir) = overflow_dir {
            let mut sink = OverflowSink::new(dir);
            sink.push_messages(&dropped);

            if sink.flush() {
                let file_path_str = sink.file_path().to_string_lossy().to_string();
                let summary_body = if dropped_has_user_turn
                    && !has_leading_summary_now
                    && !had_leading_summary
                    && summary_max_chars > 0
                {
                    let header_bytes = "对话摘要（自动压缩，以下为早期对话要点）：\n".len();
                    let used = messages_total_chars(&messages);
                    let body_byte_budget = max_chars.saturating_sub(used).saturating_sub(header_bytes);
                    let body_budget = (body_byte_budget / 3).min(summary_max_chars);
                    if body_budget >= 40 {
                        let text = build_persisted_summary_text(&dropped, body_budget);
                        if !text.trim().is_empty() {
                            Some(text)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let archive_note = build_overflow_placeholder(&file_path_str);
                let fallback_goal = dropped
                    .iter()
                    .find(|message| message.role == "user")
                    .map(|message| summarize_text(&normalize_whitespace(&value_to_string(&message.content)), 160));
                let memory_note = summary_body
                    .as_ref()
                    .filter(|s| !s.trim().is_empty())
                    .map(|summary| format!("长期记忆摘要（压缩保留）:\n{summary}"))
                    .or_else(|| {
                        fallback_goal
                            .as_ref()
                            .filter(|goal| !goal.trim().is_empty())
                            .map(|goal| format!("长期记忆摘要（压缩保留）:\n初始目标: {goal}"))
                    })
                    .unwrap_or_else(|| {
                        "长期记忆摘要（压缩保留）:\n较早原始对话已移出当前窗口；如果当前问题依赖前文细节，请读取归档文件。".to_string()
                    });

                if has_leading_summary_now {
                    let archive_idx = messages.len().min(1);
                    messages.insert(
                        archive_idx,
                        Message {
                            role: ROLE_INTERNAL_NOTE.to_string(),
                            content: Value::String(archive_note),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        },
                    );
                } else {
                    messages.insert(
                        0,
                        Message {
                            role: ROLE_INTERNAL_NOTE.to_string(),
                            content: Value::String(memory_note),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        },
                    );
                    let archive_idx = messages.len().min(1);
                    messages.insert(
                        archive_idx,
                        Message {
                            role: ROLE_INTERNAL_NOTE.to_string(),
                            content: Value::String(archive_note),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        },
                    );
                }
            }
        } else if dropped_has_user_turn
            && !has_leading_summary_now
            && !had_leading_summary
            && summary_max_chars > 0
        {
            let header_prefix = "对话摘要（自动压缩，以下为早期对话要点）：\n";
            let header_bytes = header_prefix.len();
            let used = messages_total_chars(&messages);
            let body_byte_budget = max_chars.saturating_sub(used).saturating_sub(header_bytes);
            let body_budget = (body_byte_budget / 3).min(summary_max_chars);
            if body_budget >= 40 {
                let summary_text = build_persisted_summary_text(&dropped, body_budget);
                if !summary_text.trim().is_empty() {
                    let note = Message {
                        role: ROLE_INTERNAL_NOTE.to_string(),
                        content: Value::String(format!("{header_prefix}{summary_text}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    };
                    messages.insert(0, note);
                }
            }
        }
    }

    if messages_total_chars(&messages) > max_chars {
        truncate_first_message_to_fit(&mut messages, max_chars);
    }

    keep_only_recent_reasoning_content(&mut messages);

    messages
}

#[allow(dead_code)]
fn take_leading_summary(messages: &mut Vec<Message>) -> Option<Message> {
    if messages.first().map(is_summary_message).unwrap_or(false) {
        Some(messages.remove(0))
    } else {
        None
    }
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

/// Public proxy of [`messages_total_chars`] for callers in other ai modules
/// (e.g. mid-turn compression in `turn_runtime`) that need to check budget
/// without re-implementing the same accounting.
pub(in crate::ai) fn messages_total_chars_pub(messages: &[Message]) -> usize {
    messages_total_chars(messages)
}

/// Mid-turn 渐进式压缩：在 iteration loop 中复用跨 turn 压缩管线的前几档。
/// 只做"无损/弱损"操作，不动 system / 不删除最近 keep_recent 条工具消息：
///   1. dedup_repeated_tool_results — 同 (tool, args) 旧结果折叠为 stub
///   2. prepare_tool_messages_structured — 远端 tool 结果按行裁剪到 480 字
///   3. fold_tool_call_group_to_stub  — 仍超额：远端整组 (assistant + tool) 折叠
/// 返回：(messages_after, before_chars, after_chars)
pub(in crate::ai) fn mid_turn_compress(
    messages: Vec<Message>,
    soft_threshold: usize,
) -> (Vec<Message>, usize, usize) {
    let before = messages_total_chars(&messages);
    if before <= soft_threshold {
        return (messages, before, before);
    }
    let mut out = messages;
    // 1. 同 signature 工具结果去重
    dedup_repeated_tool_results(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. 远端结构化裁剪：tool 结果中段按行折叠到 480 字/条，最近 6 条保留全文
    prepare_tool_messages_structured(&mut out, 480, 6);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 3. 仍超额：用 shrink_messages_to_fit 走"折叠 tool group + 整体兜底"
    out = shrink_messages_to_fit(out, soft_threshold);
    let after = messages_total_chars(&out);
    (out, before, after)
}

/// Mid-turn LLM 摘要兜底：当无损管线后仍超过 hard_threshold 时调用。
/// 行为：
///   - 保留 system + 最近 `keep_recent_turns` 个 user 起始的尾部窗口
///   - 把更早的部分（仅当至少含一个 user/assistant 时）调 LLM 摘要器压缩
///   - 摘要文本作为单条 `internal_note` 注入到尾部窗口前
///   - 如果 LLM 调用失败或没有可摘要的早期部分，原样返回
/// 返回 `(messages_after, before, after, did_summarize)`
pub(in crate::ai) async fn mid_turn_llm_summarize(
    app: &App,
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
) -> (Vec<Message>, usize, usize, bool) {
    let before = messages_total_chars(&messages);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return (messages, before, before, false);
    }
    let earlier = &messages[..split_at];
    let has_dialog = earlier
        .iter()
        .any(|m| m.role == "user" || m.role == "assistant");
    if !has_dialog {
        return (messages, before, before, false);
    }

    let summary = build_persisted_summary_text_with_app(app, earlier, summary_max_chars).await;
    if summary.trim().is_empty() {
        return (messages, before, before, false);
    }

    let mut out = Vec::with_capacity(messages.len() - split_at + 1);
    out.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "[mid-turn-summary] 早期工具调用与对话已被 LLM 摘要：\n{summary}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    out.extend_from_slice(&messages[split_at..]);
    let after = messages_total_chars(&out);
    (out, before, after, true)
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

async fn build_persisted_summary_text_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> String {
    let mut prepared = messages.to_vec();
    prepare_tool_messages_structured(&mut prepared, 360, KEEP_RECENT_TOOL_MESSAGES);
    redact_images_except_last(&mut prepared, 0);
    dedup_adjacent(&mut prepared);

    if let Some(summary) = request::summarize_history_via_model(app, &prepared, max_chars).await {
        let summary = normalize_whitespace(&summary);
        if !summary.is_empty() {
            return summary;
        }
    }

    build_persisted_summary_text(messages, max_chars)
}

fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent: usize,
) {
    let indices = tool_message_indices(messages);
    let protect_from = indices.len().saturating_sub(keep_recent);
    for (rank, &idx) in indices.iter().enumerate() {
        if rank >= protect_from {
            break;
        }
        let message = &mut messages[idx];
        let text = value_to_string(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        let summary = structured_tool_output_summary(&text, max_chars_per_msg);
        if !summary.is_empty() && summary != text {
            message.content = Value::String(summary);
        }
    }
}

fn structured_tool_output_summary(text: &str, max_chars: usize) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return String::new();
    }
    if lines.len() <= 8 {
        let mut out = Vec::new();
        let mut used = 0usize;
        for line in lines
            .into_iter()
            .map(tool_line_signature)
            .filter(|line| !line.is_empty())
        {
            let extra = if out.is_empty() { 0 } else { 1 };
            if used + extra + line.chars().count() > max_chars {
                break;
            }
            used += extra + line.chars().count();
            out.push(line);
        }
        return out.join("\n");
    }

    let mut sections = Vec::new();
    push_section_with_budget(
        &mut sections,
        format!("tool_output_lines: {}", lines.len()),
        max_chars,
    );

    let key_signals = lines
        .iter()
        .filter(|line| is_important_tool_line(line))
        .map(|line| tool_line_signature(line))
        .filter(|line| !line.is_empty())
        .fold(Vec::new(), |mut acc: Vec<String>, line| {
            push_unique_limited_global(&mut acc, line, 4);
            acc
        });
    if !key_signals.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("key_signals: {}", key_signals.join(" || ")),
            max_chars,
        );
    }

    let path_hints = lines
        .iter()
        .flat_map(|line| extract_path_like_tokens(line))
        .fold(Vec::new(), |mut acc: Vec<String>, token| {
            push_unique_limited_global(&mut acc, token, 4);
            acc
        });
    if !path_hints.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("paths: {}", path_hints.join(", ")),
            max_chars,
        );
    }

    let chunk_size = (lines.len() / 3).max(1);
    let mut chunk_summaries = Vec::new();
    for (chunk_index, chunk) in lines.chunks(chunk_size).take(3).enumerate() {
        let chunk_summary = summarize_tool_chunk(chunk_index + 1, chunk);
        if !chunk_summary.is_empty() {
            chunk_summaries.push(chunk_summary);
        }
    }
    if !chunk_summaries.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("chunks:\n- {}", chunk_summaries.join("\n- ")),
            max_chars,
        );
    }

    sections.join("\n")
}

fn push_section_with_budget(target: &mut Vec<String>, section: String, max_chars: usize) {
    if section.is_empty() {
        return;
    }
    let current = if target.is_empty() {
        0
    } else {
        target.join("\n").chars().count() + 1
    };
    if current + section.chars().count() <= max_chars {
        target.push(section);
        return;
    }
    if target.is_empty() {
        target.push(summarize_text(&section, max_chars));
    }
}

fn summarize_tool_chunk(chunk_index: usize, chunk: &[&str]) -> String {
    if chunk.is_empty() {
        return String::new();
    }
    let mut picks: Vec<String> = Vec::new();
    let first = tool_line_signature(chunk[0]);
    if !first.is_empty() {
        push_unique_limited_global(&mut picks, first, 4);
    }
    for line in chunk.iter().filter(|line| is_important_tool_line(line)).take(2) {
        let sig = tool_line_signature(line);
        if !sig.is_empty() {
            push_unique_limited_global(&mut picks, sig, 4);
        }
    }
    if let Some(last) = chunk.last() {
        let last = tool_line_signature(last);
        if !last.is_empty() {
            push_unique_limited_global(&mut picks, last, 4);
        }
    }
    if picks.is_empty() {
        return String::new();
    }
    format!("chunk_{chunk_index}: {}", picks.join(" | "))
}

fn tool_line_signature(line: &str) -> String {
    let normalized = normalize_whitespace(line);
    if normalized.is_empty() {
        return String::new();
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.len() <= 18 {
        return normalized;
    }

    let head = words.iter().take(12).copied().collect::<Vec<_>>().join(" ");
    let mut notable_tail = Vec::new();
    for word in words.iter().rev() {
        let token = word.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(ch, ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'')
        });
        if token.is_empty() {
            continue;
        }
        let looks_notable = token.contains('/')
            || token.contains('.')
            || token.chars().any(|ch| ch.is_ascii_digit())
            || looks_like_error_code(token);
        if looks_notable {
            push_unique_limited_global(&mut notable_tail, token.to_string(), 4);
        }
    }
    notable_tail.reverse();
    if notable_tail.is_empty() {
        return head;
    }
    format!("{head} | {}", notable_tail.join(" "))
}

fn is_important_tool_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("exception")
        || lower.contains("timeout")
        || lower.contains("not found")
        || lower.contains("traceback")
        || lower.contains("exit code")
        || lower.contains("warning")
        || lower.contains("completed")
        || lower.contains("success")
}

fn extract_path_like_tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in line.split_whitespace() {
        let token = raw.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(ch, ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'')
        });
        if token.len() > 160 || token.is_empty() {
            continue;
        }
        if token.starts_with("http://") || token.starts_with("https://") {
            continue;
        }
        let looks_like_path = token.contains('/')
            || [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift",
                ".c", ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ]
            .iter()
            .any(|suffix| token.ends_with(suffix));
        if looks_like_path {
            push_unique_limited_global(&mut out, token.to_string(), 8);
        }
    }
    out
}

fn looks_like_error_code(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 5
        && bytes[0] == b'E'
        && bytes[1..].iter().all(|byte| byte.is_ascii_digit())
}

fn push_unique_limited_global(target: &mut Vec<String>, value: String, max_items: usize) {
    if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items {
        return;
    }
    target.push(value);
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
            return extract_important_lines(text, 120);
        }
        summarize_text(&normalize_whitespace(text), 80)
    }

    fn extract_important_lines(text: &str, target_chars: usize) -> String {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return String::new();
        }
        let mut selected: Vec<&str> = Vec::new();
        let mut chars = 0usize;
        for line in &lines {
            let lowered = line.to_ascii_lowercase();
            let is_key = lowered.contains("error")
                || lowered.contains("failed")
                || lowered.contains("panic")
                || lowered.contains("exception")
                || lowered.contains("not found")
                || lowered.contains("timeout");
            if is_key || selected.is_empty() {
                if chars + line.trim().chars().count() + 2 > target_chars {
                    if selected.is_empty() {
                        let trimmed = line.trim();
                        selected.push(trimmed);
                    }
                    break;
                }
                selected.push(line.trim());
                chars += line.trim().chars().count() + 2;
            }
        }
        let result = selected.join("; ");
        if result.chars().count() <= target_chars {
            return result;
        }
        keep_ends_by_chars(&result, target_chars)
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
                && !turn.user_key.is_empty()
                && last.user_key == turn.user_key
            {
                last.count = last.count.saturating_add(turn.count.max(1));
                if last.topic_label.is_empty() && !turn.topic_label.is_empty() {
                    last.topic_label = turn.topic_label;
                    last.topic_key = turn.topic_key;
                }
                if !turn.assistant_final.is_empty()
                    && turn.assistant_final != last.assistant_final
                    && last.assistant_final.chars().count() < 200
                {
                    if last.assistant_final.is_empty() {
                        last.assistant_final = turn.assistant_final;
                    } else {
                        last.assistant_final = summarize_text(
                            &format!("{} / {}", last.assistant_final, turn.assistant_final),
                            250,
                        );
                    }
                }
                for name in turn.tool_names {
                    push_unique_limited(&mut last.tool_names, name, 6);
                }
                for h in turn.tool_highlights {
                    push_unique_limited(&mut last.tool_highlights, h, 3);
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
        line
    }

    fn render_known_tool_line(turn: &TurnSummary) -> Option<String> {
        if turn.tool_names.is_empty() {
            return None;
        }
        let mut line = String::new();
        line.push_str("- ");
        line.push_str(&turn.tool_names.join(", "));
        if !turn.topic_label.is_empty() {
            line.push_str(" @ ");
            line.push_str(&turn.topic_label);
        }
        let conclusion = if !turn.tool_highlights.is_empty() {
            turn.tool_highlights.join("；")
        } else {
            turn.assistant_final.clone()
        };
        if !conclusion.is_empty() {
            line.push_str(" => ");
            line.push_str(&conclusion);
        }
        Some(line)
    }

    fn push_line_with_budget(lines: &mut Vec<String>, mut line: String, max_chars: usize) -> bool {
        let line_chars = line.chars().count();
        if lines.is_empty() {
            if line_chars > max_chars {
                lines.push(summarize_text(&line, max_chars));
                return true;
            }
            lines.push(line);
            return true;
        }
        let current_len = lines.join("\n").chars().count();
        let remaining = max_chars.saturating_sub(current_len + 1);
        if remaining < 30 {
            return false;
        }
        if line_chars > remaining {
            line = summarize_text(&line, remaining);
        }
        if line.chars().count() <= remaining {
            lines.push(line);
            true
        } else {
            false
        }
    }

    let mut initial_goal = String::new();
    let mut pre_summary_lines: Vec<String> = Vec::new();
    let mut turns: Vec<TurnSummary> = Vec::new();
    let mut current = TurnSummary::default();

    for message in messages {
        let text = normalize_whitespace(&value_to_string(&message.content));
        match message.role.as_str() {
            role if is_system_like_role(role) => {
                let normalized = strip_summary_header(&text);
                if normalized.is_empty() || normalized.starts_with("self_note:") {
                    continue;
                }
                if normalized.contains("历史摘要") || normalized.contains("对话摘要") {
                    pre_summary_lines.push(format!(
                        "- 更早摘要: {}",
                        summarize_text(&normalized, 400)
                    ));
                    continue;
                }
                let normalized = summarize_text(&normalized, 400);
                if !normalized.is_empty() {
                    pre_summary_lines.push(format!("- 更早摘要: {normalized}"));
                }
            }
            "user" => {
                finalize_turn(&mut turns, &mut current);
                if initial_goal.is_empty() {
                    initial_goal = summarize_text(&text, 240);
                }
                current.user = summarize_text(&text, 200);
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
                    current.assistant_final = summarize_text(&text, 250);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&text) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        push_unique_limited(&mut current.tool_names, tool_call.function.name.clone(), 6);
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
                    push_unique_limited(&mut current.tool_highlights, h.clone(), 3);
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

    let recent_count = turns.len().min(3);
    let recent_turns: Vec<TurnSummary> = turns
        .iter()
        .rev()
        .take(recent_count)
        .rev()
        .cloned()
        .collect();

    let pending_tasks: Vec<String> = turns
        .iter()
        .rev()
        .take(2)
        .filter(|t| !t.user.is_empty() && t.assistant_final.is_empty())
        .map(|t| t.user.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let merged = merge_turns(turns);
    let mut known_tool_lines: Vec<String> = Vec::new();
    for t in &merged {
        if let Some(line) = render_known_tool_line(t)
            && !known_tool_lines.iter().any(|existing| existing == &line)
            && known_tool_lines.len() < 10
        {
            known_tool_lines.push(line);
        }
    }
    let reserved_tool_chars = if known_tool_lines.is_empty() {
        0
    } else {
        let tool_blob = format!("已知工具结论:\n{}", known_tool_lines.join("\n"));
        tool_blob.chars().count().min(max_chars / 3)
    };
    let body_budget = max_chars.saturating_sub(reserved_tool_chars).max(max_chars / 2);
    let mut lines: Vec<String> = Vec::new();
    if !initial_goal.is_empty()
        && !push_line_with_budget(&mut lines, format!("初始目标: {initial_goal}"), body_budget)
    {
        return summarize_text(&lines.join("\n"), max_chars);
    }
    for s in pre_summary_lines.into_iter().take(3) {
        if !push_line_with_budget(&mut lines, s, body_budget) {
            return summarize_text(&lines.join("\n"), max_chars);
        }
    }
    for t in &merged {
        if !push_line_with_budget(&mut lines, format!("- {}", render_line(t)), body_budget) {
            break;
        }
    }

    if !known_tool_lines.is_empty() {
        let _ = push_line_with_budget(&mut lines, "已知工具结论:".to_string(), max_chars);
        for line in known_tool_lines {
            if !push_line_with_budget(&mut lines, line, max_chars) {
                break;
            }
        }
    }

    if !recent_turns.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "当前工作:".to_string(), max_chars);
        for t in &recent_turns {
            let mut parts = Vec::new();
            if !t.topic_label.is_empty() {
                parts.push(format!("主题: {}", t.topic_label));
            }
            if !t.user.is_empty() {
                parts.push(format!("用户: {}", t.user));
            }
            if !t.assistant_final.is_empty() {
                parts.push(format!("助手: {}", t.assistant_final));
            }
            if !t.tool_names.is_empty() {
                parts.push(format!("工具: {}", t.tool_names.join(", ")));
            }
            if !t.tool_highlights.is_empty() {
                parts.push(format!("关键: {}", t.tool_highlights.join("；")));
            }
            let line = format!("- {}", parts.join(" | "));
            if !push_line_with_budget(&mut lines, summarize_text(&line, 600), max_chars) {
                break;
            }
        }
    }

    if !pending_tasks.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "待办任务:".to_string(), max_chars);
        for task in &pending_tasks {
            if !push_line_with_budget(
                &mut lines,
                format!("- {}", summarize_text(task, 300)),
                max_chars,
            ) {
                break;
            }
        }
    }

    summarize_text(&lines.join("\n"), max_chars)
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

fn summarize_text(text: &str, target_chars: usize) -> String {
    if target_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= target_chars {
        return text.to_string();
    }

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.len() <= 1 {
        return keep_ends_by_chars(text, target_chars);
    }

    let mut selected: Vec<&str> = Vec::new();
    let mut selected_chars = 0usize;

    let head_count = (lines.len().min(3)).min(target_chars / 20);
    for line in lines.iter().take(head_count) {
        if selected_chars + line.chars().count() + 1 > target_chars {
            break;
        }
        selected.push(line);
        selected_chars += line.chars().count() + 1;
    }

    let tail_budget = target_chars.saturating_sub(selected_chars).max(target_chars / 3);
    let tail_count = lines.len().min(3).min(tail_budget / 20);
    let tail_start = lines.len().saturating_sub(tail_count);
    if tail_start > head_count {
        for line in lines.iter().skip(tail_start) {
            if selected_chars + line.chars().count() + 1 > target_chars {
                break;
            }
            selected.push(line);
            selected_chars += line.chars().count() + 1;
        }
    }

    if selected.is_empty() {
        return keep_ends_by_chars(text, target_chars);
    }

    let result = selected.join("\n");
    if result.chars().count() <= target_chars {
        return result;
    }

    keep_ends_by_chars(&result, target_chars)
}

fn keep_ends_by_chars(text: &str, target_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= target_chars {
        return text.to_string();
    }
    let head_budget = target_chars * 3 / 5;
    let tail_budget = target_chars - head_budget - 1;
    let head: String = text.chars().take(head_budget).collect();
    let tail: String = text.chars().skip(char_count.saturating_sub(tail_budget)).collect();
    format!("{}…{}", head, tail)
}

fn first_tool_call_group(messages: &[Message]) -> Option<Vec<usize>> {
    let assistant_idx = messages.iter().position(|m| {
        m.role == "assistant" && m.tool_calls.as_ref().map_or(false, |tc| !tc.is_empty())
    })?;
    let tool_call_ids: Vec<String> = messages[assistant_idx]
        .tool_calls
        .as_ref()
        .unwrap()
        .iter()
        .map(|tc| tc.id.clone())
        .collect();
    let mut group = vec![assistant_idx];
    for (i, m) in messages.iter().enumerate() {
        if m.role == "tool" {
            if let Some(ref id) = m.tool_call_id {
                if tool_call_ids.contains(id) {
                    group.push(i);
                }
            }
        }
    }
    Some(group)
}

fn first_trim_candidate(messages: &[Message]) -> Option<usize> {
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && is_summary_message(message) {
            continue;
        }
        return Some(index);
    }
    None
}

/// 渐进式卸载：把一个 (assistant tool_calls + 配套 tool 结果) 整组折叠成单条
/// `internal_note`，保留"工具列表 + 每个工具结果首句"，便于后续轮次知道
/// 之前发生过什么、避免重复劳动；同时大幅压缩 token 占用。
fn fold_tool_call_group_to_stub(messages: &[Message], group: &[usize]) -> Option<Message> {
    if group.is_empty() {
        return None;
    }
    let assistant_idx = group[0];
    let assistant = messages.get(assistant_idx)?;
    let tool_calls = assistant.tool_calls.as_ref()?;
    if tool_calls.is_empty() {
        return None;
    }

    let mut lines = Vec::with_capacity(tool_calls.len() + 1);
    lines.push(format!(
        "compressed_tool_round: {} tool calls (folded for context budget)",
        tool_calls.len()
    ));

    for tc in tool_calls.iter().take(8) {
        let result_text = group
            .iter()
            .skip(1)
            .find_map(|idx| {
                let m = messages.get(*idx)?;
                if m.tool_call_id.as_deref() == Some(tc.id.as_str()) {
                    Some(value_to_string(&m.content))
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let first_meaningful = result_text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("");
        let one_liner = truncate_to_chars(&normalize_whitespace(first_meaningful), 160);
        lines.push(format!("- {} => {}", tc.function.name, one_liner));
    }
    if tool_calls.len() > 8 {
        lines.push(format!("- ... ({} more tools omitted)", tool_calls.len() - 8));
    }

    Some(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(lines.join("\n")),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

fn is_summary_message(message: &Message) -> bool {
    if !is_system_like_role(&message.role) {
        return false;
    }
    let text = value_to_string(&message.content);
    text.starts_with("对话摘要（自动压缩") || text.starts_with("历史摘要（自动压缩")
}

const KEEP_RECENT_TOOL_MESSAGES: usize = 6;

fn tool_message_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "tool").then_some(i))
        .collect()
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
    let mut prev_signature = String::new();
    for m in messages.drain(..) {
        let text = value_to_string(&m.content);
        // 完全相等：直接丢弃
        if m.role == prev_role && text == prev_content {
            continue;
        }
        // 模糊去重：仅对 tool 角色启用，避免误伤 assistant/user 中观感相近但实质不同的回复。
        // 同 role 且整段 text 的 tool_line_signature 相同（去掉空白噪音 + 关键 token 一致）才丢弃。
        let signature = if m.role == "tool" {
            tool_line_signature(&text)
        } else {
            String::new()
        };
        if m.role == "tool"
            && !signature.is_empty()
            && m.role == prev_role
            && signature == prev_signature
        {
            continue;
        }
        prev_role = m.role.clone();
        prev_content = text;
        prev_signature = signature;
        out.push(m);
    }
    *messages = out;
}

/// 只保留最近一条 assistant 的 reasoning_content。
/// 较老的 reasoning chain 对后续 turn 决策几乎没有帮助，但每条都要回传给厂商
/// 才能避免 400 invalid_request_error。把旧 reasoning_content 置 None 后，
/// 厂商侧并不强制要求历史 reasoning 全程保留（OpenAI 仅要求与最近一次 tool_call
/// 同回合的 reasoning 配对）。
fn keep_only_recent_reasoning_content(messages: &mut [Message]) {
    let last_with_reasoning = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| m.role == "assistant" && m.reasoning_content.is_some())
        .map(|(idx, _)| idx);
    let Some(keep_idx) = last_with_reasoning else {
        return;
    };
    for (idx, m) in messages.iter_mut().enumerate() {
        if idx == keep_idx {
            continue;
        }
        if m.role == "assistant" && m.reasoning_content.is_some() {
            m.reasoning_content = None;
        }
    }
}

/// 跨轮 tool 结果去重：同一 (tool_name, normalized_args) 在历史中出现多次时，
/// 把较早的 tool 结果替换为单行 stub（保留 tool_call_id 以维持 OpenAI tool-calls 协议正确性）。
/// 仅压缩内容，不删除消息，避免 assistant tool_calls 与 tool 响应的配对断裂。
/// 最近 KEEP_RECENT_TOOL_MESSAGES 条 tool 消息一律保留全文。
fn dedup_repeated_tool_results(messages: &mut [Message]) {
    use std::collections::HashMap;

    // 收集 (tool_name, args_signature) → 出现次数与索引
    // 通过 assistant.tool_calls 关联 tool_call_id → (name, args)
    let mut id_to_signature: HashMap<String, (String, String)> = HashMap::new();
    for message in messages.iter() {
        if let Some(tool_calls) = &message.tool_calls {
            for tc in tool_calls {
                let args_norm = serde_json::from_str::<Value>(&tc.function.arguments)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| tc.function.arguments.clone());
                id_to_signature.insert(
                    tc.id.clone(),
                    (tc.function.name.clone(), args_norm),
                );
            }
        }
    }

    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "tool").then_some(i))
        .collect();
    if tool_indices.len() <= KEEP_RECENT_TOOL_MESSAGES {
        return;
    }
    let protected_from = tool_indices.len().saturating_sub(KEEP_RECENT_TOOL_MESSAGES);

    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    for (rank, &idx) in tool_indices.iter().enumerate() {
        let signature = match messages[idx]
            .tool_call_id
            .as_ref()
            .and_then(|id| id_to_signature.get(id))
        {
            Some(sig) => sig.clone(),
            None => continue,
        };
        let count = seen.entry(signature.clone()).or_insert(0);
        *count += 1;
        if rank >= protected_from {
            continue;
        }
        // 保留首次出现，把后续重复的旧 tool 结果折叠为 stub
        if *count > 1 {
            let stub = format!(
                "[deduped: identical {} call earlier in this conversation; full result preserved at first occurrence]",
                signature.0
            );
            messages[idx].content = Value::String(stub);
        }
    }
}
