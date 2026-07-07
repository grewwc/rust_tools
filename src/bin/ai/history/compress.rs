use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::{request, types::App};

use super::types::{
    MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start,
};

const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;
/// 压缩兜底（first_trim_candidate）时保护最近 user 起始尾窗的动态上下限。
/// 小上下文优先保留 3 轮提升多阶段任务连续性；超大上下文回退到 2 轮控预算。
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN: usize = 2;
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX: usize = 3;
/// 当上下文字符数不超过该阈值时，优先保留 3 轮 user。
const KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS: usize = 48_000;

fn keep_recent_user_turns_when_trimming(messages: &[Message]) -> usize {
    if messages_total_chars(messages) <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX
    } else {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN
    }
}

/// 暴露给同 crate 的常量访问器，避免在 mod.rs 中复制阈值数字。
pub(in crate::ai) fn persisted_history_keep_recent_turns() -> usize {
    PERSISTED_HISTORY_KEEP_RECENT_TURNS
}

/// messages 数组中保留的 self_note 最大条数。
/// self_note 已经被持久化到 MemoryStore（`memory_store::AgentMemoryEntry`），
/// messages 里那条仅是同 turn 内被 LLM 看到的"冗余 inline 副本"。
/// 长 session 累计上千 turn 时这些 inline 副本会单调膨胀，需要滑窗剪裁。
const MAX_SELF_NOTES_IN_MESSAGES: usize = 8;

/// 仅保留最近 `keep_recent` 条 internal_note 中的 `self_note:` 条目。
/// 其他 internal_note（如 cache 提示、loop-breaker、历史摘要）不在剪裁范围。
fn trim_self_notes_to_recent(messages: Vec<Message>, keep_recent: usize) -> Vec<Message> {
    let total_self_notes = messages.iter().filter(|m| is_self_note_message(m)).count();
    if total_self_notes <= keep_recent {
        return messages;
    }
    let drop_count = total_self_notes - keep_recent;
    let mut dropped = 0usize;
    messages
        .into_iter()
        .filter(|m| {
            if is_self_note_message(m) && dropped < drop_count {
                dropped += 1;
                false
            } else {
                true
            }
        })
        .collect()
}

fn is_self_note_message(m: &Message) -> bool {
    if m.role != ROLE_INTERNAL_NOTE {
        return false;
    }
    let s = value_to_string(&m.content);
    s.trim_start().starts_with("self_note:")
}
const PERSISTED_HISTORY_SUMMARY_MAX_CHARS: usize = 8_000;
const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";
const PRESERVED_TOOL_OVERFLOW_DIR: &str = "tool-overflow-compressed";
const PRESERVED_USER_OVERFLOW_DIR: &str = "user-overflow-preserved";
const PRESERVED_IMAGE_OVERFLOW_DIR: &str = "image-overflow-preserved";
const PRESERVED_CONTENT_STUB_PREFIX: &str = "[[PRESERVED_CONTENT_STUB_V1]]";
const USER_OVERFLOW_SPILL_MIN_CHARS: usize = 1_024;
const IMAGE_OVERFLOW_SPILL_MIN_CHARS: usize = 512;

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
            // 仅在归档文件尚未存在时写入头部说明；后续调用走 append 模式追加新批次。
            // 每个新批次再加一个分隔行，方便人工/工具分块读取。
            if !self.path.exists() {
                self.buffer.push_str(
                    "# 溢出对话历史\n\n以下内容因超出上下文窗口而被移出，模型可使用 read_file 工具读取此文件回顾。\n\n---\n\n",
                );
            } else {
                self.buffer.push_str("\n---\n\n");
            }
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
        // append 模式：避免每次压缩都把之前归档的更早历史覆盖丢失。
        // 之前用 File::create 会清空文件，导致同一会话经历多轮压缩后只剩
        // 最后一次 flush 的内容，长期记忆退化为短期记忆。
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
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
    out.push_str("优先执行工具: read_file\n参数: file_path=\"");
    out.push_str(file_path);
    out.push_str("\", offset=1, limit=200)\n");
    out.push_str("若当前问题依赖前文细节、最初目标、之前决定、旧报错或更早工具输出，请先分段读取该归档；只有在已经定位到相关位置后，再改用更精确的行范围读取。\n");
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

    // 在做大块压缩前先剪 self_note 滑动上限，避免上千轮 turn 累积的
    // self_note（已写入 MemoryStore，messages 里那条仅是冗余备份）
    // 单调膨胀。MemoryStore 仍保留全部记录。
    let messages = trim_self_notes_to_recent(messages, MAX_SELF_NOTES_IN_MESSAGES);

    let keep_last = keep_last.min(messages.len());
    if keep_last == 0 {
        return shrink_messages_to_fit_with_summary(
            messages,
            max_chars,
            summary_max_chars,
            overflow_dir.as_deref(),
        );
    }

    let split_at = retained_turn_start(&messages, keep_last);
    let (older, recent) = messages.split_at(split_at);
    if older.is_empty() {
        return shrink_messages_to_fit_with_summary(
            recent.to_vec(),
            max_chars,
            summary_max_chars,
            overflow_dir.as_deref(),
        );
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
    let user_turns = messages
        .iter()
        .filter(|message| message.role == "user")
        .count();
    if user_turns <= MAX_HISTORY_TURNS {
        return messages;
    }

    let keep_recent_turns = PERSISTED_HISTORY_KEEP_RECENT_TURNS.min(MAX_HISTORY_TURNS - 1);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return messages;
    }

    let summary =
        build_persisted_summary_text(&messages[..split_at], PERSISTED_HISTORY_SUMMARY_MAX_CHARS);
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
    let user_turns = messages
        .iter()
        .filter(|message| message.role == "user")
        .count();
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

fn shrink_messages_to_fit(
    mut messages: Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }

    if messages.is_empty() {
        return Vec::new();
    }

    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_MESSAGES, overflow_dir);
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

    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_MESSAGES, overflow_dir);
    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    dedup_repeated_tool_results(&mut messages);

    // 先无条件外溢体量过大的旧 user/图片消息（最新一轮保护尾窗除外）。
    // 图片在预算里只按名义成本计费，单张大图不再触发超预算循环，因此必须
    // 在预算判断之前就把它们零压缩搬到文件，避免每轮请求都携带完整 base64。
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir);
    }

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }
    let had_leading_summary = messages.first().map(is_summary_message).unwrap_or(false);
    let mut dropped: Vec<Message> = Vec::new();

    while messages_total_chars(&messages) > max_chars {
        if let Some(group) = first_tool_call_group(&messages) {
            // 与 shrink_messages_to_fit 保持一致：先尝试折叠成单条 stub，
            // 让模型仍能"看见"早期发生过哪些工具调用、以什么结果收尾，
            // 避免后续轮次完全失忆而重复工作。折叠仍超额或无法构造 stub
            // 时，才把整组移入 dropped 由 OverflowSink 归档。
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
        if let Some(dir) = overflow_dir
            && try_spill_preserved_message_to_stub(&mut messages, dir)
        {
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
                    let body_byte_budget =
                        max_chars.saturating_sub(used).saturating_sub(header_bytes);
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
                let fallback_goal =
                    dropped
                        .iter()
                        .find(|message| message.role == "user")
                        .map(|message| {
                            summarize_text(
                                &normalize_whitespace(&value_to_string(&message.content)),
                                160,
                            )
                        });
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

    // 找到第一个可被截断的消息：
    // - 跳过 system（agent 指令不能被截断）；
    // - 跳过 user（用户原文零压缩）；
    // - 跳过图片消息（图片引用零压缩）。
    let target_idx = messages.iter().position(|m| {
        m.role != "system" && m.role != "user" && !message_contains_image(&m.content)
    });
    let Some(target_idx) = target_idx else {
        return; // 全是 system，没有可截断的目标
    };

    let others_chars: usize = messages
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx)
        .map(|(_, m)| value_len_chars(&m.content))
        .sum();
    let remaining_chars = max_chars.saturating_sub(others_chars).max(50);

    let target = &mut messages[target_idx];
    let text = value_to_string(&target.content);
    let truncated = truncate_to_chars(&text, remaining_chars);
    target.content = Value::String(truncated);
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
    overflow_dir: Option<&Path>,
) -> (Vec<Message>, usize, usize) {
    let before = messages_total_chars(&messages);
    if before <= soft_threshold {
        return (messages, before, before);
    }
    let mut out = messages;
    // 0. 清理过期 reasoning_content：单 turn 内 LLM 多次返回的 reasoning chain
    //    对后续决策无益，但部分厂商要求历史 reasoning 与 tool_calls 配对。
    //    只保留最后一条 assistant 的 reasoning_content，其余置 None。
    keep_only_recent_reasoning_content(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 1. 同 signature 工具结果去重
    dedup_repeated_tool_results(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. 远端结构化裁剪：tool 结果中段按行折叠到 480 字/条，最近 6 条保留全文。
    //    传入 overflow_dir 后，read_file/grep 等「不可压缩」工具的大输出会被
    //    零压缩外溢到会话文件并留 head+tail 预览 stub（与跨 turn 压缩一致），
    //    既释放上下文体积又不丢信息——模型可按 stub 里的 file_path 重新 read_file。
    prepare_tool_messages_structured(&mut out, 480, KEEP_RECENT_TOOL_MESSAGES, overflow_dir);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 3. 仍超额：用 shrink_messages_to_fit 走"折叠 tool group + 整体兜底"
    out = shrink_messages_to_fit(out, soft_threshold, overflow_dir);
    let after = messages_total_chars(&out);
    (out, before, after)
}

/// LLM 摘要"有效压缩"的最小净下降量（字符）。低于此值视为 no-op，
/// `did_summarize` 返回 false，避免调用方误存游标抑制后续重试。
/// 取 `summary_max_chars` 同量级：若净下降还不如注入的摘要文本大，
/// 说明压缩器空转（典型症状："295K 压到 294K 就停了"）。
const MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS: usize = 4_000;

/// Path C 兜底：对尾窗内单个超大非 system 消息做 head+tail 截断时的单条上限。
/// 仅在渐进式折叠后仍超 `hard_target` 时触发——宁可截断也不能让模型 4xx。
const PATH_C_PER_MSG_CAP: usize = 8_000;

/// Mid-turn LLM 摘要兜底：无损/弱损管线之后仍超阈值时调用。三条互补路径：
///   - Path A（跨轮摘要）：最近 `keep_recent_turns` 个 user 轮之前若还有对话，
///     调 LLM 摘要器把那段压成单条 `internal_note` 注入到尾窗前；同时对尾窗
///     内部较早的工具组做折叠，避免"臃肿全在最近一轮"时压不动。
///   - Path B+C（渐进式折叠）：从 `keep_recent=4` 开始（等价于原 Path B），
///     逐步缩小保护窗口到 2→1→0，直到有效压缩或降至 `hard_target` 以下。
///     解决"臃肿全在保护尾窗内、早期历史已压无可压"时压缩器空转的问题。
///   - Path C 兜底（per-message 截断）：渐进式折叠后仍超 `hard_target` 时，
///     对尾窗内单个超大非 system 消息做 head+tail 截断。这是绝对最后手段。
/// 头部所有 system / internal_note（agent 指令、工具列表、全局指引）始终原样保留。
/// 返回 `(messages_after, before, after, did_summarize)`；`did_summarize` 仅在
/// 净下降 ≥ [`MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS`] 时为 true，避免调用方误存
/// 游标抑制后续重试。
pub(in crate::ai) async fn mid_turn_llm_summarize(
    app: &App,
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
    hard_target: usize,
) -> (Vec<Message>, usize, usize, bool) {
    let before = messages_total_chars(&messages);
    // best 追踪迄今为止体积最小的结果；None 表示仍使用原始 messages。
    let mut best: Option<Vec<Message>> = None;
    let mut best_after = before;

    // === Path A：跨轮 LLM 摘要 ===
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at > 0 && split_at < messages.len() {
        // 保留头部前缀连续的 system-like 消息（agent 指令等），只摘要其后的对话
        // 区段。早期版本直接丢弃 messages[0] 的 system prompt，会让模型立刻失去
        // agent 行为指令，表现为"压缩后回复戛然而止 / 极短 / 跑偏"。
        let preserved_system_end = messages[..split_at]
            .iter()
            .position(|m| !is_system_like_role(&m.role))
            .unwrap_or(split_at);
        let earlier = &messages[preserved_system_end..split_at];
        let has_dialog = earlier
            .iter()
            .any(|m| m.role == "user" || m.role == "assistant");
        if has_dialog {
            let summary =
                build_persisted_summary_text_with_app(app, earlier, summary_max_chars).await;
            if !summary.trim().is_empty() {
                let mut out =
                    Vec::with_capacity(preserved_system_end + 1 + (messages.len() - split_at));
                // 1. 头部 system / internal_note（agent 指令等）原样保留
                out.extend_from_slice(&messages[..preserved_system_end]);
                // 2. 摘要作为 internal_note 注入（normalize_messages_for_request 会把
                //    它归类成 Summary heading 并合并进 system 消息）
                out.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: Value::String(format!(
                        "[mid-turn-summary] 早期工具调用与对话已被 LLM 摘要：\n{summary}"
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                // 3. 尾窗：保留 user 逐字 + 最近若干工具组逐字，更早工具组折叠成 stub
                let (tail, _) = fold_early_tool_groups(
                    &messages[split_at..],
                    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS,
                );
                out.extend(tail);
                let after = messages_total_chars(&out);
                if after < best_after {
                    best = Some(out);
                    best_after = after;
                }
                // 有效压缩且达标 → 直接返回
                if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS
                    && best_after <= hard_target
                {
                    return (best.unwrap(), before, best_after, true);
                }
            }
        }
    }

    // === Path B+C：渐进式工具组折叠 ===
    // 从 keep_recent=4（等价于原 Path B）开始，逐步缩小保护窗口到 2→1→0，
    // 直到有效压缩或降至 hard_target 以下。解决"臃肿全在保护尾窗内"时空转。
    // 在 best（Path A 结果或原始 messages）上链式折叠：已折叠的组变成 stub
    //（internal_note），不会被 fold_early_tool_groups 再次匹配，因此每次迭代
    // 只会折叠上一轮保留的组，逐步释放保护尾窗。
    for &keep_recent in &[
        MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS,
        2,
        1,
        0,
    ] {
        if best_after <= hard_target {
            break;
        }
        let current = best.as_ref().unwrap_or(&messages);
        let (folded, folded_groups) = fold_early_tool_groups(current, keep_recent);
        if folded_groups == 0 {
            continue;
        }
        let after = messages_total_chars(&folded);
        if after < best_after {
            best = Some(folded);
            best_after = after;
        }
    }

    // 有效压缩 → 返回（无论是否达标 hard_target，只要净下降够大就算成功）
    if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS {
        return (best.unwrap_or(messages), before, best_after, true);
    }

    // === Path C 兜底：per-message 截断 ===
    // 渐进式折叠后仍超 hard_target 且未达有效压缩：对尾窗内单个超大非 system
    // 消息做 head+tail 截断。典型场景：最近一轮对话本身就很大（巨型 user 消息
    // 或大量近期工具结果），早期历史已压无可压，臃肿全在保护尾窗内。
    // 这是绝对最后手段——宁可截断也不能让模型 4xx。
    if best_after > hard_target {
        let current = best.unwrap_or(messages);
        let capped = cap_oversized_non_system_messages(current, PATH_C_PER_MSG_CAP);
        let after = messages_total_chars(&capped);
        let savings = before.saturating_sub(after);
        return (capped, before, after, savings >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS);
    }

    // 所有路径均未达到有效压缩：返回最佳结果，did_summarize=false
    let result = best.unwrap_or(messages);
    (result, before, best_after, false)
}

/// Path C 兜底：对序列中单个超大非 system 消息做 head+tail 截断。
/// 仅在渐进式折叠后仍超 `hard_target` 时由 [`mid_turn_llm_summarize`] 调用。
/// system / agent 指令永不截断；图片按名义计费（≤ PATH_C_PER_MSG_CAP）天然跳过。
fn cap_oversized_non_system_messages(
    mut messages: Vec<Message>,
    per_msg_cap: usize,
) -> Vec<Message> {
    for message in &mut messages {
        if is_system_like_role(&message.role) {
            continue;
        }
        let chars = value_len_chars(&message.content);
        if chars <= per_msg_cap {
            continue;
        }
        let text = value_to_string(&message.content);
        let capped = keep_ends_by_chars(&text, per_msg_cap);
        message.content = Value::String(format!(
            "[context-overflow-truncated] 原文 {chars} 字符已截断为 head+tail 预览：\n{capped}"
        ));
    }
    messages
}

/// 单张图片在「字符预算」里的名义计费。
///
/// 视觉模型把一张图 tokenize 成几百~一两千 token，与其 base64 文本长度
/// （动辄数十万字符）完全脱钩。历史上 `value_len_chars` 直接按 base64 文本
/// 长度计费，导致**一张大图就把整个上下文预算吃光**：`messages_total_chars`
/// 暴涨到远超 max_chars / soft_threshold，压缩管线于是每轮都把 agent 自己的
/// 工具结果（工作记忆）挤出窗口 —— 单 turn 内表现为「失忆 + 反复重复之前的
/// 探索/计划」。这里给图片一个固定名义成本，让预算回归文本主导。
/// 注意：这只改预算**计量**，不改消息内容本身（图片仍零压缩原样发送）。
const IMAGE_BUDGET_CHARS: usize = 1_024;

/// 判断裸字符串是否是内联图片 data URL（极少数 provider 会把图片放进纯字符串）。
fn is_inline_image_data_url(s: &str) -> bool {
    let t = s.trim_start();
    t.starts_with("data:image/") && t.contains(";base64,")
}

/// 计算多模态 content 数组中单个 part 的预算字符数：图片按名义成本计费，
/// 文本按其实际字符数计费。
fn content_part_budget_chars(item: &Value) -> usize {
    let is_image = item.get("type").and_then(|t| t.as_str()) == Some("image_url")
        || item.get("image_url").is_some();
    if is_image {
        return IMAGE_BUDGET_CHARS;
    }
    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
        return text.chars().count();
    }
    item.to_string().chars().count()
}

/// 返回 Value 内容的「预算字符数」（Unicode scalar 数）。
/// 历史上这里返回的是 byte length，导致中文/emoji 场景下字符预算被高估 ~3 倍：
/// 例如 36K 字符的软阈值在中文 turn 下会被 12K 字符就误触发，反复跑压缩管线。
/// 现在统一按 `chars().count()` 计量，与外层 `cap_chars`、`max_chars`
/// 阈值的命名保持一致。图片 part 按 [`IMAGE_BUDGET_CHARS`] 名义计费，避免
/// base64 文本长度污染预算（见该常量文档）。
fn value_len_chars(v: &Value) -> usize {
    if let Some(s) = v.as_str() {
        if is_inline_image_data_url(s) {
            return IMAGE_BUDGET_CHARS;
        }
        return s.chars().count();
    }
    if let Some(arr) = v.as_array() {
        return arr.iter().map(content_part_budget_chars).sum();
    }
    v.to_string().chars().count()
}

pub(in crate::ai) fn value_to_string(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    // 多模态消息（JSON 数组）：只提取文本部分，丢弃图片 base64 数据，
    // 避免生成摘要/标题时把巨大的 base64 内容喂给模型或显示给用户。
    if let Some(arr) = v.as_array() {
        let mut text_parts = Vec::new();
        let mut has_image = false;
        for item in arr {
            if let Some(obj) = item.as_object() {
                let item_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match item_type {
                    "text" => {
                        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                            let trimmed = t.trim();
                            if !trimmed.is_empty() {
                                text_parts.push(trimmed.to_string());
                            }
                        }
                    }
                    "image_url" => has_image = true,
                    _ => {}
                }
            }
        }
        if text_parts.is_empty() && has_image {
            return "[图片]".to_string();
        }
        return text_parts.join(" ");
    }
    v.to_string()
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

fn automatic_summary_body(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    for prefix in [
        "历史摘要（自动压缩，以下为更早对话的简短语义）：",
        "对话摘要（自动压缩，以下为早期对话要点）：",
        "长期记忆摘要（压缩保留）:",
        "长期记忆摘要（压缩保留）：",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(rest.trim());
        }
    }

    if let Some(rest) = trimmed.strip_prefix("[mid-turn-summary]") {
        let rest = rest
            .trim_start()
            .strip_prefix("早期工具调用与对话已被 LLM 摘要：")
            .unwrap_or_else(|| rest.trim_start());
        return Some(rest.trim());
    }

    None
}

fn strip_nested_prior_summary_prefixes(text: &str) -> String {
    let mut current = normalize_whitespace(text);
    for _ in 0..8 {
        let trimmed = current.trim_start();
        let rest = trimmed
            .strip_prefix("- 更早摘要:")
            .or_else(|| trimmed.strip_prefix("更早摘要:"))
            .or_else(|| trimmed.strip_prefix("- 更早摘要："))
            .or_else(|| trimmed.strip_prefix("更早摘要："));
        let Some(rest) = rest else {
            break;
        };
        current = normalize_whitespace(rest);
    }
    current
}

async fn build_persisted_summary_text_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> String {
    let mut prepared = messages.to_vec();
    prepare_tool_messages_structured(&mut prepared, 360, KEEP_RECENT_TOOL_MESSAGES, None);
    redact_images_except_last(&mut prepared, 0);
    dedup_adjacent(&mut prepared);
    normalize_internal_notes_for_summary_model(&mut prepared);

    if let Some(summary) = request::summarize_history_via_model(app, &prepared, max_chars).await {
        let summary = normalize_whitespace(&summary);
        if !summary.is_empty() {
            return summary;
        }
    }

    build_persisted_summary_text(messages, max_chars)
}

fn normalize_internal_notes_for_summary_model(messages: &mut Vec<Message>) {
    let mut out = Vec::with_capacity(messages.len());
    let mut seen_auto_summary = false;

    for mut message in messages.drain(..) {
        if message.role == ROLE_INTERNAL_NOTE {
            let text = value_to_string(&message.content);
            if let Some(body) = automatic_summary_body(&text) {
                if seen_auto_summary {
                    continue;
                }
                let body = strip_nested_prior_summary_prefixes(body);
                if !body.is_empty() {
                    message.content = Value::String(format!(
                        "已有历史摘要（供本次压缩吸收，不要逐字复制）：\n{}",
                        summarize_text(&body, 2_000)
                    ));
                    out.push(message);
                    seen_auto_summary = true;
                }
                continue;
            }

            // 普通 internal_note 多为过程性提示、cache/loop 状态或 self_note 的
            // inline 副本。它们不应被当成长期历史事实交给摘要模型反复吸收。
            continue;
        }
        out.push(message);
    }

    *messages = out;
}

fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent: usize,
    overflow_dir: Option<&Path>,
) {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let indices = tool_message_indices(messages);
    let protect_from = indices.len().saturating_sub(keep_recent);
    for (rank, &idx) in indices.iter().enumerate() {
        let message = &mut messages[idx];
        let text = value_to_string(&message.content);
        if text.trim().is_empty() {
            continue;
        }

        let tool_name = message
            .tool_call_id
            .as_deref()
            .and_then(|id| id_to_tool_name.get(id))
            .map(|s| s.as_str());
        if let Some(name) = tool_name
            && is_non_compressible_tool(name)
        {
            if text.chars().count() > max_chars_per_msg
                && let Some(path) = overflow_dir
                    .and_then(|dir| write_preserved_tool_overflow_file(dir, name, &text))
            {
                message.content =
                    Value::String(build_preserved_tool_overflow_stub(&path, name, &text));
            }
            continue;
        }

        if rank >= protect_from {
            // 最近 keep_recent 条普通工具结果仍保留全文，避免误伤近端上下文。
            continue;
        }

        let summary = structured_tool_output_summary(&text, max_chars_per_msg);
        if !summary.is_empty() && summary != text {
            message.content = Value::String(summary);
        }
    }
}

fn build_tool_call_name_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.name.clone());
        }
    }
    out
}

/// 「读取/检索」类工具的输出零压缩（不行裁剪、不去重折叠、不整组删除），
/// 超阈值时只做"零压缩外溢到会话文件 + 留指针 stub"。这类输出复现代价高，
/// 一旦被压掉模型就会反复重跑同一次检索（典型失忆/原地打转症状）。
///
/// 注意：这里的名字必须与本 agent **实际注册**的工具名一致
/// （见 `src/bin/ai/tools/`）。早期版本误用了 VS Code Copilot 的工具名
/// （`file_search` / `semantic_search` / `fetch_webpage` / `read_page` /
/// `read_notebook_cell_output`）——这些工具在本 agent 里根本不存在，导致
/// `code_search` / `search_files` / `web_search` / `web_fetch` / `text_grep`
/// 等真正昂贵的检索结果统统被当作可压缩内容裁掉，引发重复检索。
fn is_non_compressible_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "read_file_lines"
            | "find_path"
            | "text_grep"
            | "search_files"
            | "code_search"
            | "web_search"
            | "web_fetch"
    )
}

fn write_preserved_tool_overflow_file(
    overflow_dir: &Path,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = tool_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let file_name = format!(
        "{}-{}-{}.txt",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        safe_tool,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(file_name);
    std::fs::write(&path, content).ok()?;
    Some(path)
}

fn build_preserved_tool_overflow_stub(path: &Path, tool_name: &str, full_content: &str) -> String {
    // 仍把全文外溢到磁盘以控制上下文体积，但在 stub 内保留 head+tail 预览，
    // 让后续 turn 拥有"召回锚点"——模型据此判断是否真的需要重新 read_file，
    // 避免早期读到的代码被搬走后出现"失忆/反复重读"。
    let preview = build_overflow_content_preview(full_content);
    format!(
        "Output preserved for non-compressible tool `{tool_name}`. Full result moved to session temp file:\n- file_path: {}\n- use read_file to inspect exact content.\n{preview}",
        path.display()
    )
}

/// 为外溢内容生成 head+tail 预览。短内容直接全量保留；长内容保留前后各若干行，
/// 中间用占位行折叠，并标注省略的行数。
fn build_overflow_content_preview(content: &str) -> String {
    const HEAD_LINES: usize = 8;
    const TAIL_LINES: usize = 4;
    const MAX_LINE_CHARS: usize = 200;

    let truncate_line = |line: &str| -> String {
        if line.chars().count() > MAX_LINE_CHARS {
            let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
            format!("{kept} …")
        } else {
            line.to_string()
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let mut out = String::from("Preview (for recall; not exhaustive):\n");
    if total <= HEAD_LINES + TAIL_LINES {
        for line in &lines {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    } else {
        for line in &lines[..HEAD_LINES] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
        out.push_str(&format!(
            "... [{} line(s) omitted; read the file above for full content] ...\n",
            total - HEAD_LINES - TAIL_LINES
        ));
        for line in &lines[total - TAIL_LINES..] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

fn is_preserved_user_or_image_stub(text: &str) -> bool {
    parse_preserved_message_stub(text).is_some()
}

fn parse_preserved_message_stub(text: &str) -> Option<(String, String)> {
    let payload = text.strip_prefix(PRESERVED_CONTENT_STUB_PREFIX)?;
    let value = serde_json::from_str::<Value>(payload).ok()?;
    let kind = value.get("kind")?.as_str()?.to_string();
    let file_path = value.get("file_path")?.as_str()?.to_string();
    if (kind == "user" || kind == "image") && !file_path.is_empty() {
        Some((kind, file_path))
    } else {
        None
    }
}

fn first_preserved_content_spill_candidate(messages: &[Message]) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for (idx, message) in messages.iter().enumerate() {
        if idx >= protected_tail_start {
            break;
        }
        if is_system_like_role(&message.role) || message.role == "tool" {
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            continue;
        }

        let text = value_to_string(&message.content);
        if is_preserved_user_or_image_stub(&text) {
            continue;
        }

        // value_to_string 会把图片 base64 折叠成 "[图片]"，无法反映真实体量。
        // 对图片消息改用原始 content 的序列化长度判断是否需要外溢，与「把大图
        // 搬到会话临时文件」的意图一致；普通文本消息仍按 value_to_string 计费。
        let char_count = if message_contains_image(&message.content) {
            message.content.to_string().chars().count()
        } else {
            text.chars().count()
        };
        if message_contains_image(&message.content) && char_count >= IMAGE_OVERFLOW_SPILL_MIN_CHARS
        {
            return Some(idx);
        }
        if message.role == "user" && char_count >= USER_OVERFLOW_SPILL_MIN_CHARS {
            return Some(idx);
        }
    }
    None
}

fn write_preserved_message_overflow_file(
    overflow_dir: &Path,
    message: &Message,
    kind: &str,
) -> Option<PathBuf> {
    let subdir = if kind == "image" {
        PRESERVED_IMAGE_OVERFLOW_DIR
    } else {
        PRESERVED_USER_OVERFLOW_DIR
    };
    let dir = overflow_dir.join(subdir);
    std::fs::create_dir_all(&dir).ok()?;
    let file_name = format!(
        "{}-{}-{}.json",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        kind,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(file_name);

    let mut payload = serde_json::Map::new();
    payload.insert("role".to_string(), Value::String(message.role.clone()));
    payload.insert("kind".to_string(), Value::String(kind.to_string()));
    payload.insert("content".to_string(), message.content.clone());
    if let Some(tool_calls) = &message.tool_calls {
        payload.insert(
            "tool_calls".to_string(),
            serde_json::to_value(tool_calls).ok()?,
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        payload.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }

    let serialized = serde_json::to_string_pretty(&Value::Object(payload)).ok()?;
    std::fs::write(&path, serialized).ok()?;
    Some(path)
}

fn build_preserved_message_overflow_stub(path: &Path, kind: &str) -> String {
    let payload = serde_json::json!({
        "kind": kind,
        "file_path": path.display().to_string(),
        "encoding": "original",
        "zero_compression": true,
        "hint": "use read_file to inspect exact content before continuing"
    });
    format!("{PRESERVED_CONTENT_STUB_PREFIX}{payload}")
}

fn try_spill_preserved_message_to_stub(messages: &mut [Message], overflow_dir: &Path) -> bool {
    let Some(idx) = first_preserved_content_spill_candidate(messages) else {
        return false;
    };
    let kind = if message_contains_image(&messages[idx].content) {
        "image"
    } else {
        "user"
    };
    let Some(path) = write_preserved_message_overflow_file(overflow_dir, &messages[idx], kind)
    else {
        return false;
    };
    messages[idx].content = Value::String(build_preserved_message_overflow_stub(&path, kind));
    true
}

/// 主动把体量过大的旧 user / 图片消息（保护尾窗之前的）搬到会话临时文件，
/// 原地替换为紧凑 stub。原文零压缩保存在磁盘上，但不再占用每轮请求的 payload。
///
/// 这与预算驱动的循环内 spill 互补：自从图片在预算里只按 [`IMAGE_BUDGET_CHARS`]
/// 名义计费后，单张大图本身不再触发 `messages_total_chars > max_chars`，于是
/// 循环内的 spill 永远不会被调用。这里改为「无论是否超预算，只要旧消息原始
/// 体量超过阈值就外溢」，既保证大图/大段用户原文被零压缩归档，又避免它们污染
/// 后续每一轮请求。最新一轮（保护尾窗内）的 user/图片永不外溢。
fn spill_oversized_preserved_messages(messages: &mut [Message], overflow_dir: &Path) {
    while try_spill_preserved_message_to_stub(messages, overflow_dir) {}
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
    for line in chunk
        .iter()
        .filter(|line| is_important_tool_line(line))
        .take(2)
    {
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
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
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
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        });
        if token.len() > 160 || token.is_empty() {
            continue;
        }
        if token.starts_with("http://") || token.starts_with("https://") {
            continue;
        }
        let looks_like_path = token.contains('/')
            || [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
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
    bytes.len() == 5 && bytes[0] == b'E' && bytes[1..].iter().all(|byte| byte.is_ascii_digit())
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
                        ',' | '.'
                            | ';'
                            | ':'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '<'
                            | '>'
                            | '"'
                            | '\''
                            | '`'
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
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
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
        if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items
        {
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
            role if role == ROLE_INTERNAL_NOTE => {
                if let Some(body) = automatic_summary_body(&text) {
                    let normalized =
                        summarize_text(&strip_nested_prior_summary_prefixes(body), 400);
                    if !normalized.is_empty() {
                        push_unique_limited(
                            &mut pre_summary_lines,
                            format!("- 更早摘要: {normalized}"),
                            3,
                        );
                    }
                }
            }
            role if is_system_like_role(role) => {}
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
                        push_unique_limited(
                            &mut current.tool_names,
                            tool_call.function.name.clone(),
                            6,
                        );
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
    let body_budget = max_chars
        .saturating_sub(reserved_tool_chars)
        .max(max_chars / 2);
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

    let tail_budget = target_chars
        .saturating_sub(selected_chars)
        .max(target_chars / 3);
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
    let tail: String = text
        .chars()
        .skip(char_count.saturating_sub(tail_budget))
        .collect();
    format!("{}…{}", head, tail)
}

fn first_tool_call_group(messages: &[Message]) -> Option<Vec<usize>> {
    let assistant_idx = messages.iter().position(|m| {
        if m.role != "assistant" {
            return false;
        }
        let Some(tool_calls) = &m.tool_calls else {
            return false;
        };
        if tool_calls.is_empty() {
            return false;
        }
        !tool_calls
            .iter()
            .any(|tc| is_non_compressible_tool(&tc.function.name))
    })?;
    let tool_call_ids: Vec<String> = messages[assistant_idx]
        .tool_calls
        .as_ref()
        .unwrap()
        .iter()
        .map(|tc| tc.id.clone())
        .collect();
    let mut group = vec![assistant_idx];
    // 仅收集紧跟 assistant 之后、连续出现的 tool 消息，遇到任何非 tool（含
    // 下一个 assistant/user/system/internal_note）即停。这样可以防止把
    // 不同 turn 中残留的同 id stub（来自 dedup_repeated_tool_results 替换）
    // 一并卷入同一 group，导致跨 turn 整组被折叠/删除。
    let mut i = assistant_idx + 1;
    while i < messages.len() && messages[i].role == "tool" {
        if let Some(ref id) = messages[i].tool_call_id {
            if tool_call_ids.contains(id) {
                group.push(i);
            } else {
                // 同位置出现了不属于本 assistant 的 tool 消息（理论上不应该
                // 发生，但若发生，停止扫描以避免破坏 OpenAI 配对协议）。
                break;
            }
        } else {
            break;
        }
        i += 1;
    }
    Some(group)
}

fn first_trim_candidate(messages: &[Message]) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);

    // 跳过头部所有 system-like（system / internal_note）消息：它们承载 agent
    // 指令、工具列表、历史摘要等关键上下文，不能被裁掉。
    // 旧实现只跳过以"对话摘要/历史摘要"前缀开头的条目，会把普通 system prompt
    // 当成可裁削目标，触发"上下文压缩后回复戛然而止"。
    // 同时把最近 N 轮 user 起始的整段尾部窗口都设为保护区，避免把多阶段任务
    // 的上一个子目标与当前子目标切开。
    let mut index = 0usize;
    while index < messages.len() && is_system_like_role(&messages[index].role) {
        index += 1;
    }

    while index < protected_tail_start {
        let message = &messages[index];

        // 已外溢到会话文件的 user/image 占位 stub 不删：它只是一个指向归档
        // 文件的指针（原文零压缩保存在磁盘上），删掉会让模型彻底失去线索。
        // 普通的 user / 含图片消息仍可被裁剪：大体量内容已由 proactive spill
        // 提前搬到文件，剩下的小体量旧 user 轮次应进入「丢弃 + 摘要」路径，
        // 否则 30+ 轮纯文本对话永远无法收敛进预算，且早期目标会被静默丢失。
        if is_preserved_user_or_image_stub(&value_to_string(&message.content)) {
            index += 1;
            continue;
        }

        // tool 不单删：否则可能破坏 assistant(tool_calls) ↔ tool 的配对关系。
        if message.role == "tool" {
            index += 1;
            continue;
        }

        // 带 tool_calls 的 assistant 不单删：保持协议配对一致性。
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            index += 1;
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
        let one_liner = tool_result_recall_one_liner(&result_text);
        lines.push(format!("- {} => {}", tc.function.name, one_liner));
    }
    if tool_calls.len() > 8 {
        lines.push(format!(
            "- ... ({} more tools omitted)",
            tool_calls.len() - 8
        ));
    }

    Some(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(lines.join("\n")),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

/// 单轮内折叠：LLM 摘要兜底时，尾窗（当前 user 轮）内部保留逐字的最近工具组数。
/// 更早的同轮工具组折叠成单行 stub，解决"臃肿全堆在一个 user 轮内、无跨轮边界
/// 可摘要"时压缩器空转的问题。
const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS: usize = 4;

/// 为折叠 stub 生成"召回锚点"单行：优先提取已外溢 tool 结果里的 `file_path:`
/// 指针（模型据此可重新 read_file），否则退回结果首个非空行。
/// 保证折叠早期 precision 工具组时仍留下可召回的线索，避免失忆式重复检索。
fn tool_result_recall_one_liner(result_text: &str) -> String {
    if let Some(path_line) = result_text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"))
    {
        return truncate_to_chars(&normalize_whitespace(path_line), 200);
    }
    let first_meaningful = result_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    truncate_to_chars(&normalize_whitespace(first_meaningful), 160)
}

/// 把消息序列中较早的 `assistant(tool_calls) + 配套 tool` 组整组折叠成单条
/// `internal_note` stub，逐字保留最近 `keep_recent_groups` 个工具组以及所有
/// 非工具组消息（user / system / internal_note / 纯文本 assistant）。
///
/// 关键不变量：折叠时把 assistant 及其全部 tool 响应**整组一起**替换成一条
/// stub，因此不会出现"留下 assistant.tool_calls 却丢掉配套 tool 响应"的配对
/// 断裂（OpenAI 协议要求二者成对）。含 `is_non_compressible_tool` 的组也会被
/// 折叠，但 stub 内保留其 `file_path:` 召回锚点。
///
/// 返回 `(folded_messages, folded_group_count)`。
fn fold_early_tool_groups(
    messages: &[Message],
    keep_recent_groups: usize,
) -> (Vec<Message>, usize) {
    // 定位所有 assistant(tool_calls) 起始位置，作为工具组锚点。
    let group_anchors: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            let has_calls = m
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false);
            (m.role == "assistant" && has_calls).then_some(idx)
        })
        .collect();
    if group_anchors.len() <= keep_recent_groups {
        return (messages.to_vec(), 0);
    }
    // 最近 keep_recent_groups 个工具组逐字保留；更早的折叠。
    let fold_before_anchor = group_anchors[group_anchors.len() - keep_recent_groups];

    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut folded_groups = 0usize;
    let mut idx = 0usize;
    while idx < messages.len() {
        let message = &messages[idx];
        let has_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| !calls.is_empty())
            .unwrap_or(false);
        // 只有位于折叠区（早于最近保留窗口）的 assistant(tool_calls) 组才折叠。
        if idx < fold_before_anchor && message.role == "assistant" && has_calls {
            let tool_call_ids: FxHashSet<&str> = message
                .tool_calls
                .as_ref()
                .unwrap()
                .iter()
                .map(|tc| tc.id.as_str())
                .collect();
            // 收集紧随其后、属于本 assistant 的连续 tool 响应，构成完整组。
            let mut group = vec![idx];
            let mut cursor = idx + 1;
            while cursor < messages.len() && messages[cursor].role == "tool" {
                match messages[cursor].tool_call_id.as_deref() {
                    Some(id) if tool_call_ids.contains(id) => group.push(cursor),
                    _ => break,
                }
                cursor += 1;
            }
            if let Some(stub) = fold_tool_call_group_to_stub(messages, &group) {
                out.push(stub);
                folded_groups += 1;
                idx = cursor;
                continue;
            }
        }
        out.push(message.clone());
        idx += 1;
    }
    (out, folded_groups)
}

fn is_summary_message(message: &Message) -> bool {
    if !is_system_like_role(&message.role) {
        return false;
    }
    let text = value_to_string(&message.content);
    text.starts_with("对话摘要（自动压缩")
        || text.starts_with("历史摘要（自动压缩")
        || text.starts_with("[mid-turn-summary]")
}

const KEEP_RECENT_TOOL_MESSAGES: usize = 6;

/// 带 tool_calls 的 assistant 消息中，保留完整 reasoning_content 的最近轮数。
/// 更早的 tool-call reasoning 置 None（DeepSeek 由 echo 兜底补空字符串占位），
/// 防止历史 reasoning 文本在长 session 里单调累积，拖慢响应并挤占上下文预算。
const KEEP_RECENT_TOOL_CALL_REASONING: usize = 3;

fn tool_message_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "tool").then_some(i))
        .collect()
}

/// 判断 message content 是否包含真正的图片附件（OpenAI Vision schema）。
/// 图片必须以 multimodal `Value::Array` 形式存在，且数组中含
/// `{"type":"image_url", "image_url":{...}}`。
/// 旧实现用 `text.contains("data:image/")` 误判：agent 在普通文本里讨论
/// `data:image/png` 字串就会被整条替换，丢信息。
fn message_contains_image(content: &Value) -> bool {
    let Some(arr) = content.as_array() else {
        return false;
    };
    arr.iter().any(|item| {
        item.get("type").and_then(|v| v.as_str()) == Some("image_url")
            || item.get("image_url").is_some()
    })
}

fn redact_images_except_last(messages: &mut [Message], keep_last: usize) {
    let _ = (messages, keep_last);
    // 用户要求图片内容零压缩：历史压缩阶段不再把旧图片替换成 [[image omitted]]。
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
        // 完全相等去重仅对 tool 启用：用户/助手/system 原文不做去重。
        if m.role == "tool" && m.role == prev_role && text == prev_content {
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

/// 裁剪历史中的 reasoning_content，只保留确有必要回传给厂商的那些。
///
/// 较老的 reasoning chain 对后续 turn 决策几乎没有帮助，去掉可节省上下文预算。
/// 但 DeepSeek thinking-mode 有一个硬约束：**凡是触发过 tool_calls 的那一回合，
/// 其 assistant 消息的 reasoning_content 必须在后续所有请求中原样回传**，否则
/// 会返回 `400 invalid_request_error: The reasoning_content in the thinking mode
/// must be passed back to the API`。因此这里的策略是：
/// - 带 `tool_calls` 的 assistant 消息：只保留最近 `KEEP_RECENT_TOOL_CALL_REASONING`
///   轮的完整 reasoning_content，更早的置 None——DeepSeek 所需的字段形状由
///   `ensure_reasoning_content_echo_for_thinking_model` 用空字符串占位补齐，既满足
///   协议校验又避免历史 reasoning 文本在长 session 里单调累积、拖慢并"变蠢"；
/// - 不带 tool_calls 的纯回答 assistant 消息：只保留最近一条的 reasoning_content，
///   其余置 None（OpenAI 等仅要求与最近一次 tool_call 同回合的 reasoning 配对，
///   旧的纯回答 reasoning 可安全丢弃）。
fn keep_only_recent_reasoning_content(messages: &mut [Message]) {
    // 最近一条「不带 tool_calls」的 assistant reasoning 索引——这一条予以保留。
    let keep_plain_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| {
            m.role == "assistant" && m.reasoning_content.is_some() && m.tool_calls.is_none()
        })
        .map(|(idx, _)| idx);

    // 带 tool_calls 的 assistant reasoning 跨轮滑窗：只保留最近 N 条的完整文本，
    // 更早的置 None（DeepSeek 会由 echo 兜底补空字符串占位）。
    let tool_call_reasoning_count = messages
        .iter()
        .filter(|m| {
            m.role == "assistant" && m.reasoning_content.is_some() && m.tool_calls.is_some()
        })
        .count();
    let drop_tool_call_reasoning_before =
        tool_call_reasoning_count.saturating_sub(KEEP_RECENT_TOOL_CALL_REASONING);
    let mut seen_tool_call_reasoning = 0usize;

    for (idx, m) in messages.iter_mut().enumerate() {
        if m.role != "assistant" || m.reasoning_content.is_none() {
            continue;
        }
        // 带 tool_calls 的回合：仅保留最近 N 条完整 reasoning，其余置 None。
        if m.tool_calls.is_some() {
            let rank = seen_tool_call_reasoning;
            seen_tool_call_reasoning += 1;
            if rank < drop_tool_call_reasoning_before {
                m.reasoning_content = None;
            }
            continue;
        }
        // 纯回答回合：只保留最近一条。
        if Some(idx) == keep_plain_idx {
            continue;
        }
        m.reasoning_content = None;
    }
}

/// 跨轮 tool 结果去重：同一 (tool_name, normalized_args) 在历史中出现多次时，
/// 把较早的 tool 结果替换为单行 stub（保留 tool_call_id 以维持 OpenAI tool-calls 协议正确性）。
/// 仅压缩内容，不删除消息，避免 assistant tool_calls 与 tool 响应的配对断裂。
/// 最近 KEEP_RECENT_TOOL_MESSAGES 条 tool 消息一律保留全文。
fn dedup_repeated_tool_results(messages: &mut [Message]) {
    use rustc_hash::FxHashMap;

    // 收集 (tool_name, args_signature) → 出现次数与索引
    // 通过 assistant.tool_calls 关联 tool_call_id → (name, args)
    let mut id_to_signature: FxHashMap<String, (String, String)> = FxHashMap::default();
    for message in messages.iter() {
        if let Some(tool_calls) = &message.tool_calls {
            for tc in tool_calls {
                let args_norm = serde_json::from_str::<Value>(&tc.function.arguments)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| tc.function.arguments.clone());
                id_to_signature.insert(tc.id.clone(), (tc.function.name.clone(), args_norm));
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

    let mut seen: FxHashMap<(String, String), usize> = FxHashMap::default();
    for (rank, &idx) in tool_indices.iter().enumerate() {
        let signature = messages[idx]
            .tool_call_id
            .as_ref()
            .and_then(|id| id_to_signature.get(id))
            .cloned();
        let signature = match signature {
            Some(sig) => sig,
            None => {
                // 孤儿 tool：找不到对应的 assistant.tool_calls（可能因为 assistant 消息
                // 已被早期裁剪/丢弃，或写入历史时配对就已经断裂）。这些消息在
                // normalize_messages_for_request 阶段会被丢掉，但在压缩阶段仍占用
                // 字符预算。最近 KEEP_RECENT_TOOL_MESSAGES 条保留全文以防误伤；
                // 较旧的孤儿一律折叠为短 stub，避免阻塞后续压缩判断。
                if rank < protected_from {
                    let tool_call_id = messages[idx].tool_call_id.clone().unwrap_or_default();
                    let stub = if tool_call_id.is_empty() {
                        "[orphan tool result: corresponding assistant.tool_calls missing; content dropped]".to_string()
                    } else {
                        format!(
                            "[orphan tool result for {}: corresponding assistant.tool_calls missing; content dropped]",
                            tool_call_id
                        )
                    };
                    messages[idx].content = Value::String(stub);
                }
                continue;
            }
        };
        let count = seen.entry(signature.clone()).or_insert(0);
        *count += 1;
        if rank >= protected_from {
            continue;
        }
        if is_non_compressible_tool(&signature.0) {
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

#[cfg(test)]
mod fold_early_tool_groups_tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn assistant_call(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_result(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    /// 构造：system + user + N 个 (assistant tool_calls + tool 结果) 组，全部在
    /// 同一个 user 轮内（只有一条 user 消息）——正是"臃肿全堆在当前轮"的场景。
    fn single_turn_with_groups(n: usize, tool_result_chars: usize) -> Vec<Message> {
        let mut messages = vec![msg("system", "system prompt"), msg("user", "干活")];
        for i in 0..n {
            let id = format!("call-{i}");
            messages.push(assistant_call(&id, "text_grep"));
            messages.push(tool_result(&id, &"x".repeat(tool_result_chars)));
        }
        messages
    }

    fn assert_tool_pairs_consistent(messages: &[Message]) {
        let mut assistant_ids: FxHashSet<String> = FxHashSet::default();
        for m in messages {
            if m.role == "assistant"
                && let Some(calls) = &m.tool_calls
            {
                for c in calls {
                    assistant_ids.insert(c.id.clone());
                }
            }
        }
        let mut tool_ids: FxHashSet<String> = FxHashSet::default();
        for m in messages {
            if m.role == "tool"
                && let Some(id) = &m.tool_call_id
            {
                tool_ids.insert(id.clone());
            }
        }
        assert_eq!(
            assistant_ids, tool_ids,
            "every assistant.tool_calls id must have a paired tool message and vice versa"
        );
    }

    #[test]
    fn folds_early_groups_in_a_single_bloated_turn() {
        let messages = single_turn_with_groups(10, 2_000);
        let before = messages_total_chars(&messages);

        let (folded, folded_groups) = fold_early_tool_groups(&messages, 4);

        // 10 组里最近 4 组逐字保留，较早 6 组被折叠。
        assert_eq!(folded_groups, 6);
        let after = messages_total_chars(&folded);
        assert!(
            after < before,
            "folding must reduce size: {after} !< {before}"
        );
        assert_tool_pairs_consistent(&folded);
    }

    #[test]
    fn preserves_user_message_verbatim() {
        let messages = single_turn_with_groups(8, 1_500);
        let (folded, _) = fold_early_tool_groups(&messages, 4);

        let user = folded
            .iter()
            .find(|m| m.role == "user")
            .expect("user message must survive");
        assert_eq!(value_to_string(&user.content), "干活");
    }

    #[test]
    fn keeps_recent_groups_verbatim() {
        let messages = single_turn_with_groups(8, 1_500);
        let (folded, _) = fold_early_tool_groups(&messages, 4);

        // 最近 4 组的 tool 结果应原样保留（未被折叠成 stub）。
        let full_tool_results = folded
            .iter()
            .filter(|m| m.role == "tool" && value_to_string(&m.content) == "x".repeat(1_500))
            .count();
        assert_eq!(full_tool_results, 4);
    }

    #[test]
    fn no_op_when_group_count_within_keep_window() {
        let messages = single_turn_with_groups(3, 1_000);
        let (folded, folded_groups) = fold_early_tool_groups(&messages, 4);

        assert_eq!(folded_groups, 0);
        assert_eq!(folded.len(), messages.len());
    }

    #[test]
    fn stub_preserves_file_path_recall_anchor() {
        let mut messages = vec![msg("system", "s"), msg("user", "干活")];
        // 早期一组：read_file 结果已外溢，含 file_path 指针，必须在 stub 中保留。
        messages.push(assistant_call("call-old", "read_file"));
        messages.push(tool_result(
            "call-old",
            "Output preserved for non-compressible tool `read_file`.\n- file_path: /tmp/session/xyz.txt\n- use read_file to inspect exact content.",
        ));
        // 追加足够多的近端组把上面那组挤进折叠区。
        for i in 0..6 {
            let id = format!("call-{i}");
            messages.push(assistant_call(&id, "text_grep"));
            messages.push(tool_result(&id, "recent"));
        }

        let (folded, folded_groups) = fold_early_tool_groups(&messages, 4);
        assert!(folded_groups >= 1);
        let stub_text: String = folded
            .iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .map(|m| value_to_string(&m.content))
            .collect();
        assert!(
            stub_text.contains("/tmp/session/xyz.txt"),
            "folded stub must retain the file_path recall anchor, got: {stub_text}"
        );
    }

    fn assistant_call_with_reasoning(id: &str, name: &str, reasoning: &str) -> Message {
        let mut m = assistant_call(id, name);
        m.reasoning_content = Some(reasoning.to_string());
        m
    }

    fn assistant_plain_with_reasoning(reasoning: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String("答复".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some(reasoning.to_string()),
        }
    }

    /// 跨轮滑窗：带 tool_calls 的 assistant reasoning 只保留最近
    /// `KEEP_RECENT_TOOL_CALL_REASONING` 条，更早的置 None；纯回答 reasoning 只留最近一条。
    #[test]
    fn keeps_only_recent_tool_call_reasoning_across_turns() {
        assert_eq!(KEEP_RECENT_TOOL_CALL_REASONING, 3);

        let mut messages = vec![
            msg("system", "s"),
            msg("user", "干活"),
            // 早期纯回答 reasoning：非最近一条，应被丢弃。
            assistant_plain_with_reasoning("early-plain"),
        ];
        // 5 组带 tool_calls 的 reasoning：rank 0/1 应丢弃，rank 2/3/4 保留。
        for i in 0..5 {
            let id = format!("call-{i}");
            messages.push(assistant_call_with_reasoning(
                &id,
                "text_grep",
                &format!("tc-{i}"),
            ));
            messages.push(tool_result(&id, "r"));
        }
        // 最近一条纯回答 reasoning：应保留。
        messages.push(assistant_plain_with_reasoning("final-plain"));

        keep_only_recent_reasoning_content(&mut messages);

        // 用 tool_call id 定位 tool-call reasoning。
        let tc_reasoning = |id: &str| -> Option<String> {
            messages
                .iter()
                .find(|m| {
                    m.tool_calls
                        .as_ref()
                        .map(|calls| calls.iter().any(|c| c.id == id))
                        .unwrap_or(false)
                })
                .and_then(|m| m.reasoning_content.clone())
        };
        assert_eq!(
            tc_reasoning("call-0"),
            None,
            "rank 0 tool-call reasoning must be dropped"
        );
        assert_eq!(
            tc_reasoning("call-1"),
            None,
            "rank 1 tool-call reasoning must be dropped"
        );
        assert_eq!(tc_reasoning("call-2").as_deref(), Some("tc-2"));
        assert_eq!(tc_reasoning("call-3").as_deref(), Some("tc-3"));
        assert_eq!(tc_reasoning("call-4").as_deref(), Some("tc-4"));

        // 纯回答 reasoning：只保留最近一条（final-plain），早期一条置 None。
        let plain_reasonings: Vec<Option<String>> = messages
            .iter()
            .filter(|m| m.role == "assistant" && m.tool_calls.is_none())
            .map(|m| m.reasoning_content.clone())
            .collect();
        assert_eq!(
            plain_reasonings,
            vec![None, Some("final-plain".to_string())]
        );
    }

    #[test]
    fn persisted_summary_absorbs_prior_summary_without_nested_prefix() {
        let messages = vec![
            msg(
                ROLE_INTERNAL_NOTE,
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n- 更早摘要: 初始目标: 修复压缩\n- 已知结论: 保留路径",
            ),
            msg("user", "继续排查 compress.rs"),
            msg("assistant", "发现摘要递归污染"),
        ];

        let summary = build_persisted_summary_text(&messages, 2_000);

        assert!(summary.contains("初始目标: 修复压缩"), "{summary}");
        assert!(
            !summary.contains("更早摘要: - 更早摘要:"),
            "summary should not recursively wrap prior summaries: {summary}"
        );
    }

    #[test]
    fn summary_model_input_drops_ephemeral_internal_notes() {
        let mut messages = vec![
            msg("user", "修复问题"),
            msg(ROLE_INTERNAL_NOTE, "self_note:\n一次性观察"),
            msg(ROLE_INTERNAL_NOTE, "tool_followup:output_truncated"),
            msg(
                ROLE_INTERNAL_NOTE,
                "对话摘要（自动压缩，以下为早期对话要点）：\n初始目标: 保留",
            ),
            msg(
                ROLE_INTERNAL_NOTE,
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n初始目标: 应去重",
            ),
        ];

        normalize_internal_notes_for_summary_model(&mut messages);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        let note = value_to_string(&messages[1].content);
        assert!(note.contains("已有历史摘要"), "{note}");
        assert!(note.contains("初始目标: 保留"), "{note}");
        assert!(!note.contains("self_note"), "{note}");
        assert!(!note.contains("tool_followup"), "{note}");
        assert!(!note.contains("应去重"), "{note}");
    }
}
