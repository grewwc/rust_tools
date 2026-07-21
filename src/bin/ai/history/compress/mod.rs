use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::types::App;

use super::types::{
    MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start,
};

pub(crate) mod llm_prune;
mod text_utils;
mod tool_groups;
mod tool_overflow;

use text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
use tool_groups::{
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS, first_tool_call_group, first_trim_candidate,
    fold_early_tool_groups, fold_tool_call_group_to_stub, recent_tool_group_message_indices,
};
#[cfg(test)]
use tool_overflow::normalize_internal_notes_for_summary_model;
use tool_overflow::{
    age_out_overflow_stub_previews, build_persisted_summary_text,
    build_persisted_summary_text_with_app, enforce_protected_precision_group_budget,
    is_non_compressible_tool, normalize_preserved_message_stubs_for_model,
    prepare_tool_messages_structured, spill_oversized_preserved_messages, tool_line_signature,
    try_spill_preserved_message_to_stub,
};

/// 所有"自动压缩摘要" note 的前缀。写入端（生成摘要 note）与识别端
/// （防重复 guard、sqlite 接续点、请求侧分组）**必须共用这一份清单**，否则
/// 会出现"写入的前缀识别端不认"的断裂——历史上 `长期记忆摘要（压缩保留）`
/// 就因未登记而绕过防重复 guard，导致每轮重复插入摘要 note、上下文预算被
/// 持续推高、压缩管线每个 turn 空转。新增摘要前缀时只改这里。
///
/// 注意：条目应为"去除前导空白后"的裸前缀；判定统一走 [`is_summary_note_text`]，
/// 它会先 `trim_start` 再逐一 `starts_with`，因此全角/半角冒号只需各列一次。
pub(in crate::ai) const SUMMARY_NOTE_PREFIXES: &[&str] = &[
    "对话摘要（自动压缩",
    "历史摘要（自动压缩",
    "长期记忆摘要（压缩保留）",
    "[mid-turn-summary]",
];

/// 归档指针 note（overflow 原文回指）的前缀。与摘要 note 成对出现，
/// P1 折叠逻辑据此识别并去重堆积的归档指针。
pub(in crate::ai) const ARCHIVE_NOTE_PREFIX: &str = "长期记忆归档";

/// 判断一段文本是否是"自动压缩摘要" note 正文（前缀匹配，容忍前导空白）。
/// 这是摘要识别的**唯一真源**，供 guard / sqlite / 请求规范化统一调用。
pub(in crate::ai) fn is_summary_note_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    SUMMARY_NOTE_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

/// 判断一段文本是否是 overflow 归档指针 note。
fn is_archive_note_text(text: &str) -> bool {
    text.trim_start().starts_with(ARCHIVE_NOTE_PREFIX)
}

const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;
/// 压缩兜底（first_trim_candidate）时保护最近 user 起始尾窗的动态上下限。
/// 小上下文优先保留 3 轮提升多阶段任务连续性；超大上下文回退到 2 轮控预算。
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN: usize = 2;
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX: usize = 3;
/// 当上下文字符数不超过该阈值时，优先保留 3 轮 user。
const KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS: usize = 48_000;

/// 计算裁剪/外溢/折叠时应完整豁免的「最近 user 起始尾窗」轮数。
///
/// 基础判定（按总量二选一）不变：≤48K → 3 轮，否则 2 轮——正常会话零行为变化。
///
/// **字节上限逃逸阀**（`budget > 0` 时生效）：保护尾窗是「完整豁免区」，它自身
/// 不应超过整个历史预算。tool-heavy agentic 会话（少 user 轮 × 每轮上百次工具
/// 调用）会让尾窗撑到 MB 级且**结构上禁止收敛**——尾窗内即便有几百条工具组也
/// 一律豁免。此时逐步收缩保护轮数，让「倒数第 2 轮及更早」的工具组暴露给
/// fold/spill 路径恢复收敛。**保底不变式：永不低于 1 轮**——最新一轮 user 及其
/// 工具组始终逐字保留（由 `KEEP_RECENT_TOOL_GROUPS` 组级保护继续兜底）。
///
/// `budget == 0` 表示调用方显式不设上限（保持旧行为），供无预算语境复用。
fn keep_recent_user_turns_when_trimming(messages: &[Message], budget: usize) -> usize {
    let mut keep = if messages_total_chars(messages) <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX
    } else {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN
    };
    if budget == 0 {
        return keep;
    }
    while keep > 1 {
        let tail_start = retained_turn_start(messages, keep);
        if messages_total_chars(&messages[tail_start..]) <= budget {
            break;
        }
        keep -= 1;
    }
    keep
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
const CONTEXT_CHECKPOINT_MARKER_PREFIX: &str = "[context_checkpoint";

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

/// checkpoint 正文已写入会话 asset；这里的短标记是模型在压缩后重新找到正文的
/// 唯一索引，因此既不能被摘要吞掉，也不能被普通裁剪删掉。
pub(super) fn is_context_checkpoint_marker(m: &Message) -> bool {
    m.role == ROLE_INTERNAL_NOTE
        && value_to_string(&m.content)
            .trim_start()
            .starts_with(CONTEXT_CHECKPOINT_MARKER_PREFIX)
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
    out.push_str(
        "长期记忆归档：更早的原始对话已移出上下文窗口，原文保存在会话归档文件中（零压缩）。\n",
    );
    out.push_str("归档文件: ");
    out.push_str(file_path);
    out.push('\n');
    out.push_str("重要：不要主动读取此归档文件。仅当当前问题确实依赖已被移出的前文细节（如最初目标、之前的决定、旧报错、或更早的工具输出）而摘要中找不到答案时，才使用 read_file 分段读取（建议 offset=1, limit=200 起步）。若当前上下文足够回答问题，忽略此归档即可。\n");
    out
}

pub(in crate::ai) fn compress_messages_for_context(
    mut messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
) -> Vec<Message> {
    // 历史库中可能仍有旧版 JSON stub。它是压缩器的内部协议，不能原样交给模型，
    // 否则模型会把它当普通用户文本甚至直接复述到最终回复中。
    normalize_preserved_message_stubs_for_model(&mut messages);
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    // 在做大块压缩前先剪 self_note 滑动上限，避免上千轮 turn 累积的
    // self_note（已写入 MemoryStore，messages 里那条仅是冗余备份）
    // 单调膨胀。MemoryStore 仍保留全部记录。
    let messages = trim_self_notes_to_recent(messages, MAX_SELF_NOTES_IN_MESSAGES);

    // 收敛历史上因防重复 guard 断裂而堆积的重复摘要/归档 note。放在请求期入口，
    // 让已经堆积了几十对 note 的旧 session 下一次请求就立刻恢复正常，无需等落盘。
    let messages = coalesce_accumulated_summary_notes(messages);

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
        let summary_source: Vec<Message> = older
            .iter()
            .filter(|message| !is_context_checkpoint_marker(message))
            .cloned()
            .collect();
        let summary = build_persisted_summary_text(&summary_source, summary_max_chars);
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
    out.extend(
        older
            .iter()
            .filter(|message| is_context_checkpoint_marker(message))
            .cloned(),
    );
    out.extend_from_slice(recent);
    shrink_messages_to_fit_with_summary(out, max_chars, summary_max_chars, overflow_dir.as_deref())
}

pub(in crate::ai) fn sanitize_message_for_persisted_history(message: &Message) -> Message {
    let mut sanitized = message.clone();
    if sanitized.role != "assistant" {
        return sanitized;
    }

    // 持久化历史只保留跨 turn 真正需要的 assistant 事实：
    // - `reasoning_content` 对后续请求没有必要保留原文，provider 需要字段形状时
    //   由 request 层统一补空字符串；
    // - 带 tool_calls 的 assistant narration 属于"本轮过程性话术"，真正的地面真相
    //   是结构化 tool_calls + tool 结果，持久化该 narration 会让单个 user turn
    //   膨胀成几十/几百条低价值 assistant 噪音。
    sanitized.reasoning_content = None;
    if sanitized
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty())
    {
        sanitized.content = Value::String(String::new());
    }
    sanitized
}

fn sanitize_persisted_history_messages(messages: Vec<Message>) -> Vec<Message> {
    let messages = coalesce_accumulated_summary_notes(messages);
    messages
        .into_iter()
        .map(|message| sanitize_message_for_persisted_history(&message))
        .collect()
}

/// 收敛历史上因防重复 guard 断裂而堆积的多条摘要 / 归档 note。
///
/// 背景：`长期记忆摘要（压缩保留）` 前缀曾未登记进 `is_summary_message`，导致
/// 每轮压缩都在开头重复插入一对「摘要 + 归档」note，长 session 可堆积几十对，
/// 既污染上下文预算又推高 `total_chars` 让压缩管线每 turn 空转。
///
/// 折叠策略（无损）：
/// - **摘要 note**：把每条正文（去 header 后）按原顺序去重拼接成**一条**，放回
///   第一条摘要原来的位置。不同轮次挤出窗口时各自记录的"初始目标"因此全部保留。
/// - **归档指针 note**：内容完全相同的只保留一条（overflow 归档文件路径唯一，
///   去重无损），紧跟合并后的摘要。
/// - 其余消息一律原样保留、顺序不变（绝不触碰非摘要/归档消息）。
///
/// 仅当摘要 + 归档 note 合计 > 2 条时才折叠，避免对正常历史做无谓改写
/// （返回值与入参逐条相等时，上层 `compacted == messages` 判定会跳过落盘）。
fn coalesce_accumulated_summary_notes(messages: Vec<Message>) -> Vec<Message> {
    let note_count = messages
        .iter()
        .filter(|m| is_summary_or_archive_note(m))
        .count();
    if note_count <= 2 {
        return messages;
    }

    // 合并所有摘要正文（去重、保序）。
    let mut merged_bodies: Vec<String> = Vec::new();
    let mut first_summary_role: Option<String> = None;
    let mut archive_note: Option<Message> = None;
    for m in &messages {
        if is_summary_message(m) {
            if first_summary_role.is_none() {
                first_summary_role = Some(m.role.clone());
            }
            let text = value_to_string(&m.content);
            let body = automatic_summary_body(&text).unwrap_or_else(|| text.trim());
            let body = body.trim();
            if !body.is_empty() && !merged_bodies.iter().any(|b| b == body) {
                merged_bodies.push(body.to_string());
            }
        } else if is_archive_note_message(m) && archive_note.is_none() {
            archive_note = Some(m.clone());
        }
    }

    let merged_summary = if merged_bodies.is_empty() {
        None
    } else {
        Some(Message {
            role: first_summary_role.unwrap_or_else(|| ROLE_INTERNAL_NOTE.to_string()),
            content: Value::String(format!(
                "长期记忆摘要（压缩保留）:\n{}",
                merged_bodies.join("\n")
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
    };

    // 重建序列：在"第一条摘要/归档 note"的位置放入合并摘要 + 单一归档指针，
    // 丢弃其余摘要/归档 note，其他消息原样保留。
    let mut out = Vec::with_capacity(messages.len());
    let mut inserted = false;
    for m in messages {
        if is_summary_or_archive_note(&m) {
            if !inserted {
                if let Some(summary) = merged_summary.clone() {
                    out.push(summary);
                }
                if let Some(archive) = archive_note.clone() {
                    out.push(archive);
                }
                inserted = true;
            }
            // 其余重复摘要/归档 note 丢弃。
        } else {
            out.push(m);
        }
    }
    out
}

fn is_summary_or_archive_note(m: &Message) -> bool {
    is_summary_message(m) || is_archive_note_message(m)
}

fn is_archive_note_message(m: &Message) -> bool {
    is_system_like_role(&m.role) && is_archive_note_text(&value_to_string(&m.content))
}

pub(in crate::ai) fn compact_persisted_history(messages: Vec<Message>) -> Vec<Message> {
    let messages = sanitize_persisted_history_messages(messages);
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

    let checkpoint_markers: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary_source: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| !is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary =
        build_persisted_summary_text(&summary_source, PERSISTED_HISTORY_SUMMARY_MAX_CHARS);
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
    out.extend(checkpoint_markers);
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
    let messages = sanitize_persisted_history_messages(messages);
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

    let checkpoint_markers: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary_source: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| !is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary = build_persisted_summary_text_with_app(
        app,
        &summary_source,
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
    out.extend(checkpoint_markers);
    out.extend_from_slice(&messages[split_at..]);
    out
}

/// 当 `first_tool_call_group` 折不动（剩余可折叠组都含 `read_file`/`code_search`
/// 等 non-compressible 工具、被它按策略拒绝）但仍超预算时的下一档手段：用
/// [`fold_early_tool_groups`] 递进折叠「保护尾窗之外」的这些组为单行
/// `compressed_tool_round` note（内含 file_path 召回锚点，模型可 read_file 回读）。
///
/// 这与 `mid_turn_llm_summarize` 的 Path B+C 复用**同一个**久经测试的折叠函数，
/// 只是把它前移到常规/落盘压缩路径——修复「tool-heavy 会话（少 user 轮 × 上百次
/// read_file/code_search）在 `compress_messages_for_context` / `shrink_*` 里永远
/// 压不掉工具组、整段历史无法收敛进预算」的总根因。
///
/// 返回是否发生了「有效折叠」（净字符数下降）。`keep_recent` 从
/// [`KEEP_RECENT_TOOL_GROUPS`] 递进收紧到 0，保证最近的工具组尽量逐字保留、
/// 只有仍超预算才逐步放宽折叠范围；每一步都要求净下降，避免无进展空转。
fn fold_noncompressible_tool_groups_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    let mut made_progress = false;
    for &keep_recent in &[KEEP_RECENT_TOOL_GROUPS, 2, 1, 0] {
        if messages_total_chars(messages) <= max_chars {
            break;
        }
        let (folded, folded_groups) = fold_early_tool_groups(messages, keep_recent, overflow_dir);
        if folded_groups == 0 {
            continue;
        }
        // 折叠必须带来净下降才采纳，否则丢弃本次结果继续收紧 keep_recent，
        // 严防「组数变了但字符没降」导致的循环空转。
        if messages_total_chars(&folded) < messages_total_chars(messages) {
            *messages = folded;
            made_progress = true;
        }
    }
    made_progress
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

    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    // dedup 必须在 offload 之前：offload 会把超阈值的旧 read_file 全文搬到磁盘并
    // 替换成带**唯一临时路径**的 stub，一旦如此，逐字节相同的重复副本就因路径不同
    // 而无法再折叠。先做内容级 dedup，把冗余全文折叠成回指 stub，再对真正需要保留
    // 的少数版本做 offload。
    dedup_repeated_tool_results(&mut messages);
    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_GROUPS, overflow_dir);
    enforce_protected_precision_group_budget(
        &mut messages,
        KEEP_RECENT_TOOL_GROUPS,
        max_chars / 2,
        overflow_dir,
    );

    // 先无条件外溢体量过大的旧 user/图片消息（保护尾窗除外），与
    // `shrink_messages_to_fit_with_summary` 保持一致。图片在预算里只按名义成本
    // 计费、大 user 原文零压缩搬盘为 stub 后，下面的裁剪循环因
    // `is_preserved_user_or_image_stub` 自动跳过它们——避免旧 user 被通用裁剪
    // 直接 `remove` 掉（那违反分类给 RecentUser 的 OffloadOnly 语义、静默丢原文）。
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // 保护尾窗之外的 overflow stub 预览体老化折叠为单行锚点（file_path 召回不丢），
    // 收敛「上百条早期 read_file 预览单调累积」的历史膨胀。放在预算判断之前，
    // 让即便未超预算的会话也能持续收敛已外溢 stub。尾窗轮数受 max_chars 字节上限
    // 约束：tool-heavy 会话尾窗过大时自动缩窗，把更早的 stub 暴露给老化折叠。
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        if let Some(group) = first_tool_call_group(&messages) {
            // 渐进式卸载：先尝试折叠为单行 stub 而不是整组删除，
            // 让模型仍能"看见"早期发生过哪些工具调用、以什么结果收尾，
            // 避免后续轮次因为完全失忆而重复工作。
            if let Some(stub) = fold_tool_call_group_to_stub(&messages, &group, overflow_dir) {
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
        // first_tool_call_group 已折不动（剩余可折叠组都含 read_file/code_search
        // 等 non-compressible 工具）但仍超预算：用 fold_early_tool_groups 递进折叠
        // 保护尾窗之外的这些组。有效折叠则继续循环重新评估；否则落到通用裁剪。
        if fold_noncompressible_tool_groups_to_fit(&mut messages, max_chars, overflow_dir) {
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages, max_chars) {
            // 旧 user（含图片的多模态 user）绝不静默删除：这是分类给 RecentUser 的
            // OffloadOnly 语义。先尝试把原文零压缩搬到归档文件、替换成回指 stub；
            // 搬盘成功则继续裁剪循环。
            if messages[idx].role == "user" {
                if let Some(dir) = overflow_dir
                    && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
                {
                    continue;
                }
                // 无法外溢（无 overflow_dir 或体量过小、上面的 proactive spill 已处理
                // 掉所有超阈值 user）：直接跳出裁剪循环，绝不 `remove` 掉 user 原文。
                // 残余的轻微超阈值交由上层硬阈值 `mid_turn_llm_summarize` 兜底，
                // 避免同一小 user 被反复选中造成死循环。
                break;
            }
            // 其余可裁候选（assistant 纯叙述等，first_trim_candidate 已排除 tool
            // 与带 tool_calls 的 assistant）保持原样删除。
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

    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    // dedup 先于 offload：理由同 shrink_messages_to_fit——避免逐字节相同的重复
    // read_file 全文各自被 offload 成唯一临时路径 stub 而失去折叠机会。
    dedup_repeated_tool_results(&mut messages);
    prepare_tool_messages_structured(&mut messages, 480, KEEP_RECENT_TOOL_GROUPS, overflow_dir);
    enforce_protected_precision_group_budget(
        &mut messages,
        KEEP_RECENT_TOOL_GROUPS,
        max_chars / 2,
        overflow_dir,
    );

    // 先无条件外溢体量过大的旧 user/图片消息（最新一轮保护尾窗除外）。
    // 图片在预算里只按名义成本计费，单张大图不再触发超预算循环，因此必须
    // 在预算判断之前就把它们零压缩搬到文件，避免每轮请求都携带完整 base64。
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // 保护尾窗之外的 overflow stub 预览体老化折叠为单行锚点（与 shrink_messages_to_fit
    // 对称）。收敛早期 read_file 预览的单调累积，file_path 召回锚点保留不丢。
    // 尾窗轮数同样受 max_chars 字节上限约束（见 keep_recent_user_turns_when_trimming）。
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);

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
            if let Some(stub) = fold_tool_call_group_to_stub(&messages, &group, overflow_dir) {
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
        // first_tool_call_group 已折不动（剩余可折叠组都含 read_file/code_search
        // 等 non-compressible 工具）但仍超预算：用 fold_early_tool_groups 递进折叠
        // 保护尾窗之外的这些组。有效折叠则继续循环重新评估；否则落到通用裁剪。
        if fold_noncompressible_tool_groups_to_fit(&mut messages, max_chars, overflow_dir) {
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages, max_chars) {
            dropped.push(messages.remove(idx));
            continue;
        }
        if let Some(dir) = overflow_dir
            && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
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
            } else {
                // flush 失败：绝不删历史。把本轮已从 messages 移除、但尚未成功
                // 落盘的 dropped 按原相对顺序放回头部（dropped 在循环中按时间升序
                // 累积，且整组 tool_call/tool 作为连续块放入，故放回头部即恢复
                // 原始时序、不破坏配对），随后立即返回——跳过摘要/归档 note 注入
                //（防止产生没有对应归档文件的悬空指针 note）、truncate 与 reasoning
                // 清理。返回值可能仍超预算，但那是可恢复的（下轮重试压缩 / 请求层
                // clamp），而数据丢失不可逆——遵守“写入失败严禁删除历史”的既有教训。
                messages.splice(0..0, dropped);
                return messages;
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
    messages.iter().map(message_billable_chars).sum::<usize>()
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
    prepare_tool_messages_structured(&mut out, 480, KEEP_RECENT_TOOL_GROUPS, overflow_dir);
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
    let overflow_dir = crate::ai::history::SessionStore::new(app.config.history_file.as_path())
        .session_assets_dir(&app.session_id);
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
                    Some(overflow_dir.as_path()),
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
    for &keep_recent in &[MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS, 2, 1, 0] {
        if best_after <= hard_target {
            break;
        }
        let current = best.as_ref().unwrap_or(&messages);
        let (folded, folded_groups) =
            fold_early_tool_groups(current, keep_recent, Some(overflow_dir.as_path()));
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
    // 或大量近期工具结果），早期历史已压无可压，臃肿全在保护尾窗内。user 消息
    // 承载任务指令，head+tail 截断会让 agent 丢失任务目标，因此永不截断——
    // 宁可让该轮带超大 user 超限发请求（由 normalize_messages_for_request 兜底
    // 或 provider 4xx 后重试），也不能把任务指令截成预览碎片。
    // 这是绝对最后手段——宁可截断也不能让模型 4xx。
    if best_after > hard_target {
        let current = best.unwrap_or(messages);
        let capped = cap_oversized_non_system_messages(current, PATH_C_PER_MSG_CAP);
        let after = messages_total_chars(&capped);
        let savings = before.saturating_sub(after);
        return (
            capped,
            before,
            after,
            savings >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS,
        );
    }

    // 所有路径均未达到有效压缩：返回最佳结果，did_summarize=false
    let result = best.unwrap_or(messages);
    (result, before, best_after, false)
}

/// Path C 兜底：对序列中单个超大非 system 消息做 head+tail 截断。
/// 仅在渐进式折叠后仍超 `hard_target` 时由 [`mid_turn_llm_summarize`] 调用。
/// system / agent 指令 / user 消息永不截断；图片按名义计费（≤ PATH_C_PER_MSG_CAP）天然跳过。
/// user 消息承载任务指令，截断成 head+tail 预览会导致 agent 丢失任务目标。
fn cap_oversized_non_system_messages(
    mut messages: Vec<Message>,
    per_msg_cap: usize,
) -> Vec<Message> {
    for message in &mut messages {
        if is_system_like_role(&message.role) || message.role == "user" {
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

/// 单条消息进入模型请求时的「计费字符数」——唯一权威口径。
///
/// 历史上多处预算只统计 `content`，把 `tool_calls[].function.arguments`
/// （典型 `apply_patch` 会把整份大补丁放进 arguments、content 为空）与
/// `reasoning_content`（thinking 模式的长思维链）完全漏算，导致大消息
/// 绕过压缩门控、TPM preflight 与 max_tokens clamp 一起低估输入。
///
/// 这里把三者合并计量，与 SQL 端 `total_message_chars_sqlite`
/// （`length(content)+length(tool_calls)+length(reasoning_content)`）对齐，
/// 使「内存态预算」与「持久化预算」共用同一口径。图片仍按
/// [`IMAGE_BUDGET_CHARS`] 名义计费（见 `value_len_chars`）。
pub(in crate::ai) fn message_billable_chars(m: &Message) -> usize {
    let content_chars = value_len_chars(&m.content);
    let tool_call_chars = m
        .tool_calls
        .as_ref()
        .map(|tc| {
            serde_json::to_string(tc)
                .map(|s| s.chars().count())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let reasoning_chars = m
        .reasoning_content
        .as_deref()
        .map(|s| s.chars().count())
        .unwrap_or(0);
    content_chars + tool_call_chars + reasoning_chars
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

fn is_summary_message(message: &Message) -> bool {
    if !is_system_like_role(&message.role) {
        return false;
    }
    is_summary_note_text(&value_to_string(&message.content))
}

/// 最近完整工具组的保护窗口。一个 assistant(tool_calls) 批次是不可拆分的证据
/// 单元：绝不能按单条 tool 消息截断，否则并行读取会只留下半批结果。
const KEEP_RECENT_TOOL_GROUPS: usize = 4;

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
/// 最近 KEEP_RECENT_TOOL_GROUPS 个完整工具组一律保留全文。
fn dedup_repeated_tool_results(messages: &mut [Message]) {
    use rustc_hash::{FxHashMap, FxHasher};
    use std::hash::{Hash, Hasher};

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

    let tool_indices = tool_message_indices(messages);
    let protected_indices = recent_tool_group_message_indices(messages, KEEP_RECENT_TOOL_GROUPS);

    // `read_file` 的 offset/limit 不同不会命中调用签名去重，但它们可能包含同一
    // 段文件。仅在两个结果都已离开近端保护窗口、同文件重叠行逐字一致时，才从较早
    // 结果删除重叠行；任一行不同（文件曾被编辑、输出格式变化等）即保持原样。
    dedup_overlapping_read_file_results(messages, &id_to_signature, &protected_indices);

    // (name, args) → 该签名下"首个保留全文"的 tool 消息序号，用于在折叠时回指。
    let mut seen: FxHashMap<(String, String), usize> = FxHashMap::default();
    // (name, args, content_hash) → 首个出现该内容版本的 tool 消息序号。
    // 内容级去重是断开"重复整篇重读"失忆环的关键：对 read_file 等
    // non-compressible 工具，同一 (文件, 参数) 被反复读取时往往返回**逐字节
    // 相同**的全文（实测占全部 tool 字节的 ~52%）。这些冗余副本可无损折叠，
    // 而内容确实变化的版本（如被编辑过的文件）因 hash 不同得以完整保留。
    let mut seen_content: FxHashMap<(String, String, u64), usize> = FxHashMap::default();
    for &idx in &tool_indices {
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
                // 字符预算。最近完整工具组的结果保留全文以防误伤；
                // 较旧的孤儿一律折叠为短 stub，避免阻塞后续压缩判断。
                if !protected_indices.contains(&idx) {
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
        if protected_indices.contains(&idx) {
            continue;
        }
        if is_non_compressible_tool(&signature.0) {
            // read_file/检索类工具**内容不同的版本**必须零压缩保留（Invariant：
            // precision 结果不做 lossy 裁剪）。但**逐字节相同**的重复副本是纯冗余，
            // 折叠它们不丢失任何信息，且能直接消除"旧全文堆积 + 近端 offload 触发
            // 重读"的失忆环。用内容 hash 区分二者：hash 首见 → 保留全文并登记；
            // hash 重现 → 折叠为回指首个全文的 stub（保留 tool_call_id 以维持协议）。
            let text = value_to_string(&messages[idx].content);
            let mut hasher = FxHasher::default();
            text.hash(&mut hasher);
            let content_key = (signature.0.clone(), signature.1.clone(), hasher.finish());
            match seen_content.get(&content_key).copied() {
                None => {
                    seen_content.insert(content_key, idx);
                }
                Some(_) => {
                    let stub = format!(
                        "[deduped: byte-identical `{}` result already present verbatim earlier in this conversation; content unchanged since then. No need to re-read — reuse the earlier full result.]",
                        signature.0
                    );
                    messages[idx].content = Value::String(stub);
                }
            }
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

#[derive(Clone)]
struct NumberedReadFileResult {
    message_idx: usize,
    path: String,
    lines: Vec<(usize, String)>,
}

fn dedup_overlapping_read_file_results(
    messages: &mut [Message],
    id_to_signature: &rustc_hash::FxHashMap<String, (String, String)>,
    protected_indices: &rustc_hash::FxHashSet<usize>,
) {
    let tool_indices = tool_message_indices(messages);
    let mut prior_results: Vec<NumberedReadFileResult> = Vec::new();

    for idx in tool_indices {
        let Some(tool_call_id) = messages[idx].tool_call_id.as_ref() else {
            continue;
        };
        let Some((tool_name, args)) = id_to_signature.get(tool_call_id) else {
            continue;
        };
        let Some(path) = read_file_path_from_args(tool_name, args) else {
            continue;
        };
        let text = value_to_string(&messages[idx].content);
        let Some(lines) = parse_numbered_read_file_lines(&text) else {
            continue;
        };

        // 近端完整工具组必须逐字保留，避免下一轮模型看到被处理过的刚读取内容。
        if protected_indices.contains(&idx) {
            prior_results.push(NumberedReadFileResult {
                message_idx: idx,
                path,
                lines,
            });
            continue;
        }

        for prior in &mut prior_results {
            if protected_indices.contains(&prior.message_idx) || prior.path != path {
                continue;
            }

            let overlapping = matching_line_numbers(&prior.lines, &lines);
            if overlapping.is_empty() {
                continue;
            }
            let removed = overlapping.len();
            prior
                .lines
                .retain(|(line_no, _)| !overlapping.contains(line_no));
            messages[prior.message_idx].content =
                Value::String(render_deduped_read_file_lines(&prior.lines, removed));
        }

        prior_results.push(NumberedReadFileResult {
            message_idx: idx,
            path,
            lines,
        });
    }
}

fn read_file_path_from_args(tool_name: &str, args: &str) -> Option<String> {
    if tool_name != "read_file" {
        return None;
    }
    serde_json::from_str::<Value>(args)
        .ok()?
        .get("file_path")?
        .as_str()
        .map(ToOwned::to_owned)
}

fn parse_numbered_read_file_lines(text: &str) -> Option<Vec<(usize, String)>> {
    let mut lines = Vec::new();
    for line in text.lines() {
        let (number, content) = line.split_once('\t')?;
        let number = number.trim().parse::<usize>().ok()?;
        lines.push((number, content.to_string()));
    }
    (!lines.is_empty()).then_some(lines)
}

/// 返回双方所有共有行号，前提是每一个共有行的内容均完全相同。
fn matching_line_numbers(
    earlier: &[(usize, String)],
    later: &[(usize, String)],
) -> rustc_hash::FxHashSet<usize> {
    let later_by_number: rustc_hash::FxHashMap<usize, &str> = later
        .iter()
        .map(|(number, content)| (*number, content.as_str()))
        .collect();
    let mut matching = rustc_hash::FxHashSet::default();
    for (number, content) in earlier {
        let Some(later_content) = later_by_number.get(number) else {
            continue;
        };
        if *later_content != content {
            return rustc_hash::FxHashSet::default();
        }
        matching.insert(*number);
    }
    matching
}

fn render_deduped_read_file_lines(lines: &[(usize, String)], removed: usize) -> String {
    if lines.is_empty() {
        return format!(
            "[overlap dedup: all {removed} numbered lines are present verbatim in a later read_file result]"
        );
    }
    let mut output = format!(
        "[overlap dedup: {removed} numbered lines are present verbatim in a later read_file result]\n"
    );
    for (number, content) in lines {
        output.push_str(&format!("{number:>6}\t{content}\n"));
    }
    output.pop();
    output
}

#[cfg(test)]
mod coalesce_summary_notes_tests;
#[cfg(test)]
mod fold_early_tool_groups_tests;
#[cfg(test)]
mod tail_window_tests;
