use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::ai::types::App;

use super::types::{
    MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE, is_internal_note_role,
    is_runtime_synthetic_user_message, is_system_like_role, last_real_user_index,
    retained_turn_start,
};

pub(crate) mod llm_prune;
mod text_utils;
mod tool_groups;
mod tool_overflow;

use text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
#[cfg(test)]
use tool_groups::{FOLDED_TOOL_GROUP_ARCHIVE_DIR, fold_early_tool_groups};
use tool_groups::{
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS, first_trim_candidate,
    is_protected_leading_system_like_message, plan_early_tool_groups,
    recent_tool_group_message_indices, recent_tool_result_groups,
};
#[cfg(test)]
use tool_overflow::normalize_internal_notes_for_summary_model;
use tool_overflow::{
    age_out_overflow_stub_previews, build_persisted_summary_text,
    build_persisted_summary_text_with_app, cap_oversized_tool_results_for_context,
    enforce_protected_precision_group_budget, is_non_compressible_tool,
    is_preserved_tool_overflow_content, is_preserved_user_or_image_stub,
    merge_old_user_overflow_stubs, minimize_overflow_stubs_for_hard_budget,
    normalize_preserved_message_stubs_for_model, prepare_tool_messages_structured,
    spill_oversized_preserved_messages, spill_protected_precision_to_fit, tool_line_signature,
    try_spill_preserved_message_to_stub,
};

/// 请求上下文中单条 raw tool result 的物理上限。canonical history 不受影响。
pub(in crate::ai) const TOOL_RESULT_RAW_HARD_CAP_CHARS: usize = 64_000;

pub(in crate::ai) fn cap_raw_tool_results_for_context(
    messages: &mut [Message],
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> usize {
    cap_oversized_tool_results_for_context(
        messages,
        TOOL_RESULT_RAW_HARD_CAP_CHARS,
        overflow_dir,
        cwd,
    )
}

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

/// 工具组折叠时生成的确定性证据 note 标记。
///
/// 这不是 LLM 生成的摘要，而是压缩器从 tool_call 参数和 tool 结果中机械提取的
/// evidence/checkpoint。它必须在二次摘要前保留，否则长工具链会退化成只有
/// file_path / original_file_path 的工具账单，模型压缩后容易重新取证。
pub(super) const COMPRESSED_TOOL_EVIDENCE_MARKER: &str = "[compressed-tool-evidence]";

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

/// 批量裁剪不能在执行中重算保护边界，因此低预算目标必须从一开始就采用
/// 48K 以下的三轮保护策略；否则第三近用户轮次可能在总量跨过 48K 前已被删除。
fn keep_recent_user_turns_for_batch(messages: &[Message], budget: usize) -> usize {
    let total_chars = messages_total_chars(messages);
    let mut keep = if budget > 0 && budget <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        3
    } else if total_chars <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        3
    } else {
        2
    };
    if budget > 0 {
        while keep > 1 {
            let tail_start = retained_turn_start(messages, keep);
            if messages_total_chars(&messages[tail_start..]) <= budget {
                break;
            }
            keep -= 1;
        }
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
/// 旧工具组的机械证据在模型上下文中逐字保留的总字符上限。
/// 更早的证据会零压缩追加到 overflow-history.md，只在 messages 中保留统一回指。
const MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS: usize = 12_000;
const CONTEXT_CHECKPOINT_MARKER_PREFIX: &str = "[context_checkpoint";

pub(in crate::ai) fn compressed_tool_evidence_inline_chars_limit() -> usize {
    MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS
}

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

pub(super) fn is_compressed_tool_evidence_note(m: &Message) -> bool {
    m.role == ROLE_INTERNAL_NOTE
        && value_to_string(&m.content)
            .trim_start()
            .contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
}

const PERSISTED_HISTORY_SUMMARY_MAX_CHARS: usize = 8_000;
const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";
const INTERNAL_NOTE_OVERFLOW_DIR: &str = "internal-note-overflow";
const PRESERVED_TOOL_OVERFLOW_DIR: &str = "tool-overflow-compressed";
const PRESERVED_USER_OVERFLOW_DIR: &str = "user-overflow-preserved";
const PRESERVED_IMAGE_OVERFLOW_DIR: &str = "image-overflow-preserved";
const PRESERVED_CONTENT_STUB_PREFIX: &str = "[[PRESERVED_CONTENT_STUB_V1]]";
const USER_OVERFLOW_SPILL_MIN_CHARS: usize = 1_024;
const IMAGE_OVERFLOW_SPILL_MIN_CHARS: usize = 512;

#[derive(Clone)]
pub(super) struct PlannedArchiveWrite {
    path: PathBuf,
    content: String,
}

impl PlannedArchiveWrite {
    pub(super) fn new(path: PathBuf, content: String) -> Self {
        Self { path, content }
    }

    /// 把规划阶段的归档原子落盘。路径包含内容指纹，因此已存在文件可直接复用；
    /// 并发 writer 即使同时写临时文件，最终也只会留下同一份确定性目标文件。
    pub(super) fn commit(&self) -> bool {
        if self.path.is_file() {
            return true;
        }
        let Some(parent) = self.path.parent() else {
            return false;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
        let Some(file_name) = self.path.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        let temporary = parent.join(format!(
            ".{file_name}.{}.tmp",
            uuid::Uuid::new_v4().simple()
        ));
        let result = (|| -> std::io::Result<()> {
            use std::io::Write;

            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(self.content.as_bytes())?;
            file.sync_data()?;
            std::fs::rename(&temporary, &self.path)
        })();
        if result.is_ok() || self.path.is_file() {
            let _ = std::fs::remove_file(&temporary);
            return true;
        }
        let _ = std::fs::remove_file(temporary);
        false
    }
}

pub(super) fn content_sha256_hex(content: &[u8]) -> String {
    Sha256::digest(content)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) struct OverflowSink {
    path: PathBuf,
    buffer: String,
    /// Offset just past the leading file header / batch separator written by
    /// `start_batch`; everything from here on is the stable batch payload used
    /// for sha256 fingerprint dedup in `flush`.
    payload_offset: usize,
    /// Hex sha256 fingerprints of payloads already known to be archived, kept
    /// as a persistent sidecar index next to the history file. Checking this
    /// tiny index instead of scanning the ever-growing archive keeps every
    /// flush O(payload) regardless of archive size. Entries are recorded only
    /// AFTER their payload landed in the archive, so a crash between the two
    /// degrades to one duplicate append rather than dropping evidence.
    seen_payloads: rustc_hash::FxHashSet<String>,
    seen_payloads_loaded: bool,
}

impl OverflowSink {
    pub(super) fn new(overflow_dir: &Path) -> Self {
        let path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
        Self {
            path,
            buffer: String::new(),
            payload_offset: 0,
            seen_payloads: rustc_hash::FxHashSet::default(),
            seen_payloads_loaded: false,
        }
    }

    /// Sidecar next to the history file listing hex sha256 digests of every
    /// payload previously appended there, one per line.
    fn fingerprint_index_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.fingerprints", self.path.to_string_lossy()))
    }

    fn start_batch(&mut self, title: &str) {
        if self.buffer.is_empty() {
            // Write the header only while the archive file does not exist yet;
            // later flushes run in append mode and add new batches only.
            // Each batch is preceded by a separator line so humans and tools can
            // read the file in blocks.
            if !self.path.exists() {
                self.buffer.push_str(
                    "# Overflow History Archive\n\nThe content below was moved out of the context window; use the read_file tool to read this file back when earlier context is needed.\n\n---\n\n",
                );
            } else {
                self.buffer.push_str("\n---\n\n");
            }
            self.payload_offset = self.buffer.len();
        }
        if !title.trim().is_empty() {
            self.buffer.push_str("## ");
            self.buffer.push_str(title.trim());
            self.buffer.push_str("\n\n");
        }
    }

    pub(super) fn push_messages(&mut self, messages: &[Message]) {
        if messages.is_empty() {
            return;
        }
        self.start_batch("Removed messages (verbatim)");
        for msg in messages {
            let text = value_to_string(&msg.content);
            match msg.role.as_str() {
                "user" => {
                    self.buffer.push_str("## User\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
                "assistant" => {
                    self.buffer.push_str("## Assistant\n\n");
                    self.buffer.push_str(&text);
                    self.buffer.push_str("\n\n");
                }
                "tool" => {
                    self.buffer.push_str("### Tool result\n\n");
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
            self.push_raw_message_json(msg);
        }
    }

    fn push_raw_message_json(&mut self, msg: &Message) {
        let Ok(json) = serde_json::to_string_pretty(msg) else {
            return;
        };
        self.buffer.push_str("raw_message_json:\n```json\n");
        self.buffer.push_str(&json);
        self.buffer.push_str("\n```\n\n");
    }

    fn push_truncated_field(&mut self, message: &Message, field: MutableMessageField) -> bool {
        let Some(original) = field.original_text(message) else {
            return false;
        };
        self.start_batch("Truncated field original text");
        self.buffer.push_str("### Field original text\n\n");
        self.buffer.push_str("- role: ");
        self.buffer.push_str(&message.role);
        self.buffer.push('\n');
        if let Some(tool_call_id) = message.tool_call_id.as_deref() {
            self.buffer.push_str("- tool_call_id: ");
            self.buffer.push_str(tool_call_id);
            self.buffer.push('\n');
        }
        self.buffer.push_str("- field: ");
        self.buffer.push_str(&field.archive_label());
        self.buffer.push_str("\n\n");
        self.buffer.push_str("Begin original text\n");
        self.buffer.push_str(&original);
        self.buffer.push_str("\nEnd original text\n\n");
        self.push_raw_message_json(message);
        true
    }

    pub(super) fn flush(&mut self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Append mode keeps every historical batch forever. Repeated reactive
        // rescues rebuild the same projection from unchanged canonical history,
        // so they can re-archive byte-identical truncated-field payloads each
        // round and grow this file once per round. Dedup consults the
        // persistent fingerprint index instead of scanning the archive body:
        // a hit skips the write yet still reports success — callers only rely
        // on the archived bytes being readable back from file_path(). Index
        // load failures merely re-enable plain appends, the pre-dedup
        // behavior.
        let payload_start = self.payload_offset.min(self.buffer.len());
        let payload = &self.buffer[payload_start..];
        if payload.is_empty() {
            // Only the reusable header/separator is pending.
            return true;
        }
        if !self.seen_payloads_loaded {
            self.seen_payloads_loaded = true;
            let fingerprints = std::fs::read_to_string(self.fingerprint_index_path())
                .map(|text| {
                    text.lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.seen_payloads.extend(fingerprints);
        }
        let digest = content_sha256_hex(payload.as_bytes());
        if self.seen_payloads.contains(&digest) {
            return true;
        }
        use std::io::Write;
        // Append mode: a later compress pass must never wipe what earlier
        // passes archived. File::create would truncate the file, leaving only
        // the final pass's batches and degrading long-term memory into
        // short-term memory across multi-pass sessions.
        let appended = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| {
                f.write_all(self.buffer.as_bytes())?;
                f.sync_data()
            })
            .is_ok();
        if !appended {
            return false;
        }
        // Record the fingerprint only after the bytes above are durably in the
        // archive; an index-first ordering could make a crash silently drop
        // the next identical payload while readers still expect its archived
        // copy. Best effort: losing an index entry just costs one duplicate
        // append later.
        self.seen_payloads.insert(digest.clone());
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.fingerprint_index_path())
            .and_then(|mut f| {
                writeln!(f, "{digest}")?;
                f.sync_data()
            });
        true
    }

    pub(super) fn file_path(&self) -> &Path {
        &self.path
    }
}

pub(super) fn archive_messages_to_overflow(
    messages: &[Message],
    overflow_dir: Option<&Path>,
) -> Option<String> {
    let dir = overflow_dir?;
    let mut sink = OverflowSink::new(dir);
    sink.push_messages(messages);
    sink.flush()
        .then(|| sink.file_path().to_string_lossy().to_string())
}

/// Internal notes may hold folded evidence, recovery instructions, or runtime
/// persisted state, so they must never be dropped silently. Each note is written
/// to its own content-addressed file, so repeated compression rounds reuse the
/// same path instead of appending the same body again.
fn archive_internal_notes_deduplicated(
    messages: &[Message],
    overflow_dir: Option<&Path>,
) -> Result<Option<PathBuf>, ()> {
    let notes: Vec<&Message> = messages
        .iter()
        .filter(|message| message.role == ROLE_INTERNAL_NOTE)
        .collect();
    if notes.is_empty() {
        return Ok(None);
    }
    let Some(root) = overflow_dir else {
        return Err(());
    };
    let archive_dir = root.join(INTERNAL_NOTE_OVERFLOW_DIR);
    for message in notes {
        let raw_json = serde_json::to_string_pretty(message).map_err(|_| ())?;
        let body = format!(
            "# Internal context note (verbatim)\n\n{}\n\nraw_message_json:\n```json\n{}\n```\n",
            value_to_string(&message.content),
            raw_json
        );
        let digest = content_sha256_hex(body.as_bytes());
        let write = PlannedArchiveWrite::new(archive_dir.join(format!("{digest}.md")), body);
        if !write.commit() {
            return Err(());
        }
    }
    Ok(Some(archive_dir))
}

fn insert_internal_note_archive_note_if_needed(
    messages: &mut Vec<Message>,
    archive_dir: Option<&Path>,
) {
    let Some(archive_dir) = archive_dir else {
        return;
    };
    let archive_note = format!(
        "{ARCHIVE_NOTE_PREFIX}\n较早的内部上下文注记已逐字归档。\n归档目录: {}\n需要回顾时使用 search_overflow（scope=all）定位，再用 read_file 精读文件。",
        archive_dir.to_string_lossy()
    );
    insert_archive_note_if_missing(messages, archive_note);
}

fn archive_truncated_field_to_overflow(
    message: &Message,
    field: MutableMessageField,
    overflow_dir: Option<&Path>,
) -> Option<String> {
    let dir = overflow_dir?;
    let mut sink = OverflowSink::new(dir);
    if !sink.push_truncated_field(message, field) {
        return None;
    }
    sink.flush()
        .then(|| sink.file_path().to_string_lossy().to_string())
}

fn insert_overflow_archive_note_if_exists(
    messages: &mut Vec<Message>,
    overflow_dir: Option<&Path>,
) {
    let Some(dir) = overflow_dir else {
        return;
    };
    let archive_path = dir.join(OVERFLOW_HISTORY_FILENAME);
    if archive_path.is_file() {
        let archive_note = build_overflow_placeholder(&archive_path.to_string_lossy());
        insert_archive_note_if_missing(messages, archive_note);
    }
}

pub(in crate::ai) fn compressed_tool_evidence_exceeds_inline_budget(messages: &[Message]) -> bool {
    messages
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .map(message_billable_chars)
        .sum::<usize>()
        > MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS
}

/// 工具组折叠 note 是旧证据的短期召回窗口，而不是永久逐条内联的账本。
/// 保留能装入固定字符预算的最近连续窗口；更早 note 先零压缩归档，写入成功后
/// 才从 messages 删除。这样长工具链不会用上百条约 1 KiB 的 note 挤占上下文。
fn trim_compressed_tool_evidence_to_inline_budget(
    mut messages: Vec<Message>,
    overflow_dir: Option<&Path>,
) -> Vec<Message> {
    if !compressed_tool_evidence_exceeds_inline_budget(&messages) {
        return messages;
    }
    let Some(overflow_dir) = overflow_dir else {
        return messages;
    };

    let evidence_sizes: Vec<usize> = messages
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .map(message_billable_chars)
        .collect();
    let mut keep_from = evidence_sizes.len();
    let mut kept_chars = 0usize;
    for (index, chars) in evidence_sizes.iter().enumerate().rev() {
        if keep_from == evidence_sizes.len()
            || kept_chars.saturating_add(*chars) <= MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS
        {
            keep_from = index;
            kept_chars = kept_chars.saturating_add(*chars);
        } else {
            break;
        }
    }
    if keep_from == 0 {
        return messages;
    }

    let dropped: Vec<Message> = messages
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .take(keep_from)
        .cloned()
        .collect();
    let mut sink = OverflowSink::new(overflow_dir);
    sink.push_messages(&dropped);
    if !sink.flush() {
        return messages;
    }

    let mut evidence_ordinal = 0usize;
    messages.retain(|message| {
        if !is_compressed_tool_evidence_note(message) {
            return true;
        }
        let keep = evidence_ordinal >= keep_from;
        evidence_ordinal += 1;
        keep
    });
    let archive_note = build_overflow_placeholder(&sink.file_path().to_string_lossy());
    insert_archive_note_if_missing(&mut messages, archive_note);
    messages
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
    cwd: Option<&Path>,
) -> Vec<Message> {
    // 历史库中可能仍有旧版 JSON stub。它是压缩器的内部协议，不能原样交给模型，
    // 否则模型会把它当普通用户文本甚至直接复述到最终回复中。
    normalize_preserved_message_stubs_for_model(&mut messages);
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    // compressed_tool_round note 本身也是压缩产物；若不设独立上限，它们会在
    // 全局 history 预算触发前逐条累积，形成另一种线性上下文膨胀。
    messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir.as_deref());

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
            cwd,
            &rustc_hash::FxHashSet::default(),
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
            cwd,
            &rustc_hash::FxHashSet::default(),
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
            .filter(|message| {
                // 摘要预算为 0（生产路径第二轮压缩等场景）时不会重建摘要；旧的摘要/
                // 归档 note 本身就是"早期对话的压缩表示"，必须与 checkpoint marker
                // 一样保留，否则 prepare_turn 已生成的摘要会在第二轮压缩中被静默丢弃。
                is_context_checkpoint_marker(message)
                    || (summary_max_chars == 0 && is_summary_or_archive_note(message))
            })
            .cloned(),
    );
    out.extend_from_slice(recent);
    shrink_messages_to_fit_with_summary(
        out,
        max_chars,
        summary_max_chars,
        overflow_dir.as_deref(),
        cwd,
        &rustc_hash::FxHashSet::default(),
    )
}

/// 持久化历史里"带 tool_calls 的 assistant narration"被截断到的字符数。
///
/// 折叠器 [`tool_groups::fold_tool_call_group_to_stub`] 把发起本轮工具调用前
/// 的可见 narration 当作 `assistant_checkpoint` 的来源。除模型协议明确要求回放的
/// continuation state 外，完整 reasoning_content 不落盘，也绝不能提升为 assistant
/// 正文；tool-call-only 消息由折叠器根据结构化 tool_calls 重建安全的操作摘要。
/// 720 字与折叠后的 checkpoint 上限同量级。
const PERSISTED_TOOL_CALL_ASSISTANT_NARRATION_MAX_CHARS: usize = 720;
pub(in crate::ai) const PERSISTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-content-replay:v1\u{1f}";

/// exact-replay continuation state 只存在于可重建的上下文投影中。payload 带来源模型，
/// 因此切换模型时不会把另一个 provider 的隐藏状态误当成当前模型的续传状态。
pub(in crate::ai) fn encode_reasoning_replay_state(model: &str, reasoning: &str) -> String {
    format!(
        "{PERSISTED_REASONING_REPLAY_PREFIX}{}",
        serde_json::json!({ "model": model, "reasoning": reasoning })
    )
}

pub(in crate::ai) fn decode_reasoning_replay_for_model(
    model: &str,
    encoded: &str,
) -> Option<String> {
    let payload = encoded.strip_prefix(PERSISTED_REASONING_REPLAY_PREFIX)?;
    let payload: Value = serde_json::from_str(payload).ok()?;
    (payload.get("model")?.as_str()? == model)
        .then(|| payload.get("reasoning")?.as_str().map(str::to_owned))?
}

/// Responses 协议加密推理回放前缀。与 exact-replay（`PERSISTED_REASONING_REPLAY_PREFIX`）
/// 分开，因为二者 payload 形状不同：exact 存明文 reasoning 字符串，加密存 provider
/// 下发的 reasoning output-item（JSON 数组，含 `encrypted_content`）。分开前缀避免
/// 请求端把两类 payload 混用、也让压缩/sanitize 层能用同一"带标记则保留"规则统一处理。
pub(in crate::ai) const PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-encrypted-replay:v1\u{1f}";

/// 加密推理跨轮回放的运行时总开关。默认开启；设 `AIOS_DISABLE_ENCRYPTED_REPLAY=1`
/// 时短路落库与请求端重建，用于对照实验复现"修复前"行为（跨轮/resume 丢失加密推理）。
/// 仅作实验脚手架，不改变默认产品行为。
pub(in crate::ai) fn encrypted_reasoning_replay_runtime_enabled() -> bool {
    std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY")
        .map(|v| v.trim().is_empty() || v == "0")
        .unwrap_or(true)
}

/// 把本轮捕获的加密 reasoning items 连同来源模型编码进单个字符串，供落库到
/// `reasoning_content`。带 model 标记：切换/回退到其他模型时，请求端解码会因
/// 模型不匹配而丢弃，避免把 A 模型的加密状态误喂给 B 模型（provider 会 400）。
pub(in crate::ai) fn encode_encrypted_reasoning_replay_state(
    model: &str,
    items: &[Value],
) -> String {
    format!(
        "{PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX}{}",
        serde_json::json!({ "model": model, "items": items })
    )
}

/// 从落库的 `reasoning_content` 解码出加密 reasoning items。仅当标记内的来源模型
/// 与当前请求模型一致时才返回；否则返回 `None`（跨模型不回放）。
pub(in crate::ai) fn decode_encrypted_reasoning_replay_for_model(
    model: &str,
    encoded: &str,
) -> Option<Vec<Value>> {
    let payload = encoded.strip_prefix(PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)?;
    let payload: Value = serde_json::from_str(payload).ok()?;
    if payload.get("model")?.as_str()? != model {
        return None;
    }
    let mut items: Vec<Value> = payload.get("items")?.as_array()?.to_vec();
    // 网关会对同一 reasoning 资源在 `.added`（部分载荷）与 `.done`（完整载荷）
    // 重复下发，修复前的累积器按全字段相等去重未能收敛，历史里可能因此落库
    // 同 id 的两项。按 id 去重、保留后到（`.done` 为协议最终权威状态）的一项，
    // 否则回放时同一资源 id 出现两次，modelhub 返回 400 (-4003 Duplicate item found)。
    dedup_reasoning_items_by_id(&mut items);
    Some(items)
}

/// 按 `id` 收敛 reasoning items：同一资源保留后到的一项。
///
/// 网关会对同一 reasoning 资源在 `.added`（部分载荷）与 `.done`（完整载荷）
/// 重复下发，二者 id 相同、内容不同，按全字段相等去重判不等，会留下重复 id；
/// 回放时同一资源 id 出现两次，modelhub 返回 400 (-4003 Duplicate item found)。
/// 这里按 id 收敛、保留后到的一项：流内 `.done` 恒晚于 `.added`，是协议最终
/// 权威状态（携带完整载荷），后到者胜天然选中它。无 `id` 的项互不合并（保留
/// 全部，避免误删）。
pub(in crate::ai) fn dedup_reasoning_items_by_id(items: &mut Vec<Value>) {
    let mut deduped: Vec<Value> = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        let id = item.get("id").cloned();
        match deduped
            .iter_mut()
            .find(|existing| id.is_some() && existing.get("id") == id.as_ref())
        {
            Some(existing) => *existing = item,
            None => deduped.push(item),
        }
    }
    *items = deduped;
}

fn sanitize_message_for_persisted_history_inner(
    message: &Message,
    replay_source_model: Option<&str>,
) -> Message {
    let mut sanitized = message.clone();
    if sanitized.role != "assistant" {
        return sanitized;
    }

    // 持久化历史只保留跨 turn 真正需要的 assistant 事实：
    // - `reasoning_content` 是隐藏推理，持久化层一律丢弃，绝不复制到可见正文；
    //   provider 需要字段形状时由 request 层统一补空字符串。
    // - 带 tool_calls 的 assistant narration 不能清空：否则
    //   [`tool_groups::fold_tool_call_group_to_stub`] 的 checkpoint 看不到任何文字，只能塌成
    //   "assistant_checkpoint: <empty; no persisted decision before these tool calls>"，
    //   压缩后模型失忆，从同一轮重启取证。
    //
    if sanitized
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty())
    {
        let narration = match &sanitized.content {
            Value::Null => String::new(),
            Value::String(text) => text.clone(),
            other => value_to_string(other),
        };
        let capped = truncate_to_chars(
            &narration,
            PERSISTED_TOOL_CALL_ASSISTANT_NARRATION_MAX_CHARS,
        );
        sanitized.content = Value::String(capped);
    }
    let has_tool_calls = sanitized
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty());
    if has_tool_calls {
        if let Some(reasoning) = sanitized.reasoning_content.as_mut() {
            if reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX)
                || reasoning.starts_with(PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)
            {
                // 已经是带内部标记的连续性状态（exact 明文 / responses 加密），保持幂等。
            } else if let Some(model) = replay_source_model {
                *reasoning = encode_reasoning_replay_state(model, reasoning);
            } else {
                sanitized.reasoning_content = None;
            }
        }
    } else {
        sanitized.reasoning_content = None;
    }
    sanitized
}

pub(in crate::ai) fn sanitize_message_for_persisted_history(message: &Message) -> Message {
    sanitize_message_for_persisted_history_inner(message, None)
}

/// 按模型协议生成持久化投影。只有显式声明需要原样回放的模型，才会为
/// tool-call assistant 保留隐藏推理；最终回答等其他消息仍一律删除。
pub(in crate::ai) fn sanitize_message_for_persisted_history_for_model(
    model: &str,
    message: &Message,
) -> Message {
    let replay_source_model =
        crate::ai::models::reasoning_content_replay_enabled(model).then_some(model);
    sanitize_message_for_persisted_history_inner(message, replay_source_model)
}

fn sanitize_persisted_history_messages(messages: Vec<Message>) -> Vec<Message> {
    let messages = coalesce_accumulated_summary_notes(messages);
    messages
        .into_iter()
        // 只有带内部标记的 reasoning 才是运行时按模型能力显式保留的连续性状态；
        // 旧历史、导入文件和其他模型的裸 reasoning 仍按原策略删除。
        .map(|message| {
            let preserve = message.reasoning_content.as_deref().is_some_and(|reasoning| {
                reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX)
                    || reasoning.starts_with(PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)
            });
            sanitize_message_for_persisted_history_inner(
                &message,
                preserve.then_some("already-tagged"),
            )
        })
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
/// - **归档指针 note**：内容完全相同的只保留一条，内容不同的全部保留，紧跟
///   合并后的摘要，避免导入/迁移会话时丢失指向其他归档文件的回指。
/// - 其余消息一律原样保留、顺序不变（绝不触碰非摘要/归档消息）。
///
/// 仅当摘要超过一条或存在内容完全相同的归档指针时才折叠，避免对正常历史做
/// 无谓改写（返回值与入参逐条相等时，上层 `compacted == messages` 判定会跳过落盘）。
fn coalesce_accumulated_summary_notes(messages: Vec<Message>) -> Vec<Message> {
    let summary_count = messages.iter().filter(|m| is_summary_message(m)).count();
    let mut seen_archive_texts = rustc_hash::FxHashSet::default();
    let has_duplicate_archive = messages
        .iter()
        .filter(|m| is_archive_note_message(m))
        .map(|m| value_to_string(&m.content))
        .any(|text| !seen_archive_texts.insert(text));
    if summary_count <= 1 && !has_duplicate_archive {
        return messages;
    }

    // 合并所有摘要正文，并对内容完全相同的归档指针去重；两者都保持原顺序。
    let mut merged_bodies: Vec<String> = Vec::new();
    let mut first_summary_role: Option<String> = None;
    let mut archive_notes: Vec<Message> = Vec::new();
    let mut seen_archive_texts = rustc_hash::FxHashSet::default();
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
        } else if is_archive_note_message(m) {
            let text = value_to_string(&m.content);
            if seen_archive_texts.insert(text) {
                archive_notes.push(m.clone());
            }
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

    // 重建序列：在"第一条摘要/归档 note"的位置放入合并摘要 + 去重后的归档指针，
    // 丢弃其余摘要/归档 note，其他消息原样保留。
    let mut out = Vec::with_capacity(messages.len());
    let mut inserted = false;
    for m in messages {
        if is_summary_or_archive_note(&m) {
            if !inserted {
                if let Some(summary) = merged_summary.clone() {
                    out.push(summary);
                }
                out.extend(archive_notes.iter().cloned());
                inserted = true;
            }
            // 其余摘要及已收集的归档 note 丢弃。
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

/// 在 leading summary 后注入归档回指；相同回指已存在时保持幂等，避免每轮压缩
/// 都在上下文头部追加一条完全相同的 `internal_note`。
fn insert_archive_note_if_missing(messages: &mut Vec<Message>, archive_note: String) {
    let already_present = messages.iter().any(|message| {
        is_archive_note_message(message) && value_to_string(&message.content) == archive_note
    });
    if already_present {
        return;
    }

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

pub(in crate::ai) fn compact_persisted_history(messages: Vec<Message>) -> Vec<Message> {
    let messages = sanitize_persisted_history_messages(messages);
    let user_turns = messages
        .iter()
        .filter(|message| {
            // 合成的 user 消息（图片 followup 等）不构成真实轮次，避免提前截断历史。
            message.role == "user" && !is_runtime_synthetic_user_message(message)
        })
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
        .filter(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
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

/// 当 `first_tool_call_group` 折不动（剩余可折叠组都含 `read_file`
/// 等 non-compressible 工具、被它按策略拒绝）但仍超预算时的下一档手段：用
/// [`fold_early_tool_groups`] 递进折叠「保护尾窗之外」的这些组为单行
/// `compressed_tool_round` note（内含 file_path 召回锚点，模型可 read_file 回读）。
///
/// 这与 `mid_turn_llm_summarize` 的 Path B+C 复用**同一个**久经测试的折叠函数，
/// 只是把它前移到常规/落盘压缩路径——修复「tool-heavy 会话（少 user 轮 × 上百次
/// read_file）在 `compress_messages_for_context` / `shrink_*` 里永远
/// 压不掉工具组、整段历史无法收敛进预算」的总根因。
///
/// 返回是否发生了「有效折叠」（净字符数下降）。`keep_recent` 从
/// [`KEEP_RECENT_TOOL_GROUPS`] 递进收紧到 [`MIN_KEEP_RECENT_TOOL_GROUPS`]（=1），
/// 保证最近的工具组尽量逐字保留、只有仍超预算才逐步放宽折叠范围；每一步都要求
/// 净下降，避免无进展空转。**不再收紧到 0**：窗口降到 0 会把最近一次工具交互也
/// 折叠成 stub，模型因此完全失去最近的结构化工具上下文；剩余超额交由 while 循环
/// 后续的 `first_trim_candidate` / `truncate_mutable_messages_to_fit` 兜底。
fn fold_noncompressible_tool_groups_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    if messages_total_chars(messages) <= max_chars {
        return false;
    }
    // Select ONE window from the descending-protection ladder instead of
    // committing every intermediate rung:
    //   1. the most protective window whose result both fits and keeps the
    //      verbatim tail at or above MIN_PROTECTED_TAIL_CHARS,
    //   2. otherwise the most protective window that merely fits - overflow
    //      matters more than the floor once the floor cannot hold,
    //   3. otherwise the deepest reducing window, so bounded progress and the
    //      historical deep-fold endpoint are preserved even when nothing fits.
    // Plans are pure functions of the messages, so candidates are recorded as
    // window sizes and re-planned exactly once before committing.
    let mut floor_safe_fitting_keep: Option<usize> = None;
    let mut fitting_keep: Option<usize> = None;
    let mut deepest_reducing_keep: Option<usize> = None;
    for &keep_recent in progressive_fold_windows().iter() {
        let plan =
            plan_early_tool_groups(messages, keep_recent, overflow_dir, protected_tool_call_ids);
        if plan.folded_groups() == 0 {
            continue;
        }
        // A plan must net a strict decrease; drop it otherwise and keep tightening
        // keep_recent to guard against livelock where the group count changes but
        // the char count does not.
        if messages_total_chars(plan.messages()) >= messages_total_chars(messages) {
            continue;
        }
        deepest_reducing_keep = Some(keep_recent);
        if messages_total_chars(plan.messages()) <= max_chars {
            if fitting_keep.is_none() {
                fitting_keep = Some(keep_recent);
            }
            if protected_tail_message_chars(plan.messages(), keep_recent)
                >= MIN_PROTECTED_TAIL_CHARS
            {
                floor_safe_fitting_keep = Some(keep_recent);
                break;
            }
        }
    }
    let chosen = floor_safe_fitting_keep
        .or(fitting_keep)
        .or(deepest_reducing_keep);
    let mut made_progress = false;
    if let Some(keep_recent) = chosen {
        let plan =
            plan_early_tool_groups(messages, keep_recent, overflow_dir, protected_tool_call_ids);
        if plan.folded_groups() > 0
            && messages_total_chars(plan.messages()) < messages_total_chars(messages)
            && plan.commit()
        {
            let (folded, _) = plan.into_result();
            *messages = folded;
            made_progress = true;
        }
    }
    made_progress
}

/// Billable chars across the tool-result messages of the most-recent
/// `keep_recent` complete tool groups - i.e. the verbatim structured evidence
/// kept outside folding under that window size (assistant anchors excluded,
/// matching what recent_tool_group_message_indices returns).
fn protected_tail_message_chars(messages: &[Message], keep_recent: usize) -> usize {
    recent_tool_group_message_indices(messages, keep_recent)
        .into_iter()
        .map(|idx| message_billable_chars(&messages[idx]))
        .sum()
}

/// 批量移除可裁的普通消息，并在一次 flush 中归档。旧实现每删一条就重新进入外层
/// 循环并 `sync_data`，tool-heavy 历史会把数百条 assistant 消息放大成数百次同步
/// 写。这里先在候选副本上完成整批裁剪；归档失败时不采纳候选，原消息保持不变。
fn trim_removable_messages_batch(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    // 单趟扫描 + 重建，替代旧实现「每轮 first_trim_candidate + Vec::remove」：
    // 旧循环每轮都会全量重扫（keep_recent_user_turns_when_trimming /
    // retained_turn_start / 头部保护 run 各 O(n)），删除又是 O(n) memmove，
    // 整体 O(n²)，数千条 tool-heavy 历史会明显卡顿。这里把保护尾窗与字符总量
    // 在移除前一次性算好，之后只做 O(n) 扫描 + O(n) 重建。
    let keep_recent_user_turns = keep_recent_user_turns_for_batch(messages, max_chars);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    let mut total = messages_total_chars(messages);
    if total <= max_chars {
        return false;
    }

    let candidate = messages.clone();
    let mut removed = Vec::new();
    let mut kept = Vec::with_capacity(candidate.len());
    let mut index = 0usize;
    let mut in_protected_leading_run = true;
    for message in candidate {
        // 头部受保护的 system-like run（系统提示词、历史摘要、归档指针、checkpoint）
        // 整段跳过，与 first_trim_candidate 语义一致。
        let head_protected =
            in_protected_leading_run && is_protected_leading_system_like_message(&message);
        if head_protected {
            kept.push(message);
            index += 1;
            continue;
        }
        in_protected_leading_run = false;

        // 与 first_trim_candidate 的可删判定一致：checkpoint / 外溢 stub / tool /
        // assistant(tool_calls) 均不可单删。user 在此路径不可删（OffloadOnly，只能
        // 外溢）——跳过而不是 break：避免首个 user 之后的大量可裁候选失去批量移除
        // 机会（旧行为 break 后只能靠 truncate 兜底，与 with_summary 的「丢弃 +
        // 归档」语义不一致）。字符总量精确维护：每删一条减去 message_billable_chars，
        // 一旦 total <= max_chars 即停止移除，与旧循环的停止条件一致。
        let removable = index < protected_tail_start
            && !is_context_checkpoint_marker(&message)
            && !is_preserved_user_or_image_stub(&value_to_string(&message.content))
            && message.role != "tool"
            && !(message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .map(|c| !c.is_empty())
                    .unwrap_or(false))
            && message.role != "user";
        if removable && total > max_chars {
            total = total.saturating_sub(message_billable_chars(&message));
            removed.push(message);
        } else {
            kept.push(message);
        }
        index += 1;
    }
    if removed.is_empty() {
        return false;
    }
    // 普通消息仍追加到统一历史归档；internal_note 单独按内容指纹写入确定性文件。
    // 后者既避免静默丢失恢复指令/持久状态，也避免重复压缩时 append 同一正文。
    let archive_candidates: Vec<Message> = removed
        .iter()
        .filter(|m| !is_internal_note_role(&m.role))
        .cloned()
        .collect();
    let internal_archive_dir = match archive_internal_notes_deduplicated(&removed, overflow_dir) {
        Ok(path) => path,
        Err(()) => return false,
    };
    let archive_ok = match overflow_dir {
        Some(dir) if !archive_candidates.is_empty() => {
            archive_messages_to_overflow(&archive_candidates, Some(dir)).is_some()
        }
        _ => true,
    };
    if !archive_ok {
        return false;
    }
    *messages = kept;
    insert_internal_note_archive_note_if_needed(messages, internal_archive_dir.as_deref());
    true
}

fn shrink_messages_to_fit(
    mut messages: Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
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
    dedup_repeated_tool_results(&mut messages, protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut messages,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
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
    // user/image 外溢 stub 没有 tool 锚点可老化：其预览本身就是单行指针，且
    // first_trim_candidate / truncate / emergency cap 都不会再触碰它们，长会话
    // 会让 stub 单调累积（尤其图片消息 512 阈值 < 名义成本时）。把保护尾窗之外
    // 的旧 stub 合并成一条带归档目录的指针，占位开销从 O(N) 收敛到 O(1)。
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // 主动精简「已成功写入」的 write_file/apply_patch 巨型 arguments：文件落盘、
    // 结果确认成功后全文不再有语义价值，无需等预算压力即可先替换为归档 stub。
    // 保护窗口内（含当前轮刚写完、模型可能立即引用正文构造后续编辑的组）与失败
    // 结果一律保留，保证 agent 效果不劣化。
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        // 一次性批量折叠超出预算的所有非保护工具组（compressible + non-compressible
        // 都通过 [`fold_early_tool_groups`] 处理）。
        // 旧实现在 `first_tool_call_group` + 单组 fold 循环里一次只折一组，且只在
        // 折无可折后才落到 `fold_noncompressible_tool_groups_to_fit` 的批 fold。Bug A
        // 让单组 fold 的字符节省极小（assistant.content 已被 sanitize 置
        // `""`/`null`，fold 出的 stub 几乎与原 group 同大）→ 外层 while 要迭代几十
        // 轮才收敛，每轮再注入一条 `<empty>` empty-checkpoint note 污染上下文（详
        // e75fc2e5 session dump 的 22 个连续 `compressed_tool_round` <empty> stub）。
        // 改成每轮优先用一个 `fold_early_tool_groups` 批把所有可折叠组一次性收掉，
        // 让收缩在数外层迭代内完成。
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
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
            // 其余可裁候选（assistant 纯叙述、compressed_tool_round 等）集中归档，
            // 避免逐条 append + sync_data。批量归档失败时原消息保持不变。
            if trim_removable_messages_batch(&mut messages, max_chars, overflow_dir) {
                continue;
            }
            break;
        }
        break;
    }

    // 裁剪 compressed_tool_evidence 时，正文会零压缩追加到统一历史归档；必须把
    // 统一回指重新放回请求，否则磁盘上虽有证据，模型却不知道归档路径。归档 note
    // 是 internal_note（受 is_system_like_role 保护、不会被下方截断裁掉），因此必须
    // 在 `truncate_unprotected_messages_to_fit` **之前**注入，才能让最后一次截断把它
    // 占用的预算从其他可裁消息中腾出，避免返回轻微超 max_chars 的 payload。这与
    // `shrink_messages_to_fit_with_summary` 先插 summary note 再 truncate 的顺序一致。
    insert_overflow_archive_note_if_exists(&mut messages, overflow_dir);

    if messages_total_chars(&messages) > max_chars {
        truncate_unprotected_messages_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        );
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
    cwd: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
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
    dedup_repeated_tool_results(&mut messages, protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut messages,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
    );
    enforce_protected_precision_group_budget(
        &mut messages,
        KEEP_RECENT_TOOL_GROUPS,
        max_chars / 2,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
        false,
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
    // 与 plain shrink 对称：合并保护尾窗之外的 user/image 外溢 stub，防止占位
    // 消息随会话时长单调累积。
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // 与 shrink_messages_to_fit 对称：成功写入的 write_file/apply_patch 巨型
    // arguments 主动替换为归档 stub（保护窗口与失败结果保留）。
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }
    let had_leading_summary = messages.first().map(is_summary_message).unwrap_or(false);
    // 归档失败时必须恢复首次删除前的完整顺序；直接把 dropped 插到头部会把它们
    // 放到原本保留的 system prompt 之前，破坏 provider 要求的消息顺序。
    let mut messages_before_first_drop: Option<Vec<Message>> = None;
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();

    // 运行期字符总量：单条删除精确减掉 message_billable_chars，折叠/外溢是整体性
    // 批量变更，各自分支里统一重算，语义与每轮 `messages_total_chars(&messages)`
    // 完全等价，但避免多轮循环时对整条消息序列反复 O(n) 全量重扫。
    let mut total = messages_total_chars(&messages);
    while total > max_chars {
        // 一次性批量折叠超出预算的所有非保护工具组（compressible + non-compressible
        // 都通过 [`fold_early_tool_groups`] 处理）——理由同
        // [`shrink_messages_to_fit`]，避免单组 fold 循环迭代几十轮注入
        // `<empty>` empty-checkpoint note（详 e75fc2e5 session dump）。
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
            total = messages_total_chars(&messages);
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages, max_chars) {
            if messages_before_first_drop.is_none() {
                messages_before_first_drop = Some(messages.clone());
            }
            let removed_msg = messages.remove(idx);
            total = total.saturating_sub(message_billable_chars(&removed_msg));
            if is_internal_note_role(&removed_msg.role) {
                dropped_internal_notes.push(removed_msg);
            } else {
                dropped.push(removed_msg);
            }
            continue;
        }
        if let Some(dir) = overflow_dir
            && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
        {
            total = messages_total_chars(&messages);
            continue;
        }
        break;
    }

    let dropped_has_user_turn = dropped.iter().any(|m| m.role == "user");
    let has_leading_summary_now = messages.first().map(is_summary_message).unwrap_or(false);
    let internal_archive_dir =
        match archive_internal_notes_deduplicated(&dropped_internal_notes, overflow_dir) {
            Ok(path) => path,
            Err(()) => return messages_before_first_drop.unwrap_or(messages),
        };

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

                if !has_leading_summary_now {
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
                }
                insert_archive_note_if_missing(&mut messages, archive_note);
            } else {
                // flush 失败：绝不删历史。恢复首次删除前的完整消息快照，随后立即
                // 返回——跳过摘要/归档 note 注入
                //（防止产生没有对应归档文件的悬空指针 note）、truncate 与 reasoning
                // 清理。返回值可能仍超预算，但那是可恢复的（下轮重试压缩 / 请求层
                // clamp），而数据丢失不可逆——遵守“写入失败严禁删除历史”的既有教训。
                return messages_before_first_drop.unwrap_or(messages);
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

    insert_internal_note_archive_note_if_needed(&mut messages, internal_archive_dir.as_deref());

    if messages_total_chars(&messages) > max_chars {
        truncate_mutable_messages_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        );
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

/// 最后一层硬预算逃逸阀：保持 system/user 与工具调用配对结构不变，只缩短可重建的
/// assistant/tool 正文、reasoning 及超大 tool arguments。先保护当前高精度结果；若
/// 仍无法达标，再允许截断这些结果。若不可裁的 system/user 自身已经超限，返回 false。
fn truncate_mutable_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    truncate_mutable_messages_to_fit_with_policy(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
        true,
    )
}

fn truncate_mutable_messages_to_fit_with_policy(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
    allow_protected_fallback: bool,
) -> bool {
    if max_chars == 0 || messages_total_chars(messages) <= max_chars {
        return true;
    }

    // overflow asset 是受保护证据的唯一真源。预算不足时先丢弃可选预览，只保留
    // 可解析的 file_path 指针；后续通用 head+tail 截断绝不能再碰该最小协议。
    minimize_overflow_stubs_for_hard_budget(messages);
    if messages_total_chars(messages) <= max_chars {
        return true;
    }

    let mut blocked_fields = rustc_hash::FxHashSet::default();
    for include_protected in [false, true] {
        if include_protected && !allow_protected_fallback {
            break;
        }
        while messages_total_chars(messages) > max_chars {
            let excess = messages_total_chars(messages).saturating_sub(max_chars);
            let mut best: Option<(usize, MutableMessageField, usize)> = None;
            for (index, message) in messages.iter().enumerate() {
                if is_system_like_role(&message.role) || message.role == "user" {
                    continue;
                }
                let is_protected = protected_tool_context_message(message, protected_tool_call_ids);
                if is_protected && !include_protected {
                    continue;
                }
                let content_chars = value_len_chars(&message.content);
                if !message_contains_image(&message.content)
                    && !is_preserved_tool_overflow_content(&message.content)
                    && content_chars > 160
                    && !blocked_fields.contains(&(index, MutableMessageField::Content))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (index, MutableMessageField::Content, content_chars - 160),
                    );
                }
                if let Some(reasoning) = message.reasoning_content.as_deref()
                    && !reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX)
                    && reasoning.chars().count() > 160
                    && !blocked_fields.contains(&(index, MutableMessageField::Reasoning))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (
                            index,
                            MutableMessageField::Reasoning,
                            reasoning.chars().count() - 160,
                        ),
                    );
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for (call_index, call) in tool_calls.iter().enumerate() {
                        let argument_chars = call.function.arguments.chars().count();
                        let field = MutableMessageField::ToolArguments(call_index);
                        if argument_chars > 160 && !blocked_fields.contains(&(index, field)) {
                            choose_larger_mutable_field(
                                &mut best,
                                (index, field, argument_chars - 160),
                            );
                        }
                    }
                }
            }

            let Some((message_index, field, reducible)) = best else {
                break;
            };
            let reduce_by = excess.min(reducible).max(1);
            if !truncate_mutable_field(
                &mut messages[message_index],
                field,
                reduce_by,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            ) {
                // A fixed marker / archive path may already be the smallest
                // form the field can reach. Skip the no-progress field and try
                // other candidates instead of re-selecting the same one.
                blocked_fields.insert((message_index, field));
                continue;
            }
            insert_overflow_archive_note_if_exists(messages, overflow_dir);
            // 首次插入 archive note 可能改变消息下标；成功缩短后重新评估候选。
            blocked_fields.clear();
        }
    }

    messages_total_chars(messages) <= max_chars
}

/// 软压缩只裁未受保护字段；当前轮高精度 tool 结果必须留给真正的 hard-target
/// 兜底处理，不能因为 soft threshold 较小就损失刚读到的精确上下文。
fn truncate_unprotected_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    truncate_mutable_messages_to_fit_with_policy(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
        false,
    )
}

/// Path C 先给每个可裁字段设置单项上限，防止一个最新结果独占整个窗口；再按总预算
/// 继续收紧。两步都不删除消息或工具调用，也不改写 exact-replay 协议状态，因此
/// assistant↔tool 配对及 reasoning continuation state 保持完整。
fn emergency_cap_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    per_field_cap: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    minimize_overflow_stubs_for_hard_budget(messages);
    let mut truncated_any = false;
    for message in messages.iter_mut() {
        if is_system_like_role(&message.role)
            || message.role == "user"
            || protected_tool_context_message(message, protected_tool_call_ids)
        {
            continue;
        }
        let content_chars = value_len_chars(&message.content);
        if !message_contains_image(&message.content)
            && !is_preserved_tool_overflow_content(&message.content)
            && content_chars > per_field_cap
        {
            truncated_any |= truncate_mutable_field(
                message,
                MutableMessageField::Content,
                content_chars - per_field_cap,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            );
        }
        if let Some(reasoning_chars) = message
            .reasoning_content
            .as_deref()
            .filter(|reasoning| !reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX))
            .map(|reasoning| reasoning.chars().count())
            && reasoning_chars > per_field_cap
        {
            truncated_any |= truncate_mutable_field(
                message,
                MutableMessageField::Reasoning,
                reasoning_chars - per_field_cap,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            );
        }
        let tool_call_count = message.tool_calls.as_ref().map(Vec::len).unwrap_or(0);
        for call_index in 0..tool_call_count {
            let argument_chars = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .map(|call| call.function.arguments.chars().count())
                .unwrap_or(0);
            if argument_chars > per_field_cap {
                truncated_any |= truncate_mutable_field(
                    message,
                    MutableMessageField::ToolArguments(call_index),
                    argument_chars - per_field_cap,
                    overflow_dir,
                FieldArchivePolicy::BestEffort,
                );
            }
        }
    }
    if truncated_any {
        insert_overflow_archive_note_if_exists(messages, overflow_dir);
    }
    let inner = truncate_mutable_messages_to_fit(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
    );
    truncated_any || inner
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MutableMessageField {
    Content,
    Reasoning,
    ToolArguments(usize),
}

impl MutableMessageField {
    fn archive_label(self) -> String {
        match self {
            MutableMessageField::Content => "content".to_string(),
            MutableMessageField::Reasoning => "reasoning_content".to_string(),
            MutableMessageField::ToolArguments(index) => {
                format!("tool_calls[{index}].function.arguments")
            }
        }
    }

    fn original_text(self, message: &Message) -> Option<String> {
        match self {
            MutableMessageField::Content => Some(value_to_string(&message.content)),
            MutableMessageField::Reasoning => message.reasoning_content.clone(),
            MutableMessageField::ToolArguments(index) => message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(index))
                .map(|call| call.function.arguments.clone()),
        }
    }
}

fn choose_larger_mutable_field(
    best: &mut Option<(usize, MutableMessageField, usize)>,
    candidate: (usize, MutableMessageField, usize),
) {
    if best
        .as_ref()
        .is_none_or(|(_, _, best_reducible)| candidate.2 > *best_reducible)
    {
        *best = Some(candidate);
    }
}

const CONTEXT_OVERFLOW_TRUNCATED_PREFIX: &str = "[context-overflow-truncated]";
const CONTEXT_OVERFLOW_UNARCHIVED_POINTER: &str = "[context-overflow-truncated] full original was not archived; inline preview omitted to meet context budget.";

/// Whether text already has the overflow-truncated marker.
fn is_context_overflow_truncated_stub(text: &str) -> bool {
    text.trim_start()
        .starts_with(CONTEXT_OVERFLOW_TRUNCATED_PREFIX)
}

/// Extract the `archived at: <path>` target embedded in an overflow stub, if
/// any. The canonical form puts the pointer on the first line; a legacy inline
/// form cuts the path at `;` or end of line.
fn embedded_archive_path(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.split_once("archived at: ").map(|(_, p)| p.trim()))
        .or_else(|| {
            text.split_once("archived at: ")
                .map(|(_, rest)| rest.split([';', '\n']).next().unwrap_or(rest).trim())
        })
}

/// Fold an existing overflow-truncated stub into its minimal terminal state:
/// keep the archive path when one exists; when none exists, state plainly that
/// the full original is not readable back, never fabricate an archive pointer.
fn build_context_overflow_pointer(text: &str, target: usize) -> Option<String> {
    let path = embedded_archive_path(text);
    if let Some(path) = path {
        // Prefer the full pointer form when it fits the target.
        let full_pointer =
            format!("{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} full original archived at: {path}\n");
        if full_pointer.chars().count() <= target {
            return Some(full_pointer);
        }
        // Otherwise keep only the single-line archive pointer.
        let minimal = format!("{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} archived at: {path}");
        return (minimal.chars().count() < text.chars().count()).then_some(minimal);
    }
    (CONTEXT_OVERFLOW_UNARCHIVED_POINTER.chars().count() <= target
        && CONTEXT_OVERFLOW_UNARCHIVED_POINTER.chars().count() < text.chars().count())
    .then(|| CONTEXT_OVERFLOW_UNARCHIVED_POINTER.to_string())
}

/// Truncated tool arguments are terminal too. When archiving failed, only the
/// preview remains; re-archiving it would falsely claim the full arguments are
/// recoverable.
fn is_context_overflow_truncated_tool_arguments(arguments: &str) -> bool {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .is_some_and(|value| {
            value
                .get("_context_overflow_truncated")
                .and_then(Value::as_bool)
                == Some(true)
                && value.get("archive_file_path").is_some()
        })
}

fn build_context_overflow_tool_arguments_pointer(arguments: &str, target: usize) -> Option<String> {
    let value = serde_json::from_str::<Value>(arguments).ok()?;
    if value
        .get("_context_overflow_truncated")
        .and_then(Value::as_bool)
        != Some(true)
        || value.get("archive_file_path").is_none()
    {
        return None;
    }
    let archived_path = value
        .get("archive_file_path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty());
    let pointer = match archived_path {
        Some(path) => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": path,
        })
        .to_string(),
        None => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": Value::Null,
            "original_unavailable": true,
        })
        .to_string(),
    };
    (pointer.chars().count() <= target && pointer.chars().count() < arguments.chars().count())
        .then_some(pointer)
}

/// Archive-write policy for [`truncate_mutable_field`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldArchivePolicy {
    /// On archive-write failure, still truncate to a preview-only inline stub.
    /// Acceptable for rebuildable assistant/tool fields: losing the full text
    /// costs precision, not the user's instruction.
    BestEffort,
    /// The field is the authoritative copy of a user instruction. Archive it
    /// through the trusted session sink before every truncation, including
    /// re-collapsing marker-prefixed text; never trust a path from its content.
    /// Refuse without mutation when that write fails.
    Required,
}

fn truncate_mutable_field(
    message: &mut Message,
    field: MutableMessageField,
    reduce_by: usize,
    overflow_dir: Option<&Path>,
    archive_policy: FieldArchivePolicy,
) -> bool {
    // The archive path can be derived directly from overflow_dir (same source
    // as OverflowSink::new), so size the stub against that path first and do
    // the real archive write last: if the stub would not be strictly shorter
    // than the original, give up immediately instead of archiving first and
    // failing the size check afterwards (which would re-archive the same field
    // every compression round and grow the overflow file without bound).
    let archive_path_hint: Option<String> = overflow_dir.map(|dir| {
        dir.join(OVERFLOW_HISTORY_FILENAME)
            .to_string_lossy()
            .into_owned()
    });
    match field {
        MutableMessageField::Content => {
            if is_preserved_tool_overflow_content(&message.content) {
                return false;
            }
            let text = value_to_string(&message.content);
            let original_chars = text.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            // BestEffort fields may reuse an embedded archive path. Required
            // fields take the trusted re-archive path below instead.
            if is_context_overflow_truncated_stub(&text) {
                // A Required field is authoritative user input. Never trust an
                // `archived at:` path embedded in that user-controlled text:
                // it may name an unrelated existing file. First prove that the
                // trusted session sink can produce a shorter pointer, then
                // archive the entire current field and point only at the path
                // returned by that successful write.
                if archive_policy == FieldArchivePolicy::Required {
                    let pointer_for_path = |path: &str| {
                        let canonical = format!(
                            "{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} full original archived at: {path}"
                        );
                        build_context_overflow_pointer(&canonical, target)
                            .filter(|pointer| pointer.chars().count() < original_chars)
                    };
                    let Some(path_hint) = archive_path_hint.as_deref() else {
                        return false;
                    };
                    if pointer_for_path(path_hint).is_none() {
                        return false;
                    }
                    let Some(archive_file_path) =
                        archive_truncated_field_to_overflow(message, field, overflow_dir)
                    else {
                        return false;
                    };
                    let Some(pointer) = pointer_for_path(&archive_file_path) else {
                        return false;
                    };
                    message.content = Value::String(pointer);
                    return true;
                }
                if let Some(pointer) = build_context_overflow_pointer(&text, target) {
                    if pointer.chars().count() < original_chars {
                        message.content = Value::String(pointer);
                        return true;
                    }
                }
                return false;
            }
            // When the preview budget is too small, the stub would be only a
            // long path with no actual content (a fake truncation): a small
            // result (e.g. a task_status poll) turned into an empty-preview
            // stub leaves the model unable to judge the real state and can
            // trap it in a "cannot confirm status → poll forever" loop. Keep
            // the original and let the hard budget bail out instead of
            // producing an information-free stub.
            const MIN_CONTENT_PREVIEW_CHARS: usize = 32;
            let build_truncated = |path: Option<&str>| -> Option<String> {
                let prefix = path
                    .map(|p| {
                        format!(
                            "[context-overflow-truncated] full original archived at: {p}\nhead+tail preview:\n"
                        )
                    })
                    .unwrap_or_else(|| {
                        "[context-overflow-truncated] head+tail preview:\n".to_string()
                    });
                let preview_budget = target.saturating_sub(prefix.chars().count());
                if preview_budget < MIN_CONTENT_PREVIEW_CHARS {
                    return None;
                }
                Some(format!(
                    "{prefix}{}",
                    keep_ends_by_chars(&text, preview_budget)
                ))
            };
            let Some(truncated) = build_truncated(archive_path_hint.as_deref())
                .filter(|candidate| candidate.chars().count() < original_chars)
            else {
                return false;
            };
            // Truncate only when the archived form keeps a meaningful preview;
            // only after the archive write actually failed may we degrade to
            // the no-path inline stub, so a good archive is never downgraded
            // to an unreadable preview.
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy (current user instruction): without an archived
            // copy the preview-only stub would be the last surviving version of
            // the instruction and could never be read back. Refuse and keep
            // the original so the caller can surface the error.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return false;
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.content = Value::String(truncated);
            true
        }
        MutableMessageField::Reasoning => {
            let Some(reasoning) = message.reasoning_content.as_deref() else {
                return false;
            };
            // Exact replay payloads must remain byte-for-byte intact. Keeping
            // the marker while truncating its payload would break decoding in
            // the request layer, so callers must shrink another field instead.
            if reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX) {
                return false;
            }
            let original_chars = reasoning.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            // As in Content, reject a stub whose long archive path leaves no
            // meaningful preview; keep the original for the hard-budget path.
            const MIN_REASONING_PREVIEW_CHARS: usize = 32;
            let build_truncated = |path: Option<&str>| -> Option<String> {
                let prefix = path
                    .map(|p| {
                        format!("[context-overflow-truncated] full original archived at: {p}; ")
                    })
                    .unwrap_or_else(|| "[context-overflow-truncated] ".to_string());
                let preview_budget = target.saturating_sub(prefix.chars().count());
                if preview_budget < MIN_REASONING_PREVIEW_CHARS {
                    return None;
                }
                Some(format!(
                    "{prefix}{}",
                    keep_ends_by_chars(reasoning, preview_budget)
                ))
            };
            let Some(truncated) = build_truncated(archive_path_hint.as_deref())
                .filter(|candidate| candidate.chars().count() < original_chars)
            else {
                return false;
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return false;
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.reasoning_content = Some(truncated);
            true
        }
        MutableMessageField::ToolArguments(call_index) => {
            let Some(arguments) = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .map(|call| call.function.arguments.clone())
            else {
                return false;
            };
            let original_chars = arguments.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            if is_context_overflow_truncated_tool_arguments(&arguments) {
                let Some(pointer) =
                    build_context_overflow_tool_arguments_pointer(&arguments, target)
                else {
                    return false;
                };
                let Some(call) = message
                    .tool_calls
                    .as_mut()
                    .and_then(|calls| calls.get_mut(call_index))
                else {
                    return false;
                };
                call.function.arguments = pointer;
                return true;
            }
            // A fixed JSON prefix plus a long archive path can consume the
            // entire target. Reject empty or near-empty previews: they contain
            // no real argument data and can make the model replay protocol keys
            // as tool arguments. Larger previews remain useful even when the
            // path consumes much of the budget.
            const MIN_TOOL_ARGS_PREVIEW_CHARS: usize = 8;
            let build_truncated = |path: Option<&str>, preview: String| {
                serde_json::json!({
                    "_context_overflow_truncated": true,
                    "original_chars": original_chars,
                    "archive_file_path": path,
                    "preview": preview,
                })
                .to_string()
            };
            let build_candidate = |path: Option<&str>| -> Option<String> {
                let fixed_chars = build_truncated(path, String::new()).chars().count();
                let mut preview_budget = target.saturating_sub(fixed_chars);
                let mut preview_text = keep_ends_by_chars(&arguments, preview_budget);
                let mut candidate = build_truncated(path, preview_text.clone());
                // JSON escaping can expand characters; tighten by the measured
                // serialized excess.
                while candidate.chars().count() > target && preview_budget > 0 {
                    let excess = candidate.chars().count() - target;
                    preview_budget = preview_budget.saturating_sub(excess.max(1));
                    preview_text = keep_ends_by_chars(&arguments, preview_budget);
                    candidate = build_truncated(path, preview_text.clone());
                }
                (preview_text.chars().count() >= MIN_TOOL_ARGS_PREVIEW_CHARS
                    && candidate.chars().count() < original_chars)
                    .then_some(candidate)
            };
            let Some(truncated) = build_candidate(archive_path_hint.as_deref()) else {
                return false;
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return false;
            }
            // Recompute the preview against the no-path stub after the archive
            // write failed, so a dead path does not eat the preview budget.
            let truncated = build_candidate(archive_file_path.as_deref()).unwrap_or(truncated);
            let Some(call) = message
                .tool_calls
                .as_mut()
                .and_then(|calls| calls.get_mut(call_index))
            else {
                return false;
            };
            // Arguments must remain valid JSON; slicing the string directly
            // would make the provider reject the request.
            call.function.arguments = truncated;
            true
        }
    }
}

/// 主动精简「已成功写入」的 write_file / apply_patch 巨型 arguments。
///
/// 文件已落盘（结果消息确认成功）之后，完整 content/patch 正文对后续轮次没有
/// 语义价值——模型引用的是文件路径而非正文——保留只会持续占用上下文。与预算
/// 压力无关：只要该组已滑出最近保护窗口（模型不再可能引用刚写入的正文来构造
/// 后续编辑），立即将其替换为 `_context_overflow_truncated` 指针 stub 并零压缩
/// 归档原文；失败结果、窗口内结果、当前轮保护 id 一律不动，保证 agent 效果
/// 不劣化（模型需要时仍可按 stub 的 archive_file_path 回读原文，或按 preview
/// 识别文件内容轮廓）。
fn shrink_successful_write_arguments(
    messages: &mut Vec<Message>,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) {
    if messages.is_empty() {
        return;
    }
    // 保护窗口：最近 KEEP_RECENT_TOOL_GROUPS 个已有结果的工具组（含当前轮刚写完、
    // 模型可能立即引用正文构造后续编辑的组），其调用一律保留完整 arguments。
    let protected_recent_call_ids: rustc_hash::FxHashSet<String> =
        recent_tool_result_groups(messages, KEEP_RECENT_TOOL_GROUPS)
            .into_iter()
            .flatten()
            .filter_map(|idx| messages[idx].tool_call_id.clone())
            .collect();
    // tool_call_id → 结果文本（判定成功/失败）。
    let result_by_call_id: rustc_hash::FxHashMap<String, String> = messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            message
                .tool_call_id
                .as_deref()
                .map(|id| (id.to_string(), value_to_string(&message.content)))
        })
        .collect();

    let mut changed = false;
    for message in messages.iter_mut() {
        if message.role != "assistant" {
            continue;
        }
        let Some(tool_calls) = message.tool_calls.as_mut() else {
            continue;
        };
        // 先收集候选，避免与 truncate_mutable_field 的独占借用冲突。
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for (call_index, call) in tool_calls.iter().enumerate() {
            let name = call.function.name.as_str();
            if name != "write_file" && name != "apply_patch" {
                continue;
            }
            if protected_recent_call_ids.contains(&call.id)
                || protected_tool_call_ids.contains(&call.id)
            {
                continue;
            }
            let arguments = &call.function.arguments;
            if arguments.contains("\"_context_overflow_truncated\"") {
                continue; // 已替换，幂等（避免重复归档/重复写文件）
            }
            let original_chars = arguments.chars().count();
            if original_chars <= 160 {
                continue;
            }
            let Some(result_text) = result_by_call_id.get(&call.id) else {
                continue;
            };
            if !is_successful_write_result(name, result_text) {
                continue;
            }
            candidates.push((call_index, original_chars));
        }
        for (call_index, original_chars) in candidates {
            if truncate_mutable_field(
                message,
                MutableMessageField::ToolArguments(call_index),
                original_chars.saturating_sub(240),
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            ) {
                changed = true;
            }
        }
    }
    if changed {
        insert_overflow_archive_note_if_exists(messages, overflow_dir);
    }
}

/// 判定 write_file / apply_patch 的结果是否成功。失败结果必须保留完整 arguments
/// 供模型依据原文修复；成功结果才是可安全精简的对象。
fn is_successful_write_result(tool_name: &str, result_text: &str) -> bool {
    let trimmed = result_text.trim_start();
    if trimmed.starts_with("Error:") || trimmed.starts_with("Exit code:") {
        return false;
    }
    match tool_name {
        "write_file" => trimmed.starts_with("Successfully wrote to"),
        "apply_patch" => trimmed.starts_with("Successfully patched"),
        _ => false,
    }
}

/// Last-resort rescue for the reactive overflow path, for the case where
/// mid-turn compression can no longer make progress: its policies never
/// truncate user messages, so an oversized current user message would
/// otherwise fail the turn outright. Offloads the middle of the last **real**
/// user message to the overflow archive and replaces it with a head+tail
/// preview stub, using the same machinery as mutable assistant fields
/// ([`truncate_mutable_field`]).
///
/// Only call this after the provider actually rejected the request: the
/// pre-request soft budget must keep the current user message intact so
/// legitimate large contexts reach the provider unchanged. Returns true when
/// the message was truncated. Refuses (returns false, message untouched) when
/// the overflow archive write fails: a preview-only stub would be the only
/// surviving copy of the user's instruction and could never be read back, so
/// the caller must surface the provider error instead of retrying on an
/// unrecoverable fragment. Marker-prefixed content is always re-archived
/// through the trusted session sink before it is collapsed, so an embedded
/// user-controlled path is never treated as provenance.
pub(in crate::ai) fn truncate_last_real_user_message_to_fit(
    messages: &mut [Message],
    target_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    let total = messages_total_chars(messages);
    if total <= target_chars {
        return false;
    }
    let Some(index) = last_real_user_index(messages) else {
        return false;
    };
    let message = &mut messages[index];
    // Multimodal (array) content must not be flattened into a text stub —
    // that would drop image parts. Only plain string content can be offloaded.
    if message.content.as_str().is_none() {
        return false;
    }
    truncate_mutable_field(
        message,
        MutableMessageField::Content,
        total - target_chars,
        overflow_dir,
        // Required: the current user instruction must stay recoverable via
        // the archive; without an archived copy the rescue must not fire.
        FieldArchivePolicy::Required,
    )
}

fn messages_total_chars(messages: &[Message]) -> usize {
    messages.iter().map(message_billable_chars).sum::<usize>()
}

fn current_turn_precision_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // 合成 user 消息不构成轮次边界：否则本轮早前轮次的 precision 工具结果
    // 会失去保护，被 Path C 有损截断。若完全没有真实 user，则整段历史都是
    // 当前合成轮，和 retained_turn_start 的保守边界保持一致。
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if is_non_compressible_tool(&tool_call.function.name)
                && crate::ai::tools::tool_history_policy(&tool_call.function.name)
                    .counts_toward_precision_inline_budget()
            {
                out.insert(tool_call.id.clone());
            }
        }
    }
    out
}

/// 收集当前 turn 内所有禁止有损压缩的工具调用。它比 precision inline 集合更宽：
/// `task_wait` 等聚合结果不参与 precision 配额，但其正文同样不能被 Path C 截断。
fn current_turn_lossless_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // 没有真实 user 时，不能把合成轮中的不可有损结果暴露给 Path C。
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if !crate::ai::tools::tool_history_policy(&tool_call.function.name)
                .allows_lossy_compress()
            {
                out.insert(tool_call.id.clone());
            }
        }
    }
    out
}

fn protected_tool_result_message(
    message: &Message,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    message.role == "tool"
        && message
            .tool_call_id
            .as_ref()
            .is_some_and(|id| protected_tool_call_ids.contains(id))
}

fn protected_tool_context_message(
    message: &Message,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    protected_tool_result_message(message, protected_tool_call_ids)
        || message.tool_calls.as_ref().is_some_and(|calls| {
            calls
                .iter()
                .any(|call| protected_tool_call_ids.contains(&call.id))
        })
}

/// Public proxy of [`messages_total_chars`] for callers in other ai modules
/// (e.g. mid-turn compression in `turn_runtime`) that need to check budget
/// without re-implementing the same accounting.
pub(in crate::ai) fn messages_total_chars_pub(messages: &[Message]) -> usize {
    messages_total_chars(messages)
}

const CONTEXT_COMPACTION_STATE_PREFIX: &str = "[runtime context state]";
const CONTEXT_COMPACTION_STATE: &str = "[runtime context state]\n\
- This request uses a compacted context projection and has passed the runtime budget guard.\n\
- Folded or truncated tool output is recoverable evidence; it does not mean the model context is full.\n\
- Prefer the stub's original_file_path/original_range. Read archive_file_path only when the original source is unavailable.\n\
- Report context exhaustion only when the provider returns an explicit context-length error.\n\
- Continue from the latest working checkpoint and verify uncertain details from the cited source.";

pub(in crate::ai) fn is_context_compaction_state(message: &Message) -> bool {
    message.role == ROLE_INTERNAL_NOTE
        && message
            .content
            .as_str()
            .is_some_and(|content| content.starts_with(CONTEXT_COMPACTION_STATE_PREFIX))
}

fn upsert_context_compaction_state(messages: &mut Vec<Message>) {
    messages.retain(|message| !is_context_compaction_state(message));
    let insert_at = last_real_user_index(messages).map_or(messages.len(), |index| index + 1);
    messages.insert(
        insert_at,
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(CONTEXT_COMPACTION_STATE.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
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
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize) {
    let before = messages_total_chars(&messages);
    let messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir);
    let after_evidence_trim = messages_total_chars(&messages);
    if after_evidence_trim <= soft_threshold {
        return (messages, before, after_evidence_trim);
    }
    let mut out = messages;
    // 把压缩状态显式交给模型，避免它把可恢复的 evidence stub 误判为上下文已满。
    // 在任何裁剪之前插入，后续预算计算会把这条固定开销一并纳入。
    upsert_context_compaction_state(&mut out);
    // 0. 清理过期 reasoning_content：单 turn 内 LLM 多次返回的 reasoning chain
    //    对后续决策无益，但部分厂商要求历史 reasoning 与 tool_calls 配对。
    //    只保留最后一条 assistant 的 reasoning_content，其余置 None。
    keep_only_recent_reasoning_content(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 1. 同 signature 工具结果去重
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&out);
    dedup_repeated_tool_results(&mut out, &protected_tool_call_ids);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. 远端结构化裁剪：tool 结果中段按行折叠到 480 字/条，最近 6 条保留全文。
    //    传入 overflow_dir 后，read_file/grep 等「不可压缩」工具的大输出会被
    //    零压缩外溢到会话文件并留 head+tail 预览 stub（与跨 turn 压缩一致），
    //    既释放上下文体积又不丢信息——模型可按 stub 里的 file_path 重新 read_file。
    prepare_tool_messages_structured(
        &mut out,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        &protected_tool_call_ids,
    );
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 3. 仍超额：用 shrink_messages_to_fit 走"折叠 tool group + 整体兜底"
    out = shrink_messages_to_fit(
        out,
        soft_threshold,
        overflow_dir,
        cwd,
        &protected_tool_call_ids,
    );
    let after = messages_total_chars(&out);
    (out, before, after)
}

/// LLM 摘要"有效压缩"的最小净下降量（字符）。低于此值视为低效，
/// `was_effective` 返回 false；硬预算兜底仍可能返回一个略小的上下文结果。
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
///     逐步缩小保护窗口到 2→1，直到有效压缩或降至 `hard_target` 以下。
///     解决"臃肿全在保护尾窗内、早期历史已压无可压"时压缩器空转的问题。
///   - Path C 兜底（per-message 截断）：渐进式折叠后仍超 `hard_target` 时，
///     对尾窗内单个超大非 system 消息做 head+tail 截断。这是绝对最后手段。
/// 头部所有 system / internal_note（agent 指令、工具列表、全局指引）始终原样保留。
/// 返回 `(messages_after, before, after, was_effective, llm_summary_inserted)`；
/// `was_effective` 仅在净下降 ≥ [`MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS`] 时为 true。
/// false 不代表返回的 messages 未变化；硬预算兜底可能产生低于有效阈值的部分下降。
/// `llm_summary_inserted` 表示 Path A 是否真的执行并注入了 `[mid-turn-summary]`：
/// 为 false 且 `after < before` 说明下降全部来自机械路径（折叠/截断/外溢），
/// 供上层报告区分"LLM 摘要已执行"与"纯机械压缩"，避免误报。
pub(in crate::ai) async fn mid_turn_llm_summarize(
    app: &App,
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
    hard_target: usize,
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize, bool, bool) {
    let before = messages_total_chars(&messages);
    let overflow_dir = crate::ai::history::SessionStore::new(app.config.history_file.as_path())
        .session_assets_dir(&app.session_id);
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&messages);
    let lossless_tool_call_ids = current_turn_lossless_tool_call_ids(&messages);
    // best 追踪迄今为止体积最小的结果；None 表示仍使用原始 messages。
    let mut best: Option<Vec<Message>> = None;
    let mut best_after = before;
    // Path A 是否真的执行并注入了 [mid-turn-summary]（见返回注释）。
    let mut llm_summary_inserted = false;

    // === Path A：跨轮 LLM 摘要 ===
    // 先按"保留最近 keep_recent_turns 个 user 轮"算切点。经过前置投影压缩后，
    // 较早的 user 消息可能已被替换成 internal_note 摘要（role != "user"），导致
    // 投影里可见的 user 边界不足 keep_recent_turns，retained_turn_start 返回 0。
    // 但这并不代表没有可压缩的旧内容--第一个 user 消息之前仍可能残留被协议配对
    // 保护的 assistant(tool_calls)/tool 记录（无法逐条删除）。此时把切点回退到
    // 第一个 user 消息位置：尾部用户轮仍受保护，前缀的 system-like 摘要/归档标记
    // 由 preserved_system_end 保留，二者之间的旧对话区段可被 LLM 摘要回收。
    let mut split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 {
        if let Some(first_user) = messages.iter().position(|m| m.role == "user") {
            if first_user > 0 {
                split_at = first_user;
            }
        }
    }
    if split_at > 0 && split_at < messages.len() {
        // 保留头部前缀连续的 system-like 消息（agent 指令等），只摘要其后的对话
        // 区段。早期版本直接丢弃 messages[0] 的 system prompt，会让模型立刻失去
        // agent 行为指令，表现为"压缩后回复戛然而止 / 极短 / 跑偏"。
        let preserved_system_end = messages[..split_at]
            .iter()
            .position(|m| !is_system_like_role(&m.role))
            .unwrap_or(split_at);
        let earlier = &messages[preserved_system_end..split_at];
        // 从待摘要区段中抽出 context checkpoint 标记：它是定位已保存检查点正文的
        // 唯一索引，绝不能被摘要吞掉。常规落盘压缩路径已做同样处理，这里补齐。
        let checkpoint_markers: Vec<Message> = earlier
            .iter()
            .filter(|m| is_context_checkpoint_marker(m))
            .cloned()
            .collect();
        let summary_source: Vec<Message> = earlier
            .iter()
            .filter(|m| !is_context_checkpoint_marker(m))
            .cloned()
            .collect();
        let has_dialog = earlier
            .iter()
            .any(|m| m.role == "user" || m.role == "assistant")
            || !checkpoint_markers.is_empty();
        if has_dialog {
            let summary =
                build_persisted_summary_text_with_app(app, &summary_source, summary_max_chars)
                    .await;
            if !summary.trim().is_empty() {
                let archive_file_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
                let tail_plan = plan_early_tool_groups(
                    &messages[split_at..],
                    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS,
                    Some(overflow_dir.as_path()),
                    &protected_tool_call_ids,
                );
                let mut out =
                    Vec::with_capacity(preserved_system_end + 2 + (messages.len() - split_at));
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
                insert_archive_note_if_missing(
                    &mut out,
                    build_overflow_placeholder(&archive_file_path.to_string_lossy()),
                );
                // 2b. 回填被抽出的 context checkpoint 标记，保留其可回读索引
                out.extend(checkpoint_markers.iter().cloned());
                // 3. 尾窗折叠先纯规划；整个 Path A 确认优于当前 best 后再统一写盘。
                out.extend_from_slice(tail_plan.messages());
                let after = messages_total_chars(&out);
                // 先提交尾窗折叠，确认候选被采纳后再归档 earlier：archive 是追加式写
                // overflow-history.md（非幂等），若提前归档而 commit 失败，`earlier`
                // 已落盘但上下文未采纳 `out`，下轮压缩会重复归档同一批消息 → 孤儿累积。
                // 短路 `&&` 保证 commit 失败时根本不会触碰归档；commit 成功后若归档
                // 失败，`best` 不更新、上下文仍保留 earlier，无数据丢失（仅剩幂等
                // 哈希命名的折叠文件）。
                if after < best_after
                    && tail_plan.commit()
                    && archive_messages_to_overflow(earlier, Some(overflow_dir.as_path())).is_some()
                {
                    best = Some(out);
                    best_after = after;
                    llm_summary_inserted = true;
                }
                // 有效压缩且达标 → 直接返回
                if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS
                    && best_after <= hard_target
                {
                    return (best.unwrap(), before, best_after, true, true);
                }
            }
        }
    }

    // === Path B+C：渐进式工具组折叠 ===
    // 从 keep_recent=4（等价于原 Path B）开始，逐步缩小保护窗口到 2→1（绝不到 0），
    // 直到有效压缩或降至 hard_target 以下。解决"臃肿全在保护尾窗内"时空转。
    // 在 best（Path A 结果或原始 messages）上链式折叠：已折叠的组变成 stub
    //（internal_note），不会被 fold_early_tool_groups 再次匹配，因此每次迭代
    // 只会折叠上一轮保留的组，逐步释放保护尾窗。窗口不降到 0（见
    // [`MIN_KEEP_RECENT_TOOL_GROUPS`]）：保留最近 1 组逐字，剩余超额由下方 Path C
    // per-message 截断兜底，避免把最近一次工具交互也 stub 化。
    for &keep_recent in progressive_fold_windows().iter() {
        if best_after <= hard_target {
            break;
        }
        let current = best.as_ref().unwrap_or(&messages);
        let plan = plan_early_tool_groups(
            current,
            keep_recent,
            Some(overflow_dir.as_path()),
            &protected_tool_call_ids,
        );
        if plan.folded_groups() == 0 {
            continue;
        }
        let after = messages_total_chars(plan.messages());
        if after < best_after && plan.commit() {
            let (folded, _) = plan.into_result();
            best = Some(folded);
            best_after = after;
        }
    }

    // 只有真正达到 hard_target 才能提前返回。旧逻辑只要净下降超过 4K 就返回，
    // 会跳过下面的硬兜底，让「旧组已省很多、最新一组仍单独超窗」继续发出超限请求。
    if best_after <= hard_target {
        let was_effective = before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS;
        return (
            best.unwrap_or(messages),
            before,
            best_after,
            was_effective,
            llm_summary_inserted,
        );
    }

    // === Path C 兜底：预算感知的结构保留截断 ===
    // 保留 system/user 与最近工具组的 assistant↔tool 配对；仅压缩可再取回的结果正文、
    // reasoning 和超大 tool arguments。与旧的「每条最多 8K」不同，这里继续按总预算
    // 收紧，因此并行工具结果很多时也能收敛。若不可裁的 system/user 本身已超预算，
    // 返回可达到的最小结果，而不是破坏用户任务原文。
    let mut result = best.unwrap_or(messages);
    // Path C 兜底前先对当前 turn 的所有禁止有损压缩结果做零压缩外溢：
    // 原文落盘为可回读 asset 并替换为 stub，避免紧接着的 `emergency_cap_messages_to_fit`
    // 把这些 grounding 证据有损截断到 8K / ~160 字符后原文不可恢复。
    spill_protected_precision_to_fit(
        &mut result,
        hard_target,
        Some(overflow_dir.as_path()),
        cwd,
        &lossless_tool_call_ids,
    );
    emergency_cap_messages_to_fit(
        &mut result,
        hard_target,
        PATH_C_PER_MSG_CAP,
        Some(overflow_dir.as_path()),
        &lossless_tool_call_ids,
    );
    let after = messages_total_chars(&result);
    let savings = before.saturating_sub(after);
    (
        result,
        before,
        after,
        savings >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS,
        llm_summary_inserted,
    )
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

/// Protection window over recent complete tool groups. An assistant(tool_calls)
/// batch is an indivisible evidence unit: it must never be truncated message by
/// message, or parallel reads would leave half a batch behind.
const KEEP_RECENT_TOOL_GROUPS: usize = 4;

/// **Minimum** protection window for progressive group folding: narrowing stops
/// here and never reaches 0.
///
/// A window of 0 folds the most recent tool interaction itself into a
/// `compressed_tool_round` stub, leaving the model without any recent structured
/// tool context (`assistant.tool_calls` + `role=tool` results): multi-step task
/// continuity suffers and runtime guards lose their freshest evidence. Keep the
/// most recent group verbatim; remaining excess is handled downstream by
/// per-message truncation / first_trim fallbacks.
const MIN_KEEP_RECENT_TOOL_GROUPS: usize = 1;

/// Floor on the protected verbatim tail once group folding converges, measured in
/// billable chars across the messages retained for the current window. Group-count
/// protection alone proved insufficient on read-heavy sessions: many large results
/// still squeezed the window down until nearly everything except the last turn was
/// a pointer stub, after which the model re-read files whose full text it had just
/// received. This floor keeps roughly 6K tokens of fresh tool evidence resident
/// whenever budget allows. It intentionally yields (folding proceeds past it) only
/// at MIN_KEEP_RECENT_TOOL_GROUPS so overflow handling always terminates.
const MIN_PROTECTED_TAIL_CHARS: usize = 30_000;

/// Window sequence for progressive folding: tightens stepwise from
/// [`KEEP_RECENT_TOOL_GROUPS`] down to [`MIN_KEEP_RECENT_TOOL_GROUPS`], widening
/// what may be folded but never reaching 0. Both progressive paths
/// ([`fold_noncompressible_tool_groups_to_fit`] and mid-turn summarization's
/// Path B+C) share one sequence so the minimum-protection policy has a single
/// source of truth.
fn progressive_fold_windows() -> Vec<usize> {
    let mut windows = Vec::new();
    let mut keep = KEEP_RECENT_TOOL_GROUPS;
    loop {
        windows.push(keep);
        if keep <= MIN_KEEP_RECENT_TOOL_GROUPS {
            break;
        }
        keep = (keep / 2).max(MIN_KEEP_RECENT_TOOL_GROUPS);
    }
    windows
}

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
    let mut prev_tool_call_id: Option<String> = None;
    // 上一条 tool 消息的 tool_call_id：只有同 tool_call_id 才视为同一次调用的重复结果。
    for m in messages.drain(..) {
        let text = value_to_string(&m.content);
        // 完全相等去重仅对 tool 启用：用户/助手/system 原文不做去重。
        // 必须同 tool_call_id：并行工具调用返回相同文本属于不同调用，不能丢弃，
        // 否则会破坏 assistant tool_call <-> tool result 的配对。
        if m.role == "tool"
            && m.role == prev_role
            && text == prev_content
            && m.tool_call_id.is_some()
            && m.tool_call_id == prev_tool_call_id
        {
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
            && m.tool_call_id.is_some()
            && m.tool_call_id == prev_tool_call_id
        {
            continue;
        }
        prev_role = m.role.clone();
        prev_content = text;
        prev_signature = signature;
        prev_tool_call_id = m.tool_call_id.clone();
        out.push(m);
    }
    *messages = out;
}

/// 裁剪历史中的 reasoning_content，只保留确有必要回传给厂商的那些。
///
/// 较老的 reasoning chain 对后续 turn 决策几乎没有帮助，去掉可节省上下文预算。
/// 部分模型对 tool-call reasoning 有回传约束，因此这里的策略是：
/// - 模型显式声明 exact replay 的 continuation state 带内部标记，只要所属
///   assistant/tool 协议组仍在上下文中就始终保留；整组被摘要替代后无需再回放；
/// - 其他带 `tool_calls` 的 assistant 消息只保留最近
///   `KEEP_RECENT_TOOL_CALL_REASONING` 轮的完整 reasoning_content，更早的置 None；
///   DeepSeek 所需的缺失字段由 request 层用空字符串占位补齐，避免历史 reasoning
///   文本在长 session 里单调累积、拖慢并"变蠢"；
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

    // 未标记的 tool-call assistant reasoning 跨轮滑窗：只保留最近 N 条完整文本。
    let tool_call_reasoning_count = messages
        .iter()
        .filter(|m| {
            m.role == "assistant"
                && m.reasoning_content.as_deref().is_some_and(|reasoning| {
                    !reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX)
                })
                && m.tool_calls.is_some()
        })
        .count();
    let drop_tool_call_reasoning_before =
        tool_call_reasoning_count.saturating_sub(KEEP_RECENT_TOOL_CALL_REASONING);
    let mut seen_tool_call_reasoning = 0usize;

    for (idx, m) in messages.iter_mut().enumerate() {
        if m.role != "assistant" || m.reasoning_content.is_none() {
            continue;
        }
        // exact replay 是所属 tool-call 消息的协议状态；消息仍在时不能单独裁掉。
        if m.reasoning_content
            .as_deref()
            .is_some_and(|reasoning| reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX))
        {
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
fn dedup_repeated_tool_results(
    messages: &mut [Message],
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) {
    use rustc_hash::{FxHashMap, FxHasher};
    use std::hash::{Hash, Hasher};

    // 收集 (tool_name, args_signature) → 出现次数与索引
    // 通过 assistant.tool_calls 关联 tool_call_id → (name, args)
    let mut id_occurrences: FxHashMap<String, usize> = FxHashMap::default();
    for message in messages.iter() {
        for tool_call in message.tool_calls.iter().flatten() {
            *id_occurrences.entry(tool_call.id.clone()).or_default() += 1;
        }
    }
    let ambiguous_ids = id_occurrences
        .into_iter()
        .filter_map(|(id, count)| (count > 1).then_some(id))
        .collect::<rustc_hash::FxHashSet<_>>();
    let mut id_to_signature: FxHashMap<String, (String, String)> = FxHashMap::default();
    let mut id_to_args_raw: FxHashMap<String, String> = FxHashMap::default();
    for message in messages.iter() {
        if let Some(tool_calls) = &message.tool_calls {
            for tc in tool_calls {
                if ambiguous_ids.contains(&tc.id) {
                    continue;
                }
                let args_norm = serde_json::from_str::<Value>(&tc.function.arguments)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| tc.function.arguments.clone());
                id_to_signature.insert(tc.id.clone(), (tc.function.name.clone(), args_norm));
                id_to_args_raw.insert(tc.id.clone(), tc.function.arguments.clone());
            }
        }
    }

    let tool_indices = tool_message_indices(messages);
    let protected_indices = recent_tool_group_message_indices(messages, KEEP_RECENT_TOOL_GROUPS);

    // `read_file` 的 offset/limit 不同不会命中调用签名去重，但它们可能包含同一
    // 段文件。仅在两个结果都已离开近端保护窗口、同文件重叠行逐字一致时，才从较早
    // 结果删除重叠行；任一行不同（文件曾被编辑、输出格式变化等）即保持原样。
    dedup_overlapping_read_file_results(
        messages,
        &id_to_signature,
        &protected_indices,
        protected_tool_call_ids,
    );

    // (name, args) → 该签名下"最新保留全文"的 tool 调用，用于在折叠时回指。
    let mut seen: FxHashMap<(String, String), DedupToolOccurrence> = FxHashMap::default();
    // (tool_name, content_hash) → 最新出现该内容版本的 tool 调用。
    // 内容级去重是断开"重复整篇重读"失忆环的关键：对 read_file 等
    // non-compressible 工具，同一 (文件) 被反复读取时往往返回**逐字节
    // 相同**的全文（实测占全部 tool 字节的 ~52%）。这些冗余副本可无损折叠，
    // 而内容确实变化的版本（如被编辑过的文件）因 hash 不同得以完整保留。
    //
    // **关键**：key 不携带 `args_norm`——历史上把 args 也纳入键，导致显式的
    // "同一查询的大小写/路径变体"（`readFileLines` vs `read_file_lines`、
    // 大小写敏感差异等）即便返回**逐字节相同**的"无命中"体也 collapse 不掉，
    // 在尾部反复堆积 6+ 份 15KB 的同内容（详 e75fc2e5 session dump）。
    // 改用 `(tool_name, content_hash)`：只要返回体本身一致就折叠——args 不同
    // 由调用签名去重的 `seen` 计数器单独管，不影响内容级折叠。
    let mut seen_content: FxHashMap<(String, u64), DedupToolOccurrence> = FxHashMap::default();
    // 从新到旧扫描，确保最新一次调用保留全文，较早的重复结果才被折叠。
    // 这对失败后重试尤其关键：成功重试不能被旧失败占据 canonical 位置后压成 stub。
    for &idx in tool_indices.iter().rev() {
        if messages[idx]
            .tool_call_id
            .as_ref()
            .is_some_and(|id| ambiguous_ids.contains(id))
        {
            // 旧历史里复用的 ID 无法可靠关联到具体 assistant occurrence；保留原文。
            continue;
        }
        let occurrence = dedup_tool_occurrence(messages, idx, &id_to_signature, &id_to_args_raw);
        let occurrence = match occurrence {
            Some(occurrence) => occurrence,
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
        // Never re-process a stub produced by an earlier projection build:
        // rendering from stub text nests stale previews/excerpts inside fresh
        // stubs, and neither copy holds real result data. Keep this marker in
        // sync with the "[deduped:" prefixes emitted by render_dedup_tool_stub.
        if value_to_string(&messages[idx].content).starts_with(DEDUP_STUB_MARKER_PREFIX) {
            continue;
        }
        let signature_key = (occurrence.tool_name.clone(), occurrence.args_norm.clone());
        let signature_canonical = seen.get(&signature_key).cloned();
        if signature_canonical.is_none() {
            seen.insert(signature_key, occurrence.clone());
        }
        // **不再豁免最近保护窗内的重复**。历史上这里 `if protected_indices.contains(&idx) continue;`
        // 让最近 N 个工具组完全跳过去重，于是 agent 不断重发同一查询、最新副本一直
        // 落到"最近窗"里 → 永不被折叠，尾部堆积 15KB × 29 份的逐字节相同结果。
        // 现在让 dedup 一视同仁跑遍所有 tool 消息：逆序首见（即最新一次）登记为
        // canonical 全文，其余较早副本一律折叠为回指 stub。模型仍能看到最新全文，
        // 同时避免旧失败覆盖后续成功重试的有效结果。
        // orphan 的保护逻辑（上面的 `!protected_indices.contains`）已经单独处理，
        // 不受这里影响。
        // 内容级去重同样作用于 current-turn precision 保护的调用（本轮内的
        // read_file 重读）：同一轮内对同一文件逐字节相同的重读是纯冗余，折叠较早
        // 副本、保留逆序首见（最新）全文即可。这不违反"precision 结果保持 raw"
        // 不变式——最新副本仍是 raw 全文，旧副本只是回指它；同时直接切断
        // "同轮内全文重读堆积 → 近端 offload → 失忆再重读"的环。
        if tool_uses_content_identity_dedup(&occurrence.tool_name) {
            // read_file/检索类工具**内容不同的版本**必须零压缩保留（Invariant：
            // precision 结果不做 lossy 裁剪）。但**逐字节相同**的重复副本是纯冗余，
            // 折叠它们不丢失任何信息，且能直接消除"旧全文堆积 + 近端 offload 触发
            // 重读"的失忆环。用内容 hash 区分二者：hash 首见 → 保留全文并登记；
            // hash 重现 → 折叠为回指最新全文的 stub（保留 tool_call_id 以维持协议）。
            let text = value_to_string(&messages[idx].content);
            // 若内容本身已是 overflow/truncation 归档 stub，它并非"完整结果"：
            // canonical（逆序首见）与本副本 byte-identical，因此 canonical 同样是
            // 截断 stub。此时折叠成"reuse the canonical full result"是谎报——真实
            // 案例：task_wait 结果先被 overflow 截断成 [context-overflow-truncated]，
            // dedup 再谎称可复用 canonical 全文，模型反复追 canonical 却永远拿不到
            // 原文（下一跳仍是 stub 的回指链）。跳过折叠：每条 stub 自带 file_path
            // 召回指针，保留它们即可让模型直接回读归档原文。
            if is_content_overflow_archived_stub(&messages[idx].content) {
                continue;
            }
            let mut hasher = FxHasher::default();
            text.hash(&mut hasher);
            let content_key = (occurrence.tool_name.clone(), hasher.finish());
            match seen_content.get(&content_key).cloned() {
                None => {
                    seen_content.insert(content_key, occurrence);
                }
                Some(canonical) => {
                    let stub = render_dedup_tool_stub(
                        DedupToolStubKind::ByteIdentical,
                        &occurrence,
                        &canonical,
                        &text,
                    );
                    messages[idx].content = Value::String(stub);
                }
            }
            continue;
        }
        // 签名级去重仍跳过 current-turn precision 保护的调用：args 变体本身携带
        // 信息（offset/limit/use_line_numbers 不同就不该折叠），避免误伤本轮正在
        // 使用的读取。上面的内容级去重已经处理了"真正逐字节相同"的情况。
        if protected_tool_call_ids.contains(&occurrence.tool_call_id) {
            continue;
        }
        // 逆序首见即最新调用；把更早的同签名结果折叠为 stub。
        if let Some(canonical) = signature_canonical {
            let text = value_to_string(&messages[idx].content);
            let stub = render_dedup_tool_stub(
                DedupToolStubKind::IdenticalCall,
                &occurrence,
                &canonical,
                &text,
            );
            messages[idx].content = Value::String(stub);
        }
    }
}

#[derive(Clone, Copy)]
enum DedupToolStubKind {
    ByteIdentical,
    IdenticalCall,
}

#[derive(Clone)]
struct DedupToolOccurrence {
    message_idx: usize,
    tool_name: String,
    tool_call_id: String,
    args_norm: String,
    args_raw: String,
    target: Option<String>,
}

fn dedup_tool_occurrence(
    messages: &[Message],
    idx: usize,
    id_to_signature: &rustc_hash::FxHashMap<String, (String, String)>,
    id_to_args_raw: &rustc_hash::FxHashMap<String, String>,
) -> Option<DedupToolOccurrence> {
    let tool_call_id = messages[idx].tool_call_id.as_deref()?;
    let (tool_name, args_norm) = id_to_signature.get(tool_call_id)?;
    let args_raw = id_to_args_raw
        .get(tool_call_id)
        .map(String::as_str)
        .unwrap_or(args_norm.as_str());
    Some(DedupToolOccurrence {
        message_idx: idx,
        tool_name: tool_name.clone(),
        tool_call_id: tool_call_id.to_string(),
        args_norm: args_norm.clone(),
        args_raw: args_raw.to_string(),
        target: dedup_tool_target_summary(tool_name, args_raw),
    })
}

fn tool_uses_content_identity_dedup(tool_name: &str) -> bool {
    is_non_compressible_tool(tool_name) || tool_name == "tree"
}

/// 内容是否为「已外溢/截断的归档 stub」——即本身就不是完整结果，只是一个指向
/// 磁盘原文的召回指针（`[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]` 或
/// `[context-overflow-truncated]`）。byte-identical dedup 遇到这类内容时必须
/// 跳过折叠：canonical 与副本逐字节相同 ⇒ canonical 同样是截断 stub，谎称
/// "reuse the canonical full result" 会把模型导向拿不到原文的回指链。
fn is_content_overflow_archived_stub(content: &Value) -> bool {
    if is_preserved_tool_overflow_content(content) {
        return true;
    }
    content.as_str().is_some_and(|text| {
        text.trim_start()
            .starts_with("[context-overflow-truncated]")
    })
}

/// Marker prefix identifying an already-folded tool-result stub produced by
/// [`render_dedup_tool_stub`]. Must stay equal to the literal "[deduped:"
/// embedded in that function's output; the dedup pass skips content starting
/// with it so persisted stubs are never re-rendered into nested stubs.
const DEDUP_STUB_MARKER_PREFIX: &str = "[deduped:";

/// Hard cap for the raw content prefix embedded in byte-identical dedup stubs.
/// Each stub carries its own bounded excerpt so it stays useful even after the
/// canonical occurrence gets folded away in a later pass (historical failure
/// mode: the stub pointed at a canonical that no longer existed verbatim in the
/// projection, so the model re-read identical data forever). Arbitrarily large
/// source results stay safe: the excerpt is truncated from the in-memory string,
/// so cost per duplicate occurrence is capped no matter how much a tool printed.
const DEDUP_STUB_EXCERPT_MAX_CHARS: usize = 1_600;

/// Char-boundary-safe prefix of at most `max_chars` characters.
fn char_prefix_capped(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &text[..byte_idx],
        None => text,
    }
}

fn render_dedup_tool_stub(
    kind: DedupToolStubKind,
    original: &DedupToolOccurrence,
    canonical: &DedupToolOccurrence,
    removed_content: &str,
) -> String {
    let mut out = match kind {
        DedupToolStubKind::ByteIdentical => format!(
            "[deduped: byte-identical `{}` result is preserved verbatim at a newer occurrence; content unchanged. No need to re-read - reuse the canonical full result.]\n",
            original.tool_name
        ),
        DedupToolStubKind::IdenticalCall => format!(
            "[deduped: identical `{}` call repeated later in this conversation; full result preserved at the newest occurrence.]\n",
            original.tool_name
        ),
    };
    out.push_str(&format!(
        "- original_tool_call_id: {}\n- canonical_tool_call_id: {}\n- canonical_message_index: {}\n",
        original.tool_call_id, canonical.tool_call_id, canonical.message_idx
    ));
    out.push_str(&format!(
        "- original_args: {}\n",
        render_dedup_args(&original.args_raw)
    ));
    if let Some(target) = original.target.as_deref() {
        out.push_str(&format!("- original_target: {target}\n"));
    }
    if original.args_norm != canonical.args_norm {
        out.push_str(&format!(
            "- canonical_args: {}\n",
            render_dedup_args(&canonical.args_raw)
        ));
    }
    if original.target != canonical.target
        && let Some(target) = canonical.target.as_deref()
    {
        out.push_str(&format!("- canonical_target: {target}\n"));
    }
    out.push_str(&format!(
        "- preview: {}",
        render_dedup_preview(removed_content)
    ));
    if matches!(kind, DedupToolStubKind::ByteIdentical) {
        // By construction removed_content equals the canonical body here, so a raw
        // prefix truthfully represents the newest copy. Keep it multi-line and
        // un-normalized: models reuse these stubs for verbatim patching, which the
        // lossy single-line preview above cannot serve.
        let body = removed_content.trim_start();
        let excerpt = char_prefix_capped(body, DEDUP_STUB_EXCERPT_MAX_CHARS);
        let total_chars = body.chars().count();
        let shown_chars = excerpt.chars().count();
        out.push_str(&format!(
            "\n- canonical_first_chars: {shown_chars} of {total_chars}\n<<<DEDUP_EXCERPT\n{excerpt}"
        ));
        if shown_chars < total_chars {
            out.push_str(
                "\n(excerpt truncated; see the canonical occurrence or original_target for the rest)",
            );
        }
        out.push_str("\nDEDUP_EXCERPT>>>");
    }
    out
}

fn render_dedup_args(args: &str) -> String {
    let rendered = serde_json::from_str::<Value>(args)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| normalize_whitespace(args));
    truncate_to_chars(&rendered, 720)
}

fn render_dedup_preview(content: &str) -> String {
    let content = content.trim();
    if content.is_empty() {
        return "<empty>".to_string();
    }
    let preview = summarize_text(content, 520);
    truncate_to_chars(&normalize_whitespace(&preview), 520)
}

fn dedup_tool_target_summary(tool_name: &str, args: &str) -> Option<String> {
    let args = serde_json::from_str::<Value>(args).ok()?;
    let mut fields = Vec::new();
    match tool_name {
        "read_file" => {
            if let Some(path) = dedup_arg_string(&args, &["file_path", "path", "filePath"]) {
                fields.push(format!(
                    "file={}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }
            if let Some(range) = dedup_read_file_range_summary(&args) {
                fields.push(range);
            }
        }
        "execute_command" | "run_command" | "shell" | "bash" => {
            if let Some(command) = dedup_arg_string(&args, &["command"]) {
                fields.push(format!(
                    "command={}",
                    truncate_to_chars(&normalize_whitespace(&command), 360)
                ));
            }
            if let Some(cwd) = dedup_arg_string(&args, &["cwd"]) {
                let cwd = normalize_whitespace(&cwd);
                if !cwd.is_empty() {
                    fields.push(format!("cwd={}", truncate_to_chars(&cwd, 240)));
                }
            }
        }
        _ => {
            for key in [
                "file_path",
                "path",
                "filePath",
                "pattern",
                "query",
                "command",
            ] {
                if let Some(value) = args.get(key).and_then(Value::as_str) {
                    fields.push(format!(
                        "{key}={}",
                        truncate_to_chars(&normalize_whitespace(value), 240)
                    ));
                }
            }
        }
    }

    (!fields.is_empty()).then(|| fields.join("; "))
}

fn dedup_arg_string(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(ToOwned::to_owned)
}

fn dedup_arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key)
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn dedup_read_file_range_summary(args: &Value) -> Option<String> {
    let start_line = dedup_arg_u64(args, "startLine");
    let end_line = dedup_arg_u64(args, "endLine");
    if let (Some(start_line), Some(end_line)) = (start_line, end_line) {
        return Some(format!("range=lines:{start_line}..{end_line}"));
    }

    let offset = dedup_arg_u64(args, "offset");
    let limit = dedup_arg_u64(args, "limit");
    match (offset, limit) {
        (Some(offset), Some(limit)) if limit > 0 => Some(format!(
            "range=lines:{}..{}",
            offset,
            offset.saturating_add(limit.saturating_sub(1))
        )),
        (Some(offset), _) => Some(format!("range=offset:{offset}")),
        (None, Some(limit)) => Some(format!("range=first:{limit}")),
        _ => None,
    }
}

#[derive(Clone)]
struct NumberedReadFileResult {
    message_idx: usize,
    tool_call_id: String,
    path: String,
    lines: Vec<(usize, String)>,
}

fn dedup_overlapping_read_file_results(
    messages: &mut [Message],
    id_to_signature: &rustc_hash::FxHashMap<String, (String, String)>,
    protected_indices: &rustc_hash::FxHashSet<usize>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
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
        let Some(lines) = parse_numbered_read_file_output_lines(&text) else {
            continue;
        };

        // 近端完整工具组必须逐字保留，避免下一轮模型看到被处理过的刚读取内容。
        if protected_indices.contains(&idx) || protected_tool_call_ids.contains(tool_call_id) {
            prior_results.push(NumberedReadFileResult {
                message_idx: idx,
                tool_call_id: tool_call_id.clone(),
                path,
                lines,
            });
            continue;
        }

        for prior in &mut prior_results {
            if protected_indices.contains(&prior.message_idx)
                || protected_tool_call_ids.contains(&prior.tool_call_id)
                || prior.path != path
            {
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
                Value::String(render_deduped_read_file_output_lines(&prior.lines, removed));
        }

        prior_results.push(NumberedReadFileResult {
            message_idx: idx,
            tool_call_id: tool_call_id.clone(),
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

fn parse_numbered_read_file_output_lines(text: &str) -> Option<Vec<(usize, String)>> {
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

fn render_deduped_read_file_output_lines(lines: &[(usize, String)], removed: usize) -> String {
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
mod dedup_adjacent_tests;
#[cfg(test)]
mod fold_early_tool_groups_tests;
#[cfg(test)]
mod overflow_stub_merge_tests;
#[cfg(test)]
mod shrink_successful_write_arguments_tests;
#[cfg(test)]
mod tail_window_tests;
#[cfg(test)]
mod truncate_last_real_user_message_tests;
