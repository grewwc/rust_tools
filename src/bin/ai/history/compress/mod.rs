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
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS, count_tool_group_anchors,
    first_trim_candidate, is_protected_leading_system_like_message, plan_early_tool_groups,
    recent_tool_group_message_indices, recent_tool_result_groups,
    ToolGroupFoldPlan,
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

/// Physical cap for a single raw tool result in the request context. Canonical
/// history is unaffected.
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

/// Prefixes of all "auto-compaction summary" notes. The writer side (which
/// generates summary notes) and the recognizer side (duplicate guard, sqlite
/// resume points, request-side grouping) **must share this one list**; otherwise
/// the two sides split apart — "prefixes written are not recognized". Historically
/// `长期记忆摘要（压缩保留）` was never registered and thus bypassed the
/// duplicate guard, causing a summary note to be re-inserted every turn, the
/// context budget to creep up continuously, and the compaction pipeline to spin
/// on every turn. When adding a new summary prefix, change only this list.
///
/// Note: entries must be bare prefixes "after leading whitespace is stripped";
/// detection uniformly goes through [`is_summary_note_text`], which `trim_start`s
/// first and then checks `starts_with` one by one, so full-width/half-width
/// colons only need to be listed once each.
pub(in crate::ai) const SUMMARY_NOTE_PREFIXES: &[&str] = &[
    "对话摘要（自动压缩",
    "历史摘要（自动压缩",
    "长期记忆摘要（压缩保留）",
    "[mid-turn-summary]",
];

/// Marker of the deterministic evidence note generated when folding a tool group.
///
/// This is not an LLM-generated summary; it is evidence/checkpoint extracted
/// mechanically by the compressor from tool_call arguments and tool results. It
/// must survive secondary summarization; otherwise long tool chains degrade into
/// a tool bill of bare file_path / original_file_path entries, and the model
/// tends to re-gather evidence after compaction.
pub(super) const COMPRESSED_TOOL_EVIDENCE_MARKER: &str = "[compressed-tool-evidence]";

/// Prefix of archive-pointer notes (back-references to overflow originals). They
/// appear paired with summary notes; the P1 folding logic relies on this to
/// recognize and dedupe piled-up archive pointers.
pub(in crate::ai) const ARCHIVE_NOTE_PREFIX: &str = "长期记忆归档";

/// Whether a piece of text is the body of an "auto-compaction summary" note
/// (prefix match, tolerant of leading whitespace). This is the **single source of
/// truth** for summary detection, shared by the guard / sqlite / request
/// normalization paths.
pub(in crate::ai) fn is_summary_note_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    SUMMARY_NOTE_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

/// Whether a piece of text is an overflow archive-pointer note.
pub(in crate::ai) fn is_archive_note_text(text: &str) -> bool {
    text.trim_start().starts_with(ARCHIVE_NOTE_PREFIX)
}

const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;
/// Dynamic bounds of the recent user-turn tail window protected during compaction
/// fallback (first_trim_candidate). Small contexts prefer keeping 3 turns to
/// improve multi-stage task continuity; very large contexts fall back to 2 turns
/// to control the budget.
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN: usize = 2;
const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX: usize = 3;
/// When the context char count is at or below this threshold, prefer keeping 3
/// recent user turns.
const KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS: usize = 48_000;

/// Number of "recent user-turn" tail-window turns to fully exempt from
/// trimming/spilling/folding.
///
/// The base rule (pick one by total size) is unchanged: <=48K -> 3 turns,
/// otherwise 2 — zero behavior change for normal sessions.
///
/// **Byte-cap escape valve** (active when `budget > 0`): the protected tail window
/// is a "full exemption zone" and must not itself exceed the entire history
/// budget. Tool-heavy agentic sessions (few user turns x hundreds of tool calls
/// per turn) can balloon the tail window to MB scale and **structurally prevent
/// convergence** — even hundreds of tool groups inside the window are all
/// exempted. In that case shrink the protected turn count step by step, exposing
/// tool groups "from the second-to-last turn and earlier" to the fold/spill paths
/// to restore convergence. **Floor invariant: never below 1 turn** — the newest
/// user turn and its tool groups are always kept verbatim (the group-level
/// protection of `KEEP_RECENT_TOOL_GROUPS` remains the backstop).
///
/// `budget == 0` means the caller explicitly sets no cap (old behavior kept), for
/// reuse in contexts without a budget.
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

/// Batch trimming cannot recompute protection boundaries mid-execution, so a
/// low-budget target must adopt the sub-48K three-turn protection policy from the
/// start; otherwise the third-most-recent user turn may already have been deleted
/// before the total crosses 48K.
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

/// Constant accessors exposed to the rest of the crate, avoiding duplicated
/// threshold numbers in mod.rs.
pub(in crate::ai) fn persisted_history_keep_recent_turns() -> usize {
    PERSISTED_HISTORY_KEEP_RECENT_TURNS
}

/// Maximum number of self_note entries kept in the messages array. self_notes are
/// already persisted to MemoryStore (`memory_store::AgentMemoryEntry`); the copy
/// in messages is only the "redundant inline copy" the LLM saw within the same
/// turn. Over a long session with thousands of turns these inline copies bloat
/// monotonically and need sliding-window pruning.
const MAX_SELF_NOTES_IN_MESSAGES: usize = 8;
/// Total char cap for keeping mechanical evidence of older tool groups verbatim
/// in the model context. Older evidence is appended to overflow-history.md with
/// zero compression, and only a unified back-reference is kept in messages.
const MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS: usize = 12_000;
const CONTEXT_CHECKPOINT_MARKER_PREFIX: &str = "[context_checkpoint";
pub(in crate::ai) const QUERY_MEMORY_INDEX_PREFIX: &str = "[query-memory-index-v1]";

pub(in crate::ai) fn compressed_tool_evidence_inline_chars_limit() -> usize {
    MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS
}

/// Keep only the `self_note:` entries among the most recent `keep_recent`
/// internal_notes. Other internal_notes (cache hints, loop-breakers, history
/// summaries) are outside the pruning scope.
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

/// The checkpoint body is already written to the session asset; this short
/// marker is the model's only index for relocating that body after
/// compression, so history compression must never swallow or drop it. The
/// request projection may replace many old markers with a hierarchical index
/// holding every exact path; that index is protected by the same rule, and the
/// original markers always stay in canonical history.
pub(in crate::ai) fn is_context_checkpoint_marker(m: &Message) -> bool {
    if m.role != ROLE_INTERNAL_NOTE {
        return false;
    }
    let content = value_to_string(&m.content);
    let content = content.trim_start();
    content.starts_with(CONTEXT_CHECKPOINT_MARKER_PREFIX)
        || content.starts_with(QUERY_MEMORY_INDEX_PREFIX)
}

pub(in crate::ai) fn is_compressed_tool_evidence_note(m: &Message) -> bool {
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

    /// Atomically persist the planning-phase archive to disk. The path contains a
    /// content fingerprint, so an already-existing file can be reused directly;
    /// even if concurrent writers write temp files at the same time, only one
    /// deterministic target file ends up remaining.
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
            // Trust an index hit only if the archive it describes is still
            // readable. `exists()` alone is insufficient: an in-place truncate
            // leaves the file present but empty, so require a non-zero length.
            let archive_readable = std::fs::metadata(&self.path)
                .map(|meta| meta.len() > 0)
                .unwrap_or(false);
            if archive_readable {
                return true;
            }
            // The archive body vanished or was emptied while the index survived
            // (external cleanup, move, or truncate of overflow-history.md).
            // Trusting a stale entry here would report success while skipping
            // payloads readers can no longer find, so reset the in-memory set
            // and drop the sidecar; both repopulate from fresh appends below.
            self.seen_payloads.clear();
            let _ = std::fs::remove_file(self.fingerprint_index_path());
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

/// Folded tool-group notes are a short-term recall window for older evidence, not
/// a ledger that stays fully inlined forever. Keep the most recent contiguous
/// window that fits a fixed char budget; older notes are archived with zero
/// compression first and removed from messages only after the write succeeds.
/// This keeps long tool chains from crowding the context with hundreds of ~1 KiB
/// notes.
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
    // The history store may still hold legacy JSON stubs. They are an internal
    // protocol of the compressor and must not be handed to the model as-is,
    // otherwise the model treats them as ordinary user text or even repeats them
    // verbatim in its final reply.
    normalize_preserved_message_stubs_for_model(&mut messages);
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    // compressed_tool_round notes are themselves compaction products; without an
    // independent cap they accumulate one by one before the global history budget
    // triggers, forming another kind of linear context bloat.
    messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir.as_deref());

    // Prune the self_note sliding cap before large-block compaction, so the
    // self_notes accumulated over thousands of turns (already written to
    // MemoryStore; the copy in messages is just a redundant backup) do not bloat
    // monotonically. MemoryStore still keeps every record.
    let messages = trim_self_notes_to_recent(messages, MAX_SELF_NOTES_IN_MESSAGES);

    // Converge duplicate summary/archive notes piled up by past duplicate-guard
    // breakage. Doing this at the request-time entry lets an old session that
    // already piled up dozens of note pairs recover on its very next request
    // instead of waiting for a flush.
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
                // When the summary budget is 0 (e.g. second-round compaction in the
                // production path) no summary is rebuilt; the old summary/archive
                // notes are themselves "the compressed representation of the early
                // conversation" and must be kept like the checkpoint marker,
                // otherwise the summary prepare_turn already produced would be
                // silently dropped in the second compaction round.
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

/// Char cap applied to "assistant narration carrying tool_calls" in the persisted
/// history.
///
/// The folder [`tool_groups::fold_tool_call_group_to_stub`] uses the visible
/// narration before this turn's tool calls as the source of
/// `assistant_checkpoint`. Except for continuation state the model protocol
/// explicitly requires replaying, full reasoning_content is never persisted and
/// must never be promoted into assistant body text; tool-call-only messages get
/// a safe operation summary rebuilt by the folder from structured tool_calls.
/// 720 chars is the same order of magnitude as the post-fold checkpoint cap.
const PERSISTED_TOOL_CALL_ASSISTANT_NARRATION_MAX_CHARS: usize = 720;
pub(in crate::ai) const PERSISTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-content-replay:v1\u{1f}";

/// exact-replay continuation state exists only in the rebuildable context
/// projection. The payload carries the originating model, so switching models
/// never mistakes another provider's hidden state for the current model's resume
/// state.
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

/// Replay prefix for encrypted reasoning under the Responses protocol. Kept
/// separate from exact-replay (`PERSISTED_REASONING_REPLAY_PREFIX`) because the
/// payload shapes differ: exact stores a plaintext reasoning string; encrypted
/// stores the reasoning output-item delivered by the provider (a JSON array with
/// `encrypted_content`). Separate prefixes prevent the request side from mixing
/// the two payload kinds and let the compaction/sanitize layers handle both with
/// the same "keep if marked" rule.
pub(in crate::ai) const PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-encrypted-replay:v1\u{1f}";

/// Runtime master switch for cross-turn encrypted reasoning replay. On by
/// default; setting `AIOS_DISABLE_ENCRYPTED_REPLAY=1` short-circuits persistence
/// and request-side rebuild, for A/B experiments reproducing the "pre-fix"
/// behavior (encrypted reasoning lost across turns/resume). Experimental
/// scaffolding only; default product behavior is unchanged.
pub(in crate::ai) fn encrypted_reasoning_replay_runtime_enabled() -> bool {
    std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY")
        .map(|v| v.trim().is_empty() || v == "0")
        .unwrap_or(true)
}

/// Encode the encrypted reasoning items captured this turn, together with the
/// originating model, into a single string for persisting into
/// `reasoning_content`. Carries a model marker: when switching/falling back to
/// another model, request-side decoding drops it on model mismatch, avoiding
/// feeding model A's encrypted state to model B (the provider would 400).
pub(in crate::ai) fn encode_encrypted_reasoning_replay_state(
    model: &str,
    items: &[Value],
) -> String {
    format!(
        "{PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX}{}",
        serde_json::json!({ "model": model, "items": items })
    )
}

/// Decode encrypted reasoning items from the persisted `reasoning_content`.
/// Returns them only when the originating model inside the marker matches the
/// current request model; otherwise returns `None` (no cross-model replay).
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
    // The gateway re-delivers the same reasoning resource twice — `.added`
    // (partial payload) and `.done` (full payload). The pre-fix accumulator
    // deduped by all-fields equality and failed to converge, so history may hold
    // two entries with the same id. Dedupe by id, keeping the later one (`.done`
    // is the protocol's final authoritative state); otherwise replay emits the
    // same resource id twice and modelhub returns 400 (-4003 Duplicate item found).
    dedup_reasoning_items_by_id(&mut items);
    Some(items)
}

/// Converge reasoning items by `id`: keep the later entry for the same resource.
///
/// The gateway re-delivers the same reasoning resource twice — `.added` (partial
/// payload) and `.done` (full payload); same id, different content, so
/// all-fields-equality dedup judges them unequal and leaves duplicate ids behind.
/// Replay then emits the same resource id twice and modelhub returns 400 (-4003
/// Duplicate item found). Converge by id here and keep the later entry: within a
/// stream `.done` always follows `.added` and is the protocol's final
/// authoritative state (carrying the full payload), so last-writer-wins naturally
/// picks it. Items without an `id` are never merged (keep all of them to avoid
/// wrong deletions).
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

    // The persisted history keeps only the assistant facts truly needed across
    // turns:
    // - `reasoning_content` is hidden reasoning; the persistence layer always
    //   drops it and never copies it into visible body text. When the provider
    //   needs the field shape, the request layer fills in an empty string.
    // - Assistant narration carrying tool_calls must not be emptied: otherwise
    //   the checkpoint of [`tool_groups::fold_tool_call_group_to_stub`] sees no
    //   text at all and collapses into
    //   "assistant_checkpoint: <empty; no persisted decision before these tool calls>"，
    //   leaving the model amnesic after compaction, re-gathering evidence from
    //   the same turn.
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
                // Already a continuation state carrying the internal marker (exact
                // plaintext / responses encrypted); stay idempotent.
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

/// Build the persisted projection according to the model protocol. Only models
/// that explicitly declare they need verbatim replay keep hidden reasoning for
/// tool-call assistant messages; final answers and other messages are still
/// always dropped.
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
        // Only reasoning carrying the internal marker is continuation state that
        // the runtime explicitly kept according to model capability; legacy
        // history, imported files, and bare reasoning from other models are still
        // dropped per the original policy.
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

/// Converge the multiple summary/archive notes piled up by past duplicate-guard
/// breakage.
///
/// Background: the `长期记忆摘要（压缩保留）` prefix was once not registered in
/// `is_summary_message`, so every compaction round re-inserted a "summary +
/// archive" note pair at the top; a long session could pile up dozens of pairs,
/// polluting the context budget and inflating `total_chars` until the compaction
/// pipeline spun on every turn.
///
/// Folding policy (lossless):
/// - **Summary notes**: dedupe and concatenate each note body (header stripped)
///   in original order into **one** note, put back where the first summary sat.
///   The "initial goal" each evicted round recorded is therefore fully kept.
/// - **Archive-pointer notes**: keep only one when contents are identical, keep
///   all when they differ, placed right after the merged summary — avoids losing
///   back-references to other archive files when importing/migrating sessions.
/// - All other messages are kept verbatim and in order (non-summary/archive
///   messages are never touched).
///
/// Fold only when there is more than one summary or identical archive pointers
/// exist, avoiding pointless rewriting of healthy history (when the return value
/// equals the input entry by entry, the caller's `compacted == messages` check
/// skips persisting).
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

    // Merge all summary bodies and dedupe archive pointers with identical
    // content; both keep their original order.
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

    // Rebuild the sequence: put the merged summary plus deduped archive pointers
    // at the position of "the first summary/archive note", drop the other
    // summary/archive notes, and keep every other message as-is.
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
            // Remaining summaries and already-collected archive notes are dropped.
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

/// Inject an archive back-reference after the leading summary; idempotent when an
/// identical back-reference already exists, so each compaction round does not
/// append another identical `internal_note` at the top of the context.
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
            // Synthetic user messages (image followups etc.) do not form a real
            // turn boundary, avoiding premature history truncation.
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

/// Proactive compaction triggered by a task boundary (a turn ended with no
/// further tool calls, meaning the agent gave its final answer). The threshold is
/// lowered from `MAX_HISTORY_TURNS` (200) to `PERSISTED_HISTORY_KEEP_RECENT_TURNS`
/// (160), so the natural "task done" boundary triggers summarization earlier
/// instead of passively switching only when the hard cap is hit. Conversations
/// below 160 turns are still never summarized, so short sessions are unaffected.
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

/// The next escalation when `first_tool_call_group` refuses to fold (all
/// remaining foldable groups contain non-compressible tools like `read_file`,
/// which it rejects by policy) but the budget is still exceeded: use
/// [`fold_early_tool_groups`] to progressively fold those groups "outside the
/// protected tail window" into single-line `compressed_tool_round` notes (each
/// carrying a file_path recall anchor the model can read back with read_file).
///
/// This reuses **the same** battle-tested folding function as Path B+C of
/// `mid_turn_llm_summarize`, just moved earlier into the regular/persisted
/// compaction path — fixing the root cause of "tool-heavy sessions (few user
/// turns x hundreds of read_file calls) never folding tool groups inside
/// `compress_messages_for_context` / `shrink_*`, leaving the whole history unable
/// to converge into the budget".
///
/// Returns whether an "effective fold" happened (net char decrease). `keep_recent`
/// tightens progressively from [`KEEP_RECENT_TOOL_GROUPS`] down to
/// [`MIN_KEEP_RECENT_TOOL_GROUPS`] (=1), keeping the most recent tool groups
/// verbatim as much as possible and widening the folding scope step by step only
/// while still over budget; every step must produce a net decrease to avoid
/// spinning without progress. **Never tightens to 0**: a window of 0 would fold
/// the most recent tool interaction into a stub too, leaving the model with no
/// structured tool context at all; the remaining excess is handled by the later
/// `first_trim_candidate` / `truncate_mutable_messages_to_fit` backstops in the
/// while loop.
fn fold_noncompressible_tool_groups_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    // The total is a pure function of the (not yet mutated) messages; compute
    // it once and reuse it for the entry guard and every comparison below.
    let base_total = messages_total_chars(messages);
    if base_total <= max_chars {
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
    // Anchor count is likewise invariant while this loop only reads messages;
    // planning (which deep-clones the entire history) is skipped outright for
    // windows whose plan would be `ToolGroupFoldPlan::unchanged`.
    let anchor_count = count_tool_group_anchors(messages);
    // Plan reuse: a plan is a pure function of the unchanged messages, so the
    // last reducing candidate can be committed directly when its window is the
    // chosen one instead of being discarded and re-planned afterwards. This
    // caps the ladder at one live plan (one whole-history clone) instead of a
    // clone per rung plus a final re-plan; every selection decision below is
    // evaluated exactly as before.
    let mut last_reducing_plan: Option<(usize, ToolGroupFoldPlan)> = None;
    for &keep_recent in progressive_fold_windows().iter() {
        if anchor_count <= keep_recent {
            continue;
        }
        let plan =
            plan_early_tool_groups(messages, keep_recent, overflow_dir, protected_tool_call_ids);
        if plan.folded_groups() == 0 {
            continue;
        }
        // A plan must net a strict decrease; drop it otherwise and keep tightening
        // keep_recent to guard against livelock where the group count changes but
        // the char count does not.
        let plan_total = messages_total_chars(plan.messages());
        if plan_total >= base_total {
            continue;
        }
        deepest_reducing_keep = Some(keep_recent);
        let plan_fits = plan_total <= max_chars;
        let floor_safe = plan_fits
            && protected_tail_message_chars(plan.messages(), keep_recent)
                >= MIN_PROTECTED_TAIL_CHARS;
        if plan_fits {
            if fitting_keep.is_none() {
                fitting_keep = Some(keep_recent);
            }
        }
        last_reducing_plan = Some((keep_recent, plan));
        if floor_safe {
            floor_safe_fitting_keep = Some(keep_recent);
            break;
        }
    }
    let chosen = floor_safe_fitting_keep
        .or(fitting_keep)
        .or(deepest_reducing_keep);
    let mut made_progress = false;
    if let Some(keep_recent) = chosen {
        // Reuse the remembered plan when it was built for the chosen window;
        // the only case it is not (a first fitting window superseded by later
        // reducing windows) falls back to the single re-plan, which produces a
        // byte-identical plan because the messages have not changed since the
        // selection loop ran.
        let plan =
            match last_reducing_plan.take() {
                Some((window, plan)) if window == keep_recent => plan,
                _ => {
                    plan_early_tool_groups(messages, keep_recent, overflow_dir,
                        protected_tool_call_ids)
                }
            };
        if plan.folded_groups() > 0
            && messages_total_chars(plan.messages()) < base_total
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

/// Batch-remove trimmable ordinary messages and archive them in a single flush.
/// The old implementation re-entered the outer loop and ran `sync_data` after
/// every single removal, so a tool-heavy history amplified hundreds of assistant
/// messages into hundreds of synchronous writes. Here the whole batch is trimmed
/// on a candidate copy first; if archiving fails the candidate is not adopted and
/// the original messages stay unchanged.
fn trim_removable_messages_batch(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    // Single-pass scan + rebuild, replacing the old "first_trim_candidate +
    // Vec::remove per round": the old loop re-scanned everything every round
    // (keep_recent_user_turns_when_trimming / retained_turn_start / the leading
    // protected run, each O(n)), and removal was an O(n) memmove — O(n²) overall,
    // visibly stalling histories with thousands of tool-heavy entries. The
    // protected tail window and total char count are computed once up front;
    // afterwards only an O(n) scan + O(n) rebuild run.
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
        // Skip the whole leading protected system-like run (system prompt, history
        // summaries, archive pointers, checkpoints), matching first_trim_candidate
        // semantics.
        let head_protected =
            in_protected_leading_run && is_protected_leading_system_like_message(&message);
        if head_protected {
            kept.push(message);
            index += 1;
            continue;
        }
        in_protected_leading_run = false;

        // Same deletability rule as first_trim_candidate: checkpoints, spill
        // stubs, tool messages, and assistant(tool_calls) cannot be removed
        // singly. user messages are not removable on this path (OffloadOnly, spill
        // only) — skip rather than break, so the many trimmable candidates after
        // the first user message keep their chance of batch removal (the old
        // behavior broke out and left everything to the truncate backstop,
        // inconsistent with the with_summary "drop + archive" semantics). The
        // total char count is maintained exactly: subtract message_billable_chars
        // per removal and stop as soon as total <= max_chars, matching the old
        // loop's stop condition.
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
    // Ordinary messages still go to the unified history archive; internal_notes
    // are written to a deterministic file keyed by content fingerprint. The
    // latter both avoids silently losing recovery instructions/persisted state
    // and avoids appending the same body again on repeated compaction.
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
    // dedup must run before offload: offload moves over-threshold old read_file
    // bodies to disk and replaces them with a stub carrying a **unique temp
    // path**; once that happens, byte-identical duplicates can no longer be folded
    // because their paths differ. Do content-level dedup first, folding redundant
    // bodies into back-reference stubs, then offload the few versions truly worth
    // keeping.
    dedup_repeated_tool_results(&mut messages, protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut messages,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
    );
    // Unconditionally spill oversized old user/image messages first (except the
    // protected tail window), consistent with
    // `shrink_messages_to_fit_with_summary`. Images are billed at nominal cost in
    // the budget, and once a large user body is moved to disk with zero
    // compression as a stub, the trimming loop below skips them automatically via
    // `is_preserved_user_or_image_stub` — preventing old user messages from being
    // outright `remove`d by generic trimming (which would violate the OffloadOnly
    // semantics assigned to RecentUser and silently lose the original text).
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // Age-fold overflow stub preview bodies outside the protected tail window into
    // single-line anchors (file_path recall is not lost), converging the
    // historical bloat of "hundreds of early read_file previews accumulating
    // monotonically". This runs before the budget check so sessions not yet over
    // budget also keep converging already-spilled stubs. The tail-window turn
    // count is bounded by the max_chars byte cap: when a tool-heavy session's tail
    // window grows too large it shrinks automatically, exposing older stubs to age
    // folding.
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);
    // user/image spill stubs have no tool anchor to age-fold: their preview is
    // already a single-line pointer, and first_trim_candidate / truncate /
    // emergency cap never touch them again, so long sessions accumulate stubs
    // monotonically (especially image messages when the 512 threshold is below
    // their nominal cost). Merge old stubs outside the protected tail window into
    // one pointer carrying the archive directory, converging placeholder overhead
    // from O(N) to O(1).
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // Proactively slim down the giant arguments of write_file/apply_patch calls
    // that were "successfully written": once the file is on disk and the result
    // confirms success, the full body no longer has semantic value, so it can be
    // replaced with an archive stub without waiting for budget pressure. Anything
    // inside the protection window (including groups just written this turn,
    // whose bodies the model may immediately reference to build follow-up edits)
    // and failed results are always kept, so agent effectiveness does not degrade.
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        // Fold all unprotected tool groups over budget in one batch (both
        // compressible and non-compressible go through [`fold_early_tool_groups`]).
        // The old implementation folded only one group per iteration in the
        // `first_tool_call_group` + single-group fold loop, and only fell through
        // to the batch fold of `fold_noncompressible_tool_groups_to_fit` after
        // nothing foldable remained. Bug A kept the per-group savings tiny
        // (assistant.content had already been sanitized to `""`/`null`, so the
        // folded stub was nearly as large as the original group) -> the outer while
        // needed dozens of iterations to converge, each round also injecting an
        // `<empty>` empty-checkpoint note that polluted the context (see the 22
        // consecutive `compressed_tool_round` <empty> stubs in the e75fc2e5
        // session dump). Now each round first uses one `fold_early_tool_groups`
        // batch to collect every foldable group at once, finishing the shrink
        // within a few outer iterations.
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages, max_chars) {
            // Old user messages (including multimodal ones with images) are never
            // silently deleted: that is the OffloadOnly semantics assigned to
            // RecentUser. First try moving the original text to the archive file
            // with zero compression and replacing it with a back-reference stub;
            // if the move succeeds, continue the trimming loop.
            if messages[idx].role == "user" {
                if let Some(dir) = overflow_dir
                    && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
                {
                    continue;
                }
                // Cannot spill (no overflow_dir, or the body is too small, or the
                // proactive spill above already handled every over-threshold
                // user): break out of the trimming loop directly and never
                // `remove` the user original text. The residual slight overage is
                // left to the upper hard-threshold `mid_turn_llm_summarize`
                // backstop, avoiding a livelock where the same small user message
                // keeps getting picked.
                break;
            }
            // Archive the remaining trimmable candidates (plain assistant
            // narration, compressed_tool_round, etc.) in one place, avoiding
            // per-entry append + sync_data. If batch archiving fails, the original
            // messages stay unchanged.
            if trim_removable_messages_batch(&mut messages, max_chars, overflow_dir) {
                continue;
            }
            break;
        }
        break;
    }

    // When compressed_tool_evidence is trimmed, its body is appended with zero
    // compression to the unified history archive; the unified back-reference must
    // be put back into the request, otherwise the evidence exists on disk but the
    // model never learns the archive path. The archive note is an internal_note
    // (protected by is_system_like_role, so the truncation below will not cut it),
    // therefore it must be injected **before** `truncate_unprotected_messages_to_fit`,
    // so the final truncation frees the budget it occupies from other trimmable
    // messages, avoiding a payload that is slightly over max_chars. This matches
    // the order in `shrink_messages_to_fit_with_summary`: insert the summary note
    // first, then truncate.
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

/// Index of the nearest alive entry at or below `from` in the real-user
/// aliveness table, or -1 when none remains. Mirrors the "previous remaining
/// user" step of `retained_turn_start` under prefix-only deletions.
fn prev_alive_user(alive: &[bool], from: isize) -> isize {
    let mut cursor = from;
    while cursor >= 0 && !alive[cursor as usize] {
        cursor -= 1;
    }
    cursor
}

/// Single-pass replacement for the sequential per-drop rounds in
/// [`shrink_messages_to_fit_with_summary`]. The old path called
/// `first_trim_candidate` plus `Vec::remove` once per dropped message — an
/// O(n) rescan (including the 48K-threshold base and the byte-capped
/// tail-window recompute inside `keep_recent_user_turns_when_trimming`) and an
/// O(n) memmove per drop, i.e. O(n²) on long histories. This helper collects
/// every candidate those rounds would have dropped in one scan and splices
/// them out once. Selection is provably identical to the sequential loop:
///
/// - candidate predicate and protected leading run match `first_trim_candidate`
///   verbatim; the leading run can only grow (its members are never
///   candidates), and the scan re-extends it after each removal exactly like
///   the sequential recompute would;
/// - deletions always sit strictly before the tail-window boundary, so
///   comparing original indices against an originally-derived boundary is
///   equivalent to the sequential recompute on the shrinking sequence;
/// - `keep_recent_user_turns` has at most two values during the stretch
///   (totals only decrease, so the 48K base flips at most 2 -> 3); the
///   byte-cap component depends only on the protected tail sums, which are
///   invariant under prefix deletions, hence both variants are precomputed
///   from `keep_recent_user_turns_when_trimming`'s own formula;
/// - `fold_noncompressible_tool_groups_to_fit` cannot flip from false to true
///   mid-stretch: removing non-tool-group singletons leaves every fold plan's
///   folded-group set and net char delta unchanged, so one fold check per
///   outer round (kept at the call site) reproduces the interleaving;
/// - `dropped` / `dropped_internal_notes` keep the sequential (ascending)
///   removal order, `total` is decremented with the same
///   `saturating_sub(billable)` arithmetic per removal, and the rollback
///   snapshot is taken immediately before the first accepted removal.
fn drop_trim_candidates_batch(
    messages: &mut Vec<Message>,
    max_chars: usize,
    total: &mut usize,
    messages_before_first_drop: &mut Option<Vec<Message>>,
    dropped: &mut Vec<Message>,
    dropped_internal_notes: &mut Vec<Message>,
) -> usize {
    let len = messages.len();
    if len == 0 || *total <= max_chars {
        return 0;
    }
    // Per-message billable chars cached once: the sequential loop re-charged
    // every message via a full `messages_total_chars` rescan each round.
    let chars: Vec<usize> = messages.iter().map(message_billable_chars).collect();
    // `keep_recent_user_turns_when_trimming` recomputed per round is equivalent
    // to this two-entry table: the byte-cap loop reads only the protected tail
    // sums (invariant under our deletions), and the 48K base depends only on
    // `total`, which moves downwards across the threshold at most once.
    let tail_chars = |keep: usize| -> usize {
        let start = retained_turn_start(messages, keep);
        chars[start..].iter().sum()
    };
    let capped_keep = |base: usize| -> usize {
        let mut keep = base;
        while keep > 1 && tail_chars(keep) > max_chars {
            keep -= 1;
        }
        keep
    };
    let keep2 = capped_keep(2);
    let keep3 = capped_keep(3);
    // Real-user positions mirror `retained_turn_start`'s input list.
    let user_positions: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, message)| {
            (message.role == "user" && !is_runtime_synthetic_user_message(message))
                .then_some(idx)
        })
        .collect();
    let user_count = user_positions.len();
    let mut alive = vec![true; user_count];
    let mut alive_users = user_count;
    let mut keep_now = if *total <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        keep3
    } else {
        keep2
    };
    // Ordinal (into user_positions) of the tail-window boundary user, or -1
    // once `retained_turn_start` would return 0 (protect everything). The
    // selected ordinal is alive_users - keep, walking down as users are
    // deleted or the window grows, exactly like recomputing
    // `retained_turn_start` per round.
    let mut boundary_ptr: isize = if user_count <= keep_now {
        -1
    } else {
        (user_count - keep_now) as isize
    };

    let mut tombstones = vec![false; len];
    let mut removed = 0usize;
    let mut user_ordinal = 0usize;
    let mut idx = 0usize;
    while idx < len && is_protected_leading_system_like_message(&messages[idx]) {
        idx += 1;
    }
    let mut head_run_end = idx;
    while idx < len {
        if idx < head_run_end {
            idx = head_run_end;
            continue;
        }
        let boundary = if boundary_ptr >= 0 {
            user_positions[boundary_ptr as usize]
        } else {
            0
        };
        if idx >= boundary || *total <= max_chars {
            break;
        }
        let message = &messages[idx];
        // Predicate chain copied from `first_trim_candidate`.
        if is_context_checkpoint_marker(message) {
            idx += 1;
            continue;
        }
        if is_preserved_user_or_image_stub(&value_to_string(&message.content)) {
            idx += 1;
            continue;
        }
        if message.role == "tool" {
            idx += 1;
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            idx += 1;
            continue;
        }

        // Accepted candidate: snapshot before the first removal (the caller's
        // sequential loop snapshotted at exactly this point), then tombstone.
        if messages_before_first_drop.is_none() {
            *messages_before_first_drop = Some(messages.clone());
        }
        tombstones[idx] = true;
        removed += 1;
        *total = total.saturating_sub(chars[idx]);
        // Deleting a real user below the window shifts the selected boundary
        // user one alive slot downwards, matching the sequential recompute.
        while user_ordinal < user_count && user_positions[user_ordinal] < idx {
            user_ordinal += 1;
        }
        if user_ordinal < user_count && user_positions[user_ordinal] == idx {
            alive[user_ordinal] = false;
            alive_users -= 1;
            user_ordinal += 1;
            if boundary_ptr >= 0 {
                boundary_ptr = prev_alive_user(&alive, boundary_ptr - 1);
            }
        }
        // Crossing the 48K threshold widens the protection window (base 2 ->
        // 3); totals never increase inside a stretch, so this fires at most
        // once per stretch.
        let new_keep = if *total <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
            keep3
        } else {
            keep2
        };
        while keep_now < new_keep {
            keep_now += 1;
            if boundary_ptr >= 0 {
                if alive_users <= keep_now {
                    boundary_ptr = -1;
                } else {
                    boundary_ptr = prev_alive_user(&alive, boundary_ptr - 1);
                }
            }
        }
        if idx == head_run_end {
            // Removing the first message after the protected leading run lets
            // that run grow over the following messages, exactly like the
            // sequential head-run recompute on the shrunk sequence.
            head_run_end = idx + 1;
            while head_run_end < len
                && is_protected_leading_system_like_message(&messages[head_run_end])
            {
                head_run_end += 1;
            }
            idx = head_run_end;
        } else {
            idx += 1;
        }
        if *total <= max_chars {
            break;
        }
    }
    if removed == 0 {
        return 0;
    }
    // One physical splice instead of one O(n) `Vec::remove` per drop. Each
    // list keeps ascending removal order, matching the sequential pushes.
    let old = std::mem::take(messages);
    let mut kept = Vec::with_capacity(old.len() - removed);
    for (index, message) in old.into_iter().enumerate() {
        if tombstones[index] {
            if is_internal_note_role(&message.role) {
                dropped_internal_notes.push(message);
            } else {
                dropped.push(message);
            }
        } else {
            kept.push(message);
        }
    }
    *messages = kept;
    removed
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
    // dedup before offload: same rationale as shrink_messages_to_fit — avoid
    // byte-identical duplicate read_file bodies each being offloaded into a
    // unique-temp-path stub and losing the chance to fold.
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

    // Unconditionally spill oversized old user/image messages first (except the
    // newest turn's protected tail window). Images are billed at nominal cost in
    // the budget, so a single large image no longer triggers the over-budget loop;
    // they must therefore be moved to files with zero compression before the
    // budget check, avoiding a full base64 payload on every request.
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // Age-fold overflow stub preview bodies outside the protected tail window into
    // single-line anchors (symmetric with shrink_messages_to_fit). Converges the
    // monotonic accumulation of early read_file previews; the file_path recall
    // anchor is kept. The tail-window turn count is likewise bounded by the
    // max_chars byte cap (see keep_recent_user_turns_when_trimming).
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);
    // Symmetric with plain shrink: merge user/image spill stubs outside the
    // protected tail window, preventing placeholder messages from accumulating
    // monotonically as the session grows.
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // Symmetric with shrink_messages_to_fit: proactively replace giant arguments
    // of successfully-written write_file/apply_patch calls with archive stubs
    // (the protection window and failed results are kept).
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }
    let had_leading_summary = messages.first().map(is_summary_message).unwrap_or(false);
    // On archive failure the full pre-removal order must be restored; inserting
    // dropped messages at the head outright would place them before the retained
    // system prompt, breaking the message order the provider requires.
    let mut messages_before_first_drop: Option<Vec<Message>> = None;
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();

    // Runtime char total: single removals subtract message_billable_chars exactly;
    // folds/spills are holistic batch changes recomputed uniformly in their own
    // branches — semantically identical to calling
    // `messages_total_chars(&messages)` every round, but avoids repeatedly
    // O(n)-rescanning the whole message sequence across loop iterations.
    let mut total = messages_total_chars(&messages);
    while total > max_chars {
        // Fold all unprotected tool groups over budget in one batch (both
        // compressible and non-compressible go through [`fold_early_tool_groups`])
        // — same rationale as [`shrink_messages_to_fit`], avoiding a single-group
        // fold loop that iterates dozens of rounds injecting `<empty>`
        // empty-checkpoint notes (see the e75fc2e5 session dump).
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
            total = messages_total_chars(&messages);
            continue;
        }
        // Batch the consecutive drop rounds: selection inside
        // `drop_trim_candidates_batch` reproduces the previous per-round
        // `first_trim_candidate` + `Vec::remove` sequence exactly, while
        // turning O(n) rescans and O(n) memmoves per drop into one scan and
        // one splice (see the helper's doc comment for the equivalence proof).
        if drop_trim_candidates_batch(
            &mut messages,
            max_chars,
            &mut total,
            &mut messages_before_first_drop,
            &mut dropped,
            &mut dropped_internal_notes,
        ) > 0
        {
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
                // flush failed: never delete history. Restore the full pre-removal
                // message snapshot and return immediately — skipping summary/archive
                // note injection (preventing dangling pointer notes without a
                // matching archive file), truncate, and reasoning cleanup. The
                // return value may still be over budget, but that is recoverable
                // (retry compaction next round / request-layer clamp), while data
                // loss is irreversible — honoring the existing lesson of "never
                // delete history when a write fails".
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

/// The last-resort hard-budget escape valve: keep the system/user and tool-call
/// pairing structure intact and only shorten rebuildable assistant/tool bodies,
/// reasoning, and oversized tool arguments. Current high-precision results are
/// protected first; if the target is still not met, truncating those results is
/// allowed. If the untrimmable system/user content itself is already over the
/// limit, return false.
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

    // The overflow asset is the single source of truth for protected evidence. When
    // the budget is short, drop the optional preview first and keep only the
    // parseable file_path pointer; the later generic head+tail truncation must
    // never touch that minimal protocol.
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
            // Inserting the archive note the first time may shift message indices;
            // re-evaluate candidates after a successful shrink.
            blocked_fields.clear();
        }
    }

    messages_total_chars(messages) <= max_chars
}

/// Soft compaction trims only unprotected fields; the current turn's
/// high-precision tool results must be left to the real hard-target backstop and
/// must not lose freshly-read precise context just because the soft threshold is
/// small.
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

/// Path C first caps each trimmable field individually so one newest result cannot
/// monopolize the whole window, then keeps tightening against the total budget.
/// Neither step deletes messages or tool calls, nor rewrites exact-replay protocol
/// state, so assistant↔tool pairing and reasoning continuation state stay intact.
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

/// Proactively slim down the giant arguments of write_file / apply_patch calls
/// that were "successfully written".
///
/// Once the file is on disk (the result message confirms success), the full
/// content/patch body has no semantic value for later turns — the model references
/// the file path, not the body — so keeping it only occupies context. This is
/// independent of budget pressure: as soon as the group has slid out of the recent
/// protection window (the model can no longer plausibly reference the just-written
/// body to construct follow-up edits), replace it with a
/// `_context_overflow_truncated` pointer stub and archive the original with zero
/// compression. Failed results, in-window results, and current-turn protected ids
/// are never touched, so agent effectiveness does not degrade (when needed, the
/// model can still read the original back via the stub's archive_file_path, or
/// recognize the file's content outline from the preview).
fn shrink_successful_write_arguments(
    messages: &mut Vec<Message>,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) {
    if messages.is_empty() {
        return;
    }
    // Protection window: the most recent KEEP_RECENT_TOOL_GROUPS tool groups that
    // already have results (including groups just written this turn, whose bodies
    // the model may immediately reference for follow-up edits) — their calls
    // always keep full arguments.
    let protected_recent_call_ids: rustc_hash::FxHashSet<String> =
        recent_tool_result_groups(messages, KEEP_RECENT_TOOL_GROUPS)
            .into_iter()
            .flatten()
            .filter_map(|idx| messages[idx].tool_call_id.clone())
            .collect();
    // tool_call_id -> result text (used to judge success/failure).
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
        // Collect candidates first, to avoid an exclusive-borrow conflict with
        // truncate_mutable_field.
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
                continue; // already replaced; idempotent (avoids duplicate archiving/duplicate file writes)
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

/// Whether a write_file / apply_patch result succeeded. Failed results must keep
/// full arguments so the model can fix from the original text; only successful
/// results are safe to slim down.
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
    // Synthetic user messages do not form a turn boundary: otherwise precision
    // tool results from earlier turns of this round would lose protection and be
    // lossy truncated by Path C. If there is no real user at all, the whole
    // history counts as the current synthetic turn, consistent with
    // retained_turn_start's conservative boundary.
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

/// Collect every tool call in the current turn that forbids lossy compaction. It
/// is wider than the precision inline set: aggregated results like `task_wait`
/// are not part of the precision quota, but their bodies likewise must not be
/// truncated by Path C.
fn current_turn_lossless_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // When there is no real user, lossless-mandatory results of the synthetic
    // turn must not be exposed to Path C.
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

/// Mid-turn progressive compaction: reuse the first tiers of the cross-turn
/// compaction pipeline inside the iteration loop. Only "lossless/weakly-lossy"
/// operations; system messages untouched and the most recent keep_recent tool
/// messages never deleted:
///   1. dedup_repeated_tool_results — older results with the same (tool, args)
///      folded into stubs
///   2. prepare_tool_messages_structured — remote tool results trimmed by line to
///      480 chars
///   3. fold_tool_call_group_to_stub  — still over budget: fold the whole remote
///      (assistant + tool) group
/// Returns: (messages_after, before_chars, after_chars)
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
    // Hand the compaction state to the model explicitly, so it does not misread
    // recoverable evidence stubs as a full context. Inserted before any trimming;
    // the later budget calculation folds this fixed overhead in.
    upsert_context_compaction_state(&mut out);
    // 0. Clean up stale reasoning_content: multiple reasoning chains returned by
    //    the LLM within one turn add nothing to later decisions, but some vendors
    //    require historical reasoning to pair with tool_calls. Keep only the last
    //    assistant's reasoning_content; set the rest to None.
    keep_only_recent_reasoning_content(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 1. Dedupe tool results with the same signature
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&out);
    dedup_repeated_tool_results(&mut out, &protected_tool_call_ids);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. Structured remote trimming: the middle of each tool result is folded by
    //    line down to 480 chars per entry; the most recent 6 keep full text.
    //    When overflow_dir is passed, large outputs of "non-compressible" tools
    //    like read_file/grep spill to the session file with zero compression and
    //    leave a head+tail preview stub (consistent with cross-turn compaction),
    //    freeing context without losing information — the model can re-read via
    //    the stub's file_path.
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
    // 3. Still over budget: use shrink_messages_to_fit for "fold tool groups +
    //    overall backstop"
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

/// Minimum net decrease (chars) for an LLM summary to count as "effective
/// compaction". Below this it is considered ineffective and `was_effective`
/// returns false; the hard-budget backstop may still return a slightly smaller
/// context result. Same order of magnitude as `summary_max_chars`: if the net
/// decrease is smaller than the injected summary text itself, the compressor is
/// spinning (typical symptom: "295K shrank to 294K and stopped").
const MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS: usize = 4_000;

/// Path C backstop: per-message cap for head+tail truncating a single oversized
/// non-system message inside the tail window. Triggers only when progressive
/// folding still leaves the context over `hard_target` — prefer truncation over
/// letting the model 4xx.
const PATH_C_PER_MSG_CAP: usize = 8_000;

/// Mid-turn LLM summary backstop: called when the lossless/weakly-lossy pipeline
/// still leaves the context over threshold. Three complementary paths:
///   - Path A (cross-turn summary): if conversation remains before the most recent
///     `keep_recent_turns` user turns, call the LLM summarizer to compress that
///     span into a single `internal_note` injected before the tail window; also
///     fold older tool groups inside the tail window, so "bloat concentrated in
///     the newest turn" can still shrink.
///   - Path B+C (progressive folding): start from `keep_recent=4` (equivalent to
///     the original Path B) and shrink the protection window step by step to 2→1,
///     until compaction is effective or the context drops below `hard_target`.
///     Fixes compressor spin when "all the bloat sits inside the protected tail
///     window and early history has nothing left to fold".
///   - Path C backstop (per-message truncation): when progressive folding still
///     exceeds `hard_target`, head+tail truncate a single oversized non-system
///     message in the tail window. This is the absolute last resort.
/// All leading system / internal_note messages (agent instructions, tool lists,
/// global guidance) are always kept verbatim. Returns
/// `(messages_after, before, after, was_effective, llm_summary_inserted)`;
/// `was_effective` is true only when the net decrease is >=
/// [`MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS`]. false does not mean the returned
/// messages are unchanged; the hard-budget backstop may produce a partial decrease
/// below the effective threshold. `llm_summary_inserted` says whether Path A
/// actually ran and injected `[mid-turn-summary]`: false with `after < before`
/// means the decrease came entirely from mechanical paths (fold/truncate/spill),
/// letting the upper report distinguish "LLM summary executed" from "purely
/// mechanical compaction" and avoid false reporting.
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
    // best tracks the smallest result so far; None means the original messages are
    // still in use.
    let mut best: Option<Vec<Message>> = None;
    let mut best_after = before;
    // Whether Path A actually ran and injected [mid-turn-summary] (see the return
    // doc).
    let mut llm_summary_inserted = false;

    // === Path A: cross-turn LLM summary ===
    // First compute the cut point as "keep the most recent keep_recent_turns user
    // turns". After upstream projection compaction, older user messages may already
    // have been replaced by internal_note summaries (role != "user"), leaving fewer
    // visible user boundaries in the projection than keep_recent_turns, so
    // retained_turn_start returns 0. That does not mean there is no compactable old
    // content — before the first user message there may still be
    // assistant(tool_calls)/tool records protected by protocol pairing (impossible
    // to delete one by one). In that case fall the cut point back to the first user
    // message position: the trailing user turns stay protected, the leading
    // system-like summary/archive markers are kept by preserved_system_end, and the
    // old conversation span between them can be reclaimed by the LLM summary.
    let mut split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 {
        if let Some(first_user) = messages.iter().position(|m| m.role == "user") {
            if first_user > 0 {
                split_at = first_user;
            }
        }
    }
    if split_at > 0 && split_at < messages.len() {
        // Keep the leading contiguous run of system-like messages (agent
        // instructions etc.) and summarize only the conversation span after them.
        // An early version dropped the messages[0] system prompt outright, which
        // made the model instantly lose its agent behavior instructions — observed
        // as "replies cut off abruptly / extremely short / off track after
        // compaction".
        let preserved_system_end = messages[..split_at]
            .iter()
            .position(|m| !is_system_like_role(&m.role))
            .unwrap_or(split_at);
        let earlier = &messages[preserved_system_end..split_at];
        // Extract context checkpoint markers from the to-be-summarized span: they
        // are the only index locating saved checkpoint bodies and must never be
        // swallowed by the summary. The regular persisted-compaction path already
        // does the same; this closes the gap here.
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
                // 1. Leading system / internal_notes (agent instructions etc.)
                //    kept verbatim
                out.extend_from_slice(&messages[..preserved_system_end]);
                // 2. The summary is injected as an internal_note
                //    (normalize_messages_for_request classifies it as a Summary
                //    heading and merges it into the system message)
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
                // 2b. Put the extracted context checkpoint markers back, keeping
                //     their re-readable index
                out.extend(checkpoint_markers.iter().cloned());
                // 3. Tail-window folding is planned only at first; disk writes
                //    happen uniformly after the whole Path A candidate is confirmed
                //    better than the current best.
                out.extend_from_slice(tail_plan.messages());
                let after = messages_total_chars(&out);
                // Commit the tail-window fold first and archive `earlier` only
                // after the candidate is confirmed adopted: archive appends to
                // overflow-history.md (non-idempotent), so archiving early and then
                // failing the commit would leave `earlier` on disk while the
                // context never adopted `out`, and the next compaction round would
                // archive the same messages again -> orphan accumulation. The
                // short-circuit `&&` guarantees the archive is never touched when
                // commit fails; if commit succeeds but archiving fails, `best` is
                // not updated and the context still keeps `earlier` — no data loss
                // (only an idempotently hash-named fold file remains).
                if after < best_after
                    && tail_plan.commit()
                    && archive_messages_to_overflow(earlier, Some(overflow_dir.as_path())).is_some()
                {
                    best = Some(out);
                    best_after = after;
                    llm_summary_inserted = true;
                }
                // Effective compaction meeting the target -> return directly
                if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS
                    && best_after <= hard_target
                {
                    return (best.unwrap(), before, best_after, true, true);
                }
            }
        }
    }

    // === Path B+C: progressive tool-group folding ===
    // Start from keep_recent=4 (equivalent to the original Path B) and shrink the
    // protection window step by step to 2→1 (never to 0), until compaction is
    // effective or the context drops below hard_target. Fixes the spin when "all
    // the bloat sits inside the protected tail window". Folding chains on best
    // (the Path A result or the original messages): already-folded groups became
    // stubs (internal_notes) and will not match fold_early_tool_groups again, so
    // each iteration folds only the groups the previous round kept, progressively
    // releasing the protected tail window. The window never drops to 0 (see
    // [`MIN_KEEP_RECENT_TOOL_GROUPS`]): the most recent 1 group stays verbatim,
    // and remaining excess is handled by the Path C per-message truncation
    // backstop below, avoiding stub-izing the most recent tool interaction too.
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

    // Only truly reaching hard_target allows an early return. The old logic
    // returned as soon as the net decrease exceeded 4K, skipping the hard backstop
    // below and letting "older groups already saved a lot, but the newest group
    // alone still overflows the window" keep sending over-limit requests.
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

    // === Path C backstop: budget-aware structure-preserving truncation ===
    // Keep the assistant↔tool pairing of system/user messages and the most recent
    // tool groups; compress only re-retrievable result bodies, reasoning, and
    // oversized tool arguments. Unlike the old "8K per message", this keeps
    // tightening against the total budget, so it converges even with many parallel
    // tool results. If the untrimmable system/user content itself is over budget,
    // return the smallest achievable result instead of corrupting the user's task
    // text.
    let mut result = best.unwrap_or(messages);
    // Before the Path C backstop, spill every current-turn result that forbids
    // lossy compaction with zero compression: persist the original as a
    // re-readable asset and replace it with a stub, so the immediately following
    // `emergency_cap_messages_to_fit` cannot lossy-truncate that grounding
    // evidence to 8K / ~160 chars, making the original unrecoverable.
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

/// Nominal billing cost of a single image in the "char budget".
///
/// A vision model tokenizes one image into a few hundred to one-or-two thousand
/// tokens, fully decoupled from its base64 text length (easily hundreds of
/// thousands of chars). Historically `value_len_chars` billed directly by base64
/// text length, so **one large image ate the entire context budget**:
/// `messages_total_chars` ballooned far past max_chars / soft_threshold, and the
/// compaction pipeline evicted the agent's own tool results (its working memory)
/// every round — within one turn this showed up as "amnesia + repeatedly
/// restating earlier exploration/plans". Give images a fixed nominal cost so the
/// budget returns to being text-dominated.
/// Note: this only changes budget **accounting**, not the message content itself
/// (images are still sent verbatim with zero compression).
const IMAGE_BUDGET_CHARS: usize = 1_024;

/// Whether a bare string is an inline image data URL (a few providers put images
/// into plain strings).
fn is_inline_image_data_url(s: &str) -> bool {
    let t = s.trim_start();
    t.starts_with("data:image/") && t.contains(";base64,")
}

/// Budget char count of a single part in a multimodal content array: images are
/// billed at nominal cost, text at its actual char count.
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

/// "Budget char count" of a Value's content (Unicode scalar count).
/// Historically this returned byte length, overestimating the char budget ~3x for
/// Chinese/emoji content: a 36K-char soft threshold could be falsely triggered by
/// a 12K-char Chinese turn, re-running the compaction pipeline over and over. Now
/// measured uniformly by `chars().count()`, consistent with the naming of the
/// outer `cap_chars` / `max_chars` thresholds. Image parts are billed nominally
/// via [`IMAGE_BUDGET_CHARS`], keeping base64 text length from polluting the
/// budget (see that constant's doc).
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

/// "Billed char count" of a single message entering a model request — the single
/// authoritative measure.
///
/// Historically many budget checks counted only `content`, entirely missing
/// `tool_calls[].function.arguments` (typically `apply_patch` puts a whole large
/// patch into arguments with empty content) and `reasoning_content` (long
/// thinking chains), letting large messages bypass compaction gating and making
/// TPM preflight and the max_tokens clamp underestimate input together.
///
/// This combines all three into one measure, aligned with the SQL side
/// `total_message_chars_sqlite`
/// (`length(content)+length(tool_calls)+length(reasoning_content)`), so the
/// "in-memory budget" and "persisted budget" share one measure. Images are still
/// billed nominally via [`IMAGE_BUDGET_CHARS`] (see `value_len_chars`).
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
    // Multimodal messages (JSON arrays): extract only the text parts and discard
    // image base64 data, avoiding feeding huge base64 content to the model or
    // showing it to the user when generating summaries/titles.
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

/// For assistant messages carrying tool_calls, how many recent turns keep full
/// reasoning_content. Older tool-call reasoning is set to None (DeepSeek fills an
/// empty-string placeholder via echo as a backstop), preventing historical
/// reasoning text from accumulating monotonically over long sessions, slowing
/// responses and squeezing the context budget.
const KEEP_RECENT_TOOL_CALL_REASONING: usize = 3;

fn tool_message_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "tool").then_some(i))
        .collect()
}

/// Whether message content contains a real image attachment (OpenAI Vision
/// schema). The image must exist as a multimodal `Value::Array`, and the array
/// must contain
/// `{"type":"image_url", "image_url":{...}}`。
/// The old implementation misjudged via `text.contains("data:image/")`: the agent
/// merely discussing the `data:image/png` string in plain text got the whole
/// message replaced, losing information.
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
    // Images are required to stay zero-compression: the history compaction stage
    // no longer replaces old images with [[image omitted]].
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
    // tool_call_id of the previous tool message: only the same tool_call_id counts
    // as a duplicate result of the same call.
    for m in messages.drain(..) {
        let text = value_to_string(&m.content);
        // Exact-equality dedup applies only to tool messages: user/assistant/system
        // originals are never deduped. Must share the same tool_call_id: parallel
        // tool calls returning identical text are different calls and must not be
        // dropped, otherwise the assistant tool_call <-> tool result pairing
        // breaks.
        if m.role == "tool"
            && m.role == prev_role
            && text == prev_content
            && m.tool_call_id.is_some()
            && m.tool_call_id == prev_tool_call_id
        {
            continue;
        }
        // Fuzzy dedup: enabled only for the tool role, avoiding false hits on
        // assistant/user replies that look similar but differ in substance. Drop
        // only when the role matches and the whole text's tool_line_signature is
        // identical (whitespace noise stripped + key tokens equal).
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

/// Trim reasoning_content in the history, keeping only what truly needs to be
/// sent back to the vendor.
///
/// Older reasoning chains barely help later turn decisions; dropping them saves
/// context budget. Some models constrain tool-call reasoning replay, so the policy
/// here is:
/// - continuation state the model explicitly declared as exact replay carries the
///   internal marker and is always kept as long as its assistant/tool protocol
///   group is still in context; once the whole group is replaced by a summary, no
///   replay is needed;
/// - other assistant messages with `tool_calls` keep full reasoning_content only
///   for the most recent `KEEP_RECENT_TOOL_CALL_REASONING` turns, older ones set
///   to None; missing fields DeepSeek requires are backfilled with empty strings
///   by the request layer, avoiding historical reasoning text accumulating
///   monotonically over long sessions, slowing responses and "getting dumber";
/// - plain-answer assistant messages without tool_calls: keep only the most
///   recent one's reasoning_content, the rest set to None (OpenAI et al. only
///   require reasoning paired with the most recent tool_call in the same turn;
///   old plain-answer reasoning can be dropped safely).
fn keep_only_recent_reasoning_content(messages: &mut [Message]) {
    // Index of the most recent assistant reasoning "without tool_calls" — this one
    // is kept.
    let keep_plain_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| {
            m.role == "assistant" && m.reasoning_content.is_some() && m.tool_calls.is_none()
        })
        .map(|(idx, _)| idx);

    // Cross-turn sliding window for unmarked tool-call assistant reasoning: keep
    // only the most recent N full texts.
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
        // exact replay is the protocol state of its tool-call message; it cannot
        // be trimmed alone while the message is still present.
        if m.reasoning_content
            .as_deref()
            .is_some_and(|reasoning| reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX))
        {
            continue;
        }
        // Turns with tool_calls: keep only the most recent N full reasonings, the
        // rest set to None.
        if m.tool_calls.is_some() {
            let rank = seen_tool_call_reasoning;
            seen_tool_call_reasoning += 1;
            if rank < drop_tool_call_reasoning_before {
                m.reasoning_content = None;
            }
            continue;
        }
        // Plain-answer turns: keep only the most recent one.
        if Some(idx) == keep_plain_idx {
            continue;
        }
        m.reasoning_content = None;
    }
}

/// Cross-turn tool result dedup: when the same (tool_name, normalized_args)
/// appears multiple times in the history, earlier tool results are replaced with
/// a single-line stub (tool_call_id kept to preserve OpenAI tool-calls protocol
/// correctness). Only content is compressed, no messages deleted, avoiding a
/// broken pairing between assistant tool_calls and tool responses. The most
/// recent KEEP_RECENT_TOOL_GROUPS complete tool groups are always kept in full.
fn dedup_repeated_tool_results(
    messages: &mut [Message],
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) {
    use rustc_hash::{FxHashMap, FxHasher};
    use std::hash::{Hash, Hasher};

    // Collect (tool_name, args_signature) -> occurrence counts and indices
    // Map tool_call_id -> (name, args) via assistant.tool_calls
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

    // `read_file` calls with different offset/limit do not hit call-signature
    // dedup, but they may contain the same file span. Only when both results have
    // left the near-end protection window and the overlapping lines of the same
    // file are verbatim identical, delete the overlapping lines from the earlier
    // result; if any line differs (file was edited, output format changed, etc.),
    // keep both as-is.
    dedup_overlapping_read_file_results(
        messages,
        &id_to_signature,
        &protected_indices,
        protected_tool_call_ids,
    );

    // (name, args) -> the "newest full-text-kept" tool call under that signature,
    // used as the fold back-reference.
    let mut seen: FxHashMap<(String, String), DedupToolOccurrence> = FxHashMap::default();
    // (tool_name, content_hash) -> the tool call where that content version most
    // recently appeared.
    // Content-level dedup is the key to breaking the "re-reading the same whole
    // file" amnesia loop: for non-compressible tools like read_file, repeated
    // reads of the same (file) often return **byte-identical** full text (measured
    // at ~52% of all tool bytes). Such redundant copies can be folded losslessly,
    // while versions whose content truly changed (e.g. an edited file) survive
    // intact because their hash differs.
    //
    // **Key point**: the key does not carry `args_norm` — historically args were
    // part of the key, so explicit "case/path variants of the same query"
    // (`readFileLines` vs `read_file_lines`, case-sensitivity differences, etc.)
    // would not collapse even when returning **byte-identical** "no hit" bodies,
    // piling up 6+ copies of 15KB identical content at the tail (see the e75fc2e5
    // session dump). Switched to `(tool_name, content_hash)`: fold whenever the
    // returned body itself matches — different args are handled separately by the
    // call-signature dedup's `seen` counter and do not affect content-level
    // folding.
    let mut seen_content: FxHashMap<(String, u64), DedupToolOccurrence> = FxHashMap::default();
    // Scan from newest to oldest so the newest call keeps full text and only older
    // duplicates get folded. Especially critical for retry-after-failure: a
    // successful retry must not be squashed into a stub after an old failure took
    // the canonical slot.
    for &idx in tool_indices.iter().rev() {
        if messages[idx]
            .tool_call_id
            .as_ref()
            .is_some_and(|id| ambiguous_ids.contains(id))
        {
            // IDs reused in old history cannot be reliably linked to a specific
            // assistant occurrence; keep the original text.
            continue;
        }
        let occurrence = dedup_tool_occurrence(messages, idx, &id_to_signature, &id_to_args_raw);
        let occurrence = match occurrence {
            Some(occurrence) => occurrence,
            None => {
                // Orphan tool: no matching assistant.tool_calls found (the
                // assistant message may have been trimmed/dropped early, or the
                // pairing was already broken when written to history). These
                // messages get dropped at normalize_messages_for_request time but
                // still consume char budget during compaction. Results of the most
                // recent complete tool groups keep full text to avoid collateral
                // damage; older orphans are always folded into short stubs so they
                // do not block later compaction decisions.
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
        // **No longer exempt duplicates inside the recent protection window**.
        // Historically `if protected_indices.contains(&idx) continue;` here let the
        // most recent N tool groups skip dedup entirely, so the agent kept
        // re-sending the same query, the newest copy always landed in the "recent
        // window" and was never folded -> 15KB x 29 byte-identical results piled up
        // at the tail. Now dedup runs uniformly over all tool messages: the first
        // seen in reverse order (i.e. the newest) is registered as the canonical
        // full text, and all earlier copies are folded into back-reference stubs.
        // The model still sees the newest full text, while an old failure can no
        // longer override a later successful retry's valid result.
        // Orphan protection (the `!protected_indices.contains` above) is handled
        // separately and is unaffected here.
        // Content-level dedup also applies to current-turn precision-protected
        // calls (re-reads within this turn): byte-identical re-reads of the same
        // file within one turn are pure redundancy — fold the earlier copies and
        // keep the reverse-order-first (newest) full text. This does not violate
        // the "precision results stay raw" invariant — the newest copy is still the
        // raw full text and older copies merely back-reference it; it also directly
        // cuts the "same-turn full re-read pile-up -> near-end offload -> amnesia
        // and re-read" loop.
        if tool_uses_content_identity_dedup(&occurrence.tool_name) {
            // For read_file/retrieval-style tools, **versions with different
            // content** must be kept with zero compression (invariant: precision
            // results get no lossy trimming). But **byte-identical** duplicates are
            // pure redundancy; folding them loses nothing and directly removes the
            // amnesia loop of "old full texts pile up + near-end offload triggers
            // re-reads". Distinguish the two by content hash: first sighting of a
            // hash -> keep full text and register it; hash reappears -> fold into a
            // stub back-referencing the newest full text (tool_call_id kept to
            // preserve the protocol).
            let text = value_to_string(&messages[idx].content);
            // If the content is already an overflow/truncation archive stub, it is
            // not a "complete result": the canonical (reverse-order-first) copy is
            // byte-identical to this one, so the canonical is also a truncation
            // stub. Folding into "reuse the canonical full result" here would be a
            // false claim — real case: a task_wait result was first
            // overflow-truncated into [context-overflow-truncated], then dedup
            // claimed the canonical full text was reusable; the model chased the
            // canonical repeatedly but never got the original (the next hop was
            // still a stub back-reference). Skip folding: each stub carries its own
            // file_path recall pointer, and keeping them lets the model read the
            // archived original directly.
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
        // Signature-level dedup still skips current-turn precision-protected
        // calls: args variants carry information themselves (different
        // offset/limit/use_line_numbers must not be folded), avoiding collateral
        // damage to reads in use this turn. The content-level dedup above already
        // handled the "truly byte-identical" cases.
        if protected_tool_call_ids.contains(&occurrence.tool_call_id) {
            continue;
        }
        // Reverse-order first sighting is the newest call; fold older
        // same-signature results into stubs.
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

/// Whether the content is an "already-spilled/truncated archive stub" — i.e. not
/// a complete result at all, just a recall pointer to the on-disk original
/// (`[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]` or `[context-overflow-truncated]`).
/// byte-identical dedup must skip folding for such content: canonical and copy
/// are byte-identical ⇒ the canonical is likewise a truncation stub, and claiming
/// "reuse the canonical full result" would lead the model into a back-reference
/// chain that never yields the original.
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

        // Near-end complete tool groups must be kept verbatim, so the model does
        // not see processed just-read content in the next round.
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

/// Returns all line numbers shared by both sides, provided every shared line's
/// content is exactly identical.
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
mod drop_trim_differential_tests;
#[cfg(test)]
mod fold_early_tool_groups_tests;
#[cfg(test)]
mod overflow_stub_merge_tests;
#[cfg(test)]
mod overflow_sink_dedup_tests;
#[cfg(test)]
mod shrink_successful_write_arguments_tests;
#[cfg(test)]
mod tail_window_tests;
#[cfg(test)]
mod truncate_last_real_user_message_tests;
