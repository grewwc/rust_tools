//! Overflow-archive helpers (extracted from compress/mod.rs).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::super::types::{Message, ROLE_INTERNAL_NOTE};
use super::{
    ARCHIVE_NOTE_PREFIX, INTERNAL_NOTE_OVERFLOW_DIR, MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS,
    MutableMessageField, OVERFLOW_HISTORY_FILENAME, insert_archive_note_if_missing,
    is_compressed_tool_evidence_note, message_billable_chars, value_to_string,
};

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
            // Content-addressed idempotence: reuse if the existing file is intact.
            // Legacy bare `fs::write` could leave a truncated file that still
            // satisfies `exists()` / `is_file()` but is shorter than the expected
            // content (crash mid-write). That stale partial file must not be
            // treated as valid, otherwise the stub permanently points at half
            // content and is never rebuilt (size / content-address check is
            // skipped). Size check is cheap and catches truncation (partial
            // write => smaller); same-size corruption is out of scope for this
            // fast-path and would require a full read/hash.
            match std::fs::metadata(&self.path) {
                Ok(meta) if meta.len() == self.content.as_bytes().len() as u64 => {
                    return true;
                }
                Ok(_) => {
                    // Size mismatch => likely truncated legacy file. Remove so the
                    // atomic temp+rename below can rebuild. If remove fails
                    // (e.g. permission), fall through to attempt atomic overwrite
                    // via `rename` which on Unix replaces the target atomically.
                    let _ = std::fs::remove_file(&self.path);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // TOCTOU: file was deleted between is_file() and metadata().
                    // Fall through to recreate it atomically below.
                }
                Err(_) => {
                    // Can't stat existing file (permission, etc.); assume it is
                    // usable rather than risking data loss by overwriting.
                    // Conservative fallback matches previous `return true` behavior.
                    return true;
                }
            }
            // If we reach here, the existing file was size-mismatched. Either it
            // was removed (now !is_file()) or remove failed and it still exists
            // but is stale. In both cases we must not return early; fall through
            // to the atomic write path which will `rename` over the stale file.
            // No early return.
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
pub(super) fn archive_internal_notes_deduplicated(
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

pub(super) fn insert_internal_note_archive_note_if_needed(
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

pub(super) fn archive_truncated_field_to_overflow(
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

pub(super) fn insert_overflow_archive_note_if_exists(
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
pub(super) fn trim_compressed_tool_evidence_to_inline_budget(
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

pub(super) fn build_overflow_placeholder(file_path: &str) -> String {
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
