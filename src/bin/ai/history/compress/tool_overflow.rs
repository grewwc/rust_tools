//! Tool overflow handling and persisted summary construction.
//!
//! - `prepare_tool_messages_structured`: structurally trims tool messages
//! - `build_persisted_summary_text` / `build_persisted_summary_text_with_app`: builds the
//!   persisted summary
//! - `write_preserved_tool_overflow_file` and friends: write overflow content to archive files
//! - `structured_tool_output_summary`: structured summary of tool results
//! - `is_non_compressible_tool` / `is_preserved_user_or_image_stub`: tool classification helpers

use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::files::extract_key_lines;
use crate::ai::{
    history::HistoryMessageSummarizer,
    tools::{storage::file_store::FileStore, tool_history_policy},
    types::App,
};

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
use super::tool_groups::{recent_tool_group_message_indices, recent_tool_result_groups};
use super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, IMAGE_OVERFLOW_SPILL_MIN_CHARS, KEEP_RECENT_TOOL_GROUPS,
    KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX, PRESERVED_CONTENT_STUB_PREFIX,
    PRESERVED_IMAGE_OVERFLOW_DIR, PRESERVED_TOOL_OVERFLOW_DIR, PRESERVED_USER_OVERFLOW_DIR,
    PlannedArchiveWrite, USER_OVERFLOW_SPILL_MIN_CHARS, automatic_summary_body, content_sha256_hex,
    dedup_adjacent, keep_recent_user_turns_when_trimming, message_contains_image,
    normalize_whitespace, redact_images_except_last, strip_nested_prior_summary_prefixes,
    tool_message_indices, value_to_string,
};

const PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str = "[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]";
const LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str =
    "Output preserved for non-compressible tool `";

pub(super) async fn build_persisted_summary_text_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> String {
    let mut prepared = messages.to_vec();
    prepare_tool_messages_structured(
        &mut prepared,
        360,
        KEEP_RECENT_TOOL_GROUPS,
        None,
        None,
        &FxHashSet::default(),
    );
    redact_images_except_last(&mut prepared, 0);
    dedup_adjacent(&mut prepared);
    normalize_internal_notes_for_summary_model(&mut prepared);

    if let Some(summary) = app.summarize_history_messages(&prepared, max_chars).await {
        let summary = normalize_whitespace(&summary);
        if !summary.is_empty() {
            return summary;
        }
    }

    build_persisted_summary_text(messages, max_chars)
}

pub(super) fn normalize_internal_notes_for_summary_model(messages: &mut Vec<Message>) {
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
                        "Existing history summary (for this compression to absorb; do not copy verbatim):\n{}",
                        summarize_text(&body, 2_000)
                    ));
                    out.push(message);
                    seen_auto_summary = true;
                }
                continue;
            }

            if text.trim_start().contains(COMPRESSED_TOOL_EVIDENCE_MARKER) {
                out.push(message);
                continue;
            }

            // Ordinary internal_notes are mostly procedural notices, cache/loop
            // state, or inline copies of self_notes. They must not be treated as
            // long-term historical facts for the summary model to absorb
            // repeatedly.
            continue;
        }
        out.push(message);
    }

    *messages = out;
}

pub(super) fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let indices = tool_message_indices(messages);
    let protected_indices = recent_tool_group_message_indices(messages, keep_recent_groups);
    for &idx in &indices {
        let message = &mut messages[idx];
        let text = value_to_string(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        // Already-spilled precision tool results are stable pointers; a stub
        // must not be spilled again as if it were the original result.
        // Otherwise every compaction round would write a new `stub -> stub`
        // file, leaking disk space and forcing the model to follow layers of
        // pointers to reach the original evidence.
        if is_preserved_tool_overflow_stub(&text) {
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
            // The most recent complete tool group is never spilled: files and
            // retrieval results read just now must remain fully visible in the
            // next request, otherwise the model sees an "evicted, please
            // re-read" stub and immediately re-issues the same read_file —
            // manifesting as endless re-reading when the session exceeds the
            // soft threshold and compaction runs every round. Only older
            // precision results outside the protected tail window are
            // zero-compression spilled to disk.
            let is_explicitly_protected = message
                .tool_call_id
                .as_ref()
                .is_some_and(|id| protected_tool_call_ids.contains(id));
            if !is_explicitly_protected
                && !protected_indices.contains(&idx)
                && text.chars().count() > max_chars_per_msg
            {
                // When reading back an asset already archived for this session,
                // reuse the existing file so "spill → read-back → spill again"
                // does not mint a new file every time (consistent with Path C's
                // reuse logic).
                let existing_asset_path = overflow_dir.and_then(|dir| {
                    message
                        .tool_call_id
                        .as_deref()
                        .and_then(|id| id_to_tool_args.get(id))
                        .and_then(|args| {
                            preserved_tool_overflow_path_in_arguments(name, args, dir, cwd)
                        })
                });
                let reused_existing = existing_asset_path.is_some();
                if let Some(path) = existing_asset_path.or_else(|| {
                    overflow_dir.and_then(|dir| {
                        write_preserved_tool_overflow_file(
                            dir,
                            message.tool_call_id.as_deref(),
                            name,
                            &text,
                        )
                    })
                }) {
                    let recall_lines = message
                        .tool_call_id
                        .as_deref()
                        .and_then(|id| id_to_tool_args.get(id))
                        .map(|args| build_tool_overflow_recall_lines(name, args))
                        .unwrap_or_default();
                    let stub =
                        build_preserved_tool_overflow_stub(&path, name, &text, &recall_lines);
                    // Anti-bloat: spilling must actually free space. Small
                    // results (e.g. a few hundred bytes of grep output) often
                    // become larger as a stub with a full preview, which only
                    // inflates usage and forces the model to read the archive
                    // back — the exact cause of "read results kept being
                    // archived as stubs". On bloat, keep the original text and
                    // delete only the file created this time (a reused asset
                    // belongs to another message's spill; deleting it would
                    // leave that message's stub dangling). Consistent with the
                    // guards in enforce_protected_precision_
                    // group_budget / spill_protected_precision_to_fit.
                    if stub.chars().count() >= text.chars().count() {
                        if !reused_existing {
                            let _ = std::fs::remove_file(&path);
                        }
                    } else {
                        message.content = Value::String(stub);
                    }
                }
            }
            continue;
        }

        if protected_indices.contains(&idx) {
            // Ordinary tool results of the most recent complete tool group keep
            // their full text, avoiding damage to near-end context.
            continue;
        }

        let summary = structured_tool_output_summary(&text, max_chars_per_msg);
        if !summary.is_empty() && summary != text {
            message.content = Value::String(summary);
        }
    }
}

/// Applies a non-bypassable physical cap to tool results in the request
/// context.
///
/// The canonical history always keeps the original text; only the request-side
/// copy is modified here. Ordinary near-end results stay raw and are spilled
/// only past the absolute cap, preventing the canonical tail after the SQLite
/// snapshot watermark from bypassing the current-turn projection and pushing
/// oversized output into the model again.
pub(super) fn cap_oversized_tool_results_for_context(
    messages: &mut [Message],
    hard_cap_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> usize {
    if hard_cap_chars == 0 {
        return 0;
    }

    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let mut capped = 0;
    for idx in tool_message_indices(messages) {
        let text = value_to_string(&messages[idx].content);
        if is_preserved_tool_overflow_stub(&text) || text.chars().nth(hard_cap_chars).is_none() {
            continue;
        }

        let tool_call_id = messages[idx].tool_call_id.as_deref();
        let tool_name = tool_call_id
            .and_then(|id| id_to_tool_name.get(id))
            .map(String::as_str)
            .unwrap_or("unknown_tool");
        let recall_lines = tool_call_id
            .and_then(|id| id_to_tool_args.get(id))
            .map(|args| build_tool_overflow_recall_lines(tool_name, args))
            .unwrap_or_default();
        // When reading back an asset already archived for this session
        // (read_file / execute_command cat pointing at a direct child of
        // tool-overflow-compressed/), reuse the existing file instead of
        // minting another randomly named one — otherwise "spill → read-back →
        // spill again" would generate a new archive on every read-back, with
        // the model forever re-reading along new pointers: an unbounded chain.
        // Consistent with Path C's reuse logic.
        let existing_asset_path = overflow_dir.and_then(|dir| {
            tool_call_id
                .and_then(|id| id_to_tool_args.get(id))
                .and_then(|args| {
                    preserved_tool_overflow_path_in_arguments(tool_name, args, dir, cwd)
                })
        });
        let replacement = existing_asset_path
            .or_else(|| {
                overflow_dir.and_then(|dir| {
                    write_preserved_tool_overflow_file(dir, tool_call_id, tool_name, &text)
                })
            })
            .map(|path| {
                build_preserved_tool_overflow_stub(&path, tool_name, &text, &recall_lines)
            })
            .unwrap_or_else(|| {
                let mut stub = format!(
                    "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
                     Output for tool `{tool_name}` exceeded the absolute request-context cap, but the session asset could not be written. Canonical history still retains the raw result."
                );
                for line in &recall_lines {
                    stub.push('\n');
                    stub.push_str(line);
                }
                stub.push('\n');
                stub.push_str(&build_overflow_content_preview(&text));
                stub
            });
        messages[idx].content = Value::String(replacement);
        capped += 1;
    }
    capped
}

/// The newest parallel batch may alone exceed the context window. Group-level
/// judgment still applies, but results registered as high-precision grounding
/// get an inline cap: results over budget are zero-compression spilled while
/// keeping a recallable stub. Aggregate results such as `task` / `task_wait`
/// do not register that flag and never crowd out the budget of evidence like
/// read_file.
pub(super) fn enforce_protected_precision_group_budget(
    messages: &mut [Message],
    keep_recent_groups: usize,
    inline_budget: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
    allow_overflow_protected: bool,
) {
    let Some(overflow_dir) = overflow_dir else {
        return;
    };
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);

    for group in recent_tool_result_groups(messages, keep_recent_groups) {
        let mut precision_results: Vec<(usize, String)> = group
            .into_iter()
            .filter_map(|idx| {
                let tool_name = messages[idx]
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_tool_name.get(id))?;
                tool_history_policy(tool_name)
                    .counts_toward_precision_inline_budget()
                    .then(|| (idx, tool_name.clone()))
            })
            .collect();

        // Results already spilled into stubs / empty text no longer count
        // toward the available budget: they are occupied by fixed protocol
        // text, and counting them would inflate total_chars and cause extra
        // spills of other results in the same group.
        let mut total_chars = precision_results
            .iter()
            .map(|(idx, _)| value_to_string(&messages[*idx].content))
            .filter(|text| !text.trim().is_empty() && !is_preserved_tool_overflow_stub(text))
            .map(|text| text.chars().count())
            .sum::<usize>();
        precision_results.sort_unstable_by_key(|(idx, _)| {
            std::cmp::Reverse(value_to_string(&messages[*idx].content).chars().count())
        });

        // Spill the largest results first, freeing enough space with the fewest
        // stubs; the rest of the group's evidence stays fully visible.
        for (idx, tool_name) in precision_results {
            if total_chars <= inline_budget {
                break;
            }
            let text = value_to_string(&messages[idx].content);
            if text.trim().is_empty() || is_preserved_tool_overflow_stub(&text) {
                continue;
            }
            // protected (current-turn) results stay as-is by default to remain
            // in context; only Path C's fallback allows a zero-compression
            // spill to an asset, so the original is recoverable instead of
            // being lost to later lossy truncation.
            if !allow_overflow_protected
                && messages[idx]
                    .tool_call_id
                    .as_ref()
                    .is_some_and(|id| protected_tool_call_ids.contains(id))
            {
                continue;
            }
            let text_len = text.chars().count();
            // When reading back an asset already archived for this session
            // (read_file pointing at a direct child of
            // tool-overflow-compressed/), reuse the existing file instead of
            // minting another randomly named one — otherwise "spill → read-back
            // → spill again" would generate a new archive on every read-back,
            // forming an unbounded chain where the model never gets stable
            // content. Consistent with Path C's reuse logic.
            let tool_call_id = messages[idx].tool_call_id.as_deref();
            let existing_asset_path = tool_call_id
                .and_then(|id| id_to_tool_args.get(id))
                .and_then(|args| {
                    preserved_tool_overflow_path_in_arguments(&tool_name, args, overflow_dir, cwd)
                });
            let (path, wrote_new) = if let Some(path) = existing_asset_path {
                (path, false)
            } else {
                let Some(path) = write_preserved_tool_overflow_file(
                    overflow_dir,
                    tool_call_id,
                    &tool_name,
                    &text,
                ) else {
                    continue;
                };
                (path, true)
            };
            let recall_lines = messages[idx]
                .tool_call_id
                .as_deref()
                .and_then(|id| id_to_tool_args.get(id))
                .map(|args| build_tool_overflow_recall_lines(&tool_name, args))
                .unwrap_or_default();
            let stub = build_preserved_tool_overflow_stub(&path, &tool_name, &text, &recall_lines);
            // Spilling must strictly free space: swapping a small result for a
            // larger stub is bloat, not compression — it inflates budget usage
            // and hides the real result from the model (an amplifying factor in
            // loops). On bloat, delete the just-written file and keep the
            // original text, consistent with Path C's anti-bloat guard.
            // Delete only files created this time: a reused asset belongs to
            // another message's spill, and deleting it would leave that
            // message's stub dangling (past sessions saw archive files deleted
            // by mistake).
            let stub_chars = stub.chars().count();
            if stub_chars >= text_len {
                if wrote_new {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
            messages[idx].content = Value::String(stub);
            // Accounting must include the stub's own size: subtracting only the
            // original underestimates remaining usage and skews budget
            // decisions.
            total_chars = total_chars
                .saturating_sub(text_len)
                .saturating_add(stub_chars);
        }
    }
}

/// Path C's global fallback: collect all protected results that forbid lossy
/// compression across tool groups and spill the largest originals first, until
/// the whole request returns to the hard target or no spillable candidates
/// remain. Candidates are widened to `!allows_lossy_compress()` (not just
/// high-precision inline-budget tools) so aggregate but non-compressible large
/// results like `task_wait` also take the lossless spill + file-pointer path
/// instead of being silently dropped by later lossy truncation.
pub(super) fn spill_protected_precision_to_fit(
    messages: &mut [Message],
    hard_target_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) -> usize {
    let Some(overflow_dir) = overflow_dir else {
        return 0;
    };
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let mut candidates = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, message)| {
            let id = message.tool_call_id.as_ref()?;
            if !protected_tool_call_ids.contains(id) {
                return None;
            }
            let tool_name = id_to_tool_name.get(id)?;
            let text = value_to_string(&message.content);
            // If the current turn reads this session's archive directly, Path C
            // must not copy the original again. Reusing the original asset's
            // pointer both preserves the hard target and avoids the "spill →
            // read-back → spill again" cycle.
            let existing_asset_path = id_to_tool_args.get(id).and_then(|args| {
                preserved_tool_overflow_path_in_arguments(&tool_name, args, overflow_dir, cwd)
            });
            (!text.trim().is_empty()
                && !is_preserved_tool_overflow_stub(&text)
                && !tool_history_policy(tool_name).allows_lossy_compress())
            .then(|| {
                (
                    idx,
                    tool_name.clone(),
                    existing_asset_path,
                    text.chars().count(),
                )
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(_, _, _, chars)| std::cmp::Reverse(*chars));

    // Running total maintained incrementally instead of re-scanning the whole
    // sequence per candidate: the only mutation inside this loop is the content
    // swap on a single message (`messages[idx].content`), so the authoritative
    // `messages_total_chars` sum changes by exactly that message's billable
    // delta. This mirrors the running-total pattern used by
    // `shrink_messages_to_fit_with_summary` in `compress/mod.rs`. Candidate
    // selection order and spilled stub contents are untouched.
    let mut total = super::messages_total_chars(messages);
    let mut spilled = 0usize;
    for (idx, tool_name, existing_asset_path, _) in candidates {
        if total <= hard_target_chars {
            break;
        }
        let text = value_to_string(&messages[idx].content);
        let (path, wrote_new_archive) = if let Some(path) = existing_asset_path {
            (path, false)
        } else {
            let Some(path) = write_preserved_tool_overflow_file(
                overflow_dir,
                messages[idx].tool_call_id.as_deref(),
                &tool_name,
                &text,
            ) else {
                continue;
            };
            (path, true)
        };
        let recall_lines = messages[idx]
            .tool_call_id
            .as_deref()
            .and_then(|id| id_to_tool_args.get(id))
            .map(|args| build_tool_overflow_recall_lines(&tool_name, args))
            .unwrap_or_default();
        let full_stub = build_preserved_tool_overflow_stub(&path, &tool_name, &text, &recall_lines);
        let replacement = if full_stub.chars().count() < text.chars().count() {
            full_stub
        } else if let Some(pointer_stub) = minimize_overflow_stub_to_pointer(&full_stub) {
            if pointer_stub.chars().count() < text.chars().count() {
                pointer_stub
            } else {
                if wrote_new_archive {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
        } else {
            if wrote_new_archive {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        };
        let before_billable = super::message_billable_chars(&messages[idx]);
        messages[idx].content = Value::String(replacement);
        let after_billable = super::message_billable_chars(&messages[idx]);
        total = total
            .saturating_sub(before_billable)
            .saturating_add(after_billable);
        // Debug-only oracle: the incremental total must always equal a full
        // rescan; compiles away in release builds.
        debug_assert_eq!(total, super::messages_total_chars(messages));
        spilled += 1;
    }
    spilled
}

pub(super) fn build_tool_call_name_index(messages: &[Message]) -> FxHashMap<String, String> {
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

pub(super) fn build_tool_call_arguments_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.arguments.clone());
        }
    }
    out
}

/// Returns the tool-overflow archive path of this session read directly by
/// `read_file`.
///
/// Ordinary tool results must be persisted to disk to preserve the hard
/// target; but an archive is already a stable asset written by the runtime, and
/// copying it again would only make the model keep producing new archives when
/// reading large files back. Only direct children of
/// `tool-overflow-compressed` are reused: files like `tmp` and checkpoints
/// under the session asset root can still be written later in the same
/// session, so keeping a pointer to them would let historical results change
/// with their source. Both ends are canonicalized before the directory check
/// to keep `..` or symlinks from crossing the boundary.
fn preserved_tool_overflow_read_file_path(arguments: &str, overflow_dir: &Path) -> Option<PathBuf> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    let raw_path = value_string_from_keys(&args, &["file_path", "path", "filePath"])?;
    archived_asset_path_from_raw(&raw_path, overflow_dir)
}

/// Uniformly recognizes direct file paths that "read this session's archived
/// asset", so both main paths and Path C can reuse the existing archive
/// instead of minting a new uuid file on every read-back and making the model
/// re-read endlessly along new pointers.
///
/// - `read_file`: the `file_path` / `path` / `filePath` fields;
/// - `execute_command`: archive paths embedded in the `command` string
///   (`cat`/`head`/`grep` etc.).
/// Other tools return no paths (reuse semantics do not apply; behavior matches
/// the old `read_file` guard).
fn preserved_tool_overflow_path_in_arguments(
    tool_name: &str,
    arguments: &str,
    overflow_dir: &Path,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    match tool_name {
        "read_file" => preserved_tool_overflow_read_file_path(arguments, overflow_dir),
        "execute_command" => {
            let command = value_string_from_keys(&args, &["command"])?;
            archived_asset_path_in_command(&command, overflow_dir, cwd)
        }
        _ => None,
    }
}

/// Validates that a raw path is a direct child of `tool-overflow-compressed`;
/// both ends are canonicalized.
fn archived_asset_path_from_raw(raw_path: &str, overflow_dir: &Path) -> Option<PathBuf> {
    let preserved_dir = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .canonicalize()
        .ok()?;
    // Must share the same relative-path resolution rules as read_file;
    // canonicalizing directly would wrongly anchor at the process cwd and
    // ignore the subagent's effective_cwd.
    let source_path = FileStore::new(PathBuf::from(raw_path))
        .path()
        .canonicalize()
        .ok()?;
    if !source_path.is_file() || source_path.parent() != Some(preserved_dir.as_path()) {
        return None;
    }
    Some(source_path)
}

/// Recognizes paths that "read this session's archived asset" inside the
/// `command` string of execute_command. Matches absolute paths of archive
/// direct children, paths relative to effective_cwd, or bare file names (the
/// name contains a uuid and is essentially unique). The archive count is
/// small, so the per-compaction scan cost is acceptable.
fn archived_asset_path_in_command(
    command: &str,
    overflow_dir: &Path,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let preserved_dir = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .canonicalize()
        .ok()?;
    for entry in std::fs::read_dir(&preserved_dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        // Consistent with the read_file path (archived_asset_path_from_raw):
        // reject symlink targets pointing outside the
        // `tool-overflow-compressed` directory.
        if !canonical.is_file() || canonical.parent() != Some(preserved_dir.as_path()) {
            continue;
        }
        let abs = canonical.to_string_lossy().into_owned();
        if command.contains(&abs) {
            return Some(canonical);
        }
        if let Some(cwd) = &cwd
            && let Ok(rel) = canonical.strip_prefix(cwd)
            && let Some(rel_str) = rel.to_str()
            && command.contains(rel_str)
        {
            return Some(canonical);
        }
        if let Some(name) = canonical.file_name().and_then(|n| n.to_str())
            && command.contains(name)
        {
            return Some(canonical);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn assistant_call(id: &str, name: &str) -> Message {
        assistant_call_args(id, name, "{}")
    }

    fn assistant_call_args(id: &str, name: &str, arguments: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
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

    #[test]
    fn preserved_tool_overflow_stub_is_not_spilled_again() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-tool-overflow-stub-{}", uuid::Uuid::new_v4()));
        let mut messages = vec![
            assistant_call("old", "read_file"),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let first_stub = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&first_stub));
        let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        assert_eq!(value_to_string(&messages[1].content), first_stub);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn pruned_stable_archive_is_content_addressed() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-pruned-stable-content-addressed-{}",
            uuid::Uuid::new_v4()
        ));

        // Same (tool, id, content) must map to one idempotent archive file.
        let first = write_preserved_tool_overflow_file_stable(
            &overflow_dir,
            "call-1",
            "read_file",
            "result body",
        )
        .unwrap();
        let second = write_preserved_tool_overflow_file_stable(
            &overflow_dir,
            "call-1",
            "read_file",
            "result body",
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "result body");

        // Same call id with different content must NOT reuse the old file:
        // a replayed tool call would otherwise read back stale bytes.
        let replayed = write_preserved_tool_overflow_file_stable(
            &overflow_dir,
            "call-1",
            "read_file",
            "replayed body",
        )
        .unwrap();
        assert_ne!(first, replayed);
        assert_eq!(std::fs::read_to_string(&replayed).unwrap(), "replayed body");
        let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 2);

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_read_file_overflow_stub_keeps_original_target_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-read-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "read_file",
                r#"{"file_path":"src/lib.rs","offset":120,"limit":40}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_file_path: src/lib.rs"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("- original_range: lines=120..159"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("Archived snapshot of an earlier read"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_file_path: src/lib.rs"),
            "anchor: {anchor}"
        );
        assert!(
            anchor.contains("Archived snapshot of an earlier read"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_stub_preview_includes_line_numbered_key_lines() {
        // read_file output carries a `{:>6}\t` line-number prefix: key_lines
        // should parse the prefix and use real line numbers for L tags (rather
        // than failing every match because of the prefix), so long files remain
        // locatable by line number after being spilled.
        let content = "\
     1\tuse std::fmt;\n\
     2\t\n\
     3\tpub fn main() {\n\
     4\t    let x = 1;\n\
     5\t}\n\
     6\tfn helper() {}\n\
     7\t//! crate docs\n\
     8\tstruct Foo;\n";
        let preview = build_overflow_content_preview(content);
        assert!(preview.contains("- key_lines (5):"), "preview: {preview}");
        assert!(preview.contains("L1: use std::fmt;"), "preview: {preview}");
        assert!(preview.contains("L3: pub fn main()"), "preview: {preview}");
        assert!(preview.contains("L6: fn helper()"), "preview: {preview}");
        assert!(preview.contains("L8: struct Foo;"), "preview: {preview}");
    }

    #[test]
    fn preserved_execute_command_overflow_stub_keeps_original_command_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-command-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "execute_command",
                r#"{"command":"git log --stat","cwd":"/repo"}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_command: git log --stat"),
            "stub: {stub}"
        );
        assert!(stub.contains("- original_cwd: /repo"), "stub: {stub}");
        assert!(
            stub.contains("Continue from `original_command` / `original_cwd`"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_command: git log --stat"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn legacy_tool_overflow_stub_is_recognized() {
        let legacy = "Output preserved for non-compressible tool `read_file`.\n\
            - file_path: /tmp/result.txt\n\
            - use read_file to inspect exact content.\n\
            Preview (for recall; not exhaustive):";
        assert!(is_preserved_tool_overflow_stub(legacy));
    }

    #[test]
    fn protected_precision_budget_excludes_aggregated_task_results() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-precision-group-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut call = assistant_call("read", "read_file");
        call.tool_calls.as_mut().unwrap().push(ToolCall {
            id: "task".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "task_wait".to_string(),
                arguments: "{}".to_string(),
            },
        });
        let mut messages = vec![
            call,
            tool_result("read", &"r".repeat(1_000)),
            tool_result("task", &"t".repeat(10_000)),
        ];

        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            200,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        assert!(is_preserved_tool_overflow_stub(&value_to_string(
            &messages[1].content
        )));
        assert_eq!(value_to_string(&messages[2].content).len(), 10_000);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn precision_budget_never_expands_small_results_into_larger_stubs() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-precision-group-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut call = assistant_call("small", "read_file");
        call.tool_calls.as_mut().unwrap().push(ToolCall {
            id: "big".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        });
        let mut messages = vec![
            call,
            tool_result("small", &"s".repeat(100)),
            tool_result("big", &"b".repeat(10_000)),
        ];

        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            200,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        // A small result (100 chars) bloats when swapped for a stub with a long
        // path: the original text must stay inline.
        assert_eq!(value_to_string(&messages[1].content), "s".repeat(100));
        // The large result is spilled into a stub, and the stub is strictly
        // shorter than the original.
        let stub = value_to_string(&messages[2].content);
        assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
        assert!(stub.chars().count() < 10_000, "{stub}");
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn enforce_group_budget_reuses_reread_archive_asset_instead_of_rearchiving() {
        // When a read_file result that read back an archived asset (no longer
        // protected after crossing turns) enters group spilling again, the
        // existing archive file must be reused (the stub points at the same
        // file) instead of minting a randomly named new one — otherwise "spill
        // → read-back → spill again" generates a new archive on every
        // read-back, forming an unbounded chain where the model never gets
        // stable content.
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-precision-group-reuse-{}", uuid::Uuid::new_v4()));
        // 1) Generate an archive asset via Path C
        let mut messages = vec![
            assistant_call("spill", "read_file"),
            // Leave ample headroom in the payload: the fingerprint adds a fixed
            // ~16-byte overhead to the stub, and if the original and stub sizes
            // were extremely close the anti-bloat guard (stub>=original) could
            // flip, making this test about reuse semantics rather than byte
            // coincidence.
            tool_result("spill", &"x".repeat(4_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];
        let mut protected = FxHashSet::default();
        protected.insert("spill".to_string());
        let stub1 = spill_protected_precision_to_fit(
            &mut messages,
            80,
            Some(&overflow_dir),
            None,
            &protected,
        );
        assert!(stub1 > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) The model reads the archive back: the result (1000 chars) exceeds
        // the group inline budget → triggers the enforce spill
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            120,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
            false,
        );

        // 3) The existing asset is reused: the directory still has exactly 1
        // file, and the stub points at the same archive_path
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn cap_oversized_reuses_reread_archive_asset_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-cap-reread-reuse-{}", uuid::Uuid::new_v4()));
        // 1) First let the cap itself write an archive asset
        let mut messages = vec![
            assistant_call("first", "read_file"),
            tool_result("first", &"y".repeat(70_000)),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert!(capped > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) The model reads the archive back (body 70k > the 64k hard cap) →
        // the existing file is reused, no new file written
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert_eq!(capped, 1);
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_reuses_reread_archive_asset_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-prepare-reread-reuse-{}", uuid::Uuid::new_v4()));
        // 1) First let prepare write an archive asset (an old read_file result,
        // over the 480 threshold and outside the tail window)
        let mut messages = vec![
            assistant_call("first", "read_file"),
            tool_result("first", &"z".repeat(2_000)),
        ];
        prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) The model reads the archive back and the result enters prepare
        // again (unprotected, outside the tail window) → the existing file is
        // reused
        let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
        let mut messages = vec![
            assistant_call_args("re-read", "read_file", &read_args.to_string()),
            tool_result("re-read", &raw),
        ];
        prepare_tool_messages_structured(
            &mut messages,
            480,
            0,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_spill_is_deterministic_across_reprojections() {
        // P1 regression: when the same canonical tool result is spilled in two
        // independent projections, it must map to the same deterministic
        // archive file rather than minting a randomly named new copy every
        // round (the old behavior caused unbounded bloat within one session:
        // 368 files for only 211 unique contents). Using **different**
        // overflow_dirs for the two projections would hide idempotence, so the
        // same dir is reused to simulate round-by-round compaction of one
        // session.
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-prepare-deterministic-spill-{}",
            uuid::Uuid::new_v4()
        ));
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        // A single very long line: after line truncation the preview makes the
        // stub significantly smaller than the original → it does spill (the
        // anti-bloat guard does not trigger).
        let big = "b".repeat(4_000);
        let build = || {
            vec![
                assistant_call("spill", "read_file"),
                tool_result("spill", &big),
                assistant_call("recent", "read_file"),
                tool_result("recent", "recent result"),
            ]
        };

        let mut first = build();
        prepare_tool_messages_structured(
            &mut first,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let first_stub = value_to_string(&first[1].content);
        assert!(is_preserved_tool_overflow_stub(&first_stub), "{first_stub}");
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);

        // Second projection: the same canonical result (same tool_call_id +
        // same body) is compacted again. Deterministic naming → the existing
        // file is hit, no new copy is added, and the stub text stays stable
        // across rounds.
        let mut second = build();
        prepare_tool_messages_structured(
            &mut second,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );
        let second_stub = value_to_string(&second[1].content);
        assert_eq!(
            second_stub, first_stub,
            "重投影后 stub 文本必须稳定（prompt cache 不断裂）"
        );
        assert_eq!(
            std::fs::read_dir(&archive_dir).unwrap().count(),
            1,
            "同一结果重复外溢不得铸造新归档文件"
        );

        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn prepare_structured_keeps_small_multiline_result_inline() {
        // P2 regression: a few-hundred-byte multi-line grep result (over
        // max_chars_per_msg but still small) becomes larger when swapped for a
        // stub with a full head/tail preview. The anti-bloat guard must keep
        // the original inline and write no archive file — otherwise the model
        // sees "evicted, please re-read" and reads back repeatedly (session
        // 9f4d0fae's "read results kept being archived as stubs" was exactly
        // this path: a 673-char grep swapped for a larger stub).
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-prepare-small-inline-{}", uuid::Uuid::new_v4()));
        // 20 lines × ~30 chars ≈ 600 chars: over max_chars_per_msg=480, but the
        // preview contains the whole body verbatim, so the stub cannot be
        // smaller than the original.
        let grep_like = (0..20)
            .map(|i| format!("src/bin/ai/mod.rs:{i}: use crate::ai::x;"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(grep_like.chars().count() > 480);
        let mut messages = vec![
            assistant_call("grep", "execute_command"),
            tool_result("grep", &grep_like),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            480,
            1,
            Some(&overflow_dir),
            None,
            &FxHashSet::default(),
        );

        let content = value_to_string(&messages[1].content);
        assert_eq!(content, grep_like, "小的多行精确结果必须保留原文内联");
        assert!(
            !is_preserved_tool_overflow_stub(&content),
            "不应被换成 stub"
        );
        // No archive file is written (on bloat the new file is deleted; here it
        // should never have been written at all).
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archived = std::fs::read_dir(&archive_dir)
            .map(|d| d.count())
            .unwrap_or(0);
        assert_eq!(archived, 0, "膨胀结果不得留下归档文件");

        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn cap_reuses_execute_command_cat_archive_instead_of_rearchiving() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-cap-cat-reuse-{}", uuid::Uuid::new_v4()));
        // 1) First let the cap write an execute_command archive asset
        let mut messages = vec![
            assistant_call("run", "execute_command"),
            tool_result("run", &"log line\n".repeat(30_000)),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert!(capped > 0);
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        let archive_path = std::fs::read_dir(&archive_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let raw = std::fs::read_to_string(&archive_path).unwrap();

        // 2) The model reads the archive body back with `cat <archive>` (over
        // the hard cap) → the existing file is reused
        let run_args = serde_json::json!({
            "command": format!("cat {}", archive_path.to_string_lossy()),
            "pty": false,
        });
        let mut messages = vec![
            assistant_call_args("re-cat", "execute_command", &run_args.to_string()),
            tool_result("re-cat", &raw),
        ];
        let capped = cap_oversized_tool_results_for_context(
            &mut messages,
            64_000,
            Some(&overflow_dir),
            None,
        );
        assert_eq!(capped, 1);
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        let stub_text = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
        assert!(
            stub_text.contains(archive_path.to_str().unwrap()),
            "{stub_text}"
        );
        let _ = std::fs::remove_dir_all(&overflow_dir);
    }

    #[test]
    fn path_c_spills_all_protected_precision_groups_without_recent_group_cap() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = Vec::new();
        let mut protected = FxHashSet::default();
        for index in 0..8 {
            let id = format!("read-{index}");
            protected.insert(id.clone());
            messages.push(assistant_call(&id, "read_file"));
            messages.push(tool_result(&id, &"line of exact evidence\n".repeat(600)));
        }

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        // Covers the second half of Path C: when still over budget after the
        // spill, the emergency cap kicks in. Every preserved stub must first
        // shrink into a non-truncatable minimal pointer and must not be run
        // through generic head/tail truncation again.
        assert!(super::super::messages_total_chars(&messages) > 4_000);
        super::super::emergency_cap_messages_to_fit(
            &mut messages,
            4_000,
            160,
            Some(&overflow_dir),
            &protected,
        );

        assert_eq!(spilled, 8);
        let stubs = messages
            .iter()
            .filter_map(|message| {
                let content = value_to_string(&message.content);
                is_preserved_tool_overflow_stub(&content).then_some(content)
            })
            .collect::<Vec<_>>();
        assert_eq!(stubs.len(), 8);
        for stub in stubs {
            let file_path = stub
                .lines()
                .find_map(|line| line.strip_prefix("- file_path: "))
                .expect("minimal overflow stub must retain file_path");
            assert!(Path::new(file_path).is_file());
            assert!(!stub.contains("Preview ("));
        }
        assert!(super::super::messages_total_chars(&messages) <= 4_000);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn path_c_reuses_reread_session_asset_instead_of_rearchiving_it() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let test_root =
            std::env::temp_dir().join(format!("ai-reread-session-asset-{}", uuid::Uuid::new_v4()));
        let effective_cwd = test_root.join("workspace");
        let overflow_dir = effective_cwd.join("session-assets");
        let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        std::fs::create_dir_all(&archive_dir).unwrap();
        let archive_path = archive_dir.join("prior-read.txt");
        let content = "previously preserved evidence\n".repeat(800);
        std::fs::write(&archive_path, &content).unwrap();
        let relative_archive_path = archive_path
            .strip_prefix(&effective_cwd)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let arguments = serde_json::json!({
            "file_path": relative_archive_path,
            "offset": 1,
            "limit": 10_000,
        })
        .to_string();
        let mut protected = FxHashSet::default();
        protected.insert("reread".to_string());
        let mut messages = vec![
            assistant_call_args("reread", "read_file", &arguments),
            tool_result("reread", &content),
        ];

        let spilled =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
                spill_protected_precision_to_fit(
                    &mut messages,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected,
                )
            });

        assert_eq!(spilled, 1);
        let stub = value_to_string(&messages[1].content);
        let file_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .expect("reused stub must retain the existing archive pointer");
        assert_eq!(Path::new(file_path), archive_path.canonicalize().unwrap());
        assert!(stub.contains("- original_range: lines=1..10000"));
        assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
        assert_eq!(std::fs::read_to_string(&archive_path).unwrap(), content);
        let _ = std::fs::remove_dir_all(test_root);
    }

    #[test]
    fn path_c_snapshots_mutable_session_temp_asset_instead_of_reusing_it() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let test_root =
            std::env::temp_dir().join(format!("ai-reread-session-temp-{}", uuid::Uuid::new_v4()));
        let effective_cwd = test_root.join("workspace");
        let overflow_dir = effective_cwd.join("session-assets");
        let temp_dir = overflow_dir.join("tmp");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let temp_path = temp_dir.join("mutable.txt");
        let content = "temporary evidence before mutation\n".repeat(800);
        std::fs::write(&temp_path, &content).unwrap();
        let relative_temp_path = temp_path
            .strip_prefix(&effective_cwd)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let arguments = serde_json::json!({
            "file_path": relative_temp_path,
            "offset": 1,
            "limit": 10_000,
        })
        .to_string();
        let mut protected = FxHashSet::default();
        protected.insert("reread".to_string());
        let mut messages = vec![
            assistant_call_args("reread", "read_file", &arguments),
            tool_result("reread", &content),
        ];

        let spilled =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
                spill_protected_precision_to_fit(
                    &mut messages,
                    0,
                    Some(&overflow_dir),
                    None,
                    &protected,
                )
            });

        assert_eq!(spilled, 1);
        let stub = value_to_string(&messages[1].content);
        let snapshot_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .map(PathBuf::from)
            .expect("mutable session file must be snapshotted into an overflow archive");
        assert_ne!(snapshot_path, temp_path.canonicalize().unwrap());
        assert!(snapshot_path.starts_with(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR)));
        assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);

        std::fs::write(&temp_path, "temporary evidence after mutation\n").unwrap();
        assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);
        let _ = std::fs::remove_dir_all(test_root);
    }

    #[test]
    fn path_c_spills_aggregated_task_wait_result_losslessly() {
        // task_wait forbids lossy compression but occupies no inline budget;
        // Path C's global fallback must spill it losslessly with a file pointer
        // left behind, rather than excluding it from candidates and letting
        // later lossy truncation lose the aggregate truth.
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-taskwait-{}",
            uuid::Uuid::new_v4()
        ));
        let mut protected = FxHashSet::default();
        protected.insert("wait".to_string());
        let mut messages = vec![
            assistant_call("wait", "task_wait"),
            tool_result("wait", &"aggregated subagent conclusion\n".repeat(600)),
        ];

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        assert_eq!(spilled, 1, "task_wait 大结果应被 Path C 无损外溢");
        let stub = value_to_string(&messages[1].content);
        assert!(
            is_preserved_tool_overflow_stub(&stub),
            "外溢后应替换为 overflow stub"
        );
        let file_path = stub
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
            .expect("overflow stub 必须保留可召回的 file_path 指针");
        assert!(Path::new(file_path.trim()).is_file(), "外溢原文必须落盘");
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn path_c_does_not_expand_short_protected_results_into_stubs() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-global-precision-short-{}",
            uuid::Uuid::new_v4()
        ));
        let mut protected = FxHashSet::default();
        protected.insert("read-short".to_string());
        let mut messages = vec![
            assistant_call("read-short", "read_file"),
            tool_result("read-short", "ok"),
        ];
        let before = super::super::messages_total_chars(&messages);

        let spilled = spill_protected_precision_to_fit(
            &mut messages,
            0,
            Some(&overflow_dir),
            None,
            &protected,
        );

        assert_eq!(spilled, 0);
        assert_eq!(value_to_string(&messages[1].content), "ok");
        assert_eq!(super::super::messages_total_chars(&messages), before);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// Builds a stub in its "first spill" shape (with a multi-line Preview
    /// body), for fold testing.
    fn overflow_stub_with_preview(file_path: &str, tool_name: &str) -> String {
        let full = (0..40)
            .map(|i| format!("line {i}: some content"))
            .collect::<Vec<_>>()
            .join("\n");
        build_preserved_tool_overflow_stub(Path::new(file_path), tool_name, &full, &[])
    }

    #[test]
    fn collapse_overflow_stub_to_anchor_drops_preview_keeps_file_path() {
        let stub = overflow_stub_with_preview("/tmp/session/read-abc.txt", "read_file");
        // Precondition: the first-spill stub really does carry a Preview body.
        assert!(stub.contains("Preview ("));

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("should collapse");
        // The preview body is discarded.
        assert!(!anchor.contains("Preview ("));
        // The file_path is kept.
        assert!(anchor.contains("- file_path: /tmp/session/read-abc.txt"));
        // The tool name is kept (the new format uses "Output preserved for
        // tool").
        assert!(anchor.contains("Output preserved for tool `read_file`"));
        // read_file-type archives carry the "usually no need to re-read" notice
        // (instead of the old leading "use read_file").
        assert!(anchor.contains("Archived snapshot of an earlier read"));
        // Still a valid stub (prefix unchanged); the downstream compaction
        // chain keeps recognizing it via the stub exemption.
        assert!(is_preserved_tool_overflow_stub(&anchor));
        // The size drops sharply.
        assert!(anchor.len() < stub.len());
    }

    #[test]
    fn preserved_stub_carries_fingerprint_line() {
        let full = "Compiling rust_tools v0.1.0 (/repo)\n\
                    warning: unused variable `root_idx`\n\
                    error[E0308]: mismatched types in sched_ctx\n";
        let stub = build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
        assert!(is_preserved_tool_overflow_stub(&stub));

        // Deterministic in content: same bytes -> byte-identical stub text.
        let stub_again = build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
        assert_eq!(stub, stub_again);

        let fp_line = stub
            .lines()
            .find_map(|l| l.trim_start().strip_prefix("- fingerprint: "))
            .expect("fingerprint line present on fresh stub");
        // sha= segment: exactly 12 hex chars.
        let sha = fp_line.split("sha=").nth(1).unwrap().split(' ').next().unwrap();
        assert_eq!(sha.len(), 12);
        assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
        // Keyword casing is preserved verbatim so tokens stay greppable in the archive.
        assert!(fp_line.contains("keys="), "keys= segment present: {fp_line}");
        assert!(fp_line.contains("rust_tools") || fp_line.contains("root_idx"), "keywords: {fp_line}");
    }

    #[test]
    fn collapse_and_minimize_carry_fingerprint_through() {
        let full = "alpha beta gamma\nE0308 mismatched_types hit\n".repeat(30);
        let stub = build_preserved_tool_overflow_stub(Path::new("/tmp/carry.txt"), "execute_command", full.as_str(), &[]);

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("collapse");
        assert!(!anchor.contains("Preview ("));
        assert!(anchor.contains("- fingerprint: "), "anchor carries fingerprint: {anchor}");

        let pointer = minimize_overflow_stub_to_pointer(&stub).expect("minimize");
        assert!(pointer.contains("- file_path: /tmp/carry.txt"));
        assert!(pointer.contains("- fingerprint: "), "pointer keeps retrieval signal");
        assert!(is_preserved_tool_overflow_stub(&pointer));

        // Legacy stubs (pre-fingerprint) minimize cleanly without fabricated fields.
        let legacy = format!(
            "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\nOutput preserved for tool `x`.\n- file_path: /tmp/legacy.txt"
        );
        let minimized = minimize_overflow_stub_to_pointer(&legacy).unwrap();
        assert!(!minimized.contains("fingerprint"));
    }

    #[test]
    fn fingerprint_skips_degenerate_gist() {
        // A single repeated character carries no recall signal; the gist segment is
        // omitted entirely so aged stubs of degenerate outputs stay minimal.
        let fp = stub_fingerprint_line(&"x".repeat(1_000)).expect("non-empty content has fingerprint");
        assert!(fp.contains("sha="), "{fp}");
        assert!(!fp.contains("gist="), "no gist for degenerate body: {fp}");

        // Real signal still gets a gist.
        let fp2 = stub_fingerprint_line("warning: sched_ctx drifted from kernel root\n");
        assert!(fp2.unwrap().contains("gist=\""), "informative line kept");
    }

    #[test]
    fn fingerprint_keywords_dedup_case_insensitively_and_stay_deterministic() {
        // Mixed-case repeats of the same token must collapse to one keyword,
        // keeping the first-seen casing; the set-based dedup must not perturb
        // ordering vs. the previous linear scan.
        let content = "sched_ctx SCHED_CTX Sched_Ctx root_idx ROOT_IDX payloadxyz\n";
        let keys = extract_fingerprint_keywords(content);
        let sched = keys.iter().filter(|k| k.eq_ignore_ascii_case("sched_ctx")).count();
        assert_eq!(sched, 1, "case-insensitive dedup collapses repeats: {keys:?}");
        assert!(keys.contains(&"sched_ctx".to_string()), "first casing kept: {keys:?}");
        assert!(keys.len() <= FINGERPRINT_KEY_COUNT);

        // Fully deterministic across calls (no RNG / hash-order leakage into output).
        assert_eq!(keys, extract_fingerprint_keywords(content));
    }

    #[test]
    fn age_out_overflow_stub_previews_is_idempotent() {
        let stub = overflow_stub_with_preview("/tmp/session/read-xyz.txt", "read_file");
        // Two user turns place the stub outside the protected tail window
        // (before retained_turn_start).
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("s", "read_file"),
            tool_result("s", "placeholder"),
            user_msg("q2"),
            user_msg("q3"),
        ];
        messages[2].content = Value::String(stub);

        age_out_overflow_stub_previews(&mut messages, 1);
        let after_first = value_to_string(&messages[2].content);
        assert!(!after_first.contains("Preview ("));

        // Run again: it is already in anchor shape and the content must not
        // change (prevents stub->stub churn).
        age_out_overflow_stub_previews(&mut messages, 1);
        assert_eq!(value_to_string(&messages[2].content), after_first);
    }

    #[test]
    fn age_out_overflow_stub_previews_respects_protected_tail() {
        // One early stub (outside the tail window) and one recent stub (inside
        // the tail window).
        let early = overflow_stub_with_preview("/tmp/session/early.txt", "read_file");
        let recent = overflow_stub_with_preview("/tmp/session/recent.txt", "read_file");
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("early", "read_file"),
            tool_result("early", "placeholder"),
            user_msg("q2"),
            assistant_call("recent", "read_file"),
            tool_result("recent", "placeholder"),
        ];
        messages[2].content = Value::String(early);
        messages[5].content = Value::String(recent.clone());

        // Protect the most recent user turn (from q2 on): the early stub folds,
        // while the recent one inside the tail window keeps its full preview.
        age_out_overflow_stub_previews(&mut messages, 1);
        assert!(!value_to_string(&messages[2].content).contains("Preview ("));
        assert_eq!(value_to_string(&messages[5].content), recent);
        assert!(value_to_string(&messages[5].content).contains("Preview ("));
    }
}

/// Output of "read/retrieval" tools is zero-compressed (no trimming, no dedup
/// folding, no whole-group deletion); over the threshold it only gets
/// "zero-compression spilled to the session file + a pointer stub". Such
/// output is expensive to reproduce, and once compressed away the model
/// re-runs the same retrieval again and again (the classic amnesia /
/// spinning-in-place symptom).
///
/// This now consults the history-retention policy declared by the tool itself
/// (`ToolHistoryPolicyRegistration`, see each tool's registration file) instead
/// of hard-coding a tool-name list here. Unregistered tools allow lossy
/// compression by default; only tools that explicitly declare
/// `lossy_compress: Never` (`read_file` / retrieval tools / `execute_command`)
/// return true. `plan` no longer forbids lossy compression: the latest version
/// is fully preserved by the recent-tool-group protection window, while older
/// versions may be summary-compressed to free context. Note this is orthogonal
/// to "whether LLM trimming is allowed" — see `llm_prune.rs`.
pub(super) fn is_non_compressible_tool(tool_name: &str) -> bool {
    !crate::ai::tools::registry::common::tool_history_policy(tool_name).allows_lossy_compress()
}

/// Plans a deterministic asset and a stable stub carrying `file_path` for
/// high-precision tool results not yet spilled.
///
/// This function only produces [`PlannedArchiveWrite`] and never touches the
/// filesystem. The caller commits the whole fold plan together after deciding
/// to adopt it, so a rejected speculative fold leaves no disk side effects.
/// Existing stubs are reused directly and produce no new writes.
pub(super) fn plan_noncompressible_tool_result_for_fold(
    overflow_dir: Option<&Path>,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
    recall_lines: &[String],
) -> Option<(String, Option<PlannedArchiveWrite>)> {
    if is_preserved_tool_overflow_stub(content) {
        return Some((content.to_string(), None));
    }
    let overflow_dir = overflow_dir?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let identity = format!("{tool_call_id}\0{content}");
    let digest = content_sha256_hex(identity.as_bytes());
    let path = overflow_dir
        .join(PRESERVED_TOOL_OVERFLOW_DIR)
        .join(format!("folded-{safe_tool}-{}.txt", &digest[..24]));
    let stub = build_preserved_tool_overflow_stub(&path, tool_name, content, recall_lines);
    Some((
        stub,
        Some(PlannedArchiveWrite::new(path, content.to_string())),
    ))
}

/// Immediate stable archive used by the LLM-prune path; independent of fold's
/// two-phase planning. The file name is deterministically derived from
/// `(tool_call_id, content)` (not a random uuid + timestamp), matching the
/// content addressing of the `spilled-` / `folded-` archive naming strategies.
///
/// LLM-guided pruning (`llm_prune::apply_pruning`) operates on the temporary
/// `messages` projection rebuilt before each model request, so the same
/// canonical tool message gets pruned again whenever a later turn rebuilds the
/// projection. Random file names would mint a fresh copy every turn and change
/// the stub text each time, which breaks the prompt cache from that point on
/// and grows disk copies monotonically. With a deterministic name, archiving is
/// idempotent: an existing file is skipped and the stub text stays stable across
/// turns. Hashing the content into the name keeps that idempotence honest when
/// the same call id is replayed with different content — the write then targets
/// a distinct file instead of blindly reusing stale bytes.
pub(super) fn preserve_pruned_tool_result_stable(
    overflow_dir: Option<&Path>,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
    recall_lines: &[String],
) -> Option<String> {
    if is_preserved_tool_overflow_stub(content) {
        return Some(content.to_string());
    }
    let path = overflow_dir.and_then(|dir| {
        write_preserved_tool_overflow_file_stable(dir, tool_call_id, tool_name, content)
    })?;
    Some(build_preserved_tool_overflow_stub(
        &path,
        tool_name,
        content,
        recall_lines,
    ))
}

/// Deterministic archive write for the pruned path; the naming rationale is
/// documented on [`preserve_pruned_tool_result_stable`].
fn write_preserved_tool_overflow_file_stable(
    overflow_dir: &Path,
    tool_call_id: &str,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let safe_id = sanitize_overflow_name_component(tool_call_id);
    let identity = format!("{tool_call_id}\0{content}");
    let digest = content_sha256_hex(identity.as_bytes());
    let file_name = format!("pruned-{safe_tool}-{safe_id}-{}.txt", &digest[..12]);
    let path = dir.join(file_name);
    // Idempotent for identical content: the same (id, content) maps to the same
    // path, so an existing file is never rewritten and the stub text (and the
    // prompt cache prefix) stays stable across turns.
    if !path.exists() {
        std::fs::write(&path, content).ok()?;
    }
    Some(path)
}

/// Normalizes a tool name / id into a safe file-name fragment containing only
/// alphanumerics and `-`/`_`.
fn sanitize_overflow_name_component(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
}

/// The immediate (non-speculative) spill path. The file name is derived
/// deterministically from `(tool_call_id, content)` (rather than a random uuid
/// + timestamp): the same canonical tool result gets re-spilled every time a
/// later turn rebuilds the projection, and random naming would make "each
/// compaction round mints a new copy" bloat without bound (measured: one
/// session produced 368 archive files for only 211 unique contents), with the
/// model's read-back pointer drifting every round so it never gets stable
/// content. With deterministic naming the same result maps idempotently to the
/// same file: if it exists it is reused without rewriting, and the stub text
/// stays stable across rounds (consistent with the `folded-` fold and
/// `pruned-` prune archive naming schemes). fold must not call this function;
/// fold's writes are committed together via `PlannedArchiveWrite` once a
/// candidate is adopted.
fn write_preserved_tool_overflow_file(
    overflow_dir: &Path,
    tool_call_id: Option<&str>,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = sanitize_overflow_name_component(tool_name);
    let identity = format!("{}\0{content}", tool_call_id.unwrap_or(""));
    let digest = content_sha256_hex(identity.as_bytes());
    let path = dir.join(format!("spilled-{safe_tool}-{}.txt", &digest[..24]));
    // Idempotent: the content does not change across rounds; if it exists it is
    // not rewritten (also keeping the prompt cache stable).
    if !path.exists() {
        std::fs::write(&path, content).ok()?;
    }
    Some(path)
}

fn build_preserved_tool_overflow_stub(
    path: &Path,
    tool_name: &str,
    full_content: &str,
    recall_lines: &[String],
) -> String {
    // The full text is still spilled to disk to control context size, but the
    // stub keeps a head+tail preview so later turns own a "recall anchor" — the
    // model can judge whether it really needs to read_file again, avoiding the
    // "amnesia / endless re-reading" that follows when early-read code is moved
    // away. The notice wording stays neutral: it clearly says "read only if the
    // full content is needed", preventing the LLM from unconditionally
    // re-reading on sight of a file_path and looping spill → re-read → spill
    // again forever.
    let preview = build_overflow_content_preview(full_content);
    let tool_hint = preserved_tool_overflow_hint(tool_name, recall_lines);
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {}",
        path.display(),
    );
    if let Some(fingerprint) = stub_fingerprint_line(full_content) {
        out.push('\n');
        out.push_str(&fingerprint);
    }
    for line in recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(tool_hint);
    out.push('\n');
    out.push_str(&preview);
    out
}

fn preserved_tool_overflow_hint(tool_name: &str, recall_lines: &[String]) -> &'static str {
    let has_original_file_path = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_file_path: "));
    let has_original_command = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_command: "));
    match tool_name {
        "read_file" if has_original_file_path => {
            "Archived snapshot of an earlier read. `original_range` marks lines already covered - for current content, read `original_file_path` past that range (identical re-reads are deduped); read `file_path` only for the exact historical output."
        }
        "read_file" => {
            "Archived snapshot of an earlier read. Read `file_path` only if the preview is insufficient and you need the exact output; identical re-reads are deduped."
        }
        "execute_command" if has_original_command => {
            "Archived command output. Continue from `original_command` / `original_cwd`; `file_path` is a text archive, not a source file - read it only for the full log."
        }
        _ => "Archived output; `file_path` holds the full text. Read it only if the preview is insufficient.",
    }
}

/// Max characters kept in the fingerprint gist; long build-log headers carry mostly
/// boilerplate past this point.
const FINGERPRINT_GIST_MAX_CHARS: usize = 72;

/// A gist qualifies only above this many distinct characters: degenerate bodies
/// (one repeated char filling an oversized command echo) carry no recall signal,
/// and padding them into every aged stub wastes bytes under tight stub budgets.
const FINGERPRINT_GIST_MIN_DISTINCT_CHARS: usize = 4;

/// How many discriminative keywords the fingerprint carries into aged stub stages.
const FINGERPRINT_KEY_COUNT: usize = 5;

/// Build the single-line `- fingerprint:` payload from archived content.
///
/// Purely deterministic in the content (no timestamps, no RNG): reprojections
/// rebuild stubs every turn, and byte-identical stubs keep prompt caches warm.
/// Layout: `sha=<12 hex> gist="<one informative line>" keys=<k1,k2,...>`; empty
/// segments are omitted rather than rendered blank.
fn stub_fingerprint_line(full_content: &str) -> Option<String> {
    if full_content.trim().is_empty() {
        return None;
    }
    let digest = content_sha256_hex(full_content.as_bytes());
    let mut parts = vec![format!("sha={}", &digest[..12])];
    if let Some(gist) = extract_fingerprint_gist(full_content) {
        parts.push(format!("gist=\"{gist}\""));
    }
    let keys = extract_fingerprint_keywords(full_content);
    if !keys.is_empty() {
        parts.push(format!("keys={}", keys.join(",")));
    }
    Some(format!("- fingerprint: {}", parts.join(" ")))
}

/// Pick the first head-region line with real alphanumeric signal as the gist,
/// whitespace-collapsed and clamped on char boundaries. Returns `None` when the
/// head region holds only decoration (separators, bare JSON braces, ...).
fn extract_fingerprint_gist(content: &str) -> Option<String> {
    content.lines().take(30).find_map(|line| {
        let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        let distinct = collapsed.chars().collect::<FxHashSet<char>>().len();
        (distinct >= FINGERPRINT_GIST_MIN_DISTINCT_CHARS).then_some(collapsed)
    }).map(|gist| {
        let head: String = gist.chars().take(FINGERPRINT_GIST_MAX_CHARS).collect();
        if head.len() < gist.len() {
            format!("{head}\u{2026}")
        } else {
            head
        }
    })
}

/// Hand-tuned heuristic salience for fingerprint keywords, matching what models
/// tend to recall from tool outputs: error codes, type/path identifiers, flags.
fn fingerprint_token_score(token: &str) -> i32 {
    let base = token.len().min(24) as i32;
    let digit_bonus = i32::from(token.chars().any(|c| c.is_ascii_digit())) * 4;
    let has_camel_boundary = token
        .as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
    let structure_bonus = i32::from(token.contains('_') || has_camel_boundary) * 3;
    // Full paths rarely survive model recollection; their trailing segment usually
    // does, so whole path-shaped tokens compete with a handicap.
    let slash_handicap = i32::from(token.contains('/')) * (-base / 2 - 2);
    base + digit_bonus + structure_bonus + slash_handicap
}

/// Extract up to `FINGERPRINT_KEY_COUNT` discriminative keywords from content.
///
/// Casing is preserved verbatim: `search_overflow` matches literally and defaults
/// to case-sensitive, so a keyword only helps retrieval if it still occurs exactly
/// in the archived text.
fn extract_fingerprint_keywords(content: &str) -> Vec<String> {
    const MIN_TOKEN_LEN: usize = 5;
    const MAX_TOKEN_LEN: usize = 32;
    // Generic prose/log vocabulary that crowds out identifiers when left unfiltered.
    const STOP_WORDS: [&str; 14] = [
        "error", "result", "output", "content", "value", "string", "unknown",
        "there", "which", "would", "about", "failed", "success", "warning",
    ];
    let mut candidates: Vec<(String, i32)> = Vec::new();
    // Case-insensitive dedup via a membership set (not a linear scan of
    // `candidates`): this runs on the full raw overflow body (up to the 64K
    // hard cap) and is rebuilt every turn during reprojection, so an
    // O(tokens × distinct) scan is a hot-path cost. Insertion order into
    // `candidates` is unchanged, and the stable sort below keeps appearance
    // order as the score tiebreak, so output stays byte-identical.
    let mut seen: FxHashSet<String> = FxHashSet::default();
    for raw in content.split(|c: char| {
        !(c.is_ascii_alphanumeric() || matches!(c, '_' | '/' | '.' | '-'))
    }) {
        let len_ok = (MIN_TOKEN_LEN..=MAX_TOKEN_LEN).contains(&raw.len());
        let wordy = raw.chars().any(|c| c.is_ascii_alphabetic());
        let diversified = !raw.chars().all(|c| c == raw.chars().next().unwrap_or('-'));
        let lowered = raw.to_ascii_lowercase();
        if !len_ok || !wordy || !diversified || STOP_WORDS.contains(&lowered.as_str()) {
            continue;
        }
        // Keep the first-seen casing; a second occurrence in any casing is skipped.
        if !seen.insert(lowered) {
            continue;
        }
        candidates.push((raw.to_string(), fingerprint_token_score(raw)));
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(FINGERPRINT_KEY_COUNT);
    candidates.into_iter().map(|(token, _)| token).collect()
}

pub(super) fn build_tool_overflow_recall_lines(tool_name: &str, arguments: &str) -> Vec<String> {
    let Ok(args) = serde_json::from_str::<Value>(arguments) else {
        return Vec::new();
    };

    match tool_name {
        "read_file" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(path) = value_string_from_keys(&args, &["file_path", "path", "filePath"]) {
                lines.push(format!(
                    "- original_file_path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }

            if let Some((label, range)) = read_file_range_summary(&args) {
                lines.push(format!("- original_range: {label}={range}"));
            }
            lines
        }
        "tree" => value_string_from_keys(&args, &["path"])
            .map(|path| {
                vec![format!(
                    "- original_path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                )]
            })
            .unwrap_or_default(),
        "execute_command" | "run_command" | "shell" | "bash" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(command) = value_string_from_keys(&args, &["command"]) {
                lines.push(format!(
                    "- original_command: {}",
                    truncate_to_chars(&normalize_whitespace(&command), 720)
                ));
            }
            if let Some(cwd) = value_string_from_keys(&args, &["cwd"]) {
                let cwd = normalize_whitespace(&cwd);
                if !cwd.is_empty() {
                    lines.push(format!("- original_cwd: {}", truncate_to_chars(&cwd, 240)));
                }
            }
            lines
        }
        _ => Vec::new(),
    }
}

fn value_string_from_keys(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(|value| value.to_string())
}

fn read_file_range_summary(args: &Value) -> Option<(&'static str, String)> {
    let start_line = args.get("startLine").and_then(Value::as_u64);
    let end_line = args.get("endLine").and_then(Value::as_u64);
    if let (Some(start_line), Some(end_line)) = (start_line, end_line) {
        return Some(("lines", format!("{start_line}..{end_line}")));
    }

    let offset = args.get("offset").and_then(Value::as_u64);
    let limit = args.get("limit").and_then(Value::as_u64);
    match (offset, limit) {
        (Some(offset), Some(limit)) if limit > 0 => Some((
            "lines",
            format!("{offset}..{}", offset + limit.saturating_sub(1)),
        )),
        (Some(offset), _) => Some(("offset", offset.to_string())),
        _ => None,
    }
}

pub(super) fn is_preserved_tool_overflow_stub(text: &str) -> bool {
    let text = text.trim_start();
    if text.starts_with(PRESERVED_TOOL_OVERFLOW_STUB_PREFIX) {
        return text.contains("\n- file_path: ");
    }
    // Legacy formats (older sessions):
    // - "Output preserved for non-compressible tool `..."  (pre-refactor)
    // - "Output preserved for tool `..."                   (new format)
    if (text.starts_with(LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX)
        || text.starts_with("Output preserved for tool `"))
        && text.contains("\n- file_path: ")
    {
        return true;
    }
    false
}

/// Collapses the head+tail preview body of an already-spilled tool overflow
/// stub into a "single-line recall anchor" (keeping only the `file_path:`
/// pointer + the read-back notice, dropping `Preview (...)` and everything
/// after it).
///
/// Old stubs' previews accumulate monotonically in long sessions (real case:
/// 800 stubs × ~1KB ≈ 849KB), while `file_path` is the only information the
/// model needs for an exact read-back — the preview is just a "first recall
/// anchor", and once the stub has drifted away from the current work focus the
/// preview body's marginal value approaches 0. After collapsing, each entry
/// shrinks from ~1KB to ~200 chars with zero loss of recall ability (the
/// original can still be read back with read_file).
///
/// Returns `None` on parse failure (file_path or tool name not found), leaving
/// the original untouched — it is never corrupted. Stubs already in anchor
/// shape (no `Preview (` section) also return `None`, guaranteeing idempotence
/// and no stub→stub churn.
fn collapse_overflow_stub_to_anchor(text: &str) -> Option<String> {
    if !is_preserved_tool_overflow_stub(text) {
        return None;
    }
    // Already in anchor shape (no preview section): idempotent; returning None
    // means no rewrite is needed.
    if !text.contains("Preview (") {
        return None;
    }
    // Parse tool_name in both the old and new formats.
    let tool_name = text
        .split_once("non-compressible tool `")
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(name, _)| name.to_string())
        .or_else(|| {
            text.split_once("Output preserved for tool `")
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(name, _)| name.to_string())
        })?;
    let file_path = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(str::trim)
        .filter(|p| !p.is_empty())?;
    let recall_lines = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("- original_"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let tool_hint = if tool_name == "read_file" {
        if recall_lines
            .iter()
            .any(|line| line.starts_with("- original_file_path: "))
        {
            "Archived snapshot of an earlier read. Read `original_file_path` for current content, `file_path` only for the exact historical output."
        } else {
            "Archived snapshot of an earlier read; read `file_path` only for the exact historical output."
        }
    } else if tool_name == "execute_command"
        && recall_lines
            .iter()
            .any(|line| line.starts_with("- original_command: "))
    {
        "Archived command output; usually no re-read needed - continue from `original_command` / `original_cwd`."
    } else {
        "Full output at `file_path`; read it only if needed."
    };
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {file_path}"
    );
    for line in &recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    if let Some(fp) = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- fingerprint: "))
    {
        out.push('\n');
        out.push_str("- fingerprint: ");
        out.push_str(fp.trim());
        out.push('\n');
    }
    out.push('\n');
    out.push_str(tool_hint);
    Some(out)
}

/// Path C's final hard-budget stage may only strip an overflow stub's preview
/// and recall notice; the sole asset pointer must never be handed to generic
/// head+tail truncation. The returned minimal stub still keeps the protocol
/// marker, tool name, and `file_path`, so the original evidence can still be
/// read back precisely.
fn minimize_overflow_stub_to_pointer(text: &str) -> Option<String> {
    if !is_preserved_tool_overflow_stub(text) {
        return None;
    }
    let tool_name = text
        .split_once("non-compressible tool `")
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(name, _)| name)
        .or_else(|| {
            text.split_once("Output preserved for tool `")
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(name, _)| name)
        })?;
    let file_path = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(str::trim)
        .filter(|path| !path.is_empty())?;
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`.\n\
         - file_path: {file_path}"
    );
    // Carry the content fingerprint into the final pointer form so even the most
    // compressed stub keeps retrieval signal; legacy stubs simply omit it.
    if let Some(fp) = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- fingerprint: "))
    {
        out.push_str("\n- fingerprint: ");
        out.push_str(fp.trim());
    }
    Some(out)
}

pub(super) fn minimize_overflow_stubs_for_hard_budget(messages: &mut [Message]) {
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        let Value::String(text) = &message.content else {
            continue;
        };
        let Some(minimal) = minimize_overflow_stub_to_pointer(text) else {
            continue;
        };
        message.content = Value::String(minimal);
    }
}

pub(super) fn is_preserved_tool_overflow_content(content: &Value) -> bool {
    content
        .as_str()
        .is_some_and(is_preserved_tool_overflow_stub)
}

/// Age-folds the preview body of overflow stubs "outside the protected tail
/// window" into single-line anchors. Applies only to already-spilled tool
/// stubs (`is_preserved_tool_overflow_stub`) and never touches original tool
/// results; stubs inside the tail window keep their full head+tail preview
/// (the current work focus still needs its recall context).
///
/// Complements budget-driven group folding: even when a stub's group escapes
/// `fold_early_tool_groups` because of near-end protection, its preview body
/// still age-collapses as the conversation advances, preventing hundreds of
/// early read_file previews from accumulating monotonically in the history.
pub(super) fn age_out_overflow_stub_previews(
    messages: &mut [Message],
    keep_recent_user_turns: usize,
) {
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for message in messages.iter_mut().take(protected_tail_start) {
        if message.role != "tool" {
            continue;
        }
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some(anchor) = collapse_overflow_stub_to_anchor(text) {
            message.content = Value::String(anchor);
        }
    }
}

/// Generates a head+tail preview for spilled content. Short content is kept in
/// full; long content keeps a few leading/trailing lines with the middle folded
/// into a placeholder line that states the omitted line count.
fn build_overflow_content_preview(content: &str) -> String {
    const HEAD_LINES: usize = 8;
    const TAIL_LINES: usize = 4;
    const MAX_LINE_CHARS: usize = 200;
    const MAX_KEY_LINES: usize = 20;

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
    // Large source/text results: attach a structural index with line-numbered
    // heads (key lines like fn/struct/impl/use/errors), aligned with the
    // capture-time overflow stub's key_lines. After the compression spill the
    // model can still locate the target region by line number and re-read only
    // the needed range, instead of blindly re-reading the middle of a
    // multi-thousand-line file.
    let key_lines = extract_key_lines(content, MAX_KEY_LINES);
    if !key_lines.is_empty() {
        out.push_str(&format!("- key_lines ({}):\n", key_lines.len()));
        for line in &key_lines {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }
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

const MERGED_PRESERVED_USER_STUB_PREFIX: &str = "较早的用户内容已归档（共 ";
const MERGED_PRESERVED_ARCHIVE_DIR_PREFIX: &str = "归档目录: ";

fn parse_merged_preserved_message_stub(text: &str) -> Option<(usize, Vec<String>)> {
    let count = text
        .strip_prefix(MERGED_PRESERVED_USER_STUB_PREFIX)?
        .split_once(" 条")?
        .0
        .parse::<usize>()
        .ok()?;
    if count == 0 {
        return None;
    }

    let mut dirs = Vec::new();
    for line in text.lines() {
        // Compatibility with the first-version merged stub's "归档文件"
        // (archive file) field; that field actually stores a directory.
        let Some(dir) = line
            .strip_prefix(MERGED_PRESERVED_ARCHIVE_DIR_PREFIX)
            .or_else(|| line.strip_prefix("归档文件: "))
            .map(str::trim)
            .filter(|dir| !dir.is_empty())
        else {
            continue;
        };
        if !dirs.iter().any(|existing| existing == dir) {
            dirs.push(dir.to_string());
        }
    }
    (!dirs.is_empty()).then_some((count, dirs))
}

fn build_merged_preserved_message_stub(count: usize, dirs: &[String]) -> String {
    let mut merged =
        format!("较早的用户内容已归档（共 {count} 条，原文零压缩保存在会话归档目录）。\n");
    for dir in dirs {
        merged.push_str(MERGED_PRESERVED_ARCHIVE_DIR_PREFIX);
        merged.push_str(dir);
        merged.push('\n');
    }
    merged.push_str(
        "这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖较早用户原文且现有摘要不足时，逐个使用 tree 列出上述归档目录，再按时间戳和类型定位 JSON 文件，最后使用 read_file 读取具体文件；不要对目录直接调用 read_file。",
    );
    merged
}

/// Merges user/image spill stubs outside the protected tail window into a
/// single pointer carrying the archive directory.
///
/// user/image stubs are role=user placeholder messages: `first_trim_candidate`
/// / truncate / emergency cap / tool-only age folding never touch them again,
/// so long sessions (especially once images are billed at nominal cost) let
/// the stubs accumulate monotonically with no convergence path. Folding old
/// stubs into a single merged pointer shrinks placeholder overhead from O(N)
/// to O(1); the originals stay zero-compressed on disk, and the
/// directory + timestamp naming allows read-back via tree + read_file. Only
/// stubs outside the protected tail window are merged; the most recent turns
/// keep per-stub pointers for precise recall.
pub(super) fn merge_old_user_overflow_stubs(
    messages: &mut Vec<Message>,
    keep_recent_user_turns: usize,
) {
    const MERGE_MIN_STUB_COUNT: usize = 4;

    // Later mid-turn summaries still split at the most recent 2/3 real user
    // boundaries. Even if the current budget has already dropped
    // keep_recent_user_turns to 1, these structural boundaries must not be
    // folded into an internal_note as well, otherwise retained_turn_start would
    // misjudge "not enough history" and the whole old segment could not enter
    // the summary.
    let structural_tail_turns =
        keep_recent_user_turns.max(KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX);
    let protected_tail_start = retained_turn_start(messages, structural_tail_turns);
    let mut stub_indices = Vec::new();
    let mut merged_stub_count = 0usize;
    let mut single_stub_count = 0usize;
    let mut archived_message_count = 0usize;
    let mut dirs: Vec<String> = Vec::new();
    for (idx, message) in messages.iter().take(protected_tail_start).enumerate() {
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some((count, merged_dirs)) = parse_merged_preserved_message_stub(text) {
            stub_indices.push(idx);
            merged_stub_count += 1;
            archived_message_count = archived_message_count.saturating_add(count);
            for dir in merged_dirs {
                if !dirs.iter().any(|existing| existing == &dir) {
                    dirs.push(dir);
                }
            }
            continue;
        }
        if message.role != "user" {
            continue;
        }
        let Some((_kind, file_path)) = parse_preserved_message_stub(text) else {
            continue;
        };
        stub_indices.push(idx);
        single_stub_count += 1;
        archived_message_count = archived_message_count.saturating_add(1);
        if let Some(parent) = Path::new(&file_path).parent() {
            let dir = parent.to_string_lossy().into_owned();
            if !dirs.iter().any(|d| d == &dir) {
                dirs.push(dir);
            }
        }
    }
    // With no merged pointer yet, fold only after at least 4 accumulate; once a
    // merged pointer exists, old and new stubs are merged back into that same
    // one, avoiding a permanent extra merged pointer for every 4 new stubs and
    // the re-degradation to O(N).
    // Snapshots from the old version may already contain a role=user merged
    // pointer (generated before the fix). Even when this round needs no
    // re-folding (a single merged pointer and no new per-stub stubs), the
    // existing merged pointer's role must be migrated to internal_note,
    // otherwise retained_turn_start keeps treating it as a real user boundary
    // and pollutes later summary split points.
    for &idx in &stub_indices {
        let is_merged = match &messages[idx].content {
            Value::String(text) => parse_merged_preserved_message_stub(text).is_some(),
            _ => false,
        };
        if is_merged && messages[idx].role != ROLE_INTERNAL_NOTE {
            messages[idx].role = ROLE_INTERNAL_NOTE.to_string();
        }
    }
    if dirs.is_empty()
        || (merged_stub_count == 0 && single_stub_count < MERGE_MIN_STUB_COUNT)
        || (merged_stub_count == 1 && single_stub_count == 0)
    {
        return;
    }

    let merged = build_merged_preserved_message_stub(archived_message_count, &dirs);

    // Delete back-to-front to keep indices valid; the merged pointer is written
    // onto the first stub's Message. It describes runtime archive metadata, not
    // a new user request; if it kept the `user` role, deleting the other stubs
    // would forge a turn boundary and make later tail/summary splitting
    // misjudge multiple rounds of old messages as one recent user turn.
    for &idx in stub_indices.iter().skip(1).rev() {
        messages.remove(idx);
    }
    messages[stub_indices[0]].role = ROLE_INTERNAL_NOTE.to_string();
    messages[stub_indices[0]].content = Value::String(merged);
}

pub(super) fn is_preserved_user_or_image_stub(text: &str) -> bool {
    parse_merged_preserved_message_stub(text).is_some()
        || parse_preserved_message_stub(text).is_some()
}

fn parse_preserved_message_stub(text: &str) -> Option<(String, String)> {
    if let Some(payload) = text.strip_prefix(PRESERVED_CONTENT_STUB_PREFIX) {
        let value = serde_json::from_str::<Value>(payload).ok()?;
        let kind = value.get("kind")?.as_str()?.to_string();
        let file_path = value.get("file_path")?.as_str()?.to_string();
        return ((kind == "user" || kind == "image") && !file_path.is_empty())
            .then_some((kind, file_path));
    }

    let kind = if text.starts_with("较早的用户图片内容已归档") {
        "image"
    } else if text.starts_with("较早的用户") {
        "user"
    } else {
        return None;
    };
    let file_path = text
        .lines()
        .find_map(|line| line.strip_prefix("归档文件: "))?
        .trim();
    (!file_path.is_empty()).then(|| (kind.to_string(), file_path.to_string()))
}

/// Converts the internal archive protocol into a context note the model can
/// understand, while staying compatible with old JSON stubs already on disk.
pub(in crate::ai) fn normalize_preserved_message_stubs_for_model(messages: &mut [Message]) {
    for message in messages {
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some((count, dirs)) = parse_merged_preserved_message_stub(text) {
            message.content = Value::String(build_merged_preserved_message_stub(count, &dirs));
            // A merged pointer is runtime archive metadata, not a user request;
            // old-version snapshots may still have it on disk as role=user, so
            // this is the fallback migration to keep user/assistant pairing
            // unpolluted.
            message.role = ROLE_INTERNAL_NOTE.to_string();
            continue;
        }
        let Some((kind, file_path)) = parse_preserved_message_stub(text) else {
            continue;
        };
        message.content = Value::String(build_preserved_message_overflow_stub(
            Path::new(&file_path),
            &kind,
        ));
    }
}

fn first_preserved_content_spill_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages, budget);
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

        // value_to_string folds an image's base64 into "[图片]" and cannot
        // reflect the real size. For image messages, the serialized length of
        // the raw content decides whether to spill, matching the intent of
        // "moving large images into session temp files"; ordinary text messages
        // are still billed by value_to_string.
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
    let content_kind = if kind == "image" { "图片" } else { "文本" };
    format!(
        "较早的用户{content_kind}内容已归档，原文未丢失。\n归档文件: {}\n这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖原文时，才使用 read_file 读取该文件。",
        path.display()
    )
}

pub(super) fn try_spill_preserved_message_to_stub(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) -> bool {
    let Some(idx) = first_preserved_content_spill_candidate(messages, budget) else {
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

/// Proactively moves oversized old user / image messages (before the protected
/// tail window) into session temp files, replacing them in place with compact
/// stubs. The originals stay zero-compressed on disk but no longer occupy each
/// request's payload.
///
/// This complements the budget-driven in-loop spill: ever since images are
/// billed nominally at [`IMAGE_BUDGET_CHARS`] in the budget, a single large
/// image no longer triggers `messages_total_chars > max_chars` by itself, so
/// the in-loop spill would never be called. This pass instead spills "whenever
/// an old message's raw size exceeds the threshold, regardless of budget",
/// both ensuring large images / long user texts get zero-compression archived
/// and keeping them from polluting every later request. The newest turn's
/// user/image messages (inside the protected tail window) are never spilled.
pub(super) fn spill_oversized_preserved_messages(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) {
    while try_spill_preserved_message_to_stub(messages, overflow_dir, budget) {}
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

pub(super) fn tool_line_signature(line: &str) -> String {
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

pub(super) fn build_persisted_summary_text(messages: &[Message], max_chars: usize) -> String {
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
            line.push_str(&format!("repeated ×{} ", turn.count));
        }
        if !turn.topic_label.is_empty() {
            line.push_str("Topic: ");
            line.push_str(&turn.topic_label);
            line.push_str(" | ");
        }
        if !turn.user.is_empty() {
            line.push_str("User: ");
            line.push_str(&turn.user);
        }
        if !turn.assistant_final.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Assistant's previous answer (not independently verified): ");
            line.push_str(&turn.assistant_final);
        }
        if !turn.tool_names.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Tools: ");
            line.push_str(&turn.tool_names.join(", "));
        }
        if !turn.tool_highlights.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("Key: ");
            line.push_str(&turn.tool_highlights.join(", "));
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
        if !turn.tool_highlights.is_empty() {
            line.push_str(" => ");
            line.push_str(&turn.tool_highlights.join(", "));
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
                            format!("- Earlier summary: {normalized}"),
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
        let tool_blob = format!(
            "Verified facts and sources:\n{}",
            known_tool_lines.join("\n")
        );
        tool_blob.chars().count().min(max_chars / 3)
    };
    let body_budget = max_chars
        .saturating_sub(reserved_tool_chars)
        .max(max_chars / 2);
    let mut lines: Vec<String> = Vec::new();
    if !initial_goal.is_empty()
        && !push_line_with_budget(
            &mut lines,
            format!("Main request: {initial_goal}"),
            body_budget,
        )
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
        let _ = push_line_with_budget(
            &mut lines,
            "Verified facts and sources:".to_string(),
            max_chars,
        );
        for line in known_tool_lines {
            if !push_line_with_budget(&mut lines, line, max_chars) {
                break;
            }
        }
    }

    if !recent_turns.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "Current work:".to_string(), max_chars);
        for t in &recent_turns {
            let mut parts = Vec::new();
            if !t.topic_label.is_empty() {
                parts.push(format!("Topic: {}", t.topic_label));
            }
            if !t.user.is_empty() {
                parts.push(format!("User: {}", t.user));
            }
            if !t.assistant_final.is_empty() {
                parts.push(format!(
                    "Assistant's previous answer (not independently verified): {}",
                    t.assistant_final
                ));
            }
            if !t.tool_names.is_empty() {
                parts.push(format!("Tools: {}", t.tool_names.join(", ")));
            }
            if !t.tool_highlights.is_empty() {
                parts.push(format!("Key: {}", t.tool_highlights.join(", ")));
            }
            let line = format!("- {}", parts.join(" | "));
            if !push_line_with_budget(&mut lines, summarize_text(&line, 600), max_chars) {
                break;
            }
        }
    }

    if !pending_tasks.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "Pending tasks:".to_string(), max_chars);
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
