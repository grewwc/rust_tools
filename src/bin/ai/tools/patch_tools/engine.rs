use super::*;
use std::fs;
use std::path::Path;
use serde_json::Value;
use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::storage::file_store::FileStore;
pub(crate) fn try_apply_hunk_at(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    start: usize,
    mode: MatchMode,
    context_policy: ContextPolicy,
) -> Option<(Vec<String>, usize)> {
    let mut out = Vec::new();
    let mut idx = start;
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(s) => {
                let cur = orig_lines.get(idx)?;
                if context_policy == ContextPolicy::Require && !lines_match(cur, s, mode) {
                    return None;
                }
                out.push(cur.clone());
                idx += 1;
            }
            UnifiedLine::Remove(s) => {
                let cur = orig_lines.get(idx)?;
                if !lines_match(cur, s, mode) {
                    return None;
                }
                idx += 1;
            }
            UnifiedLine::Add(s) => {
                out.push(s.clone());
            }
        }
    }
    Some((out, idx))
}

/// Locates the application start (0-based) of a hunk under the given match mode.
/// Returns `Ok(Some(pos))` for a unique successful location; `Ok(None)` when no
/// position in the whole file matches (the caller may retry with a more lenient
/// mode); `Err` when a position was found but is ambiguous or out of order.
pub(crate) fn locate_hunk(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
    hunk_idx: usize,
    hunk_total: usize,
) -> Result<Option<usize>, String> {
    let old_len = hunk_old_line_count(hunk);
    if old_len == 0 {
        if hunk.old_start == 0 {
            return Ok(Some(cursor));
        }
        let nominal = hunk.old_start.saturating_sub(1);
        if nominal < cursor {
            return Err(describe_hunks_out_of_order(
                orig_lines,
                hunk,
                Some(nominal),
                cursor,
                hunk_idx,
                hunk_total,
            ));
        }
        return if nominal <= orig_lines.len() {
            Ok(Some(nominal))
        } else {
            Ok(None)
        };
    }

    let nominal = hunk.old_start.saturating_sub(1);
    let nominal_ok = hunk.old_start > 0
        && nominal <= orig_lines.len()
        && nominal >= cursor
        && try_apply_hunk_at(orig_lines, hunk, nominal, mode, ContextPolicy::Require).is_some();
    if nominal_ok {
        return Ok(Some(nominal));
    }

    // When the nominal position does not match, first check how many places across
    // the whole file can match: multiple matches mean the context anchors are not
    // unique, so report the candidate positions directly instead of falling back
    // to fuzzy context and producing a confusing context mismatch.
    let positions = all_hunk_match_positions(orig_lines, hunk, mode);
    let forward: Vec<usize> = positions.iter().copied().filter(|&p| p >= cursor).collect();
    if forward.len() > 1 {
        if let Some(pos) = disambiguate_by_declared_line(&forward, hunk) {
            return Ok(Some(pos));
        }
        return Err(describe_ambiguous_hunk(
            orig_lines, &forward, hunk_idx, hunk_total,
        ));
    }
    // forward is already filtered to p >= cursor, so "hunks out of order" cannot
    // happen here. Previously falling back to find_hunk_offset (a ±50 window) would
    // falsely report context mismatch when the unique match fell outside the
    // window; just use forward's unique result directly.
    if let Some(&offset) = forward.first() {
        Ok(Some(offset))
    } else if let Some(&earliest) = positions.first() {
        // All matches are before cursor — the hunk is out of order. positions is
        // already sorted ascending, so take the earliest match as the diagnostic
        // anchor.
        Err(describe_hunks_out_of_order(
            orig_lines,
            hunk,
            Some(earliest),
            cursor,
            hunk_idx,
            hunk_total,
        ))
    } else {
        Ok(None)
    }
}

pub(crate) fn apply_unified_patch_with_hints(
    original: &str,
    patch: &str,
) -> Result<(String, Vec<String>), String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks =
        parse_unified_hunks(patch).map_err(|err| append_truncated_patch_hint(patch, err))?;
    if !hunks.iter().any(|hunk| {
        hunk.lines
            .iter()
            .any(|line| matches!(line, UnifiedLine::Remove(_) | UnifiedLine::Add(_)))
    }) {
        return Err("[NO_CHANGES] patch contains no additions or removals".to_string());
    }
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let active_hunks: Vec<&UnifiedHunk> = hunks
        .iter()
        .filter(|hunk| {
            hunk.lines
                .iter()
                .any(|line| matches!(line, UnifiedLine::Remove(_) | UnifiedLine::Add(_)))
        })
        .collect();
    let hunk_total = active_hunks.len();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for (hunk_idx, hunk) in active_hunks.iter().enumerate() {
        // First try strict matching (only tolerating trailing whitespace). When
        // strict matching cannot locate the hunk anywhere in the file, fall back
        // once to a lenient mode that ignores leading indentation — aligned with
        // `git apply --ignore-whitespace`, resolving context mismatches caused by
        // the model not reproducing markdown/nested-list/code-block indentation
        // exactly.
        let (apply_at, mode, context_policy) = match locate_hunk(
            &orig_lines,
            hunk,
            cursor,
            MatchMode::Strict,
            hunk_idx,
            hunk_total,
        )? {
            Some(at) => (at, MatchMode::Strict, ContextPolicy::Require),
            None => match locate_hunk(
                &orig_lines,
                hunk,
                cursor,
                MatchMode::IgnoreIndent,
                hunk_idx,
                hunk_total,
            )? {
                Some(at) => (at, MatchMode::IgnoreIndent, ContextPolicy::Require),
                None => match locate_hunk_with_fuzzy_context(
                    &orig_lines,
                    hunk,
                    cursor,
                    MatchMode::Strict,
                ) {
                    Ok(Some(candidate)) => (candidate.pos, MatchMode::Strict, ContextPolicy::Fuzz),
                    _ => match locate_hunk_with_fuzzy_context(
                        &orig_lines,
                        hunk,
                        cursor,
                        MatchMode::IgnoreIndent,
                    )? {
                        Some(candidate) => {
                            (candidate.pos, MatchMode::IgnoreIndent, ContextPolicy::Fuzz)
                        }
                        None => {
                            return Err(describe_context_mismatch(
                                &orig_lines,
                                hunk,
                                hunk_idx,
                                hunk_total,
                            ));
                        }
                    },
                },
            },
        };

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let (hunk_out, new_idx) =
            try_apply_hunk_at(&orig_lines, hunk, apply_at, mode, context_policy).ok_or_else(
                || describe_context_mismatch(&orig_lines, hunk, hunk_idx, hunk_total),
            )?;
        out.extend(hunk_out);
        cursor = new_idx;
    }

    out.extend_from_slice(&orig_lines[cursor..]);
    let mut s = out.join("\n");
    if had_trailing_newline {
        s.push('\n');
    }
    let hints = pure_insert_hint(original, &hunks).into_iter().collect();
    Ok((s, hints))
}

pub(crate) fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    apply_unified_patch_with_hints(original, patch).map(|(content, _)| content)
}

/// A pure-insert hunk (only `+` lines, no context/remove lines) undergoes no
/// content verification on a non-empty file; it is located only by the line number
/// declared in `@@`. When one is hit, return a hint reminding the model that if the
/// file changed after being read, it should re-verify the insert position with
/// read_file.
pub(crate) fn pure_insert_hint(original: &str, hunks: &[UnifiedHunk]) -> Option<String> {
    if original.is_empty() {
        return None;
    }
    let any_pure_insert = hunks.iter().any(|hunk| {
        !hunk.lines.is_empty()
            && hunk
                .lines
                .iter()
                .all(|line| matches!(line, UnifiedLine::Add(_)))
    });
    any_pure_insert.then(|| {
        "hunk(s) had no context lines and were located by line number only; if the file changed \
         since you read it, re-read it with read_file to confirm the insertion landed where \
         intended"
            .to_string()
    })
}

/// A lightweight truncation heuristic: when an inline patch is cut off by the
/// context manager, its tail often looks truncated (an unclosed envelope, or it
/// ends with a bare hunk header or a partial `***` marker). On a hit, append the
/// alternative `patch_file` path to the error text so the model does not keep
/// retrying the same truncated patch on a misleading parse error.
pub(crate) fn truncated_patch_hint(patch: &str) -> Option<&'static str> {
    let trimmed = patch.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let lines: Vec<&str> = trimmed.lines().collect();
    let last = lines.last().unwrap().trim_end();
    // 1) Starts with an envelope marker but is never closed
    let starts_envelope = lines
        .iter()
        .any(|line| line.trim_start().starts_with("*** Begin Patch"));
    if starts_envelope
        && !lines.iter().any(|line| {
            let t = line.trim();
            t == "*** End Patch" || t == "*** End of File"
        })
    {
        return Some("the patch starts with `*** Begin Patch` but has no closing `*** End Patch`");
    }
    // 2) Ends with a partial `***` section/closing marker (e.g. `*** End Patc`,
    //    `*** Update File: /pa`)
    let last_trimmed = last.trim();
    if last.starts_with("*** ")
        && last_trimmed != "*** End Patch"
        && last_trimmed != "*** End of File"
    {
        return Some("the patch ends with a partial `***` section marker");
    }
    // 3) Ends with a hunk header (no content lines after it): a hunk needs at least
    //    one body line that is ` ` / `-` / `+`.
    if last.trim_start().starts_with("@@") {
        return Some("the patch ends with a `@@` hunk header and no body lines after it");
    }
    None
}

/// When a parse failure hits the truncation heuristic, append the actionable
/// alternative path to the end of the error text.
pub(crate) fn append_truncated_patch_hint(patch: &str, err: String) -> String {
    match truncated_patch_hint(patch) {
        Some(hint) => format!(
            "{err}\n\nnote: {hint}. The patch was likely cut off by the context manager \
             mid-flight. Retry with a smaller inline patch, or write the patch to a temp file \
             with write_file(temp=true) and pass its path as `patch_file`."
        ),
        None => err,
    }
}

pub(crate) fn emit_stream_line(on_chunk: &mut ToolStreamWriter<'_>, line: &str) {
    let mut rendered = line.to_string();
    rendered.push('\n');
    on_chunk(rendered.as_bytes());
}

/// Strips code fences (```...``` or ~~~...~~~) the model often wraps around a
/// patch. Strips only when the whole patch is wrapped in one pair (opening fence on
/// the first line, bare closing fence on the last); fences inside the patch body
/// are left untouched to avoid damaging real patch content.
pub(crate) fn strip_code_fence(patch: &str) -> String {
    let lines: Vec<&str> = patch.lines().collect();
    if lines.len() < 3 {
        return patch.to_string();
    }
    let first = lines.first().unwrap().trim();
    let is_open_fence = first.starts_with("```") || first.starts_with("~~~");
    if !is_open_fence {
        return patch.to_string();
    }
    // Walk backwards from the end to find the first non-empty line as the closing
    // fence candidate — the model often emits extra blank lines after the closing
    // fence.
    let last_nonempty = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .unwrap_or(0);
    let last = lines.get(last_nonempty).unwrap().trim();
    let is_close_fence = last == "```" || last == "~~~";
    if !is_close_fence || last_nonempty < 2 {
        return patch.to_string();
    }
    // Strip the first/last fence lines, keep the middle content, and trim excess
    // leading/trailing whitespace.
    lines[1..last_nonempty].join("\n").trim().to_string()
}

pub(crate) fn diff_stats_for_write(write: &PreparedPatchWrite) -> (usize, usize, usize) {
    // (added, removed, total_lines_after)
    match &write.action {
        PreparedPatchAction::Write(next) => {
            let after_lines = next.lines().count();
            match &write.before {
                Some(before) => {
                    // Count additions/removals by comparing line by line
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut before_set: Vec<u64> = before
                        .lines()
                        .map(|l| {
                            let mut h = DefaultHasher::new();
                            l.hash(&mut h);
                            h.finish()
                        })
                        .collect();
                    let mut added = 0usize;
                    for l in next.lines() {
                        let mut h = DefaultHasher::new();
                        l.hash(&mut h);
                        let hash = h.finish();
                        if let Some(pos) = before_set.iter().position(|&x| x == hash) {
                            before_set.remove(pos);
                        } else {
                            added += 1;
                        }
                    }
                    let removed = before_set.len();
                    (added, removed, after_lines)
                }
                None => (after_lines, 0, after_lines),
            }
        }
        PreparedPatchAction::Delete => {
            (0, write.before.as_ref().map_or(0, |b| b.lines().count()), 0)
        }
    }
}

pub(crate) fn format_patch_success(writes: &[PreparedPatchWrite]) -> String {
    let mut message = if writes.len() == 1 {
        let (added, removed, total) = diff_stats_for_write(&writes[0]);
        format!(
            "Successfully patched {}; +{added} -{removed} ({total} lines)",
            writes[0].path.display()
        )
    } else {
        let mut m = format!("Successfully patched {} files:", writes.len());
        for write in writes {
            let (added, removed, total) = diff_stats_for_write(write);
            m.push_str(&format!(
                "\n- {}; +{added} -{removed} ({total} lines)",
                write.path.display()
            ));
        }
        m
    };
    for write in writes {
        for hint in &write.hints {
            message.push_str(&format!("\nnote: {hint}"));
        }
    }
    message
}

pub(crate) fn format_legacy_patch_dry_run(writes: &[PreparedPatchWrite]) -> String {
    if writes.len() == 1 {
        let (added, removed, total) = diff_stats_for_write(&writes[0]);
        return format!(
            "Dry run succeeded; no files changed: {}; +{added} -{removed} ({total} lines after)",
            writes[0].path.display()
        );
    }
    let mut message = format!(
        "Dry run succeeded for {} files; no files changed:",
        writes.len()
    );
    for write in writes {
        let (added, removed, total) = diff_stats_for_write(write);
        message.push_str(&format!(
            "\n- {}; +{added} -{removed} ({total} lines after)",
            write.path.display()
        ));
    }
    message
}

/// Kept only for compatibility with old sessions/history replay. `dry_run` is no
/// longer exposed to the model, but the old parameter cannot be silently ignored:
/// otherwise an old `dry_run: true` call would silently change from "validate only"
/// into a real write.
pub(crate) fn legacy_dry_run_arg(args: &Value) -> Result<bool, String> {
    match args.get("dry_run") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(value) => Err(format!(
            "[INVALID_ARGUMENT] `dry_run` must be a boolean, got {}",
            value_type_name(value)
        )),
    }
}

pub(crate) fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(crate) fn validate_delete_target(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|err| {
        format!(
            "Delete File target does not exist or cannot be inspected: {} ({err})",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "Delete File refuses symbolic links: {}. Delete the link explicitly outside apply_patch.",
            path.display()
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(format!(
            "Delete File only supports regular files, not directories or special files: {}",
            path.display()
        ));
    }
    Ok(())
}

pub(crate) fn prepare_patch_action_from_content(
    path: &Path,
    current: Option<&str>,
    envelope: &PatchEnvelope,
) -> Result<(PreparedPatchAction, Vec<String>), String> {
    match envelope.op {
        PatchEnvelopeOp::Delete => {
            if !envelope.body_lines.is_empty() {
                return Err("Delete File sections must not contain patch content".to_string());
            }
            if current.is_none() {
                return Err(format!(
                    "Delete File target does not exist: {}",
                    path.display()
                ));
            }
            if !path.exists() {
                return Err(format!(
                    "Delete File cannot delete {} because it only exists in earlier sections of the same patch. Remove the Add/Delete no-op pair instead.",
                    path.display()
                ));
            }
            validate_delete_target(path)?;
            Ok((PreparedPatchAction::Delete, Vec::new()))
        }
        PatchEnvelopeOp::ReplaceInLine => {
            let original = current.ok_or_else(|| {
                format!(
                    "Replace in line: target file does not exist: {}",
                    path.display()
                )
            })?;
            Ok((
                PreparedPatchAction::Write(apply_inline_replace(original, envelope)?),
                Vec::new(),
            ))
        }
        PatchEnvelopeOp::Update => {
            let original = current.ok_or_else(|| {
                format!(
                    "Update File patch targets a missing file: {}. Use Add File to create a new file, or correct the target path before retrying.",
                    path.display()
                )
            })?;
            let normalized_patch = normalize_patch_envelope_body(envelope)?;
            let (next, hints) = apply_unified_patch_with_hints(original, &normalized_patch)?;
            Ok((PreparedPatchAction::Write(next), hints))
        }
        PatchEnvelopeOp::Add => {
            if current.is_some() {
                return Err(
                    "Add File patch targets an existing file. Use Update File or write_file instead."
                        .to_string(),
                );
            }
            let normalized_patch = normalize_patch_envelope_body(envelope)?;
            let (next, hints) = apply_unified_patch_with_hints("", &normalized_patch)?;
            Ok((PreparedPatchAction::Write(next), hints))
        }
    }
}

pub(crate) fn prepare_patch_write(
    path: &Path,
    store: &FileStore,
    envelope: &PatchEnvelope,
) -> Result<PreparedPatchWrite, String> {
    let before = if path.exists() {
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        None
    };
    let (action, hints) = if matches!(envelope.op, PatchEnvelopeOp::Update | PatchEnvelopeOp::Add) {
        // On the first encounter of a file, keep the disk-existence check so the
        // single-section behavior is unchanged; repeated sections for the same file
        // are handled by prepare_patch_action_from_content against in-memory state.
        let normalized_patch = normalize_patch_envelope(path, envelope)?;
        let original = before.as_deref().unwrap_or_default();
        apply_unified_patch_with_hints(original, &normalized_patch)
            .map(|(next, hints)| (PreparedPatchAction::Write(next), hints))?
    } else {
        prepare_patch_action_from_content(path, before.as_deref(), envelope)?
    };
    Ok(PreparedPatchWrite {
        path: path.to_path_buf(),
        before,
        action,
        hints,
    })
}

/// Prepares a write from an already-split single-file unified diff section (used
/// for the multi-file unified diff path). Unlike prepare_patch_write, the section
/// carries its own `--- `/`+++ ` file headers and `@@` hunks, so no envelope
/// normalization is applied — it is handed straight to apply_unified_patch
/// (parse_unified_hunks skips non-`@@` lines), reusing the tolerant match
/// semantics of unified diffs as-is.
pub(crate) fn prepare_patch_write_from_section(
    path: &Path,
    store: &FileStore,
    section: &str,
) -> Result<PreparedPatchWrite, String> {
    let before = if path.exists() {
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        None
    };
    let original = before.as_deref().unwrap_or_default();
    let (next, hints) = apply_unified_patch_with_hints(original, section)?;
    let action = PreparedPatchAction::Write(next);
    Ok(PreparedPatchWrite {
        path: path.to_path_buf(),
        before,
        action,
        hints,
    })
}

/// Rejects writes whose final content is identical to the original file, so the
/// tool never reports success without actually changing anything.
pub(crate) fn ensure_patch_writes_change(writes: &[PreparedPatchWrite]) -> Result<(), String> {
    for write in writes {
        let PreparedPatchAction::Write(next) = &write.action else {
            continue;
        };
        if write.before.as_deref() == Some(next.as_str()) {
            return Err(format!(
                "[NO_CHANGES] patch would leave {} unchanged. Remove the no-op hunk or re-read the file and rebuild the patch.",
                write.path.display()
            ));
        }
    }
    Ok(())
}

pub(crate) fn verify_patch_write_is_current(write: &PreparedPatchWrite) -> Result<(), String> {
    let current = if write.path.exists() {
        Some(
            FileStore::new(write.path.clone())
                .read_to_string()
                .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    if current == write.before {
        return Ok(());
    }
    Err(format!(
        "[FILE_CHANGED] {} changed since this patch was prepared. Re-read it and rebuild the patch before retrying.",
        write.path.display()
    ))
}

pub(crate) fn apply_prepared_patch_write(write: &PreparedPatchWrite) -> Result<(), String> {
    match &write.action {
        PreparedPatchAction::Write(next) => {
            let store = FileStore::new(write.path.clone());
            // `commit_patch_writes` verifies every write is still current up front (early batch
            // abort with a friendly error before anything is applied). Passing `write.before` as
            // the snapshot hint turns the actual write into a compare-and-swap
            // (`write_all_with_before_hint` → kernel `vfs_write_if_unchanged`): the check and the
            // write share one kernel-lock hold, so other writers using that kernel cannot
            // interleave. External processes do not participate in that lock.
            //
            // A `None` `before` means Add File: the precondition is that the target does not
            // exist, so the create-only CAS (`write_new_file_if_absent` → kernel
            // `vfs_write_if_unchanged` with expected=None) enforces absence instead of writing
            // unconditionally — a file created concurrently since preparation is never
            // overwritten.
            match write.before.as_deref() {
                Some(before) => store.write_all_with_before_hint(next, Some(before)),
                None => store.write_new_file_if_absent(next),
            }
            .map_err(|err| err.to_string())
        }
        PreparedPatchAction::Delete => {
            // Delete targets always carry a before snapshot (see DeleteFile validation).
            // Conditional removal detects changed content; it is serialized with writers
            // using the same kernel, not arbitrary external processes.
            let expected = write.before.as_deref().ok_or_else(|| {
                format!(
                    "internal error: delete of {} has no before snapshot",
                    write.path.display()
                )
            })?;
            FileStore::new(write.path.clone())
                .remove_file_if_unchanged(expected)
                .map_err(|err| err.to_string())?;
            let _ = crate::ai::tools::storage::mutation_log::record(
                &write.path,
                "delete",
                write.before.as_deref(),
                None,
            );
            Ok(())
        }
    }
}

pub(crate) fn restore_prepared_patch_write(write: &PreparedPatchWrite) -> Result<(), String> {
    let store = FileStore::new(write.path.clone());
    // A failed operation may have left the original state intact. Do not write again in that
    // case. Otherwise, only undo the exact postimage this batch intended to install. Partial
    // writes and intervening edits are ambiguous and must be left intact with an error rather
    // than overwritten or deleted in the name of rollback.
    let already_restored = match &write.before {
        Some(before) => store.read_to_string().is_ok_and(|current| current == *before),
        None => matches!(
            fs::symlink_metadata(&write.path),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound
        ),
    };
    if already_restored {
        return Ok(());
    }
    let result = match (&write.before, &write.action) {
        (Some(before), PreparedPatchAction::Write(after)) => {
            store.write_all_with_before_hint(before, Some(after))
        }
        (Some(before), PreparedPatchAction::Delete) => store.write_new_file_if_absent(before),
        (None, PreparedPatchAction::Write(after)) => store.remove_file_if_unchanged(after).map(|()| {
            crate::ai::tools::storage::mutation_log::record(
                &write.path,
                "delete",
                Some(after),
                None,
            );
        }),
        (None, PreparedPatchAction::Delete) => {
            return Err(format!(
                "internal error: rollback of {} has no before snapshot",
                write.path.display()
            ));
        }
    };
    result.map_err(|err| format!("failed to restore {} during rollback: {err}", write.path.display()))
}

pub(crate) fn commit_patch_writes(writes: &[PreparedPatchWrite]) -> Result<(), String> {
    commit_patch_writes_with(writes, apply_prepared_patch_write)
}

pub(crate) fn commit_patch_writes_with(
    writes: &[PreparedPatchWrite],
    mut apply: impl FnMut(&PreparedPatchWrite) -> Result<(), String>,
) -> Result<(), String> {
    for write in writes {
        verify_patch_write_is_current(write)?;
    }

    for (idx, write) in writes.iter().enumerate() {
        if let Err(write_err) = apply(write) {
            // A [FILE_CHANGED] error means the file was concurrently modified since the
            // up-front verification and our write never happened: rolling it back to the
            // prepared snapshot would clobber that concurrent change, so the failed write
            // itself must not be restored (only earlier, already-applied writes are). Any
            // other failure may have left a changed file, so the current write is considered
            // too. Rollback only restores an exact postimage (or accepts an unchanged preimage);
            // partial/unknown content is retained and reported as an incomplete rollback.
            let restore_end = if write_err.contains("[FILE_CHANGED]") {
                idx
            } else {
                idx + 1
            };
            let restoration_errors: Vec<_> = writes[..restore_end]
                .iter()
                .rev()
                .filter_map(|written| restore_prepared_patch_write(written).err())
                .collect();
            if restoration_errors.is_empty() {
                if restore_end == 0 {
                    // Nothing was applied before the failure, so there is nothing to restore
                    // (the concurrent change is left intact).
                    return Err(format!(
                        "failed to apply {}: {write_err}",
                        write.path.display()
                    ));
                }
                return Err(format!(
                    "failed to apply {}: {write_err}; all affected files were restored",
                    write.path.display()
                ));
            }
            return Err(format!(
                "failed to apply {}: {write_err}; rollback was incomplete: {}",
                write.path.display(),
                restoration_errors.join("; ")
            ));
        }
    }

    Ok(())
}
