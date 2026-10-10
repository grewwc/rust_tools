use crate::ai::tools::common::ToolHistoryPolicy;
use crate::ai::tools::common::ToolHistoryPolicyRegistration;
use crate::ai::tools::common::ToolLossyCompressPolicy;
use crate::ai::tools::common::ToolPrunePolicy;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::ToolStreamingRegistration;
use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::storage::file_store::FileStore;
use crate::ai::tools::storage::temp_registry;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::path::PathBuf;

mod types;
pub(crate) use types::*;
mod parse;
pub(crate) use parse::*;
mod replace_utils;
pub(crate) use replace_utils::*;
mod match_utils;
pub(crate) use match_utils::*;
mod locate;
pub(crate) use locate::*;
mod diagnostics;
pub(crate) use diagnostics::*;
mod engine;
pub(crate) use engine::*;
inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "",

        execute: execute_apply_patch,
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "apply_patch",
    execute_streaming: execute_apply_patch_streaming,
});

// The apply_patch result is the only precise evidence of "which file did I just modify" for the
// current turn, and failure diagnostics echo the entire current file text (so the model can
// rebuild the patch without re-reading). Once lossily compressed or truncated, the model cannot
// see whether the patch landed, and will repeatedly suspect the tool was never called and retry
// in place — hence lossy compression is forbidden, and apply_patch consumes high-precision inline
// budget to enter the current-turn protected set. Consistent with execute_command semantics:
// stale patch results can still be pruned by the model explicitly marking them outdated, so the
// context won't grow monotonically.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "apply_patch",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

fn optional_patch_source_arg(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        // Some providers / tool bridges materialize optional schema fields as null
        // or empty strings. For mutually exclusive source arguments, both mean "not
        // provided" and must not be reported as an error when the other source is
        // valid.
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(format!(
            "{key} parameter has wrong type ({}): expected a string{}.",
            value_type_name(other),
            if key == "patch_file" {
                " path to a file containing the patch text"
            } else {
                " containing the patch text"
            }
        )),
    }
}

fn execute_apply_patch_impl(args: &Value, mut emit: impl FnMut(&str)) -> Result<String, String> {
    let legacy_dry_run = legacy_dry_run_arg(args)?;
    let inline_patch = optional_patch_source_arg(args, "patch")?;
    let patch_file = optional_patch_source_arg(args, "patch_file")?;
    let (raw_patch, from_file) = match (inline_patch, patch_file) {
        (Some(_), Some(_)) => {
            return Err(
                "pass either `patch` (inline edit text) or `patch_file` (path to a file with the \
                 patch text), not both"
                    .to_string(),
            );
        }
        (Some(p), None) => (p, false),
        (None, Some(pf)) => {
            let store = FileStore::new(PathBuf::from(&pf));
            let resolved = store.path();
            // patch_file contract: it must be a session temp file (written with
            // write_file(temp=true) and registered in the temp registry) or a file
            // under effective_cwd; anything else is rejected outright so the model
            // cannot point at arbitrary system files and get confused.
            let in_cwd = crate::ai::driver::runtime_ctx::effective_cwd()
                .map(|cwd| resolved.starts_with(&cwd))
                .unwrap_or(false);
            if !temp_registry::is_registered(&resolved.to_string_lossy()) && !in_cwd {
                return Err(format!(
                    "patch_file '{}' is not an allowed patch source: write the patch to a temp \
                     file with write_file(temp=true) and pass its path here, or use a file under \
                     the current working directory.",
                    resolved.display()
                ));
            }
            emit(&format!("reading patch from {}", resolved.display()));
            let content = store.read_to_string().map_err(|err| {
                format!(
                    "patch_file '{}' could not be read: {err}",
                    resolved.display()
                )
            })?;
            (content, true)
        }
        (None, None) => {
            return Err(
                "patch parameter is missing or empty: provide `patch` (inline edit text) or \
                 `patch_file` (path to a temp file containing the patch text, written with \
                 write_file(temp=true)). If you sent a large patch, it may have been truncated \
                 by the context manager before reaching this tool - retry with a smaller patch \
                 (split into multiple apply_patch calls), or write the patch to a temp file and \
                 pass `patch_file` instead. For a small change, `*** Replace in line:` is the \
                 most reliable format."
                    .to_string(),
            );
        }
    };
    let patch = strip_code_fence(&raw_patch);
    // Truncation heuristic: when an inline patch ends with an unclosed envelope, a
    // partial `***` marker, or a bare hunk header, it was likely cut off by the
    // context manager. Emit a hint up front regardless of whether the patch then
    // succeeds or errors, so the model does not keep retrying truncated text as if
    // it were its own syntax error.
    if let Some(hint) = truncated_patch_hint(&patch) {
        emit(&format!(
            "warning: {hint}; the patch may have been truncated by the context manager \
             mid-flight. If it applies incompletely, retry with a smaller inline patch or pass \
             the patch via `patch_file` (write_file(temp=true))."
        ));
    }
    if from_file {
        // patch_file is meant to carry large patches (inline patches get truncated
        // by the context manager), so it is exempt from the 8K inline limit; only a
        // loose safety cap is set to stop the model from misreading oversized files.
        const MAX_PATCH_FILE_CHARS: usize = 64_000;
        if patch.chars().count() > MAX_PATCH_FILE_CHARS {
            return Err(format!(
                "patch_file too large ({} chars; limit 64000). Split the patch into multiple \
                 smaller patch files and apply them sequentially, or use write_file for a \
                 full-file rewrite.",
                patch.chars().count()
            ));
        }
    } else {
        // An oversized inline patch is very likely to be truncated by the context
        // manager (appearing as a missing/partial patch or a misleading context
        // mismatch). Erroring out directly with a splitting path beats feeding
        // truncated text to the parser.
        const MAX_INLINE_PATCH_CHARS: usize = 8_000;
        if patch.chars().count() > MAX_INLINE_PATCH_CHARS {
            return Err(format!(
                "patch too large ({} chars; limit 8000). Large patches are likely to be truncated \
                 by the context manager mid-flight. Split the edit into multiple smaller \
                 apply_patch calls (a few hunks each), write the patch to a temp file with \
                 write_file(temp=true) and pass its path as `patch_file`, or use write_file for a \
                 full-file rewrite.",
                patch.chars().count()
            ));
        }
    }
    emit("parsing patch envelope");
    let initial_file_path = optional_file_path_arg(args);
    if let Some(envelopes) =
        parse_patch_envelopes(&patch).map_err(|err| append_truncated_patch_hint(&patch, err))?
    {
        // Each section inside an envelope (single-file or multi-file) already declares its own target path, so `file_path` is
        // redundant. Models often redundantly pass `file_path` for multi-file envelopes; instead of hard-erroring and wasting a round,
        // silently ignore it and use the envelope path (the envelope path is the authoritative source).
        if initial_file_path.is_some() {
            emit("note: ignoring redundant file_path arg; using paths from Begin Patch envelope");
        }
        emit(&format!("parsed {} patch section(s)", envelopes.len()));
        let mut writes: Vec<PreparedPatchWrite> = Vec::with_capacity(envelopes.len());
        let mut write_indexes: FxHashMap<PathBuf, usize> = FxHashMap::default();
        for (idx, envelope) in envelopes.iter().enumerate() {
            let target_arg = envelope.target_path.as_str();
            let store = FileStore::new(PathBuf::from(target_arg));
            emit(&format!(
                "target [{}/{}]: {}",
                idx + 1,
                envelopes.len(),
                store.path().display()
            ));
            emit("validating write access");
            store
                .validate_write_access()
                .map_err(|err| err.to_string())?;
            let path = store.path().to_path_buf();
            ensure_patch_target_matches(&path, &envelope.target_path)?;
            if envelope.op == PatchEnvelopeOp::ReplaceInLine {
                emit("applying inline replacement");
            } else if envelope.op == PatchEnvelopeOp::Delete {
                emit("preparing file deletion");
            } else {
                let hunk_count = envelope
                    .body_lines
                    .iter()
                    .filter(|line| line.starts_with("@@"))
                    .count()
                    .max(1);
                emit(&format!("applying {hunk_count} hunk(s)"));
            }
            if let Some(&write_idx) = write_indexes.get(&path) {
                emit("applying after previous section for same file");
                let (action, hints) = {
                    let current = match &writes[write_idx].action {
                        PreparedPatchAction::Write(next) => Some(next.as_str()),
                        PreparedPatchAction::Delete => None,
                    };
                    prepare_patch_action_from_content(&path, current, envelope)
                }
                .map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        envelopes.len(),
                        path.display()
                    )
                })?;
                writes[write_idx].action = action;
                writes[write_idx].hints.extend(hints);
            } else {
                let write = prepare_patch_write(&path, &store, envelope).map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        envelopes.len(),
                        path.display()
                    )
                })?;
                write_indexes.insert(path.clone(), writes.len());
                writes.push(write);
            }
        }
        ensure_patch_writes_change(&writes)?;
        if legacy_dry_run {
            let success = format_legacy_patch_dry_run(&writes);
            emit(&success);
            return Ok(success);
        }
        for write in &writes {
            match &write.action {
                PreparedPatchAction::Write(next) => {
                    emit(&format!("writing {} byte(s)", next.len()))
                }
                PreparedPatchAction::Delete => emit("deleting file"),
            }
        }
        commit_patch_writes(&writes)?;
        let success = format_patch_success(&writes);
        emit(&success);
        return Ok(success);
    }

    let header_target = parse_unified_diff_header_target(&patch);
    // Multi-file unified diff (git diff output or hand-written by the model): split by file and prepare each one,
    // then commit only after all succeed — matching the batch semantics of the Begin Patch envelope (atomic, same-path stacking).
    // Note: multiple sections for the same file (two `diff --git a/x b/x`) dedupe to paths.len()==1 after header parsing,
    // so use the split section count (>1) or the count of distinct parsed paths (>1) as the criterion — both take the split path.
    let split_sections = split_unified_diff_by_file(&patch);
    let multi_file_sections = match &split_sections {
        Ok(sections) if sections.len() > 1 || header_target.paths.len() > 1 => Some(sections),
        Ok(_) => None,
        Err(err) if header_target.paths.len() > 1 => {
            return Err(format!("multi-file unified diff could not be split: {err}"));
        }
        Err(_) => None, // 无文件头（裸 hunks 靠 file_path 定位）：单文件路径
    };
    if let Some(sections) = multi_file_sections {
        emit(&format!(
            "parsing multi-file unified diff ({} file(s))",
            sections.len()
        ));
        let mut writes: Vec<PreparedPatchWrite> = Vec::with_capacity(sections.len());
        let mut write_indexes: FxHashMap<PathBuf, usize> = FxHashMap::default();
        for (idx, (target, section)) in sections.iter().enumerate() {
            if parse_unified_diff_header_target(section).deletes_file {
                return Err(format!(
                    "[section {}/{}] unified diff deletion (`+++ /dev/null`) is not supported \
                     because unified mode writes file content; use a `*** Begin Patch` envelope \
                     with `*** Delete File: {target}` so deletion is explicit",
                    idx + 1,
                    sections.len()
                ));
            }
            let store = FileStore::new(PathBuf::from(target));
            let path = store.path().to_path_buf();
            emit(&format!(
                "target [{}/{}]: {}",
                idx + 1,
                sections.len(),
                path.display()
            ));
            emit("validating write access");
            store
                .validate_write_access()
                .map_err(|err| err.to_string())?;
            if let Some(&write_idx) = write_indexes.get(&path) {
                // Multiple sections for the same file apply in order (same semantics as the envelope branch).
                emit("applying after previous section for same file");
                let (action, hints) = {
                    let current = match &writes[write_idx].action {
                        PreparedPatchAction::Write(next) => Some(next.as_str()),
                        PreparedPatchAction::Delete => None,
                    };
                    let original = current.unwrap_or_default();
                    apply_unified_patch_with_hints(original, section)
                        .map(|(next, hints)| (PreparedPatchAction::Write(next), hints))
                }
                .map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        sections.len(),
                        path.display()
                    )
                })?;
                writes[write_idx].action = action;
                writes[write_idx].hints.extend(hints);
            } else {
                let write =
                    prepare_patch_write_from_section(&path, &store, section).map_err(|err| {
                        format!(
                            "[section {}/{}] failed while preparing patch for {}: {err}",
                            idx + 1,
                            sections.len(),
                            path.display()
                        )
                    })?;
                write_indexes.insert(path.clone(), writes.len());
                writes.push(write);
            }
        }
        ensure_patch_writes_change(&writes)?;
        for write in &writes {
            match &write.action {
                PreparedPatchAction::Write(next) => {
                    emit(&format!("writing {} byte(s)", next.len()))
                }
                PreparedPatchAction::Delete => emit("deleting file"),
            }
        }
        commit_patch_writes(&writes)?;
        let success = format_patch_success(&writes);
        emit(&success);
        return Ok(success);
    }
    if header_target.deletes_file {
        return Err(
            "unified diff deletion (`+++ /dev/null`) is not supported because unified mode writes file content; use a `*** Begin Patch` envelope with `*** Delete File: <path>` so deletion is explicit"
                .to_string(),
        );
    }
    let inferred_file_path = header_target.paths.first().cloned();
    let file_path = match initial_file_path {
        Some(path) => path.to_string(),
        // When file_path is missing, fall back to parsing the git-style diff headers (`--- a/`/`+++ b/`/`diff --git`)
        // for the paths they carry: a valid unified diff written by the model must not be rejected as "missing file_path".
        None => inferred_file_path.clone().ok_or(
            "missing file_path: provide `file_path` (or `path`) arg, wrap the patch in a \
             `*** Begin Patch` / `*** Update File: <path>` envelope, or include a git-style \
             diff header (`--- a/<path>` and `+++ b/<path>`) so the target path can be read \
             from the patch itself.",
        )?,
    };
    let store = FileStore::new(PathBuf::from(&file_path));
    if initial_file_path.is_some()
        && let Some(header_path) = inferred_file_path
    {
        let header_store = FileStore::new(PathBuf::from(&header_path));
        if header_store.path() != store.path() {
            return Err(format!(
                "file_path '{}' conflicts with unified diff header target '{}'; use one consistent target",
                store.path().display(),
                header_store.path().display()
            ));
        }
    }
    emit(&format!("target: {}", store.path().display()));
    emit("validating write access");
    store
        .validate_write_access()
        .map_err(|err| err.to_string())?;
    let path = store.path().to_path_buf();
    let hunk_count = patch
        .lines()
        .filter(|line| line.starts_with("@@"))
        .count()
        .max(1);
    emit(&format!("applying {hunk_count} hunk(s)"));
    let before = if path.exists() {
        emit("reading current file");
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        emit("creating new file from patch");
        None
    };
    let (next, hints) =
        apply_unified_patch_with_hints(before.as_deref().unwrap_or_default(), &patch)?;
    let write = PreparedPatchWrite {
        path: path.clone(),
        before,
        action: PreparedPatchAction::Write(next),
        hints,
    };
    ensure_patch_writes_change(std::slice::from_ref(&write))?;
    if legacy_dry_run {
        let success = format_legacy_patch_dry_run(&[write]);
        emit(&success);
        return Ok(success);
    }
    if let PreparedPatchAction::Write(next) = &write.action {
        emit(&format!("writing {} byte(s)", next.len()));
    }
    let success = format_patch_success(std::slice::from_ref(&write));
    commit_patch_writes(&[write])?;
    emit(&success);
    Ok(success)
}

pub(crate) fn execute_apply_patch(args: &Value) -> Result<String, String> {
    execute_apply_patch_impl(args, |_| {})
}

pub(crate) fn execute_apply_patch_streaming(
    args: &Value,
    on_chunk: &mut ToolStreamWriter<'_>,
) -> Result<String, String> {
    execute_apply_patch_impl(args, |line| emit_stream_line(on_chunk, line))
}

#[cfg(test)]
#[path = "../patch_tools_tests.rs"]
mod tests;
