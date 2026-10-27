//! Session-move path remap for asset pointers stored in persisted messages.
//!
//! Fork/branch/import copy the assets directory wholesale, so every absolute
//! asset path recorded in message text must be relocated from the source
//! session's assets directory to the target one. Only checkpoint markers were
//! remapped historically; archive pointers kept pointing at the source session
//! (cross-session reads while the source lives, dangling paths after it is
//! deleted).
//!
//! Safety model (mirrors the original checkpoint remap):
//! - Fork (source known): exact `source_assets` prefix replacement. The prefix
//!   match alone proves the pointer belongs to the source session.
//! - Import (source unknown): the recorded path must be absolute and contain a
//!   known asset-internal anchor; the tail from that anchor is joined onto the
//!   target and rewritten only when it exists there.
//! - Relative paths are left untouched: they resolve against the current
//!   session and survive moves on their own.
//! - Only writer-exact labels are recognized; project source pointers such as
//!   `original_file_path` are never rewritten.
//! - Rewritten tails must stay inside the target assets directory; a tail that
//!   escapes via `..` is left for manual recovery instead of being mirrored.

use std::path::{Component, Path, PathBuf};

use super::folded_prune;
use super::persisted::{
    ARCHIVE_DIRECTORY_PATH_PREFIX, ARCHIVE_FILE_PATH_PREFIX, PRESERVED_CONTENT_STUB_PREFIX,
};

// Legacy writer labels, frozen by persisted sessions. The canonical labels
// live in `persisted.rs`; the legacy consts there are private, so the exact
// strings are repeated here (see `LEGACY_ARCHIVE_*_PREFIX`).
const LEGACY_FILE_PREFIX: &str = "归档文件: ";
const LEGACY_DIR_PREFIX: &str = "归档目录: ";
// Writer: `folded_prune::offload` stub line `- source_projection_archive: <path>`.
const SOURCE_PROJECTION_ARCHIVE_PREFIX: &str = "source_projection_archive: ";
// Writer: `incremental::plan_incremental_summary` record
// `{"source":{"archive_file_path":"..."}}`.
const JSON_ARCHIVE_FILE_KEY: &str = "\"archive_file_path\"";
// Legacy `[[PRESERVED_CONTENT_STUB_V1]]{"kind":..,"file_path":..}` stubs are
// read-only history, but model-facing normalization rebuilds them into live
// current-format pointers reusing the same path verbatim, so their `file_path`
// must be remapped too. Matching is scoped to the stub prefix: a bare
// `"file_path"` label elsewhere is too generic to rewrite safely.

/// Asset-internal anchors accepted when the source path is unknown (import, or
/// a foreign path surviving in a forked session). Each is a directory (or the
/// `overflow-history.md` file) created by this codebase under the session
/// assets directory.
const IMPORT_ANCHORS: &[&str] = &[
    "context-checkpoints",
    "summary-sources",
    "folded-tool-groups",
    "internal-note-overflow",
    "tool-overflow-compressed",
    "user-overflow-preserved",
    "image-overflow-preserved",
    "preserved_user_overflow",
    "pruned-folds",
    "overflow-history.md",
];

/// Line-form labels whose path value runs to the end of the line.
const LINE_PREFIXES: &[&str] = &[
    ARCHIVE_FILE_PATH_PREFIX,
    ARCHIVE_DIRECTORY_PATH_PREFIX,
    SOURCE_PROJECTION_ARCHIVE_PREFIX,
    LEGACY_FILE_PREFIX,
    LEGACY_DIR_PREFIX,
];

/// Remap one recorded path to the target session's assets directory.
/// Returns `None` when the path must be left untouched.
pub(super) fn remap_asset_path(
    recorded: &Path,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> Option<PathBuf> {
    if recorded.is_relative() {
        return None;
    }
    // Exact fork rule: the source-prefix match proves ownership.
    if let Some(source) = source_assets
        && let Ok(tail) = recorded.strip_prefix(source)
        && let Some(safe) = checked_relative_tail(tail)
    {
        return Some(target_assets.join(safe));
    }
    // Anchor rule: import, or a foreign path surviving in a forked session.
    let tail = anchored_tail(recorded)?;
    let candidate = target_assets.join(&tail);
    // Existence gate: never guess a target that is not there.
    if candidate.is_file() || candidate.is_dir() {
        Some(candidate)
    } else {
        None
    }
}

/// Reject empty tails and tails escaping the target via `..`.
fn checked_relative_tail(tail: &Path) -> Option<PathBuf> {
    if tail.as_os_str().is_empty() {
        return Some(PathBuf::new());
    }
    if tail.components().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }
    Some(tail.to_path_buf())
}

/// Take the path tail starting at the last known asset-internal anchor.
fn anchored_tail(recorded: &Path) -> Option<PathBuf> {
    let parts: Vec<String> = recorded
        .components()
        .filter_map(|c| match c {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let pos = parts
        .iter()
        .rposition(|p| IMPORT_ANCHORS.contains(&p.as_str()))?;
    let mut tail = PathBuf::new();
    for part in &parts[pos..] {
        tail.push(part);
    }
    checked_relative_tail(&tail)
}

/// Remap every recognized asset pointer in `text`.
/// Returns the rewritten text, or `None` when nothing changed.
fn remap_text_pointers(
    text: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> Option<String> {
    if let Some(remapped) = remap_legacy_content_stub(text, source_assets, target_assets) {
        return Some(remapped);
    }
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    for line in text.split_inclusive('\n') {
        let (body, newline) = match line.strip_suffix('\n') {
            Some(body) => (body, "\n"),
            None => (line, ""),
        };
        let rewritten = remap_line_pointers(body, source_assets, target_assets);
        if rewritten != body {
            changed = true;
        }
        out.push_str(&rewritten);
        out.push_str(newline);
    }
    changed.then_some(out)
}

/// Remap the `file_path` inside one legacy
/// `[[PRESERVED_CONTENT_STUB_V1]]{"kind":..,"file_path":..}` stub. The whole
/// text must be the stub: the reader (`parse_preserved_message_stub`) only
/// accepts the prefix form, and anything else is left for the line handlers.
fn remap_legacy_content_stub(
    text: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> Option<String> {
    let payload = text
        .trim_start()
        .strip_prefix(PRESERVED_CONTENT_STUB_PREFIX)?;
    let mut value: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    let file_path = value.get("file_path")?.as_str()?.to_string();
    // Same acceptance rule as the reader: only user/image stubs carry paths.
    let kind = value.get("kind")?.as_str()?;
    if (kind != "user" && kind != "image") || file_path.is_empty() {
        return None;
    }
    let remapped = remap_asset_path(Path::new(&file_path), source_assets, target_assets)?;
    value["file_path"] = serde_json::Value::String(remapped.to_string_lossy().into_owned());
    let leading_len = text.len() - text.trim_start().len();
    Some(format!(
        "{}{PRESERVED_CONTENT_STUB_PREFIX}{}",
        &text[..leading_len],
        serde_json::to_string(&value).ok()?
    ))
}

/// Remap pointer labels within one line (line form and JSON key form).
fn remap_line_pointers(line: &str, source_assets: Option<&Path>, target_assets: &Path) -> String {
    let mut out = line.to_string();
    for prefix in LINE_PREFIXES {
        out = remap_line_prefix_occurrences(&out, prefix, source_assets, target_assets);
    }
    remap_json_key_occurrences(&out, source_assets, target_assets)
}

/// Rewrite the first `<prefix><path-to-EOL>` occurrence. The path keeps
/// leading whitespace trimmed and trailing whitespace trimmed; interior
/// spaces (e.g. macOS user names) are preserved. The value runs to end of
/// line by writer contract, so no second occurrence can follow on the line.
fn remap_line_prefix_occurrences(
    line: &str,
    prefix: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    let mut changed = false;
    while let Some(pos) = rest.find(prefix) {
        let (head, tail) = rest.split_at(pos + prefix.len());
        out.push_str(head);
        let raw = tail.trim_end_matches(['\r', ' ', '\t']);
        let leading_ws = tail.len() - tail.trim_start_matches([' ', '\t']).len();
        let path_text = raw.trim_start_matches([' ', '\t']);
        // The value runs to end of line by writer contract.
        if let Some(remapped) = remap_asset_path(Path::new(path_text), source_assets, target_assets)
        {
            out.push_str(&tail[..leading_ws]);
            out.push_str(&remapped.to_string_lossy());
            out.push_str(&tail[raw.len()..]);
            changed = true;
        } else {
            out.push_str(tail);
        }
        rest = "";
    }
    out.push_str(rest);
    if changed { out } else { line.to_string() }
}

/// Rewrite the first remappable `"archive_file_path" : "<path>"` occurrence;
/// candidates that fail to parse or remap are skipped and scanning continues.
/// Writers emit at most one key per line.
fn remap_json_key_occurrences(
    line: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    let mut changed = false;
    while let Some(key_pos) = rest.find(JSON_ARCHIVE_FILE_KEY) {
        let after_key = &rest[key_pos + JSON_ARCHIVE_FILE_KEY.len()..];
        let Some((path, value_start, value_end)) = parse_json_string_value(after_key) else {
            out.push_str(&rest[..key_pos + JSON_ARCHIVE_FILE_KEY.len()]);
            rest = after_key;
            continue;
        };
        let abs_start = key_pos + JSON_ARCHIVE_FILE_KEY.len() + value_start;
        let abs_end = key_pos + JSON_ARCHIVE_FILE_KEY.len() + value_end;
        if let Some(remapped) = remap_asset_path(Path::new(path), source_assets, target_assets) {
            out.push_str(&rest[..abs_start]);
            out.push_str(&remapped.to_string_lossy());
            out.push_str(&rest[abs_end..]);
            changed = true;
            rest = "";
            break;
        }
        out.push_str(&rest[..key_pos + JSON_ARCHIVE_FILE_KEY.len()]);
        rest = after_key;
    }
    out.push_str(rest);
    if changed { out } else { line.to_string() }
}

/// Parse `: "<value>"` at the start of `text` (allowing whitespace around the
/// colon). Returns the value plus its byte offsets within `text`, excluding
/// the quotes. Values with escapes are rejected: asset paths never contain
/// quotes, and a truncation would be worse than leaving the text alone.
fn parse_json_string_value(text: &str) -> Option<(&str, usize, usize)> {
    let after_ws = text.trim_start_matches([' ', '\t']);
    let mut offset = text.len() - after_ws.len();
    let rest = after_ws.strip_prefix(':')?;
    offset += 1;
    let after_ws = rest.trim_start_matches([' ', '\t']);
    offset += rest.len() - after_ws.len();
    let quoted = after_ws.strip_prefix('"')?;
    offset += 1;
    let end = quoted.find('"')?;
    let value = &quoted[..end];
    if value.contains('\\') {
        return None;
    }
    Some((value, offset, offset + end))
}

/// Relocate archive pointers in one persisted message. Checkpoint markers are
/// handled separately by the caller (original sqlite remap, behavior frozen);
/// `content_already_changed` tells provenance whether the body was rewritten
/// before this call, and `pre_move_content` is the exact pre-move body used to
/// verify the old provenance hash before rebinding (never launder an already
/// invalid sidecar into a valid one).
/// Returns `(new_content, new_reasoning)`, each `Some` only when that field
/// changed. Fold provenance is rebound atomically with the content rewrite so
/// the note keeps its pruning authority in the new session.
pub(in crate::ai) fn remap_message_for_session_move(
    content_text: &str,
    reasoning: Option<&str>,
    content_already_changed: bool,
    pre_move_content: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> (Option<String>, Option<String>) {
    let mut content = content_text.to_string();
    let mut content_changed = content_already_changed;
    if let Some(remapped) = remap_text_pointers(&content, source_assets, target_assets) {
        content = remapped;
        content_changed = true;
    }
    let new_reasoning = reasoning.and_then(|sidecar| {
        folded_prune::remap_provenance_for_session_move(
            sidecar,
            pre_move_content,
            &content,
            content_changed,
            source_assets,
            target_assets,
        )
    });
    let new_content = (content_changed && content != content_text).then_some(content);
    (new_content, new_reasoning)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_dir() -> PathBuf {
        std::env::temp_dir().join(format!("ai-remap-test-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn fork_prefix_rewrite_preserves_interior_spaces() {
        let source = PathBuf::from("/tmp/first dir/assets");
        let target = PathBuf::from("/tmp/second/assets");
        let recorded = source.join("summary-sources/a b.jsonl");
        assert_eq!(
            remap_asset_path(&recorded, Some(&source), &target),
            Some(target.join("summary-sources/a b.jsonl"))
        );
    }

    #[test]
    fn relative_paths_are_left_untouched() {
        let target = PathBuf::from("/tmp/second/assets");
        assert_eq!(
            remap_asset_path(Path::new("summary-sources/a.jsonl"), None, &target),
            None
        );
    }

    #[test]
    fn escaping_tails_are_left_for_manual_recovery() {
        let source = PathBuf::from("/tmp/first/assets");
        let target = PathBuf::from("/tmp/second/assets");
        assert_eq!(
            remap_asset_path(&source.join("../escape/x.md"), Some(&source), &target),
            None
        );
    }

    #[test]
    fn import_anchor_rewrite_requires_existence() {
        let target = target_dir();
        let missing = PathBuf::from("/export/machine/assets/summary-sources/gone.jsonl");
        assert_eq!(remap_asset_path(&missing, None, &target), None);
        let dir = target.join("summary-sources");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("here.jsonl"), "x").unwrap();
        let recorded = PathBuf::from("/export/machine/assets/summary-sources/here.jsonl");
        assert_eq!(
            remap_asset_path(&recorded, None, &target),
            Some(target.join("summary-sources/here.jsonl"))
        );
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn line_and_json_forms_all_remap() {
        let source = PathBuf::from("/s/assets");
        let target = PathBuf::from("/t/assets");
        let text = "[archive-pointer-v2]\narchive_file_path: /s/assets/overflow-history.md\narchive_directory_path: /s/assets/internal-note-overflow\n- archive_file_path: /s/assets/folded-tool-groups/a.md\n{\"source\":{\"archive_file_path\":\"/s/assets/summary-sources/b.jsonl\"}}";
        let (content, reasoning) =
            remap_message_for_session_move(text, None, false, text, Some(&source), &target);
        let content = content.expect("content must change");
        assert!(content.contains("archive_file_path: /t/assets/overflow-history.md"));
        assert!(content.contains("archive_directory_path: /t/assets/internal-note-overflow"));
        assert!(content.contains("- archive_file_path: /t/assets/folded-tool-groups/a.md"));
        assert!(content.contains("\"archive_file_path\":\"/t/assets/summary-sources/b.jsonl\""));
        assert!(!content.contains("/s/assets"));
        assert_eq!(reasoning, None);
    }

    #[test]
    fn original_file_pointers_are_never_rewritten() {
        let source = PathBuf::from("/s/assets");
        let target = PathBuf::from("/t/assets");
        let text = "original_file_path: /s/assets/overflow-history.md\noriginal_range: 1-10";
        assert_eq!(
            remap_message_for_session_move(text, None, false, text, Some(&source), &target),
            (None, None)
        );
    }

    #[test]
    fn discussion_text_without_writer_format_is_untouched() {
        let source = PathBuf::from("/s/assets");
        let target = PathBuf::from("/t/assets");
        let text = "never treat archive_file_path as a real parameter name";
        assert_eq!(
            remap_message_for_session_move(text, None, false, text, Some(&source), &target),
            (None, None)
        );
    }

    #[test]
    fn legacy_content_stub_file_path_remaps() {
        let source = PathBuf::from("/s/assets");
        let target = PathBuf::from("/t/assets");
        let text = "[[PRESERVED_CONTENT_STUB_V1]]{\"kind\":\"user\",\"file_path\":\"/s/assets/user-overflow-preserved/u.json\"}";
        let (content, reasoning) =
            remap_message_for_session_move(text, None, false, text, Some(&source), &target);
        let content = content.expect("content must change");
        assert!(
            content.contains("/t/assets/user-overflow-preserved/u.json"),
            "{content}"
        );
        assert!(!content.contains("/s/assets"), "{content}");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn invalid_provenance_is_never_laundered_into_authority() {
        use crate::ai::history::types::FOLDED_TOOL_ORIGIN;

        let source = PathBuf::from("/s/assets");
        let target = PathBuf::from("/t/assets");
        let text = "compressed_tool_round: 1 tool calls\narchive_file_path: /s/assets/f.md";
        let sidecar = format!(
            "{FOLDED_TOOL_ORIGIN}{{\"sources\":[{{\"id\":\"c\",\"name\":\"n\",\"arguments_sha256\":\"a\",\"result_sha256\":\"b\"}}],\"content_sha256\":\"deadbeef\",\"archive_file_path\":\"/s/assets/f.md\",\"offloaded\":false}}"
        );
        let (content, reasoning) = remap_message_for_session_move(
            text,
            Some(&sidecar),
            false,
            text,
            Some(&source),
            &target,
        );
        // The body pointer still moves, but the stale sidecar is left alone so
        // the note stays unauthorized exactly as before the move.
        assert!(
            content
                .expect("body must change")
                .contains("/t/assets/f.md")
        );
        assert_eq!(reasoning, None);
    }
}
