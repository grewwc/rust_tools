//! Explicit offloading of runtime-authored folded tool evidence.
//!
//! Visible marker text is never authorization. A private, content-bound sidecar
//! records every constituent call, including those omitted from the preview.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::history::types::{FOLDED_TOOL_ORIGIN, Message, ROLE_INTERNAL_NOTE};
use crate::ai::tools::registry::common::{
    ToolPrunePolicy, is_registered_tool_name, tool_history_policy,
};
use crate::ai::types::ToolCall;

use super::{COMPRESSED_TOOL_EVIDENCE_MARKER, PlannedArchiveWrite, content_sha256_hex};

pub(super) const MIN_FOLDED_CHARS: usize = 1024;
pub(super) const FOLD_ID_PREFIX: &str = "fold_";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCall {
    id: String,
    name: String,
    arguments_sha256: String,
    result_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FoldProvenance {
    sources: Vec<SourceCall>,
    content_sha256: String,
    archive_file_path: String,
    offloaded: bool,
}

/// Unknown tools (including unsupported wrappers) fail closed. A wrapper can
/// only be enabled once its complete descendant policy is explicitly supported.
pub(super) fn tool_allows_explicit_prune(name: &str) -> bool {
    is_registered_tool_name(name) && tool_history_policy(name).allows_prune()
}

/// Whether this tool's result may be explicitly offloaded *now*, given the
/// session facts the message list cannot express.
///
/// Used on the raw-result path, where the original text is still available to
/// carry the task markers an `AfterIntegration` decision needs. Unregistered
/// tools and conditional tools without a live authorization fail closed.
pub(super) fn tool_allows_explicit_prune_now(
    name: &str,
    content: &str,
    authorization: &super::PruneAuthorization,
) -> bool {
    if !is_registered_tool_name(name) {
        return false;
    }
    match tool_history_policy(name).prune_policy() {
        ToolPrunePolicy::Allow => true,
        ToolPrunePolicy::Never => false,
        ToolPrunePolicy::AfterIntegration => authorization.allows_subagent_result(content),
    }
}

/// Whether this tool's prune policy defers to session authorization.
pub(super) fn tool_prune_is_conditional(name: &str) -> bool {
    is_registered_tool_name(name)
        && matches!(
            tool_history_policy(name).prune_policy(),
            ToolPrunePolicy::AfterIntegration
        )
}

/// Retention predicate for an already-offloaded stub; see
/// [`crate::ai::tools::registry::common::ToolHistoryPolicy::retains_stub_mark`].
pub(super) fn tool_retains_stub_mark(name: &str) -> bool {
    is_registered_tool_name(name) && tool_history_policy(name).retains_stub_mark()
}

fn source_calls(calls: &[ToolCall], results: &FxHashMap<&str, &str>) -> Option<Vec<SourceCall>> {
    let mut ids = FxHashSet::default();
    let mut sources = Vec::with_capacity(calls.len());
    for call in calls {
        if call.id.is_empty() || !ids.insert(call.id.as_str()) {
            return None;
        }
        let args: Value = serde_json::from_str(&call.function.arguments).ok()?;
        if !args.is_object() {
            return None;
        }
        sources.push(SourceCall {
            id: call.id.clone(),
            name: call.function.name.clone(),
            arguments_sha256: content_sha256_hex(call.function.arguments.as_bytes()),
            // Legacy call IDs can be reused: consent for one observed result
            // must never authorize a different result of the same invocation.
            result_sha256: content_sha256_hex(results.get(call.id.as_str())?.as_bytes()),
        });
    }
    (!sources.is_empty()).then_some(sources)
}

fn complete_group_sources(messages: &[Message], group: &[usize]) -> Option<Vec<SourceCall>> {
    let assistant = messages.get(*group.first()?)?;
    if assistant.role != "assistant" {
        return None;
    }
    let calls = assistant.tool_calls.as_ref()?;
    if group.len() != calls.len() + 1 {
        return None;
    }
    let mut results = FxHashMap::default();
    for &index in &group[1..] {
        let result = messages.get(index)?;
        if result.role != "tool"
            || results
                .insert(result.tool_call_id.as_deref()?, result.content.as_str()?)
                .is_some()
        {
            return None;
        }
    }
    source_calls(calls, &results)
}

impl FoldProvenance {
    pub(super) fn id(&self) -> String {
        source_id(&self.sources)
    }

    pub(super) fn allows_prune(&self) -> bool {
        self.sources
            .iter()
            .all(|call| tool_allows_explicit_prune(&call.name))
    }

    pub(super) fn is_offloaded(&self) -> bool {
        self.offloaded
    }

    pub(super) fn label(&self) -> String {
        format!(
            "folded [{}]",
            self.sources
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    pub(super) fn source_ids(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().map(|source| source.id.as_str())
    }

    fn encode(&self) -> Option<String> {
        Some(format!(
            "{FOLDED_TOOL_ORIGIN}{}",
            serde_json::to_string(self).ok()?
        ))
    }
}

fn source_id(sources: &[SourceCall]) -> String {
    // Strings and vectors serialize infallibly; no display truncation is used.
    format!(
        "{FOLD_ID_PREFIX}{}",
        content_sha256_hex(&serde_json::to_vec(sources).expect("source calls serialize"))
    )
}

/// Called only by the fold producer; no files are written during planning.
pub(super) fn attach_provenance(
    note: &mut Message,
    messages: &[Message],
    group: &[usize],
    archive_file_path: Option<&str>,
) {
    let Some(archive_file_path) = archive_file_path else {
        return;
    };
    let Some(sources) = complete_group_sources(messages, group) else {
        return;
    };
    let Some(content) = note.content.as_str() else {
        return;
    };
    // The candidate list must identify the exact visible note, not just a tool
    // name and size shared by unrelated evidence groups.
    let content = format!("{content}\nprune_id: {}", source_id(&sources));
    let provenance = FoldProvenance {
        sources,
        content_sha256: content_sha256_hex(content.as_bytes()),
        archive_file_path: archive_file_path.to_string(),
        offloaded: false,
    };
    note.content = Value::String(content);
    note.reasoning_content = provenance.encode();
}

/// Persisted sidecars are accepted only on unchanged, structurally valid folds.
/// Legacy notes, altered summaries and marker strings inside tool results cannot
/// acquire pruning authority by resembling a runtime fold.
pub(super) fn provenance(note: &Message) -> Option<FoldProvenance> {
    if note.role != ROLE_INTERNAL_NOTE || note.tool_calls.is_some() || note.tool_call_id.is_some() {
        return None;
    }
    let encoded = note
        .reasoning_content
        .as_deref()?
        .strip_prefix(FOLDED_TOOL_ORIGIN)?;
    let meta: FoldProvenance = serde_json::from_str(encoded).ok()?;
    let content = note.content.as_str()?;
    if !content.starts_with("compressed_tool_round:")
        || !content
            .lines()
            .any(|line| line == COMPRESSED_TOOL_EVIDENCE_MARKER)
        || meta.content_sha256 != content_sha256_hex(content.as_bytes())
        || meta.archive_file_path.is_empty()
        || meta.sources.is_empty()
    {
        return None;
    }
    let mut ids = FxHashSet::default();
    for source in &meta.sources {
        if source.id.is_empty()
            || !ids.insert(source.id.as_str())
            || source.name.is_empty()
            || source.arguments_sha256.len() != 64
            || !source
                .arguments_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || source.result_sha256.len() != 64
            || !source
                .result_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return None;
        }
    }
    Some(meta)
}

/// Raw and folded groups both count toward the recent-results window. Legacy
/// folds count conservatively too, although they can never become candidates.
///
/// A raw anchor claims a slot only when at least one of its calls is answered
/// inside its own span: an interrupted call that never produced a response holds
/// no evidence, and letting it claim a slot would push the oldest group that the
/// window is supposed to protect out of that window.
pub(super) fn recent_group_indices(messages: &[Message]) -> FxHashSet<usize> {
    let mut recent = FxHashSet::default();
    for (index, message) in messages.iter().enumerate().rev() {
        if recent.len() >= super::KEEP_RECENT_TOOL_GROUPS {
            break;
        }
        let counts_as_group = if message.role == "assistant" {
            message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| raw_group_has_results(messages, index, calls))
        } else {
            super::is_compressed_tool_evidence_note(message)
        };
        if counts_as_group {
            recent.insert(index);
        }
    }
    recent
}

/// True when at least one call of the anchor is answered by a tool message inside
/// the group's own span (which ends at the next raw anchor). A partially answered
/// anchor still claims its slot: it did produce evidence, and refusing the slot
/// would strip the protection that its surviving results would otherwise keep.
fn raw_group_has_results(messages: &[Message], anchor: usize, calls: &[ToolCall]) -> bool {
    if calls.is_empty() {
        return false;
    }
    for message in messages.iter().skip(anchor + 1) {
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
        {
            break;
        }
        if message.role != "tool" {
            continue;
        }
        let Some(id) = message.tool_call_id.as_deref() else {
            continue;
        };
        if calls.iter().any(|call| call.id.as_str() == id) {
            return true;
        }
    }
    false
}

/// Fold identities that do not resolve to exactly one piece of evidence.
///
/// The identity is derived from the constituent calls only, so a reused call ID
/// with identical arguments and results lets two groups (or a fold note and a raw
/// call) answer to the same id, and a textual mark cannot say which one it means.
/// Such an id is never offered as a candidate, and the mark map drops it as well:
/// carrying the accumulated consent across such a state change would let a later
/// note be offloaded by marks earned for an earlier twin.
pub(super) fn ambiguous_fold_ids(messages: &[Message]) -> FxHashSet<String> {
    let mut carriers: FxHashMap<String, usize> = FxHashMap::default();
    // A note keeps its identity in the private sidecar; a complete raw group
    // recomputes the same identity from the group it still contains.
    for message in messages {
        if let Some(meta) = provenance(message) {
            *carriers.entry(meta.id()).or_insert(0) += 1;
        }
    }
    for id in canonical_fold_ids(messages) {
        *carriers.entry(id).or_insert(0) += 1;
    }
    // Raw results are marked by their call ID, which lives in the same textual
    // namespace: an id a raw call also answers to is not resolvable either.
    let raw_ids: FxHashSet<&str> = messages
        .iter()
        .flat_map(|message| {
            message
                .tool_calls
                .iter()
                .flatten()
                .map(|call| call.id.as_str())
                .chain(message.tool_call_id.as_deref())
        })
        .collect();
    carriers
        .into_iter()
        .filter(|(id, count)| *count != 1 || raw_ids.contains(id.as_str()))
        .map(|(id, _)| id)
        .collect()
}

pub(super) fn candidates(messages: &[Message]) -> Vec<(usize, FoldProvenance)> {
    let recent = recent_group_indices(messages);
    // An id that several items answer to cannot be marked unambiguously; the
    // colliding counterpart stays protected regardless of its own eligibility.
    let ambiguous = ambiguous_fold_ids(messages);
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let meta = provenance(message)?;
            (!recent.contains(&index)
                && !ambiguous.contains(&meta.id())
                && meta.allows_prune()
                && !meta.offloaded
                && message.content.as_str()?.chars().count() >= MIN_FOLDED_CHARS)
                .then_some((index, meta))
        })
        .collect()
}

/// Reconcile persisted fold marks against complete surviving canonical groups.
/// A rewind that removes even one response invalidates the entire group's key.
pub(super) fn canonical_fold_ids(messages: &[Message]) -> FxHashSet<String> {
    let mut ids = FxHashSet::default();
    for (index, message) in messages.iter().enumerate() {
        if message.role != "assistant" {
            continue;
        }
        let Some(calls) = message.tool_calls.as_ref() else {
            continue;
        };
        let end = index
            .saturating_add(calls.len())
            .saturating_add(1)
            .min(messages.len());
        let group = (index..end).collect::<Vec<_>>();
        let Some(sources) = complete_group_sources(messages, &group) else {
            continue;
        };
        if sources
            .iter()
            .all(|source| tool_allows_explicit_prune(&source.name))
        {
            ids.insert(source_id(&sources));
        }
    }
    ids
}

/// Preserve the exact outgoing note, not a new summary, before replacing it.
/// The earlier fold archive remains separately reachable and is explicitly a
/// projection copy: it is not claimed to contain canonical uncompressed output.
pub(super) fn offload(note: &mut Message, mut meta: FoldProvenance, dir: &Path) -> Option<usize> {
    let content = note.content.as_str()?;
    let digest = content_sha256_hex(content.as_bytes());
    let path = dir.join("pruned-folds").join(format!("{digest}.txt"));
    let checkpoint = content
        .lines()
        .find(|line| line.starts_with("assistant_checkpoint:"))?;
    let mut stub = format!(
        "compressed_tool_round: {} tool calls (explicitly offloaded; recoverable)\n{COMPRESSED_TOOL_EVIDENCE_MARKER}\nprune_id: {}\n- archive_file_path: {}\n- archive_scope: exact_folded_note_before_explicit_prune\n- source_projection_archive: {}\n{checkpoint}\n",
        meta.sources.len(),
        meta.id(),
        path.display(),
        meta.archive_file_path,
    );
    for source in &meta.sources {
        stub.push_str(&format!("- {} · {}\n", source.id, source.name));
    }
    stub.push_str("Recovery: read the exact note archive for prior previews and original_* pointers before repeating tools. Canonical history is unchanged.");
    let freed = content.chars().count().checked_sub(stub.chars().count())?;
    if freed == 0 || !PlannedArchiveWrite::new(path, content.to_string()).commit() {
        return None;
    }
    meta.offloaded = true;
    meta.content_sha256 = content_sha256_hex(stub.as_bytes());
    let encoded = meta.encode()?;
    note.content = Value::String(stub);
    note.reasoning_content = Some(encoded);
    Some(freed)
}

#[cfg(test)]
#[path = "folded_prune_tests.rs"]
mod tests;
