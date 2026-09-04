//! LLM-guided context pruning (model marks → deferred offloading).
//!
//! This is a **supplement** to the existing compression logic; it does not modify any existing compression code.
//!
//! ## How it works
//!
//! On every model call, a short prompt is appended to the system prompt asking the model to
//! use `<meta:self_note>prune:tool_call_id1,tool_call_id2</meta:self_note>` in its response to
//! mark tool messages that are no longer needed (usually stale ordinary tool results).
//! The prompt ends with a dynamic per-request section listing the exact eligible ids (see
//! [`build_prune_protocol_prompt`]), so the model does not have to recall ids from history.
//!
//! - Roles such as user / system / assistant / internal_note are never pruned even if
//!   they are marked (see `is_protected_role`).
//! - Only messages with `role == "tool"` and a `tool_call_id` can be pruned.
//! - Results whose tool declares `prune: Never` via `ToolHistoryPolicyRegistration`
//!   (e.g. `plan`) are never pruned; `read_file` / retrieval / `execute_command` results
//!   are "lossy-incompressible" but **may** still be pruned by the LLM once stale (the two dimensions are orthogonal).
//! - After being marked **PRUNE_THRESHOLD** times cumulatively, the message content is offloaded to the session asset disk and
//!   the inline text is replaced with a **recallable stub** (keeping `file_path` + a recall anchor + head/tail preview,
//!   preserving the message structure without deletion, to avoid breaking the tool_call ↔ tool_response pairing).
//!   Exception: results at or above [`PRUNE_SINGLE_MARK_OFFLOAD_CHARS`] offload after a single mark (see [`needed_marks`]).
//! - Counting semantics are "silence-tolerant + monotonically accumulating" rather than "consecutive": if the model produces no prune
//!   directives this round, the counts stay untouched (intermediate tool-only rounds do not wrongly reset them); marked ids only increase
//!   (+1), and unmarked existing entries stay **unchanged, without decay**, until the id actually leaves the context or is
//!   excluded by a protection policy. Why no decay: each round the model usually marks the results that "just became stale
//!   this round" (different ids each round), so decay would zero counts before reaching the threshold and the mechanism would
//!   almost never fire in the most typical usage. See [`update_prune_marks`].
//!
//! ## Safety guarantees (no loss of real information)
//!
//! 1. No message is deleted; the messages array length and order are unchanged.
//! 2. The existing `compress/mod.rs` / `context_budget.rs` logic is not modified.
//! 3. Pruning is **lossless and recallable**: the full pruned tool result is first written to the session asset, and the inline text
//!    is replaced only by a recall stub carrying `file_path`; the model can `read_file` the full original at any time.
//! 4. **Never prune when there is no archive directory (`overflow_dir=None`)**: prefer not compressing over doing
//!    an irreversible content drop.
//! 5. `apply_pruning` only touches the temporary `messages` projection used per model request; persistence uses the separate
//!    canonical `turn_messages`, so offloading never pollutes the real history.
//! 6. The most recent `KEEP_RECENT_TOOL_GROUPS` groups of tool results are always protected, to avoid wrongly pruning results the current round still needs.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::history::types::Message;

use super::tool_overflow::{
    build_tool_call_arguments_index, build_tool_call_name_index, build_tool_overflow_recall_lines,
    is_preserved_tool_overflow_stub, preserve_pruned_tool_result_stable,
};

/// Returns whether a tool result is marked "never LLM-pruned" by its registered policy.
/// Consults the [`ToolHistoryPolicy`] the tool itself declares (see each tool's registration file),
/// rather than hardcoding tool names. By default (unregistered) pruning is allowed; only tools that
/// explicitly declare `prune: Never` (e.g. `plan`) return true.
fn is_prune_protected_tool(tool_name: &str) -> bool {
    !crate::ai::tools::registry::common::tool_history_policy(tool_name).allows_prune()
}

/// How many cumulative marks are needed before a message is offloaded/pruned.
///
/// Uses "silence-tolerant + monotonically accumulating" semantics (see [`update_prune_marks`]), so the threshold here
/// is a **cumulative** rather than **consecutive** count. Since pruning is lossless and recallable (full text archived + stub),
/// a low threshold works: 2 marks offload the message, balancing aggressive reclamation with hysteresis against a single stray token.
///
/// Exception: results at or above [`PRUNE_SINGLE_MARK_OFFLOAD_CHARS`] offload after a single
/// mark (see [`needed_marks`]).
pub(crate) const PRUNE_THRESHOLD: u8 = 2;

/// Only old tool results reaching this size are worth exposing the pruning protocol to the model.
/// The actual replacement still compares against the generated stub, ensuring the text only ever shrinks on every path.
const PRUNE_MIN_CONTENT_CHARS: usize = 4_096;

/// Tool results at or above this size are offloaded after a **single** model
/// mark instead of [`PRUNE_THRESHOLD`].
///
/// Rationale: a result this large is re-sent in full on every round it stays
/// visible, so waiting for a second mark usually costs more tokens than the
/// rare wrong mark loses — and pruning is lossless (full text archived, a
/// recallable stub stays in place), so a wrong mark is recoverable.
const PRUNE_SINGLE_MARK_OFFLOAD_CHARS: usize = 16_384;

/// Marks required before a result of this size is offloaded: one for very
/// large results (see [`PRUNE_SINGLE_MARK_OFFLOAD_CHARS`]), [`PRUNE_THRESHOLD`]
/// otherwise. Shared by `apply_pruning` (the actual gate) and
/// [`build_prune_protocol_prompt`] (the per-candidate annotation the model sees).
fn needed_marks(content_chars: usize) -> u8 {
    if content_chars >= PRUNE_SINGLE_MARK_OFFLOAD_CHARS {
        1
    } else {
        PRUNE_THRESHOLD
    }
}

/// Per-id variant of [`needed_marks`] for display: marks still required before
/// this id offloads. For ids not found in `messages` (not an active candidate)
/// this falls back to [`PRUNE_THRESHOLD`].
pub(crate) fn needed_marks_for(messages: &[Message], tool_call_id: &str) -> u8 {
    messages
        .iter()
        .find(|message| {
            message.role == "tool" && message.tool_call_id.as_deref() == Some(tool_call_id)
        })
        .and_then(|message| message.content.as_str())
        .map(|content| needed_marks(content.chars().count()))
        .unwrap_or(PRUNE_THRESHOLD)
}

/// Maximum candidates rendered into the per-request protocol section; the
/// largest results are listed first because they free the most context.
const PRUNE_PROMPT_MAX_CANDIDATES: usize = 8;

/// The pruning-protocol instructions injected into the system prompt.
/// Kept short to avoid consuming too many tokens.
pub(crate) const PRUNE_PROTOCOL_PROMPT: &str = "\n## Context Management Protocol\n\
When your context holds outdated tool results, actively reclaim space by marking them.\n\
Each tool result in the history has a stable id (the `call_id` / `tool_call_id` shown on\n\
that tool output). Include a hidden self-note listing the ids to prune:\n\
`<meta:self_note>prune:call_abc,call_xyz</meta:self_note>`\n\
Mark any tool result that is now superseded or no longer needed — including old file\n\
reads and code/search results whose content you have already used, that you have since\n\
re-read, or that describe code you have already edited.\n\
Rules:\n\
- Never mark user messages, system instructions, assistant messages, plans, or the most recent tool results.\n\
- Marking is safe and reversible: pruning is loss-free — the full result is archived to a\n\
  session file and the kept stub shows its `file_path`, so you can re-read it anytime if you\n\
  turn out to still need it. The system only prunes after you mark an id on a couple of turns\n\
  and always protects recent results and plans.\n\
- Put the `prune:` directive on its own line; if you also write a normal self_note, keep it in the same hidden note.";

/// Returns whether messages of this role are protected (never pruned).
fn is_protected_role(role: &str) -> bool {
    !matches!(role, "tool")
    // tool is not protected; all other roles are
}

/// Same as above, written more clearly.
fn is_prunable_message(msg: &Message) -> bool {
    msg.role == "tool" && msg.tool_call_id.is_some()
}

/// Parses prune marks from the hidden_meta of a model response.
///
/// hidden_meta may span multiple lines; lines starting with `prune:` are pruning directives,
/// the rest is regular self_note content (handled by the caller).
///
/// Returns `(prune_ids, remaining_meta)`:
/// - `prune_ids`: the list of marked tool_call_ids
/// - `remaining_meta`: the hidden_meta left after removing the prune lines (for the self_note)
pub(crate) fn parse_prune_from_hidden_meta(hidden_meta: &str) -> (Vec<String>, String) {
    let mut prune_ids = Vec::new();
    let mut remaining_lines = Vec::new();
    let mut saw_prune = false;

    for line in hidden_meta.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("prune:") {
            saw_prune = true;
            // Parse the comma-separated tool_call_id list
            for id in rest.split(',') {
                let id = id.trim();
                if !id.is_empty() {
                    prune_ids.push(id.to_string());
                }
            }
        } else if !trimmed.is_empty() {
            remaining_lines.push(line.to_string());
        }
    }

    let remaining = if saw_prune {
        remaining_lines.join("\n")
    } else {
        hidden_meta.to_string()
    };
    (prune_ids, remaining)
}

/// Updates the pruning counters ("tolerate silent rounds + monotonic
/// accumulation" semantics).
///
/// - `current_marks`: this session's prune counter table (tool_call_id →
///   accumulated count)
/// - `prune_ids`: the tool_call_ids the model marked this round
/// - `active_prunable_tool_ids`: the tool_call_ids actually present in this
///   round's messages and eligible for pruning
///
/// Logic:
/// 1. **Silent rounds never touch counters**: when the model produces no valid
///    prune instruction this round (no id in `prune_ids` hits an active id),
///    the whole table stays unchanged (only stale entries that left the context
///    / are protected get cleaned up), so "back-to-back tool calls with an
///    intermediate round that wrote no self_note" does not wrongly clear
///    previously accumulated counts.
/// 2. Ids marked this round get +1 (monotonic accumulation). **Existing
///    unmarked entries stay unchanged, no decay**: each round the model usually
///    marks "results it just finished using that have now gone stale" (a
///    different id each round); applying decay to unmarked items would zero
///    them before reaching the threshold, making the mechanism almost never
///    fire under the most typical usage — exactly the hidden failure of the
///    earlier decay version. Accumulation only grows, until the id actually
///    leaves the context or is excluded by the protection policy.
/// 3. Clean up entries no longer in the current context or excluded by the
///    protection policy.
pub(crate) fn update_prune_marks(
    current_marks: &mut FxHashMap<String, u8>,
    prune_ids: &[String],
    active_prunable_tool_ids: &FxHashSet<String>,
) {
    let marked_ids = prune_ids
        .iter()
        .filter(|id| active_prunable_tool_ids.contains(*id))
        .cloned()
        .collect::<FxHashSet<_>>();

    // Increment the counters of marked tools (monotonic accumulation; a silent
    // round has empty marked_ids and is a no-op).
    for id in marked_ids {
        let count = current_marks.entry(id).or_insert(0);
        *count = count.saturating_add(1);
    }

    // Clean up entries with a zero count, no longer in the current context, or
    // excluded by the protection policy.
    current_marks.retain(|id, v| *v > 0 && active_prunable_tool_ids.contains(id));
}

/// Collects the tool_call_ids in the current context that may be pruned under
/// LLM guidance.
///
/// Protection policy:
/// - Results of the most recent complete tool group keep their full text.
/// - Results of tools whose registration policy declares `prune: Never` (e.g.
///   `plan`) are never pruned.
///   Note `read_file` / retrieval-like tools, though "not lossy-compressible",
///   **are** allowed to be pruned.
pub(crate) fn active_prunable_tool_ids(messages: &[Message]) -> FxHashSet<String> {
    let protected_ids = protected_tool_call_ids(messages);
    messages
        .iter()
        .filter_map(|message| {
            if !is_prunable_message(message) {
                return None;
            }
            if message
                .content
                .as_str()
                .is_some_and(is_preserved_tool_overflow_stub)
            {
                return None;
            }
            if message
                .content
                .as_str()
                .is_none_or(|content| content.chars().count() < PRUNE_MIN_CONTENT_CHARS)
            {
                return None;
            }
            let id = message.tool_call_id.as_ref()?;
            (!protected_ids.contains(id)).then(|| id.clone())
        })
        .collect()
}

fn protected_tool_call_ids(messages: &[Message]) -> FxHashSet<String> {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let protected_indices = super::tool_groups::recent_tool_group_message_indices(
        messages,
        super::KEEP_RECENT_TOOL_GROUPS,
    );

    let mut protected = FxHashSet::default();
    for (idx, message) in messages.iter().enumerate() {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        if protected_indices.contains(&idx) {
            protected.insert(tool_call_id.clone());
            continue;
        }
        if id_to_tool_name
            .get(tool_call_id)
            .is_some_and(|name| is_prune_protected_tool(name))
        {
            protected.insert(tool_call_id.clone());
        }
    }
    protected
}

/// Explains why a marked id was not accepted this round, so the driver can
/// surface actionable terminal feedback instead of silently dropping the mark
/// (unexplained rejections make the model repeat useless marks). Returns
/// `None` when the id is actually prunable.
///
/// Deliberately per-id (rebuilds the small indexes on each call): rejections
/// are rare (usually 0-2 per round), so clarity beats batching here. The
/// checks mirror [`active_prunable_tool_ids`] but are ordered for message
/// quality (why exactly, not merely that it is ineligible).
pub(crate) fn explain_rejected_prune_mark(
    messages: &[Message],
    tool_call_id: &str,
) -> Option<&'static str> {
    let Some(message) = messages.iter().find(|message| {
        message.role == "tool" && message.tool_call_id.as_deref() == Some(tool_call_id)
    }) else {
        return Some("no such tool result in the current context");
    };
    if message
        .content
        .as_str()
        .is_some_and(is_preserved_tool_overflow_stub)
    {
        return Some("already offloaded to the session archive");
    }
    let id_to_tool_name = build_tool_call_name_index(messages);
    if id_to_tool_name
        .get(tool_call_id)
        .is_some_and(|name| is_prune_protected_tool(name))
    {
        return Some("tool declares prune:Never");
    }
    if protected_tool_call_ids(messages).contains(&tool_call_id.to_string()) {
        return Some("inside the recent-results protection window");
    }
    if message
        .content
        .as_str()
        .is_none_or(|content| content.chars().count() < PRUNE_MIN_CONTENT_CHARS)
    {
        return Some("below the minimum size for pruning");
    }
    // All specific checks passed; re-consult the authoritative eligibility
    // set so the "None means prunable" contract cannot drift from
    // [`active_prunable_tool_ids`] (it may gain conditions later).
    if active_prunable_tool_ids(messages).contains(tool_call_id) {
        None
    } else {
        Some("not currently eligible")
    }
}

/// Pruning statistics of a single `apply_pruning` call, for the caller to
/// print a terminal notice.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PruneReport {
    /// Number of tool results offloaded to disk this time and inline-replaced
    /// with a recall stub.
    pub(crate) pruned_count: usize,
    /// Net characters freed (sum of original content lengths minus stub lengths).
    pub(crate) freed_chars: usize,
    /// Tool names involved (deduplicated, in first-appearance order).
    pub(crate) tools: Vec<String>,
}

/// Applies pruning to the messages array (lossless, recallable offload).
///
/// Tool messages whose accumulated marks reach the per-result threshold
/// (`PRUNE_THRESHOLD`, or a single mark for results at or above
/// `PRUNE_SINGLE_MARK_OFFLOAD_CHARS`) have their full text offloaded to the
/// session asset directory and are inline-replaced with a stub carrying
/// `file_path` + a recall anchor + head/tail previews; no message is deleted
/// and the array length never changes — the model can `read_file` the full
/// original at any time.
///
/// **Safety floor**: with `overflow_dir=None` (no archive directory, e.g. a
/// temporary/one-shot session), **never prune** — prefer not compressing over
/// irreversible dropping. A single entry whose archive write fails also keeps
/// its original text and is skipped.
///
/// Messages protected by `protected_tool_call_ids` (the most recent complete
/// tool group, and tools whose registration policy declares `prune: Never`,
/// e.g. `plan`) are never pruned, avoiding wrongfully pruning results needed
/// this round or task-roadmap anchors.
///
/// Returns the statistics report of this pruning run (for the caller to print
/// a terminal notice).
pub(crate) fn apply_pruning(
    messages: &mut [Message],
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
) -> PruneReport {
    let mut report = PruneReport::default();
    if prune_marks.is_empty() {
        return report;
    }
    // Safety floor: without an archive directory there is no lossless recall,
    // so do not prune at all.
    if overflow_dir.is_none() {
        return report;
    }

    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let protected_ids = protected_tool_call_ids(messages);

    for msg in messages.iter_mut() {
        if !is_prunable_message(msg) {
            continue;
        }

        let Some(tool_call_id) = msg.tool_call_id.clone() else {
            continue;
        };

        if protected_ids.contains(&tool_call_id) {
            continue;
        }

        let Some(&count) = prune_marks.get(&tool_call_id) else {
            continue;
        };

        let Some(content) = msg.content.as_str() else {
            continue;
        };
        // Very large results offload after a single mark: re-sending them in
        // full every extra round costs more than the rare wrong mark loses,
        // and the offload is lossless and recallable either way.
        if count < needed_marks(content.chars().count()) {
            continue;
        }
        // The request projection is reused across multiple model rounds; an
        // already-offloaded stub must not be counted again in the pruning report.
        if is_preserved_tool_overflow_stub(content) {
            continue;
        }
        let freed = content.chars().count();
        let tool_name = id_to_tool_name
            .get(&tool_call_id)
            .map(String::as_str)
            .unwrap_or("tool");
        let recall_lines = id_to_tool_args
            .get(&tool_call_id)
            .map(|args| build_tool_overflow_recall_lines(tool_name, args))
            .unwrap_or_default();

        // Lossless offload: full text to disk + generate a recall stub. If
        // archiving fails (including overflow_dir=None, which returned early
        // above), keep the original text and skip the entry — never drop
        // irreversibly.
        let Some(stub) = preserve_pruned_tool_result_stable(
            overflow_dir,
            &tool_call_id,
            tool_name,
            content,
            &recall_lines,
        ) else {
            continue;
        };
        // The goal of pruning is to reclaim context; swapping a short result for
        // a longer path stub would only bloat it.
        // The archive is content-addressed and idempotent, so no duplicate copies
        // are created in later requests even if already written.
        if stub.chars().count() >= freed {
            continue;
        }

        if !report.tools.iter().any(|name| name == tool_name) {
            report.tools.push(tool_name.to_string());
        }
        report.freed_chars += freed.saturating_sub(stub.chars().count());
        msg.content = Value::String(stub);
        report.pruned_count += 1;
    }

    report
}

/// Injects the protocol only when the current request actually contains
/// prunable old tool results.
///
/// This tracks the capability boundary better than a fixed message-count
/// threshold: oversized old results in a short conversation get reclaimed
/// promptly, while long tool-less conversations waste no prompt tokens.
pub(crate) fn should_inject_prune_prompt(messages: &[Message]) -> bool {
    !active_prunable_tool_ids(messages).is_empty()
}

/// Injects the pruning protocol into the model request projection on demand,
/// guaranteeing repeated calls never duplicate the prompt.
/// Recognizes the protocol message regardless of which candidate list was
/// rendered into it: the prompt is static instructions plus a dynamic
/// per-request section, so full-text equality no longer identifies it.
pub(crate) fn is_prune_protocol_message(message: &Message) -> bool {
    message.role == "system"
        && matches!(&message.content, Value::String(text) if text.starts_with(PRUNE_PROTOCOL_PROMPT))
}

/// Builds the full protocol prompt for this request: the static instructions
/// plus a dynamic section listing the currently prunable candidates (largest
/// first, capped at [`PRUNE_PROMPT_MAX_CANDIDATES`]). `prune_marks` annotates
/// each candidate with its accumulated counter (`marks/threshold`) so the
/// model can see how close a result is to being offloaded.
///
/// Listing exact ids is the main fix for the earlier static-prompt version,
/// where the model had to recall `tool_call_id`s from history and rarely
/// emitted valid marks.
fn build_prune_protocol_prompt(
    messages: &[Message],
    prune_marks: &FxHashMap<String, u8>,
) -> String {
    let active_ids = active_prunable_tool_ids(messages);
    if active_ids.is_empty() {
        return PRUNE_PROTOCOL_PROMPT.to_string();
    }
    let id_to_tool_name = build_tool_call_name_index(messages);
    let mut candidates: Vec<(String, &str, usize)> = messages
        .iter()
        .filter_map(|message| {
            let id = message.tool_call_id.as_ref()?;
            if !active_ids.contains(id) {
                return None;
            }
            let chars = message.content.as_str()?.chars().count();
            let tool = id_to_tool_name
                .get(id)
                .map(String::as_str)
                .unwrap_or("tool");
            Some((id.clone(), tool, chars))
        })
        .collect();
    // Largest first (most context freed); tie-break by id for determinism.
    candidates.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    let mut section = String::from(
        "\n### Prunable candidates in this request (id · tool · size · marks/threshold)\n",
    );
    for (id, tool, chars) in candidates.iter().take(PRUNE_PROMPT_MAX_CANDIDATES) {
        let count = prune_marks.get(id).copied().unwrap_or(0);
        section.push_str(&format!(
            "- {id} · {tool} · {chars} chars · marks {count}/{}\n",
            needed_marks(*chars)
        ));
    }
    let hidden = candidates.len().saturating_sub(PRUNE_PROMPT_MAX_CANDIDATES);
    if hidden > 0 {
        section.push_str(&format!(
            "(+{hidden} more eligible candidates not listed)\n"
        ));
    }

    let mut prompt = PRUNE_PROTOCOL_PROMPT.to_string();
    prompt.push_str(&section);
    prompt
}

/// Injects the pruning protocol into the model request projection on demand,
/// guaranteeing repeated calls never duplicate the prompt message. When the
/// protocol is already present and candidates still exist, the dynamic
/// candidate list is refreshed in place instead; when no prunable candidates
/// remain, the stale protocol message (with its outdated candidate list) is
/// removed so the model never sees ids that no longer exist.
pub(crate) fn ensure_prune_protocol_prompt(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
) -> bool {
    if let Some(existing_idx) = messages
        .iter()
        .position(|message| is_prune_protocol_message(message))
    {
        if should_inject_prune_prompt(messages) {
            let prompt = build_prune_protocol_prompt(messages, prune_marks);
            messages[existing_idx].content = Value::String(prompt);
        } else {
            // No prunable candidates remain in this projection: the candidate
            // list inside the protocol message would now name ids that are no
            // longer present, which misleads the model into re-marking them.
            // Remove the stale message instead of leaving it in place.
            messages.remove(existing_idx);
        }
        return false;
    }
    if !should_inject_prune_prompt(messages) {
        return false;
    }

    let insert_at = messages
        .iter()
        .position(|message| message.role != "system")
        .unwrap_or(messages.len());
    let prompt = build_prune_protocol_prompt(messages, prune_marks);
    messages.insert(
        insert_at,
        Message {
            role: "system".to_string(),
            content: Value::String(prompt),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
    true
}

/// Updates the transient projection before every model request: first
/// losslessly offload old tool results that reached the threshold, then inject
/// the protocol on demand.
///
/// The caller must pass a request projection kept separate from the canonical
/// `turn_messages`.
pub(crate) fn prepare_request_projection(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
) -> PruneReport {
    let report = apply_pruning(messages.as_mut_slice(), prune_marks, overflow_dir);
    ensure_prune_protocol_prompt(messages, prune_marks);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;

    fn make_tool_message(tool_call_id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            reasoning_content: None,
        }
    }

    fn make_user_message(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_assistant_message(content: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_assistant_tool_call(tool_call_id: &str, tool_name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: tool_call_id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: tool_name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn test_is_protected_role() {
        assert!(!is_protected_role("tool"));
        assert!(is_protected_role("user"));
        assert!(is_protected_role("system"));
        assert!(is_protected_role("assistant"));
        assert!(is_protected_role("internal_note"));
    }

    #[test]
    fn test_is_prunable_message() {
        let tool_msg = make_tool_message("call_1", "result");
        assert!(is_prunable_message(&tool_msg));

        let user_msg = make_user_message("hello");
        assert!(!is_prunable_message(&user_msg));

        let assistant_msg = make_assistant_message("hi");
        assert!(!is_prunable_message(&assistant_msg));

        // tool message but without a tool_call_id
        let tool_no_id = Message {
            role: "tool".to_string(),
            content: Value::String("result".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        assert!(!is_prunable_message(&tool_no_id));
    }

    #[test]
    fn test_parse_prune_from_hidden_meta() {
        let hidden_meta = "prune:call_abc,call_xyz\nDo: be concise\nAvoid: verbosity";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert_eq!(ids, vec!["call_abc", "call_xyz"]);
        assert!(remaining.contains("Do: be concise"));
        assert!(remaining.contains("Avoid: verbosity"));
        assert!(!remaining.contains("prune:"));
    }

    #[test]
    fn test_parse_prune_only() {
        let hidden_meta = "prune:call_1,call_2";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert_eq!(ids.len(), 2);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_no_prune() {
        let hidden_meta = "Do: be focused\nAvoid: tangents";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert!(ids.is_empty());
        assert_eq!(remaining, "Do: be focused\nAvoid: tangents");
    }

    #[test]
    fn test_parse_empty() {
        let (ids, remaining) = parse_prune_from_hidden_meta("");
        assert!(ids.is_empty());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_update_prune_marks_increment() {
        let mut marks = FxHashMap::default();
        let active: FxHashSet<String> = ["call_1", "call_2", "call_3"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Round 1 marks call_1, call_2
        update_prune_marks(
            &mut marks,
            &["call_1".to_string(), "call_2".to_string()],
            &active,
        );
        assert_eq!(marks.get("call_1"), Some(&1));
        assert_eq!(marks.get("call_2"), Some(&1));
        assert!(!marks.contains_key("call_3"));

        // Round 2 marks call_1, call_2
        update_prune_marks(
            &mut marks,
            &["call_1".to_string(), "call_2".to_string()],
            &active,
        );
        assert_eq!(marks.get("call_1"), Some(&2));
        assert_eq!(marks.get("call_2"), Some(&2));

        // Round 3 marks only call_1: monotonic accumulation — call_2 was not
        // marked but **stays unchanged** (no decay), while call_1 gets +1 again.
        update_prune_marks(&mut marks, &["call_1".to_string()], &active);
        assert_eq!(marks.get("call_1"), Some(&3));
        assert_eq!(marks.get("call_2"), Some(&2));
    }

    /// Realistic distribution check: each round the model marks **different**
    /// ids ("just used this round, newly stale") and never re-marks the same old
    /// id. Under monotonic accumulation each id's count only grows, so after
    /// several rounds the threshold is truly reached and pruning fires; under
    /// the old decay semantics the counts would be zeroed before reaching the
    /// threshold and never fire.
    #[test]
    fn test_update_prune_marks_distinct_ids_accumulate_monotonically() {
        let mut marks = FxHashMap::default();
        let active: FxHashSet<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();

        // Mark a different id each round (realistic model behavior).
        update_prune_marks(&mut marks, &["A".to_string()], &active);
        update_prune_marks(&mut marks, &["B".to_string()], &active);
        update_prune_marks(&mut marks, &["C".to_string()], &active);
        // Each of the three ids accumulated 1, none decayed to zero.
        assert_eq!(marks.get("A"), Some(&1));
        assert_eq!(marks.get("B"), Some(&1));
        assert_eq!(marks.get("C"), Some(&1));

        // Mark each one more round → threshold 2 reached, prunable.
        update_prune_marks(&mut marks, &["A".to_string()], &active);
        update_prune_marks(&mut marks, &["B".to_string()], &active);
        assert_eq!(marks.get("A"), Some(&2));
        assert_eq!(marks.get("B"), Some(&2));
        assert!(marks.values().any(|v| *v >= PRUNE_THRESHOLD));
    }

    /// A silent round (no valid prune marks this round) must not zero existing
    /// counters — this is the core fix of the new semantics over the old
    /// "consecutive" one: intermediate rounds of back-to-back tool calls no
    /// longer wrongly clear previously accumulated counts.
    #[test]
    fn test_update_prune_marks_silent_round_preserves_counts() {
        let mut marks = FxHashMap::default();
        marks.insert("call_1".to_string(), 1);
        let active: FxHashSet<String> =
            ["call_1", "call_2"].iter().map(|s| s.to_string()).collect();

        // Silent round: the model wrote no prune instructions.
        update_prune_marks(&mut marks, &[], &active);
        assert_eq!(marks.get("call_1"), Some(&1)); // count kept, not zeroed

        // One more mark afterwards reaches threshold 2.
        update_prune_marks(&mut marks, &["call_1".to_string()], &active);
        assert_eq!(marks.get("call_1"), Some(&2));
    }

    /// Silent rounds must still clean up stale entries that left the context /
    /// are protected, so the counter table does not grow unboundedly.
    #[test]
    fn test_update_prune_marks_silent_round_drops_stale_ids() {
        let mut marks = FxHashMap::default();
        marks.insert("call_1".to_string(), 2);
        marks.insert("stale".to_string(), 2);
        let active: FxHashSet<String> =
            ["call_1", "call_2"].iter().map(|s| s.to_string()).collect();

        update_prune_marks(&mut marks, &[], &active);

        // call_1 still active → kept; stale already left the context → cleaned up.
        assert_eq!(marks.get("call_1"), Some(&2));
        assert!(!marks.contains_key("stale"));
    }

    #[test]
    fn test_update_prune_marks_deduplicates_single_round_marks() {
        let mut marks = FxHashMap::default();
        let active: FxHashSet<String> = ["call_1"].iter().map(|s| s.to_string()).collect();

        update_prune_marks(
            &mut marks,
            &[
                "call_1".to_string(),
                "call_1".to_string(),
                "missing".to_string(),
            ],
            &active,
        );

        assert_eq!(marks.get("call_1"), Some(&1));
        assert!(!marks.contains_key("missing"));
    }

    /// Temporary archive directory for the apply_pruning tests.
    fn make_overflow_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ai-llm-prune-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_apply_pruning_replaces_content() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_keep".to_string(), 1);

        let mut messages = vec![
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message(
                "call_old",
                &"very long outdated result that should be pruned\n".repeat(100),
            ),
            make_assistant_tool_call("call_keep", "execute_command"),
            make_tool_message("call_keep", "still relevant result"),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "current turn result 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "current turn result 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "current turn result 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "current turn result 4"),
            make_user_message("what about this?"),
        ];

        let pruned = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));

        assert_eq!(pruned.pruned_count, 1);
        // call_old's content was offloaded into a recallable stub that contains
        // the full-text archive file_path.
        let stub = messages[1].content.as_str().unwrap();
        assert!(stub.contains("file_path:"));
        // The full text really is on disk and recallable (the stub itself may
        // contain head/tail previews, so we do not assert "original text absent").
        let path_line = stub
            .lines()
            .find_map(|line| line.trim().strip_prefix("- file_path: "))
            .expect("stub must carry an archived file_path");
        let archived = std::fs::read_to_string(path_line.trim()).unwrap();
        assert!(archived.contains("very long outdated result that should be pruned"));
        // call_keep's content unchanged (count < threshold)
        assert_eq!(
            messages[3].content.as_str().unwrap(),
            "still relevant result"
        );
        // Recent tool window content unchanged
        assert_eq!(
            messages[5].content.as_str().unwrap(),
            "current turn result 1"
        );
        // user message unchanged
        assert_eq!(messages[12].content.as_str().unwrap(), "what about this?");

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    /// Safety floor: with no archive directory (overflow_dir=None), never prune
    /// — keep the full text as is.
    #[test]
    fn test_apply_pruning_skips_without_overflow_dir() {
        let mut marks = FxHashMap::default();
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", "irrecoverable if dropped"),
            make_assistant_tool_call("call_r1", "execute_command"),
            make_tool_message("call_r1", "recent 1"),
            make_assistant_tool_call("call_r2", "execute_command"),
            make_tool_message("call_r2", "recent 2"),
            make_assistant_tool_call("call_r3", "execute_command"),
            make_tool_message("call_r3", "recent 3"),
            make_assistant_tool_call("call_r4", "execute_command"),
            make_tool_message("call_r4", "recent 4"),
            make_assistant_tool_call("call_r5", "execute_command"),
            make_tool_message("call_r5", "recent 5"),
        ];

        let pruned = apply_pruning(&mut messages, &marks, None);

        assert_eq!(pruned.pruned_count, 0);
        assert_eq!(
            messages[1].content.as_str().unwrap(),
            "irrecoverable if dropped"
        );
    }

    /// Idempotence: re-pruning the same pruned message keeps the stub text
    /// stable across rounds (protecting the prompt cache).
    #[test]
    fn test_apply_pruning_is_idempotent_across_turns() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);

        let build = || {
            vec![
                make_assistant_tool_call("call_old", "read_file"),
                make_tool_message("call_old", &"stable archived body\n".repeat(100)),
                make_assistant_tool_call("call_r1", "execute_command"),
                make_tool_message("call_r1", "recent 1"),
                make_assistant_tool_call("call_r2", "execute_command"),
                make_tool_message("call_r2", "recent 2"),
                make_assistant_tool_call("call_r3", "execute_command"),
                make_tool_message("call_r3", "recent 3"),
                make_assistant_tool_call("call_r4", "execute_command"),
                make_tool_message("call_r4", "recent 4"),
                make_assistant_tool_call("call_r5", "execute_command"),
                make_tool_message("call_r5", "recent 5"),
            ]
        };

        let mut m1 = build();
        apply_pruning(&mut m1, &marks, Some(overflow_dir.as_path()));
        let stub1 = m1[1].content.as_str().unwrap().to_string();

        let mut m2 = build();
        apply_pruning(&mut m2, &marks, Some(overflow_dir.as_path()));
        let stub2 = m2[1].content.as_str().unwrap().to_string();

        assert!(stub1.contains("file_path:"));
        assert_eq!(stub1, stub2, "stub must be stable across turns");

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_apply_pruning_protects_recent_tool_groups() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        marks.insert("call_last".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_prev", "execute_command"),
            make_tool_message("call_prev", "old result"),
            make_assistant_tool_call("call_last", "execute_command"),
            make_tool_message("call_last", "most recent result"),
        ];

        let pruned = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));

        // The most recent complete tool group containing call_last is protected
        // and not pruned.
        assert_eq!(pruned.pruned_count, 0);
        assert_eq!(messages[3].content.as_str().unwrap(), "most recent result");

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_apply_pruning_empty_marks() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![make_tool_message("call_1", "result")];

        let pruned = apply_pruning(
            &mut messages,
            &FxHashMap::default(),
            Some(overflow_dir.as_path()),
        );
        assert_eq!(pruned.pruned_count, 0);
        assert_eq!(messages[0].content.as_str().unwrap(), "result");
    }

    #[test]
    fn test_apply_pruning_never_touches_user_or_assistant() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        // Even when user/assistant messages carry a matching "tool_call_id",
        // they are not pruned
        marks.insert("call_1".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_2".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_user_message("important user question"),
            make_assistant_message("important assistant response"),
            make_assistant_tool_call("call_1", "execute_command"),
            make_tool_message("call_1", &"outdated tool result\n".repeat(100)),
            make_assistant_tool_call("call_2", "execute_command"),
            make_tool_message("call_2", "current tool result"),
            make_assistant_tool_call("call_3", "execute_command"),
            make_tool_message("call_3", "recent tool result 3"),
            make_assistant_tool_call("call_4", "execute_command"),
            make_tool_message("call_4", "recent tool result 4"),
            make_assistant_tool_call("call_5", "execute_command"),
            make_tool_message("call_5", "recent tool result 5"),
        ];

        let pruned = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));

        assert_eq!(pruned.pruned_count, 1);
        assert_eq!(
            messages[0].content.as_str().unwrap(),
            "important user question"
        );
        assert_eq!(
            messages[1].content.as_str().unwrap(),
            "important assistant response"
        );
        assert!(messages[3].content.as_str().unwrap().contains("file_path:"));

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_active_prunable_tool_ids_excludes_recent_groups_and_non_compressible_tools() {
        let messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"old command output\n".repeat(500)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];

        let ids = active_prunable_tool_ids(&messages);

        assert_eq!(ids.len(), 1);
        assert!(ids.contains("call_old"));
        assert!(!ids.contains("call_plan"));
        assert!(!ids.contains("call_recent_1"));
    }

    /// Decoupling invariant: `read_file` declares `lossy_compress: Never` but
    /// `prune: Allow`, so although it is "not lossy-compressible", its stale old
    /// results may still be pruned under LLM guidance. `plan` declares
    /// `prune: Never` and never becomes a pruning candidate.
    #[test]
    fn test_active_prunable_allows_read_file_but_protects_plan() {
        let messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_read", "read_file"),
            make_tool_message("call_read", &"old file contents already used\n".repeat(500)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
            make_assistant_tool_call("call_recent_5", "execute_command"),
            make_tool_message("call_recent_5", "recent 5"),
            make_assistant_tool_call("call_recent_6", "execute_command"),
            make_tool_message("call_recent_6", "recent 6"),
        ];

        let ids = active_prunable_tool_ids(&messages);

        // read_file is now prunable (it would have been excluded under the old
        // behavior).
        assert!(ids.contains("call_read"));
        // plan remains protected by its registration policy and is never pruned.
        assert!(!ids.contains("call_plan"));
    }

    #[test]
    fn test_apply_pruning_protects_non_compressible_tools() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        marks.insert("call_plan".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"old command output\n".repeat(1_000)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
            make_assistant_tool_call("call_recent_5", "execute_command"),
            make_tool_message("call_recent_5", "recent 5"),
            make_assistant_tool_call("call_recent_6", "execute_command"),
            make_tool_message("call_recent_6", "recent 6"),
        ];

        let pruned = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));

        assert_eq!(pruned.pruned_count, 1);
        assert_eq!(messages[1].content.as_str().unwrap(), "task plan");
        assert!(messages[3].content.as_str().unwrap().contains("file_path:"));

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_should_inject_prune_prompt() {
        let mut messages = vec![make_user_message("long dialog without tools")];
        assert!(!should_inject_prune_prompt(&messages));

        for index in 0..4 {
            let id = format!("call_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, &"recent result ".repeat(500)));
        }
        assert!(
            !should_inject_prune_prompt(&messages),
            "the four protected recent groups are not prune candidates"
        );

        messages.push(make_assistant_tool_call("call_4", "execute_command"));
        messages.push(make_tool_message("call_4", &"newest result ".repeat(500)));
        assert!(
            should_inject_prune_prompt(&messages),
            "once a fifth group makes an old result eligible, inject the protocol"
        );
    }

    #[test]
    fn test_prune_protocol_activates_at_same_turn_request_boundary_once() {
        let mut messages = vec![make_user_message("same-turn request")];
        for index in 0..4 {
            let id = format!("call_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, &"recent result ".repeat(500)));
        }

        assert!(!ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));
        messages.push(make_assistant_tool_call("call_4", "execute_command"));
        messages.push(make_tool_message("call_4", &"newest result ".repeat(500)));
        assert!(ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));
        assert!(!ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));
        assert_eq!(
            messages
                .iter()
                .filter(|message| is_prune_protocol_message(message))
                .count(),
            1
        );
    }

    #[test]
    fn test_prepare_request_projection_prunes_without_mutating_canonical_copy() {
        let overflow_dir = make_overflow_dir();
        let mut request_messages = vec![make_user_message("system-sized request")];
        let old_result = "old result that reached the prune threshold\n".repeat(1_000);
        request_messages.extend([
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &old_result),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            request_messages.push(make_assistant_tool_call(&id, "execute_command"));
            request_messages.push(make_tool_message(&id, "recent result"));
        }
        let canonical_messages = request_messages.clone();
        let marks = [("call_old".to_string(), PRUNE_THRESHOLD)]
            .into_iter()
            .collect();

        let first =
            prepare_request_projection(&mut request_messages, &marks, Some(overflow_dir.as_path()));
        let second =
            prepare_request_projection(&mut request_messages, &marks, Some(overflow_dir.as_path()));

        assert_eq!(first.pruned_count, 1);
        assert_eq!(second.pruned_count, 0);
        assert_eq!(
            canonical_messages[2].content.as_str(),
            Some(old_result.as_str())
        );
        let pruned_content = request_messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call_old"))
            .and_then(|message| message.content.as_str())
            .expect("pruned tool response should remain in the request projection");
        assert!(pruned_content.contains("file_path:"));
        assert_eq!(
            request_messages
                .iter()
                .filter(|message| is_prune_protocol_message(message))
                .count(),
            0,
            "after the only old candidate is pruned, the protocol prompt is no longer needed"
        );

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_pruning_never_replaces_short_result_with_longer_stub() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", "ok"),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        let marks = [("call_old".to_string(), PRUNE_THRESHOLD)]
            .into_iter()
            .collect();

        let report = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));

        assert_eq!(report.pruned_count, 0);
        assert_eq!(messages[2].content.as_str(), Some("ok"));
        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_pruned_stub_is_not_an_active_candidate() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"old result ".repeat(200)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        let marks = [("call_old".to_string(), PRUNE_THRESHOLD)]
            .into_iter()
            .collect();
        assert_eq!(
            apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path())).pruned_count,
            1
        );

        assert!(!active_prunable_tool_ids(&messages).contains("call_old"));
        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_prune_threshold_is_reasonable() {
        // Ensure the threshold is neither 0 (would fire every round) nor above
        // 10 (too conservative)
        assert!(PRUNE_THRESHOLD >= 1);
        assert!(PRUNE_THRESHOLD <= 10);
    }

    #[test]
    fn test_message_count_after_pruning_unchanged() {
        let overflow_dir = make_overflow_dir();
        let mut marks = FxHashMap::default();
        marks.insert("call_1".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_2".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_3".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_1", "execute_command"),
            make_tool_message("call_1", &"result 1\n".repeat(100)),
            make_assistant_tool_call("call_2", "execute_command"),
            make_tool_message("call_2", &"result 2\n".repeat(100)),
            make_assistant_tool_call("call_3", "execute_command"),
            make_tool_message("call_3", &"result 3\n".repeat(100)),
            make_assistant_tool_call("call_4", "execute_command"),
            make_tool_message("call_4", "old unmarked result"),
            make_assistant_tool_call("call_5", "execute_command"),
            make_tool_message("call_5", "recent result 5"),
            make_assistant_tool_call("call_6", "execute_command"),
            make_tool_message("call_6", "recent result 6"),
            make_assistant_tool_call("call_7", "execute_command"),
            make_tool_message("call_7", "recent result 7"),
            make_assistant_tool_call("call_8", "execute_command"),
            make_tool_message("call_8", "recent result 8"),
            make_assistant_tool_call("call_9", "execute_command"),
            make_tool_message("call_9", "recent result 9"),
            make_assistant_tool_call("call_10", "execute_command"),
            make_tool_message("call_10", "recent result 10"),
        ];

        let len_before = messages.len();
        let pruned = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));
        let len_after = messages.len();

        assert_eq!(len_before, len_after);
        assert_eq!(pruned.pruned_count, 3); // the most recent 4 tool groups are protected

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_protocol_prompt_lists_candidates_with_marks() {
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_big", "read_file"),
            make_tool_message("call_big", &"x".repeat(20_000)),
            make_assistant_tool_call("call_small_old", "execute_command"),
            make_tool_message("call_small_old", &"y".repeat(5_000)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        let marks = [("call_small_old".to_string(), 1u8)].into_iter().collect();

        assert!(ensure_prune_protocol_prompt(&mut messages, &marks));
        let prompt = messages
            .iter()
            .find(|message| is_prune_protocol_message(message))
            .and_then(|message| message.content.as_str())
            .expect("protocol prompt injected");
        assert!(prompt.starts_with(PRUNE_PROTOCOL_PROMPT));
        let big_pos = prompt.find("call_big").expect("large candidate listed");
        let small_pos = prompt
            .find("call_small_old")
            .expect("small candidate listed");
        assert!(big_pos < small_pos, "largest candidates are listed first");
        assert!(
            prompt.contains("marks 0/1"),
            "very large result shows the single-mark threshold"
        );
        assert!(
            prompt.contains("marks 1/2"),
            "small result shows the accumulated counter against the normal threshold"
        );
        assert!(
            !prompt.contains("call_recent_0"),
            "protected recent results are not listed as candidates"
        );
    }

    #[test]
    fn test_ensure_refreshes_candidate_list_in_place() {
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"x".repeat(5_000)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        assert!(ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));

        let prompt_before = messages
            .iter()
            .find(|message| is_prune_protocol_message(message))
            .and_then(|message| message.content.as_str())
            .expect("protocol prompt injected")
            .to_string();
        assert!(prompt_before.contains("marks 0/2"));

        // A later round of the same turn: the model has marked call_old once,
        // so the existing protocol message must be refreshed in place (updated
        // counter), never duplicated.
        let marks = [("call_old".to_string(), 1u8)].into_iter().collect();
        assert!(!ensure_prune_protocol_prompt(&mut messages, &marks));
        assert_eq!(
            messages
                .iter()
                .filter(|message| is_prune_protocol_message(message))
                .count(),
            1,
            "refresh must never duplicate the protocol message"
        );
        let prompt = messages
            .iter()
            .find(|message| is_prune_protocol_message(message))
            .and_then(|message| message.content.as_str())
            .expect("protocol prompt still present");
        assert!(prompt.contains("marks 1/2"));
        assert_ne!(prompt_before, prompt);
    }

    #[test]
    fn test_protocol_message_removed_when_no_candidates_remain() {
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"x".repeat(5_000)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        assert!(ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));
        assert_eq!(
            messages
                .iter()
                .filter(|message| is_prune_protocol_message(message))
                .count(),
            1
        );

        // A later round of the same turn: the old result is gone (offloaded or
        // rewritten), so no prunable candidate remains. The stale protocol
        // message must be removed, not left behind with an outdated list.
        messages.retain(|message| message.tool_call_id.as_deref() != Some("call_old"));
        assert!(!ensure_prune_protocol_prompt(
            &mut messages,
            &FxHashMap::default()
        ));
        assert_eq!(
            messages
                .iter()
                .filter(|message| is_prune_protocol_message(message))
                .count(),
            0,
            "protocol message with an empty candidate list is removed"
        );
    }

    #[test]
    fn test_single_mark_offloads_very_large_result() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_huge", "execute_command"),
            make_tool_message("call_huge", &"z".repeat(PRUNE_SINGLE_MARK_OFFLOAD_CHARS)),
            make_assistant_tool_call("call_small", "execute_command"),
            make_tool_message("call_small", &"w".repeat(5_000)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        let marks = [
            ("call_huge".to_string(), 1u8),
            ("call_small".to_string(), 1u8),
        ]
        .into_iter()
        .collect();

        let report = apply_pruning(&mut messages, &marks, Some(overflow_dir.as_path()));
        assert_eq!(
            report.pruned_count, 1,
            "only the very large result offloads after a single mark"
        );
        let huge_content = messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call_huge"))
            .and_then(|message| message.content.as_str())
            .expect("huge result stays in place");
        assert!(
            huge_content.contains("file_path:"),
            "offloaded to a recall stub"
        );
        let small_content = messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call_small"))
            .and_then(|message| message.content.as_str())
            .expect("small result stays in place");
        assert_eq!(
            small_content,
            "w".repeat(5_000),
            "small result still waits for the normal threshold"
        );
        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    #[test]
    fn test_needed_marks_for_matches_size_rule() {
        let messages = vec![
            make_assistant_tool_call("call_big", "execute_command"),
            make_tool_message("call_big", &"x".repeat(PRUNE_SINGLE_MARK_OFFLOAD_CHARS)),
            make_assistant_tool_call("call_small", "execute_command"),
            make_tool_message("call_small", &"y".repeat(5_000)),
        ];
        assert_eq!(needed_marks_for(&messages, "call_big"), 1);
        assert_eq!(needed_marks_for(&messages, "call_small"), PRUNE_THRESHOLD);
        assert_eq!(
            needed_marks_for(&messages, "call_missing"),
            PRUNE_THRESHOLD,
            "unknown ids fall back to the default threshold"
        );
    }

    #[test]
    fn test_explain_rejected_prune_mark() {
        let mut messages = vec![make_user_message("request")];
        messages.extend([
            make_assistant_tool_call("call_small", "execute_command"),
            make_tool_message("call_small", &"x".repeat(100)),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"y".repeat(5_000)),
        ]);
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }

        assert_eq!(
            explain_rejected_prune_mark(&messages, "call_unknown"),
            Some("no such tool result in the current context")
        );
        assert_eq!(
            explain_rejected_prune_mark(&messages, "call_recent_0"),
            Some("inside the recent-results protection window")
        );
        assert_eq!(
            explain_rejected_prune_mark(&messages, "call_small"),
            Some("below the minimum size for pruning")
        );
        assert_eq!(
            explain_rejected_prune_mark(&messages, "call_old"),
            None,
            "eligible ids get no rejection reason"
        );
    }
}
