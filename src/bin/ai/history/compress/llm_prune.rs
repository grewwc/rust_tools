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
//! Model marks are optional for the model to emit, so the runtime also reclaims on its own once the
//! eligible backlog exceeds [`PRESSURE_RECLAIM_MIN_TOTAL_CHARS`]: it tops the largest candidates up to
//! their required count and offloads them through the same lossless path
//! ([`reclaim_under_pressure_authorized`]). Every safety guarantee below is enforced by the shared
//! eligibility check rather than by the mark path, so it holds for reclaimed items as well.
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
//! 7. Tools whose policy defers to session state (`ToolPrunePolicy::AfterIntegration`, i.e. subagent
//!    results) become eligible only while [`PruneAuthorization`] proves the durable ledger recorded
//!    every task the result refers to as integrated. An unreadable ledger authorizes nothing, so those
//!    results stay inline rather than being unloaded on an unverified assumption.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::history::types::Message;

use super::PruneAuthorization;
use super::tool_overflow::{
    build_tool_call_arguments_index, build_tool_call_name_index, build_tool_overflow_recall_lines,
    is_preserved_tool_overflow_stub, preserve_pruned_tool_result_stable,
};

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

/// Total inline size of the currently eligible tool evidence above which the
/// runtime reclaims the largest candidates on its own, without waiting for a
/// model mark (see [`reclaim_under_pressure_authorized`]).
///
/// Marks stay the fine-grained signal; this is the fallback for models that
/// rarely emit them, where the eligible backlog otherwise grows while every
/// request keeps re-sending it in full. The threshold is deliberately
/// conservative (≈16K tokens of eligible evidence), so ordinary sessions keep
/// the marking contract unchanged.
const PRESSURE_RECLAIM_MIN_TOTAL_CHARS: usize = 65_536;

/// Per-request cap on how many candidates one pressure reclaim offloads, kept
/// equal to [`PRUNE_PROMPT_MAX_CANDIDATES`]: the runtime reclaims at most the
/// same largest items the model itself was shown for this request.
const PRESSURE_RECLAIM_MAX_ITEMS: usize = PRUNE_PROMPT_MAX_CANDIDATES;

/// Per-request cap on reclaimed characters. The largest candidate is always
/// included (so one oversized item can never block reclamation); capping every
/// request bounds how much of the projection prefix — and with it the prompt
/// cache — a single request rewrites, and successive requests converge until
/// the backlog falls below the trigger.
const PRESSURE_RECLAIM_MAX_CHARS: usize = 32_768;

/// The pruning-protocol instructions injected into the system prompt.
/// Kept short to avoid consuming too many tokens.
pub(crate) const PRUNE_PROTOCOL_PROMPT: &str = include_str!("prompts/prune_protocol.md");

/// Returns whether messages of this role are protected (never pruned).
fn is_protected_role(role: &str) -> bool {
    !matches!(role, "tool")
    // tool is not protected; all other roles are
}

/// Same as above, written more clearly.
fn is_prunable_message(msg: &Message) -> bool {
    msg.role == "tool" && msg.tool_call_id.is_some()
}

/// Every id a prune mark written in this request can name: the tool_call ids the
/// projection carries (call side and raw result side) plus the fold ids of the
/// folded notes in it.
///
/// Identity comes from the context rather than from the token's spelling, so a
/// provider-issued id outside the ASCII id shape — or longer than the shape bound —
/// still resolves; the shape rule only classifies tokens that name nothing here.
pub(crate) fn nameable_prune_ids(messages: &[Message]) -> FxHashSet<String> {
    let mut ids = FxHashSet::default();
    for message in messages {
        if let Some(id) = message.tool_call_id.as_deref() {
            ids.insert(id.to_string());
        }
        if let Some(calls) = message.tool_calls.as_deref() {
            for call in calls {
                ids.insert(call.id.clone());
            }
        }
        if let Some(meta) = super::folded_prune::provenance(message) {
            // The fold note and its offloaded stub render the constituent call ids, so
            // those stay nameable as well: a mark on one reaches its own rejection
            // report instead of being read as prose.
            ids.extend(meta.source_ids().map(str::to_string));
            ids.insert(meta.id());
        }
    }
    ids
}

/// Parses prune marks from the hidden_meta of a model response.
///
/// hidden_meta may span multiple lines; lines starting with `prune:` are pruning
/// directives, the rest is regular self_note content (handled by the caller). The
/// `<<<prune:ids>>>` compatibility marker is accepted here as well, so a note that
/// mixes both forms still lands every mark.
///
/// A directive line counts only when its payload is nothing but id tokens, as
/// resolved by [`resolve_id_token`] against `known_ids`: a line mixing ids with
/// prose is kept as note text for [`unrecognized_prune_fragments`] to report,
/// instead of being split into garbage ids that look like rejected marks.
///
/// `known_ids` is the id set of the request the note belongs to
/// ([`nameable_prune_ids`]); it decides which tokens are ids.
///
/// Returns `(prune_ids, remaining_meta)`:
/// - `prune_ids`: the list of marked tool_call_ids
/// - `remaining_meta`: the hidden_meta left after removing the prune lines (for the self_note)
pub(crate) fn parse_prune_from_hidden_meta(
    hidden_meta: &str,
    known_ids: &FxHashSet<String>,
) -> (Vec<String>, String) {
    // Strip the compatibility marker first: the line scan below then only has to
    // deal with `prune:` lines, and a mixed-form note keeps every mark.
    let (mut prune_ids, hidden_meta) = parse_embedded_prune_directives(hidden_meta, known_ids);
    let hidden_meta = hidden_meta.as_str();
    let mut remaining_lines = Vec::new();
    let mut saw_prune = false;

    for line in hidden_meta.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("prune:")
            && let Some(ids) = directive_payload_ids(rest, known_ids)
        {
            saw_prune = true;
            prune_ids.extend(ids);
            continue;
        }
        if !trimmed.is_empty() {
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

/// Marker some models emit instead of the documented overflow channel (a `prune:`
/// line inside `<meta:self_note>`); accepting it keeps a mark written in this shape
/// from being dropped without a trace.
const EMBEDDED_PRUNE_OPEN: &str = "<<<prune:";

/// Closing marker of [`EMBEDDED_PRUNE_OPEN`].
const EMBEDDED_PRUNE_CLOSE: &str = ">>>";

/// Wrapping characters a model may put around an id when writing prose; stripped
/// before matching so `` `call_abc` `` and `(call_abc)` still resolve.
const ID_TOKEN_DECORATION: &[char] = &['`', '"', '\'', '*', '(', ')', '[', ']', '<', '>'];

/// Sentence punctuation glued to the last id of a directive line.
const ID_TOKEN_TRAILING: &[char] = &['.', ',', ';', ':', '!', '?'];

/// Resolves one comma-separated payload token to the id it names, `None` when the
/// token carries prose.
///
/// Membership in the ids the request carries ([`nameable_prune_ids`]) is
/// authoritative: an id the model can see in this request must resolve however it is
/// spelled. The shape rule only classifies tokens that name nothing in the context,
/// where an id-looking token is still reported as a rejected mark rather than read
/// as prose. The raw token is tried before the de-decorated form, so an id that
/// itself contains a decoration character is not mangled.
fn resolve_id_token<'a>(token: &'a str, known_ids: &FxHashSet<String>) -> Option<&'a str> {
    let raw = token.trim();
    if known_ids.contains(raw) {
        return Some(raw);
    }
    let normalized = normalize_id_token(raw);
    if known_ids.contains(normalized) {
        return Some(normalized);
    }
    id_shape_matches(normalized).then_some(normalized)
}

/// Strips the wrapping characters a model may put around an id when writing prose,
/// plus sentence punctuation glued to the end of a directive line, so
/// `` `call_abc` `` and `call_abc.` still resolve.
fn normalize_id_token(token: &str) -> &str {
    token
        .trim()
        .trim_matches(ID_TOKEN_DECORATION)
        .trim_end_matches(ID_TOKEN_TRAILING)
}

/// Whether a token that names nothing in the context can still be an id. A token
/// that cannot be an id must never be reported as one: quoting the protocol
/// otherwise turns whole sentences into "rejected marks".
fn id_shape_matches(token: &str) -> bool {
    /// Longer than any provider-issued tool_call_id; keeps a runaway token bounded.
    const MAX_ID_CHARS: usize = 128;
    !token.is_empty()
        && token.len() <= MAX_ID_CHARS
        && token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Splits a directive payload into ids, or returns `None` when the payload is empty
/// or any comma-separated token is not an id. All-or-nothing keeps a prose fragment
/// from being applied as a partial mark and keeps the reported ids trustworthy.
fn directive_payload_ids(payload: &str, known_ids: &FxHashSet<String>) -> Option<Vec<String>> {
    let tokens = payload
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    let ids = tokens
        .into_iter()
        .map(|token| resolve_id_token(token, known_ids))
        .collect::<Option<Vec<_>>>()?;
    Some(ids.into_iter().map(str::to_string).collect())
}

/// Extracts `<<<prune:ids>>>` directives from model-authored text.
///
/// Only a line that starts with the marker is a directive, so the marker merely
/// quoted inside a sentence never turns into a mark. Within such a line every
/// well-formed span whose payload is a clean id list is consumed; a span that is
/// malformed or carries prose stays in the text for [`unrecognized_prune_fragments`]
/// to report.
///
/// Returns `(ids, cleaned_text)`: the marked ids in order, and the text with the
/// consumed spans removed (a line that did not carry anything else disappears
/// instead of leaving a blank gap in the narration).
///
/// `known_ids` is the id set of the request the text belongs to
/// ([`nameable_prune_ids`]); it decides which tokens are ids.
pub(crate) fn parse_embedded_prune_directives(
    text: &str,
    known_ids: &FxHashSet<String>,
) -> (Vec<String>, String) {
    let mut ids = Vec::new();
    let mut cleaned = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !line.trim_start().starts_with(EMBEDDED_PRUNE_OPEN) {
            cleaned.push_str(line);
            continue;
        }
        let mut rest = line;
        let mut line_out = String::with_capacity(line.len());
        let mut consumed_span = false;
        let mut kept_span = false;
        while let Some(start) = rest.find(EMBEDDED_PRUNE_OPEN) {
            let after_open = &rest[start + EMBEDDED_PRUNE_OPEN.len()..];
            let Some(end) = after_open.find(EMBEDDED_PRUNE_CLOSE) else {
                // Unclosed marker: keep it verbatim so it can be reported.
                kept_span = true;
                break;
            };
            let Some(span_ids) = directive_payload_ids(&after_open[..end], known_ids) else {
                // Payload with prose in it: same, keep the span for the report.
                kept_span = true;
                line_out.push_str(&rest[..start + EMBEDDED_PRUNE_OPEN.len()]);
                rest = after_open;
                continue;
            };
            line_out.push_str(&rest[..start]);
            ids.extend(span_ids);
            consumed_span = true;
            rest = &after_open[end + EMBEDDED_PRUNE_CLOSE.len()..];
            // Removing a span between two spaces would otherwise leave a double space.
            if line_out.ends_with(' ') && rest.starts_with(' ') {
                rest = &rest[1..];
            }
        }
        line_out.push_str(rest);
        if consumed_span && !kept_span && line_out.trim().is_empty() {
            continue;
        }
        cleaned.push_str(&line_out);
    }
    (ids, cleaned)
}

/// Directive-looking text that no accepted form parses, so the runtime can quote it
/// back to the model instead of dropping the attempt silently.
///
/// Recognizes the two shapes of an attempted mark left in the text: a line starting
/// with the `<<<prune:` marker whose span did not parse (missing closer, prose in the
/// payload), and a `prune:` line whose payload still holds an id. A marker merely
/// quoted inside a sentence is not an attempt and stays unreported.
///
/// An id counts when [`resolve_id_token`] resolves it against `known_ids`.
pub(crate) fn unrecognized_prune_fragments(
    text: &str,
    known_ids: &FxHashSet<String>,
) -> Vec<String> {
    /// Cap on reported snippets; the notice stays bounded and shows the shape.
    const MAX_FRAGMENTS: usize = 4;
    /// Cap on a reported snippet's length.
    const MAX_FRAGMENT_CHARS: usize = 80;
    let mut fragments: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        let attempted = if trimmed.starts_with(EMBEDDED_PRUNE_OPEN) {
            true
        } else {
            trimmed.strip_prefix("prune:").is_some_and(|payload| {
                payload
                    .split(',')
                    .map(str::trim)
                    .any(|token| resolve_id_token(token, known_ids).is_some())
            })
        };
        if !attempted {
            continue;
        }
        let snippet = trimmed.chars().take(MAX_FRAGMENT_CHARS).collect::<String>();
        if !fragments.iter().any(|existing| *existing == snippet) {
            fragments.push(snippet);
        }
        if fragments.len() >= MAX_FRAGMENTS {
            break;
        }
    }
    fragments
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
///
/// Returns whether the mark map actually changed (counter increment, new entry,
/// or removed entry), so callers can persist without cloning the whole map just
/// to detect a diff.
pub(crate) fn update_prune_marks(
    current_marks: &mut FxHashMap<String, u8>,
    prune_ids: &[String],
    active_prunable_tool_ids: &FxHashSet<String>,
) -> bool {
    let marked_ids = prune_ids
        .iter()
        .filter(|id| active_prunable_tool_ids.contains(*id))
        .cloned()
        .collect::<FxHashSet<_>>();

    // Increment the counters of marked tools (monotonic accumulation; a silent
    // round has empty marked_ids and is a no-op).
    let mut changed = false;
    for id in marked_ids {
        let count = current_marks.entry(id).or_insert(0);
        let updated = count.saturating_add(1);
        if *count != updated {
            changed = true;
        }
        *count = updated;
    }

    // Clean up entries with a zero count, no longer in the current context, or
    // excluded by the protection policy.
    let len_before_cleanup = current_marks.len();
    current_marks.retain(|id, v| *v > 0 && active_prunable_tool_ids.contains(id));
    changed |= current_marks.len() != len_before_cleanup;
    changed
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
/// - Results of tools declaring `prune: AfterIntegration` (subagent results)
///   additionally require `authorization`, which carries the ledger-derived
///   integration state this function cannot read from `messages`.
pub(crate) fn active_prunable_tool_ids_authorized(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> FxHashSet<String> {
    let mut ids = active_raw_prunable_tool_ids(messages, authorization);
    ids.extend(
        super::folded_prune::candidates(messages)
            .into_iter()
            .map(|(_, meta)| meta.id()),
    );
    ids
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`active_prunable_tool_ids_authorized`].
#[cfg(test)]
pub(crate) fn active_prunable_tool_ids(messages: &[Message]) -> FxHashSet<String> {
    active_prunable_tool_ids_authorized(messages, &PruneAuthorization::default())
}

fn active_raw_prunable_tool_ids(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> FxHashSet<String> {
    let protected_ids = protected_tool_call_ids(messages, authorization);
    let id_to_tool_name = build_tool_call_name_index(messages);
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
            // Fold consent must never become raw-result consent after replay.
            if id.starts_with(super::folded_prune::FOLD_ID_PREFIX) || protected_ids.contains(id) {
                return None;
            }
            // A tool whose policy defers to session state consults the live
            // authorization; an unresolvable name never reaches this point
            // because `protected_tool_call_ids` already protects it.
            let name = id_to_tool_name.get(id)?;
            super::folded_prune::tool_allows_explicit_prune_now(
                name,
                message.content.as_str()?,
                authorization,
            )
            .then(|| id.clone())
        })
        .collect()
}

/// Retention is separate from authorization: a completed offload must survive
/// silent rounds and canonical replay without making its stub a new candidate.
pub(crate) fn retained_prune_ids_authorized(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> FxHashSet<String> {
    // An id several items answer to cannot be resolved by a textual mark, so its
    // accumulated consent is withdrawn instead of being carried into a later state
    // where the id resolves again and would offload a note the model never saw.
    let ambiguous = super::folded_prune::ambiguous_fold_ids(messages);
    let mut ids = active_prunable_tool_ids_authorized(messages, authorization);
    ids.extend(
        super::folded_prune::canonical_fold_ids(messages)
            .into_iter()
            .filter(|id| !ambiguous.contains(id)),
    );
    let names = build_tool_call_name_index(messages);
    for message in messages {
        if let Some(meta) = super::folded_prune::provenance(message) {
            if meta.allows_prune() && !ambiguous.contains(&meta.id()) {
                ids.insert(meta.id());
            }
        }
        if message.role == "tool"
            && message
                .content
                .as_str()
                .is_some_and(is_preserved_tool_overflow_stub)
        {
            if let Some(id) = message.tool_call_id.as_ref() {
                // Retention follows how the stub was created, not current
                // eligibility: the stub no longer carries the task markers that
                // a conditional decision needs, and a conditional result could
                // only have reached a stub while it was authorized.
                if names
                    .get(id)
                    .is_some_and(|name| super::folded_prune::tool_retains_stub_mark(name))
                {
                    ids.insert(id.clone());
                }
            }
        }
    }
    ids
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`update_prune_marks_for_messages_authorized`].
#[cfg(test)]
pub(crate) fn retained_prune_ids(messages: &[Message]) -> FxHashSet<String> {
    retained_prune_ids_authorized(messages, &PruneAuthorization::default())
}

pub(crate) fn update_prune_marks_for_messages_authorized(
    current_marks: &mut FxHashMap<String, u8>,
    prune_ids: &[String],
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> bool {
    let active = active_prunable_tool_ids_authorized(messages, authorization);
    let accepted = prune_ids
        .iter()
        .filter(|id| active.contains(*id))
        .cloned()
        .collect::<Vec<_>>();
    update_prune_marks(
        current_marks,
        &accepted,
        &retained_prune_ids_authorized(messages, authorization),
    )
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`update_prune_marks_for_messages_authorized`].
#[cfg(test)]
pub(crate) fn update_prune_marks_for_messages(
    current_marks: &mut FxHashMap<String, u8>,
    prune_ids: &[String],
    messages: &[Message],
) -> bool {
    update_prune_marks_for_messages_authorized(
        current_marks,
        prune_ids,
        messages,
        &PruneAuthorization::default(),
    )
}

fn protected_tool_call_ids(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> FxHashSet<String> {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let mut protected = FxHashSet::default();
    for index in super::folded_prune::recent_group_indices(messages) {
        if let Some(calls) = &messages[index].tool_calls {
            protected.extend(calls.iter().map(|call| call.id.clone()));
        }
    }
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        if protected.contains(tool_call_id) {
            continue;
        }
        // Protection is decided per result, not per tool name: a tool whose
        // policy defers to session state (subagent results) is protected only
        // while the authorization withholds it. Unknown tools stay protected.
        if id_to_tool_name
            .get(tool_call_id)
            .is_none_or(|name| {
                !super::folded_prune::tool_allows_explicit_prune_now(
                    name,
                    message.content.as_str().unwrap_or_default(),
                    authorization,
                )
            })
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
/// checks mirror [`active_prunable_tool_ids_authorized`] but are ordered for message
/// quality (why exactly, not merely that it is ineligible).
pub(crate) fn explain_rejected_prune_mark_authorized(
    messages: &[Message],
    tool_call_id: &str,
    authorization: &PruneAuthorization,
) -> Option<&'static str> {
    if tool_call_id.starts_with(super::folded_prune::FOLD_ID_PREFIX) {
        if active_prunable_tool_ids_authorized(messages, authorization).contains(tool_call_id) {
            return None;
        }
        if super::folded_prune::ambiguous_fold_ids(messages).contains(tool_call_id) {
            return Some("fold id is shared by more than one evidence item");
        }
        return Some(
            "fold is protected, recent, already offloaded, too small, or lacks valid provenance",
        );
    }
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
    let content = message.content.as_str().unwrap_or_default();
    if let Some(name) = id_to_tool_name.get(tool_call_id)
        && !super::folded_prune::tool_allows_explicit_prune_now(name, content, authorization)
    {
        // Conditional tools report their own reason: their policy is not
        // "Never", it is "not yet" (see `ToolPrunePolicy::AfterIntegration`).
        return Some(if super::folded_prune::tool_prune_is_conditional(name) {
            "subagent result is not integrated yet"
        } else {
            "tool declares prune:Never"
        });
    }
    if protected_tool_call_ids(messages, authorization).contains(&tool_call_id.to_string()) {
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
    // [`active_prunable_tool_ids_authorized`] (it may gain conditions later).
    if active_prunable_tool_ids_authorized(messages, authorization).contains(tool_call_id) {
        None
    } else {
        Some("not currently eligible")
    }
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`explain_rejected_prune_mark_authorized`].
#[cfg(test)]
pub(crate) fn explain_rejected_prune_mark(
    messages: &[Message],
    tool_call_id: &str,
) -> Option<&'static str> {
    explain_rejected_prune_mark_authorized(messages, tool_call_id, &PruneAuthorization::default())
}

/// Pruning statistics of a single `apply_pruning` call, for the caller to
/// print a terminal notice.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PruneReport {
    /// Number of tool results or folded groups offloaded and inline-replaced
    /// with a recall stub.
    pub(crate) pruned_count: usize,
    /// Net characters freed (sum of original content lengths minus stub lengths).
    pub(crate) freed_chars: usize,
    /// Tool names involved (deduplicated, in first-appearance order).
    pub(crate) tools: Vec<String>,
    /// Subset of `pruned_count` that the runtime reclaimed under context
    /// pressure without a model mark (see [`reclaim_under_pressure_authorized`]),
    /// so a caller can attribute the offload honestly in its notice.
    pub(crate) pressure_count: usize,
}

impl PruneReport {
    /// Folds another report into this one, so a request that both applied model
    /// marks and reclaimed under pressure presents a single aggregate.
    pub(crate) fn merge(&mut self, other: PruneReport) {
        self.pruned_count += other.pruned_count;
        self.freed_chars += other.freed_chars;
        self.pressure_count += other.pressure_count;
        for tool in other.tools {
            if !self.tools.contains(&tool) {
                self.tools.push(tool);
            }
        }
    }
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
pub(crate) fn apply_pruning_authorized(
    messages: &mut [Message],
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
    authorization: &PruneAuthorization,
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
    // Replaying an accepted raw decision does not depend on the candidate-list
    // size cutoff. Recheck raw authorization without importing fold consent.
    let protected_ids = protected_tool_call_ids(messages, authorization);

    for msg in messages.iter_mut() {
        if !is_prunable_message(msg) {
            continue;
        }

        let Some(tool_call_id) = msg.tool_call_id.clone() else {
            continue;
        };

        if tool_call_id.starts_with(super::folded_prune::FOLD_ID_PREFIX)
            || protected_ids.contains(&tool_call_id)
        {
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
        // Re-derive session authorization at offload time, not only when marks
        // were recorded: a mark restored from session state (or written before
        // this policy existed) must not unload a subagent result whose task was
        // never integrated.
        if !super::folded_prune::tool_allows_explicit_prune_now(
            id_to_tool_name
                .get(&tool_call_id)
                .map(String::as_str)
                .unwrap_or("tool"),
            content,
            authorization,
        ) {
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

    // A folded group always requires two independent model responses, even if
    // its preview is large. Ordinary notes have no provenance and cannot enter.
    if let Some(dir) = overflow_dir {
        for (index, meta) in super::folded_prune::candidates(messages) {
            if prune_marks.get(&meta.id()).copied().unwrap_or(0) < PRUNE_THRESHOLD {
                continue;
            }
            let label = meta.label();
            if let Some(freed) = super::folded_prune::offload(&mut messages[index], meta, dir) {
                report.pruned_count += 1;
                report.freed_chars += freed;
                if !report.tools.contains(&label) {
                    report.tools.push(label);
                }
            }
        }
    }
    report
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`apply_pruning_authorized`].
#[cfg(test)]
pub(crate) fn apply_pruning(
    messages: &mut [Message],
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
) -> PruneReport {
    apply_pruning_authorized(
        messages,
        prune_marks,
        overflow_dir,
        &PruneAuthorization::default(),
    )
}

/// Reclaims the largest eligible tool evidence under context pressure, without
/// waiting for the model to mark it.
///
/// Model marks remain the fine-grained signal, but emitting them is optional
/// and some models never do; without this fallback the eligible backlog grows
/// while every request keeps re-sending it in full. The candidate set is
/// exactly [`active_prunable_tool_ids_authorized`], so the most recent
/// `KEEP_RECENT_TOOL_GROUPS` groups, `prune: Never` tools, and un-integrated
/// subagent results keep every protection they have under model marks: the
/// runtime only decides *when* an eligible item is reclaimed, never *whether*
/// it is protected.
///
/// The offload itself runs through [`apply_pruning_authorized`] with a local
/// marks map that tops each selected id up to its required count
/// ([`needed_marks`] for raw results, [`PRUNE_THRESHOLD`] for folded groups).
/// The caller's persisted mark map is never touched: a reclaim leaves no
/// consent behind that a later request could mistake for a model mark.
///
/// At most [`PRESSURE_RECLAIM_MAX_ITEMS`] items and
/// [`PRESSURE_RECLAIM_MAX_CHARS`] characters are reclaimed per request (largest
/// first, and always at least one item, so a single oversized candidate cannot
/// block reclamation). Successive requests converge until the eligible backlog
/// falls below [`PRESSURE_RECLAIM_MIN_TOTAL_CHARS`].
///
/// Returns a report whose `pressure_count` attributes the offloads to this
/// path. Without an archive directory it reclaims nothing: the lossless floor
/// of [`apply_pruning_authorized`] applies here too.
pub(crate) fn reclaim_under_pressure_authorized(
    messages: &mut [Message],
    overflow_dir: Option<&Path>,
    authorization: &PruneAuthorization,
) -> PruneReport {
    // Same safety floor as `apply_pruning_authorized`: with no archive
    // directory there is no lossless recall, so nothing may be offloaded.
    if overflow_dir.is_none() {
        return PruneReport::default();
    }
    let mut sized = eligible_candidate_sizes(messages, authorization);
    let total_chars: usize = sized.iter().map(|(_, chars)| *chars).sum();
    if total_chars < PRESSURE_RECLAIM_MIN_TOTAL_CHARS {
        return PruneReport::default();
    }
    // Largest first (frees the most context); tie-break by id for determinism.
    sized.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut top_up: FxHashMap<String, u8> = FxHashMap::default();
    let mut selected_chars = 0usize;
    for (id, chars) in sized {
        if top_up.len() >= PRESSURE_RECLAIM_MAX_ITEMS
            || (!top_up.is_empty() && selected_chars >= PRESSURE_RECLAIM_MAX_CHARS)
        {
            break;
        }
        let marks = if id.starts_with(super::folded_prune::FOLD_ID_PREFIX) {
            PRUNE_THRESHOLD
        } else {
            needed_marks(chars)
        };
        top_up.insert(id, marks);
        selected_chars = selected_chars.saturating_add(chars);
    }

    let mut report = apply_pruning_authorized(messages, &top_up, overflow_dir, authorization);
    report.pressure_count = report.pruned_count;
    report
}

/// Inline size (characters) of every eligible candidate in this projection:
/// raw results through their `tool_call_id`, folded notes through the note body
/// their provenance points at. Mirrors the candidate collection of
/// [`build_prune_protocol_prompt`], so pressure reclaim weighs the same items
/// the model was shown.
fn eligible_candidate_sizes(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> Vec<(String, usize)> {
    let active_ids = active_prunable_tool_ids_authorized(messages, authorization);
    let mut sizes: FxHashMap<String, usize> = FxHashMap::default();
    for message in messages {
        let (Some(id), Some(content)) = (message.tool_call_id.as_deref(), message.content.as_str())
        else {
            continue;
        };
        if active_ids.contains(id) {
            sizes.insert(id.to_string(), content.chars().count());
        }
    }
    for (index, meta) in super::folded_prune::candidates(messages) {
        let id = meta.id();
        if !active_ids.contains(&id) {
            continue;
        }
        if let Some(content) = messages[index].content.as_str() {
            sizes.insert(id, content.chars().count());
        }
    }
    sizes.into_iter().collect()
}

/// Injects the protocol only when the current request actually contains
/// prunable old tool results.
///
/// This tracks the capability boundary better than a fixed message-count
/// threshold: oversized old results in a short conversation get reclaimed
/// promptly, while long tool-less conversations waste no prompt tokens.
pub(crate) fn should_inject_prune_prompt_authorized(
    messages: &[Message],
    authorization: &PruneAuthorization,
) -> bool {
    !active_prunable_tool_ids_authorized(messages, authorization).is_empty()
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`should_inject_prune_prompt_authorized`].
#[cfg(test)]
pub(crate) fn should_inject_prune_prompt(messages: &[Message]) -> bool {
    should_inject_prune_prompt_authorized(messages, &PruneAuthorization::default())
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
    authorization: &PruneAuthorization,
) -> String {
    let active_ids = active_prunable_tool_ids_authorized(messages, authorization);
    if active_ids.is_empty() {
        return PRUNE_PROTOCOL_PROMPT.to_string();
    }
    let id_to_tool_name = build_tool_call_name_index(messages);
    let mut candidates: Vec<(String, String, usize, u8)> = messages
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
            Some((id.clone(), tool.to_string(), chars, needed_marks(chars)))
        })
        .collect();
    candidates.extend(super::folded_prune::candidates(messages).into_iter().map(
        |(index, meta)| {
            (
                meta.id(),
                meta.label(),
                messages[index].content.as_str().unwrap().chars().count(),
                PRUNE_THRESHOLD,
            )
        },
    ));
    // Largest first (most context freed); tie-break by id for determinism.
    candidates.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    let mut section = String::from(
        "\n### Prunable candidates in this request (id · tool · size · marks/threshold)\n",
    );
    for (id, tool, chars, threshold) in candidates.iter().take(PRUNE_PROMPT_MAX_CANDIDATES) {
        let count = prune_marks.get(id).copied().unwrap_or(0);
        section.push_str(&format!(
            "- {id} · {tool} · {chars} chars · marks {count}/{}\n",
            threshold
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
pub(crate) fn ensure_prune_protocol_prompt_authorized(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
    authorization: &PruneAuthorization,
) -> bool {
    if let Some(existing_idx) = messages
        .iter()
        .position(|message| is_prune_protocol_message(message))
    {
        if should_inject_prune_prompt_authorized(messages, authorization) {
            let prompt = build_prune_protocol_prompt(messages, prune_marks, authorization);
            // Refreshing identical text is not a mutation: callers reuse the
            // previous token estimate when nothing actually changed.
            if messages[existing_idx].content.as_str() == Some(prompt.as_str()) {
                return false;
            }
            messages[existing_idx].content = Value::String(prompt);
            return true;
        } else {
            // No prunable candidates remain in this projection: the candidate
            // list inside the protocol message would now name ids that are no
            // longer present, which misleads the model into re-marking them.
            // Remove the stale message instead of leaving it in place.
            messages.remove(existing_idx);
            return true;
        }
    }
    if !should_inject_prune_prompt_authorized(messages, authorization) {
        return false;
    }

    let insert_at = messages
        .iter()
        .position(|message| message.role != "system")
        .unwrap_or(messages.len());
    let prompt = build_prune_protocol_prompt(messages, prune_marks, authorization);
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

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`ensure_prune_protocol_prompt_authorized`].
#[cfg(test)]
pub(crate) fn ensure_prune_protocol_prompt(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
) -> bool {
    ensure_prune_protocol_prompt_authorized(
        messages,
        prune_marks,
        &PruneAuthorization::default(),
    )
}

/// Updates the transient projection before every model request: first
/// losslessly offload old tool results that reached the threshold, then inject
/// the protocol on demand.
///
/// Which results may be offloaded also depends on the session authorization:
/// tools whose policy defers to session state (subagent results) are unloaded
/// only while their evidence is integrated.
///
/// The caller must pass a request projection kept separate from the canonical
/// `turn_messages`.
pub(crate) fn prepare_request_projection_authorized(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
    authorization: &PruneAuthorization,
) -> PruneReport {
    let report = apply_pruning_authorized(
        messages.as_mut_slice(),
        prune_marks,
        overflow_dir,
        authorization,
    );
    ensure_prune_protocol_prompt_authorized(messages, prune_marks, authorization);
    report
}

/// Fail-closed entry point kept for tests: it authorizes no conditional result,
/// so production callers must use [`prepare_request_projection_authorized`].
#[cfg(test)]
pub(crate) fn prepare_request_projection(
    messages: &mut Vec<Message>,
    prune_marks: &FxHashMap<String, u8>,
    overflow_dir: Option<&Path>,
) -> PruneReport {
    prepare_request_projection_authorized(
        messages,
        prune_marks,
        overflow_dir,
        &PruneAuthorization::default(),
    )
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
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &FxHashSet::default());

        assert_eq!(ids, vec!["call_abc", "call_xyz"]);
        assert!(remaining.contains("Do: be concise"));
        assert!(remaining.contains("Avoid: verbosity"));
        assert!(!remaining.contains("prune:"));
    }

    #[test]
    fn test_parse_prune_only() {
        let hidden_meta = "prune:call_1,call_2";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &FxHashSet::default());

        assert_eq!(ids.len(), 2);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_no_prune() {
        let hidden_meta = "Do: be focused\nAvoid: tangents";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &FxHashSet::default());

        assert!(ids.is_empty());
        assert_eq!(remaining, "Do: be focused\nAvoid: tangents");
    }

    #[test]
    fn test_parse_empty() {
        let (ids, remaining) = parse_prune_from_hidden_meta("", &FxHashSet::default());
        assert!(ids.is_empty());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_prune_from_hidden_meta_accepts_compat_marker() {
        let hidden_meta = "<<<prune:call_a,call_b>>>\nDo: be concise";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &FxHashSet::default());

        assert_eq!(ids, vec!["call_a", "call_b"]);
        assert_eq!(remaining, "Do: be concise");
    }

    #[test]
    fn test_parse_prune_from_hidden_meta_keeps_mixed_payload_as_text() {
        // Ids mixed with prose are not applied and not invented: the line survives
        // for the rejection report instead of splitting into garbage ids.
        let hidden_meta = "prune:call_a, see the log\nkeep me";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &FxHashSet::default());

        assert!(ids.is_empty());
        assert_eq!(remaining, hidden_meta);
    }

    #[test]
    fn test_parse_embedded_prune_directives_directive_line_is_consumed() {
        let text = "<<<prune:call_1,call_2>>>\n\nNext paragraph.";
        let (ids, cleaned) = parse_embedded_prune_directives(text, &FxHashSet::default());

        assert_eq!(ids, vec!["call_1", "call_2"]);
        assert_eq!(cleaned, "\nNext paragraph.");
    }

    #[test]
    fn test_parse_embedded_prune_directives_inline_mention_is_not_a_directive() {
        let text = "The model wrote <<<prune:call_1>>> in its reply.";
        let (ids, cleaned) = parse_embedded_prune_directives(text, &FxHashSet::default());

        assert!(ids.is_empty());
        assert_eq!(cleaned, text);
    }

    #[test]
    fn test_parse_embedded_prune_directives_prose_payload_stays_for_report() {
        let text = "<<<prune:call_1, the older results>>>\n";
        let (ids, cleaned) = parse_embedded_prune_directives(text, &FxHashSet::default());

        assert!(ids.is_empty());
        assert_eq!(cleaned, text);
        assert_eq!(
            unrecognized_prune_fragments(text, &FxHashSet::default()).len(),
            1
        );
    }

    #[test]
    fn test_unrecognized_prune_fragments_reports_attempted_marks_only() {
        // An unclosed marker and a `prune:` line carrying ids plus prose are attempts.
        assert_eq!(
            unrecognized_prune_fragments("<<<prune:call_1", &FxHashSet::default()),
            vec!["<<<prune:call_1".to_string()]
        );
        assert_eq!(
            unrecognized_prune_fragments("prune:call_1, see the log", &FxHashSet::default()),
            vec!["prune:call_1, see the log".to_string()]
        );
        // Quoting the marker mid-sentence and `prune:` prose without ids are not.
        assert!(
            unrecognized_prune_fragments("write <<<prune:call_1>>> to mark", &FxHashSet::default())
                .is_empty()
        );
        assert!(
            unrecognized_prune_fragments("prune: see the protocol", &FxHashSet::default()).is_empty()
        );
    }

    #[test]
    fn test_known_ids_resolve_regardless_of_their_shape() {
        // A mark must land on an id the request carries even when the id sits outside
        // the fallback shape: providers are not bound by this parser's charset rule.
        let known: FxHashSet<String> = ["call:weird/0".to_string()].into_iter().collect();

        let (ids, remaining) = parse_prune_from_hidden_meta("prune:call:weird/0", &known);
        assert_eq!(ids, vec!["call:weird/0"]);
        assert!(remaining.is_empty());

        let (ids, cleaned) = parse_embedded_prune_directives("<<<prune:call:weird/0>>>", &known);
        assert_eq!(ids, vec!["call:weird/0"]);
        assert!(cleaned.is_empty());

        // In a request that does not carry the id the text names nothing; the marker
        // line is still reported as an attempt instead of vanishing silently.
        let empty = FxHashSet::default();
        let (ids, _) = parse_prune_from_hidden_meta("prune:call:weird/0", &empty);
        assert!(ids.is_empty());
        let (ids, _) = parse_embedded_prune_directives("<<<prune:call:weird/0>>>", &empty);
        assert!(ids.is_empty());
        assert_eq!(
            unrecognized_prune_fragments("<<<prune:call:weird/0>>>", &empty).len(),
            1
        );
    }

    #[test]
    fn test_known_id_keeps_its_decoration_characters_and_length() {
        let decorated = "call_(weird)";
        let long = "x".repeat(200);
        let known: FxHashSet<String> = [decorated.to_string(), long.clone()].into_iter().collect();

        // The raw token wins over the de-decorated form: trimming `(`/`)` would
        // otherwise corrupt an id that itself contains them.
        let (ids, _) = parse_prune_from_hidden_meta(&format!("prune:{decorated}"), &known);
        assert_eq!(ids, vec![decorated]);

        // The shape length bound exists to bound runaway prose tokens, not to reject
        // an id the request carries.
        let (ids, _) = parse_prune_from_hidden_meta(&format!("prune:{long}"), &known);
        assert_eq!(ids, vec![long]);

        // A decorated spelling still resolves through the normalized form.
        let quoted: FxHashSet<String> = ["call:weird/0".to_string()].into_iter().collect();
        let (ids, _) = parse_prune_from_hidden_meta("prune:`call:weird/0`", &quoted);
        assert_eq!(ids, vec!["call:weird/0"]);
    }

    #[test]
    fn test_known_id_mixed_with_prose_still_all_or_nothing() {
        let known: FxHashSet<String> = ["call:weird/0".to_string()].into_iter().collect();
        let hidden_meta = "prune:call:weird/0, use the older result\nkeep me";

        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta, &known);

        assert!(ids.is_empty());
        assert_eq!(remaining, hidden_meta);
    }

    #[test]
    fn test_unrecognized_prune_fragments_reports_a_known_id_line_as_an_attempt() {
        let known: FxHashSet<String> = ["call:weird/0".to_string()].into_iter().collect();

        assert_eq!(
            unrecognized_prune_fragments("prune:call:weird/0, see the log", &known),
            vec!["prune:call:weird/0, see the log".to_string()]
        );
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
        // Shortened stubs carry only the archive file name; resolve it against
        // the temp overflow dir (see `stub_archive_display_path`).
        let raw = path_line.trim();
        let archived_path = if std::path::Path::new(raw).is_absolute() {
            std::path::PathBuf::from(raw)
        } else {
            overflow_dir
                .join(super::super::PRESERVED_TOOL_OVERFLOW_DIR)
                .join(raw)
        };
        let archived = std::fs::read_to_string(&archived_path).unwrap();
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
        // The counter changed, so the refresh mutates the message and reports
        // true; callers rebuild the token estimate only on true.
        assert!(ensure_prune_protocol_prompt(&mut messages, &marks));
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
        // Refreshing with identical marks changes nothing and reports false,
        // letting callers reuse the previous token estimate.
        assert!(!ensure_prune_protocol_prompt(&mut messages, &marks));
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
        // Removal mutates the projection, so it reports true and the caller
        // remeasures.
        assert!(ensure_prune_protocol_prompt(
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

    // ---- Subagent results (`ToolPrunePolicy::AfterIntegration`) ----

    use super::PruneAuthorization;

    fn integration_authorization(task_ids: &[&str]) -> PruneAuthorization {
        PruneAuthorization::from_integrated_task_ids(
            task_ids.iter().map(|id| id.to_string()).collect(),
        )
    }

    /// A large subagent delivery shaped like the runtime's own `task_wait` output.
    fn subagent_result(task_ids: &[&str]) -> String {
        let mut content = String::new();
        for task_id in task_ids {
            content.push_str(&format!("[task_id={task_id}]\n"));
            content.push_str(&"subagent conclusion line\n".repeat(300));
        }
        content
    }

    /// One old subagent group plus four recent groups, so the subagent group sits
    /// outside the recent-results protection window.
    fn subagent_conversation(task_ids: &[&str]) -> Vec<Message> {
        let mut messages = vec![
            make_assistant_tool_call("call_sub", "task_wait"),
            make_tool_message("call_sub", &subagent_result(task_ids)),
        ];
        for index in 0..4 {
            let id = format!("call_recent_{index}");
            messages.push(make_assistant_tool_call(&id, "execute_command"));
            messages.push(make_tool_message(&id, "recent result"));
        }
        messages
    }

    #[test]
    fn test_subagent_result_is_eligible_only_after_integration() {
        let messages = subagent_conversation(&["task_a"]);

        // Un-integrated delivery: live state, so neither a candidate nor
        // advertised to the model.
        assert!(
            !active_prunable_tool_ids_authorized(&messages, &PruneAuthorization::default())
                .contains("call_sub")
        );
        assert!(!should_inject_prune_prompt_authorized(
            &messages,
            &PruneAuthorization::default()
        ));

        let integrated = integration_authorization(&["task_a"]);
        assert!(active_prunable_tool_ids_authorized(&messages, &integrated).contains("call_sub"));
        assert!(should_inject_prune_prompt_authorized(
            &messages, &integrated
        ));
    }

    #[test]
    fn test_subagent_result_with_one_unintegrated_task_stays_eligible_free() {
        let messages = subagent_conversation(&["task_a", "task_b"]);
        assert!(
            !active_prunable_tool_ids_authorized(
                &messages,
                &integration_authorization(&["task_a"])
            )
            .contains("call_sub")
        );
        assert!(
            active_prunable_tool_ids_authorized(
                &messages,
                &integration_authorization(&["task_a", "task_b"])
            )
            .contains("call_sub")
        );
    }

    #[test]
    fn test_unintegrated_subagent_result_reports_its_own_rejection_reason() {
        let messages = subagent_conversation(&["task_a"]);
        assert_eq!(
            explain_rejected_prune_mark_authorized(
                &messages,
                "call_sub",
                &PruneAuthorization::default()
            ),
            Some("subagent result is not integrated yet")
        );
        assert_eq!(
            explain_rejected_prune_mark_authorized(
                &messages,
                "call_sub",
                &integration_authorization(&["task_a"])
            ),
            None
        );
    }

    #[test]
    fn test_offload_rechecks_authorization_against_restored_marks() {
        let overflow_dir = make_overflow_dir();
        let mut messages = subagent_conversation(&["task_a"]);
        let marks = [("call_sub".to_string(), PRUNE_THRESHOLD)]
            .into_iter()
            .collect();

        // A mark restored from session state must not unload evidence whose task
        // was never integrated.
        assert_eq!(
            apply_pruning_authorized(
                &mut messages,
                &marks,
                Some(overflow_dir.as_path()),
                &PruneAuthorization::default(),
            )
            .pruned_count,
            0
        );
        assert!(
            messages[1]
                .content
                .as_str()
                .unwrap()
                .contains("task_id=task_a")
        );

        let authorization = integration_authorization(&["task_a"]);
        assert_eq!(
            apply_pruning_authorized(
                &mut messages,
                &marks,
                Some(overflow_dir.as_path()),
                &authorization,
            )
            .pruned_count,
            1
        );

        // The stub no longer carries the task markers, so retention follows how
        // the stub was created; deriving it from the stub text instead would drop
        // the mark and make the offload flap on the next request.
        assert!(retained_prune_ids_authorized(&messages, &authorization).contains("call_sub"));
        assert!(
            !active_prunable_tool_ids_authorized(&messages, &authorization).contains("call_sub")
        );

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    // ---- pressure reclaim: runtime-side offloading without a model mark ----

    /// Below the trigger the marking contract is unchanged: a model that casts
    /// no mark leaves the eligible backlog inline.
    #[test]
    fn test_pressure_reclaim_stays_silent_below_the_trigger() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![
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
        ];

        let report = reclaim_under_pressure_authorized(
            &mut messages,
            Some(overflow_dir.as_path()),
            &PruneAuthorization::default(),
        );

        assert_eq!(report.pruned_count, 0);
        assert_eq!(report.pressure_count, 0);
        assert_eq!(report.freed_chars, 0);
        assert!(
            messages[1]
                .content
                .as_str()
                .unwrap()
                .starts_with("old command output")
        );

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    /// Above the trigger the runtime offloads the largest eligible results
    /// through the same lossless path (full text archived, recallable stub),
    /// reports them as auto-reclaimed, and stops at the per-request character
    /// cap instead of rewriting the whole backlog at once.
    #[test]
    fn test_pressure_reclaim_offloads_largest_within_the_per_request_cap() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![
            make_assistant_tool_call("call_a", "execute_command"),
            make_tool_message("call_a", &"a stale output\n".repeat(2_000)),
            make_assistant_tool_call("call_b", "read_file"),
            make_tool_message("call_b", &"b stale file body\n".repeat(1_500)),
            make_assistant_tool_call("call_c", "execute_command"),
            make_tool_message("call_c", &"c stale output\n".repeat(1_000)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];
        // 30K + 27K + 15K eligible chars: above the trigger, above one cap.

        let report = reclaim_under_pressure_authorized(
            &mut messages,
            Some(overflow_dir.as_path()),
            &PruneAuthorization::default(),
        );

        assert_eq!(report.pruned_count, 2, "largest first, up to the cap");
        assert_eq!(report.pressure_count, 2);
        assert!(report.freed_chars > 0);
        let stub_a = messages[1].content.as_str().unwrap();
        assert!(stub_a.contains("file_path:"));
        assert!(messages[3].content.as_str().unwrap().contains("file_path:"));
        // `call_c` is eligible, but the per-request cap stopped the reclaim.
        assert!(
            messages[5]
                .content
                .as_str()
                .unwrap()
                .starts_with("c stale output")
        );
        // The archived full text really is recallable.
        let raw = stub_a
            .lines()
            .find_map(|line| line.trim().strip_prefix("- file_path: "))
            .expect("stub must carry an archived file_path")
            .trim();
        let archived_path = if std::path::Path::new(raw).is_absolute() {
            std::path::PathBuf::from(raw)
        } else {
            overflow_dir
                .join(super::super::PRESERVED_TOOL_OVERFLOW_DIR)
                .join(raw)
        };
        assert!(
            std::fs::read_to_string(&archived_path)
                .unwrap()
                .contains("a stale output")
        );

        // The next request finds the remainder below the trigger: reclamation
        // converges instead of re-offloading or flapping.
        let repeat = reclaim_under_pressure_authorized(
            &mut messages,
            Some(overflow_dir.as_path()),
            &PruneAuthorization::default(),
        );
        assert_eq!(repeat.pruned_count, 0);
        assert_eq!(repeat.pressure_count, 0);

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    /// Pressure never widens eligibility: the recent-results window, `prune:
    /// Never` tools, and un-integrated subagent deliveries stay inline while
    /// the runtime reclaims around them.
    ///
    /// Every protected item is larger than the largest eligible candidate, so
    /// a hole in any single protection would move that item to the front of the
    /// selection and fail its assertion instead of hiding behind the cap.
    #[test]
    fn test_pressure_reclaim_respects_protection_and_authorization() {
        let overflow_dir = make_overflow_dir();
        let mut subagent_delivery = subagent_result(&["task_a"]);
        subagent_delivery.push_str(&"\nun-integrated padding\n".repeat(2_500));
        let mut messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", &"plan body\n".repeat(6_000)),
            make_assistant_tool_call("call_sub", "task_wait"),
            make_tool_message("call_sub", &subagent_delivery),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"old output\n".repeat(4_000)),
            make_assistant_tool_call("call_old_2", "read_file"),
            make_tool_message("call_old_2", &"old file body\n".repeat(2_500)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", &"recent window body\n".repeat(3_000)),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];
        // Eligible backlog: 44K + 35K. The plan body (60K), the subagent
        // delivery (~60K), and the newest-window result (57K) are all larger
        // than the first eligible pick.

        let report = reclaim_under_pressure_authorized(
            &mut messages,
            Some(overflow_dir.as_path()),
            &PruneAuthorization::default(),
        );

        // 44K already reaches the per-request cap, so exactly one item is
        // reclaimed: the largest eligible candidate.
        assert_eq!(report.pruned_count, 1);
        assert_eq!(report.pressure_count, 1);
        assert!(
            messages[1]
                .content
                .as_str()
                .unwrap()
                .starts_with("plan body"),
            "prune: Never results are never reclaimed"
        );
        assert!(
            messages[3]
                .content
                .as_str()
                .unwrap()
                .contains("task_id=task_a"),
            "an un-integrated subagent delivery stays inline in full"
        );
        assert!(
            messages[5].content.as_str().unwrap().contains("file_path:"),
            "the largest eligible candidate is reclaimed"
        );
        assert!(
            messages[7]
                .content
                .as_str()
                .unwrap()
                .starts_with("old file body"),
            "the per-request cap stopped the reclaim before this candidate"
        );
        assert!(
            messages[9]
                .content
                .as_str()
                .unwrap()
                .starts_with("recent window body"),
            "results inside the recent-results protection window stay inline"
        );

        std::fs::remove_dir_all(&overflow_dir).ok();
    }

    /// The lossless floor outranks the pressure trigger: with no archive
    /// directory nothing is reclaimed, however large the backlog is.
    #[test]
    fn test_pressure_reclaim_requires_an_archive_directory() {
        let mut messages = vec![
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", &"irrecoverable if dropped\n".repeat(4_000)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];

        let report =
            reclaim_under_pressure_authorized(&mut messages, None, &PruneAuthorization::default());

        assert_eq!(report.pruned_count, 0);
        assert_eq!(report.freed_chars, 0);
        assert!(
            messages[1]
                .content
                .as_str()
                .unwrap()
                .starts_with("irrecoverable if dropped")
        );
    }

    /// A candidate larger than the whole per-request cap is still reclaimed:
    /// the largest pick is always included, so one oversized result cannot
    /// block reclamation.
    #[test]
    fn test_pressure_reclaim_always_includes_the_largest_candidate() {
        let overflow_dir = make_overflow_dir();
        let mut messages = vec![
            make_assistant_tool_call("call_huge", "execute_command"),
            make_tool_message("call_huge", &"huge stale output\n".repeat(5_000)),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];

        let report = reclaim_under_pressure_authorized(
            &mut messages,
            Some(overflow_dir.as_path()),
            &PruneAuthorization::default(),
        );

        assert_eq!(report.pruned_count, 1);
        assert_eq!(report.pressure_count, 1);
        assert!(messages[1].content.as_str().unwrap().contains("file_path:"));

        std::fs::remove_dir_all(&overflow_dir).ok();
    }
}
