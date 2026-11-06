//! Reasoning/thinking-mode control + prompt-cache breakpoint injection.
//!
//! Reasoning-related logic extracted from request/mod.rs:
//! - thinking wire-field parsing (per-provider-adapter field differences)
//! - `reasoning_content` echo completion for thinking models
//! - thinking-off injection for auxiliary/background requests
//! - prompt-cache breakpoint injection (`cache_control`)
//! - `reasoning_effort` tier resolution

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rustc_hash::FxHashMap;
use serde_json::{Map, Value, json};

use super::super::{
    history::{Message, is_code_inspection_memory_text, is_system_like_role},
    models,
    provider::{
        ApiProvider, ReasoningEffort, ThinkingOffCapability, reasoning_effort_reduces_thinking_for,
        request_wire_shapes, thinking_dialect_for, thinking_off_capability_for,
    },
    types::{App, ReasoningItemMap},
};
use super::types::extract_displayable_text;
use crate::commonw::{FastSet, configw};

/// Resolve the thinking/reasoning field shapes for each provider adapter.
///
/// Returns a triple:
/// 1. the top-level thinking object (or other provider-specific fields), empty
///    when nothing is injected;
/// 2. the top-level `reasoning_effort` string (some providers place it at the
///    top level);
/// 3. the nested `reasoning` object (some providers place it in `body.reasoning`).
///
/// The effort **value** is adapted per vendor by the dialect first
/// ([`ThinkingDialect::adapt_effort`]) **unless the registry declares a wire
/// placement** ([`models::reasoning_effort_wire`]): a declared placement means
/// the model's effort is vendor-verified graded (e.g. DeepSeek v4 on
/// api.deepseek.com / OpenCode Zen, DashScope DeepSeek v4), so the tier passes
/// through verbatim. Undeclared models fall back to the dialect default —
/// binary-switch vendors omit the tier, everyone else passes it through.
pub(super) fn resolve_reasoning_wire_controls<'a>(
    model: &'a str,
    endpoint: &str,
    enable_thinking: bool,
    reasoning_effort: Option<&'a str>,
) -> (Map<String, Value>, Option<&'a str>, Option<Value>) {
    let adapter_kind = models::model_adapter(model);
    let request_model = models::request_model_name(model);
    resolve_reasoning_wire_controls_from_parts(
        adapter_kind,
        &request_model,
        endpoint,
        models::reasoning_effort_wire(model),
        enable_thinking,
        reasoning_effort,
    )
}

/// Registry-free core of [`resolve_reasoning_wire_controls`]: identical wire
/// rules over pre-resolved inputs (`adapter_kind`, provider-side
/// `request_model`, declared `effort_wire`) so behavior tests pin the dialect
/// matrix without resolving registry keys.
pub(super) fn resolve_reasoning_wire_controls_from_parts<'a>(
    adapter_kind: ApiProvider,
    request_model: &str,
    endpoint: &str,
    effort_wire: Option<crate::ai::model_names::ReasoningEffortWire>,
    enable_thinking: bool,
    reasoning_effort: Option<&'a str>,
) -> (Map<String, Value>, Option<&'a str>, Option<Value>) {
    let thinking_dialect = thinking_dialect_for(adapter_kind, request_model, endpoint);
    // Per-vendor effort **value** adaptation: the dialect owns the mapping and
    // this layer never invents a wire value. A registry-declared
    // `reasoning_effort_wire` takes precedence (the model's effort is
    // vendor-verified graded and placement is explicit); only undeclared models
    // go through the dialect default, where binary-switch dialects omit the
    // tier so the placement routing below has nothing to place.
    let reasoning_effort = if effort_wire.is_some() {
        reasoning_effort
    } else {
        thinking_dialect.adapt_effort(reasoning_effort)
    };
    // The user's `enable_search` request is passed in by the builder; only the
    // reasoning/thinking triple matters here, so `None` is a placeholder — this
    // function does not depend on the returned enable_search.
    let (_, top_level_reasoning_effort, nested_reasoning) = if let Some(wire) = effort_wire {
        match wire {
            crate::ai::model_names::ReasoningEffortWire::TopLevel => (None, reasoning_effort, None),
            crate::ai::model_names::ReasoningEffortWire::Nested => (
                None,
                None,
                reasoning_effort.map(|effort| json!({ "effort": effort })),
            ),
        }
    } else {
        request_wire_shapes(adapter_kind, endpoint, None, reasoning_effort)
    };
    let thinking = thinking_dialect.fields(enable_thinking, top_level_reasoning_effort);
    (thinking, top_level_reasoning_effort, nested_reasoning)
}

/// Normalize the assistant `reasoning_content` replay policy by model capability:
/// - Models that must echo the field back (DeepSeekThinkingDialect) keep the
///   field shape on **every** replayed assistant message, including non-tool-call
///   turns and empty reasoning: an own exact-replay blob decodes back to the
///   original provider text, foreign/encrypted markers are cleared to an empty
///   string, and a missing field is filled with an empty string — a missing
///   field is rejected by the gateway with 400 "The `reasoning_content` in the
///   thinking mode must be passed back", while an empty string passes.
/// - Models declaring exact replay (GLM) only replay the original provider text
///   for tool-call rounds (by decoding the internal marker) to preserve
///   cross-tool-call continuity; non-tool-call messages strip hidden reasoning.
/// - All other models strip hidden reasoning entirely, avoiding cross-turn
///   leakage and context bloat.
pub(super) fn normalize_reasoning_content_replay_for_model(model: &str, messages: &mut [Message]) {
    let exact_replay = models::reasoning_content_replay_enabled(model);
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    normalize_reasoning_content_replay_from_parts(
        model,
        exact_replay,
        adapter_kind,
        &request_model,
        &endpoint,
        messages,
    )
}

/// Registry-free core of [`normalize_reasoning_content_replay_for_model`].
/// `model_tag` is opaque: it only tags persisted replay markers for
/// same-model decode, never looks anything up — tests pass a synthetic tag.
pub(super) fn normalize_reasoning_content_replay_from_parts(
    model_tag: &str,
    exact_replay: bool,
    adapter_kind: ApiProvider,
    request_model: &str,
    endpoint: &str,
    messages: &mut [Message],
) {
    let dialect = thinking_dialect_for(adapter_kind, request_model, endpoint);
    let shape_only_replay = dialect.requires_reasoning_content_echo();

    for message in messages.iter_mut() {
        if message.role != "assistant" {
            continue;
        }
        if shape_only_replay {
            // The DeepSeek thinking-mode gateway validates that every replayed
            // assistant message carries `reasoning_content` (see the function
            // docs), and the Anthropic Messages dialect needs the thinking
            // chain preserved for tool-round replay, so this branch covers all
            // assistant messages, not only tool-call rounds. Persisted replay
            // state (exact or encrypted markers) must never be sent verbatim to
            // the provider: an own exact marker decodes back to the original
            // text, and any other marker (cross-model / encrypted) is cleared
            // to an empty string to keep only the field shape.
            let current = message.reasoning_content.take();
            message.reasoning_content = Some(match &current {
                Some(reasoning)
                    if crate::ai::history::compress::is_persisted_reasoning_replay(reasoning) =>
                {
                    crate::ai::history::compress::decode_reasoning_replay_for_model(
                        model_tag, reasoning,
                    )
                    .unwrap_or_default()
                }
                Some(reasoning) => reasoning.clone(),
                None => String::new(),
            });
            continue;
        }
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
        if exact_replay {
            // Exact continuation state can only be produced by the same model;
            // untagged content (e.g. pre-switch GPT reasoning) and other exact
            // models' state must not be replayed across models.
            if !has_tool_calls {
                message.reasoning_content = None;
            } else {
                message.reasoning_content =
                    message.reasoning_content.as_deref().and_then(|reasoning| {
                        crate::ai::history::compress::decode_reasoning_replay_for_model(
                            model_tag, reasoning,
                        )
                    });
            }
            continue;
        }
        message.reasoning_content = None;
    }
}

/// Decoded persisted encrypted-reasoning items, keyed by source model, resolved
/// endpoint identity and the exact encoded blob.
///
/// A request projection is rebuilt for every preview and every send (several
/// times per tool round). The persisted blob is immutable once written and the
/// decode is a pure function of `(model, endpoint identity, blob)`, so caching
/// it removes repeated JSON parsing without changing which items are replayed:
/// the full blob is the key, so a rewritten payload can never hit a stale
/// entry, and the identity is part of the scope so a decision taken under one
/// backing resource can never be served to a request routed to another.
#[derive(Clone)]
enum DecodedReasoning {
    Items(Arc<Vec<Value>>),
    Missing,
}

#[derive(Default)]
struct DecodedReasoningCache {
    by_scope: FxHashMap<String, FxHashMap<String, DecodedReasoning>>,
    entries: usize,
    encoded_bytes: usize,
}

const DECODED_REASONING_CACHE_MAX_ENTRIES: usize = 64;
const DECODED_REASONING_CACHE_MAX_ENCODED_BYTES: usize = 8 * 1024 * 1024;

static DECODED_REASONING_CACHE: LazyLock<Mutex<DecodedReasoningCache>> =
    LazyLock::new(|| Mutex::new(DecodedReasoningCache::default()));

impl DecodedReasoningCache {
    fn get(&self, scope: &str, encoded: &str) -> Option<DecodedReasoning> {
        self.by_scope.get(scope)?.get(encoded).cloned()
    }

    fn insert(&mut self, scope: &str, encoded: &str, value: DecodedReasoning) {
        // A single blob larger than the whole budget is decoded on demand
        // rather than thrashing a cache it can never fit in.
        if encoded.len() > DECODED_REASONING_CACHE_MAX_ENCODED_BYTES {
            return;
        }
        if self
            .by_scope
            .get(scope)
            .is_some_and(|bucket| bucket.contains_key(encoded))
        {
            return;
        }
        if self.entries >= DECODED_REASONING_CACHE_MAX_ENTRIES
            || self.encoded_bytes + encoded.len() > DECODED_REASONING_CACHE_MAX_ENCODED_BYTES
        {
            self.by_scope.clear();
            self.entries = 0;
            self.encoded_bytes = 0;
        }
        self.by_scope
            .entry(scope.to_string())
            .or_default()
            .insert(encoded.to_string(), value);
        self.entries += 1;
        self.encoded_bytes += encoded.len();
    }
}

/// Cache scope for one decode: the model alone is not enough, because the same
/// persisted marker decodes differently depending on which backing resource the
/// current request is routed to (see the marker's endpoint identity).
fn decrypt_scope_key(model: &str, endpoint_identity: Option<&str>) -> String {
    format!("{model}\u{1f}{}", endpoint_identity.unwrap_or_default())
}

fn decode_encrypted_reasoning_cached(
    model: &str,
    endpoint_identity: Option<&str>,
    encoded: &str,
) -> Option<Arc<Vec<Value>>> {
    let scope = decrypt_scope_key(model, endpoint_identity);
    let cached = DECODED_REASONING_CACHE
        .lock()
        .ok()
        .and_then(|cache| cache.get(&scope, encoded));
    if let Some(value) = cached {
        return match value {
            DecodedReasoning::Items(items) => Some(items),
            DecodedReasoning::Missing => None,
        };
    }
    let decoded = crate::ai::history::compress::decode_encrypted_reasoning_replay_for_model(
        model,
        endpoint_identity,
        encoded,
    );
    let value = match decoded {
        Some(items) if !items.is_empty() => DecodedReasoning::Items(Arc::new(items)),
        _ => DecodedReasoning::Missing,
    };
    if let Ok(mut cache) = DECODED_REASONING_CACHE.lock() {
        cache.insert(&scope, encoded, value.clone());
    }
    match value {
        DecodedReasoning::Items(items) => Some(items),
        DecodedReasoning::Missing => None,
    }
}

/// Test convenience over
/// [`reconstruct_encrypted_reasoning_items_for_model_recorded`] that discards
/// the recorded replay owners.
#[cfg(test)]
pub(super) fn reconstruct_encrypted_reasoning_items_for_model(
    model: &str,
    endpoint: &str,
    session_id: &str,
    messages: &[Message],
    live: &ReasoningItemMap,
) -> ReasoningItemMap {
    reconstruct_encrypted_reasoning_items_for_model_recorded(
        model,
        endpoint,
        session_id,
        Path::new(""),
        messages,
        live,
        &mut SplicedReplayKeys::default(),
    )
}

/// Which side channel supplied a spliced replay owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplayTier {
    /// Decoded from a persisted marker in history.
    Persisted,
    /// Captured by the current turn in `turn_reasoning_items`.
    Live,
}

impl ReplayTier {
    /// Order in which the cross-resource repair drops tiers: persisted markers
    /// may predate a route change, while live items were captured on the route
    /// serving this turn, so they are only dropped when the rejection persists.
    const REPAIR_ORDER: [ReplayTier; 2] = [ReplayTier::Persisted, ReplayTier::Live];

    pub(super) fn label(self) -> &'static str {
        match self {
            ReplayTier::Persisted => "persisted",
            ReplayTier::Live => "live",
        }
    }
}

/// Replay owners one attempt splices into the wire body, in message order,
/// each tagged with the tier that supplied its items. Owners are the first
/// tool_call id of each assistant tool-call group with items in the merged
/// map, matching the lookup `responses_input` performs when splicing.
#[derive(Debug, Default)]
pub(super) struct SplicedReplayKeys {
    owners: Vec<(String, ReplayTier)>,
    /// Owners the repair removed from this request's replay map but has not
    /// recorded in the rejected-replay ledger yet; only an accepted resend
    /// commits them ([`commit_dropped_replay_tiers`]).
    dropped: Vec<String>,
}

impl SplicedReplayKeys {
    fn push(&mut self, first_call_id: String, tier: ReplayTier) {
        self.owners.push((first_call_id, tier));
    }

    /// Every spliced owner in message order.
    #[cfg(test)]
    fn all(&self) -> Vec<String> {
        self.owners.iter().map(|(owner, _)| owner.clone()).collect()
    }

    /// Spliced owners of one tier in message order.
    fn tier(&self, tier: ReplayTier) -> Vec<String> {
        self.owners
            .iter()
            .filter(|(_, owner_tier)| *owner_tier == tier)
            .map(|(owner, _)| owner.clone())
            .collect()
    }

    /// First tier in repair order that still has spliced owners.
    fn next_repair_tier(&self) -> Option<ReplayTier> {
        ReplayTier::REPAIR_ORDER
            .into_iter()
            .find(|tier| self.owners.iter().any(|(_, owner_tier)| owner_tier == tier))
    }

    fn take_tier(&mut self, tier: ReplayTier) -> Vec<String> {
        let owners = self.tier(tier);
        self.owners.retain(|(_, owner_tier)| *owner_tier != tier);
        owners
    }
}

/// Rebuilds the side-channel map for Responses encrypted-reasoning replay from
/// persisted messages (key = first tool_call id).
///
/// Background: an encrypted-replay model's encrypted reasoning lives in the
/// in-memory `turn_reasoning_items` during the turn that produced it, but that
/// is turn-scoped, cleared each turn, and lost on process exit. Across turns or
/// after a Ctrl+C resume, it can only be recovered from the encoded blob
/// persisted in `reasoning_content`. This function scans the current request
/// projection (already pruned naturally by compression, so it only replays
/// "recent non-folded turns", matching exact-replay's replay scope) and decodes
/// each marked assistant tool-call turn whose source model matches the current
/// model back into items, attaching them to its first tool_call id.
///
/// Only fills keys missing from `live`: the in-memory side channel (freshest
/// capture of the current turn) wins; persisted decoding only fills historical
/// gaps, so an existing key is never overwritten by a stale value. Cross-model
/// (marker model != current model) decoding returns None and is skipped
/// automatically, and so does a marker whose recorded backing-resource identity
/// differs from the endpoint resolved for this request. Owners recorded in the
/// rejected-replay ledger for this session, model and endpoint identity are
/// dropped on every path, live items included. The durable part of that ledger
/// is loaded once per `(session, history_path)` before filtering
/// ([`load_rejected_replay_keys`]), so owners rejected by an earlier process
/// stay suppressed without rewriting history.
///
/// Also reports in `spliced` which replay owners this attempt will splice into
/// the wire body, split by tier. The caller keeps them for one request: a
/// cross-resource rejection does not name the offending item, so the repair
/// drops whole tiers ([`drop_next_rejected_replay_tier`]) and records them only
/// once a resend without them is accepted ([`commit_dropped_replay_tiers`]). Live
/// items are spliced even when persisted decoding is off (model without the
/// capability or runtime replay disabled), so that path still honors the
/// ledger and records its live owners.
pub(super) fn reconstruct_encrypted_reasoning_items_for_model_recorded(
    model: &str,
    endpoint: &str,
    session_id: &str,
    history_path: &Path,
    messages: &[Message],
    live: &ReasoningItemMap,
    spliced: &mut SplicedReplayKeys,
) -> ReasoningItemMap {
    let mut merged = live.clone();
    let decode_persisted = models::reasoning_encrypted_replay_enabled(model)
        && crate::ai::history::compress::encrypted_reasoning_replay_runtime_enabled();
    if !decode_persisted && live.is_empty() {
        return merged;
    }
    // Resolved even when persisted decoding is off: the rejected-replay ledger
    // is keyed by the serving resource, so live owners are suppressed only for
    // the endpoint identity that rejected them.
    let endpoint_identity = models::endpoint_resource_identity(endpoint);
    load_rejected_replay_keys(session_id, history_path);
    for message in messages {
        if message.role != "assistant" {
            continue;
        }
        let Some(first_call_id) = message
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map(|call| call.id.clone())
        else {
            continue;
        };
        if encrypted_reasoning_replay_key_rejected(
            session_id,
            endpoint_identity.as_deref(),
            model,
            &first_call_id,
        ) {
            merged.remove(&first_call_id);
            continue;
        }
        if live.contains_key(&first_call_id) {
            spliced.push(first_call_id, ReplayTier::Live);
            continue;
        }
        if !decode_persisted || merged.contains_key(&first_call_id) {
            continue;
        }
        let Some(encoded) = message.reasoning_content.as_deref() else {
            continue;
        };
        if let Some(items) =
            decode_encrypted_reasoning_cached(model, endpoint_identity.as_deref(), encoded)
        {
            merged.insert(first_call_id.clone(), items);
            spliced.push(first_call_id, ReplayTier::Persisted);
        }
    }
    merged
}

/// One step of the cross-resource replay repair: removes the owners of the
/// next tier in [`ReplayTier::REPAIR_ORDER`] from this attempt's replay map, so
/// the retry resends the same projection without them. The owners are only
/// staged in `spliced`, not recorded: the rejection does not name the
/// offending item, so a dropped tier may hold innocent groups, and only an
/// accepted resend proves dropping it was needed ([`commit_dropped_replay_tiers`]).
/// A resend that fails for any reason leaves the ledger untouched and the staged
/// owners die with the request. Returns the dropped tier and owner count, or
/// `None` once no spliced tier remains and the rejection is final. Each call
/// consumes one tier and nothing re-adds spliced owners within a request, so a
/// request makes at most one send per tier plus the original.
pub(super) fn drop_next_rejected_replay_tier(
    reasoning_items: &mut ReasoningItemMap,
    spliced: &mut SplicedReplayKeys,
) -> Option<(ReplayTier, usize)> {
    let tier = spliced.next_repair_tier()?;
    let owners = spliced.take_tier(tier);
    for owner in &owners {
        reasoning_items.remove(owner);
    }
    let dropped = owners.len();
    spliced.dropped.extend(owners);
    Some((tier, dropped))
}

/// Records every owner the repair dropped for this request in the
/// rejected-replay ledger, both in memory and in the session's durable ledger.
/// Called only after a resend without them was accepted, so later attempts skip
/// them instead of paying the rejection again; a request that never dropped a
/// tier records nothing.
///
/// Entries are keyed by the identity of the endpoint that rejected them, so an
/// owner replays again once requests route back to a resource that accepts it.
/// Persistence is best effort: an unreadable or unwritable ledger leaves the
/// in-process state in charge and never fails the request.
pub(super) fn commit_dropped_replay_tiers(
    session_id: &str,
    model: &str,
    endpoint: &str,
    history_path: &Path,
    spliced: &SplicedReplayKeys,
) {
    if spliced.dropped.is_empty() {
        return;
    }
    let Some(endpoint_identity) = models::endpoint_resource_identity(endpoint) else {
        return;
    };
    let recorded = spliced
        .dropped
        .iter()
        .map(|first_call_id| rejected_replay_entry(&endpoint_identity, model, first_call_id))
        .collect::<Vec<_>>();
    reject_encrypted_reasoning_replay_keys(session_id, &endpoint_identity, model, &spliced.dropped);
    let mut entries =
        crate::ai::history::read_encrypted_reasoning_replay_rejections_sqlite(history_path)
            .unwrap_or_default();
    // Re-recording an owner moves it to the newest position, so the oldest
    // rejections are the ones the size cap evicts.
    let recorded_set = recorded.iter().cloned().collect::<FastSet<String>>();
    entries.retain(|entry| !recorded_set.contains(entry));
    entries.extend(recorded);
    let _ = crate::ai::history::write_encrypted_reasoning_replay_rejections_sqlite(
        history_path,
        &entries,
    );
}

/// Session-keyed set of encrypted-reasoning replay owners whose tier the
/// cross-resource repair had to drop, mirrored in the session's durable
/// history state.
///
/// A 400 `modelhub_err_code_-4003` "created under a different Azure OpenAI
/// resource" rejects the request because at least one replayed item was
/// created by a resource this route can no longer reach; resending it can
/// never succeed. The rejection does not name the item, so the repair drops a
/// whole tier at a time — persisted owners first, live owners only when the
/// rejection persists — and a dropped tier may include owners that were
/// themselves replayable. Canonical history is truth and is never rewritten
/// for this, so the marker stays on disk. Instead this ledger remembers each
/// owner (the first tool_call id of its assistant message) of the tiers whose
/// drop let a resend through, for the session + endpoint identity + model, and
/// reconstruction skips it, live or persisted, in every later attempt. Items
/// captured after the repair carry new tool_call ids and stay replayable, so a
/// resource change costs at most two extra rejected round-trips per endpoint
/// instead of one per turn.
///
/// The endpoint identity in the key is what keeps a rejection from disabling
/// valid reasoning: the same owner is only suppressed while requests route to
/// the resource that rejected it, and it replays again once the session goes
/// back to a resource that accepts it. Canonical history is never rewritten.
///
/// Entries are also persisted per session
/// (`history::write_encrypted_reasoning_replay_rejections_sqlite`) and reloaded
/// on reconstruction ([`load_rejected_replay_keys`]), so a restart — or the
/// fresh process each serve/one-shot turn uses — does not pay the rejection
/// again while the projection still splices the stale items. A missing
/// database or absent table reads as no durable entries; a failed read
/// degrades only that attempt to the process-local state and is retried on the
/// next reconstruction.
#[derive(Default)]
struct RejectedReplayKeys {
    keys: FastSet<String>,
    /// `keys` in insertion order, so the size cap evicts the oldest entry.
    order: VecDeque<String>,
    /// `(session, history path)` pairs whose durable entries were successfully
    /// merged, so reconstruction does not reopen SQLite for every request. A
    /// failed read stays out of this set so the next reconstruction retries.
    loaded: FastSet<(String, PathBuf)>,
}

impl RejectedReplayKeys {
    fn contains(&self, key: &str) -> bool {
        self.keys.contains(key)
    }

    fn insert(&mut self, key: String) {
        if self.keys.contains(&key) {
            return;
        }
        if self.order.len() >= REJECTED_REPLAY_KEYS_MAX_ENTRIES
            && let Some(oldest) = self.order.pop_front()
        {
            self.keys.remove(&oldest);
        }
        self.keys.insert(key.clone());
        self.order.push_back(key);
    }
}

/// Cheap upper bound; the ledger only holds owners of tiers the repair dropped,
/// so reaching it means several resource changes in one process. Evicting the
/// oldest entry keeps the newest rejections and only risks repeating the
/// repair once for a long-gone owner.
const REJECTED_REPLAY_KEYS_MAX_ENTRIES: usize =
    crate::ai::history::ENCRYPTED_REASONING_REPLAY_REJECTIONS_MAX_ENTRIES;

static REJECTED_REPLAY_KEYS: LazyLock<Mutex<RejectedReplayKeys>> =
    LazyLock::new(|| Mutex::new(RejectedReplayKeys::default()));

/// Marker-format discriminator prefixed to every durable rejection key.
///
/// `PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX` carries the marker shape and
/// its version (`...:v1`); using the whole prefix retires rows recorded against
/// an older marker shape instead of letting them suppress owners whose
/// persisted payload was re-encoded. The trailing segment separator is dropped
/// because the key joins segments with it.
fn rejected_replay_format_discriminator() -> &'static str {
    crate::ai::history::compress::PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX
        .trim_end_matches('\u{1f}')
}

/// Durable ledger entry shape: the replay-marker format that produced the
/// stored payload, the endpoint identity that rejected the owner, the model,
/// and the owner's first tool_call id. The session is the enclosing history
/// database file.
fn rejected_replay_entry(endpoint_identity: &str, model: &str, first_call_id: &str) -> String {
    format!(
        "{}\u{1f}{endpoint_identity}\u{1f}{model}\u{1f}{first_call_id}",
        rejected_replay_format_discriminator()
    )
}

fn rejected_replay_key(
    session_id: &str,
    endpoint_identity: &str,
    model: &str,
    first_call_id: &str,
) -> String {
    format!(
        "{session_id}\u{1f}{}",
        rejected_replay_entry(endpoint_identity, model, first_call_id)
    )
}

/// Merge the session's durable rejected-replay entries into the in-process
/// ledger on the first successful read per `(session, history path)`.
///
/// The ledger is advisory state: a missing database or absent table reads as
/// no durable entries, and a SQLite error leaves the in-memory behavior
/// unchanged without marking the scope loaded, so a later request retries the
/// read instead of losing durable suppression until restart. The read happens
/// outside the ledger lock so SQLite I/O cannot block reconstruction for
/// another session.
fn load_rejected_replay_keys(session_id: &str, history_path: &Path) {
    let scope = (session_id.to_string(), history_path.to_path_buf());
    let already_loaded = REJECTED_REPLAY_KEYS
        .lock()
        .map(|ledger| ledger.loaded.contains(&scope))
        .unwrap_or(true);
    if already_loaded {
        return;
    }
    // Only a successful read may mark the scope loaded: a transient SQLite
    // error must leave durable suppression retryable instead of disabling it
    // until the process restarts. Failing open keeps this request unblocked.
    let Ok(entries) =
        crate::ai::history::read_encrypted_reasoning_replay_rejections_sqlite(history_path)
    else {
        return;
    };
    let Ok(mut ledger) = REJECTED_REPLAY_KEYS.lock() else {
        return;
    };
    // Another reconstruction may have merged this scope while the read ran.
    if !ledger.loaded.insert(scope) {
        return;
    }
    for entry in entries {
        ledger.insert(format!("{session_id}\u{1f}{entry}"));
    }
}

fn encrypted_reasoning_replay_key_rejected(
    session_id: &str,
    endpoint_identity: Option<&str>,
    model: &str,
    first_call_id: &str,
) -> bool {
    // Without an identity there is nothing to scope suppression to; replay
    // stays enabled, matching the marker decode rule.
    let Some(endpoint_identity) = endpoint_identity else {
        return false;
    };
    REJECTED_REPLAY_KEYS
        .lock()
        .map(|ledger| {
            ledger.contains(&rejected_replay_key(
                session_id,
                endpoint_identity,
                model,
                first_call_id,
            ))
        })
        .unwrap_or(false)
}

/// Records these owners in the rejected-replay ledger once a resend without
/// their tier was accepted after a cross-resource rejection. Recorded owners
/// stay blocked for this endpoint identity (until the size cap evicts them as
/// the oldest entries); owners outside the dropped tiers keep replaying, and
/// the same owners replay again under another identity.
pub(super) fn reject_encrypted_reasoning_replay_keys(
    session_id: &str,
    endpoint_identity: &str,
    model: &str,
    first_call_ids: &[String],
) {
    let Ok(mut ledger) = REJECTED_REPLAY_KEYS.lock() else {
        return;
    };
    for first_call_id in first_call_ids {
        ledger.insert(rejected_replay_key(
            session_id,
            endpoint_identity,
            model,
            first_call_id,
        ));
    }
}

/// Merges the provider adapter's thinking fields into an auxiliary/background
/// request body.
///
/// Auxiliary (non-main-path) and background requests always turn thinking off
/// (`enable_thinking=false`); each adapter decides which keys to write
/// (`enable_thinking:false` / `thinking:{"type":"disabled"}` / or nothing), and
/// the core layer no longer discriminates by provider.
pub(crate) fn apply_aux_thinking_fields(model: &str, body: &mut Value) {
    let endpoint = models::endpoint_for_model(model, "");
    let (fields, _, _) = resolve_reasoning_wire_controls(model, &endpoint, false, None);
    if fields.is_empty() {
        return;
    }
    if let Some(map) = body.as_object_mut() {
        for (key, value) in fields {
            map.insert(key, value);
        }
    }
}

/// Whether opt-in explicit prompt-cache breakpoint injection is enabled.
///
/// `cache_control` is a provider/model-level capability declared by the
/// `explicit_prompt_cache` field in the model registry (models/); plain
/// OpenAI-compatible models may not accept this extension field.
pub(super) fn prompt_cache_enabled_for_model(model: &str) -> bool {
    prompt_cache_config_enabled() && models::explicit_prompt_cache_enabled(model)
}

fn prompt_cache_config_enabled() -> bool {
    configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_ENABLE,
            "false",
        )
        .trim()
        .eq_ignore_ascii_case("true")
}

/// Places explicit prompt-cache breakpoints on chat-completions wire messages:
/// one on the first system message (stable instructions) and one rolling
/// breakpoint on the last user/tool message, so every request writes the whole
/// conversation prefix and the next request reads it back (the provider looks
/// back from each breakpoint for the longest cached prefix). Wire-only: request
/// messages feed prompt fingerprints, and a marker that moves every request
/// would break prefix comparison at its previous position. Uses 2 of the 4
/// breakpoints Anthropic allows; idempotent.
pub(super) fn apply_prompt_cache_breakpoints(messages: &mut [Value]) {
    let role_of = |message: &Value| {
        message
            .get("role")
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    if let Some(content) = messages
        .iter_mut()
        .find(|message| role_of(message).is_some_and(|role| is_system_like_role(&role)))
        .and_then(|message| message.get_mut("content"))
    {
        mark_content_cache_breakpoint(content, is_nonempty_text_block);
    }
    if let Some(content) = messages
        .iter_mut()
        .rev()
        .find(|message| {
            matches!(role_of(message).as_deref(), Some("user" | "tool"))
                && !is_code_inspection_memory_message(message)
        })
        .and_then(|message| message.get_mut("content"))
    {
        mark_content_cache_breakpoint(content, is_nonempty_text_block);
    }
}

/// The code-inspection working-memory note is request-only tail context. Leaving a rolling
/// breakpoint on it would make every round's edited note re-bill the conversation prefix.
pub(super) fn is_code_inspection_memory_message(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("user")
        && message.get("content").is_some_and(|content| {
            is_code_inspection_memory_text(&extract_displayable_text(content))
        })
}

/// Marks `content` as a cache breakpoint: a non-empty plain string becomes a
/// single text block; a block array marks its last block accepted by
/// `cacheable`. Empty text is never marked because Anthropic rejects empty
/// text blocks. Shared by the chat-completions and Anthropic Messages dialects
/// so the breakpoint rule cannot drift between them; idempotent.
pub(super) fn mark_content_cache_breakpoint(content: &mut Value, cacheable: fn(&Value) -> bool) {
    if let Value::String(text) = content {
        if text.is_empty() {
            return;
        }
        let text = std::mem::take(text);
        *content = json!([{ "type": "text", "text": text }]);
    }
    if let Some(Value::Object(block)) = content
        .as_array_mut()
        .and_then(|blocks| blocks.iter_mut().rev().find(|block| cacheable(block)))
    {
        block.insert("cache_control".to_string(), json!({ "type": "ephemeral" }));
    }
}

/// Non-empty `text` block: the only part chat-completions gateways accept
/// `cache_control` on.
pub(super) fn is_nonempty_text_block(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("text")
        && block
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
}

/// Per-vendor thinking-off capability for the current model (see
/// [`ThinkingDialect::thinking_off_capability`]).
///
/// Deliberately resolved through the provider dialect layer rather than the
/// registry: registry fields like `reasoning_effort_wire` describe the effort
/// wire shape, not whether a dedicated thinking off-switch exists (e.g. the
/// DashScope DeepSeek entries declare `reasoning_effort_wire: "top_level"` yet
/// still turn thinking off via the `enable_thinking` switch).
pub(crate) fn model_thinking_off_capability(model: &str) -> ThinkingOffCapability {
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    thinking_off_capability_from_parts(models::model_adapter(model), &request_model, &endpoint)
}

/// Registry-free core of [`model_thinking_off_capability`]: pure dialect
/// dispatch over pre-resolved inputs.
pub(crate) fn thinking_off_capability_from_parts(
    adapter: ApiProvider,
    request_model: &str,
    endpoint: &str,
) -> ThinkingOffCapability {
    thinking_off_capability_for(adapter, request_model, endpoint)
}

/// Whether the current model's thinking has a wire effort gradation.
///
/// Registry-driven first: a model that declares `reasoning_effort_wire`
/// (models/) has vendor-verified graded effort (e.g. DeepSeek v4 on
/// api.deepseek.com / OpenCode Zen, DashScope DeepSeek v4), so its resolved
/// tier is sent and shown as a real gradation. Undeclared models fall back to
/// the thinking-dialect default, where binary-switch dialects (DeepSeek /
/// DashScope) treat effort as a no-op and omit it from the wire.
pub(crate) fn model_effort_graded(model: &str) -> bool {
    models::reasoning_effort_reduces_thinking(model)
}

/// Registry-free core of [`model_effort_graded`] over pre-resolved inputs.
pub(crate) fn model_effort_graded_from_parts(
    effort_wire: Option<crate::ai::model_names::ReasoningEffortWire>,
    adapter: ApiProvider,
    request_model: &str,
    endpoint: &str,
) -> bool {
    models::reasoning_effort_reduces_thinking_from_parts(
        effort_wire,
        adapter,
        request_model,
        endpoint,
    )
}

/// Resolve the effective reasoning intensity for the current session, highest
/// priority first:
/// 1. CLI argument `--reasoning-effort` or the `/model effort <x>` override
///    stored in [`App.cli.reasoning_effort_override`] (`Some(None)` = user
///    explicitly disabled; `None` = not set);
/// 2. The sharp agent's implied `low` (derived from the live `current_agent`,
///    so `/agent` switches and startup fallbacks take effect with no state
///    sync — an explicit override above always wins over it);
/// 3. The model registry ([models/](../../../../models)) default `reasoning_effort`;
/// 4. `None` -- no field injected, server default applies.
///
/// An explicit "off" (`Some(None)`) is treated as a request to actually turn
/// thinking off. Per-vendor adaptation, driven by the dialect's declared off
/// capability:
/// - [`ThinkingOffCapability::RealSwitch`] dialects turn thinking off in
///   `resolve_thinking` (DashScope `enable_thinking: false` / DeepSeek
///   `thinking: {"type":"disabled"}`) and omit the effort field here;
/// - [`ThinkingOffCapability::EffortNone`] dialects express off as
///   [`ReasoningEffort::None`] ("none" on the wire) — the same value the
///   truncation force-off fallback emits — for thinking-enabled models;
/// - [`ThinkingOffCapability::Unsupported`] dialects omit the field; the
///   `/effort off` handler reports that thinking stays on.
pub(crate) fn resolve_reasoning_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    resolve_reasoning_effort_from_parts(
        app.cli.reasoning_effort_override,
        &app.current_agent,
        model_thinking_off_capability(model),
        models::enable_thinking(model),
        models::default_reasoning_effort(model),
    )
}

/// Registry-free core of [`resolve_reasoning_effort`]: identical priority
/// ladder over pre-resolved inputs (`effort_override` is `Some(None)` for an
/// explicit off, `current_agent` for the sharp-agent implied `low`).
pub(crate) fn resolve_reasoning_effort_from_parts(
    effort_override: Option<Option<ReasoningEffort>>,
    current_agent: &str,
    off_capability: ThinkingOffCapability,
    thinking_enabled: bool,
    default_effort: Option<ReasoningEffort>,
) -> Option<ReasoningEffort> {
    if let Some(Some(level)) = effort_override {
        return Some(level);
    }
    if effort_override == Some(None) {
        // User explicitly typed `/effort off` (or `--reasoning-effort off`).
        match off_capability {
            ThinkingOffCapability::EffortNone if thinking_enabled => {
                return Some(ReasoningEffort::None);
            }
            _ => return None,
        }
    }
    if current_agent == "sharp" {
        // Sharp means minimal deliberation; the configured default (often
        // `max`) would defeat it. Derived here — not stored in the override
        // at CLI parse — so switching agents mid-session (`/agent sharp` /
        // `/agent build`) and startup fallbacks immediately change the
        // effective tier instead of leaving a stale value behind.
        return Some(ReasoningEffort::Low);
    }
    default_effort
}

/// Apply the truncation ladder's last-resort force-off fallback to the reasoning-effort value.
///
/// When `thinking_disabled_override` is active (set by the orchestrator after repeated
/// truncation, or by the reasoning-only recovery), the request layer must express "thinking
/// off" per the dialect's declared off capability — never by blindly sending `"none"`:
/// - [`ThinkingOffCapability::EffortNone`] dialects (OpenAI family / Responses —
///   `NoThinkingDialect` sends no thinking field): the only wire value that disables thinking
///   is `"none"`, so the effort is mapped to `ReasoningEffort::None`;
/// - [`ThinkingOffCapability::RealSwitch`] dialects (DashScope `enable_thinking: false`,
///   DeepSeek `thinking: {"type":"disabled"}`) are already turned off through
///   `enable_thinking=false`; the effort passes through unchanged;
/// - [`ThinkingOffCapability::Unsupported`] dialects (MiniMax M2.x, beam) have no off
///   mechanism — sending `"none"` is rejected or ignored there — so the current tier passes
///   through unchanged and thinking stays on. The graduated effort ladder itself deliberately
///   never sends `"none"` (orchestrator.rs) — this maps only the force-off fallback, not
///   ladder retries.
pub(crate) fn apply_thinking_force_off_effort<'a>(
    thinking_disabled_override: bool,
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
    effort: Option<&'a str>,
) -> Option<&'a str> {
    apply_thinking_force_off_effort_from_parts(
        thinking_disabled_override,
        reasoning_effort_reduces_thinking_for(provider, model, endpoint),
        thinking_off_capability_for(provider, model, endpoint),
        effort,
    )
}

/// Registry-free core of [`apply_thinking_force_off_effort`] over the
/// pre-resolved dialect predicates.
pub(crate) fn apply_thinking_force_off_effort_from_parts<'a>(
    thinking_disabled_override: bool,
    effort_reduces_thinking: bool,
    off_capability: ThinkingOffCapability,
    effort: Option<&'a str>,
) -> Option<&'a str> {
    if thinking_disabled_override
        && effort_reduces_thinking
        && matches!(off_capability, ThinkingOffCapability::EffortNone)
    {
        Some(ReasoningEffort::None.as_str())
    } else {
        effort
    }
}

/// Returns the reasoning intensity shown in the input box for the current
/// request. When no field is sent, it is explicitly labeled as the server
/// default, so a model without a default tier is not mis-displayed as a
/// specific effort.
pub(crate) fn reasoning_effort_display_label(app: &App, model: &str) -> &'static str {
    reasoning_effort_display_label_from_parts(
        app.cli.reasoning_effort_override,
        &app.current_agent,
        model_thinking_off_capability(model),
        models::enable_thinking(model),
        models::default_reasoning_effort(model),
    )
}

/// Resolve the display effort for a fresh child, whose fork clears parent overrides.
pub(crate) fn reasoning_effort_display_label_for_subagent(
    agent: &str,
    model: &str,
) -> &'static str {
    reasoning_effort_display_label_from_parts(
        None,
        agent,
        model_thinking_off_capability(model),
        models::enable_thinking(model),
        models::default_reasoning_effort(model),
    )
}

/// Registry-free core of [`reasoning_effort_display_label`] over pre-resolved
/// inputs; see [`resolve_reasoning_effort_from_parts`] for the semantics.
pub(crate) fn reasoning_effort_display_label_from_parts(
    effort_override: Option<Option<ReasoningEffort>>,
    current_agent: &str,
    off_capability: ThinkingOffCapability,
    thinking_enabled: bool,
    default_effort: Option<ReasoningEffort>,
) -> &'static str {
    if effort_override == Some(None) {
        // Explicit `/effort off`: show "off" (the user's intent) instead of the
        // underlying "none"/"server default" value. When the vendor has no off
        // mechanism at all, say so instead of implying thinking was turned off.
        return match off_capability {
            ThinkingOffCapability::Unsupported => "off (unsupported)",
            _ => "off",
        };
    }
    match resolve_reasoning_effort_from_parts(
        effort_override,
        current_agent,
        off_capability,
        thinking_enabled,
        default_effort,
    ) {
        Some(effort) => effort.as_str(),
        None => "server default",
    }
}

#[cfg(test)]
mod encrypted_replay_reconstruct_tests {
    use std::path::Path;
    use std::sync::Arc;

    use super::{
        REJECTED_REPLAY_KEYS_MAX_ENTRIES, RejectedReplayKeys, ReplayTier, SplicedReplayKeys,
        commit_dropped_replay_tiers, drop_next_rejected_replay_tier,
        reconstruct_encrypted_reasoning_items_for_model,
        reconstruct_encrypted_reasoning_items_for_model_recorded,
        reject_encrypted_reasoning_replay_keys,
    };
    use crate::ai::history::Message;
    use crate::ai::history::compress::encode_encrypted_reasoning_replay_state;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::types::{FunctionCall, ReasoningItemMap, ToolCall};
    use rustc_hash::FxHashMap;
    use serde_json::{Value, json};

    /// Endpoint every reconstruction below is routed to; markers encoded with
    /// `None` carry no identity (v1) and must keep replaying.
    const TEST_ENDPOINT: &str = "https://reasoning-replay.test/v1";
    const TEST_SESSION: &str = "reasoning-replay-test-session";

    fn test_endpoint_identity() -> String {
        crate::ai::models::endpoint_resource_identity(TEST_ENDPOINT)
            .expect("test endpoint must resolve an identity")
    }

    /// First registry key declaring `reasoning_encrypted_replay`. Hard-fails
    /// when no such entry exists so the tests below cannot silently run
    /// against the unknown-model fallback.
    fn first_encrypted_replay_model_key() -> String {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.reasoning_encrypted_replay)
            .map(|m| m.key.clone())
            .expect("registry must keep at least one reasoning_encrypted_replay entry")
    }

    fn assistant_call_with_reasoning(id: &str, reasoning: Option<String>) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: reasoning,
        }
    }

    #[test]
    fn rebuilds_items_from_encoded_history_for_encrypted_model() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Must resolve in the model registry with reasoning_encrypted_replay:
        // true; an unknown identifier makes the reconstruct gate return early
        // and the rebuild never runs.
        let model = first_encrypted_replay_model_key();
        let items = vec![json!({"type":"reasoning","encrypted_content":"ENC"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(
                &model, None, &items,
            )),
        )];
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &messages,
            &FxHashMap::default(),
        );
        assert_eq!(rebuilt.get("call-1").map(Arc::as_ref), Some(&items));
    }

    #[test]
    fn live_side_channel_takes_precedence_over_history() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let stale = vec![json!({"encrypted_content":"OLD"})];
        let fresh = Arc::new(vec![json!({"encrypted_content":"NEW"})]);
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(
                &model, None, &stale,
            )),
        )];
        let mut live: ReasoningItemMap = ReasoningItemMap::default();
        live.insert("call-1".to_string(), Arc::clone(&fresh));
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &messages,
            &live,
        );
        // The in-memory side channel (freshest for the current turn) wins;
        // archived values must never overwrite it.
        assert_eq!(rebuilt.get("call-1"), Some(&fresh));
        assert!(
            Arc::ptr_eq(rebuilt.get("call-1").expect("live entry survives"), &fresh),
            "live reasoning items must be shared, not deep-cloned"
        );
    }

    #[test]
    fn non_encrypted_model_is_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let items = vec![json!({"encrypted_content":"ENC"})];
        // Even if history carries an encrypted marker, models without the
        // encrypted-replay flag never rebuild (they return the live map
        // as-is). Cover both the explicitly-unregistered tag and a registered
        // entry that does not declare the flag.
        let registered_plain = crate::ai::model_names::all()
            .iter()
            .find(|m| !m.reasoning_encrypted_replay)
            .map(|m| m.key.clone())
            .expect("registry must keep at least one non-encrypted-replay entry");
        for model in ["synthetic-unregistered-model".to_string(), registered_plain] {
            let messages = vec![assistant_call_with_reasoning(
                "call-1",
                Some(encode_encrypted_reasoning_replay_state(
                    &model, None, &items,
                )),
            )];
            let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
                &model,
                TEST_ENDPOINT,
                TEST_SESSION,
                &messages,
                &FxHashMap::default(),
            );
            assert!(rebuilt.is_empty(), "{model}");
        }
    }

    #[test]
    fn runtime_disable_env_short_circuits_reconstruction() {
        use crate::ai::history::compress::encrypted_reasoning_replay_runtime_enabled;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save and restore the process-level env var to avoid polluting parallel tests.
        let saved = std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY").ok();

        // SAFETY: this test sets/removes the same key serially and restores it
        // before exiting; it never asserts concurrently with other tests that
        // depend on this env var.
        unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "1") };
        assert!(!encrypted_reasoning_replay_runtime_enabled());
        let model = first_encrypted_replay_model_key();
        let items = vec![json!({"encrypted_content":"ENC"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(
                &model, None, &items,
            )),
        )];
        // With replay disabled, no rebuild happens even if the model is capable
        // and history carries an encoded blob.
        assert!(
            reconstruct_encrypted_reasoning_items_for_model(
                &model,
                TEST_ENDPOINT,
                TEST_SESSION,
                &messages,
                &FxHashMap::default()
            )
            .is_empty()
        );

        unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "0") };
        assert!(encrypted_reasoning_replay_runtime_enabled());

        unsafe { std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY") };
        assert!(encrypted_reasoning_replay_runtime_enabled());

        match saved {
            Some(v) => unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", v) },
            None => unsafe { std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY") },
        }
    }

    #[test]
    fn repeated_reconstruction_reuses_cached_decoded_items() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let items = vec![json!({
            "type": "reasoning",
            "encrypted_content": "CACHE_REUSE_UNIQUE_ENC",
        })];
        let messages = vec![assistant_call_with_reasoning(
            "call-cache-reuse",
            Some(encode_encrypted_reasoning_replay_state(
                &model, None, &items,
            )),
        )];
        let first = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &messages,
            &ReasoningItemMap::default(),
        );
        let second = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &messages,
            &ReasoningItemMap::default(),
        );
        assert_eq!(first.get("call-cache-reuse").map(Arc::as_ref), Some(&items));
        assert!(
            Arc::ptr_eq(
                first.get("call-cache-reuse").expect("first decode"),
                second.get("call-cache-reuse").expect("second decode"),
            ),
            "the second reconstruction must reuse the cached decode"
        );
    }

    #[test]
    fn changed_encoded_blob_does_not_hit_the_decode_cache() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let first_items = vec![json!({"encrypted_content": "CACHE_CHANGE_FIRST"})];
        let second_items = vec![json!({"encrypted_content": "CACHE_CHANGE_SECOND"})];
        let first_messages = vec![assistant_call_with_reasoning(
            "call-cache-change",
            Some(encode_encrypted_reasoning_replay_state(
                &model,
                None,
                &first_items,
            )),
        )];
        let second_messages = vec![assistant_call_with_reasoning(
            "call-cache-change",
            Some(encode_encrypted_reasoning_replay_state(
                &model,
                None,
                &second_items,
            )),
        )];
        let first = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &first_messages,
            &ReasoningItemMap::default(),
        );
        let second = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &second_messages,
            &ReasoningItemMap::default(),
        );
        assert_eq!(
            second.get("call-cache-change").map(Arc::as_ref),
            Some(&second_items)
        );
        assert!(!Arc::ptr_eq(
            first.get("call-cache-change").expect("first decode"),
            second.get("call-cache-change").expect("second decode"),
        ));
    }

    #[test]
    fn endpoint_identity_binds_replay_to_the_serving_resource() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let items = vec![json!({"encrypted_content": "ENDPOINT_BOUND"})];
        let serving_identity = crate::ai::models::endpoint_resource_identity(TEST_ENDPOINT)
            .expect("test endpoint must parse");
        let messages = vec![
            assistant_call_with_reasoning(
                "call-resource-match",
                Some(encode_encrypted_reasoning_replay_state(
                    &model,
                    Some(&serving_identity),
                    &items,
                )),
            ),
            assistant_call_with_reasoning(
                "call-resource-mismatch",
                Some(encode_encrypted_reasoning_replay_state(
                    &model,
                    Some("a-different-resource"),
                    &items,
                )),
            ),
            assistant_call_with_reasoning(
                "call-resource-v1",
                Some(encode_encrypted_reasoning_replay_state(
                    &model, None, &items,
                )),
            ),
        ];
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
            &model,
            TEST_ENDPOINT,
            TEST_SESSION,
            &messages,
            &ReasoningItemMap::default(),
        );
        assert!(
            rebuilt.contains_key("call-resource-match"),
            "a marker created by the serving resource must replay"
        );
        assert!(
            !rebuilt.contains_key("call-resource-mismatch"),
            "a marker created by another backing resource must never replay"
        );
        assert!(
            rebuilt.contains_key("call-resource-v1"),
            "markers without an identity must keep the model-only replay rule"
        );
    }

    /// A cross-resource rejection repairs the next attempt: the rejected
    /// owner disappears while every other historical owner and the in-flight
    /// turn's fresh items keep replaying.
    #[test]
    fn rejected_cross_resource_keys_are_omitted_by_the_next_reconstruction() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        // A dedicated session keeps the process-wide rejection ledger from
        // leaking into the other reconstruction tests.
        let session = "reasoning-replay-repair-test-session";
        let stale = vec![json!({"id":"rs_stale","encrypted_content":"STALE"})];
        let other = vec![json!({"id":"rs_other","encrypted_content":"OTHER"})];
        let fresh = Arc::new(vec![json!({"id":"rs_fresh","encrypted_content":"FRESH"})]);
        let messages = vec![
            assistant_call_with_reasoning(
                "call-cross-resource",
                Some(encode_encrypted_reasoning_replay_state(
                    &model, None, &stale,
                )),
            ),
            assistant_call_with_reasoning(
                "call-other-resource",
                Some(encode_encrypted_reasoning_replay_state(
                    &model, None, &other,
                )),
            ),
            assistant_call_with_reasoning("call-fresh", None),
        ];
        let mut live: ReasoningItemMap = ReasoningItemMap::default();
        live.insert("call-fresh".to_string(), Arc::clone(&fresh));

        let mut spliced = SplicedReplayKeys::default();
        let first = reconstruct_encrypted_reasoning_items_for_model_recorded(
            &model,
            TEST_ENDPOINT,
            session,
            Path::new(""),
            &messages,
            &live,
            &mut spliced,
        );
        assert_eq!(
            spliced.all(),
            vec![
                "call-cross-resource".to_string(),
                "call-other-resource".to_string(),
                "call-fresh".to_string(),
            ],
            "the attempt must record every replay owner it splices"
        );
        assert_eq!(first.len(), 3);

        reject_encrypted_reasoning_replay_keys(
            session,
            &test_endpoint_identity(),
            &model,
            &["call-cross-resource".to_string()],
        );

        let mut retried_spliced = SplicedReplayKeys::default();
        let retried = reconstruct_encrypted_reasoning_items_for_model_recorded(
            &model,
            TEST_ENDPOINT,
            session,
            Path::new(""),
            &messages,
            &live,
            &mut retried_spliced,
        );
        assert!(
            !retried.contains_key("call-cross-resource"),
            "the rejected cross-resource owner must be omitted by the repair retry"
        );
        assert_eq!(
            retried.get("call-other-resource").map(Arc::as_ref),
            Some(&other),
            "unrelated historical owners must keep replaying"
        );
        assert_eq!(
            retried.get("call-fresh"),
            Some(&fresh),
            "the in-flight turn's fresh items must be untouched by a rejection"
        );
        assert_eq!(
            retried_spliced.all(),
            vec!["call-other-resource".to_string(), "call-fresh".to_string(),]
        );
    }

    /// The repair drops the persisted tier on the first rejection and the
    /// live tier only on the next one. Dropping only stages the owners: the
    /// ledger is untouched until a resend without them is accepted, and then
    /// both dropped tiers stay blocked for later reconstructions in the same
    /// session.
    #[test]
    fn cross_resource_repair_drops_persisted_tier_before_live_tier() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let session = "reasoning-replay-tier-order-test-session";
        let persisted_a = vec![json!({"id":"rs_tier_a","encrypted_content":"TIER_A"})];
        let persisted_b = vec![json!({"id":"rs_tier_b","encrypted_content":"TIER_B"})];
        let fresh = Arc::new(vec![
            json!({"id":"rs_tier_live","encrypted_content":"LIVE"}),
        ]);
        let messages = vec![
            assistant_call_with_reasoning(
                "call-tier-persisted-a",
                Some(encode_encrypted_reasoning_replay_state(
                    &model,
                    None,
                    &persisted_a,
                )),
            ),
            assistant_call_with_reasoning("call-tier-live", None),
            assistant_call_with_reasoning(
                "call-tier-persisted-b",
                Some(encode_encrypted_reasoning_replay_state(
                    &model,
                    None,
                    &persisted_b,
                )),
            ),
        ];
        let mut live = ReasoningItemMap::default();
        live.insert("call-tier-live".to_string(), Arc::clone(&fresh));
        let reconstruct = |spliced: &mut SplicedReplayKeys| {
            reconstruct_encrypted_reasoning_items_for_model_recorded(
                &model,
                TEST_ENDPOINT,
                session,
                Path::new(""),
                &messages,
                &live,
                spliced,
            )
        };

        let mut spliced = SplicedReplayKeys::default();
        let mut items = reconstruct(&mut spliced);
        assert_eq!(
            spliced.tier(ReplayTier::Persisted),
            vec![
                "call-tier-persisted-a".to_string(),
                "call-tier-persisted-b".to_string(),
            ]
        );
        assert_eq!(
            spliced.tier(ReplayTier::Live),
            vec!["call-tier-live".to_string()]
        );

        // First rejection: only the persisted tier is dropped.
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            Some((ReplayTier::Persisted, 2))
        );
        assert_eq!(items.len(), 1);
        assert_eq!(items.get("call-tier-live"), Some(&fresh));
        // Not committed yet: a fresh reconstruction still splices every tier.
        let mut uncommitted = SplicedReplayKeys::default();
        assert_eq!(reconstruct(&mut uncommitted).len(), 3);
        assert_eq!(uncommitted.all().len(), 3);

        // Second rejection: the live tier goes too.
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            Some((ReplayTier::Live, 1))
        );
        assert!(items.is_empty());
        // Nothing left to drop: the rejection would be final.
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            None
        );

        // The resend without both tiers is accepted.
        commit_dropped_replay_tiers(session, &model, TEST_ENDPOINT, Path::new(""), &spliced);

        // Later turns stay repaired: neither tier is rebuilt.
        let mut later = SplicedReplayKeys::default();
        assert!(reconstruct(&mut later).is_empty());
        assert!(later.all().is_empty());
    }

    /// A repair that never gets a resend through records nothing: every tier
    /// is dropped once, the final rejection stops the loop, and the innocent
    /// groups of the dropped tiers keep replaying in later attempts.
    #[test]
    fn failed_cross_resource_repair_leaves_the_ledger_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let session = "reasoning-replay-failed-repair-test-session";
        let persisted = vec![json!({"id":"rs_failed_p","encrypted_content":"FAILED_P"})];
        let fresh = Arc::new(vec![
            json!({"id":"rs_failed_live","encrypted_content":"FAILED_LIVE"}),
        ]);
        let messages = vec![
            assistant_call_with_reasoning(
                "call-failed-persisted",
                Some(encode_encrypted_reasoning_replay_state(
                    &model, None, &persisted,
                )),
            ),
            assistant_call_with_reasoning("call-failed-live", None),
        ];
        let mut live = ReasoningItemMap::default();
        live.insert("call-failed-live".to_string(), Arc::clone(&fresh));
        let reconstruct = |spliced: &mut SplicedReplayKeys| {
            reconstruct_encrypted_reasoning_items_for_model_recorded(
                &model,
                TEST_ENDPOINT,
                session,
                Path::new(""),
                &messages,
                &live,
                spliced,
            )
        };

        let mut spliced = SplicedReplayKeys::default();
        let mut items = reconstruct(&mut spliced);
        assert_eq!(items.len(), 2);
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            Some((ReplayTier::Persisted, 1))
        );
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            Some((ReplayTier::Live, 1))
        );
        assert!(items.is_empty());
        // Every tier was dropped once: the rejection is final and no further
        // resend is attempted.
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            None
        );

        // No resend was accepted, so nothing is committed and both groups
        // replay again in the next attempt.
        let mut later = SplicedReplayKeys::default();
        let rebuilt = reconstruct(&mut later);
        assert_eq!(
            later.all(),
            vec![
                "call-failed-persisted".to_string(),
                "call-failed-live".to_string(),
            ]
        );
        assert_eq!(
            rebuilt.get("call-failed-persisted").map(Arc::as_ref),
            Some(&persisted)
        );
        assert_eq!(rebuilt.get("call-failed-live"), Some(&fresh));
    }

    /// The size cap evicts the oldest owner instead of forgetting every
    /// rejection, and a repeated key neither grows the ledger nor refreshes
    /// its position.
    #[test]
    fn rejected_replay_ledger_evicts_oldest_entry_at_cap() {
        let mut ledger = RejectedReplayKeys::default();
        for index in 0..REJECTED_REPLAY_KEYS_MAX_ENTRIES {
            ledger.insert(format!("key-{index}"));
        }
        ledger.insert("key-0".to_string());
        assert_eq!(ledger.keys.len(), REJECTED_REPLAY_KEYS_MAX_ENTRIES);
        assert_eq!(ledger.order.len(), REJECTED_REPLAY_KEYS_MAX_ENTRIES);

        ledger.insert("key-newest".to_string());
        assert_eq!(ledger.keys.len(), REJECTED_REPLAY_KEYS_MAX_ENTRIES);
        assert_eq!(ledger.order.len(), REJECTED_REPLAY_KEYS_MAX_ENTRIES);
        assert!(!ledger.contains("key-0"), "the oldest entry is evicted");
        assert!(ledger.contains("key-1"));
        assert!(ledger.contains("key-newest"), "the newest entry survives");
    }

    /// Without persisted decoding (model lacks the capability, or runtime
    /// replay is disabled) live items are still spliced, so they must still be
    /// recorded for the repair and filtered by the rejected-replay ledger.
    #[test]
    fn live_items_are_recorded_and_ledger_filtered_without_persisted_decoding() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY").ok();
        let plain_model = crate::ai::model_names::all()
            .iter()
            .find(|m| !m.reasoning_encrypted_replay)
            .map(|m| m.key.clone())
            .expect("registry must keep at least one non-encrypted-replay entry");
        let encrypted_model = first_encrypted_replay_model_key();
        // Each case runs under its own session so the ledger state of one
        // cannot satisfy the other's assertions.
        for (case, model, disable_runtime) in [
            ("plain-model", plain_model, false),
            ("runtime-disabled", encrypted_model, true),
        ] {
            // SAFETY: serialized by ENV_LOCK and restored below.
            unsafe {
                if disable_runtime {
                    std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "1");
                } else {
                    std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY");
                }
            }
            let session = format!("reasoning-replay-live-only-{case}");
            let persisted = vec![json!({"encrypted_content":"NOT_DECODED"})];
            let kept = Arc::new(vec![json!({"id":"rs_kept","encrypted_content":"KEPT"})]);
            let rejected = Arc::new(vec![json!({"id":"rs_rej","encrypted_content":"REJ"})]);
            let messages = vec![
                assistant_call_with_reasoning(
                    "call-live-only-persisted",
                    Some(encode_encrypted_reasoning_replay_state(
                        &model, None, &persisted,
                    )),
                ),
                assistant_call_with_reasoning("call-live-only-kept", None),
                assistant_call_with_reasoning("call-live-only-rejected", None),
            ];
            let mut live = ReasoningItemMap::default();
            live.insert("call-live-only-kept".to_string(), Arc::clone(&kept));
            live.insert("call-live-only-rejected".to_string(), Arc::clone(&rejected));

            let mut spliced = SplicedReplayKeys::default();
            let first = reconstruct_encrypted_reasoning_items_for_model_recorded(
                &model,
                TEST_ENDPOINT,
                &session,
                Path::new(""),
                &messages,
                &live,
                &mut spliced,
            );
            assert!(
                !first.contains_key("call-live-only-persisted"),
                "{case}: persisted markers must not be decoded"
            );
            assert_eq!(first.len(), 2, "{case}");
            assert_eq!(
                spliced.tier(ReplayTier::Live),
                vec![
                    "call-live-only-kept".to_string(),
                    "call-live-only-rejected".to_string(),
                ],
                "{case}: spliced live owners must be recorded"
            );
            assert!(spliced.tier(ReplayTier::Persisted).is_empty(), "{case}");

            reject_encrypted_reasoning_replay_keys(
                &session,
                &test_endpoint_identity(),
                &model,
                &["call-live-only-rejected".to_string()],
            );
            let mut retried_spliced = SplicedReplayKeys::default();
            let retried = reconstruct_encrypted_reasoning_items_for_model_recorded(
                &model,
                TEST_ENDPOINT,
                &session,
                Path::new(""),
                &messages,
                &live,
                &mut retried_spliced,
            );
            assert!(
                !retried.contains_key("call-live-only-rejected"),
                "{case}: a ledgered live owner must be filtered"
            );
            assert_eq!(retried.get("call-live-only-kept"), Some(&kept), "{case}");
            assert_eq!(
                retried_spliced.all(),
                vec!["call-live-only-kept".to_string()],
                "{case}"
            );
        }

        match saved {
            Some(v) => unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", v) },
            None => unsafe { std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY") },
        }
    }

    /// A rejection committed to the session's history database survives the
    /// process that recorded it: a ledger scope that never saw the in-memory
    /// write reads the durable entry back and suppresses the owner only for the
    /// endpoint identity that rejected it.
    #[test]
    fn durable_rejected_replay_ledger_is_endpoint_scoped_across_a_fresh_read() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = first_encrypted_replay_model_key();
        let history_path =
            std::env::temp_dir().join(format!("ai-rejected-replay-{}.db", uuid::Uuid::new_v4()));
        let rejecting_endpoint = TEST_ENDPOINT;
        // Differs from the rejecting endpoint only by path, so the durable entry
        // must carry the whole identity instead of a host-level key.
        let other_endpoint = "https://reasoning-replay.test/other";
        let rejecting_identity = test_endpoint_identity();
        let other_identity = crate::ai::models::endpoint_resource_identity(other_endpoint)
            .expect("other test endpoint must resolve an identity");
        assert_ne!(rejecting_identity, other_identity);

        let rejected = Arc::new(vec![
            json!({"id":"rs_durable","encrypted_content":"DURABLE"}),
        ]);
        let messages = vec![assistant_call_with_reasoning("call-durable-rejected", None)];
        let mut live = ReasoningItemMap::default();
        live.insert("call-durable-rejected".to_string(), Arc::clone(&rejected));

        // The writer session records the owner after an accepted resend without
        // it, exactly like the retry loop in `transport`.
        let writer_session = "reasoning-replay-durable-writer-session";
        let mut spliced = SplicedReplayKeys::default();
        let mut items = reconstruct_encrypted_reasoning_items_for_model_recorded(
            &model,
            rejecting_endpoint,
            writer_session,
            &history_path,
            &messages,
            &live,
            &mut spliced,
        );
        assert_eq!(
            spliced.tier(ReplayTier::Live),
            vec!["call-durable-rejected".to_string()]
        );
        assert_eq!(
            drop_next_rejected_replay_tier(&mut items, &mut spliced),
            Some((ReplayTier::Live, 1))
        );
        commit_dropped_replay_tiers(
            writer_session,
            &model,
            rejecting_endpoint,
            &history_path,
            &spliced,
        );

        // The durable entry leads with the replay-marker discriminator, then
        // carries the endpoint identity, model and owner.
        let durable =
            crate::ai::history::read_encrypted_reasoning_replay_rejections_sqlite(&history_path)
                .expect("durable ledger must be readable");
        assert_eq!(
            durable,
            vec![format!(
                "{}\u{1f}{rejecting_identity}\u{1f}{model}\u{1f}call-durable-rejected",
                crate::ai::history::compress::PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX
                    .trim_end_matches('\u{1f}')
            )]
        );

        // A fresh scope (new session, first read of this database) merges the
        // durable entry instead of relying on the writer's in-memory ledger.
        let reader_session = "reasoning-replay-durable-reader-session";
        let mut reader_spliced = SplicedReplayKeys::default();
        let suppressed = reconstruct_encrypted_reasoning_items_for_model_recorded(
            &model,
            rejecting_endpoint,
            reader_session,
            &history_path,
            &messages,
            &live,
            &mut reader_spliced,
        );
        assert!(
            !suppressed.contains_key("call-durable-rejected"),
            "a durable rejection must suppress its owner on the rejecting endpoint"
        );
        assert!(reader_spliced.all().is_empty());

        // The same owner is valid on another backing resource, so it replays and
        // is spliced for that endpoint.
        let other_session = "reasoning-replay-durable-other-endpoint-session";
        let mut other_spliced = SplicedReplayKeys::default();
        let replayed = reconstruct_encrypted_reasoning_items_for_model_recorded(
            &model,
            other_endpoint,
            other_session,
            &history_path,
            &messages,
            &live,
            &mut other_spliced,
        );
        assert_eq!(
            replayed.get("call-durable-rejected"),
            Some(&rejected),
            "a rejection recorded for one endpoint must not disable another"
        );
        assert_eq!(
            other_spliced.tier(ReplayTier::Live),
            vec!["call-durable-rejected".to_string()],
            "the live tier must be retained on the non-matching endpoint"
        );

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", history_path.display()));
        }
    }

    /// A transient durable-read failure must not mark the scope loaded: the
    /// failed call still fails open, and the next reconstruction retries the
    /// read instead of losing durable suppression until the process restarts.
    #[test]
    fn durable_rejection_read_error_retries_instead_of_disabling_suppression() {
        let session = "reasoning-replay-durable-read-retry-session";
        let history_path =
            std::env::temp_dir().join(format!("ai-rejected-retry-{}.db", uuid::Uuid::new_v4()));
        std::fs::write(&history_path, b"not a sqlite database")
            .expect("corrupt fixture must be writable");
        super::load_rejected_replay_keys(session, &history_path);

        let scope = (session.to_string(), history_path.clone());
        assert!(
            !super::REJECTED_REPLAY_KEYS
                .lock()
                .unwrap()
                .loaded
                .contains(&scope),
            "a failed read must leave the scope unloaded so a later call retries"
        );

        // Replace the corrupt fixture with a real ledger, as if the transient
        // SQLite error had cleared, and reload the same scope.
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", history_path.display()));
        }
        let entry = super::rejected_replay_entry("endpoint-identity", "model", "call-retried");
        crate::ai::history::write_encrypted_reasoning_replay_rejections_sqlite(
            &history_path,
            std::slice::from_ref(&entry),
        )
        .expect("durable ledger must be writable");

        super::load_rejected_replay_keys(session, &history_path);
        let ledger = super::REJECTED_REPLAY_KEYS.lock().unwrap();
        assert!(
            ledger.loaded.contains(&scope),
            "a successful retry must mark the scope loaded"
        );
        assert!(
            ledger.contains(&format!("{session}\u{1f}{entry}")),
            "the retried read must merge the durable entries"
        );
        drop(ledger);

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", history_path.display()));
        }
    }
}

#[cfg(test)]
mod force_off_effort_tests {
    use super::apply_thinking_force_off_effort;
    use crate::ai::provider::ApiProvider;

    const MODELHUB_ENDPOINT: &str = "https://dataagent-dev-llm.bytedance.net/v1";

    #[test]
    fn responses_model_force_off_maps_to_none() {
        // gpt-5.x (compatible provider, non-DashScope modelhub endpoint) routes to
        // NoThinkingDialect: the only thinking lever is reasoning_effort, so the force-off
        // fallback must emit "none" instead of the ladder's "low".
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::Compatible,
                "gpt-5.6-sol",
                MODELHUB_ENDPOINT,
                Some("low"),
            ),
            Some("none"),
        );
    }

    #[test]
    fn force_off_inactive_passes_effort_through() {
        // Normal requests (auto-detected no-thinking, user config, etc.) must keep the
        // configured/default effort — only the explicit override maps to "none".
        assert_eq!(
            apply_thinking_force_off_effort(
                false,
                ApiProvider::Compatible,
                "gpt-5.6-sol",
                MODELHUB_ENDPOINT,
                Some("xhigh"),
            ),
            Some("xhigh"),
        );
    }

    #[test]
    fn force_off_passes_effort_through_for_unsupported_dialects() {
        // MiniMax always-on reasoning and beam (whose gateway rejects "none"
        // with 400) have no off mechanism: the force-off fallback must keep
        // the current tier instead of emitting the bogus "none".
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::OpenCode,
                "mimo-v2.5-pro",
                "https://opencode.ai/inference/go/v1/chat/completions",
                Some("low"),
            ),
            Some("low"),
        );
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::OpenCode,
                "beam-free",
                "https://opencode.ai/inference/go/v1/chat/completions",
                Some("max"),
            ),
            Some("max"),
        );
    }

    #[test]
    fn force_off_keeps_effort_for_switch_based_dialects() {
        // DashScope `enable_thinking:false` is the real off-switch there, so the effort is left
        // untouched (sending "none" would be an unverified field on that gateway).
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::Alibaba,
                "qwen3.7-max-alibaba",
                crate::ai::provider::ALIBABA_DEFAULT_ENDPOINT,
                Some("low"),
            ),
            Some("low"),
        );
        // DeepSeek `thinking:{"type":"disabled"}` is the real off-switch there as well.
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::OpenCode,
                "deepseek-flash-opencode",
                "https://api.deepseek.com/v1",
                Some("low"),
            ),
            Some("low"),
        );
    }
}
