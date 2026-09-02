//! Dedup helpers for compress (extracted from mod.rs).

use serde_json::Value;

use super::super::types::Message;
use super::text_utils::{summarize_text, truncate_to_chars};
use super::tool_overflow::{is_non_compressible_tool, is_preserved_tool_overflow_content, tool_line_signature};
use super::tool_groups::recent_tool_group_message_indices;
use super::{
    is_persisted_reasoning_replay, normalize_whitespace, tool_message_indices, value_to_string,
    KEEP_RECENT_TOOL_GROUPS, KEEP_RECENT_TOOL_CALL_REASONING,
};

pub(super) fn dedup_adjacent(messages: &mut Vec<Message>) {
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
pub(super) fn keep_only_recent_reasoning_content(messages: &mut [Message]) {
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
                    !is_persisted_reasoning_replay(reasoning)
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
        // exact/encrypted replay is the protocol state of its tool-call message;
        // it cannot be trimmed alone while the message is still present.
        if m.reasoning_content
            .as_deref()
            .is_some_and(is_persisted_reasoning_replay)
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
pub(super) fn dedup_repeated_tool_results(
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
pub(super) struct NumberedReadFileResult {
    message_idx: usize,
    tool_call_id: String,
    path: String,
    lines: Vec<(usize, String)>,
}

pub(super) fn dedup_overlapping_read_file_results(
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
