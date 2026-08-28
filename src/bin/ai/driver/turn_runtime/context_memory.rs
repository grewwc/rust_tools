use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Value, json};

use crate::ai::history::{
    Message, ROLE_INTERNAL_NOTE, is_runtime_synthetic_user_message, last_real_user_index,
    message_billable_chars, value_to_string,
};

const MIN_RECOVERABLE_MEMORY_CHARS: usize = 8_192;
const MIN_NET_SAVINGS_CHARS: usize = 2_048;
const MAX_QUERY_TOKENS: usize = 768;
const MAX_DOCUMENT_TOKENS: usize = 192;
const MAX_FULL_REPRESENTATIVES: usize = 8;
const ALWAYS_KEEP_RECENT: usize = 2;
/// Floor for consolidating earlier memory-index notes into one merged note when
/// there are no fresh omissions this round. It is deliberately much lower than
/// `MIN_NET_SAVINGS_CHARS`: consolidation exists to bound fragment growth (one
/// index note per projection) rather than to bulk-save characters.
const MIN_CONSOLIDATION_SAVINGS_CHARS: usize = 256;

#[derive(Debug, Clone, Default)]
pub(super) struct MemoryProjectionStats {
    pub(super) before_chars: usize,
    pub(super) after_chars: usize,
    pub(super) removed_messages: usize,
    pub(super) selected_messages: usize,
    pub(super) saved_chars: usize,
    pub(super) index_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RecoveryTarget {
    File(String),
    Directory(String),
}

#[derive(Debug)]
struct Candidate {
    index: usize,
    chars: usize,
    tokens: FxHashSet<String>,
    recovery_targets: BTreeSet<RecoveryTarget>,
}

#[derive(Debug)]
struct RecoveryIndex {
    roots: BTreeMap<String, Vec<String>>,
    standalone_files: Vec<String>,
    directories: Vec<String>,
}

pub(super) fn has_dense_recoverable_memory(messages: &[Message]) -> bool {
    let total_chars = total_chars(messages);
    let recoverable_chars = messages
        .iter()
        .filter_map(recoverable_candidate)
        .map(|candidate| candidate.chars)
        .sum::<usize>();
    recoverable_chars >= MIN_RECOVERABLE_MEMORY_CHARS
        && recoverable_chars.saturating_mul(4) >= total_chars
}

/// Shrinks only old, derived metadata whose exact source is already archived.
///
/// All canonical messages remain untouched outside this request projection. Non-candidate messages
/// are copied byte-for-byte and in order; the current real-user tail is never considered. Omitted
/// entries and stale memory-index notes from earlier projections are replaced by one merged
/// hierarchical path index (their targets are unioned, so nothing becomes unreachable). Any
/// validation or net-savings failure returns without changing `messages`.
pub(super) fn apply_query_aware_memory_projection(
    messages: &mut Vec<Message>,
    target_chars: usize,
) -> MemoryProjectionStats {
    let before_chars = total_chars(messages);
    let mut stats = MemoryProjectionStats {
        before_chars,
        after_chars: before_chars,
        ..MemoryProjectionStats::default()
    };
    let Some(current_user_index) = last_real_user_index(messages) else {
        return stats;
    };
    // Consolidate index notes left by earlier projections: their targets are
    // unioned into the index built this round and the notes themselves are
    // removed, so note count stays bounded instead of growing one per turn.
    // A note whose `index=` payload cannot be parsed is kept verbatim, so a
    // malformed note never causes another note's paths to be dropped.
    let (stale_consumed_indices, stale_targets) =
        collect_stale_index_notes(messages, current_user_index);
    if !has_dense_recoverable_memory(messages) && stale_consumed_indices.len() < 2 {
        return stats;
    }

    let query_tokens = active_query_tokens(messages, current_user_index);
    let candidates = messages
        .iter()
        .take(current_user_index)
        .enumerate()
        .filter_map(|(index, message)| {
            let mut candidate = recoverable_candidate(message)?;
            candidate.index = index;
            Some(candidate)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() && stale_consumed_indices.is_empty() {
        return stats;
    }

    let selected = select_representatives(&candidates, &query_tokens, target_chars);
    let omitted = candidates
        .iter()
        .filter(|candidate| !selected.contains(&candidate.index))
        .collect::<Vec<_>>();
    if omitted.is_empty() && stale_consumed_indices.len() < 2 {
        return stats;
    }

    let fresh_targets = omitted
        .iter()
        .flat_map(|candidate| candidate.recovery_targets.iter().cloned())
        .collect::<BTreeSet<_>>();
    let recovery_targets: BTreeSet<RecoveryTarget> =
        fresh_targets.union(&stale_targets).cloned().collect();
    let omitted_chars = omitted
        .iter()
        .map(|candidate| candidate.chars)
        .sum::<usize>();
    let stale_chars = stale_consumed_indices
        .iter()
        .map(|index| message_billable_chars(&messages[*index]))
        .sum::<usize>();
    let removed_chars = omitted_chars.saturating_add(stale_chars);
    let Some(index_message) = build_index_message(omitted.len(), omitted_chars, &recovery_targets)
    else {
        return stats;
    };
    let index_chars = message_billable_chars(&index_message);
    let minimum_savings = MIN_NET_SAVINGS_CHARS.max(before_chars / 100);
    let savings_floor = if omitted.is_empty() {
        MIN_CONSOLIDATION_SAVINGS_CHARS
    } else {
        minimum_savings
    };
    if removed_chars.saturating_sub(index_chars) < savings_floor {
        return stats;
    }

    let mut removed_indices = omitted
        .iter()
        .map(|candidate| candidate.index)
        .collect::<FxHashSet<_>>();
    removed_indices.extend(stale_consumed_indices.iter().copied());
    let mut projected = messages
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed_indices.contains(index))
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    let insertion_index = (0..current_user_index)
        .filter(|index| !removed_indices.contains(index))
        .count();
    projected.insert(insertion_index, index_message.clone());

    if !projection_is_valid(
        messages,
        &projected,
        &removed_indices,
        insertion_index,
        &index_message,
        current_user_index,
        &recovery_targets,
    ) {
        return stats;
    }

    let after_chars = total_chars(&projected);
    if after_chars >= before_chars {
        return stats;
    }

    *messages = projected;
    stats.after_chars = after_chars;
    stats.removed_messages = omitted.len() + stale_consumed_indices.len();
    stats.selected_messages = selected.len();
    stats.saved_chars = before_chars - after_chars;
    stats.index_chars = index_chars;
    stats
}

fn recoverable_candidate(message: &Message) -> Option<Candidate> {
    if message.role != ROLE_INTERNAL_NOTE
        || message.tool_calls.is_some()
        || message.tool_call_id.is_some()
        || message.reasoning_content.is_some()
    {
        return None;
    }
    let text = value_to_string(&message.content);
    let trimmed = text.trim_start();
    if trimmed.starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX)
        || crate::ai::history::compress::is_context_compaction_state(message)
        || !is_recoverable_metadata(message, trimmed)
    {
        return None;
    }
    let recovery_targets = extract_recovery_targets(trimmed);
    if recovery_targets.is_empty() {
        return None;
    }
    Some(Candidate {
        index: 0,
        chars: message_billable_chars(message),
        tokens: tokenize(trimmed, MAX_DOCUMENT_TOKENS),
        recovery_targets,
    })
}

fn is_recoverable_metadata(message: &Message, text: &str) -> bool {
    crate::ai::history::compress::is_context_checkpoint_marker(message)
        || crate::ai::history::compress::is_compressed_tool_evidence_note(message)
        || crate::ai::history::compress::is_archive_note_text(text)
        || text.starts_with("[context-overflow-truncated]")
        || text.starts_with("[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]")
        || text.starts_with("[[PRESERVED_CONTENT_STUB_V1]]")
        || text.starts_with("compressed_tool_round:")
}

fn active_query_tokens(messages: &[Message], current_user_index: usize) -> FxHashSet<String> {
    let mut tokens = FxHashSet::default();
    let mut real_users = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, message)| {
            message.role == "user" && !is_runtime_synthetic_user_message(message)
        })
        .take(2)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    real_users.reverse();
    // The most recent real user is the live query the model is answering, so it
    // gets token-budget priority. Iterating newest-first keeps an oversized
    // older user message from filling MAX_QUERY_TOKENS and starving the current
    // instruction out of the query set that drives representative selection.
    for index in real_users.into_iter().rev() {
        extend_tokens(
            &mut tokens,
            &value_to_string(&messages[index].content),
            MAX_QUERY_TOKENS,
        );
    }
    if let Some(message) = messages[..current_user_index]
        .iter()
        .rev()
        .find(|message| message.role == "assistant" && message.tool_calls.is_none())
    {
        extend_tokens(
            &mut tokens,
            &value_to_string(&message.content),
            MAX_QUERY_TOKENS,
        );
    }
    tokens
}

fn select_representatives(
    candidates: &[Candidate],
    query_tokens: &FxHashSet<String>,
    target_chars: usize,
) -> FxHashSet<usize> {
    let mut selected = FxHashSet::default();
    let mut covered = FxHashSet::default();
    let mut used_chars = 0usize;
    let full_budget = (target_chars / 12).clamp(2_048, 16_000);

    for candidate in candidates.iter().rev().take(ALWAYS_KEEP_RECENT).rev() {
        selected.insert(candidate.index);
        used_chars = used_chars.saturating_add(candidate.chars);
        covered.extend(candidate.tokens.iter().cloned());
    }

    let mut document_frequency: FxHashMap<&str, usize> = FxHashMap::default();
    for candidate in candidates {
        for token in &candidate.tokens {
            *document_frequency.entry(token.as_str()).or_default() += 1;
        }
    }

    while selected.len() < MAX_FULL_REPRESENTATIVES && used_chars < full_budget {
        let remaining = full_budget - used_chars;
        let best = candidates
            .iter()
            .filter(|candidate| {
                !selected.contains(&candidate.index) && candidate.chars <= remaining
            })
            .max_by(|left, right| {
                let left_gain = marginal_gain(
                    left,
                    query_tokens,
                    &covered,
                    &document_frequency,
                    candidates.len(),
                );
                let right_gain = marginal_gain(
                    right,
                    query_tokens,
                    &covered,
                    &document_frequency,
                    candidates.len(),
                );
                left_gain
                    .saturating_mul(right.chars.saturating_add(256))
                    .cmp(&right_gain.saturating_mul(left.chars.saturating_add(256)))
                    .then_with(|| left.index.cmp(&right.index))
            });
        let Some(candidate) = best else {
            break;
        };
        selected.insert(candidate.index);
        used_chars = used_chars.saturating_add(candidate.chars);
        covered.extend(candidate.tokens.iter().cloned());
    }
    selected
}

fn marginal_gain(
    candidate: &Candidate,
    query_tokens: &FxHashSet<String>,
    covered: &FxHashSet<String>,
    document_frequency: &FxHashMap<&str, usize>,
    candidate_count: usize,
) -> usize {
    let relevance = candidate
        .tokens
        .iter()
        .filter(|token| query_tokens.contains(*token))
        .map(|token| {
            let frequency = document_frequency
                .get(token.as_str())
                .copied()
                .unwrap_or(candidate_count);
            16usize.saturating_add(256 / frequency.saturating_add(1))
        })
        .sum::<usize>();
    let novel_coverage = candidate
        .tokens
        .iter()
        .filter(|token| !covered.contains(*token))
        .count();
    relevance
        .saturating_mul(8)
        .saturating_add(novel_coverage.saturating_mul(2))
        .saturating_add(candidate.index.min(32))
}

fn build_index_message(
    omitted_entries: usize,
    omitted_chars: usize,
    targets: &BTreeSet<RecoveryTarget>,
) -> Option<Message> {
    let index = build_recovery_index(targets)?;
    let payload = json!({
        "omitted_entries": omitted_entries,
        "omitted_chars": omitted_chars,
        "roots": index.roots,
        "standalone_files": index.standalone_files,
        "directories": index.directories,
    });
    let encoded = serde_json::to_string(&payload).ok()?;
    Some(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "{}\n\
             Older recoverable memory metadata was omitted only from this request projection. \
             Canonical history and archived evidence are unchanged. Search with search_overflow \
             (scope=all), then read the exact source before relying on it.\nindex={encoded}",
            crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX,
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

fn build_recovery_index(targets: &BTreeSet<RecoveryTarget>) -> Option<RecoveryIndex> {
    let mut roots: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut standalone_files = Vec::new();
    let mut directories = Vec::new();
    for target in targets {
        match target {
            RecoveryTarget::Directory(path) => directories.push(path.clone()),
            RecoveryTarget::File(path) => {
                let path_ref = Path::new(path);
                let Some(parent) = path_ref.parent().and_then(Path::to_str) else {
                    standalone_files.push(path.clone());
                    continue;
                };
                let Some(file_name) = path_ref.file_name().and_then(|name| name.to_str()) else {
                    standalone_files.push(path.clone());
                    continue;
                };
                if Path::new(parent).join(file_name).to_str() != Some(path.as_str()) {
                    standalone_files.push(path.clone());
                    continue;
                }
                roots
                    .entry(parent.to_string())
                    .or_default()
                    .push(file_name.to_string());
            }
        }
    }
    for files in roots.values_mut() {
        files.sort();
        files.dedup();
    }
    standalone_files.sort();
    standalone_files.dedup();
    directories.sort();
    directories.dedup();
    Some(RecoveryIndex {
        roots,
        standalone_files,
        directories,
    })
}

fn projection_is_valid(
    original: &[Message],
    projected: &[Message],
    removed_indices: &FxHashSet<usize>,
    insertion_index: usize,
    index_message: &Message,
    current_user_index: usize,
    recovery_targets: &BTreeSet<RecoveryTarget>,
) -> bool {
    if projected.get(insertion_index) != Some(index_message) {
        return false;
    }
    let mut without_index = projected.to_vec();
    without_index.remove(insertion_index);
    let expected = original
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed_indices.contains(index))
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    if without_index != expected {
        return false;
    }

    let removed_before_current = removed_indices
        .iter()
        .filter(|index| **index < current_user_index)
        .count();
    let projected_user_index = current_user_index
        .saturating_sub(removed_before_current)
        .saturating_add(1);
    if projected.get(projected_user_index..) != Some(&original[current_user_index..]) {
        return false;
    }

    let Some(index) = build_recovery_index(recovery_targets) else {
        return false;
    };
    recovery_targets_from_index(&index) == *recovery_targets
}

fn recovery_targets_from_index(index: &RecoveryIndex) -> BTreeSet<RecoveryTarget> {
    let mut targets = BTreeSet::new();
    for (root, files) in &index.roots {
        for file in files {
            let path = Path::new(root).join(file);
            let Some(path) = path.to_str() else {
                continue;
            };
            targets.insert(RecoveryTarget::File(path.to_string()));
        }
    }
    targets.extend(
        index
            .standalone_files
            .iter()
            .cloned()
            .map(RecoveryTarget::File),
    );
    targets.extend(
        index
            .directories
            .iter()
            .cloned()
            .map(RecoveryTarget::Directory),
    );
    targets
}

/// Parses the `index=` JSON payload of a `[query-memory-index-v1]` note back
/// into the exact recovery-target set. Returns `None` (fail-open) for any note
/// whose payload is missing or malformed; callers keep such notes verbatim so
/// their paths stay reachable.
fn index_targets_from_note(text: &str) -> Option<BTreeSet<RecoveryTarget>> {
    let encoded = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("index="))?;
    let payload: Value = serde_json::from_str(encoded).ok()?;
    let mut roots = BTreeMap::new();
    let roots_object = payload.get("roots")?.as_object()?;
    for (root, files) in roots_object {
        let files = files.as_array()?;
        let mut names = Vec::with_capacity(files.len());
        for file in files {
            names.push(file.as_str()?.to_string());
        }
        roots.insert(root.clone(), names);
    }
    let mut standalone_files = Vec::new();
    if let Some(files) = payload.get("standalone_files") {
        for file in files.as_array()? {
            standalone_files.push(file.as_str()?.to_string());
        }
    }
    let mut directories = Vec::new();
    if let Some(dirs) = payload.get("directories") {
        for dir in dirs.as_array()? {
            directories.push(dir.as_str()?.to_string());
        }
    }
    Some(recovery_targets_from_index(&RecoveryIndex {
        roots,
        standalone_files,
        directories,
    }))
}

/// Finds earlier `[query-memory-index-v1]` notes before the current user and
/// returns the indices whose `index=` payload parsed cleanly (these are removed
/// by the caller once their targets are unioned into the merged note) plus the
/// union of their targets. Unparseable notes are skipped so they stay in place:
/// consolidating must never drop a path that exists only in an old note.
fn collect_stale_index_notes(
    messages: &[Message],
    current_user_index: usize,
) -> (Vec<usize>, BTreeSet<RecoveryTarget>) {
    let mut consumed = Vec::new();
    let mut targets = BTreeSet::new();
    for (index, message) in messages.iter().enumerate().take(current_user_index) {
        if message.role != ROLE_INTERNAL_NOTE
            || message.tool_calls.is_some()
            || message.tool_call_id.is_some()
            || message.reasoning_content.is_some()
        {
            continue;
        }
        let text = value_to_string(&message.content);
        let trimmed = text.trim_start();
        if !trimmed.starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX) {
            continue;
        }
        let Some(parsed) = index_targets_from_note(trimmed) else {
            continue;
        };
        consumed.push(index);
        targets.extend(parsed);
    }
    (consumed, targets)
}

fn extract_recovery_targets(text: &str) -> BTreeSet<RecoveryTarget> {
    let mut targets = BTreeSet::new();
    let checkpoint_prefix = "[context_checkpoint path=";
    let mut remainder = text;
    while let Some(start) = remainder.find(checkpoint_prefix) {
        let value = &remainder[start + checkpoint_prefix.len()..];
        let Some((path, end)) = parse_checkpoint_target(value) else {
            remainder = value;
            continue;
        };
        insert_target(&mut targets, &path, false);
        remainder = &value[end + 1..];
    }

    for line in text.lines() {
        let line = line.trim_start();
        let line = line.strip_prefix("- ").unwrap_or(line);
        for prefix in [
            "archive_file_path: ",
            "file_path: ",
            "full original archived at: ",
            "archived at: ",
            "归档文件: ",
        ] {
            if let Some(value) = line.strip_prefix(prefix) {
                insert_target(&mut targets, value, false);
            }
        }
        if let Some(value) = line.strip_prefix("归档目录: ") {
            insert_target(&mut targets, value, true);
        }
    }
    targets
}

fn parse_checkpoint_target(value: &str) -> Option<(String, usize)> {
    let candidates = value
        .match_indices(']')
        .filter_map(|(end, _)| {
            let suffix = &value[end + 1..];
            if suffix
                .chars()
                .next()
                .is_some_and(|character| !character.is_whitespace())
            {
                return None;
            }
            let path = &value[..end];
            Path::new(path)
                .is_absolute()
                .then(|| (path.to_string(), end))
        })
        .collect::<Vec<_>>();
    if candidates.len() == 1 {
        return candidates.into_iter().next();
    }

    let mut existing = candidates
        .into_iter()
        .filter(|(path, _)| Path::new(path).exists());
    let target = existing.next()?;
    existing.next().is_none().then_some(target)
}

fn insert_target(targets: &mut BTreeSet<RecoveryTarget>, value: &str, directory: bool) {
    if !Path::new(value).is_absolute() {
        return;
    }
    if directory {
        targets.insert(RecoveryTarget::Directory(value.to_string()));
    } else {
        targets.insert(RecoveryTarget::File(value.to_string()));
    }
}

fn tokenize(text: &str, limit: usize) -> FxHashSet<String> {
    let mut tokens = FxHashSet::default();
    extend_tokens(&mut tokens, text, limit);
    tokens
}

fn extend_tokens(tokens: &mut FxHashSet<String>, text: &str, limit: usize) {
    if tokens.len() >= limit {
        return;
    }
    let mut word = String::new();
    let mut previous_cjk = None;
    for character in text.chars() {
        if is_cjk(character) {
            flush_word(tokens, &mut word, limit);
            if let Some(previous) = previous_cjk {
                insert_token(tokens, format!("{previous}{character}"), limit);
            }
            previous_cjk = Some(character);
        } else if character.is_alphanumeric()
            || matches!(character, '_' | '/' | '.' | '-' | '\\' | ':')
        {
            previous_cjk = None;
            for lower in character.to_lowercase() {
                word.push(lower);
            }
        } else {
            previous_cjk = None;
            flush_word(tokens, &mut word, limit);
        }
        if tokens.len() >= limit {
            break;
        }
    }
    flush_word(tokens, &mut word, limit);
}

fn flush_word(tokens: &mut FxHashSet<String>, word: &mut String, limit: usize) {
    if word.is_empty() {
        return;
    }
    let value = std::mem::take(word);
    insert_token(tokens, value.clone(), limit);
    for component in value.split(['/', '\\', '.', ':', '-']) {
        insert_token(tokens, component.to_string(), limit);
    }
}

fn insert_token(tokens: &mut FxHashSet<String>, token: String, limit: usize) {
    let token = token.trim_matches('_');
    if tokens.len() < limit
        && (2..=96).contains(&token.chars().count())
        && !is_boilerplate_token(token)
    {
        tokens.insert(token.to_string());
    }
}

fn is_boilerplate_token(token: &str) -> bool {
    matches!(
        token,
        "archive"
            | "archived"
            | "assistant"
            | "context"
            | "evidence"
            | "file"
            | "file_path"
            | "history"
            | "internal"
            | "message"
            | "result"
            | "source"
            | "tool"
            | "user"
    )
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}

fn total_chars(messages: &[Message]) -> usize {
    messages.iter().map(message_billable_chars).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: impl Into<String>) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn checkpoint(index: usize, topic: &str, padding: usize) -> Message {
        message(
            ROLE_INTERNAL_NOTE,
            format!(
                "[context_checkpoint path=/tmp/session.assets/context-checkpoints/{index}.md] \
                 {topic} {}",
                "x".repeat(padding)
            ),
        )
    }

    #[test]
    fn query_projection_keeps_relevant_and_recent_entries_and_indexes_every_omission() {
        let system = message("system", "system stays exact");
        let current_user = message("user", "continue the payments_router migration");
        let mut messages = vec![system.clone()];
        for index in 0..16 {
            let topic = if index == 3 {
                "payments_router migration"
            } else {
                "unrelated archived topic"
            };
            messages.push(checkpoint(index, topic, 1_200));
        }
        messages.push(message(
            ROLE_INTERNAL_NOTE,
            "important live note without an archive path",
        ));
        messages.push(current_user.clone());

        let before = messages.clone();
        let stats = apply_query_aware_memory_projection(&mut messages, 120_000);

        assert!(stats.removed_messages > 0);
        assert!(stats.saved_chars >= MIN_NET_SAVINGS_CHARS);
        assert_eq!(messages.first(), Some(&system));
        assert_eq!(messages.last(), Some(&current_user));
        assert!(messages.iter().any(|message| {
            value_to_string(&message.content).contains("payments_router migration")
        }));
        assert!(messages.iter().any(|message| {
            value_to_string(&message.content)
                .contains("important live note without an archive path")
        }));
        let index_text = messages
            .iter()
            .find_map(|message| {
                let text = value_to_string(&message.content);
                text.starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX)
                    .then_some(text)
            })
            .expect("memory index");
        for index in 0..14 {
            let original = value_to_string(&before[index + 1].content);
            if !messages
                .iter()
                .any(|message| value_to_string(&message.content) == original)
            {
                assert!(
                    index_text.contains(&format!("{index}.md")),
                    "omitted checkpoint {index} must remain exactly reachable"
                );
            }
        }
    }

    #[test]
    fn projection_preserves_the_entire_active_tail_and_tool_protocol() {
        let mut tool_call = message("assistant", "");
        tool_call.tool_calls = Some(vec![crate::ai::types::ToolCall {
            id: "call-1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{\"file_path\":\"src/main.rs\"}".to_string(),
            },
        }]);
        let mut tool_result = message("tool", "exact current result");
        tool_result.tool_call_id = Some("call-1".to_string());
        let current_user = message("user", "inspect src/main.rs");
        let tail = vec![current_user, tool_call, tool_result];
        let mut messages = vec![message("system", "system")];
        for index in 0..12 {
            messages.push(checkpoint(index, "old topic", 1_000));
        }
        messages.extend(tail.clone());

        let stats = apply_query_aware_memory_projection(&mut messages, 80_000);

        assert!(stats.removed_messages > 0);
        assert_eq!(messages[messages.len() - tail.len()..], tail);
    }

    #[test]
    fn no_net_benefit_fails_open_without_changing_messages() {
        let mut messages = vec![message("system", "system")];
        for index in 0..80 {
            messages.push(message(
                ROLE_INTERNAL_NOTE,
                format!(
                    "[context_checkpoint path=/tmp/{index}-{}/{index}.md]",
                    "unique-directory-component".repeat(8)
                ),
            ));
        }
        messages.push(message("user", "current"));
        let original = messages.clone();

        let stats = apply_query_aware_memory_projection(&mut messages, 200_000);

        assert_eq!(stats.removed_messages, 0);
        assert_eq!(messages, original);
    }

    #[test]
    fn recovery_index_round_trips_exact_targets() {
        let targets = BTreeSet::from([
            RecoveryTarget::File("/tmp/a/one.md".to_string()),
            RecoveryTarget::File("/tmp/a/two.md".to_string()),
            RecoveryTarget::File("/tmp/b/three.txt".to_string()),
            RecoveryTarget::Directory("/tmp/archive-root".to_string()),
        ]);

        let index = build_recovery_index(&targets).expect("index");

        assert_eq!(recovery_targets_from_index(&index), targets);
    }

    #[test]
    fn recovery_targets_preserve_legal_path_punctuation_exactly() {
        let checkpoint_path = r#"/tmp/session]/checkpoints/evidence,`"'.md"#;
        let file_path = r#"/tmp/archive/evidence,`"'.txt"#;
        let trailing_space_path = "/tmp/archive/evidence ";
        let directory_path = "/tmp/archive-root,";
        let text = format!(
            "[context_checkpoint path={checkpoint_path}] summary\n\
             - archive_file_path: {file_path}\n\
             - file_path: {trailing_space_path}\n\
             归档目录: {directory_path}"
        );

        let targets = extract_recovery_targets(&text);

        assert!(targets.contains(&RecoveryTarget::File(checkpoint_path.to_string())));
        assert!(targets.contains(&RecoveryTarget::File(file_path.to_string())));
        assert!(targets.contains(&RecoveryTarget::File(trailing_space_path.to_string())));
        assert!(targets.contains(&RecoveryTarget::Directory(directory_path.to_string())));
        let index = build_recovery_index(&targets).expect("index");
        assert_eq!(recovery_targets_from_index(&index), targets);
    }

    #[test]
    fn active_query_tokens_give_current_user_priority_over_older_user() {
        let current = message("user", "ledger payments_router adapter migration");
        let older = message(
            "user",
            (0..900)
                .map(|i| format!("unrelated{i:03}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let messages = vec![older, current];
        let current_user_index = 1;

        let tokens = active_query_tokens(&messages, current_user_index);

        // The older user alone exceeds MAX_QUERY_TOKENS; without newest-first
        // ordering it would fill the cap and starve the live query tokens.
        assert!(tokens.contains("payments_router"));
        assert!(tokens.contains("ledger"));
        assert!(tokens.contains("migration"));
    }

    fn index_note(targets: &BTreeSet<RecoveryTarget>) -> Message {
        build_index_message(0, 0, targets).expect("index note")
    }

    fn sample_targets(prefix: &str) -> BTreeSet<RecoveryTarget> {
        (0..10)
            .map(|index| RecoveryTarget::File(format!("/tmp/{prefix}/{index}.md")))
            .chain([
                RecoveryTarget::File(format!("/tmp/{prefix}-standalone.md")),
                RecoveryTarget::Directory(format!("/tmp/{prefix}-dir")),
            ])
            .collect()
    }

    fn sample_note(prefix: &str) -> Message {
        index_note(&sample_targets(prefix))
    }

    fn checkpoint_targets(count: usize) -> BTreeSet<RecoveryTarget> {
        (0..count)
            .map(|index| {
                RecoveryTarget::File(format!(
                    "/tmp/session.assets/context-checkpoints/{index}.md"
                ))
            })
            .collect()
    }

    fn parsed_targets(message: &Message) -> BTreeSet<RecoveryTarget> {
        index_targets_from_note(value_to_string(&message.content).trim_start())
            .expect("index note payload must parse")
    }

    fn single_index_note(messages: &[Message]) -> &Message {
        let notes = messages
            .iter()
            .filter(|message| {
                value_to_string(&message.content)
                    .trim_start()
                    .starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX)
            })
            .collect::<Vec<_>>();
        assert_eq!(notes.len(), 1, "exactly one index note must remain");
        notes[0]
    }

    #[test]
    fn stale_index_notes_merge_with_fresh_omissions_into_one_note() {
        let mut messages = vec![
            message("system", "system"),
            sample_note("stale-a"),
            sample_note("stale-b"),
        ];
        for index in 0..12 {
            messages.push(checkpoint(index, "old topic", 4_000));
        }
        messages.push(message("user", "current"));

        let stats = apply_query_aware_memory_projection(&mut messages, 80_000);

        assert!(
            stats.removed_messages >= 4,
            "removed {}",
            stats.removed_messages
        );
        let merged = single_index_note(&messages);
        let mut expected = sample_targets("stale-a");
        expected.extend(sample_targets("stale-b"));
        // ALWAYS_KEEP_RECENT keeps the last two checkpoints; the first ten are
        // omitted and must stay reachable through the merged note.
        expected.extend(checkpoint_targets(10));
        assert_eq!(parsed_targets(merged), expected);
    }

    #[test]
    fn stale_index_notes_consolidate_without_fresh_candidates() {
        let mut messages = vec![message("system", "system")];
        for index in 0..8 {
            messages.push(sample_note(&format!("stale-{index}")));
        }
        messages.push(message("user", "current"));
        let before = total_chars(&messages);

        let stats = apply_query_aware_memory_projection(&mut messages, 120_000);

        assert_eq!(stats.removed_messages, 8);
        let merged = single_index_note(&messages);
        let mut expected = BTreeSet::new();
        for index in 0..8 {
            expected.extend(sample_targets(&format!("stale-{index}")));
        }
        assert_eq!(parsed_targets(merged), expected);
        assert!(total_chars(&messages) < before);
    }

    #[test]
    fn malformed_index_note_fails_open_and_survives_consolidation() {
        let malformed = message(
            ROLE_INTERNAL_NOTE,
            format!(
                "{}\nindex=this-is-not-json",
                crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX
            ),
        );
        let malformed_text = value_to_string(&malformed.content);
        let mut messages = vec![message("system", "system"), malformed.clone()];
        for index in 0..5 {
            messages.push(sample_note(&format!("valid-{index}")));
        }
        messages.push(message("user", "current"));

        let stats = apply_query_aware_memory_projection(&mut messages, 120_000);

        assert_eq!(stats.removed_messages, 5);
        let note_texts = messages
            .iter()
            .filter(|message| {
                value_to_string(&message.content)
                    .trim_start()
                    .starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX)
            })
            .map(|message| value_to_string(&message.content))
            .collect::<Vec<_>>();
        assert_eq!(note_texts.len(), 2);
        assert!(
            note_texts.contains(&malformed_text),
            "malformed note must be kept verbatim"
        );
        let merged = note_texts
            .into_iter()
            .find(|text| *text != malformed_text)
            .expect("merged note");
        let mut expected = BTreeSet::new();
        for index in 0..5 {
            expected.extend(sample_targets(&format!("valid-{index}")));
        }
        assert_eq!(
            index_targets_from_note(merged.trim_start()).expect("merged payload"),
            expected
        );
    }
}
