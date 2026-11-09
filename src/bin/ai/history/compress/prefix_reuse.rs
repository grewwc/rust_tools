//! A single process-local projection, admitted only with a no-op certificate.
//! Exact message equality and unchanged retention boundaries replace any need
//! to trust a snapshot watermark. Unproved inputs use the original pipeline.

use std::sync::Mutex;

use crate::ai::history::tool_guidance::ToolGuidanceFilter;

use super::*;

pub(super) const CACHE_MAX_MESSAGES: usize = 16_384;
pub(super) const CACHE_MAX_BYTES: usize = 8 * 1024 * 1024;

pub(super) static CACHE: Mutex<PrefixReuseCache> = Mutex::new(PrefixReuseCache { entry: None });

#[derive(PartialEq, Eq)]
pub(super) struct PrefixReuseKey {
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
}

impl PrefixReuseKey {
    pub(super) fn new(
        max_chars: usize,
        keep_last: usize,
        summary_max_chars: usize,
        overflow_dir: Option<&Path>,
        cwd: Option<&Path>,
    ) -> Self {
        Self {
            max_chars,
            keep_last,
            summary_max_chars,
            overflow_dir: overflow_dir.map(Path::to_path_buf),
            cwd: cwd.map(Path::to_path_buf),
        }
    }
}

#[derive(Default)]
pub(super) struct PrefixReuseCache {
    pub(super) entry: Option<PrefixReuseEntry>,
}

pub(super) struct PrefixReuseEntry {
    pub(super) key: PrefixReuseKey,
    pub(super) messages: Vec<Message>,
    pub(super) chars: usize,
    pub(super) bytes: usize,
    pub(super) user_turns: usize,
    pub(super) boundaries: (usize, usize),
}

impl PrefixReuseCache {
    pub(super) fn from_outcome(key: PrefixReuseKey, outcome: &ContextCompressionOutcome) -> Self {
        Self {
            entry: PrefixReuseEntry::certify(key, outcome),
        }
    }

    pub(super) fn try_reuse(
        &mut self,
        messages: &[Message],
        key: &PrefixReuseKey,
    ) -> Option<usize> {
        let entry = self.entry.as_mut()?;
        if entry.key != *key
            || messages.len() < entry.messages.len()
            || messages.len() > CACHE_MAX_MESSAGES
            || messages[..entry.messages.len()] != entry.messages
        {
            return None;
        }
        let tail = &messages[entry.messages.len()..];
        if !tail.iter().all(is_plain_tail_message) {
            return None;
        }
        let chars = entry.chars.checked_add(messages_total_chars(tail))?;
        let bytes = entry.bytes.checked_add(projection_bytes(tail)?)?;
        let user_turns = entry.user_turns.checked_add(user_input_turn_count(tail))?;
        if chars > key.max_chars
            || bytes > CACHE_MAX_BYTES
            || (key.keep_last > 0 && user_turns > key.keep_last.min(messages.len()))
            || retention_boundaries(messages, chars) != entry.boundaries
        {
            return None;
        }

        // Evaluate the existing filter over the FULL merged input. A real user
        // in the tail can expire guidance in an otherwise identical prefix.
        let mut guidance = ToolGuidanceFilter::new(messages);
        let mut guidance_unchanged = true;
        for message in messages {
            guidance_unchanged &= guidance.keep(message);
        }
        if !guidance_unchanged {
            return None;
        }

        // No tail tool protocol means all tool indexes, call identities and
        // complete-group protection remain unchanged. No tail notes/images or
        // stubs means the certified windows and pure dedup remain unchanged too.
        // The proven identity output is itself the next certified projection;
        // extend only the new tail rather than recertifying the whole prefix.
        entry.messages.extend_from_slice(tail);
        entry.chars = chars;
        entry.bytes = bytes;
        entry.user_turns = user_turns;
        Some(chars)
    }
}

impl PrefixReuseEntry {
    fn certify(key: PrefixReuseKey, outcome: &ContextCompressionOutcome) -> Option<Self> {
        let messages = &outcome.messages;
        if outcome.status != ContextCompressionStatus::Complete
            || key.max_chars == 0
            || messages.is_empty()
            || messages.len() > CACHE_MAX_MESSAGES
            || outcome.after_chars > key.max_chars
            // Restrict the proof to text: structured/image content may change
            // serialization-sensitive dedup or preserved-content spill decisions.
            || messages.iter().any(|message| {
                message
                    .content
                    .as_str()
                    .is_none_or(is_inline_image_data_url)
            })
        {
            return None;
        }
        let bytes = projection_bytes(messages)?
            .checked_add(
                key.overflow_dir
                    .as_ref()
                    .map_or(0, |path| path.as_os_str().len()),
            )?
            .checked_add(key.cwd.as_ref().map_or(0, |path| path.as_os_str().len()))?;
        let user_turns = user_input_turn_count(messages);
        if bytes > CACHE_MAX_BYTES
            || (key.keep_last > 0 && user_turns > key.keep_last.min(messages.len()))
            || compressed_tool_evidence_exceeds_inline_budget(messages)
            || messages
                .iter()
                .filter(|message| is_incremental_summary(message))
                .map(message_billable_chars)
                .sum::<usize>()
                > MAX_INCREMENTAL_SUMMARY_INLINE_CHARS
            || messages
                .iter()
                .filter(|message| is_self_note_message(message))
                .count()
                > MAX_SELF_NOTES_IN_MESSAGES
        {
            return None;
        }

        // These are sufficient no-write conditions for the shrinker's tool
        // preparation, precision budget and successful-write argument passes.
        // Every result is protected; every call with a result therefore also
        // has protected arguments. Sum all result text conservatively, rather
        // than relying on the registry's per-group precision classifications.
        let protected = recent_tool_group_message_indices(messages, KEEP_RECENT_TOOL_GROUPS);
        let mut tool_chars = 0usize;
        for (index, message) in messages.iter().enumerate() {
            if message.role == "tool" {
                if !protected.contains(&index) {
                    return None;
                }
                tool_chars = tool_chars.checked_add(message_billable_chars(message))?;
            }
        }
        if tool_chars > key.max_chars / 2 {
            return None;
        }

        let boundaries = retention_boundaries(messages, outcome.after_chars);
        if messages[..boundaries.0].iter().any(|message| {
            message.role == "user"
                && !is_side_note_message(message)
                && !is_preserved_user_or_image_stub(message.content.as_str().unwrap())
                && message.content.as_str().unwrap().chars().count()
                    >= USER_OVERFLOW_SPILL_MIN_CHARS
        }) {
            return None;
        }

        // Probe only pure passes, in their existing order, checking each result
        // separately. Full output is not assumed to be a fixed point: a newly
        // appended increment can still exceed its independent window next time.
        let mut probe = messages.clone();
        normalize_preserved_message_stubs_for_model(&mut probe);
        if probe != *messages {
            return None;
        }
        probe = coalesce_accumulated_summary_notes(probe);
        if probe != *messages {
            return None;
        }
        dedup_adjacent(&mut probe);
        if probe != *messages {
            return None;
        }
        dedup_repeated_tool_results(&mut probe, &Default::default());
        if probe != *messages {
            return None;
        }
        age_out_overflow_stub_previews(&mut probe, trimming_turns(outcome.after_chars));
        if probe != *messages {
            return None;
        }
        merge_old_user_overflow_stubs(
            &mut probe,
            trimming_turns(outcome.after_chars),
            key.overflow_dir.as_deref(),
        );
        if probe != *messages {
            return None;
        }

        Some(Self {
            key,
            messages: probe,
            chars: outcome.after_chars,
            bytes,
            user_turns,
            boundaries,
        })
    }
}

pub(super) fn is_plain_tail_message(message: &Message) -> bool {
    if !matches!(message.role.as_str(), "user" | "assistant")
        || message.tool_calls.is_some()
        || message.tool_call_id.is_some()
        || message.reasoning_content.is_some()
    {
        return false;
    }
    let Some(text) = message.content.as_str() else {
        return false;
    };
    !is_inline_image_data_url(text)
        && !is_summary_note_text(text)
        && !is_archive_note_text(text)
        && !is_preserved_user_or_image_stub(text)
        && !is_preserved_tool_overflow_content(&message.content)
        && !text.trim_start().starts_with("self_note:")
        && !text.contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
}

fn trimming_turns(chars: usize) -> usize {
    if chars <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX
    } else {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN
    }
}

pub(super) fn retention_boundaries(messages: &[Message], chars: usize) -> (usize, usize) {
    // Both certified projections fit the entire budget, so the existing
    // byte-cap escape valve cannot reduce the base protection window.
    let keep = trimming_turns(chars);
    (
        retained_turn_start(messages, keep),
        retained_turn_start(messages, keep.max(KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX)),
    )
}

pub(super) fn projection_bytes(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .try_fold(std::mem::size_of_val(messages), |bytes, message| {
            let bytes = bytes
                .checked_add(message.role.len())?
                .checked_add(message.content.as_str()?.len())?
                .checked_add(message.tool_call_id.as_ref().map_or(0, String::len))?
                .checked_add(message.reasoning_content.as_ref().map_or(0, String::len))?;
            let Some(calls) = &message.tool_calls else {
                return Some(bytes);
            };
            calls.iter().try_fold(
                bytes.checked_add(std::mem::size_of_val(calls.as_slice()))?,
                |bytes, call| {
                    bytes
                        .checked_add(call.id.len())?
                        .checked_add(call.tool_type.len())?
                        .checked_add(call.function.name.len())?
                        .checked_add(call.function.arguments.len())
                },
            )
        })
}
