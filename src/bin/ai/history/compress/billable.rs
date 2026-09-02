//! Billable-size helpers and shared policy constants for images,
//! summary bodies, and tool-message windows.

use super::*;

/// Nominal billing cost of a single image in the "char budget".
///
/// A vision model tokenizes one image into a few hundred to one-or-two thousand
/// tokens, fully decoupled from its base64 text length (easily hundreds of
/// thousands of chars). Historically `value_len_chars` billed directly by base64
/// text length, so **one large image ate the entire context budget**:
/// `messages_total_chars` ballooned far past max_chars / soft_threshold, and the
/// compaction pipeline evicted the agent's own tool results (its working memory)
/// every round — within one turn this showed up as "amnesia + repeatedly
/// restating earlier exploration/plans". Give images a fixed nominal cost so the
/// budget returns to being text-dominated.
/// Note: this only changes budget **accounting**, not the message content itself
/// (images are still sent verbatim with zero compression).
pub(in crate::ai) const IMAGE_BUDGET_CHARS: usize = 1_024;

/// Whether a bare string is an inline image data URL (a few providers put images
/// into plain strings).
pub(in crate::ai) fn is_inline_image_data_url(s: &str) -> bool {
    let t = s.trim_start();
    t.starts_with("data:image/") && t.contains(";base64,")
}

/// Budget char count of a single part in a multimodal content array: images are
/// billed at nominal cost, text at its actual char count.
pub(in crate::ai) fn content_part_budget_chars(item: &Value) -> usize {
    let is_image = item.get("type").and_then(|t| t.as_str()) == Some("image_url")
        || item.get("image_url").is_some();
    let is_image_reference = item.get("type").and_then(|t| t.as_str()) == Some("reference")
        && item.get("kind").and_then(|kind| kind.as_str()) == Some("image");
    if is_image || is_image_reference {
        return IMAGE_BUDGET_CHARS;
    }
    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
        return text.chars().count();
    }
    item.to_string().chars().count()
}

pub(in crate::ai) fn automatic_summary_body(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    for prefix in [
        "历史摘要（自动压缩，以下为更早对话的简短语义）：",
        "对话摘要（自动压缩，以下为早期对话要点）：",
        "长期记忆摘要（压缩保留）:",
        "长期记忆摘要（压缩保留）：",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(rest.trim());
        }
    }

    if let Some(rest) = trimmed.strip_prefix("[mid-turn-summary]") {
        let rest = rest
            .trim_start()
            .strip_prefix("早期工具调用与对话已被 LLM 摘要：")
            .unwrap_or_else(|| rest.trim_start());
        return Some(rest.trim());
    }

    None
}

pub(in crate::ai) fn strip_nested_prior_summary_prefixes(text: &str) -> String {
    let mut current = normalize_whitespace(text);
    for _ in 0..8 {
        let trimmed = current.trim_start();
        let rest = trimmed
            .strip_prefix("- 更早摘要:")
            .or_else(|| trimmed.strip_prefix("更早摘要:"))
            .or_else(|| trimmed.strip_prefix("- 更早摘要："))
            .or_else(|| trimmed.strip_prefix("更早摘要："));
        let Some(rest) = rest else {
            break;
        };
        current = normalize_whitespace(rest);
    }
    current
}

/// **Minimum** protection window for progressive group folding: narrowing stops
/// here and never reaches 0.
///
/// A window of 0 folds the most recent tool interaction itself into a
/// `compressed_tool_round` stub, leaving the model without any recent structured
/// tool context (`assistant.tool_calls` + `role=tool` results): multi-step task
/// continuity suffers and runtime guards lose their freshest evidence. Keep the
/// most recent group verbatim; remaining excess is handled downstream by
/// per-message truncation / first_trim fallbacks.
pub(in crate::ai) const MIN_KEEP_RECENT_TOOL_GROUPS: usize = 1;

/// Floor on the protected verbatim tail once group folding converges, measured in
/// billable chars across the messages retained for the current window. Group-count
/// protection alone proved insufficient on read-heavy sessions: many large results
/// still squeezed the window down until nearly everything except the last turn was
/// a pointer stub, after which the model re-read files whose full text it had just
/// received. This floor keeps roughly 6K tokens of fresh tool evidence resident
/// whenever budget allows. It intentionally yields (folding proceeds past it) only
/// at MIN_KEEP_RECENT_TOOL_GROUPS so overflow handling always terminates.
pub(in crate::ai) const MIN_PROTECTED_TAIL_CHARS: usize = 30_000;

/// For assistant messages carrying tool_calls, how many recent turns keep full
/// reasoning_content. Older tool-call reasoning is set to None (DeepSeek fills an
/// empty-string placeholder via echo as a backstop), preventing historical
/// reasoning text from accumulating monotonically over long sessions, slowing
/// responses and squeezing the context budget.
pub(in crate::ai) const KEEP_RECENT_TOOL_CALL_REASONING: usize = 3;

pub(in crate::ai) fn tool_message_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "tool").then_some(i))
        .collect()
}

pub(in crate::ai) fn redact_images_except_last(messages: &mut [Message], keep_last: usize) {
    let _ = (messages, keep_last);
    // Images are required to stay zero-compression: the history compaction stage
    // no longer replaces old images with [[image omitted]].
}
