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

/// Script mix of billable text, gathered in one walk and projected onto either
/// the char budget or the prompt-token estimate, so the two measures always
/// bill exactly the same fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::ai) struct BillableCounts {
    pub(in crate::ai) ascii: usize,
    pub(in crate::ai) other: usize,
    pub(in crate::ai) images: usize,
}

/// Thirds of a token per image: the same 512 tokens the flat
/// 2-chars-per-token estimate charged for [`IMAGE_BUDGET_CHARS`].
const IMAGE_TOKEN_THIRDS: usize = IMAGE_BUDGET_CHARS / 2 * 3;
const THIRDS_PER_TOKEN: usize = 3;

impl BillableCounts {
    pub(in crate::ai) const IMAGE: Self = Self {
        ascii: 0,
        other: 0,
        images: 1,
    };

    pub(in crate::ai) const fn ascii(count: usize) -> Self {
        Self {
            ascii: count,
            other: 0,
            images: 0,
        }
    }

    /// Counts UTF-8 lead bytes instead of decoding: continuation bytes
    /// (0x80..=0xBF) contribute nothing, so `chars()` equals `chars().count()`.
    pub(in crate::ai) fn text(text: &str) -> Self {
        let mut counts = Self::default();
        for byte in text.bytes() {
            match byte {
                0x00..=0x7F => counts.ascii += 1,
                0x80..=0xBF => {}
                _ => counts.other += 1,
            }
        }
        counts
    }

    /// serde_json's compact string encoding of `text`: 2 quotes, two-char
    /// escapes for `"` `\` `\b` `\t` `\n` `\f` `\r`, `\u00xx` (6 chars) for
    /// other controls, every other scalar verbatim.
    pub(in crate::ai) fn json_string(text: &str) -> Self {
        let mut counts = Self::ascii(2);
        for byte in text.bytes() {
            match byte {
                b'"' | b'\\' | 0x08 | b'\t' | b'\n' | 0x0C | b'\r' => counts.ascii += 2,
                0x00..=0x1F => counts.ascii += 6,
                0x20..=0x7F => counts.ascii += 1,
                0x80..=0xBF => {}
                _ => counts.other += 1,
            }
        }
        counts
    }

    /// Char budget: Unicode scalars, images at their nominal cost.
    pub(in crate::ai) fn chars(self) -> usize {
        self.ascii + self.other + self.images * IMAGE_BUDGET_CHARS
    }

    /// Prompt-token estimate: ASCII prose/code/JSON at ~3 chars per token, CJK
    /// and other non-ASCII scalars at 1 token each. The previous flat
    /// 2-chars-per-token ratio overcharged ASCII by ~1.5x (starting compression
    /// early) while undercharging CJK by ~2x (provider overflow); usage
    /// feedback calibrates the verified prefix of each request.
    pub(in crate::ai) fn tokens(self) -> usize {
        (self.ascii + self.other * 3 + self.images * IMAGE_TOKEN_THIRDS).div_ceil(THIRDS_PER_TOKEN)
    }
}

impl std::ops::Add for BillableCounts {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            ascii: self.ascii + rhs.ascii,
            other: self.other + rhs.other,
            images: self.images + rhs.images,
        }
    }
}

impl std::ops::AddAssign for BillableCounts {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::iter::Sum for BillableCounts {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), std::ops::Add::add)
    }
}

/// Budget counts of a single part in a multimodal content array: images are
/// billed at nominal cost, text by its script mix.
pub(in crate::ai) fn content_part_budget_counts(item: &Value) -> BillableCounts {
    let is_image = item.get("type").and_then(|t| t.as_str()) == Some("image_url")
        || item.get("image_url").is_some();
    let is_image_reference = item.get("type").and_then(|t| t.as_str()) == Some("reference")
        && item.get("kind").and_then(|kind| kind.as_str()) == Some("image");
    if is_image || is_image_reference {
        return BillableCounts::IMAGE;
    }
    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
        return BillableCounts::text(text);
    }
    BillableCounts::text(&item.to_string())
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
