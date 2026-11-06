//! Shared rendering and budgets for auxiliary history-summary input.

use serde_json::Value;

use crate::ai::history::{Message, is_side_note_message};

/// Preserve the existing auxiliary transcript cap; incremental segments use
/// this same limit and renderer so priority sampling cannot skip a segment row.
pub(in crate::ai) const SUMMARY_TRANSCRIPT_MAX_CHARS: usize = 22_000;
pub(super) const SUMMARY_MESSAGE_MAX_CHARS: usize = 6_000;
pub(super) const SUMMARY_MESSAGE_SEPARATOR: &str = "\n\n[...chronological messages omitted...]\n\n";

fn truncate_summary_fragment(fragment: &str, max_chars: usize) -> String {
    let total = fragment.chars().count();
    if total <= max_chars {
        return fragment.to_string();
    }
    if max_chars < 80 {
        return fragment.chars().take(max_chars).collect();
    }

    let marker = "\n...[message excerpt truncated]...\n";
    let marker_chars = marker.chars().count();
    let payload = max_chars.saturating_sub(marker_chars);
    let head = payload * 2 / 3;
    let tail = payload.saturating_sub(head);
    let prefix: String = fragment.chars().take(head).collect();
    let suffix: String = fragment
        .chars()
        .rev()
        .take(tail)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}{marker}{suffix}")
}

fn push_summary_text(out: &mut String, text: &str, max_chars: usize) -> bool {
    let used = out.chars().count();
    let remaining = max_chars.saturating_sub(used);
    let text_chars = text.chars().count();
    out.extend(text.chars().take(remaining));
    text_chars <= remaining
}

pub(in crate::ai) fn render_summary_message(message: &Message, max_chars: usize) -> String {
    if is_side_note_message(message) {
        let rendered = format!(
            "### SIDE-NOTE\n\n{}",
            crate::ai::history::value_to_string(&message.content)
        );
        return if rendered.chars().count() <= max_chars {
            rendered
        } else {
            String::new()
        };
    }
    let mut out = String::with_capacity(max_chars.min(6_000));
    push_summary_text(
        &mut out,
        &format!("### {}\n\n", message.role.to_uppercase()),
        max_chars,
    );
    let content_budget = max_chars.saturating_mul(2) / 5;
    let mut content = String::with_capacity(content_budget);
    match &message.content {
        Value::String(text) => {
            push_summary_text(
                &mut content,
                &truncate_summary_fragment(text, content_budget),
                content_budget,
            );
        }
        Value::Array(parts) => {
            for part in parts {
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("content").and_then(Value::as_str));
                if let Some(text) = text {
                    if !push_summary_text(&mut content, text, content_budget)
                        || !push_summary_text(&mut content, "\n", content_budget)
                    {
                        break;
                    }
                } else if !push_summary_text(
                    &mut content,
                    "[non-text content omitted from summary input]\n",
                    content_budget,
                ) {
                    break;
                }
            }
        }
        other => {
            push_summary_text(
                &mut content,
                &truncate_summary_fragment(&other.to_string(), content_budget),
                content_budget,
            );
        }
    }
    push_summary_text(&mut out, &content, max_chars);

    if let Some(tool_calls) = &message.tool_calls {
        // Render every call identity before allocating argument budget, so huge
        // arguments cannot hide later call IDs or names from the summarizer.
        for call in tool_calls {
            let metadata = format!(
                "\nTool call: id={} type={} name={}\n",
                truncate_summary_fragment(&call.id, 256),
                truncate_summary_fragment(&call.tool_type, 128),
                truncate_summary_fragment(&call.function.name, 256),
            );
            if !push_summary_text(&mut out, &metadata, max_chars) {
                return out;
            }
        }
        let argument_budget = (max_chars / 5).max(128) / tool_calls.len().max(1);
        for call in tool_calls {
            let arguments = format!(
                "Tool arguments for {}: {}\n",
                truncate_summary_fragment(&call.id, 128),
                truncate_summary_fragment(&call.function.arguments, argument_budget),
            );
            if !push_summary_text(&mut out, &arguments, max_chars) {
                return out;
            }
        }
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        push_summary_text(
            &mut out,
            &format!(
                "\nTool result for call id: {}\n",
                truncate_summary_fragment(tool_call_id, 256)
            ),
            max_chars,
        );
    }
    if let Some(reasoning) = &message.reasoning_content
        && !crate::ai::history::has_runtime_message_metadata(message)
    {
        let remaining = max_chars.saturating_sub(out.chars().count());
        push_summary_text(
            &mut out,
            &format!(
                "\nReasoning: {}",
                truncate_summary_fragment(reasoning, remaining.saturating_sub(12))
            ),
            max_chars,
        );
    }
    out
}

/// Sample ordinary dialogue within the budget after reserving complete live
/// user guidance. Refuse partial guidance rather than silently omitting it.
pub(in crate::ai) fn build_summary_history_input(messages: &[Message], max_chars: usize) -> String {
    if messages.is_empty() || max_chars == 0 {
        return String::new();
    }

    let last_index = messages.len().saturating_sub(1);
    let mut candidates: Vec<(usize, usize, String, bool)> = messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let side_note = is_side_note_message(message);
            let raw = render_summary_message(
                message,
                if side_note {
                    max_chars
                } else {
                    SUMMARY_MESSAGE_MAX_CHARS
                },
            );
            let lower = raw.to_ascii_lowercase();
            let is_recent = last_index.saturating_sub(index) < 8;
            let is_priority_role =
                matches!(message.role.as_str(), "system" | "user" | "internal_note");
            let has_keypoint = [
                "error",
                "failed",
                "failure",
                "panic",
                "fix",
                "fixed",
                "decision",
                "root cause",
                "warning",
                "错误",
                "失败",
                "修复",
                "决定",
                "根因",
                "警告",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            let score = usize::from(is_recent) * 1_000
                + usize::from(is_priority_role) * 700
                + usize::from(has_keypoint) * 500
                + usize::from(index < 2) * 300
                + index.min(200);
            let per_message_cap = if is_priority_role {
                SUMMARY_MESSAGE_MAX_CHARS
            } else {
                SUMMARY_MESSAGE_MAX_CHARS / 2
            };
            (
                index,
                score,
                if side_note {
                    raw
                } else {
                    truncate_summary_fragment(&raw, per_message_cap)
                },
                side_note,
            )
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.3.cmp(&a.3)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.0.cmp(&a.0))
    });
    let separator = SUMMARY_MESSAGE_SEPARATOR;
    let separator_chars = separator.chars().count();
    let mut selected = vec![false; messages.len()];
    let mut used = 0usize;
    for (index, _, fragment, side_note) in &candidates {
        let fragment_chars = fragment.chars().count();
        let extra = fragment_chars + usize::from(used > 0) * separator_chars;
        if *side_note && (fragment.is_empty() || used + extra > max_chars) {
            return String::new();
        }
        if used + extra <= max_chars {
            selected[*index] = true;
            used += extra;
        }
    }

    let mut ordered: Vec<(usize, &String)> = candidates
        .iter()
        .filter(|(index, _, _, _)| selected[*index])
        .map(|(index, _, fragment, _)| (*index, fragment))
        .collect();
    ordered.sort_by_key(|(index, _)| *index);

    let mut out = String::with_capacity(used.min(max_chars));
    let mut previous_index = None;
    for (index, fragment) in ordered {
        if let Some(previous) = previous_index
            && index != previous + 1
        {
            out.push_str(separator);
        } else if previous_index.is_some() {
            out.push('\n');
        }
        out.push_str(fragment);
        previous_index = Some(index);
    }
    out
}
