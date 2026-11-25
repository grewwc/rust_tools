//! Shared rendering and budgets for auxiliary history-summary input.

use serde_json::Value;

use crate::ai::history::{
    Message, is_runtime_synthetic_user_message, is_side_note_message, is_user_input_message,
    last_real_user_index,
};

/// Bound one auxiliary request independently of source-history size. Both the
/// planner preflight and request renderer use this cap so oversized history is
/// sampled once instead of being split into serial model calls.
pub(in crate::ai) const SUMMARY_TRANSCRIPT_MAX_CHARS: usize = 22_000;
pub(super) const SUMMARY_MESSAGE_MAX_CHARS: usize = 6_000;
pub(super) const SUMMARY_MESSAGE_SEPARATOR: &str = "\n\n[...chronological messages omitted...]\n\n";
pub(super) const LATEST_USER_EXCERPT_MAX_CHARS: usize = 240;
const ARCHIVED_IMAGE_OMISSION: &str =
    "[runtime-archived image payload omitted from summary input]\n";

/// Preserve text, not inferred goals or verified corrections. A short suffix
/// favors closing instructions; explicit character ranges expose omitted text.
/// Reference bodies and attachments are not the user's own text.
pub(super) fn latest_real_user_excerpt(messages: &[Message]) -> Option<(usize, Value)> {
    let index = latest_user_content_index(messages)?;
    // Spill pointers retain their role as turn boundaries, not as authored text.
    // Do not search backwards: the archived input may supersede older inputs.
    if is_archived_message_pointer(&messages[index]) {
        return None;
    }
    let (texts, non_text_omitted): (Vec<&str>, bool) = match &messages[index].content {
        Value::String(text) => (vec![text.as_str()], false),
        Value::Array(parts) => {
            let texts: Vec<_> = parts
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            let omitted = texts.len() != parts.len();
            (texts, omitted)
        }
        _ => (Vec::new(), true),
    };
    // Iterate the joined text without allocating an unbounded intermediate copy.
    let chars = || {
        texts
            .iter()
            .enumerate()
            .flat_map(|(index, text)| (index > 0).then_some('\n').into_iter().chain(text.chars()))
    };
    let end_char = chars().count();
    let start_char = end_char.saturating_sub(LATEST_USER_EXCERPT_MAX_CHARS);
    let text: String = chars().skip(start_char).collect();
    Some((
        index,
        serde_json::json!({
            "provenance": "user_authored_unverified",
            "semantics": "unknown_not_extracted",
            "coverage": "latest_real_user_text_only",
            "text_projection": "string_or_newline_joined_text_parts",
            "text_origin": "inline_user_content_v1",
            "text": text,
            "start_char": start_char,
            "end_char": end_char,
            "partial": start_char > 0 || non_text_omitted || end_char == 0,
        }),
    ))
}

// Normalized merged pointers are internal notes, but still represent missing
// user content. Later real input supersedes them; earlier input is not a fallback.
pub(super) fn latest_user_content_index(messages: &[Message]) -> Option<usize> {
    last_real_user_index(messages)
        .into_iter()
        .chain(messages.iter().rposition(is_archived_message_pointer))
        .max()
}

fn is_archived_message_pointer(message: &Message) -> bool {
    message.content.as_str().is_some_and(|text| {
        super::tool_overflow::is_preserved_user_or_image_stub(text.trim_start())
    })
}

fn summary_role_label(message: &Message) -> String {
    if is_archived_message_pointer(message) {
        "ARCHIVE POINTER (runtime recovery metadata; original content not inline; not a user request or correction)".to_string()
    } else if is_runtime_synthetic_user_message(message) {
        "SYNTHETIC USER (runtime context, not a user correction)".to_string()
    } else if is_user_input_message(message) {
        "USER (real input; factual assertions unverified)".to_string()
    } else {
        message.role.to_uppercase()
    }
}

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

fn is_reply_reference(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("reference")
        && part.get("kind").and_then(Value::as_str) == Some("message")
}

fn has_reply_references(message: &Message) -> bool {
    message
        .content
        .as_array()
        .is_some_and(|parts| parts.iter().any(is_reply_reference))
}

/// Reserve the question and every delimiter before allocating bounded previews.
/// Refuse the fragment if even the complete structural content cannot fit.
fn render_reply_summary(message: &Message, parts: &[Value], max_chars: usize) -> String {
    const MAX_PREVIEW_CHARS: usize = 600;
    let question = parts
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    let references: Vec<_> = parts
        .iter()
        .filter(|part| is_reply_reference(part))
        .collect();
    let mut out = format!(
        "### {}\n\nUser question:\n{question}\n",
        summary_role_label(message)
    );
    if parts
        .iter()
        .any(super::tool_overflow::is_preserved_image_payload_pointer_part)
    {
        out.push_str(ARCHIVED_IMAGE_OMISSION);
    }
    let other_parts: Vec<_> = parts
        .iter()
        .filter(|part| {
            !is_reply_reference(part)
                && !super::tool_overflow::is_preserved_image_payload_pointer_part(part)
                && part.get("type").and_then(Value::as_str) != Some("text")
        })
        .collect();
    if !other_parts.is_empty() {
        out.push_str("[Other attachments omitted from summary preview]\n");
    }
    let skeleton_chars: usize = references
        .iter()
        .map(|part| {
            crate::ai::history::reply_reference::summary_preview(part, 0)
                .chars()
                .count()
        })
        .sum();
    let fixed = out.chars().count() + skeleton_chars;
    if fixed > max_chars {
        return String::new();
    }
    let body_budget = ((max_chars - fixed) / references.len().max(1)).min(MAX_PREVIEW_CHARS);
    for part in references {
        out.push_str(&crate::ai::history::reply_reference::summary_preview(
            part,
            body_budget,
        ));
    }
    out
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
    if let Value::Array(parts) = &message.content
        && parts.iter().any(is_reply_reference)
    {
        return render_reply_summary(message, parts, max_chars);
    }
    let mut out = String::with_capacity(max_chars.min(6_000));
    push_summary_text(
        &mut out,
        &format!("### {}\n\n", summary_role_label(message)),
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
                if super::tool_overflow::is_preserved_image_payload_pointer_part(part) {
                    if !push_summary_text(&mut content, ARCHIVED_IMAGE_OMISSION, content_budget) {
                        break;
                    }
                    continue;
                }
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
    let latest_user = last_real_user_index(messages);
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
                    if has_reply_references(message) || latest_user == Some(index) {
                        max_chars.min(SUMMARY_MESSAGE_MAX_CHARS)
                    } else {
                        SUMMARY_MESSAGE_MAX_CHARS
                    }
                },
            );
            let lower = raw.to_ascii_lowercase();
            let is_recent = last_index.saturating_sub(index) < 8;
            let is_priority_role = is_user_input_message(message)
                || matches!(message.role.as_str(), "system" | "internal_note");
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
                if side_note || has_reply_references(message) {
                    raw
                } else {
                    truncate_summary_fragment(&raw, per_message_cap)
                },
                side_note,
            )
        })
        .collect();

    if candidates.iter().any(|(index, _, fragment, _)| {
        has_reply_references(&messages[*index]) && fragment.is_empty()
    }) {
        return String::new();
    }
    candidates.sort_by(|a, b| {
        b.3.cmp(&a.3)
            .then_with(|| (latest_user == Some(b.0)).cmp(&(latest_user == Some(a.0))))
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
        if (*side_note || latest_user == Some(*index))
            && (fragment.is_empty() || used + extra > max_chars)
        {
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

#[cfg(test)]
mod reply_reference_tests {
    use super::*;
    use serde_json::json;

    fn referenced_question(count: i64) -> Message {
        let mut parts: Vec<_> = (0..count)
            .map(|index| {
                json!({
                    "type":"reference", "kind":"message", "source_message_id":42 + index,
                    "text":"长引用正文".repeat(1000),
                })
            })
            .collect();
        parts.push(json!({"type":"text", "text":"Explain the assumptions in every quoted reply."}));
        Message {
            role: "user".into(),
            content: Value::Array(parts),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn reply_reference_summary_long_body_preserves_question_and_end_boundary() {
        let message = referenced_question(1);
        let original = message.content.clone();
        let summary = render_summary_message(&message, 1000);
        assert!(summary.contains("Explain the assumptions in every quoted reply."));
        assert!(summary.contains("[Referenced assistant reply: m42]"));
        assert!(summary.contains("[Referenced assistant reply truncated for summary preview]"));
        assert!(summary.ends_with("[End referenced assistant reply]\n"));
        assert!(summary.chars().count() <= 1000);
        assert_eq!(message.content, original);
    }

    #[test]
    fn reply_reference_summary_multiple_long_bodies_keep_all_boundaries_in_order() {
        let message = referenced_question(4);
        let original = message.content.clone();
        for summary in [
            render_summary_message(&message, 1000),
            build_summary_history_input(&[message.clone()], 1000),
        ] {
            assert!(summary.contains("Explain the assumptions in every quoted reply."));
            let positions: Vec<_> = (42..46)
                .map(|id| {
                    summary
                        .find(&format!("[Referenced assistant reply: m{id}]"))
                        .expect("all sources survive")
                })
                .collect();
            assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(
                summary.matches("[End referenced assistant reply]").count(),
                4
            );
            assert_eq!(
                summary
                    .matches("[Referenced assistant reply truncated for summary preview]")
                    .count(),
                4
            );
            assert!(summary.chars().count() <= 1000);
        }
        assert_eq!(message.content, original);
        assert!(
            render_summary_message(&message, 20).is_empty(),
            "refuse a budget that cannot fit complete boundaries"
        );
        assert!(build_summary_history_input(&[message], 20).is_empty());
    }
}
