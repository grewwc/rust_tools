//! Exact final-response terminator for answers following a successful `complete` call.
//!
//! The marker proves that the generated body reached the boundary the model was
//! instructed to emit. It is runtime control syntax, so accepted answers lose it
//! before canonical history or terminal presentation sees the text.

use super::*;

pub(in crate::ai::driver::turn_runtime) const FINAL_TERMINATOR_RETRY_MARKER: &str =
    "[final-terminator-retry]";
pub(in crate::ai::driver::turn_runtime) const FINAL_TERMINATOR_RETRY_NOTE: &str = "[final-terminator-retry]\nThe previous final-response draft did not include the required end marker and was not stored or shown. Rewrite the complete final response from the beginning; do not continue or refer to the rejected draft. End with `[END]` on its own final line. Do not call `complete` again unless you perform another tool action first.";
pub(in crate::ai::driver::turn_runtime) const FINAL_TERMINATOR_WARNING: &str = "[Runtime warning] The model did not emit the required final-response terminator after one retry, so this answer may be incomplete.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum FinalTerminatorGateAction {
    Allow,
    Reopen,
    Warn,
}

/// Return the user-visible body only when the exact marker is the final
/// non-empty line and sits outside a Markdown fence.
fn body_before_final_terminator(text: &str) -> Option<String> {
    let trimmed = text.trim_end();
    let marker = crate::ai::tools::FINAL_RESPONSE_END_MARKER;
    let body_with_separator = trimmed.strip_suffix(marker)?;
    if !body_with_separator.ends_with('\n') || body_with_separator.trim().is_empty() {
        return None;
    }
    if markdown_fence_is_open(body_with_separator) {
        return None;
    }
    Some(body_with_separator.trim_end().to_string())
}

fn markdown_fence_is_open(text: &str) -> bool {
    let mut open: Option<(char, usize)> = None;
    for line in text.lines() {
        let line = line.trim_start();
        let Some(ch) = line.chars().next().filter(|ch| matches!(ch, '`' | '~')) else {
            continue;
        };
        let count = line
            .chars()
            .take_while(|candidate| *candidate == ch)
            .count();
        if count < 3 {
            continue;
        }
        match open {
            None => open = Some((ch, count)),
            Some((open_ch, open_count))
                if ch == open_ch && count >= open_count && line[count..].trim().is_empty() =>
            {
                open = None;
            }
            Some(_) => {}
        }
    }
    open.is_some()
}

pub(in crate::ai::driver::turn_runtime) fn final_terminator_gate_action(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    assistant_text: &mut String,
    can_reopen: bool,
) -> FinalTerminatorGateAction {
    // A successful receipt scopes the control syntax. Plain no-tool answers may
    // legitimately end in the same literal text and must remain byte-stable.
    if has_successful_complete_receipt(turn_messages)
        && let Some(body) = body_before_final_terminator(assistant_text)
    {
        *assistant_text = body;
        return FinalTerminatorGateAction::Allow;
    }
    // Stale/missing claims are owned by the outstanding-work gate. Reopening
    // here is reserved for a fresh assertion so this gate never steals that
    // gate's single-use completion state.
    if !has_fresh_completion_assertion(app, turn_messages) {
        return FinalTerminatorGateAction::Allow;
    }
    if !can_reopen || current_turn_has_internal_marker(messages, FINAL_TERMINATOR_RETRY_MARKER) {
        return FinalTerminatorGateAction::Warn;
    }
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(FINAL_TERMINATOR_RETRY_NOTE.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    FinalTerminatorGateAction::Reopen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_an_outside_final_line() {
        assert_eq!(
            body_before_final_terminator("Answer.\n[END]\n\n"),
            Some("Answer.".to_string())
        );
        assert_eq!(body_before_final_terminator("Answer [END]"), None);
        assert_eq!(body_before_final_terminator("[END]"), None);
        assert_eq!(body_before_final_terminator("```text\nAnswer\n[END]"), None);
    }
}
