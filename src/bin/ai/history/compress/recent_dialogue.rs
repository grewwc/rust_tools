//! Small verbatim dialogue window independent of the tool traffic in each turn.

use super::*;
use crate::ai::history::is_user_input_message;

const MAX_USER_TURNS: usize = 8;
const MAX_PRIOR_USER_CHARS: usize = 8_000;
const MAX_ANSWER_CHARS: usize = 4_000;

/// Apart from the current user input, reserve at most a quarter of the target
/// for dialogue. Tool groups never count toward this reservation.
#[derive(Default)]
pub(super) struct RecentDialogue(Vec<usize>);

impl RecentDialogue {
    pub(super) fn new(messages: &[Message], budget: usize) -> Self {
        let mut turns = messages
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, message)| is_user_input_message(message))
            .take(MAX_USER_TURNS);
        let Some((latest, _)) = turns.next() else {
            return Self::default();
        };
        let mut retained = vec![latest];
        let mut user_budget = MAX_PRIOR_USER_CHARS.min(budget / 8);
        let answer_budget = MAX_ANSWER_CHARS.min(budget / 16);
        let prior = turns.next();
        // A saved projection can end with a completed answer, before the next
        // question is appended. Keep that answer as well as the one preceding
        // the current user, when their questions stay visible.
        retained.extend(answer_index(
            messages,
            latest,
            messages.len(),
            answer_budget,
        ));
        for (index, message) in prior.into_iter().chain(turns) {
            let chars = message_billable_chars(message);
            if chars > user_budget
                || message
                    .content
                    .as_str()
                    .is_some_and(is_preserved_user_or_image_stub)
            {
                // Do not preferentially retain older instructions across a
                // newer input whose full content cannot stay in this window.
                break;
            }
            retained.push(index);
            user_budget -= chars;
            if prior.is_some_and(|(prior_index, _)| index == prior_index) {
                retained.extend(answer_index(messages, index, latest, answer_budget));
            }
        }
        Self(retained)
    }

    pub(super) fn contains(&self, index: usize) -> bool {
        self.0.contains(&index)
    }

    pub(super) fn shift_after_insertion(&mut self, index: usize) {
        for retained in &mut self.0 {
            if *retained >= index {
                *retained += 1;
            }
        }
    }
}

fn answer_index(messages: &[Message], start: usize, end: usize, budget: usize) -> Option<usize> {
    let index = (start + 1..end).rev().find(|&i| is_answer(messages, i))?;
    let message = &messages[index];
    let replay_chars = message
        .reasoning_content
        .as_deref()
        .filter(|reasoning| is_persisted_reasoning_replay(reasoning))
        .map_or(0, |reasoning| reasoning.chars().count());
    (value_len_chars(&message.content) + replay_chars <= budget).then_some(index)
}

fn is_answer(messages: &[Message], index: usize) -> bool {
    let message = &messages[index];
    message.role == "assistant"
        && message.tool_calls.as_ref().is_none_or(Vec::is_empty)
        && message.tool_call_id.is_none()
        && message.content.as_str().is_some_and(|text| {
            !text.trim().is_empty() && !is_summary_note_text(text) && !is_archive_note_text(text)
        })
        && !crate::ai::history::has_runtime_message_metadata(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn runtime_derived_context_cannot_displace_the_previous_answer_after_reasoning_cleanup() {
        let recall =
            crate::ai::history::runtime_derived_assistant_message("historical recall".into());
        let review = crate::ai::history::runtime_derived_assistant_message("review advice".into());
        let mut current_answer = message("assistant", "current answer");
        current_answer.reasoning_content = Some("latest model reasoning".into());
        let mut messages = vec![
            message("user", "previous question"),
            message("assistant", "previous answer"),
            recall.clone(),
            message("user", "follow-up"),
            current_answer.clone(),
            review.clone(),
        ];
        keep_only_recent_reasoning_content(&mut messages);
        assert_eq!(
            messages[2..],
            [recall, message("user", "follow-up"), current_answer, review]
        );
        let dialogue = RecentDialogue::new(&messages, 64_000);
        assert!(dialogue.contains(1));
        assert!(!dialogue.contains(2));
        assert!(dialogue.contains(4));
        assert!(!dialogue.contains(5));
    }
}
