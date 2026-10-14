use rustc_hash::FxHashSet;

use super::{Message, ROLE_INTERNAL_NOTE, last_real_user_index};

/// A projection-only filter shared by requests and rebuildable snapshots.
/// Only plain runtime tool guidance is eligible; protocol/extra payloads survive.
pub(in crate::ai) struct ToolGuidanceFilter {
    current_user_index: Option<usize>,
    index: usize,
    seen: FxHashSet<String>,
}

impl ToolGuidanceFilter {
    pub(in crate::ai) fn new(messages: &[Message]) -> Self {
        Self {
            current_user_index: last_real_user_index(messages),
            index: 0,
            seen: FxHashSet::default(),
        }
    }

    pub(in crate::ai) fn keep(&mut self, message: &Message) -> bool {
        let index = self.index;
        self.index += 1;
        let Some(current_user_index) = self.current_user_index else {
            return true;
        };
        if message.role != ROLE_INTERNAL_NOTE
            || message.tool_calls.is_some()
            || message.tool_call_id.is_some()
            || message.reasoning_content.is_some()
        {
            return true;
        }
        let Some(text) = message.content.as_str() else {
            return true;
        };
        if !text.starts_with("[tool_first_use_guidance name=") {
            return true;
        }
        // Do not deduplicate by tool name: different bodies may add constraints.
        index >= current_user_index && self.seen.insert(text.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: &str, text: &str) -> Message {
        Message {
            role: role.to_owned(),
            content: json!(text),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn clean(mut messages: Vec<Message>) -> Vec<Message> {
        let mut filter = ToolGuidanceFilter::new(&messages);
        messages.retain(|message| filter.keep(message));
        messages
    }

    #[test]
    fn tool_guidance_filter_preserves_boundaries_versions_and_payloads() {
        let guidance = message(
            ROLE_INTERNAL_NOTE,
            "[tool_first_use_guidance name=read_file]\nRead serially.",
        );
        let revised = message(
            ROLE_INTERNAL_NOTE,
            "[tool_first_use_guidance name=read_file]\nAdditional rule.",
        );
        let user = message("user", "Keep current constraints");
        let quoted = message("assistant", guidance.content.as_str().unwrap());
        let synthetic = super::super::runtime_synthetic_user_message(json!("task finished"));
        let mut payload = guidance.clone();
        payload.reasoning_content = Some("extra state".into());
        let mut protocol = guidance.clone();
        protocol.tool_call_id = Some("call-1".into());
        let current = vec![user, guidance.clone(), revised, synthetic, quoted];
        let mut input = vec![payload.clone(), protocol.clone(), guidance.clone()];
        input.extend(current.clone());
        input.push(guidance);
        let mut expected = vec![payload, protocol];
        expected.extend(current);
        assert_eq!(clean(input), expected);
        assert_eq!(clean(expected.clone()), expected);
    }

    #[test]
    fn tool_guidance_filter_keeps_unknown_boundary_and_non_guidance() {
        let guidance = message(
            ROLE_INTERNAL_NOTE,
            "[tool_first_use_guidance name=read_file]\nRule.",
        );
        let unknown = vec![guidance.clone(), guidance];
        assert_eq!(clean(unknown.clone()), unknown);
        let note = message(ROLE_INTERNAL_NOTE, "Safety constraint");
        let messages = vec![
            message("user", "old"),
            note.clone(),
            note,
            message("user", "new"),
        ];
        assert_eq!(clean(messages.clone()), messages);
    }
}
