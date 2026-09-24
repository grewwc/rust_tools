//! Request-local context calibration. Cache discounts belong only to TPM.

use std::hash::{Hash, Hasher};
use std::io::{self, Write};

use serde::Serialize;

use super::RequestBody;
use crate::ai::models;

/// A pending request becomes usable feedback only after its own response reports
/// nonzero prompt usage. Fingerprints avoid retaining another copy of history.
/// This is an estimate, not a tokenizer or proof that a request fits the window.
#[derive(Debug)]
pub(crate) struct PromptTokenFeedback {
    session_id: String,
    model: String,
    endpoint: String,
    protocol: String,
    context_window: usize,
    settings: Option<u64>,
    messages: Option<Vec<u64>>,
    estimated_prompt_tokens: usize,
    actual_prompt_tokens: Option<usize>,
    awaiting_usage: bool,
    cache_key: Option<u64>,
    supports_calibration: bool,
}

struct FingerprintWriter(rustc_hash::FxHasher);

impl Write for FingerprintWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn fingerprint(value: &impl Serialize) -> Option<u64> {
    let mut writer = FingerprintWriter(rustc_hash::FxHasher::default());
    serde_json::to_writer(&mut writer, value).ok()?;
    Some(writer.0.finish())
}

fn cache_key_fingerprint(api_key: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    api_key.trim().hash(&mut hasher);
    hasher.finish()
}

impl PromptTokenFeedback {
    pub(super) fn capture(
        session_id: &str,
        model: &str,
        endpoint: &str,
        body: &RequestBody<'_>,
    ) -> Self {
        Self {
            session_id: session_id.to_owned(),
            model: model.to_owned(),
            endpoint: endpoint.to_owned(),
            protocol: format!("{:?}", models::request_protocol_dialect(model, endpoint)),
            context_window: models::context_window_tokens(model),
            settings: fingerprint(&(
                &body.model,
                &body.thinking,
                body.enable_search,
                &body.tools,
                &body.tool_choice,
                body.reasoning_effort,
                &body.reasoning,
                body.reasoning_encrypted_replay,
                body.stream,
            )),
            messages: body.messages.iter().map(fingerprint).collect(),
            estimated_prompt_tokens: body.estimated_prompt_tokens,
            actual_prompt_tokens: None,
            awaiting_usage: true,
            cache_key: None,
            // The character estimator does not measure opaque reasoning replay.
            // Do not extrapolate its ratio when that side channel is present.
            supports_calibration: body.reasoning_items.is_none_or(|items| items.is_empty()),
        }
    }

    pub(super) fn mark_sent_with_key(&mut self, api_key: &str) {
        self.cache_key = Some(cache_key_fingerprint(api_key));
    }

    /// Consume the pending observation even when usage is missing or mismatched.
    /// A later response must never fill an earlier request's empty usage slot.
    pub(crate) fn record_usage(
        &mut self,
        session_id: &str,
        response_model: Option<&str>,
        prompt_tokens: Option<u64>,
    ) -> bool {
        if !std::mem::take(&mut self.awaiting_usage) {
            return false;
        }
        self.actual_prompt_tokens = prompt_tokens
            .filter(|tokens| *tokens > 0)
            .filter(|_| {
                self.session_id == session_id && response_model == Some(self.model.as_str())
            })
            .and_then(|tokens| usize::try_from(tokens).ok());
        self.actual_prompt_tokens.is_some()
    }

    pub(super) fn compatible_usage(&self, previous: &Self) -> Option<usize> {
        if !self.supports_calibration
            || !previous.supports_calibration
            || previous.awaiting_usage
            || self.session_id != previous.session_id
            || self.model != previous.model
            || self.endpoint != previous.endpoint
            || self.protocol != previous.protocol
            || self.context_window != previous.context_window
            || self.settings.is_none()
            || self.settings != previous.settings
            || previous.estimated_prompt_tokens == 0
            || self.estimated_prompt_tokens < previous.estimated_prompt_tokens
        {
            return None;
        }
        let (Some(messages), Some(prefix)) = (&self.messages, &previous.messages) else {
            return None;
        };
        // Compression, replacement, and reordering invalidate feedback even if
        // the resulting prompt is larger. Only an unchanged prefix may grow.
        if prefix.is_empty() || !messages.starts_with(prefix) {
            return None;
        }
        previous.actual_prompt_tokens
    }

    pub(super) fn context_prompt_tokens(&self, previous: Option<&Self>) -> usize {
        let Some((previous, actual)) = previous.and_then(|previous| {
            self.compatible_usage(previous)
                .map(|actual| (previous, actual))
        }) else {
            return self.estimated_prompt_tokens;
        };
        let growth = self
            .estimated_prompt_tokens
            .saturating_sub(previous.estimated_prompt_tokens);
        calibrated_context_tokens(previous.estimated_prompt_tokens, actual, growth)
    }

    pub(super) fn reusable_cached_tokens(
        &self,
        previous: Option<&Self>,
        cached_tokens: Option<u64>,
        api_key: &str,
    ) -> Option<u64> {
        let previous = previous?;
        let actual = self.compatible_usage(previous)?;
        if previous.cache_key != Some(cache_key_fingerprint(api_key)) {
            return None;
        }
        Some(cached_tokens?.min(u64::try_from(actual).unwrap_or(u64::MAX)))
    }
}

fn calibrated_context_tokens(previous_estimate: usize, actual: usize, growth: usize) -> usize {
    // Retain actual usage for the verified prefix, but charge at least the full
    // character estimate for new content. A small English prefix must not teach
    // a discount for a later large CJK/code/tool payload. If actual usage was
    // higher than the estimate, carry that inflation forward as well.
    let denominator = previous_estimate.max(1) as u128;
    let scaled_growth = (growth as u128)
        .saturating_mul(actual as u128)
        .div_ceil(denominator);
    let scaled_growth = usize::try_from(scaled_growth).unwrap_or(usize::MAX);
    actual.saturating_add(growth.max(scaled_growth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;
    use crate::ai::request::builder::{build_request_body, clamp_with_estimated_prompt};
    use serde_json::{Value, json};

    fn model() -> String {
        crate::ai::model_names::all()
            .iter()
            .find(|entry| models::max_output_tokens(&entry.key).is_some())
            .expect("registry must contain a declared output cap")
            .key
            .clone()
    }

    fn message(role: &str, text: &str) -> Message {
        Message {
            role: role.to_owned(),
            content: Value::String(text.to_owned()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn capture(model: &str, messages: &[Message], tools: Option<Value>) -> PromptTokenFeedback {
        let body = build_request_body(
            model, messages, true, false, None, tools, None, None, None, None, None,
        );
        PromptTokenFeedback::capture(
            "session",
            model,
            "https://example.invalid/v1/chat/completions",
            &body,
        )
    }

    fn observed(mut feedback: PromptTokenFeedback, actual: u64) -> PromptTokenFeedback {
        let model = feedback.model.clone();
        feedback.mark_sent_with_key("key-a");
        feedback.record_usage("session", Some(&model), Some(actual));
        feedback
    }

    #[test]
    fn prompt_feedback_small_request_then_large_tool_append_charges_growth() {
        let model = model();
        let mut messages = vec![message("user", "small request")];
        let tools = Some(json!([{"type":"function", "function":{"name":"read_file"}}]));
        let previous = capture(&model, &messages, tools.clone());
        let actual = (previous.estimated_prompt_tokens / 2).max(1);
        let previous = observed(previous, actual as u64);
        messages.push(message("tool", &"large tool payload ".repeat(20_000)));
        let current = capture(&model, &messages, tools);
        let growth = current.estimated_prompt_tokens - previous.estimated_prompt_tokens;
        assert_eq!(
            current.context_prompt_tokens(Some(&previous)),
            actual + growth
        );
        assert!(growth > 100_000);
        let cap = models::max_output_tokens(&model).unwrap();
        let calibrated = clamp_with_estimated_prompt(&model, actual + growth, cap);
        let stale = clamp_with_estimated_prompt(&model, actual, cap);
        assert!(calibrated <= stale);
    }

    #[test]
    fn prompt_feedback_compression_and_equal_size_rewrites_invalidate() {
        let model = model();
        let original = vec![message("user", &"original ".repeat(2_000))];
        let previous = observed(capture(&model, &original, None), 7_000);
        for messages in [
            vec![message("user", "compressed")],
            vec![message("user", &"rewritten".repeat(2_000))],
            vec![message("user", &"replacement ".repeat(4_000))],
        ] {
            let current = capture(&model, &messages, None);
            assert_eq!(
                current.context_prompt_tokens(Some(&previous)),
                current.estimated_prompt_tokens
            );
            assert_eq!(
                current.reusable_cached_tokens(Some(&previous), Some(6_000), "key-a"),
                None
            );
        }
    }

    #[test]
    fn prompt_feedback_model_protocol_tools_session_and_endpoint_changes_invalidate() {
        let model = model();
        let messages = vec![message("user", "unchanged")];
        let previous = observed(capture(&model, &messages, None), 2);
        for changed in 0..5 {
            let mut current = capture(&model, &messages, None);
            match changed {
                0 => current.model.push_str("-different"),
                1 => current.protocol.push_str("-different"),
                2 => current.session_id.push_str("-different"),
                3 => current.endpoint.push_str("/different"),
                _ => current.context_window = current.context_window.saturating_add(1),
            }
            assert_eq!(
                current.context_prompt_tokens(Some(&previous)),
                current.estimated_prompt_tokens
            );
        }
        let changed_tools = capture(&model, &messages, Some(json!([{"name":"new_tool"}])));
        assert_eq!(
            changed_tools.context_prompt_tokens(Some(&previous)),
            changed_tools.estimated_prompt_tokens
        );
    }

    #[test]
    fn prompt_feedback_missing_zero_or_wrong_response_usage_is_not_reused() {
        let model = model();
        let messages = vec![message("user", "unchanged")];
        let current = capture(&model, &messages, None);
        for usage in [None, Some(0)] {
            let mut previous = capture(&model, &messages, None);
            previous.record_usage("session", Some(&model), usage);
            previous.record_usage("session", Some(&model), Some(1));
            assert_eq!(
                current.context_prompt_tokens(Some(&previous)),
                current.estimated_prompt_tokens
            );
        }
        let mut previous = capture(&model, &messages, None);
        previous.record_usage("session", Some("different-response-model"), Some(1));
        assert_eq!(
            current.context_prompt_tokens(Some(&previous)),
            current.estimated_prompt_tokens
        );
        let pending = capture(&model, &messages, None);
        assert_eq!(
            current.context_prompt_tokens(Some(&pending)),
            current.estimated_prompt_tokens
        );
        assert_eq!(
            current.context_prompt_tokens(None),
            current.estimated_prompt_tokens
        );
    }

    #[test]
    fn prompt_feedback_observation_is_consumed_once_and_replay_is_not_calibrated() {
        let model = model();
        let messages = vec![message("user", "unchanged")];
        let mut previous = capture(&model, &messages, None);
        assert!(previous.record_usage("session", Some(&model), Some(2)));
        assert!(!previous.record_usage("session", Some(&model), Some(900)));
        assert_eq!(previous.actual_prompt_tokens, Some(2));
        let replay = rustc_hash::FxHashMap::from_iter([(
            "call".to_owned(),
            vec![json!({"encrypted_content": "opaque"})],
        )]);
        let body = build_request_body(
            &model,
            &messages,
            true,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&replay),
        );
        let current = PromptTokenFeedback::capture(
            "session",
            &model,
            "https://example.invalid/v1/chat/completions",
            &body,
        );
        assert!(!current.supports_calibration);
        assert_eq!(
            current.context_prompt_tokens(Some(&previous)),
            current.estimated_prompt_tokens
        );
        assert_eq!(
            current.reusable_cached_tokens(Some(&previous), Some(1), "key-a"),
            None
        );
        let mut mismatched = capture(&model, &messages, None);
        assert!(!mismatched.record_usage("other-session", Some(&model), Some(2)));
        assert!(!mismatched.record_usage("session", Some(&model), Some(2)));
    }

    #[test]
    fn prompt_feedback_cache_discount_is_scoped_and_never_reduces_context() {
        let model = model();
        let messages = vec![message("user", &"cached prompt ".repeat(100))];
        let previous = observed(capture(&model, &messages, None), 500);
        let current = capture(&model, &messages, None);
        let context = current.context_prompt_tokens(Some(&previous));
        assert_eq!(context, 500);
        let cached = current.reusable_cached_tokens(Some(&previous), Some(900), "key-a");
        assert_eq!(cached, Some(500));
        assert_eq!(
            super::super::token_budget::tpm_prompt_tokens(context, cached),
            1
        );
        assert_eq!(current.context_prompt_tokens(Some(&previous)), 500);
        assert_eq!(
            current.reusable_cached_tokens(Some(&previous), Some(500), "key-b"),
            None
        );
    }

    #[test]
    fn prompt_feedback_overflow_boundaries_saturate_and_inflation_is_preserved() {
        assert_eq!(calibrated_context_tokens(100, 200, 50), 300);
        assert_eq!(calibrated_context_tokens(100, 20, 50), 70);
        assert_eq!(
            calibrated_context_tokens(1, usize::MAX, usize::MAX),
            usize::MAX
        );
        assert_eq!(
            calibrated_context_tokens(usize::MAX, 1, usize::MAX),
            usize::MAX
        );
        assert_eq!(calibrated_context_tokens(0, 0, 0), 0);
        let model = model();
        let cap = models::max_output_tokens(&model).unwrap();
        assert_eq!(
            clamp_with_estimated_prompt(&model, usize::MAX, cap),
            super::super::builder::MIN_OUTPUT_TOKENS_FLOOR
        );
    }
}
