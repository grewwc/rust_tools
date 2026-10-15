//! Request-local context calibration. Cache discounts belong only to TPM.

use std::hash::{Hash, Hasher};

use super::RequestBody;
use crate::ai::history::Message;
use crate::ai::history::compress::BillableCounts;
use crate::ai::models;
use crate::ai::types::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Request-to-request prompt continuity, used to attribute provider prompt
/// cache misses to the projection change that caused them. Carries message
/// counts only; no bodies or fingerprints leave the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PrefixReuse {
    pub(crate) previous_messages: usize,
    pub(crate) current_messages: usize,
    /// Leading messages identical to the previous request; the provider cache
    /// can reuse at most this prefix (plus tools/system when settings match).
    pub(crate) common_prefix_messages: usize,
    /// Model, tools, or reasoning settings changed, which restarts the cache
    /// regardless of how many messages match.
    pub(crate) settings_changed: bool,
}

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
    message_counts: BillableCounts,
    actual_prompt_tokens: Option<usize>,
    awaiting_usage: bool,
    cache_key: Option<u64>,
    prefix_reuse: Option<PrefixReuse>,
}

/// Fallback ratio when a request carries no message text to observe.
pub(crate) const DEFAULT_CHARS_PER_TOKEN_MILLI: usize = 2_000;
/// Bounds on the derived ratio: below 0.5 chars/token only images or exotic
/// scripts remain, above 4 the estimate would promise more savings per char
/// than English prose ever yields.
const MIN_CHARS_PER_TOKEN_MILLI: usize = 500;
const MAX_CHARS_PER_TOKEN_MILLI: usize = 4_000;

/// Hash a JSON value without serializing it: no JSON escaping, no transient
/// string allocation. Object iteration follows the map's own order, which is
/// deterministic for a given value, so equal values hash equally. Numbers use
/// their display form (tiny, one numeric leaf at a time).
fn hash_value(value: &Value, state: &mut impl Hasher) {
    match value {
        Value::Null => 0u8.hash(state),
        Value::Bool(flag) => {
            1u8.hash(state);
            flag.hash(state);
        }
        Value::Number(number) => {
            2u8.hash(state);
            number.to_string().hash(state);
        }
        Value::String(text) => {
            3u8.hash(state);
            text.hash(state);
        }
        Value::Array(items) => {
            4u8.hash(state);
            items.len().hash(state);
            for item in items {
                hash_value(item, state);
            }
        }
        Value::Object(map) => {
            5u8.hash(state);
            map.len().hash(state);
            for (key, item) in map.iter() {
                key.hash(state);
                hash_value(item, state);
            }
        }
    }
}

/// Hash one request message without JSON-serializing it. Field order follows
/// the `Message` shape (`role`, `content`, `tool_calls`, `tool_call_id`,
/// `reasoning_content`); `None` vs `Some` discriminants are hashed explicitly
/// so absent and empty fields never collide.
fn fingerprint_message(message: &Message) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    message.role.hash(&mut hasher);
    hash_value(&message.content, &mut hasher);
    if let Some(calls) = message.tool_calls.as_ref() {
        true.hash(&mut hasher);
        calls.len().hash(&mut hasher);
        for call in calls {
            call.id.hash(&mut hasher);
            call.tool_type.hash(&mut hasher);
            call.function.name.hash(&mut hasher);
            call.function.arguments.hash(&mut hasher);
        }
    } else {
        false.hash(&mut hasher);
    }
    message.tool_call_id.hash(&mut hasher);
    message.reasoning_content.hash(&mut hasher);
    hasher.finish()
}

/// Hash the request settings side of the calibration identity without
/// serializing the (potentially large) tool schema. Covers the same fields as
/// the previous serialized tuple: model, thinking map, search flag, tools,
/// tool choice, reasoning effort/reasoning, encrypted-replay flag, stream.
fn fingerprint_settings(body: &RequestBody<'_>) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hash_settings_head(
        &mut hasher,
        &body.model,
        &body.thinking,
        body.enable_search,
    );
    if let Some(tools) = body.tools.as_ref() {
        true.hash(&mut hasher);
        hash_value(tools, &mut hasher);
    } else {
        false.hash(&mut hasher);
    }
    hash_settings_tail(
        &mut hasher,
        body.tool_choice.as_ref(),
        body.reasoning_effort,
        body.reasoning.as_ref(),
        body.reasoning_encrypted_replay,
        body.stream,
    );
    hasher.finish()
}

/// Settings fields hashed before the tools entry, shared by the body-based and
/// definitions-based fingerprints so field order cannot diverge between them.
fn hash_settings_head(
    state: &mut impl Hasher,
    model: &str,
    thinking: &Map<String, Value>,
    enable_search: Option<bool>,
) {
    model.hash(state);
    thinking.len().hash(state);
    for (key, value) in thinking.iter() {
        key.hash(state);
        hash_value(value, state);
    }
    enable_search.hash(state);
}

/// Settings fields hashed after the tools entry, shared like the head above.
fn hash_settings_tail(
    state: &mut impl Hasher,
    tool_choice: Option<&Value>,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
    reasoning_encrypted_replay: bool,
    stream: bool,
) {
    if let Some(choice) = tool_choice {
        true.hash(state);
        hash_value(choice, state);
    } else {
        false.hash(state);
    }
    reasoning_effort.hash(state);
    if let Some(reasoning) = reasoning {
        true.hash(state);
        hash_value(reasoning, state);
    } else {
        false.hash(state);
    }
    reasoning_encrypted_replay.hash(state);
    stream.hash(state);
}

/// Hash tool definitions exactly as `hash_value(&serde_json::to_value(defs))`
/// would, without building the intermediate `Value`. Mirrors serde's struct
/// layout in declaration order (`preserve_order`): each tool is an object with
/// `type` then `function`, whose value is an object with `name`,
/// `description`, then `parameters`. Discriminant bytes match `hash_value`
/// (`4` = array, `5` = object, `3` = string). The parity test below pins this
/// against the real serialization so preview and send fingerprints agree.
pub(super) fn hash_tool_definitions(defs: &[ToolDefinition], state: &mut impl Hasher) {
    4u8.hash(state);
    defs.len().hash(state);
    for def in defs {
        5u8.hash(state);
        2usize.hash(state);
        "type".hash(state);
        3u8.hash(state);
        def.tool_type.hash(state);
        "function".hash(state);
        5u8.hash(state);
        3usize.hash(state);
        "name".hash(state);
        3u8.hash(state);
        def.function.name.hash(state);
        "description".hash(state);
        3u8.hash(state);
        def.function.description.hash(state);
        "parameters".hash(state);
        hash_value(&def.function.parameters, state);
    }
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
            settings: Some(fingerprint_settings(body)),
            messages: Some(body.messages.iter().map(fingerprint_message).collect()),
            estimated_prompt_tokens: body.estimated_prompt_tokens,
            message_counts: body.message_counts,
            actual_prompt_tokens: None,
            awaiting_usage: true,
            cache_key: None,
            // Encrypted reasoning replay is part of the estimate (see
            // `builder::estimate_reasoning_items_tokens`), so its usage may
            // calibrate later requests like any other prompt content.
            prefix_reuse: None,
        }
    }

    /// Preview-path capture that avoids serializing the tool schema to `Value`.
    /// `tools_defs` must be `Some` exactly when the wire body would carry tools
    /// (non-empty and model-enabled, as in `agent_tools_for_request`); the
    /// settings hash and token estimate are identical to `capture` on the
    /// equivalent body (parity-tested below), so calibration continuity holds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_preview(
        session_id: &str,
        model: &str,
        endpoint: &str,
        messages: &[Message],
        thinking: &Map<String, Value>,
        enable_search: Option<bool>,
        tools_defs: Option<&[ToolDefinition]>,
        tool_choice: Option<&Value>,
        reasoning_effort: Option<&str>,
        reasoning: Option<&Value>,
        reasoning_encrypted_replay: bool,
        stream: bool,
        estimated_prompt_tokens: usize,
        message_counts: BillableCounts,
    ) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        // The body carries the request (possibly rewritten) model name, not the
        // registry key: match `fingerprint_settings` exactly, or calibration
        // continuity between preview and send budgets silently breaks.
        let request_model = models::request_model_name(model);
        hash_settings_head(&mut hasher, &request_model, thinking, enable_search);
        if let Some(defs) = tools_defs {
            true.hash(&mut hasher);
            hash_tool_definitions(defs, &mut hasher);
        } else {
            false.hash(&mut hasher);
        }
        hash_settings_tail(
            &mut hasher,
            tool_choice,
            reasoning_effort,
            reasoning,
            reasoning_encrypted_replay,
            stream,
        );
        Self {
            session_id: session_id.to_owned(),
            model: model.to_owned(),
            endpoint: endpoint.to_owned(),
            protocol: format!("{:?}", models::request_protocol_dialect(model, endpoint)),
            context_window: models::context_window_tokens(model),
            settings: Some(hasher.finish()),
            messages: Some(messages.iter().map(fingerprint_message).collect()),
            estimated_prompt_tokens,
            message_counts,
            actual_prompt_tokens: None,
            awaiting_usage: true,
            cache_key: None,
            prefix_reuse: None,
        }
    }

    pub(super) fn mark_sent_with_key(&mut self, api_key: &str) {
        self.cache_key = Some(cache_key_fingerprint(api_key));
    }

    /// Record how much of the previous request this one repeats verbatim.
    /// Unlike `compatible_usage`, this tolerates rewrites: it reports where the
    /// first rewrite happened instead of rejecting the comparison.
    pub(super) fn record_prefix_reuse(&mut self, previous: Option<&Self>) {
        self.prefix_reuse = previous.and_then(|previous| self.compare_prefix(previous));
    }

    fn compare_prefix(&self, previous: &Self) -> Option<PrefixReuse> {
        if self.session_id != previous.session_id {
            return None;
        }
        let (Some(current), Some(prior)) = (&self.messages, &previous.messages) else {
            return None;
        };
        Some(PrefixReuse {
            previous_messages: prior.len(),
            current_messages: current.len(),
            common_prefix_messages: current
                .iter()
                .zip(prior)
                .take_while(|(current, prior)| current == prior)
                .count(),
            settings_changed: self.model != previous.model
                || self.endpoint != previous.endpoint
                || self.settings != previous.settings,
        })
    }

    pub(crate) fn prefix_reuse(&self) -> Option<PrefixReuse> {
        self.prefix_reuse
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
        if previous.awaiting_usage
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

    /// Message chars per prompt token, in thousandths, for translating token
    /// deficits into char targets. Derived from this request's own script mix
    /// (ASCII-heavy history yields ~3, CJK ~1) and scaled by usage calibration
    /// when `context_prompt_tokens` differs from the raw estimate, so a
    /// provider that bills more than estimated also shrinks the ratio.
    pub(super) fn chars_per_token_milli(&self, context_prompt_tokens: usize) -> usize {
        let chars = self.message_counts.chars();
        let tokens = self.message_counts.tokens();
        if chars == 0 || tokens == 0 {
            return DEFAULT_CHARS_PER_TOKEN_MILLI;
        }
        let numerator = (chars as u128)
            .saturating_mul(1_000)
            .saturating_mul(self.estimated_prompt_tokens.max(1) as u128);
        let denominator = (tokens as u128).saturating_mul(context_prompt_tokens.max(1) as u128);
        usize::try_from(numerator / denominator)
            .unwrap_or(usize::MAX)
            .clamp(MIN_CHARS_PER_TOKEN_MILLI, MAX_CHARS_PER_TOKEN_MILLI)
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
    fn prompt_feedback_observation_is_consumed_once_and_replay_growth_is_calibrated() {
        let model = model();
        let messages = vec![message("user", "unchanged")];
        let mut previous = capture(&model, &messages, None);
        assert!(previous.record_usage("session", Some(&model), Some(2)));
        assert!(!previous.record_usage("session", Some(&model), Some(900)));
        assert_eq!(previous.actual_prompt_tokens, Some(2));
        // Reasoning replay rides on a newly appended tool round, so the verified
        // prefix still calibrates and the replay is charged as estimated growth.
        let mut grown = messages.clone();
        grown.push(Message {
            role: "assistant".to_owned(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call".to_owned(),
                tool_type: "function".to_owned(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_owned(),
                    arguments: "{}".to_owned(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        let replay = rustc_hash::FxHashMap::from_iter([(
            "call".to_owned(),
            vec![json!({"encrypted_content": "A".repeat(8_000)})],
        )]);
        let build = |items| {
            build_request_body(
                &model, &grown, true, false, None, None, None, None, None, None, items,
            )
        };
        let without_replay = build(None).estimated_prompt_tokens;
        let body = build(Some(&replay));
        assert!(body.estimated_prompt_tokens > without_replay + 1_000);
        let current = PromptTokenFeedback::capture(
            "session",
            &model,
            "https://example.invalid/v1/chat/completions",
            &body,
        );
        let growth = current.estimated_prompt_tokens - previous.estimated_prompt_tokens;
        assert_eq!(
            current.context_prompt_tokens(Some(&previous)),
            calibrated_context_tokens(previous.estimated_prompt_tokens, 2, growth)
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

    #[test]
    fn tool_definitions_hash_matches_serialized_schema() {
        use crate::ai::types::{FunctionDefinition, ToolDefinition};

        fn tool(name: &str, description: &str, parameters: Value) -> ToolDefinition {
            ToolDefinition {
                tool_type: "function".to_owned(),
                function: FunctionDefinition {
                    name: name.to_owned(),
                    description: description.to_owned(),
                    parameters,
                },
            }
        }

        fn assert_hash_parity(defs: &[ToolDefinition]) {
            let serialized = serde_json::to_value(defs).expect("tool defs must serialize");
            let mut direct = rustc_hash::FxHasher::default();
            super::hash_tool_definitions(defs, &mut direct);
            let mut through_value = rustc_hash::FxHasher::default();
            super::hash_value(&serialized, &mut through_value);
            assert_eq!(direct.finish(), through_value.finish());
        }

        assert_hash_parity(&[]);
        assert_hash_parity(&[tool("read_file", "read a file", json!({"type": "object"}))]);
        assert_hash_parity(&[
            tool(
                "a\"b\\c\n",
                "desc with \"quotes\", \\backslash\\, \u{1}control, 中文, 🙂",
                json!({"type": "object", "properties": {"p": {"type": "string"}}}),
            ),
            tool("", "", Value::Null),
            tool(
                "mcp__server__long_tool_name",
                &"x".repeat(5_000),
                json!([1, "two", {"three": [true, null, 1.5]}]),
            ),
        ]);
    }

    #[test]
    fn preview_capture_matches_body_capture_with_and_without_tools() {
        use crate::ai::types::{FunctionDefinition, ToolDefinition};

        let model = model();
        let endpoint = "https://example.invalid/v1/chat/completions";
        let messages = vec![message("user", "unchanged")];
        let defs = vec![ToolDefinition {
            tool_type: "function".to_owned(),
            function: FunctionDefinition {
                name: "budget_fixture".to_owned(),
                description: "large schema content ".repeat(100),
                parameters: json!({"type": "object", "properties": {}}),
            },
        }];
        for tools_defs in [None, Some(defs.as_slice())] {
            let tools = tools_defs.map(|defs| serde_json::to_value(defs).unwrap());
            let tool_choice = tools
                .as_ref()
                .map(|_| Value::String("auto".to_owned()));
            let body = build_request_body(
                &model, &messages, true, false, None, tools, tool_choice, None, None, None,
                None,
            );
            let expected = PromptTokenFeedback::capture("session", &model, endpoint, &body);
            // The definitions-based estimate must agree with the body's own
            // estimate, which the send path derives from the serialized value.
            assert_eq!(
                super::super::builder::estimate_request_prompt_tokens_from_definitions(
                    &messages, tools_defs, None
                ),
                body.estimated_prompt_tokens,
            );
            let actual = PromptTokenFeedback::capture_preview(
                "session",
                &model,
                endpoint,
                &messages,
                &body.thinking,
                body.enable_search,
                tools_defs,
                body.tool_choice.as_ref(),
                body.reasoning_effort,
                body.reasoning.as_ref(),
                body.reasoning_encrypted_replay,
                body.stream,
                body.estimated_prompt_tokens,
                body.message_counts,
            );
            assert_eq!(actual.settings, expected.settings);
            assert_eq!(actual.messages, expected.messages);
            assert_eq!(
                actual.estimated_prompt_tokens,
                expected.estimated_prompt_tokens
            );
            assert_eq!(actual.message_counts, expected.message_counts);
        }
    }
}
