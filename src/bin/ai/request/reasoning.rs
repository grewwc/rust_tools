//! Reasoning/thinking-mode control + prompt-cache breakpoint injection.
//!
//! Reasoning-related logic extracted from request/mod.rs:
//! - thinking wire-field parsing (per-provider-adapter field differences)
//! - `reasoning_content` echo completion for thinking models
//! - thinking-off injection for auxiliary/background requests
//! - prompt-cache breakpoint injection (`cache_control`)
//! - `reasoning_effort` tier resolution

use serde_json::{Map, Value, json};

use super::super::{
    history::{Message, is_system_like_role},
    models,
    provider::{
        ApiProvider, ReasoningEffort, ThinkingOffCapability, adapter_for,
        compatible_wire_shapes, reasoning_effort_reduces_thinking_for, thinking_dialect_for,
        thinking_off_capability_for,
    },
    types::App,
};
use crate::commonw::configw;

/// Resolve the thinking/reasoning field shapes for each provider adapter.
///
/// Returns a triple:
/// 1. the top-level thinking object (or other provider-specific fields), empty
///    when nothing is injected;
/// 2. the top-level `reasoning_effort` string (some providers place it at the
///    top level);
/// 3. the nested `reasoning` object (some providers place it in `body.reasoning`).
///
/// The effort **value** is adapted per vendor by the dialect first
/// ([`ThinkingDialect::adapt_effort`]) **unless the registry declares a wire
/// placement** ([`models::reasoning_effort_wire`]): a declared placement means
/// the model's effort is vendor-verified graded (e.g. DeepSeek v4 on
/// api.deepseek.com / OpenCode Zen, DashScope DeepSeek v4), so the tier passes
/// through verbatim. Undeclared models fall back to the dialect default —
/// binary-switch vendors omit the tier, everyone else passes it through.
pub(super) fn resolve_reasoning_wire_controls<'a>(
    model: &'a str,
    endpoint: &str,
    enable_thinking: bool,
    reasoning_effort: Option<&'a str>,
) -> (Map<String, Value>, Option<&'a str>, Option<Value>) {
    let adapter_kind = models::model_adapter(model);
    let adapter = adapter_for(adapter_kind, &endpoint);
    let request_model = models::request_model_name(model);
    let thinking_dialect = thinking_dialect_for(adapter_kind, &request_model, &endpoint);
    // Per-vendor effort **value** adaptation: the dialect owns the mapping and
    // this layer never invents a wire value. A registry-declared
    // `reasoning_effort_wire` takes precedence (the model's effort is
    // vendor-verified graded and placement is explicit); only undeclared models
    // go through the dialect default, where binary-switch dialects omit the
    // tier so the placement routing below has nothing to place.
    let reasoning_effort = if models::reasoning_effort_wire(model).is_some() {
        reasoning_effort
    } else {
        thinking_dialect.adapt_effort(reasoning_effort)
    };
    // The user's `enable_search` request is passed in by the builder; only the
    // reasoning/thinking triple matters here, so `None` is a placeholder — this
    // function does not depend on the returned enable_search.
    let (_, top_level_reasoning_effort, nested_reasoning) = if let Some(wire) =
        models::reasoning_effort_wire(model)
    {
        match wire {
            crate::ai::model_names::ReasoningEffortWire::TopLevel => (None, reasoning_effort, None),
            crate::ai::model_names::ReasoningEffortWire::Nested => (
                None,
                None,
                reasoning_effort.map(|effort| json!({ "effort": effort })),
            ),
        }
    } else if adapter_kind == ApiProvider::Compatible {
        // Compatible providers split by endpoint: DashScope uses the DashScope
        // shape, other plain OpenAI-compatible endpoints (e.g. the internal
        // modelhub) use the OpenAI shape. The adapter.reasoning_*() defaults
        // cannot be used here because the trait singleton cannot see the endpoint.
        compatible_wire_shapes(endpoint, None, reasoning_effort)
    } else {
        (
            None,
            adapter.reasoning_top_level(reasoning_effort),
            adapter.reasoning_nested(reasoning_effort),
        )
    };
    let thinking = thinking_dialect.fields(enable_thinking, top_level_reasoning_effort);
    (thinking, top_level_reasoning_effort, nested_reasoning)
}

/// Normalize the assistant `reasoning_content` replay policy by model capability:
/// - Models that must echo the field back (DeepSeekThinkingDialect) keep the
///   field shape on **every** replayed assistant message, including non-tool-call
///   turns and empty reasoning: an own exact-replay blob decodes back to the
///   original provider text, foreign/encrypted markers are cleared to an empty
///   string, and a missing field is filled with an empty string — a missing
///   field is rejected by the gateway with 400 "The `reasoning_content` in the
///   thinking mode must be passed back", while an empty string passes.
/// - Models declaring exact replay (GLM) only replay the original provider text
///   for tool-call rounds (by decoding the internal marker) to preserve
///   cross-tool-call continuity; non-tool-call messages strip hidden reasoning.
/// - All other models strip hidden reasoning entirely, avoiding cross-turn
///   leakage and context bloat.
pub(super) fn normalize_reasoning_content_replay_for_model(model: &str, messages: &mut [Message]) {
    let exact_replay = models::reasoning_content_replay_enabled(model);
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    let dialect = thinking_dialect_for(adapter_kind, &request_model, &endpoint);
    let shape_only_replay = dialect.requires_reasoning_content_echo();

    for message in messages.iter_mut() {
        if message.role != "assistant" {
            continue;
        }
        if shape_only_replay {
            // The DeepSeek thinking-mode gateway validates that every replayed
            // assistant message carries `reasoning_content` (see the function
            // docs), so this branch covers all assistant messages, not only
            // tool-call rounds. Persisted replay state (exact or encrypted
            // markers) must never be sent verbatim to the provider: an own exact
            // marker decodes back to the original text, and any other marker
            // (cross-model / encrypted) is cleared to an empty string to keep
            // only the field shape.
            let current = message.reasoning_content.take();
            message.reasoning_content = Some(match &current {
                Some(reasoning)
                    if crate::ai::history::compress::is_persisted_reasoning_replay(
                        reasoning,
                    ) =>
                {
                    crate::ai::history::compress::decode_reasoning_replay_for_model(
                        model, reasoning,
                    )
                    .unwrap_or_default()
                }
                Some(reasoning) => reasoning.clone(),
                None => String::new(),
            });
            continue;
        }
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
        if exact_replay {
            // Exact continuation state can only be produced by the same model;
            // untagged content (e.g. pre-switch GPT reasoning) and other exact
            // models' state must not be replayed across models.
            if !has_tool_calls {
                message.reasoning_content = None;
            } else {
                message.reasoning_content =
                    message.reasoning_content.as_deref().and_then(|reasoning| {
                        crate::ai::history::compress::decode_reasoning_replay_for_model(
                            model, reasoning,
                        )
                    });
            }
            continue;
        }
        message.reasoning_content = None;
    }
}

/// Rebuilds the side-channel map for Responses encrypted-reasoning replay from
/// persisted messages (key = first tool_call id).
///
/// Background: an encrypted-replay model's encrypted reasoning lives in the
/// in-memory `turn_reasoning_items` during the turn that produced it, but that
/// is turn-scoped, cleared each turn, and lost on process exit. Across turns or
/// after a Ctrl+C resume, it can only be recovered from the encoded blob
/// persisted in `reasoning_content`. This function scans the current request
/// projection (already pruned naturally by compression, so it only replays
/// "recent non-folded turns", matching exact-replay's replay scope) and decodes
/// each marked assistant tool-call turn whose source model matches the current
/// model back into items, attaching them to its first tool_call id.
///
/// Only fills keys missing from `live`: the in-memory side channel (freshest
/// capture of the current turn) wins; persisted decoding only fills historical
/// gaps, so an existing key is never overwritten by a stale value. Cross-model
/// (marker model != current model) decoding returns None and is skipped
/// automatically.
pub(super) fn reconstruct_encrypted_reasoning_items_for_model(
    model: &str,
    messages: &[Message],
    live: &rustc_hash::FxHashMap<String, Vec<Value>>,
) -> rustc_hash::FxHashMap<String, Vec<Value>> {
    let mut merged = live.clone();
    if !models::reasoning_encrypted_replay_enabled(model)
        || !crate::ai::history::compress::encrypted_reasoning_replay_runtime_enabled()
    {
        return merged;
    }
    for message in messages {
        if message.role != "assistant" {
            continue;
        }
        let Some(first_call_id) = message
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map(|call| call.id.clone())
        else {
            continue;
        };
        if merged.contains_key(&first_call_id) {
            continue;
        }
        let Some(encoded) = message.reasoning_content.as_deref() else {
            continue;
        };
        if let Some(items) =
            crate::ai::history::compress::decode_encrypted_reasoning_replay_for_model(
                model, encoded,
            )
        {
            if !items.is_empty() {
                merged.insert(first_call_id, items);
            }
        }
    }
    merged
}

/// Merges the provider adapter's thinking fields into an auxiliary/background
/// request body.
///
/// Auxiliary (non-main-path) and background requests always turn thinking off
/// (`enable_thinking=false`); each adapter decides which keys to write
/// (`enable_thinking:false` / `thinking:{"type":"disabled"}` / or nothing), and
/// the core layer no longer discriminates by provider.
pub(crate) fn apply_aux_thinking_fields(model: &str, body: &mut Value) {
    let endpoint = models::endpoint_for_model(model, "");
    let (fields, _, _) = resolve_reasoning_wire_controls(model, &endpoint, false, None);
    if fields.is_empty() {
        return;
    }
    if let Some(map) = body.as_object_mut() {
        for (key, value) in fields {
            map.insert(key, value);
        }
    }
}

/// Whether opt-in explicit prompt-cache breakpoint injection is enabled.
///
/// `cache_control` is a provider/model-level capability declared by the
/// `explicit_prompt_cache` field in the model registry (models/); plain
/// OpenAI-compatible models may not accept this extension field.
pub(super) fn prompt_cache_enabled_for_model(model: &str) -> bool {
    prompt_cache_config_enabled() && models::explicit_prompt_cache_enabled(model)
}

fn prompt_cache_config_enabled() -> bool {
    configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_ENABLE,
            "false",
        )
        .trim()
        .eq_ignore_ascii_case("true")
}

/// Rewrites the first system / internal_note message's plain-text content into
/// a content-block array carrying `cache_control`, as an explicit prompt-cache
/// breakpoint. Only converts when the content is currently a string; idempotent
/// and never touches other messages.
pub(super) fn apply_prompt_cache_breakpoint(messages: &mut [Message]) {
    for message in messages.iter_mut() {
        if !is_system_like_role(&message.role) {
            continue;
        }
        if let Value::String(text) = &message.content {
            message.content = json!([
                {
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" }
                }
            ]);
        }
        // Setting the breakpoint on the first system-like message only is sufficient.
        break;
    }
}

/// Per-vendor thinking-off capability for the current model (see
/// [`ThinkingDialect::thinking_off_capability`]).
///
/// Deliberately resolved through the provider dialect layer rather than the
/// registry: registry fields like `reasoning_effort_wire` describe the effort
/// wire shape, not whether a dedicated thinking off-switch exists (e.g. the
/// DashScope DeepSeek entries declare `reasoning_effort_wire: "top_level"` yet
/// still turn thinking off via the `enable_thinking` switch).
pub(crate) fn model_thinking_off_capability(model: &str) -> ThinkingOffCapability {
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    thinking_off_capability_for(models::model_adapter(model), &request_model, &endpoint)
}

/// Whether the current model's thinking has a wire effort gradation.
///
/// Registry-driven first: a model that declares `reasoning_effort_wire`
/// (models/) has vendor-verified graded effort (e.g. DeepSeek v4 on
/// api.deepseek.com / OpenCode Zen, DashScope DeepSeek v4), so its resolved
/// tier is sent and shown as a real gradation. Undeclared models fall back to
/// the thinking-dialect default, where binary-switch dialects (DeepSeek /
/// DashScope) treat effort as a no-op and omit it from the wire.
pub(crate) fn model_effort_graded(model: &str) -> bool {
    models::reasoning_effort_reduces_thinking(model)
}

/// Resolve the effective reasoning intensity for the current session, highest
/// priority first:
/// 1. CLI argument `--reasoning-effort` or the `/model effort <x>` override
///    stored in [`App.cli.reasoning_effort_override`] (`Some(None)` = user
///    explicitly disabled; `None` = not set);
/// 2. The model registry ([models/](../../../../models)) default `reasoning_effort`;
/// 3. `None` -- no field injected, server default applies.
///
/// An explicit "off" (`Some(None)`) is treated as a request to actually turn
/// thinking off. Per-vendor adaptation, driven by the dialect's declared off
/// capability:
/// - [`ThinkingOffCapability::RealSwitch`] dialects turn thinking off in
///   `resolve_thinking` (DashScope `enable_thinking: false` / DeepSeek
///   `thinking: {"type":"disabled"}`) and omit the effort field here;
/// - [`ThinkingOffCapability::EffortNone`] dialects express off as
///   [`ReasoningEffort::None`] ("none" on the wire) — the same value the
///   truncation force-off fallback emits — for thinking-enabled models;
/// - [`ThinkingOffCapability::Unsupported`] dialects omit the field; the
///   `/effort off` handler reports that thinking stays on.
pub(crate) fn resolve_reasoning_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    if let Some(Some(level)) = app.cli.reasoning_effort_override.as_ref() {
        return Some(*level);
    }
    if app.cli.reasoning_effort_override == Some(None) {
        // User explicitly typed `/effort off` (or `--reasoning-effort off`).
        match model_thinking_off_capability(model) {
            ThinkingOffCapability::EffortNone if models::enable_thinking(model) => {
                return Some(ReasoningEffort::None);
            }
            _ => return None,
        }
    }
    models::default_reasoning_effort(model)
}

/// Apply the truncation ladder's last-resort force-off fallback to the reasoning-effort value.
///
/// When `thinking_disabled_override` is active (set by the orchestrator after repeated
/// truncation when lowering effort alone cannot converge), thinking must actually be turned off
/// on the wire. For dialects that control thinking solely through `reasoning_effort` (OpenAI
/// family / Responses — `NoThinkingDialect` sends no thinking field), the only wire value that
/// disables thinking is `"none"`, so the effort is mapped to `ReasoningEffort::None`. Dialects
/// with a real off-switch (DashScope `enable_thinking: false`, DeepSeek
/// `thinking: {"type":"disabled"}`) are already handled through `enable_thinking=false`;
/// `reasoning_effort_reduces_thinking_for` is false for them, so their effort passes through
/// unchanged. The graduated effort ladder itself deliberately never sends `"none"`
/// (orchestrator.rs) — this maps only the force-off fallback, not ladder retries.
pub(crate) fn apply_thinking_force_off_effort<'a>(
    thinking_disabled_override: bool,
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
    effort: Option<&'a str>,
) -> Option<&'a str> {
    if thinking_disabled_override
        && reasoning_effort_reduces_thinking_for(provider, model, endpoint)
    {
        Some(ReasoningEffort::None.as_str())
    } else {
        effort
    }
}

/// Returns the reasoning intensity shown in the input box for the current
/// request. When no field is sent, it is explicitly labeled as the server
/// default, so a model without a default tier is not mis-displayed as a
/// specific effort.
pub(crate) fn reasoning_effort_display_label(app: &App, model: &str) -> &'static str {
    if app.cli.reasoning_effort_override == Some(None) {
        // Explicit `/effort off`: show "off" (the user's intent) instead of the
        // underlying "none"/"server default" value. When the vendor has no off
        // mechanism at all, say so instead of implying thinking was turned off.
        return match model_thinking_off_capability(model) {
            ThinkingOffCapability::Unsupported => "off (unsupported)",
            _ => "off",
        };
    }
    match resolve_reasoning_effort(app, model) {
        Some(effort) => effort.as_str(),
        None => "server default",
    }
}

#[cfg(test)]
mod encrypted_replay_reconstruct_tests {
    use super::reconstruct_encrypted_reasoning_items_for_model;
    use crate::ai::history::Message;
    use crate::ai::history::compress::encode_encrypted_reasoning_replay_state;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::types::{FunctionCall, ToolCall};
    use rustc_hash::FxHashMap;
    use serde_json::{Value, json};

    fn assistant_call_with_reasoning(id: &str, reasoning: Option<String>) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: reasoning,
        }
    }

    #[test]
    fn rebuilds_items_from_encoded_history_for_encrypted_model() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Must resolve in the model registry with reasoning_encrypted_replay: true
        // (models/muse-spark-1.3-contributor.json); an unknown identifier makes the
        // reconstruct gate return early and the rebuild never runs.
        let model = "muse-spark-1.3-contributor";
        let items = vec![json!({"type":"reasoning","encrypted_content":"ENC"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(model, &items)),
        )];
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
            model,
            &messages,
            &FxHashMap::default(),
        );
        assert_eq!(rebuilt.get("call-1"), Some(&items));
    }

    #[test]
    fn live_side_channel_takes_precedence_over_history() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = "muse-spark-1.3-contributor";
        let stale = vec![json!({"encrypted_content":"OLD"})];
        let fresh = vec![json!({"encrypted_content":"NEW"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(model, &stale)),
        )];
        let mut live: FxHashMap<String, Vec<Value>> = FxHashMap::default();
        live.insert("call-1".to_string(), fresh.clone());
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(model, &messages, &live);
        // The in-memory side channel (freshest for the current turn) wins;
        // archived values must never overwrite it.
        assert_eq!(rebuilt.get("call-1"), Some(&fresh));
    }

    #[test]
    fn non_encrypted_model_is_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = "glm-5.2-opencode";
        let items = vec![json!({"encrypted_content":"ENC"})];
        // Even if history carries an encrypted marker, non encrypted-replay
        // models never rebuild (they return the live map as-is).
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(
                "muse-spark-1.3-contributor",
                &items,
            )),
        )];
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(
            model,
            &messages,
            &FxHashMap::default(),
        );
        assert!(rebuilt.is_empty());
    }

    #[test]
    fn runtime_disable_env_short_circuits_reconstruction() {
        use crate::ai::history::compress::encrypted_reasoning_replay_runtime_enabled;
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save and restore the process-level env var to avoid polluting parallel tests.
        let saved = std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY").ok();

        // SAFETY: this test sets/removes the same key serially and restores it
        // before exiting; it never asserts concurrently with other tests that
        // depend on this env var.
        unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "1") };
        assert!(!encrypted_reasoning_replay_runtime_enabled());
        let model = "muse-spark-1.3-contributor";
        let items = vec![json!({"encrypted_content":"ENC"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(model, &items)),
        )];
        // With replay disabled, no rebuild happens even if the model is capable
        // and history carries an encoded blob.
        assert!(
            reconstruct_encrypted_reasoning_items_for_model(
                model,
                &messages,
                &FxHashMap::default()
            )
            .is_empty()
        );

        unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "0") };
        assert!(encrypted_reasoning_replay_runtime_enabled());

        unsafe { std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY") };
        assert!(encrypted_reasoning_replay_runtime_enabled());

        match saved {
            Some(v) => unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", v) },
            None => unsafe { std::env::remove_var("AIOS_DISABLE_ENCRYPTED_REPLAY") },
        }
    }
}

#[cfg(test)]
mod force_off_effort_tests {
    use super::apply_thinking_force_off_effort;
    use crate::ai::provider::ApiProvider;

    const MODELHUB_ENDPOINT: &str = "https://dataagent-dev-llm.bytedance.net/v1";

    #[test]
    fn responses_model_force_off_maps_to_none() {
        // gpt-5.x (compatible provider, non-DashScope modelhub endpoint) routes to
        // NoThinkingDialect: the only thinking lever is reasoning_effort, so the force-off
        // fallback must emit "none" instead of the ladder's "low".
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::Compatible,
                "gpt-5.6-sol",
                MODELHUB_ENDPOINT,
                Some("low"),
            ),
            Some("none"),
        );
    }

    #[test]
    fn force_off_inactive_passes_effort_through() {
        // Normal requests (auto-detected no-thinking, user config, etc.) must keep the
        // configured/default effort — only the explicit override maps to "none".
        assert_eq!(
            apply_thinking_force_off_effort(
                false,
                ApiProvider::Compatible,
                "gpt-5.6-sol",
                MODELHUB_ENDPOINT,
                Some("xhigh"),
            ),
            Some("xhigh"),
        );
    }

    #[test]
    fn force_off_keeps_effort_for_switch_based_dialects() {
        // DashScope `enable_thinking:false` is the real off-switch there, so the effort is left
        // untouched (sending "none" would be an unverified field on that gateway).
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::Alibaba,
                "qwen3.7-max-alibaba",
                crate::ai::provider::ALIBABA_DEFAULT_ENDPOINT,
                Some("low"),
            ),
            Some("low"),
        );
        // DeepSeek `thinking:{"type":"disabled"}` is the real off-switch there as well.
        assert_eq!(
            apply_thinking_force_off_effort(
                true,
                ApiProvider::OpenCode,
                "deepseek-flash-opencode",
                "https://api.deepseek.com/v1",
                Some("low"),
            ),
            Some("low"),
        );
    }
}
