//! Provider behavior adaptation layer.
//!
//! Converges the previously scattered "branch by provider" logic from
//! `request.rs` / `models.rs` / `stream/normalize.rs` / `stream/runtime.rs` /
//! `driver/reflection/background.rs` into a set of zero-state static singletons
//! (template method + override).
//!
//! The main pipeline keeps its free-function skeleton; only at the divergence
//! points does it call this module's hooks, so every provider's external wire
//! behavior (request-body serialization, streaming parse results, auth headers)
//! is byte-identical across providers.
//!
//! This module only carries the cross-provider public contract: the
//! [`ProviderAdapter`] trait and the [`adapter_for`] dispatch. Each concrete
//! provider implementation lives in its own file (`alibaba` / `compatible` /
//! `openai` / `openrouter` / `opencode`).
//!
//! The wire encoding of the thinking switch is an axis orthogonal to the
//! provider, moved into the [`thinking`] submodule ([`thinking_dialect_for`]);
//! provider adapters no longer participate in thinking-field encoding.

mod alibaba;
pub(in crate::ai) mod compatible;
mod openai;
mod opencode;
mod openrouter;
mod thinking;

use serde_json::Value;

use crate::ai::request::{ParsedStreamPayload, try_parse_stream_chunk_loose};

use super::ApiProvider;

pub(in crate::ai) use compatible::compatible_wire_shapes;
pub(in crate::ai) use thinking::{
    ThinkingOffCapability, reasoning_effort_reduces_thinking_for, thinking_dialect_for,
    thinking_off_capability_for,
};

use alibaba::AlibabaAdapter;
use compatible::CompatibleAdapter;
use openai::OpenAiAdapter;
use opencode::OpenCodeAdapter;
use openrouter::OpenRouterAdapter;

pub(in crate::ai) const COMPATIBLE_DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
pub(in crate::ai) const ALIBABA_DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
pub(in crate::ai) const OPENAI_DEFAULT_ENDPOINT: &str =
    "https://api.openai.com/v1/chat/completions";
pub(in crate::ai) const OPENCODE_DEFAULT_ENDPOINT: &str =
    "https://opencode.ai/zen/v1/chat/completions";
pub(in crate::ai) const OPENROUTER_ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Unified abstraction over per-LLM-provider behavioral differences. All
/// implementations are zero-state singletons.
///
/// Default methods implement the "OpenAI-compatible family" common behavior;
/// Alibaba / Compatible / OpenCode express their differences via overrides.
/// All provider differences are funneled through this trait to avoid scattering
/// `if provider == ...` across `request/*` / `stream/*`. When adding a provider
/// difference, prefer adding a hook here rather than a branch in the caller.
pub(crate) trait ProviderAdapter: Send + Sync {
    /// Label used for streaming-parse failure logs; also used in diagnostics.
    fn label(&self) -> &'static str;

    /// Value of the `enable_search` field on the main request body.
    /// Alibaba / Compatible pass the caller's switch through; the OpenAI-
    /// compatible family does not send the field (`None`).
    fn enable_search_field(&self, _requested: Option<bool>) -> Option<bool> {
        None
    }

    /// Top-level `reasoning_effort` field value (OpenAI / OpenRouter / OpenCode
    /// protocols).
    fn reasoning_top_level<'a>(&self, effort: Option<&'a str>) -> Option<&'a str> {
        effort
    }

    /// Nested `reasoning: { effort }` field value (DashScope compatible protocol).
    fn reasoning_nested(&self, _effort: Option<&str>) -> Option<Value> {
        None
    }

    /// This provider's default endpoint (used when the model does not declare
    /// one explicitly in the model registry).
    fn default_endpoint(&self) -> &'static str;

    /// Config-key candidate chain for reading the API key (in priority order).
    fn api_key_candidates(&self) -> &'static [&'static str];

    /// Unified request interception hook: rewrites `RequestBody` before
    /// serialization (provider-specific shape). Identity by default (zero
    /// behavior change); adapters may override to inject or rewrite fields.
    /// Triggered uniformly on all request paths by
    /// `request::protocol::build_http_body_for_request`.
    fn adapt_request(&self, request: &mut crate::ai::request::RequestBody<'_>) {
        let _ = request;
    }

    /// Collects all API keys available to this provider (including rotation
    /// candidates). Returns only the primary key by default; override to supply
    /// alternates (e.g. OpenCode's config entries). `primary_key` is the primary
    /// key resolved for the current model.
    fn collect_api_keys(&self, primary_key: &str) -> Vec<String> {
        vec![primary_key.to_string()]
    }

    /// Error message used when all API keys are exhausted.
    /// Providers may override with a more recognizable message (e.g. OpenCode's
    /// "all opencode keys exhausted").
    fn keys_exhausted_message(&self) -> &'static str {
        "request failed"
    }

    /// Whether to print a hint while waiting for the first visible chunk
    /// (OpenCode's first token is slow).
    fn shows_waiting_hint(&self) -> bool {
        false
    }

    /// Parses a single provider-specific streaming payload. Defaults to the
    /// generic loose parse and prints detailed diagnostic logs on failure;
    /// OpenCode overrides with a looser parse and shorter logs.
    fn parse_provider_chunk(&self, payload: &str) -> ParsedStreamPayload {
        match try_parse_stream_chunk_loose(payload) {
            Some(chunk) => ParsedStreamPayload::Chunk(chunk),
            None => {
                let err = serde_json::from_str::<serde_json::Value>(payload)
                    .err()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unable to parse stream payload".to_string());
                crate::ai::request::emit_request_diagnostic(format_args!(
                    "handleResponse error [{}] {err}",
                    self.label()
                ));
                crate::ai::request::emit_request_diagnostic(format_args!("======> response: "));
                crate::ai::request::emit_request_diagnostic(format_args!("{payload}"));
                crate::ai::request::emit_request_diagnostic(format_args!("<======"));
                ParsedStreamPayload::Ignore
            }
        }
    }
}

static ALIBABA: AlibabaAdapter = AlibabaAdapter;
static COMPATIBLE: CompatibleAdapter = CompatibleAdapter;
static OPENAI: OpenAiAdapter = OpenAiAdapter;
static OPENROUTER: OpenRouterAdapter = OpenRouterAdapter;
static OPENCODE: OpenCodeAdapter = OpenCodeAdapter;

pub(in crate::ai) fn alibaba_adapter() -> &'static dyn ProviderAdapter {
    &ALIBABA
}
pub(in crate::ai) fn compatible_adapter() -> &'static dyn ProviderAdapter {
    &COMPATIBLE
}
pub(in crate::ai) fn openai_adapter() -> &'static dyn ProviderAdapter {
    &OPENAI
}
pub(in crate::ai) fn openrouter_adapter() -> &'static dyn ProviderAdapter {
    &OPENROUTER
}
pub(in crate::ai) fn opencode_adapter() -> &'static dyn ProviderAdapter {
    &OPENCODE
}

/// Selects the adapter matching the provider and endpoint.
///
/// OpenRouter is not an independent [`ApiProvider`] variant but an endpoint
/// variant of the OpenAI protocol (endpoint contains `openrouter.ai`); its
/// streaming parse matches OpenAI and only the log label differs.
pub(in crate::ai) fn adapter_for(
    provider: ApiProvider,
    endpoint: &str,
) -> &'static dyn ProviderAdapter {
    if endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("openrouter.ai")
    {
        return openrouter_adapter();
    }
    match provider {
        ApiProvider::Alibaba => alibaba_adapter(),
        ApiProvider::Compatible => compatible_adapter(),
        ApiProvider::OpenAi => openai_adapter(),
        ApiProvider::OpenCode => opencode_adapter(),
    }
}
