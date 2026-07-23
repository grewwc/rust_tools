#[cfg(test)]
use reqwest::StatusCode;

mod aux;
mod builder;
mod error;
mod logging;
mod normalize;
mod protocol;
mod reasoning;
mod routing;
mod thinking;
mod token_budget;
mod transport;
mod types;

#[cfg(test)]
use aux::{SESSION_TITLE_BODY_TIMEOUT_SECS, SESSION_TITLE_REQUEST_TIMEOUT_SECS};
pub(crate) use aux::{
    charge_llm_usage_to_kernel, charge_llm_usage_via_kernel, generate_session_title_via_model,
    summarize_history_via_model,
};
#[cfg(test)]
pub(in crate::ai) use logging::request_diagnostics_enabled;
pub(in crate::ai) use logging::emit_request_diagnostic;
#[allow(unused_imports)]
pub(crate) use error::{
    AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS, AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS,
    REQUEST_MAX_ATTEMPTS, REQUEST_MAX_ATTEMPTS_429, REQUEST_RETRY_429_MAX_MS, RequestError,
    RequestErrorKind, RequestRetryPolicy, STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
    api_key_for_request_model, apply_request_auth, clear_stale_request_interrupt_before_request,
    config_bool_is_true, config_forces_thinking, control_model_for_aux_tasks,
    endpoint_for_request_model, is_retryable_reqwest_error, is_retryable_status_with_body,
    is_retryable_stream_error, is_transient_error, parse_retry_after, request_retry_policy,
    request_retry_policy_for_current_context, retry_delay, send_with_hedged_backup,
    should_abort_retry_wait, should_retry_status, should_rotate_key,
    should_temporarily_disable_auto_selected_model, should_temporarily_disable_model,
    should_try_model_fallback, sleep_with_cancel,
};
pub(crate) use protocol::{build_http_body_for_json_messages, extract_response_text};
#[cfg(test)]
pub(crate) use reasoning::apply_aux_thinking_fields;
#[allow(unused_imports)]
pub(crate) use routing::{extract_router_content, strip_json_fence};
pub(crate) use thinking::strip_system_reminders;
#[cfg(test)]
pub(crate) use thinking::{
    latest_user_message_text, local_thinking_decision, parse_thinking_gate_output,
};
#[allow(unused_imports)]
pub(crate) use types::{
    RequestBody, StreamChoice, StreamChunk, StreamDelta, StreamFunctionCall, StreamToolCall,
    StreamUsage, merge_reasoning_fragments,
};
// 外部 re-export（build_content 被 driver 多处调用）
#[allow(unused_imports)]
pub(crate) use builder::{build_content, clamp_max_tokens_for_prompt};

// 传输层：HTTP 请求发送、重试、超时、鉴权
pub use transport::{do_request_json, do_request_text_streaming};
pub(super) use transport::{do_request_messages, do_request_messages_without_tools, print_info};

// ── 以下 private use 仅供 `tests` 子模块通过 `use super::*;` 访问 ──
// 函数体已迁移至 `transport.rs`，mod.rs 本身不再直接使用这些项。
#[cfg(test)]
#[allow(unused_imports)]
use std::time::{Duration, Instant};

#[cfg(test)]
#[allow(unused_imports)]
use reqwest::Response;

#[cfg(test)]
#[allow(unused_imports)]
use rust_tools::commonw;

#[cfg(test)]
#[allow(unused_imports)]
use serde_json::json;

#[cfg(test)]
#[allow(unused_imports)]
use super::provider::adapter_for;

#[cfg(test)]
#[allow(unused_imports)]
use super::{
    history::{Message, SessionStore, generate_session_summary},
    models,
    types::App,
};

#[cfg(test)]
#[allow(unused_imports)]
use crate::ai::theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_SUCCESS, ACCENT_WARN, RESET};

#[cfg(test)]
#[allow(unused_imports)]
use normalize::{
    agent_tools_for_request, normalize_message_content_for_text_only_model,
    normalize_messages_for_model, normalize_messages_for_request, request_tool_names_for_model,
    strip_unavailable_tool_hints_from_messages,
};

#[cfg(test)]
#[allow(unused_imports)]
use thinking::resolve_thinking;

#[cfg(test)]
#[allow(unused_imports)]
use reasoning::{
    apply_prompt_cache_breakpoint, ensure_reasoning_content_echo_for_thinking_model,
    prompt_cache_enabled_for_model, resolve_reasoning_wire_controls,
};

#[cfg(test)]
#[allow(unused_imports)]
use builder::build_request_body;

#[cfg(test)]
#[allow(unused_imports)]
use crate::ai::request_protocol::RequestProtocolDialect;

#[cfg(test)]
#[allow(unused_imports)]
use protocol::build_responses_request_body;

#[cfg(test)]
mod tests;
