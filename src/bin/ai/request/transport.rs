//! HTTP transport layer: request sending, retries, timeouts, authentication.
//!
//! Extracted from `request/mod.rs`; its sole responsibility is "send the request and get the response",
//! kept separate from request-construction logic such as message normalization, tool schema building, and thinking dialect.

use std::time::{Duration, Instant};

use reqwest::Response;
use rust_tools::commonw;
use serde_json::Value;

use super::super::{
    history::{Message, SessionStore, generate_session_summary},
    models,
    provider::adapter_for,
    types::App,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_SUCCESS, ACCENT_WARN, RESET};

use super::aux::charge_llm_usage_to_kernel;
use super::builder::build_request_body;
use super::error::{
    REQUEST_MAX_ATTEMPTS, RequestError, RequestErrorKind, RequestRetryPolicy,
    STREAM_RESPONSE_HEADER_TIMEOUT_SECS, api_key_for_request_model, apply_request_auth,
    clear_stale_request_interrupt_before_request, config_forces_thinking,
    endpoint_for_request_model, is_retryable_reqwest_error, is_retryable_status_with_body,
    parse_retry_after, request_retry_policy_for_current_context, retry_delay, retry_delay_429,
    should_retry_status, should_rotate_key, sleep_with_cancel,
};
use super::normalize::{
    agent_tools_for_request, fold_resolved_tool_failures, normalize_messages_for_model,
    request_tool_names_for_model, strip_unavailable_tool_hints_from_messages,
};
use super::reasoning::{
    apply_prompt_cache_breakpoint, apply_thinking_force_off_effort,
    normalize_reasoning_content_replay_for_model,
    prompt_cache_enabled_for_model, reconstruct_encrypted_reasoning_items_for_model,
    resolve_reasoning_effort,
};
use super::thinking::resolve_thinking;
use super::token_budget;
use super::types::{RequestBody, StreamChunk, StreamUsage};

/// Concurrent requests (foreground turn + subagents) retry independently, and their `attempt N/M` counters
/// interleave with no way to tell them apart. An aios scheduler pid is used as a scope tag binding each retry log to
/// its process; without a pid (no TASK_PID scope) it returns an empty string and the log degrades to plain text.
fn retry_scope_tag() -> String {
    match aios_kernel::kernel::current_task_pid() {
        Some(pid) => format!("[pid {pid}] "),
        None => String::new(),
    }
}

async fn wait_for_app_request_interrupt(app: &App) {
    crate::ai::driver::signal::wait_for_interrupt_sources(
        None,
        None,
        Some(app.cancel_stream.as_ref()),
    )
    .await
}

pub(super) async fn response_text_with_cancel(
    app: &App,
    response: Response,
    cancel_reason: &'static str,
) -> Result<String, RequestError> {
    tokio::select! {
        body = response.text() => Ok(body.unwrap_or_default()),
        _ = wait_for_app_request_interrupt(app) => Err(RequestError::cancelled(cancel_reason)),
    }
}

async fn response_json_with_cancel(
    app: &App,
    response: Response,
    cancel_reason: &'static str,
) -> Result<Value, RequestError> {
    tokio::select! {
        body = response.json::<Value>() => body.map_err(RequestError::network),
        _ = wait_for_app_request_interrupt(app) => Err(RequestError::cancelled(cancel_reason)),
    }
}

fn maybe_emit_responses_reasoning_replay_diagnostic(
    model: &str,
    endpoint: &str,
    messages: &[Message],
    reasoning_items: &rustc_hash::FxHashMap<String, Vec<Value>>,
) {
    if !models::reasoning_encrypted_replay_enabled(model)
        || models::request_protocol_dialect(model, endpoint)
            != crate::ai::request_protocol::RequestProtocolDialect::Responses
    {
        return;
    }
    // Encrypted reasoning items are replayed through an in-memory side channel only within the current turn's tool chain; earlier turns
    // can no longer be replayed faithfully, and history checkpoints / tool evidence carry the semantic fidelity. Diagnostics only count
    // the current turn after the latest user message, so every request does not emit noise about old history.
    // Synthetic user messages (evidence handoff, image followup) do not form a turn boundary.
    let current_turn_start =
        crate::ai::history::last_real_user_index(messages).unwrap_or(0);
    let stats = super::protocol::responses_reasoning_replay_stats(
        &messages[current_turn_start..],
        Some(reasoning_items),
    );
    if stats.tool_call_groups == 0 || stats.missing_groups == 0 {
        return;
    }
    // super::emit_request_diagnostic(format_args!(
    //     "[Info] responses reasoning replay: replayed {}/{} assistant tool-call groups; {} group(s) missing encrypted reasoning items, falling back to function_call replay.",
    //     stats.replayed_groups,
    //     stats.tool_call_groups,
    //     stats.missing_groups
    // ));
}

/// Hedged send with TPM preflight.
///
/// The budget must be counted per actual physical send, not per logical request: when no hedged backup
/// fires, only one slot is taken; when the tail triggers a backup, every additional request re-reserves before sending.
/// This avoids 429s while preventing the one-shot `hedged_max_sends` reservation from throttling normal throughput by 3x.
async fn send_with_budgeted_hedged_backup(
    app: &App,
    model: &str,
    endpoint: &str,
    request_model_label: &str,
    api_key: &str,
    estimated_prompt_tokens: usize,
    build_request: impl Fn() -> reqwest::RequestBuilder,
    backup_after_secs: u64,
    max_sends: usize,
) -> Result<Response, RequestError> {
    use futures_util::stream::{FuturesUnordered, StreamExt};

    let max_sends = max_sends.max(1);
    let hedge = Duration::from_secs(backup_after_secs);
    let mut in_flight = FuturesUnordered::new();

    // Track the most recent retryable HTTP response / network failure: a single failure must not short-circuit other in-flight requests;
    // return only when a non-retryable response arrives or all requests have finished.
    let mut last_retryable_response: Option<Response> = None;
    let mut last_err: Option<RequestError> = None;
    for round in 1..=max_sends {
        token_budget::wait_for_request_budget(
            app,
            model,
            endpoint,
            request_model_label,
            api_key,
            estimated_prompt_tokens,
            1,
        )
        .await?;
        in_flight.push(build_request().send());
        if round == max_sends {
            break;
        }
        tokio::select! {
            result = in_flight.next() => {
                match result.expect("in_flight 非空") {
                    Ok(resp) if should_retry_status(resp.status()) => {
                        last_retryable_response = Some(resp);
                    }
                    Ok(resp) => return Ok(resp),
                    Err(e) => last_err = Some(RequestError::network(e)),
                }
            }
            _ = tokio::time::sleep(hedge) => {
                super::emit_request_diagnostic(format_args!(
                    "[Info] 第 {round} 次请求 {}s 内未返回响应头，发起 hedged backup request",
                    backup_after_secs
                ));
            }
            _ = crate::ai::driver::signal::wait_for_interrupt_sources(
                None,
                None,
                Some(app.cancel_stream.as_ref()),
            ) => {
                return Err(RequestError::cancelled(
                    "request canceled by user while waiting for response headers",
                ));
            }
        }
    }

    // All hedged requests are dispatched; keep waiting on in-flight ones. Retryable HTTP statuses (429/5xx etc.)
    // and network errors are kept only as candidate failures, so they cannot outrun requests that may still succeed.
    while !in_flight.is_empty() {
        let result = tokio::select! {
            result = in_flight.next() => {
                let Some(result) = result else {
                    break;
                };
                result
            }
            _ = crate::ai::driver::signal::wait_for_interrupt_sources(
                None,
                None,
                Some(app.cancel_stream.as_ref()),
            ) => {
                return Err(RequestError::cancelled(
                    "request canceled by user while waiting for response headers",
                ));
            }
        };
        match result {
            Ok(resp) if should_retry_status(resp.status()) => {
                last_retryable_response = Some(resp);
            }
            Ok(resp) => return Ok(resp),
            Err(e) => last_err = Some(RequestError::network(e)),
        }
    }
    if let Some(resp) = last_retryable_response {
        return Ok(resp);
    }
    Err(last_err.unwrap_or_else(|| RequestError {
        kind: RequestErrorKind::Network,
        message: "hedged request set unexpectedly empty".to_string(),
        retry_after: None,
    }))
}

/// Try a single API key for `do_request_messages`, with retry logic for
/// header timeout, network errors, and retryable server statuses (5xx / 400+upstream).
///
/// 429 (quota/rate limit) is **not** retried with backoff inside this function: it returns directly with the (clamped) `retry_after`,
/// letting the upper `do_request_messages` rotate other keys first before deciding whether to back off and retry.
/// Returns `Ok` on success or `Err` after exhausting per-key retries.
async fn request_messages_with_key(
    app: &mut App,
    model: &str,
    api_key: &str,
    request_body: &mut RequestBody<'_>,
    retry_policy: &RequestRetryPolicy,
    endpoint: &str,
) -> Result<Response, RequestError> {
    let http_body = super::protocol::build_http_body_for_request(model, endpoint, request_body);
    // Reuse the character estimate already computed inside build_request_body (same RequestBody),
    // avoiding another full traversal of the same history plus re-serializing tool schemas.
    let estimated_prompt_tokens = token_budget::calibrate_prompt_tokens_for_budget(
        request_body.estimated_prompt_tokens,
        app.last_known_prompt_tokens,
        app.last_known_cached_prompt_tokens,
    );
    for attempt in 1..=retry_policy.max_attempts {
        let client = app.client.clone();
        let build_request = || {
            apply_request_auth(client.post(endpoint), endpoint, api_key)
                .header("Content-Type", "application/json")
                .body(http_body.clone())
        };
        let response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_budgeted_hedged_backup(
                app,
                model,
                endpoint,
                &request_body.model,
                api_key,
                estimated_prompt_tokens,
                build_request,
                retry_policy.hedged_backup_after_secs(),
                retry_policy.hedged_max_sends(),
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                if attempt < retry_policy.max_attempts {
                    let delay = retry_delay(attempt);
                    super::emit_request_diagnostic(format_args!(
                        "[Warning] {}等待响应头超时 ({}s) - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        retry_policy.header_timeout_secs,
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    ));
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(RequestError {
                    kind: RequestErrorKind::Network,
                    message: format!(
                        "request timed out waiting for response headers after {} attempts",
                        retry_policy.max_attempts
                    ),
                    retry_after: None,
                });
            }
        };

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }
                let status = response.status();
                let status_code = status.as_u16();
                let retry_after_delay = if status_code == 429 {
                    parse_retry_after(response.headers())
                } else {
                    None
                };
                let body = response_text_with_cancel(
                    app,
                    response,
                    "request canceled by user while reading error response body",
                )
                .await?;
                let mut err = RequestError::status(status, body);

                // 429 (quota/rate limit) is not backed off within a single key: return immediately with the clamped retry_after,
                // let the upper layer rotate other keys first, and only then decide on backoff retry once keys are exhausted.
                if status_code == 429 {
                    err.retry_after = retry_after_delay;
                    return Err(err);
                }

                if is_retryable_status_with_body(status, &err.message)
                    && attempt < retry_policy.max_attempts
                {
                    let delay = retry_delay(attempt);
                    super::emit_request_diagnostic(format_args!(
                        "[Warning] {}{} - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        status,
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    ));
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(err);
            }
            Err(err) => {
                let retryable = match &err.kind {
                    RequestErrorKind::Network => true,
                    _ => false,
                };
                if retryable && attempt < retry_policy.max_attempts {
                    let delay = retry_delay(attempt);
                    super::emit_request_diagnostic(format_args!(
                        "[Warning] {}网络错误 - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    ));
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(err);
            }
        }
    }
    unreachable!("retry loop always returns or breaks")
}

#[commonw::debug_measure_time("do_request_message")]
pub(crate) async fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    do_request_messages_with_tool_mode(app, model, messages, stream, true).await
}

/// Send a request that exposes no tools.
///
/// Used for the wrap-up round after the tool loop or iteration cap. The `tools` field must be removed at the wire request layer here,
/// rather than relying only on the prompt to ask the model to stop calling tools.
pub(crate) async fn do_request_messages_without_tools(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    do_request_messages_with_tool_mode(app, model, messages, stream, false).await
}

async fn do_request_messages_with_tool_mode(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
    tools_enabled: bool,
) -> Result<Response, RequestError> {
    clear_stale_request_interrupt_before_request(app);

    let mut normalized_messages = normalize_messages_for_model(model, messages);
    if let Ok(outcomes) =
        crate::ai::history::read_tool_execution_outcomes_sqlite(&app.session_history_file)
    {
        fold_resolved_tool_failures(&mut normalized_messages, &outcomes);
    }
    let request_tool_names = tools_enabled
        .then(|| request_tool_names_for_model(app, model))
        .unwrap_or_default();
    strip_unavailable_tool_hints_from_messages(&mut normalized_messages, &request_tool_names);
    if prompt_cache_enabled_for_model(model) {
        apply_prompt_cache_breakpoint(&mut normalized_messages);
    }
    let (tools_value, tool_choice) = if tools_enabled {
        agent_tools_for_request(app, model)
    } else {
        (None, None)
    };
    let thinking_start = Instant::now();
    let force_thinking_requested = config_forces_thinking();
    let enable_thinking = resolve_thinking(app, model, &normalized_messages).await;
    crate::ai::agent_hang_debug!(
        "pre-fix",
        "G",
        "request::do_request_messages:resolve_thinking:end",
        "[DEBUG] resolve thinking finished",
        {
            "enable_thinking": enable_thinking,
            "elapsed_ms": thinking_start.elapsed().as_secs_f64() * 1000.0,
        },
    );
    if force_thinking_requested && !enable_thinking {
        super::emit_request_diagnostic(format_args!(
            "[Info] thinking 已请求，但当前模型 `{}` 不支持 thinking；本轮将继续以普通模式输出。",
            model
        ));
    }
    // Encrypted reasoning side-channel rebuild: must run before `normalize_reasoning_content_replay_for_model`,
    // because the latter strips the `reasoning_content` (our persisted encoded blob) for encrypted-replay models.
    // The in-memory side channel (freshest for the current turn) wins; the persisted blob only fills historical gaps. The field is then stripped from the wire projection as usual.
    let turn_reasoning_items = reconstruct_encrypted_reasoning_items_for_model(
        model,
        &normalized_messages,
        &app.turn_reasoning_items,
    );
    // The final wire projection decides reasoning_content replay semantics by model capability: GLM keeps the
    // tool-call assistant's original text, DeepSeek only fills the empty field shape, and all other models strip it.
    // Unlike this turn's enable_thinking gate, this must be unified before every request.
    normalize_reasoning_content_replay_for_model(model, &mut normalized_messages);
    let reasoning_effort = resolve_reasoning_effort(app, model).map(|e| e.as_str());
    // Some gateways (e.g. bytedance modelhub) reject /v1/chat/completions requests that carry
    // `tools` + `reasoning_effort` together (returning 400). When the model declares
    // `reasoning_effort_conflicts_with_tools` and this turn's request carries tools,
    // reasoning_effort is omitted automatically to avoid the 400; requests without tools keep it to preserve thinking.
    let reasoning_effort = if reasoning_effort.is_some()
        && tools_value.is_some()
        && models::reasoning_effort_conflicts_with_tools(model)
    {
        None
    } else {
        reasoning_effort
    };
    let endpoint = endpoint_for_request_model(app, model);
    // Truncation-ladder force-off fallback (orchestrator.rs): after repeated truncation the
    // ladder sets thinking_disabled_override to force thinking off. For effort-only dialects
    // (OpenAI family / Responses) resolve_thinking=false emits nothing on the wire
    // (NoThinkingDialect sends no thinking field) and the ladder's Low effort keeps thinking on,
    // making the fallback a no-op; map it to `reasoning_effort: "none"`, the only value that
    // actually disables thinking there. Switch-based dialects keep their effort (their off-switch
    // is already driven by enable_thinking=false).
    let reasoning_effort = apply_thinking_force_off_effort(
        app.cli.thinking_disabled_override,
        models::model_adapter(model),
        model,
        &endpoint,
        reasoning_effort,
    );
    maybe_emit_responses_reasoning_replay_diagnostic(
        model,
        &endpoint,
        &normalized_messages,
        &turn_reasoning_items,
    );
    let mut request_body = build_request_body(
        model,
        &normalized_messages,
        stream,
        enable_thinking,
        models::search_enabled(model).then_some(true),
        tools_value,
        tool_choice,
        reasoning_effort,
        app.cli.max_tokens_override,
        app.last_known_prompt_tokens,
        Some(&turn_reasoning_items),
    );
    let retry_policy = request_retry_policy_for_current_context();

    // --- Key rotation + 429 backoff ---
    // Each round first tries all keys in rotation; only when every key failed with 429 (quota/rate limit)
    // does the whole round retry with backoff (up to max_attempts_429 rounds). Other rotatable errors (401/403)
    // return directly if they still fail after key rotation; retrying is pointless.
    let primary_key = api_key_for_request_model(app, model);
    let adapter = adapter_for(models::model_adapter(model), &endpoint);
    let keys_to_try = adapter.collect_api_keys(&primary_key);
    let total_keys = keys_to_try.len();

    let mut last_key_err: Option<RequestError> = None;
    for attempt in 1..=retry_policy.max_attempts_429 {
        let mut all_rate_limited = true;
        let mut round_retry_after: Option<Duration> = None;
        for (key_idx, api_key) in keys_to_try.iter().enumerate() {
            if key_idx > 0 {
                super::emit_request_diagnostic(format_args!(
                    "[{}] key #{} failed, trying next key #{} ({} remaining)",
                    adapter.label(),
                    key_idx - 1,
                    key_idx,
                    total_keys - key_idx
                ));
            }
            match request_messages_with_key(
                app,
                model,
                api_key,
                &mut request_body,
                &retry_policy,
                &endpoint,
            )
            .await
            {
                Ok(response) => return Ok(response),
                Err(err) if should_rotate_key(&err) => {
                    if err.is_rate_limited() {
                        round_retry_after = err.retry_after.or(round_retry_after);
                    } else {
                        all_rate_limited = false;
                    }
                    last_key_err = Some(err);
                }
                Err(err) => return Err(err),
            }
        }

        // All keys failed. Retry the whole round with backoff only if every failure was 429 rate limiting and retries remain.
        if !all_rate_limited || attempt >= retry_policy.max_attempts_429 {
            break;
        }
        let delay = round_retry_after.unwrap_or_else(|| retry_delay_429(attempt));
        super::emit_request_diagnostic(format_args!(
            "[Warning] {}429 Too Many Requests - {} 个 key 均配额超限，sleep {} 秒后重试 (attempt {}/{})",
            retry_scope_tag(),
            total_keys,
            delay.as_secs_f32(),
            attempt,
            retry_policy.max_attempts_429
        ));
        if sleep_with_cancel(app, delay).await {
            return Err(RequestError::cancelled(
                "request canceled by user during retry wait",
            ));
        }
    }
    Err(last_key_err.unwrap_or_else(|| RequestError {
        kind: RequestErrorKind::Network,
        message: adapter.keys_exhausted_message().to_string(),
        retry_after: None,
    }))
}

pub(crate) fn print_info(app: &App, model: &str) {
    let search = if models::search_enabled(model) {
        "true"
    } else {
        "false"
    };
    let effort_label = if app.cli.thinking_disabled_override {
        // The truncation fallback already forced thinking off (the last resort when lowering effort is ineffective for
        // always-thinking models); label it explicitly to avoid confusion with "auto / model default".
        "off"
    } else {
        match resolve_reasoning_effort(app, model) {
            Some(e) => e.as_str(),
            None => "auto",
        }
    };

    // Print the current session summary
    let store = SessionStore::new(&app.config.history_file);
    let summary = store
        .read_session_title(&app.session_id)
        .ok()
        .flatten()
        .map(|title| crate::ai::history::normalize_generated_session_title(&title))
        .filter(|title| !title.is_empty())
        .or_else(|| {
            store
                .first_user_prompt(&app.session_id)
                .ok()
                .flatten()
                .map(|p| generate_session_summary(&p))
                .map(|summary| crate::ai::history::normalize_generated_session_title(&summary))
                .filter(|summary| !summary.is_empty())
        });
    let session_part = summary
        .filter(|s| !s.is_empty())
        .map(|s| format!("{ACCENT_MUTED} · {ACCENT_WARN}{}{RESET}", s))
        .unwrap_or_default();

    // Use println! to avoid manual-flush permission issues; model and session are merged into one line.
    println!(
        "{ACCENT_MUTED}[{ACCENT_SUCCESS}{}{ACCENT_MUTED} (search: {ACCENT_WARN}{search}{ACCENT_MUTED}, effort: {ACCENT_PRIMARY}{effort_label}{ACCENT_MUTED}){session_part}{ACCENT_MUTED}]{RESET}",
        models::model_display_label(model),
    );
}

/// Make an LLM request expecting a JSON response (used for intent recognition, knowledge-base Q&A, etc.).
///
/// `skip_reasoning_effort`: when `true`, force-omit the `reasoning_effort` field;
/// the field is ignored even when the model configures it by default. Suited to lightweight workloads such as knowledge-base Q&A.
pub async fn do_request_json(
    app: &App,
    model: &str,
    messages: &[serde_json::Value],
    stream: bool,
    skip_reasoning_effort: bool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    clear_stale_request_interrupt_before_request(app);

    let endpoint = endpoint_for_request_model(app, model);
    let request_model = models::request_model_name(model);
    let resolved_reasoning_effort = (!skip_reasoning_effort)
        .then(|| resolve_reasoning_effort(app, model).map(|effort| effort.as_str()))
        .flatten();
    // Auxiliary requests disable thinking unconditionally, but the final wire fields still follow the model/endpoint protocol dialect for the
    // HTTP body (chat-completions: messages; responses: input).
    let request_body = super::protocol::build_http_body_for_json_messages(
        model,
        &endpoint,
        messages,
        stream,
        resolved_reasoning_effort,
        stream,
    );

    // Auxiliary requests must also reuse the main request's provider key rotation. Paths like `a -n` come through here;
    // taking only the first result would skip `opencode.api_key_*` when the generic `api_key` fails.
    let primary_key = api_key_for_request_model(app, model);
    let adapter = adapter_for(models::model_adapter(model), &endpoint);
    let keys_to_try = adapter.collect_api_keys(&primary_key);
    let total_keys = keys_to_try.len();
    let mut key_idx = 0usize;
    let mut attempt = 1usize;

    loop {
        let api_key = &keys_to_try[key_idx];
        let t0 = Instant::now();
        token_budget::wait_for_request_budget(
            app,
            model,
            &endpoint,
            &request_model,
            api_key,
            token_budget::calibrate_prompt_tokens_for_budget(
                token_budget::estimate_serialized_request_tokens(&request_body),
                app.last_known_prompt_tokens,
                app.last_known_cached_prompt_tokens,
            ),
            1,
        )
        .await
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
        // Non-streaming auxiliary request: 60s timeout per attempt
        let send_future = async {
            let resp = apply_request_auth(app.client.post(&endpoint), &endpoint, api_key)
                .header("Content-Type", "application/json")
                .body(request_body.clone())
                .send()
                .await?;
            Ok::<_, reqwest::Error>(resp)
        };
        enum AuxSendWait<T> {
            Ready(T),
            TimedOut,
            Cancelled,
        }
        let response = match tokio::select! {
            result = send_future => AuxSendWait::Ready(result),
            _ = wait_for_app_request_interrupt(app) => AuxSendWait::Cancelled,
            _ = tokio::time::sleep(Duration::from_secs(60)) => AuxSendWait::TimedOut,
        } {
            AuxSendWait::Ready(result) => result,
            AuxSendWait::Cancelled => {
                return Err(Box::new(RequestError::cancelled(
                    "request canceled by user while waiting for auxiliary response headers",
                )));
            }
            AuxSendWait::TimedOut => {
                if attempt < REQUEST_MAX_ATTEMPTS {
                    super::emit_request_diagnostic(format_args!(
                        "[Warning] {}do_request_json timeout (60s), retrying (attempt {}/{})",
                        retry_scope_tag(),
                        attempt,
                        REQUEST_MAX_ATTEMPTS
                    ));
                    attempt += 1;
                    continue;
                }
                return Err("do_request_json: all attempts timed out".into());
            }
        };

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    let json = match response_json_with_cancel(
                        app,
                        response,
                        "request canceled by user while reading auxiliary response body",
                    )
                    .await
                    {
                        Ok(json) => json,
                        Err(err) => return Err(Box::new(err)),
                    };
                    // AIOS: bridge non-stream usage to kernel `/dev/llm`.
                    if let Some(usage_val) = json.get("usage") {
                        if let Ok(usage) = serde_json::from_value::<StreamUsage>(usage_val.clone())
                        {
                            let usage = usage.normalized();
                            let latency_ms = t0.elapsed().as_millis().min(u64::MAX as u128) as u64;
                            let _ = charge_llm_usage_to_kernel(app, model, &usage, latency_ms);
                        }
                    }
                    return Ok(json);
                }
                let status = response.status();
                let status_code = status.as_u16();
                // Read the Retry-After header before consuming the body
                let retry_after_delay = if status_code == 429 {
                    parse_retry_after(response.headers())
                } else {
                    None
                };
                let body = match response_text_with_cancel(
                    app,
                    response,
                    "request canceled by user while reading auxiliary error response body",
                )
                .await
                {
                    Ok(body) => body,
                    Err(err) => return Err(Box::new(err)),
                };
                let err = RequestError::status(status, body);
                if should_rotate_key(&err) && key_idx + 1 < total_keys {
                    let next_key_idx = key_idx + 1;
                    super::emit_request_diagnostic(format_args!(
                        "[{}] key #{} failed, trying next key #{} ({} remaining)",
                        adapter.label(),
                        key_idx,
                        next_key_idx,
                        total_keys - next_key_idx
                    ));
                    key_idx = next_key_idx;
                    continue;
                }
                if should_retry_status(status) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_after_delay.unwrap_or_else(|| retry_delay(attempt));
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    attempt += 1;
                    if status_code == 429 {
                        key_idx = 0;
                    }
                    continue;
                }
                return Err(err.into());
            }
            Err(err) => {
                if is_retryable_reqwest_error(&err) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    attempt += 1;
                    continue;
                }
                return Err(err.into());
            }
        }
    }
}

/// Streaming aggregation request: issues a `stream: true` request, accumulates `delta.content` chunk by chunk,
/// and finally returns the concatenated full text.
///
/// Compared with non-streaming [`do_request_json`], the streaming path returns response headers immediately and data arrives
/// incrementally per chunk, so it **cannot** be blown up by waiting for the server to finish the whole body.
/// The guard here is a per-chunk idle timeout: keep reading as long as data keeps arriving; only `STREAM_RESPONSE_HEADER_TIMEOUT_SECS`
/// consecutive seconds without any chunk count as a hang.
///
/// Suited to auxiliary tasks like knowledge curation that only need the final complete JSON, without live terminal rendering.
pub(super) fn apply_aux_stream_payload(
    payload: &str,
    event_type: Option<&str>,
    content: &mut String,
    pending_usage: &mut Option<(String, StreamUsage)>,
) {
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return;
    }
    if let Ok(value) = serde_json::from_str::<Value>(payload) {
        let event_type = event_type
            .or_else(|| value.get("type").and_then(Value::as_str))
            .map(str::trim)
            .filter(|s| !s.is_empty());
        if let Some(event_type) = event_type
            && apply_responses_stream_event(event_type, &value, content, pending_usage)
        {
            return;
        }
    }
    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(payload) {
        // Capture usage: OpenAI-compatible streaming puts the final usage in a trailing chunk with empty choices,
        // so it must be taken before the choice is read, or it would be missed.
        if let Some(usage) = chunk.usage {
            *pending_usage = Some((chunk.model.clone(), usage.normalized()));
        }
        if let Some(choice) = chunk.choices.into_iter().next() {
            content.push_str(&choice.delta.content);
        }
    }
}

fn apply_responses_stream_event(
    event_type: &str,
    value: &Value,
    content: &mut String,
    pending_usage: &mut Option<(String, StreamUsage)>,
) -> bool {
    let event_type = event_type.to_ascii_lowercase();
    if event_type == "response.completed" {
        let response = value.get("response").unwrap_or(value);
        if let Some(usage_val) = response.get("usage")
            && let Ok(usage) = serde_json::from_value::<StreamUsage>(usage_val.clone())
        {
            let model = response
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            *pending_usage = Some((model, usage.normalized()));
        }
        if content.is_empty()
            && let Some(text) = super::protocol::extract_response_text(response)
                .or_else(|| super::protocol::extract_response_text(value))
        {
            content.push_str(&text);
        }
        return true;
    }

    if event_type.contains("output_text") || event_type.contains("content") {
        if event_type.ends_with(".delta") {
            if let Some(delta) = value
                .get("delta")
                .or_else(|| value.get("text"))
                .or_else(|| value.get("content"))
                .and_then(Value::as_str)
            {
                content.push_str(delta);
            }
            return true;
        }
        if event_type.ends_with(".done") {
            if content.is_empty()
                && let Some(text) = value
                    .get("text")
                    .or_else(|| value.get("content"))
                    .and_then(Value::as_str)
            {
                content.push_str(text);
            }
            return true;
        }
    }

    if event_type.contains("refusal") {
        if event_type.ends_with(".delta") {
            if let Some(delta) = value
                .get("delta")
                .or_else(|| value.get("refusal"))
                .and_then(Value::as_str)
            {
                content.push_str(delta);
            }
            return true;
        }
        if event_type.ends_with(".done") {
            if content.is_empty()
                && let Some(text) = value
                    .get("refusal")
                    .or_else(|| value.get("text"))
                    .and_then(Value::as_str)
            {
                content.push_str(text);
            }
            return true;
        }
    }

    false
}

pub async fn do_request_text_streaming(
    app: &App,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String, Box<dyn std::error::Error>> {
    clear_stale_request_interrupt_before_request(app);

    let endpoint = endpoint_for_request_model(app, model);
    let request_model = models::request_model_name(model);
    let request_body = super::protocol::build_http_body_for_json_messages(
        model, &endpoint, messages, true, None, true,
    );

    let primary_key = api_key_for_request_model(app, model);
    let adapter = adapter_for(models::model_adapter(model), &endpoint);
    let keys_to_try = adapter.collect_api_keys(&primary_key);
    let total_keys = keys_to_try.len();
    let mut key_idx = 0usize;
    let mut attempt = 1usize;

    loop {
        let api_key = &keys_to_try[key_idx];
        let retry_policy = request_retry_policy_for_current_context();
        let estimated_prompt_tokens = token_budget::estimate_serialized_request_tokens(&request_body);
        let client = app.client.clone();
        let build_request = || {
            apply_request_auth(client.post(&endpoint), &endpoint, api_key)
                .header("Content-Type", "application/json")
                .body(request_body.clone())
        };
        // Wait for response headers: handshake + server starting to respond. This step is fast when streaming.
        // Use a hedged backup: if the primary produces no response within a short window, dispatch a backup request automatically.
        // Consistent with the non-streaming path, the header wait timeout comes from retry_policy.header_timeout_secs
        // (auto subagents use 30s instead of the hardcoded 90s); the chunk idle timeout still uses the fixed constant.
        let mut response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_budgeted_hedged_backup(
                app,
                model,
                &endpoint,
                &request_model,
                api_key,
                estimated_prompt_tokens,
                build_request,
                retry_policy.hedged_backup_after_secs(),
                retry_policy.hedged_max_sends(),
            ),
        )
        .await
        {
            Ok(Ok(resp)) => {
                if resp.status().is_success() {
                    resp
                } else {
                    let status = resp.status();
                    let body = match response_text_with_cancel(
                        app,
                        resp,
                        "request canceled by user while reading streaming error response body",
                    )
                    .await
                    {
                        Ok(body) => body,
                        Err(err) => return Err(Box::new(err)),
                    };
                    let err = RequestError::status(status, body);
                    if should_rotate_key(&err) && key_idx + 1 < total_keys {
                        let next_key_idx = key_idx + 1;
                        super::emit_request_diagnostic(format_args!(
                            "[{}] key #{} failed, trying next key #{} ({} remaining)",
                            adapter.label(),
                            key_idx,
                            next_key_idx,
                            total_keys - next_key_idx
                        ));
                        key_idx = next_key_idx;
                        continue;
                    }
                    if should_retry_status(status) && attempt < REQUEST_MAX_ATTEMPTS {
                        let delay = retry_delay(attempt);
                        if sleep_with_cancel(app, delay).await {
                            return Err(Box::new(RequestError::cancelled(
                                "request canceled by user during retry wait",
                            )));
                        }
                        attempt += 1;
                        if status.as_u16() == 429 {
                            key_idx = 0;
                        }
                        continue;
                    }
                    return Err(err.into());
                }
            }
            Ok(Err(err)) => {
                let retryable = matches!(&err.kind, RequestErrorKind::Network);
                if retryable && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    attempt += 1;
                    continue;
                }
                return Err(Box::new(err));
            }
            Err(_) => {
                if attempt < REQUEST_MAX_ATTEMPTS {
                    super::emit_request_diagnostic(format_args!(
                        "[Warning] {}do_request_text_streaming 等待响应头超时 ({}s), retrying (attempt {}/{})",
                        retry_scope_tag(),
                        retry_policy.header_timeout_secs,
                        attempt,
                        REQUEST_MAX_ATTEMPTS
                    ));
                    attempt += 1;
                    continue;
                }
                return Err("do_request_text_streaming: all attempts timed out".into());
            }
        };

        // Read and aggregate delta.content chunk by chunk.
        let mut content = String::new();
        let mut buffer: Vec<u8> = Vec::new();
        let mut sse_event_data = String::new();
        let mut sse_event_type: Option<String> = None;
        let mut idle_timed_out = false;
        // Usage carried by the final chunk (OpenAI-compatible streaming: usually returned in a trailing chunk with empty choices).
        let mut pending_usage: Option<(String, StreamUsage)> = None;
        let t0 = std::time::Instant::now();
        loop {
            enum AuxChunkWait<T> {
                Ready(T),
                TimedOut,
                Cancelled,
            }
            let chunk = match tokio::select! {
                result = response.chunk() => AuxChunkWait::Ready(result),
                _ = wait_for_app_request_interrupt(app) => AuxChunkWait::Cancelled,
                _ = tokio::time::sleep(Duration::from_secs(STREAM_RESPONSE_HEADER_TIMEOUT_SECS)) => {
                    AuxChunkWait::TimedOut
                }
            } {
                AuxChunkWait::Ready(Ok(Some(bytes))) => bytes,
                AuxChunkWait::Ready(Ok(None)) => break, // stream ended normally
                AuxChunkWait::Ready(Err(_)) => break,   // read error: keep what was aggregated so far
                AuxChunkWait::TimedOut => {
                    idle_timed_out = true;
                    break;
                }
                AuxChunkWait::Cancelled => {
                    return Err(Box::new(RequestError::cancelled(
                        "request canceled by user while reading auxiliary streaming response body",
                    )));
                }
            };
            buffer.extend_from_slice(&chunk);
            // Aggregate `data:` lines by SSE event boundaries, compatible with standard multi-line payloads.
            while let Some(pos) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    apply_aux_stream_payload(
                        &sse_event_data,
                        sse_event_type.as_deref(),
                        &mut content,
                        &mut pending_usage,
                    );
                    sse_event_data.clear();
                    sse_event_type = None;
                    continue;
                }
                if trimmed.starts_with(':') {
                    continue;
                }
                if let Some(event_type) = trimmed.strip_prefix("event:") {
                    sse_event_type = Some(event_type.trim_start().to_string());
                    continue;
                }
                let Some(payload) = trimmed.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if !sse_event_data.is_empty() {
                    sse_event_data.push('\n');
                }
                sse_event_data.push_str(payload);
            }
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if let Some(event_type) = trimmed.strip_prefix("event:") {
                sse_event_type = Some(event_type.trim_start().to_string());
            } else if let Some(payload) = trimmed.strip_prefix("data:") {
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if !sse_event_data.is_empty() {
                    sse_event_data.push('\n');
                }
                sse_event_data.push_str(payload);
            }
        }
        apply_aux_stream_payload(
            &sse_event_data,
            sse_event_type.as_deref(),
            &mut content,
            &mut pending_usage,
        );

        // AIOS: charge this streaming auxiliary request's usage to the kernel `/dev/llm`, same as the main path.
        if let Some((echoed_model, usage)) = pending_usage {
            let model_for_pricing = if echoed_model.is_empty() {
                model
            } else {
                echoed_model.as_str()
            };
            let latency_ms = t0.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let _ = charge_llm_usage_to_kernel(app, model_for_pricing, &usage, latency_ms);
        }

        if idle_timed_out && content.is_empty() && attempt < REQUEST_MAX_ATTEMPTS {
            super::emit_request_diagnostic(format_args!(
                "[Warning] {}do_request_text_streaming chunk 空闲超时 ({}s) 且无内容, retrying (attempt {}/{})",
                retry_scope_tag(),
                STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
                attempt,
                REQUEST_MAX_ATTEMPTS
            ));
            attempt += 1;
            continue;
        }
        return Ok(content);
    }
}
