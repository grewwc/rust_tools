use super::common::{test_app, first_openai_model_name};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};


#[test]
fn session_title_generation_timeouts_are_relaxed_for_background_work() {
    assert_eq!(SESSION_TITLE_REQUEST_TIMEOUT_SECS, 90);
    assert_eq!(SESSION_TITLE_BODY_TIMEOUT_SECS, 45);
}

#[test]
fn request_diagnostics_follow_terminal_suppression_scope() {
    assert!(request_diagnostics_enabled());

    let emitted = crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, || {
        emit_request_diagnostic(format_args!("hidden request diagnostic"))
    });

    assert!(!emitted);
    assert!(request_diagnostics_enabled());
}

#[test]
fn model_fallback_and_disable_statuses_are_separate() {
    let network = RequestError::cancelled("network timeout");
    assert!(should_try_model_fallback(&network));
    assert!(!should_temporarily_disable_model(&network));
    assert!(should_temporarily_disable_auto_selected_model(&network));

    let bad_request = RequestError::status(StatusCode::BAD_REQUEST, String::new());
    assert!(!should_try_model_fallback(&bad_request));
    assert!(!should_temporarily_disable_model(&bad_request));
    assert!(!should_temporarily_disable_auto_selected_model(
        &bad_request
    ));

    let unauthorized = RequestError::status(StatusCode::UNAUTHORIZED, String::new());
    assert!(should_try_model_fallback(&unauthorized));
    assert!(!should_temporarily_disable_model(&unauthorized));
    assert!(!should_temporarily_disable_auto_selected_model(
        &unauthorized
    ));

    let billing = RequestError::status(StatusCode::PAYMENT_REQUIRED, String::new());
    assert!(should_try_model_fallback(&billing));
    assert!(should_temporarily_disable_model(&billing));
    assert!(should_temporarily_disable_auto_selected_model(&billing));
}

#[test]
fn parse_retry_after_caps_oversized_server_value() {
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    let mut headers = HeaderMap::new();
    // The server returns an enormous Retry-After (simulating seconds until the
    // next quota window); it must be clamped.
    headers.insert(RETRY_AFTER, HeaderValue::from_static("243749"));
    let delay = parse_retry_after(&headers).expect("should parse numeric retry-after");
    assert_eq!(delay, Duration::from_millis(REQUEST_RETRY_429_MAX_MS));

    // Values below the cap are returned unchanged.
    let mut small = HeaderMap::new();
    small.insert(RETRY_AFTER, HeaderValue::from_static("3"));
    assert_eq!(parse_retry_after(&small), Some(Duration::from_secs(3)));

    assert!(parse_retry_after(&HeaderMap::new()).is_none());
}

#[test]
fn is_rate_limited_only_true_for_429() {
    let too_many = RequestError::status(StatusCode::TOO_MANY_REQUESTS, String::new());
    assert!(too_many.is_rate_limited());

    let unauthorized = RequestError::status(StatusCode::UNAUTHORIZED, String::new());
    assert!(!unauthorized.is_rate_limited());

    let network = RequestError::cancelled("boom");
    assert!(!network.is_rate_limited());
}

#[test]
fn tpm_budget_waits_until_old_reservations_leave_window() {
    let mut bucket = token_budget::TokenBudgetBucket::default();
    let now = Instant::now();
    let window = Duration::from_secs(60);

    assert!(matches!(
        bucket.reserve_or_delay(now, 100, 80, window),
        token_budget::BudgetDecision::Reserved
    ));
    match bucket.reserve_or_delay(now + Duration::from_secs(1), 100, 30, window) {
        token_budget::BudgetDecision::Wait(delay) => {
            assert!(delay >= Duration::from_secs(58));
            assert!(delay <= Duration::from_secs(60));
        }
        token_budget::BudgetDecision::Reserved => panic!("second reservation should wait"),
    }
    assert!(matches!(
        bucket.reserve_or_delay(now + Duration::from_secs(61), 100, 30, window),
        token_budget::BudgetDecision::Reserved
    ));
}

#[test]
fn tpm_budget_allows_oversized_single_request_after_bucket_drains() {
    let mut bucket = token_budget::TokenBudgetBucket::default();
    let now = Instant::now();
    let window = Duration::from_secs(60);

    assert!(matches!(
        bucket.reserve_or_delay(now, 100, 80, window),
        token_budget::BudgetDecision::Reserved
    ));
    assert!(matches!(
        bucket.reserve_or_delay(now + Duration::from_secs(1), 100, 150, window),
        token_budget::BudgetDecision::Wait(_)
    ));
    assert!(matches!(
        bucket.reserve_or_delay(now + Duration::from_secs(61), 100, 150, window),
        token_budget::BudgetDecision::Reserved
    ));
}

#[test]
fn tpm_budget_reservation_charges_physical_sends_without_extra_multiplier() {
    assert_eq!(token_budget::test_reservation_tokens(10_000, 1), 10_000);
    assert_eq!(token_budget::test_reservation_tokens(10_000, 3), 30_000);
}

#[test]
fn tpm_budget_uses_current_request_estimate_without_unscoped_usage() {
    assert_eq!(token_budget::tpm_prompt_tokens(46_342, None), 46_342);
    assert_eq!(token_budget::tpm_prompt_tokens(25_875, None), 25_875);
    assert_eq!(token_budget::tpm_prompt_tokens(0, None), 1);
}

#[test]
fn tpm_budget_discounts_only_validated_cache_tokens_and_saturates() {
    // Prefix and API-key compatibility are enforced before this arithmetic.
    assert_eq!(token_budget::tpm_prompt_tokens(81_370, Some(77_184)), 4_186);
    assert_eq!(token_budget::tpm_prompt_tokens(40_000, Some(77_184)), 1);
    assert_eq!(
        token_budget::tpm_prompt_tokens(usize::MAX, None),
        usize::MAX
    );
}

#[test]
fn prompt_feedback_app_forks_do_not_inherit_usage_or_pending_requests() {
    let mut app = test_app();
    let model = crate::ai::model_names::all()
        .first()
        .expect("registry must contain a model")
        .key
        .clone();
    let messages = Vec::new();
    let body = build_request_body(
        &model, &messages, true, false, None, None, None, None, None, None, None,
    );
    app.last_known_prompt_tokens = Some(PromptTokenFeedback::capture(
        &app.session_id,
        &model,
        "https://example.invalid",
        &body,
    ));
    app.last_known_cached_prompt_tokens = Some(100);
    for fork in [
        app.fork_for_subagent(),
        app.snapshot_for_driver_context(),
        app.snapshot_for_detached_helper(),
    ] {
        assert!(fork.last_known_prompt_tokens.is_none());
        assert!(fork.last_known_cached_prompt_tokens.is_none());
    }
    assert!(app.last_known_prompt_tokens.is_some());
}

#[test]
fn child_forks_do_not_inherit_reasoning_effort_controls() {
    // `/effort off` (Some(None)) is upgraded to "thinking off" on the wire; if
    // subagents/background helpers inherited it, delegated tasks would silently
    // lose thinking. The foreground driver snapshot must keep it so the current
    // turn still honors the command.
    let mut app = test_app();
    app.cli.reasoning_effort_override = Some(None);
    app.cli.thinking_disabled_override = true;
    let subagent = app.fork_for_subagent();
    let helper = app.snapshot_for_detached_helper();
    let driver = app.snapshot_for_driver_context();
    assert_eq!(subagent.cli.reasoning_effort_override, None);
    assert!(!subagent.cli.thinking_disabled_override);
    assert_eq!(helper.cli.reasoning_effort_override, None);
    assert!(!helper.cli.thinking_disabled_override);
    assert_eq!(driver.cli.reasoning_effort_override, Some(None));
    assert!(driver.cli.thinking_disabled_override);
}

#[test]
fn tpm_budget_bucket_key_distinguishes_api_keys_without_exposing_plaintext() {
    let a = token_budget::test_budget_key("https://api.example.com", "model-x", "key-a");
    let b = token_budget::test_budget_key("https://api.example.com", "model-x", "key-b");
    assert_ne!(a, b);
    assert!(!a.contains("key-a"));
    assert!(!b.contains("key-b"));
}

#[test]
fn auto_subagent_retry_policy_fails_fast_for_fallback() {
    let regular = request_retry_policy(false);
    assert_eq!(regular.max_attempts, REQUEST_MAX_ATTEMPTS);
    assert_eq!(regular.max_attempts_429, REQUEST_MAX_ATTEMPTS_429);
    assert_eq!(
        regular.header_timeout_secs,
        STREAM_RESPONSE_HEADER_TIMEOUT_SECS
    );

    let auto_subagent = request_retry_policy(true);
    assert_eq!(
        auto_subagent.max_attempts,
        AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS
    );
    assert_eq!(
        auto_subagent.max_attempts_429,
        AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS
    );
    assert_eq!(
        auto_subagent.header_timeout_secs,
        AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS
    );
    // Auto-selection failure switches models; the same sub-agent request must
    // not be duplicated across both.
    assert_eq!(auto_subagent.hedged_max_sends(), 1);
    assert_eq!(regular.hedged_max_sends(), 3);
}

#[test]
fn stream_usage_accepts_anthropic_style_field_aliases() {
    let usage: StreamUsage = serde_json::from_value(serde_json::json!({
        "input_tokens": 1200,
        "output_tokens": 345,
        "total_token_count": 1545,
    }))
    .unwrap();
    let usage = usage.normalized();
    assert_eq!(usage.prompt_tokens, 1200);
    assert_eq!(usage.completion_tokens, 345);
    assert_eq!(usage.total_tokens, 1545);
}

#[test]
fn stream_usage_derives_missing_completion_from_total() {
    let usage = StreamUsage {
        prompt_tokens: 1000,
        completion_tokens: 0,
        total_tokens: 1234,
        ..Default::default()
    }
    .normalized();
    assert_eq!(usage.completion_tokens, 234);
    assert_eq!(usage.total_tokens, 1234);
}

#[test]
fn stream_usage_recovers_output_from_reasoning_tokens() {
    let usage: StreamUsage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 800,
        "completion_tokens": 0,
        "completion_tokens_details": { "reasoning_tokens": 512 },
    }))
    .unwrap();
    let usage = usage.normalized();
    assert_eq!(usage.completion_tokens, 512);
    assert_eq!(usage.total_tokens, 1312);
}

#[test]
fn stream_usage_does_not_double_count_reasoning_when_completion_present() {
    let usage: StreamUsage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 800,
        "completion_tokens": 600,
        "completion_tokens_details": { "reasoning_tokens": 512 },
    }))
    .unwrap();
    let usage = usage.normalized();
    assert_eq!(usage.completion_tokens, 600);
    assert_eq!(usage.total_tokens, 1400);
}

#[test]
fn stream_usage_prefers_completion_derived_from_total_over_reasoning_subset() {
    let usage: StreamUsage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 800,
        "completion_tokens": 0,
        "total_tokens": 1400,
        "completion_tokens_details": { "reasoning_tokens": 512 },
    }))
    .unwrap();
    let usage = usage.normalized();
    assert_eq!(usage.prompt_tokens, 800);
    assert_eq!(usage.completion_tokens, 600);
    assert_eq!(usage.total_tokens, 1400);
}

#[test]
fn stream_usage_derives_prompt_from_total_and_reasoning_only_details() {
    let usage: StreamUsage = serde_json::from_value(serde_json::json!({
        "total_tokens": 1400,
        "completion_tokens_details": { "reasoning_tokens": 512 },
    }))
    .unwrap();
    let usage = usage.normalized();
    assert_eq!(usage.prompt_tokens, 888);
    assert_eq!(usage.completion_tokens, 512);
    assert_eq!(usage.total_tokens, 1400);
}

#[test]
fn prompt_cache_breakpoint_wraps_first_system_message() {
    let mut messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("you are helpful".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    apply_prompt_cache_breakpoint(&mut messages);

    // The first system message is rewritten into a content-block array with
    // cache_control.
    let blocks = messages[0].content.as_array().expect("array content");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "you are helpful");
    assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    // The user message stays as is.
    assert_eq!(messages[1].content, Value::String("hi".to_string()));
}

#[test]
fn prompt_cache_breakpoint_noop_without_system_message() {
    let mut messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    apply_prompt_cache_breakpoint(&mut messages);
    assert_eq!(messages[0].content, Value::String("hi".to_string()));
}

#[test]
fn dashscope_builtin_models_do_not_enable_anthropic_prompt_cache() {
    for model in ["deepseek-v4-flash-0731-alibaba", "qwen3.8-flash-alibaba"] {
        assert!(!models::explicit_prompt_cache_enabled(model), "{model}");
    }
}

#[test]
fn prompt_cache_model_support_does_not_guess_by_name() {
    assert!(!models::explicit_prompt_cache_enabled(
        "anthropic/claude-sonnet-4"
    ));
    assert!(!models::explicit_prompt_cache_enabled("claude-3-5-sonnet"));
}

#[test]
fn prompt_cache_model_support_rejects_plain_openai_model() {
    let Some(model) = first_openai_model_name() else {
        eprintln!(
            "[test] skipping prompt_cache_model_support_rejects_plain_openai_model: \
                 no OpenAi model present in model registry"
        );
        return;
    };
    assert!(!models::explicit_prompt_cache_enabled(&model));
}
