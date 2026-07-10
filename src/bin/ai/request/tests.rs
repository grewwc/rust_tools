
use super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

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
    // 服务端返回极大的 Retry-After（模拟到下个配额窗口的秒数），必须被钳制。
    headers.insert(RETRY_AFTER, HeaderValue::from_static("243749"));
    let delay = parse_retry_after(&headers).expect("should parse numeric retry-after");
    assert_eq!(delay, Duration::from_millis(REQUEST_RETRY_429_MAX_MS));

    // 小于上限的值原样返回。
    let mut small = HeaderMap::new();
    small.insert(RETRY_AFTER, HeaderValue::from_static("3"));
    assert_eq!(
        parse_retry_after(&small),
        Some(Duration::from_secs(3))
    );

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

    // 第一条 system 消息被改写为内容块数组并带 cache_control。
    let blocks = messages[0].content.as_array().expect("array content");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "you are helpful");
    assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    // user 消息保持原样。
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
fn prompt_cache_model_support_uses_models_json_flag() {
    assert!(models::explicit_prompt_cache_enabled("qwen3.7-max"));
    assert!(models::explicit_prompt_cache_enabled("qwen3.7-plus"));
    assert!(models::explicit_prompt_cache_enabled("glm-5.2"));
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
                 no OpenAi model present in models.json"
        );
        return;
    };
    assert!(!models::explicit_prompt_cache_enabled(&model));
}

fn test_app() -> App {
    App {
        cli: ParsedCli::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 0,
            history_keep_last: 0,
            history_summary_max_chars: 0,
            intent_model: None,
            agent_route_model_path: PathBuf::new(),
            skill_match_model_path: PathBuf::new(),
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: String::new(),
        current_agent: String::new(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skill: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown: Arc::new(AtomicBool::new(false)),
        streaming: Arc::new(AtomicBool::new(false)),
        cancel_stream: Arc::new(AtomicBool::new(false)),
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
    }
}

#[test]
fn test_parse_thinking_gate_output_bool() {
    let s = r#"{"thinking":true,"confidence":0.91}"#;
    assert_eq!(parse_thinking_gate_output(s), Some((true, 0.91)));
}

#[test]
fn thinking_disabled_override_forces_thinking_off() {
    // 截断兜底置位 thinking_disabled_override 后，即使模型支持 thinking，
    // resolve_thinking 也必须短路返回 false —— 这是压制 always-thinking 模型
    // （GLM 走 enable_thinking）思考链的最终手段。
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut app = test_app();
    app.cli.thinking_disabled_override = true;
    let enabled = rt.block_on(super::resolve_thinking(&app, "glm-5.2", &[]));
    assert!(!enabled, "override 置位时 thinking 必须关闭");
}

#[test]
fn test_parse_thinking_gate_output_string_bool() {
    let s = r#"{"thinking":"false","confidence":0.8}"#;
    assert_eq!(parse_thinking_gate_output(s), Some((false, 0.8)));
}

#[test]
fn test_parse_thinking_gate_output_with_fence() {
    let s = "```json\n{\"thinking\":true,\"confidence\":0.73}\n```";
    assert_eq!(parse_thinking_gate_output(s), Some((true, 0.73)));
}

#[test]
fn test_parse_thinking_gate_output_invalid() {
    let s = r#"{"confidence":0.73}"#;
    assert_eq!(parse_thinking_gate_output(s), None);
}

#[test]
fn local_thinking_decision_skips_simple_concept_questions() {
    let decision = local_thinking_decision("Rust 的 trait 是什么？");
    assert_eq!(decision, Some(false));
}

#[test]
fn local_thinking_decision_enables_for_debugging_requests() {
    let decision = local_thinking_decision(
        "帮我排查这个报错，并分析可能的修复方案\npanic: index out of bounds",
    );
    assert_eq!(decision, Some(true));
}

#[test]
fn local_thinking_decision_decides_false_locally() {
    let decision = local_thinking_decision("帮我写个函数");
    assert_eq!(decision, Some(false));
}

#[test]
fn strip_system_reminders_removes_injected_block() {
    let raw = "<system-reminder>\nlots of injected context\nmore lines\n</system-reminder>\n\nhi";
    assert_eq!(strip_system_reminders(raw), "\n\nhi");
}

#[test]
fn strip_system_reminders_handles_multiple_and_unclosed() {
    let raw = "<system-reminder>a</system-reminder>real<system-reminder>b</system-reminder> text";
    assert_eq!(strip_system_reminders(raw), "real text");

    let unclosed = "<system-reminder>never closed and then the question hi";
    assert_eq!(strip_system_reminders(unclosed), "");
}

#[test]
fn strip_system_reminders_passthrough_when_absent() {
    assert_eq!(strip_system_reminders("hi"), "hi");
}

#[test]
fn reminder_polluted_greeting_decides_locally() {
    // 模拟被 system-reminder 撑长的 "hi"：剥离后应命中本地短路（Casual+短），
    // 而不是落到耗时的模型 gate。
    let polluted = format!(
        "<system-reminder>{}</system-reminder>\n\nhi",
        "x".repeat(2000)
    );
    let clean = strip_system_reminders(&polluted);
    let clean = clean.trim();
    assert_eq!(local_thinking_decision(clean), Some(false));
}

#[test]
fn thinking_gate_uses_latest_user_message_only() {
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: Value::String(
                "请帮我排查这个复杂报错，并分析可能的修复方案\npanic: index out of bounds"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("之前的复杂问题已经回答完毕。".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("为什么天是蓝的？".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    assert_eq!(
        latest_user_message_text(&messages).as_deref(),
        Some("为什么天是蓝的？")
    );
}

#[tokio::test]
async fn sleep_with_cancel_observes_request_interrupt_source() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    let waiter = tokio::spawn(async move { sleep_with_cancel(&app, Duration::from_secs(5)).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    crate::ai::driver::signal::signal_request_interrupt();

    let cancelled = tokio::time::timeout(Duration::from_millis(200), waiter)
        .await
        .expect("retry wait should wake on interrupt")
        .expect("waiter should complete");
    assert!(cancelled);

    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn clears_stale_interrupt_for_new_request_but_keeps_active_cancel() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();

    crate::ai::driver::signal::signal_request_interrupt();
    assert!(crate::ai::driver::signal::request_interrupt_ready());
    clear_stale_request_interrupt_before_request(&app);
    assert!(!crate::ai::driver::signal::request_interrupt_ready());

    app.cancel_stream
        .store(true, std::sync::atomic::Ordering::Relaxed);
    crate::ai::driver::signal::signal_request_interrupt();
    clear_stale_request_interrupt_before_request(&app);
    assert!(crate::ai::driver::signal::request_interrupt_ready());

    app.cancel_stream
        .store(false, std::sync::atomic::Ordering::Relaxed);
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

/// 找一个真实存在的 OpenAi-provider 模型名做测试输入，避免硬编码
/// 具体模型字符串导致 models.json 变更后测试失效。
fn first_openai_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.provider == crate::ai::provider::ApiProvider::OpenAi)
        .map(|m| m.name.clone())
}

fn first_openai_vl_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.provider == crate::ai::provider::ApiProvider::OpenAi && m.is_vl)
        .map(|m| m.name.clone())
}

fn first_alibaba_vl_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.provider == crate::ai::provider::ApiProvider::Alibaba && m.is_vl)
        .map(|m| m.name.clone())
}

/// 返回该 provider 下第一个模型的 **唯一 key**（而非 `name`）。生产链路
/// 用 key 定位模型（日志里模型标识形如 `glm-5.2-opencode`），而 `name`
/// （如 `glm-5.2`）可能被多个 provider 的条目共享，按 name 查找会命中歧义
/// 条目、解析出错误的 provider 方言。测试必须与生产一致用 key。
fn first_model_key_for_provider(provider: crate::ai::provider::ApiProvider) -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.provider == provider)
        .map(|m| m.key.clone())
}

/// 逐字节 wire guard：锁死各 provider 的 `build_request_body` 序列化结果，
/// 作为 provider adapter 重构「不破坏对外 wire 行为」的可执行回归网。
/// 字段顺序由 [`RequestBody`] 声明顺序决定，serde 输出稳定可断言整串。
#[test]
fn build_request_body_wire_format_is_byte_stable_per_provider() {
    use crate::ai::provider::ApiProvider;

    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // Alibaba：嵌套 reasoning.effort + enable_thinking/enable_search，无 stream_options（非流式）。
    // 用唯一 key 定位模型（生产链路一致），避免共享 name 命中歧义条目。
    let alibaba_model = first_model_key_for_provider(ApiProvider::Alibaba)
        .expect("models.json must contain an Alibaba model");
    let alibaba = build_request_body(
        &alibaba_model,
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        Some("high"),
        None,
        None,
    );
    // wire 里的 model 字段是解析后的 request_model_name（provider 实际模型名），
    // 与用于定位的 key 可能不同。
    let alibaba_wire_model = super::super::models::request_model_name(&alibaba_model);
    // max_tokens 现按剩余上下文窗口钳制；仅当模型声明 max_output_tokens 时下发。
    // 期望值由同一钳制函数推导，保持 wire 断言随模型配置变化仍成立。
    let alibaba_max_tokens_field = expected_max_tokens_field(&alibaba_model, &messages);
    assert_eq!(
        serde_json::to_string(&alibaba).unwrap(),
        format!(
            r#"{{"model":"{alibaba_wire_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"enable_thinking":true,"enable_search":true,"reasoning":{{"effort":"high"}}{alibaba_max_tokens_field}}}"#
        )
    );

    // OpenCode 非 DeepSeek：与 OpenAI 兼容族字段一致
    // （顶层 reasoning_effort、省略扩展字段）。DeepSeek 专属的 `thinking`
    // 字段由单独的 `deepseek_request_body_uses_thinking_object` 测试覆盖。
    let non_deepseek_opencode = crate::ai::model_names::all()
        .iter()
        .find(|m| {
            m.provider == ApiProvider::OpenCode && !m.name.to_ascii_lowercase().contains("deepseek")
        })
        .map(|m| m.key.clone());
    if let Some(opencode_model) = non_deepseek_opencode {
        let opencode = build_request_body(
            &opencode_model,
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
            Some("medium"),
            None,
            None,
        );
        let opencode_wire_model = super::super::models::request_model_name(&opencode_model);
        let opencode_max_tokens_field = expected_max_tokens_field(&opencode_model, &messages);
        assert_eq!(
            serde_json::to_string(&opencode).unwrap(),
            format!(
                r#"{{"model":"{opencode_wire_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"reasoning_effort":"medium"{opencode_max_tokens_field}}}"#
            )
        );
    }
}

/// 复用生产钳制逻辑，构造 wire 断言里 `max_tokens` 字段的期望片段：
/// 模型声明 max_output_tokens 时为 `,"max_tokens":N`，否则为空串。
fn expected_max_tokens_field(model: &str, messages: &[Message]) -> String {
    match super::super::models::max_output_tokens(model) {
        Some(model_max) => {
            let clamped = clamp_max_tokens_for_prompt(model, messages, model_max, None);
            format!(r#","max_tokens":{clamped}"#)
        }
        None => String::new(),
    }
}

/// 回归：历史压缩后 `known_prompt_tokens`（上一轮服务端回填的高值）不能
/// 盖过本轮实际消息量。否则 clamp 会以为 prompt 仍占满窗口，remaining 触底
/// 到 MIN_OUTPUT_TOKENS_FLOOR，always-thinking 模型的输出预算被 reasoning
/// 吃光 → 零可见文本截断重试死循环。
#[test]
fn clamp_ignores_stale_high_known_prompt_after_compression() {
    let model = "glm-5.2-opencode";
    let Some(model_max) = super::super::models::max_output_tokens(model) else {
        return;
    };
    // 本轮消息很短（压缩后），字符估算 ~个位数 token。
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("short message after compression".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 陈旧的高 known（压缩前 ~满窗）不应触底。
    let stale_high = clamp_max_tokens_for_prompt(model, &messages, model_max, Some(259_000));
    assert!(
        stale_high > MIN_OUTPUT_TOKENS_FLOOR,
        "stale-high known_prompt_tokens should not clamp output to the floor, got {stale_high}"
    );

    // 合理的 known（与本轮估算同量级）仍被采纳：结果与不传 known 接近。
    let fresh = clamp_max_tokens_for_prompt(model, &messages, model_max, Some(20));
    let no_known = clamp_max_tokens_for_prompt(model, &messages, model_max, None);
    assert_eq!(fresh, no_known);
}

#[test]
fn build_request_body_sends_provider_model_name_for_key_handle() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let body = build_request_body(
        "deepseek-v4-flash-opencode",
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        None,
        None,
        None,
    );
    let json = serde_json::to_value(&body).unwrap();

    assert_eq!(
        json.get("model").and_then(|v| v.as_str()),
        Some("deepseek-v4-flash")
    );
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("enabled")
    );
}

#[test]
fn opencode_deepseek_reasoning_effort_suppresses_thinking_object() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    for model in [
        "deepseek-v4-flash-opencode",
        "deepseek-v4-flash-free-opencode",
    ] {
        let body = build_request_body(
            model,
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
            Some("high"),
            None,
            None,
        );
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("high"),
            "{model}"
        );
        assert!(json.get("thinking").is_none(), "{model}");
    }
}

#[test]
fn opencode_deepseek_aux_reasoning_effort_omits_disabled_thinking_object() {
    let endpoint = crate::ai::provider::OPENCODE_DEFAULT_ENDPOINT.to_string();
    let (thinking, top_level_reasoning_effort, nested_reasoning) = resolve_reasoning_wire_controls(
        "deepseek-v4-flash-opencode",
        &endpoint,
        false,
        Some("high"),
    );

    assert!(thinking.is_empty());
    assert_eq!(top_level_reasoning_effort, Some("high"));
    assert!(nested_reasoning.is_none());
}

#[test]
fn deepseek_request_body_uses_thinking_object() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 关闭：thinking={"type":"disabled"}
    let disabled = build_request_body(
        "deepseek-v4-flash-free",
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let disabled = serde_json::to_value(&disabled).unwrap();
    assert_eq!(
        disabled.get("thinking"),
        Some(&json!({ "type": "disabled" }))
    );
    // DeepSeek 不应再发送 enable_thinking（避免与 thinking 对象冲突/无效）。
    assert!(disabled.get("enable_thinking").is_none());

    // 开启：thinking={"type":"enabled"}
    let enabled = build_request_body(
        "deepseek-v4-flash-free",
        &messages,
        false,
        true,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let enabled = serde_json::to_value(&enabled).unwrap();
    assert_eq!(enabled.get("thinking"), Some(&json!({ "type": "enabled" })));
}

#[test]
fn non_deepseek_request_body_omits_thinking_object() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let body = build_request_body(
        "qwen3.7-plus",
        &messages,
        true,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();
    assert!(value.get("thinking").is_none());
}

#[test]
fn deepseek_tool_call_messages_echo_empty_reasoning_content() {
    let mut messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];

    ensure_reasoning_content_echo_for_thinking_model("deepseek-v4-flash-free", &mut messages);
    assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(
        value.get("reasoning_content").and_then(|v| v.as_str()),
        Some("")
    );
}

/// 核心回归：DashScope compatible-mode 端点的 Alibaba-provider 模型
/// （deepseek-v4-pro/flash、kimi-k2.7-code）必须按 thinking gate 决策发送
/// `enable_thinking`，否则「关闭」会被静默丢弃、模型仍 reasoning。
#[test]
fn dashscope_alibaba_provider_honors_enable_thinking_gate() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    for model in ["deepseek-v4-pro", "deepseek-v4-flash", "kimi-k2.7-code"] {
        // gate 关闭 → enable_thinking:false
        let disabled = build_request_body(
            model, &messages, false, false, None, None, None, None, None, None,
        );
        let disabled = serde_json::to_value(&disabled).unwrap();
        assert_eq!(
            disabled.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(false),
            "{model} should emit enable_thinking:false when gate disables thinking"
        );
        // 走 enable_thinking 而非 deepseek 的 thinking 对象
        assert!(disabled.get("thinking").is_none(), "{model}");

        // gate 开启 → enable_thinking:true
        let enabled = build_request_body(
            model, &messages, false, true, None, None, None, None, None, None,
        );
        let enabled = serde_json::to_value(&enabled).unwrap();
        assert_eq!(
            enabled.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(true),
            "{model} should emit enable_thinking:true when gate enables thinking"
        );
    }
}

/// 辅助（非主链路）请求对 DashScope 端点模型必须显式关闭 thinking，
/// 否则默认开启的长推理链会撑爆辅助任务超时。
#[test]
fn dashscope_aux_requests_disable_thinking_regardless_of_provider() {
    for model in ["qwen3.7-plus", "deepseek-v4-pro", "kimi-k2.7-code"] {
        let mut body = json!({ "model": model, "messages": [], "stream": false });
        apply_aux_thinking_fields(model, &mut body);
        assert_eq!(
            body.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(false),
            "{model} aux request should disable thinking via enable_thinking:false"
        );
    }

    // OpenCode 的 deepseek 不靠 enable_thinking，aux 关闭走 thinking 对象。
    let mut deepseek =
        json!({ "model": "deepseek-v4-flash-free", "messages": [], "stream": false });
    apply_aux_thinking_fields("deepseek-v4-flash-free", &mut deepseek);
    assert_eq!(
        deepseek.get("thinking"),
        Some(&json!({ "type": "disabled" }))
    );
    assert!(deepseek.get("enable_thinking").is_none());

    // OpenCode 非 deepseek 无可靠关闭开关，aux 不注入任何思考字段。
    let mut mimo = json!({ "model": "mimo-v2.5-free", "messages": [], "stream": false });
    apply_aux_thinking_fields("mimo-v2.5-free", &mut mimo);
    assert!(mimo.get("thinking").is_none());
    assert!(mimo.get("enable_thinking").is_none());
}

#[test]
fn openai_request_body_omits_nonstandard_flags() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hello".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let Some(model) = first_openai_model_name() else {
        eprintln!(
            "[test] skipping openai_request_body_omits_nonstandard_flags: \
                 no OpenAi model present in models.json"
        );
        return;
    };
    let body = build_request_body(
        &model,
        &messages,
        true,
        true,
        Some(true),
        None,
        None,
        Some("high"),
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();

    // OpenAI-provider 不发送 DashScope 扩展字段，推理强度走顶层 reasoning_effort。
    assert!(value.get("enable_thinking").is_none());
    assert!(value.get("enable_search").is_none());
    assert_eq!(
        value.get("reasoning_effort").and_then(|v| v.as_str()),
        Some("high")
    );
    assert!(value.get("reasoning").is_none());
    assert_eq!(
        value.get("model").and_then(|v| v.as_str()),
        Some(model.as_str())
    );
}

#[test]
fn alibaba_request_body_keeps_extension_flags() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hello".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let body = build_request_body(
        "qwen3.7-plus",
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        Some("high"),
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();

    assert_eq!(
        value.get("enable_thinking").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        value.get("enable_search").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert!(value.get("reasoning_effort").is_none());
    assert_eq!(
        value
            .get("reasoning")
            .and_then(|v| v.get("effort"))
            .and_then(|v| v.as_str()),
        Some("high")
    );
}

#[test]
fn normalize_messages_merges_non_leading_system_messages() {
    // Internal notes that appear AFTER the first conversational message
    // must remain in their original positions (with role normalized to
    // "system") so that older prompt-cache prefixes stay valid when new
    // notes are appended. Only notes that sit at the very top, before
    // any user/assistant/tool message, get folded into the first system.
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("history summary".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("working memory".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized[0].role, "system");
    let head_text = normalized[0].content.as_str().unwrap();
    assert!(head_text.contains("base system"));
    assert!(head_text.contains("history summary"));
    assert!(!head_text.contains("working memory"));

    assert_eq!(normalized[1].role, "user");
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(normalized[3].role, "system");
    assert_eq!(normalized[3].content.as_str(), Some("working memory"));
}

#[test]
fn normalize_messages_prioritizes_working_memory_before_summary_and_self_note() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String("self_note:\nremember style".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "对话摘要（自动压缩，以下为早期对话要点）：\nolder summary".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "Current code-inspection working memory:\n- use code_search first".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    let text = normalized[0].content.as_str().unwrap();
    let wm = text.find("## Working Memory").unwrap();
    let summary = text.find("## History Summary").unwrap();
    let self_note = text.find("## Self Notes").unwrap();
    assert!(wm < summary);
    assert!(summary < self_note);
}

#[test]
fn strip_unavailable_tool_hints_removes_code_search_correction_from_working_memory() {
    let mut messages = vec![Message {
            role: "system".to_string(),
            content: Value::String(
                "Current code-inspection working memory:\n\
                 - read_file(file=src/main.rs)\n\
                 Code-navigation correction: you have started raw inspection without `code_search`. Before another raw read, use `code_search` first to locate the relevant file/symbol/definition, then read only the specific region you need.\n\
                 Treat these findings as already-known context."
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

    let available = ["read_file", "find_path"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    let text = messages[0].content.as_str().unwrap();
    assert!(text.contains("Current code-inspection working memory:"));
    assert!(text.contains("Treat these findings as already-known context."));
    assert!(!text.contains("Code-navigation correction:"));
    assert!(!text.contains("`code_search`"));
}

#[test]
fn strip_unavailable_tool_hints_removes_tool_suggestion_lines() {
    let mut messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String(
            "Suggestion: use `code_search` to narrow the target before another raw read.\n\
                 Result: fallback kept."
                .to_string(),
        ),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
        reasoning_content: None,
    }];

    let available = ["read_file"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    let text = messages[0].content.as_str().unwrap();
    assert!(!text.contains("Suggestion:"));
    assert!(text.contains("Result: fallback kept."));
}

#[test]
fn strip_unavailable_tool_hints_keeps_regular_assistant_text() {
    let mut messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(
            "你可以之后再试 `code_search`，但这不是一条内部纠偏提示。".to_string(),
        ),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let available = ["read_file"]
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    strip_unavailable_tool_hints_from_messages(&mut messages, &available);

    assert_eq!(
        messages[0].content.as_str(),
        Some("你可以之后再试 `code_search`，但这不是一条内部纠偏提示。")
    );
}

#[test]
fn normalize_messages_drops_orphan_tool_results_and_strips_broken_tool_calls() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("later answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("stale tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized.len(), 3);
    assert_eq!(normalized[0].role, "system");
    assert_eq!(normalized[1].role, "user");
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(normalized[2].content.as_str(), Some("later answer"));
    assert!(normalized.iter().all(|message| message.role != "tool"));
}

#[test]
fn normalize_messages_keeps_contiguous_tool_call_blocks() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("fresh tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[2].role, "assistant");
    assert_eq!(
        normalized[2].tool_calls.as_ref().map(|calls| calls.len()),
        Some(1)
    );
    assert_eq!(normalized[3].role, "tool");
    assert_eq!(normalized[3].tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn normalize_messages_preserves_tool_evidence_when_malformed_tool_calls_are_dropped() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: "{\"command\":".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("Error: failed to parse arguments".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("later answer".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[2].role, "internal_note");
    let note = normalized[2].content.as_str().unwrap_or_default();
    assert!(note.contains("preserved unmatched tool outputs"));
    assert!(note.contains("failed to parse arguments"));
    assert_eq!(normalized[3].role, "assistant");
    assert_eq!(normalized[3].content.as_str(), Some("later answer"));
    assert!(normalized.iter().all(|message| message.role != "tool"));
    assert!(
        normalized
            .iter()
            .skip(1)
            .all(|message| message.tool_calls.is_none())
    );
}

#[test]
fn normalize_messages_truncates_long_internal_notes_structurally() {
    let mut long_note_lines = Vec::new();
    long_note_lines.push("Current code-inspection working memory:".to_string());
    for i in 0..80usize {
        long_note_lines.push(format!("- finding {i:02}: {}", "x".repeat(40)));
    }

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String("question".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(long_note_lines.join("\n")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(normalized.len(), 3);
    assert_eq!(normalized[2].role, "system");
    let text = normalized[2].content.as_str().unwrap_or_default();
    assert!(text.contains("Current code-inspection working memory:"));
    assert!(text.contains("[truncated:"));
    assert!(text.chars().count() <= 1_200);
}

#[test]
fn openai_image_content_uses_object_image_url_shape() {
    // 仅当 models.json 中存在一个 OpenAi-provider 且 is_vl=true 的模型时
    // 才能验证"以 {image_url:{url:...}} 对象形状下发图像"的协议契约。
    // 真实环境下没有这种模型时（例如 models.json 只有 Compatible VL），
    // 这条契约无从验证，跳过即可。
    let Some(model) = first_openai_vl_model_name() else {
        eprintln!(
            "[test] skipping openai_image_content_uses_object_image_url_shape: \
                 no OpenAi+VL model present in models.json"
        );
        return;
    };

    let path = std::env::temp_dir().join(format!("ai-openai-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"fake").unwrap();

    let value = build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

    let first = value.as_array().and_then(|items| items.first()).unwrap();
    assert_eq!(
        first.get("type").and_then(|v| v.as_str()),
        Some("image_url")
    );
    assert!(
        first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false)
    );
}

#[test]
fn alibaba_image_content_also_uses_object_image_url_shape() {
    let Some(model) = first_alibaba_vl_model_name() else {
        eprintln!(
            "[test] skipping alibaba_image_content_also_uses_object_image_url_shape: \
                 no Alibaba+VL model present in models.json"
        );
        return;
    };

    let path = std::env::temp_dir().join(format!("ai-alibaba-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"fake").unwrap();

    let value = build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

    let first = value.as_array().and_then(|items| items.first()).unwrap();
    assert_eq!(
        first.get("type").and_then(|v| v.as_str()),
        Some("image_url")
    );
    assert!(
        first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false)
    );
}

#[test]
fn normalize_messages_downgrades_image_content_for_text_only_models() {
    let Some(model) = crate::ai::model_names::all()
        .iter()
        .find(|m| !m.is_vl)
        .map(|m| m.name.clone())
    else {
        eprintln!(
            "[test] skipping normalize_messages_downgrades_image_content_for_text_only_models: no text-only model present in models.json"
        );
        return;
    };

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![
                serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": "data:image/png;base64,AAAA" }
                }),
                serde_json::json!({
                    "type": "text",
                    "text": "please explain"
                }),
            ]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_model(&model, &messages);

    assert!(
        normalized
            .iter()
            .all(|message| !matches!(message.content, Value::Array(_)))
    );
    let content = normalized[1].content.as_str().unwrap();
    assert!(content.contains("[image omitted]"));
    assert!(content.contains("please explain"));
}
