use super::common::{test_app};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};


#[test]
fn test_parse_thinking_gate_output_bool() {
    let s = r#"{"thinking":true,"confidence":0.91}"#;
    assert_eq!(parse_thinking_gate_output(s), Some((true, 0.91)));
}

#[test]
fn thinking_disabled_override_forces_thinking_off() {
    // Once truncation fallback sets thinking_disabled_override, resolve_thinking
    // must short-circuit to false even when the model supports thinking — the
    // last resort for suppressing the reasoning chains of always-thinking models
    // (GLM via enable_thinking).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut app = test_app();
    app.cli.thinking_disabled_override = true;
    let enabled = rt.block_on(super::super::resolve_thinking(
        &app,
        "deepseek-v4-flash-0731-alibaba",
        &[],
    ));
    assert!(!enabled, "override 置位时 thinking 必须关闭");
}

#[test]
fn dashscope_deepseek_defaults_to_thinking_for_simple_requests() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let app = test_app();
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("你好".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let enabled = rt.block_on(super::super::resolve_thinking(
        &app,
        "deepseek-v4-flash-0731-alibaba",
        &messages,
    ));
    assert!(enabled);
}

#[test]
fn effort_off_turns_off_thinking_for_switch_dialects() {
    // `/effort off` (Some(None)) must flip thinking off on dialects with a real
    // off-switch: DashScope `enable_thinking:false` (alibaba adapter) and the
    // DeepSeek `thinking:{"type":"disabled"}` object (official api.deepseek.com).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut app = test_app();
    app.cli.reasoning_effort_override = Some(None);
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    for model in ["deepseek-v4-flash-0731-alibaba", "deepseek-flash-official"] {
        let enabled = rt.block_on(super::super::resolve_thinking(&app, model, &messages));
        assert!(!enabled, "effort off must disable thinking for {model}");
    }
}

#[test]
fn effort_off_keeps_local_flag_for_effort_only_dialects() {
    // Volcano DeepSeek (NoThinkingDialect) has no wire off-switch: resolve_thinking
    // keeps its normal decision (the registry default-effort short-circuit), and the
    // actual thinking-off happens via `reasoning_effort: "none"` in
    // resolve_reasoning_effort — the only lever that dialect has.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut app = test_app();
    app.cli.reasoning_effort_override = Some(None);
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let enabled = rt.block_on(super::super::resolve_thinking(
        &app,
        "deepseek-v4-flash-volcano",
        &messages,
    ));
    assert!(enabled, "NoThinkingDialect keeps the local flag (wire off happens via effort none)");
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-volcano"),
        Some(crate::ai::provider::ReasoningEffort::None),
        "effort off must send reasoning_effort: none for effort-only dialects"
    );
}

#[test]
fn effort_off_omits_effort_for_switch_dialects() {
    // Switch-based dialects express thinking off through their switch; the effort
    // field is omitted (None), never "none" (an unverified value on DashScope).
    let mut app = test_app();
    app.cli.reasoning_effort_override = Some(None);
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-0731-alibaba"),
        None,
    );
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-flash-official"),
        None,
    );
}

#[test]
fn effort_off_on_minimax_is_a_noop() {
    // MiniMax M2.x (Unsupported capability) has no reliable off switch:
    // `/effort off` must NOT change the thinking decision (it stays whatever
    // auto-detection decides) and must NOT emit `reasoning_effort:"none"` (an
    // unverified value the gateway may ignore or reject).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    for model in ["mimo-v2.5-pro", "mimo-v2.5-free-opencode"] {
        let mut app = test_app();
        let baseline = rt.block_on(super::super::resolve_thinking(&app, model, &messages));
        app.cli.reasoning_effort_override = Some(None);
        let with_off = rt.block_on(super::super::resolve_thinking(&app, model, &messages));
        assert_eq!(
            with_off, baseline,
            "/effort off must not change thinking for Unsupported model {model}"
        );
        assert_eq!(
            super::super::reasoning::resolve_reasoning_effort(&app, model),
            None,
            "effort off must not emit reasoning_effort for {model}"
        );
    }
}

#[test]
fn reasoning_effort_display_label_reflects_explicit_off() {
    // The status line must show the user's explicit off, not fold Some(None)
    // back into "auto" (which would misrepresent the off state).
    let mut app = test_app();
    app.cli.reasoning_effort_override = Some(None);
    assert_eq!(
        super::super::reasoning::reasoning_effort_display_label(&app, "deepseek-v4-flash-0731-alibaba"),
        "off"
    );
    // Unsupported (MiniMax): the honest "off (unsupported)" instead of a
    // success-sounding "off".
    assert_eq!(
        super::super::reasoning::reasoning_effort_display_label(&app, "mimo-v2.5-pro"),
        "off (unsupported)"
    );
    // No override: registry default applies, or "server default" when the model
    // declares none (never the misleading "auto").
    let app2 = test_app();
    assert_eq!(
        super::super::reasoning::reasoning_effort_display_label(&app2, "deepseek-v4-flash-volcano"),
        "max"
    );
    assert_eq!(
        super::super::reasoning::reasoning_effort_display_label(&app2, "mimo-v2.5-pro"),
        "server default"
    );
}

#[test]
fn effort_defaults_to_registry_when_no_override() {
    let app = test_app();
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-volcano"),
        Some(crate::ai::provider::ReasoningEffort::Max),
    );
}

#[test]
fn sharp_agent_implies_low_effort_from_live_agent() {
    // Regression: the sharp `low` used to be written into the override at CLI
    // parse, so a live `/agent sharp` never picked it up while `/agent build`
    // (and startup fallbacks to build) kept a stale `low`. It is now derived
    // from the live agent at resolve time, so no switch path can go stale.
    let mut app = test_app();
    // Live `/agent sharp` with no explicit override.
    app.current_agent = "sharp".to_string();
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-volcano"),
        Some(crate::ai::provider::ReasoningEffort::Low),
    );
    assert_eq!(
        super::super::reasoning::reasoning_effort_display_label(&app, "deepseek-v4-flash-volcano"),
        "low",
    );
    // Switching back (`/agent build`, or a startup fallback to build after an
    // unresolvable `--agent sharp`) drops the implied low with no state to clear.
    app.current_agent = "build".to_string();
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-volcano"),
        Some(crate::ai::provider::ReasoningEffort::Max),
    );
}

#[test]
fn explicit_effort_override_wins_over_sharp_default() {
    let mut app = test_app();
    app.current_agent = "sharp".to_string();
    app.cli.reasoning_effort_override = Some(Some(crate::ai::provider::ReasoningEffort::Max));
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-volcano"),
        Some(crate::ai::provider::ReasoningEffort::Max),
    );
    // Explicit off also wins over the sharp default.
    app.cli.reasoning_effort_override = Some(None);
    assert_eq!(
        super::super::reasoning::resolve_reasoning_effort(&app, "deepseek-v4-flash-0731-alibaba"),
        None,
    );
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
    // Simulate a "hi" inflated by system-reminders: after stripping, it should
    // hit the local short-circuit (Casual + short) instead of falling to the
    // slow model gate.
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
