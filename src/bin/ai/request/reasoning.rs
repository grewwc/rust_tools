//! 推理/思考模式控制 + prompt cache 断点注入。
//!
//! 从 request/mod.rs 提取的推理相关逻辑：
//! - thinking wire 字段解析（各 provider adapter 的字段差异）
//! - thinking 模型的 reasoning_content echo 补齐
//! - 辅助/后台请求的思考关闭注入
//! - prompt cache 断点注入（cache_control）
//! - reasoning_effort 档位解析

use serde_json::{Map, Value, json};

use super::super::{
    history::{Message, is_system_like_role},
    models,
    provider::{
        ApiProvider, ReasoningEffort, adapter_for, compatible_wire_shapes,
        reasoning_effort_reduces_thinking_for, thinking_dialect_for,
    },
    types::App,
};
use crate::commonw::configw;

/// 解析各 provider adapter 对思考/推理字段的具体形状。
///
/// 返回三元组：
/// 1. 顶层 thinking 对象（或其它 provider 特定字段），空则不注入；
/// 2. 顶层 reasoning_effort 字符串（部分 provider 放顶层）；
/// 3. 嵌套 reasoning 对象（部分 provider 放 body.reasoning）。
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
    // `enable_search` 的用户请求在 builder 里传入；此处仅关心 reasoning/thinking 三元组，
    // 所以传入 `None` 占位——我们并不依赖这里返回的 enable_search。
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
        // compatible provider 按 endpoint 分流：DashScope 走 DashScope 形状，
        // 其他纯 OpenAI 兼容端点（如内部 modelhub）走 OpenAI 形状。
        // 不能直接用 adapter.reasoning_*() 默认值，因为 trait 单例看不到 endpoint。
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

/// 按模型能力归一化 tool-call assistant 的 `reasoning_content` 回放策略：
/// - GLM 等声明 exact replay 的模型保留服务端原文，维持跨工具调用连续性；
/// - DeepSeek 等要求字段回传的模型保留现有原文，缺失时补空字符串；
/// - 其余模型彻底移除隐藏 reasoning，避免跨 turn 泄漏和上下文膨胀。
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
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
        if !has_tool_calls {
            message.reasoning_content = None;
            continue;
        }
        if exact_replay {
            // exact continuation state 只能由同一模型生成。未标记内容（例如切换前
            // GPT 的 reasoning）和其他 exact 模型的状态都不能跨模型回放。
            message.reasoning_content =
                message.reasoning_content.as_deref().and_then(|reasoning| {
                    crate::ai::history::compress::decode_reasoning_replay_for_model(
                        model, reasoning,
                    )
                });
            continue;
        }
        if shape_only_replay {
            // 内部持久化状态（exact 或 encrypted 回放标记）绝不能原样发给 provider：
            // 这类 blob 编码自其他/本模型的 exact 或加密推理状态，跨模型回放既泄漏
            // 内部状态，也可能被网关当作无效推理文本拒绝。与 GLM exact 标记切到
            // DeepSeek 的既有语义一致（见 request/tests.rs 的跨模型断言），这里把
            // 两类标记统一清成空字符串，只保留字段形状。
            let carries_replay_marker = message
                .reasoning_content
                .as_deref()
                .is_some_and(crate::ai::history::compress::is_persisted_reasoning_replay);
            if carries_replay_marker {
                message.reasoning_content = Some(String::new());
            } else {
                message.reasoning_content.get_or_insert_default();
            }
        } else {
            message.reasoning_content = None;
        }
    }
}

/// 从落库消息里重建 Responses 加密推理回放的侧信道 map（key = 首个 tool_call id）。
///
/// 背景：encrypted-replay 模型的加密推理在产出当轮存于内存 `turn_reasoning_items`，
/// 但那是 turn 级、每轮清空、进程退出即失。跨轮 / Ctrl+C 后 resume 时，只能从落库到
/// `reasoning_content` 的编码 blob 恢复。此函数扫描当前请求投影（已随压缩自然裁剪，
/// 因此天然只回放"未被折叠的近期轮"，与 exact-replay 的回放范围一致），把每个带标记、
/// 且来源模型匹配当前模型的 assistant tool-call 回合解码回 items，挂到其首个 tool_call id。
///
/// 仅补充 `live` 中缺失的 key：内存侧信道（当轮最新捕获）优先，落库解码只填历史空缺，
/// 因此同一 key 不会被旧值覆盖。跨模型（标记里的模型≠当前模型）解码返回 None，自动跳过。
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

/// 把 provider adapter 给出的思考字段合并进辅助/后台请求体。
///
/// 辅助（非主链路）与后台请求固定关闭思考（`enable_thinking=false`），
/// 由各 adapter 决定具体写哪些 key（`enable_thinking:false` /
/// `thinking:{"type":"disabled"}` / 或空），核心层不再判别 provider。
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

/// 是否开启 opt-in 的显式 prompt cache 断点注入。
///
/// `cache_control` 是 provider/model 级能力，由模型注册表（models/）的
/// `explicit_prompt_cache` 字段声明；普通 OpenAI 兼容模型不一定接受该扩展字段。
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

/// 把首条 system / internal_note 消息的纯文本内容改写为带 `cache_control`
/// 的内容块数组，作为显式 prompt 缓存断点。仅在内容当前是字符串时转换，
/// 幂等且不会触碰其它消息。
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
        // 只在第一条 system-like 消息上设置断点即可。
        break;
    }
}

/// 解析当前会话生效的推理强度档位，按优先级从高到低：
/// 1. CLI 参数 `--reasoning-effort` 或 `/model effort <x>` 留下的覆盖
///    （存储在 [`App.cli.reasoning_effort_override`]，其中 `Some(None)`
///    表示用户显式关闭，`None` 表示未设置）；
/// 2. 模型注册表（[models/](../../../../models)）中该模型的默认 `reasoning_effort`；
/// 3. `None` -- 不注入字段，保持服务端默认行为。
pub(crate) fn resolve_reasoning_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    if let Some(override_value) = app.cli.reasoning_effort_override.as_ref() {
        return *override_value;
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

/// 返回输入框中展示的当前请求推理强度。未下发字段时明确标注为服务端默认值，
/// 避免把「无模型默认档位」误显示为某个具体 effort。
pub(crate) fn reasoning_effort_display_label(app: &App, model: &str) -> &'static str {
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
        let model = "muse-spark-1.2-contributor";
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
        let model = "muse-spark-1.2-contributor";
        let stale = vec![json!({"encrypted_content":"OLD"})];
        let fresh = vec![json!({"encrypted_content":"NEW"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(model, &stale)),
        )];
        let mut live: FxHashMap<String, Vec<Value>> = FxHashMap::default();
        live.insert("call-1".to_string(), fresh.clone());
        let rebuilt = reconstruct_encrypted_reasoning_items_for_model(model, &messages, &live);
        // 内存侧信道（当轮最新）优先，落库旧值不得覆盖。
        assert_eq!(rebuilt.get("call-1"), Some(&fresh));
    }

    #[test]
    fn non_encrypted_model_is_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = "glm-5.2-opencode";
        let items = vec![json!({"encrypted_content":"ENC"})];
        // 即便历史里带加密标记，非 encrypted-replay 模型也不重建（返回 live 原样）。
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(
                "muse-spark-1.2-contributor",
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
        // 保存并恢复进程级 env，避免污染并行测试。
        let saved = std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY").ok();

        // SAFETY: 单测内串行 set/remove 同一 key 并在结束前恢复；本测试不与其它
        // 依赖该 env 的测试并行断言。
        unsafe { std::env::set_var("AIOS_DISABLE_ENCRYPTED_REPLAY", "1") };
        assert!(!encrypted_reasoning_replay_runtime_enabled());
        let model = "muse-spark-1.2-contributor";
        let items = vec![json!({"encrypted_content":"ENC"})];
        let messages = vec![assistant_call_with_reasoning(
            "call-1",
            Some(encode_encrypted_reasoning_replay_state(model, &items)),
        )];
        // 关闭时即便模型 capable、历史有编码 blob，也不重建。
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
                "deepseek-v4-flash-opencode",
                "https://api.deepseek.com/v1",
                Some("low"),
            ),
            Some("low"),
        );
    }
}
