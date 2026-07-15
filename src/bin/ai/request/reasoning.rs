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
        ApiProvider, ReasoningEffort, adapter_for, compatible_wire_shapes, thinking_dialect_for,
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
    let (_, top_level_reasoning_effort, nested_reasoning) =
        if adapter_kind == ApiProvider::Compatible {
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

/// 对需要 reasoning_content echo 的 thinking 模型（如 DeepSeek-R1），
/// 为所有带 tool_calls 的 assistant 消息补上空字符串 reasoning_content。
///
/// DeepSeek 协议要求带 tool_calls 的 assistant 消息必须回传 reasoning_content 字段，
/// 否则服务端会报 400。此处统一用空字符串占位，既满足协议校验又避免历史 reasoning
/// 文本在长 session 里单调累积、拖慢并"变蠢"。
pub(super) fn ensure_reasoning_content_echo_for_thinking_model(
    model: &str,
    messages: &mut [Message],
) {
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    let dialect = thinking_dialect_for(adapter_kind, &request_model, &endpoint);
    if !dialect.requires_reasoning_content_echo() {
        return;
    }

    for message in messages.iter_mut() {
        if message.role != "assistant" || message.reasoning_content.is_some() {
            continue;
        }
        if message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            message.reasoning_content = Some(String::new());
        }
    }
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
/// `cache_control` 是 provider/model 级能力，由 `models.json` 的
/// `explicit_prompt_cache` 字段声明；普通 OpenAI 兼容模型不一定接受该扩展字段。
pub(super) fn prompt_cache_enabled_for_model(model: &str) -> bool {
    prompt_cache_config_enabled() && models::explicit_prompt_cache_enabled(model)
}

fn prompt_cache_config_enabled() -> bool {
    configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_ENABLE,
            "true",
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
/// 2. [models.json](../../../models.json) 中该模型的默认 `reasoning_effort`；
/// 3. `None` -- 不注入字段，保持服务端默认行为。
pub(crate) fn resolve_reasoning_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    if let Some(override_value) = app.cli.reasoning_effort_override.as_ref() {
        return *override_value;
    }
    models::default_reasoning_effort(model)
}
