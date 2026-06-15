//! Provider 行为适配层。
//!
//! 把原先散落在 `request.rs` / `models.rs` / `stream/normalize.rs` /
//! `stream/runtime.rs` / `driver/reflection/background.rs` 中的「按 provider 分支」
//! 逻辑，收敛到一组零状态静态单例（模板方法 + override）。
//!
//! 主链路保持自由函数骨架不变，仅在差异点调用本模块的 hook，确保各 provider
//! 对外的 wire 行为（请求体序列化、流式解析结果、鉴权头）逐字节一致。

use serde_json::{Value, json};

use crate::ai::config_schema::AiConfig;
use crate::ai::stream::{ParsedStreamPayload, try_parse_stream_chunk_loose};

use super::ApiProvider;

pub(in crate::ai) const COMPATIBLE_DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
pub(in crate::ai) const OPENAI_DEFAULT_ENDPOINT: &str =
    "https://api.openai.com/v1/chat/completions";
pub(in crate::ai) const OPENCODE_DEFAULT_ENDPOINT: &str =
    "https://opencode.ai/zen/v1/chat/completions";
pub(in crate::ai) const OPENROUTER_ENDPOINT: &str =
    "https://openrouter.ai/api/v1/chat/completions";

/// 各 LLM provider 的行为差异统一抽象。所有实现均为零状态单例。
///
/// 默认方法实现「OpenAI 兼容族」的通用行为；Compatible / OpenCode 通过
/// override 表达自身差异。
pub(in crate::ai) trait ProviderAdapter: Sync {
    /// 流式解析失败日志使用的标签，也用于诊断。
    fn label(&self) -> &'static str;

    /// 主请求体的 `enable_thinking` 字段取值。
    /// Compatible 显式声明 `Some(requested)`；OpenAI 兼容族不发送该字段（`None`）。
    fn enable_thinking_field(&self, _requested: bool) -> Option<bool> {
        None
    }

    /// 主请求体的 `enable_search` 字段取值。
    /// Compatible 透传调用方传入的开关；OpenAI 兼容族不发送该字段（`None`）。
    fn enable_search_field(&self, _requested: Option<bool>) -> Option<bool> {
        None
    }

    /// 辅助（非主链路）请求是否需要显式关闭 thinking。
    /// Compatible 默认开启 thinking 会撑爆辅助任务超时，需 `Some(false)`。
    fn aux_disable_thinking(&self) -> Option<bool> {
        None
    }

    /// 顶层 `reasoning_effort` 字段取值（OpenAI / OpenRouter / OpenCode 协议）。
    fn reasoning_top_level<'a>(&self, effort: Option<&'a str>) -> Option<&'a str> {
        effort
    }

    /// 嵌套 `reasoning: { effort }` 字段取值（DashScope compatible 协议）。
    fn reasoning_nested(&self, _effort: Option<&str>) -> Option<Value> {
        None
    }

    /// 该 provider 的默认 endpoint（模型未在 models.json 显式声明时使用）。
    fn default_endpoint(&self) -> &'static str;

    /// 读取 API key 时的配置键候选链（按优先级）。
    fn api_key_candidates(&self) -> &'static [&'static str];

    /// 是否在等待首个可见 chunk 时打印提示（OpenCode 首 token 较慢）。
    fn shows_waiting_hint(&self) -> bool {
        false
    }

    /// 解析单条 provider 专属流式 payload。默认走通用 loose 解析，失败时打印
    /// 详细诊断日志；OpenCode override 为更宽松的 loose 解析 + 简短日志。
    fn parse_provider_chunk(&self, payload: &str) -> ParsedStreamPayload {
        match try_parse_stream_chunk_loose(payload) {
            Some(chunk) => ParsedStreamPayload::Chunk(chunk),
            None => {
                let err = serde_json::from_str::<serde_json::Value>(payload)
                    .err()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unable to parse stream payload".to_string());
                eprintln!("handleResponse error [{}] {err}", self.label());
                eprintln!("======> response: ");
                eprintln!("{payload}");
                eprintln!("<======");
                ParsedStreamPayload::Ignore
            }
        }
    }
}

pub(in crate::ai) struct CompatibleAdapter;
pub(in crate::ai) struct OpenAiAdapter;
pub(in crate::ai) struct OpenRouterAdapter;
pub(in crate::ai) struct OpenCodeAdapter;

static COMPATIBLE: CompatibleAdapter = CompatibleAdapter;
static OPENAI: OpenAiAdapter = OpenAiAdapter;
static OPENROUTER: OpenRouterAdapter = OpenRouterAdapter;
static OPENCODE: OpenCodeAdapter = OpenCodeAdapter;

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

/// 根据 provider 与 endpoint 选出对应 adapter。
///
/// OpenRouter 不是独立的 [`ApiProvider`] 变体，而是 OpenAI 协议的 endpoint 变体
/// （endpoint 含 `openrouter.ai`），其流式解析与 OpenAI 一致、仅日志标签不同。
pub(in crate::ai) fn adapter_for(
    provider: ApiProvider,
    endpoint: &str,
) -> &'static dyn ProviderAdapter {
    if endpoint.trim().to_ascii_lowercase().contains("openrouter.ai") {
        return openrouter_adapter();
    }
    match provider {
        ApiProvider::Compatible => compatible_adapter(),
        ApiProvider::OpenAi => openai_adapter(),
        ApiProvider::OpenCode => opencode_adapter(),
    }
}

impl ProviderAdapter for CompatibleAdapter {
    fn label(&self) -> &'static str {
        "compatible"
    }

    fn enable_thinking_field(&self, requested: bool) -> Option<bool> {
        Some(requested)
    }

    fn enable_search_field(&self, requested: Option<bool>) -> Option<bool> {
        requested
    }

    fn aux_disable_thinking(&self) -> Option<bool> {
        Some(false)
    }

    fn reasoning_top_level<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        None
    }

    fn reasoning_nested(&self, effort: Option<&str>) -> Option<Value> {
        effort.map(|effort| json!({ "effort": effort }))
    }

    fn default_endpoint(&self) -> &'static str {
        COMPATIBLE_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_COMPATIBLE_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}

impl ProviderAdapter for OpenAiAdapter {
    fn label(&self) -> &'static str {
        "openai"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENAI_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_OPENROUTER_API_KEY,
            AiConfig::MODEL_OPENAI_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}

impl ProviderAdapter for OpenRouterAdapter {
    fn label(&self) -> &'static str {
        "openrouter"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENROUTER_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_OPENROUTER_API_KEY,
            AiConfig::MODEL_OPENAI_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}

impl ProviderAdapter for OpenCodeAdapter {
    fn label(&self) -> &'static str {
        "opencode"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENCODE_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[AiConfig::MODEL_OPENCODE_API_KEY, AiConfig::MODEL_API_KEY]
    }

    fn shows_waiting_hint(&self) -> bool {
        true
    }

    fn parse_provider_chunk(&self, payload: &str) -> ParsedStreamPayload {
        let trimmed = payload.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return ParsedStreamPayload::Ignore;
        }
        match try_parse_stream_chunk_loose(trimmed) {
            Some(chunk) => ParsedStreamPayload::Chunk(chunk),
            None => {
                eprintln!(
                    "[opencode] ignored payload, length: {}, starts_with: {:.30}",
                    trimmed.len(),
                    trimmed
                );
                ParsedStreamPayload::Ignore
            }
        }
    }
}
