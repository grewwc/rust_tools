//! 思考开关的「线缆方言」抽象。
//!
//! 「是否思考」是一个逻辑开关，但不同网关用不同的请求体字段表达它。这与
//! provider 的鉴权 / 响应消费是**正交**的另一根轴——核心层只传入逻辑开关
//! `enable`，由 [`thinking_dialect_for`] 选出方言，方言负责写哪些 key。这样
//! 各 provider adapter 不必再认识彼此的思考 wire 格式。
//!
//! [`thinking_dialect_for`] 的分派与 [`super::adapter_for`] 一一对应（同样以
//! provider + endpoint 决定网关），因此对 models.json 中所有模型的 wire 输出
//! 与旧的 per-adapter `thinking_fields` 逐字节等价。
//!
//! 三种方言：
//! - [`EnableThinkingDialect`]：DashScope compatible-mode（百炼）→ `enable_thinking: bool`
//! - [`DeepSeekThinkingDialect`]：DeepSeek（OpenCode Zen 网关）→ `thinking: {"type":...}`
//! - [`NoThinkingDialect`]：纯 OpenAI / OpenRouter / MiniMax 等 → 不发送任何字段

use serde_json::{Map, Value, json};

use super::super::ApiProvider;

/// 思考开关的线缆编码方言。零状态单例。
pub(in crate::ai) trait ThinkingDialect: Sync {
    /// 把逻辑开关 `enable` 编码成请求体字段。空 Map 表示该方言不发送任何字段。
    /// `top_level_reasoning_effort` 表示该请求最终是否会发送顶层
    /// `reasoning_effort`；仅少数特殊方言需要据此调整 wire 形状。
    fn fields(&self, enable: bool, top_level_reasoning_effort: Option<&str>) -> Map<String, Value>;

    /// 该方言是否要求 assistant tool-call 消息回传 `reasoning_content` 字段。
    /// DeepSeek thinking-mode 在工具回合续写时会校验该字段，即使网关没有给出
    /// 非空推理文本，也需要保留字段形状以通过协议校验。
    fn requires_reasoning_content_echo(&self) -> bool {
        false
    }
}

/// DashScope 协议：`{"enable_thinking": bool}`。
pub(in crate::ai) struct EnableThinkingDialect;

impl ThinkingDialect for EnableThinkingDialect {
    fn fields(
        &self,
        enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("enable_thinking".to_string(), Value::Bool(enable));
        map
    }
}

/// DeepSeek（经 OpenCode Zen 网关）：`{"thinking": {"type": "enabled"|"disabled"}}`。
/// DeepSeek 忽略 `enable_thinking`，只认这个对象。
/// 但在 OpenCode 上若已走顶层 `reasoning_effort`，该对象必须省略，避免 wire 冲突。
pub(in crate::ai) struct DeepSeekThinkingDialect;

impl ThinkingDialect for DeepSeekThinkingDialect {
    fn fields(&self, enable: bool, top_level_reasoning_effort: Option<&str>) -> Map<String, Value> {
        if top_level_reasoning_effort.is_some() {
            return Map::new();
        }
        let kind = if enable { "enabled" } else { "disabled" };
        let mut map = Map::new();
        map.insert("thinking".to_string(), json!({ "type": kind }));
        map
    }

    fn requires_reasoning_content_echo(&self) -> bool {
        true
    }
}

/// 不发送任何思考字段（纯 OpenAI / OpenRouter 仅靠 `reasoning_effort`；
/// MiniMax M2.x 为 always-on reasoning，网关无可靠关闭开关）。
pub(in crate::ai) struct NoThinkingDialect;

impl ThinkingDialect for NoThinkingDialect {
    fn fields(
        &self,
        _enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        Map::new()
    }
}

static ENABLE_THINKING: EnableThinkingDialect = EnableThinkingDialect;
static DEEPSEEK_THINKING: DeepSeekThinkingDialect = DeepSeekThinkingDialect;
static NO_THINKING: NoThinkingDialect = NoThinkingDialect;

/// 端点是否为 DashScope（阿里云百炼）compatible-mode。
/// 该端点用 `enable_thinking: bool` 控制思考。
fn is_dashscope_endpoint(endpoint: &str) -> bool {
    endpoint.contains("dashscope.aliyuncs.com")
}

/// 按网关（provider + endpoint）与 model 选出思考方言，与 provider 鉴权 /
/// 响应消费轴解耦。分派严格镜像 [`super::adapter_for`]：
/// - OpenRouter 端点 → 不发送（与 OpenAI 一致，仅靠 reasoning_effort）
/// - Alibaba → `enable_thinking`
/// - Compatible → `enable_thinking`
/// - OpenAi → DashScope 端点用 `enable_thinking`，纯 OpenAI 端点不发送
/// - OpenCode → DeepSeek 模型用 `thinking` 对象，其余不发送
pub(in crate::ai) fn thinking_dialect_for(
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
) -> &'static dyn ThinkingDialect {
    if endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("openrouter.ai")
    {
        return &NO_THINKING;
    }
    match provider {
        ApiProvider::Alibaba => &ENABLE_THINKING,
        ApiProvider::Compatible => &ENABLE_THINKING,
        ApiProvider::OpenAi => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else {
                &NO_THINKING
            }
        }
        ApiProvider::OpenCode => {
            if model.to_ascii_lowercase().contains("deepseek") {
                &DEEPSEEK_THINKING
            } else {
                &NO_THINKING
            }
        }
    }
}
