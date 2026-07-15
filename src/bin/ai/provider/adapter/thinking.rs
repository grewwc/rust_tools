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

    /// 降低 `reasoning_effort` 是否能实际缩短该方言下的思考链长度。
    ///
    /// 顶层 `reasoning_effort` 方言（OpenAI 兼容族）返回 `true`：降档能压缩推理
    /// 预算。而 [`EnableThinkingDialect`] 这类「思考仅由 `enable_thinking` 布尔
    /// 开关控制、忽略 effort」的方言返回 `false`——对它们降 effort 是空操作，
    /// 截断重试必须直接关 thinking 才能把输出预算让给可见内容。
    fn reasoning_effort_reduces_thinking(&self) -> bool {
        true
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

    fn reasoning_effort_reduces_thinking(&self) -> bool {
        // 思考仅由 `enable_thinking` 布尔开关控制，请求体里根本不带 effort，
        // 降 effort 对思考链长度零影响。
        false
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
/// 该端点用 `enable_thinking: bool` 控制思考。具体判定逻辑见
/// [`super::compatible::is_dashscope_endpoint`]——复用同一实现以保持与 compatible
/// 模块的 wire 形状判定一致（大小写 / 空白 trim）。
use super::compatible::is_dashscope_endpoint;

/// 按网关（provider + endpoint）与 model 选出思考方言，与 provider 鉴权 /
/// 响应消费轴解耦。分派严格镜像 [`super::adapter_for`]：
/// - OpenRouter 端点 → 不发送（与 OpenAI 一致，仅靠 reasoning_effort）
/// - Alibaba → `enable_thinking`（DashScope compatible-mode）
/// - Compatible → 端点为 DashScope 时走 `enable_thinking`，其余（如内部 modelhub、
///   其他 OpenAI 兼容网关）不发送思考字段，仅靠顶层 `reasoning_effort`
/// - OpenAi → DashScope 端点用 `enable_thinking`，纯 OpenAI 端点不发送
/// - OpenCode → DeepSeek 模型用 `thinking` 对象，其余不发送
///
/// 历史上 `Compatible` 全量走 DashScope 方言，导致挂在 compatible provider 下的
/// 纯 OpenAI 兼容端点（如内部 modelhub）收到未知参数 `enable_thinking` 报 400。
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
        ApiProvider::Compatible => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else {
                &NO_THINKING
            }
        }
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

/// 对指定模型，降 `reasoning_effort` 是否能实际缩短思考链。
///
/// `enable_thinking` 布尔开关方言（DashScope / compatible，如 GLM）返回 `false`：
/// 截断重试阶梯里的 effort 降档对它是空操作，必须直接关 thinking 才能收敛。
pub(in crate::ai) fn reasoning_effort_reduces_thinking_for(
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
) -> bool {
    thinking_dialect_for(provider, model, endpoint).reasoning_effort_reduces_thinking()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_thinking_dialect_effort_is_noop() {
        // DashScope compatible 端点走 enable_thinking 开关，降 effort 无效。
        assert!(!reasoning_effort_reduces_thinking_for(
            ApiProvider::Compatible,
            "glm5.2-super-relay",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ));
        assert!(!reasoning_effort_reduces_thinking_for(
            ApiProvider::Alibaba,
            "qwen-max",
            super::super::ALIBABA_DEFAULT_ENDPOINT,
        ));
    }

    #[test]
    fn non_dashscope_compatible_uses_openai_dialect() {
        // 字节 modelhub / Ollama / 自部署 vLLM 等真正 OpenAI 兼容端点，
        // 即使走 `compatible` provider，也应按 OpenAI 方言处理（reasoning_effort 生效，不发 enable_thinking）。
        assert!(reasoning_effort_reduces_thinking_for(
            ApiProvider::Compatible,
            "Kimi-K2.5",
            "https://dataagent-dev-llm.bytedance.net/api/chat/completions",
        ));
    }

    #[test]
    fn openai_family_effort_reduces_thinking() {
        // 顶层 reasoning_effort 方言：降档能压缩推理预算。
        assert!(reasoning_effort_reduces_thinking_for(
            ApiProvider::OpenAi,
            "gpt-5",
            super::super::OPENAI_DEFAULT_ENDPOINT,
        ));
    }
}
