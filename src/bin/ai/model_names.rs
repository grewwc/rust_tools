use std::path::PathBuf;
use std::sync::LazyLock;

use crate::commonw::utils::expanduser;
use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;

use super::provider::{ApiProvider, ModelQualityTier, ReasoningEffort};
use super::request_protocol::RequestProtocolDialect;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelDef {
    pub key: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub name: String,
    /// 请求 wire-profile 适配器。决定请求体/流式解析/鉴权候选链等行为。
    /// 新配置推荐使用 `adapter`；保留 `provider` 作为向后兼容别名。
    #[serde(default, alias = "provider")]
    pub adapter: ApiProvider,
    /// 平台标识，仅用于展示名、selector 后缀、日志与配置语义。
    /// 缺省时回退到 `adapter` 的 slug（如 `compatible` / `openai`）。
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default, alias = "base_url")]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_key_config_key: Option<String>,
    /// 可选：直接指定 API key 字面量（优先级高于 api_key_config_key）。
    /// 用于不想走 configw 查找的场景（如临时测试、自定义 endpoint）。
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub quality_tier: ModelQualityTier,
    pub is_vl: bool,
    pub search_enabled: bool,
    pub tools_default_enabled: bool,
    /// 是否支持在 message content block 上注入
    /// `cache_control: {"type":"ephemeral"}` 以启用显式 prompt cache。
    #[serde(default, alias = "supports_explicit_prompt_cache")]
    pub explicit_prompt_cache: bool,
    #[serde(default)]
    pub enable_thinking: bool,
    /// 可选：模型上下文窗口（token 数）。
    /// 用于 driver 的动态压缩预算估算；缺省时按 quality_tier 回退。
    #[serde(default, alias = "context_window", alias = "max_context_tokens")]
    pub context_window_tokens: Option<usize>,
    /// 可选：单次响应的最大输出 token 数，作为请求的 `max_tokens` 下发。
    /// 许多 OpenAI 兼容 provider 在客户端不指定时会用一个偏保守的补全上限，
    /// 导致大文件 `write_file` / 长文档生成被中途截断。显式声明一个接近模型
    /// 真实上限的值可缓解截断；缺省（None）时不下发 `max_tokens`，沿用历史行为。
    #[serde(default, alias = "max_tokens", alias = "max_completion_tokens")]
    pub max_output_tokens: Option<u32>,
    /// 可选：请求层的 prompt-token-per-minute 预检预算。仅当 `models.json`
    /// 显式配置此字段时，请求层才会在发送前等待；未配置（或为 0）则完全跳过
    /// TPM 预检，避免用错误的默认值误伤不同 provider / key。
    #[serde(default)]
    pub request_tpm_limit: Option<u64>,
    /// 可选：请求所使用的 HTTP 协议方言。绝大多数模型默认走
    /// `chat_completions`；只有少数模型（如 modelhub 的 GPT-5.x）显式走
    /// `responses`。缺省时会按 endpoint 形状做兼容推断，供历史配置平滑升级。
    #[serde(default)]
    pub request_protocol: Option<RequestProtocolDialect>,
    /// 可选：仅对 `responses` 协议生效。开启后，请求会带
    /// `include: ["reasoning.encrypted_content"]` 索取加密推理项，并在同一
    /// turn 的工具调用回合把服务端返回的 `reasoning` output item 原样回放到
    /// input，使模型在多步工具链中保留上一跳推理上下文。缺省关闭：未声明的
    /// 模型行为不变，网关不透传 `encrypted_content` 时回放自动退化为不回传。
    #[serde(default)]
    pub reasoning_encrypted_replay: bool,
    /// 子 agent 模型选择优先级（越大越优先）。同 tier 内按此值降序排列。
    /// 缺省为 0，用户可在 ~/.config/rust_tools/models.json 中覆盖以调整偏好，
    /// 无需重新编译。
    #[serde(default)]
    pub subagent_priority: i32,

    /// 可选：默认推理强度档位。OpenAI/OpenCode 兼容协议使用顶层
    /// `reasoning_effort`，DashScope compatible provider 使用嵌套
    /// `reasoning.effort`。CLI / `/model effort` 命令的覆盖优先级高于这里。
    ///
    /// 在 `models.json` 中可填以下值（大小写不敏感）：
    /// - `"auto"` / `"none"` / `"off"` 或字段省略：等同 `None`，请求中不带
    ///   `reasoning_effort` 字段（与历史行为兼容）；
    /// - `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` / `"max"`：对应档位。
    #[serde(
        default,
        alias = "reasoning_effort",
        rename = "default_reasoning_effort",
        deserialize_with = "deserialize_default_reasoning_effort"
    )]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// 该模型的 `reasoning_effort` 与请求体中的 `tools` 参数不兼容。
    /// 部分网关（如 bytedance modelhub）在 `/v1/chat/completions` 上拒绝
    /// 二者同时出现，返回 400 "Function tools with reasoning_effort are not
    /// supported"。设为 true 时，请求层在检测到 tools 非空时自动省略
    /// `reasoning_effort` 字段以避免 400；无 tools 的请求仍正常携带该字段，
    /// 保留 thinking 能力。
    #[serde(default)]
    pub reasoning_effort_conflicts_with_tools: bool,
}

/// 从字符串反序列化推理强度档位；接受 `auto` / `none` / `off` 等字面量作为
/// "未设置"语义，等同字段省略。
fn deserialize_default_reasoning_effort<'de, D>(
    deserializer: D,
) -> Result<Option<ReasoningEffort>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = Option::<String>::deserialize(deserializer)?;
    let Some(value) = raw else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "auto" | "none" | "off" | "default" | "null" => Ok(None),
        _ => match ReasoningEffort::parse(trimmed) {
            Some(level) => Ok(Some(level)),
            None => Err(serde::de::Error::custom(format!(
                "unknown default_reasoning_effort '{}': expected auto/minimal/low/medium/high/xhigh/max/off",
                trimmed
            ))),
        },
    }
}

static USER_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_user_models);
static BUILTIN_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_builtin_models);
static USER_BY_KEY: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_user_key_index);
static BUILTIN_BY_KEY: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_builtin_key_index);
static USER_BY_NAME: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_user_name_index);
static BUILTIN_BY_NAME: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_builtin_name_index);

fn lookup_key(value: &str) -> String {
    let mut normalized = String::new();
    let mut pending_dash = false;
    for ch in value.trim().to_ascii_lowercase().chars() {
        if ch.is_whitespace() {
            if !normalized.is_empty() {
                pending_dash = true;
            }
            continue;
        }
        if pending_dash && !normalized.ends_with('-') {
            normalized.push('-');
        }
        pending_dash = false;
        normalized.push(ch);
    }
    normalized.trim_matches('-').to_string()
}

pub fn adapter_slug(adapter: ApiProvider) -> &'static str {
    match adapter {
        ApiProvider::Alibaba => "alibaba",
        ApiProvider::Compatible => "compatible",
        ApiProvider::OpenAi => "openai",
        ApiProvider::OpenCode => "opencode",
    }
}

pub fn platform_slug(model: &ModelDef) -> String {
    model
        .platform
        .as_deref()
        .map(str::trim)
        .filter(|platform| !platform.is_empty())
        .map(lookup_key)
        .filter(|platform| !platform.is_empty())
        .unwrap_or_else(|| adapter_slug(model.adapter).to_string())
}

pub fn platform_label(model: &ModelDef) -> String {
    platform_slug(model)
}

pub fn model_handle(model: &ModelDef) -> String {
    // 如果 name 是加密格式（enc: 前缀），则使用 key 作为显示名，
    // 避免补全面板里显示乱码的 enc:xxx-<platform>。
    let is_encrypted = model.name.starts_with("enc:");
    let name = if is_encrypted {
        String::new()
    } else {
        lookup_key(&model.name)
    };
    if name.is_empty() {
        return lookup_key(&model.key);
    }
    format!("{}-{}", name, platform_slug(model))
}

pub fn legacy_adapter_handle(model: &ModelDef) -> Option<String> {
    let is_encrypted = model.name.starts_with("enc:");
    if is_encrypted {
        return None;
    }
    let name = lookup_key(&model.name);
    if name.is_empty() {
        return None;
    }
    let legacy = format!("{}-{}", name, adapter_slug(model.adapter));
    if legacy.eq_ignore_ascii_case(&model_handle(model)) {
        None
    } else {
        Some(legacy)
    }
}

fn user_config_path() -> PathBuf {
    let home = expanduser("~/.config/rust_tools/models.json");
    match home {
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
    }
}

fn builtin_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models.json")
}

fn load_models_from_path(path: &PathBuf) -> Vec<ModelDef> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
        Vec::new()
    })
}

fn load_user_models() -> Vec<ModelDef> {
    let path = user_config_path();
    if path.exists() {
        load_models_from_path(&path)
    } else {
        Vec::new()
    }
}

fn load_builtin_models() -> Vec<ModelDef> {
    let path = builtin_config_path();
    if !path.exists() {
        eprintln!(
            "[model_names] builtin models.json not found at {}",
            path.display()
        );
        std::process::exit(1);
    }
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
        std::process::exit(1);
    })
}

fn build_user_key_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in USER_MODELS.iter().enumerate() {
        insert_model_key_aliases(&mut index, m, i);
    }
    index
}

fn build_builtin_key_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in BUILTIN_MODELS.iter().enumerate() {
        insert_model_key_aliases(&mut index, m, i);
    }
    index
}

fn insert_key_alias(index: &mut SkipMap<String, usize>, alias: &str, i: usize) {
    let key = lookup_key(alias);
    if !key.is_empty() {
        index.insert(key, i);
    }
}

fn insert_model_key_aliases(index: &mut SkipMap<String, usize>, model: &ModelDef, i: usize) {
    insert_key_alias(index, &model_handle(model), i);
    if let Some(legacy) = legacy_adapter_handle(model) {
        insert_key_alias(index, &legacy, i);
    }
    insert_key_alias(index, &model.key, i);
    for alias in &model.aliases {
        insert_key_alias(index, alias, i);
    }
}

fn build_user_name_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in USER_MODELS.iter().enumerate() {
        let key = lookup_key(&m.name);
        if !key.is_empty() && !index.contains_key(&key) {
            index.insert(key, i);
        }
    }
    index
}

fn build_builtin_name_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in BUILTIN_MODELS.iter().enumerate() {
        let key = lookup_key(&m.name);
        if !key.is_empty() && !index.contains_key(&key) {
            index.insert(key, i);
        }
    }
    index
}

pub fn all() -> Vec<&'static ModelDef> {
    let mut seen = SkipSet::new(16);
    let mut result = Vec::new();

    for m in USER_MODELS.iter() {
        let key = lookup_key(&model_handle(m));
        if seen.insert(key) {
            result.push(m);
        }
    }

    for m in BUILTIN_MODELS.iter() {
        let key = lookup_key(&model_handle(m));
        if seen.insert(key) {
            result.push(m);
        }
    }

    result
}

pub fn find_by_name(name: &str) -> Option<&'static ModelDef> {
    let name_lower = lookup_key(name);

    if let Some(&i) = USER_BY_NAME.get_ref(&name_lower) {
        return Some(&USER_MODELS[i]);
    }

    BUILTIN_BY_NAME
        .get_ref(&name_lower)
        .map(|&i| &BUILTIN_MODELS[i])
}

pub fn find_by_key(key: &str) -> Option<&'static ModelDef> {
    let key_lower = lookup_key(key);

    if let Some(&i) = USER_BY_KEY.get_ref(&key_lower) {
        return Some(&USER_MODELS[i]);
    }

    BUILTIN_BY_KEY
        .get_ref(&key_lower)
        .map(|&i| &BUILTIN_MODELS[i])
}

pub fn find_by_identifier(identifier: &str) -> Option<&'static ModelDef> {
    let trimmed = identifier.trim();
    if trimmed.is_empty() {
        return None;
    }
    find_by_key(trimmed).or_else(|| find_by_name(trimmed))
}
