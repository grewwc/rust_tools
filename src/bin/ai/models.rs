use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;

use super::agents::{AgentManifest, AgentModelTier};
use super::cli::ParsedCli;
use super::config_schema::AiConfig;
use super::model_names::{self, ModelDef};
use super::provider::{self, ApiProvider, ModelQualityTier, ReasoningEffort};
use super::request_protocol::RequestProtocolDialect;
use crate::commonw::configw;

fn model_def(model: &str) -> Option<&'static ModelDef> {
    model_names::find_by_identifier(model)
}

fn model_handle(model: &ModelDef) -> String {
    model_names::model_handle(model)
}

pub(super) fn request_model_name(model: &str) -> String {
    model_def(model)
        .map(|m| maybe_decrypt(&m.name))
        .unwrap_or_else(|| model.trim().to_string())
}

pub(super) fn model_display_label(model: &str) -> String {
    match model_def(model) {
        Some(def) => model_names::model_handle(def),
        None => model.trim().to_string(),
    }
}

pub(super) fn is_vl_model(model: &str) -> bool {
    model_def(model).map(|m| m.is_vl).unwrap_or(false)
}

pub(super) fn search_enabled(model: &str) -> bool {
    model_def(model).map(|m| m.search_enabled).unwrap_or(true)
}

pub(super) fn tools_enabled(model: &str) -> bool {
    model_def(model)
        .map(|m| m.tools_default_enabled)
        .unwrap_or(true)
}

pub(super) fn explicit_prompt_cache_enabled(model: &str) -> bool {
    model_def(model)
        .map(|m| m.explicit_prompt_cache)
        .unwrap_or(false)
}

pub(super) fn enable_thinking(model: &str) -> bool {
    model_def(model).map(|m| m.enable_thinking).unwrap_or(false)
}

/// 返回该模型在 [models.json](../../../models.json) 中声明的默认推理强度
/// （`reasoning_effort`）。CLI / `/model effort` 命令的覆盖会在
/// `request::resolve_reasoning_effort` 里优先生效，此处仅给出"模型默认"。
pub(super) fn default_reasoning_effort(model: &str) -> Option<ReasoningEffort> {
    model_def(model).and_then(|m| m.reasoning_effort)
}

/// 返回该模型是否声明了 `reasoning_effort` 与 `tools` 不兼容。
/// 若为 true，请求层在携带 tools 时会自动省略 `reasoning_effort` 字段，
/// 避免部分网关（如 bytedance modelhub）返回 400。
/// 无 tools 的请求不受影响，仍正常发送 `reasoning_effort` 以保留 thinking。
pub(super) fn reasoning_effort_conflicts_with_tools(model: &str) -> bool {
    model_def(model)
        .map(|m| m.reasoning_effort_conflicts_with_tools)
        .unwrap_or(false)
}

/// 返回该模型在 [models.json](../../../models.json) 中声明的单次响应最大输出
/// token 数（`max_output_tokens`）。缺省时为 `None`，请求不下发 `max_tokens`，
/// 沿用 provider 默认补全上限。
pub(super) fn max_output_tokens(model: &str) -> Option<u32> {
    model_def(model).and_then(|m| m.max_output_tokens)
}

/// 返回该模型在 [models.json](../../../models.json) 中声明的请求层 TPM 预检预算。
/// 仅当显式配置了正整数时返回 `Some(limit)`；未配置或为 0 都视为关闭预检等待。
pub(super) fn request_tpm_limit(model: &str) -> Option<u64> {
    model_def(model)
        .and_then(|m| m.request_tpm_limit)
        .filter(|limit| *limit > 0)
}

/// 返回该模型当前生效的请求协议方言。
///
/// 优先使用 models.json 中显式声明的 `request_protocol`；未声明时按 endpoint
/// 做兼容推断，以便历史配置在不改文件时也维持原有 wire 行为。
pub(super) fn request_protocol_dialect(model: &str, endpoint: &str) -> RequestProtocolDialect {
    model_def(model)
        .and_then(|m| m.request_protocol)
        .unwrap_or_else(|| RequestProtocolDialect::infer_from_endpoint(endpoint))
}

/// 该模型是否启用「加密推理项回放」（仅 `responses` 协议有意义）。
/// 缺省关闭；开启后请求带 `include: ["reasoning.encrypted_content"]`，并在同
/// turn 工具链中回放服务端返回的 reasoning item。
pub(super) fn reasoning_encrypted_replay_enabled(model: &str) -> bool {
    model_def(model)
        .map(|m| m.reasoning_encrypted_replay)
        .unwrap_or(false)
}

pub(super) fn model_adapter(model: &str) -> ApiProvider {
    model_def(model).map(|m| m.adapter).unwrap_or_default()
}

pub(super) fn model_platform_label(model: &str) -> String {
    model_def(model)
        .map(model_names::platform_label)
        .unwrap_or_else(|| model.trim().to_string())
}

fn default_endpoint_for_adapter(adapter: ApiProvider) -> &'static str {
    provider::adapter_for(adapter, "").default_endpoint()
}

fn default_api_key_config_candidates(adapter: ApiProvider) -> &'static [&'static str] {
    provider::adapter_for(adapter, "").api_key_candidates()
}

pub(super) fn endpoint_for_model(model: &str, global_fallback: &str) -> String {
    if let Some(model_def) = model_def(model) {
        if let Some(endpoint) = model_def
            .endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
        {
            return maybe_decrypt(endpoint);
        }

        return maybe_decrypt(default_endpoint_for_adapter(model_def.adapter));
    }

    let global_fallback = global_fallback.trim();
    if !global_fallback.is_empty() {
        return maybe_decrypt(global_fallback);
    }

    maybe_decrypt(default_endpoint_for_adapter(model_adapter(model)))
}

/// 如果值以 `enc:` 开头，自动解密；否则原样返回。
/// 解密失败时打印警告并返回原始值（避免静默失败导致请求失败）。
fn maybe_decrypt(value: &str) -> String {
    if !crate::commonw::secret::is_encrypted(value) {
        return value.to_string();
    }
    match crate::commonw::secret::decrypt(value) {
        Ok(plain) => plain,
        Err(e) => {
            eprintln!("[models] failed to decrypt api_key: {e}");
            value.to_string()
        }
    }
}

pub(super) fn api_key_for_model(model: &str, global_fallback: &str) -> String {
    let cfg = configw::get_all_config();

    // 1. 字面量 api_key（models.json 直接写明，最高优先级）
    if let Some(value) = model_def(model)
        .and_then(|m| m.api_key.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return maybe_decrypt(value);
    }

    // 2. api_key_config_key（通过 configw 查找）
    if let Some(config_key) = model_def(model)
        .and_then(|m| m.api_key_config_key.as_deref())
        .map(|k| maybe_decrypt(k))
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        && let Some(value) = cfg
            .get_opt(config_key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    {
        return maybe_decrypt(&value);
    }

    // 3. adapter 默认候选 key
    for key in default_api_key_config_candidates(model_adapter(model)) {
        if let Some(value) = cfg
            .get_opt(key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            return maybe_decrypt(&value);
        }
    }

    // 4. 全局回退
    maybe_decrypt(global_fallback.trim())
}

pub(super) fn endpoint_supports_anonymous_auth(endpoint: &str) -> bool {
    let endpoint = endpoint.trim().to_ascii_lowercase();
    endpoint.starts_with("http://127.0.0.1")
        || endpoint.starts_with("http://localhost")
        || endpoint.starts_with("http://0.0.0.0")
        || endpoint.starts_with("http://[::1]")
}

pub(super) fn model_quality_tier(model: &str) -> ModelQualityTier {
    model_def(model).map(|m| m.quality_tier).unwrap_or_default()
}

fn default_context_window_tokens_for_tier(tier: ModelQualityTier) -> usize {
    match tier {
        ModelQualityTier::Flagship => 256_000,
        ModelQualityTier::Strong => 200_000,
        ModelQualityTier::Standard => 128_000,
        // Basic → 100K token = 200K 字符（CHARS_PER_TOKEN=2），作为 LLM 摘要默认阈值。
        ModelQualityTier::Basic => 100_000,
    }
}

/// 返回模型上下文窗口（token）。
/// 若 models.json 未声明，按质量档位给出保守默认值，供压缩预算动态估算使用。
pub(super) fn context_window_tokens(model: &str) -> usize {
    if let Some(def) = model_def(model) {
        return def
            .context_window_tokens
            .filter(|v| *v > 0)
            .unwrap_or_else(|| default_context_window_tokens_for_tier(def.quality_tier));
    }
    default_context_window_tokens_for_tier(model_quality_tier(model))
}

fn all_model_search_candidates() -> Vec<(String, String)> {
    model_names::all()
        .iter()
        .flat_map(|m| {
            let handle = model_handle(m);
            [(m.name.clone(), handle.clone()), (handle.clone(), handle)]
        })
        .collect()
}

fn vl_model_search_candidates() -> Vec<(String, String)> {
    model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .flat_map(|m| {
            let handle = model_handle(m);
            [(m.name.clone(), handle.clone()), (handle.clone(), handle)]
        })
        .collect()
}

fn default_model() -> String {
    // 依赖前置 [`ensure_models_available`] 在 run() 入口就检查过 models.json，
    // 这里的 fallback 路径（vl 模型缺失时）总会拿到至少一个候选。
    // 最后兜底返回空串避免 process::exit；上层若真的拿到空串会立即报错。
    choose_default_model_name(false)
        .or_else(|| choose_default_model_name(true))
        .unwrap_or_default()
}

fn disabled_model_tokens() -> Vec<String> {
    let raw = configw::get_all_config()
        .get_opt(AiConfig::MODEL_DISABLED)
        .unwrap_or_default();
    parse_disabled_model_tokens(&raw)
}

fn parse_disabled_model_tokens(raw: &str) -> Vec<String> {
    raw.split([',', '\n', ';'])
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn model_matches_disabled_tokens(model: &ModelDef, disabled: &[String]) -> bool {
    disabled.iter().any(|token| {
        let handle = model_names::model_handle(model);
        token.eq_ignore_ascii_case(&model.key)
            || token.eq_ignore_ascii_case(&model.name)
            || token.eq_ignore_ascii_case(&handle)
            || model
                .aliases
                .iter()
                .any(|alias| token.eq_ignore_ascii_case(alias))
    })
}

fn model_auto_select_enabled(model: &ModelDef, disabled: &[String]) -> bool {
    !model_matches_disabled_tokens(model, disabled) && !runtime_model_disabled(model)
}

/// 启动时调用，确保至少存在一个可用 model。
/// 在 run() 入口集中报错，避免 [`default_model`] 在更深的调用链里 panic / exit。
pub(super) fn ensure_models_available() -> Result<(), String> {
    if model_names::all().is_empty() {
        return Err(
            "[model_names] models.json is empty; please populate it before launching ai"
                .to_string(),
        );
    }
    if choose_default_model_name(false).is_none() && choose_default_model_name(true).is_none() {
        return Err("[model_names] no usable default model; check models.json entries".to_string());
    }
    Ok(())
}

pub(super) fn default_vl_model() -> String {
    choose_default_model_name(true).unwrap_or_else(default_model)
}

pub(super) fn forced_deepseek_model() -> String {
    model_names::find_by_key("DEEPSEEK_V3")
        .map(model_handle)
        .unwrap_or_else(default_model)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubagentTaskDifficulty {
    Light,
    Standard,
    Heavy,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub(crate) enum ModelStrengthTier {
    Light,
    Standard,
    Heavy,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) struct AutoModelFallbackSpec {
    pub(super) require_thinking: bool,
    pub(super) target_tier: ModelStrengthTier,
}

pub(super) struct AutoSubagentModelChoice {
    pub(super) model: String,
    pub(super) fallback: AutoModelFallbackSpec,
}

pub(super) struct SubagentModelChoice {
    pub(super) model: String,
    pub(super) is_auto_selected: bool,
    pub(super) fallback: Option<AutoModelFallbackSpec>,
}

pub(super) fn initial_model(cli: &ParsedCli) -> String {
    if let Some(ref model) = cli.model
        && !model.trim().is_empty()
    {
        return determine_model(model);
    }
    let cfg = crate::commonw::configw::get_all_config();
    cfg.get_opt(AiConfig::MODEL_DEFAULT)
        .filter(|v| !v.trim().is_empty())
        .map(|v| determine_model(&v))
        .unwrap_or_else(default_model)
}

pub(super) fn determine_model(model: &str) -> String {
    let raw = model.trim();
    if raw.is_empty() {
        return default_model();
    }
    if let Some(def) = model_names::find_by_identifier(raw) {
        return model_handle(def);
    }
    best_match_model_handle(
        &raw.to_lowercase(),
        all_model_search_candidates().into_iter(),
        default_model(),
    )
}

pub(super) fn determine_vl_model(model: &str) -> String {
    let raw = model.trim();
    let model = raw.to_lowercase();
    if model.is_empty() {
        return default_vl_model();
    }

    if let Ok(idx) = model.parse::<usize>() {
        let all = model_names::all();
        let vl = all.iter().filter(|m| m.is_vl).nth(idx);
        if let Some(vl) = vl {
            return model_handle(vl);
        }
        return default_vl_model();
    }

    if let Some(def) = model_names::find_by_identifier(&model)
        && def.is_vl
    {
        return model_handle(def);
    }

    best_match_model_handle(
        &model,
        vl_model_search_candidates().into_iter(),
        default_vl_model(),
    )
}

pub(super) fn supports_image_input(model: &str) -> bool {
    is_vl_model(model)
}

const MODEL_RUNTIME_COOLDOWN: Duration = Duration::from_secs(15 * 60);
const MODEL_RUNTIME_VERIFIED_TTL: Duration = Duration::from_secs(30 * 60);

static RUNTIME_DISABLED_MODELS: LazyLock<Mutex<FxHashMap<String, Instant>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
static RUNTIME_VERIFIED_MODELS: LazyLock<Mutex<FxHashMap<String, Instant>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

fn normalize_model_token(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

pub(super) fn mark_model_temporarily_unavailable(model: &str, reason: &str) {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return;
    }
    if let Ok(mut verified) = RUNTIME_VERIFIED_MODELS.lock() {
        verified.remove(&normalize_model_token(trimmed));
    }
    let until = Instant::now() + MODEL_RUNTIME_COOLDOWN;
    if let Ok(mut disabled) = RUNTIME_DISABLED_MODELS.lock() {
        if let Some(def) = model_names::find_by_identifier(trimmed) {
            disabled.insert(normalize_model_token(&def.name), until);
            disabled.insert(normalize_model_token(&def.key), until);
            disabled.insert(
                normalize_model_token(&model_names::model_handle(def)),
                until,
            );
            for alias in &def.aliases {
                disabled.insert(normalize_model_token(alias), until);
            }
        } else {
            disabled.insert(normalize_model_token(trimmed), until);
        }
    }
    eprintln!(
        "[model] temporarily disabled '{}' for auto-selection: {}",
        trimmed, reason
    );
}

pub(super) fn subagent_model_needs_probe(model: &str) -> bool {
    let token = normalize_model_token(model);
    if token.is_empty() {
        return false;
    }
    let now = Instant::now();
    let Ok(mut verified) = RUNTIME_VERIFIED_MODELS.lock() else {
        return true;
    };
    verified.retain(|_, until| *until > now);
    !verified.contains_key(&token)
}

pub(super) fn mark_subagent_model_verified(model: &str) {
    let token = normalize_model_token(model);
    if token.is_empty() {
        return;
    }
    if let Ok(mut verified) = RUNTIME_VERIFIED_MODELS.lock() {
        verified.insert(token, Instant::now() + MODEL_RUNTIME_VERIFIED_TTL);
    }
}

fn runtime_model_disabled(model: &ModelDef) -> bool {
    let now = Instant::now();
    let Ok(mut disabled) = RUNTIME_DISABLED_MODELS.lock() else {
        return false;
    };
    disabled.retain(|_, until| *until > now);
    let handle = model_names::model_handle(model);
    disabled.contains_key(&normalize_model_token(&model.name))
        || disabled.contains_key(&normalize_model_token(&model.key))
        || disabled.contains_key(&normalize_model_token(&handle))
        || model
            .aliases
            .iter()
            .any(|alias| disabled.contains_key(&normalize_model_token(alias)))
}

pub(super) fn auto_subagent_model_for_agent(
    agent: &AgentManifest,
    description: &str,
    prompt: &str,
) -> String {
    auto_subagent_model_choice_for_agent(agent, description, prompt).model
}

/// 未显式指定模型时，始终按任务难度和 agent 分级自动选择子 agent 模型。
/// `parent_model` 仅保留用于兼容现有调用方；父模型不再覆盖难度路由结果。
pub(super) fn choose_model_for_subagent(
    _parent_model: Option<&str>,
    agent: &AgentManifest,
    description: &str,
    prompt: &str,
) -> SubagentModelChoice {
    let auto = auto_subagent_model_choice_for_agent(agent, description, prompt);
    SubagentModelChoice {
        model: auto.model,
        is_auto_selected: true,
        fallback: Some(auto.fallback),
    }
}

pub(super) fn auto_subagent_model_choice_for_agent(
    agent: &AgentManifest,
    description: &str,
    prompt: &str,
) -> AutoSubagentModelChoice {
    let difficulty = classify_subagent_task_difficulty(description, prompt);
    let target_tier = merge_agent_tier_with_difficulty(agent_model_tier(agent), difficulty);
    let require_thinking = !matches!(target_tier, ModelStrengthTier::Light);
    AutoSubagentModelChoice {
        model: pick_subagent_model(require_thinking, target_tier),
        fallback: AutoModelFallbackSpec {
            require_thinking,
            target_tier,
        },
    }
}

pub(super) fn fallback_subagent_model_after_failure(
    failed_model: &str,
    spec: AutoModelFallbackSpec,
) -> Option<String> {
    pick_subagent_model_excluding(spec.require_thinking, spec.target_tier, Some(failed_model))
}

fn agent_model_tier(agent: &AgentManifest) -> ModelStrengthTier {
    match agent.model_tier {
        Some(AgentModelTier::Light) => ModelStrengthTier::Light,
        Some(AgentModelTier::Standard) => ModelStrengthTier::Standard,
        Some(AgentModelTier::Heavy) => ModelStrengthTier::Heavy,
        None => ModelStrengthTier::Standard,
    }
}

fn merge_agent_tier_with_difficulty(
    base_tier: ModelStrengthTier,
    difficulty: SubagentTaskDifficulty,
) -> ModelStrengthTier {
    match difficulty {
        SubagentTaskDifficulty::Heavy => ModelStrengthTier::Heavy,
        SubagentTaskDifficulty::Standard => base_tier,
        SubagentTaskDifficulty::Light => match base_tier {
            ModelStrengthTier::Heavy => ModelStrengthTier::Standard,
            other => other,
        },
    }
}

fn classify_subagent_task_difficulty(description: &str, prompt: &str) -> SubagentTaskDifficulty {
    let combined = format!("{}\n{}", description.trim(), prompt.trim());
    let lower = combined.to_lowercase();
    let char_count = combined.chars().count();
    let line_count = combined
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    let conjunction_hits = [
        " and ", " then ", " after ", "同时", "然后", "并且", "接着", "最后",
    ]
    .iter()
    .filter(|marker| lower.contains(**marker))
    .count();
    let complex_markers = [
        "multi-step",
        "end-to-end",
        "across files",
        "cross-file",
        "debug",
        "refactor",
        "migrate",
        "integrate",
        "repair",
        "fix failing",
        "panic",
        "compile error",
        "test failure",
        "run tests",
        "implement a fix",
        "implement fixes",
        "implement the change",
        "make code changes",
        "architecture",
        "design",
        "复杂",
        "端到端",
        "跨文件",
        "调试",
        "排查",
        "修复",
        "重构",
        "迁移",
        "联调",
        "实现修复",
        "实现改动",
        "跑测试",
        "报错",
    ];
    let light_markers = [
        "find",
        "search",
        "locate",
        "where",
        "list",
        "summarize",
        "show",
        "identify",
        "look up",
        "read-only",
        "read only",
        "which file",
        "which files",
        "查找",
        "定位",
        "看看",
        "看一下",
        "列出",
        "总结",
        "只读",
        "搜索",
        "梳理位置",
    ];

    let heavy = char_count >= 360
        || line_count >= 8
        || conjunction_hits >= 2
        || complex_markers.iter().any(|marker| lower.contains(marker));
    if heavy {
        return SubagentTaskDifficulty::Heavy;
    }

    let light = char_count <= 140
        && line_count <= 4
        && light_markers.iter().any(|marker| lower.contains(marker));
    if light {
        return SubagentTaskDifficulty::Light;
    }

    SubagentTaskDifficulty::Standard
}

/// 基于 models.json 中的 subagent_priority 字段选择子 agent 模型。
/// priority 越大越优先；同 priority 时按 quality_tier 和 target_tier 适配度排序。
fn pick_subagent_model(require_thinking: bool, target_tier: ModelStrengthTier) -> String {
    pick_subagent_model_excluding(require_thinking, target_tier, None).unwrap_or_else(default_model)
}

fn pick_subagent_model_excluding(
    require_thinking: bool,
    target_tier: ModelStrengthTier,
    excluded_model: Option<&str>,
) -> Option<String> {
    let disabled = disabled_model_tokens();
    let excluded = excluded_model.and_then(|model| {
        model_names::find_by_identifier(model)
            .map(|def| {
                vec![
                    normalize_model_token(&def.name),
                    normalize_model_token(&def.key),
                    normalize_model_token(&model_names::model_handle(def)),
                ]
            })
            .or_else(|| Some(vec![normalize_model_token(model)]))
    });
    let mut candidates: Vec<&ModelDef> = model_names::all()
        .into_iter()
        .filter(|model| {
            model_auto_select_enabled(model, &disabled)
                && subagent_model_eligible(model, require_thinking)
                && !model_matches_excluded_tokens(model, excluded.as_deref())
        })
        .collect();

    // 按 (满足 target_tier, priority, quality_tier) 降序排列
    candidates.sort_by(|a, b| {
        let a_satisfies = quality_tier_satisfies_target(a.quality_tier, target_tier) as u8;
        let b_satisfies = quality_tier_satisfies_target(b.quality_tier, target_tier) as u8;
        b_satisfies
            .cmp(&a_satisfies)
            .then(b.subagent_priority.cmp(&a.subagent_priority))
            .then(b.quality_tier.cmp(&a.quality_tier))
    });

    if let Some(model) = candidates.first() {
        return Some(model_handle(model));
    }

    // fallback: 放宽 thinking 要求，只要求 tools 可用
    if require_thinking {
        let mut tools_only: Vec<&ModelDef> = model_names::all()
            .into_iter()
            .filter(|model| {
                model_auto_select_enabled(model, &disabled)
                    && model.tools_default_enabled
                    && !model_matches_excluded_tokens(model, excluded.as_deref())
            })
            .collect();
        tools_only.sort_by(|a, b| {
            b.subagent_priority
                .cmp(&a.subagent_priority)
                .then(b.quality_tier.cmp(&a.quality_tier))
        });
        if let Some(model) = tools_only.first() {
            return Some(model_handle(model));
        }
    }

    None
}

fn model_matches_excluded_tokens(model: &ModelDef, excluded: Option<&[String]>) -> bool {
    excluded.is_some_and(|tokens| {
        tokens.iter().any(|token| {
            let handle = model_names::model_handle(model);
            token.eq_ignore_ascii_case(&model.key)
                || token.eq_ignore_ascii_case(&model.name)
                || token.eq_ignore_ascii_case(&handle)
                || model
                    .aliases
                    .iter()
                    .any(|alias| token.eq_ignore_ascii_case(alias))
        })
    })
}

fn subagent_model_eligible(model: &ModelDef, require_thinking: bool) -> bool {
    model.tools_default_enabled && (!require_thinking || model.enable_thinking)
}

fn choose_default_model_name(require_vl: bool) -> Option<String> {
    let disabled = disabled_model_tokens();
    let candidates = model_names::all()
        .into_iter()
        .enumerate()
        .filter(|(_, model)| {
            model.is_vl == require_vl && model_auto_select_enabled(model, &disabled)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }

    choose_best_default_candidate(
        &candidates
            .iter()
            .copied()
            .filter(|(_, model)| {
                matches!(
                    model.adapter,
                    ApiProvider::Alibaba | ApiProvider::Compatible
                )
            })
            .collect::<Vec<_>>(),
    )
    .or_else(|| choose_best_default_candidate(&candidates))
    .map(|(_, model)| model_handle(model))
}

fn choose_best_default_candidate<'a>(
    candidates: &[(usize, &'a ModelDef)],
) -> Option<(usize, &'a ModelDef)> {
    candidates
        .iter()
        .copied()
        .max_by(|(left_idx, left), (right_idx, right)| {
            default_candidate_rank(left, *left_idx).cmp(&default_candidate_rank(right, *right_idx))
        })
}

fn default_candidate_rank(
    model: &ModelDef,
    preferred_index: usize,
) -> (ModelQualityTier, u8, usize) {
    (
        model.quality_tier,
        model.tools_default_enabled as u8,
        usize::MAX - preferred_index,
    )
}

fn quality_tier_satisfies_target(
    quality_tier: ModelQualityTier,
    target_tier: ModelStrengthTier,
) -> bool {
    match target_tier {
        ModelStrengthTier::Light => quality_tier >= ModelQualityTier::Basic,
        ModelStrengthTier::Standard => quality_tier >= ModelQualityTier::Strong,
        ModelStrengthTier::Heavy => quality_tier >= ModelQualityTier::Flagship,
    }
}

fn best_match_model_handle(
    input_lowercase: &str,
    candidates: impl Iterator<Item = (String, String)>,
    default: String,
) -> String {
    let mut best = default;
    let mut best_dist = f32::MAX;
    for (candidate, handle) in candidates {
        let candidate_lower = candidate.to_ascii_lowercase();
        let dist = levenshtein(input_lowercase.as_bytes(), candidate_lower.as_bytes()) as f32
            / (input_lowercase.len() + candidate_lower.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = handle;
        }
    }
    best
}

fn levenshtein(left: &[u8], right: &[u8]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0usize; right.len() + 1];
    for (i, left_byte) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, right_byte) in right.iter().enumerate() {
            let cost = if left_byte == right_byte { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
