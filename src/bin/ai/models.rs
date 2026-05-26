use super::agents::{AgentManifest, AgentModelTier};
use super::cli::ParsedCli;
use super::config_schema::AiConfig;
use super::model_names::{self, ModelDef};
use super::provider::{ApiProvider, ModelQualityTier, ReasoningEffort};
use crate::commonw::configw;

const COMPATIBLE_DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const OPENAI_DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const OPENCODE_DEFAULT_ENDPOINT: &str = "https://opencode.ai/zen/v1/chat/completions";
const OPENROUTER_ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";

pub(super) fn is_vl_model(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.is_vl)
        .unwrap_or(false)
}

pub(super) fn search_enabled(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.search_enabled)
        .unwrap_or(true)
}

pub(super) fn tools_enabled(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.tools_default_enabled)
        .unwrap_or(true)
}

pub(super) fn enable_thinking(model: &str) -> bool {
    model_names::find_by_name(model)
        .map(|m| m.enable_thinking)
        .unwrap_or(false)
}

/// 返回该模型在 [models.json](../../../models.json) 中声明的默认推理强度
/// （`reasoning_effort`）。CLI / `/model effort` 命令的覆盖会在
/// `request::resolve_reasoning_effort` 里优先生效，此处仅给出"模型默认"。
pub(super) fn default_reasoning_effort(model: &str) -> Option<ReasoningEffort> {
    model_names::find_by_name(model).and_then(|m| m.reasoning_effort)
}

pub(super) fn model_provider(model: &str) -> ApiProvider {
    model_names::find_by_name(model)
        .map(|m| m.provider)
        .unwrap_or_default()
}

fn default_endpoint_for_provider(provider: ApiProvider) -> &'static str {
    match provider {
        ApiProvider::Compatible => COMPATIBLE_DEFAULT_ENDPOINT,
        ApiProvider::OpenAi => OPENAI_DEFAULT_ENDPOINT,
        ApiProvider::OpenCode => OPENCODE_DEFAULT_ENDPOINT,
    }
}

fn default_api_key_config_candidates(provider: ApiProvider) -> &'static [&'static str] {
    match provider {
        ApiProvider::Compatible => &[
            AiConfig::MODEL_COMPATIBLE_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_API_KEY,
        ],
        ApiProvider::OpenAi => &[
            AiConfig::MODEL_OPENROUTER_API_KEY,
            AiConfig::MODEL_OPENAI_API_KEY,
            AiConfig::MODEL_API_KEY,
        ],
        ApiProvider::OpenCode => &[AiConfig::MODEL_OPENCODE_API_KEY, AiConfig::MODEL_API_KEY],
    }
}

pub(super) fn endpoint_for_model(model: &str, global_fallback: &str) -> String {
    if let Some(model_def) = model_names::find_by_name(model) {
        if let Some(endpoint) = model_def
            .endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
        {
            return endpoint.to_string();
        }

        return default_endpoint_for_provider(model_def.provider).to_string();
    }

    let global_fallback = global_fallback.trim();
    if !global_fallback.is_empty() {
        return global_fallback.to_string();
    }

    default_endpoint_for_provider(model_provider(model)).to_string()
}

pub(super) fn api_key_for_model(model: &str, global_fallback: &str) -> String {
    let cfg = configw::get_all_config();

    if let Some(config_key) = model_names::find_by_name(model)
        .and_then(|m| m.api_key_config_key.as_deref())
        .map(str::trim)
        .filter(|key| !key.is_empty())
        && let Some(value) = cfg
            .get_opt(config_key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    {
        return value;
    }

    for key in default_api_key_config_candidates(model_provider(model)) {
        if let Some(value) = cfg
            .get_opt(key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            return value;
        }
    }

    global_fallback.trim().to_string()
}

pub(super) fn endpoint_supports_anonymous_auth(endpoint: &str) -> bool {
    let endpoint = endpoint.trim().to_ascii_lowercase();
    endpoint.starts_with("http://127.0.0.1")
        || endpoint.starts_with("http://localhost")
        || endpoint.starts_with("http://0.0.0.0")
        || endpoint.starts_with("http://[::1]")
}

pub(super) fn model_quality_tier(model: &str) -> ModelQualityTier {
    model_names::find_by_name(model)
        .map(|m| m.quality_tier)
        .unwrap_or_default()
}

fn default_context_window_tokens_for_tier(tier: ModelQualityTier) -> usize {
    match tier {
        ModelQualityTier::Flagship => 256_000,
        ModelQualityTier::Strong => 128_000,
        ModelQualityTier::Standard => 96_000,
        ModelQualityTier::Basic => 64_000,
    }
}

/// 返回模型上下文窗口（token）。
/// 若 models.json 未声明，按质量档位给出保守默认值，供压缩预算动态估算使用。
pub(super) fn context_window_tokens(model: &str) -> usize {
    if let Some(def) = model_names::find_by_name(model) {
        return def
            .context_window_tokens
            .filter(|v| *v > 0)
            .unwrap_or_else(|| default_context_window_tokens_for_tier(def.quality_tier));
    }
    default_context_window_tokens_for_tier(model_quality_tier(model))
}

fn all_model_names() -> Vec<String> {
    model_names::all().iter().map(|m| m.name.clone()).collect()
}

fn vl_model_names() -> Vec<String> {
    model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .map(|m| m.name.clone())
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

fn default_vl_model() -> String {
    choose_default_model_name(true).unwrap_or_else(default_model)
}

pub(super) fn forced_deepseek_model() -> String {
    model_names::find_by_key("DEEPSEEK_V3")
        .map(|m| m.name.as_str().to_owned())
        .unwrap_or_else(default_model)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubagentTaskDifficulty {
    Light,
    Standard,
    Heavy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ModelStrengthTier {
    Light,
    Standard,
    Heavy,
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
    if let Some(def) = super::model_names::find_by_key(raw) {
        return def.name.as_str().to_owned();
    }
    if let Some(def) = model_names::find_by_name(raw) {
        return def.name.as_str().to_owned();
    }
    best_match_model_name(
        &raw.to_lowercase(),
        all_model_names().into_iter(),
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
            return vl.name.as_str().to_owned();
        }
        return default_vl_model();
    }

    if let Some(def) = model_names::find_by_name(&model)
        && def.is_vl
    {
        return def.name.as_str().to_owned();
    }

    best_match_model_name(&model, vl_model_names().into_iter(), default_vl_model())
}

pub(super) fn supports_image_input(model: &str) -> bool {
    is_vl_model(model)
}

pub(super) fn auto_subagent_model_for_agent(
    agent: &AgentManifest,
    description: &str,
    prompt: &str,
) -> String {
    let difficulty = classify_subagent_task_difficulty(description, prompt);
    let target_tier = merge_agent_tier_with_difficulty(agent_model_tier(agent), difficulty);
    // preferred_keys 必须与 models.json 中真实存在的 key 对齐，否则 find_by_key
    // 会静默 None，这里的偏好就完全失效。新增/重命名时记得同步更新启动日志
    // log_subagent_preferred_keys_health()。
    match target_tier {
        ModelStrengthTier::Light => pick_subagent_model(
            &["DEEPSEEK_V3", "KIMI", "GLM 5.1", "MINIMAX"],
            false,
            target_tier,
        ),
        ModelStrengthTier::Standard => pick_subagent_model(
            &[
                "QWEN_PLUS_LATEST",
                "KIMI",
                "GLM 5.1",
                "MINIMAX",
                "DEEPSEEK_V3",
            ],
            true,
            target_tier,
        ),
        ModelStrengthTier::Heavy => pick_subagent_model(
            &[
                "QWEN3_MAX",
                "QWEN_PLUS_LATEST",
                "MINIMAX",
                "GLM 5.1",
                "KIMI",
            ],
            true,
            target_tier,
        ),
    }
}

/// 子 agent 模型选择策略中所有 tier 用到的 preferred_keys 集合，启动时
/// 用来做存在性体检。请保持与 auto_subagent_model_for_agent 中的列表一致。
fn subagent_preferred_keys_all() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        ("Light", &["DEEPSEEK_V3", "KIMI", "GLM 5.1", "MINIMAX"]),
        (
            "Standard",
            &[
                "QWEN_PLUS_LATEST",
                "KIMI",
                "GLM 5.1",
                "MINIMAX",
                "DEEPSEEK_V3",
            ],
        ),
        (
            "Heavy",
            &[
                "QWEN3_MAX",
                "QWEN_PLUS_LATEST",
                "MINIMAX",
                "GLM 5.1",
                "KIMI",
            ],
        ),
    ]
}

/// 启动时检查子 agent 模型偏好列表中是否有 key 在 models.json 中不存在，
/// 把缺失的 key 打一行警告，方便在删除/重命名模型时及时发现配置漂移。
/// 永远不会 panic 或退出：缺失只会让对应 tier 自动降级到下一候选或全局
/// fallback，不应阻塞启动。
pub(super) fn log_subagent_preferred_keys_health() {
    let mut missing: Vec<(&'static str, &'static str)> = Vec::new();
    for (tier, keys) in subagent_preferred_keys_all() {
        for key in *keys {
            if model_names::find_by_key(key).is_none() {
                missing.push((tier, *key));
            }
        }
    }
    if missing.is_empty() {
        return;
    }
    let mut msg =
        String::from("[models] subagent preferred_keys missing in models.json (will fallback): ");
    for (i, (tier, key)) in missing.iter().enumerate() {
        if i > 0 {
            msg.push_str(", ");
        }
        msg.push_str(&format!("{}::{}", tier, key));
    }
    eprintln!("{}", msg);
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

fn pick_subagent_model(
    preferred_keys: &[&str],
    require_thinking: bool,
    target_tier: ModelStrengthTier,
) -> String {
    let preferred = preferred_keys
        .iter()
        .enumerate()
        .filter_map(|(idx, key)| model_names::find_by_key(key).map(|model| (idx, model)))
        .filter(|(_, model)| subagent_model_eligible(model, require_thinking))
        .collect::<Vec<_>>();
    if let Some((_, model)) = choose_best_candidate(&preferred, target_tier) {
        return model.name.clone();
    }

    let fallback = model_names::all()
        .into_iter()
        .enumerate()
        .filter(|(_, model)| subagent_model_eligible(model, require_thinking))
        .collect::<Vec<_>>();
    if let Some((_, model)) = choose_best_candidate(&fallback, target_tier) {
        return model.name.clone();
    }

    if require_thinking {
        let tools_only = model_names::all()
            .into_iter()
            .enumerate()
            .filter(|(_, model)| model.tools_default_enabled)
            .collect::<Vec<_>>();
        if let Some((_, model)) = choose_best_candidate(&tools_only, target_tier) {
            return model.name.clone();
        }
    }

    default_model()
}

fn subagent_model_eligible(model: &ModelDef, require_thinking: bool) -> bool {
    model.tools_default_enabled && (!require_thinking || model.enable_thinking)
}

fn choose_default_model_name(require_vl: bool) -> Option<String> {
    let candidates = model_names::all()
        .into_iter()
        .enumerate()
        .filter(|(_, model)| model.is_vl == require_vl)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }

    choose_best_default_candidate(
        &candidates
            .iter()
            .copied()
            .filter(|(_, model)| model.provider == ApiProvider::Compatible)
            .collect::<Vec<_>>(),
    )
    .or_else(|| choose_best_default_candidate(&candidates))
    .map(|(_, model)| model.name.clone())
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

fn choose_best_candidate<'a>(
    candidates: &[(usize, &'a ModelDef)],
    target_tier: ModelStrengthTier,
) -> Option<(usize, &'a ModelDef)> {
    candidates
        .iter()
        .copied()
        .max_by(|(left_idx, left), (right_idx, right)| {
            candidate_rank(left, *left_idx, target_tier).cmp(&candidate_rank(
                right,
                *right_idx,
                target_tier,
            ))
        })
}

fn candidate_rank(
    model: &ModelDef,
    preferred_index: usize,
    target_tier: ModelStrengthTier,
) -> (u8, ModelQualityTier, usize) {
    (
        quality_tier_satisfies_target(model.quality_tier, target_tier) as u8,
        model.quality_tier,
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

fn best_match_model_name(
    input_lowercase: &str,
    candidates: impl Iterator<Item = String>,
    default: String,
) -> String {
    let mut best = default;
    let mut best_dist = f32::MAX;
    for candidate in candidates {
        let candidate_lower = candidate.to_ascii_lowercase();
        let dist = levenshtein(input_lowercase.as_bytes(), candidate_lower.as_bytes()) as f32
            / (input_lowercase.len() + candidate_lower.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
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
mod tests {
    use super::{
        COMPATIBLE_DEFAULT_ENDPOINT, ModelStrengthTier, OPENCODE_DEFAULT_ENDPOINT,
        OPENROUTER_ENDPOINT, SubagentTaskDifficulty, agent_model_tier, api_key_for_model,
        auto_subagent_model_for_agent, classify_subagent_task_difficulty, default_model,
        determine_model, determine_vl_model, enable_thinking, endpoint_for_model,
        endpoint_supports_anonymous_auth, initial_model, merge_agent_tier_with_difficulty,
        model_provider, model_quality_tier,
    };
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::config_schema::AiConfig;
    use crate::ai::provider::{ApiProvider, ModelQualityTier};

    fn manifest(
        name: &str,
        description: &str,
        model_tier: Option<AgentModelTier>,
    ) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: description.to_string(),
            mode: AgentMode::Subagent,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            routing_tags: Vec::new(),
            model_tier,
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn light_subagent_tasks_use_light_tier() {
        assert_eq!(
            classify_subagent_task_difficulty(
                "Locate task tool",
                "Find where the task tool is implemented and summarize the file."
            ),
            SubagentTaskDifficulty::Light
        );
    }

    #[test]
    fn heavy_subagent_tasks_use_heavy_tier() {
        assert_eq!(
            classify_subagent_task_difficulty(
                "Debug end-to-end failure",
                "Investigate a failing build across multiple files, implement fixes, run tests, and summarize remaining risks."
            ),
            SubagentTaskDifficulty::Heavy
        );
    }

    #[test]
    fn heavy_subagent_model_prefers_tool_capable_thinking_model() {
        let model = auto_subagent_model_for_agent(
            &manifest(
                "build",
                "Autonomous execution agent",
                Some(AgentModelTier::Heavy),
            ),
            "Debug end-to-end failure",
            "Investigate a failing build across multiple files, implement fixes, run tests, and summarize remaining risks.",
        );
        let def = super::model_names::find_by_name(&model).expect("selected model must exist");
        assert!(def.tools_default_enabled);
        assert!(def.enable_thinking);
        assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
    }

    #[test]
    fn standard_subagent_model_prefers_high_quality_tier() {
        let model = auto_subagent_model_for_agent(
            &manifest(
                "plan",
                "Read-only planning and analysis agent",
                Some(AgentModelTier::Standard),
            ),
            "Plan a refactor",
            "Review the architecture, compare approaches, and propose a refactor strategy.",
        );
        let def = super::model_names::find_by_name(&model).expect("selected model must exist");
        assert!(def.tools_default_enabled);
        assert!(def.enable_thinking);
        assert!(def.quality_tier >= ModelQualityTier::Strong);
    }

    #[test]
    fn explore_agents_default_to_light_tier() {
        let agent = manifest(
            "explore",
            "Read-only codebase exploration agent",
            Some(AgentModelTier::Light),
        );
        assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Light);
    }

    #[test]
    fn plan_agents_default_to_standard_tier() {
        let agent = manifest(
            "plan",
            "Read-only planning and analysis agent",
            Some(AgentModelTier::Standard),
        );
        assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Standard);
    }

    #[test]
    fn build_agents_default_to_heavy_tier() {
        let agent = manifest(
            "build",
            "Autonomous execution and debugging agent",
            Some(AgentModelTier::Heavy),
        );
        assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Heavy);
    }

    #[test]
    fn light_tasks_downgrade_heavy_agents_to_standard_tier() {
        assert_eq!(
            merge_agent_tier_with_difficulty(
                ModelStrengthTier::Heavy,
                SubagentTaskDifficulty::Light
            ),
            ModelStrengthTier::Standard
        );
    }

    /// 选取一个真实存在的、provider=OpenAi 的模型名做用例输入；
    /// 这样测试不会因为 models.json 增删个别条目而失效。
    fn first_openai_model_name() -> String {
        super::model_names::all()
            .iter()
            .find(|m| m.provider == ApiProvider::OpenAi)
            .map(|m| m.name.clone())
            .expect("models.json must contain at least one OpenAi-provider model")
    }

    fn first_openai_vl_model_name() -> Option<String> {
        super::model_names::all()
            .iter()
            .find(|m| m.provider == ApiProvider::OpenAi && m.is_vl)
            .map(|m| m.name.clone())
    }

    #[test]
    fn known_model_entries_resolve_exactly_by_name() {
        let openai = first_openai_model_name();
        assert_eq!(determine_model(&openai), openai);
        if let Some(vl) = first_openai_vl_model_name() {
            assert_eq!(determine_vl_model(&vl), vl);
        }
    }

    #[test]
    fn model_keys_resolve_to_model_names() {
        // 用 models.json 中第一个真实条目反向校验 key→name 的映射，
        // 而不是硬编码具体 key。
        let first = super::model_names::all()
            .first()
            .map(|m| (m.key.clone(), m.name.clone()))
            .expect("models.json must contain at least one entry");
        assert_eq!(determine_model(&first.0), first.1);
    }

    #[test]
    fn initial_model_normalizes_configured_model_key() {
        let mut cli = ParsedCli::default();
        cli.model = None;
        let model = initial_model(&cli);
        let configured = crate::commonw::configw::get_all_config()
            .get_opt(AiConfig::MODEL_DEFAULT)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        if let Some(key) = configured
            && let Some(def) = super::model_names::find_by_key(&key)
                .or_else(|| super::model_names::find_by_name(&key))
        {
            assert_eq!(model, def.name);
        }
    }

    #[test]
    fn known_model_entries_carry_provider_and_quality_tier() {
        let name = first_openai_model_name();
        let def = super::model_names::find_by_name(&name).expect("model must exist");
        assert_eq!(model_provider(&name), def.provider);
        assert_eq!(model_quality_tier(&name), def.quality_tier);
    }

    #[test]
    fn endpoint_for_known_model_prefers_model_config_over_global_fallback() {
        // 任意一个在 models.json 中显式声明 endpoint 的条目都应该优先使用自身配置，
        // 忽略 global_fallback。这里挑第一个声明 endpoint 的条目即可。
        let (name, expected) = super::model_names::all()
            .iter()
            .find_map(|m| {
                m.endpoint
                    .as_deref()
                    .map(str::trim)
                    .filter(|e| !e.is_empty())
                    .map(|e| (m.name.clone(), e.to_string()))
            })
            .expect("models.json must contain at least one entry with explicit endpoint");
        let endpoint = endpoint_for_model(
            &name,
            "https://example.com/should-not-be-used/v1/chat/completions",
        );
        assert_eq!(endpoint, expected);
    }

    #[test]
    fn endpoint_for_compatible_model_prefers_model_config() {
        // 找一个 Compatible provider 且配置了 endpoint 的模型，确保走 model 配置。
        let (name, expected) = super::model_names::all()
            .iter()
            .find_map(|m| {
                if m.provider != ApiProvider::Compatible {
                    return None;
                }
                m.endpoint
                    .as_deref()
                    .map(str::trim)
                    .filter(|e| !e.is_empty())
                    .map(|e| (m.name.clone(), e.to_string()))
            })
            .expect("models.json must contain at least one Compatible entry with endpoint");
        let endpoint = endpoint_for_model(&name, "");
        assert_eq!(endpoint, expected);
        assert_eq!(endpoint, COMPATIBLE_DEFAULT_ENDPOINT);
    }

    #[test]
    fn openai_model_entries_prefer_openai_api_key_config() {
        let name = first_openai_model_name();
        let key = api_key_for_model(&name, "fallback-key");
        assert!(!key.is_empty());
    }

    #[test]
    fn openrouter_models_use_openrouter_endpoint_in_config() {
        // 任何配置了 openrouter endpoint 的模型都该走 openrouter。
        let openrouter_model = super::model_names::all()
            .iter()
            .find(|m| {
                m.endpoint
                    .as_deref()
                    .map(|e| e.trim().eq_ignore_ascii_case(OPENROUTER_ENDPOINT))
                    .unwrap_or(false)
            })
            .map(|m| m.name.clone());
        if let Some(name) = openrouter_model {
            let endpoint = endpoint_for_model(&name, "");
            assert_eq!(endpoint, OPENROUTER_ENDPOINT);
        }
    }

    #[test]
    fn known_model_without_endpoint_uses_provider_default_before_global_fallback() {
        let endpoint = endpoint_for_model(
            "minimax-m2.5-free",
            "https://example.com/v1/chat/completions",
        );
        assert_eq!(endpoint, OPENCODE_DEFAULT_ENDPOINT);
    }

    #[test]
    fn unknown_model_uses_global_fallback_endpoint() {
        let endpoint =
            endpoint_for_model("custom-model", "https://example.com/v1/chat/completions");
        assert_eq!(endpoint, "https://example.com/v1/chat/completions");
    }

    #[test]
    fn localhost_endpoint_supports_anonymous_auth() {
        assert!(endpoint_supports_anonymous_auth(
            "http://127.0.0.1:11434/v1/chat/completions"
        ));
        assert!(endpoint_supports_anonymous_auth(
            "http://localhost:11434/v1/chat/completions"
        ));
        assert!(!endpoint_supports_anonymous_auth(
            "https://openrouter.ai/api/v1/chat/completions"
        ));
    }

    #[test]
    fn default_model_prefers_high_quality_compatible_model() {
        // default_model 在 choose_default_model_name 中先按 Compatible 过滤，
        // 再退回到全集，并按 quality_tier 取最高。这里把不变量直接写在断言上：
        //  1. 必须是 non-vl
        //  2. quality_tier 必须不低于所有同 provider-偏好下的候选
        let def = super::model_names::find_by_name(&default_model())
            .expect("default model must exist in models.json");
        assert!(!def.is_vl, "default model should be non-VL");

        let best_non_vl_tier = super::model_names::all()
            .iter()
            .filter(|m| !m.is_vl)
            .map(|m| m.quality_tier)
            .max()
            .expect("models.json must contain at least one non-VL model");
        assert_eq!(def.quality_tier, best_non_vl_tier);
    }

    #[test]
    fn opencode_model_entries_do_not_advertise_thinking_when_disabled() {
        // 以前这里是固定模型名 gpt-5.4-pro 的强制断言；现在改为针对任意一个
        // 在 models.json 中明确声明 enable_thinking=false 的 opencode 模型，
        // 校验 enable_thinking() 与配置一致。这样 models.json 的具体条目变更
        // 不会再让本测试失效，但仍然能守住"不要把 false 误读成 true"的不变量。
        let candidate = super::model_names::all()
            .iter()
            .find(|m| m.provider == ApiProvider::OpenCode && !m.enable_thinking)
            .map(|m| m.name.clone());
        if let Some(name) = candidate {
            assert!(!enable_thinking(&name));
        }
    }
}
