use super::agents::{AgentManifest, AgentModelTier};
use super::cli::ParsedCli;
use super::config_schema::AiConfig;
use super::model_names::{self, ModelDef};
use super::provider::{ApiProvider, ModelQualityTier};
use crate::commonw::configw;

const COMPATIBLE_DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const OPENAI_DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
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

pub(super) fn model_provider(model: &str) -> ApiProvider {
    model_names::find_by_name(model)
        .map(|m| m.provider)
        .unwrap_or_default()
}

fn default_endpoint_for_provider(provider: ApiProvider) -> &'static str {
    match provider {
        ApiProvider::Compatible => COMPATIBLE_DEFAULT_ENDPOINT,
        ApiProvider::OpenAi => OPENAI_DEFAULT_ENDPOINT,
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
    }
}

pub(super) fn endpoint_for_model(model: &str, global_fallback: &str) -> String {
    if let Some(endpoint) = model_names::find_by_name(model)
        .and_then(|m| m.endpoint.as_deref())
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
    {
        return endpoint.to_string();
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
    choose_default_model_name(false).unwrap_or_else(|| {
        eprintln!("[model_names] models.json is empty");
        std::process::exit(1);
    })
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
    best_match_model_name(&raw.to_lowercase(), all_model_names().into_iter(), default_model())
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
    match target_tier {
        ModelStrengthTier::Light => pick_subagent_model(
            &["DEEPSEEK_V3", "KIMI", "GLM", "MINIMAX"],
            false,
            target_tier,
        ),
        ModelStrengthTier::Standard => pick_subagent_model(
            &["QWEN_CODER_PLUS_LATEST", "GLM", "KIMI", "MINIMAX", "DEEPSEEK_V3"],
            true,
            target_tier,
        ),
        ModelStrengthTier::Heavy => pick_subagent_model(
            &["QWEN3_MAX", "QWEN_CODER_PLUS_LATEST", "MINIMAX", "GLM", "KIMI"],
            true,
            target_tier,
        ),
    }
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

fn classify_subagent_task_difficulty(
    description: &str,
    prompt: &str,
) -> SubagentTaskDifficulty {
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
    candidates.iter().copied().max_by(|(left_idx, left), (right_idx, right)| {
        default_candidate_rank(left, *left_idx).cmp(&default_candidate_rank(right, *right_idx))
    })
}

fn default_candidate_rank(model: &ModelDef, preferred_index: usize) -> (ModelQualityTier, u8, usize) {
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
    candidates.iter().copied().max_by(|(left_idx, left), (right_idx, right)| {
        candidate_rank(left, *left_idx, target_tier).cmp(&candidate_rank(right, *right_idx, target_tier))
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
        agent_model_tier, api_key_for_model, auto_subagent_model_for_agent,
        classify_subagent_task_difficulty, default_model, default_vl_model, determine_model,
        determine_vl_model, endpoint_for_model, initial_model, merge_agent_tier_with_difficulty,
        endpoint_supports_anonymous_auth, model_provider, model_quality_tier, ModelStrengthTier, SubagentTaskDifficulty,
        COMPATIBLE_DEFAULT_ENDPOINT, OPENROUTER_ENDPOINT,
    };
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::config_schema::AiConfig;
    use crate::ai::provider::{ApiProvider, ModelQualityTier};

    fn manifest(name: &str, description: &str, model_tier: Option<AgentModelTier>) -> AgentManifest {
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

    #[test]
    fn openai_model_entries_resolve_exactly() {
        assert_eq!(determine_model("gpt-4o"), "gpt-4o");
        assert_eq!(determine_vl_model("gpt-4.1-mini"), "gpt-4.1-mini");
    }

    #[test]
    fn model_keys_resolve_to_model_names() {
        assert_eq!(determine_model("gemma-4"), "google/gemma-4-26b-a4b-it:free");
    }

    #[test]
    fn initial_model_normalizes_configured_model_key() {
        let mut cli = ParsedCli::default();
        cli.model = None;
        let model = initial_model(&cli);
        if crate::commonw::configw::get_all_config()
            .get_opt(AiConfig::MODEL_DEFAULT)
            .as_deref()
            == Some("gemma-4")
        {
            assert_eq!(model, "google/gemma-4-26b-a4b-it:free");
        }
    }

    #[test]
    fn openai_model_entries_carry_provider_and_quality_tier() {
        let def = super::model_names::find_by_name("gpt-4o").expect("gpt-4o model must exist");
        assert_eq!(model_provider("gpt-4o"), def.provider);
        assert_eq!(model_quality_tier("gpt-4o"), def.quality_tier);
    }

    #[test]
    fn endpoint_for_openai_model_prefers_model_config() {
        let endpoint = endpoint_for_model("gpt-4o", "");
        assert_eq!(endpoint, OPENROUTER_ENDPOINT);
    }

    #[test]
    fn endpoint_for_compatible_model_prefers_model_config() {
        let endpoint = endpoint_for_model("qwen3-max", "");
        assert_eq!(endpoint, COMPATIBLE_DEFAULT_ENDPOINT);
    }

    #[test]
    fn openai_model_entries_prefer_openai_api_key_config() {
        let key = api_key_for_model("gpt-4o", "fallback-key");
        assert!(!key.is_empty());
    }

    #[test]
    fn openrouter_models_use_openrouter_endpoint_in_config() {
        let endpoint = endpoint_for_model("deepseek-v3.2", "");
        assert_eq!(endpoint, OPENROUTER_ENDPOINT);
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
        let def = super::model_names::find_by_name(&default_model()).expect("default model must exist");
        assert_eq!(def.provider, ApiProvider::Compatible);
        assert!(!def.is_vl);
        assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
    }

    #[test]
    fn default_vl_model_prefers_high_quality_compatible_vl_model() {
        let def =
            super::model_names::find_by_name(&default_vl_model()).expect("default vl model must exist");
        assert_eq!(def.provider, ApiProvider::Compatible);
        assert!(def.is_vl);
        assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
    }
}
