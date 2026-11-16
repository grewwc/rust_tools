use std::{fs, path::PathBuf};

use crate::ai::config_schema::AiConfig;
use crate::commonw::{configw, utils::expanduser};

use super::{models, types::AppConfig};

pub(super) fn load_config() -> Result<AppConfig, Box<dyn std::error::Error>> {
    let cfg = configw::get_all_config();
    let api_key = cfg.get_opt("api_key").unwrap_or_default();
    let opencode_api_key = cfg
        .get_opt(AiConfig::MODEL_OPENCODE_API_KEY)
        .unwrap_or_default();
    let openrouter_api_key = cfg
        .get_opt(AiConfig::MODEL_OPENROUTER_API_KEY)
        .unwrap_or_default();
    let compatible_api_key = cfg
        .get_opt(AiConfig::MODEL_COMPATIBLE_API_KEY)
        .unwrap_or_default();
    let alibaba_api_key = cfg
        .get_opt(AiConfig::MODEL_ALIBABA_API_KEY)
        .unwrap_or_default();
    let aliyun_api_key = cfg
        .get_opt(AiConfig::MODEL_ALIYUN_API_KEY)
        .unwrap_or_default();
    let openai_api_key = cfg
        .get_opt(AiConfig::MODEL_OPENAI_API_KEY)
        .unwrap_or_default();
    let default_model =
        models::determine_model(&cfg.get_opt(AiConfig::MODEL_DEFAULT).unwrap_or_default());
    let default_endpoint = models::endpoint_for_model(&default_model, "");
    let default_model_api_key = models::api_key_for_model(&default_model, &api_key);
    if api_key.trim().is_empty()
        && opencode_api_key.trim().is_empty()
        && openrouter_api_key.trim().is_empty()
        && compatible_api_key.trim().is_empty()
        && alibaba_api_key.trim().is_empty()
        && aliyun_api_key.trim().is_empty()
        && openai_api_key.trim().is_empty()
        && default_model_api_key.trim().is_empty()
        && !models::endpoint_supports_anonymous_auth(&default_endpoint)
    {
        return Err("set api_key / opencode.api_key / openrouter.api_key / compatible.api_key / alibaba.api_key / aliyun.api_key / openai.api_key or the default model's `api_key_config_key` in ~/.configW".into());
    }
    let history_file = cfg
        .get_opt("history_file")
        .unwrap_or_else(|| "~/.history_file.sqlite".to_string());
    let vl_default_model =
        models::determine_vl_model(&cfg.get_opt(AiConfig::MODEL_VL_DEFAULT).unwrap_or_default());
    let history_max_chars = history_max_chars_for_model(&cfg, &default_model);
    let history_keep_last = cfg
        .get_opt(AiConfig::HISTORY_KEEP_LAST)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    let history_summary_max_chars = cfg
        .get_opt(AiConfig::HISTORY_SUMMARY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000);
    let intent_model = cfg.get_opt(AiConfig::INTENT_MODEL);
    let history_file = PathBuf::from(expanduser(&history_file).as_ref());
    Ok(AppConfig {
        api_key,
        base_history_file: history_file.clone(),
        history_file,
        endpoint: String::new(),
        vl_default_model,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        intent_model,
    })
}

/// Fixed cross-turn live-history budget in chars when `ai.history.max_chars` is
/// unset. Deriving the budget from the model window made large-window models
/// (1M-token deepseek/minimax) keep ~2x more history resident every turn, which
/// doubled per-request input tokens without a matching quality gain. A single
/// fixed value keeps the compaction cadence identical across models and
/// independent of the declared window.
const DEFAULT_HISTORY_MAX_CHARS: usize = 200_000;

/// Cross-turn live-history budget in chars for `model`.
///
/// An explicit `ai.history.max_chars` always wins (0 keeps compression
/// disabled). Otherwise every model uses the same fixed
/// [`DEFAULT_HISTORY_MAX_CHARS`]: the budget no longer scales with the model
/// window, so a large declared window cannot inflate resident history and the
/// per-request token cost stays stable across models. The `model` argument is
/// retained for call-site compatibility and a future per-model override.
pub(super) fn history_max_chars_for_model(cfg: &configw::ConfigW, _model: &str) -> usize {
    cfg.get_opt(AiConfig::HISTORY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_HISTORY_MAX_CHARS)
}

/// Parse an on/off configuration switch. Returns `None` for unset or
/// unrecognized values so callers can apply their own default.
fn parse_switch(raw: Option<&str>) -> Option<bool> {
    match raw?.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enable" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

/// A snapshot of the opt-in agent-team optimization switches. Raw switches
/// stay private so every optimization accessor enforces the master switch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AgentTeamOptimizations {
    optimizations_enabled: bool,
    dedupe_base_prompt: bool,
    lazy_group_schemas: bool,
    persist_status_cursor: bool,
    merge_ready_results: bool,
    auto_mount_lifecycle_tools: bool,
}

impl AgentTeamOptimizations {
    /// Unset or invalid values default to off, using the existing switch parser.
    pub(super) fn from_config(cfg: &configw::ConfigW) -> Self {
        let enabled = |key| parse_switch(cfg.get_opt(key).as_deref()).unwrap_or(false);
        Self {
            optimizations_enabled: enabled(AiConfig::AGENT_TEAM_OPTIMIZATIONS_ENABLED),
            dedupe_base_prompt: enabled(AiConfig::AGENT_TEAM_DEDUPE_BASE_PROMPT),
            lazy_group_schemas: enabled(AiConfig::AGENT_TEAM_LAZY_GROUP_SCHEMAS),
            persist_status_cursor: enabled(AiConfig::AGENT_TEAM_PERSIST_STATUS_CURSOR),
            merge_ready_results: enabled(AiConfig::AGENT_TEAM_MERGE_READY_RESULTS),
            auto_mount_lifecycle_tools: enabled(AiConfig::AGENT_TEAM_AUTO_MOUNT_LIFECYCLE_TOOLS),
        }
    }

    pub(super) fn optimizations_enabled(&self) -> bool {
        self.optimizations_enabled
    }

    pub(super) fn dedupe_base_prompt(&self) -> bool {
        self.optimizations_enabled && self.dedupe_base_prompt
    }

    pub(super) fn lazy_group_schemas(&self) -> bool {
        self.optimizations_enabled && self.lazy_group_schemas
    }

    pub(super) fn persist_status_cursor(&self) -> bool {
        self.optimizations_enabled && self.persist_status_cursor
    }

    pub(super) fn merge_ready_results(&self) -> bool {
        self.optimizations_enabled && self.merge_ready_results
    }

    pub(super) fn auto_mount_lifecycle_tools(&self) -> bool {
        self.optimizations_enabled && self.auto_mount_lifecycle_tools
    }
}

/// Read one consistent snapshot from the current process's cached configuration.
pub(super) fn agent_team_optimizations() -> AgentTeamOptimizations {
    AgentTeamOptimizations::from_config(&configw::get_all_config())
}

impl AppConfig {
    /// Read the current persisted switches, including changes after a config refresh.
    pub(super) fn agent_team_optimizations(&self) -> AgentTeamOptimizations {
        agent_team_optimizations()
    }
}

/// Whether the `ai.prompt.pure_mode` configuration key requests the pure
/// agent as the startup default. Unset or invalid config means off, so a
/// missing configuration can never silently strip the normal system prompt.
/// Explicit agent selection (`--agent`, `--pure`) and `--no-pure` are applied
/// by the caller; this function only reads the persisted default.
pub(super) fn pure_startup_default(cfg: &configw::ConfigW) -> bool {
    parse_switch(cfg.get_opt(AiConfig::PURE_ENABLE).as_deref()).unwrap_or(false)
}

/// Pure-agent startup default for the current process, reading the cached
/// configuration.
pub(super) fn pure_startup_default_enabled() -> bool {
    pure_startup_default(&configw::get_all_config())
}

#[allow(dead_code)]
pub(super) fn clear_history_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
