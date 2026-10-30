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
