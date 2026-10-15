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

/// Floor for the derived history budget: the former fixed default. Every
/// cross-turn compaction rewrites the cached prompt prefix, so a budget below
/// it would compact more often than before and lose prompt-cache hits; it also
/// keeps a tiny or misdeclared window from collapsing the budget to 0, which
/// would mean "compression disabled".
const MIN_DERIVED_HISTORY_MAX_CHARS: usize = 200_000;

/// Cross-turn live-history budget in chars for `model`.
///
/// An explicit `ai.history.max_chars` always wins (0 keeps compression
/// disabled). Otherwise the budget is two thirds of the model's static soft
/// target, leaving the remaining third for the system prompt, tool schemas and
/// the current turn's growth before request-boundary compression starts, and
/// never below [`MIN_DERIVED_HISTORY_MAX_CHARS`]. Large windows thus keep far
/// more history resident (fewer prefix rewrites) while small windows keep the
/// former compaction cadence. The static default chars-per-token ratio is
/// deliberate: the value is part of the projection fingerprint, so a
/// per-request measured ratio would invalidate persisted snapshots every turn.
pub(super) fn history_max_chars_for_model(cfg: &configw::ConfigW, model: &str) -> usize {
    cfg.get_opt(AiConfig::HISTORY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            crate::ai::request::soft_target_chars_for_model(model)
                .saturating_mul(2)
                .saturating_div(3)
                .max(MIN_DERIVED_HISTORY_MAX_CHARS)
        })
}

#[allow(dead_code)]
pub(super) fn clear_history_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
