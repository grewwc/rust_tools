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
    let endpoint = cfg.get_opt(AiConfig::MODEL_ENDPOINT).unwrap_or_default();
    let default_model =
        models::determine_model(&cfg.get_opt(AiConfig::MODEL_DEFAULT).unwrap_or_default());
    let default_endpoint = models::endpoint_for_model(&default_model, &endpoint);
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
    // Live-history budget in chars (~50K tokens at ~4 chars/token). The former
    // 90_000 default folded tool results into archive pointers after only a few
    // read-heavy rounds, which forced the model to re-read files whose results it
    // had just received; 200_000 keeps roughly 10x more tool output resident while
    // still leaving headroom on 256K+-token windows after system prompt and
    // reserved output.
    let history_max_chars = cfg
        .get_opt(AiConfig::HISTORY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(200_000);
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
        endpoint,
        vl_default_model,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        intent_model,
    })
}

#[allow(dead_code)]
pub(super) fn clear_history_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
