use std::{fs, fs::File, io, path::Path, path::PathBuf};

use crate::ai::config_schema::AiConfig;
use crate::commonw::{
    configw,
    utils::{expanduser, open_file_for_write_truncate},
};

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
    let aliyun_api_key = cfg.get_opt(AiConfig::MODEL_ALIYUN_API_KEY).unwrap_or_default();
    let openai_api_key = cfg.get_opt(AiConfig::MODEL_OPENAI_API_KEY).unwrap_or_default();
    let endpoint = cfg
        .get_opt(AiConfig::MODEL_ENDPOINT)
        .unwrap_or_default();
    let default_model =
        models::determine_model(&cfg.get_opt(AiConfig::MODEL_DEFAULT).unwrap_or_default());
    let default_endpoint = models::endpoint_for_model(&default_model, &endpoint);
    if api_key.trim().is_empty()
        && opencode_api_key.trim().is_empty()
        && openrouter_api_key.trim().is_empty()
        && compatible_api_key.trim().is_empty()
        && aliyun_api_key.trim().is_empty()
        && openai_api_key.trim().is_empty()
        && !models::endpoint_supports_anonymous_auth(&default_endpoint)
    {
        println!("set api_key / opencode.api_key / openrouter.api_key / compatible.api_key / aliyun.api_key / openai.api_key in ~/.configW");
        std::process::exit(0);
    }
    let history_file = cfg
        .get_opt("history_file")
        .unwrap_or_else(|| "~/.history_file.sqlite".to_string());
    let vl_default_model =
        models::determine_vl_model(&cfg.get_opt(AiConfig::MODEL_VL_DEFAULT).unwrap_or_default());
    let history_max_chars = cfg
        .get_opt(AiConfig::HISTORY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24000);
    let history_keep_last = cfg
        .get_opt(AiConfig::HISTORY_KEEP_LAST)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    let history_summary_max_chars = cfg
        .get_opt(AiConfig::HISTORY_SUMMARY_MAX_CHARS)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000);
    let intent_model = cfg.get_opt(AiConfig::INTENT_MODEL);
    let intent_model_path = cfg
        .get_opt(AiConfig::INTENT_MODEL_PATH)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("config/intent/intent_model.json")
                .display()
                .to_string()
        });
    Ok(AppConfig {
        api_key,
        history_file: PathBuf::from(expanduser(&history_file).as_ref()),
        endpoint,
        vl_default_model,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        intent_model,
        intent_model_path: PathBuf::from(expanduser(&intent_model_path).as_ref()),
    })
}

pub(super) fn open_output_writer(path: Option<&str>) -> io::Result<Option<File>> {
    let Some(path) = path else {
        return Ok(None);
    };
    open_file_for_write_truncate(Path::new(path), 0o644).map(Some)
}

#[allow(dead_code)]
pub(super) fn clear_history_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}
