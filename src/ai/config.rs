use std::{
    fs,
    fs::{File, OpenOptions},
    io,
    path::PathBuf,
};

use crate::common::{configw, utils::expanduser};

use super::{models, types::AppConfig};

const QWEN_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";

pub(super) fn load_config() -> Result<AppConfig, Box<dyn std::error::Error>> {
    let cfg = configw::get_all_config();
    let api_key = cfg.get_opt("api_key").unwrap_or_default();
    if api_key.trim().is_empty() {
        println!("set api_key in ~/.configW");
        std::process::exit(0);
    }
    let history_file = cfg
        .get_opt("history_file")
        .unwrap_or_else(|| "~/.history_file.txt".to_string());
    let endpoint = cfg
        .get_opt("ai.model.endpoint")
        .unwrap_or_else(|| QWEN_ENDPOINT.to_string());
    let vl_default_model =
        models::determine_vl_model(&cfg.get_opt("ai.model.vl_default").unwrap_or_default());
    Ok(AppConfig {
        api_key,
        history_file: PathBuf::from(expanduser(&history_file).as_ref()),
        endpoint,
        vl_default_model,
    })
}

pub(super) fn open_output_writer(path: Option<&str>) -> io::Result<Option<File>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o644);
    }
    options.open(path).map(Some)
}

pub(super) fn clear_history_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}
