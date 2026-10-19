use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::file_store::current_session_assets_dir;

const DEFAULT_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";
const DEFAULT_MODEL: &str = "qwen3-tts-flash";
const DEFAULT_VOICE: &str = "Cherry";
const MAX_TEXT_CHARS: usize = 600;
const MAX_AUDIO_BYTES: u64 = 10 * 1024 * 1024;
const MAX_ERROR_CHARS: usize = 2_000;

pub(crate) fn execute_generate_speech(args: &Value) -> Result<String, String> {
    let text = required_text(args, "text", MAX_TEXT_CHARS)?;
    let cfg = crate::commonw::configw::get_all_config();
    let speech_key = cfg.get(AiConfig::SPEECH_API_KEY, "");
    let alibaba_key = cfg.get(AiConfig::MODEL_ALIBABA_API_KEY, "");
    let aliyun_key = cfg.get(AiConfig::MODEL_ALIYUN_API_KEY, "");
    let api_key = first_nonempty([&speech_key, &alibaba_key, &aliyun_key]).ok_or_else(|| {
        "speech generation requires ai.speech.api_key, alibaba.api_key, or aliyun.api_key; an OpenCode Go key cannot call DashScope TTS".to_string()
    })?;
    let endpoint = validate_endpoint(&cfg.get(AiConfig::SPEECH_ENDPOINT, DEFAULT_ENDPOINT))?;
    let model = optional_text(args, "model", 100)?
        .unwrap_or_else(|| cfg.get(AiConfig::SPEECH_MODEL, DEFAULT_MODEL));
    if !matches!(
        model.as_str(),
        "qwen3-tts-flash" | "qwen3-tts-instruct-flash"
    ) {
        return Err("model must be qwen3-tts-flash or qwen3-tts-instruct-flash".to_string());
    }
    let voice = optional_text(args, "voice", 100)?
        .unwrap_or_else(|| cfg.get(AiConfig::SPEECH_VOICE, DEFAULT_VOICE));
    let language_type = optional_text(args, "language_type", 40)?
        .or_else(|| nonempty(cfg.get(AiConfig::SPEECH_LANGUAGE_TYPE, "")));
    let instructions = optional_text(args, "instructions", 300)?;
    if model == "qwen3-tts-instruct-flash" && instructions.is_none() {
        return Err("instructions is required for qwen3-tts-instruct-flash".to_string());
    }

    let request = build_request(
        text,
        &model,
        &voice,
        language_type.as_deref(),
        instructions.as_deref(),
    );
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("failed to initialize speech client: {e}"))?;
    let response = client
        .post(endpoint)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .map_err(|e| format!("speech request failed: {e}"))?;
    let status = response.status();
    let response_text = response
        .text()
        .map_err(|e| format!("failed to read speech response: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "speech API returned HTTP {status}: {}",
            truncate_chars(&response_text, MAX_ERROR_CHARS)
        ));
    }
    let response_json: Value = serde_json::from_str(&response_text)
        .map_err(|e| format!("speech API returned invalid JSON: {e}"))?;
    let audio_url = normalize_audio_url(response_audio_url(&response_json)?)?;
    let (bytes, content_type) = download_audio(&client, audio_url.clone())?;
    let extension = audio_extension(audio_url.as_str(), content_type.as_deref());
    let assets = current_session_assets_dir()
        .ok_or_else(|| "speech generation requires an active session".to_string())?;
    let file_name = write_session_audio(&assets, &extension, &bytes)?;

    Ok(format!(
        "Generated speech with {model}. Session audio: {file_name}\nInclude the file name in the response so serve-chat renders its manual play control."
    ))
}

fn build_request(
    text: &str,
    model: &str,
    voice: &str,
    language_type: Option<&str>,
    instructions: Option<&str>,
) -> Value {
    let mut input = json!({ "text": text, "voice": voice });
    if let Some(language_type) = language_type {
        input["language_type"] = Value::String(language_type.to_string());
    }
    if let Some(instructions) = instructions {
        input["instructions"] = Value::String(instructions.to_string());
    }
    json!({ "model": model, "input": input })
}

fn validate_endpoint(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw.trim()).map_err(|e| format!("invalid ai.speech.endpoint: {e}"))?;
    let allowed_host = matches!(
        url.host_str(),
        Some("dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com")
    );
    if url.scheme() != "https"
        || !allowed_host
        || url.path() != "/api/v1/services/aigc/multimodal-generation/generation"
    {
        return Err("ai.speech.endpoint must be the HTTPS DashScope multimodal-generation endpoint; OpenCode Go chat endpoints do not support TTS".to_string());
    }
    Ok(url)
}

fn response_audio_url(response: &Value) -> Result<&str, String> {
    response
        .pointer("/output/audio/url")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            let detail = response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("missing output.audio.url");
            format!(
                "speech API did not return audio: {}",
                truncate_chars(detail, MAX_ERROR_CHARS)
            )
        })
}

fn normalize_audio_url(raw_url: &str) -> Result<Url, String> {
    let url = Url::parse(raw_url).map_err(|e| format!("invalid audio URL from speech API: {e}"))?;
    let trusted_host = url
        .host_str()
        .is_some_and(|host| host == "aliyuncs.com" || host.ends_with(".aliyuncs.com"));
    if !trusted_host {
        return Err("speech API returned an audio URL outside aliyuncs.com".to_string());
    }
    if url.scheme() == "https" {
        return Ok(url);
    }
    if url.scheme() != "http" {
        return Err("speech API returned an unsupported audio URL scheme".to_string());
    }

    let mut url = url;
    url.set_scheme("https")
        .map_err(|_| "failed to upgrade speech audio URL to HTTPS".to_string())?;
    Ok(url)
}

fn download_audio(client: &Client, url: Url) -> Result<(Vec<u8>, Option<String>), String> {
    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("failed to download generated speech: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "failed to download generated speech: HTTP {} (redirects are not followed)",
            response.status()
        ));
    }
    if response
        .content_length()
        .is_some_and(|len| len > MAX_AUDIO_BYTES)
    {
        return Err(format!(
            "generated speech exceeds the {} byte limit",
            MAX_AUDIO_BYTES
        ));
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let mut bytes = Vec::new();
    response
        .take(MAX_AUDIO_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read generated speech: {e}"))?;
    if bytes.len() as u64 > MAX_AUDIO_BYTES {
        return Err(format!(
            "generated speech exceeds the {} byte limit",
            MAX_AUDIO_BYTES
        ));
    }
    if bytes.is_empty() {
        return Err("speech API returned an empty audio file".to_string());
    }
    Ok((bytes, content_type))
}

fn write_session_audio(assets: &Path, extension: &str, bytes: &[u8]) -> Result<String, String> {
    std::fs::create_dir_all(assets)
        .map_err(|e| format!("failed to create session assets directory: {e}"))?;
    let file_name = format!("speech-{}.{}", Uuid::new_v4(), extension);
    let path: PathBuf = assets.join(&file_name);
    std::fs::write(&path, bytes).map_err(|e| format!("failed to save generated speech: {e}"))?;
    Ok(file_name)
}

fn audio_extension(raw_url: &str, content_type: Option<&str>) -> String {
    if let Some(ext) = content_type.and_then(extension_for_content_type) {
        return ext.to_string();
    }
    Url::parse(raw_url)
        .ok()
        .and_then(|url| url.path_segments()?.next_back().map(str::to_string))
        .and_then(|name| {
            Path::new(&name)
                .extension()?
                .to_str()
                .map(str::to_ascii_lowercase)
        })
        .filter(|ext| {
            matches!(
                ext.as_str(),
                "mp3" | "wav" | "m4a" | "aac" | "ogg" | "flac" | "opus"
            )
        })
        .unwrap_or_else(|| "mp3".to_string())
}

fn extension_for_content_type(content_type: &str) -> Option<&'static str> {
    match content_type.split(';').next()?.trim() {
        "audio/mpeg" => Some("mp3"),
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/mp4" => Some("m4a"),
        "audio/aac" => Some("aac"),
        "audio/ogg" => Some("ogg"),
        "audio/flac" => Some("flac"),
        _ => None,
    }
}

fn required_text<'a>(args: &'a Value, field: &str, max_chars: usize) -> Result<&'a str, String> {
    let value = args
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    validate_text(field, value, max_chars)?;
    if value.is_empty() {
        return Err(format!("{field} is required"));
    }
    Ok(value)
}

fn optional_text(args: &Value, field: &str, max_chars: usize) -> Result<Option<String>, String> {
    let Some(value) = args.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    let value = value.trim();
    validate_text(field, value, max_chars)?;
    Ok(nonempty(value.to_string()))
}

fn validate_text(field: &str, value: &str, max_chars: usize) -> Result<(), String> {
    if value.chars().any(char::is_control) {
        return Err(format!("{field} must not contain control characters"));
    }
    let count = value.chars().count();
    if count > max_chars {
        return Err(format!(
            "{field} is too long: {count} characters (maximum {max_chars})"
        ));
    }
    Ok(())
}

fn first_nonempty<'a>(values: impl IntoIterator<Item = &'a String>) -> Option<&'a str> {
    values
        .into_iter()
        .map(String::as_str)
        .map(str::trim)
        .find(|value| !value.is_empty())
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_string())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_matches_dashscope_tts_shape() {
        assert_eq!(
            build_request("你好", "qwen3-tts-flash", "Cherry", Some("Chinese"), None),
            json!({
                "model": "qwen3-tts-flash",
                "input": { "text": "你好", "voice": "Cherry", "language_type": "Chinese" }
            })
        );
    }

    #[test]
    fn opencode_go_endpoint_is_not_accepted_as_speech_endpoint() {
        assert!(validate_endpoint(DEFAULT_ENDPOINT).is_ok());
        assert!(validate_endpoint("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation").is_ok());
        assert!(validate_endpoint("https://opencode.ai/inference/go/v1/chat/completions").is_err());
    }

    #[test]
    fn response_requires_audio_url() {
        assert_eq!(
            response_audio_url(&json!({"output": {"audio": {"url": "https://example.com/a.wav"}}}))
                .unwrap(),
            "https://example.com/a.wav"
        );
        assert!(
            response_audio_url(&json!({"message": "bad voice"}))
                .unwrap_err()
                .contains("bad voice")
        );
    }

    #[test]
    fn dashscope_http_audio_url_is_upgraded_to_https() {
        let url = normalize_audio_url(
            "http://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/audio.wav?token=redacted",
        )
        .expect("normalize DashScope audio URL");
        assert_eq!(url.scheme(), "https");
        assert_eq!(
            url.host_str(),
            Some("dashscope-result-sh.oss-cn-shanghai.aliyuncs.com")
        );
        assert_eq!(url.query(), Some("token=redacted"));
    }

    #[test]
    fn speech_audio_url_rejects_untrusted_hosts() {
        assert!(normalize_audio_url("http://127.0.0.1/audio.wav").is_err());
        assert!(normalize_audio_url("https://example.com/audio.wav").is_err());
    }

    #[test]
    fn writes_collision_resistant_session_audio() {
        let dir = std::env::temp_dir().join(format!("a-speech-test-{}", Uuid::new_v4()));
        let name = write_session_audio(&dir, "mp3", b"audio").expect("write audio");
        assert!(name.starts_with("speech-"));
        assert!(name.ends_with(".mp3"));
        assert_eq!(
            std::fs::read(dir.join(&name)).expect("read audio"),
            b"audio"
        );
        std::fs::remove_dir_all(dir).expect("cleanup");
    }
}
