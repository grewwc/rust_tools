use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use base64::Engine as _;
use colored::Colorize;
use reqwest::blocking::{Client, Response, multipart};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    files,
    history::{Message, build_message_arr},
    models,
    types::App,
};

const FILES_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/files";

#[derive(Debug, Serialize)]
struct RequestBody {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChunk {
    #[serde(default)]
    pub(super) choices: Vec<StreamChoice>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChoice {
    #[serde(default)]
    pub(super) delta: StreamDelta,
    #[serde(default)]
    pub(super) finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamDelta {
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) content: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) reasoning_content: String,
    #[serde(default)]
    pub(super) tool_calls: Vec<StreamToolCall>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamToolCall {
    #[serde(default)]
    pub(super) index: usize,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) id: String,
    #[serde(rename = "type", default, deserialize_with = "string_or_default")]
    pub(super) tool_type: String,
    #[serde(default)]
    pub(super) function: StreamFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamFunctionCall {
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) name: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) arguments: String,
}

fn string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct UploadResponse {
    #[serde(default)]
    id: String,
}

pub(super) fn do_request(
    app: &mut App,
    model: &str,
    question: &str,
    history_count: usize,
) -> Result<Response, Box<dyn std::error::Error>> {
    let (tools_value, tool_choice) = agent_tools_for_request(app);
    let mut request_body = RequestBody {
        model: model.to_string(),
        messages: vec![Message {
            role: "system".to_string(),
            content: Value::String("You are a helpful assistant.".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: true,
        enable_thinking: app.cli.thinking,
        enable_search: models::search_enabled(model).then_some(true),
        tools: tools_value,
        tool_choice,
    };

    let should_use_long_upload = (!app.attached_binary_files.is_empty()
        || !app.uploaded_file_ids.is_empty())
        && !models::is_vl_model(model);

    if should_use_long_upload {
        request_body.model = models::qwen_long().to_string();

        let mut files_to_upload = app.attached_binary_files.clone();
        if !app.attached_image_files.is_empty() {
            files_to_upload.extend(app.attached_image_files.iter().cloned());
        }

        if !files_to_upload.is_empty() {
            app.uploaded_file_ids =
                upload_qwen_long_files(&app.client, &app.config.api_key, &files_to_upload)?;
            app.attached_binary_files.clear();
            app.attached_image_files.clear();
        }

        let file_ids = app.uploaded_file_ids.join(",");
        request_body.messages.push(Message {
            role: "system".to_string(),
            content: Value::String(format!("fileid://{file_ids}")),
            tool_calls: None,
            tool_call_id: None,
        });
    } else if !models::is_vl_model(model) {
        request_body
            .messages
            .extend(build_message_arr(history_count, &app.config.history_file)?);
    }

    request_body.messages.push(Message {
        role: "user".to_string(),
        content: build_content(&request_body.model, question, &app.attached_image_files)?,
        tool_calls: None,
        tool_call_id: None,
    });

    let response = app
        .client
        .post(&app.config.endpoint)
        .bearer_auth(&app.config.api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("request failed: {status} {body}").into());
    }

    if models::is_vl_model(&request_body.model) {
        app.attached_image_files.clear();
    }

    Ok(response)
}

pub(super) fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: Vec<Message>,
    stream: bool,
) -> Result<Response, Box<dyn std::error::Error>> {
    let (tools_value, tool_choice) = agent_tools_for_request(app);
    let request_body = RequestBody {
        model: model.to_string(),
        messages,
        stream,
        enable_thinking: app.cli.thinking,
        enable_search: models::search_enabled(model).then_some(true),
        tools: tools_value,
        tool_choice,
    };

    let response = app
        .client
        .post(&app.config.endpoint)
        .bearer_auth(&app.config.api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("request failed: {status} {body}").into());
    }

    Ok(response)
}

fn agent_tools_for_request(app: &App) -> (Option<Value>, Option<Value>) {
    let Some(ctx) = app.agent_context.as_ref() else {
        return (None, None);
    };
    if ctx.tools.is_empty() {
        return (None, None);
    }
    let tools_value = serde_json::to_value(&ctx.tools).ok();
    let tool_choice = tools_value
        .as_ref()
        .map(|_| Value::String("auto".to_string()));
    (tools_value, tool_choice)
}

pub(super) fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !models::is_vl_model(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = files::image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": format!("data:{mime};base64,{image}"),
        }));
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

pub(super) fn print_info(model: &str) {
    let search = if models::search_enabled(model) {
        "true"
    } else {
        "false"
    };
    // 使用println!避免手动flush的权限问题
    println!("[{} (search: {})]", model.green(), search.red());
}

fn upload_qwen_long_files(
    client: &Client,
    api_key: &str,
    files: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut ids = Vec::with_capacity(files.len());
    for file in files {
        ids.push(upload_single_qwen_long_file_with_retry(
            client, api_key, file, 5,
        )?);
    }
    Ok(ids)
}

fn upload_single_qwen_long_file_with_retry(
    client: &Client,
    api_key: &str,
    filename: &str,
    retry: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut last_err: Option<Box<dyn std::error::Error>> = None;
    for _ in 0..retry {
        match upload_single_qwen_long_file(client, api_key, filename) {
            Ok(id) if !id.is_empty() => return Ok(id),
            Ok(_) => last_err = Some("empty file id".into()),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| "upload failed".into()))
}

fn upload_single_qwen_long_file(
    client: &Client,
    api_key: &str,
    filename: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = PathBuf::from(filename);
    let display_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(filename);
    println!("Uploading file: {display_name}");

    let bytes = fs::read(filename)?;
    let part = multipart::Part::bytes(bytes).file_name(display_name.to_string());
    let form = multipart::Form::new()
        .part("file", part)
        .text("purpose", "file-extract");

    let response = client
        .post(FILES_ENDPOINT)
        .bearer_auth(api_key)
        .multipart(form)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("upload failed: {status} {body}").into());
    }
    let body: UploadResponse = response.json()?;
    println!("Finished upload. Fileid: {}", body.id);
    Ok(body.id)
}
