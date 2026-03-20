use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use base64::Engine as _;
use colored::Colorize;
use reqwest::blocking::{Client, Response, multipart};
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
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamDelta {
    #[serde(default)]
    pub(super) content: String,
    #[serde(default)]
    pub(super) reasoning_content: String,
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
    let mut request_body = RequestBody {
        model: model.to_string(),
        messages: vec![Message {
            role: "system".to_string(),
            content: Value::String("You are a helpful assistant.".to_string()),
        }],
        stream: true,
        enable_thinking: app.cli.thinking,
        enable_search: models::search_enabled(model).then_some(true),
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
        });
    } else if !models::is_vl_model(model) {
        request_body
            .messages
            .extend(build_message_arr(history_count, &app.config.history_file)?);
    }

    request_body.messages.push(Message {
        role: "user".to_string(),
        content: build_content(&request_body.model, question, &app.attached_image_files)?,
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

fn build_content(
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
    print!("[{} (search: {})] ", model.green(), search.red());
    let _ = io::stdout().flush();
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
