use crate::ai::{types::App, models};

pub fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        !app.attached_binary_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    // Work on the original string without cloning first
    let original_len = question.len();
    let trimmed_len = question.trim_end().len();
    
    if trimmed_len >= 6 && &question[trimmed_len -6..trimmed_len] == " -code" {
        *question = question[..trimmed_len -6].trim_end().to_string();
        return models::qwen_coder_plus_latest().to_string();
    }
    if trimmed_len >= 3 && &question[trimmed_len -3..trimmed_len] == " -d" {
        *question = question[..trimmed_len -3].trim_end().to_string();
        return models::deepseek_v3().to_string();
    }
    if let Some(selector) = trailing_model_selector(question) {
        *question = question[..original_len - 3].trim_end().to_string();
        return models::model_from_selector(selector, app.cli.thinking)
            .as_str()
            .to_string();
    }
    app.current_model.clone()
}

pub fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    has_binary_files: bool,
    vl_default_model: &str,
) -> Option<String> {
    if current_model == models::qwen_long() {
        return Some(models::qwen_long().to_string());
    }
    if has_binary_files {
        return Some(models::qwen_long().to_string());
    }
    if has_image_files && !models::is_vl_model(current_model) {
        return Some(models::determine_vl_model(vl_default_model));
    }
    None
}

pub fn trailing_model_selector(input: &str) -> Option<u8> {
    let bytes = input.as_bytes();
    if bytes.len() < 3 {
        return None;
    }
    let dash_idx = bytes.len() - 2;
    if bytes[dash_idx] != b'-' || !bytes[dash_idx + 1].is_ascii_digit() {
        return None;
    }
    if dash_idx == 0 || bytes[dash_idx - 1] != b' ' {
        return None;
    }
    Some(bytes[dash_idx + 1] - b'0')
}
