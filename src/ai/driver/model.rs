use crate::ai::{models, types::App};

pub fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    let trimmed = question.trim_end();
    if let Some(stripped) = trimmed.strip_suffix(" -code") {
        *question = stripped.trim_end().to_string();
        return models::qwen_coder_plus_latest().to_string();
    }
    if let Some(stripped) = trimmed.strip_suffix(" -d") {
        *question = stripped.trim_end().to_string();
        return models::deepseek_v3().to_string();
    }
    if let Some(selector) = trailing_model_selector(trimmed) {
        let suffix = format!(" -{}", (selector + b'0') as char);
        if let Some(stripped) = trimmed.strip_suffix(&suffix) {
            *question = stripped.trim_end().to_string();
        }
        return models::model_from_selector(selector, app.cli.thinking)
            .as_str()
            .to_string();
    }
    app.current_model.clone()
}

pub fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    vl_default_model: &str,
) -> Option<String> {
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
