use crate::ai::{models, types::App};

pub fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    // Resolution order:
    // 1) If there are image attachments, force a VL-capable model (unless already VL).
    // 2) A trailing " -d" forces the default DeepSeek model (and strips the suffix).
    // 3) A trailing " -<digit>" selects one of the built-in models (and strips the suffix).
    // 4) Otherwise, keep the current model.
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    let trimmed = question.trim_end();
    if let Some(stripped) = trimmed.strip_suffix(" -d") {
        *question = stripped.trim_end().to_string();
        return models::deepseek_v3().to_string();
    }
    app.current_model.clone()
}

pub fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    vl_default_model: &str,
) -> Option<String> {
    // Many models are text-only. When there are images, we route to a VL model to avoid
    // provider-side errors.
    if has_image_files && !models::is_vl_model(current_model) {
        return Some(models::determine_vl_model(vl_default_model));
    }
    None
}
