use crate::ai::{mcp::McpClient, models, types::App};
use serde_json::json;
use std::path::Path;

pub fn resolve_model_for_input(app: &App, _question: &mut String) -> String {
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

/// 检查模型是否支持图片输入
pub fn model_supports_images(model: &str) -> bool {
    models::supports_image_input(model)
}

/// 当模型不支持图片输入时，对图片进行 OCR 处理并返回 Markdown 内容
/// 返回格式: "<!-- OCR_IMAGE: filename -->\nocr_text\n<!-- /OCR_IMAGE -->"
pub fn ocr_images_for_text_model(
    mcp_client: &mut McpClient,
    image_files: &[String],
) -> Result<String, String> {
    if image_files.is_empty() {
        return Ok(String::new());
    }

    let mut ocr_contents = Vec::new();
    for file_path in image_files {
        let file_name = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(file_path);

        // 调用 OCR MCP 工具
        let result = mcp_client.call_tool(
            "ocr",
            "ocr_image",
            json!({
                "image_path": file_path
            }),
        );

        let ocr_text = match result {
            Ok(text) => text,
            Err(e) => {
                eprintln!("[OCR] Failed to OCR {}: {}", file_name, e);
                format!("[OCR FAILED for {}: {}]", file_name, e)
            }
        };

        // 格式化输出，带标记
        let content = format!(
            "<!-- OCR_IMAGE: {} -->\n{}\n<!-- /OCR_IMAGE -->",
            file_name, ocr_text
        );
        ocr_contents.push(content);
    }

    Ok(ocr_contents.join("\n\n"))
}
