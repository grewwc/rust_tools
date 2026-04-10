use crate::ai::{mcp::McpClient, models, types::App};
use serde_json::json;
use std::path::Path;

pub fn resolve_model_for_input(
    app: &App,
    ocr_succeeded_for_images: bool,
    _question: &mut String,
) -> String {
    // Resolution order:
    // 1) If there are image attachments, force a VL-capable model (unless already VL).
    // 2) A trailing " -d" forces the default DeepSeek model (and strips the suffix).
    // 3) A trailing " -<digit>" selects one of the built-in models (and strips the suffix).
    // 4) Otherwise, keep the current model.
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        &app.config.vl_default_model,
        ocr_succeeded_for_images,
    ) {
        return model;
    }
    app.current_model.clone()
}

pub fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    vl_default_model: &str,
    ocr_succeeded_for_images: bool,
) -> Option<String> {
    // Many models are text-only. When there are images, we route to a VL model to avoid
    // provider-side errors.
    if has_image_files && !models::is_vl_model(current_model) && !ocr_succeeded_for_images {
        return Some(models::determine_vl_model(vl_default_model));
    }
    None
}

fn preferred_ocr_image_tool_name<'a>(
    tool_names: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    fn score(name: &str) -> usize {
        match name {
            "mcp_ocr_extract_ocr_image" => 0,
            "mcp_ocr_ocr_image" => 1,
            n if n.starts_with("mcp_ocr_") && n.ends_with("_ocr_image") => 2,
            n if n.contains("_ocr_") && n.ends_with("_ocr_image") => 3,
            n if n.starts_with("mcp_") && n.ends_with("_ocr_image") => 4,
            _ => usize::MAX,
        }
    }

    tool_names
        .into_iter()
        .filter(|name| score(name) != usize::MAX)
        .min_by_key(|name| score(name))
}

fn resolve_ocr_route(mcp_client: &McpClient) -> Option<(String, String, String)> {
    let tools = mcp_client.get_all_tools();
    if let Some(full_tool_name) =
        preferred_ocr_image_tool_name(tools.iter().map(|tool| tool.function.name.as_str()))
        && let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(full_tool_name)
    {
        return Some((server_name, tool_name, full_tool_name.to_string()));
    }

    for server_name in ["ocr_extract", "ocr"] {
        let tool_name = "ocr_image";
        let full_tool_name = format!("mcp_{server_name}_{tool_name}");
        if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(&full_tool_name)
        {
            return Some((server_name, tool_name, full_tool_name));
        }
    }
    None
}

pub(in crate::ai) struct OcrExtraction {
    pub(in crate::ai) tool_name: String,
    pub(in crate::ai) content: String,
    pub(in crate::ai) images: Vec<OcrImageSummary>,
}

pub(in crate::ai) struct OcrImageSummary {
    pub(in crate::ai) file_name: String,
    pub(in crate::ai) extracted_chars: usize,
    pub(in crate::ai) error: Option<String>,
}

/// 对附加图片执行 OCR，并返回可拼接进 prompt 的 Markdown 内容。
/// 返回格式: "<!-- OCR_IMAGE: filename -->\nocr_text\n<!-- /OCR_IMAGE -->"
pub fn ocr_images_for_attached_input(
    mcp_client: &mut McpClient,
    image_files: &[String],
) -> Result<Option<OcrExtraction>, String> {
    if image_files.is_empty() {
        return Ok(None);
    }
    let Some((server_name, tool_name, full_tool_name)) = resolve_ocr_route(mcp_client) else {
        return Ok(None);
    };

    let mut ocr_contents = Vec::new();
    let mut images = Vec::new();
    for file_path in image_files {
        let file_name = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(file_path);

        let result = mcp_client.call_tool(
            &server_name,
            &tool_name,
            json!({
                "image_path": file_path
            }),
        );

        let (ocr_text, extracted_chars, error) = match result {
            Ok(text) => {
                let extracted_chars = text.chars().count();
                (text, extracted_chars, None)
            }
            Err(e) => {
                let fallback = format!("[OCR FAILED for {}: {}]", file_name, e);
                let extracted_chars = fallback.chars().count();
                (fallback, extracted_chars, Some(e))
            }
        };

        // 格式化输出，带标记
        let content = format!(
            "<!-- OCR_IMAGE: {} -->\n{}\n<!-- /OCR_IMAGE -->",
            file_name, ocr_text
        );
        ocr_contents.push(content);
        images.push(OcrImageSummary {
            file_name: file_name.to_string(),
            extracted_chars,
            error,
        });
    }

    Ok(Some(OcrExtraction {
        tool_name: full_tool_name,
        content: ocr_contents.join("\n\n"),
        images,
    }))
}

#[cfg(test)]
mod tests {
    use super::preferred_ocr_image_tool_name;

    #[test]
    fn prefers_configured_ocr_extract_image_tool_when_available() {
        let selected = preferred_ocr_image_tool_name([
            "mcp_pdf_extract_extract_pdf_text",
            "mcp_ocr_extract_ocr_image",
            "mcp_ocr_extract_ocr_pdf",
        ]);
        assert_eq!(selected, Some("mcp_ocr_extract_ocr_image"));
    }

    #[test]
    fn falls_back_to_any_mcp_ocr_image_tool() {
        let selected = preferred_ocr_image_tool_name([
            "mcp_misc_ping",
            "mcp_some_server_ocr_image",
            "mcp_other_server_ocr_pdf",
        ]);
        assert_eq!(selected, Some("mcp_some_server_ocr_image"));
    }
}
