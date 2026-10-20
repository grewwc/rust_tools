use crate::ai::mcp::{McpClient, SharedMcpClient};
use crate::ai::types::App;
use serde_json::json;
use std::path::Path;
use std::time::Duration;

pub fn resolve_model_for_input(
    app: &App,
    _has_usable_ocr_for_images: bool,
    _question: &mut String,
) -> String {
    // Resolution order:
    // 1) A trailing " -d" forces the default DeepSeek model (and strips the suffix).
    // 2) A trailing " -<digit>" selects one of the built-in models (and strips the suffix).
    // 3) Otherwise, keep the current model.
    // Image attachments no longer force a VL model switch — OCR text extraction
    // (a subagent parse run in driver/mod.rs before this function) provides text
    // for the current model.
    app.current_model.clone()
}

pub fn attachment_forced_model(
    _current_model: &str,
    _has_image_files: bool,
    _vl_default_model: &str,
    _has_usable_ocr_for_images: bool,
) -> Option<String> {
    // Text-only models with image attachments stay on the current model.
    // A subagent image parse (run in driver/mod.rs before this function) handles
    // extracting readable text for the LLM.
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

impl OcrExtraction {
    pub(in crate::ai) fn has_usable_text(&self) -> bool {
        self.images
            .iter()
            .any(|image| image.error.is_none() && image.extracted_chars > 0)
    }
}

pub(in crate::ai) struct OcrImageSummary {
    pub(in crate::ai) file_name: String,
    pub(in crate::ai) extracted_chars: usize,
    pub(in crate::ai) error: Option<String>,
}

/// Runs OCR on the attached images and returns Markdown content that can be spliced into the prompt.
/// 返回格式: "<!-- OCR_IMAGE: filename -->\nocr_text\n<!-- /OCR_IMAGE -->"
pub fn ocr_images_for_attached_input(
    mcp_client: &SharedMcpClient,
    image_files: &[String],
) -> Result<Option<OcrExtraction>, String> {
    if image_files.is_empty() {
        return Ok(None);
    }
    let (server_name, tool_name, full_tool_name) = {
        let mc = mcp_client.lock().unwrap();
        match resolve_ocr_route(&mc) {
            Some(route) => route,
            None => return Ok(None),
        }
    };

    let mut ocr_contents = Vec::new();
    let mut images = Vec::new();
    for file_path in image_files {
        let file_name = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(file_path);

        let result = {
            let mc = mcp_client.lock().unwrap();
            mc.call_tool(
                &server_name,
                &tool_name,
                json!({
                    "image_path": file_path
                }),
            )
        };

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachedImageParseRoute {
    VlSubagent,
    Ocr,
}

fn attached_image_parse_route(vl_model: &str) -> AttachedImageParseRoute {
    if crate::ai::models::supports_image_input(vl_model) {
        AttachedImageParseRoute::VlSubagent
    } else {
        AttachedImageParseRoute::Ocr
    }
}

/// When a non-VL model receives attached images, prefer dispatching a synchronous
/// subagent pinned to a VL model; fall back to static OCR when no VL model is available,
/// so image attachments are not degraded to plain-text filenames.
/// The returned OcrExtraction reuses the precomputed_ocr injection pipeline to feed the
/// parsed result to the main agent.
const IMAGE_PARSE_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Dispatch arguments for the attached-image parse subagent.
///
/// A pure transcription task does not need a reasoning chain, so the parse sends the internal
/// `"off"` control. The request layer turns that into `"none"` for `EffortNone` dialects (the
/// gpt-5.x gateways) and into the vendor switch for `RealSwitch` dialects (Qwen / DeepSeek),
/// instead of leaving them at their default tier. A dialect that cannot express off (MiniMax
/// always-on reasoning) gets no override and keeps its own default.
fn image_parse_dispatch_args(vl_model: &str, image_files: &[String]) -> serde_json::Value {
    let file_list = image_files
        .iter()
        .enumerate()
        .map(|(i, f)| format!("{}. {}", i + 1, f))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "主 agent 使用的是纯文本模型，无法直接看到图片。以下图片已作为附件直接附加到\
         本对话中，你第一轮就能看到它们，请逐张解析：\n\
         {file_list}\n\n\
         完整、忠实、详细地输出每张图片中的全部内容\
         （所有可见文字、图表、结构、界面元素、布局等），不要遗漏、不要总结。\
         图片已在对话中直接可见，无需调用 read_file 等工具。\n\n\
         输出要求：\n\
         1. 按文件顺序逐张输出，每张以 \"<!-- IMAGE: <文件名> -->\" 开头；\n\
         2. 解析结果将作为主 agent 获取图片信息的唯一来源，必须完整可读；\n\
         3. 只输出图片解析结果，不要讨论其他话题。"
    );

    let reasoning_effort = match crate::ai::request::model_thinking_off_capability(vl_model) {
        crate::ai::provider::ThinkingOffCapability::EffortNone
        | crate::ai::provider::ThinkingOffCapability::RealSwitch => Some("off"),
        crate::ai::provider::ThinkingOffCapability::Unsupported => None,
    };

    let mut args = json!({
        "description": "解析附带图片",
        "prompt": prompt,
        "agent": "build",
        "model": vl_model,
        // Hand the image files explicitly to the subagent: the VL model sees the images on the
        // first round, avoiding the redundant read_file-then-reattach-base64 round-trip (saving one
        // model request plus one duplicate image transfer).
        "image_files": image_files,
    });
    if let Some(effort) = reasoning_effort {
        args["reasoning_effort"] = json!(effort);
    }
    args
}

pub(in crate::ai) async fn parse_attached_images_via_subagent(
    mcp_client: &SharedMcpClient,
    vl_model: &str,
    image_files: &[String],
) -> Result<Option<OcrExtraction>, String> {
    if image_files.is_empty() {
        return Ok(None);
    }

    // Parse with the configured VL model (`ai.model.vl_default`, resolved into
    // `AppConfig::vl_default_model` at startup; unset means the auto-selected default). Images are
    // attached directly to the subagent's first user message via the image_files argument, so
    // the model can "see" the images on the first round, avoiding a redundant read_file
    // round-trip.
    let vl_model = vl_model.to_string();
    if attached_image_parse_route(&vl_model) == AttachedImageParseRoute::Ocr {
        return match ocr_images_for_attached_input(mcp_client, image_files) {
            Ok(Some(ocr)) => Ok(Some(ocr)),
            Ok(None) => Ok(Some(failed_image_parse_extraction(
                image_files,
                "未配置可用的 VL 模型，且未发现 OCR 工具",
            ))),
            Err(e) => Ok(Some(failed_image_parse_extraction(
                image_files,
                &format!("OCR fallback failed: {e}"),
            ))),
        };
    }

    let args = image_parse_dispatch_args(&vl_model, image_files);

    // Same as /audit: dispatch a synchronous subagent to parse the images, wait for it to
    // finish, then return to the main agent. On failure do not silently return None (otherwise
    // the main agent would have neither image content nor a failure hint and could only guess),
    // but build a placeholder OcrExtraction marked with error so the prepare stage injects a
    // visible notice.
    let result = match crate::ai::driver::tools::execute_direct_subagent_task(
        "subagent-image-parse",
        &args,
        IMAGE_PARSE_SUBAGENT_HARD_TIMEOUT,
        None,
    ) {
        Ok(r) => r,
        Err(e) => return Ok(Some(failed_image_parse_extraction(image_files, &e))),
    };

    let text = extract_subagent_output_text(&result.content);
    if text.trim().is_empty() {
        return Ok(Some(failed_image_parse_extraction(
            image_files,
            "subagent 未产出任何可用的图片解析文本",
        )));
    }

    let file_names: Vec<String> = image_files
        .iter()
        .map(|f| {
            Path::new(f)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(f)
                .to_string()
        })
        .collect();
    let images = file_names
        .iter()
        .map(|name| OcrImageSummary {
            file_name: name.clone(),
            extracted_chars: text.chars().count(),
            error: None,
        })
        .collect();
    Ok(Some(OcrExtraction {
        tool_name: "subagent:image_parse".to_string(),
        content: text,
        images,
    }))
}

/// Builds a placeholder result for image-parse failure: every image is marked with error and
/// content carries a visible failure notice, so the main agent at least sees that parsing
/// failed instead of silently losing the image content.
fn failed_image_parse_extraction(image_files: &[String], reason: &str) -> OcrExtraction {
    let file_names: Vec<String> = image_files
        .iter()
        .map(|f| {
            Path::new(f)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(f)
                .to_string()
        })
        .collect();
    let images = file_names
        .iter()
        .map(|name| OcrImageSummary {
            file_name: name.clone(),
            extracted_chars: 0,
            error: Some(reason.to_string()),
        })
        .collect();
    OcrExtraction {
        tool_name: "subagent:image_parse".to_string(),
        content: format!("[IMAGE PARSE FAILED: {reason}]"),
        images,
    }
}

/// Strips the [task_id=...] / [Task: ...] status headers and the trailing reminder from a
/// synchronous subagent's rendered output, keeping only the subagent's final parsed text.
fn extract_subagent_output_text(content: &str) -> String {
    let reminder = crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER;
    let mut text = content.to_string();
    if let Some(pos) = text.find(reminder) {
        text.truncate(pos);
    }
    // Explicitly strip deterministic wrapper lines by prefix ([task_id=...] status header,
    // agent/model selection notes, error lines, empty-output placeholder) instead of an
    // unconditional skip(1), so a non-subagent rendered first line is not deleted by mistake.
    text.lines()
        .skip_while(|line| line.starts_with("[task_id="))
        .skip_while(|line| {
            line.starts_with("[Task:")
                || line.starts_with("agent_reason=")
                || line.starts_with("model_reason=")
                || line.starts_with("Error:")
                || line.starts_with("(subagent did not produce")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        AttachedImageParseRoute, attached_image_parse_route, extract_subagent_output_text,
        failed_image_parse_extraction, image_parse_dispatch_args, preferred_ocr_image_tool_name,
    };
    use crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER;
    use serde_json::json;

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

    #[test]
    fn non_vl_default_routes_attached_images_to_ocr() {
        assert_eq!(
            attached_image_parse_route("definitely-not-a-vl-model"),
            AttachedImageParseRoute::Ocr
        );
    }

    #[test]
    fn extracts_clean_text_from_subagent_output() {
        let rendered = format!(
            "[task_id=42]\n\
             [Task: 解析附带图片 via build @ some-vl] COMPLETED after 3.2s\n\
             agent_reason=explicit agent override\n\
             model_reason=explicit model override\n\
             <!-- IMAGE: a.png -->\n\
             图片中的文字内容……\n\
             <!-- /IMAGE -->\n\
             {}",
            SUBAGENT_PARENT_SUMMARY_REMINDER
        );
        let text = extract_subagent_output_text(&rendered);
        assert_eq!(
            text,
            "<!-- IMAGE: a.png -->\n图片中的文字内容……\n<!-- /IMAGE -->"
        );
    }

    #[test]
    fn returns_empty_text_for_wrapper_only_output() {
        let rendered = format!(
            "[task_id=42]\n\
             [Task: 解析附带图片 via build @ some-vl] COMPLETED after 3.2s\n\
             agent_reason=explicit agent override\n\
             model_reason=explicit model override\n\
             {}",
            SUBAGENT_PARENT_SUMMARY_REMINDER
        );
        assert!(extract_subagent_output_text(&rendered).is_empty());
    }

    #[test]
    fn returns_empty_text_for_placeholder_only_output() {
        // Production path: when the subagent has no final text, format_subagent_output pushes a
        // placeholder line; like agent_reason/model_reason it is deterministic wrapper output and
        // must not be injected into the main agent as image content.
        let rendered = format!(
            "[task_id=42]\n\
             [Task: 解析附带图片 via build @ some-vl] COMPLETED after 3.2s\n\
             agent_reason=explicit agent override\n\
             model_reason=explicit model override\n\
             (subagent did not produce any final assistant text)\n\
             {}",
            SUBAGENT_PARENT_SUMMARY_REMINDER
        );
        assert!(extract_subagent_output_text(&rendered).is_empty());
    }

    /// Registry-driven fixture: a VL model whose dialect expresses "thinking off" as the
    /// effort floor (the plain OpenAI-compatible family the gpt-5.x gateways belong to).
    fn effort_none_vl_model_handle() -> String {
        crate::ai::model_names::all()
            .into_iter()
            .filter(|model| model.is_vl)
            .map(crate::ai::model_names::model_handle)
            .find(|handle| {
                matches!(
                    crate::ai::request::model_thinking_off_capability(handle),
                    crate::ai::provider::ThinkingOffCapability::EffortNone
                )
            })
            .expect("registry should hold a VL model with an effort-only off switch")
    }

    /// Registry-driven fixture: a VL model with always-on reasoning, where the gateway
    /// accepts `low..max` only and may reject the "none" floor.
    fn always_on_reasoning_vl_model_handle() -> String {
        crate::ai::model_names::all()
            .into_iter()
            .filter(|model| model.is_vl)
            .map(crate::ai::model_names::model_handle)
            .find(|handle| {
                matches!(
                    crate::ai::request::model_thinking_off_capability(handle),
                    crate::ai::provider::ThinkingOffCapability::Unsupported
                )
            })
            .expect("registry should hold a VL model without a thinking-off switch")
    }

    /// Registry-driven fixture: a VL model with a dedicated thinking off-switch.
    fn real_switch_vl_model_handle() -> String {
        crate::ai::model_names::all()
            .into_iter()
            .filter(|model| model.is_vl)
            .map(crate::ai::model_names::model_handle)
            .find(|handle| {
                matches!(
                    crate::ai::request::model_thinking_off_capability(handle),
                    crate::ai::provider::ThinkingOffCapability::RealSwitch
                )
            })
            .expect("registry should hold a VL model with a real thinking off-switch")
    }

    #[test]
    fn image_parse_dispatch_runs_at_the_reasoning_floor() {
        // The dispatch contract spells "thinking off" as "off"; the request layer maps it to
        // "none" for effort-only dialects and to the vendor switch for RealSwitch dialects.
        let model = effort_none_vl_model_handle();
        let args = image_parse_dispatch_args(&model, &["/tmp/shot.png".to_string()]);
        assert_eq!(args["reasoning_effort"], json!("off"));
        assert_eq!(args["model"], json!(model));
    }

    #[test]
    fn image_parse_dispatch_disables_thinking_for_real_switch_dialects() {
        let model = real_switch_vl_model_handle();
        let args = image_parse_dispatch_args(&model, &["/tmp/shot.png".to_string()]);
        assert_eq!(args["reasoning_effort"], json!("off"));
        assert_eq!(args["model"], json!(model));
    }

    #[test]
    fn image_parse_dispatch_omits_the_floor_when_the_dialect_cannot_express_it() {
        let model = always_on_reasoning_vl_model_handle();
        let args = image_parse_dispatch_args(&model, &["/tmp/shot.png".to_string()]);
        assert!(args.get("reasoning_effort").is_none());
    }

    #[test]
    fn failed_image_parse_extraction_marks_every_image_and_exposes_reason() {
        // P1-a regression: on subagent dispatch failure / empty text we must return a placeholder
        // marked with error instead of silently Ok(None), so the prepare stage can inject the
        // [IMAGE PARSE FAILED: ...] notice.
        let files = vec!["/tmp/a.png".to_string(), "/tmp/b.png".to_string()];
        let ocr = failed_image_parse_extraction(&files, "subagent 超时");

        assert_eq!(ocr.tool_name, "subagent:image_parse");
        assert!(ocr.content.contains("[IMAGE PARSE FAILED: subagent 超时]"));
        assert_eq!(ocr.images.len(), 2);
        for (img, expect_name) in ocr.images.iter().zip(["a.png", "b.png"]) {
            assert_eq!(img.file_name, expect_name);
            assert_eq!(img.extracted_chars, 0);
            assert_eq!(img.error.as_deref(), Some("subagent 超时"));
        }
        // A failure placeholder must not be consumed as usable OCR text (has_usable_text is false).
        assert!(!ocr.has_usable_text());
    }
}
