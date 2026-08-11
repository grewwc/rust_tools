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

/// 对附加图片执行 OCR，并返回可拼接进 prompt 的 Markdown 内容。
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

/// 非 VL 模型收到附带图片时，优先派发固定使用 VL 模型的同步 subagent；系统没有
/// 可用 VL 模型时改走静态 OCR，避免把图片附件降级成纯文本文件名。
/// 返回的 OcrExtraction 复用 precomputed_ocr 注入管线，把解析结果喂给主 agent。
const IMAGE_PARSE_SUBAGENT_HARD_TIMEOUT: Duration = Duration::from_secs(15 * 60);

pub(in crate::ai) async fn parse_attached_images_via_subagent(
    mcp_client: &SharedMcpClient,
    image_files: &[String],
) -> Result<Option<OcrExtraction>, String> {
    if image_files.is_empty() {
        return Ok(None);
    }

    // 固定用系统配置的默认 VL 模型，图片通过 image_files 参数直接附加到子代理首条
    // user 消息，第一轮就能直接"看到"图，省掉 read_file 的冗余往返。
    let vl_model = crate::ai::models::default_vl_model();
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

    let args = json!({
        "description": "解析附带图片",
        "prompt": prompt,
        "agent": "build",
        "model": vl_model,
        // 把图片文件显式交给子代理：VL 模型第一轮直接看到图，避免先 read_file、
        // 再在下一轮重复附加 base64 的冗余往返（省一轮模型请求 + 一次重复传图）。
        "image_files": image_files,
        // 纯转录任务不需要 high 思考链，压到 minimal 档加速解析。
        "reasoning_effort": "minimal",
    });

    // 与 /audit 相同：同步派发一个 subagent 解析图片，等它完成后再回到主 agent。
    // 失败时不静默返回 None（否则主 agent 既无图片内容也无失败提示，只能凭空猜），
    // 而是构造带 error 标记的占位 OcrExtraction，让 prepare 阶段注入可见提示。
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

/// 构造图片解析失败的占位结果：每张图标记 error，content 为可见的失败提示，
/// 使主 agent 至少能看到图片解析失败而不是静默丢失图片内容。
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

/// 从同步 subagent 的渲染结果中剥离 [task_id=...] / [Task: ...] 状态头与
/// 尾部提醒，只保留 subagent 的最终解析文本。
fn extract_subagent_output_text(content: &str) -> String {
    let reminder = crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER;
    let mut text = content.to_string();
    if let Some(pos) = text.find(reminder) {
        text.truncate(pos);
    }
    // 显式按前缀剥离确定性包装行（[task_id=...] 状态头、agent/model 选择说明、
    // 错误行、空输出占位），而不是无条件 skip(1)，避免非 subagent 渲染内容被误删首行。
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
        failed_image_parse_extraction, preferred_ocr_image_tool_name,
    };
    use crate::ai::tools::task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER;

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
        // 生产路径：subagent 无最终文本时 format_subagent_output 会 push 占位符行，
        // 该行与 agent_reason/model_reason 一样属于确定性包装，不应作为图片内容注入主 agent。
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

    #[test]
    fn failed_image_parse_extraction_marks_every_image_and_exposes_reason() {
        // P1-a 回归：subagent 派发失败 / 空文本时必须返回带 error 的占位结果，
        // 而不是静默 Ok(None)，prepare 阶段才能注入 [IMAGE PARSE FAILED: ...] 提示。
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
        // 失败占位不可作为可用 OCR 文本消费（has_usable_text 为 false）。
        assert!(!ocr.has_usable_text());
    }
}
