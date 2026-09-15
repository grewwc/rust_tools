//! Tool schema declarations + `tools/call` dispatch for the PDF server.
//!
//! Everything is returned through `content[0].text` (the host only reads that).
//! Each parse runs under `with_timeout`; timeouts return a clean error without
//! transport trigger words.

use serde_json::{Value, json};

use mcp_stdio::{JsonRpcErr, cap_text, text_content, with_timeout};

/// Per-operation timeout (ms). Default 90s, below the host request_timeout_ms (120s).
fn op_timeout_ms() -> u64 {
    std::env::var("MCP_PDF_OP_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(90_000)
}

/// initialize result.
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
        "serverInfo": { "name": "mcp-pdf", "version": "0.1.0" }
    })
}

/// tools/list result: one tool, `parse_pdf`.
pub fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "parse_pdf",
                "description": "Parse a PDF file and return its metadata (page count, title, author, subject, keywords) plus extracted text. Use this instead of read_file for .pdf files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path to the PDF file" },
                        "extract_text": { "type": "boolean", "description": "Whether to extract page text (default true; set false for metadata only)" },
                        "pages": { "type": "array", "items": { "type": "integer", "minimum": 1 }, "description": "Optional 1-based page numbers to extract; default all pages" }
                    },
                    "required": ["path"]
                }
            }
        ]
    })
}

/// tools/call dispatch: only `parse_pdf` is implemented.
pub async fn handle_tools_call(params: Option<Value>) -> Result<Value, JsonRpcErr> {
    let params = params.ok_or_else(|| JsonRpcErr::new(-32602, "missing params", None))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    match name.as_str() {
        "parse_pdf" => parse_pdf_tool(&args).await,
        _ => Err(JsonRpcErr::new(-32601, &format!("Unknown tool: {name}"), None)),
    }
}

async fn parse_pdf_tool(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| JsonRpcErr::new(-32602, "missing required string argument: path", None))?;
    let extract_text = args.get("extract_text").and_then(|v| v.as_bool()).unwrap_or(true);
    let pages = args.get("pages").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
            .collect::<Vec<u32>>()
    });

    let opts = rust_tools::pdfw::PdfParseOptions {
        extract_text,
        pages,
    };

    with_timeout(op_timeout_ms(), async move {
        let parsed = rust_tools::pdfw::parse_pdf(path, opts)
            .map_err(|e| format!("{}: {e}", pdf_err_kind(&e)))?;

        let mut out = String::new();
        out.push_str(&format!("path: {}\n", parsed.path.display()));
        out.push_str(&format!("page_count: {}\n", parsed.page_count));
        if let Some(t) = parsed.title {
            out.push_str(&format!("title: {t}\n"));
        }
        if let Some(a) = parsed.author {
            out.push_str(&format!("author: {a}\n"));
        }
        if let Some(s) = parsed.subject {
            out.push_str(&format!("subject: {s}\n"));
        }
        if let Some(k) = parsed.keywords {
            out.push_str(&format!("keywords: {k}\n"));
        }
        if let Some(text) = parsed.text {
            out.push('\n');
            out.push_str(&text);
        }
        Ok(text_content(cap_text(&out)))
    })
    .await
}

/// Short error-kind tag mirroring the three `PdfParseError` variants, so hosts
/// can distinguish "file unreadable" from "file parsed but text extraction failed".
fn pdf_err_kind(e: &rust_tools::pdfw::PdfParseError) -> &'static str {
    match e {
        rust_tools::pdfw::PdfParseError::OpenFailed(_) => "open_failed",
        rust_tools::pdfw::PdfParseError::ParseFailed(_) => "parse_failed",
        rust_tools::pdfw::PdfParseError::ExtractTextFailed(_) => "extract_text_failed",
    }
}
