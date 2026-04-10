use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use rust_tools::pdfw::ocr_image_to_text;
use serde_json::{json, Value};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(_) => return,
        };
        if n == 0 {
            return;
        }
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(err) => {
                let _ = write_json_rpc_error(&mut stdout, None, -32700, &err.to_string(), None);
                continue;
            }
        };

        let id = req.get("id").cloned();
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned();

        let res = match method.as_str() {
            "initialize" => handle_initialize(),
            "notifications/initialized" => Ok(json!({})),
            "tools/list" => handle_tools_list(),
            "tools/call" => handle_tools_call(params),
            "resources/list" => Ok(json!({"resources": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            _ => Err(json_rpc_error(
                -32601,
                "Method not found",
                Some(json!({ "method": method })),
            )),
        };

        match res {
            Ok(result) => {
                let _ = write_json_rpc_result(&mut stdout, id.as_ref(), result);
            }
            Err(err) => {
                let _ = write_json_rpc_error(
                    &mut stdout,
                    id.as_ref(),
                    err.code,
                    &err.message,
                    err.data,
                );
            }
        }
    }
}

fn handle_initialize() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
            "resources": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": "mcp-ocr",
            "version": "0.1.0"
        }
    }))
}

fn handle_tools_list() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "tools": [
            {
                "name": "ocr_image",
                "description": "Perform OCR on an image file and extract text. Supports PNG, JPEG, BMP, TIFF, WebP, GIF formats. Uses Vision on macOS, tesseract on other platforms.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file to OCR"
                        },
                        "langs": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "OCR languages: macOS Vision uses zh-Hans/en-US/ja-JP/ko-KR, tesseract uses chi_sim/eng/jpn/kor etc.",
                            "default": ["zh-Hans", "en-US"]
                        }
                    },
                    "required": ["image_path"]
                }
            },
            {
                "name": "ocr_pdf",
                "description": "Perform OCR on a PDF file and extract text as Markdown. Extracts embedded images from scanned PDF pages and runs OCR on them. Falls back to native text extraction if PDF has text layer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to the PDF file to OCR"
                        },
                        "langs": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "OCR languages, e.g. ['zh-Hans', 'en-US']",
                            "default": ["zh-Hans", "en-US"]
                        },
                        "pages": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "Specific pages to OCR (1-based). If omitted, processes all pages."
                        }
                    },
                    "required": ["pdf_path"]
                }
            },
            {
                "name": "ocr_base64_image",
                "description": "Perform OCR on a base64-encoded image and extract text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_base64": {
                            "type": "string",
                            "description": "Base64-encoded image data (may include data:image/xxx;base64, prefix)"
                        },
                        "langs": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "OCR languages",
                            "default": ["zh-Hans", "en-US"]
                        }
                    },
                    "required": ["image_base64"]
                }
            }
        ]
    }))
}

fn handle_tools_call(params: Option<Value>) -> Result<Value, JsonRpcErr> {
    let params = params.unwrap_or_else(|| json!({}));
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match name.as_str() {
        "ocr_image" => {
            let image_path = args
                .get("image_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .ok_or_else(|| json_rpc_error(-32602, "missing image_path", None))?;
            
            let langs = args
                .get("langs")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec!["zh-Hans", "en-US"]);

            let text = ocr_image_file(&image_path, &langs).map_err(|e| json_rpc_error(-32000, &e, None))?;
            
            Ok(json!({ "content": [{ "type": "text", "text": text }] }))
        }
        "ocr_pdf" => {
            let pdf_path = args
                .get("pdf_path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .ok_or_else(|| json_rpc_error(-32602, "missing pdf_path", None))?;
            
            let langs = args
                .get("langs")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec!["zh-Hans", "en-US"]);
            
            let pages = args
                .get("pages")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as u32)
                        .collect::<Vec<u32>>()
                });

            let text = if let Some(ps) = pages {
                rust_tools::pdfw::ocr_pdf_to_markdown_pages(&pdf_path, &langs, Some(ps.as_slice()))
                    .map_err(|e| json_rpc_error(-32000, &e.to_string(), None))?
            } else {
                rust_tools::pdfw::ocr_pdf_to_markdown(&pdf_path, &langs)
                    .map_err(|e| json_rpc_error(-32000, &e.to_string(), None))?
            };

            Ok(json!({ "content": [{ "type": "text", "text": text }] }))
        }
        "ocr_base64_image" => {
            let image_base64 = args
                .get("image_base64")
                .and_then(|v| v.as_str())
                .ok_or_else(|| json_rpc_error(-32602, "missing image_base64", None))?;
            
            let langs = args
                .get("langs")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec!["zh-Hans", "en-US"]);

            let text = ocr_base64_image(image_base64, &langs)
                .map_err(|e| json_rpc_error(-32000, &e, None))?;

            Ok(json!({ "content": [{ "type": "text", "text": text }] }))
        }
        _ => Err(json_rpc_error(-32601, "Unknown tool", Some(json!({ "tool": name })))),
    }
}

fn ocr_image_file(path: &PathBuf, langs: &[&str]) -> Result<String, String> {
    let img = image::open(path).map_err(|e| e.to_string())?;
    ocr_image_to_text(&img, langs)
}


fn ocr_base64_image(base64_str: &str, langs: &[&str]) -> Result<String, String> {
    use base64::Engine;

    let data = if let Some(pos) = base64_str.find("base64,") {
        let stripped = &base64_str[pos + 7..];
        base64::engine::general_purpose::STANDARD.decode(stripped).map_err(|e| e.to_string())?
    } else {
        base64::engine::general_purpose::STANDARD.decode(base64_str).map_err(|e| e.to_string())?
    };
    let img = image::load_from_memory(&data).map_err(|e| e.to_string())?;
    ocr_image_to_text(&img, langs)
}

struct JsonRpcErr {
    code: i32,
    message: String,
    data: Option<Value>,
}

fn json_rpc_error(code: i32, message: &str, data: Option<Value>) -> JsonRpcErr {
    JsonRpcErr {
        code,
        message: message.to_string(),
        data,
    }
}

fn write_json_rpc_result(stdout: &mut io::Stdout, id: Option<&Value>, result: Value) {
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    });
    writeln!(stdout, "{}", resp).ok();
}

fn write_json_rpc_error(stdout: &mut io::Stdout, id: Option<&Value>, code: i32, message: &str, data: Option<Value>) {
    let mut err_obj = json!({
        "code": code,
        "message": message
    });
    if let Some(d) = data {
        err_obj["data"] = d;
    }
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": err_obj
    });
    writeln!(stdout, "{}", resp).ok();
}