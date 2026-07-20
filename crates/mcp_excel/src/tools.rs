//! 工具 schema 声明 + `tools/call` 分发与各工具逻辑。
//!
//! 全部工具都通过 `content[0].text` 回传文本（宿主只读这一处）。
//! 每个 osascript 操作都包在 `with_timeout` 里，超时返回不含 transport 触发词的干净错误。
//!
//! # 工具集（结构化动词，不向模型开放原生 AppleScript）
//! - open_workbook / list_sheets / read_cell / read_range / write_cell /
//!   write_range / close_workbook —— 读写真实 Excel 软件，已实测可靠。
//! - export_csv —— **数据落盘主力**：AppleScript 读出数据、Rust 侧写文件，
//!   绕过 Excel 沙盒版 `save` 的 -50 限制。
//! - save_workbook —— 实验性，尝试 Excel 原生存盘，失败给出诚实的沙盒说明。

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::jsonrpc::{JsonRpcErr, cap_text, text_content, with_timeout};
use crate::osa;

/// 每操作超时（毫秒）。默认 90s，短于宿主 request_timeout_ms（建议 120s）。
fn op_timeout_ms() -> u64 {
    std::env::var("MCP_EXCEL_OP_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(90_000)
}

/// initialize 结果。
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
        "serverInfo": { "name": "mcp-excel", "version": "0.1.0" }
    })
}

/// tools/list 结果：驱动真实 Excel 软件的结构化动词工具。
pub fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "open_workbook",
                "description": "Open an .xlsx/.xls/.csv workbook in the real Excel app (idempotent: reuses it if already open) and return the list of sheet names. Excel keeps it open across calls.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path to the workbook file" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_sheets",
                "description": "List all worksheet names of a workbook (opens it if needed).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path to the workbook file" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "read_cell",
                "description": "Read a single cell value from an open workbook.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path (must be open, or will be opened)" },
                        "cell": { "type": "string", "description": "Cell address, e.g. A1" },
                        "sheet": { "type": "string", "description": "Optional worksheet name; defaults to the first sheet" }
                    },
                    "required": ["path", "cell"]
                }
            },
            {
                "name": "read_range",
                "description": "Read a rectangular range (or the whole used range if 'range' omitted) as TSV text (tab-separated, newline per row).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path (must be open, or will be opened)" },
                        "range": { "type": "string", "description": "Optional A1-style range, e.g. A1:C10; omit for the entire used range" },
                        "sheet": { "type": "string", "description": "Optional worksheet name; defaults to the first sheet" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_cell",
                "description": "Write a value into a single cell of an open workbook (in memory; use export_csv to persist to disk).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path (must be open, or will be opened)" },
                        "cell": { "type": "string", "description": "Cell address, e.g. A1" },
                        "value": { "type": "string", "description": "Value to write" },
                        "as_number": { "type": "boolean", "description": "Treat value as a number instead of text (default false)", "default": false },
                        "sheet": { "type": "string", "description": "Optional worksheet name; defaults to the first sheet" }
                    },
                    "required": ["path", "cell", "value"]
                }
            },
            {
                "name": "write_range",
                "description": "Write a 2D array of values starting at a top-left cell. Rows are applied cell by cell (in memory; use export_csv to persist).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path (must be open, or will be opened)" },
                        "start_cell": { "type": "string", "description": "Top-left cell address, e.g. A1" },
                        "rows": {
                            "type": "array",
                            "description": "2D array of strings; each inner array is a row",
                            "items": { "type": "array", "items": { "type": "string" } }
                        },
                        "sheet": { "type": "string", "description": "Optional worksheet name; defaults to the first sheet" }
                    },
                    "required": ["path", "start_cell", "rows"]
                }
            },
            {
                "name": "export_csv",
                "description": "Persist a worksheet (its used range) to a .csv file on disk. Reads the data via Excel then writes the file directly, bypassing Excel's sandboxed save. This is the reliable way to save results.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path of the source workbook (must be open, or will be opened)" },
                        "dest": { "type": "string", "description": "Absolute POSIX path of the .csv file to write" },
                        "sheet": { "type": "string", "description": "Optional worksheet name; defaults to the first sheet" }
                    },
                    "required": ["path", "dest"]
                }
            },
            {
                "name": "save_workbook",
                "description": "EXPERIMENTAL: attempt Excel's native 'save as'. On sandboxed Excel this often fails with error -50; prefer export_csv to persist data reliably.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path of the open workbook" },
                        "dest": { "type": "string", "description": "Absolute POSIX destination path" }
                    },
                    "required": ["path", "dest"]
                }
            },
            {
                "name": "close_workbook",
                "description": "Close a workbook without saving (discards in-memory changes not yet exported).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute POSIX path of the workbook to close" }
                    },
                    "required": ["path"]
                }
            }
        ]
    })
}

/// tools/call 分发。
pub async fn handle_tools_call(params: Option<Value>) -> Result<Value, JsonRpcErr> {
    let params = params.unwrap_or_else(|| json!({}));
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

    match name.as_str() {
        "open_workbook" => tool_open_workbook(&args).await,
        "list_sheets" => tool_list_sheets(&args).await,
        "read_cell" => tool_read_cell(&args).await,
        "read_range" => tool_read_range(&args).await,
        "write_cell" => tool_write_cell(&args).await,
        "write_range" => tool_write_range(&args).await,
        "export_csv" => tool_export_csv(&args).await,
        "save_workbook" => tool_save_workbook(&args).await,
        "close_workbook" => tool_close_workbook(&args).await,
        _ => Err(JsonRpcErr::new(
            -32601,
            "Unknown tool",
            Some(json!({ "tool": name })),
        )),
    }
}

// ---- 参数助手 ----

fn require_str(args: &Value, key: &str) -> Result<String, JsonRpcErr> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| JsonRpcErr::new(-32602, &format!("missing or empty '{key}'"), None))
}

fn opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// 确保 Excel 可用；不可用时返回诚实的 JSON-RPC 错误。
async fn ensure_excel() -> Result<(), JsonRpcErr> {
    osa::excel_version()
        .await
        .map(|_| ())
        .map_err(|e| JsonRpcErr::new(-32000, &format!("Microsoft Excel unavailable: {e}"), None))
}

// ---- 各工具实现 ----

async fn tool_open_workbook(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    ensure_excel().await?;
    let sheets = with_timeout(op_timeout_ms(), osa::open_workbook(&path)).await?;
    Ok(text_content(format!(
        "Opened {path}\nSheets:\n{}",
        sheets.trim()
    )))
}

async fn tool_list_sheets(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    ensure_excel().await?;
    let sheets = with_timeout(op_timeout_ms(), osa::list_sheets(&path)).await?;
    Ok(text_content(sheets.trim().to_string()))
}

async fn tool_read_cell(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let cell = require_str(args, "cell")?;
    let sheet = opt_str(args, "sheet");
    ensure_excel().await?;
    // 幂等确保已打开。
    let _ = osa::open_workbook(&path).await;
    let value = with_timeout(
        op_timeout_ms(),
        osa::read_cell(&path, sheet.as_deref(), &cell),
    )
    .await?;
    Ok(text_content(format!("{cell}={value}")))
}

async fn tool_read_range(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let range = opt_str(args, "range");
    let sheet = opt_str(args, "sheet");
    ensure_excel().await?;
    let _ = osa::open_workbook(&path).await;
    let tsv = with_timeout(
        op_timeout_ms(),
        osa::read_range(&path, sheet.as_deref(), range.as_deref()),
    )
    .await?;
    Ok(text_content(cap_text(&tsv)))
}

async fn tool_write_cell(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let cell = require_str(args, "cell")?;
    let value = require_str(args, "value")?;
    let as_number = args.get("as_number").and_then(|v| v.as_bool()).unwrap_or(false);
    let sheet = opt_str(args, "sheet");
    ensure_excel().await?;
    let _ = osa::open_workbook(&path).await;
    with_timeout(
        op_timeout_ms(),
        osa::write_cell(&path, sheet.as_deref(), &cell, &value, as_number),
    )
    .await?;
    Ok(text_content(format!("Wrote {cell}={value}")))
}

async fn tool_write_range(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let start = require_str(args, "start_cell")?;
    let sheet = opt_str(args, "sheet");
    let rows = args
        .get("rows")
        .and_then(|v| v.as_array())
        .ok_or_else(|| JsonRpcErr::new(-32602, "missing 'rows' (2D array)", None))?;

    // 解析起始单元格的列字母与行号，逐格写入。
    let (start_col, start_row) = parse_a1(&start)
        .ok_or_else(|| JsonRpcErr::new(-32602, &format!("invalid start_cell '{start}'"), None))?;

    ensure_excel().await?;
    let _ = osa::open_workbook(&path).await;

    let mut written = 0usize;
    let ms = op_timeout_ms();
    for (r, row) in rows.iter().enumerate() {
        let Some(cells) = row.as_array() else { continue };
        for (c, cell) in cells.iter().enumerate() {
            let addr = format!("{}{}", col_to_letters(start_col + c), start_row + r);
            let (val, is_num) = cell_value_literal(cell);
            with_timeout(
                ms,
                osa::write_cell(&path, sheet.as_deref(), &addr, &val, is_num),
            )
            .await?;
            written += 1;
        }
    }
    Ok(text_content(format!("Wrote {written} cells starting at {start}")))
}

async fn tool_export_csv(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let dest = require_str(args, "dest")?;
    let sheet = opt_str(args, "sheet");
    ensure_excel().await?;
    let _ = osa::open_workbook(&path).await;

    // 走已验证可靠的 read_range(used range) → TSV，再 Rust 侧转 CSV 落盘（绕过 save -50）。
    let tsv = with_timeout(op_timeout_ms(), osa::read_range(&path, sheet.as_deref(), None)).await?;
    let csv = tsv_to_csv(&tsv);

    if let Some(parent) = PathBuf::from(&dest).parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| JsonRpcErr::new(-32000, &format!("cannot create dest dir: {e}"), None))?;
    }
    std::fs::write(&dest, csv.as_bytes())
        .map_err(|e| JsonRpcErr::new(-32000, &format!("cannot write csv: {e}"), None))?;
    let bytes = csv.len();
    Ok(text_content(format!("Exported used range to {dest} ({bytes} bytes)")))
}

async fn tool_save_workbook(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    let dest = require_str(args, "dest")?;
    ensure_excel().await?;
    match with_timeout(op_timeout_ms(), osa::save_as(&path, &dest)).await {
        Ok(_) => Ok(text_content(format!("Saved workbook as {dest}"))),
        Err(e) => {
            // 沙盒 -50 是已知系统限制：给出诚实、可操作的说明，引导用 export_csv。
            Ok(text_content(format!(
                "Native save failed ({}). Microsoft Excel is sandboxed and its AppleScript 'save as' commonly returns error -50 in a non-interactive context. Use export_csv to persist data reliably instead.",
                e.message
            )))
        }
    }
}

async fn tool_close_workbook(args: &Value) -> Result<Value, JsonRpcErr> {
    let path = require_str(args, "path")?;
    ensure_excel().await?;
    let msg = with_timeout(op_timeout_ms(), osa::close_workbook(&path)).await?;
    Ok(text_content(format!("close: {msg}")))
}

// ---- A1 地址与值转换助手 ----

/// 解析 A1 地址为 (0-based 列号, 1-based 行号)。仅接受形如 `AB12`。
fn parse_a1(addr: &str) -> Option<(usize, usize)> {
    let addr = addr.trim();
    let split = addr.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = addr.split_at(split);
    if letters.is_empty() || digits.is_empty() {
        return None;
    }
    let mut col = 0usize;
    for ch in letters.chars() {
        if !ch.is_ascii_alphabetic() {
            return None;
        }
        col = col * 26 + (ch.to_ascii_uppercase() as usize - 'A' as usize + 1);
    }
    let row: usize = digits.parse().ok()?;
    Some((col - 1, row))
}

/// 0-based 列号 → 列字母（0→A, 25→Z, 26→AA）。
fn col_to_letters(mut col: usize) -> String {
    let mut s = String::new();
    loop {
        let rem = col % 26;
        s.insert(0, (b'A' + rem as u8) as char);
        if col < 26 {
            break;
        }
        col = col / 26 - 1;
    }
    s
}

/// 把 JSON cell 值转成 (字面量文本, 是否数值)。字符串按文本，数字按数值。
fn cell_value_literal(v: &Value) -> (String, bool) {
    match v {
        Value::Number(n) => (n.to_string(), true),
        Value::Bool(b) => (b.to_string(), false),
        Value::String(s) => (s.clone(), false),
        Value::Null => (String::new(), false),
        other => (other.to_string(), false),
    }
}

/// TSV → CSV：按 tab 切列、按行处理，需要时给字段加引号并转义内部引号。
fn tsv_to_csv(tsv: &str) -> String {
    let mut out = String::new();
    for line in tsv.lines() {
        // read_range 每行末尾会多一个分隔符（拼接风格），先剥掉再切列，
        // 避免尾部多出一个空字段（表现为行尾多一个逗号）。
        let line = line.trim_end_matches('\t');
        let fields: Vec<&str> = line.split('\t').collect();
        let row: Vec<String> = fields.iter().map(|f| csv_escape(f)).collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }
    out
}

/// CSV 字段转义：含逗号/引号/换行时用双引号包裹并把 `"` 转成 `""`。
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}
