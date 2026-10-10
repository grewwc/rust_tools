pub(crate) use std::io::{self, BufRead, Read, Write};
pub(crate) use std::net::{TcpListener, TcpStream};
pub(crate) use std::path::PathBuf;
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) use std::sync::mpsc;
pub(crate) use std::sync::{Mutex, OnceLock};
pub(crate) use std::time::{Duration, Instant};
pub(crate) use std::{fs, os::unix::fs::OpenOptionsExt, os::unix::fs::PermissionsExt};

pub(crate) use reqwest::blocking::Client;
pub(crate) use rust_tools::commonw::{FastMap, FastSet, configw};
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::{Value, json};
mod auth;
mod doc_create;
mod docs_read;
mod docx_render;
mod jsonrpc;
mod markdown;
mod oauth;
mod sheet_create;
mod token_store;
#[cfg(test)]
mod tests;

pub(crate) use auth::*;
pub(crate) use doc_create::*;
pub(crate) use docs_read::*;
pub(crate) use docx_render::*;
pub(crate) use jsonrpc::*;
pub(crate) use markdown::*;
pub(crate) use oauth::*;
pub(crate) use sheet_create::*;
pub(crate) use token_store::*;

pub(crate) const FEISHU_SCOPE: &str = "sheets:spreadsheet:write_only sheets:spreadsheet:create docs:document:export docx:document:readonly wiki:node:read wiki:node:retrieve sheets:spreadsheet:read offline_access contact:user.employee_id:readonly im:message:send_as_bot drive:drive.metadata:readonly docs:document.media:download base:app:create base:app:read base:app:update base:field:create base:field:delete base:field:update base:record:create base:record:delete base:record:update base:table:create base:table:delete base:table:update base:view:read base:view:write_only drive:file:download drive:file:upload docs:document.media:upload docx:document:create docx:document:write_only contact:contact.base:readonly board:whiteboard:node:read docs:document.comment:read base:table:read base:record:retrieve base:field:read im:chat:read im:chat.members:read docs:document.comment:update docs:document.comment:create task:task:read calendar:calendar.event:read calendar:calendar:read contact:user.base:readonly base:app:copy search:docs:read";


fn main() {
    // rustls has no default crypto provider (reqwest uses rustls-no-provider;
    // see root Cargo.toml); install ring before any TLS use.
    rust_tools::ensure_rustls_provider();
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
            "name": "mcp-feishu",
            "version": "0.1.0"
        }
    }))
}

fn handle_tools_list() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "tools": [
            {
                "name": "docs_get_text",
                "description": "Fetch plain text content for a Feishu doc/docx/wiki/sheet. Supports wiki (auto-resolves to underlying object) and sheet (preview). Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "docs_token": { "type": "string", "description": "docs_token from docs_search or URL" },
                        "docs_type": { "type": "string", "description": "doc, docx, wiki, or sheet. If omitted, inferred from token pattern." },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" },
                        "max_rows": { "type": "integer", "description": "For sheets: preview max rows per sheet (default 50, max 500)" },
                        "max_cols": { "type": "integer", "description": "For sheets: preview max columns per sheet (default 20, max 200)" },
                        "max_sheets": { "type": "integer", "description": "For sheets: preview max sheets (default 3, max 20)" }
                    },
                    "required": ["docs_token"]
                }
            },
            {
                "name": "docs_export_text",
                "description": "Export plain text content for a Feishu doc/docx to a local file and return the path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "docs_token": { "type": "string", "description": "docs_token from docs_search" },
                        "docs_type": { "type": "string", "description": "doc or docx" },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" },
                        "out_dir": { "type": "string", "description": "Optional output directory. Default: ~/.config/rust_tools/feishu_docs_text" }
                    },
                    "required": ["docs_token", "docs_type"]
                }
            },
            {
                "name": "docs_get_text_by_url",
                "description": "Fetch plain text content for a Feishu/Lark URL. Supports wiki/doc/docx/sheets URLs. For wiki URLs, resolves node to underlying object and then fetches content (doc/docx raw_content, sheets preview). Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "Feishu/Lark docs URL, e.g. https://bytedance.larkoffice.com/wiki/<token> or https://xxx.feishu.cn/docx/<token>" },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" },
                        "max_rows": { "type": "integer", "description": "For sheets: preview max rows per sheet (default 50, max 500)" },
                        "max_cols": { "type": "integer", "description": "For sheets: preview max columns per sheet (default 20, max 200)" },
                        "max_sheets": { "type": "integer", "description": "For sheets: preview max sheets (default 3, max 20)" }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "oauth_authorize_url",
                "description": "Build Feishu OAuth authorize URL to obtain code (for user_access_token). You must configure redirect_uri in Feishu app console.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "redirect_uri": { "type": "string", "description": "Redirect URI configured in Feishu console. Default: http://127.0.0.1:8711/callback" },
                        "scope": { "type": "string", "description": "Optional. Space-separated scopes. If omitted, defaults to the full scope set required by this MCP server's tools (docs/docx/sheets/wiki/im/base/etc.). Only pass this to request a custom subset; a narrower set may make some tools fail." },
                        "state": { "type": "string", "description": "Opaque state string for CSRF protection" },
                        "prompt": { "type": "string", "description": "Optional. Use \"consent\" to force explicit consent UI." }
                    }
                }
            },
            {
                "name": "oauth_wait_local_code",
                "description": "Start a local HTTP listener and wait for OAuth redirect to capture code. Use with redirect_uri=http://127.0.0.1:<port>/callback",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "port": { "type": "integer", "description": "Local port to listen on (default 8711)" },
                        "timeout_sec": { "type": "integer", "description": "Wait timeout in seconds (default 180)" }
                    }
                }
            },
            {
                "name": "oauth_exchange_code",
                "description": "Exchange OAuth code for user_access_token (requires client_id/client_secret; legacy app_id/app_secret is still accepted). Returns user_access_token and refresh_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": { "type": "string", "description": "OAuth authorization code (valid ~5 minutes, single-use)" },
                        "redirect_uri": { "type": "string", "description": "Must match the redirect_uri used in oauth_authorize_url. Default: http://127.0.0.1:8711/callback" }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "oauth_refresh_user_access_token",
                "description": "Refresh user_access_token using refresh_token (requires client_id/client_secret; legacy app_id/app_secret is still accepted).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh_token": { "type": "string", "description": "Refresh token. If omitted, uses FEISHU_REFRESH_TOKEN env or feishu.refresh_token in ~/.configW" }
                    }
                }
            },
            {
                "name": "sheet_create_from_csv",
                "description": "Create a new Feishu spreadsheet from CSV content and return the spreadsheet URL. Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Spreadsheet title" },
                        "csv_content": { "type": "string", "description": "CSV content to import" },
                        "folder_token": { "type": "string", "description": "Optional folder token to store the spreadsheet" }
                    },
                    "required": ["title", "csv_content"]
                }
            },
            {
                "name": "doc_create_from_markdown",
                "description": "Create a new Feishu docx document from Markdown content and return the document URL. Supports headings, lists, code blocks, tables, quotes, todo, dividers, and inline bold/italic/code. Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Document title" },
                        "markdown_content": { "type": "string", "description": "Markdown content to import" },
                        "folder_token": { "type": "string", "description": "Optional folder token to store the document" }
                    },
                    "required": ["title", "markdown_content"]
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
        "docs_get_text" => {
            let text = feishu_docs_get_text(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "docs_export_text" => {
            let text = feishu_docs_export_text(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "docs_get_text_by_url" => {
            let text = feishu_docs_get_text_by_url(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_authorize_url" => {
            let text = feishu_oauth_authorize_url(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_wait_local_code" => {
            let text = feishu_oauth_wait_local_code(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_exchange_code" => {
            let text = feishu_oauth_exchange_code(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "oauth_refresh_user_access_token" => {
            let text = feishu_oauth_refresh_user_access_token(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "sheet_create_from_csv" => {
            let result = feishu_sheet_create_from_csv(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": result }
                ]
            }))
        }
        "doc_create_from_markdown" => {
            let result = feishu_doc_create_from_markdown(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": result }
                ]
            }))
        }
        _ => Err(json_rpc_error(
            -32602,
            "Invalid params: unknown tool name",
            Some(json!({ "name": name })),
        )),
    }
}
