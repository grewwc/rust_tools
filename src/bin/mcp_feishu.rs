use std::io::{self, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{fs, os::unix::fs::PermissionsExt};

use reqwest::blocking::Client;
use rust_tools::commonw::{FastMap, configw};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

struct CachedToken {
    token: String,
    expires_at: Instant,
}

static USER_TOKEN_CACHE: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();

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
            "name": "mcp-feishu",
            "version": "0.1.0"
        }
    }))
}

fn handle_tools_list() -> Result<Value, JsonRpcErr> {
    Ok(json!({
        "tools": [
            {
                "name": "docs_search",
                "description": "Search Feishu cloud docs by keyword (requires user_access_token; supports OAuth + local token store for auto refresh)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_key": { "type": "string", "description": "Search keyword" },
                        "count": { "type": "integer", "description": "Number of results (0-50)" },
                        "offset": { "type": "integer", "description": "Offset (offset + count < 200)" },
                        "owner_ids": { "type": "array", "items": { "type": "string" }, "description": "Owner open_ids" },
                        "chat_ids": { "type": "array", "items": { "type": "string" }, "description": "Chat IDs" },
                        "docs_types": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "doc/sheet/slides/bitable/mindnote/file"
                        }
                    },
                    "required": ["search_key"]
                }
            },
            {
                "name": "messages_search",
                "description": "Search Feishu chat or thread messages by keyword from the authorized user's perspective. This works on Feishu IM messages, not docs. Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_key": { "type": "string", "description": "Keyword to search in message content" },
                        "chat_id": { "type": "string", "description": "Target chat_id. Use this for normal group/private chats." },
                        "thread_id": { "type": "string", "description": "Target thread/topic id. Use this instead of chat_id when searching a thread." },
                        "start_time": { "type": "string", "description": "Optional start time in milliseconds" },
                        "end_time": { "type": "string", "description": "Optional end time in milliseconds" },
                        "msg_types": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional message type filter, e.g. text/post/interactive/file/image/media/audio/sticker/share_chat/share_user/system"
                        },
                        "sort_order": { "type": "string", "description": "desc or asc. Default: desc" },
                        "page_size": { "type": "integer", "description": "Messages fetched per API call. Default 50, max 50" },
                        "max_pages": { "type": "integer", "description": "How many pages to scan at most. Default 5, max 20" },
                        "limit": { "type": "integer", "description": "Maximum matched messages to return. Default 20, max 100" }
                    },
                    "required": ["search_key"]
                }
            },
            {
                "name": "messages_global_search",
                "description": "Search Feishu IM messages across chats visible to the authorized user, without needing a specific chat_id. Supports substring, regex, and fuzzy matching modes. Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_key": { "type": "string", "description": "Keyword or pattern to search in message content" },
                        "max_chats": { "type": "integer", "description": "Maximum number of recent chats to scan. Default 50, max 200" },
                        "msgs_per_chat": { "type": "integer", "description": "Max messages per API page per chat. Default 50, max 50" },
                        "max_pages": { "type": "integer", "description": "Max pages to fetch per chat. Default 10, max 50. Total messages per chat = msgs_per_chat * max_pages" },
                        "msg_types": { "type": "array", "items": { "type": "string" }, "description": "Optional message type filter, e.g. text/post/interactive/file/image/media/audio/sticker/share_chat/share_user/system" },
                        "limit": { "type": "integer", "description": "Maximum matched messages to return. Default 20, max 100" },
                        "include_p2p": { "type": "boolean", "description": "Include 1-on-1 chats. Default true" },
                        "include_group": { "type": "boolean", "description": "Include group chats. Default true" },
                        "search_mode": { "type": "string", "enum": ["substring", "regex", "fuzzy"], "description": "Search mode: substring (default, case-insensitive), regex (Rust regex), fuzzy (edit distance <= 2)" }
                    },
                    "required": ["search_key"]
                }
            },
            {
                "name": "docs_get_text",
                "description": "Fetch plain text content for a Feishu doc/docx (raw_content). Requires user_access_token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "docs_token": { "type": "string", "description": "docs_token from docs_search" },
                        "docs_type": { "type": "string", "description": "doc or docx" },
                        "lang": { "type": "integer", "description": "docx raw_content lang: 0=zh,1=en,2=ja (default 0)" }
                    },
                    "required": ["docs_token", "docs_type"]
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
                        "scope": { "type": "string", "description": "Scopes separated by space. Default: offline_access im:chat:readonly im:message:readonly docs:doc:readonly docx:document:readonly wiki:wiki:readonly sheets:spreadsheet" },
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
                        "code": { "type": "string", "description": "OAuth authorization code (valid ~5 minutes, single-use)" }
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
        "docs_search" => {
            let text = feishu_docs_search(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "messages_search" => {
            let text = feishu_messages_search(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
        "messages_global_search" => {
            let text = feishu_messages_global_search(&args)?;
            Ok(json!({
                "content": [
                    { "type": "text", "text": text }
                ]
            }))
        }
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

fn feishu_docs_search(args: &Value) -> Result<String, JsonRpcErr> {
    let base_url = resolve_base_url();

    let search_key = args
        .get("search_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if search_key.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: search_key is empty",
            None,
        ));
    }

    let mut body = json!({
        "search_key": search_key,
        "count": args.get("count").and_then(|v| v.as_i64()).unwrap_or(10).clamp(0, 50),
        "offset": args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0).max(0),
    });

    if let Some(v) = args.get("owner_ids") {
        body["owner_ids"] = v.clone();
    }
    if let Some(v) = args.get("chat_ids") {
        body["chat_ids"] = v.clone();
    }
    if let Some(v) = args.get("docs_types") {
        body["docs_types"] = v.clone();
    }

    let url = format!(
        "{}/open-apis/suite/docs-api/search/object",
        base_url.trim_end_matches('/')
    );
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. docs-api search requires OAuth once.",
        |token| feishu_docs_search_with_token(&client, &url, token, &body),
    )
}

fn feishu_docs_search_with_token(
    client: &Client,
    url: &str,
    token: &str,
    body: &Value,
) -> Result<String, JsonRpcErr> {
    let (status, text) = do_docs_search_request(client, url, token, body)?;
    if !status.is_success() {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
            if code == 99991668 {
                return Err(json_rpc_error(
                    -32000,
                    "Invalid access token",
                    Some(json!({
                        "status": status.as_u16(),
                        "feishu_code": code,
                        "msg": v.get("msg").cloned().unwrap_or(Value::Null),
                    })),
                ));
            }
        }
        return Err(json_rpc_error(
            -32000,
            "Feishu API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Feishu API response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;

    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "Feishu API returned error code",
            Some(v),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let total = data.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
    let has_more = data
        .get("has_more")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let docs = data
        .get("docs_entities")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut out = String::new();
    out.push_str(&format!("total: {total}, has_more: {has_more}\n"));
    for (i, item) in docs.iter().enumerate() {
        let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let docs_type = item.get("docs_type").and_then(|v| v.as_str()).unwrap_or("");
        let docs_token = item
            .get("docs_token")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let owner_id = item.get("owner_id").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(&format!(
            "{}. [{}] {} (token: {}, owner_id: {})\n",
            i + 1,
            docs_type,
            title.trim(),
            docs_token.trim(),
            owner_id.trim()
        ));
    }
    Ok(out.trim_end().to_string())
}

fn feishu_messages_search(args: &Value) -> Result<String, JsonRpcErr> {
    let search_key = args
        .get("search_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if search_key.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: search_key is empty",
            None,
        ));
    }

    let (container_id_type, container_id) = resolve_message_container(args)?;
    let msg_type_filters = extract_string_array(args.get("msg_types"));
    let sort_type = match args
        .get("sort_order")
        .and_then(|v| v.as_str())
        .unwrap_or("desc")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "asc" => "ByCreateTimeAsc",
        _ => "ByCreateTimeDesc",
    };
    let page_size = args
        .get("page_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(50)
        .clamp(1, 50) as usize;
    let max_pages = args
        .get("max_pages")
        .and_then(|v| v.as_i64())
        .unwrap_or(5)
        .clamp(1, 20) as usize;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let start_time = get_optional_arg_string(args, "start_time");
    let end_time = get_optional_arg_string(args, "end_time");

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Message search requires OAuth once.",
        |token| {
            feishu_messages_search_with_token(
                &client,
                &base_url,
                token,
                &search_key,
                container_id_type,
                &container_id,
                &msg_type_filters,
                sort_type,
                page_size,
                max_pages,
                limit,
                start_time.as_deref(),
                end_time.as_deref(),
            )
        },
    )
}

fn feishu_messages_search_with_token(
    client: &Client,
    base_url: &str,
    token: &str,
    search_key: &str,
    container_id_type: &str,
    container_id: &str,
    msg_type_filters: &[String],
    sort_type: &str,
    page_size: usize,
    max_pages: usize,
    limit: usize,
    start_time: Option<&str>,
    end_time: Option<&str>,
) -> Result<String, JsonRpcErr> {
    let mut page_token: Option<String> = None;
    let mut matches: Vec<String> = Vec::new();
    let mut scanned_messages = 0usize;
    let mut scanned_pages = 0usize;
    let needle = search_key.to_lowercase();

    while scanned_pages < max_pages && matches.len() < limit {
        let (status, text) = do_messages_list_request(
            &client,
            &base_url,
            &token,
            container_id_type,
            &container_id,
            sort_type,
            page_size,
            page_token.as_deref(),
            start_time.as_deref(),
            end_time.as_deref(),
        )?;
        if !status.is_success() {
            return Err(json_rpc_error(
                -32000,
                "messages API returned non-success HTTP status",
                Some(json!({ "status": status.as_u16(), "body": text })),
            ));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| {
            json_rpc_error(
                -32000,
                "messages API response is not valid JSON",
                Some(json!({ "error": e.to_string(), "body": text })),
            )
        })?;
        let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(
                -32000,
                "messages API returned error code",
                Some(v),
            ));
        }

        let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
        let items = data
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let has_more = data
            .get("has_more")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let next_page_token = data
            .get("page_token")
            .and_then(|v| v.as_str())
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        scanned_pages += 1;
        scanned_messages += items.len();

        for item in &items {
            let msg_type = item
                .get("msg_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if !msg_type_filters.is_empty()
                && !msg_type_filters.iter().any(|allowed| allowed == &msg_type)
            {
                continue;
            }

            let searchable_text = build_feishu_message_searchable_text(item);
            if searchable_text.is_empty() || !searchable_text.to_lowercase().contains(&needle) {
                continue;
            }
            matches.push(render_feishu_message_hit(
                item,
                &searchable_text,
                matches.len() + 1,
            ));
            if matches.len() >= limit {
                break;
            }
        }

        if !has_more || next_page_token.is_none() {
            break;
        }
        page_token = next_page_token;
    }

    let mut out = String::new();
    out.push_str(&format!(
        "search_key: {}\ncontainer: {} {}\nscanned_pages: {}\nscanned_messages: {}\nmatched: {}",
        search_key,
        container_id_type,
        container_id,
        scanned_pages,
        scanned_messages,
        matches.len()
    ));
    if !msg_type_filters.is_empty() {
        out.push_str(&format!("\nmsg_types: {}", msg_type_filters.join(", ")));
    }
    if matches.is_empty() {
        out.push_str("\n\nNo matched message found.");
        return Ok(out);
    }
    out.push_str("\n\n");
    out.push_str(&matches.join("\n\n"));
    Ok(out)
}

/// Fuzzy match using Levenshtein distance. Returns true if the edit distance
/// between `text` and `pattern` is <= `max_distance`.
fn fuzzy_match(text: &str, pattern: &str, max_distance: usize) -> bool {
    let t_chars: Vec<char> = text.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();
    let t_len = t_chars.len();
    let p_len = p_chars.len();
    
    if p_len == 0 { return true; }
    if t_len == 0 { return false; }
    
    // Use sliding window for efficiency when pattern is much shorter than text
    let mut min_dist = usize::MAX;
    for i in 0..=(t_len.saturating_sub(p_len)) {
        let end = (i + p_len).min(t_len);
        let window = &t_chars[i..end];
        let dist = levenshtein(window, &p_chars);
        min_dist = min_dist.min(dist);
        if min_dist <= max_distance { return true; }
    }
    min_dist <= max_distance
}

/// Compute Levenshtein (edit) distance between two char slices.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)          // deletion
                .min(curr[j - 1] + 1)         // insertion
                .min(prev[j - 1] + cost);     // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[derive(Debug, Clone)]
struct ChatInfo {
    chat_id: String,
    name: String,
    recent_time: i64,
    source_order: usize,
}

/// List recent chats using the Feishu API.
fn list_recent_chats(
    client: &Client,
    base_url: &str,
    access_token: &str,
    max_chats: usize,
    include_p2p: bool,
    include_group: bool,
) -> Result<Vec<ChatInfo>, JsonRpcErr> {
    let url = format!("{}/open-apis/im/v1/chats", base_url);

    let max_chat_pages = (max_chats / 50 + 2).max(2);
    let mut all_chats = Vec::new();
    let mut source_order = 0usize;
    let mut page_token = String::new();

    for _ in 0..max_chat_pages {
        let mut req = client.get(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .query(&[("page_size", "50")]);

        if !page_token.is_empty() {
            req = req.query(&[("page_token", &page_token)]);
        }

        let resp = req.send().map_err(|e| {
            json_rpc_error(-32000, "Failed to fetch chats", Some(json!({ "error": e.to_string() })))
        })?;

        let (status, content_type, body_text) = read_response_text(resp, "chats response")?;
        let body = parse_json_response_body("chats response", status, content_type.as_deref(), &body_text)?;

        if !status.is_success() {
            return Err(json_rpc_error(-32000, "Chats list API returned error code", Some(body)));
        }

        let code = body.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(-32000, &format!("Chats API error: code={}", code), Some(body)));
        }

        let data = body.get("data").cloned().unwrap_or_else(|| json!({}));
        let items = data.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        for item in &items {
            let chat_id = item.get("chat_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let chat_mode = item.get("chat_mode").and_then(|v| v.as_str()).unwrap_or("");

            if !chat_mode_matches(chat_mode, include_p2p, include_group) {
                continue;
            }
            all_chats.push(ChatInfo {
                chat_id,
                name,
                recent_time: chat_recent_time(item),
                source_order,
            });
            source_order += 1;
        }

        if all_chats.len() >= max_chats {
            break;
        }

        page_token = data.get("page_token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if page_token.is_empty() || items.is_empty() {
            break;
        }
    }

    all_chats.sort_by(|a, b| {
        b.recent_time
            .cmp(&a.recent_time)
            .then_with(|| a.source_order.cmp(&b.source_order))
    });
    if all_chats.len() > max_chats {
        all_chats.truncate(max_chats);
    }
    Ok(all_chats)
}

/// Fetch messages from a specific chat.
fn fetch_chat_messages(
    client: &Client,
    base_url: &str,
    access_token: &str,
    chat_id: &str,
    msgs_per_chat: u64,
    max_pages: u64,
) -> Result<Vec<Value>, JsonRpcErr> {
    let mut all_messages = Vec::new();
    let mut page_token: Option<String> = None;

    for _page in 0..max_pages {
        let (status, body_text) = do_messages_list_request(
            client,
            base_url,
            access_token,
            "chat",
            chat_id,
            "ByCreateTimeDesc",
            msgs_per_chat as usize,
            page_token.as_deref(),
            None,
            None,
        )?;
        if !status.is_success() {
            return Err(json_rpc_error(
                -32000,
                "Messages list API returned error code",
                Some(json!({
                    "status": status.as_u16(),
                    "body": body_text
                })),
            ));
        }

        let body = parse_json_response_body(
            "messages response",
            status,
            Some("application/json"),
            &body_text,
        )?;

        let code = body.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(
                -32000,
                &format!("Messages API error: code={}", code),
                Some(body),
            ));
        }
        
        let data = body.get("data").cloned().unwrap_or_else(|| json!({}));
        let items = data.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        
        let is_empty = items.is_empty();
        all_messages.extend(items);
        
        page_token = data
            .get("page_token")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string())
            .filter(|v| !v.is_empty());
        if page_token.is_none() || is_empty {
            break;
        }
    }
    
    Ok(all_messages)
}

fn read_response_text(
    resp: reqwest::blocking::Response,
    label: &str,
) -> Result<(reqwest::StatusCode, Option<String>, String), JsonRpcErr> {
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_string());
    let body = resp.bytes().map_err(|e| {
        json_rpc_error(
            -32000,
            &format!("Failed to read {}", label),
            Some(json!({
                "error": e.to_string(),
                "status": status.as_u16(),
                "content_type": content_type,
            })),
        )
    })?;
    Ok((
        status,
        content_type,
        String::from_utf8_lossy(&body).to_string(),
    ))
}

fn parse_json_response_body(
    label: &str,
    status: reqwest::StatusCode,
    content_type: Option<&str>,
    body_text: &str,
) -> Result<Value, JsonRpcErr> {
    serde_json::from_str::<Value>(body_text).map_err(|e| {
        json_rpc_error(
            -32000,
            &format!("Failed to parse {}", label),
            Some(json!({
                "error": e.to_string(),
                "status": status.as_u16(),
                "content_type": content_type,
                "body": truncate_for_error_body(body_text, 1200),
            })),
        )
    })
}

fn truncate_for_error_body(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in s.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn feishu_messages_global_search(args: &Value) -> Result<String, JsonRpcErr> {
    let search_key = args
        .get("search_key")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if search_key.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: search_key is empty", None));
    }

    let max_chats: usize = args.get("max_chats").and_then(|v| v.as_u64()).unwrap_or(50).min(200) as usize;
    let msgs_per_chat: u64 = args.get("msgs_per_chat").and_then(|v| v.as_u64()).unwrap_or(50).min(50);
    let max_pages: u64 = args.get("max_pages").and_then(|v| v.as_u64()).unwrap_or(10).min(50);
    let limit: usize = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20).min(100) as usize;
    let include_p2p = args.get("include_p2p").and_then(|v| v.as_bool()).unwrap_or(true);
    let include_group = args.get("include_group").and_then(|v| v.as_bool()).unwrap_or(true);
    let msg_type_filters = extract_string_array(args.get("msg_types"));

    let search_mode = args
        .get("search_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("substring");

    // Prepare matcher based on search_mode
    let regex_opt = if search_mode == "regex" {
        Some(regex::Regex::new(&search_key).map_err(|e| {
            json_rpc_error(-32602, &format!("Invalid regex pattern: {}", e), None)
        })?)
    } else {
        None
    };
    let needle_lower = search_key.to_lowercase();
    let fuzzy_threshold = if search_mode == "fuzzy" { 2 } else { 0 };

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(-32000, "Failed to build http client", Some(json!({ "error": e.to_string() })))
        })?;

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Global message search requires OAuth once.",
        |token| {
            feishu_messages_global_search_with_token(
                &client,
                &base_url,
                token,
                search_key,
                max_chats,
                msgs_per_chat,
                max_pages,
                limit,
                include_p2p,
                include_group,
                &msg_type_filters,
                search_mode,
                regex_opt.as_ref(),
                &needle_lower,
                fuzzy_threshold,
            )
        },
    )
}

fn feishu_messages_global_search_with_token(
    client: &Client,
    base_url: &str,
    access_token: &str,
    search_key: &str,
    max_chats: usize,
    msgs_per_chat: u64,
    max_pages: u64,
    limit: usize,
    include_p2p: bool,
    include_group: bool,
    msg_type_filters: &[String],
    search_mode: &str,
    regex_opt: Option<&regex::Regex>,
    needle_lower: &str,
    fuzzy_threshold: usize,
) -> Result<String, JsonRpcErr> {
    let chats = list_recent_chats(
        client,
        base_url,
        access_token,
        max_chats,
        include_p2p,
        include_group,
    )?;

    let concurrency = 8usize.min(chats.len().max(1));
    let total_chats = chats.len();

    let mut matches: Vec<(i64, String)> = Vec::new();
    let mut scanned_chats = 0usize;
    let mut scanned_messages = 0usize;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(concurrency)
        .enable_all()
        .build()
        .map_err(|e| {
            json_rpc_error(-32000, "Failed to build tokio runtime", Some(json!({ "error": e.to_string() })))
        })?;

    let base_url_owned = base_url.to_string();
    let access_token_owned = access_token.to_string();
    let msg_type_filters_owned: Vec<String> = msg_type_filters.to_vec();
    let search_mode_owned = search_mode.to_string();
    let needle_lower_owned = needle_lower.to_string();
    let regex_pattern_owned = regex_opt.map(|re| re.as_str().to_string());

    for batch in chats.chunks(concurrency) {
        let batch_chats: Vec<ChatInfo> = batch.to_vec();

        let batch_results: Vec<Result<(usize, Vec<(i64, String)>), JsonRpcErr>> = rt.block_on(async {
            let mut handles = Vec::with_capacity(batch_chats.len());
            for chat in batch_chats {
                let base_url_c = base_url_owned.clone();
                let access_token_c = access_token_owned.clone();
                let msg_type_filters_c = msg_type_filters_owned.clone();
                let search_mode_c = search_mode_owned.clone();
                let needle_lower_c = needle_lower_owned.clone();
                let regex_pattern_c = regex_pattern_owned.clone();
                handles.push(tokio::task::spawn_blocking(move || {
                    let thread_client = Client::builder()
                        .timeout(Duration::from_secs(12))
                        .build()
                        .map_err(|e| {
                            json_rpc_error(
                                -32000,
                                "Failed to build http client",
                                Some(json!({ "error": e.to_string() })),
                            )
                        })?;
                    let re = regex_pattern_c.as_deref().map(|p| regex::Regex::new(p).unwrap());
                    let mut local_matches: Vec<(i64, String)> = Vec::new();
                    let mut local_scanned_messages = 0usize;
                    scan_chat_messages_for_matches(
                        &thread_client,
                        &base_url_c,
                        &access_token_c,
                        &chat,
                        msgs_per_chat,
                        max_pages,
                        &msg_type_filters_c,
                        &search_mode_c,
                        re.as_ref(),
                        &needle_lower_c,
                        fuzzy_threshold,
                        limit,
                        &mut local_scanned_messages,
                        &mut local_matches,
                    )?;
                    Ok((local_scanned_messages, local_matches))
                }));
            }
            let mut results = Vec::with_capacity(handles.len());
            for h in handles {
                results.push(h.await.unwrap_or_else(|e| {
                    Err(json_rpc_error(-32000, "Task panicked", Some(json!({ "error": e.to_string() }))))
                }));
            }
            results
        });

        for result in batch_results {
            let (sm, local_matches) = result?;
            scanned_chats += 1;
            scanned_messages += sm;
            matches.extend(local_matches);
        }
        matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        if matches.len() > limit {
            matches.truncate(limit);
        }

        if let Some(next_chat) = chats.get(scanned_chats) {
            if should_stop_before_chat(next_chat, &matches, limit) {
                break;
            }
        }
    }

    let mut out = String::new();
    out.push_str(&format!(
        "search_key: {} (mode: {})\nscanned_chats: {}/{}\nscanned_messages: {}\nmatched: {}",
        search_key, search_mode, scanned_chats, total_chats, scanned_messages, matches.len()
    ));
    if !msg_type_filters.is_empty() {
        out.push_str(&format!("\nfiltered_msg_types: {:?}", msg_type_filters));
    }
    out.push_str("\n\n");
    out.push_str(
        &matches
            .into_iter()
            .map(|(_, rendered)| rendered)
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    Ok(out)
}

fn scan_chat_messages_for_matches(
    client: &Client,
    base_url: &str,
    access_token: &str,
    chat: &ChatInfo,
    msgs_per_chat: u64,
    max_pages: u64,
    msg_type_filters: &[String],
    search_mode: &str,
    regex_opt: Option<&regex::Regex>,
    needle_lower: &str,
    fuzzy_threshold: usize,
    limit: usize,
    scanned_messages: &mut usize,
    matches: &mut Vec<(i64, String)>,
) -> Result<(), JsonRpcErr> {
    let mut page_token: Option<String> = None;
    for _ in 0..max_pages {
        if should_stop_inside_chat(None, matches, limit) {
            break;
        }
        let (status, body_text) = do_messages_list_request(
            client,
            base_url,
            access_token,
            "chat",
            &chat.chat_id,
            "ByCreateTimeDesc",
            msgs_per_chat as usize,
            page_token.as_deref(),
            None,
            None,
        )?;
        if !status.is_success() {
            return Err(json_rpc_error(-32000, "Messages list API returned error code", Some(json!({
                "status": status.as_u16(),
                "body": body_text
            }))));
        }
        let body = parse_json_response_body(
            "messages response",
            status,
            Some("application/json"),
            &body_text,
        )?;
        let code = body.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(
                -32000,
                &format!("Messages API error: code={}", code),
                Some(body),
            ));
        }
        let data = body.get("data").cloned().unwrap_or_else(|| json!({}));
        let items = data.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let is_empty = items.is_empty();
        let oldest_in_page = items.iter().map(message_create_time).min();

        for item in &items {
            *scanned_messages += 1;
            if !msg_type_filters.is_empty() {
                let msg_type = item.get("msg_type").and_then(|v| v.as_str()).unwrap_or("");
                if !msg_type_filters.iter().any(|f| f == msg_type) {
                    continue;
                }
            }
            let searchable_text = build_feishu_global_searchable_text(item, chat);
            if searchable_text.is_empty() {
                continue;
            }
            let matched = match search_mode {
                "regex" => regex_opt.map(|re| re.is_match(&searchable_text)).unwrap_or(false),
                "fuzzy" => fuzzy_match(&searchable_text, needle_lower, fuzzy_threshold),
                _ => searchable_text.to_lowercase().contains(needle_lower),
            };
            if !matched {
                continue;
            }
            let mut enriched = item.clone();
            if let Some(obj) = enriched.as_object_mut() {
                obj.insert("chat_name".to_string(), Value::String(chat.name.clone()));
                obj.insert("chat_id".to_string(), Value::String(chat.chat_id.clone()));
            }
            push_global_match(
                matches,
                message_create_time(item),
                serde_json::to_string_pretty(&enriched).unwrap_or_else(|_| "<error>".to_string()),
                limit,
            );
        }

        if should_stop_inside_chat(oldest_in_page, matches, limit) {
            break;
        }

        page_token = data
            .get("page_token")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string())
            .filter(|v| !v.is_empty());
        if page_token.is_none() || is_empty {
            break;
        }
    }
    Ok(())
}

fn push_global_match(
    matches: &mut Vec<(i64, String)>,
    create_time: i64,
    rendered: String,
    limit: usize,
) {
    matches.push((create_time, rendered));
    matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if matches.len() > limit {
        matches.truncate(limit);
    }
}

fn current_kth_match_time(matches: &[(i64, String)], limit: usize) -> Option<i64> {
    if matches.len() < limit {
        None
    } else {
        matches.last().map(|(ts, _)| *ts)
    }
}

fn should_stop_before_chat(chat: &ChatInfo, matches: &[(i64, String)], limit: usize) -> bool {
    current_kth_match_time(matches, limit)
        .is_some_and(|kth| chat.recent_time > 0 && chat.recent_time <= kth)
}

fn should_stop_inside_chat(
    oldest_in_page: Option<i64>,
    matches: &[(i64, String)],
    limit: usize,
) -> bool {
    match (oldest_in_page, current_kth_match_time(matches, limit)) {
        (Some(oldest), Some(kth)) => oldest <= kth,
        _ => false,
    }
}

fn message_create_time(item: &Value) -> i64 {
    item.get("create_time")
        .or_else(|| item.get("last_message_time"))
        .or_else(|| item.get("latest_message_time"))
        .or_else(|| item.get("update_time"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0)
}

fn chat_recent_time(item: &Value) -> i64 {
    item.get("last_message_time")
        .or_else(|| item.get("latest_message_time"))
        .or_else(|| item.get("update_time"))
        .or_else(|| item.get("create_time"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0)
}

fn chat_mode_matches(chat_mode: &str, include_p2p: bool, include_group: bool) -> bool {
    match chat_mode {
        "p2p" => include_p2p,
        "group" | "topic" => include_group,
        _ => include_group,
    }
}

fn resolve_message_container(args: &Value) -> Result<(&'static str, String), JsonRpcErr> {
    let chat_id = args
        .get("chat_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let thread_id = args
        .get("thread_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if !chat_id.is_empty() && !thread_id.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: chat_id and thread_id cannot both be set",
            None,
        ));
    }
    if !thread_id.is_empty() {
        return Ok(("thread", thread_id));
    }
    if !chat_id.is_empty() {
        return Ok(("chat", chat_id));
    }
    Err(json_rpc_error(
        -32602,
        "Invalid params: chat_id or thread_id is required",
        None,
    ))
}

fn get_optional_arg_string(args: &Value, key: &str) -> Option<String> {
    let value = args.get(key)?;
    match value {
        Value::String(s) => {
            let s = s.trim().to_string();
            (!s.is_empty()).then_some(s)
        }
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn extract_string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn do_messages_list_request(
    client: &Client,
    base_url: &str,
    token: &str,
    container_id_type: &str,
    container_id: &str,
    sort_type: &str,
    page_size: usize,
    page_token: Option<&str>,
    start_time: Option<&str>,
    end_time: Option<&str>,
) -> Result<(reqwest::StatusCode, String), JsonRpcErr> {
    let mut url = format!(
        "{}/open-apis/im/v1/messages?container_id_type={}&container_id={}&sort_type={}&page_size={}",
        base_url.trim_end_matches('/'),
        url_encode_component(container_id_type),
        url_encode_component(container_id),
        url_encode_component(sort_type),
        page_size
    );
    if let Some(page_token) = page_token.filter(|v| !v.trim().is_empty()) {
        url.push_str("&page_token=");
        url.push_str(&url_encode_component(page_token.trim()));
    }
    if let Some(start_time) = start_time.filter(|v| !v.trim().is_empty()) {
        url.push_str("&start_time=");
        url.push_str(&url_encode_component(start_time.trim()));
    }
    if let Some(end_time) = end_time.filter(|v| !v.trim().is_empty()) {
        url.push_str("&end_time=");
        url.push_str(&url_encode_component(end_time.trim()));
    }

    let resp = client
        .get(url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call messages API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read messages API response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    Ok((status, text))
}

fn build_feishu_message_searchable_text(item: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(v) = item.get("body") {
        collect_message_text(v, &mut parts);
    }
    let merged = parts.join(" ");
    compact_message_text(&merged)
}

fn build_feishu_global_searchable_text(item: &Value, chat: &ChatInfo) -> String {
    let _ = chat;
    build_feishu_message_searchable_text(item)
}

fn collect_message_text(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return;
            }
            if (trimmed.starts_with('{') || trimmed.starts_with('['))
                && let Ok(parsed) = serde_json::from_str::<Value>(trimmed)
            {
                collect_message_text(&parsed, out);
                return;
            }
            if let Some((_, rest)) = trimmed.split_once(':')
                && matches!(
                    trimmed.split_once(':').map(|(prefix, _)| prefix),
                    Some("text" | "post" | "card" | "interactive")
                )
            {
                let rest = rest.trim();
                if !rest.is_empty() {
                    out.push(rest.to_string());
                    return;
                }
            }
            out.push(trimmed.to_string());
        }
        Value::Array(items) => {
            for item in items {
                collect_message_text(item, out);
            }
        }
        Value::Object(map) => {
            if map
                .get("tag")
                .and_then(|v| v.as_str())
                .is_some_and(|tag| tag == "a")
            {
                if let Some(text) = map.get("text") {
                    collect_message_text(text, out);
                }
                return;
            }
            for (key, value) in map {
                if should_skip_message_text_key(key) {
                    continue;
                }
                collect_message_text(value, out);
            }
        }
        _ => {}
    }
}

fn should_skip_message_text_key(key: &str) -> bool {
    matches!(
        key,
        "href" | "url" | "pc_url" | "ios_url" | "android_url" | "card_link"
    )
}

fn compact_message_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

fn render_feishu_message_hit(item: &Value, searchable_text: &str, index: usize) -> String {
    let message_id = item
        .get("message_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let chat_id = item
        .get("chat_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let root_id = item
        .get("root_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let parent_id = item
        .get("parent_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let upper_message_id = item
        .get("upper_message_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let msg_type = item
        .get("msg_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let create_time = item
        .get("create_time")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let sender = item
        .get("sender")
        .and_then(|v| v.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let preview = truncate_for_preview(searchable_text, 200);

    let mut meta = format!(
        "{}. [{}] create_time: {} | sender: {} | chat_id: {} | message_id: {}",
        index, msg_type, create_time, sender, chat_id, message_id
    );
    if !root_id.is_empty() {
        meta.push_str(&format!(" | root_id: {}", root_id));
    }
    if !parent_id.is_empty() {
        meta.push_str(&format!(" | parent_id: {}", parent_id));
    }
    if !upper_message_id.is_empty() {
        meta.push_str(&format!(" | upper_message_id: {}", upper_message_id));
    }
    format!("{meta}\n{preview}")
}

fn truncate_for_preview(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in s.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn feishu_docs_get_text(args: &Value) -> Result<String, JsonRpcErr> {
    let docs_token = args
        .get("docs_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let docs_type = args
        .get("docs_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if docs_token.is_empty() || docs_type.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: docs_token/docs_type required",
            Some(json!({ "docs_token": docs_token, "docs_type": docs_type })),
        ));
    }

    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| {
            feishu_fetch_raw_content(
                &client,
                &base_url,
                token,
                &docs_type,
                &docs_token,
                lang,
                None,
            )
        },
    )
}

fn feishu_docs_get_text_by_url(args: &Value) -> Result<String, JsonRpcErr> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if url.is_empty() {
        return Err(json_rpc_error(-32602, "Invalid params: url is empty", None));
    }

    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let max_rows = args
        .get("max_rows")
        .and_then(|v| v.as_i64())
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let max_cols = args
        .get("max_cols")
        .and_then(|v| v.as_i64())
        .unwrap_or(20)
        .clamp(1, 200) as usize;
    let max_sheets = args
        .get("max_sheets")
        .and_then(|v| v.as_i64())
        .unwrap_or(3)
        .clamp(1, 20) as usize;

    let base_url = resolve_base_url_for_user_url(&url);
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let Some((kind, tok)) = parse_docs_url_kind_and_token(&url) else {
        return Err(json_rpc_error(
            -32602,
            "Unsupported URL: failed to extract docs token/type",
            Some(json!({ "url": url })),
        ));
    };
    let default_origin = extract_url_origin(&url);

    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| match kind.as_str() {
            "wiki" => {
                let (docs_type, docs_token) =
                    feishu_wiki_resolve_obj(&client, &base_url, token, &tok)?;
                match docs_type.as_str() {
                    "doc" | "docx" => feishu_fetch_raw_content(
                        &client,
                        &base_url,
                        token,
                        &docs_type,
                        &docs_token,
                        lang,
                        default_origin.as_deref(),
                    ),
                    "sheet" => feishu_fetch_sheet_preview_text(
                        &client,
                        &base_url,
                        token,
                        &docs_token,
                        max_rows,
                        max_cols,
                        max_sheets,
                    ),
                    other => Err(json_rpc_error(
                        -32602,
                        "Unsupported wiki node object type (supported: doc/docx/sheet for now)",
                        Some(
                            json!({ "obj_type": other, "obj_token": docs_token, "node_token": tok }),
                        ),
                    )),
                }
            }
            "doc" | "docx" => feishu_fetch_raw_content(
                &client,
                &base_url,
                token,
                &kind,
                &tok,
                lang,
                default_origin.as_deref(),
            ),
            "sheet" => feishu_fetch_sheet_preview_text(
                &client, &base_url, token, &tok, max_rows, max_cols, max_sheets,
            ),
            other => Err(json_rpc_error(
                -32602,
                "Unsupported docs URL type (supported: wiki/doc/docx/sheets for now)",
                Some(json!({ "url": url, "parsed_type": other, "parsed_token": tok })),
            )),
        },
    )
}

fn feishu_docs_export_text(args: &Value) -> Result<String, JsonRpcErr> {
    let docs_token = args
        .get("docs_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let docs_type = args
        .get("docs_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if docs_token.is_empty() || docs_type.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: docs_token/docs_type required",
            Some(json!({ "docs_token": docs_token, "docs_type": docs_type })),
        ));
    }
    let lang = args
        .get("lang")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 2);

    let out_dir = args
        .get("out_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let out_dir = if out_dir.is_empty() {
        rust_tools::commonw::utils::expanduser("~/.config/rust_tools/feishu_docs_text")
            .as_ref()
            .to_string()
    } else {
        rust_tools::commonw::utils::expanduser(&out_dir)
            .as_ref()
            .to_string()
    };

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let content = with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Fetch requires OAuth once.",
        |token| {
            feishu_fetch_raw_content(
                &client,
                &base_url,
                token,
                &docs_type,
                &docs_token,
                lang,
                None,
            )
        },
    )?;

    let dir = PathBuf::from(&out_dir);
    fs::create_dir_all(&dir).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to create output directory",
            Some(json!({ "out_dir": out_dir, "error": e.to_string() })),
        )
    })?;
    let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));

    let safe_type = docs_type.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let safe_token = docs_token.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let file_path = dir.join(format!("{safe_type}_{safe_token}.txt"));
    fs::write(&file_path, &content).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to write exported file",
            Some(json!({ "file_path": file_path.display().to_string(), "error": e.to_string() })),
        )
    })?;
    let _ = fs::set_permissions(&file_path, fs::Permissions::from_mode(0o600));

    Ok(format!("exported: {}", file_path.display()))
}

fn resolve_base_url_for_user_url(url: &str) -> String {
    let u = url.to_lowercase();
    if u.contains("larkoffice.com") || u.contains("larksuite.com") {
        "https://open.larksuite.com".to_string()
    } else {
        resolve_base_url()
    }
}

fn parse_docs_url_kind_and_token(url: &str) -> Option<(String, String)> {
    let mut s = url.trim();
    if let Some(idx) = s.find("://") {
        s = &s[idx + 3..];
    }
    let s = s.split('?').next().unwrap_or(s);
    let s = s.split('#').next().unwrap_or(s);
    let path = if let Some(idx) = s.find('/') {
        &s[idx..]
    } else {
        s
    };
    let path = path.trim_start_matches('/');

    let segs = path
        .split('/')
        .filter(|p| !p.trim().is_empty())
        .collect::<Vec<_>>();
    if segs.len() < 2 {
        return None;
    }

    for i in 0..(segs.len().saturating_sub(1)) {
        let kind = segs[i].trim().to_lowercase();
        let token = segs[i + 1].trim();
        if token.is_empty() {
            continue;
        }
        let kind = match kind.as_str() {
            "wiki" => "wiki",
            "docx" => "docx",
            "doc" | "docs" => "doc",
            "sheets" | "sheet" => "sheet",
            _ => continue,
        };
        return Some((kind.to_string(), token.to_string()));
    }
    None
}

fn feishu_wiki_resolve_obj(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    token: &str,
) -> Result<(String, String), JsonRpcErr> {
    let q_token = url_encode_component(token);
    let url = format!(
        "{}/open-apis/wiki/v2/spaces/get_node?token={}",
        base_url.trim_end_matches('/'),
        q_token
    );

    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call wiki get_node API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read wiki get_node response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "wiki get_node API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse wiki get_node JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "wiki get_node returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let node = data.get("node").cloned().unwrap_or_else(|| data.clone());

    let obj_type = node
        .get("obj_type")
        .and_then(|x| x.as_str())
        .or_else(|| node.get("objType").and_then(|x| x.as_str()))
        .unwrap_or("")
        .trim()
        .to_string();
    let obj_token = node
        .get("obj_token")
        .and_then(|x| x.as_str())
        .or_else(|| node.get("objToken").and_then(|x| x.as_str()))
        .unwrap_or("")
        .trim()
        .to_string();

    if obj_type.is_empty() || obj_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "wiki get_node response missing obj_type/obj_token",
            Some(json!({ "parsed": { "obj_type": obj_type, "obj_token": obj_token }, "body": v })),
        ));
    }

    Ok((obj_type, obj_token))
}

fn feishu_fetch_sheet_preview_text(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
    max_rows: usize,
    max_cols: usize,
    max_sheets: usize,
) -> Result<String, JsonRpcErr> {
    let sheets = feishu_query_sheet_list(client, base_url, user_access_token, spreadsheet_token)?;
    if sheets.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "No sheets found in spreadsheet",
            Some(json!({ "spreadsheet_token": spreadsheet_token })),
        ));
    }

    let mut out = String::new();
    for (idx, (sheet_id, title)) in sheets.into_iter().take(max_sheets).enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!("[sheet] {} ({})\n", title.trim(), sheet_id.trim()));

        let col = col_letters(max_cols.saturating_sub(1));
        let range = format!("{}!A1:{}{}", sheet_id.trim(), col, max_rows);
        let values = feishu_sheets_read_range_values(
            client,
            base_url,
            user_access_token,
            spreadsheet_token,
            &range,
        )?;
        out.push_str(&format_values_as_tsv(&values));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }

    Ok(out.trim_end().to_string())
}

fn feishu_query_sheet_list(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
) -> Result<Vec<(String, String)>, JsonRpcErr> {
    let url = format!(
        "{}/open-apis/sheets/v3/spreadsheets/{}/sheets/query",
        base_url.trim_end_matches('/'),
        spreadsheet_token.trim()
    );
    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to query sheets list",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read sheets list response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "Sheets list API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse sheets list JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "Sheets list returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let arr = data
        .get("sheets")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for item in arr {
        let sheet_id = item
            .get("sheet_id")
            .and_then(|x| x.as_str())
            .or_else(|| item.get("sheetId").and_then(|x| x.as_str()))
            .unwrap_or("")
            .trim()
            .to_string();
        let title = item
            .get("title")
            .and_then(|x| x.as_str())
            .or_else(|| item.get("name").and_then(|x| x.as_str()))
            .unwrap_or("")
            .trim()
            .to_string();
        if !sheet_id.is_empty() {
            out.push((sheet_id, title));
        }
    }
    Ok(out)
}

fn feishu_sheets_read_range_values(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    spreadsheet_token: &str,
    range: &str,
) -> Result<Value, JsonRpcErr> {
    let encoded_range = url_encode_component(range);
    let url = format!(
        "{}/open-apis/sheets/v2/spreadsheets/{}/values/{}?valueRenderOption=ToString&dateTimeRenderOption=FormattedString",
        base_url.trim_end_matches('/'),
        spreadsheet_token.trim(),
        encoded_range
    );

    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header("Content-Type", "application/json; charset=utf-8")
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to read spreadsheet range",
                Some(json!({ "error": e.to_string(), "range": range })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read spreadsheet range response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "Spreadsheet range API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse spreadsheet range JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
    if code != 0 {
        let msg = v
            .get("msg")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown error");
        return Err(json_rpc_error(
            -32000,
            "Spreadsheet range returned error",
            Some(json!({ "code": code, "msg": msg, "body": v })),
        ));
    }

    let values = v
        .get("data")
        .and_then(|d| d.get("valueRange"))
        .and_then(|vr| vr.get("values"))
        .cloned()
        .unwrap_or_else(|| json!([]));
    Ok(values)
}

fn format_values_as_tsv(values: &Value) -> String {
    let rows = values.as_array().cloned().unwrap_or_default();
    let mut out = String::new();
    for row in rows {
        let cells = row.as_array().cloned().unwrap_or_default();
        let mut first = true;
        for cell in cells {
            if !first {
                out.push('\t');
            }
            first = false;
            out.push_str(&cell_to_text(&cell));
        }
        out.push('\n');
    }
    out
}

fn cell_to_text(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(n) = v.as_i64() {
        return n.to_string();
    }
    if let Some(n) = v.as_f64() {
        return n.to_string();
    }
    if let Some(obj) = v.as_object() {
        if let Some(t) = obj.get("text").and_then(|x| x.as_str()) {
            return t.to_string();
        }
        if obj.get("fileToken").and_then(|x| x.as_str()).is_some()
            || obj
                .get("float_image_token")
                .and_then(|x| x.as_str())
                .is_some()
            || obj
                .get("floatImageToken")
                .and_then(|x| x.as_str())
                .is_some()
        {
            return "[image]".to_string();
        }
        if let Some(t) = obj.get("type").and_then(|x| x.as_str())
            && t == "embed-image"
        {
            return "[image]".to_string();
        }
        if let Some(s) = obj.get("value").and_then(|x| x.as_str()) {
            return s.to_string();
        }
    }
    if v.is_null() {
        return String::new();
    }
    v.to_string()
}

fn col_letters(mut idx: usize) -> String {
    let mut out = Vec::new();
    loop {
        let rem = idx % 26;
        out.push((b'A' + rem as u8) as char);
        if idx < 26 {
            break;
        }
        idx = idx / 26 - 1;
    }
    out.iter().rev().collect()
}

fn feishu_fetch_raw_content(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    docs_type: &str,
    docs_token: &str,
    lang: i64,
    default_origin: Option<&str>,
) -> Result<String, JsonRpcErr> {
    let content = feishu_fetch_raw_content_api(
        client,
        base_url,
        user_access_token,
        docs_type,
        docs_token,
        lang,
    )?;

    if docs_type != "docx" {
        return Ok(content);
    }

    let blocks_text = feishu_fetch_docx_blocks_text(
        client,
        base_url,
        user_access_token,
        docs_token,
        default_origin,
    )?;
    if should_prefer_docx_blocks_render(&content, &blocks_text) {
        Ok(blocks_text)
    } else {
        Ok(content)
    }
}

fn feishu_fetch_raw_content_api(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    docs_type: &str,
    docs_token: &str,
    lang: i64,
) -> Result<String, JsonRpcErr> {
    let (url, is_docx) = match docs_type {
        "docx" => (
            format!(
                "{}/open-apis/docx/v1/documents/{}/raw_content?lang={}",
                base_url.trim_end_matches('/'),
                docs_token,
                lang
            ),
            true,
        ),
        "doc" => (
            format!(
                "{}/open-apis/doc/v2/{}/raw_content",
                base_url.trim_end_matches('/'),
                docs_token
            ),
            false,
        ),
        _ => {
            return Err(json_rpc_error(
                -32602,
                "Unsupported docs_type (only doc/docx supported for now)",
                Some(json!({ "docs_type": docs_type })),
            ));
        }
    };

    let resp = client
        .get(url)
        .header(
            "Authorization",
            format!("Bearer {}", user_access_token.trim()),
        )
        .header(
            "Content-Type",
            if is_docx {
                "application/json; charset=utf-8"
            } else {
                "text/plain"
            },
        )
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to fetch raw_content",
                Some(json!({ "error": e.to_string() })),
            )
        })?;
    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read raw_content response",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "raw_content API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "raw_content response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "raw_content API returned error code",
            Some(v),
        ));
    }
    let content = v
        .get("data")
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Ok(content)
}

fn feishu_fetch_docx_blocks_text(
    client: &Client,
    base_url: &str,
    user_access_token: &str,
    document_id: &str,
    default_origin: Option<&str>,
) -> Result<String, JsonRpcErr> {
    let mut page_token: Option<String> = None;
    let mut items = Vec::new();

    loop {
        let mut url = format!(
            "{}/open-apis/docx/v1/documents/{}/blocks?page_size=500",
            base_url.trim_end_matches('/'),
            document_id.trim()
        );
        if let Some(token) = page_token.as_deref()
            && !token.trim().is_empty()
        {
            url.push_str("&page_token=");
            url.push_str(&url_encode_component(token.trim()));
        }

        let resp = client
            .get(&url)
            .header(
                "Authorization",
                format!("Bearer {}", user_access_token.trim()),
            )
            .header("Content-Type", "application/json; charset=utf-8")
            .send()
            .map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to fetch docx blocks",
                    Some(json!({ "error": e.to_string(), "document_id": document_id })),
                )
            })?;
        let status = resp.status();
        let text = resp.text().map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to read docx blocks response",
                Some(json!({ "error": e.to_string(), "document_id": document_id })),
            )
        })?;
        if !status.is_success() {
            return Err(json_rpc_error(
                -32000,
                "docx blocks API returned non-success HTTP status",
                Some(
                    json!({ "status": status.as_u16(), "body": text, "document_id": document_id }),
                ),
            ));
        }

        let v: Value = serde_json::from_str(&text).map_err(|e| {
            json_rpc_error(
                -32000,
                "docx blocks response is not valid JSON",
                Some(json!({ "error": e.to_string(), "body": text, "document_id": document_id })),
            )
        })?;
        let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code != 0 {
            return Err(json_rpc_error(
                -32000,
                "docx blocks API returned error code",
                Some(v),
            ));
        }

        let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
        if let Some(arr) = data.get("items").and_then(|v| v.as_array()) {
            items.extend(arr.iter().cloned());
        }

        let has_more = data
            .get("has_more")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let next_token = data
            .get("page_token")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if !has_more || next_token.is_none() {
            break;
        }
        page_token = next_token;
    }

    Ok(render_docx_blocks_as_text(&items, default_origin))
}

fn should_prefer_docx_blocks_render(raw_content: &str, blocks_text: &str) -> bool {
    let raw = raw_content.trim();
    let rendered = blocks_text.trim();
    if rendered.is_empty() {
        return false;
    }
    if raw.is_empty() {
        return true;
    }
    if rendered == raw {
        return false;
    }

    docx_blocks_text_has_non_text_placeholders(rendered)
}

fn docx_blocks_text_has_non_text_placeholders(s: &str) -> bool {
    s.contains("[流程图]")
        || s.contains("[UML 图]")
        || s.contains("[文字绘图")
        || s.contains("[图片")
        || s.contains("[文件")
        || s.contains("[思维笔记")
        || s.contains("[电子表格")
        || s.contains("[多维表格")
        || s.contains("[嵌入内容")
        || s.contains("[会话卡片")
        || s.contains("[小组件")
}

fn render_docx_blocks_as_text(items: &[Value], default_origin: Option<&str>) -> String {
    if items.is_empty() {
        return String::new();
    }

    let mut by_id: FastMap<String, &Value> = FastMap::default();
    let mut root_id = None::<String>;
    for item in items {
        if let Some(block_id) = item.get("block_id").and_then(|v| v.as_str()) {
            if item
                .get("parent_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .is_empty()
            {
                root_id = Some(block_id.to_string());
            }
            by_id.insert(block_id.to_string(), item);
        }
    }

    let Some(root_id) = root_id.or_else(|| {
        items
            .first()
            .and_then(|v| v.get("block_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }) else {
        return String::new();
    };

    let mut out = String::new();
    render_docx_block_text(&root_id, &by_id, default_origin, &mut out);
    normalize_rendered_docx_text(&out)
}

fn render_docx_block_text(
    block_id: &str,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
    out: &mut String,
) {
    let Some(block) = by_id.get(block_id).copied() else {
        return;
    };

    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let line = render_docx_block_line(block, default_origin);
    if !line.is_empty() {
        out.push_str(&line);
        out.push('\n');
    }

    if block_type == 31 {
        render_docx_table_cells(block, by_id, default_origin, out);
        return;
    }

    let children = block
        .get("children")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for child in children {
        if let Some(child_id) = child.as_str() {
            render_docx_block_text(child_id, by_id, default_origin, out);
        }
    }
}

fn render_docx_table_cells(
    block: &Value,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
    out: &mut String,
) {
    let cells = block
        .get("table")
        .and_then(|v| v.get("cells"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let col_size = block
        .get("table")
        .and_then(|v| v.get("property"))
        .and_then(|v| v.get("column_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if cells.is_empty() || col_size == 0 {
        return;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    for cell in cells {
        let Some(cell_id) = cell.as_str() else {
            continue;
        };
        let cell_text = render_docx_table_cell_text(cell_id, by_id, default_origin);
        let escaped = cell_text
            .replace('|', "\\|")
            .replace('\n', "<br>");
        row.push(escaped);
        if row.len() >= col_size {
            rows.push(row);
            row = Vec::new();
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }

    for (i, row) in rows.iter().enumerate() {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
        if i == 0 {
            out.push_str("|");
            for _ in 0..col_size {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }
}

fn render_docx_table_cell_text(
    cell_id: &str,
    by_id: &FastMap<String, &Value>,
    default_origin: Option<&str>,
) -> String {
    let Some(block) = by_id.get(cell_id).copied() else {
        return String::new();
    };

    let direct = render_text_elements(
        block
            .get("table_cell")
            .and_then(|v| v.get("elements"))
            .or_else(|| block.get("text").and_then(|v| v.get("elements"))),
        default_origin,
    );
    if !direct.is_empty() {
        return direct;
    }

    let mut parts = Vec::new();
    let children = block
        .get("children")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for child in children {
        let Some(child_id) = child.as_str() else {
            continue;
        };
        let Some(child_block) = by_id.get(child_id).copied() else {
            continue;
        };
        let line = render_docx_block_line(child_block, default_origin);
        if !line.is_empty() {
            parts.push(line);
        }
    }
    parts.join(" ")
}

fn render_docx_block_line(block: &Value, default_origin: Option<&str>) -> String {
    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    match block_type {
        1 => render_text_elements(
            block.get("page").and_then(|v| v.get("elements")),
            default_origin,
        ),
        2 => render_text_elements(
            block.get("text").and_then(|v| v.get("elements")),
            default_origin,
        ),
        3..=11 => {
            let level = (block_type - 2) as usize;
            let text = render_text_elements(
                block
                    .get(format!("heading{}", level).as_str())
                    .and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("{} {}", "#".repeat(level), text)
            }
        }
        12 => {
            let text = render_text_elements(
                block.get("bullet").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("- {}", text)
            }
        }
        13 => {
            let text = render_text_elements(
                block.get("ordered").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("1. {}", text)
            }
        }
        14 => {
            let text = render_text_elements(
                block.get("code").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("```text\n{}\n```", text)
            }
        }
        15 => {
            let text = render_text_elements(
                block.get("quote").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("> {}", text)
            }
        }
        17 => {
            let todo = block.get("todo");
            let text = render_text_elements(todo.and_then(|v| v.get("elements")), default_origin);
            let checked = todo
                .and_then(|v| v.get("style"))
                .and_then(|v| v.get("done"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if text.is_empty() {
                String::new()
            } else {
                format!("- [{}] {}", if checked { "x" } else { " " }, text)
            }
        }
        18 => render_token_placeholder(block.get("bitable"), "token", "多维表格"),
        19 => {
            let text = render_text_elements(
                block.get("callout").and_then(|v| v.get("elements")),
                default_origin,
            );
            if text.is_empty() {
                "[高亮块]".to_string()
            } else {
                format!("[高亮块] {}", text)
            }
        }
        20 => "[会话卡片]".to_string(),
        21 => render_diagram_placeholder(block),
        22 => "---".to_string(),
        23 => render_named_placeholder(block.get("file"), "name", "文件"),
        24 | 25 | 26 | 32 | 33 | 34 | 35 | 36 | 37 => String::new(),
        27 => render_token_placeholder(block.get("image"), "token", "图片"),
        28 => "[小组件]".to_string(),
        29 => render_token_placeholder(block.get("mindnote"), "token", "思维笔记"),
        30 => render_token_placeholder(block.get("sheet"), "token", "电子表格"),
        31 => String::new(),
        43 => render_token_placeholder(block.get("board"), "token", "文字绘图"),
        _ => String::new(),
    }
}

fn render_diagram_placeholder(block: &Value) -> String {
    let diagram_type = block
        .get("diagram")
        .and_then(|v| v.get("diagram_type"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    match diagram_type {
        1 => "[流程图]".to_string(),
        2 => "[UML 图]".to_string(),
        _ => "[文字绘图]".to_string(),
    }
}

fn render_named_placeholder(container: Option<&Value>, field: &str, label: &str) -> String {
    let name = container
        .and_then(|v| v.get(field))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if name.is_empty() {
        format!("[{}]", label)
    } else {
        format!("[{}: {}]", label, name)
    }
}

fn render_token_placeholder(container: Option<&Value>, field: &str, label: &str) -> String {
    let token = container
        .and_then(|v| v.get(field))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if token.is_empty() {
        format!("[{}]", label)
    } else {
        format!("[{}: {}]", label, token)
    }
}

fn extract_url_origin(url: &str) -> Option<String> {
    let raw = url.trim();
    if raw.is_empty() {
        return None;
    }
    let raw = raw.split('#').next().unwrap_or(raw);
    let raw = raw.split('?').next().unwrap_or(raw);
    let (scheme, rest) = if let Some((s, r)) = raw.split_once("://") {
        (s.trim(), r)
    } else {
        ("https", raw)
    };
    let host = rest.split('/').next().unwrap_or("").trim();
    if host.is_empty() {
        return None;
    }
    Some(format!("{}://{}", scheme, host))
}

fn render_doc_mention_as_markdown(mention: &Value, default_origin: Option<&str>) -> Option<String> {
    let title = mention
        .get("title")
        .or_else(|| mention.get("name"))
        .or_else(|| mention.get("text"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let url = mention
        .get("url")
        .and_then(|v| v.as_str())
        .or_else(|| mention.get("href").and_then(|v| v.as_str()))
        .or_else(|| {
            mention
                .get("link")
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let token = mention
        .get("token")
        .or_else(|| mention.get("obj_token"))
        .or_else(|| mention.get("docs_token"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let obj_type = mention
        .get("obj_type")
        .or_else(|| mention.get("docs_type"))
        .or_else(|| mention.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();

    let url = url.or_else(|| {
        if token.is_empty() {
            return None;
        }
        let kind = match obj_type.as_str() {
            "docx" => "docx",
            "doc" | "docs" => "doc",
            "sheet" | "sheets" => "sheets",
            "wiki" => "wiki",
            _ => "docx",
        };
        let origin = default_origin.unwrap_or("https://www.feishu.cn").trim();
        Some(format!(
            "{}/{}/{}",
            origin.trim_end_matches('/'),
            kind,
            token
        ))
    });

    let Some(url) = url else {
        if title.is_empty() {
            return None;
        }
        return Some(title);
    };

    if title.is_empty() {
        Some(url)
    } else {
        Some(format!("[{}]({})", title, url))
    }
}

fn render_text_elements(elements: Option<&Value>, default_origin: Option<&str>) -> String {
    let Some(arr) = elements.and_then(|v| v.as_array()) else {
        return String::new();
    };

    let mut out = String::new();
    for el in arr {
        // 处理 text_run，包括链接
        if let Some(text_run) = el.get("text_run") {
            let content = text_run
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // 检查是否有链接
            let link_url = text_run
                .get("text_style")
                .and_then(|v| v.get("link"))
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str());

            if let Some(url) = link_url {
                // 如果有链接，格式化为 [文本](URL) 的 Markdown 格式
                if !content.is_empty() {
                    out.push('[');
                    out.push_str(content);
                    out.push_str("](");
                    out.push_str(url);
                    out.push(')');
                } else {
                    // 如果内容为空但 URL 存在，直接显示 URL
                    out.push_str(url);
                }
            } else {
                // 没有链接，直接显示文本
                out.push_str(content);
            }
            continue;
        }

        if let Some(v) = el
            .get("equation")
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
        {
            out.push_str(v);
            continue;
        }
        if let Some(v) = el
            .get("mention_user")
            .and_then(|v| {
                v.get("user_name")
                    .or_else(|| v.get("name"))
                    .or_else(|| v.get("title"))
            })
            .and_then(|v| v.as_str())
        {
            out.push('@');
            out.push_str(v);
            continue;
        }
        if let Some(v) = el
            .get("mention_doc")
            .and_then(|v| render_doc_mention_as_markdown(v, default_origin))
        {
            out.push_str(&v);
            continue;
        }
        if let Some(v) = el
            .get("reminder")
            .and_then(|v| v.get("notify_time"))
            .and_then(|v| v.as_str())
        {
            out.push_str("[提醒:");
            out.push_str(v);
            out.push(']');
            continue;
        }
        if let Some(v) = el
            .get("file")
            .and_then(|v| {
                v.get("name")
                    .or_else(|| v.get("file_token"))
                    .or_else(|| v.get("token"))
            })
            .and_then(|v| v.as_str())
        {
            out.push_str("[附件:");
            out.push_str(v);
            out.push(']');
            continue;
        }
        if let Some(v) = el
            .get("inline_block")
            .and_then(|v| {
                v.get("block_id")
                    .or_else(|| v.get("token"))
                    .or_else(|| v.get("url"))
            })
            .and_then(|v| v.as_str())
        {
            out.push_str("[内联块:");
            out.push_str(v);
            out.push(']');
        }
    }
    out.trim().to_string()
}

fn normalize_rendered_docx_text(s: &str) -> String {
    let mut out = String::new();
    let mut blank_run = 0usize;
    for line in s.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
            continue;
        }
        blank_run = 0;
        out.push_str(trimmed);
        out.push('\n');
    }
    out.trim().to_string()
}

fn do_docs_search_request(
    client: &Client,
    url: &str,
    token: &str,
    body: &Value,
) -> Result<(reqwest::StatusCode, String), JsonRpcErr> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call Feishu API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    Ok((status, text))
}

fn resolve_base_url() -> String {
    if let Ok(v) = std::env::var("FEISHU_BASE_URL") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.base_url") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    "https://open.feishu.cn".to_string()
}

fn resolve_accounts_base_url(base_url: &str) -> String {
    if let Ok(v) = std::env::var("FEISHU_ACCOUNTS_BASE_URL") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.accounts_base_url") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    if base_url.contains("larksuite") {
        "https://accounts.larksuite.com".to_string()
    } else {
        "https://accounts.feishu.cn".to_string()
    }
}

fn resolve_user_access_token() -> Option<String> {
    if let Ok(v) = std::env::var("FEISHU_USER_ACCESS_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() && v.starts_with("u-") {
            return Some(v);
        }
    }
    if let Ok(v) = std::env::var("FEISHU_ACCESS_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() && v.starts_with("u-") {
            return Some(v);
        }
    }
    let cfg = configw::get_all_config();
    cfg.get_opt("feishu.user_access_token")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && v.starts_with("u-"))
}

fn resolve_refresh_token() -> Option<String> {
    if let Ok(v) = std::env::var("FEISHU_REFRESH_TOKEN") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    let cfg = configw::get_all_config();
    let v = cfg
        .get_opt("feishu.refresh_token")
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if v.is_some() {
        return v;
    }
    load_token_store().ok().and_then(|s| s.refresh_token)
}

fn resolve_stored_user_access_token() -> Option<(String, Option<i64>)> {
    let store = load_token_store().ok()?;
    let token = store.user_access_token?.trim().to_string();
    if token.is_empty() || !token.starts_with("u-") {
        return None;
    }
    let expires_at = store.user_access_token_expires_at_epoch_ms;
    if let Some(epoch_ms) = expires_at
        && epoch_ms > 0
        && epoch_ms <= epoch_ms_now() + 300_000
    {
        return None;
    }
    Some((token, expires_at))
}

fn resolve_client_credentials() -> Option<(String, String)> {
    let env_id = std::env::var("FEISHU_CLIENT_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .or_else(|| std::env::var("FEISHU_APP_ID").ok().map(|v| v.trim().to_string()));
    let env_secret = std::env::var("FEISHU_CLIENT_SECRET")
        .ok()
        .map(|v| v.trim().to_string())
        .or_else(|| std::env::var("FEISHU_APP_SECRET").ok().map(|v| v.trim().to_string()));
    if let (Some(id), Some(secret)) = (env_id, env_secret)
        && !id.is_empty()
        && !secret.is_empty()
    {
        return Some((id, secret));
    }

    let cfg = configw::get_all_config();
    let id = cfg
        .get_opt("feishu.client_id")
        .map(|v| v.trim().to_string())
        .or_else(|| cfg.get_opt("feishu.app_id").map(|v| v.trim().to_string()));
    let secret = cfg
        .get_opt("feishu.client_secret")
        .map(|v| v.trim().to_string())
        .or_else(|| cfg.get_opt("feishu.app_secret").map(|v| v.trim().to_string()));
    match (id, secret) {
        (Some(id), Some(secret)) if !id.is_empty() && !secret.is_empty() => Some((id, secret)),
        _ => None,
    }
}

fn get_user_access_token_cached(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    let now = Instant::now();
    if let Ok(guard) = cache.lock()
        && let Some(cached) = guard.as_ref()
        && cached.expires_at > now + Duration::from_secs(300)
    {
        return Ok(cached.token.clone());
    }
    refresh_user_access_token_and_cache(client, base_url)
}

fn cache_user_access_token(token: &str, expires_at_epoch_ms: Option<i64>) {
    if token.trim().is_empty() {
        return;
    }
    let expires_at = match expires_at_epoch_ms {
        Some(ms) if ms > epoch_ms_now() => {
            Instant::now() + Duration::from_millis(ms.saturating_sub(epoch_ms_now()) as u64)
        }
        _ => Instant::now() + Duration::from_secs(600),
    };
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken {
            token: token.trim().to_string(),
            expires_at,
        });
    }
}

fn acquire_user_access_token(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    if let Some(tok) = resolve_user_access_token() {
        return Ok(tok);
    }
    if let Some((tok, expires_at_epoch_ms)) = resolve_stored_user_access_token() {
        cache_user_access_token(&tok, expires_at_epoch_ms);
        return Ok(tok);
    }
    get_user_access_token_cached(client, base_url)
}

fn with_user_access_token<T, F>(
    client: &Client,
    base_url: &str,
    missing_message: &str,
    mut op: F,
) -> Result<T, JsonRpcErr>
where
    F: FnMut(&str) -> Result<T, JsonRpcErr>,
{
    let primary = acquire_user_access_token(client, base_url).map_err(|e| {
        json_rpc_error(
            -32000,
            missing_message,
            Some(json!({
                "detail": e.message,
                "next_steps": ["oauth_authorize_url", "oauth_wait_local_code", "oauth_exchange_code"],
                "token_store": token_store_path().display().to_string()
            })),
        )
    })?;

    let mut err = match op(&primary) {
        Ok(v) => return Ok(v),
        Err(err) => err,
    };
    if !is_invalid_user_access_token_error(&err) {
        return Err(err);
    }

    if let Some((stored, expires_at_epoch_ms)) = resolve_stored_user_access_token()
        && stored != primary
    {
        cache_user_access_token(&stored, expires_at_epoch_ms);
        match op(&stored) {
            Ok(v) => return Ok(v),
            Err(next_err) => {
                if !is_invalid_user_access_token_error(&next_err) {
                    return Err(next_err);
                }
                err = next_err;
            }
        }
    }

    if let Ok(refreshed) = refresh_user_access_token_and_cache(client, base_url)
        && refreshed != primary
    {
        match op(&refreshed) {
            Ok(v) => return Ok(v),
            Err(next_err) => err = next_err,
        }
    }

    if is_invalid_user_access_token_error(&err) {
        Err(invalid_user_access_token_error(err))
    } else {
        Err(err)
    }
}

fn is_invalid_user_access_token_error(err: &JsonRpcErr) -> bool {
    if err.message.contains("Invalid access token") {
        return true;
    }
    extract_feishu_error_code(err.data.as_ref()) == Some(99991668)
}

fn extract_feishu_error_code(data: Option<&Value>) -> Option<i64> {
    let data = data?;
    if let Some(code) = data.get("feishu_code").and_then(|v| v.as_i64()) {
        return Some(code);
    }
    if let Some(code) = data.get("code").and_then(|v| v.as_i64()) {
        return Some(code);
    }
    let body = data.get("body").and_then(|v| v.as_str())?;
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("code").and_then(|x| x.as_i64()))
}

fn extract_feishu_error_message(data: Option<&Value>) -> Option<String> {
    let data = data?;
    if let Some(msg) = data.get("msg").and_then(|v| v.as_str()) {
        return Some(msg.trim().to_string());
    }
    let body = data.get("body").and_then(|v| v.as_str())?;
    serde_json::from_str::<Value>(body).ok().and_then(|v| {
        v.get("msg")
            .and_then(|x| x.as_str())
            .map(|s| s.trim().to_string())
    })
}

fn invalid_user_access_token_error(err: JsonRpcErr) -> JsonRpcErr {
    let status = err
        .data
        .as_ref()
        .and_then(|v| v.get("status"))
        .and_then(|v| v.as_u64());
    let feishu_code = extract_feishu_error_code(err.data.as_ref());
    let msg = extract_feishu_error_message(err.data.as_ref()).unwrap_or(err.message);
    json_rpc_error(
        -32000,
        "Invalid access token. Provide a valid user_access_token or set refresh_token for automatic refresh.",
        Some(json!({
            "status": status,
            "feishu_code": feishu_code,
            "msg": msg,
            "token_store": token_store_path().display().to_string(),
            "next_steps": [
                "If you have refresh_token: call oauth_refresh_user_access_token, then update FEISHU_USER_ACCESS_TOKEN (or set FEISHU_REFRESH_TOKEN for auto refresh)",
                "Otherwise: run oauth_authorize_url -> oauth_wait_local_code -> oauth_exchange_code"
            ]
        })),
    )
}

fn refresh_user_access_token_and_cache(
    client: &Client,
    base_url: &str,
) -> Result<String, JsonRpcErr> {
    let refresh_token = resolve_refresh_token().ok_or_else(|| {
        json_rpc_error(
            -32000,
            "Missing refresh_token for refreshing user_access_token",
            Some(json!({
                "env": ["FEISHU_REFRESH_TOKEN"],
                "config_keys": ["feishu.refresh_token"],
                "token_store": token_store_path().display().to_string()
            })),
        )
    })?;
    let refreshed = refresh_user_access_token_api(client, base_url, &refresh_token)?;
    let cache = USER_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken {
            token: refreshed.user_access_token.clone(),
            expires_at: refreshed.expires_at,
        });
    }
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(refreshed.user_access_token.clone()),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(refreshed.expires_at)),
        refresh_token: Some(refreshed.refresh_token.clone()),
        refresh_token_expires_in: Some(refreshed.refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });
    Ok(refreshed.user_access_token)
}

struct RefreshedUserToken {
    user_access_token: String,
    refresh_token: String,
    expires_at: Instant,
    refresh_expires_in: i64,
}

fn refresh_user_access_token_api(
    client: &Client,
    base_url: &str,
    refresh_token: &str,
) -> Result<RefreshedUserToken, JsonRpcErr> {
    let Some((client_id, client_secret)) = resolve_client_credentials() else {
        return Err(json_rpc_error(
            -32000,
            "Missing Feishu client credentials (client_id/client_secret)",
            Some(json!({
                "env": [
                    "FEISHU_CLIENT_ID",
                    "FEISHU_CLIENT_SECRET",
                    "FEISHU_APP_ID",
                    "FEISHU_APP_SECRET"
                ],
                "config_keys": [
                    "feishu.client_id",
                    "feishu.client_secret",
                    "feishu.app_id",
                    "feishu.app_secret"
                ]
            })),
        ));
    };
    let url = format!(
        "{}/open-apis/authen/v2/oauth/token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    });
    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call refresh_access_token API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read refresh_access_token response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "refresh_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "refresh_access_token response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(
            -32000,
            "refresh_access_token API returned error code",
            Some(v),
        ));
    }
    let user_access_token = v
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let next_refresh_token = v
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = v
        .get("expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(60) as u64;
    let refresh_expires_in = v
        .get("refresh_token_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if user_access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in refresh response",
            Some(v),
        ));
    }
    Ok(RefreshedUserToken {
        user_access_token,
        refresh_token: next_refresh_token,
        expires_at: Instant::now() + Duration::from_secs(expires_in),
        refresh_expires_in,
    })
}

fn feishu_oauth_authorize_url(args: &Value) -> Result<String, JsonRpcErr> {
    let cfg = configw::get_all_config();
    let client_id = cfg
        .get_opt("feishu.client_id")
        .or_else(|| cfg.get_opt("feishu.app_id"))
        .or_else(|| std::env::var("FEISHU_CLIENT_ID").ok())
        .or_else(|| std::env::var("FEISHU_APP_ID").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            json_rpc_error(
                -32000,
                "Missing feishu.client_id / FEISHU_CLIENT_ID",
                Some(json!({
                    "legacy_env": ["FEISHU_APP_ID"],
                    "legacy_config_keys": ["feishu.app_id"]
                })),
            )
        })?;

    let base_url = resolve_base_url();
    let accounts_base = resolve_accounts_base_url(&base_url);

    let redirect_uri = args
        .get("redirect_uri")
        .and_then(|v| v.as_str())
        .unwrap_or("http://127.0.0.1:8711/callback")
        .trim()
        .to_string();
    if redirect_uri.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: redirect_uri is empty",
            None,
        ));
    }

    let scope = args
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("offline_access im:chat:readonly im:message:readonly docs:doc:readonly docx:document:readonly wiki:wiki:readonly sheets:spreadsheet")
        .trim()
        .to_string();
    let state = args
        .get("state")
        .and_then(|v| v.as_str())
        .unwrap_or("rust-tools-ai");
    let prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    let encoded_redirect = url_encode_component(&redirect_uri);
    let encoded_scope = url_encode_component(&scope);
    let encoded_state = url_encode_component(state);

    let mut url = format!(
        "{}/open-apis/authen/v1/authorize?client_id={}&response_type=code&redirect_uri={}&scope={}&state={}",
        accounts_base.trim_end_matches('/'),
        url_encode_component(&client_id),
        encoded_redirect,
        encoded_scope,
        encoded_state
    );
    if !prompt.trim().is_empty() {
        url.push_str("&prompt=");
        url.push_str(&url_encode_component(prompt.trim()));
    }
    Ok(url)
}

fn feishu_oauth_wait_local_code(args: &Value) -> Result<String, JsonRpcErr> {
    let port = args
        .get("port")
        .and_then(|v| v.as_i64())
        .unwrap_or(8711)
        .clamp(1, 65535) as u16;
    let timeout_sec = args
        .get("timeout_sec")
        .and_then(|v| v.as_i64())
        .unwrap_or(180)
        .clamp(1, 600) as u64;
    let mut listeners: Vec<TcpListener> = Vec::new();
    let addr4 = format!("127.0.0.1:{port}");
    match TcpListener::bind(&addr4) {
        Ok(l) => {
            l.set_nonblocking(true).ok();
            listeners.push(l);
        }
        Err(e) => {
            let _ = e;
        }
    }
    let addr6 = format!("[::1]:{port}");
    match TcpListener::bind(&addr6) {
        Ok(l) => {
            l.set_nonblocking(true).ok();
            listeners.push(l);
        }
        Err(e) => {
            let _ = e;
        }
    }
    if listeners.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Failed to bind local callback port",
            Some(json!({ "port": port, "addrs": [addr4, addr6] })),
        ));
    }

    let (tx, rx) = mpsc::channel::<String>();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let worker_stop_flag = Arc::clone(&stop_flag);
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout_sec);
        loop {
            if worker_stop_flag.load(Ordering::Relaxed) {
                break;
            }
            if Instant::now() >= deadline {
                break;
            }
            let mut accepted: Option<TcpStream> = None;
            for listener in &listeners {
                if let Ok((stream, _)) = listener.accept() {
                    accepted = Some(stream);
                    break;
                }
            }

            let Some(mut stream) = accepted else {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            };

            let req = read_http_request(&mut stream);
            let code = parse_oauth_code_from_http_request(&req).unwrap_or_default();
            if !code.is_empty() {
                let body = "<html><body>OK. You can close this tab.</body></html>";
                let _ = write_http_response(&mut stream, body);
                let _ = tx.send(code);
                return;
            }

            let body = r#"<html><head><meta charset="utf-8"></head><body>
<div>Waiting for OAuth code...</div>
<script>
  (function () {
    try {
      var url = new URL(window.location.href);
      var code = url.searchParams.get('code');
      if (!code && window.location.hash && window.location.hash.length > 1) {
        var hash = window.location.hash.substring(1);
        var params = new URLSearchParams(hash);
        code = params.get('code');
      }
      if (code) {
        url.hash = '';
        url.searchParams.set('code', code);
        window.location.replace(url.toString());
        return;
      }
    } catch (e) {}
    document.body.innerHTML = '<div>Missing code. If you see a "code" in the URL, copy it and paste back into the CLI.</div>';
  })();
</script>
</body></html>"#;
            let _ = write_http_response(&mut stream, body);
        }
    });

    let result = match rx.recv_timeout(Duration::from_secs(timeout_sec)) {
        Ok(code) => Ok(format!("code: {code}\nport: {port}\npath: /callback")),
        Err(_) => Err(json_rpc_error(
            -32000,
            "Timeout waiting for OAuth code",
            Some(json!({ "port": port, "timeout_sec": timeout_sec })),
        )),
    };
    stop_flag.store(true, Ordering::Relaxed);
    let _ = worker.join();
    result
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(8)));
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    let mut buf = [0u8; 1024];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if Instant::now() >= deadline || out.len() >= 16_384 {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                if out.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn write_http_response(stream: &mut TcpStream, body: &str) -> io::Result<()> {
    let bytes = body.as_bytes();
    stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .as_bytes(),
    )?;
    stream.write_all(bytes)?;
    stream.flush()
}

fn parse_oauth_code_from_http_request(req: &str) -> Option<String> {
    let first = req.lines().next()?.trim();
    let mut parts = first.split_whitespace();
    let _method = parts.next()?;
    let target = parts.next().unwrap_or("");
    if let Some(code) = parse_oauth_code_from_urlish(target) {
        return Some(code);
    }
    if let Some(idx) = req.find("\r\n\r\n") {
        let body = &req[idx + 4..];
        if let Some(code) = parse_oauth_code_from_query(body) {
            return Some(code);
        }
    }
    None
}

fn parse_oauth_code_from_urlish(target: &str) -> Option<String> {
    let target = target.trim();
    if target.is_empty() {
        return None;
    }
    let without_fragment = target.split('#').next().unwrap_or(target);
    let qidx = without_fragment.find('?')?;
    let query = &without_fragment[qidx + 1..];
    parse_oauth_code_from_query(query)
}

fn parse_oauth_code_from_query(query: &str) -> Option<String> {
    for part in query.split('&') {
        let mut it = part.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        if k == "code" && !v.trim().is_empty() {
            return url_decode_component(v);
        }
    }
    None
}

fn url_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match *b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char);
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn url_decode_component(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h1 = bytes[i + 1];
                let h2 = bytes[i + 2];
                let v1 = hex_value(h1)?;
                let v2 = hex_value(h2)?;
                out.push((v1 << 4) | v2);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn feishu_oauth_exchange_code(args: &Value) -> Result<String, JsonRpcErr> {
    let code = args
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if code.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: code is empty",
            None,
        ));
    }
    let Some((client_id, client_secret)) = resolve_client_credentials() else {
        return Err(json_rpc_error(
            -32000,
            "Missing Feishu client credentials (client_id/client_secret)",
            Some(json!({
                "env": [
                    "FEISHU_CLIENT_ID",
                    "FEISHU_CLIENT_SECRET",
                    "FEISHU_APP_ID",
                    "FEISHU_APP_SECRET"
                ],
                "config_keys": [
                    "feishu.client_id",
                    "feishu.client_secret",
                    "feishu.app_id",
                    "feishu.app_secret"
                ]
            })),
        ));
    };
    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let url = format!(
        "{}/open-apis/authen/v2/oauth/token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code
    });
    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call user_access_token API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read user_access_token response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "user_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "user_access_token response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code_num = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code_num != 0 {
        return Err(json_rpc_error(
            -32000,
            "user_access_token API returned error code",
            Some(v),
        ));
    }
    let access_token = v
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = v
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = v.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    let refresh_expires_in = v
        .get("refresh_token_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    if access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in response",
            Some(v),
        ));
    }

    let expires_at = Instant::now() + Duration::from_secs(expires_in.max(60) as u64);
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(access_token),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(expires_at)),
        refresh_token: (!refresh_token.is_empty()).then_some(refresh_token),
        refresh_token_expires_in: Some(refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });

    Ok(format!(
        "OAuth exchange success.\n- Stored tokens in: {}\n- expires_in: {}\n- refresh_expires_in: {}\n\nYou can now call docs_search without setting FEISHU_USER_ACCESS_TOKEN.",
        token_store_path().display(),
        expires_in,
        refresh_expires_in
    ))
}

fn feishu_oauth_refresh_user_access_token(args: &Value) -> Result<String, JsonRpcErr> {
    let refresh_token = args
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = if !refresh_token.is_empty() {
        Some(refresh_token)
    } else {
        resolve_refresh_token()
    }
    .ok_or_else(|| json_rpc_error(-32000, "Missing refresh_token", None))?;

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let refreshed = refresh_user_access_token_api(&client, &base_url, &refresh_token)?;
    let _ = save_token_store(&TokenStore {
        user_access_token: Some(refreshed.user_access_token.clone()),
        user_access_token_expires_at_epoch_ms: Some(epoch_ms_from_instant(refreshed.expires_at)),
        refresh_token: (!refreshed.refresh_token.is_empty())
            .then_some(refreshed.refresh_token.clone()),
        refresh_token_expires_in: Some(refreshed.refresh_expires_in),
        updated_at_epoch_ms: Some(epoch_ms_now()),
    });
    Ok(format!(
        "Refresh success.\n- Stored tokens in: {}\n- expires_in: {}\n- refresh_expires_in: {}",
        token_store_path().display(),
        refreshed
            .expires_at
            .saturating_duration_since(Instant::now())
            .as_secs() as i64,
        refreshed.refresh_expires_in
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenStore {
    user_access_token: Option<String>,
    user_access_token_expires_at_epoch_ms: Option<i64>,
    refresh_token: Option<String>,
    refresh_token_expires_in: Option<i64>,
    updated_at_epoch_ms: Option<i64>,
}

fn token_store_path() -> PathBuf {
    if let Ok(v) = std::env::var("FEISHU_TOKEN_STORE_PATH") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let cfg = configw::get_all_config();
    if let Some(v) = cfg.get_opt("feishu.token_store") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return PathBuf::from(rust_tools::commonw::utils::expanduser(&v).as_ref());
        }
    }
    PathBuf::from(
        rust_tools::commonw::utils::expanduser("~/.config/rust_tools/feishu_token.json").as_ref(),
    )
}

fn load_token_store() -> Result<TokenStore, JsonRpcErr> {
    let path = token_store_path();
    let content = fs::read_to_string(&path).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })?;
    serde_json::from_str::<TokenStore>(&content).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })
}

fn save_token_store(store: &TokenStore) -> Result<(), JsonRpcErr> {
    let path = token_store_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create token store directory",
                Some(json!({ "path": parent.display().to_string(), "error": e.to_string() })),
            )
        })?;
    }
    let s = serde_json::to_string_pretty(store).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to serialize token store",
            Some(json!({ "error": e.to_string() })),
        )
    })?;

    fs::write(&path, s).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to write token store",
            Some(json!({ "path": path.display().to_string(), "error": e.to_string() })),
        )
    })?;

    let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    Ok(())
}

fn epoch_ms_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now();
    now.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn epoch_ms_from_instant(instant: Instant) -> i64 {
    let now_instant = Instant::now();
    let now_ms = epoch_ms_now();
    if instant <= now_instant {
        return now_ms;
    }
    let delta = instant.duration_since(now_instant).as_millis() as i64;
    now_ms.saturating_add(delta)
}

fn feishu_sheet_create_from_csv(args: &Value) -> Result<String, JsonRpcErr> {
    let title = args
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let csv_content = args
        .get("csv_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let folder_token = args
        .get("folder_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if title.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: title is required",
            Some(json!({ "title": title })),
        ));
    }
    if csv_content.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: csv_content is required",
            None,
        ));
    }

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    // Step 1: Create spreadsheet
    let spreadsheet_token = with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Create spreadsheet requires OAuth once.",
        |token| {
            let mut create_body = json!({
                "title": title,
                "folder_token": folder_token
            });
            if folder_token.is_empty() {
                create_body.as_object_mut().unwrap().remove("folder_token");
            }

            let url = format!("{}/open-apis/sheets/v3/spreadsheets", base_url);
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token.trim()))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&create_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to create spreadsheet",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

            let _status = resp.status();
            let text = resp.text().map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to read response body",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let json: Value = serde_json::from_str(&text).map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to parse response JSON",
                    Some(json!({ "error": e.to_string(), "body": text })),
                )
            })?;

            let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            if code != 0 {
                let err_json = json.clone();
                return Err(json_rpc_error(
                    -32000,
                    "Failed to create spreadsheet",
                    Some(err_json),
                ));
            }

            let token = json
                .get("data")
                .and_then(|d| d.get("spreadsheet_token"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    let err_json = json.clone();
                    json_rpc_error(-32000, "No spreadsheet_token in response", Some(err_json))
                })?;

            Ok(token.to_string())
        },
    )?;

    // Step 2: Parse CSV and prepare values
    let mut values: Vec<Vec<String>> = Vec::new();
    for line in csv_content.lines() {
        let row: Vec<String> = parse_csv_line(line);
        values.push(row);
    }

    // Step 3: Batch update spreadsheet with values
    with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Update spreadsheet requires OAuth.",
        |token| {
            let range = "Sheet1!A1";
            let update_body = json!({
                "value_range": {
                    "range": range,
                    "values": values
                }
            });

            let url = format!(
                "{}/open-apis/sheets/v2/spreadsheets/{}/values/batchUpdate",
                base_url, spreadsheet_token
            );
            let resp = client
                .put(&url)
                .header("Authorization", format!("Bearer {}", token.trim()))
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&update_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to update spreadsheet values",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

            let _status = resp.status();
            let text = resp.text().map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to read response body",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let json: Value = serde_json::from_str(&text).map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to parse response JSON",
                    Some(json!({ "error": e.to_string(), "body": text })),
                )
            })?;

            let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            if code != 0 {
                return Err(json_rpc_error(
                    -32000,
                    "Failed to update spreadsheet values",
                    Some(json),
                ));
            }

            Ok(())
        },
    )?;

    // Generate URL
    let spreadsheet_url = format!(
        "https://{}.feishu.cn/sheets/{}",
        base_url
            .trim_start_matches("https://")
            .trim_end_matches("/open-apis")
            .split('.')
            .next()
            .unwrap_or("app"),
        spreadsheet_token
    );

    Ok(format!(
        "Created spreadsheet: {}\nToken: {}",
        spreadsheet_url, spreadsheet_token
    ))
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current_cell = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current_cell.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current_cell.push(c);
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => {
                    cells.push(current_cell.trim().to_string());
                    current_cell = String::new();
                }
                _ => current_cell.push(c),
            }
        }
    }
    cells.push(current_cell.trim().to_string());
    cells
}

fn make_text_element(content: &str, bold: bool, italic: bool, inline_code: bool) -> Value {
    let mut style = json!({});
    if bold {
        style["bold"] = json!(true);
    }
    if italic {
        style["italic"] = json!(true);
    }
    if inline_code {
        style["inline_code"] = json!(true);
    }
    json!({
        "text_run": {
            "content": content,
            "text_element_style": style
        }
    })
}

fn parse_inline_elements(text: &str) -> Vec<Value> {
    let mut elements = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    let mut bold_depth = 0usize;
    let mut italic_depth = 0usize;
    let mut in_code = false;

    while let Some(c) = chars.next() {
        if in_code {
            if c == '`' {
                in_code = false;
                if !current.is_empty() {
                    elements.push(make_text_element(&current, false, false, true));
                    current.clear();
                }
            } else {
                current.push(c);
            }
            continue;
        }

        match c {
            '`' => {
                if !current.is_empty() {
                    elements.push(make_text_element(
                        &current,
                        bold_depth > 0,
                        italic_depth > 0,
                        false,
                    ));
                    current.clear();
                }
                in_code = true;
            }
            '*' => {
                let peeked = chars.peek();
                if peeked == Some(&'*') {
                    chars.next();
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if bold_depth > 0 {
                        bold_depth -= 1;
                    } else {
                        bold_depth += 1;
                    }
                } else {
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if italic_depth > 0 {
                        italic_depth -= 1;
                    } else {
                        italic_depth += 1;
                    }
                }
            }
            '_' => {
                let next_is_underscore = chars.peek() == Some(&'_');
                if next_is_underscore {
                    chars.next();
                    if !current.is_empty() {
                        elements.push(make_text_element(
                            &current,
                            bold_depth > 0,
                            italic_depth > 0,
                            false,
                        ));
                        current.clear();
                    }
                    if bold_depth > 0 {
                        bold_depth -= 1;
                    } else {
                        bold_depth += 1;
                    }
                } else {
                    current.push(c);
                }
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        elements.push(make_text_element(
            &current,
            bold_depth > 0,
            italic_depth > 0,
            in_code,
        ));
    }

    if elements.is_empty() {
        elements.push(make_text_element("", false, false, false));
    }

    elements
}

#[derive(Clone)]
enum BlockOp {
    Simple(Value),
    Descendant { children_id: Vec<String>, descendants: Vec<Value> },
}

#[derive(Clone)]
enum MdNode {
    Heading { level: u8, elements: Vec<Value> },
    Paragraph { elements: Vec<Value> },
    BulletList { items: Vec<ListItem> },
    OrderedList { items: Vec<ListItem> },
    CodeBlock { lang: Option<String>, content: String },
    BlockQuote { children: Vec<MdNode> },
    Table { rows: Vec<Vec<String>> },
    Divider,
    Todo { done: bool, elements: Vec<Value> },
}

#[derive(Clone)]
struct ListItem {
    elements: Vec<Value>,
    children: Vec<MdNode>,
}

fn count_indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

fn is_table_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return false;
    }
    inner.split('|').all(|cell| {
        let c = cell.trim();
        c.starts_with('-')
            && c.ends_with('-')
            && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
    })
}

fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = if trimmed.starts_with('|') && trimmed.ends_with('|') {
        &trimmed[1..trimmed.len() - 1]
    } else if trimmed.starts_with('|') {
        &trimmed[1..]
    } else if trimmed.ends_with('|') {
        &trimmed[..trimmed.len() - 1]
    } else {
        trimmed
    };
    inner.split('|').map(|c| c.trim().to_string()).collect()
}

fn strip_ordered_list_prefix(s: &str) -> Option<&str> {
    let mut chars = s.char_indices().peekable();
    let mut found_digit = false;
    while let Some(&(idx, c)) = chars.peek() {
        if c.is_ascii_digit() {
            found_digit = true;
            chars.next();
        } else if found_digit && c == '.' {
            chars.next();
            let rest = &s[idx + 1..];
            return Some(rest.trim_start());
        } else {
            break;
        }
    }
    None
}

fn parse_heading_node(trimmed: &str) -> Option<MdNode> {
    let (level, rest) = if let Some(r) = trimmed.strip_prefix("###### ") {
        (6u8, r)
    } else if let Some(r) = trimmed.strip_prefix("##### ") {
        (5, r)
    } else if let Some(r) = trimmed.strip_prefix("#### ") {
        (4, r)
    } else if let Some(r) = trimmed.strip_prefix("### ") {
        (3, r)
    } else if let Some(r) = trimmed.strip_prefix("## ") {
        (2, r)
    } else if let Some(r) = trimmed.strip_prefix("# ") {
        (1, r)
    } else {
        return None;
    };
    Some(MdNode::Heading {
        level,
        elements: parse_inline_elements(rest),
    })
}

fn detect_list_marker(trimmed: &str) -> Option<(ListKind, &str)> {
    if let Some(rest) = trimmed.strip_prefix("- [x] ") {
        return Some((ListKind::Todo(true), rest));
    }
    if let Some(rest) = trimmed.strip_prefix("- [ ] ") {
        return Some((ListKind::Todo(false), rest));
    }
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return Some((ListKind::Bullet, rest));
    }
    if let Some(rest) = strip_ordered_list_prefix(trimmed) {
        return Some((ListKind::Ordered, rest));
    }
    None
}

#[derive(Clone, Copy, PartialEq)]
enum ListKind {
    Bullet,
    Ordered,
    Todo(bool),
}

fn parse_markdown_ast(markdown: &str) -> Vec<MdNode> {
    let lines: Vec<&str> = markdown.lines().collect();
    let mut ctx = ParseCtx { lines: &lines, pos: 0 };
    let mut nodes = Vec::new();
    while ctx.pos < lines.len() {
        if let Some(node) = parse_next_node(&mut ctx) {
            nodes.push(node);
        }
    }
    nodes
}

struct ParseCtx<'a> {
    lines: &'a [&'a str],
    pos: usize,
}

fn parse_next_node(ctx: &mut ParseCtx) -> Option<MdNode> {
    while ctx.pos < ctx.lines.len() && ctx.lines[ctx.pos].trim().is_empty() {
        ctx.pos += 1;
    }
    if ctx.pos >= ctx.lines.len() {
        return None;
    }

    let line = ctx.lines[ctx.pos];
    let trimmed = line.trim();

    if trimmed.starts_with("```") {
        return Some(parse_code_block_node(ctx));
    }

    if is_table_start_at(ctx.lines, ctx.pos) {
        return Some(parse_table_node(ctx));
    }

    if let Some(node) = parse_heading_node(trimmed) {
        ctx.pos += 1;
        return Some(node);
    }

    if trimmed.starts_with('>') {
        return Some(parse_block_quote_node(ctx));
    }

    if trimmed == "---" || trimmed == "***" || trimmed == "___" {
        ctx.pos += 1;
        return Some(MdNode::Divider);
    }

    if detect_list_marker(trimmed).is_some() {
        return Some(parse_list_node(ctx, 0));
    }

    Some(parse_paragraph_node(ctx))
}

fn parse_code_block_node(ctx: &mut ParseCtx) -> MdNode {
    parse_code_block_node_with_indent(ctx, 0)
}

fn parse_code_block_node_with_indent(ctx: &mut ParseCtx, base_indent: usize) -> MdNode {
    let first = ctx.lines[ctx.pos].trim();
    let lang = first.strip_prefix("```").map(|s| s.trim().to_string());
    let lang = lang.filter(|s| !s.is_empty());
    ctx.pos += 1;

    let mut content = String::new();
    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().starts_with("```") {
            ctx.pos += 1;
            break;
        }
        if !content.is_empty() {
            content.push('\n');
        }
        let code_line = if line.len() > base_indent && line.chars().take(base_indent).all(|c| c == ' ') {
            &line[base_indent..]
        } else {
            line.trim_start()
        };
        content.push_str(code_line);
        ctx.pos += 1;
    }

    MdNode::CodeBlock {
        lang,
        content: content.trim_end().to_string(),
    }
}

fn is_table_start_at(lines: &[&str], pos: usize) -> bool {
    let trimmed = lines[pos].trim();
    if !trimmed.contains('|') {
        return false;
    }
    if pos + 1 >= lines.len() {
        return false;
    }
    is_table_separator_line(lines[pos + 1].trim())
}

fn parse_table_node(ctx: &mut ParseCtx) -> MdNode {
    let mut rows: Vec<Vec<String>> = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let trimmed = ctx.lines[ctx.pos].trim();
        if trimmed.is_empty() {
            break;
        }
        if !trimmed.contains('|') {
            break;
        }
        if is_table_separator_line(trimmed) {
            ctx.pos += 1;
            continue;
        }
        rows.push(parse_table_row(trimmed));
        ctx.pos += 1;
    }

    MdNode::Table { rows }
}

fn parse_block_quote_node(ctx: &mut ParseCtx) -> MdNode {
    let mut inner_lines: Vec<String> = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() || !ctx.lines[j].trim().starts_with('>') {
                break;
            }
            inner_lines.push(String::new());
            ctx.pos += 1;
            continue;
        }
        if !trimmed.starts_with('>') {
            break;
        }
        let content = if trimmed == ">" {
            String::new()
        } else if let Some(rest) = trimmed.strip_prefix("> ") {
            rest.to_string()
        } else {
            trimmed[1..].to_string()
        };
        inner_lines.push(content);
        ctx.pos += 1;
    }

    let inner_text = inner_lines.join("\n");
    let children = parse_markdown_ast(&inner_text);

    MdNode::BlockQuote { children }
}

fn parse_list_node(ctx: &mut ParseCtx, base_indent: usize) -> MdNode {
    let first_trimmed = ctx.lines[ctx.pos].trim_start();
    let (first_kind, first_rest) = detect_list_marker(first_trimmed)
        .expect("parse_list_node called on non-list line");

    let mut items: Vec<ListItem> = Vec::new();
    items.push(ListItem {
        elements: parse_inline_elements(first_rest),
        children: Vec::new(),
    });
    ctx.pos += 1;


    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() {
                break;
            }
            let next_indent = count_indent(ctx.lines[j]);
            let next_trimmed = ctx.lines[j].trim_start();
            if next_indent < base_indent {
                break;
            }
            if next_indent == base_indent && detect_list_marker(next_trimmed).is_none() {
                break;
            }
            ctx.pos += 1;
            continue;
        }

        let indent = count_indent(line);
        let trimmed = line.trim_start();

        if indent < base_indent {
            break;
        }

        if indent == base_indent {
            if let Some((kind, rest)) = detect_list_marker(trimmed) {
                let same_type = match (&first_kind, &kind) {
                    (ListKind::Bullet, ListKind::Bullet) => true,
                    (ListKind::Ordered, ListKind::Ordered) => true,
                    (ListKind::Todo(_), ListKind::Todo(_)) => true,
                    _ => false,
                };
                if !same_type {
                    break;
                }
                items.push(ListItem {
                    elements: parse_inline_elements(rest),
                    children: Vec::new(),
                });
                ctx.pos += 1;
            } else {
                break;
            }
        } else {
            let nested = parse_nested_content(ctx, indent);
            if let Some(last) = items.last_mut() {
                last.children.extend(nested);
            }
        }
    }

    match first_kind {
        ListKind::Bullet => MdNode::BulletList { items },
        ListKind::Ordered => MdNode::OrderedList { items },
        ListKind::Todo(done) => {
            let todo_items: Vec<ListItem> = items.into_iter().map(|item| {
                let elements = item.elements.clone();
                ListItem {
                    elements: item.elements,
                    children: {
                        let mut ch = item.children;
                        if !ch.is_empty() {
                            ch.insert(0, MdNode::Todo { done, elements });
                            ch
                        } else {
                            ch
                        }
                    },
                }
            }).collect();
            if let Some(first) = todo_items.first() {
                MdNode::Todo {
                    done,
                    elements: first.elements.clone(),
                }
            } else {
                MdNode::Todo { done, elements: vec![make_text_element("", false, false, false)] }
            }
        }
    }
}

fn parse_nested_content(ctx: &mut ParseCtx, indent: usize) -> Vec<MdNode> {
    let mut nodes = Vec::new();

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        if line.trim().is_empty() {
            let mut j = ctx.pos + 1;
            while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                j += 1;
            }
            if j >= ctx.lines.len() || count_indent(ctx.lines[j]) < indent {
                break;
            }
            ctx.pos += 1;
            continue;
        }

        let cur_indent = count_indent(line);
        if cur_indent < indent {
            break;
        }

        if cur_indent >= indent {
            let sub_indent = cur_indent;
            let trimmed = line.trim_start();

            if let Some((kind, rest)) = detect_list_marker(trimmed) {
                let item = ListItem {
                    elements: parse_inline_elements(rest),
                    children: Vec::new(),
                };
                ctx.pos += 1;

                let mut sub_items = vec![item];
                while ctx.pos < ctx.lines.len() {
                    let sub_line = ctx.lines[ctx.pos];
                    if sub_line.trim().is_empty() {
                        let mut j = ctx.pos + 1;
                        while j < ctx.lines.len() && ctx.lines[j].trim().is_empty() {
                            j += 1;
                        }
                        if j >= ctx.lines.len() || count_indent(ctx.lines[j]) < sub_indent {
                            break;
                        }
                        ctx.pos += 1;
                        continue;
                    }
                    let si = count_indent(sub_line);
                    let st = sub_line.trim_start();
                    if si < sub_indent {
                        break;
                    }
                    if si == sub_indent {
                        if let Some((k2, r2)) = detect_list_marker(st) {
                            let same = matches!(
                                (&kind, &k2),
                                (ListKind::Bullet, ListKind::Bullet)
                                    | (ListKind::Ordered, ListKind::Ordered)
                                    | (ListKind::Todo(_), ListKind::Todo(_))
                            );
                            if !same {
                                break;
                            }
                            sub_items.push(ListItem {
                                elements: parse_inline_elements(r2),
                                children: Vec::new(),
                            });
                            ctx.pos += 1;
                        } else {
                            break;
                        }
                    } else {
                        let nested = parse_nested_content(ctx, si);
                        if let Some(last) = sub_items.last_mut() {
                            last.children.extend(nested);
                        }
                    }
                }

                let list_node = match kind {
                    ListKind::Bullet => MdNode::BulletList { items: sub_items },
                    ListKind::Ordered => MdNode::OrderedList { items: sub_items },
                    ListKind::Todo(done) => MdNode::Todo {
                        done,
                        elements: sub_items
                            .first()
                            .map(|it| it.elements.clone())
                            .unwrap_or_else(|| vec![make_text_element("", false, false, false)]),
                    },
                };
                nodes.push(list_node);
            } else if trimmed.starts_with('>') {
                nodes.push(parse_block_quote_node(ctx));
            } else if trimmed.starts_with("```") {
                nodes.push(parse_code_block_node_with_indent(ctx, indent));
            } else if let Some(node) = parse_heading_node(trimmed) {
                ctx.pos += 1;
                nodes.push(node);
            } else if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                ctx.pos += 1;
                nodes.push(MdNode::Divider);
            } else {
                let mut para = trimmed.to_string();
                ctx.pos += 1;
                while ctx.pos < ctx.lines.len() {
                    let pl = ctx.lines[ctx.pos];
                    if pl.trim().is_empty() {
                        break;
                    }
                    let pi = count_indent(pl);
                    if pi < indent {
                        break;
                    }
                    if detect_list_marker(pl.trim_start()).is_some()
                        || pl.trim_start().starts_with('>')
                        || pl.trim_start().starts_with("```")
                        || parse_heading_node(pl.trim()).is_some()
                    {
                        break;
                    }
                    para.push(' ');
                    para.push_str(pl.trim());
                    ctx.pos += 1;
                }
                nodes.push(MdNode::Paragraph {
                    elements: parse_inline_elements(&para),
                });
            }
        }
    }

    nodes
}

fn parse_paragraph_node(ctx: &mut ParseCtx) -> MdNode {
    let mut text = ctx.lines[ctx.pos].trim().to_string();
    ctx.pos += 1;

    while ctx.pos < ctx.lines.len() {
        let line = ctx.lines[ctx.pos];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with("```")
            || parse_heading_node(trimmed).is_some()
            || trimmed.starts_with('>')
            || detect_list_marker(trimmed).is_some()
            || trimmed == "---"
            || trimmed == "***"
            || trimmed == "___"
            || is_table_start_at(ctx.lines, ctx.pos)
        {
            break;
        }
        text.push(' ');
        text.push_str(trimmed);
        ctx.pos += 1;
    }

    MdNode::Paragraph {
        elements: parse_inline_elements(text.trim()),
    }
}

fn alloc_id(counter: &mut usize) -> String {
    let id = format!("blk_{}", counter);
    *counter += 1;
    id
}

fn heading_block_type(level: u8) -> i64 {
    match level {
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 6,
        5 => 7,
        6 => 8,
        _ => 2,
    }
}

fn heading_block_key(level: u8) -> &'static str {
    match level {
        1 => "heading1",
        2 => "heading2",
        3 => "heading3",
        4 => "heading4",
        5 => "heading5",
        6 => "heading6",
        _ => "text",
    }
}

fn md_node_to_block_ops(node: MdNode, id_counter: &mut usize) -> Vec<BlockOp> {
    match node {
        MdNode::Heading { level, elements } => {
            let bt = heading_block_type(level);
            let key = heading_block_key(level);
            let mut block = json!({
                "block_type": bt,
            });
            block[key] = json!({ "elements": elements });
            vec![BlockOp::Simple(block)]
        }
        MdNode::Paragraph { elements } => {
            vec![BlockOp::Simple(json!({
                "block_type": 2,
                "text": { "elements": elements }
            }))]
        }
        MdNode::CodeBlock { lang, content } => {
            let mut style = json!({});
            if let Some(l) = lang {
                style["language"] = json!(l);
            }
            vec![BlockOp::Simple(json!({
                "block_type": 14,
                "code": {
                    "elements": [{ "text_run": { "content": content, "text_element_style": {} } }],
                    "style": style
                }
            }))]
        }
        MdNode::Divider => {
            vec![BlockOp::Simple(json!({
                "block_type": 22,
                "divider": {}
            }))]
        }
        MdNode::Todo { done, elements } => {
            vec![BlockOp::Simple(json!({
                "block_type": 17,
                "todo": {
                    "elements": elements,
                    "style": { "done": done }
                }
            }))]
        }
        MdNode::BulletList { items } => {
            let mut ops = Vec::new();
            for item in items {
                ops.extend(list_item_to_block_ops(item, 12, "bullet", id_counter));
            }
            ops
        }
        MdNode::OrderedList { items } => {
            let mut ops = Vec::new();
            for item in items {
                ops.extend(list_item_to_block_ops(item, 13, "ordered", id_counter));
            }
            ops
        }
        MdNode::BlockQuote { children } => {
            if children.is_empty() {
                return vec![BlockOp::Simple(json!({
                    "block_type": 15,
                    "quote": { "elements": [make_text_element("", false, false, false)] }
                }))];
            }
            let quote_id = alloc_id(id_counter);
            let mut child_ids: Vec<String> = Vec::new();
            let mut all_descendants: Vec<Value> = Vec::new();

            for child in children {
                let (root_ids, descs) = md_node_to_descendant_blocks(child, id_counter);
                child_ids.extend(root_ids);
                all_descendants.extend(descs);
            }

            let mut quote_block = json!({
                "block_id": quote_id,
                "block_type": 15,
                "quote": { "elements": [make_text_element("", false, false, false)] },
                "children": child_ids
            });
            if child_ids.is_empty() {
                quote_block["quote"] = json!({ "elements": [make_text_element("", false, false, false)] });
                vec![BlockOp::Simple(json!({
                    "block_type": 15,
                    "quote": { "elements": [make_text_element("", false, false, false)] }
                }))]
            } else {
                let mut descendants = vec![quote_block];
                descendants.extend(all_descendants);
                vec![BlockOp::Descendant {
                    children_id: vec![quote_id],
                    descendants,
                }]
            }
        }
        MdNode::Table { rows } => {
            vec![build_table_descendant(&rows, id_counter)]
        }
    }
}

fn list_item_to_block_ops(
    item: ListItem,
    block_type: i64,
    key: &str,
    id_counter: &mut usize,
) -> Vec<BlockOp> {
    if item.children.is_empty() {
        let mut block = json!({ "block_type": block_type });
        block[key] = json!({ "elements": item.elements });
        vec![BlockOp::Simple(block)]
    } else {
        let item_id = alloc_id(id_counter);
        let mut child_ids: Vec<String> = Vec::new();
        let mut all_descendants: Vec<Value> = Vec::new();

        for child in item.children {
            let (root_ids, descs) = md_node_to_descendant_blocks(child, id_counter);
            child_ids.extend(root_ids);
            all_descendants.extend(descs);
        }

        let mut item_block = json!({
            "block_id": item_id,
            "block_type": block_type,
            "children": child_ids
        });
        item_block[key] = json!({ "elements": item.elements });

        let mut descendants = vec![item_block];
        descendants.extend(all_descendants);

        vec![BlockOp::Descendant {
            children_id: vec![item_id],
            descendants,
        }]
    }
}

fn md_node_to_descendant_blocks(
    node: MdNode,
    id_counter: &mut usize,
) -> (Vec<String>, Vec<Value>) {
    match node {
        MdNode::Heading { level, elements } => {
            let id = alloc_id(id_counter);
            let bt = heading_block_type(level);
            let key = heading_block_key(level);
            let mut block = json!({
                "block_id": id,
                "block_type": bt,
                "children": []
            });
            block[key] = json!({ "elements": elements });
            (vec![id], vec![block])
        }
        MdNode::Paragraph { elements } => {
            let id = alloc_id(id_counter);
            let block = json!({
                "block_id": id,
                "block_type": 2,
                "text": { "elements": elements },
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::CodeBlock { lang, content } => {
            let id = alloc_id(id_counter);
            let mut style = json!({});
            if let Some(l) = lang {
                style["language"] = json!(l);
            }
            let block = json!({
                "block_id": id,
                "block_type": 14,
                "code": {
                    "elements": [{ "text_run": { "content": content, "text_element_style": {} } }],
                    "style": style
                },
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::Divider => {
            let id = alloc_id(id_counter);
            let block = json!({
                "block_id": id,
                "block_type": 22,
                "divider": {},
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::Todo { done, elements } => {
            let id = alloc_id(id_counter);
            let block = json!({
                "block_id": id,
                "block_type": 17,
                "todo": {
                    "elements": elements,
                    "style": { "done": done }
                },
                "children": []
            });
            (vec![id], vec![block])
        }
        MdNode::BulletList { items } => {
            let mut root_ids = Vec::new();
            let mut all_blocks = Vec::new();
            for item in items {
                let (ids, blocks) = list_item_to_descendant(item, 12, "bullet", id_counter);
                root_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            (root_ids, all_blocks)
        }
        MdNode::OrderedList { items } => {
            let mut root_ids = Vec::new();
            let mut all_blocks = Vec::new();
            for item in items {
                let (ids, blocks) = list_item_to_descendant(item, 13, "ordered", id_counter);
                root_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            (root_ids, all_blocks)
        }
        MdNode::BlockQuote { children } => {
            let quote_id = alloc_id(id_counter);
            let mut child_ids: Vec<String> = Vec::new();
            let mut all_blocks: Vec<Value> = Vec::new();
            for child in children {
                let (ids, blocks) = md_node_to_descendant_blocks(child, id_counter);
                child_ids.extend(ids);
                all_blocks.extend(blocks);
            }
            let quote_block = json!({
                "block_id": quote_id,
                "block_type": 15,
                "quote": { "elements": [make_text_element("", false, false, false)] },
                "children": child_ids
            });
            let mut result = vec![quote_block];
            result.extend(all_blocks);
            (vec![quote_id], result)
        }
        MdNode::Table { rows } => {
            let (ids, blocks) = build_table_descendant_data(&rows, id_counter);
            (ids, blocks)
        }
    }
}

fn list_item_to_descendant(
    item: ListItem,
    block_type: i64,
    key: &str,
    id_counter: &mut usize,
) -> (Vec<String>, Vec<Value>) {
    let item_id = alloc_id(id_counter);
    let mut child_ids: Vec<String> = Vec::new();
    let mut all_blocks: Vec<Value> = Vec::new();

    for child in item.children {
        let (ids, blocks) = md_node_to_descendant_blocks(child, id_counter);
        child_ids.extend(ids);
        all_blocks.extend(blocks);
    }

    let mut item_block = json!({
        "block_id": item_id,
        "block_type": block_type,
        "children": child_ids
    });
    item_block[key] = json!({ "elements": item.elements });

    let mut result = vec![item_block];
    result.extend(all_blocks);
    (vec![item_id], result)
}

fn build_table_descendant(rows: &[Vec<String>], id_counter: &mut usize) -> BlockOp {
    let (children_id, descendants) = build_table_descendant_data(rows, id_counter);
    BlockOp::Descendant {
        children_id,
        descendants,
    }
}

fn build_table_descendant_data(
    rows: &[Vec<String>],
    id_counter: &mut usize,
) -> (Vec<String>, Vec<Value>) {
    let row_size = rows.len();
    let column_size = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if row_size == 0 || column_size == 0 {
        let id = alloc_id(id_counter);
        return (
            vec![id.clone()],
            vec![json!({
                "block_id": id,
                "block_type": 2,
                "text": { "elements": [make_text_element("", false, false, false)] },
                "children": []
            })],
        );
    }

    let table_id = alloc_id(id_counter);
    let mut cell_ids: Vec<String> = Vec::new();
    let mut descendants: Vec<Value> = Vec::new();

    for row in rows {
        for col_idx in 0..column_size {
            let cell_id = alloc_id(id_counter);
            cell_ids.push(cell_id.clone());

            let cell_text = row.get(col_idx).cloned().unwrap_or_default();
            let child_id = alloc_id(id_counter);

            let cell_block = json!({
                "block_id": cell_id,
                "block_type": 32,
                "table_cell": {},
                "children": [child_id]
            });

            let child_block = json!({
                "block_id": child_id,
                "block_type": 2,
                "text": { "elements": parse_inline_elements(&cell_text) },
                "children": []
            });

            descendants.push(cell_block);
            descendants.push(child_block);
        }
    }

    let table_block = json!({
        "block_id": table_id,
        "block_type": 31,
        "table": {
            "property": {
                "row_size": row_size,
                "column_size": column_size
            }
        },
        "children": cell_ids
    });

    let mut all_descendants = vec![table_block];
    all_descendants.extend(descendants);
    (vec![table_id], all_descendants)
}

fn convert_markdown_to_docx_blocks(markdown: &str) -> Vec<BlockOp> {
    let ast = parse_markdown_ast(markdown);
    let mut id_counter: usize = 1;
    let mut ops = Vec::new();
    for node in ast {
        ops.extend(md_node_to_block_ops(node, &mut id_counter));
    }
    ops
}

fn feishu_doc_create_from_markdown(args: &Value) -> Result<String, JsonRpcErr> {
    let title = args
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let markdown_content = args
        .get("markdown_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let folder_token = args
        .get("folder_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if title.is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: title is required",
            Some(json!({ "title": title })),
        ));
    }
    if markdown_content.trim().is_empty() {
        return Err(json_rpc_error(
            -32602,
            "Invalid params: markdown_content is required",
            None,
        ));
    }

    let base_url = resolve_base_url();
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to build http client",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let document_id = with_user_access_token(
        &client,
        &base_url,
        "Missing user_access_token. Create document requires OAuth once.",
        |token| {
            let mut create_body = json!({
                "title": title,
                "folder_token": folder_token
            });
            if folder_token.is_empty() {
                create_body
                    .as_object_mut()
                    .unwrap()
                    .remove("folder_token");
            }

            let url = format!(
                "{}/open-apis/docx/v1/documents",
                base_url.trim_end_matches('/')
            );
            let resp = client
                .post(&url)
                .header(
                    "Authorization",
                    format!("Bearer {}", token.trim()),
                )
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&create_body)
                .send()
                .map_err(|e| {
                    json_rpc_error(
                        -32000,
                        "Failed to create document",
                        Some(json!({ "error": e.to_string() })),
                    )
                })?;

            let text = resp.text().map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to read response body",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let json: Value = serde_json::from_str(&text).map_err(|e| {
                json_rpc_error(
                    -32000,
                    "Failed to parse response JSON",
                    Some(json!({ "error": e.to_string(), "body": text })),
                )
            })?;

            let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            if code != 0 {
                let err_json = json.clone();
                return Err(json_rpc_error(
                    -32000,
                    "Failed to create document",
                    Some(err_json),
                ));
            }

            let doc_id = json
                .get("data")
                .and_then(|d| d.get("document_id"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    let err_json = json.clone();
                    json_rpc_error(-32000, "No document_id in response", Some(err_json))
                })?;

            Ok(doc_id.to_string())
        },
    )?;

    let block_ops = convert_markdown_to_docx_blocks(&markdown_content);
    if block_ops.is_empty() {
        let doc_url = format!(
            "https://{}.feishu.cn/docx/{}",
            extract_domain_from_base_url(&base_url),
            document_id
        );
        return Ok(format!(
            "Created document (empty): {}\nID: {}",
            doc_url, document_id
        ));
    }

    let mut index: i64 = 0;
    let mut simple_batch: Vec<Value> = Vec::new();

    for op in &block_ops {
        match op {
            BlockOp::Simple(block) => {
                simple_batch.push(block.clone());
                if simple_batch.len() >= 50 {
                    let batch = simple_batch.clone();
                    with_user_access_token(
                        &client,
                        &base_url,
                        "Missing user_access_token. Update document requires OAuth.",
                        |token| {
                            create_children_batch(
                                &client,
                                &base_url,
                                token,
                                &document_id,
                                &batch,
                                index,
                            )?;
                            Ok(())
                        },
                    )?;
                    index += batch.len() as i64;
                    simple_batch.clear();
                }
            }
            BlockOp::Descendant {
                children_id,
                descendants,
            } => {
                if !simple_batch.is_empty() {
                    let batch = simple_batch.clone();
                    with_user_access_token(
                        &client,
                        &base_url,
                        "Missing user_access_token. Update document requires OAuth.",
                        |token| {
                            create_children_batch(
                                &client,
                                &base_url,
                                token,
                                &document_id,
                                &batch,
                                index,
                            )?;
                            Ok(())
                        },
                    )?;
                    index += batch.len() as i64;
                    simple_batch.clear();
                }
                let desc = descendants.clone();
                let cids = children_id.clone();
                with_user_access_token(
                    &client,
                    &base_url,
                    "Missing user_access_token. Update document requires OAuth.",
                    |token| {
                        create_descendant_batch(
                            &client,
                            &base_url,
                            token,
                            &document_id,
                            &cids,
                            &desc,
                            index,
                        )?;
                        Ok(())
                    },
                )?;
                index += 1;
            }
        }
    }

    if !simple_batch.is_empty() {
        let batch = simple_batch.clone();
        with_user_access_token(
            &client,
            &base_url,
            "Missing user_access_token. Update document requires OAuth.",
            |token| {
                create_children_batch(
                    &client,
                    &base_url,
                    token,
                    &document_id,
                    &batch,
                    index,
                )?;
                Ok(())
            },
        )?;
    }

    let doc_url = format!(
        "https://{}.feishu.cn/docx/{}",
        extract_domain_from_base_url(&base_url),
        document_id
    );

    Ok(format!(
        "Created document: {}\nID: {}",
        doc_url, document_id
    ))
}

fn create_children_batch(
    client: &Client,
    base_url: &str,
    token: &str,
    document_id: &str,
    children: &[Value],
    index: i64,
) -> Result<(), JsonRpcErr> {
    let url = format!(
        "{}/open-apis/docx/v1/documents/{}/blocks/{}/children",
        base_url.trim_end_matches('/'),
        document_id,
        document_id
    );
    let body = json!({
        "children": children,
        "index": index
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create document blocks",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;

    let json: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse response JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;

    let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "Failed to create document blocks",
            Some(json),
        ));
    }

    Ok(())
}

fn create_descendant_batch(
    client: &Client,
    base_url: &str,
    token: &str,
    document_id: &str,
    children_id: &[String],
    descendants: &[Value],
    index: i64,
) -> Result<(), JsonRpcErr> {
    let url = format!(
        "{}/open-apis/docx/v1/documents/{}/blocks/{}/descendant",
        base_url.trim_end_matches('/'),
        document_id,
        document_id
    );
    let body = json!({
        "index": index,
        "children_id": children_id,
        "descendants": descendants
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to create table descendant blocks",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;

    let json: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to parse response JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;

    let code = json.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "Failed to create table descendant blocks",
            Some(json),
        ));
    }

    Ok(())
}

fn extract_domain_from_base_url(base_url: &str) -> String {
    base_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .split('/')
        .next()
        .unwrap_or("app.feishu.cn")
        .to_string()
}

#[derive(Debug, Clone)]
struct JsonRpcErr {
    code: i64,
    message: String,
    data: Option<Value>,
}

fn json_rpc_error(code: i64, message: &str, data: Option<Value>) -> JsonRpcErr {
    JsonRpcErr {
        code,
        message: message.to_string(),
        data,
    }
}

fn write_json_rpc_result(out: &mut dyn Write, id: Option<&Value>, result: Value) -> io::Result<()> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "result": result
    });
    writeln!(out, "{payload}")?;
    out.flush()
}

fn write_json_rpc_error(
    out: &mut dyn Write,
    id: Option<&Value>,
    code: i64,
    message: &str,
    data: Option<Value>,
) -> io::Result<()> {
    let mut err = json!({
        "code": code,
        "message": message
    });
    if let Some(d) = data {
        err["data"] = d;
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "error": err
    });
    writeln!(out, "{payload}")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn invalid_token_err() -> JsonRpcErr {
        json_rpc_error(
            -32000,
            "raw_content API returned non-success HTTP status",
            Some(json!({
                "status": 400,
                "body": "{\"code\":99991668,\"msg\":\"Invalid access token for authorization. Please make a request with token attached.\"}"
            })),
        )
    }

    fn start_mock_http_server<F>(handler: F) -> String
    where
        F: Fn(String) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    continue;
                }
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                let body = handler(request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{}", addr)
    }

    #[test]
    fn render_docx_blocks_keeps_diagram_placeholder_near_heading() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["h2", "diagram"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "h2",
                "block_type": 4,
                "parent_id": "root",
                "heading2": {
                    "elements": [
                        { "text_run": { "content": "6.1 文字绘图", "text_element_style": {} } }
                    ]
                }
            }),
            json!({
                "block_id": "diagram",
                "block_type": 21,
                "parent_id": "root",
                "diagram": { "diagram_type": 1 }
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("## 6.1 文字绘图"));
        assert!(rendered.contains("[流程图]"));
        assert!(rendered.find("## 6.1 文字绘图") < rendered.find("[流程图]"));
    }

    #[test]
    fn prefer_blocks_render_when_special_blocks_would_be_lost() {
        let raw = "## 6.1 文字绘图";
        let rendered = "## 6.1 文字绘图\n[流程图]";
        assert!(should_prefer_docx_blocks_render(raw, rendered));
        assert!(!should_prefer_docx_blocks_render(raw, raw));
    }

    #[test]
    fn render_docx_blocks_keeps_board_placeholders_after_heading() {
        let items = vec![
            json!({
                "block_id": "root",
                "block_type": 1,
                "parent_id": "",
                "children": ["h3", "board1", "board2"],
                "page": { "elements": [] }
            }),
            json!({
                "block_id": "h3",
                "block_type": 5,
                "parent_id": "root",
                "heading3": {
                    "elements": [
                        { "text_run": { "content": "6.1. 系统架构图", "text_element_style": {} } }
                    ]
                }
            }),
            json!({
                "block_id": "board1",
                "block_type": 43,
                "parent_id": "root",
                "board": { "token": "board-token-1" }
            }),
            json!({
                "block_id": "board2",
                "block_type": 43,
                "parent_id": "root",
                "board": { "token": "board-token-2" }
            }),
        ];

        let rendered = render_docx_blocks_as_text(&items, None);
        assert!(rendered.contains("### 6.1. 系统架构图"));
        assert!(rendered.contains("[文字绘图: board-token-1]"));
        assert!(rendered.contains("[文字绘图: board-token-2]"));
        assert!(rendered.find("### 6.1. 系统架构图") < rendered.find("[文字绘图: board-token-1]"));
    }

    #[test]
    fn detects_invalid_user_access_token_from_raw_body() {
        assert!(is_invalid_user_access_token_error(&invalid_token_err()));
    }

    #[test]
    fn with_user_access_token_falls_back_to_token_store_when_env_token_is_stale() {
        let _guard = env_lock().lock().unwrap();
        let token_store_path = format!(
            "/tmp/mcp_feishu_token_test_{}_{}.json",
            std::process::id(),
            epoch_ms_now()
        );
        let old_env_token_store_path = std::env::var("FEISHU_TOKEN_STORE_PATH").ok();
        let old_env_user_token = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        let _ = fs::remove_file(&token_store_path);

        unsafe {
            std::env::set_var("FEISHU_TOKEN_STORE_PATH", &token_store_path);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-stale-token");
        }

        save_token_store(&TokenStore {
            user_access_token: Some("u-fresh-token".to_string()),
            user_access_token_expires_at_epoch_ms: Some(epoch_ms_now() + 3_600_000),
            refresh_token: None,
            refresh_token_expires_in: None,
            updated_at_epoch_ms: Some(epoch_ms_now()),
        })
        .unwrap();

        let client = Client::builder().build().unwrap();
        let mut seen = Vec::new();
        let result = with_user_access_token(
            &client,
            "https://open.feishu.cn",
            "Missing user_access_token. Fetch requires OAuth once.",
            |token| {
                seen.push(token.to_string());
                if token == "u-stale-token" {
                    Err(invalid_token_err())
                } else {
                    Ok(token.to_string())
                }
            },
        )
        .unwrap();

        assert_eq!(result, "u-fresh-token");
        assert_eq!(seen, vec!["u-stale-token", "u-fresh-token"]);

        let _ = fs::remove_file(&token_store_path);
        unsafe {
            if let Some(v) = old_env_token_store_path {
                std::env::set_var("FEISHU_TOKEN_STORE_PATH", v);
            } else {
                std::env::remove_var("FEISHU_TOKEN_STORE_PATH");
            }
            if let Some(v) = old_env_user_token {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn messages_search_uses_user_access_token() {
        let _guard = env_lock().lock().unwrap();
        let seen_auth = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_auth_clone = Arc::clone(&seen_auth);
        let base_url = start_mock_http_server(move |request| {
            if let Some(line) = request
                .lines()
                .find(|line| line.starts_with("Authorization: "))
            {
                seen_auth_clone.lock().unwrap().push(line.to_string());
            }
            serde_json::json!({
                "code": 0,
                "msg": "ok",
                "data": {
                    "items": [{
                        "message_id": "om_1",
                        "chat_id": "oc_1",
                        "msg_type": "text",
                        "create_time": "1",
                        "sender": {"id": "ou_1"},
                        "body": {"content": "{\"text\":\"dataagent ping\"}"}
                    }],
                    "has_more": false
                }
            })
            .to_string()
        });

        let old_base = std::env::var("FEISHU_BASE_URL").ok();
        let old_user = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        unsafe {
            std::env::set_var("FEISHU_BASE_URL", &base_url);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-test-user-token");
        }

        let result = feishu_messages_search(&serde_json::json!({
            "search_key": "dataagent",
            "chat_id": "oc_1"
        }))
        .unwrap();

        assert!(result.contains("matched: 1"));
        assert!(seen_auth
            .lock()
            .unwrap()
            .iter()
            .all(|line| line.contains("Bearer u-test-user-token")));

        unsafe {
            if let Some(v) = old_base {
                std::env::set_var("FEISHU_BASE_URL", v);
            } else {
                std::env::remove_var("FEISHU_BASE_URL");
            }
            if let Some(v) = old_user {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn messages_global_search_uses_user_access_token_and_includes_topic_chats() {
        let _guard = env_lock().lock().unwrap();
        let seen_auth = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_auth_clone = Arc::clone(&seen_auth);
        let base_url = start_mock_http_server(move |request| {
            if let Some(line) = request
                .lines()
                .find(|line| line.starts_with("Authorization: "))
            {
                seen_auth_clone.lock().unwrap().push(line.to_string());
            }
            let first_line = request.lines().next().unwrap_or_default().to_string();
            if first_line.contains("/open-apis/im/v1/chats?") || first_line.contains("/open-apis/im/v1/chats ") {
                return serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [{
                            "chat_id": "oc_topic",
                            "name": "dataagent",
                            "chat_mode": "topic"
                        }],
                        "page_token": ""
                    }
                })
                .to_string();
            }
            serde_json::json!({
                "code": 0,
                "msg": "ok",
                "data": {
                    "items": [{
                        "message_id": "om_topic_1",
                        "chat_id": "oc_topic",
                        "msg_type": "text",
                        "create_time": "1",
                        "sender": {"id": "ou_1"},
                        "body": {"content": "{\"text\":\"hello from thread\"}"}
                    }],
                    "page_token": ""
                }
            })
            .to_string()
        });

        let old_base = std::env::var("FEISHU_BASE_URL").ok();
        let old_user = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        unsafe {
            std::env::set_var("FEISHU_BASE_URL", &base_url);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-test-user-token");
        }

        let result = feishu_messages_global_search(&serde_json::json!({
            "search_key": "dataagent",
            "limit": 10,
            "max_chats": 10,
            "msgs_per_chat": 50,
            "max_pages": 2
        }))
        .unwrap();

        assert!(result.contains("scanned_chats: 1/1"));
        assert!(result.contains("matched: 1"));
        assert!(result.contains("\"chat_name\": \"dataagent\""));
        assert!(seen_auth
            .lock()
            .unwrap()
            .iter()
            .all(|line| line.contains("Bearer u-test-user-token")));

        unsafe {
            if let Some(v) = old_base {
                std::env::set_var("FEISHU_BASE_URL", v);
            } else {
                std::env::remove_var("FEISHU_BASE_URL");
            }
            if let Some(v) = old_user {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn messages_global_search_prioritizes_recent_chats_and_stops_early() {
        let _guard = env_lock().lock().unwrap();
        let base_url = start_mock_http_server(move |request| {
            let first_line = request.lines().next().unwrap_or_default().to_string();
            if first_line.contains("/open-apis/im/v1/chats?") || first_line.contains("/open-apis/im/v1/chats ") {
                return serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [
                            {"chat_id": "oc_old", "name": "old-chat", "chat_mode": "group", "last_message_time": "100"},
                            {"chat_id": "oc_new", "name": "new-chat", "chat_mode": "group", "last_message_time": "200"}
                        ],
                        "page_token": ""
                    }
                }).to_string();
            }
            if first_line.contains("container_id=oc_old") {
                return serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [{
                            "message_id": "om_old",
                            "chat_id": "oc_old",
                            "msg_type": "text",
                            "create_time": "100",
                            "sender": {"id": "ou_1"},
                            "body": {"content": "{\"text\":\"dataagent old\"}"}
                        }],
                        "has_more": false
                    }
                }).to_string();
            }
            serde_json::json!({
                "code": 0,
                "msg": "ok",
                "data": {
                    "items": [{
                        "message_id": "om_new",
                        "chat_id": "oc_new",
                        "msg_type": "text",
                        "create_time": "200",
                        "sender": {"id": "ou_2"},
                        "body": {"content": "{\"text\":\"dataagent new\"}"}
                    }],
                    "has_more": false
                }
            }).to_string()
        });

        let old_base = std::env::var("FEISHU_BASE_URL").ok();
        let old_user = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        unsafe {
            std::env::set_var("FEISHU_BASE_URL", &base_url);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-test-user-token");
        }

        let result = feishu_messages_global_search(&serde_json::json!({
            "search_key": "dataagent",
            "limit": 1,
            "max_chats": 2,
            "msgs_per_chat": 50,
            "max_pages": 1
        }))
        .unwrap();

        assert!(result.contains("scanned_chats: 2/2"));
        assert!(result.contains("matched: 1"));
        assert!(result.contains("\"message_id\": \"om_new\""));
        assert!(!result.contains("\"message_id\": \"om_old\""));

        unsafe {
            if let Some(v) = old_base {
                std::env::set_var("FEISHU_BASE_URL", v);
            } else {
                std::env::remove_var("FEISHU_BASE_URL");
            }
            if let Some(v) = old_user {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn messages_global_search_does_not_stop_early_when_newer_chat_may_still_win() {
        let _guard = env_lock().lock().unwrap();
        let base_url = start_mock_http_server(move |request| {
            let first_line = request.lines().next().unwrap_or_default().to_string();
            if first_line.contains("/open-apis/im/v1/chats?") || first_line.contains("/open-apis/im/v1/chats ") {
                return serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [
                            {"chat_id": "oc_hot", "name": "hot-chat", "chat_mode": "group", "last_message_time": "2000"},
                            {"chat_id": "oc_mid", "name": "mid-chat", "chat_mode": "group", "last_message_time": "1500"}
                        ],
                        "page_token": ""
                    }
                }).to_string();
            }
            if first_line.contains("container_id=oc_hot") {
                return serde_json::json!({
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [
                            {
                                "message_id": "om_hot_nonmatch",
                                "chat_id": "oc_hot",
                                "msg_type": "text",
                                "create_time": "2000",
                                "sender": {"id": "ou_1"},
                                "body": {"content": "{\"text\":\"other\"}"}
                            },
                            {
                                "message_id": "om_hot_old_match",
                                "chat_id": "oc_hot",
                                "msg_type": "text",
                                "create_time": "100",
                                "sender": {"id": "ou_1"},
                                "body": {"content": "{\"text\":\"dataagent old\"}"}
                            }
                        ],
                        "has_more": false
                    }
                }).to_string();
            }
            serde_json::json!({
                "code": 0,
                "msg": "ok",
                "data": {
                    "items": [{
                        "message_id": "om_mid_new_match",
                        "chat_id": "oc_mid",
                        "msg_type": "text",
                        "create_time": "1400",
                        "sender": {"id": "ou_2"},
                        "body": {"content": "{\"text\":\"dataagent new\"}"}
                    }],
                    "has_more": false
                }
            }).to_string()
        });

        let old_base = std::env::var("FEISHU_BASE_URL").ok();
        let old_user = std::env::var("FEISHU_USER_ACCESS_TOKEN").ok();
        unsafe {
            std::env::set_var("FEISHU_BASE_URL", &base_url);
            std::env::set_var("FEISHU_USER_ACCESS_TOKEN", "u-test-user-token");
        }

        let result = feishu_messages_global_search(&serde_json::json!({
            "search_key": "dataagent",
            "limit": 1,
            "max_chats": 2,
            "msgs_per_chat": 50,
            "max_pages": 1
        }))
        .unwrap();

        assert!(result.contains("scanned_chats: 2/2"));
        assert!(result.contains("\"message_id\": \"om_mid_new_match\""));
        assert!(!result.contains("\"message_id\": \"om_hot_old_match\""));

        unsafe {
            if let Some(v) = old_base {
                std::env::set_var("FEISHU_BASE_URL", v);
            } else {
                std::env::remove_var("FEISHU_BASE_URL");
            }
            if let Some(v) = old_user {
                std::env::set_var("FEISHU_USER_ACCESS_TOKEN", v);
            } else {
                std::env::remove_var("FEISHU_USER_ACCESS_TOKEN");
            }
        }
    }

    #[test]
    fn chat_mode_matches_keeps_topic_when_group_search_enabled() {
        assert!(chat_mode_matches("topic", true, true));
        assert!(!chat_mode_matches("topic", true, false));
    }

    #[test]
    fn message_create_time_parses_timestamp() {
        let item = json!({"create_time": "1772192845747"});
        assert_eq!(message_create_time(&item), 1_772_192_845_747);
    }
    #[test]
    fn render_text_elements_preserves_links() {
        // 测试普通文本（无链接）
        let elements_no_link = json!([
            { "text_run": { "content": "普通文本" } }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_no_link), None),
            "普通文本"
        );

        // 测试带链接的文本
        let elements_with_link = json!([
            {
                "text_run": {
                    "content": "点击这里",
                    "text_style": {
                        "link": {
                            "url": "https://example.com"
                        }
                    }
                }
            }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_with_link), None),
            "[点击这里](https://example.com)"
        );

        // 测试混合文本（有链接和无链接）
        let elements_mixed = json!([
            { "text_run": { "content": "前面文字" } },
            {
                "text_run": {
                    "content": "链接文本",
                    "text_style": {
                        "link": {
                            "url": "https://test.com"
                        }
                    }
                }
            },
            { "text_run": { "content": "后面文字" } }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_mixed), None),
            "前面文字[链接文本](https://test.com)后面文字"
        );

        // 测试只有 URL 没有文本内容的情况
        let elements_url_only = json!([
            {
                "text_run": {
                    "content": "",
                    "text_style": {
                        "link": {
                            "url": "https://bare-url.com"
                        }
                    }
                }
            }
        ]);
        assert_eq!(
            render_text_elements(Some(&elements_url_only), None),
            "https://bare-url.com"
        );
    }

    #[test]
    fn resolve_message_container_rejects_ambiguous_input() {
        let err = resolve_message_container(&json!({
            "chat_id": "oc_123",
            "thread_id": "omt_456"
        }))
        .unwrap_err();
        assert!(err.message.contains("cannot both be set"));
    }

    #[test]
    fn build_message_searchable_text_extracts_text_json() {
        let item = json!({
            "msg_type": "text",
            "body": {
                "content": "{\"text\":\"项目复盘记录\"}"
            }
        });
        let searchable = build_feishu_message_searchable_text(&item);
        assert!(searchable.contains("项目复盘记录"));
    }

    #[test]
    fn build_message_searchable_text_flattens_card_payload() {
        let item = json!({
            "msg_type": "interactive",
            "body": {
                "content": "{\"config\":{\"wide_screen_mode\":true},\"elements\":[{\"tag\":\"div\",\"text\":{\"tag\":\"plain_text\",\"content\":\"报警记录 #123\"}}]}"
            }
        });
        let searchable = build_feishu_message_searchable_text(&item);
        assert!(searchable.contains("报警记录 #123"));
    }

    #[test]
    fn build_message_searchable_text_skips_urls_and_card_links() {
        let item = json!({
            "msg_type": "interactive",
            "body": {
                "content": "{\"title\":\"告警\",\"card_link\":{\"url\":\"https://meego.larkoffice.com/dataagent/story/detail/1\"},\"elements\":[[{\"tag\":\"text\",\"text\":\"正文内容\"},{\"tag\":\"a\",\"href\":\"https://example.com/dataagent\",\"text\":\"查看详情\"}]]}"
            },
            "mentions": [{
                "name": "dataagent-bot"
            }],
            "sender": {
                "id": "dataagent-sender"
            }
        });
        let searchable = build_feishu_message_searchable_text(&item);
        assert!(searchable.contains("正文内容"));
        assert!(searchable.contains("查看详情"));
        assert!(!searchable.contains("https://meego.larkoffice.com/dataagent/story/detail/1"));
        assert!(!searchable.contains("https://example.com/dataagent"));
        assert!(!searchable.contains("dataagent-bot"));
        assert!(!searchable.contains("dataagent-sender"));
    }

    #[test]
    fn build_global_searchable_text_only_uses_body_text() {
        let item = json!({
            "msg_type": "text",
            "body": {
                "content": "{\"text\":\"hello world\"}"
            }
        });
        let chat = ChatInfo {
            chat_id: "oc_123".to_string(),
            name: "dataagent".to_string(),
            recent_time: 0,
            source_order: 0,
        };
        let searchable = build_feishu_global_searchable_text(&item, &chat);
        assert_eq!(searchable, "hello world");
    }
}
