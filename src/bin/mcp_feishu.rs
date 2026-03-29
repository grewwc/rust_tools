use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{fs, os::unix::fs::PermissionsExt};

use reqwest::blocking::Client;
use rust_tools::common::configw;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

struct CachedToken {
    token: String,
    expires_at: Instant,
}

static APP_TOKEN_CACHE: OnceLock<Mutex<Option<CachedToken>>> = OnceLock::new();
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
                        "scope": { "type": "string", "description": "Scopes separated by space. Tip: include offline_access if you need refresh_token." },
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
                "description": "Exchange OAuth code for user_access_token (requires app_id/app_secret). Returns user_access_token and refresh_token.",
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
                "description": "Refresh user_access_token using refresh_token (requires app_id/app_secret).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh_token": { "type": "string", "description": "Refresh token. If omitted, uses FEISHU_REFRESH_TOKEN env or feishu.refresh_token in ~/.configW" }
                    }
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

    let token = if let Some(tok) = resolve_user_access_token() {
        tok
    } else if let Ok(tok) = get_user_access_token_cached(&client, &base_url) {
        tok
    } else {
        return Err(json_rpc_error(
            -32000,
            "Missing user_access_token. docs-api search requires user_access_token. Use OAuth once, then keep refresh_token for automatic refresh.",
            Some(json!({
                "next_steps": [
                    "Call oauth_authorize_url to get an authorization URL",
                    "Open the URL in a browser, complete authorization",
                    "Call oauth_wait_local_code (or copy code from redirect URL)",
                    "Call oauth_exchange_code to get user_access_token + refresh_token",
                    "Then store FEISHU_REFRESH_TOKEN (and app_id/app_secret) so the token can be refreshed automatically"
                ],
                "config_keys": ["feishu.app_id", "feishu.app_secret", "feishu.user_access_token", "feishu.refresh_token"],
                "env": ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_USER_ACCESS_TOKEN", "FEISHU_REFRESH_TOKEN"]
            })),
        ));
    };

    let (mut status, mut text) = do_docs_search_request(&client, &url, &token, &body)?;
    if !status.is_success()
        && let Ok(v) = serde_json::from_str::<Value>(&text)
    {
        let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
        if code == 99991668
            && let Ok(tok) = refresh_user_access_token_and_cache(&client, &base_url)
        {
            let (s2, t2) = do_docs_search_request(&client, &url, &tok, &body)?;
            status = s2;
            text = t2;
        }
    }
    if !status.is_success() {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
            if code == 99991668 {
                return Err(json_rpc_error(
                    -32000,
                    "Invalid access token. Provide a valid user_access_token or set refresh_token for automatic refresh.",
                    Some(json!({
                        "status": status.as_u16(),
                        "feishu_code": code,
                        "msg": v.get("msg").cloned().unwrap_or(Value::Null),
                        "next_steps": [
                            "If you have refresh_token: call oauth_refresh_user_access_token, then update FEISHU_USER_ACCESS_TOKEN (or set FEISHU_REFRESH_TOKEN for auto refresh)",
                            "Otherwise: run oauth_authorize_url -> oauth_wait_local_code -> oauth_exchange_code"
                        ]
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
        let token = item
            .get("docs_token")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let owner_id = item.get("owner_id").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(&format!(
            "{}. [{}] {} (token: {}, owner_id: {})\n",
            i + 1,
            docs_type,
            title.trim(),
            token.trim(),
            owner_id.trim()
        ));
    }
    Ok(out.trim_end().to_string())
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
        |token| feishu_fetch_raw_content(&client, &base_url, token, &docs_type, &docs_token, lang),
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
            "doc" | "docx" => feishu_fetch_raw_content(&client, &base_url, token, &kind, &tok, lang),
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
        rust_tools::common::utils::expanduser("~/.config/rust_tools/feishu_docs_text")
            .as_ref()
            .to_string()
    } else {
        rust_tools::common::utils::expanduser(&out_dir)
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
        |token| feishu_fetch_raw_content(&client, &base_url, token, &docs_type, &docs_token, lang),
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

    let blocks_text =
        feishu_fetch_docx_blocks_text(client, base_url, user_access_token, docs_token)?;
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

    Ok(render_docx_blocks_as_text(&items))
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

fn render_docx_blocks_as_text(items: &[Value]) -> String {
    if items.is_empty() {
        return String::new();
    }

    let mut by_id: HashMap<String, &Value> = HashMap::new();
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
    render_docx_block_text(&root_id, &by_id, &mut out);
    normalize_rendered_docx_text(&out)
}

fn render_docx_block_text(block_id: &str, by_id: &HashMap<String, &Value>, out: &mut String) {
    let Some(block) = by_id.get(block_id).copied() else {
        return;
    };

    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let line = render_docx_block_line(block);
    if !line.is_empty() {
        out.push_str(&line);
        out.push('\n');
    }

    if block_type == 31 {
        render_docx_table_cells(block, by_id, out);
        return;
    }

    let children = block
        .get("children")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for child in children {
        if let Some(child_id) = child.as_str() {
            render_docx_block_text(child_id, by_id, out);
        }
    }
}

fn render_docx_table_cells(block: &Value, by_id: &HashMap<String, &Value>, out: &mut String) {
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

    let mut row: Vec<String> = Vec::new();
    for cell in cells {
        let Some(cell_id) = cell.as_str() else {
            continue;
        };
        let cell_text = render_docx_table_cell_text(cell_id, by_id);
        row.push(cell_text);
        if row.len() >= col_size {
            out.push_str(&row.join("\t"));
            out.push('\n');
            row.clear();
        }
    }
    if !row.is_empty() {
        out.push_str(&row.join("\t"));
        out.push('\n');
    }
}

fn render_docx_table_cell_text(cell_id: &str, by_id: &HashMap<String, &Value>) -> String {
    let Some(block) = by_id.get(cell_id).copied() else {
        return String::new();
    };

    let direct = render_text_elements(
        block
            .get("table_cell")
            .and_then(|v| v.get("elements"))
            .or_else(|| block.get("text").and_then(|v| v.get("elements"))),
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
        let line = render_docx_block_line(child_block);
        if !line.is_empty() {
            parts.push(line);
        }
    }
    parts.join(" ")
}

fn render_docx_block_line(block: &Value) -> String {
    let block_type = block
        .get("block_type")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    match block_type {
        1 => render_text_elements(block.get("page").and_then(|v| v.get("elements"))),
        2 => render_text_elements(block.get("text").and_then(|v| v.get("elements"))),
        3..=11 => {
            let level = (block_type - 2) as usize;
            let text = render_text_elements(
                block
                    .get(format!("heading{}", level).as_str())
                    .and_then(|v| v.get("elements")),
            );
            if text.is_empty() {
                String::new()
            } else {
                format!("{} {}", "#".repeat(level), text)
            }
        }
        12 => {
            let text = render_text_elements(block.get("bullet").and_then(|v| v.get("elements")));
            if text.is_empty() {
                String::new()
            } else {
                format!("- {}", text)
            }
        }
        13 => {
            let text = render_text_elements(block.get("ordered").and_then(|v| v.get("elements")));
            if text.is_empty() {
                String::new()
            } else {
                format!("1. {}", text)
            }
        }
        14 => {
            let text = render_text_elements(block.get("code").and_then(|v| v.get("elements")));
            if text.is_empty() {
                String::new()
            } else {
                format!("```text\n{}\n```", text)
            }
        }
        15 => {
            let text = render_text_elements(block.get("quote").and_then(|v| v.get("elements")));
            if text.is_empty() {
                String::new()
            } else {
                format!("> {}", text)
            }
        }
        17 => {
            let todo = block.get("todo");
            let text = render_text_elements(todo.and_then(|v| v.get("elements")));
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
            let text = render_text_elements(block.get("callout").and_then(|v| v.get("elements")));
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

fn render_text_elements(elements: Option<&Value>) -> String {
    let Some(arr) = elements.and_then(|v| v.as_array()) else {
        return String::new();
    };

    let mut out = String::new();
    for el in arr {
        if let Some(v) = el
            .get("text_run")
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
        {
            out.push_str(v);
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
            .and_then(|v| {
                v.get("title")
                    .or_else(|| v.get("obj_type"))
                    .or_else(|| v.get("token"))
            })
            .and_then(|v| v.as_str())
        {
            out.push_str(v);
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

fn resolve_app_credentials() -> Option<(String, String)> {
    let env_id = std::env::var("FEISHU_APP_ID")
        .ok()
        .map(|v| v.trim().to_string());
    let env_secret = std::env::var("FEISHU_APP_SECRET")
        .ok()
        .map(|v| v.trim().to_string());
    if let (Some(id), Some(secret)) = (env_id, env_secret)
        && !id.is_empty()
        && !secret.is_empty()
    {
        return Some((id, secret));
    }

    let cfg = configw::get_all_config();
    let id = cfg.get_opt("feishu.app_id").map(|v| v.trim().to_string());
    let secret = cfg
        .get_opt("feishu.app_secret")
        .map(|v| v.trim().to_string());
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
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("msg").and_then(|x| x.as_str()).map(|s| s.trim().to_string()))
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
    let app_access_token = get_app_access_token_cached(client, base_url)?;
    let url = format!(
        "{}/open-apis/authen/v1/refresh_access_token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    });
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", app_access_token))
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
    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let user_access_token = data
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = data
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = data
        .get("expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(60) as u64;
    let refresh_expires_in = data
        .get("refresh_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if user_access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in refresh response",
            Some(data),
        ));
    }
    Ok(RefreshedUserToken {
        user_access_token,
        refresh_token,
        expires_at: Instant::now() + Duration::from_secs(expires_in),
        refresh_expires_in,
    })
}
fn get_app_access_token_cached(client: &Client, base_url: &str) -> Result<String, JsonRpcErr> {
    let cache = APP_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    let now = Instant::now();
    if let Ok(guard) = cache.lock()
        && let Some(cached) = guard.as_ref()
        && cached.expires_at > now + Duration::from_secs(300)
    {
        return Ok(cached.token.clone());
    }

    let Some((app_id, app_secret)) = resolve_app_credentials() else {
        return Err(json_rpc_error(
            -32000,
            "Missing Feishu app credentials (app_id/app_secret)",
            Some(json!({
                "env": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
                "config_keys": ["feishu.app_id", "feishu.app_secret"]
            })),
        ));
    };

    let (token, expires_at) = fetch_app_access_token(client, base_url, &app_id, &app_secret)?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(CachedToken {
            token: token.clone(),
            expires_at,
        });
    }
    Ok(token)
}

fn fetch_app_access_token(
    client: &Client,
    base_url: &str,
    app_id: &str,
    app_secret: &str,
) -> Result<(String, Instant), JsonRpcErr> {
    let url = format!(
        "{}/open-apis/auth/v3/app_access_token/internal",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "app_id": app_id,
        "app_secret": app_secret
    });

    let resp = client
        .post(url)
        .header("Content-Type", "application/json; charset=utf-8")
        .json(&body)
        .send()
        .map_err(|e| {
            json_rpc_error(
                -32000,
                "Failed to call app_access_token API",
                Some(json!({ "error": e.to_string() })),
            )
        })?;

    let status = resp.status();
    let text = resp.text().map_err(|e| {
        json_rpc_error(
            -32000,
            "Failed to read app_access_token response body",
            Some(json!({ "error": e.to_string() })),
        )
    })?;
    if !status.is_success() {
        return Err(json_rpc_error(
            -32000,
            "app_access_token API returned non-success HTTP status",
            Some(json!({ "status": status.as_u16(), "body": text })),
        ));
    }

    let v: Value = serde_json::from_str(&text).map_err(|e| {
        json_rpc_error(
            -32000,
            "app_access_token response is not valid JSON",
            Some(json!({ "error": e.to_string(), "body": text })),
        )
    })?;
    let code = v.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
    if code != 0 {
        return Err(json_rpc_error(
            -32000,
            "app_access_token API returned error code",
            Some(v),
        ));
    }

    let token = v
        .get("app_access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "app_access_token missing in response",
            Some(v),
        ));
    }
    let expire = v
        .get("expire")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(60) as u64;
    let expires_at = Instant::now() + Duration::from_secs(expire);
    Ok((token, expires_at))
}

fn feishu_oauth_authorize_url(args: &Value) -> Result<String, JsonRpcErr> {
    let cfg = configw::get_all_config();
    let app_id = cfg
        .get_opt("feishu.app_id")
        .or_else(|| std::env::var("FEISHU_APP_ID").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| json_rpc_error(-32000, "Missing feishu.app_id / FEISHU_APP_ID", None))?;

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
        .unwrap_or("offline_access")
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
        url_encode_component(&app_id),
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
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout_sec);
        loop {
            if Instant::now() >= deadline {
                break;
            }
            let mut accepted: Option<TcpStream> = None;
            for listener in &listeners {
                match listener.accept() {
                    Ok((stream, _)) => {
                        accepted = Some(stream);
                        break;
                    }
                    Err(_) => {}
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

    match rx.recv_timeout(Duration::from_secs(timeout_sec)) {
        Ok(code) => Ok(format!("code: {code}\nport: {port}\npath: /callback")),
        Err(_) => Err(json_rpc_error(
            -32000,
            "Timeout waiting for OAuth code",
            Some(json!({ "port": port, "timeout_sec": timeout_sec })),
        )),
    }
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

    let app_access_token = get_app_access_token_cached(&client, &base_url)?;
    let url = format!(
        "{}/open-apis/authen/v1/access_token",
        base_url.trim_end_matches('/')
    );
    let body = json!({
        "grant_type": "authorization_code",
        "code": code
    });
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", app_access_token))
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
    let data = v.get("data").cloned().unwrap_or_else(|| json!({}));
    let access_token = data
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let refresh_token = data
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let expires_in = data.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    let refresh_expires_in = data
        .get("refresh_expires_in")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    if access_token.is_empty() {
        return Err(json_rpc_error(
            -32000,
            "Missing access_token in response",
            Some(data),
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
            return PathBuf::from(rust_tools::common::utils::expanduser(&v).as_ref());
        }
    }
    PathBuf::from(
        rust_tools::common::utils::expanduser("~/.config/rust_tools/feishu_token.json").as_ref(),
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
    use std::sync::{Mutex, OnceLock};

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

        let rendered = render_docx_blocks_as_text(&items);
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

        let rendered = render_docx_blocks_as_text(&items);
        assert!(rendered.contains("### 6.1. 系统架构图"));
        assert!(rendered.contains("[文字绘图: board-token-1]"));
        assert!(rendered.contains("[文字绘图: board-token-2]"));
        assert!(
            rendered.find("### 6.1. 系统架构图") < rendered.find("[文字绘图: board-token-1]")
        );
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
}
